import numpy as np
import pytest
from fastapi.testclient import TestClient

from iitls_cloud.api import create_app
from iitls_cloud.controller import Coordinator
from iitls_cloud.coordination import optimize_offsets
from iitls_cloud.edge_runtime import LocalController
from iitls_cloud.protocol import (
    Heartbeat, Event, EventType, HealthFlag, LaneStats, PhaseTiming, Plan, PlanMode, Preempt, ProtocolError, Telemetry, decode,
)
from iitls_cloud.timing import signal_delay, split_greens
from iitls_cloud.topology import Network


def corridor(n=4, spacing=(300, 450, 250)):
    def node(i):
        return {
            "id": i,
            "lanes": [{"id": 0, "approach": "W"}, {"id": 1, "approach": "W"}, {"id": 2, "approach": "E"},
                      {"id": 3, "approach": "E"}, {"id": 4, "approach": "N", "sat_flow": 1600},
                      {"id": 5, "approach": "S", "sat_flow": 1600}],
            "phases": [{"id": 0, "lanes": [0, 1, 2, 3], "min_green": 10},
                       {"id": 1, "lanes": [4, 5], "min_green": 8, "ped_min_green": 12}],
        }

    links = []
    for i in range(n - 1):
        d = spacing[i]
        links.append({"src": i, "dst": i + 1, "src_phase": 0, "dst_phase": 0, "src_lanes": [0, 1], "dst_lanes": [0, 1], "length_m": d})
        links.append({"src": i + 1, "dst": i, "src_phase": 0, "dst_phase": 0, "src_lanes": [2, 3], "dst_lanes": [2, 3], "length_m": d})
    return Network.from_dict({"nodes": [node(i) for i in range(n)], "links": links})


def feed(coord, net, t0, t1, eb=1, wb=0.3, side=0.2, healthy=True, nodes=None):
    """eb/wb/side = veículos por faixa a cada 5 s (determinístico)."""
    acc = {}
    for t in range(t0, t1, 5):
        for nid in nodes or net.nodes:
            lanes = []
            for ln, rate in ((0, eb), (1, eb), (2, wb), (3, wb), (4, side), (5, side)):
                acc[(nid, ln)] = acc.get((nid, ln), 0) + rate
                k = int(acc[(nid, ln)])
                acc[(nid, ln)] -= k
                lanes.append(LaneStats(ln, {"car": k}))
            flags = HealthFlag.CAMERA_OK if healthy else 0
            coord.ingest(decode(Telemetry(nid, 0, t, t, 5, lanes, flags).to_bytes()))


# ------------------------------------------------------------------ protocolo

def test_telemetry_roundtrip_binary_and_json():
    m = Telemetry(7, 1, 42, 1_760_000_000, 5, [LaneStats(0, {"car": 3, "bus": 1, "ped": 2}, queue=4, occupancy=0.35, speed_kmh=31, sat_obs=1650)])
    b = m.to_bytes()
    assert len(b) == 14 + 11 + 2
    for x in (decode(b), decode(m.to_json())):
        assert x.node == 7 and x.lanes[0].counts == {"car": 3, "bus": 1, "ped": 2}
        assert x.lanes[0].queue == 4 and x.lanes[0].occupancy == pytest.approx(0.35) and x.lanes[0].sat_obs == 1650


def test_crc_rejects_corruption():
    b = bytearray(Telemetry(1, 0, 1, 1, 5, [LaneStats(0, {"car": 1})]).to_bytes())
    b[5] ^= 0xFF
    with pytest.raises(ProtocolError):
        decode(bytes(b))


def test_plan_event_preempt_roundtrip():
    p = Plan(3, 9, 0, 90.0, 37.5, 1000, PlanMode.SATURATED, [PhaseTiming(0, 50.5, 3, 2), PhaseTiming(1, 29.5, 3, 2)])
    q = decode(p.to_bytes())
    assert q.cycle == 90.0 and q.offset == 37.5 and q.phases[0].green == 50.5 and q.mode == PlanMode.SATURATED
    assert decode(p.to_dict()).phases[1].green == 29.5
    e = decode(Event(1, 0, 5, 100, EventType.EMERGENCY_VEHICLE, 2, 45).to_bytes())
    assert e.kind == EventType.EMERGENCY_VEHICLE and e.value == 45
    assert decode(Preempt(1, 0, 100, 20).to_bytes()).hold == 20


# ------------------------------------------------------------------ tempos

def test_split_respects_floors_and_uses_whole_cycle():
    net = corridor()
    node = net.nodes[0]
    g = split_greens(node, 130, {0: 0.48, 1: 0.12})
    assert g[1] >= 12 and g[0] <= 90
    assert sum(g.values()) + 10 == pytest.approx(130)


def test_hcm_delay_grows_when_oversaturated():
    assert signal_delay(90, 40, 900, 1800) > 2 * signal_delay(90, 40, 600, 1800)


# ------------------------------------------------------------------ onda verde

def test_offsets_dp_matches_bruteforce_on_chain():
    rng = np.random.default_rng(0)
    C = 12
    tables = {(0, 1): [(rng.random(C), np.zeros(C))], (1, 2): [(rng.random(C), np.zeros(C))], (2, 1): [(rng.random(C), np.zeros(C))]}
    res = optimize_offsets(C, [0, 1, 2], tables)
    best = min(
        tables[(0, 1)][0][0][(b - a) % C] + tables[(1, 2)][0][0][(c - b) % C] + tables[(2, 1)][0][0][(b - c) % C]
        for a in range(C) for b in range(C) for c in range(C)
    )
    assert res.cost == pytest.approx(best)


def test_one_way_corridor_gets_forward_progression():
    net = corridor()
    coord = Coordinator(net)
    feed(coord, net, 0, 300, eb=1.0, wb=0.0, side=0.15)
    coord.replan(300)
    g = coord.groups[0]
    C = g.cycle
    for lk in net.links:
        if lk.src < lk.dst:  # leste: defasagem ≈ tempo de viagem (mod C)
            d = (g.offsets[lk.dst] - g.offsets[lk.src]) % C
            err = min(abs(d - lk.travel_time % C), C - abs(d - lk.travel_time % C))
            assert err <= 0.25 * C, (lk.src, lk.dst, d, lk.travel_time)
            assert g.aog[(lk.src, lk.dst)] > 0.8
    assert len({p.cycle for p in coord.plans.values()}) == 1  # ciclo comum


def test_modes_low_demand_saturated_and_failsafe():
    net = corridor()
    c = Coordinator(net)
    feed(c, net, 0, 300, eb=0.05, wb=0.05, side=0.02)
    c.replan(300)
    assert all(d.mode == PlanMode.LOW_DEMAND for d in c.decisions.values())
    c2 = Coordinator(net)
    feed(c2, net, 0, 300, eb=2.2, wb=1.2, side=0.5)
    c2.replan(300)
    assert any(d.mode == PlanMode.SATURATED for d in c2.decisions.values())
    assert max(p.cycle for p in c2.plans.values()) >= 100
    c3 = Coordinator(net)
    c3.replan(1000)  # nunca recebeu nada
    assert all(d.mode == PlanMode.FAILSAFE for d in c3.decisions.values())


def test_camera_failure_falls_back_to_history():
    net = corridor()
    c = Coordinator(net)
    feed(c, net, 0, 600, eb=1.0)
    c.replan(600)
    before = c.plans[1].phases[0].green
    feed(c, net, 600, 700, eb=0, wb=0, side=0, healthy=False, nodes=[1])
    feed(c, net, 600, 700, eb=1.0, nodes=[0, 2, 3])
    c.replan(700)
    assert c.decisions[1].mode != PlanMode.FAILSAFE
    assert c.plans[1].phases[0].green == pytest.approx(before, rel=0.35)


def test_emergency_opens_green_corridor_downstream():
    net = corridor()
    c = Coordinator(net)
    feed(c, net, 0, 120)
    c.replan(120)
    cmds = c.ingest(Event(0, 0, 1, 130, EventType.EMERGENCY_VEHICLE, 0, 50), now=130)
    assert [x.node for x in cmds] == [0, 1, 2, 3]
    assert all(x.phase == 0 for x in cmds)
    assert cmds[1].t_start > cmds[0].t_start
    c.replan(131)
    assert c.decisions[0].mode == PlanMode.PREEMPT


def test_incident_reduces_capacity():
    net = corridor()
    c = Coordinator(net)
    feed(c, net, 0, 300)
    c.ingest(Event(1, 0, 1, 300, EventType.INCIDENT, 0, 600), now=300)
    assert c.est.sat_flow(1, 0, 310) < 200
    c.ingest(Event(1, 0, 2, 320, EventType.INCIDENT_CLEARED, 0, 0), now=320)
    assert c.est.sat_flow(1, 0, 330) == 1800


# ------------------------------------------------------------------ executor local

def run_local(lc, t0, t1):
    seq = []
    for t in range(t0, t1):
        iv = lc.step(t)
        seq.append((t, iv.phase, iv.kind))
    return seq


def test_local_controller_safety_sequence_and_min_green():
    net = corridor()
    lc = LocalController(net.nodes[0])
    lc.receive_plan(Plan(0, 1, 0, 60, 0, 10_000, PlanMode.NORMAL, [PhaseTiming(0, 30, 3, 2), PhaseTiming(1, 20, 3, 2)]))
    seq = run_local(lc, 0, 600)
    for (_, p0, k0), (_, p1, k1) in zip(seq, seq[1:]):
        if p0 != p1:
            assert k0 == "R" and k1 == "G"  # troca de fase só após amarelo + vermelho geral
        if k1 == "Y" and k0 == "G":
            pass
    greens = {}
    for t, p, k in seq:
        if k == "G":
            greens.setdefault((p, len([1 for x in seq if x[0] <= t and x[2] == "R"])), []).append(t)
    assert min(len(v) for v in greens.values()) >= 10


def test_local_controller_converges_to_new_offset():
    net = corridor()
    lc = LocalController(net.nodes[0])
    lc.receive_plan(Plan(0, 1, 0, 60, 0, 10_000, PlanMode.NORMAL, [PhaseTiming(0, 30, 3, 2), PhaseTiming(1, 20, 3, 2)]))
    run_local(lc, 0, 120)
    lc.receive_plan(Plan(0, 2, 0, 60, 25, 10_000, PlanMode.NORMAL, [PhaseTiming(0, 30, 3, 2), PhaseTiming(1, 20, 3, 2)]))
    seq = run_local(lc, 120, 600)
    starts = [t for (t, p, k), (_, p0, k0) in zip(seq[1:], seq) if p == 0 and k == "G" and k0 != "G"]
    assert starts[-1] % 60 == 25
    assert all(b - a >= 60 * 0.8 - 1 for a, b in zip(starts, starts[1:]))  # sem salto: corrige aos poucos


def test_local_controller_skips_side_phase_without_demand():
    net = corridor()
    lc = LocalController(net.nodes[0], demand=lambda ph: ph == 0)
    lc.receive_plan(Plan(0, 1, 0, 60, 0, 10_000, PlanMode.NORMAL, [PhaseTiming(0, 30, 3, 2), PhaseTiming(1, 20, 3, 2)]))
    assert all(k == "G" and p == 0 for _, p, k in run_local(lc, 0, 300))


def test_local_controller_preempt_and_fallback():
    net = corridor()
    lc = LocalController(net.nodes[0])
    lc.receive_plan(Plan(0, 1, 0, 60, 0, 200, PlanMode.NORMAL, [PhaseTiming(0, 30, 3, 2), PhaseTiming(1, 20, 3, 2)]))
    run_local(lc, 0, 40)  # fase 1 verde
    lc.receive_preempt(Preempt(0, 0, 40, 20))
    seq = run_local(lc, 40, 80)
    first_g0 = next(t for t, p, k in seq if p == 0 and k == "G")
    assert first_g0 <= 40 + 12 + 5  # mínimo da fase atual + entreverdes
    assert all(p == 0 and k == "G" for t, p, k in seq if first_g0 <= t < first_g0 + 20)
    run_local(lc, 80, 400)  # plano venceu em t=200
    assert lc.active_plan(300).mode == PlanMode.FAILSAFE


# ------------------------------------------------------------------ API

def test_api_ingest_and_plan(monkeypatch):
    monkeypatch.setenv("IITLS_UDP_PORT", "0")
    net = corridor()
    app = create_app(net)
    with TestClient(app) as client:
        msg = Telemetry(0, 0, 1, 10, 5, [LaneStats(0, {"car": 2})])
        r = client.post("/v1/ingest", content=msg.to_bytes(), headers={"accept": "application/octet-stream"})
        assert r.status_code == 200 and decode(r.content).node == 0
        r = client.post("/v1/ingest", content=msg.to_json())
        assert r.json()["plan"]["n"] == 0
        assert client.get("/v1/plans/1").json()["n"] == 1
        assert "groups" in client.get("/v1/state").json()
        assert client.post("/v1/ingest", content=b"lixo").status_code == 400


def test_udp_returns_plan():
    import asyncio

    from iitls_cloud.api import Service, UdpProtocol

    async def go():
        loop = asyncio.get_running_loop()
        svc = Service(corridor())
        server, _ = await loop.create_datagram_endpoint(lambda: UdpProtocol(svc), local_addr=("127.0.0.1", 0))
        port = server.get_extra_info("sockname")[1]
        got = loop.create_future()

        class Client(asyncio.DatagramProtocol):
            def datagram_received(self, data, addr):
                got.set_result(data)

        client, _ = await loop.create_datagram_endpoint(Client, remote_addr=("127.0.0.1", port))
        client.sendto(Telemetry(2, 0, 1, 10, 5, [LaneStats(0, {"car": 1})]).to_bytes())
        data = await asyncio.wait_for(got, 2)
        client.close()
        server.close()
        return decode(data)

    plan = asyncio.run(go())
    assert plan.node == 2 and plan.phases


class FakeMqtt:
    def __init__(self):
        self.pub, self.subs = [], []

    def publish(self, topic, payload, qos=0, retain=False):
        self.pub.append((topic, payload, qos, retain))

    def subscribe(self, topic, qos=0):
        self.subs.append(topic)


def test_heartbeat_carries_controller_state():
    hb = decode(Heartbeat(3, 0, 9, 100, HealthFlag.CAMERA_OK, 55, 1, 12.5, 77).to_bytes())
    assert (hb.phase, hb.phase_elapsed, hb.plan_id, hb.temp_c) == (1, 12.5, 77, 55)
    assert decode(Heartbeat(3, 0, 9, 100, 1, 55, 1, 12.5, 77).to_json()).plan_id == 77


def test_mqtt_bridge_retained_plans_and_link_health():
    from iitls_cloud.api import Service
    from iitls_cloud.mqtt_bridge import CloudBridge, EdgeLink

    clock = [100.0]
    svc = Service(corridor(), clock=lambda: clock[0])
    cloud_mq = FakeMqtt()
    bridge = CloudBridge(svc, cloud_mq)
    bridge.start()
    assert cloud_mq.subs == ["iitls/+/up"]
    retained = [p for p in cloud_mq.pub if p[0].endswith("/plan")]
    assert len(retained) == 4 and all(p[3] and p[2] == 1 for p in retained)

    # borda recebe o plano retido e reporta no heartbeat que está executando ele
    edge_mq = FakeMqtt()
    lc = LocalController(corridor().nodes[1])
    edge = EdgeLink(1, lc, edge_mq)
    edge.start()
    topic, payload, *_ = next(p for p in retained if p[0] == "iitls/1/plan")
    edge.on_message(topic, payload)
    lc.step(100)
    phase, elapsed, plan_id = lc.status(100)
    edge.send(Heartbeat(1, 0, 1, 100, HealthFlag.CAMERA_OK, 50, phase, elapsed, plan_id))
    t, body, qos, retain = edge_mq.pub[-1]
    assert t == "iitls/1/up" and qos == 0 and not retain
    n_before = len(cloud_mq.pub)
    bridge.on_message(t, body)
    assert len(cloud_mq.pub) == n_before  # plano não mudou: nada republicado
    st = svc.coord.link_status(1, 102)
    assert st["online"] and st["no_plano_certo"]
    assert not svc.coord.link_status(1, 106)["online"]  # 5 s sem heartbeat

    # nó falsificado: tópico do 2 com mensagem do 1 é descartado
    bridge.on_message("iitls/2/up", body)
    assert 2 not in svc.coord.health

    # emergência vira PREEMPT nos tópicos cmd, não retidos
    edge.send(Event(1, 0, 2, 100, EventType.EMERGENCY_VEHICLE, 0, 50))
    t, body, qos, _ = edge_mq.pub[-1]
    assert qos == 1
    bridge.on_message(t, body)
    cmds = [p for p in cloud_mq.pub if p[0].endswith("/cmd")]
    assert cmds and not any(p[3] for p in cmds)
    edge.on_message("iitls/1/cmd", next(p[1] for p in cmds if p[0] == "iitls/1/cmd"))
    assert lc.preempt is not None
