"""Roda o corredor no SUMO comparando estratégias de controle.

    python -m simulacao.rodar --seeds 1 2 3

Estratégias:
    fixo     programa fixo do SUMO (90 s, sem defasagem) ~ o que a cidade tem hoje
    atuado   controlador atuado isolado (detecção de brecha), baseline forte
    isolado  IITLS adaptativo por cruzamento, sem coordenação (ablação)
    iitls    IITLS completo: adaptativo + onda verde + preempção + prioridade a ônibus

A telemetria passa pelo protocolo binário de verdade (encode -> bytes -> decode),
e cada cruzamento executa o plano com o mesmo LocalController que roda na borda.
"""

from __future__ import annotations

import argparse
import json
import statistics
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

try:
    import libsumo as traci
except ImportError:  # pragma: no cover
    import traci

from iitls_cloud.controller import Coordinator
from iitls_cloud.edge_runtime import LocalController
from iitls_cloud.protocol import Event, EventType, HealthFlag, LaneStats, Preempt, Telemetry, decode
from iitls_cloud.topology import Network

from .cenario import CAMERA_LINE_M, build

PERIODS = [  # (início, fim, leste, oeste, transversal por sentido, conversões)
    (0, 1200, 800, 650, 120, 35),
    (1200, 2400, 1300, 550, 160, 50),
    (2400, 3600, 180, 150, 30, 10),
]
TELEMETRY_S = 5
END = 4500
EMERGENCY_T = 1500
FAULT = (2, 2000, 2600)  # câmera do nó 2 falha nesse intervalo
OCC = {"car": 1.4, "moto": 1.1, "bus": 30.0, "truck": 1.2, "emergency": 3.0}


def write_routes(path: Path, seed: int, n: int) -> None:
    root = ET.Element("routes")
    ET.SubElement(root, "vType", id="car", vClass="passenger", length="4.5", accel="2.6", decel="4.5", sigma="0.5")
    ET.SubElement(root, "vType", id="moto", vClass="motorcycle", length="2.2", accel="3.5", decel="5", sigma="0.5")
    ET.SubElement(root, "vType", id="bus", vClass="bus", length="12", accel="1.2", decel="4", sigma="0.3")
    ET.SubElement(root, "vType", id="truck", vClass="truck", length="9", accel="1.3", decel="4", sigma="0.4")
    ET.SubElement(root, "vType", id="emergency", vClass="emergency", length="6", accel="2.8", decel="5", speedFactor="1.3", sigma="0")
    dist = ET.SubElement(root, "vTypeDistribution", id="mix")
    for t, p in (("car", 0.88), ("moto", 0.07), ("bus", 0.025), ("truck", 0.025)):
        dist.append(ET.Element("vType", refId=t, probability=str(p)))
    k = 0
    flows = []
    for b, e, eb, wb, side, turn in PERIODS:
        flows += [(b, e, "W_J0", f"J{n-1}_E", eb), (b, e, f"E_J{n-1}", "J0_W", wb)]
        for i in range(n):
            flows += [(b, e, f"N{i}_J{i}", f"J{i}_S{i}", side), (b, e, f"S{i}_J{i}", f"J{i}_N{i}", side),
                      (b, e, f"N{i}_J{i}", f"J{n-1}_E", turn), (b, e, f"S{i}_J{i}", "J0_W", turn),
                      (b, e, "W_J0", f"J{i}_S{i}", turn), (b, e, f"E_J{n-1}", f"J{i}_N{i}", turn)]
    for b, e, fr, to, vph in flows:
        if fr.split("_")[0] == to.split("_")[1]:
            continue
        ET.SubElement(root, "flow", id=f"f{k}", type="mix", begin=str(b), end=str(e), probability=f"{vph/3600:.5f}",
                      attrib={"from": fr, "to": to}, departLane="best", departSpeed="max")
        k += 1
    trip = ET.Element("trip", id="ambulancia", type="emergency", depart=str(EMERGENCY_T), departLane="best", departSpeed="max",
                      attrib={"from": f"E_J{n-1}", "to": "J0_W"})
    root.append(trip)
    # SUMO exige ordenação por partida
    items = [c for c in root if c.tag in ("flow", "trip")]
    for c in items:
        root.remove(c)
    for c in sorted(items, key=lambda c: float(c.get("begin", c.get("depart")))):
        root.append(c)
    ET.indent(root)
    ET.ElementTree(root).write(path)


class Edge:
    """Simula a Jetson de um cruzamento: agrega detecções e executa o plano local."""

    def __init__(self, nid: int, smap: dict, net: Network):
        self.nid = nid
        self.tls = smap["tls"]
        self.lanes = smap["lanes"]
        self.n_links = smap["n_links"]
        self.link_phase = {int(k): v for k, v in smap["link_phase"].items()}
        self.node = net.nodes[nid]
        self.local = LocalController(self.node, demand=self.has_demand, max_shift_frac=net.params.max_offset_shift_frac)
        self.phase_lanes = {ph.id: ph.lanes for ph in self.node.phases}
        self.seq = 0
        self.reset()
        self.seen_emerg: set[str] = set()
        self.seen_bus: set[str] = set()
        self.last_state = ""
        self.halting: dict[int, int] = {}
        self.stop_seen: dict[int, set] = defaultdict(set)

    def reset(self):
        self.counts = defaultdict(lambda: defaultdict(set))
        self.occ = defaultdict(float)
        self.sat_cnt = defaultdict(int)
        self.sat_secs = defaultdict(int)
        self.spd = defaultdict(list)
        self.steps = 0

    def has_demand(self, phase: int) -> bool:
        """Presença na zona da câmera (o que a Jetson vê em tempo real)."""
        return any(traci.lanearea.getLastStepVehicleNumber(f"zone_{self.nid}_{k}") > 0 for k in self.phase_lanes[phase])

    def phase_of_lane(self, k: int) -> int:
        for ph in self.node.phases:
            if k in ph.lanes:
                return ph.id
        return 0

    def sense(self, t: float, events: list):
        self.steps += 1
        for k, lane in enumerate(self.lanes):
            for vid in traci.inductionloop.getLastStepVehicleIDs(f"cam_{self.nid}_{k}"):
                vt = traci.vehicle.getTypeID(vid)
                cls = vt.split("@")[0]
                if cls == "emergency":
                    if vid not in self.seen_emerg:
                        self.seen_emerg.add(vid)
                        eta = CAMERA_LINE_M / max(traci.vehicle.getSpeed(vid), 5.0)
                        events.append(Event(self.nid, 0, self.seq, int(t), EventType.EMERGENCY_VEHICLE, k, int(eta * 10)))
                    cls = "car"
                if cls == "bus" and vid not in self.seen_bus:
                    self.seen_bus.add(vid)
                    eta = CAMERA_LINE_M / max(traci.vehicle.getSpeed(vid), 3.0)
                    self.local.bus_request(self.phase_of_lane(k), eta, t)
                self.counts[k][cls].add(vid)
            self.occ[k] += traci.lanearea.getLastStepOccupancy(f"zone_{self.nid}_{k}") / 100.0
            # saturação observada: passagens na retenção com verde e fila (após a perda de partida)
            cur = self.local.current
            if cur and cur.kind == "G" and k in self.phase_lanes.get(cur.phase, ()) and t - cur.start >= 3 and self.halting.get(k, 0) > 0:
                self.sat_secs[k] += 1
                ids = set(traci.inductionloop.getLastStepVehicleIDs(f"stop_{self.nid}_{k}"))
                self.sat_cnt[k] += len(ids - self.stop_seen[k])
                self.stop_seen[k] = ids
            self.halting[k] = traci.lanearea.getJamLengthVehicle(f"zone_{self.nid}_{k}")
            s = traci.lanearea.getLastStepMeanSpeed(f"zone_{self.nid}_{k}")
            if s >= 0:
                self.spd[k].append(s * 3.6)

    def telemetry(self, t: float, healthy: bool) -> bytes:
        lanes = []
        for k in range(len(self.lanes)):
            counts = {c: len(v) for c, v in self.counts[k].items()}
            q = traci.lanearea.getJamLengthVehicle(f"zone_{self.nid}_{k}")
            sp = statistics.fmean(self.spd[k]) if self.spd[k] else 0.0
            sat = self.sat_cnt[k] * 3600.0 / self.sat_secs[k] if self.sat_secs[k] >= 3 else 0.0
            lanes.append(LaneStats(k, counts, q, self.occ[k] / max(self.steps, 1), sp, sat))
        self.seq += 1
        flags = HealthFlag.CAMERA_OK if healthy else 0
        msg = Telemetry(self.nid, 0, self.seq, int(t), TELEMETRY_S, lanes, flags)
        self.reset()
        return msg.to_bytes()

    def actuate(self, t: float):
        iv = self.local.step(t)
        chars = []
        for idx in range(self.n_links):
            lp = self.link_phase.get(idx)
            if iv.kind == "F":
                chars.append("o")
            elif lp and lp[0] == iv.phase:
                chars.append(lp[1] if iv.kind == "G" else ("y" if iv.kind == "Y" else "r"))
            else:
                chars.append("r")
        state = "".join(chars)
        if state != self.last_state:
            traci.trafficlight.setRedYellowGreenState(self.tls, state)
            self.last_state = state


def run(strategy: str, seed: int, out: Path, log: bool = False) -> dict:
    topo = json.loads((out / "topologia.json").read_text())
    smap = json.loads((out / "sumo_map.json").read_text())
    n = len(topo["nodes"])
    routes = out / f"rotas_{seed}.rou.xml"
    if not routes.exists():
        write_routes(routes, seed, n)
    trip = out / f"trip_{strategy}_{seed}.xml"
    net_file = out / ("corredor_atuado.net.xml" if strategy == "atuado" else "corredor.net.xml")
    cmd = ["sumo", "-n", str(net_file), "-r", str(routes), "-a", str(out / "detectores.add.xml"), "--tripinfo-output", str(trip),
           "--device.emissions.probability", "1", "--seed", str(seed), "--no-step-log", "true", "--no-warnings", "true",
           "--time-to-teleport", "300", "--end", str(END)]
    traci.start(cmd)
    coord = edges = None
    if strategy in ("iitls", "isolado"):
        if strategy == "isolado":
            topo["params"]["coupling_max_travel_s"] = 0
        net = Network.from_dict(topo)
        coord = Coordinator(net, t_ref=0)
        edges = {int(k): Edge(int(k), v, net) for k, v in smap.items()}
        coord.replan(0)
        for e in edges.values():
            e.local.receive_plan(coord.plans[e.nid])
    pending = []  # (t_entrega, ação) simula latência de 1 s do enlace
    t = 0.0
    log_rows = []
    while t < END and (traci.simulation.getMinExpectedNumber() > 0 or t < 3600):
        traci.simulationStep()
        t = traci.simulation.getTime()
        if coord is None:
            continue
        events: list = []
        for e in edges.values():
            e.sense(t, events)
        for ev in events:
            # a borda reage na hora (sem depender da nuvem) ...
            node = edges[ev.node]
            node.local.receive_preempt(Preempt(ev.node, node.phase_of_lane(ev.lane), int(t), 20))
            # ... e a nuvem abre o corredor verde nos cruzamentos seguintes
            if strategy == "iitls":
                for cmd_ in coord.ingest(decode(ev.to_bytes()), t):
                    if cmd_.node != ev.node:
                        pending.append((t + 1, cmd_))
        if int(t) % TELEMETRY_S == 0:
            for e in edges.values():
                healthy = not (strategy == "iitls" and e.nid == FAULT[0] and FAULT[1] <= t < FAULT[2])
                coord.ingest(decode(e.telemetry(t, healthy)), t)
            plans = coord.maybe_replan(t)
            if plans:
                for nid, p in plans.items():
                    pending.append((t + 1, decode(p.to_bytes())))
                if log:
                    log_rows.append(coord.snapshot(t))
        due = [x for x in pending if x[0] <= t]
        pending = [x for x in pending if x[0] > t]
        for _, item in due:
            if hasattr(item, "phases"):
                edges[item.node].local.receive_plan(item)
            else:
                edges[item.node].local.receive_preempt(item)
        for e in edges.values():
            e.actuate(t)
    traci.close()
    if log:
        (out / f"log_{strategy}_{seed}.json").write_text(json.dumps(log_rows, default=str))
    return metrics(trip, n)


def metrics(trip: Path, n: int) -> dict:
    rows = []
    for ti in ET.parse(trip).getroot().iter("tripinfo"):
        em = ti.find("emissions")
        rows.append({
            "id": ti.get("id"), "type": ti.get("vType").split("@")[0], "dur": float(ti.get("duration")),
            "loss": float(ti.get("timeLoss")), "stops": int(ti.get("waitingCount")),
            "co2": float(em.get("CO2_abs")) / 1e6 if em is not None else 0.0,
            "art": ti.get("departLane").rsplit("_", 1)[0] in ("W_J0", f"E_J{n-1}") and ti.get("arrivalLane").rsplit("_", 1)[0] in ("J0_W", f"J{n-1}_E"),
        })
    art = [r for r in rows if r["art"]]
    amb = [r for r in rows if r["id"] == "ambulancia"]
    return {
        "veiculos": len(rows),
        "tempo_viagem_s": statistics.fmean(r["dur"] for r in rows),
        "atraso_s": statistics.fmean(r["loss"] for r in rows),
        "paradas": statistics.fmean(r["stops"] for r in rows),
        "atraso_arterial_s": statistics.fmean(r["loss"] for r in art),
        "paradas_arterial": statistics.fmean(r["stops"] for r in art),
        "atraso_pessoas_h": sum(r["loss"] * OCC.get(r["type"], 1.4) for r in rows) / 3600,
        "atraso_onibus_s": statistics.fmean(r["loss"] for r in rows if r["type"] == "bus") if any(r["type"] == "bus" for r in rows) else 0,
        "co2_kg": sum(r["co2"] for r in rows),
        "ambulancia_s": amb[0]["dur"] if amb else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="simulacao/out")
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--estrategias", nargs="+", default=["fixo", "atuado", "isolado", "iitls"])
    a = ap.parse_args()
    out = Path(a.saida)
    build(out)
    res = defaultdict(list)
    for s in a.seeds:
        for st in a.estrategias:
            m = run(st, s, out, log=(st == "iitls" and s == a.seeds[0]))
            res[st].append(m)
            print(st, s, {k: round(v, 2) for k, v in m.items()}, flush=True)
    summary = {st: {k: statistics.fmean(m[k] for m in ms) for k in ms[0]} for st, ms in res.items()}
    (out / "resultados.json").write_text(json.dumps(summary, indent=1))
    keys = list(next(iter(summary.values())))
    print("\n| métrica | " + " | ".join(summary) + " |")
    print("|---|" + "---|" * len(summary))
    for k in keys:
        print(f"| {k} | " + " | ".join(f"{summary[st][k]:.1f}" for st in summary) + " |")


if __name__ == "__main__":
    main()
