"""Gera o cenário SUMO (corredor arterial com N cruzamentos) e a topologia IITLS correspondente.

    python -m simulacao.cenario --saida simulacao/out

Cria: corredor.net.xml, detectores.add.xml, rotas_<seed>.rou.xml (gerado por rodar.py)
e topologia.json (mesmo formato que a nuvem usa em produção).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import sumolib

SPACING = [250, 400, 300, 350]  # distâncias entre cruzamentos (m)
SIDE_LEN = 200
SPEED = 13.89  # 50 km/h
CAMERA_LINE_M = 50  # linha virtual de contagem: 50 m antes da retenção
CAMERA_VIEW_M = 80  # zona que a câmera enxerga para fila/ocupação


def node_ids(n: int) -> list[str]:
    return [f"J{i}" for i in range(n)]


def build(out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    n = len(SPACING) + 1
    xs = [0]
    for d in SPACING:
        xs.append(xs[-1] + d)
    nodes = ET.Element("nodes")
    ET.SubElement(nodes, "node", id="W", x=str(-300), y="0")
    ET.SubElement(nodes, "node", id="E", x=str(xs[-1] + 300), y="0")
    for i, x in enumerate(xs):
        ET.SubElement(nodes, "node", id=f"J{i}", x=str(x), y="0", type="traffic_light")
        ET.SubElement(nodes, "node", id=f"N{i}", x=str(x), y=str(SIDE_LEN))
        ET.SubElement(nodes, "node", id=f"S{i}", x=str(x), y=str(-SIDE_LEN))
    edges = ET.Element("edges")

    def edge(a: str, b: str, lanes: int):
        ET.SubElement(edges, "edge", id=f"{a}_{b}", attrib={"from": a, "to": b, "numLanes": str(lanes), "speed": str(SPEED)})

    chain = ["W"] + node_ids(n) + ["E"]
    for a, b in zip(chain, chain[1:]):
        edge(a, b, 2)
        edge(b, a, 2)
    for i in range(n):
        for s in (f"N{i}", f"S{i}"):
            edge(s, f"J{i}", 1)
            edge(f"J{i}", s, 1)
    ET.ElementTree(nodes).write(out / "corredor.nod.xml")
    ET.ElementTree(edges).write(out / "corredor.edg.xml")
    subprocess.run(
        ["netconvert", "-n", str(out / "corredor.nod.xml"), "-e", str(out / "corredor.edg.xml"), "-o", str(out / "corredor.net.xml"),
         "--tls.layout", "opposites", "--tls.allred.time", "2", "--tls.yellow.time", "3", "--no-turnarounds", "--no-warnings"],
        check=True,
    )
    # baseline "atuado": mesma malha, controladores atuados isolados do próprio SUMO
    subprocess.run(
        ["netconvert", "-s", str(out / "corredor.net.xml"), "-o", str(out / "corredor_atuado.net.xml"),
         "--tls.default-type", "actuated", "--tls.rebuild", "--tls.layout", "opposites", "--tls.allred.time", "2",
         "--tls.yellow.time", "3", "--no-warnings"],
        check=True,
    )
    return make_topology(out, n)


def make_topology(out: Path, n: int) -> dict:
    net = sumolib.net.readNet(str(out / "corredor.net.xml"), withPrograms=True)
    topo_nodes, sumo_map, links = [], {}, []
    add = ET.Element("additional")
    approach_lanes: dict[tuple[int, str], list[int]] = {}
    for i in range(n):
        tls = net.getTLS(f"J{i}")
        prog = list(tls.getPrograms().values())[0]
        phases = prog.getPhases()
        conns = tls.getConnections()  # (inLane, outLane, linkIndex)
        in_lanes = []
        for c in conns:
            lid = c[0].getID()
            if lid not in in_lanes:
                in_lanes.append(lid)
        lane_idx = {lid: k for k, lid in enumerate(in_lanes)}
        green_phases = [k for k, ph in enumerate(phases) if "G" in ph.state or "g" in ph.state]
        iitls_phases = []
        link_phase = {}
        for pid, k in enumerate(green_phases):
            st = phases[k].state
            lanes = sorted({lane_idx[c[0].getID()] for c in conns if st[c[2]] in "Gg"})
            for c in conns:
                if st[c[2]] in "Gg":
                    link_phase[c[2]] = (pid, st[c[2]])
            has_side = any(net.getLane(in_lanes[ln]).getEdge().getFromNode().getID().startswith(("N", "S")) for ln in lanes)
            iitls_phases.append({"id": pid, "lanes": lanes, "min_green": 8.0 if has_side else 12.0, "max_green": 90.0,
                                 "yellow": 3.0, "all_red": 2.0, "ped_min_green": 10.0 if has_side else 0.0})
        lanes = []
        for lid, k in lane_idx.items():
            lane = net.getLane(lid)
            src = lane.getEdge().getFromNode().getID()
            appr = {"W": "W", "E": "E"}.get(src[0], src[0]) if not src.startswith("J") else ("W" if int(src[1:]) < i else "E")
            lanes.append({"id": k, "approach": appr, "movement": "LTR", "sat_flow": 1800.0 if src[0] not in "NS" else 1650.0,
                          "storage_m": round(lane.getLength(), 1)})
            approach_lanes.setdefault((i, appr), []).append(k)
            L = lane.getLength()
            ET.SubElement(add, "inductionLoop", id=f"cam_{i}_{k}", lane=lid, pos=str(max(1.0, L - CAMERA_LINE_M)), period="3600", file="NUL")
            ET.SubElement(add, "inductionLoop", id=f"stop_{i}_{k}", lane=lid, pos=str(L - 1.0), period="3600", file="NUL")
            ET.SubElement(add, "laneAreaDetector", id=f"zone_{i}_{k}", lane=lid, pos=str(max(0.0, L - CAMERA_VIEW_M)),
                          endPos=str(L - 0.1), period="3600", file="NUL")
        sumo_map[i] = {"tls": f"J{i}", "lanes": in_lanes, "n_links": len(phases[0].state),
                       "link_phase": {str(k): v for k, v in link_phase.items()}}
        topo_nodes.append({"id": i, "name": f"J{i}", "lanes": lanes, "phases": iitls_phases})
    for i in range(n - 1):
        d = SPACING[i]
        for src, dst, appr in ((i, i + 1, "W"), (i + 1, i, "E")):
            sl, dl = approach_lanes[(src, appr)], approach_lanes[(dst, appr)]
            sp = next(p["id"] for p in topo_nodes[src]["phases"] if set(sl) <= set(p["lanes"]))
            dp = next(p["id"] for p in topo_nodes[dst]["phases"] if set(dl) <= set(p["lanes"]))
            links.append({"src": src, "dst": dst, "src_phase": sp, "dst_phase": dp, "src_lanes": sl, "dst_lanes": dl,
                          "length_m": float(d), "speed_kmh": SPEED * 3.6})
    topo = {"nodes": topo_nodes, "links": links, "params": {"cycle_min": 50, "cycle_max": 130}}
    (out / "topologia.json").write_text(json.dumps(topo, indent=1), encoding="utf-8")
    (out / "sumo_map.json").write_text(json.dumps(sumo_map, indent=1), encoding="utf-8")
    ET.indent(add)
    ET.ElementTree(add).write(out / "detectores.add.xml")
    return topo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--saida", default="simulacao/out")
    build(Path(ap.parse_args().saida))
