"""Modelo da rede: cruzamentos, faixas, fases e ligações entre cruzamentos."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_PCU = {"car": 1.0, "moto": 0.4, "bus": 2.0, "truck": 2.0, "bike": 0.3}
# Ocupação média (pessoas por veículo). Otimizar atraso de pessoas, não de veículos,
# é o que dá prioridade natural ao transporte coletivo.
DEFAULT_OCCUPANCY = {"car": 1.4, "moto": 1.1, "bus": 30.0, "truck": 1.2, "bike": 1.0}


@dataclass
class Lane:
    id: int
    approach: str  # ex.: "W" (chega pelo oeste)
    movement: str = "T"  # T, L, R, TR, TL, LTR
    sat_flow: float = 1800.0  # veic/h de verde
    storage_m: float = 0.0  # comprimento disponível para fila (0 = desconhecido)
    view_m: float = 80.0  # até onde a câmera enxerga a fila a partir da retenção


@dataclass
class Phase:
    id: int
    lanes: list[int]
    min_green: float = 8.0
    max_green: float = 90.0
    yellow: float = 3.0
    all_red: float = 2.0
    ped_min_green: float = 0.0  # verde mínimo para travessia de pedestres
    startup_lost: float = 2.0

    @property
    def clearance(self) -> float:
        return self.yellow + self.all_red

    @property
    def lost_time(self) -> float:
        # perda de partida + entreverdes - aproveitamento do amarelo (~ = entreverdes)
        return self.startup_lost + self.all_red

    @property
    def floor_green(self) -> float:
        return max(self.min_green, self.ped_min_green)


@dataclass
class Node:
    id: int
    name: str
    lanes: dict[int, Lane]
    phases: list[Phase]
    allow_flash: bool = False  # amarelo piscante permitido com demanda muito baixa
    coordinated_phase: int | None = None  # fase da onda verde; None = deduzida das ligações

    def ordered_phases(self) -> list[Phase]:
        """Fases na ordem do plano: a coordenada primeiro (a defasagem marca o início dela)."""
        k = self.phase_index(self.coordinated_phase) if self.coordinated_phase is not None else 0
        return self.phases[k:] + self.phases[:k]

    def phase(self, phase_id: int) -> Phase:
        for p in self.phases:
            if p.id == phase_id:
                return p
        raise KeyError(phase_id)

    def phase_index(self, phase_id: int) -> int:
        return [p.id for p in self.phases].index(phase_id)


@dataclass
class Link:
    """Pelotão que sai de `src` na fase `src_phase` e chega em `dst` nas faixas `dst_lanes`."""

    src: int
    dst: int
    src_phase: int
    dst_phase: int
    src_lanes: list[int]
    dst_lanes: list[int]
    length_m: float
    speed_kmh: float = 50.0

    @property
    def travel_time(self) -> float:
        return self.length_m / (self.speed_kmh / 3.6)


@dataclass
class Params:
    cycle_min: float = 40.0
    cycle_max: float = 140.0
    cycle_step: float = 5.0
    low_demand_y: float = 0.15  # soma de razões críticas abaixo disso = modo baixa demanda
    saturated_y: float = 0.85
    stops_weight: float = 6.0  # 1 parada ~ 6 s de atraso (combustível, CO2, desgaste)
    person_delay: bool = True  # ponderar por ocupação (prioriza ônibus)
    coupling_max_travel_s: float = 90.0  # acima disso o pelotão se dispersa: não coordena
    replan_interval_s: float = 60.0
    plan_validity_cycles: int = 5
    stale_after_s: float = 30.0
    spillback_ratio: float = 0.8
    hysteresis: float = 0.05  # só troca de plano se ganhar > 5%
    max_offset_shift_frac: float = 0.2  # transição: corrige no máx. 20% do ciclo por ciclo
    # Usa a saturação medida pela borda (sat_obs). Desligado até validar a medição em campo:
    # na simulação a estimativa saiu enviesada para baixo e alongou ciclos sem ganho.
    learn_sat_flow: bool = False
    pcu: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_PCU))
    occupancy: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_OCCUPANCY))


@dataclass
class Network:
    nodes: dict[int, Node]
    links: list[Link]
    params: Params = field(default_factory=Params)

    @classmethod
    def from_dict(cls, d: dict) -> "Network":
        nodes = {}
        for nd in d["nodes"]:
            lanes = {ln["id"]: Lane(**ln) for ln in nd["lanes"]}
            phases = [Phase(**ph) for ph in nd["phases"]]
            nodes[nd["id"]] = Node(nd["id"], nd.get("name", str(nd["id"])), lanes, phases, nd.get("allow_flash", False),
                                   nd.get("coordinated_phase"))
        links = [Link(**lk) for lk in d.get("links", [])]
        params = Params(**d.get("params", {}))
        net = cls(nodes, links, params)
        net.validate()
        for n in nodes.values():
            if n.coordinated_phase is None:
                # fase que mais recebe/envia pelotões
                use: dict[int, int] = {}
                for lk in links:
                    if lk.src == n.id:
                        use[lk.src_phase] = use.get(lk.src_phase, 0) + 1
                    if lk.dst == n.id:
                        use[lk.dst_phase] = use.get(lk.dst_phase, 0) + 1
                n.coordinated_phase = max(use, key=use.get) if use else n.phases[0].id
        return net

    @classmethod
    def load(cls, path: str | Path) -> "Network":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return {
            "nodes": [
                {"id": n.id, "name": n.name, "allow_flash": n.allow_flash, "coordinated_phase": n.coordinated_phase,
                 "lanes": [asdict(ln) for ln in n.lanes.values()], "phases": [asdict(p) for p in n.phases]}
                for n in self.nodes.values()
            ],
            "links": [asdict(lk) for lk in self.links],
            "params": asdict(self.params),
        }

    def validate(self) -> None:
        for n in self.nodes.values():
            if not n.phases:
                raise ValueError(f"nó {n.id} sem fases")
            for p in n.phases:
                for ln in p.lanes:
                    if ln not in n.lanes:
                        raise ValueError(f"nó {n.id} fase {p.id}: faixa {ln} inexistente")
        for lk in self.links:
            for nid in (lk.src, lk.dst):
                if nid not in self.nodes:
                    raise ValueError(f"ligação com nó inexistente {nid}")
            self.nodes[lk.src].phase(lk.src_phase)
            self.nodes[lk.dst].phase(lk.dst_phase)
