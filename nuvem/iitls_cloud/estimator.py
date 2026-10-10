"""Estimativa de demanda por faixa a partir da telemetria das câmeras.

- Fluxo em PCU/h suavizado (média móvel exponencial com constante de tempo `tau`).
- Fila residual = menor fila vista ao longo do último ciclo. A fila normal do
  vermelho some no verde; se nem no verde ela zera, é demanda não atendida
  (supersaturação) e entra no cálculo como demanda extra.
- Perfil histórico por faixa e janela de 15 min do dia, usado quando a câmera
  falha ou os dados ficam velhos.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from .protocol import VEHICLE_CLASSES, HealthFlag, Telemetry
from .topology import Network

BIN_S = 900


@dataclass
class LaneState:
    flow_pcu: float = 0.0  # PCU/h
    flow_veh: float = 0.0  # veic/h
    person_weight: float = 1.4  # pessoas por PCU
    queue: float = 0.0  # veículos
    occupancy: float = 0.0
    speed_kmh: float = 0.0
    peds_ph: float = 0.0  # pedestres/h
    bus_ph: float = 0.0
    last_t: float = -math.inf
    blocked_until: float = -math.inf  # incidente
    camera_fault: bool = False
    samples: int = 0
    sat_learned: float = 0.0  # saturação aprendida (veic/h); 0 = ainda sem medida
    queue_hist: deque = field(default_factory=lambda: deque(maxlen=64))  # (t, fila)


@dataclass
class Estimator:
    net: Network
    tau: float = 180.0
    lanes: dict[tuple[int, int], LaneState] = field(default_factory=dict)
    history: dict[tuple[int, int, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for n in self.net.nodes.values():
            for lid in n.lanes:
                self.lanes[(n.id, lid)] = LaneState()

    def ingest(self, msg: Telemetry) -> None:
        p = self.net.params
        dt = max(1, msg.interval)
        alpha = 1.0 - math.exp(-dt / self.tau)
        healthy = bool(msg.flags & HealthFlag.CAMERA_OK) and not (msg.flags & HealthFlag.OCCLUSION)
        for ls in msg.lanes:
            st = self.lanes.get((msg.node, ls.lane))
            if st is None:
                continue
            if not healthy:
                st.camera_fault = True
                continue
            st.camera_fault = False
            pcu = sum(ls.counts.get(c, 0) * p.pcu.get(c, 1.0) for c in VEHICLE_CLASSES)
            veh = ls.vehicles()
            persons = sum(ls.counts.get(c, 0) * p.occupancy.get(c, 1.4) for c in VEHICLE_CLASSES)
            a = 1.0 if st.samples == 0 else alpha
            st.flow_pcu += a * (pcu * 3600.0 / dt - st.flow_pcu)
            st.flow_veh += a * (veh * 3600.0 / dt - st.flow_veh)
            if pcu > 0:
                st.person_weight += a * (persons / pcu - st.person_weight)
            st.peds_ph += a * (ls.counts.get("ped", 0) * 3600.0 / dt - st.peds_ph)
            st.bus_ph += a * (ls.counts.get("bus", 0) * 3600.0 / dt - st.bus_ph)
            if ls.sat_obs > 0:
                # saturação real muda com conversões permissivas, ônibus parando, chuva...
                b = 1.0 - math.exp(-dt / 600.0)
                st.sat_learned = ls.sat_obs if st.sat_learned == 0 else st.sat_learned + b * (ls.sat_obs - st.sat_learned)
            st.queue = ls.queue
            st.queue_hist.append((msg.t_end, ls.queue))
            st.occupancy = ls.occupancy
            st.speed_kmh = ls.speed_kmh
            st.last_t = msg.t_end
            st.samples += 1
            key = (msg.node, ls.lane, int(msg.t_end % 86400) // BIN_S)
            old = self.history.get(key)
            self.history[key] = st.flow_pcu if old is None else old + 0.05 * (st.flow_pcu - old)

    def block_lane(self, node: int, lane: int, until: float) -> None:
        if (node, lane) in self.lanes:
            self.lanes[(node, lane)].blocked_until = until

    def set_fault(self, node: int, lane: int | None, fault: bool) -> None:
        for (n, ln), st in self.lanes.items():
            if n == node and (lane is None or lane == ln):
                st.camera_fault = fault

    def is_stale(self, node: int, lane: int, now: float) -> bool:
        st = self.lanes[(node, lane)]
        return st.camera_fault or now - st.last_t > self.net.params.stale_after_s

    def flow(self, node: int, lane: int, now: float) -> float:
        """Demanda em PCU/h, com fallback para o histórico quando o dado está velho."""
        st = self.lanes[(node, lane)]
        if not self.is_stale(node, lane, now):
            return st.flow_pcu
        hist = self.history.get((node, lane, int(now % 86400) // BIN_S))
        if hist is not None:
            return hist
        return st.flow_pcu if st.samples else 300.0

    def residual_queue(self, node: int, lane: int, now: float, window: float) -> float:
        """Menor fila observada na janela (≈ 1 ciclo). 0 se a fila esvazia a cada verde."""
        st = self.lanes[(node, lane)]
        if self.is_stale(node, lane, now):
            return 0.0
        vals = [q for t, q in st.queue_hist if now - t <= window]
        return float(min(vals)) if vals else 0.0

    def sat_flow(self, node: int, lane: int, now: float) -> float:
        st = self.lanes[(node, lane)]
        base = self.net.nodes[node].lanes[lane].sat_flow
        if st.sat_learned > 0 and self.net.params.learn_sat_flow:
            base = min(max(st.sat_learned, 0.3 * base), 1.25 * base)
        return base * 0.05 if now < st.blocked_until else base

    def node_stale(self, node: int, now: float) -> bool:
        return all(self.is_stale(node, ln, now) for ln in self.net.nodes[node].lanes)
