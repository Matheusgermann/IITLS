"""Coordenador central: recebe telemetria/eventos e devolve planos para cada cruzamento.

Casos tratados:
- NORMAL: ciclo comum por grupo coordenado, verdes por Webster, defasagens ótimas.
- LOW_DEMAND: ciclo mínimo (ou amarelo piscante, se o cruzamento permitir).
- SATURATED: ciclo máximo, verde extra para escoar fila residual e controle de
  transbordo (spillback): segura o verde que alimenta uma via já cheia.
- PREEMPT: veículo de emergência. Corredor verde no cruzamento e nos seguintes.
- FAILSAFE: câmera sem dado. Usa histórico; sem histórico, plano fixo seguro.
- Incidente: faixa bloqueada perde capacidade até o evento de liberação.
- Pessoas, não veículos: atraso ponderado por ocupação (ônibus pesa mais).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .coordination import LinkInput, PhaseWindow, coordination_groups, link_cost_table, optimize_offsets
from .estimator import Estimator
from .protocol import Event, EventType, Heartbeat, PhaseTiming, Plan, PlanMode, Preempt, Telemetry
from .timing import phase_ratios, round_cycle, signal_delay, split_greens, webster_cycle
from .topology import Network, Node

VEHICLE_LENGTH_M = 7.0
EMERGENCY_HOLD_S = 20
EMERGENCY_HOPS = 3


@dataclass
class NodeDecision:
    mode: PlanMode
    ratios: dict[int, float]
    required_cycle: float
    weights: dict[int, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass
class GroupResult:
    nodes: list[int]
    cycle: float
    offsets: dict[int, int]
    cost_per_hour: float
    aog: dict[tuple[int, int], float]


class Coordinator:
    def __init__(self, net: Network, t_ref: int = 0):
        self.net = net
        self.t_ref = t_ref
        self.est = Estimator(net)
        self.plans: dict[int, Plan] = {}
        self.preempts: dict[int, Preempt] = {}
        self.decisions: dict[int, NodeDecision] = {}
        self.groups: list[GroupResult] = []
        self._plan_seq = 0
        self.last_replan = -math.inf
        self.last_eval = 0.0

    # ------------------------------------------------------------------ entrada

    def ingest(self, msg, now: float | None = None) -> list[Preempt]:
        """Processa uma mensagem da borda. Devolve comandos imediatos (preempção)."""
        if isinstance(msg, Telemetry):
            self.est.ingest(msg)
            return []
        if isinstance(msg, Heartbeat):
            return []
        if isinstance(msg, Event):
            now = msg.t if now is None else now
            return self._handle_event(msg, now)
        return []

    def _handle_event(self, ev: Event, now: float) -> list[Preempt]:
        if ev.kind == EventType.INCIDENT:
            self.est.block_lane(ev.node, ev.lane, now + (ev.value or 600))
        elif ev.kind == EventType.INCIDENT_CLEARED:
            self.est.block_lane(ev.node, ev.lane, -math.inf)
        elif ev.kind == EventType.CAMERA_FAULT:
            self.est.set_fault(ev.node, None, True)
        elif ev.kind == EventType.CAMERA_RESTORED:
            self.est.set_fault(ev.node, None, False)
        elif ev.kind == EventType.EMERGENCY_VEHICLE:
            return self._emergency_corridor(ev, now)
        return []

    def _phase_of_lane(self, node: Node, lane: int) -> int | None:
        for ph in node.phases:
            if lane in ph.lanes:
                return ph.id
        return None

    def _emergency_corridor(self, ev: Event, now: float) -> list[Preempt]:
        cmds = []
        node = self.net.nodes.get(ev.node)
        if node is None:
            return cmds
        phase = self._phase_of_lane(node, ev.lane)
        if phase is None:
            return cmds
        eta = ev.value / 10.0
        lanes = {ev.lane}
        cur, cur_phase = node.id, phase
        for hop in range(EMERGENCY_HOPS + 1):
            start = int(now + max(0.0, eta - 8.0))  # verde ~8 s antes de chegar
            cmd = Preempt(cur, cur_phase, start, EMERGENCY_HOLD_S)
            self.preempts[cur] = cmd
            cmds.append(cmd)
            nxt = [lk for lk in self.net.links if lk.src == cur and lk.src_phase == cur_phase and lanes & set(lk.src_lanes)]
            if not nxt or hop == EMERGENCY_HOPS:
                break
            lk = max(nxt, key=lambda x: len(lanes & set(x.src_lanes)))
            eta += lk.travel_time * 0.8  # viatura anda acima da velocidade da via
            cur, cur_phase, lanes = lk.dst, lk.dst_phase, set(lk.dst_lanes)
        return cmds

    # ---------------------------------------------------------------- decisões

    def _lane_inputs(self, node: Node, now: float):
        flow = {ln: self.est.flow(node.id, ln, now) for ln in node.lanes}
        sat = {ln: self.est.sat_flow(node.id, ln, now) for ln in node.lanes}
        window = max(60.0, self.plans[node.id].cycle + 5) if node.id in self.plans else 90.0
        queue = {ln: self.est.residual_queue(node.id, ln, now, window) for ln in node.lanes}
        return flow, sat, queue

    def _decide_node(self, node: Node, now: float) -> NodeDecision:
        p = self.net.params
        flow, sat, queue = self._lane_inputs(node, now)
        hint = self.plans[node.id].cycle if node.id in self.plans else 90.0
        ratios = phase_ratios(node, flow, sat, queue, hint)
        big_y = sum(ratios.values())
        req = webster_cycle(node, ratios, p)
        pre = self.preempts.get(node.id)
        if pre and now < pre.t_start + pre.hold:
            return NodeDecision(PlanMode.PREEMPT, ratios, req, notes=["preempção ativa"])
        if pre:
            del self.preempts[node.id]
        has_hist = any(k[0] == node.id for k in self.est.history)
        if self.est.node_stale(node.id, now) and not has_hist:
            return NodeDecision(PlanMode.FAILSAFE, ratios, 90.0, notes=["sem dados e sem histórico"])
        notes = []
        if self.est.node_stale(node.id, now):
            notes.append("dados velhos: usando histórico")
        if big_y < p.low_demand_y / 3 and node.allow_flash:
            return NodeDecision(PlanMode.FLASH, ratios, p.cycle_min, notes=notes + ["demanda muito baixa"])
        mode = PlanMode.NORMAL
        if big_y < p.low_demand_y:
            mode = PlanMode.LOW_DEMAND
        elif big_y > p.saturated_y:
            mode = PlanMode.SATURATED
        return NodeDecision(mode, ratios, req, notes=notes)

    def _apply_spillback(self) -> None:
        """Se a via entre A e B está quase cheia, A segura o verde que a alimenta e B prioriza escoar."""
        p = self.net.params
        for lk in self.net.links:
            if lk.src not in self.decisions or lk.dst not in self.decisions:
                continue
            # fila residual (não esvazia no verde) ocupando toda a visão da câmera = via enchendo
            _, _, resid = self._lane_inputs(self.net.nodes[lk.dst], self.last_eval)
            q = sum(resid[ln] for ln in lk.dst_lanes)
            lanes = self.net.nodes[lk.dst].lanes
            storage = sum(min(lk.length_m, lanes[ln].view_m) for ln in lk.dst_lanes) / VEHICLE_LENGTH_M
            if storage > 0 and q / storage >= p.spillback_ratio:
                src, dst = self.decisions[lk.src], self.decisions[lk.dst]
                src.weights[lk.src_phase] = src.weights.get(lk.src_phase, 1.0) * 0.6
                dst.weights[lk.dst_phase] = dst.weights.get(lk.dst_phase, 1.0) * 1.4
                for d in (src, dst):
                    if d.mode in (PlanMode.NORMAL, PlanMode.LOW_DEMAND):
                        d.mode = PlanMode.SATURATED
                src.notes.append(f"transbordo na via {lk.src}->{lk.dst}: segurando fase {lk.src_phase}")

    # ------------------------------------------------------------- otimização

    def _windows(self, node: Node, greens: dict[int, float]) -> dict[int, PhaseWindow]:
        out, t = {}, 0.0
        for ph in node.ordered_phases():
            out[ph.id] = PhaseWindow(t, greens[ph.id], ph.yellow, ph.startup_lost)
            t += greens[ph.id] + ph.clearance
        return out

    def _lane_weight(self, node_id: int, lanes: list[int]) -> float:
        if not self.net.params.person_delay:
            return 1.0
        ws = [self.est.lanes[(node_id, ln)].person_weight for ln in lanes]
        return float(np.mean(ws)) if ws else 1.0

    def _evaluate_group(self, nodes: list[int], cycle: int, now: float, current_offsets: dict[int, float] | None):
        p = self.net.params
        greens, windows = {}, {}
        for nid in nodes:
            node = self.net.nodes[nid]
            dec = self.decisions[nid]
            g = split_greens(node, cycle, dec.ratios, dec.weights)
            # se os mínimos estouram o ciclo, o grupo não cabe nesse ciclo
            if sum(g.values()) + sum(ph.clearance for ph in node.phases) > cycle + 0.5:
                return None
            greens[nid] = g
            windows[nid] = self._windows(node, g)
        tables: dict[tuple[int, int], list] = {}
        coordinated_lanes: set[tuple[int, int]] = set()
        members = set(nodes)
        for lk in self.net.links:
            if lk.src not in members or lk.dst not in members or lk.travel_time > p.coupling_max_travel_s:
                continue
            flow_s, sat_s, _ = self._lane_inputs(self.net.nodes[lk.src], now)
            flow_d, sat_d, queue_d = self._lane_inputs(self.net.nodes[lk.dst], now)
            li = LinkInput(
                link=lk,
                q_link=sum(flow_d[ln] for ln in lk.dst_lanes),
                q_src=sum(flow_s[ln] for ln in lk.src_lanes),
                s_src=sum(sat_s[ln] for ln in lk.src_lanes),
                s_dst=sum(sat_d[ln] for ln in lk.dst_lanes),
                q_dst_queue=sum(queue_d[ln] for ln in lk.dst_lanes),
                weight=self._lane_weight(lk.dst, lk.dst_lanes),
            )
            tables.setdefault((lk.src, lk.dst), []).append(
                link_cost_table(cycle, windows[lk.src][lk.src_phase], windows[lk.dst][lk.dst_phase], li, p.stops_weight)
            )
            coordinated_lanes.update((lk.dst, ln) for ln in lk.dst_lanes)
        res = optimize_offsets(cycle, nodes, tables, current_offsets)
        # faixas não coordenadas: chegada aleatória, atraso de Webster
        other = 0.0
        for nid in nodes:
            node = self.net.nodes[nid]
            flow, sat, _ = self._lane_inputs(node, now)
            for ph in node.phases:
                g_eff = greens[nid][ph.id] + ph.yellow - ph.startup_lost
                for ln in ph.lanes:
                    if (nid, ln) in coordinated_lanes:
                        continue
                    veh = flow[ln] * cycle / 3600.0
                    lam = g_eff / cycle
                    y = flow[ln] / max(sat[ln], 1e-6)
                    stop_frac = min(1.0, (1 - lam) / max(1 - y, 0.05))
                    d = signal_delay(cycle, g_eff, flow[ln], sat[ln])
                    other += self._lane_weight(nid, [ln]) * veh * (d + p.stops_weight * stop_frac)
        cost_h = (res.cost + other) * 3600.0 / cycle
        return GroupResult(nodes, float(cycle), res.offsets, cost_h, res.aog), greens

    def replan(self, now: float) -> dict[int, Plan]:
        p = self.net.params
        self.last_eval = now
        self.decisions = {nid: self._decide_node(node, now) for nid, node in self.net.nodes.items()}
        self._apply_spillback()
        eligible = {nid for nid, d in self.decisions.items() if d.mode in (PlanMode.NORMAL, PlanMode.LOW_DEMAND, PlanMode.SATURATED)}
        self.groups = []
        new_plans: dict[int, Plan] = {}
        for group in coordination_groups(self.net, eligible):
            decs = [self.decisions[n] for n in group]
            if all(d.mode == PlanMode.LOW_DEMAND for d in decs):
                c_crit = max(p.cycle_min, max(d.required_cycle for d in decs))
            else:
                c_crit = max(d.required_cycle for d in decs)
            c_crit = round_cycle(c_crit, p)
            # Webster dá o ponto de partida; o modelo de custo da rede escolhe o ciclo de fato
            candidates = sorted({min(p.cycle_max, max(p.cycle_min, c_crit + d)) for d in (-30, -20, -10, 0, 10, 20)})
            cur_cycles = {self.plans[n].cycle for n in group if n in self.plans}
            current_cycle = cur_cycles.pop() if len(cur_cycles) == 1 else None
            if current_cycle and p.cycle_min <= current_cycle <= p.cycle_max:
                candidates = sorted(set(candidates) | {current_cycle})
            current_offsets = {n: self.plans[n].offset for n in group if n in self.plans}
            results = []
            for c in candidates:
                r = self._evaluate_group(group, int(c), now, current_offsets)
                if r:
                    results.append(r)
            if not results:  # mínimos não cabem: usa o máximo permitido
                r = self._evaluate_group(group, int(p.cycle_max), now, current_offsets)
                results = [r] if r else []
            if not results:
                for n in group:
                    self.decisions[n].mode = PlanMode.FAILSAFE
                continue
            best = min(results, key=lambda r: r[0].cost_per_hour)
            if current_cycle:
                keep = [r for r in results if r[0].cycle == current_cycle]
                if keep and keep[0][0].cost_per_hour <= best[0].cost_per_hour * (1 + p.hysteresis):
                    best = keep[0]
            gres, greens = best
            self.groups.append(gres)
            for nid in group:
                new_plans[nid] = self._make_plan(nid, gres.cycle, gres.offsets[nid], greens[nid], self.decisions[nid].mode, now, exact_cycle=True)
        for nid, dec in self.decisions.items():
            if nid in new_plans:
                continue
            node = self.net.nodes[nid]
            if dec.mode == PlanMode.PREEMPT and nid in self.plans:
                new_plans[nid] = self.plans[nid]  # mantém o plano; a preempção é um comando à parte
                continue
            if dec.mode == PlanMode.FAILSAFE:
                cycle = max(90.0, sum(ph.floor_green + ph.clearance for ph in node.phases))
                greens = split_greens(node, cycle, {ph.id: 1.0 for ph in node.phases})
            else:  # isolado (sem vizinhos coordenáveis), FLASH ou PREEMPT sem plano anterior
                cycle = round_cycle(dec.required_cycle, p)
                greens = split_greens(node, cycle, dec.ratios, dec.weights)
                cycle = sum(greens.values()) + sum(ph.clearance for ph in node.phases)
            new_plans[nid] = self._make_plan(nid, cycle, 0.0, greens, dec.mode, now)
        self.plans = new_plans
        self.last_replan = now
        return new_plans

    def _make_plan(self, nid: int, cycle: float, offset: float, greens: dict[int, float], mode: PlanMode, now: float, exact_cycle: bool = False) -> Plan:
        node = self.net.nodes[nid]
        phases = [PhaseTiming(ph.id, round(greens[ph.id], 1), ph.yellow, ph.all_red) for ph in node.ordered_phases()]
        if exact_cycle:
            # o grupo inteiro precisa do MESMO ciclo; o arredondamento sobra na maior fase
            residual = round(cycle - sum(x.total for x in phases), 1)
            max(phases, key=lambda x: x.green).green += residual
        cycle = round(sum(x.total for x in phases), 1)
        old = self.plans.get(nid)
        same = old and old.mode == mode and abs(old.cycle - cycle) < 0.05 and abs(old.offset - offset) < 0.05 and all(
            abs(a.green - b.green) < 0.05 for a, b in zip(old.phases, phases)
        )
        if same:
            plan_id = old.plan_id
        else:
            self._plan_seq = (self._plan_seq + 1) & 0xFFFF
            plan_id = self._plan_seq
        valid = int(now + self.net.params.plan_validity_cycles * cycle)
        return Plan(nid, plan_id, self.t_ref, cycle, float(offset) % cycle, valid, mode, phases)

    def maybe_replan(self, now: float) -> dict[int, Plan] | None:
        if now - self.last_replan >= self.net.params.replan_interval_s:
            return self.replan(now)
        return None

    def snapshot(self, now: float) -> dict:
        return {
            "t": now,
            "nodes": {
                nid: {
                    "mode": d.mode.name,
                    "Y": round(sum(d.ratios.values()), 3),
                    "y": {ph: round(v, 3) for ph, v in d.ratios.items()},
                    "weights": d.weights,
                    "lanes": {
                        ln: {"q": round(self.est.flow(nid, ln, now)), "s": round(self.est.sat_flow(nid, ln, now)),
                             "fila": self.est.lanes[(nid, ln)].queue}
                        for ln in self.net.nodes[nid].lanes
                    },
                    "required_cycle": round(d.required_cycle, 1),
                    "notes": d.notes,
                    "plan": self.plans[nid].to_dict() if nid in self.plans else None,
                }
                for nid, d in self.decisions.items()
            },
            "groups": [
                {"nodes": g.nodes, "cycle": g.cycle, "offsets": g.offsets, "cost_person_s_per_h": round(g.cost_per_hour),
                 "arrivals_on_green": {f"{a}->{b}": round(v, 2) for (a, b), v in g.aog.items()}}
                for g in self.groups
            ],
        }
