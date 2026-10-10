"""Ciclo e divisão de verdes por cruzamento (Webster + restrições de segurança)."""

from __future__ import annotations

import math

from .topology import Node, Params


def phase_ratios(node: Node, flow: dict[int, float], sat: dict[int, float], queue: dict[int, float], cycle_hint: float) -> dict[int, float]:
    """Razão de fluxo crítica y por fase: max(q/s) entre as faixas da fase.

    A fila residual entra como demanda extra que precisa ser escoada em um ciclo.
    """
    ratios = {}
    for ph in node.phases:
        y = 0.0
        for ln in ph.lanes:
            s = max(sat.get(ln, 0.0), 1e-6)
            q = flow.get(ln, 0.0) + queue.get(ln, 0.0) * 3600.0 / max(cycle_hint, 1.0)
            y = max(y, q / s)
        ratios[ph.id] = y
    return ratios


def min_cycle(node: Node) -> float:
    return sum(p.floor_green + p.clearance for p in node.phases)


def webster_cycle(node: Node, ratios: dict[int, float], params: Params) -> float:
    lost = sum(p.lost_time for p in node.phases)
    big_y = sum(ratios.values())
    lo = max(params.cycle_min, min_cycle(node))
    if big_y >= 0.95:
        return params.cycle_max
    c0 = (1.5 * lost + 5.0) / (1.0 - big_y)
    return float(min(params.cycle_max, max(lo, c0)))


def round_cycle(c: float, params: Params) -> float:
    step = params.cycle_step
    return float(min(params.cycle_max, math.ceil(c / step) * step))


def split_greens(node: Node, cycle: float, ratios: dict[int, float], weights: dict[int, float] | None = None) -> dict[int, float]:
    """Distribui o verde disponível proporcional a y (ponderado), respeitando mínimos e máximos.

    Water-filling: calcula todos com o mesmo orçamento, fixa quem viola piso/teto,
    redistribui o resto entre os livres e repete. Se o ciclo não comporta os
    mínimos, os mínimos ganham (segurança > eficiência).
    """
    weights = weights or {}
    available = cycle - sum(p.clearance for p in node.phases)
    demand = {p.id: max(ratios.get(p.id, 0.0), 0.01) * weights.get(p.id, 1.0) for p in node.phases}
    fixed: dict[int, float] = {}
    free = {p.id: p for p in node.phases}
    while free:
        budget = available - sum(fixed.values())
        total = sum(demand[i] for i in free)
        trial = {pid: budget * demand[pid] / total for pid in free}
        low = {pid: ph.floor_green for pid, ph in free.items() if trial[pid] < ph.floor_green}
        high = {pid: ph.max_green for pid, ph in free.items() if trial[pid] > ph.max_green}
        if not low and not high:
            fixed.update(trial)
            break
        # fixa primeiro os pisos (falta de verde é o caso crítico); tetos na rodada seguinte
        for pid, g in (low or high).items():
            fixed[pid] = g
            del free[pid]
    if sum(fixed.values()) < available - 1e-6 and len(fixed) == len(node.phases):
        # todos no teto: sobra vai proporcionalmente (o ciclo pedido manda)
        extra = available - sum(fixed.values())
        tot = sum(demand.values())
        fixed = {pid: g + extra * demand[pid] / tot for pid, g in fixed.items()}
    return {ph.id: max(ph.floor_green, fixed[ph.id]) for ph in node.phases}


def signal_delay(cycle: float, green: float, flow_vph: float, sat_vph: float, period_h: float = 0.25) -> float:
    """Atraso médio de controle (s/veic), fórmula do HCM: uniforme + incremental.

    Vale também para x > 1 (supersaturação), onde o termo incremental cresce
    linearmente com a fila acumulada no período de análise.
    """
    if flow_vph <= 0:
        return 0.0
    lam = min(max(green / cycle, 0.01), 0.999)
    cap = max(sat_vph * lam, 1.0)
    x = flow_vph / cap
    d1 = 0.5 * cycle * (1 - lam) ** 2 / (1 - min(1.0, x) * lam)
    d2 = 900 * period_h * ((x - 1) + math.sqrt((x - 1) ** 2 + 4 * x / (cap * period_h)))
    return d1 + d2
