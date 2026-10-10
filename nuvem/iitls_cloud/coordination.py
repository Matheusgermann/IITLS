"""Cálculo de defasagens (onda verde) entre cruzamentos.

Modelo (inspirado no TRANSYT, simplificado para rodar em milissegundos):

1. Para cada ligação A->B, o perfil de saída do pelotão em A (descarga da fila
   na taxa de saturação, depois o fluxo de chegada) é deslocado pelo tempo de
   viagem e disperso (dispersão de pelotão de Robertson).
2. Em B, uma fila determinística segundo a segundo dá atraso e paradas para cada
   defasagem relativa possível Δ = θ_B - θ_A (0..C-1). Isso vira uma tabela de
   custo por ligação, calculada uma vez por replanejamento.
3. O custo total é a soma das tabelas. Em rede sem ciclos (corredor, árvore) a
   programação dinâmica dá o ótimo global exato. Em grade (com ciclos) parte da
   árvore geradora de maior peso e refina por descida coordenada.

Os dois sentidos de uma via entram juntos, ponderados pelo volume (e pela
ocupação de pessoas se `person_delay`), então a onda favorece o sentido mais
carregado sem abandonar o outro.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .topology import Link, Network

ROBERTSON_ALPHA = 0.35
ROBERTSON_BETA = 0.8


@dataclass
class PhaseWindow:
    start: float  # início do verde relativo ao início do ciclo (s)
    green: float
    yellow: float
    startup_lost: float = 2.0


@dataclass
class LinkInput:
    link: Link
    q_link: float  # chegada total nas faixas de destino (veic/h)
    q_src: float  # fluxo nas faixas de origem (veic/h)
    s_src: float  # saturação das faixas de origem (veic/h)
    s_dst: float  # saturação das faixas de destino (veic/h)
    q_dst_queue: float = 0.0  # fila atual no destino (veic)
    weight: float = 1.0  # pessoas por veículo


def _departure_profile(cycle: int, win: PhaseWindow, q_up: float, s_up: float) -> np.ndarray:
    """Forma do perfil de saída (soma 1) do verde da fase de origem."""
    prof = np.zeros(cycle)
    g = max(1, int(round(win.green + win.yellow - win.startup_lost)))
    start = int(round(win.start + win.startup_lost))
    red = cycle - g
    q, s = q_up / 3600.0, max(s_up / 3600.0, 1e-6)
    queue = q * red
    tq = g if s <= q else min(g, queue / (s - q))
    for k in range(g):
        prof[(start + k) % cycle] = s if k < tq else q
    total = prof.sum()
    return prof / total if total > 0 else np.full(cycle, 1.0 / cycle)


def _disperse(prof: np.ndarray, travel: float) -> np.ndarray:
    cycle = len(prof)
    t_shift = int(round(ROBERTSON_BETA * travel))
    f = 1.0 / (1.0 + ROBERTSON_ALPHA * ROBERTSON_BETA * travel)
    shifted = np.roll(prof, t_shift)
    out = np.zeros(cycle)
    prev = shifted.mean()
    for _ in range(4):  # converge o regime periódico
        for t in range(cycle):
            prev = f * shifted[t] + (1 - f) * prev
            out[t] = prev
    return out / out.sum() if out.sum() > 0 else out


def link_cost_table(cycle: int, src_win: PhaseWindow, dst_win: PhaseWindow, li: LinkInput, stops_weight: float) -> tuple[np.ndarray, np.ndarray]:
    """Custo (atraso + K*paradas, ponderado) por ciclo e fração de chegadas no verde, para Δ=0..C-1."""
    vol = li.q_link * cycle / 3600.0
    platoon = min(li.q_src, li.q_link) * cycle / 3600.0
    shape = _disperse(_departure_profile(cycle, src_win, li.q_src, li.s_src), li.link.travel_time)
    arrivals = platoon * shape + (vol - platoon) / cycle  # veic por segundo do ciclo (base de tempo de A)

    deltas = np.arange(cycle)[:, None]
    t = np.arange(cycle)[None, :]
    g = max(1, int(round(dst_win.green + dst_win.yellow - dst_win.startup_lost)))
    e0 = dst_win.start + dst_win.startup_lost
    green = ((t - deltas - e0) % cycle) < g  # [Δ, t]
    cap = li.s_dst / 3600.0

    queue = np.full(cycle, float(li.q_dst_queue))
    delay = np.zeros(cycle)
    stops = np.zeros(cycle)
    for k in range(3 * cycle):
        tt = k % cycle
        a = arrivals[tt]
        gm = green[:, tt]
        last = k >= 2 * cycle
        if last:
            stops += a * np.where(gm & (queue < 0.5), 0.0, 1.0)
        queue = queue + a
        queue = queue - np.minimum(queue, cap * gm)
        if last:
            delay += queue
    aog = (green * arrivals[None, :]).sum(axis=1) / max(arrivals.sum(), 1e-9)
    return li.weight * (delay + stops_weight * stops), aog


@dataclass
class OffsetResult:
    offsets: dict[int, int]
    cost: float  # custo por ciclo
    aog: dict[tuple[int, int], float]  # fração de chegadas no verde por ligação (src, dst)


def optimize_offsets(
    cycle: int,
    nodes: list[int],
    tables: dict[tuple[int, int], list[tuple[np.ndarray, np.ndarray]]],
    current: dict[int, float] | None = None,
) -> OffsetResult:
    """Minimiza a soma das tabelas de custo sobre θ (inteiros mod C).

    tables[(a, b)] = lista de (custo[Δ], aog[Δ]) das ligações a->b, Δ = θ_b - θ_a.
    """
    C = cycle
    # custo por par não direcionado, indexado por δ = θ_v - θ_u (u < v)
    pair: dict[tuple[int, int], np.ndarray] = {}
    for (a, b), lst in tables.items():
        u, v = (a, b) if a < b else (b, a)
        arr = pair.setdefault((u, v), np.zeros(C))
        for cost, _ in lst:
            arr += cost if a == u else np.roll(cost[::-1], 1)  # Δ_ab = -δ
    adj: dict[int, list[int]] = defaultdict(list)
    for u, v in pair:
        adj[u].append(v)
        adj[v].append(u)

    def pc(u: int, v: int) -> np.ndarray:
        """custo[θ_v - θ_u]"""
        if (u, v) in pair:
            return pair[(u, v)]
        return np.roll(pair[(v, u)][::-1], 1)

    offsets: dict[int, int] = {}
    # árvore geradora de maior "peso" (amplitude da tabela = quanto a defasagem importa)
    seen: set[int] = set()
    tree_edges: list[tuple[int, int]] = []
    for root in sorted(nodes, key=lambda n: -sum(np.ptp(pc(n, m)) for m in adj[n])):
        if root in seen:
            continue
        seen.add(root)
        frontier = [(np.ptp(pc(root, m)), root, m) for m in adj[root]]
        while frontier:
            frontier.sort()
            _, u, v = frontier.pop()
            if v in seen:
                continue
            seen.add(v)
            tree_edges.append((u, v))
            frontier += [(np.ptp(pc(v, w)), v, w) for w in adj[v] if w not in seen]
    children: dict[int, list[int]] = defaultdict(list)
    has_parent = set()
    for u, v in tree_edges:
        children[u].append(v)
        has_parent.add(v)
    roots = [n for n in nodes if n not in has_parent]

    # DP na árvore: f[v][θ_v] = custo ótimo da subárvore dado θ_v
    idx = np.arange(C)
    best_child: dict[int, np.ndarray] = {}
    f: dict[int, np.ndarray] = {}

    def solve(v: int) -> np.ndarray:
        acc = np.zeros(C)
        for w in children[v]:
            fw = solve(w)
            # M[θ_v, θ_w] = pc(v,w)[θ_w-θ_v] + f_w[θ_w]
            m = pc(v, w)[(idx[None, :] - idx[:, None]) % C] + fw[None, :]
            best_child[w] = m.argmin(axis=1)
            acc += m.min(axis=1)
        f[v] = acc
        return acc

    def assign(v: int) -> None:
        for w in children[v]:
            offsets[w] = int(best_child[w][offsets[v]])
            assign(w)

    for r in roots:
        solve(r)
        offsets[r] = int(f[r].argmin())
        assign(r)

    def node_cost(v: int, theta: np.ndarray) -> np.ndarray:
        tot = np.zeros(len(theta))
        for w in adj[v]:
            tot += pc(v, w)[(offsets[w] - theta) % C]
        return tot

    # refinamento para redes com ciclos (grade)
    if len(pair) > len(tree_edges):
        for _ in range(20):
            changed = False
            for v in nodes:
                c = node_cost(v, idx)
                b = int(c.argmin())
                if c[b] < c[offsets[v]] - 1e-9:
                    offsets[v] = b
                    changed = True
            if not changed:
                break

    # translada tudo para mexer o mínimo nos planos atuais (custo é invariante a translação)
    if current:
        shifts = np.arange(C)
        dist = np.zeros(C)
        for v in nodes:
            if v in current:
                d = (offsets[v] + shifts - current[v]) % C
                dist += np.minimum(d, C - d)
        s = int(dist.argmin())
        offsets = {v: (o + s) % C for v, o in offsets.items()}

    total = sum(float(pc(u, v)[(offsets[v] - offsets[u]) % C]) for u, v in pair)
    aog = {}
    for (a, b), lst in tables.items():
        d = (offsets[b] - offsets[a]) % C
        vals = [x[1][d] for x in lst]
        aog[(a, b)] = float(np.mean(vals)) if vals else 0.0
    return OffsetResult(offsets=offsets, cost=total, aog=aog)


def coordination_groups(net: Network, eligible: set[int]) -> list[list[int]]:
    """Componentes conexas por ligações curtas o bastante para manter o pelotão."""
    parent = {n: n for n in eligible}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for lk in net.links:
        if lk.src in eligible and lk.dst in eligible and lk.travel_time <= net.params.coupling_max_travel_s:
            parent[find(lk.src)] = find(lk.dst)
    groups: dict[int, list[int]] = defaultdict(list)
    for n in eligible:
        groups[find(n)].append(n)
    return [sorted(g) for g in groups.values()]
