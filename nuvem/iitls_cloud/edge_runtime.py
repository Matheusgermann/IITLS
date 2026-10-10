"""Executor local do plano, roda junto ao controlador semafórico (ou na Jetson).

A nuvem só PROPÕE planos (ciclo, defasagem, verdes). Quem garante a segurança e
reage em tempo real é este executor local, no esquema "coordenado-atuado":

- a fase coordenada (a da onda verde, sempre phases[0] do plano) começa no
  instante da defasagem e só cede no seu ponto de liberação;
- fases não coordenadas são atuadas pela câmera: pulam se não há ninguém
  esperando e encerram cedo (gap-out) quando a fila acaba, devolvendo o tempo
  à fase coordenada ("early return"), sem quebrar a onda;
- nunca corta verde antes do mínimo, sempre passa por amarelo e vermelho geral;
- troca de plano sem salto: corrige a defasagem aos poucos (no máx.
  `max_shift_frac` do ciclo por ciclo);
- preempção de emergência e prioridade a ônibus funcionam sem a nuvem;
- sem plano válido (enlace caiu), usa o plano fixo de segurança local.

Em campo, a saída de `step()` aciona o controlador existente (relé, NTCIP ou
protocolo do fabricante). O monitor de conflitos do controlador físico continua
sendo a última barreira de segurança, independente deste software.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .protocol import PhaseTiming, Plan, PlanMode, Preempt
from .timing import split_greens
from .topology import Node


@dataclass
class Interval:
    phase: int
    kind: str  # "G", "Y", "R" (vermelho geral após a fase), "F" (piscante)
    start: float
    end: float


class LocalController:
    def __init__(
        self,
        node: Node,
        demand: Callable[[int], bool] | None = None,
        max_shift_frac: float = 0.2,
        bus_max_extension: float = 10.0,
        gap_s: float = 2.0,
        fallback_cycle: float = 90.0,
    ):
        self.node = node
        self.demand = demand or (lambda phase: True)  # sem detecção = tempo fixo
        self.max_shift_frac = max_shift_frac
        self.bus_max_extension = bus_max_extension
        self.gap_s = gap_s
        self.fallback = self._fallback_plan(fallback_cycle)
        self.plan: Plan | None = None
        self.preempt: Preempt | None = None
        self._override: list[Interval] = []
        self.current: Interval | None = None
        # ciclo em execução
        self._order: list[PhaseTiming] = []
        self._force_off: list[float] = []
        self._cycle_start = 0.0
        self._cycle_end = 0.0
        self._idx = 0
        self._no_demand_since: float | None = None
        self._bus_ext = 0.0

    def _fallback_plan(self, cycle: float) -> Plan:
        cycle = max(cycle, sum(p.floor_green + p.clearance for p in self.node.phases))
        greens = split_greens(self.node, cycle, {p.id: 1.0 for p in self.node.phases})
        phases = [PhaseTiming(p.id, greens[p.id], p.yellow, p.all_red) for p in self.node.ordered_phases()]
        return Plan(self.node.id, 0, 0, sum(x.total for x in phases), 0.0, 2**31, PlanMode.FAILSAFE, phases)

    # ------------------------------------------------------------- comandos

    def receive_plan(self, plan: Plan) -> None:
        if plan.node == self.node.id:
            self.plan = plan

    def receive_preempt(self, cmd: Preempt) -> None:
        if cmd.node == self.node.id:
            self.preempt = cmd

    def bus_request(self, phase: int, eta: float, now: float) -> bool:
        """Estende o verde atual se o ônibus chega logo depois do fim previsto."""
        cur = self.current
        if not cur or cur.kind != "G" or cur.phase != phase or self._override:
            return False
        end = self._force_off[self._idx]
        need = now + eta + 1.0 - end
        if 0 < need and self._bus_ext + need <= self.bus_max_extension:
            self._force_off[self._idx] += need
            self._bus_ext += need
            return True
        return False

    # ------------------------------------------------------------- ciclo

    def active_plan(self, now: float) -> Plan:
        if self.plan and now <= self.plan.valid_until:
            return self.plan
        return self.fallback

    def _floor(self, phase: int) -> float:
        return self.node.phase(phase).floor_green

    def _build_cycle(self, start: float) -> None:
        plan = self.active_plan(start)
        target_c = plan.cycle
        greens = [p.green for p in plan.phases]
        if plan is not self.fallback:
            anchor = plan.t_ref + plan.offset
            err = (start - anchor) % target_c
            if err > target_c / 2:
                err -= target_c  # err > 0: atrasado -> encurta; err < 0: adiantado -> alonga
            if abs(err) > 0.5:
                limit = self.max_shift_frac * target_c
                delta = -max(-limit, min(limit, err))
                if delta < 0:
                    slack = [max(0.0, g - self._floor(p.phase)) for g, p in zip(greens, plan.phases)]
                    delta = max(delta, -sum(slack))
                else:
                    slack = [max(0.0, self.node.phase(p.phase).max_green - g) for g, p in zip(greens, plan.phases)]
                    delta = min(delta, sum(slack))
                tot = sum(slack)
                if tot > 0:
                    greens = [g + delta * s / tot for g, s in zip(greens, slack)]
        self._order = list(plan.phases)
        self._force_off = []
        t = start
        for p, g in zip(plan.phases, greens):
            self._force_off.append(t + g)
            t += g + p.yellow + p.all_red
        self._cycle_start, self._cycle_end = start, t
        self._bus_ext = 0.0

    def _start_green(self, idx: int, now: float) -> Interval:
        self._idx = idx
        self._no_demand_since = None
        return Interval(self._order[idx].phase, "G", now, self._force_off[idx])

    def _next_after(self, idx: int, now: float) -> Interval:
        """Depois do vermelho geral da fase idx: próxima fase com demanda, ou volta à coordenada."""
        for j in range(idx + 1, len(self._order)):
            if self.demand(self._order[j].phase) and now < self._force_off[j]:
                return self._start_green(j, now)
        # volta à coordenada (cedo = "early return"): ela vai até o ponto de liberação do PRÓXIMO ciclo
        late = now - self._cycle_end
        self._build_cycle(self._cycle_end if late < self._order[0].green else now)
        return self._start_green(0, now)

    # ------------------------------------------------------------- preempção

    def _build_preempt(self, now: float) -> None:
        cmd = self.preempt
        target = cmd.phase
        hold_end = max(now, cmd.t_start) + cmd.hold
        cur = self.current
        seq: list[Interval] = []
        if cur and cur.kind == "G" and cur.phase == target:
            seq.append(Interval(target, "G", cur.start, max(hold_end, now)))
        else:
            t = now
            if cur and cur.kind == "G":
                ph = self.node.phase(cur.phase)
                t = max(now, cur.start + ph.floor_green)
                seq.append(Interval(cur.phase, "G", cur.start, t))
                seq += [Interval(cur.phase, "Y", t, t + ph.yellow), Interval(cur.phase, "R", t + ph.yellow, t + ph.clearance)]
                t += ph.clearance
            elif cur and cur.kind == "Y":
                ph = self.node.phase(cur.phase)
                seq += [Interval(cur.phase, "Y", cur.start, cur.end), Interval(cur.phase, "R", cur.end, cur.end + ph.all_red)]
                t = cur.end + ph.all_red
            elif cur and cur.kind == "R":
                seq.append(Interval(cur.phase, "R", cur.start, cur.end))
                t = cur.end
            seq.append(Interval(target, "G", t, max(cmd.t_start, t) + cmd.hold))  # verde garantido de `hold` s
        t = seq[-1].end
        tp = self.node.phase(target)
        seq += [Interval(target, "Y", t, t + tp.yellow), Interval(target, "R", t + tp.yellow, t + tp.clearance)]
        self._override = seq
        self.preempt = None

    # ------------------------------------------------------------- passo

    def step(self, now: float) -> Interval:
        if self.preempt and now >= self.preempt.t_start - 1:
            self._build_preempt(now)
        if self._override:
            while self._override and now >= self._override[0].end:
                self._override.pop(0)
            if self._override:
                self.current = self._override[0]
                return self.current
            self._build_cycle(now)  # retoma; a transição realinha a defasagem
            self.current = self._start_green(0, now)
            return self.current

        plan = self.active_plan(now)
        if plan.mode == PlanMode.FLASH:
            self.current = Interval(-1, "F", now, now + 1)
            return self.current
        if self.current is None or self.current.kind == "F":
            self._build_cycle(now)
            self.current = self._start_green(0, now)

        for _ in range(2 * len(self._order) + 2):
            cur = self.current
            ph = self.node.phase(cur.phase)
            if cur.kind == "G":
                if self._idx == 0:
                    if now >= self._cycle_end:
                        self._build_cycle(self._cycle_end)
                    if now < max(self._force_off[0], cur.start + ph.floor_green):
                        break
                    others = [j for j in range(1, len(self._order)) if self.demand(self._order[j].phase)]
                    if not others:
                        break  # descansa no verde da coordenada
                else:
                    if self.demand(cur.phase):
                        self._no_demand_since = None
                    elif self._no_demand_since is None:
                        self._no_demand_since = now
                    min_done = now >= cur.start + ph.floor_green
                    gap_out = self._no_demand_since is not None and now - self._no_demand_since >= self.gap_s
                    if not (min_done and (now >= self._force_off[self._idx] or gap_out)):
                        break
                self.current = Interval(cur.phase, "Y", now, now + ph.yellow)
            elif cur.kind == "Y":
                if now < cur.end:
                    break
                self.current = Interval(cur.phase, "R", cur.end, cur.end + ph.all_red)
            elif cur.kind == "R":
                if now < cur.end:
                    break
                self.current = self._next_after(self._idx, now)
        return self.current
