"""
Contrato de mensagens IITLS entre a borda (Jetson em cada câmera) e a nuvem.

Formato primário: binário little-endian com CRC-16/CCITT no final (rádio/Starlink).
Formato secundário: JSON compacto com as mesmas chaves curtas (debug, HTTP).

Mensagens:
    Borda -> nuvem
        TELEMETRY  (0x01)  agregado por faixa a cada `interval` segundos
        EVENT      (0x02)  enviado na hora: emergência, ônibus, incidente, falha
        HEARTBEAT  (0x03)  saúde do nó quando não há telemetria
    Nuvem -> borda/controlador
        PLAN       (0x10)  plano semafórico: ciclo, defasagem, verdes por fase
        PREEMPT    (0x11)  preempção imediata (veículo de emergência)

Ver docs/protocolo-edge-nuvem.md para a especificação completa.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from enum import IntEnum

PROTOCOL_VERSION = 1

CLASSES = ("car", "moto", "bus", "truck", "bike", "ped")
VEHICLE_CLASSES = CLASSES[:5]


class MsgType(IntEnum):
    TELEMETRY = 0x01
    EVENT = 0x02
    HEARTBEAT = 0x03
    PLAN = 0x10
    PREEMPT = 0x11


class EventType(IntEnum):
    EMERGENCY_VEHICLE = 1  # value = ETA até a linha de retenção (décimos de s)
    BUS_APPROACHING = 2  # value = ETA (décimos de s)
    INCIDENT = 3  # value = duração estimada (s); faixa bloqueada
    INCIDENT_CLEARED = 4
    CAMERA_FAULT = 5  # value = código de falha
    CAMERA_RESTORED = 6


class HealthFlag(IntEnum):
    CAMERA_OK = 0x01
    LOW_LIGHT = 0x02
    OCCLUSION = 0x04
    CLOCK_UNSYNCED = 0x08
    DEGRADED_MODEL = 0x10  # rodando modelo de fallback / fps baixo


class PlanMode(IntEnum):
    NORMAL = 0
    LOW_DEMAND = 1
    SATURATED = 2
    PREEMPT = 3
    FAILSAFE = 4
    FLASH = 5


def crc16_ccitt(data: bytes, crc: int = 0xFFFF) -> int:
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            crc = ((crc << 1) ^ 0x1021) if crc & 0x8000 else (crc << 1)
            crc &= 0xFFFF
    return crc


def _u8(value: float) -> int:
    return max(0, min(255, int(round(value))))


def _u16(value: float) -> int:
    return max(0, min(0xFFFF, int(round(value))))


class ProtocolError(ValueError):
    pass


def _seal(body: bytes) -> bytes:
    return body + struct.pack("<H", crc16_ccitt(body))


def _open(data: bytes) -> bytes:
    if len(data) < 4:
        raise ProtocolError("mensagem curta demais")
    body, (crc,) = data[:-2], struct.unpack("<H", data[-2:])
    if crc16_ccitt(body) != crc:
        raise ProtocolError("CRC inválido")
    if body[0] != PROTOCOL_VERSION:
        raise ProtocolError(f"versão {body[0]} não suportada")
    return body


# ---------------------------------------------------------------------------
# Telemetria
# ---------------------------------------------------------------------------

_TEL_HEADER = struct.Struct("<BBHBHIBBB")  # 14 bytes
_TEL_LANE = struct.Struct("<B6BBBBB")  # 11 bytes


@dataclass
class LaneStats:
    lane: int
    counts: dict[str, int] = field(default_factory=dict)  # veículos que cruzaram a linha virtual
    queue: int = 0  # veículos parados na fila no fim do intervalo
    occupancy: float = 0.0  # 0..1, fração do tempo com a zona ocupada
    speed_kmh: float = 0.0  # velocidade média dos veículos em movimento
    sat_obs: float = 0.0  # veic/h escoados na retenção com fila e verde (0 = não medido)

    def vehicles(self) -> int:
        return sum(self.counts.get(c, 0) for c in VEHICLE_CLASSES)


@dataclass
class Telemetry:
    node: int
    cam: int
    seq: int
    t_end: int  # unix s do fim da janela
    interval: int  # duração da janela em s
    lanes: list[LaneStats]
    flags: int = HealthFlag.CAMERA_OK

    def to_bytes(self) -> bytes:
        body = bytearray(
            _TEL_HEADER.pack(
                PROTOCOL_VERSION,
                MsgType.TELEMETRY,
                self.node,
                self.cam,
                self.seq & 0xFFFF,
                self.t_end,
                _u8(self.interval),
                self.flags,
                len(self.lanes),
            )
        )
        for ln in self.lanes:
            body += _TEL_LANE.pack(
                ln.lane,
                *(_u8(ln.counts.get(c, 0)) for c in CLASSES),
                _u8(ln.queue),
                _u8(ln.occupancy * 200),
                _u8(ln.speed_kmh),
                _u8(ln.sat_obs / 10),
            )
        return _seal(bytes(body))

    def to_json(self) -> str:
        return json.dumps(
            {
                "v": PROTOCOL_VERSION,
                "m": int(MsgType.TELEMETRY),
                "n": self.node,
                "c": self.cam,
                "s": self.seq,
                "t": self.t_end,
                "dt": self.interval,
                "f": int(self.flags),
                "l": [
                    [ln.lane, *(ln.counts.get(c, 0) for c in CLASSES), ln.queue, _u8(ln.occupancy * 200), _u8(ln.speed_kmh), _u8(ln.sat_obs / 10)]
                    for ln in self.lanes
                ],
            },
            separators=(",", ":"),
        )

    @staticmethod
    def _lane_from_row(row) -> LaneStats:
        lane, *rest = row
        counts = {c: int(v) for c, v in zip(CLASSES, rest[:6]) if v}
        sat = rest[9] * 10.0 if len(rest) > 9 else 0.0
        return LaneStats(lane=int(lane), counts=counts, queue=int(rest[6]), occupancy=rest[7] / 200.0, speed_kmh=float(rest[8]), sat_obs=sat)

    @classmethod
    def from_bytes_body(cls, body: bytes) -> "Telemetry":
        _, _, node, cam, seq, t_end, interval, flags, n = _TEL_HEADER.unpack_from(body, 0)
        expected = _TEL_HEADER.size + n * _TEL_LANE.size
        if len(body) != expected:
            raise ProtocolError(f"tamanho {len(body)} != esperado {expected}")
        lanes = [cls._lane_from_row(_TEL_LANE.unpack_from(body, _TEL_HEADER.size + i * _TEL_LANE.size)) for i in range(n)]
        return cls(node=node, cam=cam, seq=seq, t_end=t_end, interval=interval, lanes=lanes, flags=flags)

    @classmethod
    def from_dict(cls, d: dict) -> "Telemetry":
        return cls(
            node=d["n"], cam=d.get("c", 0), seq=d.get("s", 0), t_end=d["t"], interval=d["dt"],
            flags=d.get("f", HealthFlag.CAMERA_OK), lanes=[cls._lane_from_row(r) for r in d["l"]],
        )


# ---------------------------------------------------------------------------
# Eventos e heartbeat
# ---------------------------------------------------------------------------

_EVT = struct.Struct("<BBHBHIBBH")  # 14 bytes


@dataclass
class Event:
    node: int
    cam: int
    seq: int
    t: int
    kind: EventType
    lane: int = 0
    value: int = 0

    def to_bytes(self) -> bytes:
        return _seal(_EVT.pack(PROTOCOL_VERSION, MsgType.EVENT, self.node, self.cam, self.seq & 0xFFFF, self.t, self.kind, self.lane, _u16(self.value)))

    def to_json(self) -> str:
        return json.dumps({"v": 1, "m": int(MsgType.EVENT), "n": self.node, "c": self.cam, "s": self.seq, "t": self.t, "e": int(self.kind), "ln": self.lane, "x": self.value}, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        return cls(node=d["n"], cam=d.get("c", 0), seq=d.get("s", 0), t=d["t"], kind=EventType(d["e"]), lane=d.get("ln", 0), value=d.get("x", 0))


_HB = struct.Struct("<BBHBHIBBBHH")  # 18 bytes (+2 CRC)

PHASE_UNKNOWN = 0xFF  # piscante, preempção sem fase definida ou controlador sem resposta


@dataclass
class Heartbeat:
    """Enviado a cada 1 s. Sem heartbeat por 5 s, a nuvem marca o cruzamento como fora do ar
    (a borda segue o último plano até `valid_until`)."""

    node: int
    cam: int
    seq: int
    t: int
    flags: int
    temp_c: int = 0  # temperatura da Jetson, sinal de saúde do hardware
    phase: int = PHASE_UNKNOWN  # fase verde atual no controlador
    phase_elapsed: float = 0.0  # s na fase atual
    plan_id: int = 0  # plano em execução (0 = plano de segurança local)

    def to_bytes(self) -> bytes:
        return _seal(_HB.pack(
            PROTOCOL_VERSION, MsgType.HEARTBEAT, self.node, self.cam, self.seq & 0xFFFF, self.t, self.flags,
            _u8(self.temp_c), self.phase & 0xFF, _u16(self.phase_elapsed * 10), self.plan_id & 0xFFFF,
        ))

    def to_json(self) -> str:
        return json.dumps({"v": 1, "m": int(MsgType.HEARTBEAT), "n": self.node, "c": self.cam, "s": self.seq, "t": self.t,
                           "f": int(self.flags), "tc": self.temp_c, "ph": self.phase, "pe": round(self.phase_elapsed, 1),
                           "id": self.plan_id}, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Plano (nuvem -> borda)
# ---------------------------------------------------------------------------

_PLAN_HEADER = struct.Struct("<BBHHIHHIBB")  # 20 bytes
_PLAN_PHASE = struct.Struct("<BHBB")  # 5 bytes


@dataclass
class PhaseTiming:
    phase: int
    green: float  # s
    yellow: float
    all_red: float

    @property
    def total(self) -> float:
        return self.green + self.yellow + self.all_red


@dataclass
class Plan:
    """Plano de um cruzamento.

    O ciclo começa (fase phases[0] fica verde) nos instantes
    t_ref + offset + k * cycle, para todo k inteiro. t_ref é comum à rede
    (relógio sincronizado por GPS/NTP), por isso as defasagens criam a onda verde.
    O controlador local executa o plano até `valid_until`; depois disso cai no
    plano de segurança local (ver edge_runtime.LocalController).
    """

    node: int
    plan_id: int
    t_ref: int
    cycle: float
    offset: float
    valid_until: int
    mode: PlanMode
    phases: list[PhaseTiming]

    def phase_start(self, index: int) -> float:
        return sum(p.total for p in self.phases[:index])

    def to_bytes(self) -> bytes:
        body = bytearray(
            _PLAN_HEADER.pack(
                PROTOCOL_VERSION, MsgType.PLAN, self.node, self.plan_id & 0xFFFF, self.t_ref,
                _u16(self.cycle * 10), _u16((self.offset % self.cycle) * 10), self.valid_until, self.mode, len(self.phases),
            )
        )
        for p in self.phases:
            body += _PLAN_PHASE.pack(p.phase, _u16(p.green * 10), _u8(p.yellow * 10), _u8(p.all_red * 10))
        return _seal(bytes(body))

    @classmethod
    def from_bytes_body(cls, body: bytes) -> "Plan":
        _, _, node, plan_id, t_ref, cycle, offset, valid_until, mode, n = _PLAN_HEADER.unpack_from(body, 0)
        phases = []
        for i in range(n):
            ph, g, y, ar = _PLAN_PHASE.unpack_from(body, _PLAN_HEADER.size + i * _PLAN_PHASE.size)
            phases.append(PhaseTiming(ph, g / 10, y / 10, ar / 10))
        return cls(node, plan_id, t_ref, cycle / 10, offset / 10, valid_until, PlanMode(mode), phases)

    def to_dict(self) -> dict:
        return {
            "v": 1, "m": int(MsgType.PLAN), "n": self.node, "id": self.plan_id, "tr": self.t_ref,
            "C": round(self.cycle, 1), "o": round(self.offset % self.cycle, 1), "vu": self.valid_until,
            "md": int(self.mode), "p": [[p.phase, round(p.green, 1), p.yellow, p.all_red] for p in self.phases],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(
            d["n"], d["id"], d["tr"], d["C"], d["o"], d["vu"], PlanMode(d["md"]),
            [PhaseTiming(int(a), float(b), float(c), float(e)) for a, b, c, e in d["p"]],
        )


_PRE = struct.Struct("<BBHBIB")  # 10 bytes


@dataclass
class Preempt:
    node: int
    phase: int  # fase que deve ficar verde
    t_start: int
    hold: int  # segundos de verde garantido

    def to_bytes(self) -> bytes:
        return _seal(_PRE.pack(PROTOCOL_VERSION, MsgType.PREEMPT, self.node, self.phase, self.t_start, _u8(self.hold)))

    def to_dict(self) -> dict:
        return {"v": 1, "m": int(MsgType.PREEMPT), "n": self.node, "ph": self.phase, "t": self.t_start, "h": self.hold}


# ---------------------------------------------------------------------------
# Decodificação genérica
# ---------------------------------------------------------------------------


def decode(data: bytes | str | dict):
    """Decodifica binário (com CRC) ou JSON compacto e devolve a dataclass certa."""
    if isinstance(data, (bytes, bytearray)) and data[:1] not in (b"{",):
        body = _open(bytes(data))
        kind = body[1]
        if kind == MsgType.TELEMETRY:
            return Telemetry.from_bytes_body(body)
        if kind == MsgType.EVENT:
            _, _, node, cam, seq, t, ev, lane, value = _EVT.unpack(body)
            return Event(node, cam, seq, t, EventType(ev), lane, value)
        if kind == MsgType.HEARTBEAT:
            _, _, node, cam, seq, t, flags, temp, phase, elapsed, plan_id = _HB.unpack(body)
            return Heartbeat(node, cam, seq, t, flags, temp, phase, elapsed / 10, plan_id)
        if kind == MsgType.PLAN:
            return Plan.from_bytes_body(body)
        if kind == MsgType.PREEMPT:
            _, _, node, phase, t, hold = _PRE.unpack(body)
            return Preempt(node, phase, t, hold)
        raise ProtocolError(f"tipo {kind} desconhecido")
    d = data if isinstance(data, dict) else json.loads(data)
    kind = d.get("m")
    if kind == MsgType.TELEMETRY:
        return Telemetry.from_dict(d)
    if kind == MsgType.EVENT:
        return Event.from_dict(d)
    if kind == MsgType.HEARTBEAT:
        return Heartbeat(d["n"], d.get("c", 0), d.get("s", 0), d["t"], d.get("f", 1), d.get("tc", 0),
                         d.get("ph", PHASE_UNKNOWN), d.get("pe", 0.0), d.get("id", 0))
    if kind == MsgType.PLAN:
        return Plan.from_dict(d)
    raise ProtocolError(f"tipo {kind} desconhecido")
