"""Transporte MQTT (sem TLS, dentro do túnel WireGuard) entre as bordas e a nuvem.

Tópicos (N = id do cruzamento; o payload é sempre o binário de protocol.py):

    iitls/N/up    borda -> nuvem   telemetria (5 s, QoS 0), heartbeat (1 s, QoS 0), eventos (QoS 1)
    iitls/N/plan  nuvem -> borda   PLAN, QoS 1, RETIDO: a borda recebe o último plano ao reconectar
    iitls/N/cmd   nuvem -> borda   PREEMPT, QoS 1, não retido (comando vale só no momento)

Segurança: a identidade do cruzamento é a chave WireGuard; o broker aplica ACL para que o
cruzamento N só publique em iitls/N/up e só assine iitls/N/plan e iitls/N/cmd. A nuvem ainda
descarta mensagens cujo `node` não bate com o tópico.
"""

from __future__ import annotations

import logging
from typing import Protocol

from .edge_runtime import LocalController
from .protocol import Event, Plan, Preempt, ProtocolError, decode

log = logging.getLogger("iitls.mqtt")

TOPIC_UP = "iitls/{node}/up"
TOPIC_PLAN = "iitls/{node}/plan"
TOPIC_CMD = "iitls/{node}/cmd"


class MqttClient(Protocol):  # subconjunto do paho.mqtt.client.Client que usamos
    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False): ...
    def subscribe(self, topic: str, qos: int = 0): ...


def node_from_topic(topic: str) -> int | None:
    parts = topic.split("/")
    if len(parts) == 3 and parts[0] == "iitls" and parts[1].isdigit():
        return int(parts[1])
    return None


class CloudBridge:
    """Lado nuvem: liga o broker ao Service (api.Service)."""

    def __init__(self, svc, client: MqttClient):
        self.svc = svc
        self.client = client
        self.published: dict[int, int] = {}  # nó -> plan_id já publicado

    def start(self) -> None:
        self.client.subscribe(TOPIC_UP.format(node="+"), qos=1)
        self.publish_plans()

    def on_message(self, topic: str, payload: bytes) -> None:
        node = node_from_topic(topic)
        try:
            msg = decode(payload)
        except ProtocolError as exc:
            log.warning("descartado de %s: %s", topic, exc)
            return
        if node is None or getattr(msg, "node", None) != node:
            log.warning("nó da mensagem não bate com o tópico %s", topic)
            return
        _, cmds = self.svc.handle(msg)
        for cmd in cmds:
            self.client.publish(TOPIC_CMD.format(node=cmd.node), cmd.to_bytes(), qos=1, retain=False)
        self.publish_plans()

    def publish_plans(self) -> None:
        """Publica (retido) só os planos que mudaram desde a última publicação."""
        for nid, plan in self.svc.coord.plans.items():
            if self.published.get(nid) != plan.plan_id:
                self.client.publish(TOPIC_PLAN.format(node=nid), plan.to_bytes(), qos=1, retain=True)
                self.published[nid] = plan.plan_id

    def attach_paho(self, paho_client) -> None:
        paho_client.on_message = lambda _c, _u, m: self.on_message(m.topic, m.payload)
        paho_client.on_connect = lambda *_a, **_k: self.start()  # reassina e republica ao reconectar


class EdgeLink:
    """Lado borda: entrega plano/preempção ao executor local e monta o heartbeat."""

    def __init__(self, node: int, local: LocalController, client: MqttClient):
        self.node = node
        self.local = local
        self.client = client

    def start(self) -> None:
        self.client.subscribe(TOPIC_PLAN.format(node=self.node), qos=1)
        self.client.subscribe(TOPIC_CMD.format(node=self.node), qos=1)

    def on_message(self, topic: str, payload: bytes) -> None:
        try:
            msg = decode(payload)
        except ProtocolError as exc:
            log.warning("descartado de %s: %s", topic, exc)
            return
        if isinstance(msg, Plan):
            self.local.receive_plan(msg)
        elif isinstance(msg, Preempt):
            self.local.receive_preempt(msg)

    def send(self, msg) -> None:
        qos = 1 if isinstance(msg, Event) else 0
        self.client.publish(TOPIC_UP.format(node=self.node), msg.to_bytes(), qos=qos, retain=False)
