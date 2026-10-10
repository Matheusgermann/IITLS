"""Serviço na nuvem.

- UDP :9700  -> telemetria/eventos binários das Jetsons (caminho principal, menor overhead).
               A resposta a cada mensagem é o plano vigente do nó (binário), então a
               borda recebe o plano sem precisar de conexão aberta nem polling extra.
- MQTT       -> transporte de produção (sem TLS, dentro do WireGuard), ver mqtt_bridge.py.
               Liga com IITLS_MQTT_HOST (e IITLS_MQTT_PORT, padrão 1883).
- HTTP :8000 -> mesma ingestão em JSON/binário, consulta de planos e estado (dashboard).

Rodar:  uvicorn iitls_cloud.api:app --host 0.0.0.0 --port 8000
        (variável IITLS_TOPOLOGY aponta para o JSON da rede)
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, Response

from .controller import Coordinator
from .protocol import Event, Plan, ProtocolError, Telemetry, decode
from .topology import Network

log = logging.getLogger("iitls")


class Service:
    def __init__(self, net: Network, clock=time.time):
        self.clock = clock
        self.coord = Coordinator(net, t_ref=0)
        self.coord.replan(self.clock())
        self.outbox: list = []  # preempções a entregar
        self.lock = threading.Lock()  # MQTT (thread do paho), UDP e HTTP chamam ao mesmo tempo

    def handle(self, msg) -> tuple[Plan | None, list]:
        with self.lock:
            return self._handle(msg)

    def _handle(self, msg) -> tuple[Plan | None, list]:
        now = self.clock()
        cmds = self.coord.ingest(msg, now)
        if isinstance(msg, Event) and cmds:
            self.coord.replan(now)  # tira os nós em preempção da coordenação
        else:
            self.coord.maybe_replan(now)
        self.outbox.extend(cmds)
        node = getattr(msg, "node", None)
        return self.coord.plans.get(node), cmds


class UdpProtocol(asyncio.DatagramProtocol):
    def __init__(self, svc: Service):
        self.svc = svc
        self.transport = None

    def connection_made(self, transport):
        self.transport = transport

    def datagram_received(self, data: bytes, addr):
        try:
            msg = decode(data)
        except ProtocolError as exc:
            log.warning("descartado de %s: %s", addr, exc)
            return
        plan, cmds = self.svc.handle(msg)
        for c in cmds:  # TODO: endereço de cada nó vem do registro de dispositivos
            if c.node == getattr(msg, "node", None):
                self.transport.sendto(c.to_bytes(), addr)
        if plan is not None:
            self.transport.sendto(plan.to_bytes(), addr)


def create_app(net: Network | None = None) -> FastAPI:
    net = net or Network.load(os.environ.get("IITLS_TOPOLOGY", "topologia.json"))
    svc = Service(net)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        transport = mqtt = None
        port = int(os.environ.get("IITLS_UDP_PORT", "9700"))
        if port:
            loop = asyncio.get_running_loop()
            transport, _ = await loop.create_datagram_endpoint(lambda: UdpProtocol(svc), local_addr=("0.0.0.0", port))
        host = os.environ.get("IITLS_MQTT_HOST")
        if host:
            import paho.mqtt.client as paho

            from .mqtt_bridge import CloudBridge

            mqtt = paho.Client(paho.CallbackAPIVersion.VERSION2, client_id="iitls-nuvem")
            CloudBridge(svc, mqtt).attach_paho(mqtt)
            mqtt.connect_async(host, int(os.environ.get("IITLS_MQTT_PORT", "1883")))
            mqtt.loop_start()
        yield
        if transport:
            transport.close()
        if mqtt:
            mqtt.loop_stop()
            mqtt.disconnect()

    app = FastAPI(title="IITLS nuvem", version="0.1.0", lifespan=lifespan)
    app.state.svc = svc

    @app.post("/v1/ingest")
    async def ingest(request: Request):
        raw = await request.body()
        try:
            msg = decode(raw)
        except (ProtocolError, ValueError, KeyError) as exc:
            raise HTTPException(400, str(exc))
        if not isinstance(msg, (Telemetry, Event)):
            raise HTTPException(400, "esperado telemetria ou evento")
        plan, cmds = svc.handle(msg)
        if "octet-stream" in request.headers.get("accept", ""):
            return Response(plan.to_bytes() if plan else b"", media_type="application/octet-stream")
        return {"plan": plan.to_dict() if plan else None, "preempt": [c.to_dict() for c in cmds]}

    @app.get("/v1/plans/{node}")
    def plan(node: int, request: Request):
        p = svc.coord.plans.get(node)
        if not p:
            raise HTTPException(404, "nó sem plano")
        if "octet-stream" in request.headers.get("accept", ""):
            return Response(p.to_bytes(), media_type="application/octet-stream")
        return p.to_dict()

    @app.get("/v1/state")
    def state():
        return svc.coord.snapshot(svc.clock())

    @app.post("/v1/replan")
    def replan():
        svc.coord.replan(svc.clock())
        return svc.coord.snapshot(svc.clock())

    return app


def __getattr__(name):  # `uvicorn iitls_cloud.api:app` cria o app só quando pedido
    if name == "app":
        return create_app()
    raise AttributeError(name)
