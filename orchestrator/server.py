"""Orchestrator HTTP 서버 (+ 선택: MQTT로 Node 음성·기기 명령).

    python -m orchestrator.server            # http://127.0.0.1:8765, 문서: /docs

받는 것
- POST /v2/world/sessions       World Model 세션 등록 (계약 v2.1 6-1)
- POST /v2/world/snapshots      WorldSnapshot 2.0 (202 / 409 / 422)
- POST /v2/world/device-results 기기 실행 결과 (계약 v2.1 6-3)
- POST /v1/voice/reply, /v1/voice/command  Node 음성 (MQTT 대신 HTTP로 보낼 때)
- POST /v1/control              앱의 운영자 제어: pause | resume | reset
보여 주는 것
- GET /v1/decisions, /v1/status, /v1/schemas, /health
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ValidationError

from orchestrator.agents import Models
from orchestrator.central import Orchestrator
from orchestrator.config import ROOT, Settings, build_llm
from orchestrator.contract import WorldSnapshot
from orchestrator.schemas import (
    AgentProposal, Decision, DeviceCommand, DeviceResult, LlmSelection, SpeakCommand, VoiceCommand, VoiceReply,
)

log = logging.getLogger("orchestrator")
LIVE_AGE_S = (-5, 120)


class SessionIn(BaseModel):
    home_id: str
    world_session_id: str
    clock_mode: str
    started_at: datetime | None = None


class ControlIn(BaseModel):
    action: str


class MqttBridge:
    """Node 음성 토픽(eyear/v0/{room}/speak, voice/reply, voice/command)과 기기 명령 토픽."""

    def __init__(self, host: str, port: int):
        import paho.mqtt.client as mqtt

        self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="eyear-orchestrator")
        self.host, self.port, self.orc = host, port, None
        self.client.on_connect = lambda c, u, f, rc, p=None: (
            c.subscribe("eyear/v0/+/voice/reply"), c.subscribe("eyear/v0/+/voice/command"))
        self.client.on_message = self._on_message

    def start(self, orc: Orchestrator) -> None:
        self.orc = orc
        self.client.connect(self.host, self.port)
        self.client.loop_start()

    def stop(self) -> None:
        self.client.loop_stop()
        self.client.disconnect()

    def _on_message(self, client, userdata, msg) -> None:
        try:
            if msg.topic.endswith("/voice/reply"):
                self.orc.on_voice_reply(VoiceReply.model_validate_json(msg.payload))
            elif msg.topic.endswith("/voice/command"):
                self.orc.on_voice_command(VoiceCommand.model_validate_json(msg.payload))
        except Exception as e:
            log.warning("MQTT 메시지 무시 (%s): %s", msg.topic, e)

    def speak(self, room_id: str, cmd: SpeakCommand) -> None:
        self.client.publish(f"eyear/v0/{room_id}/speak", cmd.model_dump_json(), qos=1)

    def send_command(self, cmd: DeviceCommand) -> None:
        self.client.publish(f"eyear/v0/device/{cmd.device_id}/command", cmd.model_dump_json(), qos=1)


def create_app(settings: Settings | None = None, orchestrator: Orchestrator | None = None,
               bridge: MqttBridge | None = None) -> FastAPI:
    settings = settings or Settings.load()
    app = FastAPI(title="EYEAR Orchestrator", version="0.1.0")
    spoken: list[tuple[str, SpeakCommand]] = []

    if orchestrator is None:
        if bridge is None and settings.mqtt_host:
            bridge = MqttBridge(settings.mqtt_host, settings.mqtt_port)
        holder: dict = {}

        def send(cmd: DeviceCommand) -> None:
            if bridge and settings.device_transport == "mqtt":
                bridge.send_command(cmd)
            else:  # mock: 기기가 바로 완료했다고 가정
                holder["orc"].on_device_result(DeviceResult(
                    world_session_id=cmd.world_session_id, command_id=cmd.command_id, device_id=cmd.device_id,
                    status="completed", timestamp=datetime.now(timezone.utc), reason="mock adapter"))

        def speak(room_id: str, cmd: SpeakCommand) -> None:
            spoken.append((room_id, cmd))
            if bridge:
                bridge.speak(room_id, cmd)

        trace = Path(settings.trace_path) if settings.trace_path else None
        if trace is not None and not trace.is_absolute():
            trace = ROOT / trace
        orchestrator = Orchestrator(send, speak, llm=build_llm(settings), models=Models(), trace_path=trace)
        holder["orc"] = orchestrator
    orc = orchestrator
    state = {"sessions": {}, "revision": None, "hash": None}
    stop = threading.Event()

    @asynccontextmanager
    async def lifespan(_app):
        if bridge:
            bridge.start(orc)

        def ticker():  # 수면 질문 대답 마감 확인
            while not stop.wait(1.0):
                orc.tick()
        threading.Thread(target=ticker, daemon=True).start()
        yield
        stop.set()
        if bridge:
            bridge.stop()

    app.router.lifespan_context = lifespan

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/v1/status")
    def status():
        return orc.status()

    @app.post("/v2/world/sessions", status_code=201)
    def register_session(body: SessionIn):
        if body.clock_mode not in ("live", "replay"):
            raise HTTPException(422, "clock_mode는 live 또는 replay")
        state["sessions"][body.world_session_id] = body.clock_mode
        state["revision"], state["hash"] = None, None
        orc.reset(body.world_session_id, body.clock_mode)
        return {"registered": True, "world_session_id": body.world_session_id}

    @app.post("/v2/world/snapshots", status_code=202)
    async def post_snapshot(request: Request, background: BackgroundTasks):
        raw = await request.body()
        try:
            snap = WorldSnapshot.model_validate_json(raw)
        except ValidationError as e:
            return JSONResponse(status_code=422, content={"detail": json.loads(e.json(include_url=False))})
        mode = state["sessions"].get(snap.world_session_id)
        if mode is None or orc.session_id != snap.world_session_id:
            return JSONResponse(status_code=409, content={"detail": "등록되지 않은 세션. POST /v2/world/sessions 먼저"})
        if snap.clock_mode != mode:
            return JSONResponse(status_code=409, content={"detail": f"세션 clock_mode={mode}와 다름"})
        digest = hashlib.sha256(json.dumps(json.loads(raw), sort_keys=True).encode()).hexdigest()
        rev = state["revision"]
        if rev is not None and snap.world_revision == rev:
            if digest == state["hash"]:
                return _accepted(snap)  # 재전송: 다시 판단하지 않음
            return JSONResponse(status_code=409, content={"detail": "같은 revision에 다른 내용"})
        if rev is not None and snap.world_revision < rev:
            return JSONResponse(status_code=409, content={"detail": f"지난 revision (현재 {rev})"})
        if snap.clock_mode == "live":
            age = (datetime.now(timezone.utc) - snap.as_of).total_seconds()
            if not LIVE_AGE_S[0] <= age <= LIVE_AGE_S[1]:
                return JSONResponse(status_code=422, content={"detail": f"as_of가 너무 오래됐거나 미래임 ({age:.0f}초)"})
        state["revision"], state["hash"] = snap.world_revision, digest
        background.add_task(orc.on_snapshot, snap)
        return _accepted(snap)

    def _accepted(snap):
        return {"accepted": True, "home_id": snap.home_id, "world_session_id": snap.world_session_id,
                "world_revision": snap.world_revision}

    @app.post("/v2/world/device-results", status_code=202)
    def device_result(body: DeviceResult):
        orc.on_device_result(body)
        return {"accepted": True}

    @app.post("/v1/voice/reply")
    def voice_reply(body: VoiceReply):
        d = orc.on_voice_reply(body)
        return {"handled": d is not None, "plan_id": d.plan_id if d else None}

    @app.post("/v1/voice/command")
    def voice_command(body: VoiceCommand):
        d = orc.on_voice_command(body)
        return {"handled": d is not None, "plan_id": d.plan_id if d else None}

    @app.post("/v1/control")
    def control(body: ControlIn):
        if body.action not in ("pause", "resume", "reset"):
            raise HTTPException(422, "action은 pause | resume | reset")
        orc.control(body.action)
        if body.action == "reset":
            state["revision"], state["hash"] = None, None
        return orc.status()

    @app.get("/v1/decisions", response_model=list[Decision])
    def decisions(limit: int = 20):
        return list(orc.decisions)[-limit:]

    @app.get("/v1/speech")
    def speech(limit: int = 20):
        return [{"room_id": r, **c.model_dump(mode="json")} for r, c in spoken[-limit:]]

    @app.get("/v1/schemas")
    def schemas():
        return {m.__name__: m.model_json_schema() for m in
                (WorldSnapshot, AgentProposal, LlmSelection, Decision, DeviceCommand, DeviceResult,
                 SpeakCommand, VoiceReply, VoiceCommand)}

    app.state.orchestrator = orc
    return app


def main() -> None:
    import os

    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(create_app(), host=os.environ.get("HOST", "127.0.0.1"), port=int(os.environ.get("PORT", "8765")))


if __name__ == "__main__":
    main()
