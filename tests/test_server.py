"""HTTP 수신 규칙: 세션 등록, 202 / 409 / 422, 재전송 멱등, 판단 기록 조회."""

import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient

from orchestrator.config import Settings
from orchestrator.server import create_app

RAW = json.loads((Path(__file__).resolve().parent.parent / "examples" / "snapshot_t2.json").read_text(encoding="utf-8"))
SESSION = RAW["world_session_id"]


def client():
    return TestClient(create_app(Settings(trace_path="")))


def register(c, mode="replay"):
    r = c.post("/v2/world/sessions", json={"home_id": "home_demo_01", "world_session_id": SESSION, "clock_mode": mode})
    assert r.status_code == 201


def test_unregistered_session_is_409():
    with client() as c:
        assert c.post("/v2/world/snapshots", json=RAW).status_code == 409


def test_accept_duplicate_conflict_and_old_revision():
    with client() as c:
        register(c)
        r = c.post("/v2/world/snapshots", json=RAW)
        assert r.status_code == 202 and r.json()["world_revision"] == RAW["world_revision"]
        assert c.post("/v2/world/snapshots", json=RAW).status_code == 202          # 재전송
        changed = {**RAW, "home_id": "other"}
        assert c.post("/v2/world/snapshots", json=changed).status_code == 409      # 같은 번호, 다른 내용
        assert c.post("/v2/world/snapshots", json={**RAW, "world_revision": 1}).status_code == 409
        assert c.post("/v2/world/snapshots", json={**RAW, "world_revision": 103}).status_code == 202


def test_invalid_snapshot_is_422():
    with client() as c:
        register(c)
        assert c.post("/v2/world/snapshots", json={**RAW, "unknown": 1}).status_code == 422


def test_live_snapshot_too_old_is_422():
    with client() as c:
        register(c, "live")
        assert c.post("/v2/world/snapshots", json={**RAW, "clock_mode": "live"}).status_code == 422
        fresh = {**RAW, "clock_mode": "live", "as_of": datetime.now(timezone.utc).isoformat()}
        assert c.post("/v2/world/snapshots", json=fresh).status_code == 202


def test_decision_is_recorded_and_question_asked():
    with client() as c:
        register(c)
        c.post("/v2/world/snapshots", json=RAW)
        decisions = c.get("/v1/decisions").json()
        assert decisions and decisions[-1]["world_revision"] == RAW["world_revision"]
        speech = c.get("/v1/speech").json()
        assert speech and speech[-1]["room_id"] == "room1" and speech[-1]["expect_reply"]
        reply = {"node_id": "node_room1", "room_id": "room1", "command_id": speech[-1]["command_id"],
                 "text": "응.", "confidence": 0.5, "no_response": False}
        assert c.post("/v1/voice/reply", json=reply).json()["handled"]
        last = c.get("/v1/decisions").json()[-1]
        assert {(x["device_id"], x["action"]) for x in last["commands"]} >= {("light_room1", "set_power"),
                                                                             ("curtain_room1", "set_curtain_position")}
        assert c.get("/v1/schemas").status_code == 200
