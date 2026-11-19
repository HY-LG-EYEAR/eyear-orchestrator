"""WorldSnapshot 2.0을 짧게 만드는 도구 + 시연 시나리오 t0~t7.

- 학습 데이터 시뮬레이터(dataset/simulate.py), 테스트, 시나리오 재생이 같은 도구를 쓴다.
- DemoHome은 World Model과 기기 어댑터를 흉내 낸다: 명령을 받으면 기기 상태를 바꾸고 결과를 돌려준다.
"""

from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from typing import Callable

from orchestrator.contract import WorldSnapshot
from orchestrator.schemas import DeviceCommand, DeviceResult

KST = timezone(timedelta(hours=9))
NO_EVIDENCE = {"observed_at": None, "source": "unknown", "node_ids": [], "freshness": "unavailable"}


def iso(t: datetime) -> str:
    return t.isoformat()


def ev(t: datetime | None, nodes=(), source="node", fresh="current") -> dict:
    if t is None:
        return dict(NO_EVIDENCE)
    return {"observed_at": iso(t), "source": source, "node_ids": list(nodes), "freshness": fresh}


def measurement(value=None, t=None, source="device_sensor") -> dict:
    return {"value": value, "evidence": ev(t, (), source) if value is not None else dict(NO_EVIDENCE)}


def room(room_id, room_type, t=None, node=None, occ="UNKNOWN", occ_conf=None, stable=None,
         audio="UNKNOWN", light="UNKNOWN", temp=None) -> dict:
    seen = t is not None and node is not None and occ != "UNKNOWN"
    e = ev(t, [node]) if seen else dict(NO_EVIDENCE)
    return {
        "room_id": room_id, "room_type": room_type,
        "occupancy": {"state": occ, "confidence": occ_conf, "stable_for_s": stable, "evidence": e},
        "audio": {"context": audio, "confidence": 0.8 if seen and audio != "UNKNOWN" else None, "evidence": e},
        "light": {"level": light, "confidence": None, "evidence": e},
        "temperature_c": measurement(temp, t),
        "humidity_pct": measurement(), "pm25_ug_m3": measurement(), "pm25_trend_per_min": measurement(),
    }


def person(pid, profile, room_id, identity, t, node=None, track=1, source="FACE", posture="STANDING",
           motion="SLIGHT", label="other", confirmed=True, stable=30, position=None,
           act_conf=0.86, id_conf=0.93, fresh="current") -> dict:
    src = "node" if fresh == "current" else "world_inference"
    nodes = [node] if node else []
    e = ev(t, nodes, src, fresh)
    return {
        "person_id": pid, "profile_id": profile, "room_id": room_id,
        "identity": identity, "identity_source": source if identity != "UNKNOWN" else "UNKNOWN",
        "identity_confidence": id_conf, "posture": posture,
        "posture_confidence": None if posture == "UNKNOWN" else 0.9, "motion": motion,
        "track_refs": [{"node_id": node, "track_id": track}] if node and fresh == "current" else [],
        "activity": {"label": label, "confidence": act_conf, "confirmed": confirmed,
                     "stable_for_s": stable, "evidence": e},
        "position": None if position is None else
        {"frame": "room_local", "x": position[0], "y": position[1], "evidence": e},
        "evidence": e,
    }


def _caps(actions, modes=(), temp=None, airflow=None) -> dict:
    return {"actions": list(actions), "modes": list(modes), "temperature": temp, "airflow_target": airflow}


CAPS = {
    "vacuum": _caps(["start_cleaning", "pause_cleaning", "resume_cleaning", "return_to_dock", "set_mode"], ["normal"]),
    "washer": _caps(["set_mode"], ["normal", "low_noise"]),
    "ac": _caps(["set_power", "set_mode", "set_target_temperature", "set_airflow_target"], ["cool"],
                {"min_c": 18, "max_c": 30, "step_c": 1}, "person"),
    "light": _caps(["set_power"]),
    "curtain": _caps(["set_curtain_position"]),
    "purifier": _caps(["set_power", "set_mode"], ["low", "medium", "high"]),
}


def device(device_id, room_id, kind, state, t, revision=1, online=True, manual=False, caps=None) -> dict:
    return {
        "device_id": device_id, "room_id": room_id, "kind": kind, "online": online,
        "manual_override": manual, "state_revision": revision,
        "evidence": ev(t, (), "synthetic"), "state": state,
        "capabilities": copy.deepcopy(caps or CAPS[kind]),
    }


def snapshot(t, revision, session, rooms, people, devices, profiles, routines, influence,
             nodes, clock="replay", links=(), restrictions=()) -> WorldSnapshot:
    return WorldSnapshot.model_validate({
        "schema_version": "2.0", "home_id": "home_demo_01", "world_session_id": session,
        "world_revision": revision, "as_of": iso(t), "clock_mode": clock,
        "nodes": nodes, "rooms": rooms, "people": people, "devices": devices,
        "profiles": profiles, "routines": routines, "room_influence": influence,
        "room_links": list(links), "user_restrictions": list(restrictions),
    })


def node(node_id, room_id, t, seq=1) -> dict:
    return {"node_id": node_id, "room_id": room_id, "schema_version": "0.1", "availability": "online",
            "last_event_id": f"00000000-0000-4000-8000-{seq:012d}", "last_seq": seq,
            "last_timestamp": iso(t), "received_at": iso(t), "supports_speak": True}


# ---------- 시연 시나리오: 방1에서 쉬고 거실로 나온 A의 저녁 ----------

SESSION = "592066b1-2467-47d3-a48a-f27b2e567fe7"
DAY = datetime(2026, 11, 13, tzinfo=KST)
PROFILES = [
    {"profile_id": "profile_a", "identity": "MEMBER_A", "preferred_min_c": 23, "preferred_max_c": 25,
     "quiet_rest": True, "quiet_meeting": True},
    {"profile_id": "profile_b", "identity": "MEMBER_B", "preferred_min_c": 25, "preferred_max_c": 27,
     "quiet_rest": True, "quiet_meeting": True},
]
INFLUENCE = {"laundry": {"room1": 0.8, "living": None}, "room1": {"living": 0.2}}
LINKS = [{"room_a": "room1", "room_b": "living", "passable": True},
         {"room_a": "living", "room_b": "kitchen", "passable": True},
         {"room_a": "living", "room_b": "room2", "passable": True}]


def at(hms: str) -> datetime:
    h, m, s = map(int, hms.split(":"))
    return DAY.replace(hour=h, minute=m, second=s)


class DemoHome:
    """World Model + 기기 어댑터 흉내. 명령을 받으면 기기 상태를 바꾸고 결과를 돌려준다.

    청소기 (발표 시나리오, 시뮬레이터와 맞출 동작):
    - t0: 저녁 청소 회차 진행 중, 방2 청소 중, 남은 방 [방2, 방1]
    - t1 직전: 방2 완료 → 남은 방 [방1], 아직 방1에 들어가지 않음 (t1 판단은 이 상태로)
    - return_to_dock: 즉시 completed, 상태 returning (남은 방 목록 비움) → 다음 스냅샷에서 docked
    - start_cleaning: 즉시 accepted, 상태 cleaning → 다음 스냅샷에서 청소를 마치고 completed, docked
    그 밖의 기기는 명령을 받으면 바로 상태를 바꾸고 completed를 돌려준다.
    """

    def __init__(self, on_result: Callable[[DeviceResult], None] | None = None):
        self.on_result = on_result
        self.revision = 100
        self.devices = {
            "vacuum_01": ("kitchen", "vacuum", {"status": "cleaning", "mode": "normal",
                                                "target_room_ids": ["room2", "room1"], "dock_room_id": "kitchen"}),
            "washer_01": ("laundry", "washer", {"status": "running", "mode": "normal"}),
            "ac_room1": ("room1", "ac", {"power": "off", "mode": "cool", "target_c": 24, "airflow_target_person_id": None}),
            "ac_living": ("living", "ac", {"power": "off", "mode": "cool", "target_c": 27, "airflow_target_person_id": None}),
            "light_room1": ("room1", "light", {"power": "on"}),
            "curtain_room1": ("room1", "curtain", {"position_pct": 100, "status": "stationary"}),
        }
        self.dev_rev = {k: 1 for k in self.devices}
        self.log: list[DeviceCommand] = []
        self.cleaning_cmd: DeviceCommand | None = None   # 접수했고 아직 끝나지 않은 청소 명령

    # 기기 어댑터 역할
    def send(self, cmd: DeviceCommand) -> None:
        self.log.append(cmd)
        _, kind, st = self.devices[cmd.device_id]
        p = cmd.params
        status = "completed"
        if cmd.action == "start_cleaning":
            st.update(status="cleaning", target_room_ids=list(p["room_ids"]), mode=p.get("mode", st["mode"]))
            self.cleaning_cmd, status = cmd, "accepted"    # 완료는 청소를 마친 뒤 (다음 스냅샷)
        elif cmd.action == "return_to_dock":
            st.update(status="returning", target_room_ids=[])
        elif cmd.action == "pause_cleaning":
            st["status"] = "paused"
        elif cmd.action == "resume_cleaning":
            st["status"] = "cleaning"
        elif cmd.action == "set_mode":
            st["mode"] = p["mode"]
        elif cmd.action == "set_power":
            st["power"] = p["power"]
        elif cmd.action == "set_target_temperature":
            st["target_c"] = p["target_c"]
        elif cmd.action == "set_airflow_target":
            st["airflow_target_person_id"] = p["person_id"]
        elif cmd.action == "set_curtain_position":
            st["position_pct"] = p["position_pct"]
        self.dev_rev[cmd.device_id] += 1
        self.revision += 1
        self._report(cmd, status, st)

    def _report(self, cmd: DeviceCommand, status: str, st: dict) -> None:
        if self.on_result:
            self.on_result(DeviceResult(
                world_session_id=cmd.world_session_id, world_revision=self.revision, command_id=cmd.command_id,
                device_id=cmd.device_id, status=status, timestamp=cmd.issued_at, observed_state=dict(st)))

    def _advance_vacuum(self, step: str) -> None:
        """스냅샷 사이에 청소기가 한 일 (시뮬레이터 흉내)."""
        st = self.devices["vacuum_01"][2]
        before = dict(st)
        if st["status"] == "returning":
            st.update(status="docked")
        elif self.cleaning_cmd is not None and st["status"] == "cleaning":
            st.update(status="docked", target_room_ids=[])      # 청소를 마치고 도크로 돌아옴
            cmd, self.cleaning_cmd = self.cleaning_cmd, None
            self.dev_rev["vacuum_01"] += 1
            self._report(cmd, "completed", st)
        elif step != "t0" and st["status"] == "cleaning" and st["target_room_ids"][:1] == ["room2"]:
            st["target_room_ids"] = st["target_room_ids"][1:]   # 방2 완료, 방1로 가기 전
        if st != before:
            self.dev_rev["vacuum_01"] += 1

    def _devices(self, t):
        return [device(i, r, k, dict(s), t, self.dev_rev[i]) for i, (r, k, s) in self.devices.items()]

    def snapshot(self, step: str) -> WorldSnapshot:
        t, rooms, people, due = STEPS[step]
        self._advance_vacuum(step)
        self.revision += 1
        nodes = [node("node_room1", "room1", t, self.revision), node("mock_living", "living", t, self.revision),
                 node("mock_room2", "room2", t, self.revision)]
        routine = {"routine_id": "daily_clean", "occurrence_id": "daily_clean_20261113_evening",
                   "device_id": "vacuum_01", "room_ids": ["room2", "room1"], "due": due,
                   "requested_at": iso(at("20:50:00")) if due else None,
                   "deadline_at": iso(at("23:02:00")) if due else None}
        return snapshot(t, self.revision, SESSION, rooms(t), people(t), self._devices(t), PROFILES,
                        [routine], INFLUENCE, nodes, links=LINKS)


def _rooms(r1_occ, r1_light="BRIGHT", r1_audio="SILENT", living_audio="TV"):
    def build(t):
        return [
            room("room1", "bedroom", t, "node_room1", r1_occ, 0.95, 30, r1_audio, r1_light),
            room("living", "living", t, "mock_living", "OCCUPIED", 0.95, 30, living_audio, "BRIGHT"),
            room("room2", "bedroom", t, "mock_room2", "EMPTY", 0.95, 300, "SILENT", "BRIGHT"),
            room("kitchen", "kitchen"),
            room("laundry", "laundry"),
        ]
    return build


def _b(t):
    return person("person_b", "profile_b", "living", "MEMBER_B", t, "mock_living", 1, posture="SITTING",
                  position=(0.6, 4.2))


def _people(*makers):
    return lambda t: [m(t) for m in makers]


A_ROOM1 = (2.4, 2.67)
STEPS = {
    # 이름: (시각, 방 상태, 사람, 청소 루틴 due)
    "t0": (at("21:00:00"), _rooms("EMPTY", r1_audio="LOUD_NOISE"), _people(_b), True),
    "t1": (at("21:02:10"), _rooms("OCCUPIED", r1_audio="LOUD_NOISE"), _people(
        _b, lambda t: person("person_a", "profile_a", "room1", "MEMBER_A", t, "node_room1", 1,
                             motion="ACTIVE", confirmed=False, stable=0, position=(1.4, 2.15))), True),
    "t2": (at("21:04:30"), _rooms("OCCUPIED"), _people(
        _b, lambda t: person("person_a", "profile_a", "room1", "MEMBER_A", t, "node_room1", 1,
                             posture="LYING", label="resting", stable=30, position=A_ROOM1)), True),
    # 불이 꺼진 뒤: 레이더가 이어받음 (TRACK_CARRIED, 자세 UNKNOWN)
    "t3": (at("21:04:38"), _rooms("OCCUPIED", r1_light="DARK"), _people(
        _b, lambda t: person("person_a", "profile_a", "room1", "MEMBER_A", t, "node_room1", 1,
                             source="TRACK_CARRIED", posture="UNKNOWN", label="resting", stable=38,
                             id_conf=0.85, position=A_ROOM1)), True),
    # 수면 중: 위치 레이더가 놓침 → World Model 추정 유지 (stale, confirmed=false)
    "t4": (at("21:40:00"), _rooms("OCCUPIED", r1_light="DARK"), _people(
        _b, lambda t: person("person_a", "profile_a", "room1", "MEMBER_A", t - timedelta(minutes=30), None,
                             source="TRACK_CARRIED", posture="UNKNOWN", motion="UNKNOWN", label="resting",
                             confirmed=False, stable=None, id_conf=0.85, fresh="stale")), True),
    "t5": (at("22:50:05"), _rooms("OCCUPIED", r1_audio="SPEECH"), _people(
        _b, lambda t: person("person_a", "profile_a", "room1", "MEMBER_A", t, "node_room1", 1,
                             posture="SITTING", confirmed=False, stable=0, position=(2.4, 2.85))), True),
    "t6": (at("22:55:20"), _rooms("EMPTY", r1_audio="LOUD_NOISE"), _people(
        _b, lambda t: person("person_a", "profile_a", None, "MEMBER_A", t - timedelta(seconds=7), None,
                             posture="UNKNOWN", motion="UNKNOWN", confirmed=False, stable=None,
                             fresh="stale")), True),
    "t7": (at("22:56:10"), _rooms("EMPTY", r1_audio="LOUD_NOISE"), _people(
        _b, lambda t: person("person_a", "profile_a", "living", "MEMBER_A", t, "mock_living", 2,
                             posture="SITTING", confirmed=False, stable=0, position=(0.6, 6.7))), True),
}
