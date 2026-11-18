"""WorldSnapshot 2.0 — World Model → Orchestrator 입력 계약 (v2 초안 / v2.1 문서).

모든 키는 필수이고(nullable도 null로 보냄), 정의하지 않은 키는 거부한다.
계약이 바뀌면 이 파일과 features.py만 고친다.
"""

from __future__ import annotations

import math
from typing import Annotated, Literal, Union

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, field_validator, model_validator

Conf = Annotated[float, Field(ge=0.0, le=1.0)]
Freshness = Literal["current", "stale", "unavailable"]
Source = Literal["node", "device_sensor", "world_inference", "app", "synthetic", "unknown"]
ActionName = Literal[
    "start_cleaning", "pause_cleaning", "resume_cleaning", "return_to_dock",
    "set_mode", "set_power", "set_target_temperature", "set_airflow_target", "set_curtain_position",
]
ACTIONS: tuple[str, ...] = ActionName.__args__  # type: ignore[attr-defined]


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _finite(v):
    if v is not None and not math.isfinite(v):
        raise ValueError("유한한 숫자만 허용")
    return v


class Evidence(_M):
    observed_at: AwareDatetime | None
    source: Source
    node_ids: list[str]
    freshness: Freshness


class NodeInfo(_M):
    node_id: str
    room_id: str
    schema_version: str
    availability: Literal["online", "stale", "offline", "unknown"]
    last_event_id: str | None
    last_seq: int | None = Field(ge=0)
    last_timestamp: AwareDatetime | None
    received_at: AwareDatetime | None
    supports_speak: bool


class Occupancy(_M):
    state: Literal["OCCUPIED", "EMPTY", "UNKNOWN"]
    confidence: Conf | None
    stable_for_s: int | None = Field(ge=0)
    evidence: Evidence


class Audio(_M):
    context: Literal["SILENT", "SPEECH", "CONVERSATION", "MUSIC", "TV", "LOUD_NOISE", "UNKNOWN"]
    confidence: Conf | None
    evidence: Evidence


class Light(_M):
    level: Literal["DARK", "DIM", "BRIGHT", "UNKNOWN"]
    confidence: Conf | None
    evidence: Evidence


class Measurement(_M):
    value: float | None
    evidence: Evidence

    @field_validator("value")
    @classmethod
    def _fin(cls, v):
        return _finite(v)


class Room(_M):
    room_id: str
    room_type: Literal["bedroom", "living", "kitchen", "laundry", "hallway", "other", "unknown"]
    occupancy: Occupancy
    audio: Audio
    light: Light
    temperature_c: Measurement
    humidity_pct: Measurement
    pm25_ug_m3: Measurement
    pm25_trend_per_min: Measurement

    @model_validator(mode="after")
    def _ranges(self):
        for name, lo, hi in (("temperature_c", -10, 60), ("humidity_pct", 0, 100), ("pm25_ug_m3", 0, 1000)):
            v = getattr(self, name).value
            if v is not None and not lo <= v <= hi:
                raise ValueError(f"{self.room_id}.{name} 범위 밖: {v}")
        return self


class RoomLink(_M):
    room_a: str
    room_b: str
    passable: bool | None


class TrackRef(_M):
    node_id: str
    track_id: int = Field(ge=0)


class Activity(_M):
    label: Literal["resting", "meeting", "working", "cooking", "idle", "other", "unknown"]
    confidence: Conf | None
    confirmed: bool
    stable_for_s: int | None = Field(ge=0)
    evidence: Evidence

    @model_validator(mode="after")
    def _unknown_not_confirmed(self):
        if self.label == "unknown" and self.confirmed:
            raise ValueError("activity unknown은 confirmed=false")
        return self


class Position(_M):
    frame: Literal["room_local"]
    x: float
    y: float
    evidence: Evidence


class Person(_M):
    person_id: str
    profile_id: str | None
    room_id: str | None
    identity: str
    identity_source: Literal["FACE", "VOICE", "TRACK_CARRIED", "UNKNOWN"]
    identity_confidence: Conf | None
    posture: Literal["STANDING", "SITTING", "LYING", "WALKING", "UNKNOWN"]
    posture_confidence: Conf | None
    motion: Literal["STILL", "SLIGHT", "ACTIVE", "UNKNOWN"]
    track_refs: list[TrackRef]
    activity: Activity
    position: Position | None
    evidence: Evidence

    @model_validator(mode="after")
    def _unknown_identity(self):
        if (self.identity == "UNKNOWN") != (self.identity_source == "UNKNOWN"):
            raise ValueError("identity가 UNKNOWN이면 identity_source도 UNKNOWN (그 반대도)")
        return self


# ---------- 기기 상태 (종류별) ----------

class VacuumState(_M):
    status: Literal["idle", "cleaning", "paused", "returning", "docked", "completed", "error", "unknown"]
    mode: str | None
    target_room_ids: list[str]
    dock_room_id: str | None


class WasherState(_M):
    status: Literal["off", "idle", "running", "paused", "completed", "error", "unknown"]
    mode: str | None


class AcState(_M):
    power: Literal["on", "off", "unknown"]
    mode: str | None
    target_c: float | None
    airflow_target_person_id: str | None


class LightState(_M):
    power: Literal["on", "off", "unknown"]


class CurtainState(_M):
    position_pct: float | None = Field(ge=0, le=100)
    status: Literal["stationary", "opening", "closing", "error", "unknown"]


class PurifierState(_M):
    power: Literal["on", "off", "unknown"]
    mode: str | None


STATE_BY_KIND = {
    "vacuum": VacuumState, "washer": WasherState, "ac": AcState,
    "light": LightState, "curtain": CurtainState, "purifier": PurifierState,
}


class TempRange(_M):
    min_c: float
    max_c: float
    step_c: float = Field(gt=0)

    @model_validator(mode="after")
    def _order(self):
        if self.min_c > self.max_c:
            raise ValueError("min_c > max_c")
        return self


class Capabilities(_M):
    actions: list[ActionName]
    modes: list[str]
    temperature: TempRange | None
    airflow_target: Literal["person"] | None


class Device(_M):
    device_id: str
    room_id: str | None
    kind: Literal["vacuum", "washer", "ac", "light", "curtain", "purifier"]
    online: bool | None
    manual_override: bool | None
    state_revision: int = Field(ge=0)
    evidence: Evidence
    state: Union[VacuumState, WasherState, AcState, LightState, CurtainState, PurifierState]
    capabilities: Capabilities

    @model_validator(mode="before")
    @classmethod
    def _state_by_kind(cls, data):
        if isinstance(data, dict) and isinstance(data.get("state"), dict) and data.get("kind") in STATE_BY_KIND:
            data = {**data, "state": STATE_BY_KIND[data["kind"]].model_validate(data["state"])}
        return data

    @model_validator(mode="after")
    def _kind_matches(self):
        if not isinstance(self.state, STATE_BY_KIND[self.kind]):
            raise ValueError(f"{self.device_id}: state가 kind={self.kind}와 맞지 않음")
        return self


class Profile(_M):
    profile_id: str
    identity: str
    preferred_min_c: float | None = Field(ge=-10, le=60)
    preferred_max_c: float | None = Field(ge=-10, le=60)
    quiet_rest: bool | None
    quiet_meeting: bool | None

    @model_validator(mode="after")
    def _pair(self):
        lo, hi = self.preferred_min_c, self.preferred_max_c
        if (lo is None) != (hi is None):
            raise ValueError("preferred_min_c/max_c는 둘 다 숫자 또는 둘 다 null")
        if lo is not None and lo > hi:
            raise ValueError("preferred_min_c > preferred_max_c")
        return self

    @property
    def center_c(self) -> float | None:
        return None if self.preferred_min_c is None else (self.preferred_min_c + self.preferred_max_c) / 2


class Routine(_M):
    routine_id: str
    occurrence_id: str
    device_id: str
    room_ids: list[str] = Field(min_length=1)
    due: bool
    requested_at: AwareDatetime | None
    deadline_at: AwareDatetime | None

    @model_validator(mode="after")
    def _check(self):
        if len(set(self.room_ids)) != len(self.room_ids):
            raise ValueError("routine.room_ids 중복")
        if self.requested_at and self.deadline_at and self.deadline_at < self.requested_at:
            raise ValueError("deadline_at < requested_at")
        return self


class UserRestriction(_M):
    restriction_id: str
    target_type: Literal["room", "device"]
    target_id: str
    action: ActionName | None
    expires_at: AwareDatetime | None
    reason: str


class WorldSnapshot(_M):
    schema_version: Literal["2.0"]
    home_id: str = Field(min_length=1)
    world_session_id: str
    world_revision: int = Field(ge=0)
    as_of: AwareDatetime
    clock_mode: Literal["live", "replay"]
    nodes: list[NodeInfo]
    rooms: list[Room] = Field(min_length=1)
    people: list[Person]
    devices: list[Device]
    profiles: list[Profile]
    routines: list[Routine]
    room_influence: dict[str, dict[str, Conf | None]]
    room_links: list[RoomLink]
    user_restrictions: list[UserRestriction]

    @model_validator(mode="after")
    def _references(self):
        def ids(items, attr):
            vals = [getattr(i, attr) for i in items]
            if len(vals) != len(set(vals)):
                raise ValueError(f"{attr} 중복")
            return set(vals)

        rooms = ids(self.rooms, "room_id")
        nodes = ids(self.nodes, "node_id")
        people = ids(self.people, "person_id")
        devices = ids(self.devices, "device_id")
        profiles = ids(self.profiles, "profile_id")
        ids(self.routines, "occurrence_id")
        ids(self.user_restrictions, "restriction_id")

        def need(value, pool, what):
            if value is not None and value not in pool:
                raise ValueError(f"없는 {what} 참조: {value}")

        def ev(e: Evidence):
            for n in e.node_ids:
                need(n, nodes, "node")

        for n in self.nodes:
            need(n.room_id, rooms, "room")
        for r in self.rooms:
            for e in (r.occupancy.evidence, r.audio.evidence, r.light.evidence, r.temperature_c.evidence,
                      r.humidity_pct.evidence, r.pm25_ug_m3.evidence, r.pm25_trend_per_min.evidence):
                ev(e)
        tracks = set()
        for p in self.people:
            need(p.room_id, rooms, "room")
            need(p.profile_id, profiles, "profile")
            ev(p.evidence)
            ev(p.activity.evidence)
            for t in p.track_refs:
                need(t.node_id, nodes, "node")
                key = (t.node_id, t.track_id)
                if key in tracks:
                    raise ValueError(f"한 트랙이 두 사람에 연결됨: {key}")
                tracks.add(key)
        for d in self.devices:
            need(d.room_id, rooms, "room")
            s = d.state
            if isinstance(s, VacuumState):
                need(s.dock_room_id, rooms, "room")
                for r in s.target_room_ids:
                    need(r, rooms, "room")
            if isinstance(s, AcState):
                need(s.airflow_target_person_id, people, "person")
        for r in self.routines:
            need(r.device_id, devices, "device")
            for x in r.room_ids:
                need(x, rooms, "room")
        for src, dsts in self.room_influence.items():
            need(src, rooms, "room")
            for dst in dsts:
                need(dst, rooms, "room")
        for link in self.room_links:
            need(link.room_a, rooms, "room")
            need(link.room_b, rooms, "room")
        for u in self.user_restrictions:
            need(u.target_id, rooms if u.target_type == "room" else devices, u.target_type)
        return self

    # 편의 조회
    def room(self, room_id: str) -> Room | None:
        return next((r for r in self.rooms if r.room_id == room_id), None)

    def device(self, device_id: str) -> Device | None:
        return next((d for d in self.devices if d.device_id == device_id), None)

    def person(self, person_id: str) -> Person | None:
        return next((p for p in self.people if p.person_id == person_id), None)

    def profile(self, profile_id: str | None) -> Profile | None:
        return next((p for p in self.profiles if p.profile_id == profile_id), None)

    def node_for_room(self, room_id: str) -> NodeInfo | None:
        return next((n for n in self.nodes if n.room_id == room_id), None)
