"""WorldSnapshot → 에이전트별 특징(feature) 행.

학습(dataset/)과 실행(agents.py)이 같은 함수를 쓴다. 스냅샷에 없는 값(마지막 청소 시각 등)은
History가 스냅샷 흐름과 기기 완료 결과로부터 직접 계산한다.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from orchestrator.contract import Device, WorldSnapshot

NAN = float("nan")

# 기기 모드별 상대 소음 크기 (0~1, 데모용 설정값. dBA 측정값이 아님)
NOISE_LEVEL = {
    ("vacuum", "normal"): 0.9, ("vacuum", "quiet"): 0.6,
    ("washer", "normal"): 0.7, ("washer", "low_noise"): 0.35,
    ("purifier", "high"): 0.6, ("purifier", "medium"): 0.4, ("purifier", "low"): 0.2,
}

CLEAN_FEATURES = ["hours_since_clean", "occupied_hours_since_clean", "hours_to_deadline", "is_due",
                  "is_bedroom", "is_living", "is_kitchen", "occupied_now", "hour_sin", "hour_cos"]
NOISE_FEATURES = ["level", "influence", "influence_missing", "same_room", "resting", "confirmed",
                  "quiet_pref", "quiet_pref_missing", "hour_sin", "hour_cos"]
TEMP_FEATURES = ["delta", "room_delta", "room_temp_missing", "resting", "pref_width"]
LIGHT_FEATURES = ["hour_sin", "hour_cos", "is_bedroom", "log_rest_min", "bright", "lying", "confirmed"]


def hour_feats(t: datetime) -> tuple[float, float]:
    h = t.hour + t.minute / 60
    return math.sin(2 * math.pi * h / 24), math.cos(2 * math.pi * h / 24)


@dataclass
class History:
    """스냅샷에 없는, 중앙이 직접 기억하는 값."""

    last_cleaned_at: dict[str, datetime] = field(default_factory=dict)
    occupied_s_since_clean: dict[str, float] = field(default_factory=dict)
    done_rooms: dict[str, set[str]] = field(default_factory=dict)  # occurrence_id → 청소 완료한 방
    # occurrence_id → 청소기를 도크로 돌려보낼 때 남아 있던 방. 완료 결과가 올 때까지 보존한다
    backlog: dict[str, list[str]] = field(default_factory=dict)
    last_as_of: datetime | None = None

    def observe(self, snap: WorldSnapshot) -> None:
        if self.last_as_of is not None:
            dt = max(0.0, min((snap.as_of - self.last_as_of).total_seconds(), 3600.0))
            for r in snap.rooms:
                if r.occupancy.state == "OCCUPIED":
                    self.occupied_s_since_clean[r.room_id] = self.occupied_s_since_clean.get(r.room_id, 0.0) + dt
        self.last_as_of = snap.as_of

    def mark_cleaned(self, room_id: str, t: datetime, occurrence_id: str | None = None) -> None:
        self.last_cleaned_at[room_id] = t
        self.occupied_s_since_clean[room_id] = 0.0
        if occurrence_id:
            self.done_rooms.setdefault(occurrence_id, set()).add(room_id)


def people_in(snap: WorldSnapshot, room_id: str):
    return [p for p in snap.people if p.room_id == room_id]


def room_clear(snap: WorldSnapshot, room_id: str) -> bool:
    """확실히 빈 방인가: 재실 EMPTY이고 근거가 지금(current) 것."""
    r = snap.room(room_id)
    return r is not None and r.occupancy.state == "EMPTY" and r.occupancy.evidence.freshness == "current"


def cleaning_rows(snap: WorldSnapshot, hist: History) -> list[tuple[str, dict]]:
    """청소 루틴 대상 방마다 한 행."""
    s, c = hour_feats(snap.as_of)
    rows = []
    for routine in snap.routines:
        for room_id in routine.room_ids:
            room = snap.room(room_id)
            last = hist.last_cleaned_at.get(room_id)
            rows.append((room_id, {
                "hours_since_clean": NAN if last is None else (snap.as_of - last).total_seconds() / 3600,
                "occupied_hours_since_clean": hist.occupied_s_since_clean.get(room_id, 0.0) / 3600,
                "hours_to_deadline": NAN if routine.deadline_at is None else (routine.deadline_at - snap.as_of).total_seconds() / 3600,
                "is_due": float(routine.due),
                "is_bedroom": float(room.room_type == "bedroom"),
                "is_living": float(room.room_type == "living"),
                "is_kitchen": float(room.room_type == "kitchen"),
                "occupied_now": float(room.occupancy.state == "OCCUPIED"),
                "hour_sin": s, "hour_cos": c,
            }))
    return rows


def noise_sources(snap: WorldSnapshot) -> list[tuple[Device, str, str]]:
    """(기기, 모드, 소음이 나는 방) 후보. 세탁기는 돌고 있을 때, 청소기는 루틴 대상 방마다."""
    out = []
    for d in snap.devices:
        if d.kind == "washer" and d.state.status == "running" and d.room_id:
            out += [(d, m, d.room_id) for m in d.capabilities.modes if (d.kind, m) in NOISE_LEVEL]
        if d.kind == "purifier" and d.state.power == "on" and d.room_id:
            out += [(d, m, d.room_id) for m in d.capabilities.modes if (d.kind, m) in NOISE_LEVEL]
        if d.kind == "vacuum":
            rooms = [r for rt in snap.routines if rt.device_id == d.device_id for r in rt.room_ids]
            out += [(d, m, r) for r in dict.fromkeys(rooms) for m in d.capabilities.modes if (d.kind, m) in NOISE_LEVEL]
    return out


def noise_row(snap: WorldSnapshot, dev: Device, mode: str, src_room: str, dst_room: str) -> dict:
    s, c = hour_feats(snap.as_of)
    if src_room == dst_room:
        infl = 1.0
    else:
        infl = snap.room_influence.get(src_room, {}).get(dst_room)
    people = people_in(snap, dst_room)
    resting = [p for p in people if p.activity.label in ("resting", "meeting")]
    quiet = None
    for p in resting:
        prof = snap.profile(p.profile_id)
        flag = None if prof is None else (prof.quiet_rest if p.activity.label == "resting" else prof.quiet_meeting)
        if flag is not None:
            quiet = max(quiet or 0.0, float(flag))
    return {
        "level": NOISE_LEVEL[(dev.kind, mode)],
        "influence": NAN if infl is None else infl,
        "influence_missing": float(infl is None),
        "same_room": float(src_room == dst_room),
        "resting": float(bool(resting)),
        "confirmed": float(any(p.activity.confirmed for p in resting)),
        "quiet_pref": NAN if quiet is None else quiet,
        "quiet_pref_missing": float(quiet is None),
        "hour_sin": s, "hour_cos": c,
    }


def occupied_rooms(snap: WorldSnapshot) -> list[str]:
    return [r.room_id for r in snap.rooms if r.occupancy.state == "OCCUPIED"]


def temperature_row(snap: WorldSnapshot, person, profile, room_id: str, target_c: float) -> dict:
    center = profile.center_c
    room_t = snap.room(room_id).temperature_c.value
    return {
        "delta": target_c - center,
        "room_delta": NAN if room_t is None else room_t - center,
        "room_temp_missing": float(room_t is None),
        "resting": float(person.activity.label == "resting"),
        "pref_width": profile.preferred_max_c - profile.preferred_min_c,
    }


def lighting_row(snap: WorldSnapshot, person) -> dict:
    s, c = hour_feats(snap.as_of)
    room = snap.room(person.room_id)
    stable = person.activity.stable_for_s
    return {
        "hour_sin": s, "hour_cos": c,
        "is_bedroom": float(room.room_type == "bedroom"),
        "log_rest_min": NAN if stable is None else math.log1p(stable / 60),
        "bright": float(room.light.level == "BRIGHT"),
        "lying": float(person.posture == "LYING"),
        "confirmed": float(person.activity.confirmed),
    }


def to_matrix(rows: list[dict], names: list[str]):
    import numpy as np

    return np.array([[r[n] for n in names] for r in rows], dtype=float)
