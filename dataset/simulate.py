"""가구 시뮬레이터 → 학습용 표 4개.

스냅샷은 실제 실행과 같은 WorldSnapshot 2.0 객체로 만들고, 특징도 실행과 같은 features.py로 뽑는다.
라벨은 시뮬레이터 안의 숨은 상태(바닥 먼지, 방해감, 체감 온도, 수면 의도)에서만 만든다.
스냅샷에 들어가는 값에는 관측 잡음(놓침, UNKNOWN, null)을 넣는다.

숨은 모델 (가정이며 실제 측정값이 아님)
- 청소:  먼지 D는 사람이 머문 시간만큼 쌓이고 청소하면 줄어듦.  라벨 = 1 − exp(−D)
- 소음:  P(방해) = σ(6·소음크기·실제영향 − 3.2 + 1.5·휴식 + 1.0·휴식·조용함선호 + 가구편차)
- 온도:  온열감 = (설정온도 − (선호중앙 + 0.4·휴식 + 개인편차)) / 1.5 + 잡음
- 빛:    P(수면 의도) = σ(−2 + 2.2·침실 + 2.0·밤 + 0.6·log(1+휴식분) + 가구편차)
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

from orchestrator import features as F
from orchestrator.fixtures import KST, device, node, person, room, snapshot

ROOMS = [("room1", "bedroom", "node_room1"), ("room2", "bedroom", "node_room2"), ("living", "living", "node_living"),
         ("kitchen", "kitchen", None), ("laundry", "laundry", None)]
NODE_ROOMS = {r for r, _, n in ROOMS if n}
STEP = timedelta(minutes=15)


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _where(hour: int, weekend: bool, rng: random.Random, bedroom: str) -> tuple[str | None, bool]:
    """(방 또는 None=외출, 휴식 여부)를 시간대 확률로 뽑는다."""
    if hour < 7:
        table = [(bedroom, True, .9), ("living", False, .1)]
    elif hour < 9:
        table = [("kitchen", False, .5), ("living", False, .3), (None, False, .2)]
    elif hour < 18:
        table = ([("living", False, .5), (bedroom, True, .2), ("kitchen", False, .2), (None, False, .1)] if weekend
                 else [(None, False, .6), ("living", False, .25), (bedroom, True, .15)])
    elif hour < 21:
        table = [("living", False, .5), ("kitchen", False, .25), (bedroom, True, .15), (None, False, .1)]
    else:
        table = [(bedroom, True, .5), ("living", False, .4), (None, False, .1)]
    x, acc = rng.random(), 0.0
    for r, rest, p in table:
        acc += p
        if x <= acc:
            return r, rest
    return table[-1][0], table[-1][1]


class Household:
    def __init__(self, hid: int, seed: int):
        rng = self.rng = random.Random(seed * 1000 + hid)
        self.hid = hid
        n = rng.choice([2, 2, 3])
        self.people = []
        for i in range(n):
            center = round(rng.gauss(25, 1.2) * 2) / 2
            self.people.append({
                "pid": f"person_{'abc'[i]}", "profile": f"profile_{'abc'[i]}", "identity": f"MEMBER_{'ABC'[i]}",
                "bedroom": "room1" if i == 0 else "room2", "center": center,
                "quiet": None if rng.random() < .1 else rng.random() < .7,
                "bias": rng.gauss(0, 0.5), "where": None, "rest": False, "since": 0,
            })
        self.dirt_rate = {r: math.exp(rng.gauss(math.log(0.06), 0.4)) for r, _, _ in ROOMS}
        self.dirt = {r: rng.random() for r, _, _ in ROOMS}
        self.infl_true = {("laundry", "room1"): rng.uniform(.3, .9), ("laundry", "living"): rng.uniform(0, .5),
                          ("laundry", "room2"): rng.uniform(0, .4), ("room1", "living"): rng.uniform(0, .4),
                          ("room2", "living"): rng.uniform(0, .4), ("living", "room1"): rng.uniform(0, .4),
                          ("living", "room2"): rng.uniform(0, .4), ("kitchen", "living"): rng.uniform(.2, .6)}
        self.infl_obs = {k: (round(v, 2) if rng.random() < .7 else None) for k, v in self.infl_true.items()}
        self.u_noise, self.u_sleep = rng.gauss(0, .5), rng.gauss(0, .5)
        self.hist = F.History()
        self.washer_on = 0

    def influence_true(self, src: str, dst: str) -> float:
        return 1.0 if src == dst else self.infl_true.get((src, dst), 0.05)


def simulate(households: int = 16, days: int = 14, seed: int = 7):
    """학습 행을 만든다. 반환: {agent: [(household, features, label), ...]}"""
    out = {"cleaning": [], "noise": [], "temperature": [], "lighting": []}
    for hid in range(households):
        h = Household(hid, seed)
        rng = h.rng
        t = datetime(2026, 9, 1, tzinfo=KST)
        cleaned_today: set[str] = set()
        for step in range(days * 96):
            hour, weekend = t.hour, t.weekday() >= 5
            # 사람 이동 (60%는 직전 상태 유지)
            for p in h.people:
                if p["where"] is None and step == 0 or rng.random() > .6:
                    where, rest = _where(hour, weekend, rng, p["bedroom"])
                    if (where, rest) != (p["where"], p["rest"]):
                        p["since"] = 0
                    p["where"], p["rest"] = where, rest
                else:
                    p["since"] += 15
            # 먼지 누적
            for r, _, _ in ROOMS:
                n = sum(p["where"] == r for p in h.people)
                h.dirt[r] += h.dirt_rate[r] * (0.6 * n + 0.1) * 0.25 * rng.uniform(.7, 1.3)
            # 세탁기
            h.washer_on = max(0, h.washer_on - 1) if h.washer_on else (6 if rng.random() < .03 else 0)
            # 청소 루틴: 매일 10시 요청, 4시간 안에 빈 방부터
            if hour == 0 and t.minute == 0:
                cleaned_today = set()
            due = 10 <= hour < 18 and len(cleaned_today) < 3
            snap = _snapshot(h, t, step, due, rng)
            h.hist.observe(snap)
            _rows(h, snap, out, rng)
            if due:
                for r in ("room1", "room2", "living"):
                    if r not in cleaned_today and not any(p["where"] == r for p in h.people):
                        h.dirt[r] *= 0.05
                        h.hist.mark_cleaned(r, t)
                        cleaned_today.add(r)
                        break
            t += STEP
    return out


def _snapshot(h: Household, t: datetime, step: int, due: bool, rng: random.Random):
    hour = t.hour
    dark = hour < 7 or hour >= 23
    rooms, nodes = [], []
    for r, rtype, nid in ROOMS:
        if nid is None:
            rooms.append(room(r, rtype))
            continue
        nodes.append(node(nid, r, t, step + 1))
        present = any(p["where"] == r for p in h.people)
        x = rng.random()
        occ = "UNKNOWN" if x < .03 else ("OCCUPIED" if present else "EMPTY")
        temp = round(rng.gauss(27, 1.5), 1) if rng.random() < .6 else None
        rooms.append(room(r, rtype, t, nid, occ, 0.95 if occ != "UNKNOWN" else None, 600,
                          "SILENT" if present else "UNKNOWN", "DARK" if dark else "BRIGHT", temp))
    people = []
    for i, p in enumerate(h.people):
        r = p["where"]
        if r is None or r not in NODE_ROOMS:
            continue
        if p["rest"]:
            label = "resting" if rng.random() < .85 else rng.choice(["other", "unknown"])
        else:
            label = "other" if rng.random() < .95 else rng.choice(["resting", "unknown"])
        confirmed = label != "unknown" and rng.random() < .8
        nid = next(n for rr, _, n in ROOMS if rr == r)
        people.append(person(p["pid"], p["profile"], r, p["identity"], t, nid, i + 1,
                             posture="LYING" if (p["rest"] and not dark) else ("UNKNOWN" if dark else "SITTING"),
                             label=label, confirmed=confirmed, stable=p["since"] * 60 if confirmed else None,
                             position=(1.0, 1.0)))
    devices = [
        device("vacuum_01", "kitchen", "vacuum", {"status": "docked", "mode": "normal", "target_room_ids": [],
                                                  "dock_room_id": "kitchen"}, t),
        device("washer_01", "laundry", "washer", {"status": "running" if h.washer_on else "idle", "mode": "normal"}, t),
        device("ac_room1", "room1", "ac", {"power": "on", "mode": "cool", "target_c": 25,
                                           "airflow_target_person_id": None}, t),
        device("ac_living", "living", "ac", {"power": "on", "mode": "cool", "target_c": 25,
                                             "airflow_target_person_id": None}, t),
    ]
    profiles = [{"profile_id": p["profile"], "identity": p["identity"], "preferred_min_c": p["center"] - 1,
                 "preferred_max_c": p["center"] + 1, "quiet_rest": p["quiet"], "quiet_meeting": p["quiet"]}
                for p in h.people]
    day = t.replace(hour=10, minute=0)
    routines = [{"routine_id": "daily", "occurrence_id": f"daily_{t:%Y%m%d}", "device_id": "vacuum_01",
                 "room_ids": ["room1", "room2", "living"], "due": due,
                 "requested_at": day.isoformat() if due else None,
                 "deadline_at": (day + timedelta(hours=4)).isoformat() if due else None}]
    influence = {}
    for (src, dst), v in h.infl_obs.items():
        influence.setdefault(src, {})[dst] = v
    return snapshot(t, step + 1, f"sim-{h.hid}", rooms, people, devices, profiles, routines, influence, nodes)


def _rows(h: Household, snap, out, rng: random.Random) -> None:
    hid = h.hid
    truth = {p["pid"]: p for p in h.people}
    # 청소: 방마다 숨은 먼지
    for room_id, row in F.cleaning_rows(snap, h.hist):
        out["cleaning"].append((hid, row, 1 - math.exp(-h.dirt[room_id]) + rng.gauss(0, .03)))
    # 소음: (기기, 모드, 소음 나는 방) × 사람이 있는 방
    occupied = F.occupied_rooms(snap)
    for dev, mode, src in F.noise_sources(snap):
        if dev.kind == "vacuum" and rng.random() > .15:
            continue  # 청소기 후보는 너무 많아서 일부만
        for dst in occupied:
            row = F.noise_row(snap, dev, mode, src, dst)
            here = [truth[p.person_id] for p in F.people_in(snap, dst)]
            rest = any(p["rest"] for p in here)
            quiet = any(p["rest"] and p["quiet"] for p in here)
            logit = 6 * row["level"] * h.influence_true(src, dst) - 3.2 + 1.5 * rest + 1.0 * quiet + h.u_noise
            out["noise"].append((hid, row, float(rng.random() < sigmoid(logit))))
    # 온도: 에어컨 있는 방의 사람 × 설정온도 후보 몇 개
    for p in snap.people:
        if p.room_id not in ("room1", "living"):
            continue
        tp = truth[p.person_id]
        prof = snap.profile(p.profile_id)
        for target in rng.sample(range(18, 31), 4):
            row = F.temperature_row(snap, p, prof, p.room_id, float(target))
            ideal = tp["center"] + 0.4 * tp["rest"] + tp["bias"]
            out["temperature"].append((hid, row, (target - ideal) / 1.5 + rng.gauss(0, .3)))
    # 빛: 휴식으로 관측된 사람
    for p in snap.people:
        if p.activity.label != "resting":
            continue
        tp = truth[p.person_id]
        row = F.lighting_row(snap, p)
        hr = snap.as_of.hour
        night = float(hr >= 21 or hr < 3)
        logit = -2 + 2.2 * row["is_bedroom"] + 2.0 * night + 0.6 * math.log1p(tp["since"]) + h.u_sleep
        label = float(tp["rest"] and rng.random() < sigmoid(logit))
        out["lighting"].append((hid, row, label))
