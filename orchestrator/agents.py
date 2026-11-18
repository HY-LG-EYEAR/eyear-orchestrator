"""에이전트 4개 (청소, 소음, 온도, 빛).

각 에이전트는 같은 WorldSnapshot을 읽고 AgentProposal 하나를 낸다. 명령을 직접 보내지 않는다.
학습 모델(models/*.joblib)이 있으면 점수를 모델로, 없으면 규칙 baseline으로 계산한다.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from orchestrator import features as F
from orchestrator.contract import WorldSnapshot
from orchestrator.schemas import AgentProposal, ProposedAction, Question, Restriction

if TYPE_CHECKING:
    from orchestrator.voice import DialogueManager

log = logging.getLogger(__name__)
MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DISTURB_P = 0.5      # 이 확률 이상이면 방해로 본다
ASK_SLEEP_P = 0.3    # 수면 의도 확률이 이 이상이면 질문한다


# ---------- 규칙 baseline (학습 모델이 없을 때, 그리고 학습 평가의 비교 기준) ----------

def rule_clean(row: dict) -> float:
    h = row["hours_since_clean"]
    base = 0.8 if math.isnan(h) else min(1.0, h / 24)
    return min(1.0, base + 0.1 * min(row["occupied_hours_since_clean"], 4) / 4)


def rule_noise(row: dict) -> float:
    infl = 0.0 if math.isnan(row["influence"]) else row["influence"]
    if row["level"] * infl >= 0.4:
        return 0.9
    if row["resting"] and row["confirmed"] and row["quiet_pref"] == 1.0:
        return 0.6
    return 0.1


def rule_vote(row: dict) -> float:
    return row["delta"] / 2.0  # 선호 범위 중앙보다 2°C 높으면 "조금 더움"(+1)


def rule_sleep(row: dict) -> float:
    night = row["hour_cos"] > 0.5  # 대략 21시~03시
    return 0.9 if row["is_bedroom"] and night else 0.4


# ---------- 모델 ----------

class Models:
    """models/*.joblib을 읽는다. 없으면 규칙을 쓴다."""

    EXPECTED = {"cleaning": F.CLEAN_FEATURES, "noise": F.NOISE_FEATURES,
                "temperature": F.TEMP_FEATURES, "lighting": F.LIGHT_FEATURES}

    def __init__(self, directory: Path | None = MODEL_DIR):
        self.m = {}
        if directory is not None:
            import joblib

            for name, features in self.EXPECTED.items():
                path = Path(directory) / f"{name}.joblib"
                if not path.exists():
                    continue
                try:
                    bundle = joblib.load(path)
                    if bundle.get("features") != features:
                        raise ValueError("특징 순서가 현재 코드와 다름")
                    self.m[name] = bundle
                except Exception as e:  # 깨진 파일, 버전 불일치 → 규칙으로 동작
                    log.warning("모델 %s를 읽지 못해 규칙을 씁니다: %s", name, e)

    def name(self, agent: str) -> str:
        b = self.m.get(agent)
        return f"{b['kind']}@{b['version']}" if b else "rule"

    def _predict(self, agent: str, rows: list[dict], names: list[str], proba: bool):
        b = self.m[agent]
        X = F.to_matrix(rows, names)
        return list(b["model"].predict_proba(X)[:, 1] if proba else b["model"].predict(X))

    def clean(self, rows):
        if "cleaning" in self.m and rows:
            return [float(min(1.0, max(0.0, v))) for v in self._predict("cleaning", rows, F.CLEAN_FEATURES, False)]
        return [rule_clean(r) for r in rows]

    def noise(self, rows):
        if "noise" in self.m and rows:
            return [float(v) for v in self._predict("noise", rows, F.NOISE_FEATURES, True)]
        return [rule_noise(r) for r in rows]

    def vote(self, rows):
        if "temperature" in self.m and rows:
            return [float(v) for v in self._predict("temperature", rows, F.TEMP_FEATURES, False)]
        return [rule_vote(r) for r in rows]

    def sleep(self, rows):
        if "lighting" in self.m and rows:
            return [float(v) for v in self._predict("lighting", rows, F.LIGHT_FEATURES, True)]
        return [rule_sleep(r) for r in rows]


@dataclass
class Context:
    snap: WorldSnapshot
    hist: F.History
    models: Models
    dialogue: "DialogueManager"
    protected: dict[str, set[str]]  # 소음 보호로 모드를 바꾼 기기 → 그 이유가 된 방들


def _usable(d) -> bool:
    return d is not None and d.online is True and d.manual_override is not True


def _proposal(ctx: Context, agent: str) -> AgentProposal:
    return AgentProposal(agent=agent, world_session_id=ctx.snap.world_session_id,
                         world_revision=ctx.snap.world_revision, model=ctx.models.name(agent))


# ---------- 청소 ----------

def cleaning_agent(ctx: Context) -> AgentProposal:
    """청소 루틴의 남은 방을 루틴 순서대로 청소하자고 제안한다. 재실 여부는 보지 않는다(소음 에이전트 몫)."""
    s, out = ctx.snap, _proposal(ctx, "cleaning")
    rows = F.cleaning_rows(s, ctx.hist)
    scores = ctx.models.clean([r for _, r in rows])
    for (room_id, _), v in zip(rows, scores):
        out.scores[f"urgency:{room_id}"] = round(v, 3)
    for rt in s.routines:
        if not rt.due:
            continue
        vac = s.device(rt.device_id)
        if not _usable(vac) or "start_cleaning" not in vac.capabilities.actions:
            out.reasons.append(f"VACUUM_UNAVAILABLE:{rt.device_id}")
            continue
        if vac.state.status not in ("docked", "idle", "completed"):
            out.reasons.append(f"VACUUM_BUSY:{vac.state.status}")
            continue
        remaining = [r for r in rt.room_ids if r not in ctx.hist.done_rooms.get(rt.occurrence_id, set())]
        if not remaining:
            continue
        out.actions.append(ProposedAction(device_id=vac.device_id, action="start_cleaning",
                                          params={"room_ids": remaining, "mode": "normal"},
                                          reason=f"ROUTINE_DUE:{rt.occurrence_id}"))
    return out


# ---------- 소음 ----------

def noise_agent(ctx: Context) -> AgentProposal:
    s, out = ctx.snap, _proposal(ctx, "noise")
    occupied = F.occupied_rooms(s)
    room_state = {r.room_id: r.occupancy.state for r in s.rooms}

    # 청소기: 사람이 있거나 재실을 모르는 방에서 청소 시작을 막자고 제안
    for d in s.devices:
        if d.kind != "vacuum":
            continue
        rooms = list(dict.fromkeys(r for rt in s.routines if rt.device_id == d.device_id for r in rt.room_ids))
        for r in rooms:
            st = room_state.get(r)
            if st == "OCCUPIED":
                p = ctx.models.noise([F.noise_row(s, d, "normal", r, r)])[0]
                out.scores[f"disturb:{d.device_id}:{r}"] = round(p, 3)
                rest = any(x.activity.label == "resting" for x in F.people_in(s, r))
                out.restrictions.append(Restriction(device_id=d.device_id, action="start_cleaning", room_id=r,
                                                    strength="strong",
                                                    reason="ROOM_OCCUPIED_RESTING" if rest else "ROOM_OCCUPIED"))
            elif st == "UNKNOWN":
                out.restrictions.append(Restriction(device_id=d.device_id, action="start_cleaning", room_id=r,
                                                    strength="weak", reason="OCCUPANCY_UNKNOWN"))

    # 세탁기·공기청정기: 영향받는 방에 사람이 있으면 저소음, 이유가 사라지면 원래 모드로
    for d in s.devices:
        if d.kind not in ("washer", "purifier") or not _usable(d) or d.room_id is None:
            continue
        running = d.state.status == "running" if d.kind == "washer" else d.state.power == "on"
        quiet_mode = "low_noise" if d.kind == "washer" else "low"
        loud_mode = "normal" if d.kind == "washer" else None
        if not running or quiet_mode not in d.capabilities.modes:
            continue
        cur = d.state.mode or (loud_mode or "")
        level_mode = cur if (d.kind, cur) in F.NOISE_LEVEL else (loud_mode or quiet_mode)
        reasons = []
        for r in occupied:
            p = ctx.models.noise([F.noise_row(s, d, level_mode, d.room_id, r)])[0]
            out.scores[f"disturb:{d.device_id}:{r}"] = round(p, 3)
            if p >= DISTURB_P:
                reasons.append(r)
        if reasons and cur != quiet_mode:
            out.actions.append(ProposedAction(device_id=d.device_id, action="set_mode", params={"mode": quiet_mode},
                                              reason="NEIGHBOR_OCCUPIED:" + ",".join(reasons)))
            ctx.protected[d.device_id] = set(reasons)
        elif not reasons and cur == quiet_mode and d.device_id in ctx.protected and loud_mode:
            # 보호 이유가 된 방이 모두 확실히 비었을 때만 되돌린다 (UNKNOWN은 해제 근거가 아님)
            if all(room_state.get(r) == "EMPTY" for r in ctx.protected[d.device_id]):
                out.actions.append(ProposedAction(device_id=d.device_id, action="set_mode", params={"mode": loud_mode},
                                                  reason="PROTECTION_ENDED"))
                ctx.protected.pop(d.device_id, None)
    return out


# ---------- 온도 ----------

def _grid(cap) -> list[float]:
    n = int(round((cap.max_c - cap.min_c) / cap.step_c))
    return [round(cap.min_c + i * cap.step_c, 3) for i in range(n + 1)]


def best_target(ctx: Context, people, room_id: str, cap) -> float:
    """현재 방 사람들의 예상 온열감(덥다+/춥다−) 제곱합이 가장 작은 설정 온도."""
    grid = _grid(cap)
    mean_center = sum(ctx.snap.profile(p.profile_id).center_c for p in people) / len(people)
    best, best_cost = None, None
    for t in grid:
        rows = [F.temperature_row(ctx.snap, p, ctx.snap.profile(p.profile_id), room_id, t) for p in people]
        cost = sum(v * v for v in ctx.models.vote(rows))
        key = (round(cost, 6), abs(t - mean_center))
        if best_cost is None or key < best_cost:
            best, best_cost = t, key
    return best


def temperature_agent(ctx: Context) -> AgentProposal:
    s, out = ctx.snap, _proposal(ctx, "temperature")
    for ac in s.devices:
        if ac.kind != "ac" or not _usable(ac) or ac.room_id is None:
            continue
        occ = s.room(ac.room_id).occupancy.state
        people = [p for p in F.people_in(s, ac.room_id)
                  if s.profile(p.profile_id) and s.profile(p.profile_id).center_c is not None]
        if not people:
            if occ == "EMPTY" and ac.state.power == "on" and "set_power" in ac.capabilities.actions:
                out.actions.append(ProposedAction(device_id=ac.device_id, action="set_power",
                                                  params={"power": "off"}, reason="ROOM_EMPTY"))
            continue
        cap = ac.capabilities.temperature
        if cap is None:
            continue
        target = best_target(ctx, people, ac.room_id, cap)
        out.scores[f"target:{ac.device_id}"] = target
        if ac.state.power != "on":
            out.actions.append(ProposedAction(device_id=ac.device_id, action="set_power", params={"power": "on"},
                                              reason="ROOM_OCCUPIED"))
        if ac.state.target_c != target:
            out.actions.append(ProposedAction(device_id=ac.device_id, action="set_target_temperature",
                                              params={"target_c": target},
                                              reason="PREFERENCE_" + "+".join(p.person_id for p in people)))
        if ac.capabilities.airflow_target == "person" and "set_airflow_target" in ac.capabilities.actions:
            located = [p for p in people if p.position is not None]
            if located:
                cool = min(located, key=lambda p: (s.profile(p.profile_id).center_c, p.person_id))
                if ac.state.airflow_target_person_id != cool.person_id:
                    out.actions.append(ProposedAction(device_id=ac.device_id, action="set_airflow_target",
                                                      params={"person_id": cool.person_id},
                                                      reason="PREFERS_COOLER"))
    return out


# ---------- 빛 ----------

def lighting_agent(ctx: Context) -> AgentProposal:
    s, out = ctx.snap, _proposal(ctx, "lighting")
    for p in s.people:
        if p.room_id is None:
            continue
        ep = ctx.dialogue.episode(p.person_id)
        if ep is None:
            continue
        if ep.state == "NOT_ASKED":
            node = s.node_for_room(p.room_id)
            if node is None or not node.supports_speak:
                out.reasons.append(f"NO_SPEAKER:{p.room_id}")
                continue
            prob = ctx.models.sleep([F.lighting_row(s, p)])[0]
            out.scores[f"sleep:{p.person_id}"] = round(prob, 3)
            if prob >= ASK_SLEEP_P:
                out.questions.append(Question(kind="sleep_intent", person_id=p.person_id, room_id=p.room_id))
            else:
                out.reasons.append(f"SLEEP_UNLIKELY:{p.person_id}")
        elif ep.state == "ANSWERED" and ep.answer == "yes" and not ep.consumed:
            # 소비(consumed) 표시는 중앙이 실제로 처리한 뒤에 한다 (일시정지로 막히면 재개 후 다시 제안)
            why = f"SLEEP_CONFIRMED:{p.person_id}"
            for d in s.devices:
                if d.room_id != ep.room_id:
                    continue
                if d.kind == "curtain":
                    out.actions.append(ProposedAction(device_id=d.device_id, action="set_curtain_position",
                                                      params={"position_pct": 0}, reason=why))
                if d.kind == "light":
                    out.actions.append(ProposedAction(device_id=d.device_id, action="set_power",
                                                      params={"power": "off"}, reason=why))
    return out


AGENTS = (cleaning_agent, noise_agent, temperature_agent, lighting_agent)
