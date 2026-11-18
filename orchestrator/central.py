"""중앙 판단 (Home Orchestrator).

판단 1회의 순서
1. 에이전트 4개가 같은 스냅샷으로 제안을 낸다.
2. 이미 그 상태인 제안은 뺀다. 직접 명령으로 잠근 기기의 자동 제안도 뺀다.
3. 충돌을 찾는다: 같은 기기·같은 동작에 다른 값, 또는 다른 에이전트의 제한(Restriction)에 걸리는 동작.
4. 충돌하지 않는 제안은 그대로 수용한다.
5. 충돌마다 선택지를 기계적으로 만들고, LLM이 하나를 고른다. LLM이 없거나 실패하면 결정적 fallback.
   같은 충돌·같은 상황(마감 구간, 점수 구간)이면 일정 시간 동안 앞의 선택을 재사용한다.
6. 수용한 동작을 검증(기기 능력·범위·상태 신선도·수동 조작·사용자 금지·중복)한 뒤 명령으로 보낸다.
7. 빛 에이전트가 질문을 요청하면 Node로 말할 문장을 보낸다.

일시정지(pause)는 자동 제어만 멈춘다. 사람이 직접 말한 명령("불 켜줘")은 일시정지 중에도 실행한다.
LLM 호출은 lock 밖에서 하고, 응답 후 더 새로운 스냅샷이 와 있으면 최신 상태로 다시 계산한다.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

from orchestrator import agents as A
from orchestrator.contract import WorldSnapshot
from orchestrator.features import History
from orchestrator.schemas import (
    AgentProposal, Choice, Conflict, Decision, DeviceCommand, DeviceResult, Option, ProposedAction,
    Rejected, Restriction, SpeakCommand, VoiceCommand, VoiceReply,
)
from orchestrator.voice import DialogueManager, LISTEN_SECONDS, parse_command, question_text

log = logging.getLogger(__name__)

MANUAL_HOLD = timedelta(minutes=30)   # 직접 명령 뒤 같은 기기를 자동으로 되돌리지 않는 시간
MEMO_TTL = timedelta(minutes=15)      # 같은 충돌의 선택을 재사용하는 최대 시간 (스냅샷 시각 기준)
ACCEPT_TIMEOUT_S = 30.0               # 명령을 보낸 뒤 접수(accepted) 또는 결과를 기다리는 시간
SNAPSHOT_MAX_AGE_S = 120.0            # 마지막 스냅샷을 받은 뒤 이 시간이 지나면 상태를 오래된 것으로 봄
TERMINAL = {"completed", "failed", "rejected"}
IDEMPOTENT_ACTIONS = {"set_mode", "set_power", "set_target_temperature", "set_airflow_target", "set_curtain_position"}
TRANSIENT_FAILURES = {"busy", "timeout", "temporarily_unavailable"}   # 다시 시도해도 되는 실패 이유
MAX_RETRIES = 2
RETRY_INTERVAL = timedelta(minutes=5)


def occurrence_of(a: ProposedAction) -> str | None:
    """청소 제안의 reason(ROUTINE_DUE:<occurrence_id>)에서 회차를 꺼낸다."""
    return a.reason.split(":", 1)[1] if a.reason.startswith("ROUTINE_DUE:") else None


def action_key(device_id: str, action: str, params: dict, occurrence_id: str | None) -> tuple:
    return (device_id, action, json.dumps(params, sort_keys=True, default=str), occurrence_id)


# ---------- 검증 ----------

def validate(a: ProposedAction, snap: WorldSnapshot) -> str | None:
    """실행 가능하면 None, 아니면 거절 이유."""
    d = snap.device(a.device_id)
    if d is None:
        return "UNKNOWN_DEVICE"
    if d.online is not True:
        return "DEVICE_NOT_ONLINE"
    if d.manual_override is True:
        return "MANUAL_OVERRIDE"
    if d.evidence.freshness != "current":
        return "DEVICE_STATE_STALE"
    caps, p = d.capabilities, a.params
    if a.action not in caps.actions:
        return "ACTION_NOT_SUPPORTED"
    if a.action == "start_cleaning":
        rooms = p.get("room_ids")
        if not rooms or any(snap.room(r) is None for r in rooms):
            return "BAD_ROOMS"
        if "mode" in p and p["mode"] not in caps.modes:
            return "MODE_NOT_SUPPORTED"
    elif a.action == "set_mode":
        if p.get("mode") not in caps.modes:
            return "MODE_NOT_SUPPORTED"
    elif a.action == "set_power":
        if p.get("power") not in ("on", "off"):
            return "BAD_POWER"
    elif a.action == "set_target_temperature":
        t, rng = p.get("target_c"), caps.temperature
        if rng is None or not isinstance(t, (int, float)) or not rng.min_c <= t <= rng.max_c:
            return "TEMPERATURE_OUT_OF_RANGE"
        steps = (t - rng.min_c) / rng.step_c
        if abs(steps - round(steps)) > 1e-6:
            return "TEMPERATURE_STEP"
    elif a.action == "set_airflow_target":
        person = snap.person(p.get("person_id", ""))
        if caps.airflow_target != "person" or person is None or person.room_id != d.room_id:
            return "AIRFLOW_TARGET_INVALID"
    elif a.action == "set_curtain_position":
        if not isinstance(p.get("position_pct"), (int, float)) or not 0 <= p["position_pct"] <= 100:
            return "BAD_POSITION"
    rooms = set(p.get("room_ids", [])) | ({d.room_id} if d.room_id else set())
    for u in snap.user_restrictions:
        if u.expires_at is not None and u.expires_at <= snap.as_of:
            continue
        hit = (u.target_type == "device" and u.target_id == d.device_id) or \
              (u.target_type == "room" and u.target_id in rooms)
        if hit and (u.action is None or u.action == a.action):
            return f"USER_RESTRICTION:{u.restriction_id}"
    return None


def already_done(a: ProposedAction, snap: WorldSnapshot) -> bool:
    d = snap.device(a.device_id)
    if d is None:
        return False
    s, p = d.state, a.params
    return {
        "set_mode": lambda: getattr(s, "mode", None) == p.get("mode"),
        "set_power": lambda: getattr(s, "power", None) == p.get("power"),
        "set_target_temperature": lambda: getattr(s, "target_c", None) == p.get("target_c"),
        "set_airflow_target": lambda: getattr(s, "airflow_target_person_id", None) == p.get("person_id"),
        "set_curtain_position": lambda: getattr(s, "position_pct", None) == p.get("position_pct"),
    }.get(a.action, lambda: False)()


# ---------- 충돌과 선택지 ----------

def find_conflicts(proposals: list[AgentProposal], snap: WorldSnapshot):
    """(충돌 없는 동작들, 충돌 목록)

    현재 범위: 같은 (기기, 동작)에 서로 다른 값, 그리고 청소 시작과 소음 제한의 겹침.
    동작 이름이 다른 조합(예: start_cleaning과 return_to_dock)은 아직 충돌로 보지 않는다.
    """
    by_key: dict[tuple, list[tuple[str, ProposedAction]]] = defaultdict(list)
    for prop in proposals:
        for a in prop.actions:
            by_key[(a.device_id, a.action)].append((prop.agent, a))
    restrictions: list[tuple[str, Restriction]] = [(p.agent, r) for p in proposals for r in p.restrictions]

    accepted, conflicts = [], []
    for (device_id, action), items in by_key.items():
        distinct: dict[tuple, tuple[list[str], ProposedAction]] = {}
        for agent, a in items:
            agents, _ = distinct.setdefault(a.key(), ([], a))
            agents.append(agent)
        hits = [(ag, r) for ag, r in restrictions if r.device_id == device_id and r.action == action]

        if action == "start_cleaning" and hits:
            for agents, a in distinct.values():
                blocked = [(ag, r) for ag, r in hits if r.room_id in a.params.get("room_ids", [])]
                if blocked:
                    conflicts.append(_cleaning_conflict(agents, a, blocked, snap))
                else:
                    accepted.append(a)
        elif len(distinct) > 1:
            conflicts.append(_value_conflict(device_id, action, list(distinct.values())))
        else:
            accepted.append(next(iter(distinct.values()))[1])
    return accepted, conflicts


def _cleaning_conflict(agents, a: ProposedAction, blocked, snap) -> Conflict:
    rooms = list(a.params["room_ids"])
    mode = a.params.get("mode", "normal")
    strong = [r.room_id for _, r in blocked if r.strength == "strong"]
    weak = [r.room_id for _, r in blocked if r.strength == "weak"]
    restricting = sorted({ag for ag, _ in blocked})
    cid = f"{a.device_id}:start_cleaning"

    def act(rs, m):
        return ProposedAction(device_id=a.device_id, action="start_cleaning",
                              params={"room_ids": rs, "mode": m}, reason=a.reason)

    opts = [Option(option_id="original", summary=f"제안대로 {rooms} 청소({mode})", actions=[act(rooms, mode)],
                   from_agents=agents)]
    free = [r for r in rooms if r not in strong and r not in weak]
    if free:
        opts.append(Option(option_id="skip_restricted", summary=f"{free}만 청소, {strong + weak}은 미룸",
                           actions=[act(free, mode)], deferred_rooms=strong + weak, from_agents=agents + restricting))
    free_strong = [r for r in rooms if r not in strong]
    if weak and free_strong and free_strong != free:
        opts.append(Option(option_id="skip_occupied", summary=f"{free_strong}만 청소, {strong}은 미룸",
                           actions=[act(free_strong, mode)], deferred_rooms=strong, from_agents=agents + restricting))
    dev = snap.device(a.device_id)
    if dev and "quiet" in dev.capabilities.modes and mode != "quiet":
        opts.append(Option(option_id="quiet", summary=f"{rooms}를 저소음으로 청소", actions=[act(rooms, "quiet")],
                           from_agents=agents))
    opts.append(Option(option_id="hold", summary=f"청소를 모두 미룸 {rooms}", actions=[], deferred_rooms=rooms,
                       from_agents=restricting))
    why = ", ".join(f"{r.room_id}:{r.reason}({r.strength})" for _, r in blocked)
    return Conflict(conflict_id=cid, device_id=a.device_id,
                    description=f"청소 제안 {rooms}가 소음 제한과 겹침 [{why}]", options=opts)


def _value_conflict(device_id, action, items) -> Conflict:
    opts = [Option(option_id=f"opt{i}", summary=f"{'/'.join(ag)}: {a.params}", actions=[a], from_agents=ag)
            for i, (ag, a) in enumerate(items, 1)]
    opts.append(Option(option_id="hold", summary="바꾸지 않음", actions=[], from_agents=[]))
    return Conflict(conflict_id=f"{device_id}:{action}", device_id=device_id,
                    description=f"{device_id}.{action}에 서로 다른 값 제안", options=opts)


def fallback_choice(c: Conflict) -> Choice:
    """LLM 없이: 제한을 모두 지키는 선택지 중 가장 많이 수행하는 것, 없으면 보류."""
    ids = [o.option_id for o in c.options]
    for pick in ("skip_restricted", "hold"):
        if pick in ids:
            return Choice(conflict_id=c.conflict_id, option_id=pick, reason="FALLBACK_RESPECT_RESTRICTIONS")
    return Choice(conflict_id=c.conflict_id, option_id="hold", reason="FALLBACK_HOLD")


def summarize(snap: WorldSnapshot, proposals: list[AgentProposal]) -> dict:
    """LLM에 줄 짧은 상황 요약."""
    return {
        "time": snap.as_of.isoformat(),
        "rooms": {r.room_id: {"type": r.room_type, "occupancy": r.occupancy.state} for r in snap.rooms},
        "people": [{"person_id": p.person_id, "room": p.room_id, "activity": p.activity.label,
                    "confirmed": p.activity.confirmed} for p in snap.people],
        "profiles": [{"profile_id": p.profile_id, "quiet_rest": p.quiet_rest, "quiet_meeting": p.quiet_meeting}
                     for p in snap.profiles],
        "routines": [{"rooms": r.room_ids, "due": r.due, "deadline_at": r.deadline_at} for r in snap.routines],
        "agent_scores": {p.agent: p.scores for p in proposals},
    }


def _deadline_bucket(snap: WorldSnapshot) -> list:
    out = []
    for r in snap.routines:
        if not r.due or r.deadline_at is None:
            out.append((r.occurrence_id, r.due, None))
            continue
        left = (r.deadline_at - snap.as_of).total_seconds() / 60
        out.append((r.occurrence_id, r.due, "overdue" if left < 0 else "<30m" if left < 30 else "<60m" if left < 60 else "later"))
    return out


def memo_key(c: Conflict, snap: WorldSnapshot, proposals: list[AgentProposal]) -> str:
    """같은 충돌이라도 관련 선호·사람·제한·루틴·마감 구간 중 하나라도 바뀌면 다른 상황으로 본다(보수적 무효화)."""
    return json.dumps({
        "conflict": c.model_dump(),
        "deadline": _deadline_bucket(snap),
        "routines": [(r.occurrence_id, r.due, r.room_ids) for r in snap.routines],
        "people": [(p.person_id, p.room_id, p.profile_id, p.activity.label, p.activity.confirmed) for p in snap.people],
        "profiles": [p.model_dump() for p in snap.profiles],
        "restrictions": [u.model_dump(mode="json") for u in snap.user_restrictions],
    }, sort_keys=True, default=str)


def decision_key(snap: WorldSnapshot) -> str:
    """판단에 영향을 주는 값만. 같으면 다시 판단하지 않는다(heartbeat, stable_for_s 증가 등).
    온도는 0.5°C 단위, 근거는 신선도 상태만 본다."""
    def temp(m):
        return None if m.value is None else round(m.value * 2) / 2
    return json.dumps({
        "rooms": [(r.room_id, r.occupancy.state, r.occupancy.evidence.freshness, r.light.level,
                   temp(r.temperature_c), r.temperature_c.evidence.freshness) for r in snap.rooms],
        "people": [(p.person_id, p.room_id, p.activity.label, p.activity.confirmed, p.position is not None,
                    p.evidence.freshness) for p in snap.people],
        "devices": [(d.device_id, d.online, d.manual_override, d.evidence.freshness, d.state.model_dump())
                    for d in snap.devices],
        "routines": [(r.occurrence_id, r.due, r.room_ids) for r in snap.routines],
        "deadline": _deadline_bucket(snap),
        "profiles": [p.model_dump() for p in snap.profiles],
        "restrictions": [u.model_dump(mode="json") for u in snap.user_restrictions],
        "influence": snap.room_influence,
    }, sort_keys=True, default=str)


# ---------- 중앙 ----------

@dataclass
class Memo:
    choice: Choice
    mode: str            # llm | fallback
    summary: str
    expires_at: datetime


@dataclass
class Pending:
    """발행한 명령의 진행 상태.
    sent → (accepted | in_progress | unknown) → completed / failed / rejected 결과로 종료.
    unknown: 접수 응답이 없어 실행 여부를 모름. 실패로 보지 않고, 같은 명령을 다시 보내지 않는다."""

    cmd: DeviceCommand
    sent_at: float
    occurrence_id: str | None = None   # 명령 당시 청소 회차 (완료는 이 회차에 기록)
    status: str = "sent"


@dataclass
class Plan:
    snap: WorldSnapshot
    proposals: list[AgentProposal]
    accepted: list[ProposedAction]
    conflicts: list[Conflict]
    keys: dict[str, str]
    rejected: list[Rejected] = field(default_factory=list)
    sleep_people: set[str] = field(default_factory=set)  # 수면 확인 뒤 조명·커튼을 제안받은 사람
    generation: int = 0      # reset·세션 세대. 다르면 계획을 버린다
    change_no: int = 0       # 중앙 상태 변경 번호(직접 명령·제어·대답). 다르면 다시 계산한다


class Orchestrator:
    def __init__(self, send_command: Callable[[DeviceCommand], None],
                 speak: Callable[[str, SpeakCommand], None],
                 llm=None, models: A.Models | None = None,
                 clock: Callable[[], float] = time.monotonic,
                 trace_path: Path | None = None, history_size: int = 200):
        self.send_command, self.speak, self.llm = send_command, speak, llm
        self.models = models or A.Models(None)
        self.clock, self.trace_path = clock, trace_path
        self.lock = threading.RLock()
        self.decisions: deque[Decision] = deque(maxlen=history_size)
        self.generation = 0
        self.reset()

    # ----- 상태 -----

    def reset(self, session_id: str | None = None, clock_mode: str | None = None) -> None:
        with self.lock:
            self.generation += 1
            self.change_no = 0
            self.session_id, self.clock_mode = session_id, clock_mode
            self.snap: WorldSnapshot | None = None
            self.snap_received: float | None = None
            self.failures: dict[tuple, tuple[int, datetime | None]] = {}  # 동작 키 → (실패 횟수, 다시 시도 가능 시각)
            self.hist = History()
            self.dialogue = DialogueManager()
            self.protected: dict[str, set[str]] = {}
            self.pending: dict[str, Pending] = {}
            self.holds: dict[str, datetime] = {}
            self.memo: dict[str, Memo] = {}
            self.last_key: str | None = None
            self.paused = False

    def status(self) -> dict:
        with self.lock:
            return {
                "session_id": self.session_id, "clock_mode": self.clock_mode, "paused": self.paused,
                "world_revision": self.snap.world_revision if self.snap else None,
                "llm": bool(self.llm), "models": {a: self.models.name(a) for a in
                                                 ("cleaning", "noise", "temperature", "lighting")},
                "pending_commands": {k: p.status for k, p in self.pending.items()},
                "episodes": {k: vars(v) for k, v in self.dialogue.episodes.items()},
            }

    # ----- 입력 -----

    def on_snapshot(self, snap: WorldSnapshot) -> Decision | None:
        with self.lock:
            cur = self.snap
            if cur is not None and cur.world_session_id == snap.world_session_id \
                    and snap.world_revision <= cur.world_revision:
                log.info("옛 스냅샷 무시: rev %s (현재 %s)", snap.world_revision, cur.world_revision)
                return None
            if cur is not None and cur.world_session_id != snap.world_session_id:
                self.generation += 1   # 세션이 바뀌면 진행 중인 판단은 버린다
            self.hist.observe(snap)
            self.dialogue.update(snap)
            self.snap, self.snap_received = snap, self.clock()
            self._observe_pending(snap)
            key = decision_key(snap) + json.dumps({k: (v.state, v.answer) for k, v in self.dialogue.episodes.items()})
            if key == self.last_key:
                return None
            self.last_key = key
        return self.decide("snapshot")

    def on_voice_reply(self, reply: VoiceReply) -> Decision | None:
        with self.lock:
            if self.snap is None:
                return None
            ep = self.dialogue.on_reply(reply, self.clock())
            if ep is None:
                return None
            self.last_key = None
            self.change_no += 1
        return self.decide(f"voice_reply:{ep.answer}")

    def on_voice_command(self, cmd: VoiceCommand) -> Decision | None:
        """직접 명령. 일시정지 중에도 실행한다(일시정지는 자동 제어만 멈춤)."""
        with self.lock:
            if self.snap is None:
                return None
            snap = self.snap
            node = snap.node_for_room(cmd.room_id)
            if node is None or node.node_id != cmd.node_id:
                log.warning("등록되지 않은 Node의 명령 무시: %s@%s", cmd.node_id, cmd.room_id)
                return None
            self.change_no += 1
            parsed = parse_command(cmd.text)
            accepted, rejected, speech = [], [], []
            if not self._snapshot_fresh():
                speech.append(self._say(cmd.room_id, "지금은 집 상태를 확인할 수 없어 실행하지 않았어요.", False))
                if parsed is not None:
                    dev = next((d for d in snap.devices if d.kind == parsed[0] and d.room_id == cmd.room_id), None)
                    if dev is not None:
                        rejected.append(Rejected(action=ProposedAction(device_id=dev.device_id, action=parsed[1],
                                                                       params=parsed[2], reason="DIRECT_COMMAND"),
                                                 reason="SNAPSHOT_STALE"))
            elif parsed is None:
                speech.append(self._say(cmd.room_id, "죄송해요, 잘 알아듣지 못했어요.", False))
            else:
                kind, action, params, ack = parsed
                dev = next((d for d in snap.devices if d.kind == kind and d.room_id == cmd.room_id), None)
                if dev is None:
                    speech.append(self._say(cmd.room_id, "이 방에서는 그 기기를 찾지 못했어요.", False))
                else:
                    a = ProposedAction(device_id=dev.device_id, action=action, params=params, reason="DIRECT_COMMAND")
                    reason = validate(a, snap)
                    if reason:
                        rejected.append(Rejected(action=a, reason=reason))
                        speech.append(self._say(cmd.room_id, "지금은 그 동작을 할 수 없어요.", False))
                    else:
                        accepted.append(a)
                        self.holds[dev.device_id] = snap.as_of + MANUAL_HOLD
                        speech.append(self._say(cmd.room_id, ack, False))
            plan_id = str(uuid.uuid4())
            commands = self._dispatch(accepted, snap, plan_id, ["DIRECT_COMMAND"])
            return self._record(plan_id, snap, f"voice_command:{cmd.text}", [], accepted, [], [], "none",
                                commands, rejected, speech, "직접 명령은 자동 계획보다 우선")

    def on_device_result(self, res: DeviceResult) -> None:
        with self.lock:
            p = self.pending.get(res.command_id)
            if p is None:
                return
            if p.cmd.device_id != res.device_id or p.cmd.world_session_id != res.world_session_id:
                log.warning("명령과 맞지 않는 결과 무시: %s", res.command_id)
                return
            if res.status not in TERMINAL:   # accepted: 아직 진행 중
                p.status = res.status
                return
            self.pending.pop(res.command_id)
            key = action_key(p.cmd.device_id, p.cmd.action, p.cmd.params, p.occurrence_id)
            if res.status == "completed":
                self.failures.pop(key, None)
                if p.cmd.action == "start_cleaning":
                    for room_id in p.cmd.params.get("room_ids", []):
                        self.hist.mark_cleaned(room_id, res.timestamp, p.occurrence_id)
                return
            # 알려진 일시적 실패만 간격·횟수를 제한해 다시 시도. 거절·원인 불명은 자동으로 반복하지 않는다
            count = self.failures.get(key, (0, None))[0] + 1
            transient = res.status == "failed" and (res.reason or "") in TRANSIENT_FAILURES and count <= MAX_RETRIES
            self.failures[key] = (count, res.timestamp + RETRY_INTERVAL if transient else None)

    def tick(self) -> list[str]:
        with self.lock:
            self._expire_pending()
            return [ep.person_id for ep in self.dialogue.check_deadlines(self.clock())]

    def control(self, action: str) -> Decision | None:
        """pause/resume/reset. resume은 상태가 신선하면 바로 다시 판단한다."""
        with self.lock:
            if action == "pause":
                self.paused = True
                self.change_no += 1
            elif action == "resume":
                self.paused, self.last_key = False, None
                self.change_no += 1
                if not self._snapshot_fresh():
                    return None
            elif action == "reset":
                self.reset(self.session_id, self.clock_mode)
            if action != "resume":
                return None
        return self.decide("resume")

    def _snapshot_fresh(self) -> bool:
        """마지막 스냅샷을 받은 지 SNAPSHOT_MAX_AGE_S 안인가. live·replay 모두 '받은 시각'으로 판단해
        녹화 시나리오의 과거 날짜를 현재 시각과 비교하지 않는다. live의 as_of 나이는 서버가 수신 때 검사한다."""
        return self.snap is not None and self.snap_received is not None \
            and self.clock() - self.snap_received <= SNAPSHOT_MAX_AGE_S

    # ----- 판단 -----

    def decide(self, trigger: str) -> Decision | None:
        """충돌이 새 상황이면 LLM을 lock 밖에서 부르고, 그 사이 상태가 바뀌었으면 최신 상태로 다시 계산한다."""
        fresh: dict[str, str] = {}
        for _ in range(3):
            with self.lock:
                if self.snap is None:
                    return None
                plan = self._prepare()
                missing = [c for c in plan.conflicts if self._memo_get(plan.keys[c.conflict_id], plan.snap) is None]
                if not missing:
                    return self._finish(plan, trigger, fresh)
                if self.llm is None:
                    self._remember(missing, plan, None, fresh)
                    return self._finish(plan, trigger, fresh)
                rev, context = plan.snap.world_revision, summarize(plan.snap, plan.proposals)
            sel = self.llm.select(context, missing)
            with self.lock:
                # reset·세션 변경 → 이 판단은 버린다 (이전 세션 명령을 내지 않음)
                if self.generation != plan.generation or self.snap is None \
                        or self.snap.world_session_id != plan.snap.world_session_id:
                    log.info("LLM 대기 중 reset·세션 변경 → 판단 폐기")
                    return None
                self._remember(missing, plan, sel if sel and self._valid_selection(sel, missing) else None, fresh)
                # 새 스냅샷, 직접 명령, 제어, 대답이 있었으면 최신 상태로 다시 계산 (LLM 선택은 캐시로 재사용)
                if self.snap.world_revision == rev and self.change_no == plan.change_no:
                    return self._finish(plan, trigger, fresh)
        with self.lock:   # 계속 바뀌면 마지막 상태로 fallback
            plan = self._prepare()
            self._remember([c for c in plan.conflicts
                            if self._memo_get(plan.keys[c.conflict_id], plan.snap) is None], plan, None, fresh)
            return self._finish(plan, trigger, fresh)

    def _prepare(self) -> Plan:
        snap = self.snap
        ctx = A.Context(snap, self.hist, self.models, self.dialogue, self.protected)
        proposals = [agent(ctx) for agent in A.AGENTS]
        sleep_people = {a.reason.split(":", 1)[1] for p in proposals for a in p.actions
                        if a.reason.startswith("SLEEP_CONFIRMED:")}
        rejected: list[Rejected] = []
        for p in proposals:
            kept = []
            for a in p.actions:
                if already_done(a, snap):
                    continue
                until = self.holds.get(a.device_id)
                if until and snap.as_of < until:
                    rejected.append(Rejected(action=a, reason="DIRECT_COMMAND_HOLD"))
                    continue
                count, retry_at = self.failures.get(action_key(a.device_id, a.action, a.params, occurrence_of(a)), (0, None))
                if count and (retry_at is None or snap.as_of < retry_at):
                    rejected.append(Rejected(action=a, reason="RETRY_BLOCKED" if retry_at is None else "RETRY_WAIT"))
                    continue
                kept.append(a)
            p.actions[:] = kept
        accepted, conflicts = find_conflicts(proposals, snap)
        keys = {c.conflict_id: memo_key(c, snap, proposals) for c in conflicts}
        return Plan(snap, proposals, accepted, conflicts, keys, rejected, sleep_people,
                    generation=self.generation, change_no=self.change_no)

    def _memo_get(self, key: str, snap: WorldSnapshot) -> Memo | None:
        m = self.memo.get(key)
        if m is not None and snap.as_of >= m.expires_at:
            self.memo.pop(key)
            return None
        return m

    def _remember(self, conflicts, plan: Plan, sel, fresh: dict[str, str]) -> None:
        got = {c.conflict_id: c for c in sel.choices} if sel else {}
        for c in conflicts:
            key = plan.keys[c.conflict_id]
            if c.conflict_id in got:
                self.memo[key] = Memo(got[c.conflict_id], "llm", sel.summary, plan.snap.as_of + MEMO_TTL)
            else:
                summary = "LLM 응답 없음 또는 형식 오류 → 결정적 선택" if self.llm else "LLM 없이 제한을 지키는 선택"
                self.memo[key] = Memo(fallback_choice(c), "fallback", summary, plan.snap.as_of + MEMO_TTL)
            fresh[key] = self.memo[key].mode

    def _finish(self, plan: Plan, trigger: str, fresh: dict[str, str]) -> Decision:
        snap, proposals, rejected = plan.snap, plan.proposals, list(plan.rejected)
        accepted = list(plan.accepted)
        choices, modes, summaries = [], set(), []
        for c in plan.conflicts:
            key = plan.keys[c.conflict_id]
            m = self._memo_get(key, snap)
            choices.append(m.choice)
            modes.add(fresh.get(key, "cached"))
            summaries.append(m.summary)
            opt = next(o for o in c.options if o.option_id == m.choice.option_id)
            accepted.extend(opt.actions)
        arbitration = "none" if not modes else ("llm" if "llm" in modes else "fallback" if "fallback" in modes else "cached")

        self._expire_pending()
        valid = []
        for a in accepted:
            reason = validate(a, snap)
            if reason:
                rejected.append(Rejected(action=a, reason=reason))
            elif any(p.cmd.device_id == a.device_id and p.cmd.action == a.action and p.cmd.params == a.params
                     for p in self.pending.values()):
                rejected.append(Rejected(action=a, reason="DUPLICATE_PENDING"))
            else:
                valid.append(a)
        if self.paused:
            rejected += [Rejected(action=a, reason="PAUSED") for a in valid]
            valid = []
        plan_id = str(uuid.uuid4())
        commands = self._dispatch(valid, snap, plan_id, [trigger])

        # 수면 확인 뒤의 조명·커튼 제안은 일시정지로 막혔을 때만 다시 시도하고, 그 외에는 한 번으로 끝낸다
        if not self.paused:
            for pid in plan.sleep_people:
                ep = self.dialogue.episode(pid)
                if ep:
                    ep.consumed = True

        speech = []
        for p in proposals:
            for q in p.questions:
                if self.paused or self.dialogue.awaiting_in_room(q.room_id):
                    continue
                person, node = snap.person(q.person_id), snap.node_for_room(q.room_id)
                cmd = self._say(q.room_id, question_text(person.identity), True)
                self.dialogue.asked(q.person_id, cmd.command_id, self.clock(), cmd.text, node.node_id)
                speech.append(cmd)
        return self._record(plan_id, snap, trigger, proposals, accepted, plan.conflicts, choices, arbitration,
                            commands, rejected, speech, " / ".join(dict.fromkeys(summaries)))

    @staticmethod
    def _valid_selection(sel, conflicts) -> bool:
        want = {c.conflict_id: {o.option_id for o in c.options} for c in conflicts}
        got = {c.conflict_id: c.option_id for c in sel.choices}
        return set(got) == set(want) and all(got[k] in want[k] for k in want)

    def _expire_pending(self) -> None:
        """접수 응답(accepted)이 ACCEPT_TIMEOUT_S 안에 없을 때만 처리한다. 접수된 작업은 시간으로 끝내지 않는다.
        - 값을 맞추는 동작(set_*): 다음 스냅샷의 기기 상태로 확인되므로 pending을 풀어 다시 보낼 수 있게 한다
        - 그 밖의 동작(청소 시작 등): 실행 여부를 모름(unknown)으로 두고 자동으로 다시 보내지 않는다"""
        now = self.clock()
        for cid, p in list(self.pending.items()):
            if p.status != "sent" or now - p.sent_at <= ACCEPT_TIMEOUT_S:
                continue
            if p.cmd.action in IDEMPOTENT_ACTIONS:
                log.warning("접수 응답 없음, 상태 확인 후 다시 보낼 수 있게 함: %s %s", p.cmd.device_id, p.cmd.action)
                self.pending.pop(cid)
            else:
                log.warning("접수 응답 없음, 실행 여부 모름(다시 보내지 않음): %s %s", p.cmd.device_id, p.cmd.action)
                p.status = "unknown"

    def _observe_pending(self, snap: WorldSnapshot) -> None:
        """청소 시작의 접수 여부를 기기 상태로 확인한다: 청소기가 그 방들을 청소 중이면 진행 중."""
        for p in self.pending.values():
            if p.cmd.action != "start_cleaning" or p.status not in ("sent", "unknown"):
                continue
            d = snap.device(p.cmd.device_id)
            if d is not None and getattr(d.state, "status", None) == "cleaning" \
                    and set(d.state.target_room_ids) == set(p.cmd.params.get("room_ids", [])):
                p.status = "in_progress"

    def _dispatch(self, actions, snap, plan_id: str, reasons) -> list[DeviceCommand]:
        commands = []
        for a in actions:
            cmd = DeviceCommand(plan_id=plan_id, issued_at=snap.as_of, world_session_id=snap.world_session_id,
                                based_on_revision=snap.world_revision, device_id=a.device_id, action=a.action,
                                params=a.params, reason_codes=[a.reason] + reasons)
            self.pending[cmd.command_id] = Pending(cmd, self.clock(), occurrence_of(a))
            commands.append(cmd)
        for cmd in commands:
            try:
                self.send_command(cmd)
            except Exception as e:
                log.error("명령 전송 실패 %s: %s", cmd.device_id, e)
        return commands

    def _say(self, room_id: str, text: str, expect_reply: bool) -> SpeakCommand:
        cmd = SpeakCommand(timestamp=self.snap.as_of, text=text, expect_reply=expect_reply,
                           listen_seconds=LISTEN_SECONDS)
        try:
            self.speak(room_id, cmd)
        except Exception as e:
            log.error("음성 전송 실패 %s: %s", room_id, e)
        return cmd

    def _record(self, plan_id, snap, trigger, proposals, accepted, conflicts, choices, arbitration,
                commands, rejected, speech, summary) -> Decision:
        d = Decision(plan_id=plan_id, world_session_id=snap.world_session_id, world_revision=snap.world_revision,
                     decided_at=snap.as_of, trigger=trigger, proposals=proposals, accepted=accepted,
                     conflicts=conflicts, choices=choices, arbitration=arbitration, commands=commands,
                     rejected=rejected, speech=speech, summary=summary)
        self.decisions.append(d)
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(d.model_dump_json() + "\n")
        return d
