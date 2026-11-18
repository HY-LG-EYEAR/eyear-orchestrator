"""중앙 판단 (Home Orchestrator).

판단 1회의 순서
1. 에이전트 4개가 같은 스냅샷으로 제안을 낸다.
2. 이미 그 상태인 제안은 뺀다. 직접 명령으로 잠근 기기의 자동 제안도 뺀다.
3. 충돌을 찾는다: 같은 기기·같은 동작에 다른 값, 또는 다른 에이전트의 제한(Restriction)에 걸리는 동작.
4. 충돌하지 않는 제안은 그대로 수용한다.
5. 충돌마다 선택지를 기계적으로 만들고, LLM이 하나를 고른다. LLM이 없거나 실패하면 결정적 fallback.
6. 수용한 동작을 검증(기기 능력·범위·수동 조작·사용자 금지·중복)한 뒤 명령으로 보낸다.
7. 빛 에이전트가 질문을 요청하면 Node로 말할 문장을 보낸다.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import defaultdict, deque
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

MANUAL_HOLD = timedelta(minutes=30)  # 직접 명령 뒤 같은 기기를 자동으로 되돌리지 않는 시간


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
    """(충돌 없는 동작들, 충돌 목록)"""
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


def decision_key(snap: WorldSnapshot) -> str:
    """판단에 영향을 주는 값만. 같으면 다시 판단하지 않는다(heartbeat, stable_for_s 증가 등)."""
    return json.dumps({
        "rooms": [(r.room_id, r.occupancy.state, r.light.level) for r in snap.rooms],
        "people": [(p.person_id, p.room_id, p.activity.label, p.activity.confirmed, p.position is not None)
                   for p in snap.people],
        "devices": [(d.device_id, d.online, d.manual_override, d.state.model_dump()) for d in snap.devices],
        "routines": [(r.occurrence_id, r.due, r.room_ids) for r in snap.routines],
        "profiles": [p.model_dump() for p in snap.profiles],
        "restrictions": [u.model_dump(mode="json") for u in snap.user_restrictions],
        "influence": snap.room_influence,
    }, sort_keys=True, default=str)


# ---------- 중앙 ----------

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
        self.reset()

    # ----- 상태 -----

    def reset(self, session_id: str | None = None, clock_mode: str | None = None) -> None:
        with self.lock:
            self.session_id, self.clock_mode = session_id, clock_mode
            self.snap: WorldSnapshot | None = None
            self.hist = History()
            self.dialogue = DialogueManager()
            self.protected: dict[str, set[str]] = {}
            self.pending: dict[str, DeviceCommand] = {}
            self.holds: dict[str, datetime] = {}
            self.memo: dict[str, Choice] = {}
            self.last_key: str | None = None
            self.paused = False

    def status(self) -> dict:
        with self.lock:
            return {
                "session_id": self.session_id, "clock_mode": self.clock_mode, "paused": self.paused,
                "world_revision": self.snap.world_revision if self.snap else None,
                "llm": bool(self.llm), "models": {a: self.models.name(a) for a in
                                                 ("cleaning", "noise", "temperature", "lighting")},
                "pending_commands": len(self.pending),
                "episodes": {k: vars(v) for k, v in self.dialogue.episodes.items()},
            }

    # ----- 입력 -----

    def on_snapshot(self, snap: WorldSnapshot) -> Decision | None:
        with self.lock:
            self.hist.observe(snap)
            self.dialogue.update(snap)
            self.snap = snap
            key = decision_key(snap) + json.dumps({k: (v.state, v.answer) for k, v in self.dialogue.episodes.items()})
            if key == self.last_key:
                return None
            self.last_key = key
            return self.decide("snapshot")

    def on_voice_reply(self, reply: VoiceReply) -> Decision | None:
        with self.lock:
            ep = self.dialogue.on_reply(reply, self.clock())
            if ep is None or self.snap is None:
                return None
            self.last_key = None
            return self.decide(f"voice_reply:{ep.answer}")

    def on_voice_command(self, cmd: VoiceCommand) -> Decision | None:
        with self.lock:
            if self.snap is None:
                return None
            snap, parsed = self.snap, parse_command(cmd.text)
            accepted, rejected, speech = [], [], []
            if parsed is None:
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
            commands = self._dispatch(accepted, snap, ["DIRECT_COMMAND"])
            return self._record(snap, f"voice_command:{cmd.text}", [], accepted, [], [], "none",
                                commands, rejected, speech, "직접 명령은 자동 계획보다 우선")

    def on_device_result(self, res: DeviceResult) -> None:
        with self.lock:
            cmd = self.pending.pop(res.command_id, None)
            if cmd is None or res.status != "completed" or self.snap is None:
                return
            if cmd.action == "start_cleaning":
                occ = next((r.occurrence_id for r in self.snap.routines if r.device_id == cmd.device_id and r.due), None)
                for room_id in cmd.params.get("room_ids", []):
                    self.hist.mark_cleaned(room_id, res.timestamp, occ)

    def tick(self) -> list[str]:
        with self.lock:
            return [ep.person_id for ep in self.dialogue.check_deadlines(self.clock())]

    def control(self, action: str) -> None:
        with self.lock:
            if action == "pause":
                self.paused = True
            elif action == "resume":
                self.paused, self.last_key = False, None
            elif action == "reset":
                self.reset(self.session_id, self.clock_mode)

    # ----- 판단 -----

    def decide(self, trigger: str) -> Decision:
        snap = self.snap
        ctx = A.Context(snap, self.hist, self.models, self.dialogue, self.protected)
        proposals = [agent(ctx) for agent in A.AGENTS]

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
                kept.append(a)
            p.actions[:] = kept

        accepted, conflicts = find_conflicts(proposals, snap)
        choices, arbitration, summary = self._arbitrate(snap, proposals, conflicts)
        chosen = {c.conflict_id: c.option_id for c in choices}
        for c in conflicts:
            opt = next(o for o in c.options if o.option_id == chosen[c.conflict_id])
            accepted.extend(opt.actions)

        valid = []
        for a in accepted:
            reason = validate(a, snap)
            if reason:
                rejected.append(Rejected(action=a, reason=reason))
            elif any(p.device_id == a.device_id and p.action == a.action and p.params == a.params
                     for p in self.pending.values()):
                rejected.append(Rejected(action=a, reason="DUPLICATE_PENDING"))
            else:
                valid.append(a)

        if self.paused:
            rejected += [Rejected(action=a, reason="PAUSED") for a in valid]
            valid = []
        commands = self._dispatch(valid, snap, [trigger])

        speech = []
        for p in proposals:
            for q in p.questions:
                if self.paused or self.dialogue.awaiting_in_room(q.room_id):
                    continue
                person = snap.person(q.person_id)
                cmd = self._say(q.room_id, question_text(person.identity), True)
                self.dialogue.asked(q.person_id, cmd.command_id, self.clock(), cmd.text)
                speech.append(cmd)
        return self._record(snap, trigger, proposals, accepted, conflicts, choices, arbitration,
                            commands, rejected, speech, summary)

    def _arbitrate(self, snap, proposals, conflicts):
        if not conflicts:
            return [], "none", ""
        sig = {c.conflict_id: json.dumps(c.model_dump(), sort_keys=True) for c in conflicts}
        if all(sig[c.conflict_id] in self.memo for c in conflicts):
            return [self.memo[sig[c.conflict_id]] for c in conflicts], "cached", "이전과 같은 충돌, 같은 선택 유지"
        choices, arbitration, summary = None, "fallback", "LLM 없이 제한을 지키는 선택"
        if self.llm is not None:
            sel = self.llm.select(summarize(snap, proposals), conflicts)
            if sel is not None and self._valid_selection(sel, conflicts):
                choices, arbitration, summary = sel.choices, "llm", sel.summary
            else:
                summary = "LLM 응답 없음 또는 형식 오류 → 결정적 선택"
        if choices is None:
            choices = [fallback_choice(c) for c in conflicts]
        for c in choices:
            self.memo[sig[c.conflict_id]] = c
        return choices, arbitration, summary

    @staticmethod
    def _valid_selection(sel, conflicts) -> bool:
        want = {c.conflict_id: {o.option_id for o in c.options} for c in conflicts}
        got = {c.conflict_id: c.option_id for c in sel.choices}
        return set(got) == set(want) and all(got[k] in want[k] for k in want)

    def _dispatch(self, actions, snap, reasons) -> list[DeviceCommand]:
        commands = []
        plan_id = str(time.time_ns())
        for a in actions:
            cmd = DeviceCommand(plan_id=plan_id, issued_at=snap.as_of, world_session_id=snap.world_session_id,
                                based_on_revision=snap.world_revision, device_id=a.device_id, action=a.action,
                                params=a.params, reason_codes=[a.reason] + reasons)
            self.pending[cmd.command_id] = cmd
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

    def _record(self, snap, trigger, proposals, accepted, conflicts, choices, arbitration,
                commands, rejected, speech, summary) -> Decision:
        d = Decision(world_session_id=snap.world_session_id, world_revision=snap.world_revision,
                     decided_at=snap.as_of, trigger=trigger, proposals=proposals, accepted=accepted,
                     conflicts=conflicts, choices=choices, arbitration=arbitration, commands=commands,
                     rejected=rejected, speech=speech, summary=summary)
        self.decisions.append(d)
        if self.trace_path:
            self.trace_path.parent.mkdir(parents=True, exist_ok=True)
            with self.trace_path.open("a", encoding="utf-8") as f:
                f.write(d.model_dump_json() + "\n")
        return d
