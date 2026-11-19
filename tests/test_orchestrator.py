"""계약 검증, 에이전트, 충돌·선택, 음성, 시나리오 t0~t7."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from orchestrator.agents import Models
from orchestrator.central import Orchestrator, fallback_choice, find_conflicts, validate
from orchestrator.contract import WorldSnapshot
from orchestrator.demo import run_demo
from orchestrator.fixtures import DemoHome
from orchestrator.schemas import Choice, LlmSelection, ProposedAction, VoiceReply
from datetime import timedelta
from orchestrator.voice import interpret_reply, parse_command

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def commands(decisions, step):
    return [(c.device_id, c.action, c.params) for s, d in decisions if s == step for c in d.commands]


# ---------- 계약 ----------

def test_contract_example_validates():
    snap = WorldSnapshot.model_validate_json((EXAMPLES / "snapshot_t2.json").read_text(encoding="utf-8"))
    assert snap.person("person_a").activity.label == "resting"


def test_contract_rejects_unknown_key_and_bad_reference():
    raw = json.loads((EXAMPLES / "snapshot_t2.json").read_text(encoding="utf-8"))
    with pytest.raises(ValidationError):
        WorldSnapshot.model_validate({**raw, "extra_field": 1})
    raw["people"][0]["room_id"] = "no_such_room"
    with pytest.raises(ValidationError):
        WorldSnapshot.model_validate(raw)


def test_contract_rejects_state_of_wrong_kind():
    raw = json.loads((EXAMPLES / "snapshot_t2.json").read_text(encoding="utf-8"))
    raw["devices"][0]["state"] = {"power": "on"}  # 청소기에 조명 상태
    with pytest.raises(ValidationError):
        WorldSnapshot.model_validate(raw)


# ---------- 시나리오 t0~t7 (HTML 시연과 같은 결과) ----------

@pytest.mark.parametrize("models", [Models(None), Models()], ids=["rules", "trained"])
def test_scenario_matches_demo(models):
    d = run_demo("응.", models=models)
    assert commands(d, "t0") == [("ac_living", "set_power", {"power": "on"}),
                                 ("ac_living", "set_target_temperature", {"target_c": 26.0}),
                                 ("ac_living", "set_airflow_target", {"person_id": "person_b"})]
    t1 = commands(d, "t1")
    assert ("washer_01", "set_mode", {"mode": "low_noise"}) in t1
    assert ("ac_room1", "set_power", {"power": "on"}) in t1
    assert ("vacuum_01", "return_to_dock", {}) in t1             # 방1로 가던 청소기를 도크로
    t1_dec = next(x for s, x in d if s == "t1")
    assert [o.option_id for o in t1_dec.conflicts[0].options] == ["keep_plan", "return_to_dock"]
    assert t1_dec.choices[0].option_id == "return_to_dock"
    for s in ("t2", "t2-reply", "t3", "t4", "t5"):             # 방1에 사람이 있는 동안: 보류 유지, 다시 묻지 않음
        x = next(x for st, x in d if st == s)
        assert not x.conflicts and not any(c[0] == "vacuum_01" for c in commands(d, s))
    t2 = next(x for s, x in d if s == "t2")
    assert len(t2.speech) == 1 and t2.speech[0].expect_reply   # 질문 한 번
    assert sorted(commands(d, "t2-reply"), key=str) == sorted(
        [("light_room1", "set_power", {"power": "off"}),
         ("curtain_room1", "set_curtain_position", {"position_pct": 0})], key=str)
    assert not any(x.speech for s, x in d if s in ("t3", "t4"))  # 다시 묻지 않음
    assert commands(d, "t5-voice") == [("light_room1", "set_power", {"power": "on"})]
    assert commands(d, "t5") == []                              # 커튼은 그대로
    t6 = commands(d, "t6")
    assert ("vacuum_01", "start_cleaning", {"room_ids": ["room1"], "mode": "normal"}) in t6   # 방2는 다시 안 함
    assert ("washer_01", "set_mode", {"mode": "normal"}) in t6
    assert ("ac_room1", "set_power", {"power": "off"}) in t6
    assert commands(d, "t7") == [("ac_living", "set_target_temperature", {"target_c": 25.0}),
                                 ("ac_living", "set_airflow_target", {"person_id": "person_a"})]


@pytest.mark.parametrize("reply", ["아니.", None, "음…"])
def test_no_or_missing_reply_keeps_lights(reply):
    d = run_demo(reply, models=Models(None))
    assert not any(c[0] in ("light_room1", "curtain_room1")
                   for s, x in d if s.startswith("t2") for c in commands(d, s))


# ---------- 충돌과 LLM ----------

class FakeLlm:
    def __init__(self, option_id):
        self.option_id = option_id
        self.calls = 0

    def select(self, context, conflicts):
        self.calls += 1
        return LlmSelection(choices=[Choice(conflict_id=c.conflict_id, option_id=self.option_id, reason="테스트")
                                     for c in conflicts], summary="테스트")


def test_llm_return_choice_is_used_once_and_kept():
    llm = FakeLlm("return_to_dock")
    d = run_demo("응.", llm=llm, models=Models(None))
    t1 = next(x for s, x in d if s == "t1")
    assert t1.arbitration == "llm" and ("vacuum_01", "return_to_dock", {}) in commands(d, "t1")
    assert llm.calls == 1   # 복귀 뒤에는 방1이 빌 때까지 청소 판단을 다시 묻지 않음
    assert sum(c.action == "return_to_dock" for _, x in d for c in x.commands) == 1


def test_llm_keep_plan_sends_no_vacuum_command():
    d = run_demo("응.", llm=FakeLlm("keep_plan"), models=Models(None))
    t1 = next(x for s, x in d if s == "t1")
    assert t1.arbitration == "llm" and not any(c[0] == "vacuum_01" for c in commands(d, "t1"))


def test_invalid_llm_choice_falls_back():
    d = run_demo("응.", llm=FakeLlm("no_such_option"), models=Models(None))
    t1 = next(x for s, x in d if s == "t1")
    assert t1.arbitration == "fallback" and t1.choices[0].option_id == "return_to_dock"


def test_value_conflict_and_fallback():
    home = DemoHome()
    snap = home.snapshot("t1")
    from orchestrator.schemas import AgentProposal
    a = AgentProposal(agent="temperature", world_session_id="s", world_revision=1,
                      actions=[ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 24})])
    b = AgentProposal(agent="noise", world_session_id="s", world_revision=1,
                      actions=[ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 26})])
    accepted, conflicts, _ = find_conflicts([a, b], snap)
    assert not accepted and len(conflicts) == 1 and len(conflicts[0].options) == 3
    assert fallback_choice(conflicts[0]).option_id == "hold"


# ---------- 검증 ----------

def test_validator_rejects_unsupported_and_out_of_range():
    snap = DemoHome().snapshot("t1")
    assert validate(ProposedAction(device_id="vacuum_01", action="set_mode", params={"mode": "turbo"}), snap)
    assert validate(ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 40}), snap)
    assert validate(ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 24.5}), snap)
    assert validate(ProposedAction(device_id="washer_01", action="set_power", params={"power": "off"}), snap)
    assert validate(ProposedAction(device_id="ac_living", action="set_airflow_target",
                                   params={"person_id": "person_a"}), snap)  # A는 거실에 없음
    assert validate(ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 24}), snap) is None


def test_paused_orchestrator_sends_nothing():
    home = DemoHome()
    sent = []
    orc = Orchestrator(sent.append, lambda r, c: None)
    orc.reset("s", "replay")
    orc.control("pause")
    d = orc.on_snapshot(home.snapshot("t0"))
    assert not sent and all(r.reason == "PAUSED" for r in d.rejected) and d.rejected


# ---------- 음성 ----------

@pytest.mark.parametrize("text,conf,want", [
    ("응.", 0.48, "yes"), ("네", 0.55, "yes"), ("응?", 0.29, "yes"), ("그래", 0.5, "yes"),
    ("아니.", 0.79, "no"), ("아니요 안 잘 거예요", 0.7, "no"), ("뭐라고?", 0.6, "ambiguous"),
    ("응", 0.1, "ambiguous"), (None, None, "ambiguous"),
])
def test_interpret_reply(text, conf, want):
    assert interpret_reply(text, conf) == want


def test_parse_command():
    assert parse_command("불 켜줘")[:3] == ("light", "set_power", {"power": "on"})
    assert parse_command("커튼 닫아 줘")[2] == {"position_pct": 0}
    assert parse_command("오늘 날씨 어때") is None


def test_reply_to_unknown_question_is_ignored():
    home = DemoHome()
    orc = Orchestrator(lambda c: None, lambda r, c: None)
    orc.reset("s", "replay")
    orc.on_snapshot(home.snapshot("t2"))
    assert orc.on_voice_reply(VoiceReply(node_id="n", room_id="room1", command_id="other", text="응",
                                         confidence=0.9, no_response=False)) is None


# ---------- 2026-11-18 점검에서 찾은 문제의 회귀 테스트 ----------

from orchestrator.central import decision_key
from orchestrator.fixtures import SESSION
from orchestrator.schemas import DeviceResult, VoiceCommand


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def make(llm=None):
    sent, said, clock = [], [], Clock()
    orc = Orchestrator(sent.append, lambda r, c: said.append((r, c)), llm=llm, models=Models(None), clock=clock)
    orc.reset(SESSION, "replay")
    return orc, sent, said, clock


def modified(snap, change):
    raw = json.loads(snap.model_dump_json())
    change(raw)
    return WorldSnapshot.model_validate(raw)


def result(cmd, status, **kw):
    return DeviceResult(world_session_id=kw.get("session", cmd.world_session_id), command_id=cmd.command_id,
                        device_id=kw.get("device", cmd.device_id), status=status, timestamp=cmd.issued_at)


def test_accepted_then_completed_records_cleaning():
    orc, sent, clock, s6, cmd = start_cleaning_at_t6()
    orc.on_device_result(result(cmd, "accepted"))
    assert cmd.command_id in orc.pending
    orc.on_device_result(result(cmd, "completed"))
    assert "room1" in orc.hist.last_cleaned_at and cmd.command_id not in orc.pending


def test_result_with_wrong_device_or_session_is_ignored():
    orc, sent, clock, s6, cmd = start_cleaning_at_t6()
    orc.on_device_result(result(cmd, "completed", device="washer_01"))
    orc.on_device_result(result(cmd, "completed", session="other"))
    assert "room1" not in orc.hist.last_cleaned_at


def test_pending_command_expires_and_can_be_resent():
    home = DemoHome()
    orc, sent, _, clock = make()
    orc.on_snapshot(home.snapshot("t1"))          # 결과가 오지 않음 (send는 기록만)
    n = len(sent)
    clock.t += 61
    orc.on_snapshot(modified(home.snapshot("t1"), lambda r: r["rooms"][0]["light"].update(level="DIM")))
    assert any(c.device_id == "washer_01" for c in sent[n:])


def test_plan_id_is_shared_by_decision_and_commands():
    orc, _, _, _ = make()
    d = orc.on_snapshot(DemoHome().snapshot("t0"))
    assert d.commands and {c.plan_id for c in d.commands} == {d.plan_id}


def test_older_revision_is_ignored():
    home = DemoHome()
    orc, _, _, _ = make()
    old, new = home.snapshot("t1"), home.snapshot("t6")
    orc.on_snapshot(new)
    assert orc.on_snapshot(old) is None and orc.snap.world_revision == new.world_revision


def test_decision_key_sees_temperature_freshness_and_deadline():
    s = DemoHome().snapshot("t2")
    hot = modified(s, lambda r: r["rooms"][0]["temperature_c"].update(
        value=35.0, evidence={"observed_at": r["as_of"], "source": "device_sensor", "node_ids": [], "freshness": "current"}))
    stale = modified(s, lambda r: r["rooms"][0]["occupancy"]["evidence"].update(freshness="stale"))
    soon = modified(s, lambda r: r["routines"][0].update(deadline_at="2026-11-13T21:20:00+09:00"))
    assert len({decision_key(x) for x in (s, hot, stale, soon)}) == 4


def test_cached_choice_is_not_reused_when_deadline_nears():
    llm = FakeLlm("keep_plan")
    home = DemoHome()
    orc, _, _, _ = make(llm)
    orc.on_snapshot(home.snapshot("t1"))
    late = modified(home.snapshot("t1"), lambda r: r.update(as_of="2026-11-13T22:50:00+09:00"))
    d = orc.on_snapshot(late)
    assert llm.calls == 2 and d.arbitration == "llm"


def test_stale_device_state_is_not_commanded():
    s = modified(DemoHome().snapshot("t1"), lambda r: next(
        d for d in r["devices"] if d["device_id"] == "washer_01")["evidence"].update(freshness="stale"))
    assert validate(ProposedAction(device_id="washer_01", action="set_mode", params={"mode": "low_noise"}), s) \
        == "DEVICE_STATE_STALE"


def test_reply_from_other_room_or_node_is_ignored():
    orc, _, said, _ = make()
    orc.on_snapshot(DemoHome().snapshot("t2"))
    q = said[-1][1]
    for node, room in (("node_room1", "living"), ("intruder", "room1")):
        assert orc.on_voice_reply(VoiceReply(node_id=node, room_id=room, command_id=q.command_id, text="응",
                                             confidence=0.9, no_response=False)) is None
    d = orc.on_voice_reply(VoiceReply(node_id="node_room1", room_id="room1", command_id=q.command_id, text="응",
                                      confidence=0.9, no_response=False))
    assert d is not None and d.commands


def test_voice_command_from_unregistered_node_is_ignored_but_works_while_paused():
    orc, _, _, _ = make()
    orc.on_snapshot(DemoHome().snapshot("t4"))
    orc.control("pause")
    assert orc.on_voice_command(VoiceCommand(node_id="intruder", room_id="room1", text="불 켜줘", confidence=0.9)) is None
    d = orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text="불 켜줘", confidence=0.9))
    assert [(c.device_id, c.params) for c in d.commands] == [("light_room1", {"power": "on"})]


def test_yes_while_paused_is_applied_after_resume():
    home = DemoHome()
    orc, sent, said, _ = make()
    orc.on_snapshot(home.snapshot("t2"))
    q = said[-1][1]
    orc.control("pause")
    orc.on_voice_reply(VoiceReply(node_id="node_room1", room_id="room1", command_id=q.command_id, text="응",
                                  confidence=0.9, no_response=False))
    n = len(sent)
    orc.control("resume")
    orc.on_snapshot(home.snapshot("t3"))
    assert {c.device_id for c in sent[n:]} >= {"light_room1", "curtain_room1"}


@pytest.mark.parametrize("change", [
    lambda r: r["rooms"][0]["occupancy"].update(state="EMPTY"),                       # EMPTY 방에 사람
    lambda r: r["people"][1].update(identity="UNKNOWN", identity_source="UNKNOWN"),   # UNKNOWN에 프로필
    lambda r: r["devices"][0]["evidence"].update(node_ids=["ghost"]),                 # 없는 Node
    lambda r: r["people"][1]["position"]["evidence"].update(node_ids=["ghost"]),
], ids=["empty_room_person", "unknown_with_profile", "device_ghost_node", "position_ghost_node"])
def test_contract_semantic_checks(change):
    with pytest.raises(ValidationError):
        modified(DemoHome().snapshot("t2"), change)


def test_inconsistent_voice_reply_is_rejected():
    with pytest.raises(ValidationError):
        VoiceReply(node_id="n", room_id="r", command_id="c", text="응", confidence=None, no_response=False)
    with pytest.raises(ValidationError):
        VoiceReply(node_id="n", room_id="r", command_id="c", text="응", confidence=1.5, no_response=False)


def test_broken_model_file_falls_back_to_rules(tmp_path):
    (tmp_path / "noise.joblib").write_bytes(b"broken")
    assert Models(tmp_path).name("noise") == "rule"


# ---------- 2차 점검: 진행 중 작업, LLM 대기 중 상태 변화, 회차, 재시도, 신선도 ----------

class HookLlm:
    """select가 불리는 순간(= LLM을 기다리는 동안)에 hook을 실행한다."""

    def __init__(self, hook=None, option="return_to_dock"):
        self.hook, self.option, self.calls, self.orc = hook, option, 0, None

    def select(self, context, conflicts):
        self.calls += 1
        if self.hook:
            self.hook(self.orc)
        return LlmSelection(choices=[Choice(conflict_id=c.conflict_id, option_id=self.option, reason="r")
                                     for c in conflicts], summary="s")


def make_hooked(hook):
    llm = HookLlm(hook)
    orc, sent, said, clock = make(llm)
    llm.orc = orc
    return orc, sent, said, clock, llm


def play_until(step, send=None):
    """t0부터 step까지 재생. 명령은 sent에 기록하고 DemoHome에도 반영한다(결과는 보내지 않음)."""
    home = DemoHome()
    orc, sent, said, clock = make()
    orc.send_command = send or (lambda c: (sent.append(c), home.send(c)))
    snaps = {}
    for s in ("t0", "t1", "t2", "t3", "t4", "t5", "t6"):
        snaps[s] = home.snapshot(s)
        orc.on_snapshot(snaps[s])
        if s == step:
            break
    return orc, sent, clock, snaps, home


def start_cleaning_at_t6():
    orc, sent, clock, snaps, _ = play_until("t6")
    return orc, sent, clock, snaps["t6"], next(c for c in sent if c.action == "start_cleaning")


def test_accepted_cleaning_is_not_expired_by_time():
    orc, sent, clock, _, cmd = start_cleaning_at_t6()
    orc.on_device_result(result(cmd, "accepted"))
    clock.t += 3600
    orc.tick()
    assert orc.pending[cmd.command_id].status == "accepted"
    orc.on_device_result(result(cmd, "completed"))
    assert "room1" in orc.hist.last_cleaned_at


def test_unacknowledged_cleaning_is_not_resent_and_is_confirmed_by_state():
    orc, sent, clock, s6, cmd = start_cleaning_at_t6()
    clock.t += 31
    orc.tick()
    assert orc.pending[cmd.command_id].status == "unknown"
    n = len(sent)
    orc.on_snapshot(modified(s6, lambda r: (r.update(world_revision=r["world_revision"] + 1),
                                             r["rooms"][0]["light"].update(level="DIM"))))
    assert not any(c.action == "start_cleaning" for c in sent[n:])        # 다시 보내지 않음
    cleaning = modified(s6, lambda r: (r.update(world_revision=r["world_revision"] + 2), next(
        d for d in r["devices"] if d["device_id"] == "vacuum_01")["state"].update(status="cleaning", target_room_ids=["room1"])))
    orc.on_snapshot(cleaning)
    assert orc.pending[cmd.command_id].status == "in_progress"


def test_completion_is_recorded_on_the_original_occurrence():
    orc, sent, clock, s6, cmd = start_cleaning_at_t6()
    orc.on_snapshot(modified(s6, lambda r: (r.update(world_revision=r["world_revision"] + 5),
                                             r["routines"][0].update(occurrence_id="next_occurrence"))))
    orc.on_device_result(result(cmd, "completed"))
    assert set(orc.hist.done_rooms) == {"daily_clean_20261113_evening"}
    assert "room1" in orc.hist.done_rooms["daily_clean_20261113_evening"]


def test_direct_command_during_llm_wait_is_not_overwritten():
    def hook(orc):
        orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text="에어컨 꺼줘", confidence=0.9))
    orc, sent, *_ = make_hooked(hook)
    orc.on_snapshot(DemoHome().snapshot("t1"))
    powers = [c.params["power"] for c in sent if c.device_id == "ac_room1" and c.action == "set_power"]
    assert powers == ["off"]


def test_pause_during_llm_wait_blocks_plan_and_resume_decides_immediately():
    orc, sent, *_ = make_hooked(lambda o: o.control("pause"))
    orc.on_snapshot(DemoHome().snapshot("t1"))
    assert sent == []
    orc.llm.hook = None
    d = orc.control("resume")
    assert d is not None and d.commands


def test_resume_waits_for_fresh_state_when_snapshot_is_old():
    orc, sent, said, clock = make()
    orc.control("pause")
    orc.on_snapshot(DemoHome().snapshot("t0"))
    clock.t += 121
    assert orc.control("resume") is None and sent == []


def test_reset_during_llm_wait_discards_the_plan():
    orc, sent, *_ = make_hooked(lambda o: o.reset(SESSION, "replay"))
    assert orc.on_snapshot(DemoHome().snapshot("t1")) is None
    assert sent == []


def test_session_change_during_llm_wait_discards_the_plan():
    other = modified(DemoHome().snapshot("t0"), lambda r: r.update(world_session_id="other-session"))
    orc, sent, *_ = make_hooked(lambda o: (setattr(o.llm, "hook", None), o.on_snapshot(other)))
    orc.on_snapshot(DemoHome().snapshot("t1"))
    assert all(c.world_session_id == "other-session" for c in sent)


def test_transient_failure_retries_with_limit_and_rejection_does_not():
    orc, sent, clock, s6, cmd = start_cleaning_at_t6()

    def again(minutes, rev):
        snap = modified(s6, lambda r: (r.update(world_revision=r["world_revision"] + rev),
                                       r.update(as_of=(s6.as_of + timedelta(minutes=minutes)).isoformat()),
                                       r["rooms"][0]["light"].update(level="DIM" if rev % 2 else "BRIGHT")))
        n = len(sent)
        orc.on_snapshot(snap)
        return [c for c in sent[n:] if c.action == "start_cleaning"]

    failed = DeviceResult(world_session_id=cmd.world_session_id, command_id=cmd.command_id, device_id=cmd.device_id,
                          status="failed", timestamp=cmd.issued_at, reason="busy")
    orc.on_device_result(failed)
    assert again(1, 1) == []                    # 간격 전에는 다시 보내지 않음
    retry = again(6, 2)
    assert len(retry) == 1                      # 5분 뒤 한 번 재시도
    orc.on_device_result(DeviceResult(world_session_id=cmd.world_session_id, command_id=retry[0].command_id,
                                      device_id=cmd.device_id, status="rejected", timestamp=cmd.issued_at))
    assert again(30, 3) == []                   # 거절은 자동 반복하지 않음


def test_cached_choice_is_dropped_when_preference_changes():
    llm = FakeLlm("keep_plan")
    home = DemoHome()
    orc, _, _, _ = make(llm)
    s1 = home.snapshot("t1")
    orc.on_snapshot(s1)
    orc.on_snapshot(modified(s1, lambda r: (r.update(world_revision=r["world_revision"] + 1),
                                             r["profiles"][0].update(quiet_rest=False))))
    assert llm.calls == 2


def test_direct_command_is_refused_when_state_is_old():
    orc, sent, said, clock = make()
    orc.on_snapshot(DemoHome().snapshot("t4"))
    clock.t += 121
    d = orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text="불 켜줘", confidence=0.9))
    assert d.commands == [] and d.rejected[0].reason == "SNAPSHOT_STALE"


# ---------- 3차 점검: 음성 해석, 음성 메시지 검사, 이동 후 질문 무효화, 재시도 계기, MQTT 큐 ----------

from orchestrator.voice import command_problem


@pytest.mark.parametrize("text,want", [
    ("그래서 뭐라고?", "ambiguous"), ("좋아하는 노래 틀어줘", "ambiguous"), ("어?", "ambiguous"),
    ("응 아니", "ambiguous"), ("네 아니요", "ambiguous"), ("응 잘게", "yes"), ("네네", "yes"),
    ("그래도 안 잘래", "no"),
])
def test_reply_interpretation_is_strict(text, want):
    assert interpret_reply(text, 0.6) == want


@pytest.mark.parametrize("text", ["불 켜지 마", "불 끄지 마", "에어컨 끄지 말아줘", "불 켜야 하나?", "불 켰어?"])
def test_negated_or_question_commands_are_not_parsed(text):
    assert parse_command(text) is None and command_problem(text) in ("NEGATED", "QUESTION")


def test_command_with_question_mark_but_imperative_still_works():
    assert parse_command("불 켜줘?")[:3] == ("light", "set_power", {"power": "on"})


def test_negated_command_speaks_back_without_action():
    orc, sent, said, _ = make()
    orc.on_snapshot(DemoHome().snapshot("t4"))
    n = len(sent)
    d = orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text="불 켜지 마", confidence=0.9))
    assert d.commands == [] and len(sent) == n and "실행하지 않았어요" in d.speech[0].text


def test_low_confidence_and_duplicate_commands_are_not_executed():
    orc, sent, _, _ = make()
    orc.on_snapshot(DemoHome().snapshot("t4"))
    low = orc.on_voice_command(VoiceCommand(event_id="e0", node_id="node_room1", room_id="room1", text="불 켜줘",
                                            confidence=0.1))
    assert low.commands == []
    for _ in range(2):
        orc.on_voice_command(VoiceCommand(event_id="e1", node_id="node_room1", room_id="room1", text="불 켜줘",
                                          confidence=0.9))
    assert len([c for c in sent if c.device_id == "light_room1"]) == 1


def test_old_live_voice_message_is_ignored():
    from datetime import datetime, timezone
    orc, sent, _, _ = make()
    orc.reset(SESSION, "live")
    orc.on_snapshot(DemoHome().snapshot("t4"))
    old = datetime.now(timezone.utc) - timedelta(seconds=60)
    assert orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text="불 켜줘", confidence=0.9,
                                             timestamp=old)) is None


def test_question_is_void_after_person_is_seen_in_another_room():
    orc, sent, said, _ = make()
    s2 = DemoHome().snapshot("t2")
    orc.on_snapshot(s2)
    q = said[-1][1]

    def move(r):
        r["world_revision"] += 1
        a = next(p for p in r["people"] if p["person_id"] == "person_a")
        a.update(room_id="living", track_refs=[])
        c = json.loads(json.dumps(a))
        c.update(person_id="person_c", profile_id=None, identity="UNKNOWN", identity_source="UNKNOWN", room_id="room1")
        r["people"].append(c)
    orc.on_snapshot(modified(s2, move))
    assert orc.on_voice_reply(VoiceReply(node_id="node_room1", room_id="room1", command_id=q.command_id, text="응",
                                         confidence=0.9, no_response=False)) is None


def test_late_reply_within_queue_allowance_is_accepted():
    orc, sent, said, clock = make()
    orc.on_snapshot(DemoHome().snapshot("t2"))
    q = said[-1][1]
    clock.t += 15          # Node가 질문을 줄 세웠다가 늦게 읽은 경우
    orc.tick()
    d = orc.on_voice_reply(VoiceReply(node_id="node_room1", room_id="room1", command_id=q.command_id, text="응",
                                      confidence=0.9, no_response=False))
    assert d is not None and d.commands


def test_retry_happens_when_interval_passes_even_if_state_is_unchanged():
    orc, sent, clock, s6, cmd = start_cleaning_at_t6()
    orc.on_device_result(DeviceResult(world_session_id=cmd.world_session_id, command_id=cmd.command_id,
                                      device_id=cmd.device_id, status="failed", timestamp=cmd.issued_at, reason="busy"))
    n = len(sent)
    same = modified(s6, lambda r: (r.update(world_revision=r["world_revision"] + 1),
                                   r.update(as_of=(s6.as_of + timedelta(minutes=6)).isoformat())))
    orc.on_snapshot(same)
    assert any(c.action == "start_cleaning" for c in sent[n:])


def mqtt_payload(model_cls, **kw):
    from datetime import datetime, timezone
    import uuid as _uuid
    base = dict(node_id="node_room1", room_id="room1", event_id=str(_uuid.uuid4()),
                timestamp=datetime.now(timezone.utc))
    return model_cls(**{**base, **kw}).model_dump_json().encode()


def test_mqtt_bridge_queues_messages_and_checks_topic_room_and_fields():
    from orchestrator.server import MqttBridge, parse_voice_message

    payload = mqtt_payload(VoiceReply, command_id="c", text="응", confidence=0.9, no_response=False)
    with pytest.raises(ValueError):
        parse_voice_message("eyear/v0/living/voice/reply", payload)          # 방 불일치
    no_id = VoiceReply(node_id="node_room1", room_id="room1", command_id="c", text="응", confidence=0.9,
                       no_response=False).model_dump_json().encode()
    with pytest.raises(ValueError):
        parse_voice_message("eyear/v0/room1/voice/reply", no_id)            # event_id·timestamp 없음
    bridge = MqttBridge("localhost", 1883)       # 연결하지 않음
    bridge._on_message(None, None, type("Msg", (), {"topic": "eyear/v0/room1/voice/reply", "payload": payload}))
    msg = bridge.queues["reply"].get_nowait()
    assert msg.room_id == "room1"


# ---------- 4차 점검: 문장 전체 단일 명령, LLM 대기 중 MQTT 직접 명령, 발행 실패 ----------

@pytest.mark.parametrize("text", ["불 켜는 방법 알려줘", "커튼 열지 않아도 돼", "불 끄고 켜줘",
                                  "불은 켜두고 에어컨 꺼줘", "불 켜면 안 돼"])
def test_only_whole_single_commands_are_executed(text):
    assert parse_command(text) is None
    orc, sent, said, _ = make()
    orc.on_snapshot(DemoHome().snapshot("t4"))
    n = len(sent)
    d = orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text=text, confidence=0.9))
    assert len(sent) == n and d.commands == [] and "다시 말씀해 주세요" in d.speech[0].text


@pytest.mark.parametrize("text,want", [("불 꺼 줘", {"power": "off"}), ("에어컨 켜 주세요", {"power": "on"}),
                                       ("불 켜줘?", {"power": "on"}), ("커튼 닫아 줘", {"position_pct": 0})])
def test_supported_single_commands_still_work(text, want):
    assert parse_command(text)[2] == want


def test_mqtt_direct_commands_run_while_llm_waits_and_are_not_overwritten():
    import threading as th
    from orchestrator.server import MqttBridge

    release, handled = th.Event(), []

    class BlockingLlm:
        def select(self, context, conflicts):
            release.wait(5)
            return LlmSelection(choices=[Choice(conflict_id=c.conflict_id, option_id="return_to_dock", reason="r")
                                         for c in conflicts], summary="s")

    orc, sent, said, _ = make(BlockingLlm())
    bridge = MqttBridge("localhost", 1883)
    bridge.orc = orc
    original = orc.on_voice_command

    def tracked(msg):
        d = original(msg)
        handled.append(msg.text)
        if len(handled) == 2:
            release.set()          # 두 직접 명령이 처리된 뒤에야 LLM이 돌아온다
        return d
    orc.on_voice_command = tracked
    bridge.start_workers()
    worker = th.Thread(target=orc.on_snapshot, args=(DemoHome().snapshot("t1"),))
    worker.start()                 # t1: 청소 충돌 → LLM 대기
    for text in ("에어컨 꺼줘", "불 꺼 줘"):
        bridge._on_message(None, None, type("Msg", (), {"topic": "eyear/v0/room1/voice/command",
                                                        "payload": mqtt_payload(VoiceCommand, text=text,
                                                                                confidence=0.9)}))
    worker.join(10)
    bridge._stop.set()
    assert release.is_set() and handled == ["에어컨 꺼줘", "불 꺼 줘"]      # LLM 대기 중 처리, 도착 순서 유지
    ac = [c.params["power"] for c in sent if c.device_id == "ac_room1" and c.action == "set_power"]
    assert ac == ["off"]           # 돌아온 자동 계획이 직접 명령(끄기)을 덮어쓰지 않음


def later(snap, rev_add, seconds):
    return modified(snap, lambda r: (r.update(world_revision=r["world_revision"] + rev_add),
                                     r.update(as_of=(snap.as_of + timedelta(seconds=seconds)).isoformat())))


def test_rejected_publish_is_recorded_and_question_not_awaited():
    orc, sent, said, _ = make()
    orc.speak = lambda room, cmd: False            # MQTT가 발행을 즉시 거절
    orc.send_command = lambda cmd: False
    d = orc.on_snapshot(DemoHome().snapshot("t2"))
    assert d.speech == [] and d.undelivered and d.undelivered[0].startswith("speak room1")
    assert orc.dialogue.episode("person_a").state == "NOT_ASKED"
    assert d.commands == [] and any(r.reason == "PUBLISH_FAILED" for r in d.rejected) and not orc.pending


# ---------- 5차 점검: 발행이 거절된 뒤 남는 상태 ----------

def test_refused_direct_command_does_not_leave_a_hold():
    orc, sent, _, _ = make()
    orc.on_snapshot(DemoHome().snapshot("t4"))
    orc.send_command = lambda cmd: False
    d = orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1", text="에어컨 꺼줘", confidence=0.9))
    assert d.commands == [] and "ac_room1" not in orc.holds and "보내지 못했어요" in d.speech[-1].text


def test_partly_refused_sleep_actions_retry_only_the_failed_one():
    home = DemoHome()
    orc, sent, said, _ = make()
    s2 = home.snapshot("t2")
    orc.on_snapshot(s2)
    q = said[-1][1]
    orc.send_command = lambda c: False if c.device_id == "curtain_room1" else sent.append(c)
    d = orc.on_voice_reply(VoiceReply(node_id="node_room1", room_id="room1", command_id=q.command_id, text="응",
                                      confidence=0.9, no_response=False))
    assert [c.device_id for c in d.commands] == ["light_room1"]
    assert orc.dialogue.episode("person_a").consumed is False
    orc.send_command = sent.append
    n = len(sent)
    orc.on_snapshot(later(s2, 1, 11))           # 재시도 간격(10초) 뒤 다음 스냅샷
    assert [c.device_id for c in sent[n:] if c.device_id in ("light_room1", "curtain_room1")] == ["curtain_room1"]
    assert orc.dialogue.episode("person_a").consumed is True


def test_refused_question_is_asked_again_then_given_up():
    orc, sent, said, _ = make()
    orc.speak = lambda r, c: False
    s2 = DemoHome().snapshot("t2")
    orc.on_snapshot(s2)
    orc.on_snapshot(later(s2, 1, 1))            # 같은 상태, revision만 증가 → 다시 시도
    orc.on_snapshot(later(s2, 2, 2))
    assert orc.dialogue.episode("person_a").ask_failures == 3
    assert orc.dialogue.episode("person_a").state == "EXPIRED"      # 3번 거절되면 이번 휴식에는 더 묻지 않음
    orc.speak = lambda r, c: said.append((r, c))
    orc.on_snapshot(later(s2, 3, 3))
    assert said == []

    orc2, _, said2, _ = make()
    orc2.speak = lambda r, c: False
    orc2.on_snapshot(s2)
    orc2.speak = lambda r, c: said2.append((r, c))
    d = orc2.on_snapshot(later(s2, 1, 1))
    assert d.speech and orc2.dialogue.episode("person_a").state == "AWAITING_REPLY"


def test_mqtt_reply_worker_waiting_for_llm_does_not_delay_direct_command():
    import threading as th
    from orchestrator.server import MqttBridge

    entered, release, order = th.Event(), th.Event(), []

    class BlockingLlm:
        def select(self, context, conflicts):
            entered.set()
            release.wait(5)
            order.append("llm returned")
            return LlmSelection(choices=[Choice(conflict_id=c.conflict_id, option_id="keep_plan", reason="r")
                                         for c in conflicts], summary="s")

    orc, sent, said, _ = make(FakeLlm("keep_plan"))     # 청소기는 방1로 계속 감 → 충돌이 남음
    orc.on_snapshot(DemoHome().snapshot("t2"))          # 질문 발행
    q = said[-1][1]
    orc.llm, orc.memo = BlockingLlm(), {}               # 대답 판단에서 LLM을 부르게 함
    original = orc.on_voice_command

    def tracked(msg):
        d = original(msg)
        order.append("command")
        release.set()
        return d
    orc.on_voice_command = tracked
    bridge = MqttBridge("localhost", 1883)
    bridge.orc = orc
    bridge.start_workers()
    reply = mqtt_payload(VoiceReply, command_id=q.command_id, text="응", confidence=0.9, no_response=False)
    bridge._on_message(None, None, type("Msg", (), {"topic": "eyear/v0/room1/voice/reply", "payload": reply}))
    assert entered.wait(5)                               # 대답 워커가 실제로 LLM 안에 들어감
    cmd = mqtt_payload(VoiceCommand, text="에어컨 꺼줘", confidence=0.9)
    bridge._on_message(None, None, type("Msg", (), {"topic": "eyear/v0/room1/voice/command", "payload": cmd}))
    for _ in range(50):
        if "llm returned" in order:
            break
        th.Event().wait(0.1)
    bridge._stop.set()
    assert order[:2] == ["command", "llm returned"]


# ---------- 발표 시나리오: 진행 중 청소 계획 유지 vs 도크 복귀, 같은 회차 보류 ----------

def vacuum_cmds(sent, n=0):
    return [(c.action, c.params) for c in sent[n:] if c.device_id == "vacuum_01"]


def test_t1_is_decided_with_room1_as_next_destination_after_room2():
    home = DemoHome()
    home.snapshot("t0")
    vac = home.snapshot("t1").device("vacuum_01").state
    assert vac.status == "cleaning" and vac.target_room_ids == ["room1"]


@pytest.mark.parametrize("occupancy", ["OCCUPIED", "UNKNOWN", "EMPTY_STALE"])
def test_hold_is_kept_without_asking_llm_while_room1_is_not_confirmed_empty(occupancy):
    llm = FakeLlm("return_to_dock")
    home = DemoHome()
    orc, sent, _, _ = make(llm)
    orc.send_command = lambda c: (sent.append(c), home.send(c))
    orc.on_snapshot(home.snapshot("t0"))
    orc.on_snapshot(home.snapshot("t1"))
    assert vacuum_cmds(sent) == [("return_to_dock", {})] and llm.calls == 1
    s6 = home.snapshot("t6")                                   # 청소기는 도크, A는 나간 상태를 바꿔 본다

    def change(r):
        occ = r["rooms"][0]["occupancy"]
        if occupancy == "EMPTY_STALE":
            occ["evidence"]["freshness"] = "stale"
        else:
            occ.update(state=occupancy, confidence=None if occupancy == "UNKNOWN" else 0.9)
            if occupancy == "UNKNOWN":
                occ["evidence"] = {"observed_at": None, "source": "unknown", "node_ids": [], "freshness": "unavailable"}
    d = orc.on_snapshot(modified(s6, change))
    assert vacuum_cmds(sent) == [("return_to_dock", {})] and llm.calls == 1
    assert any(r.reason.startswith("WAIT_UNTIL_EMPTY:room1") for r in d.rejected) and not d.conflicts


def test_return_is_not_asked_again_before_vacuum_state_updates():
    llm = FakeLlm("return_to_dock")
    home = DemoHome()
    orc, sent, _, _ = make(llm)
    orc.on_snapshot(home.snapshot("t0"))
    s1 = home.snapshot("t1")
    orc.on_snapshot(s1)
    orc.on_device_result(result(next(c for c in sent if c.action == "return_to_dock"), "completed"))
    d = orc.on_snapshot(later(home.snapshot("t2"), 0, 0))     # A가 쉬기 시작, 청소기 상태는 아직 갱신 안 됨
    assert llm.calls == 1 and not d.conflicts
    assert vacuum_cmds(sent) == [("return_to_dock", {})]


def test_room1_only_restarts_and_room2_is_never_added_even_after_failed_publish():
    home = DemoHome()
    orc, sent, clock, snaps, _ = play_until("t5")
    occ = "daily_clean_20261113_evening"
    assert orc.hist.backlog[occ] == ["room1"] and "room2" in orc.hist.done_rooms[occ]
    refused = []
    orc.send_command = lambda c: refused.append(c) or False if c.action == "start_cleaning" else sent.append(c)
    s6 = DemoHome().snapshot("t6")
    s6 = modified(snaps["t5"], lambda r: (r.update(world_revision=r["world_revision"] + 1, as_of=s6.as_of.isoformat()),
                                          r.update(rooms=json.loads(s6.model_dump_json())["rooms"],
                                                   people=json.loads(s6.model_dump_json())["people"])))
    d = orc.on_snapshot(s6)
    assert [c.params["room_ids"] for c in refused] == [["room1"]]
    assert any(r.reason == "PUBLISH_FAILED" for r in d.rejected) and orc.hist.backlog[occ] == ["room1"]
    orc.send_command = sent.append
    n = len(sent)
    orc.on_snapshot(later(s6, 1, 11))                           # 재시도 간격 뒤
    retry = [c for c in sent[n:] if c.action == "start_cleaning"]
    assert [c.params["room_ids"] for c in retry] == [["room1"]]
    orc.on_device_result(result(retry[0], "accepted"))
    orc.on_snapshot(later(s6, 2, 30))
    assert orc.hist.backlog[occ] == ["room1"]                   # 완료 결과 전까지 남은 작업 보존
    orc.on_device_result(result(retry[0], "completed"))
    n = len(sent)
    orc.on_snapshot(later(s6, 3, 60))
    assert not any(c.action == "start_cleaning" for c in sent[n:])   # 회차 끝, 방2를 다시 시작하지 않음
