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
    assert not any(c[0] == "vacuum_01" for c in t1)          # 사람이 있는 방1 청소는 보류
    t1_dec = next(x for s, x in d if s == "t1")
    assert t1_dec.conflicts and t1_dec.choices[0].option_id == "hold"
    t2 = next(x for s, x in d if s == "t2")
    assert len(t2.speech) == 1 and t2.speech[0].expect_reply   # 질문 한 번
    assert sorted(commands(d, "t2-reply"), key=str) == sorted(
        [("light_room1", "set_power", {"power": "off"}),
         ("curtain_room1", "set_curtain_position", {"position_pct": 0})], key=str)
    assert not any(x.speech for s, x in d if s in ("t3", "t4"))  # 다시 묻지 않음
    assert commands(d, "t5-voice") == [("light_room1", "set_power", {"power": "on"})]
    assert commands(d, "t5") == []                              # 커튼은 그대로
    t6 = commands(d, "t6")
    assert ("vacuum_01", "start_cleaning", {"room_ids": ["room1"], "mode": "normal"}) in t6
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


def test_llm_choice_is_used_and_cached():
    llm = FakeLlm("quiet")
    d = run_demo("응.", llm=llm, models=Models(None))
    t1 = next(x for s, x in d if s == "t1")
    assert t1.arbitration == "llm"
    assert ("vacuum_01", "start_cleaning", {"room_ids": ["room1"], "mode": "quiet"}) in commands(d, "t1")
    assert llm.calls <= 3   # 같은 충돌은 다시 묻지 않음


def test_invalid_llm_choice_falls_back():
    d = run_demo("응.", llm=FakeLlm("no_such_option"), models=Models(None))
    t1 = next(x for s, x in d if s == "t1")
    assert t1.arbitration == "fallback" and t1.choices[0].option_id == "hold"


def test_value_conflict_and_fallback():
    home = DemoHome()
    snap = home.snapshot("t1")
    from orchestrator.schemas import AgentProposal
    a = AgentProposal(agent="temperature", world_session_id="s", world_revision=1,
                      actions=[ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 24})])
    b = AgentProposal(agent="noise", world_session_id="s", world_revision=1,
                      actions=[ProposedAction(device_id="ac_room1", action="set_target_temperature", params={"target_c": 26})])
    accepted, conflicts = find_conflicts([a, b], snap)
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
