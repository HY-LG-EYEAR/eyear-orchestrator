"""Orchestrator가 만들고 내보내는 메시지 형식.

- AgentProposal: 에이전트 4개의 공통 출력
- Conflict / Option: 충돌할 때 LLM(또는 fallback)에 주는 선택지
- LlmSelection: LLM의 구조화 출력 (선택지 ID만 고름)
- DeviceCommand / DeviceResult: 기기 어댑터와 주고받는 메시지 (계약 v2.1 6절 제안)
- SpeakCommand / VoiceReply / VoiceCommand: Node 음성 메시지 (eyear-node schema 0.1과 같은 형식)
- Decision: 판단 1회의 기록 (앱·시뮬레이션 화면 표시용)
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import uuid4

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from orchestrator.contract import ActionName

AgentName = Literal["cleaning", "noise", "temperature", "lighting"]


def new_id() -> str:
    return str(uuid4())


class _M(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ---------- 에이전트 출력 ----------

class ProposedAction(_M):
    device_id: str
    action: ActionName
    params: dict[str, Any] = Field(default_factory=dict)
    reason: str = ""

    def key(self) -> tuple:
        return (self.device_id, self.action, tuple(sorted((k, str(v)) for k, v in self.params.items())))


class Restriction(_M):
    """다른 에이전트의 동작을 막자는 제안. 예: 휴식 중인 방에서 청소 시작 금지."""

    device_id: str
    action: ActionName
    room_id: str | None = None
    strength: Literal["strong", "weak"]
    reason: str


class KeepPlan(_M):
    """이미 진행 중인 기기 계획을 그대로 두자는 제안 (명령 없음). 예: 청소기가 남은 방으로 계속 감."""

    device_id: str
    room_ids: list[str]
    reason: str


class Question(_M):
    kind: Literal["sleep_intent"]
    person_id: str
    room_id: str


class AgentProposal(_M):
    agent: AgentName
    world_session_id: str
    world_revision: int
    actions: list[ProposedAction] = Field(default_factory=list)
    restrictions: list[Restriction] = Field(default_factory=list)
    keeps: list[KeepPlan] = Field(default_factory=list)
    questions: list[Question] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    model: str = "rule"


# ---------- 충돌과 LLM ----------

class Option(_M):
    option_id: str
    summary: str
    actions: list[ProposedAction]
    deferred_rooms: list[str] = Field(default_factory=list)
    from_agents: list[AgentName]


class Conflict(_M):
    conflict_id: str
    device_id: str
    description: str
    options: list[Option]


class Choice(_M):
    conflict_id: str
    option_id: str
    reason: str


class LlmSelection(_M):
    """LLM 구조화 출력. 주어진 conflict_id마다 option_id 하나를 고른다."""

    choices: list[Choice]
    summary: str


# ---------- 기기 ----------

class DeviceCommand(_M):
    schema_version: Literal["2.0"] = "2.0"
    command_id: str = Field(default_factory=new_id)
    plan_id: str
    issued_at: AwareDatetime
    world_session_id: str
    based_on_revision: int
    device_id: str
    action: ActionName
    params: dict[str, Any]
    reason_codes: list[str]


class DeviceResult(_M):
    world_session_id: str
    world_revision: int | None = None
    event_id: str = Field(default_factory=new_id)
    command_id: str
    device_id: str
    status: Literal["accepted", "rejected", "completed", "failed"]
    timestamp: AwareDatetime
    observed_state: dict[str, Any] | None = None
    reason: str | None = None


# ---------- 음성 (eyear-node schema 0.1) ----------

class SpeakCommand(_M):
    schema_version: Literal["0.1"] = "0.1"
    command_id: str = Field(default_factory=new_id)
    timestamp: AwareDatetime
    text: str = Field(min_length=1)
    expect_reply: bool
    listen_seconds: float = Field(default=5.0, gt=0, le=10)


class VoiceReply(BaseModel):
    """Node schema 0.x와 호환되도록 모르는 필드는 무시하되, 값의 일관성은 검사한다."""

    model_config = ConfigDict(extra="ignore")
    schema_version: str = Field(default="0.1", pattern=r"^0\.\d+$")
    event_id: str | None = None
    node_id: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    timestamp: AwareDatetime | None = None
    type: Literal["VOICE_REPLY"] = "VOICE_REPLY"
    command_id: str = Field(min_length=1)
    text: str | None
    confidence: float | None = Field(ge=0.0, le=1.0)
    no_response: bool

    @model_validator(mode="after")
    def _no_response_consistent(self):
        if self.no_response != (self.text is None) or (self.text is None) != (self.confidence is None):
            raise ValueError("no_response면 text·confidence는 null, 아니면 둘 다 값이 있어야 함")
        return self


class VoiceCommand(BaseModel):
    model_config = ConfigDict(extra="ignore")
    schema_version: str = Field(default="0.1", pattern=r"^0\.\d+$")
    event_id: str | None = None
    node_id: str = Field(min_length=1)
    room_id: str = Field(min_length=1)
    timestamp: AwareDatetime | None = None
    type: Literal["VOICE_COMMAND"] = "VOICE_COMMAND"
    text: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)


# ---------- 판단 기록 ----------

class Rejected(_M):
    action: ProposedAction
    reason: str


class Decision(_M):
    plan_id: str = Field(default_factory=new_id)
    world_session_id: str
    world_revision: int
    decided_at: AwareDatetime
    trigger: str
    proposals: list[AgentProposal]
    accepted: list[ProposedAction]
    conflicts: list[Conflict]
    choices: list[Choice]
    arbitration: Literal["none", "llm", "fallback", "cached"]
    commands: list[DeviceCommand]
    rejected: list[Rejected]
    speech: list[SpeakCommand]
    summary: str = ""
    undelivered: list[str] = Field(default_factory=list)   # 발행이 즉시 거절된 음성 (명령은 rejected에 PUBLISH_FAILED)
