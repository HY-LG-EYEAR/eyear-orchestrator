"""충돌한 제안 사이에서 LLM이 선택지 하나를 고른다 (OpenAI Structured Outputs + Pydantic).

LLM은 주어진 option_id 중 하나만 고를 수 있고, 새 기기 명령을 만들지 않는다.
결과는 central.py에서 다시 검사하고, 실패하면 결정적 fallback을 쓴다.
"""

from __future__ import annotations

import json
import logging

from orchestrator.schemas import Conflict, LlmSelection

log = logging.getLogger(__name__)

SYSTEM = """You are the central arbiter of a smart home.
Role-based agents (cleaning, noise, temperature, lighting) made proposals. Proposals that do not conflict were already accepted.
For each conflict, choose exactly one option_id from the given options. Do not invent options or device commands.
Weigh the residents' current activity and rest, their quiet preferences, cleaning urgency and deadlines, and whether occupancy is uncertain.
Avoid disturbing people who are present or resting unless the alternative is clearly worse; deferring a task keeps it for later.
Write each reason and the summary in short Korean (one sentence)."""


class LlmArbiter:
    def __init__(self, api_key: str, model: str, timeout_s: float = 8.0, reasoning_effort: str = "low"):
        from openai import OpenAI

        self.model, self.reasoning_effort = model, reasoning_effort
        self.client = OpenAI(api_key=api_key, timeout=timeout_s, max_retries=0)

    def select(self, context: dict, conflicts: list[Conflict]) -> LlmSelection | None:
        payload = {"home": context, "conflicts": [c.model_dump() for c in conflicts]}
        try:
            completion = self.client.chat.completions.parse(
                model=self.model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}],
                response_format=LlmSelection,
                reasoning_effort=self.reasoning_effort,
            )
            msg = completion.choices[0].message
            if getattr(msg, "refusal", None):
                log.warning("LLM refusal: %s", msg.refusal)
                return None
            return msg.parsed
        except Exception as e:  # 시간 초과, 네트워크, 형식 오류 → fallback
            log.warning("LLM 실패: %s", e)
            return None
