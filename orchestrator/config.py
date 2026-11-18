"""설정. 환경변수가 우선이고, 없으면 저장소 루트의 .env를 읽는다 (.env는 git에 올리지 않음)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> dict[str, str]:
    values = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                values[k.strip()] = v.strip().strip('"').strip("'")
    return values


@dataclass
class Settings:
    openai_api_key: str = ""
    openai_model: str = "gpt-6-luna"
    openai_reasoning_effort: str = "low"   # none | low | medium | high | xhigh | max (API 기본값은 medium)
    llm_timeout_s: float = 8.0
    mqtt_host: str = ""
    mqtt_port: int = 1883
    device_transport: str = "mock"   # mock | mqtt
    trace_path: str = "runs/decisions.jsonl"

    @classmethod
    def load(cls, env_file: Path = ROOT / ".env") -> "Settings":
        f = _load_dotenv(env_file)

        def get(name, default=""):
            return os.environ.get(name, f.get(name, default))

        return cls(
            openai_api_key=get("OPENAI_API_KEY"),
            openai_model=get("OPENAI_MODEL") or cls.openai_model,
            openai_reasoning_effort=get("OPENAI_REASONING_EFFORT") or cls.openai_reasoning_effort,
            llm_timeout_s=float(get("LLM_TIMEOUT_S", "8")),
            mqtt_host=get("MQTT_HOST"),
            mqtt_port=int(get("MQTT_PORT", "1883")),
            device_transport=get("DEVICE_TRANSPORT", "mock"),
            trace_path=get("TRACE_PATH", "runs/decisions.jsonl"),
        )


def build_llm(s: Settings):
    """API 키가 있을 때만 LLM을 쓴다. 없으면 None → 결정적 fallback."""
    if not (s.openai_api_key and s.openai_model):
        return None
    from orchestrator.llm import LlmArbiter

    return LlmArbiter(s.openai_api_key, s.openai_model, s.llm_timeout_s, s.openai_reasoning_effort)
