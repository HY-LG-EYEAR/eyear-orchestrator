"""시연 시나리오 t0~t7을 서버·네트워크 없이 재생한다.

    python -m orchestrator.demo              # 규칙·모델만 (LLM 끔)
    python -m orchestrator.demo --llm        # .env의 OpenAI 설정으로 충돌 선택
    python -m orchestrator.demo --reply 아니  # 수면 질문에 "아니"라고 대답
    python -m orchestrator.demo --reply none # 대답하지 않음
"""

from __future__ import annotations

import argparse
from pathlib import Path

from orchestrator.agents import Models
from orchestrator.central import Orchestrator
from orchestrator.fixtures import DemoHome, STEPS
from orchestrator.schemas import Decision, VoiceCommand, VoiceReply


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def run_demo(reply: str | None = "응.", llm=None, models: Models | None = None,
             export_dir: Path | None = None) -> list[tuple[str, Decision]]:
    """(단계, 판단) 목록. 판단이 없는 단계(변화 없음)는 빠진다."""
    clock = FakeClock()
    spoken: list[tuple[str, object]] = []
    home = DemoHome()
    orc = Orchestrator(home.send, lambda room, cmd: spoken.append((room, cmd)), llm=llm,
                       models=models or Models(None), clock=clock)
    home.on_result = orc.on_device_result
    orc.reset(session_id="demo", clock_mode="replay")

    out: list[tuple[str, Decision]] = []

    def push(step: str, d: Decision | None):
        if d is not None:
            out.append((step, d))

    for step in STEPS:
        if step == "t5":  # 잠에서 깬 A: "하이 엘지, 불 켜줘"
            push("t5-voice", orc.on_voice_command(VoiceCommand(node_id="node_room1", room_id="room1",
                                                               text="불 켜줘", confidence=0.9)))
        snap = home.snapshot(step)
        if export_dir is not None:
            (export_dir / f"snapshot_{step}.json").write_text(snap.model_dump_json(indent=2), encoding="utf-8")
        push(step, orc.on_snapshot(snap))
        # 수면 질문이 나갔으면 Node가 대답한다
        for room, cmd in list(spoken):
            if getattr(cmd, "expect_reply", False):
                spoken.remove((room, cmd))
                clock.t += 3
                r = VoiceReply(node_id="node_room1", room_id=room, command_id=cmd.command_id,
                               text=reply, confidence=None if reply is None else 0.55, no_response=reply is None)
                push(f"{step}-reply", orc.on_voice_reply(r))
    return out


def describe(d: Decision) -> list[str]:
    lines = []
    for c in d.commands:
        lines.append(f"  명령  {c.device_id}.{c.action} {c.params}")
    for s in d.speech:
        lines.append(f"  음성  \"{s.text}\"{' (대답 기다림)' if s.expect_reply else ''}")
    for c in d.conflicts:
        ch = next(x for x in d.choices if x.conflict_id == c.conflict_id)
        lines.append(f"  충돌  {c.description} → {ch.option_id} [{d.arbitration}] {ch.reason}")
    for r in d.rejected:
        lines.append(f"  거절  {r.action.device_id}.{r.action.action} {r.action.params}: {r.reason}")
    return lines or ["  (명령 없음)"]


def main() -> None:
    ap = argparse.ArgumentParser(description="EYEAR 시연 시나리오 t0~t7 재생")
    ap.add_argument("--llm", action="store_true", help=".env의 OpenAI 설정으로 충돌을 선택")
    ap.add_argument("--reply", default="응.", help='수면 질문에 대한 대답. "none"이면 무응답')
    ap.add_argument("--no-models", action="store_true", help="학습 모델 대신 규칙만 사용")
    ap.add_argument("--export", metavar="DIR", help="단계별 입력 스냅샷과 판단 기록을 JSON으로 저장")
    args = ap.parse_args()
    export = Path(args.export) if args.export else None
    if export:
        export.mkdir(parents=True, exist_ok=True)

    from orchestrator.config import Settings, build_llm

    settings = Settings.load()
    llm = build_llm(settings) if args.llm else None
    if args.llm and llm is None:
        print("LLM 설정이 없어 규칙 선택(fallback)으로 진행합니다. .env의 OPENAI_API_KEY, OPENAI_MODEL을 확인하세요.")
    models = Models(None) if args.no_models else Models()
    reply = None if args.reply.lower() == "none" else args.reply
    for step, d in run_demo(reply, llm, models, export):
        print(f"[{step}] rev {d.world_revision} · {d.trigger}")
        print("\n".join(describe(d)))
        if export:
            (export / f"decision_{step}.json").write_text(d.model_dump_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
