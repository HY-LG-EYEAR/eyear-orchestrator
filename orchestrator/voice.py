"""음성 대화: 휴식 1회당 질문 1회, 대답 해석, 직접 명령 해석.

Node(eyear-node schema 0.1)는 문장을 읽고 들은 말을 텍스트로만 보낸다. 해석은 여기서 한다.
Node에는 재생 완료 신호가 없으므로 대답 마감은 "말하는 시간 + listen_seconds + 2초"로 잡는다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from orchestrator.contract import WorldSnapshot
from orchestrator.schemas import VoiceReply

LISTEN_SECONDS = 5.0
REPLY_MARGIN_S = 2.0
QUEUE_ALLOWANCE_S = 8.0   # Node가 다른 대화 중이면 질문을 줄 세웠다가 읽으므로 그만큼 더 기다린다
MIN_REPLY_CONF = 0.2      # 짧은 대답은 신뢰도 0.3~0.6이 정상 (Node 실측)
MIN_COMMAND_CONF = 0.4    # 직접 명령은 기기를 움직이므로 대답보다 높은 신뢰도를 요구한다

# 대답은 "짧고 명확한 말"만 긍정·부정으로 본다. 질문, 긴 문장, 섞인 말은 모호(ambiguous)
QUESTION_WORDS = ("뭐", "뭔", "왜", "어떻게", "언제", "누구", "무슨")
YES_WORDS = {"응", "네", "예", "웅", "엉", "ㅇㅇ", "넵", "넹", "yes", "그래", "그래요", "맞아", "맞아요", "좋아", "좋아요"}
YES_SLEEP = {"잘게", "잘게요", "잘래", "잘래요", "자려고", "자려고요", "잘거야", "잘거예요", "잘거에요", "잘꺼야", "자야지"}
NO_WORDS = {"아니", "아니요", "아니오", "아뇨", "아냐", "싫어", "싫어요", "노"}
NO_SLEEP = ("안잘", "안자", "안잔")


def normalize(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z가-힣ㄱ-ㅎ]", "", text or "").lower()


def interpret_reply(text: str | None, confidence: float | None) -> str:
    """'yes' | 'no' | 'ambiguous'. 문장부호는 무시한다(Node 인식기가 '응?'처럼 붙임).
    "어"는 '응'과 '뭐?' 둘 다 될 수 있어 모호로 본다."""
    if not text or confidence is None or confidence < MIN_REPLY_CONF:
        return "ambiguous"
    t = normalize(text)
    if not t or any(q in t for q in QUESTION_WORDS):
        return "ambiguous"
    no = t in NO_WORDS or t.startswith(("아니", "아뇨", "아냐")) or any(x in t for x in NO_SLEEP)
    yes = t in YES_WORDS or t in YES_SLEEP or any(t == a + b for a in YES_WORDS for b in YES_SLEEP | {"응", "네"})
    if yes == no:
        return "ambiguous"
    return "yes" if yes else "no"


def display_name(identity: str) -> str:
    return identity.removeprefix("MEMBER_")


def question_text(identity: str) -> str:
    # Node 팀 제안: 대답 형태를 알려 주면 짧은 대답 인식이 안정적
    return f"{display_name(identity)}님, 주무실 거면 '네'라고 말씀해 주세요."


def speech_seconds(text: str) -> float:
    return 1.0 + 0.15 * len(text)


@dataclass
class Episode:
    person_id: str
    room_id: str
    identity: str
    state: str = "NOT_ASKED"          # NOT_ASKED → AWAITING_REPLY → ANSWERED | EXPIRED
    command_id: str | None = None
    deadline: float | None = None
    answer: str | None = None         # yes | no | ambiguous
    consumed: bool = False            # 긍정 대답에 따른 조명·커튼 제안을 중앙이 처리했는가
    node_id: str | None = None        # 질문을 보낸 Node. 대답은 이 Node·방에서 온 것만 받는다


class DialogueManager:
    """사람마다 휴식 에피소드 하나. 에피소드마다 질문은 한 번만."""

    def __init__(self):
        self.episodes: dict[str, Episode] = {}
        self.log: list[str] = []

    def episode(self, person_id: str) -> Episode | None:
        return self.episodes.get(person_id)

    def update(self, snap: WorldSnapshot) -> None:
        occ = {r.room_id: r.occupancy.state for r in snap.rooms}
        seen = set()
        for p in snap.people:
            seen.add(p.person_id)
            ep = self.episodes.get(p.person_id)
            label, confirmed = p.activity.label, p.activity.confirmed
            if ep is None:
                if label == "resting" and confirmed and p.room_id:
                    self.episodes[p.person_id] = Episode(p.person_id, p.room_id, p.identity)
                    self.log.append(f"episode start {p.person_id}@{p.room_id}")
                continue
            # 다른 방에서 지금 관측되면 원래 방에 다른 사람이 남아 있어도 이동으로 본다.
            # 위치만 모르게 된 경우(room_id=None)는 원래 방이 비었을 때만 끝낸다.
            moved = p.room_id is not None and p.room_id != ep.room_id and p.evidence.freshness == "current"
            left = moved or (p.room_id != ep.room_id and occ.get(ep.room_id) != "OCCUPIED")
            rose = confirmed and label not in ("resting", "unknown")
            if left or rose:
                self._end(ep, "left" if left else "rose")
        for pid in list(self.episodes):
            if pid not in seen and occ.get(self.episodes[pid].room_id) != "OCCUPIED":
                self._end(self.episodes[pid], "gone")

    def _end(self, ep: Episode, why: str) -> None:
        self.log.append(f"episode end {ep.person_id} ({why}, state={ep.state})")
        self.episodes.pop(ep.person_id, None)

    def asked(self, person_id: str, command_id: str, now: float, text: str, node_id: str | None = None) -> None:
        ep = self.episodes[person_id]
        ep.state, ep.command_id, ep.node_id = "AWAITING_REPLY", command_id, node_id
        ep.deadline = now + speech_seconds(text) + LISTEN_SECONDS + REPLY_MARGIN_S + QUEUE_ALLOWANCE_S

    def awaiting_in_room(self, room_id: str) -> bool:
        return any(e.state == "AWAITING_REPLY" and e.room_id == room_id for e in self.episodes.values())

    def on_reply(self, reply: VoiceReply, now: float) -> Episode | None:
        ep = next((e for e in self.episodes.values() if e.command_id == reply.command_id), None)
        if ep is None or ep.state != "AWAITING_REPLY":
            self.log.append(f"reply ignored (unknown or obsolete command {reply.command_id})")
            return None
        if reply.room_id != ep.room_id or (ep.node_id is not None and reply.node_id != ep.node_id):
            self.log.append(f"reply ignored (from {reply.node_id}@{reply.room_id}, asked {ep.node_id}@{ep.room_id})")
            return None
        if ep.deadline is not None and now > ep.deadline + 1.0:
            ep.state = "EXPIRED"
            self.log.append(f"reply too late {ep.person_id}")
            return None
        ep.state = "ANSWERED"
        ep.answer = "ambiguous" if reply.no_response else interpret_reply(reply.text, reply.confidence)
        self.log.append(f"answer {ep.person_id}: {ep.answer} ({reply.text!r})")
        return ep

    def check_deadlines(self, now: float) -> list[Episode]:
        expired = []
        for ep in self.episodes.values():
            if ep.state == "AWAITING_REPLY" and ep.deadline is not None and now > ep.deadline:
                ep.state = "EXPIRED"
                expired.append(ep)
                self.log.append(f"reply timeout {ep.person_id}")
        return expired


# ---------- 직접 명령 ("하이 엘지" 뒤 문장) ----------

NEGATION = ("지마", "지말", "말아", "말고", "하지마", "안켜", "안꺼", "안끄")
QUESTION_ENDINGS = ("나", "니", "냐", "까", "나요", "까요", "니까", "었어", "았어", "켰어", "껐어", "했어", "됐어",
                    "인가", "인가요")


def command_problem(text: str) -> str | None:
    """실행하면 안 되는 문장이면 이유. 부정("켜지 마")과 질문("켜야 하나?", "켰어?")은 실행하지 않는다.
    "불 켜줘?"처럼 인식기가 물음표만 붙인 명령은 실행한다(어미가 명령형)."""
    t = normalize(text)
    if any(n in t for n in NEGATION):
        return "NEGATED"
    if t.endswith(QUESTION_ENDINGS):
        return "QUESTION"
    return None


def parse_command(text: str) -> tuple[str, str, dict, str] | None:
    """(기기 종류, 동작, 파라미터, 응답 문장) 또는 None. 부정·질문 문장은 None."""
    if command_problem(text):
        return None
    t = normalize(text)
    on = any(w in t for w in ("켜", "킬", "틀어"))
    off = any(w in t for w in ("꺼", "끄", "끌"))
    if "불" in t or "조명" in t or "전등" in t:
        if on:
            return ("light", "set_power", {"power": "on"}, "네, 불을 켤게요.")
        if off:
            return ("light", "set_power", {"power": "off"}, "네, 불을 끌게요.")
    if "커튼" in t:
        if "열" in t:
            return ("curtain", "set_curtain_position", {"position_pct": 100}, "네, 커튼을 열게요.")
        if "닫" in t or "쳐" in t:
            return ("curtain", "set_curtain_position", {"position_pct": 0}, "네, 커튼을 닫을게요.")
    if "에어컨" in t:
        if on:
            return ("ac", "set_power", {"power": "on"}, "네, 에어컨을 켤게요.")
        if off:
            return ("ac", "set_power", {"power": "off"}, "네, 에어컨을 끌게요.")
    return None
