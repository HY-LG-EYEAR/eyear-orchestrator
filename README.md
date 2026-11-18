# EYEAR Orchestrator

[EYEAR](https://github.com/HY-LG-EYEAR)의 에이전트 4개와 중앙 판단부. **담당: 송정빈**

World Model이 보낸 집 전체 상태(WorldSnapshot 2.0)를 읽고, 청소·소음·온도·빛 에이전트가 제안을 낸 뒤, 중앙이 가전 명령과 Node 음성(질문·응답)을 결정한다.

> 상태: **초안 v0.1** — 실제로 동작하고 테스트를 통과하지만, 팀 합의(계약 v2.1, 명령·결과 경로, 앱 연결)에 따라 바뀔 수 있다.

## 흐름

```text
World Model ──WorldSnapshot 2.0 (HTTP)──▶ 에이전트 4개 ──제안──▶ 중앙 판단 ──검증──▶ 기기 명령
                                                                     │
Node ◀──말할 문장 (MQTT speak)── 중앙 ◀──대답·직접 명령 (MQTT voice/*)── Node
기기 어댑터 ──실행 결과──▶ World Model ──(HTTP)──▶ 중앙
```

**중앙 판단 방식**
1. 에이전트 4개가 같은 스냅샷으로 제안한다. 에이전트는 명령을 직접 보내지 않는다.
2. **충돌하지 않는 제안은 그대로 수용**한다.
3. 충돌(같은 기기에 다른 값, 또는 다른 에이전트의 제한에 걸리는 동작)이 있으면 선택지를 기계적으로 만들고, **LLM이 그중 하나를 고른다.** LLM은 새 명령을 만들지 못한다.
4. LLM이 없거나 실패하면 결정적 fallback(제한을 지키는 선택)을 쓴다. 같은 충돌이 이어지면 앞의 선택을 재사용한다.
5. 모든 명령은 실행 전에 검증한다: 기기 지원 동작·모드·온도 범위·수동 조작·온라인·사용자 금지·중복.

## World Model에서 읽는 값 (에이전트별)

| 에이전트 | 읽는 WorldSnapshot 필드 | 중앙이 직접 기억하는 값 |
|---|---|---|
| 청소 | `routines[]`(회차, 방 목록=청소 순서, `due`, 마감), 청소기 `state`·`capabilities`·`online`·`manual_override`, `rooms[].room_type` | 방별 마지막 청소 시각, 청소 후 재실 누적 시간 |
| 소음 | `rooms[].occupancy`, `people[].activity`(label, confirmed), `profile_id`, `profiles[].quiet_rest/quiet_meeting`, `room_influence`, 세탁기·청소기·공기청정기 상태와 모드 | 소음 보호로 모드를 바꾼 기기와 그 이유가 된 방 |
| 온도 | 에어컨 방의 `people[]`(`profile_id`, `position`), `profiles[].preferred_min/max_c`, `rooms[].temperature_c`, 에어컨 `state`·`capabilities.temperature`·`airflow_target` | — |
| 빛 | `people[].activity`(resting, confirmed, stable_for_s), `rooms[].light`, 조명·커튼 상태, `nodes[].supports_speak` | 휴식 에피소드와 질문 상태 |

필드 정의는 [`orchestrator/contract.py`](orchestrator/contract.py), 전체 예시는 [`examples/snapshot_t2.json`](examples/snapshot_t2.json).

## 에이전트 모델

모두 scikit-learn. 모델 파일은 수 KB~200 KB, CPU에서 바로 추론한다. 모델이 없으면 규칙 baseline으로 동작한다.

| 에이전트 | 모델이 내는 값 | 채택 모델 | 비교한 후보 | 쓰임 |
|---|---|---|---|---|
| 청소 | 방별 청소 필요도 0~1 | HistGradientBoosting 회귀 | Ridge, 규칙 | 충돌 시 LLM이 보는 근거 (`urgency:방`) |
| 소음 | (기기 모드, 방)별 방해 확률 | 로지스틱 회귀 | HistGradientBoosting, 규칙 | 0.5 이상이면 세탁기·공기청정기 저소음 제안 |
| 온도 | (사람, 설정온도)별 온열감(+덥다/−춥다) | Ridge 회귀 | HistGradientBoosting, 규칙 | 방 안 사람들의 온열감 제곱합이 가장 작은 온도 |
| 빛 | 휴식 중인 사람의 수면 의도 확률 | 로지스틱 회귀 | HistGradientBoosting, 규칙 | 0.3 이상이면 수면 여부 질문 |

학습 데이터는 가구 시뮬레이터([`dataset/simulate.py`](dataset/simulate.py))로 만든다. 실행과 같은 WorldSnapshot 객체와 같은 특징 함수([`orchestrator/features.py`](orchestrator/features.py))를 쓰고, 라벨은 시뮬레이터의 숨은 상태에서만 만든다. 평가는 가구 단위로 나눈다. 지표는 [`models/model_card.json`](models/model_card.json). **시뮬레이터 데이터이므로 실제 가구에서의 정확도를 뜻하지 않는다.**

## 내보내는 형식

| 메시지 | 형식 | 예시 |
|---|---|---|
| 에이전트 제안 | `AgentProposal`: actions, restrictions, questions, scores, reasons | [`decision_t1.json`](examples/scenario/decision_t1.json)의 `proposals` |
| 판단 기록 | `Decision`: 제안, 수용, 충돌·선택지·선택, 명령, 거절, 음성 | [`examples/scenario/`](examples/scenario/) |
| 기기 명령 | `DeviceCommand`: command_id, device_id, action, params, reason_codes | 동작 이름은 계약의 9개 |
| 말할 문장 | `SpeakCommand` (eyear-node schema 0.1과 같음) | `"A님, 주무실 거면 '네'라고 말씀해 주세요."` |
| LLM 출력 | `LlmSelection`: choices[{conflict_id, option_id, reason}], summary | |

모든 JSON Schema는 서버의 `GET /v1/schemas`에서 볼 수 있다. 정의는 [`orchestrator/schemas.py`](orchestrator/schemas.py).

## 주고받는 경로 (초안, 합의 필요)

| 방향 | 경로 |
|---|---|
| World Model → 중앙 | `POST /v2/world/sessions`(세션 등록) → `POST /v2/world/snapshots`(202 / 409 / 422) |
| 기기 결과 → 중앙 | `POST /v2/world/device-results` |
| Node ↔ 중앙 | MQTT `eyear/v0/{room_id}/speak`, `.../voice/reply`, `.../voice/command` (HTTP 대안: `POST /v1/voice/reply`, `/v1/voice/command`) |
| 중앙 → 기기 | MQTT `eyear/v0/device/{device_id}/command` (`DEVICE_TRANSPORT=mqtt`일 때. 기본은 mock) |
| 앱 → 중앙 | `POST /v1/control` {`action`: pause \| resume \| reset} |
| 중앙 → 앱 | `GET /v1/decisions`, `GET /v1/status`, `GET /v1/speech` |

## 실행

Python 3.11 이상. 가상환경은 저장소 안의 `.venv`에 만든다(git 제외).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # macOS·Linux: .venv/bin/python
```

`.env.example`을 `.env`로 복사하고 `OPENAI_API_KEY`를 채우면 LLM 중재를 쓴다(기본 `gpt-6-luna`, 추론 강도 `low`). 키가 없으면 결정적 선택만 쓴다.

```bash
.venv/Scripts/python -m orchestrator.demo                  # 시연 시나리오 t0~t7 재생 (네트워크 없음)
.venv/Scripts/python -m orchestrator.demo --llm             # 충돌을 LLM이 선택
.venv/Scripts/python -m orchestrator.demo --reply none      # 수면 질문에 대답하지 않을 때
.venv/Scripts/python -m dataset.train                       # 데이터 생성 + 모델 4개 학습 (약 30초)
.venv/Scripts/python -m orchestrator.server                 # http://127.0.0.1:8765, 문서 /docs
.venv/Scripts/python -m pytest                              # 테스트
```

scikit-learn 버전이 달라 모델 파일을 읽지 못하면 `python -m dataset.train`으로 다시 만들면 된다.

## 폴더

```text
orchestrator/   contract.py(입력 계약) schemas.py(출력 형식) features.py agents.py central.py
                voice.py llm.py server.py config.py fixtures.py(스냅샷 생성 도구·시나리오) demo.py
dataset/        simulate.py(가구 시뮬레이터) train.py(학습·평가)  samples/(표마다 200행)
models/         학습된 모델 4개와 model_card.json
examples/       snapshot_t2.json(계약 예시), scenario/(t0~t7 입력 스냅샷과 판단 기록)
tests/
```

## 아직 정해지지 않은 것

- World Model 계약 v2.1의 합의 (세션 등록, 기기 명령·결과 경로, `stable_since` 등)
- 앱(Spring Boot)과 주고받을 제어·표시 형식
- 시연 때 MQTT 브로커 위치, 서버 배포 위치(로컬 / 클라우드)
