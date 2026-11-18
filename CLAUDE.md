# EYEAR Orchestrator — 개발 컨텍스트

송정빈 담당 파트(에이전트 4개 + 중앙 판단)의 작업 기록. 결정에는 날짜를 붙이고, 확정/초안/미정을 구분한다.
공개 저장소이므로 키·개인 정보·개인 메모는 쓰지 않는다. 길어지면 오래된 결정은 요약한다.

## 1. 프로젝트 요약
- EYEAR: 방마다 Node(카메라·마이크·스피커·mmWave)가 의미 정보만 보내고, World Model이 집 전체 상태를 만들고, 에이전트와 중앙이 가전을 조정한다. LG ThinQ ON 맥락의 학생 프로젝트(CSE406, ITE351).
- 팀: 신반석(Node, eyear-node), 이효창(World Model·시뮬레이션, eyear-worldmodel), 송정빈(이 저장소, Development Manager 겸), 정예원(앱: Expo(React Native) + Spring Boot, eyear-app. 2026-11-16부터 Expo. 앱의 기기·루틴 모델은 아직 계약 v2와 별개).
- 시연 시나리오(최우선 기준): HTML "방1에서 쉬고 거실로 나온 A의 저녁" t0~t7. 코드상 재현은 `orchestrator/fixtures.py`의 STEPS와 `orchestrator/demo.py`.

## 2. 결정 사항
- [확정 2026-11-18] 중앙 방식: 충돌 없는 제안은 수용, 충돌만 LLM이 **주어진 선택지 중 하나**를 고름. LLM은 명령을 새로 만들지 않음. LLM 실패 시 결정적 fallback(제한을 지키는 선택). 같은 충돌은 선택 재사용(memo).
- [확정 2026-11-18] 충돌 탐지는 구조적: 같은 (기기, 동작)에 다른 값, 또는 Restriction(기기, 동작, 방)에 걸리는 동작. 시나리오별 규칙을 두지 않음.
- [확정 2026-11-18] 사전 정책은 최소: 검증기(능력·범위·온라인·수동 조작·사용자 금지·중복) + fallback. "점유 방 청소 금지"는 소음 에이전트의 strong Restriction으로 표현되고, 최종 선택은 LLM/fallback.
- [확정 2026-11-18] 청소 순서는 루틴 `room_ids` 순서 그대로. 경로 최적화 없음(계약에 거리 정보 없음, 경로는 청소기 몫).
- [확정 2026-11-18] 모델: 청소 HGB 회귀, 소음 로지스틱, 온도 Ridge(온열감), 빛 로지스틱. scikit-learn만. 모델 없으면 규칙.
- [확정 2026-11-18] 데이터셋: 가구 시뮬레이터 1개 → 표 4개(행 단위가 달라 하나로 합치지 않음). 특징은 실행과 같은 features.py, 라벨은 숨은 상태에서만. 가구 단위 분할. GPT로 데이터 생성 안 함.
- [확정 2026-11-18] 수면 질문 문장: "{A}님, 주무실 거면 '네'라고 말씀해 주세요." (Node 팀 제안). 대답 마감 = 말하는 시간 + listen_seconds + 2초. 문장부호 무시, 신뢰도 0.2 이상이면 해석.
- [확정 2026-11-18] 중앙 LLM: `gpt-6-luna`, 추론 강도 `low` (설정 기본값, .env로 변경 가능). 선택지 ID만 고르는 작업이라 가장 저렴한 모델로 충분.
- [확정 2026-11-18] 일시정지는 자동 제어만 멈춤. 직접 음성 명령은 일시정지 중에도 실행.
- [확정 2026-11-18 점검 반영] 결과는 종료 상태(completed/failed/rejected)에서만 pending 해제, 세션·기기 대조, 60초 무응답이면 유실 처리. plan_id는 판단과 명령이 공유. 같은 세션의 옛 revision 무시. 판단 키에 온도(0.5°C)·근거 신선도·마감 구간 포함. 충돌 선택 재사용은 같은 마감·점수 구간·사용자 금지일 때 15분까지. LLM은 lock 밖에서 호출하고 상태가 바뀌면 재계산. 기기 상태 근거가 current가 아니면 명령 거절. 수면 대답은 질문한 Node·방에서 온 것만. 깨진 모델 파일은 규칙으로 대체.
- [초안] 입력 계약: WorldSnapshot 2.0 (v2 초안 11-13, v2.1 문서 11-18은 송정빈이 따로 보관·이효창에게 직접 전달). 전송 형식은 2.0 그대로.
- [초안] 기기 명령 MQTT `eyear/v0/device/{id}/command`, 결과 `POST /v2/world/device-results`, 세션 `POST /v2/world/sessions`. 합의 전.

## 3. 미정 / 확인 대기
- World Model 코드가 GitHub에 없음 → 실제 출력 미검증. TRACK_CARRIED 출력, 두 Node에 같은 사람, devices/profiles/routines 공급처 질문 중.
- 계약 호환 깨는 제안: `stable_for_s`→`stable_since`, `publish_reason`, `room_links` 정리.
- 앱 제어·표시 형식, MQTT 브로커 위치, 배포(로컬/EC2), failsafe 전환.
- 실제 LLM 호출은 아직 미검증.
- eyear-team inbox: 신반석 11-06 음성 형식 요청(회신 필요), 11-11 인식 개선 알림.

## 4. 코드 지도
- `contract.py` 입력 계약 → `features.py` 특징·History → `agents.py` 4개 → `central.py` 병합·충돌·선택·검증·발행 → `voice.py` 대화 상태 → `server.py` HTTP/MQTT.
- `fixtures.py` 스냅샷 생성 도구 + DemoHome(World Model·기기 흉내) + 시나리오. `dataset/` 시뮬레이터·학습.

## 5. 작업 원칙
- 계약이 바뀌면 `contract.py`, `features.py`만 고친다. 에이전트는 features 결과만 본다.
- 모르는 값은 null/UNKNOWN 그대로. 0이나 추정으로 채우지 않는다.
- 시나리오 결과가 바뀌는 변경은 `tests/test_orchestrator.py::test_scenario_matches_demo`로 확인한다.
- 커밋은 사용자가 요청할 때만. 키는 `.env`에만.

## 6. 자주 쓰는 명령
- `.venv/Scripts/python -m pytest` / `-m orchestrator.demo [--llm] [--reply none]` / `-m dataset.train` / `-m orchestrator.server`

## 팀 공유 (eyear-team inbox)
- 다른 팀원에게 알릴 내용은 `../eyear-team/inbox/<받는 사람 id>.md`에 쓴다.
- 세션 시작 훅이 eyear-team을 pull 하고 RULES.md와 내 inbox(`inbox/jungbin.md`)를 출력해 준다.
  규칙은 RULES.md를 따르고, inbox에 항목이 있으면 작업 전에 먼저 알린다.
