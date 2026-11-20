# 시뮬레이터·World Model과 맞출 형식 (발표 시나리오 기준)

중앙(Orchestrator)이 발표 시나리오를 그대로 재현하려면 시뮬레이터·World Model 쪽이 아래 상태·명령·결과를 지켜야 한다.
범위는 발표 시나리오(t0~t7)뿐이다. 코드상 기준은 [`orchestrator/fixtures.py`](../orchestrator/fixtures.py)의 `DemoHome`이다.

## 1. 연결

| 방향 | 경로 | 비고 |
|---|---|---|
| World Model → 중앙 | `POST /v2/world/sessions` | 시작할 때 한 번. 시나리오 재생은 `clock_mode: "replay"` (과거 시각이라 `live`면 거부됨) |
| World Model → 중앙 | `POST /v2/world/snapshots` | WorldSnapshot 2.0 전체. 의미 있는 변화마다 + 변화가 없어도 30초마다. 응답은 202뿐이고 판단 결과는 응답에 없음 |
| 중앙 → 기기(시뮬레이터) | MQTT `eyear/v0/device/{device_id}/command` | 아래 DeviceCommand |
| 기기(시뮬레이터) → World Model | MQTT `eyear/v0/device/{device_id}/state`, `.../result` | 기기 상태(주기 보고)와 명령 실행 결과. 형식은 World Model 입력 형식(`ingest_device_result`: `state` 필드) |
| World Model → 중앙 | `POST /v2/world/device-results` | 결과를 상관 ID와 함께 전달. 아래 DeviceResult (`observed_state` 필드) |
| 화면 표시 | `GET /v1/decisions` | 판단 기록 (충돌, 선택지, LLM/규칙 여부, 명령, 보류 이유) |

## 2. 스냅샷 안의 청소기와 루틴

**루틴 (`routines[]`)**
- `occurrence_id`: 오늘 저녁 청소 회차 하나 (예: `daily_clean_20261113_evening`). t0부터 끝까지 같은 값.
- `device_id`: `vacuum_01`, `room_ids`: `["room2", "room1"]` (청소 순서), `due: true`는 t0부터 계속.

**청소기 (`devices[]`, `kind: "vacuum"`)**
- `state.status`: `cleaning` | `returning` | `docked` (그 밖의 값은 시나리오에서 안 씀)
- `state.target_room_ids`: **남은 방, 순서대로. 첫 번째가 지금 청소 중이거나 다음에 갈 방.** 끝낸 방은 목록에서 뺀다.
- `state.mode`: `"normal"`. `capabilities.actions`에 `start_cleaning`, `return_to_dock` 포함, `capabilities.modes: ["normal"]`.
- `online: true`, `evidence.freshness: "current"`. 시뮬레이터는 기기 상태를 MQTT `.../state`로 World Model의 10초 TTL 안에 주기적으로 보고해야 한다(오래되면 중앙이 그 기기에 명령하지 않음).

**방 재실**: 청소기는 `occupancy.state == "EMPTY"`이고 근거가 `current`인 방에만 들어간다. `OCCUPIED`, `UNKNOWN`, 오래된 `EMPTY`는 모두 "비지 않음"으로 보고 기다린다.

## 3. 장면별 청소기 상태와 중앙의 동작

| 장면 | 청소기 상태 (스냅샷) | 방1 재실 | 중앙 |
|---|---|---|---|
| t0 | `cleaning`, `["room2", "room1"]` | EMPTY | 명령 없음 (계획 유지) |
| 방2 완료 직후 | `cleaning`, `["room1"]`, 아직 방1 밖 | EMPTY | 명령 없음. 중앙은 방2를 끝난 방으로 기록 |
| **t1** | `cleaning`, `["room1"]`, 아직 방1 밖 | **OCCUPIED** | 충돌 "계획 유지 vs 도크 복귀" → LLM(실패 시 규칙: 복귀) → `return_to_dock` 한 번 |
| 복귀 중 | `returning`, `[]` | OCCUPIED | 명령 없음 |
| t2~t6-2 | `docked`, `[]` | OCCUPIED 또는 UNKNOWN | 명령 없음. 판단 기록에 `WAIT_UNTIL_EMPTY:room1(...)`. LLM을 다시 부르지 않음 |
| **t6-3** | `docked`, `[]` | **EMPTY (current)** | `start_cleaning {"room_ids": ["room1"], "mode": "normal"}` (방2는 넣지 않음) |
| 방1 청소 중 | `cleaning`, `["room1"]` | EMPTY | 명령 없음 |
| 청소 끝 | `docked` (또는 `returning` → `docked`) | — | 완료 결과를 받으면 회차 종료. 이후 청소 명령 없음 |

### t1에서 LLM을 기다리는 동안 방1에 들어가지 않게 하는 조건
1. 방2를 끝내면 `target_room_ids`를 `["room1"]`로 바꾼 스냅샷을 **방1에 들어가기 전에** 보낸다. A가 방1에 들어오는 t1 스냅샷도 청소기가 아직 방1 밖인 상태여야 한다.
2. 중앙의 판단은 스냅샷 응답(202) 뒤 비동기로 끝나고, 명령은 MQTT로 온다. LLM 시간 제한은 8초다. 그래서 **t1 스냅샷을 보낸 뒤 청소기가 방1에 들어가기까지 최소 10초**를 둔다. 또는 시뮬레이터가 방1 문 앞에서 기다렸다가, `GET /v1/decisions`에 t1의 `world_revision` 이상인 판단이 나온 뒤에 움직인다.
3. 그 사이 `return_to_dock`이 오면 방1에 들어가지 않고 도크로 돌아간다. LLM이 "계획 유지"를 고르면 명령이 오지 않으므로 그대로 방1로 들어간다(발표에서는 리허설로 확인하고, 필요하면 LLM을 끈 규칙 실행을 쓴다. 판단 기록의 `arbitration`이 `llm`인지 `fallback`인지 화면에 표시할 수 있다).

## 4. 명령 (중앙 → 기기)

MQTT `eyear/v0/device/{device_id}/command`, JSON:

```json
{"schema_version": "2.0", "command_id": "uuid", "plan_id": "uuid", "issued_at": "2026-11-13T21:02:10+09:00",
 "world_session_id": "...", "based_on_revision": 105, "device_id": "vacuum_01",
 "action": "return_to_dock", "params": {}, "reason_codes": ["NEXT_ROOM_OCCUPIED:room1", "snapshot"]}
```

시나리오에서 쓰는 동작 (HTML 이름 → 중앙 이름):

| 기기 | HTML | 중앙 `action` | `params` |
|---|---|---|---|
| 청소기 | `CLEAN_ROOM` | `start_cleaning` | `{"room_ids": ["room1"], "mode": "normal"}` |
| 청소기 | `RETURN_TO_DOCK` | `return_to_dock` | `{}` |
| 세탁기 | `SET_MODE` | `set_mode` | `{"mode": "low_noise"}` / `{"mode": "normal"}` |
| 에어컨 | `TURN_ON` / `TURN_OFF` | `set_power` | `{"power": "on"}` / `{"power": "off"}` |
| 에어컨 | `SET_TARGET` | `set_target_temperature` | `{"target_c": 25.0}` |
| 에어컨 | `SET_WIND` | `set_airflow_target` | `{"person_id": "person_a"}` |
| 조명 | `TURN_ON` / `TURN_OFF` | `set_power` | `{"power": "on"}` / `{"power": "off"}` |
| 커튼 | `OPEN` / `CLOSE` | `set_curtain_position` | `{"position_pct": 100}` / `{"position_pct": 0}` |

## 5. 결과 (기기 → World Model → 중앙)

경로는 팀 연결도(eyear-team `docs/connections.html` ⑥·⑦)를 따른다. 시뮬레이터는 MQTT `eyear/v0/device/{device_id}/result`로 World Model에 보내고, World Model이 기기 상태를 갱신한 뒤 세션·revision을 붙여 중앙에 HTTP로 전달한다. 시뮬레이터가 중앙에 직접 보내지 않는다.

World Model → 중앙 `POST /v2/world/device-results`, JSON:

```json
{"world_session_id": "...", "world_revision": 118, "event_id": "uuid", "command_id": "명령의 command_id",
 "device_id": "vacuum_01", "status": "accepted", "timestamp": "2026-11-13T22:55:21+09:00",
 "observed_state": {"status": "cleaning", "mode": "normal", "target_room_ids": ["room1"], "dock_room_id": "kitchen"},
 "reason": null}
```

- `status`: `accepted` | `completed` | `failed` | `rejected`. `command_id`, `device_id`, `world_session_id`가 명령과 같아야 한다.
- `observed_state`는 스냅샷의 `state`와 같은 필드 이름(`observed_state`, `state` 아님).
- **`start_cleaning`**: 받으면 30초 안에 `accepted`, 청소 애니메이션이 끝나면 `completed`. `completed`가 와야 중앙이 방1을 끝난 방으로 기록한다(그 전까지 회차의 남은 방 `["room1"]`을 보존).
- **`return_to_dock`**: 받으면 30초 안에 `accepted` 또는 `completed`.
- **`set_*`**: 바로 `completed` (상태도 다음 스냅샷에 반영).
- 실패는 `failed` + `reason`. `busy`, `timeout`, `temporarily_unavailable`만 5분 뒤 다시 시도하고, 나머지와 `rejected`는 자동으로 다시 보내지 않는다.
