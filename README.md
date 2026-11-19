# EYEAR Orchestrator

Role-based agents and central decision-making for [EYEAR](https://github.com/HY-LG-EYEAR). **Maintainer: Jungbin Song**

The Orchestrator reads the home-wide state sent by the World Model (WorldSnapshot 2.0). Cleaning, Noise, Temperature, and Lighting agents make proposals, and the central decision layer turns them into appliance commands and Node speech (questions and replies).

> Status: **draft v0.1** — it runs and passes its tests, but may change as the team agrees on the remaining interfaces (contract v2.1, command and result routes, application integration).

## Flow

```text
World Model ──WorldSnapshot 2.0 (HTTP)──▶ 4 agents ──proposals──▶ central decision ──validation──▶ device commands
                                                                        │
Node ◀──text to speak (MQTT speak)── central ◀──replies and direct commands (MQTT voice/*)── Node
Device adapter ──execution result──▶ World Model ──(HTTP)──▶ central
```

**How the central layer decides**
1. All four agents read the same snapshot and make proposals. Agents never send commands themselves.
2. **Proposals that do not conflict are accepted as they are.**
3. A conflict is a different value proposed for the same device, or an action blocked by another agent's restriction. For each conflict, the options are built mechanically and **the LLM chooses one of them.** The LLM cannot create new commands.
4. If the LLM is unavailable or fails, a deterministic fallback chooses the option that respects the restrictions. A choice is reused for up to 15 minutes only while the same conflict continues and the related preferences, people, restrictions, routines, and deadline band are unchanged.
5. Every command is validated before it is sent: supported action and mode, temperature range and step, fresh device state, online state, manual override, user restrictions, and duplicates of commands still in progress.

Notes
- **Pause** stops automatic control only. A command the resident speaks directly ("하이 엘지, 불 켜줘") is still carried out. **Resume** decides again immediately if the last snapshot was received within 120 seconds; otherwise it waits for the next snapshot.
- The LLM is called outside the Orchestrator's lock. If a newer snapshot, a direct command, a pause/resume, or a voice reply arrives while waiting, the decision is recomputed on the newest state. If the Orchestrator was reset or the session changed, the pending decision is discarded.
- Snapshots older than the current `world_revision` of the same session are ignored. Freshness is judged by when a snapshot was received (live and replay alike), so recorded scenarios with past dates still work; the server checks the age of live `as_of` on receipt.
- A command stays pending until a final result (`completed`, `failed`, or `rejected`) arrives for the same session and device. `accepted` and in-progress work never expire by time. If not even `accepted` arrives within 30 seconds, value-setting commands (`set_*`) may be sent again after the next state check, while other commands (such as `start_cleaning`) are marked unknown and are not resent automatically. A cleaning completion is recorded on the routine occurrence the command was issued for.
- Failures with a known transient reason (`busy`, `timeout`, `temporarily_unavailable`) are retried after 5 minutes, at most twice, even if nothing else in the home changes. Rejections and other failures are not repeated automatically.
- A direct command is refused if the last snapshot is older than 120 seconds.
- Conflict detection currently covers different values for the same device and action, and cleaning rooms blocked by a noise restriction. Other combinations (for example `start_cleaning` vs. `return_to_dock`) are not yet detected.

Voice
- Replies to the sleep question count as yes or no only when they are short and clear ("응", "네", "아니", "응 잘게"). Questions, mixed answers, and "어" are treated as ambiguous, which leaves lights and curtains unchanged. Replies are accepted only from the Node and room the question was sent to, and only while the person is still in that room.
- A direct command is not carried out when it is negated ("불 켜지 마") or phrased as a question ("불 켜야 하나?", "불 켰어?"); the Node asks the resident to say it again. Direct commands need a recognition confidence of at least 0.4 (replies 0.2).
- Voice messages with the same `(node_id, event_id)` are handled once, live messages older than 30 seconds are ignored, and an MQTT message whose topic room differs from its `room_id` is dropped.
- The reply deadline is speaking time + `listen_seconds` + 2 s + 8 s allowance for a Node that queues the question behind another conversation.
- MQTT callbacks only validate and queue messages; a separate worker thread handles them, so waiting for the LLM does not block MQTT. Subscriptions and publishes use QoS 1, and failed publishes are logged.

## Values read from the World Model (per agent)

| Agent | WorldSnapshot fields read | Values remembered by the Orchestrator |
|---|---|---|
| Cleaning | `routines[]` (occurrence, room list = cleaning order, `due`, deadline), vacuum `state` / `capabilities` / `online` / `manual_override`, `rooms[].room_type` | Last cleaning time per room, occupied time since the last cleaning |
| Noise | `rooms[].occupancy`, `people[].activity` (label, confirmed), `profile_id`, `profiles[].quiet_rest/quiet_meeting`, `room_influence`, washer / vacuum / purifier state and mode | Devices whose mode was changed for quiet protection, and the rooms that caused it |
| Temperature | `people[]` in the AC's room (`profile_id`, `position`), `profiles[].preferred_min/max_c`, `rooms[].temperature_c`, AC `state` / `capabilities.temperature` / `airflow_target` | — |
| Lighting | `people[].activity` (resting, confirmed, stable_for_s), `rooms[].light`, light and curtain state, `nodes[].supports_speak` | Rest episodes and question state |

Field definitions are in [`orchestrator/contract.py`](orchestrator/contract.py), and a full example is [`examples/snapshot_t2.json`](examples/snapshot_t2.json).

## Agent models

All models use scikit-learn. Each model file is a few KB to about 200 KB and runs on a CPU. If a model file is missing, the agent uses its rule baseline.

| Agent | What the model predicts | Chosen model | Other candidates compared | How it is used |
|---|---|---|---|---|
| Cleaning | Cleaning need per room, 0–1 (trained on the simulator's hidden dust level; not a measured contamination value) | HistGradientBoosting regression | Ridge, rule | Evidence the LLM sees in a conflict (`urgency:room`). The cleaning order itself comes from the routine |
| Noise | Disturbance probability per (device mode, room) | Logistic regression | HistGradientBoosting, rule | 0.5 or higher → propose low-noise mode for the washer or purifier |
| Temperature | Thermal sensation (+ too warm / − too cool) per (person, set temperature) | Ridge regression | HistGradientBoosting, rule | Set temperature that minimizes the sum of squared sensations of the people in the room |
| Lighting | Sleep-intent probability of a resting person | Logistic regression | HistGradientBoosting, rule | 0.3 or higher → ask whether the person is going to sleep |

Training data comes from a household simulator ([`dataset/simulate.py`](dataset/simulate.py)). It builds the same WorldSnapshot objects and uses the same feature functions ([`orchestrator/features.py`](orchestrator/features.py)) as the running service, and labels come only from the simulator's hidden state. Rows are taken every 15 simulated minutes. Evaluation is a household-level split (train / test, no separate validation set), and the adopted candidate is fixed in `dataset/train.py`. Metrics and library versions are in [`models/model_card.json`](models/model_card.json). **These are simulated data, so the metrics do not indicate accuracy in a real home.** A model file that is broken or was trained with a different feature order is ignored, and the agent falls back to its rule.

## Output formats

| Message | Format | Example |
|---|---|---|
| Agent proposal | `AgentProposal`: actions, restrictions, questions, scores, reasons | `proposals` in [`decision_t1.json`](examples/scenario/decision_t1.json) |
| Decision record | `Decision`: proposals, accepted actions, conflicts / options / choices, commands, rejections, speech | [`examples/scenario/`](examples/scenario/) |
| Device command | `DeviceCommand`: command_id, device_id, action, params, reason_codes | Action names are the nine defined in the contract |
| Text to speak | `SpeakCommand` (same as eyear-node schema 0.1) | `"A님, 주무실 거면 '네'라고 말씀해 주세요."` (Korean speech) |
| LLM output | `LlmSelection`: choices[{conflict_id, option_id, reason}], summary | |

All JSON Schemas are available from the server at `GET /v1/schemas`. Definitions are in [`orchestrator/schemas.py`](orchestrator/schemas.py).

## Interfaces (draft, to be agreed)

| Direction | Route |
|---|---|
| World Model → Orchestrator | `POST /v2/world/sessions` (session registration) → `POST /v2/world/snapshots` (202 / 409 / 422) |
| Device result → Orchestrator | `POST /v2/world/device-results` |
| Node ↔ Orchestrator | MQTT `eyear/v0/{room_id}/speak`, `.../voice/reply`, `.../voice/command` (HTTP alternative: `POST /v1/voice/reply`, `/v1/voice/command`) |
| Orchestrator → device | MQTT `eyear/v0/device/{device_id}/command` (when `DEVICE_TRANSPORT=mqtt`; the default is mock) |
| App → Orchestrator | `POST /v1/control` {`action`: pause \| resume \| reset} |
| Orchestrator → app | `GET /v1/decisions`, `GET /v1/status`, `GET /v1/speech` |

## Running

Python 3.11 or later. Create the virtual environment in `.venv` inside the repository (it is excluded from git).

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -e ".[dev]"      # macOS/Linux: .venv/bin/python
```

Copy `.env.example` to `.env` and fill in `OPENAI_API_KEY` to enable LLM arbitration (default model `gpt-6-luna`, reasoning effort `low`). Without a key, only the deterministic choice is used.

```bash
.venv/Scripts/python -m orchestrator.demo                  # replay the demo scenario t0–t7 (no network)
.venv/Scripts/python -m orchestrator.demo --llm             # let the LLM choose in conflicts
.venv/Scripts/python -m orchestrator.demo --reply none      # no answer to the sleep question
.venv/Scripts/python -m dataset.train                       # generate data and train the 4 models (about 30 s)
.venv/Scripts/python -m orchestrator.server                 # http://127.0.0.1:8765, docs at /docs
.venv/Scripts/python -m pytest                              # tests
```

If the model files cannot be loaded because of a different scikit-learn version, recreate them with `python -m dataset.train`.

## Folders

```text
orchestrator/   contract.py (input contract)  schemas.py (output formats)  features.py  agents.py  central.py
                voice.py  llm.py  server.py  config.py  fixtures.py (snapshot builder, scenario)  demo.py
dataset/        simulate.py (household simulator)  train.py (training and evaluation)  samples/ (200 rows per table)
models/         the 4 trained models and model_card.json
examples/       snapshot_t2.json (contract example), scenario/ (t0–t7 input snapshots and decision records)
tests/
```

## Not decided yet

- Agreement on World Model contract v2.1 (session registration, device command and result routes, `stable_since`, etc.)
- Control and display formats exchanged with the app (Expo app, Spring Boot backend)
- Authentication for the HTTP API and MQTT (required before deploying beyond a local network)
- MQTT broker location for the demo, and where the server is deployed (local or cloud)
