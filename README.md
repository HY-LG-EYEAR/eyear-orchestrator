# EYEAR Agents and Home Orchestrator

Owner: **Jungbin Song** | Part of [EYEAR](https://github.com/HY-LG-EYEAR)

> Status: repository scaffold and implementation plan. This upload kit does not contain runnable services, approved new contracts, trained models, or measured results. Preserve existing implementation files when adding this scaffold.

## Responsibility

Implement Cleaning, Noise, Temperature, and Lighting agents; central arbitration; deterministic policy validation; plan/task state; voice-dialogue state; and inspectable decision traces.

Agents advise; only the central service approves commands. Direct commands take precedence over automatic plans without bypassing capability or authorization checks. Distinguish plan selection, dispatch acceptance and completed appliance work.

Use a shared input interface for live World Model and dummy scenarios. Keep the GPT provider behind an adapter; implement and label deterministic fallback/hold behavior. Model choice, new output schema, and lightweight learned scorers remain subject to evaluation. Do not describe the existing three-agent prototype as the completed four-agent system.

## Inputs and outputs

Consumes WorldSnapshot plus user directives/voice messages. Produces AgentProposal, validated decisions/commands, dialogue requests and traces. Reads actual execution feedback. Spring Boot owns app-facing persistence/APIs; it must not duplicate arbitration rules.

The authoritative shared contract release is maintained in [LG-EYEAR/contracts](https://github.com/HY-LG-EYEAR/LG-EYEAR/tree/main/contracts). Coordinate changes with producers and consumers; do not create an incompatible local contract under the same version.

## Structure

| Path | Intended content |
| --- | --- |
| `src/` | Agents, feature extraction, arbitration, policy checks, task/dialogue state, provider and device adapters, API and trace. |
| `datasets/generators/` | Agent-training generators using approved observed inputs; hidden simulator state may label outcomes but cannot leak into features. |
| `datasets/configs/` | Generation settings, seeds, persona distributions and split specifications with versions. |
| `datasets/samples/` | Small synthetic sample rows only; label their origin and contract version. |
| `training/` | Offline fitting and model export. Reuse runtime feature extraction. |
| `evaluation/` | Baseline comparison, held-out evaluation, error analysis and reproducible reports. |
| `tests/` | Unit and contract tests for state transitions, invalid commands, fallbacks and duplicate events. |

## Setup and execution status

This scaffold introduces no runtime dependencies or working launch commands. Preserve existing code. The owner must add the actual prerequisites, pinned dependency installation, configuration example, start/stop commands, and one verified smoke-test procedure before marking this repository runnable. Separate existing prototype instructions from the planned integrated system.

## Verification checklist

- [ ] Shared contract version and valid/invalid examples are identified.
- [ ] Dependencies and OS/hardware versions are recorded.
- [ ] A mock-input smoke test is reproducible.
- [ ] Missing, stale, unknown and invalid input behavior is tested.
- [ ] Integration with the next component is verified and exact commits are recorded.
- [ ] Current limitations and measured results are documented without invented metrics.

## Documentation

- [Integration and scenarios](https://github.com/HY-LG-EYEAR/LG-EYEAR)
- [CSE406 engineering document](https://github.com/HY-LG-EYEAR/LG-EYEAR-docs)
- [ITE351 research blog](https://github.com/HY-LG-EYEAR/LG-EYEAR-blog)

> Independent academic prototype. EYEAR is not an official LG Electronics product. The course prototype does not modify ThinQ ON firmware, integrate with the production ThinQ application, or control physical LG appliances.
