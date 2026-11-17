# M4.5 test data — capsule bin picking

| File | Used by | What |
|---|---|---|
| `capsule_bin_picking.yaml` | `test_m45_composition.py` | the M4.5 task program (M4_5_DESIGN.md §2, §7) |
| `capsule_bin_picking_mock.yaml` | `test_m45_providers.py`, `test_m45_mock_gate.py` | its mock variant: the scripted provider, the mock EEF outcome sequence |
| `capsule_bin_picking_isaac.yaml` | `test_m45_providers.py` | its Isaac variant: the YOLO-seg segmenter + SurfaceGrasp, the SurfaceGripper stage |
| `mock_gate/*.yaml` | `test_m45_mock_gate.py` | the scripted provider's scenarios, one per gate case |

## The mock product gate (`test/test_m45_mock_gate.py`)

The M4.5 mock gate runs the GENERATED application, not a hand-written tree: the test generates the mock variant
(public UR5e sample cell) into a temp workspace, once per bundle variant, builds it, brings it up on mock and runs
it once per case of the recovery table (M4_5_DESIGN.md §11). Nothing in a generated bundle is edited.

What varies, and where:

- the **scenario** (`mock_gate/<case>.yaml`) -- a node parameter: the test copies it to the absolute path the
  bundle's `perception.yaml` names before each run, so the cases of one bundle share a build;
- the mock EEF's **outcome sequence** (`adapters.mock.outcomes`) and the provider's **`after_end`** -- generated
  into the bundle, so they define the variants: `held`/`empty`, `held`/`repeat_last`, `not_held,held`/`empty`,
  `not_held`/`repeat_last`, `lost_after`/`empty`, `lost_after,held`/`empty`, `unknown`/`empty`, and `held`/`empty`
  with the place waypoint moved out of reach (case 11).

The fixtures use the recommended cycle shape (TSA 8.6.1): `repeat until nothing_to_pick { retry 2 { retry 3 {
bin_view, detect, grasp index all + retreat + payload }, release + retreat } }` -- every bound is the outer
retry's 2 times the inner one's.

Each case asserts the `BT finished` status, the runner's exit code, the EXACT ordered list of the runner's events
(detections, candidates tried, moves, plan failures, engages, verifications, attach/detach, the program's own end
lines) -- so every count (triggers, detections, attempts, engages) is bounded and checked -- a wall bound per case,
that the runner launch ends by itself with its tree (TSA 8.6.1), clean shutdown and an empty ROS domain before
and after.

| Case | Scenario / outcomes | Expected |
|---|---|---|
| 1 success | one capsule, then empty / held | pick, place, 2 x 3 empty detections, SUCCESS, exit 0 |
| 2 no candidate | empty, one capsule / held | re-detect, pick, place, SUCCESS |
| 3 perception unavailable | `unavailable` for ever | DETECTION_TIMEOUT fault (the producer's refusal in its cause), FAILURE |
| 3b provider absent | no scenario file: the provider refuses to start | TRIGGER_UNAVAILABLE fault, FAILURE |
| 4 planning failure | [2 m away, reachable] / held | the first fails to plan, the second is picked, SUCCESS |
| 5 grasp verification | [a, b] / not_held, held | a NOT held -> retreat -> b held -> place, SUCCESS |
| 6 stale | stale, fresh / held | the stale array rejected, re-trigger, SUCCESS |
| 6b stale for ever | stale for ever | DETECTION_TIMEOUT fault, FAILURE |
| 7 retry exhaustion | [a, b, c] for ever / not_held | exactly 2 x 3 x 3 = 18 attempts, "recovery exhausted", FAILURE |
| 8 empty bin | empty for ever | 2 x 3 = 6 detections, then the presence capture finds nothing: BIN_EMPTY, SUCCESS, no motion beyond bin_view |
| 9 lost before release | a, then b / lost_after, held | OBJECT_LOST at the place -> open, detach, back to the PreRelease -> the cycle again: b placed, SUCCESS |
| 9b lost every time | a, then b / lost_after | the same retreat on both runs of the cycle, "recovery exhausted", FAILURE |
| 10 terminal fault | a, then b / held, unknown | a placed; b's VerifyGrasp STATE_UNKNOWN = an EEF fault at the second engage, never retried, FAILURE (TSA 8.6.2: with `unknown` from the start the start check stops the run first -- case 12) |
| 11 place unreachable | one capsule / held, the place 2 m away | the PreRelease has no IK solution while holding: the fault PLACE_UNREACHABLE_WHILE_HOLDING, the tool never opened, FAILURE |
| 12 unknown at start | (none reached) / unknown | TSA 8.6.2: the start check's VerifyGrasp released reads STATE_UNKNOWN: the tool is not proven empty, the run stops before any motion with that reason, FAILURE |
| 13 pickable exhausted | a, then no candidate with one object present for ever / held | TSA 8.7.0: a placed, 2 x 3 fresh empty detections, the presence capture finds the object: PICKABLE_EXHAUSTED, FAILURE (exit 1), not a fault |

TSA 8.7.0 (M4.5b): the fixtures carry `presence: capsules`, and every case also asserts its ONE `[BT] OUTCOME: <code>`
line: BIN_EMPTY (1, 2, 4, 5, 6, 8, 9), PICKABLE_EXHAUSTED (13), RECOVERY_EXHAUSTED (7, 9b), TERMINAL_FAULT (3, 3b,
6b, 10, 11, 12).

Not expressible on mock (recorded in the test's docstring): a failed `VerifyGrasp released` (the mock always reads
not held after an applied release) and an injected actuator fault (`mock.actuators.<a>.fail` is not generated by TSA).

Run it (opt-in; about 10 minutes on the reference machine: one build of the 8 variants, then 15 cases of
25-90 s each; ROS domain 75):

```bash
source /opt/ros/humble/setup.bash && source <the TrainIt workspace>/install/setup.bash
TRAINIT_LAUNCH_TEST=1 python3 -m pytest -p no:cacheprovider test/test_m45_mock_gate.py -v
# optional: TRAINIT_M45_GATE_EVIDENCE=<dir> keeps each case's report and launch logs
```
