# TrainIt Setup Assistant (v7) — full test procedure

All milestones (M0–M9) are complete. This walks you through testing the whole MVP,
from the headless generator to the GUI wizard with the live RViz gizmo.

## Key concept: robot vs scene

- **Robot + end-effector → URDF / xacro.** The assistant needs the *kinematic model*
  to build the moveit_config. For FR3WML that is `fr3wml_suction.urdf.xacro`. You do
  **not** load the robot from a `.usd`.
- **Scene (collision objects) → primitives OR a `.usd`.** The USD importer turns each
  geometry prim into a box collision object (its world AABB). So you *can* feed the
  `.usd` you use to spawn the Isaac scene — but only for the **scene**, not the robot.

Nothing to clone: the engine + bridges are already in `~/fr5_ws`.

## Two ways to start the wizard (S0) — read this first

The **S0 — Load robot** page offers two paths:

1. **Open project.yaml (recommended for FR3WML).** Starts from a *complete* project that
   already carries the **self-collision matrix** and the **cell bridges** (suction
   bridge, joint_state_merger). Use `examples/fr3wml_project.yaml`. The generated bundle
   runs end-to-end out of the box.
2. **Load a robot xacro (a fresh bootstrap, for a brand-new robot).** Auto-detects the
   group/frames/EEF, but does **NOT** auto-fill two cell-specific things:
   - the **self-collision matrix** (`collisions_updater` can't be run reliably from the
     tool here) — without it the SRDF is over-conservative and *planning fails*;
   - the **cell bridges** (sim/hardware gripper plumbing) — `mode:=mock` then has no
     `/joint_states` and the gripper action has no server.
   For a new robot you add these afterwards (compute the matrix with the real MoveIt
   Setup Assistant / `collisions_updater` once and paste the `disable_collisions` into
   the project's `robot.disable_collisions`; copy the deployment block from
   `examples/fr3wml_project.yaml` and adapt the bridge package names). This is a known
   MVP limitation — the bootstrap generates a correct *robot + app + moveit_config*, but
   the cell deployment + matrix are robot/cell data it can't invent.

> If a generated bundle fails with `PLAN FAILED (code=-31)` on the first move, or RViz
> says **"Frame [base_link] does not exist"**, you used path 2 without the matrix/bridges.
> Use path 1 (Open project) for FR3WML.

```bash
cd ~/fr5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select trainit_setup_assistant --symlink-install
source install/setup.bash
```

---

## Test A — prove the MVP equals the golden (headless, ~1 min)

```bash
ros2 run trainit_setup_assistant trainit_verify \
  ~/fr5_ws/src/trainit_setup_assistant/examples/fr3wml_project.yaml \
  --golden ~/fr5_ws/src/fr3wml_digital_twin/fr3wml_suction_camera_tsa_4_1_bundle --build
```
**Expect:** `21/21 checks passed`, `EQUIVALENCE: PASS`, `BUILD: PASS`.

Optional unit tests:
```bash
cd ~/fr5_ws/src/trainit_setup_assistant
QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
  python3 -m pytest test/ -q          # expect: 295 passed (plus 1 pre-existing network-USD failure)
```

The golden-dependent vNext M2.6 gates (`test_m2_6_eef_*.py`) skip on a checkout without
`fr3wml_digital_twin`, naming the missing file; with `TRAINIT_REQUIRE_GOLDEN=1` they FAIL
instead, so a CI job that must run them cannot report them green unrun.

The two gates that must never go red without a deliberate re-blessing:
`test_generator_golden.py` (the published bundle regenerates) and
`test_backend_bytestability.py` (the mode-dependent launch files, byte-strict). A new
per-move or per-cell option is only correctly gated when BOTH still pass with **no
regeneration and no re-blessing**; if one goes red, the gating is wrong — fix the gating,
never the golden. `test_golden_full_tree.py` holds the whole tree: every non-manifest file byte
for byte, and the manifest's CONTENT record (each file's path, action, sha256 and source, the
warnings); its PROVENANCE (`generator_version`, `capability_snapshot`, vNext M2.3) changes with
every release and catalog refresh and is checked against the committed catalog by
`test_bundle_provenance.py`, never by re-blessing as a side effect of a release or a refresh. A
provenance-only update of the golden is made only with the owner's approval, after the owner has
seen the exact diff (vNext G0, 2026-10-13: `generator_version` 7.6.0 -> 7.8.0, `catalog_sha256`,
the recorded TMR version 1.8.0 -> 2.1.0). Between a generator change and its owner-approved
re-bless, a pinned reference's gate allows exactly the pending lines of `test/pending_rebless.py`
(owner Q1, vNext M2.4) -- each substitution must apply exactly once, and any other difference
fails; the list is empty since the M2.4 re-bless (2026-10-08). Three files pin the other half, one
per option:

| file | what it pins |
|---|---|
| `test_execution_mode.py` | ADR-0015 `execution: plan \| servo` — what `servo` emits (a `<ServoTrack>` node, `config/servo.yaml`, a `servo_node` with all four `robot_description*` params), and that `plan` emits **nothing** |
| `test_cell_namespace.py` | ADR-0016 `cell_namespace` — that an empty one emits no namespace token in any launch or `.rviz`, and that a set one reaches `move_group`, the controllers (**including** the spawners' `--controller-manager`), `robot_state_publisher`, `scene_manager_node`, `servo_node`, `trainit_run_bt` and the gripper action, while `/tf`, the camera, `/perception/**` and RViz stay at the root. Read with `ast`, per node: a substring search cannot tell which node a `namespace=` belongs to, and "which node" is the whole question |
| `test_bin_picking.py` | ADR-0016 `bin_picking` — the tree's SHAPE (`Repeat` → `Fallback` → the pick, with `DetectObject` **outside** the retry), the retry count (`keep` == `num_attempts` == `tries`), the `Script`-advanced candidate index, the holding check closing each attempt, and the gated recovery |
| `test_external_node_detector.py` | D-015 the **external-node** detector's launch half — that a project with no `external_node` block regenerates the published launch file and `package.xml` **byte-for-byte**, that one bundle launches a pure-python `detector_node` and the project's own node **side by side and both global**, that the launch entry and `perception.yaml` resolve the camera contract **once** (so they cannot name different cameras), the five refusals (no package, no executable, a block on a pure-python method, an unknown input role, two detectors on one node name), that an external method with **no** block is a warning and not a refusal, that the external package is recorded in a **comment** and never as an `<exec_depend>` (ADR-0003), and that a real detector's shape (`seg_pose_node` + a parameter file keyed by it + `model:=`) is expressible and round-trips through `project.yaml` — the check against the REAL package lives in the program that owns it |
| `test_candidate_rule.py` | ADR-0017 the candidate ranking — that an **unset rule writes no key at all** (the fact the two gates above depend on, asserted directly), that a rule whose TERMS are not computable is **refused before a single file is written** for either reason (the METHOD does not produce what the term reads — the colour mask runs no 3D fit — or the PARAMETER is unset) with the message naming the missing term *and* what would supply it, that the three numbers come from their three homes (detector / end-effector / **derived** from the grasp target), that the GUI greys what it cannot run, and that a ranked bundle pins `trainit_perception >= 0.3.0` |
| `test_grasp_face_picker.py` | ADR-0018 the MEASURED grasp face — that the picker **recovers a known radius** on a synthetic disc generated inside the test (asserted against the inscribed polygon's exact equal-area radius, not against the circle, because the mesh *is* the polygon), that the flood fill takes the cap and not the parallel face 25 mm behind it, that a click is a lookup in the raster's own id buffer (so the mouse path cannot disagree with the ray path) and that the flipped view reaches the other three sides, that a millimetre file read as metres is **refused** rather than written (the silent failure of Decision 4) and that the refused band is the RUNTIME's own, that STEP is **probed** and never required (including the AppImage dispatch) and that an exact B-rep face refuses to match a tessellation whose centre, normal or area disagree, that TSA **reads the face registry from its catalog snapshot** — a `blob` face family that exists nowhere in TSA is rendered, written, pinned and refused by name, with a snapshot stood in so the test holds whatever `trainit_perception` is installed — and that an unknown face is refused even with no perception package installed (ADR-0021, M2.2), plus the dialog driven offscreen and the real capsule (skipped unless its private CAD and a headless FreeCAD are both present) |
| `test_tp_registry.py` | ADR-0021 amendment M2.2 — TSA's snapshot of trainit_perception's vocabulary (`catalog/tp_registry_snapshot.yaml`) is TP's live `vocabulary.export()` entry by entry, and every kind of drift is NAMED (committed mutations: a rule added or removed, a term renamed, a requirement, a default, an allowed value, a face and its parameter order, a floor, the schema, a classification); the snapshot's canonical bytes, the sha256 `snapshot.yaml` records, the closed schema and the boundary (names, schema, requirements -- no prose, no gate value); no candidate provider's name in the vocabulary or in what a project can select; every rule and method TSA's code names is required (AST scan) and `DetectionMethod` agrees with the snapshot both ways; TSA's computability equals TP's `computable_terms` in every configuration; the wizard's lists and colour-mask form are the snapshot's; the refresh refuses (exit 2, bytes/mtime/mode unchanged) an unclassified entry, a removal, a schema, a token TSA names going away, a release mismatch, writes all three files or none, and leaves no model built from a candidate behind it |
| `test_perception_vocabulary_pins.py` | vNext M2.2 tests-first — TSA's perception vocabulary as 7.3.0 exposed it, pinned BEFORE it was derived and not edited after: `DetectionMethod`, every 7.3.0 rule's name/value/order, the default preference, what each method provides, each rule's terms, computability over a grid, what generation writes for 26 cases (`m2_2_cases.py`: the detector's params IN ORDER and the TP pin, or the refusal word for word), and the GUI's lists and colour-mask form |
| `test_bundle_provenance.py` | vNext M2.3 (ADR-0021) — the TSA under test is this source tree; every manifest (the bundle's and Step 3's) records `capability_snapshot` right after `generator_version`, equal to the committed catalog's identity as the test computes it (the sha256 of the three catalog files, the versions and schemas `snapshot.yaml` records), a closed block that adds no dependency; one changed snapshot byte changes the identity and NOTHING else of the bundle; a catalog changed behind a running TSA (a registry, snapshot.yaml, a refresh in-process, another release, a CRLF checkout, a hand-edited label) is refused before any file; a COPY is named `package://<package>/<path>` (nearest package.xml, its `<name>`, innermost, the file as the project names it -- a symlink is not resolved) or `external:<file name>`, a directory with its own package.xml, TSA's own resources the same however TSA is installed and in a generated Gazebo cell, a policy card and its models (synthetic, and the owner's project when present); an unusable package.xml is one manifest warning; no manifest (the written file) names this machine, a candidate provider, or a path in a warning -- while the user still sees the configured paths (`local_warnings`: the CLI prints them, the GUI shows them); an empty or `.` input path is written as given, never as the cwd; a bridge's path parameter (a file, a directory, a device) is not quoted as a namespace ROOT topic |
| `test_generation_environment.py` | vNext M2.3 (T4) — three projects generated in fresh interpreters under eleven environment variants (another seed, cwd with decoy packages and an odd TMPDIR, HOME moved, `LC_ALL=C` and `C.UTF-8`, TSA and the inputs through symlinks, TP and TMR uninstalled, FAKE TP/TMR first on every path -- a planner registry TSA could read, another version, packages that raise -- no ROS environment) give the baseline's tree byte for byte, the manifest included, which records the committed snapshot; nothing is imported from or opened in TP, TMR, the ament index or the fake prefix, nothing is written outside the bundle; a positive control proves the observers see each of those; each variant is checked to have done what it claims (skips, naming why, without TP or on a merged install); two non-UTF-8 locales are strict xfails (known debt: locale-encoded text reads) |
| `test_generation_identity.py` | the M2.2 identity proof kept as a regression test — **opt-in** (`TRAINIT_IDENTITY_TEST=1`, Test A2c) |
| `test_runtime_requirements.py` | vNext M2.4 (ADR-0021 amendment M2.4) — what a bundle declares it needs to RUN: the TMR floors table pinned as curated from TMR's history (each row's version and commit) and sane (between TMR's first package.xml version and the one the snapshot was derived from; XML-comment-safe reasons; versions compared as numbers); the CLOSED WORLD -- every BT node, port, bt_params key, waypoint key, scene.yaml key, launch-helper argument, TMR executable, import and runner argument the generated bundles carry is one the file has dated, and the values of TSA's own vocabularies are the dated ones (a new one fails and needs its floor decided); an undated node stops generation (also through the Orchestrator), so does a non-numeric `keep`, a launch-helper call with `**kwargs`, or a TMR node the floor cannot read; the floor read from a package's own files (each node, port and value, keep <= 1, several trees, each scene node only when `trainit_run_bt` runs under a non-empty `namespace=`, `backend=` read as a call, a namespaced `scene_manager_node`, a canonical package's namespaced `trainit_run_bt` at 2.1.0 and nothing else moving for it -- the same package at the root, a legacy one namespaced or not, a legacy file that does not parse (vNext G0), the gripper targets, the comment's reasons in table order); a file that does not parse is named with its error kind and line and assumes every dated capability; the application matrix generated in full -- 31 cases: motion only (from scratch and scene loader), colour-mask vision, an external node, Pure / Hybrid / Residual policies with and without vision and on the scene loader, a policy whose card is missing, anchored nowhere, or inside bin picking, servo, gazebo, namespaces (with ResetScene), bin picking (and one try), a joint-position gripper, an unparseable tree, and the end-effector contract's canonical cases (the golden opted in, at the root and under /r1, canonical bin picking at 2.1.0 for its motion gate, a GripperCommand jaw commanded to a third target, the mock-only composite BUNDLE in /r1, a simple suction cell at the root and in /r3 -- each canonical runner in a cell namespace at 2.1.0 since vNext G0, its scene-loader config unchanged) -- every TrainIt dependency line of every package exactly, the hardware layer only in the package whose launch starts the adapters; the CLOSED WORLD OF DEPENDENCIES (vNext M2.7, brief section 12): every TrainIt package a generated package's own files use -- a launched node, a share lookup, a Python import, a BRIDGES row, a xacro `$(find)`, a `package://` URI, a tree or bt_params (TMR's runner; a RunPolicy step, the policy runtime) -- is one its package.xml declares at run time, a package expression the scan cannot read failing it (closed); the scan's own rules pinned on synthetic files, and its reach over the matrix (TMR, TP, the policy runtime and the hardware layer, never trainit_interfaces); the policy runtime declared iff a generated file uses it; no component a build dependency, no TMR dependency unversioned, catkin_pkg reads every `version_gte`; residual still warns; the Step-3 config equals the bundle's config; a scene-loader cell built from a synthetic base config (no golden); the negative tests (a snapshot capability, an NVIDIA candidate -- by dependency, launched node, share lookup, $(find), package:// or find_package --, an unused policy runtime, TP's snapshot release are not dependencies; the requirement module imports only its catalog constants; FAKE TMR 9.9.9 / TP 0.0.1 / policy runtime 9.9.9 first on every path, or all three uninstalled, give the same `package.xml` files, the policy bundle's included, each environment checked to be what it claims, in fresh interpreters) |
| `test_m2_6_eef_contract.py` | vNext M2.6 (ADR-0022) — the end-effector contract's model: F1 representability (the composite fixture's 6 actuators on 3 interfaces, implied capabilities, 2 sites; the mimic followers refused; the three short forms and their derivations, no 1.0/0.0 fallback); the short form stored as written and the long form derived on demand; default keys dropped inside the contract, `default_factory` fields included (the same bytes on pydantic 1.x and 2.x); a duplicate key inside the contract refused at load (R3), a legacy file loading as before; the guard token from ONE pre-validator (an overridden token named by validate), R19 at load; R2 at load (`extra='forbid'`, closed tokens); the new tokens are ones TSA 7.6.0 refuses; legacy projects save byte-identical to TSA 7.6.0 (pinned hashes); the shared conformance files pinned by sha256, TSA's emitted `eef` blocks normalising to the pinned dump, TSA refusing what the reference normaliser refuses (a swapped set_bool site, equal GripperCommand target values), the token parity; the catalog's implementation rows (no kind, no composite row), the corrected sim-grasp rows, the legacy-bridge marks as data; every identity project meets no refusal (counted: one that cannot be built here makes the test skip, naming how many were checked) |
| `test_m2_6_eef_refusals.py` | vNext M2.6 (ADR-0022) — F2: every refusal row R1-R26 by CONDITION (146 cases; vNext M2.7 inverted the M2.6 R22 pin -- the composite mock bundle generates -- into six named R22 conditions and added R24, R25 and R26), each refused from BOTH `Orchestrator.generate` and `generate_scene_loader_config` before a file exists, the row named; a meta-test traces the cases and requires every refusal site of `generator/eef_contract.py` (each `_prefix(N)` line) to be reached; the gazebo flavour read from the base config feeding R7, R15 and R16 alike; a relative (or namespaced) stage topic accepted, an absolute one under a cell namespace refused; a valid simple contract generates from both; `none` allowed for a simple unverified device; G0: every case's sentence says UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL exactly when the case is a limit of the current canonical model (`LIMITS`), four R24/R25 shapes added (a planar joint an actuator drives, a jaw whose TCP is on its finger link, a TCP below the command_joint pseudo-joint, a follower whose master is missing), the echo sentence's multi-site clause only on several sites, and each echo, jaw and pseudo-joint remedy clearing its refusal |
| `test_m2_6_eef_emission.py` | vNext M2.6 (ADR-0022) — F3: the golden opted in differs from the legacy golden ONLY in the classified files, and EVERY changed LINE matches an explicit per-class pattern ([C] the eef block, the trees' eef ports, scene.yaml's scene signal; [A] BRIDGES = the kept rows + the adapter rows, no mock server; [D] the hardware-layer and TMR floor lines, a removed legacy bridge's package that nothing in its package uses; [P] the manifest's verified records) -- anything else fails; a canonical package declares every package its scene.yaml loads meshes from; it carries the contract; a namespaced isaac cell (relative stage and reset topics, the stage warning on the resolved names); the reset topic always written with an isaac binding; the mock row's sample rate for a short max_age_ms; YAML 1.1 bool/null words quoted as keys; the Gazebo weld follows the scene signal; the composite handoff tree, ids only; two robots with identical ids resolve to disjoint names in /r1, /r2, the root and /r3 (through the adapter rows' remappings: a composite row's tool joints on the robot's own joint_states, a simple row's on `eef/<id>/joint_states`); the Pure policy bracket and the card's Trigger pair; CommandActuator; no legacy key, no empty sequence, floats that load as floats; F5 pose relevance; F7 the composite boundary tripwire (a composite mock bundle, no bypass: no tool joint, controller, SRDF group or named state); the namespaced isaac cell refused by R26 while the golden's merger input is absolute, generated once it is relative |
| `test_m2_7_eef_composite.py` | vNext M2.7 (owner Q1 option A, Q7-i E1+E2, Q7-ii, 11.1) — the composite MOCK bundle generates from both entry points (the conformance device and the section-13 tool, root and /r1; a scene-loader composite keeps its base block); R20 unchanged and R22 naming vNext M6 beyond mock; a base `<ros2_control>` block claiming a tool joint (or a mimic follower) refused, a base description that cannot be compiled refused (fail closed); the composite mock row's one remap, no remap on a simple row; the scene-loader SRDF merge filters as `srdf_builder`; R24/R25 on the preflight's cases B, C, D, F, G, H with joint, actuator, interface expected and received and the explanation each gets (an axis: the owner's E1; a grip on the path: R8's half and the jaw reading, a limit of the model), and every valid contract (a SetBool jaw off the TCP path included) passing; a mimic follower judged by its master, its remedy naming the master (listing the follower is R5's refusal and R25 still fires); a follower of the command_joint pseudo-joint a limit of the model; R26 on absolute, namespaced-absolute and private echo topics, never on a legacy project; the section-13 block normalising with the pinned reference and its bundle's launch-safe config |
| `test_m2_7_eef_fixture.py` | vNext M2.7 — the fixture the M2.7 launch tests build on (`test/eef_fixture.py`), pinned: launch-safe is an OPT-IN of the M2.6 builders (by default their projects -- and so every file generated from them and compared byte for byte elsewhere -- are unchanged; `launch_safe=True` moves exactly the three launch-safe settings: `initial_positions`, the per-joint limits, the OMPL projection); `composite_mock` is launch-safe for its own prefix in the root, /r1 and /r2 cells; the section-13 description is the tree its contract names (every actuator joint movable, both site TCPs below both axes, the jaws side branches, the lock jointless), byte for byte the one the M2.7 BEFORE launch capture used. ROS-free except xacro |
| `test_m2_7_planner.py` | vNext M2.7 (owner decision D3), scoped at vNext G0 (owner section 8: canonical projects only, a legacy project unchanged) — in a CANONICAL project (one with an end-effector contract) a blank planner has ONE meaning: `model.effective_planner` is the planner TMR's `MoveWaypoint` runs (a chosen one unchanged; a blank `ptp` -> `pilz`; a blank `free` move, or a waypoint with no segment -> the project's global planner mode, the token the runner's `planner_mode` launch argument defaults to; `lin`/`circ` -> none, TMR reads none), and generation WRITES it into `bt_params.yaml`: eef_fixture's synthetic canonical suction cell (no vision-bound move) with blank planners generates, byte for byte, the bundle of the same cell with each effective planner set explicitly, under every planner mode; a streamed move carries the same block; D-016 judges the effective planner (a blank `ptp` under global `ompl`/`ompl_chomp` is flagged, with one added clause, and every case flagged before keeps its message byte for byte) and so does the blocks page's switch to free/OMPL when a detector first takes a move over (Qt offscreen). A LEGACY project keeps the reading it had before M2.7 at all three places, pinned case by case: no planner is written for a blank move (a chosen one as it is), and D-016 and the switch read a blank planner as the global mode (`test_vision_application.py`'s v4.1 pin, inverted by M2.7, is restored). TMR is not changed |
| `test_m2_7_motion_fault.py` | vNext M2.7 (owner Q8.2, ADR-0022) — the canonical bin-picking cycle's MOTION gate: inside the cycle's absorbing Fallback every `MoveWaypoint` (the candidate moves in the retry and the moves after it) remaps TMR 2.1.0's optional `fault` output into the end-effector gate key, with grasp and release confirmation on or off, and the gate expressions unchanged; the prologue is not gated; a streamed `ServoTrack` move has no such port and gets a validation note naming it -- only inside the cycle, and never for a project without a loop detector (no cycle there; `validate()` does not raise); legacy trees and the other canonical applications carry no `fault` on any move; the pinned M2.7 tree (`test/data/m2_7_mf/bin_picking_canonical.xml`, a byte copy of TMR's) is what generation writes, and it differs from the frozen M2.6 tree by exactly the five intended lines; of these trees only the canonical bin-picking one gets `trainit_motion_runtime >= 2.1.0` for its motion gate (the M2.6 tree 2.0.0, the legacy loop 1.6.0); a canonical runner in a cell namespace gets 2.1.0 for its own reason (vNext G0, `test_runtime_requirements.py`). ROS-free; needs the vision golden |
| `test_m2_7_capsule_readiness.py` | vNext M2.7 (the capsule bin-picking readiness matrix, P1) — the TSA facts behind the M4.5 readiness rows, on a capsule-shaped canonical project (the vision golden opted in to the 3-line SetBool suction contract, `bin_picking`, the grasp move guided by an external-node detector of the real shape with `orientation: detected`, grasp confirmation as the holding check): it GENERATES (one DetectObject per cycle, keep = the retry budget and its `fault` to the gate; the candidate's SetWaypointFromDetection inside the retry; no SetReleasePolicy; the external node launched and its package NOT declared, ADR-0003); Isaac grasp confirmation needs the stage's status topic (without it R16 alone refuses; with it the device's VerifyGrasp closes each attempt inside the retry); the reference cell's `mode:=real` brings up nothing that moves (no `real_include`, no bridge row in real, the hardware-layer adapters on mock and isaac only); a second detector's explicit Detect block, which the loop emits nowhere, is REFUSED in this canonical project before any file (vNext G0, owner G0 section 6: `test_a_second_detectors_detect_block_is_refused_in_a_canonical_project`, the exact sentence from `refusals()`, `validate()`, `Orchestrator.generate`, `trainit_generate` (exit 1) and the GUI controller, one sentence per dropped block), while the same detector with no block (sampled once per cycle before the retry), the loop detector's own block (a note) and a project with no loop (the linear tree emits the block at its anchor) are not (`test_only_a_block_the_loop_would_drop_is_refused`); and one KNOWN_DEBT pin as a STRICT xfail, narrowed to a LEGACY project in vNext G0, `test_a_legacy_projects_second_detectors_detect_block_is_emitted_by_bin_picking`: legacy `bin_picking` still drops that block silently (and with it a step-relative `post_place` derivation) -- the day the M4.5 template work emits the block the test XPASSes and fails, and it is flipped then, the canonical refusal lifted with it (`raises=AssertionError`: a refusal, or a missing golden under `TRAINIT_REQUIRE_GOLDEN=1`, fails it instead). Pins, not fixes, except the G0 refusal; ROS-free; needs the vision golden |
| `test_reference_real_mode_contract.py` | vNext M0D T4 — `mode:=real` on the reference cell as a CONTRACT, evaluated and never run (no process starts, nothing connects to a robot): the action names MoveIt is configured with are the ones the FAIRINO bridges serve; as generated, `mode:=real` starts nothing that serves them (no `real_include`, no bridge in real, no ros2_control: it can plan and never move); the bridge is DEVELOPMENT / NON-PRODUCTION-READY. vNext M2.7 (P2, the capsule readiness matrix) adds `test_the_fairino_bridge_limits_are_the_recorded_ones`, read from the bridge sources: the FollowJointTrajectory server executes only the last waypoint and never reads the goal's joint names, moves at its own 95 % speed, resets controller errors before every trajectory and holds the motion mutex across a cancel, and the suction shim awaits asyncio inside its executor -- DOCUMENTED, not fixed: the test fails the day the bridge changes, so the record is revisited rather than inherited. Each fact skips when its inputs (the bridge sources, the golden, the installed reference config) are absent |
| `test_m2_6_eef_bin_picking.py` | vNext M2.6 (ADR-0022 D5) — F4: the canonical bin-picking gate walked statically (the reset Script first, `fault` inside the retry and `reason` after it, DetectObject's `fault`, `holding` set and cleared, `_failureIf` on the retried child and the stall branch, VerifyGrasp in place of CheckRobotState), the skeleton equal to the legacy loop's, and the legacy loop byte-identical to TSA 7.6.0 (pinned hashes) |
| `test_m2_6_eef_gui_guard.py` | vNext M2.6 (ADR-0022 D6) — the GUI guard: the Step-7 setters and the Step-2 re-ingest leave a canonical gripper (kind, actuation, controller_*, contract) unchanged through save and reload; a legacy gripper is edited and reclassified as before; Step 7's block editor (offscreen, on synthetic canonical projects) keeps every `actuate` action and every grasp/release `site` through page entry, an untouched Apply and save, and offers no legacy attached topic to a canonical policy block |
| `test_m2_2_regressions.py` | vNext M2.2 — the behaviours it delivers, written as strict-xfail on 7.3.0: the `edge_on` rules accepted where computable and refused for the true reason on the colour mask; the same project generating the same bundle (or refusal) with trainit_perception importable, made unimportable, and uninstalled (off `PYTHONPATH` and the ament index), in fresh interpreters; generation importing no TP module, opening no TP data or release-metadata file (pydantic's plugin scan of every distribution's `entry_points.txt` excepted) and asking the ament index nothing about it; a refresh whose derive step fails writing nothing |

### Test A2 — the namespaced cell, and the loop (v7, ~2 min, headless)

Two things ADR-0016 added that are worth seeing with your own eyes before trusting them.

```bash
cd ~/fr5_ws/src/trainit_setup_assistant
QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
  python3 -m pytest test/test_cell_namespace.py test/test_bin_picking.py \
                    test/test_generator_golden.py test/test_backend_bytestability.py -q
```

**Expect:** all pass. The last two are the byte gates and they must pass **unchanged** —
that is the acceptance criterion of the whole ADR, not a side effect.

To look at a namespaced bring-up by hand, set `robot.cell_namespace: r3` in a copy of
`examples/fr3wml_project.yaml` and run `trainit_generate` on it:

```bash
ros2 run trainit_setup_assistant trainit_generate /tmp/ns_project.yaml -o /tmp/ns_bundle
grep -n 'NS = \|namespace=NS\|tf_global\|controller-manager' \
  /tmp/ns_bundle/*_app/launch/bringup.launch.py
cat /tmp/ns_bundle/generation_manifest.yaml | grep -A2 cell_namespace
```
**Expect:** `NS = "/r3"`; `namespace=NS` on `robot_state_publisher`, `move_group`, the
`ros2_control_node` and the spawners; `tf_global` remapped on each of them; the spawners
asking for `f"{NS}/controller_manager"`; **no** `namespace=` on `rviz2` or on the
`depth_image_proc` bridge. And warnings naming what the namespace does NOT fix — the
captured bridge parameters and the arm `/joint_states` remap, which are absolute ROOT
names and never follow a namespace. Those are cell DATA: the generator reports them rather
than rewriting them behind your back.

> **A namespaced cell brought up from a GENERATED bundle**: verified from scratch by Test
> A2b (2026-10-06). The namespaced shape was first learnt on a hand-made one that runs (a
> real program's namespaced moveit_config: the tf remap, the root-level RViz and the
> scene-manager wiring). A scene-loader bundle still needs its base config's
> `ros2_controllers.yaml` keyed `/**/<name>` (see A2b); bringing one up in Isaac needs the GPU.

### Test A2b — a GENERATED bundle, built, brought up, planning and moving (vNext M0D T2, ~20 s, no GPU)

The unit tests read generated files; this one hands them to colcon and to ROS. For each of two
bundles -- the reference cell's project at the root (scene-loader flow) and the from-scratch
example under `cell_namespace: r3` -- it generates into a temp dir, `colcon build`s it over the
sourced workspace, runs `ros2 launch <app> bringup.launch.py mode:=mock rviz:=false bt:=false`
on its own `ROS_DOMAIN_ID`, and checks: the motion stack's nodes are up (in `/r3` for the second);
the planning scene holds exactly the objects of the generated `scene.yaml`; the arm controller is
ACTIVE; `plan_kinematic_path` plans to a named SRDF state and `execute_trajectory` executes it on
the mock controllers, and `joint_states` arrive within 2e-2 rad; SIGINT leaves no process behind.

```bash
source /opt/ros/humble/setup.bash && source ~/fr5_ws/install/setup.bash
cd ~/fr5_ws/src/trainit_setup_assistant
TRAINIT_LAUNCH_TEST=1 python3 -m pytest -q -s test/test_generated_bundle_launch.py
```

**Verified 2026-10-06**: both pass, four runs out of four. It is opt-in (it builds and
launches); the generation is test-time, into a temp dir it deletes.

> **The namespaced cell is now brought up from a generated bundle -- from scratch.** A
> scene-loader bundle under a namespace COPIES the base config's `ros2_controllers.yaml`, and
> the reference cell's base keys it by bare node name: under `/r3` the controller manager reports
> `The 'type' param was not defined for 'joint_state_broadcaster'` and the spawner dies
> (verified). Generation warns about exactly that; the fix is `/**/<name>` keys in the base.

### Test A2c — the generation identity proof: 43 projects × 5 environments (vNext M2.3, ~12 s, opt-in)

The M2.2 identity proof, kept: the 43 projects of `test/identity_cases.py` (40 on the golden:
measured faces, every candidate rule, noise filters, tool numbers, extra detectors and external
nodes; the example twice; the owner's policy project) each generated in five fresh interpreters
-- trainit_perception importable (twice), made unimportable, off `PYTHONPATH`, uninstalled --
must give the same tree byte for byte (manifest included) or the same refusal; no run may import
TP, TMR or the ament index, open their files or write outside its bundle; every bundle records
the one committed capability snapshot and names no machine path. A positive control proves the
observers see what they watch; the workers must run THIS source tree's TSA.

```bash
cd ~/fr5_ws/src/trainit_setup_assistant
TRAINIT_IDENTITY_TEST=1 QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest -q test/test_generation_identity.py
```

**Verified 2026-10-08** (TSA 7.5.0): 49 passed in about 12 s with 8 jobs
(`TRAINIT_IDENTITY_JOBS=N` to change). Projects the checkout lacks skip, naming the input.

### Test A2d — the composite MOCK bundles, launched (vNext M2.7 T13 + T14, ~4 min, no GPU, opt-in)

The section-13 composite tool (two FollowJointTrajectory axes, a GripperCommand gripper and a
SetBool jaw on two sites, a SetBool lock) generated three times -- at the root, in /r1 and in
/r2, IDENTICAL end-effector ids -- built in a temp workspace and brought up at once on ROS domain
83 (`test/composite_launch.py` is the harness). Phase A: the tool joints reach `<ns>/joint_states`
(K1), TF reaches both site TCPs and follows the axes (K2), MoveIt's current state holds the tool
values and an arm motion logs no "Missing" joint (K3), no namespace collision (K4), five
distinguishable actuator states, each command moving only its own (K5). Phase B: the root's
generated tree -- nominal exit 0, a FAULT on the lock failing the FIRST VerifyGrasp at tool level
(exit 1), site_b not held ending in a VERDICT (exit 1) (K6). Phase C (/r1 and /r2 trees at once)
needs TMR >= 2.1.0 and is SKIPPED below it. Every launch must end on its SIGINT and leave the
domain empty.

```bash
source /opt/ros/humble/setup.bash && source ~/fr5_ws/install/setup.bash
cd ~/fr5_ws/src/trainit_setup_assistant
TRAINIT_LAUNCH_TEST=1 python3 -m pytest -q -s test/test_m2_7_composite_mock_launch.py
```

With `TRAINIT_M27_LAUNCH_EVIDENCE=<dir>` the deterministic report, the bundle identities and every
launch log are copied to `<dir>/composite_mock/`. It never runs together with another user of
domain 83 or 84.

### Test A2e — namespaced runners on real move_groups (vNext M2.7 T17b/T17c, ~2 min, no GPU, opt-in)

The canonical `suction_simple` bundle (launch-safe; the 3-line SetBool contract on the mock
adapter) generated three times -- in /r3, /r1 and /r2, the same robot and the same frames -- built
once in a temp workspace (colcon sequential, make -j2) and launched on ROS domain 84: each bring-up
`mode:=mock rviz:=false`, then a SEPARATE runner launch `trainit_bt.launch.py use_sim_time:=false
groot_monitor:=false` (a from-scratch bring-up declares neither `bt` nor `groot_monitor`). T17b,
/r3 alone: the tree ends SUCCESS within 120 s, every MoveIt client of the runner is under /r3, and
/r3's move_group receives one MoveGroup and one ExecuteTrajectory goal per MoveWaypoint. T17c, /r1
and /r2 at once: both trees end SUCCESS, each move_group receives exactly its own tree's goals
through clients only its own move_group serves, the two runners share only /tf, /tf_static,
/rosout and /parameter_events, and no MoveIt name of either lies outside its namespace. Every
launch must end on its SIGINT (a runner launch by itself, when its tree ends) and leave the domain
empty. It needs a runner that resolves MoveIt inside its namespace (TMR >= 2.1.0): below that it
SKIPS, never passes.

```bash
source /opt/ros/humble/setup.bash && source ~/fr5_ws/install/setup.bash
cd ~/fr5_ws/src/trainit_setup_assistant
TRAINIT_LAUNCH_TEST=1 python3 -m pytest -q -s test/test_m2_7_bundle_runner_launch.py
```

**Verified 2026-10-12** (TSA 7.7.0, TMR 2.1.0): 12 passed in each of two rounds, the normalised
reports byte-identical. With `TRAINIT_M27_LAUNCH_EVIDENCE=<dir>` each case's normalised report, the
bundle identities and every launch log are copied to `<dir>/bundle_runner/<case>/`. It never runs
together with another user of domain 83 or 84.

### Test A3 — an EXTERNAL-NODE detector really launches (D-015, ~1 min, no GPU)

The unit tests read the generated launch file; this one hands the generated bytes to
`ros2 launch` and lets ROS judge them. It needs a real external-node detector package built in
the workspace (a program's own; the README's `cell_perception` is a placeholder), so it is
optional — but it
is the only check that proves the package-share resolution and the parameter ORDER are
right rather than plausible.

Generate a bundle (bundle prefix `ext` below) whose single detector is the README's
external-node entry, with `package`, `executable` and `parameter_file` set to that real
package's, then:

```bash
source /opt/ros/humble/setup.bash && source ~/fr5_ws/install/setup.bash
ros2 launch /tmp/ext_bundle/ext_app/launch/trainit_bt.launch.py --print   # structure only
ros2 launch /tmp/ext_bundle/ext_app/launch/trainit_bt.launch.py           # really start it
```

**Expect** `--print` to resolve the entry to
`ExecuteProcess(cmd=[ExecInPkg(pkg='<package>', exec='<executable>'), …])`, and
the real launch to start the node with **two** params files in this order —
`…/install/<package>/share/<package>/<parameter_file>` then
`/tmp/launch_params_*` (the inline ones). For a node that refuses a missing model, set
the inline `model` to a path that does not exist while the package's own file ships an
empty one: the launch then fails naming the path the *project* set inline. **That failure
is the pass** — the inline parameters provably override the file, which is the whole
reason they are emitted after it. Point `model` at a real `.onnx` to go further. A
program that ships such a detector documents its own concrete run of this test.

---

### Test A4 — the grasp face, measured (v7.2 / ADR-0018, ~5 s, headless)

```bash
cd ~/fr5_ws/src/trainit_setup_assistant
QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
  python3 -m pytest test/test_grasp_face_picker.py -q
```
**Expect:** all pass, with **one skip** — the exact-STEP test, unless a headless FreeCAD is
on this machine. To run that one too (the AppImage needs no installation):
```bash
TRAINIT_FREECADCMD=$HOME/Apps/FreeCAD-1.1.1.AppImage \
QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
  python3 -m pytest test/test_grasp_face_picker.py -q
```

---

## Test B — GUI wizard → generate → run (no gizmo, ~10 min)

The fastest way to see the GUI drive the whole pipeline. You type the values; the live
gizmo (Test C) is optional.

1. **Launch the wizard:**
   ```bash
   cd ~/fr5_ws && source install/setup.bash
   ros2 run trainit_setup_assistant trainit_setup_assistant
   ```
2. **S0 — Load robot (use "Open project" for FR3WML):** in **Open project.yaml** browse to
   `~/fr5_ws/src/trainit_setup_assistant/examples/fr3wml_project.yaml`.
   **Next.** → "… gripper=suction | collision-matrix: yes".
   (The *Robot xacro* field below is the bootstrap path for a brand-new robot — see
   "Two ways to start" above; it shows `collision-matrix: MISSING`.)
3. **S1 — Group & frames:** confirm base_link / tcp / fr3wml / j1..j6. **Next.**
4. **S2 — Named states:** Name `home2`, Joints
   `j1=0, j2=-1.0814, j3=-1.62, j4=-1.57, j5=1.57, j6=0` → **Add**. **Next.**
5. **S3 — Scene:** id `table`, shape `box`, dims `2.0, 2.0, 0.10`, position `0, 0, -0.08`
   → **Add**. **Next.**  (Or **Import USD…** — see Test D.)
6. **S4 — Application:** `pick_and_place`, planner `pilz`, payload `cube`, dims
   `0.02, 0.02, 0.02`, offset `0, 0, 0.01`. **Next.**
7. **S5 — Waypoints & motions:** add each move (Move name / Target / Motion / Speed /
   Role), then **Add move to sequence**:
   | name | target | values | motion | planner | speed | role | tools |
   |---|---|---|---|---|---|---|---|
   | home | named | home2 | ptp | pilz | 80 | home | |
   | pre_pick | tcp | pos `0.556,-0.029,0.167` quat `-0.707,0.707,0.008,-0.009` | free | pilz | 60 | pre_pick | |
   | pick | tcp | pos `0.558,-0.026,0.024` quat `-0.707,0.707,0.008,-0.009` | lin | | 30 | pick | ☑grasp ☑attach (payload `cube`) |
   | post_pick | tcp | pos `0.558,-0.026,0.124` same quat | lin | | 40 | post_pick | |
   | pre_place | tcp | pos `0.564,-0.275,0.248` same quat | free | pilz | 60 | pre_place | |
   | place | tcp | pos `0.558,-0.275,0.023` same quat | lin | | 30 | place | ☑release ☑detach |
   | post_place | tcp | pos `0.558,-0.275,0.123` same quat | lin | | 40 | post_place | |
   | home | named | home2 | ptp | pilz | 80 | home | |
   **Next.**
8. **S6 — Generate:** Project name `fr3wml_gui_test`, Output dir
   `~/fr5_ws/src/trainit_generated_test2` → **Generate**.
   → "Generated NN files…".
9. **Build + run** the generated bundle:
   ```bash
   cd ~/fr5_ws
   colcon build --packages-select \
     fr3wml_gui_test_description fr3wml_gui_test_moveit_config fr3wml_gui_test
   source install/setup.bash
   # Terminal A:
   ros2 launch fr3wml_gui_test bringup.launch.py mode:=mock
   # Terminal B:
   ros2 launch fr3wml_gui_test trainit_bt.launch.py planner_mode:=pilz
   ```
   **Expect:** RViz opens with FR3WML; the robot runs the validated pick&place of
   the `fr3wml_suction_camera_tsa_4_1_bundle` golden (vision-guided: detect → approach → pick → suction →
   … ). Then `mode:=isaac`.

---

## Test C — GUI wizard WITH the live RViz gizmo (the full experience)

Here you move the end-effector with the real MoveIt gizmo and capture poses into the
wizard. For FR3WML you can reuse the already-built `fr3wml_app_moveit_config` as the
live session (no bootstrap needed).

1. **Start the live config session (gives you the gizmo):**
   ```bash
   cd ~/fr5_ws && source install/setup.bash
   ros2 launch trainit_setup_assistant setup_assistant.launch.py \
       moveit_config_package:=fr3wml_suction_camera_moveit_config robot_name:=fr3wml
   ```
   RViz opens with the **MotionPlanning** panel. The interactive-marker **gizmo** is on
   the `tcp` frame.
2. **Launch the wizard** (another terminal): `ros2 run trainit_setup_assistant trainit_setup_assistant`,
   **Open project** `examples/fr3wml_project.yaml` (S0/S1 as in Test B) so you start from a
   complete, runnable base and just re-capture poses with the gizmo.
3. **Capture a named state (S2):** in RViz, drag the arm to a pose and **Plan & Execute**;
   in the wizard click **Capture current (live)** → the joint values fill in → name it
   `home2` → **Add**.
4. **Capture waypoints (S5):** for each move, in RViz drag the gizmo to the target and
   **Plan & Execute**; in the wizard click **Capture pose (live)** → position+orientation
   fill in → pick the motion (PTP/LIN/CIRC), planner, speed, role → **Add move**.
5. **Generate (S6)** as in Test B, then build + run.

> The "Capture …" buttons read live data from the running session (joint states / the
> `base_link→tcp` transform). If a capture says "capture failed", the session isn't up
> yet or the frames differ.

---

## Test D — import your Isaac scene `.usd` (collision objects)

USD positions are in the **stage world frame**; MoveIt plans in the **robot base frame**.
If the robot is mounted up in the cell (FR3WML sits at world z≈1.06 on a table), raw
world coords would place objects ~1 m above the robot in RViz. The importer fixes this:
it **aligns objects to the robot base** and **skips the robot's own prims**.

**Setup (see the imported objects live in RViz):**
```bash
cd ~/fr5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select trainit_setup_assistant --symlink-install   # if not built
source install/setup.bash

# Terminal A — live session (RViz shows the planning scene)
ros2 launch trainit_setup_assistant setup_assistant.launch.py \
    moveit_config_package:=fr3wml_suction_camera_moveit_config robot_name:=fr3wml

# Terminal B — the wizard
ros2 run trainit_setup_assistant trainit_setup_assistant
```

**In the wizard:**
1. **S0 — Open project:** `~/fr5_ws/src/trainit_setup_assistant/examples/fr3wml_project.yaml`. **Next** → S1/S2.
2. **S3 — Scene → Import USD…:** select your scene, e.g.
   `~/fr5_ws/install/fr3wml_isaac/share/fr3wml_isaac/usd/scenes/bin_picking_cell.usd`.
   Leave **"USD robot base prim"** blank → it auto-detects the robot (`/fr3wml_suction`
   from the name `fr3wml`) and aligns everything to the base. (Or type `/fr3wml_suction`.)
3. The imported objects appear in the list at robot-relative coordinates. Click an object
   to load it into the form; for the **manipulated part** tick **dynamic** and
   **Add/update object** (so it's excluded from collision planning). Remove anything you
   don't want as an obstacle.
4. **Preview in RViz (live)** → the collision boxes appear in Terminal A's RViz, now
   aligned to the robot (the part sits near the gripper, not 1 m up).
5. Continue S4/S5/Generate → build → run as in Test B.

**Verify (quick, headless):** the part lands at the flange, not 1 m up:
```bash
python3 - <<'PY'
import sys; sys.path.insert(0, '.')    # run from the trainit_setup_assistant checkout
from ament_index_python.packages import get_package_share_directory
from trainit_setup_assistant.importers import import_usd
U = get_package_share_directory('fr3wml_isaac') + '/usd/scenes/bin_picking_cell.usd'
for o in import_usd(U, robot_hint='fr3wml'):   # auto-align to the robot
    print(o.id, [round(x,3) for x in o.position], 'dims', [round(x,3) for x in o.dims])
PY
```
Expect e.g. `Cube [0.561, -0.026, 0.014] …` (≈ at the gripper), not z≈1.07.

Notes / limitations:
- The importer reads **loadable geometry prims** (Cube/Sphere/Cylinder/Cone/Mesh) and
  boxes them to their world AABB; nested transforms + scales are honoured.
- Objects that are **references to Omniverse cloud assets not downloaded locally** (e.g.
  `table`, `blue_sorting_bin` pointing at `…s3…amazonaws.com/…`) load as **empty Xforms**
  → no geometry → not imported. Download those Isaac assets (or export a flattened USD
  with geometry baked in) to import them too.
- The robot is **not** imported from the USD — load it as a project/xacro in S0; the USD
  is for the **scene** only.

---

## Troubleshooting

- `command not found: trainit_*` → use `ros2 run trainit_setup_assistant <cmd>` (ament_python
  installs scripts under `lib/<pkg>`, not bare PATH).
- GUI doesn't open → ensure a display (`echo $DISPLAY`); the wizard is Qt.
- "Capture failed" → the live session (Test C step 1) must be running first.
- Generated bundle won't build → it depends on `trainit_motion_runtime` + bridges; build
  inside `~/fr5_ws` (or overlay it) so those are on the path.
- `PLAN FAILED (code=-31)` on the first move / RViz "Frame [base_link] does not exist" →
  the bundle was made from the **xacro bootstrap** (path 2) so it lacks the collision
  matrix + cell bridges. Regenerate via **Open project** `examples/fr3wml_project.yaml`
  (path 1), or add `robot.disable_collisions` + the `deployment` block to the project.
- Output dir with `~` → the GUI now expands `~`/`$VARS`, but if in doubt use the **Browse…**
  button or an absolute path (`/home/<you>/fr5_ws/src/...`). (Older builds took `~`
  literally and created a `~` folder in the launch directory.)
