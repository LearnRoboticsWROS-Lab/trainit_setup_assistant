# TrainIt Setup Assistant

A GUI + headless generator that turns a robot + a configured application into a
buildable ROS 2 bundle (`*_description` + `*_moveit_config` + `*_app`) for the
TrainIt Motion Runtime — like the MoveIt Setup Assistant, but for the whole
**application layer**. Robot/framework-agnostic: everything is driven by the loaded
robot and the user's choices (nothing hardcoded to `fr3wml`).

## Status

| Milestone | What | State |
|-----------|------|-------|
| M0 | package skeleton + `CanonicalProject` model + `*_description` emitter | ✅ done |
| M1 | `*_moveit_config` emitter (SRDF + all config yamls + xacros) | ✅ done |
| M2 | self-collision matrix as project data | ✅ done |
| M3 | `*_app` emitter: `bt_params.yaml` + BT tree + launch + groot2 | ✅ done |
| M4 | verification: generated bundle == golden `fr3wml_app` (**32/32**) | ✅ done |
| M5 | robot auto-detection + `robot_loader` + `setup_assistant.launch.py` (live RViz gizmo) | ✅ done |
| M6 | Qt wizard (load robot → group → named states → generate) + live capture | ✅ done |
| M7 | waypoint capture + per-segment motion wizard (PTP/LIN/CIRC, planner, speed) | ✅ done |
| M8 | scene primitives (box/sphere/cyl/cone → collision objects, static/dynamic) | ✅ done |
| M9 | USD scene importer (pxr → AABB collision objects) | ✅ done |

The **whole MVP is complete and verified** (21 tests; the headless generator reproduces
the hand-coded `fr3wml_app` 32/32; the Qt wizard is offscreen-tested end-to-end; the
robot auto-detection passed an 8-archetype adversarial review). The generated bundle
`colcon build`s and bring-ups cleanly (`move_group` + controllers + bridges).

## Architecture (single source of truth)

```
gui/ (RViz + Qt, M6+)  ──edits──▶  model/CanonicalProject (project.yaml)  ──reads──▶  generator/
                                                                                         │
                                                                          *_description + *_moveit_config + *_app
```

The GUI only edits the project; the generator only reads it. `trainit_generate`
(headless) is the contract test that keeps them decoupled — a future web GUI can
replace `gui/` by emitting the same `project.yaml`.

---

# Testing the checkpoint (M0–M4)

**Prerequisite — nothing to clone.** Everything the generated bundle needs
(`trainit_motion_runtime`, `fr3wml_isaac`, `fairino_gripper`, `fairino_bridge`) is
already in your `~/fr5_ws`. You do **not** need a new workspace or to clone
`frcobot_ros2` / the fairino bridges. Just make sure the assistant is built:

```bash
cd ~/fr5_ws
source /opt/ros/humble/setup.bash
colcon build --packages-select trainit_setup_assistant --symlink-install
source install/setup.bash
```

> All commands below use **absolute paths** (`~/fr5_ws/...`) so they work from any
> directory. (The earlier error was just a relative path run from the wrong folder.)

## Test 1 — prove it EQUALS the golden `fr3wml_app` (no robot needed, ~5 s)

This is the strongest correctness check: generate the bundle and diff it against the
hand-coded `fr3wml_app`, then build it.

```bash
ros2 run trainit_setup_assistant trainit_verify \
  ~/fr5_ws/src/fr3wml_digital_twin/trainit_setup_assistant/examples/fr3wml_project.yaml \
  --golden ~/fr5_ws/src/fr3wml_digital_twin/fr3wml_app --build
```

**Expected — you should see:**
```
  OK  [byte] description/urdf/...            (meshes + xacros byte-identical)
  OK  [yaml] moveit_config/config/...        (kinematics, limits, controllers, ompl, pilz)
  OK  [srdf] moveit_config/config/fr3wml.srdf
  OK  [yaml] app/config/bt_params.yaml       (waypoints + DOF identical)
  OK  [tree] app/bt_trees/pick_place.xml     (node sequence identical)
32/32 checks passed

EQUIVALENCE: PASS
building generated bundle...
BUILD: PASS
```

Run the unit + golden tests too:
```bash
cd ~/fr5_ws/src/fr3wml_digital_twin/trainit_setup_assistant
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH python3 -m pytest test/ -q
# expected: 7 passed
```

## Test 2 — actually RUN the generated bundle in mock mode (no Isaac, no hardware)

Generate the 3 packages into your workspace (different names from `fr3wml_app`, so they
coexist with the golden), build them, and run the same pick-and-place driven by the BT.

```bash
# 1) generate into the workspace
ros2 run trainit_setup_assistant trainit_generate \
  ~/fr5_ws/src/fr3wml_digital_twin/trainit_setup_assistant/examples/fr3wml_project.yaml \
  -o ~/fr5_ws/src/trainit_generated --force

# 2) build the 3 generated packages
cd ~/fr5_ws
colcon build --packages-select \
  fr3wml_app_trainit_config_description \
  fr3wml_app_trainit_config_moveit_config \
  fr3wml_app_trainit_config
source install/setup.bash
```

**Terminal A — bring-up (mock):**
```bash
cd ~/fr5_ws && source install/setup.bash
ros2 launch fr3wml_app_trainit_config bringup.launch.py mode:=mock
```
You should see, and RViz should open with the FR3WML robot:
```
[move_group] Loading robot model 'fr3wml'...
[spawner] Configured and activated joint_state_broadcaster
[controller_manager] Configuring controller 'moveit_joint_controller'
[isaac_suction_bridge] Isaac suction bridge ready.
[joint_state_merger_suction] JointStateMerger ready
[fairino_suction_action_server] Ready. Action: /suctioncup_controller/gripper_command
```

**Terminal B — run the Behavior Tree:**
```bash
cd ~/fr5_ws && source install/setup.bash
ros2 launch fr3wml_app_trainit_config trainit_bt.launch.py planner_mode:=pilz
```
The robot executes, in RViz, the same cycle as `fr3wml_app`:
`home → pre_pick → pick → (suction ON) → post_pick → pre_place → place → (suction OFF) → post_place → home`.
Try `planner_mode:=ompl` and `:=ompl_chomp` too. (Groot2 live monitor on port 1667 if libzmq is present.)

## Test 3 — run it in Isaac (full digital twin)

Identical to how you run `fr3wml_app`: start Isaac and press **PLAY** first, then:
```bash
# Terminal A
ros2 launch fr3wml_app_trainit_config bringup.launch.py mode:=isaac
# Terminal B
ros2 launch fr3wml_app_trainit_config trainit_bt.launch.py planner_mode:=pilz
```

## Cleanup (optional)

```bash
rm -rf ~/fr5_ws/src/trainit_generated
cd ~/fr5_ws && colcon build   # or just leave the generated packages alongside fr3wml_app
```

## Optional — isolated overlay workspace

If you prefer not to add packages to `~/fr5_ws/src`, use a separate overlay that reuses
`~/fr5_ws` as the underlay (still nothing to clone):
```bash
mkdir -p ~/trainit_test_ws/src && cd ~/trainit_test_ws
ros2 run trainit_setup_assistant trainit_generate \
  ~/fr5_ws/src/fr3wml_digital_twin/trainit_setup_assistant/examples/fr3wml_project.yaml \
  -o src/trainit_generated --force
source /opt/ros/humble/setup.bash
source ~/fr5_ws/install/setup.bash      # underlay: engine + bridges
colcon build
source install/setup.bash
ros2 launch fr3wml_app_trainit_config bringup.launch.py mode:=mock
```

---

# Using the GUI Setup Assistant (M5–M9)

The wizard edits a project; the **live RViz gizmo** comes from a real `move_group`. Flow:

```bash
cd ~/fr5_ws && source install/setup.bash

# 1) Launch the wizard: Load robot xacro (auto-detects group/frames/EEF) →
#    confirm group → capture named states → scene (primitives or Import USD) →
#    application + payload → waypoints (capture pose / named state; per move pick
#    PTP/LIN/CIRC + planner + speed) → Generate.
ros2 run trainit_setup_assistant trainit_setup_assistant     # Qt wizard

# 2) To MOVE THE END-EFFECTOR WITH THE GIZMO while configuring, run the live session
#    against a generated+built moveit_config (the gizmo = MoveIt MotionPlanning marker,
#    real collision-aware IK). First Generate a bootstrap bundle from the wizard (or
#    trainit_generate), colcon build it, then:
ros2 launch trainit_setup_assistant setup_assistant.launch.py \
    moveit_config_package:=<robot>_app_moveit_config robot_name:=<robot>
#    The wizard's "Capture pose (live)" / "Capture current (live)" buttons read the
#    end-effector pose / joint values from this session.
```

Headless generation works without the GUI: build a `project.yaml` (or auto-draft one
from a robot xacro via `load_robot_spec`) and run `trainit_generate`.

## CLIs

- `trainit_generate <project.yaml> -o <dir>` — generate the bundle (pure, no ROS needed).
- `trainit_verify <project.yaml> --golden <dir> [--build]` — equivalence (auto-detects
  the golden's package names by role) + optional colcon build.
- `trainit_compute_collisions <project.yaml> [--write]` — compute `robot.disable_collisions`
  via `collisions_updater` (needs ROS; see note below).
- `trainit_setup_assistant` — the Qt Setup Assistant wizard.

## Note: the self-collision matrix

`collisions_updater` reliably hangs at DDS init when spawned head-/TTY-less, so the
self-collision matrix is treated as **project data** (`robot.disable_collisions`),
computed once and stored in `project.yaml`. Generation never spawns it. The FR3WML
example carries the matrix (identical to the golden's 13 pairs).

## Beyond the MVP

The architecture seams are in place for: more application templates (gluing,
follow-path, waypoint-replay, CNC — add an `applications/*.py`); a web GUI (swap `gui/`,
reuse the controller + generator); vendor-native motion adapters; and non-box collision
primitives once the engine's `planning_scene_manager` supports them (today they are
emitted as their AABB with a manifest warning). See the plan at
`~/.claude/plans/ora-voglio-creare-l-urdf-generic-petal.md`.
