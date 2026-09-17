# TrainIt Setup Assistant (TSA)

Configure a robotic **manipulation application** against a faithful RViz/MoveIt
environment and generate a **self-contained ROS 2 bundle** that runs on the TrainIt
Motion Runtime (TMR) with one command — in **mock**, **isaac**, or **real**.

Think "MoveIt Setup Assistant, but for the whole application layer": waypoints, motions,
planner choice, and dynamic-object handling → a Behaviour-Tree app you can watch run
immediately. Robot- and cell-agnostic; nothing is hardcoded to a specific robot.

> The headline product target is **camera-driven bin-picking with learned policies**.
> v4 ships **blind pick-and-place** AND **vision-guided motion** (2D colour-mask
> perception, live-tuned) end-to-end; PLC actuation and policies are architecture
> seams (present, not yet active). A course is included.

---

## What the TSA does — and doesn't

The TSA automates the **application layer**. It does **not** build your digital twin —
that is expert work (the *Adaptation Sprint*, a separate service):

| You/the Adaptation Sprint build (once, by hand) | The TSA generates (per application) |
|---|---|
| The **Isaac scene** (`<cell>_isaac`): robot+EE URDF, static/actuated meshes, physics, Action Graph, the grasp-physics Script Node, spawn/drive scripts | The **scene+planner MoveIt config** (mock/isaac/real + OMPL/Pilz/CHOMP + the cell obstacles) |
| The **base MoveIt config** (`<robot>_moveit_config`) with the mock/isaac/real bring-up + action-namespace parity | The **application**: waypoints, motions, planner, dynamic-object management |
| The **vendor bridges** (arm + gripper) for `real` | A **BT bundle** that runs on the TMR with one command + a README |

The boundary is deliberate: the twin needs Isaac/URDF/controller expertise; the
application is what you iterate on and ship.

## Prerequisites (in your `src/`)

```
<ros2_ws>/src/
├── <cell>_isaac/              # your Isaac scene + assets + grasp-physics Script Node   (Adaptation Sprint)
├── <robot>_moveit_config/     # hand-made base: mock/isaac/real bring-up + bridges       (Adaptation Sprint)
├── <vendor>_bridge/           # real arm + gripper bridges (only for mode:=real)         (Adaptation Sprint)
├── trainit_motion_runtime/    # the engine (BT runtime + scene loader)   — clone; keep versions bound
└── trainit_setup_assistant/   # this package                             — clone; keep versions bound
```

`trainit_setup_assistant` and `trainit_motion_runtime` **ship together** — always use
matching (committed) versions.

```bash
cd <ros2_ws>
source /opt/ros/humble/setup.bash
colcon build --packages-select trainit_motion_runtime trainit_setup_assistant --symlink-install
source install/setup.bash
```

## Dependencies (rosdep / apt — no pip needed)

TrainIt Community installs entirely through `rosdep`/`apt`; it does **not** require `pip`.
From the workspace root:

```bash
cd <ros2_ws>
rosdep install --from-paths src --ignore-src -r -y
```

That reads each `package.xml` and apt-installs every dependency, including
`python3-pydantic`, `python3-jinja2`, `python3-yaml` and `python_qt_binding` (the GUI).
The Python models are written against the **pydantic v1/v2 overlap API**, so Ubuntu 22.04's
`python3-pydantic` (1.9.x) is sufficient — no pip upgrade to pydantic 2 is required (TrainIt
runs unchanged on pydantic 1.9 and 2.x).

**Quick diagnosis.** If a command (GUI or headless) prints

```
error: missing required Python dependency 'pydantic'.
```

then that package is not installed — run the `rosdep install` above (it is a **core**
dependency, not a GUI extra). Only `python_qt_binding` is GUI-specific: the headless
`trainit_generate` / `trainit_verify` do not need it, so they run on a display-less host.

## The flow (wizard, 8 steps — the on-screen numbering)

**Phase A — build a faithful configuration environment**
1. **Robot & base config** — open a `project.yaml`, or point at the hand-made base
   MoveIt config; the assistant reads its group/frames/controllers/collision matrix.
2. **Cell scene (USD)** — preview robot+EE+objects; missing/cloud assets are flagged.
   Objects are classified (static / actuated / dynamic; grasp targets).
3. **Generate the scene+planner config** — a standalone copy of the base + planners +
   `scene.yaml`, plus the `colcon build … && source` snippet.
4. **Mode & bring-up** — mock | isaac | real, with the exact guided commands
   (open Isaac → Play → launch …). The live session powers every capture below.

**Phase B — configure the application**
5. **Perception (v4)** — configure the camera and the DETECTORS as named resources:
   pick a method (colour mask; shape/PCL/custom are roadmap), tune HSV + noise
   filters LIVE against the running bring-up (the tuner runs the SAME pure
   `detect()` the bundle will run), capture the centroid, set continuous/on-demand
   and the post-reset settle time. Skip it for a blind application.
6. **Application type** — *Blind pick and place* or *Vision guided motion* (the
   editable sequence with Vision blocks). Others are on the roadmap.
7. **Application blocks** — capture waypoint poses from RViz, motion type + planner +
   speed per move, the **dynamic-object management** (which objects the gripper
   grasps, per-move attached-collision-check, freeze/gravity on release), and — for
   vision (v4.1, D-016) — set **"Guided by camera"** on any Move block: pick a Step-5
   detector and the waypoint's position comes from the detection at run time
   (dx/dy/dz offsets, EEF-orientation policy); the sequence shows the automatic
   "detect at cycle start" row.
8. **Generate the bundle** — `<robot>_trainit_config` + `<robot>_app` +
   `<robot>_description` + a README. Build it and run — one command brings up the
   cell, one runs the app.

## Running a generated bundle

```bash
# terminal 1 — cell (move_group + planners + scene + controllers/bridges + RViz)
ros2 launch <robot>_trainit_config bringup.launch.py mode:=isaac     # or mock | real
# terminal 2 — the application (Behaviour Tree)
ros2 launch <robot>_app trainit_bt.launch.py planner_mode:=pilz      # or ompl | ompl_chomp
```
Every degree of freedom is data: waypoints/planner/speed in `<robot>_app/config/bt_params.yaml`,
obstacles + dynamic-object flags in `<robot>_trainit_config/config/scene.yaml`. See the
generated bundle's own `README.md`.

## CLIs (headless)

- `trainit_generate <project.yaml> -o <dir>` — generate the bundle (pure; no ROS needed).
- `trainit_verify <project.yaml> --golden <dir> [--build]` — equivalence + optional build.
- `trainit_setup_assistant` — the Qt wizard.

## For maintainers

See **`ARCHITECTURE.md`** for the internals (the three-config model, the emitters, the BT
vocabulary, the generalizable/sim-specific boundary) and the development roadmap.
