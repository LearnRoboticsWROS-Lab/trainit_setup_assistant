# TrainIt Setup Assistant (TSA)

Configure a robotic **manipulation application** against a faithful RViz/MoveIt
environment and generate a **self-contained ROS 2 bundle** that runs on the TrainIt
Motion Runtime (TMR) with one command — in **mock**, **isaac**, or **real**.

Think "MoveIt Setup Assistant, but for the whole application layer": waypoints, motions,
planner choice, and dynamic-object handling → a Behaviour-Tree app you can watch run
immediately. Robot- and cell-agnostic; nothing is hardcoded to a specific robot.

> The headline product target is **camera-driven bin-picking with learned policies**.
> This MVP ships **blind pick-and-place** end-to-end; vision, PLC actuation and policies
> are architecture seams (present, not yet active). A course is included.

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

## The flow (wizard, 10 steps)

**Phase A — build a faithful configuration environment**
1. **Load the USD scene** — preview robot+EE+objects; missing/cloud assets are flagged
   (load a local `.usd` or skip). Objects are classified (static / actuated / dynamic).
   Test the cell's Isaac grasp adapter here (see `resources/isaac/`).
2. **Load the base MoveIt config** — the assistant reads its group/frames/controllers.
3. **Generate the scene+planner config** — a standalone copy of the base + planners +
   `scene.yaml`. Pick an output path/name.
4. **Build snippet** — `colcon build … && source install/setup.bash`.
5. **Choose the mode** — mock | isaac | real.
6. **Guided bring-up** — the exact commands for your mode (open Isaac → Play → launch …).

**Phase B — configure the application**
7. **Perception (v4)** — configure the camera and the DETECTORS as named resources:
   pick a method (colour mask; shape/PCL/custom are roadmap), tune HSV + noise
   filters LIVE against the running bring-up (the tuner runs the SAME pure
   `detect()` the bundle will run), capture the centroid, set continuous/on-demand
   and the post-reset settle time. Skip it for a blind application.
8. **Choose the application type** — *Blind pick and place* or *Vision guided
   motion* (the editable sequence with Vision blocks). Others are on the roadmap.
9. **Configure it** — capture waypoint poses from RViz, motion type + planner + speed per
   move, the **dynamic-object management** (which objects the gripper grasps, per-move
   attached-collision-check, freeze/gravity on release), and — for vision — drop a
   **Vision block** binding a Step-7 detector to the target/approach/retreat waypoints
   with their dz offsets.
10. **Generate the bundle** — `<robot>_trainit_config` + `<robot>_app` + `<robot>_description`
   + a README. Build it and run — one command brings up the cell, one runs the app.

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
