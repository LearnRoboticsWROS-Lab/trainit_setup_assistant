# TrainIt Setup Assistant — full test procedure

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
  python3 -m pytest test/ -q          # expect: 61 passed (plus 1 pre-existing network-USD failure)
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
import sys; sys.path.insert(0,'/home/fra/fr5_ws/src/trainit_setup_assistant')
from trainit_setup_assistant.importers import import_usd
U='/home/fra/fr5_ws/install/fr3wml_isaac/share/fr3wml_isaac/usd/scenes/bin_picking_cell.usd'
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
