# TrainIt Setup Assistant (TSA v8)

Configure a robotic **manipulation application** against a faithful RViz/MoveIt
environment and generate a **self-contained ROS 2 bundle** that runs on the TrainIt
Motion Runtime (TMR) with one command — in **mock**, **isaac**, **gazebo** or **real**.

Think "MoveIt Setup Assistant, but for the whole application layer": waypoints, motions,
planner choice, planned-vs-streamed execution, and dynamic-object handling → a
Behaviour-Tree app you can watch run immediately. Robot- and cell-agnostic; nothing is
hardcoded to a specific robot.

**New here?** [Install](#install-trainit-community-from-source) takes you from a bare Ubuntu 22.04
to a running sample cell in four steps, every command ready to paste, and a fifth takes you to your
own robot.

> The headline product target is **camera-driven bin-picking with learned policies**.
> v8 ships **blind pick-and-place**, **vision-guided motion** (2D colour-mask perception,
> live-tuned) and **bin picking** — the emptying loop: ranked candidates, a retry over
> them, a verified lift and a recovery step when the bin stalls. PLC actuation and
> policies are architecture seams (present, not yet active). A course is included.

> **Version.** `8.0.0`, the product label "TSA v8" (the Canonical Task Program complete: M3). What each release changed is in
> [Version notes](#version-notes), near the end.

---

## Install (TrainIt Community, from source)

TrainIt Community is five public repositories released together: their `main` branches always
belong together. [`trainit_community.repos`](trainit_community.repos) lists them for `vcs import`.
Four steps take you from a bare Ubuntu 22.04 to a running sample cell, and a fifth to your own
robot. Their commands are written to be pasted as they are, except `~/robot_ws`, which stands for
your robot workspace: put its path there.

1. [Check your machine](#1-check-your-machine): what has to be there before TrainIt.
2. [Get TrainIt and build it](#2-get-trainit-and-build-it), in a workspace of its own or in your
   robot workspace.
3. [Check the install](#3-check-the-install).
4. [Run the public sample](#4-run-the-public-sample): a cell that needs no robot of your own.
5. [Your own robot](#5-your-own-robot).

### 1. Check your machine

TrainIt needs Ubuntu 22.04 with ROS 2 Humble, the ROS tools that fetch, resolve and build a
workspace (`vcs`, `rosdep`, `colcon`), `git` and a C++ compiler, an initialised `rosdep`, and
`curl`. MoveIt 2, BehaviorTree.CPP 4 and the Python libraries (pydantic, Jinja2, PyYAML) are
**not** prerequisites: the `rosdep install` of step 2 installs them.

Paste this into a terminal. It changes nothing, works on a machine without ROS, and prints one
line per prerequisite: `OK`, or `MISSING` with what fixes it.

```bash
if grep -qs '^VERSION_ID="22.04"' /etc/os-release; then echo 'OK      Ubuntu 22.04'
else echo "MISSING Ubuntu 22.04: this is $(grep -s '^PRETTY_NAME=' /etc/os-release | cut -d= -f2)"; fi
if [ -f /opt/ros/humble/setup.bash ]; then echo 'OK      ROS 2 Humble'
else echo 'MISSING ROS 2 Humble: https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html'; fi
for t in vcs rosdep colcon; do
  if command -v $t >/dev/null; then echo "OK      $t"
  else echo "MISSING $t: sudo apt install python3-colcon-common-extensions python3-vcstool python3-rosdep"; fi
done
if command -v git >/dev/null; then echo 'OK      git'; else echo 'MISSING git: sudo apt install git'; fi
if command -v g++ >/dev/null; then echo 'OK      g++'; else echo 'MISSING g++: sudo apt install build-essential'; fi
if [ -f /etc/ros/rosdep/sources.list.d/20-default.list ]; then echo 'OK      rosdep initialised'
else echo 'MISSING rosdep initialised: sudo rosdep init && rosdep update'; fi
if command -v curl >/dev/null; then echo 'OK      curl'; else echo 'MISSING curl: sudo apt install curl'; fi
```

Fix the `MISSING` lines from the top down, then paste the block again until every line says `OK`.
The first line has no command: ROS 2 Humble's Ubuntu packages are built for 22.04, so on another
release use a 22.04 machine or virtual machine, or the Docker image `osrf/ros:humble-desktop`,
which this install is verified on. ROS 2 Humble comes next: its install page, linked in the
block, also sets up the apt repository the ROS tools come from, and its `ros-humble-desktop` is
what TrainIt is verified on. From that repository, `sudo apt install ros-dev-tools` installs
`vcs`, `rosdep`, `colcon`, `git` and the compiler in one line.

### 2. Get TrainIt and build it

TrainIt goes into a colcon workspace in one of two layouts, A or B below. In both, the five
repositories go together: import all five from the manifest and build them together, never one
on its own (TMR needs `trainit_interfaces`, and a bundle with an end-effector contract needs the
hardware layer).

| Repository | What it is | Needed for |
|---|---|---|
| `trainit_interfaces` | the messages and services TMR and the hardware layer share | every bundle |
| `trainit_motion_runtime` | TMR, the Behaviour-Tree runtime a bundle runs on | every bundle |
| `trainit_hardware_layer` | the end-effector adapters | a bundle with an end-effector contract |
| `trainit_setup_assistant` | this package: the wizard and the headless generator | generating a bundle |
| `trainit_perception` | the detectors | vision and bin-picking applications |

Before either layout, bring the system up to date, `sudo apt update && sudo apt upgrade`: that is
the state this install was verified in. Then expect two things. `rosdep install` runs apt through
`sudo`, so it asks for your password. The first `colcon build` takes a while: about 8 minutes with
two compile jobs (`MAKEFLAGS=-j2`), nearly all of it TMR's C++. A plain `colcon build` runs one
compile job per CPU core; if that is more than the memory holds (the machine freezes, or the build
ends with `Killed signal terminated program cc1plus`), build again with two jobs and one package
at a time, `MAKEFLAGS=-j2 colcon build --executor sequential`, then `source install/setup.bash`.

No robot workspace yet, and you only want to try the public sample? Take A: the sample's commands
use `~/trainit_ws`.

**A. TrainIt in a workspace of its own**, `~/trainit_ws`, which your robot workspace then
overlays. Pick it when more than one workspace will use TrainIt, or to keep TrainIt's own build
out of a robot workspace that works: TrainIt is built once, and a robot workspace is built on top
of it.

```bash
mkdir -p ~/trainit_ws/src && cd ~/trainit_ws
curl -fsSLO https://raw.githubusercontent.com/LearnRoboticsWROS-Lab/trainit_setup_assistant/main/trainit_community.repos
vcs import src < trainit_community.repos
source /opt/ros/humble/setup.bash
rosdep update
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

From then on, in every new terminal, source ROS 2 and TrainIt, then your robot workspace if you
have one (remove the `#` of the third line), in this order. Build your robot workspace with the
first two sourced.

```bash
source /opt/ros/humble/setup.bash
source ~/trainit_ws/install/setup.bash
# source ~/robot_ws/install/setup.bash   # your robot workspace, if you have one: use its path
```

**B. TrainIt in your robot workspace**, the colcon workspace whose `src/` already holds your
robot's packages. Pick it for one workspace, one build and one `source` line. It is the layout
[Prerequisites](#prerequisites-in-your-src) draws. Replace `~/robot_ws` with your workspace's path
before you paste: each line runs only if the one before it worked, so a wrong path stops at the
first line, with `No such file or directory`, and nothing else runs.

```bash
cd ~/robot_ws &&
curl -fsSLO https://raw.githubusercontent.com/LearnRoboticsWROS-Lab/trainit_setup_assistant/main/trainit_community.repos &&
vcs import src < trainit_community.repos &&
source /opt/ros/humble/setup.bash &&
rosdep update &&
rosdep install --from-paths src --ignore-src -r -y &&
colcon build &&
source install/setup.bash
```

If a line fails, fix what it reports and paste the block again: what already worked is quick the
second time. From then on, in every new terminal: `source ~/robot_ws/install/setup.bash`, which
sources ROS 2 too.

**Where your bundles go.** A bundle TSA generates goes into the `src/` of a workspace that holds
or overlays TrainIt, and is built there: your robot workspace (in layout B, the one that holds
TrainIt), or a new workspace of its own, which is the public sample's way in either layout and
lets `rm -rf` remove it whole. In layout A, never `~/trainit_ws`, which holds TrainIt alone.

### 3. Check the install

In the terminal of step 2, or in a new one sourced as step 2 says, paste:

```bash
if ! command -v ros2 >/dev/null; then
  echo 'MISSING ros2: source your workspace as step 2 says, then paste this block again'
else
  echo 'OK      ros2'
  for p in trainit_interfaces trainit_motion_runtime trainit_hardware_layer \
           trainit_setup_assistant trainit_perception; do
    if ros2 pkg prefix $p >/dev/null 2>&1; then echo "OK      $p"
    else echo "MISSING $p: source your workspace as step 2 says; if it is sourced, its build failed"; fi
  done
  if ros2 run trainit_setup_assistant trainit_generate --help >/dev/null 2>&1; then
    echo 'OK      trainit_generate --help'; else echo 'MISSING trainit_generate (see above)'; fi
  for p in moveit_core moveit_ros_planning moveit_ros_planning_interface moveit_ros_move_group \
           moveit_configs_utils moveit_planners_ompl moveit_planners_chomp \
           pilz_industrial_motion_planner moveit_setup_assistant behaviortree_cpp xacro; do
    if ros2 pkg prefix $p >/dev/null 2>&1; then echo "OK      $p"
    else echo "MISSING $p: run the rosdep install line of step 2 again"; fi
  done
  if python3 -c 'import pydantic, jinja2, yaml' 2>/dev/null; then echo 'OK      pydantic, jinja2, yaml'
  else echo 'MISSING a Python library TSA needs: run the rosdep install line of step 2 again'; fi
fi
```

In a terminal with nothing sourced it prints only the `ros2` line. Otherwise every line should say
`OK`: the five TrainIt packages, the generator, the MoveIt 2 and BehaviorTree.CPP 4 packages TMR
and TSA run on, and TSA's Python libraries. What a bundle needs besides is not checked here:
ros2_control's controller manager and controllers, MoveIt's simple controller manager, MoveIt
Servo when a move streams. The bundle's first-run lines install them
([Running a generated bundle](#running-a-generated-bundle)).

### 4. Run the public sample

[`examples/ur5e_suction`](examples/ur5e_suction/README.md) is a complete cell for a machine with
no robot of its own: a Universal Robots UR5e from `ur_description` with a simple suction cup,
running a deterministic pick-and-place on `mode:=mock` (ros2_control's mock hardware: no
simulator, no GPU). Its README takes about five minutes: one apt package, one `trainit_generate`
into a new workspace of its own, that workspace's first `rosdep install` and `colcon build`, then
two terminals, with the log lines to expect and the harmless ones to ignore.

The other example, `examples/fr3wml_project.yaml`, describes the reference cell, whose robot
description is not public: it needs the private `fr3wml_digital_twin` repository, at the paths
the file records.

### 5. Your own robot

TSA needs one thing from your robot, and can use a second:

- **Its robot description**, a URDF or xacro file. TSA compiles it with xacro, so every package
  it names (`$(find ...)`, `package://...`) must be on the sourced index: build your robot
  workspace and source it as step 2 says before you start the wizard or `trainit_generate`.
- **A base MoveIt config**, optionally: your own `<robot>_moveit_config`, whose `config/` holds
  what the MoveIt Setup Assistant writes there, including the SRDF (`<robot>.srdf`),
  `<robot>.urdf.xacro`, `moveit_controllers.yaml` and `ros2_controllers.yaml`. The wizard reads
  the robot, its planning group, frames, named states, controllers and collision matrix from
  them, and the bundle copies that `config/`. A `launch/bringup*.launch.py` of your own is
  optional: when there is one, the wizard also takes the cell's bridges from it. A config with no
  `config/*.srdf` is refused at Step 1: `ur_moveit_config`, whose SRDF is a xacro in `srdf/`, is
  one, so a UR arm takes the from-scratch route. Without a base config, TSA writes the MoveIt
  configuration itself (*from scratch*), as for the public sample.

A bundle is three packages, `<name>_description`, `<name>_trainit_config` and `<name>_app`. Give
it a name that no package in your workspace already has: the wizard proposes your robot's name,
which clashes with a `<robot>_description` of your own, and colcon then refuses to build the
workspace (`Duplicate package names not supported`).

**With a base config**, start the wizard, a Qt window (it needs a desktop session), in a terminal
sourced as step 2 says, and follow its eight steps
([The flow](#the-flow-wizard-8-steps--the-on-screen-numbering), below). Run the commands it
prints in such a terminal too.

```bash
ros2 run trainit_setup_assistant trainit_setup_assistant
```

- **Step 1** loads the base config. Set *Base package path* to its source directory, such as
  `~/robot_ws/src/<robot>_moveit_config`, and replace the pre-filled *Base moveit_config
  package* with your config's name: the project records it.
- **Step 4** builds the configuration Step 3 wrote and brings it up, so that you can capture poses
  against RViz. Its mode starts as `isaac`: without Isaac Sim, choose `mock`, and Step 8's
  commands follow that choice. Before its bring-up, from your workspace's root, run the
  `rosdep install` line of
  [Running a generated bundle](#running-a-generated-bundle) once: the configuration Step 3 wrote
  needs ros2_control's controller manager and controllers, which Install does not install.
- **Step 8** names the bundle (*Project / bundle name*), writes it into
  `<output dir>/<name>_bundle` and prints the commands that build and run it, the first-run ones
  included. Its output directory starts as the one that holds your base config; set it to your
  workspace's `src/` if the base config lives anywhere else.

So far this route has been verified only on the reference cell's hand-made base config. A config
the MoveIt Setup Assistant wrote has the files Step 1 reads, but no test or run has used one yet.

**From scratch**, start from the public sample: copy it out of the repository, point the copy at
your description and make it yours ("Making it your own", in its README, lists what changes for
another robot). In layout B, copy it from `~/robot_ws/src/trainit_setup_assistant/` instead:

```bash
cp -r ~/trainit_ws/src/trainit_setup_assistant/examples/ur5e_suction ~/my_cell && cd ~/my_cell
```

Generate from inside the copy, since a project's relative paths resolve against the current
directory. `-o` is the bundle's own directory inside your workspace's `src/`, never `src/`
itself: `--force` deletes the `-o` directory and everything in it.

```bash
ros2 run trainit_setup_assistant trainit_generate project.yaml -o ~/robot_ws/src/my_cell_bundle
```

Either way, build and run the bundle as [Running a generated bundle](#running-a-generated-bundle)
says: its first-run line, a `rosdep install`, installs what the bundle declares.

## What the TSA does — and doesn't

The TSA automates the **application layer**. It does **not** build your digital twin —
that is expert work (the *Adaptation Sprint*, a separate service):

| You/the Adaptation Sprint build (once, by hand) | The TSA generates (per application) |
|---|---|
| The **Isaac scene** (`<cell>_isaac`): robot+EE URDF, static/actuated meshes, physics, Action Graph, the grasp-physics Script Node, spawn/drive scripts | The **scene+planner MoveIt config** (mock/isaac/real + OMPL/Pilz/CHOMP + the cell obstacles) |
| The **base MoveIt config** (`<robot>_moveit_config`, optional): its `config/` as the MoveIt Setup Assistant writes it, and for isaac/real the bring-up + action-namespace parity | The **application**: waypoints, motions, planner, dynamic-object management |
| The **vendor bridges** (arm + gripper) for `real` | A **BT bundle** that runs on the TMR with one command + a README |

The boundary is deliberate: the twin needs Isaac/URDF/controller expertise; the
application is what you iterate on and ship.

## Prerequisites (in your `src/`)

This is a cell the Adaptation Sprint built, laid out as Install's layout B: one workspace,
`~/robot_ws` here, whose `src/` holds the hand-made twin and the five TrainIt repositories. In
layout A the five `trainit_*` entries live in `~/trainit_ws/src` instead, and `~/robot_ws`, built
with `~/trainit_ws` sourced, overlays it. The public sample needs none of the three hand-made
entries, and a robot of your own needs at most a base config, which can be the MoveIt Setup
Assistant's ([Install, step 5](#5-your-own-robot)).

```
~/robot_ws/src/
├── <cell>_isaac/              # your Isaac scene + assets + grasp-physics Script Node   (Adaptation Sprint)
├── <robot>_moveit_config/     # base, here hand-made: mock/isaac/real bring-up + bridges (Adaptation Sprint)
├── <vendor>_bridge/           # real arm + gripper bridges (only for mode:=real)         (Adaptation Sprint)
├── trainit_interfaces/        # the shared messages and services         — vcs import (Install)
├── trainit_motion_runtime/    # the engine (BT runtime + scene loader)   — vcs import (Install)
├── trainit_hardware_layer/    # the end-effector adapters                — vcs import (Install)
├── trainit_perception/        # the detectors                            — vcs import (Install)
└── trainit_setup_assistant/   # this package                             — vcs import (Install)
```

The five TrainIt repositories **ship together**: build their `main` branches together
(Install, above), never one on its own -- TMR needs `trainit_interfaces`, and a bundle with an
end-effector contract needs the hardware layer. To rebuild the workspace after adding a package to
its `src/` (in layout A, `source ~/trainit_ws/install/setup.bash` first):

```bash
cd ~/robot_ws &&
source /opt/ros/humble/setup.bash &&
rosdep install --from-paths src --ignore-src -r -y &&
colcon build &&
source install/setup.bash
```

## Dependencies (rosdep / apt — no pip needed)

TrainIt Community installs entirely through `rosdep`/`apt`; it does **not** require `pip`.
From the root of the workspace whose `src/` holds TrainIt, `~/trainit_ws` in layout A (in layout
B, `~/robot_ws`):

```bash
cd ~/trainit_ws && rosdep install --from-paths src --ignore-src -r -y
```

That reads each `package.xml` and apt-installs every dependency, including
`python3-pydantic`, `python3-jinja2`, `python3-yaml` and `python_qt_binding` (the GUI).
The Python models are written against the **pydantic v1/v2 overlap API**, so Ubuntu 22.04's
`python3-pydantic` (1.8.2) is sufficient — no pip upgrade to pydantic 2 is required (TrainIt
runs unchanged on pydantic 1.8 and 2.x).

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
   This step also carries the **robot namespace** (ADR-0016) — see below.

   **Robot namespace: leave it empty unless the cell has more than one robot.** Empty
   means *you are configuring one robot*, every generated node runs at the root, and the
   bundle is byte-for-byte what TSA produced before the field existed. That is the default
   and the migration path. Fill it in (e.g. `r3`) and this robot's whole motion stack
   moves under `/r3`: `move_group`, the controllers, `robot_state_publisher`, the scene
   manager, the behaviour tree and the topics Servo derives. Two arms can then share a
   cell without fighting over one `/robot_description` or one `/controller_manager`.

   What a namespace deliberately does **not** touch, and why:

   | stays global | because |
   |---|---|
   | `/tf`, `/tf_static` | tf2 declares them absolute, and a split TF tree is a cell that cannot see itself |
   | the cameras | a camera serves a **station**, not a robot — one arm of four happens to pick from that bin |
   | `/perception/<detector>/…` | the detector that watches the bin is one detector however many arms reach into it |
   | RViz | MoveIt's MotionPlanning plugin builds no end-effector interactive marker from inside a namespace (measured). The cell is namespaced, the viewer is not; the generated bring-up remaps the two `robot_description` topics for it |

   Two things a namespace does **not** fix by itself, and the generator says so in
   `generation_manifest.yaml` rather than rewriting them behind your back: the **cell
   bridge parameters** and the **arm `/joint_states` remap** were captured verbatim from
   your base bring-up, and an absolute root name like `/joint_states_robot` never follows a
   namespace — give it the namespace in the project. And the **Gazebo** backend is not
   namespaced by this field at all: `gazebo_ros2_control` owns `/controller_manager` inside
   `gzserver` and takes the name from the URDF plugin tag, not from the launch.
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

   **The candidate rule (v7.1, ADR-0017) — WHICH object to pick.** Beside the detector's
   parameters, because that is what it is: one more detector parameter, whose only effect
   is the order the detections go out in. The **published order IS the ranking**
   (ADR-0016 Decision 1) — the runtime retries candidates in the order it was given and
   never sorts, so this one dropdown is what turns "try the next one" from a shuffle into
   a ranking.

   A rule is a product of named **terms**, kept apart on purpose so they can be compared
   rather than blended (two capsules side by side have overlapping boxes and two entirely
   visible faces; one blended number hides exactly that). The dropdown lists every shipped
   rule and **greys the ones this detector cannot compute**:

   | term | what it measures | computable when |
   |---|---|---|
   | `score` | the detector's own confidence | always |
   | `pose_quality` | inliers, residual, depth coverage, fitted size | the detector runs a **3D fit** — the colour mask does not |
   | `box_overlap` | how much of this object's box another's covers | the detector reports a pixel box |
   | `visible_fraction` | the mask's area against the area the object's own face should show | a mask area **+** the face's size, which TSA **derives** from the grasp target (no field to fill) |
   | `clearance` | how much room the tool has inside its own footprint | a 3D fit **+** the end-effector's **footprint radius** and **grip window** (Step 7) |

   So `candidate x visible face` (`score × pose_quality × visible_fraction`) is the
   **measured default** wherever a 3D fit exists, and `score x visible face` is the same
   idea for the colour mask TSA generates today. Two more facts behind the list, each
   measured rather than preferred: *clearance* read 0,00 on every candidate the reference
   picker actually chose, so it ships as a published **term** without being a default (a
   framework that shipped only the terms one cell needed would throw away the terms the
   next cell needs); and the 20 % overlap **refusal** is not offered at all, because
   measuring it halved the pick rate. `auto` lets the runtime pick the best rule it can
   compute, and `detector order` is an *explicit* "no ranking" — which is what
   distinguishes it from a rule that silently could not be computed.

   **A rule whose terms this project cannot compute is REFUSED at generation** — before a
   single file is written, naming the missing term, what would supply it, and the rules
   that *are* available. The alternative is worse than a refusal: since the order is the
   rank, an uncomputable rule would leave the cell picking in whatever order the model
   returned, under the name of a ranking, and that looks like a perception fault rather
   than a configuration one.

   A bundle that carries a rule **pins** `trainit_perception >= 0.3.0`, the version that
   reads it. Without the pin an older runtime would merge the key into its parameters,
   ignore it, and publish an unranked array in silence.

   The rules, the terms each one needs, the grasp faces and every parameter's version floor
   are trainit_perception's, read from TSA's committed snapshot of its vocabulary
   (`catalog/tp_registry_snapshot.yaml`) — never from the perception package installed on the
   machine that generates, so the same project gives the same bundle everywhere (ADR-0021).
   After trainit_perception's vocabulary changes, `python3 -m
   trainit_setup_assistant.catalog.refresh` updates the snapshot (it refuses a new method, rule
   or face the owner has not classified Community/Pro, and a token committed projects may hold
   going away), and
   `test_tp_registry.py` fails until it has run, naming what drifted.

   **Measure the grasp face — do not let the bounding box guess it (ADR-0018).** The
   winning rule divides by the area of the face the tool comes down on, and that area used
   to be approximated from the object's bounding box: on the reference capsule **28,10 mm
   against a true 20,95 mm**, because the box is inflated by the pull tab — +34 % on the
   radius and +80 % on the area. **"Measure grasp face…"** on the Perception step opens the
   object's own mesh in three orthographic views; click the face the tool comes down on and
   TSA writes the measured numbers into the detector. The flood fill highlights exactly the
   region it took, so a tolerance swallowing a fillet is visible instead of silently
   inflating the area, and a checkbox flips the views to reach the other three sides.

   - **Say what unit the file is in.** A mesh carries none, and the error is *silent*: the
     same part read as metres instead of millimetres measures a face a thousand times too
     big and nothing downstream complains. The scale defaults from the object's own
     `scale`, and a face outside the plausible range is refused rather than written.
   - **STL and OBJ always; STEP when this machine happens to have FreeCAD.** `freecadcmd`
     on `PATH`, or `TRAINIT_FREECADCMD=/path/to/FreeCAD.AppImage`, gives the EXACT B-rep
     face — true area, axis and radius, no tessellation. It is never a dependency: without
     it the picker works on meshes and says so once.
   - **The faces come from `trainit_perception`'s registry**, not from a list inside TSA:
     `disc` today, plus whatever the perception package registers. The dialog renders one
     field per parameter *that* face declares, so a new face family needs no TSA release.
     Nothing from the picker reaches the runtime but numbers — no bundle ever needs a mesh.

   **External-node detectors — the second kind of method (D-015).** `trainit_perception`
   names two kinds, and the difference is *who runs the code*. A **pure-python** method is
   a `Detector` subclass in its registry: the live tuner and the bundle run the same
   `detect()`, and TSA emits one `trainit_perception` `detector_node` per entry of
   `config/perception.yaml`. An **external-node** method is implemented by a *separate
   node* — C++/PCL, a learned model, a vendor SDK — that publishes the **same contract**
   (`vision_msgs/Detection3DArray` on `/perception/<name>/detections`), so `DetectObject`
   consumes it without knowing the difference.

   TSA used to emit **nothing at all** for the second kind: no package, no executable, no
   launch entry. A detector could be configured, recorded in `perception.yaml`, bound to a
   waypoint and ranked, and the bundle still launched nothing on its topic — `DetectObject`
   timed out, which reads as a perception fault and is a configuration one.
   `perception.detectors[].external_node` is the missing half, and it carries only what
   `launch_ros`'s `Node` needs:

   | field | what it is |
   |---|---|
   | `package` / `executable` | the node to run. **Both required** — a block missing either is refused at generation, naming which |
   | `node_name` | default `detector_<name>`. Set it when the node's own parameter file is keyed by a node name (ROS parameter files are) |
   | `parameter_file` | absolute, or relative to the package's **share** directory (`config/capsule.yaml`) |
   | `parameters` | inline node parameters, applied **after** the file so they win: a `model:=/abs/best.onnx` that lives in shell history today becomes configuration |
   | `topic_parameters` | `rgb` / `depth` / `camera_info` → the **parameter name** this node reads that role from. TSA fills the value |
   | `remappings` | `from → to`, for a node whose subscriptions are hardcoded |

   The camera topics are **not** repeated here: `rgb_topic` / `depth_topic` /
   `camera_info_topic` already live on the detector (falling back to the camera), so
   `topic_parameters` says only what the node *calls* each role and the value comes from
   the one place that has it. Otherwise the file TSA writes and the node TSA launches could
   name different cameras, silently.

   The generated `trainit_bt.launch.py` then starts that `Node` **next to** the
   pure-python `detector_node` entries — one bundle, both kinds, both global (ADR-0016).
   A project with no `external_node` block generates **byte-identical** output.

   **The external package is documented, never declared.** The generated `package.xml`
   carries a comment naming it and generation warns, but there is no `<exec_depend>`:
   ADR-0003 has Community installing with rosdep, and `rosdep install --from-paths src`
   fails with *cannot locate rosdep definition* for any package name that is neither a
   rosdep key nor in the workspace. An external detector is normally customer IP shipped
   separately, so declaring it would break the install of the **whole workspace** for
   people who never run that cell. It fails where it is *used* instead: `ros2 launch`
   names the missing package the moment you start the application.

   A worked example — a YOLO-segmentation detector shipped in a program's own package
   (`cell_perception` is a placeholder), launched by hand with
   `ros2 launch cell_perception seg_pose.launch.py model:=/abs/best.onnx`:

   ```yaml
   perception:
     detectors:
       - name: capsule
         method: custom                 # an external-node method
         rgb_topic: /bin_camera/color/image_raw
         depth_topic: /bin_camera/aligned_depth_to_color/image_raw
         camera_info_topic: /bin_camera/color/camera_info
         external_node:
           package: cell_perception
           executable: seg_pose_node
           node_name: seg_pose_node     # config/capsule.yaml is keyed by it
           parameter_file: config/capsule.yaml
           parameters:
             model: /opt/models/best.onnx
             method_3d: plane_ransac
           topic_parameters:
             rgb: rgb_topic
             depth: depth_topic
             camera_info: camera_info_topic
   ```

   Setting it is a **project edit** today: the Perception step's method combo still offers
   only the colour mask, so the external-node methods (`custom`, `pcl_cluster`) are
   configured in `project.yaml` and the wizard round-trips them untouched.
6. **Application type** — *Blind pick and place*, *Vision guided motion* (the editable
   sequence with Vision blocks) or *Bin picking* (the emptying loop — see below). Others
   are on the roadmap.

   **Bin picking: the emptying loop (ADR-0016).** A linear sequence of moves cannot empty
   a bin, and the reason is measured, not stylistic. On the reference cell, over a
   25-capsule bin: **3,33 candidates were tried per capsule delivered** (one pick in three
   does not succeed on the first candidate), **"the gripper closed" and "the capsule came
   up" differed on 1 lift in 22**, and the bin **stalled twice before it was empty**, each
   stall cleared by agitation worth six capsules out of twenty-five. So this application
   emits a loop with exactly those three features:
   - the **retry**: the detector publishes its candidates **best first** and that order
     *is* the ranking, so the tree tries them in order — candidate 0, then 1, then 2 —
     until one works. You configure *how many* ("Candidates to try"); it also becomes
     `DetectObject`'s `keep`, so the loop never asks for candidates it cannot try;
   - the **holding check**: after the lift, `CheckRobotState` asks whether the object is
     actually held. A failure here is not an error — it is what advances to the next
     candidate instead of placing nothing;
   - the **stall recovery**: "nothing detected" and "no candidate worked" are the same
     stall to an operator, so one branch treats both. You name a **cell-I/O channel** and
     the tree pulses it (`EnableProcess` → wait → `DisableProcess`). What that channel
     *actuates* is cell I/O (ADR-0013) and stays the cell's business: a shake is a machine
     with an amplitude, a frequency and a direction, and TSA emits the **step**, never the
     machine. Leave the channel empty and a stalled bin is only logged.

   Everything else is the same configuration as a vision-guided application: the standoff
   and retreat distances are the approach waypoint's camera offset and the retreat's
   step-relative offset, the rank rule lives in perception (Step 5, ADR-0014), and the
   execution mode is per move.
7. **Application blocks** — capture waypoint poses from RViz, motion type + planner +
   **execution** + speed per move, the **dynamic-object management** (which objects the
   gripper grasps, per-move attached-collision-check, freeze/gravity on release), and —
   for vision (v4.1, D-016) — set **"Guided by camera"** on any Move block: pick a Step-5
   detector and the waypoint's position comes from the detection at run time
   (dx/dy/dz offsets, EEF-orientation policy); the sequence shows the automatic
   "detect at cycle start" row.

   **Execution: `plan` | `servo` (ADR-0015) — what the two words mean.** Three questions
   get asked about every move, and they are different questions:

   | axis | the question | values |
   |---|---|---|
   | **Motion** | what SHAPE does the path have? | `ptp` · `lin` · `circ` · `free` |
   | **Planner** | HOW is free space searched? | `pilz` · `ompl` · `ompl_chomp` |
   | **Execution** | is there a PLAN at all? | `plan` · `servo` |

   The first two are questions about a plan. Execution is the question of whether a plan
   exists, which is why it is its own axis and not a fourth motion type.

   The planner values are not TSA's list: they are the modes TMR's planner registry
   (`trainit_motion_runtime/config/planner_profiles.yaml`) offers, read from TSA's committed
   byte copy `catalog/tmr_planner_registry.yaml` — never from the installed TMR (ADR-0021). After
   TMR's registry changes, `python3 -m trainit_setup_assistant.catalog.refresh` updates the copy
   and the catalog snapshot, and `test_planner_registry.py` fails until it has run.

   - **`plan` (the default) — decide the whole path first, then run it.** MoveIt computes
     a complete, collision-checked, time-parameterised trajectory to the goal and the
     controller executes it. You get the planner's guarantee: the arm's *whole route* was
     checked against the planning scene before it moved. The cost is that the goal is
     fixed at planning time. Emitted as `<MoveWaypoint>`, and it costs **zero extra bytes**
     in the bundle — an all-`plan` application generates byte-for-byte what a pre-ADR-0015
     TSA generated.
   - **`servo` — no path is planned; the goal is chased.** MoveIt Servo converts a stream
     of velocity commands into controller set-points at 100+ Hz, so the arm follows a goal
     that keeps moving. There is no trajectory, so there is no route to check in advance
     (Servo checks only collisions it is approaching, against the same planning scene).
     Emitted as a *different node*, `<ServoTrack>` — explicit in Groot2 — reading the same
     `bt_params.yaml` waypoint block. The bundle then also gets
     `<robot>_trainit_config/config/servo.yaml` and a `servo_node` in the cell bring-up,
     both **only** when at least one move chose it.

   **When each is right.** `plan` for every move that goes somewhere: the travel into a
   bin, a transfer across the cell, anything that has to get past the cell's furniture.
   `servo` only for a move whose GOAL MOVES while the arm is on its way — tracking a
   conveyor, or closing the last centimetres onto a detection that is still being
   refreshed. A streamed move to a pose that was taught once and never moves buys nothing
   and gives up the planner's route, so the assistant warns about it; `servo` with
   `lin`/`circ` is refused outright, because those shapes are Pilz LIN/CIRC and MoveIt
   Servo has no Pilz anything — one of the two choices would be silently ignored, and
   silent is the problem.

   **Arbitration is a hard rule, not a nicety.** A `JointTrajectoryController` accepts the
   servo topic while an action goal is in flight, and the two then fight over the same
   controller: the goal aborts on a tolerance violation with no useful message. **One
   owner per controller** — pause Servo before any trajectory execution on that group,
   unpause after. That is runtime work, and it belongs to TMR.

   **The end-effector dialog also carries the TOOL's geometry (v7.1, ADR-0017):** the
   **footprint radius** (`tool_radius_m` — the disc the approach has to find empty around
   the grasp point; perception measures the depth inside exactly that disc) and the
   **grip window** (`tool_reach_mm` — how far the tool can still bridge to the face and
   hold it; the reference suction cup: 8 mm). Both are properties of the tool, which is
   why they live on the end-effector and not on a detector: the same cup ranks candidates
   for every detector that watches its bin, and the grip window used to be a Python
   default inside a node — a number that **cannot be reproduced from a bundle**. Leave
   them at 0 and the clearance terms are simply not published; every other candidate rule
   is unaffected. Both are needed together: a zero grip window would make the clearance
   term identically 1.0 on every candidate, i.e. a ranking that silently collapses into
   `candidate only` — so a clearance rule with only one of the two is refused.
8. **Generate the bundle** — `<robot>_trainit_config` + `<robot>_app` +
   `<robot>_description` + a README. Build it and run — one command brings up the
   cell, one runs the app.

## A Canonical Task Program (YAML-first)

Besides the three templates, a project can describe its robot program directly:
`application.type: task_program` (`schema_version: 4`). It reads like a robot program -- WHERE
the robot goes (waypoints), HOW it gets there (move and path steps), with which limits (motion
profiles) -- and TSA checks every reference, frame, provider and limit before it writes a file.
It is written in `project.yaml` (the wizard shows it, validates it and never rewrites it). The
example is the public sample cell's `examples/ur5e_suction/task_program.yaml`; its application
(the file also gives the path a goal tolerance and sets `runtime.allowed_start_tolerance`):

```yaml
application:
  type: task_program
  motion_profiles:                       # reusable limits; % of the joint limits, TCP in m/s
    transit: {speed: 50, acceleration: 50}
    approach: {speed: 30, acceleration: 30, tcp_velocity_mps: 0.1}   # the smaller limit wins
    process: {speed: 20, acceleration: 20, tcp_velocity_mps: 0.08}
  waypoints:                             # WHERE -- never how
    - {name: home, type: joint, named: home}
    - {name: w1, frame: base, position: [-0.45, -0.25, 0.30], orientation: [0.0, 1.0, 0.0, 0.0]}
    - {name: w2, relative: {to: w1, frame: tool, dz: 0.20}}   # 200 mm along the tool axis
    - {name: w2b, frame: base, position: [-0.45, 0.0, 0.05], orientation: [0.0, 1.0, 0.0, 0.0]}
    - {name: w3, frame: base, position: [-0.45, 0.25, 0.10], orientation: [0.0, 1.0, 0.0, 0.0]}
    - {name: w4, frame: base, position: [-0.45, 0.25, 0.30], orientation: [0.0, 1.0, 0.0, 0.0]}
  paths:                                 # several waypoints as ONE motion
    - name: sweep
      states: [w1, w2, w3, w4]
      segments: [{type: lin, blend_radius_m: 0.02}, {type: circ, via: w2b, blend_radius_m: 0.02},
                 {type: lin}]            # the last segment never blends
      continuity: blend                  # or stop: halt at every state
      profile: process
  tasks:
    - name: validation_a
      steps:
        - move: {to: home, type: ptp, profile: transit}
        - move: {to: w1, type: ptp, profile: transit}
        - move: {to: w2, type: lin, profile: approach}
        - move: {to: w3, type: circ, via: w2b, profile: approach}
        - move: {to: w4, type: lin, profile: approach}
        - move: {to: home, type: ptp, profile: transit}
        - move: {to: w1, type: ptp, profile: transit}   # a path starts where the robot is
        - path: sweep
        - move: {to: home, type: ptp, profile: transit}
  main_task: validation_a
```

No provider is named: each motion gets the registry's default (`pilz_ptp`, `pilz_lin`,
`pilz_circ`, `pilz_sequence` for a path, `ompl` for `type: free`), and the bundle says which one
in comments, in `config/paths.yaml` and in its `README.md`. To choose, set
`application.providers` (`ptp`, `free`, `path`, `dense_path`), or `provider:` on a ptp/free move
or a path; a provider that cannot do what the step asks (a blend, a TCP limit, a tolerance) is
refused with the step, the missing capability and the providers that have it. The generated
`<robot>_app` holds `config/bt_params.yaml` (`task_parameters`: each waypoint and the move that
reaches it), `config/paths.yaml`, `config/motion_profiles.yaml` and `bt_trees/task_program.xml`
(one BehaviorTree per task: one `MoveWaypoint` per move, one `FollowPath` per path). It needs
`trainit_motion_runtime` 2.2.0 (2.4.0 with process events, 2.5.0 with manipulation steps). Not in this
version: a TCP other than the tip link, a dense path read from a file, a `check` step.

**Process events and captures (7.11.0).** A task coordinates motion, process actions and perception
with the same generic vocabulary -- an inspection, a dispensing or a gluing program is a composition,
never a step type of its own:

```yaml
  paths:
    - name: inspection_scan
      states: [inspection_start, w1, w2, w3]
      segments: [{type: lin}, {type: lin}, {type: lin}]
      profile: inspect
      events:                  # the robot STOPS at the anchor; the event is blocking
        - {at: {state: w1}, kind: capture, detector: inspection_cam}    # -> captures.0
        - {at: {state: w2}, kind: capture, detector: inspection_cam}    # -> captures.1
        - {at: {segment: 2, when: end}, kind: capture, detector: inspection_cam}
  tasks:
    - name: inspect
      steps:
        - move: {to: inspection_start, type: ptp, profile: transit}
        - process: {kind: process_on, channel: inspection_light}        # a LOGICAL channel id
        - path: inspection_scan
        - process: {kind: process_off, channel: inspection_light}
```

An event's `kind` is `process_on`, `process_off`, `set_value` (`value`), `external` (a logical
action, no acknowledgement) -- each on a `channel` that is a LOGICAL id (`dispenser`,
`washer_start`), never a ROS topic, a hardware address or a controller symbol -- or `capture`
(`detector`, optional `out`, `class_id`, `timeout_ms`) or, on a path, `dwell` (`ms`). A path event is
anchored at a state (`after_arrival`, the default, or `before_departure`) or at a segment's `start` /
`end`; one inside a segment (a fraction, a distance, a time) or `sync: async` is refused before any
file (R-M3-09). A capture triggers its detector on demand and keeps only a result taken after it
began, on the blackboard as `captures.<n>` (counted per task in execution order) -- nothing is
recorded. The path stays ONE path of `config/paths.yaml`; the tree runs it as event-free
`FollowPath` runs with the event nodes between them. A program with channel events runs the ROS
process controller (`<channel>/process_enabled | process_value`, latched). The inspection example is
`test/data/m3_5/validation_b.yaml`.

**Manipulation (7.12.0).** A pick and place reads as three steps; any grasp producer feeds the same
grasp (ADR-0026):

```yaml
  waypoints:
    - {name: pick_nominal, position: [0.4, 0.0, 0.3], orientation: [1, 0, 0, 0]}   # how the tool is held
    - {name: place, position: [0.3, -0.3, 0.15], orientation: [1, 0, 0, 0]}
  tasks:
    - name: pick_and_place
      steps:
        - detect: {detector: grasp_provider, source: candidates}       # -> the candidate set
        - grasp:                                                       # the best candidate of it
            orientation_from: pick_nominal
            standoff_m: 0.05
            profile: transit             # the PreGrasp transfer (PTP)
            approach_profile: approach   # the Approach and the Retract (LIN)
            retract: {along_approach: 0.1}
        - release: {at: place, standoff_m: 0.05, profile: transit, approach_profile: approach,
                    retract: {dz: 0.08}}
```

- `detect`: `source: candidates` reads a grasp producer's candidates
  (`trainit_interfaces/GraspCandidateArray`); `source: detections` projects a detector's detections onto
  the same contract (approach = -surface normal, into the surface; `approach: [x, y, z]`, base axes,
  when a detection has no fitted normal). `keep` (1), `out` (default `candidates`), `class_id`
  (detections), `timeout_ms`.
- `grasp`: PreGrasp (PTP) -> Approach (LIN) -> close the grip -> verify as the end-effector contract
  requires -> Retract (LIN, `along_approach` or `dz`). From a candidate set (`candidates`, default the
  task's latest detect; `index`, default 0, the best): the PreGrasp is `standoff_m` (else the
  candidate's own) back along the candidate's approach; the tool's orientation is `align_approach`
  -- the `orientation_from` waypoint (a pose in the base frame) tilted only as much as needed for its
  +Z to travel along the approach; a reference pointing exactly against the approach is refused. At
  a taught target (`at`), the approach is the target's own tool +Z (`target_local`). `eef` / `site`
  default to the contract's only device and site; the site's TCP must be the tip link.
- `release`: at a taught target, the same idiom with the grip opened and the release verified as the
  contract requires. `tool`: `actuate: {actuator, target}` (a named target; a grip is opened and closed
  by grasp and release) or `verify: {site?, expect: held | released}`.
- A failed verification stops the task. The reference example is `test/data/m3_6/validation_c.yaml`
  (and `validation_c_detections.yaml`: the same grasp fed by a detector).

**Task composition (8.0.0).** A program of several tasks: a task CALLS another, and three control
steps say how often and in which order a BODY of steps runs -- an automation engineer's subroutine,
loop, retry and alternative, with no condition, variable or expression of their own:

```yaml
  tasks:
    - name: cell_cycle
      description: two parts from the pick to the place, then a recovery to home
      steps:
        - move: {to: home, type: ptp, profile: transit}
        - repeat:                        # the body, 2 times (-1: until a step fails or you stop it)
            cycles: 2
            body:
              - task: transfer           # call: transfer's steps run, then the next step
        - fallback:                      # alternatives, in order, until one succeeds
            bodies:
              - - retry:                 # the body again from its first step when one fails
                    attempts: 2          # tries at most, the first included
                    body:
                      - tool: {verify: {expect: held}}
              - - task: go_home
    - name: transfer
      steps:
        - grasp: {at: pick, standoff_m: 0.05, profile: transit, approach_profile: approach, retract: {dz: 0.05}}
        - release: {at: place, standoff_m: 0.05, profile: transit, approach_profile: approach, retract: {dz: 0.05}}
    - name: go_home
      steps: [{move: {to: home, type: ptp, profile: transit}}, {dwell: {ms: 200}}]
  main_task: cell_cycle
```

- A step inside a body is the same step as anywhere else (same checks, same refusals, same nodes); it
  is named by its label: `3.1` (step 1 of the body of step 3), `5.2.1` (step 1 of the SECOND
  alternative of the fallback at step 5).
- A failing step fails its body: a `repeat` stops with that failure, a `retry` runs the body again while
  tries are left, a `fallback` runs its next alternative. ANY failure counts -- a verification verdict,
  a target the planner cannot reach, a motion fault (an end-effector fault stays latched: the run still
  ends with exit code 3).
- `inputs` / `outputs` (`{name, type}`, type `candidates` -- a detect step's set -- or `capture`) are the
  data a task takes and gives, BY NAME: the application has one blackboard, so a called task reads the
  set its caller holds under that name. TSA checks every call: the input is held there, what the called
  grasps need of it (`keep`, a standoff for projected detections), an output the task produces on every
  way it succeeds; a set a called task writes without declaring it is not held after the call; the
  calls form no cycle (`a -> b -> a` is refused, naming the chain). The main task takes no input.
- `config/tasks.yaml` is the manifest: each task's tree (`task_<name>`), description, inputs and
  outputs. Nothing reads it to run the program (`MainTree` runs `main_task`); it is the seam for a later
  task server. A task is ONE BehaviorTree, called with `SubTree`; the control steps are
  BehaviorTree.CPP's `Repeat`, `RetryUntilSuccessful` and `Fallback`. No new runtime: the floors are the
  steps' own. The example is `test/data/m3_7/composition.yaml`; the six M3 decompositions (machine
  tending, inspection, a process path, pick and place, bin picking, bottle nesting) are
  `test/data/m3_7/decompositions/`.
- The application types `gluing`, `follow_path`, `waypoint_replay` and `cnc` -- names that never
  generated a bundle -- are refused at load: each is a Canonical Task Program.

## Running a generated bundle

**The first time**, install what the bundle needs. From the workspace whose `src/` holds the
bundle, with TrainIt sourced (Install, step 2):

```bash
rosdep install --from-paths src --ignore-src -r -y
colcon build
source install/setup.bash
```

`rosdep` installs what the bundle's `package.xml` files declare (ros2_control's controller
manager and controllers, MoveIt, the controller manager the bundle's `moveit_controllers.yaml`
names, RViz) and finds the TrainIt packages in the sourced workspace. A bundle generated by TSA
7.8.0 or older (`generator_version` in its `generation_manifest.yaml`) does not declare
`moveit_simple_controller_manager`: for it, also `sudo apt install
ros-humble-moveit-simple-controller-manager`, or move_group executes nothing. A package the robot
description uses (`$(find ...)`, `package://...`) is not declared either: install it yourself
(`ur_description`, for the public sample).

Which package brings up the cell depends on how the bundle was made.

**Built on a base MoveIt config** (`robot.base_moveit_config_path` set: the wizard's flow, the
reference cell). The cell bring-up is `<robot>_trainit_config`'s:

```bash
# terminal 1 — cell (move_group + planners + scene + controllers/bridges + RViz)
ros2 launch <robot>_trainit_config bringup.launch.py mode:=isaac     # or mock | real
# terminal 2 — the application (Behaviour Tree); on mock | real add use_sim_time:=false
ros2 launch <robot>_app trainit_bt.launch.py planner_mode:=pilz      # or ompl | ompl_chomp
```

Or both with one command: `ros2 launch <robot>_app bringup.launch.py mode:=isaac` brings up the
cell and starts the tree `bt_delay` seconds later (8, or 15 where gazebo is deployed;
`bt:=false` brings up the cell alone).

**From scratch** (no base config, such as the public sample). `<robot>_trainit_config` holds
configuration only; the cell bring-up is the app's:

```bash
# terminal 1 — cell (move_group + controllers + end-effector adapters + RViz; rviz:=false: none)
ros2 launch <robot>_app bringup.launch.py mode:=mock                 # or another deployed mode
# terminal 2 — the application (Behaviour Tree)
ros2 launch <robot>_app trainit_bt.launch.py use_sim_time:=false     # mock and real
```

In both kinds, `trainit_bt.launch.py` launched on its own defaults `use_sim_time` to true, the
simulator's clock: on `mock` and `real`, which publish no `/clock`, give it `use_sim_time:=false`
(the one-command bring-up above sets it from the mode). `groot_monitor:=false` turns off the
Groot2 live monitor, which needs libzmq and a free port. The public sample's README walks through
every step, with the log lines to expect.

> **`mode:=real` is DEVELOPMENT / NON-PRODUCTION-READY.** It starts move_group (on a base
> config, also the scene manager) and **no ros2_control**: the arm moves only if the project's
> vendor bridges run in `real` (`deployment.real_include`, or, on a base config, a bridge whose
> `modes` include `real`). On the reference cell neither is configured, so as generated it can
> plan and never move; the FAIRINO bridges are started by hand, have never driven a generated
> bundle end to end, and execute only a trajectory's last waypoint (TESTING /
> `test_reference_real_mode_contract.py`).

Every degree of freedom is data: waypoints/planner/speed in `<robot>_app/config/bt_params.yaml`,
obstacles + dynamic-object flags in `<robot>_trainit_config/config/scene.yaml` (from scratch:
the tree's `AddCollisionObject` boxes, in `<robot>_app/bt_trees/`), and — only
for an application with a `servo` move — the MoveIt Servo settings in
`<robot>_trainit_config/config/servo.yaml` (the cell bring-up starts the `servo_node` with
it). See the generated bundle's own `README.md` (a bundle built on a base config has one).

With a **robot namespace** set, the same two commands need the namespace nowhere: the
generated launches carry it, and the bring-up prints it on the first line. Only RViz is
pointed at the robot by hand (its MotionPlanning display's *Move Group Namespace*), for
the plugin reason above.

## CLIs (headless)

Each runs as `ros2 run trainit_setup_assistant <command>`: an `ament_python` package installs
its commands under `lib/trainit_setup_assistant`, not on `PATH`.

- `trainit_generate <project.yaml> -o <dir>` — generate the bundle. A project without an
  end-effector contract needs no ROS. One with a contract compiles its robot description first
  (R9), so it needs `xacro` and a sourced environment that finds every package the description
  uses (`ur_description`, for the public sample). A relative path in a project resolves against
  the current directory, not against the project file. A `<dir>` that is not empty is refused;
  `--force` deletes it first.
- `trainit_verify <project.yaml> --golden <dir> [--build]` — equivalence + optional build.
- `trainit_compute_collisions <project.yaml> [--write]` — the self-collision matrix
  (`robot.disable_collisions`), from the MoveIt Setup Assistant's `collisions_updater`, so it
  needs ROS: printed, or with `--write` saved into the project, which rewrites the file without
  its comments.
- `trainit_isaac_kit --into <dir>` — copies the reusable Isaac-cell tooling (the stage validator,
  the collision-mesh exporter and two Script Node templates) into an existing `<cell>_isaac`
  package; a file already there is kept unless `--force`.
- `trainit_setup_assistant` — the Qt wizard.

## Version notes

> **Version note.** The package version is `8.0.0` and the product label is "TSA v8" —
> one number, deliberately. TSA was called v6 in commits while `package.xml` said 4.3.1,
> so the two were already telling different stories; the jump is cosmetic (nothing in the
> generated output changes with it) and is taken once so it is never ambiguous again
> (ADR-0016 Decision 5).
> `7.1.0` adds the configurable candidate ranking (ADR-0017) — additive, and nothing in
> an existing bundle's output changes with it. `7.2.0` adds the grasp-face picker and makes
> the ranking's face parameters come from `trainit_perception`'s registry instead of a list
> inside TSA (ADR-0018): also additive, with one thing to know — a project that *measures*
> its grasp face writes the measured radius where the bounding-box approximation used to
> be, so `visible_fraction` changes for every candidate in that cell. That is the point of
> the change, and it is opt-in per project: measure nothing and the output is byte-identical.
> `7.3.0` reads the planner vocabulary from a committed copy of TMR's registry, and `7.4.0`
> reads trainit_perception's vocabulary from a committed snapshot (ADR-0021): no generated
> byte changes, and generation no longer depends on which perception package is installed.
> `7.5.0` makes a bundle say what compiled it (ADR-0021, amendment M2.3): its
> `generation_manifest.yaml` records `capability_snapshot` — the versions and sha256 hashes of
> the committed catalog snapshot it was generated with — and names a copied file by its package
> (`package://<package>/<path>`, or `external:<file name>`) instead of this machine's path. Only
> the manifest changes; every other generated byte is the same.
> `7.6.0` makes a bundle say what it needs to RUN (ADR-0021, amendment M2.4), in its
> `package.xml` files: every `trainit_motion_runtime` dependency carries `version_gte` — the
> oldest TMR version that has every capability that package's files use (1.4.0 to 1.7.0
> today), with a comment naming what set it — and an app whose tree runs a policy step, or
> whose launch files start the policy runtime, declares `trainit_policy_runtime`.
> `trainit_perception` is unchanged. Only `package.xml` lines and the manifest change.
> `7.7.0` adds the end-effector CONTRACT (ADR-0022), opt-in per project: `robot.gripper.contract`
> describes the device -- its actuators on standard interfaces (SetBool, GripperCommand,
> FollowJointTrajectory), its grasp sites, the verification policy and one adapter binding per
> backend -- in three lines for a simple suction cup, or as maps for a composite tool. TSA refuses
> an incoherent contract before writing a file (R1-R23), and a canonical bundle carries an `eef`
> parameter block, `VerifyGrasp`/`CommandActuator` per the policy, the TrainIt adapter rows in its
> bring-up, `trainit_hardware_layer >= 0.2.0` and TMR >= 2.0.0. A composite BUNDLE is refused in
> 7.7.0. A project without a contract generates exactly what 7.6.0 generated, and saves
> byte-identical.
>
> **Opting in by hand-editing `project.yaml`?** Add the two silent-downgrade guards with the
> `contract:` block: `schema_version: 3` at the top and `actuation: contract` under
> `robot.gripper` -- or open the project and save it once with 7.7.0, which writes both. Without
> them an OLDER TSA does not refuse the file: it drops the `contract` block it cannot read and
> silently generates a LEGACY bundle, one without an end-effector server for the contract's
> devices (T11 finding F3). With them TSA 7.6.0 refuses the file at load (`contract` is not an
> actuation it knows), and a file an even older TSA re-saved without its contract is refused by
> 7.7.0 (`schema_version` 3 without a contract, R19).
>
> `7.8.0` (vNext M2.7) hardens what 7.7.0 started, for canonical projects; a project without a
> contract generates what 7.7.0 generated, except its manifest's provenance, and keeps 7.7.0's
> reading of a blank planner (nothing is written for it, and D-016 and the GUI's planner switch
> read it as the global mode). In a canonical project the planner a move RUNS is the one
> generation writes (owner decision D3, scoped to canonical projects at vNext G0): a blank `ptp`
> planner is written as `pilz` (TMR runs Pilz PTP there) and a blank `free` move as the project's
> global planner mode (the mode every generated launch hands TMR's runner), and D-016 and the
> GUI's planner switch judge that same planner, so its blank moves gain `planner:` lines in
> `bt_params.yaml` -- what TMR runs is written, never inherited from the move before. For
> canonical projects, a composite
> BUNDLE is generated for MOCK (R22 narrowed: every deployed mode mock and served by
> `trainit_mock`, the mock bring-up's robot description compiled, no `<ros2_control>` block
> claiming a tool joint, no pseudo-joint echo, every movable tool joint published; isaac, gazebo,
> real and mixed bindings stay refused until vNext M6), its tool joints reaching the robot's own
> `joint_states` through one relative launch remap. Three refusals are new (now R1-R26): an
> actuator that moves a joint on the path to a site's TCP is `follow_joint_trajectory`, on a
> single-DOF joint (R24); no such joint is left without one (R25); under a cell namespace, an
> absolute or private pseudo-joint echo topic is refused (R26). A refusal that marks a limit of
> the current canonical model, not an invalid design, says `UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL`
> (G0). Every `MoveWaypoint` of the
> canonical bin-picking cycle gates its motion FAULT (TMR 2.1.0's `fault` output; a streamed
> `ServoTrack` move reports none, and a note says so): a failure of the motion infrastructure ends
> the run, while a target's own failure (no IK, no plan) still moves on to the next candidate. So
> that app declares `trainit_motion_runtime >= 2.1.0`, and so does a canonical app whose runner
> runs in a cell namespace: it relies on TMR 2.1.0 resolving MoveIt there (ADR-0023; vNext G0). No
> legacy floor moves, a namespaced one included (KNOWN_DEBT). The catalog snapshot records TMR
> 2.1.0: provenance, not a floor.
>
> vNext G0 adds one refusal: a canonical bin-picking project that gives a detector other than the
> loop's an explicit Detect block is refused before any file -- the emptying loop samples once per
> cycle, before the retry, and cannot emit the block (a legacy project still drops it silently).
>
> `7.8.1` (vNext G0): a generated `<robot>_trainit_config` declares the controller manager its
> `moveit_controllers.yaml` names -- `moveit_simple_controller_manager` from scratch; on a base
> config, the released MoveIt manager the copied file names. On a machine set up by Install alone,
> `rosdep` over a 7.8.0 bundle left it out, and move_group could not load its controller manager.
> Only that `package.xml` (an `exec_depend` and its two-line comment) and the manifest change.
>
> `7.9.0` (vNext M3.2, ADR-0025): TSA reads TMR's planner registry at schema 2 (TMR 2.2.0's motion
> provider rows, their capability tokens and the default row per motion kind) and resolves a
> motion request against it: what the request requires, which provider it resolves to (its own
> choice, the application's, the registry default -- never a fallback) and a refusal before any
> file when the provider cannot honour it, naming the request, the token and the provider
> (`model/motion.py`, `model/capabilities.py`). Not yet used by any project file (M3.3's
> `task_program`): every generated byte is unchanged but the manifest's `capability_snapshot`.
>
> `7.10.0` (vNext M3.3, ADR-0024): the Canonical Task Program -- `application.type: task_program`,
> `schema_version: 4` (see [A Canonical Task Program](#a-canonical-task-program-yaml-first)):
> waypoints, motion profiles, paths and tasks of move, path and dwell steps, validated and resolved
> against TMR's providers before any file, generated as `task_parameters`, `config/paths.yaml`,
> `config/motion_profiles.yaml` and one BehaviorTree per task (needs `trainit_motion_runtime`
> 2.2.0). The wizard is YAML-first for it: it shows and validates a task program and never rewrites
> it. A template project's generated bytes are unchanged but the manifest's provenance
> (`generator_version`, `capability_snapshot`).
>
> `7.11.0` (vNext M3.5, ADR-0024): process-path semantics in a task program -- the `process` step,
> a path's `events` and the `capture` event (see above): validated and refused before any file
> (anchors at states and segment boundaries only, blocking only, a logical channel id, a triggerable
> detector), generated as `ProcessEvent`, `DetectObject trigger="true"`, `Sleep` and
> `FollowPath from= to=` runs, with `paths.yaml` recording where a path stops; needs
> `trainit_motion_runtime` 2.4.0 when used. `perception.detectors[].external_node.detect_service`
> declares an external detector's on-demand service. A program without events and every template
> bundle: unchanged but the manifest's provenance.
>
> `7.12.0` (vNext M3.6, ADR-0026): manipulation in a task program -- the `detect`, `grasp`, `release`
> and `tool` steps and `relative.frame: target_local` (see above): validated and refused before any
> file, generated as `DetectCandidates`, `SetWaypointFromCandidate`, the idioms' `MoveWaypoint`s and
> the end-effector contract's nodes, each idiom motion a derived `task_parameters` entry
> `<task>__s<i>_<role>`; needs `trainit_motion_runtime` 2.5.0 when used (which brings
> `trainit_interfaces` 0.2.0). A program without manipulation steps and every template bundle:
> unchanged but the manifest's provenance.
>
> `8.0.0` (vNext M3.7, ADR-0024; M3 decisions 6 and 7): task composition in a task program -- `task`
> calls, `inputs` / `outputs` bound by name, `repeat` / `retry` / `fallback`, an acyclic call graph, the
> `config/tasks.yaml` manifest (see above); a product label, as 7.0.0 was. BREAKING, by decision 7: the
> application types `gluing`, `follow_path`, `waypoint_replay` and `cnc` -- which never generated a
> bundle -- are refused at load, naming `task_program`. A task program's bundle gains
> `config/tasks.yaml` (and its README a row for it); no new TMR floor. Every template bundle: unchanged
> but the manifest's provenance.

## For maintainers

See **`ARCHITECTURE.md`** for the internals (the three-config model, the emitters, the BT
vocabulary, the generalizable/sim-specific boundary) and the development roadmap.
