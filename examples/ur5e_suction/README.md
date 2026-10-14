# The public sample cell: a UR5e with a suction cup

A complete TrainIt cell you can generate, build and run with no robot of your own. A Universal
Robots UR5e (the arm comes from `ur_description`) with a simple suction cup picks a 5 cm box at
one spot above a table and places it at another. It runs on `mode:=mock`: ros2_control's mock
hardware and the end-effector mock adapter, with no simulator and no GPU.

This directory is not a ROS package. It is what TSA generates from: a project and a robot
description. Step 1 generates the cell's three ROS packages (the bundle) from it, into a workspace
of your own.

| File | What it is |
|---|---|
| `project.yaml` | the TSA project: robot, end-effector contract, scene, waypoints, motions |
| `urdf/ur5e_suction.urdf.xacro` | the robot description: UR's `ur_robot` macro plus the tool |
| `urdf/suction_tool.xacro` | the suction tool: a body and a cup, the TCP at the cup lip |

Nothing of `ur_description` is copied here. The arm's macro, its parameters and its meshes are
read from the installed package (`$(find ur_description)` and `package://ur_description/...`).

## What you need

- The TrainIt Community workspace, built as the
  [Install section](../../README.md#install-trainit-community-from-source) of the TSA README says:
  ROS 2 Humble, `trainit_interfaces`, `trainit_motion_runtime` 2.0.0 or newer,
  `trainit_hardware_layer` 0.2.0 or newer and `trainit_setup_assistant` 7.8.1 or newer (an older
  one: also `sudo apt install ros-humble-moveit-simple-controller-manager`). The commands below call
  that workspace `~/trainit_ws`, as Install's layout A does. In layout B, TrainIt inside your
  robot workspace, put that workspace's path wherever they say `~/trainit_ws`.
- One apt package, once per machine:

  ```bash
  sudo apt install ros-humble-ur-description
  ```

  `ur_description` is the arm: generation compiles the robot description, and the bring-up and
  RViz read the arm's meshes from it. It was tested with version 2.13.0, and 2.14.0 gives the same
  description byte for byte. No generated package declares it (a description's own packages are
  the user's), so the `rosdep` of step 2 does not install it.

In every new terminal, first:

```bash
source /opt/ros/humble/setup.bash
source ~/trainit_ws/install/setup.bash
```

Once step 2 has built the bundle, also `source ~/trainit_sample_ws/install/setup.bash`.

## 1. Generate the bundle, from this directory

Every path in `project.yaml` is relative to this directory, and TSA resolves a relative path
against the directory you run it in, not against the project file. So run it from here. Run from
anywhere else, or before `ur_description` is installed, and generation refuses with
`the robot description cannot be read`. Its brackets tell the two apart: `FileNotFoundError` is
the wrong directory, and `package 'ur_description' not found` means the apt line of "What you
need" has not run.

The bundle goes into the `src/` of a new workspace of its own, `~/trainit_sample_ws`, outside
this repository and outside `~/trainit_ws`. Give `-o` that full path: a relative `-o` would write
the bundle in here.

```bash
cd ~/trainit_ws/src/trainit_setup_assistant/examples/ur5e_suction
mkdir -p ~/trainit_sample_ws/src
ros2 run trainit_setup_assistant trainit_generate project.yaml \
  -o ~/trainit_sample_ws/src/ur5e_suction
```

You should see `Generated 26 files into ...`, then the three packages `ur5e_suction_description`,
`ur5e_suction_trainit_config` and `ur5e_suction_app`, and no `warning:` line.

Run it again into the same directory and it refuses: `output ... is not empty (use --force to
overwrite)`. With `--force` it first deletes that whole directory, then generates it again.

## 2. Install what the bundle needs, and build it

```bash
cd ~/trainit_sample_ws
rosdep install --from-paths src --ignore-src -r -y
colcon build --base-paths src
source install/setup.bash
```

`rosdep` installs what the bundle's `package.xml` files declare: ros2_control's controller
manager and controllers, the topic-based hardware interface, MoveIt's move_group and planners,
RViz and the joint sliders. It finds the TrainIt packages in the sourced `~/trainit_ws`, so it
leaves them alone. It also installs `moveit_simple_controller_manager`, the controller manager
the bundle's `moveit_controllers.yaml` names (declared since TSA 7.8.1). It does not install
`ur_description`: that is the apt package above. colcon should end with `Summary: 3 packages
finished`.

Then, in the same terminal, check that the machine now has what the cell runs on:

```bash
if ! command -v ros2 >/dev/null; then echo 'MISSING ros2: source the setup files above first'
else
  for p in controller_manager joint_state_broadcaster joint_trajectory_controller \
           topic_based_ros2_control moveit_simple_controller_manager ur_description; do
    if ros2 pkg prefix $p >/dev/null 2>&1; then echo "OK      $p"
    else echo "MISSING $p: the apt line of What you need, then this step's rosdep install"; fi
  done
fi
```

Optional, the robot alone with joint sliders:

```bash
ros2 launch ur5e_suction_description view_robot.launch.py
```

## 3. Run it

Terminal 1, the cell on mock: move_group, ros2_control with its mock hardware, the end-effector
mock adapter and RViz. Add `rviz:=false` to leave RViz out, on a machine with no display.

```bash
ros2 launch ur5e_suction_app bringup.launch.py mode:=mock
```

The cell is up once it has logged these three lines:

```
You can start planning now!
Configured and activated moveit_joint_controller
trainit_mock adapter for /eef/main (ADR-0022): actuators [vacuum: set_bool]; ...
```

On the way, terminal 1 also prints WARN and ERROR lines that are expected: "Lines you can ignore",
below, says what each one is.

Terminal 2, the application, once. In a new terminal one `source` is enough: the sample
workspace's setup file also sources ROS 2 and `~/trainit_ws`, which were sourced when step 2 built
it.

```bash
source ~/trainit_sample_ws/install/setup.bash
ros2 launch ur5e_suction_app trainit_bt.launch.py use_sim_time:=false groot_monitor:=false
```

Pass `use_sim_time:=false` on mock: the application's launch defaults `use_sim_time` to true, for
a simulator's `/clock`, and mock publishes none. `groot_monitor:=false` turns off the Groot2 live
monitor, which needs libzmq and a free port.

The application then logs these lines, in this order (each after the node's log prefix, and among
others), and its launch ends by itself when the tree ends (the automated check below allows three
minutes for it):

```
end-effector contract: canonical, devices [main] (ADR-0022)
added collision box 'table'
MoveWaypoint 'home' (...)
MoveWaypoint 'pre_pick' (...)
MoveWaypoint 'pick' (...)
CloseGripper /eef/main/sites/main (vacuum -> engage): SUCCESS
attached 'box' to 'tcp'
MoveWaypoint 'post_pick' (...)
MoveWaypoint 'pre_place' (...)
MoveWaypoint 'place' (...)
OpenGripper /eef/main/sites/main (vacuum -> release): SUCCESS
detached 'box'
MoveWaypoint 'post_place' (...)
MoveWaypoint 'home' (...)
BT finished: SUCCESS
```

**Lines you can ignore.** Both terminals print much more than these lines, and a few of the
others look like problems without being any:

- `WARNING:root:Cannot infer SRDF from ... -- using config/ur5e_suction.srdf`, near the top of
  both launches: MoveIt's `MoveItConfigsBuilder` looks for the `.setup_assistant` file that the
  MoveIt Setup Assistant writes into a config package. TSA writes none, so the builder falls back
  to `config/ur5e_suction.srdf`, which is the SRDF TSA generated.
- `Using load_yaml() directly is deprecated. Use xacro.load_yaml() instead.`, near the top of both
  launches: xacro's notice about the bundle's own `config/ur5e_suction.ros2_control.xacro` (in
  `ur5e_suction_trainit_config`), which reads the arm's initial joint positions with the bare
  `load_yaml()`. xacro still reads them.
- `No root/virtual joint specified in SRDF. Assuming fixed joint`, an INFO line in both
  terminals: the generated SRDF declares no virtual joint, so MoveIt keeps the description's root
  link, `world`, fixed. That is right for an arm that stays where it is bolted.
- `Resolution not specified for Octomap. Assuming resolution = 0.1 instead` (WARN) and
  `No 3D sensor plugin(s) defined for octomap updates` (ERROR), in both terminals: the octomap is
  MoveIt's voxel map of what a depth camera sees, and no camera feeds this cell's MoveIt, whose
  configuration names no `sensors` and no `octomap_resolution`. The planning scene holds only the
  collision objects: the table and the box.
- `[Deprecated] Passing the robot description parameter directly ...` and
  `"allow_nonzero_velocity_at_trajectory_end" is set to true` (WARN), in terminal 1: the bring-up
  passes the robot description to ros2_control's controller manager as a parameter, and the
  trajectory controller keeps its default, `allow_nonzero_velocity_at_trajectory_end: true`. Both
  work on Humble; the notices are about what a later ros2_control release changes.
- `Could not enable FIFO RT scheduling policy ... (Operation not permitted)` (WARN), in terminal 1:
  the controller manager asks for real-time priority, which a user without real-time rights, or a
  container, does not get. It then runs at normal priority, which mock hardware does not need.
- `Failed loading deceleration limits` (WARN), twelve times in terminal 1: MoveIt's Pilz planner
  reads the joint limits three times in move_group, for the planner and for its two sequence
  capabilities. The second and third reads find the deceleration parameter already declared and
  print this once per joint. This cell sets no deceleration limit, so every read ends the same
  way: Pilz decelerates at the acceleration limit.
- `Tried to remove world object 'box', but it does not exist in this scene.` (WARN), in terminal
  2 right after `detached 'box'`: TMR releases the box in two steps, a detach in move_group's
  scene, which puts the box back into its world, and a removal from that world, which is why the
  box disappears in RViz. The application keeps its own copy of the scene to check each plan
  against, and the removal reaches that copy while the box is still attached there, so the copy
  has no world object to remove. The application takes a fresh copy from move_group before its
  next move.

In RViz, the table appears under the robot when the application starts. The arm goes above the
pick spot and moves straight down. The box appears under the cup, and the arm lifts it, swings to
the place spot and moves down. The box disappears there, because the planning scene forgets a
released box, and the arm goes home.

To stop the cell, press Ctrl-C in terminal 1. move_group does not always stop cleanly: launch may
then report `process has died [... exit code -2 ...]` for it, or `exit code -11` after a
`class_loader.ClassLoader: SEVERE WARNING!!!` line, a stack trace and a `Segmentation fault` line.
It happens while move_group shuts down, and nothing is left running.

Optional, the cup by hand (with the cell up and the application not running):

```bash
ros2 topic echo /eef/main/state --once
ros2 service call /eef/main/actuators/vacuum/set_engaged std_srvs/srv/SetBool "{data: true}"
```

To remove everything steps 1 and 2 made: `rm -rf ~/trainit_sample_ws`. The apt packages stay
installed.

## The automated checks

```bash
cd ~/trainit_ws/src/trainit_setup_assistant
PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
  python3 -m pytest -q -p no:cacheprovider test/test_example_ur5e_suction.py
TRAINIT_LAUNCH_TEST=1 PYTHONDONTWRITEBYTECODE=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
  PYTHONPATH=$PWD:$PYTHONPATH \
  python3 -m pytest -q -s -p no:cacheprovider test/test_example_ur5e_suction_launch.py
```

The first takes seconds: it generates the bundle into a temporary directory and checks its files.
The second takes a few minutes: it also builds the bundle in a temporary workspace, brings it up
on ROS domain 79 and runs the application. Nothing else may use domain 79 meanwhile: the test
refuses to launch on a domain that is not empty. Both delete what they generated, and
`PYTHONDONTWRITEBYTECODE=1` and `-p no:cacheprovider` keep them from writing `__pycache__` and
`.pytest_cache` into the source tree.

Without `ur_description`, every check that compiles the description skips and says so; the four
that only read the files of this directory still run. The second also skips when this machine
lacks a package the bundle declares, naming each one and the apt line that installs it. The apt
line of "What you need" and step 2's `rosdep install`, done once, install all of them.

## Making it your own

- Copy this whole directory and edit the copy. Its paths are relative to it, so run every command
  from inside the copy:

  ```bash
  cp -r ~/trainit_ws/src/trainit_setup_assistant/examples/ur5e_suction ~/my_cell && cd ~/my_cell
  ```

- Give the bundle names of its own (`bundle:`). A bundle is three packages, and colcon refuses to
  build a workspace where two packages share a name (`Duplicate package names not supported`), so
  none of the three may repeat a package of your workspace, such as your `<robot>_description`.
- The description leaves the `ros2_control` block to the bundle (`generate_ros2_control_tag` is
  false) and declares no xacro argument of its own: the bundle's wrapper declares
  `initial_positions_file` and `use_isaac`. That is why it calls UR's `ur_macro.xacro`, not UR's
  top-level `ur.urdf.xacro`. A description of your own needs the same two things: no
  `<ros2_control>` block for the arm's joints, since the bundle writes one for every joint of
  `planning_group`, and a default for every xacro argument it declares, since nothing sets them.
- For another robot, these change with it: the bundle names and `robot.robot_name`;
  `robot.description` (`urdf_dir`, absolute or relative to the copy and never with `~`, and
  `top_xacro`); `base_frame` and `tip_link`; `planning_group` (`name`, `base_link`, `tip_link`,
  `joints`); `gripper.eef_parent_link`; `named_states`, `initial_positions` and
  `joint_limits.per_joint`; `ompl.projection_evaluator`; `disable_collisions`; the scene's frames
  and `payload.attach_link`; and every waypoint.
- `robot.disable_collisions` is the self-collision matrix: UR's own pairs plus one for the tool.
  A different tool needs its own pairs. `ros2 run trainit_setup_assistant
  trainit_compute_collisions project.yaml` prints them; with `--write` it saves them into the
  project, which rewrites the file without its comments.
- Joint values are in radians. TCP positions are in metres in `base_link`. A waypoint orientation
  is `[qx, qy, qz, qw]`, and `[1, 0, 0, 0]` points the tool down. In the copy you type them, read
  off your robot while something moves it (your own MoveIt config's RViz, for example):
  `ros2 topic echo /joint_states --once` gives the joint values, and
  `ros2 run tf2_ros tf2_echo <base_frame> <tip_link>` a TCP pose, its quaternion as `(xyzw)`.

## Licences

`ur_description` is BSD-3-Clause; Universal Robots' separate terms cover only the UR8 Long, UR15,
UR18, UR20 and UR30 meshes. Nothing of it is redistributed here. The collision pairs in
`project.yaml` are transcribed from `ur_moveit_config` (Apache-2.0).
