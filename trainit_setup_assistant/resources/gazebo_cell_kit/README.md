# gazebo_cell_kit — author a Gazebo Classic cell for TrainIt (ADR-0008 F2)

The Gazebo backend is a **generic TrainIt capability** (ADR-0008). TSA generates the
`mode:=gazebo` bring-up; **you** provide the cell assets as *input* (never the other way
round — no consumer-specific logic lives in TrainIt). This kit is the Gazebo analogue of
`isaac_cell_kit`: the reusable pieces + the contract a Gazebo cell must satisfy.

## What TSA emits for `mode:=gazebo` (you don't write this)

The generated cell bring-up (`<cfg>/launch/bringup.launch.py`) with `mode:=gazebo`:

- launches **Gazebo Classic** (`gazebo_ros/launch/gazebo.launch.py`, `pause:=true`,
  `use_sim_time:=true`; `world:=` from the launch arg, `gui:=false` for headless);
- spawns the robot with **`spawn_entity.py -topic robot_description`**;
- starts **one** `robot_state_publisher` and **one** `move_group`
  (`build_move_group_params(backend="gazebo")` → the `GazeboSystem` `robot_description`);
- runs **spawners only** (`joint_state_broadcaster`, the arm controller, a parallel-gripper
  controller if any) against **`/controller_manager`** — the one the `gazebo_ros2_control`
  plugin creates **inside `gzserver`**. It never starts a standalone `ros2_control_node`.

So the invariants of ADR-0008 hold by construction: **one** controller_manager (the
plugin's), **one** robot_state_publisher, **one** robot_description, **one** TF tree.

## The cell you feed TSA must provide

1. **A gazebo-capable `ros2_control` xacro.** On the from-scratch path TSA generates it
   (with the `use_gazebo` arg + the `<gazebo><plugin libgazebo_ros2_control.so>` block). On
   the **base-config path** TSA copies your hand-made `moveit_config/config/` verbatim, so
   your xacro must itself select `gazebo_ros2_control/GazeboSystem` on a `use_gazebo` arg and
   carry the `<gazebo>` plugin whose `<parameters>` point at your `ros2_controllers.yaml`.
   TrainIt **preserves your controller names** — a real cell (e.g. a UR arm) may use
   `ur_controllers/ScaledJointTrajectoryController`, `GPIOController`,
   `SpeedScalingStateBroadcaster`, `ForceTorqueStateBroadcaster`; TSA drives *those*, it does
   not synthesise generic ones. Make the spawner set (in `DeploymentSpec.arm_controller` and
   the gripper) match the names in that YAML.
2. **A `.world`** with a ground plane, your workcell, and (for a grasp demo) a graspable
   object. Pass it as `world:=$(find <your_pkg>)/worlds/<cell>.world`. A minimal one is in
   `worlds/trainit_pick_cell.world`.
3. **A grasp mechanism.** A physics gripper models contact; a **suction** cell welds the
   object with the **IFRA_LinkAttacher** Gazebo plugin. Add the weld bridge
   (`scripts/link_attacher_bridge.py`) as a normal cell **bridge** in your project:

   ```yaml
   deployment:
     modes: [mock, isaac, gazebo, real]
     bridges:
       - package: <your_pkg>
         executable: link_attacher_bridge.py
         modes: [gazebo]            # runs only in the Gazebo backend
         parameters:
           - {object_model: red_box, object_link: link,
              robot_model: <robot_name>, ee_link: suction_link}
   ```

   It exposes the TrainIt suction contract (`/suction/on` → attach, `/suction/off` →
   detach), so the existing BT `CloseGripper`/`OpenGripper` drive the weld unchanged.

## Dependencies (vcs, like the rest of TrainIt's externals)

- `ros-humble-gazebo-ros-pkgs`, `ros-humble-gazebo-ros2-control` (apt).
- **IFRA_LinkAttacher** (`ros2_linkattacher` + `linkattacher_msgs`) — clone into `src/`
  (`https://github.com/IFRA-Cranfield/IFRA_LinkAttacher`, humble). Only the `gazebo` backend
  needs it; mock/isaac/real do not.

## Run the F2 Definition-of-Done smoke (you generate the bundle; TSA does not)

```bash
# 1. generate the bundle from your gazebo-capable project (MANUAL TSA step)
# 2. build + source, then headless:
ros2 launch <robot>_app bringup.launch.py mode:=gazebo gui:=false rviz:=false \
    world:=$(ros2 pkg prefix <your_pkg>)/share/<your_pkg>/worlds/trainit_pick_cell.world

# expect: gzserver up; ONE /controller_manager, ONE /robot_state_publisher; then
ros2 control list_controllers      # joint_state_broadcaster + arm (+ gripper) = active
ros2 node list | grep -c controller_manager   # -> 1
# move_group plans+executes the BT; a CloseGripper welds the object (LinkAttacher).
```

See `docs/adr/0008-backend-abstraction-and-gazebo.md` and
`docs/integration/ROS2ML_INTEGRATION_ASSESSMENT.md` §5.
