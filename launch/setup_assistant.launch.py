#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# TrainIt Setup Assistant — LIVE CONFIG SESSION.
#
# Brings up a real move_group + mock ros2_control + RViz (MoveIt MotionPlanning
# plugin) for an ALREADY-GENERATED-AND-BUILT moveit_config, so you can move the
# end-effector with the real interactive-marker gizmo (real, collision-aware IK)
# and capture poses. This is the RViz-native equivalent of the MoveIt Setup
# Assistant's interactive view — "what you configure is what you ship", because it
# uses the engine's own build_move_group_params().
#
# Typical flow:
#   1) bootstrap a bundle for your robot (robot only, empty app), e.g. via
#      trainit_generate on a project.yaml produced by load_robot_spec();
#   2) colcon build it;
#   3) ros2 launch trainit_setup_assistant setup_assistant.launch.py \
#         moveit_config_package:=<robot>_moveit_config robot_name:=<robot>
#
# This is lean (no gripper/cell bridges) — just enough to jog + capture poses.
# -----------------------------------------------------------------------------
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (DeclareLaunchArgument, OpaqueFunction,
                            RegisterEventHandler)
from launch.event_handlers import OnProcessExit
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from trainit_motion_runtime.trainit_moveit_config import build_move_group_params


def launch_setup(context, *args, **kwargs):
    moveit_config_package = LaunchConfiguration('moveit_config_package').perform(context)
    robot_name = LaunchConfiguration('robot_name').perform(context)
    arm_controller = LaunchConfiguration('arm_controller').perform(context)
    use_rviz = LaunchConfiguration('rviz').perform(context).lower() in ('true', '1')
    mode = LaunchConfiguration('mode').perform(context)

    if not moveit_config_package:
        raise RuntimeError('moveit_config_package:=<pkg> is required')
    if mode not in ('mock', 'isaac'):
        raise RuntimeError(f"mode must be 'mock' or 'isaac' (got '{mode}')")

    # mock: fake hardware (RViz only). isaac: TopicBasedSystem bridge -> jog the gizmo
    # in MoveIt and the robot moves in the running Isaac scene (Isaac must be PLAYing,
    # publishing /isaac_joint_states + /clock).
    use_isaac = (mode == 'isaac')
    params, mc = build_move_group_params(
        robot_name=robot_name, moveit_config_package=moveit_config_package,
        use_isaac=use_isaac)
    common = {'use_sim_time': use_isaac}

    moveit_share = get_package_share_directory(moveit_config_package)
    ros2_controllers_path = os.path.join(moveit_share, 'config', 'ros2_controllers.yaml')
    rviz_config_path = os.path.join(moveit_share, 'config', 'moveit.rviz')

    nodes = [
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[mc.robot_description, common]),
        Node(package='moveit_ros_move_group', executable='move_group',
             output='screen', parameters=[params, common]),
        # ros2_control (mock fake-hw, or isaac TopicBasedSystem) so the gizmo can
        # plan + execute and you see motion (in RViz, and in Isaac when mode:=isaac)
        Node(package='controller_manager', executable='ros2_control_node',
             output='screen',
             parameters=[mc.robot_description, ros2_controllers_path, common]),
    ]

    controllers = ['joint_state_broadcaster', arm_controller]
    spawners = [
        Node(package='controller_manager', executable='spawner',
             arguments=[c, '--controller-manager', '/controller_manager',
                        '--controller-manager-timeout', '60'],
             output='screen')
        for c in controllers
    ]
    nodes.append(spawners[0])
    for prev, nxt in zip(spawners, spawners[1:]):
        nodes.append(RegisterEventHandler(OnProcessExit(target_action=prev, on_exit=[nxt])))

    if use_rviz:
        nodes.append(Node(
            package='rviz2', executable='rviz2', output='log',
            arguments=['-d', rviz_config_path],
            parameters=[params, common]))

    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('moveit_config_package',
                              description='generated moveit_config package to configure'),
        DeclareLaunchArgument('robot_name', default_value='robot',
                              description='URDF/MoveItConfigsBuilder robot name'),
        DeclareLaunchArgument('arm_controller', default_value='moveit_joint_controller',
                              description='arm JointTrajectoryController name'),
        DeclareLaunchArgument('mode', default_value='mock',
                              description='mock (RViz only) | isaac (drive the Isaac scene)'),
        DeclareLaunchArgument('rviz', default_value='true'),
        OpaqueFunction(function=launch_setup),
    ])
