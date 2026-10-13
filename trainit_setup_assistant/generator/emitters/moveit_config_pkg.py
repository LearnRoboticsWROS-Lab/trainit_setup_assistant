"""Emit ``*_moveit_config``: GENERATE the SRDF, TEMPLATE the config + xacros.

The SRDF's groups/states/EEF come from the model; its ``disable_collisions`` matrix is
computed by :mod:`trainit_setup_assistant.robotmodel.collision_matrix` (M2). When that
module is unavailable or fails, the SRDF still emits (over-conservative collisions) so
the package builds.
"""

from __future__ import annotations

from typing import Optional, Sequence

from ...model.enums import Backend, GripperKind
from ...robotmodel.srdf_builder import DisablePair, build_srdf
from ..namespace import cell_namespace
from ..runtime_requirements import requirement, use_of
from ..servo_yaml import build_servo_context
from .base import Emitter, GenContext


class MoveitConfigEmitter(Emitter):
    def emit(self, project, ctx: GenContext) -> None:
        robot = project.robot
        pkg = project.bundle.moveit_config_package
        group = robot.planning_group
        # Gazebo backend (ADR-0008 F2): gated so mock/isaac/real xacros stay byte-identical.
        gazebo_supported = Backend.GAZEBO in project.deployment.modes

        def cfg(name: str) -> str:
            return f'{pkg}/config/{name}'

        # A COMPOSITE device's tool joints, controllers and named states: TSA never generates
        # them (ADR-0022 F7; generating them is vNext M6). A composite mock bundle gets nothing
        # of the tool in ros2_control, the controllers or the SRDF (test F7).
        gripper_present = (
            robot.gripper.kind not in (GripperKind.NONE, GripperKind.COMPOSITE)
            and bool(robot.gripper.controller_name)
        )

        # --- SRDF (GENERATE) ---
        disable = self._compute_disable_collisions(project, ctx)
        srdf = build_srdf(project, disable_collisions=disable)
        ctx.generate_to(cfg(f'{robot.robot_name}.srdf'), srdf, source='srdf_builder')

        # --- config yamls (TEMPLATE) ---
        ctx.render_to(cfg('kinematics.yaml'), 'moveit_config/kinematics.yaml.j2',
                      group_name=group.name, kin=robot.kinematics)
        ctx.render_to(cfg('joint_limits.yaml'), 'moveit_config/joint_limits.yaml.j2',
                      jl=robot.joint_limits)
        ctx.render_to(cfg('ompl_planning.yaml'), 'moveit_config/ompl_planning.yaml.j2',
                      group_name=group.name, ompl=robot.ompl)
        ctx.render_to(cfg('pilz_cartesian_limits.yaml'),
                      'moveit_config/pilz_cartesian_limits.yaml.j2',
                      pilz=robot.pilz_cartesian_limits)
        # node keys follow the robot slot ('/**/' under a namespace, bare at the root:
        # byte-identical) -- a bare key binds nothing under /r3 (vNext M0C #5)
        ctx.render_to(cfg('ros2_controllers.yaml'), 'moveit_config/ros2_controllers.yaml.j2',
                      arm=robot.arm_controller, joints=group.joints,
                      node_key='/**/' if cell_namespace(project) else '')
        ctx.render_to(cfg('moveit_controllers.yaml'), 'moveit_config/moveit_controllers.yaml.j2',
                      arm=robot.arm_controller, joints=group.joints,
                      gripper_present=gripper_present, grip=robot.gripper)
        ctx.render_to(cfg('initial_positions.yaml'), 'moveit_config/initial_positions.yaml.j2',
                      initial_positions=robot.initial_positions)
        # MoveIt Servo (ADR-0015): only when a move chose `execution: servo`. The
        # from-scratch flow keeps the servo_node in the app bring-up (which is where this
        # flow assembles move_group), so the app emitter gates the launch entry on the
        # same fact — an all-`plan` project emits neither file nor node.
        if project.application.uses_servo():
            ctx.render_to(cfg('servo.yaml'), 'moveit_config/servo.yaml.j2',
                          **build_servo_context(project))

        # --- xacros (TEMPLATE) ---
        ros2_control_name = f'{robot.robot_name.upper()}System'
        ctx.render_to(cfg(f'{robot.robot_name}.urdf.xacro'),
                      'moveit_config/robot.urdf.xacro.j2',
                      robot_name=robot.robot_name,
                      description_package=project.bundle.description_package,
                      moveit_config_package=pkg,
                      top_xacro=robot.description.top_xacro,
                      ros2_control_name=ros2_control_name,
                      gazebo_supported=gazebo_supported)
        ctx.render_to(cfg(f'{robot.robot_name}.ros2_control.xacro'),
                      'moveit_config/robot.ros2_control.xacro.j2',
                      robot_name=robot.robot_name, joints=group.joints,
                      moveit_config_package=pkg,
                      gazebo_supported=gazebo_supported,
                      gripper_command_joint=(robot.gripper.command_joint
                                             if gripper_present else None))
        # RViz runs at the ROOT even for a namespaced cell (the MotionPlanning plugin
        # builds no interactive marker from inside a namespace), so what it needs is the
        # move_group namespace + the namespaced scene/trajectory topics in the config.
        # Empty namespace => byte-identical to the pre-ADR-0016 file.
        ctx.render_to(cfg('moveit.rviz'), 'moveit_config/moveit.rviz.j2',
                      base_frame=robot.base_frame,
                      cell_namespace=cell_namespace(project))

        # --- package files (TEMPLATE) ---
        # This package launches nothing: TMR's launch helper READS its files, so its TMR floor
        # is the first version TMR's package.xml stated (vNext M2.4).
        ctx.render_to(f'{pkg}/package.xml', 'moveit_config/package.xml.j2',
                      package_name=pkg, meta=project.meta, robot_name=robot.robot_name,
                      servo_dep=project.application.uses_servo(),
                      description_package=project.bundle.description_package,
                      tmr=requirement(use_of()))
        ctx.render_to(f'{pkg}/CMakeLists.txt', 'moveit_config/CMakeLists.txt.j2',
                      package_name=pkg)

    def _compute_disable_collisions(
        self, project, ctx: GenContext
    ) -> Optional[Sequence[DisablePair]]:
        """Resolve the self-collision matrix from PROJECT DATA (no subprocess).

        Generation never spawns ``collisions_updater`` (it needs a live ROS context).
        The matrix is computed once by the live session / ``trainit_compute_collisions``
        and stored in ``robot.disable_collisions``; here we just consume it.
        """
        robot = project.robot
        if robot.disable_collisions:
            return [
                DisablePair(d.link1, d.link2, d.reason)
                for d in robot.disable_collisions
            ]
        if robot.source_srdf:
            return None  # user supplies their own SRDF elsewhere
        ctx.manifest.warn(
            'no disable_collisions in project: SRDF will be over-conservative. '
            'Populate robot.disable_collisions (e.g. via trainit_compute_collisions).'
        )
        return None
