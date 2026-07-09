"""Emit the STANDALONE scene+planner moveit_config (copy base + planners + scene).

The user's flow keeps a hand-made BASE moveit_config (the adaptation-sprint deliverable
that already provides the tuned mock/isaac/real bringup + vendor bridges). This emitter
produces a SELF-CONTAINED config by:

  1. COPYING that base package's ``config/`` verbatim (SRDF, urdf.xacro, ros2_control
     xacro, kinematics/joint_limits/ompl/pilz/controllers yamls, moveit.rviz) — so the
     tuned mode-switch + SRDF waypoints are preserved EXACTLY, and the URDF's only
     external ``$(find …)`` reference is the connector geometry package, not another
     moveit_config → the result is independent of the base package.
  2. GENERATING ``config/scene.yaml`` from the SceneSpec (the USD obstacles).
  3. TEMPLATING a self-referential ``bringup.launch.py`` that adds the OMPL/Pilz/CHOMP
     planners (``build_move_group_params`` on THIS package) + the scene loader node.

Invoked TWICE with the same logic: at Step 3 for the intermediate
``<robot>_scene_loader_moveit_config`` (RViz configuration), and inside the bundle for
``<robot>_trainit_config`` (deployment) — so config == deploy.
"""

from __future__ import annotations

from pathlib import Path

from ...model.enums import GripperKind
from ..scene_yaml import build_scene_yaml
from .base import Emitter, GenContext


class SceneLoaderMoveitConfigEmitter(Emitter):
    def __init__(self, package_name: str = None):
        # None => project.bundle.moveit_config_package (the bundle's _trainit_config).
        # Pass a name to emit the Step-3 intermediate under a different package name.
        self.package_name = package_name

    def emit(self, project, ctx: GenContext) -> None:
        robot = project.robot
        dep = project.deployment
        pkg = self.package_name or project.bundle.moveit_config_package

        base_path = robot.base_moveit_config_path
        if not base_path:
            ctx.manifest.warn(
                'scene_loader emitter: robot.base_moveit_config_path is unset — cannot '
                'copy the hand-made base config. Set it (Step 2) to generate a '
                'self-contained scene+planner config.')
            return
        base_config = Path(base_path) / 'config'
        if not base_config.is_dir():
            ctx.manifest.warn(f'scene_loader emitter: base config dir not found: {base_config}')
            return

        # 1) COPY the base config/ verbatim (preserves the tuned mode-switch + SRDF).
        ctx.copy_tree(base_config, f'{pkg}/config')

        # 2) GENERATE scene.yaml from the SceneSpec.
        ctx.generate_to(f'{pkg}/config/scene.yaml', build_scene_yaml(project),
                        source='scene_from_project')

        # 3) TEMPLATE the self-referential bringup (mode-switch + planners + scene loader).
        gripper_present = (robot.gripper.kind is not GripperKind.NONE
                           and bool(robot.gripper.controller_name))
        gripper_action = robot.gripper.gripper_action_ns() if gripper_present else ''
        ctx.render_to(f'{pkg}/launch/bringup.launch.py',
                      'moveit_config/scene_loader_bringup.launch.py.j2',
                      robot_name=robot.robot_name,
                      moveit_config_package=pkg,
                      arm_controller=robot.arm_controller.name,
                      valid_modes_py=repr(tuple(dep.modes)),
                      arm_js_remap_to_py=repr(dep.arm_joint_states_remap_to),
                      bridges_py=repr([b.model_dump() for b in dep.bridges]),
                      real_include_py=(repr(dep.real_include.model_dump())
                                       if dep.real_include else 'None'),
                      default_mode=dep.default_mode,
                      modes_human=' | '.join(dep.modes),
                      gripper_mock_action_py=(repr(gripper_action)
                                              if gripper_present else 'None'))

        # 4) package.xml + CMakeLists (exec_depend the framework + connector packages).
        ctx.render_to(f'{pkg}/package.xml', 'moveit_config/scene_loader_package.xml.j2',
                      package_name=pkg, meta=project.meta, robot_name=robot.robot_name,
                      bridge_packages=dep.bridge_packages())
        ctx.render_to(f'{pkg}/CMakeLists.txt', 'moveit_config/scene_loader_CMakeLists.txt.j2',
                      package_name=pkg)
