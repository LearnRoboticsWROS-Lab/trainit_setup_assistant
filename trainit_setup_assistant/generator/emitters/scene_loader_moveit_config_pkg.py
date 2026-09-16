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

from ...model.enums import Backend, GripperKind, SimGraspAdapter
from ..camera_variant import rewrite_camera_include
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

        # Safety net: a mesh object with no mesh_resource is emitted with an empty
        # mesh_path and the scene loader cannot render it — a silently invisible scene.
        empty = [o.id for o in project.scene.objects
                 if o.is_mesh() and not (o.mesh_resource or '').strip()]
        if empty:
            ctx.manifest.warn(
                f'{len(empty)} mesh scene object(s) have an EMPTY mesh_resource '
                f'({", ".join(empty[:5])}{"…" if len(empty) > 5 else ""}): the scene '
                f'loader cannot render them. Fix the mesh mapping (Step 2).')

        # 1) COPY the base config/ verbatim (preserves the tuned mode-switch + SRDF).
        ctx.copy_tree(base_config, f'{pkg}/config')

        # 1a) CAMERA VARIANT (TSA v4, D-015): swap the URDF include named by the
        # project's CameraSpec so TF carries the camera frames DetectObject
        # transforms from. Copy-then-overwrite, like the SRDF merge below.
        camera = project.perception.camera if project.perception else None
        if camera is not None:
            for src in sorted(base_config.glob('*')):
                if src.suffix not in ('.xacro', '.urdf') or not src.is_file():
                    continue
                rewritten = rewrite_camera_include(src.read_text(), camera)
                if rewritten is not None:
                    ctx.generate_to(f'{pkg}/config/{src.name}', rewritten,
                                    source='camera_variant_include')

        # 1b) MERGE assistant-captured named states into the copied SRDF. Waypoints
        # captured live (blind mode: jog with the RViz gizmo, Plan & Execute, capture)
        # exist only in the project — the app's named-target moves resolve against
        # THIS config's SRDF at runtime, so they must land here too. The same pass
        # appends the camera mount pair (base↔camera "Never"); every camera-vs-arm
        # pair stays CHECKED so planning avoids the camera.
        self._rewrite_srdf(project, ctx, base_config, pkg)

        # 1c) MoveItConfigsBuilder(<robot_name>) INFERS the URDF/SRDF from the robot name
        # (config/<robot_name>.urdf.xacro, config/<robot_name>.srdf). A third-party base
        # config may name them differently (e.g. a UR cell ships ur.urdf.xacro / ur.srdf),
        # which the builder cannot find. Ensure a <robot_name>.* copy exists so any base
        # config is consumable. No-op when the base already follows the convention (the
        # FR3WML golden), so the bundle stays byte-stable there.
        self._ensure_moveit_infer_names(project, ctx, base_config, pkg)

        # 2) GENERATE scene.yaml from the SceneSpec.
        ctx.generate_to(f'{pkg}/config/scene.yaml', build_scene_yaml(project),
                        source='scene_from_project')

        # 3) TEMPLATE the self-referential bringup (mode-switch + planners + scene loader).
        gripper_present = (robot.gripper.kind is not GripperKind.NONE
                           and bool(robot.gripper.controller_name))
        gripper_action = robot.gripper.gripper_action_ns() if gripper_present else ''
        # Synthetic coloured cloud (TSA v4): in sim it must be BUILT the way the real
        # driver publishes it (depth_image_proc, isaac branch only); in real the
        # driver already provides points_topic.
        camera_cloud = None
        # dedupe (D-016 backlog): a hand-adapted base bring-up may already carry the
        # cloud node — captured into deployment.bridges at Step 1. Emitting the
        # template block too would run the same node twice on the same topic.
        cloud_bridge = any(b.package == 'depth_image_proc' for b in dep.bridges)
        if camera is not None and camera.synthetic_cloud_in_sim and not cloud_bridge:
            camera_cloud = {'rgb': camera.rgb_topic,
                            'camera_info': camera.camera_info_topic,
                            'depth': camera.depth_topic,
                            'points': camera.points_topic}

        # W3 (ADR-0010/0011): wire the Gazebo LinkAttacher grasp adapter. When the gripper's
        # gazebo sim adapter is link_attacher and a grasp target exists, TSA GENERATES the
        # bridge (the base moveit_config does not carry it — ROS2ML stays untouched): the kit
        # script is copied into the package and run in the gazebo bring-up, welding the object
        # to the attach link (/ATTACHLINK|/DETACHLINK) on the grasp Bool. object_model = the
        # grasp target's id (its Gazebo <model> name); needs IFRA_LinkAttacher in the workspace.
        gazebo_link_attacher = None
        grasp_ids = project.scene.grasp_target_ids()
        if (Backend.GAZEBO in dep.modes and grasp_ids
                and robot.gripper.grasp_adapter_for(Backend.GAZEBO) is SimGraspAdapter.LINK_ATTACHER):
            object_link = 'link'
            if project.scene.world_path:            # auto-detect the object's <link name>
                from ...importers.world_importer import model_first_link
                object_link = model_first_link(project.scene.world_path, grasp_ids[0])
            gazebo_link_attacher = {
                'gripper_cmd_topic': project.scene.gripper_cmd_topic or '/gripper_cmd',
                'robot_model': robot.robot_name,
                'ee_link': project.scene.attach_link,
                'object_model': grasp_ids[0],
                'object_link': object_link,
            }
            from pathlib import Path as _P
            _kit = (_P(__file__).parents[2] / 'resources' / 'gazebo_cell_kit' /
                    'scripts' / 'link_attacher_bridge.py')
            ctx.copy_file(str(_kit), f'{pkg}/scripts/link_attacher_bridge.py')
        ctx.render_to(f'{pkg}/launch/bringup.launch.py',
                      'moveit_config/scene_loader_bringup.launch.py.j2',
                      robot_name=robot.robot_name,
                      world_default=(project.scene.world_path or ''),
                      gazebo_link_attacher=(gazebo_link_attacher is not None),
                      gazebo_link_attacher_py=repr(gazebo_link_attacher),
                      moveit_config_package=pkg,
                      arm_controller=robot.arm_controller.name,
                      valid_modes_py=repr(tuple(m.value for m in dep.modes)),
                      arm_js_remap_to_py=repr(dep.arm_joint_states_remap_to),
                      bridges_py=repr([b.model_dump() for b in dep.bridges]),
                      real_include_py=(repr(dep.real_include.model_dump())
                                       if dep.real_include else 'None'),
                      default_mode=dep.default_mode.value,
                      modes_human=' | '.join(m.value for m in dep.modes),
                      gripper_mock_action_py=(repr(gripper_action)
                                              if gripper_present else 'None'),
                      camera_cloud=camera_cloud,
                      # Gazebo backend (ADR-0008 F2): gated so mock/isaac/real stay
                      # byte-identical. A parallel gripper is a ros2_control controller
                      # spawned against the in-gzserver plugin CM; suction grasps via a
                      # LinkAttacher bridge (a normal DeploymentSpec.bridges entry).
                      gazebo_supported=(Backend.GAZEBO in dep.modes),
                      gazebo_cm_bootstrap=dep.gazebo_cm_bootstrap,
                      # Spawn the cell's gripper controller in Gazebo whenever the base config
                      # declares one (a GripperCommand controller captured into controller_name),
                      # not only for a PARALLEL kind — the base ingest defaults an unknown gripper
                      # to SUCTION, which wrongly dropped the Robotiq's gripper_position_controller
                      # so `ros2 control list_controllers` showed no gripper and it never moved.
                      # A suction cell with no ros2_control joint controller has an empty
                      # controller_name, so it is still not spawned (it uses the LinkAttacher).
                      gazebo_gripper_controller=(
                          robot.gripper.controller_name
                          if gripper_present and robot.gripper.controller_name
                          else None))

        # 4) package.xml + CMakeLists (exec_depend the framework + connector packages).
        ctx.render_to(f'{pkg}/package.xml', 'moveit_config/scene_loader_package.xml.j2',
                      package_name=pkg, meta=project.meta, robot_name=robot.robot_name,
                      bridge_packages=dep.bridge_packages(),
                      camera_cloud_dep=bool(camera_cloud))
        ctx.render_to(f'{pkg}/CMakeLists.txt', 'moveit_config/scene_loader_CMakeLists.txt.j2',
                      package_name=pkg, has_link_attacher=bool(gazebo_link_attacher))

    @staticmethod
    def _rewrite_srdf(project, ctx: GenContext, base_config: Path, pkg: str) -> None:
        """One rewrite pass over the copied SRDF: append captured named states missing
        from it, and (with a camera) the mount↔camera disable_collisions pair. A single
        pass because two generate_to calls on the same rel_path would lose the first."""
        import xml.etree.ElementTree as ET
        srdfs = sorted(base_config.glob('*.srdf'))
        if not srdfs:
            return
        camera = project.perception.camera if project.perception else None
        if not project.robot.named_states and camera is None:
            return
        text = srdfs[0].read_text()
        try:
            root = ET.fromstring(text)
        except ET.ParseError:
            ctx.manifest.warn(f'{srdfs[0].name}: parse failed — captured named states '
                              f'NOT merged into the copied SRDF.')
            return
        if '</robot>' not in text:
            return
        additions = []

        existing = {(gs.get('name'), gs.get('group')) for gs in root.findall('group_state')}
        missing = [s for s in project.robot.named_states
                   if (s.name, s.group) not in existing]
        if missing:
            blocks = []
            for s in missing:
                joints = '\n'.join(f'        <joint name="{j}" value="{v:.6g}"/>'
                                   for j, v in s.joint_values.items())
                blocks.append(f'    <group_state name="{s.name}" group="{s.group}">\n'
                              f'{joints}\n    </group_state>')
            additions.append(
                '    <!-- named states captured in the TrainIt Setup Assistant -->\n'
                + '\n'.join(blocks))

        if camera is not None:
            base_link = project.robot.base_frame
            pairs = {frozenset((d.get('link1'), d.get('link2')))
                     for d in root.findall('disable_collisions')}
            if frozenset((base_link, camera.link)) not in pairs:
                additions.append(
                    '    <!-- camera mount (generated): only the mount pair is '
                    'disabled; every camera-vs-arm pair stays CHECKED so planning '
                    'avoids the camera -->\n'
                    f'    <disable_collisions link1="{base_link}" '
                    f'link2="{camera.link}" reason="Never"/>')

        if not additions:
            return
        merged = text.replace('</robot>', '\n'.join(additions) + '\n</robot>')
        # Write under <robot_name>.srdf so MoveItConfigsBuilder infers it (a no-op for the
        # golden, whose base srdf is already <robot_name>.srdf); _ensure_moveit_infer_names
        # covers the no-merge case.
        ctx.generate_to(f'{pkg}/config/{project.robot.robot_name}.srdf', merged,
                        source='srdf_named_state_merge')

    @staticmethod
    def _ensure_moveit_infer_names(project, ctx: GenContext, base_config: Path, pkg: str) -> None:
        """Guarantee config/<robot_name>.urdf.xacro and config/<robot_name>.srdf exist so
        MoveItConfigsBuilder(<robot_name>) can infer them from ANY base config (not only one
        that already names its files after the robot). A no-op when they are already present
        (e.g. the FR3WML golden), so byte-stability is preserved there."""
        name = project.robot.robot_name
        out_cfg = ctx.output_root / pkg / 'config'

        if not (out_cfg / f'{name}.urdf.xacro').exists():
            xacros = sorted(base_config.glob('*.urdf.xacro'))
            if len(xacros) == 1:
                ctx.copy_file(xacros[0], f'{pkg}/config/{name}.urdf.xacro')
            elif len(xacros) > 1:
                ctx.manifest.warn(
                    f'base config has {len(xacros)} *.urdf.xacro files '
                    f'({", ".join(x.name for x in xacros)}); cannot pick the main URDF to '
                    f'name {name}.urdf.xacro — set the base config URDF to {name}.urdf.xacro '
                    f'or MoveItConfigsBuilder will not find it.')
            else:
                ctx.manifest.warn(
                    f'base config has no config/*.urdf.xacro; MoveItConfigsBuilder({name}) '
                    f'cannot infer the URDF. Provide config/{name}.urdf.xacro.')

        if not (out_cfg / f'{name}.srdf').exists():
            srdfs = sorted(base_config.glob('*.srdf'))
            if srdfs:
                ctx.copy_file(srdfs[0], f'{pkg}/config/{name}.srdf')
            else:
                ctx.manifest.warn(
                    f'base config has no config/*.srdf; MoveItConfigsBuilder({name}) cannot '
                    f'infer the semantics. Provide config/{name}.srdf.')
