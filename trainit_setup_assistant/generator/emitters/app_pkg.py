"""Emit ``*_app``: GENERATE bt_params + the BT tree, TEMPLATE launch + package files.

bt_params.yaml and the BT XML are GENERATED from the task graph (waypoints + scene +
tool actions); the launch files are TEMPLATED from the robot/deployment; the Groot2
project is generated metadata.
"""

from __future__ import annotations

from ...applications import build_bt_params_context, get_application
from ...model.enums import AppType, Backend, GripperKind
from ..perception_yaml import build_perception_yaml
from .base import Emitter, GenContext


class AppEmitter(Emitter):
    def emit(self, project, ctx: GenContext) -> None:
        app = project.application
        pkg = project.bundle.app_package
        robot = project.robot
        dep = project.deployment

        gripper_present = (robot.gripper.kind is not GripperKind.NONE
                           and bool(robot.gripper.controller_name))
        gripper_action = robot.gripper.gripper_action_ns() if gripper_present else ''

        template = get_application(app.type)

        # validation problems are surfaced (do not abort: a partial bundle still helps)
        for problem in template.validate(project):
            ctx.manifest.warn(f'application: {problem}')

        # In-tree AddCollisionObject is BOX-only, so a non-box object added in the tree
        # is emitted as its AABB. This ONLY applies to the from-scratch path; with a
        # scene loader (base set) the obstacles come from scene.yaml as real meshes.
        if not robot.base_moveit_config_path:
            for obj in project.scene.objects:
                if obj.is_planning_collision() and not obj.is_box():
                    ctx.manifest.warn(
                        f'scene object "{obj.id}" is {obj.shape.value}; engine is box-only '
                        f'-> emitted as its AABB {obj.aabb_dims()}')

        # --- bt_params.yaml (GENERATE from the task graph) ---
        bt_ctx = build_bt_params_context(project)
        bt_params_text = ctx.render('app/bt_params.yaml.j2', **bt_ctx)
        ctx.generate_to(f'{pkg}/config/bt_params.yaml', bt_params_text,
                        source='app/bt_params.yaml.j2')

        # --- BT tree XML (GENERATE by the application template) ---
        tree_name = template.tree_filename(project)
        tree_xml = template.build_tree_xml(project)
        ctx.generate_to(f'{pkg}/bt_trees/{tree_name}', tree_xml,
                        source=f'application:{app.type.value}')

        # --- learned-policy files + runtime launch (ADR-0005) ---
        # Only when the task uses a policy step; otherwise nothing is emitted and a
        # deterministic bundle stays byte-identical.
        if app.policies:
            self._emit_policies(project, ctx, pkg)

        # --- perception.yaml + the detector list (TSA v4, D-015) ---
        # Emitted whenever the project HAS detectors, whatever the app type: the
        # Perception step configures a resource; an app that never binds it simply
        # runs an idle detector. Pure-python methods get a detector_node in the
        # launch; external-node methods (C++/PCL, learned) publish the contract
        # themselves and only appear in perception.yaml for the record.
        per = project.perception
        detector_nodes = [d.name for d in per.detectors if d.emits_node] if per else []
        if per and per.detectors:
            ctx.generate_to(f'{pkg}/config/perception.yaml',
                            build_perception_yaml(project),
                            source='perception_from_project')
        # D-015 noise-filter params exist only from trainit_perception 0.2.0: on an
        # older runtime they merge silently into params and are IGNORED — the deployed
        # mask would differ from what was tuned, with no error. Pin the minimum.
        _V015_KEYS = ('blur_px', 'depth_min_m', 'depth_max_m')
        needs_v02 = any(k in d.params for d in (per.detectors if per else [])
                        for k in _V015_KEYS)
        # bootstrap flow (no base config): the camera extras live in the scene-loader
        # emitter only — say so instead of silently dropping them.
        if per and per.camera and not robot.base_moveit_config_path:
            ctx.manifest.warn(
                'bootstrap flow with a camera: the synthetic coloured cloud '
                '(depth_image_proc) and the base<->camera SRDF pair are only emitted '
                'on the scene-loader path (base config set) — add them by hand or '
                'switch to a base config.')

        # --- launch files (TEMPLATE) ---
        ctx.render_to(f'{pkg}/launch/trainit_bt.launch.py',
                      'app/trainit_bt.launch.py.j2',
                      robot_name=robot.robot_name,
                      moveit_config_package=project.bundle.moveit_config_package,
                      app_package=pkg,
                      tree_filename=tree_name,
                      default_planner_mode=app.global_planner_mode.value,
                      detectors=detector_nodes)

        if robot.base_moveit_config_path:
            # Bundle flow: the trainit_config package HAS a full cell bringup (planners,
            # modes, bridges, scene loader). The app bringup INCLUDES it and runs the
            # BT on top -> one command launches the whole stack, and the two bringups
            # can never drift apart.
            ctx.render_to(f'{pkg}/launch/bringup.launch.py',
                          'app/bringup.launch.py.j2',
                          robot_name=robot.robot_name,
                          moveit_config_package=project.bundle.moveit_config_package,
                          app_package=pkg,
                          valid_modes_py=repr(tuple(m.value for m in dep.modes)),
                          default_mode=dep.default_mode.value,
                          modes_human=' | '.join(m.value for m in dep.modes),
                          has_policies=bool(app.policies),
                          gazebo_supported=(Backend.GAZEBO in dep.modes),
                          default_planner_mode=app.global_planner_mode.value)
        else:
            # From-scratch bootstrap: no base config bringup exists — the app bringup
            # assembles move_group + controllers itself (and a mock gripper server).
            ctx.render_to(f'{pkg}/launch/bringup.launch.py',
                          'app/bringup_bootstrap.launch.py.j2',
                          robot_name=robot.robot_name,
                          moveit_config_package=project.bundle.moveit_config_package,
                          app_package=pkg,
                          arm_controller=robot.arm_controller.name,
                          valid_modes_py=repr(tuple(m.value for m in dep.modes)),
                          arm_js_remap_to_py=repr(dep.arm_joint_states_remap_to),
                          bridges_py=repr([b.model_dump() for b in dep.bridges]),
                          real_include_py=(repr(dep.real_include.model_dump())
                                           if dep.real_include else 'None'),
                          default_mode=dep.default_mode.value,
                          modes_human=' | '.join(m.value for m in dep.modes),
                          has_policies=bool(app.policies),
                          gazebo_supported=(Backend.GAZEBO in dep.modes),
                          gazebo_gripper_controller=(
                              robot.gripper.controller_name
                              if gripper_present and robot.gripper.kind is GripperKind.PARALLEL
                              else None),
                          # mock gripper: run the generated no-op server in mock mode when
                          # a gripper exists but no cell bridge serves it (empty bridges).
                          gripper_mock_action_py=(repr(gripper_action)
                                                  if gripper_present else 'None'))

        # --- mock gripper action server (GENERATE, only if the robot has a gripper) ---
        if gripper_present:
            ctx.render_to(f'{pkg}/scripts/mock_gripper_action_server.py',
                          'app/mock_gripper_action_server.py.j2',
                          make_executable=True,          # installed with install(PROGRAMS)
                          gripper_action=gripper_action)

        # --- README.md (TEMPLATE): how to build/run per mode + where to tune DOF ---
        # Emitted for the MVP scene-loader flow; the from-scratch/golden path is kept
        # byte-stable (no README) until the golden is regenerated.
        if robot.base_moveit_config_path:
            ctx.render_to(f'{pkg}/README.md', 'app/README.md.j2',
                          robot_name=robot.robot_name, app_type=app.type.value,
                          app_package=pkg,
                          moveit_config_package=project.bundle.moveit_config_package,
                          description_package=project.bundle.description_package,
                          tree_filename=template.tree_filename(project),
                          default_mode=dep.default_mode.value,
                          default_planner_mode=app.global_planner_mode.value)

        # --- Groot2 project (GENERATE metadata) ---
        groot_name = f'{robot.robot_name}_{app.type.value}'
        btproj_text = ctx.render('app/btproj.j2',
                                 groot_project_name=groot_name, tree_filename=tree_name)
        ctx.generate_to(f'{pkg}/groot2/{robot.robot_name}_{app.type.value}.btproj',
                        btproj_text, source='app/btproj.j2')

        # --- package files (TEMPLATE) ---
        ctx.render_to(f'{pkg}/package.xml', 'app/package.xml.j2',
                      package_name=pkg, meta=project.meta, robot_name=robot.robot_name,
                      app_type=app.type.value,
                      moveit_config_package=project.bundle.moveit_config_package,
                      description_package=project.bundle.description_package,
                      bridge_packages=dep.bridge_packages(),
                      has_perception=bool(detector_nodes),
                      perception_min_version=('0.2.0' if needs_v02 else None))
        ctx.render_to(f'{pkg}/CMakeLists.txt', 'app/CMakeLists.txt.j2',
                      package_name=pkg, has_gripper_script=gripper_present,
                      has_policies=bool(project.application.policies))

    def _emit_policies(self, project, ctx: GenContext, pkg: str) -> None:
        """Ship each policy step's card + exported policy file into the bundle
        (policies/<name>/) and emit a launch that starts the runtime per policy.

        The card is a DATA contract (POLICY_EXECUTION.md) copied verbatim; its
        relative `files:` still resolve because the .pt/.onnx are shipped beside it.
        The generated CMakeLists installs the whole package `share`, so policies/ ships
        with it. No torch here — TSA only moves files (ADR-0003)."""
        import os

        import yaml

        app = project.application
        shipped = []
        for pol in app.policies:
            dest_dir = f'{pkg}/policies/{pol.name}'
            if not pol.card or not os.path.isfile(pol.card):
                ctx.manifest.warn(
                    f'policy "{pol.name}": card not found at "{pol.card}" — not shipped; '
                    f'add it under {dest_dir}/ by hand')
                continue
            ctx.copy_file(pol.card, f'{dest_dir}/policy_card.yaml')
            try:
                card = (yaml.safe_load(open(pol.card)) or {}).get('policy_card', {})
            except Exception:                       # pragma: no cover - defensive
                card = {}
            files = card.get('files', {}) or {}
            card_dir = os.path.dirname(os.path.abspath(pol.card))
            shipped_any = False
            for key in ('onnx', 'jit'):
                rel = files.get(key)
                if not rel:
                    continue
                src = os.path.join(card_dir, rel)
                if os.path.isfile(src):
                    ctx.copy_file(src, f'{dest_dir}/{os.path.basename(rel)}')
                    shipped_any = True
                else:
                    ctx.manifest.warn(
                        f'policy "{pol.name}": card lists "{rel}" but it is not beside the '
                        f'card — ship it into {dest_dir}/ by hand')
            if not shipped_any:
                ctx.manifest.warn(
                    f'policy "{pol.name}": no exported policy file (.pt/.onnx) shipped — '
                    f'the runtime will not load until one is placed in {dest_dir}/')
            shipped.append((pol, card))

        if not shipped:
            return
        single = len(shipped) == 1
        # Detection wiring comes from the CELL detector (ADR-0006), not the card's
        # portable placeholder: a single detector is wired straight; with several the
        # topic/class_id is left at the runtime default and a warning asks the user to set
        # it (per policy) by hand.
        per = getattr(project, 'perception', None)
        dets = list(getattr(per, 'detectors', []) or []) if per else []
        det_topic = f'/perception/{dets[0].name}/detections' if len(dets) == 1 else None
        det_class = dets[0].class_id if len(dets) == 1 else None
        if len(dets) > 1:
            ctx.manifest.warn(
                'policy runtime: the cell has several detectors — detection topic/class_id '
                'was left at the runtime default; set it per policy by hand if a specific '
                'detector must feed the policy obs.')
        policies_ctx = []
        for p, card in shipped:
            action = card.get('action', {}) or {}
            policies_ctx.append({
                'name': p.name,
                'mode': p.mode.value,
                # one policy -> the default node name (its ~/ services match the RunPolicy
                # defaults TSA emitted); several -> distinct names (set the PolicyStep
                # run_service/status_topic/target_topic to match).
                'node_name': 'trainit_policy_runtime' if single
                             else f'trainit_policy_runtime_{p.name}',
                'detection_topic': det_topic,
                'detection_class_id': det_class,
                'tcp_frame': action.get('body') or None,          # card's action.body
                'suction_state_topic': p.attached_topic or None,
                'calibration_offset': p.calibration_offset,       # None/[] => not emitted
                'calibration_frame': p.calibration_frame.value,
            })
        if not single:
            ctx.manifest.warn(
                'multiple policy steps: each runtime node has a distinct name; set each '
                'PolicyStep run_service/status_topic/target_topic to that node namespace.')
        ctx.render_to(f'{pkg}/launch/policy_runtime.launch.py',
                      'app/policy_runtime.launch.py.j2',
                      app_package=pkg, policies=policies_ctx)
