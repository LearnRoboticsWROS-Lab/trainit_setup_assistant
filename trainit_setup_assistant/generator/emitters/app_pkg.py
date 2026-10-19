"""Emit ``*_app``: GENERATE bt_params + the BT tree, TEMPLATE launch + package files.

bt_params.yaml and the BT XML are GENERATED from the task graph (waypoints + scene +
tool actions); the launch files are TEMPLATED from the robot/deployment; the Groot2
project is generated metadata.
"""

from __future__ import annotations

from ...applications import build_bt_params_context, get_application
from ...model.enums import AppType, Backend, GripperKind
from ...model.faces import param_floors, version_key
from ..eef_contract import (HARDWARE_LAYER, bridges_with_adapters, device,
                            namespace_isaac_warning)
from ..external_node_launch import (external_node_context, external_node_packages,
                                    external_node_warnings)
from ..namespace import cell_namespace, namespace_handoff_warnings
from ..perception_yaml import build_perception_yaml, detector_params
from ..runtime_requirements import hardware_layer_requirement, requirement, use_of
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
        # The end-effector contract (ADR-0022): a canonical project has no legacy gripper action
        # and no generated mock server (the TrainIt adapter serves the device, [A]).
        eef_dev = device(project)
        legacy_gripper = gripper_present and eef_dev is None

        template = get_application(app.type)
        # vNext M3.3: a Canonical Task Program is planned ONCE, validated (the orchestrator refused
        # any problem before this emitter ran) and resolved; every file below is written from it.
        plan = None
        if app.is_task_program():
            from ...applications.task_program import plan_program
            plan = plan_program(project)
            if project.scene.payload is not None:
                # vNext M3.7 review: the grasp and release idioms do not attach a payload to the
                # planning scene in this version (KNOWN_DEBT); the text said they would (M3.6)
                ctx.manifest.warn(f'application: scene payload "{project.scene.payload.id}" is not '
                                  'used by the task program: its grasp and release steps do not attach '
                                  'an object to the planning scene in this version (the transfer is '
                                  'planned without the payload): it is not written')
        # The robot slot (ADR-0016 Decision 3). Empty => every generated node at the ROOT,
        # byte-for-byte the pre-ADR-0016 bundle. The from-scratch flow reports the
        # hand-off warnings here (the scene-loader flow reports them in its own emitter).
        cell_ns = cell_namespace(project)
        if not robot.base_moveit_config_path:
            for problem in namespace_handoff_warnings(project):
                ctx.manifest.warn(problem)

        # validation problems are surfaced (do not abort: a partial bundle still helps). A
        # problem may quote a policy card's path for the GUI: the manifest names it portably.
        cards = [pol.card for pol in (project.application.policies or []) if pol.card]
        for problem in template.validate(project):
            recorded = problem
            for card in cards:
                recorded = recorded.replace(f'"{card}"', f'"{ctx.portable(card)}"')
            ctx.manifest.warn(f'application: {recorded}', local=f'application: {problem}')

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
        if plan is None:
            bt_params_text = ctx.render('app/bt_params.yaml.j2', **bt_ctx)
            ctx.generate_to(f'{pkg}/config/bt_params.yaml', bt_params_text,
                            source='app/bt_params.yaml.j2')
        else:
            from ..program_files import (motion_profiles_yaml, paths_yaml,
                                         task_parameters_lines, tasks_yaml)
            g = dict(bt_ctx['g'], planner_mode=plan.planner_mode,
                     process_path_backend=plan.backend, process_controller=plan.process_controller)
            bt_params_text = ctx.render('app/bt_params_program.yaml.j2', g=g,
                                        task_parameters=task_parameters_lines(
                                            plan, bt_ctx['scene_params']))
            ctx.generate_to(f'{pkg}/config/bt_params.yaml', bt_params_text,
                            source='app/bt_params_program.yaml.j2')
            if plan.paths:
                ctx.generate_to(f'{pkg}/config/paths.yaml', paths_yaml(plan),
                                source='application:task_program')
            ctx.generate_to(f'{pkg}/config/motion_profiles.yaml', motion_profiles_yaml(plan),
                            source='application:task_program')
            # vNext M3.7: the task manifest (no runtime reads it: MainTree runs main_task)
            ctx.generate_to(f'{pkg}/config/tasks.yaml', tasks_yaml(project, plan),
                            source='application:task_program')

        # --- BT tree XML (GENERATE by the application template) ---
        tree_name = template.tree_filename(project)
        if plan is None:
            tree_xml = template.build_tree_xml(project)
        else:
            from ...applications.task_program import program_tree_xml
            tree_xml = program_tree_xml(project, plan)
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
        # The second kind of method (D-015): a node of the project's own, launched beside
        # the pure-python ones. Empty for every project without an external_node block, so
        # a deterministic bundle stays byte-identical. The orchestrator has already
        # REFUSED an incomplete block, so everything here is launchable.
        external_detectors = external_node_context(project)
        external_packages = external_node_packages(project)
        for problem in external_node_warnings(project):
            ctx.manifest.warn(f'perception: {problem}')
        if per and per.detectors:
            ctx.generate_to(f'{pkg}/config/perception.yaml',
                            build_perception_yaml(project),
                            source='perception_from_project')
        # A perception parameter the deployed runtime does not know MERGES SILENTLY into
        # params and is IGNORED: the cell would run with a mask, or an order, that is not
        # the one that was configured, and nothing would say so. So every generation of
        # parameters pins the trainit_perception version that reads it, and the floor is
        # the HIGHEST pin the project's parameters demand.
        #   0.2.0  D-015 noise filters (blur / depth band)
        #   0.3.0  the candidate ranking (ADR-0017): candidate_rule + the terms' numbers,
        #          and the grasp FACE (ADR-0018) — `grasp_face` is a key 0.3.0 already
        #          READS, and the face parameters come from its own registry, so the
        #          list is complete by construction instead of by maintenance.
        #          Without this pin a bundle generated WITH a rule runs on a TP that
        #          ignores it — publishing the model's own order under the name of a
        #          ranking, which is the one failure ADR-0017 Decision 3 exists to stop.
        #          A face family NEWER than the deployed TP is a different failure and a
        #          safe one: `make_face` raises on a name it does not know, so the node
        #          stops at start-up and says which faces it has. The pin exists for the
        #          SILENT case — a parameter that merges into params and is ignored.
        #          Every floor is trainit_perception's own date for the parameter, read from
        #          TSA's snapshot of its vocabulary (`param_floors`, ADR-0021 amendment M2.2):
        #          a floor kept here as a literal has to be remembered on the other side of a
        #          repository boundary -- and it already was not: `tool_max_tilt_deg` arrived
        #          in 0.4.0 and would have been written under a 0.3.0 floor, which is the
        #          silent case this exists for. The noise filters' 0.2.0 is TP's date too.
        detector_param_keys = set()
        for d in (per.detectors if per else []):
            detector_param_keys |= set(detector_params(project, d))
        floors = param_floors()
        candidates = [floors[k] for k in detector_param_keys if k in floors]
        perception_min_version = (max(candidates, key=version_key) if candidates else None)
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
                      default_planner_mode=(plan.planner_mode if plan else
                                            app.global_planner_mode.value),
                      gazebo_supported=(Backend.GAZEBO in dep.modes),
                      # vNext M3.3: a task program loads its own profiles (and paths)
                      program=plan is not None,
                      program_paths=bool(plan and plan.paths),
                      # trainit_run_bt is a CLIENT of this robot's move_group, so it runs
                      # in the robot slot; the detector nodes stay GLOBAL (ADR-0016).
                      cell_namespace=cell_ns,
                      detectors=detector_nodes,
                      external_detectors=external_detectors)

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
                          world_default=(project.scene.world_path or ''),
                          # Gazebo Classic + gazebo_ros2_control need longer than 8s to spawn
                          # controllers + move_group before the BT ticks (else the pick move or
                          # the GripperCommand fails and, coupled, the grasp Bool never fires —
                          # ADR-0012). Gated so a non-gazebo cell (the golden) keeps 8.0 exactly.
                          bt_delay_default=('15.0' if (Backend.GAZEBO in dep.modes) else '8.0'),
                          default_planner_mode=(plan.planner_mode if plan else
                                                app.global_planner_mode.value),
                          start_tolerance=plan.start_tolerance if plan else None)
        else:
            # From-scratch bootstrap: no base config bringup exists — the app bringup
            # assembles move_group + controllers itself (and a mock gripper server).
            if eef_dev is not None:
                warning = namespace_isaac_warning(project, eef_dev)
                if warning:
                    ctx.manifest.warn(warning)
            ctx.render_to(f'{pkg}/launch/bringup.launch.py',
                          'app/bringup_bootstrap.launch.py.j2',
                          robot_name=robot.robot_name,
                          # this flow's bring-up IS the cell bring-up, so the robot slot
                          # lands here (ADR-0016); empty => byte-identical.
                          cell_namespace=cell_ns,
                          # MoveIt Servo node (ADR-0015): this flow's bring-up IS the
                          # cell bring-up, so the servo_node lives here, reading the
                          # moveit_config package's config/servo.yaml. Gated on the same
                          # fact that emits that file, so an all-`plan` project is
                          # byte-identical.
                          servo=app.uses_servo(),
                          moveit_config_package=project.bundle.moveit_config_package,
                          app_package=pkg,
                          arm_controller=robot.arm_controller.name,
                          valid_modes_py=repr(tuple(m.value for m in dep.modes)),
                          arm_js_remap_to_py=repr(dep.arm_joint_states_remap_to),
                          bridges_py=repr(bridges_with_adapters(project)),
                          real_include_py=(repr(dep.real_include.dict())
                                           if dep.real_include else 'None'),
                          default_mode=dep.default_mode.value,
                          modes_human=' | '.join(m.value for m in dep.modes),
                          has_policies=bool(app.policies),
                          gazebo_supported=(Backend.GAZEBO in dep.modes),
                          world_default=(project.scene.world_path or ''),
                          gazebo_cm_bootstrap=dep.gazebo_cm_bootstrap,
                          start_tolerance=plan.start_tolerance if plan else None,
                          gazebo_gripper_controller=(
                              robot.gripper.controller_name
                              if gripper_present and robot.gripper.kind is GripperKind.PARALLEL
                              else None),
                          # mock gripper: run the generated no-op server in mock mode when
                          # a gripper exists but no cell bridge serves it (empty bridges).
                          gripper_mock_action_py=(repr(gripper_action)
                                                  if legacy_gripper else 'None'))

        # --- mock gripper action server (GENERATE, only if the robot has a gripper) ---
        if legacy_gripper:
            ctx.render_to(f'{pkg}/scripts/mock_gripper_action_server.py',
                          'app/mock_gripper_action_server.py.j2',
                          make_executable=True,          # installed with install(PROGRAMS)
                          gripper_action=gripper_action)

        # --- README.md (TEMPLATE): how to build/run per mode + where to tune DOF ---
        # Emitted for the MVP scene-loader flow; the from-scratch/golden path is kept
        # byte-stable (no README) until the golden is regenerated.
        if plan is not None:
            # vNext M3.3: a task program's README is the program, readable (both flows)
            from ..program_files import program_readme
            ctx.generate_to(f'{pkg}/README.md', program_readme(project, plan),
                            source='application:task_program')
        elif robot.base_moveit_config_path:
            ctx.render_to(f'{pkg}/README.md', 'app/README.md.j2',
                          robot_name=robot.robot_name, app_type=app.type.value,
                          # the streamed-moves section + the servo.yaml row (ADR-0015):
                          # gated, so an all-`plan` bundle's README is byte-identical
                          servo=app.uses_servo(),
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

        # --- what this package needs to RUN (vNext M2.4, ADR-0021 amendment M2.4): the oldest
        # TMR that has every capability its tree, bt_params and launch files use, and the
        # learned-policy runtime iff a tree runs RunPolicy or a launch file starts its node. Read
        # from the files this generation wrote, never from an installed component or the
        # capability snapshot.
        use = use_of(trees=ctx.written(f'{pkg}/bt_trees/', '.xml'),
                     bt_params=ctx.written(f'{pkg}/config/', 'bt_params.yaml'),
                     launches=ctx.written(f'{pkg}/launch/', '.py'),
                     profiles=ctx.written(f'{pkg}/config/', 'motion_profiles.yaml'),
                     paths=ctx.written(f'{pkg}/config/', 'paths.yaml'))
        for unreadable in use.unreadable:
            ctx.manifest.warn(f'runtime requirement: {unreadable} does not parse -- the bundle '
                              f'will not run as generated; trainit_motion_runtime is required at '
                              f'the newest version TSA dates')
        tmr = requirement(use, needs=('trainit_run_bt',))
        # ADR-0022 [D]: the hardware layer, versioned, iff THIS package's launch files start one
        # of its adapters (the from-scratch bring-up does; the scene-loader flow's config does).
        hardware = (hardware_layer_requirement(ctx.written(f'{pkg}/launch/', '.py'))
                    if eef_dev is not None else None)

        # --- package files (TEMPLATE) ---
        ctx.render_to(f'{pkg}/package.xml', 'app/package.xml.j2',
                      package_name=pkg, meta=project.meta, robot_name=robot.robot_name,
                      app_type=app.type.value,
                      moveit_config_package=project.bundle.moveit_config_package,
                      description_package=project.bundle.description_package,
                      bridge_packages=[p for p in dep.bridge_packages()
                                       if not (hardware and p == HARDWARE_LAYER)],
                      has_perception=bool(detector_nodes),
                      # DOCUMENTED, never declared (ADR-0003): an external detector's
                      # package is usually customer IP that is not in this workspace and
                      # has no rosdep key, so an <exec_depend> on it would make
                      # `rosdep install --from-paths src` fail for everyone. Empty list
                      # => the manifest is byte-identical.
                      external_packages=external_packages,
                      # the from-scratch bring-up (above) runs the servo_node from this
                      # package's launch, so it must declare moveit_servo (ADR-0015)
                      servo_dep=(app.uses_servo()
                                 and not robot.base_moveit_config_path),
                      perception_min_version=perception_min_version,
                      tmr=tmr, policy_runtime=use.policy_runtime, hardware_layer=hardware)
        ctx.render_to(f'{pkg}/CMakeLists.txt', 'app/CMakeLists.txt.j2',
                      package_name=pkg, has_gripper_script=legacy_gripper,
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
                ctx.warn_path(lambda shown: f'policy "{pol.name}": card not found at '
                                            f'"{shown}" — not shipped; add it under '
                                            f'{dest_dir}/ by hand', pol.card)
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
                    # the card's own entry may be absolute or climb out of its directory: the
                    # manifest names the file it points at portably, the user sees the entry
                    ctx.manifest.warn(
                        f'policy "{pol.name}": card lists "{ctx.portable(src)}" but it is not '
                        f'beside the card — ship it into {dest_dir}/ by hand',
                        local=f'policy "{pol.name}": card lists "{rel}" but it is not beside the '
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
