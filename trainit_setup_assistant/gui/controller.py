"""AssistantController — the ROS-free brain behind the wizard.

Holds the in-progress :class:`CanonicalProject` and exposes the operations the wizard
(or a CLI, or a future web GUI) performs: load a robot, confirm group/frames, capture
named states, edit the scene/application, and generate the bundle. No Qt, no ROS here
(xacro compilation is delegated to robotmodel and only needs a ROS env for ``$(find)``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from ..applications import get_application
from ..generator import GenerationManifest, Orchestrator
from ..model.scene import SceneSpec
from ..model import (
    BundleSpec,
    CameraSpec,
    CanonicalProject,
    DetectorSpec,
    MotionSegment,
    NamedState,
    Payload,
    PerceptionSpec,
    SceneObject,
    ToolAction,
    VisionBinding,
    Waypoint,
    load_project,
    save_project,
)
from ..model.enums import DetectionMethod
from ..model.enums import (
    AppType,
    GripperKind,
    IsaacGraspMethod,
    MotionType,
    PlannerId,
    ReleasePolicy,
    SceneObjectCategory,
    SceneObjectSource,
    ShapeType,
    ToolActionKind,
    WaypointRole,
    WaypointType,
)
from ..robotmodel import bootstrap_project, load_robot_spec


class AssistantController:
    def __init__(self, project: Optional[CanonicalProject] = None):
        self.project: Optional[CanonicalProject] = project
        # the scene+planner config package name generated at Step 3 (used by Step 4's
        # bring-up procedure so it names the package the user actually created).
        self.scene_loader_pkg: Optional[str] = None
        self.workspace_root: Optional[str] = None   # colcon build root (parent of src/)
        # cell prims + poses read from the USD (held between import and apply).
        self._usd_prims: List[dict] = []

    # ---- project lifecycle ----
    def new_from_robot(self, xacro_path, robot_name=None, group_name=None,
                       meshes_dir=None, project_name=None) -> CanonicalProject:
        """S0/S1: load + auto-detect a robot xacro into a fresh bootstrap project."""
        spec = load_robot_spec(xacro_path, robot_name=robot_name,
                               group_name=group_name, meshes_dir=meshes_dir)
        self.project = bootstrap_project(spec, project_name=project_name)
        return self.project

    def open_project(self, path) -> CanonicalProject:
        self.project = load_project(path)
        return self.project

    def new_blank_project(self, project_name: str = 'robot_app') -> CanonicalProject:
        """Bootstrap a minimal project (MVP entry): the robot is then configured by
        ``load_base_moveit_config`` and the scene/application by later steps."""
        from ..model import (
            ApplicationSpec, DeploymentSpec, DescriptionSource, PlanningGroupSpec,
            ProjectMeta, RobotSpec, SceneSpec)
        robot = RobotSpec(
            robot_name='robot',
            description=DescriptionSource(urdf_dir='.', top_xacro='robot.urdf.xacro'),
            planning_group=PlanningGroupSpec(name='manipulator'))
        self.project = CanonicalProject(
            project_name=project_name, bundle=BundleSpec.from_prefix(project_name),
            meta=ProjectMeta(), robot=robot, scene=SceneSpec(),
            application=ApplicationSpec(), deployment=DeploymentSpec())
        return self.project

    def save(self, path) -> Path:
        return save_project(self._require(), path)

    # ---- robot / group / frames (S1) ----
    def robot_summary(self) -> dict:
        p = self._require()
        r = p.robot
        return {
            'robot_name': r.robot_name,
            'base_frame': r.base_frame,
            'tip_link': r.tip_link,
            'group': r.planning_group.name,
            'arm_joints': list(r.planning_group.joints),
            'gripper_kind': r.gripper.kind.value,
            'gripper_joint': r.gripper.command_joint,
            'named_states': [s.name for s in r.named_states],
        }

    def set_frames(self, base_frame: str, tip_link: str) -> None:
        r = self._require().robot
        r.base_frame = base_frame
        r.tip_link = tip_link
        r.planning_group.base_link = base_frame
        r.planning_group.tip_link = tip_link

    def set_group(self, name: str, joints: List[str]) -> None:
        r = self._require().robot
        r.planning_group.name = name
        r.planning_group.joints = list(joints)

    def set_arm_controller(self, *, name=None, ctrl_type=None, action_ns=None,
                           update_rate=None, command_interfaces=None,
                           state_interfaces=None) -> None:
        """Edit the arm controller (drives ros2_controllers.yaml + moveit_controllers.yaml
        + the launch ARM_CONTROLLER constant)."""
        arm = self._require().robot.arm_controller
        if name:
            arm.name = name
        if ctrl_type:
            arm.type = ctrl_type
        if action_ns:
            arm.action_ns = action_ns
        if update_rate is not None:
            arm.update_rate = int(update_rate)
        if command_interfaces is not None:
            arm.command_interfaces = list(command_interfaces)
        if state_interfaces is not None:
            arm.state_interfaces = list(state_interfaces)

    def set_gripper_controller(self, *, controller_name=None, controller_type=None,
                               action_ns=None, command_joint=None,
                               grasp_action=None, release_action=None) -> None:
        """Edit the gripper controller (drives moveit_controllers.yaml + bt gripper_action)."""
        grip = self._require().robot.gripper
        if controller_name is not None:
            grip.controller_name = controller_name or None
        if controller_type:
            grip.controller_type = controller_type
        if action_ns:
            grip.action_ns = action_ns
        if command_joint is not None:
            grip.command_joint = command_joint or None
        if grasp_action:
            grip.grasp_action = grasp_action
        if release_action:
            grip.release_action = release_action

    def set_project_name(self, name: str, derive_bundle: bool = True) -> None:
        p = self._require()
        p.project_name = name
        if derive_bundle:
            p.bundle = BundleSpec.from_prefix(name)

    # ---- Step 2: load the hand-made BASE moveit_config ----
    def load_base_moveit_config(self, package: str, path) -> dict:
        """Point at the hand-made base config AND auto-configure the project from it.

        Sets ``robot.base_moveit_config_*`` (which switches generation to the standalone
        copy-base flow) and extracts the group/frames/named-states/gripper/controllers +
        the self-collision matrix from the base's SRDF + moveit_controllers.yaml, so the
        user doesn't re-type what the adaptation-sprint deliverable already declares.
        """
        from ..robotmodel.srdf_reader import read_srdf
        p = self._require()
        base = Path(path)
        config = base / 'config'
        # Guard: a GENERATED scene_loader config has config/scene.yaml + a scene_manager_node
        # bringup. Using it as the base silently drops the real cell bridges (gripper bridge,
        # joint-state merger) -> the generated package's /softgripper_controller never comes up
        # ("Action client not connected to action server"). Refuse it with a clear message.
        if (config / 'scene.yaml').exists():
            raise ValueError(
                f"'{base.name}' looks like a GENERATED scene_loader config (config/scene.yaml "
                f"present), not a hand-made base moveit_config. Point 'Base package path' at "
                f"your base (e.g. fr30_eef_moveit_config), not a *_scene_loader_moveit_config.")
        srdfs = sorted(config.glob('*.srdf'))
        if not srdfs:
            raise FileNotFoundError(f'no .srdf found in {config}')
        info = read_srdf(srdfs[0].read_text())

        p.robot.robot_name = info.robot_name
        p.robot.base_moveit_config_package = package
        p.robot.base_moveit_config_path = str(base)

        # Description source for the bundle's <robot>_description: the base config
        # keeps the robot model in config/ (MoveIt-Setup-Assistant style). Without
        # this the DescriptionSource default ('.') would ingest the whole CWD.
        model_xacros = sorted(config.glob('*.urdf.xacro')) or sorted(config.glob('*.xacro'))
        if model_xacros:
            p.robot.description.urdf_dir = str(config)
            p.robot.description.top_xacro = model_xacros[0].name

        arm = info.arm_group()
        if arm:
            p.robot.planning_group.name = arm.name
            if arm.chain and arm.chain[0]:
                self.set_frames(arm.chain[0], arm.chain[1])
            if arm.joints:
                p.robot.planning_group.joints = arm.joints
            else:
                # An SRDF group may declare its members as a bare <chain> with no
                # explicit <joint> children — that is what the MoveIt Setup Assistant
                # emits by default. Without this fallback planning_group.joints stays
                # empty, and then EVERY named state captured in the wizard is stored
                # with an empty joint_values dict (gui/wizard.py capture_joints passes
                # robot_summary()['arm_joints'] to LiveCapture). The generated SRDF
                # gets <group_state> elements with no <joint> children, and MoveIt
                # rejects them at runtime with "named target does not exist".
                self._derive_group_joints_from_chain(config)
            p.robot.named_states = [
                NamedState(name=s.name, group=s.group, joint_values=s.joint_values)
                for s in info.states_for(arm.name)]

        if info.end_effectors:
            ee = info.end_effectors[0]
            g = p.robot.gripper
            g.eef_group_name, g.eef_name, g.eef_parent_link = ee.group, ee.name, ee.parent_link
            if g.kind is GripperKind.NONE:
                g.kind = GripperKind.SUCTION
            ee_grp = info.group(ee.group)
            if ee_grp and ee_grp.joints:
                g.command_joint = ee_grp.joints[0]
            for s in info.states_for(ee.group):        # open/closed group_states
                low = s.name.lower()
                if 'open' in low:
                    g.open_state = s.name
                elif 'close' in low:
                    g.closed_state = s.name

        self._load_controllers_from_base(config)
        self._load_deployment_from_base(base)
        # Grasp targets attach to the robot's OWN tool link. Without this the scene keeps
        # the model default 'tcp', which is silently right for FR30/FR3WML and silently
        # wrong for any cell whose tip_link is named differently.
        if p.robot.tip_link:
            p.scene.attach_link = p.robot.tip_link
        self._derive_touch_links(config)
        try:
            self.import_collision_matrix_from_srdf(srdfs[0])
        except Exception:
            pass
        return {
            'robot_name': info.robot_name,
            'group': arm.name if arm else None,
            'arm_joints': list(p.robot.planning_group.joints),
            'named_states': [s.name for s in (info.states_for(arm.name) if arm else [])],
            'gripper_controller': p.robot.gripper.controller_name,
            'arm_controller': p.robot.arm_controller.name,
        }

    def _derive_group_joints_from_chain(self, config_dir: Path) -> None:
        """Fill ``planning_group.joints`` from the base->tip kinematic chain.

        Used when the SRDF group declares only a ``<chain>``. Mirrors what MoveIt
        itself does when it expands a chain group, so the captured named states carry
        the same joint set the runtime expects.
        """
        p = self._require()
        try:
            from ..robotmodel.kinematic_chain import chain_movable_joints, parse_urdf
            from ..robotmodel.xacro_loader import compile_xacro
            xacros = sorted(config_dir.glob('*.urdf.xacro')) or sorted(config_dir.glob('*.xacro'))
            if not xacros:
                return
            robot = parse_urdf(compile_xacro(xacros[0]))
            joints = chain_movable_joints(robot, p.robot.base_frame, p.robot.tip_link)
            if joints:
                p.robot.planning_group.joints = joints
        except Exception:  # noqa: BLE001 - a missing/!compiling xacro must not block Step 1
            pass

    def _derive_touch_links(self, config_dir: Path) -> None:
        """Derive the scene-loader ``touch_links`` from the robot instead of asking:
        they are the links NEAREST THE TOOL, which legitimately touch a grasped object
        (self-collision relief for the AttachedCollisionObject). Taken as the last links
        of the base_link -> tip_link chain (for the FR30: wrist3_link, end_effector, tcp).
        """
        p = self._require()
        try:
            from ..robotmodel.kinematic_chain import link_chain, parse_urdf
            from ..robotmodel.xacro_loader import compile_xacro
            xacros = sorted(config_dir.glob('*.urdf.xacro'))
            if not xacros:
                return
            robot = parse_urdf(compile_xacro(xacros[0]))
            links, _joints = link_chain(robot, p.robot.base_frame, p.robot.tip_link)
            if len(links) >= 2:
                p.scene.touch_links = links[-3:]
        except Exception:  # noqa: BLE001 - keep the default; the user can still edit it
            pass

    def _load_deployment_from_base(self, base: Path) -> None:
        """Capture the CELL BRING-UP WIRING from the base's launch file: the vendor/sim
        bridges (gripper bridge, joint-state merger, real drivers) + where the arm's
        /joint_states is remapped. Without these the generated config's /joint_states
        misses the gripper joint -> incomplete TF -> the robot flickers in RViz.
        """
        from ..model import LaunchNodeSpec
        from ..robotmodel.launch_reader import read_bringup
        launches = sorted((base / 'launch').glob('bringup*.launch.py'))
        if not launches:
            return
        try:
            info = read_bringup(launches[0])
        except Exception:  # noqa: BLE001
            return
        p = self._require()
        if info.get('arm_joint_states_remap_to'):
            p.deployment.arm_joint_states_remap_to = info['arm_joint_states_remap_to']
        bridges = []
        for b in info.get('bridges', []):
            try:
                bridges.append(LaunchNodeSpec(**b))
            except Exception:  # noqa: BLE001
                continue
        if bridges:
            p.deployment.bridges = bridges

    def _load_controllers_from_base(self, config_dir: Path) -> None:
        import yaml
        f = config_dir / 'moveit_controllers.yaml'
        if not f.is_file():
            return
        data = yaml.safe_load(f.read_text()) or {}
        mgr = data.get('moveit_simple_controller_manager', {})
        p = self._require()
        for n in mgr.get('controller_names', []):
            c = mgr.get(n, {}) or {}
            ctype = c.get('type', '')
            if ctype == 'FollowJointTrajectory':
                p.robot.arm_controller.name = n
                p.robot.arm_controller.action_ns = c.get('action_ns', 'follow_joint_trajectory')
            elif ctype == 'GripperCommand':
                p.robot.gripper.controller_name = n
                p.robot.gripper.action_ns = c.get('action_ns', 'gripper_command')

    # ---- named states (S2) ----
    def add_named_state(self, name: str, joint_values: Dict[str, float],
                        group: Optional[str] = None) -> None:
        """Add/replace a named joint configuration (SRDF group_state)."""
        p = self._require()
        grp = group or p.robot.planning_group.name
        states = [s for s in p.robot.named_states if s.name != name]
        states.append(NamedState(name=name, group=grp,
                                 joint_values={k: float(v) for k, v in joint_values.items()}))
        p.robot.named_states = states

    def remove_named_state(self, name: str) -> None:
        p = self._require()
        p.robot.named_states = [s for s in p.robot.named_states if s.name != name]

    # ---- collision matrix (S1) ----
    def import_collision_matrix_from_srdf(self, srdf_path) -> int:
        """Set robot.disable_collisions from an existing SRDF (e.g. one made by the
        real MoveIt Setup Assistant). The reliable way to get the matrix, since
        collisions_updater can't be driven headless here. Returns the pair count.
        """
        from ..robotmodel.collision_matrix import parse_disable_collisions
        from ..model.robot import DisableCollisionSpec
        pairs = parse_disable_collisions(Path(srdf_path).read_text())
        self._require().robot.disable_collisions = [
            DisableCollisionSpec(link1=p.link1, link2=p.link2, reason=p.reason)
            for p in pairs
        ]
        return len(pairs)

    def compute_collisions(self, **kwargs) -> int:
        """Populate robot.disable_collisions via collisions_updater. Returns count."""
        from ..robotmodel.collision_matrix import compute_disable_collisions
        from ..model.robot import DisableCollisionSpec
        pairs = compute_disable_collisions(self._require(), **kwargs)
        self._require().robot.disable_collisions = [
            DisableCollisionSpec(link1=p.link1, link2=p.link2, reason=p.reason)
            for p in pairs
        ]
        return len(pairs)

    # ---- application (S4/S5) ----
    def set_application(self, app_type: str, global_planner_mode: str = 'pilz') -> None:
        app = self._require().application
        app.type = AppType(app_type)
        app.global_planner_mode = PlannerId(global_planner_mode)

    def add_move(self, name: str, *, position=None, orientation=None,
                 named: Optional[str] = None, joints=None,
                 motion: str = 'free', planner: Optional[str] = None,
                 speed: int = 50, role: str = 'generic',
                 aux=None, aux_is_center: bool = False,
                 allowed_start_tolerance: float = 0.1,
                 attached_collision_check: Optional[bool] = None,
                 wait_after_ms: int = 0) -> None:
        """Define a waypoint + its incoming motion segment and append it to the tree.

        TCP target if ``position`` is given; else a joint target (``named``/``joints``).
        Call again with the same name to RE-VISIT (e.g. return home): the waypoint def
        is upserted and the name is appended to the sequence again.

        ``allowed_start_tolerance`` (rad, per waypoint): start-state drift tolerated
        before the move to this waypoint executes (0.0 disables the check).
        ``attached_collision_check`` (per-move): when the gripper holds objects,
        True => the planner routes the held payload around the static meshes; False =>
        transparent; None => inherit the current runtime state.
        """
        app = self._require().application
        wtype = WaypointType.TCP if position is not None else WaypointType.JOINT
        wp = Waypoint(
            name=name, type=wtype,
            position=list(position) if position is not None else None,
            orientation=list(orientation) if orientation is not None else None,
            joints=list(joints) if joints is not None else None,
            named=named, role=WaypointRole(role),
            allowed_start_tolerance=allowed_start_tolerance)
        app.waypoints = [w for w in app.waypoints if w.name != name] + [wp]
        seg = MotionSegment(
            to_waypoint=name, motion=MotionType(motion),
            planner=PlannerId(planner) if planner else None, speed=speed,
            aux=list(aux) if aux is not None else None, aux_is_center=aux_is_center,
            attached_collision_check=attached_collision_check,
            wait_after_ms=max(0, int(wait_after_ms)))
        app.segments = [s for s in app.segments if s.to_waypoint != name] + [seg]
        app.sequence.append(name)

    def add_tool_action(self, at_waypoint: str, kind: str,
                        payload_ref: Optional[str] = None) -> None:
        self._require().application.tool_actions.append(
            ToolAction(at_waypoint=at_waypoint, kind=ToolActionKind(kind),
                       payload_ref=payload_ref))

    def set_wait_after(self, waypoint: str, ms: int) -> None:
        """Process-layer Wait block: pause (ms) after the waypoint + its tool actions."""
        seg = self._require().application.segment_for(waypoint)
        if seg is None:
            raise ValueError(f'no move named "{waypoint}"')
        seg.wait_after_ms = max(0, int(ms))

    def set_loop(self, cycles: int, start: str = None, end: str = None) -> None:
        """Repeat a slice of the sequence. start/end are inclusive waypoint names;
        None/None keeps the historical whole-sequence behaviour."""
        app = self._require().application
        app.loop_cycles = int(cycles)
        app.loop_start = start or None
        app.loop_end = end or None

    def clear_application(self) -> None:
        app = self._require().application
        app.waypoints, app.segments, app.tool_actions, app.sequence = [], [], [], []

    # ---- perception (the dedicated Perception step, TSA v4 / D-015) ----
    def _perception(self) -> PerceptionSpec:
        p = self._require()
        if p.perception is None:
            p.perception = PerceptionSpec()
        return p.perception

    _CAMERA_CLEARABLE = ('replace_include', 'with_include')

    def set_camera(self, **fields) -> None:
        """Upsert the cell camera (topics, frames, variant includes, synthetic cloud).
        For the two include fields an EMPTY STRING clears the value (so a wrong path
        can be corrected back to 'no rewrite'); elsewhere None means 'leave as is'."""
        per = self._perception()
        cam = per.camera or CameraSpec()
        update = {}
        for k, v in fields.items():
            if k in self._CAMERA_CLEARABLE:
                if v is not None:
                    update[k] = v or None      # '' -> None (clear)
            elif v is not None:
                update[k] = v
        per.camera = cam.model_copy(update=update)

    def set_perception_timing(self, settle_ms: Optional[int] = None,
                              detect_timeout_ms: Optional[int] = None) -> None:
        per = self._perception()
        if settle_ms is not None:
            per.settle_ms = max(0, int(settle_ms))
        if detect_timeout_ms is not None:
            per.detect_timeout_ms = max(1, int(detect_timeout_ms))

    def upsert_detector(self, name: str, *, method: str = 'color_mask',
                        params: Optional[dict] = None, continuous: bool = True,
                        rate_hz: float = 10.0) -> None:
        """Add or update a named detector (the Perception step's list entries)."""
        import re
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name.strip()):
            raise ValueError(
                f'invalid detector name "{name}": it becomes the ROS node name '
                f'detector_<name> and the topic /perception/<name>/detections — '
                f'use [A-Za-z][A-Za-z0-9_]*')
        per = self._perception()
        old = per.detector_by_name(name.strip())
        spec = DetectorSpec(name=name.strip(), method=DetectionMethod(method),
                            params=dict(params or {}), continuous=continuous,
                            rate_hz=float(rate_hz),
                            # per-detector topic overrides are hand-edited data the
                            # GUI cannot set — an upsert must not strip them
                            rgb_topic=old.rgb_topic if old else None,
                            depth_topic=old.depth_topic if old else None,
                            camera_info_topic=old.camera_info_topic if old else None)
        per.detectors = [d for d in per.detectors if d.name != spec.name] + [spec]

    def remove_detector(self, name: str) -> None:
        per = self._perception()
        per.detectors = [d for d in per.detectors if d.name != name]
        for wp in self._require().application.waypoints:
            if wp.vision is not None and wp.vision.detector == name:
                wp.vision = None

    def detector_names(self) -> List[str]:
        p = self._require()
        return [d.name for d in p.perception.detectors] if p.perception else []

    def bind_vision(self, waypoint: str, detector: str, *, dx: float = 0.0,
                    dy: float = 0.0, dz: float = 0.0,
                    orientation: str = 'keep') -> None:
        """Make a waypoint vision-driven: its position is overwritten at run time
        from the detector (+ base-frame offsets); the captured pose stays as the
        recorded fallback."""
        wp = self._require().application.waypoint_by_name(waypoint)
        if wp is None:
            raise ValueError(f'no waypoint named "{waypoint}"')
        wp.relative = None                 # mutually exclusive with a relative binding
        wp.vision = VisionBinding(detector=detector, dx=float(dx), dy=float(dy),
                                  dz=float(dz), orientation=orientation)

    def unbind_vision(self, waypoint: str) -> None:
        wp = self._require().application.waypoint_by_name(waypoint)
        if wp is not None:
            wp.vision = None

    def bind_relative(self, waypoint: str, step: str, *, dx: float = 0.0,
                      dy: float = 0.0, dz: float = 0.0, droll: float = 0.0,
                      dpitch: float = 0.0, dyaw: float = 0.0) -> None:
        """Make a waypoint STEP-RELATIVE (D-017): its pose is derived at run time
        from another step's final pose (vision included) + offsets. Clears any
        vision binding (the two are mutually exclusive)."""
        from ..model import RelativeBinding
        wp = self._require().application.waypoint_by_name(waypoint)
        if wp is None:
            raise ValueError(f'no waypoint named "{waypoint}"')
        if step == waypoint:
            raise ValueError(f'"{waypoint}" cannot be relative to itself')
        wp.vision = None
        wp.relative = RelativeBinding(step=step, dx=float(dx), dy=float(dy),
                                      dz=float(dz), droll=float(droll),
                                      dpitch=float(dpitch), dyaw=float(dyaw))

    def unbind_relative(self, waypoint: str) -> None:
        wp = self._require().application.waypoint_by_name(waypoint)
        if wp is not None:
            wp.relative = None

    # ---- scene (S3) ----
    def set_payload(self, obj_id: str, dims, attach_link: Optional[str] = None,
                    attach_offset=(0.0, 0.0, 0.0)) -> None:
        p = self._require()
        p.scene.payload = Payload(
            id=obj_id, dims=list(dims),
            attach_link=attach_link or p.robot.tip_link,
            attach_offset=list(attach_offset))

    def add_scene_object(self, obj_id: str, dims, position, *, shape: str = 'box',
                         frame: Optional[str] = None, dynamic: bool = False,
                         collision: bool = True, orientation=(0.0, 0.0, 0.0, 1.0),
                         source: str = 'primitive', category: Optional[str] = None,
                         mesh_resource: Optional[str] = None, scale=(1.0, 1.0, 1.0),
                         grasp_target: bool = False, release_policy: str = 'freeze',
                         isaac_grasp_method: str = 'fixed_joint',
                         touchable_collision_ids=None, aabb_center=(0.0, 0.0, 0.0)) -> None:
        p = self._require()
        # category (static|actuated|dynamic) is authoritative when given; otherwise
        # fall back to the legacy `dynamic` flag (category is inferred from it).
        kw = dict(dynamic=dynamic, collision=collision)
        if category is not None:
            kw = dict(category=SceneObjectCategory(category))
        p.scene.objects = [o for o in p.scene.objects if o.id != obj_id] + [SceneObject(
            id=obj_id, source=SceneObjectSource(source), shape=ShapeType(shape),
            dims=list(dims), mesh_resource=mesh_resource, scale=list(scale),
            aabb_center=list(aabb_center),
            frame=frame or p.robot.base_frame,
            position=list(position), orientation=list(orientation),
            grasp_target=grasp_target, release_policy=ReleasePolicy(release_policy),
            isaac_grasp_method=IsaacGraspMethod(isaac_grasp_method),
            touchable_collision_ids=list(touchable_collision_ids or []), **kw)]

    def set_object_category(self, obj_id: str, category: str) -> None:
        """Re-classify an existing object (static | actuated | dynamic)."""
        p = self._require()
        cat = SceneObjectCategory(category)
        for o in p.scene.objects:
            if o.id == obj_id:
                o.category = cat
                o.dynamic = cat is SceneObjectCategory.DYNAMIC  # keep the flag in sync
                break

    def set_object_grasp(self, obj_id: str, *, grasp_target: Optional[bool] = None,
                         release_policy: Optional[str] = None,
                         isaac_grasp_method: Optional[str] = None,
                         touchable_collision_ids=None) -> None:
        """Scene: set the dynamic-object grasp attributes (attach on close, freeze/
        gravity on release, Isaac method, touchable meshes)."""
        for o in self._require().scene.objects:
            if o.id == obj_id:
                if grasp_target is not None:
                    o.grasp_target = bool(grasp_target)
                if release_policy is not None:
                    o.release_policy = ReleasePolicy(release_policy)
                if isaac_grasp_method is not None:
                    o.isaac_grasp_method = IsaacGraspMethod(isaac_grasp_method)
                if touchable_collision_ids is not None:
                    o.touchable_collision_ids = list(touchable_collision_ids)
                break

    @staticmethod
    def _norm_topic(value: str, what: str) -> str:
        """Normalise a ROS topic typed by a human or read out of a USD ActionGraph.

        Done HERE, not in a pydantic validator: no model sets validate_assignment, so a
        @field_validator never fires on plain attribute assignment — and every writer
        (the GUI combo, the USD auto-apply, import_scene_yaml) goes through this method.
        An empty topic would reach rclcpp's create_subscription and throw
        InvalidTopicNameError at scene_manager_node construction.
        """
        t = (value or '').strip()
        if not t:
            raise ValueError(f'{what} must not be empty')
        if any(c.isspace() for c in t):
            raise ValueError(f'{what} must not contain whitespace: {t!r}')
        return t if t.startswith('/') else '/' + t

    def set_scene_loader_params(self, *, gripper_cmd_topic=None, scene_reset_topic=None,
                                attach_link=None, touch_links=None,
                                attached_collision_check=None,
                                grasp_attach_mode=None) -> None:
        """Scene-loader (scene_manager_node) params emitted into scene.yaml."""
        s = self._require().scene
        if gripper_cmd_topic is not None:
            s.gripper_cmd_topic = self._norm_topic(gripper_cmd_topic, 'gripper_cmd_topic')
        if scene_reset_topic is not None:
            s.scene_reset_topic = self._norm_topic(scene_reset_topic, 'scene_reset_topic')
        if attach_link is not None:
            s.attach_link = attach_link
        if touch_links is not None:
            s.touch_links = list(touch_links)
        if attached_collision_check is not None:
            s.attached_collision_check = bool(attached_collision_check)
        if grasp_attach_mode is not None:
            if grasp_attach_mode not in ('remove', 'attach_box'):
                raise ValueError("grasp_attach_mode must be 'remove' or 'attach_box'")
            s.grasp_attach_mode = grasp_attach_mode

    def import_scene_yaml(self, path, replace: bool = True) -> int:
        """Load the cell scene from a scene_manager_node ``scene.yaml`` (the output of
        the adaptation-sprint's ``scene_from_usd.py``). This is the RELIABLE path: the
        meshes + base-frame poses are already resolved, so the generated config matches
        the hand-made baseline. Populates the SceneObjects (mesh/box/cyl/sphere, category
        from ``dynamic``, grasp_target from ``attach_object_ids``) + the scene-loader
        params. Returns the object count.
        """
        import yaml
        data = yaml.safe_load(Path(path).read_text()) or {}
        params = None
        for v in data.values():
            if isinstance(v, dict) and 'ros__parameters' in v:
                params = v['ros__parameters']
                break
        if params is None:
            raise ValueError('not a scene_manager_node scene.yaml (no ros__parameters)')
        p = self._require()
        if replace:
            p.scene.objects = []
        objects = params.get('objects', {}) or {}
        order = params.get('object_ids') or list(objects.keys())
        attach_ids = set(params.get('attach_object_ids', []) or [])
        for oid in order:
            o = objects.get(oid, {}) or {}
            typ = o.get('type', 'box')
            cat = 'dynamic' if bool(o.get('dynamic', False)) else 'static'
            pos = o.get('position', [0.0, 0.0, 0.0])
            quat = o.get('orientation', [0.0, 0.0, 0.0, 1.0])
            grasp = oid in attach_ids
            if typ == 'mesh':
                self.add_scene_object(oid, [1.0, 1.0, 1.0], pos, shape='mesh',
                                      mesh_resource=o.get('mesh_path', ''),
                                      scale=o.get('scale', [1.0, 1.0, 1.0]),
                                      orientation=quat, category=cat, grasp_target=grasp,
                                      source='usd')
            elif typ == 'cylinder':
                self.add_scene_object(oid, [o.get('radius', 0.05), o.get('height', 0.1)],
                                      pos, shape='cylinder', orientation=quat,
                                      category=cat, grasp_target=grasp)
            elif typ == 'sphere':
                self.add_scene_object(oid, [o.get('radius', 0.05)], pos, shape='sphere',
                                      orientation=quat, category=cat, grasp_target=grasp)
            else:  # box
                self.add_scene_object(oid, o.get('size', [0.1, 0.1, 0.1]), pos, shape='box',
                                      orientation=quat, category=cat, grasp_target=grasp)
        self.set_scene_loader_params(
            gripper_cmd_topic=params.get('gripper_cmd_topic'),
            scene_reset_topic=params.get('scene_reset_topic'),
            attach_link=params.get('attach_link'),
            touch_links=params.get('touch_links'),
            attached_collision_check=params.get('attached_collision_check'))
        if 'force_republish_hz' in params:
            p.scene.force_republish_hz = float(params['force_republish_hz'])
        return len(order)

    def import_usd_cell(self, usd_path, mesh_pkg: str, base_prim: Optional[str] = None) -> list:
        """Step 2 (recommended): read the cell prims from the USD (base-aligned poses,
        which reproduce the baseline) and AUTO-SUGGEST a collision mesh per group by
        scanning ``<mesh_pkg>/meshes``. Returns the editable group rules; the per-prim
        poses are held for :meth:`apply_usd_mapping`."""
        from ..importers.usd_scene import (build_group_rules, read_cell_prims,
                                           read_ros2_bool_topics, scan_meshes)
        p = self._require()
        if base_prim is None:
            base_prim = f'/World/{p.robot.robot_name}/{p.robot.base_frame}'
        self._usd_prims = read_cell_prims(usd_path, base_prim=base_prim,
                                          robot_hint=p.robot.robot_name)
        # surfaced to the user: if the mesh dir is not found every suggestion is empty,
        # which would build a scene the loader cannot render.
        self.mesh_scan = scan_meshes(mesh_pkg, p.robot.base_moveit_config_path)
        # The Isaac side NAMES the gripper-close signal; the base moveit_config does not.
        # Auto-apply only while the value is still the shipped default, so a project that
        # already carries a deliberate topic is never silently overwritten on reload.
        self.usd_bool_topics = read_ros2_bool_topics(
            str(usd_path), exclude=(p.scene.scene_reset_topic,))
        self.usd_base_prim_ok = getattr(read_cell_prims, 'base_prim_ok', True)
        self.usd_base_prim_tried = getattr(read_cell_prims, 'base_prim_tried', base_prim)
        default_topic = SceneSpec.model_fields['gripper_cmd_topic'].default
        if self.usd_bool_topics and p.scene.gripper_cmd_topic == default_topic:
            self.set_scene_loader_params(gripper_cmd_topic=self.usd_bool_topics[0])
        return build_group_rules(self._usd_prims, mesh_pkg,
                                 hint_path=p.robot.base_moveit_config_path)

    def apply_usd_mapping(self, rules, replace: bool = True) -> int:
        """Build the scene from the held USD prims + the (edited) group rules: one mesh
        SceneObject per included prim (pose from the USD, mesh/category/grasp from its
        group rule). Returns the object count.

        REFUSES groups that are included but have NO mesh: they would be emitted with an
        empty ``mesh_path`` and the scene loader could not render them (a silently broken
        scene). Raises ValueError naming the offending groups.
        """
        p = self._require()
        broken = [r['group'] for r in rules
                  if r.get('include') and not str(r.get('mesh', '')).strip()]
        if broken:
            raise ValueError(
                'these included groups have NO mesh resource: ' + ', '.join(broken) +
                '. Set a package:// mesh or un-tick "include". (A wrong/empty "Mesh '
                'package" makes every suggestion empty.)')
        by_group = {r['group']: r for r in rules}
        if replace:
            p.scene.objects = []
        included = [(pr, by_group[pr['group']]) for pr in self._usd_prims
                    if by_group.get(pr['group']) and by_group[pr['group']].get('include')]

        def _is_grasp(r):
            return bool(r.get('grasp')) and r.get('category') == 'dynamic'
        # non-grasp objects (structure/crate) first, grasp targets last — the baseline
        # convention. sorted() is stable, so USD order is preserved within each bucket.
        included.sort(key=lambda pr_r: _is_grasp(pr_r[1]))
        for pr, r in included:
            # everything is a MESH at its prim origin; dims carries the local AABB extents
            # only so a grasped object can attach as its bounding BOX (attach_box mode).
            self.add_scene_object(
                pr['name'], pr.get('dims') or r.get('dims') or [0.1, 0.1, 0.1],
                pr['position'], shape='mesh', mesh_resource=r.get('mesh', ''),
                orientation=pr['orientation'], category=r.get('category', 'static'),
                grasp_target=_is_grasp(r), source='usd',
                aabb_center=pr.get('local_center') or [0.0, 0.0, 0.0])
        return len(included)

    def import_usd_scene(self, usd_path, dynamic: bool = False, replace: bool = False,
                         base_prim: Optional[str] = None, category=None,
                         classify=None) -> int:
        """Import scene objects from a USD stage (AABB boxes), aligned to the robot base.

        ``base_prim`` is the USD path of the robot root/base to align to; if None it is
        auto-detected using the robot name (so cell-world coords become base-relative).
        ``category`` (default static) is assigned to every imported object; ``classify``
        may override it per prim. The user then re-classifies in the wizard. Returns the
        count added.
        """
        from ..importers import import_usd
        p = self._require()
        objs = import_usd(usd_path, default_frame=p.robot.base_frame, dynamic=dynamic,
                          base_prim=base_prim, robot_hint=p.robot.robot_name,
                          category=category, classify=classify)
        if replace:
            p.scene.objects = []
        existing = {o.id for o in p.scene.objects}
        for o in objs:
            if o.id not in existing:
                p.scene.objects.append(o)
        return len(objs)

    # ---- validation + generation (S6) ----
    def validate(self) -> List[str]:
        p = self._require()
        problems: List[str] = []
        if not p.robot.planning_group.joints:
            problems.append('planning group has no joints')
        if not p.robot.disable_collisions:
            problems.append('no self-collision matrix (SRDF will be over-conservative)')
        try:
            problems += get_application(p.application.type).validate(p)
        except KeyError as exc:
            problems.append(str(exc))
        return problems

    def generate(self, output_dir) -> GenerationManifest:
        return Orchestrator().generate(self._require(), output_dir)

    # ---- Step 3: generate the intermediate scene+planner config (for RViz config) ----
    def scene_loader_package_name(self) -> str:
        """The intermediate config package name (``<robot>_scene_loader_moveit_config``)."""
        return f'{self._require().robot.robot_name}_scene_loader_moveit_config'

    def generate_scene_loader_config(self, output_dir, package_name=None) -> GenerationManifest:
        """Step 3: emit ONLY the standalone scene+planner config (same emitter as the
        bundle's ``_trainit_config``, so what you configure == what ships)."""
        from ..generator.orchestrator import generate_scene_loader_config
        pkg = package_name or self.scene_loader_package_name()
        self.scene_loader_pkg = pkg          # remember it for Step 6's bring-up procedure
        self.workspace_root = self._workspace_root(output_dir)  # colcon build runs HERE
        return generate_scene_loader_config(self._require(), output_dir, pkg)

    @staticmethod
    def _workspace_root(output_dir) -> str:
        """The colcon workspace ROOT (the parent of ``src/``) for a package written under
        ``<ws>/src/...``. colcon build must run there, not in the chosen output dir."""
        parts = os.path.abspath(str(output_dir)).split(os.sep)
        if 'src' in parts:
            root = os.sep.join(parts[:parts.index('src')])
            return root or os.sep
        return os.path.abspath(str(output_dir))

    # ---- Step 4: mode + guided bring-up ----
    def set_mode(self, mode: str) -> None:
        """Step 4: the mode the user will configure/run in (mock | isaac | real)."""
        if mode not in ('mock', 'isaac', 'real'):
            raise ValueError(f"mode must be mock|isaac|real, got {mode!r}")
        self._require().deployment.default_mode = mode

    def build_snippet(self, package: str, ws_root: str = None) -> str:
        """The terminal snippet to build+source a generated package. colcon build
        runs at the WORKSPACE ROOT (parent of src/), not the output dir."""
        ws_root = ws_root or self.workspace_root or '<ros2_ws>'
        return (f'cd {ws_root}\n'
                f'colcon build --packages-select {package}\n'
                f'source install/setup.bash')

    def bringup_procedure(self, mode: str, config_package: str,
                          usd_path: Optional[str] = None,
                          ws_root: str = None) -> str:
        """Step 4: the guided procedure to BUILD + bring up the scene+planner config so
        the application can be configured against a faithful RViz. The application itself
        is launched later (Step 8 generates it; the bundle README has the run command)."""
        ws_root = ws_root or self.workspace_root or '<ros2_ws>'
        steps: List[str] = []
        n = 1
        if mode == 'isaac':
            steps.append(f'{n}) Open Isaac Sim, load your USD scene'
                         + (f'\n     ({usd_path})' if usd_path else '')
                         + ', run the spawn + grasp-adapter scripts, then press PLAY.')
            n += 1
        steps.append(f'{n}) Build the scene+planner config you generated at Step 3:\n'
                     f'     cd {ws_root} && source /opt/ros/humble/setup.bash\n'
                     f'     colcon build --packages-select {config_package}\n'
                     f'     source install/setup.bash')
        n += 1
        steps.append(f'{n}) Bring up the cell (move_group + planners + scene + RViz):\n'
                     f'     cd {ws_root} && source /opt/ros/humble/setup.bash && source install/setup.bash\n'
                     f'     ros2 launch {config_package} bringup.launch.py mode:={mode}')
        n += 1
        steps.append(f'{n}) Now configure the application (next steps) against this RViz. '
                     'The application is launched only AFTER you generate the bundle '
                     '(its README has the run command).')
        return '\n'.join(steps)

    # ---- helpers ----
    def has_gripper(self) -> bool:
        return self._require().robot.gripper.kind is not GripperKind.NONE

    def _require(self) -> CanonicalProject:
        if self.project is None:
            raise RuntimeError('no project loaded (call new_from_robot/open_project first)')
        return self.project
