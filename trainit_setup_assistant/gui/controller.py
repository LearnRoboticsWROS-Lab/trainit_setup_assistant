"""AssistantController — the ROS-free brain behind the wizard.

Holds the in-progress :class:`CanonicalProject` and exposes the operations the wizard
(or a CLI, or a future web GUI) performs: load a robot, confirm group/frames, capture
named states, edit the scene/application, and generate the bundle. No Qt, no ROS here
(xacro compilation is delegated to robotmodel and only needs a ROS env for ``$(find)``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from ..applications import get_application
from ..generator import GenerationManifest, Orchestrator
from ..model import (
    BundleSpec,
    CanonicalProject,
    MotionSegment,
    NamedState,
    Payload,
    SceneObject,
    ToolAction,
    Waypoint,
    load_project,
    save_project,
)
from ..model.enums import (
    AppType,
    GripperKind,
    MotionType,
    PlannerId,
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
                 allowed_start_tolerance: float = 0.1) -> None:
        """Define a waypoint + its incoming motion segment and append it to the tree.

        TCP target if ``position`` is given; else a joint target (``named``/``joints``).
        Call again with the same name to RE-VISIT (e.g. return home): the waypoint def
        is upserted and the name is appended to the sequence again.

        ``allowed_start_tolerance`` (rad, per waypoint): start-state drift tolerated
        before the move to this waypoint executes (0.0 disables the check).
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
            aux=list(aux) if aux is not None else None, aux_is_center=aux_is_center)
        app.segments = [s for s in app.segments if s.to_waypoint != name] + [seg]
        app.sequence.append(name)

    def add_tool_action(self, at_waypoint: str, kind: str,
                        payload_ref: Optional[str] = None) -> None:
        self._require().application.tool_actions.append(
            ToolAction(at_waypoint=at_waypoint, kind=ToolActionKind(kind),
                       payload_ref=payload_ref))

    def clear_application(self) -> None:
        app = self._require().application
        app.waypoints, app.segments, app.tool_actions, app.sequence = [], [], [], []

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
                         source: str = 'primitive', category: Optional[str] = None) -> None:
        p = self._require()
        # category (static|actuated|dynamic) is authoritative when given; otherwise
        # fall back to the legacy `dynamic` flag (category is inferred from it).
        kw = dict(dynamic=dynamic, collision=collision)
        if category is not None:
            kw = dict(category=SceneObjectCategory(category))
        p.scene.objects = [o for o in p.scene.objects if o.id != obj_id] + [SceneObject(
            id=obj_id, source=SceneObjectSource(source), shape=ShapeType(shape),
            dims=list(dims), frame=frame or p.robot.base_frame,
            position=list(position), orientation=list(orientation), **kw)]

    def set_object_category(self, obj_id: str, category: str) -> None:
        """Re-classify an existing object (static | actuated | dynamic)."""
        p = self._require()
        cat = SceneObjectCategory(category)
        for o in p.scene.objects:
            if o.id == obj_id:
                o.category = cat
                o.dynamic = cat is SceneObjectCategory.DYNAMIC  # keep the flag in sync
                break

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

    # ---- helpers ----
    def has_gripper(self) -> bool:
        return self._require().robot.gripper.kind is not GripperKind.NONE

    def _require(self) -> CanonicalProject:
        if self.project is None:
            raise RuntimeError('no project loaded (call new_from_robot/open_project first)')
        return self.project
