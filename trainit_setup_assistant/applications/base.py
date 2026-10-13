"""Application templates + the shared bt_params builder.

The ``bt_params.yaml`` schema is uniform across applications (a flat set of runtime
globals + per-waypoint motion specs); only the BEHAVIOR TREE differs per app. So the
bt_params context is built here once, and each :class:`ApplicationTemplate` only emits
its tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..model import CanonicalProject, effective_planner
from ..model.enums import (EndEffectorActuation, ExecutionMode, MotionType,
                           WaypointType)


# --- end-effector actuation resolution (ADR-0010) ---
def resolve_gripper_positions(robot) -> tuple:
    """(close_pos, open_pos) as floats for the ``GripperCommand`` the runtime sends on
    grasp/release. Only meaningful for JOINT_POSITION actuation; TRIGGER keeps the runtime
    defaults (1.0/0.0 == suction on/off). SRDF_STATE targets are resolved to a numeric joint
    value from ``named_states`` (the command joint's value); an unresolvable target falls
    back to the trigger default so generation never fails."""
    g = robot.gripper
    open_t, closed_t = g.joint_targets()  # (open, closed): numbers (ANGLE) | state names (SRDF)

    def _val(target, default: float) -> float:
        if isinstance(target, (int, float)):
            return float(target)
        cj = g.command_joint
        for ns in robot.named_states:
            if ns.name == target and cj and cj in ns.joint_values:
                return float(ns.joint_values[cj])
        return default

    return _val(closed_t, 1.0), _val(open_t, 0.0)


# --- scene/payload key naming (shared by bt_params AND the tree) ---
def scene_dims_key(obj_id: str) -> str:
    return f'scene_{obj_id}_dims'


def scene_pos_key(obj_id: str) -> str:
    return f'scene_{obj_id}_pos'


def payload_dims_key(payload_id: str) -> str:
    return f'payload_{payload_id}_dims'


def _token(value):
    return value.value if value is not None else None


def build_bt_params_context(project: CanonicalProject) -> dict:
    """Prepare the context the ``app/bt_params.yaml.j2`` template renders.

    Folds each waypoint's incoming MotionSegment onto the waypoint block (exactly how
    bt_params stores motion), and lays out the flat scene/payload params the tree
    references by ``{key}``.
    """
    app = project.application
    robot = project.robot

    close_pos, open_pos = resolve_gripper_positions(robot)
    # The end-effector CONTRACT (ADR-0022 D6): a canonical project carries no legacy gripper key
    # -- TMR's runner exits with code 2 on one next to the block -- and one `eef` block instead.
    # A legacy project renders every byte it rendered before (the block is '' and the gripper
    # keys keep their legacy values).
    eef_block = ''
    if robot.gripper.contract is not None:
        from ..generator.eef_contract import (device, eef_parameter_block,
                                              render_parameter_block)
        block = eef_parameter_block(device(project))
        eef_block = '\n'.join(render_parameter_block(block, indent=4))
    globals_ctx = {
        'planner_mode': app.global_planner_mode.value,
        'process_path_backend': app.process_path_backend,
        'group': robot.planning_group.name,
        'base_frame': robot.base_frame,
        'tip_link': robot.tip_link,
        # The gripper action is a CONTROLLER action, so it follows the cell namespace
        # (ADR-0016 Decision 3): /suctioncup_controller/gripper_command at the root,
        # /r3/suctioncup_controller/gripper_command under /r3. The BT node resolves it as
        # given, and it is the one name in bt_params.yaml that names a controller. Empty
        # namespace => unchanged, so the golden's bytes are untouched.
        'gripper_action': (robot.under_namespace(robot.gripper.gripper_action_ns())
                           if robot.gripper.gripper_action_ns() and not eef_block else ''),
        'process_controller': app.process_controller,
        'bt_tree_id': app.bt_tree_id,
        'enforce_validation': app.enforce_validation,
        # end-effector abstraction (ADR-0010). The template emits the actuation block ONLY
        # for JOINT_POSITION, so a TRIGGER cell (the deterministic golden) renders byte-
        # identically and the runtime keeps its 1.0/0.0 defaults. For JOINT_POSITION the
        # runtime drives the joint to these targets AND (via gripper_cmd_topic) fires the
        # per-backend sim grasp adapter (Isaac SurfaceGripper / Gazebo LinkAttacher).
        'gripper_actuation': robot.gripper.actuation.value,
        'gripper_close_position': close_pos,
        'gripper_open_position': open_pos,
        # ADR-0012 gate: a trigger-only end-effector (interacts_with_object False) publishes no
        # grasp Bool, so the runtime creates no publisher. Default True => golden byte-identical.
        'gripper_cmd_topic': (project.scene.gripper_cmd_topic
                              if robot.gripper.interacts_with_object else ''),
        # the `eef` parameter block (ADR-0022), pre-rendered; '' for a legacy project
        'eef_block': eef_block,
    }

    scene_params: List[tuple] = []
    for obj in project.scene.objects:
        if obj.is_planning_collision():
            scene_params.append((scene_dims_key(obj.id), obj.aabb_dims()))
            scene_params.append((scene_pos_key(obj.id), list(obj.position)))

    payload = None
    if project.scene.payload:
        payload = (payload_dims_key(project.scene.payload.id),
                   list(project.scene.payload.dims))

    # vNext M2.7 D3, scoped by the owner at G0 (section 8): a CANONICAL project (one with an
    # end-effector contract, ADR-0022) writes the planner TMR runs into every move block; a
    # legacy project writes only a planner the user chose, as before M2.7.
    canonical = robot.gripper.contract is not None
    waypoints = []
    for wp in app.waypoints:
        seg = app.segment_for(wp.name)
        block = {
            'name': wp.name,
            'type': wp.type.value,
            'position': list(wp.position) if wp.position else None,
            'orientation': list(wp.orientation) if wp.orientation else None,
            'joints': list(wp.joints) if wp.joints else None,
            'named': wp.named,
            'motion': seg.motion.value if seg else None,
            # canonical: the planner TMR runs, written explicitly -- never a blank one for a
            # ptp/free move, so the bundle does not rely on TMR's blank-planner rules
            'planner': (_token(effective_planner(seg.motion if seg else None,
                                                 seg.planner if seg else None,
                                                 app.global_planner_mode)) if canonical else
                        (seg.planner.value if (seg and seg.planner) else None)),
            'speed': seg.speed if seg else None,
            'aux': (list(seg.aux) if (seg and seg.aux) else None),
            'aux_is_center': (seg.aux_is_center if seg else False),
            'is_tcp': wp.type is WaypointType.TCP,
        }
        waypoints.append(block)

    return {
        'g': globals_ctx,
        'scene_params': scene_params,
        'payload': payload,
        'waypoints': waypoints,
    }


class ApplicationTemplate(ABC):
    """Maps a configured project onto one application's Behavior Tree."""

    app_type: str

    @abstractmethod
    def validate(self, project: CanonicalProject) -> List[str]:
        """Return a list of human-readable problems (empty == OK).

        Every implementation STARTS from :meth:`validate_common`, so the checks that
        hold for any application are inherited rather than re-derived per template.
        """

    # --- app-AGNOSTIC validation (inherited by every application template) ---
    def validate_common(self, project: CanonicalProject) -> List[str]:
        """Problems that do not depend on which application this is.

        Lives here so a new template gets them for free: its ``validate`` only adds the
        app-specific checks on top of this list. The end-effector contract's refusals
        (ADR-0022 R1-R26, refused again before any file) and its warnings are among them, so
        the GUI shows them before Generate is pressed.
        """
        from ..generator.eef_contract import eef_contract_problems, eef_contract_warnings
        return (self._validate_execution(project) + eef_contract_problems(project)
                + eef_contract_warnings(project))

    # --- REFUSALS: what generation must not even start with ---
    def refusals(self, project: CanonicalProject) -> List[str]:
        """Problems that REFUSE generation before a single file is written.

        :meth:`validate` returns warnings, because a partial bundle usually still helps.
        These are the cases where it does not: the bundle would build, launch and quietly do
        something other than what was configured. The orchestrator raises on them before the
        output directory exists; each one is ALSO in :meth:`validate`, so the GUI shows it
        before Generate is pressed. A template adds its own on top of these (vNext M0C).
        """
        conflicts = (self._servo_shape_conflict(seg) for seg in project.application.segments)
        return [m for m in conflicts if m]

    @staticmethod
    def _servo_shape_conflict(seg) -> str:
        """``execution: servo`` with ``motion: lin|circ`` -- one of the two would be ignored."""
        if seg.execution is not ExecutionMode.SERVO or seg.motion not in (MotionType.LIN,
                                                                           MotionType.CIRC):
            return ''
        return (f'move "{seg.to_waypoint}": execution "servo" cannot be combined '
                f'with motion "{seg.motion.value}" — MoveIt Servo has no Pilz '
                f'{seg.motion.value.upper()}, so one of the two would be silently '
                'ignored (ADR-0015). Use ptp/free, or execution "plan".')

    def _validate_execution(self, project: CanonicalProject) -> List[str]:
        """Per-move execution mode (ADR-0015, Decision 4).

        Two bounds on ``execution: servo``:

        - REFUSED with ``motion: lin|circ``. Those shapes are Pilz LIN/CIRC and MoveIt
          Servo has no Pilz anything: it streams a Cartesian/joint velocity command, so
          ONE of the two choices would be silently ignored. Silent is the problem.
        - WARNED on a waypoint with neither a vision nor a relative binding: streaming
          to a pose that was taught once and never moves buys nothing a plan does not
          already give (and gives up the planner's collision-free route).
        """
        app = project.application
        problems: List[str] = []
        for seg in app.segments:
            if seg.execution is not ExecutionMode.SERVO:
                continue
            conflict = self._servo_shape_conflict(seg)
            if conflict:
                problems.append(conflict)               # and REFUSED: see refusals()
                continue
            wp = app.waypoint_by_name(seg.to_waypoint)
            if wp is not None and wp.vision is None and wp.relative is None:
                problems.append(
                    f'move "{seg.to_waypoint}": execution "servo" on a static taught '
                    'pose — a streamed move buys nothing over a planned one (ADR-0015). '
                    'Bind the waypoint to a detector or to another step, or use '
                    'execution "plan".')
        return problems

    @abstractmethod
    def tree_filename(self, project: CanonicalProject) -> str:
        """File name for the generated BT XML (e.g. 'pick_place.xml')."""

    @abstractmethod
    def build_tree_xml(self, project: CanonicalProject) -> str:
        """Return the Behavior Tree XML for this application."""
