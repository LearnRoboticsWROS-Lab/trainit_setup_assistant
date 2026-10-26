"""The ``task_program`` application: a Canonical Task Program, validated, resolved and planned
(vNext M3.3, ADR-0024, ADR-0025).

It is not a template: the program says what the robot does (``model.program``), and this module
turns it into the data TMR 2.2.0 executes, refusing before any file is written what TMR could not
run as written:

    the program (waypoints, motion profiles, paths, tasks)
      -> REFERENCES   every waypoint, path, profile and main_task named exists; a path state is a
                      TCP pose; a relative target's chain resolves (R-M3-03, R-M3-04, R-M3-15)
      -> THE ROBOT    a frame is a link of the robot description, a tcp is the tip link, a named
                      state exists, a joint target has one value per joint (R-M3-05)
      -> PROVIDERS    each move and path is ONE M3.2 request (``model.capabilities``): its required
                      capabilities, its provider (its own, else application.providers, else the
                      registry's default) and required <= provides (R-M3-06..12)
      -> LIMITS       every motion runs with an explicit speed and acceleration (a profile or the
                      step's own); a TCP limit becomes the provider scaling of the bundle's
                      pilz_cartesian_limits (the smaller limit wins)
      -> EVENTS       a process event is anchored at a state or a segment boundary of its path (never
                      inside a segment, never asynchronous: R-M3-09), the robot stops there, a capture's
                      detector can be triggered (R-M3-16), a channel is a logical id (vNext M3.5)
      -> THE PLAN     what the bundle writes: one ``task_parameters`` entry per waypoint (the
                      target, and the move that reaches it), ``paths.yaml``, the profiles TMR loads,
                      the runner's backend and planner mode, one BehaviorTree per task; a path with
                      events is ONE path of paths.yaml run as event-free FollowPath runs with the
                      event nodes between them

Nothing is chosen at run time and nothing is degraded: an unsupported request is refused with the
step, what it asked, the provider and what to change. A problem TSA cannot prove before the robot
moves (where the robot is when a path starts, after a joint move) is a warning, and TMR refuses it
at run time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from ..catalog import declared_providers, provider_defaults, tmr_planner_registry
from ..model import CanonicalProject
from ..model.capabilities import (CapabilityRefusal, CartesianLimits, MotionRequest, Resolution,
                                  SegmentRequest, candidate_providers, check_blend_geometry,
                                  check_provider_choices, project_cartesian_limits, resolve)
from ..model.end_effector import EndEffectorCapability
from ..model.enums import AppType, Backend, MotionType, WaypointType
from ..model.motion import MotionConstraints, MotionIntent, PathContinuity
from ..model.program import (CHANNEL_KINDS, CONTROL_KINDS, AnchorWhen, DetectSource, EventKind,
                             GraspExpectation, ProcessEventSpec, RelativeFrame, TaskDataType, TaskProgram,
                             TaskWaypoint, walk_steps)
from .base import ApplicationTemplate, scene_dims_key, scene_pos_key
from .eef_tree import (canonical_device, close_nodes, continuous_everywhere, eef_attrs, open_nodes,
                       verify_grasp)

_REGISTRY = tmr_planner_registry()
# MoveWaypoint executes a percent between 1 and 100 (it clamps below 1 %, bt_nodes.cpp): a smaller
# effective scaling would run faster than asked, so it is refused instead.
MIN_MOVE_SCALING = 0.01
# FollowPath's start check (path_library.hpp PathSpec::start_tolerance), mirrored to predict it.
PATH_START_TOLERANCE = (0.005, 0.035)
# The profile name TMR always loads (its base profile): what a step that names none runs with.
TMR_BASE_PROFILE = 'default'
TREE_FILE = 'task_program.xml'
MAIN_TREE = 'MainTree'
# vNext M3.5: a capture's default blackboard key, numbered per task in execution order
CAPTURE_PREFIX = 'captures'
# the blackboard names the runner itself sets (bt_context.hpp): no capture may write them
RESERVED_BB_KEYS = ('trainit_context', 'trainit_profiles', 'ros_node', 'trainit_paths')


def task_tree_id(task: str) -> str:
    """The BehaviorTree id of a task. M3_DESIGN.md wrote ``task.<name>``; BehaviorTree.CPP 4
    refuses a '.' in a tree id ("contains forbidden character '.'", found by the M3-A launch), so
    the id is ``task_<name>`` (a task name is an identifier: one id per task, no collision with
    MainTree)."""
    return f'task_{task}'
# Resolved coordinates are rounded to 1 nm (and 1e-12 for a quaternion component): a relative
# target's float noise (0.30000000000000004 - 0.2) never reaches a generated file.
_POSITION_DIGITS, _QUATERNION_DIGITS = 9, 12


class ProgramRefusals(ValueError):
    """Every problem of a task program, at once: generation refuses before any file."""

    def __init__(self, problems: Sequence[str]):
        self.problems = tuple(problems)
        super().__init__('\n'.join(self.problems))


# --- resolved data ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """A waypoint resolved: WHERE, as the bundle writes it."""

    name: str
    type: str                                    # tcp | joint
    position: Optional[Tuple[float, ...]] = None
    orientation: Optional[Tuple[float, ...]] = None
    joints: Optional[Tuple[float, ...]] = None
    named: Optional[str] = None
    frame: str = ''                              # '' = the robot's base frame
    tcp: str = ''                                # '' = the robot's tip link
    derivation: str = ''                         # a relative target's: "w1 + ... along ..."

    def keys(self) -> Dict[str, object]:
        """The waypoint's task_parameters keys, in the order bt_params writes them."""
        out: Dict[str, object] = {'type': self.type}
        if self.type == 'tcp':
            out['position'] = list(self.position)
            out['orientation'] = list(self.orientation)
            if self.frame:
                out['frame'] = self.frame
            if self.tcp:
                out['tcp'] = self.tcp
        elif self.named:
            out['named'] = self.named
        else:
            out['joints'] = list(self.joints)
        return out


@dataclass(frozen=True)
class MovePlan:
    task: str
    step: str                                    # the step's label: '3', '3.2', '3.2.1' (1-based)
    where: str
    waypoint: str
    motion: str
    profile: str                                 # the profile TMR loads (TMR_BASE_PROFILE if none)
    profile_named: bool
    speed: float                                 # percent
    accel: float                                 # percent
    planner: Optional[str]                       # the provider row's planner MODE (ptp/free)
    aux: Optional[Tuple[float, ...]]
    via: str                                     # the via waypoint's name, or '' (inline / none)
    resolution: Resolution

    def how(self) -> Tuple:
        """What a move ADDS to its waypoint's entry: two moves with equal ``how`` share one entry."""
        return (self.motion, self.planner, self.profile, self.speed, self.accel, self.aux)


@dataclass(frozen=True)
class PlannedEvent:
    """A path event resolved onto the state the robot stops at (vNext M3.5)."""

    state: int                                   # the index of the state
    when: str                                    # after_arrival | before_departure
    event: ProcessEventSpec
    authored: str                                # the anchor as the program wrote it


@dataclass(frozen=True)
class PathPlan:
    name: str
    data: Dict[str, object]                      # the paths.yaml entry
    resolution: Resolution
    states: Tuple[str, ...]
    dense: bool
    events: Tuple[PlannedEvent, ...] = ()


@dataclass
class StepPlan:
    kind: str                                    # move | path | process | dwell | ... (STEP_KINDS)
    xml: str = ''                                # the BT node
    move: Optional[MovePlan] = None
    path: Optional[str] = None
    dwell_ms: Optional[int] = None
    nodes: List[str] = field(default_factory=list)   # vNext M3.5: several nodes (a path with events)
    event: Optional[ProcessEventSpec] = None     # a process step
    captures: List[Tuple[str, str]] = field(default_factory=list)   # (blackboard key, what) in order
    # vNext M3.6: a detect / grasp / release / tool step: (target, how, limits) as the README says it,
    # and the idiom's motions (each to an entry the idiom derives)
    summary: Optional[Tuple[str, str, str]] = None
    bound_moves: List['MovePlan'] = field(default_factory=list)
    # vNext M3.7: the step's label in its task (3, 3.2, 3.2.1), a call's task, a control step's count
    # (repeat cycles, retry attempts) and the plans of its body (repeat, retry) or alternatives (fallback)
    label: str = ''
    call: Optional[str] = None
    count: Optional[int] = None
    bodies: List[List['StepPlan']] = field(default_factory=list)


def iter_plans(steps: Sequence[StepPlan]):
    """Every step plan of a list, the plans inside control bodies included, in program order."""
    for sp in steps:
        yield sp
        for body in sp.bodies:
            yield from iter_plans(body)


def _slot(label: str) -> str:
    """A step label as part of an entry name: 3 -> 3, 3.2.1 -> 3_2_1 (a ROS parameter key)."""
    return label.replace('.', '_')


# vNext M3.7: where the robot is when a task starts -- unknown to TSA (any caller, any place)
_ENTRY = '<entry>'


@dataclass(frozen=True)
class CandidateSet:
    """A candidate set a step can read, as far as TSA knows it: how many candidates it keeps and
    whether it is projected from detections (no standoff suggested); None = not known here (a task's
    input: its callers are checked instead, see ``TaskSummary.needs``)."""

    keep: Optional[int]
    projected: Optional[bool]
    origin: str                                  # what produced it, as a message says it
    input: Optional[str] = None                  # the input of the task it is, unchanged


@dataclass
class _Flow:
    """What TSA knows at a point of a task: where the robot is (a waypoint, None, or _ENTRY) and why
    it does not know, the candidate sets and the captures held by name."""

    at: Optional[str]
    why: str = ''
    sets: Dict[str, CandidateSet] = field(default_factory=dict)
    captures: Dict[str, str] = field(default_factory=dict)

    def copy(self, at=None, why=None) -> '_Flow':
        return _Flow(self.at if at is None else at, self.why if why is None else why,
                     dict(self.sets), dict(self.captures))


@dataclass
class TaskSummary:
    """What a CALLER needs to know of a task (vNext M3.7): where it leaves the robot, the candidate
    sets it writes (its own and its callees', by name), its outputs, and what its grasps need of each
    candidate input -- (candidates kept at least, projected sets refused, where), checked at every call."""

    exit_at: Optional[str]
    exit_why: str
    writes: Tuple[str, ...]
    outputs: Dict[str, object]                   # name -> CandidateSet, or a capture's where-text
    needs: Dict[str, List[Tuple[int, bool, str]]]
    # where the task's first motion, a path, must start: checked at every call (its own warning would
    # not know the caller)
    entry_path: Optional[Tuple[str, 'PathPlan']] = None


@dataclass
class ProgramPlan:
    """Everything the bundle writes for a task program (module docstring)."""

    targets: Dict[str, Target]
    entries: Dict[str, Dict[str, object]]        # task_parameters key -> keys (ordered)
    entry_notes: Dict[str, List[str]]            # task_parameters key -> comment lines
    move_entries: Dict[Tuple[str, str], str]     # (task, step label) -> the entry its MoveWaypoint names
    paths: List[PathPlan]
    tasks: Dict[str, List[StepPlan]]
    main_task: str
    profiles: Dict[str, Dict[str, object]]       # the motion-profiles file TMR loads
    backend: str                                 # the runner's process_path_backend
    planner_mode: str                            # the runner's planner_mode (providers.free's row)
    free_provider: str
    start_tolerance: Optional[float]
    limits: Optional[CartesianLimits]
    warnings: List[str] = field(default_factory=list)
    process_controller: str = 'mock'             # vNext M3.5: 'ros' iff a process event uses a channel
    bound_entries: List[str] = field(default_factory=list)   # vNext M3.6: the entries the idioms derive
    running: List[str] = field(default_factory=list)  # vNext M3.7: main_task and the tasks it calls


# --- small geometry -------------------------------------------------------------------------------

def _rotate(q: Sequence[float], v: Sequence[float]) -> Tuple[float, float, float]:
    """``v`` rotated by the unit quaternion ``q`` = (qx, qy, qz, qw)."""
    x, y, z, w = q
    vx, vy, vz = v
    # t = 2 q_vec x v ; v' = v + w t + q_vec x t
    tx, ty, tz = 2 * (y * vz - z * vy), 2 * (z * vx - x * vz), 2 * (x * vy - y * vx)
    return (vx + w * tx + (y * tz - z * ty), vy + w * ty + (z * tx - x * tz),
            vz + w * tz + (x * ty - y * tx))


def _angle(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(p * q for p, q in zip(a, b)))
    return 2 * math.acos(min(1.0, dot))


def _fmt(v: float) -> str:
    return f'{v:g}'


# --- the robot facts --------------------------------------------------------------------------------

def _named_states(project: CanonicalProject) -> Optional[set]:
    """The named states of the planning group the bundle's SRDF will have; None = unknown."""
    robot = project.robot
    group = robot.planning_group.name
    if robot.base_moveit_config_path:
        from pathlib import Path
        from ..robotmodel.srdf_reader import read_srdf
        srdfs = sorted((Path(robot.base_moveit_config_path) / 'config').glob('*.srdf'))
        if not srdfs:
            return None
        try:
            return {s.name for s in read_srdf(srdfs[0].read_text()).states_for(group)}
        except Exception:                                      # noqa: BLE001 - unknown, not wrong
            return None
    return {s.name for s in robot.named_states if not s.group or s.group == group}


class _Frames:
    """The links of the robot description, compiled once and only when a frame other than the
    robot's base frame is named (a TF frame TSA can prove exists)."""

    def __init__(self, project: CanonicalProject):
        self.project = project
        self._links: Optional[set] = None
        self._why = ''

    def problem(self, frame: str, where: str) -> Optional[str]:
        base = self.project.robot.base_frame
        if frame == base:
            return None
        if self._links is None and not self._why:
            from ..generator.eef_contract import description_view
            view, why = description_view(self.project)
            if view is None:
                self._why = why or 'unreadable'
            else:
                self._links = set(view.links)
        if self._links is None:
            return (f"R-M3-05: {where}: frame '{frame}' cannot be checked: the robot description "
                    f'cannot be read ({self._why}); TSA proves a frame exists before the robot moves')
        if frame not in self._links:
            near = sorted(self._links)
            return (f"R-M3-05: {where}: frame '{frame}' is not a link of the robot description: TMR "
                    f'resolves a frame through TF, and TSA accepts the frames the description '
                    f"publishes (base frame '{base}', or one of: {', '.join(near)})")
        return None


# --- the planner ----------------------------------------------------------------------------------

class _Planner:
    def __init__(self, project: CanonicalProject):
        self.project = project
        self.robot = project.robot
        self.program: TaskProgram = project.application.program
        self.problems: List[str] = []
        self.warnings: List[str] = []
        self.frames = _Frames(project)
        self.waypoints = {w.name: w for w in self.program.waypoints}
        self.targets: Dict[str, Target] = {}
        self._limits: Optional[CartesianLimits] = None
        self._limits_problem: Optional[str] = None
        self._limits_read = False
        self._cycles = set()
        self._capture_keys: Dict[str, Dict[str, str]] = {}
        # vNext M3.6: the task_parameters entries the manipulation idioms derive, in program order
        self.bound: Dict[str, Tuple[Dict[str, object], List[str]]] = {}
        self._set_names: set = set()             # the candidate sets' names (blackboard prefixes)
        # vNext M3.7: task composition
        self._counters: Dict[str, List[int]] = {}   # task -> its next captures.<n>
        self._paths: Dict[str, PathPlan] = {}
        self.summaries: Dict[str, TaskSummary] = {}
        self._task: str = ''                     # the task being planned
        self._bound_task: Dict[str, str] = {}    # derived entry -> its task
        self._reads: Dict[str, set] = {}         # task -> the inputs a step reads
        self._needs: Dict[str, Dict[str, List[Tuple[int, bool, str]]]] = {}
        self._gone: Dict[str, str] = {}          # set name -> the call that replaced it (messages)
        self._cyclic: set = set()                # the tasks of a refused call cycle (checks skipped)
        self._entry_path: Dict[str, Tuple[str, PathPlan]] = {}   # task -> the path it starts with

    # -- helpers --
    def refuse(self, text: str) -> None:
        if text not in self.problems:
            self.problems.append(text)

    def limits(self) -> Optional[CartesianLimits]:
        if not self._limits_read:
            self._limits_read = True
            try:
                limits = project_cartesian_limits(self.project)
                if limits.source != 'robot.pilz_cartesian_limits':
                    # a hand-made base config's file: named portably (it reaches generated files)
                    limits = CartesianLimits(limits.max_trans_vel, limits.max_trans_acc,
                                             "the base config's pilz_cartesian_limits.yaml")
                self._limits = limits
            except CapabilityRefusal as exc:
                self._limits_problem = str(exc)
        return self._limits

    def constraints(self, profile: Optional[str], own: MotionConstraints,
                    where: str) -> Optional[MotionConstraints]:
        """The profile's limits with the step's own on top (the innermost wins)."""
        base = MotionConstraints()
        if profile is not None:
            named = self.program.motion_profiles.get(profile)
            if named is None:
                known = ', '.join(self.program.motion_profiles) or 'none defined'
                self.refuse(f"R-M3-03: {where}: profile '{profile}' is not a motion profile of this "
                            f'program (motion_profiles: {known})')
                return None
            base = MotionConstraints(**{f: getattr(named, f) for f in named.requested()})
        return base.merged(own)

    def uses_tcp_limits(self) -> bool:
        def has(c):
            return c.tcp_velocity_mps is not None or c.tcp_acceleration_mps2 is not None
        items = list(self.program.motion_profiles.values())
        for task in self.program.tasks:
            items += [s.move for s in task.steps if s.move is not None]
        for path in self.program.paths:
            items.append(path)
            items += list(path.segments or [])
        return any(has(i) for i in items)

    # -- waypoints --
    def resolve_waypoints(self) -> None:
        tip = self.robot.tip_link
        joints = len(self.robot.planning_group.joints)
        named = _named_states(self.project)
        for wp in self.program.waypoints:
            where = f"waypoint '{wp.name}'"
            if wp.type is WaypointType.JOINT:
                if wp.joints is not None and joints and len(wp.joints) != joints:
                    self.refuse(f'R-M3-05: {where}: {len(wp.joints)} joint values for the '
                                f"{joints} joints of planning group '{self.robot.planning_group.name}'"
                                f" ({', '.join(self.robot.planning_group.joints)})")
                if wp.named is not None and named is not None and wp.named not in named:
                    self.refuse(f"R-M3-05: {where}: '{wp.named}' is not a named state of planning "
                                f"group '{self.robot.planning_group.name}' (named states: "
                                f"{', '.join(sorted(named)) or 'none'})")
                continue
            if wp.tcp is not None and wp.tcp != tip:
                self.refuse(f"R-M3-05: {where}: tcp '{wp.tcp}' is not the robot's tip link '{tip}': "
                            'every motion runs for the tip link in this TSA version (another TCP is planned for '
                            'a later TSA version: UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL)')
            if wp.frame is not None:
                problem = self.frames.problem(wp.frame, where)
                if problem:
                    self.refuse(problem)
        for wp in self.program.waypoints:
            self._target(wp.name, ())

    def _target(self, name: str, chain: Tuple[str, ...]) -> Optional[Target]:
        if name in self.targets:
            return self.targets[name]
        wp: Optional[TaskWaypoint] = self.waypoints.get(name)
        if wp is None:
            return None
        if wp.type is WaypointType.JOINT:
            t = Target(name, 'joint', joints=tuple(wp.joints) if wp.joints else None,
                       named=wp.named)
        elif wp.relative is None:
            t = Target(name, 'tcp', tuple(wp.position), tuple(wp.orientation),
                       frame=(wp.frame or ''), tcp=(wp.tcp or ''))
        else:
            t = self._relative(wp, chain)
            if t is None:
                return None
        self.targets[name] = t
        return t

    def _relative(self, wp: TaskWaypoint, chain: Tuple[str, ...]) -> Optional[Target]:
        rel = wp.relative
        where = f"waypoint '{wp.name}' (relative to '{rel.to}')"
        if rel.to in chain + (wp.name,):
            cycle = frozenset(chain + (wp.name,))
            if cycle in self._cycles:
                return None
            self._cycles.add(cycle)
            self.refuse(f"R-M3-03: {where}: the relative targets form a cycle "
                        f"({' -> '.join(chain + (wp.name, rel.to))})")
            return None
        if rel.to not in self.waypoints:
            self.refuse(f"R-M3-03: {where}: no waypoint is named '{rel.to}'")
            return None
        ref = self._target(rel.to, chain + (wp.name,))
        if ref is None:
            return None
        if ref.type != 'tcp':
            self.refuse(f"R-M3-05: {where}: '{rel.to}' is a joint target: an offset needs a TCP "
                        'pose (teach the reference as a pose)')
            return None
        offset = (rel.dx, rel.dy, rel.dz)
        if rel.frame is RelativeFrame.BASE:
            base = self.robot.base_frame
            if ref.frame and ref.frame != base:
                self.refuse(f"R-M3-05: {where}: relative.frame base means the robot's base axes "
                            f"('{base}'), and '{rel.to}' is taught in frame '{ref.frame}': an offset "
                            f"along the base axes needs '{rel.to}' in '{base}' -- use relative.frame "
                            f"tool (the reference's own TCP axes), or teach '{rel.to}' in '{base}'")
                return None
            delta, axes = offset, 'the base axes'
        elif rel.frame is RelativeFrame.TARGET_LOCAL:
            # vNext M3.6: the referenced target's own local axes (for a taught target: its orientation)
            delta, axes = _rotate(ref.orientation, offset), f"{rel.to}'s local axes (target_local)"
        else:
            delta, axes = _rotate(ref.orientation, offset), f"{rel.to}'s tool axes"
        position = tuple(round(p + d, _POSITION_DIGITS) for p, d in zip(ref.position, delta))
        derivation = (f'{rel.to} + ({_fmt(rel.dx)}, {_fmt(rel.dy)}, {_fmt(rel.dz)}) m along {axes}')
        return Target(wp.name, 'tcp', position, ref.orientation, frame=ref.frame, tcp=ref.tcp,
                      derivation=derivation)

    def target(self, name: str, where: str) -> Optional[Target]:
        if name not in self.waypoints:
            known = ', '.join(self.waypoints) or 'none defined'
            self.refuse(f"R-M3-03: {where}: no waypoint is named '{name}' (waypoints: {known})")
            return None
        return self.targets.get(name)

    def aux(self, value, frame: str, where: str, what: str) -> Tuple[Optional[Tuple[float, ...]], str]:
        """A circle's auxiliary point as a position in ``frame``: a waypoint's or an inline one."""
        if value is None:
            return None, ''
        if not isinstance(value, str):
            return tuple(value), ''
        t = self.target(value, f'{where} {what}')
        if t is None:
            return None, value
        if t.type != 'tcp':
            self.refuse(f"R-M3-04: {where}: {what} '{value}' is a joint target: a circle's point is "
                        'a TCP position')
            return None, value
        if t.frame != frame:
            self.refuse(f"R-M3-04: {where}: {what} '{value}' is in frame '{t.frame or self.robot.base_frame}'"
                        f", the target in '{frame or self.robot.base_frame}': a circle's points share "
                        'one frame')
            return None, value
        return t.position, value

    # -- a request, resolved --
    def origin(self, profile: Optional[str], own: MotionConstraints, owner: str) -> str:
        """Where a request's limits come from: its profile's fields and its own (``owner``: the
        step, the path, segment i)."""
        parts = []
        if profile is not None and profile in self.program.motion_profiles:
            fields = self.program.motion_profiles[profile].requested()
            parts.append(f"{owner}'s motion profile '{profile}' ({', '.join(fields) or 'no limit'})")
        if own.requested():
            parts.append(f"{owner} itself ({', '.join(own.requested())})")
        return '; '.join(parts)

    def resolved(self, request: MotionRequest, where: str,
                 origin: str = '') -> Optional[Resolution]:
        try:
            return resolve(request, self.program.providers, self.limits())
        except CapabilityRefusal as exc:
            text = str(exc)
            if self._limits_problem and 'pilz_cartesian_limits' in text:
                text += '\n  because:  ' + self._limits_problem.splitlines()[0]
            if origin:
                text += '\n  limits:   from ' + origin
            change = _change_line(exc, request)
            if change:
                text += '\n  change:   ' + change
            self.refuse(text)
            return None

    def _scalings(self, resolution: Resolution, where: str, *, move: bool) -> Optional[Tuple]:
        v, a = resolution.velocity_scaling, resolution.acceleration_scaling
        missing = [n for n, s in (('speed', v), ('acceleration', a)) if s is None]
        if missing:
            self.refuse(f'R-M3-07: {where}: no {" and no ".join(missing)} limit: every motion runs '
                        'with an explicit speed and acceleration -- name a motion profile that sets '
                        'them, or give speed / acceleration (or tcp_velocity_mps / '
                        'tcp_acceleration_mps2) on the step')
            return None
        if move:
            for name, s in (('speed', v), ('acceleration', a)):
                if s.value < MIN_MOVE_SCALING:
                    self.refuse(f'R-M3-07: {where}: the effective {name} is a scaling of '
                                f'{s.value:.4g} ({s.derivation}), below the 1 % a move executes '
                                '(it would run faster than asked): raise the TCP limit (a '
                                'percent cannot raise it: the smaller limit wins)')
                    return None
        return v, a

    # -- moves --
    def plan_move(self, task: str, index: str, step) -> Optional[MovePlan]:
        mv = step.move
        where = f"task '{task}' step {index} (move {mv.type.value} to '{mv.to}')"
        target = self.target(mv.to, where)
        own = mv.constraints()
        inline = [f for f in ('goal_tolerance', 'planning_time_s', 'planning_attempts')
                  if getattr(own, f) is not None]
        if inline:
            self.refuse(f'R-M3-12: {where}: {", ".join(inline)} on a move step: TMR takes a move\'s '
                        'goal tolerance and planning budget from its motion profile -- set it in '
                        'a motion profile and name that profile on the step')
        constraints = self.constraints(mv.profile, own, where)
        if target is None or constraints is None or inline:
            return None
        tol = constraints.goal_tolerance
        if tol is not None and tol.joint_rad is not None:
            self.refuse(f'R-M3-12: {where}: goal_tolerance.joint_rad: TMR carries no joint goal '
                        'tolerance (position_m and orientation_rad only)')
            return None
        motion = mv.type
        if (tol is not None and motion in (MotionType.PTP, MotionType.FREE)
                and target is not None and target.type == 'joint'):
            # vNext M3.4: TMR honours a goal tolerance for a POSE goal; a joint goal has no
            # Cartesian tolerance, and TMR refuses one at run time (UNSUPPORTED_TOLERANCE)
            self.refuse(f'R-M3-12: {where}: goal_tolerance on a {motion.value} move to a joint '
                        f"target ('{mv.to}'): TMR applies a goal tolerance to a pose goal only -- "
                        'move to a TCP pose, or name a profile without goal_tolerance')
            return None
        aux, via_name = None, ''
        if motion in (MotionType.LIN, MotionType.CIRC):
            if target.type == 'joint' and target.named is None:
                self.refuse(f'R-M3-04: {where}: a {motion.value} move needs a TCP pose: '
                            f"'{mv.to}' is a joint vector (use ptp or free, or teach it as a pose "
                            'or a named state)')
                return None
        if motion is MotionType.CIRC:
            if (mv.via is None) == (mv.center is None):
                self.refuse(f'R-M3-04: {where}: a circ move needs exactly one of via or center')
                return None
            if mv.center is not None:
                self.refuse(f"R-M3-04: {where}: a circ move about a center is not executed by the "
                            'motion runtime\'s MoveWaypoint in this version: give a via point, or '
                            'use a path segment (a path takes a center)')
                return None
            frame = target.frame if target.type == 'tcp' else ''
            aux, via_name = self.aux(mv.via, frame, where, 'via')
            if aux is None:
                return None
            if target.type == 'tcp' and math.dist(aux, target.position) < 1e-4:
                self.refuse(f'R-M3-04: {where}: the via point coincides with the target')
                return None
        elif mv.via is not None or mv.center is not None:
            self.refuse(f'R-M3-04: {where}: via and center belong to a circ move')
            return None
        request = MotionRequest(where=where, intent=MotionIntent(motion.value),
                                constraints=constraints, provider=mv.provider)
        resolution = self.resolved(request, where, self.origin(mv.profile, own, 'the step'))
        if resolution is None:
            return None
        scalings = self._scalings(resolution, where, move=True)
        if scalings is None:
            return None
        planner = None
        if motion in (MotionType.PTP, MotionType.FREE):
            planner = declared_providers(_REGISTRY)[resolution.provider].get('planner_mode')
        v, a = scalings
        return MovePlan(task=task, step=index, where=where, waypoint=mv.to, motion=motion.value,
                        profile=mv.profile or TMR_BASE_PROFILE, profile_named=mv.profile is not None,
                        speed=percent(v.value), accel=percent(a.value), planner=planner,
                        aux=aux, via=via_name, resolution=resolution)

    # -- paths --
    def plan_path(self, path) -> Optional[PathPlan]:
        where = f"path '{path.name}'"
        own = path.constraints()
        constraints = self.constraints(path.profile, own, where)
        if path.tcp is not None and path.tcp != self.robot.tip_link:
            self.refuse(f"R-M3-05: {where}: tcp '{path.tcp}' is not the robot's tip link "
                        f"'{self.robot.tip_link}' (another TCP is planned for a later TSA version: "
                        'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL)')
        if path.frame is not None:
            problem = self.frames.problem(path.frame, where)
            if problem:
                self.refuse(problem)
        dense = path.source is not None
        states: List[Target] = []
        segments: List[SegmentRequest] = []
        seg_data: List[Dict[str, object]] = []
        aux_points: List[Optional[Tuple[float, ...]]] = []
        ok = constraints is not None
        if not dense:
            for i, name in enumerate(path.states):
                t = self.target(name, f'{where} state {i}')
                if t is None:
                    ok = False
                    continue
                if t.type != 'tcp':
                    self.refuse(f"R-M3-04: {where}: state {i} '{name}' is a joint target: a path "
                                'state is a TCP pose (reach a joint target with a ptp move before '
                                'the path)')
                    ok = False
                    continue
                states.append(t)
            for i, seg in enumerate(path.segments):
                swhere = f'{where} segment {i} ({path.states[i]} -> {path.states[i + 1]})'
                sc = self.constraints(seg.profile, seg.constraints(), swhere)
                if sc is None:
                    ok = False
                    continue
                segments.append(SegmentRequest(type=MotionIntent(seg.type.value), constraints=sc))
                if seg.type is MotionType.CIRC:
                    if (seg.via is None) == (seg.center is None):
                        self.refuse(f'R-M3-04: {swhere}: a circ segment needs exactly one of via '
                                    'or center')
                        ok = False
                        continue
                    # a via/center waypoint is resolved through TF like a state; an inline point is
                    # in the path's frame
                    value = seg.via if seg.via is not None else seg.center
                    aux, name = self._path_aux(value, swhere, 'via' if seg.via is not None
                                               else 'center')
                    if aux is None and name is None:
                        ok = False
                        continue
                    aux_points.append(aux)
                else:
                    if seg.via is not None or seg.center is not None:
                        self.refuse(f'R-M3-04: {swhere}: via and center belong to a circ segment')
                        ok = False
                        continue
                    aux_points.append(None)
                seg_data.append({'segment': seg, 'constraints': sc})
        if not ok:
            return None
        request = MotionRequest(where=where, intent=MotionIntent.PATH, constraints=constraints,
                                provider=path.provider, segments=tuple(segments),
                                continuity=path.continuity, dense=dense)
        origin = '; '.join(o for o in [self.origin(path.profile, path.constraints(), 'the path')] + [
            self.origin(s.profile, s.constraints(), f'segment {i}')
            for i, s in enumerate(path.segments or [])] if o)
        resolution = self.resolved(request, where, origin)
        if resolution is None:
            return None
        if not dense:
            if not self._path_geometry(path, where, states, aux_points, resolution):
                return None
            for k, seg in enumerate(resolution.segments):
                missing = [n for n, s in (('speed', seg.velocity_scaling),
                                          ('acceleration', seg.acceleration_scaling)) if s is None]
                if missing:
                    self.refuse(f'R-M3-07: {where} segment {k}: no {" and no ".join(missing)} '
                                'limit: every motion runs with an explicit speed and acceleration '
                                '-- name a motion profile on the path (or the segment) that sets '
                                'them, or give speed / acceleration there')
                    return None
        elif self._scalings(resolution, where, move=False) is None:
            return None
        data = self._path_data(path, constraints, resolution, seg_data, dense)
        events = self._path_events(path, resolution, dense)
        if events is None:
            return None
        if events:
            data['events'] = [{'at': {'state': e.state, 'when': e.when}, 'kind': e.event.kind.value}
                              for e in events]
        return PathPlan(path.name, data, resolution, tuple(path.states or ()), dense, tuple(events))

    # -- process events (vNext M3.5) --
    def _path_events(self, path, resolution: Resolution, dense: bool) -> Optional[List[PlannedEvent]]:
        """The path's events on the states the robot stops at, or None after a refusal."""
        if not path.events:
            return []
        where = f"path '{path.name}'"
        if dense:
            self.refuse(f'R-M3-09: {where}: events on a dense path are UNSUPPORTED_BY_CURRENT_CANONICAL_'
                        'MODEL: its samples are not synchronisation points -- put process steps before '
                        'and after the path step, or use a path of states')
            return None
        states = list(path.states)
        last = len(states) - 1
        out: List[PlannedEvent] = []
        ok = True
        for i, ev in enumerate(path.events):
            at = ev.at
            ewhere = f"{where} event {i} ({ev.kind.value} {at.describe()})"
            if at.state is not None:
                if isinstance(at.state, str):
                    hits = [k for k, s in enumerate(states) if s == at.state]
                    if not hits:
                        self.refuse(f"R-M3-03: {ewhere}: '{at.state}' is not a state of the path "
                                    f"(states: {', '.join(states)})")
                        ok = False
                        continue
                    if len(hits) > 1:
                        self.refuse(f"R-M3-03: {ewhere}: the path passes '{at.state}' {len(hits)} times "
                                    f"(states {', '.join(map(str, hits))}): anchor by index, 'at: {{state: "
                                    f"{hits[0]}}}'")
                        ok = False
                        continue
                    index = hits[0]
                else:
                    index = at.state
                    if index > last:
                        self.refuse(f'R-M3-03: {ewhere}: the path has no state {index} (states 0..{last}: '
                                    f"{', '.join(states)})")
                        ok = False
                        continue
                when = at.when.value
            else:
                if at.segment > last - 1:
                    self.refuse(f'R-M3-03: {ewhere}: the path has no segment {at.segment} (segments '
                                f'0..{last - 1})')
                    ok = False
                    continue
                if at.when is AnchorWhen.START:
                    index, when = at.segment, 'before_departure'
                else:
                    index, when = at.segment + 1, 'after_arrival'
            if when == 'before_departure' and index == last:
                self.refuse(f"R-M3-09: {ewhere}: before departure from the last state '{states[last]}': "
                            'the path never departs from it -- use after_arrival, or a process step after '
                            'the path')
                ok = False
                continue
            if index > 0 and resolution.segments[index - 1].blend_radius_m > 0:
                self.refuse(f"R-M3-09: {ewhere}: the robot stops at '{states[index]}' for a blocking "
                            f'event, but segment {index - 1} ({states[index - 1]} -> {states[index]}) '
                            f'blends into the next one (blend_radius_m '
                            f'{resolution.segments[index - 1].blend_radius_m:g}): set it to 0 there, or '
                            'move the event -- an event is never moved to a nearby point')
                ok = False
                continue
            if ev.kind is EventKind.CAPTURE and not self._capture_detector(ev.detector, ewhere):
                ok = False
                continue
            out.append(PlannedEvent(index, when, ev, at.describe()))
        return out if ok else None

    def _capture_detector(self, name: str, where: str) -> bool:
        """R-M3-03 / R-M3-16: the detector exists and serves the on-demand trigger a capture calls."""
        per = self.project.perception
        det = per.detector_by_name(name) if per else None
        if det is None:
            known = ', '.join(d.name for d in (per.detectors if per else [])) or 'none defined'
            self.refuse(f"R-M3-03: {where}: no detector is named '{name}' (perception.detectors: {known})")
            return False
        if det.emits_node:
            return True                          # trainit_perception's detector_node serves /detect
        ext = det.external_node
        if ext is not None and ext.detect_service:
            return True
        how = ('its external node does not declare the service (external_node.detect_service: true)'
               if ext is not None else 'no node of this bundle runs it (an external-node method without '
               'an external_node block)')
        self.refuse(f"R-M3-16: {where}: detector '{name}' cannot be triggered: a capture calls the "
                    f"detector's on-demand service /perception/{name}/detect (std_srvs/Trigger), and "
                    f'{how}')
        return False

    def _event_node(self, ev: ProcessEventSpec, key: Optional[str]) -> str:
        """The BT node of one event (a capture with its blackboard key)."""
        k = ev.kind
        if k is EventKind.CAPTURE:
            per = self.project.perception
            timeout = ev.timeout_ms or (per.detect_timeout_ms if per else 3000)
            attrs = (f'detector="{ev.detector}" trigger="true" target_frame="{self.robot.base_frame}" '
                     f'timeout_ms="{timeout}" out_key="{key}"')
            if ev.class_id:
                attrs += f' class_id="{ev.class_id}"'
            return f'<DetectObject {attrs}/>'
        if k is EventKind.DWELL:
            return f'<Sleep msec="{ev.ms}"/>'
        value = f' value="{ev.value:g}"' if k is EventKind.SET_VALUE else ''
        return f'<ProcessEvent kind="{k.value}" channel="{ev.channel}"{value}/>'

    def _capture_key(self, ev: ProcessEventSpec, task: str, counter: List[int], where: str) -> str:
        key = ev.out or f'{CAPTURE_PREFIX}.{counter[0]}'
        if ev.out is None:
            counter[0] += 1
        head = key.split('.')[0]
        clash = None
        if head in self.waypoints or any(head == n or head.startswith(n + '__') for n in self.waypoints):
            clash = f"waypoint '{head}' (its keys {head}.position, ...)"
        elif head in RESERVED_BB_KEYS or key in RESERVED_BB_KEYS:
            clash = f"the runner's own blackboard entry '{head}'"
        elif any(key == k or head == k for o in self.project.scene.objects
                 for k in (scene_dims_key(o.id), scene_pos_key(o.id))):
            clash = 'a scene parameter'
        elif head in self._set_names:
            clash = f"the candidate set '{head}'"      # vNext M3.6
        if clash:
            self.refuse(f"R-M3-02: {where}: capture key '{key}' would overwrite {clash}: name another key")
        seen = self._capture_keys.setdefault(task, {})
        if key in seen:
            self.refuse(f"R-M3-02: task '{task}': two captures write '{key}' ({seen[key]}; {where}): the "
                        'second would overwrite the first -- give each capture its own out key (a capture '
                        'without one is numbered captures.<n>, counting the task\'s captures in execution '
                        'order, and a path used twice captures twice)')
        else:
            seen[key] = where
        return key

    def _split(self, pp: PathPlan, task: str, counter: List[int], where: str) -> Tuple[List[str], List]:
        """A path with events as its BT nodes: event-free FollowPath runs with the event nodes between
        them, in execution order. At each state: the events after arrival, then those before
        departure, each group in the program's order. K event states: about K+1 runs, never a node per
        sample or per segment."""
        last = len(pp.states) - 1
        at = sorted({e.state for e in pp.events})
        nodes: List[str] = []
        captures: List[Tuple[str, str]] = []
        cur = 0
        for s in at:
            nodes.append(f'<FollowPath path="{pp.name}" from="{cur}" to="{s}"/>')
            group = ([e for e in pp.events if e.state == s and e.when == 'after_arrival'] +
                     [e for e in pp.events if e.state == s and e.when == 'before_departure'])
            for e in group:
                key = None
                if e.event.kind is EventKind.CAPTURE:
                    key = self._capture_key(e.event, task, counter, f"{where} event at '{pp.states[s]}'")
                    captures.append((key, f"{e.event.describe()}, {e.authored} of path '{pp.name}'"))
                nodes.append(self._event_node(e.event, key))
            cur = s
        if cur < last:
            nodes.append(f'<FollowPath path="{pp.name}" from="{cur}" to="{last}"/>')
        return nodes, captures

    def _path_aux(self, value, where, what):
        if isinstance(value, str):
            t = self.target(value, f'{where} {what}')
            if t is None:
                return None, None
            if t.type != 'tcp':
                self.refuse(f"R-M3-04: {where}: {what} '{value}' is a joint target: a circle's point "
                            'is a TCP position')
                return None, None
            return t.position, value
        return tuple(value), ''

    def _path_geometry(self, path, where, states, aux_points, resolution) -> bool:
        """R-M3-04/08 geometry when every state is a taught pose in ONE frame (TMR's rules,
        path_library.cpp checkPathGeometry); otherwise TMR checks it when the path runs."""
        frames = {t.frame for t in states}
        aux_frames = {self.targets[v].frame for v in [
            s.via if s.via is not None else s.center for s in path.segments
            if s.type is MotionType.CIRC and isinstance(s.via if s.via is not None else s.center,
                                                        str)]}
        inline_aux = any(s.type is MotionType.CIRC and not isinstance(
            s.via if s.via is not None else s.center, str) for s in path.segments)
        if len(frames) != 1 or (aux_frames - frames) or (inline_aux and frames != {path.frame or ''}):
            self.warnings.append(f'{where}: its states are in different frames, so TMR checks the '
                                 'path geometry (segment lengths, circle points, blend radii) when '
                                 'the path runs, after resolving every frame through TF')
            return True
        positions = [t.position for t in states]
        for i in range(len(positions) - 1):
            if math.dist(positions[i], positions[i + 1]) < 1e-6 and \
                    _angle(states[i].orientation, states[i + 1].orientation) < 1e-6:
                self.refuse(f"R-M3-04: {where}: states {i} and {i + 1} ('{path.states[i]}', "
                            f"'{path.states[i + 1]}') are the same pose: a zero-length segment "
                            'cannot be planned')
                return False
        for i, seg in enumerate(path.segments):
            aux = aux_points[i]
            if aux is None:
                continue
            a, b = positions[i], positions[i + 1]
            swhere = f'{where} segment {i} ({path.states[i]} -> {path.states[i + 1]})'
            if math.dist(a, b) < 1e-4:
                self.refuse(f'R-M3-04: {swhere}: the arc\'s two ends are the same point: a full '
                            'circle is not one circ segment')
                return False
            if seg.via is not None:
                if math.dist(aux, a) < 1e-4 or math.dist(aux, b) < 1e-4:
                    self.refuse(f'R-M3-04: {swhere}: the via point coincides with an end of the arc')
                    return False
                if _sin_between(_sub(aux, a), _sub(b, a)) < 1e-3:
                    self.refuse(f'R-M3-04: {swhere}: the via point is collinear with the arc\'s '
                                'ends: no circle passes through them')
                    return False
            else:
                r1, r2 = math.dist(a, aux), math.dist(b, aux)
                if abs(r1 - r2) > 1e-3:
                    self.refuse(f'R-M3-04: {swhere}: the center is not equidistant from the arc\'s '
                                f'ends ({r1:.4g} m vs {r2:.4g} m)')
                    return False
                if _sin_between(_sub(a, aux), _sub(b, aux)) < 1e-3:
                    self.refuse(f'R-M3-04: {swhere}: the ends and the center are collinear (a half '
                                'circle): the arc\'s plane is undefined; use a via point')
                    return False
        try:
            check_blend_geometry(where, positions, [s.blend_radius_m for s in resolution.segments])
        except CapabilityRefusal as exc:
            self.refuse(str(exc))
            return False
        return True

    def _path_data(self, path, constraints: MotionConstraints, resolution: Resolution, seg_data,
                   dense: bool) -> Dict[str, object]:
        data: Dict[str, object] = {'provider': resolution.provider}
        data['continuity'] = (path.continuity or PathContinuity.STOP).value
        data['profile'] = path.profile or TMR_BASE_PROFILE
        data['velocity_scaling'] = resolution.velocity_scaling.value \
            if resolution.velocity_scaling else None
        data['acceleration_scaling'] = resolution.acceleration_scaling.value \
            if resolution.acceleration_scaling else None
        if data['velocity_scaling'] is None:
            data.pop('velocity_scaling')
        if data['acceleration_scaling'] is None:
            data.pop('acceleration_scaling')
        tol = constraints.goal_tolerance
        if tol is not None:
            data['goal_tolerance'] = {'position_m': tol.position_m,
                                      'orientation_rad': tol.orientation_rad}
        if path.frame:
            data['frame'] = path.frame
        if path.tcp:
            data['tcp'] = path.tcp
        if dense:
            src = path.source
            gen = {k: v for k, v in json_items(src.generator) if v is not None}
            data['source'] = {'generator': gen, 'orientation': list(src.orientation)}
            return data
        data['states'] = [{'waypoint': s} for s in path.states]
        segs = []
        for i, (item, seg_res) in enumerate(zip(seg_data, resolution.segments)):
            seg = item['segment']
            row: Dict[str, object] = {'type': seg.type.value}
            if seg.type is MotionType.CIRC:
                key = 'via' if seg.via is not None else 'center'
                value = seg.via if seg.via is not None else seg.center
                row[key] = {'waypoint': value} if isinstance(value, str) else \
                    {'position': list(value)}
            own = seg.profile is not None or any(
                f in _SCALING_FIELDS for f in seg.constraints().requested())
            if seg.profile is not None:
                row['profile'] = seg.profile
            if own:
                row['velocity_scaling'] = seg_res.velocity_scaling.value
                row['acceleration_scaling'] = seg_res.acceleration_scaling.value
            if seg_res.blend_radius_m > 0:
                row['blend_radius_m'] = seg_res.blend_radius_m
            segs.append(row)
        data['segments'] = segs
        return data

    # -- manipulation (vNext M3.6, ADR-0026, M3_DESIGN §10) --
    def _device_only(self, where: str, eef: Optional[str]):
        """The contract's device (ADR-0022): the only one by default; None after a refusal."""
        dev = canonical_device(self.project)
        if dev is None:
            self.refuse(f'R-M3-13: {where}: a manipulation or tool step commands the end effector through '
                        'its contract (ADR-0022), and this project has none: adopt one in Step 7, End effector '
                        '(robot.gripper.contract)')
            return None
        if eef is not None and eef != dev.id:
            self.refuse(f"R-M3-13: {where}: no end effector is named '{eef}' (the contract's device is "
                        f"'{dev.id}')")
            return None
        return dev

    def _device(self, where: str, eef: Optional[str], site: Optional[str], need_tip: bool = True):
        """The device and grasp site a manipulation step uses (ADR-0022): the contract's only device
        by default, its only site by default; a step that MOVES to the site needs its TCP to be the
        robot's tip link (R-M3-05, R-M3-13), a tool close / open does not (``need_tip``). (None, None)
        after a refusal."""
        dev = self._device_only(where, eef)
        if dev is None:
            return None, None
        if site is None:
            s = dev.default_site()
            if s is None:
                self.refuse(f"R-M3-13: {where}: device '{dev.id}' has several grasp sites "
                            f"({', '.join(x.id for x in dev.sites)}): name the step's 'site'")
                return None, None
        else:
            s = dev.site(site)
            if s is None:
                self.refuse(f"R-M3-13: {where}: device '{dev.id}' has no grasp site '{site}' (sites: "
                            f"{', '.join(x.id for x in dev.sites)})")
                return None, None
        tip = self.robot.tip_link
        if need_tip and s.tcp != tip:
            self.refuse(f"R-M3-05: {where}: grasp site '{s.id}' holds the object at TCP '{s.tcp}', and every "
                        f"motion of this version runs for the robot's tip link '{tip}': a grasp at another "
                        'TCP is UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL (planned with the multi-TCP motion '
                        "work) -- make the site's TCP the tip link")
            return None, None
        return dev, s

    def _bound_move(self, task: str, index: str, where: str, entry: str, motion: MotionType,
                    profile: str) -> Optional[MovePlan]:
        """ONE motion of a manipulation idiom, resolved like a move step (its profile's limits, its
        provider, an explicit speed and acceleration) -- to an entry the idiom derives."""
        constraints = self.constraints(profile, MotionConstraints(), where)
        if constraints is None:
            return None
        tol = constraints.goal_tolerance
        if tol is not None and tol.joint_rad is not None:
            self.refuse(f'R-M3-12: {where}: goal_tolerance.joint_rad: TMR carries no joint goal '
                        'tolerance (position_m and orientation_rad only)')
            return None
        request = MotionRequest(where=where, intent=MotionIntent(motion.value), constraints=constraints)
        resolution = self.resolved(request, where, self.origin(profile, MotionConstraints(), 'the step'))
        if resolution is None:
            return None
        scalings = self._scalings(resolution, where, move=True)
        if scalings is None:
            return None
        planner = None
        if motion in (MotionType.PTP, MotionType.FREE):
            planner = declared_providers(_REGISTRY)[resolution.provider].get('planner_mode')
        v, a = scalings
        return MovePlan(task=task, step=index, where=where, waypoint=entry, motion=motion.value,
                        profile=profile, profile_named=True, speed=percent(v.value),
                        accel=percent(a.value), planner=planner, aux=None, via='', resolution=resolution)

    def _bind(self, name: str, keys: Dict[str, object], mp: MovePlan, note: str) -> None:
        """A task_parameters entry an idiom derives: its target keys and the move that reaches it."""
        entry = dict(keys)
        entry.update(_move_keys(mp))
        self.bound[name] = (entry, [note, move_text(mp)] + scaling_text(mp.resolution))
        self._bound_task[name] = self._task

    def _grasp_needs(self, where: str, name: str, held: CandidateSet, index: int, no_standoff: bool) -> bool:
        """A grasp on candidate ``index`` of a set, without a standoff of its own when ``no_standoff``:
        refused when the set provably cannot serve it; recorded on the task's input when the set IS an
        input (each caller is checked); a warning when TSA cannot know (TMR refuses it at run time)."""
        if held.input is not None:
            self._reads.setdefault(self._task, set()).add(held.input)
        if held.keep is not None and index >= held.keep:
            self.refuse(f"R-M3-13: {where}: candidate {index} of '{name}': {held.origin} keeps "
                        f'{held.keep} candidate(s) (raise its keep)')
            return False
        if held.projected and no_standoff:
            self.refuse(f"R-M3-13: {where}: candidates projected from detections suggest no standoff: "
                        "give the grasp step's 'standoff_m' (the PreGrasp distance back along the "
                        'approach)')
            return False
        if held.input is not None:
            if held.keep is None or (held.projected is None and no_standoff):
                self._needs.setdefault(self._task, {}).setdefault(held.input, []).append(
                    (index + 1, no_standoff, where))
        elif held.keep is None or (held.projected is None and no_standoff):
            self.warnings.append(f"{where}: TSA cannot prove that '{name}' ({held.origin}) holds candidate "
                                 f'{index}' + (' with a standoff' if no_standoff else '')
                                 + ': TMR checks it when the step runs (NO_CANDIDATE, INVALID_STANDOFF)')
        return True

    def _set_key(self, key: str, where: str) -> bool:
        """A candidate set's name is a blackboard prefix: it must not overwrite a waypoint, the
        runner's entries, the captures or a scene parameter."""
        clash = None
        captured = {k.split('.')[0] for keys in self._capture_keys.values() for k in keys}
        if '__' in key:
            clash = "the entries TSA derives ('__' is reserved for them)"
        elif key in self.waypoints or any(key == n or key.startswith(n + '__') for n in self.waypoints):
            clash = f"waypoint '{key}'"
        elif key in RESERVED_BB_KEYS or key == CAPTURE_PREFIX:
            clash = f"the blackboard entry '{key}'"
        elif key in captured:
            clash = f"the capture '{key}'"
        elif any(key == k for o in self.project.scene.objects
                 for k in (scene_dims_key(o.id), scene_pos_key(o.id))):
            clash = 'a scene parameter'
        if clash:
            self.refuse(f"R-M3-02: {where}: candidate set '{key}' would overwrite {clash}: name another 'out'")
            return False
        self._set_names.add(key)
        return True

    def plan_detect(self, task: str, index: str, step, sets: Dict[str, 'CandidateSet']) -> Optional[StepPlan]:
        d = step.detect
        where = f"task '{task}' step {index} (detect from '{d.detector}')"
        if not self._set_key(d.out, where):
            return None
        # the set exists for the steps after it even when this step is refused below: a grasp on it
        # is judged on its own, never refused again for a set its detect step does define
        sets.pop(d.out, None)
        sets[d.out] = CandidateSet(d.keep, d.source is DetectSource.DETECTIONS,  # the latest set: the last key
                                   'its detect step')
        per = self.project.perception
        det = per.detector_by_name(d.detector) if per else None
        if det is None:
            known = ', '.join(x.name for x in (per.detectors if per else [])) or 'none defined'
            self.refuse(f"R-M3-03: {where}: no detector is named '{d.detector}' (perception.detectors: "
                        f'{known})')
            return None
        if not det.emits_node and det.external_node is None:
            self.refuse(f"R-M3-03: {where}: no node of this bundle runs detector '{d.detector}' (an "
                        'external-node method without an external_node block)')
            return None
        if d.source is DetectSource.CANDIDATES and det.emits_node:
            self.refuse(f"R-M3-03: {where}: detector '{d.detector}' is trainit_perception's detector node, "
                        'which publishes DETECTIONS, not grasp candidates: use source: detections '
                        '(projected onto the candidate contract), or declare the grasp producer as an '
                        'external node')
            return None
        timeout = d.timeout_ms or (per.detect_timeout_ms if per else 3000)
        attrs = (f'source="{d.source.value}" detector="{d.detector}" '
                 f'target_frame="{self.robot.base_frame}" timeout_ms="{timeout}" keep="{d.keep}" '
                 f'out_key="{d.out}"')
        if d.class_id:
            attrs += f' class_id="{d.class_id}"'
        if d.approach is not None:
            attrs += ' default_approach="' + ';'.join(f'{v:.12g}' for v in d.approach) + '"'
        if d.source is DetectSource.CANDIDATES:
            how = f"the grasp candidates of '{d.detector}'"
        else:
            how = (f"the detections of '{d.detector}', projected onto grasp candidates (approach = "
                   f'-normal; without a normal {_vec_text(d.approach)})')
        how += f': the first {d.keep} in its order -> candidate set `{d.out}`'
        return StepPlan('detect', f'<DetectCandidates {attrs}/>', summary=('--', how, '--'))

    def plan_grasp(self, task: str, index: str, step, sets: Dict[str, 'CandidateSet']) -> Optional[StepPlan]:
        g = step.grasp
        where = f"task '{task}' step {index} ({g.describe()})"
        dev, site = self._device(where, g.eef, g.site)
        moves = [self._bound_move(task, index, f'{where} PreGrasp', f'{task}__s{_slot(index)}_pregrasp',
                                  MotionType.PTP, g.profile),
                 self._bound_move(task, index, f'{where} Approach', f'{task}__s{_slot(index)}_grasp',
                                  MotionType.LIN, g.approach_profile),
                 self._bound_move(task, index, f'{where} Retract', f'{task}__s{_slot(index)}_retract',
                                  MotionType.LIN, g.approach_profile)]
        if dev is None or None in moves:
            return None
        pre, grasp, retract = (m.waypoint for m in moves)
        r = g.retract
        if g.at is None:
            name = g.candidates if g.candidates is not None else (list(sets)[-1] if sets else None)
            held = sets.get(name) if name is not None else None
            if held is None:
                gone = self._gone.get(name) if name is not None else None
                what = (f"candidate set '{g.candidates}' (a detect step with out: {g.candidates}"
                        + ('' if gone else ', an input of the task, or an output of a task called earlier')
                        + ')') if g.candidates is not None else \
                    'a candidate set (add a detect step before the grasp)'
                self.refuse(f"R-M3-13: {where}: no detect step earlier in task '{task}' produces {what}"
                            + (f' -- {gone}' if gone else ''))
                return None
            if not self._grasp_needs(where, name, held, g.index, g.standoff_m is None):
                return None
            ref = self.target(g.orientation_from, f'{where} orientation_from')
            if ref is None:
                return None
            if ref.type != 'tcp' or (ref.frame and ref.frame != self.robot.base_frame):
                self.refuse(f"R-M3-05: {where}: orientation_from '{g.orientation_from}' must be a TCP pose "
                            f"in the robot's base frame '{self.robot.base_frame}' (the frame the "
                            'candidates are resolved in): align_approach tilts its orientation onto the '
                            'approach')
                return None
            attrs = (f'from="{name}" index="{g.index}" grasp="{grasp}" pregrasp="{pre}" '
                     f'retract="{retract}"')
            if g.standoff_m is not None:
                attrs += f' standoff_m="{_fmt(g.standoff_m)}"'
            attrs += (f' retract_along_approach="{_fmt(r.along_approach)}"' if r.along_approach is not None
                      else f' retract_dz="{_fmt(r.dz)}"')
            attrs += (f' orientation="align_approach" orientation_from="{g.orientation_from}" '
                      f'eef="{dev.id}" site="{site.id}"')
            nodes = [f'<SetWaypointFromCandidate {attrs}/>']
            src = f"candidate {g.index} of '{name}'"
            for mp, role in zip(moves, ('PreGrasp', 'the grasp pose (Approach)', 'Retract')):
                self._bind(mp.waypoint, {'type': 'tcp'}, mp,
                           f"{role} of task '{task}' step {index}: set at run time from {src} "
                           f"(SetWaypointFromCandidate, orientation align_approach on '{g.orientation_from}')")
            target_text = f'{src} (TCP orientation: align_approach on `{g.orientation_from}`)'
        else:
            if not self._taught(where, g.at, g.standoff_m, r, ('PreGrasp', 'the grasp pose', 'Retract'),
                                moves, task, index):
                return None
            nodes = []
            z = _rotate(self.targets[g.at].orientation, (0.0, 0.0, 1.0))
            target_text = f'`{g.at}` (taught; its approach is its tool +Z = {_vec_text(_round_vec(z))})'
        nodes += [f'<MoveWaypoint waypoint="{pre}"/>', f'<MoveWaypoint waypoint="{grasp}"/>']
        nodes += close_nodes(dev, site.id)
        nodes.append(f'<MoveWaypoint waypoint="{retract}"/>')
        how = (f'PreGrasp PTP ({g.profile}) {_standoff_text(g.standoff_m)} -> Approach LIN '
               f"({g.approach_profile}) -> close site '{site.id}'"
               + (' -> verify held' if dev.require_grasp_confirmation else '')
               + f' -> Retract LIN {_retract_text(r)}')
        return StepPlan('grasp', nodes=nodes, summary=(target_text, how, _limits_of(moves)),
                        bound_moves=moves)

    def _taught(self, where, at, standoff, r, roles, moves, task, index) -> bool:
        """A taught target's PreGrasp (or PreRelease) and Retract, resolved now: along its tool +Z --
        its approach -- in its target_local axes; a ``dz`` retract goes up its frame's +Z."""
        t = self.target(at, f'{where} at')
        if t is None:
            return False
        if t.type != 'tcp':
            self.refuse(f"R-M3-13: {where}: '{at}' is a joint target: a grasp or a release happens at a "
                        'TCP pose (teach it as a pose)')
            return False
        z = _rotate(t.orientation, (0.0, 0.0, 1.0))
        pre = tuple(round(p - standoff * a, _POSITION_DIGITS) for p, a in zip(t.position, z))
        if r.along_approach is not None:
            back = tuple(round(p - r.along_approach * a, _POSITION_DIGITS) for p, a in zip(t.position, z))
            back_text = f"'{at}' - {_fmt(r.along_approach)} m along its tool +Z (target_local)"
        else:
            back = (t.position[0], t.position[1], round(t.position[2] + r.dz, _POSITION_DIGITS))
            back_text = f"'{at}' + {_fmt(r.dz)} m up its frame's +Z"
        texts = (f"'{at}' - {_fmt(standoff)} m along its tool +Z (target_local)", f"'{at}'", back_text)
        for mp, position, role, text in zip(moves, (pre, t.position, back), roles, texts):
            keys: Dict[str, object] = {'type': 'tcp', 'position': list(position),
                                       'orientation': list(t.orientation)}
            if t.frame:
                keys['frame'] = t.frame
            self._bind(mp.waypoint, keys, mp, f"{role} of task '{task}' step {index}: {text} (resolved by TSA)")
        return True

    def plan_release(self, task: str, index: str, step) -> Optional[StepPlan]:
        rel = step.release
        where = f"task '{task}' step {index} ({rel.describe()})"
        dev, site = self._device(where, rel.eef, rel.site)
        moves = [self._bound_move(task, index, f'{where} PreRelease', f'{task}__s{_slot(index)}_prerelease',
                                  MotionType.PTP, rel.profile),
                 self._bound_move(task, index, f'{where} Approach', f'{task}__s{_slot(index)}_release',
                                  MotionType.LIN, rel.approach_profile),
                 self._bound_move(task, index, f'{where} Retract', f'{task}__s{_slot(index)}_retract',
                                  MotionType.LIN, rel.approach_profile)]
        if dev is None or None in moves:
            return None
        if not self._taught(where, rel.at, rel.standoff_m, rel.retract,
                            ('PreRelease', 'the release pose', 'Retract'), moves, task, index):
            return None
        pre, at, retract = (m.waypoint for m in moves)
        nodes = [f'<MoveWaypoint waypoint="{pre}"/>', f'<MoveWaypoint waypoint="{at}"/>']
        held_first = bool(dev.require_grasp_confirmation) and continuous_everywhere(self.project, dev)
        nodes += open_nodes(self.project, dev, site.id, held_guard=True)
        nodes.append(f'<MoveWaypoint waypoint="{retract}"/>')
        how = (f'PreRelease PTP ({rel.profile}) {_standoff_text(rel.standoff_m)} -> Approach LIN '
               f'({rel.approach_profile})' + (' -> verify held' if held_first else '')
               + f" -> open site '{site.id}'"
               + (' -> verify released' if dev.require_release_confirmation else '')
               + f' -> Retract LIN {_retract_text(rel.retract)}')
        z = _rotate(self.targets[rel.at].orientation, (0.0, 0.0, 1.0))
        target = f'`{rel.at}` (taught; its approach is its tool +Z = {_vec_text(_round_vec(z))})'
        return StepPlan('release', nodes=nodes, summary=(target, how, _limits_of(moves)),
                        bound_moves=moves)

    def plan_tool(self, task: str, index: str, step) -> Optional[StepPlan]:
        t = step.tool
        if t.close is not None or t.open is not None:
            return self._plan_grip(task, index, 'close' if t.close is not None else 'open', t.close or t.open)
        if t.actuate is not None:
            a = t.actuate
            where = f"task '{task}' step {index} (tool actuate '{a.actuator}' -> '{a.target}')"
            dev = self._device_only(where, a.eef)
            if dev is None:
                return None
            act = dev.actuator(a.actuator)
            if act is None:
                self.refuse(f"R-M3-13: {where}: device '{dev.id}' has no actuator '{a.actuator}' "
                            f"(actuators: {', '.join(x.id for x in dev.actuators)})")
                return None
            if a.target not in act.target_names():
                self.refuse(f"R-M3-13: {where}: actuator '{a.actuator}' has no named target '{a.target}' "
                            f"(targets: {', '.join(act.target_names())})")
                return None
            for s in dev.sites:
                if s.grip == act.id and a.target in (s.engage, s.release):
                    which = 'engage' if a.target == s.engage else 'release'
                    self.refuse(f"R-M3-13: {where}: '{a.target}' is the {which} target of grip actuator "
                                f"'{act.id}' (site '{s.id}'): a grip is closed and opened by the grasp and "
                                "release steps or 'tool: close / open' (ADR-0022 R11)")
                    return None
            return StepPlan('tool', f'<CommandActuator {eef_attrs(dev)} actuator="{a.actuator}" '
                                    f'target="{a.target}"/>',
                            summary=('--', f"actuator '{a.actuator}' -> named target '{a.target}'", '--'))
        v = t.verify
        where = f"task '{task}' step {index} (tool verify {v.expect.value})"
        dev = self._device_only(where, v.eef)
        if dev is None:
            return None
        if v.site is not None and dev.site(v.site) is None:
            self.refuse(f"R-M3-13: {where}: device '{dev.id}' has no grasp site '{v.site}' (sites: "
                        f"{', '.join(x.id for x in dev.sites)})")
            return None
        token = (EndEffectorCapability.GRASP_DETECTION if v.expect is GraspExpectation.HELD
                 else EndEffectorCapability.RELEASE_DETECTION).value
        modes = [Backend(m) for m in self.project.deployment.modes]
        if not dev.provides_everywhere(token, modes):
            self.refuse(f"R-M3-13: {where}: device '{dev.id}' cannot observe the grasp ({token}) on every "
                        f"deployed backend ({', '.join(m.value for m in modes)}): this verification would "
                        'always fail -- bind an adapter that provides it, or remove the step')
            return None
        return StepPlan('tool', verify_grasp(dev, v.site, v.expect.value),
                        summary=('--', f'verify {v.expect.value}' + (f" at site '{v.site}'" if v.site else ''),
                                 '--'))

    def _plan_grip(self, task: str, index: str, which: str, g) -> Optional[StepPlan]:
        """vNext I0 (owner decision D-E1, ADR-0024 amendment I0): a grasp's / a release's actuation where the
        robot is -- the same nodes and verification policy, no motion."""
        where = f"task '{task}' step {index} (tool {which}" + (f" at site '{g.site}'" if g.site else '') + ')'
        dev, site = self._device(where, g.eef, g.site, need_tip=False)
        if dev is None:
            return None
        if which == 'close':
            nodes = close_nodes(dev, site.id)
            how = f"close site '{site.id}' (no motion)" + (' -> verify held' if dev.require_grasp_confirmation else '')
        else:
            # no held guard: a plain open is valid with nothing held (a cycle start); the guard belongs to the
            # release idiom, or to an explicit 'tool: {verify: {expect: held}}' step (architecture review M1)
            nodes = open_nodes(self.project, dev, site.id, held_guard=False)
            how = (f"open site '{site.id}' (no motion)"
                   + (' -> verify released' if dev.require_release_confirmation else ''))
        return StepPlan('tool', nodes=nodes, summary=('--', how, '--'))

    # -- the program --
    def plan(self) -> Optional[ProgramPlan]:
        program = self.program
        if program.providers is not None:
            try:
                check_provider_choices(program.providers)
            except CapabilityRefusal as exc:
                self.refuse(str(exc))
        if self.uses_tcp_limits():
            self.limits()
        for name, prof in program.motion_profiles.items():
            if prof.goal_tolerance is not None and prof.goal_tolerance.joint_rad is not None:
                self.refuse(f"R-M3-12: motion profile '{name}': goal_tolerance.joint_rad: the motion "
                            'runtime carries no joint goal tolerance (position_m and orientation_rad '
                            'only)')
        self.resolve_waypoints()

        paths: Dict[str, PathPlan] = {}
        for path in program.paths:
            planned = self.plan_path(path)
            if planned is not None:
                paths[path.name] = planned
        self._paths = paths

        # vNext M3.7: a called task is planned before its callers (its summary is what a call
        # checks); without calls this is the declaration order
        planned_steps: Dict[str, List[StepPlan]] = {}
        planned_moves: Dict[str, List[MovePlan]] = {}
        for name in self._call_order():
            planned_steps[name], planned_moves[name] = self.plan_task(self._tasks[name])
        tasks = {t.name: planned_steps[t.name] for t in program.tasks}
        moves = [mp for t in program.tasks for mp in planned_moves[t.name]]
        rank = {t.name: i for i, t in enumerate(program.tasks)}
        self.bound = dict(sorted(self.bound.items(), key=lambda kv: rank[self._bound_task[kv[0]]]))

        running = self._running()
        self._shared_captures(running)
        unreached = [t.name for t in program.tasks if t.name not in running]
        if unreached:
            self.warnings.append(f"task {', '.join(repr(t) for t in unreached)} never runs: neither "
                                 f"main_task '{program.main_task}' nor a task it runs calls it")
        if self.problems:
            raise ProgramRefusals(self.problems)
        return self._assemble(paths, tasks, moves, running)

    # -- composition (vNext M3.7) --
    @property
    def _tasks(self):
        return {t.name: t for t in self.program.tasks}

    def _call_order(self) -> List[str]:
        """The tasks, every called task before its callers. Refuses a call to an unknown task and
        every call cycle (R-M3-15), naming the chain; a cycle's calls are then left unchecked."""
        tasks = self._tasks
        graph: Dict[str, List[Tuple[str, str]]] = {}
        for t in self.program.tasks:
            graph[t.name] = []
            for label, callee in t.calls():
                if callee not in tasks:
                    self.refuse(f"R-M3-15: task '{t.name}' step {label} (task '{callee}'): no task is named "
                                f"'{callee}' (tasks: {', '.join(tasks)})")
                elif callee not in [c for _, c in graph[t.name]]:
                    graph[t.name].append((label, callee))
        order: List[str] = []
        state: Dict[str, str] = {}
        cycles = set()

        def visit(name: str, stack: List[Tuple[str, str]]) -> None:
            state[name] = 'open'
            for label, callee in graph[name]:
                if state.get(callee) == 'open':
                    start = next(i for i, (n, _) in enumerate(stack) if n == callee) if callee != name else len(stack)
                    chain = stack[start:] + [(name, label)]
                    names = [n for n, _ in chain]
                    first = min(range(len(names)), key=lambda i: list(tasks).index(names[i]))
                    key = tuple(names[first:] + names[:first])
                    self._cyclic.update(names)
                    if key not in cycles:
                        cycles.add(key)
                        rot = chain[first:] + chain[:first]
                        calls = '; '.join(f"task '{n}' step {lab} calls '{rot[(i + 1) % len(rot)][0]}'"
                                          for i, (n, lab) in enumerate(rot))
                        self.refuse(f"R-M3-15: task call cycle: {' -> '.join(list(key) + [key[0]])} ({calls}): "
                                    'the calls of a task program form no cycle -- a task never calls itself, '
                                    "directly or through other tasks; a repetition is a 'repeat' or a 'retry' step")
                elif callee not in state:
                    visit(callee, stack + [(name, label)])
            state[name] = 'done'
            order.append(name)

        for t in self.program.tasks:
            if t.name not in state:
                visit(t.name, [])
        return order

    def _running(self) -> List[str]:
        """main_task and every task it calls, directly or not, in declaration order."""
        tasks, seen, todo = self._tasks, set(), [self.program.main_task]
        while todo:
            name = todo.pop()
            if name in seen or name not in tasks:
                continue
            seen.add(name)
            todo += [c for _, c in tasks[name].calls()]
        return [t.name for t in self.program.tasks if t.name in seen]

    def _shared_captures(self, running: List[str]) -> None:
        """Two tasks that run in one program write one blackboard: a capture key written by both
        would overwrite the other's (R-M3-02, as two captures of one task)."""
        owner: Dict[str, Tuple[str, str]] = {}
        for task in running:
            for key, where in self._capture_keys.get(task, {}).items():
                if key in owner and owner[key][0] != task:
                    self.refuse(f"R-M3-02: capture key '{key}' is written by {owner[key][1]} and by {where}: both "
                                "tasks run in this program and share the program's data, so one would overwrite "
                                "the other -- give one of the captures its own out key")
                owner.setdefault(key, (task, where))

    def _exit(self, steps, at):
        """Where a list of steps leaves the robot when it ends, from where it starts (the model alone)."""
        for step in steps:
            kind = step.kind()
            if kind == 'move':
                at = step.move.to
            elif kind == 'path':
                p = next((x for x in self.program.paths if x.name == step.path), None)
                at = p.states[-1] if p is not None and p.states else None
            elif kind in ('grasp', 'release'):
                at = None
            elif kind == 'task':
                summary = self.summaries.get(step.task)
                at = None if summary is None else (at if summary.exit_at == _ENTRY else summary.exit_at)
            elif kind in ('repeat', 'retry'):
                at = self._exit(step.bodies()[0], at)
            elif kind == 'fallback':
                ends = {self._exit(body, at) for body in step.bodies()}
                at = ends.pop() if len(ends) == 1 else None
        return at

    def plan_task(self, task) -> Tuple[List[StepPlan], List[MovePlan]]:
        """One task: its steps in program order (control bodies included), and the summary its callers
        check."""
        self._task = task.name
        self._gone = {}
        flow = _Flow(_ENTRY)
        for d in task.inputs:
            where = f"task '{task.name}' input '{d.name}'"
            if task.name == self.program.main_task:
                self.refuse(f"R-M3-15: {where} ({d.type.value}): main_task '{task.name}' is started by the "
                            'runner, which provides no data: only a called task takes inputs (its caller '
                            'provides them)')
            if d.type is TaskDataType.CANDIDATES:
                if self._set_key(d.name, where):
                    flow.sets[d.name] = CandidateSet(None, None, f"input '{d.name}' of task '{task.name}'",
                                                     input=d.name)
            else:
                flow.captures[d.name] = where
        moves: List[MovePlan] = []
        steps = self._plan_steps(task, task.steps, '', flow, moves)
        read = self._reads.get(task.name, set())
        cyclic = task.name in self._cyclic          # its calls were not checked: no follow-on refusal
        for d in task.inputs:
            if d.name in read or cyclic:
                continue
            why = ('no step of this version reads a capture: a capture is an OUTPUT (its data stays on the '
                   'blackboard for the steps after it)' if d.type is TaskDataType.CAPTURE else
                   "a grasp step reads a candidate set ('candidates: <name>'), and a call passes it to the "
                   'called task\'s input of that name')
            self.refuse(f"R-M3-03: task '{task.name}' input '{d.name}' ({d.type.value}): no step of the task reads "
                        f'it -- {why}; remove it, or name it as its reader names it')
        outputs: Dict[str, object] = {}
        for d in task.outputs:
            held = (flow.sets if d.type is TaskDataType.CANDIDATES else flow.captures).get(d.name)
            if held is None and cyclic:
                continue
            if held is None:
                how = ("a detect step with out: {n}, or a called task's output" if d.type is TaskDataType.CANDIDATES
                       else 'a capture with out: {n} (a capture without one is numbered captures.<n>), or a called '
                            "task's output").format(n=d.name)
                self.refuse(f"R-M3-03: task '{task.name}' output '{d.name}' ({d.type.value}): the task does not "
                            f'produce it on every way it can succeed ({how}) -- produce it, or remove the output')
                continue
            if d.type is TaskDataType.CANDIDATES:
                # an input given back unchanged keeps its link: the caller keeps the set it passed
                held = CandidateSet(held.keep, held.projected, f"output '{d.name}' of task '{task.name}'",
                                    input=held.input if held.input == d.name else None)
            outputs[d.name] = held
        writes = {s.detect.out for _, s in walk_steps(task.steps) if s.detect is not None}
        for _, callee in task.calls():
            if callee in self.summaries:
                writes.update(self.summaries[callee].writes)
        exit_at = flow.at
        self.summaries[task.name] = TaskSummary(exit_at, flow.why, tuple(sorted(writes)), outputs,
                                                self._needs.get(task.name, {}),
                                                self._entry_path.get(task.name))
        return steps, moves

    def _plan_steps(self, task, steps, prefix: str, flow: _Flow, moves: List[MovePlan]) -> List[StepPlan]:
        """A list of steps (a task's, or a control body's) in program order; ``flow`` is what TSA knows
        before the first and becomes what it knows after the last."""
        program = self.program
        out: List[StepPlan] = []
        captures = self._counters.setdefault(task.name, [0])     # vNext M3.5: the task's next captures.<n>
        for i, step in enumerate(steps, start=1):
            index = f'{prefix}.{i}' if prefix else str(i)
            kind = step.kind()
            if kind == 'move':
                mp = self.plan_move(task.name, index, step)
                if mp is not None:
                    moves.append(mp)
                    out.append(StepPlan('move', move=mp, label=index))
                flow.at, flow.why = step.move.to, ''
            elif kind == 'path':
                where = f"task '{task.name}' step {index} (path '{step.path}')"
                if step.path not in {p.name for p in program.paths}:
                    known = ', '.join(p.name for p in program.paths) or 'none defined'
                    self.refuse(f"R-M3-03: {where}: no path is named '{step.path}' "
                                f'(paths: {known})')
                    continue
                pp = self._paths.get(step.path)
                if pp is not None and not pp.dense:
                    self._start_check(where, flow, pp)
                    flow.at, flow.why = pp.states[-1], ''
                else:
                    flow.at, flow.why = None, ''
                if pp is not None and pp.events:
                    nodes, caps = self._split(pp, task.name, captures, where)
                    for key, what in caps:
                        flow.captures[key] = what
                    out.append(StepPlan('path', path=step.path, nodes=nodes, captures=caps, label=index))
                else:
                    out.append(StepPlan('path', f'<FollowPath path="{step.path}"/>', path=step.path,
                                        label=index))
            elif kind == 'process':
                ev = step.process
                where = f"task '{task.name}' step {index} (process {ev.kind.value})"
                key, caps = None, []
                if ev.kind is EventKind.CAPTURE:
                    if not self._capture_detector(ev.detector, where):
                        continue
                    key = self._capture_key(ev, task.name, captures, where)
                    caps = [(key, f'{ev.describe()}, {where}')]
                    flow.captures[key] = where
                out.append(StepPlan('process', self._event_node(ev, key), event=ev, captures=caps, label=index))
            elif kind == 'dwell':
                ms = step.dwell.ms
                out.append(StepPlan('dwell', f'<Sleep msec="{ms}"/>', dwell_ms=ms, label=index))
            elif kind == 'task':
                out.append(self._plan_call(task, index, step.task, flow))
            elif kind in CONTROL_KINDS:
                out.append(self._plan_control(task, index, step, flow, moves))
            else:
                # vNext M3.6: detect and tool leave the robot where it is; a grasp or a release ends
                # at its Retract, an entry the idiom derives
                plan_step = {'detect': lambda: self.plan_detect(task.name, index, step, flow.sets),
                             'grasp': lambda: self.plan_grasp(task.name, index, step, flow.sets),
                             'release': lambda: self.plan_release(task.name, index, step),
                             'tool': lambda: self.plan_tool(task.name, index, step)}[kind]
                sp = plan_step()
                if sp is not None:
                    sp.label = index
                    out.append(sp)
                if kind in ('grasp', 'release'):
                    flow.at, flow.why = None, ''
        return out

    def _plan_call(self, task, index: str, callee: str, flow: _Flow) -> StepPlan:
        """A call: the called task's inputs must be held, by name and type, where the call is; what its
        grasps need of a candidate input is checked against the set passed (or passed on to the caller's
        own input); after it the robot is where the task leaves it, and the sets it writes are its
        outputs only."""
        where = f"task '{task.name}' step {index} (task '{callee}')"
        sp = StepPlan('task', f'<SubTree ID="{task_tree_id(callee)}" _autoremap="true"/>', call=callee, label=index)
        target = self._tasks.get(callee)
        summary = self.summaries.get(callee)
        if target is None or summary is None:            # unknown, or in a cycle: refused already
            for d in (target.outputs if target is not None else []):   # no follow-on refusal for them
                if d.type is TaskDataType.CANDIDATES:
                    flow.sets[d.name] = CandidateSet(None, None, f"output '{d.name}' of task '{callee}'")
                else:
                    flow.captures[d.name] = f"output '{d.name}' of task '{callee}'"
            flow.at, flow.why = None, ''
            return sp
        if summary.entry_path is not None:              # where the called task's first path starts
            first, pp = summary.entry_path
            self._start_check(f'{where}, for {first}', flow, pp)
        for d in target.inputs:
            if d.type is TaskDataType.CAPTURE:
                if d.name not in flow.captures:
                    self.refuse(f"R-M3-03: {where}: task '{callee}' takes input capture '{d.name}', and no step "
                                'before the call produces a capture of that name')
                continue
            held = flow.sets.get(d.name)
            if held is None:
                gone = self._gone.get(d.name)
                self.refuse(f"R-M3-03: {where}: task '{callee}' takes input candidates '{d.name}', and no step "
                            'before the call holds a candidate set of that name (a detect step with out: '
                            f"{d.name}, an input of task '{task.name}', or an output of a task called earlier)"
                            + (f' -- {gone}' if gone else ''))
                continue
            for need, no_standoff, at in summary.needs.get(d.name, []):
                self._grasp_needs(f'{where}, for {at}', d.name, held, need - 1, no_standoff)
        for name in summary.writes:
            flow.sets.pop(name, None)                    # a later read would get the callee's set
            self._gone[name] = (f"the call to task '{callee}' at step {index} writes a set of that name and "
                                "does not declare it as an output: declare it in the task's outputs, or name "
                                'the sets apart')
        for d in target.outputs:
            value = summary.outputs.get(d.name)
            if (d.type is TaskDataType.CANDIDATES and value is not None and value.input == d.name
                    and d.name in flow.sets):
                continue                                 # its own input given back: the caller's set
            if d.type is TaskDataType.CANDIDATES and value is not None:
                flow.sets.pop(d.name, None)
                flow.sets[d.name] = value
                self._gone.pop(d.name, None)
            elif d.type is TaskDataType.CAPTURE:
                flow.captures[d.name] = f"output '{d.name}' of task '{callee}'"
        if summary.exit_at != _ENTRY:
            flow.at = summary.exit_at
            flow.why = summary.exit_why or (f"task '{callee}' (called at step {index}) does not end at a "
                                            'waypoint TSA can name' if summary.exit_at is None else '')
        return sp

    def _plan_control(self, task, index: str, step, flow: _Flow, moves: List[MovePlan]) -> StepPlan:
        """repeat / retry / fallback: ORCHESTRATION -- each body's steps are planned exactly as anywhere
        else (their own checks, their own refusals), the wrapper only says how often or in which order
        they run. A body is planned once, from where its FIRST run starts (the robot, the sets); what a
        later run sees differently is checked on top: a repeat's next cycle starts where the body ends
        and sees the sets the body wrote, a retry's next try starts where a failed try left the robot,
        a fallback's next alternative sees what the failed ones wrote."""
        kind = step.kind()
        bodies = step.bodies()
        sp = StepPlan(kind, label=index)
        if kind == 'fallback':
            ends, flows, written = [], [], {}
            for b, body in enumerate(bodies, start=1):
                entry = flow.copy()
                if b > 1:
                    entry.at, entry.why = None, 'an alternative of a fallback starts where the one before it failed'
                    entry.sets = _overwritten(flow.sets, written)
                sp.bodies.append(self._plan_steps(task, body, f'{index}.{b}', entry, moves))
                written.update(self._writes(body))
                flows.append(entry)
                ends.append(entry.at)
            same = len(set(ends)) == 1
            flow.at = ends[0] if same else None
            flow.why = flows[0].why if same else ("after a fallback the robot is where the alternative that "
                                                  'succeeded left it')
            flow.sets = _common(flows, 'sets')
            flow.captures = _common(flows, 'captures')
            return sp
        body = bodies[0]
        count = step.repeat.cycles if kind == 'repeat' else step.retry.attempts
        sp.count = count
        end = self._exit(body, flow.at)
        entry = flow.copy()
        if count != 1:                     # a later run sees the sets the body wrote in the run before
            entry.sets = _overwritten(flow.sets, self._writes(body))
        sp.bodies.append(self._plan_steps(task, body, index, entry, moves))
        first = self._first_path(task, body, index) if count != 1 else None
        if first is not None:
            where, pp = first
            if kind == 'repeat':
                again = flow.at if end == _ENTRY else end
                why = 'a repeat starts each next cycle where the one before ended'
                if again is None or again == _ENTRY or again != flow.at:
                    self._start_check(f'{where}, in a next cycle', _Flow(None if again == _ENTRY else again, why),
                                      pp)
            else:
                self._start_check(f'{where}, in a next try', _Flow(
                    None, 'a retry runs its body again from where a failed try left the robot'), pp)
        flow.at = end if end != _ENTRY else flow.at
        flow.why = entry.why if flow.at is None else ''
        flow.sets, flow.captures = entry.sets, entry.captures
        return sp

    def _first_path(self, task, steps, prefix: str):
        """The path a list of steps runs before any other motion, as (where, its plan); None when a
        motion, a call or a control step comes first."""
        for i, step in enumerate(steps, start=1):
            kind = step.kind()
            if kind == 'path':
                pp = self._paths.get(step.path)
                if pp is None or pp.dense:
                    return None
                label = f'{prefix}.{i}' if prefix else str(i)
                return f"task '{task.name}' step {label} (path '{step.path}')", pp
            if kind in ('move', 'grasp', 'release', 'task') or kind in CONTROL_KINDS:
                return None
        return None

    def _writes(self, steps) -> Dict[str, CandidateSet]:
        """The candidate sets a list of steps writes (its control bodies and calls included), each as the
        last write leaves it: a detect step's, a called task's output, or a set a called task writes
        without declaring it (not known here)."""
        out: Dict[str, CandidateSet] = {}
        for _, step in walk_steps(steps):
            if step.detect is not None:
                d = step.detect
                out[d.out] = CandidateSet(d.keep, d.source is DetectSource.DETECTIONS, 'its detect step')
            elif step.task is not None and step.task in self.summaries:
                summary = self.summaries[step.task]
                for name in summary.writes:
                    out[name] = CandidateSet(None, None, f"a set task '{step.task}' writes")
                for name, value in summary.outputs.items():
                    if isinstance(value, CandidateSet) and value.input is None:
                        out[name] = value
        return out

    def _start_check(self, where: str, flow: '_Flow', pp: PathPlan) -> None:
        start, at = pp.states[0], flow.at
        if at == start:
            return
        if at == _ENTRY and any(self._task == c for t in self.program.tasks for _, c in t.calls()):
            # a called task starting with a path: each call checks where its caller leaves the robot
            self._entry_path.setdefault(self._task, (where, pp))
            return
        if at is None or at == _ENTRY:
            why = flow.why if at is None and flow.why else 'nothing earlier in the task moves the robot there'
            self.warnings.append(f"{where}: the path starts at '{start}', and {why}: FollowPath refuses "
                                 'to start elsewhere (NOT_AT_PATH_START)')
            return
        here, there = self.targets.get(at), self.targets.get(start)
        if here is None or there is None or here.type != 'tcp' or here.frame != there.frame:
            self.warnings.append(f"{where}: the path starts at '{start}' and the step before ends "
                                 f"at '{at}': TSA cannot prove they are the same pose, TMR checks it "
                                 '(NOT_AT_PATH_START)')
            return
        d = math.dist(here.position, there.position)
        a = _angle(here.orientation, there.orientation)
        if d > PATH_START_TOLERANCE[0] or a > PATH_START_TOLERANCE[1]:
            self.refuse(f"R-M3-04: {where}: the path starts at '{start}', but the step before "
                        f"leaves the robot at '{at}' ({d:.4g} m, {a:.4g} rad away; FollowPath "
                        'starts within 5 mm and 0.035 rad of its first state): move to '
                        f"'{start}' first")

    # -- the bundle's data --
    def _assemble(self, paths: Dict[str, PathPlan], tasks: Dict[str, List[StepPlan]],
                  moves: List[MovePlan], running: List[str]) -> ProgramPlan:
        program = self.program
        entries: Dict[str, Dict[str, object]] = {}
        notes: Dict[str, List[str]] = {}
        move_entries: Dict[Tuple[str, str], str] = {}
        variants: Dict[str, List[Tuple]] = {}
        uses: Dict[str, List[MovePlan]] = {}
        # every move, in program order: its waypoint's own entry, or a variant for a second HOW
        order = [(m.task, m.step) for m in moves]
        by_step = {(m.task, m.step): m for m in moves}
        scene_keys = {k for o in self.project.scene.objects
                      for k in (scene_dims_key(o.id), scene_pos_key(o.id))}
        for key in order:
            mp = by_step.get(key)
            if mp is None:
                continue
            seen = variants.setdefault(mp.waypoint, [])
            if mp.how() not in seen:
                seen.append(mp.how())
            k = seen.index(mp.how())
            entry = mp.waypoint if k == 0 else f'{mp.waypoint}__{k + 1}'
            move_entries[key] = entry
            uses.setdefault(entry, []).append(mp)
        # where a waypoint is also used without being moved to: a path state, a circle's point
        also: Dict[str, List[str]] = {}
        for pp in paths.values():
            for s in pp.states:
                also.setdefault(s, []).append(f"state of path '{pp.name}'")
            for i, seg in enumerate(pp.data.get('segments') or []):
                for key in ('via', 'center'):
                    ref = (seg.get(key) or {}).get('waypoint') if isinstance(seg.get(key),
                                                                           dict) else None
                    if ref:
                        also.setdefault(ref, []).append(f"{key} point of path '{pp.name}' "
                                                         f'segment {i}')
        for mp in moves:
            if mp.via:
                also.setdefault(mp.via, []).append(f"via point of task '{mp.task}' step {mp.step}")
        # entries: library waypoints first (in declaration order), each followed by its variants
        for wp in program.waypoints:
            t = self.targets[wp.name]
            names = [wp.name] + [f'{wp.name}__{k + 1}' for k in range(1, len(variants.get(
                wp.name, [])))]
            for name in names:
                if name in scene_keys:
                    raise ProgramRefusals([f"R-M3-03: waypoint '{wp.name}': its bundle entry "
                                           f"'{name}' is also a scene parameter: rename it"])
                keys = dict(t.keys())
                lines = []
                if t.derivation:
                    lines.append(f'{wp.name} = {t.derivation} (a relative target, resolved by TSA)')
                used = uses.get(name, [])
                if used:
                    keys.update(_move_keys(used[0]))
                lines += _move_notes(used)
                if name != wp.name:
                    first = uses[wp.name][0] if uses.get(wp.name) else None
                    differ = [f for f, a, b in zip(_HOW_FIELDS, first.how(), used[0].how())
                              if a != b] if first and used else []
                    lines.insert(0, f"'{wp.name}' again, reached by a move with another "
                                    f"{', '.join(differ) or 'setting'}")
                elif also.get(wp.name):
                    lines.append('also: ' + '; '.join(dict.fromkeys(also[wp.name])))
                if uses.get(name) or name == wp.name:
                    entries[name] = keys
                    notes[name] = lines
        # vNext M3.6: the entries the manipulation idioms derive, after the library
        for name, (keys, lines) in self.bound.items():
            if name in scene_keys:
                raise ProgramRefusals([f"R-M3-03: the derived entry '{name}' is also a scene parameter: "
                                       'rename the task'])
            entries[name] = keys
            notes[name] = lines
        # README / XML lines of the steps
        for task in program.tasks:
            for sp in iter_plans(tasks[task.name]):
                if sp.kind == 'move':
                    entry = move_entries[(task.name, sp.move.step)]
                    sp.xml = f'<MoveWaypoint waypoint="{entry}"/>'
        free = (program.providers.free.value if program.providers and program.providers.free
                else provider_defaults(_REGISTRY)['free'])
        planner_mode = declared_providers(_REGISTRY)[free]['planner_mode']
        rows = declared_providers(_REGISTRY)
        backend = ('cartesian_waypoints' if any(
            rows[p.resolution.provider].get('route') == 'cartesian_path' for p in paths.values())
            else 'pilz_sequence')
        profiles = {}
        for name, prof in program.motion_profiles.items():
            entry: Dict[str, object] = {}
            if prof.planning_time_s is not None:
                entry['planning_time'] = float(prof.planning_time_s)
            if prof.planning_attempts is not None:
                entry['planning_attempts'] = int(prof.planning_attempts)
            if prof.goal_tolerance is not None:
                if prof.goal_tolerance.position_m is not None:
                    entry['goal_position_tolerance'] = float(prof.goal_tolerance.position_m)
                if prof.goal_tolerance.orientation_rad is not None:
                    entry['goal_orientation_tolerance'] = float(prof.goal_tolerance.orientation_rad)
            profiles[name] = entry
        runtime = program.runtime
        # vNext M3.5: a process event on a channel needs the ROS process controller -- the same ROS
        # contract in mock, Isaac and on the real cell (the mock controller only logs); none: as before
        # (the events that RUN: those of main_task and of the tasks it calls, vNext M3.7)
        run_steps = [sp for name in running for sp in iter_plans(tasks.get(name, []))]
        channels = any(sp.kind == 'process' and sp.event.kind in CHANNEL_KINDS for sp in run_steps) or any(
            e.event.kind in CHANNEL_KINDS for sp in run_steps if sp.kind == 'path' and sp.path in paths
            for e in paths[sp.path].events)
        return ProgramPlan(
            targets=self.targets, entries=entries, entry_notes=notes, move_entries=move_entries,
            paths=[paths[p.name] for p in program.paths if p.name in paths], tasks=tasks,
            main_task=program.main_task, profiles=profiles, backend=backend,
            planner_mode=planner_mode, free_provider=free,
            start_tolerance=runtime.allowed_start_tolerance if runtime else None,
            limits=self._limits, warnings=list(self.warnings),
            process_controller='ros' if channels else 'mock', bound_entries=list(self.bound),
            running=running)


def _merged(values: Sequence[CandidateSet], origin: str) -> CandidateSet:
    """One candidate set that may be any of ``values``: what all of them agree on."""
    if all(v == values[0] for v in values):
        return values[0]
    keeps = [v.keep for v in values]
    projected = [v.projected for v in values]
    return CandidateSet(None if None in keeps else min(keeps),
                        True if True in projected else (None if None in projected else False), origin)


def _common(flows: Sequence[_Flow], attr: str) -> Dict[str, object]:
    """What every alternative of a fallback holds when it succeeds (the first one's order); a
    candidate set that differs between them keeps only what all of them agree on."""
    maps = [getattr(f, attr) for f in flows]
    out: Dict[str, object] = {}
    for name, value in maps[0].items():
        values = [m.get(name) for m in maps]
        if any(v is None for v in values):
            continue
        out[name] = value if attr == 'captures' else _merged(
            values, 'the set of whichever alternative of a fallback succeeded')
    return out


def _overwritten(sets: Dict[str, CandidateSet], written: Dict[str, CandidateSet]) -> Dict[str, CandidateSet]:
    """The sets held when a body runs again (a next cycle or try) or a next alternative runs: a set the
    body (or a failed alternative) writes may be the earlier one or the one written there."""
    return {name: (_merged([held, written[name]], 'the set before the body or the one it wrote')
                   if name in written else held) for name, held in sets.items()}


def _change_line(exc: CapabilityRefusal, request: MotionRequest) -> str:
    """What the user can change after a capability refusal (R-M3-10/12): a provider of this kind
    of step that provides every missing capability, or dropping what requires them -- never
    applied for them."""
    if not exc.missing or exc.rule not in ('R-M3-10', 'R-M3-12'):
        return ''
    rows = declared_providers(_REGISTRY)
    kind = request.kind()
    fitting = [p for p in candidate_providers(kind, _REGISTRY)
               if p != exc.provider and all(t in rows[p]['provides'] for t in exc.missing)]
    if fitting and kind not in ('lin', 'circ'):
        where = 'provider' if request.intent is MotionIntent.PATH else 'provider (ptp/free)'
        return (f"name a {where} that provides it -- {', '.join(fitting)} -- or remove what "
                'requires the missing capability')
    return 'remove what requires the missing capability (no provider of this step provides it)'


def _vec_text(v) -> str:
    return '[' + ', '.join(_fmt(x) for x in v) + ']'


def _round_vec(v):
    return tuple(round(x, 6) + 0.0 for x in v)


def _standoff_text(standoff: Optional[float]) -> str:
    return (f'{_fmt(standoff)} m back along the approach' if standoff is not None
            else "the candidate's standoff back along the approach")


def _retract_text(r) -> str:
    return (f'{_fmt(r.along_approach)} m back along the approach' if r.along_approach is not None
            else f'{_fmt(r.dz)} m up +Z')


def _limits_of(moves) -> str:
    return ' ; '.join(f'{_fmt(m.speed)} % / {_fmt(m.accel)} %' for m in moves)


def json_items(model):
    names = list(getattr(type(model), 'model_fields', None) or type(model).__fields__)
    return [(n, getattr(model, n)) for n in names]


def _sub(a, b):
    return tuple(x - y for x, y in zip(a, b))


def _sin_between(u, v) -> float:
    nu, nv = math.sqrt(sum(x * x for x in u)), math.sqrt(sum(x * x for x in v))
    if nu * nv < 1e-12:
        return 0.0
    cx = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return math.sqrt(sum(x * x for x in cx)) / (nu * nv)


def percent(scaling: float) -> float:
    """A scaling as MoveWaypoint's percent, never above the limit it encodes (floored at 1e-6 %)."""
    pct = scaling * 100.0
    if abs(pct - round(pct)) < 1e-9:
        return int(round(pct))
    return math.floor(pct * 1e6) / 1e6


def _move_keys(mp: MovePlan) -> Dict[str, object]:
    keys: Dict[str, object] = {'motion': mp.motion}
    if mp.planner:
        keys['planner'] = mp.planner
    keys['profile'] = mp.profile
    keys['speed'] = mp.speed
    keys['accel'] = mp.accel
    if mp.aux is not None:
        keys['aux'] = list(mp.aux)
        keys['aux_is_center'] = False
    return keys


# what MovePlan.how() holds, by name (a variant entry's comment says which of them differ)
_HOW_FIELDS = ('motion', 'planner', 'profile', 'speed', 'accel', 'via point')

# the request fields that set a scaling: a segment with one (or a profile) writes its own
_SCALING_FIELDS = ('speed', 'acceleration', 'tcp_velocity_mps', 'tcp_acceleration_mps2')

_SOURCE_TEXT = {'request': 'chosen on the step', 'application': 'application.providers',
                'registry_default': 'the registry default'}


def provider_text(resolution: Resolution) -> str:
    kind = resolution.kind
    src = _SOURCE_TEXT.get(resolution.provider_source, resolution.provider_source)
    if resolution.provider_source == 'registry_default':
        src = f'the registry default for {kind}'
    elif resolution.provider_source == 'application':
        src = f'application.providers.{kind}'
    return f'provider {resolution.provider} ({src})'


def scaling_text(resolution: Resolution) -> List[str]:
    """The derivation of an effective scaling that is not a plain percent (a TCP limit won)."""
    out = []
    for label, s in (('speed', resolution.velocity_scaling),
                     ('acceleration', resolution.acceleration_scaling)):
        if s is not None and s.source.startswith('tcp_'):
            out.append(f'{label}: {s.derivation}')
    return out


def move_text(mp: MovePlan, with_motion: bool = True) -> str:
    """HOW a move reaches its waypoint, as the README and the bundle's comments say it."""
    parts = [f"via '{mp.via}'"] if mp.via else []
    parts.append(f'profile {mp.profile}' if mp.profile_named else
                 f'no profile (TMR\'s {mp.profile})')
    parts.append(provider_text(mp.resolution))
    text = ', '.join(parts)
    return f'{mp.motion.upper()} {text}' if with_motion and mp.via else (
        f'{mp.motion.upper()}, {text}' if with_motion else text)


def _move_notes(moves: Sequence[MovePlan]) -> List[str]:
    """One line per task for the moves sharing an entry (they share their HOW), then how a TCP
    limit set the speed or acceleration."""
    if not moves:
        return []
    steps: Dict[str, List[str]] = {}
    for mp in moves:
        steps.setdefault(mp.task, []).append(mp.step)
    lines = []
    for task, nums in steps.items():
        label = 'step' if len(nums) == 1 else 'steps'
        lines.append(f"task '{task}' {label} {', '.join(map(str, nums))}: {move_text(moves[0])}")
    lines += scaling_text(moves[0].resolution)
    return lines


def program_summary(program: TaskProgram) -> str:
    """One line: the main task and the size of the program (the wizard's state line)."""
    def n(count, what):
        return f'{count} {what}' + ('' if count == 1 else 's')
    return (f"main task '{program.main_task}' · {n(len(program.tasks), 'task')} · "
            f"{n(len(program.waypoints), 'waypoint')} · {n(len(program.paths), 'path')} · "
            f"{n(len(program.motion_profiles), 'motion profile')}")


def program_overview(program: TaskProgram) -> str:
    """The program as a reader sees it, from the model alone (no validation: it works for a
    program that does not validate, so the wizard can show what it holds and what is wrong)."""
    lines = [f'Main task: {program.main_task}', '']
    if any(s.kind() in ('task',) + CONTROL_KINDS for t in program.tasks for _, s in walk_steps(t.steps)):
        lines += ['(a step in a body is labelled 3.1 -- step 1 of the repeat or retry at step 3 -- or 5.2.1 --',
                  ' step 1 of the 2nd alternative of the fallback at step 5)', '']
    for task in program.tasks:
        lines.append(f'task {task.name}' + (f' -- {task.description}' if task.description else ''))
        for what, items in (('takes', task.inputs), ('gives', task.outputs)):
            if items:
                lines.append(f'  {what} ' + ', '.join(d.describe().replace('`', '') for d in items))
        lines += _overview_steps(program, task.steps, '', 2)
        lines.append('')
    lines.append(f'Waypoints ({len(program.waypoints)})')
    width = max((len(w.name) for w in program.waypoints), default=0)
    for w in program.waypoints:
        if w.type is WaypointType.JOINT:
            what = f'joint, named state {w.named}' if w.named else 'joint values'
        elif w.relative is not None:
            r = w.relative
            what = (f'relative to {r.to}: ({r.dx:g}, {r.dy:g}, {r.dz:g}) m along the '
                    f'{r.frame.value} axes')
        else:
            what = 'pose' + (f', frame {w.frame}' if w.frame else '') + \
                (f', tcp {w.tcp}' if w.tcp else '')
        lines.append(f'  {w.name:<{width}}  {what}')
    lines.append(f'Paths ({len(program.paths)}): '
                 + (', '.join(p.name for p in program.paths) or 'none'))
    lines.append(f'Motion profiles ({len(program.motion_profiles)}): '
                 + (', '.join(program.motion_profiles) or 'none'))
    chosen = []
    if program.providers is not None:
        chosen = [f'{k}={getattr(program.providers, k).value}'
                  for k in ('ptp', 'free', 'path', 'dense_path')
                  if getattr(program.providers, k) is not None]
    lines.append('Providers: ' + (', '.join(chosen) + '; the rest: ' if chosen else '')
                 + 'the registry defaults')
    if program.runtime is not None and program.runtime.allowed_start_tolerance is not None:
        lines.append(f'Start-state tolerance: {program.runtime.allowed_start_tolerance} rad')
    return '\n'.join(lines)


def _overview_steps(program: TaskProgram, steps, prefix: str, pad: int) -> List[str]:
    """The overview lines of a list of steps; a control step's body below it, indented, labelled
    3.1, 3.2.1 (vNext M3.7)."""
    lines: List[str] = []
    sp = ' ' * pad
    for i, step in enumerate(steps, start=1):
        label = f'{prefix}.{i}' if prefix else str(i)
        n = f'{label:>3}'
        if step.move is not None:
            mv = step.move
            how = []
            if mv.via is not None:
                how.append(f'via {mv.via if isinstance(mv.via, str) else list(mv.via)}')
            if mv.center is not None:
                how.append(f'center {mv.center}')
            how.append(f'profile {mv.profile}' if mv.profile else 'no profile')
            if mv.provider:
                how.append(f'provider {mv.provider}')
            extra = [f'{f} {getattr(mv, f)}' for f in mv.requested()]
            lines.append(f'{sp}{n}  move  {mv.type.value.upper():<5} -> {mv.to:<12} ' + ', '.join(how + extra))
        elif step.path is not None:
            path = next((p for p in program.paths if p.name == step.path), None)
            shape = ''
            if path is not None and path.states:
                shape = (f"{' -> '.join(path.states)}, "
                         f"{(path.continuity or PathContinuity.STOP).value}")
            elif path is not None:
                shape = 'dense path'
            lines.append(f'{sp}{n}  path  {step.path:<21} {shape}')
            for ev in (path.events if path is not None else []):
                lines.append(f'{sp}       {ev.at.describe()}: {ev.describe().replace("`", "")}')
        elif step.process is not None:
            lines.append(f'{sp}{n}  process {step.process.describe().replace("`", "")}')
        elif step.dwell is not None:
            lines.append(f'{sp}{n}  dwell {step.dwell.ms} ms')
        elif step.detect is not None:
            d = step.detect
            lines.append(f'{sp}{n}  detect {d.source.value} of {d.detector} -> {d.out} (keep {d.keep})')
        elif step.grasp is not None:
            g = step.grasp
            lines.append(f'{sp}{n}  {g.describe()}, profiles {g.profile} / {g.approach_profile}')
        elif step.release is not None:
            r = step.release
            lines.append(f'{sp}{n}  {r.describe()}, profiles {r.profile} / {r.approach_profile}')
        elif step.tool is not None:
            t = step.tool
            if t.actuate is not None:
                what = f'actuate {t.actuate.actuator} -> {t.actuate.target}'
            elif t.verify is not None:
                what = f'verify {t.verify.expect.value}'
            else:
                g = t.close or t.open
                what = t.form() + (f' site {g.site}' if g.site else '')
            lines.append(f'{sp}{n}  tool {what}')
        elif step.task is not None:
            lines.append(f'{sp}{n}  task  {step.task} (call)')
        elif step.repeat is not None:
            c = step.repeat.cycles
            lines.append(f'{sp}{n}  repeat ' + ('until a step fails' if c == -1 else f'{c} cycles') + ':')
            lines += _overview_steps(program, step.repeat.body, label, pad + 4)
        elif step.retry is not None:
            lines.append(f'{sp}{n}  retry, {step.retry.attempts} tries at most:')
            lines += _overview_steps(program, step.retry.body, label, pad + 4)
        else:
            lines.append(f'{sp}{n}  fallback, the first alternative that succeeds:')
            for b, body in enumerate(step.fallback.bodies, start=1):
                lines.append(f'{sp}      alternative {b}:')
                lines += _overview_steps(program, body, f'{label}.{b}', pad + 8)
    return lines


def resolve_targets(project: CanonicalProject) -> Tuple[Dict[str, Target], List[str]]:
    """Every waypoint of a task program resolved to WHERE (a relative target's chain followed), with the
    waypoint refusals: what the live waypoint check reads (vNext I0, owner Test C). A waypoint that cannot
    be resolved is missing from the targets; the rest of the program is not planned."""
    planner = _Planner(project)
    planner.resolve_waypoints()
    return dict(planner.targets), list(planner.problems)


def plan_program(project: CanonicalProject) -> ProgramPlan:
    """The plan of a task program, or :class:`ProgramRefusals` naming every problem."""
    if not project.application.is_task_program():
        raise ValueError(f"plan_program: application '{project.application.type.value}' is not a "
                         'task_program')
    return _Planner(project).plan()


# --- the application ------------------------------------------------------------------------------

class TaskProgramApplication(ApplicationTemplate):
    """The Canonical Task Program as an application of the registry (not a template: the program is
    the project's own). ``validate`` returns the refusals and the warnings, for the GUI;
    ``refusals`` stops generation on any problem."""

    app_type = AppType.TASK_PROGRAM.value

    def validate(self, project: CanonicalProject) -> List[str]:
        try:
            plan = plan_program(project)
        except ProgramRefusals as exc:
            return self.validate_common(project) + list(exc.problems)
        return self.validate_common(project) + [f'warning: {w}' for w in plan.warnings]

    def refusals(self, project: CanonicalProject) -> List[str]:
        try:
            plan_program(project)
        except ProgramRefusals as exc:
            return list(exc.problems)
        return []

    def tree_filename(self, project: CanonicalProject) -> str:
        return TREE_FILE

    def build_tree_xml(self, project: CanonicalProject) -> str:
        return program_tree_xml(project, plan_program(project))


def program_tree_xml(project: CanonicalProject, plan: ProgramPlan) -> str:
    """MainTree (the scene, then the main task) and one BehaviorTree per task: a move is ONE
    MoveWaypoint, a path ONE FollowPath whatever its number of states (a path with process events:
    its event-free runs and the event nodes between them), a process step ONE ProcessEvent (or a
    capture's DetectObject), a dwell ONE Sleep."""
    out = ['<?xml version="1.0" encoding="UTF-8"?>',
           '<!-- Generated by the TrainIt Setup Assistant (task_program). One BehaviorTree per task; '
           'waypoints and moves in config/bt_params.yaml, paths in config/paths.yaml. -->',
           f'<root BTCPP_format="4" main_tree_to_execute="{MAIN_TREE}">',
           f'  <BehaviorTree ID="{MAIN_TREE}">',
           '    <Sequence>']
    if not project.robot.base_moveit_config_path:
        for obj in project.scene.objects:
            if obj.is_planning_collision():
                out.append(f'      <AddCollisionObject id="{obj.id}" frame="{obj.frame}" '
                           f'dims="{{{scene_dims_key(obj.id)}}}" '
                           f'position="{{{scene_pos_key(obj.id)}}}"/>')
    # _autoremap: a task's nodes read the waypoints and the runtime from the root blackboard
    out.append(f'      <SubTree ID="{task_tree_id(plan.main_task)}" _autoremap="true"/>')
    out += ['    </Sequence>', '  </BehaviorTree>']
    for task in project.application.program.tasks:
        out.append(f'  <BehaviorTree ID="{task_tree_id(task.name)}">')
        out.append('    <Sequence>')
        out += _steps_xml(plan.tasks[task.name], '      ')
        out += ['    </Sequence>', '  </BehaviorTree>']
    out.append('</root>')
    return '\n'.join(out) + '\n'


# vNext M3.7: the control steps as BehaviorTree.CPP's own nodes; a body is always ONE Sequence
_CONTROL_XML = {'repeat': ('Repeat', 'num_cycles'), 'retry': ('RetryUntilSuccessful', 'num_attempts')}


def _steps_xml(steps: Sequence[StepPlan], indent: str) -> List[str]:
    """The BT nodes of a list of step plans: a step's node(s), a call's SubTree (``_autoremap``: the
    called task reads and writes the application's one blackboard, so its data is bound by name), a
    repeat or retry around its body, a fallback around its alternatives."""
    out: List[str] = []
    inner = indent + '  '
    for sp in steps:
        if sp.kind in _CONTROL_XML:
            tag, port = _CONTROL_XML[sp.kind]
            out.append(f'{indent}<{tag} {port}="{sp.count}">')
            out += [f'{inner}<Sequence>'] + _steps_xml(sp.bodies[0], inner + '  ') + [f'{inner}</Sequence>']
            out.append(f'{indent}</{tag}>')
        elif sp.kind == 'fallback':
            out.append(f'{indent}<Fallback>')
            for body in sp.bodies:
                out += [f'{inner}<Sequence>'] + _steps_xml(body, inner + '  ') + [f'{inner}</Sequence>']
            out.append(f'{indent}</Fallback>')
        else:
            out += [f'{indent}{node}' for node in (sp.nodes or [sp.xml])]
    return out


__all__ = ['CAPTURE_PREFIX', 'MIN_MOVE_SCALING', 'percent', 'task_tree_id', 'MovePlan', 'PathPlan', 'PlannedEvent',
           'CandidateSet', 'TaskSummary', 'iter_plans',
           'ProgramPlan', 'ProgramRefusals', 'StepPlan',
           'PATH_START_TOLERANCE', 'TMR_BASE_PROFILE', 'Target', 'TaskProgramApplication', 'plan_program',
           'resolve_targets',
           'move_text', 'program_overview', 'program_summary', 'program_tree_xml',
           'provider_text', 'scaling_text']
