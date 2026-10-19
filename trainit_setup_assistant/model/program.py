"""The Canonical Task Program: waypoints, motion profiles, paths and tasks (vNext M3.3, ADR-0024).

A ``task_program`` application describes a robot program the way an automation engineer reads one,
top to bottom:

    APPLICATION   providers (optional), motion_profiles, waypoints, paths, tasks, main_task, runtime
      TASK        a named, ordered list of STEPS, with the data it takes and gives (``inputs``,
                  ``outputs``: candidate sets and captures, by name; vNext M3.7)
        STEP      move | path | process | dwell | detect | grasp | release | tool | task
                  | repeat | retry | fallback
          move    ONE motion (ptp | lin | circ | free) to a WAYPOINT, with a PROFILE and limits
          path    ONE motion through >= 2 waypoints (lin | circ segments, stop or blend), and the
                  PROCESS EVENTS anchored at its states (vNext M3.5)
          process ONE process event where the task is: process_on | process_off | set_value |
                  external on a logical CHANNEL, or a capture by a DETECTOR (vNext M3.5)
          dwell   a hold, in milliseconds
          detect  sample a grasp producer into a CANDIDATE SET (vNext M3.6, ADR-0026): a producer's
                  grasp candidates, or a detector's detections projected onto the same contract
          grasp   the fixed GRASP idiom on a candidate or a taught target: PreGrasp, Approach, close,
                  verify (as the end-effector contract requires), Retract (vNext M3.6)
          release the fixed RELEASE idiom at a taught target (vNext M3.6)
          tool    one end-effector operation: an actuator to a named target, or a grasp verification
          task    CALL another task of the program (vNext M3.7): it runs, then the step after the
                  call; the calls form no cycle
          repeat  run a BODY of steps ``cycles`` times (-1: until a step fails or the program stops)
          retry   run a BODY of steps again when one of its steps fails, ``attempts`` tries at most
          fallback  try ALTERNATIVE bodies in order until one succeeds
                  (the three control steps are ORCHESTRATION: they never change what a step does)
      WAYPOINT    a named TARGET: WHERE, never HOW (no motion type, speed or blend on a waypoint)
      PROFILE     a named set of limits (speed, acceleration, TCP velocity, tolerance, planning budget)
      PROVIDER    which motion provider (a row of TMR's registry) runs a kind of motion; omitted =
                  the registry's default, resolved and written into the bundle by TSA

These are the STRUCTURAL models, checked when the project loads. Every one is strict: an unknown key
is refused, naming the keys the model accepts, and the keys a reader might put in the wrong place
carry a hint (a motion type on a waypoint, a template application's key, a step kind of a later
phase). The references between the sections, the robot (frames, TCP, named states), the providers
and every limit are checked by ``applications.task_program`` before a single file is written.

The motion limits are M3.2's request vocabulary (``model.motion``): a step, a path or a segment
carries the fields of ``MotionConstraints`` inline, and a profile is a ``MotionProfile``. Pydantic:
the classic v1/v2-overlap API, as every TSA model.
"""

from __future__ import annotations

import math
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, root_validator, validator

from .enums import MotionType, StrEnum, WaypointType
from .motion import (MotionConstraints, MotionProfile, PathContinuity, ProviderChoices,
                     _checked, _non_negative, _number, known_provider)

# A name that becomes a ROS parameter key, a YAML key, a BT attribute and part of a BT tree id.
NAME = re.compile(r'^[A-Za-z][A-Za-z0-9_]*$')
# The name TMR's motion-profile loader reserves for its base profile (bt_profiles.cpp).
RESERVED_PROFILE_NAMES = ('default',)
# A unit quaternion within TMR's tolerance (path_library.cpp, bt_nodes.cpp poseProblem): it is never
# normalised silently.
UNIT_QUATERNION_TOLERANCE = 1e-3

# --- hints for keys in the wrong place --------------------------------------------------------------

_HOW = ('a waypoint says WHERE, never HOW: {what} belongs to the move step or the path segment that '
        'reaches it')
WAYPOINT_HINTS = {
    'motion': _HOW.format(what="the motion type ('type: ptp | lin | circ | free')"),
    'speed': _HOW.format(what='a speed'),
    'acceleration': _HOW.format(what='an acceleration'),
    'tcp_velocity_mps': _HOW.format(what='a TCP velocity'),
    'planner': _HOW.format(what='a provider'),
    'provider': _HOW.format(what='a provider'),
    'profile': _HOW.format(what='a motion profile'),
    'blend_radius_m': _HOW.format(what='a blend radius'),
    'via': _HOW.format(what="a circle's via point"),
    'aux': _HOW.format(what="a circle's via point"),
    'role': 'a task program has no waypoint roles: its task steps say what happens where',
    'vision': "a detected target is a task step: 'detect' (a candidate set), then 'grasp: {candidates: ...}'",
    'allowed_start_tolerance': ('a task program sets the start-state tolerance once: '
                                'application.runtime.allowed_start_tolerance'),
    'pattern': ('R-M3-14: a pattern target (a grid of slots from an origin, a pitch and counts) is '
                'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL: it is reserved for a later phase -- teach each '
                'slot as a waypoint'),
}
_LATER = ("step kind '{kind}' is not available in this TSA version: {what} are planned for a "
          'later one (UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL)')
STEP_HINTS = {
    'capture': "a capture is a process event: 'process: {kind: capture, detector: <detector>}'",
    'pick': "a pick is a composition: 'detect' (or a taught target), then 'grasp', then 'move' / 'release'",
    'place': "a place is a 'release' step at a taught target (a PlaceTarget is not a step kind)",
    'check': _LATER.format(kind='check', what='robot-state checks'),
    'call': "a task calls another task with 'task: <name>'",
    'subtask': "a task calls another task with 'task: <name>'",
    'loop': "a loop is 'repeat: {cycles: <n>, body: [<steps>]}'",
    'while': ('a task program has no conditions of its own: a repeat runs its body a number of times '
              "('cycles'), a retry until it succeeds ('attempts'), a fallback until one alternative "
              'succeeds -- conditions come from what the steps observe (a detection, a verification)'),
    'if': ('a task program has no conditions of its own: a fallback tries alternative bodies in order '
           "('fallback: {bodies: [[<steps>], [<steps>]]}'), each one succeeding or failing on what its "
           'steps observe'),
    'sequence': "a task's steps already run in order: a body of steps is a list ('body: [<steps>]')",
}
TASK_HINTS = {
    'calls': "a task calls another task with a step: 'task: <name>'",
    'main': "the task that runs first is the program's 'main_task'",
}
MOVE_HINTS = {
    'planner': ("a task program names a provider ('provider: <id>', a row of TMR's registry); "
                'application.providers sets the default per kind of motion'),
    'waypoint': "the target of a move is 'to: <waypoint>'",
    'target': "the target of a move is 'to: <waypoint>'",
    'motion': "the motion type of a move is 'type: ptp | lin | circ | free'",
    'execution': 'a streamed (servo) move is not part of the task program model in M3',
}
PATH_HINTS = {
    'event': "a path's process events are a list: 'events: [{at: {state: <waypoint>}, kind: ...}]'",
    'planner': "a path names a provider ('provider: <id>')",
    'start_tolerance': ('the start tolerance of a path is TMR\'s default (5 mm, 0.035 rad) in M3: '
                        'reach the first state with a move before the path'),
}
SEGMENT_HINTS = {
    'motion': "the shape of a segment is 'type: lin | circ'",
    'aux': "a circular segment names 'via: <waypoint>' or 'center: <waypoint>'",
}
SOURCE_HINTS = {
    'file': ('a dense path read from a file is planned for a later TSA version; use a '
             'generator, or a path of states'),
}
RELATIVE_HINTS = {
    'droll': 'rotation offsets of a relative target are planned for a later TSA version (translations only: dx, dy, dz)',
    'dpitch': 'rotation offsets of a relative target are planned for a later TSA version (translations only: dx, dy, dz)',
    'dyaw': 'rotation offsets of a relative target are planned for a later TSA version (translations only: dx, dy, dz)',
    'step': "a relative target names the waypoint it is relative to: 'to: <waypoint>'",
}


def _fields_of(cls) -> List[str]:
    return list(getattr(cls, 'model_fields', None) or cls.__fields__)


def _refuse_unknown(cls, values, hints: Dict[str, str], what: str):
    """Refuse an unknown key by name, with the hint for it when one exists, else the accepted keys.
    Runs before pydantic's own ``extra = 'forbid'`` (the backstop), so the message is ours."""
    if not isinstance(values, dict):
        return values
    accepted = _fields_of(cls)
    unknown = [k for k in values if k not in accepted]
    from .motion import RESERVED_CONSTRAINT_FIELDS
    shown = [f for f in accepted if f not in RESERVED_CONSTRAINT_FIELDS]
    if not unknown:
        return values
    for key in unknown:                       # a hint that is a whole refusal of its own (R-M3-09, 14)
        hint = hints.get(key)
        if hint and hint.startswith('R-M3-') and not hint.startswith('R-M3-02'):
            raise ValueError(hint)
    lines = []
    for key in unknown:
        hint = hints.get(key)
        lines.append(f"{what}: '{key}' -- {hint}" if hint else
                     f"{what}: unknown key '{key}' (accepted: {', '.join(shown)})")
    raise ValueError('R-M3-02: ' + '; '.join(lines))


# The models that have a strict-key validator of their own, as (module, class name). Pydantic 1.x runs
# an INHERITED 'before' root validator too (the parent's first), pydantic 2.x only the override: so the
# validator acts only for the most derived class that has one -- the same refusal on both (vNext M3.7,
# found by the fresh-machine check on the apt pydantic).
_STRICT_OWNERS: set = set()


def _strict_keys(hints: Dict[str, str], what: str):
    caller = sys._getframe(1)                 # the class body it is called in: its name is the class's
    owner = (caller.f_globals.get('__name__'), caller.f_code.co_name)
    _STRICT_OWNERS.add(owner)

    def run(cls, values):  # noqa: N805
        acting = next(((k.__module__, k.__name__) for k in cls.__mro__
                       if (k.__module__, k.__name__) in _STRICT_OWNERS), owner)
        if acting != owner:
            return values                     # a subclass's own validator decides
        return _refuse_unknown(cls, values, hints, what)
    run.__name__ = '_strict_keys'
    return root_validator(pre=True, allow_reuse=True)(run)


class _Strict(BaseModel):
    class Config:
        extra = 'forbid'
        validate_assignment = True


# Words a YAML 1.1 reader (ROS 2's parameter files) reads as a boolean or null, not as a name.
_YAML_WORDS = {'y', 'yes', 'n', 'no', 'true', 'false', 'on', 'off', 'null'}


def _name(value: Any, what: str):
    if not isinstance(value, str) or not NAME.fullmatch(value):   # fullmatch: no trailing newline
        raise ValueError(f'R-M3-02: {what} {value!r} must be a name: a letter, then letters, digits '
                         'or _ (it becomes a ROS parameter key and a BehaviorTree id)')
    if value.lower() in _YAML_WORDS:
        raise ValueError(f'R-M3-02: {what} {value!r} is a word a ROS parameter file reads as a '
                         'boolean or null, not as a name: choose another name')
    return value


def _waypoint_name(value: Any, what: str):
    _name(value, what)
    if '__' in value:
        raise ValueError(f"R-M3-02: {what} {value!r} contains '__', which TSA reserves for the "
                         'bundle entries of a waypoint reached by different moves')
    return value


def _numbers(value: Any, what: str, count: Optional[int] = None):
    if value is None:
        return value
    if not isinstance(value, (list, tuple)):
        raise ValueError(f'R-M3-02: {what} must be a list of numbers, got {value!r}')
    if count is not None and len(value) != count:
        raise ValueError(f'R-M3-02: {what} must be {count} numbers, got {len(value)}')
    return [_number(v, what) for v in value]


def _quaternion(value: Any, what: str):
    value = _numbers(value, f'{what} (qx, qy, qz, qw)', 4)
    if value is None:
        return value
    norm = math.sqrt(sum(v * v for v in value))
    if abs(norm - 1.0) > UNIT_QUATERNION_TOLERANCE:
        raise ValueError(f'R-M3-02: {what} {value} is not a unit quaternion (norm {norm:.6g}): it is '
                         'not normalised silently')
    return value


def _aux_point(value: Any, what: str):
    """A circle's auxiliary point: a waypoint NAME (its position) or [x, y, z]."""
    if value is None or isinstance(value, str):
        return value
    return [float(v) for v in _numbers(value, f'{what} [x, y, z]', 3)]


# --- waypoints ------------------------------------------------------------------------------------

class RelativeFrame(StrEnum):
    """The axes a relative offset is expressed in. ``base``: the robot's base axes (ABB's Offs);
    ``tool``: the axes of the referenced target's TCP (ABB's RelTool); ``target_local`` (vNext M3.6):
    the referenced TARGET's own local axes -- neither the base axes, nor the robot's current tool
    axes, nor another TF frame. For a taught reference the two coincide (its TCP orientation IS its
    orientation, so the two give the same numbers); the manipulation idioms place a taught target's
    PreGrasp and Retract in its target_local axes (along its tool +Z). ``target_local`` is the
    portable spelling: it keeps its meaning for a reference whose own axes differ from its TCP's (an
    object frame, a detected or bound target, later versions); ``tool`` stays the taught-TCP spelling.

    These are the AXES the offset is measured along, not a TF frame: the relative target is in its
    reference's own frame."""

    BASE = 'base'
    TOOL = 'tool'
    TARGET_LOCAL = 'target_local'


class RelativeTarget(_Strict):
    """A target taught as an OFFSET from another waypoint's pose (translation, metres)."""

    to: str
    frame: RelativeFrame
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0

    _keys = _strict_keys(RELATIVE_HINTS, 'a relative target')
    _to = _checked('to', _waypoint_name, 'relative.to')
    _dx = _checked('dx', _number, 'relative.dx')
    _dy = _checked('dy', _number, 'relative.dy')
    _dz = _checked('dz', _number, 'relative.dz')


class TaskWaypoint(_Strict):
    """A named TARGET of the application: WHERE the robot goes, never how.

    tcp    ``position`` [x, y, z] (m) + ``orientation`` [qx, qy, qz, qw] in ``frame`` (default the
           robot's base frame), for ``tcp`` (default the robot's tip link); or ``relative`` to
           another waypoint (it then inherits that waypoint's frame and TCP)
    joint  ``joints`` (rad, one per joint of the planning group) or ``named`` (a named state)
    """

    name: str
    type: WaypointType = WaypointType.TCP
    position: Optional[List[float]] = None
    orientation: Optional[List[float]] = None
    joints: Optional[List[float]] = None
    named: Optional[str] = None
    frame: Optional[str] = None
    tcp: Optional[str] = None
    relative: Optional[RelativeTarget] = None

    _keys = _strict_keys(WAYPOINT_HINTS, 'a waypoint')
    _name_ok = _checked('name', _waypoint_name, 'waypoint name')

    @validator('type', pre=True)
    def _where_type(cls, v):  # noqa: N805
        if isinstance(v, str) and v in [m.value for m in MotionType]:
            raise ValueError(f"R-M3-02: a waypoint's type is tcp | joint (WHERE); '{v}' is a motion "
                             'type, which belongs to the move step or the path segment that '
                             'reaches it')
        return v
    _position = _checked('position', lambda v, w: _numbers(v, w, 3), 'position [x, y, z]')
    _orientation = _checked('orientation', _quaternion, 'orientation')
    _joints = _checked('joints', _numbers, 'joints')

    @root_validator(skip_on_failure=True)
    def _one_target(cls, values):  # noqa: N805
        name = values.get('name')
        where = f"waypoint '{name}'"
        pose = values.get('position') is not None or values.get('orientation') is not None
        if values.get('type') is WaypointType.JOINT:
            given = [k for k in ('position', 'orientation', 'frame', 'tcp', 'relative')
                     if values.get(k) is not None]
            if given:
                raise ValueError(f'R-M3-02: {where} is a joint target: {", ".join(given)} belong to '
                                 'a tcp target')
            if (values.get('joints') is None) == (values.get('named') is None):
                raise ValueError(f"R-M3-02: {where} is a joint target: give exactly one of "
                                 "'joints' (rad) or 'named' (a named state)")
            if values.get('joints') is not None and not values['joints']:
                raise ValueError(f'R-M3-02: {where}: joints is empty')
            return values
        if values.get('joints') is not None or values.get('named') is not None:
            raise ValueError(f"R-M3-02: {where} is a tcp target: 'joints' and 'named' belong to a "
                             "joint target ('type: joint')")
        if values.get('relative') is not None:
            if pose:
                raise ValueError(f'R-M3-02: {where} is relative to another waypoint: it has no '
                                 'position or orientation of its own')
            given = [k for k in ('frame', 'tcp') if values.get(k) is not None]
            if given:
                raise ValueError(f'R-M3-02: {where} is relative to another waypoint and inherits its '
                                 f'{" and ".join(given)}')
            return values
        if values.get('position') is None or values.get('orientation') is None:
            raise ValueError(f"R-M3-02: {where} is a tcp target: give 'position' [x, y, z] and "
                             "'orientation' [qx, qy, qz, qw], or 'relative' to another waypoint")
        for key in ('frame', 'tcp'):
            if values.get(key) is not None and not values[key]:
                raise ValueError(f'R-M3-02: {where}: {key} is empty (omit it for the default)')
        return values


# --- process events (vNext M3.5, M3_DESIGN §9) -------------------------------------------------------

class EventKind(StrEnum):
    """The kinds of a PROCESS EVENT: what happens, never which application it is for (a gluing, a
    dispensing or an inspection is a composition of these, not a kind of its own)."""

    PROCESS_ON = 'process_on'        # a process runs (on its logical channel)
    PROCESS_OFF = 'process_off'      # it stops
    SET_VALUE = 'set_value'          # its value (a number)
    CAPTURE = 'capture'              # a detector is sampled; its FRESH result goes on the blackboard
    DWELL = 'dwell'                  # a hold, in milliseconds
    EXTERNAL = 'external'            # a logical external action, e.g. channel washer_start: one signal
                                     # on its channel, no acknowledgement in this version


CHANNEL_KINDS = (EventKind.PROCESS_ON, EventKind.PROCESS_OFF, EventKind.SET_VALUE, EventKind.EXTERNAL)
# what each kind takes besides `kind` (and `sync`): (required, optional)
EVENT_FIELDS = {
    EventKind.PROCESS_ON: (('channel',), ()),
    EventKind.PROCESS_OFF: (('channel',), ()),
    EventKind.EXTERNAL: (('channel',), ()),
    EventKind.SET_VALUE: (('channel', 'value'), ()),
    EventKind.CAPTURE: (('detector',), ('out', 'class_id', 'timeout_ms')),
    EventKind.DWELL: (('ms',), ()),
}
_PAYLOAD = ('channel', 'value', 'detector', 'out', 'class_id', 'timeout_ms', 'ms')
# a capture's blackboard key: dot-separated names (captures.0, inspection.top)
CAPTURE_KEY = re.compile(r'^[A-Za-z][A-Za-z0-9_]*(\.[A-Za-z0-9_]+)*$')

_ADDRESS = ("a process event names a LOGICAL channel ('channel: dispenser'), never a {key}: the "
            'runtime binds the id to its ROS topic or I/O')
EVENT_HINTS = {
    'topic': _ADDRESS.format(key='topic'),
    'address': _ADDRESS.format(key='address'),
    'symbol': _ADDRESS.format(key='symbol'),
    'output': _ADDRESS.format(key='output'),
    'signal': _ADDRESS.format(key='signal'),
    'type': "the kind of a process event is 'kind: process_on | process_off | set_value | capture | "
            "dwell | external'",
    'trigger': 'a capture always triggers its detector and waits for a FRESH result',
    'keep': 'a capture keeps the first published result (candidate lists come with the detect step, '
            'planned for a later TSA version)',
    'record': 'a capture puts its result on the blackboard only: recording comes with a later phase (D0)',
    'duration_ms': "a dwell's hold is 'ms: <milliseconds>'",
}
_INSIDE = ('R-M3-09: {where}: an event placed by {what}, inside a segment, is UNSUPPORTED_BY_CURRENT_'
           'CANONICAL_MODEL: this version synchronises process events at path states and segment boundaries '
           'only (at: {{state: <waypoint>}} or at: {{segment: <i>, when: start | end}}); an exact position '
           'or time inside a moving segment needs a synchronisation with the robot controller, planned '
           'for a later phase -- split the segment with a waypoint where the event must happen')
_INSIDE_KEYS = {
    'fraction': 'a fraction of the segment', 'percent': 'a percentage of the segment',
    'ratio': 'a fraction of the segment', 'distance_m': 'a distance along the segment',
    'offset_m': 'a distance along the segment', 'time_s': 'a time into the segment',
    'time_ms': 'a time into the segment', 'at_time': 'a time into the segment',
    'position': 'a position', 'pose': 'a pose',
}


def _channel(value: Any, what: str):
    """A process channel: a LOGICAL id of the application (owner decision, CELL_SCENE_IMPORT_REVIEW
    decision 4) -- never a ROS topic path, a PLC address or a controller symbol."""
    if value is None:
        return value
    if not isinstance(value, str) or not NAME.fullmatch(value) or value.lower() in _YAML_WORDS:
        raise ValueError(f'R-M3-02: {what} {value!r} is not a logical channel id: a channel names the '
                         'process in the application\'s own terms -- a letter, then letters, digits or '
                         '_ (dispenser, washer_start, inspection_light) -- never a ROS topic path, a '
                         'hardware address or a controller symbol: the runtime binds the id to its signal')
    return value


# a detector's class id as the perception contract carries it (hypothesis.class_id): a token
CLASS_ID = re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.\-]*')


def _class_token(value: Any, what: str):
    if value is None:
        return value
    if not isinstance(value, str) or not CLASS_ID.fullmatch(value):
        raise ValueError(f'R-M3-02: {what} {value!r} must be a detector class id: letters, digits, _ . -')
    return value


def _capture_key(value: Any, what: str):
    if value is None:
        return value
    if not isinstance(value, str) or not CAPTURE_KEY.fullmatch(value):
        raise ValueError(f'R-M3-02: {what} {value!r} must be a blackboard key: names joined by dots '
                         '(captures.0, inspection.top)')
    return value


class ProcessEventSpec(_Strict):
    """ONE process event (M3_DESIGN §9): its ``kind`` and what the kind takes --

    process_on | process_off | external   ``channel`` (a logical id)
    set_value                             ``channel`` and ``value`` (a number)
    capture                               ``detector`` (a detector of the perception section),
                                          optional ``out`` (the blackboard key; default
                                          captures.<n>, n counting the task's captures from 0),
                                          ``class_id``, ``timeout_ms``
    dwell                                 ``ms``

    Every event is blocking: ``sync`` may only say ``blocking``."""

    kind: EventKind
    channel: Optional[str] = None
    value: Optional[float] = None
    detector: Optional[str] = None
    out: Optional[str] = None
    class_id: Optional[str] = None
    timeout_ms: Optional[int] = None
    ms: Optional[int] = None
    sync: Optional[str] = None

    _keys = _strict_keys(EVENT_HINTS, 'a process event')
    _channel_ok = _checked('channel', _channel, 'channel')
    _value = _checked('value', _number, 'value')
    _detector = _checked('detector', lambda v, w: v if v is None else _name(v, w), 'detector')
    _out = _checked('out', _capture_key, 'out')
    _class_id = _checked('class_id', lambda v, w: _class_token(v, w), 'class_id')
    _timeout = _checked('timeout_ms', lambda v, w: _count_or_none(v, w, 1), 'timeout_ms')
    _ms = _checked('ms', lambda v, w: _count_or_none(v, w, 1), 'ms')

    @validator('kind', pre=True)
    def _kind(cls, v):  # noqa: N805
        known = [k.value for k in EventKind]
        if v not in known:
            raise ValueError(f'R-M3-02: kind {v!r} is not a process event kind (accepted: '
                             f'{", ".join(known)}): an application is a composition of these, never '
                             'a kind of its own')
        return v

    @validator('sync', pre=True)
    def _sync(cls, v):  # noqa: N805
        if v is None or v == 'blocking':
            return v
        if v == 'async':
            raise ValueError('R-M3-09: sync: async is UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL: every '
                             'process event of this version is blocking (the task waits for it); '
                             'asynchronous events are planned for a later phase -- omit sync, or say '
                             'sync: blocking')
        raise ValueError(f"R-M3-02: sync {v!r}: a process event is 'sync: blocking' (the default)")

    @root_validator(skip_on_failure=True)
    def _fields_of_kind(cls, values):  # noqa: N805
        kind = values.get('kind')
        required, optional = EVENT_FIELDS[EventKind(kind)]
        given = [k for k in _PAYLOAD if values.get(k) is not None]
        missing = [k for k in required if values.get(k) is None]
        extra = [k for k in given if k not in required + optional]
        what = f'a {kind} event'
        if missing:
            raise ValueError(f'R-M3-02: {what} needs {" and ".join(missing)}')
        if extra:
            takes = ', '.join(required + optional)
            raise ValueError(f'R-M3-02: {what} takes no {", ".join(extra)} (it takes {takes})')
        return values

    def describe(self) -> str:
        """The event as the README and the overview say it."""
        k = self.kind
        if k is EventKind.CAPTURE:
            return f'capture by detector `{self.detector}`'
        if k is EventKind.DWELL:
            return f'dwell {self.ms} ms'
        if k is EventKind.SET_VALUE:
            return f'set_value {self.value:g} on channel `{self.channel}`'
        return f'{k.value} on channel `{self.channel}`'


class AnchorWhen(StrEnum):
    AFTER_ARRIVAL = 'after_arrival'
    BEFORE_DEPARTURE = 'before_departure'
    START = 'start'
    END = 'end'


class EventAnchor(_Strict):
    """WHERE on its path an event happens -- a synchronisation point the robot STOPS at:

    ``state``    a state of the path (its waypoint name, or its index), ``when`` after_arrival
                 (the default) or before_departure
    ``segment``  a segment index (0 = from the first state to the second), ``when`` start (= before
                 departing its first state) or end (= after arriving at its last)

    Nothing inside a segment: a fraction, a distance, a time or a position is refused (R-M3-09)."""

    state: Optional[Any] = None
    segment: Optional[int] = None
    when: Optional[AnchorWhen] = None

    # a key placing the event inside a segment is a refusal of its own (R-M3-09), whatever the order
    # pydantic runs validators in
    _keys = _strict_keys({key: _INSIDE.format(where=f'at.{key}', what=what) for key, what in _INSIDE_KEYS.items()},
                         'an event anchor (at)')
    _segment = _checked('segment', lambda v, w: _count_or_none(v, w, 0), 'at.segment')

    @validator('state', pre=True)
    def _state(cls, v):  # noqa: N805
        if v is None or isinstance(v, str):
            return v if v is None else _waypoint_name(v, 'at.state')
        return _count_or_none(v, 'at.state', 0)

    @validator('when', pre=True)
    def _when(cls, v):  # noqa: N805
        if v is None or v in [w.value for w in AnchorWhen]:
            return v
        raise ValueError(f'R-M3-09: at.when {v!r}: a state event happens after_arrival or '
                         'before_departure, a segment event at its start or its end -- nothing inside '
                         'a segment (UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL)')

    @root_validator(skip_on_failure=True)
    def _one_anchor(cls, values):  # noqa: N805
        state, segment, when = values.get('state'), values.get('segment'), values.get('when')
        if (state is None) == (segment is None):
            raise ValueError("R-M3-02: an event's 'at' names exactly one of state (a waypoint of the "
                             'path, or its index) or segment (an index)')
        if state is not None:
            if when is None:
                values['when'] = AnchorWhen.AFTER_ARRIVAL
            elif when not in (AnchorWhen.AFTER_ARRIVAL, AnchorWhen.BEFORE_DEPARTURE):
                raise ValueError(f'R-M3-02: at.when {when.value} belongs to a segment anchor; a state '
                                 'event is after_arrival or before_departure')
        else:
            if when is None:
                raise ValueError('R-M3-02: a segment event says when: start or end')
            if when not in (AnchorWhen.START, AnchorWhen.END):
                raise ValueError(f'R-M3-02: at.when {when.value} belongs to a state anchor; a segment '
                                 'event is at its start or its end')
        return values

    def describe(self) -> str:
        if self.state is not None:
            ref = f"'{self.state}'" if isinstance(self.state, str) else f'state {self.state}'
            return ('after arrival at ' if self.when is AnchorWhen.AFTER_ARRIVAL else
                    'before departure from ') + ref
        return f'at the {self.when.value} of segment {self.segment}'


class PathEvent(ProcessEventSpec):
    """A process event of a path, anchored ``at`` one of its states or segment boundaries."""

    at: EventAnchor

    _keys = _strict_keys(EVENT_HINTS, 'a path event')


STEP_EVENT_HINTS = dict(EVENT_HINTS, at="a process step happens where the task is; 'at' anchors an "
                        'event on a path (paths[].events)')


class ProcessStep(ProcessEventSpec):
    """A task step that is ONE process event, where the robot is (no anchor). A hold is a ``dwell``
    step."""

    _keys = _strict_keys(STEP_EVENT_HINTS, 'a process step')

    @validator('kind')
    def _not_dwell(cls, v):  # noqa: N805
        if v is EventKind.DWELL:
            raise ValueError("R-M3-02: a hold in a task is a dwell step ('dwell: {ms: <ms>}'); kind "
                             'dwell is for an event on a path')
        return v


# --- steps ----------------------------------------------------------------------------------------

class MoveStep(MotionConstraints):
    """ONE motion to a waypoint: ``type`` ptp | lin | circ | free, the waypoint ``to``, a ``profile``
    and/or inline limits (the inline value wins), a circle's ``via`` (or ``center``), and, for a ptp
    or free move only, an explicit ``provider``."""

    to: str
    type: MotionType
    via: Optional[Any] = None
    center: Optional[Any] = None
    profile: Optional[str] = None
    provider: Optional[str] = None

    _keys = _strict_keys(MOVE_HINTS, 'a move step')
    _to = _checked('to', _waypoint_name, 'move.to')
    _via = _checked('via', _aux_point, 'move.via')
    _center = _checked('center', _aux_point, 'move.center')
    _profile = _checked('profile', lambda v, w: v if v is None else _name(v, w), 'move.profile')
    _provider = _checked('provider', known_provider, 'move.provider')

    def constraints(self) -> MotionConstraints:
        """The step's own limits (without the profile), as M3.2's request vocabulary."""
        return MotionConstraints(**{f: getattr(self, f) for f in self.requested()})


class Dwell(_Strict):
    """A hold of ``ms`` milliseconds (BehaviorTree.CPP's Sleep)."""

    ms: int

    _keys = _strict_keys({}, 'a dwell step')

    @validator('ms', pre=True)
    def _ms(cls, v):  # noqa: N805
        v = _number(v, 'dwell.ms', integer=True)
        if v < 1:
            raise ValueError(f'R-M3-07: dwell.ms must be >= 1, got {v!r}')
        return v


# --- manipulation (vNext M3.6, ADR-0026, M3_DESIGN §10) -------------------------------------------

class DetectSource(StrEnum):
    """What a detect step samples: a grasp producer's candidates, or a detector's detections
    projected onto the same candidate contract (approach = -normal; the step's ``approach`` when a
    detection has no fitted normal)."""

    CANDIDATES = 'candidates'
    DETECTIONS = 'detections'


DETECT_HINTS = {
    'trigger': ('a detect step for a grasp takes the producer\'s next FRESH result; an on-demand '
                'capture is a process event (kind: capture) -- a triggered grasp detection is planned '
                'for a later TSA version'),
    'standoff_m': 'the standoff of a grasp belongs to the grasp step',
    'eef': 'the end effector of a grasp belongs to the grasp step',
    'site': 'the grasp site belongs to the grasp step',
    'min_score': ("a candidate's rank is the producer's order: a detect step keeps the first 'keep' "
                  'candidates as published'),
    'topic': ("a detect step names its producer ('detector: <name>' of the perception section), never "
              'a topic: the runtime derives it'),
    'default_approach': "a detections source's fallback approach is 'approach: [x, y, z]'",
}


def _needs(required: Dict[str, str], what: str):
    """A missing REQUIRED key named with what it is for (instead of pydantic's 'Field required').
    Defined before a model's ``_strict_keys``, so it runs after it (pydantic 2 runs 'before' root
    validators in reverse definition order): an unknown key is reported first."""
    def run(cls, values):  # noqa: N805
        # an unknown key is reported first: on pydantic 1.x this validator runs BEFORE _strict_keys
        if isinstance(values, dict) and any(k not in _fields_of(cls) for k in values):
            return values
        if isinstance(values, dict):
            missing = [k for k in required if values.get(k) is None]
            if missing:
                raise ValueError(f'R-M3-02: {what} needs ' + '; '.join(f"'{k}' ({required[k]})" for k in missing))
        return values
    run.__name__ = '_needs'
    return root_validator(pre=True, allow_reuse=True)(run)


def _unit_vector(value: Any, what: str):
    value = _numbers(value, f'{what} [x, y, z]', 3)
    if value is None:
        return value
    norm = math.sqrt(sum(v * v for v in value))
    if abs(norm - 1.0) > UNIT_QUATERNION_TOLERANCE:
        raise ValueError(f'R-M3-02: {what} {value} is not a unit vector (norm {norm:.6g}): the direction '
                         'the tool travels is never normalised silently')
    return [float(v) for v in value]


class DetectStep(_Strict):
    """Sample a grasp producer into a CANDIDATE SET named ``out``, the input of a grasp step.

    ``source: candidates``  the producer publishes grasp candidates (trainit_interfaces
                            GraspCandidateArray) -- any producer: a grasp method, a learned model,
                            a vendor planner, a synthetic publisher
    ``source: detections``  the detector publishes detections, projected onto the same contract:
                            approach = -(the fitted surface normal), and ``approach`` (unit, base
                            axes) for a detection without one
    ``out``: the set's name (default ``candidates``; a grasp step takes the latest set of its task
    unless it names one); ``keep`` (1): how many candidates, in the producer's order (its rank);
    ``class_id`` (detections only), ``timeout_ms`` (default: the perception section's)."""

    detector: str
    source: DetectSource
    out: str = 'candidates'
    keep: int = 1
    class_id: Optional[str] = None
    timeout_ms: Optional[int] = None
    approach: Optional[List[float]] = None

    _required = _needs({'detector': 'the producer: a detector of the perception section',
                        'source': "'candidates' (a grasp producer's) or 'detections' (a detector's, "
                                  'projected onto grasp candidates)'}, 'a detect step')
    _keys = _strict_keys(DETECT_HINTS, 'a detect step')
    _detector = _checked('detector', _name, 'detect.detector')
    _out = _checked('out', _name, 'detect.out')
    _keep = _checked('keep', lambda v, w: _count_or_none(v, w, 1), 'detect.keep')
    _class_id = _checked('class_id', lambda v, w: _class_token(v, w), 'detect.class_id')
    _timeout = _checked('timeout_ms', lambda v, w: _count_or_none(v, w, 1), 'detect.timeout_ms')
    _approach = _checked('approach', _unit_vector, 'detect.approach')

    @validator('source', pre=True)
    def _source(cls, v):  # noqa: N805
        known = [k.value for k in DetectSource]
        if v not in known:
            raise ValueError(f'R-M3-02: detect.source {v!r}: a detect step samples a grasp producer\'s '
                             "'candidates', or a detector's 'detections' (projected onto the same contract)")
        return v

    @root_validator(skip_on_failure=True)
    def _per_source(cls, values):  # noqa: N805
        source = values.get('source')
        if source is DetectSource.CANDIDATES:
            given = [k for k in ('class_id', 'approach') if values.get(k) is not None]
            if given:
                raise ValueError(f'R-M3-02: a detect step with source candidates takes no '
                                 f'{" or ".join(given)}: a grasp producer states the class and the '
                                 'approach in each candidate')
        elif values.get('approach') is None:
            raise ValueError("R-M3-02: a detect step with source detections needs 'approach: [x, y, z]' "
                             '(a unit vector in the base axes): the direction the tool travels when a '
                             'detection has no fitted surface normal (a colour mask has none)')
        return values


class RetractSpec(_Strict):
    """How a grasp or a release leaves: ``along_approach`` (metres back along the approach) or
    ``dz`` (metres up the target frame's +Z)."""

    along_approach: Optional[float] = None
    dz: Optional[float] = None

    _keys = _strict_keys({}, 'a retract')
    _along = _checked('along_approach', lambda v, w: _positive_or_none(v, w), 'retract.along_approach')
    _dz = _checked('dz', lambda v, w: _positive_or_none(v, w), 'retract.dz')

    @root_validator(skip_on_failure=True)
    def _one(cls, values):  # noqa: N805
        if (values.get('along_approach') is None) == (values.get('dz') is None):
            raise ValueError("R-M3-02: a retract is exactly one of 'along_approach: <m>' (back along the "
                             "approach) or 'dz: <m>' (up the target frame's +Z)")
        return values


ORIENTATION_ALIGN_APPROACH = 'align_approach'
_MANIPULATION_HINTS = {
    'pregrasp': ('the PreGrasp is derived, never taught: standoff_m along the approach before the '
                 'target (grasp_pose - standoff * approach)'),
    'approach': ('the approach of a grasp is the candidate\'s (a taught target: its tool +Z); its '
                 "motion profile is 'approach_profile'"),
    'candidate': "a grasp names its candidate set: 'candidates: <detect out>' (and 'index')",
    'type': ('a grasp is a fixed idiom (PreGrasp ptp, Approach lin, close, verify, Retract lin): its '
             "motions are named by 'profile' and 'approach_profile'"),
    'verify': ('a grasp verifies exactly as the end-effector contract requires '
               '(robot.gripper.contract.verification)'),
    'orientation_reference': "the reference of align_approach is 'orientation_from: <waypoint>'",
}


class GraspStep(_Strict):
    """The fixed GRASP idiom: PreGrasp (ptp, ``profile``) -> Approach (lin, ``approach_profile``) ->
    CloseGripper -> VerifyGrasp held (when the contract requires grasp confirmation) -> Retract (lin,
    ``approach_profile``).

    From a candidate set: ``candidates`` (a detect step's ``out`` earlier in the task; default: the
    task's latest detect step) and ``index`` (0, the best); the TCP orientation is ``align_approach`` -- the waypoint ``orientation_from``'s
    orientation tilted as little as possible so the tool's +Z travels along the candidate's approach
    (owner decision M3.6-1); ``standoff_m`` replaces the candidate's own.
    At a taught target: ``at`` (a TCP waypoint); its approach is its tool +Z, so the PreGrasp is
    ``standoff_m`` back along it (``target_local``).
    ``eef`` / ``site``: the device and its grasp site (default: the only ones); ``retract``."""

    candidates: Optional[str] = None
    index: int = 0
    orientation: Optional[str] = None
    orientation_from: Optional[str] = None
    at: Optional[str] = None
    eef: Optional[str] = None
    site: Optional[str] = None
    standoff_m: Optional[float] = None
    profile: str
    approach_profile: str
    retract: RetractSpec

    _required = _needs({'profile': 'the motion profile of the PreGrasp transfer (PTP)',
                        'approach_profile': 'the motion profile of the Approach and the Retract (LIN)',
                        'retract': "{along_approach: <m>} or {dz: <m>}"}, 'a grasp step')
    _keys = _strict_keys(_MANIPULATION_HINTS, 'a grasp step')
    _candidates = _checked('candidates', lambda v, w: v if v is None else _name(v, w), 'grasp.candidates')
    _index = _checked('index', lambda v, w: _count_or_none(v, w, 0), 'grasp.index')
    _from = _checked('orientation_from', lambda v, w: v if v is None else _waypoint_name(v, w),
                     'grasp.orientation_from')
    _at = _checked('at', lambda v, w: v if v is None else _waypoint_name(v, w), 'grasp.at')
    _eef = _checked('eef', lambda v, w: v if v is None else _name(v, w), 'grasp.eef')
    _site = _checked('site', lambda v, w: v if v is None else _name(v, w), 'grasp.site')
    _standoff = _checked('standoff_m', lambda v, w: _positive_or_none(v, w), 'grasp.standoff_m')
    _profile = _checked('profile', _name, 'grasp.profile')
    _approach_profile = _checked('approach_profile', _name, 'grasp.approach_profile')

    @validator('orientation', pre=True)
    def _orientation(cls, v):  # noqa: N805
        if v is None or v == ORIENTATION_ALIGN_APPROACH:
            return v
        raise ValueError(f"R-M3-02: grasp.orientation {v!r}: a candidate's TCP orientation is "
                         "'align_approach' in this version (the reference waypoint orientation_from, "
                         'tilted onto the approach); a producer-owned full orientation is planned for a '
                         'later TSA version')

    @root_validator(skip_on_failure=True)
    def _form(cls, values):  # noqa: N805
        if values.get('candidates') is not None and values.get('at') is not None:
            raise ValueError("R-M3-02: a grasp step is from a candidate set ('candidates: <detect out>', "
                             "default the task's latest detect) or at a taught target ('at: <waypoint>'), "
                             'never both')
        if values.get('at') is not None:
            given = [k for k in ('orientation', 'orientation_from') if values.get(k) is not None]
            if given or values.get('index'):
                raise ValueError('R-M3-02: a grasp at a taught target takes its orientation and approach '
                                 f'from the target itself: no {", ".join(given) or "index"}')
            if values.get('standoff_m') is None:
                raise ValueError("R-M3-02: a grasp at a taught target needs 'standoff_m' (the PreGrasp "
                                 "distance back along the target's tool +Z)")
        elif values.get('orientation_from') is None:
            raise ValueError("R-M3-02: a grasp from candidates needs 'orientation_from: <waypoint>': "
                             'align_approach tilts that taught orientation onto the candidate\'s approach '
                             '(the producer owns the approach, never the rotation about it)')
        return values

    def describe(self) -> str:
        if self.at is not None:
            return f"grasp at '{self.at}'"
        return f"grasp candidate {self.index} of " + (f"'{self.candidates}'" if self.candidates
                                                      else 'the latest detect')


class ReleaseStep(_Strict):
    """The fixed RELEASE idiom at a taught target ``at``: PreRelease (ptp, ``profile``, ``standoff_m``
    back along the target's tool +Z) -> Approach (lin, ``approach_profile``) -> [VerifyGrasp held,
    when the contract confirms grasps and its state is continuous] -> OpenGripper -> VerifyGrasp
    released (when the contract requires release confirmation) -> Retract (lin)."""

    at: str
    eef: Optional[str] = None
    site: Optional[str] = None
    standoff_m: float
    profile: str
    approach_profile: str
    retract: RetractSpec

    _required = _needs({'at': 'the taught target the object is released at',
                        'standoff_m': "the PreRelease distance back along the target's tool +Z",
                        'profile': 'the motion profile of the PreRelease transfer (PTP)',
                        'approach_profile': 'the motion profile of the Approach and the Retract (LIN)',
                        'retract': "{along_approach: <m>} or {dz: <m>}"}, 'a release step')
    _keys = _strict_keys(_MANIPULATION_HINTS, 'a release step')
    _at = _checked('at', _waypoint_name, 'release.at')
    _eef = _checked('eef', lambda v, w: v if v is None else _name(v, w), 'release.eef')
    _site = _checked('site', lambda v, w: v if v is None else _name(v, w), 'release.site')
    _standoff = _checked('standoff_m', lambda v, w: _positive_or_none(v, w), 'release.standoff_m')
    _profile = _checked('profile', _name, 'release.profile')
    _approach_profile = _checked('approach_profile', _name, 'release.approach_profile')

    def describe(self) -> str:
        return f"release at '{self.at}'"


class ToolActuate(_Strict):
    """Command ONE actuator of the device to one of its NAMED targets (ADR-0022 CommandActuator)."""

    eef: Optional[str] = None
    actuator: str
    target: str

    _required = _needs({'actuator': 'the actuator id', 'target': 'one of its named targets'}, 'a tool actuate')
    _keys = _strict_keys({}, 'a tool actuate')
    _eef = _checked('eef', lambda v, w: v if v is None else _name(v, w), 'tool.actuate.eef')
    _actuator = _checked('actuator', _name, 'tool.actuate.actuator')
    _target = _checked('target', _name, 'tool.actuate.target')


class GraspExpectation(StrEnum):
    HELD = 'held'
    RELEASED = 'released'


class ToolVerify(_Strict):
    """Verify the TOOL's grasp (ADR-0022 VerifyGrasp): ``expect`` held | released, at ``site`` or
    at tool level."""

    eef: Optional[str] = None
    site: Optional[str] = None
    expect: GraspExpectation

    _required = _needs({'expect': 'held | released'}, 'a tool verify')
    _keys = _strict_keys({}, 'a tool verify')
    _eef = _checked('eef', lambda v, w: v if v is None else _name(v, w), 'tool.verify.eef')
    _site = _checked('site', lambda v, w: v if v is None else _name(v, w), 'tool.verify.site')

    @validator('expect', pre=True)
    def _expect(cls, v):  # noqa: N805
        if v not in [k.value for k in GraspExpectation]:
            raise ValueError(f"R-M3-02: tool.verify.expect {v!r}: a grasp verification expects 'held' or "
                             "'released'")
        return v


TOOL_HINTS = {
    'open': ("a grip is opened and closed by the grasp and release steps; another actuator is "
             "'actuate: {actuator: <id>, target: <named target>}'"),
    'close': ("a grip is opened and closed by the grasp and release steps; another actuator is "
              "'actuate: {actuator: <id>, target: <named target>}'"),
    'command': "an end-effector command is 'actuate: {actuator: <id>, target: <named target>}'",
}


class ToolStep(_Strict):
    """An end-effector operation: exactly one of ``actuate`` or ``verify``."""

    actuate: Optional[ToolActuate] = None
    verify: Optional[ToolVerify] = None

    _keys = _strict_keys(TOOL_HINTS, 'a tool step')

    @root_validator(skip_on_failure=True)
    def _one(cls, values):  # noqa: N805
        if (values.get('actuate') is None) == (values.get('verify') is None):
            raise ValueError("R-M3-02: a tool step is exactly one of 'actuate' or 'verify'")
        return values


# --- composition (vNext M3.7, M3_DESIGN §4, M3_SCHEMA_PROPOSAL §1.1) --------------------------------
#
# A task CALLS another task (``task: <name>``); the three CONTROL steps wrap a BODY of steps:
# ``repeat`` (BehaviorTree.CPP Repeat), ``retry`` (RetryUntilSuccessful), ``fallback`` (Fallback) --
# BehaviorTree.CPP's own nodes, no control node of TrainIt's (ADR-0016 Decision 2). They are
# ORCHESTRATION only: a step inside a body is the same step, validated and generated as anywhere else.
# There is no condition, variable or expression: what decides is what the steps observe (a detection,
# a verification, a motion that cannot reach its target).

class TaskDataType(StrEnum):
    """What a task can take or give: the data the steps of this version put on the blackboard."""

    CANDIDATES = 'candidates'      # a detect step's candidate set (`out`), read by a grasp step
    CAPTURE = 'capture'            # a capture's result (process kind capture, its `out` key)


class TaskData(_Strict):
    """ONE input or output of a task: the NAME a step of the program gives the data (a detect step's
    ``out``, a capture's ``out``) and its ``type``. Data is bound BY NAME: the application has one
    blackboard, so a task's input is the data of that name its caller holds when it calls it, and its
    output is the data of that name it leaves for the steps after the call."""

    name: str
    type: TaskDataType
    # In this version a grasp step is the only step that READS task data (a candidate set): a capture
    # is an output only -- a capture input is refused when the program is checked (no step reads it).

    _keys = _strict_keys({'kind': "the type of a task's data is 'type: candidates | capture'",
                          'key': "a task's data is named 'name: <the step's out>'",
                          'from': ("a task's input is the data of the same name its caller holds "
                                   '(the application has one blackboard): name it as the caller names it'),
                          'default': 'a task input has no default: its caller provides it, or the call is refused'},
                         "a task's input or output")

    @validator('type', pre=True)
    def _type(cls, v):  # noqa: N805
        known = [k.value for k in TaskDataType]
        if v not in known:
            raise ValueError(f"R-M3-02: type {v!r}: a task's input or output is a candidate set ('candidates', "
                             "a detect step's out) or a capture result ('capture', a capture's out)")
        return v

    @root_validator(skip_on_failure=True)
    def _named(cls, values):  # noqa: N805
        name, kind = values.get('name'), values.get('type')
        if kind is TaskDataType.CAPTURE:
            _capture_key(name, 'capture name')
        else:
            _name(name, 'candidate set name')
        return values

    def describe(self) -> str:
        return f'{self.type.value} `{self.name}`'


def _body(value: Any, what: str):
    if not isinstance(value, list) or not value:
        raise ValueError(f'R-M3-02: {what} must be a non-empty list of steps')
    return value


REPEAT_HINTS = {
    'times': "a repeat runs its body 'cycles: <n>' times (-1: until a step fails or the program stops)",
    'count': "a repeat runs its body 'cycles: <n>' times (-1: until a step fails or the program stops)",
    'num_cycles': "a repeat runs its body 'cycles: <n>' times (-1: until a step fails or the program stops)",
    'steps': "the steps a repeat runs are its 'body: [<steps>]'",
    'until': ('a repeat has no condition: it runs its body a number of times, and stops early when a '
              'step of the body fails'),
}
RETRY_HINTS = {
    'retries': "a retry tries its body at most 'attempts: <n>' times, the first try included",
    'times': "a retry tries its body at most 'attempts: <n>' times, the first try included",
    'num_attempts': "a retry tries its body at most 'attempts: <n>' times, the first try included",
    'steps': "the steps a retry runs are its 'body: [<steps>]'",
    'step': "a retry wraps a body of steps: 'body: [<step>, ...]'",
}
FALLBACK_HINTS = {
    'body': "a fallback tries alternative bodies in order: 'bodies: [[<steps>], [<steps>], ...]'",
    'steps': "a fallback tries alternative bodies in order: 'bodies: [[<steps>], [<steps>], ...]'",
    'alternatives': "a fallback's alternatives are its 'bodies: [[<steps>], [<steps>], ...]'",
}


class RepeatStep(_Strict):
    """Run ``body`` (a list of steps) ``cycles`` times, one after the other. A step of the body that
    fails ends the repeat with that failure (the cycles left do not run). ``cycles: -1`` repeats until
    a step fails or the program is stopped (a production cycle)."""

    cycles: int
    body: List['TaskStep']

    _keys = _strict_keys(REPEAT_HINTS, 'a repeat')
    _body_ok = _checked('body', _body, 'repeat.body')

    @validator('cycles', pre=True)
    def _cycles(cls, v):  # noqa: N805
        v = _number(v, 'repeat.cycles', integer=True)
        if v is None or (v != -1 and v < 1):
            raise ValueError(f'R-M3-07: repeat.cycles must be a number of cycles >= 1, or -1 (until a step '
                             f'fails or the program stops), got {v!r}')
        return v


class RetryStep(_Strict):
    """Run ``body`` (a list of steps); when one of its steps fails, run the WHOLE body again from its
    first step, ``attempts`` tries at most (the first included). The retry succeeds on the first try
    whose body succeeds; when every try failed, it fails with the last failure."""

    attempts: int
    body: List['TaskStep']

    _keys = _strict_keys(RETRY_HINTS, 'a retry')
    _body_ok = _checked('body', _body, 'retry.body')

    @validator('attempts', pre=True)
    def _attempts(cls, v):  # noqa: N805
        v = _number(v, 'retry.attempts', integer=True)
        if v is None or v < 1:
            raise ValueError(f'R-M3-07: retry.attempts must be a number of tries >= 1 (the first try '
                             f'included), got {v!r}')
        return v


class FallbackStep(_Strict):
    """Try ``bodies`` (two or more lists of steps, the ALTERNATIVES) in order: the first body runs; if
    one of its steps fails, the next body runs from its first step; the fallback succeeds with the
    first body that succeeds and fails when every body failed. A body never changes the steps of
    another."""

    bodies: List[List['TaskStep']]

    _keys = _strict_keys(FALLBACK_HINTS, 'a fallback')

    @validator('bodies', pre=True)
    def _bodies(cls, v):  # noqa: N805
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError('R-M3-02: fallback.bodies must be a list of at least 2 alternative bodies, each a '
                             'list of steps (one body alone is not a fallback: write its steps directly)')
        for i, body in enumerate(v):
            _body(body, f'fallback.bodies[{i}]')
        return v


STEP_KINDS = ('move', 'path', 'process', 'dwell', 'detect', 'grasp', 'release', 'tool', 'task', 'repeat',
              'retry', 'fallback')
CONTROL_KINDS = ('repeat', 'retry', 'fallback')


def _task_call(value: Any, what: str):
    if isinstance(value, dict):
        raise ValueError("R-M3-02: a task call is 'task: <name>' (the task's name): its inputs and outputs "
                         'are bound by NAME -- the caller holds the data under the names the called task '
                         'declares')
    return value if value is None else _name(value, what)


class TaskStep(_Strict):
    """One step of a task: exactly one of ``move``, ``path`` (a path's name), ``process``,
    ``dwell``, ``detect``, ``grasp``, ``release``, ``tool``, ``task`` (a task's name: a call),
    ``repeat``, ``retry`` or ``fallback``."""

    move: Optional[MoveStep] = None
    path: Optional[str] = None
    process: Optional[ProcessStep] = None
    dwell: Optional[Dwell] = None
    detect: Optional[DetectStep] = None
    grasp: Optional[GraspStep] = None
    release: Optional[ReleaseStep] = None
    tool: Optional[ToolStep] = None
    task: Optional[str] = None
    repeat: Optional[RepeatStep] = None
    retry: Optional[RetryStep] = None
    fallback: Optional[FallbackStep] = None

    _keys = _strict_keys(STEP_HINTS, 'a step')
    _path = _checked('path', lambda v, w: v if v is None else _name(v, w), 'path step')
    _task = _checked('task', _task_call, 'task call')

    @root_validator(skip_on_failure=True)
    def _exactly_one(cls, values):  # noqa: N805
        kinds = [k for k in STEP_KINDS if values.get(k) is not None]
        if len(kinds) != 1:
            raise ValueError('R-M3-02: a step is exactly one of ' + ', '.join(STEP_KINDS[:-1]) + ' or '
                             + STEP_KINDS[-1] + (f' (got {", ".join(kinds)})' if kinds else ''))
        return values

    def kind(self) -> str:
        return next(k for k in STEP_KINDS if getattr(self, k) is not None)

    def bodies(self) -> List[List['TaskStep']]:
        """The bodies a control step wraps (one for repeat and retry, the alternatives of a fallback);
        none for any other step."""
        if self.repeat is not None:
            return [self.repeat.body]
        if self.retry is not None:
            return [self.retry.body]
        if self.fallback is not None:
            return list(self.fallback.bodies)
        return []


for _model in (RepeatStep, RetryStep, FallbackStep):
    # the bodies hold TaskSteps (a forward reference): resolved now, on pydantic 1.x and 2.x
    if hasattr(_model, 'model_rebuild'):
        _model.model_rebuild()
    else:
        _model.update_forward_refs()


def walk_steps(steps: List[TaskStep], label: str = ''):
    """Every step of a list of steps, the steps of the control bodies included, in program order, as
    ``(label, step)``: ``3`` is the third step, ``3.2`` the second step of the body of a repeat or a
    retry at step 3, ``3.2.1`` the first step of the SECOND alternative of a fallback at step 3."""
    for i, step in enumerate(steps, start=1):
        here = f'{label}.{i}' if label else str(i)
        yield here, step
        if step.fallback is not None:
            for b, body in enumerate(step.fallback.bodies, start=1):
                yield from walk_steps(body, f'{here}.{b}')
        else:
            for body in step.bodies():
                yield from walk_steps(body, here)


TASK_DESCRIPTION = re.compile(r'^[^\n\r]{1,200}$')


class Task(_Strict):
    """A named, ordered list of steps (one BehaviorTree, ``task_<name>``), with an optional one-line
    ``description`` and the data it takes (``inputs``) and gives (``outputs``)."""

    name: str
    description: Optional[str] = None
    inputs: List[TaskData] = []
    outputs: List[TaskData] = []
    steps: List[TaskStep]

    _keys = _strict_keys(TASK_HINTS, 'a task')
    _name_ok = _checked('name', _name, 'task name')

    @validator('description', pre=True)
    def _description(cls, v):  # noqa: N805
        if v is not None and (not isinstance(v, str) or not TASK_DESCRIPTION.fullmatch(v) or not v.strip()
                              or not v.isprintable()):
            raise ValueError(f'R-M3-02: a task description is one line of text (1..200 characters), got {v!r}')
        return v

    @validator('steps')
    def _not_empty(cls, v, values):  # noqa: N805
        if not v:
            raise ValueError(f"R-M3-02: task '{values.get('name')}' has no steps")
        return v

    @root_validator(skip_on_failure=True)
    def _data_once(cls, values):  # noqa: N805
        for what in ('inputs', 'outputs'):
            seen = set()
            for d in values.get(what) or []:
                if d.name in seen:
                    raise ValueError(f"R-M3-02: task '{values.get('name')}' declares {what[:-1]} '{d.name}' twice")
                seen.add(d.name)
        return values

    def calls(self) -> List[Tuple[str, str]]:
        """The tasks this task calls, as ``(label, task)`` in program order."""
        return [(label, s.task) for label, s in walk_steps(self.steps) if s.task is not None]


# --- paths ----------------------------------------------------------------------------------------

class PathSegment(MotionConstraints):
    """The relation between two consecutive states of a path: ``type`` lin | circ (a circ names
    ``via`` or ``center``), an optional ``profile`` and inline limits, and the ``blend_radius_m`` into
    the next segment (a ``continuity: blend`` path only; the last segment never blends)."""

    type: MotionType
    via: Optional[Any] = None
    center: Optional[Any] = None
    profile: Optional[str] = None

    _keys = _strict_keys(SEGMENT_HINTS, 'a path segment')
    _via = _checked('via', _aux_point, 'segment.via')
    _center = _checked('center', _aux_point, 'segment.center')
    _profile = _checked('profile', lambda v, w: v if v is None else _name(v, w), 'segment.profile')

    def constraints(self) -> MotionConstraints:
        return MotionConstraints(**{f: getattr(self, f) for f in self.requested()})


GENERATORS = {
    # type -> (required, optional): TMR's generators (path_library.cpp denseSource)
    'polyline': (('points',), ('closed',)),
    'circle': (('center', 'radius'), ('segments',)),
    'rectangle': (('center', 'size_x', 'size_y'), ()),
    'raster': (('center', 'size_x', 'size_y', 'rows'), ()),
}


class PathGenerator(_Strict):
    """One of TMR's path generators (a DENSE path: its samples are the states, joined by lin)."""

    type: str
    points: Optional[List[Any]] = None
    closed: Optional[bool] = None
    center: Optional[List[float]] = None
    radius: Optional[float] = None
    segments: Optional[int] = None
    size_x: Optional[float] = None
    size_y: Optional[float] = None
    rows: Optional[int] = None

    _keys = _strict_keys({}, 'a path generator')
    _center = _checked('center', lambda v, w: _numbers(v, w, 3), 'generator.center [x, y, z]')
    _radius = _checked('radius', lambda v, w: _positive_or_none(v, w), 'generator.radius')
    _size_x = _checked('size_x', lambda v, w: _positive_or_none(v, w), 'generator.size_x')
    _size_y = _checked('size_y', lambda v, w: _positive_or_none(v, w), 'generator.size_y')
    _segments = _checked('segments', lambda v, w: _count_or_none(v, w, 3), 'generator.segments')
    _rows = _checked('rows', lambda v, w: _count_or_none(v, w, 1), 'generator.rows')

    @validator('points', pre=True)
    def _points(cls, v):  # noqa: N805
        if v is None:
            return v
        if not isinstance(v, list) or len(v) < 2:
            raise ValueError('R-M3-02: generator.points must be a list of at least 2 [x, y, z]')
        return [[float(c) for c in _numbers(p, 'generator.points[] [x, y, z]', 3)] for p in v]

    @validator('closed', pre=True)
    def _closed(cls, v):  # noqa: N805
        if v is not None and not isinstance(v, bool):
            raise ValueError(f'R-M3-02: generator.closed must be true or false, got {v!r}')
        return v

    @root_validator(skip_on_failure=True)
    def _parameters(cls, values):  # noqa: N805
        kind = values.get('type')
        if kind not in GENERATORS:
            raise ValueError(f"R-M3-02: unknown generator type {kind!r} (accepted: "
                             f"{', '.join(GENERATORS)})")
        required, optional = GENERATORS[kind]
        given = {k for k in _fields_of(cls) if k != 'type' and values.get(k) is not None}
        missing = [k for k in required if k not in given]
        extra = sorted(given - set(required) - set(optional))
        if missing:
            raise ValueError(f"R-M3-02: a {kind} generator needs {', '.join(missing)}")
        if extra:
            raise ValueError(f"R-M3-02: {', '.join(extra)} is not a parameter of a {kind} generator "
                             f"(it takes {', '.join(required + optional)})")
        return values


def _positive_or_none(value, what):
    value = _number(value, what)
    if value is not None and value <= 0:
        raise ValueError(f'R-M3-07: {what} must be > 0, got {value!r}')
    return value


def _count_or_none(value, what, least):
    value = _number(value, what, integer=True)
    if value is not None and value < least:
        raise ValueError(f'R-M3-07: {what} must be >= {least}, got {value!r}')
    return value


class DenseSource(_Strict):
    """A dense path's samples: a ``generator`` and the TCP ``orientation`` every sample keeps."""

    generator: PathGenerator
    orientation: List[float]

    _keys = _strict_keys(SOURCE_HINTS, 'a dense path source')
    _orientation = _checked('orientation', _quaternion, 'source.orientation')


class PathDef(MotionConstraints):
    """A named PATH: ONE motion through ``states`` (waypoint names, the first is where the path
    starts) joined by ``segments`` (one per pair of consecutive states), with a ``continuity`` (stop
    at every state, the default, or blend through them), a ``profile`` and/or inline limits, a
    path-wide Cartesian ``goal_tolerance``, and optional ``provider``, ``frame`` and ``tcp``
    overrides. A DENSE path has a ``source`` instead of states and segments. ``events``: the process
    events anchored at its states or segment boundaries (vNext M3.5; a segment path only)."""

    name: str
    states: Optional[List[str]] = None
    segments: Optional[List[PathSegment]] = None
    continuity: Optional[PathContinuity] = None
    profile: Optional[str] = None
    provider: Optional[str] = None
    frame: Optional[str] = None
    tcp: Optional[str] = None
    source: Optional[DenseSource] = None
    events: List[PathEvent] = []

    _keys = _strict_keys(PATH_HINTS, 'a path')
    _name_ok = _checked('name', _name, 'path name')
    _profile = _checked('profile', lambda v, w: v if v is None else _name(v, w), 'path.profile')
    _provider = _checked('provider', known_provider, 'path.provider')

    @validator('states', pre=True)
    def _states(cls, v):  # noqa: N805
        if v is None:
            return v
        if not isinstance(v, list):
            raise ValueError('R-M3-02: states must be a list of waypoint names')
        for s in v:
            if not isinstance(s, str):
                raise ValueError(f'R-M3-02: a path state is a waypoint name, got {s!r} (an inline '
                                 'pose: define it as a waypoint)')
            _waypoint_name(s, 'path state')
        return v

    @root_validator(skip_on_failure=True)
    def _shape(cls, values):  # noqa: N805
        where = f"path '{values.get('name')}'"
        dense = values.get('source') is not None
        if dense and (values.get('states') is not None or values.get('segments') is not None):
            raise ValueError(f'R-M3-04: {where} is either states + segments, or a dense source')
        if not dense:
            states, segments = values.get('states'), values.get('segments')
            if states is None or segments is None:
                raise ValueError(f'R-M3-04: {where} needs states (>= 2 waypoint names) and segments '
                                 '(one per pair of consecutive states), or a dense source')
            if len(states) < 2:
                raise ValueError(f'R-M3-04: {where} needs at least 2 states (its start and one more), '
                                 f'got {len(states)}')
            if len(segments) != len(states) - 1:
                raise ValueError(f'R-M3-04: {where}: {len(states)} states need {len(states) - 1} '
                                 f'segments, got {len(segments)}')
        for key in ('frame', 'tcp'):
            if values.get(key) is not None and not values[key]:
                raise ValueError(f'R-M3-02: {where}: {key} is empty (omit it for the default)')
        return values

    def constraints(self) -> MotionConstraints:
        return MotionConstraints(**{f: getattr(self, f) for f in self.requested()})


# --- the program ----------------------------------------------------------------------------------

class RuntimeSettings(_Strict):
    """Bundle-level runtime settings. ``allowed_start_tolerance`` (rad): how far the robot's state
    may be from a planned trajectory's first point before MoveIt refuses to execute it (MoveIt's
    ``trajectory_execution.allowed_start_tolerance``; 0 disables the check)."""

    allowed_start_tolerance: Optional[float] = None

    _keys = _strict_keys({}, 'application.runtime')
    _tol = _checked('allowed_start_tolerance', _non_negative, 'runtime.allowed_start_tolerance')


APPLICATION_HINT = ("'{key}' belongs to the template applications (pick_and_place, "
                    'vision_guided_motion, bin_picking); a Canonical Task Program says the same with '
                    'waypoints, paths and tasks')
PROGRAM_HINTS = {
    'global_planner_mode': ("a task program chooses providers per kind of motion: "
                            "application.providers (ptp, free, path, dense_path)"),
    'process_path_backend': ("a task program's runner backend is derived by TSA from the providers "
                             'its paths resolve to'),
    'segments': ('a task program has no folded segments: a move step says how it reaches its '
                 'waypoint, a path its segments'),
    'sequence': "a task program's order is the steps of its tasks (main_task runs first)",
}
for _key in ('tool_actions', 'detections', 'policies', 'bin_picking', 'loop_cycles', 'loop_start',
             'loop_end', 'bt_tree_id', 'process_controller', 'enforce_validation'):
    PROGRAM_HINTS[_key] = APPLICATION_HINT.format(key=_key)


class TaskProgram(_Strict):
    """The Canonical Task Program of a ``task_program`` application (module docstring)."""

    providers: Optional[ProviderChoices] = None
    motion_profiles: Dict[str, MotionProfile] = {}
    waypoints: List[TaskWaypoint] = []
    paths: List[PathDef] = []
    tasks: List[Task]
    main_task: str
    runtime: Optional[RuntimeSettings] = None

    _keys = _strict_keys(PROGRAM_HINTS, 'a task program (application)')
    _main = _checked('main_task', _name, 'main_task')

    @validator('motion_profiles')
    def _profile_names(cls, v):  # noqa: N805
        for name in v:
            _name(name, 'motion profile name')
            if name in RESERVED_PROFILE_NAMES:
                raise ValueError(f"R-M3-02: motion profile name '{name}' is reserved: TMR's own base "
                                 'profile has it; choose another name')
        return v

    @root_validator(skip_on_failure=True)
    def _unique_and_main(cls, values):  # noqa: N805
        for what, items in (('waypoint', values.get('waypoints') or []),
                            ('path', values.get('paths') or []),
                            ('task', values.get('tasks') or [])):
            seen = set()
            for item in items:
                if item.name in seen:
                    raise ValueError(f"R-M3-15: two {what}s are named '{item.name}'")
                seen.add(item.name)
        tasks = [t.name for t in values.get('tasks') or []]
        if not tasks:
            raise ValueError('R-M3-15: a task program needs at least one task')
        if values.get('main_task') not in tasks:
            raise ValueError(f"R-M3-15: main_task '{values.get('main_task')}' is not a task of this "
                             f"program (tasks: {', '.join(tasks)})")
        return values


# The application keys of a task program, in the order a project file writes them.
PROGRAM_KEYS = tuple(_fields_of(TaskProgram))


# --- load errors in the file's own terms ---------------------------------------------------------

_NAMED_LISTS = {'waypoints': 'waypoint', 'paths': 'path', 'tasks': 'task'}
_PYDANTIC_PREFIXES = ('Value error, ', 'Assertion failed, ')
_PYDANTIC_EXTRA = ('extra fields not permitted', 'Extra inputs are not permitted')


def _place(loc, data) -> str:
    """pydantic's location of an error as the file names it: ``task 'scan' step 3 (move)``,
    ``waypoint 'w2'``, ``path 'sweep' segment 1``, ``motion profile 'approach'``."""
    parts, node, i = [], data, 0
    label, at = '', None                     # the step's label (3, 3.2, 3.2.1) and its place in parts
    loc = [x for x in loc if x not in ('program', '__root__')]
    while i < len(loc):
        key = loc[i]
        child = node.get(key) if isinstance(node, dict) else None
        # vNext M3.7: a step of a control body is named by its label, as the planner names it
        nested = 1 if key == 'body' else 2 if key == 'bodies' else 0
        if nested and label and i + nested < len(loc) and all(isinstance(x, int) for x in loc[i + 1:i + 1 + nested]):
            idx = loc[i + 1:i + 1 + nested]
            step = child
            for x in idx:
                step = step[x] if isinstance(step, list) and x < len(step) else None
            kind = next(iter(step), None) if isinstance(step, dict) and len(step) == 1 else None
            label = '.'.join([label] + [str(x + 1) for x in idx])
            parts[at] = f'step {label}' + (f' ({kind})' if kind else '')
            node, i = (step.get(kind) if kind else step), i + 1 + nested
            if i < len(loc) and loc[i] == kind:
                i += 1
            continue
        if key in _NAMED_LISTS and i + 1 < len(loc) and isinstance(loc[i + 1], int):
            item = (child or [None] * (loc[i + 1] + 1))[loc[i + 1]] if isinstance(child, list) \
                and loc[i + 1] < len(child) else None
            name = item.get('name') if isinstance(item, dict) else None
            parts.append(f"{_NAMED_LISTS[key]} '{name}'" if name else
                         f'{_NAMED_LISTS[key]} #{loc[i + 1] + 1}')
            node, i = item, i + 2
            continue
        if key == 'steps' and i + 1 < len(loc) and isinstance(loc[i + 1], int):
            step = child[loc[i + 1]] if isinstance(child, list) and loc[i + 1] < len(child) else None
            kind = next(iter(step), None) if isinstance(step, dict) and len(step) == 1 else None
            parts.append(f'step {loc[i + 1] + 1}' + (f' ({kind})' if kind else ''))
            label, at = str(loc[i + 1] + 1), len(parts) - 1
            node, i = (step.get(kind) if kind else step), i + 2
            if i < len(loc) and loc[i] == kind:
                i += 1
            continue
        if key == 'segments' and i + 1 < len(loc) and isinstance(loc[i + 1], int):
            parts.append(f'segment {loc[i + 1]}')
            node = child[loc[i + 1]] if isinstance(child, list) and loc[i + 1] < len(child) else None
            i += 2
            continue
        if key == 'motion_profiles' and i + 1 < len(loc):
            parts.append(f"motion profile '{loc[i + 1]}'")
            node = child.get(loc[i + 1]) if isinstance(child, dict) else None
            i += 2
            continue
        parts.append(str(key))
        node, i = child, i + 1
    return ' '.join(parts) if parts else 'the project'


def describe_load_error(exc: Exception, data: dict, path) -> str:
    """A task program's load refusal, one line per problem, each located in the file's terms."""
    errors = getattr(exc, 'errors', None)
    if not callable(errors):
        return f'{path}: {exc}'
    lines = []
    for err in errors():
        msg = str(err.get('msg', ''))
        for prefix in _PYDANTIC_PREFIXES:
            if msg.startswith(prefix):
                msg = msg[len(prefix):]
        loc = list(err.get('loc', ()))
        if msg in _PYDANTIC_EXTRA and loc:
            msg = f"R-M3-02: unknown key '{loc[-1]}'"
            loc = loc[:-1]
        place = _place(loc, data)
        lines.append(f'{place}: {msg}' if place != 'application' or not msg.startswith('R-M3-01')
                     else msg)
    unique = list(dict.fromkeys(lines))
    head = f'{path}: the Canonical Task Program cannot be loaded'
    return head + (':\n  ' if len(unique) > 1 else ': ') + '\n  '.join(unique)

__all__ = ['AnchorWhen', 'CAPTURE_KEY', 'CHANNEL_KINDS', 'DetectSource', 'DetectStep', 'Dwell', 'DenseSource',
           'EventAnchor', 'EventKind', 'GraspExpectation', 'GraspStep', 'ORIENTATION_ALIGN_APPROACH',
           'ReleaseStep', 'RetractSpec', 'ToolActuate', 'ToolStep', 'ToolVerify',
           'GENERATORS', 'describe_load_error', 'MoveStep', 'NAME', 'PROGRAM_KEYS', 'PathDef',
           'PathEvent', 'PathGenerator', 'PathSegment', 'ProcessEventSpec', 'ProcessStep', 'RelativeFrame',
           'RelativeTarget', 'RuntimeSettings', 'STEP_KINDS', 'Task', 'TaskProgram', 'TaskStep',
           'TaskWaypoint', 'CONTROL_KINDS', 'FallbackStep', 'RepeatStep', 'RetryStep', 'TaskData',
           'TaskDataType', 'walk_steps']
