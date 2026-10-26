"""Check the waypoints against the live MoveIt robot (vNext I0, owner Test C): before Generate, with the
planning environment running.

It COMPUTES only, through MoveIt's services (IK, FK, state validity): nothing plans, nothing moves
(ADR-0027). It answers what no offline check can:
- a TCP or relative waypoint has a collision-free joint solution -- or it is out of reach, or reachable
  only in collision (a relative target 200 mm "down" from a low pose lands inside the table);
- a joint or named waypoint is collision-free;
- a joint waypoint reaches the pose it stands for: the start of the path a PTP to it leads into (TMR
  refuses a path start further than its tolerance: NOT_AT_PATH_START), and the TCP waypoint ``X`` of a
  pair ``X`` / ``X_j`` made by "Capture TCP + joints".

Pure Python: the three computations are given as callables (``livesession.kinematics`` for ROS), so the
logic is testable without a ROS graph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..applications.task_program import PATH_START_TOLERANCE, Target, resolve_targets
from ..model.enums import MotionType
from ..model.program import CONTROL_KINDS
from ..model.project import CanonicalProject

Pose = Tuple[Sequence[float], Sequence[float]]                    # (position, orientation qx qy qz qw)
IK = Callable[[str, Sequence[float], Sequence[float], Optional[Sequence[float]], bool], Optional[List[float]]]
FK = Callable[[Sequence[float], str], Pose]                       # joints, frame -> the tip pose in it
VALID = Callable[[Sequence[float]], Tuple[bool, str]]             # joints -> (collision-free, the contacts)

_MOTION = ('move', 'path', 'grasp', 'release', 'task')            # task_program._first_path's
JOINT_SUFFIX = '_j'                                               # "Capture TCP + joints": X and X_j


class NotDecided(Exception):
    """Raised by a computation that cannot answer for THIS waypoint (MoveIt refuses the request: an unknown
    frame, group or link): the line is '?' with the reason, the other waypoints are still checked."""


@dataclass(frozen=True)
class CheckLine:
    ok: Optional[bool]                     # None: not decided (the waypoint does not resolve)
    waypoint: str                          # what a double click selects
    text: str

    def __str__(self) -> str:
        return {True: '✔', False: '✖', None: '?'}[self.ok] + ' ' + self.text


def _mm(p: Sequence[float]) -> str:
    return '(' + ', '.join(f'{v * 1000.0:.1f}' for v in p) + ') mm'


def _angle(a: Sequence[float], b: Sequence[float]) -> float:
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2.0 * math.acos(min(1.0, dot))


def named_state_values(project: CanonicalProject) -> Dict[str, Dict[str, float]]:
    """The planning group's named states, ``{name: {joint: value}}``: the base MoveIt config's SRDF when the
    project has one (what the bundle's SRDF will hold), the project's own named states merged over it --
    of THIS group only (an SRDF often has a 'home' for the gripper too)."""
    robot = project.robot
    group = robot.planning_group.name
    out: Dict[str, Dict[str, float]] = {}
    if robot.base_moveit_config_path:
        from pathlib import Path
        from ..robotmodel.srdf_reader import read_srdf
        srdfs = sorted((Path(robot.base_moveit_config_path) / 'config').glob('*.srdf'))
        try:
            if srdfs:
                out = {st.name: dict(st.joint_values) for st in read_srdf(srdfs[0].read_text()).states_for(group)}
        except Exception:  # noqa: BLE001 - unreadable: the project's own states only
            out = {}
    for st in robot.named_states:
        if not st.group or st.group == group:
            out[st.name] = dict(st.joint_values)
    return out


def _joint_values(project: CanonicalProject, t: Target, named: Dict[str, Dict[str, float]]) -> Optional[List[float]]:
    joints = list(project.robot.planning_group.joints)
    if t.joints is not None:
        return list(t.joints)
    values = named.get(t.named)
    if values is None or any(j not in values for j in joints):
        return None
    return [values[j] for j in joints]


def _path_starts(project: CanonicalProject) -> List[Tuple[str, str, str]]:
    """``(joint waypoint, path, its first state)`` for every PTP / free move to a joint waypoint followed
    by a path step, in every step list of every task."""
    program = project.application.program
    first = {p.name: p.states[0] for p in program.paths if p.states}
    out: List[Tuple[str, str, str]] = []

    def next_motion(steps, i):
        # the next step that moves the robot or may (as the planner's own path-start check reads it:
        # task_program._first_path); a process event, a dwell or a detect does not
        for step in steps[i + 1:]:
            if step.kind() in _MOTION or step.kind() in CONTROL_KINDS:
                return step
        return None

    def walk(steps) -> None:
        for i, step in enumerate(steps):
            nxt = next_motion(steps, i)
            m = step.move
            if (m is not None and m.type in (MotionType.PTP, MotionType.FREE) and nxt is not None
                    and nxt.path is not None and nxt.path in first):
                out.append((m.to, nxt.path, first[nxt.path]))
            for body in (step.fallback.bodies if step.fallback is not None else step.bodies()):
                walk(body)

    for task in program.tasks:
        walk(task.steps)
    return list(dict.fromkeys(out))


MAX_SEEDS = 4


def _seeds(candidates) -> list:
    """At most MAX_SEEDS distinct start configurations, in order; None (move_group's current state) last."""
    out: list = []
    for c in candidates:
        if c is not None and all(c != o for o in out) and len([o for o in out if o is not None]) < MAX_SEEDS - 1:
            out.append(list(c))
    return out + [None]


def check_waypoints(project: CanonicalProject, ik: IK, fk: FK, valid: VALID,
                    scene_objects: Optional[Callable[[], List[str]]] = None) -> List[CheckLine]:
    """One line per waypoint, then one per joint waypoint that stands for a pose. ``scene_objects``: the
    ids MoveIt's planning scene holds; a cell object missing from it makes every collision answer partial
    (found on a real move_group: checked before its scene manager had loaded the table, a pose inside the
    table passed)."""
    robot = project.robot
    targets, problems = resolve_targets(project)
    named = named_state_values(project)
    out: List[CheckLine] = []
    if scene_objects is not None:
        expected = [o.id for o in project.scene.objects]
        missing = [i for i in expected if i not in set(scene_objects())]
        if missing:
            out.append(CheckLine(None, '', f"the planning scene does not hold {', '.join(missing)} yet: the "
                                           'collision answers below ignore it -- wait for the environment to load '
                                           'the cell, then check again'))
    joint_values: Dict[str, List[float]] = {}
    for wp in project.application.program.waypoints:
        t = targets.get(wp.name)
        if t is None:
            out.append(CheckLine(None, wp.name, f"'{wp.name}': not resolved (see Diagnostics)"))
            continue
        if t.type == 'joint':
            values = _joint_values(project, t, named)
            if values is None:
                out.append(CheckLine(None, wp.name, f"'{wp.name}': named state '{t.named}' has no value for "
                                                    f"every joint of planning group '{robot.planning_group.name}'"))
                continue
            joint_values[wp.name] = values
            try:
                ok, contacts = valid(values)
            except NotDecided as exc:
                out.append(CheckLine(None, wp.name, f"'{wp.name}': not decided -- {exc}"))
                continue
            what = f"named state '{t.named}'" if t.named else 'joints'
            out.append(CheckLine(ok, wp.name, f"'{wp.name}' ({what}): " + (
                'collision-free' if ok else f'IN COLLISION ({contacts or "see RViz"}): teach it again')))
    previous: Optional[List[float]] = None             # the last TCP waypoint's solution (program order)
    for wp in project.application.program.waypoints:
        t = targets.get(wp.name)
        if t is None or t.type != 'tcp':
            continue
        frame = t.frame or robot.base_frame
        seed_name = wp.name + JOINT_SUFFIX
        where = f"'{wp.name}'" + (f' = {t.derivation}' if t.derivation else '') + f' {_mm(t.position)}'
        # A numerical IK finds what lies near where it starts: one start that fails proves nothing (the same
        # pose next to the bin wall was collision-free from the taught configuration and "in collision" from
        # the mock's start state). Try the pose's own taught joints, the previous waypoint's solution, the
        # joint waypoints, then move_group's current state, before saying no.
        seeds = _seeds([joint_values.get(seed_name), previous] + list(joint_values.values()) + [None])
        try:
            found = None
            for seed in seeds:
                found = ik(frame, t.position, t.orientation, seed, True)
                if found is not None:
                    break
            if found is not None:
                previous = found
                from_seed = (f" (from the configuration of '{seed_name}')"
                             if seed is not None and seed == joint_values.get(seed_name) else '')
                out.append(CheckLine(True, wp.name, f'{where}: reachable, collision-free{from_seed}'))
            elif ik(frame, t.position, t.orientation, seeds[0], False) is not None:
                out.append(CheckLine(False, wp.name, f'{where}: reachable only IN COLLISION (inside the table, a '
                                                     'fixture or the robot itself; no collision-free solution from '
                                                     f'{len(seeds)} start configurations): teach it elsewhere'))
            else:
                out.append(CheckLine(False, wp.name, f'{where}: OUT OF REACH (MoveIt finds no joint solution): '
                                                     'teach it again'))
        except NotDecided as exc:
            out.append(CheckLine(None, wp.name, f'{where}: not decided -- {exc}'))
    pairs = [(j, f"the start of path '{path}'", start) for j, path, start in _path_starts(project)]
    pairs += [(n + JOINT_SUFFIX, f"'{n}'", n) for n in targets
              if n + JOINT_SUFFIX in joint_values and not any(p[0] == n + JOINT_SUFFIX and p[2] == n for p in pairs)]
    tol_m, tol_rad = PATH_START_TOLERANCE
    for joint_wp, what, pose_wp in pairs:
        t = targets.get(pose_wp)
        values = joint_values.get(joint_wp)
        if t is None or t.type != 'tcp' or values is None:
            continue
        try:
            position, orientation = fk(values, t.frame or robot.base_frame)
        except NotDecided as exc:
            out.append(CheckLine(None, joint_wp, f"'{joint_wp}' against {what}: not decided -- {exc}"))
            continue
        d, a = math.dist(position, t.position), _angle(orientation, t.orientation)
        gap = f'{d * 1000.0:.1f} mm, {math.degrees(a):.1f}°'
        if d <= tol_m and a <= tol_rad:
            out.append(CheckLine(True, joint_wp, f"'{joint_wp}' reaches {what} ({gap})"))
        else:
            out.append(CheckLine(False, joint_wp, f"'{joint_wp}' is {gap} from {what} (TMR accepts "
                                                  f'{tol_m * 1000.0:g} mm, {math.degrees(tol_rad):.1f}°): put the '
                                                  f"robot at '{pose_wp}', then Capture TCP + joints into "
                                                  f"'{pose_wp}'"))
    return out


SEGMENT_STEP_RAD = 0.05     # a PTP's joint-space line is sampled at least this finely in every joint (~3°)
LIN_STEP_M = 0.02           # a LIN's Cartesian line: a pose at least every 2 cm ...
LIN_SAMPLES = (8, 50)       # ... and at least 8, at most 50 poses after its start
LIN_JUMP_RAD = 0.5          # a joint moving more than this between two poses of a LIN: the IK changed branch
_IDIOM = {'pregrasp': 'pre-grasp', 'prerelease': 'pre-release'}


def _unit(q: Sequence[float]) -> List[float]:
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    return [v / n for v in q]


def _slerp(a: Sequence[float], b: Sequence[float], t: float) -> List[float]:
    """The unit quaternion ``t`` of the way from ``a`` to ``b`` (qx qy qz qw), the shorter arc."""
    a, b = _unit(a), _unit(b)
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0.0:
        b, dot = [-v for v in b], -dot
    if dot > 0.9995:                                   # nearly equal: a normalised lerp
        return _unit([x + t * (y - x) for x, y in zip(a, b)])
    th = math.acos(dot)
    s = math.sin(th)
    return _unit([(math.sin((1 - t) * th) * x + math.sin(t * th) * y) / s for x, y in zip(a, b)])


def _span(hits: List[Tuple[float, str]]) -> str:
    lo, hi = round(hits[0][0] * 100), round(hits[-1][0] * 100)
    contacts = sorted({c.strip() for _t, found in hits for c in found.split(',') if c.strip()})
    where = f'{lo} %' if lo == hi else f'{lo}-{hi} %'
    return f'{where} of the way' + (f" ({', '.join(contacts)})" if contacts else '')


@dataclass
class _At:
    q: Optional[List[float]]                     # the robot's configuration there; None: not known before runtime
    name: str                                    # how a reader names the place


@dataclass
class _Lin:
    q: Optional[List[float]]                     # the configuration at the LIN's end; None: the line was left
    hits: List[Tuple[float, str]]                # (fraction of the way, contacts) of the poses in collision
    left_at: float = 0.0                         # where no joint solution was found near the line ...
    jump_at: float = 0.0                         # ... or where a joint jumped (the IK changed branch)


def check_motions(project: CanonicalProject, ik: IK, fk: FK, valid: VALID) -> List[CheckLine]:
    """The program's motions followed ALONG THEIR WAY, as the runtime will plan them (owner Test F.1, row 36:
    every waypoint passed the check, then a PTP crossed the cell camera between two of them).

    The program is walked in order -- the moves, the paths' ends, the grasp / release idioms' own motions, the
    called tasks, a repeat's next cycle, each alternative of a "Try alternatives" -- keeping the configuration
    the robot is in. A TCP target's configuration is MoveIt's IK seeded with it (what the runtime's IK starts
    from). A PTP (Pilz) is a line in joint space: sampled every SEGMENT_STEP_RAD. A LIN is a line in Cartesian
    space: sampled every LIN_STEP_M, each pose solved from the previous one. A Free move (planned around
    obstacles), a CIRC and a path's inside are checked at their end only. Where the robot is is not known
    before runtime at a task's start and after a detected candidate: the check resumes at the next joint
    waypoint (or a TCP waypoint taught with its joints, ``X`` / ``X_j``). A segment met twice (a repeat's next
    cycle, a task called twice) is computed once. One step that cannot be decided never stops the others."""
    from ..applications.task_program import ProgramRefusals, plan_program
    try:
        plan = plan_program(project)
    except ProgramRefusals:
        return [CheckLine(None, '', 'the motions between the waypoints: checked once the program is Valid')]
    robot = project.robot
    group = robot.planning_group.name
    joints = list(robot.planning_group.joints)
    named = named_state_values(project)
    bound = set(plan.bound_entries)
    paths = {pp.name: pp for pp in plan.paths}
    out: List[CheckLine] = []
    said: set = set()
    memo: Dict[tuple, object] = {}

    def say(key, line: CheckLine) -> None:
        if key not in said:
            said.add(key)
            out.append(line)

    def phase(mp) -> str:
        word = mp.where.rsplit(')', 1)[-1].strip().lower() or 'motion'
        return _IDIOM.get(word, word)

    def name_of(mp) -> str:
        return f"'{mp.waypoint}'" if mp.waypoint not in bound else f'the {phase(mp)} of step {mp.step}'

    def joint_values(entry) -> Tuple[Optional[List[float]], str]:
        if entry.get('joints') is not None:
            return [float(v) for v in entry['joints']], ''
        values = named.get(entry.get('named') or '')
        if values and all(j in values for j in joints):
            return [values[j] for j in joints], ''
        return None, (f"named state '{entry.get('named')}' has no value for every joint of planning group "
                      f"'{group}'")

    def solve_tcp(entry, seed) -> Optional[List[float]]:
        frame = entry.get('frame') or robot.base_frame
        found = ik(frame, entry['position'], entry['orientation'], seed, True)
        return found if found is not None else ik(frame, entry['position'], entry['orientation'], seed, False)

    def ptp(a, b) -> List[Tuple[float, str]]:
        key = ('ptp', tuple(round(v, 6) for v in a), tuple(round(v, 6) for v in b))
        if key not in memo:
            n = max(2, math.ceil(max(abs(y - x) for x, y in zip(a, b)) / SEGMENT_STEP_RAD))
            hits = []
            for i in range(1, n + 1):
                ok, contacts = valid([x + (i / n) * (y - x) for x, y in zip(a, b)])
                if not ok:
                    hits.append((i / n, contacts))
            memo[key] = hits
        return memo[key]

    def lin(a, entry) -> _Lin:
        key = ('lin', tuple(round(v, 6) for v in a), tuple(entry['position']), tuple(entry['orientation']),
               entry.get('frame') or '')
        if key in memo:
            return memo[key]
        frame = entry.get('frame') or robot.base_frame
        p0, o0 = fk(a, frame)
        length = math.dist(p0, entry['position'])
        n = min(LIN_SAMPLES[1], max(LIN_SAMPLES[0], math.ceil(length / LIN_STEP_M)))
        q, res = list(a), _Lin(None, [])
        for i in range(1, n + 1):
            t = i / n
            pos = [x + t * (y - x) for x, y in zip(p0, entry['position'])]
            nxt = ik(frame, pos, _slerp(o0, entry['orientation'], t), q, False)
            if nxt is None:
                res.left_at = t
                break
            if max(abs(y - x) for x, y in zip(q, nxt)) > LIN_JUMP_RAD:
                res.jump_at = t
                break
            q = nxt
            ok, contacts = valid(q)
            if not ok:
                res.hits.append((t, contacts))
        else:
            res.q = q
        memo[key] = res
        return res

    def collided(mp, at: _At, name: str, kind: str, hits) -> None:
        wp = '' if mp.waypoint in bound else mp.waypoint
        key = (mp.task, mp.step, mp.waypoint, at.name)
        if all(t >= 0.999 for t, _c in hits):        # only the end: the target itself is in collision
            if mp.waypoint in bound:
                say(key, CheckLine(False, '', f'{mp.where}: the {phase(mp)} pose is in collision ({hits[0][1]}): '
                                              'teach the target elsewhere, or change the standoff / retract'))
            else:                                    # its own waypoint line says it
                say(key, CheckLine(None, wp, f'{mp.where}: {name} is reached in collision from {at.name} -- see '
                                             f"the line of {name}"))
            return
        if mp.waypoint in bound:
            fix = ('add a joint waypoint before this step (move the robot clear in RViz, Capture joints)'
                   if kind == 'PTP' else 'teach the target elsewhere, or change the standoff / retract')
        else:
            fix = ('add a joint waypoint between them (move the robot clear in RViz, Capture joints), or make the '
                   'move Free (planned around obstacles)' if kind == 'PTP' else 'teach the two ends clear of it')
        say(key, CheckLine(False, wp, f'{mp.where}: the {kind} from {at.name} to {name} collides at {_span(hits)}: '
                                      f'{fix}'))

    def motion(mp, at: _At) -> _At:
        entry = plan.entries.get(mp.waypoint) or {}
        name = name_of(mp)
        wp = '' if mp.waypoint in bound else mp.waypoint
        key = (mp.task, mp.step, mp.waypoint, at.name)
        if entry.get('type') == 'joint':
            q, why = joint_values(entry)
            if q is None:
                say(key, CheckLine(None, wp, f'{mp.where}: {why} -- the motions from it are not checked'))
                return _At(None, name)
        elif 'position' not in entry:
            say(('runtime', mp.task, mp.step),
                CheckLine(None, '', f'{mp.where}: where the robot goes is known only at runtime (a detected '
                                    'candidate): the motions are followed again from the next joint waypoint'))
            return _At(None, name)
        elif at.q is None:
            # where the robot is is unknown: resume only where the configuration is known -- a TCP waypoint
            # taught with its joints (X / X_j), never an IK branch chosen from move_group's current state
            pair = plan.entries.get(mp.waypoint + JOINT_SUFFIX) if mp.waypoint not in bound else None
            q = joint_values(pair)[0] if pair and pair.get('type') == 'joint' else None
            return _At(q, name)
        else:
            q = None
        if at.q is not None and mp.motion == 'lin' and entry.get('type') == 'tcp':
            res = lin(at.q, entry)
            if res.q is None:
                why = (f'the IK changes configuration at {round(res.jump_at * 100)} % of the way (a joint moves more '
                       f'than {math.degrees(LIN_JUMP_RAD):.0f}°)' if res.jump_at else
                       f'no joint solution near the line at {round(res.left_at * 100)} % of the way')
                say(key, CheckLine(None, wp, f'{mp.where}: the LIN from {at.name} to {name}: {why} -- the runtime '
                                             'may refuse it; check it in RViz'))
                return _At(solve_tcp(entry, at.q), name)
            if res.hits:
                collided(mp, at, name, 'LIN', res.hits)
            return _At(res.q, name)
        if q is None:
            q = solve_tcp(entry, at.q)
            if q is None:
                fix = ('change the standoff / retract, or teach the target elsewhere' if mp.waypoint in bound else
                       'teach it again, or go there through a joint waypoint')
                say(key, CheckLine(False, wp, f'{mp.where}: {name} has no joint solution near {at.name} (out of '
                                              f'reach from there): {fix}'))
                return _At(None, name)
        if at.q is not None and mp.motion == 'ptp' and mp.planner in (None, 'pilz'):
            hits = ptp(at.q, q)
            if hits:
                collided(mp, at, name, 'PTP', hits)
        return _At(q, name)

    def path_end(name: str, at: _At) -> _At:
        pp = paths.get(name)
        where = f"the end of path '{name}'"
        q = at.q
        for state in (pp.states if pp is not None else ()):  # each state solved from the previous one
            t = plan.targets.get(state)
            if t is None:
                return _At(None, where)
            entry = t.keys()
            if entry['type'] == 'joint':
                q = joint_values(entry)[0]
            elif q is not None:
                q = solve_tcp(entry, q)
            if q is None:
                return _At(None, where)
        return _At(q if pp is not None and pp.states else None, where)

    def run(steps, at: _At, depth: int) -> _At:
        for sp in steps:
            try:
                if sp.kind == 'task' and sp.call:
                    if depth >= 8:
                        say(('deep', sp.call), CheckLine(None, '', f"task '{sp.call}': called more than 8 tasks "
                                                                   'deep -- not followed'))
                        at = _At(None, f"the end of task '{sp.call}'")
                    else:
                        at = run(plan.tasks.get(sp.call, []), at, depth + 1)
                elif sp.kind == 'fallback' and sp.bodies:
                    ends = [run(body, at, depth) for body in sp.bodies]   # each alternative from where it starts
                    same = all(e.q is not None and e.q == ends[0].q for e in ends)
                    at = ends[0] if same else _At(None, f'the end of the alternatives of step {sp.label}')
                elif sp.kind in ('repeat', 'retry') and sp.bodies:
                    end = run(sp.bodies[0], at, depth)
                    if sp.kind == 'repeat' and sp.count != 1:
                        end = run(sp.bodies[0], end, depth)        # the next cycle starts where this one ends
                    at = end
                elif sp.kind == 'path' and sp.path:
                    at = path_end(sp.path, at)
                else:
                    for mp in ([sp.move] if sp.move is not None else []) + list(sp.bound_moves):
                        at = motion(mp, at)
            except NotDecided as exc:                      # this step only: the others are still checked
                say(('nd', sp.label, str(exc)), CheckLine(None, '', f'step {sp.label}: not decided -- {exc}'))
                at = _At(None, f'step {sp.label}')
        return at

    run(plan.tasks.get(plan.main_task, []), _At(None, 'where the task starts'), 0)
    # followed to their end: every PTP, a LIN only when the line was not left
    sampled = sum(1 for key, res in memo.items() if key[0] == 'ptp' or res.q is not None)
    if sampled and not any(line.ok is False for line in out):
        unknown = sum(line.ok is None for line in out)
        text = (f"the motions followed along their way: no collision in {sampled} PTP / LIN "
                f"motion{'s' if sampled != 1 else ''} (every 3° / 2 cm; a Free move, a CIRC and a path's inside at "
                'their end only; the part held and the scene as move_group has it now)')
        out.append(CheckLine(True, '', text + (f'; {unknown} not decided above' if unknown else '')))
    return out


def summary(lines: List[CheckLine]) -> str:
    bad = sum(line.ok is False for line in lines)
    unknown = sum(line.ok is None for line in lines)
    head = (f'Live check: {len(lines) - bad - unknown} ✔, {bad} ✖' + (f', {unknown} ?' if unknown else '')
            + ' -- MoveIt IK / FK / collisions, nothing moved')
    return head


__all__ = ['CheckLine', 'JOINT_SUFFIX', 'NotDecided', 'check_motions', 'check_waypoints', 'named_state_values',
           'summary']
