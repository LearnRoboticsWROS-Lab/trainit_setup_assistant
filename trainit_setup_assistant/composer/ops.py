"""The Application Composer's model operations (vNext I0.0, I0_DESIGN.md §13).

Every edit the Composer makes to a Canonical Task Program is a function in this module. The module
uses neither Qt nor ROS.

**One model, no GUI copy.** An operation edits a COPY of the program AS WRITTEN (the ``application:``
mapping that ``save_project`` writes). The whole project is then re-validated through
``CanonicalProject.parse_obj``, the same path ``load_project`` takes.
- If it is refused, the operation raises :class:`ComposerError` (the canonical load text, located)
  and the project is unchanged.
- If it is accepted, the new project replaces the old one.

So what the Composer holds is always a project TSA could have loaded from a file, and saving it writes
exactly what was edited: nothing hidden, nothing the YAML cannot say.

**Semantic checks are not done here.** References to a missing waypoint, a call cycle and the like
are ``plan_program``'s, reported as diagnostics (:mod:`.diagnostics`). An operation only refuses what
the file itself could not hold (a step that is two kinds, a duplicate name, an unknown key).

**Steps are addressed by their canonical LABEL**, the one the validator's messages use:
- ``(3,)`` is a task's third step;
- ``(3, 2)`` is the second step of the body of a repeat / retry at step 3;
- ``(3, 2, 1)`` is the first step of the second alternative of a fallback at step 3.
"""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..model.enums import AppType
from ..model.io import project_to_dict
from ..model.program import describe_load_error
from ..model.project import CanonicalProject
from .diagnostics import Diagnostic, parse_load_error

Label = Tuple[int, ...]
CONTROL_KINDS = ('repeat', 'retry', 'fallback')
_EDITOR = 'the edited program'
# what a whole-project edit may change besides robot.gripper and deployment.bridges (checked separately)
_PROJECT_EDITABLE = ('application', 'robot', 'deployment', 'schema_version')


class ComposerError(ValueError):
    """An edit the canonical model refuses. The project is unchanged. ``diagnostics`` locate it."""

    def __init__(self, message: str, diagnostics: Sequence[Diagnostic] = ()):
        super().__init__(message)
        self.diagnostics = tuple(diagnostics)


def _refuse(message: str) -> None:
    from .diagnostics import parse
    raise ComposerError(message, (parse(message),))


def step_kind(step: dict) -> str:
    """The one kind of a step mapping (``move``, ``path``, ..., ``fallback``)."""
    if not isinstance(step, dict) or len(step) != 1:
        _refuse('R-M3-02: a step is exactly one kind')
    return next(iter(step))


def step_bodies(step: dict) -> List[list]:
    """The step lists a control step runs: one body for repeat / retry, the alternatives of a
    fallback; none for any other step."""
    kind = step_kind(step)
    if kind in ('repeat', 'retry'):
        return [step[kind].setdefault('body', [])]
    if kind == 'fallback':
        return step[kind].setdefault('bodies', [])
    return []


def walk(steps: list, label: Label = ()):
    """Every step of a list in program order, control bodies included, as ``(label, step)``: the
    canonical labels (``model.program.walk_steps`` on the written form)."""
    for i, step in enumerate(steps, start=1):
        here = label + (i,)
        yield here, step
        kind = step_kind(step)
        if kind == 'fallback':
            for b, body in enumerate(step['fallback'].get('bodies') or [], start=1):
                yield from walk(body, here + (b,))
        elif kind in ('repeat', 'retry'):
            yield from walk(step[kind].get('body') or [], here)


class Composer:
    """Holds ONE CanonicalProject whose application is a task program, and edits it.

    ``history`` (undo / redo) is a stack of whole projects, each a loadable project; it is editor
    state, never saved."""

    def __init__(self, project: CanonicalProject,
                 on_change: Optional[Callable[[CanonicalProject], None]] = None):
        if not project.application.is_task_program():
            raise ValueError('the Application Composer edits a Canonical Task Program '
                             "(application type 'task_program')")
        self.project = project
        # the ONE owner of the project (the GUI controller) is told of every new project, so the
        # Composer never holds a second copy that drifts from it
        self._on_change = on_change
        self._undo: List[CanonicalProject] = []
        self._redo: List[CanonicalProject] = []

    def _set(self, project: CanonicalProject) -> None:
        self.project = project
        if self._on_change is not None:
            self._on_change(project)

    # ---- the document ------------------------------------------------------------------------
    @property
    def program(self):
        return self.project.application.program

    def document(self) -> dict:
        """A deep copy of the program AS WRITTEN (``application:`` without ``type``)."""
        app = project_to_dict(self.project)['application']
        return {k: v for k, v in copy.deepcopy(app).items() if k != 'type'}

    def edit(self, mutate: Callable[[dict], Any]) -> Any:
        """Apply ``mutate`` to a copy of the document and commit it through the canonical model.
        Returns what ``mutate`` returned."""
        doc = self.document()
        result = mutate(doc)
        self._commit(doc)
        return result

    def edit_project(self, mutate: Callable[[dict], Any]) -> Any:
        """Apply ``mutate`` to a copy of the WHOLE project as written and commit it through the canonical
        model, in the same history as the program's edits: the end effector (``robot.gripper``) and the
        bridges it replaces (``deployment.bridges``) are edited here (vNext I0, I0_DESIGN.md §34)."""
        data = project_to_dict(self.project)
        before = copy.deepcopy(data)
        result = mutate(data)
        changed = [k for k in set(before) | set(data) if k not in _PROJECT_EDITABLE and before.get(k) != data.get(k)]
        robot_before = {k: v for k, v in (before.get('robot') or {}).items() if k != 'gripper'}
        robot_after = {k: v for k, v in (data.get('robot') or {}).items() if k != 'gripper'}
        dep_before = {k: v for k, v in (before.get('deployment') or {}).items() if k != 'bridges'}
        dep_after = {k: v for k, v in (data.get('deployment') or {}).items() if k != 'bridges'}
        if changed or robot_before != robot_after or dep_before != dep_after:
            _refuse('the Composer edits the program and the end effector (robot.gripper, deployment.bridges) only: '
                    'the robot, the scene and the deployment are set at Steps 1-4')
        self._commit_project(data)
        return result

    def _commit(self, doc: dict) -> None:
        data = project_to_dict(self.project)
        data['application'] = {'type': AppType.TASK_PROGRAM.value, **doc}
        self._commit_project(data)

    def _commit_project(self, data: dict) -> None:
        before = project_to_dict(self.project)
        try:
            new = CanonicalProject.parse_obj(data)
        except ValueError as exc:
            message = describe_load_error(exc, data, _EDITOR)
            raise ComposerError(message, parse_load_error(message)) from None
        if project_to_dict(new) == before:
            return                                   # nothing changed: no history entry
        self._undo.append(self.project)
        self._redo.clear()
        self._set(new)

    # ---- history -------------------------------------------------------------------------------
    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> None:
        if self._undo:
            self._redo.append(self.project)
            self._set(self._undo.pop())

    def redo(self) -> None:
        if self._redo:
            self._undo.append(self.project)
            self._set(self._redo.pop())

    # ---- waypoints -----------------------------------------------------------------------------
    def waypoint_names(self) -> List[str]:
        return [w.name for w in self.program.waypoints]

    def set_waypoint(self, spec: dict, index: Optional[int] = None) -> None:
        """Add a waypoint, or replace the one with the same ``name`` in place. ``index`` places a new
        one (default: last). A replacement writes ``spec`` exactly: no key of the old one is kept."""
        spec = copy.deepcopy(spec)

        def mutate(doc):
            wps = doc.setdefault('waypoints', [])
            for i, w in enumerate(wps):
                if w.get('name') == spec.get('name'):
                    wps[i] = spec
                    return
            wps.insert(len(wps) if index is None else index, spec)
        self.edit(mutate)

    def set_waypoints(self, specs: List[dict]) -> None:
        """``set_waypoint`` for several waypoints as ONE edit (one undo step): a TCP pose and the joints
        captured with it (owner Test C)."""
        specs = copy.deepcopy(list(specs))

        def mutate(doc):
            wps = doc.setdefault('waypoints', [])
            for spec in specs:
                i = next((i for i, w in enumerate(wps) if w.get('name') == spec.get('name')), None)
                if i is None:
                    wps.append(spec)
                else:
                    wps[i] = spec
        self.edit(mutate)

    def delete_waypoint(self, name: str) -> None:
        """Remove an unused waypoint. A used one is refused, naming its users."""
        users = self.waypoint_users(name)
        if users:
            _refuse(f"R-M3-03: waypoint '{name}' is used by {', '.join(users)}: change those first")
        self.edit(lambda doc: doc.__setitem__(
            'waypoints', [w for w in doc.get('waypoints', []) if w.get('name') != name]))

    def rename_waypoint(self, old: str, new: str) -> None:
        """Rename a waypoint and every reference to it (moves, paths, relative targets, grasp /
        release, path events)."""
        def mutate(doc):
            found = False
            for w in doc.get('waypoints', []):
                if w.get('name') == old:
                    w['name'], found = new, True
            if not found:
                _refuse(f"R-M3-03: no waypoint is named '{old}'")
            for _where, value, set_ in list(waypoint_refs(doc)):
                if value == old:
                    set_(new)
        self.edit(mutate)

    def waypoint_users(self, name: str) -> List[str]:
        """Where a waypoint is referenced, in the validator's terms (one entry per user)."""
        out: List[str] = []
        for where, value, _set in waypoint_refs(self.document()):
            if value == name and where not in out:
                out.append(where)
        return out

    # ---- motion profiles -----------------------------------------------------------------------
    def set_profile(self, name: str, spec: dict) -> None:
        self.edit(lambda doc: doc.setdefault('motion_profiles', {}).__setitem__(name, copy.deepcopy(spec)))

    def delete_profile(self, name: str) -> None:
        users = self.profile_users(name)
        if users:
            _refuse(f"R-M3-03: motion profile '{name}' is used by {', '.join(users)}: change those first")
        self.edit(lambda doc: (doc.get('motion_profiles') or {}).pop(name, None))

    def rename_profile(self, old: str, new: str) -> None:
        def mutate(doc):
            profiles = doc.get('motion_profiles') or {}
            if old not in profiles:
                _refuse(f"R-M3-03: no motion profile is named '{old}'")
            if new != old and new in profiles:
                _refuse(f"R-M3-02: motion profile '{new}' already exists: renaming '{old}' to it would "
                        'merge two profiles')
            doc['motion_profiles'] = {(new if k == old else k): v for k, v in profiles.items()}
            for node in _profile_holders(doc):
                for key in ('profile', 'approach_profile'):
                    if node.get(key) == old:
                        node[key] = new
        self.edit(mutate)

    def profile_users(self, name: str) -> List[str]:
        doc, out = self.document(), []
        for p in doc.get('paths', []):
            if p.get('profile') == name or any(s.get('profile') == name for s in p.get('segments') or []):
                out.append(f"path '{p['name']}'")
        for t in doc.get('tasks', []):
            for label, step in walk(t.get('steps') or []):
                body = step[step_kind(step)]
                if isinstance(body, dict) and name in (body.get('profile'), body.get('approach_profile')):
                    out.append(f"task '{t['name']}' step {'.'.join(map(str, label))}")
        return out

    # ---- paths ---------------------------------------------------------------------------------
    def set_path(self, spec: dict, index: Optional[int] = None) -> None:
        """Add a path, or replace the one with the same ``name`` in place."""
        spec = copy.deepcopy(spec)

        def mutate(doc):
            paths = doc.setdefault('paths', [])
            for i, p in enumerate(paths):
                if p.get('name') == spec.get('name'):
                    paths[i] = spec
                    return
            paths.insert(len(paths) if index is None else index, spec)
        self.edit(mutate)

    def rename_path(self, old: str, new: str) -> None:
        """Rename a path and every ``path:`` step that runs it."""
        def mutate(doc):
            paths = doc.get('paths') or []
            if not any(p.get('name') == old for p in paths):
                _refuse(f"R-M3-03: no path is named '{old}'")
            for p in paths:
                if p.get('name') == old:
                    p['name'] = new
            for t in doc.get('tasks', []):
                for _label, step in walk(t.get('steps') or []):
                    if step_kind(step) == 'path' and step['path'] == old:
                        step['path'] = new
        self.edit(mutate)

    def moves_to_path(self, task: str, label: Label, count: int, name: str, continuity: str = 'blend',
                      blend_radius_m: float = 0.005) -> None:
        """"Make these moves one path" (I0_DESIGN.md §15): ``count`` consecutive LIN / CIRC moves from
        ``label`` become ONE canonical path, run by one ``path:`` step in their place.

        - **The path's start** (its states[0]) is the waypoint the step BEFORE them moves to: FollowPath
          requires the robot there.
        - **Its states** are the moves' targets; its segments are their relations (LIN, or CIRC with its
          via / centre).
        - **Its profile** is theirs when they share one, else each segment keeps its own; per-move limits
          stay on their segment.
        - ``continuity`` is ``blend`` (with ``blend_radius_m`` on every segment but the last) or ``stop``.

        It is explicit, user-triggered and undoable; the model then validates the path."""
        if continuity not in ('blend', 'stop'):
            _refuse(f"R-M3-02: continuity is 'stop' or 'blend', got {continuity!r}")

        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            where = f"task '{task}' step {_text(label)}"
            if count < 1 or i + count > len(steps):
                _refuse(f'R-M3-03: {where}: no {count} steps from here')
            moves = steps[i:i + count]
            for k, st in enumerate(moves):
                kind = step_kind(st)
                mv = st.get('move') if kind == 'move' else None
                if mv is None or mv.get('type') not in ('lin', 'circ'):
                    what = mv.get('type') if mv else kind
                    _refuse(f"R-M3-04: task '{task}' step {_text(label[:-1] + (label[-1] + k,))} is a {what} "
                            'step: a path is LIN and CIRC motions only (a PTP move stays a move)')
                if mv.get('provider'):
                    _refuse(f"R-M3-04: task '{task}' step {_text(label[:-1] + (label[-1] + k,))} names a "
                            'provider: a path chooses its own (set it on the path)')
            before = steps[i - 1] if i > 0 else None
            start = before['move']['to'] if before is not None and step_kind(before) == 'move' else None
            if start is None:
                _refuse(f'R-M3-04: {where}: a path starts where the robot IS: the step before these moves '
                        'must be a move to a waypoint (e.g. a PTP to the path start)')
            profiles = {st['move'].get('profile') for st in moves}
            shared = profiles.pop() if len(profiles) == 1 else None
            segments = []
            for k, st in enumerate(moves):
                mv = copy.deepcopy(st['move'])
                seg = {'type': mv.pop('type')}
                mv.pop('to')
                for key in ('via', 'center'):
                    if mv.get(key) is not None:
                        seg[key] = mv.pop(key)
                if shared is None and mv.get('profile'):
                    seg['profile'] = mv['profile']
                mv.pop('profile', None)
                seg.update(mv)                                # the move's own limits, on its segment
                if continuity == 'blend' and k < count - 1:
                    seg['blend_radius_m'] = blend_radius_m
                segments.append(seg)
            path = {'name': name, 'states': [start] + [st['move']['to'] for st in moves],
                    'segments': segments}
            if continuity == 'blend':
                path['continuity'] = 'blend'
            if shared:
                path['profile'] = shared
            doc.setdefault('paths', []).append(path)
            steps[i:i + count] = [{'path': name}]
        self.edit(mutate)

    def delete_path(self, name: str) -> None:
        users = [f"task '{t}' step {'.'.join(map(str, lb))}" for t, lb, s in self._steps()
                 if step_kind(s) == 'path' and s['path'] == name]
        if users:
            _refuse(f"R-M3-03: path '{name}' is used by {', '.join(users)}: change those first")
        self.edit(lambda doc: doc.__setitem__(
            'paths', [p for p in doc.get('paths', []) if p.get('name') != name]))

    # ---- tasks ---------------------------------------------------------------------------------
    def task_names(self) -> List[str]:
        return [t.name for t in self.program.tasks]

    def add_task(self, name: str, steps: Sequence[dict], description: Optional[str] = None) -> None:
        """A new task, created WITH its first step(s): the canonical model has no empty task, and the
        Composer never holds a project the model would refuse."""
        task: Dict[str, Any] = {'name': name}
        if description:
            task['description'] = description
        task['steps'] = copy.deepcopy(list(steps))
        self.edit(lambda doc: doc.setdefault('tasks', []).append(task))

    def set_task_description(self, name: str, description: Optional[str]) -> None:
        def mutate(doc):
            t = _task(doc, name)
            if description:
                t['description'] = description
            else:
                t.pop('description', None)
        self.edit(mutate)

    def rename_task(self, old: str, new: str) -> None:
        """Rename a task, its calls and ``main_task``."""
        def mutate(doc):
            _task(doc, old)['name'] = new
            if doc.get('main_task') == old:
                doc['main_task'] = new
            for t in doc.get('tasks', []):
                for _label, step in walk(t.get('steps') or []):
                    if step_kind(step) == 'task' and step['task'] == old:
                        step['task'] = new
        self.edit(mutate)

    def delete_task(self, name: str) -> None:
        if self.program.main_task == name:
            _refuse(f"R-M3-15: task '{name}' is the main task: choose another main task first")
        callers = [f"task '{t}' step {'.'.join(map(str, lb))}" for t, lb, s in self._steps()
                   if step_kind(s) == 'task' and s['task'] == name]
        if callers:
            _refuse(f"R-M3-15: task '{name}' is called by {', '.join(callers)}: change those first")
        self.edit(lambda doc: doc.__setitem__(
            'tasks', [t for t in doc.get('tasks', []) if t.get('name') != name]))

    def set_main_task(self, name: str) -> None:
        def mutate(doc):
            _task(doc, name)
            doc['main_task'] = name
        self.edit(mutate)

    def set_task_data(self, name: str, inputs: Optional[list] = None,
                      outputs: Optional[list] = None) -> None:
        """A task's ``inputs`` / ``outputs`` (``[{name, type}]``); an empty list removes the key."""
        def mutate(doc):
            t = _task(doc, name)
            for key, value in (('inputs', inputs), ('outputs', outputs)):
                if value is None:
                    continue
                if value:
                    t[key] = copy.deepcopy(value)
                else:
                    t.pop(key, None)
        self.edit(mutate)

    # ---- steps ---------------------------------------------------------------------------------
    def step(self, task: str, label: Label) -> dict:
        """A copy of the step at ``label``."""
        doc = self.document()
        steps, i = _slot(_task(doc, task), label)
        if i >= len(steps):
            _refuse(f"R-M3-03: task '{task}' has no step {_text(label)}")
        return steps[i]

    def insert_step(self, task: str, label: Label, step: dict) -> None:
        """Insert ``step`` so that it gets ``label``: ``(4,)`` before today's fourth step, ``(5,)``
        after the last of four; ``(3, 1)`` first in the body of the control step 3; ``(3, 2, 1)``
        first in the second alternative of the fallback 3."""
        step = copy.deepcopy(step)
        step_kind(step)

        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            if i > len(steps):
                _refuse(f"R-M3-03: task '{task}': no place {_text(label)} (the list has {len(steps)} steps)")
            steps.insert(i, step)
        self.edit(mutate)

    def append_step(self, task: str, step: dict, parent: Label = (), body: int = 1) -> Label:
        """Append ``step`` to a task (``parent`` empty) or to a control step's body (``body``: the
        alternative of a fallback, 1-based). Returns the new step's label."""
        doc = self.document()
        t = _task(doc, task)
        if parent:
            ctl, i = _slot(t, parent)
            bodies = step_bodies(ctl[i])
            kind = step_kind(ctl[i])
            if kind not in CONTROL_KINDS:
                _refuse(f"R-M3-02: task '{task}' step {_text(parent)} is a {kind} step: it has no body")
            if kind == 'fallback':
                if not 1 <= body <= len(bodies):
                    _refuse(f"R-M3-03: task '{task}' step {_text(parent)} has no alternative {body}")
                label = parent + (body, len(bodies[body - 1]) + 1)
            else:
                label = parent + (len(bodies[0]) + 1,)
        else:
            label = (len(t.get('steps') or []) + 1,)
        self.insert_step(task, label, step)
        return label

    def replace_step(self, task: str, label: Label, step: dict) -> None:
        step = copy.deepcopy(step)
        step_kind(step)

        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            if i >= len(steps):
                _refuse(f"R-M3-03: task '{task}' has no step {_text(label)}")
            steps[i] = step
        self.edit(mutate)

    def delete_step(self, task: str, label: Label) -> None:
        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            if i >= len(steps):
                _refuse(f"R-M3-03: task '{task}' has no step {_text(label)}")
            del steps[i]
        self.edit(mutate)

    def move_step(self, task: str, label: Label, delta: int) -> Label:
        """Move a step up (``delta`` < 0) or down inside its own list. Returns its new label."""
        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            j = i + delta
            if not (0 <= i < len(steps) and 0 <= j < len(steps)):
                _refuse(f"R-M3-03: task '{task}' step {_text(label)} cannot move by {delta}")
            steps.insert(j, steps.pop(i))
            return label[:-1] + (j + 1,)
        return self.edit(mutate)

    def wrap_steps(self, task: str, label: Label, count: int, kind: str, times: int = 2,
                   alternatives: Sequence[list] = ()) -> None:
        """Wrap ``count`` consecutive steps from ``label`` into a ``repeat`` (``times`` cycles), a
        ``retry`` (``times`` attempts) or a ``fallback``: the steps become its FIRST alternative and
        ``alternatives`` (one step list each, at least one: a fallback has two or more) follow."""
        if kind not in CONTROL_KINDS:
            _refuse(f"R-M3-02: '{kind}' is not a control step (repeat, retry or fallback)")

        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            if count < 1 or i + count > len(steps):
                _refuse(f"R-M3-03: task '{task}': no {count} steps from step {_text(label)}")
            body = steps[i:i + count]
            if kind == 'repeat':
                ctl = {'repeat': {'cycles': times, 'body': body}}
            elif kind == 'retry':
                ctl = {'retry': {'attempts': times, 'body': body}}
            else:
                ctl = {'fallback': {'bodies': [body] + [copy.deepcopy(list(a)) for a in alternatives]}}
            steps[i:i + count] = [ctl]
        self.edit(mutate)

    def unwrap_step(self, task: str, label: Label, keep: int = 1) -> None:
        """Replace a repeat / retry by the steps of its body, or a fallback by the steps of its
        alternative ``keep`` (1-based; the other alternatives are dropped)."""
        def mutate(doc):
            steps, i = _slot(_task(doc, task), label)
            if i >= len(steps):
                _refuse(f"R-M3-03: task '{task}' has no step {_text(label)}")
            kind = step_kind(steps[i])
            bodies = step_bodies(steps[i])
            if kind not in CONTROL_KINDS:
                _refuse(f"R-M3-02: task '{task}' step {_text(label)} is a {kind} step: nothing to unwrap")
            if not 1 <= keep <= len(bodies):
                _refuse(f"R-M3-03: task '{task}' step {_text(label)} has no alternative {keep}")
            steps[i:i + 1] = bodies[keep - 1]
        self.edit(mutate)

    def add_alternative(self, task: str, label: Label, steps: Optional[list] = None) -> None:
        """A further alternative of the fallback at ``label`` (tried when the ones before it fail)."""
        def mutate(doc):
            lst, i = _slot(_task(doc, task), label)
            if i >= len(lst) or step_kind(lst[i]) != 'fallback':
                _refuse(f"R-M3-02: task '{task}' step {_text(label)} is not a fallback")
            lst[i]['fallback']['bodies'].append(copy.deepcopy(steps) if steps else [])
        self.edit(mutate)

    def remove_alternative(self, task: str, label: Label, body: int) -> None:
        def mutate(doc):
            lst, i = _slot(_task(doc, task), label)
            if i >= len(lst) or step_kind(lst[i]) != 'fallback':
                _refuse(f"R-M3-02: task '{task}' step {_text(label)} is not a fallback")
            bodies = lst[i]['fallback']['bodies']
            if not 1 <= body <= len(bodies):
                _refuse(f"R-M3-03: task '{task}' step {_text(label)} has no alternative {body}")
            del bodies[body - 1]
        self.edit(mutate)

    def set_control_count(self, task: str, label: Label, times: int) -> None:
        """A repeat's cycles or a retry's attempts."""
        def mutate(doc):
            lst, i = _slot(_task(doc, task), label)
            kind = step_kind(lst[i]) if i < len(lst) else None
            if kind == 'repeat':
                lst[i]['repeat']['cycles'] = times
            elif kind == 'retry':
                lst[i]['retry']['attempts'] = times
            else:
                _refuse(f"R-M3-02: task '{task}' step {_text(label)} is not a repeat or a retry")
        self.edit(mutate)

    # ---- helpers -------------------------------------------------------------------------------
    def _steps(self):
        doc = self.document()
        for t in doc.get('tasks', []):
            for label, step in walk(t.get('steps') or []):
                yield t['name'], label, step


def _text(label: Label) -> str:
    return '.'.join(map(str, label))


def _task(doc: dict, name: str) -> dict:
    for t in doc.get('tasks', []):
        if t.get('name') == name:
            return t
    _refuse(f"R-M3-15: no task is named '{name}'")


def _slot(task: dict, label: Label) -> Tuple[list, int]:
    """The list a step with ``label`` lives in, and its 0-based index there (which may be the
    list's length: the place after the last step). A control step's number is followed by the
    body's step number (repeat / retry) or by the alternative's number and then the step's
    (fallback)."""
    if not label or not all(isinstance(x, int) and not isinstance(x, bool) for x in label):
        _refuse(f"R-M3-03: task '{task.get('name')}': {label!r} is not a step label (numbers, 1-based)")
    steps = task.setdefault('steps', [])
    rest = list(label)
    while len(rest) > 1:
        i = rest.pop(0) - 1
        if not 0 <= i < len(steps):
            _refuse(f"R-M3-03: task '{task.get('name')}' has no step {_text(label)}")
        kind = step_kind(steps[i])
        if kind in ('repeat', 'retry'):
            steps = steps[i][kind].setdefault('body', [])
        elif kind == 'fallback':
            b = rest.pop(0) - 1 if rest else -1
            bodies = steps[i]['fallback'].setdefault('bodies', [])
            if not 0 <= b < len(bodies):
                _refuse(f"R-M3-03: task '{task.get('name')}' has no step {_text(label)}: the fallback at "
                        f"{_text(label[:len(label) - len(rest) - 1])} has {len(bodies)} alternatives")
            if not rest:
                _refuse(f"R-M3-03: task '{task.get('name')}' has no step {_text(label)}: a step inside a "
                        'fallback is <fallback>.<alternative>.<step>')
            steps = bodies[b]
        else:
            _refuse(f"R-M3-03: task '{task.get('name')}' step {_text(label[:len(label) - len(rest)])} "
                    f'is a {kind} step: it has no body')
    i = rest[0] - 1
    if i < 0:
        _refuse(f"R-M3-03: task '{task.get('name')}' has no step {_text(label)}")
    return steps, i


# The fields that NAME a waypoint, per model (model/program.py; pinned by test_i0_0_foundations):
# a waypoint reference is never matched by a key name alone outside these.
WAYPOINT_FIELDS = {
    'relative': ('to',),
    'segment': ('via', 'center'),
    'event_anchor': ('state',),
    'move': ('to', 'via', 'center'),
    'grasp': ('at', 'orientation_from'),
    'release': ('at',),
}


def _setter(node, key):
    return lambda value: node.__setitem__(key, value)


def waypoint_refs(doc: dict):
    """Every waypoint reference of a program document as ``(where, name, set)``: ``where`` in the
    validator's terms, ``set(new)`` rewrites it in place. The single walker behind
    :meth:`Composer.waypoint_users` and :meth:`Composer.rename_waypoint`."""
    for w in doc.get('waypoints', []):
        rel = w.get('relative')
        if isinstance(rel, dict) and isinstance(rel.get('to'), str):
            yield f"waypoint '{w.get('name')}'", rel['to'], _setter(rel, 'to')
        if isinstance(w.get('configuration'), str):          # ADR-0028: the configuration a pose was taught in
            yield f"waypoint '{w.get('name')}' (its configuration)", w['configuration'], _setter(w, 'configuration')
    for p in doc.get('paths', []):
        where = f"path '{p.get('name')}'"
        states = p.get('states') or []
        for i, s in enumerate(states):
            if isinstance(s, str):
                yield where, s, _setter(states, i)
        for seg in p.get('segments') or []:
            for key in WAYPOINT_FIELDS['segment']:
                if isinstance(seg.get(key), str):
                    yield where, seg[key], _setter(seg, key)
        for ev in p.get('events') or []:
            at = ev.get('at')
            if isinstance(at, dict) and isinstance(at.get('state'), str):
                yield where, at['state'], _setter(at, 'state')
    for t in doc.get('tasks', []):
        for label, step in walk(t.get('steps') or []):
            kind = step_kind(step)
            body = step[kind]
            if isinstance(body, dict):
                for key in WAYPOINT_FIELDS.get(kind, ()):
                    if isinstance(body.get(key), str):
                        yield f"task '{t.get('name')}' step {_text(label)}", body[key], _setter(body, key)


def _profile_holders(doc: dict):
    for p in doc.get('paths', []):
        yield p
        yield from (s for s in p.get('segments') or [])
    for t in doc.get('tasks', []):
        for _label, step in walk(t.get('steps') or []):
            body = step[step_kind(step)]
            if isinstance(body, dict):
                yield body


__all__ = ['CONTROL_KINDS', 'Composer', 'ComposerError', 'Label', 'WAYPOINT_FIELDS', 'step_bodies',
           'step_kind', 'walk', 'waypoint_refs']
