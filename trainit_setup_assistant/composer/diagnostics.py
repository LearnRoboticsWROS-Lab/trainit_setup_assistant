"""Structured diagnostics over the ONE canonical validator (vNext I0.0, I0_DESIGN.md §21).

TSA's validator (``plan_program``, the load-time ``describe_load_error`` and the end-effector contract
checks) reports every problem as one line in the file's own terms::

    R-M3-03: task 'transfer' step 2 (release at 'plce'): no waypoint is named 'plce' (waypoints: ...)

This module does not validate anything. It reads those canonical lines and returns a
:class:`Diagnostic`. The diagnostic keeps the line VERBATIM (``str(d) == line``, byte for byte) and
adds:

- the code;
- a :class:`Location` the editor can select: a waypoint, a path (segment / event), a task, a task
  step by its canonical label (``3.2.1``), a task input / output, or a motion profile.

A line with no recognisable place is located at the project. The canonical refusal stays authoritative;
the location only helps a user find it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

# a canonical code: at the start of a validator line, or after the place in a load refusal
# ("task 'scan' step 3 (move): R-M3-02: ...")
_CODE = re.compile(r'(?:^|:\s)(?P<code>R-M3-\d+|R-I0-\d+|R\d+[a-z]?)\b:')
# an end-effector contract refusal (ADR-0022): "end-effector contract (ADR-0022 R15): device "main" on ..."
_EEF_CODE = re.compile(r'end-effector contract \(ADR-0022 (?P<code>R\d+)\)')
_PLACES = (
    ('step', re.compile(r"task '(?P<name>[^']+)' step (?P<label>\d+(?:\.\d+)*)")),
    ('task_data', re.compile(r"task '(?P<name>[^']+)' (?P<io>input|output) '(?P<item>[^']+)'")),
    ('path', re.compile(r"path '(?P<name>[^']+)'(?: segment (?P<segment>\d+))?(?: event (?P<event>\d+))?")),
    ('waypoint', re.compile(r"waypoint '(?P<name>[^']+)'")),
    ('profile', re.compile(r"motion profile '(?P<name>[^']+)'")),
    ('task', re.compile(r"task '(?P<name>[^']+)'")),
)


@dataclass(frozen=True)
class Location:
    """Where a diagnostic points.

    ``kind`` is one of ``step``, ``task_data``, ``path``, ``waypoint``, ``profile``, ``task`` or
    ``project``. ``name`` is the task, path, waypoint or profile name. ``label`` is a step's canonical
    label, e.g. ``(3, 2, 1)``. ``segment`` and ``event`` are path indices. ``item`` is a task input /
    output name, and ``io`` is ``input`` or ``output``."""

    kind: str = 'project'
    name: str = ''
    label: Tuple[int, ...] = ()
    segment: Optional[int] = None
    event: Optional[int] = None
    io: str = ''
    item: str = ''

    def describe(self) -> str:
        if self.kind == 'step':
            return f"task '{self.name}' step {'.'.join(map(str, self.label))}"
        if self.kind == 'task_data':
            return f"task '{self.name}' {self.io} '{self.item}'"
        if self.kind == 'path':
            out = f"path '{self.name}'"
            if self.segment is not None:
                out += f' segment {self.segment}'
            if self.event is not None:
                out += f' event {self.event}'
            return out
        if self.kind == 'profile':
            return f"motion profile '{self.name}'"
        if self.kind in ('waypoint', 'task'):
            return f"{self.kind} '{self.name}'"
        if self.kind == 'end_effector':
            return 'the end-effector contract (End effector section)'
        return 'the project'


PROJECT = Location()


@dataclass(frozen=True)
class Diagnostic:
    """One canonical problem line, located. ``severity`` is ``error`` (generation refuses) or
    ``warning`` (generation proceeds). ``str()`` is the canonical line, unchanged."""

    text: str
    severity: str = 'error'
    code: str = ''
    location: Location = PROJECT

    def __str__(self) -> str:
        return self.text

    @property
    def hint(self) -> str:
        """The canonical "what to change" part, when the line carries one (after `` -- ``)."""
        _, sep, after = self.text.partition(' -- ')
        return after if sep else ''


def locate(text: str) -> Location:
    """The EARLIEST place a canonical line names: its subject comes first ("task 't' step 2 (path
    'p')" is the step, not the path)."""
    best = None
    for kind, pattern in _PLACES:
        m = pattern.search(text)
        if m is not None and (best is None or m.start() < best[0]):
            best = (m.start(), kind, m)
    if best is None:
        return PROJECT
    _, kind, m = best
    g = m.groupdict()
    if kind == 'step':
        return Location('step', g['name'], tuple(int(x) for x in g['label'].split('.')))
    if kind == 'task_data':
        return Location('task_data', g['name'], io=g['io'], item=g['item'])
    if kind == 'path':
        return Location('path', g['name'],
                        segment=int(g['segment']) if g.get('segment') else None,
                        event=int(g['event']) if g.get('event') else None)
    return Location(kind, g['name'])


def parse(text: str, severity: str = 'error') -> Diagnostic:
    """A :class:`Diagnostic` for one canonical line. The line is kept verbatim. An end-effector contract
    refusal is located at the End effector section (vNext I0, §34 E3)."""
    eef = _EEF_CODE.search(text)
    if eef is not None and _CODE.search(text) is None:
        return Diagnostic(text=text, severity=severity, code=eef.group('code'), location=Location('end_effector', ''))
    m = _CODE.search(text)
    return Diagnostic(text=text, severity=severity, code=m.group('code') if m else '',
                      location=locate(text))


def parse_all(errors: Iterable[str] = (), warnings: Iterable[str] = ()) -> List[Diagnostic]:
    """Errors first, then warnings, each in the validator's own order."""
    return ([parse(e, 'error') for e in errors] + [parse(w, 'warning') for w in warnings])


def parse_load_error(message: str) -> List[Diagnostic]:
    """The located lines of a load refusal (``describe_load_error``'s text: a head line ending in
    "cannot be loaded:" and one indented line per problem, or the head and one problem on one line)."""
    head, sep, rest = message.partition('cannot be loaded:')
    if not sep:
        return [parse(message.strip())]
    lines = [ln.strip() for ln in rest.splitlines() if ln.strip()]
    return [parse(ln) for ln in lines] or [parse(message.strip())]


# "did you mean": a misspelt NAME in a canonical refusal, and where the right names are
_UNKNOWN = (
    (re.compile(r"no waypoint is named '(?P<n>[^']+)'"), 'waypoints'),
    (re.compile(r"no path is named '(?P<n>[^']+)'"), 'paths'),
    (re.compile(r"no task is named '(?P<n>[^']+)'"), 'tasks'),
    (re.compile(r"profile '(?P<n>[^']+)' is not a motion profile"), 'profiles'),
    (re.compile(r"no motion profile is named '(?P<n>[^']+)'"), 'profiles'),
    (re.compile(r"no detector is named '(?P<n>[^']+)'"), 'detectors'),
)


def suggestion(diag: Diagnostic, doc: dict, project=None) -> Optional[str]:
    """A deterministic, UX-only hint for a misspelt name ("Did you mean 'place'?"): the closest name in
    scope (difflib), or None. The canonical refusal stays the authority."""
    import difflib
    scopes = {
        'waypoints': [w.get('name') for w in doc.get('waypoints') or []],
        'paths': [p.get('name') for p in doc.get('paths') or []],
        'tasks': [t.get('name') for t in doc.get('tasks') or []],
        'profiles': list(doc.get('motion_profiles') or {}),
        'detectors': [d.name for d in (project.perception.detectors
                                       if project is not None and project.perception else [])],
    }
    for pattern, scope in _UNKNOWN:
        m = pattern.search(diag.text)
        if m:
            close = difflib.get_close_matches(m.group('n'), [n for n in scopes[scope] if n], n=1, cutoff=0.6)
            return f"Did you mean '{close[0]}'?" if close else None
    return None


__all__ = ['Diagnostic', 'Location', 'PROJECT', 'locate', 'parse', 'parse_all', 'parse_load_error', 'suggestion']
