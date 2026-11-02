"""Steps and routes in an engineer's words (vNext I0.7, I0_DESIGN.md §13, §15, §16). Pure, no Qt.

Every canonical step kind has one line of text, without BehaviorTree words:
- a repeat is "Repeat ×N", a retry is "Retry ×N";
- a fallback is "Try alternatives";
- a task call is "Run task".

:func:`route` lists a task's motions as ``FROM → TO : HOW · profile · STOP``. Control bodies are shown
where they start, never flattened into a hidden order.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from .ops import step_kind, walk

KIND_TITLES = {
    'move': 'Move', 'path': 'Path', 'process': 'Process', 'dwell': 'Wait', 'detect': 'Detect',
    'grasp': 'Grasp', 'release': 'Release', 'tool': 'Tool', 'task': 'Run task', 'repeat': 'Repeat',
    'retry': 'Retry', 'fallback': 'Try alternatives',
}
PROCESS_TITLES = {'process_on': 'Process ON', 'process_off': 'Process OFF', 'set_value': 'Set value',
                  'capture': 'Capture', 'external': 'External process'}


def step_text(step: dict) -> str:
    kind = step_kind(step)
    body = step[kind]
    if kind == 'move':
        how = body['type'].upper()
        if body.get('via'):
            how += f" via {body['via']}"
        if body.get('center'):
            how += f" about {body['center']}"
        prof = f" · {body['profile']}" if body.get('profile') else ''
        return f"Move → {body['to']} · {how}{prof}"
    if kind == 'path':
        return f'Path {body}'
    if kind == 'task':
        return f'Run task {body}'
    if kind == 'dwell':
        return f"Wait {body.get('ms')} ms"
    if kind == 'repeat':
        return f"Repeat ×{body.get('cycles')}"
    if kind == 'retry':
        return f"Retry ×{body.get('attempts')}"
    if kind == 'fallback':
        return f"Try alternatives ({len(body.get('bodies') or [])})"
    if kind == 'process':
        title = PROCESS_TITLES.get(body.get('kind'), body.get('kind', 'process'))
        if body.get('kind') == 'capture':
            return f"{title} ({body.get('detector')})" + (f" → {body['out']}" if body.get('out') else '')
        detail = body.get('channel') or ''
        if body.get('kind') == 'set_value':
            detail += f" = {body.get('value')}"
        return f'{title} {detail}'.rstrip()
    if kind == 'detect':
        out = f" → {body['out']}" if body.get('out') else ''
        return f"Detect ({body.get('detector')}){out}"
    if kind == 'grasp':
        src = body.get('at') or (f"candidates {body['candidates']}" if body.get('candidates') else '?')
        return f'Grasp at {src}'
    if kind == 'release':
        return f"Release at {body.get('at')}"
    if kind == 'tool':
        if body.get('actuate'):
            a = body['actuate']
            return f"Tool: {a.get('actuator')} → {a.get('target')}"
        if body.get('verify'):
            return f"Tool: verify {body['verify'].get('expect')}"
        for which, title in (('close', 'Close gripper'), ('open', 'Open gripper')):
            if which in body:
                site = (body[which] or {}).get('site')
                return f'Tool: {title}' + (f' (site {site})' if site else '')
    return KIND_TITLES.get(kind, kind)


def label_text(label: Tuple[int, ...]) -> str:
    return '.'.join(map(str, label))


def route(task: dict, start: Optional[str] = None, paths: Optional[dict] = None) -> List[str]:
    """The motions of a task, in program order, as ``FROM → TO : HOW · profile · STOP``.

    Inside a control body the line says so, e.g. ``(in 3: Repeat ×2)``, and a path is one line through
    its states. ``start`` is where the robot is before the task, when it is known; ``paths`` (name ->
    path mapping) lets a path step say where it ends."""
    lines: List[str] = []
    at = start or '(where the task starts)'
    parents = {}
    for label, step in walk(task.get('steps') or []):
        kind = step_kind(step)
        if kind in ('repeat', 'retry', 'fallback'):
            parents[label] = step_text(step)
        where = ''
        for n in range(len(label) - 1, 0, -1):
            if label[:n] in parents:
                where = f'  (in {label_text(label[:n])}: {parents[label[:n]]})'
                break
        if kind == 'move':
            mv = step['move']
            how = mv['type'].upper() + (f" via {mv['via']}" if mv.get('via') else '')
            prof = f" · {mv['profile']}" if mv.get('profile') else ''
            lines.append(f"{label_text(label)}  {at} → {mv['to']} : {how}{prof} · STOP{where}")
            at = mv['to']
        elif kind == 'path':
            p = (paths or {}).get(step['path']) or {}
            states = p.get('states') or []
            through = ' → '.join(states) if states else 'its waypoints'
            end = (p.get('continuity') or 'stop').upper()
            lines.append(f"{label_text(label)}  path {step['path']}: {through} : one motion · {end}{where}")
            if states:
                at = states[-1]
        elif kind in ('grasp', 'release'):
            # owner Test F.1 (row 34): a grasp / release MOVES the robot -- above the target, in, out
            g = step[kind] or {}
            target = g.get('at') or (f"candidate {g.get('index', 0)} of "
                                     f"{g.get('candidates') or 'the latest detection'}")
            standoff = g.get('standoff_m')
            above = f'{standoff * 1000.0:g} mm above it' if standoff else 'above it'
            act = 'close' if kind == 'grasp' else 'open'
            lines.append(f"{label_text(label)}  {at} → {target} : {kind} · PTP {above} · LIN in · {act} · "
                         f"LIN out{where}")
            at = f'{target} (retracted)'
        elif kind == 'task':
            lines.append(f"{label_text(label)}  run task {step['task']} (its own route){where}")
    return lines


__all__ = ['KIND_TITLES', 'PROCESS_TITLES', 'label_text', 'route', 'step_text']
