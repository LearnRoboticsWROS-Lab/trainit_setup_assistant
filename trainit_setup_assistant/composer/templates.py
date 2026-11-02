"""Application templates (vNext I0.10, I0_DESIGN.md §19): start from a canonical program, not from blank.

A template is installed with TSA (``resources/templates/<id>.yaml``). It holds a ``template`` header
(title, description) and an ``application:`` section that is pure Canonical Task Program data, with no
template-specific runtime. Its waypoint values are placeholders that the user re-teaches.

Applying a template replaces the project's application, through the canonical model. Everything else
in the project (robot, scene, perception, deployment) stays. :func:`requirements` says what the template
expects the project to provide: detectors, MoveIt named states, logical channels. The validator then
reports what is still missing.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import yaml

from ..model.enums import AppType
from ..model.io import project_to_dict
from ..model.project import SCHEMA_VERSION_TASK_PROGRAM, CanonicalProject
from .ops import step_kind, walk

TEMPLATE_DIR = Path(__file__).resolve().parents[1] / 'resources' / 'templates'


@dataclass(frozen=True)
class Template:
    id: str
    title: str
    description: str
    application: dict


def list_templates() -> List[Template]:
    out = []
    for f in sorted(TEMPLATE_DIR.glob('*.yaml')):
        data = yaml.safe_load(f.read_text())
        head = data.get('template') or {}
        out.append(Template(f.stem, head.get('title', f.stem), head.get('description', ''), data['application']))
    order = ['simple_motion', 'pick_and_place', 'machine_tending', 'inspection', 'process_path', 'bin_picking']
    return sorted(out, key=lambda t: (order.index(t.id) if t.id in order else len(order), t.id))


def load_template(template_id: str) -> Template:
    for t in list_templates():
        if t.id == template_id:
            return t
    raise KeyError(f"no template '{template_id}' (templates: {', '.join(t.id for t in list_templates())})")


def requirements(application: dict) -> Dict[str, List[str]]:
    """What a program expects the project to provide: the detectors its captures / detect steps name,
    the MoveIt named states its waypoints name, and the logical channels it switches."""
    out = {'detectors': [], 'named_states': [], 'channels': []}

    def add(key, value):
        if value and value not in out[key]:
            out[key].append(value)
    for wp in application.get('waypoints') or []:
        add('named_states', wp.get('named'))
    for p in application.get('paths') or []:
        for ev in p.get('events') or []:
            add('channels', ev.get('channel'))
            add('detectors', ev.get('detector'))
    for t in application.get('tasks') or []:
        for _label, step in walk(t.get('steps') or []):
            kind = step_kind(step)
            body = step[kind]
            if kind in ('process', 'detect') and isinstance(body, dict):
                add('channels', body.get('channel'))
                add('detectors', body.get('detector'))
    return out


def missing(project: CanonicalProject, application: dict) -> Dict[str, List[str]]:
    """The requirements the project does not provide yet: what to configure after applying."""
    need = requirements(application)
    detectors = {d.name for d in (project.perception.detectors if project.perception else [])}
    states = {s.name for s in project.robot.named_states}
    return {'detectors': [d for d in need['detectors'] if d not in detectors],
            'named_states': [s for s in need['named_states'] if s not in states],
            'channels': need['channels']}          # a channel is a logical id: nothing to provide


def apply_template(project: CanonicalProject, template_id: str) -> CanonicalProject:
    """The project with its application replaced by the template's (a new project, through the model)."""
    data = project_to_dict(project)
    data['application'] = copy.deepcopy(load_template(template_id).application)
    data['application']['type'] = AppType.TASK_PROGRAM.value
    data['schema_version'] = SCHEMA_VERSION_TASK_PROGRAM
    return CanonicalProject.parse_obj(data)


__all__ = ['TEMPLATE_DIR', 'Template', 'apply_template', 'list_templates', 'load_template', 'missing',
           'requirements']
