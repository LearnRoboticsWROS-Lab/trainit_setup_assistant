"""Load/save a :class:`CanonicalProject` to ``project.yaml`` (round-trip safe)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel

from .enums import DEPRECATED_APP_TYPES, AppType, deprecated_app_type_refusal
from .program import describe_load_error
from .project import (SCHEMA_VERSION_EEF_CONTRACT, SCHEMA_VERSION_TASK_PROGRAM,
                      CanonicalProject)

# The ToolAction fields of the end-effector contract (ADR-0022): written only when set.
_EEF_TOOL_ACTION_KEYS = ('actuator', 'target', 'site')
# Where the contract lives in project.yaml (ADR-0022 D6), key by key.
_CONTRACT_PATH = ('robot', 'gripper', 'contract')


def load_project(path: Union[str, Path]) -> CanonicalProject:
    """Parse + validate a project.yaml into a CanonicalProject."""
    text = Path(path).read_text()
    data = yaml.safe_load(text)
    if data is None:
        raise ValueError(f'empty project file: {path}')
    application = data.get('application') if isinstance(data, dict) else None
    token = application.get('type') if isinstance(application, dict) else None
    if isinstance(token, str) and token in DEPRECATED_APP_TYPES:      # M3 decision 7 (TSA 8.0)
        raise ValueError(f'{path}: ' + deprecated_app_type_refusal(token))
    if _contract_of(data) is not None:
        refuse_duplicate_contract_keys(text, path)
    if not _is_task_program(data):
        return CanonicalProject.parse_obj(data)
    # A Canonical Task Program (vNext M3.3): a refusal names the task, step, waypoint or path in the
    # file's own terms ("task 'scan' step 3 (move)"), not pydantic's internal location.
    try:
        return CanonicalProject.parse_obj(data)
    except ValueError as exc:
        raise ValueError(describe_load_error(exc, data, path)) from None


def _is_task_program(data) -> bool:
    application = data.get('application') if isinstance(data, dict) else None
    return isinstance(application, dict) and application.get('type') == AppType.TASK_PROGRAM.value


def _contract_of(data):
    node = data
    for key in _CONTRACT_PATH:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def refuse_duplicate_contract_keys(text: str, path: Union[str, Path] = '<text>') -> None:
    """R3 'a duplicate' (ADR-0022): a contract id is a mapping KEY, so two actuators, sites or
    targets with one id can only be a duplicate key in the file -- which ``yaml.safe_load``
    silently resolves to the LAST one, dropping the first definition without a word. Refused at
    load, naming the mapping, the key and both lines. Read from the document's node tree
    (``yaml.compose``); only the contract block is checked, so a legacy project loads exactly as
    it always did."""
    loader = yaml.SafeLoader('')
    node = yaml.compose(text, Loader=yaml.SafeLoader)
    dotted = []
    for key in _CONTRACT_PATH:
        if not isinstance(node, yaml.MappingNode):
            return
        found = [v for k, v in node.value
                 if isinstance(k, yaml.ScalarNode) and k.value == key]
        if not found:
            return
        if key == _CONTRACT_PATH[-1] and len(found) > 1:
            _duplicate('.'.join(dotted), key, node, path)
        node = found[-1]                       # the one safe_load kept
        dotted.append(key)
    stack = [('.'.join(dotted), node)]
    while stack:
        where, current = stack.pop()
        if isinstance(current, yaml.MappingNode):
            seen = {}
            for k, v in current.value:
                if not isinstance(k, yaml.ScalarNode):
                    continue
                try:                                   # the key safe_load would build: 'on'
                    ident = loader.construct_object(k)  # and 'yes' are one key (True)
                except yaml.YAMLError:
                    ident = (k.tag, k.value)
                if ident in seen:
                    raise ValueError(
                        f'end-effector contract (ADR-0022 R3): {path}: {where} has the key '
                        f'{k.value!r} twice (lines {seen[ident] + 1} and '
                        f'{k.start_mark.line + 1}) -- an id names ONE actuator, site or target, '
                        'and YAML would silently keep only the last one.')
                seen[ident] = k.start_mark.line
                stack.append((f'{where}.{k.value}', v))
        elif isinstance(current, yaml.SequenceNode):
            stack.extend((f'{where}[{i}]', v) for i, v in enumerate(current.value))


def _duplicate(where: str, key: str, node, path) -> None:
    lines = [k.start_mark.line + 1 for k, _v in node.value
             if isinstance(k, yaml.ScalarNode) and k.value == key]
    raise ValueError(f'end-effector contract (ADR-0022 R3): {path}: {where} has the key '
                     f'{key!r} twice (lines {lines[0]} and {lines[-1]}) -- one device, one '
                     'contract block.')


def contract_as_written(contract) -> dict:
    """The contract as the user wrote it (ADR-0022 D6 ``io.py``): every key whose value equals its
    field's default DROPPED, ``default_factory`` fields included, inside every nested model --
    the same bytes on pydantic 1.x and 2.x. (``exclude_defaults`` alone differs: pydantic 1.x
    compares a ``default_factory`` field with None and keeps ``accepted_evidence`` / ``adapters:
    {}`` -- review TSA-4.)"""
    return _without_defaults(contract)


def _fields(model: BaseModel) -> dict:
    return getattr(type(model), 'model_fields', None) or type(model).__fields__


def _default_of(field):
    """(True, default) for a field that has one, (False, None) for a required field -- on a
    pydantic 2 FieldInfo and a pydantic 1 ModelField alike."""
    factory = getattr(field, 'default_factory', None)
    if factory is not None:
        return True, factory()
    is_required = getattr(field, 'is_required', None)
    required = is_required() if callable(is_required) else bool(getattr(field, 'required', False))
    return (False, None) if required else (True, field.default)


def _without_defaults(model: BaseModel) -> dict:
    full = json.loads(model.json())              # plain tokens, on both versions
    out = {}
    for name, field in _fields(model).items():
        if name not in full:
            continue                             # excluded from the file (Field(exclude=True))
        value = getattr(model, name)
        has_default, default = _default_of(field)
        if has_default and value == default:
            continue
        if isinstance(value, BaseModel):
            out[name] = _without_defaults(value)
        elif isinstance(value, dict) and any(isinstance(v, BaseModel) for v in value.values()):
            out[name] = {key: (_without_defaults(v) if isinstance(v, BaseModel) else plain)
                         for (key, plain), v in zip(full[name].items(), value.values())}
        else:
            out[name] = full[name]
    return out


def project_to_dict(project: CanonicalProject) -> dict:
    """Plain-data dict (enums -> their string tokens) for YAML/JSON dumping.

    ``json.loads(project.json())`` — NOT ``.dict()`` — reproduces pydantic v2's
    ``model_dump(mode='json')`` on BOTH pydantic 1.9 and 2.x: every StrEnum becomes its
    plain-string token and nested models become dicts, so ``yaml.safe_dump`` writes the
    same project.yaml. A bare ``.dict()`` would keep StrEnum *members* (a str subclass),
    which ``yaml.safe_dump`` cannot represent -> a changed/broken file. ``.json()`` is
    present (deprecated in 2.x) on both versions.

    The end-effector contract (ADR-0022 D6 ``io.py``) is written so a LEGACY project saves
    byte-identical to TSA 7.6.0: ``robot.gripper.contract`` is omitted when None, and so are the
    new ToolAction fields (``actuator``, ``target``, ``site``) and the never-saved
    ``actuation_overridden`` (dropped here too, for a pydantic that ignores ``Field(exclude=)``).
    A contract is written AS STORED -- a short form stays a short form, nothing derived is added
    -- with its default-valued keys dropped (inside the contract only,
    :func:`contract_as_written`), and ``schema_version`` is then 3."""
    data = json.loads(project.json())
    gripper = (data.get('robot') or {}).get('gripper')
    contract = project.robot.gripper.contract
    if isinstance(gripper, dict):
        gripper.pop('actuation_overridden', None)
        if contract is None:
            gripper.pop('contract', None)
        else:
            gripper['contract'] = contract_as_written(contract)
            data['schema_version'] = SCHEMA_VERSION_EEF_CONTRACT
    for action in (data.get('application') or {}).get('tool_actions') or []:
        for key in _EEF_TOOL_ACTION_KEYS:
            if action.get(key) is None:
                action.pop(key, None)
    # vNext M3.5: an external node's detect_service is written only when declared, so a project
    # saved before it existed saves byte-identical
    for det in ((data.get('perception') or {}).get('detectors') or []):
        node = det.get('external_node') if isinstance(det, dict) else None
        if isinstance(node, dict) and node.get('detect_service') is False:
            node.pop('detect_service')
    application = data.get('application')
    if isinstance(application, dict):
        application.pop('program', None)        # a template application: no program, no key
        if project.application.is_task_program():
            # vNext M3.3: the program's sections directly under `application:`, as the user wrote
            # them (every key that was set, none that was not), and schema_version 4
            data['application'] = {'type': AppType.TASK_PROGRAM.value,
                                   **program_as_written(project.application.program)}
            data['schema_version'] = SCHEMA_VERSION_TASK_PROGRAM
    return data


def program_as_written(program) -> dict:
    """A Canonical Task Program as its file states it: every key that was set (a default the user
    wrote stays written), none that was not -- the same on pydantic 1.x and 2.x. A step, a path or a
    segment writes its own keys (name, type, to, states, ...) before the limits it shares with a
    motion profile, so a saved file reads like the one the user wrote."""
    return _own_keys_first(program, json.loads(program.json(exclude_unset=True)))


def _own_keys_first(model, data):
    from .motion import MotionConstraints
    if isinstance(model, BaseModel) and isinstance(data, dict):
        names = list(_fields(model))
        shared = set(_fields(MotionConstraints())) if isinstance(model, MotionConstraints) else set()
        order = [n for n in names if n not in shared] + [n for n in names if n in shared]
        return {n: _own_keys_first(getattr(model, n), data[n]) for n in order if n in data}
    if isinstance(model, list) and isinstance(data, list):
        return [_own_keys_first(m, d) for m, d in zip(model, data)]
    if isinstance(model, dict) and isinstance(data, dict):
        return {k: _own_keys_first(model.get(k), v) for k, v in data.items()}
    return data


def save_project(project: CanonicalProject, path: Union[str, Path]) -> Path:
    """Write a project.yaml. Keys keep model order (sort_keys=False). A Canonical Task Program
    writes its innermost lists and mappings inline (``position: [0.45, 0.25, 0.3]``, ``- move: {to:
    w1, type: ptp}``), as it is written by hand; a template application's file is written exactly as
    before."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    flow = None if project.application.is_task_program() else False
    out.write_text(
        yaml.safe_dump(project_to_dict(project), sort_keys=False, default_flow_style=flow)
    )
    return out
