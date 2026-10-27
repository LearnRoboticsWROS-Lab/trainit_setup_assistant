"""The end effector's model operations (vNext I0, I0_DESIGN.md §34 E1, owner decision 2026-10-26).

Step 7 edits the robot's end-effector CONTRACT (ADR-0022: actuators and their named targets, grasp sites, the
verification policy, an adapter per deployed backend) as it edits the program: every operation changes a copy
of the project AS WRITTEN and commits it through the canonical model (``Composer.edit_project``), in the same
undo history as the program's edits. Neither Qt nor ROS.

What an operation refuses is what the file could not hold (the model's own refusals) and what would leave a
dangling reference (a step or a site naming what is deleted). The contract's semantic rules (ADR-0022 R1-R26)
stay the generator's (``generator.eef_contract``), reported as diagnostics. Renames propagate to the steps that
name the device, an actuator, a target or a site, as ``rename_waypoint`` does for waypoints.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterator, List, Optional, Tuple

from ..catalog import EEF_BRIDGE_ROLES
from ..model.end_effector import DEFAULT_ACTUATOR_ID, DEFAULT_DEVICE_ID, DEFAULT_SITE_ID
from ..model.enums import GripperKind
from .ops import Composer, _refuse, walk

LEGACY_ROLE = 'eef_legacy_bridge'


# ---- reading ------------------------------------------------------------------------------------------

def gripper_of(c: Composer) -> dict:
    """The robot's gripper as written (a copy)."""
    from ..model.io import project_to_dict
    return copy.deepcopy(project_to_dict(c.project)['robot']['gripper'])


def contract_of(c: Composer) -> Optional[dict]:
    """The contract as written (a copy), None for a legacy end effector."""
    return gripper_of(c).get('contract')


def bridge_entry(row) -> Optional[dict]:
    """The ``catalog.EEF_BRIDGE_ROLES`` entry of a deployment bridge row (a mapping or a model), None for any
    other row."""
    get = row.get if isinstance(row, dict) else (lambda key: getattr(row, key, None))
    for r in EEF_BRIDGE_ROLES:
        if get('package') == r['package'] and get('executable') == r['executable']:
            return r
    return None


def bridge_role(row) -> Optional[str]:
    """What a deployment bridge row is (``catalog.EEF_BRIDGE_ROLES``), None for any other row."""
    entry = bridge_entry(row)
    return entry['role'] if entry else None


def legacy_bridges(c: Composer) -> List[str]:
    """The deployment's legacy end-effector bridge rows (refused beside a contract, R18), as
    ``package/executable``."""
    from ..model.io import project_to_dict
    rows = (project_to_dict(c.project).get('deployment') or {}).get('bridges') or []
    return [f"{r.get('package')}/{r.get('executable')}" for r in rows if bridge_role(r) == LEGACY_ROLE]


def actuator_ids(contract: Optional[dict], kind: Optional[str] = None) -> List[str]:
    """The actuators' ids; a single actuator without an ``id`` has the kind's default one (ADR-0022 D6:
    ``vacuum`` for a suction cup, ``gripper`` for a jaw)."""
    if not contract:
        return []
    if contract.get('actuators'):
        return list(contract['actuators'])
    if contract.get('actuator') is not None:
        default = DEFAULT_ACTUATOR_ID.get(GripperKind(kind)) if kind in GripperKind._value2member_map_ else None
        return [contract['actuator'].get('id') or default or '']
    return []


def _ids(c: Composer) -> List[str]:
    return actuator_ids(contract_of(c), gripper_of(c).get('kind'))


def references(c: Composer) -> Iterator[Tuple[str, dict, str]]:
    """Every end-effector reference of the program as ``(where, mapping, key)``: the steps' ``eef``,
    ``site``, ``actuator`` and ``target`` (grasp, release, tool actuate / verify / close / open)."""
    doc = c.document()
    for task in doc.get('tasks') or []:
        for label, step in walk(task.get('steps') or []):
            where = f"task '{task.get('name')}' step {'.'.join(map(str, label))}"
            kind = next(iter(step))
            body = step[kind] or {}
            if kind in ('grasp', 'release'):
                yield where, body, 'grasp/release'
            elif kind == 'tool':
                for form in ('actuate', 'verify', 'close', 'open'):
                    if form in body:
                        yield where, body[form] if body[form] is not None else {}, f'tool {form}'


def _fix_steps(app: dict, fix) -> None:
    """Apply ``fix(form, body)`` to every end-effector reference body of an application mapping (inside an
    ``edit_project`` mutation): a grasp / release body, a tool step's actuate / verify / close / open body."""
    for task in app.get('tasks') or []:
        for _label, step in walk(task.get('steps') or []):
            kind = next(iter(step))
            body = step[kind]
            if kind in ('grasp', 'release') and body is not None:
                fix('grasp/release', body)
            elif kind == 'tool' and body is not None:
                for form in ('actuate', 'verify', 'close', 'open'):
                    if body.get(form) is not None:
                        fix(f'tool {form}', body[form])


# ---- the contract as a whole --------------------------------------------------------------------------

def adopt_contract(c: Composer, kind: str, contract: dict) -> List[str]:
    """Make the end effector canonical: its ``kind`` and ``contract``, and the deployment's legacy end-effector
    bridges removed (the contract's adapters replace them, ADR-0022 R18), the legacy per-backend grasp
    adapters R23 would refuse next to its bindings back to their defaults -- ONE edit. Returns the removed bridge rows
    (``package/executable``)."""
    from .eef_suggest import legacy_sim_adapters        # here: eef_suggest reads this module
    GripperKind(kind)
    removed: List[str] = []
    replaced = legacy_sim_adapters(c.project, contract.get('adapters') or {})

    def mutate(d):
        g = d['robot']['gripper']
        g['kind'] = kind
        g['contract'] = copy.deepcopy(contract)
        g.pop('actuation', None)                   # 'contract': the gripper's pre-validator sets it
        # a legacy per-backend grasp adapter that disagrees with its binding (R23) back to its default; any other
        # kept, Gazebo's weld always (ROS2ML-SYNC D5)
        sim = g.get('sim_grasp_adapter')
        if isinstance(sim, dict):
            for backend in replaced:
                sim.pop(backend, None)
        dep = d.setdefault('deployment', {})
        keep = []
        for row in dep.get('bridges') or []:
            if bridge_role(row) == LEGACY_ROLE:
                removed.append(f"{row.get('package')}/{row.get('executable')}")
            else:
                keep.append(row)
        if 'bridges' in dep:
            dep['bridges'] = keep
    c.edit_project(mutate)
    return removed


def set_contract(c: Composer, contract: dict) -> None:
    """Replace the contract (the kind and the bridges unchanged)."""
    if contract_of(c) is None:
        _refuse('end-effector contract: the robot has none yet -- adopt one first (End effector section)')

    def mutate(d):
        d['robot']['gripper']['contract'] = copy.deepcopy(contract)
    c.edit_project(mutate)


def remove_contract(c: Composer) -> None:
    """Back to a legacy end effector: the contract dropped and the ``actuation`` token reset by kind (a
    contract token without its block is R19). The legacy bridges removed at adoption are NOT restored."""
    users = [where for where, _body, _form in references(c)]
    if users:
        _refuse('end-effector contract: the program commands the end effector (' + ', '.join(users)
                + '): remove those steps first')

    def mutate(d):
        g = d['robot']['gripper']
        g.pop('contract', None)
        g['actuation'] = ('joint_position' if g.get('kind') == GripperKind.PARALLEL.value else 'trigger')
        if d.get('schema_version') == 3:           # 3 is the contract's number (R19); a task program keeps 4
            d['schema_version'] = 2
    c.edit_project(mutate)


def _contract_edit(c: Composer, change) -> None:
    """Change the contract in place (``change(contract)``), in one edit."""
    if contract_of(c) is None:
        _refuse('end-effector contract: the robot has none yet -- adopt one first (End effector section)')

    def mutate(d):
        change(d['robot']['gripper']['contract'])
    c.edit_project(mutate)


def set_kind(c: Composer, kind: str) -> None:
    """The end effector's kind. A single actuator without an ``id`` keeps the id it had (the kind decides the
    DEFAULT id, ADR-0022 D6, and with it the endpoint names: python review N2)."""
    GripperKind(kind)
    keep = _ids(c)[0] if contract_of(c) and not _long(contract_of(c)) and _ids(c) else None

    def mutate(d):
        d['robot']['gripper']['kind'] = kind
        k = d['robot']['gripper'].get('contract')
        if keep and k and k.get('actuator') is not None and not k['actuator'].get('id'):
            if keep != DEFAULT_ACTUATOR_ID.get(GripperKind(kind)):
                k['actuator']['id'] = keep
    c.edit_project(mutate)


def set_verification(c: Composer, grasp: bool, release: bool) -> None:
    """The two ``require_*`` keys, always written (ADR-0022 R17: false is written, never implied)."""
    def change(k):
        v = k.setdefault('verification', {})
        v['require_grasp_confirmation'] = bool(grasp)
        v['require_release_confirmation'] = bool(release)
    _contract_edit(c, change)


def set_binding(c: Composer, backend: str, binding: Any) -> None:
    """The adapter of one deployed backend: a row token (``trainit_mock``, ``none``, ...), a long form
    ``{adapter, stage_cmd_topic, ...}``, or None to remove it."""
    def change(k):
        adapters = k.setdefault('adapters', {})
        if binding is None:
            adapters.pop(backend, None)
        else:
            adapters[backend] = copy.deepcopy(binding)
        if not adapters:
            k.pop('adapters', None)
    _contract_edit(c, change)


def set_device_id(c: Composer, new: str) -> None:
    """Rename the device; the steps that name it follow."""
    if contract_of(c) is None:
        _refuse('end-effector contract: the robot has none yet -- adopt one first (End effector section)')
    _check_name(new, 'device id')
    old = contract_of(c).get('id') or DEFAULT_DEVICE_ID
    if new == old:
        return

    def mutate(d):
        d['robot']['gripper']['contract']['id'] = new
        _fix_steps(d['application'], lambda form, body: body.__setitem__('eef', new)
                   if body.get('eef') == old else None)
    c.edit_project(mutate)


# ---- actuators, targets, sites --------------------------------------------------------------------------

def _long(k: dict) -> bool:
    return k.get('actuators') is not None


def _check_name(name: str, what: str) -> None:
    """A contract id is a ROS name token (ADR-0022 R3): refused here, where it is typed, not at generation."""
    from ..model.end_effector import NAME_TOKEN
    if not isinstance(name, str) or not NAME_TOKEN.match(name):
        _refuse(f"end-effector contract: {what} {name!r} is not a name (a letter, then letters, digits or '_')")


def _derived(c: Composer):
    """The contract's derived long form (ADR-0022 D6), None when it does not derive."""
    try:
        from ..model.end_effector import derive_device
        p = c.project
        return derive_device(p.robot.gripper, p.robot.named_states,
                             stage_topic=p.scene.gripper_cmd_topic or None)
    except Exception:  # noqa: BLE001 - an incomplete contract: nothing derived
        return None


def set_actuator(c: Composer, actuator_id: str, spec: dict) -> None:
    """Add or replace an actuator: in a long form, ``actuators[actuator_id]``; in a short form, THE
    actuator (its ``id`` set when it is not the kind's default)."""
    def change(k):
        body = {key: v for key, v in copy.deepcopy(spec).items() if key != 'id'}
        if _long(k):
            k['actuators'][actuator_id] = body
        else:
            kind = gripper_of(c).get('kind')
            default = DEFAULT_ACTUATOR_ID.get(GripperKind(kind)) if kind in GripperKind._value2member_map_ else None
            if actuator_id and actuator_id != default:
                body['id'] = actuator_id
            k['actuator'] = body
    _contract_edit(c, change)


def add_actuator(c: Composer, actuator_id: str, spec: dict) -> None:
    """A NEW actuator of a multi-actuator tool (python review S3: adding never replaces one)."""
    _check_name(actuator_id, 'actuator id')
    k = contract_of(c)
    if k is None or not _long(k):
        _refuse('end-effector contract: a single-actuator device has one actuator -- make it a multi-actuator '
                'tool first to add more')
    if actuator_id in _ids(c):
        _refuse(f"end-effector contract: an actuator is already named '{actuator_id}' (edit it instead)")
    set_actuator(c, actuator_id, spec)


def actuator_users(c: Composer, actuator_id: str) -> List[str]:
    """The steps and sites that name an actuator -- a grasp, release, close / open or verify included, which
    reaches it through the site it grips (owner Test F.1, row 35: 'vacuum' showed "Used by --")."""
    out = [where for where, body, form in references(c)
           if form == 'tool actuate' and body.get('actuator') == actuator_id]
    dev = _derived(c)
    if dev is not None:
        grips = {site.id: site.grip for site in dev.sites}
        only = dev.sites[0].id if len(dev.sites) == 1 else None
        for where, body, form in references(c):
            site = body.get('site') or only
            if form != 'tool actuate' and site is not None and grips.get(site) == actuator_id:
                out.append(f"{where} (site '{site}')")
    k = contract_of(c) or {}
    out += [f"site '{s}'" for s, spec in (k.get('sites') or {}).items() if (spec or {}).get('grip') == actuator_id]
    return out


def rename_actuator(c: Composer, old: str, new: str) -> None:
    _check_name(new, 'actuator id')
    if old not in _ids(c):
        _refuse(f"end-effector contract: no actuator is named '{old}'")
    if new in _ids(c):
        _refuse(f"end-effector contract: an actuator is already named '{new}'")

    def mutate(d):
        kk = d['robot']['gripper']['contract']
        if _long(kk):
            kk['actuators'] = {(new if a == old else a): v for a, v in kk['actuators'].items()}
            for spec in (kk.get('sites') or {}).values():
                if spec and spec.get('grip') == old:
                    spec['grip'] = new
        else:
            kk['actuator']['id'] = new
        _fix_steps(d['application'], lambda form, body: body.__setitem__('actuator', new)
                       if form == 'tool actuate' and body.get('actuator') == old else None)
    c.edit_project(mutate)


def delete_actuator(c: Composer, actuator_id: str) -> None:
    users = actuator_users(c, actuator_id)
    if users:
        _refuse(f"end-effector contract: actuator '{actuator_id}' is used by {', '.join(users)}")
    k = contract_of(c) or {}
    if not _long(k):
        _refuse('end-effector contract: a single-actuator device keeps its actuator (remove the contract '
                'instead, or make the tool multi-actuator first)')

    def change(kk):
        kk['actuators'].pop(actuator_id, None)
    _contract_edit(c, change)


def target_users(c: Composer, actuator_id: str, target: str) -> List[str]:
    """The steps and the sites that name a target -- a site's DERIVED engage / release included (python review
    S1: a site without written ones derives 'closed' / 'open' from its grip)."""
    out = [where for where, body, form in references(c)
           if form == 'tool actuate' and body.get('actuator') == actuator_id and body.get('target') == target]
    dev = _derived(c)
    for site in (dev.sites if dev else []):
        if site.grip == actuator_id and target in (site.engage, site.release):
            out.append(f"site '{site.id}'")
    return out


def _actuator_body(k: dict, actuator_id: str) -> dict:
    if _long(k):
        if actuator_id not in k['actuators']:
            _refuse(f"end-effector contract: no actuator is named '{actuator_id}'")
        return k['actuators'][actuator_id]
    return k['actuator']


def set_target(c: Composer, actuator_id: str, name: str, value: Any) -> None:
    """A named target of a ``gripper_command`` (a number) or ``follow_joint_trajectory`` actuator (one value
    per joint). A ``set_bool`` actuator's targets are always ``engage`` / ``release``."""
    _check_name(name, 'target')
    dev = _derived(c)
    derived = dev.actuator(actuator_id) if dev else None

    def change(k):
        body = _actuator_body(k, actuator_id)
        if body.get('command_interface') == 'set_bool':
            _refuse(f"end-effector contract: actuator '{actuator_id}' is binary (set_bool): its targets are "
                    "always 'engage' and 'release'")
        if 'targets' not in body and derived is not None and derived.targets:
            # python review M3: writing ONE target would stop the derivation of the others (a jaw's SRDF open /
            # closed): they are written first, with their derived values
            body['targets'] = {t: (list(v) if isinstance(v, (list, tuple)) else v) for t, v in derived.targets.items()}
        body.setdefault('targets', {})[name] = copy.deepcopy(value)
    _contract_edit(c, change)


def rename_target(c: Composer, actuator_id: str, old: str, new: str) -> None:
    _check_name(new, 'target')
    dev = _derived(c)
    site_names = {s.id: (s.engage, s.release) for s in (dev.sites if dev else []) if s.grip == actuator_id}
    if not _long(contract_of(c) or {}) and any(old in names for names in site_names.values()):
        _refuse(f"end-effector contract: '{old}' is the single actuator's grip target, fixed by the short form "
                '(ADR-0022 D6) -- make it a multi-actuator tool to rename it')

    def mutate(d):
        kk = d['robot']['gripper']['contract']
        body = _actuator_body(kk, actuator_id)
        targets = body.get('targets') or {}
        if old not in targets:
            _refuse(f"end-effector contract: actuator '{actuator_id}' has no named target '{old}'")
        if new in targets:
            _refuse(f"end-effector contract: actuator '{actuator_id}' already has a target '{new}'")
        body['targets'] = {(new if t == old else t): v for t, v in targets.items()}
        for sid, spec in (kk.get('sites') or {}).items():
            if spec and spec.get('grip') == actuator_id:
                engage, release = site_names.get(sid, (None, None))
                for key, derived in (('engage', engage), ('release', release)):
                    if (spec.get(key) or derived) == old:     # written, or derived from the grip (S1)
                        spec[key] = new
        _fix_steps(d['application'], lambda form, body: body.__setitem__('target', new)
                       if form == 'tool actuate' and body.get('actuator') == actuator_id
                       and body.get('target') == old else None)
    c.edit_project(mutate)


def delete_target(c: Composer, actuator_id: str, name: str) -> None:
    users = target_users(c, actuator_id, name)
    if users:
        _refuse(f"end-effector contract: target '{name}' of '{actuator_id}' is used by {', '.join(users)}")

    def change(k):
        targets = _actuator_body(k, actuator_id).get('targets') or {}
        if name not in targets:                 # derived (ADR-0022 D6) or absent: nothing written to delete
            _refuse(f"end-effector contract: actuator '{actuator_id}' has no written target '{name}'")
        targets.pop(name)
    _contract_edit(c, change)


def site_users(c: Composer, site: str) -> List[str]:
    return [where for where, body, _form in references(c) if body.get('site') == site]


def set_site(c: Composer, site: str, spec: dict) -> None:
    """Add or replace a grasp site (a long-form contract)."""
    _check_name(site, 'site')

    def change(k):
        if not _long(k):
            _refuse('end-effector contract: a single-actuator device has one implicit site; make the tool '
                    'multi-actuator to declare sites')
        k.setdefault('sites', {})[site] = copy.deepcopy(spec)
    _contract_edit(c, change)


def rename_site(c: Composer, old: str, new: str) -> None:
    _check_name(new, 'site')

    def mutate(d):
        kk = d['robot']['gripper']['contract']
        sites = kk.get('sites') or {}
        if old not in sites:
            _refuse(f"end-effector contract: no grasp site is named '{old}'")
        if new in sites:
            _refuse(f"end-effector contract: a grasp site is already named '{new}'")
        kk['sites'] = {(new if s == old else s): v for s, v in sites.items()}
        _fix_steps(d['application'], lambda form, body: body.__setitem__('site', new)
                       if body.get('site') == old else None)
    c.edit_project(mutate)


def delete_site(c: Composer, site: str) -> None:
    users = site_users(c, site)
    if users:
        _refuse(f"end-effector contract: grasp site '{site}' is used by {', '.join(users)}")

    def change(k):
        (k.get('sites') or {}).pop(site, None)
    _contract_edit(c, change)


# ---- short -> long form ---------------------------------------------------------------------------------

def to_long_form(c: Composer) -> None:
    """Make a single-actuator contract multi-actuator: its derived values (ADR-0022 D6) written out, its
    implicit site named. A derived value that does not resolve (e.g. a jaw target with no SRDF state) is
    refused by name: write it first."""
    from ..model.end_effector import derive_device
    k = contract_of(c)
    if k is None or _long(k):
        return
    dev = derive_device(c.project.robot.gripper, c.project.robot.named_states,
                        stage_topic=c.project.scene.gripper_cmd_topic or None)
    act = dev.actuators[0]
    if act.unresolved:
        _refuse('end-effector contract: the single actuator has values that do not resolve ('
                + '; '.join(f'{key}: {why}' for key, why in act.unresolved.items()) + '): set them first')
    # what the user wrote stays as written (python review S2: e.g. confirm_timeout_ms); the derived values
    # fill only what was not written
    spec: Dict[str, Any] = {key: v for key, v in copy.deepcopy(k['actuator']).items() if key != 'id'}
    spec['command_interface'] = act.interface.value
    for key in ('action', 'joints', 'targets', 'max_effort', 'goal_tolerance', 'duration_s', 'path_tolerance'):
        value = getattr(act, key, None)
        if key not in spec and value is not None and not (act.interface.value == 'set_bool' and key == 'targets'):
            spec[key] = copy.deepcopy(list(value) if isinstance(value, tuple) else value)
    if 'result_policy' not in spec and act.result_policy is not None:
        spec['result_policy'] = getattr(act.result_policy, 'value', act.result_policy)
    if 'joint_positions' not in spec and act.joint_positions is not None:
        engaged, released = act.joint_positions
        spec['joint_positions'] = {'engaged': list(engaged), 'released': list(released)}
    site = dev.sites[0]

    def change(kk):
        kk.pop('actuator', None)
        kk['actuators'] = {act.id: spec}
        kk['sites'] = {site.id or DEFAULT_SITE_ID: {'tcp': site.tcp, 'grip': act.id}}
    _contract_edit(c, change)


__all__ = ['LEGACY_ROLE', 'actuator_ids', 'actuator_users', 'add_actuator', 'adopt_contract', 'bridge_entry', 'bridge_role', 'contract_of',
           'delete_actuator', 'delete_site', 'delete_target', 'gripper_of', 'legacy_bridges', 'references',
           'remove_contract', 'rename_actuator', 'rename_site', 'rename_target', 'set_actuator', 'set_binding',
           'set_contract', 'set_device_id', 'set_kind', 'set_site', 'set_target', 'set_verification',
           'site_users', 'target_users', 'to_long_form']
