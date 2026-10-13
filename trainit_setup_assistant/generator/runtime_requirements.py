"""What a generated package needs to RUN: the oldest TMR it can use, and the policy runtime (vNext M2.4).

ADR-0021 amendment M2.4. A package's TMR floor is the highest ``catalog.TMR_FLOORS`` version among
the TMR capabilities its OWN generated files use -- ``TMR_FIRST_VERSION`` when none is newer -- and
its package.xml declares it as ``version_gte``, with one comment naming what set it. The
capabilities are read from the files the generator wrote for that package, never from the
project's intent:

  the BT trees      every node (a node TSA does not date is a TSA defect: RuntimeError), the ports
                    and port values that arrived later (DetectObject ``keep``,
                    SetWaypointFromDetection ``index``, keep <= 1's first-published match), the
                    scene nodes when the tree runs in a namespace
  the bt_params     the joint-position gripper targets, the end-effector contract's ``eef`` block
  the launch files  the launch helper's ``backend=``, and whether trainit_run_bt and
                    scene_manager_node run under a ``namespace=`` -- parsed with ``ast``: a TMR node
                    started with a literal ``package=``/``executable=``, its namespace a non-empty
                    expression. A TMR node or a launch-helper call that cannot be read that way is a
                    TSA defect (RuntimeError), so a template refactor cannot lower a floor silently.

The learned-policy runtime is needed iff a tree runs RunPolicy or a launch file starts its node.

The end-effector contract (ADR-0022) adds two things, for a canonical project only: its TMR rows
(VerifyGrasp, CommandActuator, DetectObject's ``fault``, the ``eef`` block: TMR 2.0.0;
MoveWaypoint's ``fault``: TMR 2.1.0, vNext M2.7; and, when the package's ``trainit_run_bt`` runs
under a namespace, its MoveIt clients resolving there: TMR 2.1.0, vNext G0 -- a LEGACY namespaced
package relies on that too but keeps its floor, by the owner's rule), and
``trainit_hardware_layer`` with ``version_gte`` iff a generated launch starts one of its adapters
(:func:`hardware_layer_requirement`, fail closed: a launch it cannot read stops generation).

A generated file that does not parse (a project name that breaks the XML or the Python TSA
writes) cannot be read: the floor is then the newest version the table dates, never lower, and
the file is reported (``Use.unreadable``) for a manifest warning. "The table" is every row a
package of that contract can use: a LEGACY package (no ``eef`` block in its bt_params) never
assumes the end-effector contract's rows, so its floor is the one it had before they existed.

Pure: it reads no installed TMR, no ament index and no catalog snapshot -- the snapshot's TMR
version says what COMPILED a bundle, not what it needs (M2.4-1).
"""
from __future__ import annotations

import ast
import re
import xml.etree.ElementTree as ET
from typing import (Dict, FrozenSet, Iterable, List, Mapping, NamedTuple, Optional, Set,
                    Tuple, Union)

import yaml

from ..catalog import (BT_BUILTIN_NODES, EEF_ADAPTER_ROWS, EEF_CONTRACT_CAPABILITIES,
                       HARDWARE_LAYER, TMR_BT_NODES, TMR_FIRST_VERSION, TMR_FLOORS)

_TMR = 'trainit_motion_runtime'
_POLICY_RUNTIME = 'trainit_policy_runtime'
# the scene nodes whose default service names became relative in TMR 1.7.0 (fd4fe5a)
_SCENE_SERVICE_NODES = ('SetAttachedCollisionCheck', 'ResetScene')
_COUNT = re.compile(r'[0-9]+')
# a TMR node started under a namespace -> the fact it is
_NAMESPACE_FACTS = {'scene_manager_node': 'scene_manager_namespace',
                    'trainit_run_bt': 'tree_namespaced'}
Texts = Union[Mapping[str, str], Iterable[str]]


class Requirement(NamedTuple):
    """A package's TMR floor and the one-line package.xml comment that says what set it."""

    version: str
    comment: str


class Use(NamedTuple):
    """What one package's generated files use."""

    capabilities: Tuple[str, ...]       # TMR_FLOORS keys, in table order
    executables: FrozenSet[str]         # the TMR nodes its launch files start
    policy_runtime: bool                # a tree runs RunPolicy, or a launch file starts its node
    unreadable: Tuple[str, ...]         # '<file> (<error> at line <n>)': files that do not parse
    # its bt_params carry the end-effector contract's `eef` block (ADR-0022), or one that may
    # does not parse: only then does a file that does not parse assume the contract's rows, and
    # only then does a namespaced trainit_run_bt need TMR 2.1.0 (vNext G0)
    canonical: bool = False


def _version_key(version: str) -> tuple:
    return tuple(int(part) for part in version.split('.'))


def _named(texts: Texts, kind: str) -> List[Tuple[str, str]]:
    if isinstance(texts, Mapping):
        return sorted(texts.items())
    return [(f'{kind} {i}', text) for i, text in enumerate(texts)]


def _where(name: str, exc: Exception) -> str:
    """The file and the kind of error -- never the parser's message, which quotes project data."""
    line = getattr(exc, 'lineno', None)
    if line is None and isinstance(getattr(exc, 'position', None), tuple):
        line = exc.position[0]                       # ET.ParseError
    return f'{name} ({type(exc).__name__}' + (f' at line {line}' if line else '') + ')'


def _tree(text: str) -> ET.Element:
    start = text.find('<root')
    if start < 0:
        raise ValueError('no <root> element')
    # past the XML declaration, keeping the file's line numbers for a parse error
    return ET.fromstring('\n' * text.count('\n', 0, start) + text[start:])


def _tree_capabilities(root: ET.Element, namespaced: bool, used: Set[str]) -> bool:
    """Adds what one parsed tree uses to ``used``; True if it runs a policy step."""
    policy = False
    for el in root.iter():
        tag = el.tag
        if tag in BT_BUILTIN_NODES:
            continue
        if tag not in TMR_BT_NODES:
            raise RuntimeError(
                f'TSA emitted the BT node {tag!r}, whose TMR floor its catalog does not know: '
                f'date it in catalog TMR_BT_NODES / TMR_FLOORS (vNext M2.4)')
        if tag in TMR_FLOORS:
            used.add(tag)
        if tag == 'RunPolicy':
            policy = True
        if tag == 'DetectObject':
            keep = el.get('keep', '1')
            if not _COUNT.fullmatch(keep):
                raise RuntimeError(f'TSA emitted DetectObject keep={keep!r}: not a count')
            if 'keep' in el.attrib:
                used.add('detect_keep')
            if int(keep) <= 1:                   # TMR reads keep < 1 as 1
                used.add('detect_first_published')
        if tag == 'SetWaypointFromDetection' and 'index' in el.attrib:
            used.add('detection_index')
        if tag == 'DetectObject' and 'fault' in el.attrib:
            used.add('detect_fault')                 # ADR-0022 D5: the canonical gate
        if tag == 'MoveWaypoint' and 'fault' in el.attrib:
            used.add('move_waypoint_fault')          # ADR-0022 amendment M2.7: the motion gate
        if namespaced and tag in _SCENE_SERVICE_NODES:
            used.add('scene_services_namespace')
    return policy


def _called(call: ast.Call) -> str:
    func = call.func
    return func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')


def _literal(node) -> Optional[str]:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _launch_facts(name: str, tree: ast.AST, facts: Set[str], executables: Set[str]) -> bool:
    """Adds the TMR facts of one parsed launch file; True if it starts the policy runtime."""
    policy = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}
        spread = any(k.arg is None for k in node.keywords)
        if _called(node) == 'build_move_group_params':
            if spread:
                raise RuntimeError(f'{name}: build_move_group_params(**...) -- the TMR floor cannot '
                                   f'tell whether backend= is passed (vNext M2.4)')
            if 'backend' in kw:
                facts.add('launch_backend')
        if _called(node) != 'Node':
            continue
        package = _literal(kw.get('package'))
        if package == _POLICY_RUNTIME:
            policy = True
        if package != _TMR:
            continue                             # another package, or a bridge read from data
        executable = _literal(kw.get('executable'))
        if executable is None or spread:
            raise RuntimeError(f'{name}: a trainit_motion_runtime Node whose executable or '
                               f'namespace the TMR floor cannot read (vNext M2.4)')
        executables.add(executable)
        namespace = kw.get('namespace')
        in_slot = namespace is not None and not (isinstance(namespace, ast.Constant)
                                                 and namespace.value in ('', None))
        if in_slot and executable in _NAMESPACE_FACTS:
            facts.add(_NAMESPACE_FACTS[executable])
    return policy


def use_of(trees: Texts = (), bt_params: Texts = (), launches: Texts = ()) -> Use:
    """What one package's generated trees, bt_params and launch files use (module docstring)."""
    unreadable: List[str] = []
    facts: Set[str] = set()
    executables: Set[str] = set()
    policy = False
    for name, text in _named(launches, 'launch'):
        try:
            parsed = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            unreadable.append(_where(name, exc))
            policy |= bool(re.search(rf'package\s*=\s*["\']{_POLICY_RUNTIME}["\']', text))
            continue
        policy |= _launch_facts(name, parsed, facts, executables)
    used = {f for f in facts if f in TMR_FLOORS}
    for name, text in _named(trees, 'tree'):
        try:
            root = _tree(text)
        except (ET.ParseError, ValueError) as exc:
            unreadable.append(_where(name, exc))
            policy |= '<RunPolicy' in text
            continue
        policy |= _tree_capabilities(root, 'tree_namespaced' in facts, used)
    canonical = False
    for name, text in _named(bt_params, 'bt_params'):
        try:
            data = yaml.safe_load(text) or {}
            params = [(body or {}).get('ros__parameters') or {} for body in data.values()]
        except (yaml.YAMLError, AttributeError) as exc:
            unreadable.append(_where(name, exc))
            canonical |= bool(re.search(r'^\s+eef:\s*$', text, re.M))
            continue
        if any('gripper_close_position' in p or 'gripper_open_position' in p for p in params):
            used.add('gripper_positions')
        if any('eef' in p for p in params):
            used.add('eef_block')                    # ADR-0022: the end-effector contract
    canonical |= 'eef_block' in used
    if canonical and 'tree_namespaced' in facts:
        # vNext G0 (owner section 5, ADR-0023): a canonical runner in a cell namespace relies on
        # TMR 2.1.0's MoveIt clients resolving there; a legacy one keeps its floor (owner M2.7)
        used.add('moveit_clients_namespace')
    return Use(tuple(n for n in TMR_FLOORS if n in used), frozenset(executables), policy,
               tuple(unreadable), canonical)


def requirement(use: Use, needs: Iterable[str] = ()) -> Requirement:
    """The TMR floor of a package from what its files use. ``needs``: the TMR nodes its launch
    files must be seen to start -- a template refactor that hides one is a TSA defect."""
    missing = sorted(set(needs) - use.executables)
    if missing and not use.unreadable:
        raise RuntimeError(f'the generated launch files start {missing} in a way the TMR floor '
                           f'cannot read (not a literal Node call): read it in '
                           f'generator/runtime_requirements.py (vNext M2.4)')
    if use.unreadable:
        newest = max((row[0] for name, row in TMR_FLOORS.items()
                      if use.canonical or name not in EEF_CONTRACT_CAPABILITIES),
                     key=_version_key)
        return Requirement(newest, _comment(f'{_TMR} >= {newest}: {", ".join(use.unreadable)} '
                                            f'does not parse, so every capability TSA dates is '
                                            f'assumed'))
    floor = max([TMR_FIRST_VERSION] + [TMR_FLOORS[c][0] for c in use.capabilities],
                key=_version_key)
    if floor == TMR_FIRST_VERSION:
        return Requirement(floor, f'{_TMR} >= {floor}, the first version its package.xml states: '
                                  f'nothing newer is used here')
    reasons: List[str] = []
    for name in use.capabilities:
        version, _commit, reason = TMR_FLOORS[name]
        if version == floor and reason not in reasons:
            reasons.append(reason)
    return Requirement(floor, f'{_TMR} >= {floor} for: {"; ".join(reasons)}')


def _comment(text: str) -> str:
    return text.replace('--', '- -')                 # an XML comment holds no '--'


def capabilities(trees: Texts = (), bt_params: Texts = (), launches: Texts = ()) -> List[str]:
    """The ``TMR_FLOORS`` capabilities the given generated files use, in table order."""
    return list(use_of(trees, bt_params, launches).capabilities)


def tmr_requirement(trees: Texts = (), bt_params: Texts = (), launches: Texts = (),
                    needs: Iterable[str] = ()) -> Requirement:
    """The TMR floor of one package from its generated trees, bt_params and launch files."""
    return requirement(use_of(trees, bt_params, launches), needs)


# --- trainit_hardware_layer (ADR-0022 D6 [D]) ---------------------------------------------------

def _bridge_rows(name: str, tree: ast.AST) -> List[dict]:
    """The literal ``BRIDGES = [...]`` rows of a generated bring-up: data, not calls -- every row
    is started by one ``Node(package=b["package"], ...)`` loop. A BRIDGES that is not a literal
    is a TSA defect (fail closed)."""
    rows: List[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(getattr(t, 'id', None) == 'BRIDGES'
                                                for t in node.targets):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, SyntaxError, TypeError):
                raise RuntimeError(f'{name}: BRIDGES is not a literal, so the bundle\'s '
                                   f'{HARDWARE_LAYER} requirement cannot be read (ADR-0022)')
            if not isinstance(value, list):
                raise RuntimeError(f'{name}: BRIDGES is not a list of rows (ADR-0022)')
            rows += [r for r in value if isinstance(r, dict)]
    return rows


def hardware_layer_requirement(launches: Texts) -> Optional[Requirement]:
    """``trainit_hardware_layer``'s floor for one package, or None: iff a generated launch file
    starts one of its executables that has a dated release (the end-effector adapters,
    ``catalog.EEF_ADAPTER_ROWS``), the highest such release, with the comment naming what set it.
    Fail closed: a launch file that does not parse, or a BRIDGES that is not a literal, raises --
    a bundle must never lose this line silently. Called for canonical projects only."""
    dated = {row['executable']: row['since'] for row in EEF_ADAPTER_ROWS.values()
             if row['package'] == HARDWARE_LAYER and row['executable'] and row['since']}
    started: Dict[str, str] = {}
    for name, text in _named(launches, 'launch'):
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError) as exc:
            raise RuntimeError(f'{_where(name, exc)} does not parse, so its {HARDWARE_LAYER} '
                               'requirement cannot be read (ADR-0022): the bundle would run '
                               'without its end-effector adapters declared') from None
        executables = [str(r.get('executable')) for r in _bridge_rows(name, tree)
                       if r.get('package') == HARDWARE_LAYER]
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _called(node) == 'Node':
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                if _literal(kw.get('package')) == HARDWARE_LAYER:
                    executables.append(str(_literal(kw.get('executable'))))
        for exe in executables:
            if exe in dated:
                started[exe] = dated[exe]
    if not started:
        return None
    floor = max(started.values(), key=_version_key)
    names = ', '.join(exe for exe in sorted(started) if started[exe] == floor)
    return Requirement(floor, _comment(f'{HARDWARE_LAYER} >= {floor} for: the end-effector '
                                       f'adapter(s) {names} (ADR-0022)'))


def uses_policy_runtime(trees: Texts = (), launches: Texts = ()) -> bool:
    """A package needs the learned-policy runtime iff a tree runs RunPolicy or a launch file
    starts its node (M2.4-4)."""
    return use_of(trees=trees, launches=launches).policy_runtime
