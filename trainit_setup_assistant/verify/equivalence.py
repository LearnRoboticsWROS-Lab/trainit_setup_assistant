"""Semantic equivalence checks: a generated bundle vs a golden bundle.

Each file is compared with the right notion of equality (see the plan):
  byte-identical : copied geometry (meshes, description xacros)
  yaml-semantic  : config yamls + bt_params.yaml (order/format-independent, float tol)
  srdf-semantic  : groups + group_states + end_effector + disable_collisions SET
  tree-semantic  : the BT node sequence + ports (whitespace/comment-independent)
  valid-python   : launch files (full text intentionally not compared)

Compare with matching package names (generate a project whose bundle names equal the
golden's), so package dirs line up 1:1.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml


@dataclass
class Report:
    checks: List[tuple] = field(default_factory=list)  # (name, kind, ok, detail)

    def add(self, name: str, kind: str, ok: bool, detail: str = '') -> None:
        self.checks.append((name, kind, ok, detail))

    @property
    def ok(self) -> bool:
        return all(c[2] for c in self.checks)

    @property
    def failures(self) -> List[tuple]:
        return [c for c in self.checks if not c[2]]

    def summary(self) -> str:
        lines = []
        for name, kind, ok, detail in self.checks:
            mark = 'OK ' if ok else 'XX '
            lines.append(f'  {mark} [{kind}] {name}' + (f'  -> {detail}' if detail and not ok else ''))
        passed = sum(1 for c in self.checks if c[2])
        lines.append(f'{passed}/{len(self.checks)} checks passed')
        return '\n'.join(lines)


# --- scalar/yaml comparison ---
def _almost_equal(a, b, tol: float) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) <= tol
    return a == b


def yaml_diffs(a, b, tol: float = 1e-6, path: str = '') -> List[str]:
    """Deep diff (a=generated, b=golden). Empty list == equal."""
    diffs: List[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if key not in a:
                diffs.append(f'{path}/{key}: missing in generated')
            elif key not in b:
                diffs.append(f'{path}/{key}: extra in generated')
            else:
                diffs += yaml_diffs(a[key], b[key], tol, f'{path}/{key}')
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            diffs.append(f'{path}: list length {len(a)} != {len(b)}')
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                diffs += yaml_diffs(x, y, tol, f'{path}[{i}]')
    elif not _almost_equal(a, b, tol):
        diffs.append(f'{path}: {a!r} != {b!r}')
    return diffs


def _load_yaml(path: Path):
    return yaml.safe_load(Path(path).read_text())


# --- SRDF comparison ---
def _srdf_facts(text: str) -> dict:
    root = ET.fromstring(text)
    groups = {}
    for g in root.findall('group'):
        chain = g.find('chain')
        joints = tuple(sorted(j.get('name') for j in g.findall('joint')))
        groups[g.get('name')] = (
            (chain.get('base_link'), chain.get('tip_link')) if chain is not None else None,
            joints,
        )
    states = {}
    for gs in root.findall('group_state'):
        states[(gs.get('group'), gs.get('name'))] = {
            j.get('name'): float(j.get('value')) for j in gs.findall('joint')
        }
    eefs = {}
    for ee in root.findall('end_effector'):
        eefs[ee.get('name')] = (ee.get('parent_link'), ee.get('group'), ee.get('parent_group'))
    disable = {
        (frozenset((d.get('link1'), d.get('link2'))), d.get('reason'))
        for d in root.findall('disable_collisions')
    }
    return {'groups': groups, 'states': states, 'eefs': eefs, 'disable': disable}


def srdf_diffs(gen_text: str, gold_text: str) -> List[str]:
    a, b = _srdf_facts(gen_text), _srdf_facts(gold_text)
    diffs = []
    for key in ('groups', 'states', 'eefs'):
        if a[key] != b[key]:
            diffs.append(f'{key}: {a[key]} != {b[key]}')
    if a['disable'] != b['disable']:
        only_gen = a['disable'] - b['disable']
        only_gold = b['disable'] - a['disable']
        diffs.append(f'disable_collisions differ: +{only_gen} -{only_gold}')
    return diffs


# --- BT tree comparison ---
def _tree_nodes(text: str) -> List[tuple]:
    """Ordered list of (tag, frozenset(attr items)) for all elements under <root>."""
    root = ET.fromstring(text)
    nodes = []
    for el in root.iter():
        if el.tag in ('root', 'BehaviorTree'):
            continue
        nodes.append((el.tag, frozenset(el.attrib.items())))
    return nodes


def tree_diffs(gen_text: str, gold_text: str) -> List[str]:
    a, b = _tree_nodes(gen_text), _tree_nodes(gold_text)
    if a == b:
        return []
    diffs = []
    if len(a) != len(b):
        diffs.append(f'node count {len(a)} != {len(b)}')
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            diffs.append(f'node[{i}]: {x} != {y}')
    return diffs


# --- bundle comparison ---
DESCRIPTION = 'description'
MOVEIT = 'moveit_config'
APP = 'app'


def detect_packages(bundle_root: Path) -> Dict[str, str]:
    """Infer the role -> package-dir-name map of a bundle by what each dir contains.

    description = has urdf/ ; moveit_config = has config/*.srdf ; app = has bt_trees/.
    Lets the verifier compare bundles with DIFFERENT package names (generated vs golden).
    """
    root = Path(bundle_root)
    found: Dict[str, str] = {}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        if (d / 'bt_trees').is_dir():
            found.setdefault(APP, d.name)
        elif (d / 'config').is_dir() and any((d / 'config').glob('*.srdf')):
            found.setdefault(MOVEIT, d.name)
        elif (d / 'urdf').is_dir():
            found.setdefault(DESCRIPTION, d.name)
    return found


def compare_bundle(
    gen_root: Path,
    gold_root: Path,
    gen_packages: Dict[str, str],
    gold_packages: Optional[Dict[str, str]] = None,
) -> Report:
    """Compare a generated bundle against a golden one, by ROLE.

    ``gen_packages`` / ``gold_packages`` map role -> package-dir-name on each side, so
    the two bundles may use different package names. If ``gold_packages`` is omitted it
    is auto-detected from ``gold_root``. Reported paths are role-relative (name-neutral).
    """
    gen_root, gold_root = Path(gen_root), Path(gold_root)
    if gold_packages is None:
        gold_packages = detect_packages(gold_root)
    rep = Report()

    def gen_pkg(role: str) -> Path:
        return gen_root / gen_packages[role]

    def gold_pkg(role: str) -> Path:
        return gold_root / gold_packages[role]

    # --- description: byte-identical geometry ---
    for sub in ('urdf', 'meshes'):
        gold_sub = gold_pkg(DESCRIPTION) / sub
        if not gold_sub.is_dir():
            continue
        for gold_file in sorted(gold_sub.rglob('*')):
            if gold_file.is_file():
                rel = gold_file.relative_to(gold_pkg(DESCRIPTION))
                gen_file = gen_pkg(DESCRIPTION) / rel
                ok = gen_file.is_file() and gen_file.read_bytes() == gold_file.read_bytes()
                rep.add(f'description/{rel}', 'byte', ok,
                        'missing' if not gen_file.is_file() else 'bytes differ')

    # --- moveit_config: yaml semantic + srdf semantic ---
    for gold_file in sorted((gold_pkg(MOVEIT) / 'config').glob('*')):
        rel = gold_file.relative_to(gold_pkg(MOVEIT))
        gen_file = gen_pkg(MOVEIT) / rel
        report_name = f'moveit_config/{rel}'
        name = gold_file.name
        if name.endswith('.srdf'):
            if gen_file.is_file():
                diffs = srdf_diffs(gen_file.read_text(), gold_file.read_text())
                rep.add(report_name, 'srdf', not diffs, '; '.join(diffs))
            else:
                rep.add(report_name, 'srdf', False, 'missing')
        elif name.endswith(('.yaml', '.rviz')):
            if gen_file.is_file():
                diffs = yaml_diffs(_load_yaml(gen_file), _load_yaml(gold_file))
                rep.add(report_name, 'yaml', not diffs, '; '.join(diffs[:5]))
            else:
                rep.add(report_name, 'yaml', False, 'missing')
        else:
            rep.add(report_name, 'present', gen_file.is_file(),
                    '' if gen_file.is_file() else 'missing')

    # --- app: bt_params (yaml semantic) + BT tree (node sequence) ---
    gold_bt = gold_pkg(APP) / 'config' / 'bt_params.yaml'
    gen_bt = gen_pkg(APP) / 'config' / 'bt_params.yaml'
    if gen_bt.is_file() and gold_bt.is_file():
        diffs = yaml_diffs(_load_yaml(gen_bt), _load_yaml(gold_bt))
        rep.add('app/config/bt_params.yaml', 'yaml', not diffs, '; '.join(diffs[:8]))
    else:
        rep.add('app/config/bt_params.yaml', 'yaml', False, 'missing')

    for gold_tree in sorted((gold_pkg(APP) / 'bt_trees').glob('*.xml')):
        rel = gold_tree.relative_to(gold_pkg(APP))
        gen_tree = gen_pkg(APP) / rel
        if gen_tree.is_file():
            diffs = tree_diffs(gen_tree.read_text(), gold_tree.read_text())
            rep.add(f'app/{rel}', 'tree', not diffs, '; '.join(diffs[:8]))
        else:
            rep.add(f'app/{rel}', 'tree', False, 'missing')

    return rep
