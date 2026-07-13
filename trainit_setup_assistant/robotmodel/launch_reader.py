"""Extract the cell bring-up wiring from the hand-made base moveit_config's launch file.

The base's ``bringup.launch.py`` is the adaptation-sprint source of truth for HOW the cell
is wired per mode: which vendor/sim BRIDGES run (gripper bridge, joint-state merger, the
real arm/gripper drivers) and where the arm's ``/joint_states`` is remapped. The generated
scene+planner config must launch the SAME bridges, or ``/joint_states`` misses the gripper
joint, the TF tree is incomplete and RViz flickers.

We parse it with ``ast`` (no execution): every ``Node(package=…, executable=…)`` that is
not a framework node (robot_state_publisher / move_group / ros2_control_node / spawner /
rviz2) is a cell bridge, tagged with the mode branch it sits in.
"""

from __future__ import annotations

import ast
from typing import List, Optional

FRAMEWORK_EXECUTABLES = {
    'robot_state_publisher', 'move_group', 'ros2_control_node', 'spawner', 'rviz2',
}
ALL_MODES = ['mock', 'isaac', 'real']


def _const(node):
    return node.value if isinstance(node, ast.Constant) else None


def _dict_literal(node) -> Optional[dict]:
    if not isinstance(node, ast.Dict):
        return None
    out = {}
    for k, v in zip(node.keys, node.values):
        kk = _const(k)
        vv = _const(v)
        if kk is None or vv is None:
            return None            # non-literal (e.g. `common`) -> skip this entry
        out[kk] = vv
    return out


def _remappings(node) -> List[list]:
    out: List[list] = []
    if isinstance(node, (ast.List, ast.Tuple)):
        for el in node.elts:
            if isinstance(el, (ast.Tuple, ast.List)) and len(el.elts) == 2:
                a, b = _const(el.elts[0]), _const(el.elts[1])
                if isinstance(a, str) and isinstance(b, str):
                    out.append([a, b])
    return out


def _branch_modes(test, modes: List[str]):
    """(body_modes, else_modes) for an `if mode in (...)` / `if mode == "x"` test."""
    if isinstance(test, ast.Compare) and len(test.ops) == 1:
        left = test.left
        if isinstance(left, ast.Name) and left.id == 'mode':
            comp = test.comparators[0]
            if isinstance(test.ops[0], ast.In) and isinstance(comp, (ast.Tuple, ast.List)):
                sel = [c for c in (_const(e) for e in comp.elts) if isinstance(c, str)]
            elif isinstance(test.ops[0], ast.Eq):
                c = _const(comp)
                sel = [c] if isinstance(c, str) else []
            else:
                sel = []
            if sel:
                body = [m for m in modes if m in sel]
                other = [m for m in modes if m not in sel]
                return body, other
    return list(modes), list(modes)


def read_bringup(path) -> dict:
    """-> {'arm_joint_states_remap_to': str|None, 'bridges': [LaunchNodeSpec-like dicts]}"""
    tree = ast.parse(open(path).read())
    bridges: List[dict] = []
    arm_remap: Optional[str] = None

    def handle_call(call: ast.Call, modes: List[str]):
        nonlocal arm_remap
        if not (isinstance(call.func, ast.Name) and call.func.id == 'Node'):
            return
        kw = {k.arg: k.value for k in call.keywords if k.arg}
        pkg, exe = _const(kw.get('package')), _const(kw.get('executable'))
        if not isinstance(pkg, str) or not isinstance(exe, str):
            return
        if exe == 'ros2_control_node':          # framework, but it carries the arm remap
            for src, dst in _remappings(kw.get('remappings')):
                if src.lstrip('/') == 'joint_states':
                    arm_remap = dst
            return
        if exe in FRAMEWORK_EXECUTABLES:
            return
        spec = {'package': pkg, 'executable': exe, 'modes': list(modes),
                'parameters': [], 'remappings': _remappings(kw.get('remappings'))}
        name = _const(kw.get('name'))
        if isinstance(name, str):
            spec['name'] = name
        params = kw.get('parameters')
        if isinstance(params, (ast.List, ast.Tuple)):
            for el in params.elts:
                d = _dict_literal(el)
                if d:
                    spec['parameters'].append(d)
        bridges.append(spec)

    def visit(stmts, modes: List[str]):
        for st in stmts:
            if isinstance(st, ast.If):
                body_modes, else_modes = _branch_modes(st.test, modes)
                visit(st.body, body_modes)
                visit(st.orelse, else_modes)
                continue
            for n in ast.walk(st):
                if isinstance(n, ast.Call):
                    handle_call(n, modes)

    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            visit(fn.body, ALL_MODES)

    # de-dup (a Node assigned to a var and appended twice must not double)
    seen, uniq = set(), []
    for b in bridges:
        key = (b['package'], b['executable'], b.get('name'))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(b)
    return {'arm_joint_states_remap_to': arm_remap, 'bridges': uniq}
