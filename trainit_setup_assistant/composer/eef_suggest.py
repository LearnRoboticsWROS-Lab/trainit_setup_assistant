"""Suggest an end-effector contract from what TSA already read (vNext I0, I0_DESIGN.md §34 E2).

Step 1 read the MoveIt config (its SRDF end effector and group, the gripper controller and joint), Step 2 the
world (the stage's gripper signal). From that, a CONTRACT (ADR-0022) is proposed with the source of every value
and the decisions it leaves to the user -- never adopted silently: the user sees it and adopts it as one
undoable edit (``eef_ops.adopt_contract``).

What it recognises, by what the end effector EXPOSES (the interface follows the actuator's semantics, ADR-0022):
- a **suction / vacuum** end effector (a binary command: ``set_bool``): the stage's Bool command topic, or the
  device's names (group, joint, SRDF states ``on`` / ``off``, a legacy suction bridge);
- a **single controlled jaw** (``gripper_command``): a GripperCommand controller on one joint (a Robotiq
  2F-85-like gripper), its SRDF open / closed states as targets;
- a **multi-actuator tool** (the long form): several movable joints in the end-effector group -- one actuator
  per joint proposed, the sites and each actuator's interface left to the user.

Pure: no Qt, no ROS. The SRDF is read from the base MoveIt config when there is one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from ..model.enums import Backend
from ..model.project import CanonicalProject

SUCTION_WORDS = ('suction', 'vacuum', 'cup', 'sucker')


@dataclass
class Suggestion:
    kind: str                                    # suction | parallel | composite
    contract: dict                               # the proposed contract, as it would be written
    sources: List[str] = field(default_factory=list)       # where each value comes from
    decide: List[str] = field(default_factory=list)        # what the user must still decide
    removes: List[str] = field(default_factory=list)       # the legacy bridges adoption removes

    def summary(self) -> str:
        lines = [f'Proposed: a {self.kind} end effector.']
        lines += ['  from: ' + s for s in self.sources]
        lines += ['  to decide: ' + d for d in self.decide]
        lines += ['  adopting removes the legacy bridge ' + r for r in self.removes]
        return '\n'.join(lines)


def _srdf(project: CanonicalProject):
    base = project.robot.base_moveit_config_path
    if not base:
        return None
    srdfs = sorted((Path(base) / 'config').glob('*.srdf'))
    if not srdfs:
        return None
    try:
        from ..robotmodel.srdf_reader import read_srdf
        return read_srdf(srdfs[0].read_text())
    except Exception:  # noqa: BLE001 - unreadable: suggest from the project alone
        return None


def _row_for(mode: str, interfaces: List[str], kind: str):
    """The catalog's adapter row that serves every interface of this end effector on ``mode`` (``none`` never
    chosen here): what the adapter rows declare (``modes``, ``serves``, ``kinds``), no name in this code."""
    from ..catalog import EEF_ADAPTER_ROWS
    for row, info in EEF_ADAPTER_ROWS.items():
        if row == 'none':
            continue
        if mode in info['modes'] and all(i in info['serves'] for i in interfaces) and kind in info['kinds']:
            return row, info
    return None, None


def _bindings(project: CanonicalProject, kind: str, interfaces: List[str], facts: Dict[str, str],
              sources, decide) -> Dict[str, object]:
    """One binding per deployed backend (ADR-0022 R15: every deployed backend needs one; 'none' is written when
    no row serves this end effector there). A row's required keys are filled from what TSA read (``facts``:
    the stage command topic from the world)."""
    out: Dict[str, object] = {}
    for mode in project.deployment.modes:
        m = Backend(mode).value
        row, info = _row_for(m, interfaces, kind)
        if row is None:
            out[m] = 'none'
            decide.append(f"{m}: no TrainIt adapter serves this end effector there yet ('none': its first "
                          'end-effector command fails there)')
            continue
        missing = [key for key in info['requires'] if not facts.get(key)]
        if missing:
            out[m] = 'none'
            decide.append(f"{m}: '{row}' needs {', '.join(missing)}, which TSA did not find: bind it with "
                          "it, or keep 'none'")
            continue
        keys = {key: facts[key] for key in info['binding_keys'] if facts.get(key)}
        out[m] = {'adapter': row, **keys} if keys else row
        sources.append(f"{m}: {info['name']}" + (''.join(f", {k} '{v}'" for k, v in keys.items())))
    return out


def _mimic_followers(project: CanonicalProject) -> set:
    """The robot description's mimic joints (followers are not actuators, ADR-0022 D0); empty when the
    description cannot be compiled."""
    try:
        from ..generator.eef_contract import bringup_description, description_view
        view, _label, why = bringup_description(project) if project.robot.base_moveit_config_path else (None, '', '')
        if view is None:
            view, _why = description_view(project)
        return {j.name for j in view.joints.values() if j.mimic} if view is not None else set()
    except Exception:  # noqa: BLE001 - no description: no follower is known
        return set()


def suggest_contract(project: CanonicalProject) -> Suggestion:
    """The proposed contract (module docstring)."""
    from .eef_ops import LEGACY_ROLE, bridge_entry
    g = project.robot.gripper
    srdf = _srdf(project)
    group = srdf.group(g.eef_group_name) if (srdf is not None and g.eef_group_name) else None
    states = {st.name: st.joint_values for st in (srdf.states_for(g.eef_group_name) if group else [])}
    tool_joints = list(group.joints) if group is not None else ([g.command_joint] if g.command_joint else [])
    topic = project.scene.gripper_cmd_topic or None
    legacy = [(r, bridge_entry(r)) for r in (project.deployment.bridges or [])]
    legacy = [(r, e) for r, e in legacy if e is not None and e['role'] == LEGACY_ROLE]
    removes = [f'{r.package}/{r.executable}' for r, _e in legacy]
    legacy_suction = any(e.get('eef_kind') == 'suction' for _r, e in legacy)
    names = ' '.join(filter(None, [g.eef_group_name, g.eef_name, g.command_joint, *states])).lower()
    sources: List[str] = []
    decide: List[str] = []
    facts = {'stage_cmd_topic': topic} if topic else {}
    verification = {'require_grasp_confirmation': False, 'require_release_confirmation': False}
    note_verify = ('verification: off (confirming a grasp needs a grasp sensor on every deployed backend: the '
                   'mock simulates one, Isaac needs a stage status topic) -- turn it on where you have one')
    actuated = [j for j in tool_joints if j not in _mimic_followers(project)] if len(tool_joints) >= 2 else tool_joints

    if topic or legacy_suction or any(w in names for w in SUCTION_WORDS) or {'on', 'off'} <= set(states):
        contract = {'actuator': {'command_interface': 'set_bool'}, 'verification': verification}
        why = (f"the stage's command '{topic}'" if topic else
               'the legacy suction bridge (catalog)' if legacy_suction else "the end effector's names / states")
        sources.append(f'a binary suction command (set_bool): {why}')
        kind, interfaces = 'suction', ['set_bool']
    elif g.controller_type == 'GripperCommand' and g.command_joint:
        # a GripperCommand controller decides first: a MoveIt hand group often lists a mimic finger too
        contract = {'actuator': {'command_interface': 'gripper_command', 'result_policy': 'succeeded_or_stalled'},
                    'verification': verification}
        sources.append('a controlled jaw (gripper_command): the GripperCommand controller'
                       + (f" '{g.controller_name}'" if g.controller_name else '')
                       + f" on joint '{g.command_joint}' (MoveIt config)")
        sources.append('result policy: succeeded_or_stalled (a jaw that closes on a part stalls before its target)')
        if not g.controller_name:
            decide.append("the controller action: the gripper controller's name was not read (Step 1)")
        from ..model.end_effector import _derived_jaw_targets
        # python review S4: the derivation itself decides (a state named without a value for the jaw's joint
        # derives nothing, ADR-0022 D6), not the names alone
        targets, why = _derived_jaw_targets(g, g.command_joint, project.robot.named_states, states)
        if targets is None:
            decide.append(f"the open and closed targets of '{g.command_joint}': {why} -- type them "
                          '(End effector: Named targets)')
        else:
            sources.append(f"targets: open {targets['open']:g} / closed {targets['closed']:g} "
                           f"(the states '{g.open_state}' / '{g.closed_state}' or the gripper's angles)")
        kind, interfaces = 'parallel', ['gripper_command']
    elif len(actuated) >= 2:
        actuators = {}
        for j in actuated:
            targets = {st: [v[j]] for st, v in states.items() if j in v}
            spec = {'command_interface': 'follow_joint_trajectory', 'joints': [j],
                    'action': f'{j}_controller/follow_joint_trajectory'}
            if targets:
                spec['targets'] = targets
            actuators[j] = spec
        contract = {'id': g.eef_name or 'tool', 'actuators': actuators, 'sites': {}, 'verification': verification}
        sources.append(f"actuators: the {len(actuated)} actuated joints of the end-effector group "
                       f"'{g.eef_group_name}' (MoveIt config SRDF; mimic followers left out), one each")
        decide += ['each actuator\'s interface: an axis that moves a grasp site is follow_joint_trajectory (R24); '
                   'a jaw may be gripper_command or set_bool (binary)',
                   "each actuator's controller action and named targets",
                   'the grasp sites: a TCP link and the grip actuator of each']
        kind, interfaces = 'composite', ['follow_joint_trajectory']
    else:
        contract = {'actuator': {'command_interface': 'set_bool'}, 'verification': verification}
        decide.append('the actuator: nothing identifies it -- a binary tool (set_bool) is proposed; choose a '
                      'controlled jaw (gripper_command) or a multi-actuator tool instead if it is one')
        kind, interfaces = 'suction', ['set_bool']
    contract['adapters'] = _bindings(project, kind, interfaces, facts, sources, decide)
    sources.append(note_verify)
    if removes:
        sources.append('the legacy end-effector bridges are replaced by the adapters (ADR-0022 R18)')
    return Suggestion(kind, contract, sources, decide, removes)


__all__ = ['SUCTION_WORDS', 'Suggestion', 'suggest_contract']
