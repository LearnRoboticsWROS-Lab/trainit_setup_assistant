"""The end-effector nodes of a CANONICAL tree (ADR-0022 D2, D4, D5): logical ids only, never a
topic, a service or an action name (owner section 24.12) -- the runner resolves every endpoint
from the ``eef`` parameter block.

A LEGACY project never reaches this module: its trees keep the portless ``<CloseGripper/>`` /
``<OpenGripper/>`` and today's ``SetReleasePolicy``, byte for byte (D9).

What the verification policy emits (D4), per tool action:

* grasp    ``CloseGripper``, then ``VerifyGrasp(held)`` when grasp confirmation is required;
* release  ``VerifyGrasp(held)`` BEFORE it when grasp confirmation is required and the device's
           state is continuous on every deployed backend, ``OpenGripper``, then
           ``VerifyGrasp(released)`` when release confirmation is required;
* actuate  ``CommandActuator`` to a NAMED target (a grip's engage/release target is refused, R11);
* no ``SetReleasePolicy``: no stage or world subscribes to it (D8).

``gate`` (bin picking, D5): every end-effector node writes its ``fault`` (inside the candidate
retry) or its ``reason`` (anywhere else inside the absorbing Fallback) into the gate key.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ..model.end_effector import Device, EndEffectorCapability, derive_device
from ..model.enums import Backend, ToolActionKind

Gate = Optional[Tuple[str, str]]          # ('fault' | 'reason', blackboard key)


def canonical_device(project) -> Optional[Device]:
    """The derived device of a canonical project, None for a legacy one. The trees need no file:
    ids, targets and the verification policy are all in the project."""
    g = project.robot.gripper
    if g.contract is None:
        return None
    return derive_device(g, project.robot.named_states,
                         stage_topic=project.scene.gripper_cmd_topic or None)


def _gate(gate: Gate) -> str:
    return f' {gate[0]}="{{{gate[1]}}}"' if gate else ''


def eef_attrs(dev: Device, site: Optional[str] = None) -> str:
    return f'eef="{dev.id}"' + (f' site="{site}"' if site is not None else '')


def verify_grasp(dev: Device, site: Optional[str], expect: str, gate: Gate = None) -> str:
    return f'<VerifyGrasp {eef_attrs(dev, site)} expect="{expect}"{_gate(gate)}/>'


def continuous_everywhere(project, dev: Device) -> bool:
    """The device's state is a continuous stream on every deployed backend (``continuous_state``,
    every TrainIt adapter): the condition of the pre-release check (D4)."""
    return dev.provides_everywhere(EndEffectorCapability.CONTINUOUS_STATE.value,
                                   [Backend(m) for m in project.deployment.modes])


def close_nodes(dev: Device, site: Optional[str]) -> List[str]:
    """A grip's close (ADR-0022 D4): ``CloseGripper``, then ``VerifyGrasp held`` when the contract requires grasp
    confirmation -- the grasp idiom's actuation and a task program's ``tool close``."""
    lines = [f'<CloseGripper {eef_attrs(dev, site)}/>']
    if dev.require_grasp_confirmation:
        lines.append(verify_grasp(dev, site, 'held'))
    return lines


def open_nodes(project, dev: Device, site: Optional[str], held_guard: bool) -> List[str]:
    """A grip's open (ADR-0022 D4): [``VerifyGrasp held`` first -- ``held_guard``: the RELEASE idiom, reached with a
    part in the gripper, when grasps are confirmed and the state is continuous everywhere] + ``OpenGripper`` +
    [``VerifyGrasp released`` when release confirmation is required]. A plain ``tool open`` has no held guard: it
    is valid with nothing held, e.g. at a cycle start (ADR-0024 amendment I0)."""
    lines = []
    if held_guard and dev.require_grasp_confirmation and continuous_everywhere(project, dev):
        lines.append(verify_grasp(dev, site, 'held'))
    lines.append(f'<OpenGripper {eef_attrs(dev, site)}/>')
    if dev.require_release_confirmation:
        lines.append(verify_grasp(dev, site, 'released'))
    return lines


def action_nodes(action, project, dev: Device, gate: Gate = None,
                 verify_after_grasp: bool = True) -> List[str]:
    """The canonical nodes of one tool action, one per line, unindented."""
    kind = action.kind
    if kind is ToolActionKind.GRASP:
        lines = [f'<CloseGripper {eef_attrs(dev, action.site)}{_gate(gate)}/>']
        if dev.require_grasp_confirmation and verify_after_grasp:
            lines.append(verify_grasp(dev, action.site, 'held', gate))
        return lines
    if kind is ToolActionKind.RELEASE:
        lines = []
        if dev.require_grasp_confirmation and continuous_everywhere(project, dev):
            lines.append(verify_grasp(dev, action.site, 'held', gate))
        lines.append(f'<OpenGripper {eef_attrs(dev, action.site)}{_gate(gate)}/>')
        if dev.require_release_confirmation:
            lines.append(verify_grasp(dev, action.site, 'released', gate))
        return lines
    if kind is ToolActionKind.ACTUATE:
        return [f'<CommandActuator {eef_attrs(dev)} actuator="{action.actuator}" '
                f'target="{action.target}"{_gate(gate)}/>']
    if kind is ToolActionKind.RESET_SCENE:
        return ['<ResetScene/>']
    if kind in (ToolActionKind.ATTACH, ToolActionKind.DETACH):
        return None                         # the legacy rendering: no end-effector node
    raise ValueError(f'tool action kind {getattr(kind, "value", kind)!r} has no canonical node '
                     '(ADR-0022 R11): it must never become a silent comment')


def policy_verify(dev: Device) -> List[str]:
    """The ATTACHED half of a Pure/Residual post-policy check (ADR-0005 as ADR-0022 amends it):
    ``VerifyGrasp`` when grasp confirmation is required. The POSITION half stays CheckRobotState,
    whatever this says."""
    if dev.require_grasp_confirmation:
        return [verify_grasp(dev, None, 'held')]
    return []
