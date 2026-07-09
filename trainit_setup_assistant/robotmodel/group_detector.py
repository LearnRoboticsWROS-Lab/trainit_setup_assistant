"""Auto-detect the planning group, frames and end-effector from a URDF.

Produces a best-effort DRAFT that the user confirms/edits in the GUI (step S1). Pure
heuristics, robot-agnostic — tuned to give the obvious answer for typical 6-axis arms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from ..model.enums import GripperKind
from .kinematic_chain import (
    chain_length_from,
    chain_movable_joints,
    leaf_links,
    link_chain,
    root_link,
)

# link-name hints for the tool tip, best first
_TIP_HINTS = ('tcp', 'tool0', 'tool_tip', 'flange', 'eef', 'ee_link', 'tool', 'tip')
# gripper-kind hints (matched against gripper joint/link names)
_SUCTION_HINTS = ('suction', 'vacuum', 'suctioncup', 'sucker')
_PARALLEL_HINTS = ('finger', 'jaw', 'claw', 'gripper', 'robotiq', 'hand', 'pinch')


@dataclass
class RobotDetection:
    base_link: str
    tip_link: str
    arm_joints: List[str]
    gripper_joints: List[str] = field(default_factory=list)
    gripper_kind: GripperKind = GripperKind.NONE
    eef_parent_link: Optional[str] = None
    eef_link: Optional[str] = None
    notes: List[str] = field(default_factory=list)


def _matches(name: str, hints) -> bool:
    low = name.lower()
    return any(h in low for h in hints)


def detect_base_link(robot) -> str:
    base = root_link(robot)
    if base is None:
        raise ValueError('URDF has no root link (cyclic or empty)')
    return base


def detect_tip_link(robot, base_link: str) -> str:
    """Pick the tool tip: a name-hinted link reachable from base, else longest chain."""
    reachable = [l.name for l in robot.links if chain_length_from(robot, base_link, l.name) >= 0]
    # 1) by name hint (in hint priority order), reachable and with >=1 movable joint
    for hint in _TIP_HINTS:
        for name in reachable:
            if hint in name.lower() and chain_length_from(robot, base_link, name) >= 1:
                return name
    # 2) the leaf with the longest movable chain from base
    leaves = [l for l in leaf_links(robot) if l in reachable]
    candidates = leaves or reachable
    best = max(candidates, key=lambda n: chain_length_from(robot, base_link, n))
    return best


def detect_gripper(robot, base_link: str, tip_link: str):
    """Find gripper joints (movable joints NOT on the arm chain) + classify the kind."""
    arm = set(chain_movable_joints(robot, base_link, tip_link))
    from .kinematic_chain import all_movable_joints
    gripper_joints = [j for j in all_movable_joints(robot) if j not in arm]

    kind = GripperKind.NONE
    if gripper_joints:
        names = ' '.join(gripper_joints)
        # also consider child link names of the gripper joints
        child_links = ' '.join(j.child for j in robot.joints if j.name in gripper_joints)
        blob = (names + ' ' + child_links)
        if _matches(blob, _SUCTION_HINTS):
            kind = GripperKind.SUCTION
        elif _matches(blob, _PARALLEL_HINTS):
            kind = GripperKind.PARALLEL
        else:
            kind = GripperKind.PARALLEL  # default for an actuated EEF
    return gripper_joints, kind


def detect(robot) -> RobotDetection:
    """Full detection: base, tip, arm joints, gripper joints + kind, EEF parent."""
    base = detect_base_link(robot)
    tip = detect_tip_link(robot, base)
    arm = chain_movable_joints(robot, base, tip)
    gripper_joints, kind = detect_gripper(robot, base, tip)

    det = RobotDetection(
        base_link=base, tip_link=tip, arm_joints=arm,
        gripper_joints=gripper_joints, gripper_kind=kind,
        eef_parent_link=tip,
    )
    if gripper_joints:
        # EEF link = the child link of the first gripper joint
        for j in robot.joints:
            if j.name == gripper_joints[0]:
                det.eef_link = j.child
                break
    if not arm:
        det.notes.append('no movable joints found on the base->tip chain')
    if len(gripper_joints) > 1:
        det.notes.append(f'{len(gripper_joints)} gripper joints detected; confirm the EEF group')
    return det
