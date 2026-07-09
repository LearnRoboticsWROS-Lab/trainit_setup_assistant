"""Build an SRDF string from the canonical model.

The SRDF author is here (single source) so we control formatting. Groups, group
states, and the end effector come straight from the model; ``disable_collisions`` is
computed separately (M2, ``collision_matrix``) and passed in. This mirrors what the
MoveIt Setup Assistant writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

from ..model import CanonicalProject
from ..model.enums import GripperKind

_HEADER = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    '<!--This does not replace URDF, and is not an extension of URDF.\n'
    '    This is a format for representing semantic information about the robot structure.\n'
    '    A URDF file must exist for this robot as well, where the joints and the links '
    'that are referenced are defined\n'
    '-->\n'
)


@dataclass(frozen=True)
class DisablePair:
    """A self-collision pair to disable (an SRDF ``disable_collisions`` row)."""

    link1: str
    link2: str
    reason: str = 'Never'

    def key(self) -> frozenset:
        return frozenset((self.link1, self.link2))


def _fmt_value(value: float) -> str:
    """Format a joint value SRDF-style: integers bare, floats trimmed."""
    f = float(value)
    if f == int(f):
        return str(int(f))
    return repr(f)


def _indent(text: str, spaces: int = 4) -> str:
    pad = ' ' * spaces
    return '\n'.join(pad + line if line else line for line in text.split('\n'))


def build_srdf(
    project: CanonicalProject,
    disable_collisions: Optional[Sequence[DisablePair]] = None,
) -> str:
    """Return the SRDF XML for the project's robot."""
    robot = project.robot
    group = robot.planning_group
    grip = robot.gripper
    lines: List[str] = []

    lines.append(f'<robot name="{robot.robot_name}">')

    # --- arm group (chain) ---
    lines.append(f'    <group name="{group.name}">')
    lines.append(f'        <chain base_link="{group.base_link}" tip_link="{group.tip_link}"/>')
    lines.append('    </group>')

    # --- end-effector group (a single command joint), if any ---
    if grip.kind is not GripperKind.NONE and grip.eef_group_name and grip.command_joint:
        lines.append(f'    <group name="{grip.eef_group_name}">')
        lines.append(f'        <joint name="{grip.command_joint}"/>')
        lines.append('    </group>')

    # --- group states (named joint configurations) ---
    for state in robot.named_states:
        lines.append(f'    <group_state name="{state.name}" group="{state.group}">')
        for joint, value in state.joint_values.items():
            lines.append(f'        <joint name="{joint}" value="{_fmt_value(value)}"/>')
        lines.append('    </group_state>')

    # --- end effector ---
    if grip.kind is not GripperKind.NONE and grip.eef_name and grip.eef_group_name:
        parent_link = grip.eef_parent_link or group.tip_link
        lines.append(
            f'    <end_effector name="{grip.eef_name}" parent_link="{parent_link}" '
            f'group="{grip.eef_group_name}" parent_group="{group.name}"/>'
        )

    # --- disable collisions (computed by collisions_updater; M2) ---
    if disable_collisions is None:
        lines.append(
            '    <!-- disable_collisions populated by collisions_updater (M2) -->'
        )
    else:
        for pair in disable_collisions:
            lines.append(
                f'    <disable_collisions link1="{pair.link1}" '
                f'link2="{pair.link2}" reason="{pair.reason}"/>'
            )

    lines.append('</robot>')
    return _HEADER + '\n'.join(lines) + '\n'
