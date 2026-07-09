"""Parse a URDF and reason about its kinematic structure.

Thin wrappers over ``urdf_parser_py`` used by the group detector and the live session.
Robot-agnostic: nothing here assumes a particular robot.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

MOVABLE_JOINT_TYPES = {'revolute', 'prismatic', 'continuous'}


def parse_urdf(urdf_xml: str):
    """Parse a URDF string into a urdf_parser_py ``URDF`` object."""
    from urdf_parser_py.urdf import URDF  # lazy: heavy import
    return URDF.from_xml_string(urdf_xml)


def root_link(robot) -> Optional[str]:
    """The link with no parent joint (the kinematic root)."""
    children = {j.child for j in robot.joints}
    roots = [l.name for l in robot.links if l.name not in children]
    return roots[0] if roots else None


def leaf_links(robot) -> List[str]:
    """Links that are never a joint parent (chain tips)."""
    parents = {j.parent for j in robot.joints}
    return [l.name for l in robot.links if l.name not in parents]


def _child_to_joint(robot) -> dict:
    return {j.child: j for j in robot.joints}


def link_chain(robot, base_link: str, tip_link: str) -> Tuple[List[str], List[object]]:
    """Return (links, joints) along the path base_link -> tip_link (inclusive).

    Raises ValueError if no path exists.
    """
    c2j = _child_to_joint(robot)
    links = [tip_link]
    joints: List[object] = []
    cur = tip_link
    for _ in range(len(robot.joints) + 1):
        if cur == base_link:
            links.reverse()
            joints.reverse()
            return links, joints
        joint = c2j.get(cur)
        if joint is None:
            break
        joints.append(joint)
        links.append(joint.parent)
        cur = joint.parent
    raise ValueError(f'no kinematic path from {tip_link} up to {base_link}')


def movable_joint_names(joints) -> List[str]:
    return [j.name for j in joints if j.type in MOVABLE_JOINT_TYPES]


def chain_movable_joints(robot, base_link: str, tip_link: str) -> List[str]:
    """Movable joints on the base->tip chain, in base->tip order."""
    _, joints = link_chain(robot, base_link, tip_link)
    return movable_joint_names(joints)


def all_movable_joints(robot) -> List[str]:
    return [j.name for j in robot.joints if j.type in MOVABLE_JOINT_TYPES]


def chain_length_from(robot, base_link: str, tip_link: str) -> int:
    """Number of movable joints from base to tip, or -1 if unreachable."""
    try:
        return len(chain_movable_joints(robot, base_link, tip_link))
    except ValueError:
        return -1
