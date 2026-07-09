"""Light SRDF reader — extract groups / group_states / end_effectors from an existing
SRDF (the hand-made base moveit_config), so Step 2 can auto-configure the project
instead of asking the user to re-type what the base already declares.

Uses ElementTree (no srdfdom / ROS dependency). Pairs with
:func:`collision_matrix.parse_disable_collisions` for the self-collision matrix.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SrdfGroup:
    name: str
    joints: List[str] = field(default_factory=list)
    chain: Optional[Tuple[str, str]] = None       # (base_link, tip_link)


@dataclass
class SrdfGroupState:
    name: str
    group: str
    joint_values: Dict[str, float] = field(default_factory=dict)


@dataclass
class SrdfEndEffector:
    name: str
    parent_link: str
    group: str
    parent_group: Optional[str] = None


@dataclass
class SrdfInfo:
    robot_name: str
    groups: List[SrdfGroup] = field(default_factory=list)
    group_states: List[SrdfGroupState] = field(default_factory=list)
    end_effectors: List[SrdfEndEffector] = field(default_factory=list)

    def group(self, name: str) -> Optional[SrdfGroup]:
        return next((g for g in self.groups if g.name == name), None)

    def arm_group(self) -> Optional[SrdfGroup]:
        """The manipulation group: the one referenced by an end_effector's
        parent_group, else the group with a chain, else the one with the most joints."""
        eef_parents = {e.parent_group for e in self.end_effectors if e.parent_group}
        for g in self.groups:
            if g.name in eef_parents:
                return g
        chained = [g for g in self.groups if g.chain]
        if chained:
            return max(chained, key=lambda g: len(g.joints))
        return max(self.groups, key=lambda g: len(g.joints), default=None)

    def eef_group_name(self) -> Optional[str]:
        return self.end_effectors[0].group if self.end_effectors else None

    def states_for(self, group_name: str) -> List[SrdfGroupState]:
        return [s for s in self.group_states if s.group == group_name]


def read_srdf(text: str) -> SrdfInfo:
    root = ET.fromstring(text)
    info = SrdfInfo(robot_name=root.get('name', 'robot'))
    for g in root.findall('group'):
        grp = SrdfGroup(name=g.get('name', ''))
        chain = g.find('chain')
        if chain is not None:
            grp.chain = (chain.get('base_link', ''), chain.get('tip_link', ''))
        grp.joints = [j.get('name', '') for j in g.findall('joint')]
        info.groups.append(grp)
    for gs in root.findall('group_state'):
        st = SrdfGroupState(name=gs.get('name', ''), group=gs.get('group', ''))
        for j in gs.findall('joint'):
            try:
                st.joint_values[j.get('name', '')] = float(j.get('value', '0'))
            except ValueError:
                pass
        info.group_states.append(st)
    for ee in root.findall('end_effector'):
        info.end_effectors.append(SrdfEndEffector(
            name=ee.get('name', ''), parent_link=ee.get('parent_link', ''),
            group=ee.get('group', ''), parent_group=ee.get('parent_group')))
    return info
