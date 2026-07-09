"""M5: robot auto-detection (kinematic chain + group detector).

The core test is hermetic (a URDF string, no ROS/xacro). An extra test exercises the
real FR3WML xacro when a ROS environment is available.
"""

import os

import pytest

from trainit_setup_assistant.model.enums import GripperKind
from trainit_setup_assistant.robotmodel import detect, parse_urdf

# A minimal 2-axis arm (base->l1->tool0) with a parallel-finger gripper off tool0.
URDF_2AXIS = """<?xml version="1.0"?>
<robot name="toy_arm">
  <link name="base_link"/>
  <link name="l1"/>
  <link name="tool0"/>
  <link name="finger_link"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="l1"/>
    <limit lower="-3" upper="3" velocity="2.0" effort="100"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="l1"/><child link="tool0"/>
    <limit lower="-3" upper="3" velocity="2.5" effort="100"/>
  </joint>
  <joint name="finger_joint" type="prismatic">
    <parent link="tool0"/><child link="finger_link"/>
    <limit lower="0" upper="0.05" velocity="0.1" effort="50"/>
  </joint>
</robot>"""


def test_detect_toy_arm():
    robot = parse_urdf(URDF_2AXIS)
    det = detect(robot)
    assert det.base_link == 'base_link'
    assert det.tip_link == 'tool0'              # name-hinted tip
    assert det.arm_joints == ['j1', 'j2']       # movable joints on the chain
    assert det.gripper_joints == ['finger_joint']
    assert det.gripper_kind is GripperKind.PARALLEL  # 'finger' hint
    assert det.eef_parent_link == 'tool0'


def test_detect_no_gripper():
    urdf = URDF_2AXIS.replace(
        '<joint name="finger_joint" type="prismatic">\n'
        '    <parent link="tool0"/><child link="finger_link"/>\n'
        '    <limit lower="0" upper="0.05" velocity="0.1" effort="50"/>\n'
        '  </joint>', '')
    det = detect(parse_urdf(urdf))
    assert det.gripper_kind is GripperKind.NONE
    assert det.gripper_joints == []


def _chain_urdf(name, joints, tail_links=()):
    """Build a simple serial-chain URDF. joints = [(jname, jtype, parent, child)]."""
    links = set()
    for _, _, p, c in joints:
        links.add(p)
        links.add(c)
    links.update(tail_links)
    link_xml = ''.join(f'<link name="{l}"/>' for l in links)
    joint_xml = ''.join(
        f'<joint name="{n}" type="{t}"><parent link="{p}"/><child link="{c}"/>'
        f'<limit lower="-3" upper="3" velocity="2" effort="100"/></joint>'
        for n, t, p, c in joints)
    return f'<?xml version="1.0"?><robot name="{name}">{link_xml}{joint_xml}</robot>'


def test_detect_tip_hint_priority_tcp_over_tool0():
    # both tool0 and tcp exist; 'tcp' must win (hint priority)
    urdf = _chain_urdf('arm', [
        ('j1', 'revolute', 'base_link', 'l1'),
        ('j2', 'revolute', 'l1', 'l2'),
        ('f_tool0', 'fixed', 'l2', 'tool0'),
        ('f_tcp', 'fixed', 'tool0', 'tcp'),
    ])
    det = detect(parse_urdf(urdf))
    assert det.tip_link == 'tcp'
    assert det.arm_joints == ['j1', 'j2']


def test_detect_continuous_joint_is_movable():
    urdf = _chain_urdf('arm', [
        ('j1', 'revolute', 'base_link', 'l1'),
        ('j2', 'continuous', 'l1', 'tool0'),
    ])
    det = detect(parse_urdf(urdf))
    assert det.arm_joints == ['j1', 'j2']      # continuous counts as movable
    assert det.tip_link == 'tool0'


def test_detect_scara_prismatic_in_arm():
    urdf = _chain_urdf('scara', [
        ('j1', 'revolute', 'base_link', 'l1'),
        ('j2', 'prismatic', 'l1', 'l2'),       # vertical axis, part of the arm
        ('j3', 'revolute', 'l2', 'tool0'),
    ])
    det = detect(parse_urdf(urdf))
    assert det.arm_joints == ['j1', 'j2', 'j3']
    assert det.gripper_joints == []


FR3WML_XACRO = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/fr3wml_app/'
                'fr3wml_description/urdf/fr3wml_suction.urdf.xacro')


@pytest.mark.skipif(not os.path.isfile(FR3WML_XACRO), reason='FR3WML xacro not present')
def test_detect_fr3wml_xacro():
    try:
        from trainit_setup_assistant.robotmodel import compile_xacro
        robot = parse_urdf(compile_xacro(FR3WML_XACRO))
    except Exception as exc:  # noqa: BLE001 - needs a sourced ROS env for $(find)
        pytest.skip(f'xacro compile needs a ROS env: {exc}')
    det = detect(robot)
    assert det.base_link == 'base_link'
    assert det.tip_link == 'tcp'
    assert det.arm_joints == ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    assert det.gripper_joints == ['suctioncup_joint']
    assert det.gripper_kind is GripperKind.SUCTION


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
