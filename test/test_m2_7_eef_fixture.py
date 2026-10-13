"""vNext M2.7: the test fixture the M2.7 launch tests build on (``eef_fixture``), pinned. Fast;
ROS-free except xacro.

* Launch-safe is an OPT-IN of the M2.6 builders: by default their projects -- and so every file
  generated from them and compared byte for byte elsewhere -- are what they were, and
  ``launch_safe=True`` moves exactly the three launch-safe settings, nothing else.
* ``composite_mock`` is launch-safe for its own prefix: the root, /r1 and /r2 cells.
* The section-13 description is the tree its contract names: every actuator joint movable, both
  site TCPs below both axes, the jaws side branches, the lock jointless. Its bytes are the ones the
  M2.7 BEFORE launch capture was taken with.
"""
import hashlib
import xml.etree.ElementTree as ET

import pytest

import eef_fixture as F
from trainit_setup_assistant.model.robot import JointLimit, OmplGroupSpec

LAUNCH_SAFE_FIELDS = ['initial_positions', 'joint_limits', 'ompl']
M2_6_BUILDERS = {'suction_simple': F.suction_simple, 'jaw_simple': F.jaw_simple,
                 'jaw_binary': F.jaw_binary, 'composite': F.composite}
CELLS = [('cm_root', '', ''), ('robot_a', '/r1', 'robot_a_'), ('robot_b', '/r2', 'robot_b_')]
COMPOSITE_MOCK_XACRO_SHA256 = '8593707049d832f629eb7e99006deaab3673cf09c670e5d8fe1af41ccb1cf4a0'
LIMIT = dict(has_velocity_limits=True, max_velocity=1.0, has_acceleration_limits=True,
             max_acceleration=2.0)


def _robot(project) -> dict:
    data = F.dump(project.robot)
    data['description'].pop('urdf_dir')              # every project has its own temp dir
    return data


def _assert_launch_safe(project, joints):
    assert project.robot.initial_positions == {j: 0.0 for j in joints}
    assert project.robot.joint_limits.per_joint == {j: JointLimit(**LIMIT) for j in joints}
    assert project.robot.ompl.projection_evaluator == f'joints({",".join(joints)})'


def _assert_not_launch_safe(project):
    assert project.robot.initial_positions == {}
    assert project.robot.joint_limits.per_joint == {}
    assert project.robot.ompl.projection_evaluator == OmplGroupSpec().projection_evaluator


@pytest.mark.parametrize('name', sorted(M2_6_BUILDERS))
def test_launch_safe_is_an_opt_in_of_the_m2_6_builders(tmp_path, name):
    build = M2_6_BUILDERS[name]
    default, off = build(tmp_path / 'default'), build(tmp_path / 'off', launch_safe=False)
    on = build(tmp_path / 'on', launch_safe=True)
    assert _robot(default) == _robot(off)
    _assert_not_launch_safe(default)
    _assert_launch_safe(on, list(on.robot.planning_group.joints))
    # the three settings and nothing else: no other robot field, no other joint_limits or ompl
    # key, the same application and deployment
    a, b = _robot(default), _robot(on)
    assert sorted(k for k in a if a[k] != b[k]) == LAUNCH_SAFE_FIELDS
    assert {**a['joint_limits'], 'per_joint': None} == {**b['joint_limits'], 'per_joint': None}
    assert ({**a['ompl'], 'projection_evaluator': None} ==
            {**b['ompl'], 'projection_evaluator': None})
    assert F.dump(default.application) == F.dump(on.application)
    assert F.dump(default.deployment) == F.dump(on.deployment)


@pytest.mark.parametrize('robot, ns, p', CELLS, ids=['root', 'r1', 'r2'])
def test_composite_mock_is_launch_safe_for_its_own_prefix(tmp_path, robot, ns, p):
    project = F.composite_mock(tmp_path / 'on', robot, ns, prefix=p)
    assert project.robot.namespace() == ns
    assert list(project.robot.planning_group.joints) == [f'{p}joint1']
    _assert_launch_safe(project, [f'{p}joint1'])
    _assert_not_launch_safe(F.composite_mock(tmp_path / 'off', robot, ns, prefix=p,
                                             launch_safe=False))
    contract = F.composite_mock_contract(p)
    assert project.robot.gripper.kind.value == 'composite'
    assert contract['id'] == 'composite_tool' and contract['adapters'] == {'mock': 'trainit_mock'}
    joints = [j for a in contract['actuators'].values() for j in a.get('joints', [])]
    assert joints and all(j.startswith(p) for j in joints)


def test_the_section13_description_is_the_tree_its_contract_names(tmp_path):
    from trainit_setup_assistant.robotmodel.xacro_loader import compile_xacro
    xacro = F.DATA_M27 / 'composite_mock_cell.xacro'
    assert hashlib.sha256(xacro.read_bytes()).hexdigest() == COMPOSITE_MOCK_XACRO_SHA256, \
        'the description the M2.7 BEFORE launch capture was taken with was edited'
    p = 'robot_a_'
    project = F.composite_mock(tmp_path, 'robot_a', '/r1', prefix=p)
    desc = project.robot.description
    urdf = ET.fromstring(compile_xacro(f'{desc.urdf_dir}/{desc.top_xacro}'))
    joints = {j.get('name'): j for j in urdf.iter('joint')}
    parent = {j.find('child').get('link'): name for name, j in joints.items()}

    def path(link):                                    # the joints from the root down to link
        out = []
        while link in parent:
            out.append(parent[link])
            link = joints[parent[link]].find('parent').get('link')
        return out

    contract = F.composite_mock_contract(p)
    actuator_joints = {j for a in contract['actuators'].values() for j in a.get('joints', [])}
    active = {n for n, j in joints.items() if j.get('type') != 'fixed' and j.find('mimic') is None}
    assert active == {f'{p}joint1'} | actuator_joints
    follower = joints[f'{p}gripper_jaw_follower_joint'].find('mimic')
    assert follower is not None and follower.get('joint') == f'{p}gripper_jaw_joint'
    for site in contract['sites'].values():           # E1/E2: both TCPs move with both axes
        assert {f'{p}slide_joint', f'{p}rotation_joint'} <= set(path(site['tcp']))
        assert not {f'{p}gripper_jaw_joint', f'{p}binary_jaw_joint'} & set(path(site['tcp']))
    assert 'joints' not in contract['actuators']['binary_lock']
    assert joints[f'{p}lock_joint'].get('type') == 'fixed'
