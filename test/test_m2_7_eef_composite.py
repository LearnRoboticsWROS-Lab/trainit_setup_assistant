"""vNext M2.7 (owner Q1 option A, Q7-i E1+E2, Q7-ii a0, 11.1): the composite MOCK bundle and the
contract checks that make "allowed" mean "works". Generated into temp dirs only.

* R22 narrowed: a composite bundle generates on mock, served by trainit_mock alone, with no tool
  joint claimed by a <ros2_control> block of the robot description the mock bring-up loads (the
  project's description; on the scene-loader flow also the base config's urdf.xacro, compiled
  with the mock mapping) -- refused otherwise, each reason named, fail closed.
* The joint-state routing: the composite mock row remaps eef/<id>/joint_states onto the robot's
  joint_states (RELATIVE); every simple row carries no remap (byte identity).
* The scene-loader SRDF merge drops a composite tool's named states, as srdf_builder does.
* R24 (E1) / R25 (E2): a movable joint on the kinematic path eef_parent_link -> a site TCP is driven
  by a follow_joint_trajectory actuator, and by one; a jaw off the path keeps its own interface.
* R26 (11.1): under a cell namespace the pseudo-joint echo topic is relative.

Fast and ROS-free except the description compile (xacro, in-process).
"""
import ast
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

import eef_fixture as F
import ws_paths
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator import eef_contract as E
from trainit_setup_assistant.generator.orchestrator import generate_scene_loader_config
from trainit_setup_assistant.model import (BundleSpec, GripperSpec, LaunchNodeSpec, NamedState,
                                           load_project)

GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
needs_golden = F.needs(GOLDEN_PROJECT, reason='the golden project is not present')
REMAP = [['eef/composite_tool/joint_states', 'joint_states']]


def _bridges(launch_text: str) -> list:
    for node in ast.walk(ast.parse(launch_text)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'BRIDGES':
            return ast.literal_eval(node.value)
    raise AssertionError('no BRIDGES')


def _rows(problems, row):
    return [m for m in problems if f'(ADR-0022 R{row})' in m]


# --- a synthetic hand-made BASE config for a composite (the scene-loader flow) -----------------

def _composite_base(tmp: Path, p='robot_a_', rc_joints=None, srdf_states=(('home', 'arm'),),
                    urdf=True):
    """A minimal hand-made base moveit_config of the composite fixture: config/robot_a.urdf.xacro
    (the description + a ros2_control block of ``rc_joints``), robot_a.srdf and
    ros2_controllers.yaml."""
    import shutil
    base = tmp / 'robot_a_moveit_config'
    cfg = base / 'config'
    cfg.mkdir(parents=True, exist_ok=True)
    shutil.copy(F.DATA / 'composite_cell.xacro', cfg / 'composite_cell.xacro')
    joints = ''.join(f'<joint name="{j}"><command_interface name="position"/><state_interface '
                     f'name="position"/></joint>' for j in (rc_joints or [f'{p}joint1']))
    if urdf:
        (cfg / 'robot_a.urdf.xacro').write_text(
            '<?xml version="1.0"?>\n'
            '<robot name="robot_a" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
            '  <xacro:arg name="use_isaac" default="false"/>\n'
            '  <xacro:include filename="composite_cell.xacro"/>\n'
            f'  <xacro:composite_cell p="{p}"/>\n'
            '  <ros2_control name="ROBOT_ASystem" type="system"><hardware><plugin>'
            f'mock_components/GenericSystem</plugin></hardware>{joints}</ros2_control>\n</robot>\n')
    states = ''.join(f'<group_state name="{n}" group="{g}"><joint name="{p}joint1" value="0"/>'
                     '</group_state>' for n, g in srdf_states)
    (cfg / 'robot_a.srdf').write_text(
        f'<?xml version="1.0"?>\n<robot name="robot_a"><group name="arm"><chain '
        f'base_link="{p}base_link" tip_link="{p}tool0"/></group>{states}\n</robot>\n')
    (cfg / 'ros2_controllers.yaml').write_text(
        '/**/controller_manager:\n  ros__parameters:\n    update_rate: 100\n')
    return base


def _scene_loader_composite(tmp: Path, **base_kw):
    p = F.composite(tmp)
    base = _composite_base(tmp, **base_kw)
    p.robot.base_moveit_config_path = str(base)
    if (base / 'config' / 'robot_a.urdf.xacro').is_file():
        p.robot.description.urdf_dir = str(base / 'config')
        p.robot.description.top_xacro = 'robot_a.urdf.xacro'
    return p


# --- R22 narrowed: the pin inverts ---------------------------------------------------------------

@pytest.mark.parametrize('build', [
    lambda t: F.composite(t),                                   # the conformance device, /r1
    lambda t: F.composite(t, 'robot_root', '', prefix=''),      # ... at the root
    lambda t: F.composite_mock(t, 'cm_root', '', prefix=''),    # the section-13 tool
    lambda t: F.composite_mock(t, 'robot_a', '/r1'),
], ids=['conformance_r1', 'conformance_root', 'section13_root', 'section13_r1'])
def test_r22_a_composite_mock_bundle_generates_from_both_entry_points(tmp_path, build):
    """BEFORE (TSA 7.7.0): refused, R22 'a composite BUNDLE is refused in M2.6'. AFTER: a bundle."""
    project = build(tmp_path / 'in')
    assert E.eef_contract_problems(project) == []
    Orchestrator().generate(project, tmp_path / 'bundle')
    generate_scene_loader_config(project, tmp_path / 'step3', 'cell_scene_loader_moveit_config')
    assert (tmp_path / 'bundle' / 'generation_manifest.yaml').is_file()


def test_r22_a_scene_loader_composite_generates_and_keeps_its_base_block(tmp_path):
    p = _scene_loader_composite(tmp_path / 'in')
    assert E.eef_contract_problems(p) == []
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    cfg = out / p.bundle.moveit_config_package
    assert (cfg / 'config' / 'robot_a.urdf.xacro').read_bytes() == (
        Path(p.robot.base_moveit_config_path) / 'config' / 'robot_a.urdf.xacro').read_bytes()
    rows = _bridges((cfg / 'launch' / 'bringup.launch.py').read_text())
    assert [(r['executable'], r['remappings']) for r in rows] == [('mock_eef_adapter', REMAP)]


@pytest.mark.parametrize('modes', [('mock', 'isaac'), ('mock', 'gazebo'), ('mock', 'real'),
                                   ('isaac',)])
def test_r22_beyond_mock_r20_unchanged_and_m6_named(tmp_path, modes):
    problems = E.eef_contract_problems(F.composite(tmp_path, modes=modes))
    assert any('a composite device runs on mock only in M2.6' in m for m in _rows(problems, 20))
    assert any('vNext M6 (ADR-0020)' in m and 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in m
               for m in _rows(problems, 22))


@pytest.mark.parametrize('rc, source', [
    (['robot_a_joint1', 'robot_a_rotary_axis_joint'], "config/robot_a.urdf.xacro"),
    (['robot_a_joint1', 'robot_a_jaw_a_follower_joint'], "config/robot_a.urdf.xacro"),
])
def test_r22_a_base_ros2_control_block_claiming_a_tool_joint_is_refused(tmp_path, rc, source):
    p = _scene_loader_composite(tmp_path / 'in', rc_joints=rc)
    text = '\n'.join(_rows(E.eef_contract_problems(p), 22))
    assert f"['{rc[1]}'] are in the <ros2_control> block \"ROBOT_ASystem\"" in text, text
    assert source in text and 'a mixed binding, vNext M6' in text
    assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' not in text           # one joint, two owners


def test_r22_fails_closed_on_a_base_description_it_cannot_compile(tmp_path):
    p = _scene_loader_composite(tmp_path / 'in')
    bad = Path(p.robot.base_moveit_config_path) / 'config' / 'robot_a.urdf.xacro'
    p.robot.description = F.composite(tmp_path / 'd').robot.description   # R9 reads another
    bad.write_text(bad.read_text().replace('</robot>', '<xacro:nope/></robot>'))
    text = '\n'.join(_rows(E.eef_contract_problems(p), 22))
    assert 'cannot be compiled' in text and 'refuses rather than guess (fail closed)' in text
    assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' not in text


# --- the joint-state routing -------------------------------------------------------------------

@pytest.mark.parametrize('ns', ['', '/r1'])
def test_the_composite_mock_row_routes_its_joints_onto_joint_states(tmp_path, ns):
    p = F.composite_mock(tmp_path / 'in', 'robot_a', ns, prefix='' if not ns else None)
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    launch = (out / p.bundle.app_package / 'launch' / 'bringup.launch.py').read_text()
    rows = _bridges(launch)
    assert [(r['executable'], r['modes'], r['remappings']) for r in rows] == [
        ('mock_eef_adapter', ['mock'], REMAP)]
    # the template passes a row's remappings to its Node, after the global tf pair under a namespace
    passed = 'remappings=tf_global + [tuple(r) for r in b.get("remappings", [])]'
    assert (passed in launch) == bool(ns)


@pytest.mark.parametrize('build', [F.suction_simple, F.jaw_simple, F.jaw_binary],
                         ids=['suction', 'gripper_command_jaw', 'binary_jaw'])
def test_a_simple_canonical_row_carries_no_remap(tmp_path, build):
    p = build(tmp_path / 'in')
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    rows = _bridges((out / p.bundle.app_package / 'launch' / 'bringup.launch.py').read_text())
    assert [r['remappings'] for r in rows] == [[]]


# --- the scene-loader SRDF merge drops a composite tool's named states --------------------------

def test_the_scene_loader_srdf_merge_filters_as_srdf_builder(tmp_path):
    states = [NamedState(name='ready', group='arm', joint_values={'robot_a_joint1': 0.3}),
              NamedState(name='tool_open', group='composite_tool',
                         joint_values={'robot_a_jaw_a_joint': 0.02})]
    sl = _scene_loader_composite(tmp_path / 'sl')
    sl.robot.named_states = list(states)
    out = tmp_path / 'b'
    Orchestrator().generate(sl, out)
    srdf = ET.fromstring(
        (out / sl.bundle.moveit_config_package / 'config' / 'robot_a.srdf').read_text())
    merged = [(g.get('name'), g.get('group')) for g in srdf.findall('group_state')]
    assert merged == [('home', 'arm'), ('ready', 'arm')]                 # no tool state
    fs = F.composite(tmp_path / 'fs')
    fs.robot.named_states = list(states)
    out2 = tmp_path / 'b2'
    Orchestrator().generate(fs, out2)
    srdf2 = ET.fromstring(
        (out2 / fs.bundle.moveit_config_package / 'config' / 'robot_a.srdf').read_text())
    assert [(g.get('name'), g.get('group')) for g in srdf2.findall('group_state')] == [
        ('ready', 'arm')]


# --- R24 / R25: E1 + E2 --------------------------------------------------------------------------

def _single(project, kind, contract, actions):
    data = F.dump(project.robot.gripper)
    data.update(kind=kind, contract=contract)
    data.pop('actuation_overridden', None)
    project.robot.gripper = GripperSpec(**data)
    project.application.tool_actions = F._actions(actions)
    return project


_GRASP = [('pick', 'grasp', {}), ('place', 'release', {})]
_NOTHING = {'require_grasp_confirmation': False, 'require_release_confirmation': False}


def _parallel_on_the_axis(tmp, interface):
    grip = ({'command_interface': 'set_bool', 'joints': ['robot_a_linear_axis_joint'],
             'joint_positions': {'engaged': [0.08], 'released': [0.0]}} if interface == 'set_bool'
            else {'command_interface': 'gripper_command', 'action': 'axis_controller/gripper_cmd',
                  'joints': ['robot_a_linear_axis_joint'], 'targets': {'open': 0.0, 'closed': 0.08},
                  'result_policy': 'succeeded'})
    return _single(F.composite(tmp), 'parallel', {
        'id': 'axis_jaw', 'actuators': {'gripper': grip},
        'sites': {'main': {'tcp': 'robot_a_site_a_tcp', 'grip': 'gripper'}},
        'verification': _NOTHING, 'adapters': {'mock': 'trainit_mock'}}, _GRASP)


def _suction_slide(tmp):
    return _single(F.composite(tmp), 'suction', {
        'id': 'suction_slide',
        'actuators': {'vacuum': {'command_interface': 'set_bool'},
                      'slide': {'command_interface': 'set_bool',
                                'joints': ['robot_a_linear_axis_joint'],
                                'joint_positions': {'engaged': [0.08], 'released': [0.0]}}},
        'sites': {'main': {'tcp': 'robot_a_site_a_tcp', 'grip': 'vacuum'}},
        'verification': _NOTHING, 'adapters': {'mock': 'trainit_mock'}},
        [('pick', 'actuate', {'actuator': 'slide', 'target': 'engage'}), *_GRASP])


REFUSED = {   # the preflight's cases (report_q.md section 3) -> (joint, row/actuator, received)
    'B': (lambda t: F.with_contract(F.composite(t), **{'actuators.linear_axis': {
        'command_interface': 'set_bool', 'joints': ['robot_a_linear_axis_joint'],
        'joint_positions': {'engaged': [0.08], 'released': [0.0]}}}),
        'robot_a_linear_axis_joint', 'actuator "linear_axis"', 'received "set_bool"'),
    'C': (lambda t: F.with_contract(F.composite(t), **{'actuators.rotary_axis': {
        'command_interface': 'set_bool', 'joints': ['robot_a_rotary_axis_joint'],
        'joint_positions': {'engaged': [0.9], 'released': [0.0]}, 'actuation_feedback': True}}),
        'robot_a_rotary_axis_joint', 'actuator "rotary_axis"', 'received "set_bool"'),
    'D': (lambda t: F.with_contract(F.composite(t), **{'actuators.linear_axis': {
        'command_interface': 'set_bool', 'actuation_feedback': True}}),
        'robot_a_linear_axis_joint', 'no actuator', 'received none'),
    'F': (lambda t: _parallel_on_the_axis(t, 'set_bool'),
          'robot_a_linear_axis_joint', 'actuator "gripper"', 'received "set_bool"'),
    'G': (_suction_slide, 'robot_a_linear_axis_joint', 'actuator "slide"', 'received "set_bool"'),
    'H': (lambda t: _parallel_on_the_axis(t, 'gripper_command'),
          'robot_a_linear_axis_joint', 'actuator "gripper"', 'received "gripper_command"'),
}
# G0 (owner section 9): how each case explains itself. An axis that grips nothing is the owner's
# E1 (an invalidity); a site's grip on the path reads as an axis declared as a grip OR as a jaw
# whose TCP sits on its own finger link (a limit of the current model), and names R8's half
TAILS = {
    'B': ('(owner Q7-i E1).', False), 'C': ('(owner Q7-i E1).', False),
    'G': ('(owner Q7-i E1).', False),
    'D': ('list it in a follow_joint_trajectory actuator\'s joints, or make it fixed.', False),
    'F': ('define the TCP on a link the jaw does not move.', True),
    'H': ('define the TCP on a link the jaw does not move.', True),
}


@pytest.mark.parametrize('case', sorted(REFUSED))
def test_r24_r25_an_axis_on_the_tcp_path_must_be_follow_joint_trajectory(tmp_path, case):
    """E1 (R24): the axis's actuator is not follow_joint_trajectory; E2 (R25): nothing drives it."""
    build, joint, actuator, received = REFUSED[case]
    project = build(tmp_path)
    row = 25 if actuator == 'no actuator' else 24
    rows = _rows(E.eef_contract_problems(project), row)
    hit = [m for m in rows if f'joint "{joint}"' in m]
    assert hit, rows
    assert actuator in hit[0] and 'interface expected "follow_joint_trajectory"' in hit[0]
    assert f'interface {received}' in hit[0]
    tail, limit = TAILS[case]
    assert hit[0].endswith(tail), hit[0]
    assert ('UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in hit[0]) == limit, hit[0]
    if limit:                                    # the advice never leads into R8's refusal
        assert ('"gripper" grips site "main"' in hit[0]
                and 'a follow_joint_trajectory actuator never grips (R8)' in hit[0]
                and 'a grip of its own' in hit[0]), hit[0]
    with pytest.raises(ValueError, match=rf'\(ADR-0022 R{row}\)'):
        Orchestrator().generate(project, tmp_path / 'b')


PASSES = {
    'conformance_composite': lambda t: F.composite(t),
    'section13_composite': lambda t: F.composite_mock(t),
    'suction_simple': lambda t: F.suction_simple(t),
    'jaw_simple': lambda t: F.jaw_simple(t),
    'jaw_binary': lambda t: F.jaw_binary(t),
    # a SetBool jaw whose joint is a side branch: valid (owner Q7-ii, label != interface)
    'binary_jaw_side_branch': lambda t: _single_jaw_side_branch(t),
}


def _single_jaw_side_branch(tmp):
    p = _single(F.composite(tmp), 'parallel', {
        'id': 'jaw_long', 'actuators': {'jaw': {
            'command_interface': 'set_bool', 'joints': ['robot_a_jaw_b_joint'],
            'joint_positions': {'engaged': [0.0], 'released': [0.02]}}},
        'sites': {'main': {'tcp': 'robot_a_site_b_tcp', 'grip': 'jaw'}},
        'verification': _NOTHING, 'adapters': {'mock': 'trainit_mock'}}, _GRASP)
    p.robot.gripper.eef_parent_link = 'robot_a_wrist_link2'
    return p


@pytest.mark.parametrize('case', sorted(PASSES))
def test_r24_r25_pass_every_valid_contract(tmp_path, case):
    problems = E.eef_contract_problems(PASSES[case](tmp_path))
    assert _rows(problems, 24) == [] and _rows(problems, 25) == []


@needs_golden
def test_r24_r25_r26_pass_the_opted_in_golden(tmp_path):
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix('cell')
    assert E.eef_contract_problems(F.opt_in(p)) == []


def test_r25_judges_a_mimic_follower_by_its_master(tmp_path):
    extra = ('<link name="robot_a_m_link"/><joint name="robot_a_m_joint" type="prismatic"><parent '
             'link="robot_a_tool_base"/><child link="robot_a_m_link"/><axis xyz="1 0 0"/><limit '
             'lower="0.0" upper="0.1" effort="1.0" velocity="1.0"/></joint>'
             '<link name="robot_a_f_link"/><joint name="robot_a_f_joint" type="prismatic"><parent '
             'link="robot_a_wrist_link2"/><child link="robot_a_f_link"/><axis xyz="1 0 0"/><limit '
             'lower="0.0" upper="0.1" effort="1.0" velocity="1.0"/><mimic joint="robot_a_m_joint"/>'
             '</joint><link name="robot_a_site_d_tcp"/><joint name="robot_a_site_d_tcp_joint" '
             'type="fixed"><parent link="robot_a_f_link"/><child link="robot_a_site_d_tcp"/>'
             '</joint>')
    p = F.composite(tmp_path)
    p.robot.description = F.description(tmp_path, 'robot_a_m', 'composite_cell.xacro',
                                        '<xacro:composite_cell p="robot_a_"/>\n  ' + extra)
    p = F.with_contract(p, **{'sites.site_b.tcp': 'robot_a_site_d_tcp'})
    rows = _rows(E.eef_contract_problems(p), 25)
    assert any('joint "robot_a_f_joint" (prismatic, a mimic follower of "robot_a_m_joint")' in m
               and 'no actuator' in m for m in rows), rows
    # M2.7 final review PY-1: the remedy names the MASTER -- listing the follower is R5's refusal
    hit = [m for m in rows if 'joint "robot_a_f_joint"' in m][0]
    assert ('list its master "robot_a_m_joint" in a follow_joint_trajectory actuator\'s joints'
            in hit and 'or make "robot_a_f_joint" fixed.' in hit), hit
    assert "list it in a follow_joint_trajectory actuator's joints" not in hit, hit
    assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' not in hit, hit
    follower = F.with_contract(p, **{'actuators.wrist_mechanism.joints': [
        'robot_a_wrist_pitch_joint', 'robot_a_wrist_roll_joint', 'robot_a_f_joint'],
        'actuators.wrist_mechanism.targets': {'stow': [0.0, 0.0, 0.0],
                                              'present': [0.3, -0.3, 0.05]},
        'actuators.wrist_mechanism.goal_tolerance': [0.01, 0.01, 0.01]})
    problems = E.eef_contract_problems(follower)
    assert any('joint "robot_a_f_joint" is a mimic follower' in m for m in _rows(problems, 5))
    assert any('joint "robot_a_f_joint"' in m for m in _rows(problems, 25))
    # its master driven by an FJT actuator: nothing to say
    q = F.with_contract(p, **{'actuators.wrist_mechanism.joints': [
        'robot_a_wrist_pitch_joint', 'robot_a_wrist_roll_joint', 'robot_a_m_joint'],
        'actuators.wrist_mechanism.targets': {'stow': [0.0, 0.0, 0.0],
                                              'present': [0.3, -0.3, 0.05]},
        'actuators.wrist_mechanism.goal_tolerance': [0.01, 0.01, 0.01]})
    problems = E.eef_contract_problems(q)
    assert _rows(problems, 24) == [] and _rows(problems, 25) == []


def test_r25_a_follower_of_the_command_joint_is_a_limit_of_the_model(tmp_path):
    """G0 (owner section 9): a TCP below the command_joint pseudo-joint -- here through its
    mimic follower -- is a limit of the current canonical model; the remedy makes the FOLLOWER
    fixed and never clears command_joint (the pseudo-joint itself stays movable, R23)."""
    extra = ('<link name="robot_a_m_link"/><joint name="robot_a_m_joint" type="prismatic"><parent '
             'link="robot_a_tool_base"/><child link="robot_a_m_link"/><axis xyz="1 0 0"/><limit '
             'lower="0.0" upper="0.1" effort="1.0" velocity="1.0"/></joint>'
             '<link name="robot_a_f_link"/><joint name="robot_a_f_joint" type="prismatic"><parent '
             'link="robot_a_wrist_link2"/><child link="robot_a_f_link"/><axis xyz="1 0 0"/><limit '
             'lower="0.0" upper="0.1" effort="1.0" velocity="1.0"/><mimic joint="robot_a_m_joint"/>'
             '</joint><link name="robot_a_site_d_tcp"/><joint name="robot_a_site_d_tcp_joint" '
             'type="fixed"><parent link="robot_a_f_link"/><child link="robot_a_site_d_tcp"/>'
             '</joint>')
    p = F.composite(tmp_path)
    p.robot.description = F.description(tmp_path, 'robot_a_m', 'composite_cell.xacro',
                                        '<xacro:composite_cell p="robot_a_"/>\n  ' + extra)
    p = F.with_contract(p, **{'sites.site_b.tcp': 'robot_a_site_d_tcp'})
    data = F.dump(p.robot.gripper)
    data.update(command_joint='robot_a_m_joint')
    data.pop('actuation_overridden', None)
    p.robot.gripper = GripperSpec(**data)
    hit = [m for m in _rows(E.eef_contract_problems(p), 25) if 'joint "robot_a_f_joint"' in m]
    assert len(hit) == 1, hit
    assert ('UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in hit[0]
            and '"robot_a_m_joint" is the command_joint pseudo-joint' in hit[0]
            and hit[0].endswith('or make "robot_a_f_joint" fixed.')), hit[0]
    assert 'clear command_joint' not in hit[0], hit[0]


# --- R26: 11.1 ----------------------------------------------------------------------------------

def _merger(project, topic, modes=('mock',)):
    project.deployment.bridges = list(project.deployment.bridges) + [LaunchNodeSpec(
        package='fairino_gripper', executable='joint_state_merger',
        parameters=[{'gripper_joint_states_in': topic}], modes=list(modes))]
    return project


@pytest.mark.parametrize('ns, topic, refused', [
    ('/r3', '/suction/joint_states', True),       # absolute, outside the namespace
    ('/r3', '/r3/suction/joint_states', True),    # absolute, even under it (D7: no absolute name)
    ('/r3', '~/suction/joint_states', True),      # private: the adapter's and the merger's differ
    ('/r3', 'suction/joint_states', False),       # relative: this robot's
    ('', '/suction/joint_states', False),         # the root: the legacy name, unchanged
])
def test_r26_the_echo_topic_under_a_namespace(tmp_path, ns, topic, refused):
    p = _merger(F.suction_simple(tmp_path, ns=ns), topic)
    rows = _rows(E.eef_contract_problems(p), 26)
    assert bool(rows) == refused, rows
    if refused:
        assert (f'robot namespace "{ns}"' in rows[0] and f'({ns}/eef/suction_simple)' in rows[0]
                and f'echo topic "{topic}"' in rows[0]
                and 'expected relative form "suction/joint_states"' in rows[0]), rows[0]
        assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' not in rows[0]      # a contract rule
    else:
        out = tmp_path / 'b'
        Orchestrator().generate(p, out)
        rows = _bridges((out / p.bundle.app_package / 'launch' / 'bringup.launch.py').read_text())
        mock = [r for r in rows if r['executable'] == 'mock_eef_adapter'][0]
        assert mock['parameters'][1]['compat.pseudo_joint_topic'] == topic


def test_r26_never_reaches_a_legacy_project(tmp_path):
    """Legacy unchanged: the example (a legacy project with an absolute merger input) under a
    namespace still generates, with today's namespace warning."""
    p = load_project(str(F.DATA.parents[2] / 'examples' / 'fr3wml_project.yaml'))
    p.robot.cell_namespace = '/r3'
    assert E.eef_contract_problems(p) == []


# --- the section-13 fixture ----------------------------------------------------------------------

def test_the_section13_block_normalises_with_the_pinned_reference(tmp_path):
    p = F.composite_mock(tmp_path / 'in')
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    N = F.normalizer()
    params = out / p.bundle.app_package / 'config' / 'bt_params.yaml'
    dump = N.dump(N.parse(N.load_params(str(params))))
    for a in ('slide_axis', 'rotation_axis', 'controlled_gripper', 'binary_lock', 'binary_jaw'):
        assert f'composite_tool.actuators.{a}.command_interface' in dump


def test_the_section13_fixture_is_launchable_config(tmp_path):
    """The bring-up's robot description compiles only with every ros2_control joint's initial
    position, and launch_ros refuses a null joint_limits (WP_COMP traps 1 and 2)."""
    p = F.composite_mock(tmp_path / 'in', 'cm_root', '', prefix='')
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    cfg = out / p.bundle.moveit_config_package / 'config'
    assert yaml.safe_load((cfg / 'initial_positions.yaml').read_text()) == {
        'initial_positions': {'joint1': 0.0}}
    assert yaml.safe_load((cfg / 'joint_limits.yaml').read_text())['joint_limits']['joint1'][
        'has_acceleration_limits'] is True
    assert 'projection_evaluator: joints(joint1)' in (cfg / 'ompl_planning.yaml').read_text()
