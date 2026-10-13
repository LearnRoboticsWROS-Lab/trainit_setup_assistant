"""F2 (vNext M2.6, ADR-0022 D6 "Refusals"): every refusal row R1-R26, each refused from BOTH
generation entry points -- ``Orchestrator.generate`` and ``generate_scene_loader_config`` (the
GUI's Step 3) -- before a single file exists, with the row named and the device, actuator, site
or backend involved. R2 (an unknown key or token inside the contract) is refused at LOAD, where a
project in memory cannot even hold one; its generation-time half is a value out of range.

The refusals hold only when a contract is present, except R11 for ``actuate`` and R19 (and kind
``composite``, a token that exists only with a contract): every legacy project generates as
before (``test_m2_6_eef_contract.py`` checks every identity project).

The table has one case per refusal CONDITION, not per row: a meta-test traces every case and
requires each place ``generator/eef_contract.py`` emits a refusal (each ``_prefix(N)`` line) to be
reached by one (review TSA-3, tests lens).

Fast and ROS-free except the description compile (xacro, in-process).
"""
import re
import sys
from pathlib import Path

import pytest

import eef_fixture as F
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator import eef_contract as E
from trainit_setup_assistant.generator.orchestrator import generate_scene_loader_config
from trainit_setup_assistant.model import (AppType, Backend, GripperSpec, LaunchNodeSpec,
                                           PolicyStep, ToolAction, load_project)
from trainit_setup_assistant.model.enums import EndEffectorActuation, ToolActionKind

EXAMPLE = F.DATA.parents[2] / 'examples' / 'fr3wml_project.yaml'


def _refused_from_both(project, tmp_path, row: int, needle: str):
    out1, out2 = tmp_path / 'bundle', tmp_path / 'step3'
    with pytest.raises(ValueError) as bundle:
        Orchestrator().generate(project, out1)
    with pytest.raises(ValueError) as step3:
        generate_scene_loader_config(project, out2, 'cell_scene_loader_moveit_config')
    for err in (bundle, step3):
        text = str(err.value)
        assert f'(ADR-0022 R{row})' in text, text
        assert needle in text, text
    assert not out1.exists() and not out2.exists()


def _gripper(project, **changes):
    data = F.dump(project.robot.gripper)
    data.update(changes)
    project.robot.gripper = GripperSpec(**data)
    return project


def _jaw(tmp, **actuator):
    contract = F.jaw_contract()
    contract['actuator'].update(actuator)
    for key in [k for k, v in actuator.items() if v is F.DELETE]:
        del contract['actuator'][key]
    return F.jaw_simple(tmp, contract=contract)


def _binary(tmp, **actuator):
    contract = F.binary_jaw_contract()
    contract['actuator'].update(actuator)
    return F.jaw_binary(tmp, contract=contract)


def _composite(tmp, **changes):
    return F.with_contract(F.composite(tmp), **changes)


def _with_actions(project, *rows):
    project.application.tool_actions = list(project.application.tool_actions) + [
        ToolAction(at_waypoint=w, kind=ToolActionKind(k), **kw) for w, k, kw in rows]
    return project


def _policy(project, card, **step):
    project.application.policies = [PolicyStep(name='grasp', before_waypoint='pick', mode='pure',
                                               card=str(card), **step)]
    return project


def _card(tmp, on='/suction/on', off='/suction/off', name='card'):
    """A synthetic suction card; ``on``/``off`` are written as YAML double-quoted scalars, so a
    test can put an escape (a trailing newline) into a service name."""
    card = tmp / name / 'policy_card.yaml'
    card.parent.mkdir(parents=True, exist_ok=True)
    pair = ', '.join(f'{key}: "{value}"' for key, value in (('on_service', on),
                                                            ('off_service', off)))
    card.write_text('policy_card:\n  name: synthetic_suction\n  end_effector: suction\n'
                    f'  grasp: {{{pair}}}\n'
                    '  end_state: {tcp_pose_base: [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}\n')
    return card


def _two_policies(tmp):
    """Two policy steps whose cards name different Trigger pairs (one device serves one)."""
    project = F.suction_simple(tmp, ns='')
    project.application.policies = [
        PolicyStep(name='grasp', before_waypoint='pick', mode='pure', card=str(_card(tmp))),
        PolicyStep(name='again', before_waypoint='place', mode='pure',
                   card=str(_card(tmp, on='/vacuum/on', off='/vacuum/off', name='card_b')))]
    return project


def _ros2_control(flavour):
    return {'adapter': 'ros2_control', 'flavour': flavour}


def _legacy_example():
    return load_project(str(EXAMPLE))


def _long_suction(tmp, actuators=None, sites=None, **gripper):
    """The simple suction written in the LONG form (an actuators map and a sites map)."""
    changes = {'actuator': F.DELETE}
    if actuators is not None:
        changes['actuators'] = actuators
    if sites is not None:
        changes['sites'] = sites
    return F.with_contract(F.suction_simple(tmp, **gripper), **changes)


_VACUUM = {'vacuum': {'command_interface': 'set_bool'}}
_MAIN_SITE = {'main': {'tcp': 'tcp', 'grip': 'vacuum'}}


def _long_binary_jaw(tmp, actuators=None, sites=None):
    """The binary jaw in the LONG form."""
    gripper = {'command_interface': 'set_bool', 'joints': ['jaw_joint'],
               'joint_positions': {'engaged': [0.0], 'released': [0.02]}}
    return F.with_contract(
        F.jaw_binary(tmp), actuator=F.DELETE,
        actuators=actuators if actuators is not None else {'gripper': gripper},
        sites=sites if sites is not None else {'main': {'tcp': 'tool0', 'grip': 'gripper'}})


def _isaac(tmp, ns='', grasp=False, **binding):
    """The simple suction on mock and isaac, the isaac binding in its LONG form."""
    isaac = dict({'adapter': 'trainit_isaac_surface_gripper'}, **binding)
    return F.suction_simple(tmp, ns=ns, modes=('mock', 'isaac'), contract=F.suction_contract(
        adapters={'mock': 'trainit_mock', 'isaac': isaac}, grasp=grasp))


def _base_config(project, tmp, flavour='effort'):
    """A synthetic hand-made base config whose gripper controller is a ``<flavour>_controllers``
    GripperActionController: what TSA reads a gazebo binding's flavour from (ADR-0022 D2.B)."""
    base = tmp / 'base_moveit_config'
    (base / 'config').mkdir(parents=True, exist_ok=True)
    (base / 'config' / 'ros2_controllers.yaml').write_text(
        'controller_manager:\n  ros__parameters:\n    gripper_controller:\n'
        f'      type: {flavour}_controllers/GripperActionController\n')
    project.robot.base_moveit_config_path = str(base)
    return project


def _gazebo_jaw(tmp, binding='ros2_control', max_effort=None, flavour='effort'):
    """The GripperCommand jaw on mock and gazebo, its flavour read from the base config."""
    contract = F.jaw_contract(adapters={'mock': 'trainit_mock', 'gazebo': binding})
    if max_effort is not None:
        contract['actuator']['max_effort'] = max_effort
    return _base_config(F.jaw_simple(tmp, modes=('mock', 'gazebo'), contract=contract), tmp,
                        flavour)


def _bin(project, **cfg):
    """The project switched to bin picking, ``cfg`` set on its bin_picking block."""
    project.application.type = AppType.BIN_PICKING
    for key, value in cfg.items():
        setattr(project.application.bin_picking, key, value)
    return project


def _described(project, tmp, extra: str, tool: str = 'suction_tool'):
    """The project on the simple-tools description (its ``tool`` macro) plus ``extra`` URDF
    lines."""
    project.robot.description = F.description(tmp, 'extended', 'simple_tools.xacro',
                                              f'<xacro:{tool} p=""/>\n  ' + extra)
    return project


def _kind_in_memory(project, kind: str):
    """A tool action whose kind no TSA renders (assignment skips validation, as for R19): what
    a future kind without a canonical node would be."""
    project.application.tool_actions[0].kind = kind
    return project


CASES = {
    # R1 kind / structure
    'r1_suction_with_a_jaw': (
        1, 'kind suction is 1..N set_bool actuators',
        lambda t: F.suction_simple(
            t, contract=dict(F.suction_contract(), actuator={
                'command_interface': 'gripper_command', 'targets': {'open': 0.0, 'closed': 1.0},
                'result_policy': 'succeeded', 'action': 'x_controller/gripper_cmd',
                'joints': ['suctioncup_joint']}))),
    'r1_short_form_composite': (
        1, 'the short form is for kind suction or parallel',
        lambda t: _gripper(F.suction_simple(t), kind='composite')),
    'r1_kind_none_with_a_contract': (
        1, 'kind "none" has no contract',
        lambda t: _gripper(F.suction_simple(t), kind='none')),
    'r1_composite_without_a_contract': (
        1, 'kind "composite" without a contract block',
        lambda t: _gripper(_legacy_example(), kind='composite')),
    'r1_short_and_long_form': (
        1, 'cannot be combined with actuators/sites',
        lambda t: F.with_contract(
            F.suction_simple(t), sites={'main': {'tcp': 'tcp', 'grip': 'vacuum'}})),
    'r1_actuators_without_sites': (
        1, 'the long form needs both an actuators map and a sites map',
        lambda t: _long_suction(t, actuators=_VACUUM)),
    'r1_an_empty_actuators_map': (
        1, 'actuators is empty',
        lambda t: _long_suction(t, actuators={}, sites=_MAIN_SITE)),
    'r1_an_empty_sites_map': (
        1, 'sites is empty',
        lambda t: _long_suction(t, actuators=_VACUUM, sites={})),
    'r1_suction_with_two_sites': (
        1, 'kind suction has exactly one site, not 2',
        lambda t: _long_suction(
            t, actuators={'vacuum': {'command_interface': 'set_bool'},
                          'vacuum_b': {'command_interface': 'set_bool'}},
            sites={'main': {'tcp': 'tcp', 'grip': 'vacuum'},
                   'side': {'tcp': 'suction_body', 'grip': 'vacuum_b'}})),
    'r1_parallel_with_two_actuators': (
        1, 'kind parallel is exactly one actuator',
        lambda t: _long_binary_jaw(
            t, actuators={'gripper': {'command_interface': 'set_bool', 'joints': ['jaw_joint'],
                                      'joint_positions': {'engaged': [0.0],
                                                          'released': [0.02]}},
                          'clamp': {'command_interface': 'set_bool'}})),
    'r1_parallel_with_two_sites': (
        1, 'kind parallel has exactly one site, not 2',
        lambda t: _long_binary_jaw(
            t, sites={'main': {'tcp': 'tool0', 'grip': 'gripper'},
                      'other': {'tcp': 'tcp', 'grip': 'gripper'}})),
    # R2 a value the schema holds but the contract refuses (the unknown key/token: at load)
    'r2_command_timeout': (
        2, 'command_timeout_ms 0',
        lambda t: F.with_contract(F.suction_simple(t), command_timeout_ms=0)),
    # a NaN or an infinity passes every "> 0" check: refused as a value, before any other rule
    'r2_a_non_finite_number': (
        2, 'is not a finite number',
        lambda t: _jaw(t, targets={'open': 0.0, 'closed': float('nan')})),
    # R3 names
    'r3_device_id_eef': (
        3, 'device id \'eef\'',
        lambda t: F.with_contract(F.suction_simple(t), id='eef')),
    # eef.<id> with the id 'ids' would overwrite the block's own eef.ids list
    'r3_device_id_ids': (
        3, 'device id \'ids\'',
        lambda t: F.with_contract(F.suction_simple(t), id='ids')),
    'r3_device_id_not_a_token': (
        3, 'device id \'1st\'',
        lambda t: F.with_contract(F.suction_simple(t), id='1st')),
    # '$' would match before a trailing newline; the token rule is anchored with \Z
    'r3_device_id_with_a_trailing_newline': (
        3, 'device id \'main\\n\'',
        lambda t: F.with_contract(F.suction_simple(t), id='main\n')),
    # a name token is ASCII in every parser of the block (normalize_eef.py, TMR, the adapters)
    'r3_device_id_not_ascii': (
        3, 'device id \'mäin\'',
        lambda t: F.with_contract(F.suction_simple(t), id='mäin')),
    'r3_absolute_action': (
        3, 'is not a RELATIVE ROS name',
        lambda t: _jaw(t, action='/gripper_controller/gripper_cmd')),
    'r3_namespace_segment_eef': (
        3, 'segment named "eef"',
        lambda t: F.suction_simple(t, ns='/eef')),
    'r3_actuator_and_site_share_a_token': (
        3, 'is both an actuator and a site id',
        lambda t: _composite(
            t, **{'sites.gripper_b': {'tcp': 'robot_a_site_b_tcp', 'grip': 'gripper_b'},
                  'sites.site_b': F.DELETE})),
    'r3_an_actuator_id_not_a_token': (
        3, 'actuator id \'9vacuum\' is not a ROS name token',
        lambda t: F.with_contract(F.suction_simple(t), **{'actuator.id': '9vacuum'})),
    'r3_an_id_inside_the_actuators_map': (
        3, '"id" is the short form\'s',
        lambda t: _long_suction(
            t, actuators={'vacuum': {'command_interface': 'set_bool', 'id': 'vacuum'}},
            sites=_MAIN_SITE)),
    'r3_a_target_name_not_a_token': (
        3, 'target name \'half-way\' is not a ROS name token',
        lambda t: _jaw(t, targets={'open': 0.0, 'closed': 0.7, 'half-way': 0.35})),
    'r3_a_site_id_not_a_token': (
        3, 'site id \'2nd\' is not a ROS name token',
        lambda t: _long_suction(t, actuators=_VACUUM,
                                sites={'2nd': {'tcp': 'tcp', 'grip': 'vacuum'}})),
    # _ROS_NAME is anchored with \Z: '$' would match before a trailing newline (review TSA-7)
    'r3_a_stage_topic_with_a_trailing_newline': (
        3, 'stage_cmd_topic \'suction_cmd\\n\' is not a ROS topic name',
        lambda t: _isaac(t, stage_cmd_topic='suction_cmd\n')),
    # R4 actions
    'r4_the_arm_controllers_action': (
        4, 'is the arm controller\'s action',
        lambda t: _jaw(
            t, action='moveit_joint_controller/follow_joint_trajectory')),
    'r4_two_actuators_one_action': (
        4, 'share the action',
        lambda t: _composite(
            t, **{'actuators.rotary_axis.action':
                  'linear_axis_controller/follow_joint_trajectory'})),
    # R5 joints
    'r5_a_missing_joint': (
        5, 'joint "nope_joint" is not in the description',
        lambda t: _binary(t, joints=['nope_joint'])),
    'r5_a_mimic_follower': (
        5, 'mimic follower',
        lambda t: _composite(
            t, **{'actuators.gripper_a.joints': ['robot_a_jaw_a_follower_joint']})),
    'r5_an_arm_joint': (
        5, 'belongs to the arm group',
        lambda t: _composite(
            t, **{'actuators.gripper_b.joints': ['robot_a_joint1']})),
    'r5_a_fixed_joint': (
        5, 'is fixed',
        lambda t: _binary(t, joints=['tcp_joint'])),
    'r5_shared_by_two_actuators': (
        5, 'is also driven by actuator',
        lambda t: _composite(
            t, **{'actuators.gripper_b.joints': ['robot_a_jaw_a_joint']})),
    # R6 set_bool
    'r6_joint_positions_without_joints': (
        6, 'joints and joint_positions go together',
        lambda t: _binary(
            t, joints=None)),
    'r6_length_mismatch': (
        6, 'one number per joint',
        lambda t: _binary(
            t, joint_positions={'engaged': [0.0, 0.0], 'released': [0.02]})),
    'r6_outside_the_limits': (
        6, 'is outside its limits',
        lambda t: _binary(
            t, joint_positions={'engaged': [0.5], 'released': [0.02]})),
    'r6_confirm_timeout_zero': (
        6, 'confirm_timeout_ms 0 is not a positive integer',
        lambda t: _binary(t, confirm_timeout_ms=0)),
    # R7 gripper_command
    'r7_no_result_policy': (
        7, 'result_policy is required',
        lambda t: _jaw(t, result_policy=F.DELETE)),
    'r7_a_target_outside_the_limits': (
        7, 'is outside its limits',
        lambda t: _jaw(
            t, targets={'open': 0.0, 'closed': 2.0})),
    'r7_max_effort_not_positive': (
        7, 'must be > 0',
        lambda t: _jaw(t, max_effort=-1.0)),
    'r7_open_missing': (
        7, 'targets need "open" and "closed"',
        lambda t: _jaw(
            t, targets={'closed': 0.7, 'half': 0.3})),
    'r7_an_effort_controller_without_max_effort': (
        7, 'an effort-flavour controller',
        lambda t: F.jaw_simple(
            t, modes=('mock', 'real'), contract=F.jaw_contract(adapters={
                'mock': 'trainit_mock', 'real': _ros2_control('effort')}))),
    'r7_a_gripper_command_that_grips_nothing': (
        7, 'must be the grip of a site',
        lambda t: _composite(
            t, **{'sites.site_a.grip': 'gripper_b', 'sites.site_b.grip': 'binary_lock'})),
    # the reference normaliser refuses equal values (an adapter tells a command by its position)
    'r7_two_targets_share_a_value': (
        7, 'targets [[\'closed\', \'pinch\']] share a value',
        lambda t: _jaw(t, targets={'open': 0.0, 'closed': 0.7, 'pinch': 0.7})),
    'r7_targets_cannot_be_derived': (
        7, 'targets cannot be derived',
        lambda t: _jaw(t, targets=F.DELETE)),
    'r7_no_action': (
        7, 'actuator "gripper_a": the action is required',
        lambda t: _composite(t, **{'actuators.gripper_a.action': F.DELETE})),
    'r7_two_joints': (
        7, 'drives exactly one joint',
        lambda t: _composite(t, **{'actuators.gripper_a.joints': [
            'robot_a_jaw_a_joint', 'robot_a_jaw_b_joint']})),
    'r7_no_joints': (
        7, 'joints are required (exactly one)',
        lambda t: _composite(t, **{'actuators.gripper_a.joints': F.DELETE})),
    'r7_no_targets': (
        7, 'actuator "gripper_a": targets are required',
        lambda t: _composite(t, **{'actuators.gripper_a.targets': F.DELETE})),
    # gazebo reads the flavour from the base config (D2.B): effort there needs max_effort
    'r7_an_effort_base_config_without_max_effort': (
        7, 'an effort-flavour controller never moves with max_effort 0.0',
        lambda t: _gazebo_jaw(t)),
    # R8 follow_joint_trajectory
    'r8_no_goal_tolerance': (
        8, 'goal_tolerance is required',
        lambda t: _composite(
            t, **{'actuators.linear_axis.goal_tolerance': F.DELETE})),
    'r8_above_the_velocity_limit': (
        8, 'URDF velocity limit',
        lambda t: _composite(
            t, **{'actuators.linear_axis.duration_s': 0.01})),
    'r8_a_target_of_the_wrong_length': (
        8, 'has 1 values for 2 joints',
        lambda t: _composite(
            t, **{'actuators.wrist_mechanism.targets.stow': [0.0]})),
    'r8_used_as_a_grip': (
        8, 'never grips',
        lambda t: _composite(
            t, **{'sites.site_b.grip': 'linear_axis'})),
    'r8_no_joints': (
        8, 'actuator "linear_axis": joints are required',
        lambda t: _composite(t, **{'actuators.linear_axis.joints': F.DELETE})),
    'r8_no_action': (
        8, 'actuator "linear_axis": the action is required',
        lambda t: _composite(t, **{'actuators.linear_axis.action': F.DELETE})),
    'r8_no_targets': (
        8, 'at least one named target is required',
        lambda t: _composite(t, **{'actuators.linear_axis.targets': F.DELETE})),
    'r8_goal_tolerance_zero': (
        8, 'goal_tolerance [0.0] must be one number > 0',
        lambda t: _composite(t, **{'actuators.linear_axis.goal_tolerance': [0.0]})),
    # fixture P10: an FJT actuator without duration_s, or with one <= 0, is refused
    'r8_no_duration': (
        8, 'duration_s None must be given and > 0',
        lambda t: _composite(t, **{'actuators.linear_axis.duration_s': F.DELETE})),
    'r8_a_duration_of_zero': (
        8, 'duration_s 0.0 must be given and > 0',
        lambda t: _composite(t, **{'actuators.linear_axis.duration_s': 0.0})),
    # M9: with goal_time_tolerance_s 0 a blocked axis hangs
    'r8_goal_time_tolerance_zero': (
        8, 'goal_time_tolerance_s 0.0 must be > 0',
        lambda t: _composite(t, **{'actuators.linear_axis.goal_time_tolerance_s': 0.0})),
    'r8_path_tolerance_zero': (
        8, 'path_tolerance [0.0] must be one number > 0',
        lambda t: _composite(t, **{'actuators.rotary_axis.path_tolerance': [0.0]})),
    # R9 sites
    'r9_a_tcp_outside_the_subtree': (
        9, 'is outside the subtree',
        lambda t: _composite(
            t, **{'sites.site_a.tcp': 'robot_a_base_link'})),
    'r9_no_tcp': (
        9, 'the tcp is required',
        lambda t: F.suction_simple(t, eef_parent_link=None)),
    'r9_an_unknown_engage_target': (
        9, 'engage target \'squeeze\'',
        lambda t: _composite(
            t, **{'sites.site_a.engage': 'squeeze'})),
    'r9_one_actuator_two_sites': (
        9, 'already grips site',
        lambda t: _composite(
            t, **{'sites.site_b.grip': 'gripper_a'})),
    'r9_the_description_cannot_be_read': (
        9, 'the robot description cannot be read',
        lambda t: _top_xacro(F.suction_simple(t), 'missing.urdf.xacro')),
    'r9_an_unknown_grip': (
        9, 'grip \'nope\' is not an actuator of this device',
        lambda t: _composite(t, **{'sites.site_a.grip': 'nope'})),
    'r9_engage_equals_release': (
        9, 'engage and release are both "open"',
        lambda t: _composite(t, **{'sites.site_a.engage': 'open'})),
    # TMR sends SetBool true for "engage" and the adapter records it as ENGAGE: a swapped
    # set_bool site would grip on OpenGripper (the reference normaliser refuses it too)
    'r9_a_set_bool_site_swaps_engage_and_release': (
        9, 'its polarity is the adapter\'s, never the contract\'s',
        lambda t: _long_suction(t, actuators=_VACUUM, sites={
            'main': {'tcp': 'tcp', 'grip': 'vacuum', 'engage': 'release',
                     'release': 'engage'}})),
    'r9_a_tcp_that_is_not_a_link': (
        9, 'tcp "robot_a_nowhere" is not a link of the description',
        lambda t: _composite(t, **{'sites.site_a.tcp': 'robot_a_nowhere'})),
    'r9_a_tcp_on_a_moving_joint': (
        9, 'hangs from the prismatic joint "robot_a_linear_axis_joint"',
        lambda t: _composite(t, **{'sites.site_a.tcp': 'robot_a_carriage'})),
    # R10 IK tip
    'r10_the_tip_below_an_actuator_joint': (
        10, 'MoveIt resolves an IK tip through fixed joints only',
        lambda t: _tip(F.composite(t), 'robot_a_site_a_tcp')),
    # R11 tool actions
    'r11_actuate_without_a_contract': (
        11, 'and this project has none',
        lambda t: _with_actions(
            _legacy_example(), ('pick', 'actuate', {'actuator': 'x', 'target': 'y'}))),
    'r11_actuate_a_grip_engage_target': (
        11, 'use grasp/release',
        lambda t: _with_actions(
            F.jaw_simple(t), ('pick', 'actuate', {'actuator': 'gripper', 'target': 'closed'}))),
    'r11_an_unknown_actuator': (
        11, 'is not an actuator of',
        lambda t: _with_actions(
            F.jaw_simple(t), ('pick', 'actuate', {'actuator': 'axis', 'target': 'home'}))),
    'r11_grasp_without_a_site_on_two_sites': (
        11, 'name the site',
        lambda t: _with_actions(
            F.composite(t), ('place', 'grasp', {}))),
    'r11_actuate_an_unknown_target': (
        11, 'target \'half\' is not a named target of actuator "gripper"',
        lambda t: _with_actions(
            F.jaw_simple(t), ('flip', 'actuate', {'actuator': 'gripper', 'target': 'half'}))),
    'r11_actuate_with_a_site': (
        11, '"site" is for grasp/release',
        lambda t: _with_actions(
            _jaw(t, targets={'open': 0.0, 'closed': 0.7, 'half': 0.35}),
            ('flip', 'actuate', {'actuator': 'gripper', 'target': 'half', 'site': 'main'}))),
    'r11_grasp_with_an_actuator': (
        11, '"actuator"/"target" are for kind actuate',
        lambda t: _with_actions(F.jaw_simple(t), ('flip', 'grasp', {'actuator': 'gripper'}))),
    'r11_attach_with_a_site': (
        11, '"site" is for grasp/release',
        lambda t: _with_actions(
            F.jaw_simple(t), ('flip', 'attach', {'payload_ref': 'part', 'site': 'main'}))),
    'r11_an_unknown_kind_in_memory': (
        11, 'an unknown kind in a canonical project',
        lambda t: _kind_in_memory(F.suction_simple(t), 'teleport')),
    # R12 attach on a multi-site device
    'r12_attach_on_two_sites': (
        12, 'attaches to one link',
        lambda t: _with_actions(
            F.composite(t), ('pick', 'attach', {'payload_ref': 'part'}))),
    'r12_a_payload_on_two_sites': (
        12, 'a scene payload attaches to one link',
        lambda t: _payload(F.composite(t))),
    # R13 bin picking on a multi-site device
    'r13_bin_picking_two_sites': (
        13, 'the emptying loop verifies one grasp',
        lambda t: _app(F.composite(t), AppType.BIN_PICKING)),
    # R14 policy steps
    'r14_a_policy_under_a_namespace': (
        14, 'under the cell namespace "/r3"',
        lambda t: _policy(
            F.suction_simple(t), _card(t))),
    'r14_a_card_that_cannot_be_read': (
        14, 'the card cannot be read',
        lambda t: _policy(
            F.suction_simple(t, ns=''), t / 'nowhere' / 'policy_card.yaml')),
    'r14_a_policy_on_a_composite': (
        14, 'a policy step drives a simple device only',
        lambda t: _policy(
            F.composite(t, ns=''), _card(t))),
    'r14_a_policy_on_a_jaw': (
        14, 'the grip of device "jaw_simple" is gripper_command',
        lambda t: _policy(F.jaw_simple(t, ns=''), _card(t))),
    'r14_two_trigger_pairs': (
        14, 'differs from [\'/suction/on\', \'/suction/off\']: one device serves one pair',
        lambda t: _two_policies(t)),
    # a card Trigger name with a trailing newline is not a ROS name (review TSA-7)
    'r14_a_card_trigger_name_with_a_trailing_newline': (
        14, 'does not resolve at the root to a name the adapter can serve',
        lambda t: _policy(F.suction_simple(t, ns=''), _card(t, on='/suction/on\\n'))),
    # R15 bindings
    'r15_a_deployed_backend_without_a_binding': (
        15, 'the backend is deployed and has no binding',
        lambda t: F.suction_simple(t, modes=('mock', 'isaac'))),
    'r15_a_row_on_the_wrong_backend': (
        15, 'does not run on isaac',
        lambda t: F.suction_simple(
            t, modes=('mock', 'isaac'), contract=F.suction_contract(adapters={
                'mock': 'trainit_mock', 'isaac': 'trainit_mock'}))),
    'r15_none_for_a_verified_device': (
        15, '"none" serves nothing',
        lambda t: F.suction_simple(
            t, modes=('mock', 'real'), contract=F.suction_contract(
                adapters={'mock': 'trainit_mock', 'real': 'none'}, grasp=True))),
    'r15_ros2_control_needs_gripper_cmd': (
        15, 'serves "<controller>/gripper_cmd"',
        lambda t: F.jaw_simple(
            t, modes=('mock', 'real'), action_ns='gripper_command', contract=F.jaw_contract(
                adapters={'mock': 'trainit_mock', 'real': _ros2_control('position')}))),
    'r15_ros2_control_without_a_flavour': (
        15, 'flavour (position or effort) cannot be read',
        lambda t: F.jaw_simple(
            t, modes=('mock', 'real'), contract=F.jaw_contract(adapters={
                'mock': 'trainit_mock', 'real': 'ros2_control'}))),
    'r15_isaac_needs_a_stage_topic': (
        15, 'needs stage_cmd_topic',
        lambda t: F.suction_simple(
            t, modes=('mock', 'isaac'), contract=F.suction_contract(adapters={
                'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'}))),
    'r15_a_key_the_row_does_not_read': (
        15, 'reads no [\'flavour\']',
        lambda t: F.suction_simple(
            t, contract=F.suction_contract(adapters={
                'mock': {'adapter': 'trainit_mock', 'flavour': 'effort'}}))),
    'r15_ros2_control_for_a_binary_jaw': (
        15, '"ros2_control" does not serve set_bool',
        lambda t: F.jaw_binary(
            t, modes=('mock', 'real'), contract=F.binary_jaw_contract(adapters={
                'mock': 'trainit_mock', 'real': _ros2_control('position')}))),
    'r15_ros2_control_for_a_suction': (
        15, '"ros2_control" does not serve kind suction',
        lambda t: F.suction_simple(
            t, modes=('mock', 'real'), contract=F.suction_contract(adapters={
                'mock': 'trainit_mock', 'real': _ros2_control('position')}))),
    'r15_a_declared_flavour_the_base_config_contradicts': (
        15, 'declares flavour "position", and the base config\'s controller is "effort"',
        lambda t: _gazebo_jaw(t, binding=_ros2_control('position'))),
    # "two bindings on one simulator channel" (review TSA-2, tests lens): the adapter would read
    # its own command echo as the stage's status, or collide with the device's own names
    'r15_stage_cmd_and_status_are_one_channel': (
        15, 'stage_cmd_topic and stage_status_topic are one channel ("/suction_cmd")',
        lambda t: _isaac(t, stage_cmd_topic='suction_cmd', stage_status_topic='/suction_cmd')),
    'r15_a_stage_topic_under_the_devices_own_names': (
        15, 'stage_cmd_topic "eef/suction_simple/state" is one of the device\'s own contract',
        lambda t: _isaac(t, stage_cmd_topic='eef/suction_simple/state')),
    # D7: a latched simulator channel is per robot -- global only in a single-robot cell
    'r15_an_absolute_stage_topic_under_a_namespace': (
        15, 'stage_cmd_topic "/suction_cmd" is absolute and outside the cell namespace "/r3"',
        lambda t: _stage(F.suction_simple(
            t, modes=('mock', 'isaac'), contract=F.suction_contract(adapters={
                'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'})))),
    'r15_an_absolute_stage_status_topic_under_a_namespace': (
        15, 'bind the stage\'s status publisher to this robot\'s SurfaceGripper by prim path',
        lambda t: _isaac(t, ns='/r3', stage_cmd_topic='suction_cmd',
                         stage_status_topic='/suction/status')),
    # R16 capabilities and evidence
    'r16_grasp_confirmation_on_isaac_without_status': (
        16, 'does not provide [\'grasp_detection\']',
        lambda t: _stage(
            F.suction_simple(t, ns='', modes=('mock', 'isaac'), contract=F.suction_contract(
                adapters={'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'},
                grasp=True)))),
    'r16_sensor_only_evidence_on_isaac': (
        16, 'nothing it detects would be accepted',
        lambda t: _sensor_only_on_isaac(t)),
    'r16_max_effort_on_a_position_controller': (
        16, 'does not provide [\'effort_control\']',
        lambda t: F.jaw_simple(
            t, modes=('mock', 'real'), contract=dict(F.jaw_contract(adapters={
                'mock': 'trainit_mock', 'real': _ros2_control('position')}),
                actuator=dict(F.jaw_contract()['actuator'], max_effort=10.0)))),
    # fixture P11: actuation_feedback on a binding that does not provide it
    'r16_actuation_feedback_on_isaac': (
        16, 'actuator "vacuum" on isaac: "trainit_isaac_surface_gripper" does not provide '
            '[\'actuation_feedback\']',
        lambda t: _stage(F.with_contract(
            F.suction_simple(t, ns='', modes=('mock', 'isaac'), contract=F.suction_contract(
                adapters={'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'})),
            **{'actuator.actuation_feedback': True}))),
    # R17 verification
    'r17_a_require_key_missing': (
        17, 'verification.require_release_confirmation is required',
        lambda t: F.with_contract(
            F.suction_simple(t), **{'verification.require_release_confirmation': F.DELETE})),
    'r17_a_command_echo_accepted': (
        17, 'never evidence',
        lambda t: F.with_contract(
            F.suction_simple(t), **{'verification.accepted_evidence': ['sensor', 'command_echo']})),
    'r17_a_policy_require_attached_disagrees': (
        17, 'one key decides',
        lambda t: _require_attached(
            _policy(F.suction_simple(t, ns=''), _card(t)))),
    'r17_accepted_evidence_twice': (
        17, 'must be a non-empty list of unique classes',
        lambda t: F.with_contract(
            F.suction_simple(t), **{'verification.accepted_evidence': ['sensor', 'sensor']})),
    'r17_a_verification_timeout_of_zero': (
        17, 'verification.timeout_ms 0 is not a positive integer',
        lambda t: F.with_contract(F.suction_simple(t), **{'verification.timeout_ms': 0})),
    # D4 "one key decides": bin picking's legacy holding fields against the contract
    'r17_bin_picking_holding_check_disagrees': (
        17, 'bin_picking.holding_check is True and device "suction_simple" '
            'require_grasp_confirmation is False',
        lambda t: _bin(F.suction_simple(t, ns=''), holding_check=True)),
    'r17_bin_picking_attached_topic': (
        17, 'bin_picking.attached_topic "/suction/attached"',
        lambda t: _bin(F.suction_simple(t, ns=''), holding_check=False,
                       attached_topic='/suction/attached')),
    'r17_bin_picking_holding_timeout_disagrees': (
        17, 'bin_picking.holding_timeout_ms 5000 disagrees with verification.timeout_ms 3000',
        lambda t: _bin(F.suction_simple(t, ns=''), holding_check=False,
                       holding_timeout_ms=5000)),
    # R18 a legacy end-effector bridge row
    'r18_a_legacy_bridge_row': (
        18, 'is a legacy end-effector bridge',
        lambda t: _bridge(
            F.suction_simple(t), 'fairino_gripper', 'gripper_suction_action_server')),
    # R19 the guard without the guarded block (in memory: assignment skips validation)
    'r19_the_token_without_a_contract': (
        19, 'actuation "contract" without a contract block',
        lambda t: _assign_token(_legacy_example())),
    'r19_schema_3_without_a_contract': (
        19, 'schema_version 3 without a contract block',
        lambda t: _assign_schema(_legacy_example())),
    # R20 backends
    'r20_a_binary_jaw_on_isaac': (
        20, 'serves the suction vacuum only',
        lambda t: _stage(F.jaw_binary(
            t, modes=('mock', 'isaac'), contract=F.binary_jaw_contract(adapters={
                'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'})))),
    'r20_a_composite_on_isaac': (
        20, 'a composite device runs on mock only',
        lambda t: F.composite(
            t, modes=('mock', 'isaac'))),
    # R21 interaction
    'r21_a_contract_that_holds_nothing': (
        21, 'interacts_with_object is false',
        lambda t: F.suction_simple(
            t, interacts_with_object=False)),
    # R22 the composite bundle (vNext M2.7, owner Q1 option A): mock only, served by trainit_mock
    # alone, every tool joint's state on the robot's joint_states -- each refusal named
    'r22_a_composite_beyond_mock': (
        22, 'BUNDLE is generated for mock only (owner Q1, option A)',
        lambda t: F.composite(t, modes=('mock', 'isaac'))),
    'r22_a_composite_bound_to_none_on_mock': (
        22, 'is served by "trainit_mock" alone',
        lambda t: _composite(t, adapters={'mock': 'none'})),
    'r22_a_tool_joint_in_a_ros2_control_block': (
        22, "tool joint(s) ['robot_a_linear_axis_joint'] are in the <ros2_control> block "
            '"ToolSystem"',
        lambda t: _composite_described(t, _TOOL_RC_BLOCK)),
    'r22_the_bringup_description_cannot_be_named': (
        22, 'cannot be named: the base config has no config/robot_a.urdf.xacro',
        lambda t: _base_config(F.composite(t), t)),
    'r22_a_composite_pseudo_joint_echo': (
        22, 'which the adapter would echo as a compatibility pseudo-joint',
        lambda t: _gripper(_composite_described(t, _SPRING), command_joint='robot_a_spring_joint')),
    'r22_a_tool_joint_without_a_state_publisher': (
        22, 'tool joint "robot_a_spring_joint" (revolute) would have no state publisher',
        lambda t: _composite_described(t, _SPRING)),
    # R23 legacy fields that disagree with the contract
    'r23_controller_type': (
        23, 'controller_type "Custom"',
        lambda t: F.suction_simple(
            t, controller_type='Custom')),
    'r23_grasp_action': (
        23, 'grasp_action "suction_on"',
        lambda t: F.suction_simple(
            t, grasp_action='suction_on')),
    'r23_sim_grasp_adapter_on_mock': (
        23, 'sim_grasp_adapter.mock "link_attacher"',
        lambda t: F.suction_simple(
            t, sim_grasp_adapter={'mock': 'link_attacher'})),
    'r23_a_fixed_command_joint': (
        23, 'command_joint "tcp_joint" is a fixed joint',
        lambda t: F.suction_simple(
            t, command_joint='tcp_joint')),
    'r23_legacy_targets_that_disagree': (
        23, 'disagree on',
        lambda t: F.jaw_simple(
            t, joint_target='angle', open_angle=0.1, closed_angle=0.7)),
    'r23_a_controller_name_that_disagrees': (
        23, 'disagree with the contract action',
        lambda t: _jaw(
            t, action='other_controller/gripper_cmd')),
    'r23_a_controller_name_on_a_composite_without_jaws': (
        23, 'controller_name "gripper_a_controller" next to a composite contract',
        lambda t: _gripper(_composite(t, **{'actuators.gripper_a': F.DELETE,
                                            'sites.site_a.grip': 'binary_lock'}),
                           controller_name='gripper_a_controller')),
    'r23_a_command_joint_without_limits': (
        23, 'command_joint "spin_joint" is a continuous joint without limits',
        lambda t: _described(
            F.suction_simple(t, command_joint='spin_joint'), t,
            '<link name="spin_link"/>\n  <joint name="spin_joint" type="continuous">'
            '<parent link="suction_body"/><child link="spin_link"/><axis xyz="0 0 1"/>'
            '</joint>')),
    'r23_legacy_target_fields_next_to_a_suction': (
        23, 'are set next to a contract whose device is not a short-form GripperCommand jaw',
        lambda t: F.suction_simple(t, open_angle=0.1)),
    # R24 / R25 the kinematic path device -> TCP (owner Q7-i E1, E2; vNext M2.7)
    'r24_an_axis_on_set_bool': (
        24, 'actuator "linear_axis", interface expected "follow_joint_trajectory", interface '
            'received "set_bool"',
        lambda t: _composite(t, **{'actuators.linear_axis': {
            'command_interface': 'set_bool', 'joints': ['robot_a_linear_axis_joint'],
            'joint_positions': {'engaged': [0.08], 'released': [0.0]}}})),
    'r25_an_axis_driven_by_nothing': (
        25, 'joint "robot_a_linear_axis_joint" (prismatic) lies on the kinematic path from '
            'eef_parent_link "robot_a_tool0" to the TCP of site "site_a" ("robot_a_site_a_tcp"): '
            'no actuator, interface expected "follow_joint_trajectory", interface received none',
        lambda t: _composite(t, **{'actuators.linear_axis': {
            'command_interface': 'set_bool', 'actuation_feedback': True}})),
    'r25_a_multi_dof_joint_on_the_path': (
        25, 'interface expected "follow_joint_trajectory" on a single-DOF joint, interface '
            'received none',
        lambda t: _composite_described(t, _PLANAR, **{'sites.site_b.tcp': 'robot_a_site_c_tcp'})),
    # G0 (owner section 9): the R24/R25 shapes the M2.7 checkpoint showed as design additions
    'r24_a_multi_dof_joint_driven_on_the_path': (
        24, 'interface expected "follow_joint_trajectory" on a single-DOF joint, interface '
            'received "follow_joint_trajectory"',
        lambda t: _composite_described(t, _PLANAR, **{
            'sites.site_b.tcp': 'robot_a_site_c_tcp', 'actuators.float_axis': _FLOAT_AXIS})),
    'r24_a_jaw_whose_tcp_is_on_its_finger': (
        24, '"gripper" grips site "main", and in the current canonical model a grip never moves '
            'a TCP',
        lambda t: _described(F.with_contract(
            F.jaw_simple(t), actuator=F.DELETE, actuators={'gripper': _FINGER_JAW},
            sites={'main': {'tcp': 'finger_tcp', 'grip': 'gripper'}}), t, _FINGER_TCP,
            tool='jaw_tool')),
    'r25_a_tcp_below_the_command_joint': (
        25, '"suctioncup_joint" is the command_joint pseudo-joint',
        lambda t: _described(_long_suction(
            t, actuators=_VACUUM, sites={'main': {'tcp': 'cup_tcp', 'grip': 'vacuum'}},
            eef_parent_link='suction_body'), t, _CUP_TCP)),
    'r25_a_follower_whose_master_is_missing': (
        25, 'its <mimic> chain does not end at a joint of the description',
        lambda t: _composite_described(t, _DANGLING, **{'sites.site_b.tcp': 'robot_a_site_d_tcp'})),
    # R26 the pseudo-joint echo under a cell namespace (owner 11.1, vNext M2.7)
    'r26_an_absolute_merger_input_under_a_namespace': (
        26, 'robot namespace "/r3", pseudo-joint echo topic "/suction/joint_states" (the '
            '"gripper_joint_states_in" of the joint-state merger row fairino_gripper/'
            'joint_state_merger) is absolute: every robot of the cell would share it -- expected '
            'relative form "suction/joint_states"',
        lambda t: _merger(F.suction_simple(t), '/suction/joint_states')),
}

# A tool joint claimed by a <ros2_control> block of the description (R22, a mixed binding)
_TOOL_RC_BLOCK = ('<ros2_control name="ToolSystem" type="system"><hardware><plugin>'
                  'mock_components/GenericSystem</plugin></hardware><joint '
                  'name="robot_a_linear_axis_joint"><command_interface name="position"/>'
                  '<state_interface name="position"/></joint></ros2_control>')
# A movable side-branch joint no actuator drives (R22: no state publisher)
_SPRING = ('<link name="robot_a_spring_link"/><joint name="robot_a_spring_joint" type="revolute">'
           '<parent link="robot_a_tool_base"/><child link="robot_a_spring_link"/>'
           '<axis xyz="1 0 0"/>'
           '<limit lower="0.0" upper="0.1" effort="1.0" velocity="1.0"/></joint>')
# A planar joint between the wrist and a third TCP (R24: multi-DOF on the path)
_PLANAR = ('<link name="robot_a_float_link"/><joint name="robot_a_float_joint" type="planar">'
           '<parent link="robot_a_wrist_link2"/><child link="robot_a_float_link"/></joint>'
           '<link name="robot_a_site_c_tcp"/><joint name="robot_a_site_c_tcp_joint" type="fixed">'
           '<parent link="robot_a_float_link"/><child link="robot_a_site_c_tcp"/></joint>')


# An FJT actuator on that planar joint (R24: no interface drives a multi-DOF joint)
_FLOAT_AXIS = {'command_interface': 'follow_joint_trajectory',
               'action': 'float_axis_controller/follow_joint_trajectory',
               'joints': ['robot_a_float_joint'], 'targets': {'home': [0.0]},
               'goal_tolerance': [0.01], 'duration_s': 1.0, 'goal_time_tolerance_s': 0.5}
# A TCP on the jaw's own finger link (R24: a grip that moves its own site's TCP)
_FINGER_TCP = ('<link name="finger_tcp"/><joint name="finger_tcp_joint" type="fixed">'
               '<parent link="finger_link"/><child link="finger_tcp"/></joint>')
_FINGER_JAW = {'command_interface': 'gripper_command', 'action': 'gripper_controller/gripper_cmd',
               'joints': ['finger_joint'], 'targets': {'open': 0.0, 'closed': 0.7},
               'result_policy': 'succeeded_or_stalled'}
# A TCP below the suction pseudo-joint (R25: the command_joint echo would move it)
_CUP_TCP = ('<link name="cup_tcp"/><joint name="cup_tcp_joint" type="fixed">'
            '<parent link="suctioncup_link"/><child link="cup_tcp"/></joint>')
# A mimic follower on a TCP path whose master is not in the description (R25: a broken chain)
_DANGLING = ('<link name="robot_a_f_link"/><joint name="robot_a_f_joint" type="prismatic">'
             '<parent link="robot_a_wrist_link2"/><child link="robot_a_f_link"/>'
             '<axis xyz="1 0 0"/><limit lower="0.0" upper="0.1" effort="1.0" velocity="1.0"/>'
             '<mimic joint="robot_a_ghost_joint"/></joint><link name="robot_a_site_d_tcp"/>'
             '<joint name="robot_a_site_d_tcp_joint" type="fixed"><parent link="robot_a_f_link"/>'
             '<child link="robot_a_site_d_tcp"/></joint>')


def _composite_described(tmp, extra: str, **changes):
    """The composite fixture on its own description plus ``extra`` URDF lines."""
    project = F.composite(tmp)
    project.robot.description = F.description(tmp, 'robot_a_x', 'composite_cell.xacro',
                                              '<xacro:composite_cell p="robot_a_"/>\n  ' + extra)
    return F.with_contract(project, **changes) if changes else project


def _merger(project, topic: str, modes=('mock',)):
    """A kept joint-state merger row (catalog role gripper_state_merger) whose gripper input --
    where the adapter's pseudo-joint echo goes (ADR-0022 D6) -- is ``topic``."""
    project.deployment.bridges = list(project.deployment.bridges) + [LaunchNodeSpec(
        package='fairino_gripper', executable='joint_state_merger',
        parameters=[{'gripper_joint_states_in': topic}], modes=list(modes))]
    return project


def _sensor_only_on_isaac(tmp):
    """[sensor]-only evidence, and isaac observes sim_ground_truth (with its status topic)."""
    isaac = {'adapter': 'trainit_isaac_surface_gripper', 'stage_status_topic': 'suction/status'}
    project = F.suction_simple(tmp, ns='', modes=('mock', 'isaac'), contract=F.suction_contract(
        adapters={'mock': 'trainit_mock', 'isaac': isaac}, grasp=True))
    return _stage(F.with_contract(project, **{'verification.accepted_evidence': ['sensor']}))


def _tip(project, link):
    project.robot.tip_link = link
    project.robot.planning_group.tip_link = link
    return project


def _app(project, app_type):
    project.application.type = app_type
    return project


def _stage(project, topic='/suction_cmd'):
    project.scene.gripper_cmd_topic = topic
    return project


def _top_xacro(project, name):
    project.robot.description.top_xacro = name
    return project


def _payload(project):
    from trainit_setup_assistant.model.scene import Payload
    project.scene.payload = Payload(id='part')
    return project


def _require_attached(project):
    project.application.policies[0].require_attached = True
    return project


def _bridge(project, package, executable):
    project.deployment.bridges = list(project.deployment.bridges) + [
        LaunchNodeSpec(package=package, executable=executable, modes=['mock'])]
    return project


def _assign_token(project):
    project.robot.gripper.actuation = EndEffectorActuation.CONTRACT
    return project


def _assign_schema(project):
    project.schema_version = 3
    return project


@pytest.mark.parametrize('case', sorted(CASES))
def test_every_refusal_is_refused_from_both_entry_points(tmp_path, case):
    row, needle, build = CASES[case]
    _refused_from_both(build(tmp_path), tmp_path, row, needle)


def test_every_row_r1_to_r26_has_a_case():
    assert {row for row, _n, _b in CASES.values()} == set(range(1, 27))


def test_every_refusal_site_is_reached_by_a_case(tmp_path):
    """One case per CONDITION, not per row (review TSA-3, tests lens): every place
    ``generator/eef_contract.py`` emits a refusal -- each ``_prefix(N)`` line -- is reached by
    some case of the table above, traced while that case's problems are computed. A refusal
    added without a case, or a case that no longer reaches its line, fails here."""
    source = Path(E.__file__).read_text().splitlines()
    sites = {n for n, line in enumerate(source, 1)
             if re.search(r'_prefix\(\d+\)', line) and not line.lstrip().startswith('def ')}
    assert len(sites) > 100, 'the refusal sites were not found'
    reached = set()

    def trace(frame, event, _arg):
        if frame.f_code.co_filename != E.__file__:
            return None
        if event == 'line':
            reached.add(frame.f_lineno)
        return trace

    for name in sorted(CASES):
        project = CASES[name][2](tmp_path / name)
        previous = sys.gettrace()
        sys.settrace(trace)
        try:
            E.eef_contract_problems(project)
        finally:
            sys.settrace(previous)
    missing = sorted(sites - reached)
    assert not missing, [f'{n}: {source[n - 1].strip()}' for n in missing]


# G0 (owner section 9): the cases whose refusal marks a LIMIT of the current canonical model;
# every other case is a physical or contract invalidity and never says so
LIMITS = frozenset({
    'r22_a_composite_beyond_mock', 'r22_a_composite_bound_to_none_on_mock',
    'r22_a_composite_pseudo_joint_echo', 'r22_a_tool_joint_without_a_state_publisher',
    'r24_a_multi_dof_joint_driven_on_the_path', 'r24_a_jaw_whose_tcp_is_on_its_finger',
    'r25_a_multi_dof_joint_on_the_path', 'r25_a_tcp_below_the_command_joint'})


def test_every_limit_is_a_case():
    assert LIMITS <= set(CASES)


@pytest.mark.parametrize('case', sorted(CASES))
def test_only_a_limit_of_the_current_canonical_model_says_so(tmp_path, case):
    """A refusal that is a limit of the current canonical model says
    UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL, never that the design is invalid; a physical or
    contract invalidity does not carry the token (owner G0 section 9: A and B)."""
    row, needle, build = CASES[case]
    hits = [m for m in E.eef_contract_problems(build(tmp_path))
            if f'(ADR-0022 R{row})' in m and needle in m]
    assert hits, case
    for m in hits:
        assert ('UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in m) == (case in LIMITS), m


def _one_site(project):
    """The composite on one site (site_a): its site_b tool actions dropped."""
    project = F.with_contract(project, **{'sites.site_b': F.DELETE})
    project.application.tool_actions = [a for a in project.application.tool_actions
                                        if a.site != 'site_b']
    return project


def test_r22_the_echo_text_claims_the_adapters_refusal_only_on_several_sites(tmp_path):
    """M2.7 final review FI-5: the hardware adapter refuses a compatibility endpoint on a
    MULTI-site device only (adapter_node: default_site is None); R22's echo sentence says so only
    there, and both devices are still refused (the echo is apart from the remapped joint
    states)."""
    def echo_line(project):
        lines = [m for m in E.eef_contract_problems(project) if '(ADR-0022 R22)' in m
                 and 'which the adapter would echo as a compatibility pseudo-joint' in m]
        assert len(lines) == 1, lines
        return lines[0]
    two = echo_line(_gripper(_composite_described(tmp_path / 'two', _SPRING),
                             command_joint='robot_a_spring_joint'))
    one = echo_line(_gripper(_one_site(_composite_described(tmp_path / 'one', _SPRING)),
                             command_joint='robot_a_spring_joint'))
    assert 'the hardware adapter refuses it at start-up on a device with 2 sites' in two, two
    assert 'refuses it at start-up' not in one, one
    for line in (one, two):
        assert 'the adapter refuses it on a device with' not in line, line      # the M2.7 text
        assert 'apart from the eef/composite_tool/joint_states' in line, line


def test_r22_each_echo_remedy_clears_r22(tmp_path):
    """R22's echo remedies end the refusal: the joint listed in a set_bool actuator's joints with
    joint_positions, or command_joint cleared AND the joint fixed. Clearing command_joint alone
    is the a6 refusal (no state publisher), where the M2.7 remedy "or clear command_joint" led."""
    listed = _gripper(_composite_described(tmp_path / 'a', _SPRING, **{
        'actuators.binary_lock.joints': ['robot_a_spring_joint'],
        'actuators.binary_lock.joint_positions': {'engaged': [0.1], 'released': [0.0]}}),
        command_joint='robot_a_spring_joint')
    fixed = _composite_described(tmp_path / 'b', _SPRING.replace('type="revolute"', 'type="fixed"'))
    for project in (listed, fixed):
        assert [m for m in E.eef_contract_problems(project) if '(ADR-0022 R22)' in m] == []
    cleared = _composite_described(tmp_path / 'c', _SPRING)
    assert any('would have no state publisher' in m for m in E.eef_contract_problems(cleared)
               if '(ADR-0022 R22)' in m)


def test_r24_r25_the_remedy_of_a_limit_clears_it(tmp_path):
    """A TCP on a link the jaw does not move, or above the pseudo-joint, meets no end-effector
    refusal: the remedies the R24 jaw and R25 pseudo-joint sentences give."""
    jaw = F.with_contract(F.jaw_simple(tmp_path / 'jaw'), actuator=F.DELETE,
                          actuators={'gripper': _FINGER_JAW},
                          sites={'main': {'tcp': 'tcp', 'grip': 'gripper'}})
    cup = _long_suction(tmp_path / 'cup', actuators=_VACUUM,
                        sites={'main': {'tcp': 'tcp', 'grip': 'vacuum'}},
                        eef_parent_link='suction_body')
    for project in (jaw, cup):
        assert E.eef_contract_problems(project) == []


def test_an_effort_flavour_read_from_the_base_config_provides_effort_control(tmp_path):
    """Review TSA-1: on gazebo the flavour is read from the base config's controller type when
    the binding declares none (ADR-0022 D2.B), and R7, R15 and R16 read that ONE value -- an
    effort controller with max_effort generates, where R16 used to refuse what R7 asked for."""
    p = _gazebo_jaw(tmp_path / 'effort', max_effort=10.0)
    assert E.eef_contract_problems(p) == []
    b = E.device(p).bindings[Backend.GAZEBO]
    assert (b.flavour, b.effective_flavour) == (None, 'effort')
    assert 'effort_control' in b.provides()
    position = _gazebo_jaw(tmp_path / 'position', flavour='position')
    assert E.eef_contract_problems(position) == []          # no max_effort needed
    assert 'effort_control' not in E.device(position).bindings[Backend.GAZEBO].provides()


@pytest.mark.parametrize('ns, topic', [('/r3', 'suction_cmd'), ('/r3', '/r3/suction_cmd'),
                                       ('', '/suction_cmd'), ('', 'suction_cmd')])
def test_a_stage_topic_of_this_robot_is_accepted(tmp_path, ns, topic):
    """D7: under a cell namespace a stage channel is RELATIVE (or under the namespace), so it is
    this robot's; at the root a global one is a single-robot cell's, and allowed."""
    assert E.eef_contract_problems(_isaac(tmp_path, ns=ns, stage_cmd_topic=topic,
                                          stage_status_topic='suction_status')) == []


def test_r2_an_unknown_key_is_refused_at_load(tmp_path):
    """R2's other half: a project with an unknown key inside the contract never loads -- there
    is no project to generate (``extra='forbid'``; legacy models drop unknown keys silently)."""
    from pydantic import ValidationError
    with pytest.raises(ValidationError, match='extra'):
        F.with_contract(F.suction_simple(tmp_path), **{'actuator.duration_s': 1.0})


def test_a_refusal_names_the_device_actuator_site_and_backend(tmp_path):
    text = '\n'.join(E.eef_contract_problems(F.with_contract(
        F.composite(tmp_path), **{'actuators.linear_axis.goal_tolerance': F.DELETE})))
    assert 'device "composite_tool" actuator "linear_axis"' in text
    stage = '\n'.join(E.eef_contract_problems(_stage(F.suction_simple(
        tmp_path / 'c', ns='', modes=('mock', 'isaac'), contract=F.suction_contract(
            adapters={'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'},
            grasp=True)))))
    assert 'device "suction_simple" site "main" on isaac' in stage


@pytest.mark.parametrize('build', [F.suction_simple, F.jaw_simple, F.jaw_binary],
                         ids=['suction', 'gripper_command_jaw', 'binary_jaw'])
def test_a_valid_simple_contract_generates_from_both_entry_points(tmp_path, build):
    project = build(tmp_path)
    Orchestrator().generate(project, tmp_path / 'bundle')
    generate_scene_loader_config(project, tmp_path / 'step3', 'cell_scene_loader_moveit_config')
    assert (tmp_path / 'bundle' / 'generation_manifest.yaml').is_file()


def test_none_is_allowed_for_a_simple_device_whose_require_keys_are_false(tmp_path):
    """ADR-0022 D6: an existing simple project opts in without changing its modes -- `real`
    bound to none starts no end-effector server (SERVER_MISSING at the first command, as the
    golden's real mode today), and VALID_MODES stays as it was."""
    p = F.suction_simple(tmp_path, ns='', modes=('mock', 'real'), contract=F.suction_contract(
        adapters={'mock': 'trainit_mock', 'real': 'none'}))
    assert E.eef_contract_problems(p) == []
    Orchestrator().generate(p, tmp_path / 'bundle')
    assert Backend.REAL in p.deployment.modes
