"""ADR-0015: per-move ``execution: plan | servo`` — the GATING is the whole test.

The acceptance criterion of ADR-0015 is that ``plan`` emits **zero bytes**: the golden
bundle and the byte-stability snapshots must regenerate unchanged with no regeneration
(``test_generator_golden.py`` / ``test_backend_bytestability.py`` pin that). This file
pins the other half — that ``servo`` emits exactly three things and nothing else:

1. a DIFFERENT BT node, ``<ServoTrack waypoint="…"/>`` instead of ``<MoveWaypoint …/>``
   (explicit in Groot2, and the name ADR-0015 fixes),
2. ``config/servo.yaml`` in the cell MoveIt-config package, every frame/topic/group
   value derived from the cell model,
3. a ``servo_node`` entry in the generated bring-up, with all four
   ``robot_description*`` parameters.

Plus the validation that bounds the combination, and the deliberate ABSENCE of an
``execution:`` key in ``bt_params.yaml`` (the node choice already carries it; a second
copy would be redundant and would put the golden at risk).

Fast and ROS-free: generation is pure.
"""

import ast
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.servo_yaml import build_servo_context
from trainit_setup_assistant.model import (
    AppType, BundleSpec, ExecutionMode, MotionType, VisionBinding, load_project)
from trainit_setup_assistant.model.perception import RelativeBinding
import identity_cases
import ws_paths

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')

# The published golden (base-config / bundle flow): its cell bring-up is where a servo
# node belongs in that flow.
GOLDEN_BUNDLE = ws_paths.GOLDEN_BUNDLE
GOLDEN_PROJECT = GOLDEN_BUNDLE / 'project.yaml'


# --- fixtures -------------------------------------------------------------------

def _fromscratch(servo_on=(), motion=None):
    """The example project (NO base config -> the from-scratch path) with ``execution``
    set to servo on the named waypoints' incoming segments."""
    p = load_project(EXAMPLE)
    p.bundle = BundleSpec.from_prefix('cell')
    for seg in p.application.segments:
        if seg.to_waypoint in servo_on:
            seg.execution = ExecutionMode.SERVO
            if motion is not None:
                seg.motion = motion
            # a streamed move wants a goal that moves: bind it, so the "static taught
            # pose" warning is not what this fixture is testing
            wp = p.application.waypoint_by_name(seg.to_waypoint)
            wp.vision = VisionBinding(detector='capsules')
    return p


def _generate(project):
    tmp = tempfile.TemporaryDirectory()
    Orchestrator().generate(project, tmp.name)
    return tmp, Path(tmp.name)


def _servo_params(path: Path) -> dict:
    """The ``moveit_servo`` block of a generated servo.yaml (the prefix
    ``ServoParameters::makeServoParameters`` declares its parameters under)."""
    return yaml.safe_load(path.read_text())['/**']['ros__parameters']['moveit_servo']


# --- 1. the default emits nothing ----------------------------------------------

def test_execution_defaults_to_plan():
    p = load_project(EXAMPLE)
    assert all(s.execution is ExecutionMode.PLAN for s in p.application.segments)
    assert p.application.uses_servo() is False


def test_all_plan_project_emits_no_servo_artefact():
    p = _fromscratch()
    identity_cases.require('p2_example')
    tmp, out = _generate(p)
    with tmp:
        assert not list(out.rglob('servo.yaml')), 'servo.yaml emitted for an all-plan app'
        tree = (out / 'cell_app' / 'bt_trees' / 'pick_place.xml').read_text()
        assert '<MoveWaypoint waypoint="pick"/>' in tree
        assert 'ServoTrack' not in tree
        bringup = (out / 'cell_app' / 'launch' / 'bringup.launch.py').read_text()
        assert 'moveit_servo' not in bringup and 'servo_node' not in bringup
        for pkg in ('cell_app', 'cell_trainit_config'):
            assert 'moveit_servo' not in (out / pkg / 'package.xml').read_text()


# --- 2. servo emits the node, the yaml and the launch entry --------------------

def test_servo_segment_emits_servotrack_instead_of_movewaypoint():
    p = _fromscratch(servo_on=('pre_pick',))
    xml = get_application(p.application.type).build_tree_xml(p)
    assert '<ServoTrack waypoint="pre_pick"/>' in xml
    assert '<MoveWaypoint waypoint="pre_pick"/>' not in xml
    # every other move is untouched
    assert '<MoveWaypoint waypoint="pick"/>' in xml
    assert xml.count('ServoTrack') == 1


def test_servo_emits_servo_yaml_with_values_from_the_cell_model():
    p = _fromscratch(servo_on=('pre_pick',))
    identity_cases.require('p2_example')
    tmp, out = _generate(p)
    with tmp:
        f = out / 'cell_trainit_config' / 'config' / 'servo.yaml'
        assert f.is_file(), 'servo.yaml not emitted for a servo move'
        s = _servo_params(f)
        r = p.robot
        assert s['move_group_name'] == r.planning_group.name
        assert s['planning_frame'] == r.base_frame
        assert s['robot_link_command_frame'] == r.base_frame
        assert s['ee_frame_name'] == r.tip_link
        assert s['command_out_topic'] == f'/{r.arm_controller.name}/joint_trajectory'
        assert s['joint_topic'] == '/joint_states'
        assert s['monitored_planning_scene_topic'] == '/monitored_planning_scene'
        # the verified settings that make it work against a JointTrajectoryController
        assert s['command_in_type'] == 'speed_units'
        assert s['command_out_type'] == 'trajectory_msgs/JointTrajectory'
        assert s['publish_joint_positions'] is True       # MANDATORY with JTC defaults
        assert s['is_primary_planning_scene_monitor'] is False   # move_group owns it
        assert s['scale'] == {'linear': 0.2, 'rotational': 0.5, 'joint': 0.3}


def test_servo_yaml_follows_a_renamed_group_controller_and_frames():
    """Nothing is hardcoded: rename the cell's parts and the file follows."""
    p = _fromscratch(servo_on=('pre_pick',))
    p.robot.planning_group.name = 'r3_arm'
    p.robot.base_frame = 'r3_base'
    p.robot.tip_link = 'r3_tool0'
    p.robot.arm_controller.name = 'r3_traj_controller'
    ctx = build_servo_context(p)
    assert ctx['move_group_name'] == 'r3_arm'
    assert ctx['planning_frame'] == 'r3_base'
    assert ctx['robot_link_command_frame'] == 'r3_base'
    assert ctx['ee_frame_name'] == 'r3_tool0'
    assert ctx['command_out_topic'] == '/r3_traj_controller/joint_trajectory'


def test_servo_emits_the_launch_entry_with_all_four_descriptions():
    """From-scratch flow: that bring-up IS the cell bring-up, so the node lives there."""
    p = _fromscratch(servo_on=('pre_pick',))
    identity_cases.require('p2_example')
    tmp, out = _generate(p)
    with tmp:
        text = (out / 'cell_app' / 'launch' / 'bringup.launch.py').read_text()
        ast.parse(text)                                   # still valid python
        assert 'package="moveit_servo"' in text
        assert 'executable="servo_node_main"' in text
        assert '"config", "servo.yaml"' in text
        # all four: omitting robot_description hangs then segfaults; omitting
        # robot_description_planning (the joint limits) makes servo emit NaN
        for param in ('mc.robot_description', 'mc.robot_description_semantic',
                      'mc.robot_description_kinematics', 'mc.joint_limits'):
            assert param in text, f'servo_node is missing {param}'
        # use_sim_time comes from the backend profile, like every other node
        servo_call = text[text.index('package="moveit_servo"'):]
        assert 'common' in servo_call[:servo_call.index('))')]
        # and the package that runs it declares it
        assert '<exec_depend>moveit_servo</exec_depend>' in (
            out / 'cell_app' / 'package.xml').read_text()
        assert '<exec_depend>moveit_servo</exec_depend>' in (
            out / 'cell_trainit_config' / 'package.xml').read_text()


@pytest.mark.skipif(not GOLDEN_PROJECT.is_file(), reason='golden bundle not present')
def test_bundle_flow_puts_servo_in_the_cell_bringup():
    """Base-config flow: the cell bring-up (the "engine bay") owns the servo node, so it
    comes up with move_group and the tree's ServoTrack finds its services."""
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix('cell')
    seg = p.application.segments[1]
    seg.execution = ExecutionMode.SERVO
    seg.motion = MotionType.FREE
    tmp, out = _generate(p)
    with tmp:
        cfg = out / 'cell_trainit_config'
        assert (cfg / 'config' / 'servo.yaml').is_file()
        cell = (cfg / 'launch' / 'bringup.launch.py').read_text()
        ast.parse(cell)
        assert 'package="moveit_servo"' in cell
        assert 'executable="servo_node_main"' in cell
        assert '<exec_depend>moveit_servo</exec_depend>' in (cfg / 'package.xml').read_text()
        # the APP bring-up only includes the cell one — no second servo node
        app = (out / 'cell_app' / 'launch' / 'bringup.launch.py').read_text()
        assert 'moveit_servo' not in app


# --- 3. bt_params carries NO second copy of the decision -----------------------

def test_bt_params_has_no_execution_key():
    """Decision (iii): the NODE carries the choice. A duplicate key in bt_params would be
    redundant and would put the byte-stable golden at risk."""
    identity_cases.require('p2_example')
    for project in (_fromscratch(), _fromscratch(servo_on=('pre_pick',))):
        tmp, out = _generate(project)
        with tmp:
            text = (out / 'cell_app' / 'config' / 'bt_params.yaml').read_text()
            assert 'execution' not in text
            params = yaml.safe_load(text)['/**']['ros__parameters']
            assert 'execution' not in params
            for block in params['task_parameters'].values():
                if isinstance(block, dict):
                    assert 'execution' not in block


# --- 4. validation -------------------------------------------------------------

def test_servo_with_lin_is_refused():
    p = _fromscratch(servo_on=('pre_pick',), motion=MotionType.LIN)
    problems = get_application(p.application.type).validate(p)
    text = '\n'.join(problems)
    assert 'execution "servo" cannot be combined with motion "lin"' in text
    assert 'MoveIt Servo has no Pilz LIN' in text


def test_servo_with_lin_REFUSES_generation_before_a_file_exists(tmp_path):
    """vNext M0C #8. ADR-0015 says REFUSED; validation only warned, and the bundle was
    generated with one of the two choices silently ignored. Now it is refused like an
    uncomputable candidate rule: before any file exists, with the same message."""
    p = _fromscratch(servo_on=('pre_pick',), motion=MotionType.LIN)
    out = tmp_path / 'bundle'
    with pytest.raises(ValueError, match='cannot be combined with motion "lin"'):
        Orchestrator().generate(p, str(out))
    assert not out.exists()


def test_servo_on_a_static_taught_pose_is_only_a_warning(tmp_path):
    p = load_project(EXAMPLE)
    p.application.segment_for('pre_pick').execution = ExecutionMode.SERVO
    identity_cases.require('p2_example')
    Orchestrator().generate(p, str(tmp_path / 'bundle'))      # warned, not refused


def test_servo_with_circ_is_refused():
    p = _fromscratch(servo_on=('pre_pick',), motion=MotionType.CIRC)
    text = '\n'.join(get_application(p.application.type).validate(p))
    assert 'execution "servo" cannot be combined with motion "circ"' in text


def test_servo_on_a_static_taught_pose_warns():
    p = load_project(EXAMPLE)
    seg = p.application.segment_for('pre_pick')
    seg.execution = ExecutionMode.SERVO                  # motion is already 'free'
    text = '\n'.join(get_application(p.application.type).validate(p))
    assert 'execution "servo" on a static taught pose' in text
    assert 'buys nothing over a planned one' in text


def test_servo_on_a_bound_waypoint_is_accepted():
    """Vision-bound: clean. Step-relative: clean too (the goal still moves)."""
    p = _fromscratch(servo_on=('pre_pick',))
    text = '\n'.join(get_application(p.application.type).validate(p))
    assert 'servo' not in text

    p2 = load_project(EXAMPLE)
    seg = p2.application.segment_for('pre_pick')
    seg.execution = ExecutionMode.SERVO
    p2.application.waypoint_by_name('pre_pick').relative = RelativeBinding(step='home')
    text2 = '\n'.join(get_application(p2.application.type).validate(p2))
    assert 'servo' not in text2


def test_execution_validation_is_inherited_by_every_application_template():
    """The checks are app-AGNOSTIC, so they live in the shared base: the vision template
    reports them without re-deriving them."""
    p = _fromscratch(servo_on=('pre_pick',), motion=MotionType.LIN)
    p.application.type = AppType.VISION_GUIDED_MOTION
    text = '\n'.join(get_application(AppType.VISION_GUIDED_MOTION).validate(p))
    assert 'execution "servo" cannot be combined with motion "lin"' in text


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
