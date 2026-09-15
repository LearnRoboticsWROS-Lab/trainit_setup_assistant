"""End-effector actuation in the generator (ADR-0010 F3): the bt_params actuation block is
emitted ONLY for JOINT_POSITION (a TRIGGER cell — the deterministic golden — renders none of
it, so it stays byte-identical), with the close/open GripperCommand targets resolved from an
explicit ANGLE or from an SRDF named state. ROS-free: generation is pure."""

import pytest

from trainit_setup_assistant.applications.base import (
    build_bt_params_context, resolve_gripper_positions)
from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.model import (
    AppType, BundleSpec, CanonicalProject, MotionSegment, NamedState, ToolAction, Waypoint)
from trainit_setup_assistant.model.enums import (
    EndEffectorActuation, GripperJointTarget, GripperKind, ToolActionKind, WaypointType)
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, PlanningGroupSpec, RobotSpec)


def _project(gripper: GripperSpec, named_states=None) -> CanonicalProject:
    p = CanonicalProject(
        project_name='cell',
        bundle=BundleSpec.from_prefix('cell'),
        robot=RobotSpec(
            robot_name='cell',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            planning_group=PlanningGroupSpec(name='arm', joints=['j1', 'j2']),
            gripper=gripper,
            named_states=named_states or [],
            arm_controller=ArmControllerSpec(),
        ),
    )
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='pick', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='pick')]
    p.application.tool_actions = [
        ToolAction(at_waypoint='pick', kind=ToolActionKind.GRASP),
        ToolAction(at_waypoint='pick', kind=ToolActionKind.RELEASE)]
    p.application.sequence = ['pick']
    return p


def _render(project: CanonicalProject) -> str:
    ctx = build_bt_params_context(project)
    return build_jinja_env().get_template('app/bt_params.yaml.j2').render(**ctx)


# --- TRIGGER (suction / the golden): no actuation block ----------------------

def test_trigger_emits_no_actuation_block():
    text = _render(_project(GripperSpec(kind=GripperKind.SUCTION)))
    assert 'gripper_close_position' not in text
    assert 'gripper_open_position' not in text
    assert 'gripper_cmd_topic' not in text


# --- JOINT_POSITION via explicit ANGLE ---------------------------------------

def test_joint_position_angle_emits_resolved_targets():
    g = GripperSpec(kind=GripperKind.PARALLEL,          # defaults actuation -> joint_position
                    command_joint='robotiq_85_left_knuckle_joint',
                    controller_name='gripper_position_controller', action_ns='gripper_cmd',
                    joint_target=GripperJointTarget.ANGLE, open_angle=0.0, closed_angle=0.7691)
    assert resolve_gripper_positions(_project(g).robot) == (0.7691, 0.0)  # (close, open)
    p = _project(g)
    p.scene.gripper_cmd_topic = '/grasp_cmd'          # a wired sim adapter sets the grasp signal
    text = _render(p)
    assert 'gripper_close_position: 0.7691' in text
    assert 'gripper_open_position: 0.0' in text
    assert 'gripper_cmd_topic: "/grasp_cmd"' in text


def test_joint_position_blank_topic_omits_gripper_cmd():
    # no sim grasp adapter wired => scene.gripper_cmd_topic is '' (the new default) => the
    # actuation is still emitted, but no grasp signal line (the runtime fires no adapter).
    g = GripperSpec(kind=GripperKind.PARALLEL, joint_target=GripperJointTarget.ANGLE,
                    open_angle=0.0, closed_angle=0.7691)
    text = _render(_project(g))
    assert 'gripper_close_position: 0.7691' in text
    assert 'gripper_cmd_topic: "' not in text        # the param line (not the comment) is omitted


# --- JOINT_POSITION via SRDF named state -------------------------------------

def test_joint_position_srdf_state_resolves_from_named_states():
    cj = 'robotiq_85_left_knuckle_joint'
    g = GripperSpec(kind=GripperKind.PARALLEL, command_joint=cj,
                    joint_target=GripperJointTarget.SRDF_STATE,
                    open_state='open', closed_state='closed')
    states = [NamedState(name='open', group='gripper', joint_values={cj: 0.0}),
              NamedState(name='closed', group='gripper', joint_values={cj: 0.8})]
    assert resolve_gripper_positions(_project(g, states).robot) == (0.8, 0.0)
    text = _render(_project(g, states))
    assert 'gripper_close_position: 0.8' in text
    assert 'gripper_open_position: 0.0' in text


def test_srdf_state_unresolvable_falls_back_to_trigger_defaults():
    # no named_states carrying the command joint -> fall back to 1.0/0.0 (never fail codegen)
    g = GripperSpec(kind=GripperKind.PARALLEL, command_joint='missing_joint',
                    joint_target=GripperJointTarget.SRDF_STATE,
                    open_state='open', closed_state='closed')
    assert resolve_gripper_positions(_project(g).robot) == (1.0, 0.0)


# --- explicit TRIGGER on a parallel jaw keeps the block out ------------------

def test_explicit_trigger_on_parallel_emits_no_block():
    g = GripperSpec(kind=GripperKind.PARALLEL, actuation=EndEffectorActuation.TRIGGER)
    assert 'gripper_close_position' not in _render(_project(g))
