"""M0: the canonical model loads, validates, and round-trips through YAML."""

import os

import pytest

from trainit_setup_assistant.model import (
    AppType,
    GripperKind,
    MotionType,
    WaypointType,
    load_project,
    project_to_dict,
)
from trainit_setup_assistant.model.io import CanonicalProject

EXAMPLE = os.path.join(
    os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml'
)


def test_fr3wml_example_loads():
    project = load_project(EXAMPLE)
    assert project.project_name == 'fr3wml_app_trainit_config'
    assert project.robot.robot_name == 'fr3wml'
    assert project.robot.planning_group.joints == ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    assert project.robot.gripper.kind is GripperKind.SUCTION
    assert project.application.type is AppType.PICK_AND_PLACE
    assert len(project.application.waypoints) == 7
    assert project.application.sequence[0] == 'home'
    assert project.application.sequence[-1] == 'home'


def test_segments_cover_all_waypoints():
    project = load_project(EXAMPLE)
    names = {wp.name for wp in project.application.waypoints}
    seg_targets = {s.to_waypoint for s in project.application.segments}
    assert names == seg_targets


def test_motion_and_type_enums_parse():
    project = load_project(EXAMPLE)
    pick = project.application.waypoint_by_name('pick')
    assert pick.type is WaypointType.TCP
    assert project.application.segment_for('pick').motion is MotionType.LIN
    assert project.application.segment_for('pick').speed == 30


def test_roundtrip_yaml_dict_revalidates():
    project = load_project(EXAMPLE)
    data = project_to_dict(project)
    again = CanonicalProject.parse_obj(data)
    assert again == project


def test_gripper_action_namespace():
    project = load_project(EXAMPLE)
    assert project.robot.gripper.gripper_action_ns() == '/suctioncup_controller/gripper_command'


def test_waypoint_allowed_start_tolerance_default_and_roundtrip():
    from trainit_setup_assistant.model.task import Waypoint
    assert Waypoint(name='home').allowed_start_tolerance == 0.1        # default 0.1
    wp = Waypoint(name='approach', named='approach_prewash', allowed_start_tolerance=0.0)
    assert wp.allowed_start_tolerance == 0.0                           # 0.0 disables
    assert Waypoint.parse_obj(wp.dict()) == wp              # round-trips (v1/v2-overlap)


def test_add_move_sets_per_waypoint_start_tolerance():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.add_move('appr', named='approach_prewash', motion='ptp',
                  allowed_start_tolerance=0.25)
    assert ctrl.project.application.waypoint_by_name('appr').allowed_start_tolerance == 0.25


def test_segment_execution_default_and_roundtrip():
    """ADR-0015: non-optional with a PLAN default (mirrors ``motion`` — the semantics are
    absolute, not inherit-from-global), and it survives YAML."""
    from trainit_setup_assistant.model import ExecutionMode
    from trainit_setup_assistant.model.task import MotionSegment
    assert MotionSegment(to_waypoint='pick').execution is ExecutionMode.PLAN
    servo = MotionSegment(to_waypoint='pick', execution='servo')
    assert servo.execution is ExecutionMode.SERVO
    assert MotionSegment.parse_obj(servo.dict()) == servo   # v1/v2-overlap round-trip


def test_project_roundtrip_keeps_a_servo_segment():
    from trainit_setup_assistant.model import ExecutionMode
    project = load_project(EXAMPLE)
    project.application.segment_for('pre_pick').execution = ExecutionMode.SERVO
    again = CanonicalProject.parse_obj(project_to_dict(project))
    assert again == project
    assert again.application.segment_for('pre_pick').execution is ExecutionMode.SERVO
    assert again.application.uses_servo() is True


def test_add_move_sets_per_move_execution():
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.model import ExecutionMode
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.add_move('track', named='home', motion='free', execution='servo')
    assert ctrl.project.application.segment_for('track').execution is ExecutionMode.SERVO
    # the default is unchanged for every other caller (no global to inherit)
    ctrl.add_move('planned', named='home', motion='free')
    assert ctrl.project.application.segment_for('planned').execution is ExecutionMode.PLAN


# --- ADR-0016: the robot slot + the emptying loop ------------------------------

def test_cell_namespace_default_and_roundtrip():
    """ADR-0016 Decision 3: empty by DEFAULT, because empty is what one robot wants and
    what every bundle generated before the field existed."""
    project = load_project(EXAMPLE)
    assert project.robot.cell_namespace == ''
    assert project.robot.namespace() == ''
    project.robot.cell_namespace = 'r3'                 # canonicalised on assignment
    assert project.robot.cell_namespace == '/r3'
    again = CanonicalProject.parse_obj(project_to_dict(project))
    assert again == project
    assert again.robot.cell_namespace == '/r3'


def test_cell_namespace_rejects_a_name_ros_cannot_use():
    """Not cosmetic: an illegal namespace yields a bring-up whose nodes never find each
    other, and nothing in ROS says so at run time."""
    from trainit_setup_assistant.model.robot import RobotSpec
    project = load_project(EXAMPLE)
    with pytest.raises(Exception):
        project.robot.cell_namespace = '3r'
    assert project.robot.cell_namespace == ''           # the refusal leaves it untouched
    assert RobotSpec.parse_obj({**project.robot.dict(),
                                'cell_namespace': 'cell/r3'}).cell_namespace == '/cell/r3'


def test_bin_picking_app_type_is_a_frozen_token():
    assert AppType.BIN_PICKING.value == 'bin_picking'
    assert AppType('bin_picking') is AppType.BIN_PICKING


def test_bin_picking_spec_defaults_and_roundtrip():
    """ADR-0016 Decision 4: the loop's knobs are CONFIGURATION with measured defaults —
    3 tries (the reference cell needed 3,33 candidates per object), the holding check on
    (the gripper and the object disagree on 1 lift in 22) and no recovery channel (the
    machine behind it is the cell's, ADR-0013)."""
    from trainit_setup_assistant.model import BinPickingSpec
    cfg = BinPickingSpec()
    assert (cfg.tries, cfg.holding_check, cfg.attached_topic) == (3, True, '')
    assert (cfg.recovery_channel, cfg.recovery_dwell_ms) == ('', 1500)
    project = load_project(EXAMPLE)
    assert project.application.bin_picking == cfg       # present and inert by default
    project.application.bin_picking.tries = 5
    project.application.bin_picking.recovery_channel = 'shaker'
    again = CanonicalProject.parse_obj(project_to_dict(project))
    assert again == project
    assert again.application.bin_picking.tries == 5


def test_bin_picking_spec_refuses_a_zero_retry_budget():
    """tries = 0 is not "no retry", it is a loop that tries nothing."""
    from trainit_setup_assistant.model import BinPickingSpec
    with pytest.raises(Exception):
        BinPickingSpec(tries=0)


def test_controller_sets_the_namespace_and_the_loop():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.set_cell_namespace('r3')
    assert ctrl.cell_namespace() == '/r3'
    ctrl.set_cell_namespace('')                         # back to one robot, at the root
    assert ctrl.cell_namespace() == ''
    ctrl.set_bin_picking(tries=4, recovery_channel=' bin_agitator ',
                         attached_topic=' /suction/attached ')
    cfg = ctrl.project.application.bin_picking
    assert (cfg.tries, cfg.recovery_channel) == (4, 'bin_agitator')
    assert cfg.attached_topic == '/suction/attached'
    assert cfg.holding_check is True                    # keep-or-set: untouched


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
