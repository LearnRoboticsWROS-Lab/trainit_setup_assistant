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
    again = CanonicalProject.model_validate(data)
    assert again == project


def test_gripper_action_namespace():
    project = load_project(EXAMPLE)
    assert project.robot.gripper.gripper_action_ns() == '/suctioncup_controller/gripper_command'


def test_waypoint_allowed_start_tolerance_default_and_roundtrip():
    from trainit_setup_assistant.model.task import Waypoint
    assert Waypoint(name='home').allowed_start_tolerance == 0.1        # default 0.1
    wp = Waypoint(name='approach', named='approach_prewash', allowed_start_tolerance=0.0)
    assert wp.allowed_start_tolerance == 0.0                           # 0.0 disables
    assert Waypoint.model_validate(wp.model_dump()) == wp              # round-trips


def test_add_move_sets_per_waypoint_start_tolerance():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.add_move('appr', named='approach_prewash', motion='ptp',
                  allowed_start_tolerance=0.25)
    assert ctrl.project.application.waypoint_by_name('appr').allowed_start_tolerance == 0.25


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
