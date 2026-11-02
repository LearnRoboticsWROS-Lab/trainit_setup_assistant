"""vNext I0.3 (I0_DESIGN.md §14): waypoints visible, editable, teachable, in an engineer's units.

- An untouched value round-trips bit for bit: the quaternion, the metres and the radians as stored.
- An edited orientation is stored as a clean unit quaternion.
- The panel edits only through the Composer, so every refusal is the canonical model's.
"""
import math
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import Composer, pose
from trainit_setup_assistant.model import CanonicalProject, project_to_dict

PKG = Path(__file__).resolve().parents[1]
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'
TOOL_DOWN = [-0.7071067811865476, 0.7071067811865476, 0.0, 0.0]       # the M3 acceptance cells' orientation


def project() -> CanonicalProject:
    return CanonicalProject.parse_obj(yaml.safe_load(COMPOSITION.read_text()))


# ---- pose math -----------------------------------------------------------------------------------------

@pytest.mark.parametrize('rpy', [(0, 0, 0), (180, 0, 90), (10, -20, 30), (-90, 45, -135), (0, 89.5, 0)])
def test_rpy_and_quaternion_are_inverse(rpy):
    q = pose.rpy_to_quat(*(math.radians(a) for a in rpy))
    assert abs(sum(v * v for v in q) - 1.0) < 1e-12
    back = pose.rpy_deg(q)
    q2 = pose.rpy_to_quat(*(math.radians(a) for a in back))
    assert all(abs(a - b) < 1e-6 for a, b in zip(q, q2))


def test_the_tool_down_orientation_reads_as_an_engineer_expects():
    assert pose.rpy_deg(TOOL_DOWN) == [180.0, 0.0, -90.0] or pose.rpy_deg(TOOL_DOWN) == [-180.0, 0.0, -90.0]


def test_an_untouched_orientation_keeps_the_stored_quaternion_bit_for_bit():
    shown = pose.rpy_deg(TOOL_DOWN)
    assert pose.edited_orientation(TOOL_DOWN, shown, list(shown)) is not TOOL_DOWN
    assert pose.edited_orientation(TOOL_DOWN, shown, list(shown)) == TOOL_DOWN
    edited = pose.edited_orientation(TOOL_DOWN, shown, [shown[0], shown[1], shown[2] + 10.0])
    assert edited != TOOL_DOWN and abs(sum(v * v for v in edited) - 1.0) < 1e-8 and edited[3] >= 0


def test_an_untouched_position_and_joints_keep_their_stored_values():
    p = [0.42, -0.25, 0.22000000000000003]
    assert pose.edited_position(p, pose.mm(p), pose.mm(p)) == p
    assert pose.edited_position(p, pose.mm(p), [420.0, -250.0, 230.0]) == [0.42, -0.25, 0.23]
    j = [0.0, -2.0602, 1.2497]
    assert pose.edited_joints(j, pose.joints_deg(j), pose.joints_deg(j)) == j
    assert pose.edited_joints(j, pose.joints_deg(j), [0.0, -118.04, 90.0])[2] == round(math.pi / 2, 6)


def test_a_captured_pose_is_written_cleanly():
    wp = pose.captured_tcp_waypoint('w', [0.4200000371, -0.25, 0.22], [0.7071068, -0.7071068, 0.0, -1e-9])
    assert wp['position'] == [0.42, -0.25, 0.22]
    assert wp['orientation'][3] >= 0 and abs(sum(v * v for v in wp['orientation']) - 1.0) < 1e-8
    j = pose.captured_joint_waypoint('h', [0.1234567891, -1.5])
    assert j == {'name': 'h', 'type': 'joint', 'joints': [0.123457, -1.5]}


@pytest.mark.parametrize('wp, text', [
    ({'name': 'w1', 'position': [0.42, -0.25, 0.22], 'orientation': TOOL_DOWN}, 'XYZ (420, -250, 220) mm'),
    ({'name': 'h', 'type': 'joint', 'named': 'home2'}, "named state 'home2'"),
    ({'name': 'j', 'type': 'joint', 'joints': [0.0, math.pi / 2]}, 'joints (0, 90)°'),
    ({'name': 'w2', 'relative': {'to': 'w1', 'frame': 'tool', 'dz': 0.08}}, "relative to 'w1' along tool: dz 80 mm"),
])
def test_every_kind_has_a_readable_summary(wp, text):
    assert text in pose.summary(wp)


# ---- the panel (Qt offscreen) --------------------------------------------------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel(c):
    from trainit_setup_assistant.gui.composer.waypoints_panel import WaypointsPanel
    panel = WaypointsPanel(lambda: c)
    panel.refresh()
    return panel


def test_the_table_lists_every_waypoint_with_its_users(qapp):
    c = Composer(project())
    panel = _panel(c)
    assert panel.table.rowCount() == len(c.program.waypoints)
    for i, wp in enumerate(c.document()['waypoints']):
        assert panel.table.item(i, 0).text() == wp['name']
        assert panel.table.item(i, 2).text() == pose.summary(wp)
        users = c.waypoint_users(wp['name'])
        assert panel.table.item(i, 3).text() == (', '.join(users) or '—')


def test_apply_without_an_edit_changes_nothing(qapp):
    c = Composer(project())
    before = project_to_dict(c.project)
    panel = _panel(c)
    for wp in c.document()['waypoints']:
        panel.select(wp['name'])
        assert panel.apply()
    assert project_to_dict(c.project) == before and not c.can_undo


def test_editing_a_tcp_waypoint_in_mm_and_degrees(qapp):
    c = Composer(project())
    panel = _panel(c)
    name = panel.new_waypoint('TCP', 'taught')
    panel.select(name)
    panel.xyz[2].setValue(255.5)
    panel.rpy[2].setValue(45.0)
    assert panel.apply()
    wp = next(w for w in c.document()['waypoints'] if w['name'] == 'taught')
    assert wp['position'] == [0.4, 0.0, 0.2555]
    assert pose.rpy_deg(wp['orientation'])[2] == 45.0
    c.undo()
    assert next(w for w in c.document()['waypoints'] if w['name'] == 'taught')['position'] == [0.4, 0.0, 0.3]


def test_joints_named_and_relative_waypoints_are_edited_by_kind(qapp):
    c = Composer(project())
    panel = _panel(c)
    joints = c.project.robot.planning_group.joints
    name = panel.new_waypoint('Joints')
    panel.select(name)
    assert len(panel.joint_spins) == len(joints)
    panel.joint_spins[1].setValue(-90.0)
    assert panel.apply()
    wp = next(w for w in c.document()['waypoints'] if w['name'] == name)
    assert wp['joints'][1] == round(-math.pi / 2, 6) and wp['type'] == 'joint'
    rel = panel.new_waypoint('Relative')
    panel.select(rel)
    panel.offsets[2].setValue(-80.0)
    assert panel.apply()
    wp = next(w for w in c.document()['waypoints'] if w['name'] == rel)
    assert wp['relative']['dz'] == -0.08
    if c.project.robot.named_states:
        named = panel.new_waypoint('Named')
        assert next(w for w in c.document()['waypoints'] if w['name'] == named)['named'] == \
            c.project.robot.named_states[0].name


def test_rename_and_delete_go_through_the_model(qapp):
    c = Composer(project())
    panel = _panel(c)
    used = next(w for w in c.waypoint_names() if c.waypoint_users(w))
    assert not panel.delete(used)
    assert 'is used by' in panel.message.text()
    assert panel.rename(used, 'renamed_point')
    assert c.waypoint_users('renamed_point') and panel.selected == 'renamed_point'
    assert not panel.rename('renamed_point', '9bad name')                 # the model refuses the name
    assert 'renamed_point' in c.waypoint_names()


def test_the_composer_edits_the_controllers_one_project_and_the_program_text_follows(qapp):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(COMPOSITION)
    w = SetupWizard(ctrl)
    blocks = w.blocks_page
    blocks.initializePage()
    assert blocks.title() == 'Step 7 — Application Composer'
    shell = blocks.program_panel
    shell.select('Waypoints')
    panel = shell.waypoints
    assert panel.table.rowCount() == len(ctrl.project.application.program.waypoints)
    before = ctrl.project
    name = panel.new_waypoint('TCP', 'from_the_composer')
    assert ctrl.project is not before and ctrl.project is shell.composer().project
    assert name in [wp.name for wp in ctrl.project.application.program.waypoints]
    assert ctrl.validate_program()[0] == []                     # an unused waypoint is valid
    assert shell.undo_button.isEnabled() and not shell.redo_button.isEnabled()
    shell.undo()
    assert ctrl.project is before and shell.redo_button.isEnabled()
    assert 'from_the_composer' not in blocks.program_view.toPlainText()
    shell.redo()
    assert name in [wp.name for wp in ctrl.project.application.program.waypoints]
