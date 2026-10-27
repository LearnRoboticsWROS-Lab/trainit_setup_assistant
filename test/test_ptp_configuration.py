"""A PTP to a pose that keeps its joint configuration (ADR-0028, KNOWN_DEBT row 24 (b); TSA 8.2.0).

A TCP (or relative) waypoint may name the joint waypoint it was taught with: `configuration: w1_j`. TSA
resolves it into the bundle's entry (`configuration: [rad...]`, also on a grasp's / release's PreGrasp /
PreRelease at that target); TMR 2.6.0 reaches the pose in the configuration nearest it, or refuses.
"Capture TCP + joints" records it; the live check predicts the runtime's verdict; R-I0-05 no longer flags such
a pose. A program that names no configuration generates byte-identically (the golden gates cover it).
"""

import copy
import math
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import ProgramRefusals, plan_program
from trainit_setup_assistant.composer import Composer, pose
from trainit_setup_assistant.composer.advice import configuration_advice
from trainit_setup_assistant.composer.templates import apply_template
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.io import project_to_dict

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'
W1_J = [0.7, -1.6, -1.7, -1.4, 1.57, -0.86]
DOWN = [1.0, 0.0, 0.0, 0.0]


@pytest.fixture(autouse=True)
def _sample_paths(monkeypatch):
    monkeypatch.chdir(SAMPLE.parent)


@pytest.fixture
def qapp():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def taught(configuration='w1_j') -> Composer:
    """The simple-motion template, w1 taught WITH its configuration and reached by a PTP to the POSE."""
    c = Composer(apply_template(CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text())), 'simple_motion'))
    w1 = pose.captured_tcp_waypoint('w1', [0.45, 0.25, 0.3], DOWN)
    if configuration:
        w1['configuration'] = configuration
    c.set_waypoints([w1, pose.captured_joint_waypoint('w1_j', W1_J)])
    c.edit(lambda d: [s['move'].__setitem__('to', 'w1') for s in d['tasks'][0]['steps']
                      if 'move' in s and s['move']['to'] == 'w1_j'])
    return c


def entries(project) -> dict:
    return plan_program(project).entries


def test_the_bundle_entry_carries_the_configuration_a_pose_names():
    e = entries(taught().project)
    assert e['w1']['configuration'] == W1_J
    assert list(e['w1'])[:4] == ['type', 'position', 'orientation', 'configuration']
    assert 'configuration' not in e['w4']                    # a pose without one: unchanged


def test_a_relative_pose_and_a_named_state_configuration():
    c = taught(configuration='home')                         # a named state of the planning group
    home = {s.name: s for s in c.project.robot.named_states}['home'].joint_values
    joints = list(c.project.robot.planning_group.joints)
    assert entries(c.project)['w1']['configuration'] == [home[j] for j in joints]
    c.edit(lambda d: next(w for w in d['waypoints'] if w['name'] == 'w2').__setitem__('configuration', 'w1_j'))
    assert entries(c.project)['w2']['configuration'] == W1_J


@pytest.mark.parametrize('change, needle', [
    (lambda d: d['waypoints'][1].__setitem__('configuration', 'nowhere'), "R-M3-03: waypoint 'w1' configuration "
                                                                           "'nowhere': no waypoint is named"),
    (lambda d: d['waypoints'][1].__setitem__('configuration', 'w3'), "R-M3-05: waypoint 'w1' configuration 'w3': "
                                                                      "'w3' is a tcp target"),
    (lambda d: next(w for w in d['waypoints'] if w['name'] == 'w1_j').__setitem__('joints', [0.0, 0.0]),
     "R-M3-05: waypoint 'w1' configuration 'w1_j': 'w1_j' has 2 joint value(s)"),
])
def test_a_configuration_that_is_not_a_joint_waypoint_of_the_group_is_refused(change, needle):
    data = project_to_dict(taught().project)
    change(data['application'])
    with pytest.raises(ProgramRefusals, match=needle.replace('(', r'\(').replace(')', r'\)')):
        plan_program(CanonicalProject.parse_obj(data))


def test_a_joint_waypoint_has_no_configuration_of_its_own():
    data = project_to_dict(taught().project)
    next(w for w in data['application']['waypoints'] if w['name'] == 'w1_j')['configuration'] = 'w1_j'
    with pytest.raises(ValueError, match="R-M3-02: waypoint 'w1_j' is a joint target: configuration belong"):
        CanonicalProject.parse_obj(data)


def test_a_grasp_at_a_pose_with_a_configuration_passes_it_to_its_pregrasp():
    c = taught()
    c.edit(lambda d: d['tasks'][0]['steps'].insert(1, {'grasp': {
        'at': 'w1', 'standoff_m': 0.1, 'profile': 'transit', 'approach_profile': 'approach', 'retract': {'dz': 0.1}}}))
    e = entries(c.project)
    pre = next(k for k in e if k.endswith('_pregrasp'))
    assert e[pre]['configuration'] == W1_J                   # the PTP in the configuration it was taught in
    assert 'configuration' not in e[pre.replace('_pregrasp', '_grasp')]   # the LIN approach: none


def test_renaming_the_joint_waypoint_follows_and_a_used_one_is_not_deleted():
    from trainit_setup_assistant.composer.ops import ComposerError
    c = taught()
    c.rename_waypoint('w1_j', 'w1_cfg')
    assert next(w for w in c.document()['waypoints'] if w['name'] == 'w1')['configuration'] == 'w1_cfg'
    assert "waypoint 'w1' (its configuration)" in c.waypoint_users('w1_cfg')
    with pytest.raises(ComposerError, match='is used by'):
        c.delete_waypoint('w1_cfg')


def test_r_i0_05_no_longer_flags_a_pose_that_names_its_configuration():
    assert not [w for w in configuration_advice(taught().project) if w.startswith('R-I0-05')]
    flagged = [w for w in configuration_advice(taught(configuration=None).project) if w.startswith('R-I0-05')]
    assert flagged and 'give the pose its configuration' in flagged[0]


def test_capture_tcp_and_joints_records_the_configuration(qapp):
    from test_i0_c_configuration import FakeLive
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    c = taught(configuration=None)
    joints = list(c.project.robot.planning_group.joints)
    live = FakeLive({j: 0.1 * i for i, j in enumerate(joints)}, ([0.42, 0.05, 0.31], DOWN))
    panel = LivePanel(AssistantController(c.project), lambda: c, lambda: live, lambda: 'w1', lambda name: None)
    panel.refresh()
    panel.target.setCurrentIndex(1)                          # 'w1' (replace it)
    assert panel.capture_both() == 'w1'
    w1 = next(w for w in c.document()['waypoints'] if w['name'] == 'w1')
    assert w1['configuration'] == 'w1_j' and "configuration 'w1_j'" in pose.summary(w1)


def test_the_tcp_editor_shows_and_sets_the_configuration(qapp):
    from trainit_setup_assistant.gui.composer.waypoints_panel import NO_CONFIGURATION, WaypointsPanel
    c = taught()
    panel = WaypointsPanel(lambda: c)
    panel.refresh()
    panel.select('w1')
    assert panel.configuration.currentText() == 'w1_j'
    assert 'w4' not in [panel.configuration.itemText(i) for i in range(panel.configuration.count())]   # joints only
    panel.configuration.setCurrentText(NO_CONFIGURATION)
    assert panel.apply()
    assert 'configuration' not in next(w for w in c.document()['waypoints'] if w['name'] == 'w1')
    panel.select('w1')
    panel.configuration.setCurrentText('home')
    assert panel.apply()
    assert next(w for w in c.document()['waypoints'] if w['name'] == 'w1')['configuration'] == 'home'


class Kin:
    """IK answers a fixed solution, whatever the seed (the far branch); joints are free of collision."""

    def __init__(self, solution):
        self.solution, self.seeds = solution, []

    def ik(self, frame, position, orientation, seed, collisions):
        self.seeds.append(seed)
        return list(self.solution)

    def fk(self, joints, frame):
        return [0.45, 0.25, 0.3], DOWN

    def valid(self, joints):
        return True, ''


def test_the_live_check_predicts_the_runtimes_verdict():
    from trainit_setup_assistant.composer.live_check import check_motions, check_waypoints
    near = Kin([v + 0.01 for v in W1_J])
    lines = check_waypoints(taught().project, near.ik, near.fk, near.valid)
    assert any(l.ok and l.waypoint == 'w1' and "from the configuration of 'w1_j'" in l.text for l in lines)
    assert W1_J in near.seeds                                # solved FROM the configuration
    far = Kin([W1_J[0] + 2.0] + W1_J[1:])                    # the other shoulder: 2 rad on joint 1
    lines = check_waypoints(taught().project, far.ik, far.fk, far.valid)
    bad = [l for l in lines if l.ok is False and l.waypoint == 'w1']
    assert bad and "not reached near its configuration 'w1_j' (joint 1 2.00 rad away)" in bad[0].text
    motion = [l for l in check_motions(taught().project, far.ik, far.fk, far.valid) if l.ok is False]
    assert motion and 'is not reached near its taught configuration' in motion[0].text
    assert 'CONFIGURATION_UNREACHABLE' in motion[0].text


def test_the_configuration_round_trips_in_the_yaml():
    from trainit_setup_assistant.model.io import project_to_dict as dump
    data = dump(taught().project)
    w1 = next(w for w in data['application']['waypoints'] if w['name'] == 'w1')
    assert w1['configuration'] == 'w1_j'
    assert dump(CanonicalProject.parse_obj(copy.deepcopy(data))) == data


# ---- the python review of the TSA side (2026-10-27) ---------------------------------------------------------

def test_a_named_configuration_is_a_branch_never_an_exact_pose():
    """Review 1: a pose 200 mm from the one its configuration was captured at is still reached near it -- no
    "X_j is 200 mm from ..." line; the exact-pose check stays for the X / X_j pair only."""
    from trainit_setup_assistant.composer.live_check import check_waypoints
    c = taught()
    w9 = pose.captured_tcp_waypoint('w9', [0.45, 0.05, 0.3], DOWN)
    w9['configuration'] = 'w1_j'
    c.set_waypoint(w9)
    near = Kin([v + 0.01 for v in W1_J])
    lines = check_waypoints(c.project, near.ik, near.fk, near.valid)
    assert not [l for l in lines if 'w9' in l.text and 'from' in l.text and 'mm' in l.text and l.ok is False]


def test_a_lin_or_circ_target_ignores_its_configuration_in_the_live_check():
    """Review 2: only a ptp / free move honours a configuration; a LIN to the pose never predicts a refusal."""
    from trainit_setup_assistant.composer.live_check import check_motions
    c = taught()
    c.edit(lambda d: d['tasks'][0]['steps'].insert(1, {'move': {'to': 'w1_j', 'type': 'ptp', 'profile': 'transit'}}))
    c.edit(lambda d: d['tasks'][0]['steps'].insert(2, {'move': {'to': 'w1', 'type': 'lin', 'profile': 'approach'}}))
    far = Kin([W1_J[0] + 2.0] + W1_J[1:])
    lines = check_motions(c.project, far.ik, far.fk, far.valid)
    assert not [l for l in lines if l.ok is False and 'step 3' in l.text and 'taught configuration' in l.text]


def test_the_waypoint_check_keeps_looking_for_a_near_branch():
    """Review 3: the first seed's solution far from the configuration does not end the search."""
    from trainit_setup_assistant.composer.live_check import check_waypoints

    class FarThenNear(Kin):
        def ik(self, frame, position, orientation, seed, collisions):
            self.seeds.append(seed)
            if seed is not None and list(seed) == W1_J:
                return [W1_J[0] + 2.0] + W1_J[1:]          # the taught seed lands on the far branch
            return [v + 0.01 for v in W1_J]
    kin = FarThenNear(W1_J)
    lines = check_waypoints(taught().project, kin.ik, kin.fk, kin.valid)
    assert [l.ok for l in lines if l.waypoint == 'w1'] == [True]


def test_the_check_tolerance_is_the_runtimes_default():
    from trainit_setup_assistant.composer.live_check import CONFIGURATION_TOLERANCE_RAD
    import re
    hpp = (PKG.parent / 'trainit_motion_runtime' / 'include' / 'trainit_motion_runtime' / 'pose_configuration.hpp')
    if not hpp.is_file():
        pytest.skip('trainit_motion_runtime is not beside this checkout')
    value = float(re.search(r'kDefaultConfigurationTolerance = ([0-9.]+)', hpp.read_text()).group(1))
    assert CONFIGURATION_TOLERANCE_RAD == pytest.approx(value)


def test_capture_tcp_keeps_the_configuration_and_says_so(qapp):
    from test_i0_c_configuration import FakeLive
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    c = taught()
    joints = list(c.project.robot.planning_group.joints)
    live = FakeLive({j: 0.0 for j in joints}, ([0.44, 0.25, 0.3], DOWN))
    panel = LivePanel(AssistantController(c.project), lambda: c, lambda: live, lambda: 'w1', lambda name: None)
    panel.refresh()
    panel.target.setCurrentIndex(1)                          # 'w1' (replace it)
    assert panel.capture_tcp() == 'w1'
    w1 = next(w for w in c.document()['waypoints'] if w['name'] == 'w1')
    assert w1['position'] == [0.44, 0.25, 0.3] and w1['configuration'] == 'w1_j'
    assert "its configuration 'w1_j' kept" in panel.message.text()


def test_the_relative_editor_sets_and_clears_a_configuration(qapp):
    from trainit_setup_assistant.gui.composer.waypoints_panel import NO_CONFIGURATION, WaypointsPanel
    c = taught()
    panel = WaypointsPanel(lambda: c)
    panel.refresh()
    panel.select('w2')                                       # relative to w1
    assert panel.rel_configuration.currentText() == NO_CONFIGURATION   # never inherited from w1
    panel.rel_configuration.setCurrentText('w1_j')
    assert panel.apply()
    assert next(w for w in c.document()['waypoints'] if w['name'] == 'w2')['configuration'] == 'w1_j'
    panel.select('w2')
    panel.rel_configuration.setCurrentText(NO_CONFIGURATION)
    assert panel.apply()
    assert 'configuration' not in next(w for w in c.document()['waypoints'] if w['name'] == 'w2')
