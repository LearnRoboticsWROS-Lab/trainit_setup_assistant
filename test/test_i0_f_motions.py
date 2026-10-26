"""Owner Test F.1 (friction row 36): the live check follows the program's motions ALONG THEIR WAY.

Every waypoint passed "Check the waypoints (live)", then the run failed: the PTP from home2 to the grasp's
pre-grasp crossed the cell camera between the two (move_group: invalid states 9-11 of 17). The check now
walks the program as the runtime will plan it -- the moves, the grasp / release idioms' own motions, a
repeat's next cycle -- and samples each PTP (a joint-space line) and LIN (a Cartesian line) with MoveIt's
state validity. Pure: a fake "Cartesian robot" (its joints ARE the TCP position) with a known obstacle.
"""

import copy
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import Composer
from trainit_setup_assistant.composer.live_check import check_motions
from trainit_setup_assistant.composer.templates import apply_template
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.io import project_to_dict

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'
VALIDATION_C = PKG / 'test' / 'data' / 'm3_6' / 'validation_c.yaml'
DOWN = [1.0, 0.0, 0.0, 0.0]                              # the tool pointing down (180 deg about x)


@pytest.fixture(autouse=True)
def _sample_paths(monkeypatch):
    monkeypatch.chdir(SAMPLE.parent)


@pytest.fixture
def qapp():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def project(waypoints, steps) -> CanonicalProject:
    p = apply_template(CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text())), 'simple_motion')
    c = Composer(p)

    def mutate(d):
        d['waypoints'] = copy.deepcopy(waypoints)
        d['tasks'] = [{'name': 'cycle', 'steps': copy.deepcopy(steps)}]
        d['main_task'] = 'cycle'
        d.pop('paths', None)
    c.edit(mutate)
    return c.project


def joint(name, x, y, z):
    return {'name': name, 'type': 'joint', 'joints': [x, y, z, 0.0, 0.0, 0.0]}


def tcp(name, x, y, z):
    return {'name': name, 'position': [x, y, z], 'orientation': DOWN}


def ptp(to, kind='ptp'):
    return {'move': {'to': to, 'type': kind, 'profile': 'transit'}}


class CartesianRobot:
    """Joints 1-3 are the TCP position; the camera fills x in [0.55, 0.65] above z = 0.25."""

    def __init__(self):
        self.valid_calls = 0

    @staticmethod
    def hit(q):
        return 0.55 <= q[0] <= 0.65 and q[2] > 0.25

    def ik(self, frame, position, orientation, seed, collisions):
        q = [float(position[0]), float(position[1]), float(position[2]), 0.0, 0.0, 0.0]
        return None if collisions and self.hit(q) else q

    def fk(self, joints, frame):
        return list(joints[:3]), list(DOWN)

    def valid(self, joints):
        self.valid_calls += 1
        return (False, 'wrist2_Link / camera_link') if self.hit(joints) else (True, '')


def check(p, robot=None):
    robot = robot or CartesianRobot()
    return check_motions(p, robot.ik, robot.fk, robot.valid)


def test_a_ptp_through_the_camera_is_found_between_two_good_waypoints():
    lines = check(project([joint('start_j', 0.0, 0.0, 0.3), tcp('pick', 1.0, 0.0, 0.3)],
                          [ptp('start_j'), ptp('pick')]))
    assert [line.ok for line in lines] == [False]
    assert "step 2 (move ptp to 'pick')" in lines[0].text
    text = lines[0].text
    assert "the PTP from 'start_j' to 'pick' collides at 55-65 % of the way (wrist2_Link / camera_link)" in text
    assert 'add a joint waypoint between them' in text


def test_the_grasp_pregrasp_ptp_is_checked_and_a_joint_waypoint_clears_it():
    """The owner's case: the pre-grasp is the idiom's own target, never a waypoint of the table."""
    wps = [joint('start_j', 0.0, 0.0, 0.3), tcp('pick', 1.0, 0.0, 0.3)]
    grasp = {'grasp': {'at': 'pick', 'standoff_m': 0.15, 'profile': 'transit', 'approach_profile': 'transit',
                       'retract': {'dz': 0.15}}}
    lines = check(project(wps, [ptp('start_j'), grasp]))
    assert [line.ok for line in lines] == [False]
    assert "(grasp at 'pick') PreGrasp: the PTP from 'start_j' to the pre-grasp of step 2 collides" in lines[0].text
    assert 'add a joint waypoint before this step' in lines[0].text  # never "make it Free": an idiom's PTP
    fixed = check(project(wps + [joint('clear_j', 1.0, 0.0, 0.1)], [ptp('start_j'), ptp('clear_j'), grasp]))
    assert [line.ok for line in fixed] == [True]
    assert 'no collision in 4 PTP / LIN motions' in fixed[0].text     # clear_j, pre-grasp, approach, retract


def test_a_free_move_is_planned_around_and_checked_at_its_end_only():
    lines = check(project([joint('start_j', 0.0, 0.0, 0.3), tcp('pick', 1.0, 0.0, 0.3)],
                          [ptp('start_j'), ptp('pick', 'free')]))
    assert lines == []                                   # nothing sampled, nothing to say


def test_a_lin_through_the_camera_is_found():
    lines = check(project([joint('a_j', 1.0, 0.0, 0.3), tcp('b', 0.0, 0.0, 0.3)],
                          [ptp('a_j'), {'move': {'to': 'b', 'type': 'lin', 'profile': 'transit'}}]))
    assert [line.ok for line in lines] == [False]
    assert "the LIN from 'a_j' to 'b' collides" in lines[0].text and 'teach the two ends clear of it' in lines[0].text
    assert lines[0].waypoint == 'b'                                  # a double click opens it


def test_a_repeats_next_cycle_starts_where_the_body_ends():
    body = [ptp('a_j'), ptp('c_j'), ptp('b_j')]
    lines = check(project([joint('a_j', 0.0, 0.0, 0.3), joint('c_j', 0.0, 0.0, 0.0), joint('b_j', 1.0, 0.0, 0.3)],
                          [{'repeat': {'cycles': 2, 'body': body}}]))
    assert [line.ok for line in lines] == [False]
    assert "the PTP from 'b_j' to 'a_j' collides" in lines[0].text   # only on the way back to the first


def test_a_detected_target_stops_the_check_until_a_joint_waypoint():
    data = yaml.safe_load(VALIDATION_C.read_text())      # a contract and a detector
    detector = data['perception']['detectors'][0]['name']
    data['application'] = {
        'type': 'task_program', 'motion_profiles': {'transit': {'speed': 50, 'acceleration': 50}},
        'waypoints': [joint('start_j', 0.0, 0.0, 0.0), tcp('pick', 1.0, 0.0, 0.1), joint('home_j', 0.0, 0.0, 0.3),
                      tcp('over', 0.6, 0.0, 0.3)],
        'tasks': [{'name': 'cycle', 'steps': [
            ptp('start_j'),
            {'detect': {'detector': detector, 'source': 'detections', 'approach': [0.0, 0.0, -1.0], 'keep': 1,
                        'out': 'parts'}},
            {'grasp': {'candidates': 'parts', 'index': 0, 'orientation_from': 'pick', 'standoff_m': 0.1,
                       'profile': 'transit', 'approach_profile': 'transit', 'retract': {'dz': 0.1}}},
            ptp('over'),                                 # from an unknown place: never solved from a guess
            ptp('home_j'),
            ptp('start_j')]}],
        'main_task': 'cycle'}
    lines = check(CanonicalProject.parse_obj(data))
    assert [line.ok for line in lines] == [None, True]   # one line for the whole grasp step
    assert 'PreGrasp: where the robot goes is known only at runtime' in lines[0].text
    assert 'no collision in 1 PTP / LIN motion ' in lines[1].text and '1 not decided above' in lines[1].text


def test_a_program_that_does_not_validate_is_not_followed():
    p = project([joint('start_j', 0.0, 0.0, 0.3), tcp('pick', 1.0, 0.0, 0.3)],
                [ptp('start_j'), {'grasp': {'at': 'pick', 'standoff_m': 0.1, 'profile': 'transit',
                                            'approach_profile': 'transit', 'retract': {'dz': 0.1}}}])
    p.robot.gripper.contract = None                      # R-M3-13: the grasp has no end effector to command
    robot = CartesianRobot()
    lines = check(p, robot)
    assert [(line.ok, line.text) for line in lines] == [
        (None, 'the motions between the waypoints: checked once the program is Valid')]
    assert robot.valid_calls == 0


# ---- the python review of the motion check (2026-10-26) --------------------------------------------------

def template_project():
    return apply_template(CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text())), 'simple_motion')


def test_a_step_moveit_cannot_decide_never_stops_the_others():
    from trainit_setup_assistant.composer.live_check import NotDecided

    class Refusing(CartesianRobot):
        def ik(self, frame, position, orientation, seed, collisions):
            if abs(position[1] + 0.25) < 1e-9 and abs(position[2] - 0.3) < 1e-9:     # w4
                raise NotDecided('MoveIt refuses the IK request')
            return super().ik(frame, position, orientation, seed, collisions)
    lines = check(template_project(), Refusing())
    undecided = [line.text for line in lines if line.ok is None]
    assert 'step 5: not decided -- MoveIt refuses the IK request' in undecided       # the LIN to w4
    assert 'step 8: not decided -- MoveIt refuses the IK request' in undecided       # the path through w4


def test_a_named_state_without_every_joint_is_not_decided_never_out_of_reach():
    from trainit_setup_assistant.model.robot import NamedState
    p = project([joint('start_j', 0.0, 0.0, 0.0), {'name': 'h', 'type': 'joint', 'named': 'broken'}],
                [ptp('start_j'), ptp('h')])
    p.robot.named_states.append(NamedState(name='broken', group=p.robot.planning_group.name,
                                           joint_values={'shoulder_pan_joint': 0.0}))
    lines = check(p)
    assert [line.ok for line in lines] == [None]
    assert "named state 'broken' has no value for every joint" in lines[0].text


def test_a_tcp_entry_naming_the_tip_is_checked_like_any_other():
    p = project([joint('start_j', 0.0, 0.0, 0.3), tcp('pick', 1.0, 0.0, 0.3)], [ptp('start_j'), ptp('pick')])
    data = project_to_dict(p)
    data['application']['waypoints'][1]['tcp'] = p.robot.tip_link
    lines = check(CanonicalProject.parse_obj(data))
    assert [line.ok for line in lines] == [False] and 'collides at 55-65 %' in lines[0].text


def test_alternatives_that_end_apart_leave_the_next_motion_unchecked():
    alts = {'fallback': {'bodies': [[ptp('a_j')], [ptp('b_j')]]}}
    lines = check(project([joint('s_j', 0.0, 0.0, 0.3), joint('a_j', 0.0, 0.0, 0.2), joint('b_j', 1.0, 0.0, 0.2),
                           joint('end_j', 1.0, 0.0, 0.3)],
                          [ptp('s_j'), alts, ptp('end_j'), ptp('s_j')]))
    # s_j -> a_j and s_j -> b_j are sampled (b_j crosses at z = 0.2: below the camera); where the robot is after
    # the alternatives is unknown, so end_j is only a new start; end_j -> s_j crosses the camera
    assert [line.ok for line in lines] == [False]
    assert "the PTP from 'end_j' to 's_j' collides" in lines[0].text


def test_a_repeats_segments_are_computed_once():
    body = [ptp('a_j'), ptp('c_j'), ptp('b_j')]
    wps = [joint('a_j', 0.0, 0.0, 0.0), joint('c_j', 0.0, 0.0, 0.2), joint('b_j', 1.0, 0.0, 0.2)]
    twice, once = CartesianRobot(), CartesianRobot()
    check(project(wps, [{'repeat': {'cycles': 2, 'body': body}}]), twice)
    check(project(wps, body + [ptp('a_j')]), once)        # the same segments, each once
    assert twice.valid_calls == once.valid_calls > 0


def test_a_lin_whose_ik_changes_branch_is_not_decided():
    class Flipping(CartesianRobot):
        def ik(self, frame, position, orientation, seed, collisions):
            q = super().ik(frame, position, orientation, seed, collisions)
            if q is not None and position[0] < 0.5:
                q[4] = 2.0                                # the wrist flips past the middle of the line
            return q
    lines = check(project([joint('a_j', 1.0, 0.0, 0.1), tcp('b', 0.0, 0.0, 0.1)],
                          [ptp('a_j'), {'move': {'to': 'b', 'type': 'lin', 'profile': 'transit'}}]), Flipping())
    assert [line.ok for line in lines] == [None]
    assert 'the IK changes configuration at' in lines[0].text and 'check it in RViz' in lines[0].text


def test_slerp_is_the_shorter_arc_between_unit_quaternions():
    from trainit_setup_assistant.composer.live_check import _slerp
    half = _slerp([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], 0.5)
    assert half == pytest.approx([0.0, 0.0, 0.70710678, 0.70710678])
    q = [0.0, 0.0, 0.38268343, 0.92387953]
    assert _slerp(q, [-v for v in q], 0.5) == pytest.approx(q)              # -q is the same rotation
    assert _slerp([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 2.0, 0.0], 0.5) == pytest.approx(half)   # b normalised


def test_the_panel_keeps_the_waypoint_lines_when_the_motions_stop(qapp):
    from test_i0_c_configuration import FakeKinematics
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    p = project([joint('start_j', 0.0, 0.0, 0.3), tcp('pick', 0.5, 0.0, 0.3)], [ptp('start_j'), ptp('pick')])
    c = Composer(p)
    made = []

    class OutOfTime(FakeKinematics):
        def valid(self, joints):
            raise RuntimeError('the check stopped after 20 s (its time budget)')

    def factory(live, project_):
        made.append(1)
        return FakeKinematics(([0.5, 0.0, 0.3], DOWN)) if len(made) == 1 else OutOfTime(([0.5, 0.0, 0.3], DOWN))
    panel = LivePanel(AssistantController(c.project), lambda: c, lambda: object(), lambda: None, lambda name: None,
                      kinematics_factory=factory)
    lines = panel.check_waypoints()
    assert len(made) == 2                                 # the motions have a budget of their own
    assert [line.ok for line in lines][:2] == [True, True]                       # the waypoints' lines kept
    assert lines[-1].ok is None and 'not checked to the end -- the check stopped after 20 s' in lines[-1].text
    assert panel.check_button.isEnabled()
