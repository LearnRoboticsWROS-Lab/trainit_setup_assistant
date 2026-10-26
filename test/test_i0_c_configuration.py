"""Owner Test C (friction row 24): a PTP to a TCP pose lets IK choose the joint configuration when it runs.

The same PTP home -> w1 moved 4.4 rad of joints in one cycle and 13.4 rad in the next, and the blended path
after it found no IK solution. Owner decision (a), TSA only:
- ``R-I0-05`` warns of a PTP / free move to a TCP pose before Cartesian motion (a path, a LIN, a CIRC);
- "Capture TCP + joints" captures the pose and the joints of the SAME robot state in one edit;
- the templates reach the start of their Cartesian motion through a joint waypoint.
A configuration-consistent PTP to a pose is runtime work (KNOWN_DEBT).
"""

import copy
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import Composer, pose
from trainit_setup_assistant.composer.advice import configuration_advice
from trainit_setup_assistant.composer.diagnostics import parse
from trainit_setup_assistant.composer.templates import apply_template
from trainit_setup_assistant.model import CanonicalProject

PKG = Path(__file__).resolve().parents[1]
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _sample_paths(monkeypatch):
    monkeypatch.chdir(SAMPLE.parent)                     # the sample's relative paths


def sample() -> CanonicalProject:
    return CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text()))


def program(steps, waypoints=None) -> CanonicalProject:
    """The simple-motion template with its task's steps replaced."""
    p = apply_template(sample(), 'simple_motion')
    data = p.dict()
    app = data['application']
    doc = copy.deepcopy(yaml.safe_load(
        (PKG / 'trainit_setup_assistant' / 'resources' / 'templates' / 'simple_motion.yaml').read_text())['application'])
    if waypoints:
        doc['waypoints'] += waypoints
    doc['tasks'][0]['steps'] = steps
    c = Composer(p)
    c.edit(lambda d: d.update({k: doc[k] for k in ('waypoints', 'tasks')}))
    return c.project


def test_a_ptp_to_a_tcp_pose_before_cartesian_motion_is_flagged_where_it_is():
    p = program([{'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}},
                 {'move': {'to': 'w1', 'type': 'ptp', 'profile': 'transit'}},       # 2: before a LIN
                 {'move': {'to': 'w2', 'type': 'lin', 'profile': 'approach'}},
                 {'move': {'to': 'w4', 'type': 'free', 'profile': 'transit'}},      # 4: before a path
                 {'move': {'to': 'w1', 'type': 'ptp', 'profile': 'transit'}},       # 5: before a path
                 {'path': 'sweep'},
                 {'move': {'to': 'w3', 'type': 'ptp', 'profile': 'transit'}},       # 7: before a PTP: fine
                 {'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}])
    advice = configuration_advice(p)
    assert [parse(a, 'warning').location.label for a in advice] == [(2,), (5,)]
    assert "PTP to the TCP pose 'w1' before a LIN move" in advice[0]
    assert "before path 'sweep'" in advice[1] and 'Capture TCP + joints' in advice[1]


def test_a_joint_target_and_nested_bodies():
    p = program([{'move': {'to': 'w1_j', 'type': 'ptp', 'profile': 'transit'}},     # a joint target: fine
                 {'move': {'to': 'w2', 'type': 'lin', 'profile': 'approach'}},
                 {'repeat': {'cycles': 2, 'body': [
                     {'move': {'to': 'w1', 'type': 'ptp', 'profile': 'transit'}},   # 3.1: before a path
                     {'path': 'sweep'}]}}])
    assert [parse(a, 'warning').location.label for a in configuration_advice(p)] == [(3, 1)]


def test_the_templates_reach_their_cartesian_motion_through_a_joint_waypoint():
    for tid in ('simple_motion', 'process_path'):
        p = apply_template(sample(), tid)
        assert configuration_advice(p) == [], tid
    prog = apply_template(sample(), 'simple_motion').application.program
    steps = prog.tasks[0].steps
    assert [s.move.to for s in (steps[1], steps[6])] == ['w1_j', 'w1_j']
    # owner Test C: a placeholder in joints is robot-specific (the FR3WML preview turned to the other side of
    # the table); until re-taught, the joint waypoint is the named state every template already requires
    w1_j = next(w for w in prog.waypoints if w.name == 'w1_j')
    assert w1_j.named == 'home' and w1_j.joints is None


def test_set_waypoints_is_one_edit():
    c = Composer(apply_template(sample(), 'simple_motion'))
    before = c.document()
    c.set_waypoints([pose.captured_tcp_waypoint('w1', [0.4, 0.1, 0.3], [1.0, 0.0, 0.0, 0.0]),
                     pose.captured_joint_waypoint('w1_j', [0.1] * 6),
                     pose.captured_joint_waypoint('new_j', [0.2] * 6)])
    names = [w['name'] for w in c.document()['waypoints']]
    assert names.count('w1_j') == 1 and names[-1] == 'new_j'
    c.undo()
    assert c.document() == before


class FakeLive:
    def __init__(self, joints, pose_):
        self.joints, self.pose = joints, pose_

    def joint_states_age(self):
        return 0.1

    def current_joint_values(self, names=None, timeout_s=3.0):
        return {j: v for j, v in self.joints.items() if names is None or j in names}

    def current_pose(self, base, tip, timeout_s=3.0):
        return self.pose


def test_capture_tcp_and_joints_records_the_pose_and_its_configuration(qapp):
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    c = Composer(apply_template(sample(), 'simple_motion'))
    joints = list(c.project.robot.planning_group.joints)
    live = FakeLive({j: 0.1 * i for i, j in enumerate(joints)}, ([0.42, 0.05, 0.31], [1.0, 0.0, 0.0, 0.0]))
    ctrl = AssistantController(c.project)
    seen = []
    panel = LivePanel(ctrl, lambda: c, lambda: live, lambda: 'w1', seen.append)
    panel.refresh()
    panel.target.setCurrentIndex(1)                       # 'w1' (replace it)
    before = c.document()
    assert panel.capture_both() == 'w1'
    wps = {w['name']: w for w in c.document()['waypoints']}
    assert pose.kind(wps['w1']) == 'TCP' and wps['w1']['position'] == [0.42, 0.05, 0.31]
    assert wps['w1_j']['joints'] == pytest.approx([0.1 * i for i in range(len(joints))])
    assert "'w1' replaced and 'w1_j' replaced" in panel.message.text()
    c.undo()                                              # ONE undo reverts both
    assert c.document() == before
    panel.target.setCurrentIndex(0)                       # a new waypoint: taughtN and taughtN_j
    name = panel.capture_both()
    assert {name, f'{name}_j'} <= {w['name'] for w in c.document()['waypoints']}


# ---- Check the waypoints (live): owner decision, 2026-10-23 -------------------------------------------------

import math as _math


def taught(w1_z=0.3):
    """The simple-motion template with w1 / w1_j taught (as the owner's i0_test_c.yaml)."""
    c = Composer(apply_template(sample(), 'simple_motion'))
    c.set_waypoints([pose.captured_tcp_waypoint('w1', [0.45, 0.25, w1_z], [1.0, 0.0, 0.0, 0.0]),
                     pose.captured_joint_waypoint('w1_j', [0.7, -1.6, -1.7, -1.4, 1.57, -0.86])])
    return c


class FakeKinematics:
    """MoveIt's answers: the table top is z = 0 (below it: only in collision); farther than 0.9 m: out of
    reach; FK of any joints = ``fk_pose``."""

    def __init__(self, fk_pose, colliding=()):
        self.fk_pose, self.colliding, self.calls = fk_pose, set(colliding), []

    def ik(self, frame, position, orientation, seed, collisions):
        self.calls.append(('ik', frame, tuple(position), seed is not None, collisions))
        if _math.dist(position, (0, 0, 0)) > 0.9:
            return None
        if collisions and position[2] < 0.0:
            return None
        return [0.0] * 6

    def fk(self, joints, frame):
        return self.fk_pose

    def valid(self, joints):
        hit = tuple(joints) in self.colliding
        return (not hit), ('upperarm_Link / table' if hit else '')


def lines_of(c, kin):
    from trainit_setup_assistant.composer.live_check import check_waypoints
    return {(l.waypoint, l.ok): l.text for l in check_waypoints(c.project, kin.ik, kin.fk, kin.valid)}


def test_live_check_a_taught_program_passes_and_seeds_ik_with_the_pair():
    c = taught()
    kin = FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]))
    got = lines_of(c, kin)
    assert all(ok for (_w, ok) in got), got
    assert "(from the configuration of 'w1_j')" in got[('w1', True)]
    assert "'w1_j' reaches the start of path 'sweep' (0.0 mm, 0.0°)" == got[('w1_j', True)]
    assert ('ik', c.project.robot.base_frame, (0.45, 0.25, 0.3), True, True) in kin.calls


def test_live_check_finds_a_relative_target_inside_the_table_and_a_stale_pair():
    """The owner's case: w1 at 157 mm makes w2 (200 mm down the tool) land at -43 mm; w1_j still reaches the
    old w1."""
    c = taught(w1_z=0.157)
    got = lines_of(c, FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0])))
    assert 'reachable only IN COLLISION' in got[('w2', False)] and '-43.0) mm' in got[('w2', False)]
    stale = got[('w1_j', False)]
    assert stale.startswith("'w1_j' is 143.0 mm") and "the start of path 'sweep'" in stale
    assert 'Capture TCP + joints' in stale


def test_live_check_out_of_reach_a_colliding_joint_state_and_an_unresolved_waypoint():
    c = taught()
    c.edit(lambda d: d['waypoints'].append({'name': 'far', 'position': [1.5, 0.0, 0.3],
                                            'orientation': [1.0, 0.0, 0.0, 0.0]}))
    c.edit(lambda d: d['waypoints'].append({'name': 'ghost', 'relative': {'to': 'nowhere', 'frame': 'tool',
                                                                          'dz': 0.1}}))
    joints = (0.7, -1.6, -1.7, -1.4, 1.57, -0.86)
    got = lines_of(c, FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]), colliding=[joints]))
    assert 'OUT OF REACH' in got[('far', False)]
    assert 'IN COLLISION (upperarm_Link / table)' in got[('w1_j', False)]
    assert got[('ghost', None)] == "'ghost': not resolved (see Diagnostics)"


def test_live_kinematics_asks_moveit_and_reads_its_answers(tmp_path):
    """The ROS side against in-process stand-ins of move_group's compute services."""
    rclpy = pytest.importorskip('rclpy')
    import threading
    from geometry_msgs.msg import PoseStamped
    from moveit_msgs.srv import GetPositionFK, GetPositionIK, GetStateValidity
    from moveit_msgs.msg import ContactInformation
    from rclpy.executors import SingleThreadedExecutor
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.kinematics import LiveKinematics
    os.environ['ROS_DOMAIN_ID'] = str(120 + os.getpid() % 30)
    rclpy.init()
    p = taught().project
    joints = list(p.robot.planning_group.joints)
    seen = {}
    node = rclpy.create_node('i0_c_fake_move_group')

    def ik(req, res):
        seen['ik'] = req.ik_request
        res.error_code.val = (seen.pop('code') if 'code' in seen else
                              1 if req.ik_request.pose_stamped.pose.position.z >= 0 else -31)
        res.solution.joint_state.name = joints
        res.solution.joint_state.position = [0.1] * len(joints)
        return res

    def fk(req, res):
        seen['fk'] = req
        res.error_code.val = 1
        res.pose_stamped = [PoseStamped()]
        res.pose_stamped[0].pose.position.z = 0.3
        res.pose_stamped[0].pose.orientation.w = 1.0
        return res

    def valid(req, res):
        res.valid = False
        res.contacts = [ContactInformation(contact_body_1='upperarm_Link', contact_body_2='table')]
        return res
    node.create_service(GetPositionIK, '/compute_ik', ik)
    node.create_service(GetPositionFK, '/compute_fk', fk)
    node.create_service(GetStateValidity, '/check_state_validity', valid)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            ex.spin_once(timeout_sec=0.05)
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    live = LiveCapture('i0_c_live')
    try:
        kin = LiveKinematics(live, p)
        assert kin.ik('base_link', [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], [0.2] * 6, True) == [0.1] * 6
        r = seen['ik']
        assert r.group_name == p.robot.planning_group.name and r.ik_link_name == p.robot.tip_link
        assert r.avoid_collisions and list(r.robot_state.joint_state.position) == [0.2] * 6
        assert kin.ik('base_link', [0.4, 0.0, -0.1], [1.0, 0.0, 0.0, 0.0], [0.2] * 6, False) is None
        assert not seen['ik'].avoid_collisions
        position, orientation = kin.fk([0.3] * 6, 'base_link')
        assert position[2] == pytest.approx(0.3) and orientation[3] == pytest.approx(1.0)
        assert seen['fk'].header.frame_id == 'base_link' and list(seen['fk'].fk_link_names) == [p.robot.tip_link]
        assert kin.valid([0.0] * 6) == (False, 'upperarm_Link / table')
        # no seed: MoveIt's own current state (an empty diff), never TSA's possibly stale joint states
        kin.ik('base_link', [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], None, True)
        assert seen['ik'].robot_state.is_diff and not list(seen['ik'].robot_state.joint_state.name)
        # a request MoveIt refuses is not 'out of reach'
        seen['code'] = -21
        from trainit_setup_assistant.composer.live_check import NotDecided
        with pytest.raises(NotDecided, match='FRAME_TRANSFORM_FAILURE'):
            kin.ik('nowhere', [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0], [0.2] * 6, True)
        # the whole check has a budget: the first call past it stops the check
        late = LiveKinematics(live, p, budget_s=0.0)
        with pytest.raises(RuntimeError, match='too slowly'):
            late.fk([0.0] * 6, 'base_link')
    finally:
        live.close()
        stop.set()
        t.join(timeout=2)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


def test_the_live_panel_shows_the_check_and_opens_a_waypoint(qapp):
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    c = taught(w1_z=0.157)
    ctrl = AssistantController(c.project)
    opened = []
    kin = FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]))
    panel = LivePanel(ctrl, lambda: c, lambda: object(), lambda: None, lambda name: None,
                      kinematics_factory=lambda live, project: kin, go_to_waypoint=opened.append)
    assert not panel.check_list.isVisibleTo(panel)
    lines = panel.check_waypoints()
    assert panel.check_list.isVisibleTo(panel) and panel.check_list.count() == len(lines)
    assert panel.message.text().startswith('Live check: ') and '2 ✖' in panel.message.text()
    bad = next(i for i in range(panel.check_list.count()) if panel.check_list.item(i).text().startswith('✖'))
    panel.check_list.itemDoubleClicked.emit(panel.check_list.item(bad))
    assert opened == ['w2']
    panel = LivePanel(ctrl, lambda: c, lambda: object(), lambda: None, lambda name: None,
                      kinematics_factory=lambda live, project: (_ for _ in ()).throw(RuntimeError('no /compute_ik')))
    assert panel.check_waypoints() is None and 'start the planning environment' in panel.message.text()


# ---- the review of the live check (python-reviewer, 2026-10-26) ---------------------------------------------

def test_a_named_state_is_the_planning_groups_and_a_refused_request_is_not_decided():
    from trainit_setup_assistant.composer.live_check import NotDecided, check_waypoints
    from trainit_setup_assistant.model.robot import NamedState
    c = taught()
    p = c.project.copy(deep=True)
    arm = p.robot.planning_group.name
    joints = list(p.robot.planning_group.joints)
    p.robot.named_states = [NamedState(name='home', group='gripper', joint_values={'finger': 0.0}),
                            NamedState(name='home', group=arm, joint_values={j: 0.0 for j in joints})]
    kin = FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]))

    def ik(frame, position, orientation, seed, collisions):
        if position[0] == pytest.approx(0.45) and position[1] == pytest.approx(-0.25) and position[2] < 0.2:
            raise NotDecided('MoveIt refuses the IK request (FRAME_TRANSFORM_FAILURE)')
        return kin.ik(frame, position, orientation, seed, collisions)
    got = {(l.waypoint, l.ok): l.text for l in check_waypoints(p, ik, kin.fk, kin.valid)}
    assert got[('home', True)] == "'home' (named state 'home'): collision-free"
    assert 'not decided -- MoveIt refuses the IK request (FRAME_TRANSFORM_FAILURE)' in got[('w3', None)]
    assert ('w4', True) in got                                   # the others are still checked


def test_a_path_start_is_checked_across_a_process_step():
    from trainit_setup_assistant.composer.live_check import _path_starts
    c = taught()
    c.edit(lambda d: d['tasks'][0]['steps'].insert(7, {'process': {'kind': 'process_on', 'channel': 'glue'}}))
    assert ('w1_j', 'sweep', 'w1') in _path_starts(c.project)


def test_the_check_slot_never_raises_into_qt(qapp):
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    c = taught()
    ctrl = AssistantController(c.project)
    panel = LivePanel(ctrl, lambda: c, lambda: object(), lambda: None, lambda name: None,
                      kinematics_factory=lambda live, project: (_ for _ in ()).throw(KeyError('j1')))
    assert panel.check_waypoints() is None and panel.message.text().startswith('Check the waypoints failed: KeyError')
    ctrl.project = sample()                                      # the ur5e sample's own application
    if not ctrl.project.application.is_task_program():
        assert panel.check_waypoints() is None and 'only a task program' in panel.message.text()


def test_one_ik_start_that_fails_proves_nothing():
    """Found on a real move_group: w2b (next to the bin wall) was collision-free from the taught configuration
    and "in collision" from the mock's start state. A failure is reported only when every start fails."""
    from trainit_setup_assistant.composer.live_check import check_waypoints
    c = taught()
    starts = []

    def ik(frame, position, orientation, seed, collisions):
        starts.append(seed)
        if collisions and seed is None:                          # move_group's own state: a bad start
            return None
        return [0.3] * 6
    kin = FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]))
    got = {(l.waypoint, l.ok) for l in check_waypoints(c.project, ik, kin.fk, kin.valid)}
    assert all(ok for _w, ok in got), got
    assert None not in starts                                     # the taught / previous starts were enough

    def never(frame, position, orientation, seed, collisions):
        return None if collisions else [0.0] * 6
    lines = {(l.waypoint, l.ok): l.text for l in check_waypoints(c.project, never, kin.fk, kin.valid)}
    assert 'start configurations' in lines[('w2b', False)] and 'IN COLLISION' in lines[('w2b', False)]


def test_a_check_before_the_scene_is_loaded_says_its_collision_answers_are_partial():
    """Found on a real move_group: right after start-up its scene manager had not loaded the table yet, and a
    pose inside the table passed. The check names the cell objects the planning scene lacks."""
    from trainit_setup_assistant.composer.live_check import check_waypoints
    c = taught()
    expected = [o.id for o in c.project.scene.objects]
    kin = FakeKinematics(([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]))
    assert expected == ['table']                                  # the sample's cell
    lines = check_waypoints(c.project, kin.ik, kin.fk, kin.valid, lambda: [])
    assert lines[0].ok is None and 'does not hold table' in lines[0].text and lines[0].waypoint == ''
    full = check_waypoints(c.project, kin.ik, kin.fk, kin.valid, lambda: expected)
    assert all('does not hold' not in l.text for l in full)



# ---- owner Test C, friction row 16: a capture that changes the kind of a used waypoint asks first -------------

def _row16_panel(c, live, selected):
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    panel = LivePanel(AssistantController(c.project), lambda: c, lambda: live, lambda: selected, lambda n: None)
    panel.questions = []
    panel.answer = False
    panel._confirm = lambda q: panel.questions.append(q) or panel.answer
    panel.refresh()
    panel.target.setCurrentIndex(1)                       # '<selected>' (replace it)
    return panel


def test_capture_joints_into_a_used_tcp_waypoint_asks_and_no_changes_nothing(qapp):
    """The owner's case: Capture joints into 'w1', which 'w2' is relative to and path 'sweep' starts at."""
    c = taught()
    joints = list(c.project.robot.planning_group.joints)
    panel = _row16_panel(c, FakeLive({j: 0.2 for j in joints}, None), 'w1')
    before, depth = c.document(), len(c._undo)
    assert panel.capture_joints() is None
    q = panel.questions[0]
    assert "'w1' is a TCP waypoint and becomes Joints" in q
    assert "'w2' (relative to it)" in q and "path 'sweep' (state 0)" in q and 'may need it as TCP' in q
    assert c.document() == before and len(c._undo) == depth                    # no undo entry
    assert panel.message.text() == "not captured: 'w1' kept as TCP"
    panel.answer = True
    assert panel.capture_joints() == 'w1' and pose.kind(next(
        w for w in c.document()['waypoints'] if w['name'] == 'w1')) == 'Joints'
    assert len(c._undo) == depth + 1


def test_no_question_without_users_or_when_the_kind_is_kept(qapp):
    c = taught()
    c.edit(lambda d: d['waypoints'].append({'name': 'spare', 'position': [0.4, 0.0, 0.3],
                                            'orientation': [1.0, 0.0, 0.0, 0.0]}))
    joints = list(c.project.robot.planning_group.joints)
    panel = _row16_panel(c, FakeLive({j: 0.2 for j in joints}, ([0.41, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])), 'spare')
    assert panel.capture_joints() == 'spare' and panel.questions == []          # nobody uses it
    panel = _row16_panel(c, FakeLive({j: 0.2 for j in joints}, ([0.41, 0.2, 0.3], [1.0, 0.0, 0.0, 0.0])), 'w1')
    assert panel.capture_tcp() == 'w1' and panel.questions == []                # TCP stays TCP


def test_capture_tcp_and_joints_asks_only_when_the_partner_changes_meaning(qapp):
    """The template's 'w1_j' is the named state home until taught: Capture TCP + joints makes it Joints -- the
    designed flow, both joint targets: no question. A '<target>_j' that is a POSE some step uses does ask."""
    c = Composer(apply_template(sample(), 'simple_motion'))
    joints = list(c.project.robot.planning_group.joints)
    live = FakeLive({j: 0.2 for j in joints}, ([0.45, 0.25, 0.3], [1.0, 0.0, 0.0, 0.0]))
    panel = _row16_panel(c, live, 'w1')
    assert panel.capture_both() == 'w1' and panel.questions == []
    c.undo()
    c.set_waypoint({'name': 'w3_j', 'position': [0.45, -0.25, 0.2], 'orientation': [1.0, 0.0, 0.0, 0.0]})
    c.edit(lambda d: d['tasks'][0]['steps'].append({'move': {'to': 'w3_j', 'type': 'lin', 'profile': 'approach'}}))
    panel = _row16_panel(c, live, 'w3')
    before = c.document()
    assert panel.capture_both() is None
    assert len(panel.questions) == 1 and "'w3_j' is a TCP waypoint and becomes Joints" in panel.questions[0]
    assert c.document() == before and panel.message.text() == "not captured: 'w3_j' kept as TCP"
