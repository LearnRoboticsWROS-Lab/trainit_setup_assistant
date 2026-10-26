"""vNext I0.4 (I0_DESIGN.md §12): live capture (the CORE teaching flow) and the optional RViz Preview.

- A capture is the robot's CURRENT state: TF base -> tip, or the joint states. It becomes ONE Composer
  operation: a new waypoint, or the selected one replaced, undoable.
- The Preview publishes only a goal STATE and the planning group to RViz's MotionPlanning display (the
  spike: PASS). It never plans or executes; a TCP waypoint becomes joints through MoveIt's IK, a
  computation.
- The launcher stops a command's WHOLE process group, and its children never inherit VS Code's snap
  library paths (both found by the spike).
"""
import math
import os
import threading
import time
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import Composer, pose
from trainit_setup_assistant.composer import launch_plan
from trainit_setup_assistant.model import CanonicalProject

PKG = Path(__file__).resolve().parents[1]
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'


def project() -> CanonicalProject:
    data = yaml.safe_load(COMPOSITION.read_text())
    data['robot']['base_frame'], data['robot']['tip_link'] = 'base_link', 'tcp'
    return CanonicalProject.parse_obj(data)


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        if app is not None:
            app.processEvents()
        time.sleep(0.02)
    return cond()


# ---- the launcher: process groups and the snap environment -------------------------------------------------

def test_a_launched_command_never_inherits_the_snaps_library_paths(monkeypatch):
    monkeypatch.setenv('GTK_PATH', '/snap/code/271/usr/lib/x86_64-linux-gnu/gtk-3.0')
    monkeypatch.setenv('LOCPATH', '/snap/code/271/usr/lib/locale')
    monkeypatch.setenv('XDG_DATA_DIRS', '/home/u/snap/code/271:/usr/share')
    monkeypatch.setenv('XDG_DATA_DIRS_VSCODE_SNAP_ORIG', '/usr/share')
    env = launch_plan.child_environment(dict(os.environ))
    assert 'GTK_PATH' not in env and 'LOCPATH' not in env and env['XDG_DATA_DIRS'] == '/usr/share'
    assert launch_plan.child_environment({'GTK_PATH': '/usr/lib/gtk-3.0'}) == {'GTK_PATH': '/usr/lib/gtk-3.0'}


def test_the_launched_process_sees_the_cleaned_environment(qapp, monkeypatch):
    monkeypatch.setenv('GTK_PATH', '/snap/code/271/gtk')
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher = Launcher()
    proc = launcher.start(launch_plan.Command('build', 'b', 'echo "[${GTK_PATH:-}]"', False))
    assert _wait(qapp, lambda: proc.exit_code is not None)
    assert proc.lines[0] == '[]'


def _alive(pid: int) -> bool:
    try:
        with open(f'/proc/{pid}/stat') as f:
            return f.read().split(')')[-1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def test_stopping_a_command_stops_its_whole_group_even_a_child_that_ignores_signals(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher = Launcher()
    script = "trap '' INT TERM; sleep 120 & echo child $!; wait"
    proc = launcher.start(launch_plan.Command('environment', 'env', script, True))
    assert _wait(qapp, lambda: any(line.startswith('child ') for line in proc.lines))
    child = int(next(line for line in proc.lines if line.startswith('child ')).split()[1])
    assert _alive(child)
    started = time.time()
    launcher.stop_all()
    assert _wait(qapp, lambda: not _alive(child), timeout=15.0), 'the ignoring child survived'
    assert not launcher.running('environment') and time.time() - started < 15.0


# ---- capture (the core flow) with a fake live session ------------------------------------------------------

class FakeLive:
    def __init__(self, joints=None, pose_=None, age=0.2):
        self.joints = joints or {}
        self.pose = pose_
        self.age = age
        self.node = None

    def joint_states_age(self):
        return self.age

    def current_joint_values(self, names=None, timeout_s=3.0):
        if not self.joints:
            raise TimeoutError('no /joint_states received (is the live session running?)')
        return {j: v for j, v in self.joints.items() if names is None or j in names}

    def current_pose(self, base, tip, timeout_s=3.0):
        if self.pose is None:
            raise TimeoutError(f'TF {base}->{tip} unavailable')
        return self.pose


def _live_panel(c, live, selected=lambda: None, preview_factory=None):
    from trainit_setup_assistant.gui.composer.live_panel import LivePanel
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(c.project)
    seen = []
    panel = LivePanel(ctrl, lambda: c, lambda: live, selected, seen.append, preview_factory,
                      age=lambda: live.joint_states_age())
    panel.questions = []                     # row 16: a kind change of a used waypoint asks (yes, here)
    panel._confirm = lambda q: panel.questions.append(q) or True
    panel.refresh()
    return panel, seen, ctrl


def test_capture_tcp_creates_a_waypoint_from_the_current_pose(qapp):
    c = Composer(project())
    live = FakeLive(pose_=([0.4200001, -0.25, 0.22], [0.7071068, -0.7071068, 0.0, -1e-9]))
    panel, seen, _ = _live_panel(c, live)
    assert 'Robot: LIVE' in panel.header.text()
    name = panel.capture_tcp()
    assert name == 'taught1' and seen == ['taught1']
    wp = next(w for w in c.document()['waypoints'] if w['name'] == 'taught1')
    assert wp == pose.captured_tcp_waypoint('taught1', *live.pose)
    assert 'CURRENT TCP' in panel.message.text() and 'created' in panel.message.text()
    assert panel.capture_tcp() == 'taught2'
    c.undo()
    assert 'taught2' not in c.waypoint_names()


def test_capture_joints_replaces_the_selected_waypoint_in_group_order(qapp):
    c = Composer(project())
    joints = c.project.robot.planning_group.joints
    target = c.waypoint_names()[0]
    live = FakeLive(joints={j: 0.1 * (i + 1) for i, j in enumerate(reversed(joints))})
    panel, _, _ = _live_panel(c, live, selected=lambda: target)
    panel.target.setCurrentIndex(1)                                  # into the selected waypoint
    assert panel.capture_joints() == target
    wp = next(w for w in c.document()['waypoints'] if w['name'] == target)
    assert wp == {'name': target, 'type': 'joint', 'joints': [round(live.joints[j], 6) for j in joints]}
    assert 'replaced' in panel.message.text()


def test_a_capture_without_a_running_robot_says_what_to_do(qapp):
    c = Composer(project())
    panel, _, _ = _live_panel(c, FakeLive(age=None))
    assert 'Robot: OFFLINE' in panel.header.text()
    assert panel.capture_tcp() is None and 'start the planning environment' in panel.message.text()
    assert panel.capture_joints() is None and 'start the planning environment' in panel.message.text()
    assert not c.can_undo


def test_capture_joints_refuses_a_robot_that_is_not_the_planning_group(qapp):
    c = Composer(project())
    panel, _, _ = _live_panel(c, FakeLive(joints={'other_joint': 0.0}))
    assert panel.capture_joints() is None and 'lack' in panel.message.text()


class FakePreview:
    def __init__(self, live, listening=True):
        self._listening, self.shown = listening, []

    def listening(self):
        return self._listening

    def preview_waypoint(self, project, wp):
        self.shown.append(wp['name'])
        return f"'{wp['name']}' shown"


def test_the_preview_is_optional_and_needs_rviz_listening(qapp):
    c = Composer(project())
    name = c.waypoint_names()[0]
    holder = {}

    def deaf(live):
        holder['p'] = FakePreview(live, listening=False)
        return holder['p']
    panel, _, _ = _live_panel(c, FakeLive(), selected=lambda: name, preview_factory=deaf)
    assert not panel.preview() and 'External' in panel.message.text()
    panel2, _, _ = _live_panel(c, FakeLive(), selected=lambda: name,
                               preview_factory=lambda live: holder.setdefault('q', FakePreview(live)))
    assert panel2.preview() and holder['q'].shown == [name]
    panel3, _, _ = _live_panel(c, FakeLive(), selected=lambda: None, preview_factory=FakePreview)
    assert not panel3.preview() and 'select a waypoint' in panel3.message.text()


# ---- the RViz Preview against an in-process graph -------------------------------------------------------

@pytest.fixture
def ros_graph():
    rclpy = pytest.importorskip('rclpy')
    os.environ['ROS_DOMAIN_ID'] = str(110 + os.getpid() % 30)
    rclpy.init()
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()


def test_the_preview_sends_only_a_goal_state_and_the_group(ros_graph):
    from moveit_msgs.msg import RobotState
    from moveit_msgs.srv import GetPositionIK
    from rclpy.executors import SingleThreadedExecutor
    from std_msgs.msg import String
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.preview import GOAL_TOPIC, GROUP_TOPIC, RvizPreview
    from sensor_msgs.msg import JointState
    p = project()
    joints = list(p.robot.planning_group.joints)
    rviz = ros_graph.create_node('fake_rviz')
    got = {'goal': [], 'group': []}
    rviz.create_subscription(RobotState, GOAL_TOPIC, lambda m: got['goal'].append(m), 10)
    rviz.create_subscription(String, GROUP_TOPIC, lambda m: got['group'].append(m.data), 10)
    js = rviz.create_publisher(JointState, '/joint_states', 10)

    def ik(req, res):
        res.error_code.val = 1
        res.solution.joint_state.name = joints
        res.solution.joint_state.position = [0.5] * len(joints)
        return res
    rviz.create_service(GetPositionIK, '/compute_ik', ik)
    ex = SingleThreadedExecutor()
    ex.add_node(rviz)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            js.publish(JointState(name=joints, position=[0.0] * len(joints)))
            ex.spin_once(timeout_sec=0.05)
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    live = LiveCapture('i0_4_preview')
    try:
        preview = RvizPreview(live)
        assert _wait(None, preview.listening, 5.0)
        wp_j = {'name': 'j', 'type': 'joint', 'joints': [0.1 * i for i in range(len(joints))]}
        preview.preview_waypoint(p, wp_j)
        assert _wait(None, lambda: got['goal'], 5.0)
        msg = got['goal'][-1]
        assert msg.is_diff and list(msg.joint_state.name) == joints
        assert list(msg.joint_state.position) == pytest.approx(wp_j['joints'])
        assert got['group'][-1] == p.robot.planning_group.name
        assert _wait(None, lambda: live.joint_states_age() is not None, 5.0)
        wp_t = {'name': 't', 'position': [0.4, 0.0, 0.3], 'orientation': [0.0, 0.0, 0.0, 1.0]}
        preview.preview_waypoint(p, wp_t)
        assert _wait(None, lambda: len(got['goal']) >= 2, 5.0)
        assert list(got['goal'][-1].joint_state.position) == pytest.approx([0.5] * len(joints))
        with pytest.raises(RuntimeError, match='relative'):
            preview.preview_waypoint(p, {'name': 'r', 'relative': {'to': 'j', 'frame': 'tool'}})
        preview.close()
    finally:
        live.close()
        stop.set()
        t.join(timeout=2)
        rviz.destroy_node()


# ---- review fixes (I0.2-I0.4 python review) -----------------------------------------------------------

def test_a_failing_workspace_stops_the_line_before_anything_runs():
    import subprocess
    cmd = launch_plan.planning_environment('/nonexistent/ws', 'pkg', 'mock')
    assert 'if [ -f install/setup.bash ]; then source install/setup.bash; fi' in cmd.script
    done = subprocess.run(cmd.argv(), capture_output=True, text=True, timeout=30)
    assert done.returncode != 0 and 'ros2 launch' not in done.stdout


def test_the_capture_target_shown_is_the_one_replaced(qapp):
    c = Composer(project())
    a, b = c.waypoint_names()[:2]
    chosen = {'name': a}
    live = FakeLive(pose_=([0.4, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0]))
    panel, _, _ = _live_panel(c, live, selected=lambda: chosen['name'])
    panel.target.setCurrentIndex(1)
    chosen['name'] = b                     # the user selects another waypoint: the panel follows
    panel.refresh()
    assert panel.target.currentText() == f"'{b}' (replace it)"
    assert panel.capture_tcp() == b


def test_a_capture_that_changes_the_kind_says_so(qapp):
    c = Composer(project())
    joint_wp = next(w['name'] for w in c.document()['waypoints'] if pose.kind(w) in ('Joints', 'Named'))
    panel, _, _ = _live_panel(c, FakeLive(pose_=([0.4, 0.0, 0.3], [0.0, 0.0, 0.0, 1.0])),
                              selected=lambda: joint_wp)
    panel.target.setCurrentIndex(1)
    assert panel.capture_tcp() == joint_wp
    assert 'it WAS a' in panel.message.text() and 'it is now TCP' in panel.message.text()


def test_untouched_out_of_range_and_relative_values_stay_exact(qapp):
    from trainit_setup_assistant.gui.composer.waypoints_panel import WaypointsPanel
    c = Composer(project())
    n = len(c.project.robot.planning_group.joints)
    c.set_waypoint({'name': 'wide', 'type': 'joint', 'joints': [13.0] + [0.123456789] * (n - 1)})
    first_tcp = next(w['name'] for w in c.document()['waypoints'] if pose.kind(w) == 'TCP')
    c.set_waypoint({'name': 'rel', 'relative': {'to': first_tcp, 'frame': 'tool', 'dz': 0.0123456789,
                                                'dx': -0.0000004}})
    before = c.document()['waypoints']
    panel = WaypointsPanel(lambda: c)
    panel.refresh()
    for name in ('wide', 'rel'):
        panel.select(name)
        assert panel.apply()
    assert c.document()['waypoints'] == before
    panel.select('rel')
    panel.offsets[1].setValue(5.0)                       # dy edited: only dy changes
    assert panel.apply()
    rel = next(w for w in c.document()['waypoints'] if w['name'] == 'rel')['relative']
    assert rel == {'to': first_tcp, 'frame': 'tool', 'dx': -0.0000004, 'dy': 0.005, 'dz': 0.0123456789}


def test_a_launched_command_leads_its_own_group_never_tsas(qapp):
    """vNext I0.11: the group a stop signals is the command's own from the first instant (setsid() in
    the child before exec) -- never TSA's own group."""
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher = Launcher()
    proc = launcher.start(launch_plan.Command('environment', 'env', 'exec sleep 30', True))
    assert os.getpgid(proc.pgid) == proc.pgid != os.getpgrp()
    assert os.getsid(proc.pgid) == proc.pgid
    launcher.stop_all()
    assert not launcher.running('environment')


def test_stop_all_stops_every_group_together_quickly(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher = Launcher()
    procs = [launcher.start(launch_plan.Command(k, k, "trap '' INT; sleep 60 & wait", True))
             for k in ('environment', 'bringup', 'run')]
    time.sleep(0.3)
    started = time.time()
    launcher.stop_all()
    assert time.time() - started < 10.0                       # one shared grace, not 11 s per kind
    assert not any(launcher.running(k) for k in ('environment', 'bringup', 'run'))
    assert all(p.told or p.poll() is not None for p in procs)


def test_each_ending_is_told_once_even_when_a_kind_is_restarted(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher = Launcher()
    ends = []

    class L:
        def started(self, c):
            pass

        def line(self, k, t):
            pass

        def finished(self, k, code):
            ends.append((k, code))
    launcher.add_listener(L())
    for i in range(4):
        proc = launcher.start(launch_plan.Command('build', 'b', f'exit {i}', False))
        assert _wait(qapp, lambda: proc.told)
    assert ends == [('build', i) for i in range(4)]


def test_the_preview_tries_several_starts_and_keeps_the_solution_nearest_the_robot(ros_graph):
    """Owner Test D (friction row 33): from the mock's stretched all-zero start the one IK attempt failed on a
    pose four other starts reached. The Preview now starts from the current joints, the group's named states and
    the program's joint waypoints, and keeps the solution nearest the robot's current joints."""
    from moveit_msgs.srv import GetPositionIK
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.preview import RvizPreview
    from trainit_setup_assistant.model.robot import NamedState
    p = project()
    joints = list(p.robot.planning_group.joints)
    p.robot.named_states = [NamedState(name='far', group=p.robot.planning_group.name,
                                       joint_values={j: 1.0 for j in joints}),
                            NamedState(name='near', group=p.robot.planning_group.name,
                                       joint_values={j: 0.2 for j in joints})]
    node = ros_graph.create_node('fake_move_group_ik')
    js = node.create_publisher(JointState, '/joint_states', 10)
    seeds = []

    def ik(req, res):
        seed = list(req.ik_request.robot_state.joint_state.position)[:len(joints)]
        seeds.append(seed)
        if all(abs(v) < 1e-9 for v in seed):                 # the all-zero start: no solution
            res.error_code.val = -31
            return res
        res.error_code.val = 1
        res.solution.joint_state.name = joints
        res.solution.joint_state.position = [v + 0.01 for v in seed]   # a solution near the start
        return res
    node.create_service(GetPositionIK, '/compute_ik', ik)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            js.publish(JointState(name=joints, position=[0.0] * len(joints)))
            ex.spin_once(timeout_sec=0.05)
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    live = LiveCapture('i0_4_preview_starts')
    try:
        assert _wait(None, lambda: live.joint_states_age() is not None, 5.0)
        sol = RvizPreview(live).ik(p, [0.4, 0.0, 0.3], [1.0, 0.0, 0.0, 0.0])
        assert [sol[j] for j in joints] == pytest.approx([0.21] * len(joints))   # 'near', not 'far'
        assert len(seeds) >= 3 and all(abs(v) < 1e-9 for v in seeds[0])        # the current joints first
    finally:
        live.close()
        stop.set()
        t.join(timeout=2)
        node.destroy_node()
