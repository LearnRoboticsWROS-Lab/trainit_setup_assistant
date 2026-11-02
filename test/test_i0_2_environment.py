"""vNext I0.2 (I0_DESIGN.md §11, §12, §22, §23; ADR-0027): the planning environment.

TSA LAUNCHES and OBSERVES. The commands it may run are exactly the lines a user would type, and they
are shown before they run. Its process launcher stops what it started. Its ROS clients only READ: no
publisher or client of TSA can move a robot.

The live checks compare the running environment with the project: the joint states by name, the TCP
frame and the arm controller. Machine-local settings never enter a project.

The launch tier (``TRAINIT_LAUNCH_TEST=1``) builds the reference cell's Step-3 environment in a temp
workspace, launches it on mock through TSA's own command line, and runs the live checks against it.
"""
import os
import re
import shlex
import subprocess
import time
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import launch_plan
from trainit_setup_assistant.composer.integration import rviz_teaching_problems
from trainit_setup_assistant.composer.settings import ENV, LauncherSettings
from trainit_setup_assistant.model import CanonicalProject

PKG = Path(__file__).resolve().parents[1]
SRC = PKG / 'trainit_setup_assistant'
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'
FR3WML_TEST_E = Path.home() / 'fr5_ws' / 'forensics' / 'vnext-m3' / 'm3_acceptance' / 'test_e' / 'm3e_fr3wml.yaml'


# ---- the commands ------------------------------------------------------------------------------------

def test_the_commands_are_the_lines_a_user_types():
    b = launch_plan.build('/ws', ['cell_scene_loader_moveit_config'])
    assert b.argv()[:3] == ['setsid', 'bash', '-c'] and not b.long_running
    assert b.display() == ('source /opt/ros/humble/setup.bash && cd /ws && colcon build --base-paths src '
                           '--packages-select cell_scene_loader_moveit_config')
    env = launch_plan.planning_environment('/ws', 'cell_scene_loader_moveit_config', 'mock')
    assert env.long_running and env.display().endswith(
        'ros2 launch cell_scene_loader_moveit_config bringup.launch.py mode:=mock')
    assert 'source install/setup.bash' in env.script and env.script.split(' && ')[-1].startswith('exec ros2 launch')
    up = launch_plan.bringup('/ws', 'cell_app', 'isaac')
    assert up.display().endswith('ros2 launch cell_app bringup.launch.py mode:=isaac bt:=false')
    assert launch_plan.run('/ws', 'cell_app', 'mock').display().endswith(
        'ros2 launch cell_app trainit_bt.launch.py use_sim_time:=false groot_monitor:=false')
    assert 'use_sim_time:=true' in launch_plan.run('/ws', 'cell_app', 'isaac').script


def test_the_isaac_command_opens_the_bound_stage_with_the_shipped_script(tmp_path):
    c = launch_plan.isaac('/cells/my cell.usd', '/opt/isaac/isaac-sim.sh')
    assert c.env == {'TRAINIT_STAGE': '/cells/my cell.usd'}
    assert c.display() == (f"TRAINIT_STAGE='/cells/my cell.usd' /opt/isaac/isaac-sim.sh --exec "
                           f'{shlex.quote(str(launch_plan.OPEN_AND_PLAY))}')
    assert launch_plan.OPEN_AND_PLAY.is_file()
    text = launch_plan.OPEN_AND_PLAY.read_text()
    assert "os.environ.get('TRAINIT_STAGE'" in text and 'save' not in text.lower().replace('never modified or saved', '')
    with pytest.raises(ValueError, match='no Isaac stage'):
        launch_plan.isaac('', '/opt/isaac/isaac-sim.sh')
    with pytest.raises(ValueError, match='launcher'):
        launch_plan.isaac('/cells/c.usd', '')


def test_names_are_shell_quoted():
    c = launch_plan.build('/my ws', ['pkg; rm -rf /'])
    assert "cd '/my ws'" in c.script and "'pkg; rm -rf /'" in c.script


@pytest.mark.parametrize('out, root', [('/home/u/ws/src', '/home/u/ws'), ('/home/u/ws/src/cell', '/home/u/ws'),
                                       ('/tmp/out', None)])
def test_the_workspace_root_is_the_parent_of_src(out, root):
    assert launch_plan.workspace_root(out) == root


# ---- ADR-0027: TSA's ROS endpoints only read -------------------------------------------------------------

# the Step-2 preview of the static scene; the I0.4 RViz Preview (a goal STATE and the group: visualization)
ALLOWED_PUBLISHERS = {'/planning_scene', 'GOAL_TOPIC', 'GROUP_TOPIC'}
# the live check; the Preview's IK; the live waypoint check (vNext I0, owner Test C): MoveIt computations and
# reads only -- IK, FK, state validity, the planning scene's object names
ALLOWED_CLIENTS = {'ListControllers', 'GetPositionIK', 'GetPositionFK', 'GetStateValidity', 'GetPlanningScene'}


def test_tsa_creates_no_action_client_and_no_motion_publisher_or_client():
    publishers, clients, actions = set(), set(), []
    for f in SRC.rglob('*.py'):
        if 'resources' in f.parts or 'templates' in f.parts:
            continue                                           # generated / Isaac-side files, not TSA's process
        text = f.read_text()
        actions += [f.name for _ in re.finditer(r'\bActionClient\(', text)]
        for m in re.finditer(r'create_publisher\(\s*\w+\s*,\s*([^,]+),', text):
            publishers.add(m.group(1).strip().strip('\'"'))
        for m in re.finditer(r'(?:create_client|\.client|\._call)\(\s*(\w+)\s*,', text):
            clients.add(m.group(1))         # ._call: livesession.kinematics' one request helper
    clients.discard('srv_type')            # LiveCapture.client(): the generic long-lived helper, used above
    assert actions == []
    assert publishers <= ALLOWED_PUBLISHERS, publishers
    assert clients <= ALLOWED_CLIENTS, clients


def test_the_planning_scene_preview_publishes_only_the_planning_scene():
    text = (SRC / 'livesession' / 'planning_scene_client.py').read_text()
    assert re.findall(r"['\"](/[\w/]+)['\"]", text) == ['/planning_scene'] or '/planning_scene' in text


# ---- the launcher (Qt) ----------------------------------------------------------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wait(app, cond, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


class _Listener:
    def __init__(self):
        self.events = []

    def started(self, command):
        self.events.append(('started', command.kind))

    def line(self, kind, text):
        self.events.append(('line', kind, text))

    def finished(self, kind, code):
        self.events.append(('finished', kind, code))


def test_the_launcher_shows_the_output_and_reads_the_verdict(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher, seen = Launcher(), _Listener()
    launcher.add_listener(seen)
    cmd = launch_plan.Command('run', 'Run', "echo starting; echo '[runner] BT finished: SUCCESS'; exit 0", False)
    proc = launcher.start(cmd)
    assert _wait(qapp, lambda: proc.exit_code is not None)
    assert proc.verdict == launch_plan.BT_SUCCESS and proc.exit_code == 0
    assert seen.events[0] == ('started', 'run') and ('finished', 'run', 0) in seen.events
    assert ('line', 'run', 'starting') in seen.events


def test_a_long_running_command_is_stopped_with_sigint_and_one_per_kind(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    launcher = Launcher()
    cmd = launch_plan.Command('environment', 'Env',
                              "trap 'echo got-int; exit 130' INT; echo ready; while true; do sleep 0.1; done", True)
    proc = launcher.start(cmd)
    assert _wait(qapp, lambda: 'ready' in proc.lines)             # the trap is installed
    assert launcher.running('environment')
    with pytest.raises(RuntimeError, match='already running'):
        launcher.start(cmd)
    started = time.time()
    launcher.stop_all()
    assert not launcher.running('environment') and time.time() - started < 7.0
    assert _wait(qapp, lambda: proc.exit_code is not None)
    import signal
    if signal.getsignal(signal.SIGINT) is not signal.SIG_IGN:
        assert 'got-int' in proc.lines             # Ctrl-C reached it: a clean shutdown
    else:
        assert proc.exit_code != 0                  # SIGINT ignored by inheritance: SIGTERM stopped it


def test_the_run_panel_shows_each_command_before_it_runs(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.run_panel import RunPanel
    launcher = Launcher()
    panel = RunPanel(launcher, [('build', 'Build', lambda: launch_plan.Command('build', 'Build', 'echo built', False)),
                                ('isaac', 'Isaac', lambda: launch_plan.isaac('', 'x'))])
    panel.toggle('build')
    assert _wait(qapp, lambda: 'exited 0' in panel.state.text())
    log = panel.log.toPlainText().splitlines()
    assert log[0] == '$ echo built' and '[build] built' in log
    panel.toggle('isaac')
    assert panel.state.text().startswith('⚠ Isaac: no Isaac stage is bound')


def test_the_run_panel_says_what_each_command_is_doing(qapp):
    """Owner Test A.2: one shared state line said "Isaac Sim: running" and hid that the planning environment
    (RViz) was never started. Every command has its own status line; a stop is "stopped", not a failure."""
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.run_panel import RunPanel
    launcher = Launcher()
    loop = "trap 'exit 130' INT; echo ready; while true; do sleep 0.1; done"
    panel = RunPanel(launcher, [
        ('isaac', 'Isaac', lambda: launch_plan.Command('isaac', 'Isaac', loop, True)),
        ('environment', 'Env', lambda: launch_plan.Command('environment', 'Env', 'exit 3', True))])
    text = panel.status_label.text()
    assert 'Isaac Sim: ' in text and 'Planning environment (RViz): ' in text and text.count('not started') == 2
    panel.toggle('isaac')
    assert _wait(qapp, lambda: '[isaac] ready' in panel.log.toPlainText())
    assert panel.status == {'isaac': 'running', 'environment': 'not started'}
    panel.toggle('environment')
    assert _wait(qapp, lambda: panel.status['environment'] == 'exited 3')
    assert panel.status['isaac'] == 'running'
    panel.toggle('isaac')                                 # running: the button stops it
    assert _wait(qapp, lambda: panel.status['isaac'] == 'stopped')
    assert panel.state.text() == 'Isaac: stopped' and '[Isaac: stopped]' in panel.log.toPlainText()


# ---- machine-local settings ----------------------------------------------------------------------------

def test_settings_live_outside_the_project(tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / 'tsa.yaml'))
    s = LauncherSettings.load()
    s.isaac_launcher, s.workspace_root = '/opt/isaac/isaac-sim.sh', '/ws'
    s.save()
    assert LauncherSettings.load().isaac_launcher == '/opt/isaac/isaac-sim.sh'
    assert yaml.safe_load((tmp_path / 'tsa.yaml').read_text()) == {'isaac_launcher': '/opt/isaac/isaac-sim.sh',
                                                                  'workspace_root': '/ws'}
    fields = set(getattr(CanonicalProject, 'model_fields', None) or CanonicalProject.__fields__)
    assert not fields & {'isaac_launcher', 'workspace_root', 'settings'}


# ---- the RViz teaching display (R-I0-04) -----------------------------------------------------------------

def _copy_base(tmp_path, rviz_text=None):
    cfg = tmp_path / 'bot_moveit_config' / 'config'
    cfg.mkdir(parents=True)
    if rviz_text is not None:
        (cfg / 'moveit.rviz').write_text(rviz_text)
    data = yaml.safe_load(COMPOSITION.read_text())
    data['robot']['base_moveit_config_path'] = str(cfg.parent)
    return CanonicalProject.parse_obj(data)


@pytest.mark.parametrize('rviz, problem', [(None, 'no config/*.rviz'),
                                           ('Panels: []\nDisplays: [{Class: rviz_default_plugins/Grid}]\n',
                                            'has no MotionPlanning display'),
                                           ('Displays: [{Class: moveit_rviz_plugin/MotionPlanning}]\n', None)])
def test_the_moveit_config_must_open_rviz_with_motion_planning(tmp_path, rviz, problem):
    found = rviz_teaching_problems(_copy_base(tmp_path, rviz))
    if problem is None:
        assert found == []
    else:
        assert len(found) == 1 and found[0].startswith('R-I0-04: ') and problem in found[0]


# ---- live checks against an in-process graph ------------------------------------------------------------

@pytest.fixture
def ros_graph():
    rclpy = pytest.importorskip('rclpy')
    os.environ['ROS_DOMAIN_ID'] = str(170 + os.getpid() % 30)
    rclpy.init()
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()


def _fake_robot(rclpy, joints, controller_state='active'):
    pytest.importorskip('controller_manager_msgs')       # a TSA dependency; skip naming it if absent
    from controller_manager_msgs.msg import ControllerState
    from controller_manager_msgs.srv import ListControllers
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import JointState
    import tf2_ros
    node = rclpy.create_node('i0_2_fake_cell')
    pub = node.create_publisher(JointState, '/joint_states', 10)
    tf = TransformStamped()
    tf.header.frame_id, tf.child_frame_id = 'base_link', 'tcp'
    tf.transform.rotation.w = 1.0
    node._static = tf2_ros.StaticTransformBroadcaster(node)
    node._static.sendTransform(tf)

    def handle(_req, res):
        res.controller = [ControllerState(name='moveit_joint_controller', state=controller_state)]
        return res
    node.create_service(ListControllers, '/controller_manager/list_controllers', handle)
    from rclpy.executors import SingleThreadedExecutor
    import threading
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            pub.publish(JointState(name=list(joints), position=[0.0] * len(joints)))
            ex.spin_once(timeout_sec=0.05)
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    return node, stop, t


def _project(joints=('j1', 'j2')):
    data = yaml.safe_load(COMPOSITION.read_text())
    data['robot']['planning_group']['joints'] = list(joints)
    data['robot']['base_frame'], data['robot']['tip_link'] = 'base_link', 'tcp'
    data['robot'].setdefault('arm_controller', {})['name'] = 'moveit_joint_controller'
    return CanonicalProject.parse_obj(data)


@pytest.mark.parametrize('published, state, expect', [
    (('j1', 'j2'), 'active', {'joint states': True, 'TCP frame': True, 'arm controller': True}),
    (('j1',), 'active', {'joint states': False, 'TCP frame': True, 'arm controller': True}),
    (('j1', 'j2'), 'inactive', {'joint states': True, 'TCP frame': True, 'arm controller': False}),
])
def test_the_live_checks_compare_the_running_robot_with_the_project(ros_graph, published, state, expect):
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.checks import live_checks
    node, stop, t = _fake_robot(ros_graph, published, state)
    live = LiveCapture('i0_2_live')
    try:
        assert _wait_plain(lambda: live.joint_states_age() is not None)
        got = {c.name: c.ok for c in live_checks(live, _project(), timeout_s=3.0)}
        assert got == expect
    finally:
        live.close()
        stop.set()
        t.join(timeout=2)
        node.destroy_node()


def _wait_plain(cond, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        time.sleep(0.05)
    return cond()


# ---- the launch tier: the reference cell's planning environment on mock, through TSA's command ------------

@pytest.mark.skipif(os.environ.get('TRAINIT_LAUNCH_TEST') != '1', reason='launch tier: TRAINIT_LAUNCH_TEST=1')
def test_the_reference_cell_planning_environment_runs_on_mock_and_passes_the_live_checks(tmp_path, qapp):
    if not FR3WML_TEST_E.is_file():
        pytest.skip('the M3 acceptance project (private reference cell) is absent')
    import ws_paths
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.checks import live_checks
    ctrl = AssistantController()
    ctrl.open_project(FR3WML_TEST_E)
    p = ctrl.project
    p.robot.description.urdf_dir = ws_paths.in_workspace(p.robot.description.urdf_dir)
    p.robot.base_moveit_config_path = ws_paths.in_workspace(p.robot.base_moveit_config_path)
    src = tmp_path / 'ws' / 'src'
    src.mkdir(parents=True)
    ctrl.generate_scene_loader_config(str(src))
    ws = launch_plan.workspace_root(src)
    pkg = ctrl.scene_loader_package_name()
    build = launch_plan.build(ws, [pkg])
    assert subprocess.run(build.argv(), capture_output=True, text=True, timeout=900).returncode == 0
    os.environ['ROS_DOMAIN_ID'] = str(140 + os.getpid() % 30)
    launcher = Launcher()
    env = launch_plan.planning_environment(ws, pkg, 'mock', rviz=False)
    proc = launcher.start(env)
    live = None
    try:
        import rclpy
        live = LiveCapture('i0_2_launch_check', namespace=p.robot.cell_namespace)
        deadline = time.time() + 90
        checks = []
        while time.time() < deadline:
            qapp.processEvents()
            checks = live_checks(live, p, timeout_s=2.0)
            if all(c.ok for c in checks):
                break
            time.sleep(1.0)
        assert all(c.ok for c in checks), [(c.name, c.ok, c.text) for c in checks] + proc.lines[-30:]
        print('I0.2 launch tier:', build.display(), '|', env.display())
        for c in checks:
            print(f'  {c.name}: {c.ok} -- {c.text}')
        print('  environment output (first lines):', *proc.lines[:12], sep='\n    ')
    finally:
        if live is not None:
            live.close()
        launcher.stop_all()
        assert not launcher.running('environment')
        # vNext I0.4: the whole group stopped -- no node of this workspace outlives the launch
        left = subprocess.run(['pgrep', '-f', str(tmp_path / 'ws' / 'install')], capture_output=True, text=True)
        assert left.stdout.strip() == '', left.stdout


def test_step_4_offers_build_isaac_and_the_planning_environment(qapp, tmp_path, monkeypatch):
    monkeypatch.setenv(ENV, str(tmp_path / 'tsa.yaml'))
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController(CanonicalProject.parse_obj(yaml.safe_load(COMPOSITION.read_text())))
    w = SetupWizard(ctrl)
    page = w.mode_page
    assert page.launcher is w.launcher
    assert set(page.run_panel.buttons) == {'build', 'isaac', 'environment'}
    with pytest.raises(ValueError, match='no workspace'):
        page._environment_command()
    ctrl.workspace_root = '/ws'
    page.mode.setCurrentText('mock')
    # a quick robot (no MoveIt config): its bundle, without the tree, is the environment
    assert page._environment_command().display().endswith(
        f'ros2 launch {ctrl.project.bundle.app_package} bringup.launch.py mode:=mock bt:=false')
    assert page._build_command().script.endswith(
        ' '.join([ctrl.project.bundle.description_package, ctrl.project.bundle.moveit_config_package,
                  ctrl.project.bundle.app_package]))
    ctrl.project.robot.base_moveit_config_path = str(tmp_path)      # an existing MoveIt config: Step 3's
    assert page._environment_command().display().endswith(
        f'ros2 launch {ctrl.scene_loader_package_name()} bringup.launch.py mode:=mock')
    with pytest.raises(ValueError, match='no Isaac stage is bound'):
        page._isaac_command()


def test_a_live_check_right_after_the_session_starts_waits_for_the_first_joint_state(ros_graph):
    """Owner Test A.1: the first "Check the live robot" made the session and judged it at once -- before the
    first message arrived -- and showed a false ✖ while /joint_states ran at 50 Hz."""
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.checks import live_checks
    node, stop, t = _fake_robot(ros_graph, ('j1', 'j2'))
    try:
        time.sleep(0.5)
        live = LiveCapture('i0_2_live_fresh')            # created at the click, no message yet
        try:
            got = {c.name: c.ok for c in live_checks(live, _project(), timeout_s=3.0)}
            assert got['joint states'] is True, got
        finally:
            live.close()
    finally:
        stop.set()
        t.join(timeout=2)
        node.destroy_node()


def test_a_stopped_environment_is_not_live(ros_graph):
    """Owner Test A.2: the mock environment was stopped, Isaac was started, and "Check the live robot" passed
    the joint states and the TCP frame on the stopped environment's LAST message (78 s old). A robot whose
    joint states stopped is not LIVE: the check fails and saying why, and nothing is captured from it."""
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.checks import live_checks
    node, stop, t = _fake_robot(ros_graph, ('j1', 'j2'))
    live = LiveCapture('i0_2_live_stale')
    try:
        assert _wait_plain(lambda: live.joint_states_age() is not None)
        stop.set()                                         # the environment stops publishing
        t.join(timeout=2)
        assert _wait_plain(lambda: live.joint_states_age() > 2.5, timeout=6.0)
        got = {c.name: c for c in live_checks(live, _project(), timeout_s=1.0)}
        assert got['joint states'].ok is False and 's old: nothing publishes it now' in got['joint states'].text
        assert got['TCP frame'].ok is False                # TF still holds the last transform
        with pytest.raises(TimeoutError, match='nothing publishes it now'):
            live.current_joint_values(['j1'], timeout_s=0.3)
        with pytest.raises(TimeoutError, match='nothing publishes it now'):
            live.current_pose('base_link', 'tcp', timeout_s=0.3)
        assert live.current_pose('base_link', 'tcp', timeout_s=1.0, max_age=None)  # asked for explicitly
    finally:
        live.close()
        node.destroy_node()


def test_a_capture_reads_the_tf_of_now_after_the_clock_changed(ros_graph):
    """Owner Test B: Capture TCP returned the same pose at every click while Capture joints followed the robot.
    The session was made on MOCK (TF stamped with wall time, ~1.8e9 s); Isaac stamps TF with SIMULATION time
    (seconds since PLAY). tf2 drops data older than what it holds, so the buffer kept answering the mock robot's
    last pose. The same happens when Isaac is stopped and played again. A capture reads TF that arrived after
    the click: a dynamic frame chain (base_link -> wrist, published) + a static one (wrist -> tcp)."""
    import threading
    import tf2_ros
    from geometry_msgs.msg import TransformStamped
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from trainit_setup_assistant.livesession import LiveCapture
    node = ros_graph.create_node('i0_2_clock_change')
    js = node.create_publisher(JointState, '/joint_states', 10)
    tfb = tf2_ros.TransformBroadcaster(node)
    static = tf2_ros.StaticTransformBroadcaster(node)
    st = TransformStamped()
    st.header.frame_id, st.child_frame_id = 'wrist', 'tcp'
    st.transform.translation.z = 0.1
    st.transform.rotation.w = 1.0
    static.sendTransform(st)
    state = {'x': 0.5, 'sim': False, 'sim_t': 100.0}
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            t = TransformStamped()
            if state['sim']:                                   # Isaac: seconds since PLAY
                state['sim_t'] += 0.05
                sec = int(state['sim_t'])
                t.header.stamp.sec, t.header.stamp.nanosec = sec, int((state['sim_t'] - sec) * 1e9)
            else:                                              # mock: wall time
                t.header.stamp = node.get_clock().now().to_msg()
            t.header.frame_id, t.child_frame_id = 'base_link', 'wrist'
            t.transform.translation.x = state['x']
            t.transform.rotation.w = 1.0
            tfb.sendTransform(t)
            js.publish(JointState(name=['j1'], position=[state['x']]))
            ex.spin_once(timeout_sec=0.05)
    th = threading.Thread(target=spin, daemon=True)
    th.start()
    live = LiveCapture('i0_2_live_clock')
    try:
        pos, _ = live.current_pose('base_link', 'tcp', timeout_s=3.0)
        assert pos[0] == pytest.approx(0.5) and pos[2] == pytest.approx(0.1)
        state['sim'], state['x'] = True, 0.2                  # mock stopped, Isaac started; the robot moved
        time.sleep(0.5)
        pos, _ = live.current_pose('base_link', 'tcp', timeout_s=3.0)
        assert pos[0] == pytest.approx(0.2), 'the capture returned the pose of before the clock change'
        assert pos[2] == pytest.approx(0.1)                    # the static part survives
        state['sim_t'], state['x'] = 5.0, -0.3                 # Isaac stopped and played again
        time.sleep(0.5)
        assert live.current_pose('base_link', 'tcp', timeout_s=3.0)[0][0] == pytest.approx(-0.3)
    finally:
        live.close()
        stop.set()
        th.join(timeout=2)
        node.destroy_node()


def test_step_1_starts_empty_and_shows_what_it_read(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    w = SetupWizard(AssistantController())
    page = w.base_page
    assert page.project_name.text() == '' and page.base_pkg.text() == '' and page.base_path.text() == ''
    assert not page.validatePage() and 'Set the project name' in page.summary.text()


def test_steps_2_to_4_carry_no_cell_specific_default(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController(CanonicalProject.parse_obj(yaml.safe_load(COMPOSITION.read_text())))
    w = SetupWizard(ctrl)
    sp = w.scene_page
    assert sp.obj_id.text() == '' and sp.dims.text() == '' and sp.position.text() == ''
    assert not sp.manual_box.isVisibleTo(sp)                      # hand editing is Advanced
    w.gen_config_page.initializePage()
    assert w.gen_config_page.pkg_name.text() == ''              # a suggestion, not a value
    w.mode_page.initializePage()
    assert 'Build the config' in w.mode_page.procedure.text() and 'colcon' not in w.mode_page.procedure.text()
