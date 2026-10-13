"""A GENERATED bundle, built and brought up for real -- then asked to plan and move (vNext M0D T2).

Every other TSA test reads generated files. This one hands them to colcon and to ROS:

    generate (a temp dir) -> colcon build (a temp workspace over the sourced one)
      -> ros2 launch <app> bringup.launch.py mode:=mock rviz:=false bt:=false
      -> the motion stack is up: move_group, the scene manager, the controllers
      -> the scene the planner holds is the generated scene.yaml, object for object
      -> plan to a named state through move_group, and EXECUTE it on the mock controllers
      -> SIGINT, and every process the launch started is gone

Two bundles, the two flows that exist:

  * ``root``  -- the reference cell's project (the golden's), scene-loader flow, at the root;
  * ``r3``    -- the from-scratch example under ``cell_namespace: r3``: the first time a
                 namespaced cell is brought up from a GENERATED bundle (TESTING.md said it never
                 had been), and the runtime half of vNext M0C #5 executed.

OPT-IN: it builds and launches for minutes, so it runs only with ``TRAINIT_LAUNCH_TEST=1`` and a
sourced workspace (ROS 2 Humble + this workspace's install). Everything lives in a temp dir that
is deleted; nothing is written into the workspace. This is test-time generation, like every
golden gate here -- not a bundle for anyone to ship (MANUAL TSA GENERATION CHECKPOINT stands).

The launched graph runs on its own ROS_DOMAIN_ID, so it neither sees nor disturbs anything else
running on the machine.
"""
import os
import re
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

import ws_paths

pytestmark = pytest.mark.skipif(os.environ.get('TRAINIT_LAUNCH_TEST') != '1',
                                reason='opt-in: set TRAINIT_LAUNCH_TEST=1 (builds and launches)')

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'fr3wml_project.yaml'
DOMAIN = '87'
CASES = {
    'root': dict(source=ws_paths.GOLDEN_BUNDLE / 'project.yaml', prefix='t2root', ns=''),
    'r3': dict(source=EXAMPLE, prefix='t2ns', ns='r3'),
}


def _shell(cmd: str, *, cwd=None, timeout=900, env=None) -> subprocess.CompletedProcess:
    return subprocess.run(['bash', '-c', cmd], cwd=cwd, capture_output=True, text=True,
                          timeout=timeout, env=env)


def _generate_and_build(case: dict, ws: Path) -> dict:
    from trainit_setup_assistant.generator import Orchestrator
    from trainit_setup_assistant.model import BundleSpec, load_project
    p = load_project(str(case['source']))
    p.bundle = BundleSpec.from_prefix(case['prefix'])
    p.robot.cell_namespace = case['ns']
    Orchestrator().generate(p, ws / 'src')
    underlay = os.environ.get('COLCON_PREFIX_PATH', '').split(os.pathsep)[0]
    r = _shell(f'source /opt/ros/{os.environ.get("ROS_DISTRO", "humble")}/setup.bash && '
               f'source {underlay}/setup.bash && colcon build --event-handlers console_cohesion- '
               f'2>&1 | tail -5', cwd=ws)
    assert 'Summary:' in r.stdout and 'failed' not in r.stdout, r.stdout + r.stderr
    srdf = next((ws / 'src' / p.bundle.moveit_config_package / 'config').glob('*.srdf'))
    scene = ws / 'src' / p.bundle.moveit_config_package / 'config' / 'scene.yaml'
    return dict(project=p, srdf=srdf,
                scene_objects=(len(yaml.safe_load(scene.read_text())
                                   [next(iter(yaml.safe_load(scene.read_text())))]
                                   ['ros__parameters']['object_ids'])
                               if scene.is_file() else None),
                underlay=underlay)


def _named_state(srdf: Path, group: str) -> tuple:
    root = ET.parse(srdf).getroot()
    for name in ('home', 'ready', 'up'):
        for gs in root.findall('group_state'):
            if gs.get('name') == name and gs.get('group') == group:
                return name, {j.get('name'): float(j.get('value').split()[0])
                              for j in gs.findall('joint')}
    raise AssertionError(f'no home/ready/up state for group {group} in {srdf}')


@pytest.fixture(params=sorted(CASES), scope='module')
def brought_up(request):
    case = CASES[request.param]
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        built = _generate_and_build(case, ws)
        log = ws / 'launch.log'
        env = dict(os.environ, ROS_DOMAIN_ID=DOMAIN, RCUTILS_COLORIZED_OUTPUT='0')
        cmd = (f'source /opt/ros/{os.environ.get("ROS_DISTRO", "humble")}/setup.bash && '
               f'source {built["underlay"]}/setup.bash && source {ws}/install/setup.bash && '
               f'exec ros2 launch {built["project"].bundle.app_package} bringup.launch.py '
               f'mode:=mock rviz:=false bt:=false > {log} 2>&1')
        proc = subprocess.Popen(['bash', '-c', cmd], env=env, start_new_session=True)
        try:
            yield dict(case=case, ns=('/' + case['ns']) if case['ns'] else '', log=log,
                       proc=proc, **built)
        finally:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGINT)
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()


def _wait_log(log: Path, pattern: str, timeout_s: float, proc) -> str:
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        text = log.read_text(errors='replace') if log.exists() else ''
        m = re.search(pattern, text)
        if m:
            return m.group(0)
        if proc.poll() is not None:
            break
        time.sleep(0.5)
    raise AssertionError(f'{pattern!r} not in the launch log within {timeout_s}s:\n' +
                         (log.read_text(errors='replace')[-4000:] if log.exists() else ''))


def test_the_bundle_brings_up_plans_moves_and_shuts_down(brought_up):
    b = brought_up
    ns, log, proc = b['ns'], b['log'], b['proc']
    _wait_log(log, r'You can start planning now', 120, proc)
    if b['scene_objects'] is not None:
        _wait_log(log, r'\[INITIAL scene\] OK \(\d+ objects\)', 60, proc)

    previous_domain = os.environ.get('ROS_DOMAIN_ID')
    os.environ['ROS_DOMAIN_ID'] = DOMAIN               # this process joins the launched graph
    import rclpy
    from rclpy.action import ActionClient
    from moveit_msgs.action import ExecuteTrajectory
    from moveit_msgs.msg import Constraints, JointConstraint, MotionPlanRequest, PlanningSceneComponents
    from moveit_msgs.srv import GetMotionPlan, GetPlanningScene
    from sensor_msgs.msg import JointState
    rclpy.init()
    node = rclpy.create_node('t2_probe')
    try:
        # --- the graph -------------------------------------------------------------------
        wanted = {f'{ns}/{n}' for n in ('move_group', 'robot_state_publisher', 'controller_manager')}
        if b['scene_objects'] is not None:
            wanted.add(f'{ns}/scene_manager_node')
        end = time.monotonic() + 30
        names = []
        while time.monotonic() < end:                  # discovery is not instantaneous
            names = [f'{n_ns.rstrip("/")}/{n}' for n, n_ns in node.get_node_names_and_namespaces()]
            if wanted <= set(names):
                break
            rclpy.spin_once(node, timeout_sec=0.2)
        assert wanted <= set(names), (sorted(wanted - set(names)), sorted(names),
                                      log.read_text(errors='replace')[-3000:])
        print(f'\n[T2 {b["case"]["prefix"]}] nodes in {ns or "/"}:',
              sorted(n for n in names if n.startswith(ns + "/")))

        def call(srv_type, name, req, timeout=30.0):
            cli = node.create_client(srv_type, name)
            assert cli.wait_for_service(timeout_sec=timeout), f'{name} never appeared'
            fut = cli.call_async(req)
            rclpy.spin_until_future_complete(node, fut, timeout_sec=timeout)
            assert fut.done(), f'{name} did not answer'
            return fut.result()

        # --- the scene the planner holds IS the generated scene.yaml ------------------------
        if b['scene_objects'] is not None:
            req = GetPlanningScene.Request()
            req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
            got = call(GetPlanningScene, f'{ns}/get_planning_scene', req)
            assert len(got.scene.world.collision_objects) == b['scene_objects']
            print(f'[T2] planning scene: {len(got.scene.world.collision_objects)} objects '
                  f'== scene.yaml {b["scene_objects"]}')

        # --- the controllers are ACTIVE before anything is sent to them -------------------
        # move_group says "You can start planning now" before the spawners have activated the
        # arm controller; a trajectory sent in between fails with CONTROL_FAILED (-4), which
        # is what the app bring-up's bt_delay waits out. Wait for the fact, not for a delay.
        from controller_manager_msgs.srv import ListControllers
        arm = b['project'].robot.arm_controller.name
        end = time.monotonic() + 60
        state = None
        while time.monotonic() < end and state != 'active':
            listed = call(ListControllers, f'{ns}/controller_manager/list_controllers',
                          ListControllers.Request())
            state = next((c.state for c in listed.controller if c.name == arm), None)
            if state != 'active':
                time.sleep(0.5)
        assert state == 'active', (arm, state)

        # --- plan to a named state, then EXECUTE it on the mock controllers ----------------
        group = b['project'].robot.planning_group.name
        state_name, target = _named_state(b['srdf'], group)
        mpr = MotionPlanRequest(group_name=group, num_planning_attempts=3,
                                allowed_planning_time=10.0,
                                max_velocity_scaling_factor=0.5,
                                max_acceleration_scaling_factor=0.5)
        mpr.goal_constraints = [Constraints(joint_constraints=[
            JointConstraint(joint_name=j, position=v, tolerance_above=1e-3, tolerance_below=1e-3,
                            weight=1.0) for j, v in target.items()])]
        plan = call(GetMotionPlan, f'{ns}/plan_kinematic_path',
                    GetMotionPlan.Request(motion_plan_request=mpr))
        assert plan.motion_plan_response.error_code.val == 1, plan.motion_plan_response.error_code
        traj = plan.motion_plan_response.trajectory
        print(f'[T2] planned to "{state_name}" (group {group}): '
              f'{len(traj.joint_trajectory.points)} points')
        ac = ActionClient(node, ExecuteTrajectory, f'{ns}/execute_trajectory')
        assert ac.wait_for_server(timeout_sec=30.0)
        gf = ac.send_goal_async(ExecuteTrajectory.Goal(trajectory=traj))
        rclpy.spin_until_future_complete(node, gf, timeout_sec=30.0)
        rf = gf.result().get_result_async()
        rclpy.spin_until_future_complete(node, rf, timeout_sec=60.0)
        assert rf.result().result.error_code.val == 1, rf.result().result.error_code

        last = {}
        node.create_subscription(JointState, f'{ns}/joint_states',
                                 lambda m: last.update(zip(m.name, m.position)), 10)
        end = time.monotonic() + 10
        while time.monotonic() < end and not all(j in last for j in target):
            rclpy.spin_once(node, timeout_sec=0.2)
        for j, v in target.items():
            assert abs(last[j] - v) < 2e-2, (state_name, j, last.get(j), v)
        print(f'[T2] executed: max |joint - target| = '
              f'{max(abs(last[j] - v) for j, v in target.items()):.2e} rad')
    finally:
        node.destroy_node()
        rclpy.shutdown()
        if previous_domain is None:
            os.environ.pop('ROS_DOMAIN_ID', None)
        else:
            os.environ['ROS_DOMAIN_ID'] = previous_domain

    # --- clean shutdown: SIGINT, and nothing the launch started survives -------------------
    os.killpg(proc.pid, signal.SIGINT)
    assert proc.wait(timeout=30) is not None
    end = time.monotonic() + 15
    alive = True
    while alive and time.monotonic() < end:
        try:
            os.killpg(proc.pid, 0)                     # anything left in the launch's group?
            time.sleep(0.5)
        except ProcessLookupError:
            alive = False
    assert not alive, 'processes of the launch outlived its SIGINT:\n' + _shell(
        f'ps -o pid,cmd -g {proc.pid}').stdout
