"""P0.a, LAUNCHED (opt-in): the public sample's ROBOT TASKS -- ``examples/ur5e_suction/robot_tasks.yaml`` -- generated
by the TSA under test into a temp workspace, built, brought up on mock, SERVED by the robot task server and driven by
TMR's fake PLC, the way the owner acceptance does it (ADR-0032; overlay docs/engineering/P0_CHECKPOINT.md, OA1-OA8).

    generate the sample the README's way (from its directory), renamed to ptask_* so it never collides with a bundle
      the user built -> colcon build over the sourced workspace: 3 packages
      -> ``ros2 launch ptask_app bringup.launch.py mode:=mock rviz:=false``: move_group ready, the arm controller
         active, the trainit_mock adapter for /eef/main up
      -> ``ros2 launch ptask_app trainit_task_server.launch.py use_sim_time:=false``: the server serves the three
         exported tasks and starts FAULTED / SERVER_STARTED
      -> ``ros2 run trainit_motion_runtime trainit_fake_plc -p tasks_file:=<the bundle's tasks.yaml>`` with a script:
         RESET -> ready; home (job 1) SUCCEEDED OK; an unknown task REJECTED UNKNOWN_TASK; pick_part (job 3)
         SUCCEEDED OK -- it ends HOLDING the part; place_part (job 4) cancelled while the part is held: FAILED
         CANCELLED_WHILE_HOLDING (307), FAULTED (fail closed); RESET; home (job 5) SUCCEEDED OK; job 5 again
         REJECTED DUPLICATE_JOB
      -> every launch ends within 30 s of its SIGINT; ROS domain 81 holds no node before and none after

OPT-IN: ``TRAINIT_LAUNCH_TEST=1`` and a sourced workspace (ROS 2 Humble + the TrainIt workspace with
trainit_motion_runtime >= 2.10.0). It skips, naming the cause, with an older package than the bundle declares, without
ur_description and xacro, or when the machine lacks what the bundle declares (``launch_prereqs``). Test-time
generation in a temp workspace, removed at teardown (owner decision 2026-10-06). With ``TRAINIT_P0A_EVIDENCE=<dir>``
the report and every log are copied to ``<dir>/p0a_robot_tasks/``. NOT a safety function (ADR-0033).
"""
import os
import re
import shlex
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TRAINIT_LAUNCH_TEST') != '1',
                                reason='opt-in: set TRAINIT_LAUNCH_TEST=1 (builds and launches)')

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'ur5e_suction'
PREFIX = 'ptask'
DOMAIN = 81
EVIDENCE = 'TRAINIT_P0A_EVIDENCE'
SCRIPT = ('sleep 1500; reset; sleep 800; '
          'start home 1; sleep 6000; '
          'start 99 2; '
          'start pick_part 3; sleep 9000; '
          'start place_part 4; sleep 4000; cancel 4; sleep 4000; status; '
          'reset; sleep 800; '
          'start home 5; sleep 8000; '
          'start home 5; sleep 500; status')
EXPECTED = [                                            # the fake PLC's plain words, in this order
    'Robot needs reset (server started)',
    '-> RESET',
    'Robot ready',
    'Job 1 (home) finished: SUCCEEDED OK (100)',
    'REJECTED: START -- UNKNOWN_TASK (401)',
    'Job 3 (pick_part) finished: SUCCEEDED OK (100)',
    'Job 4 (place_part) finished: FAILED CANCELLED_WHILE_HOLDING (307)',
    'Robot FAULTED: CANCELLED_WHILE_HOLDING (307)',
    '-> RESET',
    'Robot ready',
    'Job 5 (home) finished: SUCCEEDED OK (100)',
    'REJECTED: START -- DUPLICATE_JOB (405)',
]
# the RESET's own reply and the "Robot ready" it causes come from two threads: their relative order is not fixed
ACKNOWLEDGED = ('ACCEPTED: RESET', 2)
NEVER = ('REFUSED', 'JOB LOST', 'Robot LOST', 'exit 2', 'internal error', 'PLAN FAILED', 'VALIDATION FAILED')


def _cl():
    import composite_launch as CL
    return CL


class ServedLaunch:
    """The bring-up, the robot task server, then the fake PLC's script, on the bundle built in ws."""

    def __init__(self, ws: Path):
        CL = _cl()
        self.ws = CL._outside_the_underlay(ws)
        self.logs, self.tmp, self.ros_log = self.ws / 'logs', self.ws / 'tmp', self.ws / 'ros_log'
        for d in (self.logs, self.tmp, self.ros_log):
            d.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), ROS_LOCALHOST_ONLY='1',
                        RCUTILS_COLORIZED_OUTPUT='0', ROS_LOG_DIR=str(self.ros_log), TMPDIR=str(self.tmp))
        self.lines, self.guards, self.stops, self.ready = [], {}, {}, {}
        self.plc_rc = None
        self.plc = self.server = ''

    def say(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def report(self) -> str:
        return '\n'.join(self.lines) + '\n'

    def save(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'report.txt').write_text(self.report())
        for log in sorted(self.logs.glob('*.log')):
            shutil.copy(log, dest / log.name)

    def _start(self, command: str, log: Path) -> subprocess.Popen:
        CL = _cl()
        cmd = CL._sourced(str(self.ws / 'install' / 'setup.bash')) + f' && exec {command} > {shlex.quote(str(log))} 2>&1'
        return subprocess.Popen(['bash', '-c', cmd], env=self.env, start_new_session=True, stdin=subprocess.DEVNULL)

    def _guard(self, tag: str) -> list:
        CL = _cl()
        probe = CL._Probe(DOMAIN, f'ptask_guard_{len(self.guards) + 1}', self.ros_log)
        try:
            time.sleep(CL.SETTLE_S)
            nodes = [f'{ns.rstrip("/")}/{n}' for ns, n in probe.nodes()]
        finally:
            probe.close()
        self.guards[tag] = nodes
        self.say(f'[guard] ROS domain {DOMAIN} {tag} the launches: {nodes or "no node"}')
        return nodes

    def run(self) -> 'ServedLaunch':
        CL = _cl()
        if self._guard('before'):
            raise RuntimeError(f'ROS domain {DOMAIN} is not empty: refusing to launch on it')
        up = server = None
        bringup, server_log, plc_log = self.logs / 'bringup.log', self.logs / 'server.log', self.logs / 'fake_plc.log'
        tasks = self.ws / 'install' / f'{PREFIX}_app' / 'share' / f'{PREFIX}_app' / 'config' / 'tasks.yaml'
        try:
            up = self._start(f'ros2 launch {PREFIX}_app bringup.launch.py mode:=mock rviz:=false', bringup)
            self.ready = {
                'move_group ready': CL._wait_log(bringup, r'You can start planning now', 120),
                'arm controller active': CL._wait_log(bringup, r'Configured and activated .*moveit_joint_controller', 60),
                'mock adapter up': CL._wait_log(bringup, re.escape('trainit_mock adapter for /eef/main'), 60)}
            self.say(f'[bring-up] {self.ready}')
            if all(self.ready.values()):
                time.sleep(2.0)
                server = self._start(f'ros2 launch {PREFIX}_app trainit_task_server.launch.py use_sim_time:=false',
                                     server_log)
                self.ready['server serving'] = CL._wait_log(server_log, r'RTI v1: server session', 90)
                self.say(f'[server] serving: {self.ready["server serving"]}')
                if self.ready['server serving']:
                    plc = self._start('ros2 run trainit_motion_runtime trainit_fake_plc --ros-args '
                                      f'-p tasks_file:={shlex.quote(str(tasks))} -p script:={shlex.quote(SCRIPT)}',
                                      plc_log)
                    try:
                        self.plc_rc = plc.wait(timeout=120)
                    except subprocess.TimeoutExpired:
                        self.stops['fake PLC'] = CL.CompositeLaunch._stop_outcome(plc)
        finally:
            if server is not None:
                self.stops['server'] = CL.CompositeLaunch._stop_outcome(server)
            if up is not None:
                self.stops['bring-up'] = CL.CompositeLaunch._stop_outcome(up)
            self._guard('after')
        self.plc = plc_log.read_text(errors='replace') if plc_log.is_file() else ''
        self.server = server_log.read_text(errors='replace') if server_log.is_file() else ''
        self.said = re.findall(r'^\[fake PLC [0-9:.]+\] (.*)$', self.plc, re.M)
        self.never = sorted(n for n in NEVER if n in self.plc or n in self.server)
        for line in (f'[fake PLC] exit {self.plc_rc}; said {self.said}',
                     f'[logs] lines that must never appear: {self.never or "none"}',
                     f'[shutdown] {self.stops}'):
            self.say(line)
        return self


def _in_order(said: list, expected: list) -> list:
    """The expected lines missing from `said` in this order (an empty list: all present, in order)."""
    missing, k = [], 0
    for want in expected:
        found = next((i for i in range(k, len(said)) if want in said[i]), None)
        if found is None:
            missing.append(want)
        else:
            k = found + 1
    return missing


@pytest.fixture(scope='module')
def served(tmp_path_factory):
    CL = _cl()
    if not os.environ.get('COLCON_PREFIX_PATH'):
        pytest.skip('needs a sourced workspace (ROS 2 Humble + the TrainIt workspace install)')
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.generator import Orchestrator
    from trainit_setup_assistant.model import BundleSpec, load_project
    tmp = tmp_path_factory.mktemp('p0a_ws')
    try:
        ws = CL._outside_the_underlay(tmp)
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(EXAMPLE)
            project = load_project('robot_tasks.yaml')
            project.bundle = BundleSpec.from_prefix(PREFIX)
            manifest = Orchestrator().generate(project, ws / 'src' / PREFIX)
        app_xml = ws / 'src' / PREFIX / f'{PREFIX}_app' / 'package.xml'
        floors = {e.text.strip(): tuple(int(x) for x in e.get('version_gte').split('.'))
                  for e in ET.parse(app_xml).getroot().iter('exec_depend') if e.get('version_gte')}
        for package, floor in sorted(floors.items()):
            have = CL.installed_version(package)
            if have < floor:
                pytest.skip(f'{package} {CL._v(have)} is installed; the bundle declares >= {CL._v(floor)}')
        import launch_prereqs
        launch_prereqs.require(ws / 'src', modules=CL.PROBE_MODULES)
        summary = CL.build(ws)
        run = ServedLaunch(ws)
        try:
            run.run()
        finally:
            if os.environ.get(EVIDENCE):
                run.save(Path(os.environ[EVIDENCE]) / 'p0a_robot_tasks')
        yield dict(manifest=manifest, summary=summary, run=run, floors=floors)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _why(served) -> str:
    return '\n' + served['run'].report()


def test_p0a_the_robot_tasks_generate_and_build(served):
    assert served['manifest'].warnings == []
    assert '3 packages finished' in served['summary'], served['summary']
    assert served['floors'].get('trainit_motion_runtime', (0,)) >= (2, 10, 0), served['floors']


def test_p0a_the_server_serves_the_bundle_on_mock(served):
    run = served['run']
    assert all(run.ready.values()), _why(served)
    assert '3 exported task(s)' in run.server, _why(served)
    assert 'FAULTED / SERVER_STARTED until a RESET' in run.server, _why(served)


def test_p0a_the_fake_plc_scenario_gives_the_contracts_results(served):
    run = served['run']
    assert run.plc_rc == 0, _why(served)
    assert _in_order(run.said, EXPECTED) == [], _why(served)
    assert sum(line == ACKNOWLEDGED[0] for line in run.said) == ACKNOWLEDGED[1], _why(served)
    # one line per write (the fake PLC's two threads never interleave inside a line)
    assert not any('[fake PLC' in line for line in run.said), _why(served)
    assert run.never == [], _why(served)


def test_p0a_every_launch_ends_cleanly_and_leaves_the_domain_empty(served):
    run = served['run']
    assert run.stops == {'server': 'clean', 'bring-up': 'clean'}, _why(served)
    assert run.guards.get('after') == [], _why(served)
