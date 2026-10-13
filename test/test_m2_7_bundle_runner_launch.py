"""T17b + T17c (vNext M2.7, owner section 12 "/r1 + /r2 simultaneously"; ADR-0023): the canonical
suction_simple bundle's tree, run by its own runner under a robot namespace on a real move_group.

    generate eef_fixture.suction_simple (launch-safe; the 3-line SetBool contract on the mock
      adapter) with the TSA under test: t17r3c in /r3, t17r1 in /r1 and t17r2 in /r2 -- the same
      robot, the same frames -- each into a temp workspace's src/<name>/
      -> one colcon build of the three bundles over the sourced workspace (sequential, make -j2)
      -> T17b  /r3 alone. The bring-up ``mode:=mock rviz:=false``, then a SEPARATE runner launch
               ``trainit_bt.launch.py use_sim_time:=false groot_monitor:=false`` (a from-scratch
               bring-up declares neither ``bt`` nor ``groot_monitor``): the tree ends SUCCESS
               within 120 s; every MoveIt action and service client of the runner is under /r3;
               /r3's move_group receives the tree's goals (a MoveGroup and an ExecuteTrajectory
               goal per MoveWaypoint)
      -> T17c  /r1 and /r2 at once (two bring-ups, then both runners back to back): both trees
               end SUCCESS; each move_group receives exactly its own tree's goals, through action
               clients that only its own robot's move_group serves; the two runners' endpoint sets
               are disjoint except the cell-global set (/tf, /tf_static, /rosout,
               /parameter_events); no MoveIt name of either runner lies outside its namespace
      -> every launch ends within 30 s of its SIGINT (a runner launch by itself, when its tree
         ends) and no process of it survives; the ROS domain holds no node before the first
         launch and none after the last

BEFORE (TMR 2.0.0, vNext M2.7 forensics): no tree finished within 120 s; the runners' MoveIt
clients and publishers were at the root (/move_action, /trajectory_execution_event,
/attached_collision_object) and no move_group received a goal.

How a runner's endpoints are read. Once the bring-ups are up, the graph is read after 6 s of
discovery: its nodes, and which node serves each action and service. Every node that appears
after the runners start is a runner process's node, attributed by its namespace (a node that
appears anywhere else fails the attribution check); its endpoints are read every 0.3 s while it
lives and united over the run. Node names are compared without the per-process address MoveIt and
tf2 append to their internal nodes.

OPT-IN: it builds and launches for minutes, so it runs only with ``TRAINIT_LAUNCH_TEST=1`` and a
sourced workspace (ROS 2 Humble + this workspace's install). It needs a runner that resolves
MoveIt inside its namespace (trainit_motion_runtime >= 2.1.0): below that every test SKIPS, never
passes. Every launch runs in its own session on ROS domain 84 (sequential: never together with
domain 83), ROS_LOCALHOST_ONLY=1, with TMPDIR (launch_ros writes a launch_params_* file per
parameter dict there) and ROS_LOG_DIR inside the temp workspace, which may not lie inside a
sourced workspace (composite_launch). Test-time generation, not a bundle for anyone to ship
(MANUAL TSA GENERATION CHECKPOINT stands). With ``TRAINIT_M27_LAUNCH_EVIDENCE=<dir>`` each case's
report (normalised: no stamps, pids, times or temp paths), the bundle identities and every launch
log are copied to ``<dir>/bundle_runner/<case>/`` before the temp dir goes.
"""
import os
import re
import shlex
import shutil
import subprocess
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TRAINIT_LAUNCH_TEST') != '1',
                                reason='opt-in: set TRAINIT_LAUNCH_TEST=1 (builds and launches)')

EVIDENCE = 'TRAINIT_M27_LAUNCH_EVIDENCE'
DOMAIN = 84
DEV = 'suction_simple'
CASES = {'t17b': (('t17r3c', '/r3'),),
         't17c': (('t17r1', '/r1'), ('t17r2', '/r2'))}
TREE_TIMEOUT_S = 120.0
POLL_S = 0.3
KINDS = ('action clients', 'action servers', 'service clients', 'service servers', 'publishers',
         'subscriptions')
# MoveIt's own robot-scoped names, by their last component: the clients' actions and services,
# and the topics MoveGroupInterface and the PlanningSceneMonitor use (ADR-0023 Decision 5)
MOVEIT_CLIENT_IDS = ('move_action', 'execute_trajectory', 'sequence_move_group',
                     'compute_cartesian_path', 'query_planner_interface', 'get_planner_params',
                     'set_planner_params', 'get_planning_scene', 'apply_planning_scene')
MOVEIT_IDS = MOVEIT_CLIENT_IDS + ('trajectory_execution_event', 'attached_collision_object',
                                  'planning_scene', 'collision_object', 'planning_scene_world',
                                  'monitored_planning_scene', 'joint_states')
LOG_LINE = re.compile(r'^\[(?P<proc>[A-Za-z0-9_.]+)-\d+\] \[(?P<lvl>DEBUG|INFO|WARN|ERROR|FATAL)\] '
                      r'\[\d+\.\d+\] \[(?P<logger>[^\]]+)\]: (?P<msg>.*)$')
KEEP = ('MoveWaypoint ', 'CloseGripper ', 'OpenGripper ', 'VerifyGrasp ', 'CommandActuator ',
        'BT finished', 'MoveIt start-up', 'end-effector contract', '[BT] ')
DURATION = re.compile(r'\b\d+(\.\d+)?(e-?\d+)? ?(seconds|second|sec|ms|s)\b')
# the per-process address MoveIt and tf2 append to their internal nodes, inside any name
ADDRESS = re.compile(r'_[0-9a-f]{8,}\b')
# move_group's log, one line per goal: (what, pattern)
GOALS = (('MoveGroup goals', 'move_action_capability]: Received request'),
         ('ExecuteTrajectory goals',
          'execute_trajectory_action_capability]: Received goal request'),
         ('executions SUCCEEDED',
          'execute_trajectory_action_capability]: Execution completed: SUCCEEDED'))


def _cl():
    import composite_launch
    return composite_launch


# --- generation and build -------------------------------------------------------------------------

def generate(ws: Path) -> list:
    """The three bundles into ws/src/<name>/ (projects under ws/in/), with the TSA under test.
    Returns one identity line per bundle (app package, tree sha256, file count)."""
    import eef_fixture as F
    from trainit_setup_assistant.generator import Orchestrator
    CL = _cl()
    ws = CL._outside_the_underlay(ws)
    out = []
    for name, ns in [cell for cells in CASES.values() for cell in cells]:
        project = F.suction_simple(ws / 'in' / name, ns=ns, name=name, launch_safe=True)
        tree = ws / 'src' / name
        Orchestrator().generate(project, tree)
        files = sum(1 for q in tree.rglob('*') if q.is_file())
        out.append(f'{name:7s} ns={ns} app={project.bundle.app_package} tree sha256 '
                   f'{CL.tree_sha256(tree)} ({files} files)')
    (ws / 'bundles.txt').write_text('\n'.join(out) + '\n')
    return out


def build(ws: Path) -> str:
    """colcon build of ws/src over the sourced workspace: one package at a time, make -j2,
    build/ install/ log/ inside ws (a copy install). Returns colcon's summary line."""
    CL = _cl()
    ws = CL._outside_the_underlay(ws)
    q = shlex.quote
    cmd = (f'set -o pipefail; {CL._sourced()} && cd {q(str(ws))} && [ "$(pwd -P)" = {q(str(ws))} ] '
           f'&& export MAKEFLAGS=-j2 CMAKE_BUILD_PARALLEL_LEVEL=2 && '
           f'colcon --log-base {q(str(ws / "log"))} build --base-paths {q(str(ws / "src"))} '
           f'--build-base {q(str(ws / "build"))} --install-base {q(str(ws / "install"))} '
           f'--executor sequential --event-handlers console_cohesion- 2>&1 | tail -5')
    r = subprocess.run(['bash', '-c', cmd], cwd=ws, capture_output=True, text=True, timeout=1800)
    if r.returncode != 0 or 'Summary:' not in r.stdout or 'failed' in r.stdout:
        raise RuntimeError(f'colcon build failed (rc {r.returncode}):\n{r.stdout}{r.stderr}')
    return next(line for line in r.stdout.splitlines() if 'Summary:' in line).strip()


# --- the graph ------------------------------------------------------------------------------------

def _fq(ns: str, name: str) -> str:
    return f'{ns.rstrip("/")}/{name}'


class _Graph:
    """What the probe sees on the launch domain: the bring-ups' nodes and servers (read once,
    after 6 s of discovery), then every node that appears later, with its endpoints by kind,
    read every POLL_S and united while it lives."""

    def __init__(self, probe):
        self.probe = probe
        self.lock = threading.Lock()
        self.base = set()
        self.base_nodes = []
        self.servers = {}                    # action or service name -> sorted server nodes
        self.tf_publishers = {}
        self.seen = {}                       # (ns, stable name) -> {kind: set of names}
        self._stop = threading.Event()
        self._thread = None

    def _endpoints(self, name: str, ns: str) -> dict:
        import rclpy.action
        n = self.probe.node
        kinds = {
            'action clients': rclpy.action.get_action_client_names_and_types_by_node(n, name, ns),
            'action servers': rclpy.action.get_action_server_names_and_types_by_node(n, name, ns),
            'service clients': n.get_client_names_and_types_by_node(name, ns),
            'service servers': n.get_service_names_and_types_by_node(name, ns),
            'publishers': n.get_publisher_names_and_types_by_node(name, ns),
            'subscriptions': n.get_subscriber_names_and_types_by_node(name, ns)}
        # an action's own entities (<action>/_action/...) are listed as the action itself; a
        # node's address inside a name (its parameter services) is masked
        return {k: {ADDRESS.sub('_<id>', t) for t, _types in v if '/_action/' not in t}
                for k, v in kinds.items()}

    def baseline(self, settle_s: float) -> None:
        CL = _cl()
        time.sleep(settle_s)
        self.base = set(self.probe.nodes())
        self.base_nodes = sorted(_fq(ns, CL._stable(n)) for ns, n in self.base)
        for ns, name in sorted(self.base):
            try:
                eps = self._endpoints(name, ns)
            except Exception:                    # noqa: BLE001 -- a node gone since the listing
                continue
            for kind in ('action servers', 'service servers'):
                for t in eps[kind]:
                    self.servers.setdefault(t, set()).add(_fq(ns, CL._stable(name)))
        node = self.probe.node
        for topic in ('/tf', '/tf_static'):
            self.tf_publishers[topic] = sorted(_fq(i.node_namespace, CL._stable(i.node_name))
                                               for i in node.get_publishers_info_by_topic(topic))

    def poll(self) -> None:
        CL = _cl()
        for ns, name in self.probe.nodes():
            if (ns, name) in self.base:
                continue
            try:
                eps = self._endpoints(name, ns)
            except Exception:                    # noqa: BLE001 -- a node gone since the listing
                continue
            with self.lock:
                entry = self.seen.setdefault((ns, CL._stable(name)), {k: set() for k in KINDS})
                for kind, names in eps.items():
                    entry[kind] |= names

    def start(self) -> None:
        def loop():
            while not self._stop.is_set():
                self.poll()
                self._stop.wait(POLL_S)
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10.0)

    def runner(self, ns: str) -> dict:
        """The runner process in ns: its nodes and the union of their endpoints by kind."""
        with self.lock:
            nodes = sorted(k for k in self.seen if k[0] == ns)
            eps = {kind: sorted({t for k in nodes for t in self.seen[k][kind]}) for kind in KINDS}
        return dict(nodes=[_fq(ns, n) for _ns, n in nodes], **eps)

    def outside(self, namespaces) -> list:
        with self.lock:
            return sorted(_fq(ns, n) for ns, n in self.seen if ns not in namespaces)


# --- one case -------------------------------------------------------------------------------------

def _last(name: str) -> str:
    return name.rsplit('/', 1)[-1]


class RunnerLaunch:
    """One case of CASES on the bundles built in ws: the bring-ups, then the runners."""

    def __init__(self, ws: Path, case: str, bundles: list):
        CL = _cl()
        self.ws = CL._outside_the_underlay(ws)
        self.case = case
        self.cells = CASES[case]
        self.namespaces = [ns for _name, ns in self.cells]
        self.bundles = [b for b in bundles if b.split()[0] in {n for n, _ns in self.cells}]
        self.logdir = self.ws / 'logs' / case
        self.tmp = self.ws / 'tmp'
        self.ros_log = self.ws / 'ros_log' / case
        for d in (self.logdir, self.tmp, self.ros_log):
            d.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, ROS_DOMAIN_ID=str(DOMAIN), ROS_LOCALHOST_ONLY='1',
                        RCUTILS_COLORIZED_OUTPUT='0', ROS_LOG_DIR=str(self.ros_log),
                        TMPDIR=str(self.tmp))
        self.lines = []
        self.obs = {ns: {} for ns in self.namespaces}
        self.guards = {}
        self.stops = {}
        self.graph = None
        self.launch_params = 0

    # --- the report -----------------------------------------------------------------------
    def say(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def report(self) -> str:
        return '\n'.join(self.lines) + '\n'

    def save(self, dest: Path) -> None:
        dest = Path(dest)
        (dest / 'logs').mkdir(parents=True, exist_ok=True)
        (dest / 'report.txt').write_text(self.report())
        (dest / 'bundles.txt').write_text('\n'.join(self.bundles) + '\n')
        for log in sorted(self.logdir.glob('*.log')):
            shutil.copy(log, dest / 'logs' / log.name)

    def _mask(self, text: str) -> str:
        text = text.replace(str(self.ws), '<ws>')
        return ADDRESS.sub('_<id>', DURATION.sub('<DUR>', text))

    # --- processes ------------------------------------------------------------------------
    def _launch(self, what: str, log: Path) -> subprocess.Popen:
        CL = _cl()
        cmd = (CL._sourced(str(self.ws / 'install' / 'setup.bash')) +
               f' && exec ros2 launch {what} > {shlex.quote(str(log))} 2>&1')
        return subprocess.Popen(['bash', '-c', cmd], env=self.env, start_new_session=True)

    def _guard(self, tag: str) -> bool:
        CL = _cl()
        # one name per guard: a node left over from an earlier guard is never discounted as "self"
        probe = CL._Probe(DOMAIN, f't17_guard_{self.case}_{len(self.guards) + 1}', self.ros_log)
        try:
            time.sleep(CL.SETTLE_S)
            nodes = [_fq(ns, n) for ns, n in probe.nodes()]
        finally:
            probe.close()
        self.guards[tag] = nodes
        return not nodes

    def _guard_line(self, tag: str) -> str:
        return f'[guard] ROS domain {DOMAIN} {tag}: {self.guards[tag] or "no node"}'

    # --- the whole case -------------------------------------------------------------------
    def run(self) -> 'RunnerLaunch':
        CL = _cl()
        self.say(f'# vNext M2.7 {self.case} (test_m2_7_bundle_runner_launch.py): normalised report')
        for line in self.bundles:
            self.say(f'bundle {line}')
        empty = self._guard('before the launches')
        self.say(self._guard_line('before the launches'))
        if not empty:
            raise RuntimeError(f'ROS domain {DOMAIN} is not empty: refusing to launch on it')
        ups, bts, probe = {}, {}, None
        try:
            for name, ns in self.cells:
                log = self.logdir / f'bringup_{name}.log'
                ups[ns] = self._launch(f'{name}_app bringup.launch.py mode:=mock rviz:=false', log)
            for name, ns in self.cells:
                log = self.logdir / f'bringup_{name}.log'
                adapter = re.escape(f'trainit_mock adapter for {ns}/eef/{DEV}')
                arm = r'Configured and activated .*moveit_joint_controller'
                parts = (CL._wait_log(log, r'You can start planning now', 120),
                         CL._wait_log(log, arm, 60), CL._wait_log(log, adapter, 60))
                self.obs[ns]['ready'] = all(parts)
                self.say(f'[{ns}] bring-up {name}_app bringup.launch.py mode:=mock rviz:=false: '
                         f'move_group ready {parts[0]}, arm controller active {parts[1]}, '
                         f'mock adapter up {parts[2]}')
            probe = CL._Probe(DOMAIN, 't17_probe', self.ros_log)
            self.graph = _Graph(probe)
            self.graph.baseline(CL.SETTLE_S)
            self.graph.start()
            started = {}
            for name, ns in self.cells:                     # back to back: at once
                bts[ns] = self._launch(f'{name}_app trainit_bt.launch.py use_sim_time:=false '
                                       f'groot_monitor:=false', self.logdir / f'runner_{name}.log')
                started[ns] = time.monotonic()
            for name, ns in self.cells:
                left = TREE_TIMEOUT_S - (time.monotonic() - started[ns])
                try:
                    bts[ns].wait(timeout=max(left, 0.1))
                    self.obs[ns]['timed_out'] = False
                except subprocess.TimeoutExpired:
                    self.obs[ns]['timed_out'] = True
        finally:
            if self.graph is not None:
                self.graph.stop()
            for name, ns in self.cells:
                if ns in bts:
                    self.stops[f'runner {ns}'] = CL.CompositeLaunch._stop_outcome(bts[ns])
            for name, ns in self.cells:
                if ns in ups:
                    self.stops[f'bring-up {ns}'] = CL.CompositeLaunch._stop_outcome(ups[ns])
            if probe is not None:
                probe.close()
            self.launch_params = sum(1 for _ in self.tmp.glob('launch_params_*'))
            self._guard('after the launches')
        self._read()
        return self

    # --- what the logs and the graph say --------------------------------------------------
    def _read(self) -> None:
        CL = _cl()
        g = self.graph
        self.say(f'[graph] nodes once the bring-ups were up: {g.base_nodes}')
        for topic, pubs in sorted(g.tf_publishers.items()):
            self.say(f'[graph] publishers of {topic} (TF is cell-global by design): {pubs}')
        self.say(f'[graph] nodes that appeared outside {self.namespaces}: '
                 f'{g.outside(self.namespaces) or "(none)"}')
        for name, ns in self.cells:
            o = self.obs[ns]
            text = (self.logdir / f'runner_{name}.log').read_text(errors='replace')
            parsed = CL.CompositeLaunch._parse(text, o.get('timed_out', True))
            o.update(finished=parsed['finished'], code=parsed['code'])
            tree = re.search(r"tree='([^']+)'", text)
            o['moves'] = (len(ET.parse(tree.group(1)).getroot().findall('.//MoveWaypoint'))
                          if tree and Path(tree.group(1)).is_file() else None)
            o['moved'] = len(re.findall(r"\]: MoveWaypoint '", text))
            self.say(f'[{ns}] runner {name}_app trainit_bt.launch.py use_sim_time:=false '
                     f'groot_monitor:=false: BT finished {o["finished"]}; exit {o["code"]}; '
                     f'timed out {o.get("timed_out")}')
            self.say(f'[{ns}] the tree it ran: {o["moves"]} MoveWaypoint; MoveWaypoint lines in '
                     f'its log: {o["moved"]}')
            # the runner's semantic lines in their order; its other WARN/ERROR/FATAL messages once
            # each (how often MoveIt repeats one depends on timing)
            other = []
            for line in text.splitlines():
                m = LOG_LINE.match(line)
                if not m or m['proc'] != 'trainit_run_bt':
                    continue
                said = self._mask(f'[{m["lvl"]}] [{m["logger"]}]: {m["msg"]}')
                if m['msg'].startswith(KEEP):
                    self.say(f'[{ns}]   {said}')
                elif m['lvl'] in ('WARN', 'ERROR', 'FATAL') and said not in other:
                    other.append(said)
            self.say(f'[{ns}] its other WARN/ERROR/FATAL messages, each once ({len(other)}):')
            for said in sorted(other):
                self.say(f'[{ns}]   {said}')
            up = (self.logdir / f'bringup_{name}.log').read_text(errors='replace').splitlines()
            mg = [line for line in up if line.startswith('[move_group-')]
            o['goals'] = {what: sum(pattern in line for line in mg) for what, pattern in GOALS}
            self.say(f'[{ns}] its move_group: ' +
                     ', '.join(f'{what} {n}' for what, n in o['goals'].items()))
            r = g.runner(ns)
            o['runner'] = r
            self.say(f'[{ns}] the runner process\'s nodes: {r["nodes"]}')
            for kind in KINDS:
                self.say(f'[{ns}]   {kind} ({len(r[kind])}): {r[kind]}')
            o['moveit_clients'] = sorted({t for kind in ('action clients', 'service clients')
                                          for t in r[kind] if _last(t) in MOVEIT_CLIENT_IDS})
            o['moveit_outside'] = sorted({t for kind in KINDS for t in r[kind]
                                          if _last(t) in MOVEIT_IDS
                                          and not t.startswith(ns + '/')})
            o['served_by'] = {t: sorted(g.servers.get(t, ())) for t in o['moveit_clients']}
            for t, servers in o['served_by'].items():
                self.say(f'[{ns}]   {t} is served by {servers or "(nobody)"}')
            self.say(f'[{ns}]   MoveIt names of the runner outside {ns}: '
                     f'{o["moveit_outside"] or "(none)"}')
        if len(self.cells) == 2:
            a, b = (self.obs[ns]['runner'] for ns in self.namespaces)
            names = [{t for kind in KINDS for t in r[kind]} for r in (a, b)]
            self.common = sorted(names[0] & names[1])
            self.say(f'[{" + ".join(self.namespaces)}] names both runner processes use: '
                     f'{self.common}')
        for what, outcome in self.stops.items():
            self.say(f'[shutdown] {what}: {outcome}')
        self.say(f'[hygiene] launch_params files in the workspace TMPDIR: '
                 f'{"some" if self.launch_params else "none"}')
        self.say(self._guard_line('after the launches'))

    # --- the verdicts the tests read ------------------------------------------------------
    def tree_ok(self, ns: str) -> list:
        o = self.obs[ns]
        return [what for what, ok in (
            ('the bring-up came up', o.get('ready') is True),
            ('not timed out (120 s)', o.get('timed_out') is False),
            ('BT finished: SUCCESS', o.get('finished') == 'SUCCESS'),
            ('exit 0', o.get('code') == 0),
            ('a MoveWaypoint line per move of the tree', bool(o.get('moves'))
             and o.get('moved') == o.get('moves'))) if not ok]


def _run_case(built: dict, case: str) -> RunnerLaunch:
    run = RunnerLaunch(built['ws'], case, built['bundles'])
    try:
        run.run()
    finally:
        if os.environ.get(EVIDENCE):
            run.save(Path(os.environ[EVIDENCE]) / 'bundle_runner' / case)
    return run


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    CL = _cl()
    have = CL.installed_version('trainit_motion_runtime')
    if have < CL.PHASE_C_MIN_TMR:
        pytest.skip(f'requires a namespaced runner that resolves MoveIt in its namespace '
                    f'(ADR-0023, trainit_motion_runtime >= {CL._v(CL.PHASE_C_MIN_TMR)}); '
                    f'installed {CL._v(have)}')
    ws = CL._outside_the_underlay(tmp_path_factory.mktemp('t17_ws'))
    bundles = generate(ws)
    return dict(ws=ws, bundles=bundles, summary=build(ws))


@pytest.fixture(scope='module')
def t17b(built):
    return _run_case(built, 't17b')


@pytest.fixture(scope='module')
def t17c(built):
    return _run_case(built, 't17c')


def _why(run: RunnerLaunch) -> str:
    return '\n' + run.report()


# --- the tests ------------------------------------------------------------------------------------

def test_t17_the_three_canonical_bundles_generate_and_build(built):
    assert len(built['bundles']) == 3, built['bundles']
    assert '9 packages finished' in built['summary'], built['summary']


def test_t17b_the_r3_tree_ends_success_within_120_s(t17b):
    assert t17b.tree_ok('/r3') == [], _why(t17b)


def test_t17b_every_moveit_client_of_the_runner_is_under_r3(t17b):
    o = t17b.obs['/r3']
    assert '/r3/trainit_run_bt' in o['runner']['nodes'], _why(t17b)
    assert t17b.graph.outside(['/r3']) == [], _why(t17b)           # the attribution holds
    assert {'/r3/move_action', '/r3/execute_trajectory'} <= set(o['runner']['action clients']), \
        _why(t17b)
    assert o['moveit_clients'] and all(t.startswith('/r3/') for t in o['moveit_clients']), \
        _why(t17b)
    assert o['moveit_outside'] == [], _why(t17b)


def test_t17b_r3_move_group_receives_the_trees_goals(t17b):
    o = t17b.obs['/r3']
    assert o['moves'] and o['goals'] == {what: o['moves'] for what, _p in GOALS}, _why(t17b)


@pytest.mark.parametrize('ns', ['/r1', '/r2'])
def test_t17c_both_trees_end_success_at_once(t17c, ns):
    assert t17c.tree_ok(ns) == [], _why(t17c)


@pytest.mark.parametrize('ns', ['/r1', '/r2'])
def test_t17c_each_move_group_receives_only_its_own_robots_goals(t17c, ns):
    o = t17c.obs[ns]
    assert o['moves'] and o['goals'] == {what: o['moves'] for what, _p in GOALS}, _why(t17c)
    for action in (f'{ns}/move_action', f'{ns}/execute_trajectory'):
        assert action in o['runner']['action clients'], _why(t17c)
    for client, servers in o['served_by'].items():
        assert servers and all(s.startswith(f'{ns}/move_group') for s in servers), \
            (client, servers, _why(t17c))


def test_t17c_the_runner_endpoint_sets_are_disjoint_except_the_cell_global_set(t17c):
    CL = _cl()
    assert t17c.graph.outside(['/r1', '/r2']) == [], _why(t17c)    # the attribution holds
    for ns in ('/r1', '/r2'):
        assert f'{ns}/trainit_run_bt' in t17c.obs[ns]['runner']['nodes'], _why(t17c)
    assert set(t17c.common) <= set(CL.GRAPH_EXEMPT), _why(t17c)


def test_t17c_no_moveit_name_of_either_runner_is_outside_its_namespace(t17c):
    for ns in ('/r1', '/r2'):
        o = t17c.obs[ns]
        assert o['moveit_clients'] and all(t.startswith(ns + '/') for t in o['moveit_clients']), \
            _why(t17c)
        assert o['moveit_outside'] == [], _why(t17c)


@pytest.mark.parametrize('case', ['t17b', 't17c'])
def test_t17_every_launch_ends_cleanly_and_the_domain_is_empty_before_and_after(case, request):
    run = request.getfixturevalue(case)
    assert run.guards.get('before the launches') == [], _why(run)
    assert run.guards.get('after the launches') == [], _why(run)
    want = {f'{what} {ns}' for ns in run.namespaces for what in ('runner', 'bring-up')}
    assert set(run.stops) == want, _why(run)
    assert all(outcome == 'clean' for outcome in run.stops.values()), _why(run)
    assert run.launch_params > 0, _why(run)            # launch_ros wrote into the workspace TMPDIR
