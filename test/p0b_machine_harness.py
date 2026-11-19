"""The P0.b machine-handshake LAUNCH harness (opt-in tiers M and I; overlay docs/engineering/P0B_IMPLEMENTATION_PLAN.md
WP6 / WP7): the public machine application served by the robot task server, driven by TMR's fake PLC, against the
cell's machine simulation (trainit_cell_io), observed by an in-process recorder.

    Recorder   the test's eyes and hands on the launch domain (composite_launch._Probe's pattern: its own rclpy
               context, spun by its own threads): every RTI status, execution event, machine-service status and
               request image and the cell's state, with the receipt time (time.monotonic) and the header stamp; it
               publishes the cell's OPERATOR commands (never a robot command) and holds the scripted / silent second
               requesters some cases need
    FakePlc    trainit_fake_plc, interactive (one command per stdin line), its plain words parsed from its log
    MachineRun the processes (bring-up, cell, robot task server, fake PLC), jobs, waits, and the M tier's scenes

Every wait is bounded (SceneError names what did not happen). A scene that fails records its error and the next scene
starts from a settled cell (no injection, no second requester, every service READY or DONE, the robot ready).
Not a safety function (ADR-0033).
"""
import contextlib
import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import threading
import time
import uuid
from pathlib import Path

DOMAIN_M = 80
SERVICES = ('turner', 'indexer')
MAX_AGE_S = 1.0           # channel_status_max_age_ms (TMR's default)
PERIOD_S = 0.1            # the cell's status period (the public cell file)
TICK_S = 0.01             # the runner's tick
# "at once" / "at its first tick": the node's first tick after the condition, + one tick of phase. Two ticks (20 ms)
# is the bound, not a measured value: M run 2 measured 0.1-6.1 ms for the "at once" decisions (N3 0.1, F9 0.1, N4
# 0.1, PB-P22a 0.2, PB-P22b 4.4, PB-P3a 6.1) -- the bound keeps the decision within the node's own first tick,
# whatever the tick's phase
AT_ONCE_S = 2 * TICK_S
RTI_PERIOD_S = 0.1        # the robot task server's status period
O2_S = 0.15               # O2: a machine-state change -> the RTI detail / the fake PLC
WAIT_TIMEOUT_S = 8.0      # the `wait for: done` timeout of machine_tasks.yaml's turn / index / turn_part
END_LINES_S = 3.0         # end_of: a job's RUN_ENDED event and the fake PLC's "finished" line, at most this after it
ROBOT_SEES_S = 15.0       # robot_sees: a (re)started source reaches the robot's own view (DDS discovery) within this
R1_CYCLES = 100
# the requester sessions of the test's own scripted / second requesters (never the robot's)
SCRIPTED_S1, SECOND_REQUESTER = 0x5C1D0001, 0x5C1D0011
RECORDER_MODULES = ('rclpy', 'std_msgs', 'trainit_interfaces')   # what Recorder.__init__ imports (pinned)
NEVER = ('REFUSED', 'JOB LOST', 'Robot LOST', 'exit 2', 'internal error', 'PLAN FAILED', 'VALIDATION FAILED')
# never in a user-facing machine text: a topic, a counter, an identity (P0B_DESIGN §3.5; O1)
PROTOCOL = re.compile(r'\bseq\b|heartbeat|source_session|requester_session|cycle_seq|cycle_requester|reject_|'
                      r'0x[0-9a-fA-F]{4,}|/(?:turner|indexer)/|trainit/|ChannelStatus|ChannelRequest')

# trainit_interfaces values (pinned on both sides by test_machine_channel / test_robot_task_interface)
IDLE, RUNNING, STOPPING, FAULTED = 1, 2, 3, 4
SUCCEEDED, CANCELLED, FAILED = 1, 3, 4
OK, CODE_CANCELLED, TASK_FAILED = 100, 200, 301
STATION_TIMEOUT, STATION_FAULT, STATION_UNAVAILABLE, STATION_BUSY = 305, 306, 310, 313
READY, BUSY, DONE, FAULT, HOMING = 1, 2, 3, 4, 5
STATE_NAME = {0: 'NONE', READY: 'READY', BUSY: 'BUSY', DONE: 'DONE', FAULT: 'FAULT', HOMING: 'HOMING'}
REJECT_BUSY, REJECT_FAULTED, REJECT_TWO = 1, 2, 3
REJECT_TO_CODE = {REJECT_BUSY: STATION_BUSY, REJECT_FAULTED: STATION_FAULT, REJECT_TWO: STATION_UNAVAILABLE}
ACTUATOR_TIMEOUT, SENSOR_FAULT, STATION_FAULT_CODE = 1, 2, 3
RUN_ENDED, TASK_EXITED, STEP_ENTERED, STEP_EXITED, PROCESS_EVENT = 2, 4, 5, 6, 7
EV_SUCCEEDED, EV_FAILED, EV_HALTED = 1, 2, 3


class SceneError(RuntimeError):
    """A scene could not reach the state its case needs, or something did not happen in time (named)."""


def _cl():
    import composite_launch as CL
    return CL


# --- the user-facing words ------------------------------------------------------------------------------------------

def machine_words(text: str) -> list:
    """The machine states a user-facing text shows, in order (P0B_DESIGN §3.5 words)."""
    found = []
    for word, pattern in (('REQUESTED', r'MACHINE \w+ REQUESTED'), ('WAITING', r'WAITING FOR MACHINE'),
                          ('BUSY', r'MACHINE \w+ BUSY'), ('DONE', r'MACHINE \w+ DONE'),
                          ('FAULT', r'MACHINE \w+ FAULT'), ('TIMEOUT', r'MACHINE \w+ TIMEOUT'),
                          ('UNAVAILABLE', r'MACHINE \w+ UNAVAILABLE')):
        m = re.search(pattern, text)
        if m:
            found.append((m.start(), word))
    return [w for _i, w in sorted(found)]


def collapse(seq: list) -> list:
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


def run_panel_states(lines: list) -> list:
    """What TSA's run panel shows (its state line) when fed these robot task server log lines (offscreen Qt)."""
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    from trainit_setup_assistant.gui.run_panel import RunPanel
    app = QApplication.instance() or QApplication([])

    class _Launcher:
        def add_listener(self, _listener):
            pass

        def running(self, _kind):
            return False

    panel = RunPanel(_Launcher(), [('server', 'Robot task server', lambda: None)])
    states = []
    for line in lines:
        panel.line('server', line)
        text = panel.state.text()
        if text and (not states or states[-1] != text):
            states.append(text)
    panel.deleteLater()
    app.processEvents()
    return states


# --- the recorder ---------------------------------------------------------------------------------------------------

def _delta(before: dict, after: dict, key: str):
    """after[key] - before[key] of two cell service views; None when the cell does not report `key`."""
    if key not in before or key not in after:
        return None
    return after[key] - before[key]


class Recorder:
    """The test's eyes and hands on the launch domain (module docstring)."""

    def __init__(self, domain: int, log_dir: Path):
        import rclpy
        from rclpy.executors import MultiThreadedExecutor
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String
        from trainit_interfaces.msg import ChannelRequest, ChannelStatus, ExecutionEvent, RobotTaskStatus
        self._rclpy = rclpy
        self._String, self._ChannelRequest = String, ChannelRequest
        self._saved = {k: os.environ.get(k) for k in ('ROS_LOCALHOST_ONLY', 'ROS_LOG_DIR')}
        os.environ['ROS_LOCALHOST_ONLY'] = '1'
        os.environ.setdefault('ROS_LOG_DIR', str(log_dir))
        self.context = rclpy.Context()
        rclpy.init(context=self.context, domain_id=domain)
        self.node = rclpy.create_node('p0b_recorder', context=self.context)
        self.lock = threading.Lock()
        self.data = {'rti': [], 'events': [], 'cell': []}
        for s in SERVICES:
            self.data[f'status:{s}'] = []
            self.data[f'request:{s}'] = []
        live = QoSProfile(depth=1000, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                          durability=DurabilityPolicy.VOLATILE)
        self.latched = QoSProfile(depth=1, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                                  durability=DurabilityPolicy.TRANSIENT_LOCAL)
        commands = QoSProfile(depth=10, history=HistoryPolicy.KEEP_LAST, reliability=ReliabilityPolicy.RELIABLE,
                              durability=DurabilityPolicy.VOLATILE)
        n = self.node
        n.create_subscription(RobotTaskStatus, '/trainit/robot_task/status', self._on_rti, live)
        n.create_subscription(ExecutionEvent, '/trainit/execution_events', self._on_event, live)
        for s in SERVICES:
            n.create_subscription(ChannelStatus, f'/{s}/status', lambda m, s=s: self._on_status(s, m), live)
            n.create_subscription(ChannelRequest, f'/{s}/request', lambda m, s=s: self._on_request(s, m), live)
        n.create_subscription(String, '/cell_io/state', self._on_cell, self.latched)
        self.control = n.create_publisher(String, '/cell_io/control', commands)
        self.executor = MultiThreadedExecutor(num_threads=4, context=self.context)
        self.executor.add_node(n)
        threading.Thread(target=self.executor.spin, daemon=True).start()

    # --- callbacks: small dicts, receipt time + header stamp
    @staticmethod
    def _stamp(msg) -> float:
        return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

    def _add(self, key: str, item: dict) -> None:
        with self.lock:
            self.data[key].append(item)

    def _on_rti(self, m) -> None:
        self._add('rti', dict(t=time.monotonic(), stamp=self._stamp(m), state=m.state, ready=m.ready,
                              hb=m.heartbeat, session=m.server_session, job=m.job_id, task=m.task,
                              task_name=m.task_name, result=m.result, code=m.result_code, fault=m.fault_code,
                              detail=m.detail, ack=m.ack, ack_seq=m.ack_seq, ack_code=m.ack_code))

    def _on_event(self, m) -> None:
        self._add('events', dict(t=time.monotonic(), stamp=self._stamp(m), kind=m.kind, status=m.status,
                                 task=m.task_id, step=m.step_id, step_kind=m.step_kind, channel=m.process_channel,
                                 pkind=m.process_kind, job=m.job_id, run=m.run_id, code=m.result_code))

    def _on_status(self, s: str, m) -> None:
        self._add(f'status:{s}', dict(t=time.monotonic(), stamp=self._stamp(m), session=m.source_session,
                                      hb=m.heartbeat, state=m.state, cycle=(m.cycle_requester, m.cycle_seq),
                                      done=m.cycle_done, reject=(m.reject_requester, m.reject_seq),
                                      rcode=m.reject_code, fcode=m.fault_code, detail=m.detail))

    def _on_request(self, s: str, m) -> None:
        self._add(f'request:{s}', dict(t=time.monotonic(), stamp=self._stamp(m), src=m.source_session,
                                       rs=m.requester_session, seq=m.seq))

    def _on_cell(self, m) -> None:
        try:
            data = json.loads(m.data)
        except ValueError:
            return
        if isinstance(data, dict):
            self._add('cell', dict(t=time.monotonic(), state=data))

    # --- reading
    def snap(self, key: str) -> list:
        with self.lock:
            return list(self.data[key])

    def last(self, key: str):
        with self.lock:
            items = self.data[key]
            return items[-1] if items else None

    def count_publishers(self, topic: str) -> int:
        return self.node.count_publishers(topic)

    def publisher_nodes(self, topic: str) -> list:
        return sorted(i.node_name for i in self.node.get_publishers_info_by_topic(topic))

    # --- the cell's operator surface (never the robot)
    def cell_command(self, command: dict, timeout: float = 6.0) -> dict:
        cid = uuid.uuid4().hex
        text = json.dumps(dict(command, id=cid), sort_keys=True)
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            self.control.publish(self._String(data=text))
            stop = min(end, time.monotonic() + 0.4)
            while time.monotonic() < stop:
                cell = self.last('cell')
                answer = (cell or {}).get('state', {}).get('last_command') or {}
                if answer.get('id') == cid:
                    if not answer.get('ok'):
                        raise SceneError(f'the cell refused {command}: {answer.get("text")}')
                    return answer
                time.sleep(0.01)
        raise SceneError(f'no answer from the cell to {command} within {timeout:g} s')

    # --- a second requester (scripted or silent): never the robot's
    def requester(self, service: str):
        return self.node.create_publisher(self._ChannelRequest, f'/{service}/request', self.latched)

    def publish_request(self, pub, src: int, rs: int, seq: int) -> None:
        msg = self._ChannelRequest()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.source_session, msg.requester_session, msg.seq = src, rs, seq
        pub.publish(msg)

    def drop(self, pub) -> None:
        self.node.destroy_publisher(pub)

    def close(self) -> None:
        self.executor.shutdown(timeout_sec=5.0)
        self.node.destroy_node()
        self._rclpy.try_shutdown(context=self.context)
        for key, value in self._saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


# --- the fake PLC ---------------------------------------------------------------------------------------------------

class FakePlc:
    """trainit_fake_plc driven line by line (stdin); its plain words read back from its log, with their time."""
    LINE = re.compile(r'^\[fake PLC (\d\d):(\d\d):(\d\d)\.(\d{3})\] (.*)$')

    def __init__(self, run: 'MachineRun', tasks: Path, log: Path):
        self.log = log
        self.proc = run._start('ros2 run trainit_motion_runtime trainit_fake_plc --ros-args '
                               f'-p tasks_file:={shlex.quote(str(tasks))}', log, stdin=subprocess.PIPE)

    def send(self, line: str) -> None:
        self.proc.stdin.write((line + '\n').encode())
        self.proc.stdin.flush()

    def said(self) -> list:
        """[(seconds of the local day, text)] of every '[fake PLC ...]' line."""
        text = self.log.read_text(errors='replace') if self.log.exists() else ''
        out = []
        for line in text.splitlines():
            m = self.LINE.match(line)
            if m:
                h, mi, s, ms = (int(x) for x in m.groups()[:4])
                out.append((h * 3600 + mi * 60 + s + ms / 1000.0, m.group(5)))
        return out

    def wait(self, pattern: str, since: int = 0, timeout: float = 10.0):
        """The first line at index >= since matching pattern: (index, seconds of day, text); SceneError otherwise."""
        end = time.monotonic() + timeout
        rx = re.compile(pattern)
        while True:
            said = self.said()
            for i in range(since, len(said)):
                if rx.search(said[i][1]):
                    return i, said[i][0], said[i][1]
            if time.monotonic() > end:
                raise SceneError(f'the fake PLC never said /{pattern}/ within {timeout:g} s')
            time.sleep(0.02)

    def stop(self) -> str:
        try:
            self.send('quit')
            self.proc.stdin.close()
        except (BrokenPipeError, OSError, ValueError):
            pass
        try:
            rc = self.proc.wait(timeout=15)
            return 'clean' if rc == 0 else f'exit {rc}'
        except subprocess.TimeoutExpired:
            return _cl().CompositeLaunch._stop_outcome(self.proc)


def seconds_of_day(stamp: float) -> float:
    lt = time.localtime(stamp)
    return lt.tm_hour * 3600 + lt.tm_min * 60 + lt.tm_sec + (stamp - int(stamp))


def day_delta(a: float, b: float) -> float:
    """b - a for two seconds-of-day values, across midnight."""
    d = (b - a) % 86400.0
    return d - 86400.0 if d > 43200.0 else d


def ms(a, b):
    """(b - a) in ms, None if either is missing."""
    if a is None or b is None:
        return None
    return round((b - a) * 1000.0, 1)


def kill_group(proc) -> None:
    """The last resort after a stop that raised: SIGKILL the process's group (every process started here leads its
    own session) and reap it. Never raises."""
    if getattr(proc, 'pid', None) is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except OSError:                                       # gone already, or not ours
        pass
    try:
        proc.wait(timeout=15)
    except (OSError, subprocess.SubprocessError):
        pass


# --- the run --------------------------------------------------------------------------------------------------------

class MachineRun:
    """The processes, the jobs and the waits of a machine-handshake launch tier (module docstring)."""

    def __init__(self, ws: Path, prefix: str, domain: int, cells: dict):
        CL = _cl()
        self.ws = CL._outside_the_underlay(ws)
        self.prefix, self.domain, self.cells = prefix, domain, cells
        self.logs, self.tmp, self.ros_log = self.ws / 'logs', self.ws / 'tmp', self.ws / 'ros_log'
        for d in (self.logs, self.tmp, self.ros_log):
            d.mkdir(parents=True, exist_ok=True)
        self.env = dict(os.environ, ROS_DOMAIN_ID=str(domain), ROS_LOCALHOST_ONLY='1',
                        RCUTILS_COLORIZED_OUTPUT='0', ROS_LOG_DIR=str(self.ros_log), TMPDIR=str(self.tmp))
        self.tasks = self.ws / 'install' / f'{prefix}_app' / 'share' / f'{prefix}_app' / 'config' / 'tasks.yaml'
        self.lines, self.records, self.errors = [], {}, {}
        self.guards, self.stops, self.ready, self.commands = {}, {}, {}, {}
        self.rec = None
        self.up = self.server = self.cell = None
        self.plc = None
        self.plc_logs = []        # [log, the line index the NEVER check stops at (None: the whole log)]
        self.server_logs, self.cell_logs = [], []
        self.cell_tag = None
        self.extra = []           # second requesters held by the recorder
        self.n = {'cell': 0, 'server': 0, 'plc': 0}
        self.next_job = 1
        self.jobs = {}
        self.sim_time = False     # the robot task server's use_sim_time (the I tier: true, on the stage's /clock)

    # --- reporting
    def say(self, text: str) -> None:
        self.lines.append(text)
        print(text, flush=True)

    def report(self) -> str:
        return '\n'.join(self.lines) + '\n'

    def save(self, dest: Path) -> None:
        dest.mkdir(parents=True, exist_ok=True)
        (dest / 'report.txt').write_text(self.report())
        (dest / 'records.json').write_text(json.dumps({'records': self.records, 'errors': self.errors,
                                                       'stops': self.stops, 'guards': self.guards,
                                                       'ready': self.ready, 'commands': self.commands},
                                                      indent=1, sort_keys=True, default=str))
        for log in sorted(self.logs.glob('*.log')):
            shutil.copy(log, dest / log.name)

    # --- processes
    def _start(self, command: str, log: Path, stdin=subprocess.DEVNULL) -> subprocess.Popen:
        CL = _cl()
        cmd = (CL._sourced(str(self.ws / 'install' / 'setup.bash')) +
               f' && exec {command} > {shlex.quote(str(log))} 2>&1')
        return subprocess.Popen(['bash', '-c', cmd], env=self.env, start_new_session=True, stdin=stdin)

    def guard(self, tag: str) -> list:
        CL = _cl()
        probe = CL._Probe(self.domain, f'p0b_guard_{len(self.guards) + 1}', self.ros_log)
        try:
            time.sleep(CL.SETTLE_S)
            nodes = [f'{ns.rstrip("/")}/{n}' for ns, n in probe.nodes()]
        finally:
            probe.close()
        self.guards[tag] = nodes
        self.say(f'[guard] ROS domain {self.domain} {tag} the launches: {nodes or "no node"}')
        return nodes

    def until(self, fn, timeout: float, what: str, poll: float = 0.005):
        end = time.monotonic() + timeout
        while True:
            value = fn()
            if value:
                return value
            if time.monotonic() > end:
                raise SceneError(f'timed out ({timeout:g} s) waiting for {what}')
            time.sleep(poll)

    def start_bringup(self) -> bool:
        CL = _cl()
        log = self.logs / 'bringup.log'
        command = f'ros2 launch {self.prefix}_app bringup.launch.py mode:=mock rviz:=false'
        self.commands['bringup'] = command
        self.up = self._start(command, log)
        self.ready = {
            'move_group ready': CL._wait_log(log, r'You can start planning now', 120),
            'arm controller active': CL._wait_log(log, r'Configured and activated .*moveit_joint_controller', 60),
            'mock adapter up': CL._wait_log(log, re.escape('trainit_mock adapter for /eef/main'), 60)}
        self.say(f'[bring-up] {self.ready}')
        return all(self.ready.values())

    def served(self, tag: str) -> list:
        import yaml
        data = yaml.safe_load(Path(self.cells[tag]).read_text())
        return [s for s in data.get('services', {}) if s in SERVICES]

    def server_mark(self) -> int:
        """The current robot task server log's size: a later robot_sees() reads only what follows it."""
        log = self.server_logs[-1] if self.server_logs else None
        return log.stat().st_size if log is not None and log.exists() else 0

    def robot_sees(self, services, mark: int, timeout: float = ROBOT_SEES_S) -> None:
        """The ROBOT's own view of a (re)started source, never the recorder's: TMR's registry logs "machine service
        'x': status arriving ..." when a source's status starts to reach it (its first image, a NEW session, or one
        after more than the max age without one). Waits -- condition-based and bounded -- for that line naming each
        service's CURRENT session after `mark` in the server log. A request issued before it is decided on the old,
        dead view (F9's 310 at once, M run 3: the robot discovered a restarted cell later than the recorder).
        Nothing to wait for while no robot task server runs."""
        if self.server is None or self.server.poll() is not None or not self.server_logs:
            return
        log = self.server_logs[-1]
        sessions = {s: self.rec.last(f'status:{s}')['session'] for s in services}

        def seen() -> bool:
            text = log.read_bytes()[mark:].decode('utf-8', 'replace')
            return all(re.search(rf"machine service '{s}': status arriving[^\n]*{sessions[s]:08x}", text)
                       for s in services)
        self.until(seen, timeout, f"the robot's own view of the source of {', '.join(services)}")

    def start_cell(self, tag: str) -> None:
        CL = _cl()
        self.n['cell'] += 1
        log = self.logs / f'cell_{self.n["cell"]}_{tag}.log'
        mark = self.server_mark()
        before = {s: (self.rec.last(f'status:{s}') or {}).get('session') for s in SERVICES}
        command = f'ros2 launch trainit_cell_io cell_io.launch.py config:={shlex.quote(str(self.cells[tag]))}'
        self.commands.setdefault(f'cell:{tag}', command)
        self.cell = self._start(command, log)
        self.cell_tag = tag
        self.cell_logs.append(log)
        if not CL._wait_log(log, r'cell_io: serving .* -- ready', 60):
            raise SceneError(f'the cell ({tag}) never said it serves: {log.name}')
        served = self.served(tag)
        self.until(lambda: all((self.rec.last(f'status:{s}') or {}).get('session') not in (None, before[s])
                               for s in served), 15, f'the new cell session ({tag}) on {served}')
        self.robot_sees(served, mark)
        self.say(f'[cell {self.n["cell"]}] {tag}: serving {served}, sessions '
                 + ', '.join(f'{s} {self.rec.last(f"status:{s}")["session"]:#010x}' for s in served))

    def stop_cell(self, how: str = 'int') -> str:
        proc, self.cell = self.cell, None
        tag = f'cell {self.n["cell"]} ({self.cell_tag})'
        if proc is None:
            return 'none'
        if how == 'kill':
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
            outcome = 'killed (SIGKILL)'
        else:
            outcome = _cl().CompositeLaunch._stop_outcome(proc)
        self.stops[tag] = outcome
        self.say(f'[{tag}] stopped: {outcome}')
        return outcome

    def start_server(self) -> Path:
        CL = _cl()
        self.n['server'] += 1
        log = self.logs / f'server_{self.n["server"]}.log'
        command = (f'ros2 launch {self.prefix}_app trainit_task_server.launch.py '
                   f'use_sim_time:={"true" if self.sim_time else "false"}')
        self.commands.setdefault('server', command)
        self.server = self._start(command, log)
        self.server_logs.append(log)
        if not CL._wait_log(log, r'RTI v1: server session', 90):
            raise SceneError(f'the robot task server never served: {log.name}')
        self.say(f'[server {self.n["server"]}] serving')
        return log

    def stop_server(self) -> str:
        proc, self.server = self.server, None
        outcome = _cl().CompositeLaunch._stop_outcome(proc) if proc is not None else 'none'
        self.stops[f'server {self.n["server"]}'] = outcome
        self.say(f'[server {self.n["server"]}] stopped: {outcome}')
        return outcome

    def start_plc(self, tag: str) -> None:
        self.n['plc'] += 1
        log = self.logs / f'fake_plc_{self.n["plc"]}_{tag}.log'
        self.plc = FakePlc(self, self.tasks, log)
        self.plc_logs.append([log, None])
        self.plc.wait(r'Robot needs reset|Robot ready|Robot FAULTED', 0, 30)

    def stop_plc(self) -> str:
        plc, self.plc = self.plc, None
        outcome = plc.stop() if plc is not None else 'none'
        self.stops[f'fake PLC {self.n["plc"]}'] = outcome
        return outcome

    def stop_bringup(self) -> str:
        proc, self.up = self.up, None
        outcome = _cl().CompositeLaunch._stop_outcome(proc) if proc is not None else 'none'
        self.stops['bring-up'] = outcome
        return outcome

    def stop_order(self) -> list:
        """(name, attribute, stop) of every process the run may hold, in the order they are stopped."""
        return [('fake PLC', 'plc', self.stop_plc), ('robot task server', 'server', self.stop_server),
                ('cell', 'cell', self.stop_cell), ('bring-up', 'up', self.stop_bringup)]

    def stop_all(self) -> None:
        """Every process stopped, EACH INDEPENDENTLY: a stop that raises is recorded as that process's outcome (never
        'clean'), its process group is SIGKILLed, and the next stop still runs -- a raise never leaves a launch
        running. The ExitStack runs every stop even when one is interrupted (a KeyboardInterrupt)."""
        with contextlib.ExitStack() as stack:
            for name, attr, stop in reversed(self.stop_order()):          # LIFO: the first one listed stops first
                stack.callback(self._stop_one, name, attr, stop)

    def _stop_one(self, name: str, attr: str, stop) -> None:
        held = getattr(self, attr)
        if held is None:
            return
        proc = getattr(held, 'proc', held)                                 # a FakePlc holds its process
        try:
            stop()
        except BaseException as exc:                       # noqa: B902 -- recorded; SIGKILLed; an interrupt goes on
            setattr(self, attr, None)
            self.stops[f'{name} (its stop raised)'] = f'{type(exc).__name__}: {exc}'
            self.say(f'[{name}] its stop RAISED {type(exc).__name__}: {exc} -- its process group SIGKILLed')
            kill_group(proc)
            if not isinstance(exc, Exception):
                raise                                      # a KeyboardInterrupt / SystemExit: the ExitStack goes on

    # --- the robot (RTI) through the fake PLC
    def rti_now(self):
        return self.rec.last('rti')

    def reset_robot(self, timeout: float = 15.0) -> None:
        now = self.rti_now()
        if now and now['state'] == IDLE and now['ready']:
            return
        self.plc.send('reset')
        self.until(lambda: (lambda r: r is not None and r['state'] == IDLE and r['ready'])(self.rti_now()), timeout,
                   'the robot ready after a RESET')

    def start_job(self, task: str) -> int:
        j = self.next_job
        self.next_job += 1
        self.jobs[j] = dict(id=j, task=task, t0=time.monotonic(), plc_mark=len(self.plc.said()))
        self.plc.send(f'start {task} {j}')
        return j

    def end_of(self, j: int, timeout: float = 60.0) -> dict:
        job = self.jobs[j]

        def ended():
            r = next((x for x in self.rec.snap('rti') if x['job'] == j and x['result'] != 0), None)
            if r is None and any('REJECTED: START' in text for _t, text in self.plc.said()[job['plc_mark']:]):
                raise SceneError(f'job {j} ({job["task"]}): the START was rejected')
            return r
        r = self.until(ended, timeout, f'job {j} ({job["task"]}) to end')
        # the job's LAST event (RUN_ENDED: nothing is emitted after it; or the TASK_EXITED of its task, which the
        # runner publishes right before it) and the fake PLC's "Job <j> (<task>) finished" line -- polled, bounded,
        # never assumed after a fixed sleep
        finished = re.compile(rf'^Job {j} \({re.escape(job["task"])}\) finished')

        def last_lines():
            last = any(e['job'] == j and (e['kind'] == RUN_ENDED or (e['kind'] == TASK_EXITED and
                                                                     e['task'] == job['task']))
                       for e in self.rec.snap('events'))
            return last and any(finished.search(text) for _t, text in self.plc.said()[job['plc_mark']:])
        self.until(last_lines, END_LINES_S, f'job {j} ({job["task"]}): its RUN_ENDED event and the fake PLC\'s '
                                            f'"Job {j} ({job["task"]}) finished" line', poll=0.01)
        time.sleep(0.05)          # the lines the fake PLC prints from the same status, after its "finished" line
        rti = [x for x in self.rec.snap('rti') if x['job'] == j]
        job.update(result=r['result'], code=r['code'], detail=r['detail'], t_end=r['t'], stamp_end=r['stamp'],
                   state_after=r['state'],
                   running=collapse([x['detail'] for x in rti if x['state'] == RUNNING and x['detail']]),
                   events=[e for e in self.rec.snap('events') if e['job'] == j],
                   plc=[text for _t, text in self.plc.said()[job['plc_mark']:]],
                   plc_timed=self.plc.said()[job['plc_mark']:])
        self.say(f'[job {j}] {job["task"]}: result {r["result"]} code {r["code"]} -- {r["detail"]}')
        return job

    def job(self, task: str, timeout: float = 60.0) -> dict:
        return self.end_of(self.start_job(task), timeout)

    @staticmethod
    def step(job: dict, step_id: str):
        """(entered, exited, process event) of a job's step, from its events."""
        ev = job.get('events', [])
        pick = (lambda kind: next((e for e in ev if e['kind'] == kind and e['step'] == step_id), None))
        return pick(STEP_ENTERED), pick(STEP_EXITED), pick(PROCESS_EVENT)

    # --- the machine side
    def images(self, s: str, since_t: float = None) -> list:
        return [x for x in self.rec.snap(f'status:{s}') if since_t is None or x['t'] >= since_t]

    def first(self, s: str, pred, since_t: float, timeout: float, what: str) -> dict:
        return self.until(lambda: next((x for x in self.images(s, since_t) if pred(x)), None), timeout, what)

    def robot_requests(self, s: str, since_t: float, until_t: float = None) -> list:
        return [x for x in self.rec.snap(f'request:{s}')
                if x['t'] >= since_t and (until_t is None or x['t'] <= until_t)]

    def service_view(self, s: str) -> dict:
        """The cell's operator view of `s` (its `/cell_io/state` entry: state, counters, ...), from a state published
        AFTER this call (never a stale view)."""
        t = time.monotonic()
        state = self.until(lambda: next((c for c in reversed(self.rec.snap('cell')) if c['t'] > t), None), 3.0,
                           'a fresh cell state')
        return dict(state['state'].get('services', {}).get(s, {}))

    def counters(self, s: str) -> dict:
        """The cell's counters of `s`, from a state published AFTER this call (never a stale view)."""
        return dict(self.service_view(s).get('counters', {}))

    def state_of(self, s: str):
        img = self.rec.last(f'status:{s}')
        return None if img is None else img['state']

    def inject(self, fault: str, **params) -> None:
        self.rec.cell_command(dict({'cmd': 'inject', 'fault': fault}, **params))

    def cell_reset(self, s: str) -> None:
        self.rec.cell_command({'cmd': 'reset', 'service': s})

    def cli(self, *args: str) -> tuple:
        """`cell_io_cli <args>` (the operator's own tool) on the launch domain: (exit code, output)."""
        log = self.logs / f'cli_{len(list(self.logs.glob("cli_*.log"))) + 1}.log'
        command = 'ros2 run trainit_cell_io cell_io_cli ' + ' '.join(shlex.quote(a) for a in args)
        proc = self._start(command, log)
        try:
            rc = proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            _cl().CompositeLaunch._stop_outcome(proc)
            rc = None
        out = log.read_text(errors='replace') if log.exists() else ''
        self.say(f'[cli] {command} -> {rc}: {out.strip().splitlines()[-1] if out.strip() else ""}')
        return rc, out

    def drop_extra(self) -> None:
        """Destroy the second requesters, then wait until the ROBOT no longer counts them: the recorder sees the graph
        at once, the robot's registry recounts every 100 ms (a job started sooner meets a correct 310: run 1)."""
        if not self.extra:
            return
        for pub in self.extra:
            self.rec.drop(pub)
        self.extra = []
        self.until(lambda: all(self.rec.count_publishers(f'/{s}/request') <= (1 if self.server else 0)
                               for s in SERVICES), 10, 'the second requester gone from the graph')
        time.sleep(1.0)

    def settle(self, tag: str = 'public') -> None:
        """Back to a known state: the cell up on `tag`'s file, no injection, no second requester, every served service
        READY or DONE, the robot ready (a RESET when FAULTED)."""
        self.drop_extra()
        if self.cell is None or self.cell.poll() is not None or self.cell_tag != tag:
            if self.cell is not None and self.cell.poll() is None:
                self.stop_cell()
            self.start_cell(tag)
        self.inject('clear')
        served = self.served(tag)
        for s in served:
            if self.state_of(s) == FAULT:
                self.cell_reset(s)
        self.until(lambda: all(self.state_of(s) in (READY, DONE) for s in served), 30,
                   'every service READY or DONE')
        self.until(lambda: all(self.rec.count_publishers(f'/{s}/request') <= (1 if self.server else 0)
                               for s in SERVICES), 10, 'no second requester')
        if self.plc is not None and self.server is not None:
            self.reset_robot()

    def scene(self, name: str, fn, settle_tag: str = 'public') -> None:
        self.say(f'--- scene {name}')
        try:
            fn()
        except Exception as exc:                           # noqa: BLE001 -- recorded; the next scene still runs
            self.errors[name] = f'{type(exc).__name__}: {exc}'
            self.say(f'[scene {name}] ERROR {type(exc).__name__}: {exc}')
        try:
            if settle_tag is not None:
                self.settle(settle_tag)
        except Exception as exc:                           # noqa: BLE001
            self.errors[f'{name} (settle)'] = f'{type(exc).__name__}: {exc}'
            self.say(f'[scene {name}] SETTLE ERROR {type(exc).__name__}: {exc}')

    # --- a machine job's record (request -> BUSY -> DONE -> the wait's exit)
    def machine_job(self, task: str, s: str, step_req: str, step_wait: str, timeout: float = 60.0) -> dict:
        c0 = self.counters(s)
        t0 = time.monotonic()
        prev = self.rec.last(f'status:{s}')
        job = self.job(task, timeout)
        return self.machine_record(job, s, step_req, step_wait, t0, c0, prev)

    def machine_record(self, job: dict, s: str, step_req: str, step_wait: str, t0: float, c0: dict,
                       prev: dict = None) -> dict:
        reqs = self.robot_requests(s, t0, job.get('t_end'))
        req = reqs[0] if reqs else None
        ident = (req['rs'], req['seq']) if req else None
        imgs = self.images(s, t0)
        busy = next((x for x in imgs if ident and x['cycle'] == ident and x['state'] == BUSY), None)
        done = next((x for x in imgs if ident and x['cycle'] == ident and x['done']), None)
        ent_r, ext_r, proc = self.step(job, step_req) if step_req else (None, None, None)
        ent_w, ext_w, _ = self.step(job, step_wait) if step_wait else (None, None, None)
        c1 = self.counters(s)
        return dict(
            job=job['id'], task=job['task'], result=job['result'], code=job['code'], detail=job['detail'],
            state_after=job['state_after'], ident=list(ident) if ident else None, requests_written=len(reqs),
            request_src=req['src'] if req else None,
            previous=dict(state=STATE_NAME.get(prev['state']), cycle=list(prev['cycle']), done=prev['done'])
            if prev else None,
            request_to_busy_ms=ms(proc and proc['stamp'], busy and busy['stamp']),
            request_to_done_ms=ms(proc and proc['stamp'], done and done['stamp']),
            busy_to_done_ms=ms(busy and busy['stamp'], done and done['stamp']),
            done_to_wait_exit_ms=ms(done and done['stamp'], ext_w and ext_w['stamp']),
            wait_entered_to_exit_ms=ms(ent_w and ent_w['stamp'], ext_w and ext_w['stamp']),
            wait_entered_before_done=(ent_w['stamp'] < done['stamp']) if (ent_w and done) else None,
            wait_exit_after_own_done=(ext_w['stamp'] >= done['stamp']) if (ext_w and done) else None,
            request_step_ms=ms(ent_r and ent_r['stamp'], ext_r and ext_r['stamp']),
            wait_exit_to_job_end_ms=ms(ext_w and ext_w['stamp'], job.get('stamp_end')),
            wait_status=ext_w['status'] if ext_w else None,
            process_event=dict(kind=proc['pkind'], channel=proc['channel'], job=proc['job']) if proc else None,
            station_states=collapse([STATE_NAME.get(x['state']) for x in imgs if x['t'] <= job['t_end']]),
            cycles={k: c1.get(k, 0) - c0.get(k, 0) for k in set(c0) | set(c1)},
            running=job['running'], plc=job['plc'],
            machine_states=collapse([w for d in job['running'] + [job['detail']] for w in machine_words(d)]))

    # =================================================================================================================
    # The M tier (P0B_IMPLEMENTATION_PLAN.md, WP6): scenes 0-9
    # =================================================================================================================
    def run_m(self) -> 'MachineRun':
        if self.guard('before'):
            raise RuntimeError(f'ROS domain {self.domain} is not empty: refusing to launch on it')
        self.rec = Recorder(self.domain, self.ros_log)
        try:
            if not self.start_bringup():
                raise SceneError('the bring-up never became ready')
            time.sleep(2.0)
            self.start_cell('typo')
            self.scene('0 (PB-P2b, PB-P18: no robot task server yet)', self.scene_0, settle_tag=None)
            self.scene('1 (PB-P8: the server starts beside a cell that does not serve indexer)', self.scene_1,
                       settle_tag='typo')
            self.scene('2 (PB-P19, N3 ready)', self.scene_2)
            self.scene('3 (N1, N3 done, PB-P1, N2, PB-P12, E1 / BW1 run, PB-P4)', self.scene_3)
            for name, fn in (('F4', self.scene_f4), ('F5', self.scene_f5), ('F2 + O2', self.scene_f2),
                             ('F3', self.scene_f3), ('F1 + F6', self.scene_f1_f6), ('PB-P23', self.scene_pbp23),
                             ('F8 + PB-P2a', self.scene_f8), ('PB-P14', self.scene_pbp14),
                             ('PB-P24', self.scene_pbp24), ('PB-P13', self.scene_pbp13), ('PB-P22a', self.scene_pbp22a),
                             ('PB-P22b', self.scene_pbp22b), ('PB-P11', self.scene_pbp11),
                             ('PB-P11 + the robot\'s image re-delivered', self.scene_pbp11_redelivery),
                             ('PB-P3a', self.scene_pbp3a), ('PB-P3b', self.scene_pbp3b),
                             ('PB-P10', self.scene_pbp10)):
                self.scene(f'4 ({name})', fn)
            self.scene('7 (N4: the fast cell)', self.scene_n4, settle_tag='fast')
            self.scene('7 (R1: 100 cycles, seeded random durations)', self.scene_r1, settle_tag='r1')
            self.scene('8 (F7: the cell killed during a wait; F9: no live source)', self.scene_f7_f9)
        finally:
            try:
                self.stop_all()
            finally:
                self.rec.close()
                self.rec = None
                self.guard('after')
        self.say(f'[shutdown] {self.stops}')
        self.say(f'[errors] {self.errors or "none"}')
        return self

    # --- scene 0: PB-P2(b) (a scripted single requester) and PB-P18 (the one-shot run), before the server serves
    def scene_0(self) -> None:
        s = 'turner'
        self.until(lambda: self.rec.count_publishers(f'/{s}/request') == 0, 10, 'no requester on turner')
        src = self.rec.last(f'status:{s}')['session']
        c0 = self.counters(s)
        pub = self.rec.requester(s)
        self.extra.append(pub)
        self.until(lambda: self.rec.count_publishers(f'/{s}/request') == 1, 5, 'the scripted requester visible')
        time.sleep(0.5)                                       # the cell matched it and counted one requester
        rs = SCRIPTED_S1
        t1 = time.monotonic()
        self.rec.publish_request(pub, src, rs, 1)
        busy = self.first(s, lambda x: x['cycle'] == (rs, 1) and x['state'] == BUSY, t1, 3, 'S1 accepted')
        t2 = time.monotonic()
        stamp2 = time.time()
        self.rec.publish_request(pub, src, rs, 2)
        rej = self.first(s, lambda x: x['reject'] == (rs, 2), t2, 3, 'S2 consumed')
        done = self.first(s, lambda x: x['cycle'] == (rs, 1) and x['done'], t1, 10, 'S1 DONE')
        c1 = self.counters(s)
        self.drop_extra()
        self.until(lambda: self.rec.count_publishers(f'/{s}/request') == 0, 5, 'the scripted requester gone')
        self.records['PB-P2b'] = dict(
            # the robot's rule (§3.4) maps the rejection to S2's waiter's code: a mapping, never a measured robot waiter
            reject_code=rej['rcode'], rule_code_of_s2=REJECT_TO_CODE.get(rej['rcode']),
            rejection_ms=ms(stamp2, rej['stamp']),
            at_rejection=dict(state=STATE_NAME.get(rej['state']), cycle=list(rej['cycle']), done=rej['done']),
            waiter_of_s1='SUCCESS' if done else None, busy_to_done_ms=ms(busy['stamp'], done['stamp']),
            second_cycle=any(x['cycle'] == (rs, 2) for x in self.images(s, t1)),
            cycles={k: c1.get(k, 0) - c0.get(k, 0) for k in ('accepted', 'completed', 'rejected_busy', 'aborted')})
        # PB-P18: the one-shot run (main_task turn: its FIRST step is a request), its registry just subscribed
        CL = _cl()
        log = self.logs / 'run_pbp18.log'
        t0 = time.monotonic()
        command = (f'ros2 launch {self.prefix}_app trainit_bt.launch.py use_sim_time:=false groot_monitor:=false')
        self.commands['run'] = command
        proc = self._start(command, log)
        finished = CL._wait_log(log, r'BT finished: (SUCCESS|FAILURE)', 120)
        outcome = CL.CompositeLaunch._stop_outcome(proc)
        self.stops['one-shot run'] = outcome
        text = log.read_text(errors='replace') if log.exists() else ''
        events = [e for e in self.rec.snap('events') if e['t'] >= t0]
        ent = next((e for e in events if e['kind'] == STEP_ENTERED and e['step'] == 'ask_turn'), None)
        ext = next((e for e in events if e['kind'] == STEP_EXITED and e['step'] == 'ask_turn'), None)
        written = self.robot_requests(s, t0)
        self.records['PB-P18'] = dict(
            finished=bool(finished), verdict='SUCCESS' if 'BT finished: SUCCESS' in text else
            ('FAILURE' if 'BT finished: FAILURE' in text else None), stop=outcome,
            request_step_ms=ms(ent and ent['stamp'], ext and ext['stamp']),
            request_status=ext['status'] if ext else None, written=len(written),
            machine_lines=re.findall(r'MACHINE: (.*)', text),
            startup_310=bool(re.search(r'STATION_UNAVAILABLE|UNAVAILABLE', text)))

    # --- scene 1: PB-P8 (the server's start diagnostic names the unserved service; its request 310)
    def scene_1(self) -> None:
        CL = _cl()
        log = self.start_server()
        seen = CL._wait_log(log, r"machine service 'turner': status seen", 20)
        warn = CL._wait_log(log, r"no status for machine service 'indexer'", 20)
        self.start_plc('main')
        self.reset_robot()
        t0 = time.monotonic()
        j = self.job('index', 30)
        text = log.read_text(errors='replace')
        self.records['PB-P8'] = dict(
            diagnostic_seen=bool(seen), diagnostic_warn=bool(warn),
            diagnostic_lines=[ln for ln in text.splitlines() if "machine service '" in ln][:6],
            result=j['result'], code=j['code'], detail=j['detail'], state_after=j['state_after'],
            written=len(self.robot_requests('indexer', t0)), plc=j['plc'])
        self.reset_robot()
        k = self.job('turn', 30)
        self.records['PB-P8']['turn_after'] = [k['result'], k['code']]

    # --- scene 2: PB-P19 (a healthy cell restart between jobs, no RESET) and N3 on a READY source
    def scene_2(self) -> None:
        before = self.rti_now()
        resets_before = sum('-> RESET' in t for _x, t in self.plc.said())
        old = (self.rec.last('status:turner') or {}).get('session')
        self.stop_cell()
        self.start_cell('public')
        new = self.rec.last('status:turner')['session']
        rec = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        rec.update(old_session=old, new_session=new, robot_before=dict(state=before['state'], ready=before['ready']),
                   resets_between=sum('-> RESET' in t for _x, t in self.plc.said()) - resets_before)
        self.records['PB-P19'] = rec
        # N3 (READY): the turner reset -> HOMING -> READY; a wait for ready with no request
        self.cell_reset('turner')
        self.until(lambda: self.state_of('turner') == READY, 10, 'turner READY after its reset')
        self.records['N3'] = dict(ready=self._ready_check())

    def _ready_check(self) -> dict:
        t0 = time.monotonic()
        state = STATE_NAME.get(self.state_of('turner'))
        j = self.job('ready_check', 15)
        ent, ext, _ = self.step(j, 'may_load')
        return dict(state_at_start=state, result=j['result'], code=j['code'], detail=j['detail'],
                    decided_ms=ms(ent and ent['stamp'], ext and ext['stamp']),
                    written=len(self.robot_requests('turner', t0)) + len(self.robot_requests('indexer', t0)),
                    running=j['running'], plc=j['plc'], job=j['id'],
                    machine_states=collapse([w for d in j['running'] + [j['detail']] for w in machine_words(d)]))

    # --- scene 3: the nominal block
    def scene_3(self) -> None:
        self.records['N1'] = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        self.records['N3']['done_idle'] = self._ready_check()                     # the source DONE, idle
        self.records['PB-P1'] = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        self.records['N2'] = self.machine_job('index', 'indexer', 'ask_index', 'indexed', 30)
        self.records['PB-P12'] = self.machine_job('turn_then_ready', 'turner', 'ask_turn', 'turner_ready', 30)
        # E1 + the BW1 run: the realistic task (pick, place on the station, request, move clear, wait, ...)
        t0 = time.monotonic()
        c0 = {s: self.counters(s) for s in SERVICES}
        j = self.job('turn_part', 90)
        turner = self.machine_record(j, 'turner', 'ask_turn', 'turned', t0, c0['turner'])
        indexer = self.machine_record(j, 'indexer', 'ask_index', 'indexed', t0, c0['indexer'])
        steps = []
        for e in j['events']:
            if e['kind'] in (STEP_ENTERED, STEP_EXITED, PROCESS_EVENT):
                steps.append(dict(kind=e['kind'], step=e['step'], step_kind=e['step_kind'], status=e['status'],
                                  pkind=e['pkind'], channel=e['channel'], job=e['job']))
        self.records['E1'] = dict(job=j['id'], result=j['result'], code=j['code'], steps=steps, turner=turner,
                                  indexer=indexer, all_events_job_id=all(e['job'] == j['id'] for e in j['events']),
                                  n_events=len(j['events']))
        self.records['BW1'] = dict(run_result=[j['result'], j['code']],
                                   step_kinds=sorted({e['step_kind'] for e in j['events'] if e['step_kind']}))
        # PB-P4: the robot's last turner request image re-delivered x20 (level-held): no new cycle
        last = self.rec.last('request:turner')
        v0 = self.service_view('turner')
        c0 = dict(v0.get('counters', {}))
        t1 = time.monotonic()
        pub = self.rec.requester('turner')
        self.extra.append(pub)
        self.until(lambda: self.rec.count_publishers('/turner/request') == 2, 5, 'the re-delivering publisher')
        time.sleep(0.3)
        for _ in range(20):
            self.rec.publish_request(pub, last['src'], last['rs'], last['seq'])
            time.sleep(0.02)
        time.sleep(1.0)
        v1 = self.service_view('turner')
        c1 = dict(v1.get('counters', {}))
        self.drop_extra()
        self.records['PB-P4'] = dict(
            deliveries=20, identity=[last['rs'], last['seq']],
            # PUBLISHED: the recorder's own subscription saw the identity re-published (the same DDS participant as
            # the publisher -- it proves the sending, not the cell's receipt)
            observed=sum(1 for x in self.rec.snap('request:turner')
                         if x['t'] >= t1 and (x['src'], x['rs'], x['seq']) == (last['src'], last['rs'], last['seq'])),
            # RECEIVED and IGNORED: the cell's own counts of request images handed in, and of those that re-delivered
            # an identity it had consumed already (trainit_cell_io: services.<s>.request_images / ignored_redelivered)
            received=_delta(v0, v1, 'request_images'), ignored=_delta(v0, v1, 'ignored_redelivered'),
            cycles={k: c1.get(k, 0) - c0.get(k, 0) for k in set(c0) | set(c1)},
            busy_images=sum(1 for x in self.images('turner', t1) if x['state'] == BUSY))

    # --- scene 4: faults on the public cell
    def scene_f4(self) -> None:
        self.inject('slow', device='table', factor=3)
        self.records['F4'] = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)

    def scene_f5(self) -> None:
        self.inject('slow', device='clamp', factor=6)
        self.inject('slow', device='table', factor=3.5)
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.job('turn', 30)
        rec = self.machine_record(j, 'turner', 'ask_turn', 'turned', t0, c0)
        ident = tuple(rec['ident']) if rec['ident'] else None
        late = self.first('turner', lambda x: ident and x['cycle'] == ident and x['done'], t0, 15, 'the late DONE')
        time.sleep(1.0)
        after = [x for x in self.rec.snap('rti') if x['job'] == j['id'] and x['t'] > late['t']]
        rec.update(late_done_after_end_ms=ms(j['stamp_end'], late['stamp']),
                   results_after_late_done=sorted({(x['result'], x['code']) for x in after}),
                   plc_results=[t for _x, t in self.plc.said() if f'Job {j["id"]} ' in t and 'finished' in t])
        self.records['F5'] = rec

    def scene_f2(self) -> None:
        self.inject('hold', service='turner')
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.job('turn', 30)
        rec = self.machine_record(j, 'turner', 'ask_turn', 'turned', t0, c0)
        ent_w, ext_w, _ = self.step(j, 'turned')
        rec['wait_ms'] = ms(ent_w and ent_w['stamp'], ext_w and ext_w['stamp'])
        # O2: each machine-state change -> the RTI detail; the fake PLC's first wait line; the RTI heartbeat
        rti = [x for x in self.rec.snap('rti') if x['t'] >= t0 and x['t'] <= j['t_end'] + 0.1]
        # the last image BEFORE the window is the baseline: the window's first change (DONE -> BUSY when the request is
        # taken) is measured even when no periodic image fell between t0 and it (M run 4: the window began at BUSY)
        imgs = [x for x in self.rec.snap('status:turner') if x['t'] < t0][-1:] + self.images('turner', t0)
        changes = []
        for a, b in zip(imgs, imgs[1:]):
            if b['state'] != a['state'] and b['t'] <= j['t_end']:
                word = STATE_NAME.get(b['state'])
                shown = next((x for x in rti if x['stamp'] >= b['stamp'] - 0.0005 and word in x['detail']), None)
                changes.append(dict(state=word, to_rti_detail_ms=ms(b['stamp'], shown and shown['stamp'])))
        wait_lines = [(t, text) for t, text in j['plc_timed']
                      if text.startswith(f'Robot busy: job {j["id"]}: ')
                      and re.search(r'WAITING FOR MACHINE|MACHINE \w+ BUSY', text)]
        first_wait_line = wait_lines[0] if wait_lines else None
        beats = []
        for a, b in zip(rti, rti[1:]):
            if b['hb'] != a['hb'] and b['session'] == a['session']:
                beats.append(b['stamp'])
        gaps = [round((y - x) * 1000.0, 1) for x, y in zip(beats, beats[1:])]
        rec['o2'] = dict(changes=changes,
                         plc_wait_line_ms=round(day_delta(seconds_of_day(ent_w['stamp']), first_wait_line[0]) * 1000,
                                                1) if (first_wait_line and ent_w) else None,
                         plc_wait_line=first_wait_line[1] if first_wait_line else None,
                         rti_heartbeat_max_gap_ms=max(gaps) if gaps else None, rti_heartbeats=len(beats),
                         wait_ms=ms(ent_w and ent_w['stamp'], ext_w and ext_w['stamp']))
        self.records['F2'] = rec
        self.inject('clear')
        ident = tuple(rec['ident']) if rec['ident'] else None
        self.first('turner', lambda x: ident and x['cycle'] == ident and x['done'], t0, 15, 'the held cycle DONE')

    def scene_f3(self) -> None:
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        t1 = time.monotonic()
        self.inject('station_fault', service='turner')
        fault = self.first('turner', lambda x: x['state'] == FAULT, t1, 5, 'FAULT')
        job = self.end_of(j, 30)
        rec = self.machine_record(job, 'turner', 'ask_turn', 'turned', t0, c0)
        _e, ext_w, _p = self.step(job, 'turned')
        rec.update(fault_code=fault['fcode'], fault_to_wait_exit_ms=ms(fault['stamp'], ext_w and ext_w['stamp']))
        self.records['F3'] = rec

    def scene_f1_f6(self) -> None:
        rc, out = self.cli('inject', 'stuck_actuator', 'device=clamp')
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.job('turn', 30)
        rec = self.machine_record(j, 'turner', 'ask_turn', 'turned', t0, c0)
        fault = next((x for x in self.images('turner', t0) if x['state'] == FAULT), None)
        rec.update(cli_inject=[rc, out.strip().splitlines()[-1] if out.strip() else ''],
                   fault_code=fault['fcode'] if fault else None,
                   robot_faulted=self.rti_now()['state'] == FAULTED,
                   plc_faulted=[t for t in j['plc'] if t.startswith('Robot FAULTED')])
        if fault and rec.get('ident'):
            busy = next((x for x in self.images('turner', t0) if x['cycle'] == tuple(rec['ident'])
                         and x['state'] == BUSY), None)
            rec['busy_to_fault_ms'] = ms(busy and busy['stamp'], fault['stamp'])
        self.records['F1'] = rec
        # F6: the OPERATOR clears and resets the station (the cell's CLI), the PLC resets the robot; the next job runs
        resets0 = self.counters('turner').get('resets', 0)
        rc1, out1 = self.cli('inject', 'clear')
        rc2, out2 = self.cli('reset', 'turner')
        self.until(lambda: self.state_of('turner') == READY, 15, 'turner READY after the operator RESET')
        resets1 = self.counters('turner').get('resets', 0)
        self.reset_robot()
        k = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        resets2 = self.counters('turner').get('resets', 0)
        self.records['F6'] = dict(cli=[[rc1, out1.strip()[-120:]], [rc2, out2.strip()[-120:]]],
                                  operator_resets=resets1 - resets0, resets_by_anyone_else=resets2 - resets1,
                                  control_publishers=self.rec.publisher_nodes('/cell_io/control'),
                                  next=dict(result=k['result'], code=k['code'], cycles=k['cycles']))

    def scene_pbp23(self) -> None:
        self.inject('slow', device='table', factor=3)
        t0 = time.monotonic()
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        busy = self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        self.until(lambda: time.monotonic() > busy['t'] + 2.8, 5, 'the table well on its way')
        self.inject('station_fault', service='turner')
        self.end_of(j, 30)
        self.inject('clear')
        self.inject('slow', device='table', factor=3)        # the homing is slow too: the table returns from ~180
        t1 = time.monotonic()
        self.cell_reset('turner')
        homing = self.first('turner', lambda x: x['state'] == HOMING, t1, 3, 'HOMING')
        self.reset_robot()
        c0 = self.counters('turner')
        t2 = time.monotonic()
        state_at_start = STATE_NAME.get(self.state_of('turner'))
        k = self.job('turn', 30)
        ent, ext, _p = self.step(k, 'ask_turn')
        written = self.robot_requests('turner', t2, k['t_end'])
        ready = self.first('turner', lambda x: x['state'] == READY, t1, 15, 'READY after the homing')
        time.sleep(1.5)
        c1 = self.counters('turner')
        self.records['PB-P23'] = dict(
            state_at_request=state_at_start, homing_seen=bool(homing), result=k['result'], code=k['code'],
            detail=k['detail'], written=len(written), decided_ms=ms(ent and ent['stamp'], ext and ext['stamp']),
            homing_to_ready_ms=ms(homing['stamp'], ready['stamp']), request_before_ready=k['t_end'] < ready['t'],
            cycles_after={x: c1.get(x, 0) - c0.get(x, 0) for x in ('accepted', 'rejected_busy', 'completed')},
            state_after=STATE_NAME.get(self.state_of('turner')))

    def scene_f8(self) -> None:
        self.inject('hold', service='turner')
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        self.until(lambda: any(x['job'] == j and x['state'] == RUNNING and 'BUSY' in x['detail']
                               for x in self.rec.snap('rti')), 5, 'the wait running on BUSY')
        time.sleep(0.5)
        self.plc.send(f'cancel {j}')
        job = self.end_of(j, 30)
        stopping = next((x for x in self.rec.snap('rti') if x['job'] == j and x['state'] == STOPPING), None)
        _e, ext_w, _p = self.step(job, 'turned')
        # PB-P2(a): the NEXT job while the station is still BUSY (held)
        t1 = time.monotonic()
        state_at_next = STATE_NAME.get(self.state_of('turner'))
        self.reset_robot()
        k = self.job('turn', 30)
        written = self.robot_requests('turner', t1, k['t_end'])
        ent_k, ext_k, _ = self.step(k, 'ask_turn')
        self.records['PB-P2a'] = dict(state_at_request=state_at_next, result=k['result'], code=k['code'],
                                      detail=k['detail'], written=len(written),
                                      decided_ms=ms(ent_k and ent_k['stamp'], ext_k and ext_k['stamp']))
        # the station finishes its own cycle once released; then the next job runs
        self.inject('clear')
        done = self.first('turner', lambda x: x['cycle'] == ident and x['done'], t0, 15, 'the cancelled job\'s cycle')
        c1 = self.counters('turner')
        self.reset_robot()
        n = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        self.records['F8'] = dict(
            result=job['result'], code=job['code'], detail=job['detail'], wait_status=ext_w and ext_w['status'],
            halt_ms=ms(stopping and stopping['stamp'], ext_w and ext_w['stamp']), next_code=k['code'],
            own_cycle_completed=bool(done),
            cycles={x: c1.get(x, 0) - c0.get(x, 0) for x in ('accepted', 'completed', 'aborted')},
            after_done=dict(result=n['result'], code=n['code']), plc=job['plc'])

    def scene_pbp14(self) -> None:
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.start_job('turn')
        self.until(lambda: any(e['job'] == j and e['kind'] == PROCESS_EVENT and e['pkind'] == 'request'
                               for e in self.rec.snap('events')), 10, 'the request written')
        self.plc.send(f'cancel {j}')
        job = self.end_of(j, 30)
        req = next(iter(self.robot_requests('turner', t0)), None)
        ident = (req['rs'], req['seq']) if req else None
        t1 = time.monotonic()
        self.reset_robot()
        k = self.job('turn', 30)
        written = self.robot_requests('turner', t1, k['t_end'])
        done = self.first('turner', lambda x: ident and x['cycle'] == ident and x['done'], t0, 15,
                          'the first request\'s cycle DONE')
        time.sleep(0.5)
        c1 = self.counters('turner')
        self.records['PB-P14'] = dict(
            first=dict(result=job['result'], code=job['code']),
            next=dict(result=k['result'], code=k['code'], detail=k['detail'], written=len(written)),
            first_cycle_done=bool(done),
            cycles={x: c1.get(x, 0) - c0.get(x, 0) for x in ('accepted', 'completed', 'aborted', 'rejected_busy')})

    def scene_pbp24(self) -> None:
        """PB-P24 (ADDED 2026-11-19 by the owner's decision, D19 -- the amendment of P0B_DESIGN §3.4): job A `turn`
        writes its request and is cancelled at once; the cell takes that request late (consume_delay 1500 ms); job B
        `ready_check` (its FIRST step a ready-wait, 2.0 s) starts right after A ended. B never succeeds on the READY of
        before A's request is taken: it waits ("it has not yet taken the previous request"), then A's cycle's BUSY
        keeps it waiting; on the public cell its 2.0 s timeout falls inside A's cycle: 305 "(the machine is BUSY)".
        1500 ms: B's wait is entered well before A's request is taken, and its 2.0 s timeout still falls after it."""
        self.inject('consume_delay', service='turner', ms=1500)
        t0 = time.monotonic()
        c0 = self.counters('turner')
        a = self.start_job('turn')
        self.until(lambda: any(e['job'] == a and e['kind'] == PROCESS_EVENT and e['pkind'] == 'request'
                               for e in self.rec.snap('events')), 10, "A's request written")
        self.plc.send(f'cancel {a}')
        job_a = self.end_of(a, 30)
        req = next(iter(self.robot_requests('turner', t0)), None)
        ident = (req['rs'], req['seq']) if req else None
        self.reset_robot()                                     # a no-op unless the cancel left the robot not ready
        b = self.start_job('ready_check')
        job_b = self.end_of(b, 15)
        ent, ext, _ = self.step(job_b, 'may_load')
        taken = self.first('turner', lambda x: ident is not None and (x['cycle'] == ident or x['reject'] == ident),
                           t0, 10, "A's request taken")
        self.inject('clear')
        done = self.first('turner', lambda x: x['cycle'] == ident and x['done'], t0, 15, "A's cycle DONE")
        c1 = self.counters('turner')
        self.records['PB-P24'] = dict(
            a=dict(result=job_a['result'], code=job_a['code'], job=a), identity=list(ident) if ident else None,
            b=dict(result=job_b['result'], code=job_b['code'], detail=job_b['detail'], running=job_b['running'],
                   plc=job_b['plc'], job=b,
                   machine_states=collapse([w for d in job_b['running'] + [job_b['detail']]
                                            for w in machine_words(d)])),
            taken_state=STATE_NAME.get(taken['state']), taken_as_cycle=taken['cycle'] == ident,
            # the case's condition: B's ready-wait was entered BEFORE the cell took A's request (cell stamps and robot
            # event stamps: one machine, wall time)
            wait_entered_to_taken_ms=ms(ent and ent['stamp'], taken['stamp']),
            taken_to_wait_exit_ms=ms(taken['stamp'], ext and ext['stamp']),
            wait_exit_to_a_done_ms=ms(ext and ext['stamp'], done['stamp']),
            written_by_b=len(self.robot_requests('turner', self.jobs[b]['t0'], job_b['t_end'])),
            cycles={x: c1.get(x, 0) - c0.get(x, 0) for x in ('accepted', 'completed', 'aborted', 'rejected_busy')})

    def scene_pbp13(self) -> None:
        self.inject('consume_delay', service='turner', ms=2000)
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        self.inject('station_fault', service='turner')
        fault = self.first('turner', lambda x: x['state'] == FAULT, t0, 3, 'FAULT')
        job = self.end_of(j, 30)
        _e, ext_w, _p = self.step(job, 'turned')
        rej = self.first('turner', lambda x: x['reject'] == ident, t0, 5, 'the request consumed (rejected)')
        self.inject('clear')
        t1 = time.monotonic()
        self.cell_reset('turner')
        self.first('turner', lambda x: x['state'] == READY, t1, 15, 'READY after the RESET')
        time.sleep(1.5)
        c1 = self.counters('turner')
        self.records['PB-P13'] = dict(
            result=job['result'], code=job['code'], detail=job['detail'], reject_code=rej['rcode'],
            fault_to_wait_exit_ms=ms(fault['stamp'], ext_w and ext_w['stamp']),
            busy_of_it=any(x['cycle'] == ident for x in self.images('turner', t0)),
            cycles={x: c1.get(x, 0) - c0.get(x, 0) for x in ('accepted', 'rejected_faulted', 'completed')})

    def scene_pbp22a(self) -> None:
        t0 = time.monotonic()
        j = self.start_job('turn_part')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 60, "the robot's request")
        ident = (req['rs'], req['seq'])
        self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        t1 = time.monotonic()
        self.inject('station_fault', service='turner')
        fault = self.first('turner', lambda x: x['state'] == FAULT, t1, 3, 'FAULT')
        self.inject('clear')
        self.cell_reset('turner')
        reset_img = self.first('turner', lambda x: x['state'] in (HOMING, READY) and x['t'] >= fault['t'], t1, 3,
                               'HOMING after the RESET')
        job = self.end_of(j, 60)
        ent_w, ext_w, _p = self.step(job, 'turned')
        self.records['PB-P22a'] = dict(
            result=job['result'], code=job['code'], detail=job['detail'],
            reset_before_wait=(reset_img['stamp'] < ent_w['stamp']) if ent_w else None,
            decided_ms=ms(ent_w and ent_w['stamp'], ext_w and ext_w['stamp']),
            state_at_wait=STATE_NAME.get(reset_img['state']))

    def scene_pbp22b(self) -> None:
        out = {}
        # (i) TWO_REQUESTERS: the request held back (consume_delay), a silent second requester appears meanwhile
        self.inject('consume_delay', service='turner', ms=2000)
        t0 = time.monotonic()
        j = self.start_job('turn_then_ready')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        pub = self.rec.requester('turner')
        self.extra.append(pub)
        job = self.end_of(j, 30)
        _e, ext_w, _p = self.step(job, 'turner_ready')
        rej = next((x for x in self.images('turner', t0) if x['reject'] == ident), None)
        out['two_requesters'] = dict(result=job['result'], code=job['code'], detail=job['detail'],
                                     reject_code=rej and rej['rcode'],
                                     reject_to_wait_exit_ms=ms(rej and rej['stamp'], ext_w and ext_w['stamp']),
                                     running=job['running'])
        self.drop_extra()
        self.inject('clear')
        self.until(lambda: self.rec.count_publishers('/turner/request') == 1, 5, 'the second requester gone')
        self.reset_robot()
        # (ii) FAULTED: the request held back, the station faults meanwhile
        self.inject('consume_delay', service='turner', ms=2000)
        t1 = time.monotonic()
        j = self.start_job('turn_then_ready')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t1)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        self.inject('station_fault', service='turner')
        fault = self.first('turner', lambda x: x['state'] == FAULT, t1, 3, 'FAULT')
        job = self.end_of(j, 30)
        _e, ext_w, _p = self.step(job, 'turner_ready')
        rej = self.first('turner', lambda x: x['reject'] == ident, t1, 5, 'the request consumed (rejected)')
        out['faulted'] = dict(result=job['result'], code=job['code'], detail=job['detail'], reject_code=rej['rcode'],
                              fault_to_wait_exit_ms=ms(fault['stamp'], ext_w and ext_w['stamp']))
        self.records['PB-P22b'] = out

    def scene_pbp11(self) -> None:
        pub = self.rec.requester('turner')
        self.extra.append(pub)
        self.until(lambda: self.rec.count_publishers('/turner/request') == 2, 5, 'the second requester visible')
        time.sleep(1.0)                                        # the robot's registry and the cell both count it
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.job('turn', 30)
        written = self.robot_requests('turner', t0, j['t_end'])
        src = self.rec.last('status:turner')['session']
        t1 = time.monotonic()
        self.rec.publish_request(pub, src, SECOND_REQUESTER, 1)
        rej = self.first('turner', lambda x: x['reject'] == (SECOND_REQUESTER, 1), t1, 3,
                         'the second requester rejected')
        time.sleep(1.0)
        c1 = self.counters('turner')
        self.records['PB-P11'] = dict(
            result=j['result'], code=j['code'], detail=j['detail'], written=len(written), reject_code=rej['rcode'],
            cycles={x: c1.get(x, 0) - c0.get(x, 0) for x in ('accepted', 'rejected_two_requesters')})

    def scene_pbp11_redelivery(self) -> None:
        """Right after PB-P11 (the source consumed the SECOND requester's image last): the robot's last consumed
        request image re-delivered x5 by a second publisher, then that publisher dropped -- no cycle. It exercises
        trainit_cell_io's per-requester memory: the robot's identity is the last one consumed FOR ITS REQUESTER, so
        the re-delivery is a no-op although it differs from the last identity the source consumed."""
        robot = [x for x in self.rec.snap('request:turner') if x['rs'] not in (SCRIPTED_S1, SECOND_REQUESTER)]
        if not robot:
            raise SceneError('no request image of the robot on turner to re-deliver')
        last = robot[-1]
        ident = (last['rs'], last['seq'])
        v0 = self.service_view('turner')
        c0 = dict(v0.get('counters', {}))
        t1 = time.monotonic()
        pub = self.rec.requester('turner')
        self.extra.append(pub)
        self.until(lambda: self.rec.count_publishers('/turner/request') == 2, 5, 'the re-delivering publisher')
        time.sleep(0.3)
        for _ in range(5):
            self.rec.publish_request(pub, last['src'], last['rs'], last['seq'])
            time.sleep(0.02)
        time.sleep(1.0)
        self.drop_extra()                                      # the robot's image is the source's latest now
        time.sleep(1.0)
        v1 = self.service_view('turner')
        c1 = dict(v1.get('counters', {}))
        self.records['PB-P11-redelivery'] = dict(
            deliveries=5, identity=list(ident),
            src_is_current=last['src'] == (self.rec.last('status:turner') or {}).get('session'),
            observed=sum(1 for x in self.rec.snap('request:turner')
                         if x['t'] >= t1 and (x['src'], x['rs'], x['seq']) == (last['src'], last['rs'], last['seq'])),
            received=_delta(v0, v1, 'request_images'), ignored=_delta(v0, v1, 'ignored_redelivered'),
            cycles={k: c1.get(k, 0) - c0.get(k, 0) for k in set(c0) | set(c1)},
            busy_images=sum(1 for x in self.images('turner', t1) if x['state'] == BUSY and x['cycle'] == ident))

    def scene_pbp3a(self) -> None:
        self.inject('hold', service='turner')
        t0 = time.monotonic()
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        busy = self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        time.sleep(0.5)
        t1 = time.monotonic()
        answer = self.rec.cell_command({'cmd': 'restart'})
        new = self.first('turner', lambda x: x['session'] != busy['session'], t1, 5, 'the new session')
        job = self.end_of(j, 30)
        _e, ext_w, _p = self.step(job, 'turned')
        time.sleep(1.5)
        c = self.counters('turner')
        state = STATE_NAME.get(self.state_of('turner'))
        self.inject('clear')
        self.reset_robot()
        t2 = time.monotonic()
        k = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        self.records['PB-P3a'] = dict(
            restart=answer.get('text'), old_session=busy['session'], new_session=new['session'],
            result=job['result'], code=job['code'], detail=job['detail'],
            new_session_to_wait_exit_ms=ms(new['stamp'], ext_w and ext_w['stamp']),
            new_session_counters=dict(accepted=c.get('accepted'), ignored_other_session=c.get('ignored_other_session')),
            state_after=state, stale_success=any(x['cycle'] == ident and x['done'] for x in self.images('turner', t0)),
            next=dict(result=k['result'], code=k['code'], request_src=k['request_src'], t=t2))

    def scene_pbp3b(self) -> None:
        self.inject('hold', service='turner')
        t0 = time.monotonic()
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        busy = self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        time.sleep(0.5)
        stop_outcome = self.stop_cell()                        # orderly (D11): the source GONE, then a new process
        job = self.end_of(j, 30)
        _e, ext_w, _p = self.step(job, 'turned')
        last_old = [x for x in self.images('turner', t0) if x['session'] == busy['session']][-1]
        self.start_cell('public')
        time.sleep(1.5)
        c = self.counters('turner')
        state = STATE_NAME.get(self.state_of('turner'))
        self.reset_robot()
        k = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        self.records['PB-P3b'] = dict(
            stop=stop_outcome, result=job['result'], code=job['code'], detail=job['detail'],
            last_old_image_to_wait_exit_ms=ms(last_old['stamp'], ext_w and ext_w['stamp']),
            new_session_counters=dict(accepted=c.get('accepted'), ignored_other_session=c.get('ignored_other_session')),
            state_after=state, stale_success=any(x['cycle'] == ident and x['done'] for x in self.images('turner', t0)),
            next=dict(result=k['result'], code=k['code'], request_src=k['request_src'],
                      new_session=self.rec.last('status:turner')['session']))

    def scene_pbp10(self) -> None:
        self.inject('hold', service='turner')
        t0 = time.monotonic()
        dead_job = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        old = (req['rs'], req['seq'])
        self.first('turner', lambda x: x['cycle'] == old and x['state'] == BUSY, t0, 5, 'BUSY')
        mark = len(self.plc.said())
        self.stop_server()                                     # the requesting process dies mid-job (orderly, D11)
        self.start_server()                                    # ... and comes back: a NEW requester session
        lost = self.plc.wait(r'JOB LOST', mark, 20)
        self.plc_logs[-1][1] = mark                            # D18: this PLC session's NEVER check stops here
        self.stop_plc()
        self.start_plc('after_server_restart')
        self.reset_robot()
        t1 = time.monotonic()
        k = self.job('turn', 30)
        ent, ext, _p = self.step(k, 'ask_turn')
        written = self.robot_requests('turner', t1, k['t_end'])
        self.inject('clear')
        done_old = self.first('turner', lambda x: x['cycle'] == old and x['done'], t0, 15, 'the dead request\'s DONE')
        self.reset_robot()
        n = self.machine_job('turn', 'turner', 'ask_turn', 'turned', 30)
        self.records['PB-P10'] = dict(
            dead_job=dead_job, job_lost_line=lost[2], dead_identity=list(old),
            first=dict(result=k['result'], code=k['code'], detail=k['detail'], written=len(written),
                       request_step_ms=ms(ent and ent['stamp'], ext and ext['stamp'])),
            dead_cycle_completed=bool(done_old),
            next=dict(result=n['result'], code=n['code'], ident=n['ident'], cycles=n['cycles'],
                      wait_exit_after_own_done=n['wait_exit_after_own_done']))

    # --- scene 7: N4 (DONE before the wait begins) and R1 (100 cycles)
    def scene_n4(self) -> None:
        if self.cell_tag != 'fast':
            self.stop_cell()
            self.start_cell('fast')
        self.reset_robot()
        t0 = time.monotonic()
        c0 = self.counters('turner')
        j = self.job('turn_part', 90)
        self.records['N4'] = self.machine_record(j, 'turner', 'ask_turn', 'turned', t0, c0)

    def scene_r1(self) -> None:
        if self.cell_tag != 'r1':
            self.stop_cell()
            self.start_cell('r1')
        self.reset_robot()
        c0 = self.counters('turner')
        t0 = time.monotonic()
        jobs = []
        for _ in range(R1_CYCLES):
            j = self.job('turn', 30)
            jobs.append(j)
            if j['result'] != SUCCEEDED:
                self.reset_robot()
        time.sleep(0.5)
        c1 = self.counters('turner')
        reqs = self.robot_requests('turner', t0)
        imgs = self.images('turner', t0)
        stale, lost, durations = [], [], []
        for j in jobs:
            mine = [r for r in reqs if j['t0'] <= r['t'] <= j['t_end']]
            if len(mine) != 1:
                lost.append(j['id'])
                continue
            ident = (mine[0]['rs'], mine[0]['seq'])
            busy = next((x for x in imgs if x['cycle'] == ident and x['state'] == BUSY), None)
            done = next((x for x in imgs if x['cycle'] == ident and x['done']), None)
            _e, ext_w, _p = self.step(j, 'turned')
            if done is None or ext_w is None or ext_w['stamp'] < done['stamp']:
                stale.append(j['id'])
            if busy and done:
                durations.append(round((done['stamp'] - busy['stamp']) * 1000.0, 1))
        self.records['R1'] = dict(
            cycles_run=len(jobs), ok=sum(1 for j in jobs if j['result'] == SUCCEEDED and j['code'] == OK),
            codes=sorted({j['code'] for j in jobs}), lost=lost, stale=stale,
            seqs_distinct=len({(r['rs'], r['seq']) for r in reqs}), requests=len(reqs),
            cycles={x: c1.get(x, 0) - c0.get(x, 0)
                    for x in ('accepted', 'completed', 'aborted', 'rejected_busy', 'rejected_faulted',
                              'rejected_two_requesters')},
            busy_to_done_ms=dict(min=min(durations) if durations else None, max=max(durations) if durations else None,
                                 n=len(durations), distinct=len(set(durations))))

    # --- scene 8: F7 (the source killed during a wait) then F9 (a request with no live source)
    def scene_f7_f9(self) -> None:
        if self.cell_tag != 'public' or self.cell is None:
            self.stop_cell()
            self.start_cell('public')
        self.reset_robot()
        self.inject('hold', service='turner')
        t0 = time.monotonic()
        j = self.start_job('turn')
        req = self.until(lambda: next(iter(self.robot_requests('turner', t0)), None), 10, "the robot's request")
        ident = (req['rs'], req['seq'])
        busy = self.first('turner', lambda x: x['cycle'] == ident and x['state'] == BUSY, t0, 5, 'BUSY')
        time.sleep(0.6)
        self.stop_cell(how='kill')
        job = self.end_of(j, 15)
        _e, ext_w, _p = self.step(job, 'turned')
        session = [x for x in self.images('turner', t0) if x['session'] == busy['session']]
        last_change = None
        for a, b in zip(session, session[1:]):
            if b['hb'] != a['hb']:
                last_change = b
        self.records['F7'] = dict(
            job=job['id'], result=job['result'], code=job['code'], detail=job['detail'], running=job['running'],
            plc=job['plc'],
            last_heartbeat_change_to_310_ms=ms(last_change and last_change['stamp'], ext_w and ext_w['stamp']),
            machine_states=collapse([w for d in job['running'] + [job['detail']] for w in machine_words(d)]),
            stale_success=any(x['cycle'] == ident and x['done'] for x in self.images('turner', t0)))
        # the SIGKILLed publisher leaves the graph with its DDS lease (D11)
        t_kill = time.monotonic()
        self.until(lambda: self.rec.count_publishers('/turner/status') == 0, 60, 'the dead status publisher gone')
        self.records['F7']['graph_lease_s'] = round(time.monotonic() - t_kill, 1)
        # F9: a request with no live status source
        self.reset_robot()
        t1 = time.monotonic()
        k = self.job('turn', 30)
        ent, ext, _p = self.step(k, 'ask_turn')
        self.records['F9'] = dict(result=k['result'], code=k['code'], detail=k['detail'],
                                  decided_ms=ms(ent and ent['stamp'], ext and ext['stamp']),
                                  written=len(self.robot_requests('turner', t1, k['t_end'])), plc=k['plc'])
        self.start_cell('public')
        self.reset_robot()
        n = self.job('turn', 30)
        self.records['F9']['after'] = [n['result'], n['code']]
