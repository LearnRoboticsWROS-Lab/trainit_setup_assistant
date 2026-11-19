"""The P0.b launch harness's own logic (test/p0b_machine_harness.py), WITHOUT ROS or a launch: what the opt-in M and I
tiers rely on and cannot show themselves (the Python review of the harness).

    stop_all   every process is stopped INDEPENDENTLY: a stop that raises is recorded as that process's outcome (never
               'clean'), its process group is SIGKILLed and the next stop still runs; an interrupted stop (Ctrl-C) still
               runs the others, then propagates
    end_of     a job's end waits, bounded, for its LAST event (RUN_ENDED, or its task's TASK_EXITED) and the fake PLC's
               "Job <j> (<task>) finished" line -- never a fixed sleep; neither within END_LINES_S: a SceneError naming
               what did not happen

The processes are a `sleep` (a real process group to kill) and stand-ins; the recorder and the fake PLC are fakes fed
from a thread.
"""
import subprocess
import threading
import time

import pytest

import p0b_machine_harness as H


class _Run(H.MachineRun):
    """A MachineRun with no workspace and no ROS: only the state stop_all and end_of read."""

    def __init__(self):                                   # no super(): nothing to launch
        self.lines, self.stops, self.jobs = [], {}, {}
        self.plc = self.server = self.cell = self.up = None
        self.rec = _Recorder()


class _Recorder:
    def __init__(self):
        self.lock = threading.Lock()
        self.data = {'rti': [], 'events': []}

    def add(self, key, item):
        with self.lock:
            self.data[key].append(item)

    def snap(self, key):
        with self.lock:
            return list(self.data[key])


class _Plc:
    def __init__(self):
        self.lines = []

    def said(self):
        return list(self.lines)


STOPS = (('fake PLC', 'plc', 'stop_plc'), ('robot task server', 'server', 'stop_server'), ('cell', 'cell', 'stop_cell'),
         ('bring-up', 'up', 'stop_bringup'))


def _stops(run, order, raising=None):
    """Stand-in stops (instance attributes over MachineRun's) that record their order; one named in `raising` raises
    before it drops its process, as CompositeLaunch._stop_outcome may (a group gone mid-stop, a wait timing out)."""
    raising = raising or {}

    def make(name, attr):
        def stop():
            order.append(name)
            if name in raising:
                raise raising[name]
            setattr(run, attr, None)
            run.stops[name] = 'clean'
            return 'clean'
        return stop
    for name, attr, method in STOPS:
        setattr(run, method, make(name, attr))


def test_stop_all_runs_every_stop_when_one_raises_and_kills_that_process_group():
    run, order = _Run(), []
    server = subprocess.Popen(['sleep', '60'], start_new_session=True)       # its stop raises: SIGKILLed instead
    cell = subprocess.Popen(['sleep', '60'], start_new_session=True)
    try:
        run.plc, run.server, run.cell, run.up = object(), server, cell, object()
        _stops(run, order, raising={'robot task server': ProcessLookupError('the group vanished mid-stop')})
        run.stop_all()
        assert order == ['fake PLC', 'robot task server', 'cell', 'bring-up']           # every stop ran, in order
        assert run.stops['robot task server (its stop raised)'] == 'ProcessLookupError: the group vanished mid-stop'
        assert run.stops['fake PLC'] == run.stops['cell'] == run.stops['bring-up'] == 'clean'
        assert server.poll() is not None, 'the process whose stop raised was left running'
        assert (run.plc, run.server, run.cell, run.up) == (None, None, None, None)
        assert any('its stop RAISED ProcessLookupError' in line for line in run.lines)
    finally:
        for proc in (server, cell):
            H.kill_group(proc)


def test_an_interrupted_stop_still_runs_the_others_then_propagates():
    run, order = _Run(), []
    run.plc, run.server, run.cell, run.up = object(), object(), object(), object()
    _stops(run, order, raising={'fake PLC': KeyboardInterrupt()})
    with pytest.raises(KeyboardInterrupt):
        run.stop_all()
    assert order == ['fake PLC', 'robot task server', 'cell', 'bring-up']


def test_nothing_held_nothing_stopped():
    run, order = _Run(), []
    _stops(run, order)
    run.stop_all()
    assert order == [] and run.stops == {}


def _ended_job(run, j=7, task='turn'):
    run.plc = _Plc()
    run.jobs[j] = dict(id=j, task=task, t0=time.monotonic(), plc_mark=0)
    run.rec.add('rti', dict(t=time.monotonic(), stamp=1.0, state=H.IDLE, ready=True, hb=1, session=1, job=j, task=2,
                            task_name=task, result=H.SUCCEEDED, code=H.OK, fault=0, detail='', ack=0, ack_seq=0,
                            ack_code=0))
    return j


def _later(delay, fn):
    timer = threading.Timer(delay, fn)
    timer.start()
    return timer


def test_end_of_waits_for_the_last_event_and_the_fake_plcs_finished_line():
    run = _Run()
    j = _ended_job(run)
    t0 = time.monotonic()
    timers = [_later(0.2, lambda: run.rec.add('events', dict(job=j, kind=H.RUN_ENDED, task='', step='', status=1))),
              _later(0.4, lambda: run.plc.lines.append((0.0, f'Job {j} (turn) finished: SUCCEEDED OK (100)')))]
    job = run.end_of(j, timeout=5)
    for timer in timers:
        timer.join()
    assert 0.35 <= time.monotonic() - t0 < H.END_LINES_S, 'end_of neither waited for both nor returned once seen'
    assert [e['kind'] for e in job['events']] == [H.RUN_ENDED]
    assert job['plc'] == [f'Job {j} (turn) finished: SUCCEEDED OK (100)'] and (job['result'], job['code']) == (1, 100)


def test_end_of_takes_the_tasks_exit_for_its_last_event_and_names_what_never_came(monkeypatch):
    run = _Run()
    j = _ended_job(run, task='turn_part')
    run.rec.add('events', dict(job=j, kind=H.TASK_EXITED, task='turn_part', step='', status=1))
    run.plc.lines.append((0.0, f'Job {j} (turn_part) finished: SUCCEEDED OK (100)'))
    assert run.end_of(j, timeout=5)['events'][0]['kind'] == H.TASK_EXITED
    # another job's lines, or another task's exit, never stand in for this job's: bounded, then named
    monkeypatch.setattr(H, 'END_LINES_S', 0.3)
    k = _ended_job(run, j=8, task='turn')
    run.rec.add('events', dict(job=k, kind=H.TASK_EXITED, task='turn_part', step='', status=1))
    run.plc.lines.append((0.0, f'Job {j} (turn) finished: SUCCEEDED OK (100)'))
    with pytest.raises(H.SceneError, match=rf'job {k} \(turn\): its RUN_ENDED event and the fake PLC'):
        run.end_of(k, timeout=5)
