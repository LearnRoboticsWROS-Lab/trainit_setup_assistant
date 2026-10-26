"""TSA's process launcher (vNext I0.2 / I0.11, ADR-0027): start, observe and stop the commands of
``composer.launch_plan``.

It launches and observes only. The processes are the same command lines a user would type; their
output is collected for the log pane and scanned for the runner's verdict. Closing TSA stops what TSA
started. Nothing here plans, executes motion or ticks a tree.

Each command runs in its OWN session (``start_new_session``: ``setsid()`` in the child before
``exec``), so its process group id IS its pid from the first instant. A stop then signals the whole
group, as Ctrl-C does in a terminal, never TSA's own group (vNext I0.11: a group id read from a QProcess
child before an external ``setsid`` ran was TSA's own).

Output is read by one thread per process and handed to the GUI thread by a timer (:meth:`Launcher.pump`),
so every listener runs in the GUI thread.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from collections import deque
from typing import Dict, List, Optional

from python_qt_binding.QtCore import QObject, QTimer

from ..composer.launch_plan import BT_FAILURE, BT_SUCCESS, Command, child_environment

MAX_LINES = 5000


class LaunchedProcess:
    """One started command: its process (and group), its output lines, and how it ended."""

    def __init__(self, command: Command):
        self.command = command
        self.lines: List[str] = []
        self.verdict: Optional[str] = None           # BT_SUCCESS / BT_FAILURE once seen
        self.exit_code: Optional[int] = None
        self.told = False                            # its ending was handed to the listeners
        self.stopped = False                         # TSA stopped it (not an ending of its own)
        self._incoming: deque = deque()
        argv = command.argv()
        if argv and argv[0] == 'setsid':
            argv = argv[1:]                          # the new session is made here, before exec
        self.popen = subprocess.Popen(
            argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            env=child_environment({**os.environ, **command.env}), start_new_session=True)
        self.pgid = self.popen.pid                   # session leader: pgid == pid, at once
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._reader.start()

    def _read(self) -> None:
        for raw in iter(self.popen.stdout.readline, b''):
            self._incoming.append(raw.decode('utf-8', errors='replace').rstrip('\n'))
        self.popen.stdout.close()

    def drain(self) -> List[str]:
        """The lines read since the last drain (GUI thread)."""
        out = []
        while self._incoming:
            line = self._incoming.popleft()
            self.lines.append(line)
            if BT_SUCCESS in line:
                self.verdict = BT_SUCCESS
            elif BT_FAILURE in line:
                self.verdict = BT_FAILURE
            out.append(line)
        if len(self.lines) > MAX_LINES:
            del self.lines[:len(self.lines) - MAX_LINES]
        return out

    def poll(self) -> Optional[int]:
        """The leader's exit code once it ended and its output was read -- or once its whole group is
        gone (a daemon that left the group may keep the pipe open for ever), else None."""
        if self.exit_code is None:
            code = self.popen.poll()
            if code is not None and (not self._reader.is_alive() or not _group_alive(self.popen.pid)):
                self.drain()
                self.exit_code = code
        return self.exit_code

    def running(self) -> bool:
        """The command, or a process of its group, still runs."""
        if self.popen.poll() is None:
            return True
        if self.pgid > 0 and not _group_alive(self.pgid):
            self.pgid = 0                             # never signal a dead group's number again
        return self.pgid > 0

    def stop(self, wait_ms: int = 5000) -> None:
        """Stop the command's WHOLE process group: SIGINT (what Ctrl-C does: ``ros2 launch`` shuts its
        nodes down), then SIGTERM, then SIGKILL, each given its grace period while ANY member of the
        group still runs -- a node that outlives its launch is stopped too (vNext I0.4 spike)."""
        if self.pgid <= 0:
            return
        self.stopped = True
        for sig, wait in ((signal.SIGINT, wait_ms), (signal.SIGTERM, wait_ms), (signal.SIGKILL, 1000)):
            if not _group_alive(self.pgid):
                break
            try:
                os.killpg(self.pgid, sig)
            except (ProcessLookupError, PermissionError):
                break
            deadline = time.monotonic() + wait / 1000.0
            while time.monotonic() < deadline and _group_alive(self.pgid):
                self.popen.poll()                     # reap the leader if it ended
                time.sleep(0.05)
        try:
            self.popen.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        if not _group_alive(self.pgid):
            self.pgid = 0


class Launcher(QObject):
    """At most one process per command kind (build, environment, isaac, bringup, run). Every panel that
    shows processes listens: ``add_listener(obj)`` with ``started(command)``, ``line(kind, text)`` and
    ``finished(kind, code)``, all called in the GUI thread."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.listeners: List[object] = []
        self.processes: Dict[str, LaunchedProcess] = {}
        self.timer = QTimer(self)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.pump)
        self.timer.start()

    def add_listener(self, listener) -> None:
        if listener not in self.listeners:
            self.listeners.append(listener)

    def pump(self) -> None:
        """Hand the new output and the endings to the listeners (GUI thread; the timer calls it)."""
        for kind, proc in list(self.processes.items()):
            for line in proc.drain():
                for listener in self.listeners:
                    listener.line(kind, line)
            if proc.poll() is not None and not proc.told:
                proc.told = True
                for listener in self.listeners:
                    listener.finished(kind, proc.exit_code)

    def start(self, command: Command) -> LaunchedProcess:
        self.pump()                                   # the previous one's last lines and ending first
        current = self.processes.get(command.kind)
        if current is not None and current.running():
            raise RuntimeError(f'{command.title} is already running: stop it first')
        proc = LaunchedProcess(command)               # raises (e.g. no bash) before anyone is told
        for listener in self.listeners:
            listener.started(command)
        self.processes[command.kind] = proc
        return proc

    def running(self, kind: str) -> bool:
        proc = self.processes.get(kind)
        return proc is not None and proc.running()

    def stop(self, kind: str) -> None:
        proc = self.processes.get(kind)
        if proc is not None:
            proc.stop()
            self.pump()

    def stop_all(self) -> None:
        """What TSA started, stopped TOGETHER (closing TSA must not freeze for a minute): SIGINT to every
        live group, a shared grace, then SIGTERM, then SIGKILL -- at most about 9 s in all."""
        procs = [p for p in self.processes.values() if p.pgid > 0]
        for sig, wait in ((signal.SIGINT, 5.0), (signal.SIGTERM, 3.0), (signal.SIGKILL, 1.0)):
            alive = [p for p in procs if p.pgid > 0 and _group_alive(p.pgid)]
            if not alive:
                break
            for p in alive:
                p.stopped = True
                try:
                    os.killpg(p.pgid, sig)
                except (ProcessLookupError, PermissionError):
                    pass
            deadline = time.monotonic() + wait
            while time.monotonic() < deadline and any(_group_alive(p.pgid) for p in alive if p.pgid > 0):
                for p in alive:
                    p.popen.poll()
                time.sleep(0.05)
        for p in procs:
            p.popen.poll()
            if not _group_alive(p.pgid):
                p.pgid = 0
        self.pump()


def _group_alive(pgid: int) -> bool:
    """Does a process of the group still RUN? A zombie (killed, not yet reaped by its new parent)
    still belongs to the group but runs nothing, so it does not count."""
    try:
        os.killpg(pgid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    try:
        for entry in os.listdir('/proc'):
            if not entry.isdigit():
                continue
            try:
                with open(f'/proc/{entry}/stat') as f:
                    fields = f.read().rsplit(')', 1)[1].split()
            except OSError:
                continue                                   # it ended while we looked
            if int(fields[2]) == pgid and fields[0] != 'Z':
                return True
        return False
    except OSError:
        return True                                        # no /proc: trust the signal probe


__all__ = ['Launcher', 'LaunchedProcess']
