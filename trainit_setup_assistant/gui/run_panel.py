"""A panel that starts, shows and stops launcher commands (vNext I0.2 / I0.11, ADR-0027).

Before a command starts, the panel shows its exact line (``$ ...``), which the user can copy and run
themselves. The output follows in the log, each line tagged with its process (``[isaac] ...``), and the
runner's verdict (``BT finished: SUCCESS``) is highlighted, with the run's OUTCOME and what the operator does next
(M4.5d owner acceptance F8: a PICKABLE_EXHAUSTED stop read as a hang). One status line per command says whether it
was started, runs or ended (owner Test A.2: with one shared state line, Isaac "running" hid that the
planning environment was never started). The panel launches and observes only.

TSA 8.10.0 (P0.b, decision D5): the state line follows the machine states the runtime logs -- a ``MACHINE: `` line of
the application or of the robot task server (WAITING FOR MACHINE, MACHINE <service> BUSY / DONE / FAULT / TIMEOUT /
UNAVAILABLE), and the server's ``RTI: `` lines (its state, a job's result, a job's machine activity) -- in plain
words, never a topic or a counter. Those lines are shown as PLAIN text (they were escaped for a label that rendered
them plain: ``->`` read ``-&gt;``, a quote ``&#x27;``)."""

from __future__ import annotations

import html
import re
from typing import Callable, Dict, List, Optional, Tuple

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
                                         QWidget)

from ..composer.launch_plan import BT_FAILURE, BT_SUCCESS, Command

Action = Tuple[str, str, Callable[[], Command]]       # (kind, label, the command, built when clicked)

# what the status lines call each command
NAMES = {'build': 'Build', 'isaac': 'Isaac Sim', 'environment': 'Planning environment (RViz)',
         'bringup': 'Bundle bring-up', 'run': 'Application', 'server': 'Robot task server'}
NOT_STARTED = 'not started'
# the runtime's log markers the state line reads (TMR >= 2.11.0 for MACHINE, 2.10.0 for RTI; pinned on both sides)
MACHINE_MARK = 'MACHINE: '
RTI_MARK = 'RTI: '
# TSA 8.10.0 (P0.b, case O1): the state line shows the server's words, never a protocol counter -- a command's
# sequence number ("START seq 7: ACCEPTED"), a requester's baseline seq, a run id; the log keeps the full line
_COUNTERS = re.compile(r' seq \d+(?=:)| \(baseline seq \d+\)| run [0-9a-f]{8}-\d+')


def plain_words(text: str) -> str:
    """A runtime line for the state line: as written, without the protocol's counters."""
    return _COUNTERS.sub('', text)


class RunPanel(QWidget):
    def __init__(self, launcher, actions: List[Action], parent=None):
        super().__init__(parent)
        self.launcher = launcher
        self.actions = {kind: (label, factory) for kind, label, factory in actions}
        launcher.add_listener(self)
        v = QVBoxLayout(self)
        row = QHBoxLayout()
        self.buttons = {}
        for kind, label, _factory in actions:
            b = QPushButton(label)
            b.clicked.connect(lambda _=False, k=kind: self.toggle(k))
            self.buttons[kind] = b
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        self.status = {kind: NOT_STARTED for kind, _label, _factory in actions}
        self.status_label = QLabel('')
        self.status_label.setWordWrap(True)
        v.addWidget(self.status_label)
        self.state = QLabel('')
        self.state.setWordWrap(True)
        v.addWidget(self.state)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        # kind -> (marker, lines): written ONCE per start, right after the command's output shows `marker` (TSA 8.9.0:
        # the fake PLC's command when the robot task server says it serves)
        self.after_ready = {}
        self._waiting_ready = set()
        # kind -> lines written right after the command's own line when it starts (TSA 8.10.0: the cell's machine
        # simulation command before a run that requests machine services)
        self.notes: Dict[str, List[str]] = {}
        v.addWidget(self.log, 1)
        self.verdict: Optional[str] = None
        self.outcome: Optional[str] = None          # "<CODE> -- <operator text>", from the run's OUTCOME line
        self.machine: Optional[str] = None          # TSA 8.10.0: the run's last machine state (plain words)
        self._show_status()

    def _plain(self, text: str) -> None:
        """The state line as PLAIN text: what a process said is shown as it said it, never read as markup."""
        self.state.setTextFormat(Qt.PlainText)
        self.state.setText(text)

    # ---- actions ------------------------------------------------------------
    def toggle(self, kind: str) -> None:
        if self.launcher.running(kind):
            self.launcher.stop(kind)
            return
        label, factory = self.actions[kind]
        try:
            command = factory()
        except Exception as exc:  # noqa: BLE001 - a missing setting / binding is shown, not raised
            self._plain(f'⚠ {label}: {exc}')
            return
        try:
            self.launcher.start(command)
        except RuntimeError as exc:
            self._plain(f'⚠ {exc}')
        self._buttons()

    # ---- launcher listener --------------------------------------------------
    def started(self, command: Command) -> None:
        if command.kind not in self.actions:
            return
        if command.kind == 'run':
            self.verdict = None
            self.outcome = None
            self.machine = None
        self.log.appendPlainText(f'$ {command.display()}')
        for note in self.notes.get(command.kind, []):
            self.log.appendPlainText(note)
        if command.kind in self.after_ready:
            self._waiting_ready.add(command.kind)
        self._plain(f'{command.title}: running')
        self.status[command.kind] = 'running'
        self._show_status()
        self._buttons()

    def line(self, kind: str, text: str) -> None:
        if kind not in self.actions:
            return
        self.log.appendPlainText(f'[{kind}] {text}')
        if kind in self._waiting_ready and self.after_ready[kind][0] in text:
            self._waiting_ready.discard(kind)
            for extra in self.after_ready[kind][1]:
                self.log.appendPlainText(extra)
        if kind == 'server' and RTI_MARK in text:
            # TSA 8.9.0 (ADR-0032): the robot task server's own lines -- its state, a job's result, and (TSA 8.10.0) a
            # job's machine activity ("RTI: job N: ...") -- observed, never commanded (the PLC / the fake PLC is the
            # requester); shown as plain text
            self._plain(f'Robot task server — {plain_words(text.split(RTI_MARK, 1)[1].strip())}')
        if kind in ('run', 'server') and MACHINE_MARK in text:
            # TSA 8.10.0 (P0.b, D5): the machine state, in the runtime's plain words, as it changes
            self.machine = text.split(MACHINE_MARK, 1)[1].strip()
            self._plain(f'{NAMES[kind]} — {self.machine}')
        if kind == 'run' and 'OUTCOME: ' in text:
            self.outcome = text.split('OUTCOME: ', 1)[1].strip()
            self._show_verdict()
        if BT_SUCCESS in text or BT_FAILURE in text:
            self.verdict = BT_SUCCESS if BT_SUCCESS in text else BT_FAILURE
            self._show_verdict()

    def _show_verdict(self) -> None:
        parts = []
        if self.verdict is not None:
            ok = self.verdict == BT_SUCCESS
            parts.append(f'<b style="color:{"#2e7d32" if ok else "#c62828"}">{self.verdict}</b>')
        if self.outcome:
            code, _, why = self.outcome.partition(' -- ')
            action = why.startswith('OPERATOR ACTION:')
            parts.append(f'<b>OUTCOME: {html.escape(code)}</b>'
                         + (f'<br><span style="color:{"#c62828" if action else "#424242"}">{html.escape(why)}</span>'
                            if why else ''))
        if self.verdict == BT_FAILURE and self.machine:
            # TSA 8.10.0 (P0.b): a run a machine stopped says which machine and why (the last machine state)
            parts.append(f'<span style="color:#c62828">{html.escape(self.machine)}</span>')
        self.state.setTextFormat(Qt.RichText)        # every dynamic part above is escaped
        self.state.setText('<br>'.join(parts))

    def finished(self, kind: str, code: int) -> None:
        if kind not in self.actions:
            return
        label = self.actions[kind][0]
        proc = self.launcher.processes.get(kind)
        stopped = proc is not None and proc.stopped
        self.log.appendPlainText(f'[{label}: stopped]' if stopped else f'[{label}: exited {code}]')
        if self.verdict is None or kind != 'run':
            self._plain(f'{label}: stopped' if stopped else
                        f'{label}: exited {code}' + ('' if code == 0 else ' — see the log'))
        if kind == 'run' and self.verdict is not None:
            self.status[kind] = self.verdict
        else:
            self.status[kind] = 'stopped' if stopped else f'exited {code}'
        self._show_status()
        self._buttons()

    def _show_status(self) -> None:
        colours = {'running': '#2e7d32', NOT_STARTED: '#757575', 'stopped': '#757575', 'exited 0': '#424242',
                   BT_SUCCESS: '#2e7d32'}
        self.status_label.setText('<br>'.join(
            f'{NAMES.get(kind, kind)}: <b style="color:{colours.get(state, "#c62828")}">{state}</b>'
            for kind, state in self.status.items()))

    def _buttons(self) -> None:
        for kind, b in self.buttons.items():
            label = self.actions[kind][0]
            b.setText(f'Stop: {label}' if self.launcher.running(kind) else label)
