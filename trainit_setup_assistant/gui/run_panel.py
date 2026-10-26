"""A panel that starts, shows and stops launcher commands (vNext I0.2 / I0.11, ADR-0027).

Before a command starts, the panel shows its exact line (``$ ...``), which the user can copy and run
themselves. The output follows in the log, each line tagged with its process (``[isaac] ...``), and the
runner's verdict (``BT finished: SUCCESS``) is highlighted. One status line per command says whether it
was started, runs or ended (owner Test A.2: with one shared state line, Isaac "running" hid that the
planning environment was never started). The panel launches and observes only."""

from __future__ import annotations

from typing import Callable, List, Optional, Tuple

from python_qt_binding.QtWidgets import (QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout,
                                         QWidget)

from ..composer.launch_plan import BT_FAILURE, BT_SUCCESS, Command

Action = Tuple[str, str, Callable[[], Command]]       # (kind, label, the command, built when clicked)

# what the status lines call each command
NAMES = {'build': 'Build', 'isaac': 'Isaac Sim', 'environment': 'Planning environment (RViz)',
         'bringup': 'Bundle bring-up', 'run': 'Application'}
NOT_STARTED = 'not started'


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
        v.addWidget(self.log, 1)
        self.verdict: Optional[str] = None
        self._show_status()

    # ---- actions ------------------------------------------------------------
    def toggle(self, kind: str) -> None:
        if self.launcher.running(kind):
            self.launcher.stop(kind)
            return
        label, factory = self.actions[kind]
        try:
            command = factory()
        except Exception as exc:  # noqa: BLE001 - a missing setting / binding is shown, not raised
            self.state.setText(f'⚠ {label}: {exc}')
            return
        try:
            self.launcher.start(command)
        except RuntimeError as exc:
            self.state.setText(f'⚠ {exc}')
        self._buttons()

    # ---- launcher listener --------------------------------------------------
    def started(self, command: Command) -> None:
        if command.kind not in self.actions:
            return
        if command.kind == 'run':
            self.verdict = None
        self.log.appendPlainText(f'$ {command.display()}')
        self.state.setText(f'{command.title}: running')
        self.status[command.kind] = 'running'
        self._show_status()
        self._buttons()

    def line(self, kind: str, text: str) -> None:
        if kind not in self.actions:
            return
        self.log.appendPlainText(f'[{kind}] {text}')
        if BT_SUCCESS in text or BT_FAILURE in text:
            self.verdict = BT_SUCCESS if BT_SUCCESS in text else BT_FAILURE
            ok = self.verdict == BT_SUCCESS
            self.state.setText(f'<b style="color:{"#2e7d32" if ok else "#c62828"}">{self.verdict}</b>')

    def finished(self, kind: str, code: int) -> None:
        if kind not in self.actions:
            return
        label = self.actions[kind][0]
        proc = self.launcher.processes.get(kind)
        stopped = proc is not None and proc.stopped
        self.log.appendPlainText(f'[{label}: stopped]' if stopped else f'[{label}: exited {code}]')
        if self.verdict is None or kind != 'run':
            self.state.setText(f'{label}: stopped' if stopped else
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
