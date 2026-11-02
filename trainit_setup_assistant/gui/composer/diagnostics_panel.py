"""The Composer's diagnostics (vNext I0.9, I0_DESIGN.md §21): the ONE validator's verdict, located.

Each line is the canonical text, unchanged (WHAT / WHERE / WHY / HOW is already in it), with a
deterministic "did you mean …?" where a name is misspelt. A double click (or Go to) opens the element it
names: a task step by its canonical label, a waypoint, a path, a profile or a task. The canonical
refusal stays authoritative: this panel only helps the user find it.
"""

from __future__ import annotations

from typing import Callable, List

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtGui import QBrush, QColor
from python_qt_binding.QtWidgets import (QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton,
                                         QVBoxLayout, QWidget)

from ...composer.diagnostics import Diagnostic, parse_all, suggestion

_COLOR = {'error': '#c62828', 'warning': '#8a6d3b'}


class DiagnosticsPanel(QWidget):
    def __init__(self, ctrl, composer: Callable[[], object], go_to: Callable[[Diagnostic], None], parent=None):
        super().__init__(parent)
        self.ctrl, self._composer, self._go_to = ctrl, composer, go_to
        self.diagnostics: List[Diagnostic] = []
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.summary = QLabel('')
        row.addWidget(self.summary, 1)
        self.go_button = QPushButton('Go to')
        self.go_button.clicked.connect(self.go_to_selected)
        self.validate_button = QPushButton('Validate')
        self.validate_button.clicked.connect(self.refresh)
        row.addWidget(self.go_button)
        row.addWidget(self.validate_button)
        v.addLayout(row)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.setMaximumHeight(140)
        self.list.itemDoubleClicked.connect(lambda item: self.go_to_selected())
        v.addWidget(self.list)

    def refresh(self) -> None:
        self.list.clear()
        if self.ctrl.project is None or not self.ctrl.project.application.is_task_program():
            self.diagnostics = []
            self.summary.setText('')
            return
        errors, warnings = self.ctrl.validate_program()
        self.diagnostics = parse_all(errors, warnings)
        doc = self._composer().document()
        for i, d in enumerate(self.diagnostics):
            hint = suggestion(d, doc, self.ctrl.project)
            text = ('✖ ' if d.severity == 'error' else '⚠ ') + str(d) + (f'\n    → {hint}' if hint else '')
            item = QListWidgetItem(text)
            item.setForeground(QBrush(QColor(_COLOR[d.severity])))
            item.setData(Qt.UserRole, i)
            item.setToolTip(d.location.describe())
            self.list.addItem(item)
        n_err = sum(d.severity == 'error' for d in self.diagnostics)
        n_warn = len(self.diagnostics) - n_err
        if n_err:
            self.summary.setText(f'<b style="color:#c62828">{n_err} error(s)</b>, {n_warn} warning(s) — Generate '
                                 'would refuse the program')
        elif n_warn:
            self.summary.setText(f'<b>Valid</b> · {n_warn} warning(s)')
        else:
            self.summary.setText('<b style="color:#2e7d32">Valid</b>: ready to generate')

    def go_to_selected(self) -> None:
        item = self.list.currentItem()
        if item is not None:
            self._go_to(self.diagnostics[item.data(Qt.UserRole)])
