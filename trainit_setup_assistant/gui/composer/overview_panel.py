"""The Project Overview panel (vNext I0.1, I0_DESIGN.md §18): renders ``composer.overview`` rows."""

from __future__ import annotations

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtGui import QBrush, QColor
from python_qt_binding.QtWidgets import (QLabel, QPushButton, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
                                         QVBoxLayout, QWidget)

from ...composer.integration import STATEMENT
from ...composer.overview import ERROR, INFO, OK, WARN, project_overview

_MARK = {OK: ('✔', '#2e7d32'), WARN: ('⚠', '#8a6d3b'), ERROR: ('✖', '#c62828'), INFO: ('•', '#555555')}


class OverviewPanel(QWidget):
    """One row per block (ROBOT, MOVEIT CONFIG, DEPLOYMENT, WORLD, ROBOT BINDING, SCENE, ...); a row's
    detail is its child line (progressive disclosure). Refreshed on demand: the overview is
    computed, never stored."""

    def __init__(self, ctrl, joint_states_age=lambda: None, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self._age = joint_states_age
        v = QVBoxLayout(self)
        head = QHBoxLayout()
        self.title = QLabel('<b>Project overview</b>')
        head.addWidget(self.title, 1)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.clicked.connect(self.refresh)
        head.addWidget(self.refresh_button)
        v.addLayout(head)
        self.statement = QLabel(STATEMENT)
        self.statement.setWordWrap(True)
        self.statement.setStyleSheet('color:#555555; font-style:italic;')
        v.addWidget(self.statement)
        self.tree = QTreeWidget()
        self.tree.setColumnCount(3)
        self.tree.setHeaderLabels(['', 'Block', 'State'])
        self.tree.setRootIsDecorated(True)
        self.tree.setWordWrap(True)
        v.addWidget(self.tree, 1)
        self.rows = []

    def refresh(self) -> None:
        self.tree.clear()
        if self.ctrl.project is None:
            self.rows = []
            return
        errors, warnings = self.ctrl.validate_program()
        status = self.ctrl.scene_status(self._age())
        self.rows = project_overview(self.ctrl.resolved_project(), status, errors, warnings,
                                     getattr(self.ctrl, 'last_generation_dir', None))
        for row in self.rows:
            mark, color = _MARK.get(row.mark, _MARK[INFO])
            item = QTreeWidgetItem(['', row.block, row.text])
            item.setText(0, mark)
            item.setForeground(0, QBrush(QColor(color)))
            item.setToolTip(2, row.detail or row.text)
            if row.detail:
                for line in row.detail.splitlines():
                    child = QTreeWidgetItem(['', '', line])
                    child.setFlags(child.flags() & ~Qt.ItemIsSelectable)
                    item.addChild(child)
            self.tree.addTopLevelItem(item)
        for col in (0, 1):
            self.tree.resizeColumnToContents(col)
