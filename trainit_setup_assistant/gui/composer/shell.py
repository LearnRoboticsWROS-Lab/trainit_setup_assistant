"""The Composer shell (vNext I0, I0_DESIGN.md §13): a navigation list beside one panel per section.

The shell holds the ONE ``composer.ops.Composer`` of the controller's project. Every new project a
Composer operation commits replaces the controller's ``project`` (``on_change``), so the wizard's other
pages and the Composer never hold two copies.

Sections are added slice by slice: Overview (I0.1), Waypoints (I0.3), and then paths, profiles, tasks...
The read-only program text of TSA 8.0 stays as the "Program" section until the editors replace it.
"""

from __future__ import annotations

from typing import Callable, Optional

from python_qt_binding.QtWidgets import (QHBoxLayout, QListWidget, QPushButton, QStackedWidget, QVBoxLayout,
                                         QWidget)

from ...composer.ops import Composer
from .diagnostics_panel import DiagnosticsPanel
from .eef_panel import EndEffectorPanel
from .live_panel import LivePanel
from .overview_panel import OverviewPanel
from .paths_panel import PathsPanel
from .profiles_panel import ProfilesPanel
from .tasks_panel import TasksPanel
from .waypoints_panel import WaypointsPanel
from .yaml_panel import YamlPanel


def _no_live():
    raise RuntimeError('no live session')


class ComposerWidget(QWidget):
    def __init__(self, ctrl, program_panel: QWidget, joint_states_age=lambda: None,
                 on_edit: Callable[[], None] = lambda: None, live: Callable[[], object] = _no_live,
                 preview_factory=None, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self._composer: Optional[Composer] = None
        self._on_edit = on_edit
        outer = QVBoxLayout(self)
        h = QHBoxLayout()
        outer.addLayout(h, 1)
        left = QVBoxLayout()
        self.nav = QListWidget()
        self.nav.setMaximumWidth(170)
        left.addWidget(self.nav, 1)
        self.undo_button = QPushButton('Undo')
        self.redo_button = QPushButton('Redo')
        self.undo_button.clicked.connect(self.undo)
        self.redo_button.clicked.connect(self.redo)
        for b in (self.undo_button, self.redo_button):
            b.setMaximumWidth(170)
            left.addWidget(b)
        self.stack = QStackedWidget()
        h.addLayout(left)
        h.addWidget(self.stack, 1)
        self.overview = OverviewPanel(ctrl, joint_states_age)
        self.waypoints = WaypointsPanel(self.composer, on_select=lambda name: self.live.refresh())
        self.sections = {}
        self.add_section('Overview', self.overview)
        self.add_section('Waypoints', self.waypoints)
        self.paths = PathsPanel(self.composer)
        self.add_section('Paths', self.paths)
        self.profiles = ProfilesPanel(self.composer)
        self.add_section('Profiles', self.profiles)
        # vNext I0 (§34, owner decision 2026-10-26): the end effector, in the same layer as the motion
        self.end_effector = EndEffectorPanel(self.composer, live=live)
        self.add_section('End effector', self.end_effector)
        self.tasks = TasksPanel(self.composer)
        self.add_section('Tasks', self.tasks)
        self.yaml = YamlPanel(ctrl, reloaded=self._file_reloaded)
        self.add_section('YAML', self.yaml)
        self.add_section('Program', program_panel)
        # vNext I0.4: the live panel under every section (status, Capture TCP / joints, optional Preview)
        self.live = LivePanel(ctrl, self.composer, live, lambda: self.waypoints.selected, self._captured,
                              preview_factory, age=joint_states_age, go_to_waypoint=self._open_waypoint)
        outer.addWidget(self.live)
        # vNext I0.9: the ONE validator's verdict, located; double click = go to the element
        self.diagnostics = DiagnosticsPanel(ctrl, self.composer, self.go_to)
        outer.addWidget(self.diagnostics)
        self.nav.currentRowChanged.connect(self._show)
        self.nav.setCurrentRow(0)
        self.undo_button.setEnabled(False)
        self.redo_button.setEnabled(False)

    # ---- the one project ----------------------------------------------------------------------------
    def composer(self) -> Composer:
        """The Composer of the controller's CURRENT project (a project opened elsewhere starts a new
        one, with a new undo history)."""
        if self._composer is None or self._composer.project is not self.ctrl.project:
            self._composer = Composer(self.ctrl.project, on_change=self._adopt)
        return self._composer

    def _adopt(self, project) -> None:
        self.ctrl.project = project
        self.edited()

    def go_to(self, diag) -> None:
        """Open the element a diagnostic names (I0_DESIGN.md §21)."""
        loc = diag.location
        if loc.kind in ('step', 'task', 'task_data'):
            self.select('Tasks')
            self.tasks.task = loc.name
            self.tasks.refresh()
            if loc.kind == 'step':
                self.tasks.select(loc.label)
        elif loc.kind == 'waypoint':
            self.select('Waypoints')
            self.waypoints.select(loc.name)
        elif loc.kind == 'path':
            self.select('Paths')
            self.paths.select(loc.name)
        elif loc.kind == 'profile':
            self.select('Profiles')
            self.profiles.select(loc.name)
        elif loc.kind == 'end_effector':
            self.select('End effector')
        else:
            self.select('Overview')

    def edited(self) -> None:
        """After every committed edit, undo or redo: the other views show the new project."""
        c = self._composer if self._composer is not None and self._composer.project is self.ctrl.project else None
        self.undo_button.setEnabled(bool(c and c.can_undo))
        self.redo_button.setEnabled(bool(c and c.can_redo))
        self.diagnostics.refresh()
        self._on_edit()

    def _file_reloaded(self) -> None:
        """A reload replaced the project: no panel keeps a selection of the old one."""
        self.waypoints.selected = None
        self.paths.selected = None
        self.profiles.selected = None
        self.tasks.task, self.tasks._edit_label = None, None
        self.edited()
        self.live.refresh()

    def _open_waypoint(self, name) -> None:
        if name:
            self.select('Waypoints')
            self.waypoints.select(name)

    def _captured(self, name) -> None:
        self.waypoints.selected = name
        if self.stack.currentWidget() is self.waypoints:
            self.waypoints.refresh()

    def undo(self) -> None:
        self.composer().undo()
        self.refresh()

    def redo(self) -> None:
        self.composer().redo()
        self.refresh()

    # ---- navigation ---------------------------------------------------------------------------------
    def add_section(self, name: str, widget: QWidget) -> None:
        self.sections[name] = widget
        self.nav.addItem(name)
        self.stack.addWidget(widget)

    def select(self, name: str) -> None:
        self.nav.setCurrentRow(list(self.sections).index(name))

    def _show(self, row: int) -> None:
        self.stack.setCurrentIndex(row)
        self.refresh()

    def refresh(self) -> None:
        if self.ctrl.project is None or not self.ctrl.project.application.is_task_program():
            return
        widget = self.stack.currentWidget()
        if hasattr(widget, 'refresh'):
            widget.refresh()
        self.live.refresh()
        self.diagnostics.refresh()
