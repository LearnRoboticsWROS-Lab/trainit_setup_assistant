"""The Composer's live panel (vNext I0.4, I0_DESIGN.md §12): what is live, capture, optional preview.

The CORE teaching flow:
1. In RViz, move the robot with the MotionPlanning gizmo, then Plan and Execute.
2. The deployment reaches the state.
3. In TSA, **Capture TCP** or **Capture joints** -- or **Capture TCP + joints**: the pose AND the joint
   configuration it was reached with, for a PTP that must reach the pose the same way every cycle.

A capture is the robot's CURRENT state (TF base -> tip, or the joint states), never the RViz target.
**Preview in RViz** is optional: it puts a waypoint in RViz as the goal; RViz plans and executes, TSA
never does. **Check the waypoints (live)** asks MoveIt's IK / FK / state validity about every waypoint
before Generate (``composer.live_check``): it computes, nothing moves.
"""

from __future__ import annotations

from typing import Callable, Optional

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtGui import QBrush, QColor
from python_qt_binding.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox,
                                         QPushButton,
                                         QVBoxLayout, QWidget)

from ...composer import pose
from ...composer.integration import STATEMENT
from ...composer.ops import ComposerError

NEW = 'a new waypoint'


def _changes_meaning(old_kind: str, new_kind: str) -> bool:
    """Does replacing a waypoint of ``old_kind`` with ``new_kind`` change what its users get? Named -> Joints
    does not: both are joint targets (the templates' '<name>_j' starts as the named state home precisely to
    be re-taught by Capture TCP + joints, which must not ask)."""
    return old_kind != new_kind and {old_kind, new_kind} != {'Named', 'Joints'}


class LivePanel(QWidget):
    def __init__(self, ctrl, composer: Callable[[], object], live: Callable[[], object],
                 selected: Callable[[], Optional[str]], changed: Callable[[Optional[str]], None],
                 preview_factory: Optional[Callable[[object], object]] = None,
                 age: Callable[[], Optional[float]] = lambda: None,
                 kinematics_factory: Optional[Callable[[object, object], object]] = None,
                 go_to_waypoint: Callable[[str], None] = lambda name: None, parent=None):
        super().__init__(parent)
        self.ctrl, self._composer, self._live = ctrl, composer, live
        self._age_of_running = age          # an EXISTING session's joint-state age: showing never starts one
        self._selected, self._changed = selected, changed
        self._preview_factory = preview_factory
        self._preview = None
        self._kinematics_factory = kinematics_factory
        self._go_to_waypoint = go_to_waypoint
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        self.header = QLabel('')
        self.header.setToolTip(STATEMENT)
        v.addWidget(self.header)
        row = QHBoxLayout()
        row.addWidget(QLabel('Capture into:'))
        self.target = QComboBox()
        row.addWidget(self.target)
        self.capture_tcp_button = QPushButton('Capture TCP')
        self.capture_tcp_button.setToolTip('The robot\'s CURRENT TCP pose (TF base -> tip), after Plan + Execute '
                                           'in RViz: what the robot reached, not the RViz target.')
        self.capture_tcp_button.clicked.connect(self.capture_tcp)
        self.capture_joints_button = QPushButton('Capture joints')
        self.capture_joints_button.setToolTip('The robot\'s CURRENT joint positions (planning group order).')
        self.capture_joints_button.clicked.connect(self.capture_joints)
        self.capture_both_button = QPushButton('Capture TCP + joints')
        self.capture_both_button.setToolTip(
            "The TCP pose into the target and the joints of the SAME robot state into '<target>_j': a PTP to "
            "'<target>_j' reaches the pose with this configuration every cycle (a PTP to a TCP pose lets IK "
            'choose one when it runs). Use it for the start of a path or of LIN / CIRC moves.')
        self.capture_both_button.clicked.connect(self.capture_both)
        self.preview_button = QPushButton('Preview in RViz')
        self.preview_button.setToolTip('Optional: show the selected waypoint as RViz\'s goal (needs '
                                       'MotionPlanning > "External Comm."); then Plan / Execute in RViz.')
        self.preview_button.clicked.connect(self.preview)
        for b in (self.capture_tcp_button, self.capture_joints_button, self.capture_both_button):
            row.addWidget(b)
        row.addStretch(1)
        self.check_button = QPushButton('Check the waypoints (live)')
        self.check_button.setToolTip('Before Generate: MoveIt\'s IK / FK / collision check of every waypoint '
                                     '(reachable? collision-free? does each joint waypoint reach its pose?). It '
                                     'computes; nothing moves.')
        self.check_button.clicked.connect(self.check_waypoints)
        row.addWidget(self.check_button)
        row.addWidget(self.preview_button)
        v.addLayout(row)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        v.addWidget(self.message)
        # owner Test C: the waypoints checked against the live robot (double click: the waypoint)
        self.check_list = QListWidget()
        self.check_list.setWordWrap(True)
        self.check_list.setMaximumHeight(140)
        self.check_list.itemDoubleClicked.connect(
            lambda item: self._go_to_waypoint(item.data(Qt.UserRole)))
        self.check_list.setVisible(False)
        v.addWidget(self.check_list)
        self.check_lines = []

    # ---- state -------------------------------------------------------------------------------------
    def _age(self) -> Optional[float]:
        try:
            return self._age_of_running()
        except Exception:  # noqa: BLE001 - no live session: OFFLINE
            return None

    def refresh(self) -> None:
        try:
            self.header.setText(f'<b>{self.ctrl.scene_status(self._age()).header()}</b>')
        except Exception:  # noqa: BLE001 - a status never breaks the panel
            self.header.setText('')
        sel = self._selected()
        replacing = self.target.currentData() is not None
        self.target.clear()
        self.target.addItem(NEW, None)
        if sel:                                   # the name travels WITH the item: what it says is what it does
            self.target.addItem(f"'{sel}' (replace it)", sel)
            if replacing:
                self.target.setCurrentIndex(1)

    def _name(self) -> str:
        target = self.target.currentData()
        if target:
            return target
        names = set(self._composer().waypoint_names())
        i = 1
        while f'taught{i}' in names:
            i += 1
        return f'taught{i}'

    # ---- capture (the core flow) ---------------------------------------------------------------------
    def capture_tcp(self) -> Optional[str]:
        robot = self.ctrl.project.robot
        try:
            position, orientation = self._live().current_pose(robot.base_frame, robot.tip_link)
        except Exception as exc:  # noqa: BLE001
            return self._error(f'Capture TCP: {exc} — start the planning environment (Step 4) first')
        name = self._name()
        spec = pose.captured_tcp_waypoint(name, position, orientation)
        old = next((w for w in self._composer().document().get('waypoints') or [] if w['name'] == name), None)
        what = f'TCP {robot.base_frame} -> {robot.tip_link}'
        if old and old.get('configuration'):
            # ADR-0028: never dropped silently -- the live check says whether the new pose is still reached near it
            spec['configuration'] = old['configuration']
            what += f"; its configuration '{old['configuration']}' kept (Capture TCP + joints re-teaches both)"
        return self._commit(spec, what)

    def capture_joints(self) -> Optional[str]:
        robot = self.ctrl.project.robot
        joints = list(robot.planning_group.joints)
        try:
            values = self._live().current_joint_values(joints)
        except Exception as exc:  # noqa: BLE001
            return self._error(f'Capture joints: {exc} — start the planning environment (Step 4) first')
        missing = [j for j in joints if j not in values]
        if missing:
            return self._error(f"Capture joints: the joint states lack {', '.join(missing)}: the running robot "
                               f"is not planning group '{robot.planning_group.name}'")
        spec = pose.captured_joint_waypoint(self._name(), [values[j] for j in joints])
        return self._commit(spec, 'joints')

    def capture_both(self) -> Optional[str]:
        """The TCP pose into the target and the joints of the same robot state into '<target>_j', as ONE edit
        (one Undo). Owner Test C: the same PTP home -> w1 reached w1 with 4.4 rad of joint motion in one cycle
        and 13.4 rad in the next (IK chose another configuration), and the path after it failed."""
        robot = self.ctrl.project.robot
        joints = list(robot.planning_group.joints)
        try:
            values = self._live().current_joint_values(joints)
            position, orientation = self._live().current_pose(robot.base_frame, robot.tip_link)
        except Exception as exc:  # noqa: BLE001
            return self._error(f'Capture TCP + joints: {exc} — start the planning environment (Step 4) first')
        missing = [j for j in joints if j not in values]
        if missing:
            return self._error(f"Capture TCP + joints: the joint states lack {', '.join(missing)}: the running "
                               f"robot is not planning group '{robot.planning_group.name}'")
        name = self._name()
        tcp = pose.captured_tcp_waypoint(name, position, orientation)
        js = pose.captured_joint_waypoint(f'{name}_j', [values[j] for j in joints])
        tcp['configuration'] = js['name']           # ADR-0028: a PTP / free move reaches the pose in it
        old = {w['name']: w for w in self._composer().document().get('waypoints') or []}
        if not self._kind_change_accepted([tcp, js], old):
            return None
        try:
            self._composer().set_waypoints([tcp, js])
        except ComposerError as exc:
            return self._error(str(exc))
        notes = [f"'{w['name']}' {'replaced' if w['name'] in old else 'created'}"
                 + (f" (it WAS a {pose.kind(old[w['name']])} waypoint)"
                    if w['name'] in old and pose.kind(old[w['name']]) != pose.kind(w) else '')
                 for w in (tcp, js)]
        self.message.setStyleSheet('color:#2e7d32;')
        self.message.setText(f"Captured the robot's CURRENT TCP and joints: {' and '.join(notes)} — "
                             f"{pose.summary(tcp)}; a PTP to '{name}' or to '{js['name']}' reaches it in this "
                             'configuration (Undo reverts both)')
        self._changed(name)
        return name

    def _confirm(self, question: str) -> bool:  # pragma: no cover - a modal dialog needs a user
        return QMessageBox.question(self, 'Capture', question) == QMessageBox.Yes

    def _users_text(self, name: str) -> str:
        """Who depends on a waypoint, in words: "'w2' (relative to it), path 'sweep' (state 0)"."""
        doc = self._composer().document()
        out = []
        for where in self._composer().waypoint_users(name):
            if where.startswith("waypoint '"):
                out.append(f"{where[len('waypoint '):]} (relative to it)")
            elif where.startswith("path '"):
                path = next((p for p in doc.get('paths') or [] if f"path '{p.get('name')}'" == where), {})
                states = [i for i, st in enumerate(path.get('states') or []) if st == name]
                out.append(where + (f" (state {', '.join(map(str, states))})" if states else ' (a CIRC via)'))
            else:
                out.append(where)
        return ', '.join(out)

    def _kind_change_accepted(self, specs, old: dict) -> bool:
        """Owner Test C, friction row 16: a capture that REPLACES a waypoint with another KIND while other
        elements depend on it asks first (Capture joints into a TCP waypoint that a relative target and a
        path start use left R-M3-05 / R-M3-04 behind). No: nothing changes and no undo entry is made."""
        changes = [(spec['name'], pose.kind(old[spec['name']]), pose.kind(spec)) for spec in specs
                   if spec['name'] in old and _changes_meaning(pose.kind(old[spec['name']]), pose.kind(spec))
                   and self._composer().waypoint_users(spec['name'])]
        if not changes:
            return True
        question = '\n'.join(f"'{n}' is a {a} waypoint and becomes {b}: it is used by {self._users_text(n)}, "
                              f'which may need it as {a}.' for n, a, b in changes)
        if self._confirm(question + '\n\nReplace it anyway?'):
            return True
        self.message.setStyleSheet('color:#8a6d3b;')
        self.message.setText('not captured: ' + ', '.join(f"'{n}' kept as {a}" for n, a, _b in changes))
        return False

    def _commit(self, spec: dict, what: str) -> Optional[str]:
        old = next((w for w in self._composer().document().get('waypoints') or []
                    if w['name'] == spec['name']), None)
        if old is not None and not self._kind_change_accepted([spec], {old['name']: old}):
            return None
        try:
            self._composer().set_waypoint(spec)
        except ComposerError as exc:
            return self._error(str(exc))
        verb = 'created'
        note = ''
        if old is not None:
            verb = 'replaced'
            if pose.kind(old) != pose.kind(spec):
                note = f' — it WAS a {pose.kind(old)} waypoint, it is now {pose.kind(spec)}'
            elif old.get('frame'):
                note = f" — it was in frame '{old['frame']}': a capture is in the robot base frame"
        self.message.setStyleSheet('color:#8a6d3b;' if note else 'color:#2e7d32;')
        self.message.setText(f"Captured the robot's CURRENT {what}: '{spec['name']}' {verb}{note} — "
                             f'{pose.summary(spec)} (Undo reverts it)')
        self._changed(spec['name'])
        return spec['name']

    # ---- the live waypoint check (owner Test C) ------------------------------------------------------
    def check_waypoints(self) -> Optional[list]:
        """Every waypoint against the live robot (``composer.live_check``); the lines are shown and returned."""
        from ...composer.live_check import check_motions, check_waypoints, summary
        if self.ctrl.project is None or not self.ctrl.project.application.is_task_program():
            self._error('Check the waypoints: only a task program has waypoints to check')
            return None
        self.message.setStyleSheet('')
        self.message.setText('Checking the waypoints and the motions between them against MoveIt (IK / FK / '
                             'collisions; nothing moves)...')
        from python_qt_binding.QtWidgets import QApplication
        QApplication.processEvents()                       # the message shows while MoveIt answers
        self.check_button.setEnabled(False)                # no second check from inside the first
        try:
            return self._check(check_waypoints, check_motions, summary)
        finally:
            self.check_button.setEnabled(True)

    def _check(self, check_waypoints, check_motions, summary) -> Optional[list]:
        from ...composer.live_check import CheckLine
        try:
            project = self.ctrl.resolved_project()
            factory = self._kinematics_factory
            if factory is None:
                from ...livesession.kinematics import LiveKinematics as factory
            kin = factory(self._live(), project)
            lines = check_waypoints(project, kin.ik, kin.fk, kin.valid, getattr(kin, 'scene_objects', None))
        except RuntimeError as exc:                        # no move_group, no answer, over the budget
            self.check_list.clear()                        # never a previous check's lines under the error
            self._error(f'Check the waypoints: {exc} — start the planning environment (Step 4) first')
            return None
        except Exception as exc:  # noqa: BLE001 - shown, never raised into Qt
            self.check_list.clear()
            self._error(f'Check the waypoints failed: {type(exc).__name__}: {exc}')
            return None
        # owner Test F.1 (row 36): and the motions between them, along their way -- with a time budget of their
        # own; a stop keeps the waypoints' lines
        try:
            lines += check_motions(project, *self._pumped(factory(self._live(), project)))
        except RuntimeError as exc:
            lines.append(CheckLine(None, '', f'the motions between the waypoints: not checked to the end -- {exc}'))
        except Exception as exc:  # noqa: BLE001 - shown, never raised into Qt
            lines.append(CheckLine(None, '', f'the motions between the waypoints: {type(exc).__name__}: {exc}'))
        self.check_lines = lines
        self.check_list.clear()
        colours = {True: '#2e7d32', False: '#c62828', None: '#8a6d3b'}
        for line in lines:
            item = QListWidgetItem(str(line))
            item.setForeground(QBrush(QColor(colours[line.ok])))
            item.setData(Qt.UserRole, line.waypoint)
            self.check_list.addItem(item)
        self.check_list.setVisible(True)
        bad = any(line.ok is False for line in lines)
        self.message.setStyleSheet('color:#c62828;' if bad else 'color:#2e7d32;')
        self.message.setText(summary(lines) + (' -- double click a line to open its waypoint' if bad else ''))
        return lines

    @staticmethod
    def _pumped(kin, every: int = 10):
        """``kin``'s ik / fk / valid, letting Qt repaint every ``every`` answers: a long check never freezes
        the window (the check button is disabled meanwhile)."""
        from python_qt_binding.QtWidgets import QApplication
        count = [0]

        def pump(fn):
            def call(*args):
                count[0] += 1
                if count[0] % every == 0:
                    QApplication.processEvents()
                return fn(*args)
            return call
        return pump(kin.ik), pump(kin.fk), pump(kin.valid)

    # ---- the optional preview -------------------------------------------------------------------------
    def preview(self) -> bool:
        sel = self._selected()
        wp = next((w for w in self._composer().document().get('waypoints') or [] if w['name'] == sel), None)
        if wp is None:
            self._error('Preview: select a waypoint first')
            return False
        try:
            if self._preview is None:
                factory = self._preview_factory
                if factory is None:
                    from ...livesession.preview import RvizPreview as factory
                self._preview = factory(self._live())
            if not self._preview.listening():
                self._error('Preview: RViz is not listening — in RViz, MotionPlanning panel, tick "External '
                            'Comm." (the teaching flow works without it: move the gizmo yourself)')
                return False
            text = self._preview.preview_waypoint(self.ctrl.project, wp)
        except Exception as exc:  # noqa: BLE001
            self._error(f'Preview: {exc}')
            return False
        self.message.setStyleSheet('')
        self.message.setText(text)
        return True

    def _error(self, text: str) -> None:
        self.message.setStyleSheet('color:#c62828;')
        self.message.setText(text)
        return None
