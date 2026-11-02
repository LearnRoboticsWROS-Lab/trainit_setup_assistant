"""The Waypoints section (vNext I0.3, I0_DESIGN.md §14): visible, editable, teachable.

The table shows every waypoint in an engineer's units, and an editor per kind changes the selected one.
- TCP: frame, X / Y / Z in mm, roll / pitch / yaw in degrees.
- Joints: one value per planning-group joint, in degrees.
- Named: a MoveIt named state.
- Relative: an offset from another waypoint.

Each Apply is ONE Composer operation, so the canonical model validates it. An untouched value is
stored exactly as it was (``composer.pose``).
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional

from python_qt_binding.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                                         QHeaderView, QLabel, QLineEdit, QPushButton, QStackedWidget,
                                         QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...composer import pose
from ...composer.ops import ComposerError

KINDS = ('TCP', 'Joints', 'Named', 'Relative')


NO_CONFIGURATION = '(none: nearest where the robot is)'


def _set_configuration(spec: dict, chosen: str) -> None:
    """ADR-0028: the configuration the editor shows, written or removed."""
    if chosen in ('', NO_CONFIGURATION):
        spec.pop('configuration', None)
    else:
        spec['configuration'] = chosen


def _spin(lo: float, hi: float, decimals: int = 2, suffix: str = '') -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    return s


class WaypointsPanel(QWidget):
    def __init__(self, composer: Callable[[], object], changed: Callable[[], None] = lambda: None,
                 parent=None, on_select: Callable[[Optional[str]], None] = lambda name: None):
        super().__init__(parent)
        self._composer = composer
        self._changed = changed
        self._on_select_cb = on_select
        self.selected: Optional[str] = None
        self._shown = {}                      # what the editor showed: kept values stay exact
        v = QVBoxLayout(self)
        v.addWidget(QLabel('<b>Waypoints</b> — WHERE the robot goes (how it gets there is the move or '
                           'path that uses it)'))
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Name', 'Kind', 'Values', 'Used by'])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.new_buttons = {}
        for k in KINDS:
            b = QPushButton(f'New {k}')
            b.clicked.connect(lambda _=False, kk=k: self.new_waypoint(kk))
            self.new_buttons[k] = b
            row.addWidget(b)
        self.rename_edit = QLineEdit()
        self.rename_edit.setPlaceholderText('new name')
        self.rename_button = QPushButton('Rename')
        self.rename_button.clicked.connect(lambda: self.rename(self.selected, self.rename_edit.text().strip()))
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(lambda: self.delete(self.selected))
        row.addStretch(1)
        for w in (self.rename_edit, self.rename_button, self.delete_button):
            row.addWidget(w)
        v.addLayout(row)
        self.editor = QGroupBox('Edit')
        ev = QVBoxLayout(self.editor)
        self.stack = QStackedWidget()
        self.forms = {}
        self._build_tcp()
        self._build_joints()
        self._build_named()
        self._build_relative()
        ev.addWidget(self.stack)
        arow = QHBoxLayout()
        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply)
        arow.addStretch(1)
        arow.addWidget(self.apply_button)
        ev.addLayout(arow)
        v.addWidget(self.editor)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        v.addWidget(self.message)
        self.editor.setEnabled(False)

    # ---- forms -------------------------------------------------------------------------------------
    def _build_tcp(self):
        w = QWidget()
        f = QFormLayout(w)
        self.frame = QLineEdit()
        self.frame.setPlaceholderText('(the robot base frame)')
        f.addRow('Frame', self.frame)
        self.xyz = [_spin(-100000, 100000, 2, ' mm') for _ in range(3)]
        prow = QHBoxLayout()
        for lbl, s in zip('XYZ', self.xyz):
            prow.addWidget(QLabel(lbl))
            prow.addWidget(s)
        f.addRow('Position', prow)
        self.rpy = [_spin(-180, 180, 2, '°'), _spin(-90, 90, 2, '°'), _spin(-180, 180, 2, '°')]
        orow = QHBoxLayout()
        for lbl, s in zip(('Roll', 'Pitch', 'Yaw'), self.rpy):
            orow.addWidget(QLabel(lbl))
            orow.addWidget(s)
        f.addRow('Orientation', orow)
        self.quaternion = QLabel('')
        self.quaternion.setStyleSheet('color:#777777;')
        f.addRow('Stored (qx, qy, qz, qw)', self.quaternion)
        # ADR-0028: the joint configuration the pose is reached in (a PTP / free move honours it or refuses)
        self.configuration = QComboBox()
        self.configuration.setToolTip('The joint waypoint this pose was taught with (Capture TCP + joints records '
                                      'it): a PTP or free move reaches the pose in that configuration, or refuses. '
                                      'None: the configuration nearest where the robot is.')
        f.addRow('Configuration', self.configuration)
        self.forms['TCP'] = w
        self.stack.addWidget(w)

    def _build_joints(self):
        w = QWidget()
        self.joint_form = QFormLayout(w)
        self.joint_spins: List[QDoubleSpinBox] = []
        self.forms['Joints'] = w
        self.stack.addWidget(w)

    def _build_named(self):
        w = QWidget()
        f = QFormLayout(w)
        self.named = QComboBox()
        f.addRow('Named state', self.named)
        f.addRow(QLabel('A pose of the MoveIt config (its SRDF): edited there, not here.'))
        self.forms['Named'] = w
        self.stack.addWidget(w)

    def _build_relative(self):
        w = QWidget()
        f = QFormLayout(w)
        self.rel_to = QComboBox()
        f.addRow('Relative to', self.rel_to)
        self.rel_frame = QComboBox()
        self.rel_frame.addItems(['base', 'tool', 'target_local'])
        f.addRow('Along', self.rel_frame)
        self.offsets = [_spin(-100000, 100000, 2, ' mm') for _ in range(3)]
        orow = QHBoxLayout()
        for lbl, s in zip(('dx', 'dy', 'dz'), self.offsets):
            orow.addWidget(QLabel(lbl))
            orow.addWidget(s)
        f.addRow('Offset', orow)
        # ADR-0028: a relative pose names its own configuration (it never inherits its reference's)
        self.rel_configuration = QComboBox()
        self.rel_configuration.setToolTip(self.configuration.toolTip())
        f.addRow('Configuration', self.rel_configuration)
        self.forms['Relative'] = w
        self.stack.addWidget(w)

    # ---- data --------------------------------------------------------------------------------------
    @property
    def composer(self):
        return self._composer()

    def waypoints(self) -> List[dict]:
        return list(self.composer.document().get('waypoints') or [])

    def refresh(self) -> None:
        c = self.composer
        wps = self.waypoints()
        self.table.blockSignals(True)
        self.table.setRowCount(len(wps))
        for i, wp in enumerate(wps):
            users = c.waypoint_users(wp['name'])
            for col, text in enumerate((wp['name'], pose.kind(wp), pose.summary(wp),
                                        ', '.join(users) or '—')):
                self.table.setItem(i, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        self.table.resizeColumnToContents(0)
        if self.selected in [w['name'] for w in wps]:
            self.select(self.selected)
        else:
            self.selected = None
            self.editor.setEnabled(False)

    def select(self, name: str) -> None:
        names = [w['name'] for w in self.waypoints()]
        if name not in names:
            return
        self.selected = name
        self.table.blockSignals(True)
        self.table.selectRow(names.index(name))
        self.table.blockSignals(False)
        self._load(self.waypoints()[names.index(name)])
        self._on_select_cb(name)

    def _on_select(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if rows:
            item = self.table.item(min(rows), 0)
            if item is not None:
                self.select(item.text())

    def _load(self, wp: dict) -> None:
        k = pose.kind(wp)
        self.stack.setCurrentWidget(self.forms[k])
        self.editor.setTitle(f"Edit '{wp['name']}' ({k})")
        self.editor.setEnabled(True)
        self._shown = {'kind': k, 'wp': wp}
        if k == 'TCP':
            self.frame.setText(wp.get('frame') or '')
            shown_mm = pose.mm(wp.get('position') or [0, 0, 0])
            shown_rpy = pose.rpy_deg(wp.get('orientation') or [0, 0, 0, 1])
            for s, val in zip(self.xyz, shown_mm):
                s.setValue(val)
            for s, val in zip(self.rpy, shown_rpy):
                s.setValue(val)
            # what the fields REALLY show (a spin clamps an out-of-range value): untouched = kept exact
            self._shown.update(mm=[s.value() for s in self.xyz], rpy=[s.value() for s in self.rpy])
            self.quaternion.setText(', '.join(f'{q:.6f}' for q in wp.get('orientation') or []))
            self._fill_configuration(self.configuration, wp)
        elif k == 'Joints':
            self._joint_rows(wp.get('joints') or [])
        elif k == 'Named':
            self.named.clear()
            states = [s.name for s in self.composer.project.robot.named_states]
            self.named.addItems(states)
            if wp['named'] not in states:
                self.named.addItem(wp['named'])
            self.named.setCurrentText(wp['named'])
        else:
            rel = wp['relative']
            self.rel_to.clear()
            self.rel_to.addItems([w['name'] for w in self.waypoints() if w['name'] != wp['name']])
            self.rel_to.setCurrentText(rel['to'])
            self.rel_frame.setCurrentText(rel['frame'])
            for s, key in zip(self.offsets, ('dx', 'dy', 'dz')):
                s.setValue(round(float(rel.get(key, 0.0)) * 1000.0, 2))
            self._shown['offsets'] = [s.value() for s in self.offsets]
            self._fill_configuration(self.rel_configuration, wp)

    def _fill_configuration(self, combo, wp: dict) -> None:
        """The joint waypoints a pose can name as its configuration (ADR-0028), the current one selected (kept
        even when no joint waypoint has that name any more: the validator says so)."""
        combo.clear()
        combo.addItem(NO_CONFIGURATION)
        combo.addItems([w['name'] for w in self.waypoints() if pose.kind(w) in ('Joints', 'Named')])
        if wp.get('configuration') and combo.findText(wp['configuration']) < 0:
            combo.addItem(wp['configuration'])
        combo.setCurrentText(wp.get('configuration') or NO_CONFIGURATION)

    def _joint_rows(self, values) -> None:
        while self.joint_form.rowCount():
            self.joint_form.removeRow(0)
        names = list(self.composer.project.robot.planning_group.joints)
        shown = pose.joints_deg(values)
        self.joint_spins = []
        for i, name in enumerate(names):
            s = _spin(-720, 720, 2, '°')
            s.setValue(shown[i] if i < len(shown) else 0.0)
            self.joint_form.addRow(name, s)
            self.joint_spins.append(s)
        self._shown['joints'] = [s.value() for s in self.joint_spins]

    # ---- operations --------------------------------------------------------------------------------
    def _unique(self, stem: str) -> str:
        names = {w['name'] for w in self.waypoints()}
        i = 1
        while f'{stem}{i}' in names:
            i += 1
        return f'{stem}{i}'

    def new_waypoint(self, k: str, name: Optional[str] = None) -> Optional[str]:
        """A new waypoint of kind ``k`` with neutral values, to teach or type."""
        name = name or self._unique('wp')
        robot = self.composer.project.robot
        if k == 'TCP':
            tcps = [w for w in self.waypoints() if pose.kind(w) == 'TCP']
            spec = pose.tcp_waypoint(name, [0.4, 0.0, 0.3],
                                     tcps[0]['orientation'] if tcps else [0.0, 0.0, 0.0, 1.0])
        elif k == 'Joints':
            spec = pose.joint_waypoint(name, [0.0] * len(robot.planning_group.joints))
        elif k == 'Named':
            if not robot.named_states:
                self._error('the MoveIt config has no named state')
                return None
            spec = pose.named_waypoint(name, robot.named_states[0].name)
        else:
            others = self.waypoints()
            if not others:
                self._error('a relative waypoint needs another waypoint to start from')
                return None
            spec = pose.relative_waypoint(name, others[0]['name'], 'tool', dz_mm=-50.0)
        if self._do(lambda: self.composer.set_waypoint(spec)):
            self.selected = name
            self.refresh()
            return name
        return None

    def spec_from_editor(self) -> dict:
        """The canonical waypoint the editor now describes (unchanged values kept exactly)."""
        wp = self._shown['wp']
        k = self._shown['kind']
        if k == 'TCP':
            position = pose.edited_position(wp.get('position'), self._shown['mm'], [s.value() for s in self.xyz])
            orientation = pose.edited_orientation(wp.get('orientation'), self._shown['rpy'],
                                                  [s.value() for s in self.rpy])
            out = pose.tcp_waypoint(wp['name'], position, orientation, self.frame.text().strip() or None)
            if wp.get('tcp'):
                out['tcp'] = wp['tcp']
            _set_configuration(out, self.configuration.currentText())
            return out
        if k == 'Joints':
            return pose.joint_waypoint(wp['name'], pose.edited_joints(
                wp.get('joints'), self._shown['joints'], [s.value() for s in self.joint_spins]))
        if k == 'Named':
            return pose.named_waypoint(wp['name'], self.named.currentText())
        out = copy.deepcopy(wp)                       # every other key as written
        rel = out['relative']
        rel['to'], rel['frame'] = self.rel_to.currentText(), self.rel_frame.currentText()
        for key, shown, entered in zip(('dx', 'dy', 'dz'), self._shown['offsets'], [s.value() for s in self.offsets]):
            if abs(shown - entered) < 1e-9:
                continue                              # untouched: the stored offset stays exact
            if entered == 0.0:
                rel.pop(key, None)
            else:
                rel[key] = round(entered / 1000.0, pose.POSITION_DIGITS) + 0.0
        _set_configuration(out, self.rel_configuration.currentText())
        return out

    def apply(self) -> bool:
        if not self._shown:
            return False
        spec = self.spec_from_editor()
        ok = self._do(lambda: self.composer.set_waypoint(spec))
        if ok:
            self.message.setText(f"'{spec['name']}' saved: {pose.summary(spec)}")
            self.refresh()
        return ok

    def rename(self, old: Optional[str], new: str) -> bool:
        if not old or not new:
            return False
        ok = self._do(lambda: self.composer.rename_waypoint(old, new))
        if ok:
            self.selected = new
            self.rename_edit.clear()
            self.refresh()
        return ok

    def delete(self, name: Optional[str]) -> bool:
        if not name:
            return False
        ok = self._do(lambda: self.composer.delete_waypoint(name))
        if ok:
            self.selected = None
            self.refresh()
        return ok

    def _do(self, op) -> bool:
        try:
            op()
        except ComposerError as exc:
            self._error(str(exc))
            return False
        self.message.setStyleSheet('')
        self._changed()
        return True

    def _error(self, text: str) -> None:
        self.message.setStyleSheet('color:#c62828;')
        self.message.setText(text)
