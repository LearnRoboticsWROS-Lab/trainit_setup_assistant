"""The Profiles section (vNext I0.6, I0_DESIGN.md §23): reusable application-level motion profiles.

The common view is speed, acceleration (% of the joint limits) and TCP velocity; the Advanced view adds
TCP acceleration, goal tolerance and planning time / attempts. A profile is canonical data. The "intended
usage" text of a preset is shown in the UI only, never stored. Values the user did not touch are kept
exactly, and so is every key the editor does not show.
"""

from __future__ import annotations

import copy
from typing import Callable, Dict, Optional

from python_qt_binding.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                                         QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPushButton, QSpinBox,
                                         QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...composer.ops import ComposerError

# name -> (values, intended usage shown in the UI)
PRESETS: Dict[str, tuple] = {
    'transit': ({'speed': 50, 'acceleration': 50}, 'free-space moves between stations (PTP)'),
    'approach': ({'speed': 30, 'acceleration': 30, 'tcp_velocity_mps': 0.1},
                 'the last centimetres to a part or a fixture (LIN)'),
    'process': ({'speed': 20, 'acceleration': 20, 'tcp_velocity_mps': 0.08},
                'the process itself: gluing, inspection, a path at constant speed'),
    'slow_precision': ({'speed': 10, 'acceleration': 10, 'tcp_velocity_mps': 0.02},
                       'insertion, tight clearances, first tests of a new cell'),
}


def _dspin(lo, hi, decimals, suffix, special='not set'):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setSpecialValueText(special)
    return s


class ProfilesPanel(QWidget):
    def __init__(self, composer: Callable[[], object], parent=None):
        super().__init__(parent)
        self._composer = composer
        self.selected: Optional[str] = None
        self._original: Optional[dict] = None
        self._shown: Dict[str, float] = {}
        v = QVBoxLayout(self)
        v.addWidget(QLabel('<b>Motion profiles</b> — how fast, reused by moves and paths by name'))
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(['Name', 'Speed', 'Acceleration', 'TCP velocity', 'Used by'])
        self.table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select)
        v.addWidget(self.table, 1)
        row = QHBoxLayout()
        self.preset = QComboBox()
        self.preset.addItems(list(PRESETS))
        self.new_button = QPushButton('New from preset')
        self.new_button.clicked.connect(lambda: self.new_profile(self.preset.currentText()))
        self.rename_edit = QLineEdit()
        self.rename_edit.setPlaceholderText('new name')
        self.rename_button = QPushButton('Rename')
        self.rename_button.clicked.connect(lambda: self.rename(self.selected, self.rename_edit.text().strip()))
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(lambda: self.delete(self.selected))
        for w in (self.preset, self.new_button):
            row.addWidget(w)
        row.addStretch(1)
        for w in (self.rename_edit, self.rename_button, self.delete_button):
            row.addWidget(w)
        v.addLayout(row)
        self.editor = QGroupBox('Edit')
        f = QFormLayout(self.editor)
        self.usage = QLabel('')
        self.usage.setStyleSheet('color:#555555; font-style:italic;')
        f.addRow(self.usage)
        self.speed = QSpinBox()
        self.speed.setRange(0, 100)
        self.speed.setSuffix(' %')
        self.speed.setSpecialValueText('not set')
        f.addRow('Speed (joint limits)', self.speed)
        self.accel = QSpinBox()
        self.accel.setRange(0, 100)
        self.accel.setSuffix(' %')
        self.accel.setSpecialValueText('not set')
        f.addRow('Acceleration', self.accel)
        self.tcp_vel = _dspin(0.0, 10.0, 3, ' m/s')
        f.addRow('TCP velocity (LIN / CIRC / paths)', self.tcp_vel)
        self.advanced = QCheckBox('Advanced')
        self.advanced.toggled.connect(self._show_advanced)
        f.addRow(self.advanced)
        self.tcp_acc = _dspin(0.0, 50.0, 3, ' m/s²')
        self.tol_pos = _dspin(0.0, 1000.0, 2, ' mm')
        self.planning_time = _dspin(0.0, 60.0, 2, ' s')
        self.attempts = QSpinBox()
        self.attempts.setRange(0, 100)
        self.attempts.setSpecialValueText('not set')
        self._advanced_rows = []
        for label, w in (('TCP acceleration', self.tcp_acc), ('Goal tolerance (position)', self.tol_pos),
                         ('Planning time', self.planning_time), ('Planning attempts', self.attempts)):
            lbl = QLabel(label)
            f.addRow(lbl, w)
            self._advanced_rows += [lbl, w]
        self._show_advanced(False)
        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply)
        f.addRow(self.apply_button)
        v.addWidget(self.editor)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        v.addWidget(self.message)
        self.editor.setEnabled(False)

    def _show_advanced(self, on: bool) -> None:
        for w in self._advanced_rows:
            w.setVisible(on)

    @property
    def composer(self):
        return self._composer()

    def profiles(self) -> dict:
        return dict(self.composer.document().get('motion_profiles') or {})

    def refresh(self) -> None:
        profiles = self.profiles()
        self.table.blockSignals(True)
        self.table.setRowCount(len(profiles))
        for i, (name, p) in enumerate(profiles.items()):
            vel = p.get('tcp_velocity_mps')
            cells = (name, f"{p['speed']} %" if p.get('speed') else '—',
                     f"{p['acceleration']} %" if p.get('acceleration') else '—',
                     f'{vel:g} m/s' if vel else '—', ', '.join(self.composer.profile_users(name)) or '—')
            for col, text in enumerate(cells):
                self.table.setItem(i, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        if self.selected in profiles:
            self.select(self.selected)
        else:
            self.selected, self._original = None, None
            self.editor.setEnabled(False)

    def select(self, name: str) -> None:
        profiles = self.profiles()
        if name not in profiles:
            return
        self.selected, self._original = name, copy.deepcopy(profiles[name])
        p = self._original
        self.editor.setTitle(f"Edit profile '{name}'")
        self.editor.setEnabled(True)
        self.usage.setText(PRESETS[name][1] if name in PRESETS else '')
        self.speed.setValue(int(p.get('speed') or 0))
        self.accel.setValue(int(p.get('acceleration') or 0))
        self.tcp_vel.setValue(float(p.get('tcp_velocity_mps') or 0.0))
        self.tcp_acc.setValue(float(p.get('tcp_acceleration_mps2') or 0.0))
        self.tol_pos.setValue(round(float((p.get('goal_tolerance') or {}).get('position_m') or 0.0) * 1000.0, 2))
        self.planning_time.setValue(float(p.get('planning_time_s') or 0.0))
        self.attempts.setValue(int(p.get('planning_attempts') or 0))
        self._shown = {k: w.value() for k, w in self._fields().items()}

    def _fields(self):
        return {'speed': self.speed, 'acceleration': self.accel, 'tcp_velocity_mps': self.tcp_vel,
                'tcp_acceleration_mps2': self.tcp_acc, 'position_m': self.tol_pos,
                'planning_time_s': self.planning_time, 'planning_attempts': self.attempts}

    def _on_select(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if rows:
            self.select(self.table.item(min(rows), 0).text())

    def spec_from_editor(self) -> dict:
        """The profile as edited: a field not touched keeps its stored value; 0 = not set."""
        p = copy.deepcopy(self._original)
        for key, w in self._fields().items():
            value = w.value()
            if abs(value - self._shown[key]) < 1e-9:
                continue
            if key == 'position_m':
                tol = dict(p.get('goal_tolerance') or {})
                if value:
                    tol['position_m'] = round(value / 1000.0, 6)
                else:
                    tol.pop('position_m', None)
                if tol:
                    p['goal_tolerance'] = tol
                else:
                    p.pop('goal_tolerance', None)
            elif value:
                p[key] = int(value) if key in ('speed', 'acceleration', 'planning_attempts') else float(value)
            else:
                p.pop(key, None)
        return p

    def apply(self) -> bool:
        if self._original is None:
            return False
        name, spec = self.selected, self.spec_from_editor()
        ok = self._do(lambda: self.composer.set_profile(name, spec))
        if ok:
            self.message.setText(f"'{name}' saved")
            self.refresh()
        return ok

    def new_profile(self, preset: str, name: Optional[str] = None) -> Optional[str]:
        values = PRESETS[preset][0]
        existing = self.profiles()
        name = name or preset
        i = 2
        while name in existing:
            name = f'{preset}{i}'
            i += 1
        if self._do(lambda: self.composer.set_profile(name, dict(values))):
            self.selected = name
            self.refresh()
            return name
        return None

    def rename(self, old: Optional[str], new: str) -> bool:
        if not old or not new:
            return False
        ok = self._do(lambda: self.composer.rename_profile(old, new))
        if ok:
            self.selected = new
            self.rename_edit.clear()
            self.refresh()
        return ok

    def delete(self, name: Optional[str]) -> bool:
        if not name:
            return False
        ok = self._do(lambda: self.composer.delete_profile(name))
        if ok:
            self.selected = None
            self.refresh()
        return ok

    def _do(self, op) -> bool:
        try:
            op()
        except ComposerError as exc:
            self.message.setStyleSheet('color:#c62828;')
            self.message.setText(str(exc))
            return False
        self.message.setStyleSheet('')
        return True
