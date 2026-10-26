"""The Paths section (vNext I0.5, I0_DESIGN.md §15): ONE motion through ordered waypoints.

WHERE = the waypoints (the states); HOW = each relation (LIN / CIRC + via); END BEHAVIOUR = STOP at every
state, or BLEND through them with a radius per segment. The editor rewrites only these, plus the path's
profile and goal tolerance. Every other key the path carries (events, frame, tcp, provider, per-segment
limits) is kept as written. A dense path (a generated ``source``) is shown, not edited.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtWidgets import (QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox, QHBoxLayout,
                                         QHeaderView, QLabel, QLineEdit, QListWidget, QPushButton,
                                         QTableWidget, QVBoxLayout, QWidget)

from ...composer import pose
from ...composer.ops import ComposerError

HOW = ('lin', 'circ')


def path_summary(p: dict) -> str:
    """``w1 → w2 (LIN, blend 20 mm) → w3 (CIRC via w2b) → w4 · BLEND · profile process``."""
    if p.get('source'):
        gen = (p['source'].get('generator') or {}).get('type', 'generated')
        return f'dense path ({gen}): edited in the YAML'
    states, segs = p.get('states') or [], p.get('segments') or []
    parts = [states[0]] if states else []
    for i, seg in enumerate(segs):
        how = seg.get('type', '?').upper()
        if seg.get('via'):
            how += f" via {seg['via']}"
        if seg.get('center'):
            how += f" about {seg['center']}"
        if seg.get('blend_radius_m'):
            how += f", blend {seg['blend_radius_m'] * 1000.0:g} mm"
        parts.append(f'{states[i + 1] if i + 1 < len(states) else "?"} ({how})')
    tail = [(p.get('continuity') or 'stop').upper()]
    if p.get('profile'):
        tail.append(f"profile {p['profile']}")
    return ' → '.join(parts) + ' · ' + ' · '.join(tail)


class PathsPanel(QWidget):
    def __init__(self, composer: Callable[[], object], parent=None):
        super().__init__(parent)
        self._composer = composer
        self.selected: Optional[str] = None
        self._original: Optional[dict] = None
        v = QVBoxLayout(self)
        v.addWidget(QLabel('<b>Paths</b> — ONE motion through ordered waypoints: stop at each, or blend '
                           'through them'))
        # owner Test B: "How" offered only LIN / CIRC and a joint waypoint could be picked as a state
        note = QLabel('A path is one Cartesian motion of the TCP: LIN / CIRC segments between TCP (or relative) '
                      'waypoints. PTP and Free moves, and joint / named targets, are Move steps in Tasks.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#555;')
        v.addWidget(note)
        self.list = QListWidget()
        self.list.currentTextChanged.connect(self._on_select)
        v.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.new_name = QLineEdit()
        self.new_name.setPlaceholderText('new path name')
        self.new_button = QPushButton('New path')
        self.new_button.clicked.connect(lambda: self.new_path(self.new_name.text().strip() or None))
        self.delete_button = QPushButton('Delete')
        self.delete_button.clicked.connect(lambda: self.delete(self.selected))
        for w in (self.new_name, self.new_button):
            row.addWidget(w)
        row.addStretch(1)
        row.addWidget(self.delete_button)
        v.addLayout(row)
        self.editor = QGroupBox('Edit')
        ev = QVBoxLayout(self.editor)
        form = QFormLayout()
        self.continuity = QComboBox()
        self.continuity.addItems(['stop', 'blend'])
        self.continuity.setToolTip('STOP: the robot stops at every waypoint. BLEND: it passes through them, '
                                   'rounding each corner by the segment\'s blend radius.')
        form.addRow('End behaviour', self.continuity)
        self.profile = QComboBox()
        form.addRow('Profile', self.profile)
        self.tol_pos = QDoubleSpinBox()
        self.tol_pos.setRange(0.0, 1000.0)
        self.tol_pos.setDecimals(2)
        self.tol_pos.setSuffix(' mm')
        self.tol_pos.setSpecialValueText('default')
        form.addRow('Goal tolerance (position)', self.tol_pos)
        ev.addLayout(form)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Waypoint', 'How (from the previous)', 'Via (CIRC)', 'Blend radius'])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in (1, 2, 3):                            # owner Test B (row 19): headers never truncated
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        ev.addWidget(self.table)
        trow = QHBoxLayout()
        self.add_state_button = QPushButton('Add waypoint')
        self.add_state_button.clicked.connect(lambda: self.add_state())
        self.remove_state_button = QPushButton('Remove last')
        self.remove_state_button.clicked.connect(self.remove_last_state)
        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply)
        for w in (self.add_state_button, self.remove_state_button):
            trow.addWidget(w)
        trow.addStretch(1)
        trow.addWidget(self.apply_button)
        ev.addLayout(trow)
        v.addWidget(self.editor)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        v.addWidget(self.message)
        self.editor.setEnabled(False)

    @property
    def composer(self):
        return self._composer()

    def paths(self) -> List[dict]:
        return list(self.composer.document().get('paths') or [])

    def _waypoints(self) -> List[str]:
        """What a path state or a CIRC via can be: a pose (TCP or relative), never a joint target (R-M3-04).
        A joint target a path already names stays shown (``_combo``), so the validator's error can be read."""
        doc = self.composer.document()
        return [w['name'] for w in doc.get('waypoints') or [] if pose.kind(w) in ('TCP', 'Relative')]

    def refresh(self) -> None:
        self.list.blockSignals(True)
        self.list.clear()
        for p in self.paths():
            self.list.addItem(p['name'])
            self.list.item(self.list.count() - 1).setToolTip(path_summary(p))
        self.list.blockSignals(False)
        names = [p['name'] for p in self.paths()]
        if self.selected in names:
            self.select(self.selected)
        else:
            self.selected, self._original = None, None
            self.editor.setEnabled(False)

    def select(self, name: str) -> None:
        p = next((x for x in self.paths() if x['name'] == name), None)
        if p is None:
            return
        self.selected, self._original = name, copy.deepcopy(p)
        self.list.blockSignals(True)
        items = self.list.findItems(name, Qt.MatchExactly)
        if items:
            self.list.setCurrentItem(items[0])
        self.list.blockSignals(False)
        self.editor.setTitle(f"Edit path '{name}' — {path_summary(p)}")
        dense = bool(p.get('source'))
        self.editor.setEnabled(not dense)
        if dense:
            self.message.setText(path_summary(p))
            return
        self.continuity.setCurrentText(p.get('continuity') or 'stop')
        self.profile.clear()
        self.profile.addItem('')
        self.profile.addItems(list((self.composer.document().get('motion_profiles') or {})))
        self.profile.setCurrentText(p.get('profile') or '')
        tol = (p.get('goal_tolerance') or {}).get('position_m')
        self.tol_pos.setValue(round(tol * 1000.0, 2) if tol else 0.0)
        self._fill(p.get('states') or [], p.get('segments') or [])

    def _on_select(self, name: str) -> None:
        if name:
            self.select(name)

    def _combo(self, items, current) -> QComboBox:
        c = QComboBox()
        c.addItems(list(items))
        if current and c.findText(current) < 0:
            c.addItem(current)
        c.setCurrentText(current or '')
        return c

    def _fill(self, states: List[str], segments: List[dict]) -> None:
        wps = self._waypoints()
        self.table.setRowCount(len(states))
        for i, st in enumerate(states):
            self.table.setCellWidget(i, 0, self._combo(wps, st))
            if i == 0:
                self.table.setCellWidget(i, 1, QLabel('start (the robot must be here)'))
                self.table.setCellWidget(i, 2, QLabel(''))
                self.table.setCellWidget(i, 3, QLabel(''))
                continue
            seg = segments[i - 1] if i - 1 < len(segments) else {'type': 'lin'}
            self.table.setCellWidget(i, 1, self._combo(HOW, seg.get('type', 'lin')))
            self.table.setCellWidget(i, 2, self._combo([''] + wps, seg.get('via') or ''))
            r = QDoubleSpinBox()
            r.setRange(0.0, 1000.0)
            r.setDecimals(2)
            r.setSuffix(' mm')
            r.setValue(round((seg.get('blend_radius_m') or 0.0) * 1000.0, 2))
            self.table.setCellWidget(i, 3, r)

    def _rows(self):
        out = []
        for i in range(self.table.rowCount()):
            state = self.table.cellWidget(i, 0).currentText()
            if i == 0:
                out.append((state, None, None, None))
                continue
            out.append((state, self.table.cellWidget(i, 1).currentText(),
                        self.table.cellWidget(i, 2).currentText() or None,
                        self.table.cellWidget(i, 3).value()))
        return out

    def add_state(self, waypoint: Optional[str] = None) -> None:
        rows = self._rows()
        states = [r[0] for r in rows] + [waypoint or (self._waypoints() or [''])[0]]
        segs = self._segments_from(rows) + [{'type': 'lin'}]
        self._fill(states, segs)

    def remove_last_state(self) -> None:
        rows = self._rows()
        if len(rows) > 2:
            self._fill([r[0] for r in rows[:-1]], self._segments_from(rows)[:-1])

    def _segments_from(self, rows) -> List[dict]:
        old = (self._original or {}).get('segments') or []
        segs = []
        for i, (_state, how, via, radius) in enumerate(rows[1:]):
            seg = copy.deepcopy(old[i]) if i < len(old) else {}
            seg['type'] = how
            seg.pop('via', None) if how != 'circ' or not via else seg.__setitem__('via', via)
            if how != 'circ':
                seg.pop('center', None)
            if radius:
                seg['blend_radius_m'] = round(radius / 1000.0, 6)
            else:
                seg.pop('blend_radius_m', None)
            segs.append(seg)
        return segs

    def spec_from_editor(self) -> dict:
        """The canonical path the editor describes: states, segments, continuity, profile and position
        tolerance from the editor; every other key as written. A STOP path carries no blend radius and
        the last segment never blends (the model's rules)."""
        p = copy.deepcopy(self._original)
        rows = self._rows()
        p['states'] = [r[0] for r in rows]
        segs = self._segments_from(rows)
        continuity = self.continuity.currentText()
        for i, seg in enumerate(segs):
            if continuity == 'stop' or i == len(segs) - 1:
                seg.pop('blend_radius_m', None)
        p['segments'] = segs
        if continuity == 'blend':
            p['continuity'] = 'blend'
        elif p.get('continuity'):
            p['continuity'] = 'stop'
        if self.profile.currentText():
            p['profile'] = self.profile.currentText()
        else:
            p.pop('profile', None)
        tol = dict(p.get('goal_tolerance') or {})
        if self.tol_pos.value() > 0:
            tol['position_m'] = round(self.tol_pos.value() / 1000.0, 6)
        else:
            tol.pop('position_m', None)
        if tol:
            p['goal_tolerance'] = tol
        else:
            p.pop('goal_tolerance', None)
        return p

    def apply(self) -> bool:
        if self._original is None:
            return False
        spec = self.spec_from_editor()
        ok = self._do(lambda: self.composer.set_path(spec))
        if ok:
            self.message.setText(f"'{spec['name']}' saved: {path_summary(spec)}")
            self.refresh()
        return ok

    def new_path(self, name: Optional[str] = None) -> Optional[str]:
        wps = self._waypoints()
        if len(wps) < 2:
            self._error('a path needs at least two waypoints: teach them first')
            return None
        names = {p['name'] for p in self.paths()}
        if not name:
            i = 1
            while f'path{i}' in names:
                i += 1
            name = f'path{i}'
        spec = {'name': name, 'states': wps[:2], 'segments': [{'type': 'lin'}]}
        if self._do(lambda: self.composer.set_path(spec)):
            self.selected = name
            self.new_name.clear()
            self.refresh()
            return name
        return None

    def delete(self, name: Optional[str]) -> bool:
        if not name:
            return False
        ok = self._do(lambda: self.composer.delete_path(name))
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
        return True

    def _error(self, text: str) -> None:
        self.message.setStyleSheet('color:#c62828;')
        self.message.setText(text)
