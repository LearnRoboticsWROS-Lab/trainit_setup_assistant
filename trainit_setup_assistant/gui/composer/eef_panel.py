"""The Composer's End effector section (vNext I0, I0_DESIGN.md §34 E4, owner decision 2026-10-26).

The end effector is configured in Step 7, beside the motion it works with: its CONTRACT (ADR-0022) -- the
actuators and their named targets (what an actuator can be commanded to, as a waypoint is what the arm can be
sent to), the grasp sites, the verification policy and the adapter that serves it on each deployed backend.
The Tasks section then commands it: grasp / release, Close / Open gripper, Tool (actuator / verify).

No contract yet: "Suggest from the MoveIt config and the world" proposes one (``composer.eef_suggest``), with the
source of every value and the decisions left; "Adopt" makes it the project's in one undoable edit. Every edit is
a ``composer.eef_ops`` operation, validated by the canonical model; the contract's own rules (ADR-0022 R1-R26)
are reported in Diagnostics. TSA never commands the end effector (ADR-0027).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import yaml
from python_qt_binding.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QGroupBox,
                                         QHBoxLayout, QHeaderView, QLabel, QLineEdit, QPlainTextEdit,
                                         QPushButton, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout,
                                         QWidget)

from ...catalog import EEF_ADAPTER_ROWS, EEF_INTERFACES as INTERFACES, EEF_KINDS as KINDS
from ...composer import eef_ops
from ...composer.ops import ComposerError
from ...model.end_effector import ResultPolicy

POLICIES = tuple(p.value for p in ResultPolicy)


def _table(headers: List[str]) -> QTableWidget:
    t = QTableWidget(0, len(headers))
    t.setHorizontalHeaderLabels(headers)
    for col in range(len(headers) - 1):
        t.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
    t.horizontalHeader().setSectionResizeMode(len(headers) - 1, QHeaderView.Stretch)
    t.setSelectionBehavior(QTableWidget.SelectRows)
    t.setEditTriggers(QTableWidget.NoEditTriggers)
    t.setMaximumHeight(130)
    return t


def _fail(message: str):
    raise ValueError(message)


def _numbers(text: str) -> List[float]:
    out = []
    for x in (x.strip() for x in text.replace(';', ',').split(',')):
        if x:
            try:
                out.append(float(x))
            except ValueError:
                raise ValueError(f'{x!r} is not a number') from None
    return out


def _names(text: str) -> List[str]:
    return [x.strip() for x in text.replace(';', ',').split(',') if x.strip()]


class EndEffectorPanel(QWidget):
    def __init__(self, composer: Callable[[], object], live: Optional[Callable[[], object]] = None, parent=None):
        super().__init__(parent)
        self._composer = composer
        self._live = live
        self.suggestion = None
        self.selected_actuator: Optional[str] = None
        outer = QVBoxLayout(self)
        head = QLabel('<b>End effector</b> — what the tool exposes, how each deployment drives it; '
                      'commanded from Tasks (Grasp, Release, Close / Open gripper, Tool)')
        head.setWordWrap(True)
        outer.addWidget(head)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        outer.addWidget(self.status)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        from python_qt_binding.QtCore import Qt
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)   # the content follows the width
        body = QWidget()
        self.v = QVBoxLayout(body)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        self._build_suggest()
        self._build_device()
        self._build_actuators()
        self._build_sites()
        self._build_verification()
        self._build_bindings()
        live_box = QGroupBox('Live (read-only: TSA never commands the end effector, ADR-0027)')
        lv = QVBoxLayout(live_box)
        self.check_button = QPushButton('Check the end effector (live)')
        self.check_button.setToolTip('With the bundle running (Step 8): who publishes the device state, which '
                                     'endpoints are served, the tool joints, the latest state. Nothing is commanded.')
        self.check_button.clicked.connect(self.check_live)
        lv.addWidget(self.check_button)
        self.live_result = QLabel('')
        self.live_result.setWordWrap(True)
        result_scroll = QScrollArea()                    # bounded: the sections below keep their room
        result_scroll.setWidgetResizable(True)
        result_scroll.setWidget(self.live_result)
        result_scroll.setMaximumHeight(110)
        result_scroll.setFrameShape(QScrollArea.NoFrame)
        lv.addWidget(result_scroll)
        self.contract_widgets.append(live_box)
        # owner Test F.1 (row 37): always in view, under the section's status -- not after a long scroll
        outer.insertWidget(outer.indexOf(scroll), live_box)
        self.yaml_view = QPlainTextEdit()
        self.yaml_view.setReadOnly(True)
        self.yaml_view.setMaximumHeight(160)
        yaml_label = QLabel('The contract, as project.yaml holds it (robot.gripper.contract):')
        self.contract_widgets += [yaml_label, self.yaml_view]
        self.v.addWidget(yaml_label)
        self.v.addWidget(self.yaml_view)
        self.v.addStretch(1)

    def composer(self):
        return self._composer()

    # ---- building ------------------------------------------------------------------------------------------
    def _build_suggest(self) -> None:
        box = QGroupBox('No end-effector contract yet')
        f = QVBoxLayout(box)
        intro = QLabel('Grasp, Release, Close / Open gripper and Tool steps command the end effector '
                       'through its contract. TSA proposes one from what it read (the MoveIt config at '
                       'Step 1, the world at Step 2): every value with its source, and what you decide.')
        intro.setWordWrap(True)
        f.addWidget(intro)
        row = QHBoxLayout()
        self.suggest_button = QPushButton('Suggest from the MoveIt config and the world')
        self.suggest_button.clicked.connect(self.suggest)
        self.adopt_button = QPushButton('Adopt this contract')
        self.adopt_button.clicked.connect(self.adopt)
        self.adopt_button.setEnabled(False)
        row.addWidget(self.suggest_button)
        row.addWidget(self.adopt_button)
        row.addStretch(1)
        f.addLayout(row)
        self.suggest_view = QPlainTextEdit()
        self.suggest_view.setReadOnly(True)
        self.suggest_view.setMaximumHeight(170)
        f.addWidget(self.suggest_view)
        self.suggest_box = box
        self.v.addWidget(box)

    def _build_device(self) -> None:
        box = QGroupBox('Device')
        f = QHBoxLayout(box)
        f.addWidget(QLabel('Kind'))
        self.kind = QComboBox()
        self.kind.addItems(list(KINDS))
        f.addWidget(self.kind)
        b = QPushButton('Set kind')
        b.clicked.connect(lambda: self._do(lambda: eef_ops.set_kind(self.composer(), self.kind.currentText())))
        f.addWidget(b)
        f.addWidget(QLabel('Device id'))
        self.device_id = QLineEdit()
        f.addWidget(self.device_id)
        b = QPushButton('Rename device')
        b.clicked.connect(lambda: self._do(lambda: eef_ops.set_device_id(self.composer(),
                                                                         self.device_id.text().strip())))
        f.addWidget(b)
        self.remove_button = QPushButton('Remove the contract')
        self.remove_button.clicked.connect(lambda: self._do(lambda: eef_ops.remove_contract(self.composer())))
        f.addWidget(self.remove_button)
        self.device_box = box
        self.v.addWidget(box)

    def _build_actuators(self) -> None:
        box = QGroupBox('Actuators')
        box.setToolTip('Each actuator has ONE command interface and its named targets (ADR-0022)')
        v = QVBoxLayout(box)
        self.actuators = _table(['Id', 'Interface', 'Joints', 'Named targets', 'Used by'])
        self.actuators.itemSelectionChanged.connect(self._on_actuator)
        v.addWidget(self.actuators)
        row = QHBoxLayout()
        self.long_button = QPushButton('Make it a multi-actuator tool')
        self.long_button.setToolTip('Writes the derived values of the single actuator and names its site, so '
                                    'more actuators and sites can be added')
        self.long_button.clicked.connect(lambda: self._do(lambda: eef_ops.to_long_form(self.composer())))
        self.new_actuator = QLineEdit()
        self.new_actuator.setPlaceholderText('new actuator id')
        b_add = QPushButton('Add actuator')
        b_add.clicked.connect(self.add_actuator)
        b_ren = QPushButton('Rename')
        b_ren.clicked.connect(self.rename_actuator)
        b_del = QPushButton('Delete')
        b_del.clicked.connect(lambda: self._do(lambda: eef_ops.delete_actuator(self.composer(),
                                                                               self.selected_actuator or '')))
        for w in (self.long_button, self.new_actuator, b_add, b_ren, b_del):
            row.addWidget(w)
        row.addStretch(1)
        v.addLayout(row)
        form_box = QGroupBox('Edit the selected actuator')
        f = QFormLayout(form_box)
        self.interface = QComboBox()
        self.interface.addItems(list(INTERFACES))
        self.interface.currentTextChanged.connect(self._shape_actuator)
        f.addRow('Command interface', self.interface)
        self.action = QLineEdit()
        self.action.setPlaceholderText('<controller>/<action> (empty: derived from the MoveIt config)')
        self.joints = QLineEdit()
        self.joints.setPlaceholderText('joint names, comma separated (empty: derived / none)')
        self.policy = QComboBox()
        self.policy.addItems(list(POLICIES))
        self.max_effort = QDoubleSpinBox()
        self.max_effort.setRange(0.0, 10000.0)
        self.max_effort.setSpecialValueText('not set')
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.0, 60.0)
        self.duration.setDecimals(2)
        self.duration.setSuffix(' s')
        self.duration.setSpecialValueText('not set')
        self.engaged = QLineEdit()
        self.engaged.setPlaceholderText('joint positions when engaged (a binary jaw), comma separated')
        self.released = QLineEdit()
        self.released.setPlaceholderText('joint positions when released')
        self.actuator_rows = {}
        for key, label, w in (('action', 'Controller action', self.action), ('joints', 'Joints', self.joints),
                              ('policy', 'Result policy', self.policy), ('max_effort', 'Max effort', self.max_effort),
                              ('duration', 'Duration', self.duration), ('engaged', 'Engaged positions', self.engaged),
                              ('released', 'Released positions', self.released)):
            lbl = QLabel(label)
            f.addRow(lbl, w)
            self.actuator_rows[key] = (lbl, w)
        b = QPushButton('Apply to the actuator')
        b.clicked.connect(self.apply_actuator)
        f.addRow(b)
        v.addWidget(form_box)
        tbox = QGroupBox('Named targets of the selected actuator')
        tbox.setToolTip('gripper_command: one value; follow_joint_trajectory: one value per joint; '
                        'set_bool: always engage / release')
        tv = QVBoxLayout(tbox)
        self.targets = _table(['Target', 'Value(s)'])
        self.targets.itemSelectionChanged.connect(self._on_target)
        tv.addWidget(self.targets)
        trow = QHBoxLayout()
        self.target_name = QLineEdit()
        self.target_name.setPlaceholderText('target name')
        self.target_value = QLineEdit()
        self.target_value.setPlaceholderText('value(s), comma separated')
        b_set = QPushButton('Set target')
        b_set.clicked.connect(self.set_target)
        self.target_new = QLineEdit()
        self.target_new.setPlaceholderText('new name')
        b_tren = QPushButton('Rename target')
        b_tren.clicked.connect(lambda: self._done_with(self._do(lambda: eef_ops.rename_target(
            self.composer(), self.selected_actuator or '', self.target_name.text().strip(),
            self.target_new.text().strip())), self.target_name, self.target_value, self.target_new))
        b_tdel = QPushButton('Delete target')
        b_tdel.clicked.connect(lambda: self._done_with(self._do(lambda: eef_ops.delete_target(
            self.composer(), self.selected_actuator or '', self.target_name.text().strip())),
            self.target_name, self.target_value))
        b_cap = QPushButton('Capture from the robot')
        b_cap.setToolTip("The selected actuator's joints as they are now (move the tool in RViz / MoveIt first): "
                         'the named target becomes their current value(s)')
        b_cap.clicked.connect(self.capture_target)
        for w in (self.target_name, self.target_value, b_set, b_cap, self.target_new, b_tren, b_tdel):
            trow.addWidget(w)
        tv.addLayout(trow)
        v.addWidget(tbox)
        self.actuators_box = box
        self.v.addWidget(box)

    def _build_sites(self) -> None:
        box = QGroupBox('Grasp sites')
        box.setToolTip('Where the tool holds an object: a TCP link and its grip actuator (a multi-actuator '
                       'tool names them)')
        v = QVBoxLayout(box)
        self.sites = _table(['Site', 'TCP link', 'Grip actuator', 'Engage', 'Release', 'Used by'])
        self.sites.itemSelectionChanged.connect(self._on_site)
        v.addWidget(self.sites)
        f = QHBoxLayout()
        self.site_id = QLineEdit()
        self.site_id.setPlaceholderText('site id')
        self.site_tcp = QLineEdit()
        self.site_tcp.setPlaceholderText('TCP link')
        self.site_grip = QComboBox()
        self.site_engage = QLineEdit()
        self.site_engage.setPlaceholderText('engage target (empty: derived)')
        self.site_release = QLineEdit()
        self.site_release.setPlaceholderText('release target (empty: derived)')
        b_set = QPushButton('Set site')
        b_set.clicked.connect(self.set_site)
        self.site_new = QLineEdit()
        self.site_new.setPlaceholderText('new name')
        b_ren = QPushButton('Rename')
        b_ren.clicked.connect(lambda: self._done_with(self._do(lambda: eef_ops.rename_site(
            self.composer(), self.site_id.text().strip(), self.site_new.text().strip())), self.site_id, self.site_new))
        b_del = QPushButton('Delete')
        b_del.clicked.connect(lambda: self._done_with(self._do(lambda: eef_ops.delete_site(
            self.composer(), self.site_id.text().strip())), self.site_id))
        for w in (self.site_id, self.site_tcp, self.site_grip, self.site_engage, self.site_release, b_set,
                  self.site_new, b_ren, b_del):
            f.addWidget(w)
        v.addLayout(f)
        self.sites_box = box
        self.v.addWidget(box)

    def _build_verification(self) -> None:
        box = QGroupBox('Verification')
        box.setToolTip('Confirm a grasp / a release with a sensor (ADR-0022 D4)')
        v = QVBoxLayout(box)
        row = QHBoxLayout()
        self.require_grasp = QCheckBox('Confirm every grasp (the part is held)')
        self.require_release = QCheckBox('Confirm every release (the part is gone)')
        b = QPushButton('Apply')
        b.clicked.connect(lambda: self._do(lambda: eef_ops.set_verification(
            self.composer(), self.require_grasp.isChecked(), self.require_release.isChecked())))
        for w in (self.require_grasp, self.require_release, b):
            row.addWidget(w)
        row.addStretch(1)
        v.addLayout(row)
        self.observe = QLabel('')
        self.observe.setWordWrap(True)
        v.addWidget(self.observe)
        self.verification_box = box
        self.v.addWidget(box)

    def _build_bindings(self) -> None:
        box = QGroupBox('Adapters per deployment')
        box.setToolTip('What serves the end effector on each deployed backend (ADR-0022)')
        self.bindings_form = QFormLayout(box)
        self.binding_rows: Dict[str, tuple] = {}
        self.bindings_box = box
        self.v.addWidget(box)
        self.contract_widgets = [self.device_box, self.actuators_box, self.sites_box, self.verification_box,
                                 self.bindings_box]

    # ---- state ---------------------------------------------------------------------------------------------
    def refresh(self) -> None:
        c = self.composer()
        k = eef_ops.contract_of(c)
        has = k is not None
        self.suggest_box.setVisible(not has)
        for w in self.contract_widgets:
            w.setVisible(has)
        if not has:
            self.status.setText('<b style="color:#8a6d3b">No end-effector contract</b>: the program cannot '
                                'grasp, release or command the tool yet.')
            return
        g = eef_ops.gripper_of(c)
        dev = self._device()
        n_act = len(eef_ops.actuator_ids(k, g.get('kind')))
        self.status.setText(f"<b>Device '{k.get('id') or 'main'}'</b> · {g.get('kind')} · "
                            f"{n_act} actuator{'s' if n_act != 1 else ''} · "
                            f"{len(dev.sites) if dev else '?'} site(s) · "
                            f"{'single actuator (short form)' if k.get('actuator') is not None else 'multi-actuator'}")
        self.kind.setCurrentText(g.get('kind') or 'suction')
        self.device_id.setText(k.get('id') or 'main')
        self.long_button.setVisible(k.get('actuator') is not None)
        self.sites_box.setVisible(k.get('actuators') is not None)
        self._fill_actuators(k, dev)
        self._fill_sites(k)
        ver = k.get('verification') or {}
        self.require_grasp.setChecked(bool(ver.get('require_grasp_confirmation')))
        self.require_release.setChecked(bool(ver.get('require_release_confirmation')))
        self._fill_observe(dev)
        self._fill_bindings(k)
        self.yaml_view.setPlainText(yaml.safe_dump(k, sort_keys=False, default_flow_style=None))

    def _device(self):
        try:
            from ...model.end_effector import derive_device
            p = self.composer().project
            return derive_device(p.robot.gripper, p.robot.named_states,
                                 stage_topic=p.scene.gripper_cmd_topic or None)
        except Exception:  # noqa: BLE001 - an incomplete contract: Diagnostics say why
            return None

    def _fill_actuators(self, k: dict, dev) -> None:
        c = self.composer()
        ids = eef_ops.actuator_ids(k, eef_ops.gripper_of(c).get('kind'))
        derived = {a.id: a for a in (dev.actuators if dev else [])}
        self.actuators.setRowCount(0)
        written = (k.get('actuators') or {}) if k.get('actuators') is not None else {}
        for i, aid in enumerate(ids):
            spec = written.get(aid) if written else k.get('actuator') or {}
            d = derived.get(aid)
            shown_id = aid
            targets = d.target_names() if d else list((spec.get('targets') or {}))
            joints = (d.joints if d and d.joints else spec.get('joints')) or []
            self.actuators.insertRow(i)
            users = eef_ops.actuator_users(c, shown_id)
            for col, text in enumerate((shown_id, spec.get('command_interface', ''), ', '.join(joints),
                                        ', '.join(targets), ', '.join(users) or '—')):
                self.actuators.setItem(i, col, QTableWidgetItem(text))
        if self.selected_actuator not in ids:
            self.selected_actuator = ids[0] if ids else None
        self._load_actuator()

    def _actuator_spec(self) -> dict:
        k = eef_ops.contract_of(self.composer()) or {}
        if k.get('actuators') is not None:
            return dict((k['actuators'] or {}).get(self.selected_actuator) or {})
        return dict(k.get('actuator') or {})

    def _on_actuator(self) -> None:
        rows = self.actuators.selectionModel().selectedRows() if self.actuators.selectionModel() else []
        if rows:
            self.selected_actuator = self.actuators.item(rows[0].row(), 0).text()
            self._load_actuator()

    def _load_actuator(self) -> None:
        spec = self._actuator_spec()
        self.interface.setCurrentText(spec.get('command_interface') or 'set_bool')
        self.action.setText(spec.get('action') or '')
        self.joints.setText(', '.join(spec.get('joints') or []))
        self.policy.setCurrentText(spec.get('result_policy') or 'succeeded_or_stalled')
        self.max_effort.setValue(float(spec.get('max_effort') or 0.0))
        self.duration.setValue(float(spec.get('duration_s') or 0.0))
        jp = spec.get('joint_positions') or {}
        self.engaged.setText(', '.join(str(x) for x in jp.get('engaged') or []))
        self.released.setText(', '.join(str(x) for x in jp.get('released') or []))
        self._shape_actuator(self.interface.currentText())
        self.targets.setRowCount(0)
        rows = [(n, str(v)) for n, v in (spec.get('targets') or {}).items()]
        if not rows:                         # not written: what the short form derives (ADR-0022 D6), marked so
            dev = self._device()
            act = dev.actuator(self.selected_actuator) if dev is not None else None
            if act is not None and act.targets:
                rows = [(n, f'{v} (derived)') for n, v in act.targets.items()]
        for i, (name, value) in enumerate(rows):
            self.targets.insertRow(i)
            self.targets.setItem(i, 0, QTableWidgetItem(name))
            self.targets.setItem(i, 1, QTableWidgetItem(value))

    def _shape_actuator(self, interface: str) -> None:
        shown = {'set_bool': ('joints', 'engaged', 'released'),
                 'gripper_command': ('action', 'joints', 'policy', 'max_effort'),
                 'follow_joint_trajectory': ('action', 'joints', 'duration')}.get(interface, ())
        for key, (lbl, w) in self.actuator_rows.items():
            lbl.setVisible(key in shown)
            w.setVisible(key in shown)

    def _on_target(self) -> None:
        rows = self.targets.selectionModel().selectedRows() if self.targets.selectionModel() else []
        if rows:
            self.target_name.setText(self.targets.item(rows[0].row(), 0).text())
            self.target_value.setText(self.targets.item(rows[0].row(), 1).text().replace(' (derived)', '').strip('[]'))

    def _fill_sites(self, k: dict) -> None:
        c = self.composer()
        self.sites.setRowCount(0)
        for i, (sid, spec) in enumerate((k.get('sites') or {}).items()):
            spec = spec or {}
            self.sites.insertRow(i)
            for col, text in enumerate((sid, spec.get('tcp') or '', spec.get('grip') or '', spec.get('engage') or '',
                                        spec.get('release') or '', ', '.join(eef_ops.site_users(c, sid)) or '—')):
                self.sites.setItem(i, col, QTableWidgetItem(text))
        current = self.site_grip.currentText()
        self.site_grip.clear()
        self.site_grip.addItems([a for a in eef_ops.actuator_ids(k, eef_ops.gripper_of(c).get('kind')) if a])
        if current:
            self.site_grip.setCurrentText(current)

    def _on_site(self) -> None:
        rows = self.sites.selectionModel().selectedRows() if self.sites.selectionModel() else []
        if rows:
            r = rows[0].row()
            self.site_id.setText(self.sites.item(r, 0).text())
            self.site_tcp.setText(self.sites.item(r, 1).text())
            self.site_grip.setCurrentText(self.sites.item(r, 2).text())
            self.site_engage.setText(self.sites.item(r, 3).text())
            self.site_release.setText(self.sites.item(r, 4).text())

    def _fill_observe(self, dev) -> None:
        if dev is None:
            self.observe.setText('')
            return
        modes = [getattr(m, 'value', m) for m in self.composer().project.deployment.modes]
        cells = []
        for m in modes:
            b = next((x for k_, x in dev.bindings.items() if getattr(k_, 'value', k_) == m), None)
            ok = b is not None and 'grasp_detection' in b.provides()
            cells.append(f"{m} {'✔' if ok else '✖'}")
        self.observe.setText('Can observe a grasp on: ' + ' · '.join(cells) + ' — confirming a grasp needs ✔ on '
                             'every deployed backend (Isaac: a stage status topic)')

    def _fill_bindings(self, k: dict) -> None:
        while self.bindings_form.rowCount():
            self.bindings_form.removeRow(0)
        self.binding_rows = {}
        adapters = k.get('adapters') or {}
        for m in [getattr(x, 'value', x) for x in self.composer().project.deployment.modes]:
            current = adapters.get(m)
            token = current.get('adapter') if isinstance(current, dict) else current
            combo = QComboBox()
            combo.addItem('(none chosen)', '')
            for row, info in EEF_ADAPTER_ROWS.items():
                if m in info['modes']:
                    combo.addItem(f"{row} — {info['name']}", row)
            i = combo.findData(token or '')
            combo.setCurrentIndex(max(i, 0))
            # the keys the rows of this backend read (catalog binding_keys): no backend is named here
            keys = list(dict.fromkeys(k_ for r, info in EEF_ADAPTER_ROWS.items() if m in info['modes']
                                      for k_ in info['binding_keys']))
            edits = {}
            row = QHBoxLayout()
            row.addWidget(combo)
            for key in keys:
                e = QLineEdit(str((current or {}).get(key) or '') if isinstance(current, dict) else '')
                e.setPlaceholderText(key.replace('_', ' '))
                edits[key] = e
                row.addWidget(e)
            apply_b = QPushButton('Apply')
            apply_b.clicked.connect(lambda _=False, mode=m: self.apply_binding(mode))
            row.addWidget(apply_b)
            combo.setToolTip('\n'.join(f"{r}: {EEF_ADAPTER_ROWS[r]['evidence_text']}" for r in EEF_ADAPTER_ROWS
                                       if m in EEF_ADAPTER_ROWS[r]['modes']))
            self.bindings_form.addRow(m, row)
            self.binding_rows[m] = (combo, edits)

    # ---- actions -------------------------------------------------------------------------------------------
    def _do(self, op) -> bool:
        """Run an edit; every failure is SHOWN, never raised into Qt (python review M2: a live read with no
        session, a timeout, a malformed value)."""
        try:
            result = op()
        except (ComposerError, ValueError) as exc:
            self.message.setStyleSheet('color:#c62828;')
            self.message.setText(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001 - shown, never raised into Qt
            self.message.setStyleSheet('color:#c62828;')
            self.message.setText(f'{type(exc).__name__}: {exc}')
            return False
        self.message.setStyleSheet('')
        self.message.setText(result if isinstance(result, str) else '')
        self.refresh()
        return True

    def suggest(self) -> None:
        from ...composer.eef_suggest import suggest_contract
        try:
            self.suggestion = suggest_contract(self.composer().project)
        except Exception as exc:  # noqa: BLE001 - shown
            self.suggest_view.setPlainText(f'cannot suggest: {exc}')
            self.adopt_button.setEnabled(False)
            return
        self.suggest_view.setPlainText(self.suggestion.summary() + '\n\nThe contract:\n'
                                       + yaml.safe_dump(self.suggestion.contract, sort_keys=False))
        self.adopt_button.setEnabled(True)

    def adopt(self) -> bool:
        s = self.suggestion
        if s is None:
            return False

        def op():
            removed = eef_ops.adopt_contract(self.composer(), s.kind, s.contract)
            return ('Adopted' + (f"; removed the legacy bridge(s) {', '.join(removed)} (the adapters replace "
                                 'them)' if removed else '') + (f' -- still to decide: {len(s.decide)} item(s)'
                                                                if s.decide else ''))
        ok = self._do(op)
        if ok:
            self.suggestion = None
            self.adopt_button.setEnabled(False)
        return ok

    @staticmethod
    def _done_with(ok: bool, *fields) -> bool:
        """After a rename / delete, the fields naming the OLD element are cleared (python review S5), so a later
        Set does not re-create it."""
        if ok:
            for f in fields:
                f.clear()
        return ok

    def add_actuator(self) -> bool:
        aid = self.new_actuator.text().strip()
        if not aid:
            return self._do(lambda: _fail('name the new actuator first'))
        ok = self._do(lambda: eef_ops.add_actuator(self.composer(), aid, {'command_interface': 'set_bool'}))
        if ok:
            self.selected_actuator = aid
            self.new_actuator.clear()
            self.refresh()
        return ok

    def rename_actuator(self) -> bool:
        old, new = self.selected_actuator or '', self.new_actuator.text().strip()
        ok = self._do(lambda: eef_ops.rename_actuator(self.composer(), old, new))
        if ok:
            self.selected_actuator = new                      # the renamed one stays selected (S5)
            self.new_actuator.clear()
            self.refresh()
        return ok

    def actuator_spec_from_form(self) -> dict:
        spec = self._actuator_spec()
        interface = self.interface.currentText()
        if spec.get('command_interface') != interface:
            spec = {k: v for k, v in spec.items() if k in ('id', 'targets') and interface != 'set_bool'}
        spec['command_interface'] = interface
        shown = {'set_bool': ('joints', 'engaged', 'released'),
                 'gripper_command': ('action', 'joints', 'policy', 'max_effort'),
                 'follow_joint_trajectory': ('action', 'joints', 'duration')}[interface]
        values = {'action': self.action.text().strip() or None,
                  'joints': _names(self.joints.text()) or None,
                  'result_policy': self.policy.currentText() if 'policy' in shown else None,
                  'max_effort': self.max_effort.value() or None,
                  'duration_s': self.duration.value() or None}
        for key, field_ in (('action', 'action'), ('joints', 'joints'), ('policy', 'result_policy'),
                            ('max_effort', 'max_effort'), ('duration', 'duration_s')):
            if key in shown:
                if values[field_] is None:
                    spec.pop(field_, None)
                else:
                    spec[field_] = values[field_]
            else:
                spec.pop(field_, None)
        if interface == 'set_bool' and (self.engaged.text().strip() or self.released.text().strip()):
            spec['joint_positions'] = {'engaged': _numbers(self.engaged.text()),
                                       'released': _numbers(self.released.text())}
        elif interface != 'set_bool' or not self.engaged.text().strip():
            spec.pop('joint_positions', None)
        if interface == 'set_bool':
            spec.pop('targets', None)
        return spec

    def apply_actuator(self) -> bool:
        aid = self.selected_actuator or ''
        return self._do(lambda: eef_ops.set_actuator(self.composer(), aid, self.actuator_spec_from_form()))

    def set_target(self) -> bool:
        name = self.target_name.text().strip()
        text = self.target_value.text().strip()

        def op():
            spec = self._actuator_spec()
            if not name or not text:
                raise ValueError('name the target and give its value(s)')
            values = _numbers(text)
            if not values:
                raise ValueError(f'{text!r} has no number')
            value = values[0] if spec.get('command_interface') == 'gripper_command' else values
            eef_ops.set_target(self.composer(), self.selected_actuator or '', name, value)
        return self._do(op)

    def capture_target(self) -> bool:
        """The selected actuator's joints as they are now, into the named target (observed, never commanded)."""
        name = self.target_name.text().strip()

        def op():
            spec = self._actuator_spec()
            dev = self._device()
            act = dev.actuator(self.selected_actuator) if dev is not None else None
            joints = list((act.joints if act is not None else None) or spec.get('joints') or [])
            if not name or not joints:
                raise ValueError('name the target, and give the actuator its joints, first')
            if self._live is None:
                raise ValueError('no live session')
            values = self._live().current_joint_values(joints)
            missing = [j for j in joints if j not in values]
            if missing:
                raise ValueError(f"the joint states lack {', '.join(missing)}")
            vals = [values[j] for j in joints]
            eef_ops.set_target(self.composer(), self.selected_actuator or '', name,
                               vals[0] if spec.get('command_interface') == 'gripper_command' else vals)
            return f"'{name}' captured: {', '.join(f'{v:.4f}' for v in vals)}"
        return self._do(op)

    def check_live(self) -> Optional[list]:
        from ...livesession.eef_check import eef_checks
        try:
            if self._live is None:
                raise RuntimeError('no live session')
            lines = eef_checks(self._live(), self.composer().project)
        except Exception as exc:  # noqa: BLE001 - shown, never raised into Qt
            self.live_result.setText(f'⚠ cannot check: {exc} — start the bundle (Step 8) first')
            return None
        marks = {True: '✔', False: '✖', None: '?'}
        self.live_result.setText('\n'.join(f'{marks[c.ok]} {c.name}: {c.text}' for c in lines))
        return lines

    def set_site(self) -> bool:
        sid = self.site_id.text().strip()
        spec = {key: val for key, val in (('tcp', self.site_tcp.text().strip()),
                                          ('grip', self.site_grip.currentText().strip()),
                                          ('engage', self.site_engage.text().strip()),
                                          ('release', self.site_release.text().strip())) if val}
        if not sid:
            return self._do(lambda: _fail('name the site first'))
        return self._do(lambda: eef_ops.set_site(self.composer(), sid, spec))

    def apply_binding(self, mode: str) -> bool:
        combo, edits = self.binding_rows[mode]
        token = combo.currentData()
        keys = {k: e.text().strip() for k, e in edits.items()
                if e.text().strip() and token and k in EEF_ADAPTER_ROWS[token]['binding_keys']}
        if not token:
            binding = None
        elif keys:
            binding = {'adapter': token, **keys}
        else:
            binding = token
        return self._do(lambda: eef_ops.set_binding(self.composer(), mode, binding))


__all__ = ['EndEffectorPanel']
