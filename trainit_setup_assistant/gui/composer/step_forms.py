"""The step inspectors for process, camera and tool work (vNext I0.8, I0_DESIGN.md §17): M3 scope only.

- **Process.** ``process_on`` / ``process_off`` / ``set_value`` / ``external`` act on a LOGICAL channel
  (an id of the application, never a topic or a PLC symbol). ``capture`` samples a detector.
- **Detect.** It samples a detector's detections or a producer's candidates into a named candidate set.
- **Grasp / release.** These are the M3 manipulation contract: at a taught waypoint, or (grasp) from a
  candidate set with a reference orientation. Also: eef / site, standoff, profile, approach profile,
  retract.
- **Tool.** It actuates an actuator to a named target, or verifies held / released.

Each form shows the fields of its kind. It writes back only those, plus any key it does not show,
which is kept as written. The canonical model validates every Apply, and the validator judges
references.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional

from python_qt_binding.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QLabel, QLineEdit,
                                         QSpinBox, QWidget)

PROCESS_KINDS = ('process_on', 'process_off', 'set_value', 'capture', 'external')


def _combo(items, current='', editable=False) -> QComboBox:
    c = QComboBox()
    c.setEditable(editable)
    fill(c, items, current)
    return c


def fill(c: QComboBox, items, current) -> None:
    c.clear()
    c.addItems([str(i) for i in items])
    if current and c.findText(str(current)) < 0:
        c.addItem(str(current))
    c.setCurrentText(str(current or ''))


def _mm(lo=0.0, hi=5000.0) -> QDoubleSpinBox:
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(2)
    s.setSuffix(' mm')
    return s


def context(doc: dict, project) -> dict:
    """What the forms offer in their pick-lists, from the canonical project: waypoints, profiles, the
    perception section's detectors, the channels already used, the candidate sets the detect steps
    write, and the end-effector contract's devices, actuators, targets and sites."""
    from ...composer.overview import process_channels
    from ...composer.ops import step_kind, walk
    out = {'waypoints': [w['name'] for w in doc.get('waypoints') or []],
           'profiles': list(doc.get('motion_profiles') or {}),
           'detectors': [d.name for d in (project.perception.detectors if project.perception else [])],
           'channels': process_channels(project), 'sets': [], 'eef': [], 'actuators': {}, 'sites': []}
    for t in doc.get('tasks', []):
        for d in t.get('inputs') or []:
            if d.get('type') == 'candidates' and d['name'] not in out['sets']:
                out['sets'].append(d['name'])
        for _label, step in walk(t.get('steps') or []):
            if step_kind(step) == 'detect':
                name = step['detect'].get('out', 'candidates')
                if name not in out['sets']:
                    out['sets'].append(name)
    try:
        from ...model.end_effector import derive_device
        dev = derive_device(project.robot.gripper, project.robot.named_states)
    except Exception:  # noqa: BLE001 - a broken contract is the validator's to report
        dev = None
    if dev is not None:
        out['eef'] = [dev.id]
        # the targets a 'tool actuate' may name: every named target but a grip's engage / release, which
        # grasp / release / close / open command (ADR-0022 R11); a set_bool actuator's are engage / release
        grip = {(s.grip, t) for s in dev.sites for t in (s.engage, s.release)}
        out['actuators'] = {a.id: [t for t in a.target_names() if (a.id, t) not in grip] for a in dev.actuators}
        out['sites'] = [s.id for s in dev.sites]
    return out


class StepForm(QWidget):
    kind = ''

    def load(self, step: dict, ctx: dict) -> None:
        self._original = copy.deepcopy(step)
        self.fill(step[self.kind], ctx)

    def spec(self) -> dict:
        body = copy.deepcopy(self._original[self.kind])
        self.write(body)
        return {self.kind: body}

    def fill(self, body, ctx):  # pragma: no cover - abstract
        raise NotImplementedError

    def write(self, body):  # pragma: no cover - abstract
        raise NotImplementedError

    @staticmethod
    def _set(body: dict, key: str, value) -> None:
        if value in (None, '', 0, 0.0):
            body.pop(key, None)
        else:
            body[key] = value


class ProcessForm(StepForm):
    kind = 'process'

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QFormLayout(self)
        self.pkind = _combo(PROCESS_KINDS)
        self.pkind.currentTextChanged.connect(self._shape)
        f.addRow('What', self.pkind)
        self.channel = _combo([], editable=True)
        self.channel.setToolTip('A LOGICAL id of your process (dispenser, inspection_light): the deployment '
                                'binds it to its signal; never a ROS topic or a PLC symbol')
        self.value = QDoubleSpinBox()
        self.value.setRange(-1e9, 1e9)
        self.value.setDecimals(4)
        self.detector = _combo([])
        self.out = QLineEdit()
        self.out.setPlaceholderText('captures.<n> (default)')
        self.timeout = QSpinBox()
        self.timeout.setRange(0, 600000)
        self.timeout.setSuffix(' ms')
        self.timeout.setSpecialValueText('default')
        self.rows = {}
        for key, label, w in (('channel', 'Channel', self.channel), ('value', 'Value', self.value),
                              ('detector', 'Camera detector', self.detector), ('out', 'Result key', self.out),
                              ('timeout_ms', 'Timeout', self.timeout)):
            lbl = QLabel(label)
            f.addRow(lbl, w)
            self.rows[key] = (lbl, w)
        self._shape(self.pkind.currentText())

    def _shape(self, kind: str) -> None:
        show = {'channel': kind in ('process_on', 'process_off', 'set_value', 'external'),
                'value': kind == 'set_value', 'detector': kind == 'capture', 'out': kind == 'capture',
                'timeout_ms': kind == 'capture'}
        for key, (lbl, w) in self.rows.items():
            lbl.setVisible(show[key])
            w.setVisible(show[key])

    def fill(self, body, ctx):
        self.pkind.setCurrentText(body.get('kind', 'process_on'))
        fill(self.channel, ctx['channels'], body.get('channel') or '')
        self.value.setValue(float(body.get('value') or 0.0))
        fill(self.detector, ctx['detectors'], body.get('detector') or '')
        self.out.setText(body.get('out') or '')
        self.timeout.setValue(int(body.get('timeout_ms') or 0))
        self._shown = {'value': self.value.value(), 'timeout_ms': self.timeout.value()}
        self._shape(self.pkind.currentText())

    def write(self, body):
        kind = self.pkind.currentText()
        if body.get('kind') != kind:                       # another kind: only its own fields remain
            for key in ('channel', 'value', 'detector', 'out', 'class_id', 'timeout_ms', 'ms'):
                body.pop(key, None)
        body['kind'] = kind
        if kind in ('process_on', 'process_off', 'set_value', 'external'):
            body['channel'] = self.channel.currentText().strip()
        if kind == 'set_value' and ('value' not in body or abs(self.value.value() - self._shown['value']) > 1e-12):
            body['value'] = self.value.value()           # an untouched value stays exactly as stored
        if kind == 'capture':
            body['detector'] = self.detector.currentText()
            self._set(body, 'out', self.out.text().strip())
            if self.timeout.value() != self._shown['timeout_ms'] or 'timeout_ms' not in body:
                self._set(body, 'timeout_ms', self.timeout.value())


class DetectForm(StepForm):
    kind = 'detect'

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QFormLayout(self)
        self.detector = _combo([])
        f.addRow('Detector', self.detector)
        self.source = _combo(['detections', 'candidates'])
        self.source.setToolTip('detections: a detector\'s objects (projected to candidates); candidates: a '
                               'grasp producer\'s candidates')
        f.addRow('Source', self.source)
        self.out = QLineEdit()
        f.addRow('Candidate set (out)', self.out)
        self.keep = QSpinBox()
        self.keep.setRange(1, 1000)
        f.addRow('Keep (best N)', self.keep)
        self.use_approach = QCheckBox('fallback approach direction (base frame)')
        f.addRow(self.use_approach)
        self.approach = [QDoubleSpinBox() for _ in range(3)]
        for s in self.approach:
            s.setRange(-1.0, 1.0)
            s.setDecimals(3)
        row = QWidget()
        from python_qt_binding.QtWidgets import QHBoxLayout
        h = QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0)
        for s in self.approach:
            h.addWidget(s)
        f.addRow('Approach', row)

    def fill(self, body, ctx):
        fill(self.detector, ctx['detectors'], body.get('detector') or '')
        self.source.setCurrentText(body.get('source') or 'detections')
        self.out.setText(body.get('out') or 'candidates')
        self.keep.setValue(int(body.get('keep') or 1))
        appr = body.get('approach')
        self.use_approach.setChecked(appr is not None)
        for s, v in zip(self.approach, appr or [0.0, 0.0, -1.0]):
            s.setValue(float(v))

    def write(self, body):
        body['detector'] = self.detector.currentText()
        body['source'] = self.source.currentText()
        # a default the file does not state stays unstated (out: candidates, keep: 1)
        for key, value, default in (('out', self.out.text().strip() or 'candidates', 'candidates'),
                                    ('keep', self.keep.value(), 1)):
            if key in body or value != default:
                body[key] = value
        if self.use_approach.isChecked():
            new = [s.value() for s in self.approach]
            old = body.get('approach')
            if old is None or any(abs(float(a) - b) > 5e-4 for a, b in zip(old, new)):
                body['approach'] = new
        else:
            body.pop('approach', None)


class _IdiomForm(StepForm):
    """The parts grasp and release share: eef / site, standoff, profile, approach profile, retract."""

    def _idiom_rows(self, f: QFormLayout) -> None:
        self.eef = _combo([])
        self.site = _combo([])
        f.addRow('End effector', self.eef)
        f.addRow('Site', self.site)
        self.standoff = _mm()
        f.addRow('Standoff', self.standoff)
        self.profile = _combo([])
        self.approach_profile = _combo([])
        f.addRow('Profile (to the standoff)', self.profile)
        f.addRow('Approach profile (the last LIN)', self.approach_profile)
        self.retract_mode = _combo(['dz', 'along_approach'])
        self.retract_mode.setToolTip('dz: up the target frame\'s +Z; along_approach: back along the approach')
        self.retract = _mm()
        f.addRow('Retract', self.retract_mode)
        f.addRow('Retract distance', self.retract)

    def _fill_idiom(self, body, ctx):
        fill(self.eef, [''] + ctx['eef'], body.get('eef') or '')
        fill(self.site, [''] + ctx['sites'], body.get('site') or '')
        self._standoff = round(float(body.get('standoff_m') or 0.0) * 1000.0, 2)
        self.standoff.setValue(self._standoff)
        self._standoff = self.standoff.value()
        fill(self.profile, ctx['profiles'], body.get('profile') or '')
        fill(self.approach_profile, ctx['profiles'], body.get('approach_profile') or '')
        r = body.get('retract') or {'dz': 0.1}
        mode = 'along_approach' if r.get('along_approach') is not None else 'dz'
        self.retract_mode.setCurrentText(mode)
        self.retract.setValue(round(float(r.get(mode) or 0.0) * 1000.0, 2))
        self._retract = (mode, self.retract.value())

    def _write_idiom(self, body):
        self._set(body, 'eef', self.eef.currentText())
        self._set(body, 'site', self.site.currentText())
        if abs(self.standoff.value() - self._standoff) > 1e-9 or 'standoff_m' not in body:
            if self.standoff.value() or self.kind == 'release':
                body['standoff_m'] = round(self.standoff.value() / 1000.0, 6)
            else:
                body.pop('standoff_m', None)
        body['profile'] = self.profile.currentText()
        body['approach_profile'] = self.approach_profile.currentText()
        mode, value = self.retract_mode.currentText(), self.retract.value()
        if (mode, value) != self._retract:
            body['retract'] = {mode: round(value / 1000.0, 6)}


class GraspForm(_IdiomForm):
    kind = 'grasp'

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QFormLayout(self)
        self.mode = _combo(['at a taught waypoint', 'from detected candidates'])
        self.mode.currentIndexChanged.connect(self._shape)
        f.addRow('Grasp', self.mode)
        self.at = _combo([])
        self.candidates = _combo([], editable=True)
        self.index = QSpinBox()
        self.index.setRange(0, 1000)
        self.orientation_from = _combo([])
        self.orientation_from.setToolTip('The TCP orientation: this taught waypoint\'s, tilted to the '
                                         'candidate\'s approach (align_approach)')
        self.rows = {}
        for key, label, w in (('at', 'At', self.at), ('candidates', 'Candidate set', self.candidates),
                              ('index', 'Candidate (0 = best)', self.index),
                              ('orientation_from', 'Reference orientation', self.orientation_from)):
            lbl = QLabel(label)
            f.addRow(lbl, w)
            self.rows[key] = (lbl, w)
        self._idiom_rows(f)
        self._shape(0)

    def _shape(self, i: int) -> None:
        taught = i == 0
        for key, (lbl, w) in self.rows.items():
            on = taught if key == 'at' else not taught
            lbl.setVisible(on)
            w.setVisible(on)

    def fill(self, body, ctx):
        self.mode.setCurrentIndex(1 if body.get('candidates') else 0)
        fill(self.at, ctx['waypoints'], body.get('at') or '')
        fill(self.candidates, ctx['sets'], body.get('candidates') or '')
        self.index.setValue(int(body.get('index') or 0))
        fill(self.orientation_from, ctx['waypoints'], body.get('orientation_from') or '')
        self._fill_idiom(body, ctx)
        self._shape(self.mode.currentIndex())

    def write(self, body):
        if self.mode.currentIndex() == 0:
            for key in ('candidates', 'index', 'orientation', 'orientation_from'):
                body.pop(key, None)
            body['at'] = self.at.currentText()
        else:
            body.pop('at', None)
            body['candidates'] = self.candidates.currentText().strip()
            if 'index' in body or self.index.value() != 0:      # a stated index 0 stays stated
                body['index'] = self.index.value()
            body['orientation_from'] = self.orientation_from.currentText()
        self._write_idiom(body)


class ReleaseForm(_IdiomForm):
    kind = 'release'

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QFormLayout(self)
        self.at = _combo([])
        f.addRow('At', self.at)
        self._idiom_rows(f)

    def fill(self, body, ctx):
        fill(self.at, ctx['waypoints'], body.get('at') or '')
        self._fill_idiom(body, ctx)

    def write(self, body):
        body['at'] = self.at.currentText()
        self._write_idiom(body)


TOOL_MODES = ('close', 'open', 'actuate', 'verify')


class ToolForm(StepForm):
    kind = 'tool'

    def __init__(self, parent=None):
        super().__init__(parent)
        f = QFormLayout(self)
        self.mode = _combo(list(TOOL_MODES))
        self.mode.currentTextChanged.connect(self._shape)
        f.addRow('Tool operation', self.mode)
        self.actuator = _combo([])
        self.actuator.currentTextChanged.connect(self._targets)
        self.target = _combo([])
        self.expect = _combo(['held', 'released'])
        self.site = _combo([])
        self.rows = {}
        for key, label, w in (('actuator', 'Actuator', self.actuator), ('target', 'Named target', self.target),
                              ('expect', 'Expect', self.expect), ('site', 'Site', self.site)):
            lbl = QLabel(label)
            f.addRow(lbl, w)
            self.rows[key] = (lbl, w)
        self._ctx: Dict = {}
        self._shape('actuate')

    def _shape(self, mode: str) -> None:
        shown = {'actuate': ('actuator', 'target'), 'verify': ('expect', 'site'),
                 'close': ('site',), 'open': ('site',)}.get(mode, ())
        for key, (lbl, w) in self.rows.items():
            lbl.setVisible(key in shown)
            w.setVisible(key in shown)

    def _targets(self, actuator: str) -> None:
        fill(self.target, (self._ctx.get('actuators') or {}).get(actuator, []), self.target.currentText())

    def fill(self, body, ctx):
        self._ctx = ctx
        mode = next((m for m in TOOL_MODES if m in body), 'actuate')
        self.mode.setCurrentText(mode)
        a = body.get('actuate') or {}
        fill(self.actuator, list(ctx['actuators']), a.get('actuator') or '')
        fill(self.target, ctx['actuators'].get(a.get('actuator'), []), a.get('target') or '')
        v = body.get(mode) if mode in ('verify', 'close', 'open') else {}
        v = v or {}
        self.expect.setCurrentText((body.get('verify') or {}).get('expect') or 'held')
        fill(self.site, [''] + ctx['sites'], v.get('site') or '')
        self._shape(mode)

    def write(self, body):
        mode = self.mode.currentText()
        if mode == 'actuate':
            a = copy.deepcopy(body.get('actuate') or {})
            a['actuator'], a['target'] = self.actuator.currentText(), self.target.currentText()
            body.clear()
            body['actuate'] = a
            return
        v = copy.deepcopy(body.get(mode) or {})
        if mode == 'verify':
            v['expect'] = self.expect.currentText()
        self._set(v, 'site', self.site.currentText())
        body.clear()
        body[mode] = v


FORMS = {'process': ProcessForm, 'detect': DetectForm, 'grasp': GraspForm, 'release': ReleaseForm,
         'tool': ToolForm}


def default_step(kind: str, ctx: dict, project) -> dict:
    """A NEW step of ``kind`` with values the user then sets; raises ``ValueError`` naming what is missing
    first."""
    profiles = ctx['profiles']

    def prof(prefer):
        return prefer if prefer in profiles else (profiles[0] if profiles else None)
    if kind == 'capture':
        if not ctx['detectors']:
            raise ValueError('configure a camera detector first (Step 5 — Perception)')
        return {'process': {'kind': 'capture', 'detector': ctx['detectors'][0]}}
    if kind == 'detect':
        if not ctx['detectors']:
            raise ValueError('configure a detector first (Step 5 — Perception)')
        return {'detect': {'detector': ctx['detectors'][0], 'source': 'detections', 'out': 'candidates',
                           'keep': 1, 'approach': [0.0, 0.0, -1.0]}}
    if kind in ('grasp', 'release'):
        if not ctx['waypoints'] or not profiles:
            raise ValueError('teach a waypoint and define a profile first')
        body = {'at': ctx['waypoints'][0], 'standoff_m': 0.05, 'profile': prof('transit'),
                'approach_profile': prof('approach'), 'retract': {'dz': 0.08}}
        return {kind: body}
    if kind in ('tool', 'tool_close', 'tool_open'):
        if not ctx['actuators']:
            raise ValueError('the robot has no end-effector contract: set it up in the End effector section '
                             '(Step 7, left)')
        if kind in ('tool_close', 'tool_open'):
            # vNext I0 (D-E1): close / open a grasp site where the robot is; a multi-site tool names one
            body = {'site': ctx['sites'][0]} if len(ctx['sites']) > 1 else {}
            return {'tool': {kind[len('tool_'):]: body}}
        named = [(a, t) for a, ts in ctx['actuators'].items() for t in ts]
        if named:                           # an actuator with named targets: drive it to one
            return {'tool': {'actuate': {'actuator': named[0][0], 'target': named[0][1]}}}
        return {'tool': {'verify': {'expect': 'held'}}}
    raise ValueError(kind)


__all__ = ['FORMS', 'PROCESS_KINDS', 'context', 'default_step', 'fill']
