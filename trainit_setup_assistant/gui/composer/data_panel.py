"""The Composer's Data section (TrainIt D0.8, D0_8_DESIGN.md §§1-9): Capture, Synthetic, Datasets.

TSA is the COCKPIT here, never the runtime (ADR-0029 Decision 7 + Clarification D0.8). This panel:

- AUTHORS the project's data collection (the camera's id / profile, the dataset, the triggers, the synthetic
  recipe) through ``Composer.edit_project``: one undoable, validated edit each; the ONE validator reports in the
  Diagnostics strip (R-D0-xx, located here);
- OBSERVES the live camera (the wizard's in-process preview subscription) and the recorder (its diagnostics),
  and CALLS the recorder's three Trigger services (``livesession.data_recorder``) -- the recorder itself is the
  bundle's bring-up, or a launched, copyable command;
- RUNS ``python3 -m trainit_data --json`` for every offline operation (sessions, datasets, versions, labels, YOLO
  export, recover) in a worker thread, and LAUNCHES a synthetic provider as an external process whose progress
  lines it reads (``composer.data_tools``).

It never writes the store, never builds a version, never exports, never imports the data stack or a simulator.
A control whose precondition does not hold is disabled or absent, with the reason in words; nothing is shown that
cannot work. Normal mode speaks product words; "Advanced" adds ids, topics, hashes and the exact commands.
"""

from __future__ import annotations

import json
import queue
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QColor, QImage, QPainter, QPen, QPixmap
from python_qt_binding.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                                         QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
                                         QHeaderView, QLabel, QLineEdit, QListWidget, QMessageBox,
                                         QPlainTextEdit, QProgressBar, QPushButton, QScrollArea, QSpinBox,
                                         QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from ...composer import data_tools as D
from ...composer.ops import ComposerError

OK, ERROR, WARN, HINT = '#2e7d32', '#c62828', '#8a6d3b', '#555'
# how long TSA waits for the recorder's answer (a capture waits for its frames; a close seals the session)
CALL_TIMEOUT_S = {'start_session': 15.0, 'capture': 30.0, 'close_session': 120.0}
# after a successful reply, the session buttons wait for the recorder's 1 Hz diagnostics to show the new state
STATE_SETTLE_S = 2.5
FRAMES_SILENT_S = 3.0          # a camera stream with no new frame for this long is reported silent
INSPECT_CAPTURES_READ = 5000   # Inspect reads at most this many capture records of one session
INSPECT_PROBLEMS_SHOWN = 20    # and lists at most this many not-complete ones
SYNC_FAILURES_TEXT = ('{n} capture(s) with streams missing or not taken at the same moment: kept as '
                      'Partial, with the reason (Datasets -> Inspect). Partial captures stay out of a '
                      'version unless you include them.')
KIND_LABELS = {'manual': 'Manual (Capture button)', 'interval': 'Every N milliseconds',
               'event': 'When a process channel changes', 'application_state': 'When the application reaches'}
TRANSITIONS = {'enter': 'starts', 'exit': 'ends', 'succeeded': 'succeeds', 'failed': 'fails'}
CLASSIFICATIONS = ('internal', 'customer_restricted', 'public_approved')


def _hint(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f'color:{HINT};')
    return label


def _say(label: QLabel, text: str, colour: str = HINT) -> None:
    label.setText(text)
    label.setStyleSheet(f'color:{colour};')


def describe_trigger(t: dict) -> str:
    """A trigger in product words (the table's Details column)."""
    kind = t.get('kind')
    if kind == 'manual':
        return 'when you press Capture'
    if kind == 'interval':
        limit = f', at most {t["max_captures"]} captures' if t.get('max_captures') else ''
        return f'every {t.get("period_ms")} ms{limit}'
    if kind == 'event':
        edge = {'rising': 'turns on', 'falling': 'turns off', 'any': 'changes'}.get(t.get('edge'), t.get('edge'))
        return f'when process channel "{t.get("process_channel")}" {edge}'
    if kind == 'application_state':
        what = f'step "{t["step"]}" of task "{t.get("task")}"' if t.get('step') else f'task "{t.get("task")}"'
        attempt = {'pre_grasp': ' (before the grasp)', 'post_grasp': ' (after the grasp)'}.get(t.get('pick_attempt'), '')
        return f'when {what} {TRANSITIONS.get(t.get("transition"), t.get("transition"))}{attempt}'
    return str(t)


def _np_image(arr) -> QImage:
    import numpy as np
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = np.ascontiguousarray(np.stack([arr] * 3, axis=-1))
    h, w, _ = arr.shape
    return QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888).copy()


def depth_image(depth_m) -> QImage:
    """Depth (metres) as a grey image for a person: near = bright; holes (0 / NaN / inf) black."""
    import numpy as np
    d = np.asarray(depth_m, dtype=np.float32)
    valid = np.isfinite(d) & (d > 0)
    out = np.zeros(d.shape, np.uint8)
    if valid.any():
        lo, hi = float(np.percentile(d[valid], 2)), float(np.percentile(d[valid], 98))
        span = max(hi - lo, 1e-6)
        out[valid] = (255 - np.clip((d[valid] - lo) / span, 0, 1) * 215).astype(np.uint8)
    return _np_image(out)


def mask_image(path: str) -> Optional[QImage]:
    """A 16-bit instance mask as distinct colours (0 = black)."""
    import numpy as np
    img = QImage(path)
    if img.isNull():
        return None
    img = img.convertToFormat(QImage.Format_Grayscale16)
    h, w = img.height(), img.width()
    ptr = img.constBits()
    ptr.setsize(img.sizeInBytes()) if hasattr(ptr, 'setsize') else None
    ids = np.frombuffer(bytes(ptr), np.uint16).reshape(h, img.bytesPerLine() // 2)[:, :w]
    palette = np.array([[0, 0, 0]] + [[(37 * i) % 200 + 55, (91 * i) % 200 + 55, (53 * i) % 200 + 55]
                                      for i in range(1, 256)], np.uint8)
    return _np_image(palette[ids % 256])


PREVIEW_PREFIX = 'tsa_synthetic_preview_'


def _decimals(value) -> int:
    text = repr(float(value))
    return 0 if 'e' in text else len(text.split('.')[1].rstrip('0'))


def is_preview_scratch(path) -> bool:
    """A directory TSA made for a preview: directly in the system temp dir, with TSA's prefix. Nothing else is
    ever removed by the panel (never the user's store)."""
    path = Path(path).resolve()
    return path.name.startswith(PREVIEW_PREFIX) and path.parent == Path(tempfile.gettempdir()).resolve()


def _payload(root: Path, relative: str) -> Optional[Path]:
    """A store-relative payload path under ``root`` (anything else is not shown)."""
    path = (root / relative).resolve()
    return path if path.is_relative_to(root.resolve()) else None


def _pixmap(view: QLabel, image: Optional[QImage]) -> None:
    if image is None or image.isNull():
        view.setPixmap(QPixmap())
        return
    view.setPixmap(QPixmap.fromImage(image).scaled(max(view.width(), 160), max(view.height(), 120),
                                                   Qt.KeepAspectRatio, Qt.SmoothTransformation))


class Jobs:
    """Background work for the panel: ``run(fn, done)`` runs ``fn()`` on a worker thread and calls
    ``done(result)`` in the GUI thread (a timer drains the queue). A failing ``fn`` hands its exception."""

    def __init__(self, owner: QWidget):
        self._queue: 'queue.Queue' = queue.Queue()
        self.busy = 0
        self.errors: List[str] = []
        self.timer = QTimer(owner)
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.drain)
        self.timer.start()

    def run(self, fn: Callable[[], Any], done: Callable[[Any], None]) -> None:
        self.busy += 1

        def work():
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - handed to the GUI, never raised in a thread
                result = exc
            self._queue.put((done, result))
        threading.Thread(target=work, daemon=True).start()

    def drain(self) -> None:
        while True:
            try:
                done, result = self._queue.get_nowait()
            except queue.Empty:
                return
            self.busy -= 1
            try:
                done(result)
            except Exception as exc:  # noqa: BLE001 - a Qt slot must never raise (PyQt aborts)
                self.errors.append(repr(exc))

    def wait(self, timeout_s: float = 30.0) -> None:
        """Tests: until every job has finished and been handed back."""
        deadline = time.monotonic() + timeout_s
        while self.busy and time.monotonic() < deadline:
            self.drain()
            time.sleep(0.02)
        self.drain()


# ---- trigger dialog -----------------------------------------------------------------------------------------

class TriggerDialog(QDialog):
    """Add / edit one trigger. The task and step pickers list ONLY what the program has (canonical step ids);
    the pick-attempt phase is offered only on a grasp step. The validator still judges the result."""

    def __init__(self, project, trigger: Optional[dict] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Recording trigger')
        self.project = project
        from ...generator.data_collection import _tasks
        from ...composer.overview import process_channels
        self.tasks = _tasks(project) or {}
        self.channels = process_channels(project)
        form = QFormLayout(self)
        self.id = QLineEdit((trigger or {}).get('id', ''))
        self.id.setPlaceholderText('a short name, e.g. before_grasp')
        form.addRow('Name', self.id)
        self.kind = QComboBox()
        kinds = ['manual', 'interval'] + (['event'] if self.channels else []) + \
                (['application_state'] if self.tasks else [])
        for k in kinds:
            self.kind.addItem(KIND_LABELS[k], k)
        form.addRow('Capture', self.kind)
        self.period = QSpinBox()
        self.period.setRange(1, 3_600_000)
        self.period.setSuffix(' ms')
        self.period.setValue(2000)
        self.max_captures = QSpinBox()
        self.max_captures.setRange(0, 1_000_000)
        self.max_captures.setSpecialValueText('no limit')
        self.channel = QComboBox()
        self.channel.addItems(self.channels)
        self.edge = QComboBox()
        for value, label in (('rising', 'turns on'), ('falling', 'turns off'), ('any', 'changes')):
            self.edge.addItem(label, value)
        self.task = QComboBox()
        self.task.addItems(list(self.tasks))
        self.step = QComboBox()
        self.transition = QComboBox()
        for value, label in TRANSITIONS.items():
            self.transition.addItem(label, value)
        self.attempt = QComboBox()
        self.rows = {}
        for name, widget, label in (('period', self.period, 'Every'), ('max', self.max_captures, 'At most'),
                                    ('channel', self.channel, 'Process channel'), ('edge', self.edge, 'When it'),
                                    ('task', self.task, 'Task'), ('step', self.step, 'Step'),
                                    ('transition', self.transition, 'When it'),
                                    ('attempt', self.attempt, 'Pick attempt')):
            form.addRow(label, widget)
            self.rows[name] = (form.labelForField(widget), widget)
        self.note = _hint('')
        form.addRow(self.note)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.kind.currentIndexChanged.connect(self._kind_changed)
        self.task.currentIndexChanged.connect(self._task_changed)
        self.step.currentIndexChanged.connect(self._step_changed)
        self.transition.currentIndexChanged.connect(self._step_changed)
        if trigger:
            self._load(trigger)
        self._kind_changed()

    def _load(self, t: dict) -> None:
        i = self.kind.findData(t.get('kind'))
        if i >= 0:
            self.kind.setCurrentIndex(i)
        if t.get('period_ms'):
            self.period.setValue(int(t['period_ms']))
        self.max_captures.setValue(int(t.get('max_captures') or 0))
        if t.get('process_channel') in self.channels:
            self.channel.setCurrentText(t['process_channel'])
        i = self.edge.findData(t.get('edge'))
        if i >= 0:
            self.edge.setCurrentIndex(i)
        if t.get('task') in self.tasks:
            self.task.setCurrentText(t['task'])
        self._task_changed()
        i = self.step.findData(t.get('step'))
        if i >= 0:
            self.step.setCurrentIndex(i)
        i = self.transition.findData(t.get('transition'))
        if i >= 0:
            self.transition.setCurrentIndex(i)
        self._step_changed()
        i = self.attempt.findData(t.get('pick_attempt'))
        if i >= 0:
            self.attempt.setCurrentIndex(i)

    def _show(self, *names) -> None:
        for name, (label, widget) in self.rows.items():
            visible = name in names
            widget.setVisible(visible)
            if label is not None:
                label.setVisible(visible)

    def _kind_changed(self) -> None:
        kind = self.kind.currentData()
        self._show(*{'manual': (), 'interval': ('period', 'max'), 'event': ('channel', 'edge'),
                     'application_state': ('task', 'step', 'transition', 'attempt')}[kind])
        self.note.setText({'manual': 'One capture each time Capture is pressed during a session.',
                           'event': 'Only the process channels the program drives are listed.',
                           'application_state': 'Only the steps that have an id are listed: to name a step here, '
                                                'give it an id in Tasks (select the step, Id, Apply).'}.get(kind, ''))
        if kind == 'application_state':
            self._task_changed()

    def _task_changed(self) -> None:
        steps = self.tasks.get(self.task.currentText(), {})
        self.step.blockSignals(True)
        self.step.clear()
        self.step.addItem('(the whole task)', None)
        for step_id, kind in steps.items():
            self.step.addItem(f'{step_id} ({kind})', step_id)
        self.step.blockSignals(False)
        self._step_changed()

    def _step_changed(self) -> None:
        step = self.step.currentData()
        kind = self.tasks.get(self.task.currentText(), {}).get(step)
        current = self.attempt.currentData()
        self.attempt.clear()
        self.attempt.addItem('no', None)
        if kind == 'grasp':                       # a pick attempt is a grasp step's (and its phase's transitions)
            transition = self.transition.currentData()
            if transition == 'enter':
                self.attempt.addItem('before the grasp', 'pre_grasp')
            else:
                self.attempt.addItem('after the grasp', 'post_grasp')
        i = self.attempt.findData(current)
        self.attempt.setCurrentIndex(max(i, 0))
        self.attempt.setEnabled(self.attempt.count() > 1)

    def value(self) -> dict:
        kind = self.kind.currentData()
        out = {'id': self.id.text().strip(), 'kind': kind}
        if kind == 'interval':
            out['period_ms'] = self.period.value()
            if self.max_captures.value():
                out['max_captures'] = self.max_captures.value()
        elif kind == 'event':
            out.update(process_channel=self.channel.currentText(), edge=self.edge.currentData())
        elif kind == 'application_state':
            out.update(task=self.task.currentText(), transition=self.transition.currentData())
            if self.step.currentData():
                out['step'] = self.step.currentData()
            if self.attempt.currentData():
                out['pick_attempt'] = self.attempt.currentData()
        return out


# ---- the panel ---------------------------------------------------------------------------------------------

class DataPanel(QWidget):
    def __init__(self, ctrl, composer: Callable[[], object], host: Callable[[], object] = lambda: None,
                 parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self._composer = composer
        self._host = host                        # the wizard: launcher, launcher_settings, camera_capture
        self.jobs = Jobs(self)
        self._link = None
        self._link_ns = None
        self._link_error = ''
        self._wanted_dataset: Optional[str] = None
        self._preview_dir: Optional[str] = None
        self._session_after_close: Optional[str] = None
        self._settle_until = 0.0
        self._recorder_starting = False               # "Recorder starting…" until it answers or ends
        self._recorder_tail: List[str] = []           # its last output lines (why it ended)
        self._frames: Dict[str, Tuple[int, float]] = {}  # stream -> (frames received, when that count last grew)
        self.sessions: List[dict] = []
        self.datasets: List[dict] = []
        self.versions: List[int] = []
        self.providers: List[D.Provider] = []
        self.provider_problems: List[str] = []
        self.provider: Optional[D.Provider] = None
        self.param_widgets: Dict[str, list] = {}
        self.last_cli: Optional[D.CliResult] = None
        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        title = QLabel('<b>Data</b> — capture camera data, generate synthetic data, build dataset versions and '
                       'export them')
        title.setWordWrap(True)
        head.addWidget(title, 1)
        self.advanced = QCheckBox('Advanced')
        self.advanced.toggled.connect(lambda _on: self.refresh())
        head.addWidget(self.advanced)
        outer.addLayout(head)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        outer.addWidget(self.message)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs, 1)
        self.tabs.addTab(self._scroll(self._build_capture()), 'Capture')
        self.tabs.addTab(self._scroll(self._build_synthetic()), 'Synthetic')
        self.tabs.addTab(self._scroll(self._build_datasets()), 'Datasets')
        self.tabs.currentChanged.connect(lambda _i: self.refresh())
        self.timer = QTimer(self)
        self.timer.setInterval(250)
        self.timer.timeout.connect(self._tick)
        self.timer.start()

    @staticmethod
    def _scroll(widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    # ---- shared helpers ----------------------------------------------------------------------------------
    def project(self):
        return self.ctrl.project

    def settings(self):
        host = self._host()
        settings = getattr(host, 'launcher_settings', None)
        if settings is None:
            from ...composer.settings import LauncherSettings
            settings = LauncherSettings.load()
        return settings

    def launcher(self):
        return getattr(self._host(), 'launcher', None)

    def root(self) -> str:
        return D.data_root(self.settings())

    def _edit(self, mutate, done: str = '') -> bool:
        try:
            self._composer().edit_project(mutate)
        except ComposerError as exc:
            _say(self.message, str(exc), ERROR)
            return False
        _say(self.message, done, OK)
        self.refresh()
        return True

    def cli(self, args: List[str], done: Callable[[D.CliResult], None], root: Optional[str] = None,
            timeout_s: float = 600.0) -> None:
        """One offline operation in the background; the result (and its command) kept for Advanced."""
        root = root or self.root()

        def finish(result):
            if isinstance(result, Exception):
                result = D.CliResult(argv=D.cli_argv(args, root), exit_code=None, stderr=repr(result))
            self.last_cli = result
            self._show_last_cli()
            done(result)
            self._selection_changed()
        self.jobs.run(lambda: D.run_cli(args, root=root, timeout_s=timeout_s), finish)
        self._selection_changed()

    def _show_last_cli(self) -> None:
        r = self.last_cli
        if r is None:
            return
        text = f'$ {r.display()}\nexit {r.exit_code}\n'
        if r.result is not None or r.error is not None:
            text += json.dumps({'result': r.result, 'error': r.error}, indent=2, sort_keys=True)[:20000]
        if r.stderr.strip():
            text += '\n--- stderr ---\n' + r.stderr[-4000:]
        self.cli_log.setPlainText(text)

    def _data(self) -> Optional[dict]:
        dc = self.project().data_collection if self.project() else None
        if dc is None:
            return None
        from ...model.data_collection import data_collection_as_written
        return data_collection_as_written(dc)

    def _camera(self):
        p = self.project()
        return p.perception.camera if p is not None and p.perception is not None else None

    # ---- Capture -------------------------------------------------------------------------------------------
    def _build_capture(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        # the camera
        box = QGroupBox('Camera')
        form = QFormLayout(box)
        self.camera_state = QLabel('')
        self.camera_state.setWordWrap(True)
        form.addRow(self.camera_state)
        self.camera_id = QLineEdit()
        self.camera_id.setPlaceholderText('e.g. bin_camera')
        form.addRow('Camera name', self.camera_id)
        self.camera_profile = QComboBox()
        form.addRow('Camera model', self.camera_profile)
        self.camera_apply = QPushButton('Apply')
        self.camera_apply.clicked.connect(self.apply_camera)
        form.addRow(self.camera_apply)
        self.camera_advanced = _hint('')
        form.addRow(self.camera_advanced)
        v.addWidget(box)
        # the live preview
        box = QGroupBox('Live view')
        lv = QVBoxLayout(box)
        row = QHBoxLayout()
        self.preview_button = QPushButton('Show live view')
        self.preview_button.setCheckable(True)
        self.preview_button.toggled.connect(self._preview_toggled)
        row.addWidget(self.preview_button)
        self.preview_mode = QComboBox()
        self.preview_mode.addItems(['Colour', 'Depth'])
        row.addWidget(self.preview_mode)
        row.addStretch(1)
        lv.addLayout(row)
        self.preview_state = _hint('Shows what the camera publishes now (the live ROS topics).')
        lv.addWidget(self.preview_state)
        self.preview_view = QLabel()
        self.preview_view.setMinimumSize(320, 240)
        self.preview_view.setAlignment(Qt.AlignCenter)
        self.preview_view.setVisible(False)
        lv.addWidget(self.preview_view)
        v.addWidget(box)
        # the recording setup
        box = QGroupBox('Recording setup')
        rv = QVBoxLayout(box)
        form = QFormLayout()
        self.dataset_id = QLineEdit()
        self.dataset_id.setPlaceholderText('e.g. bin_parts')
        form.addRow('Dataset', self.dataset_id)
        self.dataset_description = QLineEdit()
        form.addRow('Description', self.dataset_description)
        self.classification = QComboBox()
        for c in CLASSIFICATIONS:
            self.classification.addItem(c.replace('_', ' '), c)
        form.addRow('Data rights', self.classification)
        self.sync_timeout = QSpinBox()
        self.sync_timeout.setRange(0, 60_000)
        self.sync_timeout.setSpecialValueText('recorder default')
        self.sync_timeout.setSuffix(' ms')
        self.sync_tolerance = QDoubleSpinBox()
        self.sync_tolerance.setRange(0.0, 1000.0)
        self.sync_tolerance.setSpecialValueText('from the camera frame rate')
        self.sync_tolerance.setSuffix(' ms')
        form.addRow('Wait for frames at most', self.sync_timeout)
        form.addRow('Colour / depth tolerance', self.sync_tolerance)
        self.sync_rows = [self.sync_timeout, self.sync_tolerance]
        self.sync_form = form
        rv.addLayout(form)
        self.setup_dirty = False                       # an edit in the setup fields not saved yet: never overwritten
        for signal in (self.dataset_id.textEdited, self.dataset_description.textEdited,
                       self.classification.activated, self.sync_timeout.valueChanged,
                       self.sync_tolerance.valueChanged):
            signal.connect(self._setup_edited)
        self.setup_save = QPushButton('Save recording setup')
        self.setup_save.clicked.connect(self.save_setup)
        rv.addWidget(self.setup_save)
        rv.addWidget(QLabel('Triggers — when a capture is taken'))
        self.triggers = QTableWidget(0, 2)
        self.triggers.setHorizontalHeaderLabels(['Name', 'When'])
        self.triggers.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.triggers.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        self.triggers.setSelectionBehavior(QTableWidget.SelectRows)
        self.triggers.setEditTriggers(QTableWidget.NoEditTriggers)
        self.triggers.setMaximumHeight(150)
        self.triggers.itemSelectionChanged.connect(self._trigger_selected)
        rv.addWidget(self.triggers)
        row = QHBoxLayout()
        self.trigger_add = QPushButton('Add…')
        self.trigger_edit = QPushButton('Edit…')
        self.trigger_remove = QPushButton('Remove')
        self.trigger_add.clicked.connect(lambda: self.edit_trigger(None))
        self.trigger_edit.clicked.connect(lambda: self.edit_trigger(self.triggers.currentRow()))
        self.trigger_remove.clicked.connect(self.remove_trigger)
        for b in (self.trigger_add, self.trigger_edit, self.trigger_remove):
            row.addWidget(b)
        row.addStretch(1)
        rv.addLayout(row)
        self.setup_state = _hint('')
        rv.addWidget(self.setup_state)
        v.addWidget(box)
        # the session
        box = QGroupBox('Session')
        sv = QVBoxLayout(box)
        self.recorder_state = QLabel('')
        self.recorder_state.setWordWrap(True)
        sv.addWidget(self.recorder_state)
        row = QHBoxLayout()
        self.mode = QComboBox()
        row.addWidget(QLabel('Mode'))
        row.addWidget(self.mode)
        self.recorder_start = QPushButton('Start recorder')
        self.recorder_start.clicked.connect(self.toggle_recorder)
        row.addWidget(self.recorder_start)
        # the recorder runs the GENERATED file (what the cell deploys): before a generation, the way to Step 8
        self.bundle_generate = QPushButton('Generate…')
        self.bundle_generate.setToolTip('Open Step 8 (Generate): the recorder runs the generated recording file')
        self.bundle_generate.clicked.connect(self.open_generate)
        self.bundle_generate.setVisible(False)
        row.addWidget(self.bundle_generate)
        row.addStretch(1)
        sv.addLayout(row)
        row = QHBoxLayout()
        self.record_button = QPushButton('Record Session')
        self.capture_button = QPushButton('Capture')
        self.stop_button = QPushButton('Stop')
        self.record_button.clicked.connect(lambda: self._call('start_session'))
        self.capture_button.clicked.connect(lambda: self._call('capture'))
        self.stop_button.clicked.connect(lambda: self._call('close_session'))
        for b in (self.record_button, self.capture_button, self.stop_button):
            row.addWidget(b)
        self.open_datasets = QPushButton('Open in Datasets')
        self.open_datasets.clicked.connect(self._open_sealed_session)
        self.open_datasets.setVisible(False)
        row.addWidget(self.open_datasets)
        row.addStretch(1)
        sv.addLayout(row)
        self.counters = QLabel('')
        sv.addWidget(self.counters)
        self.session_result = QLabel('')
        self.session_result.setWordWrap(True)
        sv.addWidget(self.session_result)
        self.session_warnings = QLabel('')
        self.session_warnings.setWordWrap(True)
        self.session_warnings.setStyleSheet(f'color:{WARN};')
        sv.addWidget(self.session_warnings)
        v.addWidget(box)
        v.addStretch(1)
        self._pending = []                          # (service, future)
        return w

    def refresh_capture(self) -> None:
        p = self.project()
        camera = self._camera()
        adv = self.advanced.isChecked()
        from ...catalog import camera_profiles
        profiles = sorted(camera_profiles())
        if self.camera_profile.count() != len(profiles) + 1:
            self.camera_profile.clear()
            self.camera_profile.addItem('(choose the camera model)', None)
            for model in profiles:
                self.camera_profile.addItem(model, model)
        has_camera = camera is not None
        for widget in (self.camera_id, self.camera_profile, self.camera_apply, self.preview_button):
            widget.setEnabled(has_camera)
        if not has_camera:
            _say(self.camera_state, 'No camera in this project: add it in Perception (Step 5) first.', WARN)
        else:
            if not self.camera_id.hasFocus():
                self.camera_id.setText(camera.id or '')
            i = self.camera_profile.findData(camera.profile)
            self.camera_profile.setCurrentIndex(max(i, 0))
            ready = camera.id and camera.profile
            _say(self.camera_state, 'Ready for recording.' if ready else
                 'Name the camera and choose its model to record from it.', OK if ready else WARN)
        self.camera_advanced.setVisible(adv and has_camera)
        if has_camera:
            self.camera_advanced.setText(f'colour {camera.rgb_topic} · depth {camera.depth_topic} · info '
                                         f'{camera.camera_info_topic} · frame {camera.optical_frame}')
        data = self._data() or {}
        dataset = data.get('dataset') or {}
        if not self.setup_dirty:
            widgets = (self.dataset_id, self.dataset_description, self.classification, self.sync_timeout,
                       self.sync_tolerance)
            for widget in widgets:
                widget.blockSignals(True)
            self.dataset_id.setText(dataset.get('id', ''))
            self.dataset_description.setText(dataset.get('description', ''))
            self.classification.setCurrentIndex(max(self.classification.findData(dataset.get('classification',
                                                                                             'internal')), 0))
            sync = data.get('sync') or {}
            self.sync_timeout.setValue(int(sync.get('timeout_ms') or 0))
            self.sync_tolerance.setValue(float(sync.get('tolerance_ms') or 0.0))
            for widget in widgets:
                widget.blockSignals(False)
        self.setup_save.setText('Save recording setup *' if self.setup_dirty else 'Save recording setup')
        for widget in self.sync_rows:
            widget.setVisible(adv)
            label = self.sync_form.labelForField(widget)
            if label is not None:
                label.setVisible(adv)
        triggers = data.get('triggers') or []
        self.triggers.setRowCount(len(triggers))
        for i, t in enumerate(triggers):
            self.triggers.setItem(i, 0, QTableWidgetItem(t['id']))
            self.triggers.setItem(i, 1, QTableWidgetItem(describe_trigger(t)))
        has_data = bool(data)
        self.trigger_add.setEnabled(has_data and has_camera)
        selected = 0 <= self.triggers.currentRow() < len(triggers)
        self.trigger_edit.setEnabled(selected)
        self.trigger_remove.setEnabled(selected)
        if not has_data:
            _say(self.setup_state, 'Name the dataset and Save to start the recording setup.', HINT)
        else:
            from ...generator.data_collection import data_collection_problems
            problems = data_collection_problems(p)
            _say(self.setup_state, (f'{len(problems)} problem(s): see Diagnostics below.' if problems else
                                    'Valid. Generate the bundle to include the recorder.'),
                 ERROR if problems else OK)
        self._refresh_modes()
        self._refresh_session()

    def _setup_edited(self, *_args) -> None:
        self.setup_dirty = True
        self.setup_save.setText('Save recording setup *')

    def apply_camera(self) -> None:
        name, model = self.camera_id.text().strip() or None, self.camera_profile.currentData()

        def mutate(doc):
            doc['perception']['camera']['id'] = name
            doc['perception']['camera']['profile'] = model
            for sensor in (doc.get('data_collection') or {}).get('sensors') or []:
                sensor['camera'] = name or sensor['camera']
            for key in ('id', 'profile'):
                if doc['perception']['camera'][key] is None:
                    doc['perception']['camera'].pop(key)
        self._edit(mutate, 'Camera saved.')

    def save_setup(self) -> None:
        camera = self._camera()
        dataset = {'id': self.dataset_id.text().strip(), 'classification': self.classification.currentData()}
        if self.dataset_description.text().strip():
            dataset['description'] = self.dataset_description.text().strip()
        sync = {}
        if self.sync_timeout.value():
            sync['timeout_ms'] = self.sync_timeout.value()
        if self.sync_tolerance.value():
            sync['tolerance_ms'] = self.sync_tolerance.value()

        def mutate(doc):
            dc = doc.get('data_collection') or {}
            dc['dataset'] = dataset
            if not dc.get('sensors') and camera is not None and camera.id:
                dc['sensors'] = [{'camera': camera.id, 'streams': ['color', 'depth']}]
            if sync:
                dc['sync'] = sync
            else:
                dc.pop('sync', None)
            doc['data_collection'] = dc
        if self._edit(mutate, 'Recording setup saved.'):
            self.setup_dirty = False
            self.refresh()

    def edit_trigger(self, row: Optional[int]) -> None:
        triggers = (self._data() or {}).get('triggers') or []
        if row is not None and not 0 <= row < len(triggers):
            return                                     # Edit without a selected trigger: nothing to edit
        current = triggers[row] if row is not None else None
        dialog = TriggerDialog(self.project(), current, self)
        if dialog.exec_() != QDialog.Accepted:
            return
        value = dialog.value()

        def mutate(doc):
            dc = doc['data_collection']
            items = dc.setdefault('triggers', [])
            if current is None:
                items.append(value)
            else:
                items[row] = value
            camera = (doc.get('perception') or {}).get('camera') or {}
            if not dc.get('sensors') and camera.get('id'):
                dc['sensors'] = [{'camera': camera['id'], 'streams': ['color', 'depth']}]
        self._edit(mutate, f'Trigger "{value["id"]}" saved.')

    def _trigger_selected(self) -> None:
        count = self.triggers.rowCount()
        selected = 0 <= self.triggers.currentRow() < count and bool(self.triggers.selectedItems())
        self.trigger_edit.setEnabled(selected)
        self.trigger_remove.setEnabled(selected)

    def remove_trigger(self) -> None:
        row = self.triggers.currentRow()
        if row < 0:
            return
        self._edit(lambda doc: doc['data_collection']['triggers'].pop(row), 'Trigger removed.')

    # live preview (the wizard's in-process observation; depth units by TP's depth_to_metres)
    def _preview_toggled(self, on: bool) -> None:
        self.preview_button.setText('Hide live view' if on else 'Show live view')
        self.preview_view.setVisible(on)
        if not on:
            _pixmap(self.preview_view, None)
            _say(self.preview_state, 'Shows what the camera publishes now (the live ROS topics).')

    def _tick_preview(self) -> None:
        if not self.preview_button.isChecked():
            return
        camera, host = self._camera(), self._host()
        if camera is None or host is None or not hasattr(host, 'camera_capture'):
            _say(self.preview_state, 'Live view unavailable here (no ROS session).', WARN)
            return
        try:
            capture = host.camera_capture(camera.rgb_topic, camera.depth_topic, camera.camera_info_topic)
        except Exception as exc:  # noqa: BLE001 - no ROS here: say so, never a stale image
            _say(self.preview_state, f'Live view unavailable: {exc}', WARN)
            self.preview_button.setChecked(False)
            return
        latest = capture.latest()
        if latest is None:
            _pixmap(self.preview_view, None)
            _say(self.preview_state, 'Waiting for camera frames… (is the camera / simulation running?)', WARN)
            return
        rgb, depth, _k, frame = latest
        if self.preview_mode.currentText() == 'Depth':
            if depth is None:
                _say(self.preview_state, capture.last_error() or 'No depth.', WARN)
                _pixmap(self.preview_view, None)
                return
            _pixmap(self.preview_view, depth_image(depth))
        else:
            _pixmap(self.preview_view, _np_image(rgb))
        _say(self.preview_state, f'Live: {rgb.shape[1]}×{rgb.shape[0]}' +
             (f' · {frame}' if self.advanced.isChecked() else ''), OK)

    # the recorder
    def _generation_files(self) -> Dict[str, Path]:
        """{mode: the generated recording file} of the last generation (none before Generate)."""
        p, gen = self.project(), getattr(self.ctrl, 'last_generation_dir', None)
        if p is None or not gen:
            return {}
        config = Path(gen) / p.bundle.app_package / 'config'
        return {f.name.split('.')[1]: f for f in sorted(config.glob('data_collection.*.yaml'))}

    def _refresh_modes(self) -> None:
        files = self._generation_files()
        current = self.mode.currentText()
        if [self.mode.itemText(i) for i in range(self.mode.count())] != list(files):
            self.mode.clear()
            self.mode.addItems(list(files))
            if current in files:
                self.mode.setCurrentText(current)

    def link(self):
        from ...generator.namespace import cell_namespace
        p = self.project()
        if p is None:
            return None
        ns = cell_namespace(p)
        if self._link is not None and (self._link_ns != ns or not self._link.alive()):
            self._link.close()
            self._link = None
            self._pending = []
        if self._link is None:
            try:
                from ...livesession.data_recorder import RecorderLink
                self._link = RecorderLink(ns)
                self._link_ns = ns
                from python_qt_binding.QtWidgets import QApplication
                app = QApplication.instance()
                if app is not None and not getattr(self, '_quit_hooked', False):
                    app.aboutToQuit.connect(self._close_link)
                    self._quit_hooked = True
            except Exception as exc:  # noqa: BLE001 - no ROS here
                self._link_error = str(exc)
                return None
        return self._link

    def toggle_recorder(self) -> None:
        launcher = self.launcher()
        if launcher is None:
            return
        if launcher.running('data_recorder'):
            launcher.stop('data_recorder')
            return
        files = self._generation_files()
        mode = self.mode.currentText()
        if mode not in files:
            _say(self.message, 'Generate the bundle first: the recorder runs the generated recording file.', WARN)
            return
        from ...composer.launch_plan import workspace_root
        from ...generator.namespace import cell_namespace
        command = D.recorder_command(workspace_root(getattr(self.ctrl, 'last_generation_dir', '') or '') or None,
                                     str(files[mode]), cell_namespace(self.project()), mode in ('isaac', 'gazebo'),
                                     self.root())
        try:
            launcher.start(command)
        except RuntimeError as exc:
            _say(self.message, str(exc), ERROR)
            return
        launcher.add_listener(self)
        self._recorder_starting = True
        self._recorder_tail = []
        _say(self.message, f'Recorder starting: {command.display()}' if self.advanced.isChecked()
             else 'Recorder starting…', OK)

    def open_generate(self) -> None:
        go = getattr(self._host(), 'show_generate', None)
        if go is not None:
            go()

    def _call(self, service: str) -> None:
        link = self.link()
        if link is None:
            return
        self._pending.append((service, link.call(service), time.monotonic()))
        if service == 'start_session':
            self.open_datasets.setVisible(False)
            _say(self.session_result, 'Starting a session…')

    def _poll_calls(self) -> None:
        still = []
        for service, future, since in self._pending:
            if not future.done():
                if time.monotonic() - since > CALL_TIMEOUT_S[service]:
                    if self._link is not None:
                        self._link.forget(service, future)
                    _say(self.session_result, f'The recorder did not answer ({service}, '
                         f'{CALL_TIMEOUT_S[service]:.0f} s): is it still running?', ERROR)
                    continue
                still.append((service, future, since))
                continue
            try:
                reply = future.result()
            except Exception as exc:  # noqa: BLE001
                _say(self.session_result, f'{service} failed: {exc}', ERROR)
                continue
            if reply.success:
                self._settle_until = time.monotonic() + STATE_SETTLE_S
            if not reply.success:
                _say(self.session_result, f'Refused: {reply.message}', ERROR)
            elif service == 'start_session':
                _say(self.session_result, 'Recording session ' + (reply.message if self.advanced.isChecked()
                                                                  else 'started') + '.', OK)
            elif service == 'capture':
                from ...livesession.data_recorder import parse_capture_reply
                got = parse_capture_reply(reply.message)
                outcome = got.get('outcome') or reply.message
                _say(self.session_result, f'Captured: {outcome.lower()}' +
                     (f' ({got["capture_id"]})' if self.advanced.isChecked() and got.get('capture_id') else ''),
                     OK if outcome == 'COMPLETE' else WARN)
            elif service == 'close_session':
                try:
                    report = json.loads(reply.message)
                except ValueError:
                    report = {}
                counts = (report.get('counters') or {}).get('captures') or {}
                sid = report.get('session_id', '')
                _say(self.session_result, f'Session sealed: {counts.get("complete", 0)} complete, '
                     f'{counts.get("partial", 0)} partial, {counts.get("failed", 0)} failed' +
                     (f' ({sid})' if self.advanced.isChecked() else '') + '.', OK)
                self.open_datasets.setVisible(True)
                self._session_after_close = sid
        self._pending = still

    def _open_sealed_session(self) -> None:
        """Datasets, re-read (the store changed), with the session just sealed selected."""
        self.tabs.blockSignals(True)
        self.tabs.setCurrentIndex(2)
        self.tabs.blockSignals(False)
        self.refresh_datasets(select_session=self._session_after_close)

    def _silent_streams(self, values: dict) -> List[str]:
        """The streams whose received-frame count (the recorder's diagnostics, `received/<stream>`) has not grown
        for FRAMES_SILENT_S. A drop is a new session's reset: it restarts the watch."""
        now = time.monotonic()
        silent = []
        for key, value in values.items():
            if not key.startswith('received/'):
                continue
            try:
                count = int(value)
            except (TypeError, ValueError):
                continue
            stream = key[len('received/'):]
            last = self._frames.get(stream)
            if last is None or count != last[0]:
                self._frames[stream] = (count, now)
            elif now - last[1] >= FRAMES_SILENT_S:
                silent.append(stream)
        return silent

    def _recorder_ended(self, code: int) -> None:
        self._recorder_starting = False
        launcher = self.launcher()
        proc = launcher.processes.get('data_recorder') if launcher is not None else None
        if (proc is not None and proc.stopped) or code == 0:
            _say(self.message, 'Recorder stopped.', OK)
        else:
            last = next((t for t in reversed(self._recorder_tail) if t.strip()), '')
            _say(self.message, f'The recorder ended (exit {code})' + (f': {last.strip()}' if last else '.'), ERROR)
        self._refresh_session()

    def _refresh_session(self) -> None:
        launcher = self.launcher()
        running = bool(launcher and launcher.running('data_recorder'))
        # the link (an rclpy node) is made only when the Capture tab is on screen, never by a refresh
        link = self.link() if self.isVisible() and self.tabs.currentIndex() == 0 else self._link
        status = link.status() if link is not None else None
        reachable = bool(link is not None and link.reachable())
        # a recorder already on the graph (the bundle's bring-up runs one): never start a second one beside it
        elsewhere = reachable and not running
        self.recorder_start.setText('Stop recorder' if running else 'Start recorder')
        self.recorder_start.setEnabled(launcher is not None and not elsewhere and (running or self.mode.count() > 0))
        self.recorder_start.setToolTip('A recorder is already running (the bundle bring-up started it)'
                                       if elsewhere else '')
        self.mode.setEnabled(not running)
        values = (status or {}).get('values') or {}
        state = values.get('session_state', '')
        active = state == 'ACTIVE'
        manual = any(t.get('kind') == 'manual' for t in (self._data() or {}).get('triggers') or [])
        busy = bool(self._pending) or time.monotonic() < self._settle_until
        self.record_button.setEnabled(reachable and not active and not busy)
        self.capture_button.setEnabled(reachable and active and manual and not busy)
        self.capture_button.setToolTip('' if manual else 'Add a Manual trigger to capture by hand')
        self.stop_button.setEnabled(reachable and active and not busy)
        if link is None:
            _say(self.recorder_state, 'Recorder: no ROS session here.' +
                 (f' ({self._link_error})' if self.advanced.isChecked() and self._link_error else ''), WARN)
        elif not reachable:
            _say(self.recorder_state, 'Recorder: not running. Start the bundle (its bring-up runs it) or press '
                 'Start recorder after Generate.' if self.mode.count() or running else
                 'Recorder: not running. Generate the bundle with a recording setup first.', WARN)
        else:
            backend = (status or {}).get('backend', '')
            text = 'Recorder: recording' if active else 'Recorder: ready'
            if self.advanced.isChecked():
                text += f' · {link.base}' + (f' · {backend}' if backend else '') + \
                    (f' · {values.get("session_id")}' if values.get('session_id') else '')
            _say(self.recorder_state, text, OK)
            if self._recorder_starting:
                self._recorder_starting = False
                _say(self.message, 'Recorder ready.', OK)
        self.bundle_generate.setVisible(self.mode.count() == 0 and not running and not elsewhere
                                        and bool(self._data())
                                        and getattr(self._host(), 'show_generate', None) is not None)
        if values:
            self.counters.setText(f'Captured {values.get("captures_attempted", 0)} · Complete '
                                  f'{values.get("captures_complete", 0)} · Partial {values.get("captures_partial", 0)}'
                                  f' · Failed {values.get("captures_failed", 0)}')
        else:
            self.counters.setText('')
        warnings = []
        silent = self._silent_streams(values) if reachable else []
        if silent:
            warnings.append(f'No camera frames for {FRAMES_SILENT_S:.0f} s or more: {", ".join(silent)}. The recorder '
                            'is waiting for images; a capture now fails. Check that the camera publishes (Isaac: '
                            'the scene is playing and its camera publishes images).')
        if status is not None and status.get('level', 0) >= 1:
            for key in ('captures_lost', 'write_failures', 'queue_overruns', 'sync_failures', 'clock_jumps'):
                if values.get(key) not in (None, '', '0'):
                    warnings.append(SYNC_FAILURES_TEXT.format(n=values[key]) if key == 'sync_failures'
                                    else f'{key.replace("_", " ")}: {values[key]}')
            if values.get('last_error'):
                warnings.append(values['last_error'])
        self.session_warnings.setText(' · '.join(warnings))

    # ---- Synthetic -----------------------------------------------------------------------------------------
    def _build_synthetic(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.provider_state = QLabel('')
        self.provider_state.setWordWrap(True)
        v.addWidget(self.provider_state)
        self.provider_box = QWidget()
        pv = QVBoxLayout(self.provider_box)
        pv.setContentsMargins(0, 0, 0, 0)
        row = QHBoxLayout()
        self.provider_combo = QComboBox()
        self.provider_combo.currentIndexChanged.connect(self._provider_changed)
        row.addWidget(QLabel('Generator'))
        row.addWidget(self.provider_combo, 1)
        pv.addLayout(row)
        self.provider_about = _hint('')
        pv.addWidget(self.provider_about)
        general = QGroupBox('Generation')
        gf = QFormLayout(general)
        self.seed, self.runs, self.views = QSpinBox(), QSpinBox(), QSpinBox()
        for label, spin in (('Seed', self.seed), ('Runs (scenes)', self.runs), ('Views per run', self.views)):
            gf.addRow(label, spin)
        pv.addWidget(general)
        self.recipe_box = QWidget()
        self.recipe_layout = QVBoxLayout(self.recipe_box)
        self.recipe_layout.setContentsMargins(0, 0, 0, 0)
        pv.addWidget(self.recipe_box)
        self.recipe_state = QLabel('')
        self.recipe_state.setWordWrap(True)
        pv.addWidget(self.recipe_state)
        row = QHBoxLayout()
        self.recipe_save = QPushButton('Save recipe')
        self.preview_sample = QPushButton('Preview sample')
        self.generate_button = QPushButton('Generate')
        self.recipe_save.clicked.connect(self.save_recipe)
        self.preview_sample.clicked.connect(lambda: self.start_generation(preview=True))
        self.generate_button.clicked.connect(lambda: self.start_generation(preview=False))
        for b in (self.recipe_save, self.preview_sample, self.generate_button):
            row.addWidget(b)
        row.addStretch(1)
        pv.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        pv.addWidget(self.progress)
        self.synthetic_result = QLabel('')
        self.synthetic_result.setWordWrap(True)
        pv.addWidget(self.synthetic_result)
        self.add_to_dataset = QPushButton('Add to dataset…')
        self.add_to_dataset.setVisible(False)
        self.add_to_dataset.clicked.connect(self.add_generated_to_dataset)
        pv.addWidget(self.add_to_dataset)
        self.sample_views = QHBoxLayout()
        self.sample_rgb, self.sample_mask, self.sample_depth = QLabel(), QLabel(), QLabel()
        for view in (self.sample_rgb, self.sample_mask, self.sample_depth):
            view.setMinimumSize(200, 150)
            view.setAlignment(Qt.AlignCenter)
            self.sample_views.addWidget(view)
        pv.addLayout(self.sample_views)
        self.synthetic_log = QPlainTextEdit()
        self.synthetic_log.setReadOnly(True)
        self.synthetic_log.setMaximumBlockCount(3000)
        self.synthetic_log.setMaximumHeight(160)
        pv.addWidget(self.synthetic_log)
        v.addWidget(self.provider_box)
        v.addStretch(1)
        self._generated_session: Optional[str] = None
        self._generation_kind: Optional[str] = None
        self._refused = False
        return w

    def discover_providers(self) -> None:
        self.providers, self.provider_problems = D.find_providers(D.provider_dirs(self.settings()))

    def refresh_synthetic(self) -> None:
        p = self.project()
        if p is None:
            return
        if not self.providers and not self.provider_problems:
            self.discover_providers()
        modes = [m.value for m in p.deployment.modes]
        settings = self.settings()
        usable = [(pr, D.provider_unavailable(pr, modes, settings)) for pr in self.providers]
        if not self.providers:
            text = ('No synthetic data generator is installed (trainit_data 0.8.0 installs the reference one).' +
                    (' ' + '; '.join(self.provider_problems) if self.advanced.isChecked() and self.provider_problems
                     else ''))
            _say(self.provider_state, text, WARN)
            self.provider_box.setVisible(False)
            return
        if not any(why is None for _pr, why in usable):
            served = sorted({b for pr in self.providers for b in pr.descriptor['backends']})
            if not any(b in modes for b in served):
                _say(self.provider_state, f'Synthetic data needs a {" / ".join(served)} deployment: this project '
                     f'deploys {", ".join(modes)}. Add the mode in Step 1 to generate synthetic data here.', WARN)
            else:
                _say(self.provider_state, usable[0][1], WARN)
            self.provider_box.setVisible(False)
            return
        self.provider_box.setVisible(True)
        ids = [pr.id for pr, why in usable if why is None]
        if [self.provider_combo.itemData(i) for i in range(self.provider_combo.count())] != ids:
            self.provider_combo.blockSignals(True)
            self.provider_combo.clear()
            for pr, why in usable:
                if why is None:
                    self.provider_combo.addItem(pr.name, pr.id)
            self.provider_combo.blockSignals(False)
            saved = ((self._data() or {}).get('synthetic') or {}).get('provider')
            i = self.provider_combo.findData(saved)
            self.provider_combo.setCurrentIndex(max(i, 0))
            self._provider_changed()
        elif self.provider is not None:
            self._build_recipe()
        _say(self.provider_state, 'Generator available: ' + self.provider.name +
             (f' ({self.provider.directory})' if self.advanced.isChecked() else ''), OK)
        launcher_obj = self.launcher()
        busy = bool(launcher_obj and (launcher_obj.running('synthetic') or launcher_obj.running('synthetic_preview')))
        self.preview_sample.setEnabled(launcher_obj is not None and not busy)
        self.generate_button.setEnabled(launcher_obj is not None and not busy)
        self.generate_button.setText('Stop generation' if busy and self._generation_kind == 'synthetic'
                                     else 'Generate')
        if busy and self._generation_kind == 'synthetic':
            self.generate_button.setEnabled(True)
        self.synthetic_log.setVisible(self.advanced.isChecked())

    def _provider_changed(self) -> None:
        pid = self.provider_combo.currentData()
        self.provider = next((p for p in self.providers if p.id == pid), None)
        if self.provider is None:
            return
        d = self.provider.descriptor
        self.provider_about.setText(d['description'] + f' Writes: {", ".join(d["outputs"]["modalities"])}; labels: '
                                    f'{", ".join(d["outputs"]["annotations"])}; classes: '
                                    f'{", ".join(d["outputs"]["classes"])}.')
        saved = (self._data() or {}).get('synthetic') or {}
        same = saved.get('provider') == self.provider.id
        for name, spin in (('seed', self.seed), ('runs', self.runs), ('views', self.views)):
            g = d['general'][name]
            spin.setRange(int(g['min']), int(min(g['max'], 2**31 - 1)))
            spin.setValue(int(saved.get(name, g['default']) if same else g['default']))
            spin.setToolTip(g.get('help', ''))
        self._saved_params = dict(saved.get('parameters') or {}) if same else {}
        self.param_widgets = {}
        self._build_recipe(force=True)

    def _build_recipe(self, force: bool = False) -> None:
        """The form FROM the descriptor: Simple parameters, plus the Advanced ones when Advanced is on."""
        if self.provider is None:
            return
        level = ('simple', 'advanced') if self.advanced.isChecked() else ('simple',)
        wanted = [p for p in self.provider.descriptor['parameters'] if p['level'] in level]
        if not force and [p['key'] for p in wanted] == list(self.param_widgets):
            return
        current = {k: self._widget_value(k) for k in self.param_widgets}
        while self.recipe_layout.count():
            item = self.recipe_layout.takeAt(0)
            if item.widget() is not None:
                item.widget().deleteLater()
        self.param_widgets = {}
        groups: Dict[str, QFormLayout] = {}
        for parameter in wanted:
            group = parameter['group']
            if group not in groups:
                box = QGroupBox(group.capitalize())
                groups[group] = QFormLayout(box)
                self.recipe_layout.addWidget(box)
            value = current.get(parameter['key'], self._saved_params.get(parameter['key'],
                                                                          self.provider.default(parameter['key'])))
            widgets = self._param_widgets(parameter, value)
            row = QHBoxLayout()
            row.setContentsMargins(0, 0, 0, 0)
            for widget in widgets:
                row.addWidget(widget)
            if parameter.get('unit') and parameter['type'] != 'choice':
                row.addWidget(QLabel(parameter['unit']))
            holder = QWidget()
            holder.setLayout(row)
            holder.setToolTip(parameter.get('help', ''))
            label = parameter['label'] + (' (min – max)' if parameter['type'].endswith('_range') else '')
            groups[group].addRow(label, holder)
            self.param_widgets[parameter['key']] = widgets

    @staticmethod
    def _param_widgets(parameter: dict, value) -> list:
        kind = parameter['type']
        if kind == 'choice':
            combo = QComboBox()
            for v in parameter['values']:
                combo.addItem(' × '.join(str(x) for x in v) if isinstance(v, list) else str(v), v)
            i = next((i for i, v in enumerate(parameter['values']) if v == value), 0)
            combo.setCurrentIndex(i)
            return [combo]
        count = 1 if kind in ('int', 'float') else 2
        values = [value] if count == 1 else list(value)
        out = []
        for v in values:
            if kind.startswith('int'):
                spin = QSpinBox()
                spin.setRange(int(parameter['min']), int(parameter['max']))
                spin.setValue(int(v))
            else:
                spin = QDoubleSpinBox()
                span = float(parameter['max']) - float(parameter['min'])
                # enough decimals for the value itself (a value is never shown, nor saved, rounded)
                spin.setDecimals(min(6, max(3 if span < 10 else 1, _decimals(v))))
                spin.setSingleStep(10 ** (len(str(int(span))) - 2) if span >= 10 else 0.005)
                spin.setRange(float(parameter['min']), float(parameter['max']))
                spin.setValue(float(v))
            out.append(spin)
        return out

    def _widget_value(self, key: str):
        widgets = self.param_widgets[key]
        if isinstance(widgets[0], QComboBox):
            return widgets[0].currentData()
        values = [w.value() for w in widgets]
        return values[0] if len(values) == 1 else values

    def recipe_values(self) -> Dict[str, Any]:
        """The parameters the user set (the form's), on top of the saved recipe's hidden ones; a value equal to
        the provider's default (as its widget shows it) is not an override."""
        out = dict(getattr(self, '_saved_params', {}) or {})
        out.update({key: self._widget_value(key) for key in self.param_widgets})
        defaults = {p['key']: self.provider.default(p['key']) for p in self.provider.descriptor['parameters']}
        return {k: v for k, v in out.items() if not self._same(k, v, defaults.get(k))}

    def _same(self, key: str, value, default) -> bool:
        widgets = self.param_widgets.get(key) or []
        decimals = [w.decimals() for w in widgets if isinstance(w, QDoubleSpinBox)]
        if not decimals or default is None:
            return value == default
        pairs = zip(value, default) if isinstance(value, list) and isinstance(default, list) else [(value, default)]
        return all(round(float(a), decimals[0]) == round(float(b), decimals[0]) for a, b in pairs)

    def save_recipe(self) -> None:
        if self.provider is None:
            return
        params = self.recipe_values()
        problems = self.provider.problems(params, self.seed.value(), self.runs.value(), self.views.value())
        if problems:
            _say(self.recipe_state, '; '.join(problems), ERROR)
            return
        recipe = {'provider': self.provider.id, 'seed': self.seed.value(), 'runs': self.runs.value(),
                  'views': self.views.value(), 'parameters': params}

        def mutate(doc):
            dc = doc.get('data_collection')
            if dc is None:
                name = ''.join(c if c.isalnum() else '_' for c in self.project().project_name.lower()).strip('_')
                dc = {'dataset': {'id': (name or 'synthetic')[:64]}}
                doc['data_collection'] = dc
            dc['synthetic'] = recipe
        if self._edit(mutate, 'Recipe saved in the project.'):
            self._saved_params = dict(params)
            _say(self.recipe_state, '', HINT)

    def start_generation(self, preview: bool) -> None:
        launcher = self.launcher()
        if launcher is None or self.provider is None:
            return
        kind = 'synthetic_preview' if preview else 'synthetic'
        if not preview and launcher.running('synthetic'):
            launcher.stop('synthetic')
            return
        params = self.recipe_values()
        runs, views = (1, 1) if preview else (self.runs.value(), self.views.value())
        problems = self.provider.problems(params, self.seed.value(), runs, views)
        if problems:
            _say(self.recipe_state, '; '.join(problems), ERROR)
            return
        if preview:
            self._drop_preview()
            self._preview_dir = tempfile.mkdtemp(prefix=PREVIEW_PREFIX)
            work, store = Path(self._preview_dir), str(Path(self._preview_dir) / 'store')
        else:
            work = Path.home() / '.cache' / 'trainit' / 'synthetic'
            work.mkdir(parents=True, exist_ok=True)
            store = None
        config = work / f'{self.provider.id}-{time.strftime("%Y%m%dT%H%M%S")}.json'
        config.write_text(json.dumps(self.provider.resolved_config(params), indent=2, sort_keys=True))
        try:
            # a preview names ONLY its scratch store (TRAINIT_DATA_ROOT too): the user's store is not in its command
            command = D.provider_command(self.provider, self.settings(), str(config),
                                         self.seed.value(), runs, views, store or self.root(), store=store,
                                         title='Synthetic preview' if preview else 'Synthetic generation', kind=kind)
            launcher.add_listener(self)
            launcher.start(command)
        except (ValueError, RuntimeError) as exc:
            _say(self.recipe_state, str(exc), ERROR)
            return
        self._generation_kind = kind
        self._generated_session = None
        self._refused = False
        self.add_to_dataset.setVisible(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.synthetic_log.clear()
        self.synthetic_log.appendPlainText(f'$ {command.display()}')
        _say(self.synthetic_result, 'Rendering one sample (Isaac starts first: about half a minute)…' if preview
             else 'Generating…', HINT)
        self.refresh()

    # launcher listener (GUI thread)
    def started(self, command) -> None:
        pass

    def line(self, kind: str, text: str) -> None:
        if kind == 'data_recorder':
            self._recorder_tail = (self._recorder_tail + [text])[-5:]
            return
        if kind not in ('synthetic', 'synthetic_preview'):
            return
        self.synthetic_log.appendPlainText(text)
        event = D.parse_progress(text)
        if event is None:
            return
        phase = event['phase']
        if phase == 'run' and event.get('runs'):
            self.progress.setRange(0, int(event['runs']))
            self.progress.setValue(int(event['run']))
            self.progress.setFormat(f'run {event["run"]} / {event["runs"]}')
        elif phase == 'invalid_config':
            self._refused = True
            _say(self.synthetic_result, 'The generator refused the settings: ' + '; '.join(event.get('problems', [])),
                 ERROR)
        elif phase == 'session':
            _say(self.synthetic_result, 'Rendering…' if kind == 'synthetic_preview' else
                 f'Generating {event.get("runs")} scene(s) × {event.get("views")} view(s)…', HINT)
        elif phase == 'done':
            self._generated_session = event['session_id']

    def finished(self, kind: str, code: int) -> None:
        if kind == 'data_recorder':
            self._recorder_ended(code)
            return
        if kind not in ('synthetic', 'synthetic_preview'):
            return
        self.progress.setVisible(False)
        sid = self._generated_session
        launcher = self.launcher()
        proc = launcher.processes.get(kind) if launcher is not None else None
        if proc is not None and proc.stopped:
            _say(self.synthetic_result, 'Stopped. A session left unfinished appears as "Interrupted" in Datasets.',
                 WARN)
        elif sid is None or code != 0:
            if not self._refused:
                _say(self.synthetic_result, f'Generation failed (exit {code}): see the log (Advanced).', ERROR)
        elif kind == 'synthetic_preview' and self._preview_dir is None:
            _say(self.synthetic_result, 'The preview was discarded.', HINT)
        elif kind == 'synthetic_preview':
            _say(self.synthetic_result, 'Sample rendered (a real render, discarded when replaced).', OK)
            self.cli(['inspect', sid, '--captures', '--limit', '1'], self._show_sample,
                     root=str(Path(self._preview_dir) / 'store'))
        else:
            _say(self.synthetic_result, f'Session created' + (f': {sid}' if self.advanced.isChecked() else '') +
                 '. Add it to a dataset to label and version it.', OK)
            self.add_to_dataset.setVisible(True)
        self.refresh()

    def _show_sample(self, result: D.CliResult) -> None:
        if not result.ok or not result.result or not result.result.get('captures'):
            _say(self.synthetic_result, f'The sample could not be read: {result.message()}', ERROR)
            return
        root = Path(result.result['root'])
        if self._preview_dir is None or not root.resolve().is_relative_to(Path(self._preview_dir).resolve()):
            return                                     # an older preview's answer, or not TSA's scratch store
        capture = result.result['captures'][0]
        by_stream = {o['stream']: o for o in capture['observations']}
        color, inst, depth = by_stream.get('color'), by_stream.get('instances'), by_stream.get('depth')
        if color is not None:
            path = _payload(root, color['payload']['path'])
            image = QImage(str(path)) if path else QImage()
            if inst is not None and not image.isNull():
                painter = QPainter(image)
                painter.setPen(QPen(QColor('#00e676'), 2))
                for instance in inst['data'].get('instances', []):
                    box = instance.get('box_tight_xywh_px')
                    if box:
                        painter.drawRect(*box)
                        painter.drawText(box[0], max(box[1] - 3, 10), instance['class'])
                painter.end()
            _pixmap(self.sample_rgb, image)
        if inst is not None:
            path = _payload(root, inst['payload']['path'])
            _pixmap(self.sample_mask, mask_image(str(path)) if path else None)
        if depth is not None:
            import numpy as np
            try:
                path = _payload(root, depth['payload']['path'])
                _pixmap(self.sample_depth, depth_image(np.load(path, allow_pickle=False)) if path else None)
            except (OSError, ValueError, TypeError):
                _pixmap(self.sample_depth, None)

    def _drop_preview(self) -> None:
        """The previous preview's TSA-owned scratch store (never the user's store)."""
        if self._preview_dir and is_preview_scratch(self._preview_dir):
            shutil.rmtree(self._preview_dir, ignore_errors=True)
        self._preview_dir = None
        for view in (self.sample_rgb, self.sample_mask, self.sample_depth):
            _pixmap(view, None)

    def add_generated_to_dataset(self) -> None:
        if not self._generated_session:
            return
        self.tabs.setCurrentIndex(2)
        self.refresh_datasets(select_session=self._generated_session)

    # ---- Datasets ------------------------------------------------------------------------------------------
    def _build_datasets(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        self.store_state = _hint('')
        v.addWidget(self.store_state)
        self.interrupted = QLabel('')
        self.interrupted.setWordWrap(True)
        self.interrupted.setStyleSheet(f'color:{WARN};')
        v.addWidget(self.interrupted)
        row = QHBoxLayout()
        self.inspect_button = QPushButton('Inspect')
        self.recover_button = QPushButton('Recover…')
        self.inspect_button.clicked.connect(self.inspect_selected)
        self.recover_button.clicked.connect(self.recover_selected)
        self.refresh_button = QPushButton('Refresh')
        self.refresh_button.clicked.connect(lambda: self.refresh_datasets())
        for b in (self.refresh_button, self.inspect_button, self.recover_button):
            row.addWidget(b)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(QLabel('<b>Sessions</b> — select the ones a version should hold'))
        self.session_table = QTableWidget(0, 6)
        self.session_table.setHorizontalHeaderLabels(['When', 'Source', 'Captures', 'Complete / Partial / Failed',
                                                      'Status', 'Session'])
        self.session_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.session_table.horizontalHeader().setStretchLastSection(True)
        self.session_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.session_table.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.session_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.session_table.itemSelectionChanged.connect(self._selection_changed)
        v.addWidget(self.session_table, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel('<b>Dataset</b>'))
        self.dataset_combo = QComboBox()
        self.dataset_combo.currentIndexChanged.connect(self._dataset_changed)
        row.addWidget(self.dataset_combo, 1)
        self.new_dataset = QPushButton('New dataset…')
        self.new_dataset.clicked.connect(self.create_dataset)
        row.addWidget(self.new_dataset)
        v.addLayout(row)
        self.dataset_hint = QLabel('')
        self.dataset_hint.setWordWrap(True)
        self.dataset_hint.setVisible(False)
        v.addWidget(self.dataset_hint)
        row = QHBoxLayout()
        self.create_version_button = QPushButton('Create version…')
        self.create_version_button.clicked.connect(self.create_version)
        self.import_labels = QPushButton('Import YOLO labels…')
        self.import_labels.clicked.connect(self.import_yolo_labels)
        row.addWidget(self.create_version_button)
        row.addWidget(self.import_labels)
        row.addStretch(1)
        v.addLayout(row)
        v.addWidget(QLabel('<b>Versions</b> — immutable'))
        self.version_list = QListWidget()
        self.version_list.setMaximumHeight(110)
        self.version_list.currentRowChanged.connect(self._version_changed)
        v.addWidget(self.version_list)
        self.version_summary = QLabel('')
        self.version_summary.setWordWrap(True)
        self.version_summary.setTextInteractionFlags(Qt.TextSelectableByMouse)
        v.addWidget(self.version_summary)
        row = QHBoxLayout()
        self.export_button = QPushButton('Export YOLO Detection…')
        self.export_button.clicked.connect(self.export_yolo)
        row.addWidget(self.export_button)
        row.addStretch(1)
        v.addLayout(row)
        self.dataset_result = QLabel('')
        self.dataset_result.setWordWrap(True)
        v.addWidget(self.dataset_result)
        self.cli_log = QPlainTextEdit()
        self.cli_log.setReadOnly(True)
        self.cli_log.setMaximumHeight(180)
        v.addWidget(self.cli_log)
        self._datasets_loaded = False
        self._select_after: Optional[str] = None
        self._version_detail: Dict[int, dict] = {}
        self._dataset_ready: Optional[str] = None
        return w

    def refresh_datasets(self, select_session: Optional[str] = None) -> None:
        self._select_after = select_session or self._select_after
        self._datasets_loaded = True              # requested: a second refresh before the answers re-reads nothing
        _say(self.store_state, f'Reading the data store {self.root()}…')
        self.cli(['inspect', '--detail'], self._sessions_loaded)
        self.cli(['dataset', 'list'], self._datasets_listed)

    def _sessions_loaded(self, result: D.CliResult) -> None:
        self._datasets_loaded = True
        if not result.ok:
            code = (result.error or {}).get('code')
            _say(self.store_state, ('The data store is empty: nothing has been recorded or generated on this '
                                    f'machine yet ({self.root()}).') if code == 'STORE_NOT_INITIALIZED' else
                 f'The data store could not be read: {result.message()}', WARN if code else ERROR)
            self.sessions = []
        else:
            self.sessions = list(reversed(result.result['sessions']))       # newest first
            _say(self.store_state, f'Data store {result.result["root"]}: {len(self.sessions)} session(s).', HINT)
        self._fill_sessions()

    def _fill_sessions(self) -> None:
        recording = self._recording_session()
        self.session_table.setRowCount(len(self.sessions))
        for i, s in enumerate(self.sessions):
            when = s.get('opened_wall_utc_ns')
            when = time.strftime('%Y-%m-%d %H:%M', time.localtime(when / 1e9)) if when else s['session_id'][4:19]
            source = (f'Synthetic ({s.get("backend")})' if s.get('synthetic') else
                      'Replay of a recording' if s.get('backend') == 'replay' else
                      f'Camera ({s.get("backend")})' if s.get('backend') else '?')
            outcomes = s.get('outcomes')
            health = ' / '.join(str(outcomes[k]) for k in ('complete', 'partial', 'failed')) if outcomes else '—'
            status = s.get('status') or s.get('error', '?')
            if status == 'ACTIVE':
                status = 'recording now' if s['session_id'] == recording else 'interrupted'
            for col, text in enumerate((when, source, str(s.get('captures') if s.get('captures') is not None
                                                          else '—'), health, status.lower(), s['session_id'])):
                item = QTableWidgetItem(text)
                item.setData(Qt.UserRole, s['session_id'])
                self.session_table.setItem(i, col, item)
        self.session_table.setColumnHidden(5, not self.advanced.isChecked())
        interrupted = [s['session_id'] for s in self.sessions
                       if s.get('status') == 'ACTIVE' and s['session_id'] != recording]
        self.interrupted.setText(f'Interrupted recording found: {len(interrupted)} session(s) were left unfinished '
                                 '(select one: Inspect shows it, Recover… seals it).' if interrupted else '')
        if self._select_after:
            for row, s in enumerate(self.sessions):
                if s['session_id'] == self._select_after:
                    self.session_table.selectRow(row)
            self._select_after = None
        self._selection_changed()

    def _recording_session(self) -> Optional[str]:
        status = self._link.status() if self._link is not None else None
        values = (status or {}).get('values') or {}
        return values.get('session_id') if values.get('session_state') == 'ACTIVE' else None

    def selected_sessions(self) -> List[dict]:
        rows = sorted({i.row() for i in self.session_table.selectedItems()})
        return [self.sessions[r] for r in rows if r < len(self.sessions)]

    def _selection_changed(self) -> None:
        selected = self.selected_sessions()
        recording = self._recording_session()
        one_active = len(selected) == 1 and selected[0].get('status') == 'ACTIVE' and \
            selected[0]['session_id'] != recording
        self.inspect_button.setEnabled(len(selected) == 1)
        self.recover_button.setVisible(bool(self.interrupted.text()))
        sealed = [s for s in selected if s.get('status') in ('CLOSED', 'INTERRUPTED')]
        ready = (self.dataset_combo.currentData() is not None
                 and self._dataset_ready == self.dataset_combo.currentData() and not self.jobs.busy)
        self.create_version_button.setEnabled(bool(sealed) and len(sealed) == len(selected) and ready)
        self.export_button.setEnabled(ready and self.version_list.currentRow() >= 0)
        self.recover_button.setEnabled(one_active and not self.jobs.busy)
        self.create_version_button.setToolTip('' if self.dataset_combo.currentData() else 'Choose or create a dataset')

    def _datasets_listed(self, result: D.CliResult) -> None:
        current = self.dataset_combo.currentData()
        self.datasets = result.result['datasets'] if result.ok else []
        planned = ((self._data() or {}).get('dataset') or {}).get('id')
        wanted = self._wanted_dataset or current or planned
        self._wanted_dataset = None
        self.dataset_combo.blockSignals(True)
        self.dataset_combo.clear()
        for d in self.datasets:
            if 'error' in d:
                continue
            self.dataset_combo.addItem(f'{d["dataset_id"]} — {d["name"]}', d['dataset_id'])
        i = self.dataset_combo.findData(wanted)
        missing = planned if i < 0 and wanted == planned and planned else None
        self.dataset_combo.setCurrentIndex(i if i >= 0 else (0 if self.dataset_combo.count() and not missing else -1))
        self.dataset_combo.blockSignals(False)
        self.dataset_hint.setVisible(bool(missing))
        if missing:
            _say(self.dataset_hint, f'The recording setup\'s dataset "{missing}" does not exist yet: New dataset… '
                 'creates it. Recorded sessions join a dataset when you create a version.', WARN)
        self._dataset_changed()

    def _dataset_changed(self) -> None:
        dataset = self.dataset_combo.currentData()
        if dataset is not None:
            self.dataset_hint.setVisible(False)
        self.version_list.clear()
        self.versions = []
        self._version_detail = {}
        self.version_summary.setText('')
        self.import_labels.setEnabled(dataset is not None)
        self.export_button.setEnabled(False)
        self._dataset_ready = None                 # versions / label sets of `dataset` not read yet
        self._selection_changed()
        if dataset is None:
            return
        self.cli(['dataset', 'show', dataset], lambda r, d=dataset: self._dataset_shown(d, r))

    def _dataset_shown(self, dataset: str, result: D.CliResult) -> None:
        if dataset != self.dataset_combo.currentData():
            return                                  # an answer about a dataset no longer selected
        if not result.ok:
            _say(self.dataset_result, result.message(), ERROR)
            return
        self.versions = list(result.result['versions'])
        self._label_sets = list(result.result['label_sets'])
        self._dataset_ready = dataset
        self.version_list.clear()
        for n in self.versions:
            self.version_list.addItem(f'v{n}')
        if self.versions:
            self.version_list.setCurrentRow(len(self.versions) - 1)
        self._selection_changed()

    def _version_changed(self, row: int) -> None:
        self.export_button.setEnabled(row >= 0 and self._dataset_ready == self.dataset_combo.currentData())
        if row < 0 or row >= len(self.versions):
            return
        n = self.versions[row]
        if n in self._version_detail:
            self._show_version(self._version_detail[n])
            return
        dataset = self.dataset_combo.currentData()
        self.cli(['version', 'show', dataset, str(n)], lambda r, n=n, d=dataset: self._version_loaded(d, n, r))

    def _version_loaded(self, dataset: str, n: int, result: D.CliResult) -> None:
        if dataset != self.dataset_combo.currentData():
            return
        if result.ok:
            self._version_detail[n] = result.result['version']
            if self.versions and self.version_list.currentRow() >= 0 and \
                    self.versions[self.version_list.currentRow()] == n:
                self._show_version(result.result['version'])

    def _show_version(self, v: dict) -> None:
        c = v['content']
        sel = c['selection']
        counts = c['split']['counts']
        splits = ', '.join(f'{k} {counts[k]["captures"]}' for k in ('train', 'val', 'test') if k in counts)
        excluded = sel['excluded']
        excluded = sum(excluded.values()) if isinstance(excluded, dict) else excluded
        sources = sorted({s.get('source', {}).get('backend', '?') for s in c['sessions']})
        text = (f'<b>v{v["version"]}</b> ({v["derivation"]["op"].lower().replace("_", " ")}): '
                f'{sel["captures"]} captures included ({" + ".join(sel["filter"]["include_outcomes"]).lower()}), '
                f'{excluded} excluded · from {len(c["sessions"])} session(s) ({", ".join(sources)}) · '
                f'split {splits} · {len(c["label_sets"])} label set(s) · rights {v["rights"]["classification"]}')
        if self.advanced.isChecked():
            text += f'<br>{v["content_hash"]} · split by {c["split"].get("group_by", "session")}'
        self.version_summary.setText(text)

    def inspect_selected(self) -> None:
        selected = self.selected_sessions()
        if len(selected) != 1:
            return
        sid = selected[0]['session_id']

        def shown(result: D.CliResult) -> None:
            if not result.ok:
                _say(self.dataset_result, result.message(), ERROR)
                return
            h = result.result['session']
            committed = result.result.get('committed')
            lines = [f'Session {sid}', f'Status: {h["status"]}',
                     f'Source: {h["source"]["transport"]} / {h["source"]["backend"]}',
                     f'Sensors: {", ".join(s["sensor_id"] for s in h["sensors"])}']
            if committed is not None:
                lines.append(f'Committed so far: {committed["complete"]} complete, {committed["partial"]} partial, '
                             f'{committed["failed"]} failed')
            elif h.get('counters'):
                lines.append(f'Captures: {h["counters"]["captures"]}')
            captures = result.result.get('captures') or []
            problems = D.capture_problems(captures)
            if problems:
                lines.append('Not complete, and why:')
                lines += ['  ' + p for p in problems[:INSPECT_PROBLEMS_SHOWN]]
                if len(problems) > INSPECT_PROBLEMS_SHOWN:
                    lines.append(f'  … and {len(problems) - INSPECT_PROBLEMS_SHOWN} more')
            total = result.result.get('captures_total')
            if total is not None and len(captures) < total:
                lines.append(f'(the first {len(captures)} of {total} captures were checked)')
            QMessageBox.information(self, 'Session', '\n'.join(lines))
        self.cli(['inspect', sid, '--captures', '--limit', str(INSPECT_CAPTURES_READ)], shown)

    def recover_selected(self) -> None:
        """Explicit only, behind a confirmation that says exactly what recover does (D0_8_DESIGN.md §9)."""
        selected = self.selected_sessions()
        if len(selected) != 1 or selected[0].get('status') != 'ACTIVE':
            return
        sid = selected[0]['session_id']
        answer = QMessageBox.question(
            self, 'Recover an interrupted recording',
            f'Recover session {sid}?\n\nThis seals the session as INTERRUPTED with the captures it committed, keeps '
            'a torn last index line in a side file, and MOVES any file no capture references into the session\'s '
            'orphans/ folder. It deletes nothing. A session a running recorder still holds is refused.',
            QMessageBox.Yes | QMessageBox.Cancel, QMessageBox.Cancel)
        if answer != QMessageBox.Yes:
            return

        def done(result: D.CliResult) -> None:
            if result.ok:
                r = result.result
                _say(self.dataset_result, f'Recovered: sealed INTERRUPTED with {r["committed_captures"]} capture(s); '
                     f'{len(r["quarantined"])} file(s) moved to orphans/; nothing deleted.', OK)
            else:
                _say(self.dataset_result, f'Not recovered: {result.message()}', ERROR)
            self.refresh_datasets()
        self.cli(['recover', sid], done)

    def create_dataset(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle('New dataset')
        form = QFormLayout(dialog)
        did = QLineEdit(((self._data() or {}).get('dataset') or {}).get('id', ''))
        name = QLineEdit()
        rights = QComboBox()
        for c in ('INTERNAL', 'CUSTOMER_RESTRICTED', 'PUBLIC_APPROVED'):
            rights.addItem(c.lower().replace('_', ' '), c)
        planned = (((self._data() or {}).get('dataset') or {}).get('classification') or 'internal').upper()
        rights.setCurrentIndex(max(rights.findData(planned), 0))
        form.addRow('Id', did)
        form.addRow('Name', name)
        form.addRow('Data rights', rights)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        args = ['dataset', 'create', did.text().strip(), '--name', name.text().strip() or did.text().strip(),
                '--classification', rights.currentData()]

        def done(result: D.CliResult) -> None:
            _say(self.dataset_result, f'Dataset {did.text().strip()} created.' if result.ok else
                 f'Not created: {result.message()}', OK if result.ok else ERROR)
            self._wanted_dataset = did.text().strip()
            self.cli(['dataset', 'list'], self._datasets_listed)
        self.cli(args, done)

    def create_version(self) -> None:
        dataset = self.dataset_combo.currentData()
        selected = self.selected_sessions()
        if dataset is None or not selected:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f'New version of {dataset}')
        form = QFormLayout(dialog)
        outcomes = {}
        if self.versions:                           # a derived version keeps its parent's outcome filter
            form.addRow('Include captures', _hint(f'the same outcomes as v{self.versions[-1]}'))
        else:
            for o, default in (('COMPLETE', True), ('PARTIAL', False), ('FAILED', False)):
                box = QCheckBox(o.lower())
                box.setChecked(default)
                outcomes[o] = box
                form.addRow('Include captures' if o == 'COMPLETE' else '', box)
        synthetic = [s for s in selected if s.get('synthetic')]
        derive = QCheckBox('Label generated sessions from their synthetic ground truth (boxes)')
        derive.setChecked(bool(synthetic))
        derive.setVisible(bool(synthetic))
        form.addRow(derive)
        labels = QListWidget()
        for ls in getattr(self, '_label_sets', []):
            labels.addItem(ls)
        labels.setSelectionMode(QAbstractItemView.MultiSelection)
        labels.setMaximumHeight(80)
        if labels.count():
            form.addRow('Existing label sets', labels)
        group = QComboBox()
        group.addItem('by scene (randomization run; recorded sessions by session)', 'randomization_run')
        group.addItem('by session', 'session')
        group.setCurrentIndex(0 if synthetic else 1)
        if not self.versions:
            form.addRow('Split groups', group)
        form.addRow(_hint(f'{len(selected)} session(s) selected. ' +
                          ('A new version is created from the latest one with these sessions added.' if self.versions
                           else 'The first version of this dataset.') + ' Versions are immutable.'))
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        chosen = [o for o, box in outcomes.items() if box.isChecked()] or ['COMPLETE']
        label_sets = [i.text() for i in labels.selectedItems()]
        steps = []
        if derive.isChecked() and synthetic:
            steps.append(['labels', 'from-synthetic', dataset, '--kind', 'BBOX_2D'] +
                         [a for s in synthetic for a in ('--session', s['session_id'])])
        self._version_plan = {'dataset': dataset, 'sessions': [s['session_id'] for s in selected],
                              'outcomes': chosen, 'label_sets': label_sets, 'group_by': group.currentData()}
        self._run_steps(steps, self._make_version)

    def _run_steps(self, steps: List[List[str]], then: Callable[[], None]) -> None:
        if not steps:
            then()
            return

        def done(result: D.CliResult) -> None:
            if not result.ok:
                _say(self.dataset_result, f'Stopped: {result.message()}', ERROR)
                return
            ls = (result.result or {}).get('label_set')
            if ls:
                self._version_plan['label_sets'].append(ls['label_set_id'])
            self._run_steps(steps[1:], then)
        self.cli(steps[0], done)

    def _make_version(self) -> None:
        plan = self._version_plan
        args = ['version', 'create', plan['dataset']]
        if self.versions:
            args += ['--op', 'ADD_SESSIONS', '--from', str(self.versions[-1])]
        else:
            args += ['--group-by', plan['group_by']]
            for o in plan['outcomes']:
                args += ['--include-outcome', o]
        for sid in plan['sessions']:
            args += ['--session', sid]
        for ls in plan['label_sets']:
            args += ['--label-set', ls]

        def done(result: D.CliResult) -> None:
            if result.ok:
                v = result.result['version']
                sel = v['content']['selection']
                excluded = sel['excluded']
                excluded = sum(excluded.values()) if isinstance(excluded, dict) else excluded
                _say(self.dataset_result, f'Version v{v["version"]} created: {sel["captures"]} captures included, '
                     f'{excluded} excluded.', OK)
                self._version_detail[v['version']] = v
            else:
                _say(self.dataset_result, f'No version created: {result.message()}', ERROR)
            self._dataset_changed()
        self.cli(args, done)

    def import_yolo_labels(self) -> None:
        dataset = self.dataset_combo.currentData()
        if dataset is None:
            return
        directory = QFileDialog.getExistingDirectory(self, 'YOLO label files (<image stem>.txt)')
        if not directory:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle('Classes')
        form = QFormLayout(dialog)
        classes = QLineEdit()
        classes.setPlaceholderText('class names in YOLO index order, comma-separated')
        form.addRow('Classes', classes)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return
        args = ['labels', 'import-yolo', dataset, directory]
        for name in [c.strip() for c in classes.text().split(',') if c.strip()]:
            args += ['--class', name]

        def done(result: D.CliResult) -> None:
            _say(self.dataset_result, (f'Imported label set ({result.result["label_set"]["entries"]["count"]} '
                                       'entries).') if result.ok else f'Not imported: {result.message()}',
                 OK if result.ok else ERROR)
            self._dataset_changed()
        self.cli(args, done)

    def export_yolo(self) -> None:
        row = self.version_list.currentRow()
        dataset = self.dataset_combo.currentData()
        if dataset is None or row < 0:
            return
        n = self.versions[row]
        options = self._export_options(dataset, n)
        if options is None:
            return
        destination = QFileDialog.getExistingDirectory(self, f'Export {dataset} v{n} as YOLO detection into '
                                                             '(an empty folder)')
        if not destination:
            return
        _say(self.dataset_result, f'Exporting {dataset} v{n}…')

        def done(result: D.CliResult) -> None:
            if not result.ok:
                _say(self.dataset_result, f'Not exported: {result.message()}', ERROR)
                return
            e = result.result['export']
            splits = e['summary']['splits']
            total = sum(s['images'] for s in splits.values())
            _say(self.dataset_result, f'Exported {total} images (' + ' / '.join(
                f'{k} {s["images"]}' for k, s in splits.items()) + f') to {e["destination"]}. '
                'The canonical data was not changed.', OK)
        self.cli(['export', 'yolo', dataset, str(n), destination] + options, done, timeout_s=6 * 3600.0)

    def _export_options(self, dataset: str, n: int) -> Optional[List[str]]:
        """The export's label-set arguments, or None (nothing to export / cancelled). Box label sets of ONE origin
        export as they are; of several origins (synthetic ground truth, imported, human) the user picks which, or
        allows mixing them -- trainit_data refuses an unasked mix and records an allowed one."""
        detail = self._version_detail.get(n)
        if detail is None:                          # not read yet: trainit_data still judges the export
            return []
        boxes = [ls for ls in detail['content'].get('label_sets') or [] if ls.get('kind') == 'BBOX_2D']
        if not boxes:
            _say(self.dataset_result, f'{dataset} v{n} has no box labels to export. Label it first: generated '
                 'sessions get theirs from the synthetic ground truth when a version is created, recorded ones '
                 'with Import YOLO labels…; then create a new version.', WARN)
            return None
        if len({ls.get('source') for ls in boxes}) < 2:
            return []
        dialog = QDialog(self)
        dialog.setWindowTitle(f'Labels to export from {dataset} v{n}')
        form = QFormLayout(dialog)
        form.addRow(_hint('This version holds box labels of different origins. Export one origin, or allow '
                          'mixing them (the export records it).'))
        boxes_checked = {}
        for ls in boxes:
            box = QCheckBox(f'{ls.get("source", "?").lower().replace("_", " ")}: {ls.get("entries_selected", "?")} '
                            f'image(s) ({ls["label_set_id"]})')
            box.setChecked(True)
            boxes_checked[ls['label_set_id']] = (box, ls.get('source'))
            form.addRow(box)
        mixed = QCheckBox('Allow labels of different origins in one export')
        form.addRow(mixed)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec_() != QDialog.Accepted:
            return None
        chosen = [(i, source) for i, (box, source) in boxes_checked.items() if box.isChecked()]
        if not chosen:
            _say(self.dataset_result, 'No labels chosen: nothing exported.', WARN)
            return None
        if len({source for _, source in chosen}) > 1 and not mixed.isChecked():
            _say(self.dataset_result, 'Labels of different origins were chosen without allowing the mix: choose one '
                 'origin, or allow mixing them.', WARN)
            return None
        args = [a for i, _ in chosen for a in ('--label-set', i)]
        return args + (['--allow-mixed-provenance'] if mixed.isChecked() else [])

    # ---- refresh / timer -----------------------------------------------------------------------------------
    def refresh(self) -> None:
        if self.project() is None:
            return
        tab = self.tabs.currentIndex()
        if tab == 0:
            self.refresh_capture()
        elif tab == 1:
            self.refresh_synthetic()
        elif tab == 2:
            self.cli_log.setVisible(self.advanced.isChecked())
            if not self._datasets_loaded:
                self.refresh_datasets()
            else:
                self.session_table.setColumnHidden(5, not self.advanced.isChecked())
                self._selection_changed()

    def _tick(self) -> None:
        if not self.isVisible() or self.project() is None:
            return
        if self.tabs.currentIndex() == 0:
            self._poll_calls()
            self._tick_preview()
            self._refresh_session()

    def _close_link(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None

    def close(self) -> bool:  # pragma: no cover - GUI lifecycle
        self._drop_preview()
        self._close_link()
        return super().close()


__all__ = ['DataPanel', 'TriggerDialog', 'describe_trigger']
