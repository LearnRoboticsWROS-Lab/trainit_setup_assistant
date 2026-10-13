"""The GRASP FACE picker: load the object's CAD, click the face, keep the numbers.

ADR-0018 Decision 1 -- this adds no concept. ``SceneObject.face_radius_m()`` already
exists, already has a home in ``perception.yaml`` and already reaches the detector; the
picker replaces its APPROXIMATION with a MEASUREMENT. For the reference capsule the
bounding box says 28,10 mm where the truth is 20,95 mm (+80 % on the area that
``visible_fraction`` divides by), and nothing anywhere said so.

Decision 2 -- three orthographic views in pure numpy, not a trackball. No VTK (no rosdep
key, pulls an MPI stack, ADR-0003), no pyqtgraph (no triangle picking, so the ray cast gets
written anyway). The raster, the flood fill and the measurement live in
:mod:`trainit_setup_assistant.geometry.mesh_face`; this module is the Qt skin over them.

Decision 4 -- the unit is explicit, because getting it wrong is SILENT: a mesh file carries
no unit, this workspace mixes metres and millimetres, and the survey's own run reported a
51 m2 face and a 4042 mm radius without raising. So the scale is a field, defaulted from
``SceneObject.scale``, and a face outside the band the RUNTIME refuses is refused here.

Decision 6 -- the face FAMILY and its parameters come from
``trainit_perception``'s registry through :mod:`trainit_setup_assistant.model.faces`. The
dialog renders one spin box per parameter THAT face declares, pre-filled where the
measurement maps onto it by the shipped naming convention, so a face family added to the
registry is configurable here without a line of this file changing.

**Shape**: the live tuner's, which is the only interactive thing in the wizard today --
a ``QTimer`` polling a cache, widgets touched from nowhere but the GUI thread, and the
work redone from the form rather than from remembered state. The loading/rastering step
goes through that cache, so a 4000-triangle part cannot freeze the dialog between the
click on "Load" and the first frame.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QImage, QPixmap
from python_qt_binding.QtWidgets import (QCheckBox, QComboBox, QDialog,
                                         QDialogButtonBox, QDoubleSpinBox, QFileDialog,
                                         QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                                         QLineEdit, QListWidget, QPushButton,
                                         QVBoxLayout)

from ..geometry import mesh_face as mf
from ..geometry import step_faces as sf
from ..model import faces as registry

#: The units an STL/OBJ is actually written in, in this workspace and in general.
SCALES: Tuple[Tuple[str, float], ...] = (('metres  (scale 1)', 1.0),
                                         ('millimetres  (scale 0.001)', 0.001),
                                         ('centimetres  (scale 0.01)', 0.01),
                                         ('inches  (scale 0.0254)', 0.0254))

_STEP_SUFFIXES = ('.step', '.stp')


class OrthoView(QLabel):
    """One orthographic view: it PAINTS a numpy raster and it handles a CLICK.

    The first and only widget in the assistant that does either -- the wizard has no
    ``paintEvent``, no ``QPainter``, no ``QGraphicsView`` and no mouse handler anywhere
    else -- so it is deliberately one small class with one job, and the parent dialog keeps
    all the state.

    There is no ``paintEvent`` override even here: the raster is handed to Qt as a
    ``QImage`` and shown through ``setPixmap``, exactly the way the live tuner shows the
    camera frame and the mask. The click is mapped back through the SAME aspect-preserving
    scaling Qt applied, which is the whole of the arithmetic below.
    """

    def __init__(self, title: str, on_pick=None):
        super().__init__(title)
        self.setMinimumSize(210, 210)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet('background:#222; color:#888;')
        self.setToolTip('Click the face the tool comes down on. The marked region is '
                        'flood-filled across coplanar triangles and highlighted.')
        self._title = title
        self._render = None
        self._image_size = 0
        self.on_pick = on_pick                   # a plain callable: no Signal/Slot binding

    def show_render(self, render, rgb) -> None:
        """Display one shaded raster. ``rgb`` is (H, W, 3) uint8, contiguous."""
        self._render = render
        self._image_size = int(rgb.shape[0])
        h, w, _ = rgb.shape
        image = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.setPixmap(QPixmap.fromImage(image).scaled(
            self.width(), self.height(), Qt.KeepAspectRatio))

    def clear_render(self) -> None:
        self._render = None
        self.setPixmap(QPixmap())
        self.setText(self._title)

    def image_pixel(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """Widget coordinates -> image pixel, or ``None`` outside the rendered square.

        ``KeepAspectRatio`` centres the pixmap in the label, so undoing it needs both the
        scale and the letterbox offset. Getting this wrong does not crash, it picks the
        wrong face -- which is the one failure this whole ADR is about.
        """
        pm = self.pixmap()
        if pm is None or pm.isNull() or not self._image_size:
            return None
        ox, oy = (self.width() - pm.width()) / 2.0, (self.height() - pm.height()) / 2.0
        scale = self._image_size / float(max(pm.width(), 1))
        col, row = int((x - ox) * scale), int((y - oy) * scale)
        if not (0 <= col < self._image_size and 0 <= row < self._image_size):
            return None
        return col, row

    def mousePressEvent(self, event):            # noqa: N802 - Qt's own name
        pos = event.pos()
        hit = self.image_pixel(pos.x(), pos.y())
        if hit is None or self._render is None or self.on_pick is None:
            return
        # The z-buffer already resolved occlusion, so a click IS a triangle lookup: the
        # ray cast behind `mesh_face.ray_hit` is the programmatic path, not a second one
        # for the mouse to disagree with.
        triangle = self._render.triangle_at(hit[0], hit[1])
        if triangle >= 0:
            self.on_pick(triangle)


class GraspFacePicker(QDialog):
    """Load the grasp target's mesh, click its grasp face, read back the measurement.

    The dialog never writes into the project: :meth:`result_face` hands back
    ``(face name, params, provenance)`` and the caller (the Perception step) decides. That
    keeps it usable from a test with no project at all.
    """

    def __init__(self, parent=None, *, mesh_path: str = '', scale: float = 1.0,
                 face: str = '', params: Optional[Dict[str, float]] = None):
        super().__init__(parent)
        self.setWindowTitle('Measure the grasp face')
        self.setMinimumWidth(760)
        self._mesh: Optional[mf.Mesh] = None
        self._renders: Tuple = ()
        self._cells: List[int] = []
        self._measured: Optional[mf.FaceMeasurement] = None
        self._step_faces: Tuple[sf.StepFace, ...] = ()
        self._step_sorted: Tuple[sf.StepFace, ...] = ()   # the order the list widget shows
        self._pending: Optional[Tuple[str, float]] = None    # the cache the timer polls
        self._param_fields: Dict[str, QDoubleSpinBox] = {}
        self._said_no_freecad = False
        root = QVBoxLayout(self)

        intro = QLabel(
            'Mark WHERE on the object you intend to grasp. TSA measures that face and '
            'writes its size into the detector, where <b>visible_fraction</b> divides by '
            'it — the score then falls as what the camera sees deviates from the region '
            'you marked. Without this the face is approximated from the bounding box, '
            'which on the reference capsule is 28,10 mm against a true 20,95 mm.')
        intro.setWordWrap(True)
        root.addWidget(intro)

        form = QFormLayout()
        pick = QHBoxLayout()
        self.path = QLineEdit(mesh_path or '')
        self.path.setPlaceholderText('STL / OBJ (STEP when a headless FreeCAD is present)')
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        load = QPushButton('Load')
        load.clicked.connect(self._request_load)
        pick.addWidget(self.path, 1)
        pick.addWidget(browse)
        pick.addWidget(load)
        form.addRow('Mesh / CAD file', pick)

        self.scale = QComboBox()
        for label, value in SCALES:
            self.scale.addItem(label, value)
        self.scale.addItem('custom…', 0.0)
        self.scale.setToolTip(
            'The file\'s UNIT. A mesh carries none, and the error is SILENT: this '
            'workspace itself mixes metres (the FAIRINO STLs) and millimetres (the '
            'suction gripper, with scale=0.001 in its xacro). A STEP file carries its '
            'own unit, so this is ignored for one.')
        self.custom_scale = QDoubleSpinBox()
        self.custom_scale.setDecimals(6)
        self.custom_scale.setRange(1e-6, 1000.0)
        self.custom_scale.setValue(1.0)
        self.custom_scale.setEnabled(False)
        self.scale.currentIndexChanged.connect(
            lambda i: self.custom_scale.setEnabled(self.scale.currentData() == 0.0))
        unit = QHBoxLayout()
        unit.addWidget(self.scale, 1)
        unit.addWidget(self.custom_scale)
        form.addRow('Unit of the file', unit)
        self._select_scale(scale)

        # The face FAMILY, from the registry — never a list of names in this file.
        self.face = QComboBox()
        for name in registry.available_faces():
            self.face.addItem(name, name)
        self.face.setToolTip(
            'The shape the tool meets, from trainit_perception\'s face registry '
            f'({registry.registry_source()}). A bottle picked from '
            'above is a disc of the CAP\'s radius, not of the body\'s.')
        self.face.currentIndexChanged.connect(self._rebuild_params)
        form.addRow('Grasp face', self.face)

        self.flip = QCheckBox('look from the opposite side (-X, -Y, -Z)')
        self.flip.setToolTip(
            'Three views reach three sides. The grasp face may be on any of the six, and '
            'which three are visible depends only on how the CAD happens to be oriented — '
            'so this flips all three without a trackball (ADR-0018 Decision 2).')
        self.flip.toggled.connect(self._rerender)
        form.addRow('Point of view', self.flip)
        root.addLayout(form)

        views = QHBoxLayout()
        self.views = tuple(OrthoView(title, on_pick=self._on_pick)
                           for title, _ in mf.VIEWS)
        for view in self.views:
            views.addWidget(view)
        self.step_list = QListWidget()
        self.step_list.setToolTip('Exact B-rep faces, read by a headless FreeCAD: true '
                                  'area, axis and radius, with no tessellation and no '
                                  'tolerance. Select one to use its numbers.')
        self.step_list.setMaximumWidth(330)
        self.step_list.currentRowChanged.connect(self._on_step_row)
        self.step_list.hide()
        views.addWidget(self.step_list)
        root.addLayout(views, 1)

        self.status = QLabel('Load the grasp target\'s mesh to begin.')
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.params_box = QGroupBox('What gets written into the detector')
        self.params_form = QFormLayout(self.params_box)
        root.addWidget(self.params_box)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        # The object's own face when it has one, the REGISTRY's default otherwise — never
        # whatever happens to sort first, which is how `cylinder_side` would quietly become
        # the default the day a face family is registered ahead of `disc`.
        index = self.face.findData(face or registry.DEFAULT_FACE)
        self.face.setCurrentIndex(max(index, 0))
        self._rebuild_params(prefill=params or {})
        self._ok(False)

        # The live tuner's shape: one timer, one cache. Loading and rastering a part is
        # tens of milliseconds at grasp-target sizes and seconds on something nobody
        # should mark a face on, so it happens on a tick and not inside the click.
        self._timer = QTimer(self)
        self._timer.setInterval(60)
        self._timer.timeout.connect(self._tick)
        self._timer.start()
        if mesh_path:
            self._request_load()

    # ---- the form ----------------------------------------------------------------
    def _select_scale(self, value: float) -> None:
        index = self.scale.findData(float(value))
        if index >= 0:
            self.scale.setCurrentIndex(index)
            return
        self.scale.setCurrentIndex(self.scale.count() - 1)     # custom…
        self.custom_scale.setEnabled(True)
        self.custom_scale.setValue(float(value) or 1.0)

    def _scale_value(self) -> float:
        data = self.scale.currentData()
        return float(self.custom_scale.value() if data == 0.0 else data)

    def face_name(self) -> str:
        return str(self.face.currentData() or registry.DEFAULT_FACE)

    def _rebuild_params(self, *_args, prefill: Optional[Dict[str, float]] = None) -> None:
        """One spin box per parameter THIS face declares (ADR-0018 Decision 6).

        The registry answers what they are called; the shipped naming convention fills the
        ones it covers from the measurement; anything else is the user's to type, with the
        measurement printed above. Nothing is ever written because TSA guessed what a
        parameter meant.
        """
        keep = dict(prefill or {k: w.value() for k, w in self._param_fields.items()})
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_fields = {}
        names = registry.face_params(self.face_name())
        if not names:
            self.params_form.addRow(QLabel(
                f'The face "{self.face_name()}" is not in '
                f'{registry.registry_source()}, so its parameters cannot be rendered. '
                f'Refresh TSA\'s catalog from a trainit_perception that registers it, or '
                f'type the numbers into the detector\'s params by hand.'))
            return
        lo, hi = registry.size_band()
        for name in names:
            box = QDoubleSpinBox()
            box.setDecimals(5)
            box.setRange(0.0, hi)
            box.setSingleStep(0.001)
            box.setSuffix(' m')
            box.setValue(float(keep.get(name, 0.0) or 0.0))
            box.setToolTip(f'{name}: what the face is sized by, in METRES. 0 = unknown, '
                           f'which makes a rule that needs visible_fraction refusable '
                           f'instead of silently demoted. The runtime refuses anything '
                           f'outside {lo:g}..{hi:g} m.')
            self.params_form.addRow(name, box)
            self._param_fields[name] = box
        self._apply_measurement()

    def _ok(self, enabled: bool) -> None:
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(enabled))

    def _say(self, message: str, ok: bool = False) -> None:
        self.status.setStyleSheet('color:#2e7d32; font-weight:bold;' if ok else '')
        self.status.setText(('✓ ' if ok else '') + message)

    # ---- loading, through the cache the timer polls ------------------------------
    def _browse(self) -> None:                   # pragma: no cover - needs a file dialog
        start = os.path.dirname(self.path.text().strip()) or os.path.expanduser('~')
        chosen, _ = QFileDialog.getOpenFileName(
            self, 'Open the grasp target\'s mesh or CAD', start,
            'Mesh / CAD (*.stl *.STL *.obj *.step *.stp);;All files (*)')
        if chosen:
            self.path.setText(chosen)
            self._request_load()

    def _request_load(self) -> None:
        path = os.path.expanduser(os.path.expandvars(self.path.text().strip()))
        if not path:
            self._say('Give the path of the grasp target\'s mesh (STL or OBJ).')
            return
        self._pending = (path, self._scale_value())
        self._say(f'reading {os.path.basename(path)}…')

    def _tick(self) -> None:
        """The polled cache: exactly one load per request, never inside an event handler."""
        if self._pending is None:
            return
        path, scale = self._pending
        self._pending = None
        try:
            self._load(path, scale)
        except Exception as exc:                 # noqa: BLE001 - a dialog must not die
            self._mesh = None
            self._renders = ()
            for view in self.views:
                view.clear_render()
            self._say(str(exc))
            self._ok(False)

    def _load(self, path: str, scale: float) -> None:
        self._step_faces = self._step_sorted = ()
        self.step_list.clear()
        self.step_list.hide()
        self._cells = []
        self._measured = None
        mesh_path, note = path, ''
        if path.lower().endswith(_STEP_SUFFIXES):
            mesh_path, note = self._load_step(path)
            # FreeCAD's internal unit is the millimetre, so its tessellation is too -- and
            # a STEP file states its own unit, which is why this one is KNOWN and not a
            # question put to the user (ADR-0018 Decision 4 is about files that carry none).
            scale = sf.STEP_SCALE
        self._mesh = mf.read_mesh(mesh_path, scale)
        self._rerender()
        extents = self._mesh.extents_m() * 1000.0
        self._say(f'{os.path.basename(path)}: {self._mesh.count} triangles, '
                  f'{extents[0]:.1f} x {extents[1]:.1f} x {extents[2]:.1f} mm at scale '
                  f'{scale:g}. {note}Click the face the tool comes down on.')
        self._ok(False)

    def _load_step(self, path: str) -> Tuple[str, str]:
        """STEP: the exact path, and only when this machine happens to have FreeCAD.

        Optional by construction (ADR-0018 Decision 3) -- probed the way ``pxr`` is, never
        declared in ``package.xml``. When the probe fails the dialog says so ONCE, calmly,
        and refuses the file rather than pretending to measure something it cannot read.
        """
        if not sf.freecad_available():
            raise ValueError(
                f'{os.path.basename(path)} is a STEP file and no headless FreeCAD was '
                f'found on this machine, so it cannot be read. This is expected: STEP is '
                f'an OPTIONAL exact path (ADR-0018 Decision 3) and never a dependency. '
                f'Export the part to STL — Isaac needs a mesh anyway — or set '
                f'{sf.FREECAD_ENV} to a FreeCAD executable or AppImage and try again.')
        tess = os.path.join(
            os.environ.get('TMPDIR', '/tmp'),
            f'trainit_face_{os.path.splitext(os.path.basename(path))[0]}.stl')
        self._step_faces, written = sf.read_step_faces(path, tessellate_to=tess)
        planar = [f for f in self._step_faces if f.area_m2 > 0]
        self.step_list.clear()
        for face in sorted(planar, key=lambda f: -f.area_m2):
            self.step_list.addItem(face.label())
        self._step_sorted = tuple(sorted(planar, key=lambda f: -f.area_m2))
        self.step_list.show()
        return (written or path,
                f'FreeCAD read {len(self._step_faces)} EXACT B-rep faces — pick one on '
                f'the right for the true area, or click the views to measure the '
                f'tessellation. ')

    # ---- picking -----------------------------------------------------------------
    def _rerender(self) -> None:
        """Re-raster the three views (and re-highlight the marked face, if there is one).

        The measurement is NOT recomputed: it is a property of the face, not of the view it
        was clicked in, and recomputing it here would be the second code path that could
        disagree with the first.
        """
        if self._mesh is None:
            return
        self._renders = mf.ortho_views(self._mesh, 320,
                                       from_positive=not self.flip.isChecked())
        for view, render in zip(self.views, self._renders):
            view.show_render(render, mf.shade(self._mesh, render, self._cells))

    def _on_pick(self, triangle: int) -> None:
        if self._mesh is None:
            return
        try:
            self._cells = mf.coplanar_region(self._mesh, int(triangle))
            measured = mf.measure_region(
                self._mesh, self._cells,
                source=f'{os.path.basename(self.path.text().strip())} '
                       f'@scale {self._scale_value():g}, {len(self._cells)} triangles')
        except Exception as exc:                 # noqa: BLE001
            self._say(f'could not measure that face: {exc}')
            return
        exact = sf.match_measurement(measured, self._step_faces) if self._step_faces else None
        if exact is not None:
            # the exact numbers win; the highlight stays the tessellated region, which
            # is what the user actually clicked
            measured = exact.as_measurement()
        self._set_measurement(measured)
        for view, render in zip(self.views, self._renders):
            view.show_render(render, mf.shade(self._mesh, render, self._cells))

    def _on_step_row(self, row: int) -> None:
        if 0 <= row < len(self._step_sorted):
            self._set_measurement(self._step_sorted[row].as_measurement())

    def _set_measurement(self, measured: mf.FaceMeasurement) -> None:
        """Accept a measurement, or refuse it for being implausible (Decision 4)."""
        reason = mf.implausible_reason(measured, scale=self._scale_value(),
                                       band=registry.size_band())
        if reason:
            self._measured = None
            self._ok(False)
            self._say(reason)
            return
        self._measured = measured
        self._say(measured.summary(), ok=True)
        self._apply_measurement()

    def _apply_measurement(self) -> None:
        """Pre-fill the face's parameters the convention covers; say which it did not."""
        if self._measured is None:
            return
        filled = registry.prefill(self.face_name(), self._measured)
        for name, value in filled.items():
            if name in self._param_fields:
                self._param_fields[name].setValue(float(value))
        unknown = [n for n in self._param_fields if n not in filled]
        if unknown:
            self._say(f'{self._measured.summary()}  —  "{self.face_name()}" also takes '
                      f'{", ".join(unknown)}, which this measurement does not name: type '
                      f'{"them" if len(unknown) > 1 else "it"} from the numbers above.')
        self._ok(any(w.value() > 0 for w in self._param_fields.values()))

    # ---- the answer ---------------------------------------------------------------
    def result_face(self) -> Tuple[str, Dict[str, float], str]:
        """``(face name, its parameters, provenance)`` — what the caller writes."""
        params = {name: round(float(widget.value()), 6)
                  for name, widget in self._param_fields.items()
                  if widget.value() > 0}
        source = self._measured.source if self._measured else ''
        if self._measured is not None and self._measured.exact:
            source = f'{os.path.basename(self.path.text().strip())}: ' + source
        return self.face_name(), params, source

    def hideEvent(self, event):                  # pragma: no cover - GUI lifecycle
        self._timer.stop()
        super().hideEvent(event)
