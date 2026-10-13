"""STEP, as an OPTIONAL exact path behind a probe (ADR-0018 Decision 3).

No STEP reader survives TrainIt's rosdep/apt-only installation rule (ADR-0003):
``python3-occ`` and ``python3-pythonocc-core`` are not in jammy, and FreeCAD's python
module is not importable from the system interpreter. Isaac wants a mesh in the end anyway
and this workspace already holds 309 STL files, so the picker's normal path is the
tessellated one in :mod:`~trainit_setup_assistant.geometry.mesh_face`.

The exception is worth taking because it costs nothing. A headless FreeCAD is EXACT:
``freecadcmd`` reads the capsule STEP in 0,011 s and hands back named B-rep faces with
their true area, axis and radius -- no tessellation, no flood fill, no angle tolerance. So
STEP is an optional exact path behind a probe, the way ``pxr`` is already probed lazily,
and NEVER a declared dependency: a customer's STEP converts upstream, and a customer who
happens to have FreeCAD gets the better answer for free. When the probe fails the picker
still works on STL and says so once, calmly.

A STEP file also removes the unit question that Decision 4 exists for: the format carries
its own unit and FreeCAD normalises to millimetres, so :data:`STEP_SCALE` is a known
conversion rather than a guess.

**How the probe finds it.** ``freecadcmd`` on ``PATH`` (an apt or distribution install), or
``TRAINIT_FREECADCMD`` pointing at an executable -- including an AppImage or its extracted
``AppRun``, which dispatch to their inner ``freecadcmd`` when it is named as the first
argument. Nothing is installed and nothing is downloaded.
"""

from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .mesh_face import FaceMeasurement, plane_basis

#: STEP is normalised to millimetres by FreeCAD, so this is a FACT and not a default.
STEP_SCALE = 0.001

#: The env var that points at a FreeCAD the probe would not otherwise find (an AppImage).
FREECAD_ENV = 'TRAINIT_FREECADCMD'

_MARKER = 'TRAINIT_FACE '


@dataclass(frozen=True)
class StepFace:
    """One B-rep face, with the numbers FreeCAD computed rather than approximated."""

    index: int
    kind: str
    area_m2: float
    radius_m: float
    half_a_m: float
    half_b_m: float
    centre_m: Tuple[float, float, float]
    axis: Tuple[float, float, float]
    exact_radius_m: Optional[float] = None       # a cylinder's TRUE radius, when it is one

    def label(self) -> str:
        extra = (f'  R={self.exact_radius_m * 1000:.3f} mm'
                 if self.exact_radius_m else '')
        return (f'face[{self.index}] {self.kind:<9} area {self.area_m2 * 1e6:9.2f} mm2  '
                f'r_eq {self.radius_m * 1000:7.3f} mm{extra}')

    def as_measurement(self) -> FaceMeasurement:
        """The same scalars the mesh path produces, flagged ``exact``."""
        return FaceMeasurement(
            triangles=0, area_m2=self.area_m2,
            radius_m=self.exact_radius_m or self.radius_m,
            half_a_m=self.half_a_m, half_b_m=self.half_b_m,
            centroid_m=self.centre_m, normal=self.axis,
            source=f'FreeCAD B-rep {self.kind} face[{self.index}]', exact=True)


def freecad_command() -> Optional[List[str]]:
    """The argv prefix that runs a headless FreeCAD, or ``None``.

    Probed, never required. An AppImage (or the ``AppRun`` of an extracted one) runs its
    inner ``freecadcmd`` when that name is passed as the first argument, which is how a
    machine with no packaged FreeCAD still gets the exact path.
    """
    env = os.environ.get(FREECAD_ENV, '').strip()
    if env:
        argv = shlex.split(env)
        if not argv:
            return None
        exe = argv[0] if os.path.isfile(argv[0]) else (shutil.which(argv[0]) or '')
        if not exe:
            return None
        argv[0] = exe
        base = os.path.basename(exe).lower()
        if len(argv) == 1 and 'freecadcmd' not in base:
            argv.append('freecadcmd')            # AppImage / AppRun dispatch
        return argv
    found = shutil.which('freecadcmd')
    return [found] if found else None


def freecad_available() -> bool:
    return freecad_command() is not None


# freecadcmd does NOT forward extra argv to the script -- it treats every further
# argument as another FILE to execute, which fails with "File does not exist" on a path
# that is data. So the paths are BAKED into the generated script with ``repr``, which is
# also the only quoting that survives a filename with a space in it.
_SCRIPT = r"""
import json
import Part

src, dst, marker = %(src)r, %(dst)r, %(marker)r
shape = Part.Shape()
shape.read(src)
for i, f in enumerate(shape.Faces):
    row = {'index': i, 'kind': f.Surface.__class__.__name__, 'area_mm2': float(f.Area)}
    c = f.CenterOfMass
    row['centre_mm'] = [float(c.x), float(c.y), float(c.z)]
    axis = getattr(f.Surface, 'Axis', None)
    row['axis'] = [float(axis.x), float(axis.y), float(axis.z)] if axis else [0.0, 0.0, 0.0]
    radius = getattr(f.Surface, 'Radius', None)
    if radius is not None:
        row['radius_mm'] = float(radius)
    try:
        pts = [[float(p.x), float(p.y), float(p.z)]
               for p in f.OuterWire.discretize(Number=48)]
    except Exception:
        pts = [[float(v.Point.x), float(v.Point.y), float(v.Point.z)] for v in f.Vertexes]
    row['boundary_mm'] = pts
    print(marker + json.dumps(row))
if dst:
    import Mesh
    Mesh.Mesh(shape.tessellate(0.1)).write(dst)
    print(marker + json.dumps({'tessellated': dst}))
"""


def read_step_faces(path, *, tessellate_to=None,
                    timeout_s: float = 120.0) -> Tuple[Tuple[StepFace, ...], Optional[str]]:
    """Exact B-rep faces of a STEP file, and optionally a tessellation of it.

    Returns ``(faces, mesh_path)``. ``mesh_path`` is the STL written to
    ``tessellate_to`` -- the picker renders its three views from it, so the SAME click
    gesture works for a STEP file, while the numbers come from the B-rep beside it.

    Raises ``RuntimeError`` when no FreeCAD is on this machine (the caller probes first) or
    when it fails; never raises on a face it does not understand -- an unrecognised surface
    kind is reported with its area and nothing more.
    """
    argv = freecad_command()
    if argv is None:
        raise RuntimeError(
            f'no headless FreeCAD found: put `freecadcmd` on PATH, or set {FREECAD_ENV} '
            f'to a FreeCAD executable or AppImage. STEP is an OPTIONAL exact path '
            f'(ADR-0018 Decision 3) -- the picker works on STL and OBJ without it.')
    src = str(Path(path).expanduser())
    with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as script:
        script.write(_SCRIPT % {'src': src, 'dst': str(tessellate_to or ''),
                                'marker': _MARKER})
        script_path = script.name
    try:
        proc = subprocess.run(argv + [script_path], capture_output=True, text=True,
                              timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:            # pragma: no cover - needs FreeCAD
        raise RuntimeError(f'FreeCAD did not answer within {timeout_s:g} s') from exc
    finally:
        os.unlink(script_path)
    rows = [json.loads(line[len(_MARKER):]) for line in proc.stdout.splitlines()
            if line.startswith(_MARKER)]
    if not rows:                                         # pragma: no cover - needs FreeCAD
        tail = (proc.stderr or proc.stdout or '').strip().splitlines()[-3:]
        raise RuntimeError(f'FreeCAD read no face from {Path(src).name}: '
                           f'{" | ".join(tail) or "no output"}')
    mesh_path = None
    faces: List[StepFace] = []
    for row in rows:
        if 'tessellated' in row:
            mesh_path = row['tessellated']
            continue
        faces.append(_step_face(row))
    return tuple(faces), mesh_path


def _step_face(row: dict) -> StepFace:
    area_m2 = float(row['area_mm2']) * STEP_SCALE ** 2
    axis = tuple(float(v) for v in row.get('axis', (0.0, 0.0, 0.0)))
    centre = tuple(float(v) * STEP_SCALE for v in row.get('centre_mm', (0.0, 0.0, 0.0)))
    pts = np.asarray(row.get('boundary_mm') or [], float) * STEP_SCALE
    half_a = half_b = 0.0
    if len(pts) >= 3 and abs(np.linalg.norm(axis) - 1.0) < 1e-6:
        local = (pts - np.asarray(centre)) @ plane_basis(np.asarray(axis, float)).T
        _, vecs = np.linalg.eigh(np.cov(local.T) + np.eye(2) * 1e-18)
        proj = np.abs(local @ vecs)
        half_a, half_b = sorted((float(proj[:, 0].max()), float(proj[:, 1].max())),
                                reverse=True)
    return StepFace(
        index=int(row['index']), kind=str(row['kind']), area_m2=area_m2,
        radius_m=float(np.sqrt(area_m2 / np.pi)) if area_m2 > 0 else 0.0,
        half_a_m=half_a, half_b_m=half_b, centre_m=centre, axis=axis,
        exact_radius_m=(float(row['radius_mm']) * STEP_SCALE
                        if row.get('radius_mm') else None))


def match_measurement(measurement: FaceMeasurement, faces: Sequence[StepFace], *,
                      tol_m: float = 0.002) -> Optional[StepFace]:
    """The B-rep face a TESSELLATED measurement belongs to, or ``None``.

    So a click on the three views of a STEP file's tessellation can report the EXACT area
    instead of the flood fill's: match on the centre (within ``tol_m``), the normal and the
    area, and refuse to guess when nothing agrees -- a wrong match would silently publish
    another face's radius, which is the failure mode this whole ADR is about.
    """
    centre = np.asarray(measurement.centroid_m, float)
    normal = np.asarray(measurement.normal, float)
    best, best_d = None, float('inf')
    for face in faces:
        axis = np.asarray(face.axis, float)
        if np.linalg.norm(axis) > 0 and abs(float(axis @ normal)) < 0.985:
            continue
        if face.area_m2 <= 0 or not (0.75 <= measurement.area_m2 / face.area_m2 <= 1.3):
            continue
        d = float(np.linalg.norm(np.asarray(face.centre_m, float) - centre))
        if d < best_d:
            best, best_d = face, d
    return best if best is not None and best_d <= tol_m else None
