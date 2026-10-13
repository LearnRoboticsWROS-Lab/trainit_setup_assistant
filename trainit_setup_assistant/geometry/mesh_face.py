"""Measure a GRASP FACE on a mesh, in pure numpy (ADR-0018 Decisions 1, 2 and 4).

ADR-0017 made the grasp face the pluggable unit of the candidate ranking and shipped
``DiscFace``, which takes one number: ``expected_radius_m``. That number is the DENOMINATOR
of ``visible_fraction``, the one term the measurement showed wins and replicates on two
seeds. TSA derived it from the object's axis-aligned bounding box -- ``sqrt(x*y/pi)`` --
which for the real capsule gives **28,10 mm against a true 20,95 mm**, because the AABB is
inflated by the pull tab: +34 % on the radius and +80 % on the area. The best-measured
term in the framework was fed a denominator wrong by nearly a factor of two.

This module replaces that approximation with a MEASUREMENT, and adds no concept: the
output is the same few scalars the pipeline already carries, so there is still no runtime
mesh dependency for perception, for the motion runtime or for a generated bundle.

**Why numpy and three fixed views, not a trackball.** Measured during the survey: numpy +
``QImage`` is 93 lines and ZERO new dependencies, 24-77 ms at grasp-target sizes;
``python3-vtk9`` is faster and rotates freely but has no rosdep key and pulls an MPI stack,
which ADR-0003 forbids in ``package.xml``; ``pyqtgraph.opengl`` has the clean rosdep key
and no triangle-level picking, so the Moller-Trumbore cast below gets written anyway and
the dependency buys nothing. The slow case is not the case: a grasp target is small (the
capsule mesh is 252 triangles and rasters in 24 ms), and nobody marks a grasp face on a
whole gripper.

**The unit is explicit, because getting it wrong is SILENT.** This workspace mixes metres
and millimetres -- ``soft_gripper_suction_full.stl`` is mm with a ``scale=0.001`` xacro
while the FAIRINO STLs are metres -- STL carries no unit metadata, and nothing raises: the
survey's own run reported a 51 m2 face and a 4042 mm radius without a word. So
:func:`read_mesh` takes the scale explicitly (``SceneObject.scale`` is the precedent) and
:func:`implausible_reason` refuses a face outside a plausible range rather than letting an
absurd denominator reach a bundle.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

#: The three orthographic views, as (label, the axis looked along). Three fixed views of a
#: small part, which is what a grasp face is. Each is rendered from the POSITIVE side of its
#: axis by default and from the negative side when ``from_positive`` is False -- six
#: directions in all, because three would leave the faces on the far side of the part
#: unclickable, and which three depends on how the CAD happens to be oriented.
VIEWS: Tuple[Tuple[str, int], ...] = (('front  (from +X)', 0),
                                      ('side   (from +Y)', 1),
                                      ('top    (from +Z)', 2))

#: The range a GRASP FACE's size may plausibly fall in, in metres. The same band the
#: RUNTIME refuses on (``trainit_perception``'s ``MIN_FACE_M``/``MAX_FACE_M``), repeated
#: here only as the default: every caller inside TSA passes the registry's own band, so
#: there is no second copy to drift. Outside it the number is not a small face, it is a
#: unit mistake -- and the whole point of Decision 4 is that a unit mistake does not raise.
PLAUSIBLE_RADIUS_M: Tuple[float, float] = (1e-4, 1.0)


class ImplausibleFace(ValueError):
    """A measured face whose size cannot be what the user meant (almost always a unit)."""


@dataclass(frozen=True)
class Mesh:
    """A triangle soup in METRES, plus the two things every step below needs.

    ``tris`` is (T, 3, 3): triangle, vertex, xyz. That is the whole data structure -- an
    STL has no topology to preserve, and the one place adjacency is needed (the coplanar
    flood fill) builds it from welded vertices on demand.
    """

    tris: np.ndarray
    normals: np.ndarray
    areas: np.ndarray
    path: str = ''
    scale: float = 1.0

    @property
    def count(self) -> int:
        return int(len(self.tris))

    def bbox(self) -> Tuple[np.ndarray, np.ndarray]:
        flat = self.tris.reshape(-1, 3)
        return flat.min(axis=0), flat.max(axis=0)

    def extents_m(self) -> np.ndarray:
        lo, hi = self.bbox()
        return hi - lo


@dataclass(frozen=True)
class FaceMeasurement:
    """What a marked face IS, as the scalars the ranking consumes.

    ``radius_m`` is the EQUAL-AREA disc radius (``sqrt(area/pi)``) and not a bounding
    radius, because it is divided into a pixel area: ``visible_fraction``'s denominator is
    ``pi * (f * r / z)^2``, so an equal-area radius makes that denominator the face's real
    area. ``half_a_m`` / ``half_b_m`` are the in-plane half-extents along the face's own
    principal axes, longer first -- the two numbers a non-round face is configured by.
    """

    triangles: int
    area_m2: float
    radius_m: float
    half_a_m: float
    half_b_m: float
    centroid_m: Tuple[float, float, float]
    normal: Tuple[float, float, float]
    source: str = ''
    exact: bool = False

    def summary(self) -> str:
        kind = 'EXACT (B-rep)' if self.exact else f'measured on {self.triangles} triangles'
        return (f'area {self.area_m2 * 1e6:.1f} mm2   equal-area radius '
                f'{self.radius_m * 1000:.3f} mm   half-extents '
                f'{self.half_a_m * 1000:.2f} x {self.half_b_m * 1000:.2f} mm   '
                f'normal ({self.normal[0]:.2f}, {self.normal[1]:.2f}, '
                f'{self.normal[2]:.2f})   [{kind}]')


@dataclass(frozen=True)
class OrthoRender:
    """One orthographic view: a depth buffer, the triangle id per pixel, and the window.

    ``tid`` is why there is no second picking path in the GUI: the raster already resolved
    occlusion, so a click IS a triangle lookup. :func:`ray_hit` stays for the programmatic
    cast (and for the test, which measures a known disc along a known axis).
    """

    axis: int
    depth: np.ndarray
    tid: np.ndarray
    centre: np.ndarray
    half: float
    from_positive: bool = True

    @property
    def size(self) -> int:
        return int(self.tid.shape[0])

    def triangle_at(self, col: int, row: int) -> int:
        """The triangle under an IMAGE pixel, or -1 for background."""
        if not (0 <= row < self.tid.shape[0] and 0 <= col < self.tid.shape[1]):
            return -1
        return int(self.tid[row, col])


# --- reading -----------------------------------------------------------------------------

def read_mesh(path, scale: float = 1.0) -> Mesh:
    """Load an STL (binary or ASCII) or OBJ as triangles in METRES.

    ``scale`` multiplies every coordinate and is the explicit unit (1.0 = the file is in
    metres, 0.001 = millimetres). It is a required thought, not a nicety: see the module
    docstring.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f'no mesh at {p}')
    suffix = p.suffix.lower()
    if suffix == '.obj':
        tris = _read_obj(p)
    elif suffix in ('.stl', '.stlb'):
        tris = _read_stl(p)
    else:
        raise ValueError(
            f'{p.name}: the face picker reads STL and OBJ. A STEP file is read by a '
            f'headless FreeCAD when this machine has one (see `step_faces`); anything '
            f'else converts upstream -- Isaac needs a mesh in the end anyway.')
    if len(tris) == 0:
        raise ValueError(f'{p.name} carries no triangles')
    tris = tris * float(scale)
    nrm, areas = _normals(tris)
    return Mesh(tris=tris, normals=nrm, areas=areas, path=str(p), scale=float(scale))


def _read_stl(path: Path) -> np.ndarray:
    """Binary or ASCII STL -> (T, 3, 3). numpy only: no trimesh, no VTK, no meshio."""
    with open(path, 'rb') as f:
        head = f.read(84)
        if head[:5].lower().lstrip() == b'solid' and b'facet' in f.read(512):
            f.seek(0)
            text = f.read().decode('ascii', 'replace')
            values = [float(x) for line in text.split('\n')
                      if line.strip().startswith('vertex') for x in line.split()[1:4]]
            return np.array(values, dtype=np.float64).reshape(-1, 3, 3)
        count = int(np.frombuffer(head[80:84], '<u4')[0])
        f.seek(84)
        record = np.frombuffer(f.read(50 * count), count=count, dtype=np.dtype([
            ('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        return record['v'].astype(np.float64)


def _read_obj(path: Path) -> np.ndarray:
    """Wavefront OBJ -> (T, 3, 3). Polygons are fan-triangulated; v/vt/vn indices and
    negative (relative) indices are handled; everything else is ignored."""
    verts: List[Tuple[float, float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    for line in path.read_text(errors='replace').splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'v' and len(parts) >= 4:
            verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
        elif parts[0] == 'f' and len(parts) >= 4:
            idx = []
            for token in parts[1:]:
                raw = int(token.split('/')[0])
                idx.append(raw - 1 if raw > 0 else len(verts) + raw)
            for k in range(1, len(idx) - 1):          # fan from the first vertex
                faces.append((idx[0], idx[k], idx[k + 1]))
    if not faces:
        return np.zeros((0, 3, 3))
    return np.asarray(verts, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]


def _normals(tris: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit face normals + triangle areas (half the cross product's magnitude)."""
    cross = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    mag = np.linalg.norm(cross, axis=1, keepdims=True)
    return cross / np.maximum(mag, 1e-18), mag[:, 0] * 0.5


# --- picking and the face ----------------------------------------------------------------

def ray_hit(tris: np.ndarray, origin, direction) -> int:
    """Vectorised Moller-Trumbore over every triangle -> the NEAREST hit index, or -1.

    One pass of numpy over the whole array, which is why no acceleration structure is
    needed at grasp-target sizes, and why `pyqtgraph` would have bought nothing: its
    ``GLViewWidget`` has no triangle-level picking, so this function gets written either
    way (ADR-0018 Decision 2).
    """
    origin = np.asarray(origin, float)
    direction = np.asarray(direction, float)
    e1, e2 = tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0]
    h = np.cross(direction, e2)
    a = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(a) > 1e-14
    inv = np.zeros_like(a)
    inv[ok] = 1.0 / a[ok]
    s = origin - tris[:, 0]
    u = inv * np.einsum('ij,ij->i', s, h)
    q = np.cross(s, e1)
    v = inv * np.einsum('j,ij->i', direction, q)
    t = inv * np.einsum('ij,ij->i', e2, q)
    ok &= (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 1e-9)
    return int(np.argmin(np.where(ok, t, np.inf))) if ok.any() else -1


def _vertex_ids(tris: np.ndarray, tol: float) -> np.ndarray:
    """Weld vertices on a ``tol`` grid -> (T, 3) integer vertex ids.

    An STL repeats every vertex per triangle, so adjacency does not exist in the file and
    has to be recovered. Rounding is the recovery, and the tolerance is the same one the
    plane test uses -- one number for "the same point" and "the same plane".
    """
    keys = np.round(tris.reshape(-1, 3) / max(tol, 1e-12)).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.reshape(-1, 3)


def coplanar_region(mesh: Mesh, seed: int, angle_deg: float = 12.0,
                    pos_tol: float = 1e-4) -> List[int]:
    """Flood-fill the face containing ``seed``: shared-vertex adjacency, within a normal
    angle AND the same plane offset.

    Both tests are needed. The angle alone merges the far side of a thin part (two parallel
    faces 1 mm apart are both "within 12 degrees"); the offset alone merges a tangent
    cylinder ring. Adjacency keeps the region CONNECTED, so the cap of one capsule in a
    multi-part STL is not merged with the cap of the next.
    """
    cos_tol = float(np.cos(np.deg2rad(angle_deg)))
    seed_n = mesh.normals[seed]
    plane_d = float(seed_n @ mesh.tris[seed, 0])
    near = ((mesh.normals @ seed_n >= cos_tol)
            & (np.abs(mesh.tris[:, 0] @ seed_n - plane_d) < pos_tol))
    vid = _vertex_ids(mesh.tris, pos_tol)
    # vertex -> the triangles touching it, built once: the prototype rescanned every
    # triangle per popped cell, which is quadratic and shows up on a real part.
    by_vertex: Dict[int, List[int]] = {}
    for tri in np.flatnonzero(near):
        for v in vid[tri]:
            by_vertex.setdefault(int(v), []).append(int(tri))
    seen = {int(seed)}
    stack = [int(seed)]
    while stack:
        cur = stack.pop()
        for v in vid[cur]:
            for nxt in by_vertex.get(int(v), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return sorted(seen)


def measure_region(mesh: Mesh, cells: Sequence[int], *, source: str = '') -> FaceMeasurement:
    """The marked triangles -> the scalars the ranking consumes.

    The half-extents come from a PCA of the region's vertices IN the face plane, which is
    the same decomposition ``plane_ransac`` performs on the depth at run time -- so the
    number configured here and the number measured there are the same quantity, and
    ``radius_ratio`` (or a rectangular face's extent ratios) means what it says.
    """
    cells = [int(c) for c in cells]
    if not cells:
        raise ValueError('no triangles marked')
    area = float(mesh.areas[cells].sum())
    weights = mesh.areas[cells, None]
    centroid = (weights * mesh.tris[cells].mean(axis=1)).sum(axis=0) / max(area, 1e-18)
    normal = (weights * mesh.normals[cells]).sum(axis=0)
    normal = normal / max(float(np.linalg.norm(normal)), 1e-18)
    pts = mesh.tris[cells].reshape(-1, 3) - centroid
    pts = pts - np.outer(pts @ normal, normal)           # project into the face plane
    basis = plane_basis(normal)
    local = pts @ basis.T                                 # (N, 2)
    if len(local) >= 3:
        # principal axes of the in-plane spread; `eigh` on the 2x2 covariance is exact and
        # needs no SVD. The extent is the MAX |projection|: a CAD face has no outliers to
        # be robust against, unlike the depth image the runtime estimators are written for.
        _, vecs = np.linalg.eigh(np.cov(local.T) + np.eye(2) * 1e-18)
        proj = np.abs(local @ vecs)
        halves = sorted((float(proj[:, 0].max()), float(proj[:, 1].max())), reverse=True)
    else:                                                 # pragma: no cover - degenerate
        halves = [0.0, 0.0]
    return FaceMeasurement(
        triangles=len(cells), area_m2=area,
        radius_m=float(np.sqrt(area / np.pi)),
        half_a_m=halves[0], half_b_m=halves[1],
        centroid_m=tuple(float(v) for v in centroid),
        normal=tuple(float(v) for v in normal),
        source=source)


def plane_basis(normal: np.ndarray) -> np.ndarray:
    """Two orthonormal in-plane axes for a unit normal (any pair: PCA picks the meaning).

    Shared with `step_faces`, so a B-rep face's boundary and a tessellated face's vertices
    are projected into their plane by the SAME two lines -- two measurements that are
    compared to each other must not be computed two ways.
    """
    helper = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(normal, helper)
    u /= max(float(np.linalg.norm(u)), 1e-18)
    return np.vstack([u, np.cross(normal, u)])


def implausible_reason(measurement: FaceMeasurement, *, scale: float = 1.0,
                       band: Optional[Tuple[float, float]] = None) -> Optional[str]:
    """Why this face cannot be what the user meant, or ``None`` (ADR-0018 Decision 4).

    Almost always the unit. The sentence therefore names the unit FIRST, with the scale
    that would fix it, because "radius 4042 mm" on its own reads as a geometry problem and
    is a configuration one. ``band`` is the (min, max) the deployed runtime accepts --
    asked of the face registry by the caller, so the picker refuses exactly what the node
    would refuse rather than a range of its own.
    """
    lo, hi = band or PLAUSIBLE_RADIUS_M
    r = measurement.radius_m
    if lo <= r <= hi:
        return None
    guess = 'millimetres (scale 0.001)' if r > hi else 'metres (scale 1.0)'
    return (f'this face measures {r * 1000:.1f} mm equal-area radius '
            f'({measurement.area_m2:.4f} m2) at scale {scale:g}, which is outside the '
            f'plausible range {lo * 1000:g}..{hi * 1000:g} mm for something a tool picks. '
            f'A mesh file carries NO unit, so check it first -- this looks like a file in '
            f'{guess}. Refusing to write it: an absurd denominator in visible_fraction '
            f'does not raise anywhere downstream, it just ranks the wrong candidate first.')


def refuse_if_implausible(measurement: FaceMeasurement, *, scale: float = 1.0,
                          band: Optional[Tuple[float, float]] = None) -> FaceMeasurement:
    """The measurement back, or :class:`ImplausibleFace` -- the one-call form of the check."""
    reason = implausible_reason(measurement, scale=scale, band=band)
    if reason:
        raise ImplausibleFace(reason)
    return measurement


# --- the three views ---------------------------------------------------------------------

def raster(mesh: Mesh, axis: int, size: int = 320, pad: float = 1.06,
           from_positive: bool = True) -> OrthoRender:
    """Orthographic depth buffer along ``axis``: per-triangle bbox loop, no GL, no new dep.

    ``from_positive`` puts the viewer on the +axis side, so the nearest surface is the one
    with the LARGEST coordinate -- a "top" view shows the top. Flipped, the same raster
    reaches the three faces the other three views cannot, which is why the dialog offers it
    rather than leaving half a part unclickable.

    The whole mesh is framed (not the face), so the three views agree about the part and a
    click means the same thing in each. 24 ms on the 252-triangle capsule, 0,3 s on a
    4334-triangle one.
    """
    u, v = [i for i in range(3) if i != axis]
    flat = mesh.tris.reshape(-1, 3)
    lo, hi = flat.min(axis=0), flat.max(axis=0)
    centre = (lo + hi) / 2.0
    half = max(float((hi - lo).max()) / 2.0, 1e-9) * pad
    px = (mesh.tris[:, :, u] - (centre[u] - half)) * ((size - 1) / (2 * half))
    py = (mesh.tris[:, :, v] - (centre[v] - half)) * ((size - 1) / (2 * half))
    # one sign, applied to the depth only: "nearest" is then always the smallest value,
    # so the comparison below stays a single `<` instead of a branch per pixel.
    pz = mesh.tris[:, :, axis] * (-1.0 if from_positive else 1.0)
    depth = np.full((size, size), np.inf, np.float32)
    tid = np.full((size, size), -1, np.int32)
    for i in range(len(mesh.tris)):
        x, y, z = px[i], py[i], pz[i]
        x0, x1 = max(int(np.floor(x.min())), 0), min(int(np.ceil(x.max())) + 1, size)
        y0, y1 = max(int(np.floor(y.min())), 0), min(int(np.ceil(y.max())) + 1, size)
        if x1 <= x0 or y1 <= y0:
            continue
        det = ((y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2]))
        if abs(det) < 1e-12:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
        w0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / det
        w1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / det
        w2 = 1.0 - w0 - w1
        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not inside.any():
            continue
        zz = w0 * z[0] + w1 * z[1] + w2 * z[2]
        window = depth[y0:y1, x0:x1]
        win = inside & (zz < window)
        window[win] = zz[win]
        tid[y0:y1, x0:x1][win] = i
    return OrthoRender(axis=axis, depth=depth, tid=tid, centre=centre, half=half,
                       from_positive=from_positive)


def ortho_views(mesh: Mesh, size: int = 320,
                from_positive: bool = True) -> Tuple[OrthoRender, ...]:
    """The three fixed views, in :data:`VIEWS` order."""
    return tuple(raster(mesh, axis, size=size, from_positive=from_positive)
                 for _, axis in VIEWS)


def shade(mesh: Mesh, render: OrthoRender, highlight: Iterable[int] = ()) -> np.ndarray:
    """(H, W, 3) uint8 for a ``QImage``: Lambert grey, the marked face in green.

    The highlight is the whole feedback loop of the picker -- the user sees the region the
    flood fill actually took, which is how a 12-degree angle tolerance swallowing a fillet
    becomes visible instead of silently inflating the area.
    """
    tid = render.tid
    lit = np.abs(mesh.normals[np.clip(tid, 0, None), render.axis])
    grey = (60 + 185 * lit).astype(np.uint8)
    img = np.stack([grey, grey, grey], axis=-1)
    marked = np.zeros(mesh.count + 1, bool)
    for c in highlight:
        marked[int(c)] = True
    hit = marked[np.clip(tid, 0, None)] & (tid >= 0)
    img[hit] = np.stack([(grey[hit] * 0.25).astype(np.uint8),
                         np.minimum(255, 90 + grey[hit]).astype(np.uint8),
                         (grey[hit] * 0.35).astype(np.uint8)], axis=-1)
    img[tid < 0] = 34                                     # the live tuner's own backdrop
    return np.ascontiguousarray(img)
