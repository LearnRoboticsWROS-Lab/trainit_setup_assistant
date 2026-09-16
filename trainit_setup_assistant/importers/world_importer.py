"""Import a Gazebo Classic ``.world`` (SDF) into SceneObjects.

The Gazebo analogue of ``usd_importer`` (Isaac): each ``<model>`` in the world becomes a
SceneObject the ``scene_manager_node`` injects into the MoveIt planning scene. Static
models are CHECKED obstacles the robot must avoid; a non-static model (e.g. the red cube)
defaults to a DYNAMIC manipulation target — the user re-classifies at Step 2.

Two SDF specifics handled:
- **authoritative poses live in ``<world><state>``** (the settled poses after physics), not
  the model's initial ``<pose>``; we prefer the state pose when present.
- a model can carry several ``<collision>`` geometries (a table = a box top + cylinder
  legs); v1 takes the FIRST collision geometry per model and sums the link/collision
  translation onto the model pose. The single graspable object (one box) is exact; multi-
  part obstacles are approximate and refined by hand at Step 2.

The world is in the SIM WORLD frame; the robot is spawned separately, so its base is NOT
in the world. Pass ``base_offset`` = the robot base's world position (e.g. a UR mounted at
``[0, 0, 0.8]``) to express objects in ``base_link``; otherwise they stay world-frame and
the user shifts them at Step 2. Pure stdlib (xml.etree) — no Gazebo/ROS import.
"""

from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple

from ..model import SceneObject
from ..model.enums import SceneObjectCategory, SceneObjectSource, ShapeType
from .base import SceneImporter

# lights and the infinite floor are not planning obstacles
DEFAULT_SKIP = ('sun', 'ground_plane')


def _safe_id(raw: str) -> str:
    name = re.sub(r'[^A-Za-z0-9_]', '_', str(raw).strip())
    return name or 'world_object'


def _floats(text: str, n: int) -> List[float]:
    vals = [float(x) for x in (text or '').split()]
    return (vals + [0.0] * n)[:n]


def _rpy_to_quat(r: float, p: float, y: float) -> List[float]:
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return [sr * cp * cy - cr * sp * sy,   # x
            cr * sp * cy + sr * cp * sy,   # y
            cr * cp * sy - sr * sp * cy,   # z
            cr * cp * cy + sr * sp * sy]   # w


# --- rigid-body maths for the base_link transform (ADR-0011) -----------------
# Quaternions are [x, y, z, w]. These express a world-frame object in the robot base
# frame via the FULL rigid inverse of the base's world pose (the Gazebo mirror of Isaac's
# usd_scene M = M * binv), so both rotation AND translation are absorbed.

def _quat_mul(a: List[float], b: List[float]) -> List[float]:
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return [aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz]


def _quat_conj(q: List[float]) -> List[float]:
    return [-q[0], -q[1], -q[2], q[3]]


def _quat_rotate(q: List[float], v: List[float]) -> List[float]:
    # v' = q * (v,0) * q_conj
    qv = [v[0], v[1], v[2], 0.0]
    r = _quat_mul(_quat_mul(q, qv), _quat_conj(q))
    return [r[0], r[1], r[2]]


def _apply_base_inverse(base_pose6, xyz: List[float], quat: List[float]):
    """Express an object's world (xyz, quat) in the robot base_link frame given the robot
    base's world pose ``base_pose6`` = [x, y, z, R, P, Y] (a 3-list [x,y,z] is padded with a
    zero orientation). None / all-zero => identity (robot at world origin)."""
    if not base_pose6:
        return xyz, quat
    p = [float(v) for v in base_pose6] + [0.0] * (6 - len(base_pose6))
    if all(abs(v) < 1e-12 for v in p[:6]):
        return xyz, quat                       # identity fast path
    q_base_inv = _quat_conj(_rpy_to_quat(p[3], p[4], p[5]))
    d = [xyz[0] - p[0], xyz[1] - p[1], xyz[2] - p[2]]
    return _quat_rotate(q_base_inv, d), _quat_mul(q_base_inv, quat)


def _pose(elem) -> Tuple[List[float], List[float]]:
    """(xyz, rpy) from a <pose> element (default zeros)."""
    p = _floats(elem.text if elem is not None else '', 6)
    return p[:3], p[3:6]


def _collision_shape(geom):
    """(shape, dims, mesh_uri|None, scale) from a <geometry>. A <mesh> keeps its URI + scale
    (resolved to a package:// path later); primitives stay primitives (ADR-0011 B)."""
    box = geom.find('box')
    if box is not None:
        return ShapeType.BOX, _floats(box.findtext('size'), 3), None, [1.0, 1.0, 1.0]
    cyl = geom.find('cylinder')
    if cyl is not None:
        return (ShapeType.CYLINDER,
                [float(cyl.findtext('radius', '0.05')), float(cyl.findtext('length', '0.1'))],
                None, [1.0, 1.0, 1.0])
    sph = geom.find('sphere')
    if sph is not None:
        return ShapeType.SPHERE, [float(sph.findtext('radius', '0.05'))], None, [1.0, 1.0, 1.0]
    mesh = geom.find('mesh')
    if mesh is not None:
        return (ShapeType.MESH, [0.1, 0.1, 0.1], (mesh.findtext('uri') or '').strip(),
                _floats(mesh.findtext('scale') or '1 1 1', 3))
    return None


def _collisions(model):
    """EVERY <collision> of a model (ADR-0011 B: a table -> box top + N cylinder legs, not one
    box), each as (shape, dims, mesh_uri, scale, off_xyz, off_quat) in the MODEL frame — the
    link pose rigidly composed with the collision pose."""
    out = []
    for link in model.findall('link'):
        l_xyz, l_rpy = _pose(link.find('pose'))
        q_link = _rpy_to_quat(*l_rpy)
        for col in link.findall('collision'):
            geom = col.find('geometry')
            if geom is None:
                continue
            spec = _collision_shape(geom)
            if spec is None:
                continue
            c_xyz, c_rpy = _pose(col.find('pose'))
            rc = _quat_rotate(q_link, c_xyz)
            t = [l_xyz[i] + rc[i] for i in range(3)]         # link o collision, model frame
            q = _quat_mul(q_link, _rpy_to_quat(*c_rpy))
            shape, dims, mesh_uri, scale = spec
            out.append((shape, dims, mesh_uri, scale, t, q))
    return out


def _resolve_mesh_uri(uri: str, mesh_package: Optional[str],
                      models_subdir: str = 'models') -> Optional[str]:
    """Resolve an SDF mesh <uri> to a path the scene loader / RViz can load. ``package://`` and
    ``file://`` pass through; a Gazebo ``model://<name>/<rest>`` becomes
    ``package://<mesh_package>/<models_subdir>/<name>/<rest>`` (the cell ships its gazebo models
    under that ROS package — the Gazebo analogue of Isaac's mesh-in-package requirement).
    Returns None when it cannot be resolved (e.g. a model:// with no mesh_package)."""
    u = (uri or '').strip()
    if u.startswith(('package://', 'file://')):
        return u
    if u.startswith('/'):
        return 'file://' + u
    if u.startswith('model://') and mesh_package:
        rest = u[len('model://'):]
        return f'package://{mesh_package}/{models_subdir}/{rest}'
    return None


def model_first_link(world_path, model_name: str, default: str = 'link') -> str:
    """The first ``<link name>`` of a ``<model>`` in a ``.world`` — used as the Gazebo
    LinkAttacher ``object_link`` so TSA does not hardcode 'link' (a cube's link is often
    'link_1'). Returns ``default`` if the model or a named link is not found."""
    try:
        root = ET.parse(str(world_path)).getroot()
        world = root.find('world') if root.tag != 'world' else root
        for m in (world.findall('model') if world is not None else []):
            if m.get('name') == model_name:
                link = m.find('link')
                if link is not None and link.get('name'):
                    return link.get('name')
    except Exception:  # noqa: BLE001
        pass
    return default


def import_world(world_path, default_frame: str = 'base_link',
                 robot_base_world_pose: Optional[List[float]] = None,
                 dynamic: bool = False, category=None, classify=None,
                 skip_names=DEFAULT_SKIP, mesh_package: Optional[str] = None,
                 models_subdir: str = 'models') -> List[SceneObject]:
    """Parse a Gazebo ``.world`` into SceneObjects, expressed in the robot ``base_link`` frame.

    ``robot_base_world_pose`` = the robot base's world pose [x, y, z, R, P, Y] (a 3-list is
    padded with zero orientation): its FULL rigid inverse is applied to every object's pose
    (position AND orientation), the Gazebo mirror of Isaac's base-relative transform
    (ADR-0011). None / all-zero = identity (robot at world origin).

    Object representation (ADR-0011 B): EVERY ``<collision>`` becomes its own SceneObject (a
    table -> box top + N cylinder legs), primitives kept as primitives; a ``<mesh>`` becomes a
    ``shape=mesh`` object whose ``model://`` URI is resolved against ``mesh_package`` (the ROS
    package hosting the gazebo models, under ``models_subdir``), else it degrades to a box.

    ``category`` forces one category on all objects; ``dynamic=True`` is shorthand for
    ``category='dynamic'``; otherwise the default is per ``<static>`` (static -> STATIC
    obstacle, non-static -> DYNAMIC target). ``classify(name) -> category|None`` per model.
    """
    path = Path(world_path)
    root = ET.parse(str(path)).getroot()
    world = root.find('world') if root.tag != 'world' else root
    if world is None:
        raise ValueError(f'no <world> in {path}')

    # authoritative settled poses from <state>
    state_pose = {}
    state = world.find('state')
    if state is not None:
        for m in state.findall('model'):
            xyz, rpy = _pose(m.find('pose'))
            state_pose[m.get('name')] = (xyz, rpy)

    forced = (SceneObjectCategory(category) if category is not None
              else (SceneObjectCategory.DYNAMIC if dynamic else None))

    objects: List[SceneObject] = []
    seen = set()
    for model in world.findall('model'):
        name = model.get('name') or 'model'
        if name in skip_names:
            continue
        cols = _collisions(model)
        if not cols:
            continue
        static = (model.findtext('static', '0').strip().lower() in ('1', 'true'))
        m_xyz, m_rpy = state_pose.get(name) or _pose(model.find('pose'))
        q_model = _rpy_to_quat(*m_rpy)

        cat = forced
        if cat is None and classify is not None:
            picked = classify(name)
            cat = SceneObjectCategory(picked) if picked is not None else None
        if cat is None:
            cat = SceneObjectCategory.STATIC if static else SceneObjectCategory.DYNAMIC

        multi = len(cols) > 1
        for i, (shape, dims, mesh_uri, scale, loff, lquat) in enumerate(cols):
            # model pose o (link o collision), then base_link via the full rigid inverse.
            rc = _quat_rotate(q_model, loff)
            world_xyz = [m_xyz[k] + rc[k] for k in range(3)]
            pos, quat = _apply_base_inverse(robot_base_world_pose, world_xyz,
                                            _quat_mul(q_model, lquat))
            mesh_resource = None
            if shape is ShapeType.MESH:
                resolved = _resolve_mesh_uri(mesh_uri, mesh_package, models_subdir)
                if resolved:
                    mesh_resource = resolved
                else:                        # unresolvable model:// -> conservative box
                    shape, dims = ShapeType.BOX, [0.1, 0.1, 0.1]

            oid = _safe_id(name if not multi else f'{name}_{i}')
            while oid in seen:
                oid += '_'
            seen.add(oid)
            kw = dict(
                id=oid, source=SceneObjectSource.WORLD, shape=shape,
                dims=[float(d) for d in dims], frame=default_frame,
                position=[round(float(v), 6) for v in pos],
                orientation=[round(float(v), 6) for v in quat],
                category=cat,                # drives `dynamic` in the SceneObject validator
            )
            if mesh_resource:
                kw['mesh_resource'] = mesh_resource
                kw['scale'] = [float(s) for s in scale]
            objects.append(SceneObject(**kw))
    return objects


class WorldImporter(SceneImporter):
    def __init__(self, default_frame: str = 'base_link',
                 robot_base_world_pose: Optional[List[float]] = None, dynamic: bool = False,
                 category=None, classify=None, skip_names=DEFAULT_SKIP,
                 mesh_package: Optional[str] = None, models_subdir: str = 'models'):
        self.default_frame = default_frame
        self.robot_base_world_pose = robot_base_world_pose
        self.dynamic = dynamic
        self.category = category
        self.classify = classify
        self.skip_names = skip_names
        self.mesh_package = mesh_package
        self.models_subdir = models_subdir

    def import_objects(self, source) -> List[SceneObject]:
        return import_world(source, self.default_frame, self.robot_base_world_pose,
                            self.dynamic, category=self.category, classify=self.classify,
                            skip_names=self.skip_names, mesh_package=self.mesh_package,
                            models_subdir=self.models_subdir)
