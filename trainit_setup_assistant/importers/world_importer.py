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


def _pose(elem) -> Tuple[List[float], List[float]]:
    """(xyz, rpy) from a <pose> element (default zeros)."""
    p = _floats(elem.text if elem is not None else '', 6)
    return p[:3], p[3:6]


def _first_collision(model) -> Optional[Tuple[ShapeType, List[float], List[float]]]:
    """(shape, dims, translation-offset) of the model's first <collision> geometry, with
    the link+collision translations summed onto the offset."""
    for link in model.findall('link'):
        link_xyz, _ = _pose(link.find('pose'))
        for col in link.findall('collision'):
            col_xyz, _ = _pose(col.find('pose'))
            geom = col.find('geometry')
            if geom is None:
                continue
            off = [link_xyz[i] + col_xyz[i] for i in range(3)]
            box = geom.find('box')
            if box is not None:
                return ShapeType.BOX, _floats(box.findtext('size'), 3), off
            cyl = geom.find('cylinder')
            if cyl is not None:
                r = float(cyl.findtext('radius', '0.05'))
                h = float(cyl.findtext('length', '0.1'))
                return ShapeType.CYLINDER, [r, h], off
            sph = geom.find('sphere')
            if sph is not None:
                return ShapeType.SPHERE, [float(sph.findtext('radius', '0.05'))], off
            mesh = geom.find('mesh')
            if mesh is not None:
                # meshes need a package:// path the scene loader can resolve; a model://
                # URI can't be, so emit a conservative unit box and warn via the caller.
                return ShapeType.BOX, [0.1, 0.1, 0.1], off
    return None


def import_world(world_path, default_frame: str = 'base_link',
                 base_offset: Optional[List[float]] = None,
                 dynamic: bool = False, category=None, classify=None,
                 skip_names=DEFAULT_SKIP) -> List[SceneObject]:
    """Parse a Gazebo ``.world`` into SceneObjects.

    ``base_offset``: the robot base's world position (subtracted so objects are expressed in
    ``base_link``); omit to keep the sim world frame. ``category`` forces one category on all
    objects; ``dynamic=True`` is shorthand for ``category='dynamic'``; otherwise the default
    is per ``<static>`` (static -> STATIC obstacle, non-static -> DYNAMIC target).
    ``classify(name) -> category|None`` overrides per model.
    """
    path = Path(world_path)
    root = ET.parse(str(path)).getroot()
    world = root.find('world') if root.tag != 'world' else root
    if world is None:
        raise ValueError(f'no <world> in {path}')

    off = list(base_offset) if base_offset else [0.0, 0.0, 0.0]

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
        geom = _first_collision(model)
        if geom is None:
            continue
        shape, dims, col_off = geom
        static = (model.findtext('static', '0').strip().lower() in ('1', 'true'))
        m_xyz, m_rpy = state_pose.get(name) or _pose(model.find('pose'))
        pos = [m_xyz[i] + col_off[i] - off[i] for i in range(3)]

        cat = forced
        if cat is None and classify is not None:
            picked = classify(name)
            cat = SceneObjectCategory(picked) if picked is not None else None
        if cat is None:
            cat = SceneObjectCategory.STATIC if static else SceneObjectCategory.DYNAMIC

        oid = _safe_id(name)
        while oid in seen:
            oid += '_'
        seen.add(oid)
        objects.append(SceneObject(
            id=oid,
            source=SceneObjectSource.WORLD,
            shape=shape,
            dims=[float(d) for d in dims],
            frame=default_frame,
            position=[round(float(v), 6) for v in pos],
            orientation=_rpy_to_quat(*m_rpy),
            category=cat,                    # drives `dynamic` in the SceneObject validator
        ))
    return objects


class WorldImporter(SceneImporter):
    def __init__(self, default_frame: str = 'base_link',
                 base_offset: Optional[List[float]] = None, dynamic: bool = False,
                 category=None, classify=None, skip_names=DEFAULT_SKIP):
        self.default_frame = default_frame
        self.base_offset = base_offset
        self.dynamic = dynamic
        self.category = category
        self.classify = classify
        self.skip_names = skip_names

    def import_objects(self, source) -> List[SceneObject]:
        return import_world(source, self.default_frame, self.base_offset, self.dynamic,
                            category=self.category, classify=self.classify,
                            skip_names=self.skip_names)
