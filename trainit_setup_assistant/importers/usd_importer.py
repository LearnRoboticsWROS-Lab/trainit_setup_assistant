"""Import a USD stage into SceneObjects (axis-aligned bounding boxes).

Each geometric prim becomes a box collision object sized to its world AABB. CRUCIAL
for a digital twin: USD positions are in the STAGE WORLD frame, but MoveIt plans in the
robot BASE frame. If the robot is mounted somewhere in the cell (e.g. on a table at
z=1.05), raw world coords land the objects far from the robot in RViz. So we align each
object into the robot base frame (via the robot prim's inverse world transform) and skip
the robot's own prims. pxr (usd-core) is imported lazily and degrades gracefully.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional

from ..model import SceneObject
from ..model.enums import SceneObjectCategory, ShapeType, SceneObjectSource
from .base import SceneImporter


def usd_available() -> bool:
    try:
        import pxr  # noqa: F401
        return True
    except ImportError:
        return False


def _safe_id(raw: str) -> str:
    name = str(raw).strip('/').replace('/', '_')
    name = re.sub(r'[^A-Za-z0-9_]', '_', name)
    return name or 'usd_object'


def _guess_base_prim(stage, base_frame: str, robot_hint: Optional[str]) -> Optional[str]:
    """Find the prim to align to: a link named base_frame, else the robot root Xform."""
    from pxr import UsdGeom
    # 1) a prim named exactly like the base frame (e.g. base_link)
    for prim in stage.Traverse():
        if prim.GetName() == base_frame:
            return str(prim.GetPath())
    # 2) the shallowest Xformable whose name contains the robot hint (the robot root)
    if robot_hint:
        hint = robot_hint.lower()
        for prim in stage.Traverse():  # depth-first, top-down -> shallowest first
            if hint in prim.GetName().lower() and prim.IsA(UsdGeom.Xformable):
                return str(prim.GetPath())
    return None


def import_usd(usd_path, default_frame: str = 'base_link', dynamic: bool = False,
               base_prim: Optional[str] = None,
               robot_hint: Optional[str] = None,
               category=None, classify=None) -> List[SceneObject]:
    """Parse a USD stage into box SceneObjects, aligned to the robot base frame.

    ``base_prim``: the USD path of the robot root/base to align to (objects are
    expressed relative to it, and prims under it are skipped). If omitted it is
    auto-detected from ``default_frame`` / ``robot_hint``. With no base prim, positions
    stay in the stage world frame (and a note is attached to each object id).

    ``category``: the :class:`SceneObjectCategory` (static | actuated | dynamic) to
    assign to every imported object (default ``static``: imported prims are obstacles
    the user then re-classifies in the wizard). ``dynamic=True`` is a shorthand for
    ``category='dynamic'``. ``classify(name, path) -> category|None`` optionally
    overrides the category per prim (return None to keep the default) — this is how a
    cell can auto-tag e.g. ``bottle_*``/``crate_*`` as dynamic on load.
    """
    try:
        from pxr import Usd, UsdGeom, Gf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError('USD import needs pxr (pip install usd-core)') from exc

    path = Path(usd_path)
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f'cannot open USD stage: {path}')

    if base_prim is None:
        base_prim = _guess_base_prim(stage, default_frame, robot_hint)

    base_inv = None
    if base_prim:
        bp = stage.GetPrimAtPath(base_prim)
        if bp and bp.IsValid():
            world = UsdGeom.Xformable(bp).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default())
            base_inv = world.GetInverse()
        else:
            base_prim = None  # invalid path -> world frame

    default_cat = (SceneObjectCategory(category) if category is not None
                   else (SceneObjectCategory.DYNAMIC if dynamic
                         else SceneObjectCategory.STATIC))

    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                             [UsdGeom.Tokens.default_, UsdGeom.Tokens.render])
    objects: List[SceneObject] = []
    seen = set()
    base_prefix = (base_prim.rstrip('/') + '/') if base_prim else None
    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        # skip the robot's own prims (don't import the robot as obstacles)
        if base_prim and (prim_path == base_prim or
                          (base_prefix and prim_path.startswith(base_prefix))):
            continue
        if not prim.IsA(UsdGeom.Gprim):
            continue
        try:
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:  # noqa: BLE001
            continue
        if rng.IsEmpty():
            continue
        size = rng.GetSize()
        mid = rng.GetMidpoint()
        if base_inv is not None:
            mid = base_inv.Transform(Gf.Vec3d(mid[0], mid[1], mid[2]))
        oid = _safe_id(prim.GetName() or prim.GetPath())
        if oid in seen:
            oid = _safe_id(prim.GetPath())
        seen.add(oid)
        cat = default_cat
        if classify is not None:
            picked = classify(prim.GetName(), prim_path)
            if picked is not None:
                cat = SceneObjectCategory(picked)
        objects.append(SceneObject(
            id=oid,
            source=SceneObjectSource.USD,
            shape=ShapeType.BOX,                 # imported as world AABB
            dims=[float(size[0]), float(size[1]), float(size[2])],
            frame=default_frame,
            position=[float(mid[0]), float(mid[1]), float(mid[2])],
            category=cat,                        # drives `dynamic` in the validator
        ))
    return objects


class UsdImporter(SceneImporter):
    def __init__(self, default_frame: str = 'base_link', dynamic: bool = False,
                 base_prim: Optional[str] = None, robot_hint: Optional[str] = None,
                 category=None, classify=None):
        self.default_frame = default_frame
        self.dynamic = dynamic
        self.base_prim = base_prim
        self.robot_hint = robot_hint
        self.category = category
        self.classify = classify

    def import_objects(self, source) -> List[SceneObject]:
        return import_usd(source, self.default_frame, self.dynamic,
                          self.base_prim, self.robot_hint,
                          category=self.category, classify=self.classify)
