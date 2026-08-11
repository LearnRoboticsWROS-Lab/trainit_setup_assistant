"""USD -> cell scene (mesh objects), the reliable Step-2 path.

The USD gives the cell prim NAMES + POSES (verified to reproduce the hand-made baseline
when expressed in the robot base frame), but NOT the ROS collision-mesh paths — that
mapping is external knowledge (what scene_from_usd.py hardcodes). So we read prims +
base-aligned poses from the USD, GROUP them by name pattern (bottle_0_0..bottle_3_4 ->
"bottle"), and AUTO-SUGGEST a collision STL per group by scanning the mesh package's
meshes/ dir. The user confirms/edits the small mapping table, then the scene is built.
"""

from __future__ import annotations

import os
import re
from typing import Dict, List, Optional

from .usd_importer import _guess_base_prim, usd_available  # reuse the base-prim heuristic

_PHYSICS_TYPES = {'PhysicsScene', 'PhysicsCollisionGroup'}
_DYNAMIC_HINTS = ('bottle', 'crate', 'part', 'workpiece', 'box_part')


def group_key(name: str) -> str:
    """Collapse indexed instances to a pattern: bottle_0_0 -> bottle, crate_00 -> crate."""
    return re.sub(r'(_\d+)+$', '', name) or name


def _resolve_meshes_dir(mesh_pkg: str, hint_path: Optional[str] = None) -> Optional[str]:
    """Find <mesh_pkg>/meshes: prefer the installed share, else a src sibling of hint_path."""
    try:
        from ament_index_python.packages import get_package_share_directory
        d = os.path.join(get_package_share_directory(mesh_pkg), 'meshes')
        if os.path.isdir(d):
            return d
    except Exception:  # noqa: BLE001
        pass
    if hint_path:
        # the mesh pkg is a sibling of the base moveit_config package
        sibling = os.path.dirname(os.path.abspath(hint_path))
        d = os.path.join(sibling, mesh_pkg, 'meshes')
        if os.path.isdir(d):
            return d
    return None


def suggest_mesh(group: str, stls: List[str]) -> str:
    """Best-matching collision STL (relative to meshes/) for a group key, or ''."""
    g = group.lower()
    matches = [s for s in stls
               if s.lower().startswith(g + '/') or g in os.path.basename(s).lower()]
    if not matches:
        return ''
    coll = [s for s in matches if 'collision' in s.lower()]
    pool = coll or matches
    pool.sort(key=lambda s: (0 if 'base_collision' in s.lower() else 1, len(s)))
    return pool[0]


def read_cell_prims(usd_path: str, base_prim: Optional[str] = None,
                    robot_hint: Optional[str] = None) -> List[dict]:
    """Top-level cell prims (excluding robot, physics, unresolvable) with base-aligned
    poses. Each: {name, group, position[xyz], orientation[xyzw]}."""
    if not usd_available():
        raise RuntimeError('USD import needs pxr / usd-core')
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(usd_path, load=Usd.Stage.LoadAll)
    world = stage.GetPrimAtPath('/World')
    if not world or not world.IsValid():
        world = stage.GetPseudoRoot()

    bp = base_prim or _guess_base_prim(stage, robot_hint)
    binv = None
    robot_root = None
    if bp:
        bprim = stage.GetPrimAtPath(bp)
        if bprim and bprim.IsValid():
            binv = UsdGeom.Xformable(bprim).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()).GetInverse()
        parts = bp.strip('/').split('/')
        if len(parts) >= 2:
            robot_root = '/' + '/'.join(parts[:2])   # /World/<robot>

    # local-frame extents, so a primitive collision shape can be derived per prim
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])

    out: List[dict] = []
    for p in world.GetChildren():
        path = p.GetPath().pathString
        if robot_root and path == robot_root:
            continue
        if p.GetTypeName() in _PHYSICS_TYPES:
            continue
        if not p.IsA(UsdGeom.Xformable):
            continue
        try:
            M = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            if binv is not None:
                M = M * binv
        except Exception:  # noqa: BLE001
            continue
        t = M.ExtractTranslation()
        q = M.ExtractRotationQuat()
        im = q.GetImaginary()
        origin = [round(t[0], 4), round(t[1], 4), round(t[2], 4)]
        dims = [0.1, 0.1, 0.1]
        center = list(origin)
        try:
            rng = bbox.ComputeUntransformedBound(p).ComputeAlignedRange()
            if not rng.IsEmpty():
                s = rng.GetSize()
                dims = [round(abs(s[0]), 4), round(abs(s[1]), 4), round(abs(s[2]), 4)]
                cw = M.Transform(rng.GetMidpoint())   # AABB centre in the base frame
                center = [round(cw[0], 4), round(cw[1], 4), round(cw[2], 4)]
        except Exception:  # noqa: BLE001
            pass
        out.append({
            'name': p.GetName(),
            'group': group_key(p.GetName()),
            # origin: prim frame origin (base) -> MESH pose (verts are relative to it).
            'position': origin,
            # centre: AABB centre (base) -> PRIMITIVE pose (MoveIt shapes are centred, so
            # using the origin would sink a bottle half below its base — the USD bottle
            # origin sits at the BOTTOM, the cylinder pose is its MIDDLE).
            'center': center,
            'orientation': [round(im[0], 4), round(im[1], 4), round(im[2], 4), round(q.GetReal(), 4)],
            'dims': dims,          # local AABB (x,y,z) -> primitive collision shape
        })
    return out


def scan_meshes(mesh_pkg: str, hint_path: Optional[str] = None) -> dict:
    """Find <mesh_pkg>/meshes and list its STLs. -> {'dir': str|None, 'stls': [...]}.
    Reported to the user: a missing dir silently yields empty mesh paths, which builds a
    scene the loader cannot render — so this must be surfaced, not swallowed."""
    meshes_dir = _resolve_meshes_dir(mesh_pkg, hint_path)
    stls: List[str] = []
    if meshes_dir:
        for dp, _dirs, files in os.walk(meshes_dir):
            for f in files:
                if f.lower().endswith('.stl'):
                    stls.append(os.path.relpath(os.path.join(dp, f), meshes_dir))
    return {'dir': meshes_dir, 'stls': sorted(stls)}


def build_group_rules(prims: List[dict], mesh_pkg: str,
                      hint_path: Optional[str] = None) -> List[dict]:
    """Group the prims and auto-suggest a rule per group: mesh (scanned), category
    (heuristic), grasp, include. The user edits these in the wizard table."""
    stls = scan_meshes(mesh_pkg, hint_path)['stls']
    groups: Dict[str, int] = {}
    for pr in prims:
        groups[pr['group']] = groups.get(pr['group'], 0) + 1

    # representative local dims per group (all instances of a group share the asset)
    dims_of: Dict[str, list] = {}
    for pr in prims:
        dims_of.setdefault(pr['group'], pr.get('dims', [0.1, 0.1, 0.1]))

    rules: List[dict] = []
    for g, count in groups.items():
        mesh_rel = suggest_mesh(g, stls)
        is_dynamic = any(h in g.lower() for h in _DYNAMIC_HINTS)
        grasp = 'bottle' in g.lower()
        d = dims_of.get(g, [0.1, 0.1, 0.1])
        # COLLISION SHAPE. A grasped object becomes an AttachedCollisionObject: its BVH is
        # rebuilt/transformed with the robot on EVERY state update, so a full visual mesh
        # (a 10k-triangle bottle x20 = 200k) makes IK and RViz crawl. Grasp targets
        # therefore default to a cheap PRIMITIVE sized from the USD extents; everything
        # else keeps its mesh (built once, static in the world).
        if grasp:
            round_ish = max(d[0], d[1]) > 0 and abs(d[0] - d[1]) / max(d[0], d[1]) < 0.25
            shape = 'cylinder' if round_ish else 'box'
        else:
            shape = 'mesh'
        rules.append({
            'group': g,
            'count': count,
            'shape': shape,
            'dims': d,
            'mesh': f'package://{mesh_pkg}/meshes/{mesh_rel}' if mesh_rel else '',
            'category': 'dynamic' if is_dynamic else 'static',
            'grasp': grasp,
            # default-include only groups we found a mesh for (context/machine with no
            # collision STL is left out, matching the baseline).
            'include': bool(mesh_rel),
        })
    return rules
