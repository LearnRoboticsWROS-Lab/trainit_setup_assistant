#!/usr/bin/env python3
"""Export USD cell prims to collision STLs for the TrainIt Setup Assistant.

WHY THIS EXISTS
---------------
TSA Step 2 ("Cell scene from the USD") builds every imported object as a **mesh**
(``gui/controller.py`` -> ``apply_usd_mapping``, ``shape='mesh'``), and refuses to
include a group that has no mesh resource. The mesh must live in a ROS package as
``<mesh_pkg>/meshes/<...>.stl`` and be discoverable in that package's INSTALLED share
(``importers/usd_scene.py`` -> ``_resolve_meshes_dir``).

THE CONTRACT THIS TOOL IMPLEMENTS (verified against the BIG1500 reference cell)
------------------------------------------------------------------------------
1. Units: **metres**. STL carries no units, so ``metersPerUnit`` is baked in here.
2. Frame: the prim's **local** frame. TSA takes the object's pose from the USD prim and
   the geometry from the STL, so the STL must NOT have the prim's own transform baked in
   or the object is placed twice.
3. Size: the STL bounding box must equal ``BBoxCache.ComputeUntransformedBound(prim)``,
   because that is what TSA reads into ``dims`` (used by ``attach_box`` grasping).
   Verified on the reference cell: ``bottle_50cl.stl`` bbox == (0.0666, 0.0668, 0.2445)
   == the untransformed bound of ``/World/bottle_0_0``.
4. Geometry: prefer the asset's **collision** subtree over its visual one. Isaac assets
   commonly ship ``<asset>/Collisions/*`` (a coarse hull) next to ``<asset>/Visuals/*``
   (hundreds of thousands of points). The planning scene wants the coarse one.
5. Extension: TSA scans for ``.stl`` only; ``.dae``/``.obj`` are ignored.

NAMING (so TSA's auto-suggest finds the file)
---------------------------------------------
``importers/usd_scene.py`` -> ``suggest_mesh`` matches a group when the STL path, relative
to ``meshes/``, either starts with ``<group>/`` or contains ``<group>`` in its basename
(case-insensitive). The group key is the prim name with trailing ``_<digits>`` runs
stripped (``bottle_0_0`` -> ``bottle``). So either layout works::

    meshes/<group>/<anything>.stl        e.g. meshes/table/table_collision.stl
    meshes/<anything><group><anything>.stl

WARNING ON NON-IDENTITY SCALE
-----------------------------
If the source prim carries a non-identity ``xformOp:scale``, TSA's ``dims`` will DISAGREE
with reality, because ``ComputeUntransformedBound`` ignores the prim's own transform.
This tool detects that and warns. Fix it in the USD (bake the scale into the geometry and
set scale to 1) rather than compensating here, otherwise ``dims`` and the mesh diverge.

EXAMPLES
--------
    # one prim of the cell stage
    python3 usd_to_collision_stl.py \
        --stage ../usd/scenes/bin_picking_cell.usd \
        --prim /World/table \
        --out  ../meshes/table/table_collision.stl

    # a standalone asset file (e.g. one downloaded from the Omniverse S3 bucket)
    python3 usd_to_collision_stl.py --stage table.usd --out ../meshes/table/table_collision.stl

    # inspect first, write nothing
    python3 usd_to_collision_stl.py --stage ../usd/scenes/bin_picking_cell.usd --list
"""
from __future__ import annotations

import argparse
import os
import struct
import sys

try:
    from pxr import Gf, Usd, UsdGeom
except ImportError:  # pragma: no cover - environment guard
    sys.exit('error: pxr (usd-core) not available. Source a ROS/Isaac env that provides it.')


def _triangles(mesh: UsdGeom.Mesh, xform: Gf.Matrix4d, scale: float):
    """Yield (v0, v1, v2) triangles, transformed into the target frame and scaled."""
    points = mesh.GetPointsAttr().Get()
    counts = mesh.GetFaceVertexCountsAttr().Get()
    indices = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not counts or not indices:
        return
    pts = [Gf.Vec3f(xform.Transform(Gf.Vec3d(p)) * scale) for p in points]
    i = 0
    for c in counts:
        if c < 3:
            i += c
            continue
        # fan-triangulate any polygon (quads and n-gons are common in Isaac assets)
        for k in range(1, c - 1):
            yield pts[indices[i]], pts[indices[i + k]], pts[indices[i + k + 1]]
        i += c


def _analytic_triangles(prim: Usd.Prim, xform: Gf.Matrix4d, scale: float):
    """Tessellate an analytic UsdGeom gprim (Cube / Sphere / Cylinder) into triangles.

    Isaac cells often carry the manipulated part as an analytic ``Cube`` rather than a
    referenced mesh. TSA still needs an STL for it, so we tessellate. NOTE we deliberately
    do NOT bake the prim's own ``xformOp:scale`` (use --bake-scale to override): the
    contract is that the STL matches ``ComputeUntransformedBound``, which ignores it.
    """
    t = prim.GetTypeName()
    verts = []
    if t == 'Cube':
        sz = prim.GetAttribute('size').Get()
        h = (sz if sz is not None else 2.0) / 2.0
        c = [(-h, -h, -h), (h, -h, -h), (h, h, -h), (-h, h, -h),
             (-h, -h, h), (h, -h, h), (h, h, h), (-h, h, h)]
        faces = [(0, 3, 2, 1), (4, 5, 6, 7), (0, 1, 5, 4),
                 (1, 2, 6, 5), (2, 3, 7, 6), (3, 0, 4, 7)]
        for q in faces:
            verts.append((c[q[0]], c[q[1]], c[q[2]]))
            verts.append((c[q[0]], c[q[2]], c[q[3]]))
    elif t in ('Sphere', 'Cylinder'):
        import math
        r = prim.GetAttribute('radius').Get() or 1.0
        seg = 24
        if t == 'Sphere':
            rings = 12
            grid = [[(r * math.sin(math.pi * i / rings) * math.cos(2 * math.pi * j / seg),
                      r * math.sin(math.pi * i / rings) * math.sin(2 * math.pi * j / seg),
                      r * math.cos(math.pi * i / rings))
                     for j in range(seg + 1)] for i in range(rings + 1)]
            for i in range(rings):
                for j in range(seg):
                    a, b, c2, d = grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]
                    verts.append((a, b, c2)); verts.append((a, c2, d))
        else:
            hh = (prim.GetAttribute('height').Get() or 2.0) / 2.0
            ring = [(r * math.cos(2 * math.pi * j / seg), r * math.sin(2 * math.pi * j / seg))
                    for j in range(seg + 1)]
            for j in range(seg):
                x0, y0 = ring[j]; x1, y1 = ring[j + 1]
                verts.append(((x0, y0, -hh), (x1, y1, -hh), (x1, y1, hh)))
                verts.append(((x0, y0, -hh), (x1, y1, hh), (x0, y0, hh)))
                verts.append(((0, 0, -hh), (x1, y1, -hh), (x0, y0, -hh)))
                verts.append(((0, 0, hh), (x0, y0, hh), (x1, y1, hh)))
    else:
        return
    for tri in verts:
        yield tuple(Gf.Vec3f(xform.Transform(Gf.Vec3d(*v)) * scale) for v in tri)


def _normal(a, b, c):
    n = Gf.Cross(b - a, c - a)
    ln = n.GetLength()
    return n / ln if ln > 1e-12 else Gf.Vec3f(0.0, 0.0, 1.0)


def collect_meshes(root: Usd.Prim, prefer_collision: bool = True):
    """Meshes under ``root``; the collision subtree alone when one exists."""
    meshes = [p for p in Usd.PrimRange(root) if p.IsA(UsdGeom.Mesh)]
    if prefer_collision:
        collision = [m for m in meshes if 'collision' in m.GetPath().pathString.lower()]
        if collision:
            return collision, True
    return meshes, False


def export(stage_path: str, prim_path: str | None, out_path: str,
           prefer_collision: bool = True, dry_run: bool = False,
           bake_scale: bool = False) -> int:
    stage = Usd.Stage.Open(stage_path, load=Usd.Stage.LoadAll)
    if not stage:
        sys.exit(f'error: cannot open stage {stage_path}')
    mpu = UsdGeom.GetStageMetersPerUnit(stage)

    if prim_path:
        root = stage.GetPrimAtPath(prim_path)
        if not (root and root.IsValid()):
            sys.exit(f'error: prim {prim_path} not found in {stage_path}')
    else:
        root = stage.GetDefaultPrim() or stage.GetPseudoRoot()

    meshes, used_collision = collect_meshes(root, prefer_collision)
    analytic = [p for p in Usd.PrimRange(root)
                if p.GetTypeName() in ('Cube', 'Sphere', 'Cylinder')]
    if not meshes and not analytic:
        sys.exit(f'error: no UsdGeom.Mesh under {root.GetPath()}. '
                 f'If this prim references an unresolvable asset (https:// or a malformed '
                 f'file:/ URI), fetch it locally first and re-point the reference.')

    # target frame = the prim's LOCAL frame -> exclude the prim's own transform
    root_world = (UsdGeom.Xformable(root).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                  if root.IsA(UsdGeom.Xformable) else Gf.Matrix4d(1.0))
    root_inv = root_world.GetInverse()

    scale_op = root.GetAttribute('xformOp:scale')
    scale_val = scale_op.Get() if scale_op else None
    if scale_val and not bake_scale and any(abs(s - 1.0) > 1e-9 for s in scale_val):
        print(f'WARNING: {root.GetPath()} has xformOp:scale={tuple(scale_val)}. '
              f"TSA's dims come from ComputeUntransformedBound(), which IGNORES that scale, "
              f'so dims will not match the real object. Bake the scale into the geometry '
              f'and set scale to 1 in the USD.', file=sys.stderr)

    tris = []
    for m in meshes:
        mx = UsdGeom.Xformable(m).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tris.extend(_triangles(UsdGeom.Mesh(m), mx * root_inv, mpu))
    if not meshes:
        for g in analytic:
            if g == root:
                # the root gprim lives in its OWN local frame: identity, or scale-only
                # when the caller asked to bake the prim's scale into the geometry
                m4 = Gf.Matrix4d(1.0)
                if bake_scale and scale_val:
                    m4 = Gf.Matrix4d().SetScale(Gf.Vec3d(*scale_val))
            else:
                gx = UsdGeom.Xformable(g).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                m4 = gx * root_inv
            tris.extend(_analytic_triangles(g, m4, mpu))
        print(f'  source    : analytic gprim ({root.GetTypeName()}) tessellated'
              + ('  [scale BAKED]' if bake_scale else '  [scale NOT baked - matches TSA dims]'))

    if not tris:
        sys.exit(f'error: meshes under {root.GetPath()} carry no geometry')

    xs = [v[0] for t in tris for v in t]
    ys = [v[1] for t in tris for v in t]
    zs = [v[2] for t in tris for v in t]
    bbox = (max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs))

    print(f'  prim      : {root.GetPath()}')
    print(f'  meshes    : {len(meshes)} ({"collision" if used_collision else "VISUAL - no collision subtree found"})')
    print(f'  triangles : {len(tris)}')
    print(f'  bbox [m]  : ({bbox[0]:.4f}, {bbox[1]:.4f}, {bbox[2]:.4f})')
    print(f'  origin    : X[{min(xs):+.4f},{max(xs):+.4f}] '
          f'Y[{min(ys):+.4f},{max(ys):+.4f}] Z[{min(zs):+.4f},{max(zs):+.4f}]')

    if dry_run:
        print('  (dry run - nothing written)')
        return 0

    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or '.', exist_ok=True)
    with open(out_path, 'wb') as fh:
        fh.write(b'TrainIt collision mesh exported from USD'.ljust(80, b' '))
        fh.write(struct.pack('<I', len(tris)))
        for a, b, c in tris:
            n = _normal(a, b, c)
            fh.write(struct.pack('<12fH', n[0], n[1], n[2],
                                 a[0], a[1], a[2], b[0], b[1], b[2], c[0], c[1], c[2], 0))
    print(f'  written   : {out_path} ({os.path.getsize(out_path)/1024:.1f} KiB)')
    return 0


def list_prims(stage_path: str) -> int:
    stage = Usd.Stage.Open(stage_path, load=Usd.Stage.LoadAll)
    if not stage:
        sys.exit(f'error: cannot open stage {stage_path}')
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    world = stage.GetPrimAtPath('/World')
    if not (world and world.IsValid()):
        print('NOTE: /World is missing; TSA falls back to the pseudo-root.')
        world = stage.GetPseudoRoot()
    print(f'stage    : {stage_path}')
    print(f'units    : metersPerUnit={mpu}  upAxis={UsdGeom.GetStageUpAxis(stage)}')
    print(f'TSA scans: children of {world.GetPath()}  '
          f'(anything outside it is INVISIBLE to Step 2)\n')
    print(f'{"prim":34s} {"type":12s} {"meshes":>6s}  untransformed bound [m]')
    for p in world.GetChildren():
        if not p.IsA(UsdGeom.Xformable):
            print(f'{p.GetPath().pathString:34s} {str(p.GetTypeName()):12s} {"-":>6s}  '
                  f'SKIPPED (not Xformable)')
            continue
        n = len([q for q in Usd.PrimRange(p) if q.IsA(UsdGeom.Mesh)])
        rng = bbox.ComputeUntransformedBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            dims = 'EMPTY -> TSA falls back to 0.1x0.1x0.1'
        else:
            s = rng.GetSize()
            dims = f'({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})'
        sc = p.GetAttribute('xformOp:scale')
        scv = sc.Get() if sc else None
        warn = ''
        if scv and any(abs(v - 1.0) > 1e-9 for v in scv):
            warn = f'   <-- scale={tuple(round(v, 4) for v in scv)} IGNORED by dims!'
        print(f'{p.GetPath().pathString:34s} {str(p.GetTypeName()):12s} {n:6d}  {dims}{warn}')
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True, help='USD stage or asset file')
    ap.add_argument('--prim', help='prim to export (default: the stage default prim)')
    ap.add_argument('--out', help='output .stl path')
    ap.add_argument('--list', action='store_true',
                    help='list what TSA Step 2 would see, and exit')
    ap.add_argument('--visual', action='store_true',
                    help='use the visual meshes even when a collision subtree exists')
    ap.add_argument('--bake-scale', action='store_true',
                    help="bake the prim's own xformOp:scale into the geometry. Only do this "
                         "if you ALSO set that scale to 1 in the USD, otherwise TSA's dims "
                         "(which ignore scale) and the mesh will disagree.")
    ap.add_argument('--dry-run', action='store_true', help='report, write nothing')
    a = ap.parse_args(argv)

    if a.list:
        return list_prims(a.stage)
    if not a.out and not a.dry_run:
        ap.error('--out is required (or use --dry-run / --list)')
    return export(a.stage, a.prim, a.out or '', prefer_collision=not a.visual,
                  dry_run=a.dry_run, bake_scale=a.bake_scale)


if __name__ == '__main__':
    raise SystemExit(main())
