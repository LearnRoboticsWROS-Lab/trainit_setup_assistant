#!/usr/bin/env python3
"""Conformance checker: will the TrainIt Setup Assistant import this USD cell?

Run this on EVERY cell you author, before opening the TSA wizard. It evaluates the stage
against the Step-2 import contract reconstructed from
``trainit_setup_assistant/importers/usd_scene.py`` and ``gui/controller.py``, and exits
non-zero if anything blocking is wrong.

    python3 check_usd_for_tsa.py --stage <cell.usd> --robot-name <r> --base-frame <f> \
                                 [--mesh-pkg <pkg>]

The three inputs after --stage are exactly what the TSA project carries:
``robot.robot_name``, ``robot.base_frame`` and the "Mesh package" typed on Step 2. The
checker reproduces the importer's own resolution, so a PASS here means Step 2 will list
your objects with correct poses, dims and mesh suggestions.

Rules checked (blocking ones marked FAIL, the rest WARN):
  1  stage opens with vanilla pxr and every asset reference resolves offline
  2  metersPerUnit == 1.0 and upAxis == Z
  3  /World exists (otherwise the importer silently enumerates the stage root instead)
  4  the robot is at /World/<robot_name>/<base_frame>  -> base-relative poses
  5  every object is a DIRECT child of /World (depth 2; the importer never recurses)
  6  every object prim is Xformable, defined, active and loaded
  7  no object carries a non-identity xformOp:scale (dims ignore it)
  8  every object has a non-empty untransformed bound (else dims silently become 0.1^3)
  9  the mesh package resolves and every object group matches an STL
 10  at least one group would be auto-included (otherwise Step 2 builds 0 objects)
"""
from __future__ import annotations

import argparse
import io
import os
import sys
import tempfile

try:
    from pxr import Usd, UsdGeom
except ImportError:  # pragma: no cover
    sys.exit('error: pxr (usd-core) not available.')

PHYSICS_TYPES = {'PhysicsScene', 'PhysicsCollisionGroup'}
OK, WARN, FAIL = 'PASS', 'WARN', 'FAIL'
_results: list[tuple[str, str, str]] = []


def rec(status: str, rule: str, detail: str = '') -> None:
    _results.append((status, rule, detail))


def group_key(name: str) -> str:
    import re
    return re.sub(r'(_\d+)+$', '', name) or name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', required=True)
    ap.add_argument('--robot-name', required=True,
                    help="project robot.robot_name (the TSA builds /World/<this>/<base-frame>)")
    ap.add_argument('--base-frame', default='base_link', help='project robot.base_frame')
    ap.add_argument('--mesh-pkg', help='the "Mesh package" you will type on Step 2')
    ap.add_argument('--mesh-dir',
                    help='explicit <pkg>/meshes path, when the package is not installed in '
                         'THIS workspace (cross-workspace checks). Step 2 itself still needs '
                         'the installed share.')
    a = ap.parse_args(argv)

    # --- rule 1: opens offline, all references resolve -----------------------------
    # pxr reports resolution errors from C++, so capture at the FILE DESCRIPTOR level;
    # contextlib.redirect_stderr only rebinds sys.stderr and would miss every warning.
    saved = os.dup(2)
    tmp = tempfile.TemporaryFile(mode='w+')
    os.dup2(tmp.fileno(), 2)
    try:
        stage = Usd.Stage.Open(a.stage, load=Usd.Stage.LoadAll)
    finally:
        os.dup2(saved, 2)
        os.close(saved)
    tmp.seek(0)
    captured = tmp.read()
    tmp.close()
    if not stage:
        rec(FAIL, 'stage opens', f'cannot open {a.stage}')
        return report()
    unresolved = [ln for ln in captured.splitlines() if 'Could not open asset' in ln]
    if unresolved:
        schemes, prims = set(), set()
        for ln in unresolved:
            for tok in ln.split('@'):
                if tok.startswith(('http', 'omniverse:', 'file:')):
                    schemes.add(tok.split(':')[0] + ':')
            if '</' in ln:
                prims.add('/' + ln.split('</', 1)[1].split('>', 1)[0])
        # the importer skips the robot subtree outright (usd_scene.py:89-90), so an
        # unresolved payload under the robot never affects Step 2
        rr = f'/World/{a.robot_name}'
        def _blocking(x):
            return x.startswith('/World/') and not (x == rr or x.startswith(rr + '/'))
        inworld = sorted(x for x in prims if _blocking(x))
        outside = sorted(x for x in prims if not _blocking(x))
        rec(FAIL if inworld else WARN, 'assets resolve offline',
            f'{len(unresolved)} unresolved ({", ".join(sorted(schemes))}). '
            + (f'INSIDE /World (blocking): {", ".join(inworld)}. ' if inworld else '')
            + (f'outside /World or under the robot (harmless for Step 2): {", ".join(outside)}. '
               if outside else '')
            + 'pxr resolves neither http/omniverse nor ANY file: form — use RELATIVE local paths')
    else:
        rec(OK, 'assets resolve offline')

    # --- rule 2: units ------------------------------------------------------------
    mpu = UsdGeom.GetStageMetersPerUnit(stage)
    up = UsdGeom.GetStageUpAxis(stage)
    if abs(mpu - 1.0) < 1e-9 and up == 'Z':
        rec(OK, 'units', 'metersPerUnit=1.0, upAxis=Z')
    else:
        rec(FAIL, 'units', f'metersPerUnit={mpu}, upAxis={up} — the importer does NO conversion')

    # --- rule 3: /World -----------------------------------------------------------
    world = stage.GetPrimAtPath('/World')
    if world and world.IsValid():
        rec(OK, '/World exists')
    else:
        world = stage.GetPseudoRoot()
        rec(WARN, '/World exists',
            'no /World prim: the importer falls back to enumerating the STAGE ROOT')

    # --- rule 4: robot base prim --------------------------------------------------
    want = f'/World/{a.robot_name}/{a.base_frame}'
    bprim = stage.GetPrimAtPath(want)
    base_ok = bool(bprim and bprim.IsValid())
    if base_ok:
        rec(OK, 'robot base prim', want)
    else:
        # permissive predicate: an `over`-only base_link is still a usable base_prim
        found = [p.GetPath().pathString
                 for p in Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies(
                     Usd.PrimAllPrimsPredicate))
                 if p.GetName() == a.base_frame]
        rec(FAIL, 'robot base prim',
            f'{want} NOT FOUND -> poses degrade SILENTLY to stage-world frame. '
            + (f'candidates: {", ".join(found[:3])}' if found else 'no prim named ' + a.base_frame))

    robot_root = f'/World/{a.robot_name}'

    # --- rules 5-8: the object rows -----------------------------------------------
    bbox = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    rows, skipped, scaled, empty = [], [], [], []
    for p in world.GetChildren():
        path = p.GetPath().pathString
        if path == robot_root:
            continue
        if p.GetTypeName() in PHYSICS_TYPES:
            continue
        if not p.IsA(UsdGeom.Xformable):
            skipped.append(f'{p.GetName()} (not Xformable: {p.GetTypeName()})')
            continue
        rows.append(p)
        sc = p.GetAttribute('xformOp:scale')
        scv = sc.Get() if sc else None
        if scv and any(abs(v - 1.0) > 1e-9 for v in scv):
            scaled.append(f'{p.GetName()} scale={tuple(round(v, 4) for v in scv)}')
        rng = bbox.ComputeUntransformedBound(p).ComputeAlignedRange()
        if rng.IsEmpty():
            empty.append(p.GetName())

    deep = [c.GetName() for p in rows for c in p.GetChildren()
            if c.IsA(UsdGeom.Xformable) and c.GetChildren()]
    rec(OK if rows else FAIL, 'importable rows',
        f'{len(rows)} direct /World children would become rows'
        + (f'; skipped: {", ".join(skipped)}' if skipped else ''))
    if deep:
        rec(WARN, 'depth', f'nested Xforms are INVISIBLE (importer reads one level): {", ".join(deep[:5])}')
    rec(OK if not scaled else FAIL, 'identity scale',
        'all identity' if not scaled else
        f'dims IGNORE the prim transform, so these report the UNSCALED size: {"; ".join(scaled)}')
    rec(OK if not empty else FAIL, 'non-empty bounds',
        'all objects have geometry' if not empty else
        f'empty AABB -> dims silently fall back to 0.1x0.1x0.1: {", ".join(empty)}')

    # --- rules 9-10: the mesh package ---------------------------------------------
    if a.mesh_pkg:
        meshes_dir = None
        try:
            from ament_index_python.packages import get_package_share_directory
            d = os.path.join(get_package_share_directory(a.mesh_pkg), 'meshes')
            meshes_dir = d if os.path.isdir(d) else None
        except Exception:  # noqa: BLE001
            pass
        if not meshes_dir and a.mesh_dir and os.path.isdir(a.mesh_dir):
            meshes_dir = a.mesh_dir
            rec(WARN, 'mesh package source',
                f'using --mesh-dir {a.mesh_dir}; Step 2 resolves through the INSTALLED share, '
                f'so this passes here but would fail in the wizard until you colcon build it')
        if not meshes_dir:
            rec(FAIL, 'mesh package',
                f'"{a.mesh_pkg}" exposes no INSTALLED meshes/ dir. Add `meshes` to the '
                f'CMakeLists install(DIRECTORY ...) and colcon build it')
        else:
            stls = [os.path.relpath(os.path.join(dp, f), meshes_dir)
                    for dp, _d, fs in os.walk(meshes_dir) for f in fs
                    if f.lower().endswith('.stl')]
            rec(OK, 'mesh package', f'{meshes_dir} ({len(stls)} STL)')
            matched, unmatched = [], []
            for g in sorted({group_key(p.GetName()) for p in rows}):
                gl = g.lower()
                hit = [s for s in stls
                       if s.lower().startswith(gl + '/') or gl in os.path.basename(s).lower()]
                (matched if hit else unmatched).append(g if not hit else f'{g} -> {hit[0]}')
            rec(OK if matched else FAIL, 'auto-included groups',
                f'{len(matched)} would auto-include: {"; ".join(matched)}' if matched
                else 'NO group matches an STL -> Step 2 builds 0 objects with no error')
            if unmatched:
                rec(WARN, 'unmatched groups',
                    f'no STL for: {", ".join(unmatched)} (they stay unchecked — '
                    f'fine for decoration, blocking for real obstacles)')
    else:
        rec(WARN, 'mesh package', 'not given (--mesh-pkg) — rules 9-10 not checked')

    return report()


def report() -> int:
    width = max(len(r[1]) for r in _results)
    print()
    for status, rule, detail in _results:
        mark = {OK: '  OK  ', WARN: ' WARN ', FAIL: ' FAIL '}[status]
        print(f'[{mark}] {rule.ljust(width)}  {detail}')
    fails = sum(1 for s, _, _ in _results if s == FAIL)
    warns = sum(1 for s, _, _ in _results if s == WARN)
    print(f'\n{len(_results)} checks — {fails} FAIL, {warns} WARN')
    if fails:
        print('The TSA will NOT import this cell correctly. See '
              'docs/notes_usd_authoring_for_tsa.md')
    return 1 if fails else 0


if __name__ == '__main__':
    raise SystemExit(main())
