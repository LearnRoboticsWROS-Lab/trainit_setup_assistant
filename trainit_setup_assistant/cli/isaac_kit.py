"""``trainit_isaac_kit`` — install the Isaac cell kit into a ``<cell>_isaac`` package.

The TSA automates the APPLICATION layer; the Isaac cell is hand-made expert work. This
command ships the reusable half of that work — the stage validator, the collision-mesh
exporter and the two Script Node templates that implement the ROS contract — so a new
project starts from the same baseline instead of rediscovering it.

    ros2 run trainit_setup_assistant trainit_isaac_kit --into ~/ws/src/mycell_isaac

Existing files are never overwritten unless --force is given.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

_KIT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    'resources', 'isaac_cell_kit')


def _copy(src: str, dst: str, force: bool) -> str:
    if os.path.exists(dst) and not force:
        return 'skipped (exists)'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    if dst.endswith('.py'):
        os.chmod(dst, 0o755)
    return 'written'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--into', required=True,
                    help='the <cell>_isaac package directory (must already exist)')
    ap.add_argument('--force', action='store_true', help='overwrite existing files')
    a = ap.parse_args(argv)

    pkg = os.path.abspath(os.path.expanduser(a.into))
    if not os.path.isdir(pkg):
        sys.exit(f'error: {pkg} is not a directory. Create the package first.')
    if not os.path.isfile(os.path.join(pkg, 'package.xml')):
        print(f'warning: no package.xml in {pkg} — is this a ROS package?', file=sys.stderr)
    if not os.path.isdir(_KIT):
        sys.exit(f'error: kit not found at {_KIT} (was the TSA installed with its resources?)')

    name = os.path.basename(pkg)
    print(f'Installing the Isaac cell kit into {name}\n')
    for rel in ('scripts/check_usd_for_tsa.py', 'scripts/usd_to_collision_stl.py',
                'scripts/isaac_gripper_subscriber.py',
                'scripts/isaac_scene_reset_subscriber.py',
                'README_isaac_cell_kit.md'):
        src = os.path.join(_KIT, rel if not rel.startswith('README') else 'README.md')
        dst = os.path.join(pkg, rel)
        print(f'  {rel:44s} {_copy(src, dst, a.force)}')

    for sub in ('usd/scenes', 'usd/props', 'meshes'):
        d = os.path.join(pkg, sub)
        if not os.path.isdir(d):
            os.makedirs(d, exist_ok=True)
            print(f'  {sub + "/":44s} created')

    cml = os.path.join(pkg, 'CMakeLists.txt')
    has_meshes = os.path.isfile(cml) and 'meshes' in open(cml).read()
    print('\nNext:')
    if not has_meshes:
        print(f'  1. add `meshes` to install(DIRECTORY ...) in {name}/CMakeLists.txt —\n'
              f'     without it the TSA finds no meshes and Step 2 builds 0 objects, silently.'
              f'\n     See {name}/README_isaac_cell_kit.md for the snippet.')
    else:
        print(f'  1. CMakeLists.txt already installs meshes/ — good.')
    print(f'  2. author the stage per {name}/README_isaac_cell_kit.md')
    print(f'  3. colcon build --packages-select {name}')
    print(f'  4. python3 scripts/check_usd_for_tsa.py --stage usd/scenes/<cell>.usd \\\n'
          f'         --robot-name <robot> --base-frame <frame> --mesh-pkg {name}')
    print('     exit 0 means the wizard will import the cell cleanly.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
