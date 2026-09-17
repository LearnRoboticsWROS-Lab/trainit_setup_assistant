"""``trainit_compute_collisions`` — compute the self-collision matrix for a project.

Runs the MoveIt Setup Assistant's ``collisions_updater`` (needs a ROS environment)
and prints the ``disable_collisions`` block to paste into the project's
``robot.disable_collisions`` — or writes it back in place with ``--write``. Generation
itself never runs this; it consumes the stored matrix (project data).
"""

from __future__ import annotations

import argparse
import sys

import yaml

from ..model import load_project, save_project
from ..model.robot import DisableCollisionSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='trainit_compute_collisions',
        description='Compute robot.disable_collisions via collisions_updater.',
    )
    parser.add_argument('project', help='path to project.yaml')
    parser.add_argument('--trials', type=int, default=10000)
    parser.add_argument('--timeout', type=int, default=120, help='seconds')
    parser.add_argument('--write', action='store_true',
                        help='write the matrix back into the project file')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        project = load_project(args.project)
    except Exception as exc:  # noqa: BLE001
        print(f'error: cannot load {args.project}: {exc}', file=sys.stderr)
        return 2

    try:
        from ..robotmodel.collision_matrix import compute_disable_collisions
        pairs = compute_disable_collisions(
            project, trials=args.trials, timeout_s=args.timeout
        )
    except Exception as exc:  # noqa: BLE001
        print(f'error: {exc}', file=sys.stderr)
        return 1

    specs = [DisableCollisionSpec(link1=p.link1, link2=p.link2, reason=p.reason)
             for p in pairs]
    print(f'computed {len(specs)} disable_collisions pairs', file=sys.stderr)

    if args.write:
        project.robot.disable_collisions = specs
        save_project(project, args.project)
        print(f'wrote matrix into {args.project}', file=sys.stderr)
    else:
        block = {'disable_collisions': [s.dict() for s in specs]}
        print(yaml.safe_dump(block, sort_keys=False, default_flow_style=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
