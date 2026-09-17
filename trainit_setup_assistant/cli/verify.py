"""``trainit_verify`` — generate a project and check it against a golden bundle.

  trainit_verify <project.yaml> --golden <golden_bundle_dir> [--build]

Structural equivalence always runs (fast, no ROS). ``--build`` additionally
colcon-builds the generated bundle (needs a ROS environment).
"""

from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

# Core imports (need pydantic/jinja2/pyyaml) are deferred INTO main() so --help works with
# no deps and a MISSING dependency fails fast with a clear message, not a raw traceback.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='trainit_verify',
        description='Generate a project and verify equivalence to a golden bundle.',
    )
    parser.add_argument('project', help='path to project.yaml')
    parser.add_argument('--golden', required=True,
                        help='golden bundle directory (contains the 3 packages)')
    parser.add_argument('--build', action='store_true',
                        help='also colcon-build the generated bundle')
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ..generator import Orchestrator
        from ..model import load_project
        from ..verify import colcon_build_bundle, compare_bundle, detect_packages
        from ..verify.equivalence import APP, DESCRIPTION, MOVEIT
    except ImportError as exc:
        from ._deps import die_on_missing_dependency
        return die_on_missing_dependency(exc)

    try:
        project = load_project(args.project)
    except Exception as exc:  # noqa: BLE001
        print(f'error: cannot load {args.project}: {exc}', file=sys.stderr)
        return 2

    gen_packages = {
        DESCRIPTION: project.bundle.description_package,
        MOVEIT: project.bundle.moveit_config_package,
        APP: project.bundle.app_package,
    }
    gold_packages = detect_packages(Path(args.golden))
    missing_roles = [r for r in (DESCRIPTION, MOVEIT, APP) if r not in gold_packages]
    if missing_roles:
        print(f'error: could not find {missing_roles} packages under {args.golden}',
              file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(project, tmp)
        report = compare_bundle(Path(tmp), Path(args.golden), gen_packages, gold_packages)
        print(report.summary())
        if not report.ok:
            print('\nEQUIVALENCE: FAIL', file=sys.stderr)
            return 1
        print('\nEQUIVALENCE: PASS')

        if args.build:
            print('\nbuilding generated bundle...')
            ok, log = colcon_build_bundle(Path(tmp))
            if not ok:
                print(log[-2000:], file=sys.stderr)
                print('BUILD: FAIL', file=sys.stderr)
                return 1
            print('BUILD: PASS')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
