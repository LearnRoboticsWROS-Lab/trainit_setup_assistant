"""``trainit_generate`` — headless: project.yaml -> generated bundle.

This is the GUI-independent contract test: anything the GUI can build, it builds by
emitting a project.yaml that this command turns into the bundle.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# The core model/generator imports (which need pydantic/jinja2/pyyaml) are deferred INTO
# main(), so argparse --help works with no deps and a MISSING dependency fails fast with a
# clear, actionable message instead of a raw ImportError traceback.


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='trainit_generate',
        description='Generate a TrainIt application bundle from a project.yaml.',
    )
    parser.add_argument('project', help='path to project.yaml')
    parser.add_argument(
        '-o', '--output', required=True, help='output bundle root directory'
    )
    parser.add_argument(
        '--force', action='store_true',
        help='remove the output directory first if it already exists',
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        from ..generator import Orchestrator
        from ..model import load_project
    except ImportError as exc:
        from ._deps import die_on_missing_dependency
        return die_on_missing_dependency(exc)

    try:
        project = load_project(args.project)
    except Exception as exc:  # noqa: BLE001 - surface any load/validation error
        print(f'error: failed to load project {args.project}: {exc}', file=sys.stderr)
        return 2

    out = Path(args.output)
    if out.exists():
        if args.force:
            shutil.rmtree(out)
        elif any(out.iterdir()):
            print(
                f'error: output {out} is not empty (use --force to overwrite)',
                file=sys.stderr,
            )
            return 2

    try:
        manifest = Orchestrator().generate(project, out)
    except Exception as exc:  # noqa: BLE001
        print(f'error: generation failed: {exc}', file=sys.stderr)
        return 1

    info = manifest.as_dict()
    print(f"Generated {info['file_count']} files into {out}")
    print(f"  packages: {project.bundle.description_package}, "
          f"{project.bundle.moveit_config_package}, {project.bundle.app_package}")
    for warning in manifest.warnings:
        print(f'  warning: {warning}', file=sys.stderr)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
