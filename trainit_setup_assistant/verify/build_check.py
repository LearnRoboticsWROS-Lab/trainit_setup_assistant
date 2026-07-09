"""colcon-build a generated bundle in a throwaway overlay workspace.

This is the second verification layer (after structural equivalence): proof that the
emitted packages actually build. Needs a ROS environment; intended for the CLI / CI,
not the fast unit test.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

DEFAULT_UNDERLAY = '/home/fra/fr5_ws/install/setup.bash'
ROS_SETUP = '/opt/ros/humble/setup.bash'


def colcon_build_bundle(
    bundle_root: Path,
    underlay_setup: Optional[str] = DEFAULT_UNDERLAY,
    timeout_s: int = 600,
) -> Tuple[bool, str]:
    """Copy every package in ``bundle_root`` into a temp ws and ``colcon build``.

    Returns (ok, combined_log).
    """
    bundle_root = Path(bundle_root)
    pkgs: List[Path] = [
        p for p in sorted(bundle_root.iterdir())
        if p.is_dir() and (p / 'package.xml').is_file()
    ]
    if not pkgs:
        return False, f'no packages found under {bundle_root}'

    with tempfile.TemporaryDirectory() as ws:
        ws = Path(ws)
        src = ws / 'src'
        src.mkdir()
        for pkg in pkgs:
            shutil.copytree(pkg, src / pkg.name)

        steps = [f'source {ROS_SETUP}']
        if underlay_setup and Path(underlay_setup).is_file():
            steps.append(f'source {underlay_setup}')
        steps.append(f'cd {ws}')
        steps.append('colcon build')
        cmd = ' && '.join(steps)

        try:
            proc = subprocess.run(
                ['bash', '-lc', cmd], capture_output=True, text=True, timeout=timeout_s
            )
        except subprocess.TimeoutExpired:
            return False, f'colcon build timed out after {timeout_s}s'
        return proc.returncode == 0, proc.stdout + proc.stderr
