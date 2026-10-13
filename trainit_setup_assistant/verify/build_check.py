"""colcon-build a generated bundle in a throwaway overlay workspace.

This is the second verification layer (after structural equivalence): proof that the
emitted packages actually build. Needs a ROS environment; intended for the CLI / CI,
not the fast unit test.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

AUTO = 'auto'      # underlay_setup default: the workspace the caller has sourced


def default_underlay() -> Optional[str]:
    """The ``setup.bash`` of the workspace the caller has SOURCED, or None.

    colcon's setup scripts export ``COLCON_PREFIX_PATH``; its first entry holding a
    ``setup.bash`` is the innermost sourced workspace. This used to be one machine's absolute
    path, which on any other machine silently did not exist (vNext M0C #8).
    """
    for prefix in os.environ.get('COLCON_PREFIX_PATH', '').split(os.pathsep):
        if prefix and (Path(prefix) / 'setup.bash').is_file():
            return str(Path(prefix) / 'setup.bash')
    return None


def ros_setup() -> str:
    """The ROS setup script of the sourced distro (Humble, the supported track, if unset)."""
    return f"/opt/ros/{os.environ.get('ROS_DISTRO') or 'humble'}/setup.bash"


def colcon_build_bundle(
    bundle_root: Path,
    underlay_setup: Optional[str] = AUTO,
    timeout_s: int = 600,
) -> Tuple[bool, str]:
    """Copy every package in ``bundle_root`` into a temp ws and ``colcon build``.

    ``underlay_setup``: ``'auto'`` (the default) = the sourced workspace
    (:func:`default_underlay`); a path = that one; ``None`` = ROS alone.

    Returns (ok, combined_log).
    """
    if underlay_setup == AUTO:
        underlay_setup = default_underlay()
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

        steps = [f'source {ros_setup()}']
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
