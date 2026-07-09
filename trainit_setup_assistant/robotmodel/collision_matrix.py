"""Self-collision matrix via the MoveIt Setup Assistant's ``collisions_updater``.

This breaks the chicken-and-egg headlessly: compile the robot's geometry to a URDF,
feed it plus a groups-only SRDF to ``collisions_updater``, and parse back the
``disable_collisions`` matrix — the exact algorithm the MSA GUI uses, no GUI required.
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

from ..model import CanonicalProject
from .srdf_builder import DisablePair, build_srdf
from .xacro_loader import compile_xacro

# Matches the MoveIt Setup Assistant GUI defaults.
DEFAULT_TRIALS = 10000
DEFAULT_MIN_COLLISION_FRACTION = 1.0
DEFAULT_TIMEOUT_S = 120

_CANDIDATE_PATHS = (
    '/opt/ros/humble/lib/moveit_setup_assistant/collisions_updater',
)


def _find_collisions_updater() -> str:
    found = shutil.which('collisions_updater')
    if found:
        return found
    for cand in _CANDIDATE_PATHS:
        if Path(cand).is_file():
            return cand
    raise FileNotFoundError(
        'collisions_updater not found (install ros-humble-moveit-setup-assistant)'
    )


def compute_disable_collisions(
    project: CanonicalProject,
    trials: int = DEFAULT_TRIALS,
    min_collision_fraction: float = DEFAULT_MIN_COLLISION_FRACTION,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> List[DisablePair]:
    """Compute the self-collision matrix for the project's robot.

    Geometry is taken from the SOURCE description (copied verbatim into the bundle, so
    the matrix is identical), which must be installed for ``$(find)`` to resolve.

    ``collisions_updater`` is a ROS node that can hang at DDS init when spawned
    head-/TTY-less; this runs it in its own session with isolated DDS and a hard
    timeout, killing the whole process group on timeout so nothing is orphaned. The
    result is meant to be stored in ``robot.disable_collisions`` (project data) so
    generation itself never needs this.
    """
    desc = project.robot.description
    xacro_path = Path(desc.urdf_dir) / desc.top_xacro
    urdf = compile_xacro(xacro_path)
    base_srdf = build_srdf(project, disable_collisions=[])  # groups/states/EEF, no pairs
    updater = _find_collisions_updater()

    # Isolate DDS so a flaky discovery never blocks and never touches a live system.
    env = dict(os.environ)
    env['ROS_LOCALHOST_ONLY'] = '1'
    env.setdefault('ROS_DOMAIN_ID', '199')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        urdf_file = tmp / 'robot.urdf'
        srdf_in = tmp / 'base.srdf'
        srdf_out = tmp / 'out.srdf'
        urdf_file.write_text(urdf)
        srdf_in.write_text(base_srdf)
        cmd = [
            updater,
            '--urdf', str(urdf_file),
            '--srdf', str(srdf_in),
            '--output', str(srdf_out),
            '--trials', str(trials),
            '--min-collision-fraction', str(min_collision_fraction),
        ]
        proc = subprocess.Popen(
            cmd, stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True, env=env,
        )
        try:
            proc.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            raise RuntimeError(
                f'collisions_updater timed out after {timeout_s}s '
                '(known to hang headless; store the matrix in robot.disable_collisions)'
            )
        if proc.returncode != 0 or not srdf_out.is_file():
            raise RuntimeError(f'collisions_updater failed (rc={proc.returncode})')
        return parse_disable_collisions(srdf_out.read_text())


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:  # noqa: BLE001 - best effort cleanup
        proc.kill()


def parse_disable_collisions(srdf_text: str) -> List[DisablePair]:
    """Extract ``disable_collisions`` rows from an SRDF string (in document order)."""
    root = ET.fromstring(srdf_text)
    pairs: List[DisablePair] = []
    for dc in root.findall('disable_collisions'):
        link1 = dc.get('link1')
        link2 = dc.get('link2')
        if link1 and link2:
            pairs.append(DisablePair(link1, link2, dc.get('reason', 'Never')))
    return pairs
