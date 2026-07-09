"""Compile a xacro into a URDF string.

Primary path is the in-process xacro Python API (the same one MoveItConfigsBuilder
uses) — it resolves ``$(find pkg)`` via ament in-process and avoids the fragile xacro
console-script entry-point lookup. A subprocess fallback covers exotic environments.
The robot's description package must be installed/findable for ``$(find)`` to resolve.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Dict, Optional, Union


def compile_xacro(
    xacro_path: Union[str, Path], mappings: Optional[Dict[str, str]] = None
) -> str:
    """Return the URDF produced from ``xacro_path``.

    Raises FileNotFoundError if the file is missing, RuntimeError if xacro fails.
    """
    xacro_path = Path(xacro_path)
    if not xacro_path.is_file():
        raise FileNotFoundError(f'xacro file not found: {xacro_path}')
    str_mappings = {k: str(v) for k, v in (mappings or {}).items()}

    try:
        import xacro  # noqa: WPS433 - optional in-process path
    except ImportError:
        return _compile_via_subprocess(xacro_path, str_mappings)

    try:
        doc = xacro.process_file(str(xacro_path), mappings=str_mappings)
        return doc.toprettyxml(indent='  ')
    except Exception as exc:  # noqa: BLE001 - surface a clean error
        raise RuntimeError(f'xacro failed for {xacro_path.name}: {exc}') from exc


def _compile_via_subprocess(xacro_path: Path, mappings: Dict[str, str]) -> str:
    xacro_bin = shutil.which('xacro') or '/opt/ros/humble/bin/xacro'
    args = [xacro_bin, str(xacro_path)] + [f'{k}:={v}' for k, v in mappings.items()]
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f'xacro failed for {xacro_path.name}:\n{proc.stderr.strip()}')
    return proc.stdout
