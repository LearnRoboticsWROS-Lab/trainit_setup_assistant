"""Machine-local launcher settings (vNext I0.2, ADR-0027 Decision 5).

These record where THIS machine keeps things: the Isaac Sim launcher and the colcon workspace TSA
builds in. They are user settings, kept outside every project (``~/.config/trainit/tsa.yaml``, or the
file named by ``TRAINIT_TSA_SETTINGS``), and carry no application semantics: a project opened on another
machine is the same project.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

ENV = 'TRAINIT_TSA_SETTINGS'
_ISAAC_CANDIDATES = ('~/isaacsim/isaac-sim.sh', '~/isaac-sim/isaac-sim.sh')


def settings_file() -> Path:
    return Path(os.environ.get(ENV) or Path.home() / '.config' / 'trainit' / 'tsa.yaml')


@dataclass
class LauncherSettings:
    isaac_launcher: str = ''          # isaac-sim.sh
    workspace_root: str = ''          # the colcon workspace (the parent of src/)

    @classmethod
    def load(cls, path: Optional[Path] = None) -> 'LauncherSettings':
        import yaml
        path = Path(path) if path else settings_file()
        data = {}
        if path.is_file():
            try:
                data = yaml.safe_load(path.read_text()) or {}
            except Exception:  # noqa: BLE001 - a broken settings file falls back to the defaults
                data = {}
        out = cls(**{k: str(v) for k, v in data.items() if k in cls.__dataclass_fields__ and v})
        if not out.isaac_launcher:
            out.isaac_launcher = next((str(Path(c).expanduser()) for c in _ISAAC_CANDIDATES
                                       if Path(c).expanduser().is_file()), '')
        return out

    def save(self, path: Optional[Path] = None) -> Path:
        import yaml
        path = Path(path) if path else settings_file()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(asdict(self), sort_keys=True))
        return path


__all__ = ['ENV', 'LauncherSettings', 'settings_file']
