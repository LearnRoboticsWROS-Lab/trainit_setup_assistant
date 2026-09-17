"""Load/save a :class:`CanonicalProject` to ``project.yaml`` (round-trip safe)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Union

import yaml

from .project import CanonicalProject


def load_project(path: Union[str, Path]) -> CanonicalProject:
    """Parse + validate a project.yaml into a CanonicalProject."""
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        raise ValueError(f'empty project file: {path}')
    return CanonicalProject.parse_obj(data)


def project_to_dict(project: CanonicalProject) -> dict:
    """Plain-data dict (enums -> their string tokens) for YAML/JSON dumping.

    ``json.loads(project.json())`` — NOT ``.dict()`` — reproduces pydantic v2's
    ``model_dump(mode='json')`` on BOTH pydantic 1.9 and 2.x: every StrEnum becomes its
    plain-string token and nested models become dicts, so ``yaml.safe_dump`` writes the
    same project.yaml. A bare ``.dict()`` would keep StrEnum *members* (a str subclass),
    which ``yaml.safe_dump`` cannot represent -> a changed/broken file. ``.json()`` is
    present (deprecated in 2.x) on both versions."""
    return json.loads(project.json())


def save_project(project: CanonicalProject, path: Union[str, Path]) -> Path:
    """Write a project.yaml. Keys keep model order (sort_keys=False)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(project_to_dict(project), sort_keys=False, default_flow_style=False)
    )
    return out
