"""Load/save a :class:`CanonicalProject` to ``project.yaml`` (round-trip safe)."""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml

from .project import CanonicalProject


def load_project(path: Union[str, Path]) -> CanonicalProject:
    """Parse + validate a project.yaml into a CanonicalProject."""
    data = yaml.safe_load(Path(path).read_text())
    if data is None:
        raise ValueError(f'empty project file: {path}')
    return CanonicalProject.model_validate(data)


def project_to_dict(project: CanonicalProject) -> dict:
    """Plain-data dict (enums -> their string tokens) for YAML/JSON dumping."""
    return project.model_dump(mode='json')


def save_project(project: CanonicalProject, path: Union[str, Path]) -> Path:
    """Write a project.yaml. Keys keep model order (sort_keys=False)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump(project_to_dict(project), sort_keys=False, default_flow_style=False)
    )
    return out
