"""Project-relative paths (vNext I0.0, I0_DESIGN.md §20).

A path written in a project resolves against the PROJECT FILE's directory, so a project (or a
template) works wherever it is opened from. Until TSA 8.0 it resolved against the current
directory; that still works, with a deprecation warning, when only the current directory holds
the file. The project itself is never rewritten: :func:`resolve_project_paths` returns a resolved
COPY for generation and validation, and the file keeps the path exactly as the user wrote it.

A bundle does not depend on the form: a relative and an absolute ``urdf_dir`` generate the same
bytes (verified 2026-10-21 on the public sample). When BOTH the project directory and the current
directory hold the path, the project directory wins, silently: that is the new rule, not a fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from .project import CanonicalProject

CWD_DEPRECATION = ("resolved against the current directory: a relative path now resolves against the "
                   "project file's directory (write it relative to the project file, or absolute)")


def _robot_description(p):
    return p.robot.description


# (label, owner of the field, field name): every path-valued field generation READS FROM DISK.
# Not `scene.world_path`: the bundle writes it AS WRITTEN (the gazebo `world:=` default, often a
# `$(find ...)` substitution), so resolving it would change a generated file.
_PATH_FIELDS: Tuple[Tuple[str, Callable, str], ...] = (
    ('robot.description.urdf_dir', _robot_description, 'urdf_dir'),
    ('robot.description.meshes_dir', _robot_description, 'meshes_dir'),
    ('robot.base_moveit_config_path', lambda p: p.robot, 'base_moveit_config_path'),
    # vNext I0.1: workflow metadata no generated file reads
    ('deployment.isaac.stage', lambda p: p.deployment.isaac, 'stage'),
    ('scene.provenance.file', lambda p: p.scene.provenance, 'file'),
)


def _is_local_relative(value) -> bool:
    """A plain relative filesystem path: not empty, not absolute, not a URI, not a launch
    substitution (``$(find pkg)/...``) and not ``~``-relative."""
    if not isinstance(value, str) or not value.strip():
        return False
    text = value.strip()
    return not (Path(text).is_absolute() or '://' in text or text.startswith('$(')
                or text.startswith('~'))


def resolve_path(value: str, project_dir: Union[str, Path],
                 cwd: Optional[Union[str, Path]] = None) -> Tuple[str, Optional[str]]:
    """``(resolved, note)`` for one path as written. The project directory wins; the current
    directory is a fallback (``note`` = :data:`CWD_DEPRECATION`); a path found in neither resolves
    against the project directory, so a refusal names where TSA looked. Non-relative values are
    returned unchanged."""
    if not _is_local_relative(value):
        return value, None
    project_dir = Path(project_dir)
    cwd = Path(cwd) if cwd is not None else Path.cwd()
    beside = (project_dir / value).resolve()
    if beside.exists():
        return str(beside), None
    here = (cwd / value).resolve()
    if here.exists():
        return str(here), CWD_DEPRECATION
    return str(beside), None


def resolve_project_paths(project: CanonicalProject, project_file: Optional[Union[str, Path]],
                          cwd: Optional[Union[str, Path]] = None
                          ) -> Tuple[CanonicalProject, List[str]]:
    """A COPY of ``project`` whose relative paths are absolute, resolved against the directory of
    ``project_file`` (the current directory as a deprecated fallback), and the warnings, one per
    path that fell back. ``project_file`` None (a project never saved) resolves against ``cwd``
    without a warning: there is no project directory to prefer."""
    resolved = project.copy(deep=True)
    if project_file is None:
        base = Path(cwd) if cwd is not None else Path.cwd()
    else:
        base = Path(project_file).resolve().parent
    warnings: List[str] = []
    for label, owner_of, field in _PATH_FIELDS:
        owner = owner_of(resolved)
        if owner is None:
            continue
        value = getattr(owner, field, None)
        new, note = resolve_path(value, base, cwd)
        if new != value:
            object.__setattr__(owner, field, new)   # no validate_assignment round trip
        if note:
            warnings.append(f'{label} {value!r}: {note}')
    for i, policy in enumerate(getattr(resolved.application, 'policies', None) or []):
        new, note = resolve_path(policy.card, base, cwd)
        if new != policy.card:
            object.__setattr__(policy, 'card', new)
        if note:
            warnings.append(f'application.policies[{i}].card {policy.card!r}: {note}')
    return resolved, warnings


__all__ = ['CWD_DEPRECATION', 'resolve_path', 'resolve_project_paths']
