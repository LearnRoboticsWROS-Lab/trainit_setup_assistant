"""Orchestrator: CanonicalProject -> the 3-package bundle + manifest.

Emitters are registered incrementally as milestones land:
  M0  DescriptionEmitter
  M1+ MoveitConfigEmitter
  M3+ AppEmitter
The orchestrator owns ordering, the Jinja env, and the manifest; emitters only write
through the GenContext.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

from .. import __version__
from ..model import CanonicalProject
from .determinism import build_jinja_env
from .emitters.base import Emitter, GenContext
from .emitters.description_pkg import DescriptionEmitter
from .manifest import GenerationManifest


def default_emitters(project: Optional[CanonicalProject] = None) -> List[Emitter]:
    """The emitter pipeline for a project.

    The MoveIt-config emitter is chosen by whether the project references a hand-made
    BASE moveit_config (Step 2): when it does, the STANDALONE scene+planner emitter
    copies that base + adds planners + the scene loader (the MVP flow); otherwise the
    template-from-model emitter runs (the from-scratch / fr3wml golden path).
    """
    emitters: List[Emitter] = [DescriptionEmitter()]
    use_scene_loader = bool(project and project.robot.base_moveit_config_path)
    if use_scene_loader:
        from .emitters.scene_loader_moveit_config_pkg import SceneLoaderMoveitConfigEmitter
        emitters.append(SceneLoaderMoveitConfigEmitter())
    else:
        try:
            from .emitters.moveit_config_pkg import MoveitConfigEmitter
            emitters.append(MoveitConfigEmitter())
        except ImportError:
            pass
    try:
        from .emitters.app_pkg import AppEmitter
        emitters.append(AppEmitter())
    except ImportError:
        pass
    return emitters


class Orchestrator:
    def __init__(self, emitters: Optional[List[Emitter]] = None):
        # None => derive the emitter pipeline per-project at generate() time.
        self._emitters = emitters

    def generate(
        self, project: CanonicalProject, output_root: Union[str, Path]
    ) -> GenerationManifest:
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        manifest = GenerationManifest(
            generator_version=__version__, project_name=project.project_name
        )
        env = build_jinja_env()
        ctx = GenContext(output_root, env, manifest)

        emitters = self._emitters if self._emitters is not None else default_emitters(project)
        for emitter in emitters:
            emitter.emit(project, ctx)

        manifest.write(output_root)
        return manifest


def generate_scene_loader_config(
    project: CanonicalProject, output_root: Union[str, Path], package_name: str
) -> GenerationManifest:
    """Step 3: emit ONLY the standalone scene+planner moveit_config (for RViz config).

    Same emitter as the bundle's ``_trainit_config`` (so config == deploy), under the
    intermediate package name (e.g. ``<robot>_scene_loader_moveit_config``).
    """
    from .emitters.scene_loader_moveit_config_pkg import SceneLoaderMoveitConfigEmitter

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = GenerationManifest(
        generator_version=__version__, project_name=package_name)
    ctx = GenContext(output_root, build_jinja_env(), manifest)
    SceneLoaderMoveitConfigEmitter(package_name=package_name).emit(project, ctx)
    manifest.write(output_root)
    return manifest
