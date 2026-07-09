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


def default_emitters() -> List[Emitter]:
    """The emitter pipeline available at the current milestone."""
    emitters: List[Emitter] = [DescriptionEmitter()]
    # M1+: append MoveitConfigEmitter; M3+: append AppEmitter (wired as they land).
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
        self.emitters = emitters if emitters is not None else default_emitters()

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

        for emitter in self.emitters:
            emitter.emit(project, ctx)

        manifest.write(output_root)
        return manifest
