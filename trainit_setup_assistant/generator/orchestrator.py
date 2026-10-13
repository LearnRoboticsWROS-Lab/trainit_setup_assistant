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
from ..catalog import snapshot_identity
from ..model import CanonicalProject
from .determinism import build_jinja_env
from .eef_contract import refuse_eef_contract
from .emitters.base import Emitter, GenContext
from .emitters.description_pkg import DescriptionEmitter
from .external_node_launch import refuse_incomplete_external_nodes
from .manifest import GenerationManifest
from .perception_yaml import refuse_uncomputable_candidate_rules


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


def refuse_application_problems(project: CanonicalProject) -> None:
    """REFUSE what the application template refuses, before a single file is written.

    A hard ``ValueError``, like :func:`refuse_uncomputable_candidate_rules`: an application
    type with no template used to raise a ``KeyError`` inside the app emitter, AFTER the
    description and moveit_config packages were already on disk.
    """
    if project.application is None:
        return
    from ..applications import get_application
    try:
        template = get_application(project.application.type)
    except KeyError as exc:
        raise ValueError(exc.args[0]) from None
    problems = template.refusals(project)
    if problems:
        raise ValueError('\n'.join(problems))


class Orchestrator:
    def __init__(self, emitters: Optional[List[Emitter]] = None):
        # None => derive the emitter pipeline per-project at generate() time.
        self._emitters = emitters

    def generate(
        self, project: CanonicalProject, output_root: Union[str, Path]
    ) -> GenerationManifest:
        # REFUSALS FIRST, before a single file exists (ADR-0017 Decision 3). Validation
        # problems are normally warnings — a partial bundle still helps — but a candidate
        # rule whose terms nobody can compute is not a partial bundle, it is a bundle that
        # runs and ranks nothing while looking like it ranks.
        refuse_uncomputable_candidate_rules(project)
        # Same reasoning for the external-node LAUNCH half (D-015): a block with no
        # package or no executable leaves a bundle that builds, launches, publishes
        # nothing on /perception/<name>/detections and times out in DetectObject three
        # layers from the cause. A detector with NO block at all is a warning, not a
        # refusal — starting a vendor stack yourself is a legitimate choice.
        refuse_incomplete_external_nodes(project)
        # And the application's own (vNext M0C): a template that does not exist, a holding
        # check that can never pass, a servo move whose shape would be ignored. Each one used
        # to surface mid-generation or not at all.
        refuse_application_problems(project)
        # And the end-effector contract's (ADR-0022 R1-R26): only when the project opts into it,
        # except `actuate` without one and the guard token / schema_version 3 without one.
        refuse_eef_contract(project)
        # And what the bundle will say compiled it (vNext M2.3): the committed catalog snapshot's
        # identity -- a catalog changed behind this process is refused here, before any file.
        identity = snapshot_identity(release=__version__)
        output_root = Path(output_root)
        output_root.mkdir(parents=True, exist_ok=True)

        manifest = GenerationManifest(
            generator_version=__version__, project_name=project.project_name,
            capability_snapshot=identity,
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

    # The end-effector contract's refusals hold here too (ADR-0022): this package carries the
    # adapter rows of BRIDGES, and the GUI's Step 3 must never write what Generate would refuse.
    refuse_eef_contract(project)
    identity = snapshot_identity(release=__version__)   # before any file, as in generate()
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = GenerationManifest(
        generator_version=__version__, project_name=package_name,
        capability_snapshot=identity)
    ctx = GenContext(output_root, build_jinja_env(), manifest)
    SceneLoaderMoveitConfigEmitter(package_name=package_name).emit(project, ctx)
    manifest.write(output_root)
    return manifest
