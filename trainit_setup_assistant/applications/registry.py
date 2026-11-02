"""Application template registry. pick_and_place is the MVP; others are seams."""

from __future__ import annotations

from typing import Dict

from ..model.enums import AppType
from .base import ApplicationTemplate
from .bin_picking import BinPicking
from .pick_and_place import PickAndPlace
from .task_program import TaskProgramApplication
from .vision_guided_motion import VisionGuidedMotion

_REGISTRY: Dict[str, ApplicationTemplate] = {
    AppType.PICK_AND_PLACE.value: PickAndPlace(),
    AppType.VISION_GUIDED_MOTION.value: VisionGuidedMotion(),
    # ADR-0016 Decision 4: the emptying loop (ranked candidates + retry +
    # holding check + gated stall recovery). A subclass of the vision template,
    # so every vision/relative check is inherited rather than re-derived.
    AppType.BIN_PICKING.value: BinPicking(),
    # vNext M3.3 (ADR-0024): the Canonical Task Program -- not a template: the project's own
    # program, validated and resolved before any file (applications/task_program.py).
    AppType.TASK_PROGRAM.value: TaskProgramApplication(),
}


def get_application(app_type) -> ApplicationTemplate:
    """Return the template for an application type (enum or string)."""
    key = app_type.value if hasattr(app_type, 'value') else str(app_type)
    if key not in _REGISTRY:
        raise KeyError(
            f'no application template for "{key}" '
            f'(available: {", ".join(sorted(_REGISTRY))})'
        )
    return _REGISTRY[key]


def available_applications():
    return sorted(_REGISTRY)
