"""SceneImporter ABC — pluggable scene sources."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..model import SceneObject


class SceneImporter(ABC):
    @abstractmethod
    def import_objects(self, source) -> List[SceneObject]:
        """Return SceneObjects parsed from ``source``."""
