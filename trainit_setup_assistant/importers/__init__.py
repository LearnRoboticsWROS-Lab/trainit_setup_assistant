"""Scene importers: turn external sources into SceneObjects.

primitive (GUI-built) is the core path; USD import is optional (needs pxr / usd-core).
"""

from .base import SceneImporter
from .usd_importer import import_usd, usd_available

__all__ = ['SceneImporter', 'import_usd', 'usd_available']
