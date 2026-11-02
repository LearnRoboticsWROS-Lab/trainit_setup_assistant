"""The Application Composer's Qt views (vNext I0): Step 7 for a Canonical Task Program.

Every view reads the ONE project held by the controller and edits it only through
``composer.ops.Composer`` (no GUI copy of the program; I0_DESIGN.md §13)."""

from .shell import ComposerWidget

__all__ = ['ComposerWidget']
