"""GUI layer: a Qt wizard over the ROS-free AssistantController.

Seam: widgets only call the controller; the controller edits the CanonicalProject.
This keeps the GUI testable (drive the controller / offscreen Qt) and lets a future
web GUI reuse the controller unchanged.
"""

from .controller import AssistantController

__all__ = ['AssistantController']
