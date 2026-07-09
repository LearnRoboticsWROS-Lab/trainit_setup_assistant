"""TrainIt Setup Assistant.

A GUI + headless generator that turns a robot + a configured application into a
buildable ROS 2 bundle (description + moveit_config + app) for the TrainIt Motion
Runtime — the way the MoveIt Setup Assistant turns a robot into a moveit_config.

Architecture seam: the GUI edits a single :class:`CanonicalProject` (``model/``);
the generator (``generator/``) only reads it. They never talk directly, so a future
web GUI can replace ``gui/`` by emitting the same project. ``trainit_generate``
(headless) is the contract test that keeps them decoupled.
"""

__version__ = '0.1.0'
