"""Build the ``config/servo.yaml`` context for a cell that streams a move (ADR-0015).

MoveIt Servo is configured as an OUT-OF-PROCESS ``moveit_servo`` node (ADR-0015
Decision 5: on Humble, ``moveit_servo`` 2.5.10 is a node, not a library), so its
configuration is a cell artefact exactly like ``scene.yaml``: emitted into the cell
MoveIt-config package and handed to the ``servo_node`` the generated bring-up starts.

Only the FRAME/TOPIC/GROUP fields are derived — every one of them from the cell model,
never hardcoded. The tuning values in the template are the ones a real ``servo_node``
was brought up with against a real namespaced cell and verified streaming twists end to end;
they are cell-independent, so they live in the template as documented constants.
"""

from __future__ import annotations

from ..model import CanonicalProject
# THE hook for the cell namespace (ADR-0016 Decision 3). It lives in `namespace.py` now,
# because it is no longer only servo's business — the bring-up, the gripper action and the
# BT application derive from the same datum — and is re-exported here so this module's
# original entry point keeps working.
from .namespace import cell_namespace, topic_under as _topic

__all__ = ['cell_namespace', 'build_servo_context']


def build_servo_context(project: CanonicalProject) -> dict:
    """The template context for ``moveit_config/servo.yaml.j2``.

    Every value comes from the cell model:

    - ``move_group_name``          <- the planning group (the group Servo commands)
    - ``planning_frame`` and ``robot_link_command_frame`` <- the robot base frame
      (commands must be given in the frame of a robot link; the base is the stable one)
    - ``ee_frame_name``            <- the tip link (the EE pose Servo reports)
    - ``joint_topic``              <- the cell namespace (the COMPLETE joint state: with
      a joint-state merger in the cell it is the merged topic, which is what Servo needs)
    - ``command_out_topic``        <- the cell namespace + the arm controller name, i.e.
      the SAME ``JointTrajectoryController`` MoveIt execution already feeds, which is why
      no extra controller or bridge is needed (ADR-0015) — and why arbitration matters
    - ``monitored_planning_scene_topic`` <- the cell namespace; Servo SUBSCRIBES to
      ``move_group``'s scene (is_primary_planning_scene_monitor is false), so the one
      ``scene.yaml`` serves the planned path and the streamed path alike
    """
    robot = project.robot
    ns = cell_namespace(project)
    return {
        'move_group_name': robot.planning_group.name,
        'planning_frame': robot.base_frame,
        'robot_link_command_frame': robot.base_frame,
        'ee_frame_name': robot.tip_link,
        'joint_topic': _topic(ns, 'joint_states'),
        'command_out_topic': _topic(ns, f'{robot.arm_controller.name}/joint_trajectory'),
        'monitored_planning_scene_topic': _topic(ns, 'monitored_planning_scene'),
    }
