"""Application templates + the shared bt_params builder.

The ``bt_params.yaml`` schema is uniform across applications (a flat set of runtime
globals + per-waypoint motion specs); only the BEHAVIOR TREE differs per app. So the
bt_params context is built here once, and each :class:`ApplicationTemplate` only emits
its tree.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..model import CanonicalProject
from ..model.enums import WaypointType


# --- scene/payload key naming (shared by bt_params AND the tree) ---
def scene_dims_key(obj_id: str) -> str:
    return f'scene_{obj_id}_dims'


def scene_pos_key(obj_id: str) -> str:
    return f'scene_{obj_id}_pos'


def payload_dims_key(payload_id: str) -> str:
    return f'payload_{payload_id}_dims'


def build_bt_params_context(project: CanonicalProject) -> dict:
    """Prepare the context the ``app/bt_params.yaml.j2`` template renders.

    Folds each waypoint's incoming MotionSegment onto the waypoint block (exactly how
    bt_params stores motion), and lays out the flat scene/payload params the tree
    references by ``{key}``.
    """
    app = project.application
    robot = project.robot

    globals_ctx = {
        'planner_mode': app.global_planner_mode.value,
        'process_path_backend': app.process_path_backend,
        'group': robot.planning_group.name,
        'base_frame': robot.base_frame,
        'tip_link': robot.tip_link,
        'gripper_action': robot.gripper.gripper_action_ns(),
        'process_controller': app.process_controller,
        'bt_tree_id': app.bt_tree_id,
        'enforce_validation': app.enforce_validation,
    }

    scene_params: List[tuple] = []
    for obj in project.scene.objects:
        if obj.is_planning_collision():
            scene_params.append((scene_dims_key(obj.id), obj.aabb_dims()))
            scene_params.append((scene_pos_key(obj.id), list(obj.position)))

    payload = None
    if project.scene.payload:
        payload = (payload_dims_key(project.scene.payload.id),
                   list(project.scene.payload.dims))

    waypoints = []
    for wp in app.waypoints:
        seg = app.segment_for(wp.name)
        block = {
            'name': wp.name,
            'type': wp.type.value,
            'position': list(wp.position) if wp.position else None,
            'orientation': list(wp.orientation) if wp.orientation else None,
            'joints': list(wp.joints) if wp.joints else None,
            'named': wp.named,
            'motion': seg.motion.value if seg else None,
            'planner': (seg.planner.value if (seg and seg.planner) else None),
            'speed': seg.speed if seg else None,
            'aux': (list(seg.aux) if (seg and seg.aux) else None),
            'aux_is_center': (seg.aux_is_center if seg else False),
            'is_tcp': wp.type is WaypointType.TCP,
        }
        waypoints.append(block)

    return {
        'g': globals_ctx,
        'scene_params': scene_params,
        'payload': payload,
        'waypoints': waypoints,
    }


class ApplicationTemplate(ABC):
    """Maps a configured project onto one application's Behavior Tree."""

    app_type: str

    @abstractmethod
    def validate(self, project: CanonicalProject) -> List[str]:
        """Return a list of human-readable problems (empty == OK)."""

    @abstractmethod
    def tree_filename(self, project: CanonicalProject) -> str:
        """File name for the generated BT XML (e.g. 'pick_place.xml')."""

    @abstractmethod
    def build_tree_xml(self, project: CanonicalProject) -> str:
        """Return the Behavior Tree XML for this application."""
