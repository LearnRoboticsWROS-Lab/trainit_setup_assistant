"""Application/task schema: waypoints, motion segments, tool actions.

Mapping note (critical): the GUI keeps motion/planner/speed on the *segment* (the
"between A and B, how?" wizard), but the generator FOLDS each segment onto its
destination waypoint, because that is exactly how ``bt_params.yaml`` stores it — every
waypoint carries its own incoming-motion spec.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from .enums import AppType, MotionType, PlannerId, ToolActionKind, WaypointRole, WaypointType


class Waypoint(BaseModel):
    name: str                                  # -> task_parameters key + MoveWaypoint id
    type: WaypointType = WaypointType.TCP
    position: Optional[List[float]] = None     # tcp: [x,y,z] in base_frame
    orientation: Optional[List[float]] = None  # tcp: [qx,qy,qz,qw]
    joints: Optional[List[float]] = None       # joint: explicit values (rad)
    named: Optional[str] = None                # joint: SRDF group_state name
    role: WaypointRole = WaypointRole.GENERIC
    # Start-state tolerance (rad) for the move that REACHES this waypoint: how far the
    # current state may drift from the trajectory's first point before execution is
    # refused. Tuned per waypoint (tight-space moves want more slack); 0.0 disables the
    # check. Default 0.1 mirrors the runtime's move_group default.
    allowed_start_tolerance: float = Field(default=0.1, ge=0.0)


class MotionSegment(BaseModel):
    """How to reach ``to_waypoint`` from the previous waypoint."""

    to_waypoint: str
    motion: MotionType = MotionType.FREE
    planner: Optional[PlannerId] = None        # override; None => global planner_mode
    speed: int = Field(default=50, ge=1, le=100)
    aux: Optional[List[float]] = None          # circ only: auxiliary point [x,y,z]
    aux_is_center: bool = False                # circ only


class ToolAction(BaseModel):
    """A tool/scene side-effect anchored at a waypoint (in tree order)."""

    at_waypoint: str
    kind: ToolActionKind
    payload_ref: Optional[str] = None          # SceneSpec.payload.id for attach/detach


class ApplicationSpec(BaseModel):
    type: AppType = AppType.PICK_AND_PLACE
    global_planner_mode: PlannerId = PlannerId.PILZ
    # Runtime/bt_params globals consumed by trainit_run_bt.
    bt_tree_id: str = 'MainTree'
    process_controller: str = 'mock'
    process_path_backend: str = 'pilz_sequence'
    enforce_validation: bool = True
    waypoints: List[Waypoint] = Field(default_factory=list)
    segments: List[MotionSegment] = Field(default_factory=list)
    tool_actions: List[ToolAction] = Field(default_factory=list)
    # Explicit tree order (waypoint-name references, repeats allowed — e.g. home at
    # start AND end). Empty => the application template derives order from roles.
    # Each waypoint is DEFINED once (bt_params) but may be VISITED multiple times.
    sequence: List[str] = Field(default_factory=list)

    def segment_for(self, waypoint_name: str) -> Optional[MotionSegment]:
        """The incoming segment for a waypoint, if any."""
        for seg in self.segments:
            if seg.to_waypoint == waypoint_name:
                return seg
        return None

    def waypoint_by_name(self, name: str) -> Optional[Waypoint]:
        for wp in self.waypoints:
            if wp.name == name:
                return wp
        return None

    def actions_at(self, waypoint_name: str) -> List[ToolAction]:
        """Tool actions anchored at a waypoint, in declared order."""
        return [a for a in self.tool_actions if a.at_waypoint == waypoint_name]
