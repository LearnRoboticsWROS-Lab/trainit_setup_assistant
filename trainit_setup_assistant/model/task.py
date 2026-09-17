"""Application/task schema: waypoints, motion segments, tool actions.

Mapping note (critical): the GUI keeps motion/planner/speed on the *segment* (the
"between A and B, how?" wizard), but the generator FOLDS each segment onto its
destination waypoint, because that is exactly how ``bt_params.yaml`` stores it — every
waypoint carries its own incoming-motion spec.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, validator

from .enums import (
    AppType,
    CalibrationFrame,
    MotionType,
    PlannerId,
    PolicyMode,
    ToolActionKind,
    WaypointRole,
    WaypointType,
)
from .perception import RelativeBinding, VisionBinding


class Waypoint(BaseModel):
    name: str                                  # -> task_parameters key + MoveWaypoint id
    type: WaypointType = WaypointType.TCP
    position: Optional[List[float]] = None     # tcp: [x,y,z] in base_frame
    orientation: Optional[List[float]] = None  # tcp: [qx,qy,qz,qw]
    joints: Optional[List[float]] = None       # joint: explicit values (rad)
    named: Optional[str] = None                # joint: SRDF group_state name
    role: WaypointRole = WaypointRole.GENERIC
    # Vision-driven waypoint (D-014): when set, the tree overwrites this waypoint's
    # position (and, for a joint waypoint, promotes it to tcp) at RUN TIME via
    # SetWaypointFromDetection; the captured pose below stays as the recorded fallback.
    vision: Optional[VisionBinding] = None
    # Step-relative waypoint (D-017): pose derived at RUN TIME from another step's
    # final pose via SetWaypointRelative. Mutually exclusive with `vision`.
    relative: Optional[RelativeBinding] = None
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
    # Per-move planning-collision check for GRASPED objects: when the gripper holds
    # dynamic objects, ON => they are checked vs the static/actuated meshes (the planner
    # routes the held payload around them, e.g. into the prewash); OFF => transparent.
    # None => inherit the current runtime state (scene default). Emitted as a
    # SetAttachedCollisionCheck BT node before this move when not None.
    attached_collision_check: Optional[bool] = None
    # Process-layer pause AFTER this waypoint (and its tool actions): a Wait block in
    # the sequence editor. Emitted as the BT.CPP built-in <Sleep msec="..."/>.
    wait_after_ms: int = Field(default=0, ge=0)


class DetectPoint(BaseModel):
    """An EXPLICIT camera-sampling point in the flow (D-018): DetectObject for
    ``detector`` runs immediately BEFORE the move to ``before_waypoint``, updating
    every Move bound to that detector. Its position answers "WHEN does vision
    fire" — several detectors may sample at different points of one cycle. With no
    DetectPoint for a detector, sampling is automatic at cycle start."""

    detector: str
    before_waypoint: str


class ToolAction(BaseModel):
    """A tool/scene side-effect anchored at a waypoint (in tree order)."""

    at_waypoint: str
    kind: ToolActionKind
    payload_ref: Optional[str] = None          # SceneSpec.payload.id for attach/detach


class PolicyStep(BaseModel):
    """A learned-policy step that produces the motion INTO ``before_waypoint``
    (POLICY_EXECUTION.md, ADR-0005). The Pro package ``trainit_policy_runtime`` loads
    the policy + its card and runs it; TMR's ``RunPolicy`` / ``CheckRobotState`` nodes
    drive and check it. TSA stays config-only: it emits the nodes, reads the card for
    what to show/check, and ships the policy files with the bundle.

    Placement mirrors ``DetectPoint``: the step is anchored to the waypoint whose
    incoming motion the policy produces. By mode:
    - ``hybrid``   : the policy publishes a target pose; the deterministic MoveWaypoint
      into ``before_waypoint`` executes it (kept).
    - ``pure`` / ``residual`` : the policy drives the robot into ``before_waypoint``; the
      deterministic MoveWaypoint is suppressed.
    A ``CheckRobotState`` after asserts the card's ``end_state`` (fail-safe).
    """

    name: str                                   # unique id; also the run namespace
    before_waypoint: str                        # the policy produces the move into this wp
    mode: PolicyMode = PolicyMode.HYBRID
    card: str                                   # path to the policy card YAML (.pt sits beside it)
    # runtime wiring (defaults match trainit_policy_runtime's launch)
    run_service: str = '/trainit_policy_runtime/run'
    status_topic: str = '/trainit_policy_runtime/status'
    target_topic: str = '/trainit_policy_runtime/target_detections'   # hybrid
    timeout_ms: int = Field(default=10000, ge=100)
    # end_state check emitted as CheckRobotState AFTER the policy
    check_position: bool = True                 # assert tcp near the card's end_state pose
    position_tolerance: float = Field(default=0.03, gt=0.0)
    require_attached: bool = False              # assert the object is held (suction)
    attached_topic: Optional[str] = None        # std_msgs/Bool, true while held
    # deploy calibration (ADR-0006) — a post-hoc correction of a SYSTEMATIC policy bias,
    # applied by the runtime to the policy's decided/driven pose. A band-aid: prefer
    # retraining. None = no calibration (the golden bundle stays byte-identical).
    calibration_offset: Optional[List[float]] = None   # [dx,dy,dz, droll,dpitch,dyaw]
    calibration_frame: CalibrationFrame = CalibrationFrame.BASE
    calibration_note: str = ''                  # why it is set (for the record)

    # Classic @validator (not @field_validator): the deprecated-but-functional API that
    # runs on BOTH pydantic 1.9 (apt python3-pydantic) and 2.x, so TrainIt installs via
    # rosdep with no pip. Semantics identical (post-coercion, value-only).
    @validator('calibration_offset')
    def _check_calibration(cls, v):
        if v is not None and len(v) != 6:
            raise ValueError('calibration_offset needs 6 values '
                             '[dx, dy, dz, droll, dpitch, dyaw]')
        return v


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
    # explicit camera-sampling points (D-018); empty = automatic at cycle start
    detections: List[DetectPoint] = Field(default_factory=list)
    # learned-policy steps (ADR-0005); empty = fully deterministic (unchanged behaviour)
    policies: List[PolicyStep] = Field(default_factory=list)
    # Explicit tree order (waypoint-name references, repeats allowed — e.g. home at
    # start AND end). Empty => the application template derives order from roles.
    # Each waypoint is DEFINED once (bt_params) but may be VISITED multiple times.
    sequence: List[str] = Field(default_factory=list)
    # Process-layer loop over the WHOLE sequence (the Loop block): 0 = run once,
    # -1 = repeat forever, N>0 = repeat N times. Emitted as the BT.CPP built-in
    # <Repeat num_cycles="..."> around the move body.
    loop_cycles: int = 0
    # Which slice of `sequence` the loop wraps (inclusive waypoint names). Both None =
    # the WHOLE sequence, which is what loop_cycles alone meant before — so every
    # existing project.yaml keeps its behaviour untouched.
    loop_start: Optional[str] = None
    loop_end: Optional[str] = None

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

    def policy_before(self, waypoint_name: str) -> Optional['PolicyStep']:
        """The learned-policy step producing the move into a waypoint, if any."""
        for p in self.policies:
            if p.before_waypoint == waypoint_name:
                return p
        return None
