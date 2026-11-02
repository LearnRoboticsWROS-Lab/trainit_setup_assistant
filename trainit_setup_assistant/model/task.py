"""Application/task schema: waypoints, motion segments, tool actions.

Mapping note (critical): the GUI keeps motion/planner/speed on the *segment* (the
"between A and B, how?" wizard), but the generator FOLDS each segment onto its
destination waypoint, because that is exactly how ``bt_params.yaml`` stores it — every
waypoint carries its own incoming-motion spec.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, root_validator, validator

from .enums import (
    DEPRECATED_APP_TYPES,
    deprecated_app_type_refusal,
    AppType,
    CalibrationFrame,
    ExecutionMode,
    MotionType,
    PlannerId,
    PolicyMode,
    ToolActionKind,
    WaypointRole,
    WaypointType,
)
from .perception import RelativeBinding, VisionBinding
from .program import PROGRAM_KEYS, TaskProgram


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
    # None = no planner chosen: in a CANONICAL project generation writes the planner TMR runs
    # for it instead (effective_planner below: ptp -> pilz, free -> the global planner_mode;
    # vNext M2.7 D3, canonical-only since G0); a legacy project writes none, as before M2.7
    planner: Optional[PlannerId] = None
    # HOW this move is executed (ADR-0015): plan the trajectory and run it, or stream it
    # with MoveIt Servo. Its own axis — orthogonal to `motion` and to the planner. Like
    # `motion` (and unlike `planner`) it is NON-optional with a default: the semantics are
    # ABSOLUTE, not inherit-from-global, so there is no ApplicationSpec.global_execution.
    # `plan` emits zero bytes: the BT node name carries the choice (<MoveWaypoint> vs
    # <ServoTrack>) and servo.yaml + the servo_node launch entry are gated on it.
    execution: ExecutionMode = ExecutionMode.PLAN
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


def effective_planner(motion: Optional[MotionType], planner: Optional[PlannerId],
                      global_mode: PlannerId) -> Optional[PlannerId]:
    """The planner TMR's ``MoveWaypoint`` RUNS for a move into a waypoint -- the ONE meaning of a
    blank planner in a CANONICAL project (one with an end-effector contract, ADR-0022; vNext M2.7,
    owner decision D3, scoped to canonical projects by the owner at vNext G0). For such a project
    generation writes it explicitly into ``bt_params.yaml``, so no generated move relies on TMR's
    rules for a blank planner, and D-016 and the blocks page's switch to free/OMPL judge it. A
    legacy project keeps the reading it had before M2.7: nothing is written for a blank planner,
    and D-016 and the switch read it as the global mode. ``motion`` None = the waypoint has no
    incoming segment (TMR's motion default is free).

    * a chosen planner: itself, unchanged;
    * ``lin`` / ``circ``: None -- always Pilz LIN / CIRC; TMR never reads the planner, so nothing
      is written for a blank one (an explicit one stays, inert: KNOWN_DEBT);
    * ``ptp``, blank: ``pilz`` -- TMR runs a blank ptp as Pilz PTP (``bt_nodes.cpp``
      ``MoveWaypoint::tick``: ``else if (motion == "ptp") mode = PlannerMode::PILZ``);
    * ``free`` (or no motion), blank: ``global_mode`` -- TMR runs it with the runner's
      ``planner_mode``, which every generated launch sets from the same ``global_planner_mode``.

    The execution mode does not enter: ``bt_params`` is the same block for a planned and a streamed
    move (ServoTrack reads no planner), as before.
    """
    if planner is not None:
        return planner
    if motion in (MotionType.LIN, MotionType.CIRC):
        return None
    if motion is MotionType.PTP:
        return PlannerId.PILZ
    return global_mode


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
    # --- the end-effector contract (ADR-0022): logical ids, never an endpoint -----------------
    # Omitted from project.yaml when None (model/io.py), so a legacy project saves byte-identical.
    actuator: Optional[str] = None             # actuate: the contract actuator commanded
    target: Optional[str] = None               # actuate: one of its NAMED targets
    site: Optional[str] = None                 # grasp/release: the grasp site (multi-site device)


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


class BinPickingSpec(BaseModel):
    """The emptying loop's configuration (ADR-0016 Decision 4).

    Everything here is CONFIGURATION, not code: the number of candidates to try, whether
    the lift is verified, and the recovery step fired when the bin stalls. What the
    recovery step ACTUATES is cell I/O (ADR-0013) — a shake is a machine with an
    amplitude, a frequency and a direction; in the twin it is one line of physics and in
    a cell it is a signal on a PLC. The template emits the *step* only.

    The standoff and retreat DISTANCES are configuration too, and they already have a
    home: the approach waypoint's ``Waypoint.vision.dz`` and the retreat waypoint's
    ``Waypoint.relative.dz``. They are deliberately not duplicated here.
    """

    # Candidates tried per delivered object, in the detector's published order (ADR-0016
    # Decision 1). Also DetectObject's ``keep``: asking for more candidates than the loop
    # can try would publish blackboard entries nothing reads. Measured on the reference
    # cell: K = 3,33 candidates tried per object delivered, so 3 is the floor that makes
    # the retry worth having — one pick in three does not succeed on the first candidate.
    tries: int = Field(default=3, ge=1)
    # CheckRobotState after the retreat: what turns "the cup closed" into "the capsule
    # came up". The two differ on 1 lift in 22 on the reference cell, which is exactly the
    # rate a linear sequence would silently report as a successful pick.
    holding_check: bool = True
    # std_msgs/Bool, true while the object is held — the signal the holding check reads.
    # Empty => CheckRobotState is emitted without it and the runtime uses its own default.
    attached_topic: str = ''
    holding_timeout_ms: int = Field(default=3000, ge=100)
    # --- the stall recovery (cell I/O, ADR-0013) ---
    # The ProcessController channel pulsed when the bin stalls. EMPTY = no recovery step:
    # the tree then only logs "bin empty" and the next cycle re-detects. "No plan" and
    # "the plans do not work" are the same stall to an operator, and treating only the
    # first is why an early measurement reported N = 15 where the recovered one reports 21.
    recovery_channel: str = ''
    # How long the channel stays enabled (a <Sleep> between Enable and DisableProcess).
    recovery_dwell_ms: int = Field(default=1500, ge=0)


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
    # The emptying loop's configuration (ADR-0016 Decision 4). Read ONLY by the
    # bin_picking template, so every other application ignores it and a project that
    # never selects bin_picking generates exactly what it generated before.
    bin_picking: BinPickingSpec = Field(default_factory=BinPickingSpec)
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
    # vNext M3.3 (ADR-0024): the Canonical Task Program of a ``task_program`` application, None for
    # a template application. In project.yaml its sections sit directly under ``application:``
    # (providers, motion_profiles, waypoints, paths, tasks, main_task, runtime); the pre-validator
    # below moves them here, so the template fields above stay at their defaults and unused.
    program: Optional[TaskProgram] = None

    @root_validator(pre=True)
    def _task_program_sections(cls, values):  # noqa: N805
        """A ``task_program`` application holds the program's sections and nothing else (R-M3-01: a
        task program does not mix with a template's waypoints, segments, tool actions, policies or
        loop); any other application keeps reading as before -- a stray ``program`` key is ignored,
        as every unknown key of this model always was."""
        if not isinstance(values, dict):
            return values
        token = getattr(values.get('type'), 'value', values.get('type'))
        if isinstance(token, str) and token in DEPRECATED_APP_TYPES:      # M3 decision 7 (TSA 8.0)
            raise ValueError(deprecated_app_type_refusal(token))
        if token != AppType.TASK_PROGRAM.value:
            values = dict(values)
            values.pop('program', None)
            return values
        if set(values) <= {'type', 'program'} and isinstance(values.get('program'), TaskProgram):
            return values                     # built in code: ApplicationSpec(type=..., program=...)
        sections = {k: v for k, v in values.items() if k in PROGRAM_KEYS}
        foreign = [k for k in values if k != 'type' and k not in PROGRAM_KEYS]
        if foreign:
            from .program import PROGRAM_HINTS
            raise ValueError('R-M3-01: a Canonical Task Program (application type task_program): '
                             + '; '.join(PROGRAM_HINTS.get(k, f"has no '{k}' (its sections are "
                                                              f"{', '.join(PROGRAM_KEYS)})")
                                         for k in foreign))
        return {'type': values['type'], 'program': sections}

    @root_validator(skip_on_failure=True)
    def _program_iff_task_program(cls, values):  # noqa: N805
        is_program = values.get('type') is AppType.TASK_PROGRAM
        if is_program and values.get('program') is None:
            raise ValueError('R-M3-01: a task_program application needs its program: tasks and a '
                             'main_task at least')
        if not is_program and values.get('program') is not None:
            raise ValueError(f"R-M3-01: only a task_program application has a program (this one is "
                             f"'{values.get('type')}')")
        return values

    def is_task_program(self) -> bool:
        return self.type is AppType.TASK_PROGRAM

    def segment_for(self, waypoint_name: str) -> Optional[MotionSegment]:
        """The incoming segment for a waypoint, if any."""
        for seg in self.segments:
            if seg.to_waypoint == waypoint_name:
                return seg
        return None

    def uses_servo(self) -> bool:
        """True when at least one move is streamed with MoveIt Servo (ADR-0015).

        THE gate for every servo artefact: ``config/servo.yaml`` and the ``servo_node``
        launch entry are emitted only when this is True, so an all-``plan`` project (the
        golden) is byte-identical to a pre-ADR-0015 bundle.
        """
        return any(seg.execution is ExecutionMode.SERVO for seg in self.segments)

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
