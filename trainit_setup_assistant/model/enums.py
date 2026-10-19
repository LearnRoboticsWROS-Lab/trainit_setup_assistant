"""Enumerations for the canonical project model.

All enums derive from ``str`` so they serialize to plain scalars in ``project.yaml``
and compare cleanly against the string tokens used by the engine's ``bt_params.yaml``
(e.g. ``motion: lin``, ``planner: pilz``).

Two enums are BUILT from TSA's committed catalog (ADR-0021), because TSA only passes their tokens
through to the component that owns them and so keeps no list of its own: ``PlannerId`` from the
copy of TMR's planner registry, ``CandidateRule`` from the snapshot of trainit_perception's
vocabulary (amendment M2.2). Every other enum here is TSA's own vocabulary (``ExecutionMode``
decides which BT node TSA emits, ``DetectionMethod`` which node TSA launches), kept as code and
parity-tested against the registry where the registry also speaks (test_planner_registry.py,
test_tp_registry.py).
"""

from enum import Enum

from ..catalog import (REFRESH, missing_required_modes, offered_planner_modes,
                       tmr_planner_registry, tp_registry, tp_vocabulary_problems)


class StrEnum(str, Enum):
    """str-backed enum: value is the YAML token; ``str(x)`` is the token too."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class WaypointType(StrEnum):
    """How a waypoint goal is specified."""

    TCP = 'tcp'      # cartesian: position + orientation in base_frame
    JOINT = 'joint'  # joint-space: explicit joints[] or a named SRDF state


class MotionType(StrEnum):
    """How to move INTO a waypoint (the incoming segment), mirroring bt_params."""

    PTP = 'ptp'    # point-to-point (Pilz PTP or planner-driven)
    LIN = 'lin'    # cartesian linear (always Pilz LIN in the engine)
    CIRC = 'circ'  # cartesian circular (always Pilz CIRC; needs aux point)
    FREE = 'free'  # collision-free planning (planner selectable)


# HOW free space is searched is NOT typed in here: the members are the modes TMR's planner
# registry offers (``catalog.offered_planner_modes``: deliverable and stable), read from TSA's
# committed copy of that registry -- never from the installed TMR (ADR-0021). Names are the
# tokens upper-cased.
_PLANNERS = tmr_planner_registry()
if missing_required_modes(_PLANNERS):
    raise RuntimeError(
        f"TMR's planner registry (catalog copy) no longer offers "
        f'{missing_required_modes(_PLANNERS)}, which TSA\'s own code names (the default planner '
        f'of a project file, the D-016 vision rule): change TSA first, then {REFRESH}')
PlannerId = StrEnum('PlannerId', [(mode.upper(), mode) for mode in
                                  offered_planner_modes(_PLANNERS)],
                    module=__name__, qualname='PlannerId')
PlannerId.__doc__ = """Per-segment planner override (acts on free/ptp; lin/circ are always Pilz).

The offered modes of TMR's planner registry, from TSA's committed copy of it (ADR-0021)."""


class ExecutionMode(StrEnum):
    """How a move is EXECUTED (ADR-0015) — its own axis, orthogonal to ``MotionType``
    (what SHAPE the path has) and to ``PlannerId`` (HOW free space is searched). Both of
    those are questions about a plan; this one says WHETHER there is a plan at all.

    PLAN  : plan the whole trajectory, then execute it (every move until ADR-0015).
            Emitted as ``<MoveWaypoint>``; costs ZERO extra bytes in the bundle.
    SERVO : stream the move with MoveIt Servo — an OUT-OF-PROCESS ``moveit_servo``
            node driven by the ``ServoTrack`` BT node (Decision 5: ``moveit_servo``
            2.5.10 on Humble is a node, not a library). Adds ``config/servo.yaml`` and
            a ``servo_node`` launch entry to the cell, both gated on this value.

    Deliberately NOT folded into ``MotionType``: that would make it accidentally
    exclusive with ``lin`` and leave no room for the third value this axis already has
    in the product (a learned policy, ADR-0005, today a separate ``RunPolicy`` node).
    """

    PLAN = 'plan'
    SERVO = 'servo'


class GripperKind(StrEnum):
    """End-effector KIND -- what the tool IS, never what serves it (ADR-0022 D6: the adapter is a
    separate row, ``model.end_effector.EndEffectorAdapter``)."""

    SUCTION = 'suction'    # SurfaceGripper-style, GripperCommand -> on/off services
    PARALLEL = 'parallel'  # jaw gripper (e.g. Robotiq), GripperCommand position
    # ADR-0022: explicit actuator and grasp-site maps, in any mix of standard interfaces. Only
    # with an end-effector contract; TSA 7.6.0 refuses the token at load (a silent-downgrade guard).
    COMPOSITE = 'composite'
    NONE = 'none'          # no gripper


class AppType(StrEnum):
    """Application templates. pick_and_place is the MVP; the rest are seams.

    Tokens are FROZEN once shipped (they live in every saved project.yaml); only the
    GUI labels may change — 'pick_and_place' is shown as "Blind pick and place" (D-015).
    """

    PICK_AND_PLACE = 'pick_and_place'
    VISION_GUIDED_MOTION = 'vision_guided_motion'   # D-015: editable sequence + vision
    # ADR-0016 Decision 4: the EMPTYING LOOP — ranked candidates, a retry over the
    # candidate index, a holding check after the lift, and a gated stall recovery. The
    # three things a LINEAR sequence cannot do, each one measured rather than supposed.
    BIN_PICKING = 'bin_picking'
    # vNext M3.3 (ADR-0024): the CANONICAL application -- not a template. Its waypoints, motion
    # profiles, paths and tasks are a Canonical Task Program (model.program), in schema_version 4.
    TASK_PROGRAM = 'task_program'
    # DEPRECATED (TSA 8.0, M3 decision 7): four application NAMES that never had a model or a template
    # and never generated a bundle. They stay members (old code and saved data name them) but a project
    # that says one is REFUSED at load, naming task_program: each is a composition of the canonical
    # steps, never a type of its own (DEPRECATED_APP_TYPES).
    GLUING = 'gluing'
    FOLLOW_PATH = 'follow_path'
    WAYPOINT_REPLAY = 'waypoint_replay'
    CNC = 'cnc'


# M3 decision 7 (TSA 8.0): the removed application tokens, and what each one is in a Canonical Task
# Program. A refusal at load, never a silent rewrite; every other token loads exactly as before.
DEPRECATED_APP_TYPES = {
    'gluing': 'a path with process events on a logical channel (process_on / process_off, set_value)',
    'follow_path': "a 'path' step (states and lin / circ segments, or a dense generator)",
    'waypoint_replay': "'move' steps to taught waypoints (in a 'repeat' for a cycle)",
    'cnc': ("machine tending: moves, 'grasp' / 'release', an 'external' process event on the machine's "
            "channel, 'task' calls"),
}


def deprecated_app_type_refusal(token: str, where: str = 'application.type') -> str:
    """The load refusal of a removed application token (M3 decision 7)."""
    return (f"{where} '{token}' is no longer an application type (TSA 8.0): it never generated a bundle, "
            f'and the same application is a Canonical Task Program -- {DEPRECATED_APP_TYPES[token]}. Set '
            "'application.type: task_program' and 'schema_version: 4', and write its waypoints, paths and "
            'tasks (README: Canonical Task Program)')


# trainit_perception's vocabulary, from TSA's committed snapshot of it -- never the installed TP
# (ADR-0021, amendment M2.2). What TSA's own code names in it must still be there.
_TP = tp_registry()
if tp_vocabulary_problems(_TP):
    raise RuntimeError(
        "trainit_perception's vocabulary (catalog snapshot) no longer fits TSA's code: "
        f'{"; ".join(tp_vocabulary_problems(_TP))}. If the snapshot was changed by hand or by a '
        f'merge, restore it with git or run {REFRESH}; if trainit_perception really changed, '
        f'change TSA first, then refresh')


class DetectionMethod(StrEnum):
    """How a detector produces the perception contract (D-014 taxonomy, D-015 kinds).

    COLOR_MASK : pure-python detector in trainit_perception — live-tunable in the
                 assistant, run by the generated detector_node.
    CUSTOM     : the expert seam — a user package/executable that emits the contract
                 (``vision_msgs/Detection3DArray``); config only, never a file upload.
    PCL_CLUSTER: external-node seam (roadmap) — a C++ PCL node publishing the same
                 contract; the assistant emits NO detector_node for it.
    """

    COLOR_MASK = 'color_mask'
    CUSTOM = 'custom'
    PCL_CLUSTER = 'pcl_cluster'


# WHICH candidate the detector publishes FIRST -- and therefore the RANKING. ADR-0016 Decision 1
# made the published order of the detection array the rank; ADR-0017 Decision 1 says who computes
# it: the DETECTOR, as one more detector parameter, travelling the road every tuned parameter
# travels (the GUI -> ``DetectorSpec`` -> ``perception.yaml`` -> ``make_detector``). The tokens are
# trainit_perception's (``auto`` and its rules, in its order), read from TSA's committed snapshot:
# TSA keeps no list of its own (ADR-0021, amendment M2.2). Names are the tokens upper-cased.
CandidateRule = StrEnum('CandidateRule', [(token.upper(), token) for token in
                                          [_TP['auto']] + list(_TP['rules'])],
                        module=__name__, qualname='CandidateRule')
CandidateRule.__doc__ = """WHICH candidate the detector publishes FIRST -- so the ranking.

``AUTO`` resolves the rule from what the detector can compute, in the default preference order;
every other member is a trainit_perception rule, a PRODUCT of named terms (``CANDIDATE_RULE_TERMS``
in ``model.perception``), from TSA's committed snapshot of TP's vocabulary (ADR-0021). Tokens are
FROZEN: they are written into ``perception.yaml`` and read by the runtime's own rule check."""

# The order a configurator is offered a rule in, best first -- trainit_perception's own preference
# (ADR-0017 Decision 4), from the snapshot. "Default" is the value the GUI PRE-SELECTS among the
# rules the chosen method can compute, NOT a model default: an unset rule writes NO key, so a
# project that never configured a ranking keeps generating byte-identical output.
DEFAULT_RULE_PREFERENCE = tuple(CandidateRule(token) for token in _TP['default_preference'])
DEFAULT_CANDIDATE_RULE = DEFAULT_RULE_PREFERENCE[0]


class SceneObjectSource(StrEnum):
    """Where a scene object came from."""

    PRIMITIVE = 'primitive'  # built in the GUI primitive editor
    USD = 'usd'              # imported from a .usd stage (Isaac)
    WORLD = 'world'          # imported from a Gazebo .world (SDF)


class ShapeType(StrEnum):
    """Collision shape.

    The scene loader (``scene_manager_node``) plans ``box``/``sphere``/``cylinder``
    AND ``mesh`` (``package://…stl`` via ``createMeshFromResource``). The in-tree
    ``AddCollisionObject`` BT node is BOX-only, so non-box primitives added *in the
    tree* are emitted as their AABB; ``mesh`` objects go through the scene loader
    (``scene.yaml``), not the tree.
    """

    BOX = 'box'
    SPHERE = 'sphere'
    CYLINDER = 'cylinder'
    CONE = 'cone'
    MESH = 'mesh'      # package:// STL, loaded by scene_manager_node (not AABB'd)


class SceneObjectCategory(StrEnum):
    """Cell role chosen when the USD scene is loaded; drives collision treatment.

    STATIC   : fixed structure (e.g. the BW-0080 machine) -> a CHECKED collision
               object the robot must avoid.
    ACTUATED : a URDF station whose mobile parts are driven by an adapter to
               simulate the strokes of moving components (e.g. prewash, belt).
               A CHECKED collision object today (future: articulated); it is NOT a
               manipulation target.
    DYNAMIC  : a manipulation target that appears in the scene but is collision-
               ALLOWED against EVERYTHING (robot links + every other mesh), so the
               gripper — and the meshes it rests on — may touch it (e.g. bottles,
               crate). Emitted with ``dynamic: true`` to the scene loader, whose
               ACM lets it collide with all.
    """

    STATIC = 'static'
    ACTUATED = 'actuated'
    DYNAMIC = 'dynamic'


class ReleasePolicy(StrEnum):
    """What a grasped dynamic object does in Isaac when the grip is released.

    FREEZE  : the object stays immobile where it is (kinematic) — models a second
              actuator (e.g. a PLC-driven clamp) taking over without simulating it.
    GRAVITY : the object becomes dynamic and falls under gravity.
    (There is no FLOAT: leaving a body gravity-free in mid-air is physically absurd.)
    """

    FREEZE = 'freeze'
    GRAVITY = 'gravity'


class IsaacGraspMethod(StrEnum):
    """Which Isaac-side physics realises the grasp (metadata; the actual Script Node
    lives hand-authored in the ``*_isaac`` package — see the generalized adapter).

    FIXED_JOINT    : one rigid FixedJoint per object EE<->object (v14) — deterministic,
                     no oscillation; the method that stabilised the 20 bottles.
    SURFACE_GRIPPER : IsaacSurfaceGripper suction with N attachment points (compliant).
    """

    FIXED_JOINT = 'fixed_joint'
    SURFACE_GRIPPER = 'surface_gripper'


class WaypointRole(StrEnum):
    """Semantic role of a waypoint, used by application templates to anchor actions."""

    HOME = 'home'
    PRE_PICK = 'pre_pick'
    PICK = 'pick'
    POST_PICK = 'post_pick'
    PRE_PLACE = 'pre_place'
    PLACE = 'place'
    POST_PLACE = 'post_place'
    GENERIC = 'generic'


class ToolActionKind(StrEnum):
    """Tool/scene side-effect anchored at a waypoint."""

    GRASP = 'grasp'      # CloseGripper / SetGripper(close)
    RELEASE = 'release'  # OpenGripper / SetGripper(open)
    ATTACH = 'attach'    # AttachObject (payload follows the tcp)
    DETACH = 'detach'    # DetachObject
    RESET_SCENE = 'reset_scene'  # ResetScene: dynamic objects -> initial poses (sim loop)
    # ADR-0022: CommandActuator -- one actuator of the end-effector contract to a NAMED target
    # (``ToolAction.actuator`` / ``target``). Refused without a contract (R11).
    ACTUATE = 'actuate'


class PolicyMode(StrEnum):
    """How a trained (learned) policy is executed at a step — POLICY_EXECUTION.md,
    ADR-0005. Tokens match the trainit_policy_runtime node's `mode` and TMR's RunPolicy.

    PURE     : the policy DRIVES the robot to its end state; it replaces the
               deterministic move into the anchored waypoint.
    HYBRID   : the policy DECIDES a target pose; the deterministic runtime moves there
               (reuses DetectObject + SetWaypointFromDetection + MoveWaypoint) — smooth
               motion for free. Recommended for pick-place.
    RESIDUAL : the deterministic move runs and the policy adds a small clamped
               correction on top.
    """

    PURE = 'pure'
    HYBRID = 'hybrid'
    RESIDUAL = 'residual'


class PolicyCategory(StrEnum):
    """WHAT a learned policy does (its skill/intent) — ADR-0006. Orthogonal to PolicyMode
    (how it executes) and the end-effector (which tool). Declared in the policy CARD
    (intrinsic to the policy), read by TSA for validation/UI. Only REACH_GRASP is
    implemented; the rest are RESERVED — a new skill is a card value + one runtime branch,
    not a rewrite. Tokens match trainit_policy_runtime's policy_card.CATEGORIES.
    """

    REACH_GRASP = 'reach_grasp'    # reach a detected object and grasp it (implemented)
    PLACE = 'place'                # reserved: place a held object at a target
    INSERT = 'insert'              # reserved: contact-rich insertion (typically residual)
    PUSH = 'push'                  # reserved: non-prehensile push/align
    VISUAL_SERVO = 'visual_servo'  # reserved: servo the tcp to a viewpoint


# Categories TSA will emit/accept today; the rest are reserved (ADR-0006).
IMPLEMENTED_POLICY_CATEGORIES = (PolicyCategory.REACH_GRASP,)


class EndEffector(StrEnum):
    """WHICH tool a learned policy was trained with (ADR-0006) — intrinsic to the policy
    (declared in the card, inferred from grasp.model when absent). Only SUCTION is
    implemented; the rest are reserved. Tokens match trainit_policy_runtime's
    policy_card.END_EFFECTORS."""

    SUCTION = 'suction'                # implemented (attach-on-contact)
    PARALLEL_GRIPPER = 'parallel_gripper'   # reserved
    NONE = 'none'                      # reserved (non-prehensile)


IMPLEMENTED_END_EFFECTORS = (EndEffector.SUCTION,)


class CalibrationFrame(StrEnum):
    """Frame a deploy calibration offset is expressed in (ADR-0006)."""

    BASE = 'base_link'
    TCP = 'tcp'


class Backend(StrEnum):
    """A cell EXECUTION backend (ADR-0008) — the environment the same application runs
    in. The ROS contract is IDENTICAL across all of them (invariant 5): arm
    ``follow_joint_trajectory``, gripper/suction commands, ``/joint_states``; that is why
    one bundle runs everywhere. Selected at runtime by the generated bring-up's ``mode:=``
    launch argument; the ``DeploymentSpec.modes`` list is baked into each launch's
    ``VALID_MODES``.

    Tokens are FROZEN (they live in every saved ``project.yaml`` and every generated
    bring-up). MOCK/ISAAC/GAZEBO/REAL are all emitted by the generator (GAZEBO added in
    TSA v6 / ADR-0008 F2 — the Gazebo Classic backend the ROS2ML distribution needs; its
    live headless launch is the user's MANUAL generation step). Per-backend facts live in
    ``model/backend.py`` (``BackendProfile``).
    """

    MOCK = 'mock'      # ros2_control mock_components/GenericSystem, wall clock (RViz/CI)
    ISAAC = 'isaac'    # Isaac Sim via topic_based_ros2_control/TopicBasedSystem, sim clock
    GAZEBO = 'gazebo'  # Gazebo Classic, gazebo_ros2_control/GazeboSystem, in-gzserver CM
    REAL = 'real'      # vendor bridges impersonate the controllers; no ros2_control node


# Backends the generator emits today (all four; GAZEBO wired in ADR-0008 F2).
IMPLEMENTED_BACKENDS = (Backend.MOCK, Backend.ISAAC, Backend.GAZEBO, Backend.REAL)


class EndEffectorActuation(StrEnum):
    """How the BT commands the end-effector on grasp/release (ADR-0010; ``contract``: ADR-0022).
    Orthogonal to the sim grasp adapter."""

    TRIGGER = 'trigger'                # on/off signal to a service (suction, weld, on/off tool)
    JOINT_POSITION = 'joint_position'  # GripperCommand a joint to open/closed targets
    # ADR-0022 D6: the end-effector CONTRACT commands the tool. Set by the gripper's pre-validator
    # whenever ``contract`` is present (never chosen): the guard that makes TSA 7.6.0 refuse a
    # canonical file at load instead of silently regenerating it as legacy.
    CONTRACT = 'contract'


class GripperJointTarget(StrEnum):
    """For JOINT_POSITION actuation, how an open/closed joint target is defined (ADR-0010)."""

    SRDF_STATE = 'srdf_state'  # a named SRDF group_state (open/closed), read from the moveit_config
    ANGLE = 'angle'            # an explicit joint angle (captured live from MoveIt, or typed)


class SimGraspAdapter(StrEnum):
    """Which simulator-physics trick makes a grasped object stick (ADR-0010). Selected PER
    BACKEND: a sim gripper rarely holds by friction, so the adapter is fired on grasp and
    released on open. The adapter itself is adaptation-layer (in the USD / .world + the kit)."""

    SURFACE_GRIPPER = 'surface_gripper'  # Isaac USD SurfaceGripper property (suction/on|off)
    LINK_ATTACHER = 'link_attacher'      # Gazebo LinkAttacher weld (attach/detach; kit bridge)
    NONE = 'none'                        # no trick — real physics (mode:=real) or a physics grasp


# Sensible per-backend grasp-adapter defaults (the user overrides per backend at Step 7).
DEFAULT_SIM_GRASP_ADAPTER = {
    Backend.ISAAC: SimGraspAdapter.SURFACE_GRIPPER,
    Backend.GAZEBO: SimGraspAdapter.LINK_ATTACHER,
    Backend.REAL: SimGraspAdapter.NONE,
    Backend.MOCK: SimGraspAdapter.NONE,
}
