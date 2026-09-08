"""Enumerations for the canonical project model.

All enums derive from ``str`` so they serialize to plain scalars in ``project.yaml``
and compare cleanly against the string tokens used by the engine's ``bt_params.yaml``
(e.g. ``motion: lin``, ``planner: pilz``).
"""

from enum import Enum


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


class PlannerId(StrEnum):
    """Per-segment planner override (acts on free/ptp; lin/circ are always Pilz)."""

    PILZ = 'pilz'
    OMPL = 'ompl'
    OMPL_CHOMP = 'ompl_chomp'


class GripperKind(StrEnum):
    """End-effector actuation type."""

    SUCTION = 'suction'    # SurfaceGripper-style, GripperCommand -> on/off services
    PARALLEL = 'parallel'  # jaw gripper (e.g. Robotiq), GripperCommand position
    NONE = 'none'          # no gripper


class AppType(StrEnum):
    """Application templates. pick_and_place is the MVP; the rest are seams.

    Tokens are FROZEN once shipped (they live in every saved project.yaml); only the
    GUI labels may change — 'pick_and_place' is shown as "Blind pick and place" (D-015).
    """

    PICK_AND_PLACE = 'pick_and_place'
    VISION_GUIDED_MOTION = 'vision_guided_motion'   # D-015: editable sequence + vision
    GLUING = 'gluing'
    FOLLOW_PATH = 'follow_path'
    WAYPOINT_REPLAY = 'waypoint_replay'
    CNC = 'cnc'


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


class SceneObjectSource(StrEnum):
    """Where a scene object came from."""

    PRIMITIVE = 'primitive'  # built in the GUI primitive editor
    USD = 'usd'              # imported from a .usd stage


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
