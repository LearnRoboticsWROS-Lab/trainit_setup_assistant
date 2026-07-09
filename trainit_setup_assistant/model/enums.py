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
    """Application templates. pick_and_place is the MVP; the rest are seams."""

    PICK_AND_PLACE = 'pick_and_place'
    GLUING = 'gluing'
    FOLLOW_PATH = 'follow_path'
    WAYPOINT_REPLAY = 'waypoint_replay'
    CNC = 'cnc'


class SceneObjectSource(StrEnum):
    """Where a scene object came from."""

    PRIMITIVE = 'primitive'  # built in the GUI primitive editor
    USD = 'usd'              # imported from a .usd stage


class ShapeType(StrEnum):
    """Primitive shape. The engine plans BOX only today; others are emitted as AABB."""

    BOX = 'box'
    SPHERE = 'sphere'
    CYLINDER = 'cylinder'
    CONE = 'cone'


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
