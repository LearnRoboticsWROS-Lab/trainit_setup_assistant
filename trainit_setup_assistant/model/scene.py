"""Scene schema: collision objects + the manipulated payload.

Each object carries a ``category`` chosen when the USD scene is loaded (see
:class:`SceneObjectCategory`): ``static`` and ``actuated`` objects are CHECKED
collision objects (``AddCollisionObject`` nodes / scene-loader meshes the robot
must avoid); ``dynamic`` objects are manipulation targets that are shown but
collision-ALLOWED against everything (they are touched by the gripper and rest on
other meshes). ``category`` is authoritative and drives the ``dynamic`` flag; the
engine plans BOX only today, so non-box shapes are emitted as their AABB and
flagged in the generation manifest.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from .enums import (
    IsaacGraspMethod,
    ReleasePolicy,
    SceneObjectCategory,
    SceneObjectSource,
    ShapeType,
)


class SceneObject(BaseModel):
    id: str
    source: SceneObjectSource = SceneObjectSource.PRIMITIVE
    shape: ShapeType = ShapeType.BOX
    dims: List[float] = Field(default_factory=lambda: [0.1, 0.1, 0.1])  # box: [x,y,z]
    # package:// STL when shape is MESH (loaded by scene_manager_node; not AABB'd).
    mesh_resource: Optional[str] = None
    scale: List[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])  # mesh scale
    # local-frame AABB centre offset (from the prim origin). Used to place the grasped
    # object's bounding BOX (attach_box mode) on its centre, not on its base.
    aabb_center: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    frame: str = 'base_link'
    position: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    orientation: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    # static | actuated | dynamic — the cell role (authoritative; drives `dynamic`).
    category: SceneObjectCategory = SceneObjectCategory.STATIC
    dynamic: bool = False    # True => manipulated => excluded from collision planning
    collision: bool = True   # static & collision => emitted as AddCollisionObject
    # --- per-dynamic-object attributes (meaningful only when category is DYNAMIC) ---
    # grasp_target: the gripper grasps THIS object -> it attaches to the tool link on
    #   close (AttachedCollisionObject in RViz; scene_manager_node attach_object_ids).
    #   A dynamic object that is NOT a grasp target (e.g. the crate) is collision-
    #   allowed but never attached.
    grasp_target: bool = False
    # release_policy: Isaac behaviour when the grip opens (freeze | gravity; never float).
    release_policy: ReleasePolicy = ReleasePolicy.FREEZE
    # isaac_grasp_method: which Isaac physics realises the grasp (metadata; the Script
    #   Node itself is hand-authored in the *_isaac package).
    isaac_grasp_method: IsaacGraspMethod = IsaacGraspMethod.FIXED_JOINT
    # touchable_collision_ids: static/actuated mesh ids this held object MAY contact
    #   while grasped (e.g. the prewash mobile part) -> the ACM leaves those pairs
    #   allowed even when attached_collision_check is ON.
    touchable_collision_ids: List[str] = Field(default_factory=list)

    @model_validator(mode='before')
    @classmethod
    def _reconcile_category(cls, data):
        """Keep ``category`` and the legacy ``dynamic`` flag consistent.

        ``category`` wins when given (``dynamic`` := category is DYNAMIC). Projects
        written before ``category`` existed carry only ``dynamic``/``collision`` -> we
        infer the category from ``dynamic`` without mutating the bools. Idempotent, so
        the YAML round-trip re-validates to an equal object.
        """
        if not isinstance(data, dict):
            return data
        cat = data.get('category')
        if cat is not None:
            data['dynamic'] = SceneObjectCategory(cat) is SceneObjectCategory.DYNAMIC
        elif 'dynamic' in data:
            data['category'] = (SceneObjectCategory.DYNAMIC.value if bool(data['dynamic'])
                                else SceneObjectCategory.STATIC.value)
        return data

    def is_planning_collision(self) -> bool:
        """True if this object should become an AddCollisionObject in the tree."""
        return self.collision and not self.dynamic

    def is_dynamic(self) -> bool:
        """True => manipulation target (collision-allowed against everything)."""
        return self.category is SceneObjectCategory.DYNAMIC

    def is_actuated(self) -> bool:
        """True => URDF station moved by an adapter (checked collision today)."""
        return self.category is SceneObjectCategory.ACTUATED

    def is_grasp_target(self) -> bool:
        """True => dynamic object the gripper grasps (attaches to the tool on close)."""
        return self.is_dynamic() and self.grasp_target

    def is_box(self) -> bool:
        return self.shape is ShapeType.BOX

    def is_mesh(self) -> bool:
        """True => a package:// mesh loaded by the scene loader (not AABB'd in the tree)."""
        return self.shape is ShapeType.MESH

    def aabb_dims(self) -> list:
        """Box [x,y,z] for the engine (box-only). Non-box shapes -> their AABB.

        box: dims as-is; sphere: dims=[r] -> [2r,2r,2r]; cylinder/cone: dims=[r,h]
        -> [2r,2r,h]. Falls back to the raw dims if they don't match the shape.
        """
        d = list(self.dims)
        if self.shape is ShapeType.BOX:
            return d
        if self.shape is ShapeType.SPHERE and len(d) >= 1:
            r = d[0]
            return [2 * r, 2 * r, 2 * r]
        if self.shape in (ShapeType.CYLINDER, ShapeType.CONE) and len(d) >= 2:
            r, h = d[0], d[1]
            return [2 * r, 2 * r, h]
        return d


class Payload(BaseModel):
    """The object the robot grasps (attached to the tool on grasp)."""

    id: str
    dims: List[float] = Field(default_factory=lambda: [0.02, 0.02, 0.02])
    attach_link: str = 'tcp'
    attach_offset: List[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])


class SceneSpec(BaseModel):
    objects: List[SceneObject] = Field(default_factory=list)
    payload: Optional[Payload] = None
    # --- scene-loader (scene_manager_node) runtime params ---
    # The gripper-close signal that triggers attaching grasp_target objects to the tool.
    # Empty (the default) = NO sim grasp adapter wired: nothing attaches on close (real
    # physics, or a cell with no grasp trick). Set it (e.g. auto-detected from a USD
    # ActionGraph, or chosen at Step 7 for a Gazebo LinkAttacher) to enable the attach.
    gripper_cmd_topic: str = ''
    # Cycle-boundary signal: scene_manager_node latches a Bool here when ~/reset_scene is
    # called, so the hand-authored Isaac adapter can teleport its dynamic prims home.
    # TSA owns the CONTRACT; putting the simulated prims back is the cell author's half.
    # Empty by default (Isaac-adapter-specific — not applicable to a plain Gazebo/mock cell).
    scene_reset_topic: str = ''
    attach_link: str = 'tcp'                    # tool link grasp targets attach to
    # --- cross-simulator scene frame (ADR-0011) ---
    # Where the robot's base_link sits in the SIMULATOR WORLD frame, as [x, y, z, R, P, Y]
    # (metres + radians). In Isaac the robot is IN the USD stage, so this is read
    # automatically (binv) and left None. In Gazebo the robot is spawned SEPARATELY from the
    # .world, so the cell author declares it at Step 2: it drives BOTH the importer's full
    # rigid inverse (objects -> base_link, like Isaac) AND spawn_entity's -x/-y/-z/-R/-P/-Y,
    # so RViz and Gazebo agree by construction. None / all-zero = identity (robot at world
    # origin); mock/real keep identity (base_link IS the physical mount origin).
    robot_base_world_pose: Optional[List[float]] = None
    # links near the tool allowed to touch an attached object (self-collision relief).
    touch_links: List[str] = Field(default_factory=lambda: ['tcp'])
    # default planning-collision check for held objects (per-move nodes override it).
    attached_collision_check: bool = False
    # 0 = load once (re-asserting every tick churns the scene, disrupting Plan&Execute).
    force_republish_hz: float = 0.0
    # How a grasped object is represented in the MoveIt scene while held (meshes stay
    # meshes — no per-shape primitive guessing):
    #   remove     : it DISAPPEARS on grasp and REAPPEARS at the tool pose on release
    #                (simplest/robust; the real object is seen in Isaac). No payload
    #                collision-awareness during the transfer.
    #   attach_box : it attaches to the tool as its AABB BOUNDING BOX (one universal
    #                rule, cheap, conservative) so attached_collision_check ON/OFF works
    #                (the planner routes the held payload around obstacles); the mesh
    #                reappears at the tool pose on release.
    grasp_attach_mode: str = 'attach_box'   # default: universal AABB cuboid (payload-aware)

    def grasp_target_ids(self) -> List[str]:
        """Ids of dynamic objects the gripper grasps (attach_object_ids)."""
        return [o.id for o in self.objects if o.is_grasp_target()]

    def release_policy(self):
        """The freeze|gravity policy for released objects (first grasp target; freeze)."""
        from .enums import ReleasePolicy
        for o in self.objects:
            if o.is_grasp_target():
                return o.release_policy
        return ReleasePolicy.FREEZE
