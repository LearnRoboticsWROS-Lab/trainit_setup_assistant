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
