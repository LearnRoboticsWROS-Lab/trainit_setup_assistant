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

from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, root_validator

from .faces import DEFAULT_FACE, face_params as registered_face_params

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
    # --- the MEASURED grasp face (ADR-0018) -----------------------------------------
    # Where the user said he intends to GRASP this object, as the scalars the ranking
    # consumes. Empty (the default) = nobody measured, and ``face_radius_m()`` falls back
    # to the AABB approximation it always used.
    #   grasp_face        the FACE FAMILY, a name in trainit_perception's face registry
    #                     ('' = the registry's own default, `disc`). TSA never keeps a
    #                     list of the names (ADR-0018 Decision 6): it asks.
    #   grasp_face_params the measured numbers, keyed by THAT face's own parameter names
    #                     — so a face family added to the registry is configurable here
    #                     without TSA learning anything about it.
    #   grasp_face_source where the numbers came from (file, view, unit, exact or
    #                     tessellated). Provenance, because "measured" against
    #                     "approximated" is the whole point of ADR-0018 and a bundle that
    #                     cannot say which is back to where it started.
    grasp_face: str = ''
    grasp_face_params: Dict[str, float] = Field(default_factory=dict)
    grasp_face_source: str = ''

    # Classic @root_validator(pre=True) (not @model_validator): the v1/v2-overlap API that
    # runs on both pydantic 1.9 and 2.x. mode='before' == pre=True: both receive the raw
    # pre-validation dict (defaults not yet applied), so the mutation below is byte-identical.
    @root_validator(pre=True)
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

    def face_name(self) -> str:
        """The grasp face family measured on this object, or the registry's default.

        A name, not an enum: the registry owns the set (ADR-0018 Decision 6), and a face
        family that arrives in ``trainit_perception`` must not need a TSA release.
        """
        return (self.grasp_face or '').strip() or DEFAULT_FACE

    def has_measured_face(self) -> bool:
        """True when somebody MEASURED this object's grasp face instead of letting the
        bounding box approximate it."""
        return bool(self.grasp_face_params)

    def measured_face_params(self) -> Dict[str, float]:
        """The measured numbers, keyed by the configured face's own parameter names.

        Only the keys that face actually reads are returned, in the order the face declares
        them (TSA's snapshot of the registry, ADR-0021). For a face the snapshot does not
        register, everything measured is returned as-is rather than filtered against a
        guess — and generation refuses that face, by name.
        """
        wanted = registered_face_params(self.face_name())
        if not wanted:
            return {k: float(v) for k, v in self.grasp_face_params.items()}
        return {k: float(self.grasp_face_params[k]) for k in wanted
                if k in self.grasp_face_params}

    def face_radius_m(self) -> float:
        """The radius of the DISC the camera sees when this object is picked.

        MEASURED when the face picker measured it (ADR-0018 Decision 1), and only then
        approximated from ``shape`` + ``dims`` — which is why the ranking still needs no
        field of its own for the number itself (ADR-0017 Consequences: "the object's face
        radius needs no field: it is derivable from the grasp target's shape and
        dimensions").

        **The approximation is wrong by a lot and that is why the picker exists.** For the
        real capsule the AABB gives 28,10 mm against a true 20,95 mm — the box is inflated
        by the pull tab — which is +34 % on the radius and +80 % on the AREA, and the area
        is what ``visible_fraction`` divides by. A radius wrong by a CONSTANT factor leaves
        the ORDER untouched (it is a common factor of every candidate's term) and only
        shifts where the term saturates at 1,0; what it does cost is the number itself, and
        the ``radius_ratio`` gate that reads it.

        It feeds perception's ``visible_fraction``, whose denominator is the area a disc
        of this radius covers at the fitted depth — ``pi * (f * r / z)^2``. So what is
        wanted is an EQUIVALENT-AREA radius for the grasp FACE, not a bounding radius:

          sphere          dims=[r]      -> r              (every face is that disc)
          cylinder/cone   dims=[r, h]   -> r              (the circular end face)
          box / mesh      dims=[x,y,z]  -> sqrt(x*y/pi)   (the x-y face, equal AREA, so
                                          the denominator equals the face's real area)

        Returns 0.0 when the dims do not match the shape: 0 means "no face area", which
        is what makes a ``candidate_x_visible`` rule REFUSABLE rather than silently
        demoted to ``candidate_2d3d`` (ADR-0017 Decision 3).
        """
        import math

        measured = self.grasp_face_params.get('expected_radius_m')
        if measured is not None and float(measured) > 0:
            return float(measured)

        d = [float(v) for v in self.dims]
        if self.shape is ShapeType.SPHERE and len(d) >= 1:
            return max(d[0], 0.0)
        if self.shape in (ShapeType.CYLINDER, ShapeType.CONE) and len(d) >= 2:
            return max(d[0], 0.0)
        if self.shape in (ShapeType.BOX, ShapeType.MESH) and len(d) >= 2:
            x, y = max(d[0], 0.0), max(d[1], 0.0)
            return math.sqrt(x * y / math.pi) if x > 0 and y > 0 else 0.0
        return 0.0

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
    # The Gazebo .world that carries the cell scene (ADR-0011). Stored at Step 2 so the
    # generated gazebo bring-up DEFAULTS its `world:=` arg to it — Gazebo then opens WITH the
    # scene (the Gazebo analogue of Isaac starting in Play with the USD loaded), instead of an
    # empty world where the scene shows only in RViz. May be an absolute path or a launch
    # substitution ($(find <pkg>)/worlds/<cell>.world); empty = Gazebo's default empty world.
    world_path: Optional[str] = None
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

    def first_grasp_target(self) -> Optional[SceneObject]:
        """The object the ranking is configured from — the FIRST grasp target, matching
        ``release_policy()``. A cell with two different grasp targets does not have one
        grasp face, and quietly averaging two would be worse than the refusal the caller
        raises on nothing."""
        for o in self.objects:
            if o.is_grasp_target():
                return o
        return None

    def grasp_face(self) -> Tuple[str, Dict[str, float]]:
        """The grasp target's MEASURED face: ``(face name, its parameters)``.

        ``('', {})`` when nothing was measured — the generator then falls back to the
        bounding-box disc, which is exactly what it did before ADR-0018. The face name is
        returned as the registry's own token, so the generator writes it without
        translating it: what a face needs is the registry's answer, read from TSA's committed
        snapshot of it, never a copy kept here (ADR-0018 Decision 6, ADR-0021).
        """
        obj = self.first_grasp_target()
        if obj is None or not obj.has_measured_face():
            return '', {}
        return obj.face_name(), obj.measured_face_params()

    def grasp_face_radius_m(self) -> float:
        """The grasp target's face radius (ADR-0017) — 0.0 when there is no grasp target.

        Measured when the picker measured it, approximated from the AABB otherwise
        (ADR-0018 Decision 1 — the picker replaces the ESTIMATOR of a number the pipeline
        already carries, and changes no contract).
        """
        obj = self.first_grasp_target()
        return obj.face_radius_m() if obj is not None else 0.0

    def release_policy(self):
        """The freeze|gravity policy for released objects (first grasp target; freeze)."""
        from .enums import ReleasePolicy
        for o in self.objects:
            if o.is_grasp_target():
                return o.release_policy
        return ReleasePolicy.FREEZE
