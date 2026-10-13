"""Robot-side schema: geometry source, planning group, gripper, MoveIt config.

Every field maps to a concrete file the generator emits into ``*_moveit_config`` (or
copies into ``*_description``). Defaults reproduce the FR3WML golden where reasonable
so a minimal project still yields a sane bundle; nothing is hardcoded to ``fr3wml``.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from pydantic import BaseModel, Field, root_validator, validator

from .end_effector import EndEffectorContract
from .enums import (
    Backend,
    DEFAULT_SIM_GRASP_ADAPTER,
    EndEffectorActuation,
    GripperJointTarget,
    GripperKind,
    SimGraspAdapter,
)

# One ROS name token, per the ROS 2 topic/namespace grammar: a letter or '_' first,
# then letters, digits or '_'. A namespace is one or more of those, '/'-separated.
_NS_TOKEN = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*$')


def normalize_cell_namespace(value: str) -> str:
    """Canonical form of a cell namespace: ``''`` or ``/seg[/seg…]``.

    ``'r3'``, ``'/r3'``, ``'r3/'`` and ``' /r3/ '`` all become ``'/r3'``; ``''``, ``'/'``
    and whitespace become ``''`` (the root — one robot, which is the default and the
    migration path, ADR-0016 Decision 3). An illegal segment raises, because a bad
    namespace produces a bring-up whose nodes silently never find each other.
    """
    text = (value or '').strip().strip('/')
    if not text:
        return ''
    segments = text.split('/')
    for seg in segments:
        if not _NS_TOKEN.match(seg):
            raise ValueError(
                f'cell_namespace segment "{seg}" is not a legal ROS name '
                '(a letter or "_" first, then letters, digits or "_"). '
                'Leave the field EMPTY for a single-robot application.')
    return '/' + '/'.join(segments)


def _default_planner_configs() -> List[str]:
    return [
        'RRTConnectkConfigDefault',
        'RRTkConfigDefault',
        'RRTstarkConfigDefault',
        'PRMkConfigDefault',
        'PRMstarkConfigDefault',
        'ESTkConfigDefault',
        'KPIECEkConfigDefault',
        'BKPIECEkConfigDefault',
        'LBKPIECEkConfigDefault',
    ]


class DescriptionSource(BaseModel):
    """User's robot geometry, ingested (COPIED) verbatim into ``*_description``.

    The assistant never synthesizes geometry: it copies ``urdf_dir`` and
    ``meshes_dir`` and records the entry xacro so the moveit_config + collisions
    pipeline can compile the robot.
    """

    urdf_dir: str                       # abs path to the dir holding the xacro files
    top_xacro: str                      # entry xacro filename within urdf_dir
    meshes_dir: Optional[str] = None    # abs path to the meshes dir (optional)


class PlanningGroupSpec(BaseModel):
    """The arm planning group (SRDF ``<group><chain/></group>``)."""

    name: str
    base_link: str = 'base_link'
    tip_link: str = 'tcp'
    joints: List[str] = Field(default_factory=list)  # ordered j1..jN


class NamedState(BaseModel):
    """A named joint configuration (SRDF ``group_state``)."""

    name: str
    group: str                                  # owning group name
    joint_values: Dict[str, float] = Field(default_factory=dict)


class DisableCollisionSpec(BaseModel):
    """One SRDF ``disable_collisions`` row.

    The self-collision matrix is robot-specific project DATA (like joint positions):
    it is computed once by the MoveIt Setup Assistant's ``collisions_updater`` (see
    :mod:`trainit_setup_assistant.robotmodel.collision_matrix`) and stored here, so
    generation is deterministic and never depends on a live planning process.
    """

    link1: str
    link2: str
    reason: str = 'Never'


class KinematicsSpec(BaseModel):
    """kinematics.yaml for the planning group."""

    solver: str = 'kdl_kinematics_plugin/KDLKinematicsPlugin'
    search_resolution: float = 0.005
    timeout: float = 0.005


class JointLimit(BaseModel):
    has_velocity_limits: bool = True
    max_velocity: float = 0.0
    has_acceleration_limits: bool = True
    max_acceleration: float = 0.0


class JointLimitsSpec(BaseModel):
    """joint_limits.yaml (default scales + per-joint overrides)."""

    default_velocity_scaling_factor: float = 0.1
    default_acceleration_scaling_factor: float = 0.1
    per_joint: Dict[str, JointLimit] = Field(default_factory=dict)


class OmplGroupSpec(BaseModel):
    """Robot-specific OMPL overlay (the generic pipeline comes from the engine)."""

    default_planner_config: str = 'RRTConnectkConfigDefault'
    planner_configs: List[str] = Field(default_factory=_default_planner_configs)
    projection_evaluator: str = 'joints(j1,j2)'
    longest_valid_segment_fraction: float = 0.005


class PilzCartesianLimits(BaseModel):
    """pilz_cartesian_limits.yaml."""

    max_trans_vel: float = 1.0
    max_trans_acc: float = 2.25
    max_trans_dec: float = -5.0
    max_rot_vel: float = 1.57


class LinkAttacherConfig(BaseModel):
    """The four names the Gazebo IFRA LinkAttacher plugin welds together (ADR-0012):
    ``link2`` of ``model2`` (the grasped object) is fixed to ``link1`` of ``model1`` (the
    robot). Authored at Step 7 for a Fake-weld adapter; an empty field falls back to the
    generator's derivation (``robot_model`` = the spawned ``-entity`` name, ``object_model``
    = the grasp-target id, ``object_link`` = the ``.world``'s first link for that model)."""

    robot_model: str = ''    # model1_name: the Gazebo model the robot is spawned as (== -entity)
    robot_link: str = ''     # link1_name: the robot link the object welds to (e.g. wrist_3_link)
    object_model: str = ''   # model2_name: the Gazebo model name of the grasped object
    object_link: str = ''    # link2_name: the object's link (e.g. link_1)


class GripperSpec(BaseModel):
    """End-effector: SRDF EEF group + controller + grasp/release semantics."""

    kind: GripperKind = GripperKind.NONE
    eef_group_name: Optional[str] = None        # SRDF group, e.g. "SuctionCup"
    eef_name: Optional[str] = None              # SRDF end_effector name, e.g. "suction"
    eef_parent_link: Optional[str] = None       # e.g. "tcp"
    command_joint: Optional[str] = None         # e.g. "suctioncup_joint"
    # MoveIt controller (moveit_controllers.yaml / ros2_controllers.yaml)
    controller_name: Optional[str] = None       # e.g. "suctioncup_controller"
    controller_type: str = 'GripperCommand'
    action_ns: str = 'gripper_command'
    # how grasp/release map onto engine BT nodes:
    #   suction -> CloseGripper / OpenGripper (the engine's gripper action)
    grasp_action: str = 'close'                 # close | suction_on
    release_action: str = 'open'                # open | suction_off
    # SRDF group_states for the EEF (e.g. "on"/"off")
    open_state: Optional[str] = None
    closed_state: Optional[str] = None

    # --- end-effector abstraction (ADR-0010): actuation x sim grasp adapter ---
    # HOW the BT commands the EEF on grasp/release. Defaults from `kind` when unset
    # (parallel jaw -> joint_position; suction / none / other -> trigger); ``contract`` whenever
    # a contract is present (ADR-0022, set by the pre-validator below, never chosen).
    actuation: EndEffectorActuation = EndEffectorActuation.TRIGGER
    # --- the end-effector CONTRACT (ADR-0022), opt-in -----------------------------------------
    # None (the default) is the LEGACY contract: every field around it keeps its ADR-0010 meaning
    # and the project generates the content it generated before. Present, it alone selects the
    # canonical path; what each legacy field above then means is ADR-0022 D6 "Legacy GripperSpec
    # fields in a canonical project". Saved as written, omitted when None (model/io.py).
    contract: Optional[EndEffectorContract] = None
    # The explicit legacy ``actuation`` token a contract overrode at load, so validate() can name
    # it (ADR-0022 D6). Set ONLY by the pre-validator below, never saved (exclude).
    actuation_overridden: Optional[str] = Field(default=None, exclude=True)
    # For JOINT_POSITION: how the open/closed joint targets are defined.
    joint_target: GripperJointTarget = GripperJointTarget.SRDF_STATE
    # For joint_target == ANGLE: explicit joint angles (rad), captured live from MoveIt or typed.
    open_angle: Optional[float] = None
    closed_angle: Optional[float] = None
    # WHICH sim-physics trick attaches a grasped object, PER BACKEND (keys = backend tokens).
    # Empty -> the sensible DEFAULT_SIM_GRASP_ADAPTER (isaac->surface_gripper,
    # gazebo->link_attacher, real/mock->none). The user overrides per backend at Step 7.
    sim_grasp_adapter: Dict[str, SimGraspAdapter] = Field(default_factory=dict)
    # --- ADR-0012: end-effector <-> dynamic-object interaction ---
    # Does this end-effector grasp a dynamic object AT ALL? False -> a trigger-only
    # application (welding, inspection, dispensing): no grasp target, no grasp Bool, no
    # planning-scene attach, no per-backend adapter. Default True preserves today's
    # pick&place behaviour (the golden is unaffected: the field is inert unless the gazebo
    # branch or the grasp wiring reads it).
    interacts_with_object: bool = True
    # For a Fake-weld (LINK_ATTACHER) adapter, the four IFRA plugin names. None -> the
    # generator derives them (byte-safe fallback), so existing bundles are unchanged.
    link_attacher: Optional[LinkAttacherConfig] = None

    # --- the TOOL's own geometry (ADR-0017): two numbers that had no home ------------
    # Both are properties of the TOOL, which is why they live here and not on a detector
    # or in a node's Python defaults: the same cup ranks candidates for every detector
    # that watches its bin, and a number that exists only as a function default cannot be
    # tuned in TSA or REPRODUCED FROM A BUNDLE (ADR-0017 Consequences). The generator
    # copies them into each detector's params, where perception computes the clearance
    # terms from them (``obstruction_mm`` / ``clearance``, ADR-0014).
    #
    # tool_radius_m: the radius of the tool's FOOTPRINT around the grasp point — the disc
    #   the approach has to find empty. Perception measures the depth inside exactly that
    #   disc, so with no radius there is no footprint and no term: a `candidate_x_clearance`
    #   rule is REFUSED without it (ADR-0017 Decision 3), never silently demoted.
    # 0.0 = not declared. The clearance terms are then simply not published, and every
    #   other rule is unaffected — the reference cell's measurement says clearance buys
    #   nothing at the TOP of its ranking, so this is not a hole in the default path.
    tool_radius_m: float = Field(default=0.0, ge=0.0)
    # tool_reach_mm: THE GRIP WINDOW — how far the tool can still bridge to the face and
    #   hold it; for a suction cup, its own grip distance (the reference cup: 8 mm). An
    #   obstruction of that much or more and the tool cannot come within grasping distance
    #   however well the pose was fitted, which is precisely what turns an obstruction in
    #   millimetres into a 0..1 clearance. A ZERO window makes that term identically 1.0 on
    #   every candidate — a clearance ranking that silently collapses into `candidate_2d3d`
    #   — so it is required alongside the radius, not optional. Named for the detector
    #   parameter it becomes (`tool_reach_mm`), so one number has one name all the way from
    #   this field to the node that reads it.
    tool_reach_mm: float = Field(default=0.0, ge=0.0)

    # @root_validator(pre=True): v1/v2-overlap (mode='before' == pre=True), runs on both
    # pydantic 1.9 and 2.x; receives the raw pre-validation dict, so the default below is
    # byte-identical to the model_validator form. It is the ONLY pre-validator of this model, on
    # purpose: two of them run in definition order on pydantic 1.x and in reverse order on 2.x,
    # and the token rules below must not depend on which one runs first (ADR-0022 D6).
    @root_validator(pre=True)
    def _default_actuation_from_kind(cls, data):
        """The ``actuation`` token, decided in one place.

        With a CONTRACT (ADR-0022 D6): ``contract``, always -- it overrides an explicit
        ``trigger``/``joint_position`` (TSA has written an explicit ``actuation`` since ADR-0010,
        so adding ``contract:`` to such a file is the normal opt-in), and the overridden token is
        kept for validate() to name. Without one, ADR-0010's rule, unchanged: a parallel jaw
        defaults to JOINT_POSITION, everything else keeps the TRIGGER default, and only when
        ``actuation`` was not given, so the YAML round-trip is idempotent."""
        if not isinstance(data, dict):
            return data
        data = dict(data)
        data.pop('actuation_overridden', None)       # only this validator sets it
        if data.get('contract') is not None:
            given = data.get('actuation')
            token = getattr(given, 'value', given)
            if given is not None and token != EndEffectorActuation.CONTRACT.value:
                data['actuation_overridden'] = str(token)
            data['actuation'] = EndEffectorActuation.CONTRACT.value
            return data
        if 'actuation' not in data and data.get('kind') is not None:
            if GripperKind(data['kind']) is GripperKind.PARALLEL:
                data['actuation'] = EndEffectorActuation.JOINT_POSITION.value
        return data

    # skip_on_failure: required by pydantic 2.x for a post root_validator, accepted by 1.x.
    @root_validator(skip_on_failure=True)
    def _contract_token_needs_a_contract(cls, values):
        """R19 (ADR-0022 D6): the guard token without the block it guards -- the file lost its
        contract, and generating it would silently be LEGACY."""
        if (values.get('actuation') is EndEffectorActuation.CONTRACT
                and values.get('contract') is None):
            raise ValueError('end-effector contract (ADR-0022 R19): actuation "contract" without '
                             'a contract block -- the file was saved with an end-effector '
                             'contract that is no longer in it. Restore the contract, or set '
                             'actuation to trigger/joint_position for a legacy gripper.')
        return values

    def is_canonical(self) -> bool:
        """True iff this gripper opts into the end-effector contract (ADR-0022): ``contract`` is
        present. The ONLY test that selects the canonical path -- never the token."""
        return self.contract is not None

    def gripper_action_ns(self) -> str:
        """Full action namespace used in bt_params (``/<ctrl>/<action_ns>``)."""
        if not self.controller_name:
            return ''
        return f'/{self.controller_name}/{self.action_ns}'

    def grasp_adapter_for(self, backend) -> SimGraspAdapter:
        """The sim grasp adapter for a backend: the explicit per-backend choice if set, else
        the sensible default (isaac->surface_gripper, gazebo->link_attacher, real/mock->none)."""
        b = Backend(backend)
        got = self.sim_grasp_adapter.get(str(b))
        return got if got is not None else DEFAULT_SIM_GRASP_ADAPTER.get(b, SimGraspAdapter.NONE)

    def joint_targets(self):
        """(open, closed) targets for JOINT_POSITION actuation: SRDF state names, or explicit
        angles when ``joint_target`` is ANGLE."""
        if self.joint_target is GripperJointTarget.ANGLE:
            return self.open_angle, self.closed_angle
        return self.open_state, self.closed_state

    def link_attacher_names(self, *, robot_model, robot_link, object_model, object_link):
        """The four LinkAttacher names to emit: the Step-7 override for each field when set,
        else the passed-in generator derivation. Keeps a bundle byte-identical when the user
        never touched the fields (link_attacher is None -> every override is empty)."""
        cfg = self.link_attacher
        return {
            'robot_model': (cfg.robot_model if cfg and cfg.robot_model else robot_model),
            'robot_link': (cfg.robot_link if cfg and cfg.robot_link else robot_link),
            'object_model': (cfg.object_model if cfg and cfg.object_model else object_model),
            'object_link': (cfg.object_link if cfg and cfg.object_link else object_link),
        }


class ArmControllerSpec(BaseModel):
    """Arm trajectory controller (the launch ``ARM_CONTROLLER`` constant)."""

    name: str = 'moveit_joint_controller'
    type: str = 'joint_trajectory_controller/JointTrajectoryController'
    action_ns: str = 'follow_joint_trajectory'
    update_rate: int = 100
    command_interfaces: List[str] = Field(default_factory=lambda: ['position'])
    state_interfaces: List[str] = Field(default_factory=lambda: ['position', 'velocity'])


class RobotSpec(BaseModel):
    """Everything needed to emit ``*_description`` + ``*_moveit_config``."""

    # validate_assignment: `cell_namespace` is CANONICALISED on assignment too, not only
    # on parse, so `robot.cell_namespace = 'r3'` and a YAML round-trip agree — a field
    # whose stored value differs from its canonical value is a defect waiting for the
    # first `==`. `class Config` (not ConfigDict): honoured by pydantic 1.9 and 2.x alike.
    class Config:
        validate_assignment = True

    robot_name: str
    description: DescriptionSource
    # THE ROBOT SLOT (ADR-0016 Decision 3). Empty — the default and the migration path —
    # puts every generated node at the ROOT, which is byte-for-byte what every existing
    # bundle generates today. Non-empty namespaces this robot's whole motion stack:
    # move_group, the controllers, robot_state_publisher, the scene manager, the BT
    # application and the topic names servo derives from them.
    #
    # What it deliberately does NOT namespace: /tf and /tf_static (tf2 declares them
    # absolute, and a split TF tree is a cell that cannot see itself), the cameras (a
    # camera serves a STATION, not a robot) and /perception/<detector>/… (the detector
    # that watches the bin is one detector however many arms reach into it).
    #
    # An application with ONE robot leaves this empty and never meets it.
    cell_namespace: str = ''
    base_frame: str = 'base_link'
    tip_link: str = 'tcp'
    planning_group: PlanningGroupSpec
    gripper: GripperSpec = Field(default_factory=GripperSpec)
    named_states: List[NamedState] = Field(default_factory=list)
    kinematics: KinematicsSpec = Field(default_factory=KinematicsSpec)
    joint_limits: JointLimitsSpec = Field(default_factory=JointLimitsSpec)
    ompl: OmplGroupSpec = Field(default_factory=OmplGroupSpec)
    pilz_cartesian_limits: PilzCartesianLimits = Field(default_factory=PilzCartesianLimits)
    initial_positions: Dict[str, float] = Field(default_factory=dict)
    arm_controller: ArmControllerSpec = Field(default_factory=ArmControllerSpec)
    # Self-collision matrix (project data). When set, the generated SRDF uses it
    # verbatim and generation never spawns collisions_updater. None => emit an
    # over-conservative SRDF (build still works) and warn; populate it via the live
    # session / `trainit_compute_collisions`.
    disable_collisions: Optional[List[DisableCollisionSpec]] = None
    # Alternative fallback: a ready-made SRDF supplied by the user (skips SRDF gen).
    source_srdf: Optional[str] = None
    # The hand-made BASE moveit_config (adaptation-sprint deliverable) that already
    # provides the mock/isaac/real bringup + bridge wiring. When set, the standalone
    # scene+planner emitter COPIES this package's config/ verbatim (preserving the
    # tuned mode-switch + SRDF waypoints) instead of templating from the model, so the
    # generated intermediate/bundle config inherits real/isaac/mock for free.
    base_moveit_config_package: Optional[str] = None   # ROS pkg name, e.g. fr30_eef_moveit_config
    base_moveit_config_path: Optional[str] = None      # abs path to that package (src or share)

    # Classic @validator (not @field_validator): the deprecated-but-functional API that
    # runs on BOTH pydantic 1.9 (apt python3-pydantic) and 2.x, so TrainIt installs via
    # rosdep with no pip.
    @validator('cell_namespace')
    def _normalize_cell_namespace(cls, v):
        return normalize_cell_namespace(v)

    def namespace(self) -> str:
        """The canonical cell namespace (``''`` or ``/seg[/seg…]``) — the ONE datum every
        namespaced artefact derives from (ADR-0016 Decision 3)."""
        return normalize_cell_namespace(self.cell_namespace)

    def under_namespace(self, name: str) -> str:
        """An ABSOLUTE name under the cell namespace.

        ``'joint_states'`` -> ``/joint_states`` at the root, ``/r3/joint_states`` under
        ``/r3``. A name that already carries the namespace is returned unchanged, so
        calling it twice is safe.
        """
        ns = self.namespace()
        leaf = name if name.startswith('/') else f'/{name}'
        if ns and not (leaf == ns or leaf.startswith(f'{ns}/')):
            return f'{ns}{leaf}'
        return leaf
