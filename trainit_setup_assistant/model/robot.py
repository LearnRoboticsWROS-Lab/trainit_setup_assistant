"""Robot-side schema: geometry source, planning group, gripper, MoveIt config.

Every field maps to a concrete file the generator emits into ``*_moveit_config`` (or
copies into ``*_description``). Defaults reproduce the FR3WML golden where reasonable
so a minimal project still yields a sane bundle; nothing is hardcoded to ``fr3wml``.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, root_validator

from .enums import (
    Backend,
    DEFAULT_SIM_GRASP_ADAPTER,
    EndEffectorActuation,
    GripperJointTarget,
    GripperKind,
    SimGraspAdapter,
)


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
    # (parallel jaw -> joint_position; suction / none / other -> trigger).
    actuation: EndEffectorActuation = EndEffectorActuation.TRIGGER
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

    # @root_validator(pre=True): v1/v2-overlap (mode='before' == pre=True), runs on both
    # pydantic 1.9 and 2.x; receives the raw pre-validation dict, so the default below is
    # byte-identical to the model_validator form.
    @root_validator(pre=True)
    def _default_actuation_from_kind(cls, data):
        """A parallel jaw defaults to JOINT_POSITION actuation; everything else keeps the
        TRIGGER default. Only when ``actuation`` was not given, so the YAML round-trip is
        idempotent."""
        if isinstance(data, dict) and 'actuation' not in data and data.get('kind') is not None:
            if GripperKind(data['kind']) is GripperKind.PARALLEL:
                data['actuation'] = EndEffectorActuation.JOINT_POSITION.value
        return data

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

    robot_name: str
    description: DescriptionSource
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
