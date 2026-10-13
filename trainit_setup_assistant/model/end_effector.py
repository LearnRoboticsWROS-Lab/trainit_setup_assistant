"""The end-effector CONTRACT of a project (ADR-0022): devices, actuators, grasp sites, the
verification policy and the per-backend adapter bindings. Configuration only -- TSA configures the
contract and refuses against it; the contract itself is normative in ``trainit_interfaces`` and
``trainit_hardware_layer/contracts`` (ADR-0022 D3, D8), and TMR is its runtime (D2, D4).

OPT-IN, per project: ``robot.gripper.contract`` absent (None) is the LEGACY contract, and every
legacy project generates the content it generated before (D9). Present, it selects the canonical
path -- and ONLY its presence does (D6, the guard token ``actuation: contract`` is set by the
gripper's pre-validator, never read to decide).

Two forms, stored AS WRITTEN (D6 ``io.py``):

* the SHORT form -- one ``actuator`` and no ``sites``: a simple suction cup, a GripperCommand
  jaw, a binary jaw. Three typed lines for the simple suction::

      contract:
        actuator: {command_interface: set_bool}
        verification: {require_grasp_confirmation: false, require_release_confirmation: false}
        adapters: {mock: trainit_mock, isaac: trainit_isaac_surface_gripper}

* the LONG form -- an ``actuators`` map and a ``sites`` map, in any mix of interfaces: a
  COMPOSITE tool (a composite BUNDLE is generated for mock only: R22, vNext M2.7; its model,
  refusals, tree and parameter block are proven by tests).

The long form of a short form is DERIVED ON DEMAND (:func:`derive_device`), exactly as
``GripperSpec.gripper_action_ns()`` derives the legacy action: nothing derived is ever stored, so
editing ``eef_parent_link`` later moves the derived TCP with it instead of freezing a stale one into
the file. The derivation table is ADR-0022 D6 "Derived values of the short forms".

``extra = 'forbid'`` everywhere below: an unknown key or token inside the contract is a load-time
refusal (R2), where every legacy TSA model silently drops it. Everything the schema can hold but
the contract refuses (R1, R3-R23) is refused by ``generator.eef_contract.refuse_eef_contract``
before any file is written, from both generation entry points.

Pydantic: the classic v1/v2-overlap API (``class Config``, ``root_validator``), as every TSA model;
the actuator union is discriminated on ``command_interface`` (pydantic >= 1.9).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Annotated, Dict, List, Literal, Mapping, Optional, Sequence, Tuple, Union

from pydantic import BaseModel, Field

from ..catalog import (EEF_ADAPTER_ROWS, EEF_CAPABILITY_TOKENS, EEF_INTERFACES,
                       EEF_PSEUDO_JOINT_TOPIC)
from .enums import Backend, GripperJointTarget, GripperKind, StrEnum

# --- the closed vocabularies (shared token for token with trainit_interfaces, TMR and the
# hardware layer: the token-parity test pins them against the ADR's token file) -------------------


class CommandInterface(StrEnum):
    """The ONE standard command interface of an actuator (ADR-0022 D2)."""

    SET_BOOL = 'set_bool'                                # std_srvs/srv/SetBool
    GRIPPER_COMMAND = 'gripper_command'                  # control_msgs/action/GripperCommand
    FOLLOW_JOINT_TRAJECTORY = 'follow_joint_trajectory'  # control_msgs/action/FollowJointTrajectory


class ResultPolicy(StrEnum):
    """When a GripperCommand goal to the ENGAGE target is DONE (ADR-0022 D2.B). There is no third
    policy: no canonical server needs ABORTED or CANCELED to count as done."""

    SUCCEEDED = 'succeeded'
    SUCCEEDED_OR_STALLED = 'succeeded_or_stalled'


class EvidenceClass(StrEnum):
    """What a grasp observation rests on (``EndEffectorState.EVIDENCE_*`` without NONE). Only the
    first three can be ACCEPTED; the last two exist so a third-party adapter can label such
    evidence, and every consumer refuses them (EEF-D8, R17)."""

    SENSOR = 'sensor'
    SIM_GROUND_TRUTH = 'sim_ground_truth'
    ACTUATION_INFERENCE = 'actuation_inference'           # opt-in only, labelled as inference
    COMMAND_ECHO = 'command_echo'                         # never evidence
    SIM_ATTACHMENT = 'sim_attachment'                     # never evidence


NEVER_EVIDENCE = (EvidenceClass.COMMAND_ECHO, EvidenceClass.SIM_ATTACHMENT)
ACCEPTABLE_EVIDENCE = (EvidenceClass.SENSOR, EvidenceClass.SIM_GROUND_TRUTH,
                       EvidenceClass.ACTUATION_INFERENCE)
DEFAULT_ACCEPTED_EVIDENCE = (EvidenceClass.SENSOR, EvidenceClass.SIM_GROUND_TRUTH)


class EndEffectorCapability(StrEnum):
    """The 10 capability tokens (ADR-0022 D6): a CLOSED vocabulary, no free strings. A new token =
    a row in the adapter table + the TSA catalog + an owner classification; an ADR only when TMR
    must interpret it or an existing token changes meaning."""

    BINARY_ACTUATION = 'binary_actuation'              # implied by set_bool
    ACTUATION_FEEDBACK = 'actuation_feedback'          # required by actuation_feedback: true
    GRIPPER_CONTROL = 'gripper_control'                # implied by gripper_command
    POSITION_CONTROL = 'position_control'              # implied by gripper_command
    EFFORT_CONTROL = 'effort_control'                  # required when max_effort is declared
    JOINT_POSITION_CONTROL = 'joint_position_control'  # implied by follow_joint_trajectory
    TRAJECTORY_CONTROL = 'trajectory_control'          # implied by follow_joint_trajectory
    GRASP_DETECTION = 'grasp_detection'                # per site: grasp confirmation required
    RELEASE_DETECTION = 'release_detection'            # per site: release confirmation required
    CONTINUOUS_STATE = 'continuous_state'              # device: every TrainIt adapter provides it


if tuple(c.value for c in EndEffectorCapability) != EEF_CAPABILITY_TOKENS or tuple(
        i.value for i in CommandInterface) != EEF_INTERFACES:
    raise RuntimeError('the end-effector capability or interface tokens of the catalog '
                       '(catalog.EEF_CAPABILITY_TOKENS / EEF_INTERFACES) differ from the '
                       "model's closed vocabularies: they are one list (ADR-0022 D6)")

IMPLIED_CAPABILITIES = {
    CommandInterface.SET_BOOL: (EndEffectorCapability.BINARY_ACTUATION,),
    CommandInterface.GRIPPER_COMMAND: (EndEffectorCapability.GRIPPER_CONTROL,
                                       EndEffectorCapability.POSITION_CONTROL),
    CommandInterface.FOLLOW_JOINT_TRAJECTORY: (EndEffectorCapability.JOINT_POSITION_CONTROL,
                                               EndEffectorCapability.TRAJECTORY_CONTROL),
}


class SceneSignal(StrEnum):
    """Who publishes the scene signal ``eef/<id>/engaged`` (ADR-0022 D8): chosen by the grip's
    interface, never by the mode; a multi-site device has none."""

    ADAPTER = 'adapter'   # a set_bool grip: the adapter, on every APPLIED command
    TMR = 'tmr'           # a gripper_command grip: TMR, on command DONE
    NONE = 'none'


class ControllerFlavour(StrEnum):
    """A ros2_control GripperActionController's flavour (ADR-0022 D2.B): only ``effort`` provides
    effort_control, and an effort controller never moves with max_effort 0.0."""

    POSITION = 'position'
    EFFORT = 'effort'


# The adapter IMPLEMENTATION rows a backend binds to, from TSA's catalog (ADR-0022 D6; ADR-0021's
# named exception M2.6-1): TSA keeps no second list. Names are the tokens upper-cased.
EndEffectorAdapter = StrEnum('EndEffectorAdapter',
                             [(token.upper(), token) for token in EEF_ADAPTER_ROWS],
                             module=__name__, qualname='EndEffectorAdapter')
EndEffectorAdapter.__doc__ = """An end-effector adapter row (``catalog.EEF_ADAPTER_ROWS``): what
serves the contract on a backend. A row, never a kind (ADR-0022 amends ADR-0019 Decision 2)."""

# Defaults of the derivation table (ADR-0022 D6). One place: the parameter block TSA emits carries
# every resolved value, and normalize_eef.py fills the same defaults.
DEFAULT_DEVICE_ID = 'main'
DEFAULT_SITE_ID = 'main'
DEFAULT_ACTUATOR_ID = {GripperKind.SUCTION: 'vacuum', GripperKind.PARALLEL: 'gripper'}
SET_BOOL_TARGETS = ('engage', 'release')
GRIPPER_ENGAGE, GRIPPER_RELEASE = 'closed', 'open'
DEFAULT_COMMAND_TIMEOUT_MS = 10000
DEFAULT_CONFIRM_TIMEOUT_MS = 2000
DEFAULT_GOAL_TIME_TOLERANCE_S = 0.5
DEFAULT_VERIFY_TIMEOUT_MS = 3000
DEFAULT_MAX_AGE_MS = 500
DEFAULT_CLOCK_TOLERANCE_MS = 100

# A ROS name token, as the parameter block's own parser reads one (normalize_eef.py): an ASCII
# letter first, then ASCII letters, digits or '_'. Stricter than a namespace segment (no leading
# '_'). Anchored with \A and \Z: '$' would also match before a trailing newline ('main\n').
NAME_TOKEN = re.compile(r'\A[A-Za-z][A-Za-z0-9_]*\Z')


class _Strict(BaseModel):
    """Every model of the contract refuses an unknown key (R2)."""

    class Config:
        extra = 'forbid'


class JointPositions(_Strict):
    """A set_bool actuator's joint positions, ENGAGED and RELEASED, one number per joint."""

    engaged: List[float]
    released: List[float]


class SetBoolActuator(_Strict):
    """A binary actuator (D2.A): a valve, a lock, a pneumatic jaw. Endpoint
    ``eef/<eef_id>/actuators/<id>/set_engaged``, derived, never configured. Targets: exactly
    ``engage`` and ``release``."""

    command_interface: Literal['set_bool']
    id: Optional[str] = None                     # the SHORT form's id; a map key otherwise
    joints: Optional[List[str]] = None           # with joint_positions, or neither
    joint_positions: Optional[JointPositions] = None
    actuation_feedback: bool = False             # an end switch confirms the applied state
    confirm_timeout_ms: int = DEFAULT_CONFIRM_TIMEOUT_MS


class GripperCommandActuator(_Strict):
    """A single-joint gripper (D2.B), behind a GripperActionController or the TrainIt mock at the
    same RELATIVE action name. Never an axis."""

    command_interface: Literal['gripper_command']
    id: Optional[str] = None
    action: Optional[str] = None                 # short form: <controller_name>/<action_ns>
    joints: Optional[List[str]] = None           # exactly one; short form: [command_joint]
    targets: Optional[Dict[str, float]] = None   # open and closed required; no 1.0/0.0 fallback
    result_policy: Optional[ResultPolicy] = None  # REQUIRED (R7)
    max_effort: Optional[float] = None           # > 0; declaring it requires effort_control


class FollowJointTrajectoryActuator(_Strict):
    """An axis, or ONE coordinated mechanism on one controller (D2.C/D). Actuation, not a second
    motion method (ADR-0015 Decision 1 as ADR-0022 clarifies it). Never a grip."""

    command_interface: Literal['follow_joint_trajectory']
    id: Optional[str] = None
    action: Optional[str] = None
    joints: Optional[List[str]] = None           # exactly the controller's joints, in order
    targets: Optional[Dict[str, List[float]]] = None
    goal_tolerance: Optional[Union[float, List[float]]] = None   # > 0, carried in every goal
    duration_s: Optional[float] = None
    goal_time_tolerance_s: float = DEFAULT_GOAL_TIME_TOLERANCE_S
    path_tolerance: Optional[Union[float, List[float]]] = None


Actuator = Annotated[Union[SetBoolActuator, GripperCommandActuator,
                           FollowJointTrajectoryActuator],
                     Field(discriminator='command_interface')]


class GraspSiteSpec(_Strict):
    """Where the device holds an object (D0): a fixed TCP link inside the device's subtree, and
    exactly one GRIP actuator with its engage and release targets (derived from the grip's
    interface when not given). A site has no endpoint."""

    tcp: Optional[str] = None
    grip: Optional[str] = None
    engage: Optional[str] = None
    release: Optional[str] = None


class VerificationSpec(_Strict):
    """The verification POLICY (D4). ``require_*`` are REQUIRED keys of a canonical project: one
    key decides, for every application (R17); ``false`` is written, never implied."""

    require_grasp_confirmation: Optional[bool] = None
    require_release_confirmation: Optional[bool] = None
    accepted_evidence: List[EvidenceClass] = Field(
        default_factory=lambda: list(DEFAULT_ACCEPTED_EVIDENCE))
    timeout_ms: int = DEFAULT_VERIFY_TIMEOUT_MS
    max_age_ms: int = DEFAULT_MAX_AGE_MS
    clock_tolerance_ms: int = DEFAULT_CLOCK_TOLERANCE_MS


class AdapterBinding(_Strict):
    """The LONG form of a backend binding: the row, plus the keys that row reads
    (``catalog.EEF_ADAPTER_ROWS[...]['binding_keys']``). The short form is the bare row token."""

    adapter: EndEffectorAdapter
    stage_cmd_topic: Optional[str] = None        # isaac: derived from the Step-2 stage topic
    stage_status_topic: Optional[str] = None     # isaac: the stage publishes its status
    flavour: Optional[ControllerFlavour] = None  # ros2_control: declared on real


Binding = Union[EndEffectorAdapter, AdapterBinding]


class EndEffectorContract(_Strict):
    """ONE end-effector device on this robot's flange (ADR-0022 D0, D6), as the user wrote it.

    The short form sets ``actuator`` and leaves ``actuators``/``sites`` unset; the long form sets
    both maps. Never derive into these fields: :func:`derive_device` computes the long form on
    demand."""

    id: str = DEFAULT_DEVICE_ID
    actuator: Optional[Actuator] = None
    actuators: Optional[Dict[str, Actuator]] = None
    sites: Optional[Dict[str, GraspSiteSpec]] = None
    verification: Optional[VerificationSpec] = None
    adapters: Dict[Backend, Binding] = Field(default_factory=dict)
    command_timeout_ms: int = DEFAULT_COMMAND_TIMEOUT_MS

    def is_short_form(self) -> bool:
        return self.actuator is not None and self.actuators is None and self.sites is None


# --- the long form, derived on demand (ADR-0022 D6 "Derived values of the short forms") ----------

@dataclass
class DeviceActuator:
    """One actuator of the derived long form. ``targets`` keeps the written order; ``derived``
    names the keys the derivation filled in, ``unresolved`` the ones it could not (with why)."""

    id: str
    interface: CommandInterface
    spec: object
    joints: Optional[List[str]] = None
    action: Optional[str] = None
    targets: Optional[Dict[str, object]] = None
    result_policy: Optional[ResultPolicy] = None
    max_effort: Optional[float] = None
    actuation_feedback: bool = False
    confirm_timeout_ms: int = DEFAULT_CONFIRM_TIMEOUT_MS
    joint_positions: Optional[Tuple[List[float], List[float]]] = None
    goal_tolerance: Optional[object] = None
    duration_s: Optional[float] = None
    goal_time_tolerance_s: float = DEFAULT_GOAL_TIME_TOLERANCE_S
    path_tolerance: Optional[object] = None
    derived: Tuple[str, ...] = ()
    unresolved: Dict[str, str] = field(default_factory=dict)

    def target_names(self) -> List[str]:
        if self.interface is CommandInterface.SET_BOOL:
            return list(SET_BOOL_TARGETS)
        return list(self.targets or {})


@dataclass
class DeviceSite:
    id: str
    tcp: Optional[str]
    grip: Optional[str]
    engage: Optional[str]
    release: Optional[str]


@dataclass
class DeviceBinding:
    """One backend's binding, resolved: the row token plus the keys the row reads."""

    backend: Backend
    adapter: str
    stage_cmd_topic: Optional[str] = None
    stage_status_topic: Optional[str] = None
    flavour: Optional[str] = None                # as DECLARED by the binding (its long form)
    long_form: bool = False
    # The flavour the controller RUNS with (ADR-0022 D2.B): the declared one, else -- on gazebo --
    # the base config's controller type, which only the generator can read
    # (``generator.eef_contract.device`` resolves it once). provides() and every refusal read it.
    effective_flavour: Optional[str] = None

    @property
    def row(self) -> dict:
        return EEF_ADAPTER_ROWS[self.adapter]

    def provides(self) -> Tuple[str, ...]:
        """The capability tokens this binding provides on its backend (catalog data)."""
        row = self.row
        out = list(row['provides'])
        if self.stage_status_topic:
            out += list(row['provides_if'].get('stage_status_topic', ()))
        if self.effective_flavour == ControllerFlavour.EFFORT.value:
            out += list(row['provides_if'].get('effort_flavour', ()))
        return tuple(dict.fromkeys(out))

    def evidence(self) -> Tuple[str, ...]:
        return tuple(self.row['evidence'])


@dataclass
class Device:
    """The derived long form of a contract: what the trees, the parameter block, the adapters and
    the refusals read. Every list is in contract order."""

    id: str
    kind: GripperKind
    short_form: bool
    actuators: List[DeviceActuator]
    sites: List[DeviceSite]
    require_grasp_confirmation: Optional[bool]
    require_release_confirmation: Optional[bool]
    accepted_evidence: List[str]
    timeout_ms: int
    max_age_ms: int
    clock_tolerance_ms: int
    command_timeout_ms: int
    bindings: Dict[Backend, DeviceBinding]
    contract: EndEffectorContract

    def actuator(self, actuator_id: Optional[str]) -> Optional[DeviceActuator]:
        return next((a for a in self.actuators if a.id == actuator_id), None)

    def site(self, site_id: Optional[str]) -> Optional[DeviceSite]:
        return next((s for s in self.sites if s.id == site_id), None)

    def default_site(self) -> Optional[DeviceSite]:
        """The site a grasp, a release or a policy step addresses without naming one: the ONLY
        site of a single-site device; None on a multi-site device (a site is then required)."""
        return self.sites[0] if len(self.sites) == 1 else None

    def is_multi_site(self) -> bool:
        return len(self.sites) > 1

    def grip_of(self, site: Optional[DeviceSite]) -> Optional[DeviceActuator]:
        return self.actuator(site.grip) if site is not None else None

    def scene_signal(self) -> SceneSignal:
        """The scene signal's source, by the grip's interface (D8). Multi-site: none."""
        site = self.default_site()
        grip = self.grip_of(site)
        if grip is None:
            return SceneSignal.NONE
        if grip.interface is CommandInterface.SET_BOOL:
            return SceneSignal.ADAPTER
        if grip.interface is CommandInterface.GRIPPER_COMMAND:
            return SceneSignal.TMR
        return SceneSignal.NONE

    def provides_everywhere(self, token: str, backends: Sequence[Backend]) -> bool:
        """True when every listed backend's binding provides ``token`` (a backend without a
        binding provides nothing)."""
        return all(b in self.bindings and token in self.bindings[b].provides() for b in backends)


def _named_value(state: Optional[str], joint: Optional[str], named_states,
                 srdf_states: Optional[Mapping[str, Mapping[str, float]]]):
    """The value of ``joint`` in the SRDF state ``state``: the project's named states first (TSA's
    own copy of the SRDF states), then the base config's SRDF; None when neither has it."""
    if not state or not joint:
        return None
    for ns in named_states or ():
        if ns.name == state and joint in ns.joint_values:
            return float(ns.joint_values[joint])
    values = (srdf_states or {}).get(state) or {}
    if joint in values:
        return float(values[joint])
    return None


def _derived_jaw_targets(gripper, joint, named_states, srdf_states):
    """The jaw's {open, closed} from the legacy fields (D6): the SRDF states or the angles, with
    NO 1.0/0.0 fallback -- (targets, why) with targets None when either is unresolvable."""
    if gripper.joint_target is GripperJointTarget.ANGLE:
        if gripper.open_angle is None or gripper.closed_angle is None:
            return None, ('no targets, and joint_target "angle" has open_angle/closed_angle '
                          f'{gripper.open_angle!r}/{gripper.closed_angle!r}')
        return {GRIPPER_RELEASE: float(gripper.open_angle),
                GRIPPER_ENGAGE: float(gripper.closed_angle)}, ''
    opened = _named_value(gripper.open_state, joint, named_states, srdf_states)
    closed = _named_value(gripper.closed_state, joint, named_states, srdf_states)
    if opened is None or closed is None:
        return None, ('no targets, and the SRDF states open_state/closed_state '
                      f'{gripper.open_state!r}/{gripper.closed_state!r} give no value for joint '
                      f'{joint!r} (TSA never falls back to 1.0/0.0 for a contract)')
    return {GRIPPER_RELEASE: opened, GRIPPER_ENGAGE: closed}, ''


def _device_actuator(aid: str, spec, gripper, short: bool, named_states,
                     srdf_states) -> DeviceActuator:
    iface = CommandInterface(spec.command_interface)
    act = DeviceActuator(id=aid, interface=iface, spec=spec)
    derived: List[str] = []
    if iface is CommandInterface.SET_BOOL:
        act.joints = list(spec.joints) if spec.joints is not None else None
        if spec.joint_positions is not None:
            act.joint_positions = (list(spec.joint_positions.engaged),
                                   list(spec.joint_positions.released))
        act.actuation_feedback = bool(spec.actuation_feedback)
        act.confirm_timeout_ms = spec.confirm_timeout_ms
        return act
    act.action = spec.action
    act.joints = list(spec.joints) if spec.joints is not None else None
    act.targets = dict(spec.targets) if spec.targets is not None else None
    if iface is CommandInterface.GRIPPER_COMMAND:
        act.result_policy = spec.result_policy
        act.max_effort = spec.max_effort
        if short:
            if act.action is None:
                if gripper.controller_name:
                    act.action = f'{gripper.controller_name}/{gripper.action_ns}'
                    derived.append('action')
                else:
                    act.unresolved['action'] = ('no action, and the gripper has no '
                                                'controller_name to derive '
                                                '<controller_name>/<action_ns> from')
            if act.joints is None:
                if gripper.command_joint:
                    act.joints = [gripper.command_joint]
                    derived.append('joints')
                else:
                    act.unresolved['joints'] = 'no joints, and the gripper has no command_joint'
            if act.targets is None:
                joint = act.joints[0] if act.joints else None
                targets, why = _derived_jaw_targets(gripper, joint, named_states, srdf_states)
                if targets is None:
                    act.unresolved['targets'] = why
                else:
                    act.targets = targets
                    derived.append('targets')
    else:
        act.goal_tolerance = spec.goal_tolerance
        act.duration_s = spec.duration_s
        act.goal_time_tolerance_s = spec.goal_time_tolerance_s
        act.path_tolerance = spec.path_tolerance
    act.derived = tuple(derived)
    return act


def _site_targets(grip: Optional[DeviceActuator]):
    if grip is None:
        return None, None
    if grip.interface is CommandInterface.SET_BOOL:
        return SET_BOOL_TARGETS
    if grip.interface is CommandInterface.GRIPPER_COMMAND:
        return GRIPPER_ENGAGE, GRIPPER_RELEASE
    return None, None


def _binding(backend: Backend, value, stage_topic: Optional[str]) -> DeviceBinding:
    if isinstance(value, AdapterBinding):
        flavour = value.flavour.value if value.flavour else None
        out = DeviceBinding(backend=backend, adapter=value.adapter.value,
                            stage_cmd_topic=value.stage_cmd_topic,
                            stage_status_topic=value.stage_status_topic,
                            flavour=flavour, long_form=True, effective_flavour=flavour)
    else:
        out = DeviceBinding(backend=backend, adapter=EndEffectorAdapter(value).value)
    if out.stage_cmd_topic is None and 'stage_cmd_topic' in out.row['requires']:
        # D6: the isaac binding's stage_cmd_topic is the stage topic detected at Step 2
        out.stage_cmd_topic = stage_topic or None
    return out


def derive_device(gripper, named_states=(), srdf_states=None,
                  stage_topic: Optional[str] = None) -> Optional[Device]:
    """The contract's LONG form (ADR-0022 D6), computed from what the user wrote and the legacy
    GripperSpec fields it derives from. None for a LEGACY gripper (no contract).

    ``named_states``: the project's named states; ``srdf_states``: ``{state: {joint: value}}`` of
    the base config's SRDF (the generator passes it; the trees never need it); ``stage_topic``:
    the stage topic detected at Step 2 (``scene.gripper_cmd_topic``). Pure: no file is read.
    """
    contract = gripper.contract
    if contract is None:
        return None
    short = contract.actuator is not None
    actuators: List[DeviceActuator] = []
    sites: List[DeviceSite] = []
    if short:
        spec = contract.actuator
        aid = spec.id or DEFAULT_ACTUATOR_ID.get(gripper.kind, 'gripper')
        actuators.append(_device_actuator(aid, spec, gripper, True, named_states, srdf_states))
    for aid, spec in (contract.actuators or {}).items():
        actuators.append(_device_actuator(aid, spec, gripper, False, named_states, srdf_states))
    by_id = {a.id: a for a in actuators}
    if short and contract.sites is None:
        grip = actuators[0]
        engage, release = _site_targets(grip)
        sites.append(DeviceSite(id=DEFAULT_SITE_ID, tcp=gripper.eef_parent_link, grip=grip.id,
                                engage=engage, release=release))
    for sid, site in (contract.sites or {}).items():
        engage, release = _site_targets(by_id.get(site.grip))
        sites.append(DeviceSite(id=sid, tcp=site.tcp, grip=site.grip,
                                engage=site.engage if site.engage is not None else engage,
                                release=site.release if site.release is not None else release))
    v = contract.verification or VerificationSpec()
    return Device(
        id=contract.id, kind=gripper.kind, short_form=short, actuators=actuators, sites=sites,
        require_grasp_confirmation=v.require_grasp_confirmation,
        require_release_confirmation=v.require_release_confirmation,
        accepted_evidence=[e.value for e in v.accepted_evidence], timeout_ms=v.timeout_ms,
        max_age_ms=v.max_age_ms, clock_tolerance_ms=v.clock_tolerance_ms,
        command_timeout_ms=contract.command_timeout_ms,
        bindings={Backend(b): _binding(Backend(b), value, stage_topic)
                  for b, value in contract.adapters.items()},
        contract=contract)


def pseudo_joint_topic_default(cell_namespace: str) -> str:
    """Where the adapter publishes the suction pseudo-joint echo when no joint-state merger names
    a topic: the legacy name, absolute at the root and relative under a namespace (D7)."""
    return EEF_PSEUDO_JOINT_TOPIC if cell_namespace else '/' + EEF_PSEUDO_JOINT_TOPIC
