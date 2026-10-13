"""The end-effector contract in generation (ADR-0022 D6): what TSA REFUSES before a single file is
written, what it WARNS about, and what a canonical project's bundle CARRIES.

A LEGACY project (no ``robot.gripper.contract``) reaches none of it except two refusals that exist
precisely because a legacy file must not hide a contract: the tool-action kind ``actuate`` (R11)
and the guard token / ``schema_version`` 3 without a contract (R19). Everything else here runs only
when a contract is present, and a legacy project generates the content it generated before.

``refuse_eef_contract(project)`` is called by BOTH generation entry points
(``Orchestrator.generate`` and ``generate_scene_loader_config``, the GUI's Step-3 path) before any
file exists; every refusal
names the device, the actuator, the site and the backend involved, and the ADR-0022 row it is
(R1-R26). The same sentences reach the GUI through the application's validate().

What the bundle carries (emission, canonical only), each piece classified as ADR-0022 "Generated
changes, classified" lists it:

* [C] ``bt_params``: ``gripper_action: ""``, no joint-position keys, and the ``eef`` parameter
  block (:func:`eef_parameter_block`) -- every value RESOLVED, maps keyed by id, the order carried
  by ``ids`` / ``actuator_ids`` / ``site_ids``, never an empty sequence. Byte copies of the shared
  conformance fixture pin that it normalises to the expected dump (test data, sha256-pinned).
* [C] ``scene.yaml``: ``gripper_cmd_topic: "eef/<id>/engaged"``; a relative ``scene_reset_topic``
  under a cell namespace, always written with an isaac binding (:func:`scene_reset_topic`).
* [A] ``BRIDGES``: the TrainIt adapter rows appended (:func:`adapter_rows`), with the same block
  nested in their parameters; nothing removed (a legacy end-effector bridge row is REFUSED, R18).
  A composite mock row carries one launch remap, ``eef/<id>/joint_states`` -> ``joint_states``
  (both relative: the tool joints reach the robot's own joint_states, vNext M2.7).
* [D] ``package.xml``: ``trainit_hardware_layer`` ``version_gte`` iff a generated launch starts one
  of its adapters (``runtime_requirements.hardware_layer_requirement``), and the TMR 2.0.0 floor
  (2.1.0 for a canonical bin-picking cycle, whose moves remap MoveWaypoint's fault, vNext M2.7,
  and for a runner in a cell namespace, vNext G0); every package ``scene.yaml`` loads a mesh from,
  in the package that carries it (:func:`scene_package_references`).
"""

from __future__ import annotations

import functools
import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import yaml

from ..catalog import EEF_ADAPTER_ROWS, EEF_BRIDGE_ROLES, HARDWARE_LAYER
from ..model import CanonicalProject
from ..model.deployment import LaunchNodeSpec
from ..model.end_effector import (IMPLIED_CAPABILITIES, NAME_TOKEN,
                                  NEVER_EVIDENCE, SET_BOOL_TARGETS, CommandInterface,
                                  ControllerFlavour, Device, DeviceActuator, DeviceBinding,
                                  DeviceSite, EndEffectorCapability, derive_device,
                                  pseudo_joint_topic_default)
from ..model.enums import (DEFAULT_SIM_GRASP_ADAPTER, AppType, Backend, EndEffectorActuation,
                           GripperJointTarget, GripperKind, PolicyMode, SimGraspAdapter,
                           ToolActionKind)
from ..model.project import SCHEMA_VERSION_EEF_CONTRACT
from ..model.task import BinPickingSpec

_DEFAULT_HOLDING_TIMEOUT_MS = BinPickingSpec().holding_timeout_ms

# A composite MOCK bundle (owner Q1 option A, vNext M2.7): the trainit_mock adapter publishes the
# tool joints on its own eef/<id>/joint_states, and the generated adapter row remaps that ONE name
# onto the robot's joint_states -- both RELATIVE, so they resolve in the robot's namespace (ADR-0022
# D7): /joint_states at the root, /r1/joint_states under /r1.
ROBOT_JOINT_STATES = 'joint_states'
# What the mock bring-up hands xacro (TMR trainit_moveit_config._hardware_mappings(False, None)): a
# composite deploys mock alone, so the bring-up never renders its backend= argument.
_MOCK_XACRO_MAPPINGS = (('use_isaac', 'false'),)
_SINGLE_DOF = ('prismatic', 'revolute', 'continuous')

_GATE_KEY = 'eef_fault'
_HOLDING_KEY = 'holding'
_TRIGGER_DEFAULTS = ('/suction/on', '/suction/off')   # the policy runtime's own defaults
# A ROS topic or service name, ASCII. Anchored with \A and \Z: '$' would also match before a
# trailing newline ('/suction_cmd\n'), as it did for NAME_TOKEN.
_ROS_NAME = re.compile(r'\A/?[A-Za-z_][A-Za-z0-9_]*(/[A-Za-z_][A-Za-z0-9_]*)*\Z')
# scene_manager_node's own default reset topic (trainit_motion_runtime) is ABSOLUTE and latched. A
# canonical project names its reset topic instead (scene_reset_topic), and the isaac adapter
# listens to the same name. The default is written RELATIVE: at the root it resolves to the scene
# manager's own name, under a cell namespace to this robot's (ADR-0022 D7: no absolute canonical
# name in TSA's code).
SCENE_RESET_TOPIC_DEFAULT = 'isaac_scene_reset'
# trainit_hardware_layer's mock_eef_adapter samples every site at this rate unless told otherwise,
# and refuses to start when the rate times verification.max_age_ms is below 1000 (every sample
# must be fresh at the next one): the mock row then carries the rate it needs.
_MOCK_SAMPLE_RATE_HZ = 20.0


def _prefix(row: int) -> str:
    return f'end-effector contract (ADR-0022 R{row})'


# G0 (owner, section 9): a refusal that marks only a LIMIT of the current canonical model -- the
# configuration may be a valid design that this model does not represent -- says so with this
# token, never that the design is invalid; a physical or contract invalidity never carries it
# (ADR-0022 D6 "Refusals").
_UNSUPPORTED = ('UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL (a limit of the current canonical model, '
                'not an invalid design)')


def is_canonical(project: CanonicalProject) -> bool:
    """True iff the project opts into the end-effector contract (ADR-0022): the contract block
    is present. The token is never what decides."""
    return project.robot.gripper.contract is not None


# --- the robot description, as the contract's checks read it ------------------------------------

@dataclass(frozen=True)
class UrdfJoint:
    name: str
    type: str
    parent: str
    child: str
    lower: Optional[float]
    upper: Optional[float]
    velocity: Optional[float]
    mimic: Optional[str]


class UrdfView:
    """The links and joints of a compiled description: enough for R5, R6, R8, R9, R10 and the
    pseudo-joint echo. Built with ElementTree; nothing is resolved but names."""

    def __init__(self, urdf_xml: str):
        root = ET.fromstring(urdf_xml)
        self.links = {link.get('name') for link in root.findall('link')}
        self.joints: Dict[str, UrdfJoint] = {}
        for j in root.findall('joint'):
            limit = j.find('limit')
            mimic = j.find('mimic')

            def _f(attr, _limit=limit):
                try:
                    return float(_limit.get(attr)) if _limit is not None and _limit.get(
                        attr) is not None else None
                except ValueError:
                    return None
            parent, child = j.find('parent'), j.find('child')
            self.joints[j.get('name')] = UrdfJoint(
                name=j.get('name'), type=j.get('type', ''),
                parent=parent.get('link') if parent is not None else '',
                child=child.get('link') if child is not None else '',
                lower=_f('lower'), upper=_f('upper'), velocity=_f('velocity'),
                mimic=mimic.get('joint') if mimic is not None else None)
        self._above = {j.child: j for j in self.joints.values()}
        # {block name: [joint names]} of every <ros2_control> block (vNext M2.7, R22)
        self.ros2_control: Dict[str, List[str]] = {
            (b.get('name') or ''): [j.get('name') for j in b.findall('joint')]
            for b in root.findall('ros2_control')}

    def joint_above(self, link: str) -> Optional[UrdfJoint]:
        return self._above.get(link)

    def ancestors(self, link: str) -> List[str]:
        """The links from ``link`` up to the root, ``link`` excluded."""
        out, seen = [], {link}
        cur = link
        while cur in self._above:
            cur = self._above[cur].parent
            if cur in seen:                      # a malformed loop: stop
                break
            seen.add(cur)
            out.append(cur)
        return out

    def in_subtree(self, link: str, root: str) -> bool:
        """``link`` is ``root`` itself or below it."""
        return link == root or root in self.ancestors(link)

    def has_limits(self, joint: UrdfJoint) -> bool:
        return joint.type in ('revolute', 'prismatic') and joint.lower is not None \
            and joint.upper is not None


@functools.lru_cache(maxsize=16)
def _compiled(path: str, mtime_ns: int, size: int,
              mappings: Tuple[Tuple[str, str], ...] = ()) -> str:
    """The URDF of a description, compiled once per file version (and xacro mappings). xacro's
    own deprecation chatter (a base config's ``load_yaml``) is kept off the user's console:
    generation reports through the manifest."""
    import contextlib
    import io
    from ..robotmodel.xacro_loader import compile_xacro
    with contextlib.redirect_stderr(io.StringIO()):
        return compile_xacro(path, dict(mappings))


def description_view(project: CanonicalProject) -> Tuple[Optional[UrdfView], str]:
    """The project's robot description, compiled and parsed -- or None and why not."""
    desc = project.robot.description
    path = Path(desc.urdf_dir or '.') / (desc.top_xacro or '')
    try:
        st = os.stat(path)
        return UrdfView(_compiled(str(path), st.st_mtime_ns, st.st_size)), ''
    except Exception as exc:                                   # noqa: BLE001 - any reason
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        return None, f'{desc.top_xacro} ({type(exc).__name__}: {reason[:200]})'


def bringup_description(project: CanonicalProject) -> Tuple[Optional[UrdfView], str, str]:
    """The robot description the MOCK bring-up of a scene-loader project loads, compiled -- as
    (view, label, '') or (None, label, why). The bundle copies the base config's ``config/``, and
    MoveItConfigsBuilder loads ``config/<robot_name>.urdf.xacro`` (``_ensure_moveit_infer_names``
    copies the base's only ``*.urdf.xacro`` there when that name is absent), with the mock mapping.
    A from-scratch project has none to compile: its robot description is the project's own
    description plus the block TSA writes from the planning group
    (``robot.ros2_control.xacro.j2``)."""
    base = Path(project.robot.base_moveit_config_path or '') / 'config'
    name = project.robot.robot_name
    path = base / f'{name}.urdf.xacro'
    if not path.is_file():
        found = sorted(base.glob('*.urdf.xacro'))
        if len(found) != 1:
            return None, '', (f'the robot description its mock bring-up loads cannot be named: the '
                              f'base config has no config/{name}.urdf.xacro and {len(found)} '
                              '*.urdf.xacro files (MoveItConfigsBuilder loads '
                              'config/<robot_name>.urdf.xacro)')
        path = found[0]
    label = f'the base config\'s config/{path.name} (the mock bring-up\'s robot_description)'
    try:
        st = os.stat(path)
        return UrdfView(_compiled(str(path), st.st_mtime_ns, st.st_size,
                                  _MOCK_XACRO_MAPPINGS)), label, ''
    except Exception as exc:                                   # noqa: BLE001 - any reason
        reason = (str(exc).splitlines() or [type(exc).__name__])[0]
        return None, label, f'{label} cannot be compiled ({type(exc).__name__}: {reason[:200]})'


def tcp_path(urdf: UrdfView, root: str, tcp: str) -> Optional[List[UrdfJoint]]:
    """The joints from the device root ``root`` (eef_parent_link, excluded) down to ``tcp`` (its
    own joint included), root first: the KINEMATIC PATH device -> TCP of owner Q7-i. [] when the
    TCP is the root itself; None when ``tcp`` is not ``root`` or below it (R9 says it)."""
    if tcp == root:
        return []
    chain, link = [], tcp
    while link != root:
        joint = urdf.joint_above(link)
        if joint is None or len(chain) > len(urdf.joints):     # the URDF root, or a loop
            return None
        chain.append(joint)
        link = joint.parent
    return list(reversed(chain))


def mimic_master(urdf: UrdfView, joint: UrdfJoint) -> Optional[UrdfJoint]:
    """The joint that moves ``joint``: itself, or -- for a mimic follower -- its master (followed
    through a chain); None for a master that is not in the description, or a circular chain."""
    seen = {joint.name}
    while joint.mimic:
        master = urdf.joints.get(joint.mimic)
        if master is None or master.name in seen:
            return None
        seen.add(master.name)
        joint = master
    return joint


def base_srdf_states(project: CanonicalProject) -> Dict[str, Dict[str, float]]:
    """``{state: {joint: value}}`` of the hand-made base config's SRDF, for the jaw targets a short
    form derives from ``open_state``/``closed_state`` (ADR-0022 D6). Empty without a base."""
    base = project.robot.base_moveit_config_path
    if not base:
        return {}
    srdfs = sorted((Path(base) / 'config').glob('*.srdf'))
    if not srdfs:
        return {}
    try:
        from ..robotmodel.srdf_reader import read_srdf
        info = read_srdf(srdfs[0].read_text())
    except Exception:                                          # noqa: BLE001
        return {}
    out: Dict[str, Dict[str, float]] = {}
    for st in info.group_states:
        out.setdefault(st.name, {}).update(st.joint_values)
    return out


def device(project: CanonicalProject, *, with_srdf: bool = True) -> Optional[Device]:
    """The contract's derived long form for this project (None for a legacy project).

    A ros2_control binding's EFFECTIVE flavour is resolved here, once: the declared one, else --
    on gazebo -- the base config's controller type (ADR-0022 D2.B). R7, R15 and R16 (through
    ``DeviceBinding.provides()``) all read that one value."""
    g = project.robot.gripper
    if g.contract is None:
        return None
    dev = derive_device(g, project.robot.named_states,
                        base_srdf_states(project) if with_srdf else None,
                        stage_topic=project.scene.gripper_cmd_topic or None)
    for backend, b in dev.bindings.items():
        if backend is Backend.GAZEBO and b.adapter == 'ros2_control' and b.flavour is None:
            b.effective_flavour = _base_controller_flavour(project)
    return dev


def deployed(project: CanonicalProject) -> List[Backend]:
    return [Backend(m) for m in project.deployment.modes]


# --- what the tree addresses -------------------------------------------------------------------

def action_site(dev: Device, action) -> Optional[DeviceSite]:
    """The site a grasp or a release addresses: its own, or the only one."""
    if action.site is not None:
        return dev.site(action.site)
    return dev.default_site()


def grasped_and_released_sites(project: CanonicalProject, dev: Device) -> Tuple[set, set]:
    """The sites the application grasps and releases: its grasp/release tool actions, plus the
    default site a Pure/Residual policy grasps (ADR-0022 D4)."""
    grasped, released = set(), set()
    for action in project.application.tool_actions:
        site = action_site(dev, action)
        if site is None:
            continue
        if action.kind is ToolActionKind.GRASP:
            grasped.add(site.id)
        elif action.kind is ToolActionKind.RELEASE:
            released.add(site.id)
    if dev.default_site() is not None and any(
            p.mode in (PolicyMode.PURE, PolicyMode.RESIDUAL)
            for p in project.application.policies):
        grasped.add(dev.default_site().id)
    return grasped, released


def moved_by(urdf: UrdfView, dev: Device, site: DeviceSite) -> List[str]:
    """POSE RELEVANCE (ADR-0022 fixture P9): the actuators whose joints move ``site``'s TCP --
    every actuator with a joint on the path from the TCP up to the root, in contract order. A jaw
    beside the TCP moves no site; an axis above it does."""
    if not site.tcp or site.tcp not in urdf.links:
        return []
    path = {site.tcp} | set(urdf.ancestors(site.tcp))
    out = []
    for a in dev.actuators:
        for jn in a.joints or []:
            j = urdf.joints.get(jn)
            if j is not None and j.child in path:
                out.append(a.id)
                break
    return out


# --- the bridge rows a deployment carries, by role (catalog DATA, R18) --------------------------

def bridge_role(row: LaunchNodeSpec) -> Optional[dict]:
    for entry in EEF_BRIDGE_ROLES:
        if entry['package'] == row.package and entry['executable'] == row.executable:
            return entry
    return None


def _merger_gripper_row(project: CanonicalProject,
                        mode: str) -> Optional[Tuple[str, LaunchNodeSpec, str]]:
    """(topic, row, key) of the gripper joint-state input of the kept joint-state merger active in
    ``mode``, or None."""
    for row in project.deployment.bridges:
        role = bridge_role(row)
        if not role or role['role'] != 'gripper_state_merger' or mode not in row.modes:
            continue
        key = role.get('gripper_input')
        for params in row.parameters:
            if key in params and isinstance(params[key], str) and params[key]:
                return params[key], row, key
    return None


def _merger_gripper_input(project: CanonicalProject, mode: str) -> Optional[str]:
    """The gripper joint-state input of the kept joint-state merger active in ``mode``: where the
    adapter's pseudo-joint echo goes (ADR-0022 D6 BRIDGES: the merger row is kept, fed by it)."""
    found = _merger_gripper_row(project, mode)
    return found[0] if found else None


def pseudo_joint_echo_topic(project: CanonicalProject, mode: str) -> str:
    """Where the adapter publishes the pseudo-joint echo in ``mode``: the merger's input, else the
    legacy name (relative under a cell namespace). ONE function for adapter_rows and R26."""
    return (_merger_gripper_input(project, mode)
            or pseudo_joint_topic_default(project.robot.namespace()))


# --- the pseudo-joint echo, the card's Trigger pair --------------------------------------------

def _actuator_joints(dev: Device) -> set:
    return {j for a in dev.actuators for j in (a.joints or [])}


def pseudo_joint(project: CanonicalProject, dev: Device,
                 urdf: Optional[UrdfView]) -> Optional[UrdfJoint]:
    """The description pseudo-joint the adapter echoes (ADR-0022 D6): ``command_joint`` when it
    names a non-fixed joint of the description that no actuator drives. None otherwise."""
    cj = project.robot.gripper.command_joint
    if not cj or urdf is None or cj in _actuator_joints(dev):
        return None
    joint = urdf.joints.get(cj)
    if joint is None or joint.type == 'fixed':
        return None
    return joint


def _card_services(card_path: str) -> Tuple[Optional[Tuple[str, str]], str]:
    """The policy card's Trigger pair (``grasp.on_service`` / ``off_service``), or None and why.
    The policy runtime's own defaults apply where the card names none (EEF-D7: no change)."""
    if not card_path or not os.path.isfile(card_path):
        return None, 'the card cannot be read'
    try:
        with open(card_path) as fh:
            card = (yaml.safe_load(fh) or {}).get('policy_card', {}) or {}
        grasp = card.get('grasp', {}) or {}
    except Exception as exc:                                   # noqa: BLE001
        return None, f'the card does not parse ({type(exc).__name__})'
    pair = (grasp.get('on_service', _TRIGGER_DEFAULTS[0]),
            grasp.get('off_service', _TRIGGER_DEFAULTS[1]))
    bad = [n for n in pair if not (isinstance(n, str) and _ROS_NAME.match(n))]
    if bad:
        return None, (f'its Trigger pair {list(pair)!r} does not resolve at the root to a name '
                      f'the adapter can serve (a ROS name; never private "~")')
    return pair, ''


# --- REFUSALS -----------------------------------------------------------------------------------

def legacy_problems(project: CanonicalProject) -> List[str]:
    """The refusals that hold WITHOUT a contract (ADR-0022): ``actuate`` (R11), the guard token or
    schema_version 3 with no contract block (R19) -- and, like ``actuate`` a token that exists
    only with a contract, kind ``composite`` (R1)."""
    out: List[str] = []
    g = project.robot.gripper
    if g.contract is None:
        if g.kind is GripperKind.COMPOSITE:
            out.append(f'{_prefix(1)}: kind "composite" without a contract block: a composite '
                       'device exists only as an end-effector contract (its actuators and '
                       'sites maps). Write robot.gripper.contract.')
        for a in project.application.tool_actions:
            if a.kind is ToolActionKind.ACTUATE:
                out.append(f'{_prefix(11)}: tool action "actuate" at "{a.at_waypoint}" '
                           f'(actuator {a.actuator!r}, target {a.target!r}) commands an actuator '
                           'of the end-effector contract, and this project has none. Add '
                           'robot.gripper.contract, or remove the action.')
        if g.actuation is EndEffectorActuation.CONTRACT:
            out.append(f'{_prefix(19)}: actuation "contract" without a contract block -- the '
                       'project was saved with an end-effector contract that is no longer in '
                       'it. Restore the contract, or choose trigger/joint_position.')
        if project.schema_version == SCHEMA_VERSION_EEF_CONTRACT:
            out.append(f'{_prefix(19)}: schema_version 3 without a contract block -- saved with '
                       'an end-effector contract that is no longer in the file.')
    return out


def eef_contract_problems(project: CanonicalProject) -> List[str]:
    """Every ADR-0022 refusal of this project, as sentences (empty = generate). R2 is enforced at
    load (``extra='forbid'`` and the closed tokens): a project in memory cannot hold one."""
    out = legacy_problems(project)
    g = project.robot.gripper
    if g.contract is None:
        return out
    dev = device(project)
    urdf, urdf_why = description_view(project)
    ctx = _Checks(project, dev, urdf, urdf_why)
    if g.kind is GripperKind.NONE:
        return out + ctx.r1_kind()
    if urdf is None:
        out.append(f'{_prefix(9)}: device "{dev.id}": the robot description cannot be read '
                   f'({urdf_why}), so its joints, limits and site frames cannot be checked '
                   '(R5, R6, R8, R9, R10) -- generation refuses rather than guess (fail closed).')
    for check in (ctx.r1_kind, ctx.r2_values, ctx.r3_names, ctx.r4_actions, ctx.r5_joints,
                  ctx.r6_set_bool, ctx.r7_gripper_command, ctx.r8_trajectory, ctx.r9_sites,
                  ctx.r10_tip, ctx.r11_tool_actions, ctx.r12_attach, ctx.r13_bin_picking,
                  ctx.r14_policies, ctx.r15_bindings, ctx.r16_capabilities,
                  ctx.r17_verification, ctx.r18_bridges, ctx.r20_backends, ctx.r21_interaction,
                  ctx.r22_composite_bundle, ctx.r23_legacy_fields, ctx.r24_path_interfaces,
                  ctx.r25_path_actuated, ctx.r26_echo_topic):
        out += check()
    return out


def refuse_eef_contract(project: CanonicalProject) -> None:
    """REFUSE what the end-effector contract refuses (ADR-0022 R1-R26), before a single file is
    written -- called by both generation entry points."""
    problems = eef_contract_problems(project)
    if problems:
        raise ValueError('\n'.join(problems))


class _Checks:
    """One refusal row per method. Each returns sentences; none raises."""

    def __init__(self, project, dev: Device, urdf: Optional[UrdfView], urdf_why: str):
        self.p = project
        self.g = project.robot.gripper
        self.dev = dev
        self.urdf = urdf
        self.urdf_why = urdf_why
        self.modes = deployed(project)
        self.name = f'device "{dev.id}"'

    # helpers
    def _a(self, a: DeviceActuator) -> str:
        return f'{self.name} actuator "{a.id}"'

    def _s(self, s: DeviceSite) -> str:
        return f'{self.name} site "{s.id}"'

    def _simple(self) -> bool:
        return (self.dev.kind is not GripperKind.COMPOSITE and len(self.dev.actuators) == 1
                and len(self.dev.sites) == 1)

    def _none_exempt(self) -> bool:
        """A ``none`` binding is allowed for a simple device whose require_* keys are both false
        (ADR-0022 D6): R15/R16 exempt it."""
        return (self._simple() and self.dev.require_grasp_confirmation is False
                and self.dev.require_release_confirmation is False)

    def _bound(self, backend: Backend):
        b = self.dev.bindings.get(backend)
        return b if b is not None and b.adapter != 'none' else None

    # R1 --------------------------------------------------------------------------------------
    def r1_kind(self) -> List[str]:
        c, dev, kind, out = self.g.contract, self.dev, self.g.kind, []
        if kind is GripperKind.NONE:
            return [f'{_prefix(1)}: {self.name}: kind "none" has no contract -- choose the kind '
                    'the structure is (suction, parallel or composite).']
        if c.actuator is not None and (c.actuators is not None or c.sites is not None):
            out.append(f'{_prefix(1)}: {self.name}: the short form (actuator) cannot be combined '
                       'with actuators/sites -- write one form.')
        if c.actuator is None and (c.actuators is None or c.sites is None):
            out.append(f'{_prefix(1)}: {self.name}: the long form needs both an actuators map and '
                       'a sites map (or write the short form, actuator).')
        if c.actuators is not None and not c.actuators:
            out.append(f'{_prefix(1)}: {self.name}: actuators is empty -- a device has at least '
                       'one actuator.')
        if c.sites is not None and not c.sites:
            out.append(f'{_prefix(1)}: {self.name}: sites is empty -- a device has at least one '
                       'grasp site (process tools stay on ProcessController, ADR-0022 D0).')
        if c.actuator is not None and kind not in (GripperKind.SUCTION, GripperKind.PARALLEL):
            out.append(f'{_prefix(1)}: {self.name}: the short form is for kind suction or '
                       f'parallel; kind "{kind.value}" writes the actuators and sites maps.')
        ifaces = [a.interface for a in dev.actuators]
        if kind is GripperKind.SUCTION:
            if any(i is not CommandInterface.SET_BOOL for i in ifaces):
                out.append(f'{_prefix(1)}: {self.name}: kind suction is 1..N set_bool actuators; '
                           f'this one has {[i.value for i in ifaces]} -- a jaw or an axis makes '
                           'it parallel or composite.')
            if len(dev.sites) != 1:
                out.append(f'{_prefix(1)}: {self.name}: kind suction has exactly one site, not '
                           f'{len(dev.sites)}.')
        elif kind is GripperKind.PARALLEL:
            if len(dev.actuators) != 1 or any(i is CommandInterface.FOLLOW_JOINT_TRAJECTORY
                                              for i in ifaces):
                out.append(f'{_prefix(1)}: {self.name}: kind parallel is exactly one actuator '
                           '(gripper_command or set_bool) and one site, not '
                           f'{[i.value for i in ifaces]}.')
            if len(dev.sites) != 1:
                out.append(f'{_prefix(1)}: {self.name}: kind parallel has exactly one site, not '
                           f'{len(dev.sites)}.')
        return out

    # R2 (values the schema can hold but the contract refuses) --------------------------------
    def r2_values(self) -> List[str]:
        if not isinstance(self.dev.command_timeout_ms, int) or self.dev.command_timeout_ms <= 0:
            return [f'{_prefix(2)}: {self.name}: command_timeout_ms '
                    f'{self.dev.command_timeout_ms!r} is not a positive integer.']
        out = []
        # a NaN or an infinity passes every "> 0" check and would reach a goal or a deadline
        for a in self.dev.actuators:
            for key in ('targets', 'joint_positions', 'goal_tolerance', 'path_tolerance',
                        'duration_s', 'goal_time_tolerance_s', 'max_effort'):
                value = getattr(a, key, None)
                if value is not None and not _finite(value):
                    out.append(f'{_prefix(2)}: {self._a(a)}: {key} {value!r} holds a value that '
                               'is not a finite number.')
        return out

    # R3 --------------------------------------------------------------------------------------
    def r3_names(self) -> List[str]:
        dev, out = self.dev, []
        if not NAME_TOKEN.match(dev.id or '') or dev.id in ('eef', 'ids'):
            # 'ids' is reserved too: eef.<id> would overwrite the block's own eef.ids list
            out.append(f'{_prefix(3)}: device id {dev.id!r} is not a ROS name token (a letter '
                       'first, then letters, digits or "_"; "eef" and "ids" are reserved).')
        for seg in [s for s in (self.p.robot.namespace() or '').split('/') if s]:
            if seg == 'eef':
                out.append(f'{_prefix(3)}: the cell namespace "{self.p.robot.namespace()}" has a '
                           'segment named "eef": the device names eef/<id> would be ambiguous.')
        for a in dev.actuators:
            if not NAME_TOKEN.match(a.id or ''):
                out.append(f'{_prefix(3)}: {self.name}: actuator id {a.id!r} is not a ROS name '
                           'token.')
            if not dev.short_form and getattr(a.spec, 'id', None) is not None:
                out.append(f'{_prefix(3)}: {self._a(a)}: "id" is the short form\'s; in the '
                           'actuators map the key is the id.')
            for t in (a.targets or {}):
                if not NAME_TOKEN.match(str(t)):
                    out.append(f'{_prefix(3)}: {self._a(a)}: target name {t!r} is not a ROS name '
                               'token.')
            if a.action is not None and not _relative_name(a.action):
                out.append(f'{_prefix(3)}: {self._a(a)}: action {a.action!r} is not a RELATIVE '
                           'ROS name (no leading "/" or "~": it resolves in the cell namespace, '
                           'ADR-0022 D7).')
        for s in dev.sites:
            if not NAME_TOKEN.match(s.id or ''):
                out.append(f'{_prefix(3)}: {self.name}: site id {s.id!r} is not a ROS name '
                           'token.')
        both = {a.id for a in dev.actuators} & {s.id for s in dev.sites}
        if both:
            out.append(f'{_prefix(3)}: {self.name}: {sorted(both)} is both an actuator and a site '
                       'id -- one token, one thing.')
        for b in dev.bindings.values():
            for key in ('stage_cmd_topic', 'stage_status_topic'):
                topic = getattr(b, key)
                if topic is not None and not _ROS_NAME.match(topic):
                    out.append(f'{_prefix(3)}: {self.name} on {b.backend.value}: {key} '
                               f'{topic!r} is not a ROS topic name.')
        return out

    # R4 --------------------------------------------------------------------------------------
    def r4_actions(self) -> List[str]:
        out, seen = [], {}
        arm = self.p.robot.arm_controller
        arm_action = f'{arm.name}/{arm.action_ns}'
        for a in self.dev.actuators:
            if a.action is None:
                continue
            if a.action in seen:
                out.append(f'{_prefix(4)}: {self._a(a)} and actuator "{seen[a.action]}" share '
                           f'the action "{a.action}": one commander per controller.')
            seen.setdefault(a.action, a.id)
            if a.action.strip('/') == arm_action:
                out.append(f'{_prefix(4)}: {self._a(a)}: "{a.action}" is the arm controller\'s '
                           'action; a tool actuator never commands the planning group.')
        return out

    # R5 --------------------------------------------------------------------------------------
    def r5_joints(self) -> List[str]:
        jointed = [a for a in self.dev.actuators if a.joints]
        out: List[str] = []
        if self.urdf is None:
            return out
        parent = self.g.eef_parent_link
        if not parent or parent not in self.urdf.links:
            out.append(f'{_prefix(5)}: {self.name}: eef_parent_link {parent!r} is not a link of '
                       'the description; the device subtree hangs from it.')
        arm = set(self.p.robot.planning_group.joints)
        owner: Dict[str, str] = {}
        for a in jointed:
            for jn in a.joints:
                if jn in owner:
                    out.append(f'{_prefix(5)}: {self._a(a)}: joint "{jn}" is also driven by '
                               f'actuator "{owner[jn]}".')
                    continue
                owner[jn] = a.id
                j = self.urdf.joints.get(jn)
                if j is None:
                    out.append(f'{_prefix(5)}: {self._a(a)}: joint "{jn}" is not in the '
                               'description.')
                    continue
                if j.type == 'fixed':
                    out.append(f'{_prefix(5)}: {self._a(a)}: joint "{jn}" is fixed.')
                if j.mimic:
                    out.append(f'{_prefix(5)}: {self._a(a)}: joint "{jn}" is a mimic follower of '
                               f'"{j.mimic}": followers are never actuator joints.')
                if jn in arm:
                    out.append(f'{_prefix(5)}: {self._a(a)}: joint "{jn}" belongs to the arm '
                               f'group "{self.p.robot.planning_group.name}".')
                if parent and parent in self.urdf.links and not self.urdf.in_subtree(
                        j.parent, parent):
                    out.append(f'{_prefix(5)}: {self._a(a)}: joint "{jn}" is outside the subtree '
                               f'under eef_parent_link "{parent}".')
        return out

    def _limits(self, row: int, a: DeviceActuator, jn: str, value: float,
                what: str) -> List[str]:
        j = self.urdf.joints.get(jn) if self.urdf else None
        if j is None or not self.urdf.has_limits(j):
            return []
        if not (j.lower - 1e-12 <= value <= j.upper + 1e-12):
            return [f'{_prefix(row)}: {self._a(a)}: {what} {value!r} of joint "{jn}" is outside '
                    f'its limits [{j.lower!r}, {j.upper!r}].']
        return []

    # R6 --------------------------------------------------------------------------------------
    def r6_set_bool(self) -> List[str]:
        out = []
        for a in self.dev.actuators:
            if a.interface is not CommandInterface.SET_BOOL:
                continue
            if (a.joints is None) != (a.joint_positions is None):
                out.append(f'{_prefix(6)}: {self._a(a)}: joints and joint_positions go together '
                           '(both, or neither).')
            elif a.joints is not None:
                engaged, released = a.joint_positions
                if not a.joints or len(engaged) != len(a.joints) or len(released) != len(a.joints):
                    out.append(f'{_prefix(6)}: {self._a(a)}: joint_positions need one number per '
                               f'joint ({len(a.joints)} joints, engaged {len(engaged)}, released '
                               f'{len(released)}).')
                else:
                    for jn, ev, rv in zip(a.joints, engaged, released):
                        out += (self._limits(6, a, jn, ev, 'engaged')
                                + self._limits(6, a, jn, rv, 'released'))
            if not isinstance(a.confirm_timeout_ms, int) or a.confirm_timeout_ms <= 0:
                out.append(f'{_prefix(6)}: {self._a(a)}: confirm_timeout_ms '
                           f'{a.confirm_timeout_ms!r} is not a positive integer.')
        return out

    # R7 --------------------------------------------------------------------------------------
    def r7_gripper_command(self) -> List[str]:
        out = []
        grips = {s.grip for s in self.dev.sites}
        for a in self.dev.actuators:
            if a.interface is not CommandInterface.GRIPPER_COMMAND:
                continue
            for key, why in a.unresolved.items():
                out.append(f'{_prefix(7)}: {self._a(a)}: {key} cannot be derived: {why}.')
            if a.action is None and 'action' not in a.unresolved:
                out.append(f'{_prefix(7)}: {self._a(a)}: the action is required.')
            if a.joints is not None and len(a.joints) != 1:
                out.append(f'{_prefix(7)}: {self._a(a)}: a gripper_command actuator drives '
                           f'exactly one joint (followers move by mimic), not {a.joints}.')
            if a.joints is None and 'joints' not in a.unresolved:
                out.append(f'{_prefix(7)}: {self._a(a)}: joints are required (exactly one).')
            if a.id not in grips:
                out.append(f'{_prefix(7)}: {self._a(a)}: a gripper_command actuator must be the '
                           'grip of a site -- an axis is never a GripperCommand (EEF-D2).')
            if a.targets is not None:
                missing = [t for t in ('open', 'closed') if t not in a.targets]
                if missing:
                    out.append(f'{_prefix(7)}: {self._a(a)}: targets need "open" and "closed" '
                               f'(missing {missing}).')
                if a.joints and len(a.joints) == 1:
                    for t, v in a.targets.items():
                        out += self._limits(7, a, a.joints[0], float(v), f'target "{t}"')
                # an adapter tells ENGAGE / RELEASE / a named target apart by the goal's position
                # alone (GripperCommand carries no name): equal values make two commands one
                by_value: Dict[float, List[str]] = {}
                for t, v in a.targets.items():
                    by_value.setdefault(float(v), []).append(str(t))
                shared = [names for names in by_value.values() if len(names) > 1]
                if shared:
                    out.append(f'{_prefix(7)}: {self._a(a)}: targets {shared} share a value: a '
                               'GripperCommand goal carries only a position, so an adapter tells '
                               'the engage, the release and a named target apart by it -- give '
                               'every target its own value.')
            elif 'targets' not in a.unresolved:
                out.append(f'{_prefix(7)}: {self._a(a)}: targets are required.')
            if a.result_policy is None:
                out.append(f'{_prefix(7)}: {self._a(a)}: result_policy is required '
                           '(succeeded or succeeded_or_stalled).')
            if a.max_effort is not None and not a.max_effort > 0:
                out.append(f'{_prefix(7)}: {self._a(a)}: max_effort {a.max_effort!r} must be > 0.')
            for backend in self.modes:
                b = self.dev.bindings.get(backend)
                if (b is not None and b.adapter == 'ros2_control'
                        and self._flavour(backend, b) == ControllerFlavour.EFFORT.value
                        and a.max_effort is None):
                    out.append(f'{_prefix(7)}: {self._a(a)} on {backend.value}: an effort-flavour '
                               'controller never moves with max_effort 0.0 -- declare max_effort '
                               '> 0.')
        return out

    # R8 --------------------------------------------------------------------------------------
    def r8_trajectory(self) -> List[str]:
        out = []
        grips = {s.grip: s.id for s in self.dev.sites}
        for a in self.dev.actuators:
            if a.interface is not CommandInterface.FOLLOW_JOINT_TRAJECTORY:
                continue
            n = len(a.joints or [])
            if not a.joints:
                out.append(f'{_prefix(8)}: {self._a(a)}: joints are required.')
            if a.action is None:
                out.append(f'{_prefix(8)}: {self._a(a)}: the action is required.')
            if not a.targets:
                out.append(f'{_prefix(8)}: {self._a(a)}: at least one named target is required.')
            gt = _per_joint(a.goal_tolerance, n)
            if a.goal_tolerance is None:
                out.append(f'{_prefix(8)}: {self._a(a)}: goal_tolerance is required -- with the '
                           'default 0 the controller never checks it, and SUCCEEDED would only '
                           'mean "the time elapsed".')
            elif gt is None or any(x <= 0 for x in gt):
                out.append(f'{_prefix(8)}: {self._a(a)}: goal_tolerance {a.goal_tolerance!r} must '
                           f'be one number > 0 or {n} numbers > 0.')
            if a.duration_s is None or not a.duration_s > 0:
                out.append(f'{_prefix(8)}: {self._a(a)}: duration_s {a.duration_s!r} must be '
                           'given and > 0.')
            if not a.goal_time_tolerance_s > 0:
                out.append(f'{_prefix(8)}: {self._a(a)}: goal_time_tolerance_s '
                           f'{a.goal_time_tolerance_s!r} must be > 0.')
            if a.path_tolerance is not None:
                pt = _per_joint(a.path_tolerance, n)
                if pt is None or any(x <= 0 for x in pt):
                    out.append(f'{_prefix(8)}: {self._a(a)}: path_tolerance {a.path_tolerance!r} '
                               f'must be one number > 0 or {n} numbers > 0.')
            for t, values in (a.targets or {}).items():
                if len(values) != n:
                    out.append(f'{_prefix(8)}: {self._a(a)}: target "{t}" has {len(values)} '
                               f'values for {n} joints.')
                    continue
                for jn, v in zip(a.joints or [], values):
                    out += self._limits(8, a, jn, float(v), f'target "{t}"')
            if a.joints and self.urdf is not None and a.duration_s and a.duration_s > 0:
                for jn in a.joints:
                    j = self.urdf.joints.get(jn)
                    if j is None or not self.urdf.has_limits(j) or j.velocity is None:
                        continue
                    need = (j.upper - j.lower) / a.duration_s
                    if need > j.velocity + 1e-12:
                        out.append(f'{_prefix(8)}: {self._a(a)}: joint "{jn}" would need '
                                   f'{need:.4g} per second ((upper - lower) / duration_s) and its '
                                   f'URDF velocity limit is {j.velocity!r} -- the controller '
                                   'enforces none.')
            if a.id in grips:
                out.append(f'{_prefix(8)}: {self._a(a)}: a follow_joint_trajectory actuator never '
                           f'grips (site "{grips[a.id]}"): a squeeze is not a trajectory.')
        return out

    # R9 --------------------------------------------------------------------------------------
    def r9_sites(self) -> List[str]:
        out: List[str] = []
        by_grip: Dict[str, str] = {}
        parent = self.g.eef_parent_link
        for s in self.dev.sites:
            grip = self.dev.actuator(s.grip)
            if s.grip is None or grip is None:
                out.append(f'{_prefix(9)}: {self._s(s)}: grip {s.grip!r} is not an actuator of '
                           'this device.')
            elif s.grip in by_grip:
                out.append(f'{_prefix(9)}: {self._s(s)}: actuator "{s.grip}" already grips site '
                           f'"{by_grip[s.grip]}" -- an actuator grips at most one site.')
            if s.grip is not None:
                by_grip.setdefault(s.grip, s.id)
            if grip is not None and grip.interface is not CommandInterface.FOLLOW_JOINT_TRAJECTORY:
                names = grip.target_names()
                for key, t in (('engage', s.engage), ('release', s.release)):
                    if t is None or t not in names:
                        out.append(f'{_prefix(9)}: {self._s(s)}: {key} target {t!r} is not a '
                                   f'target of "{grip.id}" ({names}).')
                if s.engage is not None and s.engage == s.release:
                    out.append(f'{_prefix(9)}: {self._s(s)}: engage and release are both '
                               f'"{s.engage}".')
                elif (grip.interface is CommandInterface.SET_BOOL and s.engage in names
                      and s.engage != SET_BOOL_TARGETS[0]):
                    # TMR sends SetBool data true for "engage", and the adapter records it as
                    # ENGAGE: a swapped site would grip on OpenGripper (normalize_eef.py refuses it)
                    out.append(f'{_prefix(9)}: {self._s(s)}: the grip "{grip.id}" is set_bool, so '
                               f'the site engages with "engage" and releases with "release" (not '
                               f'engage "{s.engage}", release "{s.release}") -- its polarity is '
                               'the adapter\'s, never the contract\'s (ADR-0022 D2.A): leave '
                               'engage/release unset.')
            if not s.tcp:
                out.append(f'{_prefix(9)}: {self._s(s)}: the tcp is required (the short form '
                           'derives it from eef_parent_link).')
            elif self.urdf is not None:
                if s.tcp not in self.urdf.links:
                    out.append(f'{_prefix(9)}: {self._s(s)}: tcp "{s.tcp}" is not a link of the '
                               'description.')
                elif parent and not self.urdf.in_subtree(s.tcp, parent):
                    out.append(f'{_prefix(9)}: {self._s(s)}: tcp "{s.tcp}" is outside the subtree '
                               f'under eef_parent_link "{parent}" (the link itself included).')
                elif s.tcp != parent:
                    above = self.urdf.joint_above(s.tcp)
                    if above is not None and above.type != 'fixed':
                        out.append(f'{_prefix(9)}: {self._s(s)}: tcp "{s.tcp}" hangs from the '
                                   f'{above.type} joint "{above.name}" -- a TCP is a fixed frame.')
        return out

    # R10 -------------------------------------------------------------------------------------
    def r10_tip(self) -> List[str]:
        if self.urdf is None:
            return []
        out = []
        tips = dict.fromkeys([self.p.robot.tip_link, self.p.robot.planning_group.tip_link])
        for a in self.dev.actuators:
            for jn in a.joints or []:
                j = self.urdf.joints.get(jn)
                if j is None:
                    continue
                for tip in tips:
                    if tip and self.urdf.in_subtree(tip, j.child):
                        out.append(f'{_prefix(10)}: tip_link "{tip}" is below the joint "{jn}" of '
                                   f'{self._a(a)}: MoveIt resolves an IK tip through fixed '
                                   'joints only -- plan to a link above the tool axes.')
        return out

    # R11 -------------------------------------------------------------------------------------
    def r11_tool_actions(self) -> List[str]:
        out, dev = [], self.dev
        rendered = (ToolActionKind.GRASP, ToolActionKind.RELEASE, ToolActionKind.ATTACH,
                    ToolActionKind.DETACH, ToolActionKind.RESET_SCENE, ToolActionKind.ACTUATE)
        for act in self.p.application.tool_actions:
            where = f'tool action "{getattr(act.kind, "value", act.kind)}" at "{act.at_waypoint}"'
            if act.kind not in rendered:
                out.append(f'{_prefix(11)}: {where}: an unknown kind in a canonical project.')
                continue
            if act.kind is ToolActionKind.ACTUATE:
                a = dev.actuator(act.actuator)
                if act.site is not None:
                    out.append(f'{_prefix(11)}: {where}: "site" is for grasp/release.')
                if a is None:
                    out.append(f'{_prefix(11)}: {where}: actuator {act.actuator!r} is not an '
                               f'actuator of {self.name} ({[x.id for x in dev.actuators]}).')
                    continue
                if act.target is None or act.target not in a.target_names():
                    out.append(f'{_prefix(11)}: {where}: target {act.target!r} is not a named '
                               f'target of actuator "{a.id}" ({a.target_names()}).')
                for s in dev.sites:
                    if s.grip == a.id and act.target in (s.engage, s.release):
                        out.append(f'{_prefix(11)}: {where}: "{act.target}" is the '
                                   f'{"engage" if act.target == s.engage else "release"} target '
                                   f'of site "{s.id}"\'s grip -- use grasp/release '
                                   '(Close/OpenGripper), which the scene signal and the '
                                   'verification follow.')
                continue
            if act.actuator is not None or act.target is not None:
                out.append(f'{_prefix(11)}: {where}: "actuator"/"target" are for kind actuate.')
            if act.kind in (ToolActionKind.GRASP, ToolActionKind.RELEASE):
                if act.site is None and dev.is_multi_site():
                    out.append(f'{_prefix(11)}: {where}: {self.name} has '
                               f'{len(dev.sites)} sites -- name the site.')
                elif act.site is not None and dev.site(act.site) is None:
                    out.append(f'{_prefix(11)}: {where}: site {act.site!r} is not a site of '
                               f'{self.name} ({[s.id for s in dev.sites]}).')
            elif act.site is not None:
                out.append(f'{_prefix(11)}: {where}: "site" is for grasp/release.')
        return out

    # R12 -------------------------------------------------------------------------------------
    def r12_attach(self) -> List[str]:
        if not self.dev.is_multi_site():
            return []
        out = []
        for act in self.p.application.tool_actions:
            if act.kind in (ToolActionKind.ATTACH, ToolActionKind.DETACH):
                out.append(f'{_prefix(12)}: {self.name} has {len(self.dev.sites)} sites: tool '
                           f'action "{act.kind.value}" at "{act.at_waypoint}" attaches to one '
                           'link, and the scene manager has one attach link (EEF-D12).')
        if self.p.scene.payload is not None:
            out.append(f'{_prefix(12)}: {self.name} has {len(self.dev.sites)} sites: a scene '
                       'payload attaches to one link (EEF-D12).')
        return out

    # R13 -------------------------------------------------------------------------------------
    def r13_bin_picking(self) -> List[str]:
        if self.p.application.type is AppType.BIN_PICKING and self.dev.is_multi_site():
            return [f'{_prefix(13)}: bin picking with {self.name}, which has '
                    f'{len(self.dev.sites)} sites: the emptying loop verifies one grasp.']
        return []

    # R14 -------------------------------------------------------------------------------------
    def r14_policies(self) -> List[str]:
        out, dev = [], self.dev
        pairs: List[Tuple[str, str]] = []
        for pol in self.p.application.policies:
            where = f'policy step "{pol.name}"'
            if dev.kind is GripperKind.COMPOSITE or dev.is_multi_site():
                out.append(f'{_prefix(14)}: {where} on {self.name} (kind {dev.kind.value}, '
                           f'{len(dev.sites)} sites): a policy step drives a simple device only '
                           'in M2.6.')
            grip = dev.grip_of(dev.default_site())
            if grip is not None and grip.interface is not CommandInterface.SET_BOOL:
                out.append(f'{_prefix(14)}: {where}: the grip of {self.name} is '
                           f'{grip.interface.value}; the policy runtime commands a binary grip '
                           'through its card\'s Trigger pair.')
            if self.p.robot.namespace():
                out.append(f'{_prefix(14)}: {where} with the contract under the cell namespace '
                           f'"{self.p.robot.namespace()}": the policy runtime runs at the root '
                           'with absolute card services (until vNext M5).')
            card_eef = _card_end_effector(pol)
            want = {GripperKind.SUCTION: 'suction', GripperKind.PARALLEL: 'parallel_gripper'}.get(
                dev.kind)
            if card_eef and card_eef != want:
                out.append(f'{_prefix(14)}: {where}: the card was trained with end_effector '
                           f'"{card_eef}", and {self.name} is kind {dev.kind.value}.')
            pair, why = _card_services(pol.card)
            if why:
                out.append(f'{_prefix(14)}: {where}: the adapter serves the card\'s Trigger pair '
                           f'at its exact names, and {why}.')
            elif pairs and pair not in pairs:
                out.append(f'{_prefix(14)}: {where}: its card\'s Trigger pair {list(pair)} differs '
                           f'from {list(pairs[0])}: one device serves one pair.')
            if pair is not None and pair not in pairs:
                pairs.append(pair)
        return out

    # R15 -------------------------------------------------------------------------------------
    def _flavour(self, backend: Backend, b) -> Optional[str]:
        """A ros2_control binding's EFFECTIVE flavour (resolved once by :func:`device`): declared
        by the project, else read from the base config's controller type on gazebo (ADR-0022
        D2.B) -- the value provides() reads too, so R7, R15 and R16 never disagree."""
        return b.effective_flavour

    def r15_bindings(self) -> List[str]:
        out, dev = [], self.dev
        for backend in self.modes:
            b = dev.bindings.get(backend)
            where = f'{self.name} on {backend.value}'
            if b is None:
                out.append(f'{_prefix(15)}: {where}: the backend is deployed and has no binding '
                           '(contract.adapters) -- bind it, or use "none" for a simple device '
                           'whose require_* keys are false.')
                continue
            row = b.row
            if backend.value not in row['modes']:
                out.append(f'{_prefix(15)}: {where}: "{b.adapter}" does not run on '
                           f'{backend.value} (it runs on {list(row["modes"])}).')
                continue
            if b.adapter == 'none':
                if not self._none_exempt():
                    out.append(f'{_prefix(15)}: {where}: "none" serves nothing; it is allowed for '
                               'a simple device whose require_grasp_confirmation and '
                               'require_release_confirmation are both false.')
                continue
            if b.long_form:
                extra = [k for k in ('stage_cmd_topic', 'stage_status_topic', 'flavour')
                         if getattr(b, k) is not None and k not in row['binding_keys']]
                if extra:
                    out.append(f'{_prefix(15)}: {where}: "{b.adapter}" reads no {extra}.')
            for key in row['requires']:
                if not getattr(b, key):
                    out.append(f'{_prefix(15)}: {where}: "{b.adapter}" needs {key} -- set it in '
                               'the binding, or detect the stage topic at Step 2 '
                               '(scene.gripper_cmd_topic).')
            for a in dev.actuators:
                if a.interface.value not in row['serves']:
                    if (backend is Backend.ISAAC and a.interface is CommandInterface.SET_BOOL):
                        continue                      # R20 says it for isaac
                    out.append(f'{_prefix(15)}: {self._a(a)} on {backend.value}: "{b.adapter}" '
                               f'does not serve {a.interface.value} (it serves '
                               f'{list(row["serves"])}).')
            if dev.kind.value not in row['kinds'] and not (
                    backend is Backend.ISAAC or dev.kind is GripperKind.COMPOSITE):
                out.append(f'{_prefix(15)}: {where}: "{b.adapter}" does not serve kind '
                           f'{dev.kind.value}.')
            if b.adapter == 'ros2_control':
                for a in dev.actuators:
                    if a.interface is CommandInterface.GRIPPER_COMMAND and a.action is not None:
                        parts = a.action.split('/')
                        if len(parts) != 2 or parts[1] != 'gripper_cmd':
                            out.append(f'{_prefix(15)}: {self._a(a)} on {backend.value}: a '
                                       'ros2_control GripperActionController serves '
                                       f'"<controller>/gripper_cmd", not "{a.action}" (set '
                                       'action_ns: gripper_cmd).')
                read = _base_controller_flavour(self.p) if backend is Backend.GAZEBO else None
                if read and b.flavour and read != b.flavour:
                    out.append(f'{_prefix(15)}: {where}: the binding declares flavour '
                               f'"{b.flavour}", and the base config\'s controller is '
                               f'"{read}".')
                if self._flavour(backend, b) is None:
                    out.append(f'{_prefix(15)}: {where}: the ros2_control flavour (position or '
                               'effort) cannot be read'
                               + (' from the base config\'s controller type' if backend is
                                  Backend.GAZEBO else '')
                               + ' -- declare it: {adapter: ros2_control, flavour: ...}.')
            robot = self.p.robot
            if (b.stage_cmd_topic and b.stage_status_topic
                    and robot.under_namespace(b.stage_cmd_topic)
                    == robot.under_namespace(b.stage_status_topic)):
                out.append(f'{_prefix(15)}: {where}: stage_cmd_topic and stage_status_topic are '
                           f'one channel ("{robot.under_namespace(b.stage_cmd_topic)}").')
            ns = robot.namespace()
            for key in ('stage_cmd_topic', 'stage_status_topic'):
                topic = getattr(b, key)
                if not topic:
                    continue
                inside = bool(ns) and (topic == ns or topic.startswith(ns + '/'))
                if (topic[len(ns):] if inside else topic).lstrip('/').startswith(
                        f'eef/{dev.id}/'):
                    out.append(f'{_prefix(15)}: {where}: {key} "{topic}" is one of the device\'s '
                               'own contract names.')
                if ns and topic.startswith('/') and not inside:
                    # two robots of one cell would share it: latched, and the stage would drive
                    # (or report) whichever SurfaceGripper it binds to that name
                    stage = 'subscriber' if key == 'stage_cmd_topic' else 'status publisher'
                    out.append(f'{_prefix(15)}: {where}: {key} "{topic}" is absolute and outside '
                               f'the cell namespace "{ns}": a latched simulator channel exists '
                               'only per robot (ADR-0022 D7; a global one only in a single-robot '
                               f'cell) -- give a relative name (or one under "{ns}/"), and bind '
                               f'the stage\'s {stage} to this robot\'s SurfaceGripper by prim '
                               'path.')
        return out

    # R16 -------------------------------------------------------------------------------------
    def r16_capabilities(self) -> List[str]:
        out, dev = [], self.dev
        grasped, released = grasped_and_released_sites(self.p, dev)
        for backend in self.modes:
            b = self._bound(backend)
            if b is None or backend.value not in b.row['modes']:
                continue                               # R15 says it (or `none` is exempt)
            provided = set(b.provides())
            for a in dev.actuators:
                if a.interface.value not in b.row['serves']:
                    continue                           # R15/R20 say it
                need = [c.value for c in IMPLIED_CAPABILITIES[a.interface]]
                if a.actuation_feedback:
                    need.append(EndEffectorCapability.ACTUATION_FEEDBACK.value)
                if a.max_effort is not None:
                    need.append(EndEffectorCapability.EFFORT_CONTROL.value)
                missing = [c for c in need if c not in provided]
                if missing:
                    out.append(f'{_prefix(16)}: {self._a(a)} on {backend.value}: '
                               f'"{b.adapter}" does not provide {missing}.')
            for s in dev.sites:
                need = []
                if dev.require_grasp_confirmation and s.id in grasped:
                    need.append(EndEffectorCapability.GRASP_DETECTION.value)
                if dev.require_release_confirmation and s.id in released:
                    need.append(EndEffectorCapability.RELEASE_DETECTION.value)
                if not need:
                    continue
                missing = [c for c in need if c not in provided]
                if missing:
                    extra = (' (the isaac adapter detects only with the stage status topic: '
                             '{adapter: trainit_isaac_surface_gripper, stage_status_topic: ...})'
                             if b.adapter == 'trainit_isaac_surface_gripper' else '')
                    out.append(f'{_prefix(16)}: {self._s(s)} on {backend.value}: confirmation is '
                               f'required and "{b.adapter}" does not provide {missing}{extra}.')
                    continue
                classes = set(b.evidence())          # the mock's: the classes it can be set to
                if not classes & set(dev.accepted_evidence):
                    out.append(f'{_prefix(16)}: {self._s(s)} on {backend.value}: "{b.adapter}" '
                               f'observes {sorted(classes)} and the project accepts '
                               f'{dev.accepted_evidence} -- nothing it detects would be '
                               'accepted.')
        return out

    # R17 -------------------------------------------------------------------------------------
    def r17_verification(self) -> List[str]:
        out, dev = [], self.dev
        for key in ('require_grasp_confirmation', 'require_release_confirmation'):
            if getattr(dev, key) is None:
                out.append(f'{_prefix(17)}: {self.name}: verification.{key} is required -- '
                           '"false" (command success is sufficient) is written, never implied.')
        never = [e for e in dev.accepted_evidence if e in {x.value for x in NEVER_EVIDENCE}]
        if never:
            out.append(f'{_prefix(17)}: {self.name}: accepted_evidence lists {never}: a command '
                       'echo or a simulator weld is never evidence (EEF-D8).')
        if not dev.accepted_evidence or len(set(dev.accepted_evidence)) != len(
                dev.accepted_evidence):
            out.append(f'{_prefix(17)}: {self.name}: accepted_evidence {dev.accepted_evidence} '
                       'must be a non-empty list of unique classes.')
        for key in ('timeout_ms', 'max_age_ms', 'clock_tolerance_ms'):
            v = getattr(dev, key)
            if not isinstance(v, int) or v <= 0:
                out.append(f'{_prefix(17)}: {self.name}: verification.{key} {v!r} is not a '
                           'positive integer.')
        req = dev.require_grasp_confirmation
        app = self.p.application
        if app.type is AppType.BIN_PICKING and req is not None:
            bp = app.bin_picking
            if bp.holding_check != req:
                out.append(f'{_prefix(17)}: bin_picking.holding_check is {bp.holding_check} and '
                           f'{self.name} require_grasp_confirmation is {req}: one key decides '
                           f'-- set holding_check to {req}.')
            if bp.attached_topic:
                out.append(f'{_prefix(17)}: bin_picking.attached_topic "{bp.attached_topic}": the '
                           'holding check of a canonical project reads the device\'s own state '
                           '(VerifyGrasp) -- leave it empty.')
            if bp.holding_timeout_ms not in (_DEFAULT_HOLDING_TIMEOUT_MS, dev.timeout_ms):
                out.append(f'{_prefix(17)}: bin_picking.holding_timeout_ms '
                           f'{bp.holding_timeout_ms} disagrees with verification.timeout_ms '
                           f'{dev.timeout_ms}: one key decides.')
        for pol in app.policies:
            # A policy step's attached_topic is NOT refused: besides the legacy tree check (which
            # a contract replaces with VerifyGrasp), it is the unchanged policy runtime's own
            # attached-observation input (suction_state_topic), kept as it is (ADR-0022 D10).
            if pol.require_attached and req is False:
                out.append(f'{_prefix(17)}: policy step "{pol.name}": require_attached asks for a '
                           f'grasp check that {self.name} require_grasp_confirmation false does '
                           'not make -- one key decides: set require_attached false (its '
                           'attached_topic stays: the policy runtime reads it).')
        return out

    # R18 -------------------------------------------------------------------------------------
    def r18_bridges(self) -> List[str]:
        out = []
        for row in self.p.deployment.bridges:
            role = bridge_role(row)
            if role and role['role'] == 'eef_legacy_bridge':
                out.append(f'{_prefix(18)}: deployment bridge {row.package}/{row.executable} '
                           f'(modes {row.modes}) is a legacy end-effector bridge ({role["why"]}): '
                           'next to a contract the TrainIt adapter serves the device -- remove '
                           'the row (the joint-state merger stays).')
        return out

    # R20 -------------------------------------------------------------------------------------
    def r20_backends(self) -> List[str]:
        out, dev = [], self.dev
        if dev.kind is GripperKind.COMPOSITE:
            others = [m.value for m in self.modes if m is not Backend.MOCK]
            if others:
                out.append(f'{_prefix(20)}: {self.name} is composite and deploys {others}: a '
                           'composite device runs on mock only in M2.6.')
        b = self._bound(Backend.ISAAC)
        if Backend.ISAAC in self.modes and b is not None and 'set_bool' in b.row['serves']:
            if dev.kind is not GripperKind.SUCTION or len(dev.actuators) != 1:
                out.append(f'{_prefix(20)}: {self.name} on isaac: "{b.adapter}" serves the '
                           'suction vacuum only (one set_bool actuator, kind suction); this '
                           f'device is kind {dev.kind.value} with '
                           f'{[a.id for a in dev.actuators]}.')
        return out

    # R21 -------------------------------------------------------------------------------------
    def r21_interaction(self) -> List[str]:
        if not self.g.interacts_with_object:
            return [f'{_prefix(21)}: {self.name}: interacts_with_object is false -- a contract '
                    'device holds objects (process tools stay on ProcessController).']
        return []

    # R22 -------------------------------------------------------------------------------------
    def _adapter_joints(self) -> set:
        """The joints a composite mock bundle's adapter publishes, and the mimic followers that
        move with them (their state is derived from the master's)."""
        joints = set(_actuator_joints(self.dev))
        if self.urdf is not None:
            for j in self.urdf.joints.values():
                master = mimic_master(self.urdf, j) if j.mimic else None
                if master is not None and master.name in joints:
                    joints.add(j.name)
        return joints

    def _mock_descriptions(self) -> Tuple[List[Tuple[str, UrdfView]], str]:
        """The compiled descriptions the MOCK bring-up's robot_description is made of, the one it
        loads first: on the scene-loader flow the base config's urdf.xacro (compiled with the mock
        mapping), then the project's description (from scratch, the generated urdf.xacro includes
        it) -- or why the first cannot be named or compiled."""
        sources, why = [], ''
        if self.p.robot.base_moveit_config_path:
            view, label, why = bringup_description(self.p)
            if view is not None:
                sources.append((label, view))
        if self.urdf is not None:
            sources.append((f'the description "{self.p.robot.description.top_xacro}"', self.urdf))
        return sources, why

    def r22_composite_bundle(self) -> List[str]:
        """A composite BUNDLE (owner Q1 option A, vNext M2.7): generated for mock only, served by
        trainit_mock alone, with every tool joint's state reaching the robot's joint_states."""
        dev = self.dev
        if dev.kind is not GripperKind.COMPOSITE:
            return []
        out: List[str] = []
        others = [m.value for m in self.modes if m is not Backend.MOCK]
        if others:
            out.append(f'{_prefix(22)}: {self.name} is composite and deploys {others} -- '
                       f'{_UNSUPPORTED}: a composite BUNDLE is generated for mock only (owner '
                       'Q1, option A). Composite bundles on isaac, gazebo or real, and mixed '
                       'bindings (a TrainIt adapter and ros2_control on one device), are vNext M6 '
                       '(ADR-0020).')
        b = dev.bindings.get(Backend.MOCK)
        if Backend.MOCK in self.modes and b is not None and b.adapter != 'trainit_mock':
            out.append(f'{_prefix(22)}: {self.name} is composite and bound to "{b.adapter}" on '
                       f'mock -- {_UNSUPPORTED}: a composite mock bundle is served by '
                       '"trainit_mock" alone (every actuator on its standard interface, every tool '
                       'joint on the robot\'s joint_states).')
        managed = self._adapter_joints()
        sources, why = self._mock_descriptions()
        if why:
            out.append(f'{_prefix(22)}: {self.name} is composite: {why} -- TSA cannot check that '
                       'no <ros2_control> block claims a tool joint, and refuses rather than guess '
                       '(fail closed).')
        claims = [(label, block, [j for j in joints if j in managed])
                  for label, view in sources for block, joints in view.ros2_control.items()]
        if not self.p.robot.base_moveit_config_path:
            claims.append(('the generated config (the planning group\'s block)',
                           f'{self.p.robot.robot_name.upper()}System',
                           [j for j in self.p.robot.planning_group.joints if j in managed]))
        said = set()
        for label, block, hit in claims:
            # the same file compiled twice reports a claim once
            if hit and (block, tuple(hit)) not in said:
                said.add((block, tuple(hit)))
                out.append(f'{_prefix(22)}: {self.name} is composite: tool joint(s) {hit} are in '
                           f'the <ros2_control> block "{block}" of {label}. On mock the '
                           'trainit_mock adapter serves every actuator and publishes every tool '
                           'joint: a ros2_control block that also claims one publishes a second '
                           'state on joint_states and opens a second commander (a mixed binding, '
                           'vNext M6, ADR-0020) -- take the tool joints out of the block.')
        echo = pseudo_joint(self.p, dev, self.urdf)
        if echo is not None:
            # M2.7 final review FI-5: the hardware adapter refuses a compatibility endpoint on a
            # MULTI-site device only; on one site it would publish the echo, but apart from the
            # eef/<id>/joint_states that the composite row remaps onto the robot's joint_states
            refused = (', and the hardware adapter refuses it at start-up on a device with '
                       f'{len(dev.sites)} sites' if len(dev.sites) > 1 else '')
            out.append(f'{_prefix(22)}: {self.name} is composite: command_joint "{echo.name}" is a '
                       'description joint no actuator drives, which the adapter would echo as a '
                       f'compatibility pseudo-joint -- {_UNSUPPORTED}: the echo is a simple '
                       'device\'s compatibility endpoint, published apart from the '
                       f'{adapter_joint_states_topic(dev)} that a composite bundle routes onto the '
                       f'robot\'s joint_states{refused}. In a composite bundle a tool joint is '
                       'an actuator\'s: list it in a set_bool actuator\'s joints with '
                       'joint_positions, or clear command_joint and make the joint fixed.')
        out += self._unpublished_tool_joints(sources, managed, echo)
        return out

    def _unpublished_tool_joints(self, sources, managed: set, echo) -> List[str]:
        """A movable tool joint off every TCP path (R24 judges those) that nothing would publish in
        a composite mock bundle: no actuator's, no mimic follower, in no ros2_control block."""
        urdf, root = self.urdf, self.g.eef_parent_link
        if urdf is None or not root or root not in urdf.links:
            return []
        on_path = set()
        for s in self.dev.sites:                     # R24 / R25 judge the joints on a TCP path
            if s.tcp and s.tcp in urdf.links:
                on_path |= {j.name for j in (tcp_path(urdf, root, s.tcp) or [])}
        claimed = {j for _label, view in sources for joints in view.ros2_control.values()
                   for j in joints} | set(self.p.robot.planning_group.joints)
        out = []
        for j in urdf.joints.values():
            if (j.type == 'fixed' or j.mimic or not urdf.in_subtree(j.parent, root)
                    or j.name in managed or j.name in claimed or j.name in on_path
                    or (echo is not None and j.name == echo.name)):
                continue
            out.append(f'{_prefix(22)}: {self.name} is composite: tool joint "{j.name}" '
                       f'({j.type}) would have no state publisher in a mock bundle -- '
                       f'{_UNSUPPORTED}: it is no actuator\'s joint, no mimic follower of one and '
                       'in no <ros2_control> block, and the current canonical model publishes no '
                       'state for a joint nothing drives, so its child links would have no TF and '
                       'MoveIt\'s current state would stay incomplete: list it in an actuator\'s '
                       'joints, or make it fixed.')
        return out

    # R23 -------------------------------------------------------------------------------------
    def r23_legacy_fields(self) -> List[str]:
        out, g, dev = [], self.g, self.dev
        if g.controller_type != 'GripperCommand':
            out.append(f'{_prefix(23)}: controller_type "{g.controller_type}" next to a contract: '
                       'the contract names each actuator\'s interface -- leave the default.')
        for key, default in (('grasp_action', 'close'), ('release_action', 'open')):
            if getattr(g, key) != default:
                out.append(f'{_prefix(23)}: {key} "{getattr(g, key)}" next to a contract: '
                           'grasp/release are the sites\' engage/release targets -- leave the '
                           'default.')
        out += self._legacy_targets()
        actions = [a.action for a in dev.actuators
                   if a.interface is CommandInterface.GRIPPER_COMMAND and a.action
                   and 'action' not in a.derived]
        if g.controller_name and actions and f'{g.controller_name}/{g.action_ns}' not in actions:
            out.append(f'{_prefix(23)}: controller_name/action_ns '
                       f'"{g.controller_name}/{g.action_ns}" disagree with the contract action'
                       f'{"s" if len(actions) > 1 else ""} {actions}.')
        if dev.kind is GripperKind.COMPOSITE and g.controller_name and not actions:
            out.append(f'{_prefix(23)}: controller_name "{g.controller_name}" next to a composite '
                       'contract with no gripper_command action of that controller.')
        for backend in dev.bindings:
            if backend is Backend.GAZEBO:
                continue                               # authoritative for the Gazebo weld only
            got = g.sim_grasp_adapter.get(backend.value)
            if got is None:
                continue
            allowed = {DEFAULT_SIM_GRASP_ADAPTER[backend]}
            if dev.bindings[backend].adapter == 'none':
                allowed.add(SimGraspAdapter.NONE)
            if got not in allowed:
                out.append(f'{_prefix(23)}: sim_grasp_adapter.{backend.value} "{got.value}" '
                           f'disagrees with the {backend.value} binding '
                           f'"{dev.bindings[backend].adapter}" (expected '
                           f'{sorted(a.value for a in allowed)}).')
        cj = g.command_joint
        if cj and self.urdf is not None and cj not in _actuator_joints(dev):
            j = self.urdf.joints.get(cj)
            if j is None or j.type == 'fixed':
                out.append(f'{_prefix(23)}: command_joint "{cj}" is '
                           f'{"not in the description" if j is None else "a fixed joint"}: a '
                           'canonical command_joint names the non-fixed description joint the '
                           'adapter echoes.')
            elif not self.urdf.has_limits(j):
                out.append(f'{_prefix(23)}: command_joint "{cj}" is a {j.type} joint without '
                           'limits: the pseudo-joint echo publishes its upper/lower limit for '
                           'engaged/released.')
        return out

    # R24 / R25 -------------------------------------------------------------------------------
    def _tcp_path_joints(self):
        """Owner Q7-i: the movable joints on the kinematic path from the device root
        (eef_parent_link) to each site's TCP, in first-seen order, each with the joint that moves
        it (itself, or a mimic follower's master) and the actuator that drives that one (or None).
        An arm joint -- or a follower of one -- is the arm controller's: not listed. Empty when the
        description or eef_parent_link cannot be read (R9 / R5 say it)."""
        urdf, root = self.urdf, self.g.eef_parent_link
        if urdf is None or not root or root not in urdf.links:
            return []
        arm = set(self.p.robot.planning_group.joints)
        owner: Dict[str, DeviceActuator] = {}
        for a in self.dev.actuators:
            for jn in a.joints or []:
                owner.setdefault(jn, a)               # R5 refuses a joint shared by two
        found: Dict[str, Tuple[UrdfJoint, List[str]]] = {}
        for s in self.dev.sites:
            if not s.tcp or s.tcp not in urdf.links:
                continue                              # R9 says it
            for j in tcp_path(urdf, root, s.tcp) or []:
                if j.type != 'fixed' and j.name not in arm:
                    found.setdefault(j.name, (j, []))[1].append(f'site "{s.id}" ("{s.tcp}")')
        out = []
        for name, (j, sites) in found.items():
            master = mimic_master(urdf, j)
            if master is not None and master.name in arm:
                continue                              # the arm controller governs it
            what = f'joint "{name}" ({j.type}' + (f', a mimic follower of "{j.mimic}"'
                                                  if j.mimic else '') + ')'
            where = (f'{self.name}: {what} lies on the kinematic path from eef_parent_link '
                     f'"{root}" to the TCP of {", ".join(sites)}')
            out.append((j, where, owner.get(master.name) if master is not None else None))
        return out

    def r24_path_interfaces(self) -> List[str]:
        """Owner Q7-i E1 (vNext M2.7): an actuator that drives a movable joint on a TCP path is
        position/trajectory controlled -- follow_joint_trajectory, on a single-DOF joint. Command
        interface follows actuator semantics, not the functional label: a jaw OFF the path keeps
        set_bool or gripper_command. G0 (owner section 9): a multi-DOF joint, and a site's grip
        read as a jaw whose TCP sits on its own finger link, are limits of the current canonical
        model and say so; an axis that grips nothing is the owner's E1 and does not."""
        grip_of: Dict[str, str] = {}
        for s in self.dev.sites:
            if s.grip is not None:
                grip_of.setdefault(s.grip, s.id)       # R9: an actuator grips at most one site
        out = []
        for j, where, a in self._tcp_path_joints():
            multi = j.type not in _SINGLE_DOF
            if a is None or (a.interface is CommandInterface.FOLLOW_JOINT_TRAJECTORY and not multi):
                continue
            if multi:
                why = (f'{_UNSUPPORTED}: no actuator interface of the contract drives a {j.type} '
                       'joint, and MoveIt reads a multi-DOF joint from TF, which no adapter '
                       'publishes: make it fixed, or model it as single-DOF joints.')
            elif a.id in grip_of:
                # an axis declared as a grip (EEF-D2, E1), or a jaw whose TCP is on its finger:
                # the same URDF shape, so the sentence gives both readings (R8: FJT never grips)
                moved, site = (mimic_master(self.urdf, j) or j).name, grip_of[a.id]
                why = (f'"{a.id}" grips site "{site}", and in the current canonical model a grip '
                       'never moves a TCP: a prismatic or revolute axis that moves a TCP is '
                       'position/trajectory controlled, even when the application uses only two '
                       'of its positions (owner Q7-i E1), and a follow_joint_trajectory actuator '
                       f'never grips (R8). If "{moved}" is a positioning axis, drive it with a '
                       'follow_joint_trajectory actuator that grips nothing and give site '
                       f'"{site}" a grip of its own (a device with an axis is kind composite, '
                       'R1); if it is the jaw\'s own finger joint (a TCP on a finger link), the '
                       f'configuration is {_UNSUPPORTED}: define the TCP on a link the jaw does '
                       'not move.')
            else:
                why = ('a prismatic or revolute axis that moves a TCP is position/trajectory '
                       'controlled, even when the application uses only two of its positions '
                       '(owner Q7-i E1).')
            out.append(f'{_prefix(24)}: {where}: actuator "{a.id}", interface expected '
                       f'"follow_joint_trajectory"{" on a single-DOF joint" if multi else ""}, '
                       f'interface received "{a.interface.value}" -- {why}')
        return out

    def r25_path_actuated(self) -> List[str]:
        """Owner Q7-i E2 (vNext M2.7): a movable joint on a TCP path is never left without the
        actuator that governs it (a mimic follower: its master). G0 (owner section 9): a
        multi-DOF joint, and the command_joint pseudo-joint above a TCP, are limits of the current
        canonical model and say so; an undriven axis or a broken <mimic> chain does not. A
        follower's remedy names its master (M2.7 final review PY-1: R5 refuses a follower)."""
        echo = pseudo_joint(self.p, self.dev, self.urdf)
        lead = ('a movable joint that moves a TCP needs the actuator that governs it (owner '
                'Q7-i E2)')
        out = []
        for j, where, a in self._tcp_path_joints():
            if a is not None:
                continue
            multi = j.type not in _SINGLE_DOF
            master = mimic_master(self.urdf, j)        # None: a missing or circular master
            if multi:
                why = (f'{_UNSUPPORTED}: {lead}, and no actuator interface of the contract '
                       f'drives a {j.type} joint: make it fixed, or model it as single-DOF '
                       'joints.')
            elif echo is not None and echo.name in (j.name, master.name if master else None):
                why = (f'{_UNSUPPORTED}: "{echo.name}" is the command_joint pseudo-joint (the '
                       'compatibility echo of the engage/release command: engaged at its upper '
                       f'limit, released at its lower), so "{j.name}" would move the TCP with '
                       f'every engage and release: define the TCP on a link above "{j.name}" '
                       f'(for example "{j.parent}"), or make "{j.name}" fixed'
                       + (' and clear command_joint.' if j.name == echo.name else '.'))
            elif j.mimic and master is None:
                why = ('its <mimic> chain does not end at a joint of the description (a missing '
                       'or circular master), so no actuator can govern it (owner Q7-i E2): '
                       'repair the chain, or make it fixed.')
            elif j.mimic:
                why = (f'{lead}, and a mimic follower moves with its master: list its master '
                       f'"{master.name}" in a follow_joint_trajectory actuator\'s joints (a '
                       f'follower is never an actuator joint, R5), or make "{j.name}" fixed.')
            else:
                why = (f'{lead}: list it in a follow_joint_trajectory actuator\'s joints, or '
                       'make it fixed.')
            out.append(f'{_prefix(25)}: {where}: no actuator, interface expected '
                       f'"follow_joint_trajectory"{" on a single-DOF joint" if multi else ""}, '
                       f'interface received none -- {why}')
        return out

    # R26 -------------------------------------------------------------------------------------
    def r26_echo_topic(self) -> List[str]:
        """Owner 11.1 (vNext M2.7): under a cell namespace the pseudo-joint echo is this robot's;
        an absolute (or private) echo topic -- the merger row's gripper input, captured verbatim --
        would be shared by every robot of the cell. Refused, fail closed; legacy is never here."""
        ns = self.p.robot.namespace()
        if not ns:
            return []
        echo = pseudo_joint(self.p, self.dev, self.urdf)
        if echo is None or echo.upper is None or echo.lower is None:
            return []                                 # no echo is emitted (R23 says why)
        out = []
        for backend in self.modes:
            b = self.dev.bindings.get(backend)
            if b is None or not b.row['executable']:
                continue                              # no adapter row on this backend
            topic = pseudo_joint_echo_topic(self.p, backend.value)
            if not topic.startswith(('/', '~')):
                continue
            found = _merger_gripper_row(self.p, backend.value)
            source = (f'the "{found[2]}" of the joint-state merger row {found[1].package}/'
                      f'{found[1].executable}' if found else 'the default')
            rel = _relative_form(ns, topic)
            if topic.startswith('~'):
                why = ('private: it resolves under each node\'s own name, so the adapter and the '
                       'merger would never meet')
            elif topic == ns or topic.startswith(ns + '/'):
                why = 'absolute: no canonical name is absolute (ADR-0022 D7)'
            else:
                why = 'absolute: every robot of the cell would share it'
            out.append(f'{_prefix(26)}: {self.name} ({ns}/eef/{self.dev.id}) on {backend.value}: '
                       f'robot namespace "{ns}", pseudo-joint echo topic "{topic}" ({source}) is '
                       f'{why} -- expected relative form "{rel}" (it resolves to "{ns}/{rel}"); '
                       'make that input relative (owner 11.1, ADR-0022 D7).')
        return out

    def _legacy_targets(self) -> List[str]:
        g, dev = self.g, self.dev
        at_default = (g.joint_target is GripperJointTarget.SRDF_STATE and g.open_state is None
                      and g.closed_state is None and g.open_angle is None
                      and g.closed_angle is None)
        if at_default:
            return []
        jaw = dev.actuators[0] if (dev.short_form and dev.actuators and dev.actuators[0].interface
                                   is CommandInterface.GRIPPER_COMMAND) else None
        if jaw is None:
            return [f'{_prefix(23)}: joint_target/open_state/closed_state/open_angle/closed_angle '
                    'are set next to a contract whose device is not a short-form '
                    'GripperCommand jaw: nothing reads them -- leave them at their defaults.']
        if 'targets' in jaw.derived or jaw.targets is None or not jaw.joints:
            return []                                  # the targets ARE these fields
        from ..model.end_effector import _derived_jaw_targets
        legacy, _why = _derived_jaw_targets(g, jaw.joints[0], self.p.robot.named_states,
                                            base_srdf_states(self.p))
        if legacy is None:
            return []
        bad = [t for t in ('open', 'closed')
               if t in jaw.targets and abs(float(jaw.targets[t]) - legacy[t]) > 1e-9]
        if bad:
            return [f'{_prefix(23)}: {self._a(jaw)}: targets {dict(jaw.targets)} and the legacy '
                    f'target fields ({ {t: legacy[t] for t in bad} }) disagree on {bad}: targets '
                    'wins -- align or clear the legacy fields.']
        return []


def _relative_form(ns: str, topic: str) -> str:
    """The relative spelling of ``topic`` for a node in ``ns``: the namespace prefix and the
    leading '/' or '~/' dropped (/suction/joint_states and /r1/suction/joint_states both become
    suction/joint_states under /r1)."""
    name = topic[2:] if topic.startswith('~/') else topic.lstrip('/')
    prefix = ns.strip('/')
    if prefix and (name == prefix or name.startswith(prefix + '/')):
        name = name[len(prefix):].lstrip('/')
    return name


def _relative_name(name: str) -> bool:
    return bool(name) and not name.startswith(('/', '~')) and all(
        NAME_TOKEN.match(t) for t in name.split('/'))


def _finite(value) -> bool:
    """Every number in ``value`` (a number, or nested dicts / lists / tuples of them) is finite."""
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite(v) for v in value)
    return True


def _per_joint(value, n: int) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return [float(value)] * max(n, 1)
    if isinstance(value, list) and len(value) == n and n > 0:
        return [float(x) for x in value]
    return None


def _card_end_effector(pol) -> str:
    from ..applications.pick_and_place import PickAndPlace
    return PickAndPlace._card_meta(pol)[1]


def _base_controller_flavour(project: CanonicalProject) -> Optional[str]:
    """The flavour of the gripper controller the base config declares (gazebo, ADR-0022 D2.B):
    ``effort`` for an ``effort_controllers/...`` type, ``position`` for ``position_controllers/
    ...``; None when it cannot be read."""
    base = project.robot.base_moveit_config_path
    name = project.robot.gripper.controller_name
    if not base or not name:
        return None
    path = Path(base) / 'config' / 'ros2_controllers.yaml'
    try:
        data = yaml.safe_load(path.read_text()) or {}
    except Exception:                                          # noqa: BLE001
        return None
    for key, body in data.items():
        if not str(key).endswith('controller_manager') or not isinstance(body, dict):
            continue
        ctype = str(((body.get('ros__parameters') or {}).get(name) or {}).get('type', ''))
        if ctype.startswith('effort_controllers/'):
            return ControllerFlavour.EFFORT.value
        if ctype.startswith('position_controllers/'):
            return ControllerFlavour.POSITION.value
    return None


# --- WARNINGS (validate(), never refusals) ------------------------------------------------------

def eef_contract_warnings(project: CanonicalProject) -> List[str]:
    """What a canonical project should hear but may generate with."""
    g = project.robot.gripper
    if g.contract is None:
        return []
    out: List[str] = []
    dev = device(project, with_srdf=False)
    if g.actuation_overridden:
        out.append(f'end-effector contract (ADR-0022 D6): actuation "{g.actuation_overridden}" '
                   'was replaced by "contract" -- a contract commands this tool; the file is '
                   'saved with "contract".')
    canonical_modes = {b.value for b, x in dev.bindings.items()
                       if x.adapter != 'none' and b in deployed(project)}
    for row in project.deployment.bridges:
        role = bridge_role(row)
        if role and role['role'] in ('eef_legacy_bridge', 'gripper_state_merger', 'camera'):
            continue
        active = sorted(set(row.modes) & canonical_modes)
        if active:
            out.append(f'end-effector contract (ADR-0022 R18): bridge {row.package}/'
                       f'{row.executable} runs in {active}, where the contract\'s adapter serves '
                       'the device -- if it serves the gripper too, remove it.')
    stray = [b.value for b in dev.bindings if b not in deployed(project)]
    if stray:
        out.append(f'end-effector contract: bindings for {stray}, which deployment.modes does not '
                   'deploy, are ignored.')
    return out


def isaac_binding(project: CanonicalProject, dev: Optional[Device]) -> Optional[DeviceBinding]:
    """The binding a TrainIt adapter serves on isaac, when isaac is deployed (None otherwise, and
    for ``none``): what makes the stage's channels and the reset this project's to name."""
    b = dev.bindings.get(Backend.ISAAC) if dev is not None else None
    if Backend.ISAAC not in deployed(project) or b is None or b.adapter == 'none':
        return None
    return b


def namespace_isaac_warning(project: CanonicalProject, dev: Device) -> Optional[str]:
    """ADR-0022 D6: a canonical isaac project under a cell namespace -- the stage must bind THIS
    robot's SurfaceGripper by path (the reference stage binds the first one it finds), on the
    robot's own channels: the names are given RESOLVED, as the stage sees them."""
    b = isaac_binding(project, dev)
    ns = project.robot.namespace()
    if not ns or b is None:
        return None
    cmd = project.robot.under_namespace(b.stage_cmd_topic) if b.stage_cmd_topic else None
    reset = project.robot.under_namespace(scene_reset_topic(project))
    return (f'end-effector contract (ADR-0022): cell_namespace "{ns}" on isaac -- the stage must '
            f'bind this robot\'s SurfaceGripper to "{cmd}" by path, and its reset script must '
            f'listen on "{reset}".')


# --- EMISSION ------------------------------------------------------------------------------------

def _per_joint_list(value, n: int) -> List[float]:
    got = _per_joint(value, n)
    return list(got) if got is not None else []


def eef_parameter_block(dev: Device) -> Dict[str, object]:
    """The ``eef`` parameter block of one device (ADR-0022 D6, EEF_CONTRACT_SPEC section 2): what
    TMR's runner and the device's adapter both read. Every value RESOLVED (defaults included),
    maps keyed by id, the order carried by the id arrays; no empty sequence is ever emitted."""
    actuators: Dict[str, object] = {}
    for a in dev.actuators:
        entry: Dict[str, object] = {'command_interface': a.interface.value}
        if a.interface is CommandInterface.SET_BOOL:
            if a.joints:
                entry['joints'] = list(a.joints)
                entry['joint_positions'] = {'engaged': [float(x) for x in a.joint_positions[0]],
                                            'released': [float(x) for x in a.joint_positions[1]]}
            entry['actuation_feedback'] = bool(a.actuation_feedback)
            entry['confirm_timeout_ms'] = int(a.confirm_timeout_ms)
        elif a.interface is CommandInterface.GRIPPER_COMMAND:
            entry['action'] = a.action
            entry['joints'] = list(a.joints or [])
            entry['targets'] = {t: float(v) for t, v in (a.targets or {}).items()}
            entry['result_policy'] = a.result_policy.value if a.result_policy else None
            if a.max_effort is not None:
                entry['max_effort'] = float(a.max_effort)
        else:
            n = len(a.joints or [])
            entry['action'] = a.action
            entry['joints'] = list(a.joints or [])
            entry['targets'] = {t: [float(x) for x in v] for t, v in (a.targets or {}).items()}
            entry['goal_tolerance'] = _per_joint_list(a.goal_tolerance, n)
            if a.path_tolerance is not None:
                entry['path_tolerance'] = _per_joint_list(a.path_tolerance, n)
            entry['duration_s'] = float(a.duration_s) if a.duration_s is not None else None
            entry['goal_time_tolerance_s'] = float(a.goal_time_tolerance_s)
        actuators[a.id] = entry
    body = {
        'actuator_ids': [a.id for a in dev.actuators],
        'site_ids': [s.id for s in dev.sites],
        'command_timeout_ms': int(dev.command_timeout_ms),
        'scene_signal': dev.scene_signal().value,
        'actuators': actuators,
        'sites': {s.id: {'grip': s.grip, 'engage': s.engage, 'release': s.release}
                  for s in dev.sites},
        'verification': {
            'require_grasp_confirmation': bool(dev.require_grasp_confirmation),
            'require_release_confirmation': bool(dev.require_release_confirmation),
            'accepted_evidence': list(dev.accepted_evidence),
            'timeout_ms': int(dev.timeout_ms),
            'max_age_ms': int(dev.max_age_ms),
            'clock_tolerance_ms': int(dev.clock_tolerance_ms),
        },
    }
    return {'ids': [dev.id], dev.id: _prune(body)}


def _prune(value):
    """No None and no empty sequence in a ROS parameter (YAML gives them no type: the parser
    aborts); a refused contract never reaches emission, this only keeps the block well-formed."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            v = _prune(v)
            if v is None or (isinstance(v, (list, dict)) and not v):
                continue
            out[k] = v
        return out
    if isinstance(value, list):
        return [_prune(v) for v in value]
    return value


def yaml_float(value: float) -> str:
    """A float a YAML 1.1 loader (PyYAML, the ROS 2 parameter parser) reads as a double: always a
    '.', and a signed exponent when there is one (``1e-05`` would load as a string)."""
    text = repr(float(value))
    if 'e' in text or 'E' in text:
        mantissa, exp = text.lower().split('e')
        if '.' not in mantissa:
            mantissa += '.0'
        if exp[0] not in '+-':
            exp = '+' + exp
        return f'{mantissa}e{exp}'
    return text if '.' in text or 'inf' in text or 'nan' in text else text + '.0'


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return yaml_float(value)
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


# The plain words a YAML 1.1 loader reads as a bool or a null, in any case (``y``/``n`` included:
# PyYAML skips those two, other 1.1 loaders do not).
_YAML11_WORDS = frozenset(('y', 'n', 'yes', 'no', 'on', 'off', 'true', 'false', 'null', '~'))


def _yaml_key(key) -> str:
    """A mapping key as written: bare when every YAML 1.1 loader reads it back as that very
    string (every key TSA itself names), double-quoted otherwise -- an id or a target named
    ``on`` would otherwise load as the bool True in PyYAML (review TSA-6), and a number-like one
    (a digit, a sign or a dot first: ``1e3`` and ``0o17`` too, numbers in YAML 1.2) as a
    number."""
    text = str(key)
    if text[:1].isalpha() and text.lower() not in _YAML11_WORDS:
        try:
            if yaml.safe_load(f'{text}: 0') == {text: 0}:
                return text
        except yaml.YAMLError:
            pass
    return _yaml_scalar(text)


def render_parameter_block(block: Dict[str, object], indent: int) -> List[str]:
    """``eef:`` and its block as YAML lines at ``indent`` spaces (bt_params' own style: strings
    quoted, inline lists, floats always with a '.'; a key quoted only where a YAML 1.1 loader
    would not read it as itself)."""
    lines = [' ' * indent + 'eef:']

    def walk(node: Dict[str, object], depth: int):
        pad = ' ' * depth
        for key, value in node.items():
            name = _yaml_key(key)
            if isinstance(value, dict):
                lines.append(f'{pad}{name}:')
                walk(value, depth + 2)
            elif isinstance(value, list):
                lines.append(f'{pad}{name}: [' + ', '.join(_yaml_scalar(v) for v in value) + ']')
            else:
                lines.append(f'{pad}{name}: {_yaml_scalar(value)}')
    walk(block, indent + 2)
    return lines


def scene_gripper_cmd_topic(dev: Device) -> str:
    """The scene signal the scene manager reads (ADR-0022 D8): ``eef/<id>/engaged``, relative,
    always written -- for a multi-site device a name no one publishes, so the scene manager's
    absolute default is never reached."""
    return f'eef/{dev.id}/engaged'


def adapter_joint_states_topic(dev: Device) -> str:
    """Where a TrainIt adapter publishes the joints it serves (hardware layer adapter_node:
    ``eef/<id>/joint_states``, relative)."""
    return f'eef/{dev.id}/joint_states'


def relative_scene_reset_topic(project: CanonicalProject, topic: str) -> str:
    """A per-robot, RELATIVE reset topic under a cell namespace (ADR-0022 D6 scene.yaml)."""
    ns = project.robot.namespace().strip('/')
    name = topic.strip('/')
    if ns and (name == ns or name.startswith(ns + '/')):
        name = name[len(ns):].strip('/')
    return name


def scene_reset_topic(project: CanonicalProject) -> str:
    """The reset topic scene.yaml and the isaac adapter row carry. A LEGACY project: the
    project's, unchanged. A canonical one: RELATIVE and per robot under a cell namespace -- and,
    with an isaac binding, ALWAYS written (ADR-0022 D6/D7/D8): unset, the scene manager would
    publish its latched reset on its own ABSOLUTE default in every robot's namespace, and the
    adapter would never record a stage reset as a release."""
    topic = project.scene.scene_reset_topic
    if not is_canonical(project):
        return topic
    # under a cell namespace ALWAYS, isaac or not: the scene manager's own default is an absolute,
    # latched name that every robot of the cell would share (ADR-0022 D7)
    if not topic and (project.robot.namespace()
                      or isaac_binding(project, device(project, with_srdf=False)) is not None):
        topic = SCENE_RESET_TOPIC_DEFAULT
    if topic and project.robot.namespace():
        return (relative_scene_reset_topic(project, topic)
                or relative_scene_reset_topic(project, SCENE_RESET_TOPIC_DEFAULT))
    return topic


def adapter_rows(project: CanonicalProject) -> List[LaunchNodeSpec]:
    """The TrainIt adapter rows a canonical bundle APPENDS to its BRIDGES (ADR-0022 D6 [A]): one
    per deployed backend bound to a row TSA starts, in deployment order. Each carries the device
    id and the SAME ``eef`` block TMR reads (nested: launch_ros flattens it), then the row's own
    parameters. Empty for a legacy project."""
    dev = device(project)
    if dev is None:
        return []
    urdf, _ = description_view(project)
    block = eef_parameter_block(dev)
    echo = pseudo_joint(project, dev, urdf)
    trigger = None
    if project.application.policies:
        trigger, _ = _card_services(project.application.policies[0].card)
    rows: List[LaunchNodeSpec] = []
    for backend in deployed(project):
        b = dev.bindings.get(backend)
        if b is None or not b.row['executable']:
            continue
        own: Dict[str, object] = {}
        if b.adapter == 'trainit_isaac_surface_gripper':
            own['stage_cmd_topic'] = b.stage_cmd_topic
            if b.stage_status_topic:
                own['stage_status_topic'] = b.stage_status_topic
            if dev.timeout_ms < 2000:
                own['engage_timeout_ms'] = int(dev.timeout_ms)
            if dev.max_age_ms < 400:
                own['status_timeout_ms'] = int(dev.max_age_ms)
            reset = scene_reset_topic(project)
            if reset:
                own['scene_reset_topic'] = reset
        if (b.adapter == 'trainit_mock' and dev.max_age_ms > 0
                and _MOCK_SAMPLE_RATE_HZ * dev.max_age_ms < 1000):
            # the mock refuses to start unless sample_rate_hz * max_age_ms >= 1000 (CROSS-3)
            own['mock.sample_rate_hz'] = max(_MOCK_SAMPLE_RATE_HZ,
                                             float(math.ceil(1000 / dev.max_age_ms)))
        if (b.row['evidence_configurable']
                and b.row['default_evidence'] not in dev.accepted_evidence):
            chosen = next((e for e in dev.accepted_evidence if e in b.row['evidence']), None)
            if chosen:
                for s in dev.sites:
                    own[f'mock.sites.{s.id}.evidence'] = chosen
        if trigger is not None:
            own['compat.trigger_on'], own['compat.trigger_off'] = trigger
        if echo is not None and echo.upper is not None and echo.lower is not None:
            own['compat.pseudo_joint_topic'] = pseudo_joint_echo_topic(project, backend.value)
            own['compat.pseudo_joint_name'] = echo.name
            own['compat.pseudo_joint_engaged'] = float(echo.upper)
            own['compat.pseudo_joint_released'] = float(echo.lower)
        params: List[Dict[str, object]] = [{'device_id': dev.id, 'eef': block}]
        if own:
            params.append(own)
        # a composite mock bundle: the tool joints onto the robot's joint_states (owner Q1.2)
        remaps = ([[adapter_joint_states_topic(dev), ROBOT_JOINT_STATES]]
                  if dev.kind is GripperKind.COMPOSITE and b.adapter == 'trainit_mock' else [])
        rows.append(LaunchNodeSpec(package=b.row['package'], executable=b.row['executable'],
                                   name=None, parameters=params, remappings=remaps,
                                   modes=[backend.value]))
    return rows


_PACKAGE_URI = re.compile(r'package://([A-Za-z0-9_]+)/')


def scene_package_references(texts: Sequence[str]) -> List[str]:
    """[D] Every package the given generated files load from by ``package://`` (scene.yaml's
    meshes), in first-seen order. A CANONICAL package declares each one where it is used
    (review TSA-1, T11 K1): connector dependencies otherwise come from the deployment's bridge
    rows only, and the opt-in removes the legacy end-effector bridge row that may have been the
    only one declaring the mesh package. A legacy package keeps its dependencies as they were."""
    seen: List[str] = []
    for text in texts:
        for name in _PACKAGE_URI.findall(text):
            if name not in seen:
                seen.append(name)
    return seen


def bridges_with_adapters(project: CanonicalProject) -> List[dict]:
    """The bring-up's BRIDGES: the deployment's rows, then the adapter rows (canonical only)."""
    return [b.dict() for b in project.deployment.bridges] + [
        r.dict() for r in adapter_rows(project)]


def adapter_packages(project: CanonicalProject) -> Sequence[str]:
    return tuple(dict.fromkeys(r.package for r in adapter_rows(project)))


__all__ = ['HARDWARE_LAYER', 'EEF_ADAPTER_ROWS', 'ROBOT_JOINT_STATES',
           'SCENE_RESET_TOPIC_DEFAULT', 'UrdfView', 'adapter_joint_states_topic', 'adapter_rows',
           'bridges_with_adapters', 'bringup_description', 'description_view', 'device',
           'eef_contract_problems', 'eef_contract_warnings', 'mimic_master',
           'pseudo_joint_echo_topic', 'tcp_path',
           'eef_parameter_block', 'is_canonical', 'isaac_binding', 'legacy_problems', 'moved_by',
           'namespace_isaac_warning', 'refuse_eef_contract', 'render_parameter_block',
           'scene_gripper_cmd_topic', 'scene_package_references', 'scene_reset_topic',
           'yaml_float']
