"""The end-effector contract's test fixture (ADR-0022 "The R1/R2-like synthetic fixture"), TSA's
own copy -- generic names, synthetic values, nothing read from another repository.

Projects, all from-scratch (no base config), each on its own synthetic description written into a
test's temp dir from ``data/m2_6_eef``:

  composite(p, ns)      the composite device ``composite_tool``: 6 actuators on 3 interfaces, 2
                        sites; robot_a in /r1 (prefix robot_a_), robot_b in /r2 (prefix robot_b_,
                        IDENTICAL ids), the root variant (no prefix). Mock only; its BUNDLE is
                        generated for mock (R22 narrowed, vNext M2.7: no bypass any more).
  suction_simple(...)   the 3-line short form (kind suction, the description pseudo-joint)
  jaw_simple(...)       a GripperCommand jaw short form (a driven joint plus a mimic follower)
  jaw_binary(...)       a SetBool pneumatic jaw short form
  composite_mock(...)   vNext M2.7 (owner section 13): the section-13 tool, on its own
                        description from ``data/m2_7_composite``: two FJT axes, a GripperCommand
                        gripper and a SetBool jaw (one site each), a SetBool lock; cm_root at the
                        root, robot_a in /r1, robot_b in /r2. Launch-safe by default.

Launch-safe (vNext M2.7). A from-scratch fixture project GENERATES, but its bundle does
not LAUNCH: the generated ros2_control xacro reads ``initial_positions['<arm joint>']`` (empty:
xacro fails) and a null ``joint_limits`` is refused by launch_ros. ``set_launch_safe(project)``
sets exactly three settings on the arm joints: the initial position 0.0, the velocity 1.0 /
acceleration 2.0 limits (Pilz, the default planner, needs acceleration limits) and a
``joints(...)`` OMPL projection. The M2.6 builders take it as an OPT-IN (``launch_safe=True``, for
the launch harnesses): by default their projects, and so every generated file compared byte for
byte elsewhere, are unchanged.

Not collected by pytest (the name is not test_*.py).
"""
import copy
import functools
import hashlib
import importlib.util
import os
import shutil
from pathlib import Path

import pytest

from trainit_setup_assistant.model import (AppType, Backend, BundleSpec, CanonicalProject,
                                           DeploymentSpec, MotionSegment, ToolAction, Waypoint)
from trainit_setup_assistant.model.enums import ToolActionKind, WaypointType
from trainit_setup_assistant.model.robot import (ArmControllerSpec, DescriptionSource,
                                                 GripperSpec, JointLimit, PlanningGroupSpec,
                                                 RobotSpec)

DATA = Path(__file__).resolve().parent / 'data' / 'm2_6_eef'
# Byte copies of the contract files shared by TSA, TMR and the hardware layer (ADR-0022 D8,
# EEF_CONTRACT_SPEC.md): every repository pins the same bytes.
PINNED = {
    'eef_conformance.yaml': '873a1d582c4e3947a62026d153bd49ccb02d7b4d38e5cb3a5b4b3a8f5a27a04f',
    'eef_conformance.expected.txt':
        '1f6bd1d2ac2fb18865ee67a815877483537d0a325fbc9a5400cf2977710d242c',
    'eef_contract_tokens.txt': '55048ea44d466979f57c28a634295a060a79b837fe614a30c49f60572a2bd5de',
    'normalize_eef.py': '9fba00c3981762974d9e916464fd670c8f83964392649c3e0854a94149290a0b',
}
# the conformance fixture's devices, in its order
CONFORMANCE_IDS = ('composite_tool', 'suction_simple', 'jaw_simple', 'jaw_binary')

# The golden-dependent M2.6 gates (review TSA-4, tests lens): a gate that needs the demo repository
# (fr3wml_digital_twin -- the golden, its base config, the owner's test4 project) SKIPS on a
# checkout without it, and FAILS instead with TRAINIT_REQUIRE_GOLDEN=1: a CI job that must run the
# gates cannot report them green unrun.
REQUIRE_GOLDEN = os.environ.get('TRAINIT_REQUIRE_GOLDEN') == '1'


def skip_or_fail(reason: str) -> None:
    """Skip the running test -- or fail it, naming why, with TRAINIT_REQUIRE_GOLDEN=1."""
    if REQUIRE_GOLDEN:
        pytest.fail(f'{reason} -- and TRAINIT_REQUIRE_GOLDEN=1: this gate must run',
                    pytrace=False)
    pytest.skip(reason)


def needs(*paths, reason: str):
    """A test decorator: the test runs when every path exists; otherwise it skips (fails with
    TRAINIT_REQUIRE_GOLDEN=1), naming the first missing path."""
    def decorate(test):
        @functools.wraps(test)
        def guarded(*args, **kwargs):
            missing = [str(p) for p in paths if not Path(p).exists()]
            if missing:
                skip_or_fail(f'{reason}: {missing[0]} is not here')
            return test(*args, **kwargs)
        return guarded
    return decorate


def sha256(name: str) -> str:
    return hashlib.sha256((DATA / name).read_bytes()).hexdigest()


def normalizer():
    """The pinned ``normalize_eef.py`` (the parameter block's schema in executable form)."""
    spec = importlib.util.spec_from_file_location('normalize_eef_pinned', DATA / 'normalize_eef.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tokens() -> dict:
    """``{group: {name: value}}`` of the pinned token file, in file order."""
    out: dict = {}
    for line in (DATA / 'eef_contract_tokens.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        group, name, value = line.split()
        out.setdefault(group, {})[name] = value
    return out


# --- descriptions ---------------------------------------------------------------------------------

def description(root: Path, name: str, macro: str, call: str,
                data: Path = None) -> DescriptionSource:
    """A synthetic description directory: the fixture macro file plus a top xacro calling it."""
    d = root / f'{name}_description_src' / 'urdf'
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy((data or DATA) / macro, d / macro)
    (d / f'{name}.urdf.xacro').write_text(
        '<?xml version="1.0"?>\n'
        f'<robot name="{name}" xmlns:xacro="http://www.ros.org/wiki/xacro">\n'
        f'  <xacro:include filename="{macro}"/>\n'
        f'  {call}\n'
        '</robot>\n')
    return DescriptionSource(urdf_dir=str(d), top_xacro=f'{name}.urdf.xacro')


def _project(root: Path, name: str, desc: DescriptionSource, p: str, gripper: GripperSpec,
             ns: str = '', modes=('mock',), launch_safe: bool = False) -> CanonicalProject:
    project = CanonicalProject(
        project_name=name, bundle=BundleSpec.from_prefix(name),
        robot=RobotSpec(robot_name=name, description=desc, cell_namespace=ns,
                        base_frame=f'{p}base_link', tip_link=f'{p}tool0',
                        planning_group=PlanningGroupSpec(name='arm', base_link=f'{p}base_link',
                                                         tip_link=f'{p}tool0',
                                                         joints=[f'{p}joint1']),
                        gripper=gripper, arm_controller=ArmControllerSpec()),
        deployment=DeploymentSpec(modes=[Backend(m) for m in modes],
                                  default_mode=Backend(modes[0])))
    app = project.application
    app.type = AppType.PICK_AND_PLACE
    app.waypoints = [Waypoint(name=w, type=WaypointType.JOINT, joints=[v])
                     for w, v in (('home', 0.0), ('pick', 0.4), ('flip', 0.6), ('place', 0.9))]
    app.segments = [MotionSegment(to_waypoint=w.name) for w in app.waypoints]
    app.sequence = ['home', 'pick', 'flip', 'place', 'home']
    return set_launch_safe(project) if launch_safe else project


def _actions(rows):
    return [ToolAction(at_waypoint=w, kind=ToolActionKind(k), **kw) for w, k, kw in rows]


def set_launch_safe(project: CanonicalProject) -> CanonicalProject:
    """The project with the three launch-safe settings (module docstring) on every joint of its
    planning group -- the values the vNext M2.7 BEFORE launches used."""
    joints = list(project.robot.planning_group.joints)
    project.robot.initial_positions = {j: 0.0 for j in joints}
    project.robot.joint_limits.per_joint = {j: JointLimit(
        has_velocity_limits=True, max_velocity=1.0, has_acceleration_limits=True,
        max_acceleration=2.0) for j in joints}
    project.robot.ompl.projection_evaluator = f'joints({",".join(joints)})'
    return project


# --- the composite device -------------------------------------------------------------------------

def composite_contract(p: str) -> dict:
    """The device ``composite_tool`` (ADR-0022 fixture table), in TSA's project form."""
    return {
        'id': 'composite_tool',
        'actuators': {
            'linear_axis': {'command_interface': 'follow_joint_trajectory',
                            'action': 'linear_axis_controller/follow_joint_trajectory',
                            'joints': [f'{p}linear_axis_joint'],
                            'targets': {'home': [0.0], 'extended': [0.08]},
                            'goal_tolerance': [0.002], 'duration_s': 1.0,
                            'goal_time_tolerance_s': 0.5},
            'rotary_axis': {'command_interface': 'follow_joint_trajectory',
                            'action': 'rotary_axis_controller/follow_joint_trajectory',
                            'joints': [f'{p}rotary_axis_joint'],
                            'targets': {'receive': [0.0], 'transfer': [0.9]},
                            'goal_tolerance': [0.01], 'path_tolerance': [0.2], 'duration_s': 1.0,
                            'goal_time_tolerance_s': 0.5},
            'wrist_mechanism': {'command_interface': 'follow_joint_trajectory',
                                'action': 'wrist_mechanism_controller/follow_joint_trajectory',
                                'joints': [f'{p}wrist_pitch_joint', f'{p}wrist_roll_joint'],
                                'targets': {'stow': [0.0, 0.0], 'present': [0.3, -0.3]},
                                'goal_tolerance': [0.01, 0.01], 'duration_s': 1.5,
                                'goal_time_tolerance_s': 0.5},
            'gripper_a': {'command_interface': 'gripper_command',
                          'action': 'gripper_a_controller/gripper_cmd',
                          'joints': [f'{p}jaw_a_joint'],
                          'targets': {'open': 0.02, 'closed': 0.0},
                          'result_policy': 'succeeded_or_stalled', 'max_effort': 20.0},
            'gripper_b': {'command_interface': 'set_bool', 'joints': [f'{p}jaw_b_joint'],
                          'joint_positions': {'engaged': [0.0], 'released': [0.02]}},
            'binary_lock': {'command_interface': 'set_bool', 'actuation_feedback': True,
                            'confirm_timeout_ms': 2000},
        },
        'sites': {'site_a': {'tcp': f'{p}site_a_tcp', 'grip': 'gripper_a'},
                  'site_b': {'tcp': f'{p}site_b_tcp', 'grip': 'gripper_b'}},
        'verification': {'require_grasp_confirmation': True,
                         'require_release_confirmation': True, 'accepted_evidence': ['sensor']},
        'adapters': {'mock': 'trainit_mock'},
    }


# The composite handoff of ADR-0022 (judge_tsa_yaml_composite): declared order = tree order.
COMPOSITE_ACTIONS = [
    ('pick', 'actuate', {'actuator': 'rotary_axis', 'target': 'receive'}),
    ('pick', 'actuate', {'actuator': 'binary_lock', 'target': 'engage'}),
    ('pick', 'grasp', {'site': 'site_a'}),
    ('flip', 'actuate', {'actuator': 'rotary_axis', 'target': 'transfer'}),
    ('flip', 'actuate', {'actuator': 'linear_axis', 'target': 'extended'}),
    ('flip', 'actuate', {'actuator': 'wrist_mechanism', 'target': 'present'}),
    ('flip', 'grasp', {'site': 'site_b'}),
    ('flip', 'release', {'site': 'site_a'}),
    ('flip', 'actuate', {'actuator': 'linear_axis', 'target': 'home'}),
    ('place', 'release', {'site': 'site_b'}),
    ('place', 'actuate', {'actuator': 'binary_lock', 'target': 'release'}),
]


def composite(root: Path, robot: str = 'robot_a', ns: str = '/r1', prefix: str = None,
              modes=('mock',), contract: dict = None,
              launch_safe: bool = False) -> CanonicalProject:
    p = f'{robot}_' if prefix is None else prefix
    desc = description(root, robot, 'composite_cell.xacro', f'<xacro:composite_cell p="{p}"/>')
    gripper = GripperSpec(kind='composite', eef_group_name='composite_tool',
                          eef_parent_link=f'{p}tool0',
                          contract=composite_contract(p) if contract is None else contract)
    project = _project(root, robot, desc, p, gripper, ns, modes, launch_safe=launch_safe)
    project.application.tool_actions = _actions(COMPOSITE_ACTIONS)
    return project


# --- the short-form devices -----------------------------------------------------------------------

SIMPLE_ACTIONS = [('pick', 'grasp', {}), ('place', 'release', {})]


def _simple(root: Path, name: str, macro_call: str, gripper: dict, ns: str, modes,
            prefix: str = '', launch_safe: bool = False) -> CanonicalProject:
    desc = description(root, name, 'simple_tools.xacro', macro_call)
    project = _project(root, name, desc, prefix, GripperSpec(**gripper), ns, modes,
                       launch_safe=launch_safe)
    project.application.tool_actions = _actions(SIMPLE_ACTIONS)
    return project


def suction_contract(device_id='suction_simple', adapters=None, grasp=False, release=False):
    """The simple suction: three typed lines under ``contract:`` (plus the id the conformance
    fixture names its device by)."""
    return {'id': device_id, 'actuator': {'command_interface': 'set_bool'},
            'verification': {'require_grasp_confirmation': grasp,
                             'require_release_confirmation': release},
            'adapters': adapters if adapters is not None else {'mock': 'trainit_mock'}}


def suction_simple(root: Path, ns: str = '/r3', modes=('mock',), contract: dict = None,
                   name: str = 'suction_cell', launch_safe: bool = False,
                   **gripper) -> CanonicalProject:
    g = dict(kind='suction', eef_group_name='SuctionCup', eef_name='suction',
             eef_parent_link='tcp', command_joint='suctioncup_joint',
             contract=suction_contract() if contract is None else contract)
    g.update(gripper)
    return _simple(root, name, '<xacro:suction_tool p=""/>', g, ns, modes,
                   launch_safe=launch_safe)


def jaw_contract(device_id='jaw_simple', adapters=None):
    return {'id': device_id,
            'actuator': {'command_interface': 'gripper_command',
                         'targets': {'open': 0.0, 'closed': 0.7},
                         'result_policy': 'succeeded_or_stalled'},
            'verification': {'require_grasp_confirmation': False,
                             'require_release_confirmation': False},
            'adapters': adapters if adapters is not None else {'mock': 'trainit_mock'}}


def jaw_simple(root: Path, ns: str = '', modes=('mock',), contract: dict = None,
               name: str = 'jaw_cell', launch_safe: bool = False, **gripper) -> CanonicalProject:
    g = dict(kind='parallel', eef_group_name='gripper', eef_name='gripper',
             eef_parent_link='tool0', command_joint='finger_joint',
             controller_name='gripper_controller', action_ns='gripper_cmd',
             contract=jaw_contract() if contract is None else contract)
    g.update(gripper)
    return _simple(root, name, '<xacro:jaw_tool p=""/>', g, ns, modes, launch_safe=launch_safe)


def binary_jaw_contract(device_id='jaw_binary', adapters=None):
    return {'id': device_id,
            'actuator': {'command_interface': 'set_bool', 'joints': ['jaw_joint'],
                         'joint_positions': {'engaged': [0.0], 'released': [0.02]}},
            'verification': {'require_grasp_confirmation': False,
                             'require_release_confirmation': False},
            'adapters': adapters if adapters is not None else {'mock': 'trainit_mock'}}


def jaw_binary(root: Path, ns: str = '', modes=('mock',), contract: dict = None,
               name: str = 'binary_jaw_cell', launch_safe: bool = False,
               **gripper) -> CanonicalProject:
    g = dict(kind='parallel', eef_group_name='gripper', eef_name='gripper',
             eef_parent_link='tool0', command_joint='jaw_joint',
             contract=binary_jaw_contract() if contract is None else contract)
    g.update(gripper)
    return _simple(root, name, '<xacro:binary_jaw_tool p=""/>', g, ns, modes,
                   launch_safe=launch_safe)


def dump(model) -> dict:
    """A model as a dict, on pydantic 1.x (``.dict()``) and 2.x (``.model_dump()``) alike: the
    M2.6 tests run on both, as the product does (review TSA-4)."""
    return model.model_dump() if hasattr(model, 'model_dump') else model.dict()


def with_contract(project: CanonicalProject, **changes) -> CanonicalProject:
    """The project with its contract re-validated after ``changes`` (dotted keys allowed:
    ``{'verification.require_grasp_confirmation': True}``), the way a project.yaml parses."""
    data = dump(project.robot.gripper)
    contract = copy.deepcopy(_written(project.robot.gripper.contract))
    for key, value in changes.items():
        node = contract
        parts = key.split('.')
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        if value is _DELETE:
            node.pop(parts[-1], None)
        else:
            node[parts[-1]] = value
    data['contract'] = contract
    data.pop('actuation_overridden', None)
    project.robot.gripper = GripperSpec(**data)
    return project


_DELETE = object()
DELETE = _DELETE


def _written(contract) -> dict:
    from trainit_setup_assistant.model.io import contract_as_written
    return contract_as_written(contract)


# --- the golden, opted in (ADR-0022 D6 "Opting an existing project in"; temp dirs only) ---------

def opt_in(project: CanonicalProject, adapters=None, grasp=False, release=False,
           device_id='main') -> CanonicalProject:
    """The opt-in edits of the golden-shaped suction cell: the 3-line contract, and the legacy
    end-effector bridge rows removed (the catalog marks them, R18; the merger row stays)."""
    from trainit_setup_assistant.generator.eef_contract import bridge_role
    data = dump(project.robot.gripper)
    data.pop('actuation', None)                      # the golden's file carries no token
    data['contract'] = {
        'actuator': {'command_interface': 'set_bool'},
        'verification': {'require_grasp_confirmation': grasp,
                         'require_release_confirmation': release},
        'adapters': (adapters if adapters is not None else
                     {'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper',
                      'real': 'none'})}
    if device_id != 'main':
        data['contract']['id'] = device_id
    project.robot.gripper = GripperSpec(**data)
    project.deployment.bridges = [
        b for b in project.deployment.bridges
        if (bridge_role(b) or {}).get('role') != 'eef_legacy_bridge']
    return project


# --- the section-13 composite MOCK fixture (vNext M2.7, owner section 13): TSA-only test data -----

DATA_M27 = Path(__file__).resolve().parent / 'data' / 'm2_7_composite'


def composite_mock_contract(p: str) -> dict:
    """The owner's section-13 tool: two FJT axes, a GripperCommand gripper and a SetBool jaw (one
    site each: the functional label never decides the interface) and a SetBool lock."""
    return {
        'id': 'composite_tool',
        'actuators': {
            'slide_axis': {'command_interface': 'follow_joint_trajectory',
                           'action': 'slide_axis_controller/follow_joint_trajectory',
                           'joints': [f'{p}slide_joint'],
                           'targets': {'home': [0.0], 'extended': [0.08]},
                           'goal_tolerance': [0.002], 'duration_s': 1.0,
                           'goal_time_tolerance_s': 0.5},
            'rotation_axis': {'command_interface': 'follow_joint_trajectory',
                              'action': 'rotation_axis_controller/follow_joint_trajectory',
                              'joints': [f'{p}rotation_joint'],
                              'targets': {'home': [0.0], 'flipped': [1.5]},
                              'goal_tolerance': [0.01], 'path_tolerance': [0.2],
                              'duration_s': 1.0, 'goal_time_tolerance_s': 0.5},
            'controlled_gripper': {'command_interface': 'gripper_command',
                                   'action': 'controlled_gripper_controller/gripper_cmd',
                                   'joints': [f'{p}gripper_jaw_joint'],
                                   'targets': {'open': 0.02, 'closed': 0.0},
                                   'result_policy': 'succeeded_or_stalled', 'max_effort': 20.0},
            'binary_lock': {'command_interface': 'set_bool', 'actuation_feedback': True,
                            'confirm_timeout_ms': 2000},
            'binary_jaw': {'command_interface': 'set_bool', 'joints': [f'{p}binary_jaw_joint'],
                           'joint_positions': {'engaged': [0.0], 'released': [0.02]}},
        },
        'sites': {'site_a': {'tcp': f'{p}site_a_tcp', 'grip': 'controlled_gripper'},
                  'site_b': {'tcp': f'{p}site_b_tcp', 'grip': 'binary_jaw'}},
        'verification': {'require_grasp_confirmation': True,
                         'require_release_confirmation': True, 'accepted_evidence': ['sensor']},
        'adapters': {'mock': 'trainit_mock'},
    }


COMPOSITE_MOCK_ACTIONS = [
    ('pick', 'actuate', {'actuator': 'binary_lock', 'target': 'engage'}),
    ('pick', 'actuate', {'actuator': 'slide_axis', 'target': 'extended'}),
    ('pick', 'grasp', {'site': 'site_a'}),
    ('pick', 'actuate', {'actuator': 'slide_axis', 'target': 'home'}),
    ('flip', 'actuate', {'actuator': 'rotation_axis', 'target': 'flipped'}),
    ('flip', 'grasp', {'site': 'site_b'}),
    ('place', 'release', {'site': 'site_a'}),
    ('place', 'release', {'site': 'site_b'}),
    ('place', 'actuate', {'actuator': 'rotation_axis', 'target': 'home'}),
    ('place', 'actuate', {'actuator': 'binary_lock', 'target': 'release'}),
]


def composite_mock(root: Path, robot: str = 'robot_a', ns: str = '/r1', prefix: str = None,
                   modes=('mock',), contract: dict = None,
                   launch_safe: bool = True) -> CanonicalProject:
    """The section-13 tool on the 1-DOF arm stub, from scratch, LAUNCHABLE on mock: launch-safe
    by default (module docstring; an empty initial_positions.yaml does not compile, a null
    joint_limits does not launch)."""
    p = f'{robot}_' if prefix is None else prefix
    desc = description(root, robot, 'composite_mock_cell.xacro',
                       f'<xacro:composite_mock_cell p="{p}"/>', data=DATA_M27)
    gripper = GripperSpec(kind='composite', eef_group_name='composite_tool',
                          eef_parent_link=f'{p}tool0',
                          contract=composite_mock_contract(p) if contract is None else contract)
    project = _project(root, robot, desc, p, gripper, ns, modes, launch_safe=launch_safe)
    project.application.tool_actions = _actions(COMPOSITE_MOCK_ACTIONS)
    return project
