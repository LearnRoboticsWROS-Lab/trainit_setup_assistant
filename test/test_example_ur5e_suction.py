"""The public sample cell, ``examples/ur5e_suction`` (vNext G0 community follow-up, owner P2): a
Universal Robots UR5e from ur_description with a simple suction tool, running a deterministic
pick-and-place on mode:=mock -- the cell a user with no robot of their own generates and runs
(the sample's README.md).

    the committed sample is exactly its four files, names no machine path and holds no
      package.xml (colcon and rosdep never see it), and TSA depends on no part of it: package.xml,
      code, templates and catalog name neither the sample nor ur_description, and setup.py
      installs none of it (an example never becomes a framework dependency)
    the project is the end-effector contract's three-line short form (ADR-0022) on mock, at the
      root, with no description pseudo-joint
    the held box clears the table and the cup (arithmetic on the project), and the named states
      put the TCP straight above the targets with the tool down (FK of the committed description)
    TSA refuses nothing and warns nothing; generated the README's way -- FROM the sample's
      directory, its paths relative -- it writes exactly the README's 26 files, names no machine
      path, records its copied files as package://trainit_setup_assistant/examples/ur5e_suction/...
      and emits the tree, bt_params, SRDF and controllers the project describes
    the bring-up's robot_description -- the generated wrapper, compiled the way the mock bring-up's
      MoveItConfigsBuilder compiles it -- holds exactly ONE ros2_control block:
      mock_components/GenericSystem on the six arm joints, at the project's initial positions
    trainit_generate, the README's command, prints what the README says; the commands the wizard
      prints for this bundle name its launch files, with arguments those files declare

ur_description is a prerequisite of the SAMPLE, never of TSA (no package.xml entry, not even a
test_depend): a test that compiles the description SKIPS without it (or without xacro and a
sourced environment), naming the apt package -- and with TRAINIT_REQUIRE_GOLDEN=1 it FAILS
instead (eef_fixture.skip_or_fail: a fresh machine's Community dependency closure lacks it).
Generation is test-time, into a temp dir removed at teardown (owner rule 2026-10-06); a user's own
generation is the MANUAL TSA GENERATION CHECKPOINT. The launched half is
test_example_ur5e_suction_launch.py (opt-in). No ROS graph.
"""
import ast
import glob
import math
import os
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

PKG = Path(__file__).resolve().parents[1]
EXAMPLE = PKG / 'examples' / 'ur5e_suction'
URDF_DIR = EXAMPLE / 'urdf'
TOP, TOOL = 'ur5e_suction.urdf.xacro', 'suction_tool.xacro'
ROBOT = 'ur5e_suction'
DESC, CFG, APP = f'{ROBOT}_description', f'{ROBOT}_trainit_config', f'{ROBOT}_app'
ARM = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint',
       'wrist_2_joint', 'wrist_3_joint']
SEQUENCE = ['home', 'pre_pick', 'pick', 'post_pick', 'pre_place', 'place', 'post_place', 'home']
# Every file generation writes for the sample, read from the emitters (the manifest lists them).
FILES = sorted(
    [f'{DESC}/{n}' for n in ('CMakeLists.txt', 'package.xml', 'launch/view_robot.launch.py',
                             'rviz/view_robot.rviz', f'urdf/{TOOL}', f'urdf/{TOP}')]
    + [f'{CFG}/{n}' for n in ('CMakeLists.txt', 'package.xml')]
    + [f'{CFG}/config/{n}' for n in (
        'initial_positions.yaml', 'joint_limits.yaml', 'kinematics.yaml', 'moveit.rviz',
        'moveit_controllers.yaml', 'ompl_planning.yaml', 'pilz_cartesian_limits.yaml',
        'ros2_controllers.yaml', f'{ROBOT}.ros2_control.xacro', f'{ROBOT}.srdf',
        f'{ROBOT}.urdf.xacro')]
    + [f'{APP}/{n}' for n in ('CMakeLists.txt', 'package.xml', 'bt_trees/pick_place.xml',
                              'config/bt_params.yaml', f'groot2/{ROBOT}_pick_and_place.btproj',
                              'launch/bringup.launch.py', 'launch/trainit_bt.launch.py')])
# Paths of a machine: never in a committed or a generated file.
MACHINE = ('/home/', '/root/', '/tmp/', '/media/', '/mnt/', '/Users/')
# What the mock bring-up hands xacro (TMR build_move_group_params(use_isaac=False)).
MOCK_MAPPINGS = {'use_isaac': 'false'}
# The orientation (qx, qy, qz, qw) = (1, 0, 0, 0): pi about x, the tool pointing down.
DOWN = np.diag([1.0, -1.0, -1.0])


def _machine_paths(*extra) -> tuple:
    # the home as a directory prefix: as root it is /root, and every BT tree ends in '</root>'
    home = str(Path.home()).rstrip('/')
    return MACHINE + tuple(extra) + ((home + '/',) if home else ())


def needs_ur_description() -> Path:
    """ur_description's share directory -- or SKIP, naming the apt package (FAIL instead with
    TRAINIT_REQUIRE_GOLDEN=1). The sample needs it, and xacro, wherever its description is
    compiled or its bundle generated (test_example_ur5e_suction_launch.py asks here too); TSA
    itself never does."""
    try:
        import xacro  # noqa: F401 -- TSA compiles the description in process
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory('ur_description'))
    except Exception:                       # noqa: BLE001 -- not sourced, not installed, no xacro
        import eef_fixture
        eef_fixture.skip_or_fail('the public sample needs ur_description and xacro in a sourced '
                                 'ROS 2 environment (sudo apt install ros-humble-ur-description)')


def _load():
    """The sample's project, loaded the README's way: from its directory, by a relative path."""
    from trainit_setup_assistant.model import load_project
    return load_project('project.yaml')


@pytest.fixture
def project(monkeypatch):
    # TSA resolves the sample's relative paths against the current directory (the README's rule),
    # so every test that reads the description runs from the sample's directory.
    monkeypatch.chdir(EXAMPLE)
    return _load()


@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    """The sample generated once, the README's way, into a temp dir removed at teardown."""
    needs_ur_description()
    from trainit_setup_assistant.generator import Orchestrator
    tmp = tmp_path_factory.mktemp('ur5e_suction_sample')
    root = tmp / ROBOT
    try:
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(EXAMPLE)
            project = _load()
            manifest = Orchestrator().generate(project, root)
        yield dict(root=root, project=project, manifest=manifest)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _compile(path: Path, mappings=None) -> ET.Element:
    """A description compiled by TSA's own in-process xacro (a fresh mapping dict per call: xacro
    writes the defaults it applied into the dict it is given)."""
    from trainit_setup_assistant.robotmodel.xacro_loader import compile_xacro
    return ET.fromstring(compile_xacro(path, dict(mappings or {})))


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y),
                              math.sin(y))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def _fk(urdf: ET.Element, q: dict, link: str) -> np.ndarray:
    """The 4x4 pose of ``link`` in the URDF's root frame at the joint values ``q`` (fixed and
    revolute joints: all this description has)."""
    joint_of = {j.find('child').get('link'): j for j in urdf.findall('joint')}
    chain = []
    while link in joint_of:
        chain.append(joint_of[link])
        link = joint_of[link].find('parent').get('link')
    pose = np.eye(4)
    for joint in reversed(chain):
        origin = joint.find('origin')
        step = np.eye(4)
        if origin is not None:
            step[:3, :3] = _rpy(*[float(v) for v in origin.get('rpy', '0 0 0').split()])
            step[:3, 3] = [float(v) for v in origin.get('xyz', '0 0 0').split()]
        pose = pose @ step
        if joint.get('type') == 'revolute':
            a = np.array([float(v) for v in joint.find('axis').get('xyz').split()])
            k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
            angle = q[joint.get('name')]
            turn = np.eye(4)
            turn[:3, :3] = np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * k @ k
            pose = pose @ turn
    return pose


# --- the committed sample (no ROS) -------------------------------------------------------------

def test_the_committed_sample_names_no_machine_path_and_is_no_package():
    files = sorted(p for p in EXAMPLE.rglob('*') if p.is_file())
    names = {p.relative_to(EXAMPLE).as_posix() for p in files}
    assert names == {'README.md', 'project.yaml', f'urdf/{TOP}', f'urdf/{TOOL}'}, names
    assert not any(p.name == 'package.xml' for p in files)      # colcon and rosdep never see it
    for f in files:
        text = f.read_text(errors='replace')
        for needle in _machine_paths(str(PKG)):
            assert needle not in text, (f.relative_to(EXAMPLE).as_posix(), needle)
    desc = yaml.safe_load((EXAMPLE / 'project.yaml').read_text())['robot']['description']
    assert desc == {'urdf_dir': 'urdf', 'top_xacro': TOP}   # relative; no meshes_dir: UR's own
    for name in (TOP, TOOL):
        text = (URDF_DIR / name).read_text()
        assert set(re.findall(r'\$\(find ([^)\s]+)\)', text)) <= {'ur_description'}, name
        assert 'file://' not in text and '<xacro:arg' not in text, name
        for include in re.findall(r'<xacro:include filename="([^"]+)"', text):
            assert (include.startswith('$(find ur_description)/')
                    or (URDF_DIR / include).is_file()), (name, include)


def test_tsa_depends_on_no_part_of_the_sample():
    manifest = (PKG / 'package.xml').read_text()
    assert not re.search(r'\bur_(description|moveit_config|robot_driver)\b', manifest)
    # setup.py installs none of it: neither by name nor through one of its globs
    setup = (PKG / 'setup.py').read_text()
    assert not re.search(r'ur5e|examples/\*\*', setup)
    patterns = [c.args[0].value for c in ast.walk(ast.parse(setup)) if isinstance(c, ast.Call)
                and getattr(c.func, 'id', None) == 'glob' and c.args
                and isinstance(c.args[0], ast.Constant)]
    assert 'examples/*.yaml' in patterns, patterns
    installed = [m for pattern in patterns
                 for m in glob.glob(pattern, root_dir=PKG, recursive=True)]
    assert not [m for m in installed if m.startswith('examples/ur5e_suction')], installed
    named = [f.relative_to(PKG).as_posix()
             for f in sorted((PKG / 'trainit_setup_assistant').rglob('*'))
             if f.is_file() and f.suffix in ('.py', '.j2', '.yaml', '.txt')
             and re.search(r'ur_description|ur5e_suction|examples/ur5e',
                           f.read_text(errors='replace'))]
    assert named == []


def test_the_sample_is_the_three_line_suction_contract_on_mock(project):
    from trainit_setup_assistant.generator.eef_contract import device
    written = yaml.safe_load((EXAMPLE / 'project.yaml').read_text())['robot']['gripper']
    # as written: the kind, the TCP link and the three typed lines -- no pseudo-joint, no SRDF
    # end-effector group, no legacy controller
    assert written == {'kind': 'suction', 'eef_parent_link': 'tcp', 'contract': {
        'actuator': {'command_interface': 'set_bool'},
        'verification': {'require_grasp_confirmation': False,
                         'require_release_confirmation': False},
        'adapters': {'mock': 'trainit_mock'}}}
    assert project.schema_version == 3 and project.robot.gripper.is_canonical()
    assert [m.value for m in project.deployment.modes] == ['mock']
    assert project.robot.namespace() == ''
    assert project.robot.planning_group.joints == ARM
    assert project.robot.planning_group.tip_link == project.robot.tip_link == 'tcp'
    dev = device(project, with_srdf=False)
    assert dev.id == 'main'
    assert [(a.id, a.interface.value) for a in dev.actuators] == [('vacuum', 'set_bool')]
    assert [(s.id, s.tcp, s.grip) for s in dev.sites] == [('main', 'tcp', 'vacuum')]


def test_the_held_box_clears_the_table_and_the_cup(project):
    # At pick and place the tool points DOWN, so the box hangs attach_offset[2] under the TCP.
    table = next(o for o in project.scene.objects if o.id == 'table')
    top = table.position[2] + table.dims[2] / 2
    box = project.scene.payload
    assert top < 0                         # under the robot base, whose mesh starts at z = 0
    assert 0 < box.attach_offset[2] - box.dims[2] / 2 <= 0.005    # a gap: no tool link touched
    app = project.application
    for name in ('pick', 'place'):
        wp, above = app.waypoint_by_name(name), app.waypoint_by_name(f'post_{name}')
        assert wp.orientation == above.orientation == [1.0, 0.0, 0.0, 0.0], name
        bottom = wp.position[2] - box.attach_offset[2] - box.dims[2] / 2
        assert bottom - top >= 0.005, name                   # the held box never touches the table
        assert above.position[:2] == wp.position[:2], name   # the approach is vertical
        assert above.position[2] - wp.position[2] == pytest.approx(0.15), name
    assert app.sequence == SEQUENCE


# --- the description (ur_description) -----------------------------------------------------------

def test_the_description_is_the_ur5e_and_the_suction_tool(project):
    share = needs_ur_description()
    urdf = _compile(URDF_DIR / TOP)
    assert urdf.get('name') == ROBOT
    assert urdf.findall('ros2_control') == []        # the bundle writes the only block
    assert [j.get('name') for j in urdf.findall('joint') if j.get('type') != 'fixed'] == ARM
    parent = {j.find('child').get('link'): (j.get('type'), j.find('parent').get('link'))
              for j in urdf.findall('joint')}
    assert parent['tcp'] == ('fixed', 'suction_body')
    assert parent['suction_body'] == ('fixed', 'tool0')
    meshes = {m.get('filename') for m in urdf.iter('mesh')}
    assert meshes and all(u.startswith('package://ur_description/') for u in meshes), meshes
    assert all((share.parent / u[len('package://'):]).is_file() for u in meshes)
    links = {link.get('name') for link in urdf.findall('link')}
    r = project.robot
    named = {r.base_frame, r.tip_link, r.planning_group.base_link, r.planning_group.tip_link,
             r.gripper.eef_parent_link, project.scene.payload.attach_link,
             *(o.frame for o in project.scene.objects),
             *(link for d in r.disable_collisions for link in (d.link1, d.link2))}
    assert named <= links, sorted(named - links)


def test_the_named_states_put_the_tcp_straight_above_the_targets(project):
    needs_ur_description()
    urdf = _compile(URDF_DIR / TOP)
    states = {s.name: s.joint_values for s in project.robot.named_states}
    for state, waypoint in (('above_pick', 'post_pick'), ('above_place', 'post_place')):
        pose = _fk(urdf, states[state], 'tcp')
        target = project.application.waypoint_by_name(waypoint)
        assert np.abs(pose[:3, 3] - target.position).max() < 1e-3, (state, pose[:3, 3])
        cos = (np.trace(DOWN.T @ pose[:3, :3]) - 1) / 2
        assert math.acos(max(-1.0, min(1.0, cos))) < math.radians(0.1), state
    assert _fk(urdf, states['home'], 'tcp')[2, 2] < -0.9999      # home points the tool down too


def test_tsa_refuses_nothing_and_warns_nothing(project):
    needs_ur_description()
    from trainit_setup_assistant.applications import get_application
    from trainit_setup_assistant.generator.eef_contract import (eef_contract_problems,
                                                                eef_contract_warnings)
    from trainit_setup_assistant.generator.orchestrator import refuse_application_problems
    assert eef_contract_problems(project) == []      # R1-R26; R9 compiled the description
    assert eef_contract_warnings(project) == []
    assert get_application(project.application.type).validate(project) == []
    refuse_application_problems(project)              # raises on a refusal


# --- the generated bundle (ur_description) -----------------------------------------------------

def test_generation_writes_the_readme_s_26_files_and_no_warning(bundle):
    manifest, root = bundle['manifest'], bundle['root']
    assert manifest.warnings == [] and manifest.local_warnings == []
    assert sorted(e.path for e in manifest.files) == FILES
    on_disk = sorted(p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file())
    assert on_disk == sorted(FILES + ['generation_manifest.yaml'])
    assert len(FILES) == 26                           # the number the README prints


def test_the_bundle_names_no_machine_path_and_records_the_sample_portably(bundle):
    root = bundle['root']
    for f in sorted(root.rglob('*')):
        if f.is_file():
            text = f.read_text(errors='replace')
            for needle in _machine_paths(str(EXAMPLE), str(root), str(PKG)):
                assert needle not in text, (f.relative_to(root).as_posix(), needle)
    copies = {e.path: e.source for e in bundle['manifest'].files if e.action == 'COPY'}
    assert copies == {f'{DESC}/urdf/{n}':
                      f'package://trainit_setup_assistant/examples/ur5e_suction/urdf/{n}'
                      for n in (TOP, TOOL)}
    for n in (TOP, TOOL):                             # byte copies: $(find ur_description) kept
        assert (root / DESC / 'urdf' / n).read_bytes() == (URDF_DIR / n).read_bytes()


def test_the_tree_is_the_pick_and_place_the_project_describes(bundle):
    tree = ET.parse(bundle['root'] / APP / 'bt_trees' / 'pick_place.xml').getroot()
    body = tree.find('BehaviorTree').find('Sequence')
    steps = [(e.tag, e.get('waypoint') or e.get('id') or e.get('eef')) for e in body
             if e.tag != 'Log']
    assert steps == [('AddCollisionObject', 'table'),
                     *[('MoveWaypoint', w) for w in SEQUENCE[:3]],
                     ('CloseGripper', 'main'), ('AttachObject', 'box'),
                     *[('MoveWaypoint', w) for w in SEQUENCE[3:6]],
                     ('OpenGripper', 'main'), ('DetachObject', 'box'),
                     *[('MoveWaypoint', w) for w in SEQUENCE[6:]]]
    attach = body.find('AttachObject')
    assert (attach.get('link'), attach.get('position')) == ('tcp', '0;0;0.027')
    assert tree.find('.//VerifyGrasp') is None and tree.find('.//CommandActuator') is None


def test_bt_params_carry_the_moves_and_the_contract(bundle):
    text = (bundle['root'] / APP / 'config' / 'bt_params.yaml').read_text()
    params = yaml.safe_load(text)['/**']['ros__parameters']
    assert (params['group'], params['base_frame'], params['tip_link']) == (
        'ur_manipulator', 'base_link', 'tcp')
    assert params['gripper_action'] == ''             # no legacy gripper action
    assert params['eef']['ids'] == ['main']
    assert params['eef']['main']['actuators'] == {'vacuum': {
        'command_interface': 'set_bool', 'actuation_feedback': False, 'confirm_timeout_ms': 2000}}
    moves = params['task_parameters']
    assert {w: (moves[w]['motion'], moves[w].get('planner')) for w in SEQUENCE} == {
        'home': ('ptp', 'pilz'), 'pre_pick': ('ptp', 'pilz'), 'pick': ('lin', None),
        'post_pick': ('lin', None), 'pre_place': ('ptp', 'pilz'), 'place': ('lin', None),
        'post_place': ('lin', None)}
    assert (moves['pick']['position'], moves['pick']['orientation']) == (
        [0.45, 0.25, 0.055], [1.0, 0.0, 0.0, 0.0])


def test_the_bring_up_runs_the_mock_adapter_and_no_legacy_gripper(bundle):
    text = (bundle['root'] / APP / 'launch' / 'bringup.launch.py').read_text()
    consts = {t.id: ast.literal_eval(node.value) for node in ast.parse(text).body
              if isinstance(node, ast.Assign) for t in node.targets
              if isinstance(t, ast.Name) and t.id in ('BRIDGES', 'GRIPPER_MOCK_ACTION',
                                                      'VALID_MODES')}
    assert consts['VALID_MODES'] == ('mock',)
    assert consts['GRIPPER_MOCK_ACTION'] is None
    assert [(b['package'], b['executable'], b['modes'], b['parameters'][0]['device_id'])
            for b in consts['BRIDGES']] == [
        ('trainit_hardware_layer', 'mock_eef_adapter', ['mock'], 'main')]
    floors = {e.text.strip(): e.get('version_gte')
              for e in ET.parse(bundle['root'] / APP / 'package.xml').getroot().iter('exec_depend')}
    assert floors.get('trainit_motion_runtime') and floors.get('trainit_hardware_layer'), floors


def test_the_moveit_config_carries_the_project_data(bundle):
    root, project = bundle['root'], bundle['project']
    config = root / CFG / 'config'
    srdf = ET.parse(config / f'{ROBOT}.srdf').getroot()
    assert [(g.get('name'), g.find('chain').get('base_link'), g.find('chain').get('tip_link'))
            for g in srdf.findall('group')] == [('ur_manipulator', 'base_link', 'tcp')]
    assert [s.get('name') for s in srdf.findall('group_state')] == [
        'home', 'above_pick', 'above_place']
    assert srdf.findall('end_effector') == []
    assert [(d.get('link1'), d.get('link2'), d.get('reason'))
            for d in srdf.findall('disable_collisions')] == [
        (d.link1, d.link2, d.reason) for d in project.robot.disable_collisions]
    urdf = _compile(URDF_DIR / TOP)
    urdf_velocity = {j.get('name'): float(j.find('limit').get('velocity'))
                     for j in urdf.findall('joint') if j.get('name') in ARM}
    limits = yaml.safe_load((config / 'joint_limits.yaml').read_text())['joint_limits']
    assert sorted(limits) == sorted(ARM)
    # Pilz needs acceleration limits and refuses a velocity above the URDF's
    for joint, limit in limits.items():
        assert limit['has_acceleration_limits'] and limit['max_acceleration'] > 0, joint
        assert 0 < limit['max_velocity'] <= urdf_velocity[joint], joint
    initial = yaml.safe_load((config / 'initial_positions.yaml').read_text())['initial_positions']
    assert initial == project.robot.initial_positions
    controllers = yaml.safe_load((config / 'ros2_controllers.yaml').read_text())
    assert controllers['moveit_joint_controller']['ros__parameters']['joints'] == ARM


def test_the_bring_up_robot_description_has_one_mock_ros2_control_block(bundle, tmp_path,
                                                                        monkeypatch):
    """The generated wrapper compiled as the mock bring-up compiles it, its
    $(find ur5e_suction_description) served by a temporary ament prefix holding the generated
    description package."""
    root, project = bundle['root'], bundle['project']
    index = tmp_path / 'prefix' / 'share' / 'ament_index' / 'resource_index' / 'packages'
    index.mkdir(parents=True)
    (index / DESC).touch()
    (tmp_path / 'prefix' / 'share' / DESC).symlink_to(root / DESC, target_is_directory=True)
    monkeypatch.setenv('AMENT_PREFIX_PATH', os.pathsep.join(
        [str(tmp_path / 'prefix'), os.environ['AMENT_PREFIX_PATH']]))
    urdf = _compile(root / CFG / 'config' / f'{ROBOT}.urdf.xacro', MOCK_MAPPINGS)
    blocks = urdf.findall('ros2_control')
    assert [(b.get('name'), b.findtext('hardware/plugin')) for b in blocks] == [
        ('UR5E_SUCTIONSystem', 'mock_components/GenericSystem')]
    joints = blocks[0].findall('joint')
    assert [j.get('name') for j in joints] == ARM
    initial = {j.get('name'): float(j.find("state_interface[@name='position']/param").text)
               for j in joints}
    assert initial == pytest.approx(project.robot.initial_positions)
    assert [j.get('name') for j in urdf.findall('joint') if j.get('type') != 'fixed'] == ARM
    assert 'ur_robot_driver' not in ET.tostring(urdf, encoding='unicode')


def test_trainit_generate_prints_what_the_readme_says(monkeypatch, tmp_path, capsys):
    needs_ur_description()
    from trainit_setup_assistant.cli.generate import main
    monkeypatch.chdir(EXAMPLE)
    out = tmp_path / ROBOT
    try:
        assert main(['project.yaml', '-o', str(out)]) == 0
        printed = capsys.readouterr()
        assert printed.out.splitlines()[:2] == [f'Generated {len(FILES)} files into {out}',
                                                f'  packages: {DESC}, {CFG}, {APP}']
        assert [line for line in printed.err.splitlines()
                if line.lstrip().startswith('warning:')] == [], printed.err
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_the_wizard_s_commands_name_the_bundle_s_launch_files_and_arguments(bundle):
    """The commands the wizard prints after generating this bundle (gui/wizard.py imported, no
    widget shown): each ros2 launch names one of the bundle's launch files and only arguments that
    file declares, and the tree's launch, whose use_sim_time defaults to true, gets false."""
    from trainit_setup_assistant.gui.wizard import build_and_run_lines
    root = bundle['root']
    launches = [line.split('ros2 launch ', 1)[1].split()
                for line in build_and_run_lines(bundle['project'], str(root), str(root.parent))
                if 'ros2 launch ' in line]
    assert [launch[:2] for launch in launches] == [[DESC, 'view_robot.launch.py'],
                                                   [APP, 'bringup.launch.py'],
                                                   [APP, 'trainit_bt.launch.py']]
    given, defaults = {}, {}
    for package, name, *args in launches:
        text = (root / package / 'launch' / name).read_text()
        defaults[name] = dict(re.findall(
            r'DeclareLaunchArgument\("(\w+)", default_value="([^"]*)"', text))
        given[name] = dict(arg.split(':=', 1) for arg in args)
        assert set(given[name]) <= set(defaults[name]), (package, name, given[name])
    assert given['bringup.launch.py'] == {'mode': 'mock'}
    assert given['trainit_bt.launch.py'] == {'use_sim_time': 'false'}
    assert defaults['trainit_bt.launch.py']['use_sim_time'] == 'true'
