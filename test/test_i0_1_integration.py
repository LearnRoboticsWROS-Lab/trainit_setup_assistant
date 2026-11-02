"""vNext I0.1 (I0_DESIGN.md §4, §6, §7, §8): the project's integration with its authorities.

- **The MoveIt config is the robot planning authority.** The project's mirror of it is checked for
  drift.
- **The deployment world owns the physical world.** The planning projection records where it came
  from (`scene.provenance`), and a changed world shows as STALE.
- **The robot inside an Isaac stage is BOUND** (`deployment.isaac`). It is validated offline in the
  stage, never read for MoveIt semantics.
- **The scene-sync status is explicit:** LIVE / OFFLINE / CURRENT / SNAPSHOT / STALE / MANUAL. No global
  "synchronized" claim is ever made.

None of this reaches a bundle: projects without the new fields save byte for byte as before, and a
project with them generates the same tree.
"""
import copy
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer.integration import (CURRENT, LIVE, MANUAL, OFFLINE, SNAPSHOT, STALE,
                                                           STATEMENT, file_sha256, integration_problems,
                                                           isaac_binding_problems, moveit_config_drift,
                                                           scene_status)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.model import (CanonicalProject, IsaacBinding, SceneProvenance, project_to_dict,
                                           save_project)
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'
WS_SRC = PKG.parent
FR3WML_STAGE = WS_SRC / 'fr3wml_digital_twin' / 'fr3wml_isaac' / 'usd' / 'scenes' / 'bin_picking_cell.usd'
FR3WML_TEST_E = Path.home() / 'fr5_ws' / 'forensics' / 'vnext-m3' / 'm3_acceptance' / 'test_e' / 'm3e_fr3wml.yaml'


def raw(path=COMPOSITION) -> dict:
    return yaml.safe_load(Path(path).read_text())


def project(data=None, path=COMPOSITION) -> CanonicalProject:
    p = CanonicalProject.parse_obj(data if data is not None else raw(path))
    return resolve_project_paths(p, path)[0]


def assert_same_tree(a: Path, b: Path) -> None:
    from test_i0_0_foundations import assert_same_tree as same
    same(a, b)


# ---- a synthetic Isaac stage: a robot articulation with joints, a table, a cube ---------------------------

def make_stage(path: Path, joints=('j1', 'j2', 'j3', 'j4', 'j5', 'j6'), robot='/World/robot',
               articulation=True) -> Path:
    pxr = pytest.importorskip('pxr')
    from pxr import Gf, Usd, UsdGeom, UsdPhysics
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.Xform.Define(stage, '/World')
    root = UsdGeom.Xform.Define(stage, robot)
    root.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.8))
    base = UsdGeom.Xform.Define(stage, f'{robot}/base_link')
    if articulation:
        UsdPhysics.ArticulationRootAPI.Apply(base.GetPrim())
    for j in joints:
        UsdPhysics.RevoluteJoint.Define(stage, f'{robot}/joints/{j}')
    table = UsdGeom.Cube.Define(stage, '/World/table')
    table.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.0, 0.4))
    cube = UsdGeom.Cube.Define(stage, '/World/Cube')
    cube.AddTranslateOp().Set(Gf.Vec3d(0.5, 0.1, 0.82))
    stage.GetRootLayer().Save()
    del pxr
    return path


# ---- byte stability ---------------------------------------------------------------------------------------

def test_a_project_without_the_new_fields_saves_without_them(tmp_path):
    p = CanonicalProject.parse_obj(raw())
    save_project(p, tmp_path / 'p.yaml')
    data = yaml.safe_load((tmp_path / 'p.yaml').read_text())
    assert 'isaac' not in data['deployment'] and 'provenance' not in data['scene']


def test_the_new_fields_round_trip(tmp_path):
    data = raw()
    data['deployment']['isaac'] = {'stage': 'cell.usd', 'robot_prim': '/World/robot'}
    data['scene']['provenance'] = {'source': 'isaac_stage', 'file': 'cell.usd', 'sha256': 'ab' * 32,
                                   'base_prim': '/World/robot/base_link'}
    p = CanonicalProject.parse_obj(data)
    save_project(p, tmp_path / 'p.yaml')
    again = yaml.safe_load((tmp_path / 'p.yaml').read_text())
    assert again['deployment']['isaac'] == data['deployment']['isaac']
    assert again['scene']['provenance'] == data['scene']['provenance']
    assert project_to_dict(CanonicalProject.parse_obj(again)) == project_to_dict(p)


def test_an_unknown_provenance_source_is_refused():
    data = raw()
    data['scene']['provenance'] = {'source': 'blender', 'file': 'x', 'sha256': 'y'}
    with pytest.raises(ValueError, match='provenance.source'):
        CanonicalProject.parse_obj(data)


def test_the_binding_and_the_provenance_never_reach_the_bundle(tmp_path):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    plain = project()
    data = raw()
    data['deployment']['isaac'] = {'stage': str(tmp_path / 'none.usd'), 'robot_prim': '/World/robot'}
    data['scene']['provenance'] = {'source': 'isaac_stage', 'file': 'cell.usd', 'sha256': '0' * 64}
    bound = project(data)
    Orchestrator().generate(plain, tmp_path / 'plain')
    Orchestrator().generate(bound, tmp_path / 'bound')
    assert_same_tree(tmp_path / 'plain', tmp_path / 'bound')


# ---- the scene-sync status (D2) ------------------------------------------------------------------------

def _with_objects(world_file=None, sha=None):
    data = raw()
    data['scene']['objects'] = [
        {'id': 'table', 'source': 'usd', 'shape': 'box', 'dims': [1, 1, 0.1], 'position': [0.5, 0, 0],
         'category': 'static'},
        {'id': 'door', 'source': 'usd', 'shape': 'box', 'dims': [0.5, 0.02, 0.5], 'position': [0.8, 0, 0.3],
         'category': 'actuated'},
        {'id': 'Cube', 'source': 'usd', 'shape': 'box', 'dims': [0.02] * 3, 'position': [0.5, 0.1, 0.02],
         'category': 'dynamic', 'grasp_target': True},
        {'id': 'fence', 'source': 'primitive', 'shape': 'box', 'dims': [2, 0.02, 1], 'position': [0, 1, 0],
         'category': 'static'},
    ]
    if world_file is not None:
        data['scene']['provenance'] = {'source': 'isaac_stage', 'file': str(world_file),
                                       'sha256': sha or file_sha256(world_file)}
    return project(data)


def _by_id(status):
    return {o.id: o.status for o in status.objects}


def test_the_statement_is_the_owners_and_never_claims_global_sync():
    assert STATEMENT == ('The robot state is LIVE. The static planning environment is DERIVED from the '
                         'deployment world. Dynamic / actuated world-state synchronization is future work.')
    assert 'synchronized' not in STATEMENT.replace('synchronization is future work', '')


def test_imported_static_is_current_and_dynamic_and_actuated_are_snapshot(tmp_path):
    world = tmp_path / 'cell.usd'
    world.write_text('#usda 1.0\n')
    st = scene_status(_with_objects(world))
    assert st.world == CURRENT
    assert _by_id(st) == {'table': CURRENT, 'door': SNAPSHOT, 'Cube': SNAPSHOT, 'fence': MANUAL}
    assert 'NOT live-synchronized' in next(o.detail for o in st.objects if o.id == 'Cube')
    assert st.robot == OFFLINE
    assert st.header() == 'Robot: OFFLINE · Static scene: MANUAL · Dynamic/actuated: SNAPSHOT (not live-synced)'


def test_a_changed_world_makes_every_imported_object_stale(tmp_path):
    world = tmp_path / 'cell.usd'
    world.write_text('#usda 1.0\n')
    sha = file_sha256(world)
    world.write_text('#usda 1.0\n# moved the table\n')
    st = scene_status(_with_objects(world, sha))
    assert st.world == STALE and 'changed since the import' in st.world_detail
    assert _by_id(st) == {'table': STALE, 'door': STALE, 'Cube': STALE, 'fence': MANUAL}
    assert st.header().startswith('Robot: OFFLINE · Static scene: STALE · Dynamic/actuated: STALE')
    errors, warnings = integration_problems(_with_objects(world, sha), check_stage=False)
    assert any(w.startswith('R-I0-03: scene:') for w in warnings)


def test_a_missing_world_is_stale(tmp_path):
    st = scene_status(_with_objects(tmp_path / 'gone.usd', '0' * 64))
    assert st.world == STALE and 'missing' in st.world_detail


def test_a_scene_imported_before_provenance_is_manual_and_says_how_to_track_it():
    st = scene_status(_with_objects())
    assert st.world == MANUAL
    assert all(o.status == MANUAL for o in st.objects)
    assert st.header() == 'Robot: OFFLINE · Static scene: MANUAL · Dynamic/actuated: MANUAL (not live-synced)'
    assert 're-import' in next(o.detail for o in st.objects if o.id == 'table')


@pytest.mark.parametrize('age, robot', [(None, OFFLINE), (0.3, LIVE), (9.0, OFFLINE)])
def test_the_robot_is_live_only_with_fresh_joint_states(age, robot):
    assert scene_status(_with_objects(), joint_states_age=age).robot == robot


# ---- the MoveIt config drift (D1) ----------------------------------------------------------------------

SRDF = """<?xml version="1.0"?>
<robot name="cellbot">
  <group name="arm"><chain base_link="base_link" tip_link="tcp"/>
    <joint name="j1"/><joint name="j2"/></group>
  <group_state name="home" group="arm"><joint name="j1" value="0"/><joint name="j2" value="-1.5"/></group_state>
  <disable_collisions link1="base_link" link2="link1" reason="Adjacent"/>
</robot>
"""
CONTROLLERS = """moveit_simple_controller_manager:
  controller_names: [arm_controller]
  arm_controller: {type: FollowJointTrajectory, action_ns: follow_joint_trajectory, joints: [j1, j2]}
"""


def _copy_base(tmp_path, mutate=None):
    cfg = tmp_path / 'cellbot_moveit_config' / 'config'
    cfg.mkdir(parents=True)
    (cfg / 'cellbot.srdf').write_text(SRDF)
    (cfg / 'moveit_controllers.yaml').write_text(CONTROLLERS)
    data = raw()
    r = data['robot']
    r['base_moveit_config_path'] = str(cfg.parent)
    r['planning_group'] = {'name': 'arm', 'base_link': 'base_link', 'tip_link': 'tcp', 'joints': ['j1', 'j2']}
    r['named_states'] = [{'name': 'home', 'group': 'arm', 'joint_values': {'j1': 0.0, 'j2': -1.5}}]
    r['disable_collisions'] = [{'link1': 'link1', 'link2': 'base_link', 'reason': 'Adjacent'}]
    r.setdefault('arm_controller', {})['name'] = 'arm_controller'
    if mutate:
        mutate(r)
    return CanonicalProject.parse_obj(data)


def test_a_mirror_equal_to_its_moveit_config_does_not_drift(tmp_path):
    assert moveit_config_drift(_copy_base(tmp_path)) == []


@pytest.mark.parametrize('mutate, what', [
    (lambda r: r['planning_group'].__setitem__('joints', ['j1']), 'joints are j1, j2'),
    (lambda r: r['planning_group'].__setitem__('tip_link', 'tool0'), 'the chain base_link -> tcp'),
    (lambda r: r['named_states'][0]['joint_values'].__setitem__('j2', -1.0), "named state 'home'"),
    (lambda r: r.__setitem__('named_states', []), "named state 'home' is not in the project"),
    (lambda r: r.__setitem__('disable_collisions', []), 'self-collision matrix differs'),
    (lambda r: r['arm_controller'].__setitem__('name', 'other'), 'the arm controller is arm_controller'),
    (lambda r: r['planning_group'].__setitem__('name', 'manipulator'), "no planning group 'manipulator'"),
])
def test_every_drift_from_the_moveit_config_is_named(tmp_path, mutate, what):
    drift = moveit_config_drift(_copy_base(tmp_path, mutate))
    assert drift and all(d.startswith('R-I0-01: ') and 'Step 1' in d for d in drift)
    assert any(what in d for d in drift), drift


def test_a_from_scratch_robot_has_no_moveit_config_to_drift_from():
    assert moveit_config_drift(project()) == []


@pytest.mark.skipif(not FR3WML_TEST_E.is_file(), reason='the M3 acceptance project (private) is absent')
def test_the_reference_cell_mirror_matches_its_moveit_config():
    import ws_paths
    p = CanonicalProject.parse_obj(raw(FR3WML_TEST_E))
    p.robot.description.urdf_dir = ws_paths.in_workspace(p.robot.description.urdf_dir)
    p.robot.base_moveit_config_path = ws_paths.in_workspace(p.robot.base_moveit_config_path)
    assert moveit_config_drift(p) == []


# ---- the Isaac binding (Case A) ------------------------------------------------------------------------

def _bound(stage, robot_prim='/World/robot', joints=('j1', 'j2', 'j3', 'j4', 'j5', 'j6')):
    data = raw()
    data['deployment']['modes'] = ['mock', 'isaac']
    data['deployment']['isaac'] = {'stage': str(stage), 'robot_prim': robot_prim}
    data['robot']['planning_group']['joints'] = list(joints)
    return CanonicalProject.parse_obj(data)


def test_a_correct_binding_has_no_problem(tmp_path):
    stage = make_stage(tmp_path / 'cell.usda')
    p = _bound(stage)
    p.robot.base_frame = 'base_link'
    assert isaac_binding_problems(p) == ([], [])


@pytest.mark.parametrize('kw, stage_kw, what', [
    ({'robot_prim': '/World/arm'}, {}, "no prim '/World/arm'"),
    ({}, {'joints': ('j1', 'j2')}, 'has no joint j3, j4, j5, j6'),
    ({}, {'articulation': False}, 'no articulation root'),
])
def test_a_wrong_binding_is_an_error_naming_what_is_wrong(tmp_path, kw, stage_kw, what):
    stage = make_stage(tmp_path / 'cell.usda', **stage_kw)
    errors, _ = isaac_binding_problems(_bound(stage, **kw))
    assert any(what in e and e.startswith('R-I0-02: ') for e in errors), errors


def test_an_isaac_project_without_a_binding_is_warned_and_a_missing_stage_is_an_error(tmp_path):
    data = raw()
    data['deployment']['modes'] = ['mock', 'isaac']
    _, warnings = isaac_binding_problems(CanonicalProject.parse_obj(data))
    assert any('no Isaac world is bound' in w for w in warnings)
    errors, _ = isaac_binding_problems(_bound(tmp_path / 'absent.usd'))
    assert any('stage is not found' in e for e in errors)
    data['deployment']['modes'] = ['mock']
    assert isaac_binding_problems(CanonicalProject.parse_obj(data)) == ([], [])


def test_a_binding_without_a_robot_prim_is_the_unsupported_case_b(tmp_path):
    stage = make_stage(tmp_path / 'cell.usda')
    _, warnings = isaac_binding_problems(_bound(stage, robot_prim=None))
    assert any('not supported yet' in w for w in warnings)


@pytest.mark.skipif(not FR3WML_STAGE.is_file(), reason='the reference cell (private) is absent')
def test_the_reference_cell_binds_its_robot_by_name():
    data = raw()
    data['deployment']['modes'] = ['mock', 'isaac']
    data['deployment']['isaac'] = {'stage': str(FR3WML_STAGE), 'robot_prim': '/World/fr3wml_suction'}
    data['robot']['planning_group']['joints'] = ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']
    data['robot']['base_frame'] = 'base_link'
    assert isaac_binding_problems(CanonicalProject.parse_obj(data))[0] == []


# ---- the controller: import from the bound world, provenance, re-import -----------------------------------

def _controller(stage) -> AssistantController:
    ctrl = AssistantController(CanonicalProject.parse_obj(raw()))
    ctrl.project_path = str(COMPOSITION)
    ctrl.project.robot.base_frame = 'base_link'
    ctrl.set_isaac_binding(stage, '/World/robot')
    return ctrl


def _rules_with_meshes(rules):
    for r in rules:
        r['mesh'] = f"package://cell/meshes/{r['group']}.stl"
        r['include'] = True
    return rules


def test_an_import_from_the_bound_stage_records_its_provenance_and_skips_the_robot(tmp_path):
    stage = make_stage(tmp_path / 'cell.usda')
    ctrl = _controller(stage)
    rules = _rules_with_meshes(ctrl.import_usd_cell(str(stage), 'cell'))
    assert {r['group'] for r in rules} == {'table', 'Cube'}            # the robot subtree is excluded
    ctrl.apply_usd_mapping(rules)
    prov = ctrl.project.scene.provenance
    assert prov.source == 'isaac_stage' and prov.file == str(stage) and prov.sha256 == file_sha256(stage)
    assert prov.base_prim == '/World/robot/base_link'
    table = next(o for o in ctrl.project.scene.objects if o.id == 'table')
    assert table.position == pytest.approx([0.5, 0.0, -0.4])           # in base_link: the base is at z 0.8
    assert ctrl.scene_status().world == CURRENT


def test_a_re_import_keeps_the_users_categories(tmp_path):
    stage = make_stage(tmp_path / 'cell.usda')
    ctrl = _controller(stage)
    rules = _rules_with_meshes(ctrl.import_usd_cell(str(stage), 'cell'))
    for r in rules:
        if r['group'] == 'Cube':
            r['category'], r['grasp'] = 'dynamic', True
    ctrl.apply_usd_mapping(rules)
    again = {r['group']: r for r in ctrl.import_usd_cell(str(stage), 'cell')}
    assert again['Cube']['category'] == 'dynamic' and again['Cube']['grasp'] is True
    assert again['Cube']['mesh'] == 'package://cell/meshes/Cube.stl'


def test_the_editor_validation_reports_the_integration(tmp_path):
    stage = make_stage(tmp_path / 'cell.usda', joints=('j1',))
    ctrl = _controller(stage)
    ctrl.project.deployment.modes = ['mock', 'isaac']
    errors, warnings = ctrl.validate_program()
    assert any(e.startswith('R-I0-02: ') for e in errors), errors


# ---- the Project Overview (pure rows) and its Qt views -----------------------------------------------------

def test_the_overview_names_every_block_and_shows_the_statement(tmp_path):
    from trainit_setup_assistant.composer.overview import process_channels, project_overview
    world = tmp_path / 'cell.usd'
    world.write_text('#usda 1.0\n')
    p = _with_objects(world)
    rows = project_overview(p, scene_status(p), errors=[], warnings=[])
    blocks = [r.block for r in rows]
    assert blocks[:4] == ['ROBOT', 'MOVEIT CONFIG', 'DEPLOYMENT', 'WORLD']
    for b in ('ROBOT BINDING', 'SCENE', 'SCENE SYNC', 'TOOLS / EEF', 'SENSORS', 'APPLICATION',
              'PROCESS CHANNELS', 'VALIDATION'):
        assert b in blocks, b
    sync = next(r for r in rows if r.block == 'SCENE SYNC')
    assert sync.detail == STATEMENT and 'SNAPSHOT (not live-synced)' in sync.text
    assert next(r for r in rows if r.block == 'WORLD').mark == 'ok'
    assert next(r for r in rows if r.block == 'MOVEIT CONFIG').text.startswith('none: a quick robot')
    assert next(r for r in rows if r.block == 'PROCESS CHANNELS').text == (', '.join(process_channels(p))
                                                                         or 'none')


def test_the_overview_marks_drift_a_bad_binding_and_a_stale_world(tmp_path):
    from trainit_setup_assistant.composer.overview import project_overview
    stage = make_stage(tmp_path / 'cell.usda', joints=('j1',))
    p = _bound(stage)
    p.scene.provenance = SceneProvenance(source='isaac_stage', file=str(stage), sha256='0' * 64)
    errors, warnings = integration_problems(p)
    rows = {r.block: r for r in project_overview(p, scene_status(p), errors, warnings)}
    assert rows['ROBOT BINDING'].mark == 'error' and 'has no joint' in rows['ROBOT BINDING'].detail
    assert rows['WORLD'].mark == 'warn' and 'STALE' in rows['WORLD'].text
    assert rows['VALIDATION'].mark == 'error'


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_step_7_opens_the_composer_on_the_overview(qapp):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(COMPOSITION)
    w = SetupWizard(ctrl)
    blocks = w.blocks_page
    blocks.initializePage()
    composer = blocks.program_panel
    assert list(composer.sections)[0] == 'Overview' and 'Program' in composer.sections   # sections grow per slice
    assert composer.stack.currentWidget() is composer.overview
    shown = [composer.overview.tree.topLevelItem(i).text(1)
             for i in range(composer.overview.tree.topLevelItemCount())]
    assert shown[:2] == ['ROBOT', 'MOVEIT CONFIG'] and 'SCENE SYNC' in shown
    assert STATEMENT in composer.overview.statement.text()
    composer.select('Program')
    assert composer.stack.currentWidget() is not composer.overview


def test_step_2_shows_the_sync_status_and_offers_a_re_import_when_stale(qapp, tmp_path):
    from trainit_setup_assistant.gui.wizard import SetupWizard
    stage = make_stage(tmp_path / 'cell.usda')
    ctrl = _controller(stage)
    w = SetupWizard(ctrl)
    page = w.scene_page
    page.initializePage()
    assert page.robot_prim.text() == '/World/robot' and page.usd_path.text() == str(stage)
    rules = _rules_with_meshes(ctrl.import_usd_cell(str(stage), 'cell'))
    ctrl.apply_usd_mapping(rules)
    page._refresh()
    assert 'Robot: OFFLINE' in page.sync_status.text() and STATEMENT in page.sync_status.text()
    assert not page.reimport.isVisibleTo(page)
    items = [page.list.item(i).text() for i in range(page.list.count())]
    assert any(t.startswith('table ') and t.endswith('CURRENT') for t in items), items
    stage.write_text(stage.read_text() + '\n')                     # the world changes
    page._refresh()
    assert page.reimport.isVisibleTo(page) and 'STALE' in page.sync_status.text()
