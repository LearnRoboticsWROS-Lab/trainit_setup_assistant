"""The Canonical Task Program (vNext M3.3, ADR-0024, ADR-0025): the model, its validation, the
provider resolution, the generated bundle, the runtime floors and the wizard's guards.

The project under test is ``examples/ur5e_suction/task_program.yaml`` -- Validation A on the public
UR5e sample cell -- loaded the README's way (from its directory), and variants of it built from its
YAML data. Tests that compile the robot description (a frame other than base_link, the end-effector
contract at generation) need ur_description and xacro: they skip without them, naming the apt
package (fail with TRAINIT_REQUIRE_GOLDEN=1), like the sample's own tests. Every generation goes to
a pytest temp dir (test-time generation, removed at teardown). ROS-free; headless for the GUI.
"""
import copy
import json
import math
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.applications.task_program import (ProgramRefusals, plan_program,
                                                                program_overview)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.runtime_requirements import requirement, use_of
from trainit_setup_assistant.model import (CanonicalProject, load_project, project_to_dict,
                                           save_project)
from trainit_setup_assistant.model.enums import AppType

PKG = Path(__file__).resolve().parents[1]
EXAMPLE = PKG / 'examples' / 'ur5e_suction'
PROGRAM_FILE = EXAMPLE / 'task_program.yaml'
APP = 'ur5e_program_app'


def needs_ur_description():
    try:
        import xacro  # noqa: F401
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory('ur_description')
    except Exception:                                   # noqa: BLE001
        import eef_fixture
        eef_fixture.skip_or_fail('the UR5e sample cell needs ur_description and xacro in a sourced '
                                 'ROS 2 environment (sudo apt install ros-humble-ur-description)')


@pytest.fixture(autouse=True)
def _in_the_example(monkeypatch):
    # the sample's paths are relative to its directory (the README's rule)
    monkeypatch.chdir(EXAMPLE)


def raw() -> dict:
    return yaml.safe_load(PROGRAM_FILE.read_text())


def build(mutate=None) -> CanonicalProject:
    data = raw()
    if mutate is not None:
        mutate(data['application'])
    return CanonicalProject.parse_obj(data)


def base_link_frames(app):
    """The program with every pose in base_link (no description needed to check a frame)."""
    for w in app['waypoints']:
        if 'position' in w:
            w.pop('frame', None)
            x, y, z = w['position']
            w['position'] = [-x, -y, z]
            w['orientation'] = [1.0, 0.0, 0.0, 0.0]


def problems(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


def load_text(tmp_path, text_mutator) -> str:
    data = raw()
    text_mutator(data)
    f = tmp_path / 'p.yaml'
    f.write_text(yaml.safe_dump(data, sort_keys=False))
    with pytest.raises(ValueError) as exc:
        load_project(f)
    return str(exc.value)


# --- A. schema version guard ------------------------------------------------------------------------

def test_a_the_example_loads_as_a_schema_4_task_program():
    p = load_project(PROGRAM_FILE)
    assert p.schema_version == 4
    assert p.application.type is AppType.TASK_PROGRAM
    assert p.application.is_task_program()
    prog = p.application.program
    assert [w.name for w in prog.waypoints] == ['home', 'w1', 'w2', 'w2b', 'w3', 'w4']
    assert prog.main_task == 'validation_a'
    # the template fields are unused and stay at their defaults
    assert p.application.waypoints == [] and p.application.segments == []


@pytest.mark.parametrize('version', [2, 3])
def test_a_a_task_program_must_say_schema_4(version):
    data = raw()
    data['schema_version'] = version
    with pytest.raises(ValueError, match=r'R-M3-01: a Canonical Task Program .* schema_version: 4'):
        CanonicalProject.parse_obj(data)


def test_a_schema_4_without_a_task_program_is_refused():
    data = yaml.safe_load((EXAMPLE / 'project.yaml').read_text())
    data['schema_version'] = 4
    with pytest.raises(ValueError, match=r"schema_version 4 is the Canonical Task Program's, but "
                                         r"this application is 'pick_and_place'"):
        CanonicalProject.parse_obj(data)


def test_a_the_legacy_sample_still_loads_as_before():
    p = load_project(EXAMPLE / 'project.yaml')
    assert p.schema_version == 3 and p.application.type is AppType.PICK_AND_PLACE
    assert p.application.program is None
    assert 'program' not in project_to_dict(p)['application']


@pytest.mark.parametrize('key,hint', [
    ('segments', 'no folded segments'),
    ('sequence', 'main_task runs first'),
    ('global_planner_mode', 'application.providers'),
    ('tool_actions', 'template applications'),
    ('policies', 'template applications'),
    ('program', "has no 'program'"),
])
def test_a_a_task_program_does_not_mix_with_template_keys(tmp_path, key, hint):
    text = load_text(tmp_path, lambda d: d['application'].update({key: []}))
    assert 'R-M3-01' in text and hint in text


# --- B. strict, with hints ------------------------------------------------------------------------

@pytest.mark.parametrize('where,key,expected', [
    (('waypoints', 1), 'motion', 'a waypoint says WHERE, never HOW'),
    (('waypoints', 1), 'speed', 'a waypoint says WHERE, never HOW'),
    (('waypoints', 1), 'role', 'no waypoint roles'),
    (('waypoints', 1), 'allowed_start_tolerance', 'application.runtime.allowed_start_tolerance'),
    (('waypoints', 1), 'colour', "unknown key 'colour' (accepted: name, type, position"),
    (('tasks', 0, 'steps', 2, 'move'), 'planner', "a task program names a provider"),
    (('tasks', 0, 'steps', 2, 'move'), 'speedd', "unknown key 'speedd'"),
    # vNext M3.7: inputs / outputs are a task's (test_m3_7_composition.py); a near miss is hinted
    (('tasks', 0), 'calls', "a task calls another task with a step: 'task: <name>'"),
    # vNext M3.5: a path's events are available (test_m3_5_process_path.py); a near miss is hinted
    (('paths', 0), 'event', "a path's process events are a list"),
])
def test_b_an_unknown_key_is_refused_by_name(tmp_path, where, key, expected):
    def mutate(data):
        node = data['application']
        for k in where:
            node = node[k]
        node[key] = 1
    text = load_text(tmp_path, mutate)
    assert expected in text


# vNext M3.6 delivered grasp, release, detect and tool (test_m3_6_manipulation.py), vNext M3.7 task
# calls and the control steps (test_m3_7_composition.py)
@pytest.mark.parametrize('kind,what', [('check', 'robot-state')])
def test_b_a_step_kind_of_a_later_phase_is_refused_by_name(tmp_path, kind, what):
    text = load_text(tmp_path, lambda d: d['application']['tasks'][0]['steps'].append(
        {kind: {}}))
    assert f"step kind '{kind}' is not available in this TSA version" in text and what in text
    assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in text


def test_b_a_load_error_is_located_in_the_files_terms(tmp_path):
    text = load_text(tmp_path, lambda d: d['application']['tasks'][0]['steps'][2]['move'].update(
        {'profle': 'approach'}))
    assert "task 'validation_a' step 3 (move)" in text
    assert "unknown key 'profle'" in text and 'profile' in text
    text = load_text(tmp_path, lambda d: d['application']['motion_profiles']['transit'].update(
        {'velocity': 3}))
    assert "motion profile 'transit'" in text and "unknown key 'velocity'" in text


def test_b_a_dense_source_file_is_for_a_later_phase(tmp_path):
    def mutate(d):
        d['application']['paths'].append({'name': 'bead', 'source': {'file': 'x.csv'}})
    assert 'a dense path read from a file is planned for a later TSA version' in load_text(
        tmp_path, mutate)


# --- C. waypoints, frames, tcp, relative ------------------------------------------------------------

@pytest.mark.parametrize('change,expected', [
    ({'orientation': [0.0, 0.9, 0.0, 0.0]}, 'not a unit quaternion'),
    ({'position': [0.1, float('nan'), 0.2]}, 'finite'),
    ({'position': [0.1, True, 0.2]}, 'must be a number'),
    ({'position': [0.1, 0.2]}, 'must be 3 numbers'),
    ({'name': 'w1__2'}, "contains '__'"),
    ({'name': 'on'}, 'boolean or null'),
    ({'joints': [0.0] * 6}, "'joints' and 'named' belong to a joint target"),
])
def test_c_a_waypoint_is_checked_for_what_it_is(change, expected):
    def mutate(app):
        app['waypoints'][1].update(change)
    with pytest.raises(ValueError, match=expected):
        build(mutate)


def test_c_a_relative_target_inherits_its_frame_and_cannot_set_one():
    def mutate(app):
        app['waypoints'][2]['frame'] = 'base'
    with pytest.raises(ValueError, match='inherits its frame'):
        build(mutate)


def test_c_a_relative_target_is_resolved_along_the_tool_axes():
    needs_ur_description()
    plan = plan_program(load_project(PROGRAM_FILE))
    w2 = plan.targets['w2']
    assert w2.position == (-0.45, -0.25, 0.1)               # 0.2 m along the tool Z = down
    assert w2.orientation == (0.0, 1.0, 0.0, 0.0) and w2.frame == 'base'
    assert 'w1 + (0, 0, 0.2) m along w1\'s tool axes' in w2.derivation


def test_c_a_relative_target_along_base_axes_needs_its_reference_in_the_base_frame():
    needs_ur_description()

    def mutate(app):
        app['waypoints'][2]['relative']['frame'] = 'base'
    text = problems(build(mutate))
    assert "relative.frame base means the robot's base axes ('base_link')" in text
    assert "use relative.frame tool" in text


@pytest.mark.parametrize('rel,expected', [
    ({'to': 'w2', 'frame': 'tool'}, 'form a cycle'),
    ({'to': 'nowhere', 'frame': 'tool'}, "no waypoint is named 'nowhere'"),
    ({'to': 'home', 'frame': 'tool'}, "'home' is a joint target"),
    # vNext M3.6: target_local resolves (test_m3_6_manipulation.py T); a cycle is refused in it too
    ({'to': 'w2', 'frame': 'target_local'}, 'form a cycle'),
])
def test_c_relative_targets_are_refused_when_they_cannot_be_resolved(rel, expected):
    def mutate(app):
        base_link_frames(app)
        app['waypoints'][2]['relative'] = rel
    assert expected in problems(build(mutate))


def test_c_a_tcp_other_than_the_tip_link_is_refused():
    def mutate(app):
        base_link_frames(app)
        app['waypoints'][1]['tcp'] = 'tool0'
    text = problems(build(mutate))
    assert "waypoint 'w1': tcp 'tool0' is not the robot's tip link 'tcp'" in text
    assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in text


def test_c_a_frame_must_be_a_link_of_the_description():
    needs_ur_description()

    def mutate(app):
        app['waypoints'][1]['frame'] = 'part_frame'
    text = problems(build(mutate))
    assert "waypoint 'w1': frame 'part_frame' is not a link of the robot description" in text
    assert 'base_link' in text and 'tool0' in text


def test_c_a_joint_target_is_checked_against_the_robot():
    def mutate(app):
        base_link_frames(app)
        app['waypoints'][0] = {'name': 'home', 'type': 'joint', 'named': 'parked'}
        app['waypoints'].append({'name': 'j', 'type': 'joint', 'joints': [0.0, 1.0]})
    text = problems(build(mutate))
    assert "'parked' is not a named state of planning group 'ur_manipulator'" in text
    assert "waypoint 'j': 2 joint values for the 6 joints" in text


# --- D. motion profiles and limits --------------------------------------------------------------

def test_d_a_profile_named_default_is_reserved():
    with pytest.raises(ValueError, match="'default' is reserved"):
        build(lambda app: app['motion_profiles'].update({'default': {'speed': 10}}))


def test_d_an_unknown_profile_is_refused_where_it_is_named():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][1]['move']['profile'] = 'fast'
    text = problems(build(mutate))
    assert ("task 'validation_a' step 2 (move ptp to 'w1'): profile 'fast' is not a motion "
            "profile of this program (motion_profiles: transit, approach, process)") in text


def test_d_a_percent_and_a_tcp_limit_give_the_most_restrictive_scaling():
    needs_ur_description()
    plan = plan_program(load_project(PROGRAM_FILE))
    # approach: speed 30 %, tcp_velocity_mps 0.1 on max_trans_vel 1.0 -> 0.1 wins
    assert plan.entries['w2']['speed'] == 10 and plan.entries['w2']['accel'] == 30
    # process (the path): speed 20 %, tcp 0.08 -> 0.08
    sweep = plan.paths[0].data
    assert sweep['velocity_scaling'] == pytest.approx(0.08)
    assert sweep['acceleration_scaling'] == pytest.approx(0.2)


def test_d_every_motion_needs_a_speed_and_an_acceleration():
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['transit'] = {'speed': 50}
    text = problems(build(mutate))
    assert "task 'validation_a' step 1 (move ptp to 'home'): no acceleration limit" in text


def test_d_a_tcp_limit_below_one_percent_is_refused_on_a_move():
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['approach'] = {'acceleration': 30, 'tcp_velocity_mps': 0.005}
    text = problems(build(mutate))
    assert 'below the 1 % a move executes' in text


def test_d_a_goal_tolerance_or_a_planning_budget_on_a_move_step_needs_a_profile():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][3]['move']['goal_tolerance'] = {'position_m': 0.001}
    text = problems(build(mutate))
    assert 'goal_tolerance on a move step' in text and 'motion profile' in text


def test_d_a_profile_carries_planning_budget_and_tolerance_to_tmr():
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['free_move'] = {'speed': 40, 'acceleration': 40,
                                               'planning_time_s': 2.5, 'planning_attempts': 5}
        app['motion_profiles']['arc'] = {'speed': 20, 'acceleration': 20,
                                         'goal_tolerance': {'position_m': 0.002}}
        app['tasks'][0]['steps'][0]['move'].update({'type': 'free', 'profile': 'free_move'})
        app['tasks'][0]['steps'][3]['move']['profile'] = 'arc'
    plan = plan_program(build(mutate))
    assert plan.profiles['free_move'] == {'planning_time': 2.5, 'planning_attempts': 5}
    assert plan.profiles['arc'] == {'goal_position_tolerance': 0.002}
    assert plan.profiles['transit'] == {}
    assert plan.entries['home']['planner'] == 'ompl'              # free -> the ompl row's mode


def test_d_a_joint_goal_tolerance_is_refused_tmr_carries_none():
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['arc'] = {'speed': 20, 'acceleration': 20,
                                         'goal_tolerance': {'joint_rad': 0.01}}
        app['tasks'][0]['steps'][3]['move']['profile'] = 'arc'
    assert 'goal_tolerance.joint_rad' in problems(build(mutate))


def test_d_a_segment_profile_with_a_goal_tolerance_is_refused_inside_a_path():
    """Owner, M3.2 close: no per-segment goal tolerance; refused, never ignored or promoted."""
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['tight'] = {'speed': 20, 'acceleration': 20,
                                           'goal_tolerance': {'position_m': 0.001,
                                                              'orientation_rad': 0.01}}
        app['paths'][0]['segments'][0]['profile'] = 'tight'
    text = problems(build(mutate))
    assert 'R-M3-12' in text and 'segment 0 sets a goal_tolerance' in text


def test_d_a_path_level_profile_gives_the_path_wide_tolerance():
    def mutate(app):
        base_link_frames(app)
        del app['paths'][0]['goal_tolerance']
        app['motion_profiles']['process']['goal_tolerance'] = {'position_m': 0.002,
                                                               'orientation_rad': 0.02}
    plan = plan_program(build(mutate))
    assert plan.paths[0].data['goal_tolerance'] == {'position_m': 0.002, 'orientation_rad': 0.02}


# --- E. providers ---------------------------------------------------------------------------------

def test_e_omitted_providers_resolve_to_the_registry_defaults_and_say_so():
    needs_ur_description()
    plan = plan_program(load_project(PROGRAM_FILE))
    assert plan.entries['home']['planner'] == 'pilz'               # pilz_ptp's planner MODE
    assert 'planner' not in plan.entries['w2']                     # lin: Pilz LIN, no planner key
    notes = '\n'.join(plan.entry_notes['w2'])
    assert 'provider pilz_lin (the registry default for lin)' in notes
    assert plan.paths[0].resolution.provider == 'pilz_sequence'
    assert plan.paths[0].resolution.provider_source == 'registry_default'
    assert plan.backend == 'pilz_sequence' and plan.planner_mode == 'ompl'


def test_e_application_providers_and_a_step_provider_are_written():
    def mutate(app):
        base_link_frames(app)
        app['providers'] = {'free': 'ompl_chomp'}
        app['tasks'][0]['steps'][0]['move']['type'] = 'free'
        app['tasks'][0]['steps'][6]['move'].update({'type': 'free', 'provider': 'ompl'})
    plan = plan_program(build(mutate))
    assert plan.planner_mode == 'ompl_chomp'
    assert plan.entries['home']['planner'] == 'ompl_chomp'
    notes = '\n'.join(plan.entry_notes['home'])
    assert 'provider ompl_chomp (application.providers.free)' in notes
    assert plan.entries['w1__2']['planner'] == 'ompl'
    assert 'provider ompl (chosen on the step)' in '\n'.join(plan.entry_notes['w1__2'])


@pytest.mark.parametrize('mutate,expected', [
    (lambda a: a['tasks'][0]['steps'][2]['move'].update({'provider': 'ompl'}),
     ("R-M3-06 REFUSED: task 'validation_a' step 3 (move lin to 'w2') (lin): a planner on a lin "
      'move')),
    (lambda a: a['motion_profiles']['transit'].update({'jerk': 5}),
     "provider 'pilz_ptp' does not provide 'jerk_constraints'"),
    (lambda a: a.update({'providers': {'path': 'cartesian_waypoints'}}),
     "R-M3-10 REFUSED: providers.path: provider 'cartesian_waypoints' cannot serve a path step"),
    (lambda a: a['tasks'][0]['steps'][1]['move'].update({'provider': 'ompl'}),
     "provider 'ompl' does not provide 'joint_ptp'"),
])
def test_e_an_unsupported_request_is_refused_with_the_capability_diagnostic(mutate, expected):
    def change(app):
        base_link_frames(app)
        mutate(app)
    assert expected in problems(build(change))


def test_e_m3_a_the_blended_path_on_cartesian_waypoints_is_refused_naming_what_to_change():
    """The M3-A rejection: a blend requested of a provider without 'blending'."""
    def mutate(app):
        base_link_frames(app)
        app['paths'][0]['provider'] = 'cartesian_waypoints'
    text = problems(build(mutate))
    assert ("R-M3-10 REFUSED: path 'sweep' (path): provider 'cartesian_waypoints' does not provide "
            "'cartesian_linear', 'cartesian_circular', 'multi_segment_path', 'blending'") in text
    assert 'provider: cartesian_waypoints (request)' in text
    assert 'because:  blending <- continuity blend (provided by pilz_sequence)' in text
    assert ('change:   name a provider that provides it -- pilz_sequence -- or remove what '
            'requires the missing capability') in text


# --- F. paths -------------------------------------------------------------------------------------

@pytest.mark.parametrize('change,expected', [
    (lambda p: p.update(states=['w1']), 'at least 2 states'),
    (lambda p: p.update(segments=p['segments'][:2]), '4 states need 3 segments, got 2'),
])
def test_f_a_path_shape_is_checked_at_load(change, expected):
    with pytest.raises(ValueError, match=expected):
        build(lambda app: change(app['paths'][0]))


@pytest.mark.parametrize('change,expected', [
    (lambda p: p['states'].__setitem__(2, 'w9'), "path 'sweep' state 2: no waypoint is named 'w9'"),
    (lambda p: p['states'].__setitem__(0, 'home'), "state 0 'home' is a joint target"),
    (lambda p: p['segments'][1].pop('via'), 'a circ segment needs exactly one of via or center'),
    (lambda p: p['segments'][1].update(center='w2b'), 'exactly one of via or center'),
    (lambda p: p['segments'][0].update(type='ptp'), 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL'),
    (lambda p: p['segments'][2].update(blend_radius_m=0.01),
     'R-M3-08 REFUSED: path \'sweep\' (path): the last segment blends'),
    (lambda p: p.update(continuity='stop'), "a blend_radius_m on a continuity: stop path"),
    (lambda p: [s.pop('blend_radius_m', None) for s in p['segments']],
     'continuity: blend with no blend_radius_m'),
    (lambda p: p['segments'][0].update(blend_radius_m=0.15),
     'exceeds half of an adjacent segment'),
    (lambda p: p.update(tcp='tool0'), "path 'sweep': tcp 'tool0' is not the robot's tip link"),
    (lambda p: p['segments'][0].update(type='circ', via='w2'),
     'the via point coincides with an end of the arc'),
])
def test_f_a_path_is_refused_before_any_file(change, expected):
    def mutate(app):
        base_link_frames(app)
        change(app['paths'][0])
    assert expected in problems(build(mutate))


def test_f_a_via_collinear_with_the_arc_is_refused():
    def mutate(app):
        base_link_frames(app)
        app['waypoints'].append({'name': 'mid', 'position': [0.45, 0.0, 0.1],
                                 'orientation': [1.0, 0.0, 0.0, 0.0]})
        app['paths'][0]['segments'][1]['via'] = 'mid'
    assert 'collinear' in problems(build(mutate))


def test_f_a_path_that_cannot_start_where_the_robot_is_is_refused():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][6]['move']['to'] = 'w3'
    text = problems(build(mutate))
    assert ("task 'validation_a' step 8 (path 'sweep'): the path starts at 'w1', but the step "
            "before leaves the robot at 'w3'") in text


def test_f_a_path_as_first_step_is_a_warning_not_a_guess():
    def mutate(app):
        base_link_frames(app)
        app['tasks'].append({'name': 'only_path', 'steps': [{'path': 'sweep'}]})
    plan = plan_program(build(mutate))
    assert any("nothing earlier in the task moves the robot there" in w for w in plan.warnings)
    assert any("never runs" in w for w in plan.warnings)


def test_f_a_path_step_naming_no_path_is_refused():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][7] = {'path': 'swep'}
    assert "no path is named 'swep' (paths: sweep)" in problems(build(mutate))


# --- G/H/I/N. the generated bundle (Validation A) ---------------------------------------------------

@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    needs_ur_description()
    tmp = tmp_path_factory.mktemp('m3_3_validation_a')
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(EXAMPLE)
        project = load_project('task_program.yaml')
        manifest = Orchestrator().generate(project, tmp / 'a')
        Orchestrator().generate(project, tmp / 'b')
    return dict(root=tmp / 'a', again=tmp / 'b', project=project, manifest=manifest)


def test_n_validation_a_generates_the_canonical_files(bundle):
    app = bundle['root'] / APP
    names = sorted(str(p.relative_to(app)) for p in app.rglob('*') if p.is_file())
    assert names == ['CMakeLists.txt', 'README.md', 'bt_trees/task_program.xml',
                     'config/bt_params.yaml', 'config/motion_profiles.yaml', 'config/paths.yaml',
                     'config/tasks.yaml',                    # vNext M3.7: the task manifest
                     'groot2/ur5e_suction_task_program.btproj', 'launch/bringup.launch.py',
                     'launch/trainit_bt.launch.py', 'package.xml']
    assert bundle['manifest'].as_dict()['warnings'] == []
    files = json.dumps(bundle['manifest'].as_dict())
    for f in ('config/paths.yaml', 'config/motion_profiles.yaml', 'bt_trees/task_program.xml'):
        assert f in files


def test_n_generation_is_deterministic(bundle):
    a, b = bundle['root'], bundle['again']
    for f in sorted(a.rglob('*')):
        if f.is_file() and f.name != 'generation_manifest.yaml':
            assert f.read_bytes() == (b / f.relative_to(a)).read_bytes(), f


def test_g_paths_yaml_is_tmrs_schema_1_with_the_resolved_path(bundle):
    data = yaml.safe_load((bundle['root'] / APP / 'config' / 'paths.yaml').read_text())
    assert data == {'schema_version': 1, 'paths': {'sweep': {
        'provider': 'pilz_sequence', 'continuity': 'blend', 'profile': 'process',
        'velocity_scaling': 0.08, 'acceleration_scaling': 0.2,
        'goal_tolerance': {'position_m': 0.001, 'orientation_rad': 0.01},
        'states': [{'waypoint': 'w1'}, {'waypoint': 'w2'}, {'waypoint': 'w3'},
                   {'waypoint': 'w4'}],
        'segments': [{'type': 'lin', 'blend_radius_m': 0.02},
                     {'type': 'circ', 'via': {'waypoint': 'w2b'}, 'blend_radius_m': 0.02},
                     {'type': 'lin'}]}}}
    text = (bundle['root'] / APP / 'config' / 'paths.yaml').read_text()
    assert '# sweep: 4 states, continuity blend; provider pilz_sequence (the registry default' in text


def test_h_task_parameters_hold_the_waypoints_and_their_moves(bundle):
    text = (bundle['root'] / APP / 'config' / 'bt_params.yaml').read_text()
    params = yaml.safe_load(text)['/**']['ros__parameters']
    tp = params['task_parameters']
    assert tp['home'] == {'type': 'joint', 'named': 'home', 'motion': 'ptp', 'planner': 'pilz',
                          'profile': 'transit', 'speed': 50, 'accel': 50}
    assert tp['w2'] == {'type': 'tcp', 'position': [-0.45, -0.25, 0.1],
                        'orientation': [0.0, 1.0, 0.0, 0.0], 'frame': 'base', 'motion': 'lin',
                        'profile': 'approach', 'speed': 10, 'accel': 30}
    assert tp['w3']['aux'] == [-0.45, 0.0, 0.05] and tp['w3']['aux_is_center'] is False
    assert tp['w2b'] == {'type': 'tcp', 'position': [-0.45, 0.0, 0.05],
                         'orientation': [0.0, 1.0, 0.0, 0.0], 'frame': 'base'}
    assert params['planner_mode'] == 'ompl' and params['process_path_backend'] == 'pilz_sequence'
    assert "# w2 = w1 + (0, 0, 0.2) m along w1's tool axes" in text
    assert "# task 'validation_a' steps 1, 6, 9: PTP, profile transit, provider pilz_ptp" in text
    assert "# also: via point of task 'validation_a' step 4; via point of path 'sweep'" in text \
        or "via point" in text


def test_h_a_waypoint_reached_by_two_kinds_of_move_gets_a_second_entry():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][6]['move'].update({'type': 'lin', 'profile': 'approach'})
    p = build(mutate)
    plan = plan_program(p)
    assert plan.entries['w1']['motion'] == 'ptp'
    assert plan.entries['w1__2']['motion'] == 'lin'
    assert plan.entries['w1__2']['position'] == plan.entries['w1']['position']
    assert plan.move_entries[('validation_a', '7')] == 'w1__2'   # vNext M3.7: a step's label
    from trainit_setup_assistant.applications.task_program import program_tree_xml
    assert '<MoveWaypoint waypoint="w1__2"/>' in program_tree_xml(p, plan)


def test_i_the_tree_is_one_node_per_step_and_one_followpath_per_path(bundle):
    import xml.etree.ElementTree as ET
    root = ET.parse(bundle['root'] / APP / 'bt_trees' / 'task_program.xml').getroot()
    trees = {t.get('ID'): t for t in root.findall('BehaviorTree')}
    assert list(trees) == ['MainTree', 'task_validation_a']
    sub = trees['MainTree'].find('Sequence/SubTree')
    assert sub.get('ID') == 'task_validation_a' and sub.get('_autoremap') == 'true'
    steps = list(trees['task_validation_a'].find('Sequence'))
    assert [s.tag for s in steps] == ['MoveWaypoint'] * 7 + ['FollowPath', 'MoveWaypoint']
    assert steps[7].attrib == {'path': 'sweep'}
    assert [s.get('waypoint') for s in steps if s.tag == 'MoveWaypoint'] == [
        'home', 'w1', 'w2', 'w3', 'w4', 'home', 'w1', 'home']
    assert root.get('main_tree_to_execute') == 'MainTree'


def test_n_the_motion_profiles_file_and_the_launch_point_tmr_at_the_bundle(bundle):
    app = bundle['root'] / APP
    profiles = yaml.safe_load((app / 'config' / 'motion_profiles.yaml').read_text())
    assert profiles == {'motion_profiles': {'transit': {}, 'approach': {}, 'process': {}}}
    launch = (app / 'launch' / 'trainit_bt.launch.py').read_text()
    assert 'os.path.join(pkg, "config", "motion_profiles.yaml")' in launch
    assert '"paths_file": paths,' in launch and 'trainit_profiles.yaml' not in launch
    assert 'DeclareLaunchArgument("planner_mode", default_value="ompl"' in launch


def test_n_runtime_allowed_start_tolerance_reaches_move_group(bundle):
    bringup = (bundle['root'] / APP / 'launch' / 'bringup.launch.py').read_text()
    assert ('trajectory_execution={"trajectory_execution.allowed_start_tolerance": 0.05}'
            in bringup)


def test_n_the_readme_reads_as_the_program(bundle):
    readme = (bundle['root'] / APP / 'README.md').read_text()
    for text in ('# ur5e_program_app: a Canonical Task Program', '**Main task:** `validation_a`',
                 '| 8 | path | `sweep` | ONE motion, provider pilz_sequence (the registry default '
                 'for path) | 8 % / 20 % |',
                 "| `w2` | pose [-0.45, -0.25, 0.1] m, [0.0, 1.0, 0.0, 0.0] (= w1 + (0, 0, 0.2) m "
                 "along w1's tool axes) | `base` |",
                 '| `sweep` | w1 -> w2 -> w3 -> w4 | lin (blend 0.02 m); circ via w2b (blend '
                 '0.02 m); lin | blend |',
                 'trajectory_execution.allowed_start_tolerance`): 0.05 rad'):
        assert text in readme, text
    assert str(Path.home()) not in readme


# --- J. TMR floors --------------------------------------------------------------------------------

def test_j_a_task_program_requires_tmr_2_2_0_for_what_it_uses(bundle):
    xml = (bundle['root'] / APP / 'package.xml').read_text()
    assert ('<!-- trainit_motion_runtime >= 2.2.0 for: FollowPath (a path of config/paths.yaml); '
            "MoveWaypoint's profile, frame and tcp keys -->") in xml
    assert '<exec_depend version_gte="2.2.0">trainit_motion_runtime</exec_depend>' in xml


def test_j_the_floor_follows_the_files_not_the_application_type():
    tree = ('<root BTCPP_format="4"><BehaviorTree ID="MainTree"><Sequence><Sleep msec="5"/>'
            '</Sequence></BehaviorTree></root>')
    assert requirement(use_of(trees=[tree], launches=[])).version == '1.4.0'
    with_path = tree.replace('<Sleep msec="5"/>', '<FollowPath path="p"/>')
    assert requirement(use_of(trees=[with_path], launches=[])).version == '2.2.0'
    params = ('/**:\n  ros__parameters:\n    task_parameters:\n      a: {type: tcp, frame: x}\n')
    assert 'waypoint_canonical_keys' in use_of(trees=[tree], bt_params=[params]).capabilities


def test_j_a_template_file_that_does_not_parse_keeps_its_old_floor():
    broken = '<root><BehaviorTree ID="MainTree"><MoveWaypoint waypoint="a"'
    legacy = requirement(use_of(trees=[broken], launches=[]))
    assert legacy.version == '1.7.0'                     # unchanged by the M3.3 rows
    program = requirement(use_of(trees=[broken + ' <SubTree ID="task_x"'], launches=[]))
    assert program.version == '2.5.0'                    # the newest task-program row (vNext M3.6)


# --- O. refused before generation ---------------------------------------------------------------

def test_o_an_unsupported_request_is_refused_before_any_file(tmp_path):
    needs_ur_description()

    def mutate(app):
        app['paths'][0]['provider'] = 'cartesian_waypoints'
    out = tmp_path / 'refused'
    with pytest.raises(ValueError) as exc:
        Orchestrator().generate(build(mutate), out)
    assert "R-M3-10 REFUSED: path 'sweep' (path): provider 'cartesian_waypoints'" in str(exc.value)
    assert not out.exists()


# --- the cartesian_waypoints backend (M3.2 debt S2) and the limits file (N9) -----------------------

def _dense(app, continuity, provider=None):
    base_link_frames(app)
    path = {'name': 'bead', 'source': {'generator': {'type': 'polyline',
                                                     'points': [[0.45, 0.25, 0.3],
                                                                [0.45, 0.0, 0.3],
                                                                [0.45, -0.25, 0.3]]},
                                       'orientation': [1.0, 0.0, 0.0, 0.0]},
            'continuity': continuity, 'profile': 'transit'}
    if provider:
        path['provider'] = provider
    app['paths'].append(path)
    app['waypoints'].append({'name': 'bead0', 'position': [0.45, 0.25, 0.3],
                             'orientation': [1.0, 0.0, 0.0, 0.0]})
    app['tasks'][0]['steps'] += [{'move': {'to': 'bead0', 'type': 'ptp', 'profile': 'transit'}},
                                 {'path': 'bead'}]


def test_s2_a_continuous_dense_path_sets_the_runners_backend():
    plan = plan_program(build(lambda app: _dense(app, 'blend')))
    bead = next(p for p in plan.paths if p.name == 'bead')
    assert bead.resolution.provider == 'cartesian_waypoints'
    assert plan.backend == 'cartesian_waypoints'
    assert bead.data['source']['generator'] == {
        'type': 'polyline', 'points': [[0.45, 0.25, 0.3], [0.45, 0.0, 0.3], [0.45, -0.25, 0.3]]}


def test_s2_a_dense_stop_path_keeps_the_sequence_backend():
    plan = plan_program(build(lambda app: _dense(app, 'stop')))
    assert next(p for p in plan.paths if p.name == 'bead').resolution.provider == 'pilz_sequence'
    assert plan.backend == 'pilz_sequence'


def test_s2_the_generated_bt_params_and_the_provider_agree(tmp_path):
    needs_ur_description()
    out = tmp_path / 'dense'
    Orchestrator().generate(build(lambda app: _dense(app, 'blend')), out)
    params = yaml.safe_load((out / APP / 'config' / 'bt_params.yaml').read_text())
    assert params['/**']['ros__parameters']['process_path_backend'] == 'cartesian_waypoints'
    paths = yaml.safe_load((out / APP / 'config' / 'paths.yaml').read_text())['paths']
    assert paths['bead']['provider'] == 'cartesian_waypoints'


def test_n9_a_tcp_limit_is_derived_from_the_limits_the_bundle_ships(bundle):
    shipped = yaml.safe_load((bundle['root'] / 'ur5e_program_trainit_config' / 'config' /
                              'pilz_cartesian_limits.yaml').read_text())['cartesian_limits']
    from trainit_setup_assistant.model.capabilities import project_cartesian_limits
    used = project_cartesian_limits(bundle['project'])
    assert (used.max_trans_vel, used.max_trans_acc) == (shipped['max_trans_vel'],
                                                        shipped['max_trans_acc'])


# --- K/L. round trip and the wizard -------------------------------------------------------------

def test_k_a_task_program_round_trips_without_loss(tmp_path):
    p = load_project(PROGRAM_FILE)
    saved = save_project(p, tmp_path / 'project.yaml')
    again = load_project(saved)
    assert again == p
    assert yaml.safe_load(saved.read_text())['application'] == raw()['application']
    assert save_project(again, tmp_path / 'twice.yaml').read_text() == saved.read_text()


def test_k_the_template_editors_refuse_a_task_program():
    from trainit_setup_assistant.gui.controller import (TASK_PROGRAM_READ_ONLY,
                                                        AssistantController)
    ctrl = AssistantController(load_project(PROGRAM_FILE))
    before = project_to_dict(ctrl.project)
    for call in (lambda: ctrl.set_application('pick_and_place'),
                 lambda: ctrl.add_move('x', position=[0, 0, 0], orientation=[0, 0, 0, 1]),
                 lambda: ctrl.clear_application(),
                 lambda: ctrl.set_loop(2),
                 lambda: ctrl.add_tool_action('w1', 'grasp'),
                 lambda: ctrl.bind_relative('w2', 'w1', dz=0.1),
                 lambda: ctrl.set_bin_picking(tries=2)):
        with pytest.raises(ValueError, match='Canonical Task Program'):
            call()
    assert project_to_dict(ctrl.project) == before
    assert 'YAML-first' in TASK_PROGRAM_READ_ONLY


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('python_qt_binding', reason='the wizard tests need python_qt_binding')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_l_the_wizard_shows_a_task_program_and_leaves_it_as_written(qapp, tmp_path):
    needs_ur_description()               # the overview validates the program (vNext M3.7)
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(PROGRAM_FILE)
    before = project_to_dict(ctrl.project)
    w = SetupWizard(ctrl)
    page = w.application_page
    page.initializePage()
    assert not page.app_type.isVisibleTo(page) and page.program_box.isVisibleTo(page)
    assert 'Application type: task_program' in page.program_state.text()
    assert str(PROGRAM_FILE) in page.subTitle()
    assert page.validatePage()
    blocks = w.blocks_page
    blocks.initializePage()
    assert not blocks.editor.isVisibleTo(blocks) and blocks.program_panel.isVisibleTo(blocks)
    assert blocks.title() == 'Step 7 — Application Composer'          # vNext I0: no longer read-only
    assert '8  path  sweep' in blocks.program_view.toPlainText()
    assert blocks.validatePage()
    blocks.validate_program()
    assert blocks.program_result.text().startswith('Valid:')
    assert project_to_dict(ctrl.project) == before
    assert load_project(ctrl.save(tmp_path / 'saved.yaml')) == load_project(PROGRAM_FILE)


def test_l_the_wizard_shows_the_refusals_of_an_invalid_program(qapp):
    needs_ur_description()               # the overview validates the program (vNext M3.7)
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard

    def mutate(app):
        base_link_frames(app)
        app['paths'][0]['provider'] = 'cartesian_waypoints'
    ctrl = AssistantController(build(mutate))
    w = SetupWizard(ctrl)
    w.blocks_page.initializePage()
    w.blocks_page.validate_program()
    text = w.blocks_page.program_result.text()
    assert text.startswith('1 problem -- Generate would refuse the program:')
    assert "provider 'cartesian_waypoints' does not provide" in text
    assert 'c62828' in w.blocks_page.program_result.styleSheet()


def test_l_a_template_project_keeps_its_editors(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController(load_project(EXAMPLE / 'project.yaml'))
    w = SetupWizard(ctrl)
    w.application_page.initializePage()
    assert w.application_page.app_type.isVisibleTo(w.application_page)
    assert not w.application_page.program_box.isVisibleTo(w.application_page)
    w.blocks_page.initializePage()
    assert w.blocks_page.editor.isVisibleTo(w.blocks_page)
    assert not w.blocks_page.program_panel.isVisibleTo(w.blocks_page)
    assert w.blocks_page.title().startswith('Step 7 — Application blocks')


def test_l_the_overview_reads_like_a_robot_program():
    text = program_overview(load_project(PROGRAM_FILE).application.program)
    assert '    4  move  CIRC  -> w3           via w2b, profile approach' in text
    assert '  w2    relative to w1: (0, 0, 0.2) m along the tool axes' in text


def test_the_registry_has_the_task_program_application():
    assert get_application('task_program').tree_filename(None) == 'task_program.xml'


def test_the_math_of_a_tool_frame_offset():
    from trainit_setup_assistant.applications.task_program import _rotate
    # tool down (180 deg about x): +z of the tool is -z of the base
    assert _rotate((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.2)) == pytest.approx((0.0, 0.0, -0.2))
    q = (0.0, 0.0, math.sin(math.pi / 4), math.cos(math.pi / 4))      # 90 deg about z
    assert _rotate(q, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert copy.deepcopy(q) == q


def test_the_readme_example_is_the_example_file():
    """TSA's README shows the example's application block: the same program, never a second one."""
    readme = (PKG / 'README.md').read_text()
    section = readme.split('## A Canonical Task Program (YAML)', 1)[1]          # vNext I0: no longer YAML-first
    block = section.split('```yaml\n', 1)[1].split('```', 1)[0]
    shown = yaml.safe_load(block)['application']
    real = raw()['application']
    real['paths'][0].pop('goal_tolerance')          # the two keys the README names as left out
    real.pop('runtime')
    assert shown == real


def test_review_an_inline_via_is_written_as_floats():
    """A ROS parameter list is one type: an inline via of integers must not become [0, 0.1, ...]."""
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][3]['move']['via'] = [0, 0, 1]
    p = build(mutate)
    assert p.application.program.tasks[0].steps[3].move.via == [0.0, 0.0, 1.0]
    assert all(isinstance(v, float) for v in p.application.program.tasks[0].steps[3].move.via)


def test_review_a_motion_type_on_a_waypoint_is_named():
    with pytest.raises(ValueError, match="a waypoint's type is tcp | joint"):
        build(lambda app: app['waypoints'][1].update(type='ptp'))


def test_review_a_refusal_names_where_its_limits_come_from():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][1]['move']['profile'] = 'approach'     # tcp limit on a ptp move
    text = problems(build(mutate))
    assert ("limits:   from the step's motion profile 'approach' (speed, acceleration, "
            "tcp_velocity_mps)") in text


def test_review_a_variant_entry_says_what_differs():
    def mutate(app):
        base_link_frames(app)
        app['tasks'][0]['steps'][6]['move']['profile'] = 'approach'
        app['motion_profiles']['approach'] = {'speed': 30, 'acceleration': 30}
    plan = plan_program(build(mutate))
    assert plan.entry_notes['w1__2'][0] == "'w1' again, reached by a move with another profile, speed, accel"


def test_review_a_joint_tolerance_on_an_unused_profile_is_refused():
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['spare'] = {'speed': 5, 'acceleration': 5,
                                           'goal_tolerance': {'joint_rad': 0.01}}
    assert "motion profile 'spare': goal_tolerance.joint_rad" in problems(build(mutate))


def test_review_the_accepted_keys_do_not_list_reserved_fields(tmp_path):
    text = load_text(tmp_path, lambda d: d['application']['tasks'][0]['steps'][0]['move'].update(
        {'sped': 5}))
    accepted = text.split('accepted: ', 1)[1]
    assert 'jerk' not in accepted and 'duration_s' not in accepted


def test_review_a_relative_cycle_is_reported_once():
    def mutate(app):
        base_link_frames(app)
        app['waypoints'] += [{'name': 'c1', 'relative': {'to': 'c2', 'frame': 'tool'}},
                             {'name': 'c2', 'relative': {'to': 'c1', 'frame': 'tool'}}]
    assert problems(build(mutate)).count('form a cycle') == 1


def test_review_no_exponent_notation_in_generated_numbers():
    from trainit_setup_assistant.generator.program_files import _num
    assert _num(1e-07) == '0.0000001' and _num(0.08) == '0.08' and _num(50) == '50'


# --- vNext M3.4: registry parity (goal_tolerance on the MoveGroupInterface rows, TMR 2.3.0) -------

def test_m3_4_a_lin_or_ptp_move_may_carry_a_profile_tolerance_now():
    """TMR 2.3.0 honours a move's goal tolerance on the MoveGroupInterface routes, and its registry
    says so (pilz_ptp, ompl, ompl_chomp, pilz_lin provide goal_tolerance): TSA, reading that registry,
    accepts it and writes it into the profile TMR loads."""
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['precise'] = {'speed': 30, 'acceleration': 30,
                                             'goal_tolerance': {'position_m': 0.002, 'orientation_rad': 0.02}}
        app['tasks'][0]['steps'][2]['move']['profile'] = 'precise'     # lin to w2
        app['tasks'][0]['steps'][1]['move']['profile'] = 'precise'     # ptp to the pose w1
    plan = plan_program(build(mutate))
    assert plan.profiles['precise'] == {'goal_position_tolerance': 0.002, 'goal_orientation_tolerance': 0.02}


def test_m3_4_a_tolerance_on_a_move_to_a_joint_target_stays_refused():
    """A joint goal has no Cartesian tolerance (TMR refuses one at run time): TSA refuses it before
    any file, as the registry's 2.2.0 rows did (R-M3-10 then, named R-M3-12 now)."""
    def mutate(app):
        base_link_frames(app)
        app['motion_profiles']['precise'] = {'speed': 50, 'acceleration': 50,
                                             'goal_tolerance': {'position_m': 0.002}}
        app['tasks'][0]['steps'][0]['move']['profile'] = 'precise'     # ptp to home (named state)
    text = problems(build(mutate))
    assert 'R-M3-12' in text and "move to a joint target ('home')" in text


def test_m3_4_a_move_tolerance_raises_the_floor_to_tmr_2_3_0():
    tree = ('<root BTCPP_format="4"><BehaviorTree ID="MainTree"><Sequence><MoveWaypoint waypoint="a"/>'
            '</Sequence></BehaviorTree></root>')
    profiles = 'motion_profiles:\n  tol: {goal_position_tolerance: 0.002}\n  plain: {}\n'

    def floor(entry: str) -> str:
        params = '/**:\n  ros__parameters:\n    task_parameters:\n      a: ' + entry + '\n'
        return requirement(use_of(trees=[tree], bt_params=[params], profiles=[profiles])).version

    assert floor('{type: tcp, motion: lin, profile: tol}') == '2.3.0'
    assert floor('{type: tcp, motion: ptp, profile: tol}') == '2.3.0'
    assert floor('{type: tcp, profile: tol}') == '2.3.0'            # no motion key = free
    assert floor('{type: tcp, motion: circ, profile: tol}') == '2.2.0'   # the sequence route always sent it
    assert floor('{type: tcp, motion: lin, profile: plain}') == '2.2.0'
    assert floor('{type: tcp, motion: lin}') == '1.4.0'              # a legacy waypoint: no profile key


def test_m3_4_validation_a_keeps_its_floor(bundle):
    """Validation A names no goal tolerance on a move: its floor is still 2.2.0 (bundle unchanged)."""
    xml = (bundle['root'] / APP / 'package.xml').read_text()
    assert '<exec_depend version_gte="2.2.0">trainit_motion_runtime</exec_depend>' in xml
