"""vNext I0.0 (I0_DESIGN.md §20, §21, §31): the Application Composer's foundations.

- **Project-relative paths:** a project resolves its paths beside its own file, whatever the current
  directory. The cwd is a deprecated fallback. The file is never rewritten.
- **Located diagnostics:** the ONE validator's lines, parsed and kept verbatim.
- **The Composer's model operations:** every edit goes through the canonical model. An edit the model
  refuses leaves the project unchanged. Every M3 program is re-authorable through the operations alone,
  into the SAME project and the SAME bundle.
- **LiveCapture** reads the joint states in the robot's namespace (ADR-0023).
"""
import copy
import filecmp
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import ProgramRefusals, plan_program
from trainit_setup_assistant.composer import Composer, ComposerError, parse, parse_load_error, walk
from trainit_setup_assistant.composer.diagnostics import Location, locate
from trainit_setup_assistant.composer.ops import _slot, step_kind
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import CanonicalProject, load_project, project_to_dict, save_project
from trainit_setup_assistant.model.paths import (CWD_DEPRECATION, resolve_path,
                                                  resolve_project_paths)

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / 'test' / 'data'
SAMPLE = PKG / 'examples' / 'ur5e_suction'
COMPOSITION = DATA / 'm3_7' / 'composition.yaml'
# Every in-repository M3 task program: the public sample, the M3.5 / M3.6 validations, the M3.7
# composition (the six decompositions are merged into these cells in test_m3_7_decompositions).
PROGRAMS = (SAMPLE / 'task_program.yaml', DATA / 'm3_5' / 'validation_b.yaml',
            DATA / 'm3_6' / 'validation_c.yaml', DATA / 'm3_6' / 'validation_c_detections.yaml',
            COMPOSITION)
DECOMPOSITIONS = sorted((DATA / 'm3_7' / 'decompositions').glob('*.yaml'))
# the owner-acceptance projects of M3 (I0's acceptance fixtures), outside every repository: opt-in
ACCEPTANCE = Path(os.environ.get('TRAINIT_M3_ACCEPTANCE',
                                 Path.home() / 'fr5_ws' / 'forensics' / 'vnext-m3' / 'm3_acceptance'))


@pytest.fixture(autouse=True)
def _ur5e_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


@pytest.fixture
def elsewhere(tmp_path, monkeypatch):
    """A current directory that holds none of the fixtures' relative paths."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


def load(path) -> CanonicalProject:
    return load_project(path)


def resolved(path) -> CanonicalProject:
    project, notes = resolve_project_paths(load_project(path), path)
    assert notes == []
    return project


def decomposition_project(path: Path) -> dict:
    """A decomposition merged into its cell, as test_m3_7_decompositions builds it."""
    base = SAMPLE / 'project.yaml' if path.stem == 'd_pick_and_place' else COMPOSITION
    data = yaml.safe_load(base.read_text())
    dec = yaml.safe_load(path.read_text())
    data['application'] = dec['application']
    data.pop('perception', None)
    if 'perception' in dec:
        data['perception'] = dec['perception']
    data['schema_version'] = 4
    return data


# ---- project-relative paths -------------------------------------------------------------------------

def test_a_relative_path_resolves_beside_the_project_file(tmp_path):
    (tmp_path / 'proj' / 'urdf').mkdir(parents=True)
    (tmp_path / 'cwd').mkdir()
    value, note = resolve_path('urdf', tmp_path / 'proj', cwd=tmp_path / 'cwd')
    assert value == str((tmp_path / 'proj' / 'urdf').resolve()) and note is None


def test_the_current_directory_is_a_deprecated_fallback(tmp_path):
    (tmp_path / 'proj').mkdir()
    (tmp_path / 'cwd' / 'urdf').mkdir(parents=True)
    value, note = resolve_path('urdf', tmp_path / 'proj', cwd=tmp_path / 'cwd')
    assert value == str((tmp_path / 'cwd' / 'urdf').resolve()) and note == CWD_DEPRECATION


def test_a_path_found_nowhere_resolves_beside_the_project_so_the_refusal_names_it(tmp_path):
    value, note = resolve_path('urdf', tmp_path / 'proj', cwd=tmp_path)
    assert value == str((tmp_path / 'proj' / 'urdf').resolve()) and note is None


@pytest.mark.parametrize('value', ['/abs/urdf', 'package://pkg/urdf', '$(find pkg)/worlds/w.world', '~/x', '',
                                   None])
def test_absolute_uri_substitution_and_empty_values_are_left_as_written(tmp_path, value):
    assert resolve_path(value, tmp_path, cwd=tmp_path) == (value, None)


def test_resolution_returns_a_copy_and_the_file_keeps_the_path_as_written(elsewhere):
    project = load(COMPOSITION)
    copy_, notes = resolve_project_paths(project, COMPOSITION)
    assert notes == []
    assert project.robot.description.urdf_dir == '../../../examples/ur5e_suction/urdf'
    assert copy_.robot.description.urdf_dir == str((SAMPLE / 'urdf').resolve())
    out = elsewhere / 'saved.yaml'
    save_project(project, out)
    assert yaml.safe_load(out.read_text())['robot']['description']['urdf_dir'] == \
        '../../../examples/ur5e_suction/urdf'


def test_the_cli_generates_from_any_directory_the_same_bundle(elsewhere, monkeypatch):
    """The project's paths are relative to its file: generating from another directory gives the
    bundle generated from the file's own directory (TSA 8.0 needed the latter)."""
    from trainit_setup_assistant.cli.generate import main
    assert main([str(COMPOSITION), '-o', str(elsewhere / 'from_elsewhere')]) == 0
    monkeypatch.chdir(COMPOSITION.parent)
    assert main([str(COMPOSITION), '-o', str(elsewhere / 'from_beside')]) == 0
    assert_same_tree(elsewhere / 'from_elsewhere', elsewhere / 'from_beside')


# ---- located diagnostics ------------------------------------------------------------------------------

def test_a_diagnostic_is_the_canonical_line_verbatim():
    line = ("R-M3-03: task 'transfer' step 2 (release at 'plce'): no waypoint is named 'plce' "
            '(waypoints: home, pick, place)')
    d = parse(line)
    assert str(d) == line and d.code == 'R-M3-03' and d.severity == 'error'
    assert d.location == Location('step', 'transfer', (2,))


@pytest.mark.parametrize('line, where', [
    ("R-M3-05: waypoint 'w2' (relative to 'w1'): 'w1' is a joint target", Location('waypoint', 'w2')),
    ("R-M3-04: path 'sweep' segment 1 (w2 -> w3): a circ segment needs exactly one of via or center",
     Location('path', 'sweep', segment=1)),
    ("R-M3-09: path 'scan' event 0 (capture after arrival at 'p1'): x", Location('path', 'scan', event=0)),
    ("R-M3-15: task 'cell' step 4.2.1 (task 'go_home'): no task is named 'go_home'",
     Location('step', 'cell', (4, 2, 1))),
    ("R-M3-03: task 'pick' input 'cubes' (candidates): no step of the task reads it",
     Location('task_data', 'pick', io='input', item='cubes')),
    ("R-M3-12: motion profile 'slow': goal_tolerance.joint_rad: x", Location('profile', 'slow')),
    ("R-M3-15: task call cycle: a -> b -> a", Location()),
    ('R13: the contract names no actuator', Location()),
])
def test_every_canonical_place_is_located(line, where):
    assert locate(line) == where


def _refusals(mutate) -> list:
    data = yaml.safe_load(COMPOSITION.read_text())
    mutate(data['application'])
    project, _ = resolve_project_paths(CanonicalProject.parse_obj(data), COMPOSITION)
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return [parse(p) for p in exc.value.problems], project


def _resolves(project, loc: Location) -> bool:
    doc = Composer(project).document()
    if loc.kind == 'step':
        task = next(t for t in doc['tasks'] if t['name'] == loc.name)
        steps, i = _slot(copy.deepcopy(task), loc.label)
        return i < len(steps)
    if loc.kind == 'waypoint':
        return loc.name in {w['name'] for w in doc.get('waypoints', [])}
    if loc.kind == 'path':
        return loc.name in {p['name'] for p in doc.get('paths', [])}
    if loc.kind in ('task', 'task_data'):
        return loc.name in {t['name'] for t in doc['tasks']}
    if loc.kind == 'profile':
        return loc.name in doc.get('motion_profiles', {})
    return True


def _misspell_every_move_target(app):
    for t in app['tasks']:
        for _label, step in walk(t['steps']):
            if step_kind(step) == 'move':
                step['move']['to'] = step['move']['to'] + 'x'


def _unknown_profile_everywhere(app):
    for t in app['tasks']:
        for _label, step in walk(t['steps']):
            body = step[step_kind(step)]
            if isinstance(body, dict) and 'profile' in body:
                body['profile'] = 'nonexistent'


@pytest.mark.parametrize('mutate', [_misspell_every_move_target, _unknown_profile_everywhere],
                         ids=['misspelled_targets', 'unknown_profiles'])
def test_every_real_refusal_locates_an_element_that_exists(mutate):
    diags, project = _refusals(mutate)
    assert diags
    for d in diags:
        assert d.location.kind != 'project', str(d)
        assert _resolves(project, d.location), str(d)


def test_a_load_refusal_is_located_line_by_line():
    data = yaml.safe_load(COMPOSITION.read_text())
    data['application']['tasks'][0]['steps'].insert(1, {'move': {'to': 'home', 'type': 'warp'}})
    with pytest.raises(ValueError) as exc:
        from trainit_setup_assistant.model.program import describe_load_error
        try:
            CanonicalProject.parse_obj(data)
        except ValueError as e:
            raise ValueError(describe_load_error(e, data, 'p.yaml')) from None
    diags = parse_load_error(str(exc.value))
    task = data['application']['tasks'][0]['name']
    assert any(d.location == Location('step', task, (2,)) for d in diags), [str(d) for d in diags]


# ---- the Composer's operations -----------------------------------------------------------------------

def test_a_no_op_edit_keeps_the_project_byte_for_byte(elsewhere):
    for path in PROGRAMS:
        project = load(path)
        c = Composer(project)
        c.edit(lambda doc: None)
        assert project_to_dict(c.project) == project_to_dict(project), path.name
        a, b = elsewhere / 'a.yaml', elsewhere / 'b.yaml'
        save_project(project, a)
        save_project(c.project, b)
        assert a.read_bytes() == b.read_bytes(), path.name


def test_a_refused_edit_leaves_the_project_unchanged_and_is_located():
    c = Composer(load(COMPOSITION))
    before = project_to_dict(c.project)
    task = c.task_names()[0]
    with pytest.raises(ComposerError) as exc:
        c.replace_step(task, (1,), {'move': {'to': 'home', 'type': 'warp'}})
    assert project_to_dict(c.project) == before and not c.can_undo
    assert any(d.location == Location('step', task, (1,)) for d in exc.value.diagnostics)


def test_a_step_is_one_kind_and_a_task_is_never_empty():
    c = Composer(load(COMPOSITION))
    with pytest.raises(ComposerError):
        c.insert_step(c.task_names()[0], (1,), {'move': {'to': 'home', 'type': 'ptp'}, 'dwell': {'ms': 1}})
    with pytest.raises(ComposerError, match='has no steps'):
        c.add_task('empty', steps=[])


def test_rename_a_waypoint_renames_every_reference():
    c = Composer(load(COMPOSITION))
    old = next(w for w in c.waypoint_names() if c.waypoint_users(w))
    users = c.waypoint_users(old)
    c.rename_waypoint(old, 'renamed_wp')
    assert old not in c.waypoint_names() and c.waypoint_users(old) == []
    assert c.waypoint_users('renamed_wp') == users
    plan_program(resolve_project_paths(c.project, COMPOSITION)[0])    # still valid


def test_a_used_waypoint_profile_path_or_task_is_not_deleted():
    c = Composer(load(COMPOSITION))
    used_wp = next(w for w in c.waypoint_names() if c.waypoint_users(w))
    with pytest.raises(ComposerError, match='is used by'):
        c.delete_waypoint(used_wp)
    with pytest.raises(ComposerError, match='main task'):
        c.delete_task(c.program.main_task)
    for name in c.program.motion_profiles:
        if c.profile_users(name):
            with pytest.raises(ComposerError, match='is used by'):
                c.delete_profile(name)
            break


def test_wrap_move_and_unwrap_keep_canonical_nesting():
    c = Composer(load(COMPOSITION))
    task = c.program.main_task
    first, second = c.step(task, (1,)), c.step(task, (2,))
    c.wrap_steps(task, (1,), 2, 'repeat', times=3)
    assert c.step(task, (1,)) == {'repeat': {'cycles': 3, 'body': [first, second]}}
    assert c.step(task, (1, 2)) == second
    c.set_control_count(task, (1,), 4)
    assert c.step(task, (1,))['repeat']['cycles'] == 4
    c.unwrap_step(task, (1,))
    assert c.step(task, (1,)) == first and c.step(task, (2,)) == second
    assert c.move_step(task, (2,), -1) == (1,)
    assert c.step(task, (1,)) == second
    c.undo()
    assert c.step(task, (1,)) == first
    c.redo()
    assert c.step(task, (1,)) == second


def test_a_fallback_is_created_whole_and_its_alternatives_are_editable():
    c = Composer(load(COMPOSITION))
    task = c.program.main_task
    first = c.step(task, (1,))
    with pytest.raises(ComposerError):
        c.wrap_steps(task, (1,), 1, 'fallback')                 # one alternative: refused by the model
    alt = [{'dwell': {'ms': 10}}]
    c.wrap_steps(task, (1,), 1, 'fallback', alternatives=[alt])
    assert c.step(task, (1, 1, 1)) == first and c.step(task, (1, 2, 1)) == alt[0]
    c.insert_step(task, (1, 2, 1), {'dwell': {'ms': 5}})
    assert c.step(task, (1, 2, 1)) == {'dwell': {'ms': 5}} and c.step(task, (1, 2, 2)) == alt[0]
    c.add_alternative(task, (1,), [{'dwell': {'ms': 7}}])
    assert c.step(task, (1, 3, 1)) == {'dwell': {'ms': 7}}
    c.unwrap_step(task, (1,), keep=1)
    assert c.step(task, (1,)) == first


def _author_steps(c: Composer, task: str, at: tuple, steps: list) -> None:
    """Author ``steps`` into a task from label ``at`` with the operations only: a control step is
    authored as its first body's steps, then wrapped (a fallback with its further alternatives)."""
    position = at[-1]
    for step in steps:
        here = at[:-1] + (position,)
        kind = step_kind(step)
        if kind in ('repeat', 'retry'):
            body = step[kind]['body']
            _author_steps(c, task, here, body)
            c.wrap_steps(task, here, len(body), kind,
                         times=step[kind]['cycles' if kind == 'repeat' else 'attempts'])
            extra = {k: v for k, v in step[kind].items() if k not in ('body', 'cycles', 'attempts')}
            assert not extra, extra
        elif kind == 'fallback':
            bodies = step['fallback']['bodies']
            _author_steps(c, task, here, bodies[0])
            c.wrap_steps(task, here, len(bodies[0]), 'fallback', alternatives=bodies[1:])
        else:
            c.insert_step(task, here, step)
        position += 1


def reauthor(project: CanonicalProject) -> CanonicalProject:
    """The program of ``project`` re-built with the Composer's operations, from a program that has only
    the main task's first step."""
    target = Composer(project).document()
    main = next(t for t in target['tasks'] if t['name'] == target['main_task'])
    start = project_to_dict(project)
    start['application'] = {'type': 'task_program',
                            'tasks': [{'name': main['name'], 'steps': [main['steps'][0]]}],
                            'main_task': main['name']}
    c = Composer(CanonicalProject.parse_obj(start))
    for name, spec in (target.get('motion_profiles') or {}).items():
        c.set_profile(name, spec)
    for wp in target.get('waypoints') or []:
        c.set_waypoint(wp)
    for path in target.get('paths') or []:
        c.set_path(path)
    for t in target['tasks']:
        if t['name'] == main['name']:
            _author_steps(c, t['name'], (2,), t['steps'][1:])      # its first step started the program
        else:
            c.add_task(t['name'], steps=[t['steps'][0]], description=t.get('description'))
            _author_steps(c, t['name'], (2,), t['steps'][1:])
        if t.get('description') and t['name'] == main['name']:
            c.set_task_description(t['name'], t['description'])
        if t.get('inputs') or t.get('outputs'):
            c.set_task_data(t['name'], t.get('inputs'), t.get('outputs'))
    for key in ('providers', 'runtime'):
        if key in target:
            c.edit(lambda doc, k=key: doc.__setitem__(k, copy.deepcopy(target[k])))
    return c.project


def _ordered(d):
    return yaml.safe_dump(d, sort_keys=False)


@pytest.mark.parametrize('path', PROGRAMS, ids=lambda p: p.stem)
def test_every_m3_program_is_re_authored_into_the_same_project(path):
    project = load(path)
    again = reauthor(project)
    assert _ordered(project_to_dict(again)) == _ordered(project_to_dict(project))


@pytest.mark.parametrize('path', DECOMPOSITIONS, ids=lambda p: p.stem)
def test_every_m3_7_decomposition_is_re_authored_into_the_same_project(path):
    project = CanonicalProject.parse_obj(decomposition_project(path))
    assert _ordered(project_to_dict(reauthor(project))) == _ordered(project_to_dict(project))


def assert_same_tree(a: Path, b: Path) -> None:
    cmp = filecmp.dircmp(a, b)
    stack = [cmp]
    while stack:
        d = stack.pop()
        assert not (d.left_only or d.right_only or d.diff_files or d.funny_files), \
            (d.left, d.left_only, d.right_only, d.diff_files)
        _match, mismatch, errors = filecmp.cmpfiles(d.left, d.right, d.common_files, shallow=False)
        assert not mismatch and not errors, (d.left, mismatch, errors)
        stack.extend(d.subdirs.values())


@pytest.mark.parametrize('path', [COMPOSITION, DATA / 'm3_5' / 'validation_b.yaml'], ids=lambda p: p.stem)
def test_a_re_authored_program_generates_the_same_bundle(path, tmp_path):
    original = resolved(path)
    again, _ = resolve_project_paths(reauthor(load(path)), path)
    Orchestrator().generate(original, tmp_path / 'yaml')
    Orchestrator().generate(again, tmp_path / 'composer')
    assert_same_tree(tmp_path / 'yaml', tmp_path / 'composer')


@pytest.mark.parametrize('name', ['test_c/m3b_fr3wml.yaml', 'test_d/m3c_fr3wml.yaml', 'test_e/m3e_fr3wml.yaml',
                                  'test_f/m3f_fr3wml.yaml'])
def test_the_m3_owner_acceptance_projects_are_re_authored_into_the_same_bundle(name, tmp_path):
    """I0's acceptance fixtures (M3_OWNER_ACCEPTANCE.md Tests C-F): opt-in, they live outside every
    repository and use the private reference cell."""
    path = ACCEPTANCE / name
    if not path.is_file():
        pytest.skip(f'M3 acceptance project not present: {path}')
    original = resolved(path)
    again, _ = resolve_project_paths(reauthor(load(path)), path)
    assert _ordered(project_to_dict(again)) == _ordered(project_to_dict(original))
    Orchestrator().generate(original, tmp_path / 'yaml')
    Orchestrator().generate(again, tmp_path / 'composer')
    assert_same_tree(tmp_path / 'yaml', tmp_path / 'composer')


# ---- LiveCapture in the robot's namespace (ADR-0023) ---------------------------------------------------

def test_the_joint_state_topic_follows_the_robot_namespace():
    from trainit_setup_assistant.livesession import joint_states_topic
    assert joint_states_topic('') == '/joint_states'
    assert joint_states_topic('/r3') == '/r3/joint_states'
    assert joint_states_topic('/cell/r3') == '/cell/r3/joint_states'


def test_live_capture_reads_the_namespaced_robot_and_the_tcp_pose():
    rclpy = pytest.importorskip('rclpy')
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import JointState
    import tf2_ros
    from trainit_setup_assistant.livesession import LiveCapture

    domain = 200 + os.getpid() % 30                  # an isolated DDS domain for this test
    os.environ['ROS_DOMAIN_ID'] = str(domain)
    rclpy.init()
    try:
        pub_node = rclpy.create_node('i0_fake_robot')
        js_pub = pub_node.create_publisher(JointState, '/r3/joint_states', 10)
        root_pub = pub_node.create_publisher(JointState, '/joint_states', 10)
        static = tf2_ros.StaticTransformBroadcaster(pub_node)
        tf = TransformStamped()
        tf.header.frame_id, tf.child_frame_id = 'base_link', 'tcp'
        tf.transform.translation.x, tf.transform.translation.z = 0.4, 0.2
        tf.transform.rotation.x, tf.transform.rotation.w = 1.0, 0.0
        static.sendTransform(tf)
        ns_cap, root_cap = LiveCapture('i0_cap_ns', namespace='/r3'), LiveCapture('i0_cap_root')
        try:
            assert ns_cap.joint_states_age() is None
            for _ in range(40):
                msg = JointState(name=['j1', 'j2'], position=[0.1, -0.2])
                js_pub.publish(msg)
                root_pub.publish(JointState(name=['j1', 'j2'], position=[9.0, 9.0]))
                rclpy.spin_once(pub_node, timeout_sec=0.05)
                if ns_cap.joint_states_age() is not None:
                    break
            assert ns_cap.current_joint_values(['j1', 'j2'], timeout_s=2.0) == {'j1': 0.1, 'j2': -0.2}
            assert root_cap.current_joint_values(['j1'], timeout_s=2.0) == {'j1': 9.0}
            assert ns_cap.joint_states_age() < 5.0
            position, orientation = ns_cap.current_pose('base_link', 'tcp', timeout_s=3.0)
            assert position == pytest.approx([0.4, 0.0, 0.2]) and orientation == pytest.approx([1, 0, 0, 0])
        finally:
            ns_cap.close()
            root_cap.close()
            pub_node.destroy_node()
    finally:
        if rclpy.ok():
            rclpy.shutdown()


# ---- review fixes (I0.0 python review) -----------------------------------------------------------------

def _model_fields(cls) -> set:
    from trainit_setup_assistant.model.motion import MotionConstraints
    own = lambda c: set(getattr(c, 'model_fields', None) or c.__fields__)   # noqa: E731
    return own(cls) - own(MotionConstraints)


# Every field of the models a program document holds, classified: a NEW field fails here until it
# is classified -- a waypoint reference (ops.WAYPOINT_FIELDS) or not.
PINNED_FIELDS = {
    'MoveStep': {'center', 'profile', 'provider', 'to', 'type', 'via'},
    'GraspStep': {'approach_profile', 'at', 'candidates', 'eef', 'index', 'orientation', 'orientation_from',
                  'profile', 'retract', 'site', 'standoff_m'},
    'ReleaseStep': {'approach_profile', 'at', 'eef', 'profile', 'retract', 'site', 'standoff_m'},
    'RelativeTarget': {'dx', 'dy', 'dz', 'frame', 'to'},
    'PathSegment': {'center', 'profile', 'type', 'via'},
    'EventAnchor': {'segment', 'state', 'when'},
    'DetectStep': {'approach', 'class_id', 'detector', 'keep', 'out', 'source', 'timeout_ms'},
    'ToolActuate': {'actuator', 'eef', 'target'},
    'ToolVerify': {'eef', 'expect', 'site'},
    'ProcessStep': {'channel', 'class_id', 'detector', 'kind', 'ms', 'out', 'sync', 'timeout_ms', 'value'},
    'Dwell': {'ms'},
    'PathDef': {'continuity', 'events', 'frame', 'name', 'profile', 'provider', 'segments', 'source', 'states',
                'tcp'},
}


def test_the_waypoint_reference_fields_are_pinned_against_the_model():
    from trainit_setup_assistant.composer.ops import WAYPOINT_FIELDS
    from trainit_setup_assistant.model import program as P
    for name, fields in PINNED_FIELDS.items():
        assert _model_fields(getattr(P, name)) == fields, name
    assert WAYPOINT_FIELDS == {'relative': ('to',), 'segment': ('via', 'center'), 'event_anchor': ('state',),
                               'move': ('to', 'via', 'center'), 'grasp': ('at', 'orientation_from'),
                               'release': ('at',)}


def test_rename_a_profile_onto_another_is_refused():
    c = Composer(load(COMPOSITION))
    names = list(c.program.motion_profiles)
    assert len(names) >= 2
    with pytest.raises(ComposerError, match='already exists'):
        c.rename_profile(names[0], names[1])
    assert list(c.program.motion_profiles) == names


def test_a_load_refusal_keeps_its_code():
    c = Composer(load(COMPOSITION))
    with pytest.raises(ComposerError) as exc:
        c.replace_step(c.task_names()[0], (1,), {'move': {'to': 'home', 'type': 'ptp', 'speedx': 1}})
    assert any(d.code.startswith('R-M3-') for d in exc.value.diagnostics), [str(d) for d in exc.value.diagnostics]


def test_a_no_op_edit_is_not_an_undo_step_and_the_owner_is_told_of_each_change():
    seen = []
    c = Composer(load(COMPOSITION), on_change=seen.append)
    c.edit(lambda doc: None)
    assert not c.can_undo and seen == []
    wp = c.document()['waypoints'][0]
    c.set_waypoint(wp)                                 # the same waypoint: no change
    assert not c.can_undo
    c.rename_waypoint(wp['name'], 'renamed')
    assert c.can_undo and seen == [c.project]
    c.undo()
    assert seen[-1] is c.project and 'renamed' not in c.waypoint_names()


@pytest.mark.parametrize('label, match', [((1, 'a'), 'not a step label'), ((), 'not a step label'),
                                          ((True,), 'not a step label')])
def test_a_bad_step_label_is_a_composer_error(label, match):
    c = Composer(load(COMPOSITION))
    with pytest.raises(ComposerError, match=match):
        c.step(c.task_names()[0], label)


def test_a_step_inside_a_fallback_needs_its_alternative():
    c = Composer(load(COMPOSITION))
    task = c.program.main_task
    c.wrap_steps(task, (1,), 1, 'fallback', alternatives=[[{'dwell': {'ms': 10}}]])
    with pytest.raises(ComposerError, match='alternative'):
        c.step(task, (1, 3, 1))
    with pytest.raises(ComposerError, match='<fallback>.<alternative>.<step>'):
        c.step(task, (1, 2))
