"""TrainIt D0.8 (D0_8_DESIGN.md §7, ADR-0029 Decision 7): the project's DATA COLLECTION -- model, schema 5,
the ONE validator against the PINNED recording vocabulary and the project's own tasks / steps / channels, and
the generation of the recorder's ``trainit-recording/1.0`` files + its bring-up include.

The project is the M3.7 composition cell (``test/data/m3_7/composition.yaml``) with a camera and step ids added.
Every generation goes to a pytest temp dir. The cross-check against the recorder's own parser runs in a CHILD
interpreter (TSA never imports the data stack), and is skipped when trainit_data_ros is not installed.
"""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

import test_m3_7_composition as C
from test_d0_5_step_ids import with_ids
from trainit_setup_assistant.catalog import (DATA_RECORDING_FILE, camera_profiles, parse_recording_vocabulary,
                                             recording_vocabulary, snapshot)
from trainit_setup_assistant.composer.diagnostics import parse
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.data_collection import (data_collection_problems,
                                                               data_collection_warnings, recording_documents)
from trainit_setup_assistant.model import CanonicalProject, load_project, save_project
from trainit_setup_assistant.model.io import project_to_dict

CATALOG = Path(__file__).resolve().parents[1] / 'trainit_setup_assistant' / 'catalog'
CAMERA = {'id': 'bin_camera', 'profile': 'realsense_d455'}
DATA = {'dataset': {'id': 'bin_parts', 'description': 'parts in the bin'},
        'sensors': [{'camera': 'bin_camera'}],
        'triggers': [{'id': 'manual', 'kind': 'manual'},
                     {'id': 'every_2s', 'kind': 'interval', 'period_ms': 2000, 'max_captures': 50},
                     {'id': 'on_conveyor', 'kind': 'event', 'process_channel': 'conveyor', 'edge': 'rising'},
                     {'id': 'before_grasp', 'kind': 'application_state', 'task': 'transfer', 'step': 'pick',
                      'transition': 'enter', 'pick_attempt': 'pre_grasp'},
                     {'id': 'cycle_done', 'kind': 'application_state', 'task': 'cell_cycle',
                      'transition': 'succeeded'}],
        'sync': {'timeout_ms': 800}}


@pytest.fixture(autouse=True)
def _cell(monkeypatch):
    monkeypatch.chdir(C.DATA)
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def with_data(data=DATA, camera=CAMERA, modes=None, ids=True):
    def mutate(doc):
        if ids:
            with_ids()(doc)
        doc.setdefault('perception', {})['camera'] = dict(camera) if camera is not None else None
        if camera is None:
            doc['perception'].pop('camera')
        if data is not None:
            doc['data_collection'] = copy.deepcopy(data)
        if modes is not None:
            doc['deployment']['modes'] = list(modes)
            # the end-effector contract binds every deployed backend (ADR-0022 R15): a simple device elsewhere
            contract = doc['robot']['gripper']['contract']
            contract['adapters'] = {m: 'trainit_mock' if m == 'mock' else 'none' for m in modes}
            contract['verification'] = {'require_grasp_confirmation': False,
                                        'require_release_confirmation': False}
    return mutate


def build(**kw) -> CanonicalProject:
    return C.build(with_data(**kw))


def problems_of(**kw):
    return data_collection_problems(build(**kw))


def edited(**changes):
    data = copy.deepcopy(DATA)
    for path, value in changes.items():
        node = data
        *parents, leaf = path.split('__')
        for part in parents:
            node = node[int(part)] if part.isdigit() else node[part]
        if value is None:
            node.pop(leaf)
        else:
            node[int(leaf) if leaf.isdigit() else leaf] = value
    return data


# ---- the pinned vocabulary (ADR-0021) --------------------------------------------------------------------

def test_the_recording_vocabulary_is_pinned_by_the_catalog():
    vocab = recording_vocabulary()
    raw = (CATALOG / DATA_RECORDING_FILE).read_bytes()
    assert vocab == parse_recording_vocabulary(raw, 'committed')
    recorded = snapshot()['registries']['trainit_data_ros']['recording_vocabulary']
    import hashlib
    assert recorded == {'file': DATA_RECORDING_FILE, 'sha256': hashlib.sha256(raw).hexdigest(),
                        'schema': vocab['schema']}
    assert set(vocab['triggers']['kinds']) == {'MANUAL', 'INTERVAL', 'EVENT', 'APPLICATION_STATE'}
    assert 'realsense_d455' in camera_profiles()


def test_a_vocabulary_without_what_tsa_names_is_refused():
    doc = yaml.safe_load((CATALOG / DATA_RECORDING_FILE).read_text())
    doc['triggers']['kinds'].remove('EVENT')
    with pytest.raises(RuntimeError, match='EVENT'):
        parse_recording_vocabulary(yaml.safe_dump(doc).encode(), 'x')
    doc = yaml.safe_load((CATALOG / DATA_RECORDING_FILE).read_text())
    doc['schema'] = 'trainit-recording-vocabulary/2.0'
    with pytest.raises(RuntimeError, match='this TSA reads'):
        parse_recording_vocabulary(yaml.safe_dump(doc).encode(), 'x')


# ---- model + schema 5 -------------------------------------------------------------------------------------

def test_a_project_without_data_collection_is_unchanged():
    project = C.build()
    data = project_to_dict(project)
    assert 'data_collection' not in data and data['schema_version'] == 4
    assert data_collection_problems(project) == [] and recording_documents(project) == {}


def test_data_collection_content_is_schema_5_and_round_trips(tmp_path):
    project = build()
    data = project_to_dict(project)
    assert data['schema_version'] == 5 and data['perception']['camera']['id'] == 'bin_camera'
    assert data['data_collection']['triggers'][3] == DATA['triggers'][3]
    assert 'tolerance_ms' not in data['data_collection']['sync']                  # unset: absent, never null
    path = save_project(project, tmp_path / 'p.yaml')
    again = load_project(path)
    assert project_to_dict(again) == data
    # a camera id alone is schema-5 content too
    only_id = C.build(with_data(data=None))
    assert project_to_dict(only_id)['schema_version'] == 5


def test_a_5_without_data_collection_is_refused_as_a_dropped_section(tmp_path):
    path = save_project(build(), tmp_path / 'p.yaml')
    doc = yaml.safe_load(path.read_text())
    del doc['data_collection']
    del doc['perception']['camera']['id'], doc['perception']['camera']['profile']
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match='schema_version 5 without data collection'):
        load_project(path)


def test_the_number_a_hand_edit_left_loads_as_5_but_a_template_4_stays_refused(tmp_path):
    path = save_project(build(), tmp_path / 'p.yaml')
    doc = yaml.safe_load(path.read_text())
    doc['schema_version'] = 4                                  # a task program file the section was added to
    path.write_text(yaml.safe_dump(doc))
    assert load_project(path).schema_version == 5
    doc['schema_version'] = 7
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(ValueError, match='R-D0-01'):
        load_project(path)


@pytest.mark.parametrize('mutate', [
    lambda d: d['data_collection'].update(surprise=1),
    lambda d: d['data_collection']['triggers'][0].update(colour='red'),
    lambda d: d['data_collection']['triggers'][0].update(kind='telepathy'),
    lambda d: d['data_collection']['sensors'][0].update(streams=['infrared']),
    lambda d: d['data_collection']['dataset'].update(classification='secret'),
])
def test_the_section_is_strict(mutate):
    with pytest.raises(ValueError):
        C.build(lambda d: (with_data()(d), mutate(d)))


# ---- the ONE validator ------------------------------------------------------------------------------------

def test_the_full_example_is_valid_and_warns_about_nothing():
    project = build()
    assert data_collection_problems(project) == [] and data_collection_warnings(project) == []


@pytest.mark.parametrize('kw, needle', [
    (dict(camera=None), "R-D0-02: data collection sensor 'bin_camera': the project has no camera"),
    (dict(camera={'profile': 'realsense_d455'}), 'the camera has no id'),
    (dict(camera={'id': 'other_cam', 'profile': 'realsense_d455'}), "is not the project's camera"),
    (dict(camera={'id': 'bin_camera'}), 'R-D0-04: the camera has no hardware profile'),
    (dict(camera={'id': 'bin_camera', 'profile': 'kinect_v1'}), "camera profile 'kinect_v1' is not"),
    (dict(camera={'id': 'BinCam', 'profile': 'realsense_d455'}), 'R-D0-03: camera id'),
    (dict(data=edited(dataset__id='Bin Parts')), "R-D0-03: data collection dataset id 'Bin Parts'"),
    (dict(data=edited(triggers__1__period_ms=None)), "trigger 'every_2s': period_ms is required"),
    (dict(data=edited(triggers__1__period_ms=0)), 'period_ms must be an integer >= 1'),
    (dict(data=edited(triggers__0__period_ms=5)), "trigger 'manual': period_ms does not apply"),
    (dict(data=edited(triggers__2__edge='sideways')), "edge 'sideways' is not one of"),
    (dict(data=edited(triggers__1__id='manual')), "trigger id 'manual' is used twice"),
    (dict(data=edited(triggers__1=dict(DATA['triggers'][0], id='manual_2'))), 'at most 1 manual'),
    (dict(data=edited(triggers__3__task='transfre')), "no task is named 'transfre'"),
    (dict(data=edited(triggers__3__step='gone')), "task 'transfer' has no step with the id 'gone'"),
    (dict(data=edited(triggers__3__step='place')), "and step 'place' is a release step"),
    (dict(data=edited(triggers__3__transition='exit')), 'a pre_grasp pick attempt fires on enter, not exit'),
    (dict(data=edited(triggers__3__step=None)), 'a pick attempt names the grasp STEP'),
    (dict(data=edited(triggers__2__process_channel='glue')), "no process channel 'glue' is driven"),
    (dict(data=edited(sync__timeout_ms=0)), 'R-D0-07: data collection sync timeout_ms must be >= 1'),
    (dict(data=edited(sync__policy='nearest_frame')),
     "R-D0-07: data collection sync policy 'nearest_frame' is not one the recorder offers "
     '(first_aligned_at_or_after_boundary, first_at_or_after_boundary)'),
    (dict(data=edited(sensors=[{'camera': 'bin_camera'}, {'camera': 'bin_camera'}])), 'records ONE camera'),
    (dict(data=edited(sensors__0__streams=['depth'])), 'depth is registered to colour'),
    (dict(data=edited(triggers=[])), 'has no trigger'),
    (dict(modes=['gazebo']), 'declares no recording source for any deployment mode'),
])
def test_each_refusal_names_its_cause(kw, needle):
    problems = problems_of(**kw)
    assert any(needle in p for p in problems), problems


def test_a_task_or_step_deleted_after_the_trigger_was_written_is_refused_before_runtime():
    """The D0.5 debt closed: the trigger names a step; removing the step id from the program is caught here."""
    assert data_collection_problems(build()) == []
    project = C.build(with_data(ids=False))
    assert any("has no step with the id 'pick'" in p for p in data_collection_problems(project))


def test_an_application_state_trigger_needs_a_task_program():
    raw = C.raw()
    assert raw['application']['type'] == 'task_program'
    project = build()
    project.application.type = type(project.application.type)('pick_and_place')
    assert any('need a Canonical Task Program' in p for p in data_collection_problems(project))


def test_data_diagnostics_point_at_the_data_section():
    problem = problems_of(data=edited(triggers__3__task='transfre'))[0]
    diag = parse(problem)
    assert diag.code == 'R-D0-06' and diag.location.kind == 'data'
    assert diag.location.describe() == 'data collection (Data section)'


# ---- generation (temp dir) --------------------------------------------------------------------------------

def _generate(project, tmp_path):
    out = tmp_path / 'bundle'
    Orchestrator().generate(project, out)
    return out / project.bundle.app_package


def test_generation_writes_one_recording_file_per_recordable_mode(tmp_path):
    project = build(modes=['mock', 'isaac', 'gazebo'])
    assert data_collection_warnings(project) == [
        'data collection: no recording in mode(s) gazebo -- the camera profile (or the recorder) has no source '
        'for them; the bring-up starts no recorder there']
    app = _generate(project, tmp_path)
    files = sorted(p.name for p in (app / 'config').glob('data_collection.*.yaml'))
    assert files == ['data_collection.isaac.yaml', 'data_collection.mock.yaml']
    mock = yaml.safe_load((app / 'config' / 'data_collection.mock.yaml').read_text())
    assert mock == {
        'schema': 'trainit-recording/1.0', 'source': {'backend': 'mock'}, 'rights': {'classification': 'INTERNAL'},
        'sensors': [{'sensor_id': 'bin_camera', 'profile': 'realsense_d455', 'namespace': '/camera',
                     'streams': {'color': {}, 'depth': {}}}],
        'triggers': [{'id': 'manual', 'kind': 'MANUAL'},
                     {'id': 'every_2s', 'kind': 'INTERVAL', 'period_ms': 2000, 'max_captures': 50},
                     {'id': 'on_conveyor', 'kind': 'EVENT', 'process_channel': 'conveyor', 'edge': 'rising'},
                     {'id': 'before_grasp', 'kind': 'APPLICATION_STATE', 'task': 'transfer', 'step': 'pick',
                      'transition': 'enter', 'context': {'pick_attempt': {'phase': 'pre_grasp'}}},
                     {'id': 'cycle_done', 'kind': 'APPLICATION_STATE', 'task': 'cell_cycle',
                      'transition': 'succeeded'}],
        'sync': {'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 800}}
    isaac = yaml.safe_load((app / 'config' / 'data_collection.isaac.yaml').read_text())
    assert isaac['source'] == {'backend': 'isaac'}
    assert isaac['sensors'][0]['streams']['depth'] == {'info': '/camera/color/camera_info'}
    bringup = (app / 'launch' / 'bringup.launch.py').read_text()
    compile(bringup, 'bringup.launch.py', 'exec')
    assert "if mode in ('mock', 'isaac'):" in bringup and 'recorder.launch.py' in bringup
    assert 'DeclareLaunchArgument("data", default_value="true"' in bringup
    package = (app / 'package.xml').read_text()
    assert '<exec_depend version_gte="0.8.0">trainit_data_ros</exec_depend>' in package
    manifest = yaml.safe_load((tmp_path / 'bundle' / 'generation_manifest.yaml').read_text())
    assert any('data_collection.mock.yaml' in str(f) for f in json.dumps(manifest).split('"'))


def test_a_project_without_live_recording_generates_no_recorder(tmp_path):
    synthetic_only = {'dataset': {'id': 'bin_parts'},
                      'synthetic': {'provider': 'isaac_replicator_bin', 'seed': 3, 'runs': 2, 'views': 1}}
    app = _generate(build(data=synthetic_only), tmp_path)
    assert not list((app / 'config').glob('data_collection.*'))
    bringup = (app / 'launch' / 'bringup.launch.py').read_text()
    assert 'trainit_data_ros' not in bringup and 'trainit_data_ros' not in (app / 'package.xml').read_text()


def test_generation_refuses_an_invalid_data_collection_before_any_file(tmp_path):
    project = build(data=edited(triggers__3__step='gone'))
    with pytest.raises(ValueError, match="has no step with the id 'gone'"):
        Orchestrator().generate(project, tmp_path / 'bundle')
    assert not (tmp_path / 'bundle').exists()


def _recorder_parses(path: Path) -> subprocess.CompletedProcess:
    code = ('import sys, yaml\n'
            'from trainit_data.vocab import Vocabulary\n'
            'from trainit_data_ros.recording_config import load_yaml, resolve\n'
            'from ament_index_python.packages import get_package_share_directory as share\n'
            'def profile(model):\n'
            '    return yaml.safe_load(open(share("trainit_hardware_layer") + f"/profiles/camera/{model}.yaml"))\n'
            'doc, _ = load_yaml(sys.argv[1])\n'
            'r = resolve(doc, Vocabulary.default(), load_profile=profile, use_sim_time=sys.argv[2] == "1")\n'
            'import json; print(json.dumps([s["source_info"]["required"] for s in r["sensors"]]))\n')
    sim = '1' if '.isaac.' in path.name else '0'
    return subprocess.run([sys.executable, '-c', code, str(path), sim], capture_output=True, text=True,
                          env=dict(os.environ))


def test_the_recorder_itself_accepts_every_generated_file(tmp_path):
    """Cross-check against the AUTHORITY, in a child interpreter: the recorder's own parser and resolver."""
    probe = subprocess.run([sys.executable, '-c', 'import trainit_data_ros.recording_config'],
                           capture_output=True)
    if probe.returncode != 0:
        import d08_env
        d08_env.require('trainit_data_ros is not installed in this environment (source the workspace)')
    app = _generate(build(modes=['mock', 'isaac', 'real']), tmp_path)
    files = sorted((app / 'config').glob('data_collection.*.yaml'))
    assert [p.name for p in files] == ['data_collection.isaac.yaml', 'data_collection.mock.yaml',
                                       'data_collection.real.yaml']
    for path in files:
        result = _recorder_parses(path)
        assert result.returncode == 0, f'{path.name}: {result.stderr[-2000:]}'
        # only the physical camera must name its device (its source document, D0.4): the recorder resolves it so
        assert result.stdout.split() == (['[true]'] if '.real.' in path.name else ['[false]']), path.name


def test_the_composer_edits_data_collection_and_only_the_cameras_id_and_profile():
    from trainit_setup_assistant.composer.ops import Composer, ComposerError
    composer = Composer(C.build(with_data(data=None, camera={'rgb_topic': '/camera/color/image_raw'})))

    def add(doc):
        doc['perception']['camera'].update(id='bin_camera', profile='realsense_d455')
        doc['data_collection'] = copy.deepcopy(DATA)
    composer.edit_project(add)
    assert composer.project.data_collection.dataset.id == 'bin_parts' and composer.project.schema_version == 5
    assert composer.can_undo
    with pytest.raises(ComposerError, match='data collection'):
        composer.edit_project(lambda doc: doc['perception']['camera'].update(rgb_topic='/elsewhere/rgb'))
    with pytest.raises(ComposerError):
        composer.edit_project(lambda doc: doc['data_collection']['triggers'].append({'id': 'x', 'kind': 'bogus'}))
    composer.undo()
    assert composer.project.data_collection is None


# ---- review closure (D0.8): boundaries the first test set did not pin ---------------------------------------

@pytest.mark.parametrize('kw, needle', [
    (dict(data=edited(triggers__3__pick_attempt='mid_grasp')), "pick_attempt 'mid_grasp' is not one of"),
    (dict(data=edited(triggers__1__period_ms=-5)), "trigger 'every_2s': period_ms must be an integer >= 1"),
    (dict(data=edited(triggers__1__max_captures=0)), "trigger 'every_2s': max_captures must be an integer >= 1"),
    (dict(data=edited(triggers__1__max_captures=-1)), 'max_captures must be an integer >= 1'),
    (dict(data=edited(dataset__id='a' * 65)), 'longer than 64 characters'),
])
def test_boundary_refusals(kw, needle):
    problems = problems_of(**kw)
    assert any(needle in p for p in problems), problems


@pytest.mark.parametrize('kw', [
    dict(data=edited(dataset__id='a' * 64)),
    dict(data=edited(triggers__2__edge='falling')),
    dict(data=edited(triggers__2__edge='any')),
    dict(data=edited(triggers__1__max_captures=None)),
    dict(data=edited(triggers__1__period_ms=1, triggers__1__max_captures=1)),
])
def test_boundary_acceptances(kw):
    assert problems_of(**kw) == []


def test_a_real_only_project_records_from_the_real_camera_only(tmp_path):
    project = build(modes=['real'])
    assert data_collection_problems(project) == [] and data_collection_warnings(project) == []
    assert list(recording_documents(project)) == ['real']
    app = _generate(project, tmp_path)
    assert [p.name for p in (app / 'config').glob('data_collection.*.yaml')] == ['data_collection.real.yaml']
    assert "if mode in ('real',):" in (app / 'launch' / 'bringup.launch.py').read_text()
    result = _recorder_parses(app / 'config' / 'data_collection.real.yaml')
    if 'No module named' in result.stderr:
        import d08_env
        d08_env.require('trainit_data_ros is not installed in this environment')
    assert result.returncode == 0, result.stderr[-2000:]


NON_CONTRACT = [
    # no contract namespace at all: every topic explicit, depth's CameraInfo = the colour one (registered depth)
    ({'rgb_topic': '/cell_cam/rgb', 'depth_topic': '/cell_cam/depth_registered',
      'camera_info_topic': '/cell_cam/rgb_info'},
     None, {'color': {'topic': '/cell_cam/rgb', 'info': '/cell_cam/rgb_info'},
            'depth': {'topic': '/cell_cam/depth_registered', 'info': '/cell_cam/rgb_info'}}),
    # the contract's colour topic under a namespace, but its own CameraInfo and depth names
    ({'rgb_topic': '/cam2/color/image_raw', 'depth_topic': '/cam2/aligned/depth',
      'camera_info_topic': '/cam2/color/info_custom'},
     '/cam2', {'color': {'info': '/cam2/color/info_custom'}, 'depth': {'topic': '/cam2/aligned/depth'}}),
]


@pytest.mark.parametrize('topics, namespace, streams', NON_CONTRACT)
def test_non_contract_camera_topics_become_explicit_overrides_the_recorder_accepts(topics, namespace, streams,
                                                                                    tmp_path):
    camera = dict(CAMERA, **topics)
    project = build(camera=camera, modes=['mock', 'real'])
    assert data_collection_problems(project) == []
    app = _generate(project, tmp_path)
    for mode in ('mock', 'real'):
        path = app / 'config' / f'data_collection.{mode}.yaml'
        sensor = yaml.safe_load(path.read_text())['sensors'][0]
        assert sensor.get('namespace') == namespace and sensor['streams'] == streams
        result = _recorder_parses(path)
        if 'No module named' in result.stderr:
            import d08_env
            d08_env.require('trainit_data_ros is not installed in this environment')
        assert result.returncode == 0, f'{path.name}: {result.stderr[-2000:]}'


def test_relative_camera_topics_are_refused():
    problems = problems_of(camera=dict(CAMERA, rgb_topic='camera/color/image_raw'))
    assert any('the recorder needs absolute names (camera/color/image_raw)' in p for p in problems)


def test_a_synthetic_only_section_does_not_judge_the_camera_name():
    synthetic_only = {'dataset': {'id': 'bin_parts'}, 'synthetic': {'provider': 'isaac_replicator_bin'}}
    assert problems_of(data=synthetic_only, camera=dict(CAMERA, id='Bin Cam')) == []


def _render(template, **ctx):
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    return build_jinja_env().get_template(template).render(**ctx)


def test_both_bringup_templates_include_the_recorder_only_with_data(tmp_path):
    """The scene-loader flow's bring-up (bringup.launch.py.j2) and the from-scratch one
    (bringup_bootstrap.launch.py.j2, what the composition cell generates): the recorder include, the data:=
    argument and the recordable modes are there exactly when a mode records; without, nothing changes."""
    loader = dict(robot_name='r', moveit_config_package='r_cfg', app_package='r_app', valid_modes_py="('isaac',)",
                  default_mode='isaac', modes_human='isaac', gazebo_supported=False, default_planner_mode='ompl',
                  has_policies=False)
    with_data_ = _render('app/bringup.launch.py.j2', data_modes_py="('isaac',)", data_namespace='r1', **loader)
    compile(with_data_, 'bringup.launch.py', 'exec')
    assert 'recorder.launch.py' in with_data_ and "if mode in ('isaac',):" in with_data_
    assert '"namespace": "r1"' in with_data_ and 'DeclareLaunchArgument("data", default_value="true"' in with_data_
    assert '_share("r_app"), "config"' in with_data_
    plain = _render('app/bringup.launch.py.j2', **loader)
    assert _render('app/bringup.launch.py.j2', data_modes_py='', data_namespace='', **loader) == plain
    assert 'trainit_data_ros' not in plain and '"data"' not in plain
    # the bootstrap flow, through a real generation (the composition cell has no base MoveIt config)
    app = _generate(build(modes=['mock']), tmp_path)
    manifest = yaml.safe_load((tmp_path / 'bundle' / 'generation_manifest.yaml').read_text())
    entry = next(f for f in manifest['files'] if f['path'].endswith(f'{app.name}/launch/bringup.launch.py'))
    assert 'bootstrap' in entry.get('source', '')
    bootstrap = (app / 'launch' / 'bringup.launch.py').read_text()
    compile(bootstrap, 'bringup.launch.py', 'exec')
    assert "if mode in ('mock',):" in bootstrap and 'recorder.launch.py' in bootstrap
    assert 'DeclareLaunchArgument("data", default_value="true"' in bootstrap


# ---- the selectable sync policy (owner decision 2026-11-05 (b)) ---------------------------------------------

def test_the_default_policy_is_the_one_the_vocabulary_declares_not_a_list_position():
    from trainit_setup_assistant.generator.data_collection import sync_policies
    default, offered = sync_policies()
    assert default == {'name': 'FIRST_AT_OR_AFTER_BOUNDARY', 'version': 1}
    assert offered[0]['name'] == 'FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY'   # sorted first: never taken by position
    doc = yaml.safe_load((CATALOG / DATA_RECORDING_FILE).read_text())
    doc['sync']['defaults']['policy'] = {'name': 'NEAREST_WITHIN_WINDOW', 'version': 1}
    with pytest.raises(RuntimeError, match='its default sync policy among the listed ones'):
        parse_recording_vocabulary(yaml.safe_dump(doc).encode(), 'x')


def test_a_chosen_policy_is_written_with_its_version_and_no_choice_writes_the_default():
    unset = recording_documents(build(modes=['mock']))['mock']['sync']
    assert unset == {'policy': 'FIRST_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 800}
    project = build(data=edited(sync__policy='first_aligned_at_or_after_boundary'), modes=['mock'])
    assert data_collection_problems(project) == []
    chosen = recording_documents(project)['mock']['sync']
    assert chosen == {'policy': 'FIRST_ALIGNED_AT_OR_AFTER_BOUNDARY', 'policy_version': 1, 'timeout_ms': 800}


def test_the_recorder_itself_accepts_the_aligned_policy(tmp_path):
    probe = subprocess.run([sys.executable, '-c', 'import trainit_data_ros.recording_config'],
                           capture_output=True)
    if probe.returncode != 0:
        import d08_env
        d08_env.require('trainit_data_ros is not installed in this environment (source the workspace)')
    project = build(data=edited(sync__policy='first_aligned_at_or_after_boundary'), modes=['mock', 'isaac'])
    app = _generate(project, tmp_path)
    for path in sorted((app / 'config').glob('data_collection.*.yaml')):
        result = _recorder_parses(path)
        assert result.returncode == 0, f'{path.name}: {result.stderr[-2000:]}'
