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
            'resolve(doc, Vocabulary.default(), load_profile=profile, use_sim_time=sys.argv[2] == "1")\n')
    sim = '1' if '.isaac.' in path.name else '0'
    return subprocess.run([sys.executable, '-c', code, str(path), sim], capture_output=True, text=True,
                          env=dict(os.environ))


def test_the_recorder_itself_accepts_every_generated_file(tmp_path):
    """Cross-check against the AUTHORITY, in a child interpreter: the recorder's own parser and resolver."""
    probe = subprocess.run([sys.executable, '-c', 'import trainit_data_ros.recording_config'],
                           capture_output=True)
    if probe.returncode != 0:
        pytest.skip('trainit_data_ros is not installed in this environment (source the workspace)')
    app = _generate(build(modes=['mock', 'isaac', 'real']), tmp_path)
    files = sorted((app / 'config').glob('data_collection.*.yaml'))
    assert [p.name for p in files] == ['data_collection.isaac.yaml', 'data_collection.mock.yaml',
                                       'data_collection.real.yaml']
    for path in files:
        result = _recorder_parses(path)
        assert result.returncode == 0, f'{path.name}: {result.stderr[-2000:]}'


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
