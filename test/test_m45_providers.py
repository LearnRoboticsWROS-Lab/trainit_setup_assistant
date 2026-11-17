"""vNext M4.5 (M4_5_DESIGN.md §3, §4, §8, §12, §14): the grasp providers, the external segmenter, the mock outcomes.

  V  the vocabulary: TP 0.7.0's record (segmentation sources, eligibility, the scripted provider) read and held to
     what TSA's code names, per provider; the scripted provider is Community (PA45)
  I  surface_grasp on an INSTANCE source (``segmentation: {method: instance_topic, ...}``) + its workspace
     ``eligibility`` -- the node's own refusals, before any file
  E  the EXTERNAL SEGMENTER: ``{method: custom, produces: instances, external_node: {...}}``, launched by name with
     its parameters (``name_parameter`` gets the entry's name); a provider's ``source`` must name one; nothing
     consumes it as detections
  S  the SCRIPTED provider (``method: scripted_grasp``): its entry, its node, its capture-only detect, the envelope
  M  the mock adapter's outcome SEQUENCE (``outcomes: {<site>: [...]}`` on a ``trainit_mock`` binding), HW >= 0.4.0
     only when used
  P  the pins: TP 0.7.0 only for what 0.7.0 reads; no max_objects warning for an instance source
  K  mock -> Isaac: the two variants of capsule_bin_picking.yaml differ ONLY in the runtime mode, the EEF backend
     and the perception entries (the product KPI), and both generate
"""
import copy
import difflib
import re
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant import catalog
from trainit_setup_assistant.applications.task_program import ProgramRefusals, plan_program, program_tree_xml
from trainit_setup_assistant.catalog import (tp_registry, tp_registry_problems, tp_tier_problems,
                                             tp_vocabulary_problems)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.eef_contract import eef_contract_problems
from trainit_setup_assistant.generator.external_node_launch import external_node_problems
from trainit_setup_assistant.generator.perception_yaml import grasp_provider_problems
from trainit_setup_assistant.generator.runtime_requirements import hardware_layer_requirement
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.io import project_yaml
from trainit_setup_assistant.model.perception import DetectorSpec, polygon_px_problems

HERE = Path(__file__).parent
DATA = HERE / 'data' / 'm45'
MOCK = DATA / 'capsule_bin_picking_mock.yaml'
ISAAC = DATA / 'capsule_bin_picking_isaac.yaml'
PKG = HERE.parent / 'trainit_setup_assistant'


@pytest.fixture(autouse=True)
def _in_the_data_dir(monkeypatch):
    monkeypatch.chdir(DATA)


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def raw(path) -> dict:
    return yaml.safe_load(path.read_text())


def build(path, mutate=None) -> CanonicalProject:
    data = raw(path)
    if mutate is not None:
        mutate(data)
    return CanonicalProject.parse_obj(data)


def dets(d) -> list:
    return d['perception']['detectors']


def det(d, name='capsules') -> dict:
    return next(x for x in dets(d) if x['name'] == name)


def task_steps(d) -> list:
    return d['application']['tasks'][0]['steps']


def retry_body(d) -> list:
    """The inner retry's body (move, detect, grasp) of the cycle: repeat { retry 2 { retry 3 {...}, release } }."""
    return task_steps(d)[1]['repeat']['body'][0]['retry']['body'][0]['retry']['body']


def detect(d) -> dict:
    return retry_body(d)[1]['detect']


def grasp(d) -> dict:
    return retry_body(d)[2]['grasp']


def adapters(d) -> dict:
    return d['robot']['gripper']['contract']['adapters']


def problems(project) -> str:
    return '\n'.join(grasp_provider_problems(project))


def refusal(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


def generate(project, tmp_path, name='ws') -> dict:
    """Generate into a temp dir; every text file of the bundle by its relative path."""
    out = tmp_path / name
    Orchestrator().generate(project, out)
    return {p.relative_to(out).as_posix(): p.read_text() for p in out.rglob('*') if p.is_file()}


# --- V: the vocabulary -----------------------------------------------------------------------------------------

def test_the_snapshot_is_tp_0_8_0_with_both_providers():
    """TSA 8.8.0 (M4.5d): TP 0.8.0 (SurfaceGrasp's centre_estimator); the providers' sections are still 0.7.0's."""
    record = tp_registry()
    assert record['trainit_perception'] == '0.8.0'
    assert record['param_since']['centre_estimator'] == '0.8.0'
    assert list(record['grasp_providers']) == ['surface_grasp', 'scripted_grasp']
    sg = record['grasp_providers']['surface_grasp']
    assert tuple(sg) == catalog._TP_PROVIDER_KEYS
    assert sg['segmentation_methods'] == ['color_mask', 'instance_topic']
    assert sg['segmentation_parameters']['instance_topic']['since'] == '0.7.0'
    assert sg['eligibility']['since'] == '0.7.0'
    assert sg['eligibility']['parameters']['roi_px'] == {'type': 'polygon_px', 'required': True}
    sc = record['grasp_providers']['scripted_grasp']
    assert sc['node'] == 'scripted_grasp_node' and sc['grasp_modes'] == ['any']
    assert sc['segmentation_methods'] == [] and sc['segmentation_parameters'] == {} and sc['eligibility'] == {}
    assert tp_registry_problems(record) == [] and tp_vocabulary_problems(record) == []
    assert tp_tier_problems(record) == []


def test_the_scripted_provider_is_community():
    assert catalog.TP_TIERS['grasp_provider'] == {'surface_grasp': 'pro_candidate',
                                                  'scripted_grasp': 'community_candidate'}
    row = next(r for r in catalog.snapshot()['providers'] if r['id'] == 'tp.grasp.scripted_grasp')
    assert row['distribution'] == 'community_candidate' and 'scripted scenario' in row['name']


def _mutated(change) -> dict:
    record = copy.deepcopy(tp_registry())
    change(record)
    return record


def _sg(r):
    return r['grasp_providers']['surface_grasp']


@pytest.mark.parametrize('change, says', [
    (lambda r: _sg(r).pop('eligibility'), "grasp provider 'surface_grasp': keys"),
    (lambda r: _sg(r)['segmentation_methods'].append('yolo'), 'are not declared methods or segmentation sources'),
    (lambda r: _sg(r)['segmentation_parameters'].update(color_mask={'parameters': {}, 'since': '0.7.0'}),
     "has a detector method's name"),
    (lambda r: _sg(r)['segmentation_parameters']['instance_topic'].pop('since'),
     "segmentation source 'instance_topic' is not {parameters, since}"),
    (lambda r: _sg(r)['eligibility'].update(since='soon'), 'eligibility is not {parameters, since}'),
    (lambda r: _sg(r)['eligibility']['parameters']['roi_px'].update(type='polygon'), "type 'polygon' not in"),
    (lambda r: _sg(r)['segmentation_parameters']['instance_topic']['parameters']['max_objects'].pop('default'),
     'an optional parameter without a default'),
    (lambda r: _sg(r)['segmentation_parameters']['instance_topic']['parameters']['classes'].update(
        default=['Capsule!']), "default ['Capsule!'] is not"),
])
def test_a_malformed_provider_record_is_refused(change, says):
    got = tp_registry_problems(_mutated(change))
    assert any(says in p for p in got), got


@pytest.mark.parametrize('change, says', [
    (lambda r: r['grasp_providers']['scripted_grasp']['parameters'].pop('scenario_file'),
     "grasp provider 'scripted_grasp' parameter 'scenario_file' is gone"),
    (lambda r: _sg(r)['segmentation_parameters']['instance_topic']['parameters'].pop('source'),
     "segmentation source 'instance_topic' parameter 'source' is gone"),
    (lambda r: (_sg(r)['segmentation_methods'].remove('instance_topic'), _sg(r)['segmentation_parameters'].clear()),
     "segmentation source 'instance_topic' is gone"),
    (lambda r: _sg(r)['eligibility']['parameters'].pop('min_inside'),
     "eligibility parameter 'min_inside' is gone"),
    (lambda r: r['grasp_providers'].pop('scripted_grasp'), "grasp provider 'scripted_grasp' is absent"),
])
def test_what_tsa_names_of_a_provider_is_required_per_provider(change, says):
    got = tp_vocabulary_problems(_mutated(change))
    assert any(says in p for p in got), got


def test_the_scripted_provider_needs_none_of_the_surface_grasp_parameters():
    """Per provider (owner rule): no eef / site / standoff / contact radius is asked of the scripted provider."""
    assert not set(catalog.TSA_GRASP_PROVIDER_PARAMS['scripted_grasp']) & {
        'eef', 'site', 'standoff_m', 'contact_radius_m', 'grasp_mode', 'approach_cone_deg'}
    assert tp_vocabulary_problems(_mutated(
        lambda r: r['grasp_providers']['scripted_grasp']['parameters'].update(
            eef={'type': 'str', 'required': True}))) == []


# --- I: surface_grasp on an instance source, with a workspace --------------------------------------------------

def test_the_isaac_variant_has_no_problem():
    assert grasp_provider_problems(build(ISAAC)) == []
    assert external_node_problems(build(ISAAC)) == []


@pytest.mark.parametrize('change, says', [
    (lambda d: det(d)['segmentation']['params'].pop('source'), 'segmentation.params.source is required'),
    (lambda d: det(d)['segmentation']['params'].update(source='yo-lo'), 'segmentation.params.source'),
    (lambda d: det(d)['segmentation']['params'].update(classes='capsule'), 'segmentation.params.classes'),
    (lambda d: det(d)['segmentation']['params'].update(classes=['']), 'segmentation.params.classes'),
    (lambda d: det(d)['segmentation']['params'].update(min_score=1.5), 'segmentation.params.min_score'),
    (lambda d: det(d)['segmentation']['params'].update(max_objects=0), 'segmentation.params.max_objects'),
    (lambda d: det(d)['segmentation']['params'].update(h=[0, 10]), 'are not instance_topic parameters'),
    (lambda d: det(d)['eligibility'].pop('roi_px'), 'eligibility.roi_px is required'),
    (lambda d: det(d)['eligibility'].update(roi_px=[[0, 0], [10, 0]]), 'roi_px must be a list of >= 3'),
    (lambda d: det(d)['eligibility'].update(roi_px=[[0, 0], [10, 10], [10, 0], [0, 10]]), 'must be simple'),
    (lambda d: det(d)['eligibility'].update(roi_px=[[0, 0], [5, 0], [10, 0]]), 'a degenerate polygon'),
    (lambda d: det(d)['eligibility'].update(roi_px=[[0, 0], [0, 0], [10, 10]]), 'coincide'),
    (lambda d: det(d)['eligibility'].update(roi_px=[[0, 0], [10, 0], [10, float('inf')]]), 'two finite numbers'),
    (lambda d: det(d)['eligibility'].update(min_inside=0.0), 'min_inside must be in (0, 1]'),
    (lambda d: det(d)['eligibility'].update(min_inside=1.2), 'min_inside must be in (0, 1]'),
    (lambda d: det(d)['eligibility'].update(z_min_m=0.9, z_max_m=0.3), 'the depth band'),
    (lambda d: det(d)['eligibility'].update(z_max_m=21.0), 'the depth band'),
    (lambda d: det(d)['eligibility'].update(box=[0, 1]), 'eligibility has unknown key(s) box'),
])
def test_an_instance_source_or_workspace_the_node_refuses_is_refused_before_any_file(change, says, tmp_path):
    project = build(ISAAC, change)
    assert says in problems(project), problems(project)
    with pytest.raises(ValueError, match='surface_grasp'):
        Orchestrator().generate(project, tmp_path / 'ws')


def test_the_boundary_values_of_the_workspace_are_accepted():
    def edge(d):
        det(d)['eligibility'] = {'roi_px': [[0, 0], [2, 0], [0, 2]], 'min_inside': 1.0, 'z_min_m': 0.0,
                                 'z_max_m': 20.0}
    assert grasp_provider_problems(build(ISAAC, edge)) == []


def test_the_polygon_is_checked_against_the_image_when_the_size_is_known():
    roi = [[120, 80], [520, 80], [520, 400], [120, 400]]
    assert polygon_px_problems(roi) == [] and polygon_px_problems(roi, 640, 480) == []
    assert 'lie outside the 320x240 colour image' in polygon_px_problems(roi, 320, 240)[0]


def test_eligibility_belongs_to_a_grasp_provider_entry_only():
    def on_the_segmenter(d):
        det(d, 'yolo')['eligibility'] = {'roi_px': [[0, 0], [10, 0], [0, 10]]}
    assert "eligibility is a grasp provider's workspace filter" in problems(build(ISAAC, on_the_segmenter))


def test_a_colour_mask_entry_is_unchanged_and_writes_no_eligibility(tmp_path):
    """The M4 shape (color_mask segmentation, no eligibility) still generates the M4 entry, at the M4 floor."""
    def m4(d):
        dets(d)[:] = [{'name': 'capsules', 'method': 'surface_grasp', 'continuous': False, 'rate_hz': 5.0,
                       'segmentation': {'method': 'color_mask', 'params': {'class_id': 'capsule', 'max_objects': 10}},
                       'params': {'eef': 'main', 'site': 'main', 'contact_radius_m': 0.009,
                                  'approach_cone_deg': 60.0, 'standoff_m': 0.05}}]
    files = generate(build(ISAAC, m4), tmp_path)
    entry = yaml.safe_load(files['m45_app/config/perception.yaml'])['detectors']['capsules']
    assert list(entry) == ['method', 'input', 'segmentation', 'params', 'continuous', 'rate_hz']
    # TSA 8.7.0: the fixture's presence capture reads the objects TP publishes since 0.7.0 -- that, not the entry,
    # sets the floor; without presence the entry alone is at the M4 floor
    assert 'version_gte="0.7.0">trainit_perception<' in files['m45_app/package.xml']

    def m4_without_presence(d):
        m4(d)
        d['application']['tasks'][0]['steps'][1]['repeat'].pop('presence')
    files = generate(build(ISAAC, m4_without_presence), tmp_path, 'plain')
    assert 'version_gte="0.6.0">trainit_perception<' in files['m45_app/package.xml']


# --- E: the external segmenter ---------------------------------------------------------------------------------

def test_the_segmenter_is_launched_by_name_with_its_parameters(tmp_path):
    files = generate(build(ISAAC), tmp_path)
    launch = files['m45_app/launch/trainit_bt.launch.py']
    block = launch[launch.index('EXTERNAL SEGMENTER "yolo"'):launch.index('executable="trainit_run_bt"')]
    assert 'package="trainit_yolo_provider"' in block and 'executable="yolo_seg_node"' in block
    assert 'name="yolo_seg"' in block
    for line in ('"model_path": "/home/trainit/TrainItModels/capsule_seg/model.onnx",', '"output_name": "yolo",',
                 '"image_topic": "/camera/color/image_raw",', '"classes": ["capsule"],', '"conf": 0.25,',
                 '"use_sim_time": use_sim,'):
        assert line in block, line
    assert '/perception/yolo/detections' not in block
    # the bundle does not depend on it (ADR-0003): a comment, never an <exec_depend>
    package = files['m45_app/package.xml']
    assert '<exec_depend>trainit_yolo_provider' not in package and 'trainit_yolo_provider' in package
    assert 'The external SEGMENTER entr(ies) yolo publish an' in package
    # perception.yaml records it (who publishes the instance image, from which camera); no TP node reads it
    entry = yaml.safe_load(files['m45_app/config/perception.yaml'])['detectors']['yolo']
    assert entry == {'method': 'custom', 'produces': 'instances',
                     'input': {'rgb': '/camera/color/image_raw', 'depth': '/camera/depth/image_rect_raw',
                               'camera_info': '/camera/color/camera_info'}}


def test_tsa_never_imports_or_names_the_segmenter_package():
    """The external segmenter is launched BY NAME from the project: no TSA source names its package."""
    hits = [p for p in PKG.rglob('*.py') if 'trainit_yolo_provider' in p.read_text()]
    hits += [p for p in PKG.rglob('*.j2') if 'trainit_yolo_provider' in p.read_text()]
    assert hits == []


@pytest.mark.parametrize('change, says', [
    (lambda d: det(d)['segmentation']['params'].update(source='seg'),
     "source 'seg' names no entry: declare the segmenter as an entry {name: seg, method: custom, produces: "
     'instances, external_node: {package, executable, ...}} (or'),
    (lambda d: (det(d, 'yolo').pop('produces'),), "'yolo' is not an external segmenter"),
    (lambda d: det(d, 'yolo').update(method='pcl_cluster'), 'its method is custom'),
    (lambda d: det(d, 'yolo').update(produces='masks'), "produces 'masks' is not an output TSA knows"),
    (lambda d: det(d, 'yolo').update(candidate_rule='score'), 'no detections to rank'),
    (lambda d: det(d, 'yolo')['external_node'].update(detect_service=True), 'serves no /perception/yolo/detect'),
])
def test_what_reads_or_misdeclares_a_segmenter_is_refused(change, says, tmp_path):
    project = build(ISAAC, change)
    assert says in problems(project), problems(project)
    with pytest.raises(ValueError):
        Orchestrator().generate(project, tmp_path / 'ws')


@pytest.mark.parametrize('change, says', [
    (lambda e: e.update(name_parameter=' '), 'name_parameter is blank'),
    (lambda e: e['parameters'].update(output_name='other'), "would publish under another name"),
])
def test_the_name_parameter_carries_the_entry_name_only(change, says):
    project = build(ISAAC, lambda d: change(det(d, 'yolo')['external_node']))
    assert any(says in p for p in external_node_problems(project))


def test_an_explicit_matching_output_name_is_accepted():
    project = build(ISAAC, lambda d: det(d, 'yolo')['external_node']['parameters'].update(output_name='yolo'))
    assert external_node_problems(project) == []


def test_a_segmenter_the_cell_starts_itself_is_declared_and_warned(tmp_path):
    project = build(ISAAC, lambda d: det(d, 'yolo').pop('external_node'))
    assert grasp_provider_problems(project) == []
    out = tmp_path / 'ws'
    manifest = Orchestrator().generate(project, out)
    text = (out / 'generation_manifest.yaml').read_text()
    assert 'EXTERNAL SEGMENTER (produces: instances) with no external_node block' in text
    assert 'yolo_seg_node' not in (out / 'm45_app/launch/trainit_bt.launch.py').read_text()
    del manifest


def test_a_detect_step_never_consumes_the_segmenter():
    def from_the_segmenter(d):
        detect(d).update(detector='yolo')
    assert "'yolo' is an external segmenter (produces: instances)" in refusal(build(ISAAC, from_the_segmenter))


def test_the_new_keys_save_only_when_set():
    saved = yaml.safe_load(project_yaml(build(ISAAC)))
    assert det(saved, 'yolo')['produces'] == 'instances'
    assert det(saved, 'yolo')['external_node']['name_parameter'] == 'output_name'
    assert 'eligibility' in det(saved)
    plain = yaml.safe_load(project_yaml(build(DATA / 'capsule_bin_picking.yaml')))
    assert not {'produces', 'eligibility'} & set(det(plain))


# --- S: the scripted provider ----------------------------------------------------------------------------------

def test_the_mock_variant_has_no_problem_and_its_entry_is_what_the_node_reads(tmp_path):
    project = build(MOCK)
    assert grasp_provider_problems(project) == [] and eef_contract_problems(project) == []
    files = generate(project, tmp_path)
    entry = yaml.safe_load(files['m45_app/config/perception.yaml'])['detectors']['capsules']
    assert entry == {'method': 'scripted_grasp',
                     'params': {'scenario_file': '/home/trainit/TrainItData/scenarios/capsule_bin.yaml',
                                'frame_id': 'camera_color_optical_frame', 'after_end': 'empty'},
                     'continuous': False}
    launch = files['m45_app/launch/trainit_bt.launch.py']
    assert 'executable="scripted_grasp_node"' in launch and 'name="grasp_provider_capsules"' in launch
    assert 'surface_grasp_node' not in launch
    assert '"detector_name": "capsules"' in launch and '"use_sim_time": use_sim' in launch
    # the scenario is the node's (TSA never reads it): an absent file is said, never refused
    assert 'scenario_file /home/trainit/TrainItData/scenarios/capsule_bin.yaml is not on this machine' in files[
        'generation_manifest.yaml']


def test_a_scenario_that_exists_is_not_warned_about(tmp_path):
    scenario = tmp_path / 'scenario.yaml'
    scenario.write_text('steps:\n  - {candidates: []}\n')
    project = build(MOCK, lambda d: det(d)['params'].update(scenario_file=str(scenario)))
    assert 'is not on this machine' not in generate(project, tmp_path)['generation_manifest.yaml']


@pytest.mark.parametrize('change, says', [
    (lambda e: e['params'].pop('scenario_file'), 'scenario_file is required'),
    (lambda e: e['params'].update(scenario_file=''), "scenario_file = '' is not a valid str"),
    (lambda e: e['params'].update(scenario_file='scenarios/capsule_bin.yaml'), 'must be an absolute path'),
    (lambda e: e['params'].pop('frame_id'), 'frame_id is required'),
    (lambda e: e['params'].update(after_end='loop'), "after_end = 'loop' is not one of empty, repeat_last"),
    (lambda e: e['params'].update(eef='main'), "unknown parameter 'eef'"),
    (lambda e: e['params'].update(reference_x_axis=[0, 0, 2]), 'parallel to approach_cone_axis'),
    (lambda e: e['params'].update(approach_cone_axis=[0, 0, 0]), 'is not a valid vec3'),
    (lambda e: e.update(segmentation={'method': 'instance_topic', 'params': {'source': 'yolo'}}),
     'a scripted provider reads no camera'),
    (lambda e: e.update(rgb_topic='/cam/rgb'), 'rgb_topic'),
    (lambda e: e.update(rate_hz=5.0), 'rate_hz'),
    (lambda e: e.update(eligibility={'roi_px': [[0, 0], [10, 0], [0, 10]]}), 'eligibility'),
    (lambda e: e.update(external_node={'package': 'p', 'executable': 'x'}), 'no external_node block'),
])
def test_a_scripted_entry_the_node_refuses_is_refused_before_any_file(change, says, tmp_path):
    project = build(MOCK, lambda d: change(det(d)))
    assert says in problems(project), problems(project)
    with pytest.raises(ValueError, match='scripted_grasp'):
        Orchestrator().generate(project, tmp_path / 'ws')


def test_the_scripted_eef_and_site_are_the_scenarios():
    """TSA cannot read a scenario's candidates statically: no eef / site check against the device is made, and
    the grasp step keeps its own device / site (the runtime refuses a candidate for another one)."""
    assert grasp_provider_problems(build(MOCK)) == []
    assert 'for_site' not in program_tree_xml(build(MOCK), plan_program(build(MOCK)))


def test_a_scripted_detect_must_capture():
    assert "'capsules' is a scripted grasp provider: it publishes one scenario step per capture" in refusal(
        build(MOCK, lambda d: detect(d).pop('trigger')))


def test_a_scripted_grasp_keeps_the_fail_closed_envelope():
    """The envelope rule holds for EVERY provider's candidates -- the scripted ones too, eef / site unknown."""
    def no_envelope(d):
        grasp(d).pop('max_tilt_deg')
    assert "needs 'max_tilt_deg' and 'orientation_from'" in refusal(build(MOCK, no_envelope))


# --- M: the mock outcome sequence ------------------------------------------------------------------------------

def test_the_mock_adapter_gets_the_sequence_and_pins_hw_0_4_0(tmp_path):
    files = generate(build(MOCK), tmp_path)
    bringup = files['m45_app/launch/bringup.launch.py']
    assert "'mock.sites.main.outcomes': 'not_held,held'" in bringup
    package = files['m45_app/package.xml']
    assert '<exec_depend version_gte="0.4.0">trainit_hardware_layer</exec_depend>' in package
    assert 'mock_eef_adapter with its outcome sequence (ADR-0022)' in package


def test_without_a_sequence_the_hw_floor_is_unchanged(tmp_path):
    files = generate(build(MOCK, lambda d: adapters(d).update(mock='trainit_mock')), tmp_path)
    assert 'outcomes' not in files['m45_app/launch/bringup.launch.py']
    assert '<exec_depend version_gte="0.2.0">trainit_hardware_layer</exec_depend>' in files['m45_app/package.xml']


def test_the_hw_floor_reads_the_sequence_from_the_launch():
    row = "{'package': 'trainit_hardware_layer', 'executable': 'mock_eef_adapter', 'parameters': [%s]}"
    plain = 'BRIDGES = [' + row % "{'device_id': 'main'}" + ']\n'
    seq = 'BRIDGES = [' + row % "{'device_id': 'main'}, {'mock.sites.main.outcomes': 'held'}" + ']\n'
    assert hardware_layer_requirement({'a.py': plain}).version == '0.2.0'
    assert hardware_layer_requirement({'a.py': seq}).version == '0.4.0'


@pytest.mark.parametrize('binding, says', [
    ({'adapter': 'trainit_mock', 'outcomes': {'jaw': ['held']}}, 'outcomes names site "jaw"'),
    ({'adapter': 'trainit_mock', 'outcomes': {'main': []}}, 'must be a non-empty list'),
    ({'adapter': 'trainit_mock', 'outcomes': {'main': ['unavailable']}}, 'must be a non-empty list'),
])
def test_a_sequence_the_adapter_refuses_is_refused(binding, says):
    project = build(MOCK, lambda d: adapters(d).update(mock=binding))
    assert any(says in p for p in eef_contract_problems(project)), eef_contract_problems(project)


def test_only_the_mock_row_reads_a_sequence():
    def on_isaac(d):
        adapters(d)['isaac']['outcomes'] = {'main': ['held']}
    assert any("reads no ['outcomes']" in p for p in eef_contract_problems(build(ISAAC, on_isaac)))


def test_the_sequence_saves_as_written_and_a_plain_binding_saves_unchanged():
    saved = yaml.safe_load(project_yaml(build(MOCK)))
    assert adapters(saved)['mock'] == {'adapter': 'trainit_mock', 'outcomes': {'main': ['not_held', 'held']}}
    plain = yaml.safe_load(project_yaml(build(DATA / 'capsule_bin_picking.yaml')))
    assert adapters(plain) == {'mock': 'trainit_mock'}


# --- P: the pins -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize('path', [MOCK, ISAAC], ids=['scripted', 'instance_topic+eligibility'])
def test_tp_is_pinned_at_the_release_that_reads_the_entry(path, tmp_path):
    assert 'version_gte="0.7.0">trainit_perception<' in generate(build(path), tmp_path)['m45_app/package.xml']


def test_each_0_7_0_section_alone_moves_the_floor(tmp_path):
    def only_eligibility(d):
        dets(d)[:] = [det(d)]
        det(d)['segmentation'] = {'method': 'color_mask', 'params': {'class_id': 'capsule', 'max_objects': 10}}
    def only_instances(d):
        det(d).pop('eligibility')
    for i, change in enumerate((only_eligibility, only_instances)):
        package = generate(build(ISAAC, change), tmp_path, f'ws{i}')['m45_app/package.xml']
        assert 'version_gte="0.7.0">trainit_perception<' in package, change.__name__


def test_no_max_objects_warning_for_an_instance_source(tmp_path):
    def no_max(d):
        det(d)['segmentation']['params'].pop('max_objects')
    files = generate(build(ISAAC, no_max), tmp_path)
    assert 'sets no max_objects' not in files['generation_manifest.yaml']


# --- K: mock -> Isaac --------------------------------------------------------------------------------------------

def _leaf_paths(a, b, path=()):
    """The paths at which two parsed YAML documents differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in list(dict.fromkeys(list(a) + list(b))):
            out += _leaf_paths(a.get(k), b.get(k), path + (k,))
        return out
    return [] if a == b else [path]


def test_mock_and_isaac_differ_only_in_mode_eef_backend_and_perception():
    paths = _leaf_paths(raw(MOCK), raw(ISAAC))
    assert sorted(paths) == sorted([
        ('deployment', 'modes'), ('deployment', 'default_mode'),
        ('robot', 'gripper', 'contract', 'adapters', 'mock'), ('robot', 'gripper', 'contract', 'adapters', 'isaac'),
        ('perception', 'detectors')])
    text = list(difflib.unified_diff(MOCK.read_text().splitlines(), ISAAC.read_text().splitlines(), lineterm='',
                                     n=0))
    changed = [line for line in text if line[:1] in '+-' and not line.startswith(('+++', '---'))]
    assert len(changed) == 34, '\n'.join(text)          # the KPI: the lines a cell changes from mock to Isaac


def test_both_variants_generate_the_same_tree(tmp_path):
    a = generate(build(MOCK), tmp_path, 'mock')
    b = generate(build(ISAAC), tmp_path, 'isaac')
    tree = 'm45_app/bt_trees/' + next(p.split('/')[-1] for p in a if p.startswith('m45_app/bt_trees/'))
    assert a[tree] == b[tree]                                # the application does not change between them
    assert re.search(r'<DetectCandidates [^>]*detector="capsules"[^>]*trigger="true"', a[tree])
