"""vNext M4 (ADR-0026 amendment M4): a SurfaceGrasp grasp provider in a Canonical Task Program.

The M3.6 Validation C project (the public UR5e sample cell, a suction contract `main` / site `main`) with its
external synthetic producer replaced by trainit_perception's provider: ``method: surface_grasp``.

  C  configuration: the provider's params held to TP's committed schema (required, ranges, the envelope, the
     standoff band), its segmentation, its eef / site against the contract and the device kind
  D  the detect step: candidates only, a capture (trigger), the eef / site the candidates are for
  G  the grasp step: orientation candidate, the max_tilt_deg envelope, align_approach unchanged
  F  the bundle: perception.yaml's entry, the provider's node in the launch, the TP 0.6.0 / TMR 2.8.0 floors
"""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (ProgramRefusals, plan_program,
                                                                 program_tree_xml)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.perception_yaml import grasp_provider_problems
from trainit_setup_assistant.generator.runtime_requirements import requirement, use_of
from trainit_setup_assistant.model import CanonicalProject

DATA = Path(__file__).parent / 'data' / 'm3_6'
SOURCE = DATA / 'validation_c.yaml'
PROVIDER = {
    'name': 'grasp_provider', 'method': 'surface_grasp', 'continuous': False, 'rate_hz': 5.0,
    'segmentation': {'method': 'color_mask', 'params': {'class_id': 'part', 'max_objects': 10}},
    'params': {'eef': 'main', 'site': 'main', 'contact_radius_m': 0.009, 'approach_cone_deg': 45.0,
               'standoff_m': 0.05}}


@pytest.fixture(autouse=True)
def _in_the_data_dir(monkeypatch):
    monkeypatch.chdir(DATA)


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def build(mutate=None) -> CanonicalProject:
    data = yaml.safe_load(SOURCE.read_text())
    data['perception']['detectors'] = [copy.deepcopy(PROVIDER)]
    grasp(data)['max_tilt_deg'] = 60                 # the fail-closed envelope, required with a provider (review M4)
    if mutate is not None:
        mutate(data)
    return CanonicalProject.parse_obj(data)


def steps(d):
    return d['application']['tasks'][0]['steps']


def grasp(d):
    return next(s['grasp'] for s in steps(d) if 'grasp' in s)


def detect(d):
    return next(s['detect'] for s in steps(d) if 'detect' in s)


def provider(d):
    return d['perception']['detectors'][0]


def task_nodes(project) -> list:
    root = ET.fromstring(program_tree_xml(project, plan_program(project)).split('\n', 1)[1])
    task = next(t for t in root.iter('BehaviorTree') if t.get('ID').startswith('task_'))
    return [(el.tag, dict(el.attrib)) for el in task.iter() if el.tag not in ('BehaviorTree', 'Sequence')]


def refusal(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


# --- C: configuration ------------------------------------------------------------------------------------------

def test_a_valid_provider_has_no_problem():
    assert grasp_provider_problems(build()) == []


@pytest.mark.parametrize('change, needle', [
    (lambda p: p['params'].pop('approach_cone_deg'), 'approach_cone_deg is required'),
    (lambda p: p['params'].update(approach_cone_deg=90.0), 'must be in (0, 90)'),
    (lambda p: p['params'].update(approach_cone_deg=0.0), 'must be in (0, 90)'),
    (lambda p: p['params'].pop('eef'), 'eef is required'),
    (lambda p: p['params'].update(contact_radius_m=0.0), 'contact_radius_m must be > 0'),
    (lambda p: p['params'].update(standoff_m=0.6), 'standoff_m 0.6 must lie in'),
    (lambda p: p['params'].update(standoff_m=0.05, standoff_min_m=0.06), 'must lie in'),
    (lambda p: p['params'].update(grasp_mode='parallel'), 'is not one of suction'),
    (lambda p: p['params'].update(approach_cone_axis=[0, 0, 0]), 'is not a valid vec3'),
    (lambda p: p['params'].update(colour='red'), "unknown parameter 'colour'"),
    (lambda p: p.pop('segmentation'), 'needs segmentation'),
    (lambda p: p['segmentation'].update(method='pcl_cluster'), 'exposes no masks'),
    (lambda p: p['segmentation']['params'].update(hue=3), 'are not color_mask parameters'),
    (lambda p: p.update(candidate_rule='score'), 'no candidate_rule'),
    (lambda p: p['params'].update(eef='tool2'), "eef 'tool2' is not the contract's device 'main'"),
    (lambda p: p['params'].update(site='jaw'), "has no grasp site 'jaw'"),
    # review M4: every input the provider's own validation refuses is refused by TSA too (parity)
    (lambda p: p['params'].update(inlier_mm=0.0), 'inlier_mm must be > 0'),
    (lambda p: p['params'].update(max_view_angle_deg=0.0), 'max_view_angle_deg'),
    (lambda p: p['params'].update(z_min_m=5.0, z_max_m=3.0), 'the depth band'),
    (lambda p: p['params'].update(grasp_face='nope'), "grasp_face 'nope'"),
    (lambda p: p['params'].update(tool_radius_m=0.01), 'needs tool_reach_mm > 0'),
    (lambda p: p['params'].update(min_clearance=0.5), 'min_clearance > 0 needs tool_radius_m'),
    (lambda p: p['params'].update(min_visible_fraction=0.5), 'min_visible_fraction > 0 needs the size'),
    (lambda p: p['params'].update(reference_x_axis=[0, 0, 1]), 'parallel to approach_cone_axis'),
    (lambda p: p['params'].update(approach_cone_axis=[1e-12, 0, 0]), 'is not a valid vec3'),
    (lambda p: p['segmentation']['params'].update(h='x'), 'segmentation.params.h'),
    (lambda p: p['segmentation']['params'].update(max_objects=0), 'segmentation.params.max_objects'),
])
def test_a_provider_the_runtime_would_refuse_is_refused_before_any_file(change, needle, tmp_path):
    project = build(lambda d: change(provider(d)))
    text = '\n'.join(grasp_provider_problems(project))
    assert needle in text, text
    with pytest.raises(ValueError, match='surface_grasp'):
        Orchestrator().generate(project, tmp_path / 'ws')


def test_a_provider_on_a_parallel_gripper_is_eef_site_incompatible():
    def parallel(d):
        d['robot']['gripper']['kind'] = 'parallel'
    try:
        project = build(parallel)
    except ValueError:
        pytest.skip('the sample contract cannot be a parallel gripper with a set_bool actuator')
    assert any('EEF / site incompatible' in p for p in grasp_provider_problems(project))


# --- D: the detect step ------------------------------------------------------------------------------------------

def test_the_provider_is_sampled_as_candidates():
    tag, a = task_nodes(build())[0]
    assert tag == 'DetectCandidates' and a['source'] == 'candidates' and a['detector'] == 'grasp_provider'
    assert 'trigger' not in a


def test_a_provider_never_publishes_detections():
    def to_detections(d):
        detect(d).update(source='detections', approach=[0.0, 0.0, -1.0])
    assert 'it publishes grasp CANDIDATES, never detections' in refusal(build(to_detections))


def test_a_capture_samples_the_provider_on_demand():
    project = build(lambda d: detect(d).update(trigger=True))
    _, a = task_nodes(project)[0]
    assert a['trigger'] == 'true'


def test_a_capture_of_an_external_node_needs_its_detect_service():
    def external(d):
        d['perception']['detectors'] = [{'name': 'grasp_provider', 'method': 'custom', 'params': {'class_id': 'part'},
                                         'external_node': {'package': 'p', 'executable': 'e'}}]
        detect(d).update(trigger=True)
    assert 'nothing says' in refusal(build(external))


def test_trigger_is_true_or_false():
    with pytest.raises(ValueError, match='detect.trigger'):
        build(lambda d: detect(d).update(trigger='yes'))


# --- G: the grasp step -----------------------------------------------------------------------------------------

def test_align_approach_is_unchanged_and_names_its_reference():
    _, a = task_nodes(build())[1]
    assert a['orientation'] == 'align_approach' and a['orientation_from'] == 'pick_nominal'
    assert a['max_tilt_deg'] == '60'


def test_candidate_orientation_writes_the_mode_and_keeps_the_envelope_reference():
    _, a = task_nodes(build(lambda d: grasp(d).update(orientation='candidate')))[1]
    assert a['orientation'] == 'candidate' and a['orientation_from'] == 'pick_nominal' and a['max_tilt_deg'] == '60'


def test_candidate_orientation_from_another_producer_needs_no_reference():
    def external(d):
        d['perception']['detectors'] = [{'name': 'grasp_provider', 'method': 'custom', 'params': {'class_id': 'part'},
                                         'external_node': {'package': 'p', 'executable': 'e'}}]
        grasp(d).update(orientation='candidate')
        grasp(d).pop('orientation_from')
        grasp(d).pop('max_tilt_deg')
    _, a = task_nodes(build(external))[1]
    assert a['orientation'] == 'candidate' and 'orientation_from' not in a


def test_the_envelope_is_written_and_needs_a_reference():
    _, a = task_nodes(build(lambda d: grasp(d).update(max_tilt_deg=30)))[1]
    assert a['max_tilt_deg'] == '30'
    def no_ref(d):
        grasp(d).update(orientation='candidate', max_tilt_deg=30)
        grasp(d).pop('orientation_from')
    with pytest.raises(ValueError, match='max_tilt_deg needs'):
        build(no_ref)
    for bad in (0, 180, -1, 'x', True):
        with pytest.raises(ValueError, match='max_tilt_deg'):
            build(lambda d, b=bad: grasp(d).update(max_tilt_deg=b))


def test_a_grasp_from_a_provider_without_an_envelope_is_refused():
    assert "needs 'max_tilt_deg'" in refusal(build(lambda d: grasp(d).pop('max_tilt_deg')))


def test_a_grasp_for_another_site_than_the_providers_is_refused():
    def two_sites(d):
        d['robot']['gripper']['contract']['sites'] = {'main': {'tcp': 'tcp', 'grip': 'suction'},
                                                      'other': {'tcp': 'tcp', 'grip': 'suction'}}
        grasp(d).update(site='other')
    try:
        project = build(two_sites)
    except ValueError as exc:
        pytest.skip(f'the sample contract cannot carry two sites here: {exc}')
    assert 'are for main/main' in refusal(project)


def test_an_orientation_mode_outside_the_two_is_refused():
    with pytest.raises(ValueError, match="'align_approach'.*'candidate'"):
        build(lambda d: grasp(d).update(orientation='free'))


# --- F: the bundle -----------------------------------------------------------------------------------------------

def test_the_bundle_runs_the_provider_and_pins_tp_0_6_0_and_tmr_2_8_0(tmp_path):
    def full(d):
        detect(d).update(trigger=True)
        grasp(d).update(orientation='candidate', max_tilt_deg=40)
    out = tmp_path / 'ws'
    Orchestrator().generate(build(full), out)
    app_dir = out / 'm3d_app'
    per = yaml.safe_load((app_dir / 'config' / 'perception.yaml').read_text())['detectors']['grasp_provider']
    assert per['method'] == 'surface_grasp'
    assert per['segmentation'] == PROVIDER['segmentation']
    assert per['params'] == PROVIDER['params']                     # verbatim: no ranking parameter added
    assert set(per) == {'method', 'input', 'segmentation', 'params', 'continuous', 'rate_hz'}
    launch = (app_dir / 'launch' / 'trainit_bt.launch.py').read_text()
    assert 'executable="surface_grasp_node"' in launch and 'executable="detector_node"' not in launch
    package = (app_dir / 'package.xml').read_text()
    assert 'version_gte="0.6.0">trainit_perception<' in package
    tree = (app_dir / 'bt_trees' / 'task_program.xml').read_text()
    params = (app_dir / 'config' / 'bt_params.yaml').read_text()
    use = use_of(trees=[tree], bt_params=[params])
    assert {'candidate_capture', 'candidate_orientation', 'grasp_envelope'} <= set(use.capabilities)
    assert requirement(use).version == '2.8.0'
    assert 'version_gte="2.8.0">trainit_motion_runtime<' in package


def test_a_provider_grasp_always_needs_the_envelope_floor():
    tree = program_tree_xml(build(), plan_program(build()))
    use = use_of(trees=[tree])
    assert 'grasp_envelope' in use.capabilities and requirement(use).version == '2.8.0'
    assert not {'candidate_capture', 'candidate_orientation'} & set(use.capabilities)
