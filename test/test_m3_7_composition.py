"""vNext M3.7 (ADR-0024, M3_DESIGN §4, M3_SCHEMA_PROPOSAL §1.1 / §2.3): generic TASK COMPOSITION in a
Canonical Task Program -- task calls, a task's inputs and outputs, the three control steps (repeat,
retry, fallback), the acyclic call graph, the task manifest (config/tasks.yaml) -- and M3 decision 7.

The project under test is ``test/data/m3_7/composition.yaml`` (three tasks, calls, a repeat, a retry
inside a fallback, on the public UR5e sample cell with its end-effector contract), and variants of it
built from its YAML data. Composition is ORCHESTRATION: a step inside a body is the same step, so the
tests compare the nodes of a wrapped step with the nodes of the same step unwrapped. The letters follow
the M3.7 brief's focused-test list (A-W). Every generation goes to a pytest temp dir. ROS-free (the
robot description needs ur_description and xacro, as every UR5e sample test).

The control steps are BehaviorTree.CPP's own nodes; ``run`` below is a REFERENCE model of their
documented semantics (Sequence, Fallback, Repeat, RetryUntilSuccessful, SubTree), ticking a generated
tree with scripted leaf outcomes: it proves which steps run, how often and in which order FOR THE
GENERATED STRUCTURE. BehaviorTree.CPP itself runs the composition in the final end-to-end on mock
(test_m3_7_composition_launch.py).
"""
import copy
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (ProgramRefusals, plan_program,
                                                                program_overview, program_tree_xml)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.program_files import tasks_yaml
from trainit_setup_assistant.model import CanonicalProject, load_project, save_project
from trainit_setup_assistant.model.io import project_to_dict

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / 'test' / 'data' / 'm3_7'
SOURCE = DATA / 'composition.yaml'
VALIDATION_B = PKG / 'test' / 'data' / 'm3_5' / 'validation_b.yaml'
VALIDATION_C = PKG / 'test' / 'data' / 'm3_6' / 'validation_c.yaml'
APP = 'm37_app'

GRASP = {'at': 'pick', 'standoff_m': 0.05, 'profile': 'transit', 'approach_profile': 'approach',
         'retract': {'dz': 0.05}}
CAND_GRASP = {'orientation_from': 'pick', 'standoff_m': 0.05, 'profile': 'transit',
              'approach_profile': 'approach', 'retract': {'along_approach': 0.05}}
PRODUCER = {'detectors': [{'name': 'grasp_provider', 'method': 'custom', 'params': {'class_id': 'part'},
                           'continuous': True,
                           'external_node': {'package': 'm37_producer', 'executable': 'candidates'}}]}


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
def _in_the_data_dir(monkeypatch):
    monkeypatch.chdir(DATA)                              # the urdf path is relative to it


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    """Every test here plans on the public UR5e sample cell: it SKIPS, naming ur_description, on a
    machine that does not have it yet (the fresh-machine check's suite before the sample; vNext M3.7)."""
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def raw(source=SOURCE) -> dict:
    return yaml.safe_load(Path(source).read_text())


def build(mutate=None, source=SOURCE) -> CanonicalProject:
    data = raw(source)
    if mutate is not None:
        mutate(data)
    return CanonicalProject.parse_obj(data)


def with_tasks(tasks_text: str, main: str, producer: bool = False):
    """The composition cell with another program: ``tasks_text`` (YAML) and ``main``."""
    def mutate(data):
        data['application']['tasks'] = yaml.safe_load(tasks_text)
        data['application']['main_task'] = main
        if producer:
            data['perception'] = copy.deepcopy(PRODUCER)
    return mutate


def problems(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


def load_error(mutate) -> str:
    """The refusal of a mutated project as ``load_project`` reports it (located in the file's terms)."""
    import tempfile
    data = raw()
    mutate(data)
    with tempfile.TemporaryDirectory() as tmp:
        f = Path(tmp) / 'project.yaml'
        f.write_text(yaml.safe_dump(data, sort_keys=False))
        with pytest.raises(ValueError) as exc:
            load_project(f)
    return str(exc.value)


def tree(project) -> ET.Element:
    return ET.fromstring(program_tree_xml(project, plan_program(project)).split('\n', 1)[1])


def trees(project) -> dict:
    return {t.get('ID'): t for t in tree(project).iter('BehaviorTree')}


def shape(el: ET.Element) -> list:
    """A tree as nested (tag, children) -- the leaves as tag only."""
    kids = list(el)
    return (el.tag, [shape(k) for k in kids]) if kids else el.tag


# --- the reference model of BehaviorTree.CPP's control nodes ------------------------------------------

def run(project, outcomes=None):
    """Tick the generated MainTree once to completion with scripted leaf outcomes. ``outcomes`` maps a
    leaf key (``tag`` or ``tag:waypoint``) to a list of results consumed in order (default SUCCESS).
    Returns (status, the leaves that ran, in order)."""
    root = tree(project)
    defs = {t.get('ID'): t for t in root.iter('BehaviorTree')}
    script = {k: list(v) for k, v in (outcomes or {}).items()}
    ran = []

    def leaf(el):
        key = el.tag + (f":{el.get('waypoint')}" if el.get('waypoint') else '') + \
            (f":{el.get('expect')}" if el.get('expect') else '')
        ran.append(key)
        for k in (key, el.tag):
            if script.get(k):
                return script[k].pop(0)
        return True

    def tick(el):
        tag = el.tag
        if tag == 'BehaviorTree':
            return tick(el[0])
        if tag == 'Sequence':
            return all(tick(c) for c in el)                      # stops at the first failure
        if tag == 'Fallback':
            return any(tick(c) for c in el)                      # stops at the first success
        if tag == 'Repeat':
            n = int(el.get('num_cycles'))
            assert n >= 1, 'the reference model runs bounded repeats only'
            return all(tick(el[0]) for _ in range(n))
        if tag == 'RetryUntilSuccessful':
            return any(tick(el[0]) for _ in range(int(el.get('num_attempts'))))
        if tag == 'SubTree':
            return tick(defs[el.get('ID')])
        return leaf(el)

    return tick(defs['MainTree']), ran


# --- A. a simple task call -------------------------------------------------------------------------

def test_a_a_call_is_one_subtree_of_the_called_tasks_tree():
    t = trees(build())
    assert list(t) == ['MainTree', 'task_cell_cycle', 'task_transfer', 'task_go_home']
    calls = [(s.get('ID'), s.get('_autoremap')) for s in t['task_cell_cycle'].iter('SubTree')]
    assert calls == [('task_transfer', 'true'), ('task_go_home', 'true')]
    # ONE definition per task, reused by every call: the called task's nodes are never inlined
    assert not list(t['task_cell_cycle'].iter('CloseGripper'))
    assert len(list(t['task_transfer'].iter('CloseGripper'))) == 1


def test_a_the_composition_runs_its_tasks_in_program_order():
    status, ran = run(build(), {'MoveWaypoint:out_of_reach': [False, False]})
    assert status is True
    transfer = ['MoveWaypoint:transfer__s1_pregrasp', 'MoveWaypoint:transfer__s1_grasp', 'CloseGripper',
                'VerifyGrasp:held', 'MoveWaypoint:transfer__s1_retract', 'MoveWaypoint:transfer__s2_prerelease',
                'MoveWaypoint:transfer__s2_release', 'VerifyGrasp:held', 'OpenGripper', 'VerifyGrasp:released',
                'MoveWaypoint:transfer__s2_retract']
    assert ran == (['AddCollisionObject', 'MoveWaypoint:home', 'ProcessEvent'] + transfer + transfer
                   + ['ProcessEvent', 'MoveWaypoint:out_of_reach', 'MoveWaypoint:out_of_reach', 'MoveWaypoint:home',
                      'Sleep'])


# --- B / C / E / F. inputs and outputs, bound by name ---------------------------------------------------

PICKER = """
- name: main
  steps:
    - move: {to: home, type: ptp, profile: transit}
    - detect: {detector: grasp_provider, source: candidates, out: parts, keep: 2}
    - task: pick
- name: pick
  inputs: [{name: parts, type: candidates}]
  steps:
    - grasp: %s
""" % json.dumps(dict(CAND_GRASP, candidates='parts', index=1))


def test_b_a_call_passes_a_candidate_set_by_name():
    p = build(with_tasks(PICKER, 'main', producer=True))
    plan = plan_program(p)
    assert plan.warnings == []
    t = trees(p)
    assert t['task_main'].find('Sequence/DetectCandidates').get('out_key') == 'parts'
    assert t['task_pick'].find('Sequence/SetWaypointFromCandidate').get('from') == 'parts'
    assert [d.name for d in p.application.program.tasks[1].inputs] == ['parts']


def test_b_what_a_called_grasp_needs_is_checked_at_the_call():
    text = problems(build(with_tasks(PICKER.replace('keep: 2', 'keep: 1'), 'main', producer=True)))
    assert ("task 'main' step 3 (task 'pick'), for task 'pick' step 1 (grasp candidate 1 of 'parts'): "
            "candidate 1 of 'parts': its detect step keeps 1 candidate(s) (raise its keep)") in text
    # projected detections suggest no standoff: a called grasp without one is refused at the call
    def projected(data):
        with_tasks(PICKER, 'main', producer=True)(data)
        main, pick = data['application']['tasks']
        main['steps'][1]['detect'].update(source='detections', approach=[0.0, 0.0, -1.0])
        pick['steps'][0]['grasp'].pop('standoff_m')
        data['perception']['detectors'][0].pop('external_node')
        data['perception']['detectors'][0]['method'] = 'color_mask'
    text = problems(build(projected))
    assert "candidates projected from detections suggest no standoff" in text and "task 'main' step 3" in text


def test_b_a_need_passes_through_an_input_to_the_caller_of_the_caller():
    tasks = PICKER.replace('    - task: pick', '    - task: relay') + """
- name: relay
  inputs: [{name: parts, type: candidates}]
  steps:
    - task: pick
"""
    assert plan_program(build(with_tasks(tasks, 'main', producer=True))).warnings == []
    text = problems(build(with_tasks(tasks.replace('keep: 2', 'keep: 1'), 'main', producer=True)))
    assert "task 'main' step 3 (task 'relay'), for task 'relay' step 1 (task 'pick')" in text
    assert 'keeps 1 candidate(s)' in text


OUTPUT = """
- name: main
  steps:
    - move: {to: home, type: ptp, profile: transit}
    - task: look
    - grasp: %s
- name: look
  outputs: [{name: parts, type: candidates}]
  steps:
    - detect: {detector: grasp_provider, source: candidates, out: parts, keep: 1}
""" % json.dumps(dict(CAND_GRASP, candidates='parts'))


def test_c_a_called_tasks_output_is_held_after_the_call():
    p = build(with_tasks(OUTPUT, 'main', producer=True))
    assert plan_program(p).warnings == []
    assert [n.tag for n in trees(p)['task_main'].find('Sequence')][:3] == ['MoveWaypoint', 'SubTree',
                                                                          'SetWaypointFromCandidate']
    # the output keeps what its detect step says: candidate 1 of a 1-candidate set is refused
    text = problems(build(with_tasks(OUTPUT.replace('"retract"', '"index": 1, "retract"'), 'main',
                                     producer=True)))
    assert "candidate 1 of 'parts': output 'parts' of task 'look' keeps 1 candidate(s)" in text


def test_c_a_set_a_called_task_writes_without_declaring_it_is_not_held_after_the_call():
    tasks = OUTPUT.replace("  outputs: [{name: parts, type: candidates}]\n", '')
    text = problems(build(with_tasks(tasks, 'main', producer=True)))
    assert ("no detect step earlier in task 'main' produces candidate set 'parts' (a detect step with out: "
            "parts) -- the call to task 'look' at step 2 writes a set of that name and does not declare it "
            "as an output") in text


def test_e_a_missing_input_is_refused_at_the_call():
    tasks = PICKER.replace('    - detect: {detector: grasp_provider, source: candidates, out: parts, keep: 2}\n',
                           '')
    text = problems(build(with_tasks(tasks, 'main', producer=True)))
    assert ("R-M3-03: task 'main' step 2 (task 'pick'): task 'pick' takes input candidates 'parts', and no "
            "step before the call holds a candidate set of that name") in text
    # a capture input is checked the same way
    tasks = """
- name: main
  steps: [{task: sub}]
- name: sub
  inputs: [{name: captures.0, type: capture}]
  steps: [{dwell: {ms: 1}}]
"""
    text = problems(build(with_tasks(tasks, 'main')))
    assert "task 'sub' takes input capture 'captures.0', and no step before the call produces" in text


def test_e_an_input_no_step_reads_is_refused():
    tasks = PICKER.replace("grasp: ", "dwell: {ms: 1}\n#").replace('  inputs: [{name: parts',
                                                                   '  inputs: [{name: parts')
    text = problems(build(with_tasks(tasks, 'main', producer=True)))
    assert "R-M3-03: task 'pick' input 'parts' (candidates): no step of the task reads it" in text


def test_e_main_task_takes_no_input():
    tasks = """
- name: main
  inputs: [{name: parts, type: candidates}]
  steps:
    - grasp: %s
""" % json.dumps(dict(CAND_GRASP, candidates='parts'))
    text = problems(build(with_tasks(tasks, 'main', producer=True)))
    assert ("R-M3-15: task 'main' input 'parts' (candidates): main_task 'main' is started by the runner, "
            'which provides no data') in text


def test_f_an_output_the_task_does_not_produce_is_refused():
    tasks = """
- name: main
  steps: [{task: look}]
- name: look
  outputs: [{name: parts, type: candidates}, {name: inspection.top, type: capture}]
  steps: [{dwell: {ms: 1}}]
"""
    text = problems(build(with_tasks(tasks, 'main')))
    assert ("R-M3-03: task 'look' output 'parts' (candidates): the task does not produce it on every way it "
            'can succeed') in text
    assert "task 'look' output 'inspection.top' (capture)" in text


def test_f_an_output_produced_by_one_alternative_only_is_not_produced():
    tasks = """
- name: main
  steps: [{task: look}]
- name: look
  outputs: [{name: parts, type: candidates}]
  steps:
    - fallback:
        bodies:
          - [{detect: {detector: grasp_provider, source: candidates, out: parts}}]
          - [{dwell: {ms: 1}}]
"""
    assert "output 'parts' (candidates): the task does not produce it on every way" in problems(
        build(with_tasks(tasks, 'main', producer=True)))
    both = tasks.replace("[{dwell: {ms: 1}}]",
                         "[{detect: {detector: grasp_provider, source: candidates, out: parts}}]")
    assert plan_program(build(with_tasks(both, 'main', producer=True))).warnings == []


@pytest.mark.parametrize('item, needle', [
    ({'name': 'parts', 'type': 'pose'}, "is a candidate set ('candidates'"),
    ({'name': 'parts'}, 'type'),
    ({'name': 'parts', 'type': 'candidates', 'from': 'x'}, "the data of the same name its caller holds"),
    ({'name': 'captures.0', 'type': 'candidates'}, 'must be a name'),
    ({'name': 'bad key!', 'type': 'capture'}, 'must be a blackboard key'),
])
def test_f_a_malformed_input_or_output_is_refused_at_load(item, needle):
    def mutate(data):
        data['application']['tasks'][1]['outputs'] = [item]
    assert needle in load_error(mutate)


def test_f_a_task_declares_a_name_once():
    def mutate(data):
        data['application']['tasks'][1]['outputs'] = [{'name': 'a', 'type': 'candidates'}] * 2
    assert "task 'transfer' declares output 'a' twice" in load_error(mutate)


# --- D / G / H. the call graph --------------------------------------------------------------------------

def test_d_a_call_to_an_unknown_task_is_refused_before_any_file(tmp_path):
    def mutate(data):
        data['application']['tasks'][0]['steps'][2]['repeat']['body'][0]['task'] = 'transfr'
    text = problems(build(mutate))
    assert ("R-M3-15: task 'cell_cycle' step 3.1 (task 'transfr'): no task is named 'transfr' (tasks: "
            'cell_cycle, transfer, go_home)') in text
    out = tmp_path / 'out'
    with pytest.raises(ValueError):
        Orchestrator().generate(build(mutate), str(out))
    assert not out.exists()


def test_g_a_task_calling_itself_is_refused():
    tasks = """
- name: pick_part
  steps: [{dwell: {ms: 1}}, {task: pick_part}]
"""
    text = problems(build(with_tasks(tasks, 'pick_part')))
    assert ("R-M3-15: task call cycle: pick_part -> pick_part (task 'pick_part' step 2 calls 'pick_part'): "
            'the calls of a task program form no cycle') in text


def test_h_a_cycle_through_other_tasks_is_refused_once_with_its_chain(tmp_path):
    tasks = """
- name: cycle
  steps: [{task: pick_part}]
- name: pick_part
  steps:
    - retry: {attempts: 2, body: [{task: recover_pick}]}
- name: recover_pick
  steps: [{task: check_part}]
- name: check_part
  steps: [{task: pick_part}]
"""
    text = problems(build(with_tasks(tasks, 'cycle')))
    assert text.count('task call cycle') == 1
    assert ('task call cycle: pick_part -> recover_pick -> check_part -> pick_part (task \'pick_part\' step '
            "1.1 calls 'recover_pick'; task 'recover_pick' step 1 calls 'check_part'; task 'check_part' "
            "step 1 calls 'pick_part')") in text
    out = tmp_path / 'out'
    with pytest.raises(ValueError):
        Orchestrator().generate(build(with_tasks(tasks, 'cycle')), str(out))
    assert not out.exists()


def test_h_a_task_called_from_two_places_is_no_cycle():
    tasks = """
- name: main
  steps: [{task: a}, {task: b}, {task: a}]
- name: a
  steps: [{task: b}]
- name: b
  steps: [{dwell: {ms: 1}}]
"""
    plan = plan_program(build(with_tasks(tasks, 'main')))
    assert plan.running == ['main', 'a', 'b']


# --- I / J. repeat ------------------------------------------------------------------------------------------

def test_i_repeat_runs_its_body_cycles_times():
    t = trees(build())
    rep = t['task_cell_cycle'].find('Sequence/Repeat')
    assert rep.attrib == {'num_cycles': '2'} and shape(rep) == ('Repeat', [('Sequence', ['SubTree'])])
    status, ran = run(build(), {'MoveWaypoint:out_of_reach': [False, False]})
    assert ran.count('CloseGripper') == 2 and status is True


def test_i_a_failing_cycle_ends_the_repeat_with_its_failure():
    # the second cycle's grasp is not held: the repeat fails, nothing after it runs
    status, ran = run(build(), {'VerifyGrasp:held': [True, True, False]})
    assert status is False
    assert ran.count('CloseGripper') == 2 and 'ProcessEvent' not in ran[3:] and ran[-1] == 'VerifyGrasp:held'


def test_i_repeat_minus_one_is_until_a_step_fails():
    def mutate(data):
        data['application']['tasks'][0]['steps'][2]['repeat']['cycles'] = -1
    assert trees(build(mutate))['task_cell_cycle'].find('Sequence/Repeat').get('num_cycles') == '-1'


@pytest.mark.parametrize('cycles', [0, -2, 1.5, True, 'twice', None])
def test_j_an_invalid_repeat_count_is_refused_at_load(cycles):
    def mutate(data):
        data['application']['tasks'][0]['steps'][2]['repeat']['cycles'] = cycles
    text = load_error(mutate)
    assert "task 'cell_cycle' step 3 (repeat)" in text
    assert 'repeat.cycles' in text or 'cycles' in text


@pytest.mark.parametrize('body', [[], None, 'transfer'])
def test_j_a_repeat_needs_a_body_of_steps(body):
    def mutate(data):
        data['application']['tasks'][0]['steps'][2]['repeat']['body'] = body
    assert 'repeat' in load_error(mutate)


# --- K / L. retry ---------------------------------------------------------------------------------------------

def test_k_retry_runs_its_whole_body_again_until_a_try_succeeds():
    t = trees(build())
    retry = t['task_cell_cycle'].find('Sequence/Fallback/Sequence/RetryUntilSuccessful')
    assert retry.attrib == {'num_attempts': '2'}
    # the first try fails, the second succeeds: the fallback's second alternative never runs
    status, ran = run(build(), {'MoveWaypoint:out_of_reach': [False, True]})
    assert status is True and ran[-2:] == ['MoveWaypoint:out_of_reach', 'MoveWaypoint:out_of_reach']
    assert 'Sleep' not in ran


def test_l_retry_fails_when_every_try_failed():
    tasks = """
- name: main
  steps:
    - retry: {attempts: 3, body: [{move: {to: home, type: ptp, profile: transit}}, {tool: {verify: {expect: held}}}]}
    - dwell: {ms: 1}
"""
    p = build(with_tasks(tasks, 'main'))
    status, ran = run(p, {'VerifyGrasp:held': [False, False, False]})
    assert status is False
    assert ran == ['AddCollisionObject'] + ['MoveWaypoint:home', 'VerifyGrasp:held'] * 3   # the WHOLE body, 3 x


@pytest.mark.parametrize('attempts', [0, -1, 2.0, False])
def test_l_an_invalid_retry_count_is_refused_at_load(attempts):
    def mutate(data):
        data['application']['tasks'][0]['steps'][4]['fallback']['bodies'][0][0]['retry']['attempts'] = attempts
    text = load_error(mutate)
    assert "step 5.1.1 (retry)" in text and 'attempts' in text


# --- M / N. fallback ------------------------------------------------------------------------------------------

def test_m_fallback_stops_at_the_first_alternative_that_succeeds():
    status, ran = run(build(), {'MoveWaypoint:out_of_reach': [True]})
    assert status is True and ran[-1] == 'MoveWaypoint:out_of_reach' and 'Sleep' not in ran


def test_m_fallback_runs_the_next_alternative_after_a_failure():
    status, ran = run(build(), {'MoveWaypoint:out_of_reach': [False, False]})
    assert status is True and ran[-2:] == ['MoveWaypoint:home', 'Sleep']


def test_n_fallback_fails_when_every_alternative_failed():
    status, ran = run(build(), {'MoveWaypoint:out_of_reach': [False, False], 'Sleep': [False]})
    assert status is False and ran[-1] == 'Sleep'


@pytest.mark.parametrize('bodies, needle', [
    ([[{'dwell': {'ms': 1}}]], 'at least 2 alternative bodies'),
    ([[{'dwell': {'ms': 1}}], []], 'fallback.bodies[1] must be a non-empty list of steps'),
    ({'a': 1}, 'at least 2 alternative bodies'),
])
def test_n_a_malformed_fallback_is_refused_at_load(bodies, needle):
    def mutate(data):
        data['application']['tasks'][0]['steps'][4]['fallback']['bodies'] = bodies
    assert needle in load_error(mutate)


# --- O. the control steps around every canonical step -------------------------------------------------------

def _nodes(el, normalise=lambda v: v):
    return [(n.tag, {k: normalise(v) for k, v in n.attrib.items()}) for n in el.iter()
            if n.tag not in ('BehaviorTree', 'Sequence', 'Repeat', 'RetryUntilSuccessful', 'Fallback')]


@pytest.mark.parametrize('source, wrapper', [
    (VALIDATION_B, {'repeat': {'cycles': 3}}),             # move, path with capture events, process
    (VALIDATION_B, {'retry': {'attempts': 2}}),
    (VALIDATION_C, {'retry': {'attempts': 2}}),            # detect, grasp, release
    (VALIDATION_C, {'repeat': {'cycles': 2}}),
])
def test_o_a_wrapped_body_generates_exactly_the_unwrapped_steps(monkeypatch, source, wrapper):
    """Orchestration never changes a step: the same nodes, the same attributes, the same entries
    (the derived entry names carry the step's label: s2 -> s1_2)."""
    monkeypatch.chdir(Path(source).parent)
    flat = build(source=source)
    kind, args = next(iter(wrapper.items()))

    def wrap(data):
        task = data['application']['tasks'][0]
        task['steps'] = [{kind: dict(args, body=task['steps'])}]
    wrapped = build(wrap, source=source)
    task = flat.application.program.tasks[0].name
    a = _nodes(trees(flat)[f'task_{task}'])
    b = _nodes(trees(wrapped)[f'task_{task}'], lambda v: v.replace('__s1_', '__s'))
    assert a == b and a
    pa, pb = plan_program(flat), plan_program(wrapped)
    assert {k.replace('__s1_', '__s'): v for k, v in pb.entries.items()} == pa.entries


def test_o_a_wrapped_fallback_alternative_keeps_each_steps_refusal():
    """A child's validation is never suppressed by a fallback: an alternative that cannot run is
    refused like the same step anywhere else."""
    def mutate(data):
        body = data['application']['tasks'][0]['steps'][4]['fallback']['bodies'][1]
        body.insert(0, {'move': {'to': 'nowhere', 'type': 'ptp', 'profile': 'transit'}})
    assert "task 'cell_cycle' step 5.2.1 (move ptp to 'nowhere'): no waypoint is named 'nowhere'" in problems(
        build(mutate))


def test_o_a_path_at_the_top_of_a_retry_body_cannot_be_proven_and_says_why(monkeypatch):
    monkeypatch.chdir(VALIDATION_B.parent)

    def wrap(data):
        steps = data['application']['tasks'][0]['steps']
        i = next(k for k, s in enumerate(steps) if 'path' in s)
        steps[i] = {'retry': {'attempts': 2, 'body': [steps[i]]}}
    warnings = plan_program(build(wrap, source=VALIDATION_B)).warnings
    assert any('a retry runs its body again from where a failed try left the robot' in w for w in warnings)


# --- P. the generated structure -------------------------------------------------------------------------------

def test_p_one_tree_per_task_each_body_one_sequence():
    t = trees(build())
    fb = t['task_cell_cycle'].find('Sequence/Fallback')
    assert shape(fb) == ('Fallback', [
        ('Sequence', [('RetryUntilSuccessful', [('Sequence', ['MoveWaypoint'])])]),
        ('Sequence', ['SubTree'])])
    main = t['MainTree'].find('Sequence')
    assert [(n.tag, n.get('ID')) for n in main] == [('AddCollisionObject', None),
                                                     ('SubTree', 'task_cell_cycle')]


def test_p_the_derived_entries_of_a_step_in_a_body_carry_its_label():
    def mutate(data):
        data['application']['tasks'][1]['steps'] = [{'retry': {'attempts': 2, 'body': [{'grasp': GRASP}]}}]
    plan = plan_program(build(mutate))
    assert plan.bound_entries == ['transfer__s1_1_pregrasp', 'transfer__s1_1_grasp', 'transfer__s1_1_retract']


# --- Q / R. the manifest, main_task -----------------------------------------------------------------------------

def test_q_tasks_yaml_is_the_manifest_of_the_approved_schema():
    p = build()
    text = tasks_yaml(p, plan_program(p))
    data = yaml.safe_load(text)
    assert data == {'schema_version': 1, 'main_task': 'cell_cycle', 'tasks': {
        'cell_cycle': {'tree': 'task_cell_cycle',
                       'description': 'two parts from the pick to the place, then a recovery to home',
                       'inputs': [], 'outputs': []},
        'transfer': {'tree': 'task_transfer', 'description': 'one part from the pick to the place',
                     'inputs': [], 'outputs': []},
        'go_home': {'tree': 'task_go_home', 'inputs': [], 'outputs': []}}}
    assert text.startswith('# The application\'s TASKS (tasks.yaml schema 1)')


def test_q_the_manifest_states_a_tasks_data():
    p = build(with_tasks(PICKER, 'main', producer=True))
    data = yaml.safe_load(tasks_yaml(p, plan_program(p)))
    assert data['tasks']['pick'] == {'tree': 'task_pick', 'inputs': [{'name': 'parts', 'type': 'candidates'}],
                                     'outputs': []}


def test_r_main_tree_runs_main_task_and_a_task_nobody_calls_is_reported():
    t = trees(build(lambda d: d['application'].update(main_task='go_home')))
    assert t['MainTree'].find('Sequence/SubTree').get('ID') == 'task_go_home'
    plan = plan_program(build(lambda d: d['application'].update(main_task='go_home')))
    assert plan.running == ['go_home']
    assert plan.warnings == ["task 'cell_cycle', 'transfer' never runs: neither main_task 'go_home' nor a task "
                             'it runs calls it']


def test_r_the_tasks_that_run_decide_the_process_controller():
    assert plan_program(build()).process_controller == 'ros'          # conveyor, in the main task
    def moved(data):                                                  # the events only in a called task
        cyc, transfer = data['application']['tasks'][0], data['application']['tasks'][1]
        transfer['steps'] = [cyc['steps'].pop(1)] + transfer['steps'] + [cyc['steps'].pop(2)]
    assert plan_program(build(moved)).process_controller == 'ros'
    assert plan_program(build(lambda d: (moved(d), d['application'].update(main_task='go_home')))
                        ).process_controller == 'mock'


def test_r_two_running_tasks_writing_one_capture_key_are_refused(monkeypatch):
    monkeypatch.chdir(VALIDATION_B.parent)

    def mutate(data):
        app = data['application']
        first = app['tasks'][0]
        app['tasks'] = [{'name': 'main', 'steps': [{'task': first['name']}, {'task': 'again'}]},
                        first, {'name': 'again', 'steps': copy.deepcopy(first['steps'])}]
        app['main_task'] = 'main'
    text = problems(build(mutate, source=VALIDATION_B))
    assert "R-M3-02: capture key 'captures.0' is written by task" in text and "share the program's data" in text


# --- S. floors ------------------------------------------------------------------------------------------------------

def test_s_composition_adds_no_runtime_floor(tmp_path):
    """Repeat, RetryUntilSuccessful, Fallback and SubTree are BehaviorTree.CPP's own nodes and
    tasks.yaml is read by no runtime: the floor is the steps' (here ProcessEvent's, 2.4.0 -- a grasp at
    a taught target needs no candidate node), exactly as for the same steps without composition."""
    needs_ur_description()
    Orchestrator().generate(build(), str(tmp_path / 'a'))
    pkg = (tmp_path / 'a' / APP / 'package.xml').read_text()
    assert 'version_gte="2.4.0">trainit_motion_runtime<' in pkg
    flat = tmp_path / 'b'

    def flatten(data):
        cyc, transfer, home = data['application']['tasks']
        steps = cyc['steps']
        data['application']['tasks'] = [{'name': 'cell_cycle', 'steps': (
            steps[:2] + transfer['steps'] + [steps[3]] + [{'tool': {'verify': {'expect': 'held'}}}]
            + home['steps'])}]
    Orchestrator().generate(build(flatten), str(flat))

    def floors(root):
        return [line.strip() for line in (root / APP / 'package.xml').read_text().splitlines()
                if 'trainit_motion_runtime' in line and ('>=' in line or 'version_gte' in line)]
    assert floors(tmp_path / 'a') == floors(flat) and len(floors(flat)) == 2


# --- T. strict keys and hints ------------------------------------------------------------------------------------------

@pytest.mark.parametrize('step, needle', [
    ({'task': {'name': 'transfer'}}, "a task call is 'task: <name>'"),
    ({'retry': {'attempts': 2, 'step': {'dwell': {'ms': 1}}}}, "a retry wraps a body of steps: 'body: [<step>, ...]'"),
    ({'repeat': {'times': 2, 'body': [{'dwell': {'ms': 1}}]}}, "a repeat runs its body 'cycles: <n>' times"),
    ({'repeat': {'cycles': 2, 'steps': [{'dwell': {'ms': 1}}]}}, "the steps a repeat runs are its 'body"),
    ({'repeat': {'cycles': 2, 'until': 'done', 'body': [{'dwell': {'ms': 1}}]}}, 'a repeat has no condition'),
    ({'fallback': {'body': [{'dwell': {'ms': 1}}]}}, "a fallback tries alternative bodies in order"),
    ({'while': 'x'}, 'a task program has no conditions of its own'),
    ({'if': 'x'}, 'a task program has no conditions of its own'),
    ({'call': 'transfer'}, "a task calls another task with 'task: <name>'"),
    ({'loop': 2}, "a loop is 'repeat: {cycles: <n>, body: [<steps>]}'"),
    ({'task': 'transfer', 'dwell': {'ms': 1}}, 'a step is exactly one of'),
    ({'repeat': {'cycles': 2, 'body': [{'move': {'to': 'home', 'type': 'ptp', 'speedd': 3}}]}},
     "unknown key 'speedd'"),
])
def test_t_a_malformed_composition_is_refused_by_name(step, needle):
    def mutate(data):
        data['application']['tasks'][0]['steps'].append(step)
    assert needle in load_error(mutate)


def test_t_a_load_error_inside_a_body_is_located_by_its_label():
    def mutate(data):
        data['application']['tasks'][0]['steps'][4]['fallback']['bodies'][1].append(
            {'move': {'to': 'home', 'type': 'ptp', 'profle': 'transit'}})
    text = load_error(mutate)
    assert "task 'cell_cycle' step 5.2.2 (move)" in text and "unknown key 'profle'" in text


def test_t_a_task_description_is_one_line():
    def mutate(data):
        data['application']['tasks'][0]['description'] = 'two\nlines'
    assert 'a task description is one line of text' in load_error(mutate)


def test_t_a_pattern_target_is_reserved():
    def mutate(data):
        data['application']['waypoints'].append({'name': 'slot', 'pattern': {'origin': 'place'}})
    text = load_error(mutate)
    assert 'R-M3-14: a pattern target' in text and 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in text


# --- U. the YAML round trip, the wizard ------------------------------------------------------------------------------

def test_u_a_composition_round_trips_without_loss(tmp_path):
    p = load_project(SOURCE)
    saved = save_project(p, tmp_path / 'project.yaml')
    again = load_project(saved)
    assert again == p
    assert yaml.safe_load(saved.read_text())['application'] == raw()['application']
    assert save_project(again, tmp_path / 'twice.yaml').read_text() == saved.read_text()


def test_u_the_template_editors_refuse_a_composition():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(load_project(SOURCE))
    before = project_to_dict(ctrl.project)
    for call in (lambda: ctrl.set_application('pick_and_place'), lambda: ctrl.set_loop(2),
                 lambda: ctrl.clear_application()):
        with pytest.raises(ValueError, match='Canonical Task Program'):
            call()
    assert project_to_dict(ctrl.project) == before


def test_u_the_overview_reads_the_composition_top_to_bottom():
    text = program_overview(load_project(SOURCE).application.program)
    assert 'task cell_cycle -- two parts from the pick to the place, then a recovery to home' in text
    lines = [line.strip() for line in text.splitlines()]
    for expected in ('3  repeat 2 cycles:', '3.1  task  transfer (call)',
                     '5  fallback, the first alternative that succeeds:', 'alternative 1:',
                     '5.1.1  retry, 2 tries at most:', '5.1.1.1  move  PTP   -> out_of_reach profile transit',
                     'alternative 2:',
                     '5.2.1  task  go_home (call)'):
        assert expected in lines, (expected, text)


@pytest.fixture
def qapp():
    import os
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('python_qt_binding', reason='the wizard tests need python_qt_binding')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_u_the_wizard_shows_a_composition_and_leaves_it_as_written(qapp, tmp_path):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    needs_ur_description()
    ctrl = AssistantController()
    ctrl.open_project(SOURCE)
    before = project_to_dict(ctrl.project)
    w = SetupWizard(ctrl)
    w.application_page.initializePage()
    assert w.application_page.validatePage()
    blocks = w.blocks_page
    blocks.initializePage()
    assert '3.1  task  transfer (call)' in blocks.program_view.toPlainText()
    blocks.validate_program()
    assert blocks.program_result.text().startswith('Valid:')
    assert project_to_dict(ctrl.project) == before
    assert load_project(ctrl.save(tmp_path / 'saved.yaml')) == load_project(SOURCE)


def test_u_the_wizard_offers_no_removed_application_type(qapp):
    from trainit_setup_assistant.gui.wizard import ApplicationPage
    assert [t for t, _, _ in ApplicationPage.APPS] == ['pick_and_place', 'vision_guided_motion', 'bin_picking']


# --- V / W. no future-track semantics, determinism -------------------------------------------------------------------

@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    needs_ur_description()
    tmp = tmp_path_factory.mktemp('m3_7_composition')
    with pytest.MonkeyPatch.context() as mp:
        mp.chdir(DATA)
        project = load_project('composition.yaml')
        manifest = Orchestrator().generate(project, tmp / 'a')
        Orchestrator().generate(project, tmp / 'b')
    return dict(root=tmp / 'a', again=tmp / 'b', manifest=manifest)


def test_v_the_bundle_carries_no_future_track_semantics(bundle):
    app = bundle['root'] / APP
    text = '\n'.join(p.read_text() for p in app.rglob('*') if p.is_file()).lower()
    for word in ('plc', 'twincat', 'ads', 'opc', 'machinedevice', 'machine_device', 'cellspec',
                 'robottaskserver', 'task_server_', 'recording', 'policy', 'surface_grasp', 'surfacegrasp'):
        assert f' {word} ' not in f' {text} '.replace('\n', ' '), word
    params = (app / 'config' / 'bt_params.yaml').read_text()
    assert 'tasks_file' not in params               # no runtime reads the manifest: no parameter for it
    tasks = yaml.safe_load((app / 'config' / 'tasks.yaml').read_text())
    assert set(tasks) == {'schema_version', 'main_task', 'tasks'}
    assert all(set(t) <= {'tree', 'description', 'inputs', 'outputs'} for t in tasks['tasks'].values())
    assert bundle['manifest'].as_dict()['warnings'] == []
    assert 'config/tasks.yaml' in json.dumps(bundle['manifest'].as_dict())


def test_w_generation_is_deterministic(bundle):
    a = {p.relative_to(bundle['root']): p.read_bytes() for p in bundle['root'].rglob('*') if p.is_file()}
    b = {p.relative_to(bundle['again']): p.read_bytes() for p in bundle['again'].rglob('*') if p.is_file()}
    assert a.keys() == b.keys()
    differ = [str(k) for k in a if a[k] != b[k] and k.name != 'generation_manifest.yaml']
    assert differ == []


def test_w_the_readme_says_the_composition(bundle):
    readme = (bundle['root'] / APP / 'README.md').read_text()
    for needle in ('| 3 | repeat | -- | step 3.1, 2 cycles; a step that fails ends the repeat | -- |',
                   '| 3.1 | task | `transfer` | calls task `transfer`: its steps run, then the next step | -- |',
                   '| 5 | fallback | -- | 2 alternatives, in order, until one succeeds: step 5.1.1; step 5.2.1 | -- |',
                   '| 5.1.1 | retry | -- | step 5.1.1.1, again from the first when one fails: 2 tries at most | -- |',
                   '`repeat`, `retry` and `fallback` are BehaviorTree.CPP\'s',
                   '| `config/tasks.yaml` |'):
        assert needle in readme, needle


# --- M3 decision 7 --------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('token', ['gluing', 'follow_path', 'waypoint_replay', 'cnc'])
def test_d7_a_removed_application_type_is_refused_at_load_naming_task_program(tmp_path, token):
    data = yaml.safe_load((PKG / 'examples' / 'fr3wml_project.yaml').read_text())
    data['application']['type'] = token
    f = tmp_path / 'project.yaml'
    f.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError) as exc:
        load_project(f)
    text = str(exc.value)
    assert f"application.type '{token}' is no longer an application type (TSA 8.0)" in text
    assert "'application.type: task_program'" in text and str(f) in text
    with pytest.raises(ValueError, match='no longer an application type'):
        CanonicalProject.parse_obj(data)


def test_d7_every_other_application_type_loads_as_before():
    p = load_project(PKG / 'examples' / 'fr3wml_project.yaml')
    for token in ('pick_and_place', 'vision_guided_motion', 'bin_picking'):
        data = project_to_dict(p)
        data['application']['type'] = token
        assert CanonicalProject.parse_obj(data).application.type.value == token


# --- the flow through bodies and calls (review: coverage) -----------------------------------------------------

CAM = {'name': 'cam', 'method': 'custom', 'params': {'class_id': 'part'}, 'continuous': False,
       'external_node': {'package': 'm37_cam', 'executable': 'cam', 'detect_service': True}}
PART_CAM = {'name': 'part_cam', 'method': 'custom', 'params': {'class_id': 'part'}, 'continuous': True,
            'external_node': {'package': 'm37_part_cam', 'executable': 'detections'}}
PATHS = [{'name': 'line', 'states': ['pick', 'place'], 'segments': [{'type': 'lin'}], 'profile': 'approach'},
         {'name': 'back', 'states': ['place', 'pick'], 'segments': [{'type': 'lin'}], 'profile': 'approach'}]


def program(tasks_text: str, main: str = 'main'):
    """The composition cell with paths line (pick -> place) and back, the detectors grasp_provider
    (candidates), part_cam (detections) and cam (captures), and the given tasks."""
    def mutate(data):
        data['application']['tasks'] = yaml.safe_load(tasks_text)
        data['application']['main_task'] = main
        data['application']['paths'] = copy.deepcopy(PATHS)
        data['perception'] = {'detectors': copy.deepcopy(PRODUCER['detectors']) + [dict(PART_CAM), dict(CAM)]}
    return build(mutate)


def warnings_of(tasks_text: str, main: str = 'main') -> list:
    return plan_program(program(tasks_text, main)).warnings


TO_PICK = '{move: {to: pick, type: ptp, profile: transit}}'
TO_PLACE = '{move: {to: place, type: ptp, profile: transit}}'


def test_flow_a_repeat_whose_body_returns_to_its_start_keeps_the_path_start_provable():
    assert warnings_of(f"""
- name: main
  steps: [{TO_PICK}, {{repeat: {{cycles: 2, body: [{{path: line}}, {TO_PICK}]}}}}]
""") == []


def test_flow_a_repeat_whose_next_cycle_provably_starts_elsewhere_is_refused():
    """Cycle 1 starts at pick, where the path starts; cycle 2 starts where the body ended, place."""
    text = problems(program(f"""
- name: main
  steps: [{TO_PICK}, {{repeat: {{cycles: 2, body: [{{path: line}}]}}}}]
"""))
    assert ("R-M3-04: task 'main' step 2.1 (path 'line'), in a next cycle: the path starts at 'pick', but the "
            "step before leaves the robot at 'place'") in text
    # a first cycle at the wrong place is refused as without the repeat
    text = problems(program(f"""
- name: main
  steps: [{TO_PLACE}, {{retry: {{attempts: 2, body: [{{path: line}}]}}}}]
"""))
    assert "R-M3-04: task 'main' step 2.1 (path 'line'): the path starts at 'pick', but the step before" in text


def test_flow_a_next_try_cannot_be_proven_and_says_why():
    w = warnings_of(f"""
- name: main
  steps: [{TO_PICK}, {{retry: {{attempts: 2, body: [{{path: line}}]}}}}]
""")
    assert len(w) == 1 and "step 2.1 (path 'line'), in a next try" in w[0]
    assert 'a retry runs its body again from where a failed try left the robot' in w[0]
    # a repeat whose body ends where it starts: every cycle starts there
    assert warnings_of(f"""
- name: main
  steps: [{TO_PICK}, {{repeat: {{cycles: 3, body: [{{path: line}}, {{path: back}}]}}}}]
""") == []
    # one cycle, or one try: the body starts where the step before left the robot
    for wrap in ('repeat: {cycles: 1', 'retry: {attempts: 1'):
        assert warnings_of(f"""
- name: main
  steps: [{TO_PICK}, {{{wrap}, body: [{{path: line}}]}}}}]
""") == []


def test_flow_after_a_fallback_the_robot_is_known_only_when_every_alternative_agrees():
    same = f"""
- name: main
  steps: [{TO_PLACE}, {{fallback: {{bodies: [[{TO_PICK}], [{TO_PICK}]]}}}}, {{path: line}}]
"""
    assert warnings_of(same) == []
    other = same.replace(f'[[{TO_PICK}], [{TO_PICK}]]', f'[[{TO_PICK}], [{TO_PLACE}]]')
    w = warnings_of(other)
    assert len(w) == 1 and 'after a fallback the robot is where the alternative that succeeded left it' in w[0]


def test_flow_a_fallbacks_next_alternative_starts_where_the_one_before_failed():
    w = warnings_of(f"""
- name: main
  steps: [{TO_PICK}, {{fallback: {{bodies: [[{{dwell: {{ms: 1}}}}], [{{path: line}}]]}}}}]
""")
    assert len(w) == 1 and "step 2.2.1 (path 'line')" in w[0]
    assert 'an alternative of a fallback starts where the one before it failed' in w[0]


def test_flow_after_a_call_the_robot_is_where_the_called_task_leaves_it():
    ends_at_pick = f"""
- name: main
  steps: [{TO_PLACE}, {{task: sub}}, {{path: line}}]
- name: sub
  steps: [{TO_PICK}]
"""
    assert warnings_of(ends_at_pick) == []
    no_motion = f"""
- name: main
  steps: [{TO_PICK}, {{task: sub}}, {{path: line}}]
- name: sub
  steps: [{{dwell: {{ms: 1}}}}]
"""
    assert warnings_of(no_motion) == []                       # unchanged across the call
    ends_on_a_path = f"""
- name: main
  steps: [{{task: sub}}, {{path: back}}]
- name: sub
  steps: [{TO_PICK}, {{path: line}}]
"""
    assert warnings_of(ends_on_a_path) == []                  # at the path's last state
    ends_with_a_grasp = f"""
- name: main
  steps: [{{task: sub}}, {{path: line}}]
- name: sub
  steps: [{{grasp: {json.dumps(GRASP)}}}]
"""
    w = warnings_of(ends_with_a_grasp)
    assert len(w) == 1 and "task 'sub' (called at step 1) does not end at a waypoint TSA can name" in w[0]


def test_flow_after_a_fallback_a_set_is_what_every_alternative_holds():
    detect = 'detect: {{detector: grasp_provider, source: candidates, out: parts, keep: {k}}}'
    grasp = json.dumps(dict(CAND_GRASP, candidates='parts', index=2))
    tasks = f"""
- name: main
  steps:
    - fallback: {{bodies: [[{{{detect.format(k=2)}}}], [{{{detect.format(k=3)}}}]]}}
    - grasp: {grasp}
"""
    text = problems(program(tasks))
    assert ("candidate 2 of 'parts': the set of whichever alternative of a fallback succeeded keeps 2 "
            'candidate(s)') in text
    only_one = tasks.replace(f'[{{{detect.format(k=3)}}}]', '[{dwell: {ms: 1}}]')
    assert "no detect step earlier in task 'main' produces candidate set 'parts'" in problems(program(only_one))
    projected = tasks.replace(f'{{{detect.format(k=3)}}}',
                              '{detect: {detector: part_cam, source: detections, out: parts, keep: 3, '
                              'approach: [0.0, 0.0, -1.0]}}').replace('"index": 2', '"index": 0')
    projected = projected.replace('"standoff_m": 0.05, ', '')
    assert 'candidates projected from detections suggest no standoff' in problems(program(projected))


def test_flow_a_set_detected_inside_a_retry_is_held_after_it():
    """The step after a retry runs only when a try succeeded, so the try's whole body ran."""
    assert warnings_of(f"""
- name: main
  steps:
    - retry: {{attempts: 2, body: [{{detect: {{detector: grasp_provider, source: candidates, out: parts}}}}]}}
    - grasp: {json.dumps(dict(CAND_GRASP, candidates='parts'))}
""") == []


def test_flow_a_set_a_called_task_replaces_is_named_in_the_refusal():
    tasks = f"""
- name: main
  steps:
    - detect: {{detector: grasp_provider, source: candidates, out: parts}}
    - task: look
    - grasp: {json.dumps(dict(CAND_GRASP, candidates='parts'))}
- name: look
  steps: [{{detect: {{detector: grasp_provider, source: candidates, out: parts}}}}]
"""
    assert ("-- the call to task 'look' at step 2 writes a set of that name and does not declare it as an "
            'output: declare it in the task\'s outputs, or name the sets apart') in problems(program(tasks))
    declared = tasks.replace('- name: look\n', '- name: look\n  outputs: [{name: parts, type: candidates}]\n')
    assert warnings_of(declared) == []


def test_flow_a_called_set_need_inside_a_body_carries_both_labels():
    tasks = f"""
- name: main
  steps:
    - detect: {{detector: grasp_provider, source: candidates, out: parts, keep: 1}}
    - repeat: {{cycles: 2, body: [{{task: pick}}]}}
- name: pick
  inputs: [{{name: parts, type: candidates}}]
  steps:
    - retry: {{attempts: 2, body: [{{grasp: {json.dumps(dict(CAND_GRASP, candidates='parts', index=1))}}}]}}
"""
    assert ("task 'main' step 2.1 (task 'pick'), for task 'pick' step 1.1 (grasp candidate 1 of 'parts'): "
            "candidate 1 of 'parts': its detect step keeps 1 candidate(s)") in problems(program(tasks))


def test_flow_writes_propagate_through_a_relay():
    tasks = f"""
- name: main
  steps:
    - detect: {{detector: grasp_provider, source: candidates, out: parts}}
    - task: relay
    - grasp: {json.dumps(dict(CAND_GRASP, candidates='parts'))}
- name: relay
  steps: [{{task: look}}]
- name: look
  steps: [{{detect: {{detector: grasp_provider, source: candidates, out: parts}}}}]
"""
    assert "the call to task 'relay' at step 2 writes a set of that name" in problems(program(tasks))


# --- capture data ----------------------------------------------------------------------------------------------

def test_capture_an_input_is_refused_because_no_step_reads_a_capture():
    tasks = """
- name: main
  steps: [{process: {kind: capture, detector: cam, out: top}}, {task: sub}]
- name: sub
  inputs: [{name: top, type: capture}]
  steps: [{dwell: {ms: 1}}]
"""
    text = problems(program(tasks))
    assert ("R-M3-03: task 'sub' input 'top' (capture): no step of the task reads it -- no step of this "
            'version reads a capture: a capture is an OUTPUT') in text
    assert 'takes input capture' not in text                 # the caller does hold it


def test_capture_outputs_cross_a_call_and_a_path_event_produces_one():
    tasks = """
- name: main
  outputs: [{name: captures.0, type: capture}, {name: top, type: capture}]
  steps: [{task: scan}, {process: {kind: capture, detector: cam, out: top}}]
- name: scan
  outputs: [{name: captures.0, type: capture}]
  steps: [{move: {to: pick, type: ptp, profile: transit}}, {path: evented}]
"""
    p = program(tasks)
    p.application.program.paths.append(type(p.application.program.paths[0]).parse_obj(
        {'name': 'evented', 'states': ['pick', 'place'], 'segments': [{'type': 'lin'}], 'profile': 'approach',
         'events': [{'at': {'state': 'place'}, 'kind': 'capture', 'detector': 'cam'}]}))
    assert plan_program(p).warnings == []
    missing = tasks.replace("  outputs: [{name: captures.0, type: capture}]\n", '')
    q = program(missing)
    q.application.program.paths.append(p.application.program.paths[-1])
    assert "task 'main' output 'captures.0' (capture): the task does not produce it" in problems(q)


def test_capture_a_key_written_by_a_task_that_never_runs_is_no_collision():
    tasks = """
- name: main
  steps: [{process: {kind: capture, detector: cam}}]
- name: spare
  steps: [{process: {kind: capture, detector: cam}}]
"""
    assert warnings_of(tasks) == ["task 'spare' never runs: neither main_task 'main' nor a task it runs calls it"]
    called = tasks.replace("steps: [{process: {kind: capture, detector: cam}}]\n- name: spare",
                           "steps: [{process: {kind: capture, detector: cam}}, {task: spare}]\n- name: spare", 1)
    assert "capture key 'captures.0' is written by task 'main' step 1 (process capture) and by task 'spare'" in \
        problems(program(called))


# --- the call graph, more ----------------------------------------------------------------------------------------

def test_graph_two_cycles_each_once_and_no_cascade_on_a_call_into_one():
    tasks = """
- name: main
  steps: [{task: a}, {task: c}, {task: a}]
- name: a
  steps: [{task: b}]
- name: b
  steps: [{task: a}]
- name: c
  steps: [{fallback: {bodies: [[{task: c}], [{task: missing}]]}}]
"""
    text = problems(program(tasks))
    assert text.count('task call cycle') == 2
    assert 'task call cycle: a -> b -> a' in text and 'task call cycle: c -> c (task \'c\' step 1.1.1 calls' in text
    assert "R-M3-15: task 'c' step 1.2.1 (task 'missing'): no task is named 'missing'" in text
    assert text.count('\n') + 1 == 3                          # nothing else: no cascade


# --- the manifest and the README, more ---------------------------------------------------------------------------

def test_manifest_a_description_with_yaml_characters_round_trips():
    def mutate(data):
        data['application']['tasks'][1]['description'] = 'a: b # c "quoted" \'single\' \\ end'
    p = build(mutate)
    data = yaml.safe_load(tasks_yaml(p, plan_program(p)))
    assert data['tasks']['transfer']['description'] == 'a: b # c "quoted" \'single\' \\ end'


def test_readme_task_header_states_takes_gives_and_never_runs(tmp_path):
    needs_ur_description()
    tasks = PICKER + """
- name: spare
  outputs: [{name: spare_parts, type: candidates}]
  steps: [{detect: {detector: grasp_provider, source: candidates, out: spare_parts}}]
"""
    Orchestrator().generate(build(with_tasks(tasks, 'main', producer=True)), str(tmp_path / 'out'))
    readme = (tmp_path / 'out' / APP / 'README.md').read_text()
    assert ("Takes candidates `parts`; gives nothing (by name, on the application's one blackboard)."
            in readme)
    assert '(This task never runs: neither the main task nor a task it runs calls it.)' in readme
    # a program without composition keeps its README paragraph-free
    flat = tmp_path / 'flat'
    Orchestrator().generate(build(with_tasks("""
- name: main
  steps: [{move: {to: home, type: ptp, profile: transit}}]
""", 'main')), str(flat))
    assert "BehaviorTree.CPP's `Repeat`" not in (flat / APP / 'README.md').read_text()


# --- review fixes: what a later run sees, a set given back, a called task's first path --------------------------

def test_review_a_set_the_body_writes_is_what_its_next_cycle_or_try_sees():
    """Cycle 2 (or try 2) reads the set the body wrote in the run before: keep 1, not the keep 2 held
    before the body."""
    grasp = json.dumps(dict(CAND_GRASP, candidates='parts', index=1))
    for wrap in ('repeat: {cycles: 2', 'retry: {attempts: 2'):
        text = problems(program(f"""
- name: main
  steps:
    - detect: {{detector: grasp_provider, source: candidates, out: parts, keep: 2}}
    - {wrap}, body: [{{grasp: {grasp}}}, {{detect: {{detector: grasp_provider, source: candidates, out: parts, keep: 1}}}}]}}
"""))
        assert ("step 2.1 (grasp candidate 1 of 'parts'): candidate 1 of 'parts': the set before the body or "
                'the one it wrote keeps 1 candidate(s)') in text, text


def test_review_a_next_alternative_sees_what_a_failed_one_wrote():
    grasp = json.dumps(dict(CAND_GRASP, candidates='parts', index=1))
    text = problems(program(f"""
- name: main
  steps:
    - detect: {{detector: grasp_provider, source: candidates, out: parts, keep: 2}}
    - fallback:
        bodies:
          - [{{detect: {{detector: grasp_provider, source: candidates, out: parts, keep: 1}}}}, {{tool: {{verify: {{expect: held}}}}}}]
          - [{{grasp: {grasp}}}]
"""))
    assert "step 2.2.1 (grasp candidate 1 of 'parts'): candidate 1 of 'parts'" in text and 'keeps 1' in text


def test_review_an_input_given_back_unchanged_keeps_the_callers_set():
    tasks = f"""
- name: main
  steps:
    - detect: {{detector: grasp_provider, source: candidates, out: parts, keep: 1}}
    - task: relay
    - grasp: {json.dumps(dict(CAND_GRASP, candidates='parts', index=1))}
- name: relay
  inputs: [{{name: parts, type: candidates}}]
  outputs: [{{name: parts, type: candidates}}]
  steps:
    - grasp: {json.dumps(dict(CAND_GRASP, candidates='parts'))}
"""
    assert "step 3 (grasp candidate 1 of 'parts'): candidate 1 of 'parts': its detect step keeps 1" in problems(
        program(tasks))


def test_review_a_called_tasks_first_path_is_checked_at_each_call():
    tasks = f"""
- name: main
  steps: [{TO_PICK}, {{task: leg}}, {TO_PLACE}, {{task: leg}}]
- name: leg
  steps: [{{path: line}}]
"""
    text = problems(program(tasks))
    assert ("R-M3-04: task 'main' step 4 (task 'leg'), for task 'leg' step 1 (path 'line'): the path starts at "
            "'pick', but the step before leaves the robot at 'place'") in text
    assert 'step 2 (task' not in text                       # the first call is where the path starts
    ok = tasks.replace(f'{TO_PLACE}, {{task: leg}}]', f'{TO_PICK}, {{task: leg}}]')
    assert warnings_of(ok) == []                           # no warning in the called task itself
    # through a relay: the outermost caller is checked
    relay = f"""
- name: main
  steps: [{TO_PLACE}, {{task: relay}}]
- name: relay
  steps: [{{task: leg}}]
- name: leg
  steps: [{{path: line}}]
"""
    assert ("task 'main' step 2 (task 'relay'), for task 'relay' step 1 (task 'leg'), for task 'leg' step 1 "
            "(path 'line')") in problems(program(relay))


@pytest.mark.parametrize('text', ['tab\there', 'nul\x00', 'line separator'])
def test_review_a_description_is_printable(text):
    def mutate(data):
        data['application']['tasks'][0]['description'] = text
    assert 'a task description is one line of text' in load_error(mutate)


def test_review_a_cycle_causes_no_follow_on_refusal():
    tasks = f"""
- name: main
  steps: [{{task: a}}, {{grasp: {json.dumps(dict(CAND_GRASP, candidates='parts'))}}}]
- name: a
  outputs: [{{name: parts, type: candidates}}]
  steps: [{{task: b}}]
- name: b
  outputs: [{{name: parts, type: candidates}}]
  steps: [{{task: a}}]
"""
    text = problems(program(tasks))
    assert text.count('\n') == 0 and 'task call cycle: a -> b -> a' in text, text
