"""vNext M4.5 (ADR-0024 amendment M4.5, M4_5_DESIGN.md §6, §7, §9, §11): the task-program COMPOSITION additions.

All four are opt-in; a program that uses none generates exactly the tree it did before.

  F  ``application.faults: stop``: every ``fault`` output to ONE gate key, reset when the run starts; every retry
     body and fallback alternative ``_failureIf`` on it; MainTree logs the reason and ends with FAILURE
  U  ``repeat: {until: nothing_to_pick}``: the repeat's failure is its NORMAL end only on a fresh NO_CANDIDATES
     (the detect reason key, reset before each detect and at every control body's start) with the gate clear
  R  ``grasp: {on_failure: retreat}``: after the PreGrasp was reached, a failed idiom opens the tool and goes back
     to the PreGrasp (LIN, approach_profile), then fails; a failed retreat is a fault
  R2 ``release: {on_failure: retreat}`` (TSA 8.6.1, the M4.5 Isaac benchmark): after the RELEASE POSE was reached
     (TSA 8.6.2, the M4.5 review: no longer the PreRelease), a failed release (a VERDICT: the held guard's
     OBJECT_LOST, a refused retract) opens the tool (no held guard), detaches the payload, goes back to the
     PreRelease (LIN, approach_profile), then fails -- an enclosing retry re-runs the cycle; a failure before the
     release pose (the approach LIN included) is the fault PLACE_UNREACHABLE_WHILE_HOLDING with the tool closed; a
     failed retreat is a fault, a fault (a failed release verification) stays one
  A  ``grasp: {index: all}``: a Fallback over the literal indices 0..keep-1 (the step id on each attempt)
  P  ``grasp: {payload: {dims, offset}}``: AttachObject (touch_links: the TCP and scene.touch_links) after the
     verified grasp, DetachObject after each open; TSA 8.6.2 (the M4.5 review), under faults: stop: MainTree first
     proves the tool empty (VerifyGrasp released; held / engaged: the fault HOLDING_AT_START) and detaches the payload
     (a phantom attached object from an earlier run cleared)

The example program is ``test/data/m45/capsule_bin_picking.yaml`` (the M4.5 decomposition, §2), in the recommended
cycle shape since TSA 8.6.1: ``repeat until nothing_to_pick { retry 2 { retry 3 { move, detect, grasp }, release }
}``. The trees are
also RUN in BehaviorTree.CPP (``test/tools/bt_tree_harness.cpp``: the real control nodes and scripting, scripted
TMR node stubs) and, when TMR's library is current, BUILT with TMR's own node palette.
"""
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (DETECT_REASON_KEY, FAULT_NODES, HOLD_KEY, GATE_KEY,
                                                                 OPERATOR_TEXT, OUTCOME_KEY, OUTCOMES, ProgramRefusals,
                                                                 iter_plans, plan_program, program_overview,
                                                                 program_tree_xml)
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.io import project_yaml

HERE = Path(__file__).parent
DATA = HERE / 'data' / 'm45'
SOURCE = DATA / 'capsule_bin_picking.yaml'
HARNESS = HERE / 'tools' / 'bt_tree_harness.cpp'
GATED = f"{GATE_KEY} != ''"
M45_TOKENS = ('trainit__', '_failureIf', '_skipIf', 'AttachObject', 'DetachObject', 'SetWaypointRelative',
              'AlwaysFailure', '<Script', '<Log')


@pytest.fixture(autouse=True)
def _in_the_data_dir(monkeypatch):
    monkeypatch.chdir(DATA)


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def raw() -> dict:
    return yaml.safe_load(SOURCE.read_text())


def build(mutate=None) -> CanonicalProject:
    data = raw()
    if mutate is not None:
        mutate(data)
    return CanonicalProject.parse_obj(data)


def app(d):
    return d['application']


def main_steps(d):
    return app(d)['tasks'][0]['steps']


def repeat(d):
    return main_steps(d)[1]['repeat']


def drop_until(d):
    """The repeat's declared end removed -- and its presence capture (TSA 8.7.0), which needs it."""
    repeat(d).pop('until')
    repeat(d).pop('presence', None)


def cycle(d):
    """The outer retry: the whole cycle again when the place fails (TSA 8.6.1, the recommended shape)."""
    return repeat(d)['body'][0]['retry']


def retry(d):
    """The inner retry: move to bin_view, detect, grasp."""
    return cycle(d)['body'][0]['retry']


def grasp(d):
    return retry(d)['body'][2]['grasp']


def release(d):
    return cycle(d)['body'][1]['release']


def tree(project) -> str:
    return program_tree_xml(project, plan_program(project))


def root(project) -> ET.Element:
    return ET.fromstring(tree(project).split('\n', 1)[1])


def task_tree(r: ET.Element, name: str = 'empty_the_bin') -> ET.Element:
    return next(t for t in r.iter('BehaviorTree') if t.get('ID') == f'task_{name}')


def refusal(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


# --- legacy: byte-identical ------------------------------------------------------------------------------------

def _task_programs():
    for path in sorted((HERE / 'data').rglob('*.yaml')) + sorted((HERE.parent / 'examples').rglob('*.yaml')):
        if path.parent == DATA:
            continue
        if path.parent.name == 'p0a':                   # P0.a's fixture uses faults: stop on purpose (later)
            continue
        if path.name == 'robot_tasks.yaml':             # P0.a's public sample too: it tracks the held state
            continue
        try:
            data = yaml.safe_load(path.read_text())
        except yaml.YAMLError:
            continue
        if isinstance(data, dict) and (data.get('application') or {}).get('type') == 'task_program':
            yield path


@pytest.mark.parametrize('path', list(_task_programs()), ids=lambda p: p.name)
def test_a_program_without_the_additions_emits_none_of_their_constructs(path, monkeypatch):
    if path.parent.name == 'decompositions':        # an M3.7 decomposition: merged into its cell, as its own test does
        import test_m3_7_decompositions as D
        monkeypatch.chdir(D.SAMPLE if path.stem == 'd_pick_and_place' else D.DATA)
        project = D.build(path.stem)
    else:
        monkeypatch.chdir(path.parent)
        project = CanonicalProject.parse_obj(yaml.safe_load(path.read_text()))
    try:
        text = tree(project)
    except (ProgramRefusals, ValueError):
        pytest.skip(f'{path.name} is a refusal fixture')
    assert not [t for t in M45_TOKENS if t in text], path
    saved = yaml.safe_dump(yaml.safe_load(project_yaml(project))['application'])
    assert not any(k in saved for k in ('faults:', 'until:', 'on_failure:', 'payload:', 'index: all'))


def test_removing_the_additions_gives_the_m3_7_tree():
    """The capsule program without its four additions is an M3.7 program: none of their constructs appears."""
    def plain(d):
        app(d).pop('faults')
        drop_until(d)
        g = grasp(d)
        for k in ('on_failure', 'payload'):
            g.pop(k)
        g['index'] = 0
        release(d).pop('on_failure')
    text = tree(build(plain))
    assert not [t for t in M45_TOKENS if t in text]


# --- the model: load / save ------------------------------------------------------------------------------------

def test_the_capsule_program_saves_as_written_and_loads_back():
    project = build()
    text = project_yaml(project)
    again = CanonicalProject.parse_obj(yaml.safe_load(text))
    assert project_yaml(again) == text
    program = again.application.program
    assert program.faults == 'stop'
    outer = program.tasks[0].steps[1].repeat.body[0].retry
    g = outer.body[0].retry.body[2].grasp
    assert (outer.attempts, outer.body[0].retry.attempts) == (2, 3)
    assert outer.body[1].release.on_failure == 'retreat'
    assert (g.index, g.on_failure) == ('all', 'retreat')
    assert g.payload.dims == [0.042, 0.042, 0.018] and g.payload.offset == [0.0, 0.0, 0.009]
    assert program.tasks[0].steps[1].repeat.until == 'nothing_to_pick'
    assert 'until nothing is left to pick' in program_overview(program)
    assert 'Faults: stop' in program_overview(program)


@pytest.mark.parametrize('change, needle', [
    (lambda d: app(d).update(faults='retry'), "application.faults 'retry'"),
    (lambda d: repeat(d).update(until='empty'), "repeat.until 'empty'"),
    (lambda d: grasp(d).update(on_failure='abort'), "grasp.on_failure 'abort'"),
    (lambda d: release(d).update(on_failure='abort'), "release.on_failure 'abort'"),
    (lambda d: release(d).update(on_failure=True), "release.on_failure True"),
    (lambda d: grasp(d).update(index='every'), "grasp.index 'every'"),
    (lambda d: grasp(d).update(payload={'dims': [0.04, 0.0, 0.02]}), 'every side of the held box must be > 0'),
    (lambda d: grasp(d).update(payload={'dims': [0.04, 0.04]}), 'must be 3 numbers'),
    (lambda d: grasp(d).update(payload={'offset': [0, 0, 0]}), 'the held box [x, y, z]'),
    (lambda d: grasp(d).update(payload={'dims': [0.04, 0.04, 0.02], 'size': 1}), "'dims: [x, y, z]'"),
])
def test_a_malformed_addition_is_refused_at_load(change, needle):
    with pytest.raises(ValueError) as exc:
        build(change)
    assert needle in str(exc.value), str(exc.value)


def test_index_all_at_a_taught_target_is_refused_at_load():
    def taught(d):
        g = grasp(d)
        for k in ('candidates', 'orientation', 'orientation_from', 'max_tilt_deg'):
            g.pop(k)
        g['at'] = 'place'
    with pytest.raises(ValueError, match="'index: all' tries every candidate of a candidate set"):
        build(taught)


# --- F: the fault gate -----------------------------------------------------------------------------------------

FAULT_TAGS = ('MoveWaypoint', 'DetectCandidates', 'CloseGripper', 'OpenGripper', 'VerifyGrasp',
              # TSA 8.6.2 (the M4.5 review): their `fault` since TMR 2.8.1
              'AttachObject', 'DetachObject', 'SetWaypointFromCandidate')


def test_every_fault_output_is_the_gate():
    r = root(build())
    nodes = [el for el in r.iter() if el.tag in FAULT_TAGS]
    assert len(nodes) > 20 and {el.tag for el in nodes} == set(FAULT_TAGS)
    assert all(el.get('fault') == f'{{{GATE_KEY}}}' for el in nodes)
    assert set(FAULT_TAGS) <= set(FAULT_NODES)


def test_every_retry_body_and_fallback_alternative_is_gated():
    r = root(build())
    retries = list(r.iter('RetryUntilSuccessful'))
    assert retries and all(list(x)[0].get('_failureIf') == GATED for x in retries)
    # the candidates' Fallback: each alternative; the until Fallback: its "recovery exhausted" branch
    grasp_alts = [el for el in r.iter('Fallback') if any(c.get('name', '').endswith(':grasp') for s in el for c in s)]
    assert len(grasp_alts) == 1 and [s.get('_failureIf') for s in grasp_alts[0]] == [GATED] * 3


def _main(project) -> ET.Element:
    return list(next(t for t in root(project).iter('BehaviorTree') if t.get('ID') == 'MainTree'))[0]


def _gate(project) -> ET.Element:
    """MainTree's fault-gate Fallback -- TSA 8.7.0: inside the OUTCOME report when the program has an until."""
    top = _main(project)[2]
    return top[0][0] if top[0].tag == 'Sequence' and top[0][0].tag == 'Fallback' else top


def test_the_run_starts_with_the_gate_clear_and_a_fault_ends_it_with_its_reason():
    seq = _main(build())
    assert [c.tag for c in seq] == ['AddCollisionObject', 'Script', 'Fallback']
    # TSA 8.8.0 (M4.5d): a program that grips also starts with nothing held
    assert seq[1].get('code') == f"{GATE_KEY} := ''; {DETECT_REASON_KEY} := ''; {OUTCOME_KEY} := ''; {HOLD_KEY} := false"
    run, terminal = list(_gate(build()))
    assert run.tag == 'Sequence' and run[-1].tag == 'SubTree'       # TSA 8.6.2: after the start check (payload)
    assert terminal.get('_failureIf') == f"{GATE_KEY} == ''"
    assert [c.tag for c in terminal] == ['Log', 'Log', 'AlwaysFailure']
    assert terminal[1].get('message') == f'{{{GATE_KEY}}}'


def test_with_a_payload_the_run_first_proves_the_tool_empty_and_clears_a_phantom():
    """TSA 8.6.2 (the M4.5 review): a part still held from an earlier run (VerifyGrasp released fails held / engaged:
    the fault HOLDING_AT_START) never enters a new cycle, and an attached object an earlier run left in the planning
    scene is detached (idempotent) -- inside the gate's Fallback, so a fault there is logged like any other."""
    run = list(_gate(build()))[0]
    check, detach, call = list(run)
    assert check.tag == 'Fallback'
    verify, held = list(check)
    assert (verify.tag, verify.attrib) == ('VerifyGrasp', {'eef': 'main', 'site': 'main', 'expect': 'released',
                                                           'fault': f'{{{GATE_KEY}}}'})
    # a verdict, or a fault that says held / engaged: HOLDING_AT_START; a state fault keeps its own reason
    assert [(c.tag, c.get('code')) for c in held] == [
        ('Script', f"{GATE_KEY} := ({GATE_KEY} == '' || {GATE_KEY} == 'RELEASE_NOT_CONFIRMED' || "
                   f"{GATE_KEY} == 'NO_RELEASE_COMMAND') ? 'HOLDING_AT_START' : {GATE_KEY}"),
        ('AlwaysFailure', None)]
    assert held.attrib == {} and '&' not in tree(build())
    assert detach.attrib == {'id': 'held_main', 'link': 'tcp', 'fault': f'{{{GATE_KEY}}}'}
    assert call.tag == 'SubTree'


def test_without_a_payload_or_without_the_gate_there_is_no_start_check():
    def no_payload(d):
        grasp(d).pop('payload')
    assert [c.tag for c in _gate(build(no_payload))][0] == 'SubTree'
    assert 'HOLDING_AT_START' not in tree(build(no_payload))

    def no_gate(d):                                         # the additions that need the gate removed with it
        app(d).pop('faults')
        drop_until(d)
        grasp(d).pop('on_failure')
        release(d).pop('on_failure')
    text = tree(build(no_gate))
    assert 'HOLDING_AT_START' not in text and text.count('<DetachObject') == 1     # the release's only


def test_a_user_fallback_and_a_followpath_are_gated_too():
    """The gate covers every control body the program writes, not only the capsule pattern."""
    def alternatives(d):
        main_steps(d).append({'fallback': {'bodies': [[{'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}],
                                                      [{'dwell': {'ms': 10}}]]}})
    r = root(build(alternatives))
    user = [f for f in task_tree(r)[0] if f.tag == 'Fallback'][-1]
    assert [s.get('_failureIf') for s in user] == [GATED, GATED]


# --- U: repeat until nothing_to_pick ---------------------------------------------------------------------------

def _until(project) -> ET.Element:
    wrapper = task_tree(root(project))[0][1]
    assert wrapper.get('name') == 'trainit_step:cycle:repeat'      # the step id keeps the whole repeat
    return wrapper[0]


def test_until_wraps_the_repeat_in_its_normal_end():
    """Without presence (TSA 8.6.x programs): the normal end is NOTHING_PICKABLE -- TSA 8.7.0 adds only the outcome
    Scripts to the 8.6.2 shape."""
    def no_presence(d):
        repeat(d).pop('presence')
    fb = _until(build(no_presence))
    assert fb.tag == 'Fallback' and [c.tag for c in fb] == ['Repeat', 'Sequence', 'Sequence']
    nothing, exhausted = fb[1], fb[2]
    assert nothing.get('_failureIf') == f"{GATED} || {DETECT_REASON_KEY} != 'NO_CANDIDATES'"
    # TSA 8.6.2 (the M4.5 review): an empty candidate set is not an empty bin
    assert nothing[0].get('message') == ("task empty_the_bin step 2: nothing pickable: a fresh detection found no "
                                         'candidate (the bin may still hold parts no grasp can reach): the repeat '
                                         'ends normally')
    assert 'nothing to pick' not in nothing[0].get('message') and 'bin is empty' not in nothing[0].get('message')
    assert [(c.tag, c.get('code')) for c in nothing][1:] == [('Script', f"{OUTCOME_KEY} := 'NOTHING_PICKABLE'")]
    assert exhausted.get('_failureIf') == GATED
    assert 'recovery exhausted' in exhausted[0].get('message')
    assert [(c.tag, c.get('code')) for c in exhausted][1:] == [('Script', f"{OUTCOME_KEY} := 'RECOVERY_EXHAUSTED'"),
                                                               ('AlwaysFailure', None)]


def test_presence_captures_the_providers_objects_after_the_fresh_empty_detection():
    """TSA 8.7.0 (vNext M4.5b): TMR's existing DetectObject on the provider's objects, triggered through its own
    capture service, in the program's frame, keep 1, the detect step's timeout, its fault to the gate; nothing
    present (a fresh empty answer: FAILURE without a fault) = BIN_EMPTY, objects = PICKABLE_EXHAUSTED (a failure, not a
    fault), and the decided presence end is never "recovery exhausted"."""
    fb = _until(build())
    assert [c.tag for c in fb] == ['Repeat', 'Sequence', 'Sequence']
    normal, exhausted = fb[1], fb[2]
    assert normal.get('_failureIf') == f"{GATED} || {DETECT_REASON_KEY} != 'NO_CANDIDATES'"
    assert [c.tag for c in normal] == ['Log', 'Fallback', 'Script', 'Log']
    inverter, present = list(normal[1])
    assert inverter.tag == 'Inverter' and present.get('_failureIf') == GATED
    det = inverter[0]
    assert (det.tag, det.attrib) == ('DetectObject', {
        'detector': 'capsules', 'topic': '/perception/capsules/objects', 'trigger': 'true', 'target_frame': 'base_link',
        'timeout_ms': '3000', 'keep': '1', 'out_key': 'trainit__presence', 'fault': f'{{{GATE_KEY}}}'})
    assert [(c.tag, c.get('code')) for c in present] == [
        ('Script', f"{OUTCOME_KEY} := 'PICKABLE_EXHAUSTED'"), ('Log', None), ('AlwaysFailure', None)]
    assert 'pickable exhausted' in present[1].get('message')
    assert (normal[2].get('code'), normal[2].get('_failureIf')) == (f"{OUTCOME_KEY} := 'BIN_EMPTY'", GATED)
    assert 'bin empty' in normal[3].get('message')
    assert exhausted.get('_failureIf') == f"{GATED} || {DETECT_REASON_KEY} == 'NO_CANDIDATES'"


def test_the_presence_timeout_is_the_detect_steps():
    def slow(d):
        retry(d)['body'][1]['detect']['timeout_ms'] = 4500
    assert _until(build(slow))[1][1][0][0].get('timeout_ms') == '4500'


def test_the_run_reports_exactly_one_outcome_checked_against_how_it_ends():
    """TSA 8.7.0: MainTree resets the outcome, runs the gate's Fallback, then logs ONE "OUTCOME: <code>": a SUCCESS
    keeps only a success outcome, a FAILURE only a failure one, any fault is TERMINAL_FAULT."""
    report = _main(build())[2]
    ok, failed = list(report)
    assert report.tag == 'Fallback' and ok[0].tag == 'Fallback' and ok[0][-1].get('_failureIf') == f"{GATE_KEY} == ''"
    assert [(c.tag, c.get('_skipIf'), c.get('code') or c.get('message')) for c in ok[1:]] == [
        ('Script', f"{OUTCOME_KEY} == 'BIN_EMPTY' || {OUTCOME_KEY} == 'NOTHING_PICKABLE'", f"{OUTCOME_KEY} := ''"),
        ('Log', f"{OUTCOME_KEY} != 'BIN_EMPTY'", f'OUTCOME: BIN_EMPTY -- {OPERATOR_TEXT["BIN_EMPTY"]}'),
        ('Log', f"{OUTCOME_KEY} != 'NOTHING_PICKABLE'", f'OUTCOME: NOTHING_PICKABLE -- {OPERATOR_TEXT["NOTHING_PICKABLE"]}')]
    assert [(c.tag, c.get('_skipIf'), c.get('code') or c.get('message')) for c in failed] == [
        ('Script', f"{OUTCOME_KEY} == 'PICKABLE_EXHAUSTED' || {OUTCOME_KEY} == 'RECOVERY_EXHAUSTED'",
         f"{OUTCOME_KEY} := ''"),
        ('Script', f"{GATE_KEY} == ''", f"{OUTCOME_KEY} := 'TERMINAL_FAULT'"),
        ('Log', f"{OUTCOME_KEY} != 'PICKABLE_EXHAUSTED'", f'OUTCOME: PICKABLE_EXHAUSTED -- {OPERATOR_TEXT["PICKABLE_EXHAUSTED"]}'),
        ('Log', f"{OUTCOME_KEY} != 'RECOVERY_EXHAUSTED'", f'OUTCOME: RECOVERY_EXHAUSTED -- {OPERATOR_TEXT["RECOVERY_EXHAUSTED"]}'),
        ('Log', f"{OUTCOME_KEY} != 'TERMINAL_FAULT'", f'OUTCOME: TERMINAL_FAULT -- {OPERATOR_TEXT["TERMINAL_FAULT"]}'),
        ('AlwaysFailure', None, None)]
    # M4.5d F8: every outcome tells the operator what to do; the failures name an OPERATOR ACTION
    assert set(OPERATOR_TEXT) == set(OUTCOMES)
    assert all(OPERATOR_TEXT[c].startswith('OPERATOR ACTION:') for c in ('PICKABLE_EXHAUSTED', 'RECOVERY_EXHAUSTED',
                                                                          'TERMINAL_FAULT'))
    assert not any(ch in t for t in OPERATOR_TEXT.values() for ch in '<>&"{}')      # XML- and blackboard-safe
    assert set(OUTCOMES) == {'BIN_EMPTY', 'PICKABLE_EXHAUSTED', 'RECOVERY_EXHAUSTED', 'TERMINAL_FAULT',
                             'NOTHING_PICKABLE'}


def test_a_program_without_until_reports_no_outcome():
    """faults: stop alone (no until) keeps the 8.6.2 MainTree: no outcome key, no OUTCOME line."""
    def stop_only(d):
        drop_until(d)
    text = tree(build(stop_only))
    assert OUTCOME_KEY not in text and 'OUTCOME:' not in text and 'Inverter' not in text


@pytest.mark.parametrize('change, needle', [
    (lambda d: repeat(d).update(presence='nobody'), "presence 'nobody': no perception entry is named so"),
    (lambda d: drop_until(d) or repeat(d).update(presence='capsules'),
     "'presence' tells an empty bin from parts no grasp can serve at the repeat's NORMAL end, and this repeat "
     'declares none'),
])
def test_a_presence_the_planner_cannot_serve_is_refused(change, needle):
    assert needle in refusal(build(change))


def test_a_presence_on_a_non_provider_entry_is_refused():
    def colour(d):
        d['perception']['detectors'].append({'name': 'blobs', 'method': 'color_mask', 'params': {'class_id': 'part'}})
        repeat(d)['presence'] = 'blobs'
    assert "presence 'blobs': it is a color_mask entry, not a trainit_perception grasp provider" in refusal(
        build(colour))


def test_presence_without_the_gate_is_refused_through_until():
    assert "'until: nothing_to_pick' needs 'faults: stop'" in refusal(build(lambda d: app(d).pop('faults')))


@pytest.mark.parametrize('bad', ['', 'two words', 3])
def test_a_malformed_presence_is_refused_at_load(bad):
    with pytest.raises(ValueError) as exc:
        build(lambda d: repeat(d).update(presence=bad))
    assert 'repeat.presence' in str(exc.value)


def test_presence_saves_and_loads_back():
    text = project_yaml(build())
    assert 'presence: capsules' in text
    again = CanonicalProject.parse_obj(yaml.safe_load(text))
    assert again.application.program.tasks[0].steps[1].repeat.presence == 'capsules'
    assert 'then a presence capture of capsules' in program_overview(again.application.program)


def test_presence_pins_the_tp_release_that_publishes_the_objects(tmp_path):
    """A SurfaceGrasp provider alone pins TP 0.6.0; its presence capture needs the objects TP publishes since 0.7.0."""
    from trainit_setup_assistant.generator import Orchestrator

    def floor(project, out):
        Orchestrator().generate(project, out)
        xml = ET.parse(out / 'm45_app' / 'package.xml').getroot()
        return next(e.get('version_gte') for e in xml.iter('exec_depend') if e.text.strip() == 'trainit_perception')
    assert floor(build(lambda d: repeat(d).pop('presence')), tmp_path / 'a') == '0.6.0'
    assert floor(build(), tmp_path / 'b') == '0.7.0'


def test_the_detect_reason_is_fresh():
    r = root(build())
    detect = next(r.iter('DetectCandidates'))
    assert detect.get('reason') == f'{{{DETECT_REASON_KEY}}}'
    reset = f"{DETECT_REASON_KEY} := ''"
    # before the detect, and as the first step of every control body (a later failure clears it)
    parent = next(p for p in r.iter() if detect in list(p))
    assert list(parent)[list(parent).index(detect) - 1].get('code') == reset
    for ctl in list(r.iter('Repeat')) + list(r.iter('RetryUntilSuccessful')):
        assert list(ctl)[0][0].get('code') == reset


def test_until_needs_the_fault_gate():
    text = refusal(build(lambda d: app(d).pop('faults')))
    assert "'until: nothing_to_pick' needs 'faults: stop'" in text


def test_until_needs_a_detect_in_its_body():
    def no_detect(d):
        repeat(d)['body'] = [{'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}]
    assert 'no detect step runs in its body' in refusal(build(no_detect))


def test_a_detect_in_a_called_task_serves_until():
    def called(d):
        tasks = app(d)['tasks']
        r = retry(d)
        tasks.append({'name': 'look', 'outputs': [{'name': 'ranked', 'type': 'candidates'}],
                      'steps': [r['body'][1]]})
        r['body'][1] = {'task': 'look'}
    r = root(build(called))
    look = task_tree(r, 'look')
    assert [c.tag for c in look[0]] == ['Sequence']                    # its id wrapper: reset + detect
    assert look[0][0][1].get('reason') == f'{{{DETECT_REASON_KEY}}}'


# --- R: on_failure retreat -------------------------------------------------------------------------------------

def _alternative(r, i=0):
    fb = next(el for el in r.iter('Fallback') if len(el) == 3 and el[0].get('_failureIf') == GATED
              and el[0][0].get('name') == 'trainit_step:pick:grasp')
    return fb[i][0][0]                                    # alternative i -> its step wrapper -> idiom | retreat


def test_the_retreat_opens_and_goes_back_to_the_pregrasp_only_when_reached():
    idiom, back = list(_alternative(root(build())))
    flag = 'empty_the_bin__s2_1_1_3_at_pregrasp'
    tags = [(c.tag, c.get('code') or c.get('waypoint') or c.get('index')) for c in idiom]
    assert tags[:4] == [('Script', f'{flag} := 0'), ('SetWaypointFromCandidate', '0'),
                        ('MoveWaypoint', 'empty_the_bin__s2_1_1_3_pregrasp'), ('Script', f'{flag} := 1')]
    assert back.get('_skipIf') == f'{flag} == 0' and back.get('_failureIf') == GATED
    steps, mark = list(back[0])
    assert [c.tag for c in steps] == ['OpenGripper', 'VerifyGrasp', 'DetachObject', 'SetWaypointRelative',
                                      'MoveWaypoint']
    assert steps[3].get('from') == 'empty_the_bin__s2_1_1_3_pregrasp'
    assert steps[4].get('waypoint') == 'empty_the_bin__s2_1_1_3_retreat'
    assert mark.get('code') == f"{GATE_KEY} := 'RETREAT_FAILED'" and mark.get('_skipIf') == GATED
    assert back[1].tag == 'AlwaysFailure'


def test_the_retreat_is_a_lin_of_the_approach_profile():
    plan = plan_program(build())
    entry = plan.entries['empty_the_bin__s2_1_1_3_retreat']
    assert entry['type'] == 'tcp' and entry['motion'] == 'lin' and entry['profile'] == 'approach'


def test_a_taught_grasp_retreats_to_its_resolved_pregrasp():
    def taught(d):
        g = grasp(d)
        for k in ('candidates', 'index', 'orientation', 'orientation_from', 'max_tilt_deg'):
            g.pop(k)
        g['at'] = 'bin_view'
    project = build(taught)
    plan = plan_program(project)
    pre, back = plan.entries['empty_the_bin__s2_1_1_3_pregrasp'], plan.entries['empty_the_bin__s2_1_1_3_retreat']
    assert back['position'] == pre['position'] and back['orientation'] == pre['orientation']
    assert back['motion'] == 'lin' and pre['motion'] == 'ptp'
    assert 'SetWaypointRelative' not in tree(project)


def test_retreat_needs_the_fault_gate():
    def no_gate(d):
        app(d).pop('faults')
        drop_until(d)
    assert "'on_failure: retreat' needs 'faults: stop'" in refusal(build(no_gate))


# --- R2: release on_failure retreat (TSA 8.6.1, the M4.5 Isaac benchmark) --------------------------------------

PRE_RELEASE = 'empty_the_bin__s2_1_2_prerelease'
RELEASE_BACK = 'empty_the_bin__s2_1_2_retreat'


def _release(r):
    """The release step's idiom, its retreat and its unreached fault (step wrapper -> Fallback -> idiom | retreat
    | fault); returns the first two."""
    wrapper = next(el for el in r.iter('Sequence') if el.get('name') == 'trainit_step:place:release')
    assert [c.tag for c in wrapper] == ['Fallback']
    assert len(wrapper[0]) == 3
    return list(wrapper[0])[:2]


def test_a_release_failed_before_its_prerelease_is_the_fault_place_unreachable_while_holding():
    """The part is still held: the tool is not opened, and the gate stops the run (an enclosing retry never runs
    the cycle again with a part in the gripper). A node's own fault keeps its reason."""
    wrapper = next(el for el in root(build()).iter('Sequence') if el.get('name') == 'trainit_step:place:release')
    unreached = wrapper[0][2]
    flag = 'empty_the_bin__s2_1_2_at_release'
    assert unreached.get('_failureIf') == f"{GATE_KEY} != '' || {flag} != 0"
    assert [(c.tag, c.get('code')) for c in unreached] == [
        ('Script', f"{GATE_KEY} := 'PLACE_UNREACHABLE_WHILE_HOLDING'"), ('AlwaysFailure', None)]


def test_a_grasp_retreat_has_no_unreached_fault():
    """Before its PreGrasp nothing is held: the grasp's failure stays a verdict (the next candidate)."""
    fb = _alternative(root(build()))
    assert len(fb) == 2


def test_the_release_retreat_opens_detaches_and_goes_back_to_the_prerelease_only_when_reached():
    """TSA 8.6.2 (the M4.5 review): "reached" is the RELEASE POSE -- the marker is set after the approach LIN, so a
    refused approach from the PreRelease never opens the tool at the standoff."""
    idiom, back = _release(root(build()))
    flag = 'empty_the_bin__s2_1_2_at_release'
    tags = [(c.tag, c.get('code') or c.get('waypoint') or c.get('expect')) for c in idiom]
    assert tags == [('Script', f'{flag} := 0'), ('MoveWaypoint', PRE_RELEASE),
                    ('MoveWaypoint', 'empty_the_bin__s2_1_2_release'), ('Script', f'{flag} := 1'),
                    ('VerifyGrasp', 'held'),
                    ('OpenGripper', None), ('VerifyGrasp', 'released'), ('DetachObject', None),
                    ('MoveWaypoint', 'empty_the_bin__s2_1_2_retract')]
    assert back.get('_skipIf') == f'{flag} == 0' and back.get('_failureIf') == GATED
    steps, mark = list(back[0])
    # no held guard (the part may be gone), the release confirmation kept, the payload detached, back LIN
    assert [(c.tag, c.get('expect') or c.get('waypoint')) for c in steps] == [
        ('OpenGripper', None), ('VerifyGrasp', 'released'), ('DetachObject', None), ('MoveWaypoint', RELEASE_BACK)]
    assert all(c.get('fault') == f'{{{GATE_KEY}}}' for c in steps)     # DetachObject's too (TMR 2.8.1)
    assert mark.get('code') == f"{GATE_KEY} := 'RETREAT_FAILED'" and mark.get('_skipIf') == GATED
    assert back[1].tag == 'AlwaysFailure'


def test_the_release_retreat_is_a_lin_of_the_approach_profile_to_the_resolved_prerelease():
    plan = plan_program(build())
    pre, back = plan.entries[PRE_RELEASE], plan.entries[RELEASE_BACK]
    assert back['type'] == 'tcp' and back['motion'] == 'lin' and back['profile'] == 'approach'
    assert back['position'] == pre['position'] and back['orientation'] == pre['orientation']
    assert 'configuration' not in back and pre['motion'] == 'ptp'
    readme_how = next(sp.summary[1] for steps in plan.tasks.values() for sp in iter_plans(steps)
                      if sp.kind == 'release')
    assert readme_how.endswith('; on failure at the release pose: open, back to the PreRelease LIN (approach), '
                               'then fail; before it (the part still held): the fault PLACE_UNREACHABLE_WHILE_HOLDING')
    assert "release at 'place', profiles transit / approach, on failure retreat" in program_overview(
        build().application.program)


def test_a_release_retreat_without_a_payload_detaches_nothing():
    def no_payload(d):
        grasp(d).pop('payload')
    _, back = _release(root(build(no_payload)))
    assert [c.tag for c in back[0][0]] == ['OpenGripper', 'VerifyGrasp', 'MoveWaypoint']


def test_without_the_release_retreat_the_release_is_the_plain_idiom():
    """Byte for byte the TSA 8.6.0 release: no flag, no Fallback, no retreat entry."""
    def plain(d):
        release(d).pop('on_failure')
    project = build(plain)
    text = tree(project)
    assert 'at_release' not in text and 'at_prerelease' not in text and RELEASE_BACK not in text
    assert RELEASE_BACK not in plan_program(project).entries
    wrapper = next(el for el in root(project).iter('Sequence') if el.get('name') == 'trainit_step:place:release')
    assert [c.tag for c in wrapper] == ['MoveWaypoint', 'MoveWaypoint', 'VerifyGrasp', 'OpenGripper', 'VerifyGrasp',
                                        'DetachObject', 'MoveWaypoint']


def test_the_release_retreat_needs_the_fault_gate():
    def no_gate(d):
        app(d).pop('faults')
        drop_until(d)
        grasp(d).pop('on_failure')
    text = refusal(build(no_gate))
    assert "(release at 'place')" in text and "'on_failure: retreat' needs 'faults: stop'" in text


@pytest.mark.parametrize('hint', ['recovery', 'on_fail'])
def test_a_misnamed_release_recovery_names_on_failure(hint):
    with pytest.raises(ValueError, match="what a failed grasp or release does before it fails is 'on_failure: "
                                         "retreat'"):
        build(lambda d: release(d).update({hint: 'retreat'}))


# --- A: index all ----------------------------------------------------------------------------------------------

def test_index_all_unrolls_the_literal_indices_of_keep():
    r = root(build())
    assert [el.get('index') for el in r.iter('SetWaypointFromCandidate')] == ['0', '1', '2']
    assert len([el for el in r.iter('Sequence') if el.get('name') == 'trainit_step:pick:grasp']) == 3


@pytest.mark.parametrize('keep', [1, 2, 5])
def test_index_all_follows_keep(keep):
    def k(d):
        retry(d)['body'][1]['detect']['keep'] = keep
    r = root(build(k))
    assert [el.get('index') for el in r.iter('SetWaypointFromCandidate')] == [str(i) for i in range(keep)]


def test_index_all_without_retreat_or_id_is_a_plain_fallback():
    def plain(d):
        app(d).pop('faults')
        drop_until(d)
        grasp(d).pop('on_failure')
        release(d).pop('on_failure')
        retry(d)['body'][2].pop('id')
    r = root(build(plain))
    fb = next(el for el in r.iter('Fallback') if el.find('Sequence/SetWaypointFromCandidate') is not None)
    assert [s.get('_failureIf') for s in fb] == [None] * 3
    assert [s[0].tag for s in fb] == ['SetWaypointFromCandidate'] * 3


def test_index_all_on_a_task_input_is_refused():
    def called(d):
        tasks = app(d)['tasks']
        r = retry(d)
        tasks.append({'name': 'pick', 'inputs': [{'name': 'ranked', 'type': 'candidates'}],
                      'steps': [{'grasp': r['body'][2]['grasp']}]})
        r['body'][2] = {'task': 'pick'}
    assert "'index: all' tries every candidate 'ranked' keeps, and TSA cannot know" in refusal(build(called))


# --- P: payload ------------------------------------------------------------------------------------------------

def test_the_payload_is_attached_after_the_verified_grasp_and_detached_after_each_open():
    r = root(build())
    idiom = list(_alternative(r))[0]
    tags = [c.tag for c in idiom]
    at = tags.index('AttachObject')
    assert tags[at - 2:at + 2] == ['CloseGripper', 'VerifyGrasp', 'AttachObject', 'MoveWaypoint']
    attach = idiom[at]
    gate = f'{{{GATE_KEY}}}'
    assert attach.attrib == {'id': 'held_main', 'link': 'tcp', 'dims': '0.042;0.042;0.018', 'position': '0;0;0.009',
                             'touch_links': 'tcp', 'fault': gate}
    idiom, back = _release(r)
    tags = [c.tag for c in idiom]
    assert tags[tags.index('OpenGripper'):] == ['OpenGripper', 'VerifyGrasp', 'DetachObject', 'MoveWaypoint']
    assert idiom.find('DetachObject').attrib == {'id': 'held_main', 'link': 'tcp', 'fault': gate}
    assert back.find('Fallback/Sequence/DetachObject').attrib == {'id': 'held_main', 'link': 'tcp', 'fault': gate}


def test_the_attached_payload_may_touch_the_tcp_and_the_cells_links_nearest_the_tool():
    """TSA 8.6.2 (the M4.5 review): AttachObject's touch_links (TMR 2.8.1) -- the attach link first, then
    scene.touch_links (derived from the robot: the links nearest the tool), each once."""
    def links(d):
        d['scene']['touch_links'] = ['wrist_3_link', 'tcp', 'tool0']
    assert next(root(build(links)).iter('AttachObject')).get('touch_links') == 'tcp;wrist_3_link;tool0'


def test_the_capsule_program_declares_the_tmr_2_9_0_floor():
    """The fault outputs of AttachObject / DetachObject / SetWaypointFromCandidate and AttachObject's touch_links
    exist since TMR 2.8.1; TSA 8.8.0 (M4.5d): the motions' `holding` since TMR 2.9.0 -- a program that uses them
    declares the newest (an older TMR refuses the unknown ports)."""
    from trainit_setup_assistant.generator import runtime_requirements as rr
    use = rr.use_of(trees=[tree(build())])
    assert rr.requirement(use).version == '2.9.0'
    assert {'scene_fault', 'attach_touch_links', 'candidate_fault', 'held_motion_fault'} <= set(use.capabilities)


def test_a_payload_without_offset_is_centred_on_the_tcp_and_a_tool_open_detaches():
    def change(d):
        grasp(d)['payload'] = {'dims': [0.05, 0.05, 0.02]}
        main_steps(d).insert(0, {'tool': {'open': {}}})
    r = root(build(change))
    assert next(r.iter('AttachObject')).get('position') == '0;0;0'
    first = list(task_tree(r)[0])
    assert [c.tag for c in first[:3]] == ['OpenGripper', 'VerifyGrasp', 'DetachObject']
    assert next(r.iter('AttachObject')).get('touch_links') == 'tcp'


def test_a_payload_id_taken_by_a_scene_object_is_refused():
    def clash(d):
        d['scene']['objects'].append({'id': 'held_main', 'shape': 'box', 'dims': [0.1, 0.1, 0.1],
                                      'frame': 'base_link', 'position': [1.0, 1.0, 0.0]})
    assert "attached to the planning scene as 'held_main', and a scene object has that id" in refusal(build(clash))


def _r_i0_06(project):
    from trainit_setup_assistant.composer.advice import configuration_advice
    return [w for w in configuration_advice(project) if w.startswith('R-I0-06')]


def test_a_grasp_with_a_payload_is_not_warned_that_nothing_attaches():
    """TSA 8.6.1 (the M4.5 Isaac benchmark): R-I0-06 says nothing attaches to the tool in the planning scene --
    false for a grasp that declares its payload (AttachObject after the verified grasp). The scene has an object
    (the table) and no grasp target, as in the capsule cell."""
    project = build()
    assert project.scene.objects and not project.scene.grasp_target_ids()
    assert _r_i0_06(project) == []
    assert _r_i0_06(build(lambda d: grasp(d).pop('payload'))) == [
        "R-I0-06: task 'empty_the_bin' step 2.1.1.3: a grasp, and no object of the cell scene is a grasp target: "
        'nothing attaches to the tool in the planning scene (RViz does not show the part held, the planner ignores '
        "it) -- tick 'grasp' on the part you pick (Step 2)"]


def test_r_i0_06_names_the_first_grasp_without_a_payload():
    def second(d):
        g = dict(grasp(d))
        g.pop('payload')
        g['index'] = 0
        retry(d)['body'].append({'grasp': g})
    warnings = _r_i0_06(build(second))
    assert len(warnings) == 1 and warnings[0].startswith("R-I0-06: task 'empty_the_bin' step 2.1.1.4: a grasp")


# --- the example: generated -------------------------------------------------------------------------------------

def test_the_capsule_program_generates_its_bundle(tmp_path):
    from trainit_setup_assistant.generator import Orchestrator
    out = tmp_path / 'ws'
    Orchestrator().generate(build(), out)
    app_dir = out / 'm45_app'
    text = (app_dir / 'bt_trees' / 'task_program.xml').read_text()
    assert text == tree(build())
    params = yaml.safe_load((app_dir / 'config' / 'bt_params.yaml').read_text())
    flat = yaml.safe_dump(params)
    assert 'empty_the_bin__s2_1_1_3_retreat' in flat
    readme = (app_dir / 'README.md').read_text()
    assert 'ends NORMALLY when nothing is left to pick' in readme
    assert 'each candidate in turn until one succeeds' in readme


# --- the trees in BehaviorTree.CPP -------------------------------------------------------------------------------

def _bt_include() -> Path:
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '/opt/ros/humble').split(':') + ['/opt/ros/humble']:
        if (Path(prefix) / 'include' / 'behaviortree_cpp' / 'bt_factory.h').is_file():
            return Path(prefix)
    return None


def _tmr_lib() -> Path:
    explicit = os.environ.get('TRAINIT_TMR_LIB')
    if explicit:
        return Path(explicit)
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(':'):
        lib = Path(prefix) / 'lib' / 'libtrainit_motion_runtime.so'
        if lib.is_file():
            return lib
    return None


def _compile(out: Path, with_tmr: Path = None) -> Path:
    bt = _bt_include()
    if shutil.which('g++') is None or bt is None:
        pytest.skip('the BehaviorTree.CPP harness needs g++ and behaviortree_cpp (a sourced ROS 2 Humble)')
    cmd = ['g++', '-std=c++17', '-O0', '-o', str(out), str(HARNESS), f'-I{bt / "include"}', f'-L{bt / "lib"}',
           '-lbehaviortree_cpp', f'-Wl,-rpath,{bt / "lib"}']
    if with_tmr is not None:
        # DT_RPATH (not RUNPATH): this library, before any other on LD_LIBRARY_PATH (a sourced, older install)
        cmd += ['-DWITH_TMR', f'-L{with_tmr.parent}', '-ltrainit_motion_runtime', '-Wl,--allow-shlib-undefined',
                '-Wl,--disable-new-dtags', f'-Wl,-rpath,{with_tmr.parent}']
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return out


@pytest.fixture(scope='module')
def harness(tmp_path_factory):
    return _compile(tmp_path_factory.mktemp('harness') / 'bt_tree_harness')


def run_tree(harness, tmp_path, project, scenario: str):
    (tmp_path / 'tree.xml').write_text(tree(project))
    (tmp_path / 'scenario.txt').write_text(scenario)
    proc = subprocess.run([str(harness), '--run', str(tmp_path / 'tree.xml'), str(tmp_path / 'scenario.txt')],
                          capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    return proc.stdout.splitlines()


# TSA 8.7.0 (M4.5b): the presence capture after the last fresh empty detection -- a fresh empty answer (FAILURE, no
# fault): nothing present, the bin is empty
EMPTY = 'DetectObject:capsules -> FAILURE'
PRESENT = 'DetectObject:capsules -> SUCCESS'


def outcome_lines(lines):
    """The run's OUTCOME line(s) as 'LOG OUTCOME: <code>' -- each checked to carry its operator text (M4.5d F8)."""
    out = []
    for line in lines:
        if line.startswith('LOG OUTCOME: '):
            code, _, why = line.partition(' -- ')
            assert why == OPERATOR_TEXT[code[len('LOG OUTCOME: '):]], line
            out.append(code)
    return out


def bare(lines):
    """The log with each OUTCOME line cut to its code (order-sensitive checks)."""
    return [line.partition(' -- ')[0] if line.startswith('LOG OUTCOME: ') else line for line in lines]


def ticks(lines, ident):
    return [line for line in lines if line.startswith(f'TICK {ident} ')]


def test_bt_a_verdict_retreats_the_next_candidate_succeeds_and_an_empty_bin_ends_normally(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'VerifyGrasp:held -> FAILURE reason=NO_OBJECT',          # candidate 0: nothing came up (a verdict)
        'DetectCandidates:capsules -> SUCCESS x2',               # cycles 1 and 2 pick
        # then the bin is empty: 3 fresh tries of the inner retry, twice (the outer retry re-runs the cycle)
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert lines[-1] == 'RESULT SUCCESS'
    assert any('bin empty' in line for line in lines)
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 8
    assert len(ticks(lines, 'MoveWaypoint:empty_the_bin__s2_1_1_3_retreat')) == 1      # the one retreat
    assert ticks(lines, 'SetWaypointFromCandidate:1') == ['TICK SetWaypointFromCandidate:1 SUCCESS']
    assert len(ticks(lines, 'MoveWaypoint:empty_the_bin__s2_1_2_retract')) == 2          # two parts placed
    # the start check's detach (TSA 8.6.2), then the grasp retreat's and the two places'
    assert len(ticks(lines, 'AttachObject')) == 2 and len(ticks(lines, 'DetachObject')) == 1 + 3


def test_bt_a_pregrasp_never_reached_skips_the_retreat(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        # planning refused: every candidate, 3 tries, the cycle twice
        'MoveWaypoint:empty_the_bin__s2_1_1_3_pregrasp -> FAILURE x18']))
    assert lines[-1] == 'RESULT FAILURE'
    assert not ticks(lines, 'OpenGripper') and not ticks(lines, 'MoveWaypoint:empty_the_bin__s2_1_1_3_retreat')
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 6                 # 2 x R = 2 x 3 fresh detections
    assert len(ticks(lines, 'MoveWaypoint:empty_the_bin__s2_1_1_3_pregrasp')) == 18  # 2 x R x K attempts, bounded
    assert any('recovery exhausted' in line for line in lines)
    assert not any('bin empty' in line for line in lines)


def test_bt_a_fault_is_never_retried_and_ends_the_run_with_its_reason(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'DetectCandidates:capsules -> FAILURE reason=DETECTION_TIMEOUT '
                                                 'fault=DETECTION_TIMEOUT\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1
    assert 'LOG DETECTION_TIMEOUT' in lines and not any('recovery exhausted' in line for line in lines)
    assert not any('bin empty' in line for line in lines)


def test_bt_a_motion_fault_while_holding_skips_the_retreat_and_every_later_candidate(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'MoveWaypoint:empty_the_bin__s2_1_1_3_retract -> FAILURE '
                                                 'fault=MOTION_FAULT:CONTROLLER_ABORTED\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert not ticks(lines, 'OpenGripper') and not ticks(lines, 'SetWaypointFromCandidate:1')
    assert 'LOG MOTION_FAULT:CONTROLLER_ABORTED' in lines


def test_bt_a_failed_retreat_is_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'VerifyGrasp:held -> FAILURE reason=NO_OBJECT',
        'MoveWaypoint:empty_the_bin__s2_1_1_3_retreat -> FAILURE']))       # back to the PreGrasp refused (no fault)
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG RETREAT_FAILED' in lines
    assert not ticks(lines, 'SetWaypointFromCandidate:1')               # never the next candidate
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1           # never a re-detect


def test_bt_an_empty_detection_before_a_later_failure_is_not_nothing_to_pick(harness, tmp_path):
    """The reason key is fresh: an empty detection, then a try whose detect succeeds and whose candidates all
    fail, then tries that fail before their detect (the last of the inner retry, then every try of the cycle's
    second run) -- recovery exhausted, never "nothing pickable"."""
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES',
        'MoveWaypoint:empty_the_bin__s2_1_1_3_pregrasp -> FAILURE x3',
        'MoveWaypoint:bin_view -> SUCCESS x2',
        'MoveWaypoint:bin_view -> FAILURE x4']))
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 2
    assert lines[-1] == 'RESULT FAILURE'
    assert any('recovery exhausted' in line for line in lines)


HELD_OK = 'VerifyGrasp:held -> SUCCESS'                                 # the grasp's verification
DROPPED = 'VerifyGrasp:held -> FAILURE reason=OBJECT_LOST'            # the release's held guard: dropped in transport


def _after(lines, marker, n):
    i = next(k for k, line in enumerate(lines) if line.startswith(marker))
    return [line.split()[1] for line in lines[i + 1:i + 1 + n]]


def test_bt_a_part_dropped_in_transport_retreats_and_the_cycle_runs_again(harness, tmp_path):
    """The held guard says OBJECT_LOST at the place (a verdict): the tool is opened (no held guard), the payload
    detached, the arm back at the PreRelease; the release fails, the outer retry re-runs the cycle (a fresh
    detection), the next part is placed; the bin then empties -- the program ends normally."""
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        HELD_OK, DROPPED,                                                # cycle 1, try 1: dropped
        'DetectCandidates:capsules -> SUCCESS x2',                       # try 2 (re-detect) picks and places
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))   # then the bin is empty
    assert lines[-1] == 'RESULT SUCCESS'
    assert any('bin empty' in line for line in lines)
    assert _after(lines, 'TICK VerifyGrasp:held FAILURE', 4) == [
        'OpenGripper', 'VerifyGrasp:released', 'DetachObject', f'MoveWaypoint:{RELEASE_BACK}']
    assert len(ticks(lines, f'MoveWaypoint:{RELEASE_BACK}')) == 1
    assert len(ticks(lines, 'MoveWaypoint:empty_the_bin__s2_1_2_retract')) == 1     # one part placed
    assert len(ticks(lines, 'AttachObject')) == 2 and len(ticks(lines, 'DetachObject')) == 1 + 2   # + the start's
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 2 + 6
    assert not any('recovery exhausted' in line for line in lines)


def test_bt_a_part_dropped_every_time_is_bounded_and_recovery_exhausted(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([HELD_OK, DROPPED, HELD_OK, DROPPED]))
    assert lines[-1] == 'RESULT FAILURE'
    assert any('recovery exhausted' in line for line in lines)
    assert not any('bin empty' in line for line in lines)
    assert len(ticks(lines, f'MoveWaypoint:{RELEASE_BACK}')) == 2       # once per run of the cycle, 2 runs
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 2
    assert not ticks(lines, 'MoveWaypoint:empty_the_bin__s2_1_2_retract')


def test_bt_a_part_dropped_then_an_empty_bin_ends_normally(harness, tmp_path):
    """The until semantics through the nested retry: after the drop, the cycle's second run finds the bin empty
    (3 fresh detections): the latest failure is a fresh NO_CANDIDATES -- nothing pickable, SUCCESS."""
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        HELD_OK, DROPPED,
        'DetectCandidates:capsules -> SUCCESS',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x3', EMPTY]))
    assert lines[-1] == 'RESULT SUCCESS' and any('bin empty' in line for line in lines)
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 4


def test_bt_a_failed_release_retreat_is_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        HELD_OK, DROPPED, f'MoveWaypoint:{RELEASE_BACK} -> FAILURE']))   # back to the PreRelease refused
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG RETREAT_FAILED' in lines
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1          # never the cycle again
    assert not any('recovery exhausted' in line for line in lines)


def test_bt_a_failed_release_verification_stays_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'VerifyGrasp:released -> SUCCESS\n'      # the start check
                                                 'VerifyGrasp:released -> FAILURE reason=RELEASE_NOT_CONFIRMED '
                                                 'fault=RELEASE_NOT_CONFIRMED\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG RELEASE_NOT_CONFIRMED' in lines
    assert len(ticks(lines, 'OpenGripper')) == 1 and not ticks(lines, f'MoveWaypoint:{RELEASE_BACK}')
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1


def test_bt_a_place_unreachable_while_holding_is_a_fault_never_a_new_cycle(harness, tmp_path):
    """Planning to the PreRelease refused (a VERDICT, no fault of its own) while the part is held: the tool is NOT
    opened, the run stops -- the outer retry never runs the cycle again with the part in the gripper. TSA 8.8.0
    (M4.5d): the motion itself reports the fault, with its own cause ("... while holding"), which the release's
    PLACE_UNREACHABLE_WHILE_HOLDING never overwrites (a fault keeps its own reason)."""
    lines = run_tree(harness, tmp_path, build(), f'MoveWaypoint:{PRE_RELEASE} -> FAILURE x2\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG MOTION_FAULT: SCRIPTED during plan while holding' in lines
    assert not ticks(lines, 'OpenGripper') and not ticks(lines, f'MoveWaypoint:{RELEASE_BACK}')
    assert len(ticks(lines, f'MoveWaypoint:{PRE_RELEASE}')) == 1
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1           # never the cycle again
    assert not any('recovery exhausted' in line or 'bin empty' in line for line in lines)


def test_bt_a_motion_fault_on_the_way_to_the_place_keeps_its_own_reason(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), f'MoveWaypoint:{PRE_RELEASE} -> FAILURE '
                                                 'fault=MOTION_FAULT:CONTROLLER_ABORTED\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG MOTION_FAULT:CONTROLLER_ABORTED' in lines and 'LOG PLACE_UNREACHABLE_WHILE_HOLDING' not in lines
    assert not ticks(lines, 'OpenGripper') and len(ticks(lines, 'DetectCandidates:capsules')) == 1


def test_bt_the_trees_build_with_the_runtime_node_palette(tmp_path):
    """The capsule tree and its variants, built by TMR's own factory (every node name and port as the runtime
    declares it). Skips when no TMR library is found, or when the one found predates TMR 2.8.1 (a stale
    install: rebuild it, or point TRAINIT_TMR_LIB at a current build)."""
    lib = _tmr_lib()
    if lib is None or not lib.is_file():
        pytest.skip('no libtrainit_motion_runtime.so (source the workspace, or set TRAINIT_TMR_LIB)')
    if b'PLANNING_SCENE_UPDATE_FAILED' not in lib.read_bytes():
        pytest.skip(f'{lib} predates TMR 2.8.1 (no AttachObject fault / touch_links port): rebuild TMR or set '
                    'TRAINIT_TMR_LIB to a current build')
    exe = _compile(tmp_path / 'bt_tree_harness_tmr', with_tmr=lib)

    def taught(d):
        g = grasp(d)
        for k in ('candidates', 'index', 'orientation', 'orientation_from', 'max_tilt_deg'):
            g.pop(k)
        g['at'] = 'bin_view'
    for i, project in enumerate((build(), build(taught))):
        (tmp_path / f'tree{i}.xml').write_text(tree(project))
        proc = subprocess.run([str(exe), '--parse', str(tmp_path / f'tree{i}.xml')], capture_output=True,
                              text=True, timeout=60)
        assert proc.returncode == 0 and proc.stdout.startswith('PARSED'), proc.stdout + proc.stderr


# --- TSA 8.6.2 (the M4.5 independent review) ----------------------------------------------------------------------

RELEASE_POSE = 'empty_the_bin__s2_1_2_release'


def test_bt_a_refused_approach_to_the_release_pose_is_a_fault_with_the_tool_closed(harness, tmp_path):
    """The PreRelease reached, the approach LIN to the release pose refused (a VERDICT, the part still held): never
    the retreat (which would open the tool at the standoff and drop the part) -- a fault: TSA 8.8.0 (M4.5d) the
    motion's own, "... while holding" (before: PLACE_UNREACHABLE_WHILE_HOLDING)."""
    lines = run_tree(harness, tmp_path, build(), f'MoveWaypoint:{RELEASE_POSE} -> FAILURE\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG MOTION_FAULT: SCRIPTED during plan while holding' in lines
    assert ticks(lines, f'MoveWaypoint:{PRE_RELEASE}') == [f'TICK MoveWaypoint:{PRE_RELEASE} SUCCESS']
    assert not ticks(lines, 'OpenGripper') and not ticks(lines, f'MoveWaypoint:{RELEASE_BACK}')
    assert len(ticks(lines, 'DetachObject')) == 1                       # the start check's only: the part stays held
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1           # never the cycle again
    assert not any('recovery exhausted' in line or 'bin empty' in line for line in lines)


def test_bt_a_failed_retract_after_the_release_pose_still_retreats(harness, tmp_path):
    """After the release pose a VERDICT (here the retract refused) still retreats: the marker is set there."""
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'MoveWaypoint:empty_the_bin__s2_1_2_retract -> FAILURE',
        'DetectCandidates:capsules -> SUCCESS',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert len(ticks(lines, f'MoveWaypoint:{RELEASE_BACK}')) == 1
    assert 'LOG PLACE_UNREACHABLE_WHILE_HOLDING' not in lines and lines[-1] == 'RESULT SUCCESS'


@pytest.mark.parametrize('outcome, reason', [
    ('FAILURE reason=RELEASE_NOT_CONFIRMED fault=RELEASE_NOT_CONFIRMED', 'HOLDING_AT_START'),   # still held
    ('FAILURE reason=NO_RELEASE_COMMAND fault=NO_RELEASE_COMMAND', 'HOLDING_AT_START'),         # still engaged
    ('FAILURE reason=OBJECT_LOST', 'HOLDING_AT_START'),                                         # a verdict
    # the state cannot be read: the tool is not proven empty, the run stops with the state fault's own reason
    ('FAILURE reason=STATE_UNKNOWN fault=STATE_UNKNOWN', 'STATE_UNKNOWN')])
def test_bt_a_part_held_at_the_start_is_the_fault_holding_at_start(harness, tmp_path, outcome, reason):
    """A part still held from an earlier run: the run never starts a cycle -- no detach of the held part's object, no
    detection, no motion. A tool whose state cannot be read is not proven empty either: the same stop, its reason."""
    lines = run_tree(harness, tmp_path, build(), f'VerifyGrasp:released -> {outcome}\n')
    assert lines[-1] == 'RESULT FAILURE'
    assert f'LOG {reason}' in lines and ('LOG HOLDING_AT_START' in lines) == (reason == 'HOLDING_AT_START')
    assert [line.split()[1] for line in lines if line.startswith('TICK ')] == ['AddCollisionObject',
                                                                                'VerifyGrasp:released']


def test_bt_a_refused_start_detach_is_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'DetachObject -> FAILURE fault=PLANNING_SCENE_UPDATE_FAILED\n')
    assert lines[-1] == 'RESULT FAILURE' and 'LOG PLANNING_SCENE_UPDATE_FAILED' in lines
    assert not ticks(lines, 'DetectCandidates:capsules') and not ticks(lines, 'MoveWaypoint:home')


def test_bt_a_refused_attach_ends_the_run_never_the_retreat_or_the_next_candidate(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'AttachObject -> FAILURE fault=PLANNING_SCENE_UPDATE_FAILED\n')
    assert lines[-1] == 'RESULT FAILURE' and 'LOG PLANNING_SCENE_UPDATE_FAILED' in lines
    assert not ticks(lines, 'OpenGripper') and not ticks(lines, 'SetWaypointFromCandidate:1')
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1


def test_bt_a_candidate_configuration_refusal_is_a_fault_a_candidate_verdict_is_not(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'SetWaypointFromCandidate:0 -> FAILURE reason=INVALID_CONFIGURATION '
                                                 'fault=INVALID_CONFIGURATION\n')
    assert lines[-1] == 'RESULT FAILURE' and 'LOG INVALID_CONFIGURATION' in lines
    assert not ticks(lines, 'SetWaypointFromCandidate:1') and len(ticks(lines, 'DetectCandidates:capsules')) == 1
    # APPROACH_OUTSIDE_ENVELOPE: a verdict on that candidate -- the next one is tried
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'SetWaypointFromCandidate:0 -> FAILURE reason=APPROACH_OUTSIDE_ENVELOPE',
        'DetectCandidates:capsules -> SUCCESS',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert ticks(lines, 'SetWaypointFromCandidate:1') == ['TICK SetWaypointFromCandidate:1 SUCCESS']
    assert lines[-1] == 'RESULT SUCCESS'


# --- TSA 8.7.0 (vNext M4.5b): the outcomes, RUN in BehaviorTree.CPP ------------------------------------------------

def test_bt_an_empty_bin_is_bin_empty_and_success(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'DetectCandidates:capsules -> SUCCESS',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert lines[-1] == 'RESULT SUCCESS' and outcome_lines(lines) == ['LOG OUTCOME: BIN_EMPTY']
    assert ticks(lines, 'DetectObject:capsules') == ['TICK DetectObject:capsules FAILURE']
    assert any('bin empty' in line for line in lines) and not any('recovery exhausted' in line for line in lines)


def test_bt_objects_without_a_grasp_are_pickable_exhausted_a_failure_not_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', PRESENT]))
    assert lines[-1] == 'RESULT FAILURE' and outcome_lines(lines) == ['LOG OUTCOME: PICKABLE_EXHAUSTED']
    assert len(ticks(lines, 'DetectObject:capsules')) == 1 and any('pickable exhausted' in line for line in lines)
    assert not any(line.startswith('LOG FAULT') for line in lines)
    assert not any('recovery exhausted' in line or 'bin empty' in line for line in lines)


@pytest.mark.parametrize('fault', ['DETECTION_TIMEOUT', 'TRIGGER_UNAVAILABLE', 'TF_LOOKUP_FAILED'])
def test_bt_a_fault_of_the_presence_capture_is_terminal_never_an_empty_bin(harness, tmp_path, fault):
    lines = run_tree(harness, tmp_path, build(), '\n'.join([
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6',
        f'DetectObject:capsules -> FAILURE fault={fault}']))
    assert lines[-1] == 'RESULT FAILURE' and outcome_lines(lines) == ['LOG OUTCOME: TERMINAL_FAULT']
    assert f'LOG {fault}' in lines
    assert not any('bin empty' in line or 'pickable exhausted' in line or 'recovery exhausted' in line
                   for line in lines)


def test_bt_recovery_exhaustion_is_recovery_exhausted(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'MoveWaypoint:empty_the_bin__s2_1_1_3_pregrasp -> FAILURE x18\n')
    assert lines[-1] == 'RESULT FAILURE' and outcome_lines(lines) == ['LOG OUTCOME: RECOVERY_EXHAUSTED']
    assert not ticks(lines, 'DetectObject:capsules')                 # not a fresh empty detection: no presence


def test_bt_a_fault_is_terminal_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'DetectCandidates:capsules -> FAILURE reason=DETECTION_TIMEOUT '
                                                 'fault=DETECTION_TIMEOUT\n')
    assert lines[-1] == 'RESULT FAILURE' and outcome_lines(lines) == ['LOG OUTCOME: TERMINAL_FAULT']
    # the fault's own lines first, the outcome last
    assert lines.index('LOG DETECTION_TIMEOUT') < bare(lines).index('LOG OUTCOME: TERMINAL_FAULT')


def test_bt_a_fault_at_the_start_check_is_terminal_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), 'VerifyGrasp:released -> FAILURE reason=STATE_UNKNOWN '
                                                 'fault=STATE_UNKNOWN\n')
    assert lines[-1] == 'RESULT FAILURE' and outcome_lines(lines) == ['LOG OUTCOME: TERMINAL_FAULT']


def test_bt_without_presence_the_normal_end_is_nothing_pickable(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(lambda d: repeat(d).pop('presence')),
                     'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6\n')
    assert lines[-1] == 'RESULT SUCCESS' and outcome_lines(lines) == ['LOG OUTCOME: NOTHING_PICKABLE']
    assert any('nothing pickable' in line for line in lines) and not ticks(lines, 'DetectObject:capsules')


def test_bt_an_outcome_contradicted_by_the_runs_end_is_not_reported(harness, tmp_path):
    """BIN_EMPTY, then a later step fails with a verdict: the run is a FAILURE, so it reports no success outcome
    (and no failure outcome was decided): no OUTCOME line, never a BIN_EMPTY with exit 1."""
    def home_after(d):
        main_steps(d).append({'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}})
    lines = run_tree(harness, tmp_path, build(home_after), '\n'.join([
        'MoveWaypoint:home -> SUCCESS', 'MoveWaypoint:home -> FAILURE',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert lines[-1] == 'RESULT FAILURE' and outcome_lines(lines) == []
    assert any('bin empty' in line for line in lines)


# --- TSA 8.8.0 (vNext M4.5d, ADR-0024 amendment M4.5d): a motion that fails while a part is held is a FAULT ---------
# The owner's M4.5d section 19 list: planning failure while held, execution failure while held, a free motion
# failure (not held), a release path failure, a retry / fallback containing a held part, recovery, no false held,
# the fault code. The motion stubs follow TMR 2.9.0's `holding` rule (test/tools/bt_tree_harness.cpp; TMR's
# test_m45d_held_motion_fault pins the node itself).

def held_transit(d):
    """A transfer move between the grasp and the release (a reference cell's direct path shape): home, with the part."""
    cycle(d)['body'].insert(1, {'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}})


def held_fallback(d):
    """The cycle's pick + transfer + place as the FIRST alternative of a fallback whose second alternative is an
    unrelated move: a held part's failed transfer must never fall through to it."""
    body = cycle(d)['body']
    cycle(d)['body'] = [{'fallback': {'bodies': [
        [body[0], {'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}, body[1]],
        [{'move': {'to': 'place', 'type': 'ptp', 'profile': 'transit'}}]]}}]


def holds(lines):
    return [line.split()[1:] for line in lines if line.startswith('HOLD ')]


def test_the_tree_tracks_the_held_state_on_the_gripping_nodes_and_every_motion_reads_it():
    r = root(build(held_transit))
    motions = [el for tag in ('MoveWaypoint', 'FollowPath') for el in r.iter(tag)]
    assert motions and all(el.get('holding') == f'{{{HOLD_KEY}}}' for el in motions)
    assert all(el.get('_onSuccess') == f'{HOLD_KEY} := true' for el in r.iter('CloseGripper'))
    held = [el for el in r.iter('VerifyGrasp') if el.get('expect') == 'held']
    assert held and all(el.get('_onSuccess') == f'{HOLD_KEY} := true'
                        and el.get('_onFailure') == f'{HOLD_KEY} := false' for el in held)
    assert all(el.get('_onSuccess') == f'{HOLD_KEY} := false' for el in r.iter('OpenGripper'))
    released = [el for el in r.iter('VerifyGrasp') if el.get('expect') == 'released']
    assert released and not any(el.get('_onSuccess') or el.get('_onFailure') for el in released)


def test_a_program_that_never_stops_on_a_fault_tracks_nothing():
    def no_stop(d):
        drop_until(d)
        app(d).pop('faults', None)
        grasp(d).pop('on_failure', None)
        release(d).pop('on_failure', None)
    assert HOLD_KEY not in tree(build(no_stop))


def test_bt_a_planning_failure_while_held_is_a_fault_never_a_release_or_a_new_cycle(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(held_transit), '\n'.join([
        'MoveWaypoint:home -> SUCCESS',                       # the program's first move (nothing held)
        'MoveWaypoint:home -> FAILURE']))                     # the held transfer: no plan (a target verdict)
    assert lines[-1] == 'RESULT FAILURE'
    assert 'LOG MOTION_FAULT: SCRIPTED during plan while holding' in lines
    assert 'LOG OUTCOME: TERMINAL_FAULT' in bare(lines)
    assert not ticks(lines, f'MoveWaypoint:{PRE_RELEASE}') and not ticks(lines, 'OpenGripper')
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1                  # never a new cycle
    assert holds(lines)[0] == ['MoveWaypoint:home', 'free'] and ['MoveWaypoint:home', 'held'] in holds(lines)


def test_bt_an_execution_fault_while_held_keeps_its_cause(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(held_transit), '\n'.join([
        'MoveWaypoint:home -> SUCCESS',
        'MoveWaypoint:home -> FAILURE fault=MOTION_FAULT:CONTROL_FAILED_while_holding']))
    assert lines[-1] == 'RESULT FAILURE' and 'LOG MOTION_FAULT:CONTROL_FAILED_while_holding' in lines
    assert not ticks(lines, 'OpenGripper')


def unconfirmed(d):
    """The grasp is NOT confirmed (require_grasp_confirmation false): no VerifyGrasp follows the close."""
    held_transit(d)
    top = next(v for v in d.values() if isinstance(v, dict) and 'gripper' in v)
    top['gripper']['contract']['verification'] = {'require_grasp_confirmation': False,
                                                  'require_release_confirmation': False}


def test_an_unconfirmed_close_counts_as_held_fail_closed():
    """ADR-0024 amendment M4.5d (review): a CloseGripper that succeeds sets the held state even when no hold
    verification follows -- an unconfirmed hold is never assumed empty."""
    xml = tree(build(unconfirmed))
    assert 'VerifyGrasp' not in xml or 'expect="held"' not in xml
    assert f'_onSuccess="{HOLD_KEY} := true"' in xml


def test_bt_an_unconfirmed_close_makes_the_next_failure_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(unconfirmed), '\n'.join([
        'MoveWaypoint:home -> SUCCESS',                       # the program's first move (nothing held)
        'MoveWaypoint:home -> FAILURE']))                     # after the unconfirmed close: held -> a FAULT
    assert ['MoveWaypoint:home', 'held'] in holds(lines)
    assert 'LOG MOTION_FAULT: SCRIPTED during plan while holding' in lines
    assert lines[-1] == 'RESULT FAILURE' and not ticks(lines, 'OpenGripper')


def test_bt_a_free_motion_failure_stays_a_verdict(harness, tmp_path):
    """Before any grasp nothing is held: a failed move to bin_view is recovered by the retry, as in 8.7.0."""
    lines = run_tree(harness, tmp_path, build(held_transit), '\n'.join([
        'MoveWaypoint:bin_view -> FAILURE',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert ['MoveWaypoint:bin_view', 'free'] in holds(lines)
    assert not any('while holding' in line for line in lines)
    assert lines[-1] == 'RESULT SUCCESS'


def test_bt_a_release_path_failure_while_held_is_a_fault(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(), f'MoveWaypoint:{PRE_RELEASE} -> FAILURE')
    assert lines[-1] == 'RESULT FAILURE' and 'LOG OUTCOME: TERMINAL_FAULT' in bare(lines)
    assert ['MoveWaypoint:' + PRE_RELEASE, 'held'] in holds(lines)
    assert not ticks(lines, 'OpenGripper') and len(ticks(lines, 'DetectCandidates:capsules')) == 1


def test_bt_a_held_part_never_falls_through_to_another_alternative(harness, tmp_path):
    lines = run_tree(harness, tmp_path, build(held_fallback), '\n'.join([
        'MoveWaypoint:home -> SUCCESS', 'MoveWaypoint:home -> FAILURE']))
    assert lines[-1] == 'RESULT FAILURE' and 'LOG MOTION_FAULT: SCRIPTED during plan while holding' in lines
    assert not ticks(lines, 'MoveWaypoint:place')                    # the second alternative never runs
    assert len(ticks(lines, 'DetectCandidates:capsules')) == 1


def test_bt_recovery_a_failed_hold_verification_frees_the_retreat_and_the_next_cycle(harness, tmp_path):
    """No false held: a hold verification that fails clears the state, so the retreat and the next candidate move
    FREE (a failure there stays the 8.7.0 verdict / RETREAT_FAILED rule); after a release the arm moves free."""
    lines = run_tree(harness, tmp_path, build(held_transit), '\n'.join([
        'VerifyGrasp:held -> FAILURE reason=NO_OBJECT',
        'DetectCandidates:capsules -> SUCCESS x2',
        'DetectCandidates:capsules -> FAILURE reason=NO_CANDIDATES x6', EMPTY]))
    assert lines[-1] == 'RESULT SUCCESS'
    h = holds(lines)
    assert ['MoveWaypoint:empty_the_bin__s2_1_1_3_retreat', 'free'] in h
    assert ['MoveWaypoint:empty_the_bin__s2_1_1_3_retract', 'held'] in h          # the next candidate's retract
    i = max(k for k, line in enumerate(lines) if line.startswith('TICK OpenGripper'))
    later = [x for x in holds(lines[i:])]
    assert later and all(x[1] == 'free' for x in later)                           # after the last release
