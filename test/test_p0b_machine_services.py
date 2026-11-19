"""P0.b (P0B_DESIGN §3.1 / §7, TSA 8.10.0): MACHINE SERVICES in the robot application -- the T cases of the frozen P0.b
matrix (overlay docs/engineering/P0_CHECKPOINT.md, "P0.b frozen acceptance matrix"):

  X1     a request / wait on an undeclared service          refused, naming the service and the declared ones
  X2     a wait without / with an invalid timeout_ms or for  refused; no default exists
  X3     a request / wait without an id                      refused
  X4     the project schema and the Composer                 no topic, seq, PLC address, OPC UA node or I/O field
  X5     the Composer (offscreen)                            add a machine service; add request + wait; edit the
                                                             timeout: editable, validated with readable messages,
                                                             saved; the dwell labelled "Dwell"
  X6     a program without machine services                  the files of TSA 8.9.0, byte for byte (the golden gates)
  X7     a program with them                                 the floors: TMR >= 2.11.0 (trainit_interfaces through it)
  X8     not dominated; two requests; a composite            refused, readable
  X9     an exported task's wait timeouts >= its timeout_s   a warning (the job timeout would fire first)
  PB-P7  a wait for done with no earlier request (T half)    TSA refuses, readable
  BW1    the public application (T half)                     0 external and 0 dwell in the WHOLE file; every machine
                                                             interaction a request + a wait; the blind-wait report
                                                             says so

and what they rest on: the TSA <-> TMR contract (the exact XML, P0B_IMPLEMENTATION_PLAN.md), the YAML key ``for``
round trip (pydantic 1.8.2 and 2.x: run this file under both), R-P0B-07 and the warnings W-P0B-02 / 03, the
blind-wait report, the README, Step 8's cell lines and the run panel's machine lines. PUB1 is test_p0b_pub1.py.

The fixture test/data/p0b/machine.yaml (on the P0.a test arm: no ur_description) is generated in temporary
directories only, and mutated copies of it are refused; the public sample examples/ur5e_suction/machine_tasks.yaml
is read as text and as a model (its generation needs ur_description, as the sample's other tests).
"""

import copy
import os
import re
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (FAULT_NODES, ProgramRefusals, plan_program,
                                                               program_overview, program_tree_xml)
from trainit_setup_assistant.composer.diagnostics import parse, suggestion
from trainit_setup_assistant.composer.machine_waits import MachineWaits, machine_waits, machine_waits_line
from trainit_setup_assistant.model import CanonicalProject, MachineService, WaitStep, project_yaml
from trainit_setup_assistant.model.program import (EVENT_FIELDS, PROGRAM_KEYS, EventKind, TaskProgram,
                                                   describe_load_error)

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
DATA = HERE / 'data' / 'p0b'
P0A = HERE / 'data' / 'p0a'
SAMPLE = PKG / 'examples' / 'ur5e_suction'
APP = 'p0b_app'
# the protocol words no machine field, hint target or Composer text of P0.b may carry (X4, owner decision 3)
PLUMBING_KEYS = ('topic', 'request_topic', 'status_topic', 'seq', 'sequence', 'session', 'heartbeat', 'address',
                 'plc_address', 'symbol', 'tag', 'node_id', 'opcua', 'io', 'signal')
PLUMBING_WORDS = ('topic', 'seq', 'session', 'heartbeat', 'address', 'opc', 'node id', 'i/o', 'plc tag', 'symbol')


def _raw() -> dict:
    return yaml.safe_load((DATA / 'machine.yaml').read_text())


def _app(raw: dict) -> dict:
    return raw['application']


def _task(raw: dict, name: str) -> dict:
    return next(t for t in _app(raw)['tasks'] if t['name'] == name)


def _project(raw: dict) -> CanonicalProject:
    old = os.getcwd()
    os.chdir(DATA)                                       # the urdf path is relative to it
    try:
        return CanonicalProject.parse_obj(raw)
    finally:
        os.chdir(old)


def _refusal(raw: dict) -> str:
    """The MODEL's refusal (the load), located in the file's terms."""
    with pytest.raises(ValueError) as exc:
        _project(raw)
    return describe_load_error(exc.value, raw, 'machine.yaml')


def _problems(raw: dict) -> list:
    """The PLANNER's refusals (references, flow): what Validate and Generate say."""
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(_project(raw))
    return list(exc.value.problems)


def _warnings(raw: dict) -> list:
    return plan_program(_project(raw)).warnings


def _generate(raw: dict, out: Path) -> Path:
    pytest.importorskip('xacro', reason='generating the fixture compiles its test arm with xacro (ROS)')
    from trainit_setup_assistant.generator import Orchestrator
    old = os.getcwd()
    os.chdir(DATA)
    try:
        Orchestrator().generate(CanonicalProject.parse_obj(raw), out)
    finally:
        os.chdir(old)
    return out / APP


def _without_machines(raw: dict) -> dict:
    """The fixture with every machine service, request and wait removed (each task keeps a step)."""
    raw = copy.deepcopy(raw)
    app = _app(raw)
    app.pop('machine_services', None)
    for t in app['tasks']:
        t['steps'] = [s for s in t['steps'] if 'wait' not in s and (s.get('process') or {}).get('kind') != 'request']
        if not t['steps']:
            t['steps'] = [{'id': 'go_home', 'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}]
    return raw


def _request(sid: str, service: str) -> dict:
    return {'id': sid, 'process': {'kind': 'request', 'channel': service}}


def _wait(sid: str, service: str, what: str = 'done', ms: int = 8000) -> dict:
    return {'id': sid, 'wait': {'channel': service, 'for': what, 'timeout_ms': ms}}


def _code(problems: list, code: str) -> list:
    return [p for p in problems if p.startswith(code + ':')]


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    return _generate(_raw(), tmp_path_factory.mktemp('p0b'))


# --- the fixture itself: valid, no warning, the contract ------------------------------------------------------------

def test_the_fixture_plans_with_no_refusal_and_no_warning():
    plan = plan_program(_project(_raw()))
    assert plan.warnings == []
    assert plan.process_controller == 'mock'             # a request is no process channel (decision D2)


def test_the_tsa_tmr_contract_is_the_exact_xml(generated):
    """P0B_IMPLEMENTATION_PLAN.md, "The TSA <-> TMR contract (pinned on both sides)": a request is a RequestMachine, a
    wait a WaitForMachine (channel, for, timeout_ms), each in its step's Sequence; under faults: stop both carry the
    gate's fault."""
    tree = (generated / 'bt_trees' / 'task_program.xml').read_text()
    turn = tree[tree.index('<BehaviorTree ID="task_turn">'):]
    turn = turn[:turn.index('</BehaviorTree>')]
    assert turn.splitlines()[1:] == [
        '    <Sequence>',
        '      <Sequence name="trainit_step:ask_turn:process">',
        '        <RequestMachine channel="turner" fault="{trainit__fault}"/>',
        '      </Sequence>',
        '      <Sequence name="trainit_step:turned:wait">',
        '        <WaitForMachine channel="turner" for="done" timeout_ms="8000" fault="{trainit__fault}"/>',
        '      </Sequence>',
        '    </Sequence>',
        '  ']
    assert '<WaitForMachine channel="turner" for="ready" timeout_ms="2000" fault="{trainit__fault}"/>' in tree
    assert '<RequestMachine channel="indexer" fault="{trainit__fault}"/>' in tree
    assert 'ProcessEvent' not in tree                     # decision D1: never a ProcessEvent of kind request
    assert {'RequestMachine', 'WaitForMachine'} <= set(FAULT_NODES)


def test_without_the_gate_the_machine_nodes_carry_no_fault():
    raw = _raw()
    del _app(raw)['faults']
    project = _project(raw)
    tree = program_tree_xml(project, plan_program(project))
    assert '<RequestMachine channel="turner"/>' in tree
    assert '<WaitForMachine channel="turner" for="done" timeout_ms="8000"/>' in tree
    assert 'fault=' not in tree


def test_a_gated_program_with_an_external_event_gets_no_fault_on_it():
    """FAULT_NODES gained the two machine tags only: a faults: stop program's ProcessEvent stays as in 8.9.0 (a TMR 2.10.0
    ProcessEvent has no fault port)."""
    raw = _without_machines(_raw())
    _task(raw, 'turn')['steps'].insert(0, {'id': 'kick', 'process': {'kind': 'external', 'channel': 'washer_start'}})
    project = _project(raw)
    tree = program_tree_xml(project, plan_program(project))
    assert '<ProcessEvent kind="external" channel="washer_start"/>' in tree


# --- X1 --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('where', ['request', 'wait'])
def test_x1_an_undeclared_service_is_refused_naming_it_and_the_declared_ones(where):
    raw = _raw()
    steps = _task(raw, 'turn')['steps']
    if where == 'request':
        steps[0]['process']['channel'] = 'turnr'
    else:
        steps[1]['wait']['channel'] = 'turnr'
    hit = _code(_problems(raw), 'R-P0B-01')
    assert len(hit) == 1, hit
    assert "task 'turn' step" in hit[0] and "no machine service is named 'turnr'" in hit[0]
    assert '(machine_services: turner, indexer)' in hit[0]
    # located at the step, with a did-you-mean (the Composer's diagnostics)
    d = parse(hit[0])
    assert d.code == 'R-P0B-01' and d.location.kind == 'step' and d.location.name == 'turn'
    project = _project(_raw())
    from trainit_setup_assistant.composer.ops import Composer
    assert suggestion(d, Composer(project).document(), project) == "Did you mean 'turner'?"


def test_x1_a_program_that_declares_no_service_says_so():
    raw = _raw()
    del _app(raw)['machine_services']
    hit = _code(_problems(raw), 'R-P0B-01')
    assert hit and all('(machine_services: none declared)' in h for h in hit)


# --- X2 --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('mutate, words', [
    (lambda w: w.pop('timeout_ms'), "needs 'timeout_ms'"),
    (lambda w: w.update(timeout_ms=None), "needs 'timeout_ms'"),
    (lambda w: w.update(timeout_ms=0), 'got 0'),
    (lambda w: w.update(timeout_ms=-5), 'got -5'),
    (lambda w: w.update(timeout_ms=600001), 'got 600001'),
    (lambda w: w.update(timeout_ms=1.5), 'got 1.5'),
    (lambda w: w.update(timeout_ms=8000.0), 'got 8000.0'),          # never truncated
    (lambda w: w.update(timeout_ms='8000'), "got '8000'"),
    (lambda w: w.update(timeout_ms=True), 'got True'),
    (lambda w: w.pop('for'), "needs 'for'"),
    (lambda w: w.update({'for': None}), "needs 'for'"),
    (lambda w: w.update({'for': 'finished'}), "wait.for 'finished'"),
    (lambda w: w.update({'for': 'DONE'}), "wait.for 'DONE'"),
    (lambda w: w.pop('channel'), "needs 'channel'"),
], ids=['no_timeout', 'null_timeout', 'zero', 'negative', 'too_long', 'float', 'whole_float', 'string', 'bool',
        'no_for', 'null_for', 'unknown_for', 'case_for', 'no_channel'])
def test_x2_a_wait_without_or_with_an_invalid_timeout_or_for_is_refused(mutate, words):
    raw = _raw()
    mutate(_task(raw, 'turn')['steps'][1]['wait'])
    text = _refusal(raw)
    assert 'R-P0B-02' in text and words in text, text
    assert "task 'turn' step 2 (wait)" in text            # located in the file's terms


def test_x2_no_default_exists_and_the_bounds_are_accepted():
    with pytest.raises(ValueError):
        WaitStep.parse_obj({'channel': 'turner', 'for': 'done'})
    with pytest.raises(ValueError):
        WaitStep.parse_obj({'channel': 'turner', 'timeout_ms': 10})
    for ms in (1, 600000):
        raw = _raw()
        _task(raw, 'turn')['steps'][1]['wait']['timeout_ms'] = ms
        assert _project(raw).application.program.tasks[0].steps[1].wait.timeout_ms == ms
    raw = _raw()
    _task(raw, 'turn')['steps'][1]['wait']['for'] = 'ready'
    assert _project(raw).application.program.tasks[0].steps[1].wait.for_.value == 'ready'


# --- X3 --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('task, index, what', [('turn', 0, 'a machine request'), ('turn', 1, 'a wait step'),
                                               ('ready_check', 0, 'a wait step')])
def test_x3_a_request_or_a_wait_without_an_id_is_refused(task, index, what):
    raw = _raw()
    del _task(raw, task)['steps'][index]['id']
    text = _refusal(raw)
    assert 'R-P0B-03' in text and f'{what} on machine service' in text and 'needs a step id' in text, text
    assert f"task '{task}' step {index + 1}" in text


def test_x3_every_other_step_still_needs_no_id():
    raw = _raw()
    for step in _task(raw, 'turn_part')['steps']:
        if 'move' in step or 'tool' in step:
            del step['id']
    _project(raw)                                         # loads


# --- X4 --------------------------------------------------------------------------------------------------------------

def _file_keys(cls) -> set:
    fields = getattr(cls, 'model_fields', None) or cls.__fields__
    return {getattr(f, 'alias', None) or name for name, f in fields.items()}


def test_x4_the_schema_carries_no_protocol_field():
    """Owner decision 3: a logical service id, request, wait for done | ready, a mandatory timeout -- never a topic, a
    sequence number, a PLC address, an OPC UA node or a physical I/O."""
    assert _file_keys(MachineService) == {'id', 'description'}
    assert _file_keys(WaitStep) == {'channel', 'for', 'timeout_ms'}
    assert EVENT_FIELDS[EventKind.REQUEST] == (('channel',), ())
    assert 'machine_services' in PROGRAM_KEYS
    for key in PLUMBING_KEYS:
        raw = _raw()
        _app(raw)['machine_services'][0][key] = 'x'
        assert 'R-M3-02' in _refusal(raw), key
        raw = _raw()
        _task(raw, 'turn')['steps'][1]['wait'][key] = 'x'
        assert 'R-M3-02' in _refusal(raw), key
        raw = _raw()
        _task(raw, 'turn')['steps'][0]['process'][key] = 'x'
        assert 'R-M3-02' in _refusal(raw), key
    raw = _raw()
    _app(raw)['machine_services'][0]['topic'] = '/turner/request'
    assert 'LOGICAL id' in _refusal(raw)                  # a readable "never" line, not just "unknown key"
    raw = _raw()
    _app(raw)['cell_io'] = {'plant': 'mock'}
    assert "CELL's file" in _refusal(raw)
    for bad in ('/turner', 'turner/status', 'Turner station', 'true', '2turn'):
        raw = _raw()
        _app(raw)['machine_services'][0]['id'] = bad
        assert 'not a machine service id' in _refusal(raw), bad


# --- X6 / X7 ---------------------------------------------------------------------------------------------------------

def test_x6_a_program_without_machine_services_carries_none_of_them(tmp_path):
    """The machine-free fixture: no machine node, no 2.11.0 floor, no Machine services section, the bt_params of 8.9.0
    (no machine comment), and its save writes no machine_services key (an empty list is never written)."""
    raw = _without_machines(_raw())
    app = _generate(raw, tmp_path / 'free')
    tree = (app / 'bt_trees' / 'task_program.xml').read_text()
    assert 'RequestMachine' not in tree and 'WaitForMachine' not in tree
    package = (app / 'package.xml').read_text()
    assert '2.11.0' not in package and 'RequestMachine' not in package
    assert '## Machine services' not in (app / 'README.md').read_text()
    assert 'machine' not in (app / 'config' / 'bt_params.yaml').read_text()
    assert 'machine_services' not in project_yaml(_project(raw))
    raw['application']['machine_services'] = []
    assert 'machine_services' in project_yaml(_project(raw))          # what is set is written: the Composer pops it


def test_x6_the_golden_gates_hold(tmp_path):
    """The machine-free byte copies TSA 8.9.0 pinned (P0.a's served fixture, also run by TMR) regenerate byte for byte
    (test_p0a_task_export pins their digests; test_d0_5_step_ids, test_m3_6_validation_c and the template goldens pin
    the rest)."""
    import test_p0a_task_export as P
    app = P._generate(P._raw(), tmp_path / 'p0a')
    pairs = {'task_program.xml': app / 'bt_trees' / 'task_program.xml', 'tasks.yaml': app / 'config' / 'tasks.yaml',
             'bt_params.yaml': app / 'config' / 'bt_params.yaml',
             'motion_profiles.yaml': app / 'config' / 'motion_profiles.yaml'}
    for name, path in pairs.items():
        assert path.read_bytes() == (P.COPIES / name).read_bytes(), name
        assert P._sha(P.COPIES / name) == P.PINNED[name], name


def test_x7_a_program_with_machine_services_needs_tmr_2_11_0(generated):
    package = (generated / 'package.xml').read_text()
    assert '<exec_depend version_gte="2.11.0">trainit_motion_runtime</exec_depend>' in package
    comment = next(line for line in package.splitlines() if 'trainit_motion_runtime >= 2.11.0 for:' in line)
    assert 'RequestMachine' in comment and 'WaitForMachine' in comment
    assert 'trainit_interfaces >= 0.5.0 through TMR' in comment      # reached THROUGH TMR: no own exec_depend
    assert 'trainit_interfaces' not in [line.strip() for line in package.splitlines() if 'exec_depend' in line
                                        and 'trainit_interfaces</exec_depend>' in line]


def test_x7_a_wait_only_program_needs_it_too(tmp_path):
    raw = _raw()
    app = _app(raw)
    app['tasks'] = [_task(raw, 'ready_check')]
    app['main_task'] = 'ready_check'
    app['machine_services'] = [{'id': 'turner'}]
    package = (_generate(raw, tmp_path / 'ready') / 'package.xml').read_text()
    assert '<exec_depend version_gte="2.11.0">trainit_motion_runtime</exec_depend>' in package
    comment = next(line for line in package.splitlines() if '>= 2.11.0 for:' in line)
    assert 'WaitForMachine' in comment and 'RequestMachine' not in comment


# --- X8 / PB-P7 ------------------------------------------------------------------------------------------------------

def test_pb_p7_a_wait_for_done_with_no_earlier_request_is_refused():
    """PB-P7 (T half): TSA refuses it, readably; a hand-written tree fails at run time (TMR, S half)."""
    raw = _raw()
    _task(raw, 'ready_check')['steps'][0]['wait']['for'] = 'done'
    hit = _code(_problems(raw), 'R-P0B-04')
    assert len(hit) == 1, hit
    assert hit[0].startswith("R-P0B-04: task 'ready_check' step 1 (wait for done on 'turner'): no request on 'turner'")
    assert "process: {kind: request, channel: turner}" in hit[0]          # what to change


@pytest.mark.parametrize('steps', [
    # the request is AFTER the wait
    [_wait('done_first', 'turner'), _request('ask', 'turner')],
    # a request inside a repeat body dominates nothing after the repeat
    [{'repeat': {'cycles': 2, 'body': [_request('ask', 'turner'), _wait('turned_in', 'turner')]}},
     _wait('turned', 'turner')],
    # a request on ANOTHER service
    [_request('ask', 'indexer'), _wait('turned', 'turner')],
], ids=['request_after', 'request_in_a_repeat_body', 'other_service'])
def test_x8_a_wait_for_done_not_dominated_by_a_request_is_refused(steps):
    raw = _raw()
    _task(raw, 'turn')['steps'] = steps
    hit = _code(_problems(raw), 'R-P0B-04')
    assert hit and all("task 'turn' step" in h for h in hit), hit


def test_x8_a_called_tasks_request_does_not_dominate_the_callers_wait():
    """Dominance is by an earlier step of the SAME sequence or an enclosing one: a request inside a called task is not
    one (the literal rule of P0B_DESIGN §3.1)."""
    raw = _raw()
    app = _app(raw)
    app['tasks'].append({'name': 'ask_only', 'steps': [_request('ask', 'turner')]})
    _task(raw, 'turn')['steps'] = [{'task': 'ask_only'}, _wait('turned', 'turner')]
    assert _code(_problems(raw), 'R-P0B-04')


@pytest.mark.parametrize('steps, expect', [
    ([_request('a1', 'turner'), _request('a2', 'turner'), _wait('w', 'turner')], 'step 2'),
    # a READY wait between them does not count: only a wait for done
    ([_request('a1', 'turner'), _wait('r', 'turner', 'ready'), _request('a2', 'turner'), _wait('w', 'turner')],
     'step 3'),
    # a repeat whose body requests and never waits for done: its NEXT cycle requests again
    ([{'repeat': {'cycles': 2, 'body': [_request('a1', 'turner'), _wait('r', 'turner', 'ready')]}}],
     'next cycle of the repeat at step 1'),
    # a request still owed when the body of a one-cycle repeat requests
    ([_request('a1', 'turner'), {'repeat': {'cycles': 1, 'body': [_request('a2', 'turner'),
                                                                  _wait('w', 'turner')]}}], 'step 2.1'),
], ids=['two_in_a_row', 'ready_between', 'repeat_next_cycle', 'into_a_repeat'])
def test_x8_two_requests_without_a_wait_for_done_between_are_refused(steps, expect):
    raw = _raw()
    _task(raw, 'turn')['steps'] = steps
    hit = _code(_problems(raw), 'R-P0B-05')
    assert hit and any(expect in h for h in hit), hit
    assert 'STATION_BUSY 313' in hit[0] and 'wait: {channel: turner, for: done' in hit[0]


def test_x8_a_call_that_requests_while_a_request_is_owed_is_refused():
    raw = _raw()
    _app(raw)['tasks'].append({'name': 'turn_again', 'steps': [_request('ask', 'turner'), _wait('w', 'turner')]})
    _task(raw, 'turn')['steps'] = [_request('ask', 'turner'), {'task': 'turn_again'}, _wait('turned', 'turner')]
    hit = _code(_problems(raw), 'R-P0B-05')
    assert hit and "(task 'turn_again'): it requests 'turner'" in hit[0], hit


def test_x8_a_request_a_called_task_leaves_owed_is_refused_at_the_next_request():
    raw = _raw()
    _app(raw)['tasks'].append({'name': 'ask_only', 'steps': [_request('ask', 'turner')]})
    _task(raw, 'turn')['steps'] = [{'task': 'ask_only'}, _request('ask', 'turner'), _wait('turned', 'turner')]
    assert _code(_problems(raw), 'R-P0B-05')


@pytest.mark.parametrize('steps, where', [
    ([{'retry': {'attempts': 2, 'body': [_request('ask', 'turner'), _wait('turned', 'turner')]}}],
     'the retry at step 1'),
    ([{'fallback': {'bodies': [[_request('ask', 'turner'), _wait('turned', 'turner')],
                               [{'id': 'go_home', 'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}]]}}],
     'the fallback at step 1 (alternative 1)'),
    ([_request('ask', 'turner'),
      {'fallback': {'bodies': [[{'id': 'go_home', 'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}],
                               [_wait('turned', 'turner')]]}}],
     'the fallback at step 2 (alternative 2)'),
    ([{'repeat': {'cycles': 2, 'body': [{'retry': {'attempts': 2, 'body': [_wait('ready', 'turner', 'ready')]}}]}}],
     'the retry at step 1.1'),
], ids=['retry_body', 'fallback_alternative', 'fallback_second_alternative', 'retry_in_a_repeat'])
def test_x8_a_request_or_a_wait_inside_a_composite_other_than_a_sequence_or_a_repeat_body_is_refused(steps, where):
    raw = _raw()
    _task(raw, 'turn')['steps'] = steps
    hit = _code(_problems(raw), 'R-P0B-06')
    assert hit and any(f'inside {where}' in h for h in hit), hit


def test_x8_following_task_calls():
    """A request / wait in a task CALLED from a retry body or a fallback alternative is refused at the call."""
    raw = _raw()
    _task(raw, 'turn')['steps'] = [{'retry': {'attempts': 3, 'body': [{'task': 'ready_check'}]}}]
    hit = _code(_problems(raw), 'R-P0B-06')
    assert hit and "(task 'ready_check'): inside the retry at step 1, it runs task 'ready_check', which waits for " \
                   "'turner' (step 1)" in hit[0], hit


def test_x8_a_repeat_body_holds_a_request_and_its_wait():
    raw = _raw()
    _task(raw, 'turn')['export']['timeout_s'] = 60          # 3 x (8 s + 8 s) of waits: below it
    _task(raw, 'turn')['steps'] = [{'repeat': {'cycles': 3, 'body': [_request('ask', 'turner'),
                                                                     _wait('turned', 'turner'),
                                                                     _wait('ready', 'turner', 'ready')]}}]
    assert plan_program(_project(raw)).warnings == []      # no refusal, no warning
    # and a request, a wait for done, a request again, its wait
    raw = _raw()
    _task(raw, 'turn')['steps'] = [_request('a1', 'turner'), _wait('w1', 'turner'), _request('a2', 'turner'),
                                   _wait('w2', 'turner')]
    plan_program(_project(raw))


# --- X9 --------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('timeout_s, steps, total', [
    (15, [_request('a', 'turner'), _wait('w1', 'turner'), _request('b', 'indexer'), _wait('w2', 'indexer')], 16000),
    (16, [_request('a', 'turner'), _wait('w1', 'turner'), _request('b', 'indexer'), _wait('w2', 'indexer')], 16000),
    (15, [{'repeat': {'cycles': 2, 'body': [_request('a', 'turner'), _wait('w1', 'turner')]}}], 16000),
    (10, [{'task': 'ready_check'}, _request('a', 'turner'), _wait('w1', 'turner')], 10000),
], ids=['sum_above', 'sum_equal', 'repeat_multiplies', 'called_task_included'])
def test_x9_an_exported_tasks_waits_at_or_above_its_timeout_are_warned(timeout_s, steps, total):
    raw = _raw()
    turn = _task(raw, 'turn')
    turn['export']['timeout_s'] = timeout_s
    turn['steps'] = steps
    hit = [w for w in _warnings(raw) if w.startswith('W-P0B-01')]
    assert len(hit) == 1, _warnings(raw)
    assert hit[0].startswith(f"W-P0B-01: task 'turn' (PLC task #2): its machine waits add up to {total} ms")
    assert 'TASK_TIMEOUT (302)' in hit[0] and 'STATION_TIMEOUT (305)' in hit[0]


def test_x9_waits_below_the_timeout_are_not_warned():
    raw = _raw()
    _task(raw, 'turn')['export']['timeout_s'] = 8.001
    assert not [w for w in _warnings(raw) if w.startswith('W-P0B-01')]


# --- the other rules -------------------------------------------------------------------------------------------------

def test_r_p0b_07_a_service_declared_twice_is_refused():
    raw = _raw()
    _app(raw)['machine_services'].append({'id': 'turner'})
    text = _refusal(raw)
    assert "R-P0B-07: machine service 'turner' is declared twice" in text
    assert parse(text.split(': ', 2)[-1]).location.kind == 'machine_service'


def test_a_request_is_never_a_path_event():
    raw = _raw()
    _app(raw)['waypoints'] += [{'name': 'p1', 'position': [0.3, 0.0, 0.3], 'orientation': [1, 0, 0, 0]},
                               {'name': 'p2', 'position': [0.3, 0.1, 0.3], 'orientation': [1, 0, 0, 0]}]
    _app(raw)['paths'] = [{'name': 'sweep', 'states': ['p1', 'p2'], 'segments': [{'type': 'lin'}],
                           'events': [{'at': {'state': 'p2'}, 'kind': 'request', 'channel': 'turner'}]}]
    text = _refusal(raw)
    assert 'a machine request is a task step' in text and "path 'sweep'" in text


def test_w_p0b_02_a_declared_service_never_used_is_warned():
    raw = _raw()
    _app(raw)['machine_services'].append({'id': 'labeller', 'description': 'label the part'})
    hit = [w for w in _warnings(raw) if w.startswith('W-P0B-02')]
    assert hit == ["W-P0B-02: machine service 'labeller' is declared and never used: no request or wait step names "
                   'it -- remove it from machine_services, or use it']
    assert parse(hit[0], 'warning').location.kind == 'machine_service'


def test_w_p0b_03_a_request_nothing_waits_for_is_warned():
    raw = _raw()
    _task(raw, 'turn')['steps'] = [_request('ask', 'turner'),
                                   {'id': 'go_home', 'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}]
    hit = [w for w in _warnings(raw) if w.startswith('W-P0B-03')]
    assert len(hit) == 1 and hit[0].startswith("W-P0B-03: task 'turn' step 1 (request on 'turner')"), hit
    # a READY wait after it counts as waiting for it (it requires the request's completion); so does a called task's
    raw = _raw()
    _task(raw, 'turn')['steps'] = [_request('ask', 'turner'), {'task': 'ready_check'}]
    assert not [w for w in _warnings(raw) if w.startswith('W-P0B-03')]


def test_a_machine_free_program_plans_as_before():
    """No machine rule, no machine warning: the P0.a fixture (8.9.0's own) keeps exactly its one warning of 8.9.0."""
    import test_p0a_task_export as P
    assert plan_program(P._project(P._raw())).warnings == [
        "task 'inspect' never runs: neither main_task 'place_part' nor a task it runs calls it"]


# --- the YAML key `for` (a Python keyword) round-trips -- pydantic 1.8.2 AND 2.x -----------------------------------

def test_the_key_for_is_read_and_written_as_for():
    project = _project(_raw())
    wait = project.application.program.tasks[0].steps[1].wait
    assert wait.for_.value == 'done'
    text = project_yaml(project)
    assert 'wait: {channel: turner, for: done, timeout_ms: 8000}' in text
    assert 'for_' not in text
    again = CanonicalProject.parse_obj(yaml.safe_load(text))
    assert again == project and project_yaml(again) == text
    assert "machine_services:\n  - {id: turner, description: turn the part over}" in text


def test_a_file_that_writes_for_underscore_is_refused():
    raw = _raw()
    w = _task(raw, 'turn')['steps'][1]['wait']
    w['for_'] = w.pop('for')
    text = _refusal(raw)
    assert "unknown key 'for_'" in text


def test_a_wait_can_be_edited_by_assignment():
    """validate_assignment: pydantic 1.x validates {fields} with the field NAME (for_), 2.x the value."""
    project = _project(_raw())
    wait = project.application.program.tasks[0].steps[1].wait
    wait.for_ = 'ready'
    assert wait.for_.value == 'ready'
    with pytest.raises(ValueError):
        wait.timeout_ms = 0


# --- BW1 (T half) and the blind-wait report --------------------------------------------------------------------------

def _sample_raw() -> dict:
    return yaml.safe_load((SAMPLE / 'machine_tasks.yaml').read_text())


def _walk_values(node):
    if isinstance(node, dict):
        for k, v in node.items():
            yield k, v
            yield from _walk_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_values(v)


def test_bw1_the_public_application_has_no_blind_wait():
    """BW1 (T half): external = 0 and dwell = 0 in the WHOLE public application -- its text (no `dwell`, no `external`
    anywhere, comments included), its YAML (no dwell step, no external / dwell event) and the blind-wait report; every
    machine interaction a request + a wait (every request waited for, every wait for done dominated: the plan refuses
    nothing and warns nothing)."""
    text = (SAMPLE / 'machine_tasks.yaml').read_text()
    assert not re.search(r'dwell|external', text, re.IGNORECASE)
    raw = _sample_raw()
    pairs = list(_walk_values(raw['application']))
    assert not [k for k, _v in pairs if k == 'dwell']
    assert not [v for k, v in pairs if k == 'kind' and v in ('external', 'dwell')]
    program = TaskProgram.parse_obj({k: v for k, v in raw['application'].items() if k != 'type'})
    report = machine_waits(program)
    assert report == MachineWaits(requests=5, waits=7, external=0, dwell=0)
    assert report.text() == 'machine waits: 12 semantic (request + wait); 0 external; 0 dwell'
    assert report.blind == 0
    assert program.service_ids() == ['turner', 'indexer'] and program.used_services() == ['turner', 'indexer']


def test_bw1_the_public_application_plans_with_no_refusal_and_no_warning(monkeypatch):
    monkeypatch.chdir(SAMPLE)                             # the README's rule: from its directory
    from trainit_setup_assistant.model import load_project
    try:
        project = load_project('machine_tasks.yaml')
    except Exception as exc:                              # noqa: BLE001 -- no ur_description: the sample cannot load
        if 'ur_description' in str(exc):
            pytest.skip('the public sample needs ur_description')
        raise
    plan = plan_program(project)
    assert plan.warnings == []
    turn_part = project.application.program.tasks[[t.name for t in project.application.program.tasks].index(
        'turn_part')]
    assert turn_part.export.number == 5
    assert [(s.kind(), s.machine_service(), s.wait.for_.value if s.wait else None, s.wait.timeout_ms if s.wait
             else None) for s in turn_part.steps] == [
        ('move', None, None, None), ('tool', None, None, None), ('move', None, None, None), ('tool', None, None, None),
        ('process', 'turner', None, None), ('move', None, None, None), ('wait', 'turner', 'done', 8000),
        ('process', 'indexer', None, None), ('wait', 'indexer', 'done', 8000), ('wait', 'turner', 'ready', 2000)]
    # the small exported tasks the M scenarios start: wait ready only; request + done (both services); request + ready
    shapes = {t.name: [(s.kind(), s.wait.for_.value if s.wait else None) for s in t.steps]
              for t in project.application.program.exported()}
    assert ([('wait', 'ready')] in shapes.values() and [('process', None), ('wait', 'ready')] in shapes.values())
    assert [t for t, sh in shapes.items() if sh == [('process', None), ('wait', 'done')]] == ['turn', 'index']
    assert project.application.program.main_task == 'turn'            # PB-P18: the run's FIRST step is a request
    assert line_of(project) == 'machine waits: 12 semantic (request + wait); 0 external; 0 dwell'


def line_of(project) -> str:
    return machine_waits_line(project)


def test_the_report_counts_external_and_dwell_steps_and_path_events():
    """The calibration: M3.7's machine-tending decomposition (one external, one dwell) and path events."""
    data = yaml.safe_load((HERE / 'data' / 'm3_7' / 'decompositions' / 'a_machine_tending.yaml').read_text())
    program = TaskProgram.parse_obj({k: v for k, v in data['application'].items() if k != 'type'})
    assert machine_waits(program).text() == 'machine waits: 0 semantic (request + wait); 1 external; 1 dwell'
    raw = _raw()
    app = _app(raw)
    app['waypoints'] += [{'name': 'p1', 'position': [0.3, 0.0, 0.3], 'orientation': [1, 0, 0, 0]},
                         {'name': 'p2', 'position': [0.3, 0.1, 0.3], 'orientation': [1, 0, 0, 0]}]
    app['paths'] = [{'name': 'sweep', 'states': ['p1', 'p2'], 'segments': [{'type': 'lin'}],
                     'events': [{'at': {'state': 'p2'}, 'kind': 'external', 'channel': 'washer_start'},
                                {'at': {'state': 'p2'}, 'kind': 'dwell', 'ms': 100}]}]
    _task(raw, 'turn')['steps'].append({'dwell': {'ms': 50}})
    report = machine_waits(_project(raw).application.program)
    assert (report.requests, report.waits, report.external, report.dwell) == (3, 5, 1, 2)


def test_the_report_is_never_a_plan_warning():
    raw = _raw()
    _task(raw, 'turn')['steps'].append({'dwell': {'ms': 50}})
    assert not [w for w in _warnings(raw) if 'machine waits' in w]


def test_the_report_has_no_line_for_a_template_application():
    class NotAProgram:
        class application:                                 # noqa: N801
            @staticmethod
            def is_task_program():
                return False
    assert machine_waits_line(NotAProgram()) == ''


# --- the generated README, the overview, the Composer's words --------------------------------------------------------

def test_the_readme_says_what_the_machine_services_are_and_who_serves_them(generated):
    readme = (generated / 'README.md').read_text()
    section = readme[readme.index('## Machine services'):]
    section = section[:section.index('\n## ', 3)]
    assert '| `turner` | turn the part over | `turn` step 1; `turn_part` step 5 |' in section
    assert 'trainit_cell_io' in section and 'started apart' in section
    for state in ('MACHINE `<service>` FAULT', 'TIMEOUT', 'UNAVAILABLE', 'WAITING FOR MACHINE'):
        assert state in section
    assert '/request' not in section and '/status' not in section and 'heartbeat' not in section
    assert '| 2 | wait | `turner` | wait for machine service `turner`: its DONE' in readme
    assert 'Process channels' not in readme               # a request is not a process channel


def test_a_service_description_with_a_bar_stays_one_cell_of_the_readmes_table():
    """Python review MINOR 7: a description is one line of free text and may hold '|'; the README's Machine services
    table escapes it, so its row keeps four cells (a description without '|' is written as before)."""
    from trainit_setup_assistant.generator.program_files import machine_services_section
    raw = _raw()
    _app(raw)['machine_services'][0]['description'] = 'turn | flip the part over'
    project = _project(raw)
    lines = machine_services_section(project.application.program, plan_program(project))
    rows = {line.split('`')[1]: line for line in lines if line.startswith('| `')}
    assert rows['turner'].startswith('| `turner` | turn \\| flip the part over | `turn` step 1; '), rows['turner']
    assert rows['indexer'].startswith('| `indexer` | index the line by one stroke | '), rows['indexer']
    for service, row in rows.items():
        cells = re.split(r'(?<!\\)\|', row)[1:-1]         # split on the unescaped bars only
        assert len(cells) == 4, (service, cells)


def test_the_overview_and_the_words_say_wait_for_machine_and_dwell():
    program = _project(_raw()).application.program
    text = program_overview(program)
    assert '  1  request machine service turner' in text
    assert '  2  wait for machine service turner: done, at most 8000 ms' in text
    assert 'Machine services (2): turner -- turn the part over, indexer -- index the line by one stroke' in text
    from trainit_setup_assistant.composer.describe import KIND_TITLES, step_text
    assert KIND_TITLES['dwell'] == 'Dwell' and KIND_TITLES['wait'] == 'Wait for machine'
    assert step_text({'dwell': {'ms': 500}}) == 'Dwell 500 ms'
    assert step_text({'id': 'r', 'wait': {'channel': 'turner', 'for': 'ready', 'timeout_ms': 2000}}) == \
        'Wait for machine turner: ready · 2000 ms'


def test_the_overview_rows_name_the_services_and_the_report():
    from trainit_setup_assistant.composer.integration import scene_status
    from trainit_setup_assistant.composer.overview import project_overview
    project = _project(_raw())
    rows = {r.block: r for r in project_overview(project, scene_status(project))}
    assert rows['MACHINE SERVICES'].text == 'turner, indexer'
    assert rows['MACHINE WAITS'].text == 'machine waits: 8 semantic (request + wait); 0 external; 0 dwell'
    assert rows['PROCESS CHANNELS'].text == 'none'          # a request is not a process channel (decision D2)


def _requesting_composition(monkeypatch, trigger_channel: str = 'turner'):
    """The D0.8 data-collection cell (the M3.7 composition, a camera, triggers) whose cell_cycle also requests the
    machine service `turner` and waits for it, with an EVENT trigger on `trigger_channel` (edge: any)."""
    import test_d0_8_data_collection as D
    import test_m3_7_composition as C
    C.needs_ur_description()
    monkeypatch.chdir(C.DATA)
    data = copy.deepcopy(D.DATA)
    data['triggers'].append({'id': 'on_turn_request', 'kind': 'event', 'process_channel': trigger_channel,
                             'edge': 'any'})

    def mutate(doc):
        D.with_data(data=data)(doc)
        app = doc['application']
        app['machine_services'] = [{'id': 'turner', 'description': 'turn the part over'}]
        cycle = next(t for t in app['tasks'] if t['name'] == 'cell_cycle')
        cycle['steps'] += [_request('ask_turn', 'turner'), _wait('turned', 'turner')]
    return C.build(mutate)


def test_the_overview_lists_process_channels_only_and_an_event_trigger_may_name_a_requested_service(monkeypatch):
    """Python review MINOR 9: the Overview's PROCESS CHANNELS row lists the program's process channels only -- a
    machine request names a SERVICE (decision D2), shown by the MACHINE SERVICES row -- while a data-collection
    EVENT trigger may still name a requested service: a request is a PROCESS_EVENT on its id, and trainit_data
    >= 0.10.2 fires 'edge: any' on it (decision D6)."""
    from trainit_setup_assistant.composer.integration import scene_status
    from trainit_setup_assistant.composer.overview import event_channels, process_channels, project_overview
    from trainit_setup_assistant.generator.data_collection import data_collection_problems
    project = _requesting_composition(monkeypatch)
    assert process_channels(project) == ['conveyor']
    assert event_channels(project) == ['conveyor', 'turner']
    rows = {r.block: r for r in project_overview(project, scene_status(project))}
    assert rows['PROCESS CHANNELS'].text == 'conveyor' and rows['MACHINE SERVICES'].text == 'turner'
    assert data_collection_problems(project) == []
    typo = data_collection_problems(_requesting_composition(monkeypatch, 'turnr'))
    assert typo == ["R-D0-06: data collection trigger 'on_turn_request': no process channel 'turnr' is driven by the "
                    'program (channels: conveyor, turner)'], typo


# --- Step 8: the cell's command, the Isaac-plant note; nothing without machine services -----------------------------

def test_step_8_prints_the_cells_command_and_the_isaac_plant_note():
    from trainit_setup_assistant.gui.wizard import build_and_run_lines, machine_service_lines
    project = _project(_raw())
    lines = machine_service_lines(project)
    assert lines[0].startswith('  # machine services (turner, indexer): the CELL serves them')
    assert '  #                ros2 launch trainit_cell_io cell_io.launch.py config:=<your cell_io.yaml>' in lines
    assert any('examples/neutral_station/cell_io_mock.yaml' in line for line in lines)
    assert any('use_sim_time:=true' in line and "stage's /clock" in line for line in lines)
    built = build_and_run_lines(project, '/ws/src/p0b', '/ws')
    assert built[-len(lines):] == lines
    assert any('trainit_task_server.launch.py' in line for line in built)      # beside the fake PLC's
    assert machine_service_lines(_project(_without_machines(_raw()))) == []


def test_the_cells_command_is_one_spelling():
    from trainit_setup_assistant.composer import launch_plan
    assert launch_plan.cell_io_line() == 'ros2 launch trainit_cell_io cell_io.launch.py config:=<your cell_io.yaml>'
    command = launch_plan.cell_io_command('/ws')
    assert command == ('source /opt/ros/humble/setup.bash && source /ws/install/setup.bash && '
                       'ros2 launch trainit_cell_io cell_io.launch.py config:=<your cell_io.yaml>')


def test_tsa_never_imports_or_declares_the_cell():
    """ADR-0020: the cell is a cell-level process TSA prints the command of; TSA neither imports trainit_cell_io nor
    declares it (it is not published: rosdep could not resolve it)."""
    assert 'trainit_cell_io' not in (PKG / 'package.xml').read_text()
    for path in (PKG / 'trainit_setup_assistant').rglob('*.py'):
        text = path.read_text()
        assert not re.search(r'^\s*(import|from)\s+trainit_cell_io', text, re.M), path


# --- the GUI (offscreen): X4's Composer half, X5, the run panel ------------------------------------------------------

@pytest.fixture(scope='module')
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    pytest.importorskip('python_qt_binding', reason='the Composer GUI needs python_qt_binding (ROS)')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _composer(raw: dict, monkeypatch, where: Path = DATA):
    from trainit_setup_assistant.composer import Composer
    monkeypatch.chdir(where)                              # the project's relative urdf path
    return Composer(CanonicalProject.parse_obj(raw))


def _texts(widget) -> str:
    from python_qt_binding.QtWidgets import QAbstractButton, QComboBox, QGroupBox, QLabel, QLineEdit
    out = [widget.toolTip()]
    for w in widget.findChildren((QLabel, QAbstractButton, QLineEdit, QComboBox, QGroupBox)):
        out += [w.toolTip()]
        if isinstance(w, (QLabel, QAbstractButton)):
            out.append(w.text())
        elif isinstance(w, QLineEdit):
            out.append(w.placeholderText())
        elif isinstance(w, QComboBox):
            out += [w.itemText(i) for i in range(w.count())]
        elif isinstance(w, QGroupBox):
            out.append(w.title())
    return ' '.join(out).lower()


def test_x5_the_composer_adds_a_service_a_request_and_a_wait_and_edits_the_timeout(qapp, monkeypatch):
    """X5: from the machine-free P0.a fixture -- declare a machine service, add a "Machine request" and a "Wait for
    machine" (ids minted), set the timeout; every refusal is the model's readable message; the result saves and
    reloads identically. The dwell reads "Dwell"."""
    import test_p0a_task_export as P
    from trainit_setup_assistant.gui.composer.machines_panel import MachinesPanel
    from trainit_setup_assistant.gui.composer.tasks_panel import ADD_TITLES, TasksPanel
    c = _composer(P._raw(), monkeypatch, P0A)
    m = MachinesPanel(lambda: c)
    m.refresh()
    assert m.table.rowCount() == 0
    # a bad id: refused with the model's words, nothing changed
    m.service_id.setText('2 turner')
    assert not m.add() and 'not a machine service id' in m.message.text()
    m.service_id.setText('turner')
    m.description.setText('turn the part over')
    assert m.add()
    assert c.document()['machine_services'] == [{'id': 'turner', 'description': 'turn the part over'}]
    assert not m.add() and 'R-P0B-07' in m.message.text()            # declared once
    # the Tasks section: the add bar's words, a request and a wait after step 2 of place_part
    p = TasksPanel(lambda: c)
    p.task = 'place_part'
    p.refresh()
    titles = [p.add_kind.itemText(i) for i in range(p.add_kind.count())]
    assert 'Machine request' in titles and 'Wait for machine' in titles and 'Dwell' in titles and 'Wait' not in titles
    assert ADD_TITLES['dwell'] == 'Dwell'
    p.select((3,))                                                  # settle: the dwell
    assert p.form_widgets['dwell'].layout().labelForField(p.dwell_ms).text() == 'Dwell'
    assert p.add_step('request') == (4,)
    assert p.add_step('wait') == (5,)
    steps = c.document()['tasks'][0]['steps']
    assert steps[3] == {'id': 'ask_turner', 'process': {'kind': 'request', 'channel': 'turner'}}
    assert steps[4] == {'id': 'wait_turner', 'wait': {'channel': 'turner', 'for': 'done', 'timeout_ms': 10000}}
    assert p.add_step('wait') == (6,)                                # a second one: its own id
    assert c.document()['tasks'][0]['steps'][5]['id'] == 'wait_turner_2'
    p.select((6,))
    assert p.delete_selected()
    # edit the timeout in the "Wait for machine" form
    p.select((5,))
    form = p.step_forms['wait']
    assert p.forms.currentWidget() is form
    assert form.service.currentText() == 'turner' and form.wait_for.currentData() == 'done'
    form.timeout.setValue(8000)
    assert p.apply()
    assert c.document()['tasks'][0]['steps'][4]['wait'] == {'channel': 'turner', 'for': 'done', 'timeout_ms': 8000}
    # an Apply without an edit changes nothing
    undo = len(c._undo)
    p.select((5,))
    assert p.apply() and len(c._undo) == undo
    # the request in the Process form: its kind shown, its service from the declared ones
    p.select((4,))
    process = p.step_forms['process']
    assert process.pkind.currentText() == 'request' and process.service.currentText() == 'turner'
    assert process.rows['service'][1].isVisibleTo(process) and not process.rows['channel'][1].isVisibleTo(process)
    undo = len(c._undo)
    assert p.apply() and len(c._undo) == undo                        # unchanged: the request stays a request
    # validated with readable messages: a wait without its id
    p.select((5,))
    assert 'required' in p.step_id.placeholderText()
    p.step_id.setText('')
    assert not p.apply() and 'R-P0B-03' in p.message.text()
    # a used service cannot be removed (its users named); an unused one can, and the key goes with the last one
    m.refresh()
    m.select('turner')
    assert not m.remove('turner') and "task 'place_part' step 4" in m.message.text()
    # saved and reloaded identically
    text = project_yaml(c.project)
    assert 'wait: {channel: turner, for: done, timeout_ms: 8000}' in text
    assert project_yaml(CanonicalProject.parse_obj(yaml.safe_load(text))) == text
    plan_program(c.project)                                          # the planner accepts it
    p.select((5,))
    assert p.delete_selected()
    p.select((4,))
    assert p.delete_selected()
    m.refresh()
    assert m.remove('turner')
    assert 'machine_services' not in c.document()                    # never an empty list: 8.9.0's file again
    assert project_yaml(c.project) == project_yaml(P._project(P._raw()))


def test_the_wait_form_never_crashes_on_a_hand_edit_and_never_rewrites_what_the_user_left(qapp):
    """Python review MINOR 8: the "Wait for machine" form reads a stored timeout_ms defensively (a hand-edited text
    never reaches int()) and writes `for` / timeout_ms only when the user changed them -- a value its boxes cannot show
    (a text, a bool, a float, out of range, missing: QSpinBox clamps) stays as written, for the model to name."""
    from trainit_setup_assistant.gui.composer.step_forms import NEW_WAIT_TIMEOUT_MS, WaitForm
    form = WaitForm()
    ctx = {'services': ['turner', 'indexer']}
    for stored in ('abc', '8000', 0, -5, 700000, 8000.5, True, [8000], None):
        body = {'channel': 'turner', 'for': 'done', 'timeout_ms': stored}
        if stored is None:
            del body['timeout_ms']
        form.load({'id': 'turned', 'wait': copy.deepcopy(body)}, ctx)          # never raises
        assert form.timeout.value() == NEW_WAIT_TIMEOUT_MS, stored
        assert form.spec() == {'wait': body}, stored                            # untouched: kept as written
        form.timeout.setValue(8000)
        assert form.spec() == {'wait': dict(body, timeout_ms=8000)}, stored     # set by the user: written
    # a value the model accepts is shown as it is; untouched it stays, an edit is written
    body = {'channel': 'indexer', 'for': 'ready', 'timeout_ms': 2000}
    form.load({'id': 'indexed', 'wait': copy.deepcopy(body)}, ctx)
    assert (form.timeout.value(), form.wait_for.currentData(), form.service.currentText()) == (2000, 'ready', 'indexer')
    assert form.spec() == {'wait': body}
    form.wait_for.setCurrentIndex(form.wait_for.findData('done'))
    form.timeout.setValue(3000)
    assert form.spec() == {'wait': {'channel': 'indexer', 'for': 'done', 'timeout_ms': 3000}}
    # `for` the same: a hand-edited one stays as written (the model refuses it in its own words), never 'done'
    for stored in ('Done', 'soon', 7):
        body = {'channel': 'turner', 'for': stored, 'timeout_ms': 8000}
        form.load({'id': 'turned', 'wait': copy.deepcopy(body)}, ctx)
        assert form.spec() == {'wait': body}, stored
        form.wait_for.setCurrentIndex(form.wait_for.findData('ready'))
        assert form.spec()['wait']['for'] == 'ready', stored


def test_x5_adding_a_request_without_a_declared_service_says_where_to_declare_it(qapp, monkeypatch):
    import test_p0a_task_export as P
    from trainit_setup_assistant.gui.composer.tasks_panel import TasksPanel
    c = _composer(P._raw(), monkeypatch, P0A)
    p = TasksPanel(lambda: c)
    p.task = 'home'
    p.refresh()
    assert p.add_step('wait') is None
    assert 'declare a machine service first (Machine services section)' in p.bar_message.text()


def test_x4_the_composers_machine_widgets_say_no_protocol_word(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer.machines_panel import MachinesPanel
    from trainit_setup_assistant.gui.composer.step_forms import ProcessForm, WaitForm
    c = _composer(_raw(), monkeypatch)
    m = MachinesPanel(lambda: c)
    m.refresh()
    words = _texts(m) + ' ' + _texts(WaitForm())
    form = ProcessForm()
    words += ' ' + form.service.toolTip().lower() + ' ' + form.rows['service'][0].text().lower()
    for word in PLUMBING_WORDS:
        assert word not in words, word


def test_x5_the_whole_shell_edits_a_machine_program_without_a_crash(qapp, tmp_path):
    """The Composer inside the wizard (the program text and the diagnostics follow every edit): a wait added, the
    overview and the verdict refreshed; the report shown; a diagnostic on a service goes to the Machine services
    section."""
    import shutil
    pytest.importorskip('xacro', reason='the wizard validates the test arm\'s description with xacro (ROS)')
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    work = tmp_path / 'p0b'
    shutil.copytree(DATA, work)
    shutil.copytree(P0A / 'urdf', tmp_path / 'p0a' / 'urdf')
    raw = _raw()
    _app(raw)['machine_services'].append({'id': 'labeller'})          # W-P0B-02: a diagnostic on a service
    (work / 'machine.yaml').write_text(yaml.safe_dump(raw, sort_keys=False))
    ctrl = AssistantController()
    ctrl.open_project(work / 'machine.yaml')
    w = SetupWizard(ctrl)
    w.blocks_page.initializePage()
    test_x5_the_whole_shell_edits_a_machine_program_without_a_crash.keep = w
    shell = w.blocks_page.program_panel
    assert 'Machine services' in shell.sections
    panel = shell.diagnostics
    panel.refresh()
    assert panel.machine_waits.text() == 'machine waits: 8 semantic (request + wait); 0 external; 0 dwell'
    texts = [panel.list.item(i).text() for i in range(panel.list.count())]
    i = next(k for k, t in enumerate(texts) if 'W-P0B-02' in t)       # a valid program's warning, located
    panel.list.setCurrentRow(i)
    panel.go_to_selected()
    assert shell.stack.currentWidget() is shell.machines and shell.machines.selected == 'labeller'
    shell.select('Tasks')
    shell.tasks.task = 'ready_check'
    shell.tasks.refresh()
    shell.tasks.select((1,))
    assert shell.tasks.add_step('wait') == (2,)                       # _program_edited: the overview has a wait
    assert 'wait for machine service turner: done, at most 10000 ms' in w.blocks_page.program_view.toPlainText()
    assert panel.machine_waits.text() == 'machine waits: 9 semantic (request + wait); 0 external; 0 dwell'
    texts = [panel.list.item(i).text() for i in range(panel.list.count())]
    assert any('R-P0B-04' in t and "task 'ready_check' step 2" in t for t in texts)   # its wait for done: no request
    w.blocks_page.validate_program()
    assert 'machine waits: 9 semantic' in w.blocks_page.program_result.text()
    assert 'R-P0B-04' in w.blocks_page.program_result.text()


def test_the_trigger_dialog_offers_what_the_validator_accepts_and_says_so(qapp, monkeypatch):
    """The data-collection trigger dialog lists the process channels AND the machine services the program requests
    (what the validator accepts for an EVENT trigger), and its note says why a service is there; a program without
    requests reads as before (test_d0_8_data_panel)."""
    from trainit_setup_assistant.gui.composer.data_panel import TriggerDialog
    dialog = TriggerDialog(_requesting_composition(monkeypatch))
    test_the_trigger_dialog_offers_what_the_validator_accepts_and_says_so.keep = dialog
    assert [dialog.channel.itemText(i) for i in range(dialog.channel.count())] == ['conveyor', 'turner']
    dialog.kind.setCurrentIndex(dialog.kind.findData('event'))
    assert dialog.note.text() == ('Only the process channels the program drives are listed, and the machine services '
                                  'it requests (a request fires "changes").'), dialog.note.text()


def test_the_run_panel_shows_machine_and_rti_lines_as_plain_text(qapp):
    """Decision D5: a MACHINE: line (the application or the server) and the server's RTI: lines reach the state line in
    plain words -- shown as written ('->' and quotes, never '-&gt;' or '&#x27;'); a failed run keeps the machine's
    reason under its verdict."""
    from python_qt_binding.QtCore import Qt
    from trainit_setup_assistant.composer import launch_plan
    from trainit_setup_assistant.composer.launch_plan import BT_FAILURE
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.run_panel import RunPanel
    panel = RunPanel(Launcher(), [('server', 'Serve the PLC tasks', lambda: None), ('run', 'Run', lambda: None)])
    panel.line('server', '[trainit_task_server-1] [INFO] [1.2] [trainit_task_server]: RTI: state READY -> RUNNING '
                         "(job 3, turn_part)")
    assert panel.state.textFormat() == Qt.PlainText
    assert panel.state.text() == 'Robot task server — state READY -> RUNNING (job 3, turn_part)'
    panel.line('server', "[trainit_task_server-1] [INFO] [1.3] [trainit_task_server]: RTI: job 3: waiting for "
                         "machine 'turner': BUSY, 2.1 s of 8.0 s")
    assert panel.state.text() == "Robot task server — job 3: waiting for machine 'turner': BUSY, 2.1 s of 8.0 s"
    panel.line('server', '[trainit_task_server-1] [INFO] [1.4] [trainit_task_server]: MACHINE: MACHINE turner DONE '
                         '(2.4 s)')
    assert panel.state.text() == 'Robot task server — MACHINE turner DONE (2.4 s)'
    # O1: never a protocol counter on the state line (the log keeps the whole line)
    for line, shown in (('RTI: START seq 7: ACCEPTED', 'START: ACCEPTED'),
                        ('RTI: RESET seq 41: ACCEPTED', 'RESET: ACCEPTED'),
                        ('RTI: requester adopted (baseline seq 3)', 'requester adopted'),
                        ('RTI: stopping job run e1981657-26 (MotionRuntime::stop, then the halt)',
                         'stopping job (MotionRuntime::stop, then the halt)'),
                        ('RTI: job 26 (turn_then_ready #4) ended: FAILED STATION_UNAVAILABLE',
                         'job 26 (turn_then_ready #4) ended: FAILED STATION_UNAVAILABLE')):
        panel.line('server', f'[trainit_task_server-1] [INFO] [1.5] [trainit_task_server]: {line}')
        assert panel.state.text() == f'Robot task server — {shown}'
    assert 'START seq 7: ACCEPTED' in panel.log.toPlainText()
    panel.started(launch_plan.Command('run', 'Run', 'true', True))
    panel.line('run', '[trainit_run_bt-1] [INFO] [2.0] [trainit_run_bt]: MACHINE: MACHINE turner TIMEOUT: still '
                      'BUSY after 8.0 s')
    assert panel.state.text() == 'Application — MACHINE turner TIMEOUT: still BUSY after 8.0 s'
    panel.line('run', f'[trainit_run_bt-1] [INFO] [2.1] [trainit_run_bt]: {BT_FAILURE}')
    shown = panel.state.text()
    assert BT_FAILURE in shown and 'MACHINE turner TIMEOUT: still BUSY after 8.0 s' in shown
    assert panel.state.textFormat() == Qt.RichText


def test_a_run_that_requests_machine_services_notes_the_cells_command_first(qapp):
    from trainit_setup_assistant.composer import launch_plan
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.run_panel import RunPanel
    panel = RunPanel(Launcher(), [('run', 'Run', lambda: None)])
    panel.notes['run'] = ['# the cell first:', launch_plan.cell_io_command('/ws')]
    panel.started(launch_plan.Command('run', 'Run', 'exec true', True))
    log = panel.log.toPlainText().splitlines()
    assert log[0] == '$ true' and log[1] == '# the cell first:' and 'cell_io.launch.py' in log[2]


def test_step_8_writes_the_cells_command_before_the_fake_plcs_and_before_a_run(qapp, tmp_path):
    """The run panel's copy-me lines (TSA never starts the cell): the server's, written once it serves -- the cell's
    command first, then the fake PLC's; a one-shot run's, right after its own command line. None without machine
    services."""
    from types import SimpleNamespace
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.wizard import SERVING, GeneratePage
    ctrl = SimpleNamespace(project=_project(_raw()))
    page = GeneratePage(ctrl, launcher=Launcher(), settings=SimpleNamespace(workspace_root='', isaac_launcher=''))
    test_step_8_writes_the_cells_command_before_the_fake_plcs_and_before_a_run.keep = page
    bundle = tmp_path / 'ws' / 'src' / 'p0b_bundle'
    (bundle / APP / 'launch').mkdir(parents=True)
    (bundle / APP / 'launch' / 'trainit_task_server.launch.py').write_text('')
    page.out_dir_generated = str(bundle)
    page._server_command()
    marker, lines = page.run_panel.after_ready['server']
    assert marker == SERVING
    assert lines[0].startswith('# machine services (turner, indexer): the CELL serves them')
    assert lines[1].endswith('ros2 launch trainit_cell_io cell_io.launch.py config:=<your cell_io.yaml>')
    assert 'trainit_fake_plc' in lines[-1] and 'cell_io' not in lines[-1]
    page._run_command()
    assert page.run_panel.notes['run'] == lines[:2]
    ctrl.project = _project(_without_machines(_raw()))
    page._server_command()
    assert not [line for line in page.run_panel.after_ready['server'][1] if 'cell_io' in line]
    page._run_command()
    assert page.run_panel.notes['run'] == []


def test_bw1_the_public_application_generates_the_readmes_way_and_says_it_has_no_blind_wait(monkeypatch, tmp_path,
                                                                                            capsys):
    """BW1 (T half, the generator's side): `trainit_generate machine_tasks.yaml` from the sample's directory (its
    README's command) generates with no warning, prints the blind-wait report as its third line, and the bundle runs
    TMR >= 2.11.0's machine nodes -- never a ProcessEvent or a Sleep."""
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.cli.generate import main
    monkeypatch.chdir(SAMPLE)
    out = tmp_path / 'ur5e_machine'
    assert main(['machine_tasks.yaml', '-o', str(out)]) == 0
    printed = capsys.readouterr()
    lines = printed.out.splitlines()
    assert lines[1] == '  packages: ur5e_machine_description, ur5e_machine_trainit_config, ur5e_machine_app'
    assert lines[2] == '  machine waits: 12 semantic (request + wait); 0 external; 0 dwell'
    assert not [line for line in printed.err.splitlines() if line.lstrip().startswith('warning:')], printed.err
    app = out / 'ur5e_machine_app'
    tree = (app / 'bt_trees' / 'task_program.xml').read_text()
    assert tree.count('<RequestMachine ') == 5 and tree.count('<WaitForMachine ') == 7
    assert '<ProcessEvent ' not in tree and '<Sleep ' not in tree
    for number, task in ((1, 'home'), (2, 'turn'), (3, 'index'), (4, 'turn_then_ready'), (5, 'turn_part'),
                         (6, 'ready_check')):
        assert f'<BehaviorTree ID="job_{task}">' in tree, task
    assert '<exec_depend version_gte="2.11.0">trainit_motion_runtime</exec_depend>' in (app / 'package.xml').read_text()
    import yaml as _yaml
    manifest = _yaml.safe_load((out / 'generation_manifest.yaml').read_text())
    assert not manifest.get('warnings')
