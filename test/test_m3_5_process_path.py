"""Process-path semantics (vNext M3.5, ADR-0024, M3_DESIGN §9): the `process` step, a path's `events`,
the `capture` event, their validation and refusals, the split of a path into event-free FollowPath runs
with the event nodes between them, the generated files and the runtime floors.

The project under test is ``test/data/m3_5/validation_b.yaml`` -- Validation B (an inspection program)
on the public UR5e sample cell -- loaded from its directory, and variants of it built from its YAML
data. A process channel is a LOGICAL id (owner guardrail, CELL_SCENE_IMPORT_REVIEW decision 4): never a
ROS topic, a PLC address or a controller symbol. Every generation goes to a pytest temp dir. ROS-free.
"""
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (ProgramRefusals, plan_program,
                                                                program_overview, program_tree_xml)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.program_files import paths_yaml
from trainit_setup_assistant.generator.runtime_requirements import requirement, use_of
from trainit_setup_assistant.model import CanonicalProject, load_project, save_project

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / 'test' / 'data' / 'm3_5'
SOURCE = DATA / 'validation_b.yaml'
APP = 'm3c_app'


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


def raw() -> dict:
    return yaml.safe_load(SOURCE.read_text())


def build(mutate=None) -> CanonicalProject:
    data = raw()
    if mutate is not None:
        mutate(data)
    return CanonicalProject.parse_obj(data)


def app(data):
    return data['application']


def scan(data):
    return app(data)['paths'][0]


def problems(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


def load_error(mutate) -> str:
    with pytest.raises(ValueError) as exc:
        build(mutate)
    return str(exc.value)


def task_nodes(project) -> list:
    """The main task's BT nodes, as (tag, attributes)."""
    root = ET.fromstring(program_tree_xml(project, plan_program(project)).split('\n', 1)[1])
    task = next(t for t in root.iter('BehaviorTree') if t.get('ID').startswith('task_'))
    return [(el.tag, dict(el.attrib)) for el in task.iter() if el.tag not in ('BehaviorTree', 'Sequence')]


def tags(project) -> list:
    out = []
    for tag, a in task_nodes(project):
        if tag == 'FollowPath':
            out.append(f"FollowPath {a['path']} [{a.get('from', '')}..{a.get('to', '')}]")
        elif tag == 'DetectObject':
            out.append(f"capture {a['detector']} -> {a['out_key']}")
        elif tag == 'ProcessEvent':
            out.append(f"{a['kind']} {a['channel']}" + (f" {a['value']}" if 'value' in a else ''))
        elif tag == 'MoveWaypoint':
            out.append(f"move {a['waypoint']}")
        elif tag == 'Sleep':
            out.append(f"sleep {a['msec']}")
        else:
            out.append(tag)
    return out


# --- the model: what loads -------------------------------------------------------------------------

def test_validation_b_loads_and_plans():
    p = load_project(SOURCE)
    prog = p.application.program
    assert [s.kind() for s in prog.tasks[0].steps] == ['move', 'move', 'process', 'path', 'process', 'move']
    assert [e.kind.value for e in prog.paths[0].events] == ['capture'] * 3
    assert [e.at.when.value for e in prog.paths[0].events] == ['after_arrival'] * 3   # the default
    plan = plan_program(p)
    assert plan.warnings == []
    assert plan.process_controller == 'ros'


def test_the_generated_tree_is_the_split_path_with_the_events_between_its_runs():
    assert tags(build()) == [
        'move home', 'move inspection_start', 'process_on inspection_light',
        'FollowPath inspection_scan [0..1]', 'capture inspection_cam -> captures.0',
        'FollowPath inspection_scan [1..2]', 'capture inspection_cam -> captures.1',
        'FollowPath inspection_scan [2..3]', 'capture inspection_cam -> captures.2',
        'process_off inspection_light', 'move home']


def test_a_capture_is_a_triggered_fresh_detect_object_in_the_base_frame():
    caps = [a for t, a in task_nodes(build()) if t == 'DetectObject']
    assert caps[0] == {'detector': 'inspection_cam', 'trigger': 'true', 'target_frame': 'base_link',
                       'timeout_ms': '3000', 'out_key': 'captures.0'}


def test_paths_yaml_keeps_ONE_path_and_records_where_it_stops():
    plan = plan_program(build())
    data = yaml.safe_load(paths_yaml(plan))
    path = data['paths']['inspection_scan']
    assert [s['waypoint'] for s in path['states']] == ['inspection_start', 'w1', 'w2', 'w3']
    assert path['events'] == [{'at': {'state': i, 'when': 'after_arrival'}, 'kind': 'capture'}
                              for i in (1, 2, 3)]
    assert list(data['paths']) == ['inspection_scan']


@pytest.mark.parametrize('kind,extra,node', [
    ('process_on', {'channel': 'dispenser'}, 'process_on dispenser'),
    ('process_off', {'channel': 'dispenser'}, 'process_off dispenser'),
    ('set_value', {'channel': 'dispenser', 'value': 0.35}, 'set_value dispenser 0.35'),
    ('external', {'channel': 'washer_start'}, 'external washer_start'),
])
def test_a_process_step_is_ONE_generic_process_event(kind, extra, node):
    p = build(lambda d: app(d)['tasks'][0]['steps'].insert(2, {'process': {'kind': kind, **extra}}))
    assert tags(p)[2] == node


def test_a_capture_step_captures_where_the_robot_is():
    p = build(lambda d: app(d)['tasks'][0]['steps'].insert(
        2, {'process': {'kind': 'capture', 'detector': 'inspection_cam'}}))
    t = tags(p)
    # numbered in execution order across the task: the step's capture first
    assert t[2] == 'capture inspection_cam -> captures.0'
    assert [x for x in t if x.startswith('capture')][1:] == [
        'capture inspection_cam -> captures.1', 'capture inspection_cam -> captures.2',
        'capture inspection_cam -> captures.3']


def test_an_explicit_capture_key_and_class_and_timeout():
    def mutate(d):
        scan(d)['events'][1].update({'out': 'inspection.middle', 'class_id': 'part', 'timeout_ms': 5000})
    caps = [a for t, a in task_nodes(build(mutate)) if t == 'DetectObject']
    assert [c['out_key'] for c in caps] == ['captures.0', 'inspection.middle', 'captures.1']
    assert caps[1]['class_id'] == 'part' and caps[1]['timeout_ms'] == '5000'


# --- anchors -------------------------------------------------------------------------------------

def _events(d, events):
    scan(d)['events'] = events


def test_every_anchor_form_lands_on_its_state():
    def mutate(d):
        _events(d, [
            {'at': {'state': 'inspection_start'}, 'kind': 'process_on', 'channel': 'lamp'},     # 0 arrival
            {'at': {'segment': 0, 'when': 'start'}, 'kind': 'dwell', 'ms': 200},                # 0 departure
            {'at': {'segment': 0, 'when': 'end'}, 'kind': 'capture', 'detector': 'inspection_cam'},  # 1 arr
            {'at': {'state': 2, 'when': 'before_departure'}, 'kind': 'set_value', 'channel': 'lamp',
             'value': 0.5},
            {'at': {'state': 'w2'}, 'kind': 'external', 'channel': 'trigger_out'},              # 2 arrival
            {'at': {'state': 'w3'}, 'kind': 'process_off', 'channel': 'lamp'},                  # 3 arrival
        ])
    assert tags(build(mutate))[2:] == [
        'process_on inspection_light',
        'FollowPath inspection_scan [0..0]',            # the robot is at the start: nothing is sent
        'process_on lamp', 'sleep 200',                 # after arrival first, then before departure
        'FollowPath inspection_scan [0..1]', 'capture inspection_cam -> captures.0',
        'FollowPath inspection_scan [1..2]',
        'external trigger_out', 'set_value lamp 0.5',    # arrival group before the departure group
        'FollowPath inspection_scan [2..3]', 'process_off lamp',
        'process_off inspection_light', 'move home']


def test_a_trailing_run_follows_the_last_event_state():
    p = build(lambda d: _events(d, [{'at': {'state': 'w1'}, 'kind': 'capture', 'detector': 'inspection_cam'}]))
    assert [t for t in tags(p) if t.startswith(('FollowPath', 'capture'))] == [
        'FollowPath inspection_scan [0..1]', 'capture inspection_cam -> captures.0',
        'FollowPath inspection_scan [1..3]']


def test_samples_never_become_nodes_K_events_make_about_K_plus_1_runs():
    def mutate(d):
        names = [f'p{i}' for i in range(10)]
        app(d)['waypoints'] += [{'name': n, 'frame': 'base', 'position': [-0.45, -0.17 + 0.03 * i, 0.30],
                                 'orientation': [0.0, 1.0, 0.0, 0.0]} for i, n in enumerate(names)]
        scan(d).update({'states': ['inspection_start'] + names, 'segments': [{'type': 'lin'}] * 10,
                        'events': [{'at': {'state': 'p3'}, 'kind': 'capture', 'detector': 'inspection_cam'},
                                   {'at': {'state': 'p7'}, 'kind': 'capture', 'detector': 'inspection_cam'}]})
    t = tags(build(mutate))
    assert sum(x.startswith('FollowPath') for x in t) == 3          # K = 2 events -> 3 runs
    assert sum(x.startswith('capture') for x in t) == 2


def test_a_blended_path_keeps_its_blends_away_from_the_event_states():
    def mutate(d):
        scan(d).update({'continuity': 'blend', 'segments': [{'type': 'lin', 'blend_radius_m': 0.02},
                                                            {'type': 'lin'}, {'type': 'lin'}],
                        'events': [{'at': {'state': 'w2'}, 'kind': 'capture', 'detector': 'inspection_cam'}]})
    plan = plan_program(build(mutate))
    path = yaml.safe_load(paths_yaml(plan))['paths']['inspection_scan']
    assert [s.get('blend_radius_m', 0) for s in path['segments']] == [0.02, 0, 0]
    assert path['events'] == [{'at': {'state': 2, 'when': 'after_arrival'}, 'kind': 'capture'}]


# --- refusals: before any file -------------------------------------------------------------------------

@pytest.mark.parametrize('at,needle', [
    ({'segment': 1, 'fraction': 0.5}, 'R-M3-09: at.fraction: an event placed by a fraction of the segment, '
                                      'inside a segment, is UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL'),
    ({'segment': 1, 'percent': 50}, 'R-M3-09: at.percent'),
    ({'segment': 1, 'distance_m': 0.02}, 'R-M3-09: at.distance_m'),
    ({'segment': 1, 'time_s': 0.3}, 'R-M3-09: at.time_s'),
    ({'position': [0.1, 0.2, 0.3]}, 'R-M3-09: at.position'),
    ({'segment': 1, 'when': 'middle'}, "R-M3-09: at.when 'middle'"),
    ({'state': 'w1', 'when': 'during'}, "R-M3-09: at.when 'during'"),
])
def test_an_event_inside_a_segment_is_refused(at, needle):
    text = load_error(lambda d: scan(d)['events'][0].update({'at': at}))
    assert needle in text, text


def test_an_asynchronous_event_is_refused():
    text = load_error(lambda d: scan(d)['events'][0].update({'sync': 'async'}))
    assert 'R-M3-09: sync: async is UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in text
    assert 'blocking' in text
    assert build(lambda d: scan(d)['events'][0].update({'sync': 'blocking'})) is not None


@pytest.mark.parametrize('event,needle', [
    ({'kind': 'glue_on', 'channel': 'gun'}, "kind 'glue_on' is not a process event kind"),
    ({'kind': 'process_on'}, 'a process_on event needs channel'),
    ({'kind': 'process_on', 'channel': 'gun', 'value': 1}, 'a process_on event takes no value'),
    ({'kind': 'set_value', 'channel': 'gun'}, 'a set_value event needs value'),
    ({'kind': 'set_value', 'channel': 'gun', 'value': 'high'}, 'value must be a number'),
    ({'kind': 'set_value', 'channel': 'gun', 'value': float('nan')}, 'value must be a finite number'),
    ({'kind': 'capture'}, 'a capture event needs detector'),
    ({'kind': 'capture', 'detector': 'inspection_cam', 'channel': 'x'}, 'a capture event takes no channel'),
    ({'kind': 'capture', 'detector': 'inspection_cam', 'out': 'a..b'}, 'must be a blackboard key'),
    ({'kind': 'capture', 'detector': 'inspection_cam', 'out': 'captures.9\n'}, 'must be a blackboard key'),
    ({'kind': 'capture', 'detector': 'inspection_cam', 'class_id': 'it"s<'}, 'must be a detector class id'),
    ({'kind': 'dwell'}, 'a dwell event needs ms'),
    ({'kind': 'dwell', 'ms': 0}, 'ms must be >= 1'),
    ({'kind': 'capture', 'detector': 'inspection_cam', 'trigger': False}, 'always triggers its detector'),
    ({'kind': 'process_on', 'channel': 'gun', 'topic': '/gun/on'}, 'never a topic'),
])
def test_a_malformed_event_is_refused(event, needle):
    text = load_error(lambda d: scan(d)['events'].__setitem__(0, {'at': {'state': 'w1'}, **event}))
    assert needle in text, text


@pytest.mark.parametrize('channel', ['/trainit/process/dispenser', '/plc/output/12', 'MAIN.fbWasher.Cmd_Start',
                                     '192.168.1.4/ADS/851', '%QX0.3', 'dispenser/process_enabled', '1st',
                                     'washer-start', 'on', '', 'lamp\n'])
def test_a_channel_is_a_logical_id_never_a_transport_address(channel):
    text = load_error(lambda d: app(d)['tasks'][0]['steps'][2]['process'].update({'channel': channel}))
    assert 'is not a logical channel id' in text and 'never a ROS topic path, a hardware address' in text


@pytest.mark.parametrize('channel', ['dispenser', 'washer_start', 'camera_trigger', 'inspection_light', 'A1'])
def test_a_logical_channel_id_is_accepted(channel):
    p = build(lambda d: app(d)['tasks'][0]['steps'][2]['process'].update({'channel': channel}))
    assert f'process_on {channel}' in tags(p)


@pytest.mark.parametrize('at,needle', [
    ({'state': 'w9'}, "R-M3-03: path 'inspection_scan' event 0 (capture after arrival at 'w9'): 'w9' is not a "
                      'state of the path (states: inspection_start, w1, w2, w3)'),
    ({'state': 7}, 'R-M3-03: path \'inspection_scan\' event 0 (capture after arrival at state 7): the path has '
                   'no state 7 (states 0..3'),
    ({'segment': 3, 'when': 'end'}, 'the path has no segment 3 (segments 0..2)'),
    ({'state': 'w3', 'when': 'before_departure'}, "R-M3-09: path 'inspection_scan' event 0 (capture before "
                                                  "departure from 'w3'): before departure from the last state"),
])
def test_an_anchor_the_path_does_not_have_is_refused(at, needle):
    text = problems(build(lambda d: scan(d)['events'][0].update({'at': at})))
    assert needle in text, text


def test_an_ambiguous_state_name_must_be_anchored_by_index():
    def mutate(d, at):
        scan(d).update({'states': ['inspection_start', 'w1', 'w2', 'w1'],
                        'events': [{'at': at, 'kind': 'capture', 'detector': 'inspection_cam'}]})
    text = problems(build(lambda d: mutate(d, {'state': 'w1'})))
    assert "the path passes 'w1' 2 times (states 1, 3): anchor by index, 'at: {state: 1}'" in text
    assert 'FollowPath inspection_scan [0..3]' in tags(build(lambda d: mutate(d, {'state': 3})))


def test_an_event_where_the_path_blends_is_refused_never_moved():
    def mutate(d):
        scan(d).update({'continuity': 'blend', 'segments': [{'type': 'lin', 'blend_radius_m': 0.02},
                                                            {'type': 'lin'}, {'type': 'lin'}]})
    text = problems(build(mutate))
    assert ("R-M3-09: path 'inspection_scan' event 0 (capture after arrival at 'w1'): the robot stops at 'w1' "
            'for a blocking event, but segment 0 (inspection_start -> w1) blends into the next one') in text
    assert 'an event is never moved to a nearby point' in text


def test_an_event_on_a_dense_path_is_refused():
    def mutate(d):
        scan(d).pop('states'); scan(d).pop('segments')
        scan(d)['source'] = {'generator': {'type': 'polyline', 'points': [[0.45, 0.2, 0.3], [0.45, 0.1, 0.3]]},
                             'orientation': [1.0, 0.0, 0.0, 0.0]}
        scan(d)['frame'] = 'base_link'
        app(d)['tasks'][0]['steps'].pop(1)
    text = problems(build(mutate))
    assert "R-M3-09: path 'inspection_scan': events on a dense path are UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL" in text


def test_a_capture_needs_a_detector_that_can_be_triggered():
    text = problems(build(lambda d: scan(d)['events'][0].update({'detector': 'top_cam'})))
    assert "R-M3-03: path 'inspection_scan' event 0 (capture after arrival at 'w1'): no detector is named " \
           "'top_cam' (perception.detectors: inspection_cam)" in text
    text = problems(build(lambda d: d['perception']['detectors'][0]['external_node'].pop('detect_service')))
    assert "R-M3-16: path 'inspection_scan' event 0 (capture after arrival at 'w1'): detector 'inspection_cam' " \
           'cannot be triggered' in text and 'external_node.detect_service: true' in text
    text = problems(build(lambda d: d['perception']['detectors'][0].pop('external_node')))
    assert 'R-M3-16' in text and 'no node of this bundle runs it' in text
    # a trainit_perception detector (detector_node) always serves the trigger
    def color_mask(d):
        det = d['perception']['detectors'][0]
        det.pop('external_node'); det['method'] = 'color_mask'
    assert plan_program(build(color_mask)) is not None
    text = problems(build(lambda d: app(d)['tasks'][0]['steps'].insert(
        2, {'process': {'kind': 'capture', 'detector': 'nobody'}})))
    assert "R-M3-03: task 'validation_b' step 3 (process capture): no detector is named 'nobody'" in text


def test_a_capture_key_never_overwrites_a_waypoint_or_another_capture():
    text = problems(build(lambda d: scan(d)['events'][0].update({'out': 'w2'})))
    assert "capture key 'w2' would overwrite waypoint 'w2'" in text
    text = problems(build(lambda d: scan(d)['events'][0].update({'out': 'trainit_paths'})))
    assert "would overwrite the runner's own blackboard entry 'trainit_paths'" in text
    text = problems(build(lambda d: scan(d)['events'][0].update({'out': 'captures.1'})))
    assert "task 'validation_b': two captures write 'captures.1'" in text
    assert 'numbered captures.<n>' in text


def test_a_process_step_has_no_anchor_and_no_dwell():
    text = load_error(lambda d: app(d)['tasks'][0]['steps'][2]['process'].update({'at': {'state': 'w1'}}))
    assert "a process step happens where the task is; 'at' anchors an event on a path" in text
    text = load_error(lambda d: app(d)['tasks'][0]['steps'].__setitem__(2, {'process': {'kind': 'dwell', 'ms': 5}}))
    assert 'a hold in a task is a dwell step' in text


def test_an_event_inside_a_segment_generates_no_file(tmp_path):
    """M3-C: TSA refuses an event placed inside a segment BEFORE generation: no output at all."""
    out = tmp_path / 'refused'
    with pytest.raises(ValueError) as exc:
        p = build(lambda d: scan(d)['events'][1].update({'at': {'segment': 1, 'fraction': 0.5}}))
        Orchestrator().generate(p, out)
    assert 'R-M3-09' in str(exc.value) and 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in str(exc.value)
    assert not out.exists()
    # and a plan-level refusal (an anchor on a blend) stops generation the same way
    def blend(d):
        scan(d).update({'continuity': 'blend', 'segments': [{'type': 'lin', 'blend_radius_m': 0.02},
                                                            {'type': 'lin'}, {'type': 'lin'}]})
    with pytest.raises(ValueError) as exc:
        Orchestrator().generate(build(blend), out)
    assert 'R-M3-09' in str(exc.value)
    assert not out.exists()


# --- the generated bundle --------------------------------------------------------------------------

@pytest.fixture(scope='module')
def bundle(tmp_path_factory):
    needs_ur_description()
    import os
    old = os.getcwd()
    os.chdir(DATA)
    try:
        out = tmp_path_factory.mktemp('m3c') / 'm3c'
        manifest = Orchestrator().generate(load_project(SOURCE), out)
    finally:
        os.chdir(old)
    return out, manifest


def test_validation_b_generates_with_one_expected_warning(bundle):
    out, manifest = bundle
    # the only warning: the external detector's package is documented, never declared (ADR-0003)
    assert [w for w in manifest.warnings if 'm3c_fake_detector' not in w] == []
    assert (out / APP / 'config' / 'paths.yaml').is_file()
    assert (out / APP / 'bt_trees' / 'task_program.xml').is_file()


def test_the_bundle_runs_the_ros_process_controller_and_floors_tmr_2_4_0(bundle):
    out, _ = bundle
    params = yaml.safe_load((out / APP / 'config' / 'bt_params.yaml').read_text())['/**']['ros__parameters']
    assert params['process_controller'] == 'ros'
    pkg = (out / APP / 'package.xml').read_text()
    assert '<exec_depend version_gte="2.4.0">trainit_motion_runtime</exec_depend>' in pkg
    for reason in ('ProcessEvent (a process event on a logical channel)',
                   "DetectObject's trigger (a capture of a fresh result)",
                   "FollowPath from / to (a path's event-free runs)",
                   'paths.yaml events (where a path stops for a process event)'):
        assert reason in pkg


def test_the_floor_rows_are_data_gated():
    tree = ('<root BTCPP_format="4"><BehaviorTree ID="MainTree"><Sequence>'
            '<SubTree ID="task_x" _autoremap="true"/></Sequence></BehaviorTree>'
            '<BehaviorTree ID="task_x"><Sequence>{}</Sequence></BehaviorTree></root>')
    def floor(body, paths=()):
        return requirement(use_of(trees=[tree.format(body)], launches=[], paths=paths)).version
    assert floor('<FollowPath path="p"/>') == '2.2.0'
    assert floor('<FollowPath path="p" from="0" to="1"/>') == '2.4.0'
    assert floor('<ProcessEvent kind="process_on" channel="c"/>') == '2.4.0'
    assert floor('<DetectObject detector="d" trigger="true" out_key="k"/>') == '2.4.0'
    assert floor('<DetectObject detector="d" trigger="false" out_key="k"/>') == '1.7.0'
    with_events = 'schema_version: 1\npaths:\n  p:\n    events: []\n'
    without = 'schema_version: 1\npaths:\n  p:\n    states: []\n'
    assert floor('<FollowPath path="p"/>', [with_events]) == '2.4.0'
    assert floor('<FollowPath path="p"/>', [without]) == '2.2.0'


def test_a_program_without_events_writes_what_it_wrote(tmp_path):
    """Validation A (no process event): the mock controller and its comment, as in M3.3."""
    needs_ur_description()
    import os
    sample = PKG / 'examples' / 'ur5e_suction'
    os.chdir(sample)
    plan = plan_program(load_project('task_program.yaml'))
    assert plan.process_controller == 'mock'
    assert all(not pp.events for pp in plan.paths)
    assert 'events' not in paths_yaml(plan)


# --- the project file and the wizard -------------------------------------------------------------

def test_a_program_with_events_round_trips(tmp_path):
    p = load_project(SOURCE)
    f = save_project(p, tmp_path / 'again.yaml')
    text = f.read_text()
    assert 'events:' in text and 'process:' in text and 'detect_service: true' in text
    q = CanonicalProject.parse_obj(yaml.safe_load(text))
    assert q.application.program == p.application.program


def test_the_wizards_overview_shows_the_events_and_the_process_steps():
    text = program_overview(load_project(SOURCE).application.program)
    assert 'process process_on on channel inspection_light' in text
    assert "after arrival at 'w1': capture by detector inspection_cam" in text
