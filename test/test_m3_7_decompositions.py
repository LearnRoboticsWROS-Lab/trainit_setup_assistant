"""vNext M3.7 (M3_DESIGN §19, M3_IMPLEMENTATION_PLAN M3.7): the SIX canonical decompositions as
GENERATION TESTS -- machine tending, inspection, a CNC-like process path, pick and place, capsule bin
picking, bottle nesting -- each a Canonical Task Program of the generic steps only, and the SEMANTIC
comparison of two shipped templates (pick_and_place, bin_picking) with their canonical re-expression.

What they prove: the canonical language is general enough. Every decomposition is application type
``task_program``; its tree uses the generic step expansions and BehaviorTree.CPP's control nodes only;
no AppType, task subclass, BT node or runtime class exists for any of them. The decompositions live in
``test/data/m3_7/decompositions/`` (one file each, with what it shows and what is deferred) and are
merged into a test cell; nothing here changes a shipped template, the golden or the public sample: the
re-expressions are compared, never shipped and never blessed.
"""
import copy
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.applications.task_program import (ProgramRefusals, plan_program,
                                                                program_tree_xml)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import AppType, CanonicalProject, load_project
from trainit_setup_assistant.model.enums import DEPRECATED_APP_TYPES

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / 'test' / 'data' / 'm3_7'
DECOMPOSITIONS = DATA / 'decompositions'
CELL = DATA / 'composition.yaml'
SAMPLE = PKG / 'examples' / 'ur5e_suction'
BIN_PICKING_TREE = PKG / 'test' / 'data' / 'm2_7_mf' / 'bin_picking_canonical.xml'   # the template's, pinned

SIX = ('a_machine_tending', 'b_inspection', 'c_process_path', 'd_pick_and_place', 'e_bin_picking',
       'f_bottle_nesting')
# What a canonical step or a control step expands to (M3_SCHEMA_PROPOSAL §1.1), and the tree's own tags:
# the whole vocabulary a decomposition may use -- generic, none of it named after an application.
GENERIC_NODES = {'root', 'BehaviorTree', 'Sequence', 'SubTree', 'Repeat', 'RetryUntilSuccessful', 'Fallback',
                 'AddCollisionObject', 'MoveWaypoint', 'FollowPath', 'ProcessEvent', 'Sleep', 'DetectObject',
                 'DetectCandidates', 'SetWaypointFromCandidate', 'CloseGripper', 'OpenGripper', 'VerifyGrasp',
                 'CommandActuator'}


def needs_ur_description():
    try:
        import xacro  # noqa: F401
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory('ur_description')
    except Exception:                                   # noqa: BLE001
        import eef_fixture
        eef_fixture.skip_or_fail('the UR5e sample cell needs ur_description and xacro in a sourced '
                                 'ROS 2 environment (sudo apt install ros-humble-ur-description)')


def decomposition(name: str) -> dict:
    return yaml.safe_load((DECOMPOSITIONS / f'{name}.yaml').read_text())


def project_data(name: str, mutate=None) -> dict:
    """A decomposition merged into its cell: the public sample's for D (the template it is compared
    with runs there), the M3.7 composition cell for the others."""
    base = SAMPLE / 'project.yaml' if name == 'd_pick_and_place' else CELL
    data = yaml.safe_load(base.read_text())
    dec = decomposition(name)
    data['application'] = dec['application']
    data.pop('perception', None)
    if 'perception' in dec:
        data['perception'] = dec['perception']
    data['schema_version'] = 4
    if mutate is not None:
        mutate(data)
    return data


def build(name: str, mutate=None) -> CanonicalProject:
    return CanonicalProject.parse_obj(project_data(name, mutate))


@pytest.fixture(autouse=True)
def _in_the_cell_dir(monkeypatch, request):
    # the urdf paths are relative to the cell's directory
    monkeypatch.chdir(SAMPLE if 'd_pick' in request.node.name or 'pick_and_place' in request.node.name
                      else DATA)


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    """Every test here plans on the public UR5e sample cell: it SKIPS, naming ur_description, on a
    machine that does not have it yet (the fresh-machine check's suite before the sample; vNext M3.7)."""
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def tree_of(project) -> ET.Element:
    return ET.fromstring(program_tree_xml(project, plan_program(project)).split('\n', 1)[1])


def tags(project) -> set:
    return {el.tag for el in tree_of(project).iter()}


# --- the six: generic, valid, generated ----------------------------------------------------------------

@pytest.mark.parametrize('name', SIX)
def test_each_decomposition_is_a_task_program_of_generic_steps(name, monkeypatch):
    if name == 'd_pick_and_place':
        monkeypatch.chdir(SAMPLE)
    p = build(name)
    assert p.application.type is AppType.TASK_PROGRAM
    plan = plan_program(p)
    assert plan.warnings == []
    assert tags(p) <= GENERIC_NODES, tags(p) - GENERIC_NODES
    # every task of the decomposition runs (main_task and what it calls)
    assert plan.running == [t.name for t in p.application.program.tasks]


@pytest.mark.parametrize('name', SIX)
def test_each_decomposition_generates_a_bundle(name, tmp_path, monkeypatch):
    needs_ur_description()
    if name == 'd_pick_and_place':
        monkeypatch.chdir(SAMPLE)
    manifest = Orchestrator().generate(build(name), str(tmp_path / 'out'))
    expected = []
    for det in (decomposition(name).get('perception') or {}).get('detectors', []):
        expected.append(f"external detector package(s) {det['external_node']['package']} are LAUNCHED")
    if name == 'd_pick_and_place':        # the sample's payload: the canonical idioms attach none
        expected.append('scene payload "box" is not used by the task program: its grasp and release steps '
                        'do not attach an object to the planning scene')
    warnings = manifest.as_dict()['warnings']
    assert len(warnings) == len(expected) and all(any(e in w for w in warnings) for e in expected), warnings
    app = next(p for p in (tmp_path / 'out').iterdir() if p.name.endswith('_app'))
    assert (app / 'config' / 'tasks.yaml').is_file() and (app / 'bt_trees' / 'task_program.xml').is_file()


def test_no_application_specific_vocabulary_exists():
    """No AppType, task subclass, BT node or runtime class per use case: the application types are the
    three shipped templates and task_program; the four names of M3 decision 7 are refused at load."""
    live = [t.value for t in AppType if t.value not in DEPRECATED_APP_TYPES]
    assert live == ['pick_and_place', 'vision_guided_motion', 'bin_picking', 'task_program']
    from trainit_setup_assistant.catalog import TMR_BT_NODES
    words = ('tend', 'inspect', 'cnc', 'nest', 'bin', 'pick', 'place', 'glu', 'dispens', 'weld')
    canonical = GENERIC_NODES & set(TMR_BT_NODES)
    assert not [n for n in canonical if any(w in n.lower() for w in words)]
    from trainit_setup_assistant.model import program
    models = [n for n in dir(program) if isinstance(getattr(program, n), type)]
    assert not [n for n in models if any(w in n.lower() for w in words)]


# --- A. machine tending --------------------------------------------------------------------------------------

def test_a_machine_tending_is_moves_idioms_and_one_external_handshake():
    t = {b.get('ID'): b for b in tree_of(build('a_machine_tending')).iter('BehaviorTree')}
    cycle = t['task_tend'].find('Sequence/Repeat')
    assert cycle.get('num_cycles') == '-1'
    body = [(n.tag, n.get('ID') or n.get('kind') or n.get('msec')) for n in cycle.find('Sequence')]
    assert body == [('SubTree', 'task_load_machine'), ('ProcessEvent', 'external'), ('Sleep', '500'),
                    ('SubTree', 'task_unload_machine')]
    assert cycle.find('Sequence/ProcessEvent').get('channel') == 'machine_cycle_start'   # a LOGICAL id


# --- B. inspection -------------------------------------------------------------------------------------------

def test_b_inspection_is_a_path_with_captures_given_by_a_task():
    p = build('b_inspection')
    t = {b.get('ID'): b for b in tree_of(p).iter('BehaviorTree')}
    scan = [(n.tag, n.get('out_key') or n.get('kind')) for n in t['task_inspect_part'].find('Sequence')
            if n.tag in ('DetectObject', 'ProcessEvent')]
    assert scan == [('ProcessEvent', 'process_on'), ('DetectObject', 'captures.0'),
                    ('DetectObject', 'captures.1'), ('DetectObject', 'captures.2'),
                    ('ProcessEvent', 'process_off')]
    assert [d.name for d in p.application.program.tasks[1].outputs] == ['captures.0', 'captures.1',
                                                                       'captures.2']


# --- C. a CNC-like process path ---------------------------------------------------------------------------------

def test_c_a_process_path_is_one_dense_path_between_process_events():
    p = build('c_process_path')
    plan = plan_program(p)
    steps = [(n.tag, n.get('kind') or n.get('path') or n.get('waypoint'))
             for n in tree_of(p).find("BehaviorTree[@ID='task_machine_contour']/Sequence")]
    assert steps == [('MoveWaypoint', 'approach'), ('MoveWaypoint', 'start'), ('ProcessEvent', 'process_on'),
                     ('FollowPath', 'contour'), ('ProcessEvent', 'process_off'), ('MoveWaypoint', 'retract'),
                     ('MoveWaypoint', 'home')]
    assert plan.paths[0].dense and plan.paths[0].resolution.provider == 'cartesian_waypoints'
    assert plan.backend == 'cartesian_waypoints' and plan.process_controller == 'ros'


def test_c_the_designs_file_source_is_a_later_form():
    def from_file(data):
        data['application']['paths'][0]['source'] = {'file': 'paths/contour.csv'}
    with pytest.raises(ValueError, match='a dense path read from a file is planned for a later TSA version'):
        build('c_process_path', from_file)


# --- D. pick and place, compared with the shipped template --------------------------------------------------------

def _trace(nodes, entry):
    """A tree's actions as (what, motion, position): a move by its motion type and its TCP position
    (or its named state), an end-effector command, a verification, a planning-scene payload change."""
    out = []
    for el in nodes:
        if el.tag == 'MoveWaypoint':
            motion, where = entry(el.get('waypoint'))
            out.append(('move', motion, where))
        elif el.tag in ('CloseGripper', 'OpenGripper'):
            out.append(('close' if el.tag == 'CloseGripper' else 'open', None, None))
        elif el.tag == 'VerifyGrasp':
            out.append(('verify ' + el.get('expect'), None, None))
        elif el.tag in ('AttachObject', 'DetachObject'):
            out.append(('attach' if el.tag == 'AttachObject' else 'detach', None, None))
    return out


def test_d_pick_and_place_is_the_template_semantically():
    """The canonical re-expression on the SAME cell does what the shipped template does: the same
    motion types in the same order, the gripper closed after the LIN to the pick and opened after the
    LIN to the place, at the same poses, leaving along the same line. The differences are named."""
    template_project = load_project(SAMPLE / 'project.yaml')
    template = ET.fromstring(get_application('pick_and_place').build_tree_xml(template_project).split('\n', 1)[1])
    app = template_project.application
    motion = {s.to_waypoint: s.motion.value for s in app.segments}

    def template_entry(name):
        wp = app.waypoint_by_name(name)
        return motion[name], (tuple(wp.position) if wp.position else f'named:{wp.named or name}')
    tpl = _trace(template.find('BehaviorTree/Sequence'), template_entry)

    canonical = build('d_pick_and_place')
    plan = plan_program(canonical)

    def canonical_entry(name):
        e = plan.entries[name]
        return e['motion'], (tuple(e['position']) if 'position' in e else f"named:{e['named']}")
    can = _trace(tree_of(canonical).find("BehaviorTree[@ID='task_pick_and_place']/Sequence"), canonical_entry)

    # 1. the same motions, in the same order, and the gripper at the same moments
    assert [(w, m) for w, m, _ in tpl if w != 'attach' and w != 'detach'] == [(w, m) for w, m, _ in can]
    # 2. the same grasp and release poses, the same departures (post_pick / post_place = the retracts)
    def at(trace, i):
        return [p for w, _, p in trace if w == 'move'][i]
    for i in (2, 3, 5, 6):     # pick, post_pick, place, post_place
        assert at(tpl, i) == pytest.approx(at(can, i)), i
    # 3. the named differences, and no other
    assert [w for w, _, _ in tpl if w in ('attach', 'detach')] == ['attach', 'detach']   # payload in the scene:
    assert not [w for w, _, _ in can if w in ('attach', 'detach')]                     # template only (KNOWN_DEBT)
    assert at(tpl, 1) == 'named:above_pick' and at(can, 1) == pytest.approx((0.45, 0.25, 0.205))
    # (the template's PreGrasp is a taught joint state, the canonical one the pose 15 cm above the pick)


# --- E. bin picking, compared with the shipped template ---------------------------------------------------------

def _skeleton_template(root):
    """The bin_picking template's emptying loop (ADR-0016 Decision 4) as its semantic facts."""
    rep = root.find('.//Repeat')
    cycle = rep[0]
    pick, stall = list(cycle)
    detect = pick.find('DetectObject')
    retry = pick.find('RetryUntilSuccessful')
    one = retry.find('Sequence')
    try_actions = [n.tag if n.tag != 'VerifyGrasp' else 'verify ' + n.get('expect') for n in one
                   if n.tag in ('CloseGripper', 'VerifyGrasp')]
    after = list(pick)[list(pick).index(retry) + 1:]
    place = [n.tag if n.tag != 'VerifyGrasp' else 'verify ' + n.get('expect') for n in after
             if n.tag in ('OpenGripper', 'VerifyGrasp')]
    recover = stall.find('Sequence')
    # the retry advances the candidate index from 0 (Script candidate := -1, then +1 per try)
    indices = list(range(int(retry.get('num_attempts'))))
    return {'loop': (rep.tag, cycle.tag, len(cycle)),
            'samples_before_tries': [n.tag for n in pick].index('RetryUntilSuccessful') > [n.tag for n in pick].index('DetectObject'),
            'keep': int(detect.get('keep')), 'tries_in_rank_order': indices, 'try': try_actions,
            'place': place,
            'stall': [(n.tag.replace('EnableProcess', 'on').replace('DisableProcess', 'off'),
                       n.get('channel') or n.get('msec')) for n in recover if n.tag != 'Log']}


def _ranked(data):
    """E with the template's exact structure: ONE sample keeping 3, then the candidates tried in rank
    order -- a fallback over the indices (the canonical model has no variable to advance)."""
    cycle = data['application']['tasks'][0]['steps'][1]['repeat']['body'][0]['fallback']['bodies']
    grasp = data['application']['tasks'][1]['steps'][0]['grasp']
    cycle[0] = [{'move': {'to': 'bin_view', 'type': 'ptp', 'profile': 'transit'}},
                {'detect': {'detector': 'bin_cam', 'source': 'detections', 'approach': [0.0, 0.0, -1.0],
                            'keep': 3, 'out': 'ranked'}},
                {'fallback': {'bodies': [[{'grasp': dict(grasp, index=i)}] for i in range(3)]}},
                {'task': 'place_capsule'}]
    data['application']['tasks'].pop(1)                    # pick_best: inlined per index


def _skeleton_canonical(project):
    t = {b.get('ID'): b for b in tree_of(project).iter('BehaviorTree')}
    rep = t['task_empty_the_bin'].find('Sequence/Repeat')
    cycle = rep.find('Sequence/Fallback')
    pick, stall = [b for b in cycle]
    detect = pick.find('DetectCandidates')
    tries = pick.find('Fallback')
    first = tries[0]
    try_actions = [n.tag if n.tag != 'VerifyGrasp' else 'verify ' + n.get('expect') for n in first
                   if n.tag in ('CloseGripper', 'VerifyGrasp')]
    place = [n.tag if n.tag != 'VerifyGrasp' else 'verify ' + n.get('expect')
             for n in t['task_place_capsule'].find('Sequence') if n.tag in ('OpenGripper', 'VerifyGrasp')]
    return {'loop': (rep.tag, cycle.tag, len(cycle)),
            'samples_before_tries': [n.tag for n in pick].index('Fallback') > [n.tag for n in pick].index('DetectCandidates'),
            'keep': int(detect.get('keep')),
            'tries_in_rank_order': [int(b.find('SetWaypointFromCandidate').get('index')) for b in tries],
            'try': try_actions, 'place': place,
            'stall': [(n.tag, n.get('channel') or n.get('msec')) for n in stall]}


def test_e_bin_picking_ranked_is_the_template_semantically():
    """The template's emptying loop and its canonical re-expression: a repeat of a fallback between
    picking one part and a stall recovery; ONE sample keeping three candidates; the candidates tried in
    rank order 0, 1, 2, each try closing and verifying the hold; the place verifying the hold, opening
    and verifying the release; the stall a pulse on the agitator's channel."""
    tpl = _skeleton_template(ET.fromstring(BIN_PICKING_TREE.read_text().split('\n', 1)[1]))
    can = _skeleton_canonical(build('e_bin_picking', _ranked))
    assert tpl['loop'] == ('Repeat', 'Fallback', 2) and can['loop'] == ('Repeat', 'Fallback', 2)
    for key in ('samples_before_tries', 'keep', 'tries_in_rank_order', 'place'):
        assert tpl[key] == can[key], key
    assert can['try'] == ['CloseGripper', 'verify held']
    assert tpl['try'] == ['CloseGripper', 'verify held']
    assert [k for k, _ in tpl['stall']] == ['on', 'Sleep', 'off']
    assert [k for k, _ in can['stall']] == ['ProcessEvent', 'Sleep', 'ProcessEvent']
    assert tpl['stall'][0][1] == 'bin_agitator' and can['stall'][0][1] == 'bin_agitator'


def test_e_the_documented_differences_with_the_template():
    """What the template does and the canonical re-expression does not -- named, never hidden:
    the template GATES its recoveries on faults (a fault ends the run, never reads as "bin empty":
    Script + _failureIf, ADR-0022 D5), while a canonical retry or fallback recovers from ANY failure;
    the template logs "bin empty" and toggles the attached-object collision check. The canonical grasp
    adds a PreGrasp (PTP to the standoff) before its LIN approach."""
    text = BIN_PICKING_TREE.read_text()
    assert '_failureIf="eef_fault' in text and '<Script' in text and 'SetAttachedCollisionCheck' in text
    can = program_tree_xml(build('e_bin_picking', _ranked), plan_program(build('e_bin_picking', _ranked)))
    assert '_failureIf' not in can and '<Script' not in can and 'SetAttachedCollisionCheck' not in can
    assert '_pregrasp"' in can


def test_e_the_designs_retry_resamples_the_bin_each_try():
    """E as the design writes it: a retry whose body samples the bin and grasps the best candidate --
    each try a FRESH sample (the canonical model has no variable to advance an index)."""
    t = {b.get('ID'): b for b in tree_of(build('e_bin_picking')).iter('BehaviorTree')}
    retry = t['task_empty_the_bin'].find('.//RetryUntilSuccessful')
    assert retry.get('num_attempts') == '3'
    assert [n.tag for n in retry.find('Sequence')] == ['MoveWaypoint', 'DetectCandidates', 'SubTree']
    assert t['task_pick_best'].find('Sequence/SetWaypointFromCandidate').get('index') == '0'


# --- F. bottle nesting -------------------------------------------------------------------------------------------

def test_f_bottle_nesting_is_a_repeat_of_take_and_release_into_taught_slots():
    t = {b.get('ID'): b for b in tree_of(build('f_bottle_nesting')).iter('BehaviorTree')}
    rep = t['task_nest_bottles'].find('Sequence/Repeat')
    assert rep.get('num_cycles') == '6'
    assert [n.get('ID') for n in rep.find('Sequence')] == ['task_nest_into_0', 'task_nest_into_1']
    take = [n.tag for n in t['task_take_bottle'].find('Sequence')]
    assert take[:3] == ['MoveWaypoint', 'DetectCandidates', 'SetWaypointFromCandidate']


def test_f_the_two_deferred_concepts_are_refused_by_name():
    """The design's ONE missing concept, the pattern target (B-GATE), is refused at load (R-M3-14); its
    composite tool's site at another TCP needs multi-TCP motion and is refused by R-M3-05 (M6)."""
    def pattern(data):
        data['application']['waypoints'].append(
            {'name': 'nest_slot', 'pattern': {'origin': 'nest_0', 'pitch': [0.05, 0, 0], 'counts': [12, 1, 1]}})
    with pytest.raises(ValueError, match='R-M3-14: a pattern target'):
        build('f_bottle_nesting', pattern)

    def other_tcp(data):
        data['robot']['gripper']['eef_parent_link'] = 'tool0'
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(build('f_bottle_nesting', other_tcp))
    assert "grasp site 'main' holds the object at TCP 'tool0'" in str(exc.value)
    assert 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL' in str(exc.value)


# --- the shipped templates are untouched ----------------------------------------------------------------------------

def test_the_comparison_changes_no_shipped_template():
    """The public sample is still a pick_and_place project and its file is the committed one; the
    bin_picking reference tree is the pinned template output."""
    assert load_project(SAMPLE / 'project.yaml').application.type is AppType.PICK_AND_PLACE
    assert BIN_PICKING_TREE.read_text().count('<RetryUntilSuccessful num_attempts="3">') == 1
    data = copy.deepcopy(project_data('d_pick_and_place'))
    assert data['application']['type'] == 'task_program'
