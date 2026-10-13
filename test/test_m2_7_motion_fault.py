"""vNext M2.7 (owner Q8.2, ADR-0022 amendment M2.7): the canonical bin-picking cycle's MOTION gate.

Inside the cycle's absorbing Fallback every MoveWaypoint remaps its optional ``fault`` output
(TMR 2.1.0: a move that fails for a reason that is not the target's, a motion-infrastructure FAULT)
into the end-effector gate key. A class-B motion failure therefore ends the run like an
end-effector FAULT; a target's own failure (no IK, no plan, unreachable: class A) keeps the
existing next-candidate strategy, because MoveWaypoint never writes ``fault`` for it. Legacy trees
and the other canonical applications (no Fallback: a motion failure already ends their run) carry
no ``fault`` on any move.

The M2.7 tree pinned here (TSA's ``test/data/m2_7_mf/bin_picking_canonical.xml``) is a byte copy of
the one TMR ticks (TMR's file of the same name); the M2.6 tree stays frozen as the BEFORE.
ROS-free; every test needs the vision golden (eef_fixture.needs).
"""
import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

import eef_fixture as F
import test_m2_6_eef_bin_picking as M26
from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.generator import runtime_requirements as rr
from trainit_setup_assistant.model import AppType
from trainit_setup_assistant.model.enums import ExecutionMode, MotionType

needs_golden = M26.needs_golden
GATE = 'eef_fault'
DATA_M27 = Path(__file__).resolve().parent / 'data' / 'm2_7_mf'
PINNED_M27_TREE = ('bin_picking_canonical.xml',
                   '93c37df9acb6c497c1d1af1fad107f646f3d3735165034373b855171557b59a5')
# the M2.6 -> M2.7 change of the pinned project's tree: exactly these five lines
# (line, before, after)
INTENDED = [
    (38, '                <MoveWaypoint waypoint="pick"/>',
     '                <MoveWaypoint waypoint="pick" fault="{eef_fault}"/>'),
    (41, '                <MoveWaypoint waypoint="post_pick"/>',
     '                <MoveWaypoint waypoint="post_pick" fault="{eef_fault}"/>'),
    (50, '            <MoveWaypoint waypoint="pre_place"/>',
     '            <MoveWaypoint waypoint="pre_place" fault="{eef_fault}"/>'),
    (51, '            <MoveWaypoint waypoint="place"/>',
     '            <MoveWaypoint waypoint="place" fault="{eef_fault}"/>'),
    (56, '            <MoveWaypoint waypoint="post_place"/>',
     '            <MoveWaypoint waypoint="post_place" fault="{eef_fault}"/>'),
]
EEF_PARAMS = '/**:\n  ros__parameters:\n    gripper_action: ""\n    eef:\n      ids: ["main"]\n'


def _moves(root):
    return [el for el in root.iter() if el.tag == 'MoveWaypoint']


@pytest.mark.parametrize('grasp, release', [(True, False), (True, True), (False, False),
                                            (False, True)])
@needs_golden
def test_mf1_every_move_inside_the_cycle_writes_its_fault_to_the_gate(grasp, release):
    root = M26._root(M26._tree(M26._canonical(grasp, release)))
    cycle = root.find('.//Repeat/Fallback')
    inside = [el for el in cycle.iter() if el.tag == 'MoveWaypoint']
    assert inside and all(set(el.attrib) == {'waypoint', 'fault'}
                          and el.get('fault') == f'{{{GATE}}}' for el in inside)
    # the prologue, outside the Repeat, is not gated (a failure there already ends the run)
    prologue = [el for el in _moves(root) if el not in inside]
    assert prologue and all(set(el.attrib) == {'waypoint'} for el in prologue)
    # the gate expressions themselves are unchanged
    retry = cycle[0].find('RetryUntilSuccessful')
    assert retry[0].get('_failureIf') == f"{GATE} != ''"
    assert cycle[1].get('_failureIf') == f"{GATE} != '' || holding == 1"


@pytest.mark.parametrize('servo_at, noted', [('ready', False),        # the prologue: no cycle there
                                              ('pick', True),           # a candidate move
                                              ('post_place', True)])    # after the release
@needs_golden
def test_mf2_servo_moves_carry_no_fault_and_get_the_note(servo_at, noted):
    p = M26.pinned_canonical_project()
    seg = p.application.segment_for(servo_at)
    seg.motion, seg.execution = MotionType.FREE, ExecutionMode.SERVO
    root = M26._root(M26._tree(p))
    servo = [el for el in root.iter() if el.tag == 'ServoTrack']
    assert [el.attrib for el in servo] == [{'waypoint': servo_at}]        # no fault port to remap
    cycle = root.find('.//Repeat/Fallback')
    cycle_moves = [el for el in cycle.iter() if el.tag == 'MoveWaypoint']
    assert cycle_moves and all(el.get('fault') == f'{{{GATE}}}' for el in cycle_moves)
    app = get_application(AppType.BIN_PICKING)
    note = [m for m in app.validate(p) if 'reports no motion fault' in m]
    head = (f'note: {AppType.BIN_PICKING.value}: {servo_at} stream(s) through ServoTrack, which '
            'reports no motion fault')
    assert (len(note) == 1 and note[0].startswith(head)) if noted else note == []
    assert not [m for m in app.validate(M26.pinned_canonical_project())
                if 'reports no motion fault' in m]


@needs_golden
def test_mf2_a_canonical_project_without_a_loop_detector_gets_no_servo_note():
    """No loop detector, no cycle: the tree is the linear sequence, where any failure ends the run,
    so the note would be false. validate() must not raise either (the sequence is read locally)."""
    p = M26.pinned_canonical_project()
    for wp in p.application.waypoints:
        wp.vision = None
    seg = p.application.segment_for('pick')
    seg.motion, seg.execution = MotionType.FREE, ExecutionMode.SERVO
    app = get_application(AppType.BIN_PICKING)
    assert app._loop_detector(p) is None
    assert app._candidate_bounds(p)[0] is not None      # validate() reaches the note's guard
    problems = app.validate(p)
    assert not [m for m in problems if 'reports no motion fault' in m]
    assert M26._root(M26._tree(p)).find('.//Repeat/Fallback') is None


@needs_golden
def test_mf3_legacy_and_the_other_canonical_applications_carry_no_motion_fault():
    import test_bin_picking as tb
    import test_m2_6_eef_emission as E
    legacy = M26._tree(tb._bin_project())
    moves = [line for line in legacy.splitlines() if '<MoveWaypoint' in line]
    assert moves and all('fault=' not in line for line in moves)
    golden = F.opt_in(E._golden())
    xml = get_application(golden.application.type).build_tree_xml(golden)
    moves = _moves(M26._root(xml))
    assert moves and all('fault' not in el.attrib for el in moves)


@needs_golden
def test_mf4_the_m2_7_tree_is_pinned_and_differs_from_m2_6_by_exactly_the_intended_lines():
    name, sha = PINNED_M27_TREE
    data = (DATA_M27 / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, 'the pinned copy was edited'
    generated = M26._tree(M26.pinned_canonical_project())
    assert generated.encode('utf-8') == data
    m26 = (F.DATA / M26.PINNED_CANONICAL_TREE[0]).read_text().splitlines()
    m27 = generated.splitlines()
    assert len(m26) == len(m27)
    changed = [(i + 1, a, b) for i, (a, b) in enumerate(zip(m26, m27)) if a != b]
    assert changed == INTENDED


@needs_golden
def test_mf5_only_the_canonical_bin_picking_package_gets_2_1_0():
    tree = M26._tree(M26.pinned_canonical_project())
    req = rr.tmr_requirement(trees={'cell_app/bt_trees/bin_picking.xml': tree},
                             bt_params=[EEF_PARAMS])
    assert (req.version, req.comment) == (
        '2.1.0', "trainit_motion_runtime >= 2.1.0 for: MoveWaypoint's fault output")
    m26 = (F.DATA / M26.PINNED_CANONICAL_TREE[0]).read_text()
    assert rr.tmr_requirement(trees={'t.xml': m26}, bt_params=[EEF_PARAMS]).version == '2.0.0'
    import test_bin_picking as tb
    assert rr.tmr_requirement(trees={'t.xml': M26._tree(tb._bin_project())}).version == '1.6.0'
    # the parse sees the port as written: ET reads the same attribute the scanner dates
    assert any('fault' in el.attrib for el in ET.fromstring(tree[tree.index('<root'):]).iter()
               if el.tag == 'MoveWaypoint')
