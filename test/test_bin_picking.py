"""ADR-0016 Decision 4: ``bin_picking`` — the emptying loop.

The template's whole value is the SHAPE of the tree it emits, so the shape is what this
file pins, and each assertion stands for a measured fact rather than a preference:

- the **retry** over the candidate index — the reference cell needed K = 3,33 candidates
  per object delivered, so a linear sequence loses two picks in three;
- the **holding check** after the lift — "the gripper closed" and "the object came up"
  differ on 1 lift in 22, and a failure here is what advances the index;
- the **gated stall recovery** — "nothing detected" and "no candidate worked" are the same
  stall to an operator, and the recovery STEP is emitted while the machine behind it stays
  cell I/O (ADR-0013);
- and the **index mechanism**: ``Script`` sets the key to -1 and increments it as the first
  step of each attempt, so attempt 1 reads candidate 0 — the only moving part Decision 2
  needs beyond ``DetectObject``'s ``keep`` and ``SetWaypointFromDetection``'s ``index``.

Byte gate: selecting ``bin_picking`` changes NOTHING for the other applications, and
``test_generator_golden.py`` / ``test_backend_bytestability.py`` pin that a project that
never selects it regenerates unchanged.

Fast and ROS-free: generation is pure.
"""

import os
import re
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import (AppType, BundleSpec, ExecutionMode,
                                           MotionType, load_project)
import ws_paths

HERE = os.path.dirname(__file__)
GOLDEN_BUNDLE = ws_paths.GOLDEN_BUNDLE
GOLDEN_PROJECT = GOLDEN_BUNDLE / 'project.yaml'

pytestmark = pytest.mark.skipif(not GOLDEN_PROJECT.is_file(),
                                reason='the vision golden project is not present')


# --- fixtures -------------------------------------------------------------------

def _bin_project(**cfg):
    """The published VISION project, switched to bin_picking. It already has what the loop
    needs — a detector, a camera-guided grasp, a step-relative retreat and a grasp/release
    pair — which is the point: bin picking is a TEMPLATE over the same configuration, not
    a different kind of project."""
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix('cell')
    p.application.type = AppType.BIN_PICKING
    bp = p.application.bin_picking
    bp.attached_topic = '/suction/attached'
    bp.recovery_channel = 'bin_agitator'
    for key, value in cfg.items():
        setattr(bp, key, value)
    return p


def _tree(project):
    return get_application(project.application.type).build_tree_xml(project)


def _root(xml: str):
    return ET.fromstring(xml[xml.index('<root'):])


def _path_to(root, tag):
    """The chain of element tags from <root> down to the first ``tag``, so a test can
    assert the NESTING and not merely the presence of a node."""
    def walk(node, chain):
        for child in node:
            if child.tag == tag:
                return chain + [node.tag, child.tag]
            got = walk(child, chain + [node.tag])
            if got:
                return got
        return None
    return walk(root, []) or []


# --- 1. the loop's shape --------------------------------------------------------

def test_the_tree_is_a_repeat_over_a_fallback_over_the_pick():
    """ADR-0016's figure, node for node: Repeat -> Fallback (pick | stall) -> Sequence."""
    root = _root(_tree(_bin_project()))
    repeat = root.find('.//Repeat')
    assert repeat is not None, 'no emptying loop'
    assert len(repeat) == 1, 'Repeat is a decorator: exactly one child'
    cycle = repeat[0]
    assert cycle.tag == 'Fallback', 'the cycle must fall back to the stall branch'
    assert [c.tag for c in cycle] == ['Sequence', 'Fallback'], \
        'the cycle is "pick, or recover from the stall"'
    assert cycle[0].get('name') == 'PickOneObject'
    assert cycle[1].get('name') == 'BinStalled'


def test_detect_fires_once_per_cycle_outside_the_retry():
    """The candidates are ranked ONCE and then tried in the published order (Decision 1).
    A DetectObject inside the retry would re-rank between attempts and the index would
    point at a different object each time."""
    root = _root(_tree(_bin_project()))
    pick = root.find('.//Fallback/Sequence')
    tags = [c.tag for c in pick]
    assert tags.count('DetectObject') == 1
    assert tags.index('DetectObject') < tags.index('RetryUntilSuccessful')
    retry = pick.find('RetryUntilSuccessful')
    assert retry.find('.//DetectObject') is None, 'DetectObject must be OUTSIDE the retry'


def test_detect_asks_for_exactly_as_many_candidates_as_the_loop_can_try():
    """keep = tries. Asking for more publishes blackboard entries nothing reads; asking
    for fewer makes the last attempts index past the end of the list."""
    for tries in (2, 3, 7):
        root = _root(_tree(_bin_project(tries=tries)))
        det = root.find('.//DetectObject')
        assert det.get('keep') == str(tries)
        assert root.find('.//RetryUntilSuccessful').get('num_attempts') == str(tries)


def test_the_retry_wraps_one_candidate_and_nothing_else():
    """Inside: the binding, the approach/grasp/retreat and the holding check. Outside:
    the place. Retrying the place would drop the object twice."""
    root = _root(_tree(_bin_project()))
    retry = root.find('.//RetryUntilSuccessful')
    assert len(retry) == 1 and retry[0].tag == 'Sequence'
    inside = [c.tag for c in retry[0]]
    assert inside[0] == 'Script'                       # the index advances first
    assert 'SetWaypointFromDetection' in inside
    assert 'MoveWaypoint' in inside
    assert 'CloseGripper' in inside
    assert 'CheckRobotState' in inside
    moved = [c.get('waypoint') for c in retry[0] if c.tag in ('MoveWaypoint',
                                                              'ServoTrack')]
    assert 'pick' in moved and 'post_pick' in moved
    assert 'place' not in moved and 'pre_place' not in moved
    pick_seq = root.find('.//Fallback/Sequence')
    after = [c.get('waypoint') for c in pick_seq if c.tag == 'MoveWaypoint']
    assert 'place' in after and 'pre_place' in after


def test_the_candidate_index_is_a_script_counter_the_retry_advances():
    """Decision 2 adds two PORTS and no control node: BT.CPP v4's own Script advances the
    key that `index` reads. -1 before the retry, +1 as the first step of each attempt, so
    attempt 1 reads candidate 0."""
    xml = _tree(_bin_project())
    root = _root(xml)
    pick = root.find('.//Fallback/Sequence')
    reset = [c for c in pick if c.tag == 'Script']
    assert len(reset) == 1 and reset[0].get('code') == 'candidate := -1'
    bump = root.find('.//RetryUntilSuccessful/Sequence/Script')
    assert bump.get('code') == 'candidate := candidate + 1'
    for setter in root.findall('.//SetWaypointFromDetection'):
        assert setter.get('index') == '{candidate}', \
            'the index must be READ from the blackboard, not hardcoded'


def test_the_grasp_binding_keeps_its_offsets_and_orientation_policy():
    """The standoff is configuration, and it already lives on the waypoint's own vision
    binding — the template reads it, it does not re-invent a field for it."""
    p = _bin_project()
    wp = next(w for w in p.application.waypoints if w.vision is not None)
    setter = next(s for s in _root(_tree(p)).findall('.//SetWaypointFromDetection')
                  if s.get('waypoint') == wp.name)
    assert setter.get('dz') == f'{wp.vision.dz:.3f}'
    assert setter.get('orientation') == wp.vision.orientation
    assert setter.get('from') == f'detected.{wp.vision.detector}'


def test_the_retreat_is_rederived_inside_the_retry():
    """The lift hangs off the grasp it ACTUALLY took, so its pose has to be recomputed per
    attempt — a retreat derived once would point at the first candidate for ever."""
    root = _root(_tree(_bin_project()))
    inner = root.find('.//RetryUntilSuccessful/Sequence')
    rel = [c for c in inner if c.tag == 'SetWaypointRelative']
    assert rel, 'the step-relative retreat is not re-derived per attempt'
    assert rel[0].get('from') == 'pick'
    # and the relative step hanging off a TAUGHT pose is derived once, at the cycle top
    cycle_rel = [c for c in root.find('.//Fallback/Sequence')
                 if c.tag == 'SetWaypointRelative']
    assert [c.get('waypoint') for c in cycle_rel] == ['post_place']


# --- 2. the holding check -------------------------------------------------------

def test_the_holding_check_closes_the_candidate_attempt():
    root = _root(_tree(_bin_project()))
    inner = root.find('.//RetryUntilSuccessful/Sequence')
    assert [c.tag for c in inner][-1] == 'CheckRobotState', \
        'the attempt must END on the verification: its FAILURE is what retries'
    chk = inner.find('CheckRobotState')
    assert chk.get('require_attached') == 'true'
    assert chk.get('attached_topic') == '/suction/attached'
    assert chk.get('timeout_ms') == '3000'


def test_the_holding_check_can_be_switched_off_and_then_emits_nothing():
    root = _root(_tree(_bin_project(holding_check=False)))
    assert root.find('.//CheckRobotState') is None
    problems = '\n'.join(get_application(AppType.BIN_PICKING).validate(
        _bin_project(holding_check=False)))
    assert 'holding_check is OFF' in problems
    assert '1 lift in 22' in problems


def test_a_holding_check_with_no_topic_is_REFUSED_before_a_file_exists(tmp_path):
    """vNext M0C #3. CheckRobotState has NO default topic: ``require_attached`` without
    ``attached_topic`` fails on every tick (TMR logs "require_attached set but no
    'attached_topic'"). The loop it closes could therefore never deliver one object, and
    every candidate would look like a missed pick. That is not a partial bundle that still
    helps: generation is refused, naming the field, before a single file exists — and the
    validation the GUI shows says the same instead of promising a runtime default."""
    p = _bin_project(attached_topic='')
    out = tmp_path / 'bundle'
    with pytest.raises(ValueError, match='attached_topic'):
        Orchestrator().generate(p, str(out))
    assert not out.exists()
    text = '\n'.join(get_application(AppType.BIN_PICKING).validate(p))
    assert 'attached_topic' in text and 'has no default topic' in text
    assert 'falls back to the runtime default topic' not in text      # the old, false note


def test_the_holding_check_switched_off_needs_no_topic(tmp_path):
    p = _bin_project(attached_topic='', holding_check=False)
    Orchestrator().generate(p, str(tmp_path / 'bundle'))      # no refusal


# --- 3. the gated recovery ------------------------------------------------------

def test_the_recovery_is_a_pulse_on_a_named_cell_io_channel():
    """The template emits the STEP; what the channel actuates is cell I/O (ADR-0013). The
    pulse is the semantic Cmd_/Sts_ handshake the ProcessController already speaks."""
    root = _root(_tree(_bin_project(recovery_channel='bin_agitator',
                                    recovery_dwell_ms=1200)))
    stalled = root.find(".//Fallback[@name='BinStalled']")
    assert [c.tag for c in stalled] == ['Sequence', 'Log']
    recover = stalled[0]
    assert [c.tag for c in recover] == ['Log', 'EnableProcess', 'Sleep',
                                        'DisableProcess']
    assert recover.find('EnableProcess').get('channel') == 'bin_agitator'
    assert recover.find('DisableProcess').get('channel') == 'bin_agitator'
    assert recover.find('Sleep').get('msec') == '1200'
    assert stalled[1].get('message') == 'bin empty'


def test_the_recovery_is_GATED_on_the_channel_being_configured():
    """No channel, no step — and the stall branch degrades to the honest thing: a log."""
    root = _root(_tree(_bin_project(recovery_channel='')))
    stalled = root.find(".//Fallback[@name='BinStalled']")
    assert [c.tag for c in stalled] == ['Log']
    assert stalled[0].get('message') == 'bin empty'
    assert root.find('.//EnableProcess') is None
    assert root.find('.//DisableProcess') is None
    assert 'no recovery_channel' in '\n'.join(
        get_application(AppType.BIN_PICKING).validate(_bin_project(recovery_channel='')))


def test_a_zero_dwell_emits_no_sleep():
    root = _root(_tree(_bin_project(recovery_dwell_ms=0)))
    recover = root.find(".//Sequence[@name='RecoverFromStall']")
    assert [c.tag for c in recover] == ['Log', 'EnableProcess', 'DisableProcess']


def test_the_generated_tree_says_the_agitator_is_not_in_the_template():
    """The boundary has to be visible where someone reads the tree, not only in a doc."""
    xml = _tree(_bin_project())
    assert 'CELL I/O' in xml and 'ADR-0013' in xml
    assert 'not this application' in xml
    import trainit_setup_assistant.applications.bin_picking as mod
    assert 'ADR-0013' in type(get_application(AppType.BIN_PICKING)).__doc__
    assert 'ADR-0013' in mod.__doc__
    assert 'does NOT contain: the agitator' in mod.__doc__


# --- 4. configuration, not code -------------------------------------------------

def test_the_loop_count_comes_from_loop_cycles():
    """-1 runs until the operator stops it; a bounded run is configuration."""
    assert _root(_tree(_bin_project())).find('.//Repeat').get('num_cycles') == '-1'
    p = _bin_project()
    p.application.loop_cycles = 5
    assert _root(_tree(p)).find('.//Repeat').get('num_cycles') == '5'


def test_the_per_move_execution_mode_still_applies_inside_the_loop():
    """ADR-0015's axis is orthogonal to the loop: a streamed approach is still a
    <ServoTrack>, inside the retry like anywhere else."""
    p = _bin_project()
    seg = p.application.segment_for('pick')
    seg.execution = ExecutionMode.SERVO
    seg.motion = MotionType.FREE
    inner = _root(_tree(p)).find('.//RetryUntilSuccessful/Sequence')
    assert inner.find("ServoTrack[@waypoint='pick']") is not None
    assert inner.find("MoveWaypoint[@waypoint='pick']") is None


def test_the_loop_configuration_round_trips_and_defaults_are_inert():
    from trainit_setup_assistant.model import CanonicalProject, project_to_dict
    p = load_project(str(GOLDEN_PROJECT))
    cfg = p.application.bin_picking
    assert (cfg.tries, cfg.holding_check, cfg.recovery_channel) == (3, True, '')
    p.application.bin_picking.tries = 5
    p.application.bin_picking.recovery_channel = 'shaker'
    again = CanonicalProject.parse_obj(project_to_dict(p))
    assert again == p
    assert again.application.bin_picking.tries == 5
    assert again.application.bin_picking.recovery_channel == 'shaker'


@pytest.mark.parametrize('app_type', [AppType.PICK_AND_PLACE,
                                      AppType.VISION_GUIDED_MOTION])
def test_the_loop_configuration_is_ignored_by_every_other_application(app_type):
    """Only bin_picking reads it, which is why the field can default without touching one
    byte of what the other templates emit."""
    p = load_project(str(GOLDEN_PROJECT))
    p.application.type = app_type
    before = get_application(app_type).build_tree_xml(p)
    p.application.bin_picking.tries = 9
    p.application.bin_picking.recovery_channel = 'shaker'
    p.application.bin_picking.holding_check = False
    assert get_application(app_type).build_tree_xml(p) == before


# --- 5. validation --------------------------------------------------------------

def test_a_single_try_is_called_out_as_an_inert_retry():
    text = '\n'.join(get_application(AppType.BIN_PICKING).validate(_bin_project(tries=1)))
    assert 'tries = 1' in text and '3,33' in text


def test_no_grasp_and_no_roles_means_no_loop_and_it_says_so():
    """The retry needs a slice to wrap. With neither a pick role nor a grasp there is
    none, so the template degrades to the linear vision tree AND says why at the top of
    the file — a tree that silently lost its loop is worse than no tree."""
    p = _bin_project()
    p.application.tool_actions = [a for a in p.application.tool_actions
                                  if a.kind.value != 'grasp']
    text = '\n'.join(get_application(AppType.BIN_PICKING).validate(p))
    assert 'no grasp tool action' in text
    assert 'the retry has no slice to wrap' in text
    xml = _tree(p)
    assert xml.startswith('<!-- bin_picking: NO retry loop was emitted')
    assert '<RetryUntilSuccessful' not in xml


def test_policy_steps_are_refused_loudly_rather_than_dropped_silently():
    from trainit_setup_assistant.model import PolicyStep
    p = _bin_project()
    p.application.policies = [PolicyStep(name='grasp_policy', before_waypoint='pick',
                                         card='/nonexistent.yaml')]
    text = '\n'.join(get_application(AppType.BIN_PICKING).validate(p))
    assert 'does not emit learned-policy steps' in text
    assert 'RunPolicy' not in _tree(p)


def test_the_execution_mode_checks_are_inherited():
    """validate_common (ADR-0015) is inherited, not re-derived, like every vision check."""
    p = _bin_project()
    seg = p.application.segment_for('pick')
    seg.execution = ExecutionMode.SERVO
    seg.motion = MotionType.LIN
    text = '\n'.join(get_application(AppType.BIN_PICKING).validate(p))
    assert 'execution "servo" cannot be combined with motion "lin"' in text


# --- 6. the bundle --------------------------------------------------------------

def test_the_bundle_ships_the_loop_as_bin_picking_xml():
    p = _bin_project()
    with tempfile.TemporaryDirectory() as tmp:
        manifest = Orchestrator().generate(p, tmp)
        out = Path(tmp)
        tree = out / 'cell_app' / 'bt_trees' / 'bin_picking.xml'
        assert tree.is_file(), 'the loop is not in the bundle'
        ET.fromstring(tree.read_text()[tree.read_text().index('<root'):])
        assert 'RetryUntilSuccessful' in tree.read_text()
        # the launch and the Groot2 project point at it
        assert 'bin_picking.xml' in (
            out / 'cell_app' / 'launch' / 'trainit_bt.launch.py').read_text()
        assert (out / 'cell_app' / 'groot2' /
                f'{p.robot.robot_name}_bin_picking.btproj').is_file()
        # bt_params is the SAME schema as every other application
        params = (out / 'cell_app' / 'config' / 'bt_params.yaml').read_text()
        assert 'task_parameters:' in params
        assert 'tries' not in params and 'recovery_channel' not in params
        del manifest


def test_the_loop_survives_a_namespaced_cell():
    """The two ADR-0016 decisions are independent: the tree does not change, and the
    gripper action in bt_params follows the namespace."""
    p = _bin_project()
    p.robot.cell_namespace = 'r3'
    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(p, tmp)
        out = Path(tmp)
        assert _root((out / 'cell_app' / 'bt_trees' / 'bin_picking.xml').read_text()
                     ).find('.//RetryUntilSuccessful') is not None
        params = (out / 'cell_app' / 'config' / 'bt_params.yaml').read_text()
        assert re.search(r'gripper_action: "/r3/\S+"', params)


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
