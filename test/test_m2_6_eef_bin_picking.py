"""F4 (vNext M2.6, ADR-0022 D5): the canonical bin-picking loop's verdict / fault / holding gate,
walked statically -- and the LEGACY loop, byte for byte what TSA 7.6.0 generated.

The static rule of D5, checked here (and ticked as XML by TMR's own M5 regression):

* the reset Script (both keys) is the FIRST step of every cycle;
* EEF nodes inside the retry write ``fault``; EEF nodes elsewhere inside the absorbing Fallback
  write ``reason``; DetectObject's ``fault`` goes to the gate too;
* ``holding`` is set right after the retry and cleared right after the cycle's release (after its
  ``VerifyGrasp(released)`` when one is emitted);
* every retried child, and every later Fallback child, carries the gate (``_failureIf``);
* ``VerifyGrasp`` stands in place of the legacy ``CheckRobotState``.

So a VERDICT inside the retry still advances it and a FRESH empty detection still reaches the
stall branch ("bin empty"), while a fault, any end-effector failure after the retry, a silent
detector and any failure while holding end the run -- they can never become "bin empty".

Fast and ROS-free.
"""
import hashlib
import xml.etree.ElementTree as ET

import pytest

import eef_fixture as F
import ws_paths
from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.model import AppType
from trainit_setup_assistant.model.enums import Backend

# every case is the vision golden switched to bin picking: it skips without the demo repository,
# and FAILS with TRAINIT_REQUIRE_GOLDEN=1 (eef_fixture.needs)
needs_golden = F.needs(ws_paths.GOLDEN_BUNDLE / 'project.yaml',
                       reason='the vision golden project is not present')

GATE, HOLDING = 'eef_fault', 'holding'
EEF_TAGS = ('CloseGripper', 'OpenGripper', 'VerifyGrasp', 'CommandActuator')


def _canonical(grasp=True, release=False, **cfg):
    import test_bin_picking as tb
    p = tb._bin_project(attached_topic='', holding_check=grasp, **cfg)
    p.deployment.modes, p.deployment.default_mode = [Backend.MOCK], Backend.MOCK
    return F.opt_in(p, adapters={'mock': 'trainit_mock'}, grasp=grasp, release=release)


def _tree(project) -> str:
    return get_application(AppType.BIN_PICKING).build_tree_xml(project)


def _root(xml: str):
    return ET.fromstring(xml[xml.index('<root'):])


def _cycle_parts(root):
    cycle = root.find('.//Repeat/Fallback')
    pick, stall = cycle[0], cycle[1]
    return cycle, pick, stall


@pytest.mark.parametrize('grasp, release', [(True, False), (True, True), (False, False),
                                            (False, True)])
@needs_golden
def test_f4_the_gate_walker(grasp, release):
    from trainit_setup_assistant.generator.eef_contract import eef_contract_problems
    p = _canonical(grasp, release)
    assert eef_contract_problems(p) == []
    root = _root(_tree(p))
    cycle, pick, stall = _cycle_parts(root)
    # the reset, first step of every cycle, both keys
    first = pick[0]
    assert (first.tag, first.get('code')) == ('Script', f"{GATE} := ''; {HOLDING} := 0")
    # the retried child carries the gate; so does the later Fallback child
    retry = pick.find('RetryUntilSuccessful')
    assert len(retry) == 1 and retry[0].get('_failureIf') == f"{GATE} != ''"
    assert stall.tag == 'Fallback' and stall.get('name') == 'BinStalled'
    assert stall.get('_failureIf') == f"{GATE} != '' || {HOLDING} == 1"
    # inside the retry: fault; elsewhere in the absorbing Fallback: reason
    inside = [el for el in retry.iter() if el.tag in EEF_TAGS]
    after = [el for el in list(pick)[list(pick).index(retry) + 1:]
             for el in el.iter() if el.tag in EEF_TAGS]
    assert inside and after
    assert all(el.get('fault') == f'{{{GATE}}}' and 'reason' not in el.attrib for el in inside)
    assert all(el.get('reason') == f'{{{GATE}}}' and 'fault' not in el.attrib for el in after)
    # DetectObject's fault gates too (a silent or stale detector is not an empty bin)
    detects = [el for el in pick.iter() if el.tag == 'DetectObject']
    assert detects and all(el.get('fault') == f'{{{GATE}}}' for el in detects)
    # holding: set right after the retry, cleared right after the release (and its check)
    steps = list(pick)
    i = steps.index(retry)
    assert (steps[i + 1].tag, steps[i + 1].get('code')) == ('Script', f'{HOLDING} := 1')
    opened = next(k for k, el in enumerate(steps) if el.tag == 'OpenGripper')
    clear = opened + (2 if release else 1)
    if release:
        assert (steps[opened + 1].tag, steps[opened + 1].get('expect')) == ('VerifyGrasp',
                                                                            'released')
    assert (steps[clear].tag, steps[clear].get('code')) == ('Script', f'{HOLDING} := 0')
    # the holding check is the contract's, never CheckRobotState
    assert root.find('.//CheckRobotState') is None
    held_in_retry = [el for el in inside if el.tag == 'VerifyGrasp']
    assert [el.get('expect') for el in held_in_retry] == (['held'] if grasp else [])
    # the stall branch still ends in "bin empty" -- only a fresh empty detection reaches it
    assert stall[-1].tag == 'Log' and stall[-1].get('message') == 'bin empty'


@needs_golden
def test_f4_the_canonical_loop_is_the_legacy_loop_plus_the_gate():
    """Same skeleton, node for node, as the legacy loop: the gate adds Scripts, attributes and the
    contract's nodes, and removes CheckRobotState and SetReleasePolicy -- nothing else moves."""
    import test_bin_picking as tb
    legacy = _root(_tree(tb._bin_project()))
    canonical = _root(_tree(_canonical()))

    def skeleton(root):
        drop = {'Script', 'CheckRobotState', 'SetReleasePolicy', 'VerifyGrasp'}
        return [(el.tag, el.get('name'), el.get('waypoint')) for el in root.iter()
                if el.tag not in drop]
    assert skeleton(legacy) == skeleton(canonical)


# The canonical loop TMR's M2.6 tests tick: its test/data/m2_6_contract/bin_picking_canonical.xml is
# a byte copy of test/data/m2_6_eef/bin_picking_canonical.xml, which is what TSA 7.7.0 generated for
# this project (vNext M2.7 generates the motion-gated tree pinned in test_m2_7_motion_fault) -- the
# vision golden switched to bin picking and opted in to the contract, made a bounded and quick run
# (2 cycles, 100 ms dwells, a 1.5 s detection timeout) so TMR's M5 regression can tell "bin empty"
# from a FAILURE in seconds, with every node but the motion real.
PINNED_CANONICAL_TREE = ('bin_picking_canonical.xml',
                         '25fb4e2533423d150c040dee2a19bb9d65905d2a9ff247c0c8e7cf6456716b5b')


def pinned_canonical_project():
    p = _canonical(grasp=True, release=True)
    p.application.loop_cycles = 2
    p.application.bin_picking.recovery_dwell_ms = 100
    p.perception.detect_timeout_ms = 1500
    p.application.segment_for('post_place').wait_after_ms = 100
    return p


@needs_golden
def test_f4_the_canonical_loop_tmr_ticks_is_the_one_tsa_generates():
    """The M2.6 tree, frozen: TMR's M2.6 tests tick this file. Since vNext M2.7 it is the BEFORE of
    the motion gate (owner Q8.2); TSA now generates the M2.7 tree, which test_m2_7_motion_fault's
    test_mf4 pins and compares with this one line by line. This file is never edited."""
    name, sha = PINNED_CANONICAL_TREE
    data = (F.DATA / name).read_bytes()
    assert hashlib.sha256(data).hexdigest() == sha, 'the pinned copy was edited'


@needs_golden
def test_f4_a_legacy_project_gets_no_gate():
    import test_bin_picking as tb
    xml = _tree(tb._bin_project())
    assert GATE not in xml and '_failureIf' not in xml and 'VerifyGrasp' not in xml
    assert '<CheckRobotState require_attached="true"' in xml


# The legacy loop, as TSA 7.6.0 (351eb8c) itself generated it (vNext M2.6 forensics, executed).
LEGACY_TREES_BY_760 = {
    'default': ({}, '49c87437024da756cd5f8a8730890cbf25dd8ccd0f361132ca3beb8f1b0aad34'),
    'no_holding_check': ({'holding_check': False},
                         '963ad9635ebae05452c40f8de1c806e9b40909ed5d641dd89cd9f1342e0420be'),
    'one_try': ({'tries': 1}, '0f06a2dc022c7d67b9792b4bb121c7a6cfdc9ddb71f41f71cdbc3ee350d0e7a1'),
}


@needs_golden
@pytest.mark.parametrize('name', sorted(LEGACY_TREES_BY_760))
def test_f4_the_legacy_loop_is_byte_identical_to_tsa_760(name):
    import test_bin_picking as tb
    cfg, want = LEGACY_TREES_BY_760[name]
    assert hashlib.sha256(_tree(tb._bin_project(**cfg)).encode()).hexdigest() == want
