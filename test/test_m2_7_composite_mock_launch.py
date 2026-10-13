"""T13 + T14 (vNext M2.7, owner Q1 option A and section 13): the composite MOCK bundles, generated,
built and brought up for real -- and judged check by check.

    generate the three section-13 cells (a temp dir; eef_fixture.composite_mock: cm_root at the
      root, robot_a in /r1, robot_b in /r2, IDENTICAL end-effector ids)
      -> colcon build in a temp workspace over the sourced one
      -> phase A: the three bring-ups AT ONCE (mode:=mock rviz:=false)
           K1 the tool joints reach <ns>/joint_states (the composite mock row's remap)
           K2 TF reaches both site TCPs, and they follow the axes
           K3 MoveIt's current state holds the tool values; an arm motion logs no "Missing"
           K4 no namespace collision (names, identities, the node and topic rule)
           K5 five distinguishable actuator states; each command moves only its own actuator
      -> phase B: the root's generated tree, one fresh bring-up per run
           K6 nominal: exit 0, the six VerifyGrasp SUCCESS lines; a FAULT on the lock: the FIRST
              VerifyGrasp fails at tool level, exit 1; site_b not held: a VERDICT at site_b
      -> phase C: /r1 and /r2 trees at once -- only on a runner that resolves MoveIt inside its
         namespace (trainit_motion_runtime >= 2.1.0); below it the phase is SKIPPED, never PASS
      -> every launch ends on its SIGINT, nothing survives it, the ROS domain is empty after

The phases, the checks and the report are those of ``composite_launch.py`` (the helper, not
collected); this module runs it once per session and gives each check its own test.

OPT-IN: it builds and launches for minutes, so it runs only with ``TRAINIT_LAUNCH_TEST=1`` and a
sourced workspace (ROS 2 Humble + this workspace's install). Every launch runs on ROS domain 83
(sequential, never together with domain 84), ROS_LOCALHOST_ONLY=1, with TMPDIR and ROS_LOG_DIR
inside the temp workspace. Everything lives in a pytest temp dir; nothing is written into the
workspace. This is test-time generation -- not a bundle for anyone to ship (MANUAL TSA GENERATION
CHECKPOINT stands). With ``TRAINIT_M27_LAUNCH_EVIDENCE=<dir>`` the report, the bundle identities
and every launch log are copied to ``<dir>/composite_mock/`` before the temp dir goes.
"""
import os
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(os.environ.get('TRAINIT_LAUNCH_TEST') != '1',
                                reason='opt-in: set TRAINIT_LAUNCH_TEST=1 (builds and launches)')

EVIDENCE = 'TRAINIT_M27_LAUNCH_EVIDENCE'


@pytest.fixture(scope='module')
def composite_run(tmp_path_factory):
    import composite_launch as CL
    ws = tmp_path_factory.mktemp('t13_ws')
    bundles = CL.generate(ws)
    summary = CL.build(ws)
    run = CL.CompositeLaunch(ws)
    try:
        run.run('ABC')
    finally:
        if os.environ.get(EVIDENCE):
            run.save(Path(os.environ[EVIDENCE]) / 'composite_mock')
    return dict(run=run, bundles=bundles, summary=summary,
                checks={cid: (status, detail) for cid, status, detail in run.checks})


def _verdict(composite_run, cid: str) -> None:
    checks = composite_run['checks']
    assert cid in checks, f'no check {cid!r}: {sorted(checks)}'
    status, detail = checks[cid]
    assert status == 'PASS', f'{cid}: {status} -- {detail}\n' + composite_run['run'].report()


def test_t13_the_three_composite_mock_bundles_generate_and_build(composite_run):
    """BEFORE (TSA 7.7.0): every cell refused, R22 'a composite BUNDLE is refused in M2.6'."""
    assert len(composite_run['bundles']) == 3, composite_run['bundles']
    assert '9 packages finished' in composite_run['summary'], composite_run['summary']


@pytest.mark.parametrize('ns', ['/', '/r1', '/r2'])
def test_t14_k1_the_tool_joints_reach_the_robots_joint_states(composite_run, ns):
    _verdict(composite_run, f'K1 joint states propagate {ns}')


@pytest.mark.parametrize('ns', ['/', '/r1', '/r2'])
def test_t14_k2_tf_reaches_both_site_tcps(composite_run, ns):
    _verdict(composite_run, f'K2 TF reaches the tool TCPs {ns}')


@pytest.mark.parametrize('ns', ['/', '/r1'])
def test_t14_k3_moveit_sees_the_tool_joints_and_misses_none(composite_run, ns):
    _verdict(composite_run, f'K3 MoveIt sees the right joints {ns}')


def test_t13_k4_no_namespace_collision(composite_run):
    _verdict(composite_run, 'K4 no namespace collision')


@pytest.mark.parametrize('ns', ['/', '/r1'])
def test_t13_k5_actuator_states_stay_distinguishable(composite_run, ns):
    _verdict(composite_run, f'K5 actuator states distinguishable {ns}')


@pytest.mark.parametrize('cid', [
    'K6 nominal: verification is tool-level, the tree succeeds',
    'K6 fault on the lock: the FIRST VerifyGrasp fails at tool level',
    'K6 site_b not held: a VERDICT at site_b, every command before it SUCCESS',
], ids=['nominal', 'fault_on_the_lock', 'site_b_not_held'])
def test_t13_k6_grasp_verification_is_tool_level(composite_run, cid):
    _verdict(composite_run, cid)


def test_t13_phase_c_two_namespaced_trees_at_once(composite_run):
    """Needs a runner that resolves MoveIt in its namespace: SKIPPED below that, never PASS."""
    checks = composite_run['checks']
    skipped = checks.get('C two namespaced trees at once')
    if skipped is not None:
        assert skipped[0] == 'SKIPPED', skipped
        pytest.skip(skipped[1])
    for ns in ('/r1', '/r2'):
        _verdict(composite_run, f'C {ns} tree with /r1 and /r2 at once')


def test_t13_every_launch_shuts_down_cleanly_and_the_domain_ends_empty(composite_run):
    checks = composite_run['checks']
    for cid in [c for c in checks if c.startswith(('shutdown ', 'guard '))]:
        _verdict(composite_run, cid)
    assert 'guard before the launches' in checks and 'guard after the launches' in checks


def test_t13_no_check_failed(composite_run):
    """Every check the helper ran, the ones named above and any it gains later."""
    failed = [(cid, detail) for cid, (status, detail) in composite_run['checks'].items()
              if status == 'FAIL']
    assert not failed, failed
