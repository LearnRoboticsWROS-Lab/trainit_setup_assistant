"""vNext M2.7 D3, scoped at vNext G0: a CANONICAL project's generated move never relies on a blank
planner (owner decision D3; owner G0 section 8: canonical projects only, legacy unchanged).

In a canonical project (one with an end-effector contract, ADR-0022) TSA resolves a blank planner
to the one TMR's MoveWaypoint RUNS -- ``ptp`` -> ``pilz`` (TMR ``bt_nodes.cpp`` MoveWaypoint::tick:
a blank ptp is PlannerMode::PILZ), ``free`` or no segment -> the project's ``global_planner_mode``,
the token every generated launch hands the runner as ``planner_mode`` -- and WRITES it into
bt_params; ``lin``/``circ`` read no planner, so a blank one stays blank. D-016 and the blocks page's
switch to free/OMPL judge the same effective planner. A LEGACY project keeps the reading it had
before M2.7 at all three places: nothing is written for a blank planner, and D-016 and the switch
read it as the global mode (KNOWN_DEBT: TMR runs a blank ptp as Pilz PTP all the same). TMR is
untouched: its own pins of the blank semantics (M2.5 WP-FREE-03, T5-01..04, T3-01/02) stay as they
are.

Fast and ROS-free; the GUI tests need Qt (offscreen). The canonical cases run on eef_fixture's
synthetic suction cell (golden-free); the D-016 table reads the golden bundle (it skips without it,
and fails instead with TRAINIT_REQUIRE_GOLDEN=1).
"""
import os
import tempfile
from pathlib import Path

import pytest
import yaml

import eef_fixture
import identity_cases
import ws_paths
from trainit_setup_assistant.applications import build_bt_params_context, get_application
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import (AppType, BundleSpec, DetectorSpec, PerceptionSpec,
                                           VisionBinding, effective_planner, load_project)
from trainit_setup_assistant.model.enums import ExecutionMode, MotionType, PlannerId

EXAMPLE = str(identity_cases.EXAMPLE)
GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
M, P = MotionType, PlannerId


# --- 1. the table ---------------------------------------------------------------------------------

@pytest.mark.parametrize('motion, planner, mode, want', [
    (M.PTP, None, P.OMPL, P.PILZ), (M.PTP, None, P.PILZ, P.PILZ),
    (M.PTP, None, P.OMPL_CHOMP, P.PILZ),
    (M.FREE, None, P.OMPL, P.OMPL), (M.FREE, None, P.PILZ, P.PILZ),
    (M.FREE, None, P.OMPL_CHOMP, P.OMPL_CHOMP),
    (None, None, P.OMPL, P.OMPL),                       # no segment: TMR's motion default is free
    (M.LIN, None, P.OMPL, None), (M.CIRC, None, P.PILZ, None),
    (M.PTP, P.OMPL, P.PILZ, P.OMPL), (M.FREE, P.PILZ, P.OMPL, P.PILZ),
    (M.LIN, P.PILZ, P.OMPL, P.PILZ),                    # an explicit one stays, inert (KNOWN_DEBT)
])
def test_the_effective_planner(motion, planner, mode, want):
    assert effective_planner(motion, planner, mode) is want


# --- 2. generation: the planner TMR runs in a canonical project, a chosen one only in a legacy one

def _example(mode, blank=True):
    """The LEGACY example project (no contract): ptp, free and lin moves."""
    p = load_project(EXAMPLE)
    p.bundle = BundleSpec.from_prefix('cell')
    p.application.global_planner_mode = mode
    if blank:
        for seg in p.application.segments:
            seg.planner = None
    return p


def _canonical(root, mode, explicit=False):
    """A CANONICAL cell: eef_fixture's synthetic suction cell at the root, its blank moves made
    home ptp, pick free, flip lin, place free (every branch of the rule) under global ``mode``;
    ``explicit``: each planner set to its effective value."""
    p = eef_fixture.suction_simple(root, ns='')
    p.application.global_planner_mode = mode
    p.application.segment_for('home').motion = MotionType.PTP
    p.application.segment_for('flip').motion = MotionType.LIN
    if explicit:
        for seg in p.application.segments:
            seg.planner = effective_planner(seg.motion, None, mode)
    return p


def _generate(p):
    tmp = tempfile.TemporaryDirectory()
    Orchestrator().generate(p, tmp.name)
    return tmp, Path(tmp.name)


def _files(root: Path) -> dict:
    return {f.relative_to(root).as_posix(): f.read_bytes() for f in sorted(root.rglob('*'))
            if f.is_file()}


def _blocks(root: Path, app: str = 'cell_app') -> dict:
    params = yaml.safe_load((root / app / 'config' / 'bt_params.yaml').read_text())
    return {k: v for k, v in params['/**']['ros__parameters']['task_parameters'].items()
            if isinstance(v, dict)}


@pytest.mark.parametrize('mode', list(PlannerId))
def test_a_canonical_blank_planner_is_written_as_the_one_tmr_runs(tmp_path, mode):
    p = _canonical(tmp_path, mode)
    tmp, out = _generate(p)
    with tmp:
        blocks = _blocks(out, p.bundle.app_package)
    assert {b['motion'] for b in blocks.values()} >= {'ptp', 'free', 'lin'}   # every branch
    for name, block in blocks.items():
        if block['motion'] == 'ptp':
            assert block['planner'] == 'pilz', name
        elif block['motion'] == 'free':
            assert block['planner'] == mode.value, name
        else:                                           # lin / circ: no planner is read
            assert 'planner' not in block, name


@pytest.mark.parametrize('mode', list(PlannerId))
def test_a_canonical_blank_planner_generates_exactly_the_explicit_effective_one(tmp_path, mode):
    """The strongest form of 'no second meaning': this canonical cell's bundle with blank planners
    is BYTE-IDENTICAL to its bundle with each planner set to its effective value (the cell has no
    vision-bound move, so no D-016 warning can differ either)."""
    a, out_a = _generate(_canonical(tmp_path, mode))
    b, out_b = _generate(_canonical(tmp_path, mode, explicit=True))
    with a, b:
        assert _files(out_a) == _files(out_b)


def test_a_canonical_waypoint_without_a_segment_gets_the_global_planner(tmp_path):
    p = _canonical(tmp_path, PlannerId.OMPL)
    p.application.segments = [s for s in p.application.segments if s.to_waypoint != 'pick']
    tmp, out = _generate(p)
    with tmp:
        block = _blocks(out, p.bundle.app_package)['pick']
    assert 'motion' not in block and block['planner'] == 'ompl'


def test_a_canonical_streamed_move_carries_the_same_block_as_a_planned_one(tmp_path):
    planned, streamed = _canonical(tmp_path, PlannerId.OMPL), _canonical(tmp_path, PlannerId.OMPL)
    streamed.application.segment_for('pick').execution = ExecutionMode.SERVO
    pick = [next(b for b in build_bt_params_context(p)['waypoints'] if b['name'] == 'pick')
            for p in (planned, streamed)]
    assert pick[0] == pick[1] and pick[0]['planner'] == 'ompl'


@pytest.mark.parametrize('mode', list(PlannerId))
def test_a_legacy_blank_planner_is_not_written(mode):
    """Owner G0 section 8: a legacy project generates what it generated before M2.7 -- no planner
    for a blank move (TMR's own blank-planner rules apply to it, as they always did)."""
    tmp, out = _generate(_example(mode))
    with tmp:
        blocks = _blocks(out)
    assert {b['motion'] for b in blocks.values()} >= {'ptp', 'free', 'lin'}   # every branch
    assert [name for name, block in blocks.items() if 'planner' in block] == []


def test_a_legacy_project_writes_its_chosen_planners_as_they_are():
    p = _example(PlannerId.OMPL, blank=False)      # the example's own: ptp/free pilz, lin blank
    tmp, out = _generate(p)
    with tmp:
        blocks = _blocks(out)
    for seg in p.application.segments:
        want = seg.planner.value if seg.planner else None
        assert blocks[seg.to_waypoint].get('planner') == want, seg.to_waypoint


def test_a_legacy_waypoint_without_a_segment_gets_no_planner():
    p = _example(PlannerId.OMPL, blank=False)
    p.application.segments = [s for s in p.application.segments if s.to_waypoint != 'pre_pick']
    tmp, out = _generate(p)
    with tmp:
        block = _blocks(out)['pre_pick']
    assert 'motion' not in block and 'planner' not in block


@pytest.mark.parametrize('mode', list(PlannerId))
def test_the_runner_receives_the_global_planner_token(mode):
    """The free-move half of the rule: bt_params' planner_mode and the trainit_bt launch
    argument's default (which overrides it: node_params come after bt_params) are both the global
    token."""
    tmp, out = _generate(_example(mode))
    with tmp:
        params = yaml.safe_load((out / 'cell_app' / 'config' / 'bt_params.yaml').read_text())
        launch = (out / 'cell_app' / 'launch' / 'trainit_bt.launch.py').read_text()
    assert params['/**']['ros__parameters']['planner_mode'] == mode.value
    assert f'DeclareLaunchArgument("planner_mode", default_value="{mode.value}",' in launch
    assert 'parameters=[params, bt_params, node_params' in launch


# --- 3. D-016: the effective planner in a canonical project, the old reading in a legacy one -----

_OLD = ('waypoint "pick" is vision-driven but reached with ptp/pilz — a blind joint '
        'interpolation that cannot avoid obstacles (it hit the camera on the reference cell). '
        'Use motion "free" with OMPL for the travel into a vision goal.')
_NEW = ('waypoint "pick" is vision-driven but reached with ptp/pilz (no planner set: a blank '
        'ptp planner runs Pilz PTP, whatever the global planner) — a blind joint interpolation '
        'that cannot avoid obstacles (it hit the camera on the reference cell). Use motion '
        '"free" with OMPL for the travel into a vision goal.')


def _golden(canonical):
    p = load_project(str(GOLDEN_PROJECT))
    if canonical:                                         # the golden, opted in (ADR-0022 D6)
        p.bundle = BundleSpec.from_prefix('cell')
        eef_fixture.opt_in(p)
    return p


@pytest.mark.parametrize('canonical, planner, mode, want', [
    (True, None, P.OMPL, _NEW),      # the case M2.7 closes: TMR runs Pilz PTP, TSA said nothing
    (True, None, P.OMPL_CHOMP, _NEW),
    (True, None, P.PILZ, _OLD),      # flagged before: its message is unchanged
    (True, P.PILZ, P.OMPL, _OLD),
    (True, P.OMPL, P.PILZ, None),    # an explicit collision-aware planner: never flagged
    (False, None, P.OMPL, None),     # legacy (owner G0 section 8): blank = the global mode, as
    (False, None, P.OMPL_CHOMP, None),   # before M2.7 (KNOWN_DEBT: TMR runs Pilz PTP there)
    (False, None, P.PILZ, _OLD),
    (False, P.PILZ, P.OMPL, _OLD),
    (False, P.OMPL, P.PILZ, None),
])
@eef_fixture.needs(GOLDEN_PROJECT, reason='the golden bundle (its camera-bound move)')
def test_d016_judges_the_planner_tmr_runs(canonical, planner, mode, want):
    p = _golden(canonical)
    p.application.global_planner_mode = mode
    seg = p.application.segment_for('pick')               # the golden's camera-bound move
    seg.motion, seg.planner = MotionType.PTP, planner
    got = [x for x in get_application(p.application.type).validate(p) if 'vision-driven' in x]
    assert got == ([want] if want else [])


def test_d016_flags_a_blank_ptp_vision_move_of_a_synthetic_canonical_cell(tmp_path):
    """The canonical half without the golden (the legacy half: test_vision_application.py's
    test_ptp_with_inherited_ompl_is_not_flagged): eef_fixture's suction cell made vision guided,
    'pick' camera-bound and reached ptp with a blank planner under global ompl."""
    p = eef_fixture.suction_simple(tmp_path, ns='')
    p.application.type = AppType.VISION_GUIDED_MOTION
    p.application.global_planner_mode = PlannerId.OMPL
    p.perception = PerceptionSpec(detectors=[DetectorSpec(name='cube',
                                                          params={'class_id': 'cube'})])
    p.application.waypoint_by_name('pick').vision = VisionBinding(detector='cube')
    p.application.segment_for('pick').motion = MotionType.PTP
    got = [x for x in get_application(p.application.type).validate(p) if 'vision-driven' in x]
    assert got == [_NEW]


# --- 4. the GUI auto-switch: the same scope -------------------------------------------------------

@pytest.fixture(scope='module')
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    yield QApplication.instance() or QApplication([])


@pytest.mark.parametrize('canonical, mode, planner, switched', [
    (True, 'ompl', None, True),      # canonical: a blank ptp is Pilz PTP under any global mode
    (True, 'pilz', None, True), (True, 'ompl', 'pilz', True), (True, 'pilz', 'pilz', True),
    (True, 'ompl', 'ompl', False), (True, 'pilz', 'ompl', False),
    (False, 'ompl', None, False),    # legacy (owner G0 section 8): blank = the global mode
    (False, 'pilz', None, True), (False, 'ompl', 'pilz', True), (False, 'pilz', 'pilz', True),
    (False, 'ompl', 'ompl', False), (False, 'pilz', 'ompl', False),
])
def test_attaching_a_detector_to_a_pilz_ptp_move_switches_it_to_free_ompl(
        qapp, tmp_path, canonical, mode, planner, switched):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    if canonical:                       # the GUI has no contract editor: a built project, opened
        ctrl = AssistantController(eef_fixture.suction_simple(tmp_path, ns=''))
    else:
        ctrl = AssistantController()
        ctrl.open_project(EXAMPLE)
    ctrl.project.application.global_planner_mode = PlannerId(mode)
    ctrl.upsert_detector('cube', params={'class_id': 'cube'})
    ctrl.add_move('approach', named='home', motion='ptp', planner=planner)
    wiz = SetupWizard(ctrl)                               # keep it: its pages die with it
    bp = wiz.blocks_page
    bp.initializePage()
    row = next(i for i, b in enumerate(bp.blocks)
               if b['kind'] == 'move' and b['name'] == 'approach')
    bp.seq.setCurrentRow(row)
    bp.m_detector.setCurrentText('cube')
    bp.apply_inspector()
    seg = ctrl.project.application.segment_for('approach')
    assert (seg.motion, seg.planner) == ((MotionType.FREE, PlannerId.OMPL) if switched else
                                         (MotionType.PTP, PlannerId(planner) if planner else None))
    assert ('motion set to free/OMPL' in bp.status.text()) is switched
