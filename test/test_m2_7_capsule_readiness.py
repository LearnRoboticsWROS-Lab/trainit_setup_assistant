"""vNext M2.7 (capsule bin-picking readiness matrix, P1): the TSA facts behind the M4.5 readiness
rows 1 (TSA), 4 (end effector), 8 (Isaac) and 10 (real / FAIRINO), pinned.

The capsule-shaped canonical project: the vision golden, opted in to the end-effector contract
(``eef_fixture.opt_in``: the 3-line SetBool suction contract, the legacy end-effector bridge rows
removed, the merger kept), switched to ``bin_picking``, its grasp move guided by an EXTERNAL-NODE
detector of the real shape (``test_external_node_detector.SEG_NODE``: ``cell_perception`` is a
placeholder for the program's own package) with ``orientation: detected``, and
``bin_picking.holding_check`` equal to ``require_grasp_confirmation`` (R17: one key decides).

What is pinned:

1. It GENERATES: one DetectObject per cycle (keep = the retry budget, its ``fault`` to the gate),
   the candidate's SetWaypointFromDetection inside the retry with the detected orientation, no
   SetReleasePolicy, the external node launched, its package NOT declared (ADR-0003).
2. Isaac grasp confirmation needs the stage's status topic: without it R16 refuses the project;
   with it the holding check is the device's own VerifyGrasp inside the retry.
3. The reference cell's ``mode:=real`` brings up nothing that moves: no real_include, no bridge
   row in real, the hardware-layer adapters on mock and isaac only.
4. A SECOND detector's explicit Detect block, which the loop emits nowhere: REFUSED before any file
   in this canonical project (vNext G0, owner G0 section 6), while nothing else is (the same
   detector with no block, the loop detector's own block, a project with no loop). A LEGACY project
   still drops it silently: KNOWN_DEBT, a strict xfail on the same shape without the contract. The
   day the M4.5 template work emits the block, that xfail XPASSes and fails: flip it then, and lift
   the canonical refusal with it.

Pins, not fixes (owner decision, vNext M2.7: document and defer), except item 4's refusal (vNext
G0). ROS-free: generation is pure.
Every test needs the vision golden (``eef_fixture.needs``: it skips without the demo repository and
FAILS with TRAINIT_REQUIRE_GOLDEN=1); every bundle is generated into the test's own ``tmp_path``.
"""
import ast
import xml.etree.ElementTree as ET

import pytest

import eef_fixture as F
import ws_paths
from test_external_node_detector import SEG_NODE
from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.cli.generate import main as trainit_generate
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.eef_contract import eef_contract_problems
from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.model import (AppType, BundleSpec, DetectorSpec, DetectPoint,
                                           ExternalNodeSpec, VisionBinding, load_project,
                                           save_project)
from trainit_setup_assistant.model.enums import Backend, DetectionMethod

GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
needs_golden = F.needs(GOLDEN_PROJECT, reason='the vision golden project is not present')

PREFIX = 'capsule'
APP, CONFIG = f'{PREFIX}_app', f'{PREFIX}_trainit_config'
GATE = '{eef_fault}'
# A second, IN-HAND detector: the capsule cycle measures the held object over an upward camera
# before it places it. Its own node, in the same placeholder package.
INSPECTION_NODE = dict(package='cell_perception', executable='inspection_node',
                       node_name='inspection_node')
ISAAC_WITH_STATUS = {'adapter': 'trainit_isaac_surface_gripper',
                     'stage_status_topic': '/suction_status'}


def _detector(name: str, node: dict) -> DetectorSpec:
    return DetectorSpec(name=name, method=DetectionMethod.CUSTOM, params={'class_id': 'capsule'},
                        external_node=ExternalNodeSpec(**node))


def _capsule(grasp: bool = False, modes=None, adapters=None, legacy: bool = False):
    """The capsule-shaped canonical bin-picking project (module docstring); ``legacy``: the same
    shape without the end-effector contract (no opt-in), item 4's legacy pin."""
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix(PREFIX)
    if modes is not None:
        p.deployment.modes = [Backend(m) for m in modes]
        p.deployment.default_mode = Backend(modes[0])
    if not legacy:
        F.opt_in(p, adapters=adapters, grasp=grasp)
    app = p.application
    app.type = AppType.BIN_PICKING
    app.bin_picking.holding_check = grasp                       # R17: one key decides
    p.perception.detectors = [_detector('capsule_yolo', SEG_NODE)]
    pick = app.waypoint_by_name('pick')
    pick.vision = VisionBinding(detector='capsule_yolo', dz=pick.vision.dz, orientation='detected')
    # the golden's Detect block samples the colour-mask detector this project replaces; the loop
    # owns its own detector's sampling instant (once per cycle, before the retry)
    app.detections = []
    return p


def _root(xml: str):
    return ET.fromstring(xml[xml.index('<root'):])


def _generated_tree(project, out):
    Orchestrator().generate(project, out)
    return (out / APP / 'bt_trees' / 'bin_picking.xml').read_text()


def _launched(launch_file) -> list:
    """(package, executable) of every ``Node(...)`` in a generated launch file, read with ast."""
    out = []
    for node in ast.walk(ast.parse(launch_file.read_text())):
        if isinstance(node, ast.Call) and getattr(node.func, 'id', None) == 'Node':
            kw = {k.arg: k.value for k in node.keywords if k.arg}
            out.append(tuple(kw[k].value if isinstance(kw.get(k), ast.Constant) else None
                             for k in ('package', 'executable')))
    return out


def _constants(launch_file, *names) -> dict:
    """The module-level literal assignments ``names`` of a generated launch file, never run."""
    out = {}
    for node in ast.parse(launch_file.read_text()).body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id in names:
                out[target.id] = ast.literal_eval(node.value)
    assert sorted(out) == sorted(names), (launch_file, sorted(out))
    return out


# vNext G0 (owner G0 section 6): the exact sentence item 4's refusal gives the probe's case B.
DETECT_REFUSAL = (
    'bin_picking: the Detect block for "place_correction" before "place" is not supported in a '
    'canonical project (one with an end-effector contract, ADR-0022). The emptying loop samples '
    'only its own detector "capsule_yolo", once per cycle before the retry, and emits no other '
    'Detect block, so this one would be dropped: no DetectObject for "place_correction", "place" '
    'not updated from the camera, "post_place" not re-derived. Remove the block: a detector that '
    'guides a move then samples once per cycle, before the retry and before anything is held. A '
    'sampling point inside or after the pick is not supported by bin_picking yet (vNext G0; '
    'KNOWN_DEBT).')


def _with_place_correction(p, block: bool = True):
    """A second, in-hand detector guiding ``place`` -- with its explicit Detect block before
    ``place`` (the M2.7 readiness probe's case B) or without one (case B2)."""
    p.perception.detectors.append(_detector('place_correction', INSPECTION_NODE))
    p.application.waypoint_by_name('place').vision = VisionBinding(detector='place_correction')
    if block:
        p.application.detections.append(
            DetectPoint(detector='place_correction', before_waypoint='place'))
    return p


@needs_golden
def test_a_capsule_shaped_canonical_bin_picking_bundle_generates(tmp_path):
    p = _capsule()
    assert eef_contract_problems(p) == []
    root = _root(_generated_tree(p, tmp_path / 'bundle'))
    assert [(el.get('detector'), el.get('keep'), el.get('fault'))
            for el in root.iter('DetectObject')] == [('capsule_yolo', '3', GATE)]
    retry = root.find('.//RetryUntilSuccessful')

    def swfd(scope):
        return [(el.get('waypoint'), el.get('index'), el.get('orientation'))
                for el in scope.iter('SetWaypointFromDetection')]
    assert swfd(retry) == swfd(root) == [('pick', '{candidate}', 'detected')]
    assert root.find('.//SetReleasePolicy') is None             # no stage subscribes to it (D8)
    assert root.find('.//VerifyGrasp') is None                  # command success is the grasp
    out = tmp_path / 'bundle' / APP
    launch_file = out / 'launch' / 'trainit_bt.launch.py'
    assert ('cell_perception', 'seg_pose_node') in _launched(launch_file)
    declared = [(el.text or '').strip()
                for el in ET.parse(out / 'package.xml').getroot().iter('exec_depend')]
    assert 'cell_perception' not in declared                    # a comment names it (ADR-0003)


@needs_golden
def test_isaac_grasp_confirmation_needs_the_stage_status_topic():
    short = _capsule(grasp=True, modes=('mock', 'isaac'),
                     adapters={'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper'})
    problems = eef_contract_problems(short)
    assert len(problems) == 1, problems
    assert all(token in problems[0] for token in ('R16', 'grasp_detection', 'stage_status_topic'))
    long = _capsule(grasp=True, modes=('mock', 'isaac'),
                    adapters={'mock': 'trainit_mock', 'isaac': ISAAC_WITH_STATUS})
    assert eef_contract_problems(long) == []
    retry = _root(get_application(AppType.BIN_PICKING).build_tree_xml(long)).find(
        './/RetryUntilSuccessful')
    assert [(el.get('eef'), el.get('expect'), el.get('fault'))
            for el in retry.iter('VerifyGrasp')] == [('main', 'held', GATE)]


@needs_golden
def test_the_reference_cell_gets_no_real_bring_up(tmp_path):
    p = _capsule()
    Orchestrator().generate(p, tmp_path / 'bundle')
    bring_up = tmp_path / 'bundle' / CONFIG / 'launch' / 'bringup.launch.py'
    c = _constants(bring_up, 'VALID_MODES', 'REAL_INCLUDE', 'BRIDGES')
    assert 'real' in c['VALID_MODES']                           # the cell deploys real ...
    assert c['REAL_INCLUDE'] is None                            # ... and real includes nothing
    assert [b for b in c['BRIDGES'] if 'real' in b['modes']] == []
    assert sorted((b['executable'], tuple(b['modes'])) for b in c['BRIDGES']
                  if b['package'] == 'trainit_hardware_layer') == [
        ('isaac_eef_adapter', ('isaac',)), ('mock_eef_adapter', ('mock',))]


@needs_golden
def test_a_second_detectors_detect_block_is_refused_in_a_canonical_project(tmp_path, capsys):
    """vNext G0 (owner G0 section 6): the block the loop emits nowhere is REFUSED before any file
    instead of dropped in silence -- the same sentence from refusals(), validate() (the list the GUI
    shows), Orchestrator.generate, trainit_generate (exit 1) and the GUI controller. The end-effector
    contract has nothing to say (the refusal is the template's). With the loop detector's own block
    beside it there are two blocks and still one sentence: only a block the loop would drop is
    refused, and the other stays a note."""
    p = _with_place_correction(_capsule())
    assert eef_contract_problems(p) == []
    app = get_application(AppType.BIN_PICKING)
    assert app.refusals(p) == [DETECT_REFUSAL]
    assert DETECT_REFUSAL in app.validate(p)
    out = tmp_path / 'bundle'
    with pytest.raises(ValueError) as refused:
        Orchestrator().generate(p, out)
    assert str(refused.value) == DETECT_REFUSAL
    assert not out.exists()                                     # refused before any file
    path = save_project(p, tmp_path / 'project.yaml')
    assert trainit_generate([str(path), '-o', str(tmp_path / 'cli')]) == 1
    assert f'error: generation failed: {DETECT_REFUSAL}' in capsys.readouterr().err
    assert not (tmp_path / 'cli').exists()
    ctrl = AssistantController()
    ctrl.open_project(path)
    assert DETECT_REFUSAL in ctrl.validate()
    with pytest.raises(ValueError):
        ctrl.generate(tmp_path / 'gui')
    assert not (tmp_path / 'gui').exists()
    p.application.detections.insert(0, DetectPoint(detector='capsule_yolo',
                                                    before_waypoint='pick'))
    assert app.refusals(p) == [DETECT_REFUSAL]                  # two blocks, one sentence
    assert any(m.startswith('note: the Detect block for "capsule_yolo" is ignored')
               for m in app.validate(p))


@needs_golden
def test_only_a_block_the_loop_would_drop_is_refused():
    """What item 4's refusal leaves alone, in memory (no file is written). The same detector with
    NO block samples once per cycle, before the retry, its fault to the gate -- the refusal's
    remedy, the readiness probe's case B2; the loop detector's own block is only noted (the loop
    samples that detector anyway); with no loop detector the tree is the linear vision sequence,
    which emits the block at its anchor."""
    app = get_application(AppType.BIN_PICKING)
    remedy = _with_place_correction(_capsule(), block=False)
    assert app.refusals(remedy) == []
    cycle = _root(app.build_tree_xml(remedy)).find('.//Repeat/Fallback/Sequence')
    tags = [el.tag for el in cycle]
    assert [(el.get('detector'), el.get('fault')) for el in cycle.findall('DetectObject')] == [
        ('capsule_yolo', GATE), ('place_correction', GATE)]
    assert (max(i for i, tag in enumerate(tags) if tag == 'DetectObject')
            < tags.index('RetryUntilSuccessful'))
    own = _capsule()
    own.application.detections = [DetectPoint(detector='capsule_yolo', before_waypoint='pick')]
    assert app.refusals(own) == []
    assert any(m.startswith('note: the Detect block for "capsule_yolo" is ignored')
               for m in app.validate(own))
    no_loop = _with_place_correction(_capsule())
    no_loop.application.waypoint_by_name('pick').vision = None
    assert app._loop_detector(no_loop) is None
    assert app.refusals(no_loop) == []
    assert 'detector="place_correction"' in app.build_tree_xml(no_loop)


@pytest.mark.xfail(strict=True, raises=AssertionError,
                   reason='KNOWN_DEBT (vNext M2.7, narrowed to legacy in vNext G0): a LEGACY '
                          "bin_picking project drops a second detector's Detect block silently "
                          '(BinPicking._move_block never calls _before_move; _detect_lines skips '
                          'every detector that has a block); a canonical one is refused (item 4). '
                          'The M4.5 template work emits it at its anchor: then flip this test and '
                          'lift the canonical refusal.')
@needs_golden
def test_a_legacy_projects_second_detectors_detect_block_is_emitted_by_bin_picking(tmp_path):
    """The M2.7 pin, narrowed in vNext G0 to the same shape WITHOUT the end-effector contract
    (owner G0 section 6: legacy behaviour unchanged). Today the bundle generates with no refusal
    and no warning naming the drop: no DetectObject and no SetWaypointFromDetection for
    ``place_correction``, so ``place`` runs on its captured pose and the step-relative
    ``post_place`` loses its derivation too, while the detector's node is still launched. Only the
    one assertion below may fail (``raises=AssertionError``): a refusal -- the canonical rule
    reaching a legacy project -- or a missing golden under TRAINIT_REQUIRE_GOLDEN=1 fails this test
    instead of xfailing it."""
    p = _with_place_correction(_capsule(legacy=True))
    assert 'detector="place_correction"' in _generated_tree(p, tmp_path / 'bundle')
