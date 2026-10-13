"""The GUI guard (vNext M2.6, ADR-0022 D6 "The GUI").

TSA's GUI offers two actuations (trigger, joint_position) and its Step-2 base-config ingest
reclassifies the kind -- either would silently turn a canonical gripper back into a legacy one.
So ``AssistantController``'s gripper setters and the Step-2 ingest leave ``kind``, ``actuation``,
``controller_*`` and ``contract`` untouched when ``gripper.contract`` is present: a Step-7 save and
a Step-2 re-ingest keep the canonical gripper exactly as it was. A LEGACY gripper is edited as
before.

Step 7's block editor rebuilds the tool actions from blocks when the page is entered (review
TSA-2): a canonical project's ``actuate`` action and the ``site`` of a grasp or a release must
come back exactly as they were -- ``actuate`` has its own (read-only) block, and the site rides the
block dict -- instead of folding into a grasp. Proven on SYNTHETIC canonical projects
(eef_fixture), headless (``QT_QPA_PLATFORM=offscreen``), with no golden.

ROS-free. The ingest tests read the golden's hand-made base config from the workspace: they skip
without it, and fail with TRAINIT_REQUIRE_GOLDEN=1.
"""
import json
import os

import pytest
import yaml

import eef_fixture as F
import ws_paths
from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.model import BundleSpec, PolicyStep, ToolAction, load_project
from trainit_setup_assistant.model.enums import (EndEffectorActuation, GripperKind,
                                                 ToolActionKind)

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
BASE = ws_paths.ws_src('fr3wml_digital_twin', 'fr3wml_suction_camera_moveit_config')
needs_golden = F.needs(GOLDEN_PROJECT, BASE,
                       reason='the golden project and its base config are not present')
EEF_TAGS = ('CloseGripper', 'OpenGripper', 'VerifyGrasp', 'CommandActuator')


def _canonical():
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix('cell')
    return F.opt_in(p)


def _gripper_block(path) -> dict:
    return yaml.safe_load(path.read_text())['robot']['gripper']


@needs_golden
def test_step_7_setters_leave_a_canonical_gripper_unchanged(tmp_path):
    p = _canonical()
    ctrl = AssistantController(p)
    before = _gripper_block(ctrl.save(tmp_path / 'before.yaml'))
    ctrl.set_gripper_actuation(actuation='trigger')
    ctrl.set_gripper_actuation(actuation='joint_position')
    ctrl.set_gripper_controller(controller_name='other_controller', controller_type='Custom',
                                action_ns='other_ns')
    g = p.robot.gripper
    assert g.actuation is EndEffectorActuation.CONTRACT
    assert (g.controller_name, g.controller_type, g.action_ns) == (
        'suctioncup_controller', 'GripperCommand', 'gripper_command')
    after = ctrl.save(tmp_path / 'after.yaml')
    assert _gripper_block(after) == before
    again = load_project(after)
    assert again.robot.gripper.contract == p.robot.gripper.contract
    assert again.robot.gripper.actuation is EndEffectorActuation.CONTRACT
    assert again.schema_version == 3


@needs_golden
def test_step_2_ingest_leaves_a_canonical_gripper_unchanged(tmp_path):
    """The re-ingest of the golden's own base config: it would reclassify this suction (a
    GripperCommand controller on a real joint) as PARALLEL + joint_position; it must not."""
    p = _canonical()
    ctrl = AssistantController(p)
    before = _gripper_block(ctrl.save(tmp_path / 'before.yaml'))
    ctrl.load_base_moveit_config('fr3wml_suction_camera_moveit_config', str(BASE))
    g = p.robot.gripper
    assert g.kind is GripperKind.SUCTION
    assert g.actuation is EndEffectorActuation.CONTRACT
    assert g.contract is not None
    assert _gripper_block(ctrl.save(tmp_path / 'after.yaml')) == before


@needs_golden
def test_a_legacy_gripper_is_still_edited_and_reclassified(tmp_path):
    """The guard is canonical-only: the legacy setters and ingest behave as before."""
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix('cell')
    ctrl = AssistantController(p)
    ctrl.set_gripper_controller(controller_name='other_controller')
    assert p.robot.gripper.controller_name == 'other_controller'
    ctrl.load_base_moveit_config('fr3wml_suction_camera_moveit_config', str(BASE))
    assert p.robot.gripper.kind is GripperKind.PARALLEL               # the legacy rule
    assert p.robot.gripper.actuation is EndEffectorActuation.JOINT_POSITION
    assert 'contract' not in json.loads(p.robot.gripper.json()) or \
        json.loads(p.robot.gripper.json())['contract'] is None


# --- Step 7, the block editor: canonical tool actions round-trip (review TSA-2) ----------------

@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _actions(project) -> dict:
    """The tool actions of each waypoint, in their declared order: what a tree renders (the fold
    lists them waypoint by waypoint in sequence order, as it always has)."""
    out: dict = {}
    for a in project.application.tool_actions:
        out.setdefault(a.at_waypoint, []).append((a.kind.value, a.payload_ref, a.actuator,
                                                  a.target, a.site))
    return out


def _eef_nodes(project) -> list:
    tree = get_application(project.application.type).build_tree_xml(project)
    return [line.strip() for line in tree.splitlines()
            if line.strip().startswith(tuple(f'<{tag} ' for tag in EEF_TAGS))]


def _jaw_with_actuate(tmp):
    """The GripperCommand jaw with a CommandActuator to a named target (F3's case)."""
    contract = F.jaw_contract()
    contract['actuator']['targets'] = {'open': 0.0, 'closed': 0.7, 'half': 0.35}
    p = F.jaw_simple(tmp, contract=contract)
    p.application.tool_actions.append(ToolAction(at_waypoint='flip', kind=ToolActionKind.ACTUATE,
                                                 actuator='gripper', target='half'))
    return p


@pytest.mark.parametrize('build', [lambda t: F.composite(t), _jaw_with_actuate],
                         ids=['composite_sites_and_actuators', 'jaw_actuate'])
def test_step_7_keeps_a_canonical_projects_tool_actions(qapp, tmp_path, build):
    """Entering Step 7 (initializePage folds the blocks back into the model) and leaving it
    (validatePage) keep every tool action -- kind, actuator, target, site, payload -- and the
    end-effector nodes of the tree exactly as they were; selecting a block and pressing Apply
    with no edit too. The composite: 7 actuate actions and 4 site-addressed grasps/releases."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    p = build(tmp_path)
    actions, nodes = _actions(p), _eef_nodes(p)
    assert any(kind == 'actuate' for acts in actions.values() for kind, *_r in acts)
    ctrl = AssistantController(p)
    wizard = SetupWizard(ctrl)                 # alive: it owns the page's widgets
    page = wizard.blocks_page
    page.initializePage()
    kinds = [b['kind'] for b in page.blocks]
    assert kinds.count('actuate') == sum(1 for acts in actions.values() for k, *_r in acts
                                         if k == 'actuate')
    assert _actions(ctrl.project) == actions
    assert page.validatePage()
    assert _actions(ctrl.project) == actions and _eef_nodes(ctrl.project) == nodes
    for row, block in enumerate(page.blocks):           # an untouched Apply is an identity
        if block['kind'] in ('actuate', 'gripper'):
            page.seq.setCurrentRow(row)
            page.apply_inspector()
    assert page.validatePage()
    assert _actions(ctrl.project) == actions and _eef_nodes(ctrl.project) == nodes
    saved = load_project(ctrl.save(tmp_path / 'saved.yaml'))
    assert _actions(saved) == actions and _eef_nodes(saved) == nodes


def test_step_7_keeps_a_legacy_projects_blocks_as_before(qapp):
    """A legacy project's gripper blocks carry no site and fold into exactly the actions they
    were (the round trip the GUI tests prove for every block kind, here for the gripper keys)."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(str(F.DATA.parents[2] / 'examples' / 'fr3wml_project.yaml'))
    before = json.loads(ctrl.project.json())['application']['tool_actions']
    wizard = SetupWizard(ctrl)                 # alive: it owns the page's widgets
    page = wizard.blocks_page
    page.initializePage()
    assert all('site' not in b for b in page.blocks if b['kind'] == 'gripper')
    assert not [b for b in page.blocks if b['kind'] == 'actuate']
    assert page.validatePage()
    assert json.loads(ctrl.project.json())['application']['tool_actions'] == before


def test_step_7_keeps_a_canonical_policy_block_as_a_legacy_one(qapp, tmp_path):
    """A policy step's attached_topic is the unchanged policy runtime's own attached-observation
    input (suction_state_topic, ADR-0022 D10): next to a contract it is neither refused nor
    hidden. The policy block behaves as in a legacy project (the runtime's default topic), and an
    untouched Apply keeps the step generable."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    from trainit_setup_assistant.generator.eef_contract import eef_contract_problems
    card = tmp_path / 'card' / 'policy_card.yaml'
    card.parent.mkdir()
    card.write_text('policy_card:\n  name: synthetic_suction\n  end_effector: suction\n'
                    '  end_state: {tcp_pose_base: [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}\n')
    p = F.suction_simple(tmp_path, ns='')
    p.application.policies = [PolicyStep(name='grasp', before_waypoint='pick', mode='pure',
                                         card=str(card))]
    assert eef_contract_problems(p) == []
    ctrl = AssistantController(p)
    wizard = SetupWizard(ctrl)                 # alive: it owns the page's widgets
    page = wizard.blocks_page
    page.initializePage()
    row = next(i for i, b in enumerate(page.blocks) if b['kind'] == 'policy')
    page.seq.setCurrentRow(row)
    assert page.p_attach_topic.text() == '/suction/attached' and page.p_attach_topic.isEnabled()
    page.apply_inspector()
    assert page.validatePage()
    assert ctrl.project.application.policies[0].attached_topic == '/suction/attached'
    assert eef_contract_problems(ctrl.project) == []
    page.add_block('policy')
    assert page.blocks[-1]['attached_topic'] == '/suction/attached'
