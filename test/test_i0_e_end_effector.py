"""The end effector in Step 7 (vNext I0, I0_DESIGN.md §34, owner decision 2026-10-26).

D-E1 -- a plain open / close of a grasp site where the robot is (``tool: {close|open: {eef?, site?}}``,
ADR-0024 amendment I0): the grasp's / release's actuation without its motion, the same nodes and
verification policy; nothing moves, so the site's TCP need not be the tip link.
"""

import copy
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import ProgramRefusals, plan_program
from trainit_setup_assistant.model import CanonicalProject

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'
VALIDATION_C = PKG / 'test' / 'data' / 'm3_6' / 'validation_c.yaml'


@pytest.fixture(autouse=True)
def _sample_paths(monkeypatch):
    monkeypatch.chdir(SAMPLE.parent)


def program(data: dict, steps, waypoints=None) -> CanonicalProject:
    """``data`` (a project mapping whose robot has a contract) with a task program of ``steps``."""
    d = copy.deepcopy(data)
    d['schema_version'] = max(int(d.get('schema_version', 4)), 4)
    d['application'] = {
        'type': 'task_program',
        'motion_profiles': {'transit': {'speed': 50, 'acceleration': 50}},
        'waypoints': waypoints or [{'name': 'home_wp', 'type': 'joint',
                                    'joints': [0.0, -1.5708, 1.5708, -1.5708, -1.5708, 0.0]}],
        'tasks': [{'name': 'main', 'steps': steps}],
        'main_task': 'main'}
    return CanonicalProject.parse_obj(d)


def tree(project) -> str:
    plan = plan_program(project)
    from trainit_setup_assistant.applications.task_program import iter_plans
    out = []
    for task, steps in plan.tasks.items():
        for sp in iter_plans(steps):
            out += list(sp.nodes or [sp.xml])
    return '\n'.join(out)


def sample() -> dict:
    return yaml.safe_load(SAMPLE.read_text())


def test_close_and_open_where_the_robot_is():
    p = program(sample(), [{'move': {'to': 'home_wp', 'type': 'ptp', 'profile': 'transit'}},
                           {'tool': {'open': {}}},
                           {'tool': {'close': {}}},
                           {'dwell': {'ms': 200}},
                           {'tool': {'open': {}}}])
    xml = tree(p)
    assert xml.count('<CloseGripper eef="main" site="main"/>') == 1
    assert xml.count('<OpenGripper eef="main" site="main"/>') == 2
    assert 'VerifyGrasp' not in xml                      # the sample's contract confirms nothing
    assert 'MoveWaypoint waypoint' in xml and xml.count('MoveWaypoint') == 1   # no motion of its own


def test_close_and_open_follow_the_contracts_verification_policy():
    data = yaml.safe_load(VALIDATION_C.read_text())        # grasp AND release confirmed
    data.pop('application')
    p = program(data, [{'tool': {'close': {}}}, {'tool': {'open': {'site': 'main'}}}])
    xml = tree(p).splitlines()
    assert xml[0] == '<CloseGripper eef="main" site="main"/>'
    assert xml[1].startswith('<VerifyGrasp eef="main" site="main" expect="held"')
    assert '<OpenGripper eef="main" site="main"/>' in xml
    assert xml[-1].startswith('<VerifyGrasp eef="main" site="main" expect="released"')


def test_a_site_need_not_be_the_tip_and_an_unknown_one_is_refused():
    from eef_fixture import composite_mock_contract
    data = sample()
    data['robot']['gripper'] = {'kind': 'composite', 'contract': composite_mock_contract('')}
    p = program(data, [{'tool': {'close': {'site': 'site_a'}}}, {'tool': {'open': {'site': 'site_b'}}}])
    xml = tree(p)
    assert '<CloseGripper eef="composite_tool" site="site_a"/>' in xml
    assert '<OpenGripper eef="composite_tool" site="site_b"/>' in xml     # no R-M3-05: nothing moves
    with pytest.raises(ProgramRefusals, match="name the step's 'site'"):
        plan_program(program(data, [{'tool': {'close': {}}}]))            # two sites: say which
    with pytest.raises(ProgramRefusals, match="has no grasp site 'site_z'"):
        plan_program(program(data, [{'tool': {'open': {'site': 'site_z'}}}]))


def test_the_tool_step_is_exactly_one_form_and_round_trips():
    from trainit_setup_assistant.model.io import project_to_dict
    with pytest.raises(ValueError, match="exactly one of 'actuate', 'verify', 'close' or 'open'"):
        program(sample(), [{'tool': {'close': {}, 'open': {}}}])
    p = program(sample(), [{'tool': {'close': {'site': 'main'}}}])
    assert project_to_dict(p)['application']['tasks'][0]['steps'] == [{'tool': {'close': {'site': 'main'}}}]


def test_no_contract_is_refused_with_the_end_effector_section_named():
    data = sample()
    data['robot']['gripper'].pop('contract')
    with pytest.raises(ProgramRefusals, match='R-M3-13'):
        plan_program(program(data, [{'tool': {'close': {}}}]))


# ---- E1: the contract's model operations (composer/eef_ops.py) ---------------------------------------------

from trainit_setup_assistant.composer import Composer                     # noqa: E402
from trainit_setup_assistant.composer import eef_ops                      # noqa: E402
from trainit_setup_assistant.composer.ops import ComposerError            # noqa: E402
from trainit_setup_assistant.composer.templates import apply_template     # noqa: E402
from trainit_setup_assistant.model.io import project_to_dict              # noqa: E402

LEGACY_ROWS = [
    {'package': 'fr3wml_isaac', 'executable': 'isaac_suction_bridge.py', 'modes': ['mock', 'isaac']},
    {'package': 'fairino_gripper', 'executable': 'joint_state_merger', 'name': 'joint_state_merger_suction',
     'modes': ['mock', 'isaac']},
    {'package': 'fairino_gripper', 'executable': 'gripper_suction_action_server', 'modes': ['mock', 'isaac']}]
SUCTION = {'actuator': {'command_interface': 'set_bool'},
           'verification': {'require_grasp_confirmation': False, 'require_release_confirmation': False},
           'adapters': {'mock': 'trainit_mock'}}


def legacy_composer() -> Composer:
    """A task program on a LEGACY gripper (no contract) with the reference cell's bridge rows."""
    d = sample()
    d['robot']['gripper'].pop('contract')
    d['robot']['gripper']['actuation'] = 'trigger'
    d['schema_version'] = 2                       # a legacy project (3 is the contract's number, R19)
    d.setdefault('deployment', {})['bridges'] = copy.deepcopy(LEGACY_ROWS)
    return Composer(apply_template(CanonicalProject.parse_obj(d), 'simple_motion'))


def test_adopting_a_contract_is_one_undoable_edit_that_drops_the_legacy_bridges():
    c = legacy_composer()
    before = project_to_dict(c.project)
    assert eef_ops.contract_of(c) is None and len(eef_ops.legacy_bridges(c)) == 2
    removed = eef_ops.adopt_contract(c, 'suction', SUCTION)
    assert removed == ['fr3wml_isaac/isaac_suction_bridge.py', 'fairino_gripper/gripper_suction_action_server']
    after = project_to_dict(c.project)
    assert [r['executable'] for r in after['deployment']['bridges']] == ['joint_state_merger']   # the merger stays
    assert after['robot']['gripper']['contract'] == SUCTION and c.project.robot.gripper.is_canonical()
    c.undo()
    assert project_to_dict(c.project) == before


def test_the_contract_edits_and_their_refusals():
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'suction', SUCTION)
    eef_ops.set_verification(c, True, False)
    eef_ops.set_binding(c, 'isaac', {'adapter': 'trainit_isaac_surface_gripper', 'stage_cmd_topic': '/suction_cmd'})
    eef_ops.set_binding(c, 'gazebo', 'none')
    k = eef_ops.contract_of(c)
    assert k['verification'] == {'require_grasp_confirmation': True, 'require_release_confirmation': False}
    assert k['adapters']['isaac']['stage_cmd_topic'] == '/suction_cmd' and k['adapters']['gazebo'] == 'none'
    eef_ops.set_binding(c, 'gazebo', None)
    assert 'gazebo' not in eef_ops.contract_of(c)['adapters']
    with pytest.raises(ComposerError, match='binary'):
        eef_ops.set_target(c, eef_ops.actuator_ids(eef_ops.contract_of(c))[0] or 'vacuum', 'half', 0.5)
    with pytest.raises(ComposerError):                                     # the model refuses a bad token
        eef_ops.set_binding(c, 'mock', 'no_such_adapter')
    assert eef_ops.contract_of(c)['adapters']['mock'] == 'trainit_mock'    # unchanged after the refusal


def test_renames_follow_into_the_steps_and_a_used_element_is_not_deleted():
    from eef_fixture import composite_mock_contract
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'composite', composite_mock_contract(''))
    c.edit(lambda doc: doc['tasks'][0]['steps'].extend([
        {'tool': {'actuate': {'actuator': 'slide_axis', 'target': 'extended'}}},
        {'tool': {'close': {'site': 'site_a'}}},
        {'tool': {'open': {'eef': 'composite_tool', 'site': 'site_a'}}}]))
    eef_ops.rename_actuator(c, 'slide_axis', 'slide')
    eef_ops.rename_target(c, 'slide', 'extended', 'out')
    eef_ops.rename_site(c, 'site_a', 'front')
    eef_ops.set_device_id(c, 'tool')
    steps = c.document()['tasks'][0]['steps'][-3:]
    assert steps == [{'tool': {'actuate': {'actuator': 'slide', 'target': 'out'}}},
                     {'tool': {'close': {'site': 'front'}}},
                     {'tool': {'open': {'eef': 'tool', 'site': 'front'}}}]
    k = eef_ops.contract_of(c)
    assert 'slide' in k['actuators'] and 'out' in k['actuators']['slide']['targets'] and 'front' in k['sites']
    with pytest.raises(ComposerError, match="used by task 'cycle' step"):
        eef_ops.delete_actuator(c, 'slide')
    with pytest.raises(ComposerError, match="grasp site 'front' is used by"):
        eef_ops.delete_site(c, 'front')
    with pytest.raises(ComposerError, match="is used by site 'site_b'"):
        eef_ops.delete_actuator(c, 'binary_jaw')
    eef_ops.delete_actuator(c, 'binary_lock')                               # nobody uses it
    assert 'binary_lock' not in eef_ops.contract_of(c)['actuators']
    with pytest.raises(ComposerError, match='commands the end effector'):
        eef_ops.remove_contract(c)


def test_a_single_actuator_becomes_multi_actuator_with_its_derived_values_written():
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'suction', SUCTION)
    eef_ops.to_long_form(c)
    k = eef_ops.contract_of(c)
    assert 'actuator' not in k and list(k['actuators']) == ['vacuum']
    assert k['actuators']['vacuum']['command_interface'] == 'set_bool'
    assert k['sites'] == {'main': {'tcp': c.project.robot.gripper.eef_parent_link or 'tcp', 'grip': 'vacuum'}}
    plan_program(c.project)                                                 # still a valid program


def test_removing_the_contract_resets_the_legacy_token():
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'suction', SUCTION)
    eef_ops.remove_contract(c)
    g = eef_ops.gripper_of(c)
    assert 'contract' not in g and g['actuation'] == 'trigger'


# ---- E2: the suggestion from what TSA read (composer/eef_suggest.py) ---------------------------------------

from trainit_setup_assistant.composer.eef_suggest import suggest_contract   # noqa: E402

SRDF_COMPOSITE = '''<?xml version="1.0"?>
<robot name="ur5e_suction">
  <group name="tool"><joint name="slide_joint"/><joint name="jaw_joint"/></group>
  <group_state name="home" group="tool"><joint name="slide_joint" value="0"/><joint name="jaw_joint" value="0.02"/></group_state>
  <group_state name="out" group="tool"><joint name="slide_joint" value="0.08"/><joint name="jaw_joint" value="0"/></group_state>
  <end_effector name="tool" parent_link="tool0" group="tool"/>
</robot>
'''


def legacy_project(**gripper) -> CanonicalProject:
    d = sample()
    d['robot']['gripper'].pop('contract')
    d['robot']['gripper'].update({'actuation': 'trigger', **gripper})
    d['schema_version'] = 2
    return CanonicalProject.parse_obj(d)


def test_a_suction_cup_from_the_stage_signal_with_a_binding_per_deployed_backend():
    p = legacy_project()
    p.scene.gripper_cmd_topic = '/suction_cmd'
    p.deployment.modes = [type(p.deployment.modes[0])(m) for m in ('mock', 'isaac', 'gazebo')]
    s = suggest_contract(p)
    assert s.kind == 'suction' and s.contract['actuator'] == {'command_interface': 'set_bool'}
    assert s.contract['adapters'] == {'mock': 'trainit_mock', 'gazebo': 'none',
                                      'isaac': {'adapter': 'trainit_isaac_surface_gripper',
                                                'stage_cmd_topic': '/suction_cmd'}}
    assert s.contract['verification'] == {'require_grasp_confirmation': False, 'require_release_confirmation': False}
    assert any("'/suction_cmd'" in line for line in s.sources) and any(line.startswith('gazebo:') for line in s.decide)


def test_a_controlled_jaw_from_its_gripper_command_controller():
    p = legacy_project(kind='parallel', actuation='joint_position', controller_type='GripperCommand',
                       controller_name='gripper_controller', command_joint='finger_joint',
                       eef_group_name='gripper', eef_name='hand')
    s = suggest_contract(p)
    assert s.kind == 'parallel'
    assert s.contract['actuator'] == {'command_interface': 'gripper_command', 'result_policy': 'succeeded_or_stalled'}
    assert any('open and closed targets' in d for d in s.decide)            # no SRDF states named


def test_a_multi_actuator_tool_from_the_end_effector_group(tmp_path):
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'robot.srdf').write_text(SRDF_COMPOSITE)
    p = legacy_project(eef_group_name='tool', eef_name='tool')
    p.robot.base_moveit_config_path = str(tmp_path)
    s = suggest_contract(p)
    assert s.kind == 'composite' and set(s.contract['actuators']) == {'slide_joint', 'jaw_joint'}
    assert s.contract['actuators']['slide_joint']['targets'] == {'home': [0.0], 'out': [0.08]}
    assert any('grasp sites' in d for d in s.decide)


def test_the_suggestion_on_the_reference_cell_is_the_probes_contract_and_adopts_cleanly():
    """The owner's reference project (when present): the contract the feasibility probe ran on mock with
    grasp + release SUCCESS (forensics/vnext-i0/eef_design), adopted with no contract problem."""
    path = Path(os.environ.get('TRAINIT_I0_REFERENCE_PROJECT', ''))
    if not os.environ.get('TRAINIT_I0_REFERENCE_PROJECT') or not path.is_file():
        pytest.skip('TRAINIT_I0_REFERENCE_PROJECT: the owner reference project (not in the repository)')
    from trainit_setup_assistant.generator.eef_contract import eef_contract_problems
    from trainit_setup_assistant.model.io import load_project
    from trainit_setup_assistant.model.paths import resolve_project_paths
    p, _ = resolve_project_paths(load_project(path), path)
    s = suggest_contract(p)
    assert s.kind == 'suction' and s.contract['adapters']['isaac']['stage_cmd_topic'] == '/suction_cmd'
    c = Composer(p)
    assert eef_ops.adopt_contract(c, s.kind, s.contract) == s.removes
    assert eef_contract_problems(c.project) == []


# ---- E3 / E4: the End effector section, the Tool forms, the diagnostics ----------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_the_section_suggests_adopts_and_edits_the_contract(qapp):
    from trainit_setup_assistant.gui.composer.eef_panel import EndEffectorPanel
    c = legacy_composer()
    c.project.scene.gripper_cmd_topic = '/suction_cmd'
    panel = EndEffectorPanel(lambda: c)
    panel.refresh()
    assert panel.suggest_box.isVisibleTo(panel) and not panel.actuators_box.isVisibleTo(panel)
    panel.suggest()
    assert panel.adopt_button.isEnabled() and 'suction' in panel.suggest_view.toPlainText()
    assert panel.adopt()
    assert 'removed the legacy bridge' in panel.message.text() and eef_ops.contract_of(c) is not None
    assert panel.actuators_box.isVisibleTo(panel) and panel.actuators.item(0, 0).text() == 'vacuum'
    panel.require_grasp.setChecked(True)
    panel.verification_box.findChildren(type(panel.suggest_button))[0].click()
    assert eef_ops.contract_of(c)['verification']['require_grasp_confirmation'] is True
    combo, edits = panel.binding_rows['mock']
    assert combo.findData('trainit_mock') >= 0 and edits == {}              # the mock rows read no key
    panel.long_button.click()                                                # multi-actuator
    assert panel.sites_box.isVisibleTo(panel) and panel.sites.rowCount() == 1
    panel.new_actuator.setText('lock')
    assert panel.add_actuator()
    panel.selected_actuator = 'lock'
    panel.interface.setCurrentText('follow_joint_trajectory')
    panel.action.setText('lock_controller/follow_joint_trajectory')
    panel.joints.setText('lock_joint')
    assert panel.apply_actuator()
    panel.target_name.setText('locked')
    panel.target_value.setText('0.01')
    assert panel.set_target()
    lock = eef_ops.contract_of(c)['actuators']['lock']
    assert lock['command_interface'] == 'follow_joint_trajectory' and lock['targets'] == {'locked': [0.01]}
    assert 'lock:' in panel.yaml_view.toPlainText()
    c.undo()                                                                 # every edit is undoable
    assert 'targets' not in eef_ops.contract_of(c)['actuators']['lock']


def test_close_and_open_are_offered_in_tasks_and_a_refusal_is_shown_by_the_buttons(qapp):
    from trainit_setup_assistant.gui.composer.tasks_panel import ADD_TITLES, TasksPanel
    assert ADD_TITLES['tool_close'] == 'Close gripper' and ADD_TITLES['tool_open'] == 'Open gripper'
    c = legacy_composer()
    panel = TasksPanel(lambda: c)
    panel.task = 'cycle'
    panel.refresh()
    assert panel.add_step('tool_close') is None                              # no contract yet
    assert 'End effector section' in panel.bar_message.text()
    eef_ops.adopt_contract(c, 'suction', SUCTION)
    panel.refresh()
    label = panel.add_step('tool_close')
    assert label is not None and c.document()['tasks'][0]['steps'][-1] == {'tool': {'close': {}}}


def test_the_tool_form_writes_close_open_and_offers_only_actuatable_targets(qapp):
    from trainit_setup_assistant.gui.composer.step_forms import ToolForm, context
    from eef_fixture import composite_mock_contract
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'composite', composite_mock_contract(''))
    ctx = context(c.document(), c.project)
    assert ctx['actuators']['slide_axis'] == ['home', 'extended']
    assert ctx['actuators']['binary_lock'] == ['engage', 'release']         # a set_bool's targets are named
    assert 'open' not in ctx['actuators']['controlled_gripper']             # a grip's targets: close / open
    form = ToolForm()
    form.load({'tool': {'close': {'site': 'site_b'}}}, ctx)
    assert form.mode.currentText() == 'close' and form.site.currentText() == 'site_b'
    form.mode.setCurrentText('open')
    assert form.spec() == {'tool': {'open': {'site': 'site_b'}}}


def test_a_contract_refusal_is_located_at_the_end_effector_section():
    from trainit_setup_assistant.composer.diagnostics import parse
    d = parse('end-effector contract (ADR-0022 R15): device "main" on real: the backend is deployed and has no '
              'binding (contract.adapters) -- bind it, or use "none"')
    assert d.code == 'R15' and d.location.kind == 'end_effector'
    assert 'End effector section' in d.location.describe()


# ---- E5: the read-only live check and the capture of a target -------------------------------------------

def test_the_live_check_reads_the_graph_and_the_state_and_commands_nothing(monkeypatch):
    rclpy = pytest.importorskip('rclpy')
    pytest.importorskip('trainit_interfaces')
    import threading
    from rclpy.executors import SingleThreadedExecutor
    from std_srvs.srv import SetBool
    from trainit_interfaces.msg import EndEffectorState
    from trainit_setup_assistant.livesession import LiveCapture
    from trainit_setup_assistant.livesession.eef_check import eef_checks
    monkeypatch.setenv('ROS_DOMAIN_ID', str(150 + os.getpid() % 40))     # never the owner's domain
    rclpy.init()
    p = CanonicalProject.parse_obj(sample())                    # a suction cup, mock adapter, short form
    calls = []
    node = rclpy.create_node('fake_mock_eef_adapter')
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    # the hardware layer adapter's state QoS (eef/adapter_node.py STATE_QOS; TMR's registry reads it alike)
    state_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    pub = node.create_publisher(EndEffectorState, '/eef/main/state', state_qos)
    node.create_service(SetBool, '/eef/main/actuators/vacuum/set_engaged',
                        lambda req, res: calls.append(req) or res)
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    stop = threading.Event()

    def spin():
        while not stop.is_set():
            pub.publish(EndEffectorState(end_effector='main', adapter='trainit_mock', status=1,
                                         sites=['main'], site_grip=['vacuum'], site_grasp=[2], site_evidence=[0]))
            ex.spin_once(timeout_sec=0.05)
    t = threading.Thread(target=spin, daemon=True)
    t.start()
    live = LiveCapture('i0_e_live')
    try:
        deadline = __import__('time').time() + 5
        got = {}
        while __import__('time').time() < deadline:
            got = {c.name: c for c in eef_checks(live, p, timeout_s=1.0)}
            if got.get('state') and got['state'].ok and got.get("actuator 'vacuum'") and got["actuator 'vacuum'"].ok:
                break
        assert got['state'].ok and "'fake_mock_eef_adapter'" in got['state'].text
        assert got["actuator 'vacuum'"].ok and 'set_engaged: served' in got["actuator 'vacuum'"].text
        assert got['latest state'].ok and 'READY' in got['latest state'].text and 'main: not held' in got['latest state'].text
        assert calls == []                                       # observed, never commanded (ADR-0027)
    finally:
        live.close()
        stop.set()
        t.join(timeout=2)
        node.destroy_node()
    live = LiveCapture('i0_e_live_empty')
    try:
        got = {c.name: c for c in eef_checks(live, p, timeout_s=0.5)}
        assert got['state'].ok is False and 'nobody publishes it' in got['state'].text
        assert got["actuator 'vacuum'"].ok is False
    finally:
        live.close()
        if rclpy.ok():
            rclpy.shutdown()


def test_a_named_target_is_captured_from_the_tools_current_joints(qapp):
    from trainit_setup_assistant.gui.composer.eef_panel import EndEffectorPanel
    from eef_fixture import composite_mock_contract

    class Live:
        def current_joint_values(self, names, timeout_s=3.0):
            return {'slide_joint': 0.042, 'rotation_joint': 1.2}
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'composite', composite_mock_contract(''))
    panel = EndEffectorPanel(lambda: c, live=lambda: Live())
    panel.refresh()
    panel.selected_actuator = 'slide_axis'
    panel.target_name.setText('mid')
    assert panel.capture_target(), panel.message.text()
    assert eef_ops.contract_of(c)['actuators']['slide_axis']['targets']['mid'] == [0.042]
    assert "'mid' captured: 0.0420" in panel.message.text()


def test_a_plain_open_has_no_held_guard_even_when_grasps_are_confirmed():
    """Architecture review M1: 'open at the start of a cycle' holds nothing; the release idiom's pre-open 'held'
    guard (ADR-0022 D4) would fail it (NO_ENGAGE_COMMAND). A held guard is an explicit tool verify."""
    data = yaml.safe_load(VALIDATION_C.read_text())        # grasp AND release confirmed
    data.pop('application')
    xml = tree(program(data, [{'tool': {'open': {}}}, {'tool': {'verify': {'expect': 'held'}}},
                              {'tool': {'open': {}}}])).splitlines()
    assert xml[0] == '<OpenGripper eef="main" site="main"/>'               # nothing checked before
    assert 'expect="released"' in xml[1]
    assert 'expect="held"' in xml[2]                                          # the explicit guard, as written


SRDF_HAND = '''<?xml version="1.0"?>
<robot name="ur5e_suction">
  <group name="hand"><joint name="finger_joint1"/><joint name="finger_joint2"/></group>
  <group_state name="open" group="hand"><joint name="finger_joint1" value="0.04"/><joint name="finger_joint2" value="0.04"/></group_state>
  <group_state name="close" group="hand"><joint name="finger_joint1" value="0"/><joint name="finger_joint2" value="0"/></group_state>
  <end_effector name="hand" parent_link="tool0" group="hand"/>
</robot>
'''


def test_a_two_finger_moveit_hand_with_a_gripper_command_controller_is_a_jaw(tmp_path):
    """Architecture review S1: the usual MoveIt hand group lists both fingers; its GripperCommand controller
    makes it ONE controlled jaw, never a composite tool (a mimic follower is not an actuator, ADR-0022 D0)."""
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'robot.srdf').write_text(SRDF_HAND)
    p = legacy_project(kind='parallel', actuation='joint_position', controller_type='GripperCommand',
                       controller_name='hand_controller', command_joint='finger_joint1',
                       eef_group_name='hand', eef_name='hand', open_state='open', closed_state='close')
    p.robot.base_moveit_config_path = str(tmp_path)
    p.deployment.modes = [type(p.deployment.modes[0])(m) for m in ('mock', 'isaac', 'gazebo', 'real')]
    s = suggest_contract(p)
    assert s.kind == 'parallel' and s.contract['actuator']['command_interface'] == 'gripper_command'
    assert s.contract['adapters'] == {'mock': 'trainit_mock', 'isaac': 'none', 'gazebo': 'ros2_control',
                                      'real': 'none'}                         # from the catalog's rows
    assert any(line.startswith('isaac:') for line in s.decide)
    # real: ros2_control needs its flavour declared (only gazebo's is read from the base config) -- never a
    # binding R15 refuses (ROS2ML-SYNC D4)
    assert any(line.startswith("real: 'ros2_control' serves it once its controller flavour is declared")
               for line in s.decide)


def _course_like_jaw(tmp_path, **gripper):
    """A Robotiq-2F-85-like jaw on a GripperCommand controller in a Gazebo cell that sends its LinkAttacher
    weld on the grasp-signal topic (the ROS2ML course cell), with the legacy grasp adapters of TSA v6."""
    (tmp_path / 'config').mkdir(parents=True)
    (tmp_path / 'config' / 'robot.srdf').write_text(SRDF_HAND)
    (tmp_path / 'config' / 'ros2_controllers.yaml').write_text(
        'controller_manager:\n  ros__parameters:\n    hand_controller:\n'
        '      type: position_controllers/GripperActionController\n')
    p = legacy_project(**{**dict(kind='parallel', actuation='joint_position', controller_type='GripperCommand',
                                 controller_name='hand_controller', action_ns='gripper_cmd',
                                 command_joint='finger_joint1', eef_group_name='hand', eef_name='hand',
                                 open_state='open', closed_state='close'), **gripper})
    p.robot.base_moveit_config_path = str(tmp_path)
    p.deployment.modes = [type(p.deployment.modes[0])(m) for m in ('mock', 'isaac', 'gazebo', 'real')]
    p.scene.gripper_cmd_topic = '/gripper_cmd'
    p.robot.gripper.sim_grasp_adapter = _legacy_adapters(mock='link_attacher', isaac='surface_gripper', real='none',
                                                         gazebo='link_attacher')
    return p


def _legacy_adapters(**by_backend):
    from trainit_setup_assistant.model.enums import SimGraspAdapter
    return {k: SimGraspAdapter(v) for k, v in by_backend.items()}


def _plain(adapters):
    return {k: getattr(v, 'value', v) for k, v in adapters.items()}


def test_a_grasp_signal_topic_does_not_make_a_gripper_command_jaw_a_suction_cup(tmp_path):
    """ROS2ML-SYNC D4: the course cell's '/gripper_cmd' is its Gazebo weld signal, not a suction command."""
    s = suggest_contract(_course_like_jaw(tmp_path))
    assert s.kind == 'parallel' and s.contract['actuator']['command_interface'] == 'gripper_command'
    assert s.contract['adapters']['gazebo'] == 'ros2_control'
    # a suction name still decides first, GripperCommand or not (a suction action server is one)
    s = suggest_contract(_course_like_jaw(tmp_path / 'b', eef_group_name='suction', eef_name='suction'))
    assert s.kind == 'suction'
    # an Isaac SurfaceGripper cell whose only evidence is its stage topic stays a suction cup, even behind a
    # GripperCommand action server: no Gazebo weld reads that topic there (python review S1)
    p = _course_like_jaw(tmp_path / 'c', command_joint='tool_joint', eef_group_name='tool', eef_name='tool')
    p.deployment.modes = [type(p.deployment.modes[0])(m) for m in ('mock', 'isaac')]
    s = suggest_contract(p)
    assert s.kind == 'suction' and s.contract['adapters']['isaac']['stage_cmd_topic'] == '/gripper_cmd'


def test_adopting_resets_the_legacy_grasp_adapters_but_the_gazebo_weld(tmp_path):
    """ROS2ML-SYNC D5: TSA v6's sim_grasp_adapter.mock 'link_attacher' disagrees with a 'trainit_mock' binding
    (R23): adopting resets every backend's but Gazebo's, says so, and leaves no binding / legacy problem."""
    from trainit_setup_assistant.generator.eef_contract import eef_contract_problems
    c = Composer(apply_template(_course_like_jaw(tmp_path), 'pick_and_place'))   # a student's first program
    s = suggest_contract(c.project)
    assert any("adopting resets sim_grasp_adapter.mock 'link_attacher'" in line for line in s.sources)
    eef_ops.adopt_contract(c, s.kind, s.contract)
    assert _plain(c.project.robot.gripper.sim_grasp_adapter) == {
        'isaac': 'surface_gripper', 'real': 'none', 'gazebo': 'link_attacher'}     # the defaults kept, the weld kept
    # no binding problem (R15) and no legacy disagreement (R23); this sample's description has no jaw joint,
    # so the description rules (R5, R9) are the fixture's, not the adoption's
    assert [p for p in eef_contract_problems(c.project) if 'R23' in p or 'R15' in p] == []


def test_a_whole_project_edit_is_the_program_and_the_end_effector_only():
    """Architecture review S6: Composer.edit_project must not become a general project editor that bypasses
    Steps 1-4."""
    c = legacy_composer()
    before = project_to_dict(c.project)
    with pytest.raises(ComposerError, match='the program and the end effector'):
        c.edit_project(lambda d: d['robot'].__setitem__('tip_link', 'tool0'))
    with pytest.raises(ComposerError, match='the program and the end effector'):
        c.edit_project(lambda d: d['scene'].__setitem__('objects', []))
    assert project_to_dict(c.project) == before and not c.can_undo


# ---- the python review's fixes (M2, M3, S1-S6) -----------------------------------------------------------

def jaw_composer() -> Composer:
    """A short-form controlled jaw whose open / closed targets DERIVE from the project's named states."""
    d = sample()
    d['robot']['gripper'] = {
        'kind': 'parallel', 'eef_parent_link': 'tcp', 'actuation': 'joint_position',
        'controller_type': 'GripperCommand', 'controller_name': 'hand_controller', 'command_joint': 'finger_joint',
        'open_state': 'hand_open', 'closed_state': 'hand_closed',
        'contract': {'actuator': {'command_interface': 'gripper_command'},
                     'verification': {'require_grasp_confirmation': False, 'require_release_confirmation': False},
                     'adapters': {'mock': 'trainit_mock'}}}
    d['robot']['named_states'] += [{'name': 'hand_open', 'group': 'hand', 'joint_values': {'finger_joint': 0.04}},
                                   {'name': 'hand_closed', 'group': 'hand', 'joint_values': {'finger_joint': 0.0}}]
    return Composer(apply_template(CanonicalProject.parse_obj(d), 'simple_motion'))


def _jaw_id(c) -> str:
    return eef_ops.actuator_ids(eef_ops.contract_of(c), 'parallel')[0]


def test_writing_one_target_on_a_jaw_keeps_its_derived_open_and_closed():
    """M3: the first written target would otherwise stop the derivation of 'open' / 'closed' (ADR-0022 D6)."""
    c = jaw_composer()
    aid = _jaw_id(c)
    eef_ops.set_target(c, aid, 'half', 0.02)
    assert eef_ops.contract_of(c)['actuator']['targets'] == {'open': 0.04, 'closed': 0.0, 'half': 0.02}
    plan_program(c.project)
    c.undo()
    assert 'targets' not in eef_ops.contract_of(c)['actuator']


def test_a_sites_derived_engage_and_release_count_as_users():
    """S1: a site with no written engage / release uses its grip's 'closed' / 'open' all the same."""
    c = jaw_composer()
    aid = _jaw_id(c)
    assert eef_ops.target_users(c, aid, 'closed') == ["site 'main'"]
    with pytest.raises(ComposerError, match="used by site 'main'"):
        eef_ops.delete_target(c, aid, 'closed')
    with pytest.raises(ComposerError, match='fixed by the short form'):
        eef_ops.rename_target(c, aid, 'open', 'wide')
    with pytest.raises(ComposerError, match="no written target 'half'"):
        eef_ops.delete_target(c, aid, 'half')
    eef_ops.to_long_form(c)                                   # long form: the grip's names are the site's
    eef_ops.rename_target(c, aid, 'closed', 'shut')
    k = eef_ops.contract_of(c)
    assert k['actuators'][aid]['targets'] == {'open': 0.04, 'shut': 0.0}
    assert k['sites']['main']['engage'] == 'shut' and 'release' not in k['sites']['main']
    plan_program(c.project)


def test_the_long_form_keeps_what_was_written_and_adding_never_replaces():
    """S2: a written value (here confirm_timeout_ms) survives the long form. S3: Add refuses an existing id and
    a single-actuator device. R3: an id is a name token, refused where it is typed."""
    c = legacy_composer()
    contract = copy.deepcopy(SUCTION)
    contract['actuator']['confirm_timeout_ms'] = 1500
    eef_ops.adopt_contract(c, 'suction', contract)
    with pytest.raises(ComposerError, match='has one actuator'):
        eef_ops.add_actuator(c, 'lock', {'command_interface': 'set_bool'})
    eef_ops.to_long_form(c)
    assert eef_ops.contract_of(c)['actuators']['vacuum']['confirm_timeout_ms'] == 1500
    with pytest.raises(ComposerError, match="already named 'vacuum'"):
        eef_ops.add_actuator(c, 'vacuum', {'command_interface': 'set_bool'})
    eef_ops.add_actuator(c, 'lock', {'command_interface': 'set_bool'})
    assert list(eef_ops.contract_of(c)['actuators']) == ['vacuum', 'lock']
    for bad in ('', '2x', 'a b', 'main\n'):
        with pytest.raises(ComposerError, match='is not a name'):
            eef_ops.rename_actuator(c, 'lock', bad)
    with pytest.raises(ComposerError, match='is not a name'):
        eef_ops.set_site(c, 'front site', {'tcp': 'tcp', 'grip': 'lock'})
    with pytest.raises(ComposerError, match='is not a name'):
        eef_ops.set_device_id(c, 'my-tool')


def test_the_kind_change_keeps_a_short_forms_actuator_name():
    """N2: the kind decides the DEFAULT actuator id, and with it the endpoint names; a kind change keeps it."""
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'suction', SUCTION)
    eef_ops.set_kind(c, 'parallel')
    assert eef_ops.actuator_ids(eef_ops.contract_of(c), 'parallel') == ['vacuum']


def test_the_panel_shows_every_failure_and_follows_a_rename(qapp):
    """M2: a failure of a live read is shown, never raised into Qt. S5: a rename keeps the renamed element
    selected and clears the fields naming the old one. M3 (visibility): derived targets are listed, marked."""
    from python_qt_binding.QtWidgets import QPushButton
    from trainit_setup_assistant.gui.composer.eef_panel import EndEffectorPanel
    from eef_fixture import composite_mock_contract

    class StoppedLive:
        def current_joint_values(self, names, timeout_s=3.0):
            raise TimeoutError('no joint states within 3 s')
    c = legacy_composer()
    eef_ops.adopt_contract(c, 'composite', composite_mock_contract(''))
    panel = EndEffectorPanel(lambda: c, live=lambda: StoppedLive())
    panel.refresh()
    panel.selected_actuator = 'slide_axis'
    panel.target_name.setText('mid')
    assert not panel.capture_target()
    assert panel.message.text() == 'TimeoutError: no joint states within 3 s'
    panel.target_value.setText('0.1, abc')
    assert not panel.set_target() and panel.message.text() == "'abc' is not a number"
    panel.new_actuator.setText('slide')
    assert panel.rename_actuator()
    assert panel.selected_actuator == 'slide' and panel.new_actuator.text() == ''
    panel.site_id.setText('site_a')
    panel.site_new.setText('front')
    rename = [b for b in panel.sites_box.findChildren(QPushButton) if b.text() == 'Rename'][0]
    rename.click()
    assert 'front' in eef_ops.contract_of(c)['sites'] and panel.site_id.text() == panel.site_new.text() == ''
    jaw = jaw_composer()
    panel = EndEffectorPanel(lambda: jaw)
    panel.refresh()
    rows = {panel.targets.item(i, 0).text(): panel.targets.item(i, 1).text() for i in range(panel.targets.rowCount())}
    assert rows == {'open': '0.04 (derived)', 'closed': '0.0 (derived)'}


def test_the_live_check_reads_the_joint_states_once_and_says_when_it_cannot():
    """S6: one read for every actuator (the GUI waits once); a read that fails is said, not mistaken for
    missing joints."""
    from eef_fixture import composite_mock_contract
    from trainit_setup_assistant.livesession.eef_check import eef_checks

    class Node:
        def get_publishers_info_by_topic(self, topic):
            return []

        def get_service_names_and_types(self):
            return []

        def get_topic_names_and_types(self):
            return []

    class Live:
        node = Node()

        def __init__(self, fail=False):
            self.calls, self.fail = [], fail

        def current_joint_values(self, names, timeout_s=3.0):
            self.calls.append(list(names))
            if self.fail:
                raise TimeoutError('no joint states within 0.1 s')
            return {'slide_joint': 0.0, 'rotation_joint': 0.0}
    data = sample()
    data['robot']['gripper'] = {'kind': 'composite', 'contract': composite_mock_contract('')}
    p = program(data, [{'dwell': {'ms': 10}}])
    live = Live()
    got = {c.name: c for c in eef_checks(live, p, timeout_s=0.1)}
    assert len(live.calls) == 1
    assert got["joints of 'slide_axis'"].ok and got["joints of 'rotation_axis'"].ok
    assert got["joints of 'controlled_gripper'"].ok is False and 'gripper_jaw_joint' in got["joints of 'controlled_gripper'"].text
    live = Live(fail=True)
    got = {c.name: c for c in eef_checks(live, p, timeout_s=0.1)}
    assert len(live.calls) == 1
    assert got["joints of 'slide_axis'"].ok is None and 'cannot be read: no joint states' in got["joints of 'slide_axis'"].text


def test_a_jaw_suggestion_says_what_does_not_resolve(tmp_path):
    """S4: a GripperCommand jaw with no controller name, or whose named states give no value for its joint,
    leaves those to decide (never a contract that fails at generation)."""
    (tmp_path / 'config').mkdir()
    (tmp_path / 'config' / 'robot.srdf').write_text(SRDF_HAND)
    p = legacy_project(kind='parallel', actuation='joint_position', controller_type='GripperCommand',
                       command_joint='finger_joint', eef_group_name='hand', eef_name='hand',
                       open_state='open', closed_state='close')
    p.robot.base_moveit_config_path = str(tmp_path)
    s = suggest_contract(p)
    assert s.kind == 'parallel'
    assert any(d.startswith('the controller action') for d in s.decide)
    assert any("give no value for joint 'finger_joint'" in d for d in s.decide)


def test_adopting_resets_only_what_r23_refuses_and_undo_restores_it(tmp_path):
    """Python review S2/S3: a deliberate 'none' next to a 'none' binding, a backend not deployed and a key that
    names no backend are kept (R23 reads none of them); one undo brings the legacy map back."""
    p = _course_like_jaw(tmp_path)
    p.deployment.modes = [type(p.deployment.modes[0])(m) for m in ('mock', 'isaac', 'gazebo')]     # no real
    p.robot.gripper.sim_grasp_adapter = _legacy_adapters(mock='link_attacher', isaac='none', real='link_attacher',
                                                         gazebo='link_attacher')
    p.robot.gripper.sim_grasp_adapter['sim'] = p.robot.gripper.sim_grasp_adapter['mock']      # an old hand edit
    c = Composer(apply_template(p, 'pick_and_place'))
    before = _plain(c.project.robot.gripper.sim_grasp_adapter)
    s = suggest_contract(c.project)
    assert s.contract['adapters']['isaac'] == 'none'
    eef_ops.adopt_contract(c, s.kind, s.contract)
    assert _plain(c.project.robot.gripper.sim_grasp_adapter) == {
        'isaac': 'none', 'real': 'link_attacher', 'gazebo': 'link_attacher', 'sim': 'link_attacher'}   # mock reset
    c.undo()
    assert _plain(c.project.robot.gripper.sim_grasp_adapter) == before and c.project.robot.gripper.contract is None
