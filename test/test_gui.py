"""M6: controller + Qt wizard, driven headless (QT_QPA_PLATFORM=offscreen)."""

import os
import tempfile

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from trainit_setup_assistant.gui.controller import AssistantController  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')
FR3WML_XACRO = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/fr3wml_app/'
                'fr3wml_description/urdf/fr3wml_suction.urdf.xacro')
FR3WML_MESHES = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/fr3wml_app/'
                 'fr3wml_description/meshes')


# ---- controller (ROS-free) ----
def test_controller_open_edit_generate():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.add_named_state('park', {'j1': 0.0, 'j2': -1.0, 'j3': 0.0,
                                  'j4': -1.5, 'j5': 0.0, 'j6': 0.0})
    assert 'park' in ctrl.robot_summary()['named_states']
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('ctrl_gen_test')
        manifest = ctrl.generate(tmp)
        srdf = os.path.join(tmp, 'ctrl_gen_test_moveit_config', 'config', 'fr3wml.srdf')
        assert os.path.isfile(srdf)
        assert 'park' in open(srdf).read()
        assert manifest.as_dict()['file_count'] > 30


def test_controller_requires_project():
    with pytest.raises(RuntimeError):
        AssistantController().robot_summary()


def test_mock_gripper_server_generated_for_bootstrap():
    """A gripper robot with no cell bridges gets a generated mock gripper action server."""
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.deployment.bridges = []          # bootstrap: no cell bridges
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('mockgrip_test')
        ctrl.generate(tmp)
        app = os.path.join(tmp, 'mockgrip_test')
        script = os.path.join(app, 'scripts', 'mock_gripper_action_server.py')
        assert os.path.isfile(script)
        assert 'GripperCommand' in open(script).read()
        bringup = open(os.path.join(app, 'launch', 'bringup.launch.py')).read()
        assert 'GRIPPER_MOCK_ACTION' in bringup
        assert 'mock_gripper_action_server.py' in bringup
        assert 'mock_gripper_action_server.py' in open(os.path.join(app, 'CMakeLists.txt')).read()


GOLDEN_SRDF = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/fr3wml_app/'
               'fr3wml_app_moveit_config/config/fr3wml.srdf')


@pytest.mark.skipif(not os.path.isfile(GOLDEN_SRDF), reason='golden SRDF not present')
def test_controller_import_collision_matrix():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.robot.disable_collisions = None
    n = ctrl.import_collision_matrix_from_srdf(GOLDEN_SRDF)
    assert n == 13
    assert len(ctrl.project.robot.disable_collisions) == 13


QUAT = [-0.707, 0.707, 0.008, -0.009]
GOLDEN_ROOT = '/home/fra/fr5_ws/src/fr3wml_digital_twin/fr3wml_app'


@pytest.mark.skipif(not os.path.isdir(GOLDEN_ROOT), reason='golden not present')
def test_controller_rebuilds_golden_app():
    """Build the FR3WML pick&place app entirely via controller calls -> == golden app."""
    from trainit_setup_assistant.model import BundleSpec
    from trainit_setup_assistant.verify import compare_bundle
    from trainit_setup_assistant.verify.equivalence import APP, DESCRIPTION, MOVEIT

    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)               # robot half (incl. collision matrix)
    ctrl.clear_application()
    ctrl.project.scene.objects = []
    ctrl.project.scene.payload = None

    ctrl.set_application('pick_and_place', 'pilz')
    ctrl.add_scene_object('table', [2.0, 2.0, 0.10], [0.0, 0.0, -0.08])
    ctrl.set_payload('cube', [0.02, 0.02, 0.02], 'tcp', [0.0, 0.0, 0.01])

    ctrl.add_move('home', named='home2', motion='ptp', planner='pilz', speed=80, role='home')
    ctrl.add_move('pre_pick', position=[0.556, -0.029, 0.167], orientation=QUAT,
                  motion='free', planner='pilz', speed=60, role='pre_pick')
    ctrl.add_move('pick', position=[0.558, -0.026, 0.024], orientation=QUAT,
                  motion='lin', speed=30, role='pick')
    ctrl.add_tool_action('pick', 'grasp')
    ctrl.add_tool_action('pick', 'attach', 'cube')
    ctrl.add_move('post_pick', position=[0.558, -0.026, 0.124], orientation=QUAT,
                  motion='lin', speed=40, role='post_pick')
    ctrl.add_move('pre_place', position=[0.564, -0.275, 0.248], orientation=QUAT,
                  motion='free', planner='pilz', speed=60, role='pre_place')
    ctrl.add_move('place', position=[0.558, -0.275, 0.023], orientation=QUAT,
                  motion='lin', speed=30, role='place')
    ctrl.add_tool_action('place', 'release')
    ctrl.add_tool_action('place', 'detach', 'cube')
    ctrl.add_move('post_place', position=[0.558, -0.275, 0.123], orientation=QUAT,
                  motion='lin', speed=40, role='post_place')
    ctrl.add_move('home', named='home2', motion='ptp', planner='pilz', speed=80, role='home')

    ctrl.project.project_name = 'fr3wml_app'
    ctrl.project.bundle = BundleSpec(description_package='fr3wml_description',
                                     moveit_config_package='fr3wml_app_moveit_config',
                                     app_package='fr3wml_app')
    pkgs = {DESCRIPTION: 'fr3wml_description', MOVEIT: 'fr3wml_app_moveit_config',
            APP: 'fr3wml_app'}
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.generate(tmp)
        report = compare_bundle(tmp, GOLDEN_ROOT, pkgs, pkgs)
        app_checks = [c for c in report.checks if c[0].startswith('app/')]
        failed = [c for c in app_checks if not c[2]]
        assert not failed, '\n'.join(f'{c[0]}: {c[3]}' for c in failed)


# ---- Qt wizard (offscreen) ----
@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.mark.skipif(not os.path.isfile(FR3WML_XACRO), reason='FR3WML xacro not present')
def test_wizard_offscreen_full_flow(qapp):
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    wiz = SetupWizard(ctrl)

    # S0 load robot
    wiz.load_page.xacro.setText(FR3WML_XACRO)
    wiz.load_page.robot_name.setText('fr3wml')
    wiz.load_page.group_name.setText('fr3wml')
    wiz.load_page.meshes.setText(FR3WML_MESHES)
    if not wiz.load_page.validatePage():
        pytest.skip('xacro compile needs a ROS env: ' + wiz.load_page.summary.text())
    assert ctrl.robot_summary()['tip_link'] == 'tcp'
    assert ctrl.robot_summary()['arm_joints'] == ['j1', 'j2', 'j3', 'j4', 'j5', 'j6']

    # S1 group/frames
    wiz.group_page.initializePage()
    assert wiz.group_page.base.text() == 'base_link'
    assert wiz.group_page.validatePage()

    # S2 named states
    wiz.states_page.initializePage()
    wiz.states_page.state_name.setText('home2')
    wiz.states_page.state_joints.setText(
        'j1=0, j2=-1.0814, j3=-1.62, j4=-1.57, j5=1.57, j6=0')
    wiz.states_page.add_state()
    assert 'home2' in ctrl.robot_summary()['named_states']

    # S6 generate
    with tempfile.TemporaryDirectory() as tmp:
        wiz.generate_page.initializePage()
        wiz.generate_page.project_name.setText('fr3wml_wizard_test')
        wiz.generate_page.output_dir.setText(tmp)
        wiz.generate_page.generate()
        srdf = os.path.join(tmp, 'fr3wml_wizard_test_moveit_config', 'config', 'fr3wml.srdf')
        assert os.path.isfile(srdf), wiz.generate_page.result.toPlainText()
        assert 'home2' in open(srdf).read()
        # description geometry copied
        assert os.path.isfile(os.path.join(
            tmp, 'fr3wml_wizard_test_description', 'urdf', 'fr3wml_suction.urdf.xacro'))


def test_wizard_open_project_keeps_matrix(qapp):
    """S0 'Open project.yaml' starts from a complete project (matrix + bridges kept)."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    wiz = SetupWizard(ctrl)
    wiz.load_page.project_file.setText(EXAMPLE)
    assert wiz.load_page.validatePage()
    assert ctrl.project.robot.disable_collisions, 'matrix should carry over from project'
    assert 'collision-matrix: yes' in wiz.load_page.summary.text()
    # regenerating yields a buildable bundle WITH the matrix
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('reopened_test')
        ctrl.generate(tmp)
        srdf = os.path.join(tmp, 'reopened_test_moveit_config', 'config', 'fr3wml.srdf')
        assert open(srdf).read().count('<disable_collisions') == 13


def test_scene_category_change_applies_immediately(qapp):
    """Selecting an object + changing its category applies without 'Add/update object'."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.add_scene_object('part', [0.02, 0.02, 0.02], [0.5, 0.0, 0.05])  # static
    wiz = SetupWizard(ctrl)
    sp = wiz.scene_page
    sp.initializePage()
    item = next(sp.list.item(i) for i in range(sp.list.count())
                if sp.list.item(i).text().startswith('part'))
    sp._on_select(item)
    assert sp.category.currentText() == 'static'
    sp.category.setCurrentText('dynamic')            # change -> applies live
    part = next(o for o in ctrl.project.scene.objects if o.id == 'part')
    assert part.is_dynamic()
    assert not part.is_planning_collision()          # collision-allowed now
    sp.category.setCurrentText('actuated')           # actuated -> checked again
    part = next(o for o in ctrl.project.scene.objects if o.id == 'part')
    assert part.is_actuated() and part.is_planning_collision()


def test_wizard_controllers_page(qapp):
    """Editing controllers in S2 propagates into the generated controller yamls."""
    import yaml
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    wiz = SetupWizard(ctrl)
    cp = wiz.controllers_page
    cp.initializePage()
    assert cp.arm_name.text() == 'moveit_joint_controller'
    assert cp.grip_label.text() == 'suction'
    cp.arm_name.setText('my_arm_ctrl')
    cp.arm_update_rate.setValue(200)
    assert cp.validatePage()
    assert ctrl.project.robot.arm_controller.name == 'my_arm_ctrl'
    assert ctrl.project.robot.arm_controller.update_rate == 200
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('ctrl_page_test')
        ctrl.generate(tmp)
        rc = yaml.safe_load(open(os.path.join(
            tmp, 'ctrl_page_test_moveit_config', 'config', 'ros2_controllers.yaml')))
        params = rc['controller_manager']['ros__parameters']
        assert 'my_arm_ctrl' in params
        assert params['update_rate'] == 200


def test_wizard_app_pages_build_app(qapp):
    """Drive S4 (application+payload) and S5 (waypoints) offscreen, then generate."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)          # robot half ready
    ctrl.clear_application()
    ctrl.project.scene.objects = []
    ctrl.project.scene.payload = None
    wiz = SetupWizard(ctrl)

    # S4 application + payload
    wiz.application_page.app_type.setCurrentText('pick_and_place')
    wiz.application_page.planner.setCurrentText('pilz')
    wiz.application_page.payload_id.setText('cube')
    wiz.application_page.payload_dims.setText('0.02, 0.02, 0.02')
    wiz.application_page.payload_offset.setText('0, 0, 0.01')
    assert wiz.application_page.validatePage()
    assert ctrl.project.scene.payload.id == 'cube'

    # S5 waypoints — add home (named) then a pick (tcp) with grasp+attach
    wp = wiz.waypoints_page
    wp.move_name.setText('home')
    wp.target_mode.setCurrentText('named')
    wp.named.setText('home2')
    wp.motion.setCurrentText('ptp')
    wp.move_planner.setCurrentText('pilz')
    wp.speed.setValue(80)
    wp.role.setCurrentText('home')
    wp.add_move()

    wp.move_name.setText('pick')
    wp.target_mode.setCurrentText('tcp')
    wp.pos.setText('0.558, -0.026, 0.024')
    wp.quat.setText('-0.707, 0.707, 0.008, -0.009')
    wp.motion.setCurrentText('lin')
    wp.move_planner.setCurrentText('')
    wp.speed.setValue(30)
    wp.role.setCurrentText('pick')
    wp.tool_cbs['grasp'].setChecked(True)
    wp.tool_cbs['attach'].setChecked(True)
    wp.payload_ref.setText('cube')
    wp.add_move()

    assert ctrl.project.application.sequence == ['home', 'pick']
    kinds = {a.kind.value for a in ctrl.project.application.tool_actions}
    assert {'grasp', 'attach'} <= kinds

    with tempfile.TemporaryDirectory() as tmp:
        wiz.generate_page.initializePage()
        wiz.generate_page.project_name.setText('fr3wml_wiz_app')
        wiz.generate_page.output_dir.setText(tmp)
        wiz.generate_page.generate()
        import yaml
        bt = os.path.join(tmp, 'fr3wml_wiz_app', 'config', 'bt_params.yaml')
        data = yaml.safe_load(open(bt))['/**']['ros__parameters']['task_parameters']
        assert data['home']['named'] == 'home2'
        assert data['pick']['motion'] == 'lin'
        assert data['pick']['speed'] == 30
        tree = open(os.path.join(tmp, 'fr3wml_wiz_app', 'bt_trees', 'pick_place.xml')).read()
        assert '<CloseGripper/>' in tree and 'AttachObject' in tree


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
