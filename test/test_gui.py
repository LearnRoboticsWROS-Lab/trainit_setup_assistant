"""M6: controller + Qt wizard, driven headless (QT_QPA_PLATFORM=offscreen)."""

import os
import tempfile

import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from trainit_setup_assistant.gui.controller import AssistantController  # noqa: E402

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')
# The robot description lives in fr5_app, the URDF and mesh root. It used to be read
# from the hand-written golden bundle fr3wml_app, which was removed on 2026-08-24.
FR3WML_XACRO = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
                'fr5_app/urdf/fr3wml_suction.urdf.xacro')
FR3WML_MESHES = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
                 'fr5_app/meshes')


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
        srdf = os.path.join(tmp, 'ctrl_gen_test_trainit_config', 'config', 'fr3wml.srdf')
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
        app = os.path.join(tmp, 'mockgrip_test_app')
        script = os.path.join(app, 'scripts', 'mock_gripper_action_server.py')
        assert os.path.isfile(script)
        assert 'GripperCommand' in open(script).read()
        bringup = open(os.path.join(app, 'launch', 'bringup.launch.py')).read()
        assert 'GRIPPER_MOCK_ACTION' in bringup
        assert 'mock_gripper_action_server.py' in bringup
        assert 'mock_gripper_action_server.py' in open(os.path.join(app, 'CMakeLists.txt')).read()


GOLDEN_SRDF = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
               'fr3wml_suction_moveit_config/config/fr3wml_suction.srdf')


@pytest.mark.skipif(not os.path.isfile(GOLDEN_SRDF), reason='golden SRDF not present')
def test_controller_import_collision_matrix():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.robot.disable_collisions = None
    n = ctrl.import_collision_matrix_from_srdf(GOLDEN_SRDF)
    assert n == 13
    assert len(ctrl.project.robot.disable_collisions) == 13


# NOTE: test_controller_rebuilds_golden_app was removed on 2026-08-24 together with the
# hand-written golden bundle fr3wml_app. Unlike the other golden-dependent tests it could
# not be re-pointed: it was tied to that bundle by CONTENT -- the hard-coded 'home2' named
# state and pose quaternions were the old bundle's, not any current one.
#
# What it asserted -- "an app built through the controller API equals a known-good bundle"
# -- is now asserted by test_generator_golden.py against fr3wml_suction_tsa_32_bundle, a
# real TSA output validated in Isaac. The controller API itself stays covered by
# test_wizard_mvp_flow.py and the remaining cases here (add_move, add_tool_action,
# set_payload, add_scene_object, set_application, generate).


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
        bundle = os.path.join(tmp, 'fr3wml_wizard_test_bundle')
        srdf = os.path.join(bundle, 'fr3wml_wizard_test_trainit_config', 'config', 'fr3wml.srdf')
        assert os.path.isfile(srdf), wiz.generate_page.result.toPlainText()
        assert 'home2' in open(srdf).read()
        # description geometry copied
        assert os.path.isfile(os.path.join(
            bundle, 'fr3wml_wizard_test_description', 'urdf', 'fr3wml_suction.urdf.xacro'))


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
        srdf = os.path.join(tmp, 'reopened_test_trainit_config', 'config', 'fr3wml.srdf')
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
            tmp, 'ctrl_page_test_trainit_config', 'config', 'ros2_controllers.yaml')))
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
    ctrl.set_payload('cube', [0.02, 0.02, 0.02], attach_offset=[0, 0, 0.01])  # scene defines the payload
    wiz = SetupWizard(ctrl)

    # S5 application type (planner/payload are no longer here: per-waypoint + scene)
    wiz.application_page.app_type.setCurrentText('pick_and_place')
    assert wiz.application_page.validatePage()
    assert ctrl.project.application.type.value == 'pick_and_place'

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
        wiz.generate_page.project_name.setText('fr3wml_wiz')
        wiz.generate_page.output_dir.setText(tmp)
        wiz.generate_page.generate()
        import yaml
        bundle = os.path.join(tmp, 'fr3wml_wiz_bundle')
        bt = os.path.join(bundle, 'fr3wml_wiz_app', 'config', 'bt_params.yaml')
        data = yaml.safe_load(open(bt))['/**']['ros__parameters']['task_parameters']
        assert data['home']['named'] == 'home2'
        assert data['pick']['motion'] == 'lin'
        assert data['pick']['speed'] == 30
        tree = open(os.path.join(bundle, 'fr3wml_wiz_app', 'bt_trees', 'pick_place.xml')).read()
        assert '<CloseGripper/>' in tree and 'AttachObject' in tree


FR30_BASE = ('/home/fra/BIG1500_tending_nesting/src/big1500_digital_twin/'
             'fr30_eef_moveit_config')


@pytest.mark.skipif(not os.path.isdir(FR30_BASE), reason='fr30_eef base not present')
def test_wizard_mvp_flow_offscreen(qapp):
    """Phase 4b: drive the MVP wizard — base config -> scene (mesh + grasp target) ->
    waypoints (per-move attached-check) -> generate a bundle with the flag nodes."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    wiz = SetupWizard(ctrl)

    # Step 2: base config (bootstraps + auto-configures the robot)
    wiz.base_page.project_name.setText('big1500')
    wiz.base_page.base_pkg.setText('fr30_eef_moveit_config')
    wiz.base_page.base_path.setText(FR30_BASE)
    assert wiz.base_page.validatePage()
    assert ctrl.project.robot.robot_name == 'fr30_eef'
    assert 'pre_pick20' in ctrl.robot_summary()['named_states']

    # Step 1/8: scene — a static mesh + a dynamic grasp-target mesh
    sp = wiz.scene_page
    sp.obj_id.setText('bottle_0_0')
    sp.shape.setCurrentText('mesh')
    sp.mesh_resource.setText('package://big1500_isaac/meshes/dynamic/bottle_50cl.stl')
    sp.position.setText('0.5, 0.2, 0.7')
    sp.category.setCurrentText('dynamic')
    sp.grasp_target.setChecked(True)
    sp.release_policy.setCurrentText('freeze')
    sp.add_object()
    bottle = next(o for o in ctrl.project.scene.objects if o.id == 'bottle_0_0')
    assert bottle.is_mesh() and bottle.is_grasp_target()

    # Step 5: application type (planner is per-waypoint; objects come from the scene)
    wiz.application_page.app_type.setCurrentText('pick_and_place')
    assert wiz.application_page.validatePage()

    # Step 8: waypoints (named SRDF states) with a per-move attached-check ON
    wp = wiz.waypoints_page
    for name, acc in [('pick_20', 'off'), ('approach_prewash', 'on')]:
        wp.move_name.setText(name)
        wp.target_mode.setCurrentText('named')
        wp.named.setText(name)
        wp.motion.setCurrentText('free')
        wp.move_planner.setCurrentText('ompl')
        wp.attached_check.setCurrentText(acc)
        if name == 'pick_20':
            wp.tool_cbs['grasp'].setChecked(True)
        else:
            wp.tool_cbs['grasp'].setChecked(False)
            wp.tool_cbs['release'].setChecked(True)
        wp.add_move()
    seg = ctrl.project.application.segment_for('approach_prewash')
    assert seg.attached_collision_check is True

    # Step 9: generate the bundle (generate_page derives bundle names from the project
    # name via BundleSpec.from_prefix -> app='big1500_app', config='big1500_trainit_config')
    with tempfile.TemporaryDirectory() as tmp:
        wiz.generate_page.initializePage()
        wiz.generate_page.project_name.setText('big1500')
        wiz.generate_page.output_dir.setText(tmp)
        wiz.generate_page.generate()
        bundle = os.path.join(tmp, 'big1500_bundle')
        tree = open(os.path.join(bundle, 'big1500_app', 'bt_trees', 'pick_place.xml')).read()
        assert '<SetAttachedCollisionCheck value="true"/>' in tree
        assert '<SetReleasePolicy policy="freeze"/>' in tree
        assert os.path.isfile(os.path.join(
            bundle, 'big1500_trainit_config', 'config', 'fr30_eef.srdf'))
        # bundle flow: the app bringup INCLUDES the trainit_config cell bringup
        bringup = open(os.path.join(bundle, 'big1500_app', 'launch', 'bringup.launch.py')).read()
        assert 'IncludeLaunchDescription' in bringup
        assert 'big1500_trainit_config' in bringup


USD_SCENE = ('/home/fra/BIG1500_tending_nesting/src/big1500_digital_twin/big1500_isaac/'
             'usd/scenes/big1500_tending_nesting_prewash.usd')


@pytest.mark.skipif(not (os.path.isdir(FR30_BASE) and os.path.isfile(USD_SCENE)),
                    reason='fr30 base or USD not present')
def test_wizard_usd_mapping_builds_baseline_scene(qapp):
    """Step 2 from the USD: load USD -> auto-suggested mesh mapping table -> apply ->
    the scene reproduces the baseline (23 mesh objects, bottles as grasp targets)."""
    try:
        from trainit_setup_assistant.importers.usd_scene import usd_available
    except Exception:
        pytest.skip('usd_scene import failed')
    if not usd_available():
        pytest.skip('pxr / usd-core not available')
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    wiz = SetupWizard(ctrl)
    wiz.base_page.project_name.setText('big1500')
    wiz.base_page.base_pkg.setText('fr30_eef_moveit_config')
    wiz.base_page.base_path.setText(FR30_BASE)
    assert wiz.base_page.validatePage()

    sp = wiz.scene_page
    sp.usd_path.setText(USD_SCENE)
    sp.mesh_pkg.setText('big1500_isaac')
    sp.load_usd_mapping()
    # table filled; bottles + crate + prewash + belt auto-suggested; BW_0080_context off
    assert sp.map_table.rowCount() >= 4
    groups = {sp.map_table.item(i, 1).text().split()[0] for i in range(sp.map_table.rowCount())}
    assert {'bottle', 'crate', 'prewash_station', 'belt_conveyor'} <= groups
    sp.apply_usd_mapping_table()
    ids = {o.id for o in ctrl.project.scene.objects}
    assert 'bottle_0_0' in ids and 'crate_00' in ids and 'prewash_station' in ids
    assert len(ctrl.project.scene.objects) == 23        # BW_0080_context excluded
    # everything stays a MESH (no per-shape primitive guessing); a grasped object is
    # removed/attached-as-box at RUNTIME, not converted here. dims holds the AABB extents.
    bottle = next(o for o in ctrl.project.scene.objects if o.id == 'bottle_0_0')
    assert bottle.is_grasp_target() and bottle.is_mesh()
    assert bottle.mesh_resource.endswith('dynamic/bottle_50cl.stl')
    assert len(bottle.dims) == 3 and abs(bottle.dims[2] - 0.2445) < 1e-3   # AABB for the box
    prewash = next(o for o in ctrl.project.scene.objects if o.id == 'prewash_station')
    assert prewash.is_mesh()
    assert prewash.mesh_resource.endswith('prewash_station/base_collision.stl')
    assert ctrl.project.scene.grasp_attach_mode == 'attach_box'   # default: universal AABB cuboid


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


def test_blocks_page_builds_app_with_wait_and_loop(qapp):
    """v3 UX: drive the block editor (palette|sequence|inspector) offscreen — moves +
    gripper + wait + loop fold into the canonical model and the BT gets Sleep/Repeat."""
    from trainit_setup_assistant.applications.registry import get_application
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.clear_application()
    ctrl.set_payload('cube', [0.02, 0.02, 0.02])
    wiz = SetupWizard(ctrl)
    bp = wiz.blocks_page

    # move "home" (named) -> gripper close -> wait 250ms -> move "place" (tcp)
    # -> gripper open -> loop forever
    bp.add_block('move')
    bp.m_name.setText('home')
    bp.m_target.setCurrentIndex(0)
    bp.m_named.setText('home')
    bp.m_motion.setCurrentText('ptp')
    bp.apply_inspector()

    bp.add_block('gripper')
    bp.g_action.setCurrentIndex(0)          # close (grasp)
    bp.apply_inspector()

    bp.add_block('wait')
    bp.w_ms.setValue(250)
    bp.apply_inspector()

    bp.add_block('move')
    bp.m_name.setText('place')
    bp.m_target.setCurrentIndex(1)          # tcp
    bp.m_pos.setText('0.3, 0.3, 0.1')
    bp.m_quat.setText('0, 1, 0, 0')
    bp.m_motion.setCurrentText('lin')
    bp.m_planner.setCurrentText('pilz')
    bp.m_check.setCurrentText('on')
    bp.apply_inspector()

    bp.add_block('gripper')
    bp.g_action.setCurrentIndex(1)          # open (release)
    bp.apply_inspector()

    bp.add_block('reset')                   # scene reset at the cycle boundary
    bp.add_block('loop')                    # forever (default)
    bp.apply_inspector()
    assert bp.validatePage()

    app = ctrl.project.application
    assert app.sequence == ['home', 'place']
    assert app.segment_for('home').wait_after_ms == 250
    assert app.segment_for('place').attached_collision_check is True
    assert app.loop_cycles == -1
    kinds = [a.kind.value for a in app.tool_actions]
    assert kinds == ['grasp', 'release', 'reset_scene']

    xml = get_application(app.type).build_tree_xml(ctrl.project)
    assert '<Sleep msec="250"/>' in xml
    assert '<Repeat num_cycles="-1">' in xml
    assert xml.index('<Repeat') < xml.index('<MoveWaypoint')
    # scene reset renders INSIDE the loop, as the cycle boundary (after the release)
    assert '<ResetScene/>' in xml
    assert xml.index('<ResetScene/>') > xml.index('OpenGripper')
    assert xml.index('<ResetScene/>') < xml.index('</Repeat>')

    # a second loop block is refused; reload round-trip rebuilds the same blocks
    bp.add_block('loop')
    assert sum(1 for b in bp.blocks if b['kind'] == 'loop') == 1
    bp.blocks = []
    bp.initializePage()
    rebuilt = [b['kind'] for b in bp.blocks]
    assert rebuilt == ['move', 'gripper', 'wait', 'move', 'gripper', 'reset', 'loop']


def test_wizard_vision_flow_offscreen(qapp):
    """TSA v4: Perception step -> vision app -> Vision block -> generated bundle."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    wiz = SetupWizard(ctrl)

    # Step 5 — Perception: tune + save a detector (no live session: form only)
    pp = wiz.perception_page
    pp.det_name.setText('cube')
    pp.p_class.setText('cube')
    pp.h_lo.setValue(170); pp.h_hi.setValue(10)
    pp.c_replace.setText('$(find x)/a.xacro')
    pp.c_with.setText('$(find x)/a_camera.xacro')
    pp._add_detector()
    assert pp.validatePage()
    per = ctrl.project.perception
    assert per.detectors[0].name == 'cube'
    assert per.detectors[0].params['h'] == [170, 10]
    assert per.camera.with_include == '$(find x)/a_camera.xacro'

    # Step 6 — Application: vision app suggested (detectors exist) and applied
    wiz.application_page.initializePage()
    assert wiz.application_page.validatePage()
    assert ctrl.project.application.type.value == 'vision_guided_motion'

    # Step 7 — blocks (D-016): vision is a PROPERTY of the Move block
    bp = wiz.blocks_page
    bp.blocks = [
        {'kind': 'move', 'name': 'approach', 'target': 'named', 'named': 'ready',
         'pos': '', 'quat': '0, 0, 0, 1', 'motion': 'free', 'planner': 'ompl',
         'speed': 50, 'tol': 0.1, 'check': 'inherit',
         'detector': 'cube', 'vdx': 0.0, 'vdy': 0.0, 'vdz': 0.08,
         'vori': 'same', 'vori_ref': 'pick'},
        {'kind': 'move', 'name': 'pick', 'target': 'tcp', 'pos': '0.5, 0.0, 0.03',
         'quat': '-0.707, 0.707, 0, 0', 'named': '', 'motion': 'lin', 'planner': 'pilz',
         'speed': 50, 'tol': 0.1, 'check': 'inherit',
         'detector': 'cube', 'vdx': 0.01, 'vdy': 0.0, 'vdz': 0.006,
         'vori': 'keep', 'vori_ref': ''},
    ]
    bp._sync_model()
    app = ctrl.project.application
    assert app.waypoint_by_name('pick').vision.dz == 0.006
    assert app.waypoint_by_name('pick').vision.dx == 0.01
    assert app.waypoint_by_name('approach').vision.orientation == 'from:pick'
    # the automatic detect row is visible and names the detector
    bp._refresh()
    assert bp.auto_detect.isVisible() or 'cube' in bp.auto_detect.text()

    # generate: the bundle carries perception.yaml + the vision tree lines
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('vision_gui_test')
        ctrl.generate(tmp)
        appdir = os.path.join(tmp, 'vision_gui_test_app')
        per_yaml = open(os.path.join(appdir, 'config', 'perception.yaml')).read()
        assert 'cube' in per_yaml and 'color_mask' in per_yaml
        tree = open(os.path.join(appdir, 'bt_trees', 'pick_place.xml')).read()
        assert 'DetectObject detector="cube"' in tree
        assert 'SetWaypointFromDetection waypoint="approach"' in tree
        launch = open(os.path.join(appdir, 'launch', 'trainit_bt.launch.py')).read()
        assert 'detector_node' in launch and 'detector_cube' in launch


def test_blocks_page_preserves_golden_loop_start_and_tree(qapp):
    """Regression (verification blocker): merely ENTERING Step 7 on a reopened
    project must not fold loop_start away — the regenerated tree must still equal
    the golden's shape (Repeat opening at pre_pick, detection inside the cycle)."""
    GOLDEN_PROJECT = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
                      'fr3wml_suction_tsa_32_bundle/project.yaml')
    if not os.path.isfile(GOLDEN_PROJECT):
        pytest.skip('golden bundle not present')
    from trainit_setup_assistant.applications import get_application
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(GOLDEN_PROJECT)
    before = get_application(ctrl.project.application.type).build_tree_xml(ctrl.project)
    wiz = SetupWizard(ctrl)
    wiz.blocks_page.initializePage()           # enter Step 7 (this used to null it)
    assert wiz.blocks_page.validatePage()
    assert ctrl.project.application.loop_start == 'pre_pick'
    after = get_application(ctrl.project.application.type).build_tree_xml(ctrl.project)
    from trainit_setup_assistant.verify.equivalence import tree_diffs
    assert not tree_diffs(after, before), tree_diffs(after, before)[:5]


def test_blocks_page_no_edit_apply_is_identity_on_the_golden(qapp):
    """v4.1 regression: selecting a vision-bound move and clicking Apply with NO
    edits must not mutate the model (the auto free/OMPL switch fires only when the
    detector is NEWLY set; a literal orientation would ride vori_raw)."""
    GOLDEN_PROJECT = ('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
                      'fr3wml_suction_tsa_32_bundle/project.yaml')
    if not os.path.isfile(GOLDEN_PROJECT):
        pytest.skip('golden bundle not present')
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(GOLDEN_PROJECT)
    wiz = SetupWizard(ctrl)
    bp = wiz.blocks_page
    bp.initializePage()
    before = ctrl.project.model_dump(mode='json')
    row = next(i for i, b in enumerate(bp.blocks)
               if b['kind'] == 'move' and b.get('detector'))
    bp.seq.setCurrentRow(row)
    bp.apply_inspector()
    assert bp.validatePage()
    assert ctrl.project.model_dump(mode='json') == before


def test_literal_quaternion_binding_survives_step7(qapp):
    """A hand-edited literal orientation must survive page entry AND an untouched
    Apply (it rides the block's vori_raw passthrough)."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.upsert_detector('red_cube', params={'class_id': 'cube'})
    ctrl.add_move('pick', position=[0.5, 0.0, 0.03],
                  orientation=[-0.707, 0.707, 0.0, 0.0], motion='lin', planner='pilz')
    ctrl.bind_vision('pick', 'red_cube', dz=0.006, orientation='0;0;0;1')
    wiz = SetupWizard(ctrl)
    bp = wiz.blocks_page
    bp.initializePage()
    assert ctrl.project.application.waypoint_by_name('pick').vision.orientation == '0;0;0;1'
    row = next(i for i, b in enumerate(bp.blocks)
               if b['kind'] == 'move' and b['name'] == 'pick')
    bp.seq.setCurrentRow(row)
    bp.apply_inspector()
    assert bp.validatePage()
    assert ctrl.project.application.waypoint_by_name('pick').vision.orientation == '0;0;0;1'


def test_blocks_page_folds_and_reopens_relative_steps(qapp):
    """D-017: a Move with 'Relative to step' folds into Waypoint.relative and a
    reopened project rebuilds the block fields."""
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.upsert_detector('red_cube', params={'class_id': 'cube'})
    wiz = SetupWizard(ctrl)
    bp = wiz.blocks_page
    base_move = {'kind': 'move', 'target': 'tcp', 'named': '',
                 'quat': '-0.707, 0.707, 0, 0', 'motion': 'lin', 'planner': 'pilz',
                 'speed': 50, 'tol': 0.1, 'check': 'inherit', 'detector': '',
                 'vdx': 0.0, 'vdy': 0.0, 'vdz': 0.0, 'vori': 'keep', 'vori_ref': '',
                 'vori_raw': '', 'rel_step': '', 'rdx': 0.0, 'rdy': 0.0, 'rdz': 0.0,
                 'rroll': 0.0, 'rpitch': 0.0, 'ryaw': 0.0}
    pick = dict(base_move, name='pick', pos='0.5, 0.0, 0.03',
                detector='red_cube', vdz=0.006)
    post = dict(base_move, name='post_pick', pos='0.5, 0.0, 0.11',
                rel_step='pick', rdz=0.08, ryaw=45.0)
    bp.blocks = [pick, post]
    bp._sync_model()
    wp = ctrl.project.application.waypoint_by_name('post_pick')
    assert wp.relative.step == 'pick' and wp.relative.dz == 0.08
    assert wp.relative.dyaw == 45.0 and wp.vision is None
    # reopen: the block dict comes back with the same fields
    bp.blocks = []
    bp.initializePage()
    rb = next(b for b in bp.blocks if b.get('name') == 'post_pick')
    assert rb['rel_step'] == 'pick' and rb['rdz'] == 0.08 and rb['ryaw'] == 45.0
