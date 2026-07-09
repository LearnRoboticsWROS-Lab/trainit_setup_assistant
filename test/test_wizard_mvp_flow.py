"""Phase 4: the controller drives the MVP flow — load the hand-made base config,
configure the scene + application (per-move flags, grasp targets), and generate the
standalone bundle. Uses the real fr30_eef_moveit_config as the base (skips otherwise).
"""

import os
import tempfile

import pytest

from trainit_setup_assistant.gui.controller import AssistantController

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')
BASE = ('/home/fra/BIG1500_tending_nesting/src/big1500_digital_twin/'
        'fr30_eef_moveit_config')

pytestmark = pytest.mark.skipif(not os.path.isdir(BASE),
                                reason='base fr30_eef_moveit_config not present')


def _configured():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    # Step 2: load the hand-made base -> auto-config group/frames/named-states/controllers
    info = ctrl.load_base_moveit_config('fr30_eef_moveit_config', BASE)
    ctrl.set_project_name('big1500', derive_bundle=False)
    ctrl.project.bundle.moveit_config_package = 'big1500_trainit_config'
    ctrl.project.bundle.app_package = 'big1500_app'
    # Step 5
    ctrl.set_mode('isaac')
    # scene: a static mesh + a dynamic grasp-target mesh
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('prewash_station', [1, 1, 1], [0.58, -0.60, 1.57], shape='mesh',
                          mesh_resource='package://big1500_isaac/meshes/prewash_station/base_collision.stl',
                          category='static')
    ctrl.add_scene_object('bottle_0_0', [1, 1, 1], [0.5, 0.2, 0.7], shape='mesh',
                          mesh_resource='package://big1500_isaac/meshes/dynamic/bottle_50cl.stl',
                          category='dynamic')
    ctrl.set_object_grasp('bottle_0_0', grasp_target=True, release_policy='freeze')
    ctrl.set_scene_loader_params(touch_links=['end_effector', 'tcp', 'wrist3_link'])
    # Step 8: application with per-move flags + grasp/release
    ctrl.clear_application()
    ctrl.set_application('pick_and_place', 'pilz')
    ctrl.add_move('pre_pick20', named='pre_pick20', motion='free', planner='ompl')
    ctrl.add_move('pick_20', named='pick_20', motion='free', planner='ompl',
                  attached_collision_check=False)
    ctrl.add_tool_action('pick_20', 'grasp')
    ctrl.add_move('pre_pick20', named='pre_pick20', motion='free', planner='ompl',
                  attached_collision_check=False)          # lift clear (flag still OFF)
    ctrl.add_move('approach_prewash', named='approach_prewash', motion='free', planner='ompl',
                  attached_collision_check=True)            # held transfer -> avoid prewash
    ctrl.add_move('place_prewash', named='place_prewash', motion='lin', planner='pilz')
    ctrl.add_tool_action('place_prewash', 'release')
    return ctrl, info


def test_base_config_autoconfig():
    ctrl, info = _configured()
    assert info['robot_name'] == 'fr30_eef'
    assert info['group'] == 'fr30_eef_group'
    # named states extracted from the base SRDF
    assert 'pre_pick20' in info['named_states'] and 'place_prewash' in info['named_states']
    # controllers extracted from moveit_controllers.yaml
    assert ctrl.project.robot.arm_controller.name == 'moveit_joint_controller'
    assert ctrl.project.robot.gripper.controller_name == 'softgripper_controller'
    assert ctrl.project.robot.base_moveit_config_path == BASE


def test_step3_intermediate_config_from_controller():
    ctrl, _ = _configured()
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.generate_scene_loader_config(tmp)
        pkg = os.path.join(tmp, 'fr30_eef_scene_loader_moveit_config')
        assert os.path.isfile(os.path.join(pkg, 'config', 'scene.yaml'))
        assert os.path.isfile(os.path.join(pkg, 'launch', 'bringup.launch.py'))


def test_full_bundle_from_controller_has_per_move_flags():
    ctrl, _ = _configured()
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.generate(tmp)
        tree = open(os.path.join(tmp, 'big1500_app', 'bt_trees', 'pick_place.xml')).read()
        # per-move flags emitted automatically
        assert '<SetAttachedCollisionCheck value="true"/>' in tree
        assert '<SetReleasePolicy policy="freeze"/>' in tree
        # the standalone config was copied from the base
        assert os.path.isfile(os.path.join(tmp, 'big1500_trainit_config', 'config', 'fr30_eef.srdf'))
        assert os.path.isfile(os.path.join(tmp, 'big1500_trainit_config', 'config', 'scene.yaml'))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
