"""Phase 3: the standalone scene+planner moveit_config emitter + scene.yaml + the
app tree wiring the per-move dynamic-object nodes.

Uses the real hand-made base ``fr30_eef_moveit_config`` as the copy source when it is
present (skips otherwise), so the test also proves the copied config is self-contained.
"""

import os
import tempfile

import pytest

from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.generator.orchestrator import (
    Orchestrator, generate_scene_loader_config)
from trainit_setup_assistant.generator.scene_yaml import build_scene_yaml
from trainit_setup_assistant.model import SceneObject

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')
BASE = ('/home/fra/BIG1500_tending_nesting/src/big1500_digital_twin/'
        'fr30_eef_moveit_config')


def _project_with_base():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    p = ctrl.project
    p.robot.robot_name = 'fr30_eef'
    p.robot.base_moveit_config_package = 'fr30_eef_moveit_config'
    p.robot.base_moveit_config_path = BASE
    p.bundle.moveit_config_package = 'big1500_trainit_config'
    p.bundle.app_package = 'big1500_app'
    p.scene.touch_links = ['end_effector', 'tcp', 'wrist3_link']
    p.scene.objects = [
        SceneObject(id='prewash_station', shape='mesh',
                    mesh_resource='package://big1500_isaac/meshes/prewash_station/base_collision.stl',
                    category='static', position=[0.58, -0.60, 1.57]),
        SceneObject(id='crate_00', shape='mesh',                       # dynamic, NOT grasped
                    mesh_resource='package://big1500_isaac/meshes/dynamic/crate_20x50cl.stl',
                    category='dynamic', position=[0.5, 0.2, 0.6]),
        SceneObject(id='bottle_0_0', shape='mesh',                     # dynamic grasp target
                    mesh_resource='package://big1500_isaac/meshes/dynamic/bottle_50cl.stl',
                    category='dynamic', grasp_target=True, release_policy='freeze',
                    position=[0.5, 0.2, 0.7]),
    ]
    # per-move flag: turn the check ON for the transfer into pre_place
    for seg in p.application.segments:
        if seg.to_waypoint == 'pre_place':
            seg.attached_collision_check = True
    return ctrl, p


# ---- scene.yaml generation (no base needed) ----
def test_scene_yaml_has_meshes_and_attach_ids():
    _, p = _project_with_base()
    y = build_scene_yaml(p)
    assert 'type: mesh' in y
    assert 'mesh_path: "package://big1500_isaac/meshes/dynamic/bottle_50cl.stl"' in y
    assert 'attach_object_ids: [bottle_0_0]' in y            # only the grasp target
    assert 'object_ids: [prewash_station, crate_00, bottle_0_0]' in y
    assert 'dynamic: true' in y and 'dynamic: false' in y     # crate/bottle vs prewash
    import yaml
    parsed = yaml.safe_load(y)['scene_manager_node']['ros__parameters']
    assert parsed['frame_id'] == 'base_link'
    assert parsed['objects']['prewash_station']['type'] == 'mesh'


# ---- app tree wires the new per-move nodes + no in-tree AddCollisionObject ----
def test_app_tree_emits_flag_nodes_and_skips_add_collision():
    ctrl, p = _project_with_base()
    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(p, tmp)
        tree = open(os.path.join(tmp, 'big1500_app', 'bt_trees', 'pick_place.xml')).read()
        assert '<SetAttachedCollisionCheck value="true"/>' in tree     # per-move flag
        assert '<SetReleasePolicy policy="freeze"/>' in tree           # before OpenGripper
        assert '<OpenGripper/>' in tree
        # scene loader owns the obstacles -> the tree must NOT re-add them
        assert 'AddCollisionObject' not in tree


# ---- the standalone config: copies the base + adds scene + self-referential bringup ----
@pytest.mark.skipif(not os.path.isdir(BASE), reason='base fr30_eef_moveit_config not present')
def test_bundle_trainit_config_is_standalone():
    ctrl, p = _project_with_base()
    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(p, tmp)
        cfg = os.path.join(tmp, 'big1500_trainit_config')
        # base config copied verbatim (SRDF + urdf + controllers)
        assert os.path.isfile(os.path.join(cfg, 'config', 'fr30_eef.srdf'))
        assert os.path.isfile(os.path.join(cfg, 'config', 'moveit_controllers.yaml'))
        # generated scene.yaml + self-referential bringup
        assert os.path.isfile(os.path.join(cfg, 'config', 'scene.yaml'))
        bringup = open(os.path.join(cfg, 'launch', 'bringup.launch.py')).read()
        assert 'MOVEIT_CONFIG_PACKAGE = "big1500_trainit_config"' in bringup   # self
        assert 'scene_manager_node' in bringup
        assert 'fr30_eef_moveit_config' not in bringup                          # independent
        # the copied URDF's only external $(find …) is a connector pkg, not a moveit_config
        urdf = open(os.path.join(cfg, 'config', 'fr30_eef.urdf.xacro')).read()
        assert 'fr30_eef_moveit_config' not in urdf


@pytest.mark.skipif(not os.path.isdir(BASE), reason='base fr30_eef_moveit_config not present')
def test_step3_intermediate_config():
    ctrl, p = _project_with_base()
    with tempfile.TemporaryDirectory() as tmp:
        generate_scene_loader_config(p, tmp, 'fr30_eef_scene_loader_moveit_config')
        cfg = os.path.join(tmp, 'fr30_eef_scene_loader_moveit_config')
        assert os.path.isfile(os.path.join(cfg, 'config', 'scene.yaml'))
        assert os.path.isfile(os.path.join(cfg, 'launch', 'bringup.launch.py'))
        assert os.path.isfile(os.path.join(cfg, 'package.xml'))


BASELINE_SCENE = ('/home/fra/BIG1500_tending_nesting/src/big1500_digital_twin/'
                  'fr30_eef_scene_loader_moveit_config/config/scene.yaml')


@pytest.mark.skipif(not (os.path.isdir(BASE) and os.path.isfile(BASELINE_SCENE)),
                    reason='baseline scene_loader config not present')
def test_import_scene_yaml_reproduces_baseline():
    """The acceptance criterion: import the baseline scene.yaml -> generate the config ->
    its scene.yaml is functionally identical to the hand-made baseline."""
    import yaml
    from trainit_setup_assistant.generator.scene_yaml import build_scene_yaml
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.load_base_moveit_config('fr30_eef_moveit_config', BASE)
    n = ctrl.import_scene_yaml(BASELINE_SCENE)
    assert n == 23
    got = yaml.safe_load(build_scene_yaml(ctrl.project))['scene_manager_node']['ros__parameters']
    want = yaml.safe_load(open(BASELINE_SCENE))['scene_manager_node']['ros__parameters']
    assert got['object_ids'] == want['object_ids']
    assert got['attach_object_ids'] == want['attach_object_ids']       # only the bottles
    assert got['touch_links'] == want['touch_links']
    for oid in want['object_ids']:
        a, b = got['objects'][oid], want['objects'][oid]
        assert a['type'] == b['type'] == 'mesh'
        assert a['mesh_path'] == b['mesh_path']
        assert a['dynamic'] == b['dynamic']
        assert [round(x, 4) for x in a['position']] == [round(x, 4) for x in b['position']]
        assert [round(x, 4) for x in a['orientation']] == [round(x, 4) for x in b['orientation']]


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
