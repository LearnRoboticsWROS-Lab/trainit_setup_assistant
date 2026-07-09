"""M8 (scene primitives) + M9 (USD import)."""

import os
import tempfile

import pytest

from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.importers import usd_available

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')


# ---- M8: scene primitives ----
def test_scene_box_becomes_collision_object():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('table', [2.0, 2.0, 0.10], [0.0, 0.0, -0.08])
    ctrl.add_scene_object('part', [0.02, 0.02, 0.02], [0.5, 0.0, 0.05], dynamic=True)
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('scene_test')
        ctrl.generate(tmp)
        tree = open(os.path.join(tmp, 'scene_test', 'bt_trees', 'pick_place.xml')).read()
        # static collision object emitted; dynamic excluded from planning
        assert 'AddCollisionObject id="table"' in tree
        assert 'id="part"' not in tree


def test_non_box_shape_emits_aabb_with_warning():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('ball', [0.1], [0.3, 0.0, 0.2], shape='sphere')  # r=0.1
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('aabb_test')
        manifest = ctrl.generate(tmp)
        import yaml
        bt = os.path.join(tmp, 'aabb_test', 'config', 'bt_params.yaml')
        params = yaml.safe_load(open(bt))['/**']['ros__parameters']['task_parameters']
        assert params['scene_ball_dims'] == [0.2, 0.2, 0.2]   # sphere AABB
        assert any('box-only' in w for w in manifest.warnings)


# ---- M9: USD import ----
@pytest.mark.skipif(not usd_available(), reason='pxr/usd-core not available')
def test_import_usd_scene():
    from pxr import Usd, UsdGeom, Gf
    with tempfile.TemporaryDirectory() as tmp:
        usd_path = os.path.join(tmp, 'scene.usda')
        stage = Usd.Stage.CreateNew(usd_path)
        # two cubes at different places/sizes
        c1 = UsdGeom.Cube.Define(stage, '/World/box_a')
        c1.GetSizeAttr().Set(0.4)
        UsdGeom.XformCommonAPI(c1).SetTranslate(Gf.Vec3d(1.0, 0.0, 0.2))
        c2 = UsdGeom.Cube.Define(stage, '/World/box_b')
        c2.GetSizeAttr().Set(0.2)
        UsdGeom.XformCommonAPI(c2).SetTranslate(Gf.Vec3d(-0.5, 0.5, 0.1))
        stage.GetRootLayer().Save()

        ctrl = AssistantController()
        ctrl.open_project(EXAMPLE)
        ctrl.project.scene.objects = []
        n = ctrl.import_usd_scene(usd_path)
        assert n == 2
        ids = {o.id for o in ctrl.project.scene.objects}
        assert 'box_a' in ids and 'box_b' in ids
        box_a = next(o for o in ctrl.project.scene.objects if o.id == 'box_a')
        # cube size 0.4 -> AABB ~0.4; centre ~ (1.0, 0, 0.2)
        assert abs(box_a.dims[0] - 0.4) < 1e-6
        assert abs(box_a.position[0] - 1.0) < 1e-6
        # generates a collision object
        with tempfile.TemporaryDirectory() as out:
            ctrl.set_project_name('usd_test')
            ctrl.generate(out)
            tree = open(os.path.join(out, 'usd_test', 'bt_trees', 'pick_place.xml')).read()
            assert 'AddCollisionObject id="box_a"' in tree


BIN_USD = ('/home/fra/fr5_ws/install/fr3wml_isaac/share/fr3wml_isaac/'
           'usd/scenes/bin_picking_cell.usd')


@pytest.mark.skipif(not (usd_available() and os.path.isfile(BIN_USD)),
                    reason='bin_picking_cell.usd not present')
def test_usd_import_aligns_to_robot_base():
    """Cell-world USD coords must be expressed in the robot base frame."""
    from trainit_setup_assistant.importers import import_usd
    # raw (no base prim) -> stage world frame: the Cube sits ~1.07 m up (on the table)
    raw = import_usd(BIN_USD)
    cube_raw = next((o for o in raw if o.id.lower().startswith('cube')), None)
    assert cube_raw is not None
    assert cube_raw.position[2] > 1.0          # world frame, far above base

    # aligned to the robot prim -> the Cube lands near the flange (~0.01 m), not 1.07
    aligned = import_usd(BIN_USD, base_prim='/fr3wml_suction')
    cube = next(o for o in aligned if o.id.lower().startswith('cube'))
    assert abs(cube.position[2] - 0.014) < 0.05

    # auto-detect via the robot name hint gives the same alignment
    auto = import_usd(BIN_USD, robot_hint='fr3wml')
    cube_auto = next(o for o in auto if o.id.lower().startswith('cube'))
    assert abs(cube_auto.position[2] - cube.position[2]) < 1e-6


# ---- 3-way cell classification (static / actuated / dynamic) ----
def test_category_drives_collision_role():
    from trainit_setup_assistant.model import SceneObject
    from trainit_setup_assistant.model.enums import SceneObjectCategory
    static = SceneObject(id='machine', category=SceneObjectCategory.STATIC)
    actuated = SceneObject(id='belt', category=SceneObjectCategory.ACTUATED)
    dyn = SceneObject(id='bottle', category=SceneObjectCategory.DYNAMIC)
    # static + actuated are CHECKED collisions; dynamic is collision-allowed
    assert static.is_planning_collision() and not static.is_dynamic()
    assert actuated.is_planning_collision() and actuated.is_actuated()
    assert dyn.is_dynamic() and not dyn.is_planning_collision()


def test_legacy_dynamic_flag_infers_category_and_roundtrips():
    from trainit_setup_assistant.model import SceneObject
    from trainit_setup_assistant.model.enums import SceneObjectCategory
    o = SceneObject(id='part', dynamic=True)          # legacy: only `dynamic` given
    assert o.category is SceneObjectCategory.DYNAMIC
    assert SceneObject.model_validate(o.model_dump()) == o   # idempotent round-trip
    s = SceneObject(id='wall')                        # defaults -> static
    assert s.category is SceneObjectCategory.STATIC and not s.dynamic


def test_actuated_checked_dynamic_allowed_in_tree():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('belt', [1.0, 0.4, 0.3], [0.5, 0.0, 0.0], category='actuated')
    ctrl.add_scene_object('crate', [0.3, 0.2, 0.3], [0.4, 0.0, 0.1], category='dynamic')
    ctrl.set_object_category('belt', 'actuated')      # re-classify path
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('cat_test')
        ctrl.generate(tmp)
        tree = open(os.path.join(tmp, 'cat_test', 'bt_trees', 'pick_place.xml')).read()
        assert 'AddCollisionObject id="belt"' in tree   # actuated -> checked collision
        assert 'id="crate"' not in tree                 # dynamic -> collision-allowed


@pytest.mark.skipif(not usd_available(), reason='pxr/usd-core not available')
def test_import_classify_assigns_dynamic():
    from pxr import Usd, UsdGeom, Gf
    from trainit_setup_assistant.importers import import_usd
    from trainit_setup_assistant.model.enums import SceneObjectCategory
    with tempfile.TemporaryDirectory() as tmp:
        usd_path = os.path.join(tmp, 'cell.usda')
        stage = Usd.Stage.CreateNew(usd_path)
        b = UsdGeom.Cube.Define(stage, '/World/bottle_0'); b.GetSizeAttr().Set(0.1)
        UsdGeom.XformCommonAPI(b).SetTranslate(Gf.Vec3d(0.4, 0.0, 0.1))
        m = UsdGeom.Cube.Define(stage, '/World/machine'); m.GetSizeAttr().Set(0.5)
        UsdGeom.XformCommonAPI(m).SetTranslate(Gf.Vec3d(-0.6, 0.0, 0.2))
        stage.GetRootLayer().Save()
        objs = import_usd(usd_path,
                          classify=lambda name, path: 'dynamic'
                          if name.startswith('bottle') else None)
        by = {o.id: o for o in objs}
        assert by['bottle_0'].is_dynamic()                              # classified
        assert by['machine'].category is SceneObjectCategory.STATIC     # default


# ---- MVP v1: mesh objects + per-dynamic attributes + per-move flag ----
def test_mesh_object_and_dynamic_attributes_roundtrip():
    from trainit_setup_assistant.model import (
        IsaacGraspMethod, ReleasePolicy, SceneObject,
    )
    # a grasp-target bottle: dynamic mesh that attaches, freezes on release
    bottle = SceneObject(id='bottle_0', shape='mesh',
                         mesh_resource='package://p/bottle.stl', category='dynamic',
                         grasp_target=True, release_policy='freeze',
                         touchable_collision_ids=['prewash_station'])
    assert bottle.is_mesh() and bottle.is_dynamic() and bottle.is_grasp_target()
    assert not bottle.is_box()
    assert bottle.release_policy is ReleasePolicy.FREEZE
    assert bottle.isaac_grasp_method is IsaacGraspMethod.FIXED_JOINT   # default
    assert bottle.touchable_collision_ids == ['prewash_station']
    assert SceneObject.model_validate(bottle.model_dump(mode='json')) == bottle

    # the crate: dynamic (collision-allowed) but NOT a grasp target -> never attached
    crate = SceneObject(id='crate', shape='mesh',
                        mesh_resource='package://p/crate.stl', category='dynamic')
    assert crate.is_dynamic() and not crate.is_grasp_target()

    # a static mesh (machine) is a CHECKED collision object
    machine = SceneObject(id='big1500', shape='mesh',
                          mesh_resource='package://p/machine.stl', category='static')
    assert machine.is_planning_collision() and not machine.is_dynamic()


def test_segment_attached_collision_check_flag():
    from trainit_setup_assistant.model import MotionSegment
    on = MotionSegment(to_waypoint='approach_prewash', attached_collision_check=True)
    off = MotionSegment(to_waypoint='pick', attached_collision_check=False)
    default = MotionSegment(to_waypoint='home')
    assert on.attached_collision_check is True
    assert off.attached_collision_check is False
    assert default.attached_collision_check is None   # inherit runtime state
    assert MotionSegment.model_validate(on.model_dump()) == on


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
