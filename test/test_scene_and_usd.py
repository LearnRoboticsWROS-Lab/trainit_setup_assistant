"""M8 (scene primitives) + M9 (USD import)."""

import json
import os
import tempfile

import pytest

from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.importers import usd_available
import identity_cases
import ws_paths

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')


# ---- M8: scene primitives ----
def test_scene_box_becomes_collision_object():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('table', [2.0, 2.0, 0.10], [0.0, 0.0, -0.08])
    ctrl.add_scene_object('part', [0.02, 0.02, 0.02], [0.5, 0.0, 0.05], dynamic=True)
    identity_cases.require('p2_example')
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('scene_test')
        ctrl.generate(tmp)
        tree = open(os.path.join(tmp, 'scene_test_app', 'bt_trees', 'pick_place.xml')).read()
        # static collision object emitted; dynamic excluded from planning
        assert 'AddCollisionObject id="table"' in tree
        assert 'id="part"' not in tree


def test_non_box_shape_emits_aabb_with_warning():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('ball', [0.1], [0.3, 0.0, 0.2], shape='sphere')  # r=0.1
    identity_cases.require('p2_example')
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('aabb_test')
        manifest = ctrl.generate(tmp)
        import yaml
        bt = os.path.join(tmp, 'aabb_test_app', 'config', 'bt_params.yaml')
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
        identity_cases.require('p2_example')
        with tempfile.TemporaryDirectory() as out:
            ctrl.set_project_name('usd_test')
            ctrl.generate(out)
            tree = open(os.path.join(out, 'usd_test_app', 'bt_trees', 'pick_place.xml')).read()
            assert 'AddCollisionObject id="box_a"' in tree


BIN_USD = str(ws_paths.installed_share('fr3wml_isaac', 'usd', 'scenes',
                                      'bin_picking_cell.usd'))


@pytest.mark.skipif(not usd_available(), reason='pxr/usd-core not available')
def test_usd_import_aligns_to_robot_base():
    """Cell-world USD coords must be expressed in the robot base frame.

    vNext M0C #6. This used to read the reference cell's installed stage and assert that the
    RAW import put the Cube "~1.07 m up (on the table)". That stage was re-authored with the
    robot at the stage origin, so raw and aligned now agree and the premise could not hold
    any more: the test failed on a correct importer. The behaviour is pinned here on a stage
    authored in the test, with the robot root raised AND turned, so world and base differ
    in both translation and rotation.
    """
    from pxr import Gf, Usd, UsdGeom
    from trainit_setup_assistant.importers import import_usd
    with tempfile.TemporaryDirectory() as tmp:
        usd_path = os.path.join(tmp, 'cell.usda')
        stage = Usd.Stage.CreateNew(usd_path)
        robot = UsdGeom.Xform.Define(stage, '/fr3wml_suction')      # the robot root, on a table
        UsdGeom.XformCommonAPI(robot).SetTranslate(Gf.Vec3d(0.2, -0.1, 1.0))
        UsdGeom.XformCommonAPI(robot).SetRotate(Gf.Vec3f(0.0, 0.0, 90.0))
        UsdGeom.Cube.Define(stage, '/fr3wml_suction/flange_visual')   # part of the robot
        cube = UsdGeom.Cube.Define(stage, '/World/Cube')
        cube.GetSizeAttr().Set(0.05)
        UsdGeom.XformCommonAPI(cube).SetTranslate(Gf.Vec3d(0.7, -0.1, 1.07))
        stage.GetRootLayer().Save()

        # raw (no base prim found) -> the stage world frame
        raw = {o.id: o for o in import_usd(usd_path)}
        assert raw['Cube'].position == pytest.approx([0.7, -0.1, 1.07], abs=1e-6)

        # aligned to the robot root: R^-1 (p - t), with the root turned +90 deg about Z
        aligned = {o.id: o for o in import_usd(usd_path, base_prim='/fr3wml_suction')}
        assert aligned['Cube'].position == pytest.approx([0.0, -0.5, 0.07], abs=1e-6)
        assert 'flange_visual' not in aligned               # the robot's own prims are skipped

        # auto-detected from the robot name hint: the same alignment
        auto = {o.id: o for o in import_usd(usd_path, robot_hint='fr3wml')}
        assert auto['Cube'].position == pytest.approx(aligned['Cube'].position, abs=1e-9)


@pytest.mark.skipif(not usd_available(), reason='pxr/usd-core not available')
def test_an_explicit_base_prim_that_is_not_on_the_stage_is_refused():
    """vNext M0C #6. An explicit base prim missing from the stage silently fell back to the
    WORLD frame: on the reference cell, whose robot moved from /fr3wml_suction to
    /World/fr3wml_suction, every object came back 1,06 m off and nothing said so. A path the
    caller named is a mistake when it is not there, not a choice -- refused, naming it."""
    from pxr import Usd, UsdGeom
    from trainit_setup_assistant.importers import import_usd
    with tempfile.TemporaryDirectory() as tmp:
        usd_path = os.path.join(tmp, 'cell.usda')
        stage = Usd.Stage.CreateNew(usd_path)
        UsdGeom.Cube.Define(stage, '/World/Cube')
        stage.GetRootLayer().Save()
        with pytest.raises(ValueError, match='/robot_that_moved'):
            import_usd(usd_path, base_prim='/robot_that_moved')
        assert import_usd(usd_path)                  # no base prim at all: the world frame


@pytest.mark.skipif(not (usd_available() and os.path.isfile(BIN_USD)),
                    reason='bin_picking_cell.usd not present')
def test_the_reference_cell_stage_aligns_the_part_at_the_flange():
    """What holds on the reference cell's own stage: aligned to the robot, by its base link's
    path or by name hint, the Cube sits at the flange (~0.014 m), and the two paths agree."""
    from trainit_setup_assistant.importers import import_usd
    aligned = import_usd(BIN_USD, base_prim='/World/fr3wml_suction/base_link')
    cube = next(o for o in aligned if o.id.lower().startswith('cube'))
    assert abs(cube.position[2] - 0.014) < 0.05
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
    assert SceneObject.parse_obj(o.dict()) == o   # idempotent round-trip (v1/v2-overlap)
    s = SceneObject(id='wall')                        # defaults -> static
    assert s.category is SceneObjectCategory.STATIC and not s.dynamic


def test_actuated_checked_dynamic_allowed_in_tree():
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects = []
    ctrl.add_scene_object('belt', [1.0, 0.4, 0.3], [0.5, 0.0, 0.0], category='actuated')
    ctrl.add_scene_object('crate', [0.3, 0.2, 0.3], [0.4, 0.0, 0.1], category='dynamic')
    ctrl.set_object_category('belt', 'actuated')      # re-classify path
    identity_cases.require('p2_example')
    with tempfile.TemporaryDirectory() as tmp:
        ctrl.set_project_name('cat_test')
        ctrl.generate(tmp)
        tree = open(os.path.join(tmp, 'cat_test_app', 'bt_trees', 'pick_place.xml')).read()
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
    assert SceneObject.parse_obj(json.loads(bottle.json())) == bottle

    # the crate: dynamic (collision-allowed) but NOT a grasp target -> never attached
    crate = SceneObject(id='crate', shape='mesh',
                        mesh_resource='package://p/crate.stl', category='dynamic')
    assert crate.is_dynamic() and not crate.is_grasp_target()

    # a static mesh (machine) is a CHECKED collision object
    machine = SceneObject(id='machine', shape='mesh',
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
    assert MotionSegment.parse_obj(on.dict()) == on


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
