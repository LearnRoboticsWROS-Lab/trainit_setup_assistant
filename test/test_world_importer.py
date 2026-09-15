"""Gazebo .world scene import (the Gazebo analogue of the USD importer). ROS-free.

Pins: box/cylinder extraction, <state> pose priority over the model <pose>, static ->
STATIC / non-static -> DYNAMIC default, robot_base_world_pose -> base_link, skip of sun/ground_plane,
and the source tag.
"""
from pathlib import Path

from trainit_setup_assistant.importers import import_world
from trainit_setup_assistant.model.enums import (
    SceneObjectCategory, SceneObjectSource, ShapeType)

WORLD = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="w">
    <model name="ground_plane"><static>1</static>
      <link name="l"><collision name="c"><geometry><plane/></geometry></collision></link></model>
    <model name="table"><static>1</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="l"><collision name="c"><pose>0 0 1 0 0 0</pose>
        <geometry><box><size>1.5 0.8 0.03</size></box></geometry></collision></link></model>
    <model name="red_cube"><static>0</static>
      <pose>0.5 0 1.1 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><box><size>0.02 0.02 0.2</size></box></geometry></collision></link></model>
    <model name="post"><static>1</static>
      <pose>0.2 0.2 0.3 0 0 0</pose>
      <link name="l"><collision name="c"><geometry><cylinder><radius>0.05</radius><length>0.6</length></cylinder></geometry></collision></link></model>
    <state world_name="w">
      <model name="red_cube"><pose>0.5 0 1.115 0 0 0</pose></model>
    </state>
  </world>
</sdf>
"""


def _world_file(tmp_path) -> str:
    f = Path(tmp_path) / "cell.world"
    f.write_text(WORLD)
    return str(f)


def test_import_world_extracts_models(tmp_path):
    objs = {o.id: o for o in import_world(_world_file(tmp_path))}
    # sun/ground_plane/plane geometry are skipped (lights + infinite floor)
    assert set(objs) == {"table", "red_cube", "post"}
    for o in objs.values():
        assert o.source is SceneObjectSource.WORLD

    # box top: model pose 0 + collision offset z=1 -> z=1.0; static -> STATIC obstacle
    assert objs["table"].shape is ShapeType.BOX
    assert objs["table"].dims == [1.5, 0.8, 0.03]
    assert objs["table"].position[2] == 1.0
    assert objs["table"].category is SceneObjectCategory.STATIC

    # cylinder -> (radius, length)
    assert objs["post"].shape is ShapeType.CYLINDER
    assert objs["post"].dims == [0.05, 0.6]

    # red_cube: <state> pose (1.115) wins over the model <pose> (1.1); non-static -> DYNAMIC
    assert objs["red_cube"].position == [0.5, 0.0, 1.115]
    assert objs["red_cube"].category is SceneObjectCategory.DYNAMIC
    assert objs["red_cube"].dynamic is True


def test_base_offset_expresses_in_base_link(tmp_path):
    objs = {o.id: o for o in import_world(_world_file(tmp_path), robot_base_world_pose=[0, 0, 0.8])}
    # world z 1.115 - mount 0.8 -> base_link z 0.315
    assert objs["red_cube"].position == [0.5, 0.0, 0.315]


def test_category_override(tmp_path):
    # force everything static (the user re-classifies the target at Step 2)
    objs = import_world(_world_file(tmp_path), category="static")
    assert all(o.category is SceneObjectCategory.STATIC for o in objs)
    # classify hook: only red_cube dynamic
    objs2 = {o.id: o for o in import_world(
        _world_file(tmp_path),
        classify=lambda name: "dynamic" if name == "red_cube" else "static")}
    assert objs2["red_cube"].category is SceneObjectCategory.DYNAMIC
    assert objs2["table"].category is SceneObjectCategory.STATIC


def test_full_rigid_inverse_absorbs_rotation(tmp_path):
    import math

    import pytest
    # robot base at (0,0,0.8) yawed +90deg about z: a world object folds into base_link with
    # BOTH translation AND rotation absorbed (the Gazebo mirror of Isaac's binv, ADR-0011).
    objs = {o.id: o for o in import_world(
        _world_file(tmp_path), robot_base_world_pose=[0, 0, 0.8, 0, 0, math.pi / 2])}
    cube = objs["red_cube"]                       # world (0.5, 0, 1.115), identity orientation
    assert cube.position[0] == pytest.approx(0.0, abs=1e-6)
    assert cube.position[1] == pytest.approx(-0.5, abs=1e-6)   # x folds onto -y under Rz(-90)
    assert cube.position[2] == pytest.approx(0.315, abs=1e-6)
    # orientation = inverse base rotation = Rz(-90deg): quat [0, 0, -0.7071, 0.7071]
    assert cube.orientation[2] == pytest.approx(-0.70711, abs=1e-4)
    assert cube.orientation[3] == pytest.approx(0.70711, abs=1e-4)


def test_identity_pose_keeps_world_frame(tmp_path):
    # None / all-zero pose = robot at world origin: objects stay in world coords (no shift)
    objs = {o.id: o for o in import_world(_world_file(tmp_path), robot_base_world_pose=None)}
    assert objs["red_cube"].position == [0.5, 0.0, 1.115]
    zero = {o.id: o for o in import_world(_world_file(tmp_path),
                                          robot_base_world_pose=[0, 0, 0, 0, 0, 0])}
    assert zero["red_cube"].position == [0.5, 0.0, 1.115]


_MULTI_WORLD = """<?xml version="1.0" ?>
<sdf version="1.6">
  <world name="w">
    <model name="table"><static>1</static>
      <pose>0.5 0 0 0 0 0</pose>
      <link name="l">
        <collision name="top"><pose>0 0 0.4 0 0 0</pose>
          <geometry><box><size>1.0 0.6 0.05</size></box></geometry></collision>
        <collision name="leg1"><pose>0.4 0.25 0.2 0 0 0</pose>
          <geometry><cylinder><radius>0.03</radius><length>0.4</length></cylinder></geometry></collision>
        <collision name="leg2"><pose>-0.4 -0.25 0.2 0 0 0</pose>
          <geometry><cylinder><radius>0.03</radius><length>0.4</length></cylinder></geometry></collision>
      </link></model>
    <model name="pallet"><static>1</static>
      <pose>-0.9 -0.3 0 0 0 0</pose>
      <link name="l"><collision name="c">
        <geometry><mesh><uri>model://euro_pallet/meshes/pallet.dae</uri><scale>0.1 0.1 0.1</scale></mesh></geometry>
      </collision></link></model>
  </world>
</sdf>
"""


def _multi_file(tmp_path) -> str:
    f = Path(tmp_path) / "multi.world"
    f.write_text(_MULTI_WORLD)
    return str(f)


def test_multi_collision_emits_one_object_per_collision(tmp_path):
    # a table with a box top + 2 cylinder legs -> 3 SceneObjects (ADR-0011 B), not one box
    objs = {o.id: o for o in import_world(_multi_file(tmp_path))}
    assert {"table_0", "table_1", "table_2"} <= set(objs)
    assert objs["table_0"].shape is ShapeType.BOX          # top
    assert objs["table_1"].shape is ShapeType.CYLINDER     # a leg
    # top world z = model 0 + collision 0.4 -> 0.4
    assert objs["table_0"].position[2] == 0.4


def test_mesh_resolved_to_package_uri(tmp_path):
    # model://euro_pallet/... -> package://<mesh_pkg>/models/euro_pallet/... + scale kept
    objs = {o.id: o for o in import_world(_multi_file(tmp_path), mesh_package='lrwros_ur5_workcell')}
    pallet = objs["pallet"]
    assert pallet.shape is ShapeType.MESH
    assert pallet.mesh_resource == 'package://lrwros_ur5_workcell/models/euro_pallet/meshes/pallet.dae'
    assert pallet.scale == [0.1, 0.1, 0.1]


def test_mesh_without_package_degrades_to_box(tmp_path):
    # no mesh_package -> a model:// mesh can't resolve -> conservative box (scene still loads)
    objs = {o.id: o for o in import_world(_multi_file(tmp_path))}
    assert objs["pallet"].shape is ShapeType.BOX
