"""Gazebo .world scene import (the Gazebo analogue of the USD importer). ROS-free.

Pins: box/cylinder extraction, <state> pose priority over the model <pose>, static ->
STATIC / non-static -> DYNAMIC default, base_offset -> base_link, skip of sun/ground_plane,
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
    objs = {o.id: o for o in import_world(_world_file(tmp_path), base_offset=[0, 0, 0.8])}
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
