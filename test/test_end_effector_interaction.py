"""End-effector <-> dynamic-object interaction (ADR-0012). ROS-free / pure generation.

Covers the pieces layered on ADR-0010:
- the ``interacts_with_object`` gate: a trigger-only end-effector wires NO grasp Bool, so the
  topic is dropped from bt_params AND scene.yaml even if a stale value lingers;
- the four LinkAttacher names: a Step-7 override wins, an untouched config derives (byte-safe),
  and ``robot_model`` stays coherent with spawn_entity's ``-entity`` name;
- the Gazebo readiness fix: the app bt_delay default is larger for a gazebo cell, 8.0 otherwise
  (so the non-gazebo golden is byte-identical).
"""

import tempfile
from pathlib import Path

from trainit_setup_assistant.applications.base import build_bt_params_context
from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.generator.emitters.app_pkg import AppEmitter
from trainit_setup_assistant.generator.emitters.base import GenContext
from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
    SceneLoaderMoveitConfigEmitter)
from trainit_setup_assistant.generator.manifest import GenerationManifest
from trainit_setup_assistant.generator.scene_yaml import build_scene_yaml
from trainit_setup_assistant.model import (
    AppType, Backend, BundleSpec, CanonicalProject, DeploymentSpec, MotionSegment, SceneObject,
    ToolAction, Waypoint)
from trainit_setup_assistant.model.enums import (
    GripperKind, SceneObjectCategory, ShapeType, ToolActionKind, WaypointType)
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, LinkAttacherConfig, PlanningGroupSpec,
    RobotSpec)


# --- model ------------------------------------------------------------------

def test_interacts_with_object_defaults_true():
    """Default True preserves today's pick&place behaviour (and the golden's bytes)."""
    assert GripperSpec().interacts_with_object is True


def test_link_attacher_names_override_wins_else_derives():
    g = GripperSpec()
    # untouched: every field falls back to the passed-in derivation
    assert g.link_attacher_names(robot_model='ur', robot_link='wrist_3_link',
                                 object_model='red_cube', object_link='link_1') == {
        'robot_model': 'ur', 'robot_link': 'wrist_3_link',
        'object_model': 'red_cube', 'object_link': 'link_1'}
    # a Step-7 override wins per field, empties still derive
    g.link_attacher = LinkAttacherConfig(robot_model='cobot', object_link='link')
    assert g.link_attacher_names(robot_model='ur', robot_link='wrist_3_link',
                                 object_model='red_cube', object_link='link_1') == {
        'robot_model': 'cobot', 'robot_link': 'wrist_3_link',
        'object_model': 'red_cube', 'object_link': 'link'}


# --- the interacts gate drops the grasp topic -------------------------------

def _bt_project(interacts: bool) -> CanonicalProject:
    g = GripperSpec(kind=GripperKind.PARALLEL, interacts_with_object=interacts)
    p = CanonicalProject(
        project_name='cell', bundle=BundleSpec.from_prefix('cell'),
        robot=RobotSpec(robot_name='cell',
                        description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
                        planning_group=PlanningGroupSpec(name='arm', joints=['j1']),
                        gripper=g, arm_controller=ArmControllerSpec()))
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='pick', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='pick')]
    p.application.tool_actions = [ToolAction(at_waypoint='pick', kind=ToolActionKind.GRASP)]
    p.application.sequence = ['pick']
    p.scene.gripper_cmd_topic = '/gripper_cmd'
    return p


def test_gate_off_drops_gripper_cmd_from_bt_params():
    tpl = build_jinja_env().get_template('app/bt_params.yaml.j2')
    on = tpl.render(**build_bt_params_context(_bt_project(True)))
    off = tpl.render(**build_bt_params_context(_bt_project(False)))
    assert 'gripper_cmd_topic: "/gripper_cmd"' in on
    assert 'gripper_cmd_topic: "' not in off      # trigger-only: no grasp Bool wired


def test_gate_off_drops_gripper_cmd_from_scene_yaml():
    assert 'gripper_cmd_topic: "/gripper_cmd"' in build_scene_yaml(_bt_project(True))
    assert 'gripper_cmd_topic:' not in build_scene_yaml(_bt_project(False))


def test_primitive_grasp_target_emits_grasp_box():
    """A PRIMITIVE (box) grasp target must emit grasp_box/grasp_box_center so scene_manager_node
    builds the PURPLE AttachedCollisionObject (attach_box) instead of just REMOVE-ing it (which
    makes the object disappear in RViz)."""
    p = _bt_project(True)
    p.scene.objects = [SceneObject(id='red_cube', shape=ShapeType.BOX, dims=[0.02, 0.02, 0.2],
        position=[0.5, 0, 0.3], category=SceneObjectCategory.DYNAMIC, grasp_target=True)]
    y = build_scene_yaml(p)
    assert 'grasp_box: [0.02, 0.02, 0.2]' in y
    assert 'grasp_box_center: [0.0, 0.0, 0.0]' in y


# --- scene-loader emitter: override + spawn coherence + gate ----------------

def _base_config(tmp_path: Path) -> Path:
    cfg = tmp_path / 'base' / 'config'
    cfg.mkdir(parents=True)
    (cfg / 'ur.urdf.xacro').write_text('<?xml version="1.0"?>\n'
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"/>\n')
    (cfg / 'ur.srdf').write_text('<?xml version="1.0"?>\n<robot name="r"></robot>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml',
              'moveit_controllers.yaml', 'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    return tmp_path / 'base'


def _sl_project(tmp_path: Path, *, interacts=True, link_attacher=None) -> CanonicalProject:
    g = GripperSpec(kind=GripperKind.PARALLEL, controller_name='gripper_position_controller',
                    command_joint='robotiq_85_left_knuckle_joint',
                    interacts_with_object=interacts, link_attacher=link_attacher)
    p = CanonicalProject(project_name='ur5', bundle=BundleSpec.from_prefix('ur5'),
        robot=RobotSpec(robot_name='ur',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            base_frame='base_link', tip_link='tcp',
            base_moveit_config_path=str(_base_config(tmp_path)),
            planning_group=PlanningGroupSpec(name='arm', joints=['j1']), gripper=g,
            arm_controller=ArmControllerSpec()))
    p.deployment = DeploymentSpec(modes=[Backend.MOCK, Backend.GAZEBO],
                                  default_mode=Backend.GAZEBO)
    p.scene.gripper_cmd_topic = '/gripper_cmd'
    p.scene.attach_link = 'wrist_3_link'
    p.scene.objects = [SceneObject(id='red_cube', shape=ShapeType.BOX, dims=[0.05, 0.05, 0.05],
        position=[0.5, 0, 0.3], category=SceneObjectCategory.DYNAMIC, grasp_target=True)]
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='home', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='home')]
    p.application.sequence = ['home']
    return p


def _emit_scene_loader(project) -> str:
    out = Path(tempfile.mkdtemp())
    SceneLoaderMoveitConfigEmitter().emit(
        project, GenContext(out, build_jinja_env(), GenerationManifest('ur5', 'ur5')))
    return (out / 'ur5_trainit_config' / 'launch' / 'bringup.launch.py').read_text()


def test_link_attacher_override_drives_bridge_and_spawn_entity(tmp_path):
    """robot_model override must reach BOTH the LinkAttacher model1 AND spawn_entity's -entity
    name — else the weld's model1 never resolves against the spawned model."""
    la = LinkAttacherConfig(robot_model='cobot', object_link='link')
    launch = _emit_scene_loader(_sl_project(tmp_path, link_attacher=la))
    assert "'robot_model': 'cobot'" in launch
    assert "'object_link': 'link'" in launch                     # override
    assert "'object_model': 'red_cube'" in launch                # derived
    assert '"-entity", "cobot", "-topic", "robot_description"' in launch  # single datum


def test_default_spawn_entity_is_robot_name(tmp_path):
    launch = _emit_scene_loader(_sl_project(tmp_path))
    assert '"-entity", "ur", "-topic", "robot_description"' in launch
    assert "'robot_model': 'ur'" in launch


def test_gate_off_emits_no_link_attacher_bridge(tmp_path):
    launch = _emit_scene_loader(_sl_project(tmp_path, interacts=False))
    assert 'link_attacher_bridge.py' not in launch


# --- app bt_delay readiness default -----------------------------------------

def _app_bringup(tmp_path: Path, modes) -> str:
    p = _sl_project(tmp_path)
    p.deployment = DeploymentSpec(modes=modes, default_mode=modes[0])
    out = Path(tempfile.mkdtemp())
    AppEmitter().emit(p, GenContext(out, build_jinja_env(), GenerationManifest('ur5', 'ur5')))
    return (out / 'ur5_app' / 'launch' / 'bringup.launch.py').read_text()


def test_bt_delay_larger_for_gazebo(tmp_path):
    gz = _app_bringup(tmp_path, [Backend.MOCK, Backend.GAZEBO])
    assert 'DeclareLaunchArgument("bt_delay", default_value="15.0"' in gz


def test_bt_delay_default_for_non_gazebo(tmp_path):
    plain = _app_bringup(tmp_path, [Backend.MOCK, Backend.ISAAC, Backend.REAL])
    assert 'DeclareLaunchArgument("bt_delay", default_value="8.0"' in plain


# --- selectable grasp target (generalise beyond the Step-2 flag) -------------

def test_set_sole_grasp_target_selects_one_and_clears_others():
    from trainit_setup_assistant.gui.controller import AssistantController
    p = CanonicalProject(project_name='cell', bundle=BundleSpec.from_prefix('cell'),
        robot=RobotSpec(robot_name='cell',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            planning_group=PlanningGroupSpec(name='arm', joints=['j1']),
            gripper=GripperSpec(), arm_controller=ArmControllerSpec()))
    p.scene.objects = [
        SceneObject(id='cube_a', shape=ShapeType.BOX, dims=[0.05, 0.05, 0.05],
                    category=SceneObjectCategory.DYNAMIC, grasp_target=True),
        SceneObject(id='cube_b', shape=ShapeType.BOX, dims=[0.05, 0.05, 0.05],
                    category=SceneObjectCategory.DYNAMIC, grasp_target=False)]
    ctrl = AssistantController(project=p)
    ctrl.set_sole_grasp_target('cube_b')
    assert p.scene.grasp_target_ids() == ['cube_b']       # selected wins, other cleared
    ctrl.set_sole_grasp_target('missing')                 # unknown id: no-op
    assert p.scene.grasp_target_ids() == ['cube_b']
