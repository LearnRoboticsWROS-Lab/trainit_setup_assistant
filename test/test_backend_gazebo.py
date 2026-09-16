"""Gazebo backend emission (ADR-0008 F2). ROS-free.

Verifies the generator emits a correct Gazebo Classic cell for a bundle whose
DeploymentSpec.modes include Backend.GAZEBO, on BOTH paths (from-scratch bootstrap and
base-config scene-loader), while mock/isaac/real stay byte-identical (the byte gate in
test_backend_bytestability.py proves the latter). Correctness contract (the F2 DoD):

- Gazebo Classic launched (gazebo.launch.py) + the robot spawned via spawn_entity.py.
- The gazebo_ros2_control plugin owns /controller_manager INSIDE gzserver -> the bring-up
  runs SPAWNERS ONLY against it, NEVER a standalone ros2_control_node (unlike mock/isaac).
- use_sim_time true for gazebo; move_group built with backend=mode (GazeboSystem xacro).
- exactly one robot_state_publisher; the xacro selects gazebo_ros2_control/GazeboSystem on
  use_gazebo and emits the <gazebo> plugin whose <parameters> single-source the controllers.
"""

import os
import tempfile
from pathlib import Path

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.generator.emitters.app_pkg import AppEmitter
from trainit_setup_assistant.generator.emitters.base import GenContext
from trainit_setup_assistant.generator.emitters.moveit_config_pkg import MoveitConfigEmitter
from trainit_setup_assistant.generator.manifest import GenerationManifest
from trainit_setup_assistant.model import (
    AppType, Backend, BundleSpec, CanonicalProject, DeploymentSpec, MotionSegment,
    ToolAction, Waypoint, load_project)
from trainit_setup_assistant.model.enums import GripperKind, ToolActionKind, WaypointType
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, PlanningGroupSpec, RobotSpec)

GOLDEN_BUNDLE = Path('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
                     'fr3wml_suction_camera_tsa_4_1_bundle')
GOLDEN_PROJECT = GOLDEN_BUNDLE / 'project.yaml'
GZ_MODES = [Backend.MOCK, Backend.ISAAC, Backend.GAZEBO, Backend.REAL]


def _gz_fromscratch_project(gripper: GripperSpec = None,
                            cm_bootstrap: bool = False) -> CanonicalProject:
    p = CanonicalProject(
        project_name='gzcell', bundle=BundleSpec.from_prefix('gzcell'),
        robot=RobotSpec(
            robot_name='gzcell',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            base_frame='base_link', tip_link='tcp',
            planning_group=PlanningGroupSpec(name='arm', base_link='base_link',
                                             tip_link='tcp', joints=['j1', 'j2']),
            gripper=gripper or GripperSpec(),
            arm_controller=ArmControllerSpec()))
    p.deployment = DeploymentSpec(modes=GZ_MODES, gazebo_cm_bootstrap=cm_bootstrap)
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='home', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='home')]
    p.application.sequence = ['home']
    return p


def _render_fromscratch(project) -> Path:
    tmp = tempfile.mkdtemp()
    out = Path(tmp)
    for Em in (MoveitConfigEmitter, AppEmitter):
        ctx = GenContext(out, build_jinja_env(), GenerationManifest('gz', 'gzcell'))
        Em().emit(project, ctx)
    return out


def _gazebo_branch(launch: str) -> str:
    """The runtime code of the `elif mode == 'gazebo':` block (excludes the comment)."""
    start = launch.index('elif mode == "gazebo":')
    end = launch.index('else:  # real', start)
    return launch[start:end]


# --- from-scratch (bootstrap) path --------------------------------------------

def test_fromscratch_gazebo_bringup_wiring():
    out = _render_fromscratch(_gz_fromscratch_project())
    launch = (out / 'gzcell_app' / 'launch' / 'bringup.launch.py').read_text()
    gz = _gazebo_branch(launch)

    # Gazebo launched + robot spawned the Classic way
    assert 'gazebo.launch.py' in gz
    assert 'spawn_entity.py' in gz
    assert '"-entity", ROBOT_NAME, "-topic", "robot_description"' in gz

    # spawners ONLY against the in-gzserver plugin CM; NO standalone ros2_control_node
    assert '"--controller-manager", "/controller_manager"' in gz
    assert 'executable="spawner"' in gz
    assert 'executable="ros2_control_node"' not in gz     # the plugin owns the CM

    # the mock/isaac branch is UNCHANGED (still starts the standalone node)
    assert 'executable="ros2_control_node"' in launch

    # sim time + the GazeboSystem robot_description
    assert 'use_sim_time = use_isaac or mode == "gazebo"' in launch
    assert 'backend=mode' in launch

    # exactly one robot_state_publisher for the whole cell
    assert launch.count('executable="robot_state_publisher"') == 1

    # world/gui knobs (headless via gui:=false)
    assert 'DeclareLaunchArgument("world"' in launch
    assert 'DeclareLaunchArgument("gui"' in launch


def test_gazebo_cm_bootstrap_opt_in():
    # default OFF: spawners only, no standalone controller_manager node in the gz branch
    off = _gazebo_branch((_render_fromscratch(_gz_fromscratch_project())
                          / 'gzcell_app' / 'launch' / 'bringup.launch.py').read_text())
    assert 'executable="ros2_control_node"' not in off

    # opt-in ON (deployment.gazebo_cm_bootstrap): a transient standalone controller_manager
    # is added and the spawners start on ITS start (the UR init-race workaround)
    on = _gazebo_branch((_render_fromscratch(_gz_fromscratch_project(cm_bootstrap=True))
                         / 'gzcell_app' / 'launch' / 'bringup.launch.py').read_text())
    assert 'executable="ros2_control_node"' in on
    assert 'OnProcessStart(target_action=gz_cm, on_start=[gz_spawners[0]])' in on


def test_fromscratch_gazebo_trainit_bt_threads_backend():
    # the BT node's robot_description must be built with backend=mode too (not a stale
    # use_isaac), so every node's hardware plugin is uniform (arch-review follow-up).
    out = _render_fromscratch(_gz_fromscratch_project())
    bt = (out / 'gzcell_app' / 'launch' / 'trainit_bt.launch.py').read_text()
    assert 'backend=LaunchConfiguration("mode").perform(context)' in bt
    assert 'DeclareLaunchArgument("mode"' in bt


def test_fromscratch_gazebo_ros2_control_xacro():
    out = _render_fromscratch(_gz_fromscratch_project())
    xacro = (out / 'gzcell_trainit_config' / 'config' / 'gzcell.ros2_control.xacro').read_text()

    # 3-way hardware plugin selector: gazebo | isaac | mock
    assert 'gazebo_ros2_control/GazeboSystem' in xacro
    assert 'topic_based_ros2_control/TopicBasedSystem' in xacro      # isaac kept
    assert 'mock_components/GenericSystem' in xacro                   # mock kept
    assert 'use_gazebo' in xacro

    # the <gazebo> plugin that creates the in-gzserver controller_manager, single-sourcing
    # the controllers the spawners load
    assert 'libgazebo_ros2_control.so' in xacro
    assert '<parameters>$(find gzcell_trainit_config)/config/ros2_controllers.yaml' in xacro

    urdf = (out / 'gzcell_trainit_config' / 'config' / 'gzcell.urdf.xacro').read_text()
    assert '<xacro:arg name="use_gazebo"' in urdf
    assert 'use_gazebo="$(arg use_gazebo)"' in urdf


def test_fromscratch_gazebo_parallel_gripper_spawns_controller():
    par = GripperSpec(kind=GripperKind.PARALLEL, controller_name='gripper_controller',
                      command_joint='finger_joint')
    out = _render_fromscratch(_gz_fromscratch_project(gripper=par))
    gz = _gazebo_branch((out / 'gzcell_app' / 'launch' / 'bringup.launch.py').read_text())
    assert 'gz_controllers.append("gripper_controller")' in gz


def test_fromscratch_suction_gripper_spawns_no_extra_controller():
    suc = GripperSpec(kind=GripperKind.SUCTION, controller_name='suction_controller')
    out = _render_fromscratch(_gz_fromscratch_project(gripper=suc))
    gz = _gazebo_branch((out / 'gzcell_app' / 'launch' / 'bringup.launch.py').read_text())
    # suction grasps via a LinkAttacher bridge, not a ros2_control gripper controller
    assert 'gz_controllers.append' not in gz


# --- base-config (scene-loader) path ------------------------------------------

def _skip_if_no_golden():
    import pytest
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')


def test_base_config_gazebo_bringup_wiring():
    _skip_if_no_golden()
    project = load_project(str(GOLDEN_PROJECT))
    project.bundle = BundleSpec(description_package='g_desc', moveit_config_package='g_cfg',
                                app_package='g_app')
    project.deployment.modes = GZ_MODES     # add gazebo to the golden's modes
    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(project, tmp)
        cell = Path(tmp) / 'g_cfg' / 'launch' / 'bringup.launch.py'
        launch = cell.read_text()
        gz = _gazebo_branch(launch)
        assert 'gazebo.launch.py' in gz and 'spawn_entity.py' in gz
        assert 'executable="ros2_control_node"' not in gz          # plugin owns the CM
        assert 'executable="ros2_control_node"' in launch          # mock/isaac unchanged
        assert 'backend=mode' in launch
        assert 'DeclareLaunchArgument("world"' in launch
        assert launch.count('executable="robot_state_publisher"') == 1


def test_gazebo_absent_emits_no_gazebo_branch():
    # a bundle WITHOUT gazebo in modes must not carry any gazebo wiring (byte-stability)
    p = _gz_fromscratch_project()
    p.deployment = DeploymentSpec(modes=[Backend.MOCK, Backend.ISAAC, Backend.REAL])
    out = _render_fromscratch(p)
    launch = (out / 'gzcell_app' / 'launch' / 'bringup.launch.py').read_text()
    assert 'gazebo' not in launch.lower()
    xacro = (out / 'gzcell_trainit_config' / 'config' / 'gzcell.ros2_control.xacro').read_text()
    assert 'GazeboSystem' not in xacro
    assert 'use_gazebo' not in xacro


def test_gazebo_link_attacher_bridge_generated_from_step7(tmp_path):
    """W3 (ADR-0010): when the gripper's gazebo sim adapter is link_attacher and a grasp target
    exists, TSA GENERATES the LinkAttacher bridge into the scene_loader (the base moveit_config
    does not carry it): the kit script is copied + installed, and a node runs it with the grasp
    topic / robot / ee_link / object params. mock/isaac/real (no gazebo adapter) get nothing."""
    from trainit_setup_assistant.generator.emitters.base import GenContext
    from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
        SceneLoaderMoveitConfigEmitter)
    from trainit_setup_assistant.generator.manifest import GenerationManifest
    from trainit_setup_assistant.model import (Backend, BundleSpec, CanonicalProject,
        DeploymentSpec, MotionSegment, SceneObject, Waypoint)
    from trainit_setup_assistant.model.enums import (AppType, GripperKind,
        SceneObjectCategory, ShapeType, WaypointType)
    from trainit_setup_assistant.model.robot import (ArmControllerSpec, DescriptionSource,
        GripperSpec, PlanningGroupSpec, RobotSpec)

    cfg = tmp_path / 'base' / 'config'
    cfg.mkdir(parents=True)
    (cfg / 'ur.urdf.xacro').write_text('<?xml version="1.0"?>\n'
        '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"/>\n')
    (cfg / 'ur.srdf').write_text('<?xml version="1.0"?>\n<robot name="r"></robot>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml',
              'moveit_controllers.yaml', 'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    g = GripperSpec(kind=GripperKind.PARALLEL, controller_name='gripper_position_controller',
                    command_joint='robotiq_85_left_knuckle_joint')       # gazebo -> link_attacher
    p = CanonicalProject(project_name='ur5', bundle=BundleSpec.from_prefix('ur5'),
        robot=RobotSpec(robot_name='ur',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            base_frame='base_link', tip_link='tcp', base_moveit_config_path=str(tmp_path / 'base'),
            planning_group=PlanningGroupSpec(name='arm', joints=['j1']), gripper=g,
            arm_controller=ArmControllerSpec()))
    p.deployment = DeploymentSpec(modes=[Backend.MOCK, Backend.GAZEBO], default_mode=Backend.GAZEBO)
    p.scene.gripper_cmd_topic = '/gripper_cmd'
    p.scene.attach_link = 'wrist_3_link'
    p.scene.objects = [SceneObject(id='red_cube', shape=ShapeType.BOX, dims=[0.05, 0.05, 0.05],
        position=[0.5, 0, 0.3], category=SceneObjectCategory.DYNAMIC, grasp_target=True)]
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='home', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='home')]
    p.application.sequence = ['home']

    out = Path(tempfile.mkdtemp())
    SceneLoaderMoveitConfigEmitter().emit(
        p, GenContext(out, build_jinja_env(), GenerationManifest('ur5', 'ur5')))
    launch = (out / 'ur5_trainit_config' / 'launch' / 'bringup.launch.py').read_text()
    assert 'link_attacher_bridge.py' in launch
    assert "'ee_link': 'wrist_3_link'" in launch and "'object_model': 'red_cube'" in launch
    assert (out / 'ur5_trainit_config' / 'scripts' / 'link_attacher_bridge.py').is_file()
    assert 'link_attacher_bridge.py' in (out / 'ur5_trainit_config' / 'CMakeLists.txt').read_text()
