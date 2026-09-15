"""Base-config ingest (ADR-0008 F2): MoveItConfigsBuilder(<robot_name>) infers
config/<robot_name>.urdf.xacro and config/<robot_name>.srdf. A third-party base
moveit_config may name them differently (e.g. a UR cell ships ur.urdf.xacro / ur.srdf),
which the builder cannot find. The scene-loader emitter must ensure a <robot_name>.* copy
exists so ANY base config is consumable — and be a no-op / byte-stable when the base already
follows the convention (the FR3WML golden; covered by the byte gate + generator golden).
ROS-free.
"""
import tempfile
from pathlib import Path

from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.generator.emitters.base import GenContext
from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
    SceneLoaderMoveitConfigEmitter)
from trainit_setup_assistant.generator.manifest import GenerationManifest
from trainit_setup_assistant.model import (
    AppType, Backend, BundleSpec, CanonicalProject, DeploymentSpec, MotionSegment, Waypoint)
from trainit_setup_assistant.model.enums import WaypointType
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, PlanningGroupSpec, RobotSpec)


def _base_config(tmp: str, urdf_name: str, srdf_name: str) -> str:
    cfg = Path(tmp) / 'base' / 'config'
    cfg.mkdir(parents=True)
    (cfg / urdf_name).write_text(
        '<?xml version="1.0"?>\n<robot xmlns:xacro="http://www.ros.org/wiki/xacro" '
        'name="r"/>\n')
    (cfg / srdf_name).write_text('<?xml version="1.0"?>\n<robot name="r"></robot>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml',
              'moveit_controllers.yaml', 'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    return str(Path(tmp) / 'base')


def _emit(base: str) -> Path:
    p = CanonicalProject(
        project_name='gzc', bundle=BundleSpec.from_prefix('gzc'),
        robot=RobotSpec(
            robot_name='gzc',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            base_frame='base_link', tip_link='tcp', base_moveit_config_path=base,
            planning_group=PlanningGroupSpec(name='arm', base_link='base_link',
                                             tip_link='tcp', joints=['j1']),
            gripper=GripperSpec(), arm_controller=ArmControllerSpec()))
    p.deployment = DeploymentSpec(modes=[Backend.GAZEBO], default_mode=Backend.GAZEBO)
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='home', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='home')]
    p.application.sequence = ['home']
    out = Path(tempfile.mkdtemp())
    SceneLoaderMoveitConfigEmitter().emit(
        p, GenContext(out, build_jinja_env(), GenerationManifest('gzc', 'gzc')))
    return out / 'gzc_trainit_config' / 'config'


def test_mismatched_base_names_get_robot_named_copies(tmp_path):
    # a third-party cell names its files ur.* (robot_name is gzc): the builder infers
    # gzc.urdf.xacro / gzc.srdf, so those must be created from the ur.* originals.
    cfg = _emit(_base_config(str(tmp_path), 'ur.urdf.xacro', 'ur.srdf'))
    assert (cfg / 'gzc.urdf.xacro').is_file()
    assert (cfg / 'gzc.srdf').is_file()
    assert (cfg / 'ur.urdf.xacro').is_file()   # original kept (harmless, unused)


def test_matching_base_names_add_nothing_spurious(tmp_path):
    # base already follows the <robot_name>.* convention -> no ur.* file, no extra copies
    cfg = _emit(_base_config(str(tmp_path), 'gzc.urdf.xacro', 'gzc.srdf'))
    assert (cfg / 'gzc.urdf.xacro').is_file()
    assert (cfg / 'gzc.srdf').is_file()
    assert not (cfg / 'ur.urdf.xacro').exists()
