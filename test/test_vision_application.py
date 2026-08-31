"""TSA v4 vision integration units: model round-trip, tree emission, validation,
perception.yaml, camera-variant rewrite. The golden test covers the full bundle;
these pin the pieces in isolation. ROS-free."""

import pytest

from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.generator.camera_variant import rewrite_camera_include
from trainit_setup_assistant.generator.perception_yaml import build_perception_yaml
from trainit_setup_assistant.model import (
    AppType,
    BundleSpec,
    CameraSpec,
    CanonicalProject,
    DetectorSpec,
    MotionSegment,
    PerceptionSpec,
    ToolAction,
    VisionBinding,
    Waypoint,
)
from trainit_setup_assistant.model.enums import ToolActionKind, WaypointType
from trainit_setup_assistant.model.io import load_project, save_project
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, PlanningGroupSpec, RobotSpec)


def _project(with_vision=True, loop=-1, with_reset=True) -> CanonicalProject:
    wps = [
        Waypoint(name='home', type=WaypointType.JOINT, named='home'),
        Waypoint(name='pre_pick', type=WaypointType.JOINT, named='pre_pick',
                 vision=VisionBinding(detector='cube', dz=0.08,
                                      orientation='from:pick') if with_vision else None),
        Waypoint(name='pick', type=WaypointType.TCP,
                 position=[0.56, -0.02, 0.03], orientation=[-0.707, 0.707, 0.0, 0.0],
                 vision=VisionBinding(detector='cube', dz=0.006,
                                      orientation='keep') if with_vision else None),
        Waypoint(name='place', type=WaypointType.TCP,
                 position=[0.56, -0.24, 0.036], orientation=[-0.707, 0.707, 0.0, 0.0]),
    ]
    actions = [ToolAction(at_waypoint='pick', kind=ToolActionKind.GRASP),
               ToolAction(at_waypoint='place', kind=ToolActionKind.RELEASE)]
    if with_reset:
        actions.append(ToolAction(at_waypoint='place', kind=ToolActionKind.RESET_SCENE))
    p = CanonicalProject(
        project_name='cell',
        bundle=BundleSpec.from_prefix('cell'),
        robot=RobotSpec(
            robot_name='cell',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            base_frame='base_link', tip_link='tcp',
            planning_group=PlanningGroupSpec(name='arm', base_link='base_link',
                                             tip_link='tcp', joints=['j1']),
            gripper=GripperSpec(),
            arm_controller=ArmControllerSpec(),
        ),
        perception=PerceptionSpec(
            camera=CameraSpec(replace_include='$(find x)/a.xacro',
                              with_include='$(find x)/a_camera.xacro'),
            detectors=[DetectorSpec(name='cube',
                                    params={'class_id': 'cube', 'h': [170, 10]})],
        ),
    )
    p.application.type = AppType.VISION_GUIDED_MOTION
    p.application.waypoints = wps
    p.application.segments = [MotionSegment(to_waypoint=w.name) for w in wps]
    p.application.tool_actions = actions
    p.application.sequence = ['home', 'pre_pick', 'pick', 'place']
    p.application.loop_cycles = loop
    p.application.loop_start = 'pre_pick'
    return p


# --- model round-trip ---------------------------------------------------------

def test_perception_round_trips_through_yaml(tmp_path):
    p = _project()
    f = tmp_path / 'project.yaml'
    save_project(p, f)
    q = load_project(f)
    assert q.perception.detectors[0].params['h'] == [170, 10]
    assert q.application.waypoint_by_name('pick').vision.dz == 0.006
    assert q.schema_version == 2


def test_old_projects_without_perception_still_load(tmp_path):
    p = _project(with_vision=False)
    p.perception = None
    f = tmp_path / 'project.yaml'
    save_project(p, f)
    text = f.read_text()
    assert 'vision:' not in text.replace('vision: null', '')
    q = load_project(f)
    assert q.perception is None


# --- tree emission ------------------------------------------------------------

def test_vision_tree_has_detect_bindings_and_settle():
    xml = get_application(AppType.VISION_GUIDED_MOTION).build_tree_xml(_project())
    assert ('<DetectObject detector="cube" class_id="cube" target_frame="base_link" '
            'timeout_ms="3000" out_key="detected.cube"/>') in xml
    assert ('<SetWaypointFromDetection waypoint="pre_pick" from="detected.cube" '
            'dz="0.080" orientation="from:pick"/>') in xml
    assert ('<SetWaypointFromDetection waypoint="pick" from="detected.cube" '
            'dz="0.006" orientation="keep"/>') in xml
    # settle AFTER the reset, inside the cycle
    reset_at = xml.index('<ResetScene/>')
    assert '<Sleep msec="1500"/>' in xml[reset_at:]
    # detection at the TOP of the cycle, before the first move
    assert xml.index('<DetectObject') < xml.index('<MoveWaypoint waypoint="pre_pick"')


def test_vision_tree_without_loop_still_detects_first():
    xml = get_application(AppType.VISION_GUIDED_MOTION).build_tree_xml(
        _project(loop=0, with_reset=False))
    assert xml.index('<DetectObject') < xml.index('<MoveWaypoint waypoint="home"')
    assert '<Sleep msec="1500"' not in xml            # no reset -> no settle


def test_no_bindings_emits_the_blind_tree():
    p = _project(with_vision=False)
    xml = get_application(AppType.VISION_GUIDED_MOTION).build_tree_xml(p)
    assert '<DetectObject' not in xml and 'SetWaypointFromDetection' not in xml


# --- validation ---------------------------------------------------------------

def test_validate_flags_unknown_detector_and_bad_orientation():
    p = _project()
    p.application.waypoint_by_name('pre_pick').vision.detector = 'ghost'
    p.application.waypoint_by_name('pick').vision.orientation = 'from:nowhere'
    problems = get_application(AppType.VISION_GUIDED_MOTION).validate(p)
    text = '\n'.join(problems)
    assert 'unknown detector "ghost"' in text
    assert 'unknown waypoint "nowhere"' in text


def test_validate_flags_keep_without_orientation():
    p = _project()
    p.application.waypoint_by_name('pre_pick').vision.orientation = 'keep'
    problems = get_application(AppType.VISION_GUIDED_MOTION).validate(p)
    assert any('orientation "keep" but the waypoint has none' in x for x in problems)


def test_validate_does_not_require_gripper_actions():
    p = _project()
    p.application.tool_actions = []
    problems = get_application(AppType.VISION_GUIDED_MOTION).validate(p)
    assert not any('grasp' in x or 'release' in x for x in problems)


# --- perception.yaml + camera variant -----------------------------------------

def test_perception_yaml_shape():
    import yaml
    doc = yaml.safe_load(build_perception_yaml(_project()))
    cube = doc['detectors']['cube']
    assert cube['method'] == 'color_mask'
    assert cube['input']['rgb'] == '/camera/color/image_raw'
    assert cube['params']['h'] == [170, 10]
    assert cube['continuous'] is True and cube['rate_hz'] == 10.0


def test_camera_include_rewrite():
    cam = CameraSpec(replace_include='$(find x)/a.xacro',
                     with_include='$(find x)/a_camera.xacro')
    text = '<robot>\n    <xacro:include filename="$(find x)/a.xacro" />\n</robot>\n'
    out = rewrite_camera_include(text, cam)
    assert 'a_camera.xacro' in out and '"$(find x)/a.xacro"' not in out
    assert rewrite_camera_include('<robot/>', cam) is None
    assert rewrite_camera_include(text, None) is None


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


# --- adversarial-review regressions (2026-08-31 verification pass) --------------

def test_validate_flags_on_demand_bound_detector_and_detected_orientation():
    p = _project()
    p.perception.detectors[0].continuous = False
    p.application.waypoint_by_name('pick').vision.orientation = 'detected'
    problems = get_application(AppType.VISION_GUIDED_MOTION).validate(p)
    text = '\n'.join(problems)
    assert 'on-demand' in text and 'time out' in text
    assert 'identity orientation' in text


def test_custom_method_emits_no_detector_node():
    d = DetectorSpec(name='mine', method='custom')
    assert d.emits_node is False               # runtime registry knows only color_mask


def test_detector_name_validation():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project())
    with pytest.raises(ValueError):
        ctrl.upsert_detector('red cube')
    with pytest.raises(ValueError):
        ctrl.upsert_detector('1cube')
    ctrl.upsert_detector('red_cube')           # valid


def test_camera_rewrite_refuses_double_include():
    cam = CameraSpec(replace_include='$(find x)/a.xacro',
                     with_include='$(find x)/a_camera.xacro')
    both = ('<robot>\n<xacro:include filename="$(find x)/a.xacro" />\n'
            '<xacro:include filename="$(find x)/a_camera.xacro" />\n</robot>\n')
    assert rewrite_camera_include(both, cam) is None


# --- D-016 (v4.1): move-level vision, offsets, planner guard --------------------

def test_dx_dy_emitted_only_when_set():
    p = _project()
    p.application.waypoint_by_name('pick').vision.dx = 0.01
    xml = get_application(AppType.VISION_GUIDED_MOTION).build_tree_xml(p)
    assert ('waypoint="pick" from="detected.cube" dx="0.010" dz="0.006" '
            'orientation="keep"' in xml)
    # pre_pick has no dx/dy -> golden-shaped dz-only attributes
    assert ('waypoint="pre_pick" from="detected.cube" dz="0.080" '
            'orientation="from:pick"' in xml)


def test_validate_flags_ptp_pilz_into_a_vision_goal():
    from trainit_setup_assistant.model.enums import MotionType, PlannerId
    p = _project()
    seg = p.application.segment_for('pick')
    seg.motion = MotionType.PTP
    seg.planner = PlannerId.PILZ
    problems = get_application(AppType.VISION_GUIDED_MOTION).validate(p)
    assert any('ptp/pilz' in x and 'vision-driven' in x for x in problems)


def test_align_policy_gated_by_detector_metadata():
    d = DetectorSpec(name='cube')
    assert d.gives_orientation is False        # colour mask: identity only
