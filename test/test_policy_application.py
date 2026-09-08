"""Learned-policy step emission (ADR-0005, POLICY_EXECUTION.md). ROS-free.

Pins the three execution modes in the emitted BT, the card-driven CheckRobotState,
validation, and — critically — that a project with NO policy step emits exactly the
deterministic tree (additive, byte-stable)."""

import textwrap

from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.model import (
    AppType,
    BundleSpec,
    CanonicalProject,
    MotionSegment,
    PolicyStep,
    ToolAction,
    Waypoint,
)
from trainit_setup_assistant.model.enums import PolicyMode, ToolActionKind, WaypointType
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, PlanningGroupSpec, RobotSpec)


_CARD = textwrap.dedent("""
    policy_card:
      name: fr5_suction_pickplace
      trained_for: "reach + grasp + carry to pre-place with the vacuum active"
      files: {jit: policy.pt, onnx: policy.onnx}
      end_state:
        tcp_pose_base: [0.4267, -0.278, 0.1589, -0.006, -0.7006, 0.7134, 0.0117]
        object_attached: true
        suction_active: true
""")


def _write_card(tmp_path):
    f = tmp_path / 'policy_card.yaml'
    f.write_text(_CARD)
    return str(f)


def _project(policy: PolicyStep = None) -> CanonicalProject:
    wps = [
        Waypoint(name='home', type=WaypointType.JOINT, named='home'),
        Waypoint(name='pick', type=WaypointType.TCP,
                 position=[0.56, -0.02, 0.03], orientation=[-0.707, 0.707, 0.0, 0.0]),
        Waypoint(name='pre_place', type=WaypointType.TCP,
                 position=[0.43, -0.28, 0.25], orientation=[-0.006, -0.70, 0.71, 0.01]),
    ]
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
    )
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = wps
    p.application.segments = [MotionSegment(to_waypoint=w.name) for w in wps]
    p.application.tool_actions = [
        ToolAction(at_waypoint='pick', kind=ToolActionKind.GRASP),
        ToolAction(at_waypoint='pre_place', kind=ToolActionKind.RELEASE)]
    p.application.sequence = ['home', 'pick', 'pre_place']
    if policy is not None:
        p.application.policies = [policy]
    return p


def _emit(project):
    return get_application(AppType.PICK_AND_PLACE).build_tree_xml(project)


# --- byte-stability: no policy => the deterministic tree, unchanged -----------

def test_no_policy_emits_deterministic_tree():
    xml = _emit(_project())
    assert 'RunPolicy' not in xml
    assert 'CheckRobotState' not in xml
    assert '<MoveWaypoint waypoint="pre_place"/>' in xml


# --- hybrid: decide a target, deterministic move executes it -------------------

def test_hybrid_policy_reuses_detection_path(tmp_path):
    pol = PolicyStep(name='grasp2preplace', before_waypoint='pre_place',
                     mode=PolicyMode.HYBRID, card=_write_card(tmp_path),
                     require_attached=True, attached_topic='/suction/attached')
    xml = _emit(_project(pol))
    assert '<RunPolicy service="/trainit_policy_runtime/run" ' in xml
    assert 'mode="hybrid"' in xml
    assert ('<DetectObject topic="/trainit_policy_runtime/target_detections" '
            'class_id="grasp2preplace:target"') in xml
    assert ('<SetWaypointFromDetection waypoint="pre_place" from="policy_grasp2preplace" '
            'orientation="detected"/>') in xml
    # hybrid KEEPS the deterministic move that executes the decided target
    assert '<MoveWaypoint waypoint="pre_place"/>' in xml
    # ordering: RunPolicy -> DetectObject -> SetWaypoint -> MoveWaypoint -> CheckRobotState
    assert (xml.index('<RunPolicy') < xml.index('<DetectObject')
            < xml.index('<SetWaypointFromDetection waypoint="pre_place"')
            < xml.index('<MoveWaypoint waypoint="pre_place"/>')
            < xml.index('<CheckRobotState'))
    # CheckRobotState carries the card's end_state pose + the attach assertion
    assert 'position="0.4267;-0.278;0.1589"' in xml
    assert 'require_attached="true"' in xml
    assert 'attached_topic="/suction/attached"' in xml


# --- pure: the policy drives the robot; no deterministic move ------------------

def test_pure_policy_replaces_move(tmp_path):
    pol = PolicyStep(name='p', before_waypoint='pre_place', mode=PolicyMode.PURE,
                     card=_write_card(tmp_path))
    xml = _emit(_project(pol))
    assert 'mode="pure"' in xml
    assert '<MoveWaypoint waypoint="pre_place"/>' not in xml   # policy drove it
    assert '<MoveWaypoint waypoint="pick"/>' in xml            # other moves untouched
    assert xml.index('<RunPolicy') < xml.index('<CheckRobotState')


# --- residual: nominal move kept, policy corrects on top -----------------------

def test_residual_policy_keeps_nominal_move(tmp_path):
    pol = PolicyStep(name='p', before_waypoint='pre_place', mode=PolicyMode.RESIDUAL,
                     card=_write_card(tmp_path))
    xml = _emit(_project(pol))
    assert 'mode="residual"' in xml
    assert '<MoveWaypoint waypoint="pre_place"/>' in xml       # nominal trajectory kept
    assert 'residual:' in xml                                  # the caveat comment


# --- validation ----------------------------------------------------------------

def test_validation_flags_unknown_waypoint_and_missing_card():
    app = get_application(AppType.PICK_AND_PLACE)
    pol = PolicyStep(name='p', before_waypoint='nope', card='/does/not/exist.yaml')
    problems = app.validate(_project(pol))
    assert any('unknown waypoint' in m for m in problems)
    assert any('card not found' in m for m in problems)


# --- bundle shipping: card + .pt/.onnx + a runtime launch ---------------------

def test_bundle_ships_policy_files_and_launch(tmp_path):
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    from trainit_setup_assistant.generator.emitters.app_pkg import AppEmitter
    from trainit_setup_assistant.generator.emitters.base import GenContext
    from trainit_setup_assistant.generator.manifest import GenerationManifest

    card = _write_card(tmp_path)
    (tmp_path / 'policy.onnx').write_bytes(b'ONNX')     # exported file beside the card
    pol = PolicyStep(name='grasp2preplace', before_waypoint='pre_place',
                     mode=PolicyMode.HYBRID, card=card)
    project = _project(pol)
    out = tmp_path / 'out'
    ctx = GenContext(out, build_jinja_env(), GenerationManifest('test', 'cell'))
    pkg = project.bundle.app_package
    AppEmitter()._emit_policies(project, ctx, pkg)

    base = out / pkg / 'policies' / 'grasp2preplace'
    assert (base / 'policy_card.yaml').is_file()
    assert (base / 'policy.onnx').is_file()             # shipped beside the card
    launch = (out / pkg / 'launch' / 'policy_runtime.launch.py').read_text()
    assert 'policy_runtime_node' in launch
    assert '"hybrid"' in launch
    assert 'grasp2preplace' in launch


# --- model persistence (what the GUI relies on) -------------------------------

def test_policy_step_round_trips_through_yaml(tmp_path):
    from trainit_setup_assistant.model.io import load_project, save_project
    pol = PolicyStep(name='p1', before_waypoint='pre_place', mode=PolicyMode.PURE,
                     card='/tmp/c.yaml', require_attached=True,
                     attached_topic='/suction/attached', check_position=False)
    p = _project(pol)
    f = tmp_path / 'project.yaml'
    save_project(p, f)
    q = load_project(f)
    assert len(q.application.policies) == 1
    got = q.application.policies[0]
    assert got.name == 'p1' and got.mode is PolicyMode.PURE
    assert got.before_waypoint == 'pre_place'
    assert got.require_attached is True and got.check_position is False
    assert q.application.policy_before('pre_place').card == '/tmp/c.yaml'


def test_controller_add_policy_and_clear():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController()
    ctrl.project = _project()
    ctrl.add_policy('p', 'pre_place', mode='hybrid', card='/tmp/c.yaml',
                    require_attached=True, attached_topic='/suction/attached')
    assert len(ctrl.project.application.policies) == 1
    assert ctrl.project.application.policies[0].attached_topic == '/suction/attached'
    ctrl.clear_application()
    assert ctrl.project.application.policies == []
