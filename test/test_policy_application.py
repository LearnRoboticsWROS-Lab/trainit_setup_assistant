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
    # class_id is the CARD name (what the runtime publishes), not the step name
    assert ('<DetectObject topic="/trainit_policy_runtime/target_detections" '
            'class_id="fr5_suction_pickplace:target"') in xml
    assert ('<SetWaypointFromDetection waypoint="pre_place" from="policy_grasp2preplace" '
            'orientation="detected"/>') in xml
    # hybrid KEEPS the deterministic move that executes the decided target
    assert '<MoveWaypoint waypoint="pre_place"/>' in xml
    # ordering: RunPolicy -> DetectObject -> SetWaypoint -> MoveWaypoint (the grasp)
    assert (xml.index('<RunPolicy') < xml.index('<DetectObject')
            < xml.index('<SetWaypointFromDetection waypoint="pre_place"')
            < xml.index('<MoveWaypoint waypoint="pre_place"/>'))
    # HYBRID decides only the grasp; the deterministic MoveWaypoint is self-checking, and
    # the card end_state is the FULL-SKILL (pure) end, NOT the grasp -> no end_state
    # CheckRobotState is emitted for hybrid (ADR-0006). It would wrongly fail otherwise.
    assert '<CheckRobotState' not in xml
    assert '0.4267;-0.278;0.1589' not in xml   # the end_state pose is not asserted


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


def test_hybrid_detectobject_matches_card_name_not_step_name(tmp_path):
    """The runtime tags its target with '<card name>:target'; DetectObject must match the
    CARD name, not the step name (they can differ — regression: they did, and the tree
    could not find the published target)."""
    pol = PolicyStep(name='grasp_policy', before_waypoint='pre_place',
                     mode=PolicyMode.HYBRID, card=_write_card(tmp_path))
    xml = _emit(_project(pol))
    assert 'class_id="fr5_suction_pickplace:target"' in xml    # the CARD name
    assert 'grasp_policy:target' not in xml                    # NOT the step name


# --- validation ----------------------------------------------------------------

def test_validation_flags_unknown_waypoint_and_missing_card():
    app = get_application(AppType.PICK_AND_PLACE)
    pol = PolicyStep(name='p', before_waypoint='nope', card='/does/not/exist.yaml')
    problems = app.validate(_project(pol))
    assert any('unknown waypoint' in m for m in problems)
    assert any('card not found' in m for m in problems)


def _write_card_with(tmp_path, name, **extra):
    """Write a card variant (ADR-0006 axes) to a unique file so validation reads it."""
    import yaml
    doc = yaml.safe_load(_CARD)
    doc['policy_card'].update(extra)
    f = tmp_path / f'{name}.yaml'
    f.write_text(yaml.safe_dump(doc))
    return str(f)


def test_validation_flags_reserved_category_and_end_effector(tmp_path):
    app = get_application(AppType.PICK_AND_PLACE)
    ins = PolicyStep(name='p', before_waypoint='pre_place',
                     card=_write_card_with(tmp_path, 'ins', category='insert'))
    assert any('category "insert" is reserved' in m for m in app.validate(_project(ins)))
    grip = PolicyStep(name='p', before_waypoint='pre_place',
                      card=_write_card_with(tmp_path, 'grip', end_effector='parallel_gripper'))
    assert any('end-effector "parallel_gripper"' in m
               for m in app.validate(_project(grip)))
    ok = PolicyStep(name='p', before_waypoint='pre_place',
                    card=_write_card_with(tmp_path, 'ok', category='reach_grasp',
                                          end_effector='suction'))
    probs = app.validate(_project(ok))
    assert not any('reserved' in m or 'end-effector' in m for m in probs)


def test_validation_flags_residual_mode(tmp_path):
    app = get_application(AppType.PICK_AND_PLACE)
    res = PolicyStep(name='p', before_waypoint='pre_place', mode=PolicyMode.RESIDUAL,
                     card=_write_card(tmp_path))
    assert any('residual' in m and 'not available' in m
               for m in app.validate(_project(res)))
    hyb = PolicyStep(name='p', before_waypoint='pre_place', mode=PolicyMode.HYBRID,
                     card=_write_card(tmp_path))
    assert not any('residual' in m for m in app.validate(_project(hyb)))


def test_calibration_offset_must_have_six_values():
    import pytest
    from pydantic import ValidationError
    for bad in ([0.02], [0.0] * 7):
        with pytest.raises(ValidationError):
            PolicyStep(name='p', before_waypoint='w', card='/tmp/c.yaml',
                       calibration_offset=bad)


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


def _emit_policy_launch(tmp_path, pol, *, detectors=None):
    """Ship + emit the policy runtime launch, return its text (ADR-0006 helper)."""
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    from trainit_setup_assistant.generator.emitters.app_pkg import AppEmitter
    from trainit_setup_assistant.generator.emitters.base import GenContext
    from trainit_setup_assistant.generator.manifest import GenerationManifest
    from trainit_setup_assistant.model.perception import DetectorSpec, PerceptionSpec

    (tmp_path / 'policy.onnx').write_bytes(b'ONNX')
    project = _project(pol)
    if detectors:
        project.perception = PerceptionSpec(
            detectors=[DetectorSpec(name=n, params={'class_id': c}) for n, c in detectors])
    out = tmp_path / 'out'
    ctx = GenContext(out, build_jinja_env(), GenerationManifest('test', 'cell'))
    pkg = project.bundle.app_package
    AppEmitter()._emit_policies(project, ctx, pkg)
    return (out / pkg / 'launch' / 'policy_runtime.launch.py').read_text()


def test_bundle_emits_calibration_and_wires_cell_detector(tmp_path):
    from trainit_setup_assistant.model.enums import CalibrationFrame
    pol = PolicyStep(name='grasp', before_waypoint='pre_place', mode=PolicyMode.HYBRID,
                     card=_write_card(tmp_path),
                     calibration_offset=[0.022, 0.0, 0.0, 0.0, 0.0, 0.0],
                     calibration_frame=CalibrationFrame.BASE)
    # a distinctive class_id (NOT the DetectorSpec default 'object') proves the wiring
    # read the cell detector rather than falling back to a default
    launch = _emit_policy_launch(tmp_path, pol, detectors=[('red_object', 'red_widget')])
    # deploy calibration (ADR-0006) flows to the runtime as the FULL 6-vector param
    assert '[0.022, 0.0, 0.0, 0.0, 0.0, 0.0]' in launch
    assert 'calibration_frame' in launch and 'base_link' in launch
    # detection is wired to the CELL detector, not the card's placeholder
    assert '/perception/red_object/detections' in launch
    assert '"red_widget"' in launch


def test_bundle_without_calibration_omits_the_param(tmp_path):
    pol = PolicyStep(name='grasp', before_waypoint='pre_place', mode=PolicyMode.HYBRID,
                     card=_write_card(tmp_path))
    launch = _emit_policy_launch(tmp_path, pol)
    assert 'calibration_offset' not in launch   # no calibration => param not emitted


def test_app_bringup_includes_policy_runtime_only_when_policies_exist():
    """One command runs the whole stack: the app bringup starts the policy runtime WITH
    the cell so RunPolicy finds its service. No policy => the bringup is byte-identical
    (the {% if has_policies %} block leaves nothing behind)."""
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    tpl = build_jinja_env().get_template('app/bringup.launch.py.j2')
    ctx = dict(robot_name='r', moveit_config_package='r_cfg', app_package='r_app',
               valid_modes_py="('isaac',)", default_mode='isaac', modes_human='isaac',
               gazebo_supported=False, default_planner_mode='ompl')
    with_pol = tpl.render(has_policies=True, **ctx)
    assert 'policy_runtime.launch.py' in with_pol
    assert 'DeclareLaunchArgument("policy"' in with_pol
    without = tpl.render(has_policies=False, **ctx)
    assert 'policy_runtime.launch.py' not in without   # golden bringup unchanged
    assert '"policy"' not in without


def test_app_cmakelists_installs_policies_only_when_policies_exist():
    """The card + .pt/.onnx are shipped under policies/; the CMakeLists must INSTALL that
    dir into share, or the runtime cannot find the card. No policy => byte-identical."""
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    tpl = build_jinja_env().get_template('app/CMakeLists.txt.j2')
    with_pol = tpl.render(package_name='r_app', has_gripper_script=False, has_policies=True)
    assert 'install(DIRECTORY policies' in with_pol
    without = tpl.render(package_name='r_app', has_gripper_script=False, has_policies=False)
    assert 'policies' not in without   # golden CMakeLists unchanged


# --- model persistence (what the GUI relies on) -------------------------------

def test_policy_step_round_trips_through_yaml(tmp_path):
    from trainit_setup_assistant.model.io import load_project, save_project
    pol = PolicyStep(name='p1', before_waypoint='pre_place', mode=PolicyMode.PURE,
                     card='/tmp/c.yaml', require_attached=True,
                     attached_topic='/suction/attached', check_position=False,
                     calibration_offset=[0.022, 0.0, 0.0, 0.0, 0.0, 0.0],
                     calibration_note='systematic +2cm x bias, run-7')
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
    # ADR-0006 calibration fields survive the round-trip
    assert got.calibration_offset == [0.022, 0.0, 0.0, 0.0, 0.0, 0.0]
    assert got.calibration_note == 'systematic +2cm x bias, run-7'


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


def test_controller_parses_calibration_string():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController()
    ctrl.project = _project()
    ctrl.add_policy('p', 'pre_place', card='/tmp/c.yaml', calibration='0.022, 0, 0, 0, 0, 0')
    assert ctrl.project.application.policies[0].calibration_offset == [0.022, 0, 0, 0, 0, 0]
    # an all-zero offset means "no calibration" (keeps the bundle byte-stable)
    ctrl.clear_application()
    ctrl.add_policy('p', 'pre_place', card='/tmp/c.yaml', calibration='0,0,0,0,0,0')
    assert ctrl.project.application.policies[0].calibration_offset is None
