"""ADR-0016 Decision 3: ``cell_namespace`` — the robot slot.

Two halves, and the first one is the acceptance criterion of the whole ADR:

1. **Empty emits nothing.** The default puts every node at the root and not one byte of a
   generated bundle differs from a pre-ADR-0016 one. ``test_generator_golden.py`` and
   ``test_backend_bytestability.py`` pin that at bundle granularity with no regeneration;
   this file pins the absence of every namespace token in the files that would carry one.

2. **Set, it reaches exactly the right nodes.** ``move_group``, the controllers (the
   ``ros2_control_node`` AND the spawners' ``--controller-manager``),
   ``robot_state_publisher``, ``scene_manager_node``, the ``servo_node``, the topics servo
   derives, the robot's own bridges, the gripper action in ``bt_params.yaml`` and the BT
   application — and NOT ``/tf``/``/tf_static``, the camera, the detector nodes or RViz.

The per-node assertions read the generated launch files with ``ast``, not with ``in``: a
substring search cannot tell which node a ``namespace=`` belongs to, and "which node" is
the entire question here.

Fast and ROS-free: generation is pure.
"""

import ast
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications import build_bt_params_context
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.namespace import (cell_namespace,
                                                         namespace_handoff_warnings)
from trainit_setup_assistant.generator.servo_yaml import build_servo_context
from trainit_setup_assistant.model import (BundleSpec, CanonicalProject, ExecutionMode,
                                           MotionType, VisionBinding, load_project,
                                           normalize_cell_namespace, project_to_dict)
from trainit_setup_assistant.model.robot import RobotSpec
import identity_cases
import ws_paths

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')
GOLDEN_BUNDLE = ws_paths.GOLDEN_BUNDLE
GOLDEN_PROJECT = GOLDEN_BUNDLE / 'project.yaml'

# Every token the namespaced bring-up introduces. NONE of them may appear when the field
# is empty — that is what "empty emits nothing" means, file by file.
_NS_TOKENS = ('NS = ', 'namespace=NS', 'tf_global', 'CAMERA_BRIDGE_PACKAGES',
              'publish_robot_description_semantic')


# --- helpers --------------------------------------------------------------------

def _nodes(text: str):
    """Every ``Node(...)`` call in a generated launch file, as a dict of its literal
    keywords. ``namespace=NS`` comes back as ``'<NS>'`` (the launch's own constant)."""
    out = []
    for node in ast.walk(ast.parse(text)):
        if not (isinstance(node, ast.Call)
                and getattr(node.func, 'id', None) == 'Node'):
            continue
        kw = {k.arg: k.value for k in node.keywords if k.arg}

        def lit(key):
            v = kw.get(key)
            if v is None:
                return None
            if isinstance(v, ast.Constant):
                return v.value
            if isinstance(v, ast.Name):
                return f'<{v.id}>'
            return '<expr>'

        out.append({
            'package': lit('package'),
            'executable': lit('executable'),
            'name': lit('name'),
            'namespace': lit('namespace'),
            'remappings': (ast.unparse(kw['remappings']) if 'remappings' in kw else ''),
        })
    return out


def _motion_planning(rviz_text: str) -> dict:
    """The MotionPlanning display of a generated moveit.rviz. Parsed, not grepped: the
    .rviz is YAML, RViz will not start on a broken one, and the namespaced version adds
    comments inside a nested mapping — exactly where a hand-written template goes wrong.
    """
    displays = yaml.safe_load(rviz_text)['Visualization Manager']['Displays']
    return next(d for d in displays
                if d['Class'] == 'moveit_rviz_plugin/MotionPlanning')


def _node(nodes, executable):
    found = [n for n in nodes if n['executable'] == executable]
    assert found, f'no Node(executable="{executable}") in the generated launch'
    return found[0]


def _project(source, namespace='', servo=False):
    p = load_project(str(source))
    p.bundle = BundleSpec.from_prefix('cell')
    p.robot.cell_namespace = namespace
    if servo:
        seg = p.application.segment_for('pre_pick')
        seg.execution = ExecutionMode.SERVO
        seg.motion = MotionType.FREE
        # a streamed move wants a goal that MOVES: bind it, so the fixture is not
        # testing the "static taught pose" warning instead
        det = (p.perception.detectors[0].name
               if (p.perception and p.perception.detectors) else 'capsules')
        p.application.waypoint_by_name('pre_pick').vision = VisionBinding(detector=det)
    return p


def _generate(project):
    tmp = tempfile.TemporaryDirectory()
    manifest = Orchestrator().generate(project, tmp.name)
    return tmp, Path(tmp.name), manifest


# --- 1. the field itself --------------------------------------------------------

def test_default_is_empty_and_means_the_root():
    p = load_project(EXAMPLE)
    assert p.robot.cell_namespace == ''
    assert p.robot.namespace() == ''
    assert cell_namespace(p) == ''
    # the one-robot case never meets the field: an absolute name stays absolute
    assert p.robot.under_namespace('joint_states') == '/joint_states'
    assert p.robot.under_namespace('/joint_states') == '/joint_states'


@pytest.mark.parametrize('raw,want', [
    ('', ''), ('/', ''), ('   ', ''),
    ('r3', '/r3'), ('/r3', '/r3'), ('r3/', '/r3'), (' /r3/ ', '/r3'),
    ('cell/r3', '/cell/r3'), ('/cell/r3/', '/cell/r3'), ('_r3', '/_r3'),
])
def test_namespace_is_canonicalised(raw, want):
    assert normalize_cell_namespace(raw) == want
    assert RobotSpec.parse_obj(
        {**_min_robot(), 'cell_namespace': raw}).cell_namespace == want


@pytest.mark.parametrize('bad', ['3r', 'r3 arm', '/r-3', 'r3//arm', 'a/3b'])
def test_an_illegal_namespace_is_refused(bad):
    """A bad namespace produces a bring-up whose nodes silently never find each other —
    so it fails at load, not at run time."""
    with pytest.raises(Exception):
        normalize_cell_namespace(bad)


def _min_robot():
    return {'robot_name': 'r', 'description': {'urdf_dir': '/tmp', 'top_xacro': 'r.xacro'},
            'planning_group': {'name': 'arm'}}


def test_namespace_survives_the_yaml_round_trip():
    p = load_project(EXAMPLE)
    p.robot.cell_namespace = 'r3'
    again = CanonicalProject.parse_obj(project_to_dict(p))
    assert again == p
    assert again.robot.cell_namespace == '/r3'


def test_under_namespace_is_idempotent():
    p = load_project(EXAMPLE)
    p.robot.cell_namespace = 'r3'
    once = p.robot.under_namespace('/suctioncup_controller/gripper_command')
    assert once == '/r3/suctioncup_controller/gripper_command'
    assert p.robot.under_namespace(once) == once       # calling twice cannot double it


# --- 2. empty emits nothing -----------------------------------------------------

@pytest.mark.parametrize('source', [EXAMPLE, GOLDEN_PROJECT])
def test_empty_namespace_emits_no_namespace_token_anywhere(source):
    if not Path(source).is_file():
        pytest.skip('golden bundle not present')
    if source == EXAMPLE:                         # the golden param keeps its own skip above
        identity_cases.require('p2_example')
    tmp, out, _ = _generate(_project(source, '', servo=True))
    with tmp:
        touched = sorted(out.rglob('*.launch.py')) + sorted(out.rglob('*.rviz'))
        assert touched
        for f in touched:
            text = f.read_text()
            for token in _NS_TOKENS:
                assert token not in text, f'{f.name} carries "{token}" with no namespace'
            if f.name == 'moveit.rviz':      # view_robot.rviz has no MotionPlanning
                mp = _motion_planning(text)
                assert 'Move Group Namespace' not in mp
                assert mp['Planning Scene Topic'] == 'monitored_planning_scene'
        # and the names that derive from it stay at the root
        s = yaml.safe_load(
            (out / 'cell_trainit_config' / 'config' / 'servo.yaml').read_text())
        s = s['/**']['ros__parameters']['moveit_servo']
        assert s['joint_topic'] == '/joint_states'
        assert s['monitored_planning_scene_topic'] == '/monitored_planning_scene'


def test_empty_namespace_produces_no_handoff_warning():
    """The warnings exist to catch what a namespace does NOT fix. With no namespace there
    is nothing to catch, so a single-robot project's manifest is unchanged."""
    assert namespace_handoff_warnings(load_project(EXAMPLE)) == []


# --- 3. set: it reaches the robot's motion stack --------------------------------

def test_namespace_reaches_every_node_of_the_motion_stack():
    identity_cases.require('p2_example')
    tmp, out, _ = _generate(_project(EXAMPLE, 'r3', servo=True))
    with tmp:
        text = (out / 'cell_app' / 'launch' / 'bringup.launch.py').read_text()
        ast.parse(text)                                   # still valid python
        assert 'NS = "/r3"' in text
        nodes = _nodes(text)
        for exe in ('robot_state_publisher', 'move_group', 'ros2_control_node',
                    'spawner', 'servo_node_main'):
            assert _node(nodes, exe)['namespace'] == '<NS>', f'{exe} is not namespaced'
        # the controllers are reached through the controller MANAGER's name too: a
        # spawner that still asks for /controller_manager finds nobody.
        assert '"--controller-manager", f"{NS}/controller_manager"' in text
        assert '"--controller-manager", "/controller_manager"' not in text


def test_namespace_reaches_the_scene_manager_in_the_cell_bringup():
    """The scene manager drives THIS robot's planning scene. TMR 1.5.0 asks for
    apply/get_planning_scene by RELATIVE name, so the namespace is the whole fix — two
    robots stop writing into one global service."""
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, _ = _generate(_project(GOLDEN_PROJECT, 'r3'))
    with tmp:
        text = (out / 'cell_trainit_config' / 'launch' / 'bringup.launch.py').read_text()
        ast.parse(text)
        nodes = _nodes(text)
        assert _node(nodes, 'scene_manager_node')['namespace'] == '<NS>'
        for exe in ('robot_state_publisher', 'move_group', 'ros2_control_node',
                    'spawner'):
            assert _node(nodes, exe)['namespace'] == '<NS>', f'{exe} is not namespaced'


def test_namespace_reaches_the_bt_application_but_not_the_detectors():
    """trainit_run_bt is a CLIENT of this robot's move_group (MoveGroupInterface resolves
    the action against its own node namespace), so it lives in the slot. The detectors do
    not: they publish /perception/<name>/detections, which DetectObject reads by absolute
    name, and one detector serves every arm reaching into the bin."""
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, _ = _generate(_project(GOLDEN_PROJECT, 'r3'))
    with tmp:
        text = (out / 'cell_app' / 'launch' / 'trainit_bt.launch.py').read_text()
        ast.parse(text)
        nodes = _nodes(text)
        assert _node(nodes, 'trainit_run_bt')['namespace'] == '<NS>'
        detectors = [n for n in nodes if n['executable'] == 'detector_node']
        assert detectors, 'the golden has a detector; its node should be generated'
        for d in detectors:
            assert d['namespace'] is None, 'a detector must stay GLOBAL (ADR-0016)'


def test_namespace_reaches_the_topics_servo_derives():
    p = _project(EXAMPLE, 'r3', servo=True)
    ctx = build_servo_context(p)
    assert ctx['joint_topic'] == '/r3/joint_states'
    assert ctx['command_out_topic'] == '/r3/moveit_joint_controller/joint_trajectory'
    assert ctx['monitored_planning_scene_topic'] == '/r3/monitored_planning_scene'
    # frames are NOT topics: they must not grow a namespace
    assert ctx['planning_frame'] == p.robot.base_frame
    assert ctx['ee_frame_name'] == p.robot.tip_link


def test_namespace_reaches_the_gripper_action_in_bt_params():
    """The one name in bt_params.yaml that names a CONTROLLER, so it is the one that has
    to follow the controllers."""
    p = _project(EXAMPLE, '')
    assert build_bt_params_context(p)['g']['gripper_action'] == \
        '/suctioncup_controller/gripper_command'
    p.robot.cell_namespace = 'r3'
    assert build_bt_params_context(p)['g']['gripper_action'] == \
        '/r3/suctioncup_controller/gripper_command'


def test_a_cell_with_no_gripper_emits_an_empty_action_under_a_namespace():
    """Empty must stay empty: '/r3' alone is not an action name."""
    from trainit_setup_assistant.model.enums import GripperKind
    p = _project(EXAMPLE, 'r3')
    p.robot.gripper.controller_name = None
    p.robot.gripper.kind = GripperKind.NONE
    assert build_bt_params_context(p)['g']['gripper_action'] == ''


# --- 4. set: and NOT the things it must not touch -------------------------------

def test_tf_stays_global_on_every_namespaced_node():
    """tf2 declares /tf and /tf_static absolute, and a split TF tree is a cell that cannot
    see itself. Every namespaced node says so explicitly."""
    identity_cases.require('p2_example')
    tmp, out, _ = _generate(_project(EXAMPLE, 'r3', servo=True))
    with tmp:
        text = (out / 'cell_app' / 'launch' / 'bringup.launch.py').read_text()
        assert 'tf_global = [("/tf", "/tf"), ("/tf_static", "/tf_static")]' in text
        nodes = _nodes(text)
        for exe in ('robot_state_publisher', 'move_group', 'servo_node_main'):
            assert 'tf_global' in _node(nodes, exe)['remappings'], \
                f'{exe} does not keep tf global'
        # never a namespaced tf
        assert 'f"{NS}/tf"' not in text and '"/r3/tf"' not in text


def test_the_camera_stays_at_the_root():
    """A camera serves a STATION, not a robot: in the reference cell the bin camera
    publishes under its own name while serving one arm of four."""
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, _ = _generate(_project(GOLDEN_PROJECT, 'r3'))
    with tmp:
        text = (out / 'cell_trainit_config' / 'launch' / 'bringup.launch.py').read_text()
        assert 'CAMERA_BRIDGE_PACKAGES = ("depth_image_proc",)' in text
        assert 'if b["package"] not in CAMERA_BRIDGE_PACKAGES:' in text
        # the camera's own topics are absolute and untouched
        assert '/camera/depth/color/points' in text
        assert '/r3/camera' not in text


def test_perception_topics_are_never_namespaced():
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, _ = _generate(_project(GOLDEN_PROJECT, 'r3'))
    with tmp:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.suffix in ('.py', '.yaml', '.xml'):
                assert '/r3/perception' not in f.read_text(), f'{f.name}'


def test_rviz_runs_at_the_root_with_the_description_topics_remapped():
    """Measured, not stylistic: MoveIt's MotionPlanning plugin builds no end-effector
    interactive marker from inside a namespace, so the CELL is namespaced and the VIEWER
    is not — and what RViz needs is pointed at this robot by hand."""
    identity_cases.require('p2_example')
    tmp, out, _ = _generate(_project(EXAMPLE, 'r3'))
    with tmp:
        text = (out / 'cell_app' / 'launch' / 'bringup.launch.py').read_text()
        rviz = _node(_nodes(text), 'rviz2')
        assert rviz['namespace'] is None, 'RViz must NOT be namespaced'
        # ast.unparse normalises the quoting, so compare on the f-string's body
        assert "{NS}/robot_description'" in rviz['remappings']
        assert "{NS}/robot_description_semantic'" in rviz['remappings']
        # The .rviz is YAML and RViz will not start on a broken one, so parse it
        # rather than grepping it: the namespaced version adds comments inside a
        # nested mapping, which is exactly where a hand-written template goes wrong.
        mp = _motion_planning(
            (out / 'cell_trainit_config' / 'config' / 'moveit.rviz').read_text())
        assert mp['Move Group Namespace'] == '/r3'
        assert mp['Planning Scene Topic'] == '/r3/monitored_planning_scene'
        assert mp['Planned Path']['Trajectory Topic'] == '/r3/display_planned_path'
        # "Robot Description" must NOT be absolute: the plugin puts it inside a node
        # name, where a '/' is illegal — the bring-up remaps the topic instead.
        assert mp['Robot Description'] == 'robot_description'


# --- 5. what a namespace does NOT fix is SAID, not silently rewritten -----------

def test_captured_absolute_cell_data_is_reported_not_rewritten():
    """The bridge parameters and the arm /joint_states remap were captured VERBATIM from
    the hand-made base bring-up. An absolute root name never follows a namespace, and a
    silent rewrite of a captured name is how two halves of a cell start disagreeing — so
    it is reported instead."""
    p = _project(EXAMPLE, 'r3')
    warnings = '\n'.join(namespace_handoff_warnings(p))
    assert 'arm_joint_states_remap_to' in warnings
    assert '/joint_states_robot' in warnings
    assert 'absolute ROOT name' in warnings
    assert 'gripper_suction_action_server' in warnings
    # and the camera-side bridge is NOT reported: it is meant to stay at the root
    assert 'depth_image_proc' not in warnings


def test_the_warnings_reach_the_generation_manifest():
    identity_cases.require('p2_example')
    tmp, out, manifest = _generate(_project(EXAMPLE, 'r3'))
    with tmp:
        text = '\n'.join(manifest.as_dict().get('warnings', []))
        assert 'cell_namespace "/r3"' in text


def test_gazebo_is_declared_unverified_under_a_namespace():
    """gazebo_ros2_control owns /controller_manager inside gzserver and the name comes
    from the URDF plugin tag, not from the launch — so this field cannot namespace it, and
    saying so beats generating a cell that cannot start."""
    from trainit_setup_assistant.model.enums import Backend
    p = _project(EXAMPLE, 'r3')
    p.deployment.modes = list(p.deployment.modes) + [Backend.GAZEBO]
    text = '\n'.join(namespace_handoff_warnings(p))
    assert 'GAZEBO backend is NOT namespaced' in text


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


# --- the parameter FILES follow the namespace too (vNext M0C #5) ------------------
# Verified by execution (2026-10-06, rclpy on Humble): a params file keyed by a BARE node
# name binds that node in / and binds NOTHING in /r3; `/**/<name>` binds both. A namespaced
# bundle that keeps bare keys brings the node up with its defaults and reports it ready --
# the first customer cell measured "loading 0 object(s)" from its scene manager exactly so.

def _top_keys(path: Path) -> list:
    return list(yaml.safe_load(path.read_text()))


def test_the_scene_yaml_binds_a_namespaced_scene_manager():
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, _ = _generate(_project(GOLDEN_PROJECT, 'r3'))
    with tmp:
        assert _top_keys(out / 'cell_trainit_config' / 'config' / 'scene.yaml') == [
            '/**/scene_manager_node']


def test_at_the_root_the_scene_yaml_key_is_the_bare_one_it_always_was():
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, _ = _generate(_project(GOLDEN_PROJECT, ''))
    with tmp:
        assert _top_keys(out / 'cell_trainit_config' / 'config' / 'scene.yaml') == [
            'scene_manager_node']


def test_the_generated_controller_file_binds_a_namespaced_controller_manager():
    """From scratch, TSA renders ros2_controllers.yaml itself: every node key it writes
    must reach the controller manager and the arm controller in the robot's slot."""
    identity_cases.require('p2_example')
    tmp, out, _ = _generate(_project(EXAMPLE, 'r3'))
    with tmp:
        keys = _top_keys(out / 'cell_trainit_config' / 'config' / 'ros2_controllers.yaml')
    assert keys and all(k.startswith('/**/') for k in keys), keys
    tmp, out, _ = _generate(_project(EXAMPLE, ''))
    with tmp:
        keys = _top_keys(out / 'cell_trainit_config' / 'config' / 'ros2_controllers.yaml')
    assert keys and not any(k.startswith('/') for k in keys), keys       # root: unchanged


def test_a_copied_controller_file_with_bare_keys_is_reported_under_a_namespace():
    """The scene-loader flow COPIES the base's ros2_controllers.yaml -- cell data, which the
    generator reports rather than rewrites (namespace_handoff_warnings). Bare keys there
    bind nothing under the slot, so they are named in a warning."""
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    tmp, out, manifest = _generate(_project(GOLDEN_PROJECT, 'r3'))
    with tmp:
        warned = [w for w in manifest.warnings if 'ros2_controllers.yaml' in w]
    assert warned and 'controller_manager' in warned[0] and '/**/' in warned[0], warned
    tmp, out, manifest = _generate(_project(GOLDEN_PROJECT, ''))
    with tmp:
        assert not [w for w in manifest.warnings if 'ros2_controllers.yaml' in w]
