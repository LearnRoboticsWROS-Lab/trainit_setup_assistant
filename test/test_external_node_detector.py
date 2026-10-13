"""D-015's second kind of method: the EXTERNAL-NODE detector's LAUNCH half.

``trainit_perception``'s registry names two kinds of detection method — ``pure-python`` (a
``Detector`` subclass, which TSA emits a ``detector_node`` for) and ``external-node`` (a
method implemented by a separate node, C++/PCL or a learned model or a vendor SDK, that
publishes the SAME contract on ``/perception/<name>/detections``). For the second kind TSA
emitted nothing at all and had no field for a package, an executable or a parameter file.

What this file pins:

1. **No external detector => byte-identical.** The acceptance criterion. Rendering the
   golden project's ``trainit_bt.launch.py`` and ``package.xml`` must produce the exact
   bytes the published bundle carries, with no regeneration and no re-blessing.
   ``test_generator_golden.py`` and ``test_backend_bytestability.py`` are the bundle-level
   gates; this file asserts the absence of every new token directly, which is the fact
   those gates depend on.
2. **Both kinds launch the same way.** One bundle, one ``trainit_bt.launch.py``, a
   ``Node`` for each: ``trainit_perception``/``detector_node`` for the pure-python entry
   and the project's own package/executable for the external one, both GLOBAL (ADR-0016)
   and both carrying ``use_sim_time``.
3. **The refusals.** A block with no package or no executable is refused BEFORE a file
   exists, naming what is missing; so are an ``external_node`` block on a pure-python
   method, an unknown input role, an empty parameter name, and two detectors resolving to
   one node name.
4. **The input topics are not duplicated.** The launch entry's topic parameters and the
   generated ``perception.yaml``'s ``input:`` block come from ONE resolution, so they can
   never name different cameras.
5. **No ``<exec_depend>`` on the external package** (ADR-0003), but a comment naming it
   and a generation warning.
6. **A real detector's shape.** A segmentation node of the kind a program ships in its
   own package — ``seg_pose_node``, launched by hand with a ``model:=`` argument and a
   parameter file keyed by its node name — is expressible in these fields, and the
   launch entry it produces matches what its hand-written ``seg_pose.launch.py`` does.
   The package name here is a placeholder; the check against the REAL package lives in
   that program's own (private) repository.

The launch assertions read the generated file with ``ast``, not with ``in``: a substring
cannot say which ``Node`` a keyword belongs to, and "which node" is the whole question.

Fast and ROS-free: generation is pure.
"""

import ast
import os
import tempfile
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.generator.external_node_launch import (
    external_node_context, external_node_packages, external_node_problems,
    external_node_warnings, refuse_incomplete_external_nodes)
from trainit_setup_assistant.model import (BundleSpec, CameraSpec, DetectorSpec,
                                           ExternalNodeSpec, PerceptionSpec,
                                           load_project, project_to_dict)
from trainit_setup_assistant.model.project import CanonicalProject
import pending_rebless
import ws_paths

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')
GOLDEN_BUNDLE = ws_paths.GOLDEN_BUNDLE
GOLDEN_PROJECT = GOLDEN_BUNDLE / 'project.yaml'
G_APP = 'fr3wml_suction_camera_tsa_4_1_app'

# Every token the external-node launch half introduces into a generated file. NONE of them
# may appear in a bundle that has no external detector — that is what "byte-identical"
# means, token by token, and it is cheaper to read than a hash when it breaks.
_NEW_TOKENS = ('EXTERNAL-NODE', 'external_node', 'exec_depend> of this bundle')


# --- a real-shaped external detector, as a project would carry it -----------------
# The shape of a real program's detector; `cell_perception` is a placeholder. Launched by hand:
#   ros2 launch cell_perception seg_pose.launch.py model:=/abs/best.onnx
# which is a Node(package='cell_perception', executable='seg_pose_node',
# name='seg_pose_node') with config/capsule.yaml plus two inline parameters. `node_name`
# is explicit because capsule.yaml is a ROS parameter file and opens with `seg_pose_node:`
# — a different node name there means every parameter in the file is silently ignored.
SEG_NODE = dict(
    package='cell_perception',
    executable='seg_pose_node',
    node_name='seg_pose_node',
    parameter_file='config/capsule.yaml',
    parameters={'model': '/opt/models/best.onnx', 'method_3d': 'plane_ransac'},
    topic_parameters={'rgb': 'rgb_topic', 'depth': 'depth_topic',
                      'camera_info': 'camera_info_topic'},
)
SEG_TOPICS = dict(rgb_topic='/bin_camera/color/image_raw',
                     depth_topic='/bin_camera/aligned_depth_to_color/image_raw',
                     camera_info_topic='/bin_camera/color/camera_info')


# --- helpers ----------------------------------------------------------------------

def _nodes(text: str):
    """Every ``Node(...)`` call in a generated launch file, keyword by keyword.

    ``parameters``/``remappings`` come back as source text (``ast.unparse``) because what
    matters about them is the whole expression, including the ``use_sim`` variable and the
    ``get_package_share_directory`` call that a literal-only reader would drop.
    """
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
            return v.value if isinstance(v, ast.Constant) else f'<{ast.unparse(v)}>'

        out.append({
            'package': lit('package'),
            'executable': lit('executable'),
            'name': lit('name'),
            'namespace': lit('namespace'),
            'parameters': (ast.unparse(kw['parameters']) if 'parameters' in kw else ''),
            'remappings': (ast.unparse(kw['remappings']) if 'remappings' in kw else ''),
        })
    return out


def _q(text: str) -> str:
    """Quote-insensitive source, because ``ast.unparse`` normalises to single quotes while
    the template (and the golden) are written with double ones. The quoting is not the
    thing under test; which parameter carries which topic is."""
    return text.replace('"', "'")


def _node(nodes, executable):
    found = [n for n in nodes if n['executable'] == executable]
    assert found, f'no Node(executable="{executable}") in the generated launch'
    assert len(found) == 1, f'{executable} launched {len(found)} times'
    return found[0]


def _project(detectors, *, camera=True):
    p = load_project(EXAMPLE)
    p.bundle = BundleSpec.from_prefix('cell')
    p.perception = p.perception or PerceptionSpec()
    if camera and p.perception.camera is None:
        p.perception.camera = CameraSpec()
    p.perception.detectors = list(detectors)
    return p


def _seg_project(**overrides):
    """A project whose single detector is the real-shaped external-node detector."""
    spec = dict(SEG_NODE)
    spec.update(overrides)
    det = DetectorSpec(name='capsule', method='custom', **SEG_TOPICS,
                       external_node=ExternalNodeSpec(**spec))
    return _project([det])


def _generate(project):
    out = tempfile.mkdtemp()
    manifest = Orchestrator().generate(project, out)
    return Path(out), manifest


def _app_launch(root, pkg='cell_app'):
    return (root / pkg / 'launch' / 'trainit_bt.launch.py').read_text()


# --- 1. no external detector => not one byte changes ------------------------------

@pytest.mark.skipif(not GOLDEN_PROJECT.is_file(), reason='golden bundle not present')
def test_a_project_without_an_external_detector_is_byte_identical():
    """THE acceptance criterion, at byte granularity on the two files that could change.

    The golden carries one pure-python detector and no ``external_node`` block anywhere,
    so the launch file and the manifest must come out exactly as published. Byte-strict on
    purpose: a semantic diff would forgive a stray blank line, and a stray blank line is
    precisely what a new ``{% for %}`` loop adds.
    """
    project = load_project(GOLDEN_PROJECT)
    project.bundle = BundleSpec(
        description_package='fr3wml_suction_camera_tsa_4_1_description',
        moveit_config_package='fr3wml_suction_camera_tsa_4_1_trainit_config',
        app_package=G_APP)
    root, _ = _generate(project)
    for rel in ('launch/trainit_bt.launch.py', 'package.xml'):
        got = (root / G_APP / rel).read_bytes()
        # the published bytes, with exactly the pending lines if any (owner Q1; none since the
        # M2.4 re-bless)
        want = pending_rebless.forward((GOLDEN_BUNDLE / G_APP / rel).read_bytes(),
                                       pending_rebless.GOLDEN, f'{G_APP}/{rel}')
        assert got == want, f'{rel} is no longer byte-identical to the published bundle'


def test_the_new_tokens_appear_nowhere_without_an_external_node_block():
    """The absence the two byte gates depend on, asserted directly and per token."""
    det = DetectorSpec(name='red_object', method='color_mask')
    root, manifest = _generate(_project([det]))
    launch = _app_launch(root)
    manifest_xml = (root / 'cell_app' / 'package.xml').read_text()
    for token in _NEW_TOKENS:
        assert token not in launch, f'"{token}" leaked into a launch with no external node'
        assert token not in manifest_xml, f'"{token}" leaked into package.xml'
    # and the only Node entries are the two that were always there
    assert {n['executable'] for n in _nodes(launch)} == {'detector_node',
                                                         'trainit_run_bt'}
    assert external_node_context(_project([det])) == []
    assert external_node_warnings(_project([det])) == []


# --- 2. both kinds, one bundle, launched the same way -----------------------------

def test_one_bundle_launches_a_pure_python_and_an_external_detector():
    """The point of the feature: ``DetectObject`` cannot tell them apart, and neither can
    the launch — one ``Node`` each, side by side, both global, both on the same clock."""
    pure = DetectorSpec(name='red_object', method='color_mask')
    root, _ = _generate(_project([pure, _seg_project().perception.detectors[0]]))
    launch = _app_launch(root)
    ast.parse(launch)                                  # the bundle must still be valid

    nodes = _nodes(launch)
    pure_node = _node(nodes, 'detector_node')
    ext_node = _node(nodes, 'seg_pose_node')
    assert pure_node['package'] == 'trainit_perception'
    assert ext_node['package'] == 'cell_perception'
    assert ext_node['name'] == 'seg_pose_node'
    # GLOBAL, like the pure-python detectors (ADR-0016): one bin, one detector, however
    # many arms reach into it.
    assert pure_node['namespace'] is None and ext_node['namespace'] is None
    # the same clock: a sim-time mismatch makes every detection's stamp unusable to TF,
    # and nothing in the log says why
    assert "'use_sim_time': use_sim" in _q(ext_node['parameters'])
    # the external node comes AFTER the pure-python ones and BEFORE the BT, which is the
    # order a reader expects and the order the tree's consumer is started in
    order = [n['executable'] for n in nodes]
    assert order == ['detector_node', 'seg_pose_node', 'trainit_run_bt']


def test_the_parameter_file_is_resolved_against_the_package_share():
    """A deployed bundle can only find the file where ``install`` put it.

    ``get_package_share_directory`` + ``os.path.join`` are used because the launch
    template ALREADY imports both — a ``FindPackageShare`` would have added an import to
    every generated launch file and broken the byte gates for bundles that have no
    external detector at all.
    """
    root, _ = _generate(_seg_project())
    launch = _app_launch(root)
    ext = _node(_nodes(launch), 'seg_pose_node')
    assert ("os.path.join(get_package_share_directory('cell_perception'), "
            "'config', 'capsule.yaml')") in _q(ext['parameters'])
    assert 'FindPackageShare' not in launch
    # an ABSOLUTE path is used verbatim: a cell's tuned file may live outside any package
    root2, _ = _generate(_seg_project(parameter_file='/etc/cell/capsule.yaml'))
    ext2 = _node(_nodes(_app_launch(root2)), 'seg_pose_node')
    assert "'/etc/cell/capsule.yaml'" in _q(ext2['parameters'])
    assert 'get_package_share_directory' not in ext2['parameters']


def test_inline_parameters_win_over_the_file_and_are_written_sorted():
    """``model:=`` is CONFIGURATION, not shell history — and the bytes must be stable.

    Inline parameters come after the file in ``parameters=[...]``, which is how ROS 2
    resolves a later value over an earlier one, and which is exactly what a real
    ``seg_pose.launch.py`` does with ``model`` and ``method_3d``.
    """
    root, _ = _generate(_seg_project())
    ext = _node(_nodes(_app_launch(root)), 'seg_pose_node')
    src = ext['parameters']
    assert src.index('capsule.yaml') < src.index('method_3d'), (
        'the inline parameters must come AFTER the parameter file, or the file wins')
    keys = [k for k in ('camera_info_topic', 'depth_topic', 'method_3d', 'model',
                        'rgb_topic', 'use_sim_time')]
    positions = [src.index(k) for k in keys]
    assert positions == sorted(positions), (
        'parameters are written in sorted key order: a dict from YAML carries file order, '
        'not meaning, and the same project must produce the same bytes')


def test_remappings_are_emitted_only_when_asked_and_sorted():
    """For a node whose subscriptions are hardcoded — the node-agnostic way to point any
    node at a topic, needing no cooperation from its author."""
    plain = _node(_nodes(_app_launch(_generate(_seg_project())[0])), 'seg_pose_node')
    assert plain['remappings'] == '', 'no remappings asked for, none emitted'

    root, _ = _generate(_seg_project(
        remappings={'points': '/bin_camera/depth/color/points',
                    'image_raw': '/bin_camera/color/image_raw'}))
    ext = _node(_nodes(_app_launch(root)), 'seg_pose_node')
    assert _q(ext['remappings']) == (
        "[('image_raw', '/bin_camera/color/image_raw'), "
        "('points', '/bin_camera/depth/color/points')]")


# --- 3. the refusals --------------------------------------------------------------

@pytest.mark.parametrize('field', ['package', 'executable'])
def test_a_block_with_no_package_or_no_executable_is_refused(field):
    """Today's failure is a bundle that launches nothing and says nothing. This one names
    the missing field, before a single file exists."""
    project = _seg_project(**{field: ''})
    with pytest.raises(ValueError) as exc:
        refuse_incomplete_external_nodes(project)
    msg = str(exc.value)
    assert f'no {field}' in msg
    assert 'capsule' in msg and '/perception/capsule/detections' in msg

    out = tempfile.mkdtemp()
    with pytest.raises(ValueError):
        Orchestrator().generate(project, out)
    assert not list(Path(out).rglob('*.py')), (
        'the refusal must come BEFORE the first file is written')


def test_an_external_block_on_a_pure_python_method_is_refused():
    """Both halves would publish ``/perception/<name>/detections``, interleaved, and the
    consumer could not tell the two streams apart."""
    det = DetectorSpec(name='red_object', method='color_mask',
                       external_node=ExternalNodeSpec(**SEG_NODE))
    with pytest.raises(ValueError) as exc:
        refuse_incomplete_external_nodes(_project([det]))
    assert 'PURE-PYTHON' in str(exc.value)
    assert 'both would publish' in str(exc.value).lower()


def test_an_unknown_input_role_is_refused_not_ignored():
    """A typo in ``topic_parameters`` would drop a topic SILENTLY — the node would keep
    whatever its own parameter file said, and nothing would say which camera it read."""
    project = _seg_project(topic_parameters={'colour': 'rgb_topic'})
    with pytest.raises(ValueError) as exc:
        refuse_incomplete_external_nodes(project)
    msg = str(exc.value)
    assert 'colour' in msg and 'rgb, depth, camera_info' in msg


def test_an_empty_parameter_name_is_refused():
    project = _seg_project(topic_parameters={'rgb': '  '})
    with pytest.raises(ValueError) as exc:
        refuse_incomplete_external_nodes(project)
    assert 'empty parameter name' in str(exc.value)


def test_a_blank_parameter_file_is_refused():
    """``os.path.join(share, )`` evaluates to the share DIRECTORY, and launch would then
    report "parameter file does not exist" about a path nobody wrote."""
    project = _seg_project(parameter_file='   ')
    with pytest.raises(ValueError) as exc:
        refuse_incomplete_external_nodes(project)
    assert 'parameter_file is blank' in str(exc.value)
    # removing the key entirely is fine: a node may take all its parameters inline
    ok = _seg_project(parameter_file=None)
    refuse_incomplete_external_nodes(ok)
    root, _ = _generate(ok)
    params = _node(_nodes(_app_launch(root)), 'seg_pose_node')['parameters']
    assert 'get_package_share_directory' not in params


def test_the_context_refuses_for_any_caller_not_only_the_orchestrator():
    """An emitter can be driven directly (the byte-stability gate does), so the refusal is
    re-checked where the launch entry is actually built — the pattern
    ``build_perception_yaml`` already follows for its own."""
    with pytest.raises(ValueError):
        external_node_context(_seg_project(executable=''))


def test_two_detectors_cannot_share_one_node_name():
    """Two nodes with one name collide on their parameters and their services, and the
    second to start wins silently."""
    a = DetectorSpec(name='capsule_a', method='custom',
                     external_node=ExternalNodeSpec(package='p', executable='e',
                                                    node_name='seg'))
    b = DetectorSpec(name='capsule_b', method='custom',
                     external_node=ExternalNodeSpec(package='p', executable='e',
                                                    node_name='seg'))
    with pytest.raises(ValueError) as exc:
        refuse_incomplete_external_nodes(_project([a, b]))
    assert 'capsule_a, capsule_b' in str(exc.value) and '"seg"' in str(exc.value)

    # and the DEFAULT node name is per detector, so two blocks without one are fine
    a.external_node.node_name = None
    b.external_node.node_name = None
    refuse_incomplete_external_nodes(_project([a, b]))
    assert sorted(e['node_name'] for e in external_node_context(_project([a, b]))) == [
        'detector_capsule_a', 'detector_capsule_b']


def test_an_external_method_without_a_block_is_warned_about_not_refused():
    """Starting a vendor stack yourself stays legitimate — but the silence does not.

    This was the old behaviour for EVERY external-node method, and the only thing said
    about it was a note from one application template's ``validate()``. It is now said
    app-agnostically, wherever ``perception.yaml`` is emitted.
    """
    det = DetectorSpec(name='capsule', method='custom')
    project = _project([det])
    refuse_incomplete_external_nodes(project)                # not a refusal
    assert external_node_problems(project) == []
    root, manifest = _generate(project)
    warnings = manifest.as_dict().get('warnings', [])
    hit = [w for w in warnings if 'EXTERNAL-NODE method with no external_node block' in w]
    assert hit, warnings
    assert 'perception.detectors["capsule"].external_node' in hit[0]
    # nothing was launched for it, and the file still records it
    assert {n['executable'] for n in _nodes(_app_launch(root))} == {'trainit_run_bt'}
    entry = yaml.safe_load(
        (root / 'cell_app' / 'config' / 'perception.yaml').read_text())
    assert entry['detectors']['capsule']['method'] == 'custom'


# --- 4. the input topics are resolved ONCE ----------------------------------------

def test_the_launch_and_perception_yaml_name_the_same_camera():
    """The one fault no log would ever show: the file TSA writes and the node TSA launches
    looking at different cameras. Both come from ``DetectorSpec.input_topics``."""
    root, _ = _generate(_seg_project())
    entry = yaml.safe_load(
        (root / 'cell_app' / 'config' / 'perception.yaml').read_text())
    inputs = entry['detectors']['capsule']['input']
    assert inputs == {'rgb': SEG_TOPICS['rgb_topic'],
                      'depth': SEG_TOPICS['depth_topic'],
                      'camera_info': SEG_TOPICS['camera_info_topic']}
    params = _node(_nodes(_app_launch(root)), 'seg_pose_node')['parameters']
    for role, param in SEG_NODE['topic_parameters'].items():
        assert f"'{param}': '{inputs[role]}'" in _q(params)


def test_the_detector_falls_back_to_the_cell_camera_then_to_the_defaults():
    """The override chain is the model's, not the generator's, and the external node gets
    the same answer the generated file records."""
    det = DetectorSpec(name='capsule', method='custom',
                       external_node=ExternalNodeSpec(
                           package='p', executable='e',
                           topic_parameters={'rgb': 'rgb_topic'}))
    cam = CameraSpec(rgb_topic='/bin_camera/color/image_raw')
    assert det.input_topics(cam)['rgb'] == '/bin_camera/color/image_raw'
    assert det.input_topics(None)['rgb'] == '/camera/color/image_raw'
    det.rgb_topic = '/override/rgb'
    assert det.input_topics(cam)['rgb'] == '/override/rgb'
    # and nothing about the camera contract was duplicated onto the launch model
    fields = getattr(ExternalNodeSpec, 'model_fields', None) or ExternalNodeSpec.__fields__
    for banned in ('rgb_topic', 'depth_topic', 'camera_info_topic'):
        assert banned not in fields, (
            f'{banned} already lives on DetectorSpec; a second home for it is how the '
            f'file and the node end up looking at different cameras')


# --- 5. the dependency is DOCUMENTED, never declared ------------------------------

def test_the_external_package_is_not_an_exec_depend_but_is_recorded():
    """ADR-0003: Community installs with rosdep. ``rosdep install --from-paths src`` fails
    with "cannot locate rosdep definition" for a package name that is neither a rosdep key
    nor in the workspace — and an external detector is normally customer IP shipped
    separately, so declaring it would break the install of the WHOLE workspace for people
    who never run that cell. It fails where it is USED instead: ``ros2 launch`` names the
    missing package the moment you start the application.
    """
    root, manifest = _generate(_seg_project())
    pkg_xml = (root / 'cell_app' / 'package.xml').read_text()
    import xml.etree.ElementTree as ET
    declared = {e.text.strip() for e in ET.fromstring(pkg_xml).iter('exec_depend')}
    assert 'cell_perception' not in declared
    # recorded where a person looks for dependencies
    assert 'cell_perception' in pkg_xml
    assert 'ADR-0003' in pkg_xml
    assert external_node_packages(_seg_project()) == ['cell_perception']
    hit = [w for w in manifest.as_dict().get('warnings', [])
           if 'cell_perception' in w and 'exec_depend' in w]
    assert hit, 'generation must say which package the bundle launches but will not install'


# --- 6. the real worked example ---------------------------------------------------

def test_a_real_shaped_detector_is_expressible_and_round_trips():
    """The shape is the one a real program's detector has (the real package is checked in
    that program's own repository).

    ``ros2 launch cell_perception seg_pose.launch.py model:=/abs/best.onnx`` is a
    ``Node(package='cell_perception', executable='seg_pose_node', name='seg_pose_node')``
    with ``config/capsule.yaml`` plus the two parameters the launch declares as arguments
    (``model``, ``method_3d``) — every piece of which has a field here. Round-tripped
    through ``project.yaml``, because a field a saved project loses is not configuration.
    """
    project = _seg_project()
    back = CanonicalProject.parse_obj(project_to_dict(project))
    ext = back.perception.detectors[0].external_node
    assert ext == project.perception.detectors[0].external_node
    assert (ext.package, ext.executable) == ('cell_perception', 'seg_pose_node')
    assert ext.parameters['model'].endswith('best.onnx')
    assert ext.resolved_node_name('capsule') == 'seg_pose_node'
    assert back.perception.detectors[0].launches_external_node is True
    assert back.perception.detectors[0].emits_node is False

    # the launch entry it produces, in full
    root, _ = _generate(back)
    ext_node = _node(_nodes(_app_launch(root)), 'seg_pose_node')
    assert ext_node == {
        'package': 'cell_perception',
        'executable': 'seg_pose_node',
        'name': 'seg_pose_node',
        'namespace': None,
        'parameters': ext_node['parameters'],
        'remappings': '',
    }
    for want in ("'model': '/opt/models/best.onnx'",
                 "'method_3d': 'plane_ransac'",
                 "'rgb_topic': '/bin_camera/color/image_raw'",
                 "'use_sim_time': use_sim"):
        assert want in _q(ext_node['parameters']), want


def test_pcl_cluster_is_no_longer_a_name_with_no_way_to_be_used():
    """ADR-0018 notes it as a reserved name with no implementation, and the C++ PCL it
    promises is already installed. The implementation is still somebody's to write, but
    the name can now carry a node: the reservation was useless while TSA had no field for
    the package that would answer to it.
    """
    det = DetectorSpec(name='bin_clusters', method='pcl_cluster',
                       external_node=ExternalNodeSpec(
                           package='trainit_perception_pcl',
                           executable='cluster_node',
                           remappings={'cloud_in': '/bin_camera/depth/color/points'}))
    project = _project([det])
    refuse_incomplete_external_nodes(project)
    root, _ = _generate(project)
    ext = _node(_nodes(_app_launch(root)), 'cluster_node')
    assert ext['package'] == 'trainit_perception_pcl'
    assert ext['name'] == 'detector_bin_clusters'          # the default convention
    assert _q(ext['remappings']) == "[('cloud_in', '/bin_camera/depth/color/points')]"


def test_generation_is_reproducible_with_an_external_detector():
    """Same project + same generator => byte-identical output, the rule the whole
    generator is built on. Mappings came from YAML, so this is the test that catches a
    forgotten ``sorted()``."""
    a, _ = _generate(_seg_project(
        parameters={'b': 2, 'a': 1, 'c': 3},
        remappings={'z': '/z', 'a': '/a', 'm': '/m'}))
    b, _ = _generate(_seg_project(
        parameters={'c': 3, 'a': 1, 'b': 2},
        remappings={'m': '/m', 'z': '/z', 'a': '/a'}))
    assert _app_launch(a, 'cell_app') == _app_launch(b, 'cell_app')


def test_the_launch_template_still_renders_with_no_external_detectors_variable():
    """``StrictUndefined`` is on: a caller that forgets the new variable must fail LOUDLY
    at generation rather than silently render a bundle with no detectors in it."""
    from jinja2 import UndefinedError
    env = build_jinja_env()
    tpl = env.get_template('app/trainit_bt.launch.py.j2')
    common = dict(robot_name='cell', moveit_config_package='cell_trainit_config',
                  app_package='cell_app', tree_filename='pick_place.xml',
                  default_planner_mode='ompl', gazebo_supported=False,
                  cell_namespace='', detectors=[])
    with pytest.raises(UndefinedError):
        tpl.render(**common)
    assert 'EXTERNAL-NODE' not in tpl.render(**common, external_detectors=[])


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))


# --- 7. the GUI does not strip what it cannot edit (vNext M0C #8) -------------------

def _reapply_as_the_perception_page_does(ctrl, det, method=None):
    """The Perception page re-applies EVERY detector it loaded when the user leaves it --
    untouched ones included -- with the fields the page edits, through upsert_detector."""
    ctrl.upsert_detector(det.name, method=method or det.method.value, params=dict(det.params),
                         continuous=det.continuous, rate_hz=det.rate_hz,
                         candidate_rule=det.candidate_rule.value if det.candidate_rule else None)
    return ctrl.project.perception.detector_by_name(det.name)


def test_crossing_the_perception_page_keeps_the_external_node_block():
    """upsert_detector rebuilt the spec from the page's fields and DROPPED ``external_node``:
    walking through the page was enough to lose a hand-written launch block, and the bundle
    then launched nothing for that detector. The topic overrides were already kept."""
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_seg_project())
    det = ctrl.project.perception.detectors[0].copy(deep=True)
    after = _reapply_as_the_perception_page_does(ctrl, det)
    assert after.external_node == det.external_node
    assert (after.rgb_topic, after.depth_topic, after.camera_info_topic) == (
        det.rgb_topic, det.depth_topic, det.camera_info_topic)


def test_changing_the_method_in_the_gui_drops_the_old_methods_launch_block():
    """The block launches the node that IMPLEMENTS the method; switching the method on the page
    is a deliberate change, and a kept block would only make generation refuse it."""
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_seg_project())
    det = ctrl.project.perception.detectors[0].copy(deep=True)
    after = _reapply_as_the_perception_page_does(ctrl, det, method='color_mask')
    assert after.external_node is None
    assert after.rgb_topic == det.rgb_topic                  # the topic overrides still stay
