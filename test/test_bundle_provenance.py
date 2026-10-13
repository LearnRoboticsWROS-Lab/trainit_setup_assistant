"""Every bundle says which capability snapshot compiled it, and names no machine (vNext M2.3).

ADR-0021, amendment M2.3: ``generation_manifest.yaml`` records, right after ``generator_version``,
``capability_snapshot`` -- the identity of the committed catalog this TSA built its vocabulary
from: ``catalog_version`` and the sha256 of ``catalog/snapshot.yaml``; for TMR's planner registry
and trainit_perception's vocabulary, the release, the schema and the sha256 of TSA's committed
copy. Hashes and versions only: never the catalog's content, never what is installed. And a
COPY's source is ``package://<package>/<path>`` or ``external:<file name>``, never the
generating machine's path (owner answer Q1). These tests hold:

1. the TSA under test is THIS source tree -- an older install earlier on the path would generate
   the same bundles but for the version, and prove nothing;
2. the identity is the committed catalog's, computed by the test itself (``catalog_identity``):
   the same closed block in every bundle and in the Step-3 config, the first keys of the
   manifest, consistent with what the catalog records, and no dependency of the bundle;
3. one changed snapshot byte changes the identity -- and NOTHING else of the bundle: every other
   file and the manifest's content record are the same (the brief's "M2.3 may change provenance
   metadata, not application behaviour", held semantically); a catalog changed behind a running
   TSA is refused before a file is written;
4. the source rule: on synthetic trees (nearest package.xml, its <name>, innermost package,
   symlinks resolved, relative paths, an unusable package.xml), for TSA's own resources however
   TSA is installed, and for the golden's twelve copied files;
5. no manifest names this machine, a candidate provider, or a path in a warning.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from importlib import resources
from pathlib import Path

import pytest
import yaml

import catalog_identity
import identity_cases
import trainit_setup_assistant as T
import ws_paths
from trainit_setup_assistant import catalog
from trainit_setup_assistant.catalog import REFRESH
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator import orchestrator
from trainit_setup_assistant.generator.determinism import canonical_yaml
from trainit_setup_assistant.generator.orchestrator import generate_scene_loader_config
from trainit_setup_assistant.catalog import refresh as refresh_tool
from trainit_setup_assistant.generator.provenance import SourceNamer, portable_source

TEST = Path(__file__).resolve().parent
PKG = TEST.parent
WORKER = TEST / 'identity_worker.py'
MANIFEST = 'generation_manifest.yaml'
HEX = re.compile(r'^[0-9a-f]{64}$')
RELEASE = re.compile(r'^\d+\.\d+\.\d+$')
PORTABLE = re.compile(r'^package://[a-zA-Z0-9][a-zA-Z0-9_-]*/[^\s]+$|^external:[^/\\]+$')
KIT = (PKG / 'trainit_setup_assistant' / 'resources' / 'gazebo_cell_kit' / 'scripts'
       / 'link_attacher_bridge.py')
KIT_SOURCE = ('package://trainit_setup_assistant/trainit_setup_assistant/resources/'
              'gazebo_cell_kit/scripts/link_attacher_bridge.py')
GOLDEN_COPIES = ['fr3wml_suction.ros2_control.xacro', 'fr3wml_suction.urdf.xacro',
                 'initial_positions.yaml', 'fr3wml_suction.ros2_control.xacro',
                 'fr3wml_suction.srdf', 'fr3wml_suction.urdf.xacro', 'initial_positions.yaml',
                 'joint_limits.yaml', 'kinematics.yaml', 'moveit.rviz', 'moveit_controllers.yaml',
                 'ompl_planning.yaml', 'pilz_cartesian_limits.yaml',
                 'pilz_industrial_motion_planner_planning.yaml', 'ros2_controllers.yaml']


_CARD = textwrap.dedent("""
    policy_card:
      name: synthetic_pick
      files: {jit: policy.pt, onnx: policy.onnx}
      end_state:
        tcp_pose_base: [0.4, -0.2, 0.15, 0.0, -0.7, 0.71, 0.01]
        object_attached: true
        suction_active: true
""")


def _policy_project(card_dir: Path, card: str = _CARD, models=('policy.onnx', 'policy.pt')):
    """The golden with one policy step whose card (and models) sit in ``card_dir``."""
    from trainit_setup_assistant.model import PolicyStep
    from trainit_setup_assistant.model.enums import PolicyMode
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / 'policy_card.yaml').write_text(card)
    for name in models:
        (card_dir / name).write_bytes(b'MODEL')
    p = identity_cases.golden()
    p.application.policies = [PolicyStep(
        name='grasp2preplace', before_waypoint='pre_place', mode=PolicyMode.PURE,
        card=str(card_dir / 'policy_card.yaml'), require_attached=True,
        attached_topic='/suction/attached')]
    return p


def _gazebo_kit_manifest(root: Path) -> dict:
    """A Step-3 config of a Gazebo link-attacher cell: it COPIES TSA's own kit script."""
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    from trainit_setup_assistant.generator.emitters.base import GenContext
    from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
        SceneLoaderMoveitConfigEmitter)
    from trainit_setup_assistant.generator.manifest import GenerationManifest
    from trainit_setup_assistant.model import (Backend, BundleSpec, CanonicalProject,
                                               DeploymentSpec, MotionSegment, SceneObject,
                                               Waypoint)
    from trainit_setup_assistant.model.enums import (AppType, GripperKind, SceneObjectCategory,
                                                     ShapeType, WaypointType)
    from trainit_setup_assistant.model.robot import (ArmControllerSpec, DescriptionSource,
                                                     GripperSpec, PlanningGroupSpec, RobotSpec)
    cfg = root / 'gz' / 'base' / 'config'
    cfg.mkdir(parents=True)
    (root / 'gz' / 'urdf').mkdir()
    (cfg / 'ur.urdf.xacro').write_text('<?xml version="1.0"?>\n<robot '
                                      'xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"/>\n')
    (cfg / 'ur.srdf').write_text('<?xml version="1.0"?>\n<robot name="r"></robot>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml',
              'moveit_controllers.yaml', 'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    gripper = GripperSpec(kind=GripperKind.PARALLEL, controller_name='gripper_position_controller',
                          command_joint='robotiq_85_left_knuckle_joint')    # gazebo: link_attacher
    p = CanonicalProject(project_name='ur5', bundle=BundleSpec.from_prefix('ur5'), robot=RobotSpec(
        robot_name='ur', description=DescriptionSource(urdf_dir=str(root / 'gz' / 'urdf'),
                                                       top_xacro='x.urdf.xacro'),
        base_frame='base_link', tip_link='tcp', base_moveit_config_path=str(root / 'gz' / 'base'),
        planning_group=PlanningGroupSpec(name='arm', joints=['j1']), gripper=gripper,
        arm_controller=ArmControllerSpec()))
    p.deployment = DeploymentSpec(modes=[Backend.MOCK, Backend.GAZEBO],
                                  default_mode=Backend.GAZEBO)
    p.scene.gripper_cmd_topic = '/gripper_cmd'
    p.scene.attach_link = 'wrist_3_link'
    p.scene.objects = [SceneObject(id='red_cube', shape=ShapeType.BOX, dims=[0.05, 0.05, 0.05],
                                   position=[0.5, 0, 0.3], category=SceneObjectCategory.DYNAMIC,
                                   grasp_target=True)]
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='home', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='home')]
    p.application.sequence = ['home']
    manifest = GenerationManifest('ur5', 'ur5')
    SceneLoaderMoveitConfigEmitter().emit(p, GenContext(root / 'gz' / 'out', build_jinja_env(),
                                                        manifest))
    return manifest.as_dict()


def _no_package_above(path: Path) -> bool:
    return not any((parent / 'package.xml').is_file() for parent in [path, *path.parents])


def _released() -> str:
    return ET.parse(PKG / 'package.xml').getroot().findtext('version')


def _generate(project, out: Path) -> dict:
    Orchestrator().generate(project, out)
    return yaml.safe_load((out / MANIFEST).read_bytes())


@pytest.fixture(scope='module')
def bundles():
    """The manifests of the bundles these tests read, generated once (the example is 45 MB)."""
    root = Path(tempfile.mkdtemp(prefix='tsa_provenance_'))
    out = {}
    try:
        for case in ('p1_golden', 'p2_example', 'pol1_policy_test4_project'):
            if not identity_cases.missing(case):
                bundle = root / case
                out[case] = _generate(identity_cases.CASES[case](), bundle)
                out[case + ':text'] = (bundle / MANIFEST).read_text(encoding='utf-8')
                out[case + ':files'] = {p.relative_to(bundle).as_posix(): p.read_bytes()
                                        for p in bundle.rglob('*') if p.is_file()}
        if not identity_cases.missing('p1_golden'):
            # a policy card with its models in no package (the case trainit_lab is in), and a
            # Step-3 config that copies TSA's own Gazebo kit -- both built here, not untracked
            bundle = root / 'policy_synthetic'
            out['policy_synthetic'] = _generate(_policy_project(root / 'cards' / 'pick'), bundle)
            out['policy_synthetic:text'] = (bundle / MANIFEST).read_text(encoding='utf-8')
            step3 = root / 'step3'
            generate_scene_loader_config(identity_cases.golden(), step3,
                                         'fr3wml_suction_scene_loader_moveit_config')
            out['step3'] = yaml.safe_load((step3 / MANIFEST).read_bytes())
            out['step3:text'] = (step3 / MANIFEST).read_text(encoding='utf-8')
        # a synthetic project: needs nothing of the workspace
        out['gazebo_kit'] = _gazebo_kit_manifest(root)
        out['gazebo_kit:text'] = canonical_yaml(out['gazebo_kit'])
        out['root'] = str(root)
        out['no_package_above_root'] = _no_package_above(root)
        yield out
    finally:
        shutil.rmtree(root, ignore_errors=True)


def _manifests(bundles):
    found = [k for k in ('p1_golden', 'p2_example', 'pol1_policy_test4_project', 'step3',
                         'policy_synthetic') if k in bundles]
    if not found:
        pytest.skip('needs the golden sibling checkout or fr3wml_digital_twin/fr5_app')
    return found


# --- 1. the TSA under test ------------------------------------------------------------------------

def test_the_tsa_under_test_is_this_source_tree():
    assert Path(T.__file__).resolve() == PKG / 'trainit_setup_assistant' / '__init__.py', (
        f'{T.__file__} is not this source tree: run the tests from {PKG} with '
        f'PYTHONPATH=$PWD:$PYTHONPATH (TESTING.md)')
    assert T.__version__ == _released()
    assert Path(str(resources.files('trainit_setup_assistant.catalog'))).resolve() == \
        catalog_identity.CATALOG


# --- 2. the identity ------------------------------------------------------------------------------

def test_every_manifest_records_the_committed_snapshot_identity(bundles):
    expected = catalog_identity.committed_identity()
    for name in _manifests(bundles):
        manifest = bundles[name]
        assert list(manifest)[:2] == ['generator_version', 'capability_snapshot'], name
        assert manifest['generator_version'] == _released(), name
        assert manifest['capability_snapshot'] == expected, name
        assert bundles[name + ':text'].startswith(f'generator_version: {_released()}\n'
                                                  f'capability_snapshot:\n'), name


def test_the_identity_is_a_closed_block_of_versions_and_hashes():
    identity = catalog.snapshot_identity()
    assert set(identity) == {'catalog_version', 'catalog_sha256', 'trainit_motion_runtime',
                             'trainit_perception'}
    assert set(identity['trainit_motion_runtime']) == {
        'version', 'planner_registry_schema', 'planner_registry_sha256'}
    assert set(identity['trainit_perception']) == {'version', 'vocabulary_schema',
                                                   'vocabulary_sha256'}
    assert HEX.match(identity['catalog_sha256'])
    for component in ('trainit_motion_runtime', 'trainit_perception'):
        block = identity[component]
        assert RELEASE.match(block['version']), block
        for key, value in block.items():
            if key.endswith('_sha256'):
                assert HEX.match(value), (component, key)
            elif key.endswith('_schema'):
                assert isinstance(value, int) and not isinstance(value, bool), (component, key)
    assert isinstance(identity['catalog_version'], int)
    assert identity == catalog_identity.committed_identity()
    assert catalog.snapshot_identity() is not catalog.snapshot_identity()   # never shared


def test_the_identity_agrees_with_what_the_catalog_records():
    snap = yaml.safe_load((catalog_identity.CATALOG / 'snapshot.yaml').read_bytes())
    tp = yaml.safe_load((catalog_identity.CATALOG / 'tp_registry_snapshot.yaml').read_bytes())
    tmr = yaml.safe_load((catalog_identity.CATALOG / 'tmr_planner_registry.yaml').read_bytes())
    identity = catalog.snapshot_identity()
    assert identity['trainit_perception']['version'] == tp['trainit_perception'] == \
        snap['derived_with']['trainit_perception']
    assert identity['trainit_perception']['vocabulary_schema'] == tp['schema_version']
    assert identity['trainit_motion_runtime']['planner_registry_schema'] == tmr['schema_version']
    registries = snap['registries']
    assert registries['trainit_motion_runtime']['planner_registry']['sha256'] == \
        identity['trainit_motion_runtime']['planner_registry_sha256']
    assert registries['trainit_perception']['vocabulary']['sha256'] == \
        identity['trainit_perception']['vocabulary_sha256']
    assert snap['derived_with']['trainit_setup_assistant'] == T.__version__


def test_the_identity_is_provenance_not_a_dependency(bundles):
    """A bundle that uses no perception gains no trainit_perception dependency from recording the
    vocabulary identity of the TSA that compiled it, and a bundle with no policy step names no
    policy runtime (Pro) at all: the block is the compilation universe, not a list of needs."""
    if 'p2_example' not in bundles:
        pytest.skip('needs fr3wml_digital_twin/fr5_app')
    files = bundles['p2_example:files']
    project = identity_cases.CASES['p2_example']()
    assert not (project.perception and project.perception.detectors)
    for rel, data in files.items():
        if rel.endswith('package.xml'):
            assert b'trainit_perception' not in data, rel
        assert b'trainit_policy_runtime' not in data, rel
    manifest = dict(bundles['p2_example'])
    manifest.pop('capability_snapshot')
    assert 'trainit_perception' not in json.dumps(manifest)


# --- 3. one byte of the snapshot -----------------------------------------------------------------

def _tsa_copy(root: Path) -> Path:
    """A copy of this TSA's Python package, importable from ``root``."""
    shutil.copytree(PKG / 'trainit_setup_assistant', root / 'trainit_setup_assistant',
                    ignore=shutil.ignore_patterns('__pycache__'))
    return root


def _one_more_byte(path: Path) -> None:
    data = path.read_bytes()
    first = data.index(b'\n')
    assert data.startswith(b'#'), path            # a comment line: the meaning is untouched
    path.write_bytes(data[:first] + b' ' + data[first:])


def _record(copy_catalog: Path, name: str) -> None:
    """What the refresh does after a registry changed: snapshot.yaml records its new sha256."""
    snap = copy_catalog / 'snapshot.yaml'
    old = catalog_identity.sha256(name)
    new = catalog_identity.sha256(name, copy_catalog)
    text = snap.read_bytes()
    assert text.count(old.encode()) == 1, name
    snap.write_bytes(text.replace(old.encode(), new.encode()))


def _worker(pkg: Path, case: str, root: Path) -> dict:
    run = Path(tempfile.mkdtemp(dir=root))
    (run / 'tmp').mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(run / 'tmp'),
               PYTHONPATH=os.pathsep.join([str(pkg), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-B', str(WORKER), '--case', case, '--out',
                        str(run / 'out'), '--pkg', str(pkg)],
                       capture_output=True, text=True, cwd=str(run), env=env, timeout=120)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    assert line, r.stdout + r.stderr
    result = json.loads(line[-1][len('RESULT '):])
    shutil.rmtree(run)
    return result


@pytest.fixture(scope='module')
def changed_root():
    root = Path(tempfile.mkdtemp(prefix='tsa_changed_snapshot_'))
    yield root
    shutil.rmtree(root, ignore_errors=True)


def _content_record(manifest: dict) -> dict:
    return {k: v for k, v in manifest.items() if k not in ('generator_version',
                                                            'capability_snapshot')}


@pytest.mark.parametrize('name, changed', [
    ('snapshot.yaml', {('catalog_sha256',)}),
    ('tmr_planner_registry.yaml', {('catalog_sha256',),
                                   ('trainit_motion_runtime', 'planner_registry_sha256')}),
    ('tp_registry_snapshot.yaml', {('catalog_sha256',),
                                   ('trainit_perception', 'vocabulary_sha256')}),
])
def test_one_changed_snapshot_byte_changes_the_identity_and_nothing_else(changed_root, name,
                                                                          changed):
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    copy = _tsa_copy(Path(tempfile.mkdtemp(dir=changed_root)))
    copy_catalog = copy / 'trainit_setup_assistant' / 'catalog'
    _one_more_byte(copy_catalog / name)
    if name != 'snapshot.yaml':
        _record(copy_catalog, name)
    base, other = _worker(PKG, 'p1_golden', changed_root), _worker(copy, 'p1_golden', changed_root)
    assert base['outcome'] == other['outcome'] == 'generated', (other.get('crash'),)
    assert other['tsa_file_real'].startswith(str(copy.resolve()))
    differing = sorted(k for k in set(base['files']) | set(other['files'])
                       if base['files'].get(k) != other['files'].get(k))
    assert differing == [MANIFEST], differing            # only the provenance moved
    assert _content_record(base['manifest']) == _content_record(other['manifest'])
    a, b = base['manifest']['capability_snapshot'], other['manifest']['capability_snapshot']
    moved = {(k,) for k in a if not isinstance(a[k], dict) and a[k] != b[k]}
    moved |= {(k, s) for k in a if isinstance(a[k], dict) for s in a[k] if a[k][s] != b[k][s]}
    assert moved == changed
    assert b == catalog_identity.committed_identity(copy_catalog)
    shutil.rmtree(copy)


@pytest.mark.parametrize('name', ['tmr_planner_registry.yaml', 'tp_registry_snapshot.yaml'])
def test_a_catalog_changed_behind_a_running_tsa_is_refused_before_any_file(monkeypatch,
                                                                            tmp_path, name):
    """The vocabulary was built from bytes the snapshot no longer records (a refresh while TSA
    was running, a hand edit): the identity would be a lie, so nothing is generated."""
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    project = identity_cases.golden()
    monkeypatch.setitem(catalog._PARSED, name, catalog._parsed(name) + b'# changed\n')
    with pytest.raises(RuntimeError) as exc:
        Orchestrator().generate(project, tmp_path / 'out')
    message = str(exc.value)
    assert f'catalog/{name} is not the file catalog/snapshot.yaml records' in message and \
        REFRESH in message and 'start TSA again' in message, message
    assert not (tmp_path / 'out').exists()


def test_a_refresh_inside_a_running_tsa_is_refused_not_recorded(monkeypatch):
    """A refresh in this process rewrites TP's snapshot and snapshot.yaml, and its
    ``_reread_catalog`` clears the readers' caches: a lazy reader (the faces) re-reads the new
    vocabulary while the model keeps the one it was built with at import. The identity stays the
    FIRST bytes' (write-once, surviving ``_reread_catalog``), so the bundle is refused instead of
    recording an identity no part of the model has."""
    monkeypatch.setattr(catalog, '_PARSED', dict(catalog._PARSED))   # this test's own copy
    first = catalog._parsed(catalog.TP_REGISTRY_FILE)
    new_tp = first + b'# refreshed\n'
    old_sha, new_sha = (hashlib.sha256(x).hexdigest() for x in (first, new_tp))
    real_snapshot = (catalog_identity.CATALOG / 'snapshot.yaml').read_bytes()
    assert real_snapshot.count(old_sha.encode()) == 1
    new_snapshot = real_snapshot.replace(old_sha.encode(), new_sha.encode())
    catalog._PARSED[catalog.SNAPSHOT_FILE] = real_snapshot          # TSA had generated before
    monkeypatch.setattr(catalog, '_committed', lambda name, _real=catalog._committed: (
        new_tp if name == catalog.TP_REGISTRY_FILE else
        new_snapshot if name == catalog.SNAPSHOT_FILE else _real(name)))
    try:
        refresh_tool._reread_catalog()             # what the refresh does after its write
        catalog.tp_registry()                      # the re-read
        assert catalog._PARSED[catalog.TP_REGISTRY_FILE] == first
        with pytest.raises(RuntimeError, match=r'snapshot\.yaml changed on disk while this TSA '
                                               r'was running'):
            catalog.snapshot_identity()
        del catalog._PARSED[catalog.SNAPSHOT_FILE]  # had TSA not generated before the refresh,
        with pytest.raises(RuntimeError, match=r'is not the file catalog/snapshot\.yaml '
                                               r'records'):  # the registry's hash refuses it
            catalog.snapshot_identity()
    finally:
        monkeypatch.undo()
        refresh_tool._reread_catalog()             # the next reader reads the real files
    assert catalog.snapshot_identity() == catalog_identity.committed_identity()


def test_a_refused_snapshot_is_not_pinned_so_its_repair_needs_no_restart(monkeypatch):
    """The refusal says 'restore it with git': after the restore, the next generation works."""
    monkeypatch.setattr(catalog, '_PARSED', dict(catalog._PARSED))
    catalog._PARSED.pop(catalog.SNAPSHOT_FILE, None)                # not read yet
    text = (catalog_identity.CATALOG / 'snapshot.yaml').read_text()
    edit = f'      sha256: {_TP_SHA}\n      schema_version: 1'
    edited = text.replace(edit, edit[:-1] + '7').encode()
    real = catalog._committed
    monkeypatch.setattr(catalog, '_committed', lambda name: (
        edited if name == catalog.SNAPSHOT_FILE else real(name)))
    with pytest.raises(RuntimeError, match='the vocabulary schema 7'):
        catalog.snapshot_identity()
    monkeypatch.setattr(catalog, '_committed', real)               # restored with git
    assert catalog.snapshot_identity() == catalog_identity.committed_identity()


def test_a_snapshot_changed_on_disk_while_tsa_runs_is_refused(monkeypatch):
    catalog.snapshot_identity()                    # this process has read it: it is pinned
    pinned = catalog._PARSED[catalog.SNAPSHOT_FILE]
    monkeypatch.setattr(catalog, '_committed', lambda name, _real=catalog._committed: (
        pinned + b'# next release\n' if name == catalog.SNAPSHOT_FILE else _real(name)))
    with pytest.raises(RuntimeError, match=r'snapshot\.yaml changed on disk while this TSA was '
                                           r'running: start TSA again'):
        catalog.snapshot_identity()


def test_a_snapshot_another_release_derived_is_refused(monkeypatch, tmp_path):
    """TSA updated while running, first generation after the update: snapshot.yaml is pinned at
    its first identity, so only the release refuses it -- in both producers, before any file."""
    with pytest.raises(RuntimeError, match=r'was derived by TSA .*, and this is TSA 0\.0\.1'):
        catalog.snapshot_identity(release='0.0.1')
    assert catalog.snapshot_identity(release=T.__version__) == \
        catalog_identity.committed_identity()
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    monkeypatch.setattr(orchestrator, '__version__', '0.0.1')     # the code that runs
    with pytest.raises(RuntimeError, match='was derived by TSA'):
        Orchestrator().generate(identity_cases.golden(), tmp_path / 'bundle')
    with pytest.raises(RuntimeError, match='was derived by TSA'):
        generate_scene_loader_config(identity_cases.golden(), tmp_path / 'step3', 'x_config')
    assert not (tmp_path / 'bundle').exists() and not (tmp_path / 'step3').exists()


_TP_SHA = catalog_identity.sha256('tp_registry_snapshot.yaml')
_TMR_SHA = catalog_identity.sha256('tmr_planner_registry.yaml')
_TP_RELEASE = catalog_identity.committed_identity()['trainit_perception']['version']
_TMR_RELEASE = catalog_identity.committed_identity()['trainit_motion_runtime']['version']


@pytest.mark.parametrize('edit, says', [
    ((f'      sha256: {_TP_SHA}\n      schema_version: 1',
      f'      sha256: {_TP_SHA}\n      schema_version: 7'), 'the vocabulary schema 7'),
    ((f'  trainit_perception:\n    version: {_TP_RELEASE}\n    vocabulary:',
      '  trainit_perception:\n    version: 9.9.9\n    vocabulary:'),
     "the trainit_perception release '9.9.9'"),
    ((f'      file: tmr_planner_registry.yaml\n      sha256: {_TMR_SHA}\n      schema_version: 1',
      f'      file: tmr_planner_registry.yaml\n      sha256: {_TMR_SHA}\n      schema_version: 2'),
     'the planner registry schema 2'),
    ((f'  trainit_motion_runtime: {_TMR_RELEASE}\n  trainit_perception:',
      f'  trainit_motion_runtime: "{_TMR_RELEASE}\\n"\n  trainit_perception:'),
     'records no usable versions or schemas'),
], ids=['vocabulary_schema', 'tp_release', 'planner_schema', 'version_with_a_newline'])
def test_labels_the_snapshot_records_must_be_the_parsed_files_own(monkeypatch, edit, says):
    """A hand edit or a bad merge of snapshot.yaml that leaves the hashes alone: the labels it
    records must be the ones the files it names carry."""
    text = (catalog_identity.CATALOG / 'snapshot.yaml').read_text()
    assert text.count(edit[0]) == 1, 'the snapshot.yaml layout moved: update this edit'
    edited = text.replace(edit[0], edit[1]).encode()
    monkeypatch.setattr(catalog, '_committed', lambda name, _real=catalog._committed: (
        edited if name == catalog.SNAPSHOT_FILE else _real(name)))
    monkeypatch.setitem(catalog._PARSED, catalog.SNAPSHOT_FILE, edited)   # as if read at start
    with pytest.raises(RuntimeError, match=re.escape(says)):
        catalog.snapshot_identity()


def test_a_crlf_checkout_is_named_in_the_refusal(monkeypatch):
    crlf = catalog._parsed(catalog.REGISTRY_FILE).replace(b'\n', b'\r\n')
    monkeypatch.setitem(catalog._PARSED, catalog.REGISTRY_FILE, crlf)
    with pytest.raises(RuntimeError, match='CRLF line endings: the checkout converted them'):
        catalog.snapshot_identity()


def test_an_unreadable_snapshot_is_refused_with_the_repair(monkeypatch):
    monkeypatch.setattr(catalog, '_committed', lambda name: b'<<<<<<< ours\n')
    with pytest.raises(RuntimeError, match=r'catalog/snapshot\.yaml cannot be read as the '
                                           r'catalog snapshot .*restore it with git'):
        catalog.snapshot_identity()


# --- 4. the source rule -------------------------------------------------------------------------

def _package(root: Path, name: str, manifest: str = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / 'package.xml').write_text(manifest if manifest is not None else
                                      f'<?xml version="1.0"?>\n<package format="3">'
                                      f'<name>{name}</name><version>1.0.0</version></package>\n')
    return root


def _file(path: Path, data: bytes = b'x') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.fixture
def no_package_above(tmp_path):
    for parent in tmp_path.parents:
        if (parent / 'package.xml').is_file():
            pytest.skip(f'{parent} holds a package.xml: tmp_path is inside a package here')
    return tmp_path


def test_a_file_is_named_by_the_nearest_package_and_its_path_inside_it(no_package_above):
    pkg = _package(no_package_above / 'ws' / 'src' / 'my_cell_description', 'my_cell_description')
    assert portable_source(_file(pkg / 'urdf' / 'robot.urdf.xacro')) == \
        'package://my_cell_description/urdf/robot.urdf.xacro'
    assert portable_source(_file(pkg / 'meshes' / 'visual' / 'link 1.dae')) == \
        'package://my_cell_description/meshes/visual/link 1.dae'
    assert portable_source(_file(pkg / 'CMakeLists.txt')) == \
        'package://my_cell_description/CMakeLists.txt'
    assert portable_source(pkg / 'package.xml') == 'package://my_cell_description/package.xml'


def test_the_name_is_the_package_xmls_never_the_directorys(no_package_above):
    pkg = _package(no_package_above / 'fairino_bridge_master_class', '  fairino_bridge \n')
    assert portable_source(_file(pkg / 'config' / 'a.yaml')) == \
        'package://fairino_bridge/config/a.yaml'


def test_the_innermost_package_holds_the_file(no_package_above):
    outer = _package(no_package_above / 'outer', 'outer_pkg')
    inner = _package(outer / 'vendor', 'inner_pkg')
    assert portable_source(_file(inner / 'meshes' / 'a.stl')) == 'package://inner_pkg/meshes/a.stl'
    assert portable_source(_file(outer / 'meshes' / 'b.stl')) == 'package://outer_pkg/meshes/b.stl'


def test_a_file_in_no_package_is_named_by_its_file_name_alone(no_package_above):
    card = _file(no_package_above / 'lab' / 'policies' / 'pick' / 'policy_card.yaml')
    (no_package_above / 'lab' / 'COLCON_IGNORE').write_text('')
    assert portable_source(card) == 'external:policy_card.yaml'


def test_a_symlink_is_named_as_the_project_names_it_never_by_its_target(no_package_above):
    """The bundle ships a linked file under its own name; the manifest names it the same way. A
    target the project never names (another package, a private store) stays out of it."""
    pkg_a = _package(no_package_above / 'pkg_a', 'pkg_a')
    pkg_b = _package(no_package_above / 'pkg_b', 'pkg_b')
    target = _file(pkg_b / 'config' / 'c.yaml')
    (pkg_a / 'config').mkdir()
    (pkg_a / 'config' / 'link.yaml').symlink_to(target)
    assert portable_source(pkg_a / 'config' / 'link.yaml') == 'package://pkg_a/config/link.yaml'
    secret = _file(no_package_above / 'store' / 'part_4711_jaw.stl')
    (pkg_a / 'meshes').mkdir()
    (pkg_a / 'meshes' / 'jaw.stl').symlink_to(secret)
    assert portable_source(pkg_a / 'meshes' / 'jaw.stl') == 'package://pkg_a/meshes/jaw.stl'
    leaf = no_package_above / 'outside' / 'urdf_link'          # a urdf_dir given as a link
    leaf.parent.mkdir()
    leaf.symlink_to(pkg_b / 'config', target_is_directory=True)
    assert portable_source(leaf / 'c.yaml') == 'external:c.yaml'
    workspace = no_package_above / 'ws_link'                   # the workspace reached by a link
    workspace.symlink_to(no_package_above, target_is_directory=True)
    assert portable_source(workspace / 'pkg_b' / 'config' / 'c.yaml') == \
        'package://pkg_b/config/c.yaml'
    # a '..' right after a link climbs from the link's TARGET, as the copy reads it
    _file(pkg_b / 'meshes' / 'real.stl')
    (pkg_a / 'sub_link').symlink_to(pkg_b / 'config', target_is_directory=True)
    assert portable_source(f'{pkg_a}/sub_link/../meshes/real.stl') == \
        'package://pkg_b/meshes/real.stl'


def test_a_directory_is_named_with_its_own_package_xml(no_package_above):
    pkg = _package(no_package_above / 'my_cell_description', 'my_cell_description')
    (pkg / 'urdf').mkdir()
    assert portable_source(pkg, directory=True) == 'package://my_cell_description/'
    assert portable_source(pkg / 'urdf', directory=True) == 'package://my_cell_description/urdf'
    assert portable_source(pkg / 'missing' / 'meshes', directory=True) == \
        'package://my_cell_description/missing/meshes'
    assert portable_source(no_package_above / 'nowhere', directory=True) == 'external:nowhere'


def test_a_relative_path_is_read_where_the_copy_reads_it(no_package_above, monkeypatch):
    pkg = _package(no_package_above / 'pkg', 'pkg')
    _file(pkg / 'config' / 'x.yaml')
    monkeypatch.chdir(pkg)
    assert portable_source('config/x.yaml') == 'package://pkg/config/x.yaml'
    assert portable_source('config/../config/./x.yaml') == 'package://pkg/config/x.yaml'


@pytest.mark.parametrize('manifest', [
    '<package><name>broken',                                        # not XML
    '',                                                             # empty
    '<manifest><name>catkin_era</name></manifest>',                 # not a package manifest
    '<package format="3"><name>a</name><name>b</name></package>',   # two names
    '<package format="3"><description/></package>',                # no name
    '<package format="3"><name>my pkg/x</name></package>',          # not a package name
], ids=['not_xml', 'empty', 'wrong_root', 'two_names', 'no_name', 'bad_name'])
def test_an_unusable_package_xml_gives_a_file_name_and_one_warning(no_package_above, manifest):
    pkg = _package(no_package_above / 'odd_pkg', '', manifest=manifest)
    names = SourceNamer()
    assert names(_file(pkg / 'a.stl')) == 'external:a.stl'
    assert names(_file(pkg / 'sub' / 'b.stl')) == 'external:b.stl'
    assert len(names.problems) == 1, names.problems               # one per package.xml
    assert '"odd_pkg"' in names.problems[0] and str(no_package_above) not in names.problems[0]
    above = _package(no_package_above / 'outer', 'outer_pkg')     # never walks past a package.xml
    shutil.move(str(pkg), str(above / 'odd_pkg'))
    assert SourceNamer()(above / 'odd_pkg' / 'a.stl') == 'external:a.stl'


def test_what_a_generation_reads_it_reads_again_in_the_next_one(no_package_above):
    pkg = _package(no_package_above / 'pkg', 'first_name')
    f = _file(pkg / 'a.yaml')
    assert SourceNamer()(f) == 'package://first_name/a.yaml'
    _package(pkg, 'second_name')
    assert SourceNamer()(f) == 'package://second_name/a.yaml'


def test_tsa_s_own_resources_are_named_the_same_however_tsa_is_installed(no_package_above):
    assert portable_source(KIT) == KIT_SOURCE
    link = no_package_above / 'linked_tsa'
    link.symlink_to(PKG / 'trainit_setup_assistant', target_is_directory=True)
    assert portable_source(link / 'resources' / 'gazebo_cell_kit' / 'scripts' /
                           'link_attacher_bridge.py') == KIT_SOURCE
    # an installed copy: site-packages has no package.xml above the package
    site = _tsa_copy(no_package_above / 'site-packages')
    code = ('from trainit_setup_assistant.generator.provenance import portable_source\n'
            'import trainit_setup_assistant, os\n'
            'kit = os.path.join(os.path.dirname(trainit_setup_assistant.__file__), "resources",'
            ' "gazebo_cell_kit", "scripts", "link_attacher_bridge.py")\n'
            'print(trainit_setup_assistant.__file__)\nprint(portable_source(kit))\n')
    r = subprocess.run([sys.executable, '-B', '-c', code], capture_output=True, text=True,
                       cwd=str(no_package_above),
                       env=dict(os.environ, PYTHONPATH=str(site), PYTHONDONTWRITEBYTECODE='1'))
    assert r.returncode == 0, r.stderr
    imported, named = r.stdout.splitlines()[-2:]
    assert imported.startswith(str(site)) and named == KIT_SOURCE


def test_the_goldens_copies_are_named_by_their_package(bundles):
    if 'p1_golden' not in bundles:
        pytest.skip('needs the golden sibling checkout')
    copies = [f['source'] for f in bundles['p1_golden']['files'] if f['action'] == 'COPY']
    assert copies == [f'package://fr3wml_suction_camera_moveit_config/config/{name}'
                      for name in GOLDEN_COPIES]
    if 'p2_example' not in bundles:
        pytest.skip('needs fr3wml_digital_twin/fr5_app for the example half')
    sources = [f['source'] for f in bundles['p2_example']['files'] if f['action'] == 'COPY']
    assert len(sources) == 31 and all(s.startswith(('package://fr5_app/urdf/',
                                                    'package://fr5_app/meshes/'))
                                      for s in sources), sources


def _policy_sources(manifest: dict) -> dict:
    return {f['path'].rsplit('/', 1)[-1]: f['source'] for f in manifest['files']
            if f['action'] == 'COPY' and '/policies/' in f['path']}


def test_a_policy_card_and_its_models_are_named_by_their_file_names(bundles):
    """A card trained outside any package (trainit_lab's case): its file names alone, never the
    directory it was trained in -- built here, so a clean checkout checks it too."""
    if 'policy_synthetic' not in bundles:
        pytest.skip('needs the golden sibling checkout')
    if not bundles['no_package_above_root']:
        pytest.skip('the temporary directory sits inside a ROS package here')
    assert _policy_sources(bundles['policy_synthetic']) == {
        'policy_card.yaml': 'external:policy_card.yaml', 'policy.onnx': 'external:policy.onnx',
        'policy.pt': 'external:policy.pt'}


def test_the_owners_policy_project_names_its_card_files_by_file_name(bundles):
    if 'pol1_policy_test4_project' not in bundles:
        pytest.skip(identity_cases.missing('pol1_policy_test4_project'))
    assert _policy_sources(bundles['pol1_policy_test4_project']) == {
        'policy_card.yaml': 'external:policy_card.yaml', 'policy.onnx': 'external:policy.onnx',
        'policy.pt': 'external:policy.pt'}


def test_a_generated_gazebo_cell_names_tsa_s_kit_by_its_tsa_path(bundles):
    kit = [f['source'] for f in bundles['gazebo_kit']['files']
           if f['path'].endswith('scripts/link_attacher_bridge.py')]
    assert kit == [KIT_SOURCE]


def test_an_unusable_package_xml_above_the_inputs_is_one_manifest_warning(tmp_path):
    """The copied files are named by their file names, and the manifest says why -- once, naming
    the directory, never its path."""
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    if not _no_package_above(tmp_path):
        pytest.skip('tmp_path is inside a ROS package here')
    base = tmp_path / 'catkin_era_pkg'
    shutil.copytree(ws_paths.ws_src('fr3wml_digital_twin', 'fr3wml_suction_camera_moveit_config'),
                    base)
    (base / 'package.xml').write_text('<package><name>broken')
    project = identity_cases.golden()
    project.robot.description.urdf_dir = str(base / 'config')
    project.robot.base_moveit_config_path = str(base)
    manifest = _generate(project, tmp_path / 'out')
    copies = [f['source'] for f in manifest['files'] if f['action'] == 'COPY']
    assert copies and all(s.startswith('external:') for s in copies), copies
    problems = [w for w in manifest['warnings'] if 'not a valid ROS package manifest' in w]
    assert problems == ['copied files are recorded as external:<file name>: the package.xml in a '
                        'directory named "catkin_era_pkg" is not a valid ROS package manifest']


# --- 5. what a manifest must not say ---------------------------------------------------------------

def _machine_roots(*extra):
    roots = {str(PKG), str(PKG.resolve()), str(ws_paths.WS_SRC), str(Path.home()),
             tempfile.gettempdir(), os.path.realpath(tempfile.gettempdir()), *map(str, extra)}
    return sorted(r for r in roots if r and r != '/')


def _candidate_tokens():
    """Every candidate provider's id, package and product name, and the product words that are
    not ordinary words (ADR-0019: candidates stay external metadata, never in a bundle)."""
    generic = {'ess', 'segment', 'nvidia', 'isaac', 'ros', 'anything', 'core', 'deploy', 'for',
               'lab', 'manipulation', 'sim', 'stereo'}
    exact, words = set(), set()
    for c in catalog.candidates():
        product = c['id'].split('.', 1)[-1].lower()
        exact |= {c['id'].lower(), product} | {p.lower() for p in c.get('packages', [])}
        words |= {w for w in product.split('_') if w not in generic and len(w) > 2}
    assert {'cumotion', 'curobo', 'foundationpose'} <= words, 'the deny-list lost NVIDIA words'
    return exact, words


def _leaks(text: str, roots) -> list:
    out = [root for root in roots if root in text]
    out += [m.group(0) for m in re.finditer(r'(?<![\w.:/-])/(home|root|tmp|opt|mnt|media|srv|'
                                            r'var|usr)/\S*', text)]
    exact, words = _candidate_tokens()
    tokens = set(re.split(r'[^a-z0-9_]+', text.lower()))
    out += sorted(tokens & exact)
    out += sorted(t for t in tokens if set(t.split('_')) & words)
    return out


def test_no_manifest_names_this_machine_or_a_candidate(bundles):
    roots = _machine_roots(bundles['root'])
    for name in _manifests(bundles) + ['gazebo_kit']:
        text = bundles[name + ':text']
        assert not _leaks(text, roots), (name, _leaks(text, roots))
        for record in bundles[name]['files']:
            source = record.get('source', '')
            if record['action'] == 'COPY':
                assert PORTABLE.match(source), (name, record)
            else:
                assert not source.startswith(('/', '~')) and ':\\' not in source, (name, record)


def test_the_warnings_name_no_machine_path_and_the_user_still_sees_it(no_package_above):
    """Every generator warning that mentions a file or directory -- a missing meshes_dir, an empty
    urdf_dir, a missing base config, a missing policy card, a card listing a model that is not
    there (absolute or climbing out of its directory) -- names it, in the MANIFEST, the way a COPY
    is named; the user who generates is shown the paths as given (CLI and GUI read
    ``local_warnings``), as before M2.3."""
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    tmp = no_package_above
    empty = tmp / 'inputs' / 'empty_urdf'
    empty.mkdir(parents=True)
    a = identity_cases.golden()
    a.robot.description.urdf_dir = str(empty)
    a.robot.description.meshes_dir = str(tmp / 'inputs' / 'nowhere' / 'meshes')
    a.robot.base_moveit_config_path = str(tmp / 'inputs' / 'no_base')
    ma = Orchestrator().generate(a, tmp / 'a')
    card = textwrap.dedent(f"""
        policy_card:
          name: lost_models
          files: {{jit: ../../trained_on_another_machine/runs/r7/policy.pt,
                  onnx: {tmp}/models_of_another_machine/policy.onnx}}
    """)
    b = _policy_project(tmp / 'cards' / 'lost', card=card, models=())
    mb = Orchestrator().generate(b, tmp / 'b')
    c = _policy_project(tmp / 'cards' / 'gone')
    c.application.policies[0].card = str(tmp / 'inputs' / 'gone' / 'policy_card.yaml')
    mc = Orchestrator().generate(c, tmp / 'c')
    recorded = '\n'.join(ma.warnings + mb.warnings + mc.warnings)
    for expected in ('meshes_dir not found, skipped: external:meshes',
                     'no urdf files copied from external:empty_urdf',
                     'scene_loader emitter: base config dir not found: external:config',
                     'card lists "external:policy.pt" but it is not beside the card',
                     'card lists "external:policy.onnx" but it is not beside the card',
                     'card not found at "external:policy_card.yaml"'):
        assert expected in recorded, (expected, recorded)
    assert not _leaks(recorded, _machine_roots(tmp)), _leaks(recorded, _machine_roots(tmp))
    assert ('trained_on_another_machine' not in recorded
            and 'models_of_another_machine' not in recorded)
    shown = '\n'.join(ma.local_warnings + mb.local_warnings + mc.local_warnings)
    for path in (str(empty), str(tmp / 'inputs' / 'nowhere' / 'meshes'),
                 str(tmp / 'inputs' / 'no_base' / 'config'),
                 '../../trained_on_another_machine/runs/r7/policy.pt',
                 str(tmp / 'inputs' / 'gone' / 'policy_card.yaml')):
        assert path in shown, (path, shown)
    for out, manifest in ((tmp / 'a', ma), (tmp / 'b', mb), (tmp / 'c', mc)):
        written = yaml.safe_load((out / MANIFEST).read_bytes())      # what the bundle CARRIES
        assert written['warnings'] == manifest.warnings
        assert not _leaks('\n'.join(written['warnings']), _machine_roots(tmp))


def test_the_cli_shows_the_user_the_paths_the_manifest_does_not_name(no_package_above):
    """trainit_generate prints the warnings with this machine's paths (local_warnings); the
    bundle's manifest names the same inputs portably."""
    if identity_cases.missing('p2_example'):
        pytest.skip(identity_cases.missing('p2_example'))
    from trainit_setup_assistant.model.io import save_project
    project = identity_cases.CASES['p2_example']()
    missing = no_package_above / 'typo' / 'meshes'
    project.robot.description.meshes_dir = str(missing)
    save_project(project, no_package_above / 'project.yaml')
    r = subprocess.run([sys.executable, '-B', '-m', 'trainit_setup_assistant.cli.generate',
                        str(no_package_above / 'project.yaml'), '-o', str(no_package_above / 'out')],
                       capture_output=True, text=True, cwd=str(PKG), timeout=300,
                       env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
                           [str(PKG), os.environ.get('PYTHONPATH', '')])))
    assert r.returncode == 0, r.stderr
    assert f'meshes_dir not found, skipped: {missing}' in r.stderr, r.stderr
    written = (no_package_above / 'out' / MANIFEST).read_text()
    assert 'meshes_dir not found, skipped: external:meshes' in written
    assert str(missing) not in written


def test_the_gui_shows_the_user_the_local_warnings():
    """The Step-3 page prints what the user who generated it can act on (static: the page is
    driven offscreen elsewhere, without a missing input)."""
    source = (PKG / 'trainit_setup_assistant' / 'gui' / 'wizard.py').read_text()
    assert 'for w in manifest.local_warnings:' in source
    assert "as_dict().get('warnings'" not in source


def test_an_empty_or_dot_input_path_is_written_as_given_not_as_the_cwd(tmp_path, monkeypatch):
    """'' or '.' (the GUI's default urdf_dir) would otherwise name the cwd -- a login name."""
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    cwd = tmp_path / 'jdoe'
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    project = identity_cases.golden()
    project.robot.description.urdf_dir = ''
    manifest = Orchestrator().generate(project, tmp_path / 'out')
    text = '\n'.join(manifest.warnings)
    assert 'no urdf files copied from .' in text and 'jdoe' not in text, text


def test_a_namespaced_cell_does_not_quote_a_bridge_s_file_parameter_as_a_topic(tmp_path):
    """The namespace hand-off warning names a bridge's absolute ROOT topics; a parameter that is a
    file path (not a ROS name: it has a '.') is not one -- quoting it would also put the path in
    the manifest."""
    if identity_cases.missing('p1_golden'):
        pytest.skip(identity_cases.missing('p1_golden'))
    project = identity_cases.golden()
    project.robot.cell_namespace = '/r1'
    merger = next(b for b in project.deployment.bridges if b.executable == 'joint_state_merger')
    paths = ['/home/someone/cal/calibration.yaml',      # a file (a '.')
             '/home/someone/models/run_42',             # a directory: a valid ROS name's shape
             '/dev/ttyUSB0', '/srv/cell/calibration']   # a device, another filesystem root
    merger.parameters.append({f'p{i}': v for i, v in enumerate(paths)})
    manifest = Orchestrator().generate(project, tmp_path / 'out')
    handoff = [w for w in manifest.warnings if 'joint_state_merger runs IN the namespace' in w]
    assert handoff and '/joint_states_robot' in handoff[0], manifest.warnings
    assert not [v for v in paths if v in '\n'.join(manifest.warnings)], manifest.warnings
