"""What a CANONICAL bundle carries (vNext M2.6, ADR-0022 D6 "Emission"): F3 emission, F5 pose
relevance, F7 the composite boundary tripwire. Generated into temp dirs only; nothing here writes
a committed bundle.

F3  the golden-shaped suction opted in (the 3-line contract, the two legacy bridge rows removed):
    every generated file is the legacy golden's byte for byte EXCEPT the classified ones --
    [C] bt_params (gripper_action "", the eef block), the tree (eef ports, no SetReleasePolicy),
    scene.yaml (gripper_cmd_topic eef/main/engaged); [A] BRIDGES (the adapter rows appended) and
    no generated mock server; [D] package.xml (trainit_hardware_layer >= 0.2.0, TMR >= 2.0.0);
    [P] the manifest. The motion nodes, the planner config, URDF/SRDF and perception.yaml are
    untouched. The composite trees carry ids only; two robots with identical ids resolve to
    DISJOINT names in /r1 and /r2 (and the root, and /r3).
F5  moved_by(site): the axes that move a site's TCP, from the description; a jaw moves none.
F7  a composite (mock) bundle writes no tool joint in ros2_control, no tool controller, no SRDF
    tool group and no named tool state (vNext M2.7: R22 narrowed, no bypass any more).
"""
import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

import eef_fixture as F
import ws_paths
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator import eef_contract as E
from trainit_setup_assistant.generator.orchestrator import generate_scene_loader_config
from trainit_setup_assistant.model import (BundleSpec, GripperSpec, PolicyStep, ToolAction,
                                           load_project)
from trainit_setup_assistant.model.enums import Backend, ToolActionKind

GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
# skips without the demo repository; FAILS with TRAINIT_REQUIRE_GOLDEN=1 (eef_fixture.needs)
needs_golden = F.needs(GOLDEN_PROJECT, reason='the golden project is not present')
EEF_TAGS = ('CloseGripper', 'OpenGripper', 'VerifyGrasp', 'CommandActuator')


def _files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob('*')) if p.is_file()}


def _bridges(launch_text: str) -> list:
    for node in ast.walk(ast.parse(launch_text)):
        if isinstance(node, ast.Assign) and getattr(node.targets[0], 'id', '') == 'BRIDGES':
            return ast.literal_eval(node.value)
    raise AssertionError('no BRIDGES')


def _golden(prefix='cell'):
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix(prefix)
    return p


# --- F3: the golden-shaped suction, opted in ----------------------------------------------------

APP, CFG = 'cell_app', 'cell_trainit_config'
CLASSIFIED = {   # the only files a canonical candidate may differ by, and why (ADR-0022)
    f'{APP}/config/bt_params.yaml': 'C',
    f'{APP}/bt_trees/pick_place.xml': 'C',
    f'{CFG}/config/scene.yaml': 'C',
    f'{CFG}/launch/bringup.launch.py': 'A',
    f'{APP}/scripts/mock_gripper_action_server.py': 'A',     # not emitted
    f'{APP}/CMakeLists.txt': 'A',                            # ... nor installed
    f'{CFG}/package.xml': 'D',
    f'{APP}/package.xml': 'D',
    'generation_manifest.yaml': 'P',
}


_PORTLESS = re.compile(r'^\s*<(CloseGripper|OpenGripper)/>$')
_RELEASE_POLICY = re.compile(r'^\s*<SetReleasePolicy policy="[a-z_]+"/>$')
_TMR_FLOOR = re.compile(r'^  (<!-- trainit_motion_runtime >= (?P<c>\d+\.\d+\.\d+) for: '
                        r'(?P<why>.*) -->|<exec_depend version_gte="(?P<d>\d+\.\d+\.\d+)">'
                        r'trainit_motion_runtime</exec_depend>)$')
_EEF_FLOOR_REASONS = ('VerifyGrasp (the end-effector contract)',
                      'CommandActuator (the end-effector contract)',
                      "DetectObject's fault output", 'the eef parameter block (ADR-0022)')
_HW_DEPEND = re.compile(r'^  (<!-- trainit_hardware_layer >= 0\.2\.0 for: the end-effector '
                        r'adapter\(s\) [a-z_]+(, [a-z_]+)* \(ADR-0022\) -->|<exec_depend '
                        r'version_gte="0\.2\.0">trainit_hardware_layer</exec_depend>)$')
_PLAIN_DEPEND = re.compile(r'^  <exec_depend>(?P<p>[A-Za-z0-9_]+)</exec_depend>$')
_MOCK_SERVER = f'{APP}/scripts/mock_gripper_action_server.py'
_MOCK_INSTALL = ('# mock gripper action server (no-op, used in mock mode)',
                 'install(PROGRAMS scripts/mock_gripper_action_server.py',
                 '  DESTINATION lib/${PROJECT_NAME})')
_EEF_BLOCK_COMMENT = (
    '    # The end-effector contract (ADR-0022): every actuator, grasp site and the verification',
    '    # policy, resolved. trainit_run_bt and the device\'s adapter read this same block; the',
    '    # actuators are commanded on their own standard interfaces at RELATIVE names.')


def _changed_lines(a: bytes, b: bytes) -> list:
    """Every changed line of one file, as (side, index, text): difflib without autojunk."""
    from difflib import SequenceMatcher
    old = a.decode('utf-8').splitlines() if a is not None else []
    new = b.decode('utf-8').splitlines() if b is not None else []
    out = []
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, old, new, autojunk=False).get_opcodes():
        if tag != 'equal':
            out += [('-', i, old[i]) for i in range(i1, i2)]
            out += [('+', j, new[j]) for j in range(j1, j2)]
    return out


def _manifest_entry(lines: list, index: int):
    """The ``files:`` entry (its path) line ``index`` of a manifest belongs to, or None."""
    for line in reversed(lines[:index + 1]):
        if line.startswith('- path: '):
            return line[len('- path: '):]
        if not line.startswith(('  ', '- ')):
            return None
    return None


class _Opted:
    """What the line classes may know: both trees, the canonical project, the removed rows."""

    def __init__(self, old: dict, new: dict, canonical, legacy):
        import hashlib
        self.old, self.new, self.project = old, new, canonical
        self.sha = hashlib.sha256
        dev = E.device(canonical)
        self.dev = dev
        self.eef_lines = list(_EEF_BLOCK_COMMENT) + E.render_parameter_block(
            E.eef_parameter_block(dev), indent=4)
        kept = {(b.package, b.executable) for b in canonical.deployment.bridges}
        self.removed = [b for b in legacy.deployment.bridges
                        if (b.package, b.executable) not in kept]
        self.remaining_packages = {b.package for b in canonical.deployment.bridges}
        self.legacy_scene_topic = legacy.scene.gripper_cmd_topic
        self.reset = E.scene_reset_topic(canonical)

    def text(self, side: str, rel: str) -> list:
        data = (self.old if side == '-' else self.new).get(rel)
        return data.decode('utf-8').splitlines() if data is not None else []

    def references(self, rel: str, package: str) -> bool:
        """A file of ``rel``'s package in the canonical tree uses ``package``."""
        top = rel.split('/', 1)[0] + '/'
        return any(f'package://{package}/'.encode() in data or f'$(find {package})'.encode() in data
                   for path, data in self.new.items() if path.startswith(top))


def _line_class(rel: str, side: str, index: int, text: str, ctx: _Opted) -> str:
    """The class ([C] [A] [D] [P]) of ONE changed line, by an explicit per-file pattern; '' when
    none matches (ADR-0022 "Generated changes, classified": anything else fails the gate)."""
    dev = re.escape(ctx.dev.id)
    if rel.endswith('config/bt_params.yaml'):
        if side == '-' and re.match(r'^    (gripper_action|gripper_cmd_topic|'
                                    r'gripper_close_position|gripper_open_position): ', text):
            return 'C'                      # the legacy gripper keys go
        if side == '+' and (text == '    gripper_action: ""' or text in ctx.eef_lines):
            return 'C'                      # gripper_action "" and the eef block (D6)
    elif rel.endswith('bt_trees/pick_place.xml'):
        if side == '-' and (_PORTLESS.match(text) or _RELEASE_POLICY.match(text)):
            return 'C'                      # the port-less legacy nodes; no SetReleasePolicy
        if side == '+' and re.match(rf'^\s*<(CloseGripper|OpenGripper|VerifyGrasp|CommandActuator) '
                                    rf'eef="{dev}"( [a-z_]+="[^"<>]*")*/>$', text):
            return 'C'                      # the contract's nodes, logical ids only
    elif rel.endswith('config/scene.yaml'):
        signal = (f'gripper_cmd_topic: "eef/{ctx.dev.id}/engaged"' if side == '+' else
                  f'gripper_cmd_topic: "{ctx.legacy_scene_topic}"')
        if text.strip() == signal:
            return 'C'                      # the scene signal eef/<id>/engaged
        if text.strip().startswith('scene_reset_topic: ') and (
                side == '-' or text.strip() == f'scene_reset_topic: "{ctx.reset}"'):
            return 'C'                      # the per-robot reset topic
    elif rel.endswith('launch/bringup.launch.py'):
        if text.startswith('BRIDGES = '):
            return 'A'                      # the adapter rows appended (checked below)
    elif rel == _MOCK_SERVER:
        return 'A' if side == '-' and rel not in ctx.new else ''
    elif rel == f'{APP}/CMakeLists.txt':
        return 'A' if side == '-' and text in _MOCK_INSTALL else ''
    elif rel.endswith('package.xml'):
        floor = _TMR_FLOOR.match(text)
        if _HW_DEPEND.match(text) and side == '+':
            return 'D'                      # trainit_hardware_layer: a launch starts an adapter
        if floor and side == '-':
            return 'D'                      # the old TMR floor goes ...
        if floor and (floor.group('d') == '2.0.0' or (
                floor.group('c') == '2.0.0' and all(
                    why in _EEF_FLOOR_REASONS for why in floor.group('why').split('; ')))):
            return 'D'                      # ... for the 2.0.0 floor of an ADR-0022 capability
        plain = _PLAIN_DEPEND.match(text)
        if plain and side == '-':
            package = plain.group('p')
            if (package in {b.package for b in ctx.removed}
                    and package not in ctx.remaining_packages
                    and not ctx.references(rel, package)):
                return 'D'                  # a removed legacy bridge's package, used nowhere here
    elif rel == 'generation_manifest.yaml':
        lines = ctx.text(side, rel)
        if re.match(r'^(generator_version: |  catalog_sha256: |    version: )', text):
            return 'P'
        if text.startswith('file_count: '):
            doc = yaml.safe_load('\n'.join(lines))
            return 'P' if int(text.split(': ')[1]) == len(doc['files']) else ''
        entry = _manifest_entry(lines, index)
        if entry == _MOCK_SERVER and side == '-' and _MOCK_SERVER not in ctx.new:
            return 'A'                      # the mock server's own entry
        if entry and text.startswith('  sha256: '):
            data = (ctx.old if side == '-' else ctx.new).get(entry)
            if (data is not None and ctx.old.get(entry) != ctx.new.get(entry)
                    and text == f'  sha256: {ctx.sha(data).hexdigest()}'):
                return 'P'                  # a changed file's own record, verified
        if text.startswith('- \'end-effector contract (ADR-0022') and side == '+':
            return 'C'                      # the contract's warnings (validate, D6)
        if text in ('warnings: []', 'warnings:'):
            return 'C'
    return ''


@needs_golden
def test_f3_the_opted_in_golden_differs_only_where_classified(tmp_path):
    """Every changed LINE of every changed file is [C], [A], [D] or [P] by an explicit pattern
    (ADR-0022 "Generated changes, classified"; review TSA-1 tests lens, T11 K1): a classified
    FILE holding an unclassified line fails -- e.g. a package.xml losing the dependency its
    scene.yaml still loads meshes from."""
    legacy = tmp_path / 'legacy'
    canonical = tmp_path / 'canonical'
    Orchestrator().generate(_golden(), legacy)
    Orchestrator().generate(F.opt_in(_golden()), canonical)
    old, new = _files(legacy), _files(canonical)
    changed = sorted(rel for rel in set(old) | set(new) if old.get(rel) != new.get(rel))
    assert set(changed) <= set(CLASSIFIED), sorted(set(changed) - set(CLASSIFIED))
    assert _MOCK_SERVER in old and _MOCK_SERVER not in new
    ctx = _Opted(old, new, F.opt_in(_golden()), _golden())
    unexpected, classes = [], {}
    for rel in changed:
        for side, index, text in _changed_lines(old.get(rel), new.get(rel)):
            cls = _line_class(rel, side, index, text, ctx)
            if cls != CLASSIFIED[rel] and not (cls and rel == 'generation_manifest.yaml'):
                unexpected.append(f'{rel}:{side}{index + 1} [{cls or "?"}] {text[:120]}')
            classes[cls] = classes.get(cls, 0) + 1
    assert not unexpected, unexpected
    # the classes are what the opt-in is: a contract, adapter rows, dependencies, provenance
    assert set(classes) == {'A', 'C', 'D', 'P'}, classes
    # [C] the eef block is there whole, in order; [A] BRIDGES = the kept rows + the adapter rows
    added = [t for s, _i, t in _changed_lines(old[f'{APP}/config/bt_params.yaml'],
                                              new[f'{APP}/config/bt_params.yaml']) if s == '+']
    assert added == ['    gripper_action: ""'] + ctx.eef_lines
    rows_old = _bridges(old[f'{CFG}/launch/bringup.launch.py'].decode())
    rows_new = _bridges(new[f'{CFG}/launch/bringup.launch.py'].decode())
    removed = {(b.package, b.executable) for b in ctx.removed}
    assert rows_new == [r for r in rows_old if (r['package'], r['executable']) not in removed] + [
        r.dict() for r in E.adapter_rows(ctx.project)]
    # [D] the scene's mesh package is declared where scene.yaml is used (T11 K1, fixed at the
    # source): the config package keeps fr3wml_isaac although no bridge row names it any more
    cfg_xml = new[f'{CFG}/package.xml'].decode()
    for package in E.scene_package_references([new[f'{CFG}/config/scene.yaml'].decode()]):
        assert f'<exec_depend>{package}</exec_depend>' in cfg_xml, package
    # what stays: motion, planner config, description, perception
    for rel in old:
        if rel.endswith(('.srdf', '.urdf.xacro', 'ros2_controllers.yaml', 'moveit_controllers.yaml',
                         'ompl_planning.yaml', 'perception.yaml', 'trainit_bt.launch.py')):
            assert old[rel] == new[rel], rel


@needs_golden
def test_f3_the_opted_in_golden_carries_the_contract(tmp_path):
    out = tmp_path / 'b'
    Orchestrator().generate(F.opt_in(_golden()), out)
    params = yaml.safe_load((out / APP / 'config' / 'bt_params.yaml').read_text())[
        '/**']['ros__parameters']
    assert params['gripper_action'] == ''
    assert not {'gripper_cmd_topic', 'gripper_close_position',
                'gripper_open_position'} & set(params)
    assert params['eef']['ids'] == ['main']
    assert params['eef']['main']['scene_signal'] == 'adapter'
    scene = yaml.safe_load((out / CFG / 'config' / 'scene.yaml').read_text())
    assert scene['scene_manager_node']['ros__parameters']['gripper_cmd_topic'] == \
        'eef/main/engaged'
    tree = (out / APP / 'bt_trees' / 'pick_place.xml').read_text()
    assert '<CloseGripper eef="main"/>' in tree and '<OpenGripper eef="main"/>' in tree
    assert 'SetReleasePolicy' not in tree and 'VerifyGrasp' not in tree    # the project: false
    rows = _bridges((out / CFG / 'launch' / 'bringup.launch.py').read_text())
    assert [(r['package'], r['executable'], r['modes']) for r in rows] == [
        ('fairino_gripper', 'joint_state_merger', ['mock', 'isaac']),
        ('depth_image_proc', 'point_cloud_xyzrgb_node', ['isaac']),
        ('trainit_hardware_layer', 'mock_eef_adapter', ['mock']),
        ('trainit_hardware_layer', 'isaac_eef_adapter', ['isaac'])]
    mock, isaac = rows[2]['parameters'], rows[3]['parameters']
    assert mock[0] == isaac[0] == {'device_id': 'main', 'eef': params['eef']}
    assert isaac[1]['stage_cmd_topic'] == '/suction_cmd'            # detected at Step 2
    assert isaac[1]['scene_reset_topic'] == '/isaac_scene_reset'
    echo = {k: v for k, v in isaac[1].items() if k.startswith('compat.pseudo_joint')}
    assert echo == {'compat.pseudo_joint_topic': '/suction/joint_states',     # the merger's input
                    'compat.pseudo_joint_name': 'suctioncup_joint',
                    'compat.pseudo_joint_engaged': 1.0, 'compat.pseudo_joint_released': 0.0}
    assert {k: v for k, v in mock[1].items() if k.startswith('compat.')} == echo
    assert not any('trigger' in k for k in mock[1])                 # no policy step: no pair
    cfg_xml = (out / CFG / 'package.xml').read_text()
    assert ('<exec_depend version_gte="0.2.0">trainit_hardware_layer</exec_depend>' in cfg_xml)
    assert cfg_xml.count('trainit_hardware_layer</exec_depend>') == 1
    app_xml = (out / APP / 'package.xml').read_text()
    assert '<exec_depend version_gte="2.0.0">trainit_motion_runtime</exec_depend>' in app_xml
    assert 'trainit_hardware_layer' not in app_xml                  # its launch starts none


@needs_golden
def test_f3_a_namespaced_canonical_isaac_cell(tmp_path):
    """Under /r1: the stage channel is RELATIVE (an absolute one is refused, R15 / D7), the reset
    topic becomes relative and per robot, the scene signal is relative, and the manifest says the
    stage must bind THIS robot's SurfaceGripper by path, on the robot's resolved names."""
    p = F.opt_in(_golden(), adapters={
        'mock': 'trainit_mock', 'real': 'none',
        'isaac': {'adapter': 'trainit_isaac_surface_gripper', 'stage_cmd_topic': 'suction_cmd'}})
    p.robot.cell_namespace = '/r1'
    # owner 11.1 (vNext M2.7): the golden's merger input /suction/joint_states is absolute, so the
    # echo would be shared by every robot of the cell -- refused (R26) until it is made relative
    refused = '\n'.join(E.eef_contract_problems(p))
    assert ('(ADR-0022 R26)' in refused and 'robot namespace "/r1"' in refused
            and '(/r1/eef/main)' in refused and 'echo topic "/suction/joint_states"' in refused
            and 'expected relative form "suction/joint_states"' in refused), refused
    merger = next(b for b in p.deployment.bridges if b.executable == 'joint_state_merger')
    merger.parameters = [{k: ('suction/joint_states' if k == 'gripper_joint_states_in' else v)
                          for k, v in d.items()} for d in merger.parameters]
    out = tmp_path / 'b'
    manifest = Orchestrator().generate(p, out)
    scene = yaml.safe_load((out / CFG / 'config' / 'scene.yaml').read_text())[
        '/**/scene_manager_node']['ros__parameters']
    assert scene['scene_reset_topic'] == 'isaac_scene_reset'
    assert scene['gripper_cmd_topic'] == 'eef/main/engaged'
    isaac = [r for r in _bridges((out / CFG / 'launch' / 'bringup.launch.py').read_text())
             if r['executable'] == 'isaac_eef_adapter'][0]
    assert isaac['parameters'][1]['stage_cmd_topic'] == 'suction_cmd'
    assert isaac['parameters'][1]['scene_reset_topic'] == 'isaac_scene_reset'
    assert isaac['parameters'][1]['compat.pseudo_joint_topic'] == 'suction/joint_states'
    assert any('the stage must bind this robot\'s SurfaceGripper to "/r1/suction_cmd" by path, '
               'and its reset script must listen on "/r1/isaac_scene_reset"' in w
               for w in manifest.warnings), manifest.warnings
    # the Step-2 stage topic as detected (absolute, global) is refused under the namespace
    q = F.opt_in(_golden())
    q.robot.cell_namespace = '/r1'
    assert any('R15' in m and 'stage_cmd_topic "/suction_cmd" is absolute' in m
               for m in E.eef_contract_problems(q))


# relative in both cases: at the root it resolves to the scene manager's own /isaac_scene_reset
@pytest.mark.parametrize('ns, reset', [('', 'isaac_scene_reset'), ('/r3', 'isaac_scene_reset')])
def test_a_canonical_isaac_cell_always_names_its_reset_topic(tmp_path, ns, reset):
    """Review TSA-7 (cross lens): with an isaac binding the reset topic is ALWAYS written -- in
    scene.yaml and in the isaac adapter row -- even when the project leaves it blank: the scene
    manager's own default is ABSOLUTE and latched (one channel for every robot of a cell), and
    the adapter must hear a stage reset to record it as a release (D8)."""
    p = F.suction_simple(tmp_path, ns=ns, modes=('mock', 'isaac'), contract=F.suction_contract(
        adapters={'mock': 'trainit_mock', 'isaac': {'adapter': 'trainit_isaac_surface_gripper',
                                                    'stage_cmd_topic': 'suction_cmd'}}))
    assert p.scene.scene_reset_topic == ''
    assert E.eef_contract_problems(p) == []
    assert E.scene_reset_topic(p) == reset
    from trainit_setup_assistant.generator.scene_yaml import build_scene_yaml
    assert f'    scene_reset_topic: "{reset}"' in build_scene_yaml(p).splitlines()
    isaac = [r for r in E.adapter_rows(p) if r.executable == 'isaac_eef_adapter'][0]
    assert isaac.parameters[1]['scene_reset_topic'] == reset
    # without an isaac binding: at the root nothing is added (one robot, the scene manager's own
    # default); under a cell namespace the per-robot relative name is written anyway (D7: no
    # latched name shared by the robots of a cell). A LEGACY project keeps its blank topic.
    mock_only = F.suction_simple(tmp_path / 'mock', ns=ns)
    assert E.scene_reset_topic(mock_only) == ('isaac_scene_reset' if ns else '')
    legacy = load_project(str(F.DATA.parents[2] / 'examples' / 'fr3wml_project.yaml'))
    assert E.scene_reset_topic(legacy) == legacy.scene.scene_reset_topic


@pytest.mark.parametrize('max_age_ms', [1, 7, 30, 40, 49, 50, 500])
def test_the_mock_row_samples_fast_enough_for_max_age(tmp_path, max_age_ms):
    """Review CROSS-3: the mock adapter refuses to start unless sample_rate_hz * max_age_ms >=
    1000, at 20 Hz by default -- so the row carries the rate it needs (only when 20 Hz would be
    too slow: the default rows stay as they were)."""
    p = F.with_contract(F.suction_simple(tmp_path, ns=''),
                        **{'verification.max_age_ms': max_age_ms})
    assert E.eef_contract_problems(p) == []
    mock = [r for r in E.adapter_rows(p) if r.executable == 'mock_eef_adapter'][0]
    own = mock.parameters[1] if len(mock.parameters) > 1 else {}
    rate = own.get('mock.sample_rate_hz', 20.0)
    assert rate * max_age_ms >= 1000, (max_age_ms, own)
    assert ('mock.sample_rate_hz' in own) == (max_age_ms < 50)
    if 'mock.sample_rate_hz' in own:
        assert isinstance(own['mock.sample_rate_hz'], float)


@pytest.mark.parametrize('word', ['on', 'Off', 'yes', 'No', 'y', 'n', 'true', 'NULL'])
def test_a_key_yaml_would_not_read_as_itself_is_quoted(tmp_path, word):
    """Review TSA-6: a target (or an id) named like a YAML 1.1 bool/null word is a valid name
    token, and a bare key would load as True/False/None in PyYAML -- the reference normaliser
    would see another key. It is quoted; every other key stays bare."""
    contract = F.jaw_contract()
    contract['actuator']['targets'] = {'open': 0.0, 'closed': 0.7, word: 0.35}
    p = F.jaw_simple(tmp_path, contract=contract)
    assert E.eef_contract_problems(p) == []
    lines = E.render_parameter_block(E.eef_parameter_block(E.device(p)), indent=4)
    assert f'              "{word}": 0.35' in lines
    assert '      ids: ["jaw_simple"]' in lines and '              open: 0.0' in lines
    N = F.normalizer()
    path = tmp_path / 'bt_params.yaml'
    path.write_text('/**:\n  ros__parameters:\n' + '\n'.join(lines) + '\n')
    flat = N.load_params(str(path))
    assert flat[f'eef.jaw_simple.actuators.gripper.targets.{word}'] == 0.35
    assert N.parse(flat)[f'jaw_simple.actuators.gripper.targets.{word}'] == 0.35


def test_a_device_id_yaml_would_not_read_as_itself_is_quoted(tmp_path):
    p = F.suction_simple(tmp_path, contract=F.suction_contract(device_id='yes'))
    lines = E.render_parameter_block(E.eef_parameter_block(E.device(p)), indent=4)
    assert '      "yes":' in lines and '      ids: ["yes"]' in lines
    N = F.normalizer()
    path = tmp_path / 'bt_params.yaml'
    path.write_text('/**:\n  ros__parameters:\n' + '\n'.join(lines) + '\n')
    assert N.parse(N.load_params(str(path)))['ids'] == ['yes']


def _scene_loader_suction(root, contract=True):
    """The simple suction as a SCENE-LOADER cell (a minimal synthetic base config, no golden)
    with a static mesh obstacle loaded from ``package://cell_meshes/`` -- a package no bridge row
    names (the dependency a legacy bundle took only from its bridges)."""
    from trainit_setup_assistant.model.enums import ShapeType
    from trainit_setup_assistant.model.scene import SceneObject
    p = F.suction_simple(root, ns='')
    if not contract:
        p.robot.gripper = GripperSpec(**dict(F.dump(p.robot.gripper), contract=None,
                                             actuation='trigger'))
        p.schema_version = 2                    # a legacy file's number (R19 otherwise)
    cfg = root / 'base' / 'config'
    cfg.mkdir(parents=True)
    (cfg / 'r.urdf.xacro').write_text('<?xml version="1.0"?>\n<robot '
                                      'xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"/>\n')
    (cfg / 'r.srdf').write_text('<?xml version="1.0"?>\n<robot name="r"></robot>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml',
              'moveit_controllers.yaml', 'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    p.robot.base_moveit_config_path = str(root / 'base')
    p.scene.objects = [SceneObject(id='table', shape=ShapeType.MESH, dims=[1.0, 1.0, 0.1],
                                   mesh_resource='package://cell_meshes/meshes/table.stl')]
    return p


def test_a_canonical_package_declares_the_mesh_packages_its_scene_loads(tmp_path):
    """Review TSA-1 (tests lens), T11 K1, fixed at the source: every package a canonical
    scene.yaml loads a mesh from is declared by the package that carries scene.yaml -- a
    dependency no bridge row names any more. A LEGACY package keeps its dependencies as they
    were (from the bridge rows only)."""
    out = tmp_path / 'canonical'
    generate_scene_loader_config(_scene_loader_suction(tmp_path / 'c'), out, 'cell_cfg')
    scene = (out / 'cell_cfg' / 'config' / 'scene.yaml').read_text()
    assert 'package://cell_meshes/meshes/table.stl' in scene
    assert '  <exec_depend>cell_meshes</exec_depend>' in (
        out / 'cell_cfg' / 'package.xml').read_text()
    legacy = tmp_path / 'legacy'
    generate_scene_loader_config(_scene_loader_suction(tmp_path / 'l', contract=False), legacy,
                                 'cell_cfg')
    assert 'cell_meshes' not in (legacy / 'cell_cfg' / 'package.xml').read_text()
    assert E.scene_package_references(['a: "package://p_1/x"\nb: package://q/y package://p_1/z'
                                       ]) == ['p_1', 'q']


@needs_golden
def test_f3_the_gazebo_weld_follows_the_scene_signal(tmp_path):
    p = F.opt_in(_golden(), adapters={'mock': 'trainit_mock', 'isaac':
                                      'trainit_isaac_surface_gripper', 'real': 'none',
                                      'gazebo': 'none'})
    p.deployment.modes = list(p.deployment.modes) + [Backend.GAZEBO]
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    launch = (out / CFG / 'launch' / 'bringup.launch.py').read_text()
    assert "'gripper_cmd_topic': 'eef/main/engaged'" in launch


# --- F3: the composite trees, ids only; two robots, disjoint names ------------------------------

COMPOSITE_TREE = [
    'MoveWaypoint home', 'MoveWaypoint pick',
    'CommandActuator rotary_axis receive', 'CommandActuator binary_lock engage',
    'CloseGripper site_a', 'VerifyGrasp site_a held',
    'MoveWaypoint flip',
    'CommandActuator rotary_axis transfer', 'CommandActuator linear_axis extended',
    'CommandActuator wrist_mechanism present',
    'CloseGripper site_b', 'VerifyGrasp site_b held',      # the guard: site_a opens only after
    'VerifyGrasp site_a held', 'OpenGripper site_a', 'VerifyGrasp site_a released',
    'CommandActuator linear_axis home',
    'MoveWaypoint place',
    'VerifyGrasp site_b held', 'OpenGripper site_b', 'VerifyGrasp site_b released',
    'CommandActuator binary_lock release',
    'MoveWaypoint home']


def _steps(tree_xml: str) -> list:
    root = ET.fromstring(tree_xml[tree_xml.index('<root'):])
    out = []
    for el in root.iter():
        if el.tag == 'MoveWaypoint':
            out.append(f'MoveWaypoint {el.get("waypoint")}')
        elif el.tag == 'CommandActuator':
            out.append(f'CommandActuator {el.get("actuator")} {el.get("target")}')
        elif el.tag in ('CloseGripper', 'OpenGripper'):
            out.append(f'{el.tag} {el.get("site")}')
        elif el.tag == 'VerifyGrasp':
            out.append(f'VerifyGrasp {el.get("site")} {el.get("expect")}')
    return out


def test_f3_the_composite_tree_is_the_handoff(tmp_path):
    out = tmp_path / 'b'
    Orchestrator().generate(F.composite(tmp_path), out)
    tree = (out / 'robot_a_app' / 'bt_trees' / 'pick_place.xml').read_text()
    assert _steps(tree) == COMPOSITE_TREE
    assert all(el.get('eef') == 'composite_tool'
               for el in ET.fromstring(tree[tree.index('<root'):]).iter() if el.tag in EEF_TAGS)


def _resolved(ns: str, block: dict, remaps: dict) -> set:
    """Every endpoint name a device's block resolves to in ``ns`` (ADR-0022 D7: all relative),
    through the adapter row's launch remappings (a composite mock row routes its joints onto the
    robot's joint_states, vNext M2.7)."""
    names = set()
    for dev in block['ids']:
        d = block[dev]
        joints = remaps.get(f'eef/{dev}/joint_states', f'eef/{dev}/joint_states')
        names |= {f'{ns}/eef/{dev}/state', f'{ns}/{joints}'}
        if d['scene_signal'] != 'none':
            names.add(f'{ns}/eef/{dev}/engaged')
        for a in d['actuator_ids']:
            act = d['actuators'][a]
            if act['command_interface'] == 'set_bool':
                names.add(f'{ns}/eef/{dev}/actuators/{a}/set_engaged')
            else:
                assert not act['action'].startswith(('/', '~'))
                names.add(f'{ns}/{act["action"]}')
    return names


def test_f3_two_robots_with_identical_ids_resolve_apart(tmp_path):
    """robot_a in /r1 and robot_b in /r2 (identical eef, actuator and site ids), the root variant
    and suction_simple in /r3, generated at once: every end-effector name exists once per
    namespace, none outside one, and no tree carries a '/' -- trees carry ids only."""
    projects = {'/r1': F.composite(tmp_path, 'robot_a', '/r1'),
                '/r2': F.composite(tmp_path, 'robot_b', '/r2'),
                '': F.composite(tmp_path, 'robot_root', '', prefix=''),
                '/r3': F.suction_simple(tmp_path, ns='/r3')}
    names, trees = {}, []
    for ns, p in projects.items():
        out = tmp_path / f'out{ns.replace("/", "_") or "_root"}'
        Orchestrator().generate(p, out)
        pkg = p.bundle.app_package
        block = yaml.safe_load((out / pkg / 'config' / 'bt_params.yaml').read_text())[
            '/**']['ros__parameters']['eef']
        trees.append((out / pkg / 'bt_trees' / 'pick_place.xml').read_text())
        bringup = (out / pkg / 'launch' / 'bringup.launch.py').read_text()
        assert (f'NS = "{ns}"' in bringup) == bool(ns)
        remaps = {}
        for row in _bridges(bringup):
            if row['package'] == 'trainit_hardware_layer':
                assert row['parameters'][0]['eef'] == block
                remaps.update({a: b for a, b in row['remappings']})
        names[ns] = _resolved(ns, block, remaps)
    pairs = [(a, b) for a in names for b in names if a < b]
    assert all(not names[a] & names[b] for a, b in pairs), pairs
    assert '/r1/eef/composite_tool/state' in names['/r1']
    assert '/r2/linear_axis_controller/follow_joint_trajectory' in names['/r2']
    assert '/eef/composite_tool/state' in names['']
    # the composite rows route the tool joints onto the robot's own joint_states, per namespace
    assert {'/r1/joint_states', '/r2/joint_states'} <= names['/r1'] | names['/r2']
    assert '/joint_states' in names['']
    assert '/r3/eef/suction_simple/joint_states' in names['/r3']          # a simple row: no remap
    for tree in trees:
        for el in ET.fromstring(tree[tree.index('<root'):]).iter():
            if el.tag in EEF_TAGS:
                assert not any('/' in v for v in el.attrib.values()), el.attrib


# --- F3: the policy bracket and the card's Trigger pair -----------------------------------------

def test_f3_a_pure_policy_in_a_canonical_cell(tmp_path):
    """ADR-0022 D4/D10: the POSITION half stays CheckRobotState; the ATTACHED half is VerifyGrasp;
    the adapter serves the card's Trigger pair at its exact names (the runtime is unchanged)."""
    card = tmp_path / 'card' / 'policy_card.yaml'
    card.parent.mkdir()
    card.write_text('policy_card:\n  name: synthetic_suction\n  end_effector: suction\n'
                    '  grasp: {on_service: /vacuum/on, off_service: /vacuum/off}\n'
                    '  end_state: {tcp_pose_base: [0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0]}\n')
    p = F.suction_simple(tmp_path, ns='', contract=F.suction_contract(grasp=True))
    p.application.policies = [PolicyStep(name='grasp', before_waypoint='pick', mode='pure',
                                         card=str(card), attached_topic='/vacuum/attached')]
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    pkg = p.bundle.app_package
    # the step's attached_topic is the UNCHANGED policy runtime's attached-observation input
    # (ADR-0022 D10): carried to its launch exactly as in a legacy project, never refused
    policy_launch = (out / pkg / 'launch' / 'policy_runtime.launch.py').read_text()
    assert '"suction_state_topic": "/vacuum/attached",' in policy_launch
    tree = (out / pkg / 'bt_trees' / 'pick_place.xml').read_text()
    check = re.search(r'<CheckRobotState [^>]*/>', tree).group(0)
    assert 'position=' in check and 'require_attached' not in check
    assert tree.index('<RunPolicy') < tree.index(check) < tree.index(
        '<VerifyGrasp eef="suction_simple" expect="held"/>')
    mock = [r for r in _bridges((out / pkg / 'launch' / 'bringup.launch.py').read_text())
            if r['executable'] == 'mock_eef_adapter'][0]
    assert (mock['parameters'][1]['compat.trigger_on'],
            mock['parameters'][1]['compat.trigger_off']) == ('/vacuum/on', '/vacuum/off')


def test_f3_command_actuator_to_a_named_target(tmp_path):
    contract = F.jaw_contract()
    contract['actuator']['targets'] = {'open': 0.0, 'closed': 0.7, 'half': 0.35}
    p = F.jaw_simple(tmp_path, contract=contract)
    p.application.tool_actions.append(ToolAction(at_waypoint='flip', kind=ToolActionKind.ACTUATE,
                                                 actuator='gripper', target='half'))
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    tree = (out / p.bundle.app_package / 'bt_trees' / 'pick_place.xml').read_text()
    assert '<CommandActuator eef="jaw_simple" actuator="gripper" target="half"/>' in tree
    params = yaml.safe_load((out / p.bundle.app_package / 'config' / 'bt_params.yaml').read_text())[
        '/**']['ros__parameters']
    assert params['eef']['jaw_simple']['scene_signal'] == 'tmr'
    assert params['eef']['jaw_simple']['actuators']['gripper']['targets'] == {
        'open': 0.0, 'closed': 0.7, 'half': 0.35}
    assert not (out / p.bundle.app_package / 'scripts').exists()


def test_f3_no_legacy_gripper_key_and_no_empty_sequence(tmp_path):
    """TMR's runner exits with code 2 on a legacy gripper key next to the block, and a YAML
    empty sequence has no type: neither is ever emitted (EEF_CONTRACT_SPEC section 2)."""
    for build in (F.suction_simple, F.jaw_simple, F.jaw_binary):
        p = build(tmp_path / build.__name__)
        out = tmp_path / f'out_{build.__name__}'
        Orchestrator().generate(p, out)
        text = (out / p.bundle.app_package / 'config' / 'bt_params.yaml').read_text()
        params = yaml.safe_load(text)['/**']['ros__parameters']
        assert params['gripper_action'] == ''
        assert not {'gripper_cmd_topic', 'gripper_close_position',
                    'gripper_open_position'} & set(params)

        def walk(node):
            for v in (node.values() if isinstance(node, dict) else node):
                assert v != [] and v != {} and v is not None
                if isinstance(v, (dict, list)):
                    walk(v)
        walk(params['eef'])


def _not_well_formed(root: Path) -> list:
    """Every XML file of a generated bundle, parsed WHOLE by a strict parser (expat). A comment with
    two dashes in a row, for one, is not well-formed: xmllint refuses it, while BT.CPP's tinyxml2
    reads it, so only a strict parse of the whole document catches it (the tree checks above
    parse from ``<root`` on, past the header comment)."""
    import xml.parsers.expat
    bad = []
    for p in sorted(root.rglob('*')):
        if p.is_file() and p.suffix in ('.xml', '.srdf', '.urdf', '.xacro'):
            try:
                xml.parsers.expat.ParserCreate().Parse(p.read_bytes(), True)
            except xml.parsers.expat.ExpatError as e:
                bad.append(f'{p.relative_to(root).as_posix()}: {e}')
    return bad


def _bin_picking(canonical: bool):
    import test_bin_picking as tb
    import test_m2_6_eef_bin_picking as tbp
    return tbp._canonical(grasp=True, release=True) if canonical else tb._bin_project()


WELL_FORMED_CASES = {
    'golden': lambda t: _golden(),
    'golden_canonical': lambda t: F.opt_in(_golden()),
    'bin_picking': lambda t: _bin_picking(False),
    'bin_picking_canonical': lambda t: _bin_picking(True),
    'suction_simple': lambda t: F.suction_simple(t),
    'jaw_simple': lambda t: F.jaw_simple(t),
    'jaw_binary': lambda t: F.jaw_binary(t),
}


# the cases built on the golden (bin picking is the golden switched to it); the others are synthetic
GOLDEN_CASES = ('golden', 'golden_canonical', 'bin_picking', 'bin_picking_canonical')


@pytest.mark.parametrize('case', sorted(WELL_FORMED_CASES))
def test_f3_every_generated_xml_file_is_well_formed(tmp_path, case):
    if case in GOLDEN_CASES and not GOLDEN_PROJECT.is_file():
        F.skip_or_fail(f'the golden project is not present: {GOLDEN_PROJECT}')
    out = tmp_path / 'b'
    Orchestrator().generate(WELL_FORMED_CASES[case](tmp_path / 'in'), out)
    assert _not_well_formed(out) == []


def test_f3_the_composite_bundle_xml_is_well_formed(tmp_path):
    out = tmp_path / 'b'
    Orchestrator().generate(F.composite(tmp_path), out)
    assert _not_well_formed(out) == []


def test_yaml_floats_always_load_as_floats():
    for v in (0.0, 1.0, 0.02, 1e-05, 2.5e-07, 1e20, -0.3, 123456789.0):
        text = E.yaml_float(v)
        assert isinstance(yaml.safe_load(f'x: {text}')['x'], float), text
        assert float(text) == v


# --- F5: pose relevance -------------------------------------------------------------------------

def test_f5_which_actuators_move_a_site(tmp_path):
    p = F.composite(tmp_path)
    urdf, why = E.description_view(p)
    assert urdf is not None, why
    dev = E.device(p)
    assert E.moved_by(urdf, dev, dev.site('site_a')) == ['linear_axis', 'rotary_axis']
    assert E.moved_by(urdf, dev, dev.site('site_b')) == ['wrist_mechanism']
    movers = set(E.moved_by(urdf, dev, dev.site('site_a'))) | set(
        E.moved_by(urdf, dev, dev.site('site_b')))
    assert not movers & {'gripper_a', 'gripper_b', 'binary_lock'}      # the jaws move no site
    p.robot.tip_link = p.robot.planning_group.tip_link = 'robot_a_site_a_tcp'
    assert any('R10' in m for m in E.eef_contract_problems(p))


# --- F7: the composite boundary tripwire --------------------------------------------------------

TOOL_JOINTS = ('linear_axis_joint', 'rotary_axis_joint', 'wrist_pitch_joint', 'wrist_roll_joint',
               'jaw_a_joint', 'jaw_b_joint', 'jaw_a_follower_joint', 'jaw_b_follower_joint')


def test_f7_composite_emission_writes_no_tool_controller_joint_group_or_state(tmp_path):
    """A composite (mock) bundle gets NOTHING of its tool in the MoveIt /
    ros2_control config: generating tool joints, controllers, SRDF groups or named states is the
    owner's section-23 question (ADR-0022 Q1), not a side effect of emission."""
    p = F.composite(tmp_path)
    p.robot.gripper.command_joint = 'robot_a_jaw_a_joint'      # tempting, and still nothing
    p.robot.gripper.controller_name = 'gripper_a_controller'
    p.robot.gripper.action_ns = 'gripper_cmd'
    from trainit_setup_assistant.model import NamedState
    p.robot.named_states = [
        NamedState(name='home', group='arm', joint_values={'robot_a_joint1': 0.0}),
        NamedState(name='tool_open', group='composite_tool',
                   joint_values={'robot_a_jaw_a_joint': 0.02})]
    p.robot.gripper.eef_name = 'composite_tool'
    out = tmp_path / 'b'
    Orchestrator().generate(p, out)
    cfg = out / 'robot_a_trainit_config' / 'config'
    ros2_control = (cfg / 'robot_a.ros2_control.xacro').read_text()
    assert re.findall(r'<joint name="([^"]+)"', ros2_control) == ['robot_a_joint1']
    controllers = {k.replace('/**/', ''): v for k, v in yaml.safe_load(
        (cfg / 'ros2_controllers.yaml').read_text()).items()}
    assert sorted(controllers) == ['controller_manager', 'moveit_joint_controller']
    assert sorted(controllers['controller_manager']['ros__parameters']) == [
        'joint_state_broadcaster', 'moveit_joint_controller', 'update_rate']
    moveit = yaml.safe_load((cfg / 'moveit_controllers.yaml').read_text())
    assert moveit['moveit_simple_controller_manager']['controller_names'] == [
        'moveit_joint_controller']
    srdf = ET.fromstring((cfg / 'robot_a.srdf').read_text())
    assert [g.get('name') for g in srdf.findall('group')] == ['arm']
    assert srdf.findall('end_effector') == []
    assert [s.get('group') for s in srdf.findall('group_state')] == ['arm']
    for f in cfg.iterdir():
        text = f.read_text()
        assert not any(j in text for j in TOOL_JOINTS), f.name
