"""The end-effector contract in TSA's model (vNext M2.6, ADR-0022 D6): F1 representability, the
short form stored as written, the guard token and its one validator, legacy saves byte-identical,
the shared conformance fixture and the token parity, and the catalog's adapter rows.

* F1  the composite fixture derives 6 actuators on 3 interfaces with their implied capabilities
      and 2 sites; the mimic followers are refused as actuator joints; the simple suction is three
      typed lines; a GripperCommand jaw and a binary jaw are as short;
* the short form is STORED as written and the long form derived on demand: an edit of
  ``eef_parent_link`` moves the derived TCP and the file stays three lines;
* the token ``actuation: contract``: ONE pre-validator sets it whenever a contract is present
  (over an explicit trigger/joint_position, which validate() then names); refused without a
  contract (R19), like ``schema_version: 3`` without one;
* a LEGACY project saves byte-identical to TSA 7.6.0 (the hashes 7.6.0 itself wrote, pinned);
* the conformance fixture (byte copies pinned by sha256): the ``eef`` blocks TSA emits for the four
  conformance devices normalise -- with the pinned ``normalize_eef.py`` -- to the pinned dump;
  the token parity (TSA's vocabularies and catalog rows against the pinned token file);
* the catalog: implementation rows, never kinds (no composite row), corrected sim-grasp rows.

Fast and ROS-free except the description compile (xacro, in-process).
"""
import hashlib
import json
from pathlib import Path

import pytest
import yaml
from pydantic import BaseModel, Field, ValidationError

import eef_fixture as F
import ws_paths
from trainit_setup_assistant import catalog
from trainit_setup_assistant.applications.base import build_bt_params_context
from trainit_setup_assistant.generator import eef_contract as E
from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.model import (CommandInterface, EndEffectorCapability,
                                           EvidenceClass, GripperSpec, ResultPolicy,
                                           SceneSignal, ToolAction, load_project, save_project)
from trainit_setup_assistant.model import end_effector as M
from trainit_setup_assistant.model.enums import (EndEffectorActuation, GripperKind,
                                                 ToolActionKind)

TEST = Path(__file__).resolve().parent
PKG = TEST.parent
EXAMPLE = PKG / 'examples' / 'fr3wml_project.yaml'
GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
TEST4_PROJECT = ws_paths.ws_src('fr3wml_digital_twin', 'fr3wml_suction_policy_test4_bundle',
                                'project.yaml')


# --- the shared conformance files: byte copies, pinned ------------------------------------------

@pytest.mark.parametrize('name', sorted(F.PINNED))
def test_the_shared_contract_files_are_the_pinned_bytes(name):
    """ADR-0022 D8: TSA, TMR and the hardware layer each keep byte copies of the conformance
    fixture, its expected dump, the token file and the reference normaliser, pinned by sha256."""
    assert F.sha256(name) == F.PINNED[name]


def _bt_params(project) -> str:
    return build_jinja_env().get_template('app/bt_params.yaml.j2').render(
        **build_bt_params_context(project))


def _conformance_projects(root):
    return {'composite_tool': F.composite(root),
            'suction_simple': F.suction_simple(root),
            'jaw_simple': F.jaw_simple(root),
            'jaw_binary': F.jaw_binary(root)}


def test_the_emitted_eef_blocks_normalise_to_the_conformance_dump(tmp_path):
    """TSA's generator side of the conformance proof: the ``eef`` block of each conformance device,
    as TSA renders it into bt_params.yaml, merged in the fixture's device order, normalises with
    the pinned normalize_eef.py to the pinned expected dump, byte for byte. Each block alone also
    passes the normaliser (no refusal key)."""
    N = F.normalizer()
    merged = {'gripper_action': ''}
    ids = []
    for device_id, project in _conformance_projects(tmp_path).items():
        path = tmp_path / f'{device_id}_bt_params.yaml'
        path.write_text(_bt_params(project))
        flat = N.load_params(str(path))
        assert flat['gripper_action'] == ''
        assert not {'gripper_cmd_topic', 'gripper_close_position', 'gripper_open_position'} & set(
            flat)
        single = N.parse(flat)                              # no Refused
        assert single['ids'] == [device_id]
        ids += flat['eef.ids']
        merged.update({k: v for k, v in flat.items() if k.startswith('eef.') and k != 'eef.ids'})
    merged['eef.ids'] = ids
    assert tuple(ids) == F.CONFORMANCE_IDS
    assert N.dump(N.parse(merged)) == (F.DATA / 'eef_conformance.expected.txt').read_text()


def test_the_conformance_fixture_itself_normalises_to_its_dump():
    """The pinned pair agrees with the pinned normaliser (what the two other parsers prove too)."""
    N = F.normalizer()
    got = N.dump(N.parse(N.load_params(str(F.DATA / 'eef_conformance.yaml'))))
    assert got == (F.DATA / 'eef_conformance.expected.txt').read_text()


@pytest.mark.parametrize('case, row, key', [
    ('swapped_set_bool_site', 9, 'eef.suction_simple.sites.main.engage'),
    ('equal_target_values', 7, 'eef.jaw_simple.actuators.gripper.targets'),
])
def test_tsa_refuses_what_the_reference_normaliser_refuses(tmp_path, case, row, key):
    """The two rules the reference normaliser gained (sha256 9fba00c3...): a set_bool site's
    engage is "engage" (its polarity is the adapter's), and a gripper_command actuator's target
    VALUES are pairwise distinct (an adapter tells a command by its position). TSA refuses both
    before emission with its own row (R9, R7); the block it would have emitted is the one the
    normaliser refuses, with that key."""
    if case == 'swapped_set_bool_site':
        p = F.with_contract(F.suction_simple(tmp_path), actuator=F.DELETE,
                            actuators={'vacuum': {'command_interface': 'set_bool'}},
                            sites={'main': {'tcp': 'tcp', 'grip': 'vacuum', 'engage': 'release',
                                            'release': 'engage'}})
    else:
        contract = F.jaw_contract()
        contract['actuator']['targets'] = {'open': 0.0, 'closed': 0.7, 'pinch': 0.7}
        p = F.jaw_simple(tmp_path, contract=contract)
    assert [m for m in E.eef_contract_problems(p) if f'(ADR-0022 R{row})' in m]
    N = F.normalizer()
    path = tmp_path / 'bt_params.yaml'
    path.write_text('/**:\n  ros__parameters:\n    gripper_action: ""\n' + '\n'.join(
        E.render_parameter_block(E.eef_parameter_block(E.device(p)), indent=4)) + '\n')
    with pytest.raises(N.Refused) as refused:
        N.parse(N.load_params(str(path)))
    assert refused.value.key == key


def test_the_token_parity():
    """TSA's vocabularies and its catalog's adapter rows, token for token against the pinned token
    file (ADR-0022 D6: one closed list, shared with trainit_interfaces, TMR, the hardware layer)."""
    t = F.tokens()
    assert [c.value for c in CommandInterface] == [
        k for k in t['command_interface'] if k != 'unknown']
    assert [int(t['command_interface'][c.value]) for c in CommandInterface] == [1, 2, 3]
    assert [c.value for c in EndEffectorCapability] == list(t['capability'])
    assert [c.value for c in ResultPolicy] == list(t['result_policy'])
    assert [c.value for c in SceneSignal] == list(t['scene_signal'])
    assert sorted(e.value for e in M.ACCEPTABLE_EVIDENCE) == sorted(t['accepted_evidence'])
    assert {e.value for e in EvidenceClass} == set(t['evidence']) - {'none'}
    assert {e.value for e in M.NEVER_EVIDENCE} == {'command_echo', 'sim_attachment'}
    assert [e.value for e in M.DEFAULT_ACCEPTED_EVIDENCE] == ['sensor', 'sim_ground_truth']
    for token, row in catalog.EEF_ADAPTER_ROWS.items():
        assert set(row['serves']) <= set(t['command_interface']) - {'unknown'}, token
        provided = set(row['provides']) | {x for v in row['provides_if'].values() for x in v}
        assert provided <= set(t['capability']), token
        assert set(row['evidence']) <= set(t['accepted_evidence']), token


# --- F1: representable ---------------------------------------------------------------------------

def test_f1_the_composite_fixture_is_representable(tmp_path):
    """6 actuators on 3 interfaces, in contract order, with their implied capabilities; 2 sites,
    each with its grip; the device's scene signal is none (multi-site)."""
    dev = E.device(F.composite(tmp_path))
    assert [(a.id, a.interface.value) for a in dev.actuators] == [
        ('linear_axis', 'follow_joint_trajectory'), ('rotary_axis', 'follow_joint_trajectory'),
        ('wrist_mechanism', 'follow_joint_trajectory'), ('gripper_a', 'gripper_command'),
        ('gripper_b', 'set_bool'), ('binary_lock', 'set_bool')]
    assert {a.interface for a in dev.actuators} == set(CommandInterface)
    implied = {a.id: [c.value for c in M.IMPLIED_CAPABILITIES[a.interface]]
               for a in dev.actuators}
    assert implied['wrist_mechanism'] == ['joint_position_control', 'trajectory_control']
    assert implied['gripper_a'] == ['gripper_control', 'position_control']
    assert implied['binary_lock'] == ['binary_actuation']
    assert [(s.id, s.tcp, s.grip, s.engage, s.release) for s in dev.sites] == [
        ('site_a', 'robot_a_site_a_tcp', 'gripper_a', 'closed', 'open'),
        ('site_b', 'robot_a_site_b_tcp', 'gripper_b', 'engage', 'release')]
    assert dev.scene_signal() is SceneSignal.NONE
    assert dev.actuator('wrist_mechanism').joints == ['robot_a_wrist_pitch_joint',
                                                      'robot_a_wrist_roll_joint']
    # vNext M2.7 (owner Q1 option A): the composite MOCK bundle is no longer refused (R22 narrowed)
    assert E.eef_contract_problems(F.composite(tmp_path / 'b')) == []


def test_f1_a_mimic_follower_is_never_an_actuator_joint(tmp_path):
    p = F.with_contract(F.composite(tmp_path),
                        **{'actuators.gripper_a.joints': ['robot_a_jaw_a_follower_joint']})
    problems = '\n'.join(E.eef_contract_problems(p))
    assert 'R5' in problems and 'mimic follower' in problems and 'robot_a_jaw_a_joint' in problems


@pytest.mark.parametrize('build, kind, interface, actuator, site, scene', [
    (F.suction_simple, 'suction', 'set_bool', 'vacuum', 'tcp', 'adapter'),
    (F.jaw_simple, 'parallel', 'gripper_command', 'gripper', 'tool0', 'tmr'),
    (F.jaw_binary, 'parallel', 'set_bool', 'gripper', 'tool0', 'adapter'),
], ids=['suction', 'gripper_command_jaw', 'binary_jaw'])
def test_f1_the_simple_devices_are_short(tmp_path, build, kind, interface, actuator, site, scene):
    """Owner section 24.1-3: one actuator, one implicit site ``main`` whose TCP is
    eef_parent_link; the scene signal's source by the grip's interface (ADR-0022 D8)."""
    p = build(tmp_path)
    assert E.eef_contract_problems(p) == []
    dev = E.device(p)
    assert (dev.kind.value, [(a.id, a.interface.value) for a in dev.actuators]) == (
        kind, [(actuator, interface)])
    assert [(s.id, s.grip, s.tcp) for s in dev.sites] == [('main', actuator, site)]
    assert dev.scene_signal().value == scene


def test_f1_the_jaw_short_form_derives_action_joints_and_targets(tmp_path):
    """D6 derivation table: the jaw action is <controller_name>/<action_ns>, RELATIVE (never the
    absolute gripper_action_ns()); joints [command_joint]; targets from the angles with NO
    1.0/0.0 fallback -- an unresolvable target is a refusal, not a default."""
    contract = F.jaw_contract()
    del contract['actuator']['targets']
    p = F.jaw_simple(tmp_path, contract=contract, joint_target='angle', open_angle=0.05,
                     closed_angle=0.6)
    a = E.device(p).actuators[0]
    assert (a.action, a.joints, a.targets) == ('gripper_controller/gripper_cmd',
                                               ['finger_joint'], {'open': 0.05, 'closed': 0.6})
    assert set(a.derived) == {'action', 'joints', 'targets'}
    assert p.robot.gripper.gripper_action_ns() == '/gripper_controller/gripper_cmd'   # unused
    q = F.jaw_simple(tmp_path / 'q', contract=F.jaw_contract())
    q = F.with_contract(q, **{'actuator.targets': F.DELETE})
    problems = '\n'.join(E.eef_contract_problems(q))
    assert 'R7' in problems and 'never falls back to 1.0/0.0' in problems


# --- the short form stored as written ------------------------------------------------------------

def _gripper_block(text: str) -> dict:
    return yaml.safe_load(text)['robot']['gripper']


def test_the_short_form_is_stored_as_written_and_derived_on_demand(tmp_path):
    p = F.suction_simple(tmp_path)
    path = save_project(p, tmp_path / 'project.yaml')
    saved = _gripper_block(path.read_text())
    assert saved['actuation'] == 'contract'
    assert saved['contract'] == {
        'id': 'suction_simple', 'actuator': {'command_interface': 'set_bool'},
        'verification': {'require_grasp_confirmation': False,
                         'require_release_confirmation': False},
        'adapters': {'mock': 'trainit_mock'}}
    assert yaml.safe_load(path.read_text())['schema_version'] == 3
    again = load_project(path)
    assert again.robot.gripper.contract == p.robot.gripper.contract
    # an edit of what a short form derives from moves the derived value, and the file stays short
    again.robot.gripper.eef_parent_link = 'suction_body'
    assert E.device(again).sites[0].tcp == 'suction_body'
    path2 = save_project(again, tmp_path / 'project2.yaml')
    assert _gripper_block(path2.read_text())['contract'] == saved['contract']


def test_default_keys_are_dropped_inside_the_contract_only(tmp_path):
    p = F.with_contract(F.suction_simple(tmp_path), **{'verification.timeout_ms': 3000,
                                                       'command_timeout_ms': 10000})
    saved = _gripper_block(save_project(p, tmp_path / 'p.yaml').read_text())
    assert 'timeout_ms' not in saved['contract']['verification']
    assert 'command_timeout_ms' not in saved['contract']
    assert saved['tool_radius_m'] == 0.0                     # outside the contract: kept


def test_default_factory_keys_are_dropped_too_on_every_pydantic(tmp_path):
    """Review TSA-4: ``exclude_defaults`` alone keeps a ``default_factory`` field on pydantic 1.x
    (it compares it with None), so the same project saved different bytes on 1.x and 2.x. The
    contract is written by walking its fields against their defaults, factories included:
    ``accepted_evidence`` at its default and an empty ``adapters`` are dropped on both, while a
    value that differs from its default is kept as written."""
    p = F.with_contract(F.suction_simple(tmp_path), adapters={}, **{
        'verification.accepted_evidence': ['sensor', 'sim_ground_truth'],
        'actuator.confirm_timeout_ms': 2000})
    saved = _gripper_block(save_project(p, tmp_path / 'p.yaml').read_text())['contract']
    assert saved == {'id': 'suction_simple', 'actuator': {'command_interface': 'set_bool'},
                     'verification': {'require_grasp_confirmation': False,
                                      'require_release_confirmation': False}}
    q = F.with_contract(F.composite(tmp_path / 'c'),
                        **{'verification.accepted_evidence': ['sim_ground_truth']})
    written = _gripper_block(save_project(q, tmp_path / 'q.yaml').read_text())['contract']
    assert written['verification']['accepted_evidence'] == ['sim_ground_truth']
    assert written['adapters'] == {'mock': 'trainit_mock'}
    assert 'goal_time_tolerance_s' not in written['actuators']['linear_axis']   # its default
    assert written['actuators']['binary_lock'] == {'command_interface': 'set_bool',
                                                   'actuation_feedback': True}
    assert load_project(tmp_path / 'q.yaml').robot.gripper.contract == q.robot.gripper.contract


# --- R3 'a duplicate': a contract id is a mapping KEY, so a duplicate is a duplicate YAML key --

def test_a_duplicate_key_inside_the_contract_is_refused_at_load(tmp_path):
    """Review TSA-9: ``yaml.safe_load`` keeps the LAST of two equal keys, so a copy-pasted
    actuator would silently replace the first one. Inside ``robot.gripper.contract`` that is an
    R3 refusal at load, naming the mapping, the key and both lines."""
    path = save_project(F.composite(tmp_path), tmp_path / 'project.yaml')
    text = path.read_text()
    head = '      actuators:\n'
    assert text.count(head) == 1
    copy = text.replace(head, head + '        gripper_b:\n          command_interface: set_bool\n',
                        1)
    path.write_text(copy)
    with pytest.raises(ValueError, match=r'R3\).*robot\.gripper\.contract\.actuators has the '
                                         r'key \'gripper_b\' twice'):
        load_project(path)
    # a target named twice, deeper inside the contract
    target = '            home:\n'
    deep = text.replace(target, target.replace('home', 'extended') + '            - 0.05\n' +
                        target, 1)
    assert deep != text
    path.write_text(deep)
    with pytest.raises(ValueError, match=r'actuators\.linear_axis\.targets has the key '
                                         r'\'extended\' twice'):
        load_project(path)


def test_a_legacy_project_with_a_duplicate_key_loads_exactly_as_before(tmp_path):
    """The duplicate check reads the contract block only: a legacy file -- which has none -- loads
    as it always did (YAML's last-wins), byte for byte the same project."""
    text = EXAMPLE.read_text()
    gripper = '  gripper:\n'
    assert text.count(gripper) == 1 and '    eef_name: suction\n' in text
    path = tmp_path / 'project.yaml'
    path.write_text(text.replace(gripper, gripper + '    eef_name: overridden\n', 1))
    p = load_project(path)
    assert p.robot.gripper.contract is None
    assert p.robot.gripper.eef_name == 'suction'                    # the last one, as ever
    assert json.loads(p.json()) == json.loads(load_project(EXAMPLE).json())


# --- the guard token, the one pre-validator, schema_version ------------------------------------

def _gripper(**kw) -> GripperSpec:
    return GripperSpec(kind='suction', eef_parent_link='tcp', command_joint='suctioncup_joint',
                       **kw)


class _ExcludeProbe(BaseModel):
    hidden: int = Field(default=0, exclude=True)


# Whether this pydantic honours Field(exclude=True) in a model's own JSON, asked of pydantic itself:
# Ubuntu 22.04's python3-pydantic 1.8.2 does not, and keeps the field there. What TSA SAVES never
# carries it on any pydantic: model/io.py drops it.
FIELD_EXCLUDE = 'hidden' not in json.loads(_ExcludeProbe().json())


def test_the_token_is_set_whenever_a_contract_is_present_and_overrides_a_legacy_one(tmp_path):
    project = F.suction_simple(tmp_path)
    for given in (None, 'trigger', 'joint_position', 'contract'):
        kw = {} if given is None else {'actuation': given}
        g = _gripper(contract=F.suction_contract(), **kw)
        assert g.actuation is EndEffectorActuation.CONTRACT, given
        assert g.actuation_overridden == (given if given in ('trigger', 'joint_position')
                                          else None)
        project.robot.gripper = g                       # never SAVED, on every pydantic
        path = save_project(project, tmp_path / f'{given or "none"}.yaml')
        saved = _gripper_block(path.read_text())
        assert 'actuation_overridden' not in saved, given
        if FIELD_EXCLUDE:                               # nor in the model's own JSON
            assert 'actuation_overridden' not in json.loads(g.json())
    # a canonical PARALLEL short form: never joint_position, whatever validator order would do
    jaw = GripperSpec(kind='parallel', contract=F.jaw_contract())
    assert jaw.actuation is EndEffectorActuation.CONTRACT
    # with no contract, ADR-0010's rule, unchanged
    assert GripperSpec(kind='parallel').actuation is EndEffectorActuation.JOINT_POSITION
    assert GripperSpec(kind='suction').actuation is EndEffectorActuation.TRIGGER


def test_there_is_one_pre_validator_on_the_gripper():
    """Two pre root validators run in definition order on pydantic 1.x and in reverse on 2.x: the
    token rules live in ONE (ADR-0022 D6), so their order cannot matter."""
    source = (PKG / 'trainit_setup_assistant' / 'model' / 'robot.py').read_text()
    start = source.index('class GripperSpec')
    body = source[start:source.index('\nclass ', start + 1)]
    import re
    assert len(re.findall(r'^\s+@root_validator\(pre=True\)', body, re.M)) == 1


def test_the_overridden_token_is_named_by_validate(tmp_path):
    p = F.suction_simple(tmp_path)
    data = F.dump(p.robot.gripper)
    data['actuation'] = 'trigger'
    data['contract'] = F.suction_contract()
    p.robot.gripper = GripperSpec(**data)
    warnings = E.eef_contract_warnings(p)
    assert any('actuation "trigger" was replaced by "contract"' in w for w in warnings)


def test_r19_at_load_the_token_or_schema_3_without_a_contract(tmp_path):
    p = load_project(EXAMPLE)
    data = json.loads(p.json())
    data['robot']['gripper']['actuation'] = 'contract'
    with pytest.raises(ValidationError, match='R19'):
        type(p).parse_obj(data)
    data = json.loads(p.json())
    data['schema_version'] = 3
    with pytest.raises(ValidationError, match='R19.*schema_version 3'):
        type(p).parse_obj(data)
    # a contract loads as schema 3 whatever number its file had
    data = json.loads(p.json())
    data['schema_version'] = 2
    data['robot']['gripper']['contract'] = F.suction_contract(adapters={
        'mock': 'trainit_mock', 'isaac': 'trainit_isaac_surface_gripper', 'real': 'none'})
    assert type(p).parse_obj(data).schema_version == 3


def test_an_unknown_key_or_token_inside_the_contract_is_refused_at_load():
    """R2: ``extra='forbid'`` inside the contract and closed tokens -- where every legacy model
    silently drops an unknown key."""
    for bad in ({'actuator': {'command_interface': 'set_bool', 'duration_s': 1.0}},
                {'actuator': {'command_interface': 'magic'}},
                {'actuator': {'command_interface': 'set_bool'}, 'adapterz': {}},
                {'actuator': {'command_interface': 'set_bool'},
                 'adapters': {'mock': 'trainit_composite'}},
                {'actuator': {'command_interface': 'set_bool'},
                 'verification': {'accepted_evidence': ['vibes']}},
                {'actuator': {'command_interface': 'set_bool'},
                 'adapters': {'mock': {'adapter': 'trainit_mock', 'stage_topic': 'x'}}}):
        with pytest.raises(ValidationError):
            _gripper(contract=bad)


def test_the_new_tokens_are_tokens_an_older_tsa_refuses():
    """The silent-downgrade guard (ADR-0022 D6): ``contract``, ``composite`` and ``actuate`` are
    NEW members of closed enums, so TSA 7.6.0 -- whose members these are -- refuses a canonical
    file at load instead of regenerating it as legacy (executed against a pristine 7.6.0 copy in
    the M2.6 forensics)."""
    tsa_760 = {EndEffectorActuation: ['trigger', 'joint_position'],
               GripperKind: ['suction', 'parallel', 'none'],
               ToolActionKind: ['grasp', 'release', 'attach', 'detach', 'reset_scene']}
    new = {EndEffectorActuation: 'contract', GripperKind: 'composite', ToolActionKind: 'actuate'}
    for enum, old in tsa_760.items():
        values = [m.value for m in enum]
        assert [v for v in values if v != new[enum]] == old
        assert new[enum] in values and new[enum] not in old


# --- a legacy project saves byte-identical -------------------------------------------------------

# load_project + save_project, as TSA 7.6.0 (351eb8c) itself wrote them (vNext M2.6 forensics,
# executed): 7.7.0 must write the same bytes.
SAVED_BY_760 = {
    'example': (EXAMPLE, 'fdbbe2aa9ed1949ee542fcb747e0a07c13f0eaffa27069bb15e99de3ec7400a8'),
    'golden': (GOLDEN_PROJECT,
               '0037051cfb7e3f25aefadd1d640214954a2da6ba83f098265d6b21af8778ec2d'),
    'test4': (TEST4_PROJECT, '62869f1c9ead28e3dc6e587ba0d11fc5b40259671afa8e3fb641f10727097a16'),
}


@pytest.mark.parametrize('name', sorted(SAVED_BY_760))
def test_a_legacy_project_saves_byte_identical_to_tsa_760(tmp_path, name):
    path, want = SAVED_BY_760[name]
    if not Path(path).is_file():
        F.skip_or_fail(f'{name} project not present: {path}')
    p = load_project(str(path))
    assert p.robot.gripper.contract is None
    out = save_project(p, tmp_path / 'p.yaml')
    text = out.read_text()
    assert 'contract' not in text and 'actuator:' not in text and '  site:' not in text
    assert hashlib.sha256(out.read_bytes()).hexdigest() == want


def test_the_new_tool_action_fields_are_written_only_when_set(tmp_path):
    p = load_project(EXAMPLE)
    p.application.tool_actions.append(ToolAction(at_waypoint='pick', kind=ToolActionKind.GRASP,
                                                 site='main'))
    actions = yaml.safe_load(save_project(p, tmp_path / 'p.yaml').read_text())[
        'application']['tool_actions']
    assert actions[-1] == {'at_waypoint': 'pick', 'kind': 'grasp', 'payload_ref': None,
                           'site': 'main'}
    assert all(set(a) == {'at_waypoint', 'kind', 'payload_ref'} for a in actions[:-1])


# --- the catalog: implementation rows ------------------------------------------------------------

def test_the_eef_adapter_rows_are_implementations_never_kinds():
    rows = {a['id']: a for a in catalog.adapters() if a['adapter_type'] == 'eef'}
    assert sorted(rows) == ['adapter.eef.none', 'adapter.eef.ros2_control',
                            'adapter.eef.trainit_isaac_surface_gripper', 'adapter.eef.trainit_mock']
    assert not [k for k in rows if k.split('.')[-1] in ('suction', 'parallel', 'composite')]
    assert [m.value for m in M.EndEffectorAdapter] == list(catalog.EEF_ADAPTER_ROWS)
    isaac = rows['adapter.eef.trainit_isaac_surface_gripper']
    assert (isaac['modes'], isaac['serves'], isaac['kinds']) == (['isaac'], ['set_bool'],
                                                                 ['suction'])
    assert 'grasp_detection' not in isaac['provides']           # only with the status topic
    assert isaac['provides_if'] == {'stage_status_topic': ['grasp_detection',
                                                           'release_detection']}
    assert rows['adapter.eef.ros2_control']['evidence_classes'] == []
    assert rows['adapter.eef.none']['serves'] == []


def test_the_sim_grasp_rows_are_on_the_one_backend_each_exists_on():
    rows = {a['id']: a['modes'] for a in catalog.adapters()
            if a['adapter_type'] == 'eef_simulation'}
    assert rows == {'adapter.sim_grasp.surface_gripper': ['isaac'],
                    'adapter.sim_grasp.link_attacher': ['gazebo']}


def test_the_legacy_bridge_marks_are_catalog_data():
    """R18 identifies a legacy end-effector bridge by catalog DATA (package + executable + role),
    never by a literal in the code that refuses it."""
    roles = {(r['package'], r['executable']): r['role'] for r in catalog.EEF_BRIDGE_ROLES}
    assert roles[('fairino_gripper', 'gripper_suction_action_server')] == 'eef_legacy_bridge'
    assert roles[('fairino_gripper', 'joint_state_merger')] == 'gripper_state_merger'
    source = (PKG / 'trainit_setup_assistant' / 'generator' / 'eef_contract.py').read_text()
    for package, executable in roles:
        assert executable not in source, executable


def test_every_legacy_identity_project_loads_with_no_new_refusal():
    """F6 (in the suite): every project of the generation identity proof is LEGACY and meets no
    end-effector refusal -- the contract is opt-in. A case whose inputs are not on this checkout
    is COUNTED, never passed over in silence: the test then skips naming how many it could not
    check (and fails with TRAINIT_REQUIRE_GOLDEN=1)."""
    import identity_cases
    checked, missing = [], {}
    for name, build in identity_cases.CASES.items():
        why = identity_cases.missing(name)
        if why:
            missing[name] = why
            continue
        p = build()
        assert p.robot.gripper.contract is None, name
        assert E.eef_contract_problems(p) == [], name
        assert E.eef_contract_warnings(p) == [], name
        checked.append(name)
    assert len(checked) + len(missing) == len(identity_cases.CASES) > 0
    if missing:
        first = next(iter(missing))
        F.skip_or_fail(f'{len(checked)} of {len(identity_cases.CASES)} identity projects checked; '
                       f'{len(missing)} cannot be built here (first: {first}: {missing[first]})')
