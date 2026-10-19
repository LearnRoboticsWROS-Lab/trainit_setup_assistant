"""TSA's planner vocabulary comes from TMR's registry, through TSA's committed copy (vNext M2.1).

ADR-0021: TMR registers the planner vocabulary once (``trainit_motion_runtime/config/
planner_profiles.yaml``); TSA commits a BYTE copy of it (``catalog/tmr_planner_registry.yaml``,
written by ``python3 -m trainit_setup_assistant.catalog.refresh``) and builds from the copy --
never from the installed TMR. These tests hold the copy to TMR, every TSA consumer to the copy,
prove that a changed registry is noticed with a message that says what to do, and that no
candidate provider can become selectable. The pre-M2.1 values themselves are pinned in
test_planner_vocabulary_pins.py.
"""
import ast
import hashlib
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest

from trainit_setup_assistant import catalog
from trainit_setup_assistant.applications import get_application
from trainit_setup_assistant.applications.pick_and_place import MOVE_NODES, PickAndPlace
from trainit_setup_assistant.catalog import (REFRESH, REGISTRY_FILE, TSA_REQUIRED_PLANNER_MODES,
                                             declared_planner_modes, missing_required_modes,
                                             offered_planner_modes, parse_planner_registry,
                                             tmr_planner_registry)
from trainit_setup_assistant.catalog import refresh as refresh_tool
from trainit_setup_assistant.model import load_project
from trainit_setup_assistant.model import enums as E
from trainit_setup_assistant.model.robot import OmplGroupSpec
from trainit_setup_assistant.model.task import ApplicationSpec

import catalog_identity
import identity_cases
import ws_paths

PKG = Path(__file__).resolve().parents[1]
CATALOG_DIR = PKG / 'trainit_setup_assistant' / 'catalog'
COPY = CATALOG_DIR / REGISTRY_FILE
TEMPLATES = PKG / 'trainit_setup_assistant' / 'generator' / 'templates'

# ADR-0019 Decision 7: MoveIt's in-tree planners are Community. Only these plugins may back a
# registry pipeline; anything else (a candidate, a Pro plugin) enters through an extension point.
COMMUNITY_MOVEIT_PLANNERS = {'moveit_planners_ompl', 'pilz_industrial_motion_planner',
                             'moveit_planners_chomp', 'moveit_planners_stomp'}


def _tmr_registry_files():
    """TMR's registry where this checkout can see it: the sibling source and the installed copy."""
    found = {}
    src = ws_paths.ws_src('trainit_motion_runtime', 'config', 'planner_profiles.yaml')
    if Path(src).is_file():
        found['source'] = Path(src)
    try:
        from ament_index_python.packages import get_package_share_directory
        installed = Path(get_package_share_directory('trainit_motion_runtime')) / 'config' / \
            'planner_profiles.yaml'
        if installed.is_file():
            found['installed'] = installed
    except Exception:                                           # noqa: BLE001 - not sourced
        pass
    return found


def copy_problems(copy_bytes: bytes, tmr_bytes: bytes, where: str):
    """Why TSA's copy is not TMR's registry, as an instruction. Empty = it is."""
    if copy_bytes == tmr_bytes:
        return []
    return [f"catalog/{REGISTRY_FILE} differs from TMR's {where} planner registry: TSA would offer "
            f"a stale vocabulary. Rebuild TMR if you edited its source, then run `{REFRESH}`."]


def execution_problems(registry: dict):
    """Where TSA's execution axis and the registry's disagree, as an instruction."""
    out = []
    tsa = [m.value for m in E.ExecutionMode]
    tmr = list(registry['execution'])
    for token in tmr:
        if token not in tsa:
            out.append(f"TMR's registry declares execution '{token}', which TSA's ExecutionMode "
                       'does not have. A learned policy is never such an entry (ADR-0021 D6: it '
                       'is RunPolicy); a MoveIt-realised mode needs the member AND its BT node in '
                       'MOVE_NODES (and its validation), or a move in that mode cannot be generated')
    for token in tsa:
        if token not in tmr:
            out.append(f"TSA's ExecutionMode '{token}' is not in TMR's registry")
    for member in E.ExecutionMode:
        node = registry['execution'].get(member.value, {}).get('node')
        if node is not None and MOVE_NODES.get(member) != node:
            out.append(f"execution '{member.value}': TSA emits <{MOVE_NODES.get(member)}>, TMR's "
                       f'registry says <{node}>')
    return out


def candidate_tokens():
    """Exact tokens of every candidate provider (ids, packages), and the product words of the
    motion-planning ones (cumotion, curobo) -- the deny-list."""
    exact, words = set(), set()
    for c in catalog.candidates():
        exact.add(c['id'].lower())
        exact.update(p.lower() for p in c.get('packages', []))
        if c['family'] == 'motion_planning':
            words.add(c['id'].split('.', 1)[1].split('_')[0].lower())
    return exact, words


def candidate_problems(registry: dict):
    """How a registry would let a candidate (or a non-Community plugin) in. Empty = it cannot."""
    exact, words = candidate_tokens()
    out = []
    for name, spec in registry['pipelines'].items():
        if spec['plugin_package'] not in COMMUNITY_MOVEIT_PLANNERS:
            out.append(f"pipeline '{name}': plugin {spec['plugin_package']} is not a Community "
                       'MoveIt planner (ADR-0019 D7); a candidate or Pro planner registers through '
                       'an extension point, never in the Community registry (ADR-0021 D7)')
    keys = list(registry['pipelines']) + list(registry['planner_modes']) + \
        [s['plugin_package'] for s in registry['pipelines'].values()]
    for key in keys:
        k = key.lower()
        if k in exact or any(w in k for w in words):
            out.append(f"'{key}' is a candidate provider's name: candidates are metadata, never "
                       'registry entries (ADR-0019)')
    return out


# --- the copy is TMR's registry -------------------------------------------------------------------

def test_the_copy_is_tmrs_registry_byte_for_byte():
    found = _tmr_registry_files()
    if not found:
        pytest.skip('TMR is neither a sibling checkout nor installed: run in the workspace')
    problems = []
    for where, path in found.items():
        problems += copy_problems(COPY.read_bytes(), path.read_bytes(), where)
    assert problems == [], '\n'.join(problems)


def test_the_snapshot_records_the_copy_it_was_made_with():
    rec = catalog.snapshot()['registries']['trainit_motion_runtime']['planner_registry']
    assert rec == dict(file=REGISTRY_FILE, sha256=hashlib.sha256(COPY.read_bytes()).hexdigest(),
                       schema_version=catalog.PLANNER_REGISTRY_SCHEMA)


# --- every TSA consumer is the copy ------------------------------------------------------------

def test_planner_id_is_what_the_copy_offers():
    offered = offered_planner_modes(tmr_planner_registry())
    assert [m.value for m in E.PlannerId] == offered
    assert catalog.snapshot()['registries']['tsa_selectable']['planner'] == offered  # a fixed point


def test_the_bundle_texts_list_the_copys_modes():
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    env = build_jinja_env()
    assert env.globals['tsa_planner_modes'] == ' | '.join(m.value for m in E.PlannerId)
    assert env.globals['tmr_planner_modes'] == ' | '.join(
        declared_planner_modes(tmr_planner_registry()))
    for template, var in (('app/bringup.launch.py.j2', 'tsa_planner_modes'),
                          ('app/README.md.j2', 'tsa_planner_modes'),
                          ('app/trainit_bt.launch.py.j2', 'tmr_planner_modes')):
        text = (TEMPLATES / template).read_text()
        assert '{{ ' + var + ' }}' in text, template
        assert 'ompl_chomp' not in text, f'{template} types a planner list again'


def test_execution_modes_and_their_nodes_are_the_registrys():
    registry = tmr_planner_registry()
    assert execution_problems(registry) == []
    for member in E.ExecutionMode:
        assert PickAndPlace._move_node(types.SimpleNamespace(execution=member)) == \
            registry['execution'][member.value]['node']
    assert PickAndPlace._move_node(None) == registry['execution']['plan']['node']


def test_an_execution_mode_tsa_cannot_emit_is_refused_not_planned():
    """A third execution value (a learned policy, ADR-0005) must never fall back to a planned
    <MoveWaypoint>: the closed map refuses it, and the parity check says what to add."""
    policy = E.StrEnum('Fake', [('POLICY', 'policy')]).POLICY
    with pytest.raises(ValueError, match="execution 'policy' has no BT node in TSA"):
        PickAndPlace._move_node(types.SimpleNamespace(execution=policy))
    registry = tmr_planner_registry()
    registry['execution']['policy'] = dict(planned=False, node='RunPolicy')
    (msg,) = execution_problems(registry)
    assert "declares execution 'policy'" in msg and 'MOVE_NODES' in msg


def test_the_model_default_is_the_registrys():
    """TSA's own schema (what a project file without the key means), held to TMR's default."""
    assert ApplicationSpec().global_planner_mode.value == \
        tmr_planner_registry()['default_planner_mode'], (
            "TMR changed its default planner mode: TSA's schema default (what a project file "
            'without global_planner_mode means, model/task.py) is PlannerId.PILZ -- change it '
            'deliberately, it changes what old projects generate')


def test_the_ompl_planner_tsa_configures_is_the_one_tmr_requests():
    requested = tmr_planner_registry()['planner_modes']['ompl']['planner']
    spec = OmplGroupSpec()
    assert spec.default_planner_config == requested and requested in spec.planner_configs


@pytest.mark.parametrize('template', ['moveit_config/package.xml.j2',
                                      'moveit_config/scene_loader_package.xml.j2'])
def test_a_bundle_depends_on_every_loaded_planner_plugin_and_no_other(template):
    deps = set(re.findall(r'<exec_depend>([^<]+)</exec_depend>', (TEMPLATES / template).read_text()))
    pipelines = tmr_planner_registry()['pipelines'].values()
    loaded = {s['plugin_package'] for s in pipelines if s['loaded']}
    not_loaded = {s['plugin_package'] for s in pipelines if not s['loaded']}
    assert loaded <= deps, f'{template} misses {loaded - deps}'
    assert not (not_loaded & deps), f'{template} depends on unloaded {not_loaded & deps}'


# D-016: Pilz PTP is a blind joint interpolation, so a ptp move into a vision goal is flagged when
# it resolves to pilz. Every OFFERED mode must have an answer here: a new one forces the decision.
D016_FLAGS_PTP_INTO_A_VISION_GOAL = {'pilz': True, 'ompl': False, 'ompl_chomp': False}


@pytest.mark.skipif(not ws_paths.GOLDEN_BUNDLE.is_dir(), reason='needs the golden project')
@pytest.mark.parametrize('mode', list(E.PlannerId), ids=lambda m: m.value)
def test_the_vision_rule_has_an_answer_for_every_offered_mode(mode):
    assert mode.value in D016_FLAGS_PTP_INTO_A_VISION_GOAL, (
        f"'{mode.value}' is a newly offered planner mode: decide whether a ptp move into a vision "
        'goal with it is blind (D-016) and add it to D016_FLAGS_PTP_INTO_A_VISION_GOAL')
    expected = D016_FLAGS_PTP_INTO_A_VISION_GOAL[mode.value]
    p = load_project(str(ws_paths.GOLDEN_BUNDLE / 'project.yaml'))
    seg = p.application.segment_for('pick')
    seg.motion, seg.planner = E.MotionType.PTP, mode
    problems = get_application(p.application.type).validate(p)
    assert any('ptp/pilz' in x for x in problems) is expected


# --- the offer rule, on registries that changed ------------------------------------------------

def _with(registry, **modes):
    registry['planner_modes'].update(modes)
    return registry


def test_what_the_rule_offers_when_the_registry_changes():
    r = tmr_planner_registry
    assert offered_planner_modes(r()) == ['pilz', 'ompl', 'ompl_chomp']
    # a new stable mode on a loaded pipeline is offered, at its place in the registry
    assert offered_planner_modes(_with(r(), rrtstar=dict(
        pipeline='ompl', planner='RRTstarkConfigDefault', stability='stable')))[-1] == 'rrtstar'
    # an experimental one is not, loaded or not
    assert 'trial' not in offered_planner_modes(_with(r(), trial=dict(
        pipeline='ompl', planner='PRMkConfigDefault', stability='experimental')))
    # a mode whose optimizer pipeline is not loaded degrades at runtime, so it is not offered
    reg = r()
    reg['pipelines']['chomp']['loaded'] = False
    assert offered_planner_modes(reg) == ['pilz', 'ompl']
    # stomp, made loaded, stays out while it is experimental
    reg = r()
    reg['pipelines']['stomp']['loaded'] = True
    assert 'stomp' not in offered_planner_modes(reg)
    # a mode whose own pipeline is not loaded is not offered (every OMPL mode needs ompl)
    reg = r()
    reg['pipelines']['ompl']['loaded'] = False
    assert offered_planner_modes(reg) == ['pilz']
    # stomp made stable is still not offered while its pipeline is not loaded (ADR-0021 D5)
    reg = r()
    reg['planner_modes']['stomp']['stability'] = 'stable'
    assert 'stomp' not in offered_planner_modes(reg)
    # an explicit null optimizer means no optimizer, as TMR's checker and derive() read it
    reg = r()
    reg['planner_modes']['ompl']['optimizer_pipeline'] = None
    assert offered_planner_modes(reg) == ['pilz', 'ompl', 'ompl_chomp']
    # a removed mode disappears from the offer -- and TSA's code names it, so it is required
    reg = r()
    del reg['planner_modes']['pilz']
    assert 'pilz' not in offered_planner_modes(reg)
    assert missing_required_modes(reg) == ['pilz']
    assert missing_required_modes(r()) == [] and set(TSA_REQUIRED_PLANNER_MODES) <= {
        m.value for m in E.PlannerId}


# --- no candidate can become selectable (vNext M2 T2) -------------------------------------------

def test_the_registry_admits_no_candidate():
    assert candidate_problems(tmr_planner_registry()) == []
    reg = tmr_planner_registry()
    reg['pipelines']['cumotion'] = dict(plugin_package='isaac_ros_cumotion_moveit',
                                        parameters='cumotion_planning.yaml', loaded=True)
    reg['planner_modes']['cumotion'] = dict(pipeline='cumotion', planner='cuMotion',
                                            stability='stable')
    problems = candidate_problems(reg)
    assert any('not a Community MoveIt planner' in p for p in problems), problems
    assert any("'cumotion' is a candidate provider's name" in p for p in problems), problems


def test_no_selectable_vocabulary_holds_a_candidate():
    exact, words = candidate_tokens()
    assert {'cumotion', 'curobo'} <= words
    assert {'isaac_ros_cumotion_moveit', 'curobo_core', 'cumotion_vendor'} <= exact
    for vocab in (E.PlannerId, E.ExecutionMode, E.DetectionMethod, E.CandidateRule, E.PolicyMode,
                  E.Backend):
        for m in vocab:
            v = m.value.lower()
            assert v not in exact and not any(w in v for w in words), (vocab.__name__, v)


def test_every_moveit_pipeline_candidate_names_its_packages():
    for c in catalog.candidates():
        if c['integration'] == 'moveit_pipeline':
            assert c.get('packages'), f"{c['id']} must list its packages, so the guard is exact"


# --- generation reads the copy, and only modules that read no environment -------------------------

def test_generation_reads_the_copy(tmp_path):
    """The positive half of test_catalog's isolation rule: the vocabulary IS read from the copy."""
    identity_cases.require('p2_example')
    code = (
        'import sys\n'
        'opened = []\n'
        'sys.addaudithook(lambda e, a: opened.append(str(a[0])) if e == "open" and a else None)\n'
        f'sys.path.insert(0, {str(PKG / "test")!r})\n'
        'import conftest  # noqa: F401\n'
        'from trainit_setup_assistant.generator import Orchestrator\n'
        'from trainit_setup_assistant.model import load_project\n'
        f'Orchestrator().generate(load_project({str(PKG / "examples" / "fr3wml_project.yaml")!r}),'
        f' {str(tmp_path / "out")!r})\n'
        f'print("COPY_READ" if any(f.endswith({REGISTRY_FILE!r}) for f in opened) else "NO_COPY")\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                       cwd=str(ws_paths.WS_SRC))
    assert 'COPY_READ' in r.stdout, r.stdout + r.stderr


def test_the_catalog_is_a_leaf_and_the_vocabulary_reads_no_environment():
    tree = ast.parse((CATALOG_DIR / '__init__.py').read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert node.level == 0 and not (node.module or '').startswith('trainit_setup_assistant'), \
                'catalog/__init__.py must import nothing from TSA (the model imports it)'
    for path in (CATALOG_DIR / '__init__.py', PKG / 'trainit_setup_assistant' / 'model' / 'enums.py',
                 PKG / 'trainit_setup_assistant' / 'generator' / 'determinism.py'):
        assert not re.search(r'\bos\.environ\b|\bgetenv\b', path.read_text()), path.name


def test_one_refresh_is_a_fixed_point(monkeypatch):
    if not _tmr_registry_files().get('installed'):
        pytest.skip('needs the installed workspace')
    pytest.importorskip('trainit_perception.ranking')
    from trainit_setup_assistant.catalog import refresh
    # A package snapshot.yaml records facts of may not be installed here (the private policy
    # runtime; MoveIt Servo, outside the Community closure). derive() runs in the refresh's CHILD
    # interpreter, out of a monkeypatch's reach, so the record the child returns takes exactly
    # those values as committed: every other byte of the three files is still compared.
    absent = catalog_identity.absent_packages()
    if absent:
        derive_record, committed = refresh.derive_record, catalog.snapshot()
        monkeypatch.setattr(refresh, 'derive_record', lambda **inputs: (
            catalog_identity.as_recorded(derive_record(**inputs), committed, absent)))
    assert refresh.main(['--check']) == 0
    catalog_identity.require_installed(absent)


# --- one reader, one refusal: a registry of another schema --------------------------------------

def test_a_registry_of_another_schema_is_refused_by_every_tsa_reader():
    newer = COPY.read_bytes().replace(b'schema_version: 2', b'schema_version: 3')
    assert newer != COPY.read_bytes()
    with pytest.raises(RuntimeError, match=r"schema_version 3, and this TSA reads schemas "
                                           r"\[1, 2\].*readers updated first"):
        parse_planner_registry(newer, 'the copy')        # catalog/__init__.py AND derive.py


def test_a_copy_that_drops_a_required_mode_stops_the_model_with_an_instruction(tmp_path):
    code = (
        'import trainit_setup_assistant.catalog as c\n'
        'reg = c.tmr_planner_registry()\n'
        "reg['planner_modes']['pilz']['stability'] = 'experimental'\n"
        'c._registry_copy = lambda: reg\n'
        'import trainit_setup_assistant.model\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "no longer offers ['pilz']" in r.stderr and 'change TSA first' in r.stderr, r.stderr


# --- the refresh tool: validates, then writes; --check writes nothing ---------------------------

@pytest.fixture
def scratch_catalog(tmp_path, monkeypatch):
    """refresh_tool pointed at a scratch catalog dir and a scratch 'installed' TMR registry.

    vNext M2.2 (ADR-0021 amendment M2.2-5): the same refresh now also writes TP's vocabulary
    snapshot, from ``tp_export``, and hands ``derive_record`` the inputs it validated -- so the
    scratch catalog holds TSA's committed TP snapshot, TP's export answers with it (current, its
    installed release the one it names) and the fake record accepts those inputs. Every assertion
    below is M2.1's, unchanged."""
    here = tmp_path / 'catalog'
    here.mkdir()
    (here / REGISTRY_FILE).write_bytes(COPY.read_bytes())
    (here / catalog.TP_REGISTRY_FILE).write_bytes(
        (CATALOG_DIR / catalog.TP_REGISTRY_FILE).read_bytes())
    tmr = tmp_path / 'planner_profiles.yaml'
    tmr.write_bytes(COPY.read_bytes())
    record = dict(fake=True, registries=dict(trainit_perception=dict(version='0')))
    (here / refresh_tool.SNAPSHOT_FILE).write_text(refresh_tool.snapshot_text(record))
    monkeypatch.setattr(refresh_tool, 'catalog_dir', lambda: here)
    monkeypatch.setattr(refresh_tool, 'tmr_registry_path', lambda: tmr)
    monkeypatch.setattr(refresh_tool, 'derive_record', lambda **inputs: record)
    monkeypatch.setattr(refresh_tool, 'tp_available', lambda: True)
    monkeypatch.setattr(refresh_tool, 'tp_export', catalog.tp_registry)
    monkeypatch.setattr(refresh_tool, 'tp_installed_release',
                        lambda: catalog.tp_registry()['trainit_perception'])
    files = lambda: {f.name: f.read_bytes() for f in here.iterdir()}  # noqa: E731
    return types.SimpleNamespace(here=here, tmr=tmr, files=files)


def test_refresh_check_on_a_current_catalog_writes_nothing(scratch_catalog):
    before = scratch_catalog.files()
    assert refresh_tool.main(['--check']) == 0
    assert scratch_catalog.files() == before


def test_refresh_reports_a_stale_copy_and_then_writes_it_byte_for_byte(scratch_catalog, capsys):
    scratch_catalog.tmr.write_bytes(COPY.read_bytes() + b'# TMR moved\n')
    before = scratch_catalog.files()
    assert refresh_tool.main(['--check']) == 1
    assert f'catalog DRIFT: {REGISTRY_FILE}' in capsys.readouterr().out
    assert scratch_catalog.files() == before                         # --check wrote nothing
    assert refresh_tool.main([]) == 0
    assert (scratch_catalog.here / REGISTRY_FILE).read_bytes() == scratch_catalog.tmr.read_bytes()


@pytest.mark.parametrize('edit, why', [
    (lambda b: b.replace(b'schema_version: 2', b'schema_version: 3'), 'readers updated first'),
    (lambda b: b.replace(b'stability: stable', b'stability: experimental', 1),   # pilz: first
     "no longer offers ['pilz']"),
    (lambda b: b.replace(b'reserved: [orientation_constraints, jerk_constraints, ',
                         b'reserved: [orientation_constraints, '),
     "no longer declares [\"capability 'jerk_constraints'\"]"),
], ids=['another_schema', 'drops_a_required_mode', 'drops_a_capability_tsa_names'])
def test_refresh_refuses_a_registry_this_tsa_cannot_read_and_writes_nothing(
        scratch_catalog, capsys, edit, why):
    registry = edit(COPY.read_bytes())
    assert registry != COPY.read_bytes()
    scratch_catalog.tmr.write_bytes(registry)
    before = scratch_catalog.files()
    assert refresh_tool.main([]) == 2
    assert why in capsys.readouterr().err
    assert scratch_catalog.files() == before


def test_refresh_refuses_to_write_a_snapshot_without_tp(scratch_catalog, monkeypatch, capsys):
    monkeypatch.setattr(refresh_tool, 'tp_available', lambda: False)
    scratch_catalog.tmr.write_bytes(COPY.read_bytes() + b'# TMR moved\n')
    before = scratch_catalog.files()
    assert refresh_tool.main([]) == 2
    assert 'trainit_perception is not importable' in capsys.readouterr().err
    assert scratch_catalog.files() == before


# --- the GUI's own vision rule, unchanged (it names pilz and ompl: TSA_REQUIRED_PLANNER_MODES) -----

@pytest.fixture(scope='module')
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_binding_a_camera_to_a_ptp_pilz_block_switches_it_to_free_ompl(qapp):
    """wizard.py's auto-switch (D-016): a move a camera FIRST takes over, resolving to ptp/pilz,
    becomes free/OMPL -- unchanged by M2.1, pinned because it names two planner tokens."""
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    from trainit_setup_assistant.model import DetectorSpec
    from trainit_setup_assistant.model.perception import PerceptionSpec

    ctrl = AssistantController()
    ctrl.open_project(str(PKG / 'examples' / 'fr3wml_project.yaml'))
    ctrl.clear_application()
    ctrl.project.perception = PerceptionSpec(detectors=[DetectorSpec(name='cube')])
    wiz = SetupWizard(ctrl)
    bp = wiz.blocks_page
    bp.add_block('move')
    bp.m_name.setText('pick')
    bp.m_target.setCurrentIndex(1)                      # tcp
    bp.m_pos.setText('0.5, 0.0, 0.1')
    bp.m_quat.setText('0, 1, 0, 0')
    bp.m_motion.setCurrentText('ptp')
    bp.m_planner.setCurrentText('pilz')
    bp.apply_inspector()
    bp.m_detector.setCurrentText('cube')                # the camera takes the move over
    bp.apply_inspector()
    block = next(b for b in bp.blocks if b.get('name') == 'pick')
    assert (block['motion'], block['planner']) == ('free', 'ompl')
