"""P0.a (ADR-0032 Decision 3, TSA 8.9.0): TASK EXPORT -- a task the cell's PLC may request through the Robot Task
Interface. The frozen P0.a matrix's TSA cases (overlay docs/engineering/P0_CHECKPOINT.md):

  X1  two tasks with the same export number         refused, naming both tasks
  X2  a number outside 1..65535                     refused
  X3  timeout_s missing, <= 0, above 86400          refused; no default exists (and a millisecond resolution)
  X4  exporting a task that has inputs              refused
  X5  the export table id                           TSA's equals the C++ rule's (TMR rti_contract); a fixture pins it
  B1  with and without one task's export            only the documented additions differ; every task_<name> tree
                                                    byte-identical; the file diff recorded
  B3  an export-free project                        the files of TSA 8.8.0 (no job tree, no server launch, the 8.8.0
                                                    tasks.yaml header); the golden byte gates pin the rest

The fixture test/data/p0a/served.yaml (on the M2.5 test arm) is generated in a temporary directory; its tree,
tasks.yaml, bt_params and motion profiles are the BYTE COPIES kept in test/data/p0a/served/ here and in TMR
test/data/p0a/served/ (TMR's test_p0a_task_server runs them): sha256-pinned on both sides. A change here is a TSA
generation change: regenerate, copy to both repositories, re-pin both sides, and say why.
"""

import copy
import hashlib
import os
import zlib
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.determinism import export_table_id, export_table_text
from trainit_setup_assistant.generator.program_files import TASKS_HEADER
from trainit_setup_assistant.model import CanonicalProject, TaskExport, load_project

DATA = Path(__file__).parent / 'data' / 'p0a'
COPIES = DATA / 'served'
TMR_COPIES = Path(__file__).resolve().parents[2] / 'trainit_motion_runtime' / 'test' / 'data' / 'p0a' / 'served'
APP = 'p0a_app'
PINNED = {   # the digests TMR's test_p0a_task_server.cpp pins too
    'task_program.xml': 'f864d0e3b046dec1c4be8f700b7d3c80e8e8db0d977b8e0623ee1c2b57247c1b',
    'tasks.yaml': '6f02aa4347736cfc0be8ed8a3de16d09bdfd004d3e6df63696e98fd7274e4e5e',
    'bt_params.yaml': 'ee5f12cc8fe8988836d9e3dd909f1687a928907694bf2c059257d13b6b3d9716',
    'motion_profiles.yaml': '8b0b793ffc1d549259474fcbfa120cea3b19b53ec1abc9a237330f074c803750',
}
# X5: the fixture's table and its id (zlib.crc32; TMR's crc32 is pinned to zlib's check value "123456789" ->
# 0xCBF43926 in test_p0a_rti_state_machine)
FIXTURE_TABLE_TEXT = '3:pick_part:10000\n4:carry:10000\n7:home:5000\n12:place_part:30000\n'
FIXTURE_TABLE_ID = 3767158916
# review (architecture, finding 8): more vectors of the SAME rule, pinned identically in TMR's
# test_p0a_rti_state_machine (RtiContract.TheSharedVectorsAreTheSameOnBothSides, with the fixture's table too)
SHARED_VECTORS = [
    ([(1, 'a', 1)], '1:a:1\n', 429640448),
    ([(65535, 'z', 86400000), (300, 'm', 2500)], '300:m:2500\n65535:z:86400000\n', 3443090808),
    ([(2, 'job_b', 1500), (1, 'job_a', 1)], '1:job_a:1\n2:job_b:1500\n', 2330640754),
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw() -> dict:
    return yaml.safe_load((DATA / 'served.yaml').read_text())


def _project(raw: dict) -> CanonicalProject:
    old = os.getcwd()
    os.chdir(DATA)                                       # the urdf path is relative to it
    try:
        return CanonicalProject.parse_obj(raw)
    finally:
        os.chdir(old)


def _generate(raw: dict, out: Path) -> Path:
    old = os.getcwd()
    os.chdir(DATA)
    try:
        Orchestrator().generate(CanonicalProject.parse_obj(raw), out)
    finally:
        os.chdir(old)
    return out / APP


def _task(raw: dict, name: str) -> dict:
    return next(t for t in raw['application']['tasks'] if t['name'] == name)


def _refusal(raw: dict) -> str:
    with pytest.raises(Exception) as exc:
        _project(raw)
    return str(exc.value)


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    return _generate(_raw(), tmp_path_factory.mktemp('p0a'))


# --- the shared fixture ----------------------------------------------------------------------------------------------

def test_the_copies_are_the_pinned_ones():
    for name, digest in PINNED.items():
        assert _sha(COPIES / name) == digest, name


def test_tsa_generates_exactly_the_copies_tmr_runs(generated):
    pairs = {'task_program.xml': generated / 'bt_trees' / 'task_program.xml',
             'tasks.yaml': generated / 'config' / 'tasks.yaml',
             'bt_params.yaml': generated / 'config' / 'bt_params.yaml',
             'motion_profiles.yaml': generated / 'config' / 'motion_profiles.yaml'}
    for name, path in pairs.items():
        assert path.read_bytes() == (COPIES / name).read_bytes(), name


def test_tmrs_copies_are_the_same_bytes():
    if not TMR_COPIES.is_dir():
        pytest.skip('trainit_motion_runtime is not beside this repository')
    for name in PINNED:
        assert (TMR_COPIES / name).read_bytes() == (COPIES / name).read_bytes(), name


def test_an_exported_bundle_carries_its_job_trees_table_and_server_launch(generated):
    tree = (generated / 'bt_trees' / 'task_program.xml').read_text()
    for task in ('home', 'place_part', 'pick_part', 'carry'):
        assert f'<BehaviorTree ID="job_{task}">' in tree
    assert 'job_inspect' not in tree                       # internal: not exported
    # a job tree never resets the held state and has no start check (the server carries it, ADR-0032 Decision 8)
    jobs = tree[tree.index('<BehaviorTree ID="job_home">'):]
    assert 'trainit__holding := false' not in jobs
    assert 'trainit__holding := false' in tree[:tree.index('<BehaviorTree ID="task_place_part">')]   # MainTree
    tasks = yaml.safe_load((generated / 'config' / 'tasks.yaml').read_text())
    assert tasks['export_table_id'] == FIXTURE_TABLE_ID
    assert tasks['tasks']['place_part']['export'] == {'number': 12, 'timeout_s': 30.0}
    assert tasks['tasks']['home']['export'] == {'number': 7, 'timeout_s': 5.0}
    assert tasks['tasks']['pick_part']['export'] == {'number': 3, 'timeout_s': 10.0}
    assert tasks['tasks']['carry']['export'] == {'number': 4, 'timeout_s': 10.0}
    assert 'export' not in tasks['tasks']['inspect']
    launch = (generated / 'launch' / 'trainit_task_server.launch.py').read_text()
    assert 'executable="trainit_task_server"' in launch and '"tasks_file": tasks' in launch
    assert 'executable="trainit_run_bt"' not in launch
    assert 'end_with_tree' not in launch                   # a server never ends by itself
    package = (generated / 'package.xml').read_text()
    assert '<exec_depend version_gte="2.10.0">trainit_motion_runtime</exec_depend>' in package


# --- X1 - X5 ---------------------------------------------------------------------------------------------------------

def test_x1_two_tasks_with_one_number_are_refused_naming_both():
    raw = _raw()
    _task(raw, 'home')['export']['number'] = 12
    text = _refusal(raw)
    assert 'R-P0-03' in text and "'place_part'" in text and "'home'" in text


@pytest.mark.parametrize('number', [0, -1, 65536, 1.5, '12', True])
def test_x2_a_number_outside_1_65535_is_refused(number):
    raw = _raw()
    _task(raw, 'home')['export']['number'] = number
    assert 'R-P0-01' in _refusal(raw)


@pytest.mark.parametrize('timeout', [0, -1, 86400.001, 86401, float('nan'), float('inf'), 0.0005, '30', None,
                                     1e-10, 10 ** 400])   # review: sub-millisecond -> 0 ms; a huge int (no overflow)
def test_x3_a_timeout_missing_non_positive_too_long_or_finer_than_a_millisecond_is_refused(timeout):
    raw = _raw()
    if timeout is None:
        del _task(raw, 'home')['export']['timeout_s']    # missing: no default exists
        text = _refusal(raw)
        assert 'timeout_s' in text and 'mandatory' in text   # named with what it is for, not "Field required"
        return
    _task(raw, 'home')['export']['timeout_s'] = timeout
    assert 'R-P0-02' in _refusal(raw)


def test_x3_the_bounds_themselves_are_accepted():
    for timeout in (0.001, 86400, 2.5):
        raw = _raw()
        _task(raw, 'home')['export']['timeout_s'] = timeout
        home = next(t for t in _project(raw).application.program.exported() if t.name == 'home')
        assert home.export.timeout_s == float(timeout)


def test_an_exported_program_that_does_not_track_the_held_state_is_warned():
    """P0.a review (architecture): the robot task server can report a cancelled job CANCELLED (200) only when the
    program tracks the held state; without it every CANCEL from the PLC fails closed (309, ADR-0032 Decision 8) -- the
    owner is told at plan time. The served fixture tracks it (faults: stop, pick_part grips): no warning."""
    from trainit_setup_assistant.applications.task_program import plan_program

    def warned(raw):
        old = os.getcwd()
        os.chdir(DATA)
        try:
            return [w for w in plan_program(CanonicalProject.parse_obj(raw)).warnings if '(309)' in w]
        finally:
            os.chdir(old)

    assert warned(_raw()) == []
    raw = _raw()
    del raw['application']['faults']                     # no fault policy: the held state is not tracked
    w = warned(raw)
    assert len(w) == 1, w
    assert all(f"'{name}'" in w[0] for name in ('place_part', 'home', 'pick_part', 'carry'))
    assert "'inspect'" not in w[0] and 'CANCELLED_HOLD_UNKNOWN' in w[0]
    raw = _raw()
    del raw['application']['faults']
    for task in raw['application']['tasks']:
        task.pop('export', None)
    assert warned(raw) == []                             # nothing exported: nothing to warn about


def test_x4_a_task_with_inputs_cannot_be_exported():
    raw = _raw()
    inspect = _task(raw, 'inspect')
    inspect['inputs'] = [{'name': 'parts', 'type': 'candidates'}]
    inspect['export'] = {'number': 9, 'timeout_s': 10}
    text = _refusal(raw)
    assert 'R-P0-04' in text and "'inspect'" in text


def test_x5_the_export_table_id_is_the_shared_rule():
    entries = [(12, 'place_part', 30000), (7, 'home', 5000), (4, 'carry', 10000), (3, 'pick_part', 10000)]
    assert export_table_text(entries) == FIXTURE_TABLE_TEXT       # sorted by number, milliseconds
    assert export_table_id(entries) == FIXTURE_TABLE_ID
    assert export_table_id(entries) == zlib.crc32(FIXTURE_TABLE_TEXT.encode())
    for table, text, crc in SHARED_VECTORS:
        assert export_table_text(table) == text and export_table_id(table) == crc, text
    assert zlib.crc32(b'123456789') == 0xCBF43926                 # the check value TMR's crc32 is pinned to


def test_x6_the_project_schema_carries_no_protocol_field():
    # ADR-0032 Decision 14: an export is intent -- a number and a timeout -- never protocol plumbing
    assert set(TaskExport.__fields__) == {'number', 'timeout_s'}
    for key in ('session', 'requester_session', 'cmd_seq', 'ack_seq', 'heartbeat', 'topic', 'node_id', 'recipe',
                'parameters', 'result_code'):
        raw = _raw()
        _task(raw, 'home')['export'][key] = 1
        _refusal(raw)


def test_an_unexported_project_refuses_nothing_and_exports_nothing():
    raw = _raw()
    for t in raw['application']['tasks']:
        t.pop('export', None)
    assert _project(raw).application.program.exported() == []


# --- B1 / B3: progression by configuration, never by task logic -------------------------------------------------------

def _blocks(tree: str) -> dict:
    """BehaviorTree id -> its exact text."""
    out, rest = {}, tree
    while '<BehaviorTree ID="' in rest:
        start = rest.index('<BehaviorTree ID="')
        end = rest.index('</BehaviorTree>', start) + len('</BehaviorTree>')
        block = rest[start:end]
        out[block.split('"', 2)[1]] = block
        rest = rest[end:]
    return out


def _files(app: Path) -> dict:
    return {str(p.relative_to(app)): p.read_bytes() for p in app.rglob('*') if p.is_file()}


def test_b1_exporting_a_task_adds_only_the_documented_files_and_changes_no_task_tree(tmp_path, record_property):
    with_all = _generate(_raw(), tmp_path / 'all')
    raw = _raw()
    del _task(raw, 'home')['export']
    without_home = _generate(raw, tmp_path / 'without_home')
    raw = _raw()
    for t in raw['application']['tasks']:
        t.pop('export', None)
    without_any = _generate(raw, tmp_path / 'without_any')

    trees = {k: _blocks((app / 'bt_trees' / 'task_program.xml').read_text())
             for k, app in (('all', with_all), ('without_home', without_home), ('without_any', without_any))}
    for name in ('MainTree', 'task_place_part', 'task_home', 'task_pick_part', 'task_carry', 'task_inspect'):
        assert trees['all'][name] == trees['without_home'][name] == trees['without_any'][name], name
    assert set(trees['all']) - set(trees['without_home']) == {'job_home'}
    assert set(trees['all']) - set(trees['without_any']) == {'job_home', 'job_place_part', 'job_pick_part', 'job_carry'}

    # the file-level difference, recorded: only the documented additions
    documented = {'bt_trees/task_program.xml', 'config/tasks.yaml', 'launch/trainit_task_server.launch.py',
                  'package.xml', 'generation_manifest.yaml', 'README.md'}
    diffs = {}
    for other_name, other in (('without_home', without_home), ('without_any', without_any)):
        a, b = _files(with_all), _files(other)
        changed = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
        diffs[other_name] = changed
        assert set(changed) <= documented, (other_name, changed)
    record_property('b1_diff_vs_without_home', ','.join(diffs['without_home']))
    record_property('b1_diff_vs_without_any', ','.join(diffs['without_any']))
    assert 'launch/trainit_task_server.launch.py' not in _files(without_any)
    assert 'launch/trainit_task_server.launch.py' in _files(without_home)   # one export left


def test_b3_an_export_free_project_generates_the_files_of_8_8_0(tmp_path):
    raw = _raw()
    for t in raw['application']['tasks']:
        t.pop('export', None)
    app = _generate(raw, tmp_path / 'free')
    tree = (app / 'bt_trees' / 'task_program.xml').read_text()
    assert 'job_' not in tree
    tasks_text = (app / 'config' / 'tasks.yaml').read_text()
    assert tasks_text.startswith(TASKS_HEADER)                    # 8.8.0's header, byte for byte
    assert 'export' not in tasks_text
    assert not (app / 'launch' / 'trainit_task_server.launch.py').exists()
    assert '2.10.0' not in (app / 'package.xml').read_text()      # no task-server floor


# --- X6: the Composer -- intent, never protocol plumbing ---------------------------------------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _composer(monkeypatch):
    from trainit_setup_assistant.composer import Composer
    monkeypatch.chdir(DATA)                              # the urdf path is relative to it
    return Composer(CanonicalProject.parse_obj(_raw()))


def _task_doc(c, name):
    return next(t for t in c.document()['tasks'] if t['name'] == name)


def test_x6_the_composer_exposes_a_task_with_a_number_and_a_timeout_only(qapp, monkeypatch):
    from python_qt_binding.QtWidgets import QAbstractButton, QLabel
    from trainit_setup_assistant.gui.composer.tasks_panel import TasksPanel
    c = _composer(monkeypatch)
    p = TasksPanel(lambda: c)
    p.task = 'inspect'
    p.refresh()
    assert p.plc_box.isEnabled()
    assert not p.plc_expose.isChecked()                  # internal
    assert p.plc_number.value() == 1                     # the FIRST free number (3, 4, 7 and 12 are taken)
    # expose it: a number and a timeout, nothing else
    p.plc_expose.setChecked(True)
    p.plc_number.setValue(9)
    p.plc_timeout.setValue(12.5)
    assert p.apply_export()
    assert _task_doc(c, 'inspect')['export'] == {'number': 9, 'timeout_s': 12.5}
    texts = [p.task_list.item(i).text() for i in range(p.task_list.count())]
    assert any(t.startswith('inspect') and 'PLC #9' in t for t in texts)
    assert any(t.startswith('place_part') and 'main' in t and 'PLC #12' in t for t in texts)
    # validated with the model's readable message, nothing changed
    p.plc_number.setValue(12)
    assert not p.apply_export()
    assert 'R-P0-03' in p.message.text() and "'place_part'" in p.message.text()
    assert _task_doc(c, 'inspect')['export']['number'] == 9
    # an edit elsewhere never discards what was typed in the PLC group (review): refresh keeps it
    p.plc_number.setValue(44)
    p.refresh()
    assert p.plc_number.value() == 44
    # internal again
    p.plc_expose.setChecked(False)
    assert p.apply_export()
    assert 'export' not in _task_doc(c, 'inspect')
    # making an internal task internal is no edit (no undo entry)
    undo = len(c._undo)
    c.set_task_export('inspect', None)
    assert len(c._undo) == undo
    # the PLC group says nothing of the protocol
    words = ' '.join(w.text() + ' ' + w.toolTip() for w in p.plc_box.findChildren((QLabel, QAbstractButton))).lower()
    words += ' ' + p.plc_box.toolTip().lower() + ' ' + p.plc_number.toolTip().lower() + ' ' + p.plc_timeout.toolTip().lower()
    for word in ('session', 'sequence', 'heartbeat', 'topic', 'node id', 'ack', 'seq', 'code', 'result', 'acknowledg'):
        assert word not in words, word


def test_x6_a_saved_export_reloads_identically(monkeypatch, tmp_path):
    from trainit_setup_assistant.model import project_yaml
    c = _composer(monkeypatch)
    c.set_task_export('inspect', 9, 12.5)
    text = project_yaml(c.project)
    again = yaml.safe_load(text)
    pick = next(t for t in again['application']['tasks'] if t['name'] == 'inspect')
    assert pick['export'] == {'number': 9, 'timeout_s': 12.5}
    assert list(pick).index('export') < list(pick).index('steps')   # written before the steps (field order)
    assert project_yaml(CanonicalProject.parse_obj(again)) == text

# --- the public sample: robot tasks a PLC requests (the owner acceptance's project) ---------------------------------

SAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'ur5e_suction'


def _sample_project():
    old = os.getcwd()
    os.chdir(SAMPLE)                                     # the README's rule: relative paths, from its directory
    try:
        return load_project('robot_tasks.yaml')
    finally:
        os.chdir(old)


def test_the_robot_tasks_sample_exposes_three_tasks_and_tracks_the_held_state():
    """examples/ur5e_suction/robot_tasks.yaml (TSA 8.9.0): neutral, three tasks exposed (home #1, pick_part #2,
    place_part #3), inspect internal (the acceptance exposes it in the Composer), a cycle as the one-shot main task;
    faults: stop + a gripping step -> the held state is tracked: no plan warning at all."""
    pytest.importorskip('xacro', reason='the public sample needs ur_description and xacro')
    try:
        project = _sample_project()
    except Exception as exc:                             # noqa: BLE001 -- no ur_description: the sample cannot load
        if 'ur_description' in str(exc):
            pytest.skip('the public sample needs ur_description')
        raise
    program = project.application.program
    assert [(t.name, t.export.number, t.export.timeout_s) for t in program.exported()] == [
        ('home', 1, 15.0), ('pick_part', 2, 20.0), ('place_part', 3, 30.0)]
    assert {t.name for t in program.tasks} - {t.name for t in program.exported()} == {'inspect', 'cycle'}
    from trainit_setup_assistant.applications.task_program import plan_program, tracks_held_state
    assert tracks_held_state(program)
    old = os.getcwd()
    os.chdir(SAMPLE)
    try:
        assert plan_program(project).warnings == []
    finally:
        os.chdir(old)


def test_the_fake_plc_command_is_written_when_the_server_says_it_serves(qapp):
    """P0.a owner acceptance, finding 2: the fake PLC's command was written BEFORE the server started and its
    start-up lines buried it. It is now written once per start, right after the server's "serving" line."""
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.run_panel import RunPanel
    from trainit_setup_assistant.gui.wizard import SERVING
    from trainit_setup_assistant.composer import launch_plan
    panel = RunPanel(Launcher(), [('server', 'Serve the PLC tasks', lambda: None)])
    panel.after_ready['server'] = (SERVING, ['# the fake PLC:', 'ros2 run trainit_motion_runtime trainit_fake_plc'])
    panel.started(launch_plan.Command('server', 'Serve the PLC tasks', 'true', True))
    panel.line('server', '[trainit_task_server-1] [INFO] BB[scene_table_pos] = 0.3;0;-0.03')
    assert 'trainit_fake_plc' not in panel.log.toPlainText()
    panel.line('server', f'[trainit_task_server-1] [INFO] [trainit_task_server]: {SERVING} d299da23, 4 exported '
                         f'task(s) ...')
    panel.line('server', '[trainit_task_server-1] [INFO] RTI exported task #1 home ...')
    log = panel.log.toPlainText().splitlines()
    at = next(i for i, line in enumerate(log) if SERVING in line)
    assert log[at + 1:at + 3] == ['# the fake PLC:', 'ros2 run trainit_motion_runtime trainit_fake_plc']
    panel.line('server', f'[trainit_task_server-1] {SERVING} again')   # once per start
    assert panel.log.toPlainText().count('trainit_fake_plc') == 1
