"""TrainIt D0.8: the Composer's Data section (D0_8_DESIGN.md §§1-9), offscreen.

What is pinned here is the PRODUCT behaviour TSA owns -- authoring through the canonical model, capability
states (no control that cannot work), the trigger pickers fed by the program, the exact offline commands TSA
launches, recover only behind a confirmation, the export's words -- with the offline tool replaced by a recorder
of the argv it was given (the real tool runs in test_d0_8_runtime.py). The owner's GUI acceptance (flows A-E)
is the real proof; this is its regression net.
"""
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import test_m3_7_composition as C
from test_d0_8_data_collection import DATA, with_data
from trainit_setup_assistant.composer import data_tools as D
from trainit_setup_assistant.composer.ops import Composer

REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / 'trainit_data' / 'examples' / 'isaac_replicator_bin'


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _cell(monkeypatch):
    monkeypatch.chdir(C.DATA)
    import test_example_ur5e_suction as S
    S.needs_ur_description()


class FakeLauncher:
    def __init__(self):
        self.started, self.listeners, self.processes = [], [], {}

    def add_listener(self, listener):
        if listener not in self.listeners:
            self.listeners.append(listener)

    def running(self, kind):
        return False

    def start(self, command):
        self.started.append(command)


class Cli:
    """Stands in for ``python3 -m trainit_data --json``: records every argv, answers from a table."""

    def __init__(self, answers=None):
        self.calls, self.answers = [], answers or {}

    def __call__(self, args, root=None, timeout_s=600.0, env=None):
        self.calls.append(list(args))
        key = ' '.join(args[:2])
        ok, result, error = self.answers.get(key, (True, {}, None))
        if callable(result):
            result = result(args)
        return D.CliResult(argv=D.cli_argv(args, root), exit_code=0 if ok else 1, ok=ok, command=key,
                           result=result, error=error)


def make_panel(qapp, monkeypatch, project=None, cli=None, settings=None, modes=None):
    from trainit_setup_assistant.gui.composer.data_panel import DataPanel
    project = project or C.build(with_data(modes=modes) if modes else with_data())
    ctrl = SimpleNamespace(project=project, last_generation_dir=None)
    composer = Composer(project, on_change=lambda p: setattr(ctrl, 'project', p))
    host = SimpleNamespace(launcher=FakeLauncher(),
                           launcher_settings=settings or SimpleNamespace(isaac_launcher='', data_root='/tmp/store',
                                                                         synthetic_provider_dirs=''))
    cli = cli or Cli()
    monkeypatch.setattr(D, 'run_cli', cli)
    monkeypatch.setattr(D, 'provider_dirs', lambda settings=None: [EXAMPLE.parent] if EXAMPLE.is_dir() else [])
    panel = DataPanel(ctrl, lambda: composer, host=lambda: host)
    return panel, ctrl, host, cli


def _wait(panel):
    panel.jobs.wait()


# ---- Capture ------------------------------------------------------------------------------------------------

def test_the_capture_tab_shows_the_recording_setup_in_product_words(qapp, monkeypatch):
    panel, ctrl, _host, _cli = make_panel(qapp, monkeypatch)
    panel.refresh()
    assert panel.camera_state.text() == 'Ready for recording.'
    assert panel.camera_id.text() == 'bin_camera' and panel.camera_profile.currentData() == 'realsense_d455'
    assert panel.dataset_id.text() == 'bin_parts'
    rows = [panel.triggers.item(i, 1).text() for i in range(panel.triggers.rowCount())]
    assert rows == ['when you press Capture', 'every 2000 ms, at most 50 captures',
                    'when process channel "conveyor" turns on',
                    'when step "pick" of task "transfer" starts (before the grasp)', 'when task "cell_cycle" succeeds']
    assert panel.setup_state.text().startswith('Valid.')
    assert not panel.camera_advanced.isVisibleTo(panel) and not panel.sync_timeout.isVisibleTo(panel)
    panel.advanced.setChecked(True)
    assert panel.camera_advanced.isVisibleTo(panel) and '/camera/color/image_raw' in panel.camera_advanced.text()
    # no ROS recorder here: the session buttons are disabled, and the reason is said
    assert not panel.record_button.isEnabled() and not panel.capture_button.isEnabled()
    assert not panel.recorder_start.isEnabled()                    # nothing generated yet


def test_a_project_without_a_camera_offers_nothing_that_cannot_work(qapp, monkeypatch):
    project = C.build(with_data(data=None, camera=None))
    panel, *_ = make_panel(qapp, monkeypatch, project=project)
    panel.refresh()
    assert 'No camera in this project' in panel.camera_state.text()
    assert not panel.camera_apply.isEnabled() and not panel.preview_button.isEnabled()
    assert not panel.trigger_add.isEnabled()


def test_naming_the_camera_and_saving_the_setup_are_validated_edits(qapp, monkeypatch):
    project = C.build(with_data(data=None, camera={'rgb_topic': '/camera/color/image_raw'}))
    panel, ctrl, *_ = make_panel(qapp, monkeypatch, project=project)
    panel.refresh()
    panel.camera_id.setText('bin_camera')
    panel.camera_profile.setCurrentIndex(panel.camera_profile.findData('realsense_d455'))
    panel.apply_camera()
    assert ctrl.project.perception.camera.id == 'bin_camera' and ctrl.project.schema_version == 5
    panel.dataset_id.setText('bin_parts')
    panel.save_setup()
    dc = ctrl.project.data_collection
    assert dc.dataset.id == 'bin_parts' and [s.camera for s in dc.sensors] == ['bin_camera']
    panel.camera_id.setText('Bin Camera!')              # kept (it is the user's), and the ONE validator flags it
    panel.apply_camera()
    from trainit_setup_assistant.generator.data_collection import data_collection_problems
    assert ctrl.project.perception.camera.id == 'Bin Camera!'
    assert ctrl.project.data_collection.sensors[0].camera == 'Bin Camera!'          # renamed with the camera
    assert any(p.startswith("R-D0-03: camera id 'Bin Camera!'") for p in data_collection_problems(ctrl.project))
    panel.refresh()
    assert 'problem(s)' in panel.setup_state.text()


def test_the_trigger_dialog_offers_only_the_programs_tasks_steps_and_channels(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer.data_panel import TriggerDialog
    project = C.build(with_data())
    dialog = TriggerDialog(project)
    kinds = [dialog.kind.itemData(i) for i in range(dialog.kind.count())]
    assert kinds == ['manual', 'interval', 'event', 'application_state']
    assert [dialog.channel.itemText(i) for i in range(dialog.channel.count())] == ['conveyor']
    dialog.kind.setCurrentIndex(3)
    dialog.task.setCurrentText('transfer')
    steps = [dialog.step.itemData(i) for i in range(dialog.step.count())]
    assert steps == [None, 'pick', 'place']
    dialog.step.setCurrentIndex(1)                                  # pick: a grasp step
    assert dialog.attempt.isEnabled() and dialog.attempt.itemData(1) == 'pre_grasp'
    dialog.transition.setCurrentIndex(dialog.transition.findData('exit'))
    assert dialog.attempt.itemData(1) == 'post_grasp'               # the phase follows the transition
    dialog.step.setCurrentIndex(2)                                  # place: a release step, no attempt
    assert not dialog.attempt.isEnabled()
    dialog.step.setCurrentIndex(1)
    dialog.transition.setCurrentIndex(dialog.transition.findData('enter'))
    dialog.attempt.setCurrentIndex(1)
    dialog.id.setText('before_grasp_2')
    assert dialog.value() == {'id': 'before_grasp_2', 'kind': 'application_state', 'task': 'transfer',
                              'transition': 'enter', 'step': 'pick', 'pick_attempt': 'pre_grasp'}


def test_a_template_application_offers_no_application_trigger(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer.data_panel import TriggerDialog
    project = C.build(with_data())
    project.application.type = type(project.application.type)('pick_and_place')
    dialog = TriggerDialog(project)
    assert [dialog.kind.itemData(i) for i in range(dialog.kind.count())] == ['manual', 'interval']


def test_adding_and_removing_a_trigger_goes_through_the_composer(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, ctrl, *_ = make_panel(qapp, monkeypatch)
    panel.refresh()

    class Accept:
        def __init__(self, project, current, parent):
            pass

        def exec_(self):
            return P.QDialog.Accepted

        def value(self):
            return {'id': 'every_5s', 'kind': 'interval', 'period_ms': 5000}
    monkeypatch.setattr(P, 'TriggerDialog', Accept)
    panel.edit_trigger(None)
    assert [t.id for t in ctrl.project.data_collection.triggers][-1] == 'every_5s'
    panel.triggers.selectRow(panel.triggers.rowCount() - 1)
    panel.remove_trigger()
    assert 'every_5s' not in [t.id for t in ctrl.project.data_collection.triggers]


def test_the_recorder_starts_only_from_a_generated_file_as_a_copyable_command(qapp, monkeypatch, tmp_path):
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)
    config = tmp_path / 'bundle' / ctrl.project.bundle.app_package / 'config'
    config.mkdir(parents=True)
    (config / 'data_collection.mock.yaml').write_text('schema: trainit-recording/1.0\n')
    ctrl.last_generation_dir = str(tmp_path / 'bundle')
    panel.refresh()
    assert panel.mode.count() == 1 and panel.mode.currentText() == 'mock'
    assert panel.recorder_start.isEnabled()
    panel.toggle_recorder()
    command = host.launcher.started[-1]
    assert command.kind == 'data_recorder' and 'recorder.launch.py' in command.display()
    assert str(config / 'data_collection.mock.yaml') in command.display()


# ---- Synthetic ----------------------------------------------------------------------------------------------

def _needs_example():
    if not (EXAMPLE / 'provider.json').is_file():
        pytest.skip('the trainit_data repository (its reference provider) is not beside TSA')


def test_a_mock_only_project_gets_an_explanation_not_controls(qapp, monkeypatch):
    _needs_example()
    panel, *_ = make_panel(qapp, monkeypatch)
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    assert 'Synthetic data needs a isaac deployment' in panel.provider_state.text()
    assert not panel.provider_box.isVisibleTo(panel)


def test_no_isaac_on_this_machine_says_what_is_missing(qapp, monkeypatch):
    _needs_example()
    panel, *_ = make_panel(qapp, monkeypatch, modes=['mock', 'isaac'])
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    assert 'Isaac Sim is not installed here' in panel.provider_state.text()
    assert not panel.provider_box.isVisibleTo(panel)


def _isaac_settings(tmp_path):
    (tmp_path / 'isaac-sim.sh').write_text('#!/bin/sh\n')
    (tmp_path / 'python.sh').write_text('#!/bin/sh\n')
    return SimpleNamespace(isaac_launcher=str(tmp_path / 'isaac-sim.sh'), data_root=str(tmp_path / 'store'),
                           synthetic_provider_dirs='')


def test_the_recipe_form_is_built_from_the_descriptor(qapp, monkeypatch, tmp_path):
    _needs_example()
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch, modes=['mock', 'isaac'], settings=_isaac_settings(tmp_path))
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    assert panel.provider_box.isVisibleTo(panel) and panel.provider.id == 'isaac_replicator_bin'
    simple = [p['key'] for p in panel.provider.descriptor['parameters'] if p['level'] == 'simple']
    assert list(panel.param_widgets) == simple
    assert panel.seed.value() == 20261102 and panel.runs.value() == 24 and panel.views.value() == 2
    panel.advanced.setChecked(True)
    assert list(panel.param_widgets) == [p['key'] for p in panel.provider.descriptor['parameters']]
    # every control is a real field: nothing the descriptor does not declare
    assert 'orientation' not in ' '.join(panel.param_widgets) and 'distractor' not in ' '.join(panel.param_widgets)
    panel.param_widgets['objects_per_run'][1].setValue(5)
    panel.runs.setValue(3)
    panel.save_recipe()
    recipe = ctrl.project.data_collection.synthetic
    assert recipe.provider == 'isaac_replicator_bin' and recipe.runs == 3
    assert recipe.parameters == {'objects_per_run': [4, 5]}            # only what differs from the base config


def test_generate_and_preview_launch_the_provider_and_read_its_progress(qapp, monkeypatch, tmp_path):
    _needs_example()
    sid = 'ses-20261103T101500Z-a1b2c3d4'
    panel, ctrl, host, cli = make_panel(qapp, monkeypatch, modes=['mock', 'isaac'],
                                        settings=_isaac_settings(tmp_path))
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    panel.runs.setValue(4)
    panel.start_generation(preview=False)
    command = host.launcher.started[-1]
    assert command.kind == 'synthetic' and '--runs 4' in command.display() and command.display().startswith('env -i')
    for line in ('TRAINIT_PROGRESS {"phase": "starting"}', 'TRAINIT_PROGRESS {"phase": "session", "runs": 4, '
                 '"views": 2, "session_id": "' + sid + '"}', 'TRAINIT_PROGRESS {"phase": "run", "run": 2, "runs": 4}'):
        panel.line('synthetic', line)
    assert panel.progress.maximum() == 4 and panel.progress.value() == 2
    panel.line('synthetic', f'TRAINIT_SESSION {sid}')
    panel.finished('synthetic', 0)
    assert 'Session created' in panel.synthetic_result.text() and panel.add_to_dataset.isVisibleTo(panel)
    # the preview: 1 run x 1 view into a TSA-owned scratch store, removed when replaced
    panel.start_generation(preview=True)
    preview = host.launcher.started[-1]
    assert preview.kind == 'synthetic_preview' and '--runs 1 --views 1 --store' in preview.display()
    scratch = Path(panel._preview_dir)
    assert scratch.name.startswith('tsa_synthetic_preview_') and scratch.is_dir()
    panel.start_generation(preview=True)
    assert not scratch.exists()                                         # the previous preview's scratch: gone
    panel._drop_preview()


def test_a_refused_configuration_is_reported_in_words(qapp, monkeypatch, tmp_path):
    _needs_example()
    panel, *_ = make_panel(qapp, monkeypatch, modes=['mock', 'isaac'], settings=_isaac_settings(tmp_path))
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    panel.start_generation(preview=False)
    panel.line('synthetic', 'TRAINIT_PROGRESS {"phase": "invalid_config", "problems": ["tray.inner_m: too big"]}')
    panel.line('synthetic', 'TRAINIT_FAILED')
    panel.finished('synthetic', 2)
    assert 'refused the settings: tray.inner_m: too big' in panel.synthetic_result.text()


# ---- Datasets -----------------------------------------------------------------------------------------------

SESSIONS = [
    {'session_id': 'ses-20261103T090000Z-00000001', 'status': 'CLOSED', 'transport': 'LIVE_ROS', 'backend': 'real',
     'captures': 12, 'opened_wall_utc_ns': 1_793_000_000_000_000_000, 'classification': 'INTERNAL',
     'sensors': ['bin_camera'], 'outcomes': {'complete': 11, 'partial': 1, 'failed': 0}, 'synthetic': False},
    {'session_id': 'ses-20261103T091000Z-00000002', 'status': 'CLOSED', 'transport': 'INGEST', 'backend': 'isaac',
     'captures': 8, 'opened_wall_utc_ns': 1_793_000_600_000_000_000, 'classification': 'INTERNAL',
     'sensors': ['bin_camera'], 'outcomes': {'complete': 8, 'partial': 0, 'failed': 0}, 'synthetic': True},
    {'session_id': 'ses-20261103T092000Z-00000003', 'status': 'ACTIVE', 'transport': 'LIVE_ROS', 'backend': 'mock',
     'captures': None, 'opened_wall_utc_ns': 1_793_001_200_000_000_000, 'classification': 'INTERNAL',
     'sensors': ['bin_camera'], 'outcomes': None, 'synthetic': False},
]
VERSION = {'version': 1, 'dataset_id': 'bin_parts', 'content_hash': 'sha256:' + '0' * 64,
           'derivation': {'op': 'CREATE'}, 'rights': {'classification': 'INTERNAL'}, 'content_kind': 'OBSERVATIONS',
           'content': {'selection': {'captures': 19, 'excluded': {'COMPLETE': 0, 'PARTIAL': 1, 'FAILED': 0},
                                     'filter': {'include_outcomes': ['COMPLETE']}},
                       'sessions': [{'session_id': s['session_id'], 'source': {'backend': s['backend']}}
                                    for s in SESSIONS[:2]],
                       'label_sets': [{'label_set_id': 'ls-1'}],
                       'split': {'group_by': 'randomization_run', 'counts': {
                           'train': {'groups': 3, 'captures': 15}, 'val': {'groups': 1, 'captures': 2},
                           'test': {'groups': 1, 'captures': 2}}}}}


def datasets_cli(versions=()):
    return Cli({
        'inspect --detail': (True, {'root': '/tmp/store', 'store_id': 's', 'schemas': {}, 'sessions': SESSIONS}, None),
        'dataset list': (True, {'datasets': [{'dataset_id': 'bin_parts', 'name': 'Bin parts',
                                              'classification': 'INTERNAL', 'versions': list(versions)}]}, None),
        'dataset show': (True, {'dataset': {}, 'versions': list(versions), 'label_sets': ['ls-1']}, None),
        'version show': (True, {'version': VERSION}, None),
        'labels from-synthetic': (True, {'label_set': {'label_set_id': 'ls-synth'}}, None),
        'version create': (True, {'version': dict(VERSION, version=len(versions) + 1)}, None),
        'recover ses-20261103T092000Z-00000003': (True, {'session_id': SESSIONS[2]['session_id'],
                                                          'committed_captures': 2, 'quarantined': [{'from': 'a',
                                                                                                    'to': 'b'}]},
                                                  None),
        'export yolo': (True, {'export': {'destination': '/tmp/out', 'summary': {'splits': {
            'train': {'images': 15, 'boxes': 40}, 'val': {'images': 2, 'boxes': 5},
            'test': {'images': 2, 'boxes': 6}}}}}, None),
    })


def _datasets(qapp, monkeypatch, versions=()):
    cli = datasets_cli(versions)
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch, cli=cli)
    panel.tabs.setCurrentIndex(2)
    panel.refresh()
    _wait(panel)
    _wait(panel)
    return panel, cli


def test_the_sessions_table_says_source_health_and_interruption(qapp, monkeypatch):
    panel, cli = _datasets(qapp, monkeypatch)
    assert ['inspect', '--detail'] in cli.calls and ['dataset', 'list'] in cli.calls
    rows = [[panel.session_table.item(r, c).text() for c in range(5)] for r in range(panel.session_table.rowCount())]
    assert [r[1:] for r in rows] == [['Camera (mock)', '—', '—', 'interrupted'],
                                     ['Synthetic (isaac)', '8', '8 / 0 / 0', 'closed'],
                                     ['Camera (real)', '12', '11 / 1 / 0', 'closed']]
    assert panel.session_table.isColumnHidden(5)                       # session ids: Advanced only
    assert 'Interrupted recording found: 1 session' in panel.interrupted.text()
    assert panel.dataset_combo.currentData() == 'bin_parts'


def test_recover_happens_only_after_the_explicit_confirmation(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, cli = _datasets(qapp, monkeypatch)
    panel.session_table.selectRow(0)
    assert panel.recover_button.isEnabled()
    asked = []
    monkeypatch.setattr(P.QMessageBox, 'question', lambda *a, **k: asked.append(a[2]) or P.QMessageBox.Cancel)
    panel.recover_selected()
    _wait(panel)
    assert not any(c[0] == 'recover' for c in cli.calls)
    assert 'deletes nothing' in asked[0] and 'orphans/' in asked[0]
    monkeypatch.setattr(P.QMessageBox, 'question', lambda *a, **k: P.QMessageBox.Yes)
    panel.session_table.selectRow(0)
    panel.recover_selected()
    _wait(panel)
    assert ['recover', 'ses-20261103T092000Z-00000003'] in cli.calls
    assert 'nothing deleted' in panel.dataset_result.text()


def test_a_sealed_session_cannot_be_recovered_and_versions_take_only_sealed_ones(qapp, monkeypatch):
    panel, _ = _datasets(qapp, monkeypatch)
    panel.session_table.selectRow(2)
    assert not panel.recover_button.isEnabled() and panel.create_version_button.isEnabled()
    panel.session_table.selectRow(0)                                    # the interrupted one
    assert not panel.create_version_button.isEnabled()


def _accept_dialogs(monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    monkeypatch.setattr(P.QDialog, 'exec_', lambda self: P.QDialog.Accepted)


def test_create_version_labels_generated_sessions_then_builds_v1_complete_only(qapp, monkeypatch):
    _accept_dialogs(monkeypatch)
    panel, cli = _datasets(qapp, monkeypatch)
    from python_qt_binding.QtCore import QItemSelectionModel
    panel.session_table.selectRow(1)
    model = panel.session_table.selectionModel()
    model.select(panel.session_table.model().index(2, 0), QItemSelectionModel.Select | QItemSelectionModel.Rows)
    panel.create_version()
    _wait(panel)
    _wait(panel)
    labels = next(c for c in cli.calls if c[:2] == ['labels', 'from-synthetic'])
    assert labels == ['labels', 'from-synthetic', 'bin_parts', '--kind', 'BBOX_2D', '--session',
                      'ses-20261103T091000Z-00000002']
    version = next(c for c in cli.calls if c[:2] == ['version', 'create'])
    assert version == ['version', 'create', 'bin_parts', '--group-by', 'randomization_run',
                       '--session', 'ses-20261103T091000Z-00000002', '--session', 'ses-20261103T090000Z-00000001',
                       '--include-outcome', 'COMPLETE', '--label-set', 'ls-synth']
    assert 'Version v1 created: 19 captures included, 1 excluded.' in panel.dataset_result.text()


def test_a_dataset_with_versions_gets_a_derived_version(qapp, monkeypatch):
    _accept_dialogs(monkeypatch)
    panel, cli = _datasets(qapp, monkeypatch, versions=[1])
    _wait(panel)
    assert panel.version_list.count() == 1 and 'v1' in panel.version_summary.text()
    assert '19 captures included' in panel.version_summary.text()
    panel.session_table.selectRow(2)
    panel.create_version()
    _wait(panel)
    version = next(c for c in cli.calls if c[:2] == ['version', 'create'])
    assert version[3:7] == ['--op', 'ADD_SESSIONS', '--from', '1']


def test_export_names_the_split_and_says_the_canonical_data_is_unchanged(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, cli = _datasets(qapp, monkeypatch, versions=[1])
    _wait(panel)
    monkeypatch.setattr(P.QFileDialog, 'getExistingDirectory', lambda *a, **k: '/tmp/out')
    panel.export_yolo()
    _wait(panel)
    assert ['export', 'yolo', 'bin_parts', '1', '/tmp/out'] in cli.calls
    assert panel.dataset_result.text() == ('Exported 19 images (train 15 / val 2 / test 2) to /tmp/out. The '
                                           'canonical data was not changed.')


def test_an_empty_store_is_said_plainly(qapp, monkeypatch):
    cli = Cli({'inspect --detail': (False, None, {'code': 'STORE_NOT_INITIALIZED', 'message': 'no store'}),
               'dataset list': (False, None, {'code': 'STORE_NOT_INITIALIZED', 'message': 'no store'})})
    panel, *_ = make_panel(qapp, monkeypatch, cli=cli)
    panel.tabs.setCurrentIndex(2)
    panel.refresh()
    _wait(panel)
    assert 'The data store is empty' in panel.store_state.text()
    assert panel.session_table.rowCount() == 0 and not panel.create_version_button.isEnabled()


def test_the_data_diagnostic_opens_the_data_section(qapp, monkeypatch):
    from trainit_setup_assistant.composer.diagnostics import parse
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(C.SOURCE)
    w = SetupWizard(ctrl)
    w.blocks_page.initializePage()
    shell = w.blocks_page.program_panel
    shell.go_to(parse("R-D0-06: data collection trigger 'x': no task is named 'y' (tasks: a)"))
    assert shell.stack.currentWidget() is shell.data and shell.data.tabs.currentIndex() == 0
