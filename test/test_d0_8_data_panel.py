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
        import d08_env
        d08_env.require('the trainit_data repository (its reference provider) is not beside TSA')


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
                       'label_sets': [{'label_set_id': 'ls-1', 'kind': 'BBOX_2D', 'source': 'SYNTHETIC_GROUND_TRUTH',
                                       'entries_selected': 19}],
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
                       '--include-outcome', 'COMPLETE',
                       '--session', 'ses-20261103T091000Z-00000002', '--session', 'ses-20261103T090000Z-00000001',
                       '--label-set', 'ls-synth']
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
    # ADD_SESSIONS inherits the parent's outcome filter: trainit_data refuses --include-outcome with it
    # (DERIVATION_INVALID, found while preparing owner flow 3)
    assert version == ['version', 'create', 'bin_parts', '--op', 'ADD_SESSIONS', '--from', '1',
                       '--session', 'ses-20261103T090000Z-00000001']


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


def test_back_from_generate_shows_the_new_recorder_modes(qapp, monkeypatch, tmp_path):
    """Owner acceptance flow 1: Generate (Step 8), then Back to Step 7. QWizard does not re-run
    initializePage on Back; showing the page must still offer the modes the generation wrote."""
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(C.SOURCE)
    w = SetupWizard(ctrl)
    page = w.blocks_page
    page.initializePage()
    shell = page.program_panel
    shell.select('Data')
    assert shell.data.mode.count() == 0
    config = tmp_path / 'bundle' / ctrl.project.bundle.app_package / 'config'
    config.mkdir(parents=True)
    (config / 'data_collection.isaac.yaml').write_text('schema: trainit-recording/1.0\n')
    ctrl.last_generation_dir = str(tmp_path / 'bundle')       # what Step 8's Generate records
    from python_qt_binding.QtGui import QShowEvent
    page.showEvent(QShowEvent())                              # Back: shown again, no initializePage
    assert [shell.data.mode.itemText(i) for i in range(shell.data.mode.count())] == ['isaac']


# ---- review closure (D0.8) ----------------------------------------------------------------------------------

def test_a_real_only_project_offers_live_capture_and_no_synthetic_controls(qapp, monkeypatch, tmp_path):
    _needs_example()
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch, modes=['real'], settings=_isaac_settings(tmp_path))
    panel.refresh()
    assert panel.camera_state.text() == 'Ready for recording.' and panel.setup_state.text().startswith('Valid.')
    assert panel.trigger_add.isEnabled() and 'Isaac' not in panel.recorder_state.text()
    config = tmp_path / 'bundle' / ctrl.project.bundle.app_package / 'config'
    config.mkdir(parents=True)
    (config / 'data_collection.real.yaml').write_text('schema: trainit-recording/1.0\n')
    ctrl.last_generation_dir = str(tmp_path / 'bundle')
    panel.refresh()
    assert [panel.mode.itemText(i) for i in range(panel.mode.count())] == ['real'] and panel.recorder_start.isEnabled()
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    assert 'Synthetic data needs a isaac deployment: this project deploys real' in panel.provider_state.text()
    assert not panel.provider_box.isVisibleTo(panel)               # no controls, even with Isaac installed here


def test_the_preview_writes_only_into_its_own_scratch_store(qapp, monkeypatch, tmp_path):
    """The preview's command names ONLY a fresh TSA scratch directory -- as --store AND as TRAINIT_DATA_ROOT,
    under env -i (an exported TRAINIT_DATA_ROOT cannot leak in) -- never the user's store; and the provider gives
    --store precedence over the environment (pinned on the provider side: test_d0_8_provider.py)."""
    _needs_example()
    import tempfile
    from trainit_setup_assistant.gui.composer.data_panel import is_preview_scratch
    user_store = tmp_path / 'TrainItData'
    settings = _isaac_settings(tmp_path)
    settings.data_root = str(user_store)
    monkeypatch.setenv('TRAINIT_DATA_ROOT', str(user_store))
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch, modes=['mock', 'isaac'], settings=settings)
    panel.tabs.setCurrentIndex(1)
    panel.refresh()
    panel.start_generation(preview=True)
    shown = host.launcher.started[-1].display()
    scratch = Path(panel._preview_dir)
    assert is_preview_scratch(scratch) and scratch.parent == Path(tempfile.gettempdir()).resolve()
    assert str(user_store) not in shown and shown.startswith('env -i ')
    assert f'TRAINIT_DATA_ROOT={scratch}/store ' in shown and f'--store {scratch}/store' in shown
    assert not user_store.exists()
    # the generation proper writes to the user's store, by design (and only it)
    panel.start_generation(preview=False)
    assert f'TRAINIT_DATA_ROOT={user_store} ' in host.launcher.started[-1].display()
    assert '--store' not in host.launcher.started[-1].display()
    panel._drop_preview()
    assert not scratch.exists()


def test_only_a_tsa_scratch_directory_is_ever_removed(tmp_path):
    import tempfile
    from trainit_setup_assistant.gui.composer.data_panel import is_preview_scratch
    assert not is_preview_scratch(tmp_path / 'tsa_synthetic_preview_x')          # right name, wrong place
    assert not is_preview_scratch(Path(tempfile.gettempdir()) / 'TrainItData')  # right place, wrong name
    assert not is_preview_scratch(Path.home() / 'TrainItData')


def test_stale_cli_answers_about_another_dataset_are_ignored(qapp, monkeypatch):
    panel, cli = _datasets(qapp, monkeypatch, versions=[1])
    _wait(panel)
    assert panel.versions == [1]
    panel._dataset_shown('another_dataset', D.CliResult(argv=[], exit_code=0, ok=True, result={
        'versions': [1, 2, 3], 'label_sets': []}))
    assert panel.versions == [1] and panel.version_list.count() == 1
    shows = [c for c in cli.calls if c[:2] == ['dataset', 'show']]
    assert len(shows) == 1                                            # one dataset change, one read


def test_an_unanswered_recorder_call_times_out_and_frees_the_buttons(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, *_ = make_panel(qapp, monkeypatch)
    forgotten = []
    panel._link = SimpleNamespace(forget=lambda s, f: forgotten.append(s), status=lambda: None,
                                  reachable=lambda: True, alive=lambda: True, base='/trainit_data_recorder')
    never = SimpleNamespace(done=lambda: False)
    panel._pending = [('start_session', never, P.time.monotonic() - P.CALL_TIMEOUT_S['start_session'] - 1)]
    panel._poll_calls()
    assert panel._pending == [] and forgotten == ['start_session']
    assert 'did not answer' in panel.session_result.text()


def test_a_second_recorder_is_never_started_beside_a_running_one(qapp, monkeypatch, tmp_path):
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)
    config = tmp_path / 'bundle' / ctrl.project.bundle.app_package / 'config'
    config.mkdir(parents=True)
    (config / 'data_collection.mock.yaml').write_text('x: 1\n')
    ctrl.last_generation_dir = str(tmp_path / 'bundle')
    panel._link = SimpleNamespace(status=lambda: None, reachable=lambda: True, alive=lambda: True,
                                  base='/trainit_data_recorder')
    panel._link_ns = ''
    panel._refresh_modes()
    panel._refresh_session()
    assert not panel.recorder_start.isEnabled() and 'already running' in panel.recorder_start.toolTip()


def test_unsaved_setup_edits_survive_a_refresh(qapp, monkeypatch):
    panel, ctrl, *_ = make_panel(qapp, monkeypatch)
    panel.refresh()
    panel.dataset_description.setText('typed, not saved')
    panel.dataset_description.textEdited.emit('typed, not saved')
    panel.advanced.setChecked(True)                                   # a refresh
    assert panel.dataset_description.text() == 'typed, not saved' and panel.setup_dirty
    panel.save_setup()
    assert ctrl.project.data_collection.dataset.description == 'typed, not saved' and not panel.setup_dirty


def test_edit_needs_a_selected_trigger(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, *_ = make_panel(qapp, monkeypatch)
    panel.refresh()
    panel.triggers.clearSelection()
    panel.triggers.setCurrentCell(-1, -1)
    panel._trigger_selected()
    assert not panel.trigger_edit.isEnabled()
    opened = []
    monkeypatch.setattr(P, 'TriggerDialog', lambda *a: opened.append(a) or (_ for _ in ()).throw(AssertionError))
    panel.edit_trigger(-1)
    assert opened == []


# ---- owner acceptance flow 1 findings (D0.8) ---------------------------------------------------------------

class FakeLink:
    base = '/trainit_data_recorder'

    def reachable(self):
        return True

    def status(self, max_age_s=None):
        return {'level': 0, 'backend': 'isaac', 'values': {'session_state': 'IDLE'}}

    def alive(self):
        return True

    def close(self):
        pass


def test_recorder_starting_is_replaced_by_ready_or_by_why_it_ended(qapp, monkeypatch, tmp_path):
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)
    config = tmp_path / 'bundle' / ctrl.project.bundle.app_package / 'config'
    config.mkdir(parents=True)
    (config / 'data_collection.isaac.yaml').write_text('schema: trainit-recording/1.0\n')
    ctrl.last_generation_dir = str(tmp_path / 'bundle')
    panel.refresh()
    panel.toggle_recorder()
    assert panel in host.launcher.listeners and panel.message.text() == 'Recorder starting…'
    # it ends at once (the flow 1 defect: a launch argument refused): the reason replaces "starting"
    panel.line('data_recorder', "malformed launch argument 'namespace:=', expected format '<name>:=<value>'")
    panel.line('data_recorder', '')
    panel.finished('data_recorder', 1)
    assert panel.message.text().startswith("The recorder ended (exit 1): malformed launch argument 'namespace:='")
    # started again, and it answers: "starting" becomes "ready"
    panel.toggle_recorder()
    panel._link = FakeLink()
    panel._refresh_session()
    assert panel.message.text() == 'Recorder ready.' and panel.recorder_state.text().startswith('Recorder: ready')
    panel._refresh_session()
    assert panel.message.text() == 'Recorder ready.'


def test_the_recording_setups_dataset_not_created_yet_is_said_not_replaced(qapp, monkeypatch):
    cli = datasets_cli()
    cli.answers['dataset list'] = (True, {'datasets': [{'dataset_id': 'd07_other', 'name': 'Another dataset',
                                                        'classification': 'INTERNAL', 'versions': [1]}]}, None)
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch, cli=cli)
    panel.tabs.setCurrentIndex(2)
    panel.refresh()
    _wait(panel)
    _wait(panel)
    planned = ctrl.project.data_collection.dataset.id
    assert panel.dataset_combo.currentIndex() == -1 and panel.dataset_combo.count() == 1   # never d07_other by itself
    assert not panel.dataset_hint.isHidden() and f'"{planned}" does not exist yet' in panel.dataset_hint.text()
    assert not panel.create_version_button.isEnabled()
    panel.dataset_combo.setCurrentIndex(0)                       # the user's own choice: allowed, hint gone
    _wait(panel)
    assert panel.dataset_combo.currentData() == 'd07_other' and panel.dataset_hint.isHidden()


def test_generate_is_offered_only_before_a_generation_and_opens_step_8(qapp, monkeypatch, tmp_path):
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)
    opened = []
    host.show_generate = lambda: opened.append(True)
    panel.refresh()
    assert not panel.bundle_generate.isHidden() and panel.mode.count() == 0
    panel.bundle_generate.clicked.emit(False)                 # the button's own signal
    assert opened == [True]
    config = tmp_path / 'bundle' / ctrl.project.bundle.app_package / 'config'
    config.mkdir(parents=True)
    (config / 'data_collection.mock.yaml').write_text('schema: trainit-recording/1.0\n')
    ctrl.last_generation_dir = str(tmp_path / 'bundle')
    panel.refresh()
    assert panel.bundle_generate.isHidden() and panel.mode.count() == 1


def test_no_generate_button_without_a_host_that_opens_step_8(qapp, monkeypatch):
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)        # no show_generate: absent, not decorative
    panel.refresh()
    assert panel.bundle_generate.isHidden()


def test_the_wizard_opens_step_8_from_the_composer(qapp, monkeypatch):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(C.SOURCE)
    w = SetupWizard(ctrl)
    moved = []
    monkeypatch.setattr(w, 'currentPage', lambda: w.blocks_page)
    monkeypatch.setattr(w, 'next', lambda: moved.append('next'))
    w.show_generate()
    assert moved == ['next']
    monkeypatch.setattr(w, 'currentPage', lambda: w.generate_page)
    w.show_generate()
    assert moved == ['next']                                    # anywhere else: nothing


def test_step_8_says_where_the_recorder_starts_after_generate(qapp, monkeypatch, tmp_path):
    """Generate then Back is not obvious (owner, flow 1): Step 8's result says it, for the modes it wrote."""
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    from test_d0_8_data_collection import with_data
    ctrl = AssistantController()
    ctrl.project = C.build(with_data(modes=['mock', 'isaac']))
    w = SetupWizard(ctrl)
    page = w.generate_page
    page.initializePage()
    page.project_name.setText(ctrl.project.project_name)
    page.output_dir.setText(str(tmp_path))
    page.generate()
    text = page.result.toPlainText()
    assert 'Generated' in text, text
    assert ('Data recording (isaac, mock): < Back to Step 7 -> Data -> Capture -> Start recorder (no build '
            'needed), or start the bundle: its bring-up runs it.') in text


# ---- owner acceptance flow 2 findings (D0.8) ---------------------------------------------------------------

def test_a_step_gets_keeps_and_loses_its_id_in_the_tasks_inspector(qapp):
    """The trigger dialog lists only steps with an id; the GUI must be able to give one (owner, flow 2)."""
    import test_i0_7_tasks as T
    c = T.composer(T.SAMPLE)
    main = c.program.main_task
    panel = T._panel(c, main)
    panel.select((2,))
    assert panel.step_id.text() == ''
    panel.step_id.setText('approach_w1')
    assert panel.apply()
    assert c.step(main, (2,))['id'] == 'approach_w1' and c.can_undo
    panel.select((2,))
    assert panel.step_id.text() == 'approach_w1'
    panel.move_profile.setCurrentText('approach')                # another edit keeps the id
    assert panel.apply() and c.step(main, (2,))['id'] == 'approach_w1'
    panel.select((3,))
    panel.step_id.setText('approach_w1')                         # unique in its task: refused, nothing changed
    before = c.step(main, (3,))
    assert not panel.apply() and c.step(main, (3,)) == before
    panel.select((2,))
    panel.step_id.setText('')                                    # cleared: removed
    assert panel.apply() and 'id' not in c.step(main, (2,))
    c.undo()
    assert c.step(main, (2,))['id'] == 'approach_w1'


def test_a_silent_camera_is_reported_before_a_capture_fails(qapp, monkeypatch):
    """Flow 2: Isaac's image publishers stopped; the recorder was ready and every capture timed out. The
    recorder's own frame counters (diagnostics) now say so in the Session box."""
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)
    counts = {'color': 0, 'depth': 0}

    class Link(FakeLink):
        def status(self, max_age_s=None):
            return {'level': 0, 'backend': 'isaac',
                    'values': {'session_state': 'IDLE', 'received/cell_camera/color': str(counts['color']),
                               'received/cell_camera/depth': str(counts['depth'])}}
    panel._link = Link()
    clock = [100.0]
    monkeypatch.setattr(P.time, 'monotonic', lambda: clock[0])
    counts.update(color=10, depth=10)
    panel._refresh_session()
    assert panel.session_warnings.text() == ''
    clock[0] += P.FRAMES_SILENT_S + 0.5
    counts['color'] = 40                                         # colour flows, depth is silent
    panel._refresh_session()
    text = panel.session_warnings.text()
    assert 'No camera frames' in text and 'cell_camera/depth' in text and 'cell_camera/color' not in text
    counts.update(color=0, depth=0)                              # a new session resets the counters: a new watch
    panel._refresh_session()
    assert panel.session_warnings.text() == ''


def test_the_inspector_forgets_a_deleted_step(qapp):
    """Flow 2: after Delete, the Inspector still showed the deleted step's form (Id approach_w2) as if editable."""
    import test_i0_7_tasks as T
    c = T.composer(T.SAMPLE)
    panel = T._panel(c, c.program.main_task)
    panel.select((2,))
    panel.step_id.setText('approach_w1')
    assert panel.apply()
    panel.select((2,))
    assert panel.delete_selected()
    assert not panel.inspector.isEnabled() and panel.step_id.text() == '' and panel.inspector.title() == 'Inspector'
    assert not panel.apply()                                     # nothing stale left to apply


def test_a_sync_failure_is_said_in_product_words(qapp, monkeypatch):
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch)

    class Link(FakeLink):
        def status(self, max_age_s=None):
            return {'level': 1, 'backend': 'isaac', 'values': {'session_state': 'RECORDING', 'sync_failures': '1'}}
    panel._link = Link()
    panel._refresh_session()
    text = panel.session_warnings.text()
    assert 'sync failures' not in text
    assert text.startswith('1 capture(s) with streams missing or not taken at the same moment: kept as Partial')
    assert 'Datasets -> Inspect' in text


# the two records of the owner's flow 2 session ses-20261104T163012Z-fe31fad1, trimmed to what Inspect reads
FLOW2_CAPTURES = [
    {'seq': 1, 'outcome': 'PARTIAL', 'missing': [], 'trigger': {'trigger_id': 'at_w2'},
     'sync': {'status': 'OUT_OF_TOLERANCE', 'max_skew_ns': 50000002, 'tolerance_ns': 16666667}},
    {'seq': 2, 'outcome': 'COMPLETE', 'missing': [], 'trigger': {'trigger_id': 'at_w2'},
     'sync': {'status': 'SYNCED', 'max_skew_ns': 0, 'tolerance_ns': 16666667}},
    {'seq': 3, 'outcome': 'FAILED', 'trigger': {'trigger_id': 'manual'}, 'sync': None,
     'missing': [{'stream': 'cell_camera/color', 'reason': 'TIMEOUT'},
                 {'stream': 'cell_camera/depth', 'reason': 'TIMEOUT'}]},
]


def test_capture_problems_say_why_a_capture_is_not_complete():
    assert D.capture_problems(FLOW2_CAPTURES) == [
        '#1 PARTIAL (at_w2): streams taken 50 ms apart (allowed 17 ms) [OUT_OF_TOLERANCE]',
        '#3 FAILED (manual): cell_camera/color: TIMEOUT; cell_camera/depth: TIMEOUT']
    assert D.capture_problems(FLOW2_CAPTURES[1:2]) == []


def test_inspect_lists_the_captures_that_are_not_complete_and_why(qapp, monkeypatch):
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, cli = _datasets(qapp, monkeypatch)
    sid = SESSIONS[0]['session_id']
    header = {'status': 'CLOSED', 'source': {'transport': 'LIVE_ROS', 'backend': 'real'},
              'sensors': [{'sensor_id': 'bin_camera'}], 'counters': {'captures': {'attempted': 3}}}
    cli.answers[f'inspect {sid}'] = (True, {'session': header, 'captures': FLOW2_CAPTURES, 'captures_total': 3}, None)
    shown = []
    monkeypatch.setattr(P.QMessageBox, 'information', lambda *a, **k: shown.append(a[2]))
    panel.session_table.selectRow(2)
    panel.inspect_selected()
    _wait(panel)
    assert ['inspect', sid, '--captures', '--limit', str(P.INSPECT_CAPTURES_READ)] in cli.calls
    assert 'Not complete, and why:\n  #1 PARTIAL (at_w2): streams taken 50 ms apart (allowed 17 ms)' in shown[0]
    assert '#2' not in shown[0] and 'checked)' not in shown[0]


# ---- owner flow 4 preparation: the label sets an export takes ------------------------------------------------

def _export_with(qapp, monkeypatch, label_sets, dialog=None):
    """Export v1 whose label sets are ``label_sets``; ``dialog(widget)`` sets the label dialog before OK."""
    from trainit_setup_assistant.gui.composer import data_panel as P
    panel, cli = _datasets(qapp, monkeypatch, versions=[1])
    _wait(panel)
    panel._version_detail[1] = dict(VERSION, content=dict(VERSION['content'], label_sets=label_sets))
    asked = []
    monkeypatch.setattr(P.QFileDialog, 'getExistingDirectory', lambda *a, **k: asked.append(1) or '/tmp/out')

    def exec_(self):
        if dialog is not None:
            dialog(self)
        return P.QDialog.Accepted
    monkeypatch.setattr(P.QDialog, 'exec_', exec_)
    panel.export_yolo()
    _wait(panel)
    return panel, [c for c in cli.calls if c[:2] == ['export', 'yolo']], asked


SYNTH = {'label_set_id': 'ls-s', 'kind': 'BBOX_2D', 'source': 'SYNTHETIC_GROUND_TRUTH', 'entries_selected': 48}
MASKS = {'label_set_id': 'ls-m', 'kind': 'INSTANCE_MASK', 'source': 'SYNTHETIC_GROUND_TRUTH', 'entries_selected': 48}
IMPORTED = {'label_set_id': 'ls-i', 'kind': 'BBOX_2D', 'source': 'IMPORTED', 'entries_selected': 4}


def test_a_version_without_box_labels_is_said_before_any_folder_is_asked(qapp, monkeypatch):
    """Owner flow 3: d08_live v2 has recorded sessions only; trainit_data would refuse (EXPORT_REFUSED)."""
    panel, calls, asked = _export_with(qapp, monkeypatch, [MASKS])
    assert not calls and not asked
    assert panel.dataset_result.text().startswith('bin_parts v1 has no box labels to export.')


def test_box_labels_of_one_origin_export_as_they_are(qapp, monkeypatch):
    panel, calls, asked = _export_with(qapp, monkeypatch, [SYNTH, MASKS])
    assert calls == [['export', 'yolo', 'bin_parts', '1', '/tmp/out']]


def _tick(text, value):
    from trainit_setup_assistant.gui.composer import data_panel as P

    def set_(dialog):
        for box in dialog.findChildren(P.QCheckBox):
            if box.text().startswith(text):
                box.setChecked(value)
    return set_


def test_mixed_origins_need_a_choice_or_an_explicit_mix(qapp, monkeypatch):
    """d07_mixed_real_synthetic v1: synthetic ground truth + imported boxes; trainit_data refuses an unasked mix."""
    panel, calls, asked = _export_with(qapp, monkeypatch, [SYNTH, MASKS, IMPORTED])
    assert not calls and not asked and 'without allowing the mix' in panel.dataset_result.text()
    panel, calls, _ = _export_with(qapp, monkeypatch, [SYNTH, MASKS, IMPORTED], _tick('imported', False))
    assert calls == [['export', 'yolo', 'bin_parts', '1', '/tmp/out', '--label-set', 'ls-s']]
    panel, calls, _ = _export_with(qapp, monkeypatch, [SYNTH, MASKS, IMPORTED], _tick('Allow labels', True))
    assert calls == [['export', 'yolo', 'bin_parts', '1', '/tmp/out', '--label-set', 'ls-s', '--label-set', 'ls-i',
                      '--allow-mixed-provenance']]


def test_a_new_dataset_appears_and_is_chosen_without_a_refresh(qapp, monkeypatch):
    """Owner acceptance 2026-11-05 (flows 3 and 5): after New dataset… the list kept the old datasets until
    Refresh. The answer callback read the dialog's Id field after Qt had deleted the closed dialog; the error was
    swallowed. Driven here with the REAL dialog, closed by a real OK click (no ``exec_`` stand-in)."""
    from python_qt_binding.QtCore import QTimer
    from trainit_setup_assistant.gui.composer import data_panel as P
    cli = datasets_cli()
    created = []
    cli.answers['dataset create'] = (True, lambda a: created.append(a[2]) or {'dataset': {'dataset_id': a[2]}},
                                     None)
    cli.answers['dataset list'] = (True, lambda a: {'datasets': [
        {'dataset_id': d, 'name': d, 'classification': 'INTERNAL', 'versions': []}
        for d in ['bin_parts'] + created]}, None)
    panel, ctrl, host, _ = make_panel(qapp, monkeypatch, cli=cli)
    panel.show()
    panel.tabs.setCurrentIndex(2)
    panel.refresh()
    _wait(panel)
    _wait(panel)

    def fill():
        dialog = qapp.activeModalWidget()
        ident, name = dialog.findChildren(P.QLineEdit)[:2]
        ident.setText('d08_synthetic')
        name.setText('D0.8 synthetic acceptance')
        dialog.findChild(P.QDialogButtonBox).button(P.QDialogButtonBox.Ok).click()
    QTimer.singleShot(0, fill)
    panel.create_dataset()
    _wait(panel)
    _wait(panel)
    assert panel.jobs.errors == []
    assert ['dataset', 'create', 'd08_synthetic', '--name', 'D0.8 synthetic acceptance',
            '--classification', 'INTERNAL'] in cli.calls
    assert panel.dataset_combo.currentData() == 'd08_synthetic'
    assert panel.dataset_result.text() == 'Dataset d08_synthetic created.'


def test_a_failing_answer_is_said_not_swallowed(qapp, monkeypatch, capsys):
    panel, ctrl, host, cli = make_panel(qapp, monkeypatch)

    def broken(result):
        raise RuntimeError('wrapped C/C++ object has been deleted')
    panel.cli(['dataset', 'list'], broken)
    _wait(panel)
    assert panel.jobs.errors and 'deleted' in panel.jobs.errors[0]
    assert 'Internal error in the Data panel' in panel.message.text() and 'Refresh' in panel.message.text()
    assert 'RuntimeError' in capsys.readouterr().err                # the traceback is in the terminal
