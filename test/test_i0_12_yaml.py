"""vNext I0.12 (I0_DESIGN.md §20; owner decision D4): the YAML view, the save, the reload.

- The YAML view shows EXACTLY the text a save writes.
- D4: a GUI save over a file with comments asks first and writes ``<file>.bak`` (byte-identical to the
  original) before overwriting. A "no", or a backup that cannot be written, saves nothing. A file without
  comments saves without a question.
- A hand edit of the file is detected, and "Reload from file" shows it in the GUI. The owner's
  Acceptance H: GUI → save → one YAML edit → reload → the GUI reflects it, with no hidden state.
"""
import os
import shutil
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer.files import has_comments, save_with_backup
from trainit_setup_assistant.model import CanonicalProject, project_yaml, save_project

PKG = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PKG / 'examples' / 'ur5e_suction'


@pytest.fixture
def sample(tmp_path) -> Path:
    """The public sample's commented task program, copied with its urdf (paths stay project-relative)."""
    shutil.copytree(SAMPLE_DIR / 'urdf', tmp_path / 'urdf')
    shutil.copy(SAMPLE_DIR / 'task_program.yaml', tmp_path / 'task_program.yaml')
    return tmp_path / 'task_program.yaml'


@pytest.mark.parametrize('text, expected', [
    ('a: 1  # a comment\n', True), ('# header\na: 1\n', True), ('a: "b#c"\n', False), ('a: b#c\n', False),
    ('a: 1\nb: [1, 2]\n', False), ('  # indented\n', True), ("a: it's fine # c\n", True),
    ("description: don't stop  # note\n", True), ('a:\t# tab\n', True),
    ("a: 'x # y'\n", True),                              # prudent: a false "yes" only asks and backs up
])
def test_comments_are_detected_prudently(text, expected):
    assert has_comments(text) is expected


def test_the_yaml_view_is_exactly_what_a_save_writes(sample, tmp_path):
    p = CanonicalProject.parse_obj(yaml.safe_load(sample.read_text()))
    out = save_project(p, tmp_path / 'out.yaml')
    assert out.read_text() == project_yaml(p)


def test_d4_a_commented_file_is_kept_as_bak_before_the_save(sample):
    original = sample.read_bytes()
    assert has_comments(original.decode())
    p = CanonicalProject.parse_obj(yaml.safe_load(original))
    asked = []
    saved, backup, msg = save_with_backup(p, sample, lambda q: asked.append(q) or False)
    assert not saved and backup is None and sample.read_bytes() == original and asked     # "no": nothing
    saved, backup, msg = save_with_backup(p, sample, lambda q: True)
    assert saved and backup == sample.with_name('task_program.yaml.bak')
    assert backup.read_bytes() == original                                              # the only copy kept
    assert not has_comments(sample.read_text()) and '.bak' in msg
    asked.clear()
    saved, backup, _ = save_with_backup(p, sample, lambda q: asked.append(q) or True)    # no comments now
    assert saved and backup is None and asked == []


def test_d4_a_backup_that_cannot_be_written_aborts_the_save(sample, monkeypatch):
    import trainit_setup_assistant.composer.files as files
    original = sample.read_bytes()

    def full_disk(*_a, **_k):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(files.shutil, 'copyfile', full_disk)
    p = CanonicalProject.parse_obj(yaml.safe_load(original))
    saved, backup, msg = save_with_backup(p, sample, lambda q: True)
    assert not saved and 'backup' in msg and sample.read_bytes() == original


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_acceptance_h_gui_save_yaml_edit_reload(qapp, sample):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(sample)
    w = SetupWizard(ctrl)
    w.blocks_page.initializePage()
    shell = w.blocks_page.program_panel
    # GUI edit -> save (the commented original kept as .bak, D4)
    shell.select('Waypoints')
    assert shell.waypoints.new_waypoint('TCP', 'from_gui')
    shell.select('YAML')
    shell.yaml._confirm = lambda q: True
    assert 'from_gui' in shell.yaml.text.toPlainText()
    assert shell.yaml.save() and sample.with_name('task_program.yaml.bak').is_file()
    assert not ctrl.file_changed_on_disk()
    # one supported hand edit of the YAML: a profile's speed
    data = yaml.safe_load(sample.read_text())
    data['application']['motion_profiles']['transit']['speed'] = 42
    sample.write_text(yaml.safe_dump(data, sort_keys=False))
    shell.yaml.refresh()
    assert ctrl.file_changed_on_disk() and 'changed on disk' in shell.yaml.changed.text()
    assert shell.yaml.reload()
    assert ctrl.project.application.program.motion_profiles['transit'].speed == 42
    shell.select('Profiles')
    row = {shell.profiles.table.item(i, 0).text(): shell.profiles.table.item(i, 1).text()
           for i in range(shell.profiles.table.rowCount())}
    assert row['transit'] == '42 %'                                  # the GUI shows the file's model
    assert 'from_gui' in [wp.name for wp in ctrl.project.application.program.waypoints]


def test_a_hand_edit_the_model_refuses_keeps_the_last_valid_project(qapp, sample):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.composer.yaml_panel import YamlPanel
    ctrl = AssistantController()
    ctrl.open_project(sample)
    before = ctrl.project
    panel = YamlPanel(ctrl)
    sample.write_text(sample.read_text().replace('type: ptp', 'type: warp', 1))
    assert not panel.reload() and 'cannot be loaded' in panel.message.text()
    assert ctrl.project is before


def test_d4_a_hand_edit_since_the_read_is_also_asked_and_kept(sample, tmp_path):
    p = CanonicalProject.parse_obj(yaml.safe_load(sample.read_text()))
    save_project(p, sample)                                       # no comments any more
    edited = sample.read_text().replace('speed: 50', 'speed: 51', 1)
    sample.write_text(edited)
    asked = []
    saved, backup, _ = save_with_backup(p, sample, lambda q: asked.append(q) or True, changed_on_disk=True)
    assert saved and asked and 'changed on disk' in asked[0] and backup.read_text() == edited


def test_a_backup_never_overwrites_another(sample):
    p = CanonicalProject.parse_obj(yaml.safe_load(sample.read_text()))
    first_original = sample.read_bytes()
    _, b1, _ = save_with_backup(p, sample, lambda q: True)
    sample.write_text('# a new hand comment\n' + sample.read_text())
    second_original = sample.read_bytes()
    _, b2, _ = save_with_backup(p, sample, lambda q: True)
    assert b1 != b2 and b1.read_bytes() == first_original and b2.read_bytes() == second_original


def test_reload_asks_before_discarding_unsaved_edits(qapp, sample):
    from trainit_setup_assistant.composer import Composer
    from trainit_setup_assistant.gui.composer.yaml_panel import YamlPanel
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController()
    ctrl.open_project(sample)
    c = Composer(ctrl.project, on_change=lambda pr: setattr(ctrl, 'project', pr))
    c.rename_waypoint(c.waypoint_names()[1], 'edited_here')
    assert ctrl.has_unsaved_changes()
    panel = YamlPanel(ctrl, confirm=lambda q: False)
    assert not panel.reload() and 'edits are kept' in panel.message.text()
    assert 'edited_here' in [w.name for w in ctrl.project.application.program.waypoints]
    panel._confirm = lambda q: True
    assert panel.reload() and 'edited_here' not in [w.name for w in ctrl.project.application.program.waypoints]


def test_a_project_never_saved_is_saved_where_the_user_chooses(qapp, sample, tmp_path):
    """Owner Test B: a TSA restart (to pick up a fix) lost the waypoints taught at Step 7, because "Save project"
    only pointed to Step 8. A project never saved asks where, and the file becomes the project's file."""
    from trainit_setup_assistant.gui.composer.yaml_panel import YamlPanel
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.model.io import load_project
    ctrl = AssistantController(load_project(sample))
    ctrl.project_path = None
    panel = YamlPanel(ctrl, confirm=lambda q: True)
    panel._choose_path = lambda: None
    assert not panel.save() and 'no file chosen' in panel.message.text()
    target = tmp_path / 'taught.yaml'
    panel._choose_path = lambda: str(target)
    assert panel.save() and target.is_file()
    assert ctrl.project_path == str(target) and str(target) in panel.where.text()
    assert load_project(target).application.program is not None
