"""vNext I0.9 (I0_DESIGN.md §21): the ONE validator's verdict, located and actionable in the Composer.

- The canonical text is shown unchanged; a deterministic "did you mean …?" is added for a misspelt name
  (the owner's M3 Test A: ``release at 'plce'`` -> "Did you mean 'place'?").
- Go to opens the element a line names: the task step by its label, a waypoint, a path, a profile.
- The verdict follows every edit.
"""
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import ProgramRefusals, plan_program
from trainit_setup_assistant.composer import Composer, parse
from trainit_setup_assistant.composer.diagnostics import suggestion
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'


@pytest.fixture(autouse=True)
def _ur5e_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def _misspelt(tmp_path, mutate) -> Path:
    data = yaml.safe_load(COMPOSITION.read_text())
    mutate(data['application'])
    data['robot']['description']['urdf_dir'] = str((COMPOSITION.parent / data['robot']['description']['urdf_dir'])
                                                   .resolve())
    out = tmp_path / 'project.yaml'
    out.write_text(yaml.safe_dump(data, sort_keys=False))
    return out


def _release_plce(app):
    for t in app['tasks']:
        for step in t['steps']:
            if 'release' in step:
                step['release']['at'] = 'plce'


def test_the_owners_test_a_misspelling_gets_a_did_you_mean(tmp_path):
    path = _misspelt(tmp_path, _release_plce)
    project = CanonicalProject.parse_obj(yaml.safe_load(path.read_text()))
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(resolve_project_paths(project, path)[0])
    line = next(p for p in exc.value.problems if "'plce'" in p)
    d = parse(line)
    assert d.location.kind == 'step' and d.location.name == 'transfer'
    assert suggestion(d, Composer(project).document(), project) == "Did you mean 'place'?"


@pytest.mark.parametrize('text, hint', [
    ("R-M3-03: task 't' step 1 (path 'sweap'): no path is named 'sweap' (paths: sweep)", None),
    ("R-M3-03: task 't' step 1 (task 'transfr'): no task is named 'transfr'", "Did you mean 'transfer'?"),
    ("R-M3-03: task 't' step 1 (move ptp to 'home'): profile 'transitt' is not a motion profile of this program",
     "Did you mean 'transit'?"),
    ("R-M3-03: task 't' step 1 (move ptp to 'xyz'): no waypoint is named 'xyz'", None),   # nothing close
])
def test_a_hint_only_when_a_close_name_exists(text, hint):
    project = CanonicalProject.parse_obj(yaml.safe_load(COMPOSITION.read_text()))
    assert suggestion(parse(text), Composer(project).document(), project) == hint


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _shell(path):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(path)
    w = SetupWizard(ctrl)
    w.blocks_page.initializePage()
    _shell.keep = w                                  # the wizard owns the widgets: keep it alive
    return ctrl, w.blocks_page.program_panel


def test_the_composer_shows_the_verdict_and_goes_to_the_step(qapp, tmp_path):
    ctrl, shell = _shell(_misspelt(tmp_path, _release_plce))
    panel = shell.diagnostics
    assert 'error' in panel.summary.text()
    texts = [panel.list.item(i).text() for i in range(panel.list.count())]
    line = next(t for t in texts if "'plce'" in t)
    assert "→ Did you mean 'place'?" in line
    i = texts.index(line)
    panel.list.setCurrentRow(i)
    panel.go_to_selected()
    assert shell.stack.currentWidget() is shell.tasks and shell.tasks.task == 'transfer'
    label = panel.diagnostics[i].location.label
    assert shell.tasks.selected_labels() == [label]
    # fix it in the inspector: the verdict follows the edit
    form = shell.tasks.step_forms['release']
    form.at.setCurrentText('place')
    assert shell.tasks.apply()
    assert not any("'plce'" in panel.list.item(k).text() for k in range(panel.list.count()))


def test_go_to_a_waypoint_a_path_and_a_profile(qapp, tmp_path):
    from trainit_setup_assistant.composer.diagnostics import Location
    ctrl, shell = _shell(COMPOSITION)
    doc = shell.composer().document()
    wp = doc['waypoints'][1]['name']
    shell.go_to(parse(f"R-M3-05: waypoint '{wp}': x"))
    assert shell.stack.currentWidget() is shell.waypoints and shell.waypoints.selected == wp
    prof = next(iter(doc['motion_profiles']))
    shell.go_to(parse(f"R-M3-12: motion profile '{prof}': x"))
    assert shell.stack.currentWidget() is shell.profiles and shell.profiles.selected == prof
    shell.go_to(parse('R-M3-15: task call cycle: a -> b -> a'))
    assert shell.stack.currentWidget() is shell.overview
    assert Location().kind == 'project'
