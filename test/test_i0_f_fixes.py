"""Owner Test F.1's friction rows 34, 35, 37, 38 and 39 (TSA part), fixed after the test.

34 -- the Route ignored that a grasp / release MOVES the robot ("5 home -> home").
35 -- an actuator reached through its site (a grasp, release, close / open) showed "Used by --".
37 -- "Check the end effector (live)" sat after a long scroll.
38 -- a grasp, and no object of the cell scene a grasp target: nothing attaches in the planning scene (R-I0-06).
39 -- Step 2 always showed "attach_box" and saved the grasp handling only with "Apply mapping".
"""

import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import Composer, eef_ops
from trainit_setup_assistant.composer.advice import configuration_advice
from trainit_setup_assistant.composer.describe import route
from trainit_setup_assistant.composer.diagnostics import parse
from trainit_setup_assistant.composer.templates import apply_template
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.io import project_to_dict

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'


@pytest.fixture(autouse=True)
def _sample_paths(monkeypatch):
    monkeypatch.chdir(SAMPLE.parent)


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def pick_and_place() -> Composer:
    """The sample (a suction short form, mock) with the Pick and place template and an Open gripper first."""
    c = Composer(apply_template(CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text())), 'pick_and_place'))
    c.edit(lambda d: d['tasks'][0]['steps'].insert(1, {'tool': {'open': {}}}))
    return c


def test_the_route_follows_the_grasp_and_the_release():
    lines = route(pick_and_place().document()['tasks'][0])
    assert lines == [
        '1  (where the task starts) → home : PTP · transit · STOP',
        '3  home → pick : grasp · PTP 150 mm above it · LIN in · close · LIN out',
        '4  pick (retracted) → place : release · PTP 150 mm above it · LIN in · open · LIN out',
        '5  place (retracted) → home : PTP · transit · STOP']


def test_a_detected_grasp_reads_its_candidate():
    task = {'steps': [{'grasp': {'candidates': 'parts', 'index': 0, 'standoff_m': 0.05}}]}
    assert route(task) == ['1  (where the task starts) → candidate 0 of parts : grasp · PTP 50 mm above it · '
                           'LIN in · close · LIN out']
    task = {'steps': [{'grasp': {'index': 1}}]}                      # the task's latest detection
    assert route(task)[0].startswith('1  (where the task starts) → candidate 1 of the latest detection : grasp')


def test_the_actuator_is_used_through_its_site():
    users = eef_ops.actuator_users(pick_and_place(), 'vacuum')
    assert users == ["task 'pick_and_place' step 2 (site 'main')", "task 'pick_and_place' step 3 (site 'main')",
                     "task 'pick_and_place' step 4 (site 'main')"]


def test_a_grasp_without_a_grasp_target_in_the_scene_is_a_warning_at_the_grasp():
    c = pick_and_place()
    warnings = [w for w in configuration_advice(c.project) if w.startswith('R-I0-06')]
    assert warnings == ["R-I0-06: task 'pick_and_place' step 3: a grasp, and no object of the cell scene is a grasp "
                        'target: nothing attaches to the tool in the planning scene (RViz does not show the part '
                        "held, the planner ignores it) -- tick 'grasp' on the part you pick (Step 2)"]
    d = parse(warnings[0])
    assert d.code == 'R-I0-06' and d.location.kind == 'step'
    data = project_to_dict(c.project)
    data['scene']['objects'].append({'id': 'part', 'shape': 'box', 'dims': [0.02, 0.02, 0.02],
                                     'position': [0.5, 0.0, 0.01], 'category': 'dynamic', 'dynamic': True,
                                     'grasp_target': True})
    assert not [w for w in configuration_advice(CanonicalProject.parse_obj(data)) if w.startswith('R-I0-06')]


def test_the_end_effector_live_check_is_above_the_scroll(qapp):
    from python_qt_binding.QtWidgets import QScrollArea
    from trainit_setup_assistant.gui.composer.eef_panel import EndEffectorPanel
    c = pick_and_place()
    panel = EndEffectorPanel(lambda: c)
    panel.refresh()
    scroll = next(s for s in panel.findChildren(QScrollArea) if s.widget().isAncestorOf(panel.actuators))
    assert panel.check_button.isVisibleTo(panel)
    assert not scroll.widget().isAncestorOf(panel.check_button)
    rows = {panel.actuators.item(0, 0).text(): panel.actuators.item(0, 4).text()}
    assert rows == {'vacuum': "task 'pick_and_place' step 2 (site 'main'), task 'pick_and_place' step 3 (site "
                              "'main'), task 'pick_and_place' step 4 (site 'main')"}


def test_step_2_shows_and_saves_the_grasp_handling(qapp):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(SAMPLE)
    ctrl.project.scene.grasp_attach_mode = 'remove'
    wizard = SetupWizard(ctrl)                                      # the page's parent, kept alive
    page = wizard.scene_page
    saves = []
    original = ctrl.set_scene_loader_params
    ctrl.set_scene_loader_params = lambda **kw: saves.append(kw) or original(**kw)
    page.initializePage()
    assert page.grasp_mode.currentText() == 'remove'                # as saved, not the default
    assert not [kw for kw in saves if 'grasp_attach_mode' in kw]    # showing it saves nothing
    page.grasp_mode.setCurrentText('attach_box')
    assert ctrl.project.scene.grasp_attach_mode == 'attach_box'     # saved when changed, no "Apply mapping"


# ---- the end-of-I0 polish pass (friction rows 11, 19, 29) ------------------------------------------------

def test_the_polish_pass(qapp):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from python_qt_binding.QtWidgets import QApplication, QHeaderView
    from trainit_setup_assistant.gui.composer.paths_panel import PathsPanel
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(SAMPLE)
    wizard = SetupWizard(ctrl)
    g = QApplication.instance().primaryScreen().availableGeometry()
    assert wizard.width() == min(g.width(), max(1280, int(g.width() * 0.8)))          # row 11: most of the screen
    c = pick_and_place()
    paths = PathsPanel(lambda: c)                                                   # kept alive
    header = paths.table.horizontalHeader()
    assert all(header.sectionResizeMode(col) == QHeaderView.ResizeToContents for col in (1, 2, 3))   # row 19
    page = wizard.application_page
    page.initializePage()
    page._template_changed(max(0, page.template.currentIndex()))
    text = page.template_text.text()                                                 # row 29
    assert text.startswith("Template '") and 'would replace the program' in text and text.endswith('Step 7.')
