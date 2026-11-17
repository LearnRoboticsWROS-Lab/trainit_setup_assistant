"""M4.5d owner acceptance (2026-11-17): the frictions the owner met in the real TSA, fixed at the source.

    F1  Step 5 dressed a project-file detector (SurfaceGrasp) in the colour-mask form; "Save detector" overwrote it
    F2  Step 6 pre-selected a template ("Simple motion") for a project that already has its program
    F3  Step 7 rendered an unbounded repeat as "Repeat x-1"
    F5  the generated YAML's BYTES depended on in-memory sharing (anchors from the CLI, none from the GUI)
    F8  the run's OUTCOME did not say what the operator does next; the run panel did not show it

(F4 and F6 live in the private cell repository; F7 is P0 scope.)
"""
import copy
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import OPERATOR_TEXT, OUTCOMES
from trainit_setup_assistant.composer.describe import step_text
from trainit_setup_assistant.composer.launch_plan import BT_FAILURE
from trainit_setup_assistant.generator.determinism import canonical_yaml, generated_yaml
from trainit_setup_assistant.model import CanonicalProject

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'
M3_6 = HERE / 'data' / 'm3_6'
PROVIDER = {
    'name': 'grasp_provider', 'method': 'surface_grasp', 'continuous': False, 'rate_hz': 5.0,
    'segmentation': {'method': 'color_mask', 'params': {'class_id': 'part', 'max_objects': 10}},
    'params': {'eef': 'main', 'site': 'main', 'contact_radius_m': 0.009, 'approach_cone_deg': 45.0,
               'standoff_m': 0.05}}
MASK = {'name': 'red_parts', 'method': 'color_mask', 'continuous': True, 'rate_hz': 10.0,
        'params': {'class_id': 'red', 'h': [170, 10], 's': [40, 255], 'v': [60, 255]}}


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _provider_project(monkeypatch) -> CanonicalProject:
    """The M4 SurfaceGrasp project (the public UR5e sample cell) plus a colour mask beside the provider."""
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    monkeypatch.chdir(M3_6)
    data = yaml.safe_load((M3_6 / 'validation_c.yaml').read_text())
    data['perception']['detectors'] = [copy.deepcopy(PROVIDER), copy.deepcopy(MASK)]
    next(s['grasp'] for s in data['application']['tasks'][0]['steps'] if 'grasp' in s)['max_tilt_deg'] = 60
    return CanonicalProject.parse_obj(data)


# ---- F1 ---------------------------------------------------------------------------------------------------------

def test_f1_a_project_file_detector_is_shown_read_only_and_never_overwritten(qapp, monkeypatch):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import PerceptionPage
    ctrl = AssistantController(_provider_project(monkeypatch))
    before = ctrl.project.perception.detector_by_name('grasp_provider').dict()
    page = PerceptionPage(ctrl)
    page.initializePage()
    page._refresh_list(select='grasp_provider')
    page._on_pick_detector('grasp_provider')
    # its OWN method is shown, the colour-mask form is locked, the page says where to edit it
    assert page.METHODS[page.method.currentIndex()][0] == 'surface_grasp'
    assert not page.save_btn.isEnabled() and not page.params_box.isEnabled()
    assert page.readonly_note.isVisibleTo(page) and 'project file' in page.readonly_note.text()
    # a Save (or an Add under its name) is refused, never a colour mask under the provider's name
    page._save_detector()
    page.det_name.setText('grasp_provider')
    page._add_detector()
    assert page._detectors['grasp_provider']['method'] == 'surface_grasp'
    assert page.validatePage()
    assert ctrl.project.perception.detector_by_name('grasp_provider').dict() == before
    page.initializePage()
    # removing it leaves the editable form, not a locked page (review)
    page._refresh_list(select='grasp_provider')
    page._remove_detector()
    assert page.save_btn.isEnabled() and not page.readonly_note.isVisibleTo(page)
    # a colour mask is editable again
    page._on_pick_detector('red_parts')
    assert page.METHODS[page.method.currentIndex()][0] == 'color_mask'
    assert page.save_btn.isEnabled() and page.params_box.isEnabled() and not page.readonly_note.isVisibleTo(page)


# ---- F2 ---------------------------------------------------------------------------------------------------------

def test_f2_an_existing_program_preselects_no_template(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(SAMPLE)
    ctrl.apply_template('machine_tending')                   # a project that HAS its task program
    program = copy.deepcopy(ctrl.project.application.program.dict())
    w = SetupWizard(ctrl)
    page = w.application_page
    page.initializePage()
    assert page.template.currentData() is None and page.template.currentIndex() == 0
    assert not page.restart_button.isEnabled()
    assert 'already has its program' in page.template_text.text()
    page.restart_from_template()                             # the placeholder applies nothing
    page.restart_from_template()
    assert page.validatePage()
    assert ctrl.project.application.program.dict() == program
    # choosing a real template arms the (two-click) restart, as before
    page.template.setCurrentIndex(page.template.findData('inspection'))
    assert page.restart_button.isEnabled() and page.template_text.text().startswith("Template '")


def test_f2_a_new_task_program_still_starts_from_the_chosen_template(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(SAMPLE)
    w = SetupWizard(ctrl)
    page = w.application_page
    page.initializePage()
    assert page.template.itemData(0) is not None            # no placeholder outside an existing program
    page.app_type.setCurrentIndex([t for t, _, _ in page.APPS].index('task_program'))
    page.template.setCurrentIndex(page.template.findData('machine_tending'))
    assert page.validatePage() and ctrl.project.application.program.main_task == 'tend'


# ---- F3 ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('step, text', [
    ({'repeat': {'cycles': -1, 'until': 'nothing_to_pick', 'body': []}}, 'Repeat until nothing to pick'),
    ({'repeat': {'cycles': -1, 'body': []}}, 'Repeat until stopped'),
    ({'repeat': {'cycles': 3, 'body': []}}, 'Repeat ×3'),
])
def test_f3_an_unbounded_repeat_is_never_shown_as_a_count(step, text):
    assert step_text(step) == text


# ---- F5 ---------------------------------------------------------------------------------------------------------

def test_f5_generated_yaml_has_no_anchors_whatever_the_sharing():
    roi = [[328, 162], [518, 162]]
    shared = {'a': {'roi_px': roi}, 'b': {'roi_px': roi}}
    copied = {'a': {'roi_px': copy.deepcopy(roi)}, 'b': {'roi_px': copy.deepcopy(roi)}}
    for dump in (lambda d: generated_yaml(d, sort_keys=False), canonical_yaml):
        text = dump(shared)
        assert '&id' not in text and '*id' not in text
        assert text == dump(copied)                          # same content -> same bytes
    assert yaml.safe_dump(shared) != yaml.safe_dump(copied)  # what PyYAML does by default (the defect)


def test_f5_perception_yaml_is_byte_identical_for_shared_and_copied_detector_data(monkeypatch):
    """The M4.5d case: data the detectors hold in common (there, one ``roi_px`` polygon) is ONE object when the
    project is parsed from a YAML with anchors (the CLI) and a copy per detector after the GUI re-applies them."""
    from trainit_setup_assistant.generator.perception_yaml import build_perception_yaml
    project = _provider_project(monkeypatch)
    second = project.perception.detectors[0].copy(deep=True)
    second.name = 'grasp_provider_b'
    project.perception.detectors.append(second)
    hue = [170, 10]
    shared, copied = project.copy(deep=True), project.copy(deep=True)
    for d in shared.perception.detectors:
        if d.segmentation:
            d.segmentation['params']['h'] = hue                    # one list object held twice
    for d in copied.perception.detectors:
        if d.segmentation:
            d.segmentation['params']['h'] = copy.deepcopy(hue)     # a copy each
    text = build_perception_yaml(shared)
    assert text.count('h: [170, 10]') >= 2 and '&id' not in text and '*id' not in text
    assert text == build_perception_yaml(copied)


# ---- F8 ---------------------------------------------------------------------------------------------------------

def test_f8_every_outcome_says_what_the_operator_does_next():
    assert set(OPERATOR_TEXT) == set(OUTCOMES)
    for code in ('PICKABLE_EXHAUSTED', 'RECOVERY_EXHAUSTED', 'TERMINAL_FAULT'):
        assert OPERATOR_TEXT[code].startswith('OPERATOR ACTION:')
    assert 'wall' in OPERATOR_TEXT['PICKABLE_EXHAUSTED']


def test_f8_the_run_panel_shows_the_outcome_and_the_operator_action(qapp):
    from trainit_setup_assistant.gui.launcher import Launcher
    from trainit_setup_assistant.gui.run_panel import RunPanel
    panel = RunPanel(Launcher(), [('run', 'Run', lambda: None)])
    panel.line('run', f"[trainit_run_bt-5] [INFO] [r3.trainit_run_bt]: [BT] OUTCOME: PICKABLE_EXHAUSTED -- "
                      f"{OPERATOR_TEXT['PICKABLE_EXHAUSTED']}")
    panel.line('run', f'[trainit_run_bt-5] [INFO] [r3.trainit_run_bt]: {BT_FAILURE}')
    shown = panel.state.text()
    assert BT_FAILURE in shown and 'OUTCOME: PICKABLE_EXHAUSTED' in shown and 'OPERATOR ACTION:' in shown
