"""vNext I0.10 (I0_DESIGN.md §19): start from a template, not from blank.

- Every installed template is pure canonical data. Applied to a robot project (the public sample cell),
  it validates with ``plan_program`` once the project provides what the template names (a detector for
  the camera templates), and it generates.
- Templates are customer-neutral.
- Step 6 offers "Task program (Application Composer)" from a template; Step 7 then opens the Composer
  on it.
"""
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import plan_program
from trainit_setup_assistant.composer import Composer
from trainit_setup_assistant.composer.templates import (TEMPLATE_DIR, apply_template, list_templates, missing,
                                                        requirements)
from trainit_setup_assistant.model import CanonicalProject, project_to_dict
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'project.yaml'          # a template application project
DECOMPOSITIONS = PKG / 'test' / 'data' / 'm3_7' / 'decompositions'
IDS = ['simple_motion', 'pick_and_place', 'machine_tending', 'inspection', 'process_path', 'bin_picking']


@pytest.fixture(autouse=True)
def _ur5e_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def sample() -> CanonicalProject:
    return CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text()))


def test_the_catalogue_lists_the_six_templates_in_order():
    assert [t.id for t in list_templates()] == IDS
    assert all(t.title and t.description for t in list_templates())


def test_templates_are_customer_neutral():
    """No program, product or vendor of a real cell: a programme code (letters then three or four digits,
    e.g. ``abc1234``), the reference cell's own words. The customer names are never spelt here: this file is
    published."""
    import re
    code = re.compile(r'\b[a-z]{2,4}[ _-]?\d{3,4}\b')
    for f in TEMPLATE_DIR.glob('*.yaml'):
        text = f.read_text().lower()
        assert not code.search(text), (f.name, code.search(text).group(0))
        for word in ('capsule', 'bottle', 'washer', 'fairino', 'fr3wml', 'fr5'):
            assert word not in text, (f.name, word)


def _with_detectors(project: CanonicalProject, names) -> CanonicalProject:
    """The camera templates name a detector: the project provides one of that name (Step 5)."""
    if not names:
        return project
    data = project_to_dict(project)
    data['perception'] = {'detectors': [{'name': n, 'method': 'color_mask'} for n in names]}
    return CanonicalProject.parse_obj(data)


@pytest.mark.parametrize('tid', IDS)
def test_every_template_applies_validates_and_keeps_its_canonical_data(tid):
    applied = apply_template(sample(), tid)
    template = next(t for t in list_templates() if t.id == tid)
    app = project_to_dict(applied)['application']
    assert app['main_task'] == template.application['main_task']
    assert [t['name'] for t in app['tasks']] == [t['name'] for t in template.application['tasks']]
    assert applied.robot == sample().robot                         # only the application changed
    need = missing(applied, template.application)
    assert need['named_states'] == [], need                        # 'home' is a named state of the sample
    ready = _with_detectors(applied, need['detectors'])
    plan_program(resolve_project_paths(ready, SAMPLE)[0])


def test_requirements_name_what_a_template_expects():
    by_id = {t.id: t for t in list_templates()}
    assert requirements(by_id['inspection'].application)['detectors']
    assert requirements(by_id['machine_tending'].application)['channels']
    assert requirements(by_id['simple_motion'].application) == {'detectors': [], 'named_states': ['home'],
                                                                'channels': []}


def test_a_template_generates_a_bundle(tmp_path):
    from trainit_setup_assistant.generator import Orchestrator
    p = resolve_project_paths(apply_template(sample(), 'simple_motion'), SAMPLE)[0]
    Orchestrator().generate(p, tmp_path / 'b')
    assert next((tmp_path / 'b').rglob('task_program.xml')).is_file()


def test_the_composer_edits_a_template_program():
    c = Composer(apply_template(sample(), 'pick_and_place'))
    c.rename_waypoint(c.waypoint_names()[1], 'taught_pick')
    assert 'taught_pick' in c.waypoint_names()


# ---- Step 6 (Qt offscreen) -----------------------------------------------------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_step_6_starts_a_task_program_from_a_template(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(SAMPLE)
    w = SetupWizard(ctrl)
    page = w.application_page
    page.initializePage()
    page.app_type.setCurrentIndex([t for t, _, _ in page.APPS].index('task_program'))
    assert page.template_box.isVisibleTo(page)
    page.template.setCurrentIndex(IDS.index('machine_tending'))
    assert 'Load a raw part' in page.template_text.text()
    assert page.validatePage()
    assert ctrl.is_task_program() and ctrl.project.application.program.main_task == 'tend'
    w.blocks_page.initializePage()
    assert w.blocks_page.title() == 'Step 7 — Application Composer'
    # re-entering Step 6 shows the program, and offers starting over from a template
    page.initializePage()
    assert page.program_box.isVisibleTo(page) and page.restart_button.isVisibleTo(page)
