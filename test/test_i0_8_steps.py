"""vNext I0.8 (I0_DESIGN.md §17): process, camera, detect, grasp, release and tool authoring, M3 scope.

- Every form shows the fields of its kind. Its pick-lists come from the canonical project: waypoints,
  profiles, detectors, the candidate sets the detect steps write, and the end-effector contract's
  actuators / targets / sites.
- A channel is a LOGICAL id (the model refuses a topic path).
- An Apply without an edit changes nothing; keys a form does not show are kept.
- Each new step (capture, detect, grasp, release, tool) is added complete, then edited in its form.
"""
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import plan_program
from trainit_setup_assistant.composer import Composer, walk
from trainit_setup_assistant.model import CanonicalProject, project_to_dict
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / 'test' / 'data'
VALIDATION_C = DATA / 'm3_6' / 'validation_c.yaml'            # detect + grasp from candidates + release
VALIDATION_B = DATA / 'm3_5' / 'validation_b.yaml'            # process events + captures
DECOMPOSITIONS = DATA / 'm3_7' / 'decompositions'


def composer(path) -> Composer:
    return Composer(CanonicalProject.parse_obj(yaml.safe_load(path.read_text())))


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _ur5e_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def _panel(c, task=None):
    from trainit_setup_assistant.gui.composer.tasks_panel import TasksPanel
    p = TasksPanel(lambda: c)
    if task:
        p.task = task
    p.refresh()
    return p


def _decomposition(name) -> Composer:
    from test_i0_0_foundations import decomposition_project
    return Composer(CanonicalProject.parse_obj(decomposition_project(DECOMPOSITIONS / f'{name}.yaml')))


@pytest.mark.parametrize('source', ['validation_c', 'validation_b', 'e_bin_picking', 'd_pick_and_place',
                                    'b_inspection', 'a_machine_tending'])
def test_apply_without_an_edit_changes_nothing_in_every_m3_step_kind(qapp, source):
    c = composer(VALIDATION_C) if source == 'validation_c' else composer(VALIDATION_B) \
        if source == 'validation_b' else _decomposition(source)
    before = project_to_dict(c.project)
    kinds = set()
    for t in c.document()['tasks']:
        panel = _panel(c, t['name'])
        for label, step in walk(t['steps']):
            kinds.add(next(iter(step)))
            panel.select(label)
            assert panel.apply(), (source, label, panel.message.text())
    assert project_to_dict(c.project) == before, source
    assert not c.can_undo


def test_the_pick_lists_come_from_the_project(qapp):
    from trainit_setup_assistant.gui.composer.step_forms import context
    c = composer(VALIDATION_C)
    ctx = context(c.document(), c.project)
    assert ctx['detectors'] == ['grasp_provider'] and 'parts' in ctx['sets']
    assert ctx['eef'] and ctx['sites'] and ctx['actuators']
    assert 'pick_nominal' in ctx['waypoints'] and 'transit' in ctx['profiles']


def test_a_process_event_on_a_logical_channel_and_a_capture(qapp):
    c = composer(VALIDATION_B)
    task = c.program.main_task
    panel = _panel(c, task)
    label = panel.add_step('process')
    panel.select(label)
    form = panel.step_forms['process']
    form.pkind.setCurrentText('set_value')
    form.channel.setCurrentText('dispenser_pressure')
    form.value.setValue(2.5)
    assert panel.apply()
    assert c.step(task, label) == {'process': {'kind': 'set_value', 'channel': 'dispenser_pressure', 'value': 2.5}}
    form.channel.setCurrentText('/plc/out/3')                    # a topic path is not a logical channel
    assert not panel.apply() and 'logical channel' in panel.message.text()
    cap = panel.add_step('capture')
    assert c.step(task, cap)['process']['kind'] == 'capture'
    plan_program(resolve_project_paths(c.project, VALIDATION_B)[0])


def test_detect_grasp_release_and_tool_are_authored_in_forms(qapp):
    c = composer(VALIDATION_C)
    task = c.program.main_task
    panel = _panel(c, task)
    det = next(l for l, s in walk(next(t for t in c.document()['tasks'] if t['name'] == task)['steps'])
               if 'detect' in s)
    panel.select(det)
    f = panel.step_forms['detect']
    f.keep.setValue(3)
    assert panel.apply() and c.step(task, det)['detect']['keep'] == 3
    grasp = next(l for l, s in walk(next(t for t in c.document()['tasks'] if t['name'] == task)['steps'])
                 if 'grasp' in s)
    panel.select(grasp)
    g = panel.step_forms['grasp']
    assert g.mode.currentIndex() == 1 and g.candidates.currentText() == 'parts'
    g.standoff.setValue(60.0)
    g.retract_mode.setCurrentText('along_approach')
    g.retract.setValue(90.0)
    assert panel.apply()
    body = c.step(task, grasp)['grasp']
    assert body['standoff_m'] == 0.06 and body['retract'] == {'along_approach': 0.09}
    assert body['candidates'] == 'parts' and body['orientation_from'] == 'pick_nominal'
    plan_program(resolve_project_paths(c.project, VALIDATION_C)[0])
    for kind in ('release', 'tool', 'detect', 'grasp'):
        label = panel.add_step(kind)
        assert label is not None, (kind, panel.message.text())
        assert next(iter(c.step(task, label))) == kind


def test_a_new_capture_without_a_detector_says_where_to_configure_one(qapp):
    from test_i0_7_tasks import composer as composition
    c = composition()
    if c.project.perception is not None and c.project.perception.detectors:
        pytest.skip('the fixture has detectors')
    panel = _panel(c)
    assert panel.add_step('capture') is None and 'Perception' in panel.message.text()


@pytest.mark.parametrize('name', ['test_c/m3b_fr3wml.yaml', 'test_d/m3c_fr3wml.yaml', 'test_e/m3e_fr3wml.yaml',
                                  'test_f/m3f_fr3wml.yaml'])
def test_the_m3_acceptance_programs_survive_an_apply_of_every_step(qapp, name):
    """I0's acceptance fixtures (opt-in: outside every repository): every step of every task, loaded
    into its form and applied without an edit, leaves the project as it was."""
    from test_i0_0_foundations import ACCEPTANCE
    path = ACCEPTANCE / name
    if not path.is_file():
        pytest.skip(f'M3 acceptance project not present: {path}')
    c = composer(path)
    before = project_to_dict(c.project)
    for t in c.document()['tasks']:
        panel = _panel(c, t['name'])
        for label, _step in walk(t['steps']):
            panel.select(label)
            assert panel.apply(), (name, label, panel.message.text())
    assert project_to_dict(c.project) == before


def test_an_untouched_process_value_stays_exact(qapp):
    from trainit_setup_assistant.composer import Composer
    c = composer(VALIDATION_B)
    task = c.program.main_task
    c.append_step(task, {'process': {'kind': 'set_value', 'channel': 'pressure', 'value': 0.123456789}})
    label = (len(next(t for t in c.document()['tasks'] if t['name'] == task)['steps']),)
    panel = _panel(c, task)
    panel.select(label)
    assert panel.apply()
    assert c.step(task, label)['process']['value'] == 0.123456789
