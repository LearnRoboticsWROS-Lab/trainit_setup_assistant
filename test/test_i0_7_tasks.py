"""vNext I0.7 (I0_DESIGN.md §16): task composition, visually, with the canonical nesting.

- The tree IS the program's nesting: a task's steps, a repeat / retry body, each alternative of a "Try
  alternatives". Every canonical label is one node. The words are an engineer's: no BT terms.
- Every operation is one Composer operation, refused by the model when the file could not hold it: add
  (after / inside), move, delete, wrap (repeat / retry / try alternatives), unwrap, make one path,
  new / rename / delete / main task, and the inspector's edits.
- An Apply without an edit changes nothing.
"""
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import plan_program
from trainit_setup_assistant.composer import Composer, walk
from trainit_setup_assistant.composer.describe import route, step_text
from trainit_setup_assistant.model import CanonicalProject, project_to_dict
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
COMPOSITION = PKG / 'test' / 'data' / 'm3_7' / 'composition.yaml'
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'task_program.yaml'
BT_WORDS = ('Sequence', 'Fallback', 'RetryUntilSuccessful', 'SubTree', 'BehaviorTree', 'Repeat node')


def composer(path=COMPOSITION) -> Composer:
    return Composer(CanonicalProject.parse_obj(yaml.safe_load(path.read_text())))


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel(c, task=None):
    from trainit_setup_assistant.gui.composer.tasks_panel import TasksPanel
    p = TasksPanel(lambda: c)
    if task:
        p.task = task
    p.refresh()
    return p


# ---- words ------------------------------------------------------------------------------------------------

@pytest.mark.parametrize('step, text', [
    ({'move': {'to': 'w3', 'type': 'circ', 'via': 'w2b', 'profile': 'approach'}}, 'Move → w3 · CIRC via w2b · approach'),
    ({'path': 'sweep'}, 'Path sweep'),
    ({'task': 'go_home'}, 'Run task go_home'),
    ({'repeat': {'cycles': 2, 'body': []}}, 'Repeat ×2'),
    ({'retry': {'attempts': 3, 'body': []}}, 'Retry ×3'),
    ({'fallback': {'bodies': [[], []]}}, 'Try alternatives (2)'),
    ({'process': {'kind': 'process_on', 'channel': 'cycle_light'}}, 'Process ON cycle_light'),
    ({'process': {'kind': 'capture', 'detector': 'red_object'}}, 'Capture (red_object)'),
    ({'dwell': {'ms': 500}}, 'Wait 500 ms'),
    ({'detect': {'detector': 'red_object', 'out': 'cubes'}}, 'Detect (red_object) → cubes'),
    ({'release': {'at': 'place'}}, 'Release at place'),
])
def test_every_step_reads_without_bt_words(step, text):
    assert step_text(step) == text
    assert not any(w in text for w in BT_WORDS)


def test_the_route_reads_from_to_how():
    c = composer(SAMPLE)
    doc = c.document()
    task = doc['tasks'][0]
    lines = route(task, paths={p['name']: p for p in doc['paths']})
    assert lines[0] == '1  (where the task starts) → home : PTP · transit · STOP'
    assert lines[3] == '4  w2 → w3 : CIRC via w2b · approach · STOP'
    assert lines[7] == '8  path sweep: w1 → w2 → w3 → w4 : one motion · BLEND'
    assert lines[8] == '9  w4 → home : PTP · transit · STOP'


# ---- the tree --------------------------------------------------------------------------------------------

def test_the_tree_is_the_canonical_nesting(qapp):
    c = composer()
    for t in c.document()['tasks']:
        panel = _panel(c, t['name'])
        labels = [label for label, _step in walk(t['steps'])]
        assert sorted(panel.items) == sorted(labels)
        for label, step in walk(t['steps']):
            node = panel.items[label]
            assert node.text(0) == step_text(step)
            assert node.text(1) == '.'.join(map(str, label))
            parent = node.parent()
            if len(label) > 1:
                data = parent.data(0, 256)                     # Qt.UserRole
                assert data[0] in ('step', 'alt')              # under its control step or its alternative


def test_apply_without_an_edit_changes_nothing(qapp):
    c = composer()
    before = project_to_dict(c.project)
    for t in c.document()['tasks']:
        panel = _panel(c, t['name'])
        for label in list(panel.items):
            panel.select(label)
            assert panel.apply()
    assert project_to_dict(c.project) == before and not c.can_undo


# ---- operations ------------------------------------------------------------------------------------------

def test_add_after_and_inside_a_repeat_body_and_an_alternative(qapp):
    c = composer()
    main = c.program.main_task
    panel = _panel(c, main)
    rep = next(l for l, s in walk(next(t for t in c.document()['tasks'] if t['name'] == main)['steps'])
               if 'repeat' in s)
    panel.select(rep)
    body_len = len(c.step(main, rep)['repeat']['body'])
    assert panel.add_step('dwell', inside=True) == rep + (body_len + 1,)
    fb = next(l for l, s in walk(next(t for t in c.document()['tasks'] if t['name'] == main)['steps'])
              if 'fallback' in s)
    alt_item = panel.items[fb].child(1)                         # Alternative 2
    panel.tree.clearSelection()
    alt_item.setSelected(True)
    new = panel.add_step('dwell', inside=True)
    assert new[:-1] == fb + (2,) and c.step(main, new) == {'dwell': {'ms': 500}}
    panel.select((1,))
    assert panel.add_step('dwell') == (2,) and c.step(main, (2,)) == {'dwell': {'ms': 500}}
    assert panel.add_step('move') == (3,) and c.step(main, (3,))['move']['profile']   # a new move has limits


def test_wrap_in_repeat_retry_and_try_alternatives_then_unwrap(qapp):
    c = composer(SAMPLE)
    main = c.program.main_task
    c.add_task('recover', steps=[{'move': {'to': 'home', 'type': 'ptp', 'profile': 'transit'}}])
    panel = _panel(c, main)
    first, second = c.step(main, (2,)), c.step(main, (3,))
    for item in (panel.items[(2,)], panel.items[(3,)]):
        item.setSelected(True)
    panel.wrap_kind.setCurrentIndex(0)                          # Repeat
    panel.wrap_times.setValue(3)
    assert panel.wrap_selection()
    assert c.step(main, (2,)) == {'repeat': {'cycles': 3, 'body': [first, second]}}
    panel.select((2,))
    assert panel.unwrap_selected()
    panel.select((2,))
    panel.wrap_kind.setCurrentIndex(2)                          # Try alternatives
    panel.alternative_task.setCurrentText('recover')
    assert panel.wrap_selection()
    assert c.step(main, (2,)) == {'fallback': {'bodies': [[first], [{'task': 'recover'}]]}}
    assert [panel.items[(2,)].child(i).text(0) for i in range(2)] == ['Alternative 1 (tried first)',
                                                                      'Alternative 2']
    import test_example_ur5e_suction as S
    S.needs_ur_description()                             # the sample's frames are checked from its description
    plan_program(resolve_project_paths(c.project, SAMPLE)[0])


def test_make_one_path_from_the_selected_moves(qapp):
    c = composer(SAMPLE)
    main = c.program.main_task
    panel = _panel(c, main)
    for label in ((3,), (4,), (5,)):
        panel.items[label].setSelected(True)
    assert panel.make_path('from_ui', radius_m=0.02)
    assert c.step(main, (3,)) == {'path': 'from_ui'}
    assert 'path from_ui: w1 → w2 → w3 → w4' in panel.route_view.toPlainText()


def test_move_delete_and_the_inspector(qapp):
    c = composer(SAMPLE)
    main = c.program.main_task
    panel = _panel(c, main)
    second = c.step(main, (2,))
    panel.select((2,))
    assert panel.move(1)
    assert c.step(main, (3,)) == second
    panel.select((3,))
    panel.move_type.setCurrentText('lin')
    panel.move_profile.setCurrentText('approach')
    assert panel.apply()
    assert c.step(main, (3,)) == {'move': {'to': 'w1', 'type': 'lin', 'profile': 'approach'}}
    n = len(next(t for t in c.document()['tasks'] if t['name'] == main)['steps'])
    panel.select((3,))
    assert panel.delete_selected()
    assert len(next(t for t in c.document()['tasks'] if t['name'] == main)['steps']) == n - 1


def test_tasks_new_rename_main_and_a_called_task_is_not_deleted(qapp):
    c = composer()
    panel = _panel(c)
    called = next(s['task'] for t in c.document()['tasks'] for _l, s in walk(t['steps']) if 'task' in s)
    assert not panel.delete_task(called) and 'is called by' in panel.message.text()
    assert panel.rename_task(called, 'renamed_task')
    assert any(s.get('task') == 'renamed_task' for t in c.document()['tasks'] for _l, s in walk(t['steps']))
    assert panel.new_task('extra')
    assert 'extra' in c.task_names() and panel.task == 'extra'
    assert panel.set_main('extra') and c.program.main_task == 'extra'
    assert not panel.new_task('extra') and panel.message.text()                 # a duplicate name: refused


def test_the_composer_has_every_section(qapp):
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(COMPOSITION)
    w = SetupWizard(ctrl)
    w.blocks_page.initializePage()
    shell = w.blocks_page.program_panel
    assert list(shell.sections) == ['Overview', 'Waypoints', 'Paths', 'Profiles', 'End effector', 'Tasks', 'YAML', 'Program']
    shell.select('Tasks')
    assert shell.tasks.task == ctrl.project.application.program.main_task
