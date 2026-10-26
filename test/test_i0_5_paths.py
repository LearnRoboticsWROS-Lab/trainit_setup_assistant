"""vNext I0.5 (I0_DESIGN.md §15): paths and "make these moves one path".

- WHERE = waypoints, HOW = the relation (LIN / CIRC + via), END BEHAVIOUR = STOP or BLEND with a radius.
- Converting consecutive LIN / CIRC moves into ONE path reproduces, from the public sample's own moves,
  the blended path that sample writes by hand (states, segments, via, radii).
- The converted program validates with ``plan_program`` and generates.
- The Paths editor rewrites only what it shows; every other key of a path is kept.
"""
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import plan_program
from trainit_setup_assistant.composer import Composer, ComposerError
from trainit_setup_assistant.model import CanonicalProject, project_to_dict
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'task_program.yaml'


@pytest.fixture(autouse=True)
def _ur5e_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def composer() -> Composer:
    return Composer(CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text())))


def resolved(c: Composer) -> CanonicalProject:
    return resolve_project_paths(c.project, SAMPLE)[0]


def test_the_samples_moves_become_the_samples_blended_path():
    c = composer()
    task = c.program.main_task
    c.moves_to_path(task, (3,), 3, 'sweep_from_moves', continuity='blend', blend_radius_m=0.02)
    doc = c.document()
    made = next(p for p in doc['paths'] if p['name'] == 'sweep_from_moves')
    by_hand = next(p for p in doc['paths'] if p['name'] == 'sweep')
    assert made['states'] == by_hand['states'] == ['w1', 'w2', 'w3', 'w4']
    assert made['segments'] == [{'type': 'lin', 'blend_radius_m': 0.02},
                                {'type': 'circ', 'via': 'w2b', 'blend_radius_m': 0.02},
                                {'type': 'lin'}] == by_hand['segments']
    assert made['continuity'] == 'blend' and made['profile'] == 'approach'   # the moves' shared profile
    steps = next(t for t in doc['tasks'] if t['name'] == task)['steps']
    assert steps[1] == {'move': {'to': 'w1', 'type': 'ptp', 'profile': 'transit'}}
    assert steps[2] == {'path': 'sweep_from_moves'}
    plan_program(resolved(c))                                       # the canonical validator accepts it


def test_a_stop_path_carries_no_blend_and_undo_restores_the_moves():
    c = composer()
    before = project_to_dict(c.project)
    c.moves_to_path(c.program.main_task, (3,), 3, 'stop_path', continuity='stop')
    made = next(p for p in c.document()['paths'] if p['name'] == 'stop_path')
    assert 'continuity' not in made and all('blend_radius_m' not in s for s in made['segments'])
    plan_program(resolved(c))
    c.undo()
    assert project_to_dict(c.project) == before


@pytest.mark.parametrize('label, count, match', [
    ((1,), 2, 'is a ptp step'),                       # a PTP move stays a move
    ((3,), 9, 'no 9 steps'),
])
def test_only_lin_and_circ_moves_after_a_move_become_a_path(label, count, match):
    c = composer()
    with pytest.raises(ComposerError, match=match):
        c.moves_to_path(c.program.main_task, label, count, 'p')


def test_a_path_must_start_where_the_robot_is():
    c = composer()
    task = c.program.main_task
    c.delete_step(task, (2,))                          # now the LIN to w2 follows the PTP to home
    c.delete_step(task, (1,))                          # and now it is the first step
    with pytest.raises(ComposerError, match='starts where the robot IS'):
        c.moves_to_path(task, (1,), 2, 'p')


def test_renaming_a_path_renames_its_steps():
    c = composer()
    c.rename_path('sweep', 'blended_sweep')
    steps = next(t for t in c.document()['tasks'] if t['name'] == c.program.main_task)['steps']
    assert {'path': 'blended_sweep'} in steps and {'path': 'sweep'} not in steps


# ---- the Paths editor (Qt offscreen) --------------------------------------------------------------------

@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel(c):
    from trainit_setup_assistant.gui.composer.paths_panel import PathsPanel
    p = PathsPanel(lambda: c)
    p.refresh()
    return p


def test_apply_without_an_edit_keeps_the_path(qapp):
    from trainit_setup_assistant.gui.composer.paths_panel import path_summary
    c = composer()
    before = project_to_dict(c.project)
    panel = _panel(c)
    panel.select('sweep')
    assert panel.apply()
    assert project_to_dict(c.project) == before
    assert path_summary(c.document()['paths'][0]) == ('w1 → w2 (LIN, blend 20 mm) → w3 (CIRC via w2b, blend 20 mm) '
                                                     '→ w4 (LIN) · BLEND · profile process')


def test_switching_to_stop_removes_every_blend_and_keeps_the_other_keys(qapp):
    c = composer()
    panel = _panel(c)
    panel.select('sweep')
    panel.continuity.setCurrentText('stop')
    assert panel.apply()
    p = next(x for x in c.document()['paths'] if x['name'] == 'sweep')
    assert p['continuity'] == 'stop' and all('blend_radius_m' not in s for s in p['segments'])
    assert p['goal_tolerance'] == {'position_m': 0.001, 'orientation_rad': 0.01}   # kept as written
    plan_program(resolved(c))


def test_a_new_path_and_a_state_added_by_the_editor(qapp):
    c = composer()
    panel = _panel(c)
    name = panel.new_path('scan')
    assert name == 'scan'
    panel.select('scan')
    panel.add_state('w4')
    panel.table.cellWidget(2, 1).setCurrentText('lin')
    assert panel.apply()
    p = next(x for x in c.document()['paths'] if x['name'] == 'scan')
    assert p['states'][-1] == 'w4' and len(p['segments']) == 2


def test_a_last_segment_never_blends_whatever_the_editor_says(qapp):
    c = composer()
    panel = _panel(c)
    panel.select('sweep')
    panel.table.cellWidget(3, 3).setValue(15.0)        # a radius typed on the LAST segment
    assert panel.apply()
    p = next(x for x in c.document()['paths'] if x['name'] == 'sweep')
    assert 'blend_radius_m' not in p['segments'][-1]


def test_the_converted_program_generates_a_followpath(tmp_path):
    from trainit_setup_assistant.generator import Orchestrator
    c = composer()
    c.moves_to_path(c.program.main_task, (3,), 3, 'sweep_from_moves', blend_radius_m=0.02)
    Orchestrator().generate(resolved(c), tmp_path / 'b')
    paths = yaml.safe_load(next((tmp_path / 'b').rglob('paths.yaml')).read_text())
    assert 'sweep_from_moves' in str(paths)
    tree = next((tmp_path / 'b').rglob('task_program.xml')).read_text()
    assert 'path="sweep_from_moves"' in tree


def test_a_path_offers_only_poses_and_keeps_showing_a_joint_state_it_already_names(qapp):
    """Owner Test B: a new path's first state was set to 'home' (a named JOINT target) and the program got
    R-M3-04. A path state or a CIRC via is a pose: the editor offers TCP / relative waypoints only; a joint
    target a path already names stays visible so the validator's error can be read."""
    c = composer()
    kinds = {w['name']: w for w in c.document()['waypoints']}
    joint_names = [n for n, w in kinds.items() if w.get('type') == 'joint']
    assert joint_names
    panel = _panel(c)
    panel.select('sweep')
    state = panel.table.cellWidget(0, 0)
    offered = [state.itemText(i) for i in range(state.count())]
    via = panel.table.cellWidget(2, 2)
    offered_via = [via.itemText(i) for i in range(via.count())]
    assert not set(joint_names) & set(offered) and not set(joint_names) & set(offered_via)
    panel._fill([joint_names[0], 'w1'], [{'type': 'lin'}])
    assert panel.table.cellWidget(0, 0).currentText() == joint_names[0]
