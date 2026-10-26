"""vNext I0.6 (I0_DESIGN.md §23): reusable motion profiles, in an engineer's terms.

- Speed and acceleration are % of the joint limits; TCP velocity is in m/s; the Advanced view holds the
  rest.
- A field the user did not touch keeps its stored value exactly, and keys the editor does not show are
  kept.
- The presets' "intended usage" text is UI-only.
- Rename and delete go through the Composer: a used profile is never deleted, and two profiles are never
  merged.
"""
import os
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import plan_program
from trainit_setup_assistant.composer import Composer
from trainit_setup_assistant.model import CanonicalProject, project_to_dict
from trainit_setup_assistant.model.paths import resolve_project_paths

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'task_program.yaml'


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _panel():
    from trainit_setup_assistant.gui.composer.profiles_panel import ProfilesPanel
    c = Composer(CanonicalProject.parse_obj(yaml.safe_load(SAMPLE.read_text())))
    p = ProfilesPanel(lambda: c)
    p.refresh()
    return c, p


def test_the_table_shows_each_profile_in_an_engineers_terms(qapp):
    c, p = _panel()
    assert p.table.rowCount() == 3
    row = {p.table.item(i, 0).text(): [p.table.item(i, k).text() for k in range(1, 5)] for i in range(3)}
    assert row['approach'][:3] == ['30 %', '30 %', '0.1 m/s']
    assert 'step' in row['transit'][3]                      # used by moves


def test_apply_without_an_edit_keeps_every_profile(qapp):
    c, p = _panel()
    before = project_to_dict(c.project)
    for name in ('transit', 'approach', 'process'):
        p.select(name)
        assert p.apply()
    assert project_to_dict(c.project) == before and not c.can_undo


def test_editing_speed_and_an_advanced_tolerance(qapp):
    c, p = _panel()
    p.select('approach')
    assert p.usage.text().startswith('the last centimetres')            # UI-only text
    p.speed.setValue(25)
    p.advanced.setChecked(True)
    p.tol_pos.setValue(0.5)
    assert p.apply()
    prof = c.document()['motion_profiles']['approach']
    assert prof == {'speed': 25, 'acceleration': 30, 'tcp_velocity_mps': 0.1, 'goal_tolerance': {'position_m': 0.0005}}
    p.select('approach')
    p.tcp_vel.setValue(0.0)                                             # 0 = not set
    assert p.apply()
    assert 'tcp_velocity_mps' not in c.document()['motion_profiles']['approach']
    sample = resolve_project_paths(c.project, SAMPLE)[0]
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    plan_program(sample)


def test_new_from_preset_rename_and_delete(qapp):
    c, p = _panel()
    name = p.new_profile('slow_precision')
    assert name == 'slow_precision'
    assert c.document()['motion_profiles']['slow_precision'] == {'speed': 10, 'acceleration': 10,
                                                                 'tcp_velocity_mps': 0.02}
    assert p.new_profile('transit') == 'transit2'                        # never overwrites
    assert not p.rename('transit2', 'approach') and 'already exists' in p.message.text()
    assert not p.delete('transit') and 'is used by' in p.message.text()
    assert p.delete('transit2') and 'transit2' not in c.document()['motion_profiles']
