"""The planner and execution vocabularies TSA exposes, pinned (vNext M2.1).

Written and run against TSA 7.2.0, BEFORE TSA took its planner vocabulary from TMR's registry
(ADR-0021): the registry changes who KNOWS the tokens, never which ones a project may select, the
order the GUI lists them in, or what a project file without the key means. Changing any of them on
purpose edits these pins in the same commit, with the reason.
"""
import copy
import os
import pickle
from pathlib import Path

import pytest
from pydantic import ValidationError

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from trainit_setup_assistant.model.enums import ExecutionMode, PlannerId  # noqa: E402
from trainit_setup_assistant.model.task import ApplicationSpec, MotionSegment  # noqa: E402

EXAMPLE = str(Path(__file__).resolve().parents[1] / 'examples' / 'fr3wml_project.yaml')

OFFERED_PLANNERS = [('PILZ', 'pilz'), ('OMPL', 'ompl'), ('OMPL_CHOMP', 'ompl_chomp')]
EXECUTIONS = [('PLAN', 'plan'), ('SERVO', 'servo')]


@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_the_planner_modes_a_project_can_select():
    assert [(m.name, m.value) for m in PlannerId] == OFFERED_PLANNERS
    assert PlannerId('pilz') is PlannerId.PILZ and PlannerId.OMPL == 'ompl'
    assert str(PlannerId.OMPL_CHOMP) == 'ompl_chomp' and f'{PlannerId.OMPL}' == 'ompl'
    assert all(isinstance(m, str) for m in PlannerId)


def test_the_execution_modes_a_move_can_take():
    assert [(m.name, m.value) for m in ExecutionMode] == EXECUTIONS
    assert str(ExecutionMode.SERVO) == 'servo'


@pytest.mark.parametrize('member', list(PlannerId) + list(ExecutionMode))
def test_every_member_survives_pickle_and_copy(member):
    assert pickle.loads(pickle.dumps(member)) is member
    assert copy.deepcopy(member) is member


@pytest.mark.parametrize('token', ['stomp', 'cumotion', 'curobo', 'chomp', 'PILZ', 'Pilz', ''])
def test_a_token_that_is_not_offered_is_refused(token):
    """stomp: declared by TMR but not deliverable on Humble. cumotion/curobo: candidate providers
    (ADR-0019), never selectable. chomp: a pipeline, not a mode. Tokens are case-sensitive."""
    with pytest.raises(ValidationError):
        MotionSegment(to_waypoint='x', planner=token)
    with pytest.raises(ValidationError):
        ApplicationSpec(global_planner_mode=token)


def test_what_a_file_without_the_keys_means():
    assert ApplicationSpec().global_planner_mode is PlannerId.PILZ
    seg = MotionSegment(to_waypoint='x')
    assert seg.planner is None                     # None => the global planner_mode
    assert seg.execution is ExecutionMode.PLAN


def test_the_gui_lists_and_the_global_mode_the_wizard_writes(qapp):
    """Both move editors list the offered modes after a blank (= global), and the execution axis;
    the application page writes 'ompl' as the global mode of every project it touches."""
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard

    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.clear_application()
    wiz = SetupWizard(ctrl)

    def items(combo):
        return [combo.itemText(i) for i in range(combo.count())]

    planners = [''] + [v for _, v in OFFERED_PLANNERS]
    assert items(wiz.waypoints_page.move_planner) == planners
    assert items(wiz.blocks_page.m_planner) == planners
    assert items(wiz.waypoints_page.move_execution) == [v for _, v in EXECUTIONS]
    assert items(wiz.blocks_page.m_execution) == [v for _, v in EXECUTIONS]

    wiz.application_page.app_type.setCurrentText('pick_and_place')
    assert wiz.application_page.validatePage()
    assert ctrl.project.application.global_planner_mode is PlannerId.OMPL
