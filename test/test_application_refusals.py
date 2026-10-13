"""Generation is refused BEFORE a file exists when the application cannot be generated.

vNext M0C #8. An ``AppType`` with no template (``gluing``, ``follow_path``,
``waypoint_replay``, ``cnc``: seams in the enum, offered greyed out by the wizard) reached
``get_application`` only inside the app emitter -- after the description and moveit_config
packages had already been written -- and stopped there with a ``KeyError``, leaving a partial
bundle on disk. It is now refused with the other refusals, before the output directory exists.

The application-specific refusals (a holding check with no topic, ``execution: servo`` with
``motion: lin|circ``) are pinned next to the features they belong to, in
``test_bin_picking.py`` and ``test_execution_mode.py``, and a canonical bin-picking project's
Detect block the loop cannot emit (vNext G0), in ``test_m2_7_capsule_readiness.py``.

Fast and ROS-free: generation is pure.
"""
import os

import pytest

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import AppType, load_project

EXAMPLE = os.path.join(os.path.dirname(__file__), os.pardir, 'examples', 'fr3wml_project.yaml')


@pytest.mark.parametrize('app_type', [AppType.GLUING, AppType.FOLLOW_PATH,
                                      AppType.WAYPOINT_REPLAY, AppType.CNC])
def test_an_application_with_no_template_is_refused_before_a_file_exists(tmp_path, app_type):
    p = load_project(EXAMPLE)
    p.application.type = app_type
    out = tmp_path / 'bundle'
    with pytest.raises(ValueError, match=f'no application template for "{app_type.value}"'):
        Orchestrator().generate(p, str(out))
    assert not out.exists()


def test_the_golden_project_is_not_refused(tmp_path):
    """The refusal stage adds nothing to a valid project: the reference app generates."""
    Orchestrator().generate(load_project(EXAMPLE), str(tmp_path / 'bundle'))
