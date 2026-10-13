"""The tests resolve the files they read without one machine's paths (vNext M0B/M0D).

test/ws_paths.py and test/conftest.py are what let a checkout on another machine run the same
tests: the workspace is found from the test's own location, and the absolute paths the reference
projects recorded (the golden's frozen project.yaml among them) are rebased into this workspace
when, and only when, they do not exist here.
"""
from pathlib import Path

import ws_paths
from trainit_setup_assistant.model import load_project


def test_a_path_that_exists_is_left_exactly_as_it_is():
    here = str(ws_paths.WS_SRC / 'trainit_setup_assistant')
    assert ws_paths.in_workspace(here) == here


def test_a_path_recorded_on_another_machine_lands_in_this_workspace():
    elsewhere = '/home/someone_else/other_ws/src/trainit_setup_assistant/examples'
    got = ws_paths.in_workspace(elsewhere)
    assert got == str(ws_paths.WS_SRC / 'trainit_setup_assistant' / 'examples')
    assert Path(got).is_dir()


def test_a_path_outside_any_workspace_package_is_returned_unchanged():
    assert ws_paths.in_workspace('/nonexistent/at/all') == '/nonexistent/at/all'


def test_records_compare_workspace_relative():
    assert ws_paths.workspace_relative(
        '/home/a/ws/src/trainit_setup_assistant/x.yaml') == 'trainit_setup_assistant/x.yaml'
    assert ws_paths.workspace_relative('moveit_config/package.xml.j2') == \
        'moveit_config/package.xml.j2'                       # a template name, not a path


def test_loaded_projects_are_rebased(tmp_path):
    """The conftest wrapper: a project recording another machine's urdf_dir loads with this
    workspace's (here the example's own examples/ dir stands in for a description dir)."""
    src = Path(__file__).resolve().parents[1] / 'examples' / 'fr3wml_project.yaml'
    text = src.read_text()
    moved = tmp_path / 'p.yaml'
    import yaml
    data = yaml.safe_load(text)
    data['robot']['description']['urdf_dir'] = '/elsewhere/ws/src/trainit_setup_assistant/examples'
    moved.write_text(yaml.safe_dump(data))
    p = load_project(str(moved))
    assert p.robot.description.urdf_dir == str(ws_paths.WS_SRC / 'trainit_setup_assistant'
                                               / 'examples')
