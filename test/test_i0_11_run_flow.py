"""vNext I0.11 (I0_DESIGN.md §22, §23, §30; ADR-0027): validate → generate → build → start → run →
observe, from Step 8.

- Step 8's commands are launch_plan's lines for THIS bundle: the three packages built at the workspace
  root, the bundle brought up without its tree in the chosen mode, the run, and Isaac on the bound stage.
- The reference copy of project.yaml written into the bundle does not move the project's own file:
  relative paths keep resolving beside it.
- Launch tier (``TRAINIT_LAUNCH_TEST=1``): the public sample's task program, generated, built, brought
  up on MOCK and run through TSA's own launcher until the runner prints ``BT finished: SUCCESS``. The
  bundle needs no TSA to run.
"""
import os
import shutil
import time
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.composer import launch_plan

PKG = Path(__file__).resolve().parents[1]
SAMPLE = PKG / 'examples' / 'ur5e_suction' / 'task_program.yaml'


@pytest.fixture(autouse=True)
def _ur5e_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


@pytest.fixture
def qapp():
    os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
    from python_qt_binding.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def _wizard(tmp_path, monkeypatch):
    from trainit_setup_assistant.composer.settings import ENV
    monkeypatch.setenv(ENV, str(tmp_path / 'tsa.yaml'))
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(SAMPLE)
    w = SetupWizard(ctrl)
    return ctrl, w


def test_step_8_runs_the_generated_bundle_with_launch_plans_lines(qapp, tmp_path, monkeypatch):
    ctrl, w = _wizard(tmp_path, monkeypatch)
    page = w.generate_page
    page.initializePage()
    assert page.launcher is w.launcher
    assert [page.run_mode.itemText(i) for i in range(page.run_mode.count())] == ['mock']
    with pytest.raises(ValueError, match='generate the bundle'):
        page._build_command()
    src = tmp_path / 'ws' / 'src'
    src.mkdir(parents=True)
    page.output_dir.setText(str(src))
    page.generate()
    assert 'Generated' in page.result.toPlainText(), page.result.toPlainText()
    ws = str(tmp_path / 'ws')
    b = ctrl.project.bundle
    assert page._build_command().display() == launch_plan.build(
        ws, [b.description_package, b.moveit_config_package, b.app_package]).display()
    assert page._bringup_command().display() == launch_plan.bringup(ws, b.app_package, 'mock').display()
    assert page._run_command().display() == launch_plan.run(ws, b.app_package, 'mock').display()
    with pytest.raises(ValueError, match='no Isaac stage is bound'):
        page._isaac_command()
    # the reference copy in the bundle does not become the project's file
    assert ctrl.project_path == str(SAMPLE.resolve())
    assert list(src.rglob('project.yaml'))


@pytest.mark.skipif(os.environ.get('TRAINIT_LAUNCH_TEST') != '1', reason='launch tier: TRAINIT_LAUNCH_TEST=1')
def test_generate_build_start_run_observe_on_mock(qapp, tmp_path, monkeypatch):
    ctrl, w = _wizard(tmp_path, monkeypatch)
    page = w.generate_page
    page.initializePage()
    src = tmp_path / 'ws' / 'src'
    src.mkdir(parents=True)
    page.output_dir.setText(str(src))
    page.generate()
    assert 'Generated' in page.result.toPlainText(), page.result.toPlainText()
    os.environ['ROS_DOMAIN_ID'] = str(150 + os.getpid() % 30)
    launcher = page.launcher

    def wait(cond, timeout):
        end = time.time() + timeout
        while time.time() < end and not cond():
            qapp.processEvents()
            time.sleep(0.05)
        return cond()

    build = launcher.start(page._build_command())
    assert wait(lambda: build.exit_code is not None, 900) and build.exit_code == 0, build.lines[-30:]
    print('I0.11: built', flush=True)
    up = launcher.start(launch_plan.bringup(page._ws_root(), ctrl.project.bundle.app_package, 'mock', rviz=False))
    try:
        assert wait(lambda: any('You can start planning now' in ln or 'Ready to take commands' in ln
                                for ln in up.lines), 120), up.lines[-40:]
        print('I0.11: bundle up', flush=True)
        run = launcher.start(page._run_command())
        print('I0.11: running', page._run_command().display(), flush=True)
        assert wait(lambda: run.verdict is not None or run.exit_code is not None, 300), run.lines[-40:]
        assert run.verdict == launch_plan.BT_SUCCESS, run.lines[-60:]
        print('I0.11 launch tier:', page._run_command().display(), '->', run.verdict)
    finally:
        launcher.stop_all()
    assert not launcher.running('bringup') and not launcher.running('run')


# ---- owner Test C: Step 8's defaults and the running Step 4 environment ----------------------------------------

def test_step_8_names_the_bundle_after_the_project_and_keeps_what_the_user_typed(qapp, tmp_path, monkeypatch):
    """Owner Test C, friction row 22: Step 8 pre-filled the ROBOT's name and Generate silently renamed the project
    (and its bundle) to it. The pre-fill is the project's own name; a name typed here survives a re-entry; the
    same name does not touch the bundle."""
    ctrl, w = _wizard(tmp_path, monkeypatch)
    page = w.generate_page
    page.initializePage()
    assert page.project_name.text() == ctrl.project.project_name == 'ur5e_program_sample'
    page.project_name.setText('my_cell')
    page.initializePage()                                   # Back, then Next again
    assert page.project_name.text() == 'my_cell'
    page.project_name.setText(ctrl.project.project_name)
    bundle = ctrl.project.bundle.copy()
    src = tmp_path / 'ws' / 'src'
    src.mkdir(parents=True)
    page.output_dir.setText(str(src))
    page.generate()
    assert 'Generated' in page.result.toPlainText(), page.result.toPlainText()
    assert ctrl.project.project_name == 'ur5e_program_sample' and ctrl.project.bundle == bundle


def test_step_8_output_is_the_step_3_workspace_never_the_base_config_repository(qapp, tmp_path, monkeypatch):
    """Owner Test C, friction row 22: the output defaulted NEXT TO the base MoveIt config, i.e. inside its source
    repository beside committed bundles. It defaults to the src/ of the workspace Step 3 generated into, else
    stays empty (the placeholder says what is expected)."""
    ctrl, w = _wizard(tmp_path, monkeypatch)
    page = w.generate_page
    repo = tmp_path / 'repo'
    (repo / 'robot_moveit_config').mkdir(parents=True)
    ctrl.project.robot.base_moveit_config_path = str(repo / 'robot_moveit_config')
    ctrl.workspace_root = None
    page.initializePage()
    assert page.output_dir.text() == '' and 'workspace' in page.output_dir.placeholderText()
    ws = tmp_path / 'ws'
    (ws / 'src').mkdir(parents=True)
    ctrl.workspace_root = str(ws)
    page.initializePage()
    assert page.output_dir.text() == str(ws / 'src')
    assert str(repo) not in page.output_dir.text()


def test_step_8_replaces_an_existing_bundle_only_when_the_user_says_so(qapp, tmp_path, monkeypatch):
    """Owner Test C, friction row 22: generating into an existing bundle folder overwrote it without asking. A
    refusal leaves the folder untouched; a yes regenerates it."""
    ctrl, w = _wizard(tmp_path, monkeypatch)
    page = w.generate_page
    page.initializePage()
    src = tmp_path / 'ws' / 'src'
    src.mkdir(parents=True)
    page.output_dir.setText(str(src))
    asked = []
    page._confirm = lambda q: asked.append(q) or False
    page.generate()                                        # a new folder: nothing to confirm
    assert 'Generated' in page.result.toPlainText() and not asked
    bundle_dir = src / f'{ctrl.project.project_name}_bundle'
    marker = bundle_dir / 'my_notes.txt'
    marker.write_text('mine')
    before = {p: p.stat().st_mtime_ns for p in bundle_dir.rglob('*') if p.is_file()}
    page.generate()
    assert len(asked) == 1 and str(bundle_dir) in asked[0] and 'REPLACES' in asked[0]
    assert page.result.toPlainText().startswith('Not generated')
    assert {p: p.stat().st_mtime_ns for p in bundle_dir.rglob('*') if p.is_file()} == before
    page._confirm = lambda q: True
    page.generate()
    assert 'Generated' in page.result.toPlainText(), page.result.toPlainText()


def test_step_8_stops_the_step_4_environment_and_never_starts_the_bundle_beside_it(qapp, tmp_path, monkeypatch):
    """Owner Test C, friction row 23: Step 8 could start the bundle while Step 4's planning environment still ran
    (two move_group and controller managers on one robot). Step 8 shows that environment, its button stops it,
    and the bundle refuses to start beside it."""
    ctrl, w = _wizard(tmp_path, monkeypatch)
    page = w.generate_page
    page.initializePage()
    panel, launcher = page.run_panel, page.launcher
    assert 'environment' in panel.buttons and panel.status['environment'] == 'not started'
    panel.toggle('environment')                            # not running: Step 8 never starts it
    assert panel.state.text().startswith('⚠ Planning environment (Step 4): start it at Step 4')
    assert not launcher.running('environment')
    env = launcher.start(launch_plan.Command(
        'environment', 'Env', "trap 'exit 130' INT; echo ready; while true; do sleep 0.1; done", True))
    try:
        end = time.time() + 5
        while time.time() < end and 'ready' not in env.lines:
            qapp.processEvents()
            launcher.pump()
            time.sleep(0.05)
        assert launcher.running('environment')
        with pytest.raises(ValueError, match=r'planning environment \(Step 4\) is running: stop it first'):
            page._bringup_command()
        panel.toggle('environment')                        # running: the button stops it
        assert not launcher.running('environment') and panel.status['environment'] == 'stopped'
    finally:
        launcher.stop_all()
    src = tmp_path / 'ws' / 'src'
    src.mkdir(parents=True)
    page.output_dir.setText(str(src))
    page.generate()
    assert page._bringup_command().display().endswith('bringup.launch.py mode:=mock bt:=false')
