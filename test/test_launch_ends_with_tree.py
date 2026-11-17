"""TSA 8.6.1 (found by the M4.5 Isaac benchmark): the application launch ENDS WITH ITS TREE.

``trainit_bt.launch.py`` starts the runner (``trainit_run_bt``) and, for a perception application, the perception
nodes beside it (detector_node, a grasp provider such as surface_grasp_node / scripted_grasp_node, an external
segmenter such as yolo_seg_node). Those nodes never exit by themselves, so the launch outlived the tree --
contradicting "its launch ends by itself when the tree ends". The template now registers, on the runner action,
an ``OnProcessExit`` handler that emits ``Shutdown`` (the reason names the runner's exit and return code).

The rule, the same for an application with or without perception:

  * a STANDALONE ``trainit_bt.launch.py`` ends when the runner exits (without perception it always did: the runner
    was its only process; with perception, by the handler -- ``end_with_tree``, default true);
  * the app ``bringup.launch.py`` that INCLUDES it (a bundle on a base MoveIt config) keeps the cell up after the
    tree ends, as it always did: it passes ``end_with_tree: false``, so the perception nodes stay up with the cell.

A bundle WITHOUT perception nodes generates both files byte-identically (none of the new tokens appears).

Checked here at the text / ``ast`` level, and by RUNNING the generated launch file in ``launch.LaunchService`` with
every ``Node`` replaced by a stand-in process (the runner exits after 0.5 s, every perception node sleeps): no
ROS graph, no build, a few seconds.
"""
import ast
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import CanonicalProject

HERE = Path(__file__).parent
M45 = HERE / 'data' / 'm45'
SAMPLE = HERE.parent / 'examples' / 'ur5e_suction'
NEW_TOKENS = ('end_with_tree', 'OnProcessExit', 'RegisterEventHandler', 'Shutdown', 'ENDS WITH THE TREE')


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def _generate(path: Path, out: Path, monkeypatch) -> Path:
    monkeypatch.chdir(path.parent)
    project = CanonicalProject.parse_obj(yaml.safe_load(path.read_text()))
    Orchestrator().generate(project, out)
    return out / project.bundle.app_package / 'launch'


@pytest.fixture(scope='module')
def bundles(tmp_path_factory):
    # a module-scoped fixture runs BEFORE the function-scoped autouse skip: check the sample's prerequisite here too
    # (a fresh machine without ur_description skips instead of erroring -- the M4.5 fresh-machine check found it)
    import test_example_ur5e_suction as S
    S.needs_ur_description()
    mp = pytest.MonkeyPatch()
    try:
        root = tmp_path_factory.mktemp('launch_ends')
        return {name: _generate(path, root / name, mp) for name, path in (
            ('mock', M45 / 'capsule_bin_picking_mock.yaml'),          # scripted_grasp_node
            ('isaac', M45 / 'capsule_bin_picking_isaac.yaml'),        # surface_grasp_node + yolo_seg_node
            ('plain', SAMPLE / 'task_program.yaml'))}                 # no perception node
    finally:
        mp.undo()


def _node_calls(tree: ast.AST):
    return [n for n in ast.walk(tree) if isinstance(n, ast.Call) and getattr(n.func, 'id', '') == 'Node']


def _kw(call: ast.Call, name: str):
    return next((ast.literal_eval(k.value) for k in call.keywords if k.arg == name), None)


# --- the generated text ----------------------------------------------------------------------------------------

@pytest.mark.parametrize('name, perception', [('mock', ['scripted_grasp_node']),
                                              ('isaac', ['surface_grasp_node', 'yolo_seg_node'])])
def test_a_perception_launch_shuts_down_on_the_runners_exit(bundles, name, perception):
    text = (bundles[name] / 'trainit_bt.launch.py').read_text()
    tree = ast.parse(text)
    setup = next(f for f in tree.body if isinstance(f, ast.FunctionDef) and f.name == 'launch_setup')
    # the actions list: the perception nodes first, the runner LAST (the handler's target is actions[-1])
    assign = next(s for s in setup.body if isinstance(s, ast.Assign) and isinstance(s.targets[0], ast.Name)
                  and s.targets[0].id == 'actions')
    execs = [_kw(c, 'executable') for c in assign.value.elts]
    assert execs == perception + ['trainit_run_bt']
    # the handler: on the runner, Shutdown with the runner's return code in its reason, under end_with_tree
    guard = next(s for s in setup.body if isinstance(s, ast.If) and 'end_with_tree' in ast.unparse(s.test))
    assert setup.body.index(guard) == len(setup.body) - 2           # after the list, before the return
    body = ast.unparse(guard)
    assert 'runner = actions[-1]' in body
    assert 'RegisterEventHandler(OnProcessExit(target_action=runner' in body
    assert 'Shutdown(reason=f' in body and 'event.returncode' in body
    assert ast.unparse(setup.body[-1]) == 'return actions'
    # the argument, true by default
    declared = {_kw(c, 'default_value') for c in ast.walk(tree) if isinstance(c, ast.Call)
                and getattr(c.func, 'id', '') == 'DeclareLaunchArgument'
                and c.args and ast.literal_eval(c.args[0]) == 'end_with_tree'}
    assert declared == {'true'}
    assert 'from launch.event_handlers import OnProcessExit' in text


def test_a_launch_without_perception_is_unchanged(bundles):
    """Its runner is its only process: it always ended with the tree, and none of the new tokens appears."""
    text = (bundles['plain'] / 'trainit_bt.launch.py').read_text()
    assert [_kw(c, 'executable') for c in _node_calls(ast.parse(text))] == ['trainit_run_bt']
    assert not [t for t in NEW_TOKENS if t in text]
    assert '    return [\n' in text


def test_the_bringup_that_includes_it_keeps_the_cell_up():
    """A base-config bundle with perception (the golden project): its app bringup includes trainit_bt.launch.py
    with end_with_tree false -- the cell (and its perception) stays up after the tree, as without perception."""
    import ws_paths
    from trainit_setup_assistant.model.io import load_project
    project_file = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
    if not project_file.is_file():
        pytest.skip('golden bundle not present (set TRAINIT_WS_SRC)')
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        project = load_project(str(project_file))
        Orchestrator().generate(project, tmp)
        launch = Path(tmp) / project.bundle.app_package / 'launch'
        bringup = (launch / 'bringup.launch.py').read_text()
        assert 'detector_node' in (launch / 'trainit_bt.launch.py').read_text()
    include = next(n for n in ast.walk(ast.parse(bringup)) if isinstance(n, ast.Dict)
                   and any(isinstance(k, ast.Constant) and k.value == 'planner_mode' for k in n.keys))
    pairs = {k.value: v for k, v in zip(include.keys, include.values)}
    assert ast.literal_eval(pairs['end_with_tree']) == 'false'


# --- RUN: the generated launch file in launch.LaunchService, every Node a stand-in process ------------------------

RUNNER_SECONDS = 0.5
SAFETY_SECONDS = 6.0


def _run(launch_file: Path, monkeypatch, tmp_path, *args: str):
    """Run ``launch_file`` with its ``Node``s replaced by plain processes: the runner sleeps RUNNER_SECONDS and
    exits, every other node sleeps 60 s. A safety timer (a thread: a TimerAction would itself keep the launch
    busy) shuts the service down after SAFETY_SECONDS. Returns (seconds the launch ran, the executables
    started, in order)."""
    import threading

    from launch import LaunchDescription, LaunchService
    from launch.actions import ExecuteProcess, IncludeLaunchDescription
    from launch.launch_description_sources import PythonLaunchDescriptionSource

    fake = types.ModuleType('trainit_motion_runtime.trainit_moveit_config')
    fake.build_move_group_params = lambda **_kw: ({}, None)
    fake.available_pipeline_names = lambda: []
    try:
        import trainit_motion_runtime  # noqa: F401 -- the real package, when sourced
    except ImportError:
        monkeypatch.setitem(sys.modules, 'trainit_motion_runtime', types.ModuleType('trainit_motion_runtime'))
    monkeypatch.setitem(sys.modules, 'trainit_motion_runtime.trainit_moveit_config', fake)
    import ament_index_python.packages as aip
    monkeypatch.setattr(aip, 'get_package_share_directory', lambda _pkg: str(tmp_path))
    import launch_ros.actions as lra
    started = []

    def stand_in(*, executable, name=None, **_kw):
        seconds = RUNNER_SECONDS if executable == 'trainit_run_bt' else 60
        started.append(executable)
        return ExecuteProcess(cmd=[sys.executable, '-c', f'import time; time.sleep({seconds})'],
                              name=executable, output='log', sigterm_timeout='2', sigkill_timeout='2')
    monkeypatch.setattr(lra, 'Node', stand_in)

    ls = LaunchService(argv=list(args))
    ls.include_launch_description(LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(launch_file)),
                                 launch_arguments=[tuple(a.split(':=')) for a in args])]))
    safety = threading.Timer(SAFETY_SECONDS, ls.shutdown)
    safety.start()
    t0 = time.monotonic()
    try:
        assert ls.run() == 0
    finally:
        safety.cancel()
    return time.monotonic() - t0, started


@pytest.mark.parametrize('name', ['mock', 'isaac'])
def test_run_a_perception_launch_ends_when_the_runner_exits(bundles, name, monkeypatch, tmp_path):
    seconds, started = _run(bundles[name] / 'trainit_bt.launch.py', monkeypatch, tmp_path, 'use_sim_time:=false')
    assert started[-1] == 'trainit_run_bt' and len(started) >= 2
    assert seconds < SAFETY_SECONDS - 1.0, f'the launch outlived its tree ({seconds:.1f} s)'


def test_run_included_with_end_with_tree_false_the_perception_stays_up(bundles, monkeypatch, tmp_path):
    seconds, _ = _run(bundles['isaac'] / 'trainit_bt.launch.py', monkeypatch, tmp_path, 'use_sim_time:=false',
                      'end_with_tree:=false')
    assert seconds >= SAFETY_SECONDS - 0.5, f'the launch ended with the tree ({seconds:.1f} s)'


def test_run_a_launch_without_perception_still_ends_with_its_tree(bundles, monkeypatch, tmp_path):
    seconds, started = _run(bundles['plain'] / 'trainit_bt.launch.py', monkeypatch, tmp_path, 'use_sim_time:=false')
    assert started == ['trainit_run_bt']
    assert seconds < SAFETY_SECONDS - 1.0
