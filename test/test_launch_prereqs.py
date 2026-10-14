"""The opt-in launch tiers' prerequisite check, ``launch_prereqs`` (vNext G0 community follow-up,
owner P2), on synthetic generated cells in a temp dir: no ROS, no generation, no build.

    a no-op where everything is here (the reference machine)
    a SKIP naming every package the cells declare and the ament index lacks, the controller
      manager a moveit_controllers.yaml names, and every harness module this interpreter cannot
      import, with the apt line -- and a FAIL instead with TRAINIT_REQUIRE_GOLDEN=1
    never the cells' own packages, nor those of another cell generated beside them, nor a key that
      is no ROS package name (python3-...)
    the named plugin is 'declared' once a package.xml declares it, and not missing once installed
    a missing Community package is built in the TrainIt workspace, and a Pro or experimental one
      is named as such: neither gets an apt line
    composite_launch's module lists are the imports of the code they guard (_Probe.__init__, and
      with _probe_cells phase A), and the other two tiers import no ROS module beyond the probe's:
      a new import there would fail a fresh machine after minutes of building, unnamed

The machine is simulated: the ament index and the importable modules are monkeypatched.
"""
import ast
from pathlib import Path

import pytest

import eef_fixture
import launch_prereqs as LP

HERE = Path(__file__).resolve().parent
RG = ' -- and TRAINIT_REQUIRE_GOLDEN=1: this gate must run'
PLUGIN = 'moveit_simple_controller_manager/MoveItSimpleControllerManager'
MODULES = ('rclpy', 'controller_manager_msgs')
EVERYTHING = {'controller_manager', 'joint_state_publisher_gui', 'moveit_ros_move_group',
              'moveit_simple_controller_manager', 'trainit_motion_runtime'}


def _package(path: Path, *depends: str, tag: str = 'exec_depend') -> None:
    path.mkdir(parents=True)
    (path / 'package.xml').write_text(
        '<?xml version="1.0"?>\n<package format="3">\n'
        f'  <name>{path.name}</name>\n  <version>0.1.0</version>\n'
        '  <buildtool_depend>ament_cmake</buildtool_depend>\n'
        + ''.join(f'  <{tag}>{d}</{tag}>\n' for d in depends) + '</package>\n')


@pytest.fixture
def src(tmp_path):
    """Two cells generated into one temp workspace, as a launch tier writes them: cell_a's app
    declares its own packages, cell_b's config, ros2_control, TMR and a system key; its
    description a ``depend``; its config names the controller-manager plugin."""
    src = tmp_path / 'src'
    _package(src / 'cell_a' / 'cell_a_app', 'cell_a_description', 'cell_a_config',
             'cell_b_config', 'controller_manager', 'python3-yaml', 'trainit_motion_runtime')
    _package(src / 'cell_a' / 'cell_a_description', 'joint_state_publisher_gui', tag='depend')
    _package(src / 'cell_a' / 'cell_a_config', 'moveit_ros_move_group')
    (src / 'cell_a' / 'cell_a_config' / 'config').mkdir()
    (src / 'cell_a' / 'cell_a_config' / 'config' / 'moveit_controllers.yaml').write_text(
        f'moveit_controller_manager: {PLUGIN}\n')
    _package(src / 'cell_b' / 'cell_b_config')
    return src


def _machine(monkeypatch, index, importable=MODULES, require_golden=False):
    monkeypatch.setattr(LP, '_on_the_index', lambda: set(index))
    monkeypatch.setattr(LP, '_importable', lambda module: module in importable)
    monkeypatch.setattr(eef_fixture, 'REQUIRE_GOLDEN', require_golden)
    monkeypatch.setenv('ROS_DISTRO', 'humble')


# What a fresh Humble machine set up by README.md "Install" has of these: MoveIt and TrainIt, no
# ros2_control and no controller-manager plugin (a fresh osrf/ros:humble-desktop container, as
# measured at vNext G0).
FRESH = {'moveit_ros_move_group', 'trainit_motion_runtime'}
FRESH_LACKS = [('controller_manager', 'declared'), ('joint_state_publisher_gui', 'declared'),
               ('moveit_simple_controller_manager', 'named'),
               ('controller_manager_msgs', 'imported')]


def test_where_everything_is_here_it_does_nothing(src, monkeypatch):
    _machine(monkeypatch, EVERYTHING, require_golden=True)
    assert LP.missing(src, modules=MODULES) == []
    assert LP.require(src, modules=MODULES) is None


def test_a_fresh_machine_skips_naming_what_it_lacks_and_the_fix(src, monkeypatch):
    _machine(monkeypatch, FRESH, importable=('rclpy',))
    assert LP.missing(src, modules=MODULES) == FRESH_LACKS
    with pytest.raises(pytest.skip.Exception) as stop:
        LP.require(src, modules=MODULES)
    message = str(stop.value)
    assert message.startswith('this launch tier stops before the build, as this machine lacks '
                              'controller_manager, joint_state_publisher_gui (the generated '
                              'package.xml files exec_depend on them'), message
    assert "moveit_simple_controller_manager (move_group's controller manager" in message
    assert 'the Python module controller_manager_msgs (the launch harness imports it)' in message
    assert ('-- fix: sudo apt install ros-humble-controller-manager '
            'ros-humble-controller-manager-msgs ros-humble-joint-state-publisher-gui '
            'ros-humble-moveit-simple-controller-manager.') in message
    assert '`rosdep install --from-paths <the bundle> --ignore-src -r -y`' in message
    assert not message.endswith(RG)


def test_with_require_golden_it_fails_instead(src, monkeypatch):
    _machine(monkeypatch, FRESH, importable=('rclpy',), require_golden=True)
    with pytest.raises(pytest.fail.Exception) as stop:
        LP.require(src, modules=MODULES)
    assert str(stop.value).endswith(RG)
    assert 'controller_manager_msgs' in str(stop.value)


def test_the_cells_own_packages_and_system_keys_are_never_reported(src, monkeypatch):
    _machine(monkeypatch, index=())                 # nothing at all on the index
    assert LP.missing(src) == [
        ('controller_manager', 'declared'), ('joint_state_publisher_gui', 'declared'),
        ('moveit_ros_move_group', 'declared'), ('trainit_motion_runtime', 'declared'),
        ('moveit_simple_controller_manager', 'named')]


def test_the_named_plugin_is_declared_once_a_package_declares_it(src, monkeypatch):
    manifest = src / 'cell_a' / 'cell_a_config' / 'package.xml'
    manifest.write_text(manifest.read_text().replace(
        '</package>', '  <exec_depend>moveit_simple_controller_manager</exec_depend>\n</package>'))
    _machine(monkeypatch, FRESH | {'controller_manager', 'joint_state_publisher_gui'})
    assert LP.missing(src) == [('moveit_simple_controller_manager', 'declared')]
    _machine(monkeypatch, EVERYTHING)
    assert LP.missing(src) == []


def test_a_missing_trainit_package_is_built_or_named_never_apt_installed():
    message = LP.reason([('trainit_motion_runtime', 'declared'),
                         ('trainit_policy_runtime', 'declared')])
    assert ('build trainit_motion_runtime in the TrainIt workspace (README.md, "Install")'
            in message), message
    assert ('trainit_policy_runtime is not TrainIt Community (Pro or experimental): a bundle '
            'under a Community launch tier should not declare it') in message
    assert 'trainit-' not in message


def _imports(node: ast.AST) -> set:
    """The top-level module of every absolute import under ``node``."""
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Import):
            names |= {a.name.split('.')[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.level == 0:
            names.add(n.module.split('.')[0])
    return names


def test_the_harness_module_lists_are_the_imports_they_guard():
    import composite_launch as CL
    tree = ast.parse((HERE / 'composite_launch.py').read_text())
    method = {f'{c.name}.{f.name}': f for c in tree.body if isinstance(c, ast.ClassDef)
              for f in c.body if isinstance(f, ast.FunctionDef)}
    probe = _imports(method['_Probe.__init__'])
    assert sorted(CL.PROBE_MODULES) == sorted(probe)
    assert sorted(CL.PHASE_A_MODULES) == sorted(
        probe | _imports(method['CompositeLaunch._probe_cells']))


def test_the_other_tiers_import_no_ros_module_beyond_the_probe_s():
    import composite_launch as CL
    helpers = {p.stem for p in HERE.glob('*.py')} | {'trainit_setup_assistant'}
    for module in ('test_m2_7_bundle_runner_launch', 'test_example_ur5e_suction_launch'):
        tree = ast.parse((HERE / f'{module}.py').read_text())
        inner = set().union(*(_imports(f) for f in ast.walk(tree)
                              if isinstance(f, ast.FunctionDef)))
        # what a fixture or a test imports at run time, beyond TSA and this directory's helpers
        assert inner - helpers <= set(CL.PROBE_MODULES), (module, sorted(inner - helpers))
