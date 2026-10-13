"""The bundle build check finds the workspace it runs in, not one machine's path.

vNext M0C #8. ``verify/build_check.py`` defaulted its underlay to
``/home/fra/fr5_ws/install/setup.bash`` and its ROS setup to Humble's: on any other machine the
default underlay silently did not exist, so ``trainit_verify --build`` built the bundle against
ROS alone and failed on every TrainIt dependency, three layers from the cause. The default is
now the workspace the caller has SOURCED (colcon's ``COLCON_PREFIX_PATH``), and the ROS setup
follows ``ROS_DISTRO``. An explicit ``None`` still means "no underlay".

Fast: nothing is built here.
"""
import inspect

from trainit_setup_assistant.verify import build_check


def test_no_machine_path_in_the_build_check():
    assert '/home/' not in inspect.getsource(build_check)


def test_the_default_underlay_is_the_sourced_workspace(tmp_path, monkeypatch):
    (tmp_path / 'setup.bash').write_text('')
    monkeypatch.setenv('COLCON_PREFIX_PATH', f'/nonexistent:{tmp_path}')
    assert build_check.default_underlay() == str(tmp_path / 'setup.bash')


def test_no_sourced_workspace_means_no_underlay(monkeypatch):
    monkeypatch.delenv('COLCON_PREFIX_PATH', raising=False)
    assert build_check.default_underlay() is None


def test_the_ros_setup_follows_the_distro(monkeypatch):
    monkeypatch.setenv('ROS_DISTRO', 'jazzy')
    assert build_check.ros_setup() == '/opt/ros/jazzy/setup.bash'
    monkeypatch.delenv('ROS_DISTRO', raising=False)
    assert build_check.ros_setup() == '/opt/ros/humble/setup.bash'    # the supported track
