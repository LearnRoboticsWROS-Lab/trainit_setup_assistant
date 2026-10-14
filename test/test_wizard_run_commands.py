"""The commands the wizard prints after generating (GeneratePage, "Build & run";
``wizard.build_and_run_lines``), for both kinds of bundle and every backend (vNext G0 community
follow-up, owner P2): what the emitters write.

    the first run installs what the bundle declares (rosdep over the bundle; since TSA 7.8.1 that
      includes move_group's controller manager, so no apt line), before the build
    built on a base config: the cell is the trainit_config's bringup.launch.py, the tree the app's
      trainit_bt.launch.py, and the app's bringup.launch.py brings up both at once
    from scratch: the cell is the app's bringup.launch.py (the trainit_config has no launch file)
      and the tree its trainit_bt.launch.py
    trainit_bt.launch.py defaults use_sim_time to true, so it is given the mode profile's value
      (mock and real publish no /clock), and mode:= only where gazebo is deployed (the one case it
      declares mode)

The projects are the public sample's, loaded from its directory (nothing compiled, nothing
generated), with the backends and a base config set per case; the paths are strings, nothing is
written. wizard.py imports Qt; no widget is made. test_example_ur5e_suction.py checks the same
lines against a generated bundle's launch files.
"""
from pathlib import Path

import pytest

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'ur5e_suction'
DESC, CFG, APP = 'ur5e_suction_description', 'ur5e_suction_trainit_config', 'ur5e_suction_app'
# id: (built on a base config, the deployed modes, the default mode, the package of the cell's
# bring-up, the arguments of the tree's launch)
CASES = {
    'scratch-mock': (False, ['mock'], 'mock', APP, 'use_sim_time:=false'),
    'scratch-isaac': (False, ['mock', 'isaac'], 'isaac', APP, 'use_sim_time:=true'),
    'scratch-real': (False, ['mock', 'real'], 'real', APP, 'use_sim_time:=false'),
    'scratch-gazebo': (False, ['mock', 'gazebo'], 'gazebo', APP,
                       'use_sim_time:=true mode:=gazebo'),
    'scratch-mock-gazebo_deployed': (False, ['mock', 'gazebo'], 'mock', APP,
                                     'use_sim_time:=false mode:=mock'),
    'base-mock': (True, ['mock', 'isaac', 'real'], 'mock', CFG, 'use_sim_time:=false'),
    'base-isaac': (True, ['mock', 'isaac', 'real'], 'isaac', CFG, 'use_sim_time:=true'),
    'base-real': (True, ['mock', 'isaac', 'real'], 'real', CFG, 'use_sim_time:=false'),
    'base-gazebo': (True, ['mock', 'isaac', 'gazebo'], 'gazebo', CFG,
                    'use_sim_time:=true mode:=gazebo'),
}


@pytest.mark.parametrize('case', sorted(CASES))
def test_the_printed_commands_are_the_bundle_s(case, monkeypatch, tmp_path):
    from trainit_setup_assistant.gui.wizard import build_and_run_lines
    from trainit_setup_assistant.model import load_project
    base, modes, default, cell, tree = CASES[case]
    monkeypatch.chdir(EXAMPLE)               # the sample's paths are relative to its directory
    project = load_project('project.yaml')
    project.deployment.modes, project.deployment.default_mode = modes, default
    if base:
        project.robot.base_moveit_config_path = str(tmp_path / 'base_moveit_config')
    ws = tmp_path / 'ws'
    out = ws / 'src' / 'ur5e_suction_bundle'
    assert build_and_run_lines(project, str(out), str(ws)) == [
        f'  cd {ws}',
        "  # first time only: the bundle's dependencies",
        f'  rosdep install --from-paths {out} --ignore-src -r -y',
        f'  colcon build --packages-select {DESC} {CFG} {APP}',
        '  source install/setup.bash',
        f'  # robot alone:   ros2 launch {DESC} view_robot.launch.py',
        f'  # cell + RViz:   ros2 launch {cell} bringup.launch.py mode:={default}',
        f'  # application:   ros2 launch {APP} trainit_bt.launch.py {tree}',
    ] + ([f'  # or both:       ros2 launch {APP} bringup.launch.py mode:={default}']
         if base else [])
