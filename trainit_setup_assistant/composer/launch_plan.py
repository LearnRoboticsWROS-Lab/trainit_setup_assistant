"""What TSA's launcher may run (vNext I0.2 / I0.11, ADR-0027): the exact command lines.

TSA LAUNCHES and OBSERVES; it never hosts the runtime or executes motion. Every process it starts is
one of the commands built here. Each is a plain shell line a user could type themselves, and the UI
always shows it so it can be copied. The commands are:
- build the generated packages;
- bring up the planning environment or the bundle;
- start Isaac on the bound stage;
- run the generated behaviour tree.

The generated bundle never needs TSA. Pure Python: no Qt, no ROS.
"""

from __future__ import annotations

import os
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

ROS_SETUP = '/opt/ros/humble/setup.bash'
OPEN_AND_PLAY = Path(__file__).resolve().parents[1] / 'resources' / 'isaac_cell_kit' / 'scripts' / 'open_and_play.py'
KINDS = ('build', 'environment', 'isaac', 'bringup', 'run')


@dataclass(frozen=True)
class Command:
    """One process TSA may start. ``script`` is run as ``bash -c script``; its last command is
    ``exec``'d, so a stop signal reaches it (``ros2 launch`` then shuts its nodes down)."""

    kind: str                    # build | environment | isaac | bringup | run
    title: str
    script: str
    long_running: bool
    env: Dict[str, str] = field(default_factory=dict)

    def argv(self) -> List[str]:
        """``setsid``: the command runs in its OWN process group, so stopping it reaches every node it
        started, as Ctrl-C does in a terminal (vNext I0.4 spike: a launch stopped alone left its scene
        manager and RViz running)."""
        return ['setsid', 'bash', '-c', self.script]

    def display(self) -> str:
        """The copyable form: what a user types in a terminal to do the same."""
        prefix = ''.join(f'{k}={shlex.quote(v)} ' for k, v in self.env.items())
        script = self.script[len('exec '):] if self.script.startswith('exec ') else self.script
        return prefix + script.replace(' && exec ', ' && ')


# A process started from VS Code's snap inherits variables that point INTO the snap; RViz then loads the
# snap's libraries and dies ("symbol lookup error ... /snap/core20 ... libpthread", vNext I0.4 spike).
_SNAP_ONLY = ('GTK_PATH', 'GTK_EXE_PREFIX', 'LOCPATH', 'GIO_MODULE_DIR', 'GDK_PIXBUF_MODULEDIR',
              'GDK_PIXBUF_MODULE_FILE', 'GTK_IM_MODULE_FILE', 'GSETTINGS_SCHEMA_DIR')


def child_environment(env: Dict[str, str]) -> Dict[str, str]:
    """The environment of a launched command: this process's, without the variables a snap (VS Code)
    pointed into itself, and with the XDG values the snap saved restored."""
    out = dict(env)
    for key in _SNAP_ONLY:
        value = out.get(key, '')
        if '/snap/' in value:
            out.pop(key)
    for key in ('XDG_DATA_DIRS', 'XDG_CONFIG_DIRS'):
        orig = out.get(f'{key}_VSCODE_SNAP_ORIG')
        if orig is not None:
            out[key] = orig
    return out


def _sourced(ws_root: Optional[str], tail: str) -> str:
    parts = [f'source {ROS_SETUP}']
    if ws_root:
        root = shlex.quote(str(ws_root))
        # the overlay is optional (not built yet); a failing ROS source or cd still stops the line
        parts += [f'cd {root}', 'if [ -f install/setup.bash ]; then source install/setup.bash; fi']
    return ' && '.join(parts + [f'exec {tail}'])


def build(ws_root: str, packages: Sequence[str]) -> Command:
    """``colcon build`` of the generated packages, at the workspace ROOT (the parent of ``src/``), looking
    for packages under ``src/`` only (``--base-paths src``): a workspace root may hold other trees with
    thousands of package.xml files (build outputs, test runs) that colcon would otherwise crawl."""
    if not packages:
        raise ValueError('nothing to build: no package named')
    pkgs = ' '.join(shlex.quote(p) for p in packages)
    script = ' && '.join([f'source {ROS_SETUP}', f'cd {shlex.quote(str(ws_root))}',
                          f'exec colcon build --base-paths src --packages-select {pkgs}'])
    return Command('build', f'Build {", ".join(packages)}', script, long_running=False)


def planning_environment(ws_root: str, config_package: str, mode: str, rviz: bool = True) -> Command:
    """The teaching / planning environment: the Step-3 scene+planner config's bring-up (move_group, the
    scene projection, the controllers, RViz with MotionPlanning), no behaviour tree."""
    args = f'mode:={shlex.quote(mode)}' + ('' if rviz else ' rviz:=false')
    return Command('environment', f'Planning environment ({mode})',
                   _sourced(ws_root, f'ros2 launch {shlex.quote(config_package)} bringup.launch.py {args}'),
                   long_running=True)


def isaac(stage: str, isaac_launcher: str) -> Command:
    """Isaac Sim opening the bound stage and pressing PLAY (``open_and_play.py``, shipped with TSA)."""
    if not stage:
        raise ValueError('no Isaac stage is bound (deployment.isaac.stage)')
    if not isaac_launcher:
        raise ValueError('the Isaac Sim launcher (isaac-sim.sh) is not set (Settings)')
    script = f'exec {shlex.quote(str(isaac_launcher))} --exec {shlex.quote(str(OPEN_AND_PLAY))}'
    return Command('isaac', 'Isaac Sim (open the stage, PLAY)', script, long_running=True,
                   env={'TRAINIT_STAGE': str(stage)})


def bringup(ws_root: str, app_package: str, mode: str, rviz: bool = True) -> Command:
    """The generated bundle's bring-up without its behaviour tree (``bt:=false``)."""
    args = f'mode:={shlex.quote(mode)} bt:=false' + ('' if rviz else ' rviz:=false')
    return Command('bringup', f'Bundle bring-up ({mode})',
                   _sourced(ws_root, f'ros2 launch {shlex.quote(app_package)} bringup.launch.py {args}'),
                   long_running=True)


def run(ws_root: str, app_package: str, mode: str) -> Command:
    """The generated behaviour tree (``trainit_bt.launch.py``): simulated time in a simulator, wall
    time on mock; Groot monitoring off."""
    sim_time = 'false' if mode == 'mock' else 'true'
    return Command('run', 'Run the application',
                   _sourced(ws_root, f'ros2 launch {shlex.quote(app_package)} trainit_bt.launch.py '
                                     f'use_sim_time:={sim_time} groot_monitor:=false'),
                   long_running=True)


def workspace_root(output_dir) -> Optional[str]:
    """The colcon workspace ROOT for a package written under ``<ws>/src/...`` (None: not under a src/)."""
    parts = os.path.abspath(str(output_dir)).split(os.sep)
    if 'src' not in parts:
        return None
    i = len(parts) - 1 - parts[::-1].index('src')
    return os.sep.join(parts[:i]) or os.sep


# What a run's output says about its outcome (TMR's runner; the M3 acceptance logs).
BT_SUCCESS = 'BT finished: SUCCESS'
BT_FAILURE = 'BT finished: FAILURE'


__all__ = ['BT_FAILURE', 'BT_SUCCESS', 'Command', 'child_environment', 'KINDS', 'OPEN_AND_PLAY', 'ROS_SETUP', 'bringup', 'build',
           'isaac', 'planning_environment', 'run', 'workspace_root']
