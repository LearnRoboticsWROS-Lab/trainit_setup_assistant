"""`mode:=real` on the reference cell, as a CONTRACT -- evaluated, never run (vNext M0D T4).

No process is started and nothing connects to a robot: the generated bring-up's launch function
is EVALUATED in a LaunchContext with ``mode=real`` and the actions it would start are read back.
Three facts, pinned as the baseline they are on 2026-10-06:

1. NAMES -- the action names MoveIt is configured with (the bundle's moveit_controllers.yaml)
   are the ones the FAIRINO bridges serve: ``moveit_joint_controller/follow_joint_trajectory``
   (fairino_bridge, all three server variants) and ``suctioncup_controller/gripper_command``
   (fairino_gripper). The naming half of the real contract holds.
2. WHAT REAL STARTS -- nothing that serves them. The reference project declares no real
   bring-up (``real_include``) and no bridge runs in ``real``: the bundle brings up
   robot_state_publisher, move_group and the scene manager, and no ros2_control (by design:
   "the vendor bridges impersonate the same action names") -- so, as generated, ``mode:=real``
   can PLAN and can never MOVE. The FAIRINO bridges are started by hand.
3. MATURITY -- the FAIRINO bridge is DEVELOPMENT / NON-PRODUCTION-READY: ``mode:=real`` has
   never been executed end to end, and the roadmap records that the bridge drops the
   trajectory (R0). Nothing here claims otherwise; this test fails the day the bundle starts
   hardware, so that the claim is revisited rather than inherited.
4. THE BRIDGE'S LIMITS -- what the vNext M2.7 capsule readiness matrix records against the real
   cell (DOCUMENTED, not fixed): the FollowJointTrajectory bridge executes only the last waypoint,
   never reads the goal's joint names, moves at its own 95 % speed, resets controller errors
   before every trajectory, holds the motion mutex across a cancel, and the suction shim awaits
   asyncio inside its executor. Read from the source, never run; it fails the day the bridge
   changes, by the same rule.

Skipped when the reference cell's packages are not installed (a checkout of this repo alone).
"""
import importlib.util
import re
from pathlib import Path

import pytest
import yaml

import ws_paths

GOLDEN = ws_paths.GOLDEN_BUNDLE
CFG_PKG = 'fr3wml_suction_camera_tsa_4_1_trainit_config'
BRIDGE_SRC = ws_paths.ws_src('fairino_bridge_master_class', 'src')
SUCTION_SRV = ws_paths.ws_src('fairino_bridge_gripper_master_class', 'fairino_gripper',
                              'gripper_suction_action_server.py')
SERVER3 = BRIDGE_SRC / 'follow_joint_trajectory_server3.cpp'


def _installed(pkg: str) -> bool:
    try:
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory(pkg)
        return True
    except Exception:
        return False


def _moveit_actions() -> dict:
    mscm = yaml.safe_load((GOLDEN / CFG_PKG / 'config' / 'moveit_controllers.yaml').read_text())[
        'moveit_simple_controller_manager']
    return {c: (mscm[c]['type'], f"{c}/{mscm[c]['action_ns']}") for c in mscm['controller_names']}


@pytest.mark.skipif(not (BRIDGE_SRC.is_dir() and SUCTION_SRV.is_file() and GOLDEN.is_dir()),
                    reason='the reference cell and the FAIRINO bridge sources are not present')
def test_the_bridges_serve_the_names_moveit_is_configured_with():
    actions = _moveit_actions()
    assert actions == {
        'moveit_joint_controller': ('FollowJointTrajectory',
                                    'moveit_joint_controller/follow_joint_trajectory'),
        'suctioncup_controller': ('GripperCommand', 'suctioncup_controller/gripper_command')}
    servers = [p for p in sorted(BRIDGE_SRC.glob('follow_joint_trajectory_server*.cpp'))]
    assert servers, 'no FollowJointTrajectory server in fairino_bridge'
    for src in servers:
        names = re.findall(r'create_server<\w+>\(\s*[^,]+,\s*"([^"]+)"', src.read_text())
        assert names == ['moveit_joint_controller/follow_joint_trajectory'], (src.name, names)
    suction = re.findall(r'GripperCommand,\s*"([^"]+)"', SUCTION_SRV.read_text())
    assert suction == ['suctioncup_controller/gripper_command']


@pytest.mark.skipif(not _installed(CFG_PKG), reason=f'{CFG_PKG} is not installed')
def test_real_mode_as_generated_starts_nothing_that_moves_the_robot():
    from launch import LaunchContext
    from launch.actions import OpaqueFunction
    from ament_index_python.packages import get_package_share_directory
    launch_file = Path(get_package_share_directory(CFG_PKG)) / 'launch' / 'bringup.launch.py'
    spec = importlib.util.spec_from_file_location('ref_bringup', launch_file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ctx = LaunchContext()
    ctx.launch_configurations.update({'mode': 'real', 'rviz': 'false', 'scene_config': '',
                                      'start_tolerance': ''})
    started = []
    for entity in mod.generate_launch_description().entities:
        if isinstance(entity, OpaqueFunction):
            for action in entity.execute(ctx) or []:
                pkg = getattr(action, 'node_package', None)
                exe = getattr(action, 'node_executable', None)
                if pkg:
                    started.append((pkg, exe))
    packages = {p for p, _ in started}
    assert ('moveit_ros_move_group', 'move_group') in started             # it can PLAN ...
    assert 'controller_manager' not in packages                            # no ros2_control
    assert not any(p.startswith('fairino') for p in packages), started    # ... and never MOVE
    assert mod.REAL_INCLUDE is None
    assert not [b for b in mod.BRIDGES if 'real' in b.get('modes', [])]


@pytest.mark.skipif(not (SERVER3.is_file() and SUCTION_SRV.is_file()),
                    reason='the FAIRINO bridge sources are not present')
def test_the_fairino_bridge_limits_are_the_recorded_ones():
    """Fact 4: the server the reference suction bring-up starts (server3), and the suction shim."""
    src = SERVER3.read_text()
    assert 'traj.back()' in src                        # only the LAST waypoint becomes one MoveJ
    assert 'joint_names' not in src                    # positions by index, the names never read
    assert re.search(r'motion_speed_percent_\(95\.0f\)', src)            # its own speed, 95 %
    assert 'declare_parameter<bool>("auto_reset_errors", true)' in src   # a fault cleared ...
    assert 'send_command_blocking("ResetAllError()")' in src             # ... by the next goal
    lock = 'lock_guard<std::mutex> lock(motion_mutex_)'
    assert lock in src[src.index('handle_cancel('):src.index('handle_accepted(')]   # a cancel
    assert lock in src[src.index('void execute('):]    # waits for the mutex the motion holds
    assert 'asyncio.wait_for' in SUCTION_SRV.read_text()   # awaited inside its own executor
