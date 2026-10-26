"""Live checks of the running planning environment against the project (vNext I0.2, I0_DESIGN.md §6).

They OBSERVE only: they read the joint states and TF, and list the controllers (``list_controllers``).
They check that the deployment robot the environment drives is the project's MoveIt robot:
- the planning-group joints are published;
- the TCP frame resolves;
- the arm controller is active.
rclpy is imported lazily.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ..composer.integration import LIVE_MAX_AGE_S
from .capture import stale_message


@dataclass(frozen=True)
class LiveCheck:
    name: str
    ok: Optional[bool]          # None: could not be decided (nothing running yet)
    text: str


def controller_manager_service(namespace: str = '') -> str:
    ns = (namespace or '').strip('/')
    return f'/{ns}/controller_manager/list_controllers' if ns else '/controller_manager/list_controllers'


def list_controllers(live, namespace: str = '', timeout_s: float = 2.0) -> Optional[dict]:
    """``{name: state}`` from the controller manager, or None when it does not answer."""
    import time
    from controller_manager_msgs.srv import ListControllers
    client = live.client(ListControllers, controller_manager_service(namespace))   # long-lived
    if not client.wait_for_service(timeout_sec=timeout_s):
        return None
    future = client.call_async(ListControllers.Request())
    deadline = time.time() + timeout_s
    while not future.done() and time.time() < deadline:
        time.sleep(0.02)                       # the LiveCapture executor spins the node
    if not future.done() or future.result() is None:
        return None
    return {c.name: c.state for c in future.result().controller}


def live_checks(live, project, timeout_s: float = 2.0) -> List[LiveCheck]:
    """Joint states, TCP frame and arm controller of the running environment, against the project."""
    robot = project.robot
    out: List[LiveCheck] = []
    age = live.joint_states_age()
    if age is None:                       # a session made a moment ago: give the first message its time
        try:
            live.current_joint_values(None, timeout_s=timeout_s)
        except TimeoutError:
            pass
        age = live.joint_states_age()
    if age is None:
        out.append(LiveCheck('joint states', False, f'no joint states on {live.joint_states_topic}: is the '
                                                    'planning environment running?'))
    elif age > LIVE_MAX_AGE_S:            # what a stopped environment left behind: not LIVE (owner Test A.2)
        out.append(LiveCheck('joint states', False, stale_message(live.joint_states_topic, age)))
    else:
        try:
            values = live.current_joint_values(None, timeout_s=timeout_s)
        except TimeoutError as exc:
            values = {}
            out.append(LiveCheck('joint states', False, str(exc)))
        missing = [j for j in robot.planning_group.joints if j not in values]
        if values:
            out.append(LiveCheck('joint states', not missing,
                                 f'{live.joint_states_topic} ({age:.1f} s old) carries every joint of '
                                 f"'{robot.planning_group.name}'" if not missing else
                                 f"{live.joint_states_topic} lacks {', '.join(missing)}: the running robot "
                                 'is not the project\'s MoveIt robot (joints are matched by name)'))
    try:
        live.current_pose(robot.base_frame, robot.tip_link, timeout_s=timeout_s)
        out.append(LiveCheck('TCP frame', True, f'TF {robot.base_frame} -> {robot.tip_link} resolves'))
    except TimeoutError as exc:
        out.append(LiveCheck('TCP frame', False, str(exc)))
    try:
        states = list_controllers(live, robot.cell_namespace, timeout_s)
    except Exception as exc:  # noqa: BLE001 - shown, never raised
        states = None
        out.append(LiveCheck('arm controller', None, f'cannot list the controllers: {exc}'))
    else:
        name = robot.arm_controller.name
        if states is None:
            out.append(LiveCheck('arm controller', False, f'no controller manager answers on '
                                                          f'{controller_manager_service(robot.cell_namespace)}: '
                                                          'is the planning environment running?'))
        elif states.get(name) == 'active':
            out.append(LiveCheck('arm controller', True, f"'{name}' is active"))
        else:
            out.append(LiveCheck('arm controller', False,
                                 f"'{name}' is {states.get(name, 'not loaded')} (controllers: "
                                 f"{', '.join(f'{k}={v}' for k, v in sorted(states.items())) or 'none'})"))
    return out


__all__ = ['LiveCheck', 'controller_manager_service', 'list_controllers', 'live_checks']
