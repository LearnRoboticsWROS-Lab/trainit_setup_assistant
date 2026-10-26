"""The end effector checked against the running deployment, READ-ONLY (vNext I0, I0_DESIGN.md §34 E5).

ADR-0027: TSA never commands the end effector. This only LOOKS: the ROS graph (who publishes the device's state,
which endpoints are served) and the latest state the adapter published (``trainit_interfaces``
EndEffectorState, subscribed for one message). The names are the contract's (ADR-0022 D7), relative to the
robot's namespace: ``eef/<device>/state``, ``eef/<device>/actuators/<id>/set_engaged`` for a binary actuator,
the actuator's ``action`` for a jaw or an axis.
"""

from __future__ import annotations

import time
from typing import List

from .checks import LiveCheck

def _names(prefix: str) -> dict:
    """``{value: name}`` of the EndEffectorState constants with ``prefix`` (the message is the authority)."""
    from trainit_interfaces.msg import EndEffectorState
    return {v: k[len(prefix):] for k, v in vars(EndEffectorState).items()
            if k.startswith(prefix) and isinstance(v, int)}


def _resolve(namespace: str, name: str) -> str:
    ns = (namespace or '').strip('/')
    name = name.lstrip('/')
    return f'/{ns}/{name}' if ns else f'/{name}'


def latest_state(live, topic: str, timeout_s: float = 2.0):
    """One EndEffectorState from ``topic``, or None (no message within the timeout, or no interfaces)."""
    from trainit_interfaces.msg import EndEffectorState          # ImportError: the caller says so
    from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
    got = []
    # the adapter's state QoS (reliable, transient local): the latest state, even between its heartbeats
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    sub = live.node.create_subscription(EndEffectorState, topic, got.append, qos)
    try:
        deadline = time.monotonic() + timeout_s
        while not got and time.monotonic() < deadline:
            time.sleep(0.05)                                   # the live session's executor spins
        return got[-1] if got else None
    finally:
        try:
            live.node.destroy_subscription(sub)
        except Exception:  # noqa: BLE001
            pass


def eef_checks(live, project, timeout_s: float = 2.0) -> List[LiveCheck]:
    """The device's state publisher, its endpoints, its tool joints, its latest state."""
    from ..model.end_effector import derive_device
    robot = project.robot
    dev = derive_device(robot.gripper, robot.named_states, stage_topic=project.scene.gripper_cmd_topic or None)
    if dev is None:
        return [LiveCheck('end effector', False, 'no end-effector contract (End effector section)')]
    ns = robot.cell_namespace
    node = live.node
    out: List[LiveCheck] = []
    state_topic = _resolve(ns, f'eef/{dev.id}/state')
    pubs = node.get_publishers_info_by_topic(state_topic)
    if len(pubs) == 1:
        out.append(LiveCheck('state', True, f"{state_topic}: published by '{pubs[0].node_name}' (the adapter)"))
    elif not pubs:
        out.append(LiveCheck('state', False, f'{state_topic}: nobody publishes it -- is the bundle running, with its '
                                             'end-effector adapter (Step 8: Start the bundle)?'))
    else:
        out.append(LiveCheck('state', False, f'{state_topic}: {len(pubs)} publishers (' + ', '.join(
            p.node_name for p in pubs) + '): one device, one adapter'))
    services = {name for name, _types in node.get_service_names_and_types()}
    topics = {name for name, _types in node.get_topic_names_and_types()}
    tool_joints = [j for act in dev.actuators for j in (act.joints or [])]
    joint_values, joints_why = {}, ''
    if tool_joints:                       # ONE read for every actuator (python review S6: the GUI waits once)
        try:
            joint_values = live.current_joint_values(tool_joints, timeout_s=timeout_s)
        except Exception as exc:  # noqa: BLE001 - said, not mistaken for missing joints
            joints_why = str(exc)
    for act in dev.actuators:
        if act.interface.value == 'set_bool':
            name = _resolve(ns, f'eef/{dev.id}/actuators/{act.id}/set_engaged')
            ok = name in services
        else:
            name = _resolve(ns, act.action or '')
            ok = f'{name}/_action/status' in topics
        out.append(LiveCheck(f"actuator '{act.id}'", ok, f'{act.interface.value} at {name}: '
                             + ('served' if ok else 'not served (no server: the first command would fail)')))
        if act.joints:
            if joints_why:
                out.append(LiveCheck(f"joints of '{act.id}'", None, f'the joint states cannot be read: {joints_why}'))
                continue
            missing = [j for j in act.joints if j not in joint_values]
            out.append(LiveCheck(f"joints of '{act.id}'", not missing,
                                 'in the joint states' if not missing else
                                 f"not in the joint states: {', '.join(missing)} (MoveIt then sees an incomplete "
                                 'robot; a joint-state merger publishes them)'))
    try:
        msg = latest_state(live, state_topic, timeout_s) if pubs else None
        status_names, grasp_names = _names('STATUS_'), _names('GRASP_')
    except ImportError:
        out.append(LiveCheck('latest state', None, 'trainit_interfaces is not importable here: source the '
                                                   'workspace that has it'))
        return out
    if msg is not None:
        sites = ', '.join(f"{s}: {grasp_names.get(g, g).lower().replace('_', ' ')}"
                          for s, g in zip(msg.sites, msg.site_grasp))
        status = status_names.get(msg.status, str(msg.status))
        out.append(LiveCheck('latest state', status in ('READY', 'BUSY'),
                             f"{status} (adapter '{msg.adapter}')" + (f'; sites: {sites}' if sites else '')))
    elif pubs:
        out.append(LiveCheck('latest state', None, f'no message on {state_topic} within {timeout_s:g} s'))
    return out


__all__ = ['eef_checks', 'latest_state']
