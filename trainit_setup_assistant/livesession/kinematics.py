"""MoveIt's computation services for the live waypoint check (vNext I0, owner Test C;
``composer.live_check``): ``compute_ik``, ``compute_fk`` and ``check_state_validity`` of the running
``move_group``, in the robot's namespace (ADR-0023).

They compute; nothing plans or moves (ADR-0027). rclpy and the MoveIt messages are imported lazily.
"""

from __future__ import annotations

import time
from typing import List, Optional, Sequence, Tuple

from ..composer.live_check import NotDecided

# a whole check gives up after this: the GUI waits for it (a slow move_group must not freeze TSA for minutes)
CHECK_BUDGET_S = 20.0              # per check: the waypoints, then the motions with a budget of their own
IK_TIMEOUT_NS = 500_000_000          # per IK request: KDL restarts from random seeds within it


def move_group_service(name: str, namespace: str = '') -> str:
    ns = (namespace or '').strip('/')
    return f'/{ns}/{name}' if ns else f'/{name}'


class LiveKinematics:
    """IK, FK and state validity of the project's planning group, through the live session's node."""

    def __init__(self, live, project, timeout_s: float = 2.0, budget_s: float = CHECK_BUDGET_S):
        self.live, self.project, self.timeout_s = live, project, timeout_s
        self.budget_s = budget_s
        self.deadline = time.monotonic() + budget_s
        robot = project.robot
        self.group, self.tip = robot.planning_group.name, robot.tip_link
        self.joints = list(robot.planning_group.joints)
        self.namespace = robot.cell_namespace

    def _call(self, srv_type, name: str, request):
        """One request; a missing service, a request without an answer or the whole check over its budget
        stops the check (RuntimeError): the first slow answer ends it, never N of them."""
        service = move_group_service(name, self.namespace)
        if time.monotonic() > self.deadline:
            raise RuntimeError(f'the check stopped after {self.budget_s:g} s (its time budget): MoveIt answers too '
                               'slowly, or there is much to check')
        client = self.live.client(srv_type, service)                 # long-lived (the session's)
        if not client.wait_for_service(timeout_sec=self.timeout_s):
            raise RuntimeError(f'no {service} (is the planning environment running?)')
        future = client.call_async(request)
        deadline = time.monotonic() + self.timeout_s + 1.0
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.02)                                          # the session's executor spins
        if not future.done() or future.result() is None:
            remove = getattr(client, 'remove_pending_request', None)
            if remove is not None:
                try:
                    remove(future)                                    # a late answer is not kept
                except Exception:  # noqa: BLE001
                    pass
            raise RuntimeError(f'{service} did not answer')
        return future.result()

    @staticmethod
    def _code_name(code: int) -> str:
        from moveit_msgs.msg import MoveItErrorCodes
        names = {v: k for k, v in vars(MoveItErrorCodes).items() if k.isupper() and isinstance(v, int)}
        return names.get(code, str(code))

    def _state(self, msg, joints: Sequence[float]) -> None:
        msg.joint_state.name = list(self.joints)
        msg.joint_state.position = [float(v) for v in joints]

    def ik(self, frame: str, position: Sequence[float], orientation: Sequence[float],
           seed: Optional[Sequence[float]], collisions: bool) -> Optional[List[float]]:
        """The group's joints placing the tip at the pose, or None when MoveIt finds no solution. Seeded with
        ``seed``, else MoveIt's own current state; ``collisions``: the solution must be collision-free in the
        planning scene. A request MoveIt refuses (an unknown frame, group or link) raises NotDecided."""
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionIK
        req = GetPositionIK.Request()
        req.ik_request.group_name = self.group
        req.ik_request.ik_link_name = self.tip
        req.ik_request.avoid_collisions = bool(collisions)
        req.ik_request.timeout.nanosec = IK_TIMEOUT_NS
        if seed is None:
            req.ik_request.robot_state.is_diff = True                 # empty: move_group's current state
        else:
            self._state(req.ik_request.robot_state, seed)
        ps = PoseStamped()
        ps.header.frame_id = frame
        (ps.pose.position.x, ps.pose.position.y, ps.pose.position.z) = [float(v) for v in position]
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = [float(v) for v in orientation]
        req.ik_request.pose_stamped = ps
        res = self._call(GetPositionIK, 'compute_ik', req)
        code = res.error_code.val
        if code in (-31, -6):                                         # NO_IK_SOLUTION, TIMED_OUT
            return None
        if code != 1:
            raise NotDecided(f'MoveIt refuses the IK request ({self._code_name(code)}: a frame, the group or the '
                             'tip unknown to it?)')
        values = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
        return [values[j] for j in self.joints]

    def fk(self, joints: Sequence[float], frame: str) -> Tuple[List[float], List[float]]:
        """The tip pose in ``frame`` for the group's joints."""
        from moveit_msgs.srv import GetPositionFK
        req = GetPositionFK.Request()
        req.header.frame_id = frame
        req.fk_link_names = [self.tip]
        self._state(req.robot_state, joints)
        res = self._call(GetPositionFK, 'compute_fk', req)
        if res.error_code.val != 1 or not res.pose_stamped:
            raise NotDecided(f'MoveIt computes no FK for these joints ({self._code_name(res.error_code.val)})')
        p = res.pose_stamped[0].pose
        return ([p.position.x, p.position.y, p.position.z],
                [p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w])

    def scene_objects(self) -> List[str]:
        """The ids of the world objects in move_group's planning scene."""
        from moveit_msgs.msg import PlanningSceneComponents
        from moveit_msgs.srv import GetPlanningScene
        req = GetPlanningScene.Request()
        req.components.components = PlanningSceneComponents.WORLD_OBJECT_NAMES
        res = self._call(GetPlanningScene, 'get_planning_scene', req)
        return [o.id for o in res.scene.world.collision_objects]

    def valid(self, joints: Sequence[float]) -> Tuple[bool, str]:
        """Is the state collision-free in the planning scene? (and the contacts when it is not)"""
        from moveit_msgs.srv import GetStateValidity
        req = GetStateValidity.Request()
        req.group_name = self.group
        self._state(req.robot_state, joints)
        res = self._call(GetStateValidity, 'check_state_validity', req)
        contacts = sorted({f'{c.contact_body_1} / {c.contact_body_2}' for c in res.contacts})
        return bool(res.valid), ', '.join(contacts)


__all__ = ['CHECK_BUDGET_S', 'LiveKinematics', 'move_group_service']
