"""Preview a waypoint in RViz (vNext I0.4, I0_DESIGN.md §12; OPTIONAL, gated by the live spike: PASS).

What it does: it puts the waypoint in RViz's MotionPlanning display as the GOAL state, so the engineer
sees it. It sends two messages, both visualization requests:
- the planning group on ``/rviz/moveit/select_planning_group``;
- the goal state on ``/rviz/moveit/update_custom_goal_state``.

What it never does: plan, execute, or send a trajectory (ADR-0027). The engineer presses Plan / Execute
in RViz. The display listens only when its "External Comm." option is on; :meth:`RvizPreview.listening`
says whether it does.

A TCP waypoint is turned into joints by MoveIt's IK service (``compute_ik``): a computation, not a
motion. It is seeded with the robot's current joints and must be collision-free.
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional, Sequence

GOAL_TOPIC = '/rviz/moveit/update_custom_goal_state'
GROUP_TOPIC = '/rviz/moveit/select_planning_group'


def compute_ik_service(namespace: str = '') -> str:
    ns = (namespace or '').strip('/')
    return f'/{ns}/compute_ik' if ns else '/compute_ik'


class RvizPreview:
    def __init__(self, live):
        from moveit_msgs.msg import RobotState
        from std_msgs.msg import String
        self.live = live
        self._goal = live.node.create_publisher(RobotState, GOAL_TOPIC, 1)
        self._group = live.node.create_publisher(String, GROUP_TOPIC, 1)

    def listening(self) -> bool:
        """RViz's MotionPlanning display subscribes ("External Comm." on)."""
        return self._goal.get_subscription_count() > 0

    def show_joints(self, group: str, names: Sequence[str], values: Sequence[float]) -> None:
        from moveit_msgs.msg import RobotState
        from std_msgs.msg import String
        self._group.publish(String(data=group))
        time.sleep(0.3)                                  # the display switches its group first
        msg = RobotState()
        msg.is_diff = True
        msg.joint_state.name = list(names)
        msg.joint_state.position = [float(v) for v in values]
        self._goal.publish(msg)

    MAX_STARTS = 5                                    # the current joints + at most 4 taught configurations

    def ik(self, project, position: Sequence[float], orientation: Sequence[float],
           frame: Optional[str] = None, timeout_s: float = 2.0) -> Dict[str, float]:
        """The planning group's joints placing the TCP at the pose (collision-checked IK): the solution NEAREST
        the robot's current joints among the ones found from the current joints, the group's named states and
        the program's joint waypoints. Owner Test D (friction row 33): one IK start (the mock's stretched
        all-zero start) failed on a pose four other starts reached, or found an odd wrist. Raises
        ``RuntimeError`` naming the reason when MoveIt finds none."""
        from geometry_msgs.msg import PoseStamped
        from moveit_msgs.srv import GetPositionIK
        robot = project.robot
        joints = list(robot.planning_group.joints)
        client = self.live.client(GetPositionIK, compute_ik_service(robot.cell_namespace))   # long-lived
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f'no IK service on {compute_ik_service(robot.cell_namespace)} '
                               '(is the planning environment running?)')
        current = self.live.current_joint_values(None, timeout_s=timeout_s)
        ps = PoseStamped()
        ps.header.frame_id = frame or robot.base_frame
        (ps.pose.position.x, ps.pose.position.y, ps.pose.position.z) = [float(v) for v in position]
        (ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z,
         ps.pose.orientation.w) = [float(v) for v in orientation]
        starts = [(list(current), [float(v) for v in current.values()])]
        taught: List[List[float]] = []
        for values in self._taught_configurations(project):
            if len(starts) >= self.MAX_STARTS:
                break
            if any(max(abs(a - b) for a, b in zip(values, seen)) < 1e-6 for seen in taught):
                continue                                  # the same configuration twice
            taught.append(values)
            starts.append((joints, values))
        found, last = [], None
        for names, values in starts:
            req = GetPositionIK.Request()
            req.ik_request.group_name = robot.planning_group.name
            req.ik_request.ik_link_name = robot.tip_link
            req.ik_request.avoid_collisions = True
            req.ik_request.timeout.nanosec = 200_000_000
            req.ik_request.robot_state.joint_state.name = list(names)
            req.ik_request.robot_state.joint_state.position = list(values)
            req.ik_request.pose_stamped = ps
            future = client.call_async(req)
            deadline = time.time() + timeout_s + 1.0
            while not future.done() and time.time() < deadline:
                time.sleep(0.02)
            res = future.result() if future.done() else None
            if res is None:
                raise RuntimeError('the IK service did not answer')
            last = res.error_code.val
            if last == 1:
                sol = dict(zip(res.solution.joint_state.name, res.solution.joint_state.position))
                found.append({j: sol[j] for j in joints if j in sol})
        if not found:
            raise RuntimeError(f'MoveIt finds no collision-free joint solution for this pose from {len(starts)} '
                               f'start configurations (error {last})')
        here = [current.get(j, 0.0) for j in joints]
        return min(found, key=lambda sol: sum(abs(sol.get(j, 0.0) - h) for j, h in zip(joints, here)))

    @staticmethod
    def _taught_configurations(project) -> List[List[float]]:
        """The group's named states, then the program's joint waypoints: configurations the cell is taught in."""
        from ..composer.live_check import named_state_values
        joints = list(project.robot.planning_group.joints)
        out = [[v[j] for j in joints] for v in named_state_values(project).values() if all(j in v for j in joints)]
        app = project.application
        if app.is_task_program():
            out += [list(w.joints) for w in app.program.waypoints
                    if w.joints is not None and len(w.joints) == len(joints)]
        return out

    def preview_waypoint(self, project, wp: dict) -> str:
        """Show a waypoint (its canonical mapping) as RViz's goal state; returns what was shown."""
        robot = project.robot
        group, joints = robot.planning_group.name, list(robot.planning_group.joints)
        if wp.get('relative'):
            raise RuntimeError('a relative waypoint is resolved when the program runs: preview the waypoint '
                               'it is relative to')
        if wp.get('type') == 'joint' and wp.get('named'):
            state = next((s for s in robot.named_states if s.name == wp['named']), None)
            if state is None:
                raise RuntimeError(f"named state '{wp['named']}' is not in the project's MoveIt mirror")
            values = [state.joint_values[j] for j in joints]
        elif wp.get('type') == 'joint':
            values = list(wp['joints'])
        else:
            sol = self.ik(project, wp['position'], wp['orientation'], wp.get('frame'))
            values = [sol[j] for j in joints]
        self.show_joints(group, joints, values)
        return f"'{wp['name']}' shown as RViz's goal: press Plan, then Execute, in RViz"

    def close(self) -> None:
        for pub in (self._goal, self._group):
            try:
                self.live.node.destroy_publisher(pub)
            except Exception:  # noqa: BLE001
                pass


__all__ = ['GOAL_TOPIC', 'GROUP_TOPIC', 'RvizPreview', 'compute_ik_service']
