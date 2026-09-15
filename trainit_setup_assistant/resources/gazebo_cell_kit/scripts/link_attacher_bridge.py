#!/usr/bin/env python3
"""LinkAttacher grasp bridge for the TrainIt Gazebo backend (ADR-0008 F2).

Gazebo Classic has no suction physics, so a suction grasp is simulated by a rigid weld:
the IFRA_LinkAttacher plugin's ``/ATTACHLINK`` / ``/DETACHLINK`` services. This node adapts
that to the **TrainIt suction contract** so the existing BT (CloseGripper/OpenGripper ->
``/suction/on`` / ``/suction/off``) drives the weld unchanged, exactly like the real cell's
vacuum services and the Isaac SurfaceGripper adapter.

It runs ONLY in the ``gazebo`` backend (add it to ``DeploymentSpec.bridges`` with
``modes: [gazebo]``); mock/isaac/real never load it. It depends on ``linkattacher_msgs``
(clone IFRA_LinkAttacher into the workspace — Gazebo backend only).

Parameters:
  object_model (str)  the Gazebo model welded on grasp (e.g. "red_box")
  object_link  (str)  its link name (default "link")
  robot_model  (str)  the spawned robot model name (spawn_entity -entity <name>)
  ee_link      (str)  the end-effector link the object welds to (e.g. "suction_link")
  on_service   (str)  grasp service (default "/suction/on")
  off_service  (str)  release service (default "/suction/off")
"""
import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

try:
    from linkattacher_msgs.srv import AttachLink, DetachLink
except ImportError:  # pragma: no cover - only importable with IFRA_LinkAttacher present
    AttachLink = DetachLink = None


class LinkAttacherBridge(Node):
    def __init__(self):
        super().__init__('link_attacher_bridge')
        self.declare_parameter('object_model', 'red_box')
        self.declare_parameter('object_link', 'link')
        self.declare_parameter('robot_model', 'robot')
        self.declare_parameter('ee_link', 'suction_link')
        self.declare_parameter('on_service', '/suction/on')
        self.declare_parameter('off_service', '/suction/off')
        g = lambda n: self.get_parameter(n).get_parameter_value().string_value
        self._obj_model, self._obj_link = g('object_model'), g('object_link')
        self._robot_model, self._ee_link = g('robot_model'), g('ee_link')

        if AttachLink is None:
            self.get_logger().error(
                'linkattacher_msgs not found: clone IFRA_LinkAttacher into the workspace '
                '(Gazebo backend only). The suction services are inert until then.')
        self._attach = self.create_client(AttachLink, '/ATTACHLINK') if AttachLink else None
        self._detach = self.create_client(DetachLink, '/DETACHLINK') if DetachLink else None

        self.create_service(Trigger, g('on_service'), self._on)
        self.create_service(Trigger, g('off_service'), self._off)
        self.get_logger().info(
            f'LinkAttacher bridge: {g("on_service")}/{g("off_service")} weld '
            f'{self._robot_model}/{self._ee_link} <-> {self._obj_model}/{self._obj_link}')

    def _call(self, client, srv_type, resp):
        if client is None:
            resp.success = False
            resp.message = 'linkattacher_msgs unavailable (IFRA_LinkAttacher not built)'
            return resp
        if not client.wait_for_service(timeout_sec=2.0):
            resp.success = False
            resp.message = f'{client.srv_name} not available'
            return resp
        req = srv_type.Request()
        req.model1_name, req.link1_name = self._robot_model, self._ee_link
        req.model2_name, req.link2_name = self._obj_model, self._obj_link
        client.call_async(req)   # fire-and-forget; the weld applies asynchronously
        resp.success = True
        resp.message = f'{client.srv_name} requested'
        return resp

    def _on(self, _req, resp):
        return self._call(self._attach, AttachLink, resp)

    def _off(self, _req, resp):
        return self._call(self._detach, DetachLink, resp)


def main(args=None):
    rclpy.init(args=args)
    node = LinkAttacherBridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
