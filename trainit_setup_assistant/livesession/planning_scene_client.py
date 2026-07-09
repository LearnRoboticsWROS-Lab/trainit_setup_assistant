"""Publish scene collision objects to the live /planning_scene for RViz preview.

Static+collision objects become box CollisionObjects (the engine plans boxes; non-box
primitives are previewed as their AABB). Dynamic objects are skipped (they are not
planning collisions). rclpy is imported lazily so the package works without ROS.
"""

from __future__ import annotations


class PlanningScenePublisher:
    def __init__(self, node_name: str = 'trainit_planning_scene_pub'):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, DurabilityPolicy
        from moveit_msgs.msg import PlanningScene

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()
        self._node = Node(node_name)
        qos = QoSProfile(depth=1)
        qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self._pub = self._node.create_publisher(PlanningScene, '/planning_scene', qos)

    def publish_scene(self, scene_spec) -> int:
        """Publish all static+collision objects. Returns the count published."""
        from moveit_msgs.msg import CollisionObject, PlanningScene
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose

        msg = PlanningScene()
        msg.is_diff = True
        count = 0
        for obj in scene_spec.objects:
            if not obj.is_planning_collision():
                continue
            co = CollisionObject()
            co.id = obj.id
            co.header.frame_id = obj.frame
            box = SolidPrimitive()
            box.type = SolidPrimitive.BOX
            box.dimensions = [float(d) for d in obj.aabb_dims()]
            pose = Pose()
            pose.position.x, pose.position.y, pose.position.z = (
                float(obj.position[0]), float(obj.position[1]), float(obj.position[2]))
            q = obj.orientation
            pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = (
                float(q[0]), float(q[1]), float(q[2]), float(q[3]))
            co.primitives = [box]
            co.primitive_poses = [pose]
            co.operation = CollisionObject.ADD
            msg.world.collision_objects.append(co)
            count += 1
        self._pub.publish(msg)
        return count

    def close(self) -> None:
        try:
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
