"""Read live joint states + TF poses from the running config session.

Used by the wizard's "Capture" buttons: capture a named state (current joint values)
or a waypoint pose (current base->tip transform). Spins its own node on a background
executor thread so it coexists with the Qt event loop.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple


class LiveCapture:
    """A small rclpy node that caches the latest /joint_states and serves TF lookups.

    Construct it while the live session runs; call ``current_joint_values`` /
    ``current_pose``; call ``close()`` when done. rclpy is imported here so importing
    this module never requires ROS.
    """

    def __init__(self, node_name: str = 'trainit_live_capture'):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        import tf2_ros

        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        self._node = Node(node_name)
        self._latest_js = None
        self._js_lock = threading.Lock()
        self._node.create_subscription(JointState, '/joint_states', self._on_js, 10)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._spinning and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_js(self, msg):
        with self._js_lock:
            self._latest_js = msg

    def current_joint_values(
        self, joint_names: Optional[List[str]] = None, timeout_s: float = 3.0
    ) -> Dict[str, float]:
        """Latest joint positions (filtered to ``joint_names`` if given)."""
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._js_lock:
                js = self._latest_js
            if js is not None:
                values = dict(zip(js.name, js.position))
                if joint_names is None:
                    return values
                return {j: values[j] for j in joint_names if j in values}
            time.sleep(0.05)
        raise TimeoutError('no /joint_states received (is the live session running?)')

    def current_pose(
        self, base_frame: str, tip_link: str, timeout_s: float = 3.0
    ) -> Tuple[List[float], List[float]]:
        """Current base_frame->tip_link transform as ([x,y,z], [qx,qy,qz,qw])."""
        from rclpy.time import Time
        deadline = time.time() + timeout_s
        last_exc = None
        while time.time() < deadline:
            try:
                tf = self._tf_buffer.lookup_transform(base_frame, tip_link, Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                return [t.x, t.y, t.z], [q.x, q.y, q.z, q.w]
            except Exception as exc:  # noqa: BLE001 - TF may not be ready yet
                last_exc = exc
                time.sleep(0.05)
        raise TimeoutError(f'TF {base_frame}->{tip_link} unavailable: {last_exc}')

    def close(self) -> None:
        self._spinning = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
