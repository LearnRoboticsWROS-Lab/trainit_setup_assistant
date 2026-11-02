"""Read live joint states + TF poses from the running config session.

Used by the wizard's "Capture" buttons: capture a named state (current joint values)
or a waypoint pose (current base->tip transform). Spins its own node on a background
executor thread so it coexists with the Qt event loop.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, List, Optional, Tuple

from ..composer.integration import LIVE_MAX_AGE_S


def stale_message(topic: str, age: float) -> str:
    return (f'the last message on {topic} is {age:.0f} s old: nothing publishes it now '
            '(is the planning environment running?)')


def joint_states_topic(namespace: str = '') -> str:
    """The robot's joint-state topic (ADR-0023): ``joint_states`` resolved in the robot's namespace
    (``cell_namespace``, canonical form ``/a/b`` or empty), so a namespaced robot is read, not the
    root. TF stays global (``/tf``, ``/tf_static``), as ADR-0023 keeps it."""
    ns = (namespace or '').strip('/')
    return f'/{ns}/joint_states' if ns else '/joint_states'


class LiveCapture:
    """A small rclpy node that caches the latest joint states and serves TF lookups.

    Construct it while the live session runs; call ``current_joint_values`` /
    ``current_pose``; call ``close()`` when done. rclpy is imported here so importing
    this module never requires ROS. ``namespace`` is the robot's ``cell_namespace``
    (vNext I0.0: the joint states are read in it, ADR-0023).
    """

    def __init__(self, node_name: str = 'trainit_live_capture', namespace: str = ''):
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
        self._latest_js_at: Optional[float] = None          # time.monotonic() of the last message
        self._js_lock = threading.Lock()
        self.joint_states_topic = joint_states_topic(namespace)
        self._node.create_subscription(JointState, self.joint_states_topic, self._on_js, 10)
        self._tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self._tf_buffer, self._node)

        self._clients = {}
        self.last_error: Optional[Exception] = None
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    @property
    def node(self):
        """The capture's node (spun by its own executor thread): live checks create their clients on
        it."""
        return self._node

    def client(self, srv_type, name: str):
        """A service client on this node, created ONCE and kept until :meth:`close` (never destroyed while
        the executor waits on it; vNext I0.4 review)."""
        key = (srv_type, name)
        if key not in self._clients:
            self._clients[key] = self._node.create_client(srv_type, name)
        return self._clients[key]

    def _spin(self):
        while self._spinning and self._rclpy.ok():
            try:
                self._executor.spin_once(timeout_sec=0.1)
            except Exception as exc:  # noqa: BLE001 - one bad callback must not end every later read
                self.last_error = exc
                time.sleep(0.1)

    def _on_js(self, msg):
        with self._js_lock:
            self._latest_js = msg
            self._latest_js_at = time.monotonic()

    def joint_states_age(self) -> Optional[float]:
        """Seconds since the last joint-state message (None: none yet). What "the robot is LIVE"
        rests on (I0_DESIGN.md §8)."""
        with self._js_lock:
            at = self._latest_js_at
        return None if at is None else time.monotonic() - at

    def current_joint_values(
        self, joint_names: Optional[List[str]] = None, timeout_s: float = 3.0,
        max_age: Optional[float] = LIVE_MAX_AGE_S,
    ) -> Dict[str, float]:
        """Latest joint positions (filtered to ``joint_names`` if given), from a message at most
        ``max_age`` seconds old (None: any age). A robot whose joint states stopped is not LIVE: its last
        values are never handed out as the current ones (owner Test A.2: a stopped environment's last
        message, 78 s old, passed the live check)."""
        deadline = time.time() + timeout_s
        js = at = None
        while True:
            with self._js_lock:
                js, at = self._latest_js, self._latest_js_at
            if js is not None and (max_age is None or time.monotonic() - at <= max_age):
                values = dict(zip(js.name, js.position))
                if joint_names is None:
                    return values
                return {j: values[j] for j in joint_names if j in values}
            if time.time() >= deadline:
                break
            time.sleep(0.05)
        if js is not None:
            raise TimeoutError(stale_message(self.joint_states_topic, time.monotonic() - at))
        raise TimeoutError(f'no {self.joint_states_topic} received (is the live session running?)')

    def current_pose(
        self, base_frame: str, tip_link: str, timeout_s: float = 3.0,
        max_age: Optional[float] = LIVE_MAX_AGE_S,
    ) -> Tuple[List[float], List[float]]:
        """Current base_frame->tip_link transform as ([x,y,z], [qx,qy,qz,qw]), of a LIVE robot: TF keeps
        the last transform of a robot that stopped, so the joint states (what the robot's TF is computed
        from) must be at most ``max_age`` seconds old (None: no such check)."""
        from rclpy.time import Time
        if max_age is not None:
            self.current_joint_values(None, timeout_s=timeout_s, max_age=max_age)
        # Read the TF of NOW: forget what the buffer holds (the static transforms are kept) and wait for what
        # arrives after the click. tf2 drops data older than the newest it holds, so after a clock change --
        # a mock session (wall time) followed by Isaac (simulation time), or Isaac stopped and played again --
        # a long-lived buffer answers the OLD pose for ever (owner Test B: Capture TCP never changed).
        self._tf_buffer.clear()
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
            for c in self._clients.values():
                self._node.destroy_client(c)
            self._clients.clear()
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
