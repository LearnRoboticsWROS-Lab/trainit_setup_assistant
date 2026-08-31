"""Read live camera frames (rgb + depth + intrinsics) from the running session.

Used by the Perception step's live tuner: the ROS callbacks cache the LATEST
synchronized (rgb, depth-in-metres, K) triple under a lock on a background executor
thread; the Qt side POLLS the cache from a QTimer — widgets are never touched from
the ROS thread, and the GUI never blocks on frame delivery.

QoS matters and is silent when wrong: the sim camera publishes images BEST_EFFORT
(a RELIABLE subscriber never pairs and shows a black canvas with no error), while
CameraInfo is RELIABLE. The constants mirror trainit_perception's detector_node —
the tuner must see exactly what the runtime node sees.

rclpy/numpy are imported in ``__init__`` so importing this module never needs ROS.
"""

from __future__ import annotations

import threading
from typing import Optional, Tuple


class LiveCameraCapture:
    """Caches the latest synchronized camera frame from the live session."""

    def __init__(self,
                 rgb_topic: str = '/camera/color/image_raw',
                 depth_topic: str = '/camera/depth/image_rect_raw',
                 camera_info_topic: str = '/camera/color/camera_info',
                 node_name: str = 'trainit_camera_capture'):
        import numpy as np
        import rclpy
        from message_filters import ApproximateTimeSynchronizer, Subscriber
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
        from sensor_msgs.msg import CameraInfo, Image

        self._np = np
        self._rclpy = rclpy
        self._owns_context = not rclpy.ok()
        if self._owns_context:
            rclpy.init()

        self._node = Node(node_name)
        self._lock = threading.Lock()
        self._latest: Optional[tuple] = None      # (rgb, depth_m)
        self._K = None
        self._frame_id: Optional[str] = None
        self._last_error: Optional[str] = None    # why frames are being dropped

        sensor_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.BEST_EFFORT,
                                history=HistoryPolicy.KEEP_LAST)
        info_qos = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE,
                              history=HistoryPolicy.KEEP_LAST)
        self._sub_rgb = Subscriber(self._node, Image, rgb_topic, qos_profile=sensor_qos)
        self._sub_depth = Subscriber(self._node, Image, depth_topic,
                                     qos_profile=sensor_qos)
        self._sync = ApproximateTimeSynchronizer([self._sub_rgb, self._sub_depth],
                                                 queue_size=5, slop=0.05)
        self._sync.registerCallback(self._on_pair)
        self._node.create_subscription(CameraInfo, camera_info_topic,
                                       self._on_info, info_qos)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._spinning and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    # --- callbacks (executor thread) ---------------------------------------------
    def _on_pair(self, rgb_msg, depth_msg):
        np = self._np
        try:
            if rgb_msg.encoding not in ('rgb8', 'bgr8'):
                raise ValueError(f'unsupported rgb encoding {rgb_msg.encoding!r}')
            h, w = rgb_msg.height, rgb_msg.width
            # honour the row stride: cameras may pad rows (step != width * bpp)
            rgb = np.frombuffer(rgb_msg.data, np.uint8).reshape(
                h, rgb_msg.step)[:, :w * 3].reshape(h, w, 3)
            if rgb_msg.encoding == 'bgr8':
                rgb = rgb[:, :, ::-1]
            dh, dw = depth_msg.height, depth_msg.width
            if depth_msg.encoding == '32FC1':
                depth = np.frombuffer(depth_msg.data, np.uint8).reshape(
                    dh, depth_msg.step)[:, :dw * 4].reshape(dh, dw * 4)
                depth = depth.view(np.float32).reshape(dh, dw).copy()
            elif depth_msg.encoding == '16UC1':
                depth = np.frombuffer(depth_msg.data, np.uint8).reshape(
                    dh, depth_msg.step)[:, :dw * 2].reshape(dh, dw * 2)
                depth = depth.view(np.uint16).reshape(dh, dw).astype(np.float32) / 1000.0
            else:
                raise ValueError(f'unsupported depth encoding {depth_msg.encoding!r}')
            with self._lock:
                self._latest = (rgb.copy(), depth)
                self._frame_id = rgb_msg.header.frame_id
                self._last_error = None
        except Exception as exc:  # noqa: BLE001 — a bad frame must not kill the thread
            with self._lock:
                self._last_error = repr(exc)      # …but the drop reason is surfaced

    def _on_info(self, msg):
        with self._lock:
            self._K = self._np.array(msg.k, dtype=float).reshape(3, 3)

    # --- Qt-side polling ----------------------------------------------------------
    def latest(self) -> Optional[Tuple]:
        """(rgb HxWx3 uint8 RGB, depth_m HxW float32, K 3x3, frame_id) or None."""
        with self._lock:
            if self._latest is None or self._K is None:
                return None
            rgb, depth = self._latest
            return rgb, depth, self._K, self._frame_id

    def last_error(self) -> Optional[str]:
        """Why the newest frame was dropped (None when frames decode fine)."""
        with self._lock:
            return self._last_error

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
