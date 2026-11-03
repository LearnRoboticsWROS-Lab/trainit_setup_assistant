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

Depth UNITS are converted by trainit_perception's ``depth_to_metres`` (TrainIt D0.8, the KNOWN_DEBT rule: one
depth decoding per runtime side -- TSA reuses TP's, it does not keep its own). This module only lays the
message's rows out as an array (the row stride); without trainit_perception the depth is unavailable and
``last_error`` says why (the colour still shows).
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
        try:
            from trainit_perception.detectors.base import depth_to_metres
        except Exception as exc:  # noqa: BLE001 - the colour preview works without it
            depth_to_metres = None
            self._depth_unavailable = f'depth needs trainit_perception (its depth_to_metres): {exc!r}'
        self._depth_to_metres = depth_to_metres
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
            depth, error = None, getattr(self, '_depth_unavailable', None)
            if self._depth_to_metres is not None:
                dh, dw = depth_msg.height, depth_msg.width
                pixel = {'32FC1': np.float32, '16UC1': np.uint16}.get(depth_msg.encoding)
                if pixel is None:
                    raise ValueError(f'unsupported depth encoding {depth_msg.encoding!r}')
                size = np.dtype(pixel).itemsize
                raw = np.frombuffer(depth_msg.data, np.uint8).reshape(
                    dh, depth_msg.step)[:, :dw * size].copy()
                depth = self._depth_to_metres(raw.view(pixel).reshape(dh, dw), depth_msg.encoding)
            with self._lock:
                self._latest = (rgb.copy(), depth)
                self._frame_id = rgb_msg.header.frame_id
                self._last_error = error
        except Exception as exc:  # noqa: BLE001 — a bad frame must not kill the thread
            with self._lock:
                self._last_error = repr(exc)      # …but the drop reason is surfaced

    def _on_info(self, msg):
        with self._lock:
            self._K = self._np.array(msg.k, dtype=float).reshape(3, 3)

    # --- Qt-side polling ----------------------------------------------------------
    def latest(self) -> Optional[Tuple]:
        """(rgb HxWx3 uint8 RGB, depth_m HxW float32 (None without trainit_perception), K 3x3, frame_id) or
        None."""
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


def sniff_camera_topics(timeout_s: float = 2.0) -> dict:
    """One-shot read of the live ROS graph: find the camera's rgb/depth/camera_info
    topics by TYPE (sensor_msgs Image/CameraInfo) and NAME heuristics (depth vs
    color). Returns {'rgb': ..., 'depth': ..., 'camera_info': ...} with '' for
    anything not found. Own short-lived node; shares an already-init'd context."""
    import time
    import rclpy
    from rclpy.node import Node

    owns = not rclpy.ok()
    if owns:
        rclpy.init()
    node = Node('trainit_topic_sniffer')
    try:
        deadline = time.time() + timeout_s
        images, infos = [], []
        while time.time() < deadline:
            pairs = node.get_topic_names_and_types()
            images = [n for n, ts in pairs if 'sensor_msgs/msg/Image' in ts]
            infos = [n for n, ts in pairs if 'sensor_msgs/msg/CameraInfo' in ts]
            if images and infos:
                break
            time.sleep(0.2)

        def _is_depth(name: str) -> bool:
            return 'depth' in name.lower()

        depth = next((n for n in images if _is_depth(n)), '')
        rgb = next((n for n in images if not _is_depth(n)), '')
        # prefer the CameraInfo that shares a prefix with the rgb topic
        info = ''
        if infos:
            base = rgb.rsplit('/', 1)[0] if rgb else ''
            info = next((n for n in infos if base and n.startswith(base)),
                        next((n for n in infos if not _is_depth(n)), infos[0]))
        return {'rgb': rgb, 'depth': depth, 'camera_info': info}
    finally:
        node.destroy_node()
        if owns and rclpy.ok():
            rclpy.shutdown()
