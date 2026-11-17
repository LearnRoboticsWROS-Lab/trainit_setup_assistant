"""The camera -> robot frame chain, checked on the live session (TSA 8.8.0, vNext M4.5d; the owner's brief §23).

A perception observation is measured in the camera's OPTICAL frame; the motion runtime needs it in the robot's
planning frame. The runtime transforms it with TF2 at the image's timestamp (the M4 contract, TMR's
SetWaypointFromCandidate / DetectObject). TSA does NOT repeat that mathematics: this module only ASKS the live TF2
tree the same question the runtime will ask, and reports what the user needs to trust the chain:

    PERCEPTION FRAME   the CameraInfo's frame_id, and whether it is the optical frame the project names
    TARGET FRAME       the robot's planning frame (the project's base frame)
    TF STATUS          AVAILABLE (the transform at the latest image's stamp, its translation and rotation)
                       or MISSING (and TF2's own reason)
    CALIBRATION        intrinsics VALID (a CameraInfo with a non-degenerate K) or MISSING; the extrinsics ARE the
                       TF chain above (a camera mount in the description, or a published calibration)

`evaluate()` is pure (no ROS: unit-tested); `probe()` reads the live graph (CameraInfo once, TF2 at its stamp).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

AVAILABLE, MISSING = 'AVAILABLE', 'MISSING'
VALID = 'VALID'


@dataclass
class CameraFrameReport:
    perception_frame: str = ''
    expected_optical_frame: str = ''
    target_frame: str = ''
    tf_status: str = MISSING
    tf_detail: str = ''
    translation_m: Optional[Sequence[float]] = None
    rotation_xyzw: Optional[Sequence[float]] = None
    calibration: str = MISSING
    calibration_detail: str = ''
    stamp: str = ''
    problems: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.tf_status == AVAILABLE and self.calibration == VALID and not self.problems

    def lines(self) -> list:
        t = ', '.join(f'{v:+.4f}' for v in self.translation_m) if self.translation_m else '-'
        out = [f'PERCEPTION FRAME   {self.perception_frame or "(no CameraInfo)"}',
               f'TARGET FRAME       {self.target_frame}',
               f'TF STATUS          {self.tf_status}' + (f' at {self.stamp}: translation [{t}] m' if
                                                         self.tf_status == AVAILABLE else f' ({self.tf_detail})'),
               f'CALIBRATION        {self.calibration} ({self.calibration_detail})']
        out += [f'PROBLEM            {p}' for p in self.problems]
        return out


def intrinsics_status(k: Sequence[float], width: int, height: int) -> tuple:
    """(VALID | MISSING, why) of a CameraInfo's K (row-major 3x3) and size."""
    if not k or len(k) != 9 or not all(math.isfinite(float(x)) for x in k):
        return MISSING, 'no K'
    fx, fy, cx, cy = float(k[0]), float(k[4]), float(k[2]), float(k[5])
    if fx <= 0 or fy <= 0:
        return MISSING, f'fx {fx:g}, fy {fy:g}: not a calibrated camera'
    if width <= 0 or height <= 0 or not (0 < cx < width and 0 < cy < height):
        return MISSING, f'principal point ({cx:g}, {cy:g}) outside the {width}x{height} image'
    return VALID, f'fx {fx:.1f} fy {fy:.1f} cx {cx:.1f} cy {cy:.1f}, {width}x{height}'


def evaluate(info: Optional[Dict], target_frame: str, expected_optical_frame: str,
             transform: Optional[Dict], tf_error: str = '') -> CameraFrameReport:
    """The report from what the live graph answered: `info` = {frame_id, k, width, height, stamp} of one CameraInfo
    (None: none arrived); `transform` = {translation, rotation_xyzw} of TF2's lookup target <- frame_id at the stamp
    (None: refused, `tf_error` = TF2's reason)."""
    r = CameraFrameReport(target_frame=target_frame, expected_optical_frame=expected_optical_frame)
    if info is None:
        r.tf_detail = 'no CameraInfo arrived: is the camera publishing?'
        r.calibration_detail = 'no CameraInfo'
        r.problems.append('no CameraInfo on the configured topic')
        return r
    r.perception_frame = str(info.get('frame_id') or '')
    r.stamp = str(info.get('stamp', ''))
    r.calibration, r.calibration_detail = intrinsics_status(info.get('k') or (), int(info.get('width', 0)),
                                                            int(info.get('height', 0)))
    if expected_optical_frame and r.perception_frame != expected_optical_frame:
        r.problems.append(f"the CameraInfo's frame is '{r.perception_frame}', the project names "
                          f"'{expected_optical_frame}' (Step 5, Optical frame)")
    if transform is None:
        r.tf_status, r.tf_detail = MISSING, tf_error or 'TF2 has no transform'
    else:
        r.tf_status = AVAILABLE
        r.translation_m = [float(x) for x in transform['translation']]
        r.rotation_xyzw = [float(x) for x in transform['rotation_xyzw']]
    return r


def probe(camera_info_topic: str, target_frame: str, expected_optical_frame: str,
          timeout_s: float = 2.0) -> CameraFrameReport:  # pragma: no cover - needs a live ROS graph
    """Read one CameraInfo and ask TF2 for target_frame <- its frame at its stamp (the runtime's own question).

    Scope: TF2's ABSOLUTE /tf and /tf_static (a robot whose TF is remapped into a namespace, ADR-0023, is not
    followed: TF MISSING then names the frame TF2 could not find); a relative topic name resolves in the root."""
    import time

    import rclpy
    import tf2_ros
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import qos_profile_sensor_data
    from rclpy.time import Time
    from sensor_msgs.msg import CameraInfo

    if not rclpy.ok():
        rclpy.init()
    node = Node('trainit_camera_frames_probe')
    # its OWN executor: the probe runs on a worker thread (the GUI stays responsive) and must not spin, nor be
    # spun by, the global executor another part of the session may use
    ex = SingleThreadedExecutor()
    ex.add_node(node)
    try:
        buf = tf2_ros.Buffer()
        tf2_ros.TransformListener(buf, node)
        got = {}
        # sensor-data QoS (BEST_EFFORT): it matches a RELIABLE publisher too; a RELIABLE subscriber would never
        # match a camera driver publishing camera_info best-effort (a healthy camera reported as silent)
        node.create_subscription(CameraInfo, camera_info_topic, lambda m: got.setdefault('m', m),
                                 qos_profile_sensor_data)
        end = time.monotonic() + timeout_s
        while 'm' not in got and time.monotonic() < end:
            ex.spin_once(timeout_sec=0.05)
        m = got.get('m')
        if m is None:
            return evaluate(None, target_frame, expected_optical_frame, None)
        info = {'frame_id': m.header.frame_id, 'k': list(m.k), 'width': m.width, 'height': m.height,
                'stamp': f'{m.header.stamp.sec}.{m.header.stamp.nanosec:09d}'}
        stamp = Time.from_msg(m.header.stamp)
        tf, err = None, ''
        end = time.monotonic() + timeout_s
        while time.monotonic() < end:
            ex.spin_once(timeout_sec=0.05)
            try:
                t = buf.lookup_transform(target_frame, m.header.frame_id, stamp, timeout=Duration(seconds=0.0))
                tr, q = t.transform.translation, t.transform.rotation
                tf = {'translation': [tr.x, tr.y, tr.z], 'rotation_xyzw': [q.x, q.y, q.z, q.w]}
                break
            except Exception as exc:  # noqa: BLE001 - TF2's own reason is the report
                err = str(exc).splitlines()[0]
        return evaluate(info, target_frame, expected_optical_frame, tf, err)
    finally:
        ex.remove_node(node)
        node.destroy_node()
