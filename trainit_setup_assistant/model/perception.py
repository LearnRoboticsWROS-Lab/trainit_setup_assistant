"""Perception model (D-014 / D-015): the camera and the detectors as NAMED RESOURCES.

The dedicated Perception step configures *what the camera sees and how* (this module);
the application step only BINDS a detector to waypoints (``Waypoint.vision``). One
detector serves any number of applications.

Everything here is DATA consumed by three emitters:
- ``perception.yaml`` + one ``detector_node`` per pure-python detector in ``_app``;
- the camera-variant URDF include rewrite + the base↔camera SRDF pair in
  ``_description`` and ``_trainit_config``;
- the synthetic coloured cloud (``depth_image_proc``) in the isaac bring-up branch.

Generation stays pure: nothing here touches ROS. The measured cell values (settle
1500 ms, dz offsets) are PROJECT data with defaults, never generator constants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .enums import DetectionMethod


class CameraSpec(BaseModel):
    """The cell camera: frames, topics, and how the robot description gains it."""

    optical_frame: str = 'camera_color_optical_frame'
    # The camera's root link in the URDF — used for the SRDF pair (mount↔camera
    # disabled "Never"; every camera-vs-arm pair stays CHECKED so planning avoids it).
    link: str = 'camera_link'
    rgb_topic: str = '/camera/color/image_raw'
    depth_topic: str = '/camera/depth/image_rect_raw'
    camera_info_topic: str = '/camera/color/camera_info'
    points_topic: str = '/camera/depth/color/points'
    # URDF variant rewrite (both copied packages): the include whose ``filename``
    # equals ``replace_include`` is swapped for ``with_include``. Data, not a generator
    # rule — so the generator stays robot-agnostic and the project carries the paths.
    replace_include: Optional[str] = None
    with_include: Optional[str] = None
    # sim builds the coloured cloud (depth_image_proc, isaac branch only); in real the
    # vendor driver already publishes ``points_topic``.
    synthetic_cloud_in_sim: bool = True


class DetectorSpec(BaseModel):
    """One named detector: a method plus its tuned parameters.

    ``params`` mirrors the runtime profile VERBATIM (``trainit_perception``'s
    ``defaults()`` keys, ``class_id`` included) — the live tuner and the generated
    ``perception.yaml`` serialize the same dict, which is the "TSA generates what you
    tuned" guarantee.
    """

    name: str                                   # -> /perception/<name>/detections
    method: DetectionMethod = DetectionMethod.COLOR_MASK
    params: Dict[str, Any] = Field(default_factory=dict)
    # Input overrides; None => the camera's topics.
    rgb_topic: Optional[str] = None
    depth_topic: Optional[str] = None
    camera_info_topic: Optional[str] = None
    # Trigger model (D-015): continuous at rate_hz keeps the pipeline warm (default,
    # proof-2 validated); on demand the same node answers the std_srvs/Trigger service.
    continuous: bool = True
    rate_hz: float = 10.0

    @property
    def class_id(self) -> str:
        return str(self.params.get('class_id', 'object'))

    @property
    def emits_node(self) -> bool:
        """True when the assistant generates a detector_node for this method
        (pure-python methods). External-node methods publish the contract themselves."""
        return self.method in (DetectionMethod.COLOR_MASK, DetectionMethod.CUSTOM)


class VisionBinding(BaseModel):
    """Binds a waypoint to a detector: SetWaypointFromDetection, as data.

    ``orientation``: ``keep`` (the waypoint's own), ``detected``, ``from:<waypoint>``
    (borrow another waypoint's orientation — how a joint-named approach pose is
    promoted to tcp), or a literal ``qx;qy;qz;qw``.
    """

    detector: str
    dz: float = 0.0                             # metres, along base-frame Z
    orientation: str = 'keep'


class PerceptionSpec(BaseModel):
    camera: Optional[CameraSpec] = None
    detectors: List[DetectorSpec] = Field(default_factory=list)
    # Settle barrier after a scene-mutating block that precedes DetectObject in the
    # cycle: arrival freshness is not content freshness (measured 0.4 s on the cell;
    # 1500 ms = 3.7x margin). Emitted as <Sleep> right after <ResetScene/>.
    settle_ms: int = Field(default=1500, ge=0)
    detect_timeout_ms: int = Field(default=3000, ge=1)

    def detector_by_name(self, name: str) -> Optional[DetectorSpec]:
        for d in self.detectors:
            if d.name == name:
                return d
        return None
