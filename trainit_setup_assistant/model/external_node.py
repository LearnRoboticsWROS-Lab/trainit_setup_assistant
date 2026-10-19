"""How an EXTERNAL-NODE detector is LAUNCHED (D-015's second kind of method).

``trainit_perception``'s registry names two kinds of detection method, and the difference
is *who runs the code*:

- **pure-python** — a ``Detector`` subclass in the registry. TSA emits one
  ``trainit_perception`` ``detector_node`` per entry of ``config/perception.yaml``, and the
  live tuner and the bundle run the SAME ``detect()``.
- **external-node** — a method implemented by a SEPARATE node (C++/PCL, a learned model, a
  vendor SDK) that publishes the same contract (``vision_msgs/Detection3DArray`` on
  ``/perception/<name>/detections``). ``make_detector`` raises ``KeyError`` on it by
  design: there is no python class to construct.

Until this module, TSA could *name* the second kind (``DetectionMethod.CUSTOM`` /
``PCL_CLUSTER``, ``EXTERNAL_METHODS`` in the runtime registry) and emit **nothing at all**
for it: no package, no executable, no parameter file, no launch entry. So a real
YOLO-segmentation detector — which is exactly an external-node method — could be
configured, recorded in ``perception.yaml``, bound to a waypoint and ranked, and the
generated bundle still launched nothing that published its topic. The tree then timed out
in ``DetectObject``, which reads as a perception fault and is a configuration one.

:class:`ExternalNodeSpec` is the missing half: the launch half. It carries exactly what
``launch_ros.actions.Node`` needs and NOTHING ELSE —

======================  ==========================================================
``package``             the ROS package that holds the node (required)
``executable``          its entry point / console script (required)
``node_name``           the node name; default ``detector_<detector name>``, the same
                        convention the pure-python ``detector_node`` already follows
``parameter_file``      an absolute path, or a path relative to the package's SHARE
                        directory (``config/capsule.yaml``) — the file the node's own
                        author ships and maintains
``parameters``          inline node parameters, applied AFTER the file so they win;
                        this is where a ``model:=/abs/best.onnx`` command-line
                        argument lands as configuration instead of a habit
``topic_parameters``    role -> the PARAMETER NAME this node reads that role from, for
                        a node that takes its inputs as parameters
``remappings``          ``from -> to``, for a node whose subscriptions are hardcoded
``detect_service``      the node serves ``/perception/<name>/detect`` (std_srvs/Trigger), the
                        contract's on-demand service a capture calls (vNext M3.5)
======================  ==========================================================

Two deliberate omissions, both because the field already exists elsewhere:

1. **The input topics are not here.** ``DetectorSpec.rgb_topic`` / ``depth_topic`` /
   ``camera_info_topic`` (falling back to ``CameraSpec``) are already the project's single
   home for the camera contract, and the generated ``perception.yaml`` already records the
   resolved values. Repeating them here would let the file TSA writes and the node TSA
   launches disagree about which camera they are looking at — silently. So the project says
   only what the node CALLS them (``topic_parameters``) or what it subscribes to
   (``remappings``), and the VALUES come from the one place that has them.
2. **No class_id / ranking / rate.** Those are detector parameters and they already travel
   in ``DetectorSpec.params`` into ``perception.yaml``. An external node that wants them
   reads that file or declares its own; TSA does not re-author another node's parameters.

Pure data: nothing here imports ROS or touches the filesystem.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from pydantic import BaseModel, Field

# The camera contract's three roles, named with the SAME tokens the generated
# ``perception.yaml`` uses in its ``input:`` block — a configurator should not have to learn
# a second vocabulary for the same three topics.
INPUT_ROLES: Tuple[str, ...] = ('rgb', 'depth', 'camera_info')


class ExternalNodeSpec(BaseModel):
    """What an external-node detector needs in order to be LAUNCHED by the bundle.

    Present => the generated ``trainit_bt.launch.py`` starts this node next to the
    pure-python ``detector_node`` entries, so one bundle launches BOTH kinds of method the
    same way. Absent on an external-node method => TSA emits nothing for it and SAYS SO
    (a manifest warning): launching it yourself stays a legitimate choice, because the
    external node may be a vendor stack with its own bring-up.
    """

    package: str = ''
    executable: str = ''
    # None => ``detector_<detector name>``. Named explicitly only when the node's own
    # parameter file is keyed by a node name (ROS parameter files are), which is exactly
    # the real case: a node's ``capsule.yaml`` opens with ``seg_pose_node:``.
    node_name: Optional[str] = None
    # Absolute, or relative to ``package``'s share directory. Resolved in the generated
    # launch with ``get_package_share_directory`` — which is already imported there, so a
    # bundle with no external detector gains not one byte.
    parameter_file: Optional[str] = None
    # Inline parameters, applied after ``parameter_file``. A path a human types on the
    # command line today (``model:=/abs/best.onnx``) is CONFIGURATION: it belongs in the
    # project, where it is saved, reviewed and regenerated, not in shell history.
    parameters: Dict[str, Any] = Field(default_factory=dict)
    # role -> parameter name, e.g. ``{'rgb': 'rgb_topic'}``. The VALUE is filled by the
    # generator from the detector's own resolved input topics, so the project stays the
    # single source of truth for the camera contract (see this module's docstring).
    topic_parameters: Dict[str, str] = Field(default_factory=dict)
    # ``from -> to`` for a node whose subscriptions are hardcoded. Sorted by the generator
    # before it is written, because the same project must produce the same bytes.
    remappings: Dict[str, str] = Field(default_factory=dict)
    # vNext M3.5 (R-M3-16): the node serves the perception contract's on-demand service at EXACTLY
    # ``/perception/<detector>/detect`` (std_srvs/Trigger; the capture calls that name, and reads
    # ``/perception/<detector>/detections``), so a task program may CAPTURE with it. Declared, never
    # assumed: TSA cannot see what another package's node serves. Generates nothing.
    detect_service: bool = False

    def missing(self) -> Tuple[str, ...]:
        """The required fields that are empty, in declaration order.

        A block missing either one cannot be launched at all, and the failure it produces
        without this check is the worst kind: a bundle that starts, publishes nothing on
        ``/perception/<name>/detections``, and times out in ``DetectObject``.
        """
        out = []
        if not (self.package or '').strip():
            out.append('package')
        if not (self.executable or '').strip():
            out.append('executable')
        return tuple(out)

    def unknown_roles(self) -> Tuple[str, ...]:
        """``topic_parameters`` keys that are not one of :data:`INPUT_ROLES`.

        A typo here would drop a topic SILENTLY — the node would keep whatever its own
        parameter file said — so it is refused rather than ignored.
        """
        return tuple(sorted(k for k in self.topic_parameters if k not in INPUT_ROLES))

    def resolved_node_name(self, detector_name: str) -> str:
        """The node name to launch under: the explicit one, or ``detector_<name>``."""
        return (self.node_name or '').strip() or f'detector_{detector_name}'
