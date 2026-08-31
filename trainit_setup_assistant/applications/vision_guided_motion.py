"""vision_guided_motion application template (TSA v4, D-014/D-015).

The blind pick-and-place skeleton plus vision: waypoints carrying a
``Waypoint.vision`` binding are overwritten AT RUN TIME from the camera. The
template injects, at the top of every cycle, one ``DetectObject`` per detector in
use and one ``SetWaypointFromDetection`` per bound waypoint — and, whenever a
``ResetScene`` precedes the next detection in the loop, the settle barrier
(``<Sleep>``): arrival freshness is not content freshness (the reset returns before
the camera pipeline has re-rendered the world; measured 0.4 s on the cell).

Everything else — sequence, motion per segment, planners, tool actions, loop —
is inherited unchanged from :class:`PickAndPlace`, so a project with no vision
bindings emits exactly the blind tree.
"""

from __future__ import annotations

from typing import Dict, List

from ..model import CanonicalProject, Waypoint
from ..model.enums import AppType
from .pick_and_place import PickAndPlace


class VisionGuidedMotion(PickAndPlace):
    app_type = AppType.VISION_GUIDED_MOTION.value

    # --- validation -------------------------------------------------------------
    def validate(self, project: CanonicalProject) -> List[str]:
        problems = self._validate_sequence(project)
        problems += self._validate_payload_refs(project)
        # NO grasp/release requirement: a vision-guided motion may carry no gripper.
        per = project.perception
        bound = self._vision_waypoints(project)
        if per is None or not per.detectors:
            problems.append(f'{self.app_type}: no perception block '
                            '(configure a detector in the Perception step)')
            return problems
        if not bound:
            problems.append(f'{self.app_type}: no waypoint has a vision binding '
                            '(the application would be blind)')
        names = {wp.name for wp in project.application.waypoints}
        for wp in bound:
            b = wp.vision
            if per.detector_by_name(b.detector) is None:
                problems.append(f'waypoint "{wp.name}" binds unknown detector '
                                f'"{b.detector}"')
            if b.orientation.startswith('from:'):
                ref = b.orientation.split(':', 1)[1]
                if ref not in names:
                    problems.append(f'waypoint "{wp.name}" orientation references '
                                    f'unknown waypoint "{ref}"')
            elif b.orientation == 'keep' and not wp.orientation:
                # holds for joint waypoints too: the binding promotes them to tcp,
                # and a tcp goal with no orientation fails at run time (that is why
                # the golden's joint pre_pick borrows "from:pick").
                problems.append(f'waypoint "{wp.name}": orientation "keep" but the '
                                'waypoint has none — the runtime would fail')
        for d in per.detectors:
            if not d.emits_node:
                problems.append(f'note: detector "{d.name}" ({d.method.value}) is an '
                                'external-node method — its node is not generated; '
                                'launch it yourself so it publishes the contract')
        return problems

    # --- tree hooks -------------------------------------------------------------
    def _cycle_prologue(self, project: CanonicalProject, indent: str) -> List[str]:
        per = project.perception
        bound = self._vision_waypoints(project)
        if per is None or not bound:
            return []
        lines: List[str] = [
            f'{indent}<!-- Vision (D-014): these waypoints are overwritten from the '
            'camera each cycle; bt_params keeps the captured poses as fallback. -->'
        ]
        for det_name, waypoints in self._by_detector(bound).items():
            d = per.detector_by_name(det_name)
            if d is None:
                continue
            out_key = f'detected.{d.name}'
            lines.append(
                f'{indent}<DetectObject detector="{d.name}" class_id="{d.class_id}" '
                f'target_frame="{project.robot.base_frame}" '
                f'timeout_ms="{per.detect_timeout_ms}" out_key="{out_key}"/>')
            for wp in waypoints:
                b = wp.vision
                lines.append(
                    f'{indent}<SetWaypointFromDetection waypoint="{wp.name}" '
                    f'from="{out_key}" dz="{b.dz:.3f}" '
                    f'orientation="{b.orientation}"/>')
        return lines

    def _after_scene_reset(self, project: CanonicalProject, indent: str) -> List[str]:
        per = project.perception
        if per is None or per.settle_ms <= 0 or not self._vision_waypoints(project):
            return []
        return [
            f'{indent}<!-- settle: the reset returns before the camera pipeline has '
            're-rendered the world — without this pause the next DetectObject '
            'latches a stale pose. -->',
            f'{indent}<Sleep msec="{per.settle_ms}"/>',
        ]

    # --- helpers ----------------------------------------------------------------
    def _vision_waypoints(self, project: CanonicalProject) -> List[Waypoint]:
        """Vision-bound waypoints in SEQUENCE order (first occurrence)."""
        app = project.application
        seq = app.sequence or self._derived_sequence(project)
        seen, out = set(), []
        for name in seq:
            if name in seen:
                continue
            seen.add(name)
            wp = app.waypoint_by_name(name)
            if wp is not None and wp.vision is not None:
                out.append(wp)
        return out

    @staticmethod
    def _by_detector(bound: List[Waypoint]) -> Dict[str, List[Waypoint]]:
        """Group bound waypoints by detector, in first-use order."""
        groups: Dict[str, List[Waypoint]] = {}
        for wp in bound:
            groups.setdefault(wp.vision.detector, []).append(wp)
        return groups
