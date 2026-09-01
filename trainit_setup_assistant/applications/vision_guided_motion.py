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
        rel = self._relative_waypoints(project)
        names = {wp.name for wp in project.application.waypoints}

        # --- step-relative bindings (D-017) ---
        for wp in rel:
            r = wp.relative
            if wp.vision is not None:
                problems.append(f'waypoint "{wp.name}" has BOTH a vision and a '
                                'relative binding — pick one')
            if r.step not in names:
                problems.append(f'waypoint "{wp.name}" is relative to unknown '
                                f'step "{r.step}"')
                continue
            if r.step == wp.name:
                problems.append(f'waypoint "{wp.name}" cannot be relative to itself')
                continue
            ref = project.application.waypoint_by_name(r.step)
            if ref.relative is not None:
                problems.append(f'waypoint "{wp.name}": reference "{r.step}" is '
                                'itself relative — chains are not supported yet')
            elif ref.vision is None and not (ref.position and ref.orientation):
                problems.append(f'waypoint "{wp.name}": reference "{r.step}" has no '
                                'runtime pose (it must be a tcp waypoint with an '
                                'orientation, or camera-guided)')

        if per is None or not per.detectors:
            if not rel:
                problems.append(f'{self.app_type}: no perception block '
                                '(configure a detector in the Perception step)')
            return problems
        if not bound and not rel:
            problems.append(f'{self.app_type}: no waypoint has a vision binding '
                            '(the application would be blind)')
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
        bound_detectors = {wp.vision.detector for wp in bound}
        for d in per.detectors:
            if not d.emits_node:
                problems.append(f'note: detector "{d.name}" ({d.method.value}) is an '
                                'external-node method — its node is not generated; '
                                'launch it yourself so it publishes the contract')
            if d.name in bound_detectors and not d.continuous:
                problems.append(
                    f'detector "{d.name}" is on-demand (continuous: false) but the '
                    'generated tree only consumes the continuous stream — every '
                    'DetectObject would time out. Enable continuous, or add your own '
                    'Trigger caller.')
        from ..model.enums import DetectionMethod, MotionType, PlannerId
        for wp in bound:
            d = per.detector_by_name(wp.vision.detector)
            if (d is not None and wp.vision.orientation == 'detected'
                    and d.method is DetectionMethod.COLOR_MASK):
                problems.append(
                    f'waypoint "{wp.name}": orientation "detected" with a colour-mask '
                    'detector — a colour mask publishes an identity orientation, which '
                    'after TF becomes the camera-to-base rotation: an arbitrary TCP '
                    'orientation. Use "keep" or "from:<waypoint>".')
            # D-016 (found live): Pilz PTP is a blind joint interpolation — it swept
            # the wrist through the camera on the way to a vision-computed approach.
            # A vision goal lands anywhere, so the travel INTO it must be
            # collision-aware.
            seg = project.application.segment_for(wp.name)
            if seg is not None and seg.motion is MotionType.PTP:
                planner = seg.planner or project.application.global_planner_mode
                if planner is PlannerId.PILZ:
                    problems.append(
                        f'waypoint "{wp.name}" is vision-driven but reached with '
                        'ptp/pilz — a blind joint interpolation that cannot avoid '
                        'obstacles (it hit the camera on the reference cell). Use '
                        'motion "free" with OMPL for the travel into a vision goal.')
        return problems

    # --- tree hooks -------------------------------------------------------------
    def _cycle_prologue(self, project: CanonicalProject, indent: str) -> List[str]:
        per = project.perception
        bound = self._vision_waypoints(project)
        lines: List[str] = []
        if per is None or not bound:
            return lines + self._relative_lines(project, indent)
        lines.append(
            f'{indent}<!-- Vision (D-014): these waypoints are overwritten from the '
            'camera each cycle; bt_params keeps the captured poses as fallback. -->')
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
                # dx/dy only when set, so the golden (dz-only) stays byte-stable
                offsets = ''.join(f'{k}="{v:.3f}" ' for k, v in
                                  (('dx', b.dx), ('dy', b.dy))
                                  if round(v, 3) != 0.0)
                lines.append(
                    f'{indent}<SetWaypointFromDetection waypoint="{wp.name}" '
                    f'from="{out_key}" {offsets}dz="{b.dz:.3f}" '
                    f'orientation="{b.orientation}"/>')
        return lines + self._relative_lines(project, indent)

    def _relative_lines(self, project: CanonicalProject, indent: str) -> List[str]:
        """Step-relative waypoints (D-017), emitted AFTER the vision lines so the
        reference pose is already final when SetWaypointRelative reads it."""
        rel = self._relative_waypoints(project)
        if not rel:
            return []
        lines = [f'{indent}<!-- Relative steps (D-017): pose derived from another '
                 "step's FINAL pose (vision included), at run time. -->"]
        for wp in rel:
            r = wp.relative
            offs = ''.join(f'{k}="{v:.3f}" ' for k, v in
                           (('dx', r.dx), ('dy', r.dy)) if round(v, 3) != 0.0)
            angs = ''.join(f'{k}="{v:.1f}" ' for k, v in
                           (('droll', r.droll), ('dpitch', r.dpitch),
                            ('dyaw', r.dyaw)) if round(v, 1) != 0.0)
            lines.append(
                f'{indent}<SetWaypointRelative waypoint="{wp.name}" '
                f'from="{r.step}" {offs}dz="{r.dz:.3f}" {angs}'.rstrip() + '/>')
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
    def _relative_waypoints(self, project: CanonicalProject) -> List[Waypoint]:
        """Relative-bound waypoints in SEQUENCE order (first occurrence)."""
        app = project.application
        seq = app.sequence or self._derived_sequence(project)
        seen, out = set(), []
        for name in seq:
            if name in seen:
                continue
            seen.add(name)
            wp = app.waypoint_by_name(name)
            if wp is not None and wp.relative is not None:
                out.append(wp)
        return out

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
