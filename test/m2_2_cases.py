"""The projects the vNext M2.2 tests build (ADR-0021, amendment M2.2).

The example project with one detector, configured each way trainit_perception's vocabulary reaches
a bundle: a grasp face measured on the grasp target (disc, rect, cylinder_side -- and the same
cylinder with its numbers given in the other order, because the ORDER the face declares its
parameters in reaches perception.yaml), a face through the detector's own params, the noise
filters, the tool's numbers, an external-node method, and the configurations generation refuses.

Shared by test_perception_vocabulary_pins.py, test_m2_2_regressions.py and their subprocess
workers (which import it from this directory). Building a case never generates anything.
"""
import os

from trainit_setup_assistant.gui.controller import AssistantController
from trainit_setup_assistant.model import (BundleSpec, DetectorSpec, PerceptionSpec, SceneObject,
                                           load_project)

HERE = os.path.dirname(os.path.abspath(__file__))
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')
DETECTOR = 'capsules'
DISC_R = 0.02095          # the reference capsule's measured cap radius (ADR-0018), metres
TILT = 30.0               # tool_max_tilt_deg: any positive value switches the edge_on term on
TOOL = (0.012, 4.0)       # the tool's footprint radius (m) and grip window (mm)
RECT = {'expected_half_a_m': 0.03, 'expected_half_b_m': 0.02}
CYLINDER = {'expected_radius_m': 0.0325, 'expected_height_m': 0.06}


def project(*, method='color_mask', rule=None, params=None, measured=None, tool=None,
            target=True):
    """The example project, one detector; a cylinder grasp target unless ``target`` is False.

    ``measured`` = (face, numbers) goes through the GUI's own path for a measured face
    (``AssistantController.set_grasp_face``), and ``tool`` through the End-effector dialog's
    (``set_tool_geometry``), so the cases exercise what a person actually produces."""
    p = load_project(EXAMPLE)
    p.bundle = BundleSpec.from_prefix('cell')
    p.perception = p.perception or PerceptionSpec()
    p.perception.detectors = [DetectorSpec(name=DETECTOR, method=method,
                                           params=dict(params or {}), candidate_rule=rule)]
    if target:
        p.scene.objects = [o for o in p.scene.objects if not o.is_grasp_target()] + [
            SceneObject(id='capsule', shape='cylinder', dims=[0.0135, 0.04],
                        category='dynamic', grasp_target=True)]
    ctrl = AssistantController(p)
    if measured is not None:
        ctrl.set_grasp_face(measured[0], dict(measured[1]), 'vNext M2.2 test')
    if tool is not None:
        ctrl.set_tool_geometry(tool_radius_m=tool[0], tool_reach_mm=tool[1])
    return p


CASES = {
    'plain': lambda: project(),
    'tilt': lambda: project(params={'tool_max_tilt_deg': TILT}),
    'rule_bbox_disc': lambda: project(rule='score_x_visible'),
    'rule_auto_bbox_disc': lambda: project(rule='auto'),
    'rule_disc_measured': lambda: project(
        rule='score_x_visible', measured=('disc', {'expected_radius_m': DISC_R})),
    'rule_rect_measured': lambda: project(rule='score_x_visible', measured=('rect', RECT)),
    'rule_rect_measured_extra_key': lambda: project(
        rule='score_x_visible', measured=('rect', dict(RECT, expected_radius_m=0.05))),
    'rule_cylinder_measured': lambda: project(
        rule='score_x_visible', measured=('cylinder_side', CYLINDER)),
    'rule_cylinder_measured_reversed': lambda: project(
        rule='score_x_visible',
        measured=('cylinder_side', dict(reversed(list(CYLINDER.items()))))),
    'cylinder_measured_no_rule': lambda: project(measured=('cylinder_side', CYLINDER)),
    'rule_cylinder_params': lambda: project(
        rule='score_x_visible', params=dict(CYLINDER, grasp_face='cylinder_side')),
    'rule_disc_tilt': lambda: project(rule='score_x_visible', params={'tool_max_tilt_deg': TILT},
                                      measured=('disc', {'expected_radius_m': DISC_R})),
    'noise_all': lambda: project(params={'blur_px': 3, 'depth_min_m': 0.2, 'depth_max_m': 1.5}),
    'noise_depth_min': lambda: project(params={'depth_min_m': 0.25}),
    'noise_and_rule': lambda: project(rule='score_x_visible', params={'blur_px': 3},
                                      measured=('disc', {'expected_radius_m': DISC_R})),
    'tool_numbers': lambda: project(tool=TOOL),
    'tool_numbers_tuned_override': lambda: project(tool=TOOL, params={'tool_radius_m': 0.02}),
    'external_rule': lambda: project(method='custom', rule='candidate_2d3d'),
    'external_clearance_tilt': lambda: project(method='custom', rule='score_x_clearance',
                                               params={'tool_max_tilt_deg': TILT}, tool=TOOL),
    'refused_unknown_face_no_rule': lambda: project(params={'grasp_face': 'blob'}),
    'refused_unknown_face_rule': lambda: project(
        rule='score_x_visible', params={'grasp_face': 'blob', 'expected_radius_m': DISC_R}),
    'refused_unknown_face_rule_score': lambda: project(rule='score',
                                                       params={'grasp_face': 'blob'}),
    'refused_measured_unknown_face_rule': lambda: project(
        rule='score_x_visible', measured=('blob', {'expected_radius_m': DISC_R})),
    'refused_rect_half_missing_rule': lambda: project(
        rule='score_x_visible', measured=('rect', {'expected_half_a_m': 0.03})),
    'refused_colour_mask_pose_rule': lambda: project(rule='candidate_x_visible'),
    'refused_no_target_rule': lambda: project(rule='score_x_visible', target=False),
}
