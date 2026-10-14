"""The 43 projects of the generation identity proof (vNext M2.2, kept as a regression test in M2.3).

The vNext M2.2 identity proof generated these 43 projects in five environments -- trainit_perception
importable (twice), made unimportable, off the interpreter's path, uninstalled -- and found every
bundle byte-identical across them, and identical to TSA 7.3.0's. The owner asked for that proof to
stay a regression test: ``test_generation_identity.py`` runs these builders (opt-in) and compares the
environments with each other; the historical comparison with 7.3.0 stays with the M2.2 evidence.

The builders are the M2.2 harness's, without its machine paths: the golden project (40 cases: its
colour-mask detector ``red_object``, faces measured on its grasp target, every candidate rule TSA
offers, the noise filters, the gripper's tool numbers, extra detectors and external nodes), the
example project (2) and the owner's policy project (1, untracked: the case skips without it).
Projects are loaded through ``model.load_project`` AFTER ``conftest`` rebased it, so a checkout
elsewhere runs the same cases. ``missing(case)`` says why a case cannot run on this checkout, and
``require(case)`` is how a test that needs the case stops there (vNext G0 P1). Not collected by
pytest (the name is not test_*.py).
"""
from pathlib import Path

import yaml

import ws_paths

TEST = Path(__file__).resolve().parent
PKG = TEST.parent

GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
EXAMPLE = PKG / 'examples' / 'fr3wml_project.yaml'
POLICY_PROJECT = ws_paths.ws_src('fr3wml_digital_twin', 'fr3wml_suction_policy_test4_bundle',
                                 'project.yaml')

GOLDEN_NAMES = dict(description_package='fr3wml_suction_camera_tsa_4_1_description',
                    moveit_config_package='fr3wml_suction_camera_tsa_4_1_trainit_config',
                    app_package='fr3wml_suction_camera_tsa_4_1_app')
DISC_R = 0.02095
TILT = 30.0
# Every candidate rule TSA offered when the proof was made (7.3.0), on the golden colour mask.
RULES = ('auto', 'candidate_x_visible', 'score_x_visible', 'candidate_2d3d',
         'candidate_x_nonoverlap', 'score_x_nonoverlap', 'candidate_x_clearance',
         'score_x_clearance', 'score', 'detector_order')


def _model():
    from trainit_setup_assistant import model     # after conftest: its load_project rebases
    return model


def golden():
    m = _model()
    p = m.load_project(str(GOLDEN_PROJECT))
    p.bundle = m.BundleSpec(**GOLDEN_NAMES)
    return p


_UNSET = object()


def _dump(obj):
    return obj.model_dump() if hasattr(obj, 'model_dump') else obj.dict()


def with_detector(p, idx=0, *, candidate_rule=_UNSET, params_update=None):
    det = p.perception.detectors[idx]
    data = _dump(det)
    if params_update:
        data['params'] = {**data['params'], **params_update}
    if candidate_rule is not _UNSET:
        data['candidate_rule'] = candidate_rule
    p.perception.detectors[idx] = _model().DetectorSpec(**data)
    return p


def measured(p, face, params, source='identity harness'):
    """The GUI's own path for a MEASURED face (AssistantController.set_grasp_face), Qt-free."""
    from trainit_setup_assistant.gui.controller import AssistantController
    AssistantController(p).set_grasp_face(face, params, source)
    return p


def tool(p, radius_m=0.012, reach_mm=4.0):
    from trainit_setup_assistant.gui.controller import AssistantController
    AssistantController(p).set_tool_geometry(tool_radius_m=radius_m, tool_reach_mm=reach_mm)
    return p


def seg_project():
    import test_external_node_detector as ten
    return ten._seg_project()


def seg_detector(name='seg', **kw):
    import test_external_node_detector as ten
    m = _model()
    return m.DetectorSpec(name=name, method='custom', **ten.SEG_TOPICS,
                          external_node=m.ExternalNodeSpec(**dict(ten.SEG_NODE,
                                                                  node_name=f'{name}_node')),
                          **kw)


def golden_plus(*dets):
    p = golden()
    p.perception.detectors = list(p.perception.detectors) + list(dets)
    return p


def detector(**kw):
    return _model().DetectorSpec(**kw)


CASES = {
    'p1_golden': golden,
    'p2_example': lambda: _model().load_project(str(EXAMPLE)),
    'p3_golden_rule_score_x_visible_disc': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'grasp_face': 'disc', 'expected_radius_m': DISC_R}),
    'p4_golden_tool_max_tilt_deg': lambda: with_detector(
        golden(), params_update={'tool_max_tilt_deg': TILT}),
    'p5_external_node_seg': seg_project,
    's1_golden_rule_disc_plus_tilt': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'grasp_face': 'disc', 'expected_radius_m': DISC_R,
                       'tool_max_tilt_deg': TILT}),
    's2_golden_rule_rect_face': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'grasp_face': 'rect', 'expected_half_a_m': 0.03,
                       'expected_half_b_m': 0.02}),
    's3_golden_unknown_face_no_rule': lambda: with_detector(
        golden(), params_update={'grasp_face': 'blob'}),
    's4_golden_rule_rect_face_half_b_missing': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'grasp_face': 'rect', 'expected_half_a_m': 0.03}),
    # faces MEASURED on the grasp target: the face parameters' order reaches the bytes
    'm1_measured_disc_rule': lambda: with_detector(
        measured(golden(), 'disc', {'expected_radius_m': DISC_R}),
        candidate_rule='score_x_visible'),
    'm2_measured_rect_rule': lambda: with_detector(
        measured(golden(), 'rect', {'expected_half_a_m': 0.03, 'expected_half_b_m': 0.02}),
        candidate_rule='score_x_visible'),
    'm3_measured_cylinder_side_rule': lambda: with_detector(
        measured(golden(), 'cylinder_side', {'expected_radius_m': 0.0325,
                                             'expected_height_m': 0.06}),
        candidate_rule='score_x_visible'),
    'm4_measured_cylinder_side_reversed_rule': lambda: with_detector(
        measured(golden(), 'cylinder_side', {'expected_height_m': 0.06,
                                             'expected_radius_m': 0.0325}),
        candidate_rule='score_x_visible'),
    'm5_measured_cylinder_side_no_rule': lambda: measured(
        golden(), 'cylinder_side', {'expected_radius_m': 0.0325, 'expected_height_m': 0.06}),
    'm6_measured_disc_rule_candidate_x_visible': lambda: with_detector(
        measured(golden(), 'disc', {'expected_radius_m': DISC_R}),
        candidate_rule='candidate_x_visible'),
    'm7_measured_unknown_face_rule': lambda: with_detector(
        measured(golden(), 'blob', {'expected_radius_m': DISC_R}),
        candidate_rule='score_x_visible'),
    'm8_measured_rect_extra_key_rule': lambda: with_detector(
        measured(golden(), 'rect', {'expected_half_a_m': 0.03, 'expected_half_b_m': 0.02,
                                    'expected_radius_m': 0.05}),
        candidate_rule='score_x_visible'),
    # a face through the detector's params
    'd1_params_cylinder_side_rule': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'grasp_face': 'cylinder_side', 'expected_radius_m': 0.0325,
                       'expected_height_m': 0.06}),
    'd2_bbox_default_face_rule_candidate_x_visible': lambda: with_detector(
        golden(), candidate_rule='candidate_x_visible'),
    # an unknown face WITH a rule
    'u1_unknown_face_params_rule_score_x_visible': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'grasp_face': 'blob', 'expected_radius_m': DISC_R}),
    'u2_unknown_face_params_rule_score': lambda: with_detector(
        golden(), candidate_rule='score', params_update={'grasp_face': 'blob'}),
    # the noise filters
    'n1_noise_filters_all': lambda: with_detector(
        golden(), params_update={'blur_px': 3, 'depth_min_m': 0.2, 'depth_max_m': 1.5}),
    'n2_noise_depth_min_only': lambda: with_detector(
        golden(), params_update={'depth_min_m': 0.25}),
    'n3_noise_plus_rule': lambda: with_detector(
        golden(), candidate_rule='score_x_visible',
        params_update={'blur_px': 3, 'grasp_face': 'disc', 'expected_radius_m': DISC_R}),
    # the gripper's tool numbers (GripperSpec -> clearance params)
    'g1_gripper_tool_numbers': lambda: tool(golden()),
    'g2_gripper_tool_numbers_rule_score_x_clearance': lambda: with_detector(
        tool(golden()), candidate_rule='score_x_clearance'),
    'g3_gripper_tool_numbers_tuned_override': lambda: with_detector(
        tool(golden()), params_update={'tool_radius_m': 0.02}),
    # several detectors, external nodes, the other external method
    'x1_colour_mask_plus_external_rule': lambda: tool(golden_plus(
        seg_detector('seg', candidate_rule='candidate_x_visible'))),
    'x2_two_colour_masks': lambda: with_detector(golden_plus(detector(
        name='blue_object', method='color_mask', candidate_rule='score_x_nonoverlap',
        params={'class_id': 'blue', 'h': [100, 130], 'depth_max_m': 1.2})),
        candidate_rule='score'),
    'x3_external_rule_clearance_tool': lambda: tool(golden_plus(
        seg_detector('seg', candidate_rule='score_x_clearance'))),
    'x4_pcl_cluster_no_launch_block': lambda: golden_plus(detector(
        name='cluster', method='pcl_cluster')),
    'x5_custom_no_launch_block_rule': lambda: golden_plus(detector(
        name='mine', method='custom', candidate_rule='candidate_2d3d')),
    # the owner's policy project (untracked; read, never written: it generates into a temp dir)
    'pol1_policy_test4_project': lambda: _model().load_project(str(POLICY_PROJECT)),
}
for _rule in RULES:
    CASES[f'r_{_rule}'] = (lambda r: lambda: with_detector(
        measured(golden(), 'disc', {'expected_radius_m': DISC_R}), candidate_rule=r))(_rule)

# What each case did in the M2.2 proof (and still does): a builder that silently stops reaching
# its branch -- a refusal that starts generating, or the reverse -- fails, even though every
# environment would still agree with every other.
REFUSED = {'s3_golden_unknown_face_no_rule', 's4_golden_rule_rect_face_half_b_missing',
           'm6_measured_disc_rule_candidate_x_visible', 'm7_measured_unknown_face_rule',
           'd2_bbox_default_face_rule_candidate_x_visible',
           'u1_unknown_face_params_rule_score_x_visible', 'u2_unknown_face_params_rule_score',
           'g2_gripper_tool_numbers_rule_score_x_clearance', 'r_candidate_x_visible',
           'r_candidate_2d3d', 'r_candidate_x_nonoverlap', 'r_candidate_x_clearance',
           'r_score_x_clearance'}

SOURCE = {name: ('example' if name in ('p2_example', 'p5_external_node_seg')
                 else 'policy' if name == 'pol1_policy_test4_project' else 'golden')
          for name in CASES}


def _inputs(project_file: Path):
    """The input directories and files a project records, rebased the way conftest rebases them."""
    data = yaml.safe_load(project_file.read_text(encoding='utf-8')) or {}
    robot = data.get('robot') or {}
    desc = robot.get('description') or {}
    out = [ws_paths.in_workspace(desc.get('urdf_dir') or '')]
    if desc.get('meshes_dir'):
        out.append(ws_paths.in_workspace(desc['meshes_dir']))
    if robot.get('base_moveit_config_path'):
        out.append(ws_paths.in_workspace(robot['base_moveit_config_path']))
    for policy in ((data.get('application') or {}).get('policies') or []):
        card = Path(ws_paths.in_workspace(policy.get('card') or ''))
        out.append(str(card))
        if card.is_file():                        # the product reads policy_card.files
            doc = yaml.safe_load(card.read_text(encoding='utf-8')) or {}
            files = (doc.get('policy_card') or {}).get('files') or {}
            out += [str(card.parent / rel) for rel in files.values() if isinstance(rel, str)]
    return [p for p in out if p]


def missing(case: str) -> str:
    """Why a case cannot run on this checkout ('' when it can): the first input that is absent."""
    project_file = {'golden': GOLDEN_PROJECT, 'example': EXAMPLE,
                    'policy': POLICY_PROJECT}[SOURCE[case]]
    if not project_file.is_file():
        return f'{project_file} is not here'
    for path in _inputs(project_file):
        if not Path(path).exists():
            return f'{path} (recorded by {project_file.name}) is not here'
    return ''


def require(case: str) -> None:
    """Stop the running test when ``case`` cannot run on this checkout: the way a test says that a
    reference project, or an input it records, is not here (vNext G0 P1; the older
    ``pytest.skip(missing(case))`` sites only skip). It SKIPS naming the first absent input
    (``missing()``'s message, the one the suite's other skips print); with
    TRAINIT_REQUIRE_GOLDEN=1 it FAILS instead (``eef_fixture.skip_or_fail``), so a job that must
    run the test cannot report it green unrun. A no-op where everything is present."""
    why = missing(case)
    if why:
        import eef_fixture        # pytest and TSA's model: imported only by a test that stops
        eef_fixture.skip_or_fail(why)
