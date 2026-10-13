"""TSA's perception vocabulary as TSA 7.3.0 exposes it -- pinned BEFORE vNext M2.2 derives it.

ADR-0021, amendment M2.2: TSA's rule vocabulary, the terms a rule needs, what a method provides,
the faces and the version floors come from a committed snapshot of trainit_perception's export
instead of hand-kept tables and an import of the installed TP. Everything a project, a bundle and
the GUI could see is frozen here first, written and run on 7.3.0 WITH TP importable (the answers
TP gave then), and not edited by the implementation that follows -- as M2.1 pinned the planner
vocabulary (test_planner_vocabulary_pins.py). M2.2 may ADD vocabulary (two TP rules, the term
``edge_on``); it may not change or drop one of these answers, in any environment.

What is pinned:

- ``DetectionMethod`` exactly (it stays TSA code, ADR-0021 Decision 6); every ``CandidateRule`` of
  7.3.0 with its name, value and relative order; both as strings, through pickle and copy; the
  ``DetectorSpec`` defaults; typos and candidate product words refused at the model boundary;
- the default rule preference, what each method provides, each rule's terms;
- ``computable_terms`` / ``computable_rules`` / the pre-selected rule over a grid of methods and
  configurations, restricted to 7.3.0's terms and rules (``test/data``);
- what generation writes for each M2.2 case (``m2_2_cases.py``): the detector's params, in order,
  and the trainit_perception pin -- or the refusal, word for word (``test/data``);
- the GUI (offscreen): the method list, the first ten rule rows, which of them are enabled, the
  colour-mask form's defaults and ranges, and what an untouched form saves.

Regenerate the table only deliberately, never because this test failed::

    python3 test/test_perception_vocabulary_pins.py > test/data/perception_vocabulary_m2_2.txt
"""
import copy
import itertools
import json
import os
import pickle
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
if __name__ == '__main__':                                       # pragma: no cover
    sys.path[:0] = [HERE, os.path.join(HERE, os.pardir)]
    import conftest  # noqa: F401  (the same project rebasing as a test session)

import pytest  # noqa: E402
import yaml  # noqa: E402

import m2_2_cases  # noqa: E402
from trainit_setup_assistant.generator import Orchestrator  # noqa: E402
from trainit_setup_assistant.model import DetectorSpec  # noqa: E402
from trainit_setup_assistant.model import enums as E  # noqa: E402
from trainit_setup_assistant.model import perception as P  # noqa: E402

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

DATA = os.path.join(HERE, 'data', 'perception_vocabulary_m2_2.txt')

METHODS_7_3 = [('COLOR_MASK', 'color_mask'), ('CUSTOM', 'custom'), ('PCL_CLUSTER', 'pcl_cluster')]
RULES_7_3 = ['auto', 'detector_order', 'score', 'candidate_2d3d', 'candidate_x_visible',
             'candidate_x_nonoverlap', 'candidate_x_clearance', 'score_x_visible',
             'score_x_nonoverlap', 'score_x_clearance']
TERMS_7_3 = ['score', 'pose_quality', 'box_overlap', 'visible_fraction', 'clearance']
PREFERENCE_7_3 = ['candidate_x_visible', 'score_x_visible', 'candidate_2d3d',
                  'score_x_nonoverlap', 'score', 'detector_order']
RULE_TERMS_7_3 = {
    'auto': [], 'detector_order': [], 'score': ['score'],
    'candidate_2d3d': ['score', 'pose_quality'],
    'candidate_x_visible': ['score', 'pose_quality', 'visible_fraction'],
    'candidate_x_nonoverlap': ['score', 'pose_quality', 'box_overlap'],
    'candidate_x_clearance': ['score', 'pose_quality', 'clearance'],
    'score_x_visible': ['score', 'visible_fraction'],
    'score_x_nonoverlap': ['score', 'box_overlap'],
    'score_x_clearance': ['score', 'clearance'],
}


# --- the enums -------------------------------------------------------------------------------

def test_detection_method_is_exactly_7_3_0s():
    assert [(m.name, m.value) for m in E.DetectionMethod] == METHODS_7_3


def test_every_rule_of_7_3_0_keeps_its_name_value_and_relative_order():
    members = [(m.name, m.value) for m in E.CandidateRule]
    today = [(v.upper(), v) for v in RULES_7_3]
    missing = [t for t in today if t not in members]
    assert not missing, f'7.3.0 rules gone: {missing}'
    positions = [members.index(t) for t in today]
    assert positions == sorted(positions), members


@pytest.mark.parametrize('enum', [E.CandidateRule, E.DetectionMethod], ids=lambda e: e.__name__)
def test_the_tokens_are_strings_and_survive_pickle_and_copy(enum):
    for m in enum:
        assert isinstance(m, str) and str(m) == m.value and m == m.value
        assert enum(m.value) is m and enum[m.name] is m
        assert pickle.loads(pickle.dumps(m)) is m
        assert copy.copy(m) is m and copy.deepcopy(m) is m


def test_the_detector_spec_defaults():
    d = DetectorSpec(name='x')
    assert d.method is E.DetectionMethod.COLOR_MASK
    assert d.candidate_rule is None


@pytest.mark.parametrize('field, value', [
    ('candidate_rule', 'candidate_2d3D'), ('candidate_rule', 'best'),
    ('candidate_rule', 'foundationpose'), ('candidate_rule', 'segment_anything'),
    ('method', 'foundationpose'), ('method', 'rtdetr'), ('method', 'onnx_seg'),
    ('method', 'shape_mask'),
])
def test_a_typo_or_a_candidate_product_is_refused_at_the_model_boundary(field, value):
    with pytest.raises(Exception):
        DetectorSpec(name='x', **{field: value})


def test_the_default_rule_preference():
    assert [r.value for r in E.DEFAULT_RULE_PREFERENCE] == PREFERENCE_7_3
    assert E.DEFAULT_CANDIDATE_RULE.value == 'candidate_x_visible'


def test_what_each_method_provides():
    got = {k.value: (None if v is None else set(v)) for k, v in P.METHOD_PROVIDES.items()}
    assert got == {'color_mask': {'score', 'box_px', 'mask_area_px'}, 'custom': None,
                   'pcl_cluster': None}


def test_the_terms_of_every_rule_of_7_3_0():
    got = {r.value: list(t) for r, t in P.CANDIDATE_RULE_TERMS.items() if r.value in RULES_7_3}
    assert got == RULE_TERMS_7_3


# --- the table: computability over a grid, and what generation writes -------------------------

def _grid():
    faces = [('-', {}), ('-', {'expected_radius_m': 0.02})]
    for name, params in (('disc', ('expected_radius_m',)),
                         ('rect', ('expected_half_a_m', 'expected_half_b_m')),
                         ('cylinder_side', ('expected_radius_m', 'expected_height_m'))):
        for k in range(len(params) + 1):
            for subset in itertools.combinations(params, k):
                faces.append((name, {p: 0.02 for p in subset}))
    faces.append(('blob', {'expected_radius_m': 0.02}))
    tools = (('tool_radius_m', 0.012), ('tool_reach_mm', 4.0), ('tool_max_tilt_deg', 30.0))
    for method in [m for _, m in METHODS_7_3]:
        for face, sized in faces:
            for on in itertools.product((0, 1), repeat=len(tools)):
                params = dict(sized)
                if face != '-':
                    params['grasp_face'] = face
                params.update({n: v for (n, v), o in zip(tools, on) if o})
                yield (f'{method} | face={face} {",".join(sorted(sized)) or "-"} | '
                       f'tools={"".join(map(str, on))}'), E.DetectionMethod(method), params


def _generated(case):
    """(perception.yaml params of the detector, the trainit_perception exec_depend line), or the
    refusal -- generated into a temp dir that is removed."""
    project = m2_2_cases.CASES[case]()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            Orchestrator().generate(project, tmp)
        except ValueError as exc:
            return 'refused ' + json.dumps(str(exc))
        root = Path(tmp) / 'cell_app'
        det = yaml.safe_load((root / 'config' / 'perception.yaml').read_text())['detectors']
        pin = [ln.strip() for ln in (root / 'package.xml').read_text().splitlines()
               if 'trainit_perception' in ln and 'exec_depend' in ln]
        return (f'ok pin={json.dumps(pin)} '
                f'params={json.dumps(list(det[m2_2_cases.DETECTOR]["params"].items()))}')


def table():
    out = ['[computable]']
    for key, method, params in _grid():
        terms = P.computable_terms(method, params)
        rules = [r.value for r in P.computable_rules(method, params) if r.value in RULES_7_3]
        out.append(f'{key} | terms={",".join(t for t in TERMS_7_3 if t in terms)} | '
                   f'rules={",".join(rules)} | '
                   f'preferred={P.preferred_candidate_rule(method, params).value}')
    out.append('[generated]')
    out += [f'{case} | {_generated(case)}' for case in m2_2_cases.CASES]
    return out


def test_tsa_answers_exactly_what_7_3_0_answered():
    with open(DATA) as fh:
        want = fh.read().splitlines()
    got = table()
    if got != want:
        changed = [f'  pinned: {w}\n  now:    {g}' for w, g in zip(want, got) if w != g][:20]
        extra = len(got) - len(want)
        raise AssertionError(
            'TSA\'s perception answers moved (vNext M2.2 may add vocabulary, not change it):\n'
            + '\n'.join(changed) + (f'\n  ... and {extra:+d} lines' if extra else ''))


# --- the GUI (offscreen) -----------------------------------------------------------------------

@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def _perception_page(project):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    wiz = SetupWizard(AssistantController(project))
    page = wiz.perception_page
    page.initializePage()
    return wiz, page                      # the wizard is held: a collected one deletes its widgets


def test_the_wizard_lists_7_3_0s_methods_and_rules(qapp):
    from trainit_setup_assistant.gui.wizard import PerceptionPage
    assert [tuple(m) for m in PerceptionPage.METHODS] == [
        ('color_mask', 'Colour mask (HSV)', True), ('shape_mask', 'Shape mask (contours)', False),
        ('pcl_cluster', 'PCL 3D cluster (C++ node)', False),
        ('custom', 'Custom node (expert)', False)]
    assert [tok for tok, _ in PerceptionPage.RULES[:10]] == [   # the GUI's own order
        'auto', 'candidate_x_visible', 'score_x_visible', 'candidate_2d3d',
        'candidate_x_nonoverlap', 'score_x_nonoverlap', 'candidate_x_clearance',
        'score_x_clearance', 'score', 'detector_order']
    assert [label for _, label in PerceptionPage.RULES[:10]] == [
        'auto  (the best rule this detector can compute)',
        'candidate x visible face  (measured best, needs a 3D fit)',
        'score x visible face  (the same idea without a 3D fit)',
        'candidate only  (score x pose quality)',
        'candidate x non-overlap  (boxes)',
        'score x non-overlap  (boxes only — any detector)',
        'candidate x tool clearance  (needs the tool numbers)',
        'score x tool clearance',
        'score alone',
        "detector order  (an EXPLICIT 'no ranking')"]


@pytest.mark.parametrize('case, enabled', [
    ('plain', {'auto', 'score_x_visible', 'score_x_nonoverlap', 'score', 'detector_order'}),
    ('tool_numbers', {'auto', 'score_x_visible', 'score_x_nonoverlap', 'score', 'detector_order'}),
    ('rule_rect_measured', {'auto', 'score_x_visible', 'score_x_nonoverlap', 'score',
                            'detector_order'}),
])
def test_the_rule_combo_enables_what_7_3_0_enabled(qapp, case, enabled):
    wiz, page = _perception_page(m2_2_cases.CASES[case]())
    model = page.p_rule.model()
    on = {tok for i, (tok, _) in enumerate(page.RULES) if model.item(i + 1).isEnabled()}
    assert {t for t in on if t in RULES_7_3} == enabled


def test_the_colour_mask_form_has_7_3_0s_defaults_and_ranges(qapp):
    wiz, page = _perception_page(m2_2_cases.CASES['plain']())
    got = {name: (getattr(page, name).minimum(), getattr(page, name).maximum(),
                  getattr(page, name).value())
           for name in ('h_lo', 'h_hi', 's_lo', 's_hi', 'v_lo', 'v_hi', 'p_area', 'p_max',
                        'p_morph', 'p_win', 'p_blur', 'p_zmin', 'p_zmax')}
    assert got == {'h_lo': (0, 179, 170), 'h_hi': (0, 179, 10), 's_lo': (0, 255, 40),
                   's_hi': (0, 255, 255), 'v_lo': (0, 255, 60), 'v_hi': (0, 255, 255),
                   'p_area': (1, 100000, 60), 'p_max': (1, 20, 1), 'p_morph': (0, 15, 3),
                   'p_win': (1, 15, 5), 'p_blur': (0, 31, 0), 'p_zmin': (0.0, 10.0, 0.0),
                   'p_zmax': (0.0, 10.0, 0.0)}
    assert page._params_from_form() == {
        'class_id': 'object', 'h': [170, 10], 's': [40, 255], 'v': [60, 255],
        'min_area_px': 60, 'max_objects': 1, 'morph_kernel': 3, 'depth_window_px': 5}


if __name__ == '__main__':                                       # pragma: no cover
    sys.stdout.write('\n'.join(table()) + '\n')
