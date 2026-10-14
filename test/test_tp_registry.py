"""TSA reads trainit_perception's vocabulary from a committed snapshot, never from TP (vNext M2.2).

ADR-0021, amendment M2.2: TP registers its configuration vocabulary once (its built-in tables,
read by ``trainit_perception.vocabulary.export()``); TSA commits the canonical YAML of it
(``catalog/tp_registry_snapshot.yaml``, written by ``python3 -m
trainit_setup_assistant.catalog.refresh``) and builds from the snapshot alone. These tests hold:

1. the snapshot to TP's live export, entry by entry, and a drift says WHAT is out of sync -- with
   committed MUTATION tests for every kind of drift the owner named (a rule added or removed, a
   term renamed, a requirement, a default, an allowed value, a face, a floor, the schema version,
   a classification);
2. the snapshot itself: canonical bytes, the sha256 the catalog records, the closed schema and the
   boundary (ADR-0021, Decision 9: names, kinds, schema, requirements, floors -- no prose, no
   formula, no gate value), and no candidate provider's name anywhere in it or in what TSA lets a
   project select;
3. what TSA's code names in TP's vocabulary is required (an AST scan of the product code), and
   ``DetectionMethod`` -- TSA code -- agrees with the snapshot both ways;
4. TSA's computability model equals TP's ``computable_terms`` over every configuration, and the
   GUI's hand-written colour-mask form equals the snapshot's schema;
5. the refresh: READ, VALIDATE, COMPARE, WRITE -- refusing (exit 2, nothing written: bytes, mtime
   and mode) an unclassified entry, a removal, a schema it cannot read, a token TSA names going
   away, a release mismatch; writing all three files or none; leaving no global state behind.
"""
import ast
import copy
import hashlib
import itertools
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import pytest
import yaml

import catalog_identity
import identity_cases
from trainit_setup_assistant import catalog
from trainit_setup_assistant.catalog import (REFRESH, TP_REGISTRY_FILE, TP_TIERS,
                                             TSA_COLOR_MASK_FORM, TSA_OWN_METHODS,
                                             TSA_REQUIRED_TP_RULES, TSA_TP_METHODS,
                                             parse_tp_registry, tp_drift, tp_registry,
                                             tp_registry_problems, tp_registry_text,
                                             tp_tier_problems, tp_vocabulary_problems)
from trainit_setup_assistant.catalog import refresh as refresh_tool
from trainit_setup_assistant.model import enums as E
from trainit_setup_assistant.model import perception as P

PKG = Path(__file__).resolve().parents[1]
CATALOG_DIR = PKG / 'trainit_setup_assistant' / 'catalog'
TP_EXPORT = refresh_tool.tp_export          # captured before any fixture stands in for it
TP_RELEASE = refresh_tool.tp_installed_release
SNAPSHOT = CATALOG_DIR / TP_REGISTRY_FILE
SOURCE = PKG / 'trainit_setup_assistant'

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')


def _live_tp():
    try:
        from trainit_perception.vocabulary import export
    except Exception:                                       # noqa: BLE001
        pytest.skip('trainit_perception (0.5.0 or newer) is not installed here')
    return export()


# --- 1. the snapshot is TP's vocabulary, and drift is named --------------------------------------

def test_the_snapshot_is_trainit_perceptions_vocabulary():
    live = _live_tp()
    drift = tp_drift(tp_registry(), live)
    assert not drift, ('\n'.join(drift) + f'\n-> refresh the catalog: {REFRESH}')
    assert tp_registry_text(live).encode() == SNAPSHOT.read_bytes(), (
        f'the snapshot is not the canonical text of TP\'s vocabulary -> {REFRESH}')


def _mutated(change):
    live = tp_registry()
    change(live)
    return live


def _rename_term(r, old, new):
    r['terms'][new] = r['terms'].pop(old)
    for rule, terms in r['rules'].items():
        r['rules'][rule] = [new if t == old else t for t in terms]


DRIFTS = [
    ('rule added', lambda r: r['rules'].update(candidate_x_new=['score']),
     "TP registry drift: rule 'candidate_x_new' exists in TP but is absent from TSA snapshot"),
    ('rule removed', lambda r: r['rules'].pop('score_x_clearance'),
     "TP registry drift: rule 'score_x_clearance' is in TSA snapshot but TP no longer exports it"),
    ('term renamed', lambda r: _rename_term(r, 'edge_on', 'edge_tilt'),
     "TP registry drift: term 'edge_on' is in TSA snapshot but TP no longer exports it "
     "(renamed 'edge_tilt'?)"),
    ('term requirement changed',
     lambda r: r['terms']['clearance'].update(positive_params=['tool_radius_m']),
     "TP registry drift: term 'clearance' positive_params is ['tool_radius_m'] in TP but "
     "['tool_radius_m', 'tool_reach_mm'] in TSA snapshot"),
    ('rule factors changed', lambda r: r['rules']['score_x_visible'].append('edge_on'),
     "TP registry drift: rule 'score_x_visible' multiplies ['score', 'visible_fraction', "
     "'edge_on'] in TP but ['score', 'visible_fraction'] in TSA snapshot"),
    ('default changed', lambda r: r['detector_methods']['color_mask']['parameters'][
        'min_area_px'].update(default=50),
     'TP registry drift: parameter color_mask.min_area_px default is 50 in TP but 60 in TSA '
     'snapshot'),
    ('allowed value changed', lambda r: r['detector_methods']['color_mask']['parameters'][
        'h'].update(max=180),
     'TP registry drift: parameter color_mask.h allowed values (max) is 180 in TP but 179 in '
     'TSA snapshot'),
    ('face added', lambda r: r['faces'].update(
        ellipse={'params': ['expected_half_a_m', 'expected_half_b_m'],
                 'enables_terms': ['visible_fraction']}),
     "TP registry drift: face 'ellipse' exists in TP but is absent from TSA snapshot"),
    ('face removed', lambda r: r['faces'].pop('rect'),
     "TP registry drift: face 'rect' is in TSA snapshot but TP no longer exports it"),
    ('face parameter order changed', lambda r: r['faces']['cylinder_side'].update(
        params=['expected_height_m', 'expected_radius_m']),
     "TP registry drift: face 'cylinder_side' params is ['expected_height_m', "
     "'expected_radius_m'] in TP but ['expected_radius_m', 'expected_height_m'] in TSA snapshot"),
    ('floor changed', lambda r: r['param_since'].update(tool_max_tilt_deg='0.5.0'),
     "TP registry drift: parameter floor 'tool_max_tilt_deg' arrived in '0.5.0' in TP but "
     "'0.4.0' in TSA snapshot"),
    ('default preference changed', lambda r: r['default_preference'].reverse(),
     'TP registry drift: default_preference is '),
    ('method kind changed', lambda r: r['detector_methods']['color_mask'].update(
        kind='external_node'),
     "TP registry drift: method 'color_mask' kind is 'external_node' in TP but 'pure_python'"),
    ('release changed', lambda r: r.update(trainit_perception='0.6.0'),
     "TP registry drift: trainit_perception is '0.6.0' in TP but"),
    ('rules reordered', lambda r: r.update(rules=dict(reversed(list(r['rules'].items())))),
     'TP registry drift: the rules are listed in another order in TP'),
    ('parameters reordered', lambda r: r['detector_methods']['color_mask'].update(
        parameters=dict(reversed(list(r['detector_methods']['color_mask']['parameters'].items())))),
     "TP registry drift: method 'color_mask' parameters are listed in another order in TP"),
    ('allowed value (min) changed', lambda r: r['detector_methods']['color_mask']['parameters'][
        'h'].update(min=1),
     'TP registry drift: parameter color_mask.h allowed values (min) is 1 in TP but 0'),
    ('allowed values (choices) changed', lambda r: r['detector_methods']['color_mask'][
        'parameters']['class_id'].update(choices=['object', 'part']),
     "TP registry drift: parameter color_mask.class_id allowed values (choices) is ['object', "
     "'part'] in TP but None"),
    ('type changed', lambda r: r['detector_methods']['color_mask']['parameters'][
        'blur_px'].update(type='float'),
     "TP registry drift: parameter color_mask.blur_px type is 'float' in TP but 'int'"),
    ('rule added and rules reordered', lambda r: r.update(rules=dict(
        [('candidate_x_new', ['score'])] + list(reversed(list(r['rules'].items()))))),
     'TP registry drift: the rules are listed in another order in TP'),
]


@pytest.mark.parametrize('what, change, says', DRIFTS, ids=[d[0] for d in DRIFTS])
def test_every_kind_of_drift_is_named(what, change, says):
    drift = tp_drift(tp_registry(), _mutated(change))
    assert any(says in line for line in drift), drift
    assert tp_drift(tp_registry(), tp_registry()) == []


def test_the_drift_m2_2_closed_is_the_one_the_owner_named():
    """Before M2.2 TSA had no rule ``candidate_x_square``: against today's TP that is exactly
    the owner's example sentence."""
    before = tp_registry()
    before['rules'].pop('candidate_x_square')
    assert ("TP registry drift: rule 'candidate_x_square' exists in TP but is absent from TSA "
            'snapshot') in tp_drift(before, tp_registry())


def test_an_incompatible_schema_is_refused_by_every_reader():
    newer = SNAPSHOT.read_bytes().replace(b'schema_version: 1', b'schema_version: 2', 1)
    assert newer != SNAPSHOT.read_bytes()
    with pytest.raises(RuntimeError, match=r'schema_version 2, and this TSA reads schema 1.*'
                                           r'readers updated first'):
        parse_tp_registry(newer, 'the snapshot')


@pytest.mark.parametrize('change, says', [
    (lambda r: r['rules']['score'].__setitem__(0, 'Score Alone'), 'not all declared terms'),
    (lambda r: r['terms']['score'].update(note='every detection carries one'), "term 'score'"),
    (lambda r: r['detector_methods']['color_mask']['parameters']['h'].update(
        label='Hue'), "parameter 'h'"),
    (lambda r: r['detector_methods']['color_mask']['parameters'].update(
        pose_quality_min_inliers={'type': 'int', 'default': 40}), 'gate or tool parameter'),
    (lambda r: r.update(notes=['measured on a real cell']), 'differ from schema'),
    (lambda r: r['faces']['disc'].update(extent_term_keys=['radius_ratio']), "face 'disc'"),
], ids=['prose_as_a_name', 'term_note', 'parameter_label', 'gate_default', 'top_level_prose',
        'extent_terms'])
def test_the_reader_refuses_what_must_not_cross_the_boundary(change, says):
    record = _mutated(change)
    problems = tp_registry_problems(record)
    assert any(says in p for p in problems), problems
    with pytest.raises(RuntimeError, match='not a trainit_perception vocabulary this TSA can read'):
        parse_tp_registry(tp_registry_text(record).encode(), 'mutated')


@pytest.mark.parametrize('change', [
    lambda r: r.update(default_face=['disc']),
    lambda r: r['detector_methods']['color_mask'].update(parameters=['h', 's']),
    lambda r: r['terms'].update(visible_fraction=['mask_area_px']),
    lambda r: r['terms']['clearance'].update(positive_params=[['tool_radius_m']]),
    lambda r: r.update(rules=['score']),
], ids=['default_face_list', 'parameters_list', 'term_list', 'unhashable_params', 'rules_list'])
def test_a_malformed_record_is_refused_with_sentences_never_a_crash(change):
    record = _mutated(change)
    problems = tp_registry_problems(record)
    assert problems and all(isinstance(p, str) for p in problems), problems
    with pytest.raises(RuntimeError, match='not a trainit_perception vocabulary'):
        parse_tp_registry(tp_registry_text(record).encode(), 'mutated')


def test_a_snapshot_that_is_not_yaml_says_how_to_repair_it():
    broken = SNAPSHOT.read_bytes().replace(
        b'trainit_perception: ', b'<<<<<<< ours\ntrainit_perception: ', 1) + b'>>>>>>> theirs\n'
    with pytest.raises(RuntimeError, match=r'not YAML .*' + re.escape(REFRESH)):
        parse_tp_registry(broken, 'the snapshot')


# --- 2. the snapshot itself --------------------------------------------------------------------

def test_the_snapshot_is_the_canonical_text_of_its_own_record():
    data = SNAPSHOT.read_bytes()
    assert tp_registry_text(parse_tp_registry(data, 'snapshot')).encode() == data


def test_the_catalog_records_the_snapshot_it_was_made_with():
    tp = catalog.snapshot()['registries']['trainit_perception']
    assert tp['vocabulary'] == {'file': TP_REGISTRY_FILE,
                                'sha256': hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
                                'schema_version': 1}
    assert tp['version'] == tp_registry()['trainit_perception']


def _leaves(node, path=()):
    if isinstance(node, dict):
        for k, v in node.items():
            yield path + (k,), k
            yield from _leaves(v, path + (k,))
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _leaves(v, path + (i,))
    else:
        yield path, node


def test_the_snapshot_holds_names_schema_and_requirements_only():
    """ADR-0021 Decision 9, checked on the committed bytes (TP's test checks its export): every
    string a name or a release, numbers only in a parameter's schema and the face band, no prose
    key, no gate value or tool default."""
    record = tp_registry()
    assert tp_registry_problems(record) == []
    for path, value in _leaves(record):
        if isinstance(value, bool):
            assert path[0] == 'terms' and path[-1] == 'face_sized', path
        elif isinstance(value, (int, float)):
            assert path in (('schema_version',),) or path[0] == 'face_size_band_m' or (
                path[0] == 'detector_methods' and path[2] == 'parameters'
                and path[4] in ('default', 'min', 'max')), (path, value)
        elif isinstance(value, str):
            assert re.match(r'^[a-z][a-z0-9_]*$|^\d+\.\d+\.\d+$', value), (path, value)
        else:
            assert value is None and path[0] == 'detector_methods', (path, value)
    keys = {k for p, k in _leaves(record) if isinstance(k, str) and p[-1] == k}
    assert not keys & {'note', 'notes', 'label', 'description', 'complement', 'extent_term_keys',
                       'weight', 'value'}


def _candidate_words():
    """Whole tokens of every perception / grasp / scene candidate (ADR-0021 D5 for TP's
    vocabulary): ids, packages, and the product word of each id. Generic tokens that the
    derivation produces from a product name ('ess' from ess_stereo, 'segment' from
    segment_anything) are not product words and would forbid ordinary English."""
    families = ('perception_', 'grasp_generation', 'scene')
    stop = {'ess', 'segment', 'nvidia', 'isaac', 'ros'}
    exact, words = set(), set()
    for c in catalog.candidates():
        if not c['family'].startswith(families):
            continue
        exact.add(c['id'].lower())
        exact.update(p.lower() for p in c.get('packages', []))
        product = c['id'].split('.', 1)[-1].lower()
        exact.add(product)
        words.update(w for w in product.split('_') if w not in stop and len(w) > 2)
    return exact, words


def _names(record):
    for path, value in _leaves(record):
        if isinstance(value, str) and not re.match(r'^\d', value):
            yield value


def test_no_candidate_provider_enters_the_vocabulary():
    exact, words = _candidate_words()
    assert {'foundationpose', 'rtdetr'} & words, 'the deny-list lost the NVIDIA perception words'
    offered = ([m.value for m in E.DetectionMethod] + [r.value for r in E.CandidateRule]
               + list(tp_registry()['faces']) + list(tp_registry()['terms']))
    for name in list(_names(tp_registry())) + offered:
        tokens = set(name.lower().split('_'))
        assert name.lower() not in exact and not tokens & words, name


def test_the_guard_would_catch_a_candidate_as_a_method():
    exact, words = _candidate_words()
    record = _mutated(lambda r: r['detector_methods'].update(
        foundationpose_tracker={'kind': 'external_node', 'provides': None, 'parameters': None}))
    hits = [n for n in _names(record) if set(n.split('_')) & words or n in exact]
    assert 'foundationpose_tracker' in hits


# --- 3. what TSA's code names is required; DetectionMethod agrees with the snapshot -------------

def _attribute_tokens(enum_name):
    out = set()
    for path in SOURCE.rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == enum_name and node.attr.isupper()):
                out.add((node.attr.lower(), str(path.relative_to(PKG))))
    return out


def test_every_rule_and_method_tsa_code_names_is_required():
    named_rules = _attribute_tokens('CandidateRule')
    assert named_rules, 'the scan found no CandidateRule.X at all'
    unrequired = sorted((t, f) for t, f in named_rules
                        if t not in TSA_REQUIRED_TP_RULES and t != catalog.TSA_AUTO)
    assert not unrequired, f'named in TSA code but not required of TP: {unrequired}'
    named_methods = _attribute_tokens('DetectionMethod')
    assert {t for t, _ in named_methods} <= set(TSA_TP_METHODS) | set(TSA_OWN_METHODS)


def test_detection_method_is_tsa_code_and_agrees_with_the_snapshot_both_ways():
    assert set(TSA_TP_METHODS) | set(TSA_OWN_METHODS) == {m.value for m in E.DetectionMethod}
    methods = tp_registry()['detector_methods']
    assert {m: s['kind'] for m, s in methods.items()} == TSA_TP_METHODS
    for m in E.DetectionMethod:
        if m.value in methods:      # a node is emitted exactly for the pure-python ones
            assert P.DetectorSpec(name='x', method=m).emits_node is (
                methods[m.value]['kind'] == 'pure_python'), m
    assert P.DetectorSpec(name='x', method='custom').emits_node is False


@pytest.mark.parametrize('change, says', [
    (lambda r: r['rules'].pop('score_x_visible'), "rule 'score_x_visible' is gone"),
    (lambda r: r['detector_methods'].update(
        custom={'kind': 'external_node', 'provides': None, 'parameters': None}),
     "method 'custom' has the name of a method TSA owns"),
    (lambda r: r['detector_methods'].update(
        shape_mask={'kind': 'pure_python', 'provides': ['score'], 'parameters': {}}),
     "method 'shape_mask' is one TSA's DetectionMethod does not handle yet"),
    (lambda r: r['detector_methods']['pcl_cluster'].update(kind='pure_python'),
     "method 'pcl_cluster' is pure_python"),
    (lambda r: r['faces']['disc'].update(params=['expected_diameter_m']), 'the default face'),
    (lambda r: r['detector_methods']['color_mask']['parameters'].pop('blur_px'),
     "colour-mask parameter 'blur_px' is gone"),
    (lambda r: r.update(auto='resolve'), 'TSA writes'),
], ids=['rule_tsa_names', 'reserved_method', 'unhandled_method', 'method_kind', 'default_face',
        'form_parameter', 'auto_token'])
def test_a_vocabulary_tsa_code_cannot_work_with_is_refused(change, says):
    problems = tp_vocabulary_problems(_mutated(change))
    assert any(says in p for p in problems), problems
    assert tp_vocabulary_problems(tp_registry()) == []


def test_a_snapshot_that_drops_a_token_tsa_names_stops_the_model_with_an_instruction():
    code = (
        'import trainit_setup_assistant.catalog as c\n'
        'rec = c.tp_registry()\n'
        "rec['rules'].pop('score_x_visible')\n"
        'c._tp_snapshot = lambda: rec\n'
        'import trainit_setup_assistant.model\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env)
    assert r.returncode != 0
    assert "rule 'score_x_visible' is gone" in r.stderr, r.stderr
    assert 'restore it with git' in r.stderr and 'change TSA first' in r.stderr, r.stderr


def test_every_term_has_tsa_words_and_none_are_tps():
    assert set(P.TERM_HINTS) == set(tp_registry()['terms'])
    assert not re.search(r'plane_ransac|centroid', P.TERM_HINTS['edge_on'])


# --- 4. the term model is TP's, and the GUI's form is the snapshot's ---------------------------

def _configs():
    faces = [{}, {'expected_radius_m': 0.02}]
    for name, row in tp_registry()['faces'].items():
        for k in range(len(row['params']) + 1):
            for subset in itertools.combinations(row['params'], k):
                faces.append(dict({p: 0.02 for p in subset}, grasp_face=name))
    tools = (('tool_radius_m', 0.012), ('tool_reach_mm', 4.0), ('tool_max_tilt_deg', 30.0))
    for face in faces:
        for on in itertools.product((0, 1), repeat=len(tools)):
            yield dict(face, **{n: v for (n, v), o in zip(tools, on) if o})


def test_tsas_computable_terms_are_trainit_perceptions():
    """For the colour mask (what it provides, from the snapshot) and for an external node (TSA
    cannot know what it provides, so everything is assumed) -- every face state, every tool
    state: TSA's answer from the snapshot is the runtime's answer."""
    try:
        from trainit_perception.ranking.rules import computable_terms
    except Exception:                                       # noqa: BLE001
        pytest.skip('trainit_perception is not installed here')
    provides_all = tp_registry()['provides']
    checked = 0
    for params in _configs():
        assert P.computable_terms(E.DetectionMethod.COLOR_MASK, params) == computable_terms(
            params, tp_registry()['detector_methods']['color_mask']['provides']), params
        assert P.computable_terms(E.DetectionMethod.CUSTOM, params) == computable_terms(
            params, provides_all), params
        checked += 1
    faces = tp_registry()['faces'].values()
    assert checked == (2 + sum(2 ** len(f['params']) for f in faces)) * 8   # every state, once


def test_edge_on_is_computable_where_tp_computes_it():
    tilt = {'tool_max_tilt_deg': 30.0}
    assert 'edge_on' in P.computable_terms(E.DetectionMethod.CUSTOM, tilt)
    assert 'edge_on' in P.computable_terms(E.DetectionMethod.PCL_CLUSTER, tilt)
    assert 'edge_on' not in P.computable_terms(E.DetectionMethod.COLOR_MASK, tilt)
    assert 'edge_on' not in P.computable_terms(E.DetectionMethod.CUSTOM, {})
    assert P.missing_candidate_terms(E.CandidateRule('candidate_x_square'),
                                     E.DetectionMethod.COLOR_MASK, tilt) == ('pose_quality',
                                                                             'edge_on')


@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_the_wizards_lists_and_colour_mask_form_are_the_snapshots(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import PerceptionPage, SetupWizard
    import m2_2_cases
    rules = [tok for tok, _ in PerceptionPage.RULES]
    assert sorted(rules) == sorted(r.value for r in E.CandidateRule) and len(set(rules)) == len(
        rules), 'every rule TSA offers is listed once in the wizard'
    methods = [tok for tok, _, _ in PerceptionPage.METHODS]
    # shape_mask: a declared roadmap placeholder, greyed, and refused by the model
    assert set(methods) - {'shape_mask'} == {m.value for m in E.DetectionMethod}
    assert [tok for tok, _, on in PerceptionPage.METHODS if on] == [
        m for m, s in tp_registry()['detector_methods'].items() if s['kind'] == 'pure_python']
    wiz = SetupWizard(AssistantController(m2_2_cases.CASES['plain']()))
    page = wiz.perception_page
    page.initializePage()
    schema = tp_registry()['detector_methods']['color_mask']['parameters']
    for (lo, hi), key in (((page.h_lo, page.h_hi), 'h'), ((page.s_lo, page.s_hi), 's'),
                          ((page.v_lo, page.v_hi), 'v')):
        for widget, default in ((lo, schema[key]['default'][0]), (hi, schema[key]['default'][1])):
            assert (widget.minimum(), widget.maximum(), widget.value()) == (
                schema[key]['min'], schema[key]['max'], default), key
    for widget, key in ((page.p_area, 'min_area_px'), (page.p_max, 'max_objects'),
                        (page.p_morph, 'morph_kernel'), (page.p_win, 'depth_window_px'),
                        (page.p_blur, 'blur_px'), (page.p_zmin, 'depth_min_m'),
                        (page.p_zmax, 'depth_max_m')):
        assert (widget.minimum(), widget.maximum(), widget.value()) == (
            schema[key]['min'], schema[key]['max'], schema[key]['default']), key
    assert set(page._params_from_form()) <= set(TSA_COLOR_MASK_FORM)
    assert page._params_from_form() == {k: schema[k]['default'] for k in page._params_from_form()}


@pytest.mark.parametrize('case', ['plain', 'tilt'])
def test_the_colour_mask_page_greys_the_edge_on_rules(qapp, case):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    import m2_2_cases
    wiz = SetupWizard(AssistantController(m2_2_cases.CASES[case]()))
    page = wiz.perception_page
    page.initializePage()
    model = page.p_rule.model()
    enabled = {tok for i, (tok, _) in enumerate(page.RULES) if model.item(i + 1).isEnabled()}
    assert not enabled & {'candidate_x_square', 'candidate_x_visible_x_square'}, enabled
    assert 'score_x_visible' in enabled


def test_an_external_node_ranking_by_edge_on_generates_a_whole_bundle(tmp_path):
    import m2_2_cases
    from trainit_setup_assistant.generator import Orchestrator
    p = m2_2_cases.project(method='custom', params={'tool_max_tilt_deg': m2_2_cases.TILT})
    p.perception.detectors[0] = P.DetectorSpec(
        name=m2_2_cases.DETECTOR, method='custom', candidate_rule='candidate_x_visible_x_square',
        params={'tool_max_tilt_deg': m2_2_cases.TILT})
    identity_cases.require('p2_example')
    Orchestrator().generate(p, str(tmp_path))
    det = yaml.safe_load((tmp_path / 'cell_app' / 'config' / 'perception.yaml').read_text())[
        'detectors'][m2_2_cases.DETECTOR]
    assert det['params']['candidate_rule'] == 'candidate_x_visible_x_square'
    assert det['params']['tool_max_tilt_deg'] == m2_2_cases.TILT
    xml = (tmp_path / 'cell_app' / 'package.xml').read_text()      # an external node: no TP pin
    assert not [ln for ln in xml.splitlines() if 'exec_depend' in ln and 'trainit_perception' in ln]


def test_a_face_that_does_not_produce_a_term_never_makes_it_computable(monkeypatch):
    """TP's semantics, not just its shipped data: a face-sized term is computable only through a
    face that ENABLES it -- every shipped face enables visible_fraction, so it takes a stand-in."""
    from trainit_setup_assistant.model import faces as F
    record = tp_registry()
    record['faces']['blob'] = {'params': ['expected_radius_m'], 'enables_terms': []}
    record = parse_tp_registry(tp_registry_text(record).encode(), 'the stand-in')
    monkeypatch.setattr(catalog, '_tp_snapshot', lambda: record)
    F.face_registry(refresh=True)
    try:
        sized = {'grasp_face': 'blob', 'expected_radius_m': 0.02}
        assert 'visible_fraction' not in P.computable_terms(E.DetectionMethod.COLOR_MASK, sized)
        assert 'visible_fraction' in P.computable_terms(
            E.DetectionMethod.COLOR_MASK, {'expected_radius_m': 0.02})
    finally:
        monkeypatch.undo()
        F.face_registry(refresh=True)


# --- 5. the refresh: validate everything, compare, then write all three files or none ----------

def _tree(here):
    return {f.name: (f.read_bytes(), f.stat().st_mtime_ns, f.stat().st_mode)
            for f in sorted(here.iterdir())}


@pytest.fixture
def tp_catalog(tmp_path, monkeypatch):
    """The refresh pointed at a scratch copy of the committed catalog, an 'installed' TMR registry
    equal to the committed copy, and a TP whose export the test chooses (default: the snapshot)."""
    here = tmp_path / 'catalog'
    here.mkdir()
    for name in (catalog.REGISTRY_FILE, TP_REGISTRY_FILE, refresh_tool.SNAPSHOT_FILE):
        (here / name).write_bytes((CATALOG_DIR / name).read_bytes())
    tmr = tmp_path / 'planner_profiles.yaml'
    tmr.write_bytes((CATALOG_DIR / catalog.REGISTRY_FILE).read_bytes())
    state = types.SimpleNamespace(here=here, tmr=tmr, tp=tp_registry(),
                                  release=tp_registry()['trainit_perception'])

    def record(tmr_bytes=None, tp_vocabulary=None):
        # a deterministic stand-in for derive(): it says which inputs it was handed
        return dict(fake=True, tp=tp_vocabulary['trainit_perception'],
                    rules=list(tp_vocabulary['rules']),
                    tmr=hashlib.sha256(tmr_bytes).hexdigest())

    first = refresh_tool.snapshot_text(record(tmr.read_bytes(), state.tp)).encode()
    (here / refresh_tool.SNAPSHOT_FILE).write_bytes(first)
    monkeypatch.setattr(refresh_tool, 'catalog_dir', lambda: here)
    monkeypatch.setattr(refresh_tool, 'tmr_registry_path', lambda: tmr)
    monkeypatch.setattr(refresh_tool, 'tp_available', lambda: True)
    monkeypatch.setattr(refresh_tool, 'tp_export', lambda: copy.deepcopy(state.tp))
    monkeypatch.setattr(refresh_tool, 'tp_installed_release', lambda: state.release)
    monkeypatch.setattr(refresh_tool, 'derive_record', record)
    return state


def test_a_current_catalog_is_left_alone(tp_catalog, capsys):
    before = _tree(tp_catalog.here)
    assert refresh_tool.main(['--check']) == 0
    assert refresh_tool.main([]) == 0
    assert 'nothing changed' in capsys.readouterr().out
    assert _tree(tp_catalog.here) == before


@pytest.mark.parametrize('change, says', [
    (lambda r: r['rules'].update(candidate_x_new=['score']),
     "rule 'candidate_x_new' exists in TP but has no Community/Pro classification"),
    (lambda r: r['rules'].pop('score_x_clearance'),
     "rule 'score_x_clearance' is gone, and TSA's code names it"),
    (lambda r: r['terms']['score'].update(note='measured'), 'not a trainit_perception vocabulary'),
    (lambda r: r.update(schema_version=2), 'readers updated first'),
    (lambda r: r['detector_methods']['color_mask']['parameters'].pop('blur_px'),
     "colour-mask parameter 'blur_px' is gone"),
    (lambda r: r['faces'].update(oval={'params': ['expected_radius_m'],
                                       'enables_terms': ['visible_fraction']}),
     "face 'oval' exists in TP but has no Community/Pro classification"),
], ids=['unclassified_rule', 'removed_rule', 'prose', 'schema', 'tsa_token', 'unclassified_face'])
def test_the_refresh_refuses_and_writes_nothing(tp_catalog, capsys, change, says):
    change(tp_catalog.tp)
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    err = capsys.readouterr().err
    assert says in err and 'nothing written' in err, err
    assert refresh_tool.main(['--check']) == 2
    assert _tree(tp_catalog.here) == before            # bytes, mtime and mode; no temp file


def test_the_refresh_refuses_a_removal_committed_projects_may_hold(tp_catalog, capsys,
                                                                  monkeypatch):
    """A face TSA's code does not name, even with its classification row gone too: projects may
    persist it, so its removal is refused at COMPARE -- after everything else validated."""
    tiers = copy.deepcopy(TP_TIERS)
    del tiers['face']['rect']
    monkeypatch.setattr(catalog, 'TP_TIERS', tiers)
    tp_catalog.tp['faces'].pop('rect')
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    err = capsys.readouterr().err
    assert "TP no longer exports faces 'rect'" in err and 'migrate those projects first' in err
    assert _tree(tp_catalog.here) == before


@pytest.mark.parametrize('export, says', [
    (lambda: {}['gamma'], "export() failed (KeyError: 'gamma')"),
    (lambda: __import__('no_such_module_cv2'), 'export() failed (ModuleNotFoundError'),
    (lambda: dict(tp_registry(), provides=frozenset({'score'})), 'is not plain data'),
    (lambda: dict(tp_registry(), default_preference=[]), 'the default rule preference is empty'),
    (lambda: {**tp_registry(), 'param_since': {k: v for k, v in tp_registry()[
        'param_since'].items() if k != 'blur_px'}}, "parameter 'blur_px' has no release floor"),
], ids=['keyerror', 'import_error', 'frozenset', 'empty_preference', 'noise_floor_gone'])
def test_a_broken_tp_export_is_refused_never_a_traceback(tp_catalog, monkeypatch, capsys,
                                                          export, says):
    """The refresh's own READ (``tp_export``) around a broken ``vocabulary.export()``: whatever
    it raises or returns, the answer is 'REFUSED, nothing written' (exit 2), never a traceback."""
    vocabulary = pytest.importorskip('trainit_perception.vocabulary')
    monkeypatch.setattr(refresh_tool, 'tp_export', TP_EXPORT)        # the real one, not a stub
    monkeypatch.setattr(vocabulary, 'export', export)
    before = _tree(tp_catalog.here)
    for argv in (['--check'], []):
        assert refresh_tool.main(argv) == 2
        assert says in capsys.readouterr().err
    assert _tree(tp_catalog.here) == before


@pytest.mark.parametrize('installed, says', [
    (None, 'is not on the ament index'),
    ('<package format="3"><name>trainit_perception</name></package>', 'names no <version>'),
    ('<package format="3"><version>', 'cannot be read'),
], ids=['not_on_the_index', 'no_version', 'not_xml'])
def test_the_refresh_refuses_a_tp_no_installed_package_xml_confirms(
        tp_catalog, monkeypatch, capsys, tmp_path, installed, says):
    """The real release check (``tp_installed_release``): an environment problem, refused with its
    own remedy -- never the 'change TSA' advice that a vocabulary problem gets."""
    aip = pytest.importorskip('ament_index_python.packages')
    pytest.importorskip('trainit_perception')
    monkeypatch.setattr(refresh_tool, 'tp_installed_release', TP_RELEASE)
    real = aip.get_package_share_directory

    def share(package, *args, **kwargs):
        if package != 'trainit_perception':
            return real(package, *args, **kwargs)
        if installed is None:
            raise aip.PackageNotFoundError(package)
        (tmp_path / 'package.xml').write_text(installed)
        return str(tmp_path)
    monkeypatch.setattr(aip, 'get_package_share_directory', share)
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    err = capsys.readouterr().err
    assert says in err and 'change TSA' not in err, err
    if installed is None:
        assert 'imported from' in err
    assert _tree(tp_catalog.here) == before


def test_an_unreadable_committed_snapshot_is_reported_and_replaced_whole(tp_catalog, capsys):
    (tp_catalog.here / refresh_tool.SNAPSHOT_FILE).write_bytes(
        (CATALOG_DIR / refresh_tool.SNAPSHOT_FILE).read_bytes())      # the names to judge by
    target = tp_catalog.here / TP_REGISTRY_FILE
    target.write_bytes(b'<<<<<<< ours\n' + target.read_bytes() + b'>>>>>>> theirs\n')
    assert refresh_tool.main(['--check']) == 1
    out = capsys.readouterr().out
    assert 'TSA snapshot unreadable' in out and 'not YAML' in out, out
    assert 'the refresh replaces it whole' in out
    assert refresh_tool.main([]) == 0
    assert target.read_bytes() == tp_registry_text(tp_catalog.tp).encode()


def test_an_unreadable_snapshot_still_refuses_a_removal_projects_may_hold(tp_catalog, monkeypatch,
                                                                         capsys):
    """When the committed TP snapshot cannot be read (a merge conflict, say) the removals are
    judged against the names snapshot.yaml records -- the conflict does not let a face go."""
    target = tp_catalog.here / TP_REGISTRY_FILE
    target.write_bytes(b'<<<<<<< ours\n' + target.read_bytes() + b'>>>>>>> theirs\n')
    (tp_catalog.here / refresh_tool.SNAPSHOT_FILE).write_bytes(
        (CATALOG_DIR / refresh_tool.SNAPSHOT_FILE).read_bytes())      # names rect among faces
    tiers = copy.deepcopy(TP_TIERS)
    del tiers['face']['rect']
    monkeypatch.setattr(catalog, 'TP_TIERS', tiers)
    tp_catalog.tp['faces'].pop('rect')
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    assert "TP no longer exports faces 'rect'" in capsys.readouterr().err
    assert _tree(tp_catalog.here) == before


def test_a_snapshot_changed_outside_the_refresh_is_named_and_replaced(tp_catalog, capsys):
    """Schema-valid but not what the refresh wrote (a key renamed by a bad merge): its sha256 is not
    the one snapshot.yaml records, so the refresh says so and writes the canonical file back."""
    (tp_catalog.here / refresh_tool.SNAPSHOT_FILE).write_bytes(
        (CATALOG_DIR / refresh_tool.SNAPSHOT_FILE).read_bytes())      # records the real sha256
    target = tp_catalog.here / TP_REGISTRY_FILE
    target.write_bytes(target.read_bytes().replace(b'candidate_x_nonoverlap:',
                                                   b'candida2e_x_nonoverlap:'))
    assert refresh_tool.main(['--check']) == 1
    assert 'was changed outside the refresh' in capsys.readouterr().out
    assert refresh_tool.main([]) == 0
    assert target.read_bytes() == SNAPSHOT.read_bytes()


def test_an_unreadable_snapshot_with_no_names_to_judge_by_is_refused(tp_catalog, capsys):
    target = tp_catalog.here / TP_REGISTRY_FILE
    target.write_bytes(b'<<<<<<< ours\n' + target.read_bytes() + b'>>>>>>> theirs\n')
    (tp_catalog.here / refresh_tool.SNAPSHOT_FILE).write_bytes(b'<<<<<<< not yaml either\n')
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    assert 'restore both files with git' in capsys.readouterr().err
    assert _tree(tp_catalog.here) == before


@pytest.mark.parametrize('break_tmr, says', [
    (lambda b: b'<<<<<<< ours\n' + b, 'not YAML'),
    (lambda b: b'- a list\n- not a registry\n', 'not a planner registry'),
    (lambda b: b'schema_version: 1\n', 'without its pipelines and planner_modes'),
    (lambda b: b'trainit_perception: 2026-13-45\n', 'not YAML'),
], ids=['conflict', 'a_list', 'no_sections', 'bad_date'])
def test_a_broken_installed_tmr_registry_is_refused_never_a_traceback(tp_catalog, capsys,
                                                                      break_tmr, says):
    tp_catalog.tmr.write_bytes(break_tmr(tp_catalog.tmr.read_bytes()))
    before = _tree(tp_catalog.here)
    for argv in (['--check'], []):
        assert refresh_tool.main(argv) == 2
        assert says in capsys.readouterr().err
    assert _tree(tp_catalog.here) == before


def test_the_refresh_names_a_reordered_vocabulary_and_writes_it(tp_catalog, capsys):
    tp_catalog.tp['rules'] = dict(reversed(list(tp_catalog.tp['rules'].items())))
    assert refresh_tool.main(['--check']) == 1
    out = capsys.readouterr().out
    assert 'the rules are listed in another order in TP' in out
    assert f'catalog DRIFT: {TP_REGISTRY_FILE}' in out
    assert refresh_tool.main([]) == 0
    assert refresh_tool.main(['--check']) == 0


def test_a_snapshot_whose_bytes_alone_differ_is_named(tp_catalog, capsys):
    target = tp_catalog.here / TP_REGISTRY_FILE
    target.write_bytes(target.read_bytes() + b'# a stray comment\n')
    assert refresh_tool.main(['--check']) == 1
    assert 'TSA snapshot is not the canonical text' in capsys.readouterr().out


def test_a_derive_step_that_fails_writes_nothing(tp_catalog, monkeypatch):
    tp_catalog.tp['param_since']['tool_max_tilt_deg'] = '0.5.0'

    def broken(**inputs):
        raise RuntimeError('derive failed after the validation passed')
    monkeypatch.setattr(refresh_tool, 'derive_record', broken)
    before = _tree(tp_catalog.here)
    with pytest.raises(RuntimeError, match='derive failed'):
        refresh_tool.refresh(False)
    assert _tree(tp_catalog.here) == before


def test_the_refresh_refuses_a_release_its_package_xml_does_not_declare(tp_catalog, capsys):
    tp_catalog.release = '0.4.0'
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    err = capsys.readouterr().err
    assert 'installed package.xml says 0.4.0' in err and 'change TSA' not in err, err
    assert _tree(tp_catalog.here) == before


def test_an_invalid_tmr_registry_leaves_tps_snapshot_alone_too(tp_catalog, capsys):
    tp_catalog.tmr.write_bytes(tp_catalog.tmr.read_bytes().replace(
        b'schema_version: 1', b'schema_version: 2'))
    tp_catalog.tp['detector_methods']['color_mask']['parameters']['h']['max'] = 180
    before = _tree(tp_catalog.here)
    assert refresh_tool.main([]) == 2
    assert _tree(tp_catalog.here) == before


def test_a_changed_vocabulary_is_reported_then_written_whole(tp_catalog, capsys):
    tp_catalog.tp['detector_methods']['color_mask']['parameters']['min_area_px']['default'] = 50
    before = _tree(tp_catalog.here)
    assert refresh_tool.main(['--check']) == 1
    out = capsys.readouterr().out
    assert 'parameter color_mask.min_area_px default is 50 in TP but 60' in out
    assert f'catalog DRIFT: {TP_REGISTRY_FILE}' in out
    assert _tree(tp_catalog.here) == before            # --check wrote nothing
    assert refresh_tool.main([]) == 0
    after = _tree(tp_catalog.here)
    assert after[TP_REGISTRY_FILE][0] == tp_registry_text(tp_catalog.tp).encode()
    assert after[TP_REGISTRY_FILE][2] == before[TP_REGISTRY_FILE][2]      # the mode is kept
    assert set(after) == set(before)                                       # no temp file left
    assert refresh_tool.main(['--check']) == 0                             # a fixed point


def test_a_write_that_fails_half_way_puts_every_file_back(tp_catalog, monkeypatch):
    tp_catalog.tmr.write_bytes(tp_catalog.tmr.read_bytes() + b'# TMR moved\n')
    tp_catalog.tp['param_since']['tool_max_tilt_deg'] = '0.5.0'
    before = {name: data for name, (data, _, mode) in _tree(tp_catalog.here).items()}
    real, calls = os.replace, []

    def flaky(src, dst):
        calls.append(dst)
        if len(calls) == 2:
            raise OSError('disk full, on the second file')
        return real(src, dst)

    monkeypatch.setattr(refresh_tool.os, 'replace', flaky)
    with pytest.raises(OSError):
        refresh_tool.refresh(False)
    assert {name: data for name, (data, _, _) in _tree(tp_catalog.here).items()} == before


def test_the_file_that_records_the_others_is_written_last_from_non_yaml_temps(tp_catalog,
                                                                              monkeypatch):
    tp_catalog.tmr.write_bytes(tp_catalog.tmr.read_bytes() + b'# TMR moved\n')
    tp_catalog.tp['param_since']['tool_max_tilt_deg'] = '0.5.0'
    real, moves = os.replace, []

    def record(src, dst):
        moves.append((os.path.basename(src), os.path.basename(dst)))
        return real(src, dst)

    monkeypatch.setattr(refresh_tool.os, 'replace', record)
    assert refresh_tool.main([]) == 0
    assert [dst for _, dst in moves] == [catalog.REGISTRY_FILE, TP_REGISTRY_FILE,
                                         refresh_tool.SNAPSHOT_FILE]
    assert not any(src.endswith('.yaml') for src, _ in moves), moves


def test_a_refresh_leaves_no_global_state_behind(tp_catalog):
    tmr_before, tp_before = catalog.tmr_planner_registry(), tp_registry()
    tp_catalog.tp['rules'].pop('score')            # refused (a removal)
    assert refresh_tool.main([]) == 2
    tp_catalog.tp = tp_registry()
    tp_catalog.tp['param_since']['tool_max_tilt_deg'] = '0.5.0'
    assert refresh_tool.main(['--check']) == 1
    assert refresh_tool.main([]) == 0
    assert catalog.tmr_planner_registry() == tmr_before
    assert tp_registry() == tp_before == parse_tp_registry(SNAPSHOT.read_bytes(), 'committed')


def _workspace_installed():
    """The refresh's own READ of TMR and TP works here (the tests below run it for real)."""
    try:
        refresh_tool.tmr_registry_path()
        TP_RELEASE()
        return True
    except Exception:                                       # noqa: BLE001
        return False


def _run(code, cwd, pythonpath):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([str(pythonpath), os.environ.get('PYTHONPATH', '')]))
    return subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                          cwd=str(cwd))


def test_no_model_built_from_a_candidate_outlives_the_refresh(tmp_path):
    """The CLI path, in a fresh interpreter: the refresh DERIVES in a child interpreter, so after a
    --check against a candidate vocabulary (twice: pydantic 1.x refuses a model declared twice in
    one process) the process imports the model from the committed snapshot -- by the dotted name
    and through the package attribute alike."""
    if not _workspace_installed():
        pytest.skip('needs the installed workspace (TMR, TP on the ament index)')
    here = tmp_path / 'catalog'
    here.mkdir()
    for name in (catalog.REGISTRY_FILE, TP_REGISTRY_FILE, refresh_tool.SNAPSHOT_FILE):
        (here / name).write_bytes((CATALOG_DIR / name).read_bytes())
    code = (
        'import copy, sys\n'
        'from pathlib import Path\n'
        'from trainit_setup_assistant import catalog as C\n'
        'from trainit_setup_assistant.catalog import refresh as R\n'
        'assert "trainit_setup_assistant.model" not in sys.modules\n'
        f'R.catalog_dir = lambda: Path({str(here)!r})\n'
        'live = C.tp_registry()\n'
        "live['rules']['score_x_visible'] = ['score', 'visible_fraction', 'clearance']\n"
        'R.tp_export = lambda: copy.deepcopy(live)\n'
        "print('CHECK', R.main(['--check']), R.main(['--check']))\n"
        "print('MODEL IMPORTED BY THE REFRESH', 'trainit_setup_assistant.model' in sys.modules)\n"
        'from trainit_setup_assistant.model import perception as P, enums as E\n'
        'import trainit_setup_assistant as T\n'
        "rule = E.CandidateRule('score_x_visible')\n"
        "print('TERMS', list(P.CANDIDATE_RULE_TERMS[rule]), "
        "list(T.model.perception.CANDIDATE_RULE_TERMS[rule]))\n")
    r = _run(code, tmp_path, PKG)
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'CHECK 1 1' in r.stdout, r.stdout + r.stderr
    assert 'MODEL IMPORTED BY THE REFRESH False' in r.stdout, r.stdout
    assert ("TERMS ['score', 'visible_fraction'] ['score', 'visible_fraction']"
            in r.stdout), r.stdout


def test_the_cli_repairs_a_snapshot_the_model_cannot_read(tmp_path):
    """The real command on a copy of this package whose committed TP snapshot is conflict-marked:
    the model cannot import, --check names the file, the refresh writes it back, and the model
    imports again."""
    if not _workspace_installed():
        pytest.skip('needs the installed workspace (TMR, TP on the ament index)')
    import shutil
    copy_root = tmp_path / 'copy'
    shutil.copytree(SOURCE, copy_root / 'trainit_setup_assistant',
                    ignore=shutil.ignore_patterns('__pycache__'))
    target = copy_root / 'trainit_setup_assistant' / 'catalog' / TP_REGISTRY_FILE
    target.write_bytes(b'<<<<<<< ours\n' + target.read_bytes() + b'>>>>>>> theirs\n')
    cli = ('import sys\nfrom trainit_setup_assistant.catalog import refresh as R\n'
           'sys.exit(R.main(sys.argv[1:]))\n')
    model = 'import trainit_setup_assistant.model\n'
    broken = _run(model, tmp_path, copy_root)
    assert broken.returncode != 0 and 'not YAML' in broken.stderr, broken.stderr
    check = subprocess.run([sys.executable, '-c', cli, '--check'], capture_output=True,
                           text=True, cwd=str(tmp_path), env=dict(
                               os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
                                   [str(copy_root), os.environ.get('PYTHONPATH', '')])))
    assert check.returncode == 1 and 'TSA snapshot unreadable' in check.stdout, (
        check.stdout + check.stderr)
    write = subprocess.run([sys.executable, '-c', cli], capture_output=True, text=True,
                           cwd=str(tmp_path), env=dict(
                               os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONPATH=os.pathsep.join(
                                   [str(copy_root), os.environ.get('PYTHONPATH', '')])))
    assert write.returncode == 0, write.stdout + write.stderr
    assert target.read_bytes() == SNAPSHOT.read_bytes()
    assert _run(model, tmp_path, copy_root).returncode == 0


def test_after_a_write_the_face_registry_is_read_again_lazily():
    from trainit_setup_assistant.model import faces as F
    F.face_registry()
    assert F._cache is not None
    refresh_tool._reread_catalog()
    assert F._cache is None                    # not re-read here: a broken file must not raise
    assert F.face_registry() == F.face_registry(refresh=True)


def test_a_missing_generated_file_says_how_to_get_it_back():
    with pytest.raises(RuntimeError, match=r'catalog/no_such\.yaml is missing.*restore it with git'):
        catalog._committed('no_such.yaml')
    with pytest.raises(RuntimeError, match='not YAML'):
        catalog.parse_planner_registry(b'<<<<<<< ours\nschema_version: 1\n', 'a copy')
    with pytest.raises(RuntimeError, match='not YAML'):
        parse_tp_registry(b'trainit_perception: 2026-13-45\n', 'a snapshot')


def test_the_tiers_classify_exactly_what_tp_ships():
    assert tp_tier_problems(tp_registry()) == []
    reclassified = copy.deepcopy(TP_TIERS)
    del reclassified['face']['rect']
    reclassified['rule']['retired_rule'] = 'pro_candidate'
    problems = tp_tier_problems(tp_registry(), reclassified)
    assert any("face 'rect' exists in TP but has no Community/Pro classification" in p
               for p in problems), problems
    assert any("classifies rule 'retired_rule', which TP does not ship" in p for p in problems)


def test_a_changed_classification_is_named_by_the_catalog_drift(monkeypatch):
    """Brief section 7: a Community/Pro classification is represented in the catalog
    (``snapshot.yaml``), so changing one is caught -- and the failure names the row and field."""
    pytest.importorskip('ament_index_python')
    from trainit_setup_assistant.catalog import derive as D
    try:
        derived = D.derive(tp_vocabulary=tp_registry())
    except Exception as exc:                               # noqa: BLE001
        pytest.skip(f'needs the installed workspace (TMR, MoveIt): {exc}')
    # the baseline without what this checkout cannot derive (catalog_identity): the claim below
    # runs in full on any checkout, and the test then says what the baseline did not compare
    absent = catalog_identity.absent_packages()
    committed, derived = (catalog_identity.without_packages(r, absent)
                          for r in (catalog.snapshot(), derived))
    assert D.catalog_drift(committed, derived) == []
    tiers = copy.deepcopy(TP_TIERS)
    tiers['rule']['score'] = 'pro_candidate'
    monkeypatch.setattr(D, 'TP_TIERS', tiers)
    drift = D.catalog_drift(catalog.snapshot(), D.derive(tp_vocabulary=tp_registry()))
    assert ("catalog drift: tp.rule.score distribution is 'community_candidate' in snapshot.yaml "
            "but 'pro_candidate' from derive()") in drift, drift
    catalog_identity.require_installed(absent)
