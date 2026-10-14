"""ADR-0017: the candidate ranking as CONFIGURATION — and the gating is half the test.

What this pins, per ADR-0017 Decision:

1. **Decision 1** — the rank is a DETECTOR parameter: ``candidate_rule`` is enum-validated
   the way ``method`` is, and it travels ``DetectorSpec`` -> ``params`` -> the
   ``perception.yaml`` entry the runtime reads. Nothing is written onto the application.
2. **"only when set"** — an unset rule writes NO key, so a project that never configured a
   ranking generates byte-identical output (``test_generator_golden.py`` /
   ``test_backend_bytestability.py`` are the gates; this file proves the key's absence
   directly, which is the fact those gates depend on).
3. **Decision 3** — a rule whose TERMS this detector cannot compute is REFUSED at
   generation time, before a single file exists, and the message names the missing term,
   what would supply it, and the rules that ARE available. Two reasons a term can be
   uncomputable and both are configuration: the METHOD does not produce what the term
   reads (the colour mask runs no 3D fit, so ``pose_quality`` — and with it every
   ``candidate_*`` rule — is out), or the PARAMETER it needs is not set.
4. **the two homeless parameters** — the tool's footprint radius and grip window live on
   ``GripperSpec`` (a tool property on the tool) and are copied into every detector's
   params; the object's FACE RADIUS gets no field at all and is derived from the grasp
   target's shape and dims.
5. **the version pin** — a bundle carrying a rule pins the ``trainit_perception`` that
   understands it, through the mechanism that already pins the noise filters.

Fast and ROS-free: generation is pure.
"""

import os
import tempfile
from pathlib import Path

import pytest
import yaml

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from trainit_setup_assistant.generator import Orchestrator  # noqa: E402
from trainit_setup_assistant.generator.perception_yaml import (  # noqa: E402
    build_perception_yaml, detector_params, ranking_param_names,
    refuse_uncomputable_candidate_rules)
from trainit_setup_assistant.model import (BundleSpec, CandidateRule,  # noqa: E402
                                           DetectorSpec, PerceptionSpec, SceneObject,
                                           load_project, project_to_dict)
from trainit_setup_assistant.model.project import CanonicalProject  # noqa: E402
import identity_cases  # noqa: E402
import ws_paths  # noqa: E402

HERE = os.path.dirname(__file__)
EXAMPLE = os.path.join(HERE, os.pardir, 'examples', 'fr3wml_project.yaml')
GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'


@pytest.fixture(scope='module')
def qapp():
    from python_qt_binding.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


# --- fixtures -------------------------------------------------------------------

_DIMS = {'cylinder': [0.0135, 0.04], 'sphere': [0.02], 'box': [0.03, 0.02, 0.01],
         'mesh': [0.03, 0.02, 0.01]}

# The colour mask publishes a score, a box and a mask area and runs NO 3D fit, so every
# `candidate_*` rule (they all take `pose_quality`) is refused on it. An EXTERNAL-NODE
# method declares no capabilities to TSA, so only its parameters are judged — which is how
# a test reaches the `candidate_*` family at all.
MASK_RULE = CandidateRule.SCORE_X_VISIBLE
FIT_METHOD = 'custom'


def _project(rule=None, *, radius=0.0, window=0.0, target='cylinder', params=None,
             method='color_mask'):
    """The example project with one detector, optionally ranking, plus a grasp target.

    ``target``: the grasp target's shape — ``None`` removes it, which is how a project
    whose ``visible_fraction`` has no face area is built.
    """
    p = load_project(EXAMPLE)
    p.bundle = BundleSpec.from_prefix('cell')
    p.perception = p.perception or PerceptionSpec()
    p.perception.detectors = [DetectorSpec(name='capsules', method=method,
                                           params=dict(params or {}),
                                           candidate_rule=rule)]
    p.robot.gripper.tool_radius_m = radius
    p.robot.gripper.tool_reach_mm = window
    if target is not None:
        p.scene.objects = [o for o in p.scene.objects if not o.is_grasp_target()] + [
            SceneObject(id='capsule', shape=target, dims=_DIMS[target],
                        category='dynamic', grasp_target=True)]
    return p


def _entry(project, name='capsules'):
    return yaml.safe_load(build_perception_yaml(project))['detectors'][name]


def _generate(project):
    identity_cases.require('p2_example')          # every caller generates the example
    tmp = tempfile.TemporaryDirectory()
    Orchestrator().generate(project, tmp.name)
    return tmp, Path(tmp.name)


# --- 1. the rank is a DETECTOR parameter, enum-validated (Decision 1) -----------

def test_candidate_rule_is_enum_validated_like_the_detection_method():
    """A typo is not a worse ranking, it is NO ranking — so it is refused at the model
    boundary, exactly where an unknown ``method`` is."""
    with pytest.raises(Exception):
        DetectorSpec(name='capsules', candidate_rule='candidate_2d3D')   # plausible typo
    with pytest.raises(Exception):
        DetectorSpec(name='capsules', candidate_rule='best')
    for rule in CandidateRule:
        assert DetectorSpec(name='capsules',
                            candidate_rule=rule.value).candidate_rule is rule


def test_the_rule_reaches_the_runtime_through_the_params_dict():
    entry = _entry(_project(CandidateRule.SCORE_X_NONOVERLAP))
    assert entry['params']['candidate_rule'] == 'score_x_nonoverlap'
    # it rides in params, where make_detector reads it — not as a sibling key
    assert 'candidate_rule' not in {k for k in entry if k != 'params'}


def test_the_rule_round_trips_through_project_yaml():
    p = _project(MASK_RULE)
    back = CanonicalProject.parse_obj(project_to_dict(p))
    assert back.perception.detectors[0].candidate_rule is MASK_RULE


def test_the_application_layer_is_untouched():
    """ADR-0017 Decision 1: the rank is computed by the detector, so nothing about it may
    appear in the generated tree or bt_params — TMR 1.6.0 consumes an ORDER and no
    vocabulary (ADR-0016 Decision 1)."""
    tmp, root = _generate(_project(CandidateRule.SCORE_X_NONOVERLAP))
    with tmp:
        app = root / 'cell_app'
        tree = (app / 'bt_trees' / 'pick_place.xml').read_text()
        bt_params = (app / 'config' / 'bt_params.yaml').read_text()
        for text in (tree, bt_params):
            assert 'candidate_rule' not in text
            assert 'visible_fraction' not in text


# --- 2. written ONLY when set (the byte gate's precondition) --------------------

def test_an_unset_rule_writes_no_key_at_all():
    """The precedent is ``blur_px`` / the depth bounds: perception.yaml is compared
    YAML-semantically against the golden, so a key that always landed would silently
    re-bless every published bundle."""
    entry = _entry(_project(None, radius=0.0, window=0.0))
    assert set(entry['params']) == set()
    assert 'candidate_rule' not in entry['params']
    assert 'expected_radius_m' not in entry['params']


def test_an_unset_rule_derives_no_face_radius_even_with_a_grasp_target():
    """The face radius is DERIVED, so it must not appear in a bundle whose ranking was
    never configured — the golden has a grasp target and no rule."""
    assert 'expected_radius_m' not in _entry(_project(None))['params']


def test_the_golden_project_is_untouched_by_the_feature():
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    p = load_project(GOLDEN_PROJECT)
    assert p.perception.detectors[0].candidate_rule is None
    assert p.robot.gripper.tool_radius_m == 0.0 and p.robot.gripper.tool_reach_mm == 0.0
    params = detector_params(p, p.perception.detectors[0])
    assert params == dict(p.perception.detectors[0].params)       # nothing added


# --- 3. the refusal (Decision 3) ------------------------------------------------

def test_a_rule_the_METHOD_cannot_compute_is_refused():
    """The colour mask runs no 3D fit, so every ``candidate_*`` rule is out — and that is
    a configuration fact TSA knows before any frame arrives. Offering those four on it
    would offer four rules that stop the node."""
    for rule in (CandidateRule.CANDIDATE_2D3D, CandidateRule.CANDIDATE_X_VISIBLE,
                 CandidateRule.CANDIDATE_X_NONOVERLAP,
                 CandidateRule.CANDIDATE_X_CLEARANCE):
        p = _project(rule, radius=0.012, window=8.0)
        with pytest.raises(ValueError) as exc:
            refuse_uncomputable_candidate_rules(p)
        msg = str(exc.value)
        assert 'pose_quality' in msg and '3D FIT' in msg
        assert 'score_x_' in msg                      # names what IS available


def test_an_external_node_method_is_not_refused_on_a_capability_tsa_cannot_know():
    """An external-node method publishes the contract from its own node, so TSA has no
    field for what it produces. Refusing it on a capability TSA merely failed to know
    would be refusing the expert seam; the node makes the final check on its own file."""
    refuse_uncomputable_candidate_rules(
        _project(CandidateRule.CANDIDATE_2D3D, method=FIT_METHOD))


def test_visible_without_a_face_area_is_refused_not_demoted():
    """The asymmetry ADR-0017 Decision 3 closes: an unset radius used to turn
    ``candidate_x_visible`` into ``candidate_2d3d`` in silence (both 0,7000 on the same
    frame), dropping the pose gate with it."""
    p = _project(MASK_RULE, target=None)
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    msg = str(exc.value)
    assert 'score_x_visible' in msg and 'visible_fraction' in msg
    assert 'grasp target' in msg                      # says WHERE to fix it


def test_a_grasp_target_with_no_usable_dims_is_refused_too():
    p = _project(MASK_RULE)
    p.scene.objects[-1].dims = [0.0, 0.0]
    with pytest.raises(ValueError):
        refuse_uncomputable_candidate_rules(p)


@pytest.mark.parametrize('radius,window', [(0.0, 0.0), (0.012, 0.0), (0.0, 8.0)])
def test_clearance_without_the_tool_numbers_is_refused(radius, window):
    """Both numbers are required: a zero grip window makes the clearance term identically
    1.0, i.e. a ranking that silently collapses into ``candidate_2d3d`` — the same failure
    as an unknown rule, so the same refusal."""
    p = _project(CandidateRule.SCORE_X_CLEARANCE, radius=radius, window=window,
                 method=FIT_METHOD)
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    msg = str(exc.value)
    assert 'clearance' in msg
    assert 'End-effector' in msg                      # says WHERE to fix it


def test_clearance_with_both_tool_numbers_is_accepted():
    p = _project(CandidateRule.SCORE_X_CLEARANCE, radius=0.012, window=8.0,
                 method=FIT_METHOD)
    refuse_uncomputable_candidate_rules(p)                # does not raise
    params = _entry(p)['params']
    assert params['candidate_rule'] == 'score_x_clearance'
    assert params['tool_radius_m'] == 0.012 and params['tool_reach_mm'] == 8.0


@pytest.mark.parametrize('rule', [CandidateRule.AUTO, CandidateRule.DETECTOR_ORDER,
                                  CandidateRule.SCORE, CandidateRule.SCORE_X_NONOVERLAP])
def test_the_rules_that_need_nothing_configured_are_always_computable(rule):
    """``auto`` is resolved by the runtime from what the detector can compute, and
    ``detector_order`` is an EXPLICIT "no ranking" — which is exactly what distinguishes it
    from a rule that silently could not be computed."""
    refuse_uncomputable_candidate_rules(_project(rule, target=None))


def test_generation_refuses_BEFORE_writing_a_single_file():
    """A partial bundle still helps for a warning; a bundle that ranks nothing while
    looking like it ranks does not — so the refusal runs before the first emitter."""
    p = _project(MASK_RULE, target=None)
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(ValueError):
            Orchestrator().generate(p, os.path.join(tmp, 'bundle'))
        assert not os.path.exists(os.path.join(tmp, 'bundle'))


def test_the_emitter_refuses_on_its_own_too():
    """``build_perception_yaml`` is reachable from the live tuner and from tests, so the
    name is checked where it is READ and not only in the orchestrator."""
    with pytest.raises(ValueError):
        build_perception_yaml(_project(MASK_RULE, target=None))


# --- 4. the three numbers: two on the TOOL, one DERIVED ------------------------

@pytest.mark.parametrize('shape,expected', [
    ('cylinder', 0.0135),                       # dims=[r, h] -> the circular end face
    ('sphere', 0.02),                           # dims=[r]    -> every face is that disc
    ('box', (0.03 * 0.02 / 3.141592653589793) ** 0.5),    # equal-AREA disc of the x-y face
    ('mesh', (0.03 * 0.02 / 3.141592653589793) ** 0.5),   # its AABB's x-y face
])
def test_the_face_radius_is_derived_from_the_grasp_targets_shape_and_dims(shape, expected):
    p = _project(MASK_RULE, target=shape)
    assert _entry(p)['params']['expected_radius_m'] == pytest.approx(expected, abs=1e-5)


def test_only_the_grasp_target_supplies_the_face_radius():
    """A dynamic object that is not a grasp target (the crate) is not what is picked."""
    p = _project(None, target=None)
    p.scene.objects.append(SceneObject(id='crate', shape='box', dims=[0.4, 0.3, 0.2],
                                       category='dynamic', grasp_target=False))
    assert p.scene.grasp_face_radius_m() == 0.0


def test_a_hand_edited_params_value_always_wins():
    """``params`` is hand-editable data the GUI cannot see; a derivation must not clobber
    a number the configurator measured."""
    p = _project(MASK_RULE, radius=0.012, window=8.0,
                 params={'expected_radius_m': 0.009, 'tool_radius_m': 0.004})
    params = _entry(p)['params']
    assert params['expected_radius_m'] == 0.009          # not the derived 0.0135
    assert params['tool_radius_m'] == 0.004              # not the gripper's 0.012
    assert params['tool_reach_mm'] == 8.0                # the one it did not set


def test_the_tool_numbers_travel_without_a_rule_because_clearance_is_a_TERM():
    """ADR-0017 Decision 4: ``clearance`` ships as a published term without being a
    ranking default — a cell that declares its tool gets the term whatever it ranks by."""
    params = _entry(_project(None, radius=0.012, window=8.0))['params']
    assert params['tool_radius_m'] == 0.012 and params['tool_reach_mm'] == 8.0
    assert 'candidate_rule' not in params


def test_the_tool_numbers_are_absent_when_the_tool_declares_none():
    params = _entry(_project(CandidateRule.SCORE))['params']
    assert 'tool_radius_m' not in params and 'tool_reach_mm' not in params


def test_the_tool_geometry_round_trips_and_refuses_a_negative():
    p = _project(None, radius=0.012, window=8.0)
    back = CanonicalProject.parse_obj(project_to_dict(p))
    assert back.robot.gripper.tool_radius_m == 0.012
    assert back.robot.gripper.tool_reach_mm == 8.0
    with pytest.raises(Exception):
        type(p.robot.gripper)(tool_radius_m=-0.01)


# --- 5. the version pin ---------------------------------------------------------

def _app_package_xml(project) -> str:
    tmp, root = _generate(project)
    with tmp:
        return (root / 'cell_app' / 'package.xml').read_text()


def test_a_rule_pins_the_trainit_perception_that_understands_it():
    xml = _app_package_xml(_project(MASK_RULE))
    assert '<exec_depend version_gte="0.3.0">trainit_perception</exec_depend>' in xml


def test_the_tool_numbers_pin_it_too_because_they_are_ranking_parameters():
    xml = _app_package_xml(_project(None, radius=0.012, window=8.0))
    assert 'version_gte="0.3.0"' in xml


def test_the_noise_filter_pin_is_unchanged_on_its_own():
    xml = _app_package_xml(_project(None, params={'blur_px': 3}))
    assert '<exec_depend version_gte="0.2.0">trainit_perception</exec_depend>' in xml


def test_the_floor_is_the_HIGHEST_pin_the_parameters_demand():
    xml = _app_package_xml(_project(CandidateRule.SCORE,
                                    params={'blur_px': 3, 'depth_max_m': 1.2}))
    assert 'version_gte="0.3.0"' in xml and 'version_gte="0.2.0"' not in xml


def test_no_ranking_and_no_filters_pins_nothing():
    xml = _app_package_xml(_project(None))
    assert '<exec_depend>trainit_perception</exec_depend>' in xml
    # trainit_perception carries no floor (trainit_motion_runtime's is vNext M2.4's)
    assert not [line for line in xml.splitlines()
                if 'version_gte' in line and 'trainit_perception' in line]


def test_every_ranking_parameter_is_covered_by_the_pin():
    """The pin is derived from the same list the generator writes from, so a new ranking
    parameter cannot reach a bundle without moving the floor that guarantees it is read.

    The list is ASKED of the face registry (ADR-0018 Decision 6), so this also covers the
    parameters of a face family TSA has never heard of: what the generator can write and
    what the pin covers are the same question asked once.
    """
    from trainit_setup_assistant.model.faces import param_floors

    names = ranking_param_names()
    p = _project(CandidateRule.SCORE_X_CLEARANCE, radius=0.012, window=8.0,
                 method=FIT_METHOD)
    written = set(detector_params(p, p.perception.detectors[0]))
    assert written <= set(names), written - set(names)
    assert {'candidate_rule', 'grasp_face', 'expected_radius_m',
            'tool_radius_m', 'tool_reach_mm'} <= set(names)

    # EVERY writable parameter must be DATED. An undated one is the silent failure: written into
    # a bundle, it merges into an older runtime's params and does nothing.
    floors = param_floors()
    undated = sorted(set(names) - set(floors))
    assert not undated, f'undated ranking parameters: {undated}'

    # and each one ALONE moves the floor to ITS OWN, not to a fixed number. Pinning them all at
    # 0.3.0 is what let `tool_max_tilt_deg` (0.4.0) be written under a floor that ignores it.
    for name in names:
        value = ('score' if name == 'candidate_rule'
                 else 'disc' if name == 'grasp_face' else 0.01)
        want = f'version_gte="{floors[name]}"'
        assert want in _app_package_xml(_project(None, params={name: value})), (name, want)


# --- 6. a hand-edited params key is still TSA's problem ------------------------

def test_a_hand_edited_rule_in_params_is_refused_like_the_field():
    """``params`` is a verbatim runtime profile a configurator may edit by hand, so the
    key can arrive that way — and a rule TSA cannot see is a rule TSA cannot refuse."""
    p = _project(None, target=None, params={'candidate_rule': 'score_x_visible'})
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    assert 'visible_fraction' in str(exc.value)


def test_a_hand_edited_rule_in_params_still_gets_the_derived_radius():
    p = _project(None, params={'candidate_rule': 'score_x_visible'})
    refuse_uncomputable_candidate_rules(p)           # computable: a grasp target exists
    assert _entry(p)['params']['expected_radius_m'] == pytest.approx(0.0135)


def test_an_unknown_hand_edited_rule_token_is_refused_by_name():
    p = _project(None, params={'candidate_rule': 'best_one'})
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    msg = str(exc.value)
    assert 'best_one' in msg and 'candidate_x_visible' in msg


# --- 7. the GUI (headless) ------------------------------------------------------

def test_the_controller_previews_the_refusal_the_generator_would_raise():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project(None, target=None))
    assert ctrl.candidate_rule_problem('') is None
    assert ctrl.candidate_rule_problem('score') is None
    assert 'visible_fraction' in ctrl.candidate_rule_problem('score_x_visible')
    assert 'pose_quality' in ctrl.candidate_rule_problem('candidate_2d3d')
    ctrl = AssistantController(_project(None, radius=0.012, window=8.0))
    assert ctrl.candidate_rule_problem('score_x_clearance', method=FIT_METHOD) is None


def test_the_controller_offers_only_the_rules_this_detector_can_run():
    """The GUI greys the rest, so a configurator is never offered a rule that would stop
    the node (ADR-0017 Decision 3), and the pre-selected one is the measured best among
    what IS available (Decision 4)."""
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project(None))          # a grasp target, no tool numbers
    offered, preferred = ctrl.candidate_rules_for()
    tokens = {r.value for r in offered}
    assert 'score_x_visible' in tokens                  # mask area + a derived face radius
    assert 'score_x_nonoverlap' in tokens               # boxes
    assert 'candidate_x_visible' not in tokens          # no 3D fit on a colour mask
    assert 'score_x_clearance' not in tokens            # no tool numbers
    assert preferred.value == 'score_x_visible'
    # with no grasp target the face area goes, and the fallback is the box rule
    _, fallback = AssistantController(_project(None, target=None)).candidate_rules_for()
    assert fallback.value == 'score_x_nonoverlap'


def test_the_controller_sets_the_tool_geometry_and_keeps_what_it_is_not_given():
    from trainit_setup_assistant.gui.controller import AssistantController
    ctrl = AssistantController(_project(None))
    ctrl.set_tool_geometry(tool_radius_m=0.012)
    assert ctrl.project.robot.gripper.tool_radius_m == 0.012
    ctrl.set_tool_geometry(tool_reach_mm=8.0)
    assert ctrl.project.robot.gripper.tool_radius_m == 0.012     # kept
    assert ctrl.project.robot.gripper.tool_reach_mm == 8.0


def test_the_perception_step_saves_and_reloads_the_rule(qapp):
    """The rule is a field beside ``params``, so it has to survive the page's own
    save -> list -> re-pick -> push cycle."""
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController()
    ctrl.open_project(EXAMPLE)
    ctrl.project.scene.objects.append(
        SceneObject(id='capsule', shape='cylinder', dims=[0.0135, 0.04],
                    category='dynamic', grasp_target=True))
    wiz = SetupWizard(ctrl)
    pp = wiz.perception_page
    pp.det_name.setText('capsules')
    pp.p_rule.setCurrentIndex(pp.p_rule.findData(MASK_RULE.value))
    pp._add_detector()
    assert pp.validatePage()
    det = ctrl.project.perception.detector_by_name('capsules')
    assert det.candidate_rule is MASK_RULE
    # re-entering the page restores the selection instead of silently unsetting it
    pp._detectors.clear()
    pp.initializePage()
    pp._on_pick_detector('capsules')
    assert pp.p_rule.currentData() == MASK_RULE.value
    # and the derived radius lands in the generated file
    params = _entry(ctrl.project)['params']
    assert params['candidate_rule'] == MASK_RULE.value
    assert params['expected_radius_m'] == pytest.approx(0.0135)


def test_the_perception_step_greys_a_rule_this_detector_cannot_run(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import SetupWizard
    ctrl = AssistantController(_project(None))
    wiz = SetupWizard(ctrl)                 # held: a collected wizard deletes its widgets
    pp = wiz.perception_page
    pp.initializePage()
    model = pp.p_rule.model()
    enabled = {tok for i, (tok, _) in enumerate(pp.RULES)
               if model.item(i + 1).isEnabled()}
    assert 'score_x_visible' in enabled
    assert 'candidate_x_visible' not in enabled      # the colour mask runs no 3D fit
    assert 'score_x_clearance' not in enabled        # no tool numbers declared
    assert pp.p_rule.currentData() == ''             # nothing armed: still byte-identical


def test_the_end_effector_dialog_saves_the_two_tool_numbers(qapp):
    from trainit_setup_assistant.gui.controller import AssistantController
    from trainit_setup_assistant.gui.wizard import EndEffectorDialog
    ctrl = AssistantController(_project(None))
    dlg = EndEffectorDialog(ctrl, wiz=None)
    assert dlg.g_tool_r.value() == 0.0 and dlg.g_grip_w.value() == 0.0
    dlg.g_tool_r.setValue(0.012)
    dlg.g_grip_w.setValue(8.0)
    dlg.accept()
    assert ctrl.project.robot.gripper.tool_radius_m == pytest.approx(0.012)
    assert ctrl.project.robot.gripper.tool_reach_mm == pytest.approx(8.0)
    # reopening shows what was saved
    assert EndEffectorDialog(ctrl, wiz=None).g_tool_r.value() == pytest.approx(0.012)


# --- 8. the loop consumes the ORDER, so an unranked loop says so ---------------

def test_bin_picking_notes_a_loop_whose_detector_has_no_rule():
    """ADR-0016 Decision 2: the runtime retries candidates in the order it was GIVEN and
    never sorts, so without a rule "try the next one" is a shuffle, not a ranking. A note,
    not a refusal — the rule is a perception choice and an unset one still runs."""
    if not GOLDEN_PROJECT.is_file():
        pytest.skip('golden bundle not present')
    from trainit_setup_assistant.applications import get_application
    from trainit_setup_assistant.model import AppType
    p = load_project(GOLDEN_PROJECT)
    p.bundle = BundleSpec.from_prefix('cell')
    p.application.type = AppType.BIN_PICKING
    p.application.bin_picking.attached_topic = '/suction/attached'
    template = get_application(p.application.type)
    assert 'no candidate_rule' in '\n'.join(template.validate(p))
    # setting the rule silences it
    p.perception.detectors[0].candidate_rule = MASK_RULE
    assert 'no candidate_rule' not in '\n'.join(template.validate(p))


if __name__ == '__main__':
    raise SystemExit(pytest.main([__file__, '-v']))
