"""vNext M3.6 (ADR-0026, M3_DESIGN §10): the manipulation steps of a Canonical Task Program --
``detect``, ``grasp``, ``release``, ``tool`` -- and ``relative.frame: target_local``.

Validation C (test/data/m3_6/validation_c*.yaml, checkpoint M3-D) is the reference: ONE grasp step fed
by two producers (a grasp-candidate producer, a detector projected onto the same contract), a release at
a taught target, on the public UR5e sample cell with grasp and release confirmation.

  M  the strict models: each step's keys, forms and refusals, the hints for keys in the wrong place
  G  the generated idioms: the node sequence, the verification the contract requires (D4), the derived
     entries (bound at run time from a candidate, resolved by TSA at a taught target -- along its tool
     +Z, target_local), the two producers converging onto the same grasp step
  R  the refusals before any file (R-M3-02/03/05/13), each naming what, where, why and how to fix
  T  target_local on a relative waypoint
  F  the bundle: bt_params entries, the README, the TMR floor (data-gated, 2.5.0)
"""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (ProgramRefusals, plan_program,
                                                                 program_overview, program_tree_xml)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.runtime_requirements import requirement, use_of
from trainit_setup_assistant.model import CanonicalProject

DATA = Path(__file__).parent / 'data' / 'm3_6'
SOURCE = DATA / 'validation_c.yaml'
DETECTIONS = DATA / 'validation_c_detections.yaml'


@pytest.fixture(autouse=True)
def _in_the_data_dir(monkeypatch):
    monkeypatch.chdir(DATA)                              # the urdf path is relative to it


@pytest.fixture(autouse=True)
def _the_ur5e_sample_cell():
    """Every test here plans on the public UR5e sample cell: it SKIPS, naming ur_description, on a
    machine that does not have it yet (the fresh-machine check's suite before the sample; vNext M3.7)."""
    import test_example_ur5e_suction as S
    S.needs_ur_description()


def raw(source=SOURCE) -> dict:
    return yaml.safe_load(source.read_text())


def build(mutate=None, source=SOURCE) -> CanonicalProject:
    data = raw(source)
    if mutate is not None:
        mutate(data)
    return CanonicalProject.parse_obj(data)


def app(data):
    return data['application']


def steps(data):
    return app(data)['tasks'][0]['steps']


def grasp_of(data):
    return next(s['grasp'] for s in steps(data) if 'grasp' in s)


def problems(project) -> str:
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    return str(exc.value)


def load_error(mutate) -> str:
    with pytest.raises(ValueError) as exc:
        build(mutate)
    return str(exc.value)


def task_nodes(project) -> list:
    root = ET.fromstring(program_tree_xml(project, plan_program(project)).split('\n', 1)[1])
    task = next(t for t in root.iter('BehaviorTree') if t.get('ID').startswith('task_'))
    return [(el.tag, dict(el.attrib)) for el in task.iter() if el.tag not in ('BehaviorTree', 'Sequence')]


def tags(project) -> list:
    out = []
    for tag, a in task_nodes(project):
        if tag == 'MoveWaypoint':
            out.append(f"move {a['waypoint'].split('__')[1]}")
        elif tag == 'VerifyGrasp':
            out.append(f"verify {a['expect']}")
        else:
            out.append(tag)
    return out


# --- G: the idioms ----------------------------------------------------------------------------------

GRASP_AND_RELEASE = [
    'DetectCandidates', 'SetWaypointFromCandidate', 'move s2_pregrasp', 'move s2_grasp', 'CloseGripper',
    'verify held', 'move s2_retract',
    'move s3_prerelease', 'move s3_release', 'verify held', 'OpenGripper', 'verify released', 'move s3_retract']


def test_validation_c_plans_to_the_fixed_idioms():
    assert tags(build()) == GRASP_AND_RELEASE
    assert tags(build(source=DETECTIONS)) == GRASP_AND_RELEASE


def test_the_two_producers_converge_onto_the_same_grasp_step():
    """Only the detect step differs: the candidate contract is the one seam (ADR-0026)."""
    a = program_tree_xml(build(), plan_program(build())).splitlines()
    b = program_tree_xml(build(source=DETECTIONS), plan_program(build(source=DETECTIONS))).splitlines()
    differ = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and len(differ) == 1
    assert '<DetectCandidates source="candidates" detector="grasp_provider"' in differ[0][0]
    assert ('<DetectCandidates source="detections" detector="part_cam" target_frame="base_link" '
            'timeout_ms="3000" keep="1" out_key="parts" default_approach="0;0;-1"/>') in differ[0][1]
    pa, pb = plan_program(build()), plan_program(build(source=DETECTIONS))
    assert pa.entries == pb.entries


def test_the_detect_step_is_one_DetectCandidates_in_the_base_frame():
    tag, a = task_nodes(build())[0]
    assert tag == 'DetectCandidates'
    assert a == {'source': 'candidates', 'detector': 'grasp_provider', 'target_frame': 'base_link',
                 'timeout_ms': '3000', 'keep': '1', 'out_key': 'parts'}


def test_the_grasp_binds_its_entries_from_the_candidate_with_align_approach():
    _, a = task_nodes(build())[1]
    assert a == {'from': 'parts', 'index': '0', 'grasp': 'validation_c__s2_grasp',
                 'pregrasp': 'validation_c__s2_pregrasp', 'retract': 'validation_c__s2_retract',
                 'standoff_m': '0.05', 'retract_along_approach': '0.1', 'orientation': 'align_approach',
                 'orientation_from': 'pick_nominal', 'eef': 'main', 'site': 'main'}
    plan = plan_program(build())
    # WHERE only at run time: the entries carry the MOVE that reaches each (motion, provider, limits)
    assert plan.entries['validation_c__s2_pregrasp'] == {
        'type': 'tcp', 'motion': 'ptp', 'planner': 'pilz', 'profile': 'transit', 'speed': 50, 'accel': 50}
    assert plan.entries['validation_c__s2_grasp'] == {
        'type': 'tcp', 'motion': 'lin', 'profile': 'approach', 'speed': 20, 'accel': 20}
    assert plan.entries['validation_c__s2_retract'] == plan.entries['validation_c__s2_grasp']
    assert plan.bound_entries[:3] == ['validation_c__s2_pregrasp', 'validation_c__s2_grasp',
                                      'validation_c__s2_retract']


def test_a_release_at_a_taught_target_is_resolved_along_its_tool_z():
    plan = plan_program(build())
    pre, at, back = (plan.entries[f'validation_c__s3_{r}'] for r in ('prerelease', 'release', 'retract'))
    # place (0.3, -0.3, 0.15), tool down: its tool +Z is -Z, so 0.05 m back along it is UP
    assert pre['position'] == [0.3, -0.3, 0.2]
    assert at['position'] == [0.3, -0.3, 0.15]
    assert back['position'] == [0.3, -0.3, 0.23]           # retract dz 0.08: up the frame's +Z
    assert pre['orientation'] == at['orientation'] == back['orientation'] == [1.0, 0.0, 0.0, 0.0]
    assert (pre['motion'], at['motion'], back['motion']) == ('ptp', 'lin', 'lin')
    assert 'target_local' in plan.entry_notes['validation_c__s3_prerelease'][0]


# place tilted 30 deg about Y: q = Ry(30) * Rx(180), so its tool +Z is Ry(30) * (0, 0, -1)
TILTED_Z = (-math.sin(math.radians(30)), 0.0, -math.cos(math.radians(30)))


def _tilted_place(data):
    half = math.radians(30) / 2

    def qmul(a, b):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return [aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz]
    q = qmul([0.0, math.sin(half), 0.0, math.cos(half)], [1.0, 0.0, 0.0, 0.0])
    app(data)['waypoints'][1]['orientation'] = [round(v, 15) for v in q]
    return q


def test_a_taught_targets_pregrasp_follows_its_own_axes_not_the_base_axes():
    q = []
    project = build(lambda d: q.append(_tilted_place(d)))
    plan = plan_program(project)
    pre = plan.entries['validation_c__s3_prerelease']['position']
    want = [p - 0.05 * a for p, a in zip((0.3, -0.3, 0.15), TILTED_Z)]
    assert pre == pytest.approx(want, abs=1e-9)
    assert pre[0] > 0.3                                     # back along the TILTED tool axis, not straight up


@pytest.mark.parametrize('grasp_c, release_c, verifies', [
    (True, True, ['verify held', 'verify held', 'verify released']),
    (True, False, ['verify held', 'verify held']),
    (False, True, ['verify released']),
    (False, False, []),
])
def test_the_verification_is_exactly_what_the_contract_requires(grasp_c, release_c, verifies):
    def mutate(d):
        d['robot']['gripper']['contract']['verification'] = {
            'require_grasp_confirmation': grasp_c, 'require_release_confirmation': release_c}
    got = [t for t in tags(build(mutate)) if t.startswith('verify')]
    assert got == verifies


def test_a_grasp_at_a_taught_target_needs_no_detect_and_no_reference():
    def mutate(d):
        s = steps(d)
        s.pop(0)
        s[0] = {'grasp': {'at': 'place', 'standoff_m': 0.04, 'profile': 'transit', 'approach_profile': 'approach',
                          'retract': {'along_approach': 0.06}}}
    project = build(mutate)
    assert tags(project)[:5] == ['move s1_pregrasp', 'move s1_grasp', 'CloseGripper', 'verify held',
                                 'move s1_retract']
    plan = plan_program(project)
    assert plan.entries['validation_c__s1_pregrasp']['position'] == [0.3, -0.3, 0.19]
    assert plan.entries['validation_c__s1_retract']['position'] == [0.3, -0.3, 0.21]   # along the approach


def test_a_tool_step_is_one_end_effector_node():
    def mutate(d):
        steps(d).append({'tool': {'verify': {'expect': 'released'}}})
    assert task_nodes(build(mutate))[-1] == ('VerifyGrasp', {'eef': 'main', 'expect': 'released'})


# --- M: the strict models ------------------------------------------------------------------------------

@pytest.mark.parametrize('step, needle', [
    ({'detect': {'detector': 'grasp_provider', 'source': 'candidates', 'out': 'p', 'trigger': True}},
     "'trigger' -- a detect step for a grasp takes the producer's next FRESH result"),
    ({'detect': {'detector': 'grasp_provider', 'source': 'images', 'out': 'p'}}, "detect.source 'images'"),
    ({'detect': {'detector': 'part_cam', 'source': 'detections', 'out': 'p'}}, "needs 'approach: [x, y, z]'"),
    ({'detect': {'detector': 'part_cam', 'source': 'detections', 'out': 'p', 'approach': [0, 0, -2]}},
     'is not a unit vector'),
    ({'detect': {'detector': 'grasp_provider', 'source': 'candidates', 'out': 'p', 'approach': [0, 0, -1]}},
     'takes no approach'),
    ({'detect': {'detector': 'grasp_provider', 'source': 'candidates', 'out': 'p', 'standoff_m': 0.1}},
     'the standoff of a grasp belongs to the grasp step'),
    ({'detect': {'detector': 'grasp_provider', 'source': 'candidates', 'out': 'p', 'keep': 0}}, 'detect.keep'),
    ({'grasp': {'candidates': 'p', 'profile': 't', 'approach_profile': 'a', 'retract': {'dz': 0.1}}},
     "needs 'orientation_from: <waypoint>'"),
    ({'grasp': {'candidates': 'p', 'orientation': 'candidate', 'orientation_from': 'nominal', 'profile': 't',
                'approach_profile': 'a', 'retract': {'dz': 0.1}}}, "a candidate's TCP orientation is 'align_approach'"),
    ({'grasp': {'at': 'place', 'profile': 't', 'approach_profile': 'a', 'retract': {'dz': 0.1}}},
     "needs 'standoff_m'"),
    ({'grasp': {'at': 'place', 'orientation_from': 'nominal', 'standoff_m': 0.1, 'profile': 't', 'approach_profile': 'a',
                'retract': {'dz': 0.1}}}, 'takes its orientation and approach from the target itself'),
    ({'grasp': {'at': 'place', 'candidates': 'p', 'standoff_m': 0.1, 'profile': 't', 'approach_profile': 'a',
                'retract': {'dz': 0.1}}}, 'never both'),
    ({'grasp': {'at': 'place', 'standoff_m': 0.1, 'profile': 't', 'approach_profile': 'a',
                'retract': {'dz': 0.1, 'along_approach': 0.1}}}, 'a retract is exactly one of'),
    ({'grasp': {'at': 'place', 'standoff_m': 0.0, 'profile': 't', 'approach_profile': 'a', 'retract': {'dz': 0.1}}},
     'grasp.standoff_m must be > 0'),
    ({'grasp': {'at': 'place', 'standoff_m': 0.1, 'profile': 't', 'approach_profile': 'a', 'retract': {'dz': 0.1},
                'pregrasp': 'x'}}, 'the PreGrasp is derived, never taught'),
    ({'release': {'at': 'place', 'profile': 't', 'approach_profile': 'a', 'retract': {'dz': 0.1}}}, 'standoff_m'),
    ({'tool': {'actuate': {'actuator': 'a', 'target': 'b'}, 'verify': {'expect': 'held'}}}, 'exactly one of'),
    ({'tool': {'verify': {'expect': 'maybe'}}}, "expects 'held' or 'released'"),
    ({'tool': {'close': {}}}, 'a grip is opened and closed by the grasp and release steps'),
    ({'pick': {'at': 'place'}}, 'a pick is a composition'),
    ({'place': {'at': 'place'}}, "a place is a 'release' step"),
])
def test_the_models_are_strict_and_say_what_to_write(step, needle):
    def mutate(d):
        steps(d).append(step)
    error = load_error(mutate)
    assert 'R-M3-0' in error and needle in error, error


# --- R: refusals before any file ---------------------------------------------------------------------

def test_a_grasp_without_an_end_effector_contract_is_refused():
    def mutate(d):
        d['robot']['gripper'].pop('contract')
    text = problems(build(mutate))
    assert 'R-M3-13' in text and 'add robot.gripper.contract' in text


@pytest.mark.parametrize('change, needle', [
    (lambda g: g.update(eef='other'), "no end effector is named 'other'"),
    (lambda g: g.update(site='cup'), "has no grasp site 'cup'"),
    (lambda g: g.update(candidates='nothing'), "no detect step earlier in task 'validation_c' produces candidate set 'nothing'"),
    (lambda g: g.update(index=1), "its detect step keeps 1 candidate(s) (raise its keep)"),
    (lambda g: g.update(orientation_from='nowhere'), "no waypoint is named 'nowhere'"),
    (lambda g: g.update(profile='fast'), "profile 'fast' is not a motion profile"),
])
def test_a_grasp_names_what_exists(change, needle):
    text = problems(build(lambda d: change(grasp_of(d))))
    assert needle in text, text


def test_projected_candidates_need_the_steps_standoff():
    text = problems(build(lambda d: grasp_of(d).pop('standoff_m'), source=DETECTIONS))
    assert 'R-M3-13' in text and "give the grasp step's 'standoff_m'" in text
    # a producer may suggest its own: the step's standoff is optional there
    plan_program(build(lambda d: grasp_of(d).pop('standoff_m')))


def test_the_align_approach_reference_is_a_pose_in_the_base_frame():
    def joint(d):
        app(d)['waypoints'].append({'name': 'home', 'type': 'joint', 'named': 'home'})
        grasp_of(d)['orientation_from'] = 'home'
    assert "orientation_from 'home' must be a TCP pose in the robot's base frame" in problems(build(joint))

    def other_frame(d):
        app(d)['waypoints'][0]['frame'] = 'base'
    assert "orientation_from 'pick_nominal' must be a TCP pose in the robot's base frame 'base_link'" in \
        problems(build(other_frame))


def test_a_site_whose_tcp_is_not_the_tip_link_is_refused():
    def mutate(d):
        d['robot']['gripper']['eef_parent_link'] = 'tool0'
    text = problems(build(mutate))
    assert "grasp site 'main' holds the object at TCP 'tool0'" in text and 'UNSUPPORTED' in text


def test_a_detect_names_a_producer_of_the_bundle():
    def unknown(d):
        steps(d)[0]['detect']['detector'] = 'nobody'
    assert "no detector is named 'nobody'" in problems(build(unknown))

    def tp_node(d):
        d['perception']['detectors'][0] = {'name': 'grasp_provider', 'method': 'color_mask',
                                           'params': {'class_id': 'part'}}
    text = problems(build(tp_node))
    assert 'publishes DETECTIONS, not grasp candidates' in text and 'use source: detections' in text


def test_a_candidate_set_never_overwrites_a_waypoint():
    def mutate(d):
        steps(d)[0]['detect']['out'] = 'place'
        grasp_of(d)['candidates'] = 'place'
    assert "candidate set 'place' would overwrite waypoint 'place'" in problems(build(mutate))


def test_a_grip_is_never_actuated_by_a_tool_step():
    def grip(d):
        steps(d).append({'tool': {'actuate': {'actuator': 'vacuum', 'target': 'engage'}}})
    text = problems(build(grip))
    assert "'engage' is the engage target of grip actuator 'vacuum' (site 'main')" in text, text
    assert 'a grip is closed and opened by the grasp and release steps' in text

    def unknown(d):
        steps(d).append({'tool': {'actuate': {'actuator': 'axis', 'target': 'home'}}})
    assert "has no actuator 'axis'" in problems(build(unknown))

    def site(d):
        steps(d).append({'tool': {'verify': {'site': 'cup', 'expect': 'held'}}})
    assert "has no grasp site 'cup'" in problems(build(site))


# --- T: target_local ------------------------------------------------------------------------------

def test_target_local_offsets_along_the_referenced_targets_own_axes():
    def mutate(d):
        _tilted_place(d)
        app(d)['waypoints'].append({'name': 'above_place', 'relative': {'to': 'place', 'frame': 'target_local',
                                                                         'dz': -0.05}})
        app(d)['waypoints'].append({'name': 'above_base', 'relative': {'to': 'place', 'frame': 'base', 'dz': 0.05}})
    plan = plan_program(build(mutate))
    local = plan.targets['above_place']
    # dz -0.05 along the target's own +Z (TILTED_Z): 0.05 m back along its tool axis
    assert list(local.position) == pytest.approx([p - 0.05 * a for p, a in zip((0.3, -0.3, 0.15), TILTED_Z)],
                                                 abs=1e-9)
    assert 'local axes (target_local)' in local.derivation
    assert list(plan.targets['above_base'].position) == pytest.approx([0.3, -0.3, 0.2])   # the base axes differ
    assert local.orientation == plan.targets['place'].orientation


# --- F: the bundle ---------------------------------------------------------------------------------

def test_the_bundle_declares_tmr_2_5_0_and_documents_the_manipulation(tmp_path):
    out = tmp_path / 'ws'
    Orchestrator().generate(build(), out)
    app_dir = out / 'm3d_app'
    tree = (app_dir / 'bt_trees' / 'task_program.xml').read_text()
    params = (app_dir / 'config' / 'bt_params.yaml').read_text()
    use = use_of(trees=[tree], bt_params=[params])
    assert 'DetectCandidates' in use.capabilities and 'SetWaypointFromCandidate' in use.capabilities
    assert requirement(use).version == '2.5.0'
    package = (app_dir / 'package.xml').read_text()
    assert 'version_gte="2.5.0">trainit_motion_runtime<' in package
    entries = yaml.safe_load(params)['/**']['ros__parameters']['task_parameters']
    assert entries['validation_c__s2_pregrasp'] == {
        'type': 'tcp', 'motion': 'ptp', 'planner': 'pilz', 'profile': 'transit', 'speed': 50, 'accel': 50}
    assert entries['pick_nominal']['orientation'] == [1.0, 0.0, 0.0, 0.0]
    readme = (app_dir / 'README.md').read_text()
    assert '## Manipulation' in readme and 'align_approach' in readme and '`validation_c__s2_pregrasp`' in readme
    assert '| 2 | grasp | candidate 0 of \'parts\'' in readme


def test_a_task_program_without_manipulation_keeps_its_floor():
    use = use_of(trees=['<root><BehaviorTree ID="t"><Sequence><MoveWaypoint waypoint="w"/></Sequence>'
                        '</BehaviorTree></root>'])
    assert 'DetectCandidates' not in use.capabilities and 'SetWaypointFromCandidate' not in use.capabilities


def test_the_wizard_overview_reads_the_manipulation_steps():
    text = program_overview(build().application.program)
    assert '  1  detect candidates of grasp_provider -> parts (keep 1)' in text
    assert "  2  grasp candidate 0 of 'parts', profiles transit / approach" in text
    assert "  3  release at 'place', profiles transit / approach" in text


def test_a_program_with_manipulation_steps_round_trips(tmp_path):
    from trainit_setup_assistant.model import load_project, save_project
    for source in (SOURCE, DETECTIONS):
        p = load_project(source)
        f = save_project(p, tmp_path / f'again_{source.stem}.yaml')
        text = f.read_text()
        assert 'detect:' in text and 'grasp:' in text and 'release:' in text and 'orientation_from' in text
        q = CanonicalProject.parse_obj(yaml.safe_load(text))
        assert q.application.program == p.application.program


# --- review fixes (M3.6 review) -------------------------------------------------------------------

def test_a_refused_detect_step_is_reported_once_never_again_by_its_grasp():
    def mutate(d):
        steps(d)[0]['detect']['detector'] = 'nope'
    text = problems(build(mutate))
    assert "no detector is named 'nope'" in text
    assert 'no detect step earlier' not in text, text


@pytest.mark.parametrize('out, needle', [
    ('validation_c__s2_grasp', "would overwrite the entries TSA derives ('__' is reserved for them)"),
])
def test_a_candidate_set_never_overwrites_a_derived_entry(out, needle):
    def mutate(d):
        steps(d)[0]['detect']['out'] = out
        grasp_of(d)['candidates'] = out
    assert needle in problems(build(mutate))


def test_a_capture_and_a_candidate_set_never_share_a_name():
    def triggerable(d):
        d['perception']['detectors'][0]['external_node']['detect_service'] = True

    def capture_after(d):
        triggerable(d)
        steps(d).insert(1, {'process': {'kind': 'capture', 'detector': 'grasp_provider', 'out': 'parts'}})
    assert "capture key 'parts' would overwrite the candidate set 'parts'" in problems(build(capture_after))

    def capture_before(d):
        triggerable(d)
        steps(d).insert(0, {'process': {'kind': 'capture', 'detector': 'grasp_provider', 'out': 'parts'}})
    assert "candidate set 'parts' would overwrite the capture 'parts'" in problems(build(capture_before))


def test_the_common_case_needs_no_set_name():
    """detect without out, grasp without candidates: the grasp takes the latest detect's set."""
    def mutate(d):
        steps(d)[0]['detect'].pop('out')
        grasp_of(d).pop('candidates')
    project = build(mutate)
    nodes = task_nodes(project)
    assert nodes[0][1]['out_key'] == 'candidates' and nodes[1][1]['from'] == 'candidates'

    def latest(d):
        s = steps(d)
        s.insert(1, {'detect': {'detector': 'grasp_provider', 'source': 'candidates', 'out': 'other'}})
        grasp_of(d).pop('candidates')
    assert task_nodes(build(latest))[2][1]['from'] == 'other'

    def none(d):
        steps(d).pop(0)
        grasp_of(d).pop('candidates')
    assert 'add a detect step before the grasp' in problems(build(none))


def test_a_taught_target_in_another_frame_keeps_its_frame_and_its_axes():
    def mutate(d):
        app(d)['waypoints'][1]['frame'] = 'base'          # UR's base: 180 deg about Z from base_link
    plan = plan_program(build(mutate))
    for role in ('prerelease', 'release', 'retract'):
        assert plan.entries[f'validation_c__s3_{role}']['frame'] == 'base'
    assert plan.entries['validation_c__s3_prerelease']['position'] == [0.3, -0.3, 0.2]   # its own tool +Z


def test_the_readme_states_a_taught_targets_approach():
    tag_plan = plan_program(build())
    step = tag_plan.tasks['validation_c'][2]
    assert step.summary[0] == '`place` (taught; its approach is its tool +Z = [0, 0, -1])'
