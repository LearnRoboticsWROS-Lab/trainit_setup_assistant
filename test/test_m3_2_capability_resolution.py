"""Constraint and provider capability resolution (vNext M3.2; ADR-0025, M3_DESIGN.md §7-§8, §15).

The canonical rule, held here end to end on TSA's committed copy of TMR's registry:

    request -> required tokens -> provider (the request's, the application's, the registry
    default) -> eligible and required <= provides ? -> a Resolution, or a refusal before any file

    B  the strict request models: valid and boundary values, non-finite numbers, illegal percents,
       unknown keys, the reserved fields
    C  the derivation: every intent and every constraint field -> its exact required set
    D  the provider x capability matrix: every registry row against every token, and every row as
       the default of every request kind (accepted iff the expected required set <= provides)
    E  the three documented equivalences: speed %, absolute TCP limits, continuity: stop
    F  one focused test per refusal rule this phase owns (R-M3-02, 04, 06, 07, 08, 10, 11, 12)
       and the diagnostic every capability refusal carries

Pure: no ROS, no generation, no file outside the test's tmp_path.
"""
import copy
import math
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant import catalog
from trainit_setup_assistant.catalog import (TSA_CAPABILITY_TOKENS, capability_vocabulary,
                                             declared_providers, tmr_planner_registry)
from trainit_setup_assistant.model import (GoalTolerance, MotionConstraints, MotionIntent,
                                           MotionProfile, PathContinuity, PlannerId,
                                           ProviderChoices, ProviderId, load_project)
from trainit_setup_assistant.model import capabilities as cap
from trainit_setup_assistant.model.capabilities import (CapabilityRefusal, CapabilityRefusals,
                                                        CartesianLimits, MotionRequest,
                                                        SegmentRequest, check_blend_geometry,
                                                        check_provider_choices,
                                                        missing_capabilities,
                                                        project_cartesian_limits,
                                                        required_capabilities, resolve,
                                                        resolve_all)

SAMPLE = Path(__file__).resolve().parents[1] / 'examples' / 'ur5e_suction' / 'project.yaml'
LIMITS = CartesianLimits(max_trans_vel=1.0, max_trans_acc=2.25,
                         source='robot.pilz_cartesian_limits')
REGISTRY = tmr_planner_registry()
ROWS = declared_providers(REGISTRY)
VOCAB = capability_vocabulary(REGISTRY)
SCALING = {'velocity_scaling', 'acceleration_scaling'}
PTP, FREE, LIN, CIRC, PATH = (MotionIntent.PTP, MotionIntent.FREE, MotionIntent.LIN,
                              MotionIntent.CIRC, MotionIntent.PATH)
BLEND, STOP = PathContinuity.BLEND, PathContinuity.STOP


def c(**kw):
    return MotionConstraints(**kw)


def seg(kind, **kw):
    return SegmentRequest(MotionIntent(kind), MotionConstraints(**kw))


def refused(request, rule, *says, **kw):
    """The refusal of ``request``: its rule, and every ``says`` in its text."""
    with pytest.raises(CapabilityRefusal) as exc:
        resolve(request, limits=kw.pop('limits', LIMITS), **kw)
    assert exc.value.rule == rule, str(exc.value)
    for text in says:
        assert text in str(exc.value), str(exc.value)
    return exc.value


# --- B. the strict request models ----------------------------------------------------------------

@pytest.mark.parametrize('kw', [
    dict(speed=1), dict(speed=100), dict(acceleration=1), dict(acceleration=100),
    dict(tcp_velocity_mps=1e-6), dict(tcp_acceleration_mps2=2.25), dict(blend_radius_m=0),
    dict(blend_radius_m=0.0), dict(blend_radius_m=0.01), dict(planning_time_s=0.5),
    dict(planning_attempts=1), dict(goal_tolerance={'position_m': 1e-4}),
    dict(goal_tolerance={'position_m': 0.001, 'orientation_rad': 0.01, 'joint_rad': 0.01}),
    dict(tcp_velocity_mps=1),                                  # a YAML int is a number
], ids=lambda kw: '-'.join(f'{k}={v}' for k, v in kw.items()))
def test_valid_and_boundary_values_are_accepted_as_given(kw):
    model = MotionConstraints(**kw)
    for key, value in kw.items():
        got = getattr(model, key)
        assert (got.dict(exclude_none=True) if isinstance(got, GoalTolerance) else got) == value


@pytest.mark.parametrize('kw, says', [
    (dict(speed=0), 'R-M3-07: speed is a percent in 1..100, got 0'),
    (dict(speed=101), 'R-M3-07: speed is a percent in 1..100, got 101'),
    (dict(acceleration=-5), 'R-M3-07: acceleration is a percent in 1..100'),
    (dict(speed=50.5), 'R-M3-02: speed must be a whole number, got 50.5'),
    (dict(speed=True), 'R-M3-02: speed must be a number, got True'),
    (dict(speed='50'), "R-M3-02: speed must be a number, got '50'"),
    (dict(tcp_velocity_mps=float('nan')), 'R-M3-02: tcp_velocity_mps must be a finite number'),
    (dict(tcp_acceleration_mps2=float('inf')), 'R-M3-02: tcp_acceleration_mps2 must be a finite'),
    (dict(planning_time_s=float('-inf')), 'R-M3-02: planning_time_s must be a finite number'),
    (dict(tcp_velocity_mps=0), 'R-M3-07: tcp_velocity_mps must be > 0, got 0'),
    (dict(tcp_acceleration_mps2=-0.1), 'R-M3-07: tcp_acceleration_mps2 must be > 0'),
    (dict(blend_radius_m=-0.001), 'R-M3-07: blend_radius_m must be >= 0'),
    (dict(blend_radius_m=float('nan')), 'R-M3-02: blend_radius_m must be a finite number'),
    (dict(planning_time_s=0), 'R-M3-07: planning_time_s must be > 0'),
    (dict(planning_attempts=0), 'R-M3-07: planning_attempts must be >= 1'),
    (dict(planning_attempts=2.0), 'R-M3-02: planning_attempts must be a whole number'),
    (dict(goal_tolerance={}), 'R-M3-07: goal_tolerance sets none of'),
    (dict(goal_tolerance={'position_m': 0}), 'R-M3-07: goal_tolerance.position_m must be > 0'),
    (dict(goal_tolerance={'joint_rad': float('nan')}),
     'R-M3-02: goal_tolerance.joint_rad must be a finite number'),
    (dict(goal_tolerance={'position_m': 0.001, 'yaw_rad': 0.1}), 'yaw_rad'),
    (dict(velocity_scaling=0.5), 'velocity_scaling'),          # TMR's word, not a request field
    (dict(profile_shape='trapezoidal'), 'profile_shape'),       # the app never names a shape
], ids=lambda x: x if isinstance(x, str) else '-'.join(map(str, x)))
def test_an_invalid_value_or_an_unknown_key_is_refused_by_name(kw, says):
    with pytest.raises(ValueError) as exc:
        MotionConstraints(**kw)
    assert says in str(exc.value), str(exc.value)


def test_unknown_keys_are_refused_in_every_new_model_never_dropped():
    for model, kw in ((MotionConstraints, dict(sped=50)), (MotionProfile, dict(name='x')),
                      (GoalTolerance, dict(position=0.1)), (ProviderChoices, dict(lin='pilz_lin'))):
        with pytest.raises(ValueError, match='[Ee]xtra'):
            model(**kw)


def test_the_reserved_fields_are_held_so_resolution_refuses_them_by_name():
    model = MotionConstraints(jerk=50, orientation_constraint={'axis': 'z'}, duration_s=2.0)
    assert model.requested() == ('jerk', 'orientation_constraint', 'duration_s')


def test_a_profile_is_a_constraint_set_without_a_segment_blend():
    assert MotionProfile(speed=80, acceleration=50).requested() == ('speed', 'acceleration')
    with pytest.raises(ValueError, match='R-M3-02: blend_radius_m belongs to a path segment'):
        MotionProfile(blend_radius_m=0.01)
    with pytest.raises(ValueError, match='R-M3-07'):
        MotionProfile(speed=0)                                   # the constraints' own checks


def test_merging_is_innermost_wins_field_by_field():
    profile = MotionProfile(speed=20, acceleration=20,
                            goal_tolerance={'position_m': 0.001, 'orientation_rad': 0.02})
    inner = c(speed=5, blend_radius_m=0.01, goal_tolerance={'orientation_rad': 0.01})
    merged = profile.merged(inner)
    assert (merged.speed, merged.acceleration, merged.blend_radius_m) == (5, 20, 0.01)
    assert (merged.goal_tolerance.position_m, merged.goal_tolerance.orientation_rad) == \
        (0.001, 0.01)
    assert profile.merged(None) == MotionConstraints(**profile.dict(exclude_none=True))
    assert type(merged) is MotionConstraints


def test_provider_choices_take_registry_rows_only_and_no_lin_or_circ():
    choices = ProviderChoices(ptp='pilz_ptp', free='ompl_chomp', path='pilz_sequence',
                              dense_path='cartesian_waypoints')
    assert choices.free is ProviderId.OMPL_CHOMP
    with pytest.raises(ValueError, match=r"R-M3-06: providers.path: provider 'spline' is not in "
                                         r"TMR's registry snapshot"):
        ProviderChoices(path='spline')
    with pytest.raises(ValueError, match='R-M3-06'):
        ProviderChoices(free='pilz')                        # a planner MODE is not a provider id


def test_provider_ids_are_the_registry_rows_and_planner_modes_are_unchanged():
    """Two vocabularies, kept apart (M3_DESIGN.md §5, §8): the legacy PlannerId is untouched."""
    assert [p.value for p in ProviderId] == list(ROWS)
    assert [m.value for m in PlannerId] == ['pilz', 'ompl', 'ompl_chomp']


# --- C. the derivation: request -> exact required set --------------------------------------------

def _required(request):
    return set(required_capabilities(request))


@pytest.mark.parametrize('request_, expected', [
    (MotionRequest('m', PTP), {'joint_ptp'}),
    (MotionRequest('m', FREE), {'collision_free_planning'}),
    (MotionRequest('m', LIN), {'cartesian_linear'}),
    (MotionRequest('m', CIRC), {'cartesian_circular'}),
    (MotionRequest('p', PATH, segments=(seg('lin'), seg('lin'))),
     {'multi_segment_path', 'cartesian_linear'}),
    (MotionRequest('p', PATH, segments=(seg('lin'), seg('circ'), seg('lin')), continuity=STOP),
     {'multi_segment_path', 'cartesian_linear', 'cartesian_circular'}),
    (MotionRequest('p', PATH, segments=(seg('lin', blend_radius_m=0.01), seg('lin')),
                   continuity=BLEND),
     {'multi_segment_path', 'cartesian_linear', 'blending'}),
    (MotionRequest('p', PATH, dense=True, continuity=BLEND), {'continuous_path'}),
    (MotionRequest('p', PATH, dense=True), {'multi_segment_path', 'cartesian_linear'}),
    (MotionRequest('p', PATH, dense=True, continuity=STOP),
     {'multi_segment_path', 'cartesian_linear'}),
], ids=['ptp', 'free', 'lin', 'circ', 'path_lin', 'path_stop_lin_circ', 'path_blend',
        'dense_blend', 'dense_default', 'dense_stop'])
def test_every_intent_and_shape_requires_exactly_its_tokens(request_, expected):
    assert _required(request_) == expected | SCALING


@pytest.mark.parametrize('field, value, adds', [
    ('speed', 20, set()),                                   # velocity_scaling: always required
    ('acceleration', 20, set()),
    ('tcp_velocity_mps', 0.1, {'tcp_limits'}),
    ('tcp_acceleration_mps2', 0.5, {'tcp_limits'}),
    ('blend_radius_m', 0.01, {'blending', 'multi_segment_path'}),
    ('blend_radius_m', 0.0, set()),                         # a zero blend is no blend
    ('goal_tolerance', {'position_m': 0.001}, {'goal_tolerance'}),
    ('planning_time_s', 2.0, {'collision_free_planning'}),
    ('planning_attempts', 3, {'collision_free_planning'}),
    ('jerk', 10.0, {'jerk_constraints'}),
    ('orientation_constraint', {'axis': 'z'}, {'orientation_constraints'}),
], ids=lambda x: str(x))
def test_every_constraint_field_adds_exactly_its_tokens(field, value, adds):
    required = required_capabilities(MotionRequest('m', LIN, c(**{field: value})))
    assert set(required) == {'cartesian_linear'} | SCALING | adds
    for token in adds:
        assert field in required[token]                      # the reason names the field


def test_the_derivation_is_one_table_naming_only_registry_tokens():
    named = {t for tokens in cap.INTENT_TOKENS.values() for t in tokens}
    named |= set(cap.SEGMENT_TOKENS.values()) | set(cap.ALWAYS_TOKENS)
    named |= {t for tokens in cap.CONSTRAINT_TOKENS.values() for t in tokens}
    named |= {t for tokens in cap.KIND_TOKENS.values() for t in tokens}
    named |= {cap.PATH_TOKEN, cap.BLEND_TOKEN, cap.CONTINUOUS_TOKEN}       # a path's shape
    assert named == set(TSA_CAPABILITY_TOKENS)
    assert named <= set(VOCAB['live']) | set(VOCAB['reserved'])
    from trainit_setup_assistant.model.motion import (CONSTRAINT_FIELDS,
                                                      RESERVED_CONSTRAINT_FIELDS)
    assert set(cap.CONSTRAINT_TOKENS) == set(CONSTRAINT_FIELDS + RESERVED_CONSTRAINT_FIELDS)


def test_a_paths_derivation_is_tmrs_and_a_segment_constraint_counts():
    """TMR's requiredCapabilities (path_library.cpp): a path = multi_segment_path + its segments'
    shapes + blending iff continuity blend + the two scalings + goal_tolerance iff set."""
    request = MotionRequest('p', PATH, c(goal_tolerance={'position_m': 0.001,
                                                         'orientation_rad': 0.01}),
                            segments=(seg('circ', tcp_velocity_mps=0.1, blend_radius_m=0.01),
                                      seg('lin')), continuity=BLEND)
    required = required_capabilities(request)
    assert list(required) == ['cartesian_linear', 'cartesian_circular', 'multi_segment_path',
                              'blending', 'velocity_scaling', 'acceleration_scaling',
                              'tcp_limits', 'goal_tolerance']          # the registry's order
    assert required['tcp_limits'] == ['segment 0 tcp_velocity_mps']
    assert required['blending'] == ['continuity blend']                # not the radius itself


# --- D. the provider x capability matrix ---------------------------------------------------------

@pytest.mark.parametrize('provider', list(ROWS))
@pytest.mark.parametrize('token', VOCAB['live'] + VOCAB['reserved'])
def test_every_row_against_every_token(provider, token):
    """required <= provides, one token at a time: compatible iff the row provides it."""
    provides = ROWS[provider]['provides']
    assert (missing_capabilities(provider, [token]) == []) is (token in provides)
    assert missing_capabilities(provider, provides) == []


# the request of each kind, and what it requires (the design's table, stated independently)
KIND_REQUESTS = {
    'ptp': (MotionRequest('m', PTP), {'joint_ptp'}),
    'free': (MotionRequest('m', FREE), {'collision_free_planning'}),
    'lin': (MotionRequest('m', LIN), {'cartesian_linear'}),
    'circ': (MotionRequest('m', CIRC), {'cartesian_circular'}),
    'path': (MotionRequest('p', PATH, segments=(seg('lin', blend_radius_m=0.01), seg('circ')),
                           continuity=BLEND),
             {'multi_segment_path', 'cartesian_linear', 'cartesian_circular', 'blending'}),
    'dense_path': (MotionRequest('p', PATH, dense=True, continuity=BLEND), {'continuous_path'}),
}


def _with_default(kind, provider):
    registry = copy.deepcopy(REGISTRY)
    registry['provider_defaults'][kind] = provider
    return registry


@pytest.mark.parametrize('provider', list(ROWS))
@pytest.mark.parametrize('kind', list(KIND_REQUESTS))
def test_every_row_as_the_default_of_every_kind(kind, provider):
    """Default resolution validates the default like any choice: accepted iff it provides what
    the kind requires; never a search for another row."""
    request, base = KIND_REQUESTS[kind]
    registry = _with_default(kind, provider)
    ok = (base | SCALING) <= set(ROWS[provider]['provides'])
    if ok:
        result = resolve(request, limits=LIMITS, registry=registry)
        assert (result.provider, result.provider_source) == (provider, 'registry_default')
    else:
        with pytest.raises(CapabilityRefusal) as exc:
            resolve(request, limits=LIMITS, registry=registry)
        assert exc.value.rule in ('R-M3-10', 'R-M3-12') and exc.value.provider == provider


EXPECTED_DEFAULTS = {'ptp': 'pilz_ptp', 'free': 'ompl', 'lin': 'pilz_lin', 'circ': 'pilz_circ',
                     'path': 'pilz_sequence', 'dense_path': 'cartesian_waypoints'}


@pytest.mark.parametrize('kind', list(KIND_REQUESTS))
def test_default_resolution_is_the_registrys_and_deterministic(kind):
    request, _ = KIND_REQUESTS[kind]
    first = resolve(request, limits=LIMITS)
    assert (first.provider, first.provider_source, first.kind) == \
        (EXPECTED_DEFAULTS[kind], 'registry_default', kind)
    assert all(resolve(request, limits=LIMITS) == first for _ in range(3))


def test_a_dense_stop_path_defaults_to_the_path_row_as_tmr_does():
    result = resolve(MotionRequest('p', PATH, dense=True, continuity=STOP), limits=LIMITS)
    assert (result.kind, result.provider) == ('path', 'pilz_sequence')


@pytest.mark.parametrize('provider', list(ROWS))
def test_an_explicit_path_provider_is_accepted_iff_it_provides_the_path(provider):
    request = MotionRequest('p', PATH, provider=provider,
                            segments=(seg('lin', blend_radius_m=0.01), seg('lin')),
                            continuity=BLEND)
    required = {'multi_segment_path', 'cartesian_linear', 'blending'} | SCALING
    if required <= set(ROWS[provider]['provides']):
        assert resolve(request, limits=LIMITS).provider_source == 'request'
    else:
        refused(request, 'R-M3-10', f"provider '{provider}' does not provide")


@pytest.mark.parametrize('intent, provider, ok', [
    (PTP, 'pilz_ptp', True), (PTP, 'ompl', False), (FREE, 'ompl', True),
    (FREE, 'ompl_chomp', True), (FREE, 'pilz_ptp', False),
])
def test_an_explicit_move_planner_must_provide_its_motion(intent, provider, ok):
    request = MotionRequest('m', intent, provider=provider)
    if ok:
        assert resolve(request).provider == provider
    else:
        refused(request, 'R-M3-10', f"provider '{provider}' does not provide")


def test_the_application_choice_comes_between_the_request_and_the_default():
    choices = ProviderChoices(free='ompl_chomp')
    assert resolve(MotionRequest('m', FREE), choices).provider == 'ompl_chomp'
    assert resolve(MotionRequest('m', FREE), choices).provider_source == 'application'
    request = MotionRequest('m', FREE, provider='ompl')
    assert (resolve(request, choices).provider, resolve(request, choices).provider_source) == \
        ('ompl', 'request')
    assert resolve(MotionRequest('m', PTP), choices).provider_source == 'registry_default'


def test_resolution_reads_the_committed_copy_not_the_environment(monkeypatch):
    """No installed package, no importability, no environment variable enters the result."""
    before = resolve(KIND_REQUESTS['path'][0], limits=LIMITS)
    monkeypatch.setenv('AMENT_PREFIX_PATH', '/nonexistent')
    monkeypatch.setitem(__import__('sys').modules, 'trainit_motion_runtime', None)
    assert resolve(KIND_REQUESTS['path'][0], limits=LIMITS) == before


def test_check_provider_choices_checks_every_choice_even_unused():
    check_provider_choices(ProviderChoices(ptp='pilz_ptp', free='ompl', path='pilz_sequence',
                                           dense_path='cartesian_waypoints'))
    with pytest.raises(CapabilityRefusal) as exc:
        check_provider_choices(ProviderChoices(path='cartesian_waypoints'))
    assert exc.value.rule == 'R-M3-10' and 'multi_segment_path' in str(exc.value)
    with pytest.raises(CapabilityRefusal, match="R-M3-10 .*providers.free.*'pilz_ptp' cannot "
                                                'serve a free step'):
        check_provider_choices(ProviderChoices(free='pilz_ptp'))
    registry = copy.deepcopy(REGISTRY)
    registry['providers']['ompl_chomp']['stability'] = 'experimental'
    with pytest.raises(CapabilityRefusal, match='R-M3-11'):
        check_provider_choices(ProviderChoices(free='ompl_chomp'), registry)


# --- E. the documented equivalences --------------------------------------------------------------

def _eq(result):
    """equivalence id -> its derivations (one per axis or segment that used it)"""
    out = {}
    for key, text in result.equivalences:
        out.setdefault(key, []).append(text)
    return {k: ' | '.join(v) for k, v in out.items()}


def test_speed_percent_is_velocity_scaling():
    result = resolve(MotionRequest('m', PTP, c(speed=20, acceleration=50)), limits=None)
    assert (result.velocity_scaling.value, result.velocity_scaling.source) == (0.2, 'speed')
    assert (result.acceleration_scaling.value, result.acceleration_scaling.source) == \
        (0.5, 'acceleration')
    assert 'speed 20 % = scaling 0.2' in _eq(result)[cap.EQ_SPEED_PERCENT]
    unset = resolve(MotionRequest('m', PTP))
    assert unset.velocity_scaling is None and unset.equivalences == ()     # the provider default


def test_an_absolute_tcp_limit_is_scaling_of_the_configured_limit():
    limits = CartesianLimits(0.5, 2.0, 'robot.pilz_cartesian_limits')
    result = resolve(MotionRequest('m', LIN, c(tcp_velocity_mps=0.1, tcp_acceleration_mps2=0.5)),
                     limits=limits)
    assert result.velocity_scaling.value == pytest.approx(0.2)
    assert result.acceleration_scaling.value == pytest.approx(0.25)
    assert result.velocity_scaling.source == 'tcp_velocity_mps'
    assert ('tcp_velocity_mps 0.1 / robot.pilz_cartesian_limits.max_trans_vel 0.5 = scaling 0.2'
            == result.velocity_scaling.derivation)
    at_limit = resolve(MotionRequest('m', LIN, c(tcp_velocity_mps=0.5)), limits=limits)
    assert at_limit.velocity_scaling.value == 1.0                  # the limit itself: accepted


def test_a_percent_and_an_absolute_limit_together_honour_both():
    """Both are LIMITS: the effective scaling is the smaller, and the result says which."""
    tight = resolve(MotionRequest('m', LIN, c(speed=20, tcp_velocity_mps=0.1)), limits=LIMITS)
    assert (tight.velocity_scaling.value, tight.velocity_scaling.source) == \
        (0.1, 'tcp_velocity_mps')
    assert 'the smaller of two limits' in tight.velocity_scaling.derivation
    loose = resolve(MotionRequest('m', LIN, c(speed=5, tcp_velocity_mps=0.1)), limits=LIMITS)
    assert (loose.velocity_scaling.value, loose.velocity_scaling.source) == (0.05, 'speed')
    assert set(_eq(tight)) == {cap.EQ_SPEED_PERCENT, cap.EQ_TCP_LIMITS}


def test_continuity_stop_is_a_sequence_of_single_motions():
    result = resolve(MotionRequest('p', PATH, segments=(seg('lin'), seg('circ'), seg('lin')),
                                   continuity=STOP), limits=LIMITS)
    assert [s.blend_radius_m for s in result.segments] == [0.0, 0.0, 0.0]
    assert 'equivalent to a sequence of single motions' in _eq(result)[cap.EQ_CONTINUITY_STOP]
    assert 'blending' not in result.required
    implicit = resolve(MotionRequest('p', PATH, segments=(seg('lin'), seg('lin'))))
    assert cap.EQ_CONTINUITY_STOP in _eq(implicit)                       # stop is the default
    blended = resolve(KIND_REQUESTS['path'][0])
    assert [s.blend_radius_m for s in blended.segments] == [0.01, 0.0]
    assert cap.EQ_CONTINUITY_STOP not in _eq(blended)


def test_a_segment_scaling_is_its_own_over_the_paths():
    result = resolve(MotionRequest('p', PATH, c(speed=50),
                                   segments=(seg('lin', speed=10), seg('lin',
                                                                       tcp_velocity_mps=0.2))),
                     limits=LIMITS)
    assert [s.velocity_scaling.value for s in result.segments] == [0.1, pytest.approx(0.2)]
    assert result.velocity_scaling.value == 0.5


def test_project_cartesian_limits_read_the_canonical_source(tmp_path):
    project = load_project(str(SAMPLE))               # the public sample: from-scratch flow
    assert not project.robot.base_moveit_config_path
    project.robot.pilz_cartesian_limits.max_trans_vel = 0.7
    limits = project_cartesian_limits(project)
    assert (limits.max_trans_vel, limits.max_trans_acc, limits.source) == \
        (0.7, 2.25, 'robot.pilz_cartesian_limits')
    base = tmp_path / 'base_moveit_config'
    (base / 'config').mkdir(parents=True)
    (base / 'config' / 'pilz_cartesian_limits.yaml').write_text(yaml.safe_dump(
        {'cartesian_limits': {'max_trans_vel': 0.3, 'max_trans_acc': 1.0, 'max_trans_dec': -1.0,
                              'max_rot_vel': 1.0}}))
    project.robot.base_moveit_config_path = str(base)
    limits = project_cartesian_limits(project)
    assert (limits.max_trans_vel, limits.max_trans_acc) == (0.3, 1.0)
    assert limits.source.endswith('pilz_cartesian_limits.yaml')
    project.robot.base_moveit_config_path = str(tmp_path / 'missing')
    with pytest.raises(CapabilityRefusal, match='R-M3-07 .*the TCP limits cannot be read'):
        project_cartesian_limits(project)


@pytest.mark.parametrize('bad', [0.0, -1.0, float('nan'), float('inf'), True, '1.0'])
def test_an_invalid_configured_limit_is_refused_not_used(bad):
    limits = CartesianLimits(bad, 2.25, 'robot.pilz_cartesian_limits')
    refused(MotionRequest('m', LIN, c(tcp_velocity_mps=0.1)), 'R-M3-07',
            'robot.pilz_cartesian_limits.max_trans_vel is', 'a finite limit > 0', limits=limits)
    assert resolve(MotionRequest('m', LIN, c(speed=10)), limits=limits)     # unused: not read


# --- F. one focused test per refusal rule --------------------------------------------------------

def test_r_m3_04_a_path_without_segments_or_a_move_with_path_fields():
    refused(MotionRequest('p', PATH), 'R-M3-04', 'at least 2 states')
    refused(MotionRequest('p', PATH, dense=True, segments=(seg('lin'),)), 'R-M3-04',
            'a dense path has no segments')
    refused(MotionRequest('m', LIN, continuity=BLEND), 'R-M3-04', 'belong to a path')


def test_r_m3_06_unknown_provider_planner_on_lin_circ_and_ptp_segment():
    refused(MotionRequest('m', FREE, provider='rrt_star'), 'R-M3-06',
            "provider 'rrt_star' is not in TMR's registry snapshot")
    refused(MotionRequest('m', LIN, provider='pilz_lin'), 'R-M3-06', 'a planner on a lin move')
    refused(MotionRequest('m', CIRC, provider='ompl'), 'R-M3-06', 'a planner on a circ move')
    refused(MotionRequest('p', PATH, segments=(seg('ptp'), seg('lin'))), 'R-M3-06',
            'segment 0 is ptp', 'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL')


def test_r_m3_07_a_tcp_limit_above_the_configuration_is_refused_not_clamped():
    error = refused(MotionRequest('task t step 3', LIN, c(tcp_velocity_mps=1.5)), 'R-M3-07',
                    'tcp_velocity_mps 1.5 exceeds the configured '
                    'robot.pilz_cartesian_limits.max_trans_vel 1', 'refused, not clamped')
    assert error.intent == 'lin' and error.where == 'task t step 3'
    refused(MotionRequest('m', CIRC, c(tcp_acceleration_mps2=3.0)), 'R-M3-07',
            'tcp_acceleration_mps2 3 exceeds', 'max_trans_acc 2.25')
    refused(MotionRequest('p', PATH, segments=(seg('lin', tcp_velocity_mps=2.0), seg('lin'))),
            'R-M3-07', 'segment 0', 'tcp_velocity_mps 2 exceeds')
    refused(MotionRequest('m', LIN, c(tcp_velocity_mps=0.1)), 'R-M3-07', 'none was given',
            limits=None)


def test_r_m3_07_a_path_goal_tolerance_is_cartesian():
    path = (seg('lin'), seg('lin'))
    refused(MotionRequest('p', PATH, c(goal_tolerance={'position_m': 0.001}), segments=path),
            'R-M3-07', 'needs position_m and orientation_rad')
    refused(MotionRequest('p', PATH, c(goal_tolerance={'position_m': 0.001,
                                                       'orientation_rad': 0.01,
                                                       'joint_rad': 0.01}), segments=path),
            'R-M3-07', 'no joint_rad')
    ok = resolve(MotionRequest('p', PATH, c(goal_tolerance={'position_m': 0.001,
                                                            'orientation_rad': 0.01}),
                               segments=path))
    assert 'goal_tolerance' in ok.required and ok.provider == 'pilz_sequence'


def test_r_m3_08_blend_structure_and_geometry():
    refused(MotionRequest('p', PATH, segments=(seg('lin'), seg('lin', blend_radius_m=0.01)),
                          continuity=BLEND), 'R-M3-08', 'the last segment blends',
            'a path ends at rest')
    refused(MotionRequest('p', PATH, segments=(seg('lin', blend_radius_m=0.01), seg('lin'))),
            'R-M3-08', 'continuity: stop path')
    refused(MotionRequest('p', PATH, segments=(seg('lin'), seg('lin')), continuity=BLEND),
            'R-M3-08', 'with no blend_radius_m > 0')
    refused(MotionRequest('p', PATH, c(blend_radius_m=0.01), segments=(seg('lin'), seg('lin'))),
            'R-M3-08', "blend_radius_m is a segment's")
    states = [(0, 0, 0), (0.1, 0, 0), (0.1, 0.02, 0)]          # segments of 0.1 m and 0.02 m
    check_blend_geometry('p', states, [0.01, 0.0])               # half of 0.02: accepted
    with pytest.raises(CapabilityRefusal, match=r'R-M3-08 .*segment 0: blend_radius_m 0.011 '
                                                r'exceeds half of an adjacent segment'):
        check_blend_geometry('p', states, [0.011, 0.0])


def test_r_m3_10_the_brief_example_lin_blend_on_cartesian_waypoints():
    """REQUEST lin + blend, PROVIDER cartesian_waypoints -> missing blending (the brief's
    example): the refusal names the request, the intent, the required and provided sets, the
    missing tokens and where each requirement comes from."""
    request = MotionRequest("task 'weld' step 4 (path 'seam')", PATH,
                            provider='cartesian_waypoints',
                            segments=(seg('lin', blend_radius_m=0.005), seg('lin')),
                            continuity=BLEND)
    error = refused(request, 'R-M3-10', "provider 'cartesian_waypoints' does not provide")
    assert error.where == "task 'weld' step 4 (path 'seam')" and error.intent == 'path'
    assert error.provider == 'cartesian_waypoints' and error.provider_source == 'request'
    assert set(error.required) == {'multi_segment_path', 'cartesian_linear', 'blending'} | SCALING
    assert error.provided == ('continuous_path', 'velocity_scaling', 'acceleration_scaling')
    assert error.missing == ('cartesian_linear', 'multi_segment_path', 'blending')
    text = str(error)
    for line in ("R-M3-10 REFUSED: task 'weld' step 4 (path 'seam') (path)",
                 '  required: ', '  provider: cartesian_waypoints (request)',
                 '  provides: continuous_path, velocity_scaling, acceleration_scaling',
                 '  missing:  cartesian_linear, multi_segment_path, blending',
                 '  because:  blending <- continuity blend (provided by pilz_sequence)'):
        assert line in text, text


def test_r_m3_10_a_reserved_constraint_is_refused_by_its_token():
    for field, token in (('jerk', 'jerk_constraints'),
                         ('orientation_constraint', 'orientation_constraints')):
        error = refused(MotionRequest('m', LIN, c(**{field: 1})), 'R-M3-10', repr(token),
                        f'{token} <- {field} (reserved: no provider provides it)')
        assert error.missing == (token,)


def test_r_m3_12_a_lin_move_cannot_blend_a_blend_needs_a_path():
    """A lin move chooses no provider (R-M3-06): its only candidate is the default, so a field
    that row lacks is R-M3-12, and the hints name no row the move could not choose."""
    error = refused(MotionRequest('m', LIN, c(blend_radius_m=0.01)), 'R-M3-12',
                    'no lin provider can honour blend_radius_m',
                    'multi_segment_path <- blend_radius_m (no lin provider provides it)')
    assert 'pilz_sequence' not in str(error)


def test_r_m3_12_goal_tolerance_is_refused_where_the_route_ignores_it():
    """ADR-0025 Decision 3: goal_tolerance was on pilz_circ and pilz_sequence only until M3.4; since
    TMR 2.3.0 the MoveGroupInterface rows honour it too. The continuous dense path still does not."""
    tol = c(goal_tolerance={'position_m': 0.001})
    assert resolve(MotionRequest('m', LIN, tol)).provider == 'pilz_lin'
    assert resolve(MotionRequest('m', PTP, tol)).provider == 'pilz_ptp'
    assert resolve(MotionRequest('m', FREE, tol)).provider == 'ompl'
    refused(MotionRequest('d', PATH, c(goal_tolerance={'position_m': 0.001,
                                                       'orientation_rad': 0.01}),
                          dense=True, continuity=BLEND), 'R-M3-12',
            'no dense_path provider can honour goal_tolerance')
    assert resolve(MotionRequest('m', CIRC, tol)).provider == 'pilz_circ'


def test_the_candidates_of_a_kind_are_the_rows_it_could_choose():
    assert cap.candidate_providers('ptp') == ['pilz_ptp']
    assert cap.candidate_providers('free') == ['ompl', 'ompl_chomp']
    assert cap.candidate_providers('lin') == ['pilz_lin']           # the default only
    assert cap.candidate_providers('circ') == ['pilz_circ']
    assert cap.candidate_providers('path') == ['pilz_sequence']
    assert cap.candidate_providers('dense_path') == ['cartesian_waypoints']


def test_r_m3_10_another_row_of_the_kind_has_it():
    """free + a token one free row lacks and another has: the chosen row is refused (R-M3-10),
    naming the row that has it -- here on a registry where only ompl_chomp provides tcp_limits."""
    registry = copy.deepcopy(REGISTRY)
    registry['providers']['ompl_chomp']['provides'].append('tcp_limits')
    refused(MotionRequest('m', FREE, c(tcp_velocity_mps=0.1)), 'R-M3-10',
            "provider 'ompl' does not provide 'tcp_limits'",
            'tcp_limits <- tcp_velocity_mps (provided by ompl_chomp)', registry=registry)


def test_the_compatibility_rule_refuses_an_unknown_provider_by_name():
    with pytest.raises(CapabilityRefusal, match="R-M3-06 .*provider 'spline'"):
        missing_capabilities('spline', ['joint_ptp'])


def test_an_edit_is_validated_like_a_load():
    model = MotionConstraints(speed=20)
    with pytest.raises(ValueError, match='R-M3-07'):
        model.speed = 500
    assert model.speed == 20


def test_r_m3_11_a_provider_that_is_not_stable_or_not_deliverable():
    registry = copy.deepcopy(REGISTRY)
    registry['providers']['ompl_chomp']['stability'] = 'experimental'
    refused(MotionRequest('m', FREE, provider='ompl_chomp'), 'R-M3-11',
            "provider 'ompl_chomp' is not eligible", registry=registry)
    registry = copy.deepcopy(REGISTRY)
    registry['pipelines']['pilz_industrial_motion_planner']['loaded'] = False
    refused(MotionRequest('m', LIN), 'R-M3-11', "provider 'pilz_lin' is not eligible",
            registry=registry)                                   # a default is checked too
    one = copy.deepcopy(REGISTRY)
    for section in ('capabilities', 'providers', 'provider_defaults'):
        one.pop(section)
    one['schema_version'] = 1
    refused(MotionRequest('m', PTP), 'R-M3-11', 'schema 1) has no provider for a ptp request',
            registry=one)


def test_r_m3_12_a_constraint_no_provider_of_the_step_honours():
    refused(MotionRequest('m', FREE, c(tcp_velocity_mps=0.1)), 'R-M3-12',
            'no free provider can honour tcp_velocity_mps',
            'tcp_limits <- tcp_velocity_mps (no free provider provides it)')
    # vNext M3.4: pilz_ptp provides goal_tolerance (TMR 2.3.0); a joint_rad tolerance is refused by
    # the task program (TMR carries none), the resolution's example is a ptp TCP limit
    refused(MotionRequest('m', PTP, c(tcp_velocity_mps=0.1)), 'R-M3-12',
            'no ptp provider can honour tcp_velocity_mps')
    refused(MotionRequest('m', LIN, c(planning_time_s=2.0)), 'R-M3-12',
            'no lin provider can honour planning_time_s')
    refused(MotionRequest('m', LIN, c(duration_s=2.0)), 'R-M3-12', 'duration_s is reserved')
    refused(MotionRequest('p', PATH, segments=(seg('lin', goal_tolerance={
        'position_m': 0.001, 'orientation_rad': 0.01}), seg('lin'))), 'R-M3-12',
        'segment 0 sets a goal_tolerance')
    free = resolve(MotionRequest('m', FREE, c(planning_time_s=2.0, planning_attempts=3)))
    assert free.provider == 'ompl'                                  # a free move honours them


def test_nothing_is_dropped_every_request_field_is_required_or_refused():
    """No silent degradation: for every field, on every intent, either a token the resolved
    provider provides was required for it, or the request is refused."""
    fields = dict(speed=20, acceleration=20, tcp_velocity_mps=0.1, tcp_acceleration_mps2=0.5,
                  blend_radius_m=0.01, goal_tolerance={'position_m': 0.001,
                                                       'orientation_rad': 0.01},
                  planning_time_s=1.0, planning_attempts=2, jerk=1.0,
                  orientation_constraint={'axis': 'z'}, duration_s=1.0)
    for intent in (PTP, FREE, LIN, CIRC):
        for name, value in fields.items():
            request = MotionRequest('m', intent, c(**{name: value}))
            try:
                result = resolve(request, limits=LIMITS)
            except CapabilityRefusal:
                continue
            tokens = cap.CONSTRAINT_TOKENS[name]
            assert tokens and all(t in result.required and t in result.provided
                                  for t in tokens), (intent, name)


def test_resolve_all_reports_every_refusal_at_once():
    requests = [MotionRequest('a', PTP), MotionRequest('b', FREE, c(tcp_velocity_mps=0.1)),
                MotionRequest('c', LIN, c(jerk=1))]
    with pytest.raises(CapabilityRefusals) as exc:
        resolve_all(requests, limits=LIMITS)
    assert [r.rule for r in exc.value.refusals] == ['R-M3-12', 'R-M3-10']
    assert [r.where for r in exc.value.refusals] == ['b', 'c']
    assert [r.provider for r in resolve_all(requests[:1])] == ['pilz_ptp']


def test_a_refusal_is_a_value_error_like_every_generation_refusal():
    assert issubclass(CapabilityRefusal, ValueError) and issubclass(CapabilityRefusals,
                                                                    ValueError)


def test_the_model_refuses_a_copy_without_the_vocabulary_tsa_names(tmp_path):
    """As for a planner mode TSA names (test_planner_registry.py): the model stops at import."""
    import os
    import subprocess
    import sys
    pkg = Path(__file__).resolve().parents[1]
    code = (
        'import trainit_setup_assistant.catalog as c\n'
        'reg = c.tmr_planner_registry()\n'
        "reg['capabilities']['reserved'].remove('jerk_constraints')\n"
        'c._registry_copy = lambda: reg\n'
        'import trainit_setup_assistant.model\n')
    path = os.pathsep.join([str(pkg), os.environ.get('PYTHONPATH', '')])
    env = dict(os.environ, PYTHONPATH=path)
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                       cwd=str(tmp_path))
    assert r.returncode != 0
    assert "no longer declares [\"capability 'jerk_constraints'\"]" in r.stderr, r.stderr
    assert 'change TSA first' in r.stderr


def test_numbers_in_refusals_are_finite_and_scalings_in_range():
    for request in (MotionRequest('m', LIN, c(speed=1, tcp_velocity_mps=1.0)),
                    MotionRequest('m', PTP, c(speed=100, acceleration=1))):
        result = resolve(request, limits=LIMITS)
        for s in (result.velocity_scaling, result.acceleration_scaling):
            assert s is None or (math.isfinite(s.value) and 0 < s.value <= 1)
    assert catalog.PLANNER_REGISTRY_SCHEMA == 2
