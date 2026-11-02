"""TSA reads TMR's planner registry at schema 2 (vNext M3.2, ADR-0025 Decision 7; ADR-0021).

Schema 2 is schema 1 plus three sections: ``capabilities`` (live and reserved tokens),
``providers`` (one row per MoveIt route: what it provides) and ``provider_defaults`` (the row a
motion kind gets when nothing chooses one). TMR owns all three; TSA reads them from its committed
byte copy, never from the installed TMR, and checks them structurally so no reader half-reads
them. These tests pin what the committed copy says, that a schema-1 registry still reads (the
planner vocabulary unchanged, no providers), that every structural defect is refused by name, and
which rows a request may resolve to (stable and deliverable, as ADR-0021 Decision 5 says of planner
modes). The byte copy and the drift test are test_planner_registry.py's and test_catalog.py's.
"""
import copy

import pytest
import yaml

from trainit_setup_assistant import catalog
from trainit_setup_assistant.catalog import (TSA_CAPABILITY_TOKENS, TSA_MOTION_KINDS,
                                             capability_vocabulary, declared_providers,
                                             eligible_providers, missing_capability_vocabulary,
                                             offered_planner_modes, parse_planner_registry,
                                             provider_defaults, provider_problems,
                                             tmr_planner_registry)

LIVE = ['joint_ptp', 'collision_free_planning', 'cartesian_linear', 'cartesian_circular',
        'multi_segment_path', 'blending', 'continuous_path', 'velocity_scaling',
        'acceleration_scaling', 'tcp_limits', 'goal_tolerance', 'process_path']
RESERVED = ['orientation_constraints', 'jerk_constraints', 'precomputed_trajectory']
# TMR's rows (ADR-0025 Decision 2), and what each provides; vNext M3.4 (TMR 2.3.0): goal_tolerance on
# the MoveGroupInterface rows, which honour it since
PROVIDES = {
    'pilz_ptp': ['joint_ptp', 'velocity_scaling', 'acceleration_scaling', 'goal_tolerance'],
    'ompl': ['collision_free_planning', 'velocity_scaling', 'acceleration_scaling', 'goal_tolerance'],
    'ompl_chomp': ['collision_free_planning', 'velocity_scaling', 'acceleration_scaling',
                   'goal_tolerance'],
    'pilz_lin': ['cartesian_linear', 'velocity_scaling', 'acceleration_scaling', 'tcp_limits',
                 'goal_tolerance'],
    'pilz_circ': ['cartesian_circular', 'velocity_scaling', 'acceleration_scaling', 'tcp_limits',
                  'goal_tolerance'],
    'pilz_sequence': ['multi_segment_path', 'blending', 'cartesian_linear', 'cartesian_circular',
                      'velocity_scaling', 'acceleration_scaling', 'tcp_limits', 'goal_tolerance',
                      'process_path'],
    'cartesian_waypoints': ['continuous_path', 'velocity_scaling', 'acceleration_scaling'],
}
DEFAULTS = {'ptp': 'pilz_ptp', 'free': 'ompl', 'lin': 'pilz_lin', 'circ': 'pilz_circ',
            'path': 'pilz_sequence', 'dense_path': 'cartesian_waypoints'}


def _dump(registry: dict) -> bytes:
    return yaml.safe_dump(registry, sort_keys=False).encode()


def schema_1(registry: dict) -> dict:
    """The same registry as schema 1: the three schema-2 sections removed."""
    out = copy.deepcopy(registry)
    for section in ('capabilities', 'providers', 'provider_defaults'):
        out.pop(section)
    out['schema_version'] = 1
    return out


# --- what the committed copy says -------------------------------------------------------------

def test_the_copy_is_schema_2_with_its_providers_tokens_and_defaults():
    registry = tmr_planner_registry()
    assert registry['schema_version'] == 2 == catalog.PLANNER_REGISTRY_SCHEMA
    assert capability_vocabulary(registry) == {'live': LIVE, 'reserved': RESERVED}
    assert {name: row['provides'] for name, row in declared_providers(registry).items()} == \
        PROVIDES
    assert list(declared_providers(registry)) == list(PROVIDES)              # registry order
    assert provider_defaults(registry) == DEFAULTS
    assert eligible_providers(registry) == list(PROVIDES)                   # all stable today
    assert provider_problems(registry) == [] and missing_capability_vocabulary(registry) == []


def test_tsa_names_only_tokens_and_kinds_tmr_declares():
    registry = tmr_planner_registry()
    assert set(TSA_CAPABILITY_TOKENS) <= set(LIVE) | set(RESERVED)
    assert set(TSA_MOTION_KINDS) == set(provider_defaults(registry))


def test_schema_2_left_the_planner_vocabulary_alone():
    """ADR-0025 Decision 1: nothing above the three new sections changed."""
    registry = tmr_planner_registry()
    assert offered_planner_modes(registry) == offered_planner_modes(schema_1(registry))


# --- schema 1 still reads (ADR-0025 Decision 7) -------------------------------------------------

def test_a_schema_1_registry_reads_with_its_planner_vocabulary_and_no_providers():
    one = parse_planner_registry(_dump(schema_1(tmr_planner_registry())), 'a schema-1 registry')
    assert one['schema_version'] == 1
    assert offered_planner_modes(one) == ['pilz', 'ompl', 'ompl_chomp']
    assert declared_providers(one) == {} and eligible_providers(one) == []
    assert provider_defaults(one) == {}
    assert capability_vocabulary(one) == {'live': [], 'reserved': []}
    assert missing_capability_vocabulary(one) == []


# --- every structural defect is refused by name --------------------------------------------------

def _mutated(edit):
    registry = tmr_planner_registry()
    edit(registry)
    return registry


@pytest.mark.parametrize('edit, says', [
    (lambda r: r.update(schema_1(r), providers={}),
     "schema 1 carries the schema-2 section 'providers'"),
    (lambda r: r.pop('provider_defaults'), "schema 2 without its 'provider_defaults' section"),
    (lambda r: r.pop('capabilities'), "schema 2 without its 'capabilities' section"),
    (lambda r: r['capabilities'].update(extra=['x']), 'exactly live and reserved'),
    (lambda r: r['capabilities']['live'].append('joint_ptp'), 'each a list of distinct tokens'),
    (lambda r: r['capabilities']['reserved'].append('blending'),
     "capability 'blending' is both live and reserved"),
    (lambda r: r['providers']['pilz_lin']['provides'].append('jerk_constraints'),
     "provider 'pilz_lin' provides the reserved capability 'jerk_constraints'"),
    (lambda r: r['providers']['pilz_lin']['provides'].append('teleport'),
     "provider 'pilz_lin' provides 'teleport', which capabilities does not declare"),
    (lambda r: r['providers']['pilz_lin']['provides'].append('tcp_limits'),
     "provider 'pilz_lin': provides must be a list of distinct tokens"),
    (lambda r: r['providers']['pilz_lin'].update(cartesian='spiral'),
     "provider 'pilz_lin' points at cartesian 'spiral', which the registry does not declare"),
    (lambda r: r['providers']['ompl'].update(planner_mode='rrt_star'),
     "provider 'ompl' points at planner_mode 'rrt_star', which the registry does not declare"),
    (lambda r: r['providers']['ompl'].update(cartesian='linear'),
     "provider 'ompl' points at both a planner_mode and a cartesian entry"),
    (lambda r: r['providers']['ompl'].pop('stability'), "provider 'ompl' has no route or "
                                                        'stability'),
    (lambda r: r['providers'].pop('cartesian_waypoints'),
     "provider_defaults.dense_path names 'cartesian_waypoints', which is not a provider row"),
    (lambda r: r['capabilities']['live'].append('teleport'),
     "live capability 'teleport' is provided by no row"),
    (lambda r: r['provider_defaults'].update(lin='pilz_linear'),
     "provider_defaults.lin names 'pilz_linear', which is not a provider row"),
    (lambda r: r['provider_defaults'].update(lin=['pilz_lin']),
     "provider_defaults.lin names ['pilz_lin'], which is not a provider row"),
    (lambda r: r['providers']['ompl'].update(planner_mode=['ompl']),
     "provider 'ompl' points at planner_mode ['ompl'], which the registry does not declare"),
], ids=['schema_1_with_providers', 'no_defaults', 'no_capabilities', 'capabilities_extra_key',
        'duplicate_token', 'live_and_reserved', 'provides_reserved', 'provides_undeclared',
        'provides_twice', 'dangling_cartesian', 'dangling_planner_mode', 'two_pointers',
        'no_stability', 'default_to_removed_row', 'unprovided_live_token', 'default_unknown',
        'default_not_a_string', 'pointer_not_a_string'])
def test_a_malformed_schema_2_section_is_refused_by_every_reader(edit, says):
    registry = _mutated(edit)
    assert any(says in p for p in provider_problems(registry)), provider_problems(registry)
    with pytest.raises(RuntimeError, match='run python3 -m trainit_setup_assistant'):
        parse_planner_registry(_dump(registry), 'the copy')       # catalog AND derive read it


@pytest.mark.parametrize('edit, missing', [
    (lambda r: r['capabilities']['reserved'].remove('jerk_constraints'),
     ["capability 'jerk_constraints'"]),
    (lambda r: r['provider_defaults'].pop('dense_path'), ['provider_defaults.dense_path']),
], ids=['token_tsa_derives', 'kind_tsa_resolves'])
def test_a_registry_that_drops_what_tsa_names_is_named(edit, missing):
    assert missing_capability_vocabulary(_mutated(edit)) == missing


# --- which rows a request may resolve to (R-M3-11) ----------------------------------------------

@pytest.mark.parametrize('edit, gone', [
    (lambda r: r['providers']['ompl_chomp'].update(stability='experimental'), 'ompl_chomp'),
    (lambda r: r['pipelines']['chomp'].update(loaded=False), 'ompl_chomp'),      # mode not offered
    (lambda r: r['providers']['ompl'].update(planner_mode='stomp'), 'ompl'),      # never offered
    (lambda r: r['pipelines']['pilz_industrial_motion_planner'].update(loaded=False),
     'pilz_lin'),
], ids=['experimental_row', 'optimizer_not_loaded', 'experimental_mode', 'cartesian_unloaded'])
def test_a_row_that_is_not_stable_or_not_deliverable_is_not_eligible(edit, gone):
    registry = _mutated(edit)
    assert gone in declared_providers(registry)
    assert gone not in eligible_providers(registry)
    assert 'cartesian_waypoints' in eligible_providers(registry)   # no pipeline: always deliverable
