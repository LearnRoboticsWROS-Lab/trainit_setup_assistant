"""The TrainIt provider catalog -- DATA, read-only (vNext M1, ADR-0019).

A provider is a ROW, not a class: a capability TrainIt can use, and the facts a deployment
decision needs about it -- who makes it, under which licence and how sure we are of that, what it
needs to run, how it plugs in. Hardware and simulator ADAPTERS are rows of their own kind and are
NOT providers (owner decision: the FR3WML interface is a robot adapter, the surface gripper an
EEF adapter, cuMotion a motion provider).

Four data files beside this module:

  snapshot.yaml    what TrainIt HAS today, derived from the registries that already exist
                   (``derive.derive()``) and committed. The drift test fails when a registry
                   moves and the snapshot does not.
  candidates.yaml  what TrainIt MIGHT use -- the NVIDIA stack first. External metadata only:
                   nothing in it is installed, configured or benchmarked, and every licence claim
                   carries its status and its exact source.
  tmr_planner_registry.yaml
                   a BYTE copy of TMR's planner registry (``config/planner_profiles.yaml``):
                   TSA's planner vocabulary is built from it, never from the installed TMR
                   (ADR-0021).
  tp_registry_snapshot.yaml
                   trainit_perception's configuration vocabulary as this TSA release supports it:
                   the canonical YAML of ``trainit_perception.vocabulary.export()`` (ADR-0021,
                   amendment M2.2). TSA's rule vocabulary, the terms a rule needs, what a method
                   provides, the grasp faces and the version floors are built from it, never from
                   the installed TP. A closed schema: this module refuses a key it does not know.

One more table is committed as code, not as a file: ``TMR_FLOORS``, the oldest TMR release that
has each TMR capability a bundle can use -- what a bundle needs to RUN, which its package.xml
files declare (vNext M2.4, ADR-0021 amendment M2.4). It is kept by hand, as ``TP_TIERS`` is:
each row names the TMR commit it was read from. Since vNext M2.6 two more: ``EEF_ADAPTER_ROWS``,
what serves the end-effector contract on each backend, and ``EEF_BRIDGE_ROLES``, what a bridge
row of a deployment is (ADR-0022; ADR-0021's named exception M2.6-1). ``derive`` copies the
adapter rows into ``snapshot.yaml``.

The two registry files and ``snapshot.yaml`` are written by one explicit maintenance step, all
three or none, never while generating: ``python3 -m trainit_setup_assistant.catalog.refresh``
(``--check`` only compares). ``candidates.yaml`` is curated by hand, with its sources. Generation
reads the two registry files, and the bytes of ``snapshot.yaml`` for the identity every bundle
records (:func:`snapshot_identity`, vNext M2.3), and nothing else here: it never imports
``derive`` and never reads ``candidates.yaml``.

This module is a LEAF: it imports nothing from the rest of TSA, so the model can build its
vocabulary from it without an import cycle.

Licence status, for every row:

  VERIFIED         the licence facts were read at an exact source (a tag-pinned LICENSE or
                   package.xml, a model card) on ``checked_on``
  REQUIRES_REVIEW  the sources conflict, or reading them needs a legal decision
  UNKNOWN          the source could not be found or does not state it

A status is never upgraded by assumption: no source, no VERIFIED.
"""
from __future__ import annotations

import copy
import functools
import hashlib
import re
from importlib import resources
from typing import Dict, List, Optional

import yaml

LICENSE_STATUS = ('VERIFIED', 'REQUIRES_REVIEW', 'UNKNOWN')
DISTRIBUTION = ('community_candidate', 'pro_candidate', 'extension_point', 'external')
FAMILY = ('motion_planning', 'motion_execution', 'perception_detection', 'perception_pose',
          'perception_depth', 'perception_segmentation', 'perception_fiducials', 'grasp_ranking',
          'grasp_generation', 'scene', 'policy_execution', 'policy_model', 'policy_training',
          'simulation', 'reference_workflow')
KIND = ('in_tree', 'candidate')
MATURITY = ('verified', 'verified_manually', 'development', 'non_production_ready',
            'as_its_backend')


REGISTRY_FILE = 'tmr_planner_registry.yaml'
# The registry schemas this TSA reads (ADR-0025 Decision 7): 1 (the planner vocabulary) and 2
# (schema 1 + ``capabilities``, ``providers``, ``provider_defaults``). Any other number is refused
# by name -- a newer TMR schema needs TSA's readers first (ADR-0021's landing order).
PLANNER_REGISTRY_SCHEMAS = (1, 2)
PLANNER_REGISTRY_SCHEMA = max(PLANNER_REGISTRY_SCHEMAS)     # the newest, the committed copy's
SNAPSHOT_FILE = 'snapshot.yaml'
REFRESH = 'python3 -m trainit_setup_assistant.catalog.refresh'

# The planner modes TSA's own code names (the default of a project file without the key is
# PlannerId.PILZ; the D-016 vision rule keys on PILZ and switches to OMPL). TMR's registry must keep
# offering them, or TSA changes first: the refresh refuses, and the model says so at import.
TSA_REQUIRED_PLANNER_MODES = ('pilz', 'ompl')

# --- registry schema 2: motion provider capabilities (ADR-0025) ------------------------------
#
# TMR owns the capability vocabulary and the provider rows; TSA reads them from this copy and
# names only what its capability resolution derives (``model.capabilities``). Like the planner
# modes above, what TSA's code names must still be in the registry, or TSA changes first: the
# refresh refuses, and the model says so at import.
#   TSA_CAPABILITY_TOKENS  every token the required-capability derivation can produce
#   TSA_MOTION_KINDS       every kind of motion request TSA resolves a default provider for --
#                          the keys of the registry's ``provider_defaults``
TSA_CAPABILITY_TOKENS = ('joint_ptp', 'collision_free_planning', 'cartesian_linear',
                         'cartesian_circular', 'multi_segment_path', 'blending', 'continuous_path',
                         'velocity_scaling', 'acceleration_scaling', 'tcp_limits', 'goal_tolerance',
                         'jerk_constraints', 'orientation_constraints')
TSA_MOTION_KINDS = ('ptp', 'free', 'lin', 'circ', 'path', 'dense_path')
_SCHEMA_2_SECTIONS = ('capabilities', 'providers', 'provider_defaults')
_POINTED = {'planner_mode': 'planner_modes', 'cartesian': 'cartesian'}  # pointer -> section


def _load(name: str):
    return yaml.safe_load(resources.files(__name__).joinpath(name).read_bytes())


def parse_planner_registry(data: bytes, where: str) -> dict:
    """A planner registry from its bytes (UTF-8, one loader for every reader), schema-checked.
    Bytes that are not YAML, or not a registry at all, are a RuntimeError that says how to repair
    the committed copy (vNext M2.2: the TP reader beside it does the same). Schema 2's provider
    sections are checked structurally here (:func:`provider_problems`), so no reader half-reads
    them."""
    try:
        registry = yaml.safe_load(data) or {}
    except (yaml.YAMLError, ValueError) as exc:
        raise RuntimeError(f'{where}: not YAML ({str(exc).splitlines()[0]}). A committed copy is '
                           f'generated: run {REFRESH}') from None
    if not isinstance(registry, dict):
        raise RuntimeError(f'{where}: not a planner registry (a {type(registry).__name__}). A '
                           f'committed copy is generated: run {REFRESH}')
    schema = registry.get('schema_version')
    if isinstance(schema, bool) or schema not in PLANNER_REGISTRY_SCHEMAS:
        raise RuntimeError(
            f'{where}: planner registry schema_version {schema!r}, and this TSA reads schemas '
            f'{list(PLANNER_REGISTRY_SCHEMAS)}. A newer TMR schema needs TSA\'s readers updated '
            'first (catalog/__init__.py, catalog/derive.py); a copy that is merely stale is fixed '
            'by ' + REFRESH)
    if not (isinstance(registry.get('pipelines'), dict)
            and isinstance(registry.get('planner_modes'), dict)):
        raise RuntimeError(f'{where}: a planner registry without its pipelines and planner_modes '
                           f'mappings. A committed copy is generated: run {REFRESH}')
    problems = provider_problems(registry)
    if problems:
        raise RuntimeError(f'{where}: ' + '; '.join(problems) + '. A committed copy is '
                           f'generated: run {REFRESH}')
    return registry


def provider_problems(registry: dict) -> List[str]:
    """Why a registry's schema-2 sections cannot be read as ADR-0025 defines them. Empty = they
    can. Schema 1 must not carry them (a half-upgraded file); schema 2 must carry all three:

    * ``capabilities``: ``live`` and ``reserved``, disjoint lists of distinct tokens;
    * ``providers``: each row a ``route``, at most one pointer (``planner_mode`` or ``cartesian``)
      to an entry of the sections above (none: a route that runs no MoveIt pipeline, e.g.
      ``cartesian_path``), a ``stability``, and ``provides`` -- distinct live
      tokens, never a reserved one; every live token provided by at least one row;
    * ``provider_defaults``: kind -> a declared row."""
    present = [s for s in _SCHEMA_2_SECTIONS if s in registry]
    if registry.get('schema_version') == 1:
        return [f'schema 1 carries the schema-2 section {s!r}' for s in present]
    out = [f'schema 2 without its {s!r} section' for s in _SCHEMA_2_SECTIONS if s not in present]
    if out:
        return out
    caps, rows, defaults = (registry[s] for s in _SCHEMA_2_SECTIONS)
    if not (isinstance(caps, dict) and set(caps) == {'live', 'reserved'}
            and all(_tokens(caps[k]) and len(set(caps[k])) == len(caps[k]) for k in caps)):
        return ['capabilities must be exactly live and reserved, each a list of distinct tokens']
    live, reserved = caps['live'], caps['reserved']
    out += [f'capability {t!r} is both live and reserved' for t in live if t in reserved]
    if not (isinstance(rows, dict) and rows and all(isinstance(r, dict) for r in rows.values())):
        return out + ['providers must be a mapping of rows']
    provided = set()
    for name, row in rows.items():
        pointers = [k for k in ('planner_mode', 'cartesian') if k in row]
        if not _TOKEN.match(str(name)):
            out.append(f'provider {name!r} is not a token')
        if not isinstance(row.get('route'), str) or not isinstance(row.get('stability'), str):
            out.append(f'provider {name!r} has no route or stability')
        if len(pointers) > 1:
            out.append(f'provider {name!r} points at both a planner_mode and a cartesian entry')
        elif pointers and (not isinstance(row[pointers[0]], str) or row[pointers[0]] not in
                           (registry.get(_POINTED[pointers[0]]) or {})):
            out.append(f'provider {name!r} points at {pointers[0]} {row[pointers[0]]!r}, which '
                       'the registry does not declare')
        tokens = row.get('provides')
        if not (_tokens(tokens) and len(set(tokens)) == len(tokens)):
            out.append(f'provider {name!r}: provides must be a list of distinct tokens')
            continue
        out += [f'provider {name!r} provides the reserved capability {t!r}' for t in tokens
                if t in reserved]
        out += [f'provider {name!r} provides {t!r}, which capabilities does not declare'
                for t in tokens if t not in live and t not in reserved]
        provided.update(tokens)
    out += [f'live capability {t!r} is provided by no row' for t in live if t not in provided]
    if not isinstance(defaults, dict):
        return out + ['provider_defaults must be a mapping']
    out += [f'provider_defaults.{kind} names {row!r}, which is not a provider row'
            for kind, row in defaults.items() if not isinstance(row, str) or row not in rows]
    return out


def missing_capability_vocabulary(registry: dict) -> List[str]:
    """What TSA's capability resolution names that a schema-2 registry no longer declares: a
    token (``TSA_CAPABILITY_TOKENS``) or a motion kind's default (``TSA_MOTION_KINDS``). A
    schema-1 registry declares no capabilities: nothing is missing, there is nothing to resolve."""
    if registry.get('schema_version') == 1:
        return []
    caps = registry['capabilities']
    declared = set(caps['live']) | set(caps['reserved'])
    return ([f'capability {t!r}' for t in TSA_CAPABILITY_TOKENS if t not in declared]
            + [f'provider_defaults.{k}' for k in TSA_MOTION_KINDS
               if k not in registry['provider_defaults']])


def capability_vocabulary(registry: dict) -> Dict[str, List[str]]:
    """The registry's capability tokens, ``live`` and ``reserved``, in registry order; empty lists
    for schema 1."""
    caps = registry.get('capabilities') or {}
    return {'live': list(caps.get('live') or []), 'reserved': list(caps.get('reserved') or [])}


def declared_providers(registry: dict) -> Dict[str, dict]:
    """Every provider row TMR declares, by id, in registry order (empty for schema 1)."""
    return {name: dict(row) for name, row in (registry.get('providers') or {}).items()}


def eligible_providers(registry: dict) -> List[str]:
    """The provider rows a project may be resolved to, in registry order: STABLE and DELIVERABLE
    (ADR-0021 Decision 5, as ADR-0025 extends it to rows) -- a row pointing at a planner mode needs
    that mode offered (:func:`offered_planner_modes`), a row pointing at a Cartesian shape needs
    the shape's pipeline loaded, a row with no pointer runs no planning pipeline (it is
    deliverable). Any other declared row is refused by name (R-M3-11)."""
    loaded = {name for name, spec in registry['pipelines'].items() if spec.get('loaded')}
    offered = set(offered_planner_modes(registry))
    shapes = registry.get('cartesian') or {}
    out = []
    for name, row in declared_providers(registry).items():
        if row.get('stability') != 'stable':
            continue
        if 'planner_mode' in row:
            deliverable = row['planner_mode'] in offered
        elif 'cartesian' in row:
            deliverable = (shapes.get(row['cartesian']) or {}).get('pipeline') in loaded
        else:
            deliverable = True
        if deliverable:
            out.append(name)
    return out


def provider_defaults(registry: dict) -> Dict[str, str]:
    """The registry's default provider row per motion kind (empty for schema 1)."""
    return dict(registry.get('provider_defaults') or {})


def missing_required_modes(registry: dict) -> List[str]:
    """The planner modes TSA's code names that this registry does not offer."""
    offered = offered_planner_modes(registry)
    return [m for m in TSA_REQUIRED_PLANNER_MODES if m not in offered]


def _committed(name: str) -> bytes:
    """A generated catalog file as committed, or a RuntimeError that says how to get it back."""
    try:
        return resources.files(__name__).joinpath(name).read_bytes()
    except OSError as exc:
        raise RuntimeError(f'catalog/{name} is missing or unreadable ({exc.__class__.__name__}): '
                           f'it is generated -- restore it with git, or run {REFRESH}') from None


# The bytes this PROCESS first parsed, by file name -- written once: the identity a bundle records
# is the one of the vocabulary the model was built from at import, even if the files changed on
# disk since or a reader re-read them (vNext M2.3); snapshot.yaml is pinned at its first read.
_PARSED: Dict[str, bytes] = {}


@functools.lru_cache(maxsize=None)
def _registry_copy() -> dict:
    data = _committed(REGISTRY_FILE)
    _PARSED.setdefault(REGISTRY_FILE, data)
    return parse_planner_registry(data, f'catalog/{REGISTRY_FILE}')


def tmr_planner_registry() -> dict:
    """TMR's planner registry as TSA committed it (ADR-0021): never the installed TMR."""
    return copy.deepcopy(_registry_copy())


def declared_planner_modes(registry: dict) -> List[str]:
    """Every planner_mode token TMR declares, in registry order (TMR parses all of them)."""
    return list(registry['planner_modes'])


def offered_planner_modes(registry: dict) -> List[str]:
    """The modes a project may select, in registry order: DELIVERABLE (the mode's pipeline, and
    its optimizer pipeline if it has one, are in the default loaded set) and STABLE. Any other
    declared mode degrades at runtime or is experimental, so TSA never offers it."""
    loaded = {name for name, spec in registry['pipelines'].items() if spec.get('loaded')}
    return [mode for mode, spec in registry['planner_modes'].items()
            if spec['pipeline'] in loaded
            and (spec.get('optimizer_pipeline') or spec['pipeline']) in loaded
            and spec.get('stability') == 'stable']


# --- what a bundle needs to RUN: TMR's version floors (ADR-0021, amendment M2.4) -----------------
#
# The oldest trainit_motion_runtime release that certainly has each TMR capability a generated
# bundle can use: committed compatibility data, curated from TMR's history with the owner's rule
# (vNext M2.4, 2026-10-08). A capability counts from the first package.xml version that certainly
# contains it; one added while package.xml already said X counts from the next bump; a BEHAVIOUR a
# bundle relies on counts like a capability. 0.0.0 certifies nothing: every TMR tree before
# d7a9fde says it -- the tagged v1.0.0 to v1.3.0 included -- and so do copies outside this history,
# so the oldest floor is TMR_FIRST_VERSION, the first version TMR's package.xml ever stated.
# Generation reads this table and the files it wrote (generator/runtime_requirements.py): never an
# installed TMR, never the snapshot's TMR version -- that one says what COMPILED a bundle, not what
# the bundle needs. The table is TSA code: a bundle's generator_version identifies it.
#
# An EXCEPTION to ADR-0021 Decisions 1 and 9 (a vocabulary and its floors are exported by the
# component that owns them): TMR has no export of the version that brought each node, port and
# behaviour, so TSA curates the dates (amendment M2.4-3). Closes when TMR exports them. A new TMR
# capability TSA emits needs its row here; test/test_runtime_requirements.py names a missing one.
TMR_FIRST_VERSION = '1.4.0'
# Every TMR BT node TSA can emit. Present in TMR_FIRST_VERSION unless TMR_FLOORS dates it later:
# the motion, gripper, process, scene and log nodes of TMR's first commit (6961738), the
# dynamic-object flags (ebb5a67), ResetScene (cdc131b), the vision nodes (6a65790) and the relative
# waypoint (d7a9fde). An emitted node outside this tuple and BT_BUILTIN_NODES stops generation.
TMR_BT_NODES = ('MoveWaypoint', 'ServoTrack', 'OpenGripper', 'CloseGripper', 'EnableProcess',
                'DisableProcess', 'AddCollisionObject', 'AttachObject', 'DetachObject',
                'SetAttachedCollisionCheck', 'ResetScene', 'SetReleasePolicy', 'DetectObject',
                'SetWaypointFromDetection', 'SetWaypointRelative', 'RunPolicy', 'CheckRobotState',
                'Log', 'VerifyGrasp', 'CommandActuator', 'FollowPath', 'ProcessEvent',
                'DetectCandidates', 'SetWaypointFromCandidate')
# BehaviorTree.CPP's own nodes TSA emits, and the tags of the tree file itself: no TMR floor
BT_BUILTIN_NODES = ('root', 'BehaviorTree', 'Sequence', 'Fallback', 'Sleep', 'Script', 'Repeat',
                    'RetryUntilSuccessful', 'SubTree')
# The one placeholder the floors table admits, for a row whose TMR commit does not exist yet
# (test_runtime_requirements.py names every row that still carries it). vNext M2.6 used it until
# the TMR 2.0.0 commit adf94b0; no row carries it since.
TMR_EEF_EVIDENCE_PENDING = 'PENDING-TMR-2.0.0'
# capability -> (floor, the TMR commit that is its evidence, what the package.xml comment says).
# The comment text goes into an XML comment: never two dashes in a row.
TMR_FLOORS = {
    'ServoTrack': ('1.5.0', 'b56fbe3', 'ServoTrack (execution: servo)'),
    'RunPolicy': ('1.5.0', '45ebe5c', 'RunPolicy (a policy step)'),
    'CheckRobotState': ('1.5.0', '45ebe5c', 'CheckRobotState'),
    'launch_backend': ('1.5.0', '37c87d9', "the launch helper's backend= (gazebo)"),
    'gripper_positions': ('1.5.0', 'ff165f8', 'the joint-position gripper targets'),
    'scene_manager_namespace': ('1.5.0', '51670ed', 'scene_manager_node under a namespace'),
    'detect_keep': ('1.6.0', 'c47a724', 'DetectObject keep (a candidate list)'),
    'detection_index': ('1.6.0', 'c47a724', 'SetWaypointFromDetection index'),
    'detect_first_published': ('1.7.0', '5593e5b',
                               'DetectObject keep=1 takes the first published match'),
    'scene_services_namespace': ('1.7.0', 'fd4fe5a',
                                 'SetAttachedCollisionCheck/ResetScene reach the cell namespace'),
    # The end-effector contract (ADR-0022): only a bundle that opts into it uses these, so only
    # such a bundle gets the 2.0.0 floor. The evidence is the TMR commit that ships them.
    'VerifyGrasp': ('2.0.0', 'adf94b0', 'VerifyGrasp (the end-effector contract)'),
    'CommandActuator': ('2.0.0', 'adf94b0', 'CommandActuator (the end-effector contract)'),
    'detect_fault': ('2.0.0', 'adf94b0', "DetectObject's fault output"),
    'eef_block': ('2.0.0', 'adf94b0', 'the eef parameter block (ADR-0022)'),
    # vNext M2.7 (owner Q8.2): only the canonical bin-picking cycle remaps MoveWaypoint's fault.
    # The port shipped with TMR 2d230bc; 2a76a55 made a colliding start a FAULT, which the cycle
    # relies on.
    'move_waypoint_fault': ('2.1.0', '2a76a55', "MoveWaypoint's fault output"),
    # vNext G0 (owner section 5, ADR-0023): MoveGroupInterface resolves in the runner's namespace
    # since c2c9f01 (package.xml 2.0.0 there, so 2.1.0); before it a namespaced trainit_run_bt
    # waited for a ROOT move_group. A canonical-only row: a package gets it iff its bt_params carry
    # the eef block AND its launch starts trainit_run_bt under a namespace. A legacy namespaced
    # package relies on it too, yet keeps its floor (owner M2.7 sections 4 and 17; KNOWN_DEBT).
    'moveit_clients_namespace': (
        '2.1.0', 'c2c9f01', 'trainit_run_bt reaches MoveIt in the cell namespace (ADR-0023)'),
    # vNext M3.1 (ADR-0024), used by a Canonical Task Program (vNext M3.3): FollowPath, the runner's
    # paths_file, and MoveWaypoint's profile / frame / tcp keys arrived with TMR 2.2.0 (b399a33).
    'FollowPath': ('2.2.0', 'b399a33', 'FollowPath (a path of config/paths.yaml)'),
    'waypoint_canonical_keys': ('2.2.0', 'b399a33', "MoveWaypoint's profile, frame and tcp keys"),
    # vNext M3.4 (ADR-0025 Decision 3, owner D3): a ptp / free / lin move whose motion profile
    # carries a goal tolerance relies on TMR honouring it on the MoveGroupInterface routes, which
    # 8ed8a36 made true (TMR 2.3.0); before it the tolerance was dropped. A circ move or a path
    # never needed it (the sequence route always sent it).
    'move_goal_tolerance': ('2.3.0', '8ed8a36', "a move's goal tolerance on a MoveGroupInterface route"),
    # vNext M3.5 (M3_DESIGN §9), used by a task program with process events: the generic ProcessEvent
    # node, DetectObject's trigger (a fresh capture), FollowPath's event-free runs (from / to), and
    # paths.yaml's events key (TMR 2.3.0's loader refuses it as an unknown key) arrived with 6a2414e.
    'ProcessEvent': ('2.4.0', '6a2414e', 'ProcessEvent (a process event on a logical channel)'),
    'detect_trigger': ('2.4.0', '6a2414e', "DetectObject's trigger (a capture of a fresh result)"),
    'follow_path_range': ('2.4.0', '6a2414e', "FollowPath from / to (a path's event-free runs)"),
    'path_events': ('2.4.0', '6a2414e', "paths.yaml events (where a path stops for a process event)"),
    # vNext M3.6 (ADR-0026), used by a task program with manipulation steps: the grasp-candidate
    # contract's two nodes arrived with cb4d1a2 (TMR 2.5.0, which depends on trainit_interfaces >= 0.2.0,
    # the GraspCandidate messages: the bundle reaches the interface through TMR, never directly).
    'DetectCandidates': ('2.5.0', 'cb4d1a2',
                         'DetectCandidates (a grasp-candidate set; trainit_interfaces >= 0.2.0 through TMR)'),
    'SetWaypointFromCandidate': ('2.5.0', 'cb4d1a2', "SetWaypointFromCandidate (a grasp idiom's targets)"),
}
# The TMR_FLOORS rows only a CANONICAL package can be given -- one whose bt_params carry the
# end-effector contract's eef block (ADR-0022, its amendment M2.7; ADR-0021 amendment G0): a
# LEGACY package never gets them, so a legacy file that does not parse assumes every OTHER dated
# capability -- its floor stays what it was before the contract existed.
EEF_CONTRACT_CAPABILITIES = ('VerifyGrasp', 'CommandActuator', 'detect_fault', 'eef_block',
                             'move_waypoint_fault', 'moveit_clients_namespace')
# The TMR_FLOORS rows only a CANONICAL TASK PROGRAM's package can be given (vNext M3.3): a template
# application's package never gets them, so a template's file that does not parse keeps the floor it
# had before them.
TASK_PROGRAM_CAPABILITIES = ('FollowPath', 'waypoint_canonical_keys', 'move_goal_tolerance',
                             'ProcessEvent', 'detect_trigger', 'follow_path_range', 'path_events',
                             'DetectCandidates', 'SetWaypointFromCandidate')


# --- the end-effector ADAPTER rows (ADR-0022 D6) ------------------------------------------------
#
# What serves the end-effector contract on a backend: IMPLEMENTATION rows, never kinds (ADR-0019
# Decision 2, as ADR-0022 amends it -- there is no row for ``composite``). A canonical project
# binds each deployed backend to one row by its token (``robot.gripper.contract.adapters``), and
# TSA refuses a binding that does not run in that mode, does not serve an actuator's interface or
# the device's kind, or does not provide a required capability or an accepted evidence class
# (ADR-0022 R15/R16). The capability, interface and evidence tokens are the closed vocabularies of
# ``model.end_effector`` (a parity test holds them together, and both against the ADR's token file).
#
#   modes            the backends the row can serve
#   serves           the command interfaces it serves
#   kinds            the device kinds it serves
#   provides         the capability tokens it always provides
#   provides_if      tokens it provides only under a binding condition: ``stage_status_topic``
#                    (the stage publishes its status) or ``effort_flavour`` (an effort-flavour
#                    controller)
#   evidence         the evidence classes its grasp observations carry
#   evidence_configurable  the class is chosen per site by an adapter parameter (the mock)
#   binding_keys     the keys a binding to this row may set (``{adapter: <row>, <key>: ...}``)
#   requires         the binding keys it cannot run without (TSA derives stage_cmd_topic)
#   flavours         the controller flavours a binding declares (ros2_control)
#   package/executable/since  the node TSA starts for it, and the first release that has it
#
# An EXCEPTION to ADR-0021 Decisions 1 and 9 (amendment M2.6-1, named in ADR-0022): these facts
# belong to trainit_hardware_layer, which does not export them yet, so TSA types them here, as
# M2.4-3 did for TMR_FLOORS. Closes when the hardware layer exports them and TSA reads a fourth
# committed snapshot.
EEF_INTERFACES = ('set_bool', 'gripper_command', 'follow_joint_trajectory')
EEF_KINDS = ('suction', 'parallel', 'composite')
EEF_CAPABILITY_TOKENS = ('binary_actuation', 'actuation_feedback', 'gripper_control',
                         'position_control', 'effort_control', 'joint_position_control',
                         'trajectory_control', 'grasp_detection', 'release_detection',
                         'continuous_state')
HARDWARE_LAYER = 'trainit_hardware_layer'
EEF_ADAPTER_ROWS = {
    'trainit_mock': dict(
        name='TrainIt mock end-effector adapter', modes=('mock',), serves=EEF_INTERFACES,
        kinds=EEF_KINDS, provides=EEF_CAPABILITY_TOKENS, provides_if={},
        evidence=('sensor', 'sim_ground_truth', 'actuation_inference'),
        evidence_configurable=True, default_evidence='sensor', binding_keys=(), requires=(),
        flavours=(), package=HARDWARE_LAYER, executable='mock_eef_adapter', since='0.2.0',
        maturity='development',
        evidence_text='ADR-0022 D8: serves every actuator on its standard interface at the '
                      'canonical names; every capability and the evidence class are '
                      'configurable by node parameters'),
    'trainit_isaac_surface_gripper': dict(
        name='TrainIt Isaac SurfaceGripper adapter', modes=('isaac',), serves=('set_bool',),
        kinds=('suction',), provides=('binary_actuation', 'continuous_state'),
        provides_if={'stage_status_topic': ('grasp_detection', 'release_detection')},
        evidence=('sim_ground_truth',), evidence_configurable=False, default_evidence=None,
        binding_keys=('stage_cmd_topic', 'stage_status_topic'), requires=('stage_cmd_topic',),
        flavours=(), package=HARDWARE_LAYER, executable='isaac_eef_adapter', since='0.2.0',
        maturity='development',
        evidence_text='ADR-0022 D8 and EEF-D5: binary suction only; a grasp is observed only '
                      'with the stage status topic, never inferred from the command echo'),
    'ros2_control': dict(
        name='ros2_control GripperActionController (third-party)', modes=('gazebo', 'real'),
        serves=('gripper_command',), kinds=('parallel', 'composite'),
        provides=('gripper_control', 'position_control'),
        provides_if={'effort_flavour': ('effort_control',)}, evidence=(),
        evidence_configurable=False, default_evidence=None, binding_keys=('flavour',),
        requires=(), flavours=('position', 'effort'), package=None, executable=None, since=None,
        maturity='as_its_backend',
        evidence_text='ADR-0022 D6: the controller IS the adapter; TSA spawns it on gazebo, the '
                      'vendor bring-up provides it on real; it publishes no EndEffectorState, so '
                      'it detects nothing'),
    'none': dict(
        name='no end-effector server', modes=('mock', 'isaac', 'gazebo', 'real'), serves=(),
        kinds=EEF_KINDS, provides=(), provides_if={}, evidence=(), evidence_configurable=False,
        default_evidence=None, binding_keys=(), requires=(), flavours=(), package=None,
        executable=None, since=None, maturity='as_its_backend',
        evidence_text='ADR-0022 D6: allowed for a simple device whose require_* keys are both '
                      'false; the first end-effector command ends SERVER_MISSING'),
}

# Bridge rows a canonical project's deployment may carry, by what they are -- catalog DATA, so no
# TSA code names a reference application's executable (ADR-0022 R18). A row marked
# ``eef_legacy_bridge`` serves the legacy end-effector contract and is REFUSED next to a contract
# (the canonical adapter replaces it); a ``gripper_state_merger`` is kept, and its gripper input is
# where the adapter's pseudo-joint echo goes; a ``camera`` row is not an end-effector at all. Any
# other row active in a mode with a canonical binding gets a validate() warning naming it.
EEF_BRIDGE_ROLES = (
    dict(package='fr3wml_isaac', executable='isaac_suction_bridge.py', role='eef_legacy_bridge',
         why='the reference cell\'s Isaac suction bridge (the legacy Trigger pair and stage Bool)'),
    dict(package='fairino_gripper', executable='gripper_suction_action_server',
         role='eef_legacy_bridge', why='the FAIRINO GripperCommand-to-Trigger suction shim'),
    dict(package='fairino_gripper', executable='gripper_action_server',
         role='eef_legacy_bridge', why='the FAIRINO soft-gripper GripperCommand shim'),
    dict(package='fairino_gripper', executable='joint_state_merger', role='gripper_state_merger',
         gripper_input='gripper_joint_states_in', why='merges the gripper joint into joint_states'),
    dict(package='depth_image_proc', executable='point_cloud_xyzrgb_node', role='camera',
         why='the camera point cloud'),
    dict(package=HARDWARE_LAYER, executable='mock_camera', role='camera',
         why='the mock camera'),
)
# The legacy name of the suction pseudo-joint echo (hardware layer contracts/gripper.md), used
# when no joint-state merger names one: the adapter publishes the echo there, as the legacy bridge
# did (ADR-0022 D7: relative under a cell namespace).
EEF_PSEUDO_JOINT_TOPIC = 'suction/joint_states'


# --- trainit_perception's configuration vocabulary (ADR-0021, amendment M2.2) --------------------

TP_REGISTRY_FILE = 'tp_registry_snapshot.yaml'
TP_REGISTRY_SCHEMA = 1
TP_HEADER = (
    "# trainit_perception's configuration vocabulary, as this TSA release supports it (ADR-0021).\n"
    '# GENERATED by trainit_setup_assistant.catalog.refresh from\n'
    '# trainit_perception.vocabulary.export() -- do not edit by hand.\n')

_TP_KEYS = ('schema_version', 'trainit_perception', 'provides', 'detector_methods', 'auto',
            'rules', 'default_preference', 'terms', 'faces', 'default_face', 'face_size_band_m',
            'param_since')
_TP_KINDS = ('pure_python', 'external_node')
_TP_TYPES = ('str', 'int', 'float', 'range')
_TOKEN = re.compile(r'^[a-z][a-z0-9_]*$')
_RELEASE = re.compile(r'^\d+\.\d+\.\d+$')
# Parameters whose DEFAULT is a measured gate or a tool's own number: names may cross the boundary
# (a configurator writes them), values never do (ADR-0021, Decision 9).
_NO_DEFAULT = re.compile(r'^(pose_quality_|min_pose_quality|tool_)')

# What TSA's own code names in trainit_perception's vocabulary. TP must keep it, or TSA changes
# first: the refresh refuses, and the model says so at import (the M2.1 rule for planner modes).
#   methods   DetectionMethod is TSA code (ADR-0021 D6): every member but TSA's own `custom` is a TP
#             method of exactly this kind -- `emits_node` keys on the pure-python one -- and TP
#             may ship no method TSA does not handle yet, nor one named like TSA's own
#   rules     every rule the wizard lists by name (gui/wizard.py), and the `auto` token
#   face      the face a bundle without `grasp_face` gets, and the one parameter TSA derives
#             from a bounding box when nothing was measured (generator/perception_yaml.py)
#   dated     the parameters TSA writes itself and pins the runtime for: the ranking's
#             (FIXED_RANKING_PARAMS, the tool's numbers, the default face's size) and the noise
#             filters the wizard's form writes -- their floor was a TSA literal until vNext M2.2
#   form      the colour-mask parameters the wizard's form edits
TSA_TP_METHODS = {'color_mask': 'pure_python', 'pcl_cluster': 'external_node'}
TSA_OWN_METHODS = ('custom',)
TSA_REQUIRED_TP_RULES = ('detector_order', 'score', 'candidate_2d3d', 'candidate_x_visible',
                         'candidate_x_nonoverlap', 'candidate_x_clearance', 'candidate_x_square',
                         'candidate_x_visible_x_square', 'score_x_visible', 'score_x_nonoverlap',
                         'score_x_clearance')
TSA_AUTO = 'auto'
TSA_DEFAULT_FACE = ('disc', ('expected_radius_m',))
TSA_DATED_TP_PARAMS = ('candidate_rule', 'grasp_face', 'tool_radius_m', 'tool_reach_mm',
                       'expected_radius_m', 'blur_px', 'depth_min_m', 'depth_max_m')
TSA_COLOR_MASK_FORM = ('class_id', 'h', 's', 'v', 'min_area_px', 'max_objects', 'morph_kernel',
                       'depth_window_px', 'blur_px', 'depth_min_m', 'depth_max_m')

# The Community/Pro classification of every trainit_perception entry the catalog lists: an OWNER
# decision (ADR-0019 Decision 7), explicit since vNext M2.2 (owner answer C4) so a new TP entry is
# classified by a person and never by a default -- the refresh refuses an entry without a row, and
# a row for an entry TP no longer ships. Today's split, unchanged: the basic deterministic ranking
# is Community; every rule that ranks by the grasp-surface terms and every grasp face is the
# TrainIt surface grasp, Pro candidate IP; an external-node method is an extension point.
TP_TIERS = {
    'method': {'color_mask': 'community_candidate', 'pcl_cluster': 'extension_point'},
    'rule': {'detector_order': 'community_candidate', 'score': 'community_candidate',
             'candidate_2d3d': 'pro_candidate', 'candidate_x_visible': 'pro_candidate',
             'candidate_x_nonoverlap': 'pro_candidate', 'candidate_x_clearance': 'pro_candidate',
             'candidate_x_square': 'pro_candidate',
             'candidate_x_visible_x_square': 'pro_candidate', 'score_x_visible': 'pro_candidate',
             'score_x_nonoverlap': 'pro_candidate', 'score_x_clearance': 'pro_candidate'},
    'face': {'disc': 'pro_candidate', 'rect': 'pro_candidate', 'cylinder_side': 'pro_candidate'},
}
_TIER_SECTIONS = (('method', 'detector_methods'), ('rule', 'rules'), ('face', 'faces'))


def tp_registry_text(record: dict) -> str:
    """The canonical bytes of a TP vocabulary record: one serialisation, so a refresh is diffable
    and a re-serialised snapshot is byte-identical to the committed one."""
    return TP_HEADER + yaml.safe_dump(record, sort_keys=False, width=100)


def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _tokens(value) -> bool:
    return isinstance(value, list) and all(isinstance(v, str) and _TOKEN.match(v) for v in value)


def tp_registry_problems(record) -> List[str]:
    """Why ``record`` is not a TP vocabulary this TSA reads, as sentences. Empty = it is.

    A CLOSED schema: every level must have exactly its keys, every string be a name or a release,
    numbers sit only in a parameter's schema and the face size band (ADR-0021, Decision 9) -- an
    unknown key is refused, which is also how a schema TSA cannot read fails. Then the record
    must agree with itself: every term a rule multiplies, every output a term reads, every
    parameter a face or a term needs, the default face and the preference are declared. A record
    too malformed to inspect (a list where a mapping belongs, say) is one sentence, never an
    exception: the reader's answer is what the refresh acts on."""
    try:
        return _registry_problems(record)
    except Exception as exc:                                     # noqa: BLE001
        return [f'malformed ({type(exc).__name__}: {exc})']


def _registry_problems(record) -> List[str]:
    if not isinstance(record, dict):
        return ['not a mapping']
    out: List[str] = []
    if set(record) != set(_TP_KEYS):
        out.append(f'keys {sorted(set(record) ^ set(_TP_KEYS))} differ from schema '
                   f'{TP_REGISTRY_SCHEMA}')
        return out
    if not _RELEASE.match(str(record['trainit_perception'])):
        out.append(f'trainit_perception {record["trainit_perception"]!r} is not a release')
    provides = record['provides']
    if not _tokens(provides):
        out.append('provides is not a list of names')
        provides = []
    for name, method in _mapping(out, record, 'detector_methods').items():
        if not isinstance(method, dict) or set(method) != {'kind', 'provides', 'parameters'}:
            keys = sorted(method) if isinstance(method, dict) else method
            out.append(f'method {name!r}: keys {keys}')
            continue
        if method['kind'] not in _TP_KINDS:
            out.append(f'method {name!r}: kind {method["kind"]!r} not in {_TP_KINDS}')
        if method['kind'] == 'external_node':
            if method['provides'] is not None or method['parameters'] is not None:
                out.append(f'method {name!r}: an external node declares no outputs or parameters')
            continue
        if not _tokens(method['provides']) or not set(method['provides']) <= set(provides):
            out.append(f'method {name!r}: provides {method["provides"]!r} is not a list of '
                       f'declared outputs')
        for key, row in (method['parameters'] or {}).items():
            out += [f'method {name!r} parameter {key!r}: {p}'
                    for p in _parameter_problems(key, row)]
    terms = _mapping(out, record, 'terms')
    for name, req in terms.items():
        if not isinstance(req, dict) or set(req) != {'provides', 'positive_params', 'face_sized'}:
            out.append(f'term {name!r}: keys {sorted(req) if isinstance(req, dict) else req}')
            continue
        if not _tokens(req['provides']) or not set(req['provides']) <= set(provides):
            out.append(f'term {name!r}: provides {req["provides"]!r} is not declared outputs')
        if not _tokens(req['positive_params']) or not isinstance(req['face_sized'], bool):
            out.append(f'term {name!r}: positive_params/face_sized malformed')
    rules = _mapping(out, record, 'rules')
    for name, factors in rules.items():
        if not _tokens(factors) or not set(factors) <= set(terms):
            out.append(f'rule {name!r} multiplies {factors!r}, which are not all declared terms')
    if not _TOKEN.match(str(record['auto'])) or record['auto'] in rules:
        out.append(f'auto {record["auto"]!r} is not a name, or names a rule')
    if not _tokens(record['default_preference']) or not set(
            record['default_preference']) <= set(rules):
        out.append(f'default_preference {record["default_preference"]!r} names a rule that is '
                   f'not declared')
    floors = _mapping(out, record, 'param_since')
    for key, version in floors.items():
        if not isinstance(version, str) or not _RELEASE.match(version):
            out.append(f'param_since {key!r}: {version!r} is not a release')
    faces = _mapping(out, record, 'faces')
    for name, face in faces.items():
        if not isinstance(face, dict) or set(face) != {'params', 'enables_terms'}:
            out.append(f'face {name!r}: keys {sorted(face) if isinstance(face, dict) else face}')
            continue
        if not _tokens(face['params']) or not face['params'] or not set(face['params']) <= set(
                floors):
            out.append(f'face {name!r}: params {face["params"]!r} are not dated parameter names')
        if not _tokens(face['enables_terms']) or not all(
                terms.get(t, {}).get('face_sized') for t in face['enables_terms']):
            out.append(f'face {name!r}: enables {face["enables_terms"]!r}, not all face-sized '
                       f'terms')
    for name, req in terms.items():
        if isinstance(req, dict) and not set(req.get('positive_params') or ()) <= set(floors):
            out.append(f'term {name!r}: needs undated parameters {req["positive_params"]!r}')
    if record['default_face'] not in faces:
        out.append(f'default_face {record["default_face"]!r} is not a declared face')
    band = record['face_size_band_m']
    if not (isinstance(band, list) and len(band) == 2 and all(_is_number(b) for b in band)
            and 0.0 < band[0] < band[1]):
        out.append(f'face_size_band_m {band!r} is not [min, max] with 0 < min < max')
    return out


def _mapping(out: List[str], record: dict, key: str) -> dict:
    value = record[key]
    if not isinstance(value, dict) or not all(isinstance(k, str) and _TOKEN.match(k)
                                              for k in value):
        out.append(f'{key} is not a mapping of names')
        return {}
    return value


def _parameter_problems(key: str, row) -> List[str]:
    if not _TOKEN.match(str(key)):
        return ['not a name']
    if not isinstance(row, dict) or not {'type', 'default'} <= set(row) <= {
            'type', 'default', 'min', 'max', 'choices'}:
        return [f'keys {sorted(row) if isinstance(row, dict) else row}']
    out = []
    if row['type'] not in _TP_TYPES:
        out.append(f'type {row["type"]!r} not in {_TP_TYPES}')
    if _NO_DEFAULT.match(key):
        out.append('a gate or tool parameter carries a default across the boundary')
    default = row['default']
    if not (_is_number(default) or (isinstance(default, str) and _TOKEN.match(default))
            or (isinstance(default, list) and all(_is_number(v) for v in default))):
        out.append(f'default {default!r} is not a number, a list of numbers or a name')
    for bound in ('min', 'max'):
        if bound in row and not _is_number(row[bound]):
            out.append(f'{bound} {row[bound]!r} is not a number')
    if 'choices' in row and not _tokens(row['choices']):
        out.append(f'choices {row["choices"]!r} are not names')
    return out


def parse_tp_registry(data: bytes, where: str) -> dict:
    """A TP vocabulary record from its bytes, held to the schema this TSA reads. Anything else --
    not YAML at all (a merge conflict, say), another schema, a broken record -- is a RuntimeError
    that says how to repair it."""
    try:
        record = yaml.safe_load(data) or {}
    except (yaml.YAMLError, ValueError) as exc:     # ValueError: a constructor (a date, !!int)
        raise RuntimeError(f'{where}: not YAML ({str(exc).splitlines()[0]}). It is generated: '
                           f'run {REFRESH}') from None
    version = record.get('schema_version') if isinstance(record, dict) else None
    if version != TP_REGISTRY_SCHEMA:
        raise RuntimeError(
            f'{where}: trainit_perception vocabulary schema_version {version!r}, and this TSA '
            f'reads schema {TP_REGISTRY_SCHEMA}. A newer TP schema needs TSA\'s readers updated '
            'first (catalog/__init__.py); a snapshot that is merely stale is fixed by ' + REFRESH)
    problems = tp_registry_problems(record)
    if problems:
        raise RuntimeError(f'{where}: not a trainit_perception vocabulary this TSA can read -- '
                           + '; '.join(problems[:12]) + f'. Fix it at its source, then {REFRESH}')
    return record


def tp_vocabulary_problems(record: dict) -> List[str]:
    """What in a (schema-valid) TP vocabulary TSA's own code cannot work with. Empty = nothing."""
    out = []
    methods = record['detector_methods']
    for name, kind in TSA_TP_METHODS.items():
        got = methods.get(name, {}).get('kind')
        if got != kind:
            out.append(f'method {name!r} is {got or "absent"}, and TSA\'s DetectionMethod needs '
                       f'it {kind}')
    for name in methods:
        if name in TSA_OWN_METHODS:
            out.append(f'method {name!r} has the name of a method TSA owns')
        elif name not in TSA_TP_METHODS:
            out.append(f'method {name!r} is one TSA\'s DetectionMethod does not handle yet')
    if record['auto'] != TSA_AUTO:
        out.append(f'the token for "resolve the rule" is {record["auto"]!r}, TSA writes '
                   f'{TSA_AUTO!r}')
    out += [f'rule {r!r} is gone, and TSA\'s code names it' for r in TSA_REQUIRED_TP_RULES
            if r not in record['rules']]
    face, params = TSA_DEFAULT_FACE
    declared = record['default_face']
    if declared != face or record['faces'].get(face, {}).get('params') != list(params):
        out.append(f'the default face is {declared!r} sized by '
                   f'{record["faces"].get(declared, {}).get("params")!r}; TSA writes {face!r} '
                   f'sized by {list(params)!r}')
    if not record['default_preference']:
        out.append('the default rule preference is empty, and TSA pre-selects its first rule')
    out += [f'parameter {k!r} has no release floor, and TSA writes it' for k in TSA_DATED_TP_PARAMS
            if k not in record['param_since']]
    form = (methods.get('color_mask') or {}).get('parameters') or {}
    out += [f'colour-mask parameter {k!r} is gone, and the wizard\'s form edits it'
            for k in TSA_COLOR_MASK_FORM if k not in form]
    return out


def tp_tier_problems(record: dict, tiers: Optional[dict] = None) -> List[str]:
    """Every TP entry without a Community/Pro row, and every row for an entry TP does not ship."""
    tiers = TP_TIERS if tiers is None else tiers
    out = []
    for kind, section in _TIER_SECTIONS:
        exported, classified = list(record[section]), tiers.get(kind, {})
        out += [f'TP registry drift: {kind} {name!r} exists in TP but has no Community/Pro '
                f'classification in TSA\'s catalog (owner decision, ADR-0019 Decision 7): '
                f'classify it in catalog/__init__.py TP_TIERS' for name in exported
                if name not in classified]
        out += [f'TP registry drift: TSA\'s catalog classifies {kind} {name!r}, which TP does '
                f'not ship' for name in classified if name not in exported]
        out += [f'{kind} {name!r}: tier {tier!r} not in {DISTRIBUTION}'
                for name, tier in classified.items() if tier not in DISTRIBUTION]
    return out


def tp_drift(snapshot: Optional[dict], live: dict) -> List[str]:
    """How TP's live vocabulary differs from TSA's snapshot of it, one sentence per entry.

    The sentence says WHAT is out of sync: "TP registry drift: rule 'X' exists in TP but is absent
    from TSA snapshot", "... is in TSA snapshot but TP no longer exports it", a term's
    requirement, a parameter's default or allowed values, a face's parameters (their ORDER is
    what a generator writes), a floor, the version. Empty = in sync."""
    if snapshot is None:
        return ['TP registry drift: TSA has no snapshot of trainit_perception\'s vocabulary']
    out: List[str] = []
    for key in ('schema_version', 'trainit_perception'):
        if snapshot.get(key) != live.get(key):
            out.append(f'TP registry drift: {key} is {live.get(key)!r} in TP but '
                       f'{snapshot.get(key)!r} in TSA snapshot')
    for section, what in (('detector_methods', 'method'), ('rules', 'rule'), ('terms', 'term'),
                          ('faces', 'face'), ('param_since', 'parameter floor')):
        old, new = snapshot.get(section) or {}, live.get(section) or {}
        gone = [k for k in old if k not in new]
        came = [k for k in new if k not in old]
        out += [f'TP registry drift: {what} {k!r} exists in TP but is absent from TSA snapshot'
                for k in came]
        out += [f'TP registry drift: {what} {k!r} is in TSA snapshot but TP no longer exports it'
                + ''.join(f' (renamed {n!r}?)' for n in came if new[n] == old[k])
                for k in gone]
        common_new, common_old = [k for k in new if k in old], [k for k in old if k in new]
        if common_new != common_old:
            # the ORDER is vocabulary too: CandidateRule follows the rules' order, and with it
            # every refusal that lists them; a face's parameters are written in theirs
            out.append(f'TP registry drift: the {what}s are listed in another order in TP '
                       f'({common_new}) than in TSA snapshot ({common_old})')
        for k in new:
            if k in old and _ordered(new[k]) != _ordered(old[k]):
                out += _entry_drift(what, k, old[k], new[k])
    for key in ('provides', 'auto', 'default_preference', 'default_face', 'face_size_band_m'):
        if snapshot.get(key) != live.get(key):
            out.append(f'TP registry drift: {key} is {live.get(key)!r} in TP but '
                       f'{snapshot.get(key)!r} in TSA snapshot')
    return out


def _ordered(value):
    """A value compared WITH its mappings' key order, which dict equality ignores."""
    if isinstance(value, dict):
        return [(k, _ordered(v)) for k, v in value.items()]
    if isinstance(value, list):
        return [_ordered(v) for v in value]
    return value


def _entry_drift(what: str, name: str, old, new) -> List[str]:
    if what == 'method' and isinstance(old, dict) and isinstance(new, dict):
        out = [f'TP registry drift: method {name!r} {k} is {new.get(k)!r} in TP but '
               f'{old.get(k)!r} in TSA snapshot' for k in ('kind', 'provides')
               if old.get(k) != new.get(k)]
        oldp, newp = old.get('parameters') or {}, new.get('parameters') or {}
        out += [f'TP registry drift: parameter {name}.{p} exists in TP but is absent from TSA '
                f'snapshot' for p in newp if p not in oldp]
        out += [f'TP registry drift: parameter {name}.{p} is in TSA snapshot but TP no longer '
                f'exports it' for p in oldp if p not in newp]
        common_new, common_old = [k for k in newp if k in oldp], [k for k in oldp if k in newp]
        if common_new != common_old:
            out.append(f'TP registry drift: method {name!r} parameters are listed in another '
                       f'order in TP ({common_new}) than in TSA snapshot ({common_old})')
        for p in newp:
            if p in oldp and newp[p] != oldp[p]:
                for field in ('type', 'default', 'min', 'max', 'choices'):
                    if newp[p].get(field) != oldp[p].get(field):
                        label = 'default' if field == 'default' else (
                            'allowed values (' + field + ')' if field != 'type' else 'type')
                        out.append(f'TP registry drift: parameter {name}.{p} {label} is '
                                   f'{newp[p].get(field)!r} in TP but {oldp[p].get(field)!r} '
                                   f'in TSA snapshot')
        return out
    if isinstance(old, dict) and isinstance(new, dict):
        return [f'TP registry drift: {what} {name!r} {k} is {new.get(k)!r} in TP but '
                f'{old.get(k)!r} in TSA snapshot' for k in new if new.get(k) != old.get(k)]
    verb = {'rule': 'multiplies', 'parameter floor': 'arrived in'}.get(what, 'is')
    return [f'TP registry drift: {what} {name!r} {verb} {new!r} in TP but {old!r} in TSA '
            f'snapshot']


@functools.lru_cache(maxsize=None)
def _tp_snapshot() -> dict:
    data = _committed(TP_REGISTRY_FILE)
    _PARSED.setdefault(TP_REGISTRY_FILE, data)
    return parse_tp_registry(data, f'catalog/{TP_REGISTRY_FILE}')


def tp_registry() -> dict:
    """trainit_perception's vocabulary as TSA committed it (ADR-0021): never the installed TP."""
    return copy.deepcopy(_tp_snapshot())


def snapshot() -> dict:
    """The committed in-tree catalog (providers, adapters, the registries they came from)."""
    return _load(SNAPSHOT_FILE)


def _parsed(name: str) -> bytes:
    """The bytes the model was built from: read by its cached reader (first use triggers it).
    A reader replaced by a stand-in reads nothing: the committed bytes answer then."""
    if name not in _PARSED:
        {REGISTRY_FILE: _registry_copy, TP_REGISTRY_FILE: _tp_snapshot}[name]()
    return _PARSED[name] if name in _PARSED else _committed(name)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def snapshot_identity(release: Optional[str] = None) -> dict:
    """The identity of the capability snapshot this TSA compiles with, as a bundle records it.

    vNext M2.3 (ADR-0021): versions and sha256 hashes of TSA's COMMITTED catalog files, never their
    content and never what is installed -- ``catalog_sha256`` over ``snapshot.yaml``, and for each
    component registry TSA keeps a copy of, its release (as ``snapshot.yaml`` records it), its
    schema and the sha256 of the bytes the model PARSED. Every hash is over the exact bytes, read
    as bytes: any byte -- a comment, a newline -- changes the identity, by design.

    The identity must describe what this process compiles with, or it would be a lie. A
    RuntimeError that says how to repair it, before any file is written, when:

    * ``snapshot.yaml`` cannot be read, or changed on disk since this process first used it
      (it is pinned at its first identity that passed every check);
    * the sha256, schema or release ``snapshot.yaml`` records for a registry is not the one of
      the bytes the model was built from (changed outside the refresh, a merge, a CRLF checkout,
      or a refresh while TSA was running);
    * ``release`` is given (the generator's: the running TSA) and is not the release that
      derived ``snapshot.yaml`` (TSA updated while running, or a version bump not refreshed).
    """
    data = _committed(SNAPSHOT_FILE)
    try:
        catalog = yaml.safe_load(data)
        registries = catalog['registries']
        tmr = registries['trainit_motion_runtime']['planner_registry']
        tp = registries['trainit_perception']
        identity = {
            'catalog_version': catalog['catalog_version'],
            'catalog_sha256': _sha256(data),
            'trainit_motion_runtime': {
                'version': catalog['derived_with']['trainit_motion_runtime'],
                'planner_registry_schema': tmr['schema_version'],
                'planner_registry_sha256': _sha256(_parsed(REGISTRY_FILE))},
            'trainit_perception': {
                'version': tp['version'],
                'vocabulary_schema': tp['vocabulary']['schema_version'],
                'vocabulary_sha256': _sha256(_parsed(TP_REGISTRY_FILE))},
        }
        recorded = {REGISTRY_FILE: tmr, TP_REGISTRY_FILE: tp['vocabulary']}
    except (yaml.YAMLError, ValueError, KeyError, TypeError) as exc:
        reason = (str(exc).splitlines() or [repr(exc)])[0]
        raise RuntimeError(f'catalog/{SNAPSHOT_FILE} cannot be read as the catalog snapshot '
                           f'({type(exc).__name__}: {reason}): it is generated -- restore it with '
                           f'git, or run {REFRESH}') from None
    if _PARSED.get(SNAPSHOT_FILE, data) != data:
        raise RuntimeError(f'catalog/{SNAPSHOT_FILE} changed on disk while this TSA was running: '
                           f'start TSA again (a bundle must name the catalog it was compiled with)')
    for name, record in recorded.items():
        parsed = _parsed(name)
        actual = _sha256(parsed)
        if record.get('file') != name or record.get('sha256') != actual:
            crlf = (' Its bytes have CRLF line endings: the checkout converted them (git '
                    'core.autocrlf); check it out with LF.') if b'\r\n' in parsed else ''
            raise RuntimeError(
                f'catalog/{name} is not the file catalog/{SNAPSHOT_FILE} records (sha256 '
                f'{actual[:12]}..., recorded {str(record.get("sha256"))[:12]}...): the catalog was '
                f'changed outside the refresh, or refreshed while this TSA was running -- restore '
                f'it with git or run {REFRESH}, then start TSA again.{crlf}')
    versions = [identity[c]['version'] for c in ('trainit_motion_runtime', 'trainit_perception')]
    numbers = [identity['catalog_version'], identity['trainit_motion_runtime'][
        'planner_registry_schema'], identity['trainit_perception']['vocabulary_schema']]
    if not all(isinstance(v, str) and _RELEASE.fullmatch(v) for v in versions) or not all(
            isinstance(n, int) and not isinstance(n, bool) for n in numbers):
        raise RuntimeError(f'catalog/{SNAPSHOT_FILE} records no usable versions or schemas '
                           f'({versions}, {numbers}): it is generated -- restore it with git, or '
                           f'run {REFRESH}')
    # the labels snapshot.yaml records must be the parsed files' own (a hand edit, a bad merge)
    tmr_doc = yaml.safe_load(_parsed(REGISTRY_FILE)) or {}
    tp_doc = yaml.safe_load(_parsed(TP_REGISTRY_FILE)) or {}
    for label, recorded_value, own in (
            ('the planner registry schema', identity['trainit_motion_runtime'][
                'planner_registry_schema'], tmr_doc.get('schema_version')),
            ('the vocabulary schema', identity['trainit_perception']['vocabulary_schema'],
             tp_doc.get('schema_version')),
            ('the trainit_perception release', identity['trainit_perception']['version'],
             tp_doc.get('trainit_perception'))):
        if recorded_value != own:
            raise RuntimeError(f'catalog/{SNAPSHOT_FILE} records {label} {recorded_value!r}, and '
                               f'the file it names says {own!r}: it was changed outside the '
                               f'refresh -- restore it with git, or run {REFRESH}')
    derived_by = (catalog.get('derived_with') or {}).get('trainit_setup_assistant')
    if release is not None and derived_by != release:
        raise RuntimeError(f'catalog/{SNAPSHOT_FILE} was derived by TSA {derived_by}, and this is '
                           f'TSA {release}: TSA changed while it was running (start it again), or '
                           f'a version change was not followed by {REFRESH}')
    _PARSED.setdefault(SNAPSHOT_FILE, data)     # pinned only once it passed every check
    return identity


def candidates() -> List[dict]:
    """External candidate providers, with provenance. Never installed by TrainIt."""
    return list(_load('candidates.yaml')['candidates'])


def providers(family: Optional[str] = None) -> List[dict]:
    """Every provider row, in-tree then candidates; optionally one family."""
    rows = list(snapshot()['providers']) + candidates()
    return [r for r in rows if family is None or r['family'] == family]


def adapters() -> List[dict]:
    """Robot, EEF and simulator/backend adapters. NOT providers."""
    return list(snapshot()['adapters'])


def problems(rows: Optional[List[dict]] = None) -> List[str]:
    """What is wrong with the catalog, as sentences. Empty = valid."""
    out: List[str] = []
    rows = providers() if rows is None else rows
    seen: Dict[str, int] = {}
    for r in rows:
        rid = r.get('id', '<no id>')
        seen[rid] = seen.get(rid, 0) + 1
        for field, allowed in (('family', FAMILY), ('kind', KIND),
                               ('license_status', LICENSE_STATUS),
                               ('distribution', DISTRIBUTION)):
            if r.get(field) not in allowed:
                out.append(f'{rid}: {field} {r.get(field)!r} not in {allowed}')
        if r.get('kind') == 'candidate':
            if r.get('distribution') != 'external':
                out.append(f'{rid}: a candidate is external until it is integrated, not '
                           f'{r.get("distribution")!r}')
            if r.get('installed') or r.get('configured') or r.get('benchmarked'):
                out.append(f'{rid}: a candidate is never installed, configured or benchmarked '
                           f'by the catalog')
        if r.get('license_status') == 'VERIFIED':
            srcs = r.get('license_source')
            srcs = srcs if isinstance(srcs, list) else [srcs] if srcs else []
            if not srcs or not all(s.get('checked_on') and (s.get('url') or s.get('file'))
                                   for s in srcs):
                out.append(f'{rid}: VERIFIED needs a license_source with url or file and '
                           f'checked_on')
    out += [f'{rid}: duplicate id ({n}x)' for rid, n in seen.items() if n > 1]
    for a in adapters():
        if a.get('maturity') not in MATURITY:
            out.append(f'{a.get("id")}: maturity {a.get("maturity")!r} not in {MATURITY}')
        if any(a.get('id') == r.get('id') for r in rows):
            out.append(f'{a.get("id")}: an adapter is not a provider')
    return out
