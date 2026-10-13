"""What TrainIt HAS today, derived from the registries that already exist (vNext M1).

Nothing here is typed in twice. Each row is read from the place that owns the fact:

  TSA          its own enums: what a project can SELECT (DetectionMethod, CandidateRule,
               PolicyMode, ExecutionMode, Backend, SimGraspAdapter), and the end-effector
               adapter rows it types for the hardware layer (``catalog.EEF_ADAPTER_ROWS``,
               ADR-0022; ADR-0021's named exception M2.6-1)
  TMR          its installed planner registry ``config/planner_profiles.yaml`` (ADR-0021) and
               ``config/pipelines/``; the planner modes a project may select are
               ``catalog.offered_planner_modes`` of THAT registry -- never TSA's ``PlannerId``,
               which is built from the committed copy, so one refresh is a fixed point
  TP           its installed configuration vocabulary, ``trainit_perception.vocabulary.export()``
               (ADR-0021, amendment M2.2): detection methods, candidate rules, grasp faces, the
               version floors, the face size band -- the same record the refresh commits as
               ``tp_registry_snapshot.yaml``, so one refresh is a fixed point
  MoveIt       the installed planner and servo packages' ``package.xml`` (version, licence)
  the policy runtime, TMR, TP: their installed ``package.xml``

The only facts that are not derivable are the ADAPTERS' maturity and the provisional
Community/Pro classification (an owner decision, vNext approval 2026-10-06; for TP's entries an
explicit table, ``catalog.TP_TIERS``, owner answer C4 of vNext M2.2); they are constants, each with
the evidence it rests on.

``derive()`` is what ``snapshot.yaml`` is made from (``catalog.refresh`` writes it, handing it the
TMR registry and TP vocabulary it has just validated), and the drift test compares the two: when a
registry moves and the snapshot does not, it fails -- ``catalog_drift`` says which row and which
field. It READS the installed workspace and never writes anything; generation never imports it.
"""
from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .. import __version__ as TSA_VERSION
from ..model import enums as E
from . import (EEF_ADAPTER_ROWS, REGISTRY_FILE, TP_REGISTRY_FILE, TP_TIERS,
               declared_planner_modes, offered_planner_modes, parse_planner_registry,
               tp_registry_text)

CHECKED_ON = '2026-10-06'

# --- the provisional Community/Pro split (owner decision, vNext approval) ----------------------
# A PRODUCT classification, not a licence: "candidate" because nothing is decided until the
# owner splits the repositories (vNext M0B report, licence plan L1/L2).
COMMUNITY = 'community_candidate'
PRO = 'pro_candidate'
EXTENSION_POINT = 'extension_point'     # TrainIt supplies the seam, an application the implementation



def _tier(kind: str, name: str) -> str:
    """A TP entry's Community/Pro classification -- the owner's, from ``catalog.TP_TIERS``.

    Never a default: an unclassified entry stops the derivation (the refresh refuses it first,
    with the instruction)."""
    try:
        return TP_TIERS[kind][name]
    except KeyError:
        raise RuntimeError(f'{kind} {name!r} has no Community/Pro classification in '
                           f'catalog.TP_TIERS (owner decision, ADR-0019 Decision 7)') from None

# --- adapters: NOT providers (owner decision: the FR3WML interface is a robot adapter, the
# surface gripper an EEF adapter). Maturity is evidence, not a registry, so it is stated here.
_BACKEND_FACTS = {
    'mock': dict(maturity='verified',
                 evidence='TSA test_generated_bundle_launch.py (vNext M0D T2): a generated bundle '
                          'brought up, planning and executing on mock ros2_control, 2026-10-06'),
    'isaac': dict(maturity='verified_manually',
                  evidence='the reference cell, end to end in Isaac Sim through the GUI (M1/M2 '
                           'manual protocol); physics baseline fr3wml_isaac/scripts/'
                           'baseline_cell_physics.py (vNext M0D T3)'),
    'gazebo': dict(maturity='development',
                   evidence='ADR-0008 native Gazebo backend; a namespaced cell is unverified on '
                            'gazebo (generation warns)'),
    'real': dict(maturity='non_production_ready',
                 evidence='vNext M0D T4: as generated, mode:=real starts no hardware interface; '
                          'the FAIRINO bridge executes only a trajectory\'s last waypoint (R0); '
                          'never run end to end (owner classification DEVELOPMENT / '
                          'NON-PRODUCTION-READY)'),
}
_ROBOT_ADAPTERS = [dict(
    id='adapter.robot.fairino_fr3wml', adapter_type='robot', name='FAIRINO FR3WML bridge',
    packages=['fairino_bridge', 'fairino_gripper', 'frcobot_ros2'],
    serves=['moveit_joint_controller/follow_joint_trajectory',
            'suctioncup_controller/gripper_command'],
    modes=['real'], maturity='non_production_ready',
    evidence='TSA test_reference_real_mode_contract.py: the action names match MoveIt\'s; '
             'docs/TRAINIT_HARDWARE_BASELINE.md limits (last waypoint only, x86-64 only)')]


def _package_xml(package: str) -> Optional[Path]:
    try:
        from ament_index_python.packages import get_package_share_directory
        p = Path(get_package_share_directory(package)) / 'package.xml'
        return p if p.is_file() else None
    except Exception:                               # noqa: BLE001 - not installed / not sourced
        return None


def _package_facts(package: str) -> Dict[str, object]:
    """version + declared licence of an INSTALLED package, or installed: False."""
    p = _package_xml(package)
    if p is None:
        return dict(package=package, installed=False)
    root = ET.parse(p).getroot()
    return dict(package=package, installed=True, version=root.findtext('version'),
                license=', '.join(sorted(e.text.strip() for e in root.findall('license')
                                         if e.text)))


def _tmr_registry(raw: Optional[bytes] = None) -> dict:
    """TMR's INSTALLED planner registry, strictly (ADR-0021): a snapshot derived without it, or
    from another schema, would be wrong rather than partial, so this raises. ``raw`` is the
    registry the refresh has just read and validated, so the snapshot records exactly it."""
    from ament_index_python.packages import get_package_share_directory
    share = Path(get_package_share_directory('trainit_motion_runtime'))
    path = share / 'config' / 'planner_profiles.yaml'
    raw = path.read_bytes() if raw is None else raw
    registry = parse_planner_registry(raw, str(path))
    return dict(registry=registry, sha256=hashlib.sha256(raw).hexdigest(),
                pipeline_files=sorted(f.name for f in (share / 'config' / 'pipelines')
                                      .glob('*.yaml')))


def _motion_planning(tmr: dict) -> List[dict]:
    registry = tmr['registry']
    offered = set(offered_planner_modes(registry))
    rows = []
    for mode, spec in sorted(registry['planner_modes'].items()):
        pipeline = spec['pipeline']
        plugin = _package_facts(registry['pipelines'][pipeline]['plugin_package'])
        rows.append(dict(
            id=f'moveit.{mode}', family='motion_planning', kind='in_tree',
            integration='moveit_pipeline', name=f'MoveIt {mode} (TMR planner_mode)',
            vendor='MoveIt 2', tmr_planner_mode=mode, pipeline_id=pipeline,
            planner_id=spec.get('planner'), optimizer=spec.get('optimizer_pipeline'),
            tsa_selectable=mode in offered, plugin=plugin,
            license_status='VERIFIED' if plugin.get('installed') else 'UNKNOWN',
            license_source=dict(file='package.xml (installed)', package=plugin['package'],
                                version=plugin.get('version'), checked_on=CHECKED_ON),
            distribution=COMMUNITY, requirements=dict(gpu=None, ros_distro='humble')))
    return rows


def _servo() -> dict:
    facts = _package_facts('moveit_servo')
    return dict(
        id='moveit.servo', family='motion_execution', kind='in_tree', integration='ros_node',
        name='MoveIt Servo (execution: servo, ADR-0015)', vendor='MoveIt 2',
        tsa_selectable='servo' in {m.value for m in E.ExecutionMode}, plugin=facts,
        license_status='VERIFIED' if facts.get('installed') else 'UNKNOWN',
        license_source=dict(file='package.xml (installed)', package='moveit_servo',
                            version=facts.get('version'), checked_on=CHECKED_ON),
        distribution=COMMUNITY, requirements=dict(gpu=None, ros_distro='humble'))


def _trainit_license(package: str) -> dict:
    facts = _package_facts(package)
    return dict(license_status='REQUIRES_REVIEW',
                license_source=dict(file='package.xml (installed)', package=package,
                                    version=facts.get('version'),
                                    declared=facts.get('license'),
                                    note='declared only: the repository has no LICENSE file '
                                         '(vNext M0B licence plan L1)',
                                    checked_on=CHECKED_ON))


def _perception(tp: dict) -> List[dict]:
    rows = []
    selectable = {m.value for m in E.DetectionMethod}
    for method in tp.get('in_process_methods', []):
        rows.append(dict(
            id=f'tp.detector.{method}', family='perception_detection', kind='in_tree',
            integration='tp_in_process', name=f'trainit_perception {method}',
            vendor='TrainIt', tsa_selectable=method in selectable,
            distribution=_tier('method', method), requirements=dict(gpu=None),
            **_trainit_license('trainit_perception')))
    for method in tp.get('external_methods', []):
        rows.append(dict(
            id=f'tp.detector.{method}', family='perception_detection', kind='in_tree',
            integration='external_node', name=f'external-node method {method}',
            vendor='the application', tsa_selectable=method in selectable,
            distribution=_tier('method', method), requirements=dict(gpu=None),
            **_trainit_license('trainit_perception')))
    # A method TSA lets a project select that TP's registry does not list (``custom``): an
    # external node by construction. Shown, not hidden -- it is a vocabulary drift (M2).
    known = set(tp.get('in_process_methods', [])) | set(tp.get('external_methods', []))
    for method in sorted(selectable - known) if tp else []:
        rows.append(dict(
            id=f'tsa.detector.{method}', family='perception_detection', kind='in_tree',
            integration='external_node', name=f'external-node method {method} (TSA only, '
                                               'not in the trainit_perception registry)',
            vendor='the application', tsa_selectable=True, distribution=EXTENSION_POINT,
            requirements=dict(gpu=None), **_trainit_license('trainit_setup_assistant')))
    return rows


def _grasp(tp: dict, selectable: List[str]) -> List[dict]:
    # A rule is selectable iff it is a CandidateRule -- and CandidateRule is built from the
    # vocabulary this record is derived from (ADR-0021, M2.2), so `selectable` comes from that
    # vocabulary and never from the enum the running process happened to import: one refresh is a
    # fixed point.
    rows = []
    for rule in tp.get('rules', []):
        rows.append(dict(
            id=f'tp.rule.{rule}', family='grasp_ranking', kind='in_tree', integration='tp_rule',
            name=f'candidate rule {rule}', vendor='TrainIt', tsa_selectable=rule in selectable,
            distribution=_tier('rule', rule),
            requirements=dict(gpu=None), **_trainit_license('trainit_perception')))
    for face in tp.get('faces', []):
        rows.append(dict(
            id=f'tp.face.{face}', family='grasp_ranking', kind='in_tree', integration='tp_face',
            name=f'grasp face {face} (TrainIt surface grasp)', vendor='TrainIt',
            tsa_selectable=True, distribution=_tier('face', face), requirements=dict(gpu=None),
            **_trainit_license('trainit_perception')))
    return rows


def _scene(tmr_facts: dict) -> dict:
    return dict(
        id='tmr.scene_manager', family='scene', kind='in_tree', integration='ros_node',
        name='scene_manager_node (scene.yaml into the MoveIt planning scene)', vendor='TrainIt',
        tsa_selectable=True, distribution=COMMUNITY, requirements=dict(gpu=None),
        **_trainit_license('trainit_motion_runtime'))


def _policy() -> List[dict]:
    return [dict(
        id=f'policy.{mode.value}', family='policy_execution', kind='in_tree',
        integration='ros_node', name=f'trainit_policy_runtime {mode.value} (ADR-0005)',
        vendor='TrainIt', tsa_selectable=True, distribution=PRO,
        requirements=dict(gpu=None), **_trainit_license('trainit_policy_runtime'))
        for mode in E.PolicyMode]


def installed_tp_vocabulary() -> dict:
    """trainit_perception's INSTALLED configuration vocabulary, strictly (ADR-0021, M2.2): a
    snapshot derived without it would be wrong rather than partial, so this raises."""
    try:
        from trainit_perception.vocabulary import export
    except Exception as exc:                        # noqa: BLE001 - not installed / not sourced
        raise RuntimeError(
            'trainit_perception.vocabulary is not importable here: source the workspace that '
            f'installs trainit_perception 0.5.0 or newer ({exc})') from exc
    return export()


def _tp_record(vocabulary: dict) -> dict:
    """The catalog's record of trainit_perception, from its vocabulary. The sha256 is of the
    canonical text the refresh commits, so the catalog says which snapshot it was made with."""
    methods = vocabulary['detector_methods']
    return dict(
        version=vocabulary['trainit_perception'],
        vocabulary=dict(file=TP_REGISTRY_FILE,
                        sha256=hashlib.sha256(tp_registry_text(vocabulary).encode()).hexdigest(),
                        schema_version=vocabulary['schema_version']),
        in_process_methods=sorted(m for m, s in methods.items() if s['kind'] == 'pure_python'),
        external_methods=sorted(m for m, s in methods.items() if s['kind'] == 'external_node'),
        rules=sorted(vocabulary['rules']), faces=sorted(vocabulary['faces']),
        param_since=dict(sorted(vocabulary['param_since'].items())),
        face_size_band_m=[float(b) for b in vocabulary['face_size_band_m']])


def _eef_adapter_row(token: str, row: dict) -> dict:
    """One end-effector IMPLEMENTATION row (ADR-0022 D6) as the snapshot lists it: the facts of
    ``catalog.EEF_ADAPTER_ROWS`` (TSA's named exception M2.6-1 to ADR-0021), never a kind."""
    out = dict(id=f'adapter.eef.{token}', adapter_type='eef', name=row['name'],
               modes=list(row['modes']), serves=list(row['serves']), kinds=list(row['kinds']),
               provides=list(row['provides']),
               provides_if={k: list(v) for k, v in row['provides_if'].items()},
               evidence_classes=list(row['evidence']),
               evidence_configurable=row['evidence_configurable'])
    if row['package']:
        out.update(package=row['package'], executable=row['executable'], since=row['since'])
    if row['flavours']:
        out['flavours'] = list(row['flavours'])
    out.update(maturity=row['maturity'], evidence=row['evidence_text'])
    return out


def _adapters() -> List[dict]:
    rows = [dict(id=f'adapter.backend.{b.value}', adapter_type='simulator_or_backend',
                 name=f'backend {b.value}', modes=[b.value], **_BACKEND_FACTS[b.value])
            for b in E.Backend]
    # A KIND is not an adapter (ADR-0022, amending ADR-0019 Decision 2): the EEF rows are the
    # implementations that serve the contract, so a new kind (``composite``) adds no row.
    rows += [_eef_adapter_row(token, row) for token, row in EEF_ADAPTER_ROWS.items()]
    # The simulator-physics tricks, each on the ONE backend it exists on (ADR-0022 D6 corrects
    # the earlier rows, which listed both on isaac AND gazebo): the per-backend default map is the
    # source, so the row and the generator's default can never disagree.
    sim_modes: Dict[str, List[str]] = {}
    for backend, adapter in E.DEFAULT_SIM_GRASP_ADAPTER.items():
        if adapter is not E.SimGraspAdapter.NONE:
            sim_modes.setdefault(adapter.value, []).append(backend.value)
    rows += [dict(id=f'adapter.sim_grasp.{a.value}', adapter_type='eef_simulation',
                  name=f'simulated grasp {a.value}',
                  modes=[b.value for b in E.Backend if b.value in sim_modes.get(a.value, [])],
                  maturity='as_its_backend', evidence='TSA SimGraspAdapter (ADR-0010)')
             for a in E.SimGraspAdapter if a.value != 'none']
    return rows + [dict(r) for r in _ROBOT_ADAPTERS]


def derive(tmr_bytes: Optional[bytes] = None, tp_vocabulary: Optional[dict] = None) -> dict:
    """The in-tree part of the catalog, as the installed workspace reports it.

    The refresh passes the TMR registry and the TP vocabulary it has just read and validated, so
    the record describes exactly the files it is written with; without them, both are read from
    the installed workspace (the drift test)."""
    tmr = _tmr_registry(tmr_bytes)
    vocabulary = installed_tp_vocabulary() if tp_vocabulary is None else tp_vocabulary
    tp = _tp_record(vocabulary)
    selectable_rules = [vocabulary['auto']] + list(vocabulary['rules'])
    providers = (_motion_planning(tmr) + [_servo()] + _perception(tp)
                 + _grasp(tp, selectable_rules) + [_scene(tmr)] + _policy())
    return dict(
        catalog_version=1,
        derived_with=dict(trainit_setup_assistant=TSA_VERSION,
                          trainit_motion_runtime=_package_facts(
                              'trainit_motion_runtime').get('version'),
                          trainit_perception=tp.get('version'),
                          trainit_policy_runtime=_package_facts(
                              'trainit_policy_runtime').get('version')),
        providers=providers,
        adapters=_adapters(),
        registries=dict(
            tsa_selectable=dict(
                planner=offered_planner_modes(tmr['registry']),
                detection_method=[m.value for m in E.DetectionMethod],
                candidate_rule=selectable_rules,
                execution=[m.value for m in E.ExecutionMode],
                policy_mode=[m.value for m in E.PolicyMode],
                backend=[m.value for m in E.Backend]),
            trainit_motion_runtime=dict(
                planner_registry=dict(file=REGISTRY_FILE, sha256=tmr['sha256'],
                                      schema_version=tmr['registry']['schema_version']),
                planner_modes=declared_planner_modes(tmr['registry']),
                pipeline_files=tmr['pipeline_files']),
            trainit_perception=tp))


def catalog_drift(committed: dict, derived: dict) -> List[str]:
    """How a derived catalog record differs from the committed one, one sentence per difference:
    which row, which field, which values -- a changed Community/Pro classification included
    ("catalog drift: tp.rule.score distribution is 'community_candidate' in snapshot.yaml but
    'pro_candidate' from derive()"). Empty = identical."""
    out: List[str] = []
    for section in ('providers', 'adapters'):
        old = {r['id']: r for r in committed.get(section) or []}
        new = {r['id']: r for r in derived.get(section) or []}
        out += [f'catalog drift: {i} is derived but not in snapshot.yaml' for i in new
                if i not in old]
        out += [f'catalog drift: {i} is in snapshot.yaml but no longer derived' for i in old
                if i not in new]
        for i in new:
            if i in old and new[i] != old[i]:
                out += [f'catalog drift: {i} {k} is {old[i].get(k)!r} in snapshot.yaml but '
                        f'{new[i].get(k)!r} from derive()'
                        for k in sorted(set(old[i]) | set(new[i]))
                        if old[i].get(k) != new[i].get(k)]
        if list(old) != list(new) and set(old) == set(new):
            out.append(f'catalog drift: the {section} are listed in another order')
    for key in sorted((set(committed) | set(derived)) - {'providers', 'adapters'}):
        out += _leaf_drift(key, committed.get(key), derived.get(key))
    return out


def _leaf_drift(path: str, old, new) -> List[str]:
    """One sentence per changed leaf of a nested mapping, named by its dotted path."""
    if isinstance(old, dict) and isinstance(new, dict):
        out = []
        for k in list(old) + [k for k in new if k not in old]:
            out += _leaf_drift(f'{path}.{k}', old.get(k), new.get(k))
        return out
    if old != new:
        return [f'catalog drift: {path} is {old!r} in snapshot.yaml but {new!r} from derive()']
    return []
