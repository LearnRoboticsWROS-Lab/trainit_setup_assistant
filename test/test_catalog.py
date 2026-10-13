"""The provider catalog: DATA, read-only, and true to the registries it was derived from (vNext M1).

What this pins:

1. the catalog is VALID: closed vocabularies (family, kind, licence status, distribution), no
   duplicate ids, every VERIFIED licence claim with a source and the date it was read;
2. the snapshot is what the registries SAY -- ``derive()`` against the installed workspace equals
   the committed ``snapshot.yaml``; when a registry moves and the snapshot does not, this fails;
3. adapters are NOT providers, and every candidate is external metadata: never installed,
   configured or benchmarked, never a Community dependency;
4. generation reads ONLY what is committed (ADR-0021, which replaces the vNext M1 rule "the
   generator never reads the catalog"): of the catalog, exactly TSA's committed copies of the
   component registries and -- since vNext M2.3, for the identity every bundle records -- the
   bytes of ``snapshot.yaml``; nothing that depends on the machine -- never ``catalog.derive`` or
   ``catalog.refresh``, never the installed TMR's ``config/``, never the candidate catalog (the
   full-tree golden test is the byte half).
"""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from trainit_setup_assistant import catalog
from trainit_setup_assistant.catalog.derive import derive

import ws_paths

PKG = Path(__file__).resolve().parents[1]


def _installed(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:                                       # noqa: BLE001
        return False


def test_the_catalog_is_valid():
    assert catalog.problems() == []


@pytest.mark.skipif(not (_installed('trainit_perception.ranking')
                         and _installed('ament_index_python')),
                    reason='needs the workspace installed (trainit_perception, TMR, MoveIt)')
def test_the_snapshot_is_what_the_registries_say():
    derived, committed = derive(), catalog.snapshot()
    if derived['derived_with'] != committed['derived_with']:
        pytest.fail(f'the workspace moved: {derived["derived_with"]} vs the snapshot\'s '
                    f'{committed["derived_with"]} -- regenerate snapshot.yaml from derive()')
    from trainit_setup_assistant.catalog.derive import catalog_drift
    assert derived == committed, '\n'.join(catalog_drift(committed, derived))


def test_adapters_are_not_providers():
    provider_ids = {r['id'] for r in catalog.providers()}
    adapter_ids = {a['id'] for a in catalog.adapters()}
    assert not provider_ids & adapter_ids
    assert 'adapter.robot.fairino_fr3wml' in adapter_ids
    real = next(a for a in catalog.adapters() if a['id'] == 'adapter.backend.real')
    assert real['maturity'] == 'non_production_ready'


def test_every_candidate_is_external_metadata_and_never_community():
    cands = catalog.candidates()
    assert cands and all(c['kind'] == 'candidate' and c['distribution'] == 'external'
                         for c in cands)
    assert not any(c['installed'] or c['configured'] or c['benchmarked'] for c in cands)
    assert not [c for c in cands if c['trainit_class'] == 'community_candidate']


def test_the_licence_facts_that_decide_the_nvidia_plan():
    """The review's conclusions, as the catalog holds them (owner-approved, vNext 2026-10-06)."""
    by = {c['id']: c for c in catalog.candidates()}
    humble_cumotion = by['nvidia.cumotion_isaac_ros_3_2']
    assert (humble_cumotion['evaluation_use'], humble_cumotion['commercial_use']) == (
        'allowed', 'not_allowed')
    assert by['nvidia.cumotion_isaac_ros_4_4']['humble_available'] is False
    assert by['nvidia.curobo_v2']['commercial_use'] == 'allowed'
    assert by['nvidia.gr00t_n1_7']['license_status'] == 'REQUIRES_REVIEW'
    assert by['nvidia.rtdetr_syntheticadetr']['license_status'] == 'UNKNOWN'


def test_the_snapshot_records_the_vocabulary_drifts_m2_will_close():
    """Facts the catalog made visible, not hid: TSA cannot select every TMR planner mode, and one
    TSA detection method is not in TP's registry. The TP rules TSA could not select are the drift
    vNext M2.2 CLOSED (ADR-0021, amendment M2.2): TSA's rule vocabulary is TP's, from its snapshot,
    so the row now says selectable -- the one assertion of this test M2.2 changes."""
    rows = {r['id']: r for r in catalog.snapshot()['providers']}
    assert rows['moveit.stomp']['tsa_selectable'] is False
    assert rows['moveit.stomp']['plugin']['installed'] is False     # not on this Humble
    assert rows['tp.rule.candidate_x_square']['tsa_selectable'] is True
    assert 'tsa.detector.custom' in rows


@pytest.mark.parametrize('project', [
    PKG / 'examples' / 'fr3wml_project.yaml',
    pytest.param(ws_paths.GOLDEN_BUNDLE / 'project.yaml', marks=pytest.mark.skipif(
        not ws_paths.GOLDEN_BUNDLE.is_dir(), reason='needs the golden project (sibling checkout)')),
], ids=['from_scratch', 'scene_loader'])
def test_generation_reads_only_the_committed_catalog(tmp_path, project):
    """ADR-0021: generation is a function of the project, the TSA release and what TSA commits.
    Every file opened and every module imported -- from the first import on -- by a generation in a
    fresh interpreter is recorded: under catalog/ it opens exactly the three committed snapshot
    files (an allow-list, M2.3), it imports neither the derive tool nor the refresh, and it opens no
    file of TMR's config/ (the source or an installed copy) and not the candidate catalog."""
    code = (
        'import sys\n'
        'opened = []\n'
        'def _audit(event, args):\n'
        '    if event == "open" and args and isinstance(args[0], (str, bytes)):\n'
        '        opened.append(args[0] if isinstance(args[0], str) else args[0].decode())\n'
        'sys.addaudithook(_audit)\n'
        f'sys.path.insert(0, {str(PKG / "test")!r})\n'
        'import conftest  # noqa: F401  (the same path rebasing as this session)\n'
        'from trainit_setup_assistant.generator import Orchestrator\n'
        'from trainit_setup_assistant.model import load_project\n'
        f'Orchestrator().generate(load_project({str(project)!r}),'
        f' {str(tmp_path / "out")!r})\n'
        'bad = [f for f in opened if "trainit_motion_runtime/config" in f.replace("\\\\", "/")'
        ' or f.endswith("candidates.yaml")]\n'
        'print("DERIVE_IMPORTED" if "trainit_setup_assistant.catalog.derive" in sys.modules'
        ' or "trainit_setup_assistant.catalog.refresh" in sys.modules else "DERIVE_NOT_IMPORTED")\n'
        'import os, trainit_setup_assistant.catalog as C\n'
        'here = os.path.dirname(os.path.realpath(C.__file__))\n'
        'print("CATALOG_FILES", sorted({os.path.basename(f) for f in opened'
        ' if os.path.dirname(os.path.realpath(f)) == here and not f.endswith((".py", ".pyc"))}))\n'
        'print("FORBIDDEN_FILES", bad)\n')
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                       cwd=str(ws_paths.WS_SRC))
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'DERIVE_NOT_IMPORTED' in r.stdout, r.stdout + r.stderr
    assert 'FORBIDDEN_FILES []' in r.stdout, r.stdout
    assert ("CATALOG_FILES ['snapshot.yaml', 'tmr_planner_registry.yaml', "
            "'tp_registry_snapshot.yaml']") in r.stdout, r.stdout
