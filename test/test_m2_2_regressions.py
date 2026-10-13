"""What vNext M2.2 changes, as tests that fail on 7.3.0 (ADR-0021, amendment M2.2).

Each test here states a behaviour M2.2 delivers. On 7.3.0 the ones that do not hold yet are
``xfail(strict=True)`` -- they must FAIL there, and an unexpected pass is an error -- and the
implementation removes the markers; nothing else in this file changes. The tests that already
hold on 7.3.0 carry no marker: generation with and without trainit_perception is already identical
for those cases, and must stay so.

1. ``edge_on``: the two TP rules TSA refused with a false message are vocabulary TSA accepts, an
   external node with a tilt limit can rank by them, and the colour mask refuses them for the TRUE
   reason (no 3D fit), not as "a rule no detector publishes".
2. The same project generates the same bundle -- or the same refusal -- with trainit_perception
   importable, with it made unimportable, and with it uninstalled (off ``PYTHONPATH`` and the
   ament index): generation must not query the installed TP.
3. Generation imports no trainit_perception module, opens no file of it, and asks the ament index
   nothing about it.
4. The catalog refresh writes all of its files or none: on 7.3.0 a refresh whose derive step fails
   has already rewritten the planner registry copy (the M2.1 half-refresh M2.2 fixes forward).
"""
import json
import os
import subprocess
import sys
import tempfile
import types
from pathlib import Path

import pytest
import yaml

import m2_2_cases
from trainit_setup_assistant.catalog import REGISTRY_FILE
from trainit_setup_assistant.catalog import refresh as refresh_tool
from trainit_setup_assistant.generator.perception_yaml import (build_perception_yaml,
                                                               refuse_uncomputable_candidate_rules)
from trainit_setup_assistant.model import DetectorSpec


PKG = Path(__file__).resolve().parents[1]
TEST = Path(__file__).resolve().parent
COPY = PKG / 'trainit_setup_assistant' / 'catalog' / REGISTRY_FILE
EDGE_RULES = ('candidate_x_square', 'candidate_x_visible_x_square')


# --- 1. edge_on ---------------------------------------------------------------------------------

@pytest.mark.parametrize('rule', EDGE_RULES)
def test_the_edge_on_rules_are_vocabulary_tsa_accepts(rule):
    assert DetectorSpec(name='x', candidate_rule=rule).candidate_rule.value == rule


def test_an_external_node_with_a_tilt_limit_ranks_by_candidate_x_square():
    p = m2_2_cases.project(method='custom', params={'tool_max_tilt_deg': m2_2_cases.TILT})
    p.perception.detectors[0] = DetectorSpec(
        name=m2_2_cases.DETECTOR, method='custom', candidate_rule='candidate_x_square',
        params={'tool_max_tilt_deg': m2_2_cases.TILT})
    refuse_uncomputable_candidate_rules(p)
    params = yaml.safe_load(build_perception_yaml(p))['detectors'][m2_2_cases.DETECTOR]['params']
    assert params['candidate_rule'] == 'candidate_x_square'


@pytest.mark.parametrize('rule', EDGE_RULES)
def test_the_colour_mask_refuses_an_edge_on_rule_for_the_true_reason(rule):
    p = m2_2_cases.project(params={'candidate_rule': rule,
                                   'tool_max_tilt_deg': m2_2_cases.TILT})
    with pytest.raises(ValueError) as exc:
        refuse_uncomputable_candidate_rules(p)
    message = str(exc.value)
    assert 'is not a rule any detector publishes' not in message
    assert 'pose_quality' in message and 'edge_on' in message, message


# --- 2. and 3. generation does not depend on, or query, the installed trainit_perception --------

_WORKER = r'''
import json, os, shutil, sys, tempfile, hashlib, traceback
mode, case, test_dir, pkg = sys.argv[1:5]
seen, opened, ament = [], [], []

class _TP:
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0].startswith('trainit_perception'):
            seen.append(name)
            if mode == 'blocked':
                raise ModuleNotFoundError(f'{name!r} made unimportable', name=name)
        return None

sys.meta_path.insert(0, _TP())
import importlib.util
try:                                # a real package, not a checkout seen as a namespace package
    spec = importlib.util.find_spec('trainit_perception')
    tp_importable = spec is not None and spec.origin not in (None, 'namespace')
except ModuleNotFoundError:
    tp_importable = False
seen.clear()                        # that lookup was this worker's, not TSA's

def _audit(event, args):
    # A DATA file of trainit_perception (a profile, its package.xml, anything in its share, its
    # release metadata). Not a module (the finder above sees those), and not the one file
    # pydantic's plugin loader reads from EVERY installed distribution the first time a model
    # validates -- entry_points.txt, a scan of the whole environment, not a question asked of TP.
    if event == 'open' and args and isinstance(args[0], (str, bytes)):
        p = args[0].decode() if isinstance(args[0], bytes) else args[0]
        if 'trainit_perception' not in p or p.endswith(('.py', '.pyc', '.so')):
            return
        if (p.endswith(('.egg-info/entry_points.txt', '.dist-info/entry_points.txt'))
                and any('/pydantic/' in f.filename for f in traceback.extract_stack())):
            return
        opened.append(p)

sys.addaudithook(_audit)
try:
    import ament_index_python.packages as _aip
    for _fn in ('get_package_share_directory', 'get_package_prefix'):
        _orig = getattr(_aip, _fn)
        def _wrap(name, *a, _orig=_orig, **k):
            if 'trainit_perception' in str(name):
                ament.append(str(name))
            return _orig(name, *a, **k)
        setattr(_aip, _fn, _wrap)
except Exception:
    pass
sys.path[:0] = [test_dir, pkg]
import conftest  # noqa: F401  (the project rebasing every test session uses)
import m2_2_cases
from trainit_setup_assistant.generator import Orchestrator
project = m2_2_cases.CASES[case]()
before = list(seen)
out = tempfile.mkdtemp()
result = {'tp_importable': tp_importable,
          'tp_origin': str(getattr(spec, 'origin', None)) if tp_importable else None}
try:
    Orchestrator().generate(project, out)
    files = {}
    for d, _, names in os.walk(out):
        for n in names:
            full = os.path.join(d, n)
            files[os.path.relpath(full, out)] = hashlib.sha256(open(full, 'rb').read()).hexdigest()
    result['files'] = dict(sorted(files.items()))
except ValueError as exc:
    result['refused'] = str(exc)
finally:
    shutil.rmtree(out, ignore_errors=True)    # a bundle is ~45 MB: never leave one behind
result['tp_lookups'] = sorted(set(seen))
result['tp_lookups_during_generation'] = sorted(set(seen[len(before):]))
result['tp_files_opened'] = sorted(set(opened))
result['ament_tp_lookups'] = sorted(set(ament))
print('RESULT ' + json.dumps(result))
'''


def _without_tp(paths):
    return os.pathsep.join(p for p in paths.split(os.pathsep)
                           if p and 'trainit_perception' not in p)


def _generate(mode, case):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1',
               PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    if mode == 'absent':            # uninstalled: off the interpreter's path AND the ament index
        env['PYTHONPATH'] = _without_tp(env['PYTHONPATH'])
        env['AMENT_PREFIX_PATH'] = _without_tp(env.get('AMENT_PREFIX_PATH', ''))
    # outside src/: a `python -c` puts its cwd on sys.path, where the trainit_perception checkout
    # would import as an empty namespace package and look installed
    r = subprocess.run([sys.executable, '-c', _WORKER, mode, case, str(TEST), str(PKG)],
                       capture_output=True, text=True, env=env, cwd=tempfile.gettempdir(),
                       timeout=600)
    line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
    assert r.returncode == 0 and line, r.stdout + r.stderr
    return json.loads(line[0][len('RESULT '):])


def _comparable(result):
    """What must not depend on the environment: every file's bytes (the manifest included -- the
    same TSA writes the same version), or the refusal, word for word."""
    return {k: result[k] for k in ('files', 'refused') if k in result}


def _difference(a, b):
    """Which bundle files (or which refusal) differ -- the failure message."""
    if 'files' in a and 'files' in b:
        return sorted(k for k in set(a['files']) | set(b['files'])
                      if a['files'].get(k) != b['files'].get(k))
    return {'one': a.get('refused', 'generated'), 'other': b.get('refused', 'generated')}


# On 7.3.0 these already generate the same with and without TP; they must stay so.
SAME_TODAY = ['plain', 'rule_bbox_disc', 'rule_disc_measured', 'noise_all', 'tool_numbers',
              'external_rule']
# On 7.3.0 these do not: a version pin, a face's parameter order, or a refusal depends on TP.
DIFFERENT_TODAY = ['tilt', 'rule_rect_measured', 'rule_rect_measured_extra_key',
                   'rule_cylinder_measured_reversed', 'rule_cylinder_params',
                   'refused_unknown_face_rule', 'refused_measured_unknown_face_rule',
                   'refused_rect_half_missing_rule']


@pytest.mark.parametrize('case', SAME_TODAY + DIFFERENT_TODAY)
def test_the_same_project_generates_the_same_with_and_without_trainit_perception(case):
    importable, blocked = _generate('normal', case), _generate('blocked', case)
    if not importable['tp_importable']:
        pytest.skip('trainit_perception is not installed here, so "with it" cannot be run')
    assert blocked['tp_importable'] is False
    assert _comparable(blocked) == _comparable(importable), _difference(importable, blocked)


@pytest.mark.parametrize('case', SAME_TODAY + DIFFERENT_TODAY)
def test_the_same_project_generates_the_same_with_trainit_perception_uninstalled(case):
    installed, absent = _generate('normal', case), _generate('absent', case)
    if not installed['tp_importable']:
        pytest.skip('trainit_perception is not installed here, so "with it" cannot be run')
    if absent['tp_importable']:
        pytest.skip('cannot simulate an uninstall here: trainit_perception still imports from '
                    f'{absent["tp_origin"]} (a merged install shares its prefix)')
    for result in (installed, absent):             # and no run asked TP anything
        assert not (result['tp_lookups'] or result['tp_files_opened']
                    or result['ament_tp_lookups']), result
    assert _comparable(absent) == _comparable(installed), _difference(installed, absent)


@pytest.mark.parametrize('case', ['plain', 'rule_cylinder_measured', 'tilt',
                                  'refused_unknown_face_rule'])
def test_generation_does_not_query_the_installed_trainit_perception(case):
    result = _generate('normal', case)
    assert result['tp_lookups_during_generation'] == [], result
    assert result['tp_lookups'] == [], result
    assert result['tp_files_opened'] == [], result
    assert result['ament_tp_lookups'] == [], result


# --- 4. the refresh writes all or nothing -------------------------------------------------------

@pytest.fixture
def scratch_catalog(tmp_path, monkeypatch):
    """The refresh pointed at a scratch catalog and a scratch 'installed' TMR registry."""
    here = tmp_path / 'catalog'
    here.mkdir()
    (here / REGISTRY_FILE).write_bytes(COPY.read_bytes())
    for name in os.listdir(COPY.parent):
        if name.endswith('.yaml') and name != REGISTRY_FILE:
            (here / name).write_bytes((COPY.parent / name).read_bytes())
    tmr = tmp_path / 'planner_profiles.yaml'
    tmr.write_bytes(COPY.read_bytes() + b'# TMR moved\n')
    monkeypatch.setattr(refresh_tool, 'catalog_dir', lambda: here)
    monkeypatch.setattr(refresh_tool, 'tmr_registry_path', lambda: tmr)
    monkeypatch.setattr(refresh_tool, 'tp_available', lambda: True)
    # TP answers with the committed vocabulary, so the refresh reaches its derive step whether or
    # not trainit_perception is installed where the test runs
    from trainit_setup_assistant import catalog
    monkeypatch.setattr(refresh_tool, 'tp_export', catalog.tp_registry)
    monkeypatch.setattr(refresh_tool, 'tp_installed_release',
                        lambda: catalog.tp_registry()['trainit_perception'])
    return types.SimpleNamespace(
        here=here, files=lambda: {f.name: f.read_bytes() for f in here.iterdir()})


def test_a_refresh_whose_derive_step_fails_writes_nothing(scratch_catalog, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError('derive failed after the validation passed')
    monkeypatch.setattr(refresh_tool, 'derive_record', broken)
    before = scratch_catalog.files()
    with pytest.raises(RuntimeError, match='derive failed after the validation passed'):
        refresh_tool.refresh(False)
    assert scratch_catalog.files() == before, 'a failed refresh rewrote part of the catalog'
