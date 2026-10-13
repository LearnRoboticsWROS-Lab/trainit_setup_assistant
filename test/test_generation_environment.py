"""The same project gives the same bundle whatever the machine around TSA is doing (vNext M2.3, T4).

Generation is a function of the project, the TSA release and TSA's committed catalog snapshot
(ADR-0021) -- never of the shell it runs in. Each project here is generated once as a baseline,
then once per VARIANT of the environment, every run in a fresh interpreter
(``identity_worker.py``), and every variant must give the baseline's tree byte for byte, the
manifest included:

  seed_cwd_tmpdir   another hash seed; a cwd holding decoy trainit_setup_assistant and
                    trainit_perception packages; a TMPDIR with spaces and non-ASCII characters
                    under a planted package.xml
  home              HOME moved (the user site kept: pydantic lives there)
  locale_c          LC_ALL=C (Python's UTF-8 mode)
  locale_c_utf8     LC_ALL=C.UTF-8
  tsa_symlink       this TSA imported through a symlink
  inputs_symlink    the project's inputs reached through a symlink to the workspace: the
                    manifest names a copied file by its package (vNext M2.3), not by the path
                    that reached it
  no_components     trainit_perception and trainit_motion_runtime off PYTHONPATH and every
                    prefix variable (uninstalled)
  fake_components   a FAKE prefix first: trainit_motion_runtime 9.9.9 with another planner
                    registry, trainit_perception 0.0.1, and Python packages of both that raise
                    when imported -- generation must not see any of it
  no_ros            no ROS environment at all
  locale_latin1, locale_c_no_utf8 -- KNOWN DEBT (docs/engineering/KNOWN_DEBT.md): nine text
                    reads use the locale's encoding, so a non-UTF-8 locale changes the golden's
                    merged SRDF or stops it; strict xfails, so the day they are fixed this file
                    says so

In every run: the worker imported THIS source tree's TSA; the manifest records the committed
capability snapshot (``catalog_identity``, M2.3) -- the fake components' versions and registry
never reach it; nothing imported trainit_perception, trainit_motion_runtime or
ament_index_python, opened one of their files or a file under the fake prefix, or wrote outside
its bundle and TMPDIR.
"""
import concurrent.futures as cf
import json
import os
import shutil
import site
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

import catalog_identity
import identity_cases
import ws_paths

TEST = Path(__file__).resolve().parent
PKG = TEST.parent
WORKER = TEST / 'identity_worker.py'
PROJECTS = ('p1_golden', 'p2_example', 'pol1_policy_test4_project')
VARIANTS = ('seed_cwd_tmpdir', 'home', 'locale_c', 'locale_c_utf8', 'tsa_symlink',
            'inputs_symlink', 'no_components', 'fake_components', 'no_ros', 'locale_latin1',
            'locale_c_no_utf8')
KNOWN_LOCALE_DEBT = {('p1_golden', 'locale_latin1'), ('p1_golden', 'locale_c_no_utf8'),
                     ('pol1_policy_test4_project', 'locale_latin1'),
                     ('pol1_policy_test4_project', 'locale_c_no_utf8')}
COMPONENTS = ('trainit_perception', 'trainit_motion_runtime')
PREFIX_VARS = ('AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH')


def _locales():
    try:
        return set(subprocess.run(['locale', '-a'], capture_output=True, text=True,
                                  timeout=30).stdout.split())
    except Exception:                                       # noqa: BLE001
        return set()


def _strip(paths: str, *names: str) -> str:
    return os.pathsep.join(p for p in paths.split(os.pathsep)
                           if p and not any(n in p for n in names))


def _fake_components(root: Path) -> dict:
    """A prefix and a Python directory where both components exist, in versions and with
    vocabularies the committed snapshot does not have, and blow up when imported."""
    prefix, python = root / 'fake_prefix', root / 'fake_python'
    index = prefix / 'share' / 'ament_index' / 'resource_index' / 'packages'
    index.mkdir(parents=True)
    for name, version in (('trainit_motion_runtime', '9.9.9'), ('trainit_perception', '0.0.1')):
        (index / name).write_text('')
        share = prefix / 'share' / name
        share.mkdir(parents=True)
        (share / 'package.xml').write_text(
            f'<?xml version="1.0"?>\n<package format="3"><name>{name}</name>'
            f'<version>{version}</version></package>\n')
        (python / name).mkdir(parents=True)
        (python / name / '__init__.py').write_text(
            f'raise RuntimeError("the fake {name} was imported")\n')
    # a registry TSA could read: another stable mode, another default -- reading it would change
    # the bundle's planner lists, so a TSA that looked at it could not pass unnoticed
    registry = yaml.safe_load((PKG / 'trainit_setup_assistant' / 'catalog' /
                               'tmr_planner_registry.yaml').read_bytes())
    registry['planner_modes']['fake_mode'] = dict(registry['planner_modes']['ompl'])
    registry['default_planner_mode'] = 'fake_mode'
    (prefix / 'share' / 'trainit_motion_runtime' / 'config').mkdir()
    (prefix / 'share' / 'trainit_motion_runtime' / 'config' / 'planner_profiles.yaml').write_text(
        yaml.safe_dump(registry, sort_keys=False))
    return {'prefix': prefix, 'python': python}


def _decoy_cwd(root: Path) -> Path:
    cwd = root / 'decoy_cwd'
    for name in ('trainit_setup_assistant', 'trainit_perception'):
        (cwd / name).mkdir(parents=True)
        (cwd / name / '__init__.py').write_text(f'raise RuntimeError("decoy {name} imported")\n')
    return cwd


def _odd_tmpdir(root: Path) -> Path:
    planted = root / 'planted_pkg'
    planted.mkdir()
    (planted / 'package.xml').write_text('<package format="3"><name>planted</name></package>\n')
    tmp = planted / 'tmp dir éàü'
    tmp.mkdir()
    return tmp


def _environment(variant: str, shared: dict, run_dir: Path):
    """(env, cwd, extra worker arguments) for one variant; the baseline is variant None."""
    tmp = run_dir / 'tmp'
    tmp.mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0', TMPDIR=str(tmp),
               PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    cwd, extra = run_dir, []
    if variant == 'seed_cwd_tmpdir':
        odd = Path(tempfile.mkdtemp(dir=shared['odd_tmpdir']))
        env.update(PYTHONHASHSEED='4242', TMPDIR=str(odd))
        cwd = shared['decoy_cwd']
    elif variant == 'home':
        home = run_dir / 'home'
        home.mkdir()
        env.update(HOME=str(home), PYTHONUSERBASE=site.getuserbase())
    elif variant == 'locale_c':
        env.update(LC_ALL='C', LANG='C')
    elif variant == 'locale_c_utf8':
        env.update(LC_ALL='C.UTF-8', LANG='C.UTF-8')
    elif variant == 'locale_latin1':
        env.update(LC_ALL='en_US.ISO-8859-1', LANG='en_US.ISO-8859-1')
    elif variant == 'locale_c_no_utf8':
        env.update(LC_ALL='C', LANG='C', PYTHONUTF8='0')
    elif variant == 'tsa_symlink':
        extra = ['--pkg', str(shared['tsa_link'])]
        env['PYTHONPATH'] = os.pathsep.join([str(shared['tsa_link']),
                                             os.environ.get('PYTHONPATH', '')])
    elif variant == 'inputs_symlink':
        extra = ['--alias', f'{ws_paths.WS_SRC}={shared["ws_link"]}']
    elif variant == 'no_components':
        env['PYTHONPATH'] = _strip(env['PYTHONPATH'], *COMPONENTS)
        for var in PREFIX_VARS:
            env[var] = _strip(env.get(var, ''), *COMPONENTS)
    elif variant == 'fake_components':
        fake = shared['fake']
        env['PYTHONPATH'] = os.pathsep.join([str(PKG), str(fake['python']),
                                             os.environ.get('PYTHONPATH', '')])
        for var in PREFIX_VARS:
            env[var] = os.pathsep.join([str(fake['prefix']), env.get(var, '')])
        extra = ['--forbid', str(fake['prefix']), '--forbid', str(fake['python'])]
    elif variant == 'no_ros':
        for var in list(env):
            if var.startswith(('AMENT', 'COLCON', 'ROS_', 'RMW', 'CMAKE_PREFIX')) or \
                    var in ('LD_LIBRARY_PATH', 'PYTHONPATH'):
                del env[var]
        env['PYTHONPATH'] = str(PKG)
    return env, cwd, extra


def _run(root: Path, shared: dict, project: str, variant: str = None) -> dict:
    """One worker; ``project`` None runs the positive control in that variant."""
    with tempfile.TemporaryDirectory(dir=root) as run_dir:
        run_dir = Path(run_dir)
        env, cwd, extra = _environment(variant, shared, run_dir)
        what = ['--control'] if project is None else ['--case', project]
        cmd = [sys.executable, '-B', str(WORKER), *what, '--out', str(run_dir / 'out')] + extra
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(cwd), env=env,
                           timeout=120)
        line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
        result = (json.loads(line[-1][len('RESULT '):]) if line
                  else {'outcome': 'crashed', 'crash': {'type': 'no RESULT line'}})
        result['stderr_tail'] = r.stderr[-1500:]
        result['left_in_tmpdir'] = sorted(os.listdir(env.get('TMPDIR')))
    return result


@pytest.fixture(scope='module')
def runs():
    root = Path(tempfile.mkdtemp(prefix='tsa_environment_'))
    shared = {'fake': _fake_components(root), 'decoy_cwd': _decoy_cwd(root),
              'odd_tmpdir': _odd_tmpdir(root), 'tsa_link': root / 'tsa_link',
              'ws_link': root / 'ws_link'}
    shared['tsa_link'].symlink_to(PKG, target_is_directory=True)
    shared['ws_link'].symlink_to(ws_paths.WS_SRC, target_is_directory=True)
    locales = {name.lower().replace('-', '') for name in _locales()}
    jobs = [(p, v) for p in PROJECTS if not identity_cases.missing(p)
            for v in (None,) + VARIANTS
            if v != 'locale_latin1' or 'en_us.iso88591' in locales]
    jobs.append((None, 'fake_components'))       # the positive control, with the fakes there
    try:
        with cf.ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 1)) as pool:
            futures = {pool.submit(_run, root, shared, p, v): (p, v) for p, v in jobs}
            results = {futures[f]: f.result() for f in cf.as_completed(futures)}
        results['shared'] = {'fake_python': str(shared['fake']['python']),
                             'tsa_link': str(shared['tsa_link'])}
        yield results
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert not root.exists()


def _content(result: dict):
    if result.get('outcome') not in ('generated', 'refused'):
        return ('crashed', result.get('crash'), result.get('stderr_tail', '')[-600:])
    return (result['outcome'], result.get('refusal'), result['files'], result['executables'])


@pytest.mark.parametrize('variant', VARIANTS)
@pytest.mark.parametrize('project', PROJECTS)
def test_the_environment_does_not_reach_the_bundle(runs, request, project, variant):
    why = identity_cases.missing(project)
    if why:
        pytest.skip(why)
    if (project, variant) not in runs:
        pytest.skip(f'{variant}: the locale is not installed here')
    if (project, variant) in KNOWN_LOCALE_DEBT:
        request.applymarker(pytest.mark.xfail(
            strict=True, reason='known debt: text reads use the locale encoding'))
    base, run = runs[(project, None)], runs[(project, variant)]
    assert base.get('outcome') == 'generated', (_content(base), base.get('stderr_tail'))
    assert base['manifest']['capability_snapshot'] == catalog_identity.committed_identity()
    here = str((PKG / 'trainit_setup_assistant' / '__init__.py').resolve())
    assert run.get('tsa_file_real') == base.get('tsa_file_real') == here, run.get('tsa_file')
    assert _content(run) == _content(base), sorted(
        k for k in set(run.get('files', {})) | set(base['files'])
        if run.get('files', {}).get(k) != base['files'].get(k)) or _content(run)[:2]
    queried = {f'import {k}': v for k, v in run['imports'].items() if v}
    queried.update({f'open {k}': v for k, v in run['opened'].items() if v})
    if run['writes_outside']:
        queried['writes outside the bundle'] = run['writes_outside']
    if run['left_in_tmpdir']:
        queried['left in TMPDIR'] = run['left_in_tmpdir']
    assert not queried, queried


def test_the_observers_see_what_the_variants_hide(runs):
    """The positive control, in the fake-components environment: an import of each watched
    package, a read of TP's and TMR's shares and of a fake file, a write outside the bundle --
    all seen. Without it a blind observer would make 'nothing queried' true by construction."""
    r = runs[(None, 'fake_components')]
    assert r.get('outcome') == 'control', (r.get('crash'), r.get('stderr_tail'))
    for family in ('tp', 'tmr', 'ament'):
        assert r['imports'][family], (family, r['imports'])
    assert r['opened']['forbidden'] and r['opened']['tp'] and r['opened']['tmr'], r['opened']
    assert any('control_write.txt' in w for w in r['writes_outside']), r['writes_outside']


def _present():
    present = [p for p in PROJECTS if not identity_cases.missing(p)]
    if not present:
        pytest.skip('no project of this matrix is on this checkout')
    return present


def test_the_variants_did_what_they_claim(runs):
    """A variant that silently did nothing would pass vacuously (these hold with or without TP)."""
    for project in _present():
        base = runs[(project, None)]
        assert runs[(project, 'tsa_symlink')]['tsa_file'].startswith(runs['shared']['tsa_link'])
        assert runs[(project, 'inputs_symlink')].get('aliased', 0) > 0, project
        assert runs[(project, 'home')]['tsa_file_real'] == base['tsa_file_real']
        # the fakes were there to be found: trainit_perception resolves to the fake package
        assert (runs[(project, 'fake_components')]['tp_origin'] or '').startswith(
            runs['shared']['fake_python']), runs[(project, 'fake_components')]['tp_origin']


def _merged(origin) -> bool:
    """An install whose path entry names no package (colcon --merge-install): no strip works."""
    entry = os.path.dirname(os.path.dirname(origin or ''))
    return bool(origin) and not any(c in entry for c in COMPONENTS)


def test_the_uninstalled_variant_took_tp_and_tmr_away(runs):
    for project in _present():
        base, gone = runs[(project, None)], runs[(project, 'no_components')]
        if not base['tp_importable_real_package']:
            pytest.skip('trainit_perception is not installed here: nothing to take away')
        if _merged(gone.get('tp_origin')) or _merged(gone.get('tmr_origin')):
            pytest.skip('a merged install: TP or TMR shares a path entry with other packages')
        assert base['tmr_importable'] and base['tp_on_ament_index']
        assert not gone['tp_importable_real_package'] and not gone['tmr_importable'], gone
        assert not gone['tp_on_ament_index'], 'TP is still on the ament index'
