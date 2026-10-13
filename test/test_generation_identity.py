"""The generation identity proof, as a regression test: 43 projects x 5 environments (opt-in).

    TRAINIT_IDENTITY_TEST=1 python3 -m pytest test/test_generation_identity.py
    (optional TRAINIT_IDENTITY_JOBS=N, default min(8, CPUs); about 12 s with 8 jobs)

vNext M2.2 proved, outside the repositories, that TSA generates the same bundle whatever
trainit_perception is doing on the machine: the 43 projects of ``identity_cases.py`` in five
environments, every bundle identical byte for byte. The owner asked for the proof to stay
(vNext M2.3); this is it. Each (project, environment) runs ``identity_worker.py`` in a FRESH
interpreter with its own random hash seed, its own output directory and its own TMPDIR:

  A, A2  the environment as sourced, twice (two interpreters: what the seed may change)
  B      trainit_perception made unimportable (a sys.meta_path finder)
  B2     trainit_perception off PYTHONPATH (still on the ament index)
  B3     trainit_perception off PYTHONPATH and the ament/CMake/colcon prefixes (uninstalled)

and for every project:

* all five environments generate the SAME tree -- every file's bytes, the manifest included and
  unmasked (one TSA, one version), the executable bits -- or refuse with the same ValueError, word
  for word, leaving no output directory;
* no run imports trainit_perception, trainit_motion_runtime or ament_index_python, opens one of
  their files or writes anything outside its bundle and its TMPDIR, from interpreter start to exit;
* it still does what it did in the proof (30 projects generate, 13 refuse).

And for the whole matrix: the workers ran THIS source tree's TSA (not an older install earlier on
the path), the observers see what they watch (a positive control in every environment), every
bundle of every project in every environment records ONE capability snapshot -- the committed
catalog's, as the test computes it -- and names its copied files without a path of this machine
(vNext M2.3), and nothing is left in the temporary directory. The environments are compared with each other only:
the comparison with TSA 7.3.0 belongs to the M2.2 evidence. Projects the checkout does not have
skip (the golden sibling, fr3wml_digital_twin/fr5_app, the owner's untracked policy project).
"""
import os

import pytest

if os.environ.get('TRAINIT_IDENTITY_TEST') != '1':
    pytest.skip('opt-in: set TRAINIT_IDENTITY_TEST=1 (43 projects x 5 environments, fresh '
                'interpreters, about 12 s with 8 jobs)', allow_module_level=True)

import concurrent.futures as cf          # noqa: E402
import json                              # noqa: E402
import random                            # noqa: E402
import shutil                            # noqa: E402
import subprocess                        # noqa: E402
import sys                               # noqa: E402
import tempfile                          # noqa: E402
import xml.etree.ElementTree as ET       # noqa: E402
from pathlib import Path                 # noqa: E402

import catalog_identity                  # noqa: E402
import identity_cases                    # noqa: E402
import re                                # noqa: E402
import trainit_setup_assistant as T      # noqa: E402

TEST = Path(__file__).resolve().parent
PKG = TEST.parent
WORKER = TEST / 'identity_worker.py'
MODES = ('A', 'A2', 'B', 'B2', 'B3')
JOBS = int(os.environ.get('TRAINIT_IDENTITY_JOBS') or min(8, os.cpu_count() or 1))
CASES = list(identity_cases.CASES)
_SEEDS = random.SystemRandom()


def _without_tp(paths: str) -> str:
    return os.pathsep.join(p for p in paths.split(os.pathsep)
                           if p and 'trainit_perception' not in p)


def env_for(mode: str, tmp: Path, seed: int) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED=str(seed),
               TMPDIR=str(tmp),
               PYTHONPATH=os.pathsep.join([str(PKG), os.environ.get('PYTHONPATH', '')]))
    if mode in ('B2', 'B3'):
        env['PYTHONPATH'] = _without_tp(env['PYTHONPATH'])
    if mode == 'B3':
        for var in ('AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH'):
            env[var] = _without_tp(env.get(var, ''))
    return env


def run(root: Path, mode: str, case: str = None) -> dict:
    """One worker; its directory (bundle + TMPDIR) is gone when this returns."""
    seed = _SEEDS.randint(1, 2 ** 32 - 1)
    with tempfile.TemporaryDirectory(dir=root) as run_dir:
        run_dir = Path(run_dir)
        tmp = run_dir / 'tmp'
        tmp.mkdir()
        cmd = [sys.executable, '-B', str(WORKER), '--out', str(run_dir / 'out')]
        cmd += ['--control'] if case is None else ['--case', case]
        if mode == 'B':
            cmd.append('--block-tp')
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir),
                               env=env_for(mode, tmp, seed), timeout=120)
            line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
            result = (json.loads(line[-1][len('RESULT '):]) if line
                      else {'outcome': 'crashed', 'crash': {'type': 'no RESULT line'}})
            result['returncode'] = r.returncode
            result['stderr_tail'] = r.stderr[-2000:]
        except subprocess.TimeoutExpired:
            result = {'outcome': 'crashed', 'crash': {'type': 'timeout after 120 s'}}
        result['left_in_tmpdir'] = sorted(os.listdir(tmp))
    result['seed'] = seed
    result['reproduce'] = (f'PYTHONHASHSEED={seed} (mode {mode} env) {" ".join(cmd)}')
    return result


@pytest.fixture(scope='module')
def matrix():
    root = Path(tempfile.mkdtemp(prefix='tsa_identity_'))
    jobs = [(case, mode) for case in CASES if not identity_cases.missing(case) for mode in MODES]
    jobs += [(None, mode) for mode in MODES]                   # the positive controls
    try:
        with cf.ThreadPoolExecutor(max_workers=JOBS) as pool:
            futures = {pool.submit(run, root, mode, case): (case, mode) for case, mode in jobs}
            results = {futures[f]: f.result() for f in cf.as_completed(futures)}
        results['left_in_root'] = sorted(os.listdir(root))
        yield results
    finally:
        shutil.rmtree(root, ignore_errors=True)
    assert not root.exists(), f'{root} survived the matrix'


def _queries(result: dict) -> dict:
    found = {f'import {k}': v for k, v in result.get('imports', {}).items() if v}
    found.update({f'open {k}': v for k, v in result.get('opened', {}).items() if v})
    found.update({f'module at exit {k}': v for k, v in result.get('modules_at_exit', {}).items()
                  if v})
    if result.get('writes_outside'):
        found['writes outside the bundle'] = result['writes_outside']
    if result.get('left_in_tmpdir'):
        found['left in TMPDIR'] = result['left_in_tmpdir']
    return found


def _outcome(result: dict):
    if result.get('outcome') == 'generated':
        return ('generated', result['files'], result['executables'])
    if result.get('outcome') == 'refused':
        return ('refused', result['refusal']['type'], result['refusal']['message'],
                result['files'])
    return ('crashed', result.get('crash'), result.get('returncode'),
            result.get('stderr_tail', '')[-800:])


def _difference(a: tuple, b: tuple):
    if a[0] == b[0] == 'generated':
        return sorted(k for k in set(a[1]) | set(b[1]) if a[1].get(k) != b[1].get(k)) or \
            {'executable bits': [a[2], b[2]]}
    return {'one': a[:3], 'other': b[:3]}


# --- the whole matrix -----------------------------------------------------------------------------

def test_the_tsa_under_test_is_this_source_tree(matrix):
    """An older TSA earlier on the path (the install space) would generate the same bundles but
    for generator_version, and prove nothing about this tree: every worker ran THIS tree's TSA."""
    here = (PKG / 'trainit_setup_assistant' / '__init__.py').resolve()
    released = ET.parse(PKG / 'package.xml').getroot().findtext('version')
    assert Path(T.__file__).resolve() == here and T.__version__ == released
    ran = {(r.get('tsa_file_real'), r.get('tsa_version'))
           for key, r in matrix.items() if key != 'left_in_root' and key[0] is not None}
    assert ran == {(str(here), released)}, ran
    assert all(r.get('dont_write_bytecode') for key, r in matrix.items()
               if key != 'left_in_root')


def test_the_observers_see_what_they_watch(matrix):
    """The positive control does, on purpose, each thing the per-project items assert never
    happens. Without it a broken observer would make 'nothing queried' true by construction."""
    tp_here = matrix[(None, 'A')].get('tp_importable_real_package')
    for mode in MODES:
        r = matrix[(None, mode)]
        assert r.get('outcome') == 'control', (mode, r.get('crash'), r.get('stderr_tail'))
        for family in ('tp', 'tmr', 'ament'):
            assert r['imports'][family], (mode, family, r['imports'])
        assert any('control_write.txt' in w for w in r['writes_outside']), (mode, r)
        assert r['opened']['tmr'], (mode, r['opened'])     # TMR stays installed in every mode
        if mode == 'B3':                    # uninstalled: the ament index has no TP share left
            assert not r['opened']['tp'], (mode, r['opened'])
        elif tp_here:
            assert r['opened']['tp'], (mode, r['opened'])


def test_the_environments_are_what_they_claim(matrix):
    probes = {mode: matrix[(None, mode)].get('tp_importable_real_package') for mode in MODES}
    assert probes['B'] is False, 'the blocking finder let trainit_perception import'
    if not (probes['A'] and probes['A2']):
        pytest.skip('trainit_perception is not installed here: the matrix proves determinism '
                    'across five interpreters, not independence from trainit_perception')
    for mode in ('B2', 'B3'):
        if probes[mode]:
            pytest.skip(f'cannot take trainit_perception off the path here ({mode} still imports '
                        f'it from {matrix[(None, mode)].get("tp_origin")}: a merged install)')


def _generated(matrix):
    return [(key, r) for key, r in matrix.items()
            if key != 'left_in_root' and key[0] is not None and r.get('outcome') == 'generated']


def test_every_bundle_records_one_capability_snapshot(matrix):
    seen = {json.dumps(r['manifest'].get('capability_snapshot'), sort_keys=True)
            for _key, r in _generated(matrix)}
    assert seen == {json.dumps(catalog_identity.committed_identity(), sort_keys=True)}, seen


def test_no_copied_file_is_named_by_a_path_of_this_machine(matrix):
    portable = re.compile(r'^package://[a-zA-Z0-9][a-zA-Z0-9_-]*/\S+$|^external:[^/]+$')
    bad = {(key, f['source']) for key, r in _generated(matrix) for f in r['manifest']['files']
           if f['action'] == 'COPY' and not portable.match(f['source'])}
    assert not bad, sorted(bad)[:10]


def test_nothing_is_left_in_the_temporary_directory(matrix):
    assert matrix['left_in_root'] == []
    leftovers = {key: r['left_in_tmpdir'] for key, r in matrix.items()
                 if key != 'left_in_root' and r.get('left_in_tmpdir')}
    assert not leftovers, leftovers


# --- one item per project -----------------------------------------------------------------------

@pytest.mark.parametrize('case', CASES)
def test_every_environment_generates_the_same(matrix, case):
    why = identity_cases.missing(case)
    if why:
        pytest.skip(why)
    runs = {mode: matrix[(case, mode)] for mode in MODES}
    outcomes = {mode: _outcome(r) for mode, r in runs.items()}
    crashed = {mode: o for mode, o in outcomes.items() if o[0] == 'crashed'}
    assert not crashed, {m: (o, runs[m]['reproduce']) for m, o in crashed.items()}
    for mode in MODES[1:]:
        assert outcomes[mode] == outcomes['A'], (
            f'{case}: mode {mode} differs from mode A', _difference(outcomes['A'], outcomes[mode]),
            runs[mode]['reproduce'])
    kind = outcomes['A'][0]
    expected = 'refused' if case in identity_cases.REFUSED else 'generated'
    assert kind == expected, (case, kind, outcomes['A'][1:3])
    if kind == 'refused':
        assert outcomes['A'][1] == 'ValueError' and outcomes['A'][3] == {}, outcomes['A']
    queried = {mode: q for mode, q in ((m, _queries(r)) for m, r in runs.items()) if q}
    assert not queried, queried
