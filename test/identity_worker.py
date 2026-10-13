"""One generation in a FRESH interpreter, watched -- the worker of the identity tests (vNext M2.3).

    python3 -B identity_worker.py --case <name> --out <dir> [--pkg <TSA root>] [--block-tp]
                                  [--alias <from>=<to>] [--forbid <root>]...
    python3 -B identity_worker.py --control --out <dir> [...]

Run as a SCRIPT (``test_generation_identity.py``, ``test_generation_environment.py``): a script's
own directory, not the cwd, is the first entry of sys.path, so the caller may vary the cwd freely
without a checkout there shadowing the TSA under test. It prints ONE line ``RESULT <json>``.

Before TSA is imported it installs three observers, so what it reports covers the whole run:

* a ``sys.meta_path`` finder that records every import of ``trainit_perception*``,
  ``trainit_motion_runtime*`` and ``ament_index_python*`` -- generation must import none of them
  (``--block-tp`` also makes trainit_perception unimportable);
* an audit hook that records every file OPENED under a trainit_perception or
  trainit_motion_runtime path, or under a ``--forbid`` root, that is not a module (the finder
  sees those) and not the entry_points.txt pydantic's plugin loader reads from EVERY installed
  distribution when the first model is built -- a scan of the environment, not a question asked of
  a component;
* the same hook records every WRITE outside ``--out`` and ``$TMPDIR``: an open for writing and
  every os/shutil call that creates, moves, removes or changes a file.

``--control`` does, deliberately, one of each thing the observers watch, so a test can prove they
see it. ``--alias FROM=TO`` re-roots the project's input paths (FROM a directory, TO another path
to it, e.g. a symlink) after conftest rebased them. Writes nothing outside --out and $TMPDIR.
"""
import argparse
import hashlib
import json
import os
import stat
import sys
import traceback

ap = argparse.ArgumentParser()
ap.add_argument('--case')
ap.add_argument('--control', action='store_true')
ap.add_argument('--out', required=True)
ap.add_argument('--pkg', default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument('--block-tp', action='store_true')
ap.add_argument('--alias', default='')
ap.add_argument('--forbid', action='append', default=[])
args = ap.parse_args()

TEST = os.path.dirname(os.path.abspath(__file__))
WATCHED = {'tp': 'trainit_perception', 'tmr': 'trainit_motion_runtime',
           'ament': 'ament_index_python'}
R = {'imports': {k: [] for k in WATCHED}, 'opened': {'tp': [], 'tmr': [], 'forbidden': []},
     'writes_outside': [], 'dont_write_bytecode': bool(sys.flags.dont_write_bytecode)}


def _family(module):
    top = module.split('.')[0]
    return next((k for k, v in WATCHED.items() if top.startswith(v)), None)


class _Watch:
    """Records the watched imports; with --block-tp, trainit_perception cannot be imported."""

    def find_spec(self, name, path=None, target=None):
        family = _family(name)
        if family is None:
            return None
        R['imports'][family].append(name)
        if family == 'tp' and args.block_tp:
            raise ModuleNotFoundError(f'{name!r} made unimportable', name=name)
        return None

    def invalidate_caches(self):
        pass


sys.meta_path.insert(0, _Watch())

_ALLOWED = [os.path.realpath(args.out), os.path.realpath(os.environ.get('TMPDIR') or '/nonexistent')]
_FORBID = [os.path.realpath(p) for p in args.forbid]
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC
_MUTATIONS = {'os.mkdir', 'os.rename', 'os.remove', 'os.rmdir', 'os.chmod', 'os.chown',
              'os.link', 'os.symlink', 'os.truncate', 'os.utime', 'shutil.copyfile',
              'shutil.copymode', 'shutil.copystat', 'shutil.copytree', 'shutil.move',
              'shutil.rmtree', 'shutil.chown'}
_busy = False


def _under(path, roots):
    return any(path == r or path.startswith(r + os.sep) for r in roots)


def _pydantic_scan():
    frame = sys._getframe(2)
    while frame is not None:
        if f'{os.sep}pydantic{os.sep}' in frame.f_code.co_filename:
            return True
        frame = frame.f_back
    return False


def _changed(event, a):
    """The path(s) an event creates, changes or removes -- or, for an open, opens."""
    if event in ('os.rename', 'shutil.move'):
        picked = a[:2]                       # the old name goes, the new one appears
    elif event.startswith('shutil.copy') or event in ('os.symlink', 'os.link'):
        picked = a[1:2]                      # the destination; the source is only read
    else:
        picked = a[:1]
    return [os.fsdecode(x) for x in picked if isinstance(x, (str, bytes, os.PathLike))]


def _writing(event, a):
    if event != 'open':
        return True
    flags = a[2] if len(a) > 2 and isinstance(a[2], int) else 0
    mode = a[1] if len(a) > 1 and isinstance(a[1], str) else ''
    return bool(flags & _WRITE_FLAGS) or any(c in mode for c in 'wax+')


def _audit(event, a):
    global _busy
    if _busy or not (event == 'open' or event in _MUTATIONS):
        return
    _busy = True
    try:
        writing = _writing(event, a)
        for p in _changed(event, a):
            if not writing and p.endswith(('.py', '.pyc', '.so')):
                continue                     # a module load: the finder watches those
            full = os.path.realpath(os.path.abspath(p))
            if writing and not _under(full, _ALLOWED):
                R['writes_outside'].append(f'{event} {p}')
            if event != 'open':
                continue
            if _under(full, _FORBID):
                R['opened']['forbidden'].append(p)
            for key in ('tp', 'tmr'):
                if WATCHED[key] in p and not (p.endswith('entry_points.txt') and _pydantic_scan()):
                    R['opened'][key].append(p)
    finally:
        _busy = False


sys.addaudithook(_audit)

import importlib.util  # noqa: E402

try:                         # a real package, not a checkout seen as a namespace package
    _spec = importlib.util.find_spec('trainit_perception')
    R['tp_importable_real_package'] = bool(_spec is not None
                                           and _spec.origin not in (None, 'namespace'))
    R['tp_origin'] = None if _spec is None else str(_spec.origin)
except ModuleNotFoundError:
    R['tp_importable_real_package'], R['tp_origin'] = False, None
try:                         # and is trainit_motion_runtime there at all
    _tmr = importlib.util.find_spec('trainit_motion_runtime')
    R['tmr_importable'], R['tmr_origin'] = _tmr is not None, (_tmr and str(_tmr.origin))
except ModuleNotFoundError:
    R['tmr_importable'], R['tmr_origin'] = False, None
# on the ament index? its marker file, looked at without importing the index's library
R['tp_on_ament_index'] = any(os.path.isfile(os.path.join(
    prefix, 'share', 'ament_index', 'resource_index', 'packages', 'trainit_perception'))
    for prefix in os.environ.get('AMENT_PREFIX_PATH', '').split(os.pathsep) if prefix)
for _k in R['imports']:      # those lookups were this worker's, not TSA's
    R['imports'][_k].clear()
for _k in R['opened']:
    R['opened'][_k].clear()

sys.path[:0] = [TEST, args.pkg]


def _files(root):
    out, executables = {}, []
    for dirpath, _dirs, names in os.walk(root):
        for n in names:
            full = os.path.join(dirpath, n)
            rel = os.path.relpath(full, root).replace(os.sep, '/')
            with open(full, 'rb') as fh:
                out[rel] = hashlib.sha256(fh.read()).hexdigest()
            if os.stat(full).st_mode & stat.S_IXUSR:
                executables.append(rel)
    return dict(sorted(out.items())), sorted(executables)


def _realias(project):
    """Reach the project's inputs through another path to the same directory (--alias)."""
    src, dst = args.alias.split('=', 1)

    def moved(path):
        if path and (path == src or path.startswith(src + os.sep)):
            R['aliased'] = R.get('aliased', 0) + 1
            return dst + path[len(src):]
        return path
    desc = project.robot.description
    desc.urdf_dir = moved(desc.urdf_dir)
    desc.meshes_dir = moved(desc.meshes_dir) if desc.meshes_dir else desc.meshes_dir
    if project.robot.base_moveit_config_path:
        project.robot.base_moveit_config_path = moved(project.robot.base_moveit_config_path)
    for policy in (project.application.policies if project.application else []):
        policy.card = moved(policy.card)


def control():
    """One of each thing the observers watch, done on purpose."""
    for module in ('trainit_perception', 'trainit_motion_runtime', 'ament_index_python'):
        try:
            __import__(module)
        except Exception:                                   # noqa: BLE001
            pass
    for package in ('trainit_perception', 'trainit_motion_runtime'):
        try:
            from ament_index_python.packages import get_package_share_directory
            share = get_package_share_directory(package)
            with open(os.path.join(share, 'package.xml'), 'rb'):
                pass
        except Exception:                                   # noqa: BLE001 - uninstalled here
            pass
    for root in _FORBID:
        for dirpath, _dirs, names in os.walk(root):
            for n in names[:1]:
                with open(os.path.join(dirpath, n), 'rb'):
                    pass
            break
    with open(os.path.join(os.getcwd(), 'control_write.txt'), 'w') as fh:
        fh.write('a write outside the bundle, on purpose')


def run_case():
    import conftest  # noqa: F401  (the session's project rebasing; the first TSA import)
    import trainit_setup_assistant as tsa
    R['tsa_file'] = tsa.__file__
    R['tsa_file_real'] = os.path.realpath(tsa.__file__)
    R['tsa_version'] = tsa.__version__
    import identity_cases
    from trainit_setup_assistant.generator import Orchestrator
    project = identity_cases.CASES[args.case]()
    if args.alias:
        _realias(project)
    try:
        manifest = Orchestrator().generate(project, args.out)
        R['outcome'] = 'generated'
        R['warnings'] = list(manifest.warnings)
    except ValueError as exc:
        R['outcome'] = 'refused'
        R['refusal'] = {'type': type(exc).__name__, 'message': str(exc)}
    if os.path.isdir(args.out):
        R['files'], R['executables'] = _files(args.out)
        path = os.path.join(args.out, 'generation_manifest.yaml')
        if os.path.isfile(path):
            import yaml
            with open(path, 'rb') as fh:
                R['manifest'] = yaml.safe_load(fh.read())
    else:
        R['files'], R['executables'] = {}, []


try:
    if args.control:
        control()
        R['outcome'] = 'control'
    else:
        run_case()
except Exception as exc:                                    # noqa: BLE001 - a crash, reported
    R['outcome'] = 'crashed'
    R['crash'] = {'type': type(exc).__name__, 'message': str(exc),
                  'traceback': traceback.format_exc()[-4000:]}
R['modules_at_exit'] = {k: sorted(n for n in sys.modules if _family(n) == k) for k in WATCHED}
print('RESULT ' + json.dumps(R, sort_keys=True))
