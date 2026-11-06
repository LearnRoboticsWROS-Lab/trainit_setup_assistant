"""The OFFLINE side of TSA's Data section (TrainIt D0.8, ADR-0029 Decision 7 + Clarification D0.8): launched
processes TSA shows, observes and parses -- never a runtime of its own (no Qt, no ROS here).

- The canonical data engine: ``python3 -m trainit_data --json ...`` (never ``ros2 run``: its stdout is not
  machine-clean), one ``trainit-cli/1`` object on stdout, its stderr and exit status kept for the Advanced
  details. TSA never imports ``trainit_data`` and never touches the store itself.
- The recorder: ``ros2 launch trainit_data_ros recorder.launch.py ...`` as a launched, copyable command (when the
  bundle's bring-up does not already run it).
- A synthetic PROVIDER (``trainit-synthetic-provider/1.0``): found by its ``provider.json`` in the installed
  ``share/trainit_data/synthetic_providers/`` and the machine-local extra directories; its form rendered and its
  values validated FROM the descriptor (TSA's own strict reader of that contract); its resolved configuration
  written; the provider launched as an external process whose ``trainit-provider-progress/1`` lines TSA reads.
  The provider validates the same configuration again before its runtime starts.
"""

from __future__ import annotations

import copy
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .launch_plan import Command

CLI_SCHEMA = 'trainit-cli/1'
PROVIDER_SCHEMA = 'trainit-synthetic-provider'
PROVIDER_MAJOR = 1
PROGRESS_PROTOCOL = 'trainit-provider-progress/1'
DESCRIPTOR_FILE = 'provider.json'
TYPES = ('int', 'float', 'int_range', 'float_range', 'float_pair', 'choice')
GROUPS = ('objects', 'lighting', 'appearance', 'camera', 'scene', 'quality')
LEVELS = ('simple', 'advanced')
GENERAL = ('seed', 'runs', 'views')
# The runtime kinds TSA can launch a provider with: kind -> how THIS machine's settings resolve its interpreter
# (None + the reason when it is not installed). A new runtime kind is one row here; the panel, the validator and
# the canonical project never name a runtime.
RUNTIME_LAUNCHERS = {}
_IDENT = re.compile(r'^[a-z0-9]+(?:_[a-z0-9]+)*$')


# ---- the data store and the CLI ---------------------------------------------------------------------------

def data_root(settings=None) -> str:
    """The store this machine uses: the TSA setting, else ``TRAINIT_DATA_ROOT``, else ``~/TrainItData``."""
    configured = getattr(settings, 'data_root', '') if settings is not None else ''
    return str(Path(configured or os.environ.get('TRAINIT_DATA_ROOT') or '~/TrainItData').expanduser())


@dataclass
class CliResult:
    """One ``trainit-data --json`` run: the parsed ``trainit-cli/1`` object (or why there is none)."""

    argv: List[str]
    exit_code: Optional[int]
    ok: bool = False
    command: str = ''
    result: Optional[dict] = None
    error: Optional[dict] = None
    stderr: str = ''
    stdout: str = ''

    def display(self) -> str:
        return ' '.join(shlex.quote(a) for a in self.argv)

    def message(self) -> str:
        """One line for a person: the refusal, or what went wrong running the tool."""
        if self.error:
            return f'{self.error.get("code")}: {self.error.get("message")}'
        if self.result is None:
            tail = (self.stderr.strip().splitlines() or ['no output'])[-1]
            return f'trainit-data did not answer (exit {self.exit_code}): {tail}'
        return 'ok' if self.ok else 'finished with findings'


def cli_argv(args: Sequence[str], root: Optional[str] = None, python: str = 'python3') -> List[str]:
    argv = [python, '-m', 'trainit_data', '--json']
    if root:
        argv += ['--root', str(root)]
    return argv + [str(a) for a in args]


def parse_cli_output(argv: List[str], exit_code: Optional[int], stdout: str, stderr: str) -> CliResult:
    out = CliResult(argv=argv, exit_code=exit_code, stderr=stderr, stdout=stdout)
    try:
        doc = json.loads(stdout)
    except ValueError:
        return out
    if not isinstance(doc, dict) or doc.get('schema') != CLI_SCHEMA:
        return out
    out.ok = bool(doc.get('ok')) and doc.get('error') is None
    out.command = doc.get('command') or ''
    out.result = doc.get('result')
    out.error = doc.get('error')
    return out


def run_cli(args: Sequence[str], root: Optional[str] = None, timeout_s: float = 600.0,
            env: Optional[Dict[str, str]] = None) -> CliResult:
    """Run ``trainit-data --json`` to completion (call it from a worker thread, never the GUI thread)."""
    argv = cli_argv(args, root)
    try:
        proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout_s,
                              env=dict(os.environ, **(env or {})), stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CliResult(argv=argv, exit_code=None, stderr=f'{type(exc).__name__}: {exc}')
    return parse_cli_output(argv, proc.returncode, proc.stdout, proc.stderr)


def cli_version(timeout_s: float = 30.0) -> Optional[str]:
    """``trainit-data``'s version when the offline tool answers here, else None (capability detection)."""
    try:
        proc = subprocess.run(['python3', '-m', 'trainit_data', '--version'], capture_output=True, text=True,
                              timeout=timeout_s, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.TimeoutExpired):
        return None
    words = proc.stdout.split()
    return words[-1] if proc.returncode == 0 and words else None


def recorder_command(ws_root: Optional[str], config_file: str, namespace: str, use_sim_time: bool,
                     root: str) -> Command:
    """The recorder as a launched, copyable command (when the bundle's bring-up does not run it)."""
    from .launch_plan import _sourced
    # no slot = the launch file's default: `ros2 launch` refuses an empty `namespace:=` (owner acceptance flow 1)
    args = (f'config_file:={shlex.quote(str(config_file))} '
            + (f'namespace:={shlex.quote(namespace)} ' if namespace else '')
            + f'use_sim_time:={"true" if use_sim_time else "false"}')
    return Command('data_recorder', 'Data recorder', _sourced(ws_root, f'ros2 launch trainit_data_ros '
                                                                       f'recorder.launch.py {args}'),
                   long_running=True, env={'TRAINIT_DATA_ROOT': str(root)})


# ---- synthetic providers ----------------------------------------------------------------------------------

class DescriptorError(ValueError):
    """A provider descriptor TSA cannot use (strict: unknown keys, bad types, another major)."""


def _fail(where: str, message: str):
    raise DescriptorError(f'{where}: {message}')


def _keys(value, where, required, optional=()):
    if not isinstance(value, dict):
        _fail(where, 'expected a JSON object')
    missing = [k for k in required if k not in value]
    extra = [k for k in value if k not in required and k not in optional]
    if missing or extra:
        _fail(where, f'missing {missing}' if missing else f'unknown field(s) {extra}')
    return value


def _is_number(value, scalar):
    if isinstance(value, bool):
        return False
    return isinstance(value, int) if scalar == 'int' else isinstance(value, (int, float))


def _parameter(value, where, general=False) -> dict:
    optional = ('unit', 'help') + (('type',) if general else ())
    base = ('label',) + (('default',) if general else ('key', 'type', 'group', 'level'))
    _keys(value, where, base, optional + ('min', 'max', 'values'))
    kind = 'int' if general else value['type']
    if general and value.get('type', 'int') != 'int':
        _fail(where, 'seed / runs / views are integers')
    if kind not in TYPES:
        _fail(where, f'type {kind!r} is not one of {", ".join(TYPES)}')
    out = {k: v for k, v in value.items()}
    out['type'] = kind
    if kind == 'choice':
        if not isinstance(value.get('values'), list) or not value['values'] or 'min' in value or 'max' in value:
            _fail(where, 'a choice lists its values (and has no min / max)')
    else:
        scalar = 'int' if kind.startswith('int') else 'float'
        if 'values' in value or not all(_is_number(value.get(b), scalar) for b in ('min', 'max')):
            _fail(where, f'a {kind} has numeric min and max')
        if value['min'] > value['max']:
            _fail(where, 'min > max')
    if not general:
        if not isinstance(value['key'], str) or any(not p for p in value['key'].split('.')):
            _fail(where, f'{value["key"]!r} is not a dotted key path')
        if value['group'] not in GROUPS or value['level'] not in LEVELS:
            _fail(where, f'group / level {value["group"]!r} / {value["level"]!r} unknown')
    if general and value_problem(out, value['default']):
        _fail(where, f'default {value_problem(out, value["default"])}')
    for text in ('label', 'unit', 'help'):
        if text in value and not isinstance(value[text], str):
            _fail(where, f'{text} must be a string')
    return out


def parse_descriptor(doc, where: str = DESCRIPTOR_FILE) -> dict:
    """TSA's strict reader of ``trainit-synthetic-provider/1.x`` (the contract, not the provider's code)."""
    _keys(doc, where, ('schema', 'id', 'name', 'version', 'description', 'runtime', 'backends', 'entry',
                       'base_config', 'general', 'parameters', 'outputs', 'progress'))
    name, _, version = str(doc['schema']).partition('/')
    if name != PROVIDER_SCHEMA or version.split('.')[0] != str(PROVIDER_MAJOR):
        _fail(where, f'schema {doc["schema"]!r}: this TSA reads {PROVIDER_SCHEMA}/{PROVIDER_MAJOR}.x')
    if not isinstance(doc['id'], str) or not _IDENT.match(doc['id']):
        _fail(where, f'id {doc["id"]!r} is not a provider id')
    for text in ('name', 'version', 'description'):
        if not isinstance(doc[text], str) or not doc[text].strip():
            _fail(where, f'{text} must be a non-empty string')
    if any(not isinstance(b, str) or not _IDENT.match(b) for b in doc['backends'] or [None]) \
            or len(set(doc['backends'])) != len(doc['backends']):
        _fail(where, 'backends: unique deployment backend ids')
    runtime = _keys(doc['runtime'], f'{where}.runtime', ('kind',), ('min_version',))
    if runtime['kind'] not in RUNTIME_LAUNCHERS:
        _fail(where, f'runtime {runtime["kind"]!r} is not one TSA can launch ({", ".join(RUNTIME_LAUNCHERS)})')
    if not isinstance(doc['backends'], list) or not doc['backends']:
        _fail(where, 'backends: at least one deployment backend')
    for rel in ('entry', 'base_config'):
        value = doc[rel]
        if not isinstance(value, str) or value.startswith('/') or '..' in value.split('/'):
            _fail(where, f'{rel} {value!r} is not a path inside the provider')
    general = _keys(doc['general'], f'{where}.general', GENERAL)
    out = dict(doc)
    out['general'] = {k: _parameter(general[k], f'{where}.general.{k}', general=True) for k in GENERAL}
    if not isinstance(doc['parameters'], list):
        _fail(where, 'parameters must be a list')
    out['parameters'] = [_parameter(p, f'{where}.parameters[{i}]') for i, p in enumerate(doc['parameters'])]
    keys = [p['key'] for p in out['parameters']]
    if len(set(keys)) != len(keys):
        _fail(where, 'a parameter key is declared twice')
    outputs = _keys(doc['outputs'], f'{where}.outputs', ('modalities', 'annotations', 'classes'))
    for name, items in outputs.items():
        if not isinstance(items, list) or any(not isinstance(i, str) for i in items) or len(set(items)) != len(items):
            _fail(where, f'outputs.{name}: a list of unique names')
    if doc['progress'] != PROGRESS_PROTOCOL:
        _fail(where, f'progress {doc["progress"]!r}: this TSA reads {PROGRESS_PROTOCOL}')
    return out


def value_problem(parameter: dict, value) -> Optional[str]:
    """None if ``value`` meets the parameter's type and bounds, else a short reason (the provider's rule)."""
    kind = parameter['type']
    if kind == 'choice':
        return None if value in parameter['values'] else f'{value!r} is not one of {parameter["values"]}'
    lo, hi = parameter['min'], parameter['max']
    scalar = 'int' if kind.startswith('int') else 'float'
    if kind in ('int', 'float'):
        if not _is_number(value, scalar):
            return f'{value!r} is not {"an integer" if scalar == "int" else "a number"}'
        return None if lo <= value <= hi else f'{value} is outside [{lo}, {hi}]'
    if not isinstance(value, list) or len(value) != 2 or not all(_is_number(v, scalar) for v in value):
        return f'{value!r} is not a pair of {"integers" if scalar == "int" else "numbers"}'
    if not all(lo <= v <= hi for v in value):
        return f'{value} is outside [{lo}, {hi}]'
    if kind != 'float_pair' and value[0] > value[1]:
        return f'{value}: the lower bound exceeds the upper one'
    return None


def get_path(config: dict, key: str):
    value = config
    for part in key.split('.'):
        if not isinstance(value, dict):
            raise KeyError(key)
        value = value[part]
    return value


def _set_path(config: dict, key: str, value) -> None:
    *parents, leaf = key.split('.')
    node = config
    for part in parents:
        node = node[part]
    node[leaf] = value


@dataclass
class Provider:
    directory: Path
    descriptor: dict
    base_config: dict = field(default_factory=dict)

    @property
    def id(self) -> str:
        return self.descriptor['id']

    @property
    def name(self) -> str:
        return self.descriptor['name']

    def parameter(self, key: str) -> dict:
        return next(p for p in self.descriptor['parameters'] if p['key'] == key)

    def default(self, key: str):
        return copy.deepcopy(get_path(self.base_config, key))

    def problems(self, parameters: Dict[str, Any], seed=None, runs=None, views=None) -> List[str]:
        """Every value TSA would send that the descriptor refuses (an unknown key too). Empty = valid."""
        out = []
        known = {p['key']: p for p in self.descriptor['parameters']}
        for key, value in parameters.items():
            if key not in known:
                out.append(f'{key}: provider {self.id} has no such parameter')
                continue
            problem = value_problem(known[key], value)
            if problem:
                out.append(f'{known[key]["label"]}: {problem}')
        for name, value in (('seed', seed), ('runs', runs), ('views', views)):
            if value is not None:
                problem = value_problem(self.descriptor['general'][name], value)
                if problem:
                    out.append(f'{self.descriptor["general"][name]["label"]}: {problem}')
        return out

    def resolved_config(self, parameters: Dict[str, Any]) -> dict:
        """The base configuration with the given parameters set (what the provider is launched with)."""
        config = copy.deepcopy(self.base_config)
        for key, value in parameters.items():
            _set_path(config, key, copy.deepcopy(value))
        return config


def load_provider(directory) -> Provider:
    directory = Path(directory)
    try:
        doc = json.loads((directory / DESCRIPTOR_FILE).read_text())
    except (OSError, ValueError) as exc:
        raise DescriptorError(f'{directory / DESCRIPTOR_FILE}: not a readable JSON descriptor ({exc})') from None
    descriptor = parse_descriptor(doc, str(directory / DESCRIPTOR_FILE))
    try:
        base = json.loads((directory / descriptor['base_config']).read_text())
    except (OSError, ValueError) as exc:
        raise DescriptorError(f'{directory}: base configuration unreadable ({exc})') from None
    provider = Provider(directory, descriptor, base)
    for parameter in descriptor['parameters']:
        try:
            problem = value_problem(parameter, get_path(base, parameter['key']))
        except KeyError:
            problem = 'absent from the base configuration'
        if problem:
            raise DescriptorError(f'{directory}: parameter {parameter["key"]}: {problem}')
    if not (directory / descriptor['entry']).is_file():
        raise DescriptorError(f'{directory}: entry {descriptor["entry"]} is missing')
    return provider


def provider_dirs(settings=None) -> List[Path]:
    """Where providers are looked for: the installed ``share/trainit_data/synthetic_providers`` (when
    trainit_data is on the ament index), then the machine-local extra directories."""
    out = []
    try:
        from ament_index_python.packages import get_package_share_directory
        out.append(Path(get_package_share_directory('trainit_data')) / 'synthetic_providers')
    except Exception:  # noqa: BLE001 - not installed / not sourced: only the extra directories
        pass
    extra = getattr(settings, 'synthetic_provider_dirs', '') if settings is not None else ''
    out += [Path(p).expanduser() for p in extra.split(os.pathsep) if p]
    return out


def find_providers(dirs: Sequence[Path]) -> Tuple[List[Provider], List[str]]:
    """(the providers found, why a candidate was not usable). A directory is a provider when it holds
    ``provider.json``; otherwise its sub-directories are searched (one level)."""
    providers, problems, seen = [], [], set()
    for base in dirs:
        try:
            candidates = [base] if (base / DESCRIPTOR_FILE).is_file() else (
                sorted(p for p in base.iterdir() if (p / DESCRIPTOR_FILE).is_file()) if base.is_dir() else [])
        except OSError as exc:
            problems.append(f'{base}: cannot be searched ({exc})')
            continue
        for directory in candidates:
            try:
                provider = load_provider(directory)
            except DescriptorError as exc:
                problems.append(str(exc))
                continue
            if provider.id in seen:
                problems.append(f'{directory}: provider {provider.id} is already found earlier in the search path')
                continue
            seen.add(provider.id)
            providers.append(provider)
    return providers, problems


def isaac_python(isaac_launcher: str) -> Optional[Path]:
    """Isaac Sim's ``python.sh`` beside its ``isaac-sim.sh`` launcher (the TSA setting), or None."""
    if not isaac_launcher:
        return None
    path = Path(isaac_launcher).expanduser().with_name('python.sh')
    return path if path.is_file() else None


def _isaac_sim_python(settings) -> Tuple[Optional[Path], str]:
    python = isaac_python(getattr(settings, 'isaac_launcher', '') or '')
    return python, ('' if python else 'Isaac Sim is not installed here, or its launcher is not set (Settings: '
                                      'isaac-sim.sh, with python.sh beside it)')


RUNTIME_LAUNCHERS['isaac_sim_python'] = _isaac_sim_python


def runtime_python(provider: Provider, settings) -> Tuple[Optional[Path], str]:
    """The interpreter this machine runs the provider's runtime kind with, or (None, why not)."""
    resolve = RUNTIME_LAUNCHERS.get(provider.descriptor['runtime']['kind'])
    if resolve is None:
        return None, f'runtime {provider.descriptor["runtime"]["kind"]!r} is not one TSA can launch'
    return resolve(settings)


def provider_unavailable(provider: Provider, modes: Sequence[str], settings) -> Optional[str]:
    """Why this provider cannot run for this project on this machine (None = available)."""
    served = [b for b in provider.descriptor['backends'] if b in modes]
    if not served:
        return (f'{provider.name} generates data for {", ".join(provider.descriptor["backends"])} cells; this '
                f'project deploys {", ".join(modes) or "nothing"}')
    python, why = runtime_python(provider, settings)
    return None if python is not None else why


def provider_command(provider: Provider, settings, config_file: str, seed: int, runs: int, views: int,
                     root: str, store: Optional[str] = None, title: str = 'Synthetic generation',
                     kind: str = 'synthetic') -> Command:
    """The provider as an external process (``env -i``: a sourced ROS environment stays out of its runtime)."""
    python, why = runtime_python(provider, settings)
    if python is None:
        raise ValueError(why)
    entry = provider.directory / provider.descriptor['entry']
    args = [str(python), str(entry), '--config', str(config_file), '--seed', str(seed), '--runs', str(runs),
            '--views', str(views)]
    if store:
        args += ['--store', str(store)]
    script = ('exec env -i HOME="$HOME" PATH=/usr/bin:/bin TRAINIT_DATA_ROOT=' + shlex.quote(str(root)) + ' '
              + ' '.join(shlex.quote(a) for a in args))
    return Command(kind, title, script, long_running=True)


def parse_progress(line: str) -> Optional[dict]:
    """One ``trainit-provider-progress/1`` line -> ``{phase, ...}``; None for any other output."""
    line = line.strip()
    if line.startswith('TRAINIT_PROGRESS '):
        try:
            doc = json.loads(line[len('TRAINIT_PROGRESS '):])
        except ValueError:
            return None
        return doc if isinstance(doc, dict) and isinstance(doc.get('phase'), str) else None
    if line.startswith('TRAINIT_SESSION '):
        return {'phase': 'done', 'session_id': line.split(None, 1)[1].strip()}
    if line == 'TRAINIT_FAILED':
        return {'phase': 'failed'}
    return None


_SYNC_PROBLEMS = {'OUT_OF_TOLERANCE': 'taken {skew} ms apart (allowed {tol} ms)',
                  'INCOMPATIBLE_CLOCKS': 'stamped on different clocks'}


def _ms(ns) -> str:
    return f'{(ns or 0) / 1e6:.0f}'


def capture_problems(captures: Sequence[dict]) -> List[str]:
    """``inspect --captures`` records -> one line per capture that is not COMPLETE, saying why in the recorder's
    own words (its stream reasons and sync verdict); [] when every capture is complete."""
    lines = []
    for c in captures:
        if not isinstance(c, dict) or c.get('outcome') == 'COMPLETE':
            continue
        why = [f'{m.get("stream")}: {m.get("reason")}' for m in c.get('missing') or [] if isinstance(m, dict)]
        sync = c.get('sync') or {}
        if sync.get('status') in _SYNC_PROBLEMS:
            text = _SYNC_PROBLEMS[sync['status']].format(skew=_ms(sync.get('max_skew_ns')),
                                                         tol=_ms(sync.get('tolerance_ns')))
            why.append(f'streams {text} [{sync["status"]}]')
        trigger = (c.get('trigger') or {}).get('trigger_id')
        lines.append(f'#{c.get("seq")} {c.get("outcome")}' + (f' ({trigger})' if trigger else '') + ': '
                     + ('; '.join(why) or 'no reason recorded'))
    return lines


__all__ = ['CliResult', 'DescriptorError', 'Provider', 'capture_problems', 'cli_argv', 'cli_version', 'data_root',
           'find_providers', 'isaac_python', 'load_provider', 'parse_cli_output', 'parse_descriptor', 'parse_progress',
           'provider_command', 'provider_dirs', 'provider_unavailable', 'recorder_command', 'run_cli',
           'value_problem']
