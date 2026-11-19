"""P0.b PUB1 (T): what P0.b adds to the PUBLIC repositories names nothing private -- no private program's name, value
or asset, and no private (absolute) path (the frozen P0.b matrix, overlay docs/engineering/P0_CHECKPOINT.md).

Scanned, in every public repository P0.b touches that is checked out beside this one:

  trainit_setup_assistant, trainit_motion_runtime,   every line ADDED since the P0.a tag (the working tree against
  trainit_interfaces, trainit_data                   the tag, and the files git does not track yet) and every commit
                                                     message since it
  trainit_cell_io                                    (new in P0.b) every file and every commit message

against four rules: the publication guard's PATTERN; the private program's tokens; an absolute machine path; and a
captured-data payload file (the guard's PAYLOAD_EXT).

Like trainit_data's fixture scrub, the guard has two layers. The PUBLIC one is this file: the scanning logic, self-
tested with SYNTHETIC markers -- no protected word is written here, a guard must not put into public history the words
it exists to keep out. The PRIVATE one is read at RUN TIME from the ENVIRONMENT only, never copied into a repository
and never looked for at a default location: the guard's PATTERN from the pre-push hook TRAINIT_PUB1_HOOK names (unset:
this repository's installed pre-push hook, when there is one), and the private token list TRAINIT_PUB1_TOKENS names
(one token per line, '#' comments). Without them -- a fresh machine -- the scan SKIPS, naming the two variables; a
variable that names no file FAILS (a misspelt path never passes as a skip); the publication gate runs the guard
itself. A hit names the file, the line and the rule, never the matched text.
"""
import os
import re
import subprocess
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

import pytest

import ws_paths

REPO = Path(__file__).resolve().parents[1]        # this repository: its installed pre-push hook is the only fallback
HOOK_ENV, TOKENS_ENV = 'TRAINIT_PUB1_HOOK', 'TRAINIT_PUB1_TOKENS'
# the public repositories P0.b touches, and the P0.a publication tag their P0.b additions are counted from
SINCE = {'trainit_setup_assistant': 'v8.9.0', 'trainit_motion_runtime': 'v2.10.0', 'trainit_interfaces': 'v0.4.0',
         'trainit_data': 'v0.10.1'}
WHOLE = ('trainit_cell_io',)                      # new in P0.b: every file is P0.b's
# an absolute path of a machine (PRE_PUBLISH_CHECKLIST §12): a home directory, Linux or macOS
ABSOLUTE = re.compile(r'/(home|Users)/[A-Za-z]')
# the guard's own payload rule, when its hook cannot be read
PAYLOAD_DEFAULT = (r'\.(png|jpe?g|bmp|tiff?|exr|npy|npz|h5|hdf5|parquet|pcd|ply|bag|db3|mcap|mp4|avi|mkv|webm|pt|'
                   r'pth|ckpt|onnx|engine|safetensors)$')
# the self-test's markers: invented, never a real one
SYNTHETIC_PATTERN = 'pub1_synthetic_program|pub1[ _-]?blocked'
SYNTHETIC_TOKENS = ('pub1_private_joint', 'Pub1SecretCycle')

Hit = Tuple[str, int, str]                        # (file, line, rule)


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(['git', '-C', str(repo), *args], check=True, capture_output=True, text=True,
                          errors='replace').stdout


def _text_lines(path: Path) -> Iterator[Tuple[int, str]]:
    try:
        data = path.read_bytes()
    except OSError:
        return
    if b'\0' in data:                                # a binary file: the payload rule judges it
        return
    yield from enumerate(data.decode('utf-8', 'replace').splitlines(), 1)


def added_lines(repo: Path, since: Optional[str]) -> Iterator[Tuple[str, int, str]]:
    """(file, line, text) of every line P0.b adds: since ``since``, the working tree's diff to it and every line of the
    files git does not track yet (ignored ones aside); without ``since``, every line of every file."""
    if since is None:
        names = _git(repo, 'ls-files', '--cached', '--others', '--exclude-standard').splitlines()
        for name in names:
            for number, text in _text_lines(repo / name):
                yield name, number, text
        return
    diff = _git(repo, 'diff', '--no-color', '--no-ext-diff', '--unified=0', since, '--')
    name, number = None, 0
    for row in diff.splitlines():
        if row.startswith('+++ '):
            name = row[6:] if row.startswith('+++ b/') else None
        elif row.startswith('@@'):
            match = re.search(r'\+(\d+)', row)
            number = int(match.group(1)) if match else 0
        elif row.startswith('+') and name is not None:
            yield name, number, row[1:]
            number += 1
    for untracked in _git(repo, 'ls-files', '--others', '--exclude-standard').splitlines():
        for number, text in _text_lines(repo / untracked):
            yield untracked, number, text


def added_files(repo: Path, since: Optional[str]) -> List[str]:
    """The files P0.b adds or changes (the payload rule)."""
    if since is None:
        return _git(repo, 'ls-files', '--cached', '--others', '--exclude-standard').splitlines()
    changed = _git(repo, 'diff', '--name-only', '--diff-filter=AMR', since, '--').splitlines()
    return changed + _git(repo, 'ls-files', '--others', '--exclude-standard').splitlines()


def messages(repo: Path, since: Optional[str]) -> List[str]:
    """Every commit message P0.b adds (since ``since``, or all of the repository's)."""
    head = _git(repo, 'rev-parse', '--verify', '-q', 'HEAD').strip() if _has_commits(repo) else ''
    if not head:
        return []
    out = _git(repo, 'log', '--format=%H%x00%B%x01', f'{since}..HEAD' if since else 'HEAD')
    return [entry.split('\0', 1)[1].strip() for entry in out.split('\x01') if '\0' in entry]


def _has_commits(repo: Path) -> bool:
    return subprocess.run(['git', '-C', str(repo), 'rev-parse', '--verify', '-q', 'HEAD'],
                          capture_output=True).returncode == 0


def rules(pattern: str, tokens: List[str], payload: str) -> List[Tuple[str, re.Pattern]]:
    """The text rules, each with the name a hit reports (never the matched text)."""
    out = [('the guard pattern', re.compile(pattern, re.IGNORECASE))]
    for i, token in enumerate(tokens, 1):
        # a token inside a longer identifier counts (flanked by _ . / -), inside a longer word does not
        out.append((f'private token #{i}', re.compile(rf'(?<![A-Za-z0-9]){re.escape(token)}(?![A-Za-z0-9])',
                                                      re.IGNORECASE)))
    out.append(('an absolute machine path', ABSOLUTE))
    return out


def scan(repo: Path, since: Optional[str], text_rules: List[Tuple[str, re.Pattern]], payload: str) -> List[Hit]:
    hits: List[Hit] = []
    for name, number, text in added_lines(repo, since):
        for rule, regex in text_rules:
            if regex.search(text):
                hits.append((name, number, rule))
    for i, message in enumerate(messages(repo, since), 1):
        for number, text in enumerate(message.splitlines(), 1):
            for rule, regex in text_rules:
                if regex.search(text):
                    hits.append((f'commit message #{i}', number, rule))
    payload_re = re.compile(payload, re.IGNORECASE)
    hits += [(name, 0, 'a payload file') for name in added_files(repo, since) if payload_re.search(name)]
    return hits


# --- the private layer, read at run time from the environment only --------------------------------------------------

def _installed_hook() -> Optional[Path]:
    """This repository's installed pre-push hook, when there is one."""
    try:
        installed = Path(_git(REPO, 'rev-parse', '--git-path', 'hooks/pre-push').strip())
    except (subprocess.CalledProcessError, OSError):     # not a git checkout, or no git at all
        return None
    if not installed.is_absolute():
        installed = REPO / installed
    return installed if installed.is_file() else None


def _hook() -> Optional[Path]:
    """The guard's hook: the file TRAINIT_PUB1_HOOK names, else this repository's installed hook (None: neither)."""
    given = os.environ.get(HOOK_ENV)
    return Path(given) if given else _installed_hook()


def _token_list() -> Optional[Path]:
    """The private token list: the file TRAINIT_PUB1_TOKENS names -- never a default location (None: unset)."""
    given = os.environ.get(TOKENS_ENV)
    return Path(given) if given else None


def _hook_value(text: str, name: str) -> Optional[str]:
    match = re.search(rf"^{name}='([^']+)'", text, re.MULTILINE)
    return match.group(1) if match else None


def _tokens(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(errors='replace').splitlines()
            if line.strip() and not line.strip().startswith('#')]


def private_layer() -> Tuple[List[Tuple[str, re.Pattern]], str]:
    """The private layer's text rules and payload rule, from the environment; SKIP when a part is not set (a fresh
    machine), FAIL when a variable names no file or the files hold nothing to scan for."""
    for env in (HOOK_ENV, TOKENS_ENV):
        given = os.environ.get(env)
        if given and not Path(given).is_file():
            pytest.fail(f'{env} names no file: PUB1 cannot read the private layer it points to')
    hook, tokens = _hook(), _token_list()
    if hook is None or tokens is None:
        unset = ([f'{TOKENS_ENV} (the private token list) is not set'] if tokens is None else []) + \
                ([f'{HOOK_ENV} (the guard\'s pre-push hook) is not set and this repository has no installed one']
                 if hook is None else [])
        pytest.skip(f'PUB1 private layer absent: {"; ".join(unset)} -- it is read at run time from {HOOK_ENV} and '
                    f'{TOKENS_ENV} only (a fresh clone has neither); the publication gate runs the guard itself')
    text = hook.read_text(errors='replace')
    pattern = _hook_value(text, 'PATTERN')
    if pattern is None:
        pytest.fail(f'the guard hook ({HOOK_ENV} or the installed one) holds no PATTERN line: PUB1 cannot read the '
                    'guard pattern')
    listed = _tokens(tokens)
    if not listed:
        pytest.fail(f'the private token list ({TOKENS_ENV}) is empty: PUB1 would scan for nothing')
    payload = _hook_value(text, 'PAYLOAD_EXT') or PAYLOAD_DEFAULT
    return rules(pattern, listed, payload), payload


@pytest.fixture(scope='module')
def private():
    return private_layer()


@pytest.mark.parametrize('repo', list(SINCE) + list(WHOLE))
def test_pub1_what_p0b_adds_names_nothing_private(repo, private):
    text_rules, payload = private
    path = ws_paths.WS_SRC / repo
    if not (path / '.git').exists():
        pytest.skip(f'{repo} is not checked out beside this repository')
    since = SINCE.get(repo)
    if since is not None and subprocess.run(['git', '-C', str(path), 'rev-parse', '--verify', '-q',
                                             f'{since}^{{commit}}'], capture_output=True).returncode != 0:
        pytest.fail(f'{repo}: the P0.a tag {since} is missing, so PUB1 cannot tell what P0.b added')
    hits = scan(path, since, text_rules, payload)
    assert not hits, f'{repo}: {len(hits)} hit(s) -- ' + '; '.join(f'{f}:{n} ({rule})' for f, n, rule in hits)


# --- the public layer: the scanner, self-tested with SYNTHETIC markers ------------------------------------------------

def _repo(tmp: Path) -> Path:
    def git(*args):
        subprocess.run(['git', '-C', str(tmp), *args], check=True, capture_output=True)
    git('init', '-q')
    git('config', 'user.email', 'pub1@example.com')
    git('config', 'user.name', 'pub1 self-test')
    (tmp / 'kept.py').write_text('OLD = "pub1_synthetic_program was here before the tag"\n')   # before: not P0.b's
    git('add', '-A')
    git('commit', '-q', '-m', 'before')
    git('tag', 'v0')
    (tmp / 'kept.py').write_text('OLD = "pub1_synthetic_program was here before the tag"\nNEW = "neutral turner"\n'
                                 'JOINT = "a.pub1_private_joint_state"\n')
    home = '/' + 'home' + '/someone/ws'              # built here: a public file never holds a machine path
    (tmp / 'new.py').write_text('CYCLE = "pub1secretcycle"\nWORDS = "pub1_private_jointed"\n'
                                f'WHERE = "{home}"\nNEUTRAL = "/opt/ros/humble"\n')
    (tmp / 'frame.png').write_bytes(b'\x89PNG\0\0')
    git('add', 'kept.py')
    git('commit', '-q', '-m', 'add the pub1 blocked program\n\nneutral body')
    return tmp


def test_pub1_the_scanner_finds_each_rule_and_only_what_was_added(tmp_path):
    repo = _repo(tmp_path)
    text_rules = rules(SYNTHETIC_PATTERN, list(SYNTHETIC_TOKENS), PAYLOAD_DEFAULT)
    hits = sorted(scan(repo, 'v0', text_rules, PAYLOAD_DEFAULT))
    assert hits == sorted([
        ('kept.py', 3, 'private token #1'),           # inside an identifier (flanked by . and _): counted
        ('new.py', 1, 'private token #2'),            # case-insensitive
        ('new.py', 3, 'an absolute machine path'),
        ('commit message #1', 1, 'the guard pattern'),
        ('frame.png', 0, 'a payload file'),
    ]), hits
    # the line that was there before the tag is not P0.b's; a token inside a longer WORD is not the token
    assert not [h for h in hits if h[0] == 'kept.py' and h[1] == 1]
    assert not [h for h in hits if h[0] == 'new.py' and h[1] == 2]
    # a NEW repository: every file and every message is P0.b's (the line before the tag too)
    whole = scan(repo, None, text_rules, PAYLOAD_DEFAULT)
    assert ('kept.py', 1, 'the guard pattern') in whole


def test_pub1_the_private_markers_are_read_at_run_time_never_written_here():
    """This file names the private layer by its environment variables only -- never a private file's location or
    content; the synthetic markers are the only ones it holds."""
    text = Path(__file__).read_text()
    assert ABSOLUTE.search(text) is None              # no machine path in a public file
    tokens = _token_list()
    if tokens is not None and tokens.is_file():
        for i, token in enumerate(_tokens(tokens), 1):
            assert token.lower() not in text.lower(), f'private token #{i} is written in this file'


def test_pub1_the_private_layer_comes_from_the_environment_only(monkeypatch, tmp_path):
    """Unset, the variables are NAMED by the skip and no default location is consulted (the hook's only fallback is
    this repository's own installed hook); set, the files they name are read; a variable naming no file fails."""
    monkeypatch.delenv(HOOK_ENV, raising=False)
    monkeypatch.delenv(TOKENS_ENV, raising=False)
    assert _token_list() is None
    hook = _hook()
    assert hook is None or hook == _installed_hook(), 'the hook comes from the environment or this repository only'
    with pytest.raises(pytest.skip.Exception) as skipped:
        private_layer()
    assert f'{TOKENS_ENV} (the private token list) is not set' in str(skipped.value), skipped.value
    assert HOOK_ENV in str(skipped.value), skipped.value
    # set: the synthetic layer is read from the files the variables name
    (tmp_path / 'hook').write_text(f"#!/bin/sh\nPATTERN='{SYNTHETIC_PATTERN}'\n")
    (tmp_path / 'tokens').write_text('# synthetic\n' + '\n'.join(SYNTHETIC_TOKENS) + '\n')
    monkeypatch.setenv(HOOK_ENV, str(tmp_path / 'hook'))
    monkeypatch.setenv(TOKENS_ENV, str(tmp_path / 'tokens'))
    text_rules, payload = private_layer()
    assert payload == PAYLOAD_DEFAULT                 # the hook states no PAYLOAD_EXT: the guard's own rule
    assert [name for name, _rx in text_rules] == ['the guard pattern', 'private token #1', 'private token #2',
                                                  'an absolute machine path']
    assert text_rules[0][1].search('the PUB1 Blocked program') and text_rules[1][1].search('x.pub1_private_joint')
    # a variable that names no file is a FAILURE, never a skip
    monkeypatch.setenv(TOKENS_ENV, str(tmp_path / 'missing'))
    with pytest.raises(pytest.fail.Exception, match=TOKENS_ENV):
        private_layer()
