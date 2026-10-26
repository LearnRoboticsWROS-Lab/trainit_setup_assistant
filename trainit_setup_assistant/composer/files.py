"""The project file as the Composer sees it (vNext I0.12, I0_DESIGN.md §20; owner decision D4).

- **D4: never destroy the only copy silently.** A GUI save rewrites the file with PyYAML, which drops
  YAML comments. The save asks first, and writes a backup (a byte copy) BEFORE overwriting, when the
  file on disk:
  - has comments, or
  - was changed by hand since TSA read it.

  It aborts if the backup cannot be written. A backup never overwrites another backup. Comment-preserving
  YAML is future UX debt.
- **External changes.** A hand edit of the YAML while TSA has the project open is detected by the file's
  fingerprint (mtime + SHA-256). The Composer offers "Reload from file", so the YAML and the GUI stay two
  views of ONE model.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from pathlib import Path
from typing import Callable, Optional, Tuple

from ..model.io import save_project
from ..model.project import CanonicalProject

Fingerprint = Tuple[float, str]


def fingerprint(path) -> Optional[Fingerprint]:
    p = Path(path)
    if not p.is_file():
        return None
    return p.stat().st_mtime, hashlib.sha256(p.read_bytes()).hexdigest()


def has_comments(text: str) -> bool:
    """Could the file hold a YAML comment? PRUDENT, as D4 needs: any ``#`` at the start of a line or
    after a blank counts, even one inside a quoted scalar. A false "yes" only asks once more and keeps a
    backup; a false "no" would lose a comment. Only a ``#`` glued to a word (``a#b``) is not counted."""
    return re.search(r'(^|[ \t])#', text, re.MULTILINE) is not None


def backup_path(path) -> Path:
    """``<file>.bak``, or a time-stamped name when that one exists: a backup never overwrites another."""
    p = Path(path)
    first = p.with_name(p.name + '.bak')
    if not first.exists():
        return first
    stamp = time.strftime('%Y%m%d-%H%M%S')
    candidate = p.with_name(f'{p.name}.{stamp}.bak')
    n = 1
    while candidate.exists():
        candidate = p.with_name(f'{p.name}.{stamp}-{n}.bak')
        n += 1
    return candidate


def save_with_backup(project: CanonicalProject, path, confirm: Callable[[str], bool],
                     changed_on_disk: bool = False) -> Tuple[bool, Optional[Path], str]:
    """Save ``project`` to ``path`` under D4. Returns ``(saved, backup, message)``.

    ``confirm(question)`` is asked when the file on disk has comments, or was changed since TSA read it
    (``changed_on_disk``). A "no" saves nothing; a "yes" first keeps the file as a backup."""
    if path is None:
        return False, None, 'not saved: the project has no file yet'
    p = Path(path)
    reasons = []
    if p.is_file() and has_comments(p.read_text(errors='replace')):
        reasons.append('it has YAML comments, and a save from TSA rewrites it WITHOUT them (the program is kept '
                       'exactly)')
    if p.is_file() and changed_on_disk:
        reasons.append('it was changed on disk since TSA read it, and a save replaces that change with the program '
                       'shown here')
    backup = None
    if reasons:
        backup = backup_path(p)
        if not confirm(f"{p.name}: {'; '.join(reasons)}. The current file is kept as {backup.name}. Save?"):
            return False, None, 'not saved: the file stays as it is'
        try:
            shutil.copyfile(p, backup)
        except OSError as exc:
            return False, None, f'not saved: the backup {backup} could not be written ({exc})'
    save_project(project, p)
    note = f' (the previous file is in {backup.name})' if backup else ''
    return True, backup, f'saved {p}{note}'


__all__ = ['Fingerprint', 'backup_path', 'fingerprint', 'has_comments', 'save_with_backup']
