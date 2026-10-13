"""Where a bundle's files came from, in words that name no machine (vNext M2.3, ADR-0021).

A bundle's manifest says what produced each file. A file COPIED from the project's inputs used to
be recorded by its absolute path on the generating machine (``/home/<user>/<ws>/src/...``): the
same project gave another manifest on every machine, and a manifest shared with a bundle carried
the user's home and the layout of a private workspace. :func:`portable_source` records it instead
as the package that holds it and the path inside that package, or by its file name alone:

  ``package://<name>/<path>``  the nearest directory above the file with a ``package.xml`` is a
                               ROS package; ``<name>`` is that package.xml's ``<name>`` and
                               ``<path>`` the file's path inside the package (POSIX). The file is
                               named as the project names it: a symlink is NOT resolved -- its
                               target, which the bundle never names, stays out of the manifest
                               -- and a workspace reached through a link names the same package;
                               a '..' is applied the way the OS reads it (after a link, from the
                               link's target).
  ``external:<file name>``     no package.xml above the file; or the nearest one is not a valid
                               package manifest (then one manifest warning names its directory,
                               without a path, and the walk stops there)

TSA's own resources are ``package://trainit_setup_assistant/trainit_setup_assistant/<path>``
however TSA is installed (an installed copy has no package.xml above it; a symlink to TSA is
resolved for this check alone): the record depends on the file, not on where TSA was imported
from. A DIRECTORY (``directory=True``, the warnings about an input directory) is named the same
way, its own package.xml included: a package root is ``package://<name>/``.

A ``package://`` record is PROVENANCE -- which package, which file in its tree -- not a promise
that ament can resolve it. Two different files may share an ``external:`` name; the entry's path
and sha256 tell them apart. The walk reads only the package.xml files above the copied files (the
project's own inputs): never an installed component, never the ament index, never the environment.
"""
from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from pathlib import PurePath
from typing import Dict, List, Optional, Tuple, Union

# the trainit_setup_assistant Python package (generator/ is one of its subpackages)
_TSA_PACKAGE = os.path.realpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_TSA_PREFIX = 'package://trainit_setup_assistant/trainit_setup_assistant/'
# REP 149 / catkin_pkg: a name it accepts (lowercase is only recommended)
_PACKAGE_NAME = re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]*$')

PathLike = Union[str, 'os.PathLike[str]']


def _package_name(manifest: str) -> Optional[str]:
    """The <name> of a valid package.xml, or None."""
    try:
        root = ET.parse(manifest).getroot()
    except (ET.ParseError, OSError, UnicodeDecodeError, ValueError):
        return None
    names = root.findall('name') if root.tag == 'package' else []
    name = (names[0].text or '').strip() if len(names) == 1 else ''
    return name if _PACKAGE_NAME.match(name) else None


class SourceNamer:
    """:func:`portable_source` for one generation: what it read is remembered for that generation
    only, and an unusable package.xml is reported once, in :attr:`problems` (path-free)."""

    def __init__(self) -> None:
        self._roots: Dict[str, Optional[Tuple[str, Optional[str]]]] = {}
        self.problems: List[str] = []

    def __call__(self, path: PathLike, directory: bool = False) -> str:
        given = _as_read(path)
        real = os.path.realpath(given)
        if real == _TSA_PACKAGE or real.startswith(_TSA_PACKAGE + os.sep):
            return _TSA_PREFIX + _inside(real, _TSA_PACKAGE)
        found = self._package_of(given if directory else os.path.dirname(given))
        if found is None or found[1] is None:
            return 'external:' + os.path.basename(given)
        root, name = found
        return f'package://{name}/' + _inside(given, root)

    def _package_of(self, directory: str) -> Optional[Tuple[str, Optional[str]]]:
        """(package root, its name -- None if the package.xml is unusable) or None: no package."""
        if directory in self._roots:
            return self._roots[directory]
        if os.path.isfile(os.path.join(directory, 'package.xml')):
            name = _package_name(os.path.join(directory, 'package.xml'))
            if name is None:
                self.problems.append(
                    f'copied files are recorded as external:<file name>: the package.xml in a '
                    f'directory named "{os.path.basename(directory)}" is not a valid ROS package '
                    f'manifest')
            found: Optional[Tuple[str, Optional[str]]] = (directory, name)
        else:
            parent = os.path.dirname(directory)
            found = None if parent == directory else self._package_of(parent)
        self._roots[directory] = found
        return found

    def take_problems(self) -> List[str]:
        out, self.problems = self.problems, []
        return out


def _as_read(path: PathLike) -> str:
    """The absolute path the OS reads (a relative one from the cwd), with link names kept: a
    '..' is applied the way the OS applies it -- after the links before it -- so only the part up
    to the last '..' is resolved; the rest stays as the project names it."""
    raw = os.fspath(path)
    if '..' not in raw.split(os.sep):
        return os.path.abspath(raw)
    parts = (raw if os.path.isabs(raw) else os.path.join(os.getcwd(), raw)).split(os.sep)
    last = len(parts) - 1 - parts[::-1].index('..')
    head = os.path.realpath(os.sep.join(parts[:last + 1]) or os.sep)
    return os.path.normpath(os.path.join(head, *parts[last + 1:]))


def _inside(path: str, root: str) -> str:
    rel = os.path.relpath(path, root)
    return '' if rel == '.' else PurePath(rel).as_posix()


def portable_source(path: PathLike, directory: bool = False) -> str:
    """Where a file (or directory) comes from, as a manifest records it (module docstring)."""
    return SourceNamer()(path, directory=directory)
