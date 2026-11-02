"""Where the files a test reads live -- without a machine path in any test.

Some of TSA's tests read files OUTSIDE this package: the golden bundle and the reference
application's robot description (both in ``fr3wml_digital_twin``, a sibling in the workspace),
an installed share, and a few fixtures that are not in this workspace at all. They used to name
all of them by absolute path, which runs on exactly one machine. They resolve them here instead:

* **the workspace** -- ``src/`` is two levels above this file
  (``<ws>/src/trainit_setup_assistant/test/``), or ``TRAINIT_WS_SRC`` when this repository was
  checked out somewhere else;
* **an installed share** -- through the ament index of the sourced workspace, falling back to the
  colcon layout next to ``src/``;
* **a fixture outside the workspace** -- an environment variable. Unset, the path is empty and the
  test that needs it SKIPS, naming the variable.

A test that needs something that is not there SKIPS and names it; nothing in this module raises.
The test says it: ``identity_cases.require(case)`` for a reference project or an input it records,
``eef_fixture.needs`` / ``skip_or_fail`` for a fixed path, ``catalog_identity.require_installed``
for a package that ``snapshot.yaml`` records facts of but that is not installed here. Those FAIL
instead with TRAINIT_REQUIRE_GOLDEN=1 (TESTING.md); the older plain skips
(``golden bundle not present`` and their like) only skip (vNext G0 P1). A clean checkout of this
repository alone therefore runs every test that needs nothing else.
"""
import os
from pathlib import Path

WS_SRC = Path(os.environ.get('TRAINIT_WS_SRC') or Path(__file__).resolve().parents[2])

# The golden bundle every byte-stability gate compares against (fr3wml_digital_twin).
GOLDEN_BUNDLE = WS_SRC / 'fr3wml_digital_twin' / 'fr3wml_suction_camera_tsa_4_1_bundle'


def ws_src(*parts: str) -> Path:
    """A path under the workspace's ``src/``."""
    return WS_SRC.joinpath(*parts)


def installed_share(package: str, *parts: str) -> Path:
    """A path under an installed package's share directory."""
    try:
        from ament_index_python.packages import get_package_share_directory
        return Path(get_package_share_directory(package)).joinpath(*parts)
    except Exception:          # not sourced, or not installed: the conventional layout
        return WS_SRC.parent.joinpath('install', package, 'share', package, *parts)


def external(variable: str) -> str:
    """A fixture outside this workspace, named by an environment variable ('' when unset)."""
    return os.environ.get(variable, '')


def in_workspace(path: str) -> str:
    """A path a reference project recorded on ANOTHER machine, rebased into this workspace.

    The projects these tests read -- the golden's project.yaml (frozen) and
    examples/fr3wml_project.yaml -- record the absolute paths of the machine that made them.
    A path that exists is returned UNCHANGED (on that machine this is a no-op); one that does
    not is rebased on its first component that is a top-level directory of this workspace's
    src/: ``/any/where/src/fr3wml_digital_twin/fr5_app/urdf`` becomes
    ``<WS_SRC>/fr3wml_digital_twin/fr5_app/urdf``. Anything else is returned unchanged -- a
    RELATIVE path too: it resolves beside its project file (vNext I0.0), nothing to rebase (its
    ``..`` used to be taken for a top-level directory of src/).
    """
    if not path or not Path(path).is_absolute() or Path(path).exists():
        return path
    parts = Path(path).parts
    for i, part in enumerate(parts[1:], start=1):
        if (WS_SRC / part).is_dir():
            return str(WS_SRC.joinpath(*parts[i:]))
    return path


def workspace_relative(path: str) -> str:
    """An ABSOLUTE path with its machine's workspace prefix removed, for comparing records.

    ``/home/x/ws/src/fr3wml_digital_twin/a.yaml`` -> ``fr3wml_digital_twin/a.yaml``. Relative
    strings (template names, source tags) come back unchanged.
    """
    if not path or not path.startswith('/'):
        return path
    parts = Path(path).parts
    for i, part in enumerate(parts[1:], start=1):
        if (WS_SRC / part).is_dir():
            return '/'.join(parts[i:])
    return path
