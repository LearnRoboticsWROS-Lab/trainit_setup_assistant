"""Shared: turn a missing Python dependency into one clear, honest, actionable message.

Every TrainIt entrypoint (GUI and headless) imports the core model/generator, which need a
handful of third-party packages. When one is missing we must (a) NAME the package, (b) give
the exact fix, and (c) NOT mislabel a core dependency as a "GUI" dependency. TrainIt Community
is designed to install via rosdep/apt (no pip): the model uses only the pydantic v1/v2-overlap
API, so ``python3-pydantic`` (1.9.x on Ubuntu 22.04) is enough.
"""

from __future__ import annotations

import sys


def _missing_name(exc: ImportError) -> str:
    """The importable module name that could not be loaded (e.g. 'pydantic', 'jinja2',
    'yaml', 'python_qt_binding')."""
    name = getattr(exc, 'name', None)
    return name or 'a required package'


def missing_dependency_message(exc: ImportError) -> str:
    """A self-explanatory error for a missing dependency: names it and gives the exact fix."""
    name = _missing_name(exc)
    return (
        f"error: missing required Python dependency '{name}'.\n"
        f"TrainIt Community needs it at runtime — a CORE dependency, not optional.\n"
        f"Fix (ROS 2 / Ubuntu, no pip needed): from your workspace root run\n"
        f"    rosdep install --from-paths src --ignore-src -r -y\n"
        f"  which reads package.xml and apt-installs the framework's dependencies\n"
        f"  (python3-pydantic, python3-jinja2, python3-yaml, python_qt_binding, ...).\n"
        f"Outside ROS you can install it directly (apt/pip package names may differ, "
        f"e.g. python3-pydantic on apt, pyyaml on pip for 'yaml')."
    )


def die_on_missing_dependency(exc: ImportError) -> int:
    """Print the actionable message and return a non-zero exit code (fail-fast)."""
    print(missing_dependency_message(exc), file=sys.stderr)
    return 1
