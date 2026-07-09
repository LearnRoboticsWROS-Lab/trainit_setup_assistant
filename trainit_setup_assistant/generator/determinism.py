"""Determinism helpers: stable hashing, a configured Jinja2 env, canonical YAML.

Reproducibility rule: same project + same generator => byte-identical output. We get
there with stable ordering (sorted file lists, ordered model traversal), fixed Jinja2
whitespace, and a canonical YAML dumper with a fixed float format.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader, StrictUndefined

TEMPLATES_DIR = Path(__file__).parent / 'templates'


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode('utf-8'))


def build_jinja_env() -> Environment:
    """Jinja2 env rooted at the package templates dir.

    ``keep_trailing_newline`` preserves the final newline ROS files expect; strict
    undefined turns a missing variable into an error instead of silent ``''``.
    """
    env = Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        keep_trailing_newline=True,
        trim_blocks=True,
        lstrip_blocks=True,
        undefined=StrictUndefined,
    )
    env.filters['fmtnum'] = fmtnum
    env.filters['fmtlist'] = fmtlist
    return env


def fmtnum(value: float) -> str:
    """Format a number the way the golden YAML does: ints stay ints, floats trim.

    ``2.0 -> '2.0'``, ``0.1 -> '0.1'``, ``-0.029 -> '-0.029'``, ``5 -> '5'``.
    """
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, int):
        return str(value)
    f = float(value)
    if f == int(f):
        return f'{f:.1f}'
    # round-trip via repr to avoid trailing FP noise, then strip
    return repr(f)


def fmtlist(values) -> str:
    """Format a numeric list inline: ``[0.556, -0.029, 0.167]``."""
    return '[' + ', '.join(fmtnum(v) for v in values) + ']'


class _CanonicalDumper(yaml.SafeDumper):
    """SafeDumper that keeps mapping insertion order (no key sorting)."""


def _represent_dict(dumper: yaml.Dumper, data: dict):
    return dumper.represent_mapping('tag:yaml.org,2002:map', data.items())


_CanonicalDumper.add_representer(dict, _represent_dict)


def canonical_yaml(data: Any, *, default_flow_style: bool = False) -> str:
    """Deterministic YAML dump (insertion order preserved, no key sorting)."""
    return yaml.dump(
        data,
        Dumper=_CanonicalDumper,
        sort_keys=False,
        default_flow_style=default_flow_style,
        width=4096,
    )
