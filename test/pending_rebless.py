"""The exact lines pinned bundles may differ by until the owner re-blesses them (owner Q1, vNext M2.4).

A generator change that moves the bytes of a pinned reference (the golden bundle, a pinned tree)
lists here, per file, the exact substitutions it makes: until the owner re-blesses the reference,
its gate is byte for byte EXCEPT these substitutions -- each one exact, each one required to apply
exactly once; any other difference fails. At the re-bless the pinned bytes take the new lines and
the list empties.

vNext M2.4 used it for the package.xml runtime requirement; the owner re-blessed the golden and the
policy reference trees on 2026-10-08, so the list is EMPTY: every gate that routes through it is
byte for byte.

Not collected by pytest (the name is not test_*.py).
"""
from typing import Dict, List, Tuple

Substitutions = Dict[str, List[Tuple[str, str]]]

# the golden bundle's pending lines, by bundle path: none since the M2.4 re-bless
GOLDEN: Substitutions = {}


def _apply(data: bytes, subs: List[Tuple[str, str]], rel: str, backward: bool) -> bytes:
    if not subs:
        return data                                  # not a pending file: its bytes, untouched
    text = data.decode('utf-8')
    for old, new in subs:
        find, put = (new, old) if backward else (old, new)
        assert text.count(find) == 1, (
            f'{rel}: the pending substitution does not apply exactly once '
            f'({text.count(find)} matches of {find!r})')
        text = text.replace(find, put)
    return text.encode('utf-8')


def forward(pinned: bytes, subs: Substitutions, rel: str) -> bytes:
    """A pinned file with the pending lines applied: what this TSA must generate."""
    return _apply(pinned, subs.get(rel, []), rel, backward=False)


def backward(generated: bytes, subs: Substitutions, rel: str) -> bytes:
    """A generated file with the pending lines put back: what the pinned tree recorded."""
    return _apply(generated, subs.get(rel, []), rel, backward=True)
