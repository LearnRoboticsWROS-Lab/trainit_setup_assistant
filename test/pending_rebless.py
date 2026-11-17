"""The exact lines pinned bundles may differ by until the owner re-blesses them (owner Q1, vNext M2.4).

A generator change that moves the bytes of a pinned reference (the golden bundle, a pinned tree)
lists here, per file, the exact substitutions it makes: until the owner re-blesses the reference,
its gate is byte for byte EXCEPT these substitutions -- each one exact, each one required to apply
exactly once; any other difference fails. At the re-bless the pinned bytes take the new lines and
the list empties.

vNext M2.4 used it for the package.xml runtime requirement; the owner re-blessed the golden and the
policy reference trees on 2026-10-08, so the list was EMPTY.

TSA 8.6.1 (the M4.5 Isaac benchmark's launch defect) lists the golden's two launch files: its
trainit_bt.launch.py starts a detector_node beside trainit_run_bt, so the launch never ended with
the tree; it now shuts itself down when the runner exits (``end_with_tree``, default true), and the
app bringup.launch.py that includes it passes ``end_with_tree: false`` (the cell stays up after the
tree, as before). Nothing else of the golden moves (generation_manifest.yaml: those two files'
sha256 only). The policy reference trees (the golden project, generated as 'cell') move in the same two
files: POLICY_REFERENCE. The owner re-blessed the golden and re-pinned the policy reference trees at M4.5b
(PB-7, 2026-11-10: the two launch files and their two manifest sha256, each changed line verified against a temp
candidate first), so the lists are EMPTY again.

Not collected by pytest (the name is not test_*.py).
"""
from typing import Dict, List, Tuple

Substitutions = Dict[str, List[Tuple[str, str]]]


# the golden bundle's pending lines, by bundle path (empty: nothing pending since the M4.5b re-bless)
GOLDEN: Substitutions = {}
# the policy reference trees (test_policy_reference_trees.py: the golden project as 'cell')
POLICY_REFERENCE: Substitutions = {}


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
