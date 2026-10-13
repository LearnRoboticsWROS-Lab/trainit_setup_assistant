"""The golden bundle, compared as a WHOLE TREE, byte for byte (vNext M0D T1).

``test_generator_golden.py`` compares the three packages semantically (URDF, YAML, the tree);
``test_backend_bytestability.py`` pins chosen files. Neither says that EVERY byte TSA writes is
the byte the published golden carries. This file does, with the one difference there is,
named:

    the golden bundle (fr3wml_digital_twin/fr3wml_suction_camera_tsa_4_1_bundle), 35 files:
      34 generated  -- regenerated here: 33 byte-identical, and generation_manifest.yaml whose
                       CONTENT record is the golden's: everything the bundle IS (the project
                       name, the warnings, every file's path, action, sha256 and source, in
                       order) -- but not its PROVENANCE, what compiled it (``generator_version``
                       and ``capability_snapshot``: the golden records TSA 7.8.0's, updated
                       in vNext G0), which changes with every release and every catalog
                       refresh and is checked against the committed catalog by
                       test_bundle_provenance.py
      1 input       -- project.yaml, the project the bundle was generated FROM (provenance)

Measured 2026-10-06: exactly that, and two regenerations give the same tree hash. A COPY's
source is compared as the manifest records it since M2.3 (``package://<package>/<path>``); the
golden records that form too since its re-bless, and an absolute source (a golden of the older
form) would be rebased into this workspace and named with the same rule -- the gate says "the
same file was copied", on any machine.

The golden itself is frozen: its canonical tree hash is pinned below. It was frozen in vNext M0A
and re-blessed three times, each time owner-approved and with its exact diff shown first:

  vNext M2.3 (2026-10-08)  only generation_manifest.yaml: generator_version 0.1.0 (TSA v4.3.1)
                           -> 7.5.0, the capability_snapshot block, its fifteen COPY sources
                           made portable (tree 97b0c2fa...);
  vNext M2.4 (2026-10-08)  the two package.xml files that depend on trainit_motion_runtime, each
                           by its version floor and the comment naming what set it (app >= 1.7.0,
                           trainit_config >= 1.4.0), and the manifest's provenance and those two
                           sha256 (tree 5a5735d2...);
  vNext G0 (2026-10-13)    provenance only, for publication: three lines of
                           generation_manifest.yaml -- generator_version 7.6.0 -> 7.8.0, the
                           catalog_sha256 of TSA 7.8.0's snapshot, the recorded
                           trainit_motion_runtime version 1.8.0 -> 2.1.0 (tree adfdb238...).

The other 32 files are the M0A bytes. Re-blessing the golden is an explicit, explained change of
that constant -- never a side effect of a generator change. Between a generator change and its
re-bless, ``pending_rebless.GOLDEN`` lists the exact lines the golden may differ by (owner answer
Q1, vNext M2.4); it is empty since the M2.4 re-bless.

Fast and ROS-free: generation is pure.
"""
import hashlib
import tempfile
from pathlib import Path

import pytest
import yaml

import pending_rebless
import trainit_setup_assistant
import ws_paths
from test_generator_golden import _generate_with_golden_names
from trainit_setup_assistant.generator.provenance import portable_source

GOLDEN = ws_paths.GOLDEN_BUNDLE
MANIFEST = 'generation_manifest.yaml'
INPUT = 'project.yaml'
# What compiled a bundle, not what it is: never compared with the golden (see the docstring).
PROVENANCE = ('generator_version', 'capability_snapshot')
# sha256 over "<sha256>  <posix relative path>\n", paths sorted as Python strings (locale-free)
GOLDEN_TREE_SHA256 = 'adfdb238ca9486d9cdaa40d0af77950afda2181f8d2ed8448b62374b8d573469'

pytestmark = pytest.mark.skipif(not GOLDEN.is_dir(), reason='golden bundle not present')


def _files(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in sorted(root.rglob('*')) if p.is_file()}


def _tree_sha256(files: dict) -> str:
    listing = ''.join(f'{hashlib.sha256(b).hexdigest()}  {rel}\n'
                      for rel, b in sorted(files.items()))
    return hashlib.sha256(listing.encode()).hexdigest()


def _regenerated() -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        _generate_with_golden_names(tmp)
        return _files(Path(tmp))


def test_the_committed_golden_is_the_frozen_one():
    assert _tree_sha256(_files(GOLDEN)) == GOLDEN_TREE_SHA256


def _expected(gold: dict) -> dict:
    """The golden's files as this TSA must generate them: the pending M2.4 lines applied."""
    return {rel: pending_rebless.forward(data, pending_rebless.GOLDEN, rel)
            for rel, data in gold.items()}


def test_the_pending_list_names_only_golden_package_xml_files():
    gold = _files(GOLDEN)
    assert set(pending_rebless.GOLDEN) <= set(gold)
    assert all(rel.endswith('/package.xml') for rel in pending_rebless.GOLDEN)


def test_every_generated_file_is_the_goldens_byte_for_byte():
    gen, gold = _regenerated(), _expected(_files(GOLDEN))
    assert set(gen) == set(gold) - {INPUT}                 # nothing missing, nothing extra
    differing = sorted(rel for rel in gen if rel != MANIFEST and gen[rel] != gold[rel])
    assert not differing, differing


def _content_record(manifest: dict) -> dict:
    """The manifest without its provenance; a COPY's source as M2.3 records it (portable)."""
    record = {k: v for k, v in manifest.items() if k not in PROVENANCE}
    record['files'] = [dict(f, source=_portable(f['source'])) if f.get('action') == 'COPY' else f
                       for f in record['files']]
    return record


def _portable(source: str) -> str:
    if source.startswith(('package://', 'external:')):
        return source
    return portable_source(ws_paths.in_workspace(source))   # the golden's recorded machine path


def _pending_record(gold_manifest: dict, gold_files: dict) -> dict:
    """The golden's manifest with the sha256 of each pending file as this TSA must write it."""
    expected = _expected(gold_files)
    files = [dict(f, sha256=hashlib.sha256(expected[f['path']]).hexdigest())
             if f['path'] in pending_rebless.GOLDEN and f.get('action') != 'COPY' else f
             for f in gold_manifest['files']]
    return dict(gold_manifest, files=files)


def test_the_manifests_content_record_is_the_goldens():
    gen = yaml.safe_load(_regenerated()[MANIFEST])
    gold_files = _files(GOLDEN)
    gold = _pending_record(yaml.safe_load(gold_files[MANIFEST]), gold_files)
    assert gen['generator_version'] == trainit_setup_assistant.__version__
    assert gold['generator_version'] == '7.8.0'   # the release of the G0 provenance update
    # both say what compiled them, in the same shape; the VALUES are provenance and move with
    # every release and refresh (test_bundle_provenance.py checks the generated one)
    assert list(gen)[:2] == list(gold)[:2] == ['generator_version', 'capability_snapshot']
    assert {k: sorted(v) if isinstance(v, dict) else None
            for k, v in gen['capability_snapshot'].items()} == \
        {k: sorted(v) if isinstance(v, dict) else None
         for k, v in gold['capability_snapshot'].items()}
    # every file's path, action, sha256 and source, as an ORDERED list: one path is recorded
    # twice (the srdf COPY, then its GENERATE), and the order is the record's
    assert _content_record(gen) == _content_record(gold)


def test_regeneration_is_deterministic():
    assert _tree_sha256(_regenerated()) == _tree_sha256(_regenerated())
