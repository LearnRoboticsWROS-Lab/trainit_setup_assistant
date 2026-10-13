"""Generation manifest: a per-file record of what the generator did.

Written as ``generation_manifest.yaml`` at the bundle root. Enables reproducibility
checks ("did anything change?"), the verification harness's structural comparison,
and surfacing of silent caps (e.g. a non-box primitive emitted as its AABB).

Two parts (vNext M2.3, ADR-0021). PROVENANCE -- what compiled the bundle: ``generator_version``
(the TSA release) and ``capability_snapshot`` (the identity of the committed catalog snapshot it
compiled with: versions and sha256 hashes, ``catalog.snapshot_identity()``; the same for every
bundle of a TSA build, never a dependency list). CONTENT -- what the bundle is: the project name,
the warnings and one record per written file (path, action, sha256, source). A COPY's source is
``package://<package>/<path>`` or ``external:<file name>`` (``provenance.portable_source``), never
a path of the machine that generated it; a warning that names a file or directory names it the
same way. The user who generates sees the same warnings with the paths of THEIR machine
(``local_warnings``, printed by the CLI and shown by the GUI, never written to the bundle).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .determinism import canonical_yaml

# action vocabulary
COPY = 'COPY'          # byte-for-byte from a source file/tree
TEMPLATE = 'TEMPLATE'  # rendered from a Jinja2 template
GENERATE = 'GENERATE'  # synthesized from the task graph (bt_params, BT xml, srdf)


@dataclass
class FileEntry:
    path: str            # relative to bundle root, POSIX separators
    action: str          # COPY | TEMPLATE | GENERATE
    sha256: str
    source: Optional[str] = None  # template name or source path

    def as_dict(self) -> dict:
        d = {'path': self.path, 'action': self.action, 'sha256': self.sha256}
        if self.source is not None:
            d['source'] = self.source
        return d


@dataclass
class GenerationManifest:
    generator_version: str
    project_name: str
    files: List[FileEntry] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    capability_snapshot: Optional[dict] = None     # catalog.snapshot_identity() (M2.3)
    local_warnings: List[str] = field(default_factory=list)   # what THIS machine's user reads

    def record(self, path: str, action: str, sha256: str, source: Optional[str] = None) -> None:
        self.files.append(FileEntry(Path(path).as_posix(), action, sha256, source))

    def warn(self, message: str, local: Optional[str] = None) -> None:
        """``message`` goes into the manifest; ``local`` (default: the same text) is what the user
        who generates is shown -- it may name this machine's paths, the manifest never does."""
        self.warnings.append(message)
        self.local_warnings.append(message if local is None else local)

    def as_dict(self) -> dict:
        ordered = sorted(self.files, key=lambda e: e.path)
        provenance = {'generator_version': self.generator_version}
        if self.capability_snapshot is not None:
            provenance['capability_snapshot'] = copy.deepcopy(self.capability_snapshot)
        return {
            **provenance,
            'project_name': self.project_name,
            'file_count': len(ordered),
            'warnings': list(self.warnings),
            'files': [e.as_dict() for e in ordered],
        }

    def write(self, bundle_root: Path) -> Path:
        out = Path(bundle_root) / 'generation_manifest.yaml'
        out.write_text(canonical_yaml(self.as_dict()))
        return out
