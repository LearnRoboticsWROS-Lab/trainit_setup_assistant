"""Generation manifest: a per-file record of what the generator did.

Written as ``generation_manifest.yaml`` at the bundle root. Enables reproducibility
checks ("did anything change?"), the verification harness's structural comparison,
and surfacing of silent caps (e.g. a non-box primitive emitted as its AABB).
"""

from __future__ import annotations

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

    def record(self, path: str, action: str, sha256: str, source: Optional[str] = None) -> None:
        self.files.append(FileEntry(Path(path).as_posix(), action, sha256, source))

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def as_dict(self) -> dict:
        ordered = sorted(self.files, key=lambda e: e.path)
        return {
            'generator_version': self.generator_version,
            'project_name': self.project_name,
            'file_count': len(ordered),
            'warnings': list(self.warnings),
            'files': [e.as_dict() for e in ordered],
        }

    def write(self, bundle_root: Path) -> Path:
        out = Path(bundle_root) / 'generation_manifest.yaml'
        out.write_text(canonical_yaml(self.as_dict()))
        return out
