"""Emitter base + the generation context shared by all emitters.

An :class:`Emitter` turns part of a :class:`CanonicalProject` into files under one of
the three packages. All file writes go through :class:`GenContext` so every byte is
recorded in the manifest with its action (COPY/TEMPLATE/GENERATE).
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Union

from jinja2 import Environment

from ..determinism import sha256_bytes
from ..manifest import COPY, GENERATE, TEMPLATE, GenerationManifest


class GenContext:
    """Carries the output root, the Jinja env, and the manifest through emission."""

    def __init__(self, output_root: Path, env: Environment, manifest: GenerationManifest):
        self.output_root = Path(output_root)
        self.env = env
        self.manifest = manifest

    # --- low-level write (always records in the manifest) ---
    def write_bytes(self, rel_path: str, data: bytes, action: str, source: str = None,
                    make_executable: bool = False) -> Path:
        dest = self.output_root / rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        if make_executable:
            # Anything the generated CMakeLists installs with install(PROGRAMS) MUST be
            # executable at the source. install(PROGRAMS) chmods the installed COPY, but
            # `colcon build --symlink-install` installs a symlink instead, so the source
            # file's mode is what ros2 launch actually sees -- and a non-executable target
            # fails with "executable '<name>' not found on the libexec directory", which
            # reads like a missing file rather than a wrong permission bit.
            dest.chmod(dest.stat().st_mode | 0o111)
        self.manifest.record(rel_path, action, sha256_bytes(data), source)
        return dest

    def write_text(self, rel_path: str, text: str, action: str, source: str = None,
                   make_executable: bool = False) -> Path:
        return self.write_bytes(rel_path, text.encode('utf-8'), action, source,
                                make_executable=make_executable)

    # --- rendering ---
    def render(self, template_name: str, **variables) -> str:
        return self.env.get_template(template_name).render(**variables)

    def render_to(self, rel_path: str, template_name: str, make_executable: bool = False,
                  **variables) -> Path:
        """Render a Jinja2 template and write it as a TEMPLATE file.

        Pass make_executable=True for anything installed with install(PROGRAMS).
        """
        return self.write_text(
            rel_path, self.render(template_name, **variables), TEMPLATE,
            source=template_name, make_executable=make_executable
        )

    def generate_to(self, rel_path: str, text: str, source: str = None) -> Path:
        """Write synthesized content (bt_params, BT xml, srdf) as a GENERATE file."""
        return self.write_text(rel_path, text, GENERATE, source=source)

    # --- copying (byte-identical ingestion of robot geometry) ---
    def copy_file(self, src: Union[str, Path], rel_dest: str) -> Path:
        data = Path(src).read_bytes()
        return self.write_bytes(rel_dest, data, COPY, source=str(src))

    def copy_tree(self, src_dir: Union[str, Path], rel_dest_dir: str) -> int:
        """Copy a directory tree verbatim, recording each file. Returns file count."""
        src_dir = Path(src_dir)
        count = 0
        for src in sorted(src_dir.rglob('*')):
            if src.is_file():
                rel = Path(rel_dest_dir) / src.relative_to(src_dir)
                self.copy_file(src, str(rel))
                count += 1
        return count


class Emitter(ABC):
    """Produces files for one package of the bundle."""

    @abstractmethod
    def emit(self, project, ctx: GenContext) -> None:
        ...
