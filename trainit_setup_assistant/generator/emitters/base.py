"""Emitter base + the generation context shared by all emitters.

An :class:`Emitter` turns part of a :class:`CanonicalProject` into files under one of
the three packages. All file writes go through :class:`GenContext` so every byte is
recorded in the manifest with its action (COPY/TEMPLATE/GENERATE).
"""

from __future__ import annotations

import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, Union

from jinja2 import Environment

from ..determinism import sha256_bytes
from ..manifest import COPY, GENERATE, TEMPLATE, GenerationManifest
from ..provenance import SourceNamer


class GenContext:
    """Carries the output root, the Jinja env, and the manifest through emission."""

    def __init__(self, output_root: Path, env: Environment, manifest: GenerationManifest):
        self.output_root = Path(output_root)
        self.env = env
        self.manifest = manifest
        self.sources = SourceNamer()      # a COPY's source, as the manifest records it

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
    def portable(self, path: Union[str, Path], directory: bool = False) -> str:
        """A file (or directory) as the manifest names it -- package://<package>/<path> or
        external:<name>, never this machine's path (vNext M2.3); an unusable package.xml on the
        way is reported once, as a manifest warning."""
        name = self.sources(path, directory=directory)
        for problem in self.sources.take_problems():
            self.manifest.warn(problem)
        return name

    def warn_path(self, render: Callable[[str], str], path: Union[str, Path],
                  directory: bool = False) -> None:
        """A warning about a file or directory: the manifest names it portably, the user who
        generates sees the path as given."""
        raw = os.fspath(path) if path is not None else ''
        # '' or '.' (Path('') is '.') would name the cwd -- a login name, another package
        shown = raw if raw in ('', '.') else self.portable(path, directory=directory)
        self.manifest.warn(render(shown), local=render(raw))

    def copy_file(self, src: Union[str, Path], rel_dest: str, make_executable: bool = False) -> Path:
        """Copy a file byte-identically. The copy is written, not cloned, so it never keeps the
        source's mode: pass make_executable=True for anything installed with install(PROGRAMS)."""
        data = Path(src).read_bytes()
        return self.write_bytes(rel_dest, data, COPY, source=self.portable(src),
                                make_executable=make_executable)

    def written(self, prefix: str, suffix: str = '') -> Dict[str, str]:
        """The text of every file THIS generation rendered or generated (never a COPY) whose
        bundle path starts with ``prefix`` and ends with ``suffix``, by path: what a later step
        may read back from the files it wrote (vNext M2.4, the runtime requirement)."""
        paths = sorted({e.path for e in self.manifest.files if e.action in (TEMPLATE, GENERATE)
                        and e.path.startswith(prefix) and e.path.endswith(suffix)})
        return {p: (self.output_root / p).read_text(encoding='utf-8') for p in paths}

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
