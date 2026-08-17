"""Emit ``*_description``: COPY the robot geometry, TEMPLATE the package files.

The assistant ingests the user's existing description (urdf/xacro + meshes) verbatim
— it never synthesizes geometry. Only package.xml + CMakeLists.txt are templated.
"""

from __future__ import annotations

from pathlib import Path

from .base import Emitter, GenContext


class DescriptionEmitter(Emitter):
    def emit(self, project, ctx: GenContext) -> None:
        pkg = project.bundle.description_package
        desc = project.robot.description

        urdf_dir = Path(desc.urdf_dir)
        if not urdf_dir.is_dir():
            raise FileNotFoundError(f'description urdf_dir not found: {urdf_dir}')

        # COPY the robot MODEL files (*.urdf, *.xacro) — never a verbatim tree:
        # urdf_dir may be a mixed dir (the base moveit_config's config/, or even a
        # workspace root) and build/install/git trees must never be ingested.
        n_urdf = self._copy_model_files(ctx, urdf_dir, f'{pkg}/urdf')
        has_meshes = False
        if desc.meshes_dir:
            meshes_dir = Path(desc.meshes_dir)
            if meshes_dir.is_dir():
                ctx.copy_tree(meshes_dir, f'{pkg}/meshes')
                has_meshes = True
            else:
                ctx.manifest.warn(f'meshes_dir not found, skipped: {meshes_dir}')

        if n_urdf == 0:
            ctx.manifest.warn(f'no urdf files copied from {urdf_dir}')

        install_dirs = 'urdf meshes' if has_meshes else 'urdf'

        # TEMPLATE view_robot.launch.py + a minimal RViz config — "get familiar with
        # the robot ALONE" (rsp + joint sliders + RViz; no MoveIt, no controllers).
        xacro_file = (desc.top_xacro
                      if desc.top_xacro and (urdf_dir / desc.top_xacro).is_file()
                      else self._main_xacro(urdf_dir))
        if xacro_file:
            ctx.render_to(
                f'{pkg}/launch/view_robot.launch.py',
                'description/view_robot.launch.py.j2',
                package_name=pkg,
                robot_name=project.robot.robot_name,
                xacro_file=xacro_file,
            )
            ctx.render_to(
                f'{pkg}/rviz/view_robot.rviz',
                'description/view_robot.rviz.j2',
                base_frame=project.robot.base_frame,
            )
            install_dirs += ' launch rviz'
        else:
            ctx.manifest.warn(f'no .urdf.xacro/.xacro/.urdf in {urdf_dir}: '
                              f'view_robot.launch.py not generated')

        # TEMPLATE package.xml + CMakeLists.txt
        ctx.render_to(
            f'{pkg}/package.xml',
            'description/package.xml.j2',
            package_name=pkg,
            meta=project.meta,
            robot_name=project.robot.robot_name,
        )
        ctx.render_to(
            f'{pkg}/CMakeLists.txt',
            'description/CMakeLists.txt.j2',
            package_name=pkg,
            install_dirs=install_dirs,
        )

    @staticmethod
    def _main_xacro(urdf_dir: Path) -> str:
        """The robot's top-level model file (relative to urdf/), preferring xacro."""
        for pattern in ('*.urdf.xacro', '*.xacro', '*.urdf'):
            found = sorted(urdf_dir.glob(pattern))
            if found:
                return found[0].name
        return ''

    _SKIP_DIRS = {'build', 'install', 'log', '.git', '.claude', '.vscode',
                  '__pycache__', 'node_modules'}

    @classmethod
    def _copy_model_files(cls, ctx: GenContext, src_dir: Path, rel_dest: str) -> int:
        """COPY only *.urdf / *.xacro (recursive, skipping build trees) — plus the
        ``initial_positions*.yaml`` data files a MoveIt-SA ros2_control xacro loads
        with a RELATIVE ``load_yaml()`` (without them the copied xacro won't compile)."""
        count = 0
        for src in sorted(src_dir.rglob('*')):
            if not src.is_file():
                continue
            model = src.suffix in ('.urdf', '.xacro')
            xacro_data = (src.suffix in ('.yaml', '.yml')
                          and src.stem.startswith('initial_positions'))
            if not (model or xacro_data):
                continue
            rel = src.relative_to(src_dir)
            if any(part in cls._SKIP_DIRS for part in rel.parts[:-1]):
                continue
            ctx.copy_file(src, str(Path(rel_dest) / rel))
            count += 1
        return count
