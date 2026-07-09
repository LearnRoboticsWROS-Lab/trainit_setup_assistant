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

        # COPY urdf/ (xacro) + meshes/ verbatim
        n_urdf = ctx.copy_tree(urdf_dir, f'{pkg}/urdf')
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
