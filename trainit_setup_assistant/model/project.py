"""The root canonical project — the single source of truth.

The GUI edits a ``CanonicalProject``; the generator only reads it. Persisted as
``project.yaml`` (see :mod:`trainit_setup_assistant.model.io`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .deployment import DeploymentSpec
from .robot import RobotSpec
from .scene import SceneSpec
from .task import ApplicationSpec

SCHEMA_VERSION = 1


class BundleSpec(BaseModel):
    """Names of the three generated ROS 2 packages.

    Independently settable so the output can either match a golden bundle exactly or
    use a fresh prefix (e.g. ``fr3wml_app_trainit_config``).
    """

    description_package: str
    moveit_config_package: str
    app_package: str

    @classmethod
    def from_prefix(cls, prefix: str) -> 'BundleSpec':
        return cls(
            description_package=f'{prefix}_description',
            moveit_config_package=f'{prefix}_moveit_config',
            app_package=prefix,
        )


class ProjectMeta(BaseModel):
    """Package metadata templated into every package.xml."""

    maintainer_name: str = 'fra'
    maintainer_email: str = 'ros.master.ai@gmail.com'
    license: str = 'Apache-2.0'
    version: str = '0.0.0'


class CanonicalProject(BaseModel):
    schema_version: int = SCHEMA_VERSION
    project_name: str
    bundle: BundleSpec
    meta: ProjectMeta = Field(default_factory=ProjectMeta)
    robot: RobotSpec
    scene: SceneSpec = Field(default_factory=SceneSpec)
    application: ApplicationSpec = Field(default_factory=ApplicationSpec)
    deployment: DeploymentSpec = Field(default_factory=DeploymentSpec)
