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
    """Names of the three generated ROS 2 packages of a bundle.

    The prefix is the ROBOT name, so a bundle is ``<robot>_description`` +
    ``<robot>_trainit_config`` + ``<robot>_app``. ``moveit_config_package`` holds the
    self-contained *trainit_config* — the "engine bay": the MoveIt config + Pilz/OMPL/
    CHOMP planners + mock/isaac/real wiring + gripper adapters + the scene loader, on
    which ``<robot>_app`` (BehaviorTree, consuming the TMR) sits. Fields stay
    independently settable so a golden bundle can still be matched exactly.
    """

    description_package: str
    moveit_config_package: str   # value is the bundle's self-contained <robot>_trainit_config
    app_package: str

    @classmethod
    def from_prefix(cls, prefix: str) -> 'BundleSpec':
        return cls(
            description_package=f'{prefix}_description',
            moveit_config_package=f'{prefix}_trainit_config',
            app_package=f'{prefix}_app',
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
