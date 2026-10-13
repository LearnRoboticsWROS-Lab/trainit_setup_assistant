"""The root canonical project — the single source of truth.

The GUI edits a ``CanonicalProject``; the generator only reads it. Persisted as
``project.yaml`` (see :mod:`trainit_setup_assistant.model.io`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, root_validator

from typing import Optional

from .deployment import DeploymentSpec
from .perception import PerceptionSpec
from .robot import RobotSpec
from .scene import SceneSpec
from .task import ApplicationSpec

# 2: adds the top-level ``perception`` block + ``Waypoint.vision`` (TSA v4, D-015).
# Old projects load unchanged (the new fields are optional with None defaults).
SCHEMA_VERSION = 2
# 3: written IF AND ONLY IF the project carries an end-effector contract (ADR-0022 D6, the second
# silent-downgrade guard). A TSA copy older than ADR-0010 keeps schema_version as a plain integer
# and re-saves it even when it drops the contract it cannot read, so a 3 without a contract tells
# this TSA the file lost one (R19). A project without a contract keeps the number it has.
SCHEMA_VERSION_EEF_CONTRACT = 3


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
    # Perception is a TOP-LEVEL resource (the dedicated step): the camera + named
    # detectors, configured once, bound by any application via Waypoint.vision.
    perception: Optional[PerceptionSpec] = None
    application: ApplicationSpec = Field(default_factory=ApplicationSpec)
    deployment: DeploymentSpec = Field(default_factory=DeploymentSpec)

    # skip_on_failure: required by pydantic 2.x for a post root_validator, accepted by 1.x.
    @root_validator(skip_on_failure=True)
    def _schema_version_follows_the_contract(cls, values):
        """``schema_version`` 3 if and only if an end-effector contract exists (ADR-0022 D6):
        a contract loads as 3 whatever number its file had; a 3 without one is refused (R19)."""
        robot = values.get('robot')
        has_contract = bool(robot is not None and robot.gripper.contract is not None)
        if has_contract:
            values['schema_version'] = SCHEMA_VERSION_EEF_CONTRACT
        elif values.get('schema_version') == SCHEMA_VERSION_EEF_CONTRACT:
            raise ValueError('end-effector contract (ADR-0022 R19): schema_version 3 without a '
                             'contract block -- the file was saved with an end-effector contract '
                             'that is no longer in it (an older TSA dropped it on save?). Restore '
                             'robot.gripper.contract, or set schema_version 2 for a legacy '
                             'project.')
        return values
