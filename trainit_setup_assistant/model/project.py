"""The root canonical project — the single source of truth.

The GUI edits a ``CanonicalProject``; the generator only reads it. Persisted as
``project.yaml`` (see :mod:`trainit_setup_assistant.model.io`).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, root_validator

from typing import Optional

from .data_collection import DataCollectionSpec
from .deployment import DeploymentSpec
from .perception import PerceptionSpec
from .robot import RobotSpec
from .scene import SceneSpec
from .enums import AppType
from .task import ApplicationSpec

# 2: adds the top-level ``perception`` block + ``Waypoint.vision`` (TSA v4, D-015).
# Old projects load unchanged (the new fields are optional with None defaults).
SCHEMA_VERSION = 2
# 3: written IF AND ONLY IF the project carries an end-effector contract (ADR-0022 D6, the second
# silent-downgrade guard). A TSA copy older than ADR-0010 keeps schema_version as a plain integer
# and re-saves it even when it drops the contract it cannot read, so a 3 without a contract tells
# this TSA the file lost one (R19). A project without a contract keeps the number it has.
SCHEMA_VERSION_EEF_CONTRACT = 3
# 4: written IF AND ONLY IF the application is a Canonical Task Program (vNext M3.3, ADR-0024,
# R-M3-01): its sections are strict and new, so a TSA that cannot read them refuses the file by its
# number (and by the application type) instead of dropping them. A task program may carry an
# end-effector contract: the number stays 4.
SCHEMA_VERSION_TASK_PROGRAM = 4
# 5: written IF AND ONLY IF the project carries DATA COLLECTION content -- a ``data_collection`` section or a
# camera ``id`` / ``profile`` (TrainIt D0.8). A task program with it is 5 too (a TSA that reads 4 refuses
# the file instead of dropping the section). A 5 without that content is refused: a TSA older than D0.8
# keeps the number it read and drops a section it does not know, so a bare 5 says the file lost one.
SCHEMA_VERSION_DATA_COLLECTION = 5


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
    # TrainIt D0.8: what this cell collects data with (a camera, triggers, a synthetic recipe); TSA
    # authors and validates it, the recorder and trainit_data run it (ADR-0029 Decision 7).
    data_collection: Optional[DataCollectionSpec] = None

    def has_data_collection(self) -> bool:
        """The schema-5 content: a data_collection section, or a camera id / profile."""
        return has_data_collection_content(self.data_collection, self.perception)

    # skip_on_failure: required by pydantic 2.x for a post root_validator, accepted by 1.x.
    @root_validator(skip_on_failure=True)
    def _schema_version_follows_the_contract(cls, values):
        """``schema_version`` 4 if and only if the application is a Canonical Task Program (vNext
        M3.3, R-M3-01: the file must say 4, a 4 without one is refused); otherwise 3 if and only if
        an end-effector contract exists (ADR-0022 D6): a contract loads as 3 whatever number its
        file had; a 3 without one is refused (R19)."""
        has_data = has_data_collection_content(values.get('data_collection'), values.get('perception'))
        if has_data:
            # D0.8: 5 whatever the application. A file says 5, or the number its application had before the
            # section was added by hand (a task program 4, a template application 2 or 3); it loads as 5.
            program = getattr(values.get('application'), 'type', None) is AppType.TASK_PROGRAM
            allowed = ((SCHEMA_VERSION_TASK_PROGRAM,) if program
                       else (SCHEMA_VERSION, SCHEMA_VERSION_EEF_CONTRACT)) + (SCHEMA_VERSION_DATA_COLLECTION,)
            if values.get('schema_version') not in allowed:
                raise ValueError(f"R-D0-01: schema_version {values.get('schema_version')!r}: a project with "
                                 f'data collection is schema_version {SCHEMA_VERSION_DATA_COLLECTION}')
            values['schema_version'] = SCHEMA_VERSION_DATA_COLLECTION
            return values
        if values.get('schema_version') == SCHEMA_VERSION_DATA_COLLECTION:
            raise ValueError('R-D0-01: schema_version 5 without data collection content -- the file was '
                             'saved with a data_collection section (or a camera id / profile) that is no '
                             'longer in it (a TSA older than D0.8 dropped it on save?). Restore it, or set '
                             'the schema_version the rest of the project needs')
        application = values.get('application')
        if application is not None and application.type is AppType.TASK_PROGRAM:
            if values.get('schema_version') != SCHEMA_VERSION_TASK_PROGRAM:
                raise ValueError(
                    'R-M3-01: a Canonical Task Program (application type task_program) is '
                    f"schema_version: {SCHEMA_VERSION_TASK_PROGRAM}, the file says "
                    f"{values.get('schema_version')!r}: set schema_version: "
                    f'{SCHEMA_VERSION_TASK_PROGRAM}')
            return values
        if values.get('schema_version') == SCHEMA_VERSION_TASK_PROGRAM:
            kind = getattr(application, 'type', None)
            raise ValueError(
                f'R-M3-01: schema_version {SCHEMA_VERSION_TASK_PROGRAM} is the Canonical Task '
                f"Program's, but this application is '{getattr(kind, 'value', kind)}': a template "
                'application is schema_version 2 (3 with an end-effector contract)')
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


def has_data_collection_content(data_collection, perception) -> bool:
    """A ``data_collection`` section, or a camera ``id`` / ``profile`` (the schema-5 content, D0.8)."""
    camera = getattr(perception, 'camera', None)
    return bool(data_collection is not None
                or (camera is not None and (camera.id is not None or camera.profile is not None)))
