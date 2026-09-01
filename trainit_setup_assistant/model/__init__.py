"""Canonical project data model (the single source of truth)."""

from .enums import (
    AppType,
    DetectionMethod,
    GripperKind,
    IsaacGraspMethod,
    MotionType,
    PlannerId,
    ReleasePolicy,
    SceneObjectCategory,
    SceneObjectSource,
    ShapeType,
    ToolActionKind,
    WaypointRole,
    WaypointType,
)
from .deployment import DeploymentSpec, LaunchNodeSpec, RealIncludeSpec
from .io import load_project, project_to_dict, save_project
from .perception import (CameraSpec, DetectorSpec, PerceptionSpec,
                         RelativeBinding, VisionBinding)
from .project import BundleSpec, CanonicalProject, ProjectMeta, SCHEMA_VERSION
from .robot import (
    ArmControllerSpec,
    DescriptionSource,
    DisableCollisionSpec,
    GripperSpec,
    JointLimit,
    JointLimitsSpec,
    KinematicsSpec,
    NamedState,
    OmplGroupSpec,
    PilzCartesianLimits,
    PlanningGroupSpec,
    RobotSpec,
)
from .scene import Payload, SceneObject, SceneSpec
from .task import ApplicationSpec, MotionSegment, ToolAction, Waypoint

__all__ = [
    'SCHEMA_VERSION',
    'CanonicalProject',
    'BundleSpec',
    'ProjectMeta',
    'RobotSpec',
    'DescriptionSource',
    'PlanningGroupSpec',
    'NamedState',
    'DisableCollisionSpec',
    'KinematicsSpec',
    'JointLimit',
    'JointLimitsSpec',
    'OmplGroupSpec',
    'PilzCartesianLimits',
    'GripperSpec',
    'ArmControllerSpec',
    'SceneSpec',
    'SceneObject',
    'Payload',
    'ApplicationSpec',
    'Waypoint',
    'MotionSegment',
    'ToolAction',
    'DeploymentSpec',
    'LaunchNodeSpec',
    'RealIncludeSpec',
    # perception (TSA v4, D-015)
    'PerceptionSpec',
    'CameraSpec',
    'DetectorSpec',
    'VisionBinding',
    'RelativeBinding',
    'DetectionMethod',
    # enums
    'WaypointType',
    'MotionType',
    'PlannerId',
    'GripperKind',
    'AppType',
    'SceneObjectSource',
    'SceneObjectCategory',
    'ShapeType',
    'ReleasePolicy',
    'IsaacGraspMethod',
    'WaypointRole',
    'ToolActionKind',
    # io
    'load_project',
    'save_project',
    'project_to_dict',
]
