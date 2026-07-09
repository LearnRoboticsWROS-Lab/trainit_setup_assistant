"""Turn a loaded robot xacro into a draft model (the headless core of GUI step S0/S1).

`load_robot_spec` compiles the xacro, auto-detects the planning group/EEF, and extracts
joint limits from the URDF, producing a `RobotSpec` the user then confirms/edits.
`bootstrap_project` wraps it into a minimal buildable `CanonicalProject` so a live RViz
session can be brought up before any waypoints exist.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from ..model import (
    BundleSpec,
    CanonicalProject,
    DescriptionSource,
    GripperSpec,
    JointLimit,
    PlanningGroupSpec,
    RobotSpec,
)
from ..model.enums import GripperKind
from .group_detector import RobotDetection, detect
from .kinematic_chain import parse_urdf
from .xacro_loader import compile_xacro

DEFAULT_MAX_ACCEL = 5.0


def _extract_joint_limits(robot, joint_names) -> dict:
    """Per-joint limits from the URDF (velocity from URDF; accel default)."""
    out = {}
    by_name = {j.name: j for j in robot.joints}
    for name in joint_names:
        j = by_name.get(name)
        vel = getattr(getattr(j, 'limit', None), 'velocity', None) if j else None
        if vel and vel > 0:
            out[name] = JointLimit(has_velocity_limits=True, max_velocity=float(vel),
                                   has_acceleration_limits=True,
                                   max_acceleration=DEFAULT_MAX_ACCEL)
        else:
            out[name] = JointLimit(has_velocity_limits=False, max_velocity=0.0,
                                   has_acceleration_limits=False, max_acceleration=0.0)
    return out


def _draft_gripper(det: RobotDetection) -> GripperSpec:
    if det.gripper_kind is GripperKind.NONE or not det.gripper_joints:
        return GripperSpec(kind=GripperKind.NONE)
    cmd = det.gripper_joints[0]
    group = 'SuctionCup' if det.gripper_kind is GripperKind.SUCTION else 'Gripper'
    return GripperSpec(
        kind=det.gripper_kind,
        eef_group_name=group,
        eef_name=group.lower(),
        eef_parent_link=det.eef_parent_link or det.tip_link,
        command_joint=cmd,
        controller_name=f'{cmd.replace("_joint", "")}_controller',
        controller_type='GripperCommand',
        action_ns='gripper_command',
        grasp_action='close',
        release_action='open',
    )


def load_robot_spec(
    xacro_path,
    robot_name: Optional[str] = None,
    group_name: Optional[str] = None,
    meshes_dir: Optional[str] = None,
) -> RobotSpec:
    """Compile + auto-detect a robot xacro into a draft RobotSpec."""
    xacro_path = Path(xacro_path)
    urdf = compile_xacro(xacro_path)
    robot = parse_urdf(urdf)
    det = detect(robot)

    rname = robot_name or robot.name
    gname = group_name or rname
    all_joints = det.arm_joints + det.gripper_joints

    spec = RobotSpec(
        robot_name=rname,
        description=DescriptionSource(
            urdf_dir=str(xacro_path.parent),
            top_xacro=xacro_path.name,
            meshes_dir=meshes_dir,
        ),
        base_frame=det.base_link,
        tip_link=det.tip_link,
        planning_group=PlanningGroupSpec(
            name=gname, base_link=det.base_link, tip_link=det.tip_link,
            joints=det.arm_joints),
        gripper=_draft_gripper(det),
        initial_positions={j: 0.0 for j in all_joints},
    )
    spec.joint_limits.per_joint = _extract_joint_limits(robot, all_joints)
    if len(det.arm_joints) >= 2:
        spec.ompl.projection_evaluator = (
            f'joints({det.arm_joints[0]},{det.arm_joints[1]})')
    return spec


def bootstrap_project(
    robot_spec: RobotSpec, project_name: Optional[str] = None
) -> CanonicalProject:
    """Wrap a RobotSpec into a minimal buildable project (robot only, empty app).

    Enough to generate + build a bundle and bring up a live RViz session for pose
    capture, before any waypoints/scene are configured.
    """
    name = project_name or f'{robot_spec.robot_name}_app'
    return CanonicalProject(
        project_name=name,
        bundle=BundleSpec.from_prefix(name),
        robot=robot_spec,
    )
