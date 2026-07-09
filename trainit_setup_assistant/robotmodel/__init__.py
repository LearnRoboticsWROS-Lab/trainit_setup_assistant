"""In-memory robot model: xacro loading, kinematic chain, SRDF, collision matrix.

These modules let the assistant reason about a robot the way the MoveIt Setup
Assistant does — building an SRDF and a self-collision matrix headlessly — without a
running move_group. The live RViz session (``livesession/``) consumes their output.
"""

from .collision_matrix import compute_disable_collisions, parse_disable_collisions
from .group_detector import RobotDetection, detect
from .robot_loader import bootstrap_project, load_robot_spec
from .kinematic_chain import (
    all_movable_joints,
    chain_movable_joints,
    link_chain,
    parse_urdf,
    root_link,
)
from .srdf_builder import DisablePair, build_srdf
from .xacro_loader import compile_xacro

__all__ = [
    'build_srdf',
    'DisablePair',
    'compile_xacro',
    'compute_disable_collisions',
    'parse_disable_collisions',
    'parse_urdf',
    'root_link',
    'link_chain',
    'chain_movable_joints',
    'all_movable_joints',
    'detect',
    'RobotDetection',
    'load_robot_spec',
    'bootstrap_project',
]
