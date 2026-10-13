"""The cell namespace: one datum, read in one place (ADR-0016 Decision 3).

Every namespaced artefact in a generated bundle derives from ``RobotSpec.cell_namespace``
and from nothing else — the bring-up's ``NS`` constant, the servo topics, the gripper
action in ``bt_params.yaml`` and the BT application's own node namespace. That is why the
derivation lives in a module of its own instead of being spelt out per emitter: a second
spelling is a second place for the two halves of a cell to disagree.

**Empty is the default and the migration path.** It puts every node at the ROOT, which is
byte-for-byte what every bundle generated before this ADR, and the golden-bundle and
byte-stability gates pin exactly that.

What a non-empty namespace reaches, and what it deliberately does not:

================================  ==========================================================
namespaced                        ``move_group``, the controllers (``ros2_control_node`` +
                                  the spawners' ``--controller-manager``),
                                  ``robot_state_publisher``, ``scene_manager_node``, the
                                  ``servo_node`` and the topics servo derives, the robot's
                                  own cell bridges, and ``trainit_run_bt`` (a client of
                                  this robot's ``move_group``)
left GLOBAL                       ``/tf`` and ``/tf_static`` (tf2 declares them absolute,
                                  and a split TF tree is a cell that cannot see itself),
                                  the cameras (a camera serves a STATION, not a robot),
                                  ``/perception/<detector>/…`` (one detector serves every
                                  arm reaching into the bin), and RViz (MoveIt's
                                  MotionPlanning plugin builds no interactive marker from
                                  inside a namespace — measured)
================================  ==========================================================
"""

from __future__ import annotations

import re
from typing import List

from ..model import CanonicalProject
from ..model.enums import Backend


def cell_namespace(project: CanonicalProject) -> str:
    """The canonical cell namespace: ``''`` (the root — one robot) or ``/seg[/seg…]``."""
    return project.robot.namespace()


def topic_under(namespace: str, name: str) -> str:
    """An ABSOLUTE topic under a namespace (``/joint_states`` when it is empty)."""
    return f'{namespace.rstrip("/")}/{name.lstrip("/")}'


def _root_absolute(value) -> bool:
    """True for a string that is an ABSOLUTE ROS name at the root (``/joint_states``).

    Such a name does NOT follow a namespace: it is the one shape that keeps pointing at
    the root when the node that uses it moves.
    """
    return isinstance(value, str) and value.startswith('/')


# What a ROS 2 name may contain (alphanumerics, '_', '/', '~' and '{...}' substitutions). A
# parameter value outside it -- '/srv/cell/calibration.yaml' -- or under a filesystem root --
# '/home/me/models/run_42', '/dev/ttyUSB0' -- is a path, not a topic: no namespace could move it,
# and quoting it would put the path in the manifest (M2.3).
_ROS_NAME = re.compile(r'/[A-Za-z0-9_/~{}]*')
_FILESYSTEM_ROOTS = {'bin', 'boot', 'dev', 'etc', 'home', 'lib', 'media', 'mnt', 'opt', 'proc',
                     'root', 'run', 'sbin', 'snap', 'srv', 'sys', 'tmp', 'usr', 'var'}


def _ros_topic(value: str) -> bool:
    return bool(_ROS_NAME.fullmatch(value)) and value.split('/')[1] not in _FILESYSTEM_ROOTS


def param_file_node_key(project: CanonicalProject, node_name: str) -> str:
    """The key a ROS 2 parameter FILE uses for ``node_name``, so it binds in the robot's slot.

    A bare key binds the node only at the ROOT: under ``/r3`` it binds NOTHING, and the node
    comes up with its defaults and reports ready (verified by execution on Humble, 2026-10-06;
    the first customer cell's scene manager loaded 0 objects exactly so). ``/**/<name>``
    binds it in any namespace. Empty namespace => the bare key, byte for byte as before
    (vNext M0C #5).
    """
    return f'/**/{node_name}' if cell_namespace(project) else node_name


def bare_param_file_keys(path) -> List[str]:
    """The top-level keys of a parameter file that would bind nothing under a namespace."""
    import yaml
    try:
        doc = yaml.safe_load(open(path)) or {}
    except (OSError, yaml.YAMLError):
        return []
    return [str(k) for k in doc if isinstance(doc, dict) and not str(k).startswith('/')]


def namespace_handoff_warnings(project: CanonicalProject) -> List[str]:
    """What a non-empty namespace does NOT fix by itself, named precisely.

    The generator namespaces the nodes it starts. It does **not** rewrite cell DATA: the
    bridge parameters and the arm ``/joint_states`` remap were captured verbatim from the
    hand-made base bring-up (Step 1), and a value like ``/joint_states_robot`` keeps
    pointing at the root however the node that reads it is launched. That asymmetry is
    reported here rather than papered over with a silent rewrite, because a silent rewrite
    of a captured name is exactly how two halves of a cell start disagreeing.

    Empty namespace => empty list, so nothing changes for a single-robot project.
    """
    ns = cell_namespace(project)
    if not ns:
        return []
    out: List[str] = []
    dep = project.deployment

    if _root_absolute(dep.arm_joint_states_remap_to):
        out.append(
            f'cell_namespace "{ns}": deployment.arm_joint_states_remap_to is '
            f'"{dep.arm_joint_states_remap_to}", an absolute ROOT name, so the arm '
            f'broadcaster publishes OUTSIDE the namespace while move_group reads '
            f'{ns}/joint_states — planning would have no start state. Give it the '
            f'namespace ("{ns}{dep.arm_joint_states_remap_to}"), or drop it and let the '
            f'broadcaster publish {ns}/joint_states directly.')

    for b in dep.bridges:
        names = [v for d in b.parameters for v in d.values()
                 if _root_absolute(v) and _ros_topic(v)]
        names += [r[1] for r in b.remappings
                  if len(r) == 2 and _root_absolute(r[1])]
        offenders = [n for n in names if not (n == ns or n.startswith(f'{ns}/'))]
        if offenders and b.package not in _CAMERA_BRIDGE_PACKAGES:
            out.append(
                f'cell_namespace "{ns}": bridge {b.package}/{b.executable} runs IN the '
                f'namespace but names {len(offenders)} absolute ROOT topic(s)/service(s) '
                f'({", ".join(sorted(set(offenders))[:4])}) — captured verbatim from the '
                f'base bring-up, and an absolute name never follows a namespace. Prefix '
                f'them with "{ns}" in the project, or make them relative.')

    if Backend.GAZEBO in dep.modes:
        out.append(
            f'cell_namespace "{ns}": the GAZEBO backend is NOT namespaced by this field. '
            'gazebo_ros2_control owns /controller_manager inside gzserver and the name '
            'comes from the URDF <gazebo> plugin tag, not from the launch — so a '
            'namespaced cell is unverified on gazebo. Use mock/isaac/real, or declare the '
            'namespace in the plugin tag yourself.')

    if project.robot.base_moveit_config_path:
        out.append(
            f'cell_namespace "{ns}": the cell\'s moveit.rviz is COPIED from your base '
            'config, so it is not rewritten. In the MotionPlanning display set "Move '
            f'Group Namespace" to {ns} and point "Planning Scene Topic" / "Trajectory '
            f'Topic" at {ns}/monitored_planning_scene and {ns}/display_planned_path; the '
            'generated bring-up already remaps the two robot_description topics for RViz.')

    return out


# Packages whose nodes serve the CAMERA, not the robot, so they stay at the root. Kept
# here beside the warning that consults it, and emitted into the bring-up as
# CAMERA_BRIDGE_PACKAGES so the generated launch states the same rule in the open.
_CAMERA_BRIDGE_PACKAGES = ('depth_image_proc',)
