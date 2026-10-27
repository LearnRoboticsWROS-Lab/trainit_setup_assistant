"""What a bundle declares it needs to RUN (vNext M2.4, ADR-0021 amendment M2.4).

A component is a runtime dependency of a generated package only if that package's own files use
it, and its version floor is the oldest version that certainly has everything they use:

  trainit_motion_runtime  every package that depends on it declares ``version_gte`` -- the
                          highest ``catalog.TMR_FLOORS`` version among the TMR capabilities its
                          files use, the first version TMR's package.xml stated (1.4.0) when none
                          is newer -- with one comment naming what set it (owner rule Q2:
                          behaviours count)
  trainit_perception      the app, iff it launches a TP detector_node (unchanged since M2.2)
  trainit_policy_runtime  the app, iff a tree runs RunPolicy or a launch file starts its node;
                          unversioned (0.1.0 is the only version its package.xml has stated)

and never because the capability snapshot (what COMPILED the bundle, M2.3) knows a component.

1. the floors table, pinned as curated from TMR's history, and sane;
2. the closed world: every BT node, port, bt_params key, scene.yaml key, launch-helper argument
   and TMR executable a bundle can carry is one this file has dated -- a new one fails here and
   needs its floor decided; an undated node stops generation;
3. the requirement read from a package's own files, unit by unit (and a file that does not parse);
4. the application matrix, generated in full: every TrainIt dependency line of every package, and
   the closed world of dependencies -- every TrainIt package a package's own files use is one its
   package.xml declares (vNext M2.7);
5. the Step-3 config, and a scene-loader cell that needs no golden;
6. the negative tests: a snapshot capability, an NVIDIA candidate, an unused policy runtime, TP's
   snapshot release are not dependencies, and the installed components do not move a requirement.

Fast and ROS-free, except the last test (fresh interpreters, a few seconds).
"""
import ast
import contextlib
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

import catalog_identity
import identity_cases
import ws_paths
from trainit_setup_assistant import catalog
from trainit_setup_assistant.catalog import (BT_BUILTIN_NODES, EEF_CONTRACT_CAPABILITIES,
                                             TASK_PROGRAM_CAPABILITIES, TMR_BT_NODES,
                                             TMR_EEF_EVIDENCE_PENDING, TMR_FIRST_VERSION,
                                             TMR_FLOORS)
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator import runtime_requirements as rr
from trainit_setup_assistant.generator.orchestrator import generate_scene_loader_config
from trainit_setup_assistant.model import (Backend, BundleSpec, ExecutionMode, MotionType,
                                           PolicyStep, ToolAction, load_project)
from trainit_setup_assistant.model import enums as E
from trainit_setup_assistant.model.enums import (EndEffectorActuation, GripperJointTarget,
                                                 GripperKind, PolicyMode, ToolActionKind)

TEST = Path(__file__).resolve().parent
PKG = TEST.parent
EXAMPLE = PKG / 'examples' / 'fr3wml_project.yaml'
GOLDEN_PROJECT = ws_paths.GOLDEN_BUNDLE / 'project.yaml'
WORKER = TEST / 'identity_worker.py'
COMPONENTS = ('trainit_motion_runtime', 'trainit_perception', 'trainit_policy_runtime')


def _key(version: str) -> tuple:
    return tuple(int(p) for p in version.split('.'))


# --- 1. the table --------------------------------------------------------------------------------

# As vNext M2.4 curated it from TMR's history (the floor scan in the M2.4 forensics: every probe of
# the TMR surface a bundle uses, at every package.xml version bump). A change is a deliberate edit
# of both, with its TMR commit.
PINNED_FLOORS = {
    'ServoTrack': ('1.5.0', 'b56fbe3'),
    'RunPolicy': ('1.5.0', '45ebe5c'),
    'CheckRobotState': ('1.5.0', '45ebe5c'),
    'launch_backend': ('1.5.0', '37c87d9'),
    'gripper_positions': ('1.5.0', 'ff165f8'),
    'scene_manager_namespace': ('1.5.0', '51670ed'),
    'detect_keep': ('1.6.0', 'c47a724'),
    'detection_index': ('1.6.0', 'c47a724'),
    'detect_first_published': ('1.7.0', '5593e5b'),
    'scene_services_namespace': ('1.7.0', 'fd4fe5a'),
    # vNext M2.6 (ADR-0022): the end-effector contract, TMR 2.0.0, shipped by TMR adf94b0.
    'VerifyGrasp': ('2.0.0', 'adf94b0'),
    'CommandActuator': ('2.0.0', 'adf94b0'),
    'detect_fault': ('2.0.0', 'adf94b0'),
    'eef_block': ('2.0.0', 'adf94b0'),
    # vNext M2.7 (owner Q8.2): MoveWaypoint's fault, TMR 2.1.0 (the port 2d230bc; 2a76a55 the rule
    # the canonical cycle relies on: a colliding start is a FAULT).
    'move_waypoint_fault': ('2.1.0', '2a76a55'),
    # vNext G0 (owner section 5, ADR-0023): a canonical runner in a cell namespace, TMR 2.1.0 (its
    # MoveIt clients built with the node's namespace in c2c9f01, under package.xml 2.0.0)
    'moveit_clients_namespace': ('2.1.0', 'c2c9f01'),
    # vNext M3.3 (ADR-0024): a Canonical Task Program's FollowPath and MoveWaypoint's profile /
    # frame / tcp keys, TMR 2.2.0 (b399a33, the M3.1 surfaces)
    'FollowPath': ('2.2.0', 'b399a33'),
    'waypoint_canonical_keys': ('2.2.0', 'b399a33'),
    # vNext M3.4 (ADR-0025 Decision 3): a ptp / free / lin move's goal tolerance, honoured on the
    # MoveGroupInterface routes since TMR 2.3.0 (8ed8a36)
    'move_goal_tolerance': ('2.3.0', '8ed8a36'),
    # vNext M3.5 (M3_DESIGN §9): a task program's process events -- the ProcessEvent node,
    # DetectObject's trigger, FollowPath's from / to and paths.yaml's events, TMR 2.4.0 (6a2414e)
    'ProcessEvent': ('2.4.0', '6a2414e'),
    'detect_trigger': ('2.4.0', '6a2414e'),
    'follow_path_range': ('2.4.0', '6a2414e'),
    'path_events': ('2.4.0', '6a2414e'),
    # vNext M3.6 (ADR-0026): the grasp-candidate contract's nodes, TMR 2.5.0
    'DetectCandidates': ('2.5.0', 'cb4d1a2'),
    'SetWaypointFromCandidate': ('2.5.0', 'cb4d1a2'),
    # ADR-0028: a pose's taught joint configuration, TMR 2.6.0 (5635584)
    'pose_configuration': ('2.6.0', '5635584'),
}


def test_the_floors_are_the_curated_ones():
    assert TMR_FIRST_VERSION == '1.4.0'        # 0.0.0 certifies nothing (ADR-0021 M2.4-3)
    assert {name: row[:2] for name, row in TMR_FLOORS.items()} == PINNED_FLOORS


def test_every_floor_is_a_version_the_compiling_tmr_had():
    """A floor above the TMR the snapshot was derived from would date a capability this TSA
    cannot know; a floor at or below the first version says nothing.

    A row whose TMR commit does not exist yet carries the PENDING placeholder, and this test
    FAILS, naming it, until the commit is filled (here in PINNED_FLOORS and in catalog
    TMR_FLOORS) and the catalog refreshed for that TMR (vNext M2.6 did so for TMR 2.0.0)."""
    compiled = catalog.snapshot()['derived_with']['trainit_motion_runtime']
    pending = sorted(n for n, row in TMR_FLOORS.items() if row[1] == TMR_EEF_EVIDENCE_PENDING)
    for name, (version, commit, reason) in TMR_FLOORS.items():
        assert re.fullmatch(r'\d+\.\d+\.\d+', version), name
        # it goes into an XML comment
        assert reason and '--' not in reason and '\n' not in reason, name
        if name in pending:
            continue
        assert _key(TMR_FIRST_VERSION) < _key(version) <= _key(compiled), name
        assert re.fullmatch(r'[0-9a-f]{7}', commit), name
    assert not pending, (
        f'TMR_FLOORS rows {pending} still carry {TMR_EEF_EVIDENCE_PENDING!r}: fill the TMR 2.0.0 '
        f'commit that ships them, then refresh the catalog for TMR 2.0.0 (the snapshot records TMR '
        f'{compiled})')


def test_the_node_rows_name_tmr_nodes_and_the_builtins_are_not_tmr():
    assert {n for n in TMR_FLOORS if n[0].isupper()} <= set(TMR_BT_NODES)
    assert not set(BT_BUILTIN_NODES) & set(TMR_BT_NODES)
    assert len(set(TMR_BT_NODES)) == len(TMR_BT_NODES)


def test_versions_are_compared_as_numbers(monkeypatch):
    monkeypatch.setitem(rr.TMR_FLOORS, 'ServoTrack', ('1.10.0', 'b56fbe3', 'ServoTrack'))
    tree = _tree('<ServoTrack waypoint="a"/><DetectObject detector="d"/>')
    assert rr.tmr_requirement(trees=[tree]).version == '1.10.0'


# --- 2. the closed world -------------------------------------------------------------------------

# Everything a generated bundle can carry that TMR reads, each checked against TMR's history by
# the M2.4 floor scan: present in 1.4.0 unless TMR_FLOORS (or the note) says otherwise. A name
# that is not here fails ``test_every_tmr_surface_a_bundle_carries_is_dated``: date it first.
DATED_PORTS = {
    'MoveWaypoint': {'waypoint', 'fault'},           # fault: 2.1.0 (ADR-0022 amendment M2.7)
    'ServoTrack': {'waypoint'},                                            # ServoTrack: 1.5.0
    # the end-effector contract's ports (ADR-0022, 2.0.0): a canonical tree's package always
    # carries the eef block, whose row dates them
    'OpenGripper': {'eef', 'site', 'fault', 'reason'},
    'CloseGripper': {'eef', 'site', 'fault', 'reason'}, 'ResetScene': set(),
    'VerifyGrasp': {'eef', 'site', 'expect', 'fault', 'reason'},                # 2.0.0
    'CommandActuator': {'eef', 'actuator', 'target', 'fault', 'reason'},        # 2.0.0
    'EnableProcess': {'channel'}, 'DisableProcess': {'channel'},
    'AddCollisionObject': {'id', 'frame', 'dims', 'position'},
    'AttachObject': {'id', 'link', 'dims', 'position'}, 'DetachObject': {'id', 'link'},
    'SetAttachedCollisionCheck': {'value'}, 'SetReleasePolicy': {'policy'},
    'DetectObject': {'detector', 'class_id', 'target_frame', 'timeout_ms', 'out_key',
                     'topic', 'keep',                                        # keep: 1.6.0
                     'fault',                                      # fault: 2.0.0 (ADR-0022 D5)
                     'trigger'},                                   # trigger: 2.4.0 (vNext M3.5)
    'SetWaypointFromDetection': {'waypoint', 'from', 'dx', 'dy', 'dz', 'orientation',
                                 'index'},                                     # index: 1.6.0
    'SetWaypointRelative': {'waypoint', 'from', 'dx', 'dy', 'dz', 'droll', 'dpitch', 'dyaw'},
    'RunPolicy': {'service', 'status_topic', 'mode', 'timeout_ms'},           # 1.5.0
    'CheckRobotState': {'position', 'position_tolerance', 'require_attached', 'attached_topic',
                        'timeout_ms'},                                         # 1.5.0
    'Log': {'message'},
    # BehaviorTree.CPP's own
    'root': {'BTCPP_format', 'main_tree_to_execute'}, 'BehaviorTree': {'ID'},
    # _failureIf: BehaviorTree.CPP's own precondition, the canonical bin-picking gate (ADR-0022 D5)
    'Sequence': {'name', '_failureIf'}, 'Fallback': {'name', '_failureIf'}, 'Sleep': {'msec'},
    'Script': {'code'},
    'Repeat': {'num_cycles'}, 'RetryUntilSuccessful': {'num_attempts'},
    # vNext M3.3: a task program's path (TMR 2.2.0) and its task subtrees (BehaviorTree.CPP's own)
    'FollowPath': {'path',
                   'from', 'to'},                       # vNext M3.5: follow_path_range, 2.4.0
    'SubTree': {'ID', '_autoremap'},
    # vNext M3.5: the generic process event (2.4.0)
    'ProcessEvent': {'kind', 'channel', 'value'},
    # vNext M3.6 (ADR-0026), 2.5.0
    'DetectCandidates': {'source', 'detector', 'target_frame', 'timeout_ms', 'keep', 'out_key', 'class_id',
                         'default_approach'},
    'SetWaypointFromCandidate': {'from', 'index', 'grasp', 'pregrasp', 'retract', 'standoff_m',
                                 'retract_along_approach', 'retract_dz', 'orientation', 'orientation_from',
                                 'eef', 'site'},
}
DATED_BT_PARAMS = {'planner_mode', 'process_path_backend', 'group', 'base_frame', 'tip_link',
                   'gripper_action', 'process_controller', 'bt_tree_id', 'enforce_validation',
                   'gripper_cmd_topic', 'task_parameters',
                   'gripper_close_position', 'gripper_open_position',          # these two: 1.5.0
                   'eef'}                                                      # 2.0.0 (ADR-0022)
DATED_WAYPOINT_KEYS = {'type', 'position', 'orientation', 'named', 'joints', 'motion', 'planner',
                       'speed', 'aux', 'aux_is_center',
                       'accel',                       # MoveWaypoint's since TMR_FIRST_VERSION
                       'profile', 'frame', 'tcp'}     # vNext M3.3: waypoint_canonical_keys, 2.2.0
DATED_SCENE_KEYS = {'frame_id', 'force_republish_hz', 'scene_update_wait_ms', 'gripper_cmd_topic',
                    'scene_reset_topic', 'attach_link', 'attached_collision_check',
                    'grasp_attach_mode', 'touch_links', 'attach_object_ids', 'object_ids',
                    'objects'}
# touchable_collision_ids: metadata no TMR version reads (no floor)
DATED_SCENE_OBJECT_KEYS = {'type', 'mesh_path', 'scale', 'radius', 'height', 'size', 'position',
                           'orientation', 'dynamic', 'grasp_box', 'grasp_box_center',
                           'touchable_collision_ids'}
DATED_HELPER_ARGS = {'robot_name', 'moveit_config_package', 'use_isaac', 'trajectory_execution',
                     'backend'}                                               # backend: 1.5.0
DATED_TMR_EXECUTABLES = {'trainit_run_bt', 'scene_manager_node'}
DATED_TMR_IMPORTS = {'build_move_group_params', 'available_pipeline_names'}
DATED_RUN_BT_ARGS = {'bt_tree_file', 'planner_mode', 'motion_profiles_file', 'available_pipelines',
                     'groot_monitor',
                     'paths_file'}       # vNext M3.3: with FollowPath (2.2.0), never without it
# the values TSA's own vocabularies write into what TMR reads (free-text fields pass through)
DATED_VALUES = {
    'MotionType': {'ptp', 'lin', 'circ', 'free'},
    'PlannerId': {'pilz', 'ompl', 'ompl_chomp'},
    'ReleasePolicy': {'freeze', 'gravity'},
    'PolicyMode': {'pure', 'hybrid', 'residual'},          # passed through to the policy runtime
    'WaypointType': {'tcp', 'joint'},
}


def _tags_in_the_emitters() -> set:
    tags = set()
    for source in sorted((PKG / 'trainit_setup_assistant' / 'applications').glob('*.py')):
        tags |= set(re.findall(r'<([A-Z][A-Za-z]+)[\s/>]', source.read_text(encoding='utf-8')))
    return tags


def test_every_bt_node_the_applications_name_has_a_floor():
    """A static scan of the emitters' literal tags (a tag computed at run time -- MoveWaypoint
    or ServoTrack -- is seen by the matrix below and by the runtime check)."""
    tags = _tags_in_the_emitters()
    assert {'MoveWaypoint', 'DetectObject', 'RunPolicy', 'EnableProcess', 'Sequence'} <= tags
    assert tags - set(BT_BUILTIN_NODES) <= set(TMR_BT_NODES), \
        sorted(tags - set(BT_BUILTIN_NODES) - set(TMR_BT_NODES))
    assert set(DATED_PORTS) == set(TMR_BT_NODES) | set(BT_BUILTIN_NODES)


def test_the_values_tsa_writes_are_the_dated_ones():
    for enum, values in DATED_VALUES.items():
        assert {m.value for m in getattr(E, enum)} == values, enum


def _tree(body: str) -> str:
    return ('<?xml version="1.0"?>\n<root BTCPP_format="4">\n  <BehaviorTree ID="MainTree">\n'
            f'    <Sequence name="main">\n      {body}\n    </Sequence>\n  </BehaviorTree>\n'
            '</root>\n')


def test_an_undated_node_stops_generation():
    with pytest.raises(RuntimeError, match="'MoveToTheMoon'"):
        rr.tmr_requirement(trees=[_tree('<MoveToTheMoon waypoint="a"/>')])


def test_an_undated_node_stops_a_real_generation(monkeypatch, tmp_path):
    """Through the Orchestrator: the generator dates every node it writes, or says which not."""
    monkeypatch.setattr(rr, 'TMR_BT_NODES', tuple(n for n in TMR_BT_NODES if n != 'Log'))
    identity_cases.require('p2_example')
    with pytest.raises(RuntimeError, match="'Log'"):
        Orchestrator().generate(_example(), tmp_path / 'out')


@pytest.mark.parametrize('keep', ['{n}', '\u00b2', '-1', ''])
def test_a_keep_that_is_not_a_count_stops_generation(keep):
    with pytest.raises(RuntimeError, match='not a count'):
        rr.tmr_requirement(trees=[_tree(f'<DetectObject detector="d" keep="{keep}"/>')])


# --- 3. the requirement, read from a package's own files ----------------------------------------

FIRST = ('trainit_motion_runtime >= 1.4.0, the first version its package.xml states: nothing '
         'newer is used here')
KEEP_1 = 'DetectObject keep=1 takes the first published match'
KEEP = 'DetectObject keep (a candidate list)'
INDEX = 'SetWaypointFromDetection index'
SERVO = 'ServoTrack (execution: servo)'
POLICY = 'RunPolicy (a policy step)'
CHECK = 'CheckRobotState'
BACKEND = "the launch helper's backend= (gazebo)"
GRIPPER = 'the joint-position gripper targets'
SCENE_MANAGER_NS = 'scene_manager_node under a namespace'
SCENE_SERVICES_NS = 'SetAttachedCollisionCheck/ResetScene reach the cell namespace'
VERIFY = 'VerifyGrasp (the end-effector contract)'
COMMAND = 'CommandActuator (the end-effector contract)'
DETECT_FAULT = "DetectObject's fault output"
EEF_BLOCK = 'the eef parameter block (ADR-0022)'
WAYPOINT_FAULT = "MoveWaypoint's fault output"
RUN_BT_NS = 'trainit_run_bt reaches MoveIt in the cell namespace (ADR-0023)'
FOLLOW_PATH = 'FollowPath (a path of config/paths.yaml)'                   # vNext M3.3
CANONICAL_KEYS = "MoveWaypoint's profile, frame and tcp keys"            # vNext M3.3
MOVE_TOLERANCE = "a move's goal tolerance on a MoveGroupInterface route"   # vNext M3.4
PROCESS_EVENT = 'ProcessEvent (a process event on a logical channel)'      # vNext M3.5
DETECT_TRIGGER = "DetectObject's trigger (a capture of a fresh result)"
PATH_RANGE = "FollowPath from / to (a path's event-free runs)"
PATH_EVENTS = 'paths.yaml events (where a path stops for a process event)'
DETECT_CANDIDATES = 'DetectCandidates (a grasp-candidate set; trainit_interfaces >= 0.2.0 through TMR)'   # M3.6
CANDIDATE_TARGETS = "SetWaypointFromCandidate (a grasp idiom's targets)"


def _for(version: str, *reasons: str) -> str:
    return f'trainit_motion_runtime >= {version} for: {"; ".join(reasons)}'


@pytest.mark.parametrize('body, version, used', [
    ('<MoveWaypoint waypoint="a"/><CloseGripper/><OpenGripper/><Log message="m"/>', '1.4.0', []),
    ('<AddCollisionObject id="t" frame="b" dims="1;1;1" position="0;0;0"/>'
     '<AttachObject id="c" link="tcp" dims="1;1;1" position="0;0;0"/>'
     '<DetachObject id="c" link="tcp"/><EnableProcess channel="x"/>'
     '<DisableProcess channel="x"/>', '1.4.0', []),
    ('<SetAttachedCollisionCheck value="true"/><SetReleasePolicy policy="freeze"/><ResetScene/>',
     '1.4.0', []),
    ('<SetWaypointFromDetection waypoint="a" from="k" dz="0.010" orientation="keep"/>'
     '<SetWaypointRelative waypoint="b" from="a" dz="0.050"/>', '1.4.0', []),
    ('<ServoTrack waypoint="a"/>', '1.5.0', ['ServoTrack']),
    ('<RunPolicy service="s" status_topic="t" mode="pure" timeout_ms="100"/>', '1.5.0',
     ['RunPolicy']),
    ('<CheckRobotState require_attached="true" timeout_ms="100"/>', '1.5.0', ['CheckRobotState']),
    ('<DetectObject detector="d" keep="3" out_key="k"/>', '1.6.0', ['detect_keep']),
    ('<SetWaypointFromDetection waypoint="a" from="k" index="{i}" dz="0.0"/>', '1.6.0',
     ['detection_index']),
    ('<DetectObject detector="d" out_key="k"/>', '1.7.0', ['detect_first_published']),
    ('<DetectObject topic="/p/t" class_id="c" out_key="k"/>', '1.7.0',
     ['detect_first_published']),
    ('<DetectObject detector="d" keep="1" out_key="k"/>', '1.7.0',
     ['detect_keep', 'detect_first_published']),
    ('<DetectObject detector="d" keep="0" out_key="k"/>', '1.7.0',       # TMR reads keep < 1 as 1
     ['detect_keep', 'detect_first_published']),
    # vNext M2.7: MoveWaypoint's fault, wherever a tree remaps it (TMR < 2.1.0 refuses the port)
    ('<MoveWaypoint waypoint="a" fault="{g}"/>', '2.1.0', ['move_waypoint_fault']),
])
def test_the_floor_of_a_tree(body, version, used):
    assert rr.capabilities(trees=[_tree(body)]) == used
    assert rr.tmr_requirement(trees=[_tree(body)]).version == version


def test_every_tree_of_a_package_counts():
    trees = [_tree('<MoveWaypoint waypoint="a"/>'), _tree('<DetectObject detector="d" keep="2"/>')]
    assert rr.tmr_requirement(trees=trees) == rr.Requirement('1.6.0', _for('1.6.0', KEEP))


RUN_BT = 'Node(package="trainit_motion_runtime", executable="trainit_run_bt", output="screen"{})\n'


@pytest.mark.parametrize('node', ['<SetAttachedCollisionCheck value="false"/>', '<ResetScene/>'])
def test_each_scene_node_needs_1_7_only_when_the_tree_runs_in_a_namespace(node):
    tree = _tree(node)
    in_slot = rr.tmr_requirement(trees=[tree], launches=['NS = "/r3"\n' +
                                                         RUN_BT.format(', namespace=NS')])
    assert in_slot == rr.Requirement('1.7.0', _for('1.7.0', SCENE_SERVICES_NS))
    for launch in (RUN_BT.format(''), RUN_BT.format(', namespace=""'),
                   RUN_BT.format(', namespace=None'),
                   # another package's node in a namespace does not put the tree there
                   'NS = "/r3"\nNode(package="moveit_servo", executable="trainit_run_bt", '
                   'namespace=NS)\n'):
        assert rr.tmr_requirement(trees=[tree], launches=[launch]) == \
            rr.Requirement('1.4.0', FIRST), launch


def test_the_launch_helpers_backend_argument_is_read_as_a_call_not_as_text():
    called = 'build_move_group_params(robot_name="r", use_isaac=True, backend=mode)\n'
    mentioned = ('# build_move_group_params(backend=mode) would be the gazebo form\n'
                 'build_move_group_params(robot_name="r", use_isaac=True)\n'
                 'BACKEND = "backend=mode"\n')
    assert rr.tmr_requirement(launches=[called]) == rr.Requirement('1.5.0', _for('1.5.0', BACKEND))
    assert rr.tmr_requirement(launches=[mentioned]).version == '1.4.0'


def test_a_launch_the_floor_cannot_read_is_a_tsa_defect():
    """Fail closed: a template refactor must not lower a floor silently."""
    with pytest.raises(RuntimeError, match='backend='):
        rr.tmr_requirement(launches=['build_move_group_params(**kwargs)\n'])
    with pytest.raises(RuntimeError, match='cannot read'):
        rr.tmr_requirement(launches=['Node(package="trainit_motion_runtime", executable=EXE)\n'])
    with pytest.raises(RuntimeError, match='cannot read'):
        rr.tmr_requirement(launches=['Node(package="trainit_motion_runtime", **node_args)\n'])
    # the node the package must be seen to start, started through a constant
    hidden = 'TMR = "trainit_motion_runtime"\nNode(package=TMR, executable="trainit_run_bt")\n'
    with pytest.raises(RuntimeError, match="'trainit_run_bt'"):
        rr.tmr_requirement(launches=[hidden], needs=('trainit_run_bt',))
    assert rr.tmr_requirement(launches=[RUN_BT.format('')], needs=('trainit_run_bt',)).version \
        == '1.4.0'


def test_a_scene_manager_under_a_namespace_needs_1_5():
    launch = ('nodes.append(Node(package="trainit_motion_runtime", executable="scene_manager_node",'
              ' name="scene_manager_node", namespace=NS))\n')
    assert rr.tmr_requirement(launches=[launch]) == rr.Requirement(
        '1.5.0', _for('1.5.0', SCENE_MANAGER_NS))
    assert rr.tmr_requirement(launches=[launch.replace(', namespace=NS', '')]).version == '1.4.0'


_EEF_PARAMS = '/**:\n  ros__parameters:\n    gripper_action: ""\n    eef:\n      ids: ["main"]\n'
_LEGACY_PARAMS = '/**:\n  ros__parameters:\n    group: "arm"\n'
_RUN_BT_IN_R1 = 'NS = "/r1"\n' + RUN_BT.format(', namespace=NS')


def test_a_canonical_runner_in_a_cell_namespace_needs_2_1():
    """vNext G0 (owner section 5; ADR-0023; ADR-0021 amendment G0): a package whose bt_params
    carry the end-effector contract's eef block AND whose trainit_run_bt runs under a non-empty
    namespace= relies on TMR 2.1.0's MoveIt clients resolving in that namespace (c2c9f01) -- any
    namespace expression, as for the other namespace facts. The comment names that reason alone
    (the contract's rows sit at 2.0.0); with the motion gate, both 2.1.0 reasons in table order."""
    want = rr.Requirement('2.1.0', _for('2.1.0', RUN_BT_NS))
    assert rr.tmr_requirement(bt_params=[_EEF_PARAMS], launches=[_RUN_BT_IN_R1],
                              needs=('trainit_run_bt',)) == want
    assert rr.capabilities(bt_params=[_EEF_PARAMS], launches=[_RUN_BT_IN_R1]) == \
        ['eef_block', 'moveit_clients_namespace']
    for slot in (', namespace="/r3"', ', namespace=LaunchConfiguration("ns")'):
        assert rr.tmr_requirement(bt_params=[_EEF_PARAMS],
                                  launches=[RUN_BT.format(slot)]) == want, slot
    gate = _tree('<MoveWaypoint waypoint="a" fault="{g}"/>')
    assert rr.tmr_requirement(trees=[gate], bt_params=[_EEF_PARAMS],
                              launches=[_RUN_BT_IN_R1]) == \
        rr.Requirement('2.1.0', _for('2.1.0', WAYPOINT_FAULT, RUN_BT_NS))


def test_a_root_or_legacy_runner_keeps_the_floor_it_had():
    """vNext G0, the other half (owner section 5: a non-consuming legacy or other package keeps
    its previous floor, existing capabilities included; owner M2.7 sections 4 and 17): the same
    canonical package at the root; a LEGACY package, namespaced or not -- its runner relies on the
    same TMR 2.1.0 behaviour, yet its floor stays (KNOWN_DEBT); a namespaced scene_manager_node
    (the config package: no runner); another package's node in a namespace; a legacy file that
    does not parse (the row is canonical-only)."""
    for root in (RUN_BT.format(''), RUN_BT.format(', namespace=""'),
                 RUN_BT.format(', namespace=None')):
        assert rr.tmr_requirement(bt_params=[_EEF_PARAMS], launches=[root]) == \
            rr.Requirement('2.0.0', _for('2.0.0', EEF_BLOCK)), root
    for launch in (_RUN_BT_IN_R1, RUN_BT.format('')):
        assert rr.tmr_requirement(bt_params=[_LEGACY_PARAMS], launches=[launch]) == \
            rr.Requirement('1.4.0', FIRST), launch
    assert rr.tmr_requirement(trees=[_tree('<ResetScene/>')], bt_params=[_LEGACY_PARAMS],
                              launches=[_RUN_BT_IN_R1]) == \
        rr.Requirement('1.7.0', _for('1.7.0', SCENE_SERVICES_NS))
    scene_manager = ('NS = "/r1"\nNode(package="trainit_motion_runtime", '
                     'executable="scene_manager_node", namespace=NS)\n')
    assert rr.tmr_requirement(launches=[scene_manager]) == \
        rr.Requirement('1.5.0', _for('1.5.0', SCENE_MANAGER_NS))
    other = 'NS = "/r1"\nNode(package="moveit_servo", executable="trainit_run_bt", namespace=NS)\n'
    assert rr.tmr_requirement(bt_params=[_EEF_PARAMS], launches=[other]) == \
        rr.Requirement('2.0.0', _for('2.0.0', EEF_BLOCK))
    assert 'moveit_clients_namespace' in EEF_CONTRACT_CAPABILITIES
    broken = rr.use_of(trees={'t.xml': _tree('<MoveWaypoint waypoint="a&b"/>')},
                       bt_params=[_LEGACY_PARAMS], launches=[_RUN_BT_IN_R1])
    assert broken.unreadable and not broken.canonical
    assert rr.requirement(broken).version == '1.7.0'


def test_the_joint_position_gripper_targets_need_1_5():
    params = textwrap.dedent("""\
        /**:
          ros__parameters:
            gripper_close_position: 0.7
            gripper_open_position: 0.0
        """)
    assert rr.tmr_requirement(bt_params=[params]) == rr.Requirement('1.5.0', _for('1.5.0', GRIPPER))
    assert rr.tmr_requirement(bt_params=['/**:\n  ros__parameters:\n    group: "arm"\n']) \
        == rr.Requirement('1.4.0', FIRST)


def test_the_comment_names_only_what_set_the_floor_in_table_order():
    tree = _tree('<ServoTrack waypoint="a"/><DetectObject detector="d" keep="2"/>'
                 '<SetWaypointFromDetection waypoint="a" from="k" index="{i}"/>'
                 '<DetectObject detector="e" keep="4"/>')
    assert rr.tmr_requirement(trees=[tree]) == rr.Requirement('1.6.0', _for('1.6.0', KEEP, INDEX))


def test_the_policy_runtime_is_needed_by_a_tree_or_by_a_launch_file():
    policy_tree = _tree('<RunPolicy service="s" status_topic="t" mode="pure" timeout_ms="1"/>')
    starts = 'Node(package="trainit_policy_runtime", executable="policy_runtime_node")\n'
    assert rr.uses_policy_runtime(trees=[policy_tree])
    assert rr.uses_policy_runtime(trees=[_tree('<MoveWaypoint waypoint="a"/>')], launches=[starts])
    assert not rr.uses_policy_runtime(trees=[_tree('<MoveWaypoint waypoint="a"/>')],
                                      launches=[RUN_BT.format('')])


def test_a_file_that_does_not_parse_assumes_every_dated_capability():
    """A project name that breaks the XML or the Python TSA writes: the file cannot be read, so
    the floor is the newest version TSA dates -- never lower -- and the file is named (with the
    kind of error, never the parser's message: it quotes project data). "Dated" for a LEGACY
    package: never the end-effector contract's rows (ADR-0022) -- only a package whose bt_params
    carry the eef block can use them -- so its floor is the one it had before they existed."""
    newest = max((row[0] for name, row in TMR_FLOORS.items()
                  if name not in EEF_CONTRACT_CAPABILITIES
                  and name not in TASK_PROGRAM_CAPABILITIES), key=_key)     # vNext M3.3: likewise
    bad_tree = _tree('<MoveWaypoint waypoint="a&b"/>')
    bad_launch = 'x = 1\nNode(package="trainit_policy_runtime", x=")\n'
    use = rr.use_of(trees={'cell_app/bt_trees/t.xml': bad_tree},
                    launches={'cell_app/launch/l.py': bad_launch})
    assert use.unreadable == ('cell_app/launch/l.py (SyntaxError at line 2)',
                              'cell_app/bt_trees/t.xml (ParseError at line 5)')
    assert use.policy_runtime                         # read from the text it could not parse
    req = rr.requirement(use, needs=('trainit_run_bt',))
    assert req.version == newest
    assert req.comment == (f'trainit_motion_runtime >= {newest}: {", ".join(use.unreadable)} '
                           f'does not parse, so every capability TSA dates is assumed')


def test_a_canonical_file_that_does_not_parse_assumes_the_contract_rows_too():
    """The same unreadable tree in a package whose bt_params carry the end-effector contract's
    eef block (ADR-0022): every dated row, the contract's included -- TMR 2.1.0 (vNext M2.7: the
    motion gate's row is a contract row too)."""
    bad_tree = _tree('<MoveWaypoint waypoint="a&b"/>')
    params = '/**:\n  ros__parameters:\n    gripper_action: ""\n    eef:\n      ids: ["main"]\n'
    use = rr.use_of(trees={'cell_app/bt_trees/t.xml': bad_tree}, bt_params=[params])
    assert use.canonical and 'eef_block' in use.capabilities
    req = rr.requirement(use)
    # vNext M3.3: never a task program's rows (FollowPath, the canonical waypoint keys: 2.2.0) --
    # only a package with FollowPath, a SubTree or those keys can use them
    assert req.version == max((row[0] for name, row in TMR_FLOORS.items()
                               if name not in TASK_PROGRAM_CAPABILITIES), key=_key) == '2.1.0'


# --- 4. the application matrix -------------------------------------------------------------------

_CARD = textwrap.dedent("""
    policy_card:
      name: fr5_suction_pickplace
      files: {jit: policy.pt, onnx: policy.onnx}
      end_state:
        tcp_pose_base: [0.4267, -0.278, 0.1589, -0.006, -0.7006, 0.7134, 0.0117]
        object_attached: true
""")


UR5E = PKG / 'examples' / 'ur5e_suction'


def _task_program(_card=None):
    """vNext M3.3: the public cell's Canonical Task Program (Validation A), loaded the README's way
    (from its directory) and its description directory made absolute, so it generates from any
    directory."""
    with _cwd(UR5E):
        p = load_project('task_program.yaml')
    p.robot.description.urdf_dir = str(UR5E / 'urdf')
    p.bundle = BundleSpec.from_prefix('cell')
    return p


def _task_program_move_tolerance(_card=None):
    """vNext M3.4: the same program, its lin/circ `approach` profile carrying a goal tolerance (the
    lin moves w2 and w4 then rely on TMR 2.3.0 honouring it on the MoveGroupInterface route)."""
    from trainit_setup_assistant.model.motion import GoalTolerance
    p = _task_program()
    p.application.program.motion_profiles['approach'].goal_tolerance = GoalTolerance(position_m=0.002)
    return p


M3_5 = PKG / 'test' / 'data' / 'm3_5'


def _task_program_process(_card=None):
    """vNext M3.5: Validation B (an inspection program: process steps on a logical channel, a path
    with three captures), plus a set_value step -- every M3.5 surface a bundle can carry."""
    from trainit_setup_assistant.model.program import ProcessStep, TaskStep
    with _cwd(M3_5):
        p = load_project('validation_b.yaml')
    p.robot.description.urdf_dir = str(UR5E / 'urdf')
    p.bundle = BundleSpec.from_prefix('cell')
    p.application.program.tasks[0].steps.insert(3, TaskStep(process=ProcessStep(
        kind='set_value', channel='inspection_light', value=0.8)))
    return p


M3_6 = PKG / 'test' / 'data' / 'm3_6'


def _task_program_manipulation(_card=None):
    """vNext M3.6: Validation C (a detect, a grasp from candidates, a release at a taught target) plus
    a tool step -- every M3.6 surface a bundle can carry."""
    from trainit_setup_assistant.model.program import TaskStep, ToolStep, ToolVerify
    with _cwd(M3_6):
        p = load_project('validation_c.yaml')
    p.robot.description.urdf_dir = str(UR5E / 'urdf')
    p.bundle = BundleSpec.from_prefix('cell')
    p.application.program.tasks[0].steps.append(TaskStep(tool=ToolStep(verify=ToolVerify(expect='released'))))
    return p


@contextlib.contextmanager
def _cwd(path):
    old = os.getcwd()
    if path is not None:
        os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _ur_description() -> bool:
    try:
        import xacro  # noqa: F401
        from ament_index_python.packages import get_package_share_directory
        get_package_share_directory('ur_description')
        return True
    except Exception:                                   # noqa: BLE001
        return False


def _example():
    p = load_project(str(EXAMPLE))
    p.bundle = BundleSpec.from_prefix('cell')
    return p


def _golden():
    p = load_project(str(GOLDEN_PROJECT))
    p.bundle = BundleSpec.from_prefix('cell')
    return p


def _no_vision(p):
    """The golden's scene-loader cell without perception: no detector, no camera-guided pose."""
    p.perception.detectors = []
    p.application.detections = []
    for wp in p.application.waypoints:
        wp.vision = None
    return p


def _policy(p, card: Path, mode: str, before: str):
    p.application.policies = [PolicyStep(name='grasp', before_waypoint=before,
                                         mode=PolicyMode(mode), card=str(card),
                                         require_attached=True,
                                         attached_topic='/suction/attached')]
    return p


def _servo(p):
    seg = p.application.segment_for('pre_pick')
    seg.execution, seg.motion = ExecutionMode.SERVO, MotionType.FREE
    return p


def _gazebo(p):
    p.deployment.modes = [Backend.MOCK, Backend.ISAAC, Backend.GAZEBO, Backend.REAL]
    return p


def _namespaced(p):
    p.robot.cell_namespace = 'r3'
    return p


def _reset_scene(p):
    p.application.tool_actions.append(ToolAction(at_waypoint='home',
                                                 kind=ToolActionKind.RESET_SCENE))
    return p


def _parallel_jaw(p):
    g = p.robot.gripper
    g.kind, g.actuation = GripperKind.PARALLEL, EndEffectorActuation.JOINT_POSITION
    g.joint_target, g.open_angle, g.closed_angle = GripperJointTarget.ANGLE, 0.0, 0.7
    p.scene.gripper_cmd_topic = '/grasp_cmd'
    return p


def _renamed_waypoint(p, old, new):
    """A waypoint name the generated tree cannot hold (TSA does not escape it: known debt)."""
    app = p.application
    app.waypoint_by_name(old).name = new
    for seg in app.segments:
        seg.to_waypoint = new if seg.to_waypoint == old else seg.to_waypoint
    app.sequence = [new if n == old else n for n in app.sequence]
    for action in app.tool_actions:
        action.at_waypoint = new if action.at_waypoint == old else action.at_waypoint
    return p


def _bin_picking(**cfg):
    import test_bin_picking
    return test_bin_picking._bin_project(**cfg)


def _seg():
    return identity_cases.seg_project()


# --- the end-effector contract (ADR-0022): canonical cases, generated in temp dirs only ---------

def _canonical_golden():
    import eef_fixture
    return eef_fixture.opt_in(_golden())


def _canonical_bin_picking():
    """The golden's bin picking, opted in with grasp confirmation -- mock only: on isaac a grasp
    is observed only with a stage status topic (EEF-D5), and the contract refuses it without."""
    import eef_fixture
    p = _bin_picking(attached_topic='')
    p.deployment.modes, p.deployment.default_mode = [Backend.MOCK], Backend.MOCK
    return eef_fixture.opt_in(p, adapters={'mock': 'trainit_mock'}, grasp=True)


def _canonical_jaw_actuate(card):
    """A GripperCommand jaw (from scratch, synthetic) commanded to a third named target."""
    import eef_fixture
    contract = eef_fixture.jaw_contract()
    contract['actuator']['targets'] = {'open': 0.0, 'closed': 0.7, 'half': 0.35}
    p = eef_fixture.jaw_simple(card.parent.parent / 'eef_jaw', contract=contract)
    p.bundle = BundleSpec.from_prefix('cell')
    p.application.tool_actions.append(ToolAction(at_waypoint='flip',
                                                 kind=ToolActionKind.ACTUATE,
                                                 actuator='gripper', target='half'))
    return p


def _composite_mock_r1(card):
    """vNext M2.7 (WP_COMP section 7): the section-13 composite tool, robot_a in /r1 -- a mock-only
    composite BUNDLE (R22 narrowed), from scratch (synthetic): its app starts the mock adapter."""
    import eef_fixture
    p = eef_fixture.composite_mock(card.parent.parent / 'composite_mock', 'robot_a', '/r1')
    p.bundle = BundleSpec.from_prefix('cell')
    return p


def _canonical_suction(card, ns):
    """vNext G0 (owner section 5): eef_fixture's simple suction cell (from scratch, synthetic,
    mock only), the same project at the root and in a cell namespace."""
    import eef_fixture
    where = card.parent.parent / ('suction' + (ns.replace('/', '_') or '_root'))
    p = eef_fixture.suction_simple(where, ns=ns)
    p.bundle = BundleSpec.from_prefix('cell')
    return p


def _canonical_golden_r1():
    """vNext G0: the golden opted in under /r1 (the T16 golden_ns stage C cell): the isaac stage
    channel relative (R15) and the merger's gripper input relative (R26), as
    test_m2_6_eef_emission.test_f3_a_namespaced_canonical_isaac_cell builds it."""
    import eef_fixture
    p = eef_fixture.opt_in(_golden(), adapters={
        'mock': 'trainit_mock', 'real': 'none',
        'isaac': {'adapter': 'trainit_isaac_surface_gripper', 'stage_cmd_topic': 'suction_cmd'}})
    p.robot.cell_namespace = '/r1'
    merger = next(b for b in p.deployment.bridges if b.executable == 'joint_state_merger')
    merger.parameters = [{k: ('suction/joint_states' if k == 'gripper_joint_states_in' else v)
                          for k, v in d.items()} for d in merger.parameters]
    return p


def _missing(card: Path) -> Path:
    return card.parent.parent / 'nowhere' / 'policy_card.yaml'


CASES = {   # name -> (needs the golden, builder(card))
    'A_motion_only': (False, lambda c: _example()),
    'A_motion_only_scene_loader': (True, lambda c: _no_vision(_golden())),
    'B_vision_colour_mask': (True, lambda c: _golden()),
    'C_external_node': (False, lambda c: _seg()),
    'D_pure_policy': (False, lambda c: _policy(_example(), c, 'pure', 'pick')),
    'D_pure_policy_scene_loader': (True, lambda c: _policy(_no_vision(_golden()), c, 'pure',
                                                           'pre_place')),
    'D_pure_policy_with_vision': (True, lambda c: _policy(_golden(), c, 'pure', 'pre_place')),
    'E_hybrid_policy': (False, lambda c: _policy(_example(), c, 'hybrid', 'pick')),
    'E_hybrid_policy_with_vision': (True, lambda c: _policy(_golden(), c, 'hybrid', 'pre_place')),
    'E_residual_policy': (False, lambda c: _policy(_example(), c, 'residual', 'pick')),
    'P_policy_card_missing': (False, lambda c: _policy(_example(), _missing(c), 'pure', 'pick')),
    'P_policy_anchored_nowhere': (True, lambda c: _policy(_golden(), c, 'pure', 'nowhere')),
    'P_policy_in_bin_picking': (True, lambda c: _policy(_bin_picking(), c, 'pure', 'pre_place')),
    'S_servo': (False, lambda c: _servo(_example())),
    'S_servo_scene_loader': (True, lambda c: _servo(_no_vision(_golden()))),
    'Z_gazebo': (False, lambda c: _gazebo(_example())),
    'Z_gazebo_with_scene_loader': (True, lambda c: _gazebo(_golden())),
    'N_namespace': (False, lambda c: _namespaced(_example())),
    'N_namespace_reset_scene': (False, lambda c: _reset_scene(_namespaced(_example()))),
    'N_namespace_with_scene_loader': (True, lambda c: _namespaced(_golden())),
    'K_bin_picking': (True, lambda c: _bin_picking()),
    'K_bin_picking_one_try': (True, lambda c: _bin_picking(tries=1)),
    'J_joint_position_gripper': (False, lambda c: _parallel_jaw(_example())),
    'U_unparseable_tree': (False, lambda c: _renamed_waypoint(_example(), 'post_place',
                                                               'post&place')),
    # the end-effector contract (ADR-0022): only an opted-in bundle gets the TMR 2.0.0 floor
    'Q_canonical_suction': (True, lambda c: _canonical_golden()),
    'Q_canonical_bin_picking': (True, lambda c: _canonical_bin_picking()),
    'Q_canonical_jaw_actuate': (False, _canonical_jaw_actuate),
    'Q_composite_mock_r1': (False, _composite_mock_r1),
    # vNext G0 (owner section 5): a canonical runner in a cell namespace, and its root twin
    'Q_canonical_suction_root': (False, lambda c: _canonical_suction(c, '')),
    'Q_canonical_suction_r3': (False, lambda c: _canonical_suction(c, '/r3')),
    'Q_canonical_golden_r1': (True, lambda c: _canonical_golden_r1()),
    # vNext M3.3: a Canonical Task Program (FollowPath, the canonical waypoint keys, paths_file)
    'T_task_program': (False, _task_program),
    # vNext M3.4: a lin move whose profile carries a goal tolerance (TMR 2.3.0)
    'T_task_program_move_tolerance': (False, _task_program_move_tolerance),
    # vNext M3.5: process events (ProcessEvent, DetectObject trigger, FollowPath from/to, path events)
    'T_task_program_process': (False, _task_program_process),
    'T_task_program_manipulation': (False, _task_program_manipulation),
}
# F (no policy) is every case without a policy step; G (no perception) every case without a TP
# detector node: A, C, D, E, P (but the golden-based ones), S, Z, N, J, U.

APP, CFG, DESC = 'cell_app', 'cell_trainit_config', 'cell_description'
BASE = ('1.4.0', FIRST)
V17 = ('1.7.0', _for('1.7.0', KEEP_1))
PURE = ('1.5.0', _for('1.5.0', POLICY, CHECK))
# the line is the first one of the generated tree that holds the broken name (see _unreadable)
UNREADABLE = ('1.7.0', 'trainit_motion_runtime >= 1.7.0: cell_app/bt_trees/pick_place.xml '
                       '(ParseError at line {line}) does not parse, so every capability TSA '
                       'dates is assumed')
# case -> ({package: (TMR floor, its comment)}, the app's TP line ('plain' | None), policy runtime)
EXPECTED = {
    'A_motion_only': ({APP: BASE, CFG: BASE}, None, False),
    'A_motion_only_scene_loader': ({APP: BASE, CFG: BASE}, None, False),
    'B_vision_colour_mask': ({APP: V17, CFG: BASE}, 'plain', False),
    'C_external_node': ({APP: BASE, CFG: BASE}, None, False),
    'D_pure_policy': ({APP: PURE, CFG: BASE}, None, True),
    'D_pure_policy_scene_loader': ({APP: PURE, CFG: BASE}, None, True),
    'D_pure_policy_with_vision': ({APP: V17, CFG: BASE}, 'plain', True),
    'E_hybrid_policy': ({APP: V17, CFG: BASE}, None, True),
    'E_hybrid_policy_with_vision': ({APP: V17, CFG: BASE}, 'plain', True),
    'E_residual_policy': ({APP: PURE, CFG: BASE}, None, True),
    # RunPolicy is in the tree; no card shipped, so no launch file starts the node
    'P_policy_card_missing': ({APP: PURE, CFG: BASE}, None, True),
    # no RunPolicy in the tree, but policy_runtime.launch.py ships and bring-up starts it
    'P_policy_anchored_nowhere': ({APP: V17, CFG: BASE}, 'plain', True),
    'P_policy_in_bin_picking': ({APP: ('1.6.0', _for('1.6.0', KEEP, INDEX)), CFG: BASE}, 'plain',
                                True),
    'S_servo': ({APP: ('1.5.0', _for('1.5.0', SERVO)), CFG: BASE}, None, False),
    'S_servo_scene_loader': ({APP: ('1.5.0', _for('1.5.0', SERVO)), CFG: BASE}, None, False),
    'Z_gazebo': ({APP: ('1.5.0', _for('1.5.0', BACKEND)), CFG: BASE}, None, False),
    'Z_gazebo_with_scene_loader': ({APP: V17, CFG: ('1.5.0', _for('1.5.0', BACKEND))}, 'plain',
                                   False),
    # LEGACY namespaced bundles: their runner relies on TMR 2.1.0 too, yet their floors stay
    # (owner M2.7 sections 4 and 17; vNext G0 raises only the canonical one: KNOWN_DEBT)
    'N_namespace': ({APP: BASE, CFG: BASE}, None, False),
    'N_namespace_reset_scene': ({APP: ('1.7.0', _for('1.7.0', SCENE_SERVICES_NS)), CFG: BASE},
                                None, False),
    'N_namespace_with_scene_loader': ({APP: ('1.7.0', _for('1.7.0', KEEP_1, SCENE_SERVICES_NS)),
                                       CFG: ('1.5.0', _for('1.5.0', SCENE_MANAGER_NS))},
                                      'plain', False),
    'K_bin_picking': ({APP: ('1.6.0', _for('1.6.0', KEEP, INDEX)), CFG: BASE}, 'plain', False),
    'K_bin_picking_one_try': ({APP: V17, CFG: BASE}, 'plain', False),
    'J_joint_position_gripper': ({APP: ('1.5.0', _for('1.5.0', GRIPPER)), CFG: BASE}, None,
                                 False),
    'U_unparseable_tree': ({APP: UNREADABLE, CFG: BASE}, None, False),
    'Q_canonical_suction': ({APP: ('2.0.0', _for('2.0.0', EEF_BLOCK)), CFG: BASE}, 'plain',
                            False),
    'Q_canonical_bin_picking': ({APP: ('2.1.0', _for('2.1.0', WAYPOINT_FAULT)), CFG: BASE},
                                'plain', False),           # vNext M2.7: the motion gate
    'Q_canonical_jaw_actuate': ({APP: ('2.0.0', _for('2.0.0', COMMAND, EEF_BLOCK)), CFG: BASE},
                                None, False),
    # the mock-only composite bundle in /r1 (vNext M2.7): its runner is namespaced, so since vNext
    # G0 it needs TMR 2.1.0 -- the contract's rows (2.0.0) are below the floor and not named
    'Q_composite_mock_r1': ({APP: ('2.1.0', _for('2.1.0', RUN_BT_NS)), CFG: BASE}, None, False),
    # vNext G0 (owner section 5): the same simple suction cell at the root and in /r3, and the
    # golden opted in under /r1 (its scene-loader config keeps its 1.5.0 scene-manager floor)
    'Q_canonical_suction_root': ({APP: ('2.0.0', _for('2.0.0', EEF_BLOCK)), CFG: BASE}, None,
                                 False),
    'Q_canonical_suction_r3': ({APP: ('2.1.0', _for('2.1.0', RUN_BT_NS)), CFG: BASE}, None,
                               False),
    'Q_canonical_golden_r1': ({APP: ('2.1.0', _for('2.1.0', RUN_BT_NS)),
                               CFG: ('1.5.0', _for('1.5.0', SCENE_MANAGER_NS))}, 'plain', False),
    # vNext M3.3: a Canonical Task Program needs TMR 2.2.0 for what it uses (its eef rows, 2.0.0,
    # are below the floor and not named)
    'T_task_program': ({APP: ('2.2.0', _for('2.2.0', FOLLOW_PATH, CANONICAL_KEYS)), CFG: BASE},
                       None, False),
    'T_task_program_move_tolerance': ({APP: ('2.3.0', _for('2.3.0', MOVE_TOLERANCE)), CFG: BASE},
                                      None, False),
    # vNext M3.5: process events need TMR 2.4.0 for all four surfaces
    'T_task_program_process': ({APP: ('2.4.0', _for('2.4.0', PROCESS_EVENT, DETECT_TRIGGER, PATH_RANGE,
                                                    PATH_EVENTS)), CFG: BASE}, None, False),
    # vNext M3.6: the manipulation idioms need TMR 2.5.0 for the candidate contract's two nodes (the tool
    # step's VerifyGrasp, 2.0.0, is below the floor and not named)
    'T_task_program_manipulation': ({APP: ('2.5.0', _for('2.5.0', DETECT_CANDIDATES, CANDIDATE_TARGETS)),
                                     CFG: BASE}, None, False),
}
# the package whose launch starts the end-effector adapters, per canonical case (ADR-0022 [D])
HARDWARE = {'Q_canonical_suction': (CFG, 'isaac_eef_adapter, mock_eef_adapter'),
            'Q_canonical_bin_picking': (CFG, 'mock_eef_adapter'),
            'Q_canonical_jaw_actuate': (APP, 'mock_eef_adapter'),
            'Q_composite_mock_r1': (APP, 'mock_eef_adapter'),
            'Q_canonical_suction_root': (APP, 'mock_eef_adapter'),
            'Q_canonical_suction_r3': (APP, 'mock_eef_adapter'),
            'Q_canonical_golden_r1': (CFG, 'isaac_eef_adapter, mock_eef_adapter'),
            'T_task_program': (APP, 'mock_eef_adapter'),
            'T_task_program_move_tolerance': (APP, 'mock_eef_adapter'),
            'T_task_program_process': (APP, 'mock_eef_adapter'),
            'T_task_program_manipulation': (APP, 'mock_eef_adapter')}


# the generated text a package reference can live in: the closed world of dependencies (vNext
# M2.7) reads the descriptions and the RViz files too ($(find ...), package://); meshes, CAD files,
# the README and CMakeLists.txt are not read
TEXT_SUFFIXES = ('.xml', '.py', '.yaml', '.xacro', '.urdf', '.srdf', '.rviz')


def _read(out: Path) -> dict:
    return {f.relative_to(out).as_posix(): f.read_text(encoding='utf-8')
            for f in sorted(out.rglob('*')) if f.is_file() and f.suffix in TEXT_SUFFIXES}


@pytest.fixture(scope='module')
def bundles():
    """Every case generated once: its files (xml, py, yaml) as text, by bundle path."""
    if identity_cases.missing('p2_example'):
        pytest.skip(identity_cases.missing('p2_example'))
    out = {}
    with tempfile.TemporaryDirectory(prefix='tsa_m2_4_') as tmp:
        tmp = Path(tmp)
        card = tmp / 'card' / 'policy_card.yaml'
        card.parent.mkdir()
        card.write_text(_CARD)
        (card.parent / 'policy.pt').write_bytes(b'JIT')
        (card.parent / 'policy.onnx').write_bytes(b'ONNX')
        for name, (needs_golden, build) in CASES.items():
            if needs_golden and not GOLDEN_PROJECT.is_file():
                continue
            if name.startswith('T_') and not _ur_description():
                continue                                   # the UR5e cell needs ur_description
            manifest = Orchestrator().generate(build(card), tmp / name)
            out[name] = dict(_read(tmp / name), _warnings=list(manifest.warnings))
    return out


def _case(bundles, name):
    if name not in bundles:
        pytest.skip('the golden bundle is not present')
    return bundles[name]


def _files(bundles):
    for name, files in bundles.items():
        for rel, text in files.items():
            if rel != '_warnings':
                yield name, rel, text


def _deps(xml: str) -> list:
    root = ET.fromstring(xml[xml.index('<package'):])
    return [(el.tag, (el.text or '').strip(), dict(el.attrib)) for el in root
            if el.tag.endswith('depend')]


def _tmr_lines(xml: str) -> list:
    """(version_gte, the comment right above) of each TMR dependency line."""
    lines = xml.splitlines()
    return [(re.search(r'version_gte="([^"]*)"', line).group(1) if 'version_gte' in line
             else None, lines[i - 1].strip())
            for i, line in enumerate(lines)
            if line.strip().endswith('>trainit_motion_runtime</exec_depend>')]


def _broken_line(files) -> int:
    """The file line where the generated tree first holds the name it cannot carry."""
    lines = files[f'{APP}/bt_trees/pick_place.xml'].splitlines()
    return next(i for i, line in enumerate(lines, 1) if 'post&place' in line)


@pytest.mark.parametrize('name', list(EXPECTED))
def test_the_application_matrix(bundles, name):
    files = _case(bundles, name)
    tmr, tp, policy = EXPECTED[name]
    for pkg, (version, comment) in tmr.items():
        comment = comment.format(line=_broken_line(files)) if '{line}' in comment else comment
        assert _tmr_lines(files[f'{pkg}/package.xml']) == [(version, f'<!-- {comment} -->')], pkg
    app = _deps(files[f'{APP}/package.xml'])
    tp_lines = [attrs for tag, dep, attrs in app if dep == 'trainit_perception']
    assert tp_lines == ({'plain': [{}], None: []}[tp]), tp_lines
    assert [dep for tag, dep, attrs in app if dep == 'trainit_policy_runtime'] == \
        (['trainit_policy_runtime'] if policy else [])
    # TP and the policy runtime only ever in the app; the description package names no component
    for pkg in (CFG, DESC):
        assert not {dep for _t, dep, _a in _deps(files[f'{pkg}/package.xml'])} & \
            {'trainit_perception', 'trainit_policy_runtime'}, pkg
    assert not {dep for _t, dep, _a in _deps(files[f'{DESC}/package.xml'])} & set(COMPONENTS)


def _hardware_lines(xml: str) -> list:
    lines = xml.splitlines()
    return [(re.search(r'version_gte="([^"]*)"', line).group(1) if 'version_gte' in line
             else None, lines[i - 1].strip())
            for i, line in enumerate(lines)
            if line.strip().endswith('>trainit_hardware_layer</exec_depend>')]


@pytest.mark.parametrize('name', list(EXPECTED))
def test_the_hardware_layer_is_declared_iff_a_launch_starts_an_adapter(bundles, name):
    """ADR-0022 [D]: trainit_hardware_layer version_gte 0.2.0, with the comment naming the
    adapters that set it, in the package whose generated launch starts them -- and nowhere else,
    and never in a legacy bundle."""
    files = _case(bundles, name)
    for pkg in (APP, CFG, DESC):
        rel = f'{pkg}/package.xml'
        if rel not in files:
            continue
        want = []
        if name in HARDWARE and HARDWARE[name][0] == pkg:
            want = [('0.2.0', f'<!-- trainit_hardware_layer >= 0.2.0 for: the end-effector '
                              f'adapter(s) {HARDWARE[name][1]} (ADR-0022) -->')]
        assert _hardware_lines(files[rel]) == want, (name, pkg)


def test_the_policy_runtime_is_declared_iff_a_generated_file_uses_it(bundles):
    """The M2.4-4 invariant, over every case: the app names trainit_policy_runtime iff one of
    the bundle's generated trees or launch files does (RunPolicy's service is
    /trainit_policy_runtime/run; the launch starts its node)."""
    for name, files in bundles.items():
        declared = 'trainit_policy_runtime' in [d for _t, d, _a in _deps(files[f'{APP}/package.xml'])]
        used = any('trainit_policy_runtime' in text for rel, text in files.items()
                   if rel.endswith(('.py', '.xml')) and not rel.endswith('package.xml'))
        assert declared == used, name


@pytest.mark.parametrize('name', list(EXPECTED))
def test_no_component_is_a_build_dependency_and_none_stays_unversioned(bundles, name):
    for rel, text in _case(bundles, name).items():
        if not rel.endswith('package.xml'):
            continue
        for tag, dep, attrs in _deps(text):
            if dep in COMPONENTS:
                assert tag == 'exec_depend', (rel, tag, dep)
            if dep == 'trainit_motion_runtime':
                assert set(attrs) == {'version_gte'}, (rel, attrs)


def test_a_file_that_does_not_parse_is_reported_and_generation_completes(bundles):
    files = _case(bundles, 'U_unparseable_tree')
    warnings = files['_warnings']
    assert (f'runtime requirement: cell_app/bt_trees/pick_place.xml (ParseError at line '
            f'{_broken_line(files)}) does not parse -- the bundle will not run as generated; '
            f'trainit_motion_runtime is required at the newest version TSA dates') in warnings
    assert 'post&place' not in ' '.join(w for w in warnings
                                         if w.startswith('runtime requirement'))


def test_residual_stays_not_usable(bundles):
    """Declaring what the generated launch starts does not make residual usable: TSA still
    warns, as before M2.4."""
    warnings = _case(bundles, 'E_residual_policy')['_warnings']
    assert any('mode "residual"' in w and 'not available yet' in w for w in warnings), warnings


def test_the_package_xml_files_are_read_by_ros_tooling_as_written(bundles):
    """catkin_pkg -- what colcon and rosdep read -- sees each version_gte TSA writes."""
    parse = pytest.importorskip('catkin_pkg.package').parse_package_string
    for name, rel, text in _files(bundles):
        if not rel.endswith('package.xml'):
            continue
        package = parse(text)
        package.validate()
        for dep in package.exec_depends:
            if dep.name == 'trainit_motion_runtime':
                assert dep.version_gte == _tmr_lines(text)[0][0], (name, rel)


def _surfaces(bundles) -> dict:
    """Every TMR surface the generated bundles carry, by kind."""
    out = {k: set() for k in ('ports', 'bt_params', 'waypoint', 'scene', 'scene_object',
                              'helper_args', 'executables', 'imports', 'run_bt_args')}
    for _name, rel, text in _files(bundles):
        if '/bt_trees/' in rel and rel.endswith('.xml') and '&' not in text:
            for el in ET.fromstring(text[text.index('<root'):]).iter():
                out['ports'] |= {(el.tag, a) for a in el.attrib} | {(el.tag, None)}
        elif rel.endswith('config/bt_params.yaml'):
            for body in yaml.safe_load(text).values():
                params = body['ros__parameters']
                out['bt_params'] |= set(params)
                for wp in (params.get('task_parameters') or {}).values():
                    if isinstance(wp, dict):
                        out['waypoint'] |= set(wp)
        elif rel.endswith('config/scene.yaml'):
            for body in yaml.safe_load(text).values():
                params = body['ros__parameters']
                out['scene'] |= set(params)
                for obj in (params.get('objects') or {}).values():
                    out['scene_object'] |= set(obj)
        elif rel.endswith('.py') and '/launch/' in rel:
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.ImportFrom) and (node.module or '').startswith(
                        'trainit_motion_runtime'):
                    out['imports'] |= {a.name for a in node.names}
                if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == \
                        'build_move_group_params':
                    out['helper_args'] |= {k.arg for k in node.keywords}
                if isinstance(node, ast.Call) and getattr(node.func, 'id', '') == 'Node':
                    kw = {k.arg: k.value for k in node.keywords}
                    if getattr(kw.get('package'), 'value', '') == 'trainit_motion_runtime':
                        out['executables'].add(kw['executable'].value)
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and \
                        [getattr(t, 'id', '') for t in node.targets] == ['node_params']:
                    out['run_bt_args'] |= {k.value for k in node.value.keys}
    return out


def test_every_tmr_surface_a_bundle_carries_is_dated(bundles):
    s = _surfaces(bundles)
    undated = sorted((tag, a) for tag, a in s['ports']
                     if tag not in DATED_PORTS or (a is not None and a not in DATED_PORTS[tag]))
    assert not undated, f'date these in TMR_FLOORS first, then here: {undated}'
    assert s['bt_params'] <= DATED_BT_PARAMS, s['bt_params'] - DATED_BT_PARAMS
    assert s['waypoint'] <= DATED_WAYPOINT_KEYS, s['waypoint'] - DATED_WAYPOINT_KEYS
    assert s['scene'] <= DATED_SCENE_KEYS, s['scene'] - DATED_SCENE_KEYS
    assert s['scene_object'] <= DATED_SCENE_OBJECT_KEYS, s['scene_object'] - DATED_SCENE_OBJECT_KEYS
    assert s['helper_args'] <= DATED_HELPER_ARGS, s['helper_args'] - DATED_HELPER_ARGS
    assert s['executables'] == DATED_TMR_EXECUTABLES, s['executables']
    assert s['imports'] == DATED_TMR_IMPORTS, s['imports']
    assert s['run_bt_args'] == DATED_RUN_BT_ARGS, s['run_bt_args']
    # the matrix reaches what is dated: a pin nothing exercises would be dead
    assert {tag for tag, _a in s['ports']} == set(DATED_PORTS)
    assert {'gripper_close_position', 'gripper_open_position'} <= s['bt_params']
    assert 'backend' in s['helper_args']


# --- the closed world of dependencies (vNext M2.7, brief section 12) -----------------------------

TMR, TP, POLICY_RT, HW = ('trainit_motion_runtime', 'trainit_perception', 'trainit_policy_runtime',
                          'trainit_hardware_layer')
RUNTIME_DEPENDS = ('depend', 'exec_depend', 'run_depend')       # the tags a run time resolves
_NODE_CALLS = ('Node', 'ComposableNode', 'LifecycleNode', 'ExecutableInPackage')     # package=
_SHARE_CALLS = ('get_package_share_directory', 'get_package_share_path', 'get_package_prefix',
                'FindPackage', 'FindPackagePrefix', 'FindPackageShare')        # the package first
_PROCESS_CALLS = ('ExecuteProcess',)            # a command line: the package it runs is not read
_FIND = re.compile(r'\$\(find(?:-pkg-share)?\s+([\w-]+)\)')
_PACKAGE_URI = re.compile(r'package://([\w-]+)/')
_SUBSTITUTED = re.compile(r'\$\(find(?:-pkg-share)?\s+\$|package://\$')   # a name xacro computes
_TREE = re.compile(r'^[^/]+/bt_trees/[^/]+\.xml$')


def _called(func) -> str:
    return func.id if isinstance(func, ast.Name) else getattr(func, 'attr', '')


def _python_uses(rel: str, text: str):
    """(uses, unreadable) of a generated Python file: ``(form, package)`` for every import, every
    launched node's ``package=``, every share or prefix lookup and every BRIDGES row. A package
    argument is a literal, a module-level literal constant (``MOVEIT_CONFIG_PACKAGE``,
    ``REAL_INCLUDE["package"]``; a constant that is None names nothing: the code it guards never
    runs) or the BRIDGES loop's ``b["package"]``. Anything else -- another expression (a call, a
    ``**`` spread), a process command line, a file that does not parse -- is what this scan cannot
    read: it is returned as unreadable, and the test fails on it."""
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        return set(), [f'{rel}: does not parse (SyntaxError at line {exc.lineno})']
    consts = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and \
                isinstance(node.targets[0], ast.Name):
            try:
                consts[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass                                    # not a literal: never read as a name
    loops = {}                                          # for <variable> in <CONSTANT>: ...
    for node in ast.walk(tree):
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and \
                isinstance(node.iter, ast.Name):
            loops.setdefault(node.target.id, set()).add(node.iter.id)

    def rows_of(name):
        """What ``name[...]`` reads from: a constant, or every row of the list constants a loop
        variable runs over -- None when the scan cannot tell."""
        if name in consts:
            return [consts[name]]
        lists = [consts.get(it) for it in sorted(loops.get(name, ()))]
        if lists and all(isinstance(v, list) for v in lists):
            return [row for v in lists for row in v]
        return None

    def names(expr):
        """The package names ``expr`` evaluates to -- None when the scan cannot tell."""
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return {expr.value}
        if isinstance(expr, ast.Name) and expr.id in consts:
            value = consts[expr.id]
            if value is None or isinstance(value, str):
                return set() if value is None else {value}
        if isinstance(expr, ast.Subscript) and isinstance(expr.value, ast.Name) and \
                isinstance(expr.slice, ast.Constant):
            rows, key = rows_of(expr.value.id), expr.slice.value
            if rows is not None and all(row is None or isinstance(row, dict)
                                        and isinstance(row.get(key), str) for row in rows):
                return {row[key] for row in rows if row is not None}
        return None

    uses, unreadable = set(), []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            uses |= {('import', a.name.split('.')[0]) for a in node.names}
        elif isinstance(node, ast.ImportFrom) and not node.level:
            uses.add(('import', node.module.split('.')[0]))
        elif isinstance(node, ast.Call) and _called(node.func) in _PROCESS_CALLS:
            unreadable.append(f'{rel}:{node.lineno}: a process command line: {ast.unparse(node)}')
        elif isinstance(node, ast.Call) and _called(node.func) in _NODE_CALLS + _SHARE_CALLS:
            kw = {k.arg: k.value for k in node.keywords}
            if _called(node.func) == 'ExecutableInPackage':
                form, arg = 'node', kw.get('package', node.args[1] if len(node.args) > 1 else None)
            elif _called(node.func) in _NODE_CALLS:
                form, arg = 'node', kw.get('package')
                if arg is None and None not in kw:
                    continue                            # no package=: an executable on the PATH
            else:
                form, arg = 'share', node.args[0] if node.args else kw.get('package')
            found = None if arg is None else names(arg)
            if found is None:
                unreadable.append(f'{rel}:{node.lineno}: {ast.unparse(arg or node)}')
            else:
                uses |= {(form, name) for name in found}
    for row in consts.get('BRIDGES') or []:
        if isinstance(row, dict) and isinstance(row.get('package'), str):
            uses.add(('bridges', row['package']))
        else:
            unreadable.append(f'{rel}: a BRIDGES row without a package: {row!r}')
    return uses, unreadable


def _uses(rel: str, text: str):
    """(uses, unreadable) of one generated file of a package (``rel`` = ``<package>/<path>``): the
    Python forms above, a xacro ``$(find X)`` / ``$(find-pkg-share X)`` and a ``package://X/`` URI
    in any text, and a behaviour tree or ``bt_params.yaml``, which only TMR's runner reads (a tree
    with a RunPolicy step also calls the policy runtime's service: ADR-0021 M2.4-4)."""
    uses, unreadable = set(), []
    if rel.endswith('.py'):
        uses, unreadable = _python_uses(rel, text)
    if not rel.endswith('/package.xml'):                # the declarations, not a use
        uses |= {('find', m) for m in _FIND.findall(text)}
        uses |= {('package://', m) for m in _PACKAGE_URI.findall(text)}
        unreadable += [f'{rel}: {m.group(0)}...' for m in _SUBSTITUTED.finditer(text)]
    if _TREE.match(rel) or rel.endswith('/config/bt_params.yaml'):
        uses.add(('bt', TMR))
        if _TREE.match(rel) and re.search(r'<RunPolicy[\s/>]', text):
            uses.add(('bt', POLICY_RT))
    return uses, unreadable


def _package_files(files: dict) -> dict:
    """{package directory: {rel: text}} of one bundle: a package is a top-level directory with a
    package.xml; the bundle's own manifest lies outside every package."""
    out = {rel.split('/')[0]: {} for rel in files if rel.count('/') == 1
           and rel.endswith('/package.xml')}
    for rel, text in files.items():
        if '/' in rel:
            out.setdefault(rel.split('/')[0], {})[rel] = text
    return out


@pytest.mark.parametrize('name', list(EXPECTED))
def test_every_trainit_package_a_generated_file_uses_is_declared_by_its_package(bundles, name):
    """The closed world of dependencies (vNext M2.7, owner brief section 12): for every package of
    every matrix case -- the composite mock bundle in /r1 included (WP_COMP section 7) -- the
    TrainIt packages its OWN generated files use (a launched node, a share lookup, a Python import,
    a BRIDGES row, a xacro $(find), a package:// URI, a tree or bt_params for TMR's runner) are
    among the packages its package.xml declares at run time. A subset, not an equality: the
    from-scratch moveit_config declares TMR by design (the planner pipelines come from it at launch
    time); the converse stays per component (the matrix and the hardware-layer tests)."""
    packages = _package_files({rel: text for rel, text in _case(bundles, name).items()
                               if rel != '_warnings'})
    for pkg, files in sorted(packages.items()):
        assert f'{pkg}/package.xml' in files, f'{pkg}: a generated directory that is no package'
        used, unreadable = set(), []
        for rel, text in files.items():
            u, bad = _uses(rel, text)
            used |= u
            unreadable += bad
        assert not unreadable, (pkg, unreadable)
        trainit = {p for _form, p in used if p.startswith('trainit_')}
        declared = {dep for tag, dep, _a in _deps(files[f'{pkg}/package.xml'])
                    if tag in RUNTIME_DEPENDS}
        assert trainit <= declared, (f'{pkg} uses TrainIt packages its package.xml does not '
                                     f'declare: {sorted(trainit - declared)}')


def test_the_dependency_scan_reads_every_form_the_bundles_carry(bundles):
    """The closed world above is not vacuous: over the matrix's packages each form names the
    TrainIt packages it is there for, the xacro and URI forms read the copied descriptions' own
    references (none of them TrainIt's), and the TrainIt packages generated files use are exactly
    these four -- trainit_interfaces comes with TMR and the hardware layer, never from a bundle
    (ADR-0021 M2.4-1). The bundle's own manifest is no package's file: the sources it records
    (``package://<package>/...``) are provenance, not uses."""
    missing = sorted(set(EXPECTED) - set(bundles))
    if missing:
        pytest.skip(f'the golden bundle is not present: {len(missing)} cases not generated')
    seen, others = {form: set() for form in ('import', 'node', 'share', 'bridges', 'bt')}, set()
    for name, files in bundles.items():
        packages = _package_files({rel: text for rel, text in files.items() if rel != '_warnings'})
        for rel, text in (item for pkg_files in packages.values() for item in pkg_files.items()):
            uses, unreadable = _uses(rel, text)
            assert not unreadable, (name, unreadable)
            for form, pkg in uses:
                if pkg.startswith('trainit_'):
                    seen.setdefault(form, set()).add(pkg)
                else:
                    others.add(form)
    assert seen['import'] >= {TMR} and seen['share'] >= {TMR}
    assert seen['node'] >= {TMR, TP, POLICY_RT, HW}         # HW: the BRIDGES loop's b["package"]
    assert seen['bridges'] >= {HW} and seen['bt'] >= {TMR, POLICY_RT}
    assert {'find', 'package://'} <= others
    assert set().union(*seen.values()) == {TMR, TP, POLICY_RT, HW}


_SCANNED_LAUNCH = textwrap.dedent('''\
    from trainit_policy_runtime.x import y
    MOVEIT_CONFIG_PACKAGE = "cell_trainit_config"
    REAL_INCLUDE = None
    BRIDGES = [{"package": "trainit_hardware_layer", "executable": "mock_eef_adapter"}]
    get_package_share_directory(MOVEIT_CONFIG_PACKAGE)
    get_package_share_directory(REAL_INCLUDE["package"])
    for b in BRIDGES:
        Node(package=b["package"], executable=b["executable"])
    Node(package="trainit_perception", executable="detector_node")
    ''')


@pytest.mark.parametrize('rel, text, uses, unreadable', [
    ('a/launch/l.py', _SCANNED_LAUNCH,
     {('import', POLICY_RT), ('share', 'cell_trainit_config'), ('bridges', HW), ('node', HW),
      ('node', TP)}, []),
    ('a/launch/l.py', 'Node(package=LaunchConfiguration("p"))\n', set(),
     ['a/launch/l.py:1: LaunchConfiguration(\'p\')']),
    ('a/launch/l.py', 'P = "trainit_x" + ""\nFindPackageShare(P)\n', set(),
     ['a/launch/l.py:2: P']),
    ('a/urdf/r.xacro', '<x a="$(find trainit_x)/u" b="package://trainit_y/m.stl"/>',
     {('find', 'trainit_x'), ('package://', 'trainit_y')}, []),
    ('a/urdf/r.xacro', '<x a="$(find ${pkg})/u"/> <!-- package:// paths -->', set(),
     ['a/urdf/r.xacro: $(find $...']),
    ('a/bt_trees/t.xml', '<root><RunPolicy service="s"/></root>', {('bt', TMR), ('bt', POLICY_RT)},
     []),
    ('a/config/bt_params.yaml', '/**: {}\n', {('bt', TMR)}, []),
    ('a/package.xml', '<!-- $(find ${p}) package://$x -->', set(), []),     # never a use
    ('a/launch/l.py', 'get_package_share_path("trainit_a")\nFindPackagePrefix("trainit_b")\n'
     'ExecutableInPackage("exe", "trainit_c")\nFindPackage(package="trainit_d")\n',
     {('share', 'trainit_a'), ('share', 'trainit_b'), ('node', 'trainit_c'),
      ('share', 'trainit_d')}, []),
    ('a/launch/l.py', 'Node(executable="tool")\nNode(**args)\n', set(),
     ['a/launch/l.py:2: Node(**args)']),
    ('a/launch/l.py', 'ExecuteProcess(cmd=["ros2", "run", "trainit_x", "y"])\n', set(),
     ["a/launch/l.py:1: a process command line: "
      "ExecuteProcess(cmd=['ros2', 'run', 'trainit_x', 'y'])"]),
    ('a/launch/l.py', 'Node(package="trainit_x"\n', set(),
     ['a/launch/l.py: does not parse (SyntaxError at line 1)']),
])
def test_the_dependency_scan_reads_each_form_and_fails_closed(rel, text, uses, unreadable):
    """The scan's own rules, on synthetic files: what each form names, a constant or BRIDGES
    indirection followed, a None constant naming nothing, a node with no package naming nothing,
    and whatever it cannot read -- an expression, a ``**`` spread, a process command line, a xacro
    substitution, a file that does not parse -- reported, never skipped."""
    assert _uses(rel, text) == (uses, unreadable)


# --- 5. the Step-3 config, and a scene-loader cell without the golden ---------------------------

@pytest.mark.skipif(not GOLDEN_PROJECT.is_file(), reason='golden bundle not present')
@pytest.mark.parametrize('variant, want', [
    ('plain', BASE),
    ('gazebo_namespace', ('1.5.0', _for('1.5.0', BACKEND, SCENE_MANAGER_NS))),
])
def test_the_step_3_config_declares_what_its_bring_up_uses(variant, want, tmp_path):
    """Step 3 renders the same package as the bundle's config (orchestrator
    ``generate_scene_loader_config``): the same TMR lines."""
    build = _golden if variant == 'plain' else (lambda: _namespaced(_gazebo(_golden())))
    generate_scene_loader_config(build(), tmp_path / 'step3', 'cell_scene_loader_moveit_config')
    step3 = (tmp_path / 'step3' / 'cell_scene_loader_moveit_config' / 'package.xml').read_text()
    assert _tmr_lines(step3) == [(want[0], f'<!-- {want[1]} -->')]
    Orchestrator().generate(build(), tmp_path / 'bundle')
    bundle = (tmp_path / 'bundle' / CFG / 'package.xml').read_text()
    assert step3.replace('cell_scene_loader_moveit_config', CFG) == bundle


def _synthetic_scene_loader(root: Path, *, namespace='', gazebo=False):
    """A scene-loader cell built from a minimal synthetic base config (no golden needed)."""
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    from trainit_setup_assistant.generator.emitters.app_pkg import AppEmitter
    from trainit_setup_assistant.generator.emitters.base import GenContext
    from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
        SceneLoaderMoveitConfigEmitter)
    from trainit_setup_assistant.generator.manifest import GenerationManifest
    from trainit_setup_assistant.model import (CanonicalProject, DeploymentSpec, MotionSegment,
                                               Waypoint)
    from trainit_setup_assistant.model.enums import AppType, WaypointType
    from trainit_setup_assistant.model.robot import (ArmControllerSpec, DescriptionSource,
                                                     GripperSpec, PlanningGroupSpec, RobotSpec)
    cfg = root / 'base' / 'config'
    cfg.mkdir(parents=True)
    (cfg / 'r.urdf.xacro').write_text('<?xml version="1.0"?>\n'
                                      '<robot xmlns:xacro="http://www.ros.org/wiki/xacro" '
                                      'name="r"/>\n')
    (cfg / 'r.srdf').write_text('<?xml version="1.0"?>\n<robot name="r"></robot>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml',
              'moveit_controllers.yaml', 'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    p = CanonicalProject(project_name='syn', bundle=BundleSpec.from_prefix('cell'),
                         robot=RobotSpec(robot_name='syn',
                                         description=DescriptionSource(urdf_dir=str(cfg),
                                                                       top_xacro='r.urdf.xacro'),
                                         base_frame='base_link', tip_link='tcp',
                                         base_moveit_config_path=str(root / 'base'),
                                         planning_group=PlanningGroupSpec(name='arm',
                                                                          joints=['j1']),
                                         gripper=GripperSpec(kind=GripperKind.NONE),
                                         arm_controller=ArmControllerSpec(),
                                         cell_namespace=namespace))
    p.deployment = DeploymentSpec(modes=[Backend.MOCK] + ([Backend.GAZEBO] if gazebo else []),
                                  default_mode=Backend.MOCK)
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = [Waypoint(name='home', type=WaypointType.JOINT, named='home')]
    p.application.segments = [MotionSegment(to_waypoint='home')]
    p.application.sequence = ['home']
    p.application.tool_actions = [ToolAction(at_waypoint='home', kind=ToolActionKind.RESET_SCENE)]
    out = root / 'out'
    for emitter in (SceneLoaderMoveitConfigEmitter(), AppEmitter()):
        emitter.emit(p, GenContext(out, build_jinja_env(), GenerationManifest('syn', 'syn')))
    return ((out / APP / 'package.xml').read_text(), (out / CFG / 'package.xml').read_text())


@pytest.mark.parametrize('namespace, gazebo, app, cfg', [
    ('', False, BASE, BASE),
    ('r3', False, ('1.7.0', _for('1.7.0', SCENE_SERVICES_NS)),
     ('1.5.0', _for('1.5.0', SCENE_MANAGER_NS))),
    ('', True, ('1.5.0', _for('1.5.0', BACKEND)), ('1.5.0', _for('1.5.0', BACKEND))),
])
def test_a_scene_loader_cell_without_the_golden(namespace, gazebo, app, cfg, tmp_path):
    app_xml, cfg_xml = _synthetic_scene_loader(tmp_path, namespace=namespace, gazebo=gazebo)
    assert _tmr_lines(app_xml) == [(app[0], f'<!-- {app[1]} -->')]
    assert _tmr_lines(cfg_xml) == [(cfg[0], f'<!-- {cfg[1]} -->')]


# --- 6. negative tests ---------------------------------------------------------------------------

def test_1_a_snapshot_capability_is_not_a_dependency(bundles):
    """The snapshot knows TP, the policy runtime, MoveIt Servo and TMR 2.3.0; a motion-only
    bundle needs none of them, and its TMR floor is not the snapshot's TMR."""
    snap = catalog.snapshot()
    ids = {row['id'] for row in snap['providers']}
    assert {'moveit.servo', 'policy.pure', 'tp.detector.color_mask'} <= ids
    files = _case(bundles, 'A_motion_only')
    names = {dep for rel, text in files.items() if rel.endswith('package.xml')
             for _t, dep, _a in _deps(text)}
    assert not names & {'trainit_perception', 'trainit_policy_runtime', 'moveit_servo'}
    assert snap['derived_with']['trainit_motion_runtime'] == '2.6.0'      # refreshed for ADR-0028 (TMR 2.6.0)
    assert {v for rel, text in files.items() if rel.endswith('package.xml')
            for v, _c in _tmr_lines(text)} == {'1.4.0'}


def test_1_a_refresh_moves_no_runtime_floor(bundles, monkeypatch, tmp_path):
    """Owner brief M2.5 section 13: COMPILE SNAPSHOT VERSION != AUTOMATIC RUNTIME FLOOR. The TMR
    release the catalog snapshot records is provenance: recording ANOTHER release -- what a
    catalog refresh does -- regenerates every case with every TMR dependency line (its floor and
    the comment naming what set it) unchanged; only the manifest's provenance follows. Causal, not
    a value comparison: a floor that legitimately equals some release stays allowed."""
    release = catalog.snapshot()['derived_with']['trainit_motion_runtime']
    text = (catalog_identity.CATALOG / catalog.SNAPSHOT_FILE).read_text()
    line = f'  trainit_motion_runtime: {release}\n  trainit_perception:'
    assert text.count(line) == 1, 'the snapshot.yaml layout moved: update this edit'
    edited = text.replace(line, '  trainit_motion_runtime: 99.0.0\n  trainit_perception:').encode()
    monkeypatch.setattr(catalog, '_committed', lambda name, _real=catalog._committed: (
        edited if name == catalog.SNAPSHOT_FILE else _real(name)))
    monkeypatch.setitem(catalog._PARSED, catalog.SNAPSHOT_FILE, edited)   # as if read at start
    card = tmp_path / 'card' / 'policy_card.yaml'
    card.parent.mkdir()
    card.write_text(_CARD)
    (card.parent / 'policy.pt').write_bytes(b'JIT')
    (card.parent / 'policy.onnx').write_bytes(b'ONNX')
    compared = 0
    for name, (_needs_golden, build) in CASES.items():
        if name not in bundles:
            continue                                   # the golden is absent: skipped there too
        manifest = Orchestrator().generate(build(card), tmp_path / name)
        assert manifest.capability_snapshot['trainit_motion_runtime']['version'] == '99.0.0'
        other = _read(tmp_path / name)
        for rel, xml in bundles[name].items():
            if rel.endswith('package.xml'):
                assert _tmr_lines(other[rel]) == _tmr_lines(xml), (name, rel)
                compared += 1
    assert compared >= 2


def test_1_the_requirement_never_reads_the_catalog_snapshot():
    """By construction: the module reads its catalog constants at import and nothing else of
    the catalog -- no snapshot, no installed component, no ament index."""
    source = (PKG / 'trainit_setup_assistant' / 'generator' /
              'runtime_requirements.py').read_text(encoding='utf-8')
    tree = ast.parse(source)
    relative = {(n.level, n.module) for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.level}
    assert relative == {(2, 'catalog')}
    assert {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.level
            for a in n.names} == {'BT_BUILTIN_NODES', 'TMR_BT_NODES', 'TMR_FIRST_VERSION',
                                  'TMR_FLOORS', 'EEF_CONTRACT_CAPABILITIES', 'EEF_ADAPTER_ROWS',
                                  'HARDWARE_LAYER',       # vNext M2.6: constants, like the others
                                  'TASK_PROGRAM_CAPABILITIES'}                      # vNext M3.3
    absolute = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
                and not n.level} | {a.name for n in ast.walk(tree) if isinstance(n, ast.Import)
                                    for a in n.names}
    assert absolute <= {'__future__', 'typing', 'ast', 're', 'xml.etree.ElementTree', 'yaml'}
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | \
        {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for forbidden in ('snapshot', 'snapshot_identity', 'derived_with', 'importlib', 'metadata',
                      'get_package_share_directory', 'open', 'Path'):
        assert forbidden not in names, forbidden


def _candidate_tokens():
    """Every candidate provider's id, packages and product name (catalog/candidates.yaml)."""
    exact, words = set(), set()
    generic = {'ess', 'segment', 'nvidia', 'isaac', 'ros', 'ros2', 'moveit', 'node', 'core', 'sim',
               'lab', 'base', 'model', 'models', 'anything', 'stereo', 'pose', 'humble', 'jazzy'}
    for c in catalog.candidates():
        exact.add(c['id'].lower())
        exact.update(p.lower() for p in c.get('packages', []))
        product = c['id'].split('.', 1)[-1].lower()
        exact.add(product)
        words.update(w for w in re.split(r'[_\W]+', product)
                     if w not in generic and len(w) > 3 and not w.isdigit())
    return exact, words


_REFERENCES = re.compile(r'package\s*=\s*["\']([\w-]+)["\']|get_package_share_directory\(\s*'
                         r'["\']([\w-]+)["\']|FindPackageShare\(\s*["\']([\w-]+)["\']|'
                         r'\$\(find(?:-pkg-share)? ([\w-]+)\)|package://([\w-]+)/|'
                         r'find_package\(\s*([\w-]+)')


def test_2_no_nvidia_candidate_becomes_a_dependency(bundles):
    """No package a bundle declares or reaches -- by package.xml, a launched node, a share
    lookup, a xacro $(find), a package:// URI or a CMake find_package -- is a candidate's."""
    exact, words = _candidate_tokens()
    assert {'cumotion', 'foundationpose', 'nvblox'} <= words, 'the deny-list lost NVIDIA words'
    for name, rel, text in _files(bundles):
        used = {dep for _t, dep, _a in _deps(text)} if rel.endswith('package.xml') else set()
        used |= {g for m in _REFERENCES.finditer(text) for g in m.groups() if g}
        for dep in {u.lower() for u in used}:
            assert dep not in exact and not set(re.split(r'[_\W]+', dep)) & words, \
                (name, rel, dep)


def test_3_an_unused_policy_runtime_is_not_a_dependency(bundles):
    """No file of a bundle without a policy step names the policy runtime (ADR-0009)."""
    for name, files in bundles.items():
        if EXPECTED[name][2]:
            continue
        for rel, text in files.items():
            if rel != '_warnings':
                assert 'trainit_policy_runtime' not in text, (name, rel)


def test_4_tp_is_a_dependency_only_where_a_tp_node_runs(bundles):
    """TP's snapshot release (0.5.0) is never a floor, and a bundle that launches no TP
    detector_node -- motion only, an external node, a policy without a detector -- names no TP."""
    tp_release = catalog.snapshot()['derived_with']['trainit_perception']
    assert tp_release == '0.5.0'
    for name, files in bundles.items():
        app = files[f'{APP}/package.xml']
        launches_tp = 'package="trainit_perception"' in files[f'{APP}/launch/trainit_bt.launch.py']
        named = [attrs for _t, dep, attrs in _deps(app) if dep == 'trainit_perception']
        assert bool(named) == launches_tp == (EXPECTED[name][1] is not None), name
        assert all(attrs.get('version_gte') != tp_release for attrs in named), name


# negative test 5: the installed components, generated in fresh interpreters

PREFIX_VARS = ('AMENT_PREFIX_PATH', 'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH')
FAKES = {'trainit_motion_runtime': '9.9.9', 'trainit_perception': '0.0.1',
         'trainit_policy_runtime': '9.9.9'}


def _fakes(root: Path):
    """A prefix and a Python directory where the three components exist, in versions nothing
    else has, and raise when imported."""
    prefix, python = root / 'fake_prefix', root / 'fake_python'
    index = prefix / 'share' / 'ament_index' / 'resource_index' / 'packages'
    index.mkdir(parents=True)
    for name, version in FAKES.items():
        (index / name).write_text('')
        (prefix / 'share' / name).mkdir(parents=True)
        (prefix / 'share' / name / 'package.xml').write_text(
            f'<?xml version="1.0"?>\n<package format="3"><name>{name}</name>'
            f'<version>{version}</version></package>\n')
        (python / name).mkdir(parents=True)
        (python / name / '__init__.py').write_text(
            f'raise RuntimeError("the fake {name} was imported")\n')
    return prefix, python


def _strip(paths: str) -> str:
    return os.pathsep.join(p for p in paths.split(os.pathsep)
                           if p and not any(c in p for c in COMPONENTS))


def _environment(variant: str, root: Path) -> dict:
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(root))
    if variant == 'fake':
        prefix, python = _fakes(root)
        env['PYTHONPATH'] = os.pathsep.join([str(PKG), str(python), env.get('PYTHONPATH', '')])
        for var in PREFIX_VARS:
            env[var] = os.pathsep.join([str(prefix), env.get(var, '')])
    else:                                            # 'none': uninstalled
        env['PYTHONPATH'] = os.pathsep.join([str(PKG), _strip(env.get('PYTHONPATH', ''))])
        for var in PREFIX_VARS:
            env[var] = _strip(env.get(var, ''))
    return env


def _found(env: dict) -> dict:
    """Where each component resolves in ``env`` (None: nowhere), asked in that environment."""
    code = ('import importlib.util, json; print(json.dumps({n: (lambda s: s and s.origin)('
            'importlib.util.find_spec(n)) for n in %r}))' % (COMPONENTS,))
    r = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env,
                       timeout=60)
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_5_the_installed_components_do_not_move_a_requirement():
    """With FAKE trainit_motion_runtime 9.9.9, trainit_perception 0.0.1 and
    trainit_policy_runtime 9.9.9 first on every prefix and on PYTHONPATH, and with the three
    uninstalled, every package.xml is the one this process generates -- the policy bundle's
    included. Each environment is checked to be what it claims."""
    cases = [c for c in ('p1_golden', 'p2_example', 'pol1_policy_test4_project')
             if not identity_cases.missing(c)]
    if 'p2_example' not in cases:
        pytest.skip('needs the example inputs')
    with tempfile.TemporaryDirectory(prefix='tsa_m2_4_env_') as root:
        root = Path(root)
        envs = {}
        for variant in ('fake', 'none'):
            (root / variant).mkdir()
            envs[variant] = _environment(variant, root / variant)
        found = {v: _found(env) for v, env in envs.items()}
        # each environment is what it claims: the fakes first, or no component at all
        assert all((found['fake'][n] or '').startswith(str(root / 'fake')) for n in COMPONENTS), \
            found['fake']
        assert not any(found['none'].values()), found['none']
        for case in cases:
            here = root / f'{case}_here'
            Orchestrator().generate(identity_cases.CASES[case](), here)
            want = {f.relative_to(here).as_posix(): hashlib.sha256(f.read_bytes()).hexdigest()
                    for f in here.glob('*/package.xml')}
            for variant, env in envs.items():
                out = root / f'{case}_{variant}'
                r = subprocess.run([sys.executable, '-B', str(WORKER), '--case', case, '--out',
                                    str(out)], capture_output=True, text=True, env=env,
                                   cwd=str(root), timeout=120)
                line = [ln for ln in r.stdout.splitlines() if ln.startswith('RESULT ')]
                assert line, r.stderr[-1500:]
                result = json.loads(line[-1][len('RESULT '):])
                assert result['outcome'] == 'generated', (case, variant, result.get('crash'))
                got = {rel: sha for rel, sha in result['files'].items()
                       if rel.endswith('/package.xml')}
                assert got == want, (case, variant)
            for xml in here.glob('*/package.xml'):
                assert not {'9.9.9', '0.0.1'} & set(re.findall(r'\d+\.\d+\.\d+',
                                                               xml.read_text())), xml


# --- ADR-0028: a pose's taught joint configuration ----------------------------------------------------

def test_a_pose_with_a_taught_configuration_needs_tmr_2_6_0_and_one_without_keeps_its_floor():
    """An older TMR ignores the entry key `configuration` and the taught configuration is dropped silently: a
    package whose bt_params carry it declares TMR >= 2.6.0; the same package without the key keeps its floor."""
    tree = _tree('<MoveWaypoint waypoint="w1"/>')
    entry = '      w1: {type: tcp, position: [0.4, 0.0, 0.3], orientation: [1.0, 0.0, 0.0, 0.0], profile: transit'
    without = '/**:\n  ros__parameters:\n    task_parameters:\n' + entry + '}\n'
    with_key = '/**:\n  ros__parameters:\n    task_parameters:\n' + entry + ', configuration: [0.1, 0.2]}\n'
    plain = rr.use_of(trees={'t.xml': tree}, bt_params=[without])
    taught = rr.use_of(trees={'t.xml': tree}, bt_params=[with_key])
    assert 'pose_configuration' not in plain.capabilities and rr.requirement(plain).version == '2.2.0'
    assert 'pose_configuration' in taught.capabilities and rr.requirement(taught).version == '2.6.0'
