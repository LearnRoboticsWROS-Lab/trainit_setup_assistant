"""End-effector abstraction model (ADR-0010 F1): the two orthogonal axes — actuation
(trigger | joint_position{srdf_state | angle}) x sim_grasp_adapter (per backend). ROS-free.
Byte-stability of the golden is covered by the byte gate + generator golden (the new fields
are inert data in F1; the generator is unchanged)."""

import json

import pytest

from trainit_setup_assistant.model import (
    Backend, EndEffectorActuation, GripperJointTarget, SimGraspAdapter)
from trainit_setup_assistant.model.enums import DEFAULT_SIM_GRASP_ADAPTER, GripperKind
from trainit_setup_assistant.model.robot import GripperSpec


def test_enum_tokens_frozen():
    assert [e.value for e in EndEffectorActuation] == ['trigger', 'joint_position']
    assert [e.value for e in GripperJointTarget] == ['srdf_state', 'angle']
    assert [e.value for e in SimGraspAdapter] == ['surface_gripper', 'link_attacher', 'none']


# --- actuation defaults from kind (byte-safe: suction stays TRIGGER) -----------

def test_actuation_defaults_from_kind():
    assert GripperSpec(kind=GripperKind.SUCTION).actuation is EndEffectorActuation.TRIGGER
    assert GripperSpec(kind=GripperKind.NONE).actuation is EndEffectorActuation.TRIGGER
    # a parallel jaw (Robotiq) defaults to joint-position actuation
    assert GripperSpec(kind=GripperKind.PARALLEL).actuation is EndEffectorActuation.JOINT_POSITION


def test_explicit_actuation_is_kept():
    # a parallel gripper the user configured as a trigger (e.g. a custom on/off jaw)
    g = GripperSpec(kind=GripperKind.PARALLEL, actuation=EndEffectorActuation.TRIGGER)
    assert g.actuation is EndEffectorActuation.TRIGGER


# --- sim grasp adapter, per backend ------------------------------------------

def test_grasp_adapter_defaults_per_backend():
    g = GripperSpec(kind=GripperKind.PARALLEL)
    assert g.grasp_adapter_for(Backend.ISAAC) is SimGraspAdapter.SURFACE_GRIPPER
    assert g.grasp_adapter_for(Backend.GAZEBO) is SimGraspAdapter.LINK_ATTACHER
    assert g.grasp_adapter_for(Backend.REAL) is SimGraspAdapter.NONE
    assert g.grasp_adapter_for(Backend.MOCK) is SimGraspAdapter.NONE
    assert g.grasp_adapter_for('gazebo') is SimGraspAdapter.LINK_ATTACHER   # token accepted


def test_grasp_adapter_explicit_override():
    g = GripperSpec(kind=GripperKind.PARALLEL,
                    sim_grasp_adapter={'gazebo': SimGraspAdapter.NONE})
    assert g.grasp_adapter_for(Backend.GAZEBO) is SimGraspAdapter.NONE     # overridden
    assert g.grasp_adapter_for(Backend.ISAAC) is SimGraspAdapter.SURFACE_GRIPPER  # default kept


# --- joint targets: SRDF state vs explicit angle ------------------------------

def test_joint_targets_srdf_vs_angle():
    srdf = GripperSpec(kind=GripperKind.PARALLEL, joint_target=GripperJointTarget.SRDF_STATE,
                       open_state='open', closed_state='closed')
    assert srdf.joint_targets() == ('open', 'closed')
    ang = GripperSpec(kind=GripperKind.PARALLEL, joint_target=GripperJointTarget.ANGLE,
                      open_angle=0.0, closed_angle=0.7691)
    assert ang.joint_targets() == (0.0, 0.7691)


# --- YAML round-trip (idempotent) ---------------------------------------------

def test_round_trip_idempotent():
    g = GripperSpec(kind=GripperKind.PARALLEL, command_joint='robotiq_85_left_knuckle_joint',
                    controller_name='gripper_position_controller', action_ns='gripper_cmd',
                    joint_target=GripperJointTarget.SRDF_STATE, open_state='open',
                    closed_state='closed', sim_grasp_adapter={'gazebo': SimGraspAdapter.LINK_ATTACHER})
    d = json.loads(g.json())   # v1/v2-overlap for model_dump(mode='json')
    assert d['actuation'] == 'joint_position'        # scalar tokens, not enum reprs
    assert d['sim_grasp_adapter'] == {'gazebo': 'link_attacher'}
    g2 = GripperSpec(**d)
    assert g2 == g                                   # re-validates equal (idempotent)
