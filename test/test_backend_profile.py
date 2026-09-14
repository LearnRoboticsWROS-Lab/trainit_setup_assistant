"""Backend / BackendProfile — the first-class backend abstraction (ADR-0008 F1).

Pins the abstraction that replaces the free-string ``mode``: the frozen token set, the
reserved GAZEBO member, the per-backend facts (which must MATCH what the launch templates
+ ros2_control xacro encode today), and that DeploymentSpec coerces/validates/round-trips
Backend transparently. ROS-free.
"""

import tempfile
from pathlib import Path

import pytest
from pydantic import ValidationError

from trainit_setup_assistant.model import (
    Backend, DeploymentSpec, BackendProfile, profile_for)
from trainit_setup_assistant.model.backend import (
    CM_EXTERNAL_PLUGIN, CM_NONE, CM_STANDALONE)
from trainit_setup_assistant.model.enums import IMPLEMENTED_BACKENDS


# --- the enum: frozen tokens + reserved gazebo --------------------------------

def test_backend_tokens_are_frozen():
    assert [b.value for b in Backend] == ['mock', 'isaac', 'gazebo', 'real']
    # StrEnum: a member IS its token (comparisons/joins used across the code rely on it)
    assert Backend.ISAAC == 'isaac'
    assert ' | '.join(Backend) == 'mock | isaac | gazebo | real'


def test_only_mock_isaac_real_are_implemented():
    assert IMPLEMENTED_BACKENDS == (Backend.MOCK, Backend.ISAAC, Backend.REAL)
    assert Backend.GAZEBO not in IMPLEMENTED_BACKENDS


# --- BackendProfile: facts must match today's template/xacro behaviour --------

def test_profiles_match_current_generated_behaviour():
    # isaac: sim clock + TopicBasedSystem + a standalone ros2_control_node (mock/isaac path)
    isaac = profile_for(Backend.ISAAC)
    assert isaac.use_sim_time is True
    assert isaac.hardware_plugin == 'topic_based_ros2_control/TopicBasedSystem'
    assert isaac.controller_manager == CM_STANDALONE
    assert isaac.implemented is True

    # mock: wall clock + GenericSystem + standalone CM
    mock = profile_for(Backend.MOCK)
    assert mock.use_sim_time is False
    assert mock.hardware_plugin == 'mock_components/GenericSystem'
    assert mock.controller_manager == CM_STANDALONE

    # real: wall clock + no ros2_control node (vendor bridges impersonate it)
    real = profile_for(Backend.REAL)
    assert real.use_sim_time is False
    assert real.controller_manager == CM_NONE
    assert real.implemented is True


def test_gazebo_profile_is_reserved_but_declared():
    gz = profile_for(Backend.GAZEBO)
    assert isinstance(gz, BackendProfile)
    assert gz.implemented is False              # reserved: wired in TSA v6 (F2)
    assert gz.use_sim_time is True              # sim clock, like isaac
    assert gz.controller_manager == CM_EXTERNAL_PLUGIN   # the in-gzserver plugin owns the CM
    assert gz.hardware_plugin == 'gazebo_ros2_control/GazeboSystem'


def test_profile_for_accepts_token_and_rejects_unknown():
    assert profile_for('isaac') is profile_for(Backend.ISAAC)
    with pytest.raises(ValueError):
        profile_for('webots')


# --- DeploymentSpec: transparent coercion / validation / round-trip -----------

def test_deployment_defaults_unchanged():
    dep = DeploymentSpec()
    assert dep.modes == [Backend.MOCK, Backend.ISAAC, Backend.REAL]
    assert dep.default_mode == Backend.ISAAC


def test_deployment_coerces_string_inputs():
    dep = DeploymentSpec(modes=['mock', 'isaac', 'real'], default_mode='isaac')
    assert all(isinstance(m, Backend) for m in dep.modes)
    assert isinstance(dep.default_mode, Backend)


def test_deployment_rejects_unknown_backend():
    with pytest.raises(ValidationError):
        DeploymentSpec(modes=['mock', 'webots'])


def test_validate_assignment_coerces_raw_string():
    # controller.set_mode() assigns a raw string; validate_assignment must coerce it so
    # `.value` is safe downstream in the emitters.
    dep = DeploymentSpec()
    dep.default_mode = 'mock'
    assert dep.default_mode is Backend.MOCK
    assert dep.default_mode.value == 'mock'


def test_deployment_round_trips_to_scalars(tmp_path):
    from trainit_setup_assistant.model.io import project_to_dict
    from trainit_setup_assistant.model import (
        BundleSpec, CanonicalProject, DescriptionSource, PlanningGroupSpec,
        ArmControllerSpec, GripperSpec, RobotSpec, load_project, save_project)
    p = CanonicalProject(
        project_name='cell', bundle=BundleSpec.from_prefix('cell'),
        robot=RobotSpec(robot_name='cell',
                        description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
                        base_frame='base_link', tip_link='tcp',
                        planning_group=PlanningGroupSpec(name='arm', base_link='base_link',
                                                         tip_link='tcp', joints=['j1']),
                        gripper=GripperSpec(), arm_controller=ArmControllerSpec()))
    # serialises to plain scalars (StrEnum), not enum reprs
    d = project_to_dict(p)
    assert d['deployment']['modes'] == ['mock', 'isaac', 'real']
    assert d['deployment']['default_mode'] == 'isaac'
    # full file round-trip preserves Backend typing
    f = Path(tmp_path) / 'project.yaml'
    save_project(p, f)
    assert 'default_mode: isaac' in f.read_text()      # scalar in the YAML, no enum repr
    q = load_project(f)
    assert q.deployment.default_mode is Backend.ISAAC
    assert q.deployment.modes == [Backend.MOCK, Backend.ISAAC, Backend.REAL]
