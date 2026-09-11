"""Byte-stability gate for the backend refactor (ADR-0008 F1: mode -> Backend).

The backend refactor promotes the free-string ``mode`` into a first-class ``Backend``
enum + ``BackendProfile``. It MUST NOT change one byte of the generated mock/isaac/real
output. This test pins that, at byte granularity, on the mode-dependent artifacts:

- GATE A (base-config path): regenerate the checked-in golden project and assert the
  cell bringup, app bringup and trainit_bt launch files are byte-identical to the
  published golden bundle (the same 3 files whose context is built from DeploymentSpec).
- GATE B (from-scratch path): render the bootstrap bringup + the ros2_control/urdf
  xacros and assert byte-identity against committed snapshots under
  ``test/golden/backend_bootstrap/`` (captured pre-refactor).

Complements ``test_generator_golden.py`` (which is deliberately *semantic* / whitespace-
tolerant): this one is *byte* strict, because the refactor's risk is a serialization
change (``repr(tuple(dep.modes))`` and friends) that a semantic diff would forgive.

Fast and ROS-free: generation is pure. To (re)capture GATE B fixtures after an
*intentional* change, run this file with ``--record``.
"""

import hashlib
import os
import sys
import tempfile
from pathlib import Path

import pytest

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.determinism import build_jinja_env
from trainit_setup_assistant.generator.emitters.app_pkg import AppEmitter
from trainit_setup_assistant.generator.emitters.base import GenContext
from trainit_setup_assistant.generator.emitters.moveit_config_pkg import MoveitConfigEmitter
from trainit_setup_assistant.generator.manifest import GenerationManifest
from trainit_setup_assistant.model import (
    AppType, BundleSpec, CanonicalProject, MotionSegment, ToolAction, Waypoint,
    load_project)
from trainit_setup_assistant.model.enums import ToolActionKind, WaypointType
from trainit_setup_assistant.model.robot import (
    ArmControllerSpec, DescriptionSource, GripperSpec, PlanningGroupSpec, RobotSpec)

HERE = os.path.dirname(__file__)
GOLDEN_BUNDLE = Path('/home/fra/fr5_ws/src/fr3wml_digital_twin/'
                     'fr3wml_suction_camera_tsa_4_1_bundle')
GOLDEN_PROJECT = GOLDEN_BUNDLE / 'project.yaml'
G_DESC = 'fr3wml_suction_camera_tsa_4_1_description'
G_CFG = 'fr3wml_suction_camera_tsa_4_1_trainit_config'
G_APP = 'fr3wml_suction_camera_tsa_4_1_app'

# The mode-dependent launch files whose context is built from DeploymentSpec.
GATE_A_FILES = [
    (G_CFG, 'launch/bringup.launch.py'),   # scene_loader_bringup.launch.py.j2
    (G_APP, 'launch/bringup.launch.py'),   # app/bringup.launch.py.j2
    (G_APP, 'launch/trainit_bt.launch.py'),  # app/trainit_bt.launch.py.j2
]

BOOTSTRAP_GOLDEN = Path(HERE) / 'golden' / 'backend_bootstrap'
# (rendered rel-path under the temp out dir  ->  fixture filename)
GATE_B_FILES = {
    'cell_app/launch/bringup.launch.py': 'app_bringup_bootstrap.launch.py',
    'cell_trainit_config/config/cell.ros2_control.xacro': 'cell.ros2_control.xacro',
    'cell_trainit_config/config/cell.urdf.xacro': 'cell.urdf.xacro',
}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fromscratch_project() -> CanonicalProject:
    """A minimal project with NO base_moveit_config_path -> the from-scratch/bootstrap
    path (renders bringup_bootstrap + the ros2_control/urdf xacros)."""
    wps = [
        Waypoint(name='home', type=WaypointType.JOINT, named='home'),
        Waypoint(name='pick', type=WaypointType.TCP,
                 position=[0.56, -0.02, 0.03], orientation=[-0.707, 0.707, 0.0, 0.0]),
        Waypoint(name='pre_place', type=WaypointType.TCP,
                 position=[0.43, -0.28, 0.25], orientation=[-0.006, -0.70, 0.71, 0.01]),
    ]
    p = CanonicalProject(
        project_name='cell',
        bundle=BundleSpec.from_prefix('cell'),
        robot=RobotSpec(
            robot_name='cell',
            description=DescriptionSource(urdf_dir='/tmp', top_xacro='x.urdf.xacro'),
            base_frame='base_link', tip_link='tcp',
            planning_group=PlanningGroupSpec(name='arm', base_link='base_link',
                                             tip_link='tcp', joints=['j1', 'j2']),
            gripper=GripperSpec(),
            arm_controller=ArmControllerSpec(),
        ),
    )
    p.application.type = AppType.PICK_AND_PLACE
    p.application.waypoints = wps
    p.application.segments = [MotionSegment(to_waypoint=w.name) for w in wps]
    p.application.tool_actions = [
        ToolAction(at_waypoint='pick', kind=ToolActionKind.GRASP),
        ToolAction(at_waypoint='pre_place', kind=ToolActionKind.RELEASE)]
    p.application.sequence = ['home', 'pick', 'pre_place']
    return p


def _render_fromscratch(out: Path) -> None:
    """Drive the from-scratch emitters into ``out`` (moveit_config xacros + app launch)."""
    p = _fromscratch_project()
    assert p.robot.base_moveit_config_path is None, 'fixture must take the bootstrap path'
    for Em in (MoveitConfigEmitter, AppEmitter):
        ctx = GenContext(out, build_jinja_env(), GenerationManifest('bytegate', 'cell'))
        Em().emit(p, ctx)


# --- GATE A: base-config path == published golden bundle (byte) -----------------

@pytest.mark.skipif(not GOLDEN_PROJECT.is_file(),
                    reason='golden bundle not present')
def test_base_config_launch_files_byte_identical_to_golden():
    project = load_project(str(GOLDEN_PROJECT))
    project.bundle = BundleSpec(description_package=G_DESC, moveit_config_package=G_CFG,
                                app_package=G_APP)
    with tempfile.TemporaryDirectory() as tmp:
        Orchestrator().generate(project, tmp)
        for pkg, rel in GATE_A_FILES:
            gen = Path(tmp) / pkg / rel
            gold = GOLDEN_BUNDLE / pkg / rel
            assert gen.is_file(), f'generator did not emit {pkg}/{rel}'
            assert gold.is_file(), f'golden missing {pkg}/{rel}'
            gb, xb = gen.read_bytes(), gold.read_bytes()
            assert gb == xb, (
                f'{pkg}/{rel} changed vs golden: '
                f'gen sha={_sha(gb)[:16]} ({len(gb)}B) != golden sha={_sha(xb)[:16]} '
                f'({len(xb)}B). The backend refactor must not alter mock/isaac/real output.')


# --- GATE B: from-scratch path == committed pre-refactor snapshot (byte) --------

def test_from_scratch_launch_and_xacros_byte_identical_to_snapshot():
    assert BOOTSTRAP_GOLDEN.is_dir(), (
        f'missing GATE B fixtures at {BOOTSTRAP_GOLDEN} — run this file with --record')
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _render_fromscratch(out)
        for rel, fixture in GATE_B_FILES.items():
            gen = out / rel
            fix = BOOTSTRAP_GOLDEN / fixture
            assert gen.is_file(), f'from-scratch render did not emit {rel}'
            assert fix.is_file(), f'missing fixture {fixture} — run with --record'
            gb, xb = gen.read_bytes(), fix.read_bytes()
            assert gb == xb, (
                f'{rel} changed vs snapshot: gen sha={_sha(gb)[:16]} ({len(gb)}B) '
                f'!= snapshot sha={_sha(xb)[:16]} ({len(xb)}B). The backend refactor '
                f'must not alter the from-scratch mock/isaac/real output.')


def _record() -> None:
    """Capture the GATE B fixtures from the CURRENT generator output."""
    BOOTSTRAP_GOLDEN.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp)
        _render_fromscratch(out)
        for rel, fixture in GATE_B_FILES.items():
            src = out / rel
            dst = BOOTSTRAP_GOLDEN / fixture
            dst.write_bytes(src.read_bytes())
            print(f'recorded {fixture}  sha={_sha(src.read_bytes())[:16]}  '
                  f'({src.stat().st_size}B)')


if __name__ == '__main__':
    if '--record' in sys.argv:
        _record()
    else:
        raise SystemExit(pytest.main([__file__, '-v']))
