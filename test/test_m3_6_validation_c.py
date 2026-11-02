"""vNext M3.6 / M3-D (ADR-0026): Validation C's generated files are the BYTE COPIES TMR runs.

TMR's ``test_m3_6_validation_c`` ticks the Validation C trees with the real nodes against its fake
move_group and fake end effector, with two synthetic producers, and checks the captured PreGrasp,
Approach and Retract goals against the designed values. It reads byte copies of what TSA generates
(``trainit_motion_runtime/test/data/m3_6/validation_c/``); this test regenerates both projects in a
temporary directory and proves the copies kept here (``test/data/m3_6/m3d/``) are exactly that output,
and pins the same sha256 digests TMR pins. A change here is a TSA generation change: regenerate, copy
to both repositories, re-pin both sides, and say why (TMR ``test/data/m3_6/README.md``).
"""

import hashlib
from pathlib import Path

import pytest

from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.model import load_project

DATA = Path(__file__).parent / 'data' / 'm3_6'
COPIES = DATA / 'm3d'
PINNED = {   # the digests TMR's test_m3_6_validation_c.cpp pins too
    'candidates.xml': '0c954d63dc62f227fc7850d8d65ad3a3c482b18f254b553effb2cca419b4a757',
    'detections.xml': 'bfa1d674d5025801c040c728767135480ae002450c0ec71d444100f83a715a9d',
    'bt_params.yaml': 'e90956740ce6fff21ba79c81c47a89b3ace8e0817b9c0ba71a70bc8134b2ed9e',
    'motion_profiles.yaml': '507b77a7feca6be0dd0b25a512c20ac480e9445b0a8b04d0e01c4db30b5bc587',
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_the_copies_are_the_pinned_ones():
    for name, digest in PINNED.items():
        assert _sha(COPIES / name) == digest, name


@pytest.fixture(scope='module')
def generated(tmp_path_factory):
    import test_example_ur5e_suction as S
    S.needs_ur_description()            # the UR5e sample cell (vNext M3.7: skip, naming it, without it)
    out = tmp_path_factory.mktemp('m3d')
    import os
    old = os.getcwd()
    os.chdir(DATA)                                       # the urdf path is relative to it
    try:
        for project, which in (('validation_c.yaml', 'candidates'), ('validation_c_detections.yaml', 'detections')):
            Orchestrator().generate(load_project(project), out / which)
    finally:
        os.chdir(old)
    return out


def test_tsa_generates_exactly_the_copies_tmr_runs(generated):
    app = 'm3d_app'
    pairs = {
        'candidates.xml': generated / 'candidates' / app / 'bt_trees' / 'task_program.xml',
        'detections.xml': generated / 'detections' / app / 'bt_trees' / 'task_program.xml',
        'bt_params.yaml': generated / 'candidates' / app / 'config' / 'bt_params.yaml',
        'motion_profiles.yaml': generated / 'candidates' / app / 'config' / 'motion_profiles.yaml',
    }
    for name, path in pairs.items():
        assert path.read_bytes() == (COPIES / name).read_bytes(), name
    # the two producers share every parameter: only the detect step differs
    for name in ('bt_params.yaml', 'motion_profiles.yaml'):
        sub = Path('config') / name
        assert (generated / 'candidates' / app / sub).read_bytes() == (generated / 'detections' / app / sub).read_bytes()
