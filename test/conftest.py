"""Test-session plumbing: the reference projects were recorded on another machine.

The golden's project.yaml (frozen with the golden) and examples/fr3wml_project.yaml record the
absolute description and base-config paths of the machine that made them, and the generator
needs those directories to exist. Every project these tests LOAD is therefore rebased into this
workspace (``ws_paths.in_workspace``): a path that exists is left exactly as it is -- on the
machine that recorded it nothing changes -- and one that does not is moved onto this checkout's
``src/``. Product code is untouched; only what the tests load is rebased (vNext M0D).

A Qt test's parentless widgets are collected when the test ends (vNext I0): left to a later cyclic
collection, a Composer panel was freed inside a ROS message import of another test and crashed the
interpreter (a pydantic-1 run, 2026-10-26).
"""
import gc

import pytest

import trainit_setup_assistant.model as _model
import trainit_setup_assistant.model.io as _io

import ws_paths

_load_project = _io.load_project


def _load_project_in_this_workspace(path):
    p = _load_project(path)
    desc = p.robot.description
    desc.urdf_dir = ws_paths.in_workspace(desc.urdf_dir)
    if desc.meshes_dir:
        desc.meshes_dir = ws_paths.in_workspace(desc.meshes_dir)
    if p.robot.base_moveit_config_path:
        p.robot.base_moveit_config_path = ws_paths.in_workspace(p.robot.base_moveit_config_path)
    for policy in (p.application.policies if p.application else []):
        policy.card = ws_paths.in_workspace(policy.card)   # a missing card is only a warning
    return p


_io.load_project = _model.load_project = _load_project_in_this_workspace


@pytest.fixture(autouse=True)
def _collect_qt_widgets(request):
    yield
    if 'qapp' in request.fixturenames:
        gc.collect()
