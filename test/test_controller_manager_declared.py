"""A bundle declares the controller manager its moveit_controllers.yaml names (vNext G0, TSA 7.8.1).

move_group loads the plugin ``moveit_controllers.yaml`` names (``moveit_controller_manager:
<package>/<class>``); without its package it cannot execute a single trajectory. Before 7.8.1 no
generated package.xml declared it, so ``rosdep install`` over a bundle left it out on a machine
that README.md "Install" alone had set up (measured on a fresh osrf/ros:humble-desktop). Since then
the package that ships the file declares the plugin's package:

    from scratch       the template's own moveit_controllers.yaml names
                       moveit_simple_controller_manager, and moveit_config/package.xml.j2
                       declares it
    on a base config   the copied file names whatever the base chose; the scene-loader emitter reads
                       it (moveit_controller_manager_package) and declares it when it is a released
                       MoveIt manager (a rosdep key)

Generation goes into pytest temp dirs only.
"""
import re
from pathlib import Path

import eef_fixture as F
import identity_cases
from trainit_setup_assistant.generator import Orchestrator
from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
    MOVEIT_CONTROLLER_MANAGERS, moveit_controller_manager_package)

DECLARE = re.compile(r'<(?:exec_)?depend[^>]*>([^<]+)</(?:exec_)?depend>')


def _named_and_declared(bundle: Path):
    """(the plugin package each moveit_controllers.yaml names, what the bundle's package.xml
    files declare)."""
    named = {moveit_controller_manager_package(f) for f in bundle.rglob('moveit_controllers.yaml')}
    declared = {d.strip() for p in bundle.rglob('package.xml')
                for d in DECLARE.findall(p.read_text())}
    return named, declared


def test_the_plugin_package_is_read_from_the_file(tmp_path):
    f = tmp_path / 'moveit_controllers.yaml'
    for plugin, want in (
            ('moveit_simple_controller_manager/MoveItSimpleControllerManager',
             'moveit_simple_controller_manager'),
            ('moveit_ros_control_interface/Ros2ControlManager', 'moveit_ros_control_interface'),
            ('my_vendor_manager/VendorManager', None)):            # not a released MoveIt one
        f.write_text(f'moveit_controller_manager: {plugin}\n'
                     'moveit_simple_controller_manager: {}\n')
        assert moveit_controller_manager_package(f) == want, plugin
    f.write_text('controller_names: []\n')                         # no manager named
    assert moveit_controller_manager_package(f) is None
    f.write_text('moveit_controller_manager: [unclosed\n')         # not YAML
    assert moveit_controller_manager_package(f) is None
    assert moveit_controller_manager_package(tmp_path / 'absent.yaml') is None
    assert set(MOVEIT_CONTROLLER_MANAGERS) == {'moveit_simple_controller_manager',
                                               'moveit_ros_control_interface'}


def test_a_bundle_from_scratch_declares_its_controller_manager(tmp_path):
    out = tmp_path / 'bundle'
    Orchestrator().generate(F.suction_simple(tmp_path), str(out))
    named, declared = _named_and_declared(out)
    assert named == {'moveit_simple_controller_manager'}, named
    assert named <= declared, sorted(declared)


def test_a_bundle_on_a_base_config_declares_the_manager_the_copied_file_names(tmp_path):
    identity_cases.require('p1_golden')                  # the reference cell: a base-config bundle
    out = tmp_path / 'bundle'
    Orchestrator().generate(identity_cases.golden(), str(out))
    named, declared = _named_and_declared(out)
    assert named == {'moveit_simple_controller_manager'}, named
    assert named <= declared, sorted(declared)
