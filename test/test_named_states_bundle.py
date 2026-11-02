"""The named states a task program may name are the ones the bundle's SRDF will hold (ROS2ML-SYNC D1).

On a project built on an existing MoveIt config, the generator appends the project's own named states the
config's SRDF lacks and keeps the SRDF's own where both name one
(``scene_loader_moveit_config_pkg._rewrite_srdf``). The task program's name check (R-M3-05) and the values
it resolves (``named_state_values``: the live check, a pose's configuration) read that same set. Found on the
ROS2ML course cell: its 'ready', added in TSA, was refused although the course bundle's SRDF has it.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest
import yaml

from trainit_setup_assistant.applications.task_program import (ProgramRefusals, named_state_values,
                                                                 plan_program)
from trainit_setup_assistant.model import CanonicalProject
from trainit_setup_assistant.model.robot import NamedState

PKG = Path(__file__).resolve().parents[1]
EXAMPLE = PKG / 'examples' / 'ur5e_suction'
GROUP = 'ur_manipulator'
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint', 'wrist_1_joint', 'wrist_2_joint',
          'wrist_3_joint']


@pytest.fixture(autouse=True)
def _in_the_example(monkeypatch):
    monkeypatch.chdir(EXAMPLE)          # the sample's paths are relative to its directory


def _values(*q):
    return dict(zip(JOINTS, q))


def _base_config(root: Path) -> Path:
    """A MoveIt config whose SRDF has 'home' and 'up' for the arm and 'open' for a gripper group."""
    cfg = root / 'base' / 'config'
    cfg.mkdir(parents=True)

    def state(name, group, values):
        joints = ''.join(f'<joint name="{j}" value="{v}"/>' for j, v in values.items())
        return f'<group_state name="{name}" group="{group}">{joints}</group_state>'
    (cfg / 'cell.srdf').write_text(
        '<?xml version="1.0"?>\n<robot name="cell">\n'
        f'  <group name="{GROUP}"><chain base_link="base_link" tip_link="tcp"/></group>\n'
        '  <group name="gripper"><joint name="finger_joint"/></group>\n'
        f'  {state("home", GROUP, _values(0, -1.5708, 1.5708, -1.5708, -1.5708, 0))}\n'
        f'  {state("up", GROUP, _values(0, -1.5708, 0, -1.5708, 0, 0))}\n'
        f'  {state("open", "gripper", {"finger_joint": 0.0})}\n'
        '</robot>\n')
    (cfg / 'pilz_cartesian_limits.yaml').write_text(
        'cartesian_limits:\n  max_trans_vel: 1.0\n  max_trans_acc: 2.25\n  max_trans_dec: -5.0\n'
        '  max_rot_vel: 1.57\n')
    return root / 'base'


def _project(tmp_path, named='ready') -> CanonicalProject:
    """The sample task program on that MoveIt config; the project adds 'ready' and gives 'home' other
    values; the first joint waypoint names ``named``."""
    data = yaml.safe_load((EXAMPLE / 'task_program.yaml').read_text())
    for w in data['application']['waypoints']:
        w.pop('frame', None)            # poses in the base frame: no robot description needed to plan
    project = CanonicalProject.parse_obj(data)
    project.robot.base_moveit_config_path = str(_base_config(tmp_path))
    project.robot.named_states = [
        NamedState(name='home', group=GROUP, joint_values=_values(0.3, -1.2, 1.2, -1.5, -1.5, 0)),
        NamedState(name='ready', group=GROUP, joint_values=_values(0.1, -1.4, 1.4, -1.5, -1.5, 0)),
        NamedState(name='closed', group='gripper', joint_values={'finger_joint': 0.7})]
    joint = next(w for w in project.application.program.waypoints if w.named is not None)
    joint.named = named
    return project


def test_a_state_the_project_adds_is_one_a_waypoint_may_name(tmp_path):
    plan_program(_project(tmp_path))                 # no refusal at all


@pytest.mark.parametrize('group', ['', 'gripper'])
def test_a_state_of_no_group_or_another_group_is_not_the_arms(tmp_path, group):
    """Both generators write a state with the group it names (``group=""`` stays no group's): a 'ready'
    that is not the planning group's is not one the bundle's arm has."""
    project = _project(tmp_path)
    project.robot.named_states[1].group = group
    assert 'ready' not in named_state_values(project)
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(project)
    assert "'ready' is not a named state of planning group 'ur_manipulator' (named states: home, up)" in str(
        exc.value)


def test_a_state_neither_holds_is_refused_naming_what_the_bundle_will_have(tmp_path):
    with pytest.raises(ProgramRefusals) as exc:
        plan_program(_project(tmp_path, named='nowhere'))
    assert ("'nowhere' is not a named state of planning group 'ur_manipulator' "
            '(named states: home, ready, up)') in str(exc.value)


def test_the_values_are_the_bundles_the_srdf_s_own_where_both_name_one(tmp_path):
    states = named_state_values(_project(tmp_path))
    assert set(states) == {'home', 'up', 'ready'}                    # the arm's only: no 'open' / 'closed'
    assert states['home'] == pytest.approx(_values(0, -1.5708, 1.5708, -1.5708, -1.5708, 0))
    assert states['ready'] == pytest.approx(_values(0.1, -1.4, 1.4, -1.5, -1.5, 0))


def test_the_check_reads_what_the_generator_writes(tmp_path):
    """The generator's SRDF and ``named_state_values`` agree, state for state (the copy-base emitter run
    on the same project; its other inputs minimal)."""
    from trainit_setup_assistant.generator.determinism import build_jinja_env
    from trainit_setup_assistant.generator.emitters.base import GenContext
    from trainit_setup_assistant.generator.emitters.scene_loader_moveit_config_pkg import (
        SceneLoaderMoveitConfigEmitter)
    from trainit_setup_assistant.generator.manifest import GenerationManifest
    project = _project(tmp_path)
    cfg = Path(project.robot.base_moveit_config_path) / 'config'
    (cfg / 'ur.urdf.xacro').write_text('<?xml version="1.0"?>\n<robot '
                                      'xmlns:xacro="http://www.ros.org/wiki/xacro" name="r"/>\n')
    for f in ('kinematics.yaml', 'joint_limits.yaml', 'ompl_planning.yaml', 'moveit_controllers.yaml',
              'ros2_controllers.yaml'):
        (cfg / f).write_text('{}\n')
    out = tmp_path / 'out'
    SceneLoaderMoveitConfigEmitter().emit(
        project, GenContext(out, build_jinja_env(), GenerationManifest('cell', 'cell')))
    # the merged SRDF is written as <robot_name>.srdf, the one MoveItConfigsBuilder loads (the base's
    # own file is copied beside it, unchanged)
    (srdf,) = list(out.rglob(f'{project.robot.robot_name}.srdf'))
    written = {gs.get('name'): {j.get('name'): float(j.get('value')) for j in gs.findall('joint')}
               for gs in ET.parse(srdf).getroot().findall('group_state') if gs.get('group') == GROUP}
    expected = named_state_values(project)
    assert set(written) == set(expected)
    for name, values in expected.items():
        assert written[name] == pytest.approx(values, abs=1e-5), name
