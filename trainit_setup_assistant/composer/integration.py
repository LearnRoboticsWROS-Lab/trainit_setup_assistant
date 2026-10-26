"""Project integration checks and the scene-sync status (vNext I0.1, I0_DESIGN.md §4, §6, §7, §8).

These check how the project is INTEGRATED with its authorities, never the application:
- the MoveIt config (robot planning semantics: the project's mirror must not drift from it);
- the deployment world (the planning projection must not be stale);
- the robot bound inside the world.

They are editor warnings and errors with ``R-I0-*`` codes. The canonical program validation
(``plan_program``) is unchanged, and no generated file depends on these checks.

The **scene-sync status** is COMPUTED, never stored (owner decision D2). It states what is live and
what is not. TSA never claims that the deployment world and RViz are synchronized.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..model.enums import SceneObjectCategory
from ..model.project import CanonicalProject

# The accurate I0 statement (owner, 2026-10-21): shown verbatim in the UI and the docs.
STATEMENT = ('The robot state is LIVE. The static planning environment is DERIVED from the deployment '
             'world. Dynamic / actuated world-state synchronization is future work.')

LIVE, OFFLINE = 'LIVE', 'OFFLINE'
CURRENT, SNAPSHOT, STALE, MANUAL = 'CURRENT', 'SNAPSHOT', 'STALE', 'MANUAL'
LIVE_MAX_AGE_S = 2.0          # joint states older than this: the robot is not LIVE


def file_sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


# ---- scene-sync status ------------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectStatus:
    id: str
    category: str               # static | dynamic | actuated
    status: str                 # CURRENT | SNAPSHOT | STALE | MANUAL
    detail: str = ''


@dataclass
class SceneStatus:
    robot: str = OFFLINE         # LIVE | OFFLINE
    robot_detail: str = ''
    world: str = MANUAL          # CURRENT | STALE | MANUAL (the projection's source)
    world_detail: str = ''
    objects: List[ObjectStatus] = field(default_factory=list)

    def of_category(self, *categories: str) -> List[ObjectStatus]:
        return [o for o in self.objects if o.category in categories]

    def static_status(self) -> str:
        """The static scene's status: STALE wins, then MANUAL; CURRENT when every static object is
        derived from a current world; '' when there is no static object."""
        statuses = {o.status for o in self.of_category('static')}
        for s in (STALE, MANUAL, CURRENT):
            if s in statuses:
                return s
        return ''

    def header(self) -> str:
        """``Robot: LIVE · Static scene: CURRENT · Dynamic/actuated: SNAPSHOT (not live-synced)``."""
        parts = [f'Robot: {self.robot}']
        static = self.static_status()
        if static:
            parts.append(f'Static scene: {static}')
        moving = self.of_category('dynamic', 'actuated')
        if moving:
            statuses = {o.status for o in moving}
            # STALE wins; SNAPSHOT when any was imported from a world; MANUAL when all are authored
            shown = STALE if STALE in statuses else SNAPSHOT if SNAPSHOT in statuses else MANUAL
            parts.append(f'Dynamic/actuated: {shown} (not live-synced)')
        return ' · '.join(parts)


def _category(obj) -> str:
    cat = getattr(obj, 'category', None)
    return cat.value if isinstance(cat, SceneObjectCategory) else str(cat or 'static')


def world_status(project: CanonicalProject) -> Tuple[str, str]:
    """``(status, detail)`` of the projection's source world: CURRENT, STALE (changed or missing
    since the import) or MANUAL (no recorded import). ``project`` has its paths resolved."""
    prov = project.scene.provenance
    if prov is None:
        return MANUAL, 'no world import recorded: the scene is authored in the project'
    if not os.path.isfile(prov.file):
        return STALE, f'the imported world file is missing: {prov.file}'
    if file_sha256(prov.file) != prov.sha256:
        return STALE, f'the world changed since the import ({prov.file}): re-import it'
    return CURRENT, f'derived from {prov.file}'


def scene_status(project: CanonicalProject, joint_states_age: Optional[float] = None,
                 live_max_age: float = LIVE_MAX_AGE_S) -> SceneStatus:
    """The scene-sync status (I0_DESIGN.md §8) of a project whose paths are resolved.
    ``joint_states_age`` is the live session's (``LiveCapture.joint_states_age``); None = no live
    session."""
    out = SceneStatus()
    if joint_states_age is not None and joint_states_age <= live_max_age:
        out.robot, out.robot_detail = LIVE, f'joint states {joint_states_age:.1f} s old'
    elif joint_states_age is not None:
        out.robot_detail = f'no joint states for {joint_states_age:.0f} s'
    else:
        out.robot_detail = 'no planning environment running'
    out.world, out.world_detail = world_status(project)
    prov = project.scene.provenance
    for obj in project.scene.objects:
        cat = _category(obj)
        imported = prov is not None and getattr(obj.source, 'value', obj.source) in ('usd', 'world')
        moving = cat in ('dynamic', 'actuated')
        if not imported:
            detail = 'authored in the project'
            if getattr(obj.source, 'value', obj.source) in ('usd', 'world') and prov is None:
                detail = 'imported before TSA recorded the world: re-import it to track the world'
            if moving:
                detail += '; NOT live-synchronized'
            out.objects.append(ObjectStatus(obj.id, cat, MANUAL, detail))
        elif out.world == STALE:
            out.objects.append(ObjectStatus(obj.id, cat, STALE, out.world_detail))
        elif moving:
            out.objects.append(ObjectStatus(
                obj.id, cat, SNAPSHOT, 'the imported initial pose only: NOT live-synchronized (the '
                'deployment may move it, the planning scene will not follow)'))
        else:
            out.objects.append(ObjectStatus(obj.id, cat, CURRENT, out.world_detail))
    return out


# ---- the MoveIt config (robot planning authority) -----------------------------------------------------

def _base_config(project: CanonicalProject) -> Optional[Path]:
    base = project.robot.base_moveit_config_path
    return Path(base) / 'config' if base else None


def moveit_config_drift(project: CanonicalProject) -> List[str]:
    """Where the project's MIRROR of the referenced MoveIt config differs from it (copy-base flow;
    a from-scratch project has no authority to drift from). Warnings, ``R-I0-01``."""
    config = _base_config(project)
    if config is None:
        return []
    fix = ' -- re-read the robot from its MoveIt config (Step 1)'
    srdfs = sorted(config.glob('*.srdf')) if config.is_dir() else []
    if not srdfs:
        return [f'R-I0-01: robot: the MoveIt config {config.parent} has no config/*.srdf' + fix]
    from ..robotmodel.collision_matrix import parse_disable_collisions
    from ..robotmodel.srdf_reader import read_srdf
    text = srdfs[0].read_text()
    info = read_srdf(text)
    robot, out = project.robot, []
    where = f'robot (MoveIt config {srdfs[0].name})'
    group = info.group(robot.planning_group.name)
    if group is None:
        out.append(f"R-I0-01: {where}: no planning group '{robot.planning_group.name}' "
                   f"(groups: {', '.join(g.name for g in info.groups) or 'none'})" + fix)
    else:
        if group.chain and tuple(group.chain) != (robot.planning_group.base_link, robot.planning_group.tip_link):
            out.append(f"R-I0-01: {where}: group '{group.name}' is the chain {group.chain[0]} -> "
                       f'{group.chain[1]}, the project says {robot.planning_group.base_link} -> '
                       f'{robot.planning_group.tip_link}' + fix)
        if group.joints and list(group.joints) != list(robot.planning_group.joints):
            out.append(f"R-I0-01: {where}: group '{group.name}' joints are {', '.join(group.joints)}, "
                       f"the project says {', '.join(robot.planning_group.joints)}" + fix)
        mine = {s.name: s for s in robot.named_states if s.group == group.name}
        for st in info.states_for(group.name):
            ours = mine.get(st.name)
            if ours is None:
                out.append(f"R-I0-01: {where}: named state '{st.name}' is not in the project" + fix)
            elif set(ours.joint_values) != set(st.joint_values) or any(
                    abs(ours.joint_values[j] - v) > 1e-9 for j, v in st.joint_values.items()):
                out.append(f"R-I0-01: {where}: named state '{st.name}' has other joint values in the "
                           'project' + fix)
    theirs = {frozenset((p.link1, p.link2)) for p in parse_disable_collisions(text)}
    ours = {frozenset((d.link1, d.link2)) for d in robot.disable_collisions}
    if theirs != ours:
        out.append(f'R-I0-01: {where}: the self-collision matrix differs ({len(theirs - ours)} pair(s) '
                   f'only in the SRDF, {len(ours - theirs)} only in the project)' + fix)
    ctl = config / 'moveit_controllers.yaml'
    if ctl.is_file():
        import yaml
        mgr = (yaml.safe_load(ctl.read_text()) or {}).get('moveit_simple_controller_manager', {}) or {}
        arms = [n for n in mgr.get('controller_names', []) or []
                if (mgr.get(n) or {}).get('type') == 'FollowJointTrajectory']
        if arms and robot.arm_controller.name not in arms:
            out.append(f"R-I0-01: robot (MoveIt config moveit_controllers.yaml): the arm controller is "
                       f"{', '.join(arms)}, the project says '{robot.arm_controller.name}'" + fix)
    return out


# ---- the Isaac world binding (Case A: the stage contains the robot) ------------------------------------

_STAGES: Dict[Tuple[str, float], object] = {}


def _open_stage(path: str):
    from pxr import Usd
    key = (path, os.path.getmtime(path))
    if key not in _STAGES:
        _STAGES.clear()                          # one stage at a time: they are large
        _STAGES[key] = Usd.Stage.Open(path, load=Usd.Stage.LoadAll)
    return _STAGES[key]


def isaac_binding_problems(project: CanonicalProject) -> Tuple[List[str], List[str]]:
    """``(errors, warnings)`` of the Isaac binding, checked OFFLINE in the stage (pxr): the stage
    exists; the robot prim exists and holds an articulation; its joints include the planning group's;
    the base prim the scene import used lies in it. ``R-I0-02``. A project without Isaac has none."""
    errors: List[str] = []
    warnings: List[str] = []
    dep = project.deployment
    isaac_mode = any(getattr(m, 'value', m) == 'isaac' for m in dep.modes)
    binding = dep.isaac
    if binding is None:
        if isaac_mode:
            warnings.append('R-I0-02: deployment (isaac): no Isaac world is bound -- bind the stage and the '
                            'robot prim (Step 2) so TSA can check the robot in it and track the scene')
        return errors, warnings
    where = 'deployment (isaac binding)'
    if not os.path.isfile(binding.stage):
        errors.append(f'R-I0-02: {where}: the stage is not found: {binding.stage}')
        return errors, warnings
    if not binding.robot_prim:
        warnings.append(f'R-I0-02: {where}: no robot prim: an Isaac world without the robot (the robot '
                        'spawned separately) is not supported yet -- name the robot prim in the stage')
        return errors, warnings
    try:
        from pxr import UsdPhysics
        stage = _open_stage(binding.stage)
    except ImportError:
        warnings.append(f'R-I0-02: {where}: pxr is not available: the robot in the stage cannot be '
                        'checked offline')
        return errors, warnings
    except Exception as exc:  # noqa: BLE001 - a broken stage is reported, not raised
        errors.append(f'R-I0-02: {where}: the stage cannot be opened: {exc}')
        return errors, warnings
    prim = stage.GetPrimAtPath(binding.robot_prim) if stage else None
    if not prim or not prim.IsValid():
        top = [c.GetPath().pathString for c in stage.GetPrimAtPath('/World').GetChildren()] \
            if stage and stage.GetPrimAtPath('/World') else []
        errors.append(f"R-I0-02: {where}: no prim '{binding.robot_prim}' in the stage"
                      + (f" (/World holds: {', '.join(top)})" if top else ''))
        return errors, warnings
    from pxr import Usd
    subtree = list(Usd.PrimRange(prim))
    if not any(p.HasAPI(UsdPhysics.ArticulationRootAPI) for p in subtree):
        errors.append(f"R-I0-02: {where}: '{binding.robot_prim}' holds no articulation root: it is not a "
                      'simulated robot')
    joints = {p.GetName() for p in subtree if p.IsA(UsdPhysics.Joint)}
    missing = [j for j in project.robot.planning_group.joints if j not in joints]
    if missing:
        errors.append(f"R-I0-02: {where}: the robot in the stage has no joint {', '.join(missing)} of "
                      f"planning group '{project.robot.planning_group.name}' (joints found: "
                      f"{', '.join(sorted(joints)) or 'none'}) -- joints are matched by NAME")
    base = f'{binding.robot_prim.rstrip("/")}/{project.robot.base_frame}'
    if not stage.GetPrimAtPath(base):
        warnings.append(f"R-I0-02: {where}: no base prim '{base}' (the robot's base frame in the stage)")
    prov = project.scene.provenance
    if prov is not None and prov.base_prim and not prov.base_prim.startswith(binding.robot_prim.rstrip('/') + '/'):
        warnings.append(f"R-I0-02: {where}: the scene was imported relative to '{prov.base_prim}', not to "
                        f"the bound robot '{binding.robot_prim}' -- re-import the scene")
    return errors, warnings


MOTION_PLANNING_DISPLAY = 'moveit_rviz_plugin/MotionPlanning'


def rviz_teaching_problems(project: CanonicalProject) -> List[str]:
    """Teaching happens in RViz's MotionPlanning display (I0_DESIGN.md §12): the copy-base flow ships
    the MoveIt config's own RViz file, which must carry it (a from-scratch config always does, from
    TSA's template). ``R-I0-04``."""
    config = _base_config(project)
    if config is None or not config.is_dir():
        return []
    rviz = sorted(config.glob('*.rviz'))
    if not rviz:
        return [f'R-I0-04: robot (MoveIt config {config.parent.name}): no config/*.rviz: the planning '
                'environment opens RViz without the MotionPlanning display -- add one with MoveIt Setup '
                'Assistant']
    if not any(MOTION_PLANNING_DISPLAY in f.read_text(errors='replace') for f in rviz):
        return [f"R-I0-04: robot (MoveIt config {config.parent.name}): {', '.join(f.name for f in rviz)} "
                f'has no MotionPlanning display: teaching needs its interactive marker and Plan / Execute '
                '-- add the display in RViz and save the config']
    return []


def integration_problems(project: CanonicalProject, check_stage: bool = True) -> Tuple[List[str], List[str]]:
    """Every integration check of a project with its paths resolved: ``(errors, warnings)``."""
    errors, warnings = (isaac_binding_problems(project) if check_stage else ([], []))
    warnings = list(moveit_config_drift(project)) + rviz_teaching_problems(project) + list(warnings)
    status, detail = world_status(project)
    if status == STALE:
        warnings.append(f'R-I0-03: scene: {detail} (Step 2: Re-import world)')
    return list(errors), warnings


__all__ = ['CURRENT', 'LIVE', 'LIVE_MAX_AGE_S', 'MANUAL', 'OFFLINE', 'ObjectStatus', 'SNAPSHOT', 'STALE',
           'STATEMENT', 'SceneStatus', 'file_sha256', 'integration_problems', 'isaac_binding_problems',
           'moveit_config_drift', 'rviz_teaching_problems', 'scene_status', 'world_status']
