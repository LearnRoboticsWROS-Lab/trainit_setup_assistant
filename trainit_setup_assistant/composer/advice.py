"""Composer advice (vNext I0, owner Test C): programs the canonical validator accepts that may still not
repeat at run time. They are editor WARNINGS with ``R-I0-*`` codes; ``plan_program`` is unchanged and no
generated file depends on them.

``R-I0-05`` -- a PTP or free move to a TCP pose, followed by a path or a LIN / CIRC move. The pose does not
say which joint configuration reaches it, so IK chooses one when the move runs, and it can change between
cycles: the same PTP home -> w1 moved 4.4 rad of joints in one cycle and 13.4 rad in the next, and the
blended path after it found no IK solution (``pilz_sequence: FAILED``). The Cartesian motion that follows is
planned from that configuration. A PTP to a JOINT waypoint taught at the same pose (Step 7: Capture TCP +
joints) reaches it the same way every cycle. A configuration-consistent PTP to a pose is runtime work
(KNOWN_DEBT).

``R-I0-06`` -- a grasp, and no object of the cell scene (when it has one) is a grasp target (Step 2). The grasp commands the end
effector, but nothing attaches to the tool in MoveIt's planning scene: RViz does not show the part held and
the planner ignores it (owner Test F.1, friction row 38: "the cube was not grasped").
"""

from __future__ import annotations

from typing import List

from ..model.enums import MotionType, WaypointType
from ..model.program import walk_steps
from ..model.project import CanonicalProject

_CARTESIAN = (MotionType.LIN, MotionType.CIRC)


def configuration_advice(project: CanonicalProject) -> List[str]:
    """The ``R-I0-05`` and ``R-I0-06`` warnings of a task program (none for another application)."""
    app = project.application
    if not app.is_task_program():
        return []
    program = app.program
    # ADR-0028: a pose that names its configuration is reached the same way every cycle -- not flagged
    poses = {w.name for w in program.waypoints if w.type != WaypointType.JOINT and not w.configuration}
    out: List[str] = []

    def what(step) -> str:
        if step.path is not None:
            return f"path '{step.path}'"
        if step.move is not None and step.move.type in _CARTESIAN:
            return f"a {step.move.type.value.upper()} move"
        return ''

    def walk(task: str, steps, label: str) -> None:
        # labels as model.program.walk_steps gives them: 3.2 in a repeat / retry, 3.2.1 in a fallback
        for i, step in enumerate(steps, start=1):
            here = f'{label}.{i}' if label else str(i)
            move, after = step.move, (what(steps[i]) if i < len(steps) else '')
            if move is not None and move.type in (MotionType.PTP, MotionType.FREE) and move.to in poses and after:
                out.append(f"R-I0-05: task '{task}' step {here}: {move.type.value.upper()} to the TCP pose "
                           f"'{move.to}' before {after}: IK chooses the joint configuration when it runs and it "
                           'can change between cycles; give the pose its configuration (Step 7: Capture TCP + joints '
                           'records it), or move to a JOINT waypoint taught at that pose')
            if step.fallback is not None:
                for b, body in enumerate(step.fallback.bodies, start=1):
                    walk(task, body, f'{here}.{b}')
            else:
                for body in step.bodies():
                    walk(task, body, here)

    for task in program.tasks:
        walk(task.name, task.steps, '')
    if project.scene.objects and not project.scene.grasp_target_ids():
        first = next(((task.name, label) for task in program.tasks
                      for label, step in walk_steps(task.steps) if step.grasp is not None), None)
        if first is not None:
            out.append(f"R-I0-06: task '{first[0]}' step {first[1]}: a grasp, and no object of the cell scene is a "
                       "grasp target: nothing attaches to the tool in the planning scene (RViz does not show the "
                       "part held, the planner ignores it) -- tick 'grasp' on the part you pick (Step 2)")
    return out


__all__ = ['configuration_advice']
