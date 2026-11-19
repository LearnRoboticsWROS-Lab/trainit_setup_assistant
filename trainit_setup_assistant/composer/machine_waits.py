"""The BLIND-WAIT report of a Canonical Task Program (TSA 8.10.0, P0.b, P0B_DESIGN §7, owner decision 6). Pure: no Qt,
no ROS, no planning.

How does the program wait for a machine? With a SEMANTIC interaction -- a ``request`` on a machine service and a
``wait`` for it (the machine says when it is done) -- or BLIND: an ``external`` signal nobody acknowledges, or a
``dwell`` that guesses how long the machine takes. One line says it::

    machine waits: 5 semantic (request + wait); 0 external; 0 dwell

counted over the WHOLE program (every task, running or not, and every path's events):

- semantic = request steps + wait steps;
- external = process steps of kind ``external`` + path events of kind ``external``;
- dwell    = ``dwell`` steps + path events of kind ``dwell``.

It is a REPORT, never a plan warning (a program with dwells is valid): Validate, Step 8, the Overview, the diagnostics
and ``trainit_generate`` show it, and the gate's static check (BW1) reads this same function.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..model.program import EventKind, TaskProgram, walk_steps


@dataclass(frozen=True)
class MachineWaits:
    """What a program's machine waits are: its requests and waits (semantic), its external signals and its dwells."""

    requests: int
    waits: int
    external: int
    dwell: int

    @property
    def semantic(self) -> int:
        return self.requests + self.waits

    @property
    def blind(self) -> int:
        """The blind waits: every external signal and every dwell."""
        return self.external + self.dwell

    def text(self) -> str:
        return (f'machine waits: {self.semantic} semantic (request + wait); {self.external} external; '
                f'{self.dwell} dwell')


def machine_waits(program: TaskProgram) -> MachineWaits:
    """The blind-wait report of ``program`` (module docstring)."""
    requests = waits = external = dwell = 0
    for task in program.tasks:
        for _label, step in walk_steps(task.steps):
            if step.wait is not None:
                waits += 1
            elif step.dwell is not None:
                dwell += 1
            elif step.process is not None:
                if step.process.kind is EventKind.REQUEST:
                    requests += 1
                elif step.process.kind is EventKind.EXTERNAL:
                    external += 1
    for path in program.paths:
        for event in path.events:
            if event.kind is EventKind.EXTERNAL:
                external += 1
            elif event.kind is EventKind.DWELL:
                dwell += 1
    return MachineWaits(requests, waits, external, dwell)


def machine_waits_line(project) -> str:
    """The report's line for a project ('' for a project whose application is not a task program)."""
    application = getattr(project, 'application', None)
    if application is None or not application.is_task_program():
        return ''
    return machine_waits(application.program).text()


__all__ = ['MachineWaits', 'machine_waits', 'machine_waits_line']
