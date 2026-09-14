"""Deployment schema: the bring-up topology (controllers + cell bridges).

The bring-up is cell-specific (suction bridges, joint-state mergers, a real-robot
bridge). Modelling it as DATA keeps the generator generic: any robot's bridges are a
list of nodes, and the FR3WML golden is reproduced by carrying its bridges in the
project. Defaults are empty (a minimal mock/isaac bring-up still works).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from .enums import Backend


class LaunchNodeSpec(BaseModel):
    """A bridge/helper node added to bring-up in the given modes."""

    package: str
    executable: str
    name: Optional[str] = None
    # each dict becomes one entry in the node's `parameters=[...]` list
    parameters: List[Dict[str, Any]] = Field(default_factory=list)
    remappings: List[List[str]] = Field(default_factory=list)  # [[from, to], ...]
    # backend tokens this node runs in; kept as plain strings (matched at runtime against
    # the launch `mode:=` string) so `.model_dump()` serialises byte-identically.
    modes: List[str] = Field(default_factory=lambda: ['mock', 'isaac'])


class RealIncludeSpec(BaseModel):
    """A launch file included when mode == 'real' (the hardware bridge)."""

    package: str
    launch_file: str


class DeploymentSpec(BaseModel):
    # validate_assignment: coerce a raw-string assignment (e.g. controller.set_mode)
    # back into a Backend member, so `default_mode` is always a Backend downstream.
    model_config = ConfigDict(validate_assignment=True)

    # The execution backends this cell supports (ADR-0008). A first-class Backend enum
    # instead of a free string, so an unknown backend token is rejected at load. Serialises
    # to plain scalars in project.yaml (StrEnum) and to the same VALID_MODES tuple in the
    # generated bring-up. GAZEBO is reserved (wired in TSA v6 / F2).
    modes: List[Backend] = Field(
        default_factory=lambda: [Backend.MOCK, Backend.ISAAC, Backend.REAL])
    default_mode: Backend = Backend.ISAAC
    # Remap the arm ros2_control node's /joint_states ONLY when a merger republishes it
    # to /joint_states (FR3WML suction pattern). Default None: with no merger bridge the
    # broadcaster must publish /joint_states directly, else there is no TF and planning
    # has no start state. Set this (e.g. '/joint_states_robot') together with a merger.
    arm_joint_states_remap_to: Optional[str] = None
    # Extra cell bridges (suction bridge, joint_state_merger, gripper action server...).
    bridges: List[LaunchNodeSpec] = Field(default_factory=list)
    # Hardware bringup include for mode == 'real'.
    real_include: Optional[RealIncludeSpec] = None

    def bridge_packages(self) -> List[str]:
        """Unique bridge packages (for package.xml exec_depend), in first-seen order."""
        seen: List[str] = []
        for node in self.bridges:
            if node.package not in seen:
                seen.append(node.package)
        if self.real_include and self.real_include.package not in seen:
            seen.append(self.real_include.package)
        return seen
