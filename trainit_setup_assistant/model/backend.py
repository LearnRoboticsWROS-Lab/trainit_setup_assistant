"""Backend profiles — the bring-up facts each execution backend implies (ADR-0008).

The backend used to be a bare ``mode`` string branched by literal comparison
(``mode == "isaac"``) scattered across the launch templates + the ros2_control xacro.
``BackendProfile`` promotes those facts to a single, first-class source of truth: for a
given :class:`~trainit_setup_assistant.model.enums.Backend`, what ``use_sim_time`` to
use, WHO owns the ``controller_manager``, and which ros2_control hardware plugin the
xacro must select.

**F1 (this file) is additive and behaviour-preserving:** the generated mock/isaac/real
launch topology is UNCHANGED — the templates still branch as before. The profiles simply
capture, in one place, the facts those branches already encode, so that TSA v6 (F2) can
drive the new Gazebo branch (and generalise the TMR ``use_isaac`` seam + the xacro plugin
selector) from data instead of adding more scattered literals. See
``docs/integration/ROS2ML_INTEGRATION_ASSESSMENT.md`` §5 and ADR-0008.
"""

from __future__ import annotations

from dataclasses import dataclass

from .enums import Backend

# Who provides/owns the ros2_control controller_manager for a backend.
CM_STANDALONE = 'standalone'            # a controller_manager/ros2_control_node WE launch
CM_EXTERNAL_PLUGIN = 'external_plugin'  # provided in-process by the sim plugin (Gazebo)
CM_NONE = 'none'                        # none: vendor bridges impersonate the controllers


@dataclass(frozen=True)
class BackendProfile:
    """The bring-up facts a backend implies. Consumed by the generator (F2) and the TMR
    move-group builder; the single source of truth replacing scattered ``mode ==`` tests.

    - ``use_sim_time``      : nodes run on the sim ``/clock`` (True) or wall clock (False).
    - ``controller_manager``: CM_STANDALONE | CM_EXTERNAL_PLUGIN | CM_NONE (see above).
    - ``hardware_plugin``   : the ros2_control ``<hardware><plugin>`` id the xacro selects.
    - ``implemented``       : the generator emits/runs this backend today (False = reserved).
    """

    backend: Backend
    use_sim_time: bool
    controller_manager: str
    hardware_plugin: str
    implemented: bool = True


# The single source of truth. mock/isaac/real reproduce EXACTLY today's behaviour
# (isaac -> sim time + TopicBasedSystem; mock/real -> mock_components/GenericSystem;
# real launches no controller_manager). gazebo is RESERVED (wired in F2).
_PROFILES = {
    Backend.MOCK: BackendProfile(
        Backend.MOCK, use_sim_time=False, controller_manager=CM_STANDALONE,
        hardware_plugin='mock_components/GenericSystem'),
    Backend.ISAAC: BackendProfile(
        Backend.ISAAC, use_sim_time=True, controller_manager=CM_STANDALONE,
        hardware_plugin='topic_based_ros2_control/TopicBasedSystem'),
    Backend.GAZEBO: BackendProfile(
        Backend.GAZEBO, use_sim_time=True, controller_manager=CM_EXTERNAL_PLUGIN,
        hardware_plugin='gazebo_ros2_control/GazeboSystem', implemented=False),
    Backend.REAL: BackendProfile(
        Backend.REAL, use_sim_time=False, controller_manager=CM_NONE,
        hardware_plugin='mock_components/GenericSystem'),
}


def profile_for(backend) -> BackendProfile:
    """Return the :class:`BackendProfile` for a backend (a :class:`Backend` or its token).

    Raises ``ValueError`` for an unknown backend token (via ``Backend(...)``).
    """
    return _PROFILES[Backend(backend)]
