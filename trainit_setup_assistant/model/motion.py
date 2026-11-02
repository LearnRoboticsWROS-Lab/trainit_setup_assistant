"""The canonical motion REQUEST vocabulary: constraints, profiles, provider choices (vNext M3.2).

The application states INTENT and LIMITS; a motion provider implements them (M3_DESIGN.md §7,
ADR-0025). These models are what a request may say -- M3_SCHEMA_PROPOSAL.md §1.3 -- and nothing
else: every one is STRICT (``extra = 'forbid'``, R-M3-02), so an unknown key is a refusal and
never a silent drop, and every number is checked for what it is (a percent, a finite positive
limit) instead of being coerced. What a request REQUIRES of its provider, and whether the provider
it resolves to honours it, is ``model.capabilities`` -- one derivation, one resolution.

    MotionConstraints  the limits of one motion, a path or a path segment: speed and acceleration
                       (percent of the robot's limits), absolute TCP velocity and acceleration,
                       a segment's blend radius, a goal tolerance, a planning budget; and the
                       RESERVED fields (jerk, orientation_constraint, duration_s), held so a
                       request for one is refused by name, never dropped
    MotionProfile      a named constraint set of the application (never a segment's blend)
    ProviderChoices    the application's provider per motion kind (registry provider ids)
    ProviderId         the provider rows of TMR's registry, from TSA's committed copy (ADR-0021)
    MotionIntent       what a request asks for: ptp | free | lin | circ | path

Keep three things apart (M3_DESIGN.md §5, §8): the INTENT (``MotionIntent``), the PROVIDER that
implements it (``ProviderId``, a registry row: ``pilz_lin``, ``ompl``, ``pilz_sequence``, ...)
and the MoveIt pipeline/planner a row points at (registry data, never named here). The legacy
``PlannerId`` (a planner MODE of the legacy templates' ``planner`` and ``global_planner_mode``) is
unchanged: a provider id is not a planner mode, and a legacy project never reads one.

These models are not yet part of any project file: M3.3's ``task_program`` application uses them.
Pydantic: the classic v1/v2-overlap API (``class Config``, ``validator``), as every TSA model.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from pydantic import BaseModel, root_validator, validator

from ..catalog import (REFRESH, declared_providers, missing_capability_vocabulary,
                       tmr_planner_registry)
from .enums import StrEnum

# TMR's registry, from TSA's committed copy -- never the installed TMR (ADR-0021). What TSA's
# capability resolution names must still be there, or TSA changes first.
_REGISTRY = tmr_planner_registry()
if missing_capability_vocabulary(_REGISTRY):
    raise RuntimeError(
        f"TMR's planner registry (catalog copy) no longer declares "
        f'{missing_capability_vocabulary(_REGISTRY)}, which TSA\'s capability resolution names '
        f'(model/capabilities.py): change TSA first, then {REFRESH}')

ProviderId = StrEnum('ProviderId', [(name.upper(), name) for name in
                                    declared_providers(_REGISTRY)],
                     module=__name__, qualname='ProviderId')
ProviderId.__doc__ = """A motion provider: a row of TMR's registry (ADR-0025), from TSA's copy.

Every DECLARED row, so a row that exists but is not stable or not deliverable is refused by name
(R-M3-11) rather than as an unknown token (R-M3-06)."""


class MotionIntent(StrEnum):
    """What a motion request asks for (M3_DESIGN.md §5.1). A path's default provider kind is
    ``path``, or ``dense_path`` for a dense source executed as one continuous motion."""

    PTP = 'ptp'      # joint-interpolated, collision-CHECKED (Pilz PTP)
    FREE = 'free'    # collision-AVOIDING (a sampling planner)
    LIN = 'lin'      # a straight TCP line
    CIRC = 'circ'    # an arc through a via point or about a centre
    PATH = 'path'    # >= 2 states as ONE motion: segments, or a dense source


class PathContinuity(StrEnum):
    """How a path passes its inner states: ``stop`` at each (every blend radius 0 -- documented as
    equivalent to a sequence of single motions), or ``blend`` through them."""

    STOP = 'stop'
    BLEND = 'blend'


class _Strict(BaseModel):
    """Every model of the canonical request refuses an unknown key (R-M3-02)."""

    class Config:
        extra = 'forbid'
        validate_assignment = True       # an edit is checked like a load


def _number(value: Any, field: str, *, integer: bool = False):
    """A YAML number as itself: never a bool, a string or a float truncated to an int (pydantic
    1.x would truncate 50.5 to 50), and never NaN or an infinity (R-M3-02)."""
    if value is None:
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'R-M3-02: {field} must be a number, got {value!r}')
    if integer and not isinstance(value, int):
        raise ValueError(f'R-M3-02: {field} must be a whole number, got {value!r}')
    if not math.isfinite(value):
        raise ValueError(f'R-M3-02: {field} must be a finite number, got {value!r}')
    return value


def _percent(value: Any, field: str):
    value = _number(value, field, integer=True)
    if value is not None and not 1 <= value <= 100:
        raise ValueError(f'R-M3-07: {field} is a percent in 1..100, got {value!r}')
    return value


def _positive(value: Any, field: str):
    value = _number(value, field)
    if value is not None and value <= 0:
        raise ValueError(f'R-M3-07: {field} must be > 0, got {value!r}')
    return value


def _non_negative(value: Any, field: str):
    value = _number(value, field)
    if value is not None and value < 0:
        raise ValueError(f'R-M3-07: {field} must be >= 0, got {value!r}')
    return value


def _at_least_one(value: Any, field: str):
    value = _number(value, field, integer=True)
    if value is not None and value < 1:
        raise ValueError(f'R-M3-07: {field} must be >= 1, got {value!r}')
    return value


def _checked(name: str, check, label: Optional[str] = None):
    """A pre-validator of ONE field running ``check(value, label)`` (the classic API on pydantic
    1.x and 2.x; 2.x no longer hands a validator its field, so each field gets its own)."""
    def run(cls, value):  # noqa: N805
        return check(value, label or name)
    run.__name__ = f'_check_{name}'
    return validator(name, pre=True, allow_reuse=True)(run)


class GoalTolerance(_Strict):
    """How close the goal must be reached. Each bound is finite and > 0; at least one is set.
    A provider honours it only if its row provides ``goal_tolerance`` (ADR-0025 Decision 3)."""

    position_m: Optional[float] = None
    orientation_rad: Optional[float] = None
    joint_rad: Optional[float] = None

    _position = _checked('position_m', _positive, 'goal_tolerance.position_m')
    _orientation = _checked('orientation_rad', _positive, 'goal_tolerance.orientation_rad')
    _joint = _checked('joint_rad', _positive, 'goal_tolerance.joint_rad')

    @root_validator(skip_on_failure=True)
    def _something(cls, values):  # noqa: N805
        if all(values.get(k) is None for k in ('position_m', 'orientation_rad', 'joint_rad')):
            raise ValueError('R-M3-07: goal_tolerance sets none of position_m, orientation_rad, '
                             'joint_rad')
        return values


# The request fields, in the order the proposal lists them, and the RESERVED ones: held so that a
# request for one is refused BY NAME by the capability resolution (no provider honours them).
CONSTRAINT_FIELDS = ('speed', 'acceleration', 'tcp_velocity_mps', 'tcp_acceleration_mps2',
                     'blend_radius_m', 'goal_tolerance', 'planning_time_s', 'planning_attempts')
RESERVED_CONSTRAINT_FIELDS = ('jerk', 'orientation_constraint', 'duration_s')


class MotionConstraints(_Strict):
    """The limits of one motion, a path or a path segment (M3_SCHEMA_PROPOSAL.md §1.3); the
    innermost set value wins (:meth:`merged`). Unset = not requested.

    speed, acceleration      PERCENT 1..100 of the robot's limits (provider velocity /
                             acceleration scaling: a documented equivalence)
    tcp_velocity_mps,        absolute TCP limits, > 0: the provider scaling is derived from the
    tcp_acceleration_mps2    configuration's pilz_cartesian_limits, never above them
    blend_radius_m           >= 0, a path SEGMENT's blend into the next one
    goal_tolerance           see GoalTolerance
    planning_time_s > 0,     the planning budget of a collision-avoiding search
    planning_attempts >= 1
    jerk, orientation_constraint, duration_s
                             RESERVED: accepted by the schema only to be refused by name
    """

    speed: Optional[int] = None
    acceleration: Optional[int] = None
    tcp_velocity_mps: Optional[float] = None
    tcp_acceleration_mps2: Optional[float] = None
    blend_radius_m: Optional[float] = None
    goal_tolerance: Optional[GoalTolerance] = None
    planning_time_s: Optional[float] = None
    planning_attempts: Optional[int] = None
    jerk: Optional[Any] = None
    orientation_constraint: Optional[Any] = None
    duration_s: Optional[Any] = None

    _speed = _checked('speed', _percent)
    _acceleration = _checked('acceleration', _percent)
    _tcp_velocity = _checked('tcp_velocity_mps', _positive)
    _tcp_acceleration = _checked('tcp_acceleration_mps2', _positive)
    _blend = _checked('blend_radius_m', _non_negative)
    _planning_time = _checked('planning_time_s', _positive)
    _attempts = _checked('planning_attempts', _at_least_one)

    def requested(self) -> tuple:
        """The fields this constraint set sets, request fields first, in a fixed order."""
        return tuple(f for f in CONSTRAINT_FIELDS + RESERVED_CONSTRAINT_FIELDS
                     if getattr(self, f) is not None)

    def merged(self, inner: Optional['MotionConstraints']) -> 'MotionConstraints':
        """These constraints with ``inner``'s set values on top: the innermost wins, field by
        field (a goal tolerance bound by bound). Returns a new MotionConstraints of THIS class's
        base type: merging is how a profile, a path and a segment combine into one request."""
        values = {f: getattr(self, f) for f in self.requested()}
        for f in (inner.requested() if inner is not None else ()):
            if f == 'goal_tolerance' and values.get(f) is not None:
                outer, new = values[f], inner.goal_tolerance
                values[f] = GoalTolerance(**{
                    b: getattr(new, b) if getattr(new, b) is not None else getattr(outer, b)
                    for b in ('position_m', 'orientation_rad', 'joint_rad')})
            else:
                values[f] = getattr(inner, f)
        return MotionConstraints(**values)


class MotionProfile(MotionConstraints):
    """A named constraint set of the application (``motion_profiles: {<name>: {...}}``, M3.3).
    The application never names a velocity-profile SHAPE (trapezoidal, TOTG, Ruckig): that is a
    provider row's ``time_parameterization``, informational. A blend radius is a segment's own."""

    @validator('blend_radius_m')
    def _no_blend(cls, v):  # noqa: N805
        if v is not None:
            raise ValueError('R-M3-02: blend_radius_m belongs to a path segment, not to a motion '
                             'profile (M3_SCHEMA_PROPOSAL.md §1.3)')
        return v


class ProviderChoices(_Strict):
    """The application's provider per motion kind (M3_SCHEMA_PROPOSAL.md §1 ``providers``); an
    unset kind gets the registry's ``provider_defaults``. A ``lin`` or ``circ`` move has no
    choice: it is always the registry's default (a ``planner`` on one is R-M3-06). Whether a
    chosen row is eligible and provides the kind is ``capabilities.check_provider_choices``."""

    ptp: Optional[ProviderId] = None
    free: Optional[ProviderId] = None
    path: Optional[ProviderId] = None
    dense_path: Optional[ProviderId] = None

    _ptp = _checked('ptp', lambda v, where: known_provider(v, where), 'providers.ptp')
    _free = _checked('free', lambda v, where: known_provider(v, where), 'providers.free')
    _path = _checked('path', lambda v, where: known_provider(v, where), 'providers.path')
    _dense = _checked('dense_path', lambda v, where: known_provider(v, where),
                      'providers.dense_path')
    # (a lambda: known_provider is defined below this class)


def known_provider(value: Any, where: str):
    """A provider token that names a row of TMR's registry snapshot, or R-M3-06 by name."""
    if value is None:
        return value
    known = [p.value for p in ProviderId]
    if not isinstance(value, str) or value not in known:
        raise ValueError(f'R-M3-06: {where}: provider {value!r} is not in TMR\'s registry snapshot '
                         f'(catalog/tmr_planner_registry.yaml); its providers are {known}')
    return value
