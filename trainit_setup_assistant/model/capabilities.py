"""Capability resolution: what a motion request REQUIRES, which provider honours it (vNext M3.2).

ADR-0025, M3_DESIGN.md §7-§8 and §15:

    REQUEST (intent + constraints [+ the provider it names])
        -> REQUIRED capability tokens          :func:`required_capabilities` -- the ONE derivation
        -> the provider: the request's own choice, else the application's (ProviderChoices),
           else the registry's ``provider_defaults`` -- never a search, never a fallback
        -> eligible (stable and deliverable) and required <= provides ?
        -> YES: a :class:`Resolution` (the effective provider, its scalings and the documented
           equivalences, for M3.3 to write into the bundle)
           NO:  a :class:`CapabilityRefusal`, BEFORE any file, naming the request, the token,
           the provider, what it provides, the constraint and why

TMR owns the vocabulary (the tokens, the provider rows, the defaults: registry schema 2); TSA reads
it from its committed copy (ADR-0021) and invents none: every token named below is in
``catalog.TSA_CAPABILITY_TOKENS``, which the refresh and the model import hold against the copy.
The derivation of a PATH is the one TMR's path library applies at load (``requiredCapabilities``),
so a path TSA accepts is never one TMR refuses for a capability.

Every motion is executed with a velocity and an acceleration scaling (TMR applies one whether or
not the request names it), so both tokens are required of every request, as TMR requires them of
every path. The three documented equivalences (M3_DESIGN.md §7) -- and no other:

    speed_percent      ``speed``/``acceleration`` N % == provider scaling N / 100
    tcp_limits         ``tcp_velocity_mps`` v == scaling v / pilz_cartesian_limits.max_trans_vel
                       (``tcp_acceleration_mps2`` a == a / max_trans_acc); a value above the
                       configured limit is REFUSED (R-M3-07), never clamped. Both a percent and an
                       absolute limit on one motion: both are LIMITS, so the effective scaling is
                       the smaller one (each stays honoured), and the result says which.
    continuity_stop    a ``continuity: stop`` path == a sequence of single motions: every blend
                       radius 0

TrainIt states intent; the provider owns HOW (no time-parameterization here).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from ..catalog import (capability_vocabulary, declared_providers, eligible_providers,
                       provider_defaults, tmr_planner_registry)
from .motion import (GoalTolerance, MotionConstraints, MotionIntent, PathContinuity,
                     ProviderChoices, ProviderId)

# --- the derivation tables (the only place TSA names a capability token) -------------------------

# what each intent requires by itself (a path's own tokens are derived from its shape below)
INTENT_TOKENS = {
    MotionIntent.PTP: ('joint_ptp',),
    MotionIntent.FREE: ('collision_free_planning',),
    MotionIntent.LIN: ('cartesian_linear',),
    MotionIntent.CIRC: ('cartesian_circular',),
}
# a path segment's shape
SEGMENT_TOKENS = {MotionIntent.LIN: 'cartesian_linear', MotionIntent.CIRC: 'cartesian_circular'}
# a path's own shape (TMR's requiredCapabilities): a path of segments, its blend, a dense path
# executed as one continuous motion
PATH_TOKEN, BLEND_TOKEN, CONTINUOUS_TOKEN = 'multi_segment_path', 'blending', 'continuous_path'
# what every motion requires: TMR executes every motion with a velocity and an acceleration scaling
ALWAYS_TOKENS = ('velocity_scaling', 'acceleration_scaling')
# what a constraint field, when set, requires of the provider. ``blend_radius_m`` only when > 0.
# ``duration_s`` maps to no token: no provider honours a segment duration (R-M3-12 by name).
CONSTRAINT_TOKENS = {
    'speed': ('velocity_scaling',),
    'acceleration': ('acceleration_scaling',),
    'tcp_velocity_mps': ('tcp_limits',),
    'tcp_acceleration_mps2': ('tcp_limits',),
    'blend_radius_m': (PATH_TOKEN, BLEND_TOKEN),
    'goal_tolerance': ('goal_tolerance',),
    'planning_time_s': ('collision_free_planning',),
    'planning_attempts': ('collision_free_planning',),
    'jerk': ('jerk_constraints',),
    'orientation_constraint': ('orientation_constraints',),
    'duration_s': (),
}
# the tokens a chosen provider must provide for its KIND, whatever the request (ProviderChoices)
KIND_TOKENS = {**{intent.value: tokens for intent, tokens in INTENT_TOKENS.items()},
               'path': (PATH_TOKEN,), 'dense_path': (CONTINUOUS_TOKEN,)}

# the documented equivalences, by id
EQ_SPEED_PERCENT = 'speed_percent'
EQ_TCP_LIMITS = 'tcp_limits'
EQ_CONTINUITY_STOP = 'continuity_stop'

# where the effective provider came from
FROM_REQUEST = 'request'
FROM_APPLICATION = 'application'
FROM_REGISTRY_DEFAULT = 'registry_default'

_REGISTRY = tmr_planner_registry()


# --- the request ---------------------------------------------------------------------------------

@dataclass(frozen=True)
class SegmentRequest:
    """One segment of a path: its shape (``lin`` | ``circ``; ``ptp`` is reserved, R-M3-06) and its
    own constraints (they win over the path's)."""

    type: MotionIntent
    constraints: MotionConstraints = field(default_factory=MotionConstraints)


@dataclass(frozen=True)
class MotionRequest:
    """One motion as the application asks for it, for resolution -- not a schema (M3.3 builds it
    from a task step). ``where`` names it in every refusal (``task 'scan' step 2``).

    ``constraints`` are the effective ones (a profile merged with the step's own:
    ``MotionConstraints.merged``). ``provider`` is the request's OWN choice: a ``ptp``/``free``
    move's ``planner`` or a path's ``provider``. A path has ``segments`` (states - 1) and a
    ``continuity`` (default stop), or is ``dense`` (a sampled source: no segments; ``blend`` =
    one continuous motion, ``stop`` = a stop at every sample)."""

    where: str
    intent: MotionIntent
    constraints: MotionConstraints = field(default_factory=MotionConstraints)
    provider: Optional[str] = None
    segments: Tuple[SegmentRequest, ...] = ()
    continuity: Optional[PathContinuity] = None
    dense: bool = False

    def kind(self) -> str:
        """The registry's ``provider_defaults`` key for this request (TMR's own rule)."""
        if self.intent is MotionIntent.PATH:
            dense_blend = self.dense and self.continuity is PathContinuity.BLEND
            return 'dense_path' if dense_blend else 'path'
        return self.intent.value


@dataclass(frozen=True)
class CartesianLimits:
    """The configuration's pilz_cartesian_limits a TCP limit is derived from, and where they were
    read (the project, or a hand-made base moveit_config's file)."""

    max_trans_vel: float
    max_trans_acc: float
    source: str


# --- the result ----------------------------------------------------------------------------------

@dataclass(frozen=True)
class Scaling:
    """An effective provider scaling in (0, 1], the field it comes from and its derivation."""

    value: float
    source: str
    derivation: str


@dataclass(frozen=True)
class SegmentResolution:
    index: int
    type: MotionIntent
    velocity_scaling: Optional[Scaling]
    acceleration_scaling: Optional[Scaling]
    blend_radius_m: float


@dataclass(frozen=True)
class Resolution:
    """A request resolved: the ONE effective provider and why it was chosen, what the request
    requires and what the provider provides, the effective scalings (None = not requested: the
    provider's default), a path's segments and the documented equivalences applied, as
    ``(id, derivation)``. Everything M3.3 needs to write the effective request into a bundle."""

    where: str
    intent: MotionIntent
    kind: str
    provider: str
    provider_source: str
    required: Tuple[str, ...]
    provided: Tuple[str, ...]
    velocity_scaling: Optional[Scaling]
    acceleration_scaling: Optional[Scaling]
    segments: Tuple[SegmentResolution, ...] = ()
    equivalences: Tuple[Tuple[str, str], ...] = ()


class CapabilityRefusal(ValueError):
    """A request TSA refuses before any file (an R-M3 rule), with everything a diagnosis needs.
    A ValueError: generation's refusal type."""

    def __init__(self, rule: str, where: str, reason: str, *, intent: Optional[str] = None,
                 provider: Optional[str] = None, provider_source: Optional[str] = None,
                 required: Sequence[str] = (), provided: Sequence[str] = (),
                 missing: Sequence[str] = (), because: Sequence[str] = ()):
        self.rule, self.where, self.reason = rule, where, reason
        self.intent, self.provider, self.provider_source = intent, provider, provider_source
        self.required, self.provided = tuple(required), tuple(provided)
        self.missing, self.because = tuple(missing), tuple(because)
        super().__init__(self._text())

    def _text(self) -> str:
        head = f'{self.rule} REFUSED: {self.where}'
        if self.intent:
            head += f' ({self.intent})'
        lines = [f'{head}: {self.reason}']
        if self.required:
            lines.append(f'  required: {", ".join(self.required)}')
        if self.provider:
            src = f' ({self.provider_source})' if self.provider_source else ''
            lines.append(f'  provider: {self.provider}{src}')
            lines.append(f'  provides: {", ".join(self.provided) or "(nothing)"}')
        if self.missing:
            lines.append(f'  missing:  {", ".join(self.missing)}')
        lines += [f'  because:  {b}' for b in self.because]
        return '\n'.join(lines)


class CapabilityRefusals(ValueError):
    """Every refusal of a set of requests, at once (:func:`resolve_all`)."""

    def __init__(self, refusals: Sequence[CapabilityRefusal]):
        self.refusals = tuple(refusals)
        super().__init__('\n'.join(str(r) for r in self.refusals))


# --- the derivation ------------------------------------------------------------------------------

def _ordered(tokens: Iterable[str], registry: dict) -> Tuple[str, ...]:
    """Tokens in the registry's vocabulary order (live, then reserved): one deterministic order."""
    vocab = capability_vocabulary(registry)
    order = {t: i for i, t in enumerate(vocab['live'] + vocab['reserved'])}
    return tuple(sorted(set(tokens), key=lambda t: (order.get(t, len(order)), t)))


def _constraint_reasons(c: MotionConstraints, prefix: str, out: Dict[str, List[str]],
                        *, in_path: bool) -> None:
    for f in c.requested():
        if f == 'blend_radius_m' and (in_path or not c.blend_radius_m):
            continue                      # a path's blends are its continuity's; a 0 blend: none
        for token in CONSTRAINT_TOKENS[f]:
            out.setdefault(token, []).append(f'{prefix}{f}')


def _shape(request: MotionRequest) -> Dict[str, List[str]]:
    """What the request's intent and shape require, by themselves (not its constraint fields)."""
    out: Dict[str, List[str]] = {}
    if request.intent is not MotionIntent.PATH:
        for token in INTENT_TOKENS[request.intent]:
            out.setdefault(token, []).append(f'motion {request.intent.value}')
        return out
    blend = request.continuity is PathContinuity.BLEND
    if request.dense:
        tokens = (CONTINUOUS_TOKEN,) if blend else (PATH_TOKEN, SEGMENT_TOKENS[MotionIntent.LIN])
        for token in tokens:
            out.setdefault(token, []).append(
                f'a dense path, continuity {"blend" if blend else "stop"}')
        return out
    out.setdefault(PATH_TOKEN, []).append('a path')
    for i, seg in enumerate(request.segments):
        if seg.type in SEGMENT_TOKENS:
            out.setdefault(SEGMENT_TOKENS[seg.type], []).append(f'segment {i} {seg.type.value}')
    if blend:
        out.setdefault(BLEND_TOKEN, []).append('continuity blend')
    return out


def required_capabilities(request: MotionRequest,
                          registry: Optional[dict] = None) -> Dict[str, List[str]]:
    """What ``request`` requires of its provider: token -> the reasons (the intent, the shape or
    the constraint field that requires it), in the registry's vocabulary order. THE derivation:
    every rule of M3_DESIGN.md §7 is a row of the tables above, and a path's is TMR's own."""
    registry = _REGISTRY if registry is None else registry
    out = _shape(request)
    in_path = request.intent is MotionIntent.PATH
    _constraint_reasons(request.constraints, '', out, in_path=in_path)
    for i, seg in enumerate(request.segments):
        _constraint_reasons(seg.constraints, f'segment {i} ', out, in_path=True)
    for token in ALWAYS_TOKENS:
        out.setdefault(token, []).append('every motion is executed with a scaling')
    return {t: out[t] for t in _ordered(out, registry)}


# --- the configuration a TCP limit is derived from -----------------------------------------------

def project_cartesian_limits(project) -> CartesianLimits:
    """The pilz_cartesian_limits the bundle of ``project`` will carry: the hand-made base
    moveit_config's ``config/pilz_cartesian_limits.yaml`` when the project names one (the
    scene-loader flow copies that file), else ``robot.pilz_cartesian_limits`` (the from-scratch
    flow renders it). A file that cannot be read is R-M3-07 by name, never a default."""
    robot = project.robot
    if robot.base_moveit_config_path:
        path = Path(robot.base_moveit_config_path) / 'config' / 'pilz_cartesian_limits.yaml'
        try:
            limits = (yaml.safe_load(path.read_text()) or {})['cartesian_limits']
            return CartesianLimits(limits['max_trans_vel'], limits['max_trans_acc'], str(path))
        except Exception as exc:                                  # noqa: BLE001 - named below
            raise CapabilityRefusal(
                'R-M3-07', 'the project', f'the TCP limits cannot be read from {path} '
                f'({type(exc).__name__}: {exc}): a TCP velocity or acceleration needs them'
            ) from None
    pilz = robot.pilz_cartesian_limits
    return CartesianLimits(pilz.max_trans_vel, pilz.max_trans_acc,
                           'robot.pilz_cartesian_limits')


def _limit(limits: Optional[CartesianLimits], name: str, where: str,
           intent: Optional[str]) -> float:
    if limits is None:
        raise CapabilityRefusal('R-M3-07', where, 'an absolute TCP limit needs the '
                                'configuration\'s pilz_cartesian_limits, and none was given',
                                intent=intent)
    value = getattr(limits, name)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or \
            not math.isfinite(value) or value <= 0:
        raise CapabilityRefusal('R-M3-07', where, f'the configuration\'s {limits.source}.{name} '
                                f'is {value!r}: a TCP limit is derived from a finite limit > 0',
                                intent=intent)
    return float(value)


# --- the equivalences ----------------------------------------------------------------------------

def _scaling(c: MotionConstraints, limits: Optional[CartesianLimits], where: str,
             intent: str, percent: str, absolute: str,
             limit: str) -> Tuple[Optional[Scaling], List[Tuple]]:
    """The effective scaling of one axis (velocity or acceleration) and the equivalences used."""
    options, used = [], []
    if getattr(c, percent) is not None:
        p = getattr(c, percent)
        options.append(Scaling(p / 100.0, percent, f'{percent} {p} % = scaling {p / 100.0:g}'))
        used.append((EQ_SPEED_PERCENT, options[-1].derivation))
    if getattr(c, absolute) is not None:
        v, cap = getattr(c, absolute), _limit(limits, limit, where, intent)
        if v > cap:
            raise CapabilityRefusal(
                'R-M3-07', where, f'{absolute} {v:g} exceeds the configured '
                f'{limits.source}.{limit} {cap:g}: it is refused, not clamped', intent=intent,
                because=[f'{absolute} {v:g} > {limit} {cap:g}'])
        options.append(Scaling(v / cap, absolute, f'{absolute} {v:g} / {limits.source}.{limit} '
                               f'{cap:g} = scaling {v / cap:g}'))
        used.append((EQ_TCP_LIMITS, options[-1].derivation))
    if not options:
        return None, []
    chosen = min(options, key=lambda s: (s.value, s.source))
    if len(options) > 1:
        other = next(s for s in options if s is not chosen)
        chosen = Scaling(chosen.value, chosen.source,
                         f'{chosen.derivation} (the smaller of two limits; {other.derivation})')
    return chosen, used


def _scalings(c, limits, where, intent):
    v, used_v = _scaling(c, limits, where, intent, 'speed', 'tcp_velocity_mps', 'max_trans_vel')
    a, used_a = _scaling(c, limits, where, intent, 'acceleration', 'tcp_acceleration_mps2',
                         'max_trans_acc')
    return v, a, used_v + used_a


# --- structure (before any token) ----------------------------------------------------------------

def _structure(request: MotionRequest) -> None:
    """The refusals that need no provider: a request's own shape (R-M3-04/06/07/08)."""
    where, c = request.where, request.constraints
    if request.intent is not MotionIntent.PATH:
        if request.segments or request.continuity is not None or request.dense:
            raise CapabilityRefusal('R-M3-04', where, 'segments, continuity and dense belong to '
                                    'a path', intent=request.intent.value)
        if request.provider is not None and request.intent in (MotionIntent.LIN,
                                                               MotionIntent.CIRC):
            raise CapabilityRefusal(
                'R-M3-06', where, f'a planner on a {request.intent.value} move: a '
                f'{request.intent.value} move is always the registry default '
                f'({provider_defaults(_REGISTRY).get(request.intent.value)}), it chooses no '
                'provider', intent=request.intent.value)
        return
    if c.blend_radius_m is not None:
        raise CapabilityRefusal('R-M3-08', where, 'blend_radius_m is a segment\'s, not a '
                                'path\'s: give it to the segment it blends out of',
                                intent='path')
    if request.dense:
        if request.segments:
            raise CapabilityRefusal('R-M3-04', where, 'a dense path has no segments (its samples '
                                    'are joined by implicit lin segments)', intent='path')
    else:
        if not request.segments:
            raise CapabilityRefusal('R-M3-04', where, 'a path needs at least 2 states, so at '
                                    'least one segment', intent='path')
        for i, seg in enumerate(request.segments):
            if seg.type not in SEGMENT_TOKENS:
                raise CapabilityRefusal(
                    'R-M3-06', where, f'segment {i} is {seg.type.value}: a path segment is lin or '
                    'circ (a joint-space segment inside a path is reserved: '
                    'UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL)', intent='path')
        radii = [s.constraints.blend_radius_m or 0.0 for s in request.segments]
        if radii[-1] > 0:
            raise CapabilityRefusal('R-M3-08', where, f'the last segment blends '
                                    f'(blend_radius_m {radii[-1]:g}): a path ends at rest',
                                    intent='path')
        blend = request.continuity is PathContinuity.BLEND
        if not blend and any(radii):
            raise CapabilityRefusal('R-M3-08', where, 'a blend_radius_m on a continuity: stop '
                                    'path: say continuity: blend, or remove the radius',
                                    intent='path')
        if blend and not any(radii):
            raise CapabilityRefusal('R-M3-08', where, 'continuity: blend with no blend_radius_m '
                                    '> 0: the path would stop at every state; say continuity: '
                                    'stop, or give the radii', intent='path')
    for i, seg in enumerate(request.segments):
        if seg.constraints.goal_tolerance is not None:
            raise CapabilityRefusal('R-M3-12', where, f'segment {i} sets a goal_tolerance: a path '
                                    'has ONE goal, its last state, so the tolerance is the '
                                    'path\'s (TMR\'s path library takes none per segment)',
                                    intent='path')
    if c.goal_tolerance is not None:
        tol = c.goal_tolerance
        if tol.position_m is None or tol.orientation_rad is None or tol.joint_rad is not None:
            raise CapabilityRefusal('R-M3-07', where, 'a path\'s goal_tolerance is Cartesian: it '
                                    'needs position_m and orientation_rad, and no joint_rad '
                                    '(TMR\'s path library)', intent='path')


def check_blend_geometry(where: str, positions: Sequence[Sequence[float]],
                         radii: Sequence[float]) -> None:
    """R-M3-08's geometry, when every state's position is known (inline or taught poses; a state
    bound at run time is checked by TMR when it ticks): a blend radius larger than half of either
    segment it joins would swallow a state. The rule and the chord lengths are TMR's
    (``checkPathGeometry``)."""
    for i, r in enumerate(radii):
        if r <= 0 or i + 2 >= len(positions):
            continue
        a, b, c = positions[i], positions[i + 1], positions[i + 2]
        shortest = min(math.dist(a, b), math.dist(b, c))
        if r > 0.5 * shortest:
            raise CapabilityRefusal('R-M3-08', where, f'segment {i}: blend_radius_m {r:g} exceeds '
                                    f'half of an adjacent segment ({shortest:g} m): the blend '
                                    'would swallow a state', intent='path')


# --- the resolution ------------------------------------------------------------------------------

def missing_capabilities(provider: str, required: Iterable[str],
                         registry: Optional[dict] = None) -> List[str]:
    """THE compatibility rule, ``required <= provides``: the required tokens ``provider``'s row
    does not provide, in the registry's order. Empty = compatible."""
    registry = _REGISTRY if registry is None else registry
    rows = declared_providers(registry)
    if provider not in rows:
        raise CapabilityRefusal('R-M3-06', f'provider {provider!r}', 'not in TMR\'s registry '
                                f'snapshot; its providers are {list(rows)}')
    provides = set(rows[provider]['provides'])
    return [t for t in _ordered(required, registry) if t not in provides]


def _choice(request: MotionRequest, choices: Optional[ProviderChoices],
            registry: dict) -> Tuple[Optional[str], str]:
    """(the provider, where it came from): the request's own, the application's, the registry's
    default for the request's kind -- in that order, and nothing else."""
    if request.provider is not None:
        return request.provider, FROM_REQUEST
    chosen = getattr(choices, request.kind(), None) if choices is not None else None
    if chosen is not None:
        return chosen.value if isinstance(chosen, ProviderId) else chosen, FROM_APPLICATION
    return provider_defaults(registry).get(request.kind()), FROM_REGISTRY_DEFAULT


def resolve(request: MotionRequest, choices: Optional[ProviderChoices] = None,
            limits: Optional[CartesianLimits] = None,
            registry: Optional[dict] = None) -> Resolution:
    """The ONE effective provider of ``request``, or a CapabilityRefusal naming why (module doc).
    Deterministic: a function of the request, the choices, the limits and TSA's committed registry
    copy -- never of what is installed or importable."""
    registry = _REGISTRY if registry is None else registry
    where, intent = request.where, request.intent.value
    _structure(request)
    if request.provider is not None:
        known = list(declared_providers(registry))
        if request.provider not in known:
            raise CapabilityRefusal('R-M3-06', where, f'provider {request.provider!r} is not in '
                                    f'TMR\'s registry snapshot; its providers are {known}',
                                    intent=intent)
    if request.constraints.duration_s is not None or any(
            s.constraints.duration_s is not None for s in request.segments):
        raise CapabilityRefusal('R-M3-12', where, 'duration_s is reserved: no provider honours a '
                                'segment duration (UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL)',
                                intent=intent)
    required = required_capabilities(request, registry)
    provider, source = _choice(request, choices, registry)
    rows = declared_providers(registry)
    if provider is None or provider not in rows:
        raise CapabilityRefusal('R-M3-11', where, f'TMR\'s registry snapshot (schema '
                                f'{registry.get("schema_version")}) has no provider for a '
                                f'{request.kind()} request', intent=intent,
                                required=list(required))
    provided = _ordered(rows[provider]['provides'], registry)
    common = dict(intent=intent, provider=provider, provider_source=source,
                  required=list(required), provided=provided)
    if provider not in eligible_providers(registry):
        raise CapabilityRefusal('R-M3-11', where, f'provider {provider!r} is not eligible: its '
                                f'row is {rows[provider].get("stability")!r} or its pipeline is '
                                'not in the loaded set (ADR-0021 Decision 5)', **common)
    missing = missing_capabilities(provider, required, registry)
    if missing:
        raise _missing(request, required, missing, provider, registry, common)
    # the effective request: scalings, a path's segments, the equivalences
    v, a, used = _scalings(request.constraints, limits, where, intent)
    segments: List[SegmentResolution] = []
    if request.intent is MotionIntent.PATH:
        stop = request.continuity is not PathContinuity.BLEND
        for i, seg in enumerate(request.segments):
            eff = request.constraints.merged(seg.constraints)
            sv, sa, s_used = _scalings(eff, limits, f'{where} segment {i}', intent)
            used += s_used
            segments.append(SegmentResolution(
                i, seg.type, sv, sa, 0.0 if stop else float(seg.constraints.blend_radius_m or 0.0)))
        if stop:
            dense = ' (a stop at every sample)' if request.dense else ''
            used.append((EQ_CONTINUITY_STOP, 'continuity stop: every blend radius 0, equivalent '
                         f'to a sequence of single motions{dense}'))
    return Resolution(where=where, intent=request.intent, kind=request.kind(), provider=provider,
                      provider_source=source, required=tuple(required), provided=provided,
                      velocity_scaling=v, acceleration_scaling=a, segments=tuple(segments),
                      equivalences=tuple(dict.fromkeys(used)))


def candidate_providers(kind: str, registry: Optional[dict] = None) -> List[str]:
    """The eligible rows a request of ``kind`` could resolve to by ANY choice: a ``ptp``, ``free``,
    ``path`` or ``dense_path`` request may name a row (its own choice or the application's), so
    every eligible row providing the kind's tokens; a ``lin`` or ``circ`` move names none
    (R-M3-06), so only the registry default."""
    registry = _REGISTRY if registry is None else registry
    rows, eligible = declared_providers(registry), eligible_providers(registry)
    if kind in ('lin', 'circ'):
        default = provider_defaults(registry).get(kind)
        return [default] if default in eligible else []
    return [p for p in eligible if all(t in rows[p]['provides'] for t in KIND_TOKENS[kind])]


def _missing(request, required, missing, provider, registry, common) -> CapabilityRefusal:
    """R-M3-10 (the chosen provider lacks a token another provider of this kind of step has, or
    lacks the step's own shape, or a token is reserved), or R-M3-12 (a constraint field no
    provider the step could choose can honour), naming every missing token, where it comes from
    and the rows of this kind that provide it."""
    rows = declared_providers(registry)
    reserved = set(capability_vocabulary(registry)['reserved'])
    shape = set(_shape(request))
    kind = request.kind()
    candidates = candidate_providers(kind, registry)
    because = []
    for t in missing:
        having = [p for p in candidates if t in rows[p]['provides']]
        note = 'reserved: no provider provides it' if t in reserved else (
            f'provided by {", ".join(having)}' if having else f'no {kind} provider provides it')
        because.append(f'{t} <- {"; ".join(required[t])} ({note})')
    if any(t in reserved for t in missing) or any(t in shape for t in missing) or any(
            all(t in rows[p]['provides'] for t in missing) for p in candidates):
        reason = f'provider {provider!r} does not provide {", ".join(repr(t) for t in missing)}'
        return CapabilityRefusal('R-M3-10', request.where, reason, missing=missing,
                                 because=because, **common)
    fields = sorted({w for t in missing for w in required[t]})
    return CapabilityRefusal('R-M3-12', request.where,
                             f'no {kind} provider can honour {", ".join(fields)}',
                             missing=missing, because=because, **common)


def resolve_all(requests: Sequence[MotionRequest], choices: Optional[ProviderChoices] = None,
                limits: Optional[CartesianLimits] = None,
                registry: Optional[dict] = None) -> List[Resolution]:
    """Every request resolved, or ONE CapabilityRefusals naming every refusal (none is hidden
    behind the first)."""
    out, refused = [], []
    for request in requests:
        try:
            out.append(resolve(request, choices, limits, registry))
        except CapabilityRefusal as exc:
            refused.append(exc)
    if refused:
        raise CapabilityRefusals(refused)
    return out


def check_provider_choices(choices: ProviderChoices, registry: Optional[dict] = None) -> None:
    """The application's provider choices, before any request uses them: each chosen row must be
    eligible (R-M3-11) and provide its kind (R-M3-10) -- a choice no step uses is still checked."""
    registry = _REGISTRY if registry is None else registry
    rows, eligible = declared_providers(registry), eligible_providers(registry)
    for kind in ('ptp', 'free', 'path', 'dense_path'):
        chosen = getattr(choices, kind)
        if chosen is None:
            continue
        name = chosen.value if isinstance(chosen, ProviderId) else chosen
        where = f'providers.{kind}'
        if name not in rows:
            raise CapabilityRefusal('R-M3-06', where, f'provider {name!r} is not in TMR\'s '
                                    f'registry snapshot; its providers are {list(rows)}')
        provided = _ordered(rows[name]['provides'], registry)
        common = dict(provider=name, provider_source=FROM_APPLICATION, provided=provided,
                      required=list(KIND_TOKENS[kind]))
        if name not in eligible:
            raise CapabilityRefusal('R-M3-11', where, f'provider {name!r} is not eligible (not '
                                    'stable, or its pipeline is not loaded)', **common)
        missing = missing_capabilities(name, KIND_TOKENS[kind], registry)
        if missing:
            raise CapabilityRefusal('R-M3-10', where, f'provider {name!r} cannot serve a {kind} '
                                    f'step: it does not provide {", ".join(missing)}',
                                    missing=missing, **common)


__all__ = ['ALWAYS_TOKENS', 'CONSTRAINT_TOKENS', 'CapabilityRefusal', 'CapabilityRefusals',
           'CartesianLimits', 'EQ_CONTINUITY_STOP', 'EQ_SPEED_PERCENT', 'EQ_TCP_LIMITS',
           'FROM_APPLICATION', 'FROM_REGISTRY_DEFAULT', 'FROM_REQUEST', 'GoalTolerance',
           'INTENT_TOKENS', 'KIND_TOKENS', 'MotionRequest', 'Resolution', 'SEGMENT_TOKENS',
           'Scaling', 'SegmentRequest', 'SegmentResolution', 'check_blend_geometry',
           'candidate_providers', 'check_provider_choices', 'missing_capabilities',
           'project_cartesian_limits',
           'required_capabilities',
           'resolve', 'resolve_all']
