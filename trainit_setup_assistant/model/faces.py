"""The GRASP FACE registry, read from TSA's committed snapshot (ADR-0021, amendment M2.2).

``trainit_perception`` owns the faces: a name in a configuration file, a class, and
``register_face`` for one that lives outside the package. A face declares what it needs and what
it unlocks --

    ``params``          the parameter keys it reads out of the detector's flat params, in the
                        order the face declares them (the order a generator writes them in)
    ``enables_terms``   the candidate TERMS that become computable once those parameters are set

-- and that is what a configurator has to know. TSA used to keep its own copy of the answer for
ONE face, and a hand-written ``rect`` face was accepted by the runtime and refused by TSA
(ADR-0018 Decision 6). Then it ASKED the installed perception package, with a one-face fallback
when the package was missing -- and the same project generated differently on two machines.

So this module READS: ``trainit_perception.vocabulary.export()`` is committed as TSA's snapshot
(``catalog/tp_registry_snapshot.yaml``, refreshed by ``python3 -m
trainit_setup_assistant.catalog.refresh``), and the faces, the size band and the parameters'
version floors come from it. A face family added to the registry is configurable here once the
owner has classified it (a row in the catalog's ``TP_TIERS``, owner decision C4) and the catalog
is refreshed -- no line of this module or the picker changes; a face the snapshot does not
register is refused, on every machine, as TP's own registry refuses it. Extent terms and how a
face measures anything are not in the snapshot: TSA does not need them (ADR-0021, Decision 9).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from ..catalog import tp_registry

#: The face the runtime assumes when ``grasp_face`` is unset. It is a CONTRACT value, not
#: a preference: ``trainit_perception``'s ``computable_terms`` reads
#: ``params.get('grasp_face', 'disc')``, so a bundle that omits the key gets this one --
#: which is why TSA writes the key only when the face is something else, the way
#: ``blur_px`` and the depth bounds land only when they are on. The snapshot's default face must
#: be this one (``catalog.tp_vocabulary_problems`` refuses another).
DEFAULT_FACE = 'disc'


@dataclass(frozen=True)
class FaceKind:
    """What a configurator needs to know about one grasp face."""

    name: str
    params: Tuple[str, ...] = ()
    enables_terms: Tuple[str, ...] = ()

    def missing_params(self, params: Mapping) -> Tuple[str, ...]:
        """The face's own parameters that carry no usable positive number."""
        return tuple(name for name in self.params if not _positive(params.get(name)))

    def is_sized(self, params: Mapping) -> bool:
        """True when every parameter the face needs is set -- i.e. when the face has a
        known area and ``visible_fraction`` is therefore computable."""
        return not self.missing_params(params)


_cache: Optional[Dict[str, FaceKind]] = None


def _positive(value) -> bool:
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def _from_snapshot() -> Dict[str, FaceKind]:
    return {name: FaceKind(name, tuple(row['params']), tuple(row['enables_terms']))
            for name, row in tp_registry()['faces'].items()}


def param_floors() -> Dict[str, str]:
    """param -> the earliest ``trainit_perception`` release that understands it, from the snapshot.

    A generator has to pin the runtime a bundle needs, and the failure it pins against is SILENT:
    a parameter written by a newer assistant merges into an older runtime's params dict and is
    ignored -- no exception, no log, a knob that does nothing. TP dates every parameter it adds
    (the ranking's and the colour mask's own), and TSA reads the dates; it keeps none.
    """
    return dict(tp_registry()['param_since'])


def version_key(version: str) -> Tuple[int, ...]:
    """'0.10.0' sorts above '0.9.0'. A string compare does not, and would pin the wrong floor."""
    out = []
    for part in str(version).split('.'):
        digits = ''.join(c for c in part if c.isdigit())
        out.append(int(digits) if digits else 0)
    return tuple(out)


def face_registry(*, refresh: bool = False) -> Dict[str, FaceKind]:
    """name -> :class:`FaceKind`, from TSA's snapshot of trainit_perception's registry.

    Cached; ``refresh=True`` re-reads it (the tests use it after standing in a snapshot)."""
    global _cache
    if _cache is None or refresh:
        _cache = _from_snapshot()
    return dict(_cache)


def registry_source() -> str:
    """Where the face registry came from: TSA's snapshot, and the TP release it was taken of."""
    return f"TSA's snapshot of trainit_perception {tp_registry()['trainit_perception']}"


def size_band() -> Tuple[float, float]:
    """The (min, max) a face size may plausibly take, in METRES.

    The runtime refuses a size outside its own ``MIN_FACE_M``..``MAX_FACE_M`` inside
    ``make_detector``, so the picker refuses the same band at measuring time -- from the
    snapshot, the band the runtime declared.
    """
    low, high = tp_registry()['face_size_band_m']
    return float(low), float(high)


def available_faces() -> Tuple[str, ...]:
    return tuple(sorted(face_registry()))


def face_kind(name: str = DEFAULT_FACE) -> Optional[FaceKind]:
    """The face, or ``None`` when the snapshot does not register it."""
    return face_registry().get(str(name or DEFAULT_FACE))


def face_params(name: str = DEFAULT_FACE) -> Tuple[str, ...]:
    """The parameter keys a face reads -- what a configurator renders a form from."""
    kind = face_kind(name)
    return kind.params if kind else ()


def _face_sized_terms() -> Tuple[str, ...]:
    return tuple(t for t, req in tp_registry()['terms'].items() if req['face_sized'])


def face_enables(term: str, params: Mapping) -> Optional[bool]:
    """Whether the face this detector configures produces ``term`` at all; ``None`` when the
    face is not in the snapshot (not judged here: it is refused once, by name, as a face)."""
    kind = face_kind(face_of(params))
    return None if kind is None else term in kind.enables_terms


def face_term_requirement(term: str, params: Mapping) -> Optional[Tuple[str, ...]]:
    """The parameters ``term`` needs FROM THE FACE this detector configures.

    ``()`` when the term does not come from the face; the face's own parameters when it is a
    term the face ENABLES (``visible_fraction`` today, whatever a future face declares
    tomorrow); ``None`` when the configured face is not in the snapshot, which means "do not
    judge" -- :func:`face_problem` refuses that face, once.
    """
    kind = face_kind(face_of(params))
    if kind is None:
        return None if term in _face_sized_terms() else ()
    return kind.params if term in kind.enables_terms else ()


def all_face_params() -> Tuple[str, ...]:
    """Every parameter name any registered face can read, sorted.

    The version pin is derived from this (plus the rule and the tool's two numbers), so a
    face family that arrives in the perception package cannot smuggle a parameter into a
    bundle without the floor that guarantees the runtime reads it.
    """
    names = set()
    for kind in face_registry().values():
        names |= set(kind.params)
    return tuple(sorted(names))


def face_of(params: Mapping) -> str:
    """The face a detector's params select -- the runtime's own resolution of the key."""
    return str(params.get('grasp_face') or DEFAULT_FACE)


def face_param_requirement(params: Mapping) -> Optional[Tuple[str, ...]]:
    """The parameters ``visible_fraction`` needs for THIS detector's face.

    ``None`` means the configured face is not in the snapshot, so its parameters are unknown and
    must not be judged -- the same answer
    :data:`trainit_setup_assistant.model.perception.METHOD_PROVIDES` gives for an external-node
    method, and for the same reason.
    """
    kind = face_kind(face_of(params))
    return kind.params if kind else None


def face_problem(params: Mapping) -> Optional[str]:
    """Why this face cannot be used, or ``None``.

    The snapshot always answers, so a face it does not register is refused on every machine --
    the node would stop at start-up when it builds the face.
    """
    name = face_of(params)
    if face_kind(name) is not None:
        return None
    return (f'grasp_face "{name}" is not a face trainit_perception registers, so the '
            f'detector node would stop at start-up when it builds the face. Registered '
            f'here: {", ".join(available_faces())}.')


# --- the measurement -> parameters convention -------------------------------------------
# The registry says what a face's parameters are CALLED; it cannot say what they MEAN, and
# inventing a meaning for a name TSA has never seen is exactly the hardcoding Decision 6
# removes. So the picker fills the ones that follow the shipped NAMING CONVENTION below
# and leaves the rest to the user, with the measurement printed beside the field: a new
# face family is configurable the day it appears in the registry, and nothing is ever
# written into a bundle because TSA guessed what a parameter meant.
#
#   <prefix>radius_m        the EQUAL-AREA disc radius (visible_fraction's denominator)
#   <prefix>area_m2         the measured face area
#   <prefix>half_<a|x>_m    the longer in-plane half-extent
#   <prefix>half_<b|y>_m    the shorter in-plane half-extent
_CONVENTION = (
    (re.compile(r'(^|_)radius_m$'), 'radius_m'),
    (re.compile(r'(^|_)area_m2$'), 'area_m2'),
    (re.compile(r'(^|_)half_(a|x|long)_m$'), 'half_a_m'),
    (re.compile(r'(^|_)half_(b|y|short)_m$'), 'half_b_m'),
)


def measured_quantity(param: str) -> Optional[str]:
    """The :class:`~trainit_setup_assistant.geometry.mesh_face.FaceMeasurement` attribute
    that fills this parameter, or ``None`` when the convention does not cover it."""
    for pattern, attr in _CONVENTION:
        if pattern.search(str(param)):
            return attr
    return None


def prefill(face: str, measurement) -> Dict[str, float]:
    """The face's parameters TSA can fill from a measurement, by the convention above.

    Returns only what it is sure of. A parameter the convention does not cover is absent
    from the result, which is how the dialog knows to ask for it.
    """
    out: Dict[str, float] = {}
    for name in face_params(face):
        attr = measured_quantity(name)
        value = getattr(measurement, attr, None) if attr else None
        if value is not None:
            out[name] = float(value)
    return out
