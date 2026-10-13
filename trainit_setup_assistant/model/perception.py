"""Perception model (D-014 / D-015): the camera and the detectors as NAMED RESOURCES.

The dedicated Perception step configures *what the camera sees and how* (this module);
the application step only BINDS a detector to waypoints (``Waypoint.vision``). One
detector serves any number of applications.

Everything here is DATA consumed by three emitters:
- ``perception.yaml`` + one ``detector_node`` per pure-python detector in ``_app``;
- the camera-variant URDF include rewrite + the base↔camera SRDF pair in
  ``_description`` and ``_trainit_config``;
- the synthetic coloured cloud (``depth_image_proc``) in the isaac bring-up branch.

Generation stays pure: nothing here touches ROS. The measured cell values (settle
1500 ms, dz offsets) are PROJECT data with defaults, never generator constants.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from ..catalog import tp_registry
from .enums import CandidateRule, DetectionMethod
from .external_node import INPUT_ROLES, ExternalNodeSpec
from .faces import face_enables, face_of, face_term_requirement

# The stock RealSense names, used when a project has no ``CameraSpec`` at all. Here rather
# than in the generator because they are the DEFAULT VALUE of project data, and the
# external-node launch entry resolves the same three topics the generated
# ``perception.yaml`` does (see :meth:`DetectorSpec.input_topics`).
DEFAULT_INPUT_TOPICS: Dict[str, str] = {
    'rgb': '/camera/color/image_raw',
    'depth': '/camera/depth/image_rect_raw',
    'camera_info': '/camera/color/camera_info',
}

assert set(DEFAULT_INPUT_TOPICS) == set(INPUT_ROLES), (
    'the camera contract has exactly three roles and both the perception.yaml input block '
    'and an external node\'s topic_parameters key off the SAME tokens')

# --- the candidate ranking's computability contract (ADR-0017 Decision 3) ------------
# A rule is a PRODUCT of named TERMS, and a term is computable only when two things are
# true: the DETECTOR produces what the term reads (a box, a mask area, a 3D fit), and the
# PARAMETERS the term needs are set. Both are configuration a person chose in the wizard, so a
# rule that cannot be computed is refused HERE, before a bundle exists — since the published
# ORDER is the ranking (ADR-0016 Decision 1), it must never quietly mean "whatever order the
# model returned".
#
# WHAT each method provides, WHICH terms each rule multiplies and WHAT each term needs are
# trainit_perception's declarations, read from TSA's committed snapshot of its vocabulary
# (ADR-0021, amendment M2.2) — never from the installed TP, and never a copy typed in here. What
# stays TSA's is the sentence a configurator reads when a term is missing (``TERM_HINTS``).
_TP = tp_registry()

# What each METHOD produces per detection. ``None`` = TSA cannot know: an external-node
# method publishes the contract from its own node (TSA emits no detector for it and has no
# field for its capabilities), so its rule is NOT refused on capability grounds — only on
# the parameters, and the node itself makes the final check on the file it reads. TSA's own
# ``custom`` is such a method; ``DetectionMethod`` stays TSA code (ADR-0021 Decision 6).
METHOD_PROVIDES: Dict[DetectionMethod, Optional[Tuple[str, ...]]] = {
    method: (None if (_TP['detector_methods'].get(method.value) or {}).get('provides') is None
             else tuple(_TP['detector_methods'][method.value]['provides']))
    for method in DetectionMethod}

# The terms each rule multiplies, in the rule's own order. AUTO takes none: it is resolved by the
# runtime from what the detector can compute, so it is always configurable — and
# ``detector_order`` takes none either: an EXPLICIT "no ranking", which is exactly what
# distinguishes it from a rule that silently could not be computed.
CANDIDATE_RULE_TERMS: Dict[CandidateRule, Tuple[str, ...]] = {
    CandidateRule.AUTO: (),
    **{CandidateRule(rule): tuple(terms) for rule, terms in _TP['rules'].items()}}

# A rule without a row above would raise a bare KeyError out of the refusal path — the
# ranking would break where it is supposed to explain itself. Checked at import, so the
# enum and the table cannot drift.
assert set(CANDIDATE_RULE_TERMS) == set(CandidateRule), (
    'every CandidateRule needs a CANDIDATE_RULE_TERMS row (even an empty one): it is what '
    'makes the rule refusable instead of silently unrankable')

# The sentence that tells a configurator where to set what a term needs — TSA's own words, about
# TSA's own steps. "expected_radius_m is missing" is not actionable; "the grasp target has no
# usable shape and dims" is. A term with no sentence here gets one built from its requirements.
TERM_HINTS: Dict[str, str] = {
    'score': 'every detection carries one',
    'pose_quality': (
        'a detector that runs a 3D FIT over the depth inside the mask — the colour mask '
        'does not, so choose a "score x …" rule for it, or an external-node method that '
        'fits a pose'),
    'box_overlap': 'a detector that reports a pixel bounding box',
    'visible_fraction': (
        "a detector that reports a MASK AREA, plus the grasp face's size — MEASURE it on "
        'the object (Perception step -> "Measure grasp face…"), or mark the picked object '
        'as a dynamic grasp target in the Scene step and give it a shape with non-zero '
        'dimensions, which lets TSA approximate the face from its bounding box'),
    'clearance': (
        "a 3D fit plus the TOOL's footprint radius and grip window, both on the "
        'end-effector (Step 7 -> End-effector dialog)'),
    'edge_on': (
        "a 3D FIT over the depth inside the mask plus the TOOL's tilt limit "
        '(tool_max_tilt_deg > 0 in the detector\'s params) — the colour mask runs no fit, so '
        'choose an external-node method that fits a pose'),
}


def _generic_hint(req: Dict[str, Any]) -> str:
    parts = [f'a detector that provides {", ".join(req["provides"])}'] if req['provides'] else []
    if req['positive_params']:
        parts.append(f'{" and ".join(req["positive_params"])} > 0')
    if req['face_sized']:
        parts.append("the grasp face's size")
    return ', plus '.join(parts) or 'nothing a detector has to be configured for'


# Per term: what the detector must produce, which parameters must be set, and the sentence. The
# parameters a FACE-SIZED term needs belong to the configured face, so they are read per detector
# (``term_param_needs``) and the tuple here holds only the term's own.
TERM_NEEDS: Dict[str, Tuple[Tuple[str, ...], Tuple[str, ...], str]] = {
    term: (tuple(req['provides']), tuple(req['positive_params']),
           TERM_HINTS.get(term) or _generic_hint(req))
    for term, req in _TP['terms'].items()}

# The terms that come from the configured grasp FACE (computable when that face enables them and
# every one of its parameters is set).
FACE_SIZED_TERMS = frozenset(term for term, req in _TP['terms'].items() if req['face_sized'])


def _positive(value: Any) -> bool:
    """True when a parameter carries a usable positive number (a missing / 0 / non-numeric
    value is 'not computable', not 'zero')."""
    try:
        return float(value) > 0.0
    except (TypeError, ValueError):
        return False


def term_param_needs(term: str, params: Dict[str, Any]) -> Optional[Tuple[str, ...]]:
    """The PARAMETERS a term needs, for a detector configured THIS way.

    Two halves, added together. The static half is in :data:`TERM_NEEDS` — the tool's two
    numbers for ``clearance``, the tilt limit for ``edge_on``, nothing for the rest. The other
    half comes from the configured GRASP FACE: a face declares the TERMS it enables
    (``visible_fraction`` today) and the PARAMETERS it is sized by — trainit_perception's
    declarations, read from TSA's snapshot of them (ADR-0021), not a copy that is right for
    exactly one face.

    ``None`` means "not judged here": the configured face is not in the snapshot, and that is
    refused once, by name, as a face (``faces.face_problem``) — the same answer
    :data:`METHOD_PROVIDES` gives for an external-node method, whose node makes the final check.
    """
    from_face = face_term_requirement(term, params)
    if from_face is None:
        return None
    return tuple(from_face) + TERM_NEEDS[term][1]


def term_needs_sentence(term: str, params: Dict[str, Any]) -> str:
    """What would supply ``term``, naming the configured face's own parameters.

    "expected_radius_m is missing" is not actionable; "the grasp target has no usable
    shape and dims" is — and for a face that is not the default, neither is any use unless
    the sentence says which face and which of ITS parameters.
    """
    sentence = TERM_NEEDS[term][2]
    if term != 'visible_fraction':
        return sentence
    face = face_of(params)
    needs = term_param_needs(term, params)
    if needs:
        sentence += (f'. The configured face is "{face}", which is sized by: '
                     f'{", ".join(needs)}')
    elif needs is None:
        sentence += (f'. The configured face "{face}" is not in TSA\'s snapshot of '
                     f'trainit_perception\'s face registry, so its parameters were not judged')
    return sentence


def computable_terms(method: DetectionMethod, params: Dict[str, Any]) -> set:
    """Which terms a detector of this METHOD, configured with these PARAMS, can compute.

    trainit_perception's own ``computable_terms`` over its declarations (ADR-0021, amendment
    M2.2): a term needs what it reads from the detection and its parameters above 0, and a
    FACE-SIZED term also needs the configured face to enable it and every one of that face's
    parameters set (test_tp_registry.py holds the two to each other when TP is installed).
    An external-node method declares no capabilities to TSA, so every capability is assumed
    present for it and only the parameters are judged: refusing a rule on a capability TSA
    merely failed to know would be refusing the expert seam. A face that is not in the snapshot
    is not judged either: it is refused once, by name, as a face.
    """
    provides = METHOD_PROVIDES.get(DetectionMethod(method))
    unknown = provides is None
    have = set(provides or ())
    out = set()
    for term, (needs_provides, _, _) in TERM_NEEDS.items():
        if not unknown and not set(needs_provides) <= have:
            continue
        if term in FACE_SIZED_TERMS and face_enables(term, params) is False:
            continue                     # the configured face does not produce this term at all
        needs_params = term_param_needs(term, params)
        if needs_params is None or all(_positive(params.get(n)) for n in needs_params):
            out.add(term)
    return out


def computable_rules(method: DetectionMethod,
                     params: Dict[str, Any]) -> Tuple[CandidateRule, ...]:
    """Every rule this detector can compute as configured — what a configurator offers."""
    have = computable_terms(method, params)
    return tuple(rule for rule in CandidateRule
                 if set(CANDIDATE_RULE_TERMS[rule]) <= have)


def preferred_candidate_rule(method: DetectionMethod,
                             params: Dict[str, Any]) -> CandidateRule:
    """The best rule this detector can compute, in the measured preference order — the
    value the GUI pre-selects (ADR-0017 Decision 4: visible wherever a face area is
    available, box overlap as the fallback)."""
    from .enums import DEFAULT_RULE_PREFERENCE
    offered = set(computable_rules(method, params))
    for rule in DEFAULT_RULE_PREFERENCE:
        if rule in offered:
            return rule
    return CandidateRule.AUTO


def effective_candidate_rule(det: 'DetectorSpec') -> Optional[CandidateRule]:
    """The rule this detector will actually run with: the field, or a HAND-EDITED
    ``params['candidate_rule']`` when the field is unset.

    ``params`` is a verbatim runtime profile that a configurator may edit by hand, so the
    key can arrive that way — and a rule TSA cannot see is a rule TSA cannot refuse, which
    would reopen exactly the hole ADR-0017 Decision 3 closes. An unknown token raises
    ``ValueError`` here, the same answer the node gives for the same file.
    """
    if det.candidate_rule is not None:
        return det.candidate_rule
    raw = det.params.get('candidate_rule')
    if raw in (None, ''):
        return None
    try:
        return CandidateRule(raw)
    except ValueError:
        raise ValueError(
            f'detector "{det.name}": params["candidate_rule"] = {raw!r} is not a rule any '
            f'detector publishes, so nothing could rank by it — and the published order IS '
            f'the ranking (ADR-0016 Decision 1), so an unknown rule cannot simply be '
            f'ignored. Accepted: '
            f'{", ".join(r.value for r in CandidateRule)}') from None


def missing_candidate_terms(rule: Optional[CandidateRule], method: DetectionMethod,
                            params: Dict[str, Any]) -> Tuple[str, ...]:
    """The terms ``rule`` needs that this detector cannot compute, in the rule's own order.

    Pure, and shared by the GUI's live hint and the generation-time refusal, so the two can
    never disagree about what is computable.
    """
    if rule is None:
        return ()
    have = computable_terms(method, params)
    return tuple(term for term in CANDIDATE_RULE_TERMS[CandidateRule(rule)]
                 if term not in have)


def candidate_rule_refusal(detector: str, rule: CandidateRule, method: DetectionMethod,
                           missing: Tuple[str, ...],
                           params: Optional[Dict[str, Any]] = None) -> str:
    """The message for a rule whose terms are not computable (ADR-0017 Decision 3).

    The sentence per term comes from :func:`term_needs_sentence`, so a missing FACE
    parameter is named by its real name and its real face — "set expected_half_b_m on the
    rect face" instead of a sentence written when ``disc`` was the only face there was.
    """
    needs = '; '.join(f'{term} needs {term_needs_sentence(term, params or {})}'
                      for term in missing)
    offered = computable_rules(method, params or {})
    return (f'detector "{detector}": candidate_rule "{CandidateRule(rule).value}" needs the '
            f'term(s) {", ".join(missing)}, which this detector cannot compute as '
            f'configured — {needs}. The PUBLISHED ORDER of the detections IS the ranking '
            f'(ADR-0016 Decision 1), so a rule whose terms nobody can compute would leave '
            f'the cell picking in whatever order the model returned, under the name of a '
            f'ranking — and that looks like a perception fault, not a configuration one. '
            f'Rules available for this detector as configured: '
            f'{", ".join(r.value for r in offered)}.')


class CameraSpec(BaseModel):
    """The cell camera: frames, topics, and how the robot description gains it."""

    optical_frame: str = 'camera_color_optical_frame'
    # The camera's root link in the URDF — used for the SRDF pair (mount↔camera
    # disabled "Never"; every camera-vs-arm pair stays CHECKED so planning avoids it).
    link: str = 'camera_link'
    rgb_topic: str = '/camera/color/image_raw'
    depth_topic: str = '/camera/depth/image_rect_raw'
    camera_info_topic: str = '/camera/color/camera_info'
    points_topic: str = '/camera/depth/color/points'
    # URDF variant rewrite (both copied packages): the include whose ``filename``
    # equals ``replace_include`` is swapped for ``with_include``. Data, not a generator
    # rule — so the generator stays robot-agnostic and the project carries the paths.
    replace_include: Optional[str] = None
    with_include: Optional[str] = None
    # sim builds the coloured cloud (depth_image_proc, isaac branch only); in real the
    # vendor driver already publishes ``points_topic``.
    synthetic_cloud_in_sim: bool = True


class DetectorSpec(BaseModel):
    """One named detector: a method plus its tuned parameters.

    ``params`` mirrors the runtime profile VERBATIM (``trainit_perception``'s
    ``defaults()`` keys, ``class_id`` included) — the live tuner and the generated
    ``perception.yaml`` serialize the same dict, which is the "TSA generates what you
    tuned" guarantee.
    """

    name: str                                   # -> /perception/<name>/detections
    method: DetectionMethod = DetectionMethod.COLOR_MASK
    params: Dict[str, Any] = Field(default_factory=dict)
    # WHICH candidate goes out first, and therefore THE RANKING (ADR-0017 Decision 1:
    # the rank is computed BY THE DETECTOR, as a detector parameter). Enum-validated the
    # way ``method`` is, because a typo here is not a worse ranking, it is NO ranking.
    #
    # None (the default and the migration path) writes NO key into perception.yaml, the
    # way ``blur_px`` and the depth bounds already land only when they are on: the file
    # is compared YAML-semantically against the golden, so a key that always appeared
    # would re-bless every published bundle. An unset rule leaves the runtime's own
    # default in charge; a SET one is carried verbatim and refused at generation time if
    # its terms are not computable (see ``missing_candidate_params``).
    candidate_rule: Optional[CandidateRule] = None
    # Input overrides; None => the camera's topics.
    rgb_topic: Optional[str] = None
    depth_topic: Optional[str] = None
    camera_info_topic: Optional[str] = None
    # Trigger model (D-015): continuous at rate_hz keeps the pipeline warm (default,
    # proof-2 validated); on demand the same node answers the std_srvs/Trigger service.
    continuous: bool = True
    rate_hz: float = 10.0
    # HOW an external-node method is LAUNCHED (D-015's second kind of method): the
    # package, the executable, an optional parameter file and the remappings its contract
    # needs. ``None`` for a pure-python method, and ``None`` is also a legitimate answer
    # for an external-node method whose node you start yourself — in which case the
    # generator says so instead of silently emitting nothing. See
    # :mod:`trainit_setup_assistant.model.external_node`.
    external_node: Optional[ExternalNodeSpec] = None

    @property
    def class_id(self) -> str:
        return str(self.params.get('class_id', 'object'))

    @property
    def gives_orientation(self) -> bool:
        """Whether this detector publishes a REAL object orientation. A colour mask
        (and any 3D-only method) cannot — it publishes identity — so the GUI greys
        the "align to detected object" policy for it (D-016). Becomes method
        metadata when 6-DoF methods (fiducial/PCL/learned) land."""
        return False

    @property
    def emits_node(self) -> bool:
        """True when the assistant generates a detector_node for this method — only
        the pure-python methods the runtime registry actually knows. CUSTOM and
        PCL_CLUSTER are external-node methods: their node publishes the contract
        itself, and generating a detector_node for them would crash at startup
        (make_detector raises KeyError on an unknown method)."""
        return self.method is DetectionMethod.COLOR_MASK

    @property
    def launches_external_node(self) -> bool:
        """True when the bundle must start this detector's OWN node (D-015).

        The complement of :attr:`emits_node` for the methods that have a launch block:
        ``emits_node`` says "TSA runs ``trainit_perception``'s detector_node for it",
        this says "TSA runs the node the project named". Both false means the detector is
        recorded in ``perception.yaml`` and launched by nobody, which is a legitimate
        choice (a vendor stack with its own bring-up) and is reported, never assumed.
        """
        return self.external_node is not None and not self.emits_node

    def input_topics(self, camera: Optional['CameraSpec']) -> Dict[str, str]:
        """The camera contract this detector actually reads: role -> topic.

        The ONE resolution of the per-detector override -> the cell camera -> the stock
        RealSense names. Both consumers call it — the ``input:`` block of the generated
        ``perception.yaml`` and the ``topic_parameters`` of an external node's launch
        entry — because the file TSA writes and the node TSA launches disagreeing about
        which camera they are looking at is a fault no log would ever show.
        """
        return {
            'rgb': self.rgb_topic or (camera.rgb_topic if camera
                                      else DEFAULT_INPUT_TOPICS['rgb']),
            'depth': self.depth_topic or (camera.depth_topic if camera
                                          else DEFAULT_INPUT_TOPICS['depth']),
            'camera_info': self.camera_info_topic or (
                camera.camera_info_topic if camera
                else DEFAULT_INPUT_TOPICS['camera_info']),
        }


class VisionBinding(BaseModel):
    """Binds a waypoint to a detector: SetWaypointFromDetection, as data (D-016:
    vision is a PROPERTY of the Move — the waypoint's position comes from the
    detection at run time; the captured pose stays as the recorded fallback).

    ``orientation``: ``keep`` (the waypoint's own), ``detected``, ``from:<waypoint>``
    (borrow another waypoint's orientation — how a joint-named approach pose is
    promoted to tcp), or a literal ``qx;qy;qz;qw``.
    """

    detector: str
    dx: float = 0.0                             # metres, base-frame axes (D-016)
    dy: float = 0.0
    dz: float = 0.0
    orientation: str = 'keep'


class RelativeBinding(BaseModel):
    """A waypoint DERIVED from another step's FINAL pose (D-017): position =
    reference + dx/dy/dz (base-frame metres), orientation = rpy delta (DEGREES,
    extrinsic base-frame) composed onto the reference orientation. Resolved at RUN
    TIME by ``SetWaypointRelative`` from the blackboard — so it follows the
    reference wherever vision sent it this cycle. The canonical use: the retreat
    (post_pick = pick + 8 cm) without a second detection, since at retreat time the
    arm occludes the object anyway. Mutually exclusive with ``Waypoint.vision``."""

    step: str                                   # the reference waypoint name
    dx: float = 0.0
    dy: float = 0.0
    dz: float = 0.0
    droll: float = 0.0                          # degrees
    dpitch: float = 0.0
    dyaw: float = 0.0


class PerceptionSpec(BaseModel):
    camera: Optional[CameraSpec] = None
    detectors: List[DetectorSpec] = Field(default_factory=list)
    # Settle barrier after a scene-mutating block that precedes DetectObject in the
    # cycle: arrival freshness is not content freshness (measured 0.4 s on the cell;
    # 1500 ms = 3.7x margin). Emitted as <Sleep> right after <ResetScene/>.
    settle_ms: int = Field(default=1500, ge=0)
    detect_timeout_ms: int = Field(default=3000, ge=1)

    def detector_by_name(self, name: str) -> Optional[DetectorSpec]:
        for d in self.detectors:
            if d.name == name:
                return d
        return None
