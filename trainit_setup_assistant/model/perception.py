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
    # TrainIt D0.8 (D0_DESIGN.md §4 D3): the LOGICAL sensor id data collection names this camera by, and
    # the hardware-layer camera PROFILE model (trainit_hardware_layer profiles/camera/<model>.yaml) its
    # nominal characteristics come from. Both optional; written only when set (a project without them saves
    # byte-identical). Intrinsics, units, calibration and extrinsics never live here: the recorder records
    # what the sensor published.
    id: Optional[str] = None
    profile: Optional[str] = None


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
    # vNext M4, method surface_grasp only: the TP detector the provider fits its surfaces on --
    # ``{method: <a TP method that exposes masks>, params: {...}}``, written verbatim into perception.yaml.
    # ``params`` then holds the PROVIDER's own parameters (eef, site, contact_radius_m, approach_cone_deg,
    # standoff_m, ...), held to TP's committed schema (``surface_grasp_problems``).
    segmentation: Optional[Dict[str, Any]] = None
    # vNext M4.5, method surface_grasp only: the workspace ELIGIBILITY filter (TP >= 0.7.0) -- ``{roi_px: [[u, v],
    # ...], min_inside, z_min_m, z_max_m}``, written verbatim into the entry beside ``segmentation``. None = none.
    eligibility: Optional[Dict[str, Any]] = None
    # vNext M4.5: WHAT this entry's node publishes, when it is not the perception contract's detections. Only
    # ``instances`` (:data:`PRODUCES_INSTANCES`) exists: an EXTERNAL SEGMENTER (method ``custom`` + an
    # ``external_node`` block) publishing an instance image ``/perception/<name>/instances`` (16UC1) and its labels
    # ``/perception/<name>/instance_info`` (vision_msgs/Detection2DArray) -- what a grasp provider's
    # ``segmentation: {method: instance_topic, params: {source: <name>}}`` reads. Never detections, so no detect
    # step, capture or ranking consumes it. None = the method's own output (byte-identical when unset).
    produces: Optional[str] = None

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
    def is_grasp_provider(self) -> bool:
        """True for a trainit_perception GRASP PROVIDER (vNext M4): the bundle runs the provider's own node,
        which publishes grasp CANDIDATES on /perception/<name>/grasp_candidates, never detections."""
        return self.method in GRASP_PROVIDER_METHODS

    @property
    def produces_instances(self) -> bool:
        """True for an EXTERNAL SEGMENTER entry (vNext M4.5): its node publishes an instance image, never
        detections."""
        return self.produces == PRODUCES_INSTANCES

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


# vNext M4.5: the methods whose node is a trainit_perception GRASP PROVIDER (TSA launches TP's own node, named in
# TP's vocabulary, ``grasp_providers.<method>.node``)
GRASP_PROVIDER_METHODS = (DetectionMethod.SURFACE_GRASP, DetectionMethod.SCRIPTED_GRASP)
# vNext M4.5: the one value of ``DetectorSpec.produces`` -- an external segmenter's instance image
PRODUCES_INSTANCES = 'instances'


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


# --- vNext M4: the SurfaceGrasp provider's configuration, held to TP's committed schema -----------------------
def grasp_provider_schema(name: str) -> Optional[Dict[str, Any]]:
    """TP's record of a grasp provider (TSA's snapshot, ADR-0021): node, modes, segmentation methods,
    parameters. None when the snapshot has no such provider."""
    return (_TP.get('grasp_providers') or {}).get(name)


def _number(v) -> bool:
    import math
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def surface_grasp_problems(det: 'DetectorSpec') -> List[str]:
    """Why a ``surface_grasp`` detector entry cannot be generated, as sentences (empty = it can). The same
    refusals the provider's node makes at start-up, made before a bundle exists: a provider without an end
    effector, an envelope or a valid standoff never starts, so a bundle carrying one would fail three layers
    from its cause."""
    where = f'detector "{det.name}" (surface_grasp)'
    schema = grasp_provider_schema(DetectionMethod.SURFACE_GRASP.value)
    if schema is None:
        return [f'{where}: TSA\'s snapshot of trainit_perception has no surface_grasp provider: refresh the catalog '
                'from trainit_perception >= 0.6.0']
    out: List[str] = []
    if det.external_node is not None:
        out.append(f'{where}: a grasp provider runs trainit_perception\'s own node: no external_node block')
    if det.candidate_rule is not None or 'candidate_rule' in det.params:
        out.append(f'{where}: a grasp provider ranks its candidates by its own terms: no candidate_rule')
    seg = det.segmentation
    if not isinstance(seg, dict) or not seg.get('method'):
        out.append(f'{where}: needs segmentation: {{method: <one of {", ".join(schema["segmentation_methods"])}>, '
                   'params: {...}}} -- the detector whose masks the surfaces are fitted on')
    else:
        stray = sorted(set(seg) - {'method', 'params'})
        if stray:
            out.append(f'{where}: segmentation has unknown key(s) {", ".join(stray)} (method, params)')
        if seg['method'] not in schema['segmentation_methods']:
            out.append(f'{where}: segmentation.method {seg["method"]!r} exposes no masks (accepted: '
                       f'{", ".join(schema["segmentation_methods"])})')
        elif seg['method'] in (schema.get('segmentation_parameters') or {}):
            # vNext M4.5: a segmentation SOURCE that is not a TP detector (`instance_topic`: an external segmenter's
            # instance image), held to its own schema -- the same rules as the node's InstanceConfig
            out += _source_problems(where, seg['method'], seg.get('params') or {},
                                    schema['segmentation_parameters'][seg['method']]['parameters'])
        else:
            known = (_TP['detector_methods'][seg['method']].get('parameters') or {})
            unknown = sorted(k for k in (seg.get('params') or {}) if k not in known)
            if unknown:
                out.append(f'{where}: segmentation.params {", ".join(unknown)} are not {seg["method"]} parameters')
            for key, v in (seg.get('params') or {}).items():           # held to the method's schema, like params
                row = known.get(key)
                if row is None:
                    continue
                t = row['type']
                if t == 'range':
                    ok = (isinstance(v, list) and len(v) == 2 and all(isinstance(x, int) and not isinstance(x, bool)
                                                                     for x in v))
                    vals = v if ok else []
                elif t == 'int':
                    ok, vals = isinstance(v, int) and not isinstance(v, bool), [v]
                elif t == 'float':
                    ok, vals = _number(v), [v]
                else:
                    ok, vals = isinstance(v, str), []
                if not ok:
                    out.append(f'{where}: segmentation.params.{key} = {v!r} is not a valid {t}')
                    continue
                if any(('min' in row and x < row['min']) or ('max' in row and x > row['max']) for x in vals):
                    out.append(f'{where}: segmentation.params.{key} = {v!r} is outside [{row.get("min")}, '
                               f'{row.get("max")}]')
    if det.eligibility is not None:
        out += eligibility_problems(det)
    params = schema['parameters']
    for key in sorted(set(det.params) - set(params)):
        out.append(f'{where}: unknown parameter {key!r} (the provider refuses it)')
    for key, row in params.items():
        if key not in det.params:
            if row.get('required'):
                out.append(f'{where}: {key} is required ({row["type"]}): the provider has no default for it')
            continue
        v, t = det.params[key], row['type']
        if t == 'str':
            ok = isinstance(v, str) and (bool(v) or not row.get('required'))
        elif t == 'int':
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif t == 'float':
            ok = _number(v)
        else:                                                    # vec3
            ok = (isinstance(v, list) and len(v) == 3 and all(_number(x) for x in v)
                  and sum(x * x for x in v) ** 0.5 > 1e-9)
        if not ok:
            out.append(f'{where}: {key} = {v!r} is not a valid {t}')
            continue
        if 'choices' in row and v not in row['choices']:
            out.append(f'{where}: {key} = {v!r} is not one of {", ".join(row["choices"])}')
        if t in ('int', 'float') and (('min' in row and v < row['min']) or ('max' in row and v > row['max'])):
            out.append(f'{where}: {key} = {v!r} is outside [{row.get("min")}, {row.get("max")}]')
    # the semantic refusals the provider makes (M4_DESIGN.md §8-§10) -- the SAME rules as TP's
    # SurfaceGraspConfig.from_params, so a bundle TSA generates is never refused by its node at start-up (review M4)
    def val(key):
        return det.params.get(key, (params.get(key) or {}).get('default'))

    def num(key, default=0.0):
        v = val(key)
        return float(v) if _number(v) else default
    if _number(det.params.get('inlier_mm')) and not det.params['inlier_mm'] > 0:
        out.append(f'{where}: inlier_mm must be > 0')
    view = det.params.get('max_view_angle_deg')
    if _number(view) and not 0 < view <= 90:
        out.append(f'{where}: max_view_angle_deg {view} must be in (0, 90]')
    if not (0 <= num('z_min_m') < num('z_max_m', 1.0)):
        out.append(f'{where}: the depth band [z_min_m, z_max_m] = [{val("z_min_m")}, {val("z_max_m")}] is empty')
    face = val('grasp_face') or 'disc'
    faces = _TP.get('faces') or {}
    if face not in faces:
        out.append(f'{where}: grasp_face {face!r} is not a face trainit_perception registers '
                   f'({", ".join(sorted(faces))})')
    if num('tool_radius_m') > 0 and not num('tool_reach_mm') > 0:
        out.append(f'{where}: tool_radius_m > 0 needs tool_reach_mm > 0 (the clearance term needs the grip window)')
    if num('min_clearance') > 0 and not num('tool_radius_m') > 0:
        out.append(f'{where}: min_clearance > 0 needs tool_radius_m > 0 (the clearance term is not evaluated)')
    if num('min_visible_fraction') > 0 and face in faces and not all(
            num(k) > 0 for k in faces[face].get('params') or ()):
        out.append(f'{where}: min_visible_fraction > 0 needs the size of face {face!r} '
                   f'({", ".join(faces[face].get("params") or ())}): visibility is not evaluated without it')
    axis, xref = val('approach_cone_axis'), val('reference_x_axis')
    if isinstance(axis, list) and isinstance(xref, list) and len(axis) == len(xref) == 3 \
            and all(_number(x) for x in axis + xref):
        na = sum(x * x for x in axis) ** 0.5
        nx = sum(x * x for x in xref) ** 0.5
        if na > 1e-9 and nx > 1e-9:
            c = abs(sum(a * b for a, b in zip(axis, xref))) / (na * nx)
            if (1 - min(1.0, c) ** 2) ** 0.5 < 1e-6:
                out.append(f'{where}: reference_x_axis is parallel to approach_cone_axis: the rotation about the '
                           'approach would be undefined')
    cone = det.params.get('approach_cone_deg')
    if _number(cone) and not 0 < cone < 90:
        out.append(f'{where}: approach_cone_deg {cone} must be in (0, 90) deg (an envelope, never a default)')
    if _number(det.params.get('contact_radius_m')) and not det.params['contact_radius_m'] > 0:
        out.append(f'{where}: contact_radius_m must be > 0 (the contact the grasp site needs on the surface)')
    s = det.params.get('standoff_m')
    lo = det.params.get('standoff_min_m', (params.get('standoff_min_m') or {}).get('default'))
    hi = det.params.get('standoff_max_m', (params.get('standoff_max_m') or {}).get('default'))
    if _number(s) and _number(lo) and _number(hi) and not (0 < lo <= s <= hi):
        out.append(f'{where}: standoff_m {s} must lie in [standoff_min_m, standoff_max_m] = [{lo}, {hi}] (> 0): '
                   'never clamped')
    return out


# --- vNext M4.5: instance_topic segmentation, workspace eligibility, the scripted provider -------------------------
_ROS_NAME_TOKEN = __import__('re').compile(r'^[A-Za-z][A-Za-z0-9_]*$')


def _source_problems(where: str, method: str, params: Dict[str, Any], rows: Dict[str, Any]) -> List[str]:
    """A segmentation SOURCE's params (``instance_topic``) against TP's schema for it -- the same refusals as the
    node's ``InstanceConfig.from_params``: the source one ROS name token, the classes a list of non-empty names,
    the numbers in their ranges."""
    out: List[str] = []
    unknown = sorted(k for k in params if k not in rows)
    if unknown:
        out.append(f'{where}: segmentation.params {", ".join(unknown)} are not {method} parameters')
    for key, row in rows.items():
        if key not in params:
            if row.get('required'):
                out.append(f'{where}: segmentation.params.{key} is required ({row["type"]}): {method} has no '
                           'default for it')
            continue
        v, t = params[key], row['type']
        if t == 'str':
            ok = isinstance(v, str) and (not row.get('required') or bool(_ROS_NAME_TOKEN.match(v)))
        elif t == 'str_list':
            ok = v is None or (isinstance(v, list) and all(isinstance(x, str) and x for x in v))
        elif t == 'int':
            ok = isinstance(v, int) and not isinstance(v, bool)
        elif t == 'float':
            ok = _number(v)
        else:
            ok = False                                       # a type this TSA does not know how to check
        if not ok:
            out.append(f'{where}: segmentation.params.{key} = {v!r} is not a valid {t}'
                       + (' (one ROS name token: the segmenter\'s entry name)' if key == 'source' else ''))
            continue
        if t in ('int', 'float') and (('min' in row and v < row['min']) or ('max' in row and v > row['max'])):
            out.append(f'{where}: segmentation.params.{key} = {v!r} is outside [{row.get("min")}, {row.get("max")}]')
    return out


def _cross(p1, p2, q1, q2) -> bool:
    """Whether the closed segments p1p2 and q1q2 share a point (TP's eligibility rule, the same tolerances)."""
    def orient(a, b, c):
        v = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        return 0 if abs(v) < 1e-12 else (1 if v > 0 else -1)

    def on(a, b, c):
        return (min(a[0], b[0]) - 1e-12 <= c[0] <= max(a[0], b[0]) + 1e-12
                and min(a[1], b[1]) - 1e-12 <= c[1] <= max(a[1], b[1]) + 1e-12)

    o1, o2, o3, o4 = orient(p1, p2, q1), orient(p1, p2, q2), orient(q1, q2, p1), orient(q1, q2, p2)
    if o1 != o2 and o3 != o4:
        return True
    return ((o1 == 0 and on(p1, p2, q1)) or (o2 == 0 and on(p1, p2, q2))
            or (o3 == 0 and on(q1, q2, p1)) or (o4 == 0 and on(q1, q2, p2)))


def polygon_px_problems(roi, width: Optional[int] = None, height: Optional[int] = None) -> List[str]:
    """Why ``roi`` is not a usable workspace polygon in colour pixels (empty = it is): >= 3 vertices of two finite
    numbers, no two consecutive ones equal, simple (no two non-adjacent edges meet), at least 1 px^2 -- the rules of
    TP's ``eligibility.polygon_problems``. With the colour image's size, every vertex on the image too. TSA has no
    camera resolution in a project today, so generation passes none: the node checks it against its CameraInfo."""
    if not isinstance(roi, list) or len(roi) < 3:
        return [f'roi_px must be a list of >= 3 [u, v] vertices, got {roi!r}']
    pts = []
    for i, p in enumerate(roi):
        if not isinstance(p, list) or len(p) != 2 or not all(_number(x) for x in p):
            return [f'roi_px vertex {i} must be two finite numbers [u, v], got {p!r}']
        pts.append((float(p[0]), float(p[1])))
    out: List[str] = []
    n = len(pts)
    same = next((i for i in range(n) if pts[i] == pts[(i + 1) % n]), None)
    if same is not None:
        out.append(f'roi_px vertices {same} and {(same + 1) % n} coincide')
    else:
        crossing = next(((i, j) for i in range(n) for j in range(i + 2, n) if not (i == 0 and j == n - 1)
                         and _cross(pts[i], pts[(i + 1) % n], pts[j], pts[(j + 1) % n])), None)
        if crossing is not None:
            out.append(f'roi_px edges {crossing[0]} and {crossing[1]} intersect: the polygon must be simple')
        else:
            area = abs(sum(pts[i][0] * pts[(i + 1) % n][1] - pts[(i + 1) % n][0] * pts[i][1]
                           for i in range(n))) / 2
            if area < 1.0:
                out.append(f'roi_px encloses {area:g} px^2: a degenerate polygon (collinear vertices) admits no region')
    if width is not None and height is not None:
        bad = [i for i, (u, v) in enumerate(pts) if not (0.0 <= u <= width and 0.0 <= v <= height)]
        if bad:
            out.append(f'roi_px vertex(es) {bad} lie outside the {width}x{height} colour image')
    return out


def eligibility_problems(det: 'DetectorSpec', width: Optional[int] = None,
                         height: Optional[int] = None) -> List[str]:
    """Why a provider entry's ``eligibility`` block would be refused by its node (empty = it would not): a mapping
    of the provider's eligibility parameters (TP's committed schema), ``roi_px`` required and a usable polygon,
    ``min_inside`` in (0, 1], the depth band ordered: 0 <= z_min_m < z_max_m <= 20."""
    where = f'detector "{det.name}" ({det.method.value})'
    schema = grasp_provider_schema(det.method.value) or {}
    section = schema.get('eligibility') or {}
    if not section:
        return [f'{where}: this provider has no workspace eligibility filter (trainit_perception\'s vocabulary '
                'declares none): remove the eligibility block']
    block = det.eligibility
    if not isinstance(block, dict):
        return [f'{where}: eligibility must be a mapping {{roi_px, min_inside, z_min_m, z_max_m}}, got {block!r}']
    rows = section['parameters']
    out: List[str] = []
    unknown = sorted(set(block) - set(rows))
    if unknown:
        out.append(f'{where}: eligibility has unknown key(s) {", ".join(map(str, unknown))} '
                   f'(have: {", ".join(rows)})')
    if 'roi_px' not in block:
        out.append(f'{where}: eligibility.roi_px is required: the workspace polygon in colour pixels (no default: a '
                   'workspace is the cell\'s)')
    else:
        out += [f'{where}: eligibility: {p}' for p in polygon_px_problems(block['roi_px'], width, height)]
    mi = block.get('min_inside', 0.9)
    if not _number(mi) or not 0.0 < mi <= 1.0:
        out.append(f'{where}: eligibility.min_inside must be in (0, 1], got {mi!r}')
    zmin = block.get('z_min_m', (rows.get('z_min_m') or {}).get('default', 0.0))
    zmax = block.get('z_max_m', (rows.get('z_max_m') or {}).get('default', 20.0))
    if not (_number(zmin) and _number(zmax) and 0.0 <= zmin < zmax <= 20.0):
        out.append(f'{where}: eligibility: the depth band [z_min_m, z_max_m] = [{zmin!r}, {zmax!r}] must satisfy '
                   '0 <= z_min_m < z_max_m <= 20')
    return out


def scripted_grasp_problems(det: 'DetectorSpec') -> List[str]:
    """Why a ``scripted_grasp`` entry cannot be generated (empty = it can): the node's own start-up refusals made
    before a bundle exists -- its params held to TP's schema (scenario_file and frame_id non-empty, after_end one
    of its choices, the two axes valid and not parallel), and the entry shape it accepts: ``{method, params,
    continuous: false}`` only (it reads no camera, has no rate and answers the capture trigger only).

    The scenario FILE is not read: its candidates' eef / site, poses and steps are validated by the node at
    start-up (TSA never parses a TP scenario format). A relative path is the node's working directory's, so an
    absolute one is asked for."""
    where = f'detector "{det.name}" (scripted_grasp)'
    schema = grasp_provider_schema(DetectionMethod.SCRIPTED_GRASP.value)
    if schema is None:
        return [f'{where}: TSA\'s snapshot of trainit_perception has no scripted_grasp provider: refresh the '
                'catalog from trainit_perception >= 0.7.0']
    out: List[str] = []
    if det.external_node is not None:
        out.append(f'{where}: a grasp provider runs trainit_perception\'s own node: no external_node block')
    if det.candidate_rule is not None:
        out.append(f'{where}: a grasp provider ranks its candidates by its own terms: no candidate_rule')
    stray = sorted(k for k in ('segmentation', 'eligibility', 'produces', 'rgb_topic', 'depth_topic',
                               'camera_info_topic') if getattr(det, k) is not None)
    if det.rate_hz != DetectorSpec.__fields__['rate_hz'].default:
        stray.append('rate_hz')
    if stray:
        out.append(f'{where}: {", ".join(stray)}: a scripted provider reads no camera and no segmenter, and has no '
                   'rate -- its entry is {method, params, continuous: false}')
    # `continuous` is not judged: the entry is always written `continuous: false` (the node refuses true), and a
    # detect step on it must capture (`trigger: true`), which the task program refuses otherwise (R-M3-03)
    params = schema['parameters']
    for key in sorted(set(det.params) - set(params)):
        out.append(f'{where}: unknown parameter {key!r} (the provider refuses it)')
    for key, row in params.items():
        if key not in det.params:
            if row.get('required'):
                out.append(f'{where}: {key} is required ({row["type"]}): the provider has no default for it')
            continue
        v, t = det.params[key], row['type']
        if t == 'str':
            ok = isinstance(v, str) and (bool(v) or not row.get('required'))
        elif t == 'vec3':
            ok = (isinstance(v, list) and len(v) == 3 and all(_number(x) for x in v)
                  and sum(x * x for x in v) ** 0.5 > 1e-9)
        else:
            ok = isinstance(v, (int, float)) and not isinstance(v, bool)
        if not ok:
            out.append(f'{where}: {key} = {v!r} is not a valid {t}')
            continue
        if 'choices' in row and v not in row['choices']:
            out.append(f'{where}: {key} = {v!r} is not one of {", ".join(row["choices"])}')
    scenario = det.params.get('scenario_file')
    if isinstance(scenario, str) and scenario and not scenario.startswith('/'):
        out.append(f'{where}: scenario_file {scenario!r} must be an absolute path: the node opens it from its own '
                   'working directory, which the bundle does not fix')
    axis = det.params.get('approach_cone_axis', (params.get('approach_cone_axis') or {}).get('default'))
    xref = det.params.get('reference_x_axis', (params.get('reference_x_axis') or {}).get('default'))
    if isinstance(axis, list) and isinstance(xref, list) and len(axis) == len(xref) == 3 \
            and all(_number(x) for x in axis + xref):
        na, nx = sum(x * x for x in axis) ** 0.5, sum(x * x for x in xref) ** 0.5
        if na > 1e-9 and nx > 1e-9:
            c = abs(sum(a * b for a, b in zip(axis, xref))) / (na * nx)
            if (1 - min(1.0, c) ** 2) ** 0.5 < 1e-6:
                out.append(f'{where}: reference_x_axis is parallel to approach_cone_axis: the rotation about the '
                           'approach would be undefined')
    return out


def grasp_provider_entry_problems(det: 'DetectorSpec') -> List[str]:
    """The provider-specific refusals of one grasp-provider entry, by its method."""
    if det.method is DetectionMethod.SCRIPTED_GRASP:
        return scripted_grasp_problems(det)
    return surface_grasp_problems(det)
