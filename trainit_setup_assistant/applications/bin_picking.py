"""bin_picking application template — the EMPTYING LOOP (ADR-0016 Decision 4).

A linear ``MoveWaypoint`` sequence cannot empty a bin, and the reason is measured rather
than supposed. On the reference cell, over a 25-capsule bin:

- **K = 3,33 candidates tried per object delivered** — one pick in three does not succeed
  on the first candidate, so without a retry two picks in three stop at the first refusal;
- **"the cup closed" and "the object came up" differ on 1 lift in 22** — so the lift needs
  a verification, not an assumption;
- **the bin stalls twice before it is empty**, each stall recovered by agitation worth six
  capsules out of twenty-five. "No plan" and "the plans do not work" are the same stall to
  an operator, and treating only the first is why an early measurement reported N = 15
  where the recovered one reports N = 21.

So this template emits a LOOP, and the three things above are its three features:

.. code-block:: xml

    <Repeat num_cycles="N">                   <!-- the emptying loop -->
      <Fallback>                              <!-- pick, or recover from the stall -->
        <Sequence>                            <!-- the pick -->
          <DetectObject keep="tries"/>        <!-- ranked, best first (Decision 1) -->
          <Script code="candidate := -1"/>
          <RetryUntilSuccessful num_attempts="tries">
            <Sequence>                        <!-- ONE candidate, in rank order -->
              <Script code="candidate := candidate + 1"/>
              <SetWaypointFromDetection index="{candidate}"/>
              approach / grasp / CloseGripper / retreat
              <CheckRobotState/>              <!-- did it actually come up -->
            </Sequence>
          </RetryUntilSuccessful>
          place / OpenGripper
        </Sequence>
        <Fallback>                            <!-- the bin stalled -->
          <recovery>                          <!-- cell I/O: an agitator, ADR-0013 -->
          <Log message="bin empty"/>
        </Fallback>
      </Fallback>
    </Repeat>

**How the index advances, with no new runtime node.** ADR-0016 Decision 2 gives
``DetectObject`` a ``keep`` port (it writes ``<out_key>.count`` and ``<out_key>.<i>.*``)
and ``SetWaypointFromDetection`` an ``index`` port — and nothing else, because nothing
else is needed: BehaviorTree.CPP v4's own ``Script`` node advances the blackboard key the
``index`` port reads, so "try the next candidate" is a decorator over a counter rather
than a control node TMR has to grow. ``candidate`` is set to ``-1`` before the retry and
incremented as the FIRST step of each attempt, so attempt 1 reads candidate 0.

**What is configuration, not code** (``ApplicationSpec.bin_picking``): the number of
tries, whether the lift is verified, and the recovery step. The RANK rule is configuration
too but it lives in perception (ADR-0014), and the standoff/retreat DISTANCES live on the
waypoints' own vision/relative bindings — none of them is duplicated here.

**What this template deliberately does NOT contain: the agitator.** A shake is a machine
with an amplitude, a frequency and a direction; in the twin it is one line of physics and
in a cell it is a signal on a PLC (ADR-0013). The template emits the *step* — an
``EnableProcess`` / ``Sleep`` / ``DisableProcess`` pulse on a named ``ProcessController``
channel, the semantic ``Cmd_``/``Sts_`` handshake ADR-0013 fixes — and what that channel
ACTUATES is cell I/O, owned by the cell, not by this application. The generated tree says
so at the point of use, so nobody reading it mistakes the step for the machine.
"""

from __future__ import annotations

from typing import List, Optional

from ..model import CanonicalProject, Waypoint
from ..model.enums import AppType, ExecutionMode, ToolActionKind, WaypointRole
from .eef_tree import canonical_device, verify_grasp
from .pick_and_place import MOVE_NODES
from .vision_guided_motion import VisionGuidedMotion

# The moves that belong to ONE CANDIDATE: everything between the approach and the lift is
# retried with the next candidate when the lift does not hold. Roles, not names, so the
# template works on any cell that labels its moves.
_CANDIDATE_ROLES = (WaypointRole.PRE_PICK, WaypointRole.PICK, WaypointRole.POST_PICK)

# The blackboard key the retry advances and ``SetWaypointFromDetection index="{…}"`` reads.
_INDEX_KEY = 'candidate'
# The end-effector contract's gate (ADR-0022 D5), canonical projects only: the key every EEF node
# (and DetectObject) writes its fault -- or, outside the retry, its reason -- into, and the flag
# that says the cycle holds an object. A fault, or any failure while holding, can never become
# "bin empty".
_GATE_KEY = 'eef_fault'
_HOLDING_KEY = 'holding'


class BinPicking(VisionGuidedMotion):
    """The emptying loop: ranked candidates, a retry over them, a verified lift, and a
    gated stall recovery whose MACHINE is cell I/O (ADR-0013), not this template.

    A subclass of :class:`VisionGuidedMotion` on purpose — every vision and step-relative
    check it already makes still holds here and is inherited rather than re-derived. What
    this class replaces is only the tree's SHAPE, which is the whole of ADR-0016
    Decision 4. The module docstring carries the measurement that forced each piece.
    """

    app_type = AppType.BIN_PICKING.value

    def tree_filename(self, project: CanonicalProject) -> str:
        return 'bin_picking.xml'

    # --- refusals ---------------------------------------------------------------
    def refusals(self, project: CanonicalProject) -> List[str]:
        return (super().refusals(project)
                + self._holding_check_conflicts(project.application.bin_picking, project)
                + self._dropped_detect_blocks(project))

    def _holding_check_conflicts(self, cfg, project: CanonicalProject = None) -> List[str]:
        """A holding check with no topic can never pass (vNext M0C #3).

        CheckRobotState has NO default topic: ``require_attached`` without
        ``attached_topic`` FAILS on every tick. The loop would try every candidate, report
        each one as a missed pick and deliver nothing -- three layers from the cause. A
        canonical project's check is VerifyGrasp on the device's own state (ADR-0022), which
        needs no topic: the end-effector contract's own rules (R17) hold its fields instead.
        """
        if project is not None and project.robot.gripper.contract is not None:
            return []
        if not (cfg.holding_check and not cfg.attached_topic):
            return []
        return [f'{self.app_type}: holding_check is on but attached_topic is empty. '
                'CheckRobotState has no default topic, so the check would FAIL on every '
                'pick and the loop would deliver nothing. Name the std_msgs/Bool that is '
                'true while the object is held (bin_picking.attached_topic), or switch '
                'holding_check off and accept that a closed gripper counts as a pick.']

    def _dropped_detect_blocks(self, project: CanonicalProject) -> List[str]:
        """The Detect blocks a CANONICAL emptying loop cannot emit: REFUSED, never dropped in
        silence (vNext G0, owner G0 section 6).

        The loop samples its own detector once per cycle, before the retry, and every other
        detector that guides a move there too -- but only one with no Detect block of its own
        (``_detect_lines``), and ``_move_block`` emits no block at its anchor. So the block of any
        detector but the loop's would vanish: no DetectObject, its moves not updated from the
        camera, the step-relative moves off them not re-derived. Not refused: the loop detector's
        own block (the loop samples that detector anyway; validate() notes it), a project with no
        loop (the linear vision tree emits every block at its anchor) and a LEGACY project, which
        keeps the drop (KNOWN_DEBT; ADR-0022 D9).
        """
        app = project.application
        if project.robot.gripper.contract is None or not app.detections:
            return []
        first, _last = self._candidate_bounds(project)
        det = self._loop_detector(project)
        if first is None or det is None:
            return []
        out: List[str] = []
        for pt in app.detections:
            if pt.detector == det:
                continue
            moves = [wp.name for wp in self._vision_waypoints(project)
                     if wp.vision.detector == pt.detector]
            derived = [wp.name for wp in self._relative_waypoints(project)
                       if wp.relative.step in moves]
            lost = [f'no DetectObject for "{pt.detector}"']
            if moves:
                lost.append(', '.join(f'"{n}"' for n in moves) + ' not updated from the camera')
            if derived:
                lost.append(', '.join(f'"{n}"' for n in derived) + ' not re-derived')
            out.append(
                f'{self.app_type}: the Detect block for "{pt.detector}" before '
                f'"{pt.before_waypoint}" is not supported in a canonical project (one with an '
                'end-effector contract, ADR-0022). The emptying loop samples only its own '
                f'detector "{det}", once per cycle before the retry, and emits no other Detect '
                f'block, so this one would be dropped: {", ".join(lost)}. Remove the block: a '
                'detector that guides a move then samples once per cycle, before the retry and '
                'before anything is held. A sampling point inside or after the pick is not '
                f'supported by {self.app_type} yet (vNext G0; KNOWN_DEBT).')
        return out

    # --- validation -------------------------------------------------------------
    def validate(self, project: CanonicalProject) -> List[str]:
        # Everything vision_guided_motion checks still holds (a vision binding per guided
        # move, a continuous detector, no blind ptp/pilz into a vision goal, …) and is
        # inherited rather than re-derived.
        problems = super().validate(project)
        app = project.application
        cfg = app.bin_picking

        if ToolActionKind.GRASP not in {a.kind for a in app.tool_actions}:
            problems.append(f'{self.app_type}: no grasp tool action — the loop would '
                            'approach a candidate and never pick it up')
        if ToolActionKind.RELEASE not in {a.kind for a in app.tool_actions}:
            problems.append(f'{self.app_type}: no release tool action — the loop would '
                            'carry the first object for ever')

        first, last = self._candidate_bounds(project)
        if first is None:
            problems.append(
                f'{self.app_type}: the retry has no slice to wrap — no move carries a '
                'pick role (pre_pick / pick / post_pick) AND no grasp is anchored at a '
                'move. The generated tree falls back to a linear vision sequence. Add '
                'the grasp where the gripper closes, or give the approach, the grasp and '
                'the retreat their roles.')
            return problems

        det = self._loop_detector(project)
        if det is None:
            problems.append(
                f'{self.app_type}: no pick-role waypoint is guided by a detector — the '
                'loop would retry the same taught pose N times. Bind the grasp move to a '
                'Step-5 detector.')
        else:
            seq = app.sequence or self._derived_sequence(project)
            others = {wp.vision.detector
                      for wp in self._vision_waypoints(project)
                      if wp.name in seq[first:last + 1] and wp.vision.detector != det}
            if others:
                problems.append(
                    f'{self.app_type}: the candidate moves are guided by more than one '
                    f'detector ({det}, {", ".join(sorted(others))}). One sampling feeds '
                    'one retry loop, so every move inside it must read the same '
                    'detector; sample the others before the loop.')
            if any(pt.detector == det for pt in app.detections):
                problems.append(
                    f'note: the Detect block for "{det}" is ignored by {self.app_type} — '
                    'the loop owns that detector\'s sampling instant (once per cycle, '
                    'before the retry), which is the whole point of the loop.')
            problems += self._dropped_detect_blocks(project)   # and REFUSED: see refusals()
            # The loop retries candidates IN THE ORDER IT WAS GIVEN (ADR-0016 Decision 2:
            # TMR filters and truncates and never sorts), so with no candidate_rule the
            # order is whatever the model returned and "try the next one" is not a ranking
            # — it is a shuffle. A note, not a refusal: the rule is a perception choice and
            # an unset one still runs (ADR-0017 Decision 1).
            spec = (project.perception.detector_by_name(det)
                    if project.perception else None)
            if spec is not None and spec.candidate_rule is None:
                problems.append(
                    f'note: {self.app_type}: detector "{det}" has no candidate_rule, so '
                    'the loop retries candidates in whatever order the detector returned '
                    '— the published ORDER is the ranking (ADR-0016), and the runtime '
                    'never sorts. Set the rule on the Perception step; the measured '
                    'default is "candidate x visible face".')

        if cfg.tries <= 1:
            problems.append(
                f'{self.app_type}: tries = {cfg.tries}, so the retry is inert and the '
                'application is a linear pick with extra nodes. The reference cell needed '
                'K = 3,33 candidates per object delivered; 3 is the measured floor.')
        if canonical_device(project) is not None and det is not None:
            # vNext M2.7 (owner Q8.2): only MoveWaypoint reports a motion FAULT; a move streamed
            # through ServoTrack inside the cycle cannot. Without a loop detector there is no cycle
            # (the tree falls back to the linear sequence, where any failure ends the run), so no
            # note. The sequence is read here, not reused from the detector branch above.
            cyc = (app.sequence or self._derived_sequence(project))[first:]
            streamed = [n for n in cyc
                        if self._move_node(app.segment_for(n)) == MOVE_NODES[ExecutionMode.SERVO]]
            if streamed:
                problems.append(
                    f'note: {self.app_type}: {", ".join(streamed)} stream(s) through ServoTrack, '
                    'which reports no motion fault: a servo outage inside the canonical cycle '
                    'reads as a refused candidate or a stalled bin, not as a FAULT (vNext M2.7, '
                    'KNOWN_DEBT).')
        problems += self._holding_check_conflicts(cfg, project)   # and REFUSED: see refusals()
        if not cfg.holding_check and project.robot.gripper.contract is None:
            problems.append(
                f'note: {self.app_type}: holding_check is OFF, so a closed gripper counts '
                'as a successful pick. On the reference cell the two differ on 1 lift in '
                '22, and every one of those would be placed empty and counted as good.')
        if not cfg.recovery_channel:
            problems.append(
                f'note: {self.app_type}: no recovery_channel, so a stalled bin is only '
                'logged ("bin empty") and the next cycle re-detects. On the reference '
                'cell agitation after a stall was worth 6 capsules out of 25.')

        if app.policies:
            problems.append(
                f'{self.app_type} does not emit learned-policy steps: a policy inside a '
                'retry has no defined interaction with the candidate index (ADR-0005/'
                'ADR-0016). The '
                f'{len(app.policies)} configured step(s) are NOT in the generated tree — '
                'use vision_guided_motion for a policy-driven pick.')
        return problems

    # --- the tree ---------------------------------------------------------------
    def build_tree_xml(self, project: CanonicalProject) -> str:
        app = project.application
        cfg = app.bin_picking
        seq = app.sequence or self._derived_sequence(project)
        first, last = self._candidate_bounds(project)
        det_name = self._loop_detector(project)
        if first is None or det_name is None:
            # Nothing to retry: emit the honest thing (the inherited vision sequence) and
            # say why at the top of the file. validate() has already reported it.
            why = ('no pick-role waypoint' if first is None
                   else 'no detector on the grasp move')
            return (f'<!-- {self.app_type}: NO retry loop was emitted — {why}. This is '
                    'the linear vision sequence; see the generation manifest. -->\n'
                    + super().build_tree_xml(project))

        if canonical_device(project) is not None:
            return self._build_canonical_tree(project, first, last, det_name)

        name = _camel(project.robot.robot_name) + 'BinPicking'
        label = f'{project.robot.robot_name} bin picking (BT, data-driven)'
        loop = app.loop_cycles if app.loop_cycles != 0 else -1

        out: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
        out.append(f'<!-- Generated by the TrainIt Setup Assistant ({self.app_type}). '
                   'The emptying loop of ADR-0016: ranked candidates, a retry over the')
        out.append('     candidate index, a holding check after the lift, and a gated '
                   'stall recovery. Every DOF is in config/bt_params.yaml. -->')
        out.append(f'<root BTCPP_format="4" main_tree_to_execute="{app.bt_tree_id}">')
        out.append(f'  <BehaviorTree ID="{app.bt_tree_id}">')
        out.append(f'    <Sequence name="{name}">')
        out.append(f'      <Log message="{label}: start"/>')
        out.append('')

        # --- the prologue: the moves before the first candidate move, run ONCE ---
        for wp_name in seq[:first]:
            out += self._move_block(project, wp_name, '      ')
        if first:
            out.append('')

        # --- the loop ---
        out.append('      <!-- The emptying loop. num_cycles="-1" runs until the '
                   'operator stops it; set')
        out.append('           application.loop_cycles for a bounded run. -->')
        out.append(f'      <Repeat num_cycles="{loop}">')
        out.append(f'        <Fallback name="{name}Cycle">')
        out.append('          <Sequence name="PickOneObject">')
        out += self._cycle_relative_lines(project, set(seq[first:last + 1]),
                                          '            ')
        out += self._detect_lines(project, det_name, cfg, '            ')
        out.append('')
        out.append('            <!-- The candidate index the retry advances. Set to -1 '
                   'and incremented as the')
        out.append('                 FIRST step of each attempt, so attempt 1 reads '
                   'candidate 0 (ADR-0016). -->')
        out.append(f'            <Script code="{_INDEX_KEY} := -1"/>')
        out.append(f'            <RetryUntilSuccessful num_attempts="{cfg.tries}">')
        out.append('              <Sequence name="OneCandidate">')
        out.append(f'                <Script code="{_INDEX_KEY} := {_INDEX_KEY} + 1"/>')
        out += self._candidate_bindings(project, det_name, seq[first:last + 1],
                                        '                ')
        for wp_name in seq[first:last + 1]:
            out += self._move_block(project, wp_name, '                ')
        out += self._holding_lines(cfg, '                ')
        out.append('              </Sequence>')
        out.append('            </RetryUntilSuccessful>')
        out.append('')

        # --- after the retry: the delivery (and whatever closes the cycle) ---
        for wp_name in seq[last + 1:]:
            out += self._move_block(project, wp_name, '            ')
        out.append('          </Sequence>')
        out += self._stall_lines(project, cfg, '          ')
        out.append('        </Fallback>')
        out.append('      </Repeat>')

        out.append('')
        out.append(f'      <Log message="{label}: complete"/>')
        out.append('    </Sequence>')
        out.append('  </BehaviorTree>')
        out.append('</root>')
        return '\n'.join(out) + '\n'

    def _build_canonical_tree(self, project: CanonicalProject, first: int, last: int,
                              det_name: str) -> str:
        """The emptying loop of a CANONICAL project (ADR-0022 D5): the same loop, plus the
        verdict / fault / holding gate. Standard BehaviorTree.CPP only -- a Script that resets
        the gate as the FIRST step of every cycle, ``_failureIf`` on the retried child and on
        the stall branch, the EEF nodes' ``fault`` (inside the retry) and ``reason`` (after it)
        remapped to the gate key, DetectObject's ``fault`` too:

        * a VERDICT inside the retry (NO_OBJECT: nothing came up) still advances it, and a FRESH
          empty detection still reaches the stall branch -- "bin empty", as today;
        * a FAULT inside the retry, any end-effector failure after it, a silent or stale
          detector, and ANY failure while the cycle holds an object end the run with FAILURE
          and its reason (HOLD, EEF-D11) -- never "bin empty", never a recovery pulse;
        * a motion FAULT (MoveWaypoint's ``fault``, vNext M2.7: a failure that is not the
          target's) ends the run too; a target's own motion failure keeps the existing strategy.
        """
        app = project.application
        cfg = app.bin_picking
        dev = canonical_device(project)
        seq = app.sequence or self._derived_sequence(project)
        name = _camel(project.robot.robot_name) + 'BinPicking'
        label = f'{project.robot.robot_name} bin picking (BT, data-driven)'
        loop = app.loop_cycles if app.loop_cycles != 0 else -1
        fault, reason = ('fault', _GATE_KEY), ('reason', _GATE_KEY)

        out: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
        out.append(f'<!-- Generated by the TrainIt Setup Assistant ({self.app_type}). '
                   'The emptying loop of ADR-0016: ranked candidates, a retry over the')
        out.append('     candidate index, a holding check after the lift, and a gated '
                   'stall recovery. Every DOF is in config/bt_params.yaml.')
        out.append(f'     End-effector contract (ADR-0022 D5): device "{dev.id}". A fault, or '
                   'any failure while the cycle')
        # never two dashes in a row inside an XML comment (the document would not be well-formed)
        out.append('     holds an object, ends the run with FAILURE: it never reads as '
                   '"bin empty". -->')
        out.append(f'<root BTCPP_format="4" main_tree_to_execute="{app.bt_tree_id}">')
        out.append(f'  <BehaviorTree ID="{app.bt_tree_id}">')
        out.append(f'    <Sequence name="{name}">')
        out.append(f'      <Log message="{label}: start"/>')
        out.append('')

        for wp_name in seq[:first]:
            out += self._move_block(project, wp_name, '      ')
        if first:
            out.append('')

        out.append('      <!-- The emptying loop. num_cycles="-1" runs until the '
                   'operator stops it; set')
        out.append('           application.loop_cycles for a bounded run. -->')
        out.append(f'      <Repeat num_cycles="{loop}">')
        out.append(f'        <Fallback name="{name}Cycle">')
        out.append('          <Sequence name="PickOneObject">')
        out.append(f'            <!-- The gate, reset as the FIRST step of every cycle: '
                   f'{_GATE_KEY} holds the reason of a fault,')
        out.append(f'                 {_HOLDING_KEY} says the cycle holds an object '
                   '(ADR-0022 D5). -->')
        out.append(f'            <Script code="{_GATE_KEY} := \'\'; {_HOLDING_KEY} := 0"/>')
        out += self._cycle_relative_lines(project, set(seq[first:last + 1]),
                                          '            ')
        out += self._detect_lines(project, det_name, cfg, '            ', fault=_GATE_KEY)
        out.append('')
        out.append('            <!-- The candidate index the retry advances. Set to -1 '
                   'and incremented as the')
        out.append('                 FIRST step of each attempt, so attempt 1 reads '
                   'candidate 0 (ADR-0016). -->')
        out.append(f'            <Script code="{_INDEX_KEY} := -1"/>')
        out.append(f'            <RetryUntilSuccessful num_attempts="{cfg.tries}">')
        out.append(f'              <Sequence name="OneCandidate" '
                   f'_failureIf="{_GATE_KEY} != \'\'">')
        out.append(f'                <Script code="{_INDEX_KEY} := {_INDEX_KEY} + 1"/>')
        out += self._candidate_bindings(project, det_name, seq[first:last + 1],
                                        '                ')
        for wp_name in seq[first:last + 1]:
            out += self._move_block(project, wp_name, '                ', gate=fault,
                                    verify_after_grasp=False, motion_gate=_GATE_KEY)
        out += self._canonical_holding_lines(dev, '                ', fault)
        out.append('              </Sequence>')
        out.append('            </RetryUntilSuccessful>')
        out.append(f'            <Script code="{_HOLDING_KEY} := 1"/>')
        out.append('')

        for wp_name in seq[last + 1:]:
            out += self._move_block(project, wp_name, '            ', gate=reason,
                                    after_release=[f'<Script code="{_HOLDING_KEY} := 0"/>'],
                                    motion_gate=_GATE_KEY)
        out.append('          </Sequence>')
        out += self._stall_lines(project, cfg, '          ',
                                 gate=f"{_GATE_KEY} != '' || {_HOLDING_KEY} == 1")
        out.append('        </Fallback>')
        out.append('      </Repeat>')

        out.append('')
        out.append(f'      <Log message="{label}: complete"/>')
        out.append('    </Sequence>')
        out.append('  </BehaviorTree>')
        out.append('</root>')
        return '\n'.join(out) + '\n'

    @staticmethod
    def _canonical_holding_lines(dev, indent: str, gate) -> List[str]:
        """The canonical holding check (ADR-0022 D4): VerifyGrasp on the device's own state, in
        place of CheckRobotState, when the project requires grasp confirmation. A VERDICT here
        (nothing came up) sends the retry to the next candidate; a fault writes the gate."""
        if not dev.require_grasp_confirmation:
            return [f'{indent}<!-- require_grasp_confirmation is false: command success is '
                    f'the grasp, as the project declares (ADR-0022 D4). -->']
        return [
            f'{indent}<!-- Did it actually come up: the device\'s own grasp state, fresh and '
            f'with accepted evidence.',
            f'{indent}     A verdict (NO_OBJECT) sends the retry to the next candidate; a '
            f'fault ends the run. -->',
            f'{indent}{verify_grasp(dev, None, "held", gate)}',
        ]

    # --- pieces -----------------------------------------------------------------
    def _detect_lines(self, project: CanonicalProject, det_name: str, cfg,
                      indent: str, fault: str = '') -> List[str]:
        """One sampling per cycle, OUTSIDE the retry: the candidates are ranked once and
        then tried in the order they were published (ADR-0016 Decision 1)."""
        per = project.perception
        d = per.detector_by_name(det_name)
        out_key = f'detected.{d.name}'
        lines = [
            f'{indent}<!-- One sampling per cycle, OUTSIDE the retry: the detector '
            f'publishes BEST FIRST and that',
            f'{indent}     order IS the ranking (ADR-0016 Decision 1). keep = the retry '
            f'budget, so every',
            f'{indent}     candidate the loop can try has a blackboard entry and no '
            f'more. -->',
            f'{indent}<DetectObject detector="{d.name}" class_id="{d.class_id}" '
            f'target_frame="{project.robot.base_frame}" '
            f'timeout_ms="{per.detect_timeout_ms}" keep="{cfg.tries}" '
            f'out_key="{out_key}"' + (f' fault="{{{fault}}}"' if fault else '') + '/>',
        ]
        # Any OTHER detector in use (e.g. one that finds the place target) samples here
        # too, before the retry — it has nothing to do with the candidate index.
        # A detector with a Detect block of its own is skipped here and emitted nowhere else:
        # refused in a canonical project (_dropped_detect_blocks), dropped in a legacy one.
        explicit = {pt.detector for pt in project.application.detections}
        others = [n for n in self._by_detector(self._vision_waypoints(project))
                  if n != det_name and n not in explicit]
        for other in others:
            lines.append('')
            lines.append(f'{indent}<!-- "{other}" is not the bin detector: it samples '
                         f'once per cycle and is not indexed. -->')
            lines += self._detector_group(project, other, indent, fault=fault)
        return lines

    def _cycle_relative_lines(self, project: CanonicalProject, region: set,
                              indent: str) -> List[str]:
        """Step-relative moves whose reference is a TAUGHT pose: derived once per cycle,
        at the top. The ones that hang off a CANDIDATE move are not here — they are
        re-derived inside the retry, per attempt, because their reference moves."""
        app = project.application
        taught = [wp for wp in self._relative_waypoints(project)
                  if wp.name not in region and wp.relative.step not in region
                  and (app.waypoint_by_name(wp.relative.step) is not None
                       and app.waypoint_by_name(wp.relative.step).vision is None)]
        if not taught:
            return []
        return ([f'{indent}<!-- Relative steps (D-017) off a taught pose: derived once '
                 f'per cycle. -->']
                + [self._relative_line(wp, indent) for wp in taught]
                + [''])

    def _candidate_bindings(self, project: CanonicalProject, det_name: str,
                            region: List[str], indent: str) -> List[str]:
        """Point this attempt's moves at candidate ``{candidate}`` — and re-derive the
        step-relative moves that hang off them, so the retreat follows the grasp it
        actually took."""
        app = project.application
        out_key = f'detected.{det_name}'
        lines: List[str] = []
        bound: List[Waypoint] = []
        for wp_name in region:
            wp = app.waypoint_by_name(wp_name)
            if wp is not None and wp.vision is not None and wp.vision.detector == det_name:
                b = wp.vision
                offsets = ''.join(f'{k}="{v:.3f}" ' for k, v in
                                  (('dx', b.dx), ('dy', b.dy)) if round(v, 3) != 0.0)
                lines.append(
                    f'{indent}<SetWaypointFromDetection waypoint="{wp.name}" '
                    f'from="{out_key}" index="{{{_INDEX_KEY}}}" {offsets}'
                    f'dz="{b.dz:.3f}" orientation="{b.orientation}"/>')
                bound.append(wp)
        ref_names = {wp.name for wp in bound}
        for wp_name in region:
            wp = app.waypoint_by_name(wp_name)
            if (wp is not None and wp.relative is not None
                    and wp.relative.step in ref_names):
                lines.append(self._relative_line(wp, indent))
        return lines

    def _holding_lines(self, cfg, indent: str) -> List[str]:
        """Did it actually come up. The difference between a closed gripper and a held
        object is 1 lift in 22 on the reference cell — and a FAILURE here is what sends
        the retry to the next candidate instead of placing nothing."""
        if not cfg.holding_check:
            return [f'{indent}<!-- holding_check is OFF: a closed gripper counts as a '
                    f'pick (bin_picking.holding_check). -->']
        attrs = ['require_attached="true"']
        if cfg.attached_topic:
            attrs.append(f'attached_topic="{cfg.attached_topic}"')
        attrs.append(f'timeout_ms="{cfg.holding_timeout_ms}"')
        return [
            f'{indent}<!-- Did it actually come up: this is what turns "the gripper '
            f'closed" into "the object',
            f'{indent}     came up". A FAILURE here is not an error — it is what sends '
            f'the retry to the next',
            f'{indent}     candidate. -->',
            f'{indent}<CheckRobotState {" ".join(attrs)}/>',
        ]

    def _stall_lines(self, project: CanonicalProject, cfg, indent: str,
                     gate: str = '') -> List[str]:
        """The bin stalled: no candidate was detected, or none of them worked. Those are
        the same stall to an operator, which is why ONE branch treats both. ``gate``: the
        canonical ``_failureIf`` (ADR-0022 D5) -- a fault, or a failure while holding, is not a
        stall."""
        inner = indent + '  '
        lines = [
            f'{indent}<!-- THE BIN STALLED: either nothing was detected, or every '
            f'candidate refused. Those are',
            f'{indent}     the same stall to an operator, so one branch treats both. -->',
            f'{indent}<Fallback name="BinStalled"'
            + (f' _failureIf="{gate}"' if gate else '') + '>',
        ]
        if cfg.recovery_channel:
            lines += [
                f'{inner}<!-- The recovery STEP. What the channel "{cfg.recovery_channel}" '
                f'ACTUATES is CELL I/O',
                f'{inner}     (ADR-0013), not this application: a shake is a machine with '
                f'an amplitude, a',
                f'{inner}     frequency and a direction, and it lives in the cell\'s own '
                f'I/O program. TSA emits',
                f'{inner}     the semantic Cmd_/Sts_ handshake and nothing about the '
                f'machine behind it. -->',
                f'{inner}<Sequence name="RecoverFromStall">',
                f'{inner}  <Log message="bin stalled: pulsing '
                f'{cfg.recovery_channel}"/>',
                f'{inner}  <EnableProcess channel="{cfg.recovery_channel}"/>',
            ]
            if cfg.recovery_dwell_ms > 0:
                lines.append(f'{inner}  <Sleep msec="{cfg.recovery_dwell_ms}"/>')
            lines += [
                f'{inner}  <DisableProcess channel="{cfg.recovery_channel}"/>',
                f'{inner}</Sequence>',
            ]
        else:
            lines.append(f'{inner}<!-- No recovery step configured '
                         f'(bin_picking.recovery_channel is empty). -->')
        lines += [
            f'{inner}<Log message="bin empty"/>',
            f'{indent}</Fallback>',
        ]
        return lines

    def _move_block(self, project: CanonicalProject, wp_name: str,
                    indent: str, gate=None, verify_after_grasp: bool = True,
                    after_release: List[str] = (), motion_gate: str = '') -> List[str]:
        """One move with everything anchored at it: the per-move attached-collision check,
        the move node (``MoveWaypoint`` or ``ServoTrack`` — ADR-0015), its tool actions,
        the process pause, and a cycle-boundary ``ResetScene`` + settle barrier last.
        ``gate``, ``verify_after_grasp`` and ``after_release`` shape a CANONICAL loop's
        end-effector nodes (ADR-0022 D5), ``motion_gate`` its moves' motion fault (vNext M2.7);
        a legacy project never passes them."""
        app = project.application
        seg = app.segment_for(wp_name)
        lines: List[str] = []
        if seg is not None and seg.attached_collision_check is not None:
            val = 'true' if seg.attached_collision_check else 'false'
            lines.append(f'{indent}<SetAttachedCollisionCheck value="{val}"/>')
        node = self._move_node(seg)
        # vNext M2.7 (owner Q8.2, ADR-0022 amendment M2.7): inside a canonical cycle every
        # MoveWaypoint writes its `fault` -- a failure that is not the target's -- into the gate
        # key. ServoTrack has no such output (KNOWN_DEBT); a legacy project never passes
        # motion_gate.
        fault = (f' fault="{{{motion_gate}}}"'
                 if motion_gate and node == MOVE_NODES[ExecutionMode.PLAN] else '')
        lines.append(f'{indent}<{node} waypoint="{wp_name}"{fault}/>')
        acts = app.actions_at(wp_name)
        for action in acts:
            if action.kind is not ToolActionKind.RESET_SCENE:
                lines += self._action_lines(action, project, indent, gate, verify_after_grasp)
                if action.kind is ToolActionKind.RELEASE:
                    lines += [indent + extra for extra in after_release]
        if seg is not None and seg.wait_after_ms > 0:
            lines.append(f'{indent}<Sleep msec="{seg.wait_after_ms}"/>')
        for action in acts:
            if action.kind is ToolActionKind.RESET_SCENE:
                lines += self._action_lines(action, project, indent)
                lines += self._after_scene_reset(project, indent)
        return lines

    def _action_lines(self, action, project: CanonicalProject,
                      indent: str, gate=None, verify_after_grasp: bool = True) -> List[str]:
        """``_render_action`` returns one or two lines at a FIXED indent (the linear
        template's); re-indent them for the loop's deeper nesting. A canonical project's
        end-effector nodes come one per line (ADR-0022)."""
        if canonical_device(project) is not None:
            return [indent + line.strip() for line in self._tool_action_lines(
                action, project, '', gate, verify_after_grasp)]
        return [indent + line.strip()
                for line in self._render_action(action, project).split('\n')]

    # --- helpers ----------------------------------------------------------------
    # Which moves belong to ONE CANDIDATE is read from the project in two ways, and the
    # order matters: an EXPLICIT role wins, and otherwise the slice is DERIVED from what
    # the project already says. Both are needed, because the MVP block editor (Step 7)
    # does not ask for roles at all — a template that only understood roles would quietly
    # emit no loop for every bundle built in the GUI.

    def _grasp_index(self, project: CanonicalProject, seq: List[str]) -> Optional[int]:
        """Sequence index of the move the GRASP is anchored at — the one unambiguous
        landmark of a pick, and one the GUI always produces."""
        grasped = {a.at_waypoint for a in project.application.tool_actions
                   if a.kind is ToolActionKind.GRASP}
        for i, name in enumerate(seq):
            if name in grasped:
                return i
        return None

    def _candidate_bounds(self, project: CanonicalProject):
        """(first, last) sequence indices of the slice the retry wraps.

        Explicit roles (``pre_pick`` / ``pick`` / ``post_pick``) win when any is set. With
        no roles the slice is derived: it ENDS at the retreat (a step-relative move hanging
        off the grasp — a lift whose pose follows the grasp it actually took) and STARTS at
        the earliest camera-guided move at or before the grasp, because a move to a FIXED
        taught pose (an "above the bin" approach) gains nothing from being retried.
        ``(None, None)`` when the project has neither roles nor a grasp.
        """
        app = project.application
        seq = app.sequence or self._derived_sequence(project)
        by_role = [i for i, n in enumerate(seq)
                   if (app.waypoint_by_name(n) is not None
                       and app.waypoint_by_name(n).role in _CANDIDATE_ROLES)]
        if by_role:
            return by_role[0], by_role[-1]

        grasp = self._grasp_index(project, seq)
        if grasp is None:
            return None, None
        det = self._loop_detector(project)
        first = grasp
        if det is not None:
            for i, n in enumerate(seq[:grasp + 1]):
                wp = app.waypoint_by_name(n)
                if wp is not None and wp.vision is not None and wp.vision.detector == det:
                    first = i
                    break
        inside = set(seq[first:grasp + 1])
        last = grasp
        for i in range(grasp + 1, len(seq)):
            wp = app.waypoint_by_name(seq[i])
            if wp is not None and wp.relative is not None and wp.relative.step in inside:
                last = i
            else:
                break
        return first, last

    def _loop_detector(self, project: CanonicalProject) -> Optional[str]:
        """The detector whose candidate LIST the loop iterates.

        The grasp move's own binding when it has one; otherwise the nearest camera-guided
        move before it (an approach computed from the detection, with the grasp relative
        to it); otherwise the ``pick``-role move's binding.
        """
        app = project.application
        per = project.perception
        if per is None:
            return None
        seq = app.sequence or self._derived_sequence(project)

        def _det(name):
            wp = app.waypoint_by_name(name)
            if wp is None or wp.vision is None:
                return None
            return (wp.vision.detector
                    if per.detector_by_name(wp.vision.detector) is not None else None)

        grasp = self._grasp_index(project, seq)
        if grasp is not None:
            for name in reversed(seq[:grasp + 1]):
                got = _det(name)
                if got:
                    return got
        for name in seq:
            wp = app.waypoint_by_name(name)
            if wp is not None and wp.role is WaypointRole.PICK:
                got = _det(name)
                if got:
                    return got
        return None


def _camel(text: str) -> str:
    return ''.join(part.capitalize()
                   for part in text.replace('-', '_').split('_') if part)
