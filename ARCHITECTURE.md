# TrainIt Setup Assistant — internal architecture & roadmap

Internal notes (not shipped to customers). How the TSA is built behind the scenes, the
key decisions, and where to take it next. Pairs with the runtime notes in
`trainit_motion_runtime/docs/`.

## 1. The product in one paragraph

The TSA turns a **CanonicalProject** (`model/`) into a 3-package **bundle** that runs on
the TMR. The GUI (`gui/`) only edits the project; the generator (`generator/`) only reads
it; `trainit_generate` is the headless contract between them. What's automatable is the
**application layer**; the **digital twin** (Isaac scene + base MoveIt config + vendor
bridges) is hand-made expert work — the *Adaptation Sprint*. That boundary is the product.

## 2. The three-config model (the core decision)

A cell has **three** MoveIt configs with distinct roles:

| Config | Who makes it | Role |
|---|---|---|
| `<robot>_moveit_config` (**base**) | hand, Adaptation Sprint | The mock/isaac/real bring-up + bridge wiring + SRDF waypoints. The source of truth for the connectors. |
| `<robot>_scene_loader_moveit_config` (**intermediate**) | TSA Step 3 | A standalone copy of the base + planners + `scene.yaml`, used **only** to configure the app against a faithful RViz. |
| `<robot>_trainit_config` (**bundle**) | TSA Step 8 | The same standalone copy, shipped in the bundle. **Independent** of the other two configs. |

**Why the bundle is a standalone COPY (not an overlay):** the intermediate/bundle configs
are produced by `SceneLoaderMoveitConfigEmitter`, which **copies the base's `config/`
verbatim** (SRDF, urdf.xacro, ros2_control.xacro, all yamls, moveit.rviz). This preserves
the hand-tuned mode-switch + SRDF exactly, and — crucially — the copied URDF's only
external `$(find …)` reference is the **connector geometry package** (`<cell>_isaac`), not
another moveit_config. So the copy is self-contained: it needs the connector/bridge
packages (legitimate) but **not** the base or intermediate config packages. `real`/`isaac`/
`mock` come for free because they were copied from the base.

**Action-namespace parity** (the invariant that keeps MoveIt byte-identical across modes):
the arm action `…/moveit_joint_controller/follow_joint_trajectory` and the gripper action
`…/<gripper>_controller/gripper_command` are served under identical namespaces by whatever
backend is active. `mock↔isaac` swap the ros2_control hardware plugin (`GenericSystem ↔
topic_based_ros2_control`); **`real` bypasses ros2_control entirely** and launches the
vendor bridges under the same namespaces.

## 3. Generation (`generator/`)

- `orchestrator.default_emitters(project)` chooses the MoveIt-config emitter by whether
  `robot.base_moveit_config_path` is set: **scene-loader** (copy-base, the MVP flow) vs
  the legacy **template-from-model** `MoveitConfigEmitter` (from-scratch / fr3wml golden).
- `emitters/scene_loader_moveit_config_pkg.py` → copy base `config/` + generate
  `scene.yaml` (`generator/scene_yaml.py`) + template a self-referential `bringup.launch.py`
  (`templates/moveit_config/scene_loader_bringup.launch.py.j2`; `build_move_group_params`
  on **itself** + `scene_manager_node`). Invoked for the bundle config and, via
  `generate_scene_loader_config()`, for the Step-3 intermediate (same logic → config==deploy).
- `emitters/app_pkg.py` + `applications/pick_and_place.py` (and
  `applications/vision_guided_motion.py`, which inherits the blind skeleton via a
  registry keyed by `AppType` and adds the vision prologue + settle; and
  `applications/bin_picking.py`, which inherits *that* and replaces only the tree's
  SHAPE — see §4b) → BT XML + `bt_params.yaml` + `perception.yaml` + launch + Groot2 +
  README. Every DOF is in `bt_params.yaml`; the tree is uniform except for bin picking,
  whose whole value is that it is not.
- `generator/namespace.py` → the cell namespace, read in ONE place (§4c). Every
  namespaced artefact derives from `RobotSpec.cell_namespace` and from nothing else, so
  there is no second spelling for the two halves of a cell to disagree over.
- All writes go through `GenContext` → recorded in `generation_manifest.yaml`
  (COPY/TEMPLATE/GENERATE + sha256). Determinism: same project ⇒ byte-identical output.
  Since 7.5.0 (vNext M2.3, ADR-0021) the manifest has two parts: PROVENANCE —
  `generator_version` and `capability_snapshot`, the identity of the committed catalog snapshot
  TSA compiled with (`catalog.snapshot_identity()`: the sha256 of `catalog/snapshot.yaml` and of
  the two registry copies, their schemas and the component releases; hashes and versions,
  never the catalog's content) — and CONTENT, one record per file. A COPY's source is
  `package://<package>/<path inside it>` (the nearest `package.xml`; the file as the project names
  it, a symlink not resolved, a `..` read as the OS reads it) or `external:<file name>`
  (`generator/provenance.py`), never the
  generating machine's path; the warnings that mention a file name it the same way in the
  manifest, while the user who generates sees the paths as given (`local_warnings`, CLI and GUI).
  A catalog changed while TSA runs, or derived by another release, is refused before any file.
- What a bundle needs to RUN (7.6.0, vNext M2.4, ADR-0021 amendment M2.4) is declared in its
  `package.xml` files, never in the manifest, and never because the snapshot knows a component.
  `generator/runtime_requirements.py` reads the files THIS generation wrote for each package
  (`GenContext.written`, never a COPY): the BT trees (their nodes, the ports and port values that
  arrived later), `bt_params.yaml`, and the launch files, parsed with `ast` (the launch helper's
  `backend=`; `trainit_run_bt` and `scene_manager_node` under a non-empty `namespace=`). The
  package's `trainit_motion_runtime` floor is the highest `catalog.TMR_FLOORS` version among
  them, or the first version TMR's package.xml stated (1.4.0). It is written as `version_gte`,
  with a comment naming what set it. The table is curated by hand from TMR's history, each row
  with its TMR commit, and identified by the TSA release (`generator_version`), not by the
  catalog hash; `0.0.0` certifies nothing. The app declares `trainit_policy_runtime` iff a tree
  runs RunPolicy or a launch file starts its node. `trainit_perception` keeps its M2.2 rule:
  declared iff a TP `detector_node` is launched, floored by TP's own `since` of each parameter
  written. It FAILS CLOSED: a BT node the table does not date, a TMR node or launch-helper call
  that is not a literal call, or a package whose TMR node cannot be seen, stops generation (a TSA
  defect). A generated file that does not parse (a project name the XML or Python cannot hold)
  is a manifest warning, and the floor is the newest version the table dates.

## 4. Dynamic-object management (the Step-8 crux)

Model (`model/scene.py`, `model/task.py`):
- `SceneObject`: `category` (static/actuated/dynamic) drives collision role; per-dynamic
  `grasp_target` (attaches on close), `release_policy` (freeze|gravity — never float),
  `isaac_grasp_method` (metadata), `touchable_collision_ids`.
- `MotionSegment.attached_collision_check` (per-move; None = inherit).
- `MotionSegment.execution` (per-move, ADR-0015): `plan` | `servo`, non-optional with a
  `plan` default — absolute semantics, so there is deliberately no global to inherit from.
  Its own axis, orthogonal to `motion` and to the planner. The value picks the BT NODE
  (`MoveWaypoint` vs `ServoTrack`), which is why `bt_params.yaml` carries no `execution`
  key: a second copy would be redundant. `ApplicationSpec.uses_servo()` is THE gate for
  the two servo artefacts — `config/servo.yaml` (`generator/servo_yaml.py` +
  `templates/moveit_config/servo.yaml.j2`, every frame/topic/group derived from the cell
  model) and the `servo_node` entry in the generated cell bring-up. Both sit behind
  `{% if servo %}`, so an all-`plan` project emits neither byte and the published golden
  bundle regenerates unchanged (the ADR's acceptance criterion).

Runtime (TMR `bt_nodes.cpp`, added for this): the tree drives the flags automatically —
- **`SetAttachedCollisionCheck`** calls `scene_manager_node`'s `~/attached_collision_check`
  SetBool service and **blocks on the response** so the ACM is applied before the next
  plan. Emitted before a transfer that must route the held payload around static meshes.
- **`SetReleasePolicy`** publishes a latched Bool on `/isaac_release_policy` (freeze=true),
  emitted before `OpenGripper`. A no-op in mock/real.

Both are gated on the scene-loader flow (a base config being set), so the from-scratch
golden path stays byte-stable.

## 4b. The bin_picking template (ADR-0016 Decision 4) — the one tree that is not linear

`applications/bin_picking.py` subclasses `VisionGuidedMotion`, so every vision and
step-relative check is inherited; what it replaces is the tree's shape:

```
Repeat                                   <- loop_cycles (-1 = until the operator stops)
  Fallback                               <- pick, or recover from the stall
    Sequence                             <- the pick
      DetectObject keep="{tries}"        <- ranked ONCE, best first (Decision 1)
      Script  candidate := -1
      RetryUntilSuccessful num_attempts="{tries}"
        Sequence                         <- ONE candidate, in rank order
          Script  candidate := candidate + 1
          SetWaypointFromDetection index="{candidate}"
          (SetWaypointRelative for the retreat — re-derived per attempt)
          approach / grasp / CloseGripper / retreat
          CheckRobotState require_attached    <- did it actually come up
      place / OpenGripper
    Fallback                             <- the bin stalled
      Sequence  EnableProcess / Sleep / DisableProcess   <- gated on recovery_channel
      Log "bin empty"
```

Three design points worth keeping:

- **The index advances with BT.CPP's own `Script`.** ADR-0016 Decision 2 gives TMR two
  PORTS (`DetectObject.keep`, `SetWaypointFromDetection.index`) and no control node,
  because none is needed: `Script` sets the blackboard key to `-1` before the retry and
  increments it as the FIRST step of each attempt, so attempt 1 reads candidate 0. Verified
  against BehaviorTree.CPP 4.10.0 with the TrainIt leaves stubbed: the tree loads, the
  indices seen are `0 1 2`, and the stall branch fires when none of them holds.
- **`DetectObject` is OUTSIDE the retry, on purpose.** Inside it, the candidates would be
  re-ranked between attempts and the index would point at a different object each time.
- **Which moves the retry wraps is read from the project in two ways, and the order
  matters.** An explicit role (`pre_pick`/`pick`/`post_pick`) wins; with no roles the slice
  is DERIVED — it ends at the step-relative retreat hanging off the grasp and starts at the
  earliest camera-guided move at or before the grasp, because a move to a fixed taught pose
  gains nothing from being retried. Both paths are needed: the MVP block editor (Step 7)
  does not ask for roles at all, so a template that only understood roles would quietly
  emit no loop for every bundle built in the GUI.

The agitator is **not** in the template. The template emits the *step* — a pulse on a
named `ProcessController` channel, the semantic `Cmd_`/`Sts_` handshake of ADR-0013 — and
the generated tree says so at the point of use, so nobody reading it mistakes the step for
the machine.

**A Detect block the loop cannot emit is refused, not dropped (vNext G0, owner G0 section 6).** The
loop samples its own detector once per cycle, before the retry, and every other detector that
guides a move there too -- but only one with no Detect block (D-018): no move of the loop emits a
block at its anchor. In a canonical project (one with an end-effector contract) such a block is
refused before any file (`BinPicking.refusals()`, the same sentence in `validate()`), naming the
block, the loop detector and what would be lost. The loop detector's own block stays a note (the
loop samples that detector anyway), a project with no loop gets the linear vision tree, which emits
every block at its anchor, and a legacy project still drops such a block silently (KNOWN_DEBT;
ADR-0022 D9).

## 4c. The cell namespace (ADR-0016 Decision 3) — one datum, one place

`RobotSpec.cell_namespace`, canonicalised on parse AND on assignment (`r3` → `/r3`, `/` →
`''`, an illegal ROS name raises — a bad namespace yields a bring-up whose nodes silently
never find each other). `generator/namespace.py::cell_namespace()` is the only reader;
`servo_yaml.cell_namespace` re-exports it so the servo entry point is unchanged.

| namespaced | left GLOBAL |
|---|---|
| `move_group`, the controllers (`ros2_control_node` **and** the spawners' `--controller-manager`), `robot_state_publisher`, `scene_manager_node`, `servo_node` + the topics servo derives, the robot's own cell bridges, `trainit_run_bt`, and the gripper action in `bt_params.yaml` | `/tf` + `/tf_static`, the cameras (incl. the camera-side bridge, via the generated `CAMERA_BRIDGE_PACKAGES`), `/perception/<detector>/…`, and RViz |

Two notes that are easy to lose:

- **`trainit_run_bt` is namespaced and the detector nodes are not**, in the same launch
  file. The BT is a client of this robot's `move_group`: since TMR 2.1.0 its MoveGroupInterface
  resolves the actions and services in its own node namespace (ADR-0023), so a root-level BT
  finds no `move_group` there; it warns every 5 s with the names it waits for and exits 2 after
  `moveit_wait_timeout_s` (60 s by default; TMR 2.0.0 waited for ever, in silence). The detectors
  publish `/perception/<name>/detections`, which `DetectObject` reads by absolute name.
- **`scene_manager_node` needs only the namespace**, not the remapping a hand-made
  namespaced bring-up carries: TMR 1.5.0 asks for `apply_planning_scene`/`get_planning_scene` by
  RELATIVE name, so a namespaced instance reaches its own `move_group` and two robots stop
  writing into one global service.

**Gating.** Every namespaced site is behind a Jinja `{% if cell_namespace %}` (or a
`{% set %}`-built fragment — `{{ }}` expressions are not touched by `lstrip_blocks`, so an
empty namespace renders the pre-ADR-0016 bytes exactly). What a namespace does NOT fix is
reported into `generation_manifest.yaml` instead of being silently rewritten: captured
bridge parameters and `arm_joint_states_remap_to` that are absolute ROOT names, the copied
`moveit.rviz`, and the Gazebo backend (whose `/controller_manager` name comes from the URDF
plugin tag, not the launch).

## 4d. The candidate ranking (ADR-0017) — a DETECTOR parameter, assembled from three parts

WHICH object to pick is configuration (ADR-0014) and the PUBLISHED ORDER of the detection
array is where the answer is carried (ADR-0016 Decision 1). ADR-0017 fixes who computes
it: the **detector**, as one more detector parameter. TSA therefore grows one enum field
and no new concept — `DetectorSpec.candidate_rule`, enum-validated the way `method` is,
travelling the road every tuned parameter already travels (GUI → `DetectorSpec` →
`perception.yaml` → `make_detector`).

Nothing is written onto the application. A runtime that ranked would have to know what
`visible_fraction` means, and the perception contract exists precisely so it does not:
TMR 1.6.0 filters and truncates and never sorts.

**The rule needs three numbers, and they come from three different parts of the model** —
which is why the GENERATOR is where they are joined (`generator/perception_yaml.py`):

| number | lives on | why there |
|---|---|---|
| `candidate_rule` | `DetectorSpec` | it is a detector parameter (Decision 1) |
| `tool_radius_m`, `tool_reach_mm` | `GripperSpec` | a tool property belongs on the tool; the grip window (`tool_reach_mm`) was a Python default inside a node, i.e. not reproducible from a bundle |
| the GRASP FACE — `grasp_face` + the parameters *that* face reads | the grasp-target `SceneObject` | it is a property of the object that is picked: `grasp_face_params` when the face was MEASURED on its CAD (§4f), `SceneObject.face_radius_m()` from shape + dims when it was not. The equal-AREA disc, because that is the denominator `visible_fraction` divides by |

**Computability has TWO halves, and TSA models both** (`model/perception.py`:
`METHOD_PROVIDES`, `CANDIDATE_RULE_TERMS`, `TERM_NEEDS`, built from TSA's committed snapshot of
trainit_perception's vocabulary — ADR-0021, amendment M2.2 — and held to the runtime's
`computable_terms()` over every configuration by `test_tp_registry.py`). A term is computable when
the METHOD produces what it reads *and* the PARAMETERS it needs are set. **Which parameters those
are is the face registry's answer, not a table here** (ADR-0018 Decision 6 — see §4f). Modelling only the parameters would have been a real
defect rather than a simplification: the only detector TSA generates a node for is the
colour mask, which publishes a score, a box and a mask area and runs **no 3D fit** — so
`pose_quality`, and with it every `candidate_*` rule, is uncomputable on it, and a GUI
offering those four would have offered four rules that stop the node. That is why the
`score_*` family is in the enum and why the dropdown greys rather than lists. An
EXTERNAL-NODE method declares no capabilities to TSA, so it is judged on its parameters
only — refusing it on a capability TSA merely failed to know would be refusing the expert
seam, and the node still checks the file it reads.

**Gating is the same rule as everywhere else in v7:** each key lands only when it has a
value. `candidate_rule` only when set; the derived radius only when a rule is set (a
derived number must not appear in a bundle whose ranking was never configured — the golden
*has* a grasp target and no rule); the two tool numbers whenever the tool declares them,
because `clearance` ships as a published **term** without being a ranking default
(Decision 4). `perception.yaml` is compared YAML-semantically against the golden, so a key
that always appeared would silently re-bless every published bundle.

**The refusal (Decision 3) is the one place TSA stops instead of warning.** Every other
validation problem is a manifest warning, because a partial bundle still helps. A rule
whose terms are not computable is not a partial bundle: it generates, it runs, it publishes
the model's own order under the name of a ranking, and it looks like a perception fault.
So `refuse_uncomputable_candidate_rules()` raises — from `Orchestrator.generate()` before
the first file exists, and again inside `build_perception_yaml()` so no other caller can
produce the file either — with the missing parameter and its home named. The GUI previews
the same pure check (`AssistantController.candidate_rule_problem`) while the user is still
choosing.

**The version pin rides the existing mechanism.** A parameter the deployed runtime does
not know merges silently into its params and is IGNORED, so the app's `package.xml` pins
`trainit_perception` to the floor the project's parameters demand: `0.2.0` for the D-015
noise filters, `0.3.0` or `0.4.0` for the parameters in `ranking_param_names()`, and the HIGHEST
of them when several are used. Every floor is trainit_perception's own date for the parameter,
read from TSA's snapshot of its vocabulary (ADR-0021, M2.2), and the pin list is derived from
the same call the generator writes from — so neither a new ranking parameter nor a new face
family's parameters can reach a bundle without moving the floor. A face family newer than
the deployed runtime is a different and safe failure: `make_face` raises on a name it does
not know, so the node stops at start-up and names the faces it has. The pin exists for the
SILENT case.

**What stays out of TSA:** the loop. The retry over candidates, the stall recovery and the
refill are the cell's business logic, they are the behaviour tree §4b generates, and the
`bin_picking` template only *notes* a loop whose detector carries no rule — because the
runtime retries in the order it was given, so an unranked loop is a shuffle wearing a
ranking's name.

## 4e. The two kinds of detection method (D-015) — and the LAUNCH half of the second

`trainit_perception`'s registry names two kinds, and the difference is **who runs the
code**:

| kind | implementation | what TSA emits |
|---|---|---|
| `pure-python` | a `Detector` subclass in the runtime registry (`color_mask`) | the `perception.yaml` entry **and** one `trainit_perception` `detector_node` per entry |
| `external-node` | a separate node — C++/PCL, a learned model, a vendor SDK — publishing the SAME `Detection3DArray` contract on `/perception/<name>/detections` | the `perception.yaml` entry **and** that node, from `DetectorSpec.external_node` |

Until the second row's right-hand column existed, TSA could *name* an external-node method
(`DetectionMethod.CUSTOM` / `PCL_CLUSTER`) and emit **nothing at all** for it. That was not
a missing convenience: a project could configure a detector, record it in
`perception.yaml`, bind it to a waypoint and rank it, and the bundle launched nothing on its
topic — so `DetectObject` timed out, three layers from the cause, and it read as a
perception fault. `pcl_cluster` was a reserved name with no way to be used at all.

`model/external_node.py` carries exactly what `launch_ros`'s `Node` needs — `package`,
`executable`, an optional `node_name` and `parameter_file`, inline `parameters`,
`topic_parameters` and `remappings` — and `generator/external_node_launch.py` turns it into
the `Node` entry the app emitter puts into `trainit_bt.launch.py` **next to** the
pure-python ones, GLOBAL like them (§4c). One bundle, both kinds, one way to launch them.

**The camera contract is resolved once.** The input topics are *not* duplicated onto the
new model: `DetectorSpec.rgb_topic`/`depth_topic`/`camera_info_topic` (falling back to
`CameraSpec`) are already the project's single home for them, so `topic_parameters` says
only what the node *calls* each role and the value comes from
`DetectorSpec.input_topics()`, which the `input:` block of `perception.yaml` now uses too.
A second home for the same three topics is how the file TSA writes and the node TSA
launches end up looking at different cameras, silently.

**Refused before the first file** (the `refuse_uncomputable_candidate_rules` pattern, §4d):
a block with no `package` or no `executable` (naming which), a block on a *pure-python*
method (both halves would publish the same topic, interleaved), an unknown input role, an
empty parameter name, and two detectors resolving to one node name. An external-node method
with **no** block at all stays a **warning**: starting a vendor stack yourself is a
legitimate choice, and the app emitter says so app-agnostically — `perception.yaml` is
emitted whatever the application type, so the fact does not belong in one template's
`validate()`.

**The external package is documented, never declared** — the one judgement call here. The
generated `package.xml` gains a comment naming it and generation warns, but no
`<exec_depend>`: ADR-0003 has Community installing via rosdep/apt, and `rosdep install` over
a source workspace fails with *cannot locate rosdep definition* for a package name that is
neither a rosdep key nor present in that workspace. An external detector is normally
customer IP shipped separately, so the manifest entry would turn "this detector is not
installed" — a readable `ros2 launch` failure, at the moment you use it, naming the package
— into "the whole workspace will not install", for everyone, including people who never run
that cell. The dependency is real, but it is a **deployment** fact, so it is recorded where
a person looks for dependencies instead of declared where a tool would enforce it.

**What stays out:** the external node's own parameters. TSA passes the file its author
ships and the inline values the project sets; it does not re-author another package's
parameter schema. Only the camera contract is unified, because only the camera contract is
a *contract*.

## 4f. The grasp face is MEASURED on the object (ADR-0018)

§4d's best-measured term, `visible_fraction`, divides by the grasp face's area. TSA derived
that area from the object's axis-aligned bounding box — `sqrt(x·y/π)` — which on the
reference capsule gives **28,10 mm against a true 20,95 mm**, because the box is inflated
by the pull tab: +34 % on the radius and **+80 % on the area**. The picker replaces the
ESTIMATOR and changes no contract: `SceneObject.face_radius_m()` already existed, already
had a home in `perception.yaml` and already reached the detector.

**The picker is a BUILD-TIME tool whose entire output is a few scalars.** A `GraspFace`
answers three questions and all three return numbers, so nothing at runtime ever needs a
mesh — not the detector, not the motion runtime, not a generated bundle.

- `geometry/mesh_face.py` — binary/ASCII STL and OBJ in pure numpy, a Möller–Trumbore ray
  cast, a coplanar flood fill (normal angle **and** plane offset, over welded-vertex
  adjacency, so a parallel face 1 mm away is not merged in), three orthographic rasters and
  the measurement: area, the equal-area radius, the in-plane half-extents from a PCA in the
  face plane — the same decomposition `plane_ransac` performs on the depth at run time.
- `gui/face_picker.py` — the Qt skin: `OrthoView` is the first widget in the wizard that
  paints and handles a mouse, and it stays one small class. The raster's own id buffer
  answers the click, so the mouse path cannot disagree with the ray path. It follows the
  live tuner's shape — a `QTimer` polling a cache, widgets touched only from the GUI
  thread — rather than inventing a second pattern.
- **Zero new dependencies** (ADR-0018 Decision 2, measured): VTK is faster and rotates
  freely but `python3-vtk9` has no rosdep key and pulls an MPI stack, which ADR-0003
  forbids; `pyqtgraph.opengl` has the clean rosdep key and no triangle-level picking, so
  the ray cast gets written anyway and the dependency buys nothing. Three fixed views (six
  with the flip) of a small part are interactive in numpy: 24 ms on a 252-triangle capsule,
  0,3 s on a 4334-triangle one.
- **The unit is explicit, because getting it wrong is SILENT** (Decision 4): a mesh carries
  no unit, this workspace itself mixes metres and millimetres, and the survey's own run
  reported a 51 m² face and a 4042 mm radius without raising. The scale is a field
  (defaulted from `SceneObject.scale`) and a face outside the band the RUNTIME refuses —
  the runtime's own band, from TSA's snapshot of its registry — is refused here instead of
  becoming an absurd denominator in a bundle.
- **STEP is an optional exact path behind a probe** (Decision 3): `freecadcmd` on `PATH` or
  `TRAINIT_FREECADCMD` (an AppImage works), probed the way `pxr` is, never declared in
  `package.xml`. It returns named B-rep faces with their true area, axis and radius, so a
  click on the tessellation can report the EXACT number — and `match_measurement` refuses
  to guess when centre, normal and area do not all agree, because a wrong match would
  publish another face's radius. Without FreeCAD the picker works on STL/OBJ and says so.

**The measurement lands on the OBJECT** (`SceneObject.grasp_face`, `grasp_face_params`,
`grasp_face_source`), where `shape` and `dims` already live, because that is what was
measured; the generator turns it into detector parameters exactly as it always turned the
bounding box into them. Clearing it returns to the approximation, which is the only honest
way back. `grasp_face_source` is provenance: "measured" vs "approximated" is the whole
point, and a bundle that cannot say which is back where it started.

**TSA reads the face registry from a committed snapshot, and keeps no hand-written copy**
(Decision 6, superseded for the vocabulary by ADR-0021's amendment M2.2: a versioned snapshot,
never the installed package). `model/faces.py` answers from TSA's committed
snapshot of trainit_perception's vocabulary (`catalog/tp_registry_snapshot.yaml`, the
canonical YAML of `trainit_perception.vocabulary.export()`): the parameters a face takes (in
the order it declares them, the order a generator writes them), the terms it makes computable,
the plausible size band and every parameter's version floor. This closes a verified asymmetry
— a hand-written `rect` face was **accepted by the runtime and refused by TSA**, because the
ranking-parameter list was a fixed four-tuple that never wrote `grasp_face` (a key the
runtime already read) and the computable-terms mirror hardcoded `expected_radius_m`. A face
family added to the registry is configurable, pinned and refusable here once the owner has
classified it (a `TP_TIERS` row in the catalog, owner decision C4) and the catalog is
refreshed — no TSA code changes: the picker renders one field per parameter *that* face
declares. (Refusal texts list the registered faces, so the 7.3.0 pin file
`test/data/perception_vocabulary_m2_2.txt` is re-pinned deliberately when a face arrives.)

Until vNext M2.2 the registry was PROBED at generation, with a `disc`-only fallback when no
perception package could be imported — and the same project then generated differently on
two machines (a face TSA could not look up was accepted on one and refused on the other; a
0.4.0 parameter lost its pin). The snapshot always answers: a face it does not register is
refused everywhere, as the node would refuse it at start-up, and generation imports nothing of
trainit_perception (`test_m2_2_regressions.py` proves it in a fresh interpreter with TP made
unimportable). What a face measures (its extent terms) is not part of what TSA knows.

## 4g. The Canonical Task Program (ADR-0024, vNext M3.3) — a program, not a template

`application.type: task_program` (`schema_version: 4`, R-M3-01) is the canonical application: the
project says what the robot does as an automation engineer reads it -- APPLICATION -> TASK -> STEP
-> WAYPOINT / PATH -> PROFILE -> PROVIDER -- and TSA turns it into the data TMR 2.2.0 executes. It
sits beside the three templates, which are untouched (their generated bytes included).

- *The model* (`model/program.py`, strict: an unknown key is refused naming the accepted keys, and a
  key in the wrong place carries a hint). A WAYPOINT is a target only (`tcp`: position +
  orientation in a `frame`, for a `tcp`; or `relative` to another waypoint along the `base` or
  `tool` axes; `joint`: `joints` or `named`) -- a motion type, a speed or a blend on a waypoint is
  refused with "a waypoint says WHERE, never HOW". A TASK is an ordered list of steps: `move`
  (`type: ptp | lin | circ | free`, `to`, `profile` and/or inline limits, a circle's `via`, a
  `provider` for ptp/free), `path` (a path's name) and `dwell` (`ms`); `process` (M3.5, §4g.1) and
  `detect`/`grasp`/`release`/`tool` (M3.6, §4g.2); `task` calls and the control steps `repeat`/
  `retry`/`fallback` (M3.7, §4g.3); `check` is refused by name with its phase. A PATH is ONE motion through >= 2 waypoint states joined by
  `lin | circ` segments, `continuity: stop | blend`, a path-level profile and Cartesian
  `goal_tolerance`, `provider`, `frame`, `tcp`; a dense path has a `source.generator` (TMR's
  generators; a file source is M3.4). `motion_profiles` are M3.2's `MotionProfile`s (`default` is
  TMR's own base profile: reserved). `main_task` runs; `runtime.allowed_start_tolerance`.
  In memory the program is `ApplicationSpec.program`; in the file its sections sit directly under
  `application:` (a pre-validator moves them), and a save writes exactly the keys that were set.
- *Validation and resolution* (`applications/task_program.py`, `plan_program`): every reference
  (R-M3-03), the robot (a `frame` must be a link of the compiled description -- compiled only when
  a frame other than the base frame is named; `tcp` = the tip link until M3.4; named states;
  joint counts, R-M3-05), the shape and geometry of each path (TMR's rules, mirrored where every
  state is a taught pose in one frame), and each move and path as ONE M3.2 request
  (`model.capabilities.resolve`: its own provider, else `application.providers`, else the registry
  default). Every motion must resolve an explicit speed AND acceleration (no hidden default); a TCP
  limit is the scaling of the bundle's `pilz_cartesian_limits`, the smaller of a percent and a TCP
  limit wins, and a move's effective scaling below 1 % is refused (MoveWaypoint clamps there). A
  move's goal tolerance or planning budget comes from its profile (TMR reads them only there); a
  segment-level goal tolerance inside a path is refused (owner, M3.2 close). A refusal names the
  step, the request, the provider and what to change; what TSA cannot prove (where the robot is
  when a path starts after a joint move) is a warning, and TMR checks it at run time.
- *The bundle* (`generator/program_files.py`, from the validated plan only): `bt_params.yaml`
  `task_parameters` -- one entry per waypoint, its target and the move that reaches it (`motion`,
  `planner` = the resolved row's planner MODE, `profile`, `speed`/`accel` in percent, `aux`); a
  waypoint reached by two kinds of move gets a second entry `<waypoint>__2`; a comment above each
  entry names the steps and the resolved provider. `config/paths.yaml` (TMR's schema 1, provider and
  scalings written). `config/motion_profiles.yaml` (what TMR takes only from a profile: planning
  budget, goal tolerance), loaded by the launch instead of TMR's default file. `bt_trees/
  task_program.xml`: `MainTree` (the scene, then `<SubTree ID="task_<main>" _autoremap="true"/>`)
  and one `task_<name>` BehaviorTree per task (BehaviorTree.CPP 4 refuses the design's `task.<name>`:
  a '.' is forbidden in a tree id) -- a move is ONE `MoveWaypoint`, a path ONE
  `FollowPath`, a dwell ONE `Sleep`. The runner's `process_path_backend` is `cartesian_waypoints`
  iff a path resolves to that provider (the M3.2 debt S2, closed); `planner_mode` is the free
  provider's mode. `runtime.allowed_start_tolerance` is move_group's
  `trajectory_execution.allowed_start_tolerance`. `README.md` is the program, readable.
- *Floors:* `FollowPath` and MoveWaypoint's `profile`/`frame`/`tcp` keys are TMR 2.2.0 (b399a33),
  read from the generated files like every floor; a template's file that does not parse keeps its
  old floor (`TASK_PROGRAM_CAPABILITIES`).
- *The wizard* (YAML-first, decision 5): Step 6 shows the program's type and size instead of the
  type selector; Step 7 shows a read-only overview and a "Validate the program" action (generation's
  refusals in red, the run-time warnings in amber); the controller's template setters refuse a task
  program. Nothing in the wizard rewrites it; a save round-trips it key for key.

### 4g.1 Process events and captures (vNext M3.5, TMR 2.4.0)

- *The model* (`model/program.py`): ONE generic `ProcessEventSpec` -- `kind` process_on |
  process_off | set_value | capture | dwell | external, each with exactly the fields it takes
  (`channel`, `value`; `detector`, `out`, `class_id`, `timeout_ms`; `ms`), `sync: blocking` only
  (`async` is R-M3-09). `channel` is a LOGICAL id (owner guardrail, CELL_SCENE_IMPORT_REVIEW
  decision 4): `NAME`, never a topic path, a hardware address or a symbol (refused by name, a key such
  as `topic`/`address` too). A task step `process` is one event where the robot is (no `at`, no
  dwell: that is the `dwell` step); a path's `events` are `PathEvent`s anchored `at` a state (name or
  index; after_arrival default, before_departure) or a segment boundary (start / end); a key that
  places an event inside a segment (`fraction`, `distance_m`, `time_s`, `position`, ...) is R-M3-09
  with the fix ("split the segment with a waypoint").
- *Planning* (`applications/task_program.py`): each anchor lands on the STATE the robot stops at (a
  segment's start = before departing its first state, its end = after arriving at its last);
  refused: an unknown or ambiguous state (anchor by index), a segment out of range, before_departure
  at the last state, a blend into an event state (never moved), events on a dense path, a capture's
  unknown detector (R-M3-03) or one that cannot be triggered (R-M3-16: a `trainit_perception`
  detector serves `/perception/<name>/detect`; an external node must declare
  `external_node.detect_service: true`), a capture key that would overwrite a waypoint, the runner's
  entries or another capture. The split: at each event state, a `FollowPath path= from= to=` run up to
  it, then its events (after-arrival group, then before-departure group, each in program order), and
  a trailing run -- K event states, about K+1 runs; never a node per sample. Captures are
  `captures.<n>` per task in execution order (or `out`). `process_controller: ros` iff an event of
  the main task uses a channel.
- *Bundle*: `paths.yaml` `events: [{at: {state, when}, kind}]` (where it stops; the action is the BT
  node) with comments in the program's terms; the README's "Process events" and "Process and
  perception" sections. *Floors*: `ProcessEvent`, `detect_trigger`, `follow_path_range`,
  `path_events` (TMR 2.4.0, 6a2414e), read from the trees and `paths.yaml`.

### 4g.2 Manipulation (vNext M3.6, TMR 2.5.0, ADR-0026)

- *The model* (`model/program.py`, strict): `DetectStep` (`detector`, `source: candidates |
  detections`, `out` default `candidates`, `keep`, `class_id`/`approach` for detections only -- a
  detections source names its fallback `approach`, a unit vector in the base axes), `GraspStep` (a
  candidate set -- `candidates`, default the task's latest detect, `index`, `orientation_from`,
  `orientation: align_approach` the only value -- or a taught target `at`; `standoff_m`, `profile`,
  `approach_profile`, `retract: {along_approach | dz}`, `eef`, `site`), `ReleaseStep` (`at`),
  `ToolStep` (`actuate | verify`); `relative.frame: target_local` (a taught reference: its own axes,
  numerically equal to `tool`; the portable spelling for later bound targets). A missing key names
  what it is for; a key in the wrong place gets a hint (`trigger`, `pregrasp`, `pick`, `place`, ...).
- *Planning* (`applications/task_program.py`): `plan_detect` -> ONE `DetectCandidates` in the base
  frame; `plan_grasp` -> `SetWaypointFromCandidate` (from a set; the target's PreGrasp / Retract
  resolved by TSA along its tool +Z at a taught target) + the idiom (PreGrasp `MoveWaypoint` ptp ->
  Approach lin -> `CloseGripper` -> `VerifyGrasp held` iff require_grasp_confirmation -> Retract lin);
  `plan_release` (PreRelease ptp -> Approach lin -> [`VerifyGrasp held` iff grasp confirmation and
  continuous state] -> `OpenGripper` -> [`VerifyGrasp released`] -> Retract lin, ADR-0022 D4);
  `plan_tool` -> `CommandActuator` / `VerifyGrasp`. Each idiom motion is resolved like a move step
  (`_bound_move`: profile, provider, explicit limits) into a derived entry `<task>__s<i>_<role>`
  ('__' is reserved: no user name has it). Refused before any file (R-M3-02/03/05/13): no contract,
  an unknown eef / site / actuator / target, a site TCP other than the tip link, a set no earlier
  detect produces, an index past its keep, projected candidates without the step's standoff, a
  reference that is not a base-frame pose, a TP detector node as a candidate producer, a set name
  colliding with a waypoint, a derived entry or a capture, a grip actuated by a tool step, a
  verification no deployed adapter can observe. A refused detect still defines its set (no second
  refusal on its grasp).
- *Bundle*: the derived entries in `task_parameters` (with their comment), the README's rows and
  "Manipulation" section. *Floors*: `DetectCandidates`, `SetWaypointFromCandidate` (TMR 2.5.0,
  cb4d1a2), read from the trees; TMR 2.5.0 brings `trainit_interfaces` 0.2.0.

### 4g.3 Task composition (vNext M3.7, TSA 8.0.0)

- *The model* (`model/program.py`): `task: <name>` (a call; a mapping is refused: data is bound by
  name), `RepeatStep {cycles >= 1 | -1, body}`, `RetryStep {attempts >= 1, body}`, `FallbackStep
  {bodies: >= 2}` (forward references resolved for pydantic 1.x and 2.x), `Task.description`
  (one line), `Task.inputs` / `outputs` (`TaskData {name, type: candidates | capture}`, each name
  once). `walk_steps` labels a step `3`, `3.2` (a repeat or retry body) or `3.2.1` (the second
  alternative of a fallback); `_place` locates a load error with the same label. Hints refuse a
  condition (`while`, `if`, `until`), a `call`, a `loop`, a retry's `step:`.
- *Planning* (`applications/task_program.py`): `_call_order` refuses an unknown callee and every
  cycle (R-M3-15, the chain once, rotated to the first declared task) and orders the tasks callee
  first -- declaration order without calls, so a call-free program plans exactly as before (moves,
  `__2` variants, derived entries reordered by declaration). `plan_task` plans a task's steps
  (`_plan_steps`, recursive over bodies, labels as above; derived entries `<task>__s<label_>`) and
  leaves a `TaskSummary` its callers check: where it leaves the robot (`_ENTRY` = unchanged), the
  candidate sets it writes (transitive), its outputs, and what its grasps need of each candidate input
  (`_grasp_needs`: an index below `keep`, a standoff for projected detections) -- checked at each call
  against the set passed, or passed on to the caller's own input. `_plan_call`: inputs held by name and
  type (R-M3-03), the needs, the callee's undeclared sets dropped after the call, its outputs held.
  `_plan_control`: each body planned exactly as a task's steps; where the robot is at a body's start
  is what every way into it agrees on (`_exit`, a pure function of the model) -- else unknown, with
  the reason (a retry, a next cycle, a fallback's next alternative) in the path-start warning; after a
  fallback what EVERY alternative holds (`_common`). After all tasks: `_running` (main_task and what it
  calls), `_shared_captures` (one capture key written by two running tasks: R-M3-02), the "never runs"
  warning. The main task takes no input (R-M3-15); an input no step reads, an output not produced on
  every way the task succeeds: R-M3-03.
- *Bundle*: one `BehaviorTree ID="task_<name>"` per task; a call `<SubTree ID="task_<name>"
  _autoremap="true"/>` (one blackboard); `Repeat num_cycles` / `RetryUntilSuccessful num_attempts` /
  `Fallback`, each body one `Sequence` (`_steps_xml`). `config/tasks.yaml` (`program_files.tasks_yaml`,
  schema 1: `main_task`, each task's `tree`, `description`, `inputs`, `outputs`) -- read by no runtime:
  no runner parameter, no TMR reader, no floor. README rows carry the labels; a paragraph explains the
  control steps when used. *Decision 7*: `model/enums.py` `DEPRECATED_APP_TYPES` -- `gluing`,
  `follow_path`, `waypoint_replay`, `cnc` stay enum members (old code and data name them) and are
  refused at load (`model/io.py`, `ApplicationSpec`) naming `task_program`; the wizard no longer lists
  them.

## 5. The Isaac grasp adapter (sim-specific, NOT generated)

Isaac-internal physics can't be synthesized generically, so the grasp adapter is a
**hand-authored Script Node in the `<cell>_isaac` package**. The TSA owns only the ROS
**contract** (`/isaac_gripper_cmd` Bool = close/open; `/isaac_release_policy` Bool =
freeze/gravity) + the config, and ships a **reference template**:
`resources/isaac/isaac_manipulation_adapter.py` — a parametric fixed-joint grasp (the v14
method that stabilised the 20 bottles) that reads `db.inputs.cmd` + `db.inputs.freeze`.
The user copies it into their `*_isaac`, edits the CONFIG block, wires two OmniGraph ROS2
Subscribers, and tests it in Step 1. It was distilled from an earlier cell's
`isaac_dynamic_attach_script.py` (not in this workspace).

## 6. TMR BT vocabulary (what an application may use)

25+ leaf nodes; the assistant emits: `MoveWaypoint` (data-driven), `Close/OpenGripper`,
`AttachObject/DetachObject`, `SetAttachedCollisionCheck`/`SetReleasePolicy`,
`AddCollisionObject` (from-scratch path only), — vNext M3.3 — `FollowPath` (TMR 2.2.0) and the
BT.CPP built-in `SubTree` for a Canonical Task Program (§4g), — vNext M3.5 — `ProcessEvent`,
`DetectObject trigger="true"` and `FollowPath from= to=` (TMR 2.4.0, §4g.1), — vNext M3.6 —
`DetectCandidates` and `SetWaypointFromCandidate` (TMR 2.5.0, §4g.2), — vNext M3.7 — `SubTree`
calls and `Repeat` / `RetryUntilSuccessful` / `Fallback` in a task program (BT.CPP built-ins, §4g.3), — v4 (D-015) — `DetectObject` +
`SetWaypointFromDetection` plus the post-reset settle `Sleep`, — ADR-0015 — `ServoTrack`
for a move configured `execution: servo`, and — ADR-0016 — `CheckRobotState` +
`EnableProcess`/`DisableProcess` in the emptying loop. Trees are linear for every
application **except `bin_picking`**, which uses the BT.CPP v4 built-ins `Repeat`,
`Fallback`, `RetryUntilSuccessful` and `Script` (§4b) — the first generated tree with real
control flow, and it needed no new TMR node to get it. The TMR can run pick-and-place /
vision-guided picking / bin picking / tending / palletizing / kitting today, and cartesian
process paths (gluing/follow-path) with a bit more wiring. No CNC-as-machine, no force
control.

ADR-0016 Decision 2 is the other outstanding runtime contract: `DetectObject` gains a
`keep` port (writing `<out_key>.count` and `<out_key>.<i>.*`) and
`SetWaypointFromDetection` gains `index`. The `bin_picking` template emits against that
contract; until it lands in TMR the generated loop loads but reads candidate 0 every time.

`ServoTrack` is the ONE node here the assistant emits that TMR does not implement yet:
TSA configures and launches the out-of-process `moveit_servo` node (ADR-0015 Decision 5 —
on Humble `moveit_servo` 2.5.10 is a node, not a library), and the BT node that drives it
over ROS, plus the controller arbitration it owes (pause Servo before any trajectory
execution on the group, unpause after — a `JointTrajectoryController` accepts the servo
topic while an action goal is in flight and the two fight silently), is TMR work.

## 7. Status & roadmap

**Done (committed):**
- Phase 0 — TSA + TMR under git (one repo each), versions bound.
- Phase 1 — model extensions (mesh objects, per-dynamic attributes, per-move flag, base ref).
- Phase 2 — TMR BT nodes `SetAttachedCollisionCheck` / `SetReleasePolicy` (+ gtest).
- Phase 3 — `SceneLoaderMoveitConfigEmitter` + `scene.yaml` + app wiring + bundle README.
- Phase 5 — this doc, the public README, the generalized Isaac adapter.

**Done (v3/v4):** the two-phase 8-page wizard (`gui/`), and — v4.0.0 (D-015) —
the dedicated **Perception step** (live HSV tuner running trainit_perception's pure
`detect()`, noise filters, continuous/on-demand trigger, settle) + camera guidance as a MOVE property (v4.1/D-016: detector dropdown +
dx/dy/dz + EEF-orientation policy on the Move inspector; automatic detect row) + the vision emitters (perception.yaml,
detector_node, depth_image_proc isaac branch, camera-variant URDF/SRDF rewrite).

**Done (v7 / ADR-0015 + ADR-0016):** the per-move `execution` axis (`plan | servo`) with
`servo.yaml` + a gated `servo_node`; the `cell_namespace` robot slot (§4c); the
`bin_picking` emptying loop (§4b); the package version reunited with the product label at
`7.0.0`.

**Done (v7.1 / ADR-0017):** the candidate ranking as configuration (§4d) — `candidate_rule`
on the detector, the tool's footprint radius + grip window on the end-effector, the face
radius derived from the grasp target, a generation-time refusal for a rule whose terms are
not computable, and the `trainit_perception >= 0.3.0` pin that stops a ranked bundle from
running on a runtime that ignores the rule.

**Done (v7.2 / ADR-0018):** the grasp face is MEASURED on the object (§4f) — the
three-orthographic-view picker in pure numpy with no new dependency, the explicit unit and
the refusal of an implausible face, STEP behind an optional `freecadcmd` probe, the
measurement stored on the grasp target with its provenance, and the end of TSA's own copy
of what a face needs: the parameters, the extent keys, the terms a face enables and the
plausible size band are all asked of `trainit_perception`'s face registry, so `rect` (or
anything registered after it) is configurable, pinned and refusable without a TSA change.
*(7.4.0, vNext M2.2: the registry is read from TSA's committed snapshot of it instead, §4f —
the same project now generates the same bundle with or without a perception package; a new
face needs the snapshot refresh and the owner's classification row.)*

**Done (7.5.0 / vNext M2.3, ADR-0021):** a bundle is self-describing — its manifest records the
identity of the capability snapshot that compiled it and names no machine (§3, the manifest).
The golden's manifest is compared as a CONTENT record (paths, actions, hashes, sources); its
provenance is checked against the committed catalog by `test_bundle_provenance.py`.

**Done (7.6.0 / vNext M2.4, ADR-0021):** a bundle says what it needs to run, in its
`package.xml` files: the TMR floor of each package (with its reason), the policy runtime iff the
package uses it (§3). The golden and the policy reference trees were re-blessed with it (owner,
2026-10-08); between a generator change and its re-bless, `test/pending_rebless.py` lists the
exact lines a pinned reference may differ by.

**Done (7.7.0 / vNext M2.6, ADR-0022):** the end-effector contract, opt-in per project
(`model/end_effector.py`, `generator/eef_contract.py`). The short form is stored as written and
the long form derived on demand; one pre-validator sets the guard token `actuation: contract`,
and `schema_version` is 3 iff a contract exists. `refuse_eef_contract` runs before any file in
both generation entry points (R1-R23, each naming the device, actuator, site and backend). A
canonical bundle carries the `eef` block (byte copies of the shared conformance fixture pin its
normalised form), trees with logical ids only (CloseGripper/OpenGripper `eef`/`site`,
CommandActuator, VerifyGrasp per `require_*`, no SetReleasePolicy, the bin-picking
verdict/fault/holding gate), `scene.yaml`'s `eef/<id>/engaged`, the TrainIt adapter rows appended
to BRIDGES, and its dependencies. The EEF adapter rows are typed in TSA's catalog (ADR-0021's
named exception M2.6-1); legacy end-effector bridge rows are identified by catalog data (R18). The
GUI changes were guards only (vNext I0 adds the End effector section of Step 7: ADR-0022 amendment I0): the
gripper setters and the Step-2 ingest leave a canonical gripper untouched, and Step 7 keeps a canonical project's tool actions as they are (an `actuate` action in
its own read-only block, a grasp's `site` carried through the block). A composite BUNDLE is
refused (R22); legacy projects generate the same content.

**Done (7.8.0 / vNext M2.7, ADR-0022, ADR-0015 M2.7-3 and M2.7-4):** three changes, none of them
for a project without a contract (the effective planner since vNext G0, owner section 8).
- *The composite mock bundle (ADR-0022 R22 narrowed, R24-R26).* R22 now allows a composite bundle
  when every deployed mode is mock and `trainit_mock` serves it, the robot description its mock
  bring-up loads can be named and compiled (fail closed), no `<ros2_control>` block of that
  description (nor the block TSA writes from the planning group) claims an actuator's joint or a
  mimic follower of one, there is no `command_joint` pseudo-joint echo, and every movable tool
  joint off the TCP paths has a state publisher; isaac, gazebo, real and mixed bindings stay
  refused (vNext M6, ADR-0020; R20 unchanged). The composite mock adapter row carries ONE relative
  launch remap, `eef/<id>/joint_states` -> `joint_states`, so the tool joints reach the robot's own
  `joint_states` (robot_state_publisher, MoveIt) in its namespace; the scene-loader SRDF merge
  applies `srdf_builder`'s composite filter (only the planning group's captured states). For every
  canonical kind: R24 (owner Q7-i E1) -- an actuator that drives a movable joint on the kinematic
  path `eef_parent_link` -> a site's TCP is `follow_joint_trajectory`, on a single-DOF joint; R25
  (E2) -- such a joint (a mimic follower judged by its master; planar/floating included) is never
  left without one. The command interface follows actuator semantics, not the functional label: a
  jaw off the TCP path keeps SetBool or GripperCommand. R26 (owner 11.1) -- under a cell namespace
  the pseudo-joint echo topic (the kept merger row's gripper input, or the default) is relative;
  an absolute or private one is refused, naming the robot namespace, the end effector, the topic
  with its source and the expected relative form. G0 (owner section 9) ratifies R22, R24, R25 and
  R26 as canonical validation rules and words each refusal by its kind. A limit of the current
  canonical model, where the design may be valid, says `UNSUPPORTED_BY_CURRENT_CANONICAL_MODEL`:
  - R22: a composite beyond mock, a mock binding other than `trainit_mock`, a pseudo-joint echo,
    a tool joint nothing publishes;
  - R24 and R25: a planar or floating joint on a TCP path;
  - R24: a site's grip whose joint moves a TCP, read as a jaw whose TCP is on its finger link;
  - R25: a TCP below the `command_joint` pseudo-joint.

  A physical or contract invalidity carries no token. R25 names a mimic follower's master, and
  R22 states the adapter's echo refusal only for a multi-site device. No rule moved. None of this
  changes a legacy project's bytes, nor a simple canonical candidate's.
- *The effective planner (owner D3, ADR-0015 M2.7-3; canonical projects only since vNext G0, owner
  section 8).* One meaning of a blank planner in a canonical project:
  `model.effective_planner(motion, planner, global_mode)` is the planner TMR's `MoveWaypoint` runs
  -- a chosen planner unchanged; a blank `ptp` -> `pilz`; a blank `free` move, or a waypoint with
  no segment -> the project's global planner mode (the token every generated launch hands the
  runner as `planner_mode`); `lin`/`circ` -> none, TMR reads none. Generation writes it into a
  canonical project's `bt_params.yaml` (`applications/base.py`), so no canonical free or ptp move
  relies on TMR's blank-planner rules; D-016 (`applications/vision_guided_motion.py`: a blank `ptp`
  vision move under a global mode other than `pilz` is flagged now) and the blocks page's switch to
  free/OMPL (`gui/wizard.py`) judge the same planner there. A legacy project keeps the reading it
  had before M2.7 at all three places: nothing is written for a blank planner, and D-016 and the
  switch read it as the global mode (KNOWN_DEBT: TMR runs a blank `ptp` as Pilz PTP all the same).
  TMR is unchanged. A canonical project with blank planners gains the explicit `planner:` lines;
  the identity corpora, which have none, generate byte-identical.
- *The canonical bin-picking motion gate (owner Q8.2, ADR-0015 M2.7-4).* Inside the canonical
  cycle (`Repeat` -> `Fallback`) every `MoveWaypoint` remaps TMR 2.1.0's optional `fault` output
  into the end-effector gate key (`applications/bin_picking.py`): a failure of the motion
  infrastructure (class B: MoveIt's servers, a controller or the execution failed or timed out)
  ends the run like an end-effector FAULT, while a target's own failure (class A: no IK, no plan,
  unreachable) writes no `fault` and keeps the next-candidate strategy. The prologue, `ServoTrack`
  (it has no such output; a validation note says so when the cycle streams a move) and legacy
  trees are untouched; the pinned project's tree is a byte copy of the one TMR's tests tick
  (`test/data/m2_7_mf/bin_picking_canonical.xml`). `TMR_FLOORS['move_waypoint_fault']` (2.1.0, a
  contract row) dates the port, so a canonical bin-picking app declares
  `trainit_motion_runtime >= 2.1.0`. vNext G0 adds `TMR_FLOORS['moveit_clients_namespace']`
  (2.1.0, `c2c9f01`, canonical-only): a canonical app whose `trainit_run_bt` runs under a
  non-empty `namespace=` declares it too, since its MoveIt clients resolve in the namespace only
  from TMR 2.1.0 (ADR-0023). Its config package and every legacy floor stay, a legacy namespaced
  bundle included (KNOWN_DEBT; owner M2.7 §4 and §17). The catalog snapshot records TMR
  2.1.0 -- provenance: a snapshot version is not a floor (ADR-0021).

**Done (8.0.0 / vNext M3.7, ADR-0024):** task composition in a task program -- `task` calls,
`inputs` / `outputs` by name, `repeat` / `retry` / `fallback`, the acyclic call graph, `config/tasks.yaml`
(§4g.3); decision 7 (four application names refused at load); the six M3_DESIGN §19 decompositions and
the pick_and_place / bin_picking semantic comparisons as tests. No TMR change; a call-free program and
every template: unchanged but `tasks.yaml` (task programs) and the manifest's provenance.

**Done (7.12.0 / vNext M3.6, ADR-0026):** manipulation in a task program -- the `detect`, `grasp`,
`release` and `tool` steps, `target_local`, their refusals (R-M3-13), the fixed idioms, the TMR 2.5.0
floors (§4g.2). M3-D offline: Validation C's generated trees run in TMR's fake cell with two synthetic
producers to the designed PreGrasp / Approach / Retract goals; a program without manipulation steps
and every template: unchanged.

**Done (7.11.0 / vNext M3.5, ADR-0024):** process-path semantics in a task program -- the `process`
step, path `events`, `capture`, their refusals (R-M3-09, R-M3-16), the split into `FollowPath from=
to=` runs, the TMR 2.4.0 floors (§4g.1). M3-C on mock: Validation B (three fresh captures,
`BT finished: SUCCESS`); a program without events and every template: unchanged.

**Done (7.10.0 / vNext M3.3, ADR-0024):** the Canonical Task Program (`task_program`, schema 4):
the model, its validation and M3.2 resolution, the generated `task_parameters` / `paths.yaml` /
`motion_profiles.yaml` / per-task trees, the TMR 2.2.0 floors, the YAML-first wizard guards (§4g).
M3-A: Validation A runs on mock from a generated bundle (`BT finished: SUCCESS`). Template bundles
byte-identical but the manifest's provenance.

**Done (7.9.0 / vNext M3.2, ADR-0025):** constraint and provider capability resolution -- TSA
answers "what does this request require, and can its provider do it?" before any file. No
generated byte moves (only the manifest's `capability_snapshot` provenance of the refresh).
- *Registry schema 2.* The catalog reads TMR's planner registry at schemas 1 and 2 and refuses
  any other number by name. Schema 2's `capabilities` (live / reserved tokens), `providers` (one
  row per MoveIt route: `route`, at most one pointer into `planner_modes` or `cartesian`,
  `provides`, `stability`) and `provider_defaults` are checked structurally by every reader
  (`catalog.provider_problems`); `eligible_providers` = stable and deliverable rows (ADR-0021
  Decision 5 extended to rows; a row with no pointer runs no pipeline). The refresh refuses a
  registry that drops a token or a kind TSA's resolution names (`TSA_CAPABILITY_TOKENS`,
  `TSA_MOTION_KINDS`), and so does the model at import. The committed copy is TMR 2.2.0's.
- *The request models* (`model/motion.py`, strict, not yet in any project file -- M3.3's
  `task_program` uses them): `MotionConstraints` (speed / acceleration percent, TCP velocity /
  acceleration, a segment's blend radius, `goal_tolerance`, planning time / attempts; the
  reserved `jerk`, `orientation_constraint`, `duration_s` held so they are refused by name),
  `MotionProfile` (no segment blend), `ProviderChoices` (`ptp`, `free`, `path`, `dense_path`),
  `ProviderId` (the registry's rows) and `MotionIntent` (`ptp | free | lin | circ | path`). The
  legacy `PlannerId` (planner modes) is untouched: a provider id is another vocabulary.
- *The resolution* (`model/capabilities.py`): ONE derivation (`required_capabilities`: an
  intent/shape table and a constraint-field table; a path's derivation is TMR's own), ONE
  compatibility rule (`missing_capabilities`: required <= provides), and `resolve` -- the
  request's provider, else the application's choice, else the registry default, never a search
  or a fallback; a refusal (`CapabilityRefusal`, a ValueError) names the request, intent,
  required and provided sets, the missing tokens and where each requirement comes from
  (R-M3-04, 06, 07, 08, 10, 11, 12; R-M3-02 at load). The three documented equivalences come back
  with their derivation for M3.3 to write: speed % = scaling, absolute TCP limits = scaling of the
  bundle's `pilz_cartesian_limits` (`project_cartesian_limits`: the base moveit_config's file, or
  `robot.pilz_cartesian_limits`; above the limit refused, never clamped; with a percent too, the
  smaller of the two), and `continuity: stop` = every blend radius 0.

**Next:**
- **Touchable-ACM**: honour `touchable_collision_ids` in `scene_manager_node`
  (`applyAttachedAcm`) — the metadata is already emitted into `scene.yaml`.
- **Real generalization**: for cells whose real branch is more than one include, extend the
  bringup template to spawn a list of real bridges (today: `deployment.real_include`).

**Seams (present, not active):**
- **PLC actuation** — a bundle package driving robot + PLC components via BT with per-mode
  adapters (`Isaac_Adapter_PLC_Actuator`, `PLC_Beckhoff_Adapter`) behind the same contract
  as the gripper adapter.
- **Vision 3D / learned** — the 2D colour-mask path SHIPPED in v4 (TMR v1.3
  `DetectObject`/`SetWaypointFromDetection`, proven live 2026-08-31). PCL 3D, shape and
  learned methods are still roadmap as *implementations*, but they are no longer roadmap
  as *configuration*: the external-node **launch half** ships (§4e), so a bundle launches
  such a node beside the generated `detector_node`. What is missing is the GUI — the
  Perception step's method combo still offers only the colour mask, so an external-node
  detector is configured in `project.yaml` (which the wizard round-trips untouched).
- **Docker** packaging; **policies** (Isaac-Lab-trained weights) as selectable per-task
  motion strategies.

## 8. Testing

`QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
python3 -m pytest test/ -q` (the plugin-autoload flag avoids the ament lint plugins). The
fr3wml golden equivalence regenerates the 3.2 VISION bundle from its shipped
project.yaml (21/21 checks — the scene-loader path); `test_scene_loader_emitter.py`
covers the MVP copy-base flow; `test_bt_nodes.cpp` (TMR) covers node registration.

**The two gates that must never go red without a deliberate re-blessing** are
`test_generator_golden.py` and `test_backend_bytestability.py`. Every per-move or per-cell
option added since is only correctly GATED when both still pass with no regeneration:
`test_execution_mode.py` pins ADR-0015's half (what `servo` emits, and that `plan` emits
nothing), `test_cell_namespace.py` pins ADR-0016's (empty emits no namespace token
anywhere; set, it reaches every node that should carry it and none that should not — read
with `ast`, per node, because a substring search cannot tell which node a `namespace=`
belongs to), and `test_bin_picking.py` pins the loop's shape, its retry count and its
gated recovery. `test_candidate_rule.py` pins ADR-0017's: that an unset rule writes no key
at all (the fact the two gates depend on), that a rule whose terms are not computable is
refused before a single file is written, that the three numbers come from their three
homes, and that a ranked bundle pins the runtime that reads the rule.
