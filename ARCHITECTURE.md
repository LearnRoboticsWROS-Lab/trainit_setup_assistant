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
| `<robot>_trainit_config` (**bundle**) | TSA Step 9 | The same standalone copy, shipped in the bundle. **Independent** of the other two configs. |

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
- `emitters/app_pkg.py` + `applications/pick_and_place.py` → BT XML + `bt_params.yaml` +
  launch + Groot2 + README. Every DOF is in `bt_params.yaml`; the tree is uniform.
- All writes go through `GenContext` → recorded in `generation_manifest.yaml`
  (COPY/TEMPLATE/GENERATE + sha256). Determinism: same project ⇒ byte-identical output.

## 4. Dynamic-object management (the Step-8 crux)

Model (`model/scene.py`, `model/task.py`):
- `SceneObject`: `category` (static/actuated/dynamic) drives collision role; per-dynamic
  `grasp_target` (attaches on close), `release_policy` (freeze|gravity — never float),
  `isaac_grasp_method` (metadata), `touchable_collision_ids`.
- `MotionSegment.attached_collision_check` (per-move; None = inherit).

Runtime (TMR `bt_nodes.cpp`, added for this): the tree drives the flags automatically —
- **`SetAttachedCollisionCheck`** calls `scene_manager_node`'s `~/attached_collision_check`
  SetBool service and **blocks on the response** so the ACM is applied before the next
  plan. Emitted before a transfer that must route the held payload around static meshes.
- **`SetReleasePolicy`** publishes a latched Bool on `/isaac_release_policy` (freeze=true),
  emitted before `OpenGripper`. A no-op in mock/real.

Both are gated on the scene-loader flow (a base config being set), so the from-scratch
golden path stays byte-stable.

## 5. The Isaac grasp adapter (sim-specific, NOT generated)

Isaac-internal physics can't be synthesized generically, so the grasp adapter is a
**hand-authored Script Node in the `<cell>_isaac` package**. The TSA owns only the ROS
**contract** (`/isaac_gripper_cmd` Bool = close/open; `/isaac_release_policy` Bool =
freeze/gravity) + the config, and ships a **reference template**:
`resources/isaac/isaac_manipulation_adapter.py` — a parametric fixed-joint grasp (the v14
method that stabilised the 20 bottles) that reads `db.inputs.cmd` + `db.inputs.freeze`.
The user copies it into their `*_isaac`, edits the CONFIG block, wires two OmniGraph ROS2
Subscribers, and tests it in Step 1. The BIG1500 reference is
`big1500_isaac/scripts/isaac_dynamic_attach_script.py`.

## 6. TMR BT vocabulary (what an application may use)

25+ leaf nodes; the assistant emits: `MoveWaypoint` (data-driven), `Close/OpenGripper`,
`AttachObject/DetachObject`, and now `SetAttachedCollisionCheck`/`SetReleasePolicy`, plus
`AddCollisionObject` (from-scratch path only). Trees are **linear** (no control-flow /
sensor nodes yet). The TMR can run pick-and-place / tending / palletizing / kitting today,
and cartesian process paths (gluing/follow-path) with a bit more wiring. No CNC-as-machine,
no vision, no force control, no online servo yet.

## 7. Status & roadmap

**Done (committed):**
- Phase 0 — TSA + TMR under git (one repo each), versions bound.
- Phase 1 — model extensions (mesh objects, per-dynamic attributes, per-move flag, base ref).
- Phase 2 — TMR BT nodes `SetAttachedCollisionCheck` / `SetReleasePolicy` (+ gtest).
- Phase 3 — `SceneLoaderMoveitConfigEmitter` + `scene.yaml` + app wiring + bundle README.
- Phase 5 — this doc, the public README, the generalized Isaac adapter.

**Next:**
- **Phase 4 — the wizard** (`gui/`): the two-phase, 9-step flow; Step-1 USD preview +
  missing-asset handling; Step-2 base-config extraction; Step-3 generate + build snippet;
  Step-8 dynamic-object management UI. This is the largest remaining piece.
- **Touchable-ACM**: honour `touchable_collision_ids` in `scene_manager_node`
  (`applyAttachedAcm`) — the metadata is already emitted into `scene.yaml`.
- **Real generalization**: for cells whose real branch is more than one include, extend the
  bringup template to spawn a list of real bridges (today: `deployment.real_include`).

**Seams (present, not active):**
- **PLC actuation** — a bundle package driving robot + PLC components via BT with per-mode
  adapters (`Isaac_Adapter_PLC_Actuator`, `PLC_Beckhoff_Adapter`) behind the same contract
  as the gripper adapter.
- **Vision** — 2D (OpenCV) / 3D (PCL) perception feeding grasp poses; needs Condition/
  perception BT nodes in the TMR (not present).
- **Docker** packaging; **policies** (Isaac-Lab-trained weights) as selectable per-task
  motion strategies.

## 8. Testing

`QT_QPA_PLATFORM=offscreen PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=$PWD:$PYTHONPATH \
python3 -m pytest test/ -q` (the plugin-autoload flag avoids the ament lint plugins). The
fr3wml golden equivalence is the from-scratch-path regression guard; `test_scene_loader_emitter.py`
covers the MVP copy-base flow; `test_bt_nodes.cpp` (TMR) covers node registration.
