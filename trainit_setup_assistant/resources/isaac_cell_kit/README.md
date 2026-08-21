# TrainIt Isaac cell kit

Drop-in tooling that makes a `<cell>_isaac` package — and the USD stage inside it —
importable by the TrainIt Setup Assistant.

Install it into a package with:

```bash
ros2 run trainit_setup_assistant trainit_isaac_kit --into <path/to/<cell>_isaac>
```

That copies `scripts/` and prints the checklist below. Nothing is overwritten unless you
pass `--force`.

---

## What the TSA needs from a cell, and which file covers it

| Requirement | Covered by |
|---|---|
| The stage is laid out so Step 2 can read it | `scripts/check_usd_for_tsa.py` (validator) |
| Every included object has a collision STL, in metres, in the prim's local frame | `scripts/usd_to_collision_stl.py` (exporter) |
| The gripper-close signal reaches `scene_manager_node` | `scripts/isaac_gripper_subscriber.py` (Script Node) |
| `ResetScene` puts the SIMULATED prims back, not just the planning scene | `scripts/isaac_scene_reset_subscriber.py` (Script Node) |
| `meshes/` reaches the installed share | `CMakeLists.snippet` |

The two Script Nodes are **templates**: copy their text into an OmniGraph Script Node in
your stage and edit the `CONFIG` block at the top (search for `CHANGE_ME`). They cannot be
generated, because Isaac-internal physics is not expressible in the TSA project model —
the TSA owns the ROS **contract**, the cell author owns the implementation.

---

## The ROS contract

Three topics, all `std_msgs/Bool`, all latched (`transient_local`):

| Topic | Direction | Meaning |
|---|---|---|
| gripper cmd (`scene.gripper_cmd_topic`) | cell → `scene_manager_node` | `true` = closed. Attaches the grasp targets in the planning scene |
| scene reset (`scene.scene_reset_topic`) | `scene_manager_node` → cell | **rising edge** on `~/reset_scene`. Teleport the dynamic prims home |
| `/isaac_release_policy` | `scene_manager_node` → cell | `true` = freeze on release, `false` = gravity |

> **Act on the TRANSITION, never on the level.** The reset topic is latched: a handler that
> acts while `reset == True` runs every tick and pins the prims home forever. The runtime
> emits `true` then `false` ~300 ms later precisely so you can trigger on the edge.

The first two are configurable at wizard Step 2, and the gripper one is **auto-detected**
from the stage's ActionGraph (the TSA reads every `std_msgs/Bool` `ROS2Subscriber`), so
naming your subscriber correctly is usually all it takes.

---

## Procedure for a new cell

### 1 — package skeleton

```
<cell>_isaac/
├── usd/scenes/<cell>.usd        the stage you author in Isaac
├── usd/props/*.usd              assets pulled out of the cloud, referenced RELATIVELY
├── meshes/<group>/<group>_collision.stl
├── scripts/                     this kit
└── CMakeLists.txt               see CMakeLists.snippet
```

### 2 — author the stage (Isaac)

- create `/World`; put the robot under it as `/World/<robot_name>`, where `<robot_name>`
  is **exactly** `<robot name="…">` from your base moveit_config's SRDF — the TSA looks for
  `/World/<robot_name>/<base_frame>` and there is no override field;
- every cell object is a **direct child of `/World`** (the importer reads one level, never
  recurses) with `xformOp:scale = (1,1,1)` — size belongs in the geometry;
- name the pick target with `part`, `workpiece`, `bottle`, `crate` or `box_part` to get
  `dynamic` + `grasp` classified automatically;
- pull any `https://` / `omniverse://` asset into `usd/props/` and re-point the reference
  **relatively**: plain `usd-core` resolves neither of those schemes, nor any `file:` form.

### 3 — collision meshes

```bash
python3 scripts/usd_to_collision_stl.py --stage usd/scenes/<cell>.usd --list
python3 scripts/usd_to_collision_stl.py --stage usd/scenes/<cell>.usd \
        --prim /World/<object> --out meshes/<group>/<group>_collision.stl
```

Export **from the cell stage**, never from a downloaded asset: cells routinely override an
asset's collision box with `over` specs, and exporting the bare asset silently loses them.

### 4 — the two Script Nodes (Isaac)

For each, add a `ROS2Subscriber` (`std_msgs` / `Bool`, your topic) + a Script Node with a
bool input, wire `data → input`, paste the template, edit `CHANGE_ME`, save the stage.

| Script Node | input attr | subscriber topic |
|---|---|---|
| `isaac_gripper_subscriber.py` | `cmd` | your gripper cmd topic |
| `isaac_scene_reset_subscriber.py` | `reset` | `isaac_scene_reset` |

### 5 — build and validate

```bash
colcon build --packages-select <cell>_isaac && source install/setup.bash

python3 scripts/check_usd_for_tsa.py --stage usd/scenes/<cell>.usd \
        --robot-name <robot_name> --base-frame <base_frame> --mesh-pkg <cell>_isaac
echo $?          # 0 = the wizard will import this cell cleanly
```

### 6 — smoke-test the contract, before the wizard

With the sim in Play and no application running:

```bash
ros2 topic pub --once <gripper cmd topic> std_msgs/msg/Bool '{data: true}'   # gripper closes
ros2 topic pub --once /isaac_scene_reset  std_msgs/msg/Bool '{data: true}'   # prims go home
```

Publish the reset **three times**. If it only works once, a value-dedupe guard survived
somewhere and the loop will break at cycle 2.

---

## One thing the kit cannot do for you

The TSA also needs a **base `moveit_config`** with a `launch/bringup.launch.py` carrying the
mock/isaac/real switch — that is where it reads the cell bridges and the arm's
`/joint_states` remap (parsed with `ast`, never executed). Keep it parseable:
`Node(package="literal", executable="literal")` inside `if mode in (...)` branches, literal
dicts in `parameters`, literal 2-tuples in `remappings`.

Also make sure the SRDF's planning group lists its joints — either explicitly, or as a
`<chain>` the TSA can walk. A group with neither yields empty captured named states.
