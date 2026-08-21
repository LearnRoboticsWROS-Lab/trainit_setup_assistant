# -----------------------------------------------------------------------------
# Isaac-side SCENE RESET receiver — OmniGraph SCRIPT NODE code (NOT the Script Editor!)
#
# THE OTHER HALF OF THE RESET CONTRACT.
# trainit_motion_runtime's scene_manager_node restores the MoveIt PLANNING scene when
# its ~/reset_scene service is called, and latches a RISING EDGE on
# `scene_reset_topic` (default /isaac_scene_reset). Putting the SIMULATED prims back is
# the cell author's half — that is this file. Without it the planning scene resets and
# the Isaac world does not, so cycle 2 of a looping application picks at thin air.
#
# WHY NOT rclpy: Isaac Sim 5.1 runs Python 3.11, ROS 2 Humble's rclpy is built for 3.10,
# so `import rclpy` fails inside Isaac. Talk to ROS 2 through the ros2_bridge OmniGraph
# nodes. This code goes into an OmniGraph *Script Node* fed by a *ROS2 Subscriber*
# (std_msgs/Bool, topic `isaac_scene_reset`).
#
# HOW TO WIRE IT (mirror the suction pair exactly):
#   1. Action Graph -> add ROS2 Subscriber:  messagePackage std_msgs, messageName Bool,
#      topicName `isaac_scene_reset`, context <- ros2_context, execIn <- on_playback_tick
#   2. add a Script Node, give it a BOOL input attribute named 'reset', wire the
#      subscriber's 'data' output into it, paste this file into its script, File > Save.
#
# TRIGGER ON THE EDGE, NEVER ON THE LEVEL. The topic is latched: a handler that acts on
# `reset == True` runs every tick and pins the prims home forever; one that dedupes
# against a permanently-true level fires once per session. scene_manager_node therefore
# emits true -> false, and this node acts on the FALSE->TRUE transition only.
# -----------------------------------------------------------------------------

# --- CONFIG ------------------------------------------------------------------
# Dynamic prims to send home. Keep in step with the `dynamic: true` ids in the
# generated config/scene.yaml.
RESET_PRIMS = ["/World/Cube"]
# Released before teleporting, so a reset that lands mid-grasp does not fight the
# SurfaceGripper attachment. Set to None to skip.
GRIPPER_PATH = "/World/fr3wml_suction/suction_body/tcp/SurfaceGripper"
# Optional hard override, {prim_path: ([x,y,z], [w,x,y,z])}. Use it when the pose saved in
# the stage is not the home you want; otherwise leave empty and the saved pose wins.
HOME_POSES = {}
# Two resets closer than this are treated as one. A pick-and-place cycle boundary is
# seconds apart, so 1 s swallows duplicates without ever dropping a real reset.
DEBOUNCE_S = 1.0
# -----------------------------------------------------------------------------


def _home_pose(stage, path):
    """The AUTHORED home pose, re-read FROM DISK.

    Not the live prim pose: by the time a reset arrives the object has already been picked
    and placed, so a runtime capture would enshrine wherever physics left it.

    And not the in-memory root layer either — that was the first version of this and it is
    WRONG: dragging a prim in the viewport authors straight into the session's root layer,
    so an unsaved manual move silently becomes "home". Re-opening the layer file gives the
    pose the cell author actually saved, immune both to PhysX and to live edits.

    HOME_POSES overrides everything, for a cell whose saved pose is not the wanted home.
    """
    from pxr import Gf, UsdGeom, Usd, Sdf
    if path in HOME_POSES:
        pos, quat = HOME_POSES[path]
        return tuple(pos), tuple(quat)
    layer = stage.GetRootLayer()
    try:
        disk = Sdf.Layer.OpenAsAnonymous(layer.identifier)    # fresh read, ignores edits
        if disk:
            layer = disk
    except Exception as e:                                   # noqa: BLE001
        print(f"[reset] could not re-read {layer.identifier}: {e} — using the live layer")
    t = layer.GetAttributeAtPath(f"{path}.xformOp:translate")
    o = layer.GetAttributeAtPath(f"{path}.xformOp:orient")
    pos = tuple(t.default) if t and t.default is not None else None
    quat = None
    if o and o.default is not None:
        q = o.default                       # Gf.Quatd -> (w, x, y, z) for Isaac core
        quat = (q.GetReal(), *q.GetImaginary())
    if pos is None:                         # fallback: composed pose, may be a moved one
        prim = stage.GetPrimAtPath(path)
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        tr = m.ExtractTranslation()
        qq = m.ExtractRotationQuat()
        pos = (tr[0], tr[1], tr[2])
        quat = (qq.GetReal(), *qq.GetImaginary())
        print(f"[reset] WARNING {path}: no authored xformOp:translate, using the live pose")
    return pos, (quat or (1.0, 0.0, 0.0, 0.0))


def _teleport(path, pos, quat_wxyz):
    """Move a dynamic rigid body and STOP it.

    Uses the physics-aware API: writing xformOps while the sim is playing does not
    reliably move a rigid body (PhysX owns the transform) and can leave the rendered mesh
    behind. Velocities must be zeroed too, or the cube teleports home and immediately
    resumes falling with its pre-reset velocity.
    """
    import numpy as np
    try:
        from isaacsim.core.prims import SingleRigidPrim as _RigidPrim      # Isaac 5.x
    except ImportError:
        from omni.isaac.core.prims import RigidPrim as _RigidPrim          # Isaac 4.x
    rp = _RigidPrim(path)
    rp.set_world_pose(position=np.array(pos), orientation=np.array(quat_wxyz))
    try:
        rp.set_linear_velocity(np.zeros(3))
        rp.set_angular_velocity(np.zeros(3))
    except Exception as e:                                   # noqa: BLE001
        print(f"[reset] {path}: could not zero velocity: {e}")


def _open_gripper():
    if not GRIPPER_PATH:
        return
    try:
        from isaacsim.robot.surface_gripper import GripperView
        GripperView(paths=GRIPPER_PATH).apply_gripper_action([-1.0])   # -1 = open
    except Exception as e:                                   # noqa: BLE001
        print(f"[reset] gripper open skipped: {e}")


def setup(db):
    db.per_instance_state.last_t = 0.0
    db.per_instance_state.home = None


# Set False once the cell is proven, to stop logging every edge.
VERBOSE = True


def compute(db):
    # EVERYTHING is wrapped: an unhandled exception in an OmniGraph Script Node is easy
    # to miss in the Console, and a reset that fails silently looks exactly like a reset
    # that never arrived. Be loud instead.
    try:
        try:
            cmd = bool(db.inputs.reset)
        except Exception as e:                               # noqa: BLE001
            print(f"[reset] cannot read inputs.reset: {e} "
                  f"(is the bool input named exactly 'reset' and wired to the "
                  f"subscriber's 'data'?)")
            return True

        st = db.per_instance_state
        # TIME debounce, NOT value dedupe. A `cmd == last` guard desynchronises: any
        # `true` that is never followed by a `false` (a hand-published test message, a
        # re-delivered latched value, a graph reload mid-pulse) leaves `last` stuck at
        # True and every later reset is swallowed — silently. Debouncing on time has no
        # state to lose: act on any `true`, ignore repeats inside one pulse.
        import time
        now = time.monotonic()
        last_t = getattr(st, 'last_t', 0.0)
        if VERBOSE:
            print(f"[reset] compute: reset={cmd} dt={now - last_t:.2f}s")
        if not cmd:                       # trailing edge / idle level
            return True
        if now - last_t < DEBOUNCE_S:     # same pulse, or a re-delivered latched message
            return True
        st.last_t = now

        import omni.usd
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            print("[reset] no stage")
            return True
        if getattr(st, 'home', None) is None:      # cache the authored poses on first use
            st.home = {p: _home_pose(stage, p) for p in RESET_PRIMS
                       if stage.GetPrimAtPath(p).IsValid()}
            missing = [p for p in RESET_PRIMS if p not in st.home]
            if missing:
                print(f"[reset] WARNING prims not found: {missing}")
            print(f"[reset] home poses: {st.home}")

        _open_gripper()
        for path, (pos, quat) in st.home.items():
            try:
                _teleport(path, pos, quat)
                print(f"[reset] {path} -> {tuple(round(v, 4) for v in pos)}")
            except Exception as e:                           # noqa: BLE001
                import traceback
                print(f"[reset] {path}: teleport FAILED: {e}")
                traceback.print_exc()
        print(f"[reset] {len(st.home)} prim(s) -> home")
    except Exception:                                        # noqa: BLE001
        import traceback
        print("[reset] UNHANDLED:")
        traceback.print_exc()
    return True


def cleanup(db):
    db.per_instance_state.home = None
