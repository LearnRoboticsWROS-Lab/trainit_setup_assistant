# -----------------------------------------------------------------------------
# Isaac-side suction receiver — OmniGraph SCRIPT NODE code (NOT the Script Editor!)
#
# WHY NOT rclpy: Isaac Sim 5.1 runs Python 3.11, ROS2 Humble's rclpy is built for
# 3.10 -> `import rclpy` fails inside Isaac. The supported way to talk to ROS2
# from Isaac is the ros2_bridge OmniGraph nodes. This code goes into an OmniGraph
# *Script Node* fed by a *ROS2 Subscriber* (std_msgs/Bool, topic <your gripper cmd topic>).
#
# Because it lives in the Action Graph, it is SAVED in the USD and runs
# automatically on Play — you do NOT re-run it each session (unlike the drives,
# which are also saved; runtime scripts are not).
#
# HOW TO USE: see the "Build the suction Action Graph" steps in the chat / notes.
# Add a BOOL input attribute named 'cmd' to the Script Node and wire the ROS2
# Subscriber's 'data' output into it.
# -----------------------------------------------------------------------------
GRIPPER_PATH = "/World/CHANGE_ME/.../SurfaceGripper"   # <-- your SurfaceGripper prim


def setup(db):
    db.per_instance_state.gv = None
    db.per_instance_state.last = None


def compute(db):
    cmd = bool(db.inputs.cmd)              # bool from the ROS2 Subscriber 'data'
    st = db.per_instance_state
    if cmd == st.last:                     # act only on change
        return True
    st.last = cmd
    try:
        from isaacsim.robot.surface_gripper import GripperView
        if st.gv is None:
            st.gv = GripperView(paths=GRIPPER_PATH)
        st.gv.apply_gripper_action([1.0 if cmd else -1.0])   # 1=close, -1=open
        print(f"[gripper] {'CLOSE' if cmd else 'OPEN'} -> {st.gv.get_surface_gripper_status()}")
    except Exception as e:
        print(f"[gripper] error: {e} (is the sim PLAYING?)")
    return True


def cleanup(db):
    db.per_instance_state.gv = None
