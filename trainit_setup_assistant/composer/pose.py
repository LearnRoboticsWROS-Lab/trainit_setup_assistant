"""Waypoint values as an engineer reads them (vNext I0.3, I0_DESIGN.md §14). Pure math, no Qt.

- **Position:** stored in metres, shown in millimetres.
- **Orientation:** stored as the canonical unit quaternion ``[qx, qy, qz, qw]``, shown as roll / pitch /
  yaw in degrees (ROS RPY: fixed X, then Y, then Z).

**The stored quaternion is kept EXACTLY** unless the user changes the orientation:
:func:`edited_orientation` returns the original when the shown angles were not edited. An untouched
waypoint therefore round-trips bit for bit, and raw quaternion editing is never the normal way.

Joints are stored in radians and shown in degrees, under the same rule.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

MM_DIGITS = 2                 # 0.01 mm shown
DEG_DIGITS = 2                # 0.01 deg shown
POSITION_DIGITS = 6           # a captured or typed position: 1 µm
QUATERNION_DIGITS = 9         # a computed quaternion: unit within 1e-9 (the model accepts 1e-3)
JOINT_DIGITS = 6              # a captured or typed joint: 1 µrad


def quat_to_rpy(q: Sequence[float]) -> Tuple[float, float, float]:
    """``[qx, qy, qz, qw]`` -> (roll, pitch, yaw) in radians (fixed-axis X-Y-Z, as tf2 / URDF)."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def rpy_to_quat(roll: float, pitch: float, yaw: float) -> List[float]:
    """(roll, pitch, yaw) in radians -> a unit ``[qx, qy, qz, qw]`` with ``qw >= 0``."""
    cr, sr = math.cos(roll / 2.0), math.sin(roll / 2.0)
    cp, sp = math.cos(pitch / 2.0), math.sin(pitch / 2.0)
    cy, sy = math.cos(yaw / 2.0), math.sin(yaw / 2.0)
    q = [sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy,
         cr * cp * cy + sr * sp * sy]
    if q[3] < 0.0:
        q = [-v for v in q]
    return q


def rpy_deg(q: Sequence[float]) -> List[float]:
    """The orientation as shown: degrees, rounded to :data:`DEG_DIGITS`."""
    return [round(math.degrees(a), DEG_DIGITS) + 0.0 for a in quat_to_rpy(q)]


def mm(position: Sequence[float]) -> List[float]:
    return [round(float(v) * 1000.0, MM_DIGITS) + 0.0 for v in position]


def metres(values_mm: Sequence[float]) -> List[float]:
    return [round(float(v) / 1000.0, POSITION_DIGITS) + 0.0 for v in values_mm]


def clean_quaternion(q: Sequence[float]) -> List[float]:
    """A computed or captured quaternion as written: normalized, ``qw >= 0``, rounded."""
    x, y, z, w = (float(v) for v in q)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0.0:
        raise ValueError('a zero quaternion is not an orientation')
    out = [x / n, y / n, z / n, w / n]
    if out[3] < 0.0:
        out = [-v for v in out]
    return [round(v, QUATERNION_DIGITS) + 0.0 for v in out]


def edited_orientation(stored: Optional[Sequence[float]], shown_deg: Sequence[float],
                       entered_deg: Sequence[float]) -> List[float]:
    """The quaternion to store after the orientation fields were (maybe) edited: the STORED one when
    the entered angles equal the shown ones, else the entered angles' quaternion."""
    if stored is not None and all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(shown_deg, entered_deg)):
        return list(stored)
    return clean_quaternion(rpy_to_quat(*(math.radians(float(a)) for a in entered_deg)))


def edited_position(stored: Optional[Sequence[float]], shown_mm: Sequence[float],
                    entered_mm: Sequence[float]) -> List[float]:
    """The same rule for the position (stored metres, shown millimetres)."""
    if stored is not None and all(abs(float(a) - float(b)) < 1e-9 for a, b in zip(shown_mm, entered_mm)):
        return list(stored)
    return metres(entered_mm)


def joints_deg(joints: Sequence[float]) -> List[float]:
    return [round(math.degrees(float(v)), DEG_DIGITS) + 0.0 for v in joints]


def edited_joints(stored: Optional[Sequence[float]], shown_deg: Sequence[float],
                  entered_deg: Sequence[float]) -> List[float]:
    if stored is not None and len(stored) == len(entered_deg) and all(
            abs(float(a) - float(b)) < 1e-9 for a, b in zip(shown_deg, entered_deg)):
        return list(stored)
    return [round(math.radians(float(v)), JOINT_DIGITS) + 0.0 for v in entered_deg]


# ---- canonical waypoint mappings (model/program.py TaskWaypoint) -------------------------------------

def tcp_waypoint(name: str, position: Sequence[float], orientation: Sequence[float],
                 frame: Optional[str] = None) -> dict:
    wp = {'name': name, 'position': [float(v) for v in position], 'orientation': [float(v) for v in orientation]}
    if frame:
        wp['frame'] = frame
    return wp


def captured_tcp_waypoint(name: str, position: Sequence[float], orientation: Sequence[float],
                          frame: Optional[str] = None) -> dict:
    """A waypoint from the robot's CURRENT TCP pose (TF base -> tip): 1 µm, a clean unit quaternion."""
    return tcp_waypoint(name, [round(float(v), POSITION_DIGITS) + 0.0 for v in position],
                        clean_quaternion(orientation), frame)


def joint_waypoint(name: str, joints: Sequence[float]) -> dict:
    return {'name': name, 'type': 'joint', 'joints': [float(v) for v in joints]}


def captured_joint_waypoint(name: str, joints: Sequence[float]) -> dict:
    """A waypoint from the robot's CURRENT joint state, in planning-group order: 1 µrad."""
    return joint_waypoint(name, [round(float(v), JOINT_DIGITS) + 0.0 for v in joints])


def named_waypoint(name: str, state: str) -> dict:
    return {'name': name, 'type': 'joint', 'named': state}


def relative_waypoint(name: str, to: str, frame: str, dx_mm: float = 0.0, dy_mm: float = 0.0,
                      dz_mm: float = 0.0) -> dict:
    rel = {'to': to, 'frame': frame}
    for key, v in (('dx', dx_mm), ('dy', dy_mm), ('dz', dz_mm)):
        if float(v) != 0.0:
            rel[key] = round(float(v) / 1000.0, POSITION_DIGITS) + 0.0
    return {'name': name, 'relative': rel}


def kind(wp: dict) -> str:
    """TCP | Joints | Named | Relative."""
    if wp.get('relative'):
        return 'Relative'
    if wp.get('type') == 'joint':
        return 'Named' if wp.get('named') else 'Joints'
    return 'TCP'


def summary(wp: dict) -> str:
    """One line, in an engineer's units."""
    k = kind(wp)
    if k == 'TCP':
        p, r = mm(wp.get('position') or [0, 0, 0]), rpy_deg(wp.get('orientation') or [0, 0, 0, 1])
        frame = f" in {wp['frame']}" if wp.get('frame') else ''
        return (f'XYZ ({p[0]:g}, {p[1]:g}, {p[2]:g}) mm · RPY ({r[0]:g}, {r[1]:g}, {r[2]:g})°{frame}')
    if k == 'Joints':
        return 'joints (' + ', '.join(f'{v:g}' for v in joints_deg(wp.get('joints') or [])) + ')°'
    if k == 'Named':
        return f"named state '{wp['named']}' (MoveIt config)"
    rel = wp['relative']
    offs = ', '.join(f'{a} {rel[a] * 1000.0:g} mm' for a in ('dx', 'dy', 'dz') if rel.get(a))
    return f"relative to '{rel['to']}' along {rel['frame']}" + (f': {offs}' if offs else ' (no offset)')


__all__ = ['captured_joint_waypoint', 'captured_tcp_waypoint', 'clean_quaternion', 'edited_joints',
           'edited_orientation', 'edited_position', 'joint_waypoint', 'joints_deg', 'kind', 'metres', 'mm',
           'named_waypoint', 'quat_to_rpy', 'relative_waypoint', 'rpy_deg', 'rpy_to_quat', 'summary',
           'tcp_waypoint']
