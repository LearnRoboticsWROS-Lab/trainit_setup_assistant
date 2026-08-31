"""Camera-variant URDF rewrite (TSA v4, D-015).

A camera cell needs the robot description to CARRY the camera frames (TF is what
``DetectObject`` transforms through), yet the base description is ingested verbatim.
The project's :class:`~..model.perception.CameraSpec` names the swap as DATA:
the ``xacro:include`` whose ``filename`` equals ``replace_include`` becomes
``with_include``. Applied with the copy-then-overwrite precedent (the SRDF
named-state merge): the same rel_path is COPY'd then GENERATE'd.

Applied in BOTH packages that copy the model (``_description`` and
``_trainit_config``) — a single-sided rewrite would make the two copies drift.
"""

from __future__ import annotations

from typing import Optional

_MARK = ('    <!-- Camera variant (generated): the camera frames ride the robot '
         'description,\n         so TF carries the optical frame DetectObject '
         'transforms from. -->\n')


def rewrite_camera_include(text: str, camera) -> Optional[str]:
    """Swap the include target; ``None`` when nothing applies (file untouched)."""
    if camera is None or not camera.replace_include or not camera.with_include:
        return None
    needle = f'filename="{camera.replace_include}"'
    if needle not in text:
        return None
    replacement = f'filename="{camera.with_include}"'
    out = text.replace(needle, replacement)
    # a short generated marker above the first rewritten include, for the reader
    line_start = out.index(replacement)
    line_start = out.rfind('\n', 0, line_start) + 1
    return out[:line_start] + _MARK + out[line_start:]
