"""Application templates: map a configured project onto a Behavior Tree."""

from .base import (
    ApplicationTemplate,
    build_bt_params_context,
    payload_dims_key,
    scene_dims_key,
    scene_pos_key,
)
from .bin_picking import BinPicking
from .registry import available_applications, get_application

__all__ = [
    'ApplicationTemplate',
    'BinPicking',
    'build_bt_params_context',
    'scene_dims_key',
    'scene_pos_key',
    'payload_dims_key',
    'get_application',
    'available_applications',
]
