"""Verification: prove a generated bundle EQUALS a golden one (+ that it builds)."""

from .build_check import colcon_build_bundle
from .equivalence import Report, compare_bundle, detect_packages

__all__ = ['compare_bundle', 'detect_packages', 'Report', 'colcon_build_bundle']
