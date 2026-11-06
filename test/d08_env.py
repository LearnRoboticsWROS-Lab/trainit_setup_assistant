"""TrainIt D0.8 test environment rule: an acceptance-critical integration dependency is REQUIRED where it should
exist, so a mis-built environment fails loudly instead of skipping silently.

In a sourced ROS workspace (``AMENT_PREFIX_PATH`` set) -- the environment the D0.8 acceptance runs in -- the data
stack (``trainit_data``, ``trainit_data_ros``), ``rclpy``, ``trainit_perception`` and the ``trainit_data``
repository beside TSA (its reference provider) must all be there: a missing one FAILS, naming it. Outside ROS (a
plain-Python TSA check) such a test skips, naming why; ``TRAINIT_REQUIRE_D08=1`` makes it fail there too.
"""
import os

import pytest


def required() -> bool:
    return bool(os.environ.get('AMENT_PREFIX_PATH')) or os.environ.get('TRAINIT_REQUIRE_D08') == '1'


def require(reason: str) -> None:
    """Fail (a D0.8 integration environment) or skip (plain Python), naming the missing dependency."""
    if required():
        pytest.fail(f'D0.8 integration dependency missing: {reason} (a sourced workspace must provide it; check '
                    'PYTHONPATH keeps the ROS paths: PYTHONPATH=$PWD:$PWD/test:$PYTHONPATH)', pytrace=False)
    pytest.skip(reason)


def import_or_require(module: str):
    try:
        return __import__(module, fromlist=['_'])
    except Exception as exc:  # noqa: BLE001
        require(f'{module} is not importable ({type(exc).__name__}: {exc})')
