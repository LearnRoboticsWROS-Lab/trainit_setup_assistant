"""The capability snapshot identity a bundle must record, computed by the TESTS (vNext M2.3).

Independent of ``catalog.snapshot_identity``: the sha256 of the three committed catalog files of
THIS source tree, read by path (not through the imported package), and the versions and schemas
``snapshot.yaml`` records. A test that compares a manifest with this pins the meaning of the
identity -- the hash of the committed bytes -- not just whatever the product function returns.
Not collected by pytest (the name is not test_*.py).
"""
import hashlib
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
CATALOG = PKG / 'trainit_setup_assistant' / 'catalog'
FILES = ('snapshot.yaml', 'tmr_planner_registry.yaml', 'tp_registry_snapshot.yaml')


def sha256(name: str, catalog: Path = CATALOG) -> str:
    return hashlib.sha256((catalog / name).read_bytes()).hexdigest()


def committed_identity(catalog: Path = CATALOG) -> dict:
    snap = yaml.safe_load((catalog / 'snapshot.yaml').read_bytes())
    registries = snap['registries']
    return {
        'catalog_version': snap['catalog_version'],
        'catalog_sha256': sha256('snapshot.yaml', catalog),
        'trainit_motion_runtime': {
            'version': snap['derived_with']['trainit_motion_runtime'],
            'planner_registry_schema':
                registries['trainit_motion_runtime']['planner_registry']['schema_version'],
            'planner_registry_sha256': sha256('tmr_planner_registry.yaml', catalog)},
        'trainit_perception': {
            'version': registries['trainit_perception']['version'],
            'vocabulary_schema': registries['trainit_perception']['vocabulary']['schema_version'],
            'vocabulary_sha256': sha256('tp_registry_snapshot.yaml', catalog)},
    }
