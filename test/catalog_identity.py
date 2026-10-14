"""The capability snapshot identity a bundle must record, computed by the TESTS (vNext M2.3).

Independent of ``catalog.snapshot_identity``: the sha256 of the three committed catalog files of
THIS source tree, read by path (not through the imported package), and the versions and schemas
``snapshot.yaml`` records. A test that compares a manifest with this pins the meaning of the
identity -- the hash of the committed bytes -- not just whatever the product function returns.

What a checkout cannot derive (vNext G0 P1). ``snapshot.yaml`` records facts that derive() reads
from the package.xml of installed packages. Where one of those packages is not installed (the
private policy runtime; a package outside the Community dependency closure, such as MoveIt Servo),
derive() answers differently for exactly those facts. ``absent_packages()`` names such packages,
``without_packages`` masks exactly those values in a record, ``as_recorded`` takes them from
another record instead, and ``require_installed`` then says what was not compared. Unlike the
identity above, this half deliberately asks the product: derive()'s own reader, so that it and a
derived record can never disagree about what is installed.

Not collected by pytest (the name is not test_*.py).
"""
import copy
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


# --- what a checkout without a package cannot derive (vNext G0 P1) ------------------------------

# The derived_with entries derive() reads from an installed package.xml; the other two are TSA's
# own release and the release TP's vocabulary states (catalog/derive.py, derive()).
DERIVED_WITH_INSTALLED = ('trainit_motion_runtime', 'trainit_policy_runtime')


def _installed_values(record: dict):
    """Every value of a catalog record that derive() reads from an installed package.xml -- what
    the callers of ``derive._package_facts`` put there, nothing else -- as ``(package, where,
    mapping, key, without)``, ``without`` being what derive() answers when the package is not
    installed: its ``derived_with`` entry; the release and the declared licence of the rows it
    licenses (``_trainit_license``); the plugin facts, licence status and release of the rows
    it backs as a plugin (``_motion_planning``, ``_servo``)."""
    derived_with = record.get('derived_with') or {}
    for name in DERIVED_WITH_INSTALLED:
        if name in derived_with:
            yield name, ('derived_with', name), derived_with, name, None
    for row in record.get('providers') or []:
        source = row.get('license_source') or {}
        package = source.get('package')
        if not package:
            continue
        rid = row.get('id')
        yield package, (rid, 'version'), source, 'version', None
        if 'declared' in source:
            yield package, (rid, 'declared'), source, 'declared', None
        if 'plugin' in row:
            yield package, (rid, 'plugin'), row, 'plugin', dict(package=package, installed=False)
            yield package, (rid, 'license_status'), row, 'license_status', 'UNKNOWN'


def absent_packages(record: dict = None) -> list:
    """The packages ``record`` (the committed ``snapshot.yaml`` by default) records an INSTALLED
    fact of -- a release, a licence, a plugin -- that are not installed HERE, sorted. Asked of
    derive()'s own reader (the ament index, no fallback), so this list and a derived record can
    never disagree. Empty where every one of them is installed (the owner's workspace)."""
    from trainit_setup_assistant.catalog.derive import _package_facts
    if record is None:
        record = yaml.safe_load((CATALOG / 'snapshot.yaml').read_bytes())
    recorded = {package for package, _, mapping, key, without in _installed_values(record)
                if mapping.get(key) != without}
    return sorted(p for p in recorded if not _package_facts(p).get('installed'))


def without_packages(record: dict, absent: list) -> dict:
    """A deep copy of ``record`` with EXACTLY the values derive() reads from the package.xml of a
    package in ``absent`` set to what derive() answers without it; nothing else changes, and an
    empty ``absent`` changes nothing. Applied to both records, it compares everything else."""
    out = copy.deepcopy(record)
    for package, _, mapping, key, without in list(_installed_values(out)):
        if package in absent:
            mapping[key] = copy.deepcopy(without)
    return out


def as_recorded(record: dict, recorded: dict, absent: list) -> dict:
    """A deep copy of ``record`` with each value ``without_packages`` would mask taken from
    ``recorded`` instead (the same row, the same field): what derive() answers where the
    ``absent`` packages are installed as ``recorded`` says. Nothing else changes."""
    want = {where: mapping.get(key) for package, where, mapping, key, _
            in _installed_values(recorded) if package in absent}
    out = copy.deepcopy(record)
    for package, where, mapping, key, _ in list(_installed_values(out)):
        if package in absent and where in want:
            mapping[key] = copy.deepcopy(want[where])
    return out


def require_installed(absent: list) -> None:
    """Stop the running test when ``absent`` (``absent_packages()``) names a package -- called
    LAST, once everything else was compared. It SKIPS naming them and what was not compared; with
    TRAINIT_REQUIRE_GOLDEN=1 it FAILS instead (``eef_fixture.skip_or_fail``). A no-op when none is
    absent."""
    if absent:
        import eef_fixture        # pytest and TSA's model: imported only by a test that stops
        one = len(absent) == 1
        eef_fixture.skip_or_fail(
            f'{", ".join(absent)} {"is" if one else "are"} not installed here (not on the ament '
            f'index): what snapshot.yaml records from {"its" if one else "their"} package.xml -- '
            'release, licence, plugin facts -- was not compared')
