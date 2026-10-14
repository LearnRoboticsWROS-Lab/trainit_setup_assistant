"""What GENERATED bundles declare, and the launch harness imports, that this machine lacks: the
prerequisites of the opt-in launch tiers (vNext G0 community follow-up, the P1 contract applied to
``TRAINIT_LAUNCH_TEST=1``). Not collected by pytest (the name is not test_*.py); its own tests are
test_launch_prereqs.py.

A bundle's package.xml files declare what it runs on, and its first run installs that with
``rosdep install --from-paths <the bundle> --ignore-src -r -y`` (README.md, "Running a generated
bundle"). A launch tier generates its bundles into a temp dir, where rosdep never looks: on a
machine set up by README.md "Install" alone it used to build for minutes and then fail. Called
after generation and before the build, ``require`` stops it in seconds instead, naming

  * every package the bundles' package.xml files exec_depend on that is not on this machine's
    ament index -- the bundles' own packages, and every other package generated into the same
    temp workspace, excepted. A declared package counts even where a tier never uses it
    (joint_state_publisher_gui serves only view_robot.launch.py): a tier runs a bundle the way
    its user does after that first rosdep install, which installs everything declared;
  * the plugin package each generated ``moveit_controllers.yaml`` names as move_group's
    controller manager (``moveit_simple_controller_manager``), which no generated package.xml
    declares yet;
  * every Python module of the launch harness that this interpreter cannot import
    (``controller_manager_msgs``, for one: ``composite_launch.PHASE_A_MODULES``).

It SKIPS naming them and how to install them; with TRAINIT_REQUIRE_GOLDEN=1 it FAILS instead
(``eef_fixture.skip_or_fail``). Where everything is installed (the reference machine) it does
nothing. Humble has no package named mock_components to look for: the mock_components/GenericSystem
plugin ships in hardware_interface, a dependency of controller_manager.
"""
import importlib.util
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

# A ROS package name (REP 144). A dependency key that is not one (python3-..., lib...) is a system
# package that rosdep resolves and no ament index lists.
PACKAGE_NAME = re.compile(r'^[a-z][a-z0-9_]*$')
RUN_DEPENDS = ('depend', 'exec_depend')
# A Python module and the ROS package that installs it, where the two names differ.
MODULE_PACKAGE = {'tf2_ros': 'tf2_ros_py'}
# TrainIt Community: the repositories this manifest lists, one package each (README.md
# "Install"). Any other trainit_* package is Pro or experimental.
COMMUNITY_REPOS = Path(__file__).resolve().parents[1] / 'trainit_community.repos'


def _on_the_index() -> set:
    """Every package on this machine's ament index (none when no workspace is sourced)."""
    try:
        from ament_index_python.packages import get_packages_with_prefixes
        return set(get_packages_with_prefixes())
    except Exception:                       # noqa: BLE001 -- AMENT_PREFIX_PATH unset
        return set()


def _importable(module: str) -> bool:
    """Whether this interpreter can import the top-level ``module`` (found, not imported)."""
    return importlib.util.find_spec(module) is not None


def missing(*roots: Path, modules: tuple = ()) -> list:
    """What the bundles under ``roots`` declare, or the harness imports (``modules``), and this
    machine lacks, as ``(name, why)`` pairs: ``why`` is 'declared' (a package.xml exec_depends on
    it), 'named' (a moveit_controllers.yaml names its plugin, and no package.xml declares it) or
    'imported' (one of ``modules``). Sorted by name within each kind; [] where everything is
    here."""
    own, declared, named = set(), set(), set()
    for root in roots:
        for manifest in sorted(Path(root).rglob('package.xml')):
            package = ET.parse(manifest).getroot()
            own.add((package.findtext('name') or '').strip())
            declared |= {e.text.strip() for tag in RUN_DEPENDS for e in package.iter(tag)
                         if e.text and e.text.strip()}
        for controllers in sorted(Path(root).rglob('moveit_controllers.yaml')):
            data = yaml.safe_load(controllers.read_text())
            manager = data.get('moveit_controller_manager') if isinstance(data, dict) else None
            if isinstance(manager, str) and '/' in manager:
                named.add(manager.split('/', 1)[0])
    index = _on_the_index()
    out = [(n, 'declared') for n in sorted(declared - own)
           if PACKAGE_NAME.match(n) and n not in index]
    out += [(n, 'named') for n in sorted(named - declared - own) if n not in index]
    out += [(m, 'imported') for m in sorted(set(modules)) if not _importable(m)]
    return out


def _names(lacks: list, why: str) -> list:
    return [name for name, kind in lacks if kind == why]


def reason(lacks: list) -> str:
    """The skip message for ``missing()``'s answer: what is missing, why it is listed, and the
    fix -- the apt line for this machine, and what installs it for a bundle of one's own."""
    distro = os.environ.get('ROS_DISTRO', 'humble')
    declared, named, imported = (_names(lacks, w) for w in ('declared', 'named', 'imported'))
    what = []
    if declared:
        what.append(f'{", ".join(declared)} (the generated package.xml files exec_depend on '
                    f'{"it" if len(declared) == 1 else "them"}; not on the ament index)')
    if named:
        what.append(f'{", ".join(named)} (move_group\'s controller manager: the generated '
                    'moveit_controllers.yaml names it, and no generated package declares it yet)')
    if imported:
        what.append(f'the Python module{"s" if len(imported) > 1 else ""} {", ".join(imported)} '
                    f'(the launch harness imports {"it" if len(imported) == 1 else "them"})')
    packages = sorted({MODULE_PACKAGE.get(n, n) if kind == 'imported' else n
                       for n, kind in lacks})
    apt = [p for p in packages if not p.startswith('trainit_')]
    community = set(yaml.safe_load(COMMUNITY_REPOS.read_text())['repositories'])
    built = [p for p in packages if p.startswith('trainit_') and p in community]
    beyond = [p for p in packages if p.startswith('trainit_') and p not in community]
    fix = []
    if apt:
        fix.append('sudo apt install ' + ' '.join(f'ros-{distro}-{p.replace("_", "-")}'
                                                  for p in apt))
    if built:
        fix.append(f'build {", ".join(built)} in the TrainIt workspace (README.md, "Install")')
    if beyond:
        one = len(beyond) == 1
        fix.append(f'{", ".join(beyond)} {"is" if one else "are"} not TrainIt Community (Pro or '
                   'experimental): a bundle under a Community launch tier should not declare '
                   f'{"it" if one else "them"}')
    return (f'this launch tier stops before the build, as this machine lacks '
            f'{"; ".join(what)} -- fix: {"; ".join(fix)}. For a bundle of your own, '
            '`rosdep install --from-paths <the bundle> --ignore-src -r -y` installs what it '
            f'declares, and `sudo apt install ros-{distro}-moveit-simple-controller-manager` the '
            'controller manager (README.md, "Running a generated bundle")')


def require(*roots: Path, modules: tuple = ()) -> None:
    """Stop the running test when the bundles under ``roots`` declare, or the launch harness's
    ``modules`` are, what this machine lacks -- called after generation and before the build.
    It SKIPS naming each missing package or module and the fix; with TRAINIT_REQUIRE_GOLDEN=1 it
    FAILS instead (``eef_fixture.skip_or_fail``). A no-op where everything is here."""
    lacks = missing(*roots, modules=modules)
    if lacks:
        import eef_fixture        # pytest and TSA's model: imported only by a test that stops
        eef_fixture.skip_or_fail(reason(lacks))
