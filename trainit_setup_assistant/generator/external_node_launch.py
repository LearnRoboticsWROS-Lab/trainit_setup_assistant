"""Launch the EXTERNAL-NODE detectors: refusals, warnings, and the launch context.

``trainit_perception`` names two kinds of detection method (D-015). TSA has always emitted
the first — one ``detector_node`` per pure-python entry of ``config/perception.yaml`` — and
for the second it emitted **nothing at all**, with no field for a package, an executable or
a parameter file. A project could therefore configure a real YOLO-segmentation detector,
record it in ``perception.yaml``, bind it to a waypoint and rank it, and the bundle would
launch nothing that published ``/perception/<name>/detections``: ``DetectObject`` timed out,
which reads as a perception fault and is a configuration one.

This module turns :class:`~trainit_setup_assistant.model.external_node.ExternalNodeSpec`
into the ``Node`` entry that goes into the generated ``trainit_bt.launch.py`` beside the
pure-python ones, so **one bundle launches both kinds of method the same way**.

Three things it is careful about.

**Byte-identity.** A project with no external-node launch block contributes an EMPTY list,
the template's loop renders nothing, and no import is added — so the published bundles
regenerate byte-for-byte (``test_generator_golden.py`` and
``test_backend_bytestability.py`` are the gates, and they pass unchanged). The parameter
file is resolved with ``os.path.join(get_package_share_directory(...))``, both of which the
launch template already imports, precisely so that no new import can leak into a bundle
that has no external detector.

**Determinism.** Every mapping is written in SORTED key order. The same project must
produce the same bytes, and a dict that came from YAML carries file order, not meaning.

**No build dependency on the external package.** The generated ``package.xml`` does NOT
gain an ``<exec_depend>`` on it; it gains a COMMENT naming it, and generation emits a
warning. Reason, and it is ADR-0003: Community installs with ``rosdep``, and
``rosdep install --from-paths src --ignore-src`` on a bundle whose manifest names
a program's own detector package fails with *cannot locate rosdep definition* whenever that package is
not in the same workspace — which is the normal case, because the external detector is
customer IP that is shipped separately or not at all. A manifest entry would therefore
convert "the detector is not installed" (a readable ``ros2 launch`` failure, at the moment
you try to use it, naming the package) into "the whole workspace will not install" for
everyone, including people who never run that cell. The dependency is real but it is a
DEPLOYMENT fact, so it is documented where a person looks for dependencies instead of
declared where a tool would enforce it.
"""

from __future__ import annotations

from typing import Any, Dict, List

from ..model import CanonicalProject
from ..model.external_node import INPUT_ROLES
from ..model.perception import DetectorSpec


class _UseSimTime:
    """Marker for "write the launch file's own ``use_sim`` variable here", not a value.

    A sentinel object rather than a magic string, so no project value can ever collide
    with it and be turned into source code by accident.
    """

    __slots__ = ()


_USE_SIM = _UseSimTime()

# The generated launch file's own indentation: a Node(...) keyword, one of its list
# arguments, one dict item inside that list.
_KW = ' ' * 12
_ARG = ' ' * 16
_ITEM = ' ' * 20


# --- python source helpers (the launch file is python, so the context carries source) ---
def _py(value: Any) -> str:
    """A python literal for ``value``, double-quoted like the rest of the template."""
    if isinstance(value, bool):
        return 'True' if value else 'False'
    if value is None:
        return 'None'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_py(v) for v in value) + ']'
    if isinstance(value, dict):
        return '{' + ', '.join(f'{_py(k)}: {_py(v)}' for k, v in value.items()) + '}'
    text = str(value)
    if '"' in text or '\\' in text:             # fall back rather than emit broken source
        return repr(text)
    return f'"{text}"'


def _parameter_file_py(package: str, path: str) -> str:
    """Source that evaluates to the parameter file's path at launch time.

    An absolute path is used verbatim (a cell's tuned file may live outside any package);
    a relative one is resolved against the external package's SHARE directory, which is
    where ``setup.py``/``CMakeLists`` install it and the only place a *deployed* bundle can
    find it. ``get_package_share_directory`` raises ``PackageNotFoundError`` naming the
    package if it is not installed — the readable failure the missing ``exec_depend``
    deliberately leaves in place (see the module docstring).
    """
    if path.startswith('/'):
        return _py(path)
    parts = [p for p in path.split('/') if p]
    joined = ', '.join(_py(p) for p in parts)
    return f'os.path.join(get_package_share_directory({_py(package)}), {joined})'


def _resolved_node_names(project: CanonicalProject) -> Dict[str, List[str]]:
    """node name -> the detectors that would launch under it.

    Two nodes with one name is a genuine ROS fault (parameters and services collide), and
    the only way to get it here is an explicit ``node_name`` — so it is checked, not hoped.
    """
    out: Dict[str, List[str]] = {}
    per = project.perception
    for det in (per.detectors if per else []):
        if det.emits_node:
            name = f'detector_{det.name}'
        elif det.external_node is not None:
            name = det.external_node.resolved_node_name(det.name)
        else:
            continue
        out.setdefault(name, []).append(det.name)
    return out


# --- refusals -----------------------------------------------------------------------
def external_node_problems(project: CanonicalProject) -> List[str]:
    """Every external-node launch block that cannot be launched, as refusal messages.

    Pure, so a GUI can show the same text the generator would raise.
    """
    problems: List[str] = []
    per = project.perception
    for det in (per.detectors if per else []):
        ext = det.external_node
        if ext is None:
            continue
        if det.emits_node:
            # Both halves would run and both would publish /perception/<name>/detections:
            # two publishers on one topic, interleaved, and the consumer cannot tell.
            problems.append(
                f'detector "{det.name}": method "{det.method.value}" is a PURE-PYTHON '
                f'method — TSA already launches trainit_perception\'s detector_node for '
                f'it — but it also carries an external_node block '
                f'({ext.package or "?"}/{ext.executable or "?"}). Both would publish '
                f'/perception/{det.name}/detections, and nothing downstream could tell '
                f'the two streams apart. Choose one: drop the external_node block, or set '
                f'the method to an external-node one (custom / pcl_cluster).')
            continue
        missing = ext.missing()
        if missing:
            problems.append(
                f'detector "{det.name}": the external_node block has no '
                f'{" and no ".join(missing)} — so there is nothing to launch. An '
                f'external-node method publishes the contract from ITS OWN node, so the '
                f'bundle needs the ROS package and the executable to start '
                f'(ros2 run <package> <executable>). Without them the bundle launches '
                f'nothing on /perception/{det.name}/detections and DetectObject times '
                f'out, which looks like a perception fault and is a configuration one. '
                f'Set perception.detectors["{det.name}"].external_node.'
                f'{{{", ".join(missing)}}}.')
        unknown = ext.unknown_roles()
        if unknown:
            problems.append(
                f'detector "{det.name}": external_node.topic_parameters names the input '
                f'role(s) {", ".join(unknown)}, which are not part of the camera '
                f'contract. The contract has exactly three: '
                f'{", ".join(INPUT_ROLES)} — the same tokens the generated '
                f'perception.yaml uses in its input block. A role nobody recognises would '
                f'be dropped silently and the node would keep whatever its own parameter '
                f'file said.')
        if ext.parameter_file is not None and not ext.parameter_file.strip():
            # `os.path.join(share, )` would evaluate to the share DIRECTORY, and launch
            # would report "parameter file does not exist" about a path nobody wrote.
            problems.append(
                f'detector "{det.name}": external_node.parameter_file is blank. Give the '
                f'file a path (absolute, or relative to {ext.package or "the package"}\'s '
                f'share directory, e.g. "config/{det.name}.yaml"), or remove the key — '
                f'a node may well take all its parameters inline.')
        empty = sorted(role for role, param in ext.topic_parameters.items()
                       if not str(param).strip())
        if empty:
            problems.append(
                f'detector "{det.name}": external_node.topic_parameters maps '
                f'{", ".join(empty)} to an empty parameter name. Name the parameter the '
                f'node reads that topic from (e.g. "rgb_topic"), or remove the '
                f'entry and let the node take the topic from its own parameter file.')

    for node_name, owners in sorted(_resolved_node_names(project).items()):
        if len(owners) > 1:
            problems.append(
                f'detectors {", ".join(sorted(owners))} would all launch under the node '
                f'name "{node_name}". Two nodes with one name collide on their '
                f'parameters and their services, and the second one to start wins '
                f'silently. Give each external_node block its own node_name.')
    return problems


def refuse_incomplete_external_nodes(project: CanonicalProject) -> None:
    """REFUSE a launch block that cannot be launched, before a single file is written.

    A hard ``ValueError``, the same answer :func:`refuse_uncomputable_candidate_rules`
    gives and for the same reason: most generation problems leave a partial bundle that
    still helps, but a bundle whose detector never starts is worse than no bundle — it
    builds, it launches, and it fails in ``DetectObject`` three layers away from the
    cause.
    """
    problems = external_node_problems(project)
    if problems:
        raise ValueError('\n'.join(problems))


# --- warnings (NOT refusals: launching it yourself is a legitimate choice) -----------
def external_node_warnings(project: CanonicalProject) -> List[str]:
    """What the generated bundle does NOT do for an external-node detector.

    Two kinds, both reported rather than assumed:

    - a detector whose method is external-node and which carries NO launch block: TSA
      emits nothing for it. That is allowed — a vendor stack usually has its own bring-up
      — but it must be said, because the alternative is a silent bundle.
    - every external package the bundle LAUNCHES but does not DEPEND on (see the module
      docstring for why the ``exec_depend`` is deliberately absent).
    """
    out: List[str] = []
    per = project.perception
    launched = []
    for det in (per.detectors if per else []):
        if det.emits_node:
            continue
        if det.external_node is None:
            out.append(
                f'detector "{det.name}" ({det.method.value}) is an EXTERNAL-NODE method '
                f'with no external_node block: the bundle launches nothing for it and '
                f'/perception/{det.name}/detections stays silent unless you start its '
                f'node yourself. Set perception.detectors["{det.name}"].external_node '
                f'(package + executable) to have the bundle launch it.')
        else:
            launched.append((det.name, det.external_node))
    pkgs = sorted({ext.package for _, ext in launched if ext.package})
    if pkgs:
        out.append(
            'external detector package(s) ' + ', '.join(pkgs) + ' are LAUNCHED by this '
            'bundle but deliberately NOT declared as <exec_depend>: an undeclared-to-'
            'rosdep package name would make `rosdep install --from-paths src` fail for '
            'the whole workspace (ADR-0003), including for people who never run this '
            'cell. Install them beside the bundle; `ros2 launch` names the missing one '
            'if you do not. The generated package.xml records them in a comment.')
    return out


# --- the launch context -------------------------------------------------------------
def _entry(project: CanonicalProject, det: DetectorSpec) -> Dict[str, Any]:
    ext = det.external_node
    cam = project.perception.camera if project.perception else None
    topics = det.input_topics(cam)

    # Inline parameters: the project's own, then the camera topics under the parameter
    # names the node reads them from, then use_sim_time. Sorted, so the bytes are stable.
    inline: Dict[str, Any] = dict(ext.parameters)
    for role in INPUT_ROLES:                      # fixed order, not the dict's
        param = ext.topic_parameters.get(role)
        if param:
            inline[str(param).strip()] = topics[role]
    # The pure-python detector_node is launched with use_sim_time; an external detector on
    # the same camera needs the same clock or its stamps never match the TF buffer. An
    # explicit value in `parameters` wins — somebody who wrote it meant it.
    inline.setdefault('use_sim_time', _USE_SIM)

    # One parameter per line, sorted: a cell's parameter list is read by a person far more
    # often than it is written, and `parameters=[{...}]` on one 300-column line is where a
    # wrong topic hides.
    lines = [f'{_KW}parameters=[']
    if ext.parameter_file:
        lines.append(_ARG + _parameter_file_py(ext.package, ext.parameter_file.strip())
                     + ',')
    lines.append(_ARG + '{')
    for key in sorted(inline):
        value = inline[key]
        rendered = 'use_sim' if value is _USE_SIM else _py(value)
        lines.append(f'{_ITEM}{_py(key)}: {rendered},')
    lines.append(_ARG + '},')
    lines.append(_KW + '],')
    if ext.remappings:
        lines.append(f'{_KW}remappings=[')
        for key in sorted(ext.remappings):
            lines.append(f'{_ARG}({_py(key)}, {_py(ext.remappings[key])}),')
        lines.append(_KW + '],')

    return {
        'detector': det.name,
        'package': ext.package.strip(),
        'executable': ext.executable.strip(),
        'node_name': ext.resolved_node_name(det.name),
        # Pre-indented source, inserted at column 0 by the template: jinja indents only
        # the first line of a multi-line expression, so the indentation is built here.
        'arguments_block': '\n'.join(lines),
    }


def external_node_context(project: CanonicalProject) -> List[Dict[str, Any]]:
    """One dict per external detector the bundle launches, in detector order.

    Empty for every project that has no external_node block — which is what keeps the
    published bundles byte-identical.

    Re-checks the refusals, the way :func:`build_perception_yaml` re-checks its own: the
    orchestrator calls them before the first file is written, but an emitter can be driven
    directly (the byte-stability gate does exactly that), and a launch entry with no
    package must not be producible by any caller.
    """
    refuse_incomplete_external_nodes(project)
    per = project.perception
    return [_entry(project, det) for det in (per.detectors if per else [])
            if det.launches_external_node]


def external_node_packages(project: CanonicalProject) -> List[str]:
    """The external packages this bundle launches, sorted — for the package.xml comment."""
    per = project.perception
    return sorted({det.external_node.package.strip()
                   for det in (per.detectors if per else [])
                   if det.launches_external_node and det.external_node.package.strip()})
