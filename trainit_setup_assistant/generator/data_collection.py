"""DATA COLLECTION: the ONE validator of a project's ``data_collection`` section, and its generation into the
bundle as the recorder's ``trainit-recording/1.0`` files (TrainIt D0.8, D0_8_DESIGN.md §7).

TSA is the control plane (ADR-0027 as narrowly amended by ADR-0029 Decision 7): it AUTHORS the recorder's
configuration and refuses what the recorder would refuse -- BEFORE generation, in the file's own terms -- against the
recorder's EXPORTED vocabulary as TSA's catalog pinned it (``catalog.recording_vocabulary()``, ADR-0021), and against
the project itself: the camera, the tasks and their canonical step ids (D0.5), the process channels. It never
imports ``trainit_data`` / ``trainit_data_ros``; the recorder re-validates the generated file at start-up and stays
the authority.

Generation (only when the section records live data: a camera and at least one trigger): one file per deployment
mode the camera's hardware profile can record from (``config/data_collection.<mode>.yaml``: the recorder's
``source.backend`` is a fact of the file, and the clock follows the mode), and the bring-up starts the recorder in
the robot slot with the file of the mode it runs (``bringup.launch.py``, ``data:=false`` to leave it out). The data
store is not in the bundle: ``TRAINIT_DATA_ROOT`` at launch.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

import yaml

from ..model import CanonicalProject
from ..model.data_collection import DataCollectionSpec, DataTrigger

CODE_SHAPE = 'R-D0-01'          # the model's own shape refusals (model/data_collection.py)
CODE_CAMERA = 'R-D0-02'
CODE_NAME = 'R-D0-03'
CODE_PROFILE = 'R-D0-04'
CODE_TRIGGER = 'R-D0-05'
CODE_REFERENCE = 'R-D0-06'
CODE_SYNC = 'R-D0-07'
CODE_SYNTHETIC = 'R-D0-08'
# the camera contract's stream names under a namespace (trainit_hardware_layer contracts/camera.md)
_CONTRACT = {'color': ('color/image_raw', 'color/camera_info'), 'depth': ('depth/image_rect_raw', None)}
_SIM_MODES = ('isaac', 'gazebo')
FILE_PATTERN = 'config/data_collection.{mode}.yaml'


def _vocabulary() -> dict:
    from ..catalog import recording_vocabulary
    return recording_vocabulary()


def sync_policies(vocab: Optional[dict] = None) -> Tuple[dict, List[dict]]:
    """(the recorder's default sync policy, every policy it offers) from the pinned recording vocabulary; each
    ``{'name': 'FIRST_AT_OR_AFTER_BOUNDARY', 'version': 1}``. The default is the one the vocabulary DECLARES,
    never a list position (a new row sorted first must not change what existing projects record)."""
    vocab = vocab or _vocabulary()
    return vocab['sync']['defaults']['policy'], list(vocab['sync']['policies'])


def chosen_sync_policy(dc: DataCollectionSpec, vocab: Optional[dict] = None) -> dict:
    """The policy a recording document names: the project's choice (its newest offered version), else the
    recorder's declared default."""
    default, offered = sync_policies(vocab)
    wanted = dc.sync.policy if dc.sync is not None else None
    if wanted is None:
        return default
    rows = [row for row in offered if row['name'].lower() == wanted]
    if not rows:
        raise ValueError(f'sync policy {wanted!r} is not offered by the recorder')
    return max(rows, key=lambda row: row['version'])


def _profiles() -> Dict[str, dict]:
    from ..catalog import camera_profiles
    return camera_profiles()


def _trigger_where(trigger: DataTrigger) -> str:
    return f"data collection trigger '{trigger.id}'"


def _tasks(project: CanonicalProject):
    """{task name: {canonical step id: step kind}} of a task program, or None for a template application."""
    if not project.application.is_task_program():
        return None
    from ..model.program import walk_steps
    out = {}
    for task in project.application.program.tasks:
        out[task.name] = {step.id: step.kind() for _, step in walk_steps(task.steps) if step.id}
    return out


def recordable_modes(project: CanonicalProject, vocab: Optional[dict] = None,
                     profiles: Optional[Dict[str, dict]] = None) -> Tuple[List[str], List[str]]:
    """(the deployment modes a recording file is generated for, the modes left without one)."""
    vocab = vocab or _vocabulary()
    profiles = _profiles() if profiles is None else profiles
    camera = project.perception.camera if project.perception else None
    row = profiles.get(camera.profile) if camera is not None and camera.profile else None
    supported = set(row['backends']) if row else set()
    # a simulated mode runs the recorder on use_sim_time: the profile must name the simulator clock there, or
    # the recorder refuses the session at start-up (recording_config.resolve)
    clocked = set((row or {}).get('sim_sources') or {})
    modes = [m.value for m in project.deployment.modes]
    recordable = [m for m in modes if m in vocab['backends'] and m in supported
                  and (m not in _SIM_MODES or m in clocked)]
    return recordable, [m for m in modes if m not in recordable]


def data_collection_problems(project: CanonicalProject) -> List[str]:
    """Every refusal of the project's data collection, one line each in its own terms (empty = valid).
    Generation refuses on any of them; the GUI shows them in the Diagnostics strip before Generate."""
    dc = project.data_collection
    camera = project.perception.camera if project.perception else None
    problems: List[str] = []
    if dc is None:
        if camera is not None and camera.profile is not None and camera.profile not in _profiles():
            problems.append(f'{CODE_PROFILE}: camera profile {camera.profile!r} is not a hardware-layer camera '
                            f'profile TSA knows ({", ".join(sorted(_profiles())) or "none"})')
        return problems
    vocab = _vocabulary()
    ident = re.compile(vocab['identifier_pattern'])

    def name(value: str, what: str) -> None:
        if not ident.fullmatch(value):
            problems.append(f'{CODE_NAME}: {what} {value!r} cannot be a recording identifier: lower-case '
                            'letters and digits, words joined by single "_" (the recorder refuses anything '
                            'else)')

    name(dc.dataset.id, 'data collection dataset id')
    if len(dc.dataset.id) > 64:
        problems.append(f'{CODE_NAME}: data collection dataset id {dc.dataset.id!r} is longer than 64 characters')
    if dc.dataset.classification.upper() not in vocab['classifications']:
        problems.append(f'{CODE_SHAPE}: data collection classification {dc.dataset.classification!r} is not '
                        f'one the recorder knows')

    # -- the camera: perception.camera, by its logical id; exactly one at D0.8 --
    if len(dc.sensors) > 1:
        problems.append(f'{CODE_CAMERA}: data collection records ONE camera (D0.8); the section lists '
                        f'{len(dc.sensors)} -- synchronized multi-camera recording is not offered yet')
    for sensor in dc.sensors:
        if camera is None:
            problems.append(f"{CODE_CAMERA}: data collection sensor '{sensor.camera}': the project has no "
                            'camera (configure it in Perception)')
            continue
        if camera.id is None:
            problems.append(f"{CODE_CAMERA}: data collection sensor '{sensor.camera}': the camera has no id "
                            '(give it one in Data -> Capture)')
        elif sensor.camera != camera.id:
            problems.append(f"{CODE_CAMERA}: data collection sensor '{sensor.camera}' is not the project's "
                            f"camera (its id is '{camera.id}')")
        relative = [topic for topic in (camera.rgb_topic, camera.depth_topic, camera.camera_info_topic)
                    if not topic.startswith('/')]
        if relative:
            problems.append(f"{CODE_CAMERA}: data collection sensor '{sensor.camera}': camera topics are "
                            f"cell-global, the recorder needs absolute names ({', '.join(relative)})")
        if 'depth' in sensor.streams and 'color' not in sensor.streams:
            problems.append(f"{CODE_CAMERA}: data collection sensor '{sensor.camera}': depth is registered to "
                            'colour -- record both streams')
    if dc.sensors and camera is not None and camera.id is not None:
        name(camera.id, 'camera id')
    if dc.sensors and camera is not None:
        profiles = _profiles()
        if camera.profile is None:
            problems.append(f'{CODE_PROFILE}: the camera has no hardware profile: recording needs one '
                            f'({", ".join(sorted(profiles)) or "none known"})')
        elif camera.profile not in profiles:
            problems.append(f'{CODE_PROFILE}: camera profile {camera.profile!r} is not a hardware-layer camera '
                            f'profile TSA knows ({", ".join(sorted(profiles)) or "none"})')
        elif not recordable_modes(project, vocab, profiles)[0]:
            problems.append(f'{CODE_PROFILE}: camera profile {camera.profile!r} declares no recording source '
                            f'for any deployment mode of this project '
                            f'({", ".join(m.value for m in project.deployment.modes)})')

    # -- triggers: the recorder's vocabulary, then the project's references --
    if dc.sensors and not dc.triggers:
        problems.append(f'{CODE_TRIGGER}: data collection records a camera but has no trigger: add one '
                        '(manual, interval, event or application state)')
    if dc.triggers and not dc.sensors:
        problems.append(f'{CODE_TRIGGER}: data collection has triggers but no camera to record')
    fields = vocab['triggers']['fields']
    ids = [t.id for t in dc.triggers]
    for dup in sorted({i for i in ids if ids.count(i) > 1}):
        problems.append(f"{CODE_TRIGGER}: data collection trigger id '{dup}' is used twice")
    for kind, most in vocab['triggers']['max_per_kind'].items():
        if sum(1 for t in dc.triggers if t.kind.upper() == kind) > most:
            problems.append(f'{CODE_TRIGGER}: data collection allows at most {most} {kind.lower()} trigger(s)')
    tasks = _tasks(project)
    channels = None
    for trigger in dc.triggers:
        where = _trigger_where(trigger)
        name(trigger.id, 'data collection trigger id')
        kind = trigger.kind.upper()
        if kind not in vocab['triggers']['kinds']:
            problems.append(f'{CODE_TRIGGER}: {where}: kind {trigger.kind} is not one the recorder implements')
            continue
        for stray in trigger.stray_fields():
            problems.append(f'{CODE_TRIGGER}: {where}: {stray} does not apply to a {trigger.kind} trigger')
        given = trigger.set_fields()
        for field, spec in fields[kind].items():
            if field == 'context':
                continue
            value = given.get(field)
            if value is None:
                if spec.get('required'):
                    problems.append(f'{CODE_TRIGGER}: {where}: {field} is required')
                continue
            if spec['type'] == 'int' and (isinstance(value, bool) or not isinstance(value, int)
                                          or value < spec['min']):
                problems.append(f'{CODE_TRIGGER}: {where}: {field} must be an integer >= {spec["min"]}')
            elif spec['type'] == 'enum' and value not in spec['values']:
                problems.append(f'{CODE_TRIGGER}: {where}: {field} {value!r} is not one of '
                                f'{", ".join(spec["values"])}')
            elif spec['type'] == 'identifier' and not ident.fullmatch(str(value)):
                problems.append(f'{CODE_NAME}: {where}: {field} {value!r} cannot be named by the recorder '
                                '(lower-case letters and digits, words joined by single "_")')
        if kind == 'APPLICATION_STATE' and trigger.pick_attempt is not None:
            phases = vocab['triggers']['pick_attempt_phases']
            if trigger.pick_attempt not in phases:
                problems.append(f'{CODE_TRIGGER}: {where}: pick_attempt {trigger.pick_attempt!r} is not one of '
                                f'{", ".join(phases)}')
            elif trigger.transition is not None and trigger.transition not in phases[trigger.pick_attempt]:
                problems.append(f'{CODE_TRIGGER}: {where}: a {trigger.pick_attempt} pick attempt fires on '
                                f'{" / ".join(phases[trigger.pick_attempt])}, not {trigger.transition}')
            if trigger.step is None:
                problems.append(f'{CODE_TRIGGER}: {where}: a pick attempt names the grasp STEP (a task has no '
                                'attempt)')
        # the application's own references: a deleted or renamed task / step is refused HERE, not at runtime
        if kind == 'APPLICATION_STATE' and trigger.task is not None:
            if tasks is None:
                problems.append(f'{CODE_REFERENCE}: {where}: application-state triggers need a Canonical Task '
                                'Program (this application has no tasks the runtime reports)')
            elif trigger.task not in tasks:
                problems.append(f"{CODE_REFERENCE}: {where}: no task is named '{trigger.task}' (tasks: "
                                f"{', '.join(tasks) or 'none'})")
            elif trigger.step is not None:
                steps = tasks[trigger.task]
                if trigger.step not in steps:
                    problems.append(f"{CODE_REFERENCE}: {where}: task '{trigger.task}' has no step with the id "
                                    f"'{trigger.step}' (step ids: {', '.join(steps) or 'none -- give a step an id'})")
                elif trigger.pick_attempt is not None and steps[trigger.step] != 'grasp':
                    problems.append(f"{CODE_REFERENCE}: {where}: a pick attempt is a grasp step's, and step "
                                    f"'{trigger.step}' is a {steps[trigger.step]} step")
        if kind == 'EVENT' and trigger.process_channel is not None:
            if channels is None:
                from ..composer.overview import process_channels
                channels = process_channels(project)
            if trigger.process_channel not in channels:
                problems.append(f"{CODE_REFERENCE}: {where}: no process channel '{trigger.process_channel}' is "
                                f"driven by the program (channels: {', '.join(channels) or 'none'})")

    if dc.sync is not None:
        if dc.sync.timeout_ms is not None and dc.sync.timeout_ms < 1:
            problems.append(f'{CODE_SYNC}: data collection sync timeout_ms must be >= 1')
        if dc.sync.tolerance_ms is not None and not dc.sync.tolerance_ms > 0:
            problems.append(f'{CODE_SYNC}: data collection sync tolerance_ms must be > 0')
        if dc.sync.policy is not None:
            offered = sorted({row['name'].lower() for row in sync_policies()[1]})
            if dc.sync.policy not in offered:
                problems.append(f'{CODE_SYNC}: data collection sync policy {dc.sync.policy!r} is not one the '
                                f'recorder offers ({", ".join(offered)})')

    if dc.synthetic is not None:
        recipe = dc.synthetic
        if not re.fullmatch(r'[a-z0-9]+(?:_[a-z0-9]+)*', recipe.provider):
            problems.append(f'{CODE_SYNTHETIC}: synthetic provider {recipe.provider!r} is not a provider id')
        for label, value, least in (('seed', recipe.seed, 0), ('runs', recipe.runs, 1),
                                    ('views', recipe.views, 1)):
            if value is not None and value < least:
                problems.append(f'{CODE_SYNTHETIC}: synthetic {label} must be >= {least}')
        for key in recipe.parameters:
            if not re.fullmatch(r'[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*', key):
                problems.append(f'{CODE_SYNTHETIC}: synthetic parameter {key!r} is not a dotted key path')
    return problems


def data_collection_warnings(project: CanonicalProject) -> List[str]:
    """What generation will do that the user may not expect (never a refusal)."""
    dc = project.data_collection
    if dc is None or not dc.records_live() or data_collection_problems(project):
        return []
    _, left = recordable_modes(project)
    if left:
        return [f'data collection: no recording in mode(s) {", ".join(left)} -- the camera profile (or the '
                'recorder) has no source for them; the bring-up starts no recorder there']
    return []


def _namespace_and_overrides(camera) -> Tuple[Optional[str], Dict[str, dict]]:
    """The camera namespace the CameraSpec's topics follow, and per-stream overrides where they do not."""
    suffix = '/' + _CONTRACT['color'][0]
    namespace = camera.rgb_topic[:-len(suffix)] if camera.rgb_topic.endswith(suffix) else None
    overrides: Dict[str, dict] = {'color': {}, 'depth': {}}
    if not namespace:
        # no contract namespace: every topic explicit; depth's CameraInfo is the colour one (contracts/camera.md,
        # "Registered depth": depth carries the colour grid and intrinsics), never a profile-namespace default
        overrides['color']['topic'] = camera.rgb_topic
        overrides['color']['info'] = camera.camera_info_topic
        overrides['depth']['topic'] = camera.depth_topic
        overrides['depth']['info'] = camera.camera_info_topic
        return None, overrides
    if camera.camera_info_topic != f'{namespace}/{_CONTRACT["color"][1]}':
        overrides['color']['info'] = camera.camera_info_topic
    if camera.depth_topic != f'{namespace}/{_CONTRACT["depth"][0]}':
        overrides['depth']['topic'] = camera.depth_topic
    return namespace, overrides


def recording_documents(project: CanonicalProject) -> Dict[str, dict]:
    """{mode: the trainit-recording/1.0 document} for every recordable mode; {} when nothing records live.
    Only what the user set is written; every default stays the recorder's (it records the effective one)."""
    dc = project.data_collection
    if dc is None or not dc.records_live():
        return {}
    vocab = _vocabulary()
    camera = project.perception.camera
    sensor = dc.sensors[0]
    namespace, overrides = _namespace_and_overrides(camera)
    policy = chosen_sync_policy(dc, vocab)
    triggers = []
    for t in dc.triggers:
        doc = {'id': t.id, 'kind': t.kind.upper()}
        for field in ('period_ms', 'max_captures', 'process_channel', 'edge', 'task', 'step', 'transition'):
            value = getattr(t, field)
            if value is not None:
                doc[field] = value
        if t.pick_attempt is not None:
            doc['context'] = {'pick_attempt': {'phase': t.pick_attempt}}
        triggers.append(doc)
    sync = {'policy': policy['name'], 'policy_version': policy['version']}
    if dc.sync is not None:
        for field in ('timeout_ms', 'tolerance_ms'):
            if getattr(dc.sync, field) is not None:
                sync[field] = getattr(dc.sync, field)
    out = {}
    for mode in recordable_modes(project, vocab)[0]:
        streams = {}
        for stream in sensor.streams:
            block = dict(overrides[stream])
            if stream == 'depth' and mode == 'isaac' and 'info' not in block:
                # contracts/camera.md, `isaac`: the Camera Helper publishes no depth/camera_info yet; under
                # "Registered depth" the colour CameraInfo IS the depth stream's (the D0.3 decision)
                block['info'] = camera.camera_info_topic
            streams[stream] = block
        entry = {'sensor_id': sensor.camera, 'profile': camera.profile}
        if namespace is not None:
            entry['namespace'] = namespace
        entry['streams'] = streams
        if mode == 'real':
            # a physical camera's session must name its device (D0.4, D0-C): the recorder refuses one without the
            # camera's source document (`<namespace>/source_info`, published by the hardware layer's launch)
            entry['source_info'] = {'required': True}
        out[mode] = {'schema': vocab['recording_schema'], 'source': {'backend': mode},
                     'rights': {'classification': dc.dataset.classification.upper()},
                     'sensors': [entry], 'triggers': triggers, 'sync': sync}
    return out


HEADER = ('# trainit-recording/1.0 -- the data recorder of this cell in mode {mode} (trainit_data_ros).\n'
          '# Generated by the TrainIt Setup Assistant from the project\'s data_collection section: edit the\n'
          '# project, not this file. Only what was configured is written; the recorder resolves every default\n'
          '# (topics and frames from the camera contract and the hardware profile) and records the EFFECTIVE\n'
          '# configuration in each session. The data store is TRAINIT_DATA_ROOT (default ~/TrainItData).\n')


def recording_yaml(mode: str, document: dict) -> str:
    return HEADER.format(mode=mode) + yaml.safe_dump(document, sort_keys=False, width=100)


def sim_time_modes() -> Tuple[str, ...]:
    return _SIM_MODES
