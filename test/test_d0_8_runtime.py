"""TrainIt D0.8 (ADR-0029 Decision 7 + Clarification D0.8): TSA's RUNTIME side of the Data section -- the
cockpit, never the runtime.

- boundary: TSA never imports the data stack or a simulator; its ONLY clients of the recorder are the three
  std_srvs/Trigger services, created in livesession/data_recorder.py;
- the recorder link against an in-process stand-in (services + diagnostics), on a private domain;
- the offline CLI as a launched process (``python3 -m trainit_data --json``) and its ``trainit-cli/1`` parse;
- the synthetic provider contract: TSA's own descriptor reader AGREES with the provider-side reader
  (``trainit_data.synthetic_provider``, run in a CHILD interpreter), discovery, the resolved configuration the
  provider itself accepts, the command, the progress protocol;
- the live preview's depth units come from trainit_perception's ``depth_to_metres`` (one decoding per side).
"""
import ast
import copy
import json
import os
import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import d08_env

from trainit_setup_assistant.composer import data_tools as D
from trainit_setup_assistant.livesession.data_recorder import (SERVICES, diagnostics_topic, parse_capture_reply,
                                                               recorder_base)

SRC = Path(__file__).resolve().parents[1] / 'trainit_setup_assistant'
REPO = Path(__file__).resolve().parents[2]
EXAMPLE = REPO / 'trainit_data' / 'examples' / 'isaac_replicator_bin'
# pxr is NOT here: TSA's USD scene importer (importers/, vNext I0.1) reads a stage file offline with it --
# a file reader, not the simulator or Replicator runtime
FORBIDDEN = ('trainit_data', 'trainit_data_ros', 'omni', 'isaacsim', 'carb')


# ---- boundary -------------------------------------------------------------------------------------------

def _imports(path: Path):
    tree = ast.parse(path.read_text(), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module


def test_tsa_never_imports_the_data_stack_or_a_simulator():
    found = []
    for f in SRC.rglob('*.py'):
        if 'resources' in f.parts:                     # Isaac-side scripts TSA ships, never imported by TSA
            continue
        found += [f'{f.relative_to(SRC)}: {m}' for m in _imports(f) if m.split('.')[0] in FORBIDDEN]
    assert found == []


def test_the_only_trigger_clients_are_the_three_recorder_services_of_the_data_link():
    owners = []
    for f in SRC.rglob('*.py'):
        if re.search(r'create_client\(\s*Trigger\b', f.read_text()):
            owners.append(f.relative_to(SRC).as_posix())
    assert owners == ['livesession/data_recorder.py']
    text = (SRC / 'livesession' / 'data_recorder.py').read_text()
    assert len(re.findall(r'create_client\(', text)) == 1          # one comprehension, over SERVICES
    assert SERVICES == ('start_session', 'capture', 'close_session')
    assert 'create_publisher' not in text and 'create_service' not in text


def test_recorder_names_resolve_in_the_robot_slot():
    assert recorder_base('') == '/trainit_data_recorder'
    assert recorder_base('r1') == recorder_base('/r1/') == '/r1/trainit_data_recorder'
    assert diagnostics_topic('cell/r1') == '/cell/r1/diagnostics' and diagnostics_topic('') == '/diagnostics'
    assert parse_capture_reply('ses-20261103T101500Z-a1b2c3d4/000003 PARTIAL') == {
        'capture_id': 'ses-20261103T101500Z-a1b2c3d4/000003', 'outcome': 'PARTIAL'}
    assert parse_capture_reply('NO_ACTIVE_SESSION: start one')['raw'].startswith('NO_ACTIVE')


# ---- the recorder link (an in-process stand-in on a private domain) --------------------------------------

@pytest.fixture
def ros_graph():
    rclpy = d08_env.import_or_require('rclpy')
    d08_env.import_or_require('std_srvs.srv')
    os.environ['ROS_DOMAIN_ID'] = str(200 + os.getpid() % 30)
    rclpy.init()
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()


def test_the_link_calls_the_three_services_and_reads_the_recorders_diagnostics(ros_graph):
    from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from std_srvs.srv import Trigger
    from trainit_setup_assistant.livesession.data_recorder import RecorderLink

    calls = []
    fake = Node('trainit_data_recorder', namespace='d08')

    def handler(name, reply):
        def serve(_request, response):
            calls.append(name)
            response.success, response.message = True, reply
            return response
        return serve
    fake.create_service(Trigger, '~/start_session', handler('start', 'ses-20261103T101500Z-a1b2c3d4'))
    fake.create_service(Trigger, '~/capture', handler('capture', 'ses-20261103T101500Z-a1b2c3d4/000001 COMPLETE'))
    fake.create_service(Trigger, '~/close_session', handler('close', '{"session_id": "x"}'))
    diag = fake.create_publisher(DiagnosticArray, 'diagnostics', 10)
    other = Node('another_node', namespace='d08')
    other_diag = other.create_publisher(DiagnosticArray, 'diagnostics', 10)
    executor = SingleThreadedExecutor()
    executor.add_node(fake)
    executor.add_node(other)
    stop = threading.Event()
    spinner = threading.Thread(target=lambda: [executor.spin_once(timeout_sec=0.05) for _ in iter(stop.is_set, True)],
                               daemon=True)
    spinner.start()
    link = RecorderLink('d08', node_name='trainit_data_panel_test')
    try:
        deadline = time.monotonic() + 10
        while not link.reachable() and time.monotonic() < deadline:
            time.sleep(0.05)
        assert link.reachable()
        for service, reply in (('start_session', 'ses-'), ('capture', 'COMPLETE'), ('close_session', 'session_id')):
            future = link.call(service)
            deadline = time.monotonic() + 5
            while not future.done() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert future.done() and future.result().success and reply in future.result().message
        assert calls == ['start', 'capture', 'close']
        with pytest.raises(ValueError):
            link.call('shutdown')
        assert link.status() is None
        status = DiagnosticStatus(name='trainit_data_recorder: /d08/trainit_data_recorder', hardware_id='mock',
                                  message='ACTIVE ses-x', level=DiagnosticStatus.OK,
                                  values=[KeyValue(key='captures_complete', value='3')])
        foreign = DiagnosticStatus(name='something_else: /d08/another_node', message='noise')
        deadline = time.monotonic() + 5
        while link.status() is None and time.monotonic() < deadline:
            other_diag.publish(DiagnosticArray(status=[foreign]))
            diag.publish(DiagnosticArray(status=[status]))
            time.sleep(0.1)
        got = link.status()
        assert got['message'] == 'ACTIVE ses-x' and got['values'] == {'captures_complete': '3'}
        assert got['backend'] == 'mock' and got['level'] == 0
    finally:
        link.close()
        stop.set()
        spinner.join(timeout=2)
        fake.destroy_node()
        other.destroy_node()


# ---- the offline CLI (a launched process) -----------------------------------------------------------------

def _cli_available():
    if D.cli_version() is None:
        d08_env.require('trainit_data is not importable by python3 here (source the workspace)')


def test_the_cli_runs_as_a_process_and_its_refusal_is_structured(tmp_path):
    _cli_available()
    result = D.run_cli(['dataset', 'list'], root=str(tmp_path / 'nowhere'))
    assert result.argv[:4] == ['python3', '-m', 'trainit_data', '--json'] and '--root' in result.argv
    assert result.exit_code == 1 and not result.ok and result.error['code'] == 'STORE_NOT_INITIALIZED'
    assert result.command == 'dataset list' and 'STORE_NOT_INITIALIZED' in result.message()
    assert not (tmp_path / 'nowhere').exists()
    assert re.fullmatch(r'\d+\.\d+\.\d+', D.cli_version())


def test_a_tool_that_does_not_answer_in_json_is_reported_not_parsed():
    bad = D.parse_cli_output(['x'], 2, 'usage: trainit-data ...', 'error: argument')
    assert bad.result is None and not bad.ok and 'did not answer' in bad.message()
    other = D.parse_cli_output(['x'], 0, json.dumps({'schema': 'trainit-cli/2', 'ok': True}), '')
    assert other.result is None and not other.ok


def test_the_recorder_command_is_launched_and_copyable():
    cmd = D.recorder_command('/ws', '/ws/install/a_app/share/a_app/config/data_collection.mock.yaml', 'r1', False,
                             '/data/store')
    assert cmd.kind == 'data_recorder' and cmd.long_running
    shown = cmd.display()
    assert shown.startswith("TRAINIT_DATA_ROOT=/data/store source /opt/ros/humble/setup.bash")
    assert 'ros2 launch trainit_data_ros recorder.launch.py config_file:=' in shown and 'namespace:=r1' in shown
    assert 'ros2 run' not in shown
    # a project without a robot slot: no `namespace:=` at all (`ros2 launch` refuses an empty value)
    flat = D.recorder_command('/ws', '/ws/x/data_collection.isaac.yaml', '', True, '/data/store').display()
    assert 'namespace:=' not in flat and flat.endswith('config_file:=/ws/x/data_collection.isaac.yaml '
                                                       'use_sim_time:=true')


def test_the_store_root_follows_settings_then_environment(monkeypatch):
    monkeypatch.setenv('TRAINIT_DATA_ROOT', '/env/store')
    assert D.data_root(SimpleNamespace(data_root='')) == '/env/store'
    assert D.data_root(SimpleNamespace(data_root='/set/store')) == '/set/store'
    monkeypatch.delenv('TRAINIT_DATA_ROOT')
    assert D.data_root(None) == str(Path('~/TrainItData').expanduser())


# ---- the synthetic provider contract ---------------------------------------------------------------------

def _example():
    if not (EXAMPLE / 'provider.json').is_file():
        d08_env.require('the trainit_data repository (its reference provider) is not beside TSA')
    return D.load_provider(EXAMPLE)


MUTATIONS = [
    lambda d: d.update(surprise=1),
    lambda d: d.update(schema='trainit-synthetic-provider/2.0'),
    lambda d: d.update(progress='other/1'),
    lambda d: d['parameters'].append(dict(d['parameters'][0])),
    lambda d: d['parameters'][0].update(type='vector'),
    lambda d: d['parameters'][0].update(level='expert'),
    lambda d: d['parameters'][0].update(key='lights..x'),
    lambda d: d['parameters'][1].update(min=1.0, max=0.5),
    lambda d: d['general']['runs'].update(default=0),
    lambda d: d['general'].pop('views'),
    lambda d: d.update(entry='/abs/generate.py'),
    lambda d: d['runtime'].update(kind='docker'),
    lambda d: None,                                         # the descriptor as shipped: both accept it
]


def _core_accepts(doc) -> bool:
    """The PROVIDER-side reader's verdict, in a child interpreter (TSA never imports trainit_data)."""
    code = ('import json, sys\nfrom trainit_data.synthetic_provider import parse_descriptor\n'
            'try:\n    parse_descriptor(json.loads(sys.stdin.read()))\nexcept Exception:\n    sys.exit(3)\n')
    proc = subprocess.run([sys.executable, '-c', code], input=json.dumps(doc), capture_output=True, text=True)
    if proc.returncode not in (0, 3):
        d08_env.require(f'trainit_data is not importable here: {proc.stderr[-300:]}')
    return proc.returncode == 0


@pytest.mark.parametrize('mutate', MUTATIONS)
def test_tsas_descriptor_reader_agrees_with_the_providers(mutate):
    _example()
    doc = json.loads((EXAMPLE / 'provider.json').read_text())
    mutate(doc)
    try:
        D.parse_descriptor(copy.deepcopy(doc))
        tsa = True
    except D.DescriptorError:
        tsa = False
    assert tsa == _core_accepts(doc)


def test_the_resolved_configuration_is_what_the_provider_itself_accepts(tmp_path):
    provider = _example()
    params = {'objects_per_run': [2, 3], 'lights.dome_intensity': [100.0, 200.0], 'resolution': [848, 480]}
    assert provider.problems(params, seed=5, runs=1, views=1) == []
    resolved = provider.resolved_config(params)
    assert resolved['objects_per_run'] == [2, 3] and resolved['lights']['key_intensity'] == \
        provider.base_config['lights']['key_intensity']
    code = ('import json, sys\nfrom trainit_data.synthetic_provider import config_problems, load_descriptor\n'
            'print(json.dumps(config_problems(load_descriptor(sys.argv[1]), json.loads(sys.stdin.read()))))\n')
    proc = subprocess.run([sys.executable, '-c', code, str(EXAMPLE)], input=json.dumps(resolved),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        d08_env.require(f'trainit_data is not importable here: {proc.stderr[-300:]}')
    assert json.loads(proc.stdout) == []
    assert provider.problems({'objects_per_run': [0, 3], 'nope': 1}, runs=0) == [
        'Objects per run: [0, 3] is outside [1, 20]', 'nope: provider isaac_replicator_bin has no such parameter',
        'Runs: 0 is outside [1, 10000]']


def test_providers_are_discovered_installed_and_machine_local(tmp_path):
    provider = _example()
    found, problems = D.find_providers([EXAMPLE.parent, tmp_path / 'missing'])
    assert [p.id for p in found] == ['isaac_replicator_bin'] and problems == []
    broken = tmp_path / 'broken'
    broken.mkdir()
    (broken / 'provider.json').write_text('{"schema": "trainit-synthetic-provider/1.0"}')
    found, problems = D.find_providers([EXAMPLE, broken, EXAMPLE])
    assert [p.id for p in found] == ['isaac_replicator_bin'] and len(problems) == 2
    assert any('already found earlier' in p for p in problems)
    installed = [d for d in D.provider_dirs(None) if d.name == 'synthetic_providers']
    if installed and installed[0].is_dir():
        found, _ = D.find_providers(installed)
        assert provider.id in [p.id for p in found]


def test_availability_and_the_launched_command(tmp_path):
    provider = _example()
    none = SimpleNamespace(isaac_launcher='')
    assert 'this project deploys mock' in D.provider_unavailable(provider, ['mock'], none)
    missing = SimpleNamespace(isaac_launcher=str(tmp_path / 'isaac-sim.sh'))
    assert 'Isaac Sim is not installed' in D.provider_unavailable(provider, ['isaac'], missing)
    (tmp_path / 'isaac-sim.sh').write_text('#!/bin/sh\n')
    (tmp_path / 'python.sh').write_text('#!/bin/sh\n')
    settings = SimpleNamespace(isaac_launcher=str(tmp_path / 'isaac-sim.sh'))
    assert D.provider_unavailable(provider, ['mock', 'isaac'], settings) is None
    cmd = D.provider_command(provider, settings, '/tmp/x/resolved.json', 7, 3, 2, '/data/store', store='/tmp/x/s')
    shown = cmd.display()
    assert cmd.kind == 'synthetic' and shown.startswith('env -i HOME="$HOME" PATH=/usr/bin:/bin '
                                                        'TRAINIT_DATA_ROOT=/data/store ')
    assert f'{tmp_path}/python.sh {EXAMPLE}/generate.py --config /tmp/x/resolved.json --seed 7 --runs 3 ' \
           f'--views 2 --store /tmp/x/s' in shown


def test_the_progress_protocol():
    assert D.parse_progress('TRAINIT_PROGRESS {"phase": "run", "run": 3, "runs": 24}') == {
        'phase': 'run', 'run': 3, 'runs': 24}
    assert D.parse_progress('TRAINIT_SESSION ses-20261103T101500Z-a1b2c3d4') == {
        'phase': 'done', 'session_id': 'ses-20261103T101500Z-a1b2c3d4'}
    assert D.parse_progress('TRAINIT_FAILED') == {'phase': 'failed'}
    for other in ('run 3/24: 5 objects', 'TRAINIT_PROGRESS not json', 'TRAINIT_PROGRESS {"run": 1}'):
        assert D.parse_progress(other) is None


# ---- the live preview reuses TP's depth units -------------------------------------------------------------

def _msg(encoding, array, pad=0):
    h, w = array.shape[:2]
    row = array.reshape(h, -1).view(np.uint8)
    data = np.concatenate([row, np.zeros((h, pad), np.uint8)], axis=1) if pad else row
    return SimpleNamespace(encoding=encoding, height=h, width=w, step=data.shape[1], data=data.tobytes(),
                           header=SimpleNamespace(frame_id='cam_optical'))


@pytest.mark.parametrize('encoding, raw, metres', [
    ('16UC1', np.array([[1000, 2500], [0, 65535]], np.uint16), np.array([[1.0, 2.5], [0.0, 65.535]], np.float32)),
    ('32FC1', np.array([[0.5, np.nan], [1.25, 3.0]], np.float32), np.array([[0.5, np.nan], [1.25, 3.0]], np.float32)),
])
def test_the_preview_converts_depth_with_tps_depth_to_metres(encoding, raw, metres):
    d08_env.import_or_require('trainit_perception.detectors.base')
    from trainit_perception.detectors import base as tp
    from trainit_setup_assistant.livesession.camera_capture import LiveCameraCapture
    cap = LiveCameraCapture.__new__(LiveCameraCapture)
    cap._np, cap._lock, cap._depth_to_metres = np, threading.Lock(), tp.depth_to_metres
    rgb = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    cap._on_pair(_msg('rgb8', rgb, pad=2), _msg(encoding, raw, pad=4))
    got_rgb, depth = cap._latest
    assert getattr(cap, '_last_error', None) is None
    np.testing.assert_array_equal(got_rgb, rgb)
    np.testing.assert_array_equal(depth, metres)
    np.testing.assert_array_equal(depth, tp.depth_to_metres(raw, encoding))


def test_without_tp_the_colour_still_shows_and_the_reason_is_given():
    from trainit_setup_assistant.livesession.camera_capture import LiveCameraCapture
    cap = LiveCameraCapture.__new__(LiveCameraCapture)
    cap._np, cap._lock, cap._depth_to_metres = np, threading.Lock(), None
    cap._depth_unavailable = 'depth needs trainit_perception'
    rgb = np.zeros((2, 2, 3), np.uint8)
    cap._on_pair(_msg('rgb8', rgb), _msg('16UC1', np.zeros((2, 2), np.uint16)))
    assert cap._latest[1] is None and cap._last_error == 'depth needs trainit_perception'


def test_the_preview_keeps_no_decoder_of_its_own():
    text = (SRC / 'livesession' / 'camera_capture.py').read_text()
    assert '/ 1000.0' not in text and 'depth_to_metres' in text



def test_the_runtime_is_dispatched_by_kind_never_named_by_the_panel(tmp_path, monkeypatch):
    """A provider's runtime is resolved through the ONE registry keyed by its descriptor's runtime kind: a new
    kind is a registry row; the panel and the validator name no runtime."""
    provider = _example()
    panel_source = (SRC / 'gui' / 'composer' / 'data_panel.py').read_text()
    assert 'isaac_launcher' not in panel_source and 'isaac_python' not in panel_source
    other = copy.deepcopy(provider)
    other.descriptor['runtime'] = {'kind': 'example_runtime'}
    assert 'not one TSA can launch' in D.provider_unavailable(other, ['isaac'], SimpleNamespace())
    monkeypatch.setitem(D.RUNTIME_LAUNCHERS, 'example_runtime', lambda settings: (tmp_path / 'py', ''))
    assert D.provider_unavailable(other, ['isaac'], SimpleNamespace()) is None
    cmd = D.provider_command(other, SimpleNamespace(), '/c.json', 1, 1, 1, '/root')
    assert f'{tmp_path}/py {EXAMPLE}/generate.py' in cmd.display()


# ---- the real CLI, happy path, through the exact invocation TSA uses ---------------------------------------

def _store(root):
    code = 'import sys\nfrom trainit_data.store import Store\nStore.open(sys.argv[1], create=True)\n'
    proc = subprocess.run([sys.executable, '-c', code, str(root)], capture_output=True, text=True)
    if proc.returncode != 0:
        d08_env.require(f'trainit_data is not importable here: {proc.stderr[-300:]}')


def test_the_real_cli_happy_path_is_parsed_and_propagated(tmp_path):
    """A real temporary store (made by trainit_data in a child process: TSA never writes a store), then
    python3 -m trainit_data --json as TSA runs it: exit 0, one trainit-cli/1 object, its result in TSA's hands."""
    _cli_available()
    root = tmp_path / 'store'
    _store(root)
    made = D.run_cli(['dataset', 'create', 'bin_parts', '--name', 'Bin parts'], root=str(root))
    assert made.exit_code == 0 and made.ok and made.error is None and made.command == 'dataset create'
    assert made.result['dataset']['dataset_id'] == 'bin_parts' and made.message() == 'ok'
    listed = D.run_cli(['dataset', 'list'], root=str(root))
    assert listed.ok and listed.exit_code == 0 and listed.stderr == ''
    assert [d['dataset_id'] for d in listed.result['datasets']] == ['bin_parts']
    detail = D.run_cli(['inspect', '--detail'], root=str(root))
    assert detail.ok and detail.result['sessions'] == [] and detail.result['root'] == str(root)
    again = D.run_cli(['dataset', 'create', 'bin_parts', '--name', 'Bin parts'], root=str(root))
    assert again.exit_code == 1 and not again.ok and again.error['code'] == 'DATASET_EXISTS'
