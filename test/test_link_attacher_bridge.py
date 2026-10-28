"""The Gazebo LinkAttacher bridge reads the plugin's answer (D8, found 2026-10-28).

The bridge used to call /ATTACHLINK fire-and-forget and log "ATTACH -> /ATTACHLINK" whatever the
plugin answered, so a weld the plugin refused (a model or link it cannot find, a weld it already
holds) was invisible, and the cycle still ended SUCCESS. Now every answer is logged, a refusal is
tried once more, and a second refusal is an ERROR that names the plugin's reason; an answer or a
retry of a grasp signal that has changed since is dropped, and a refused release with nothing
welded (the opening command at start-up) is no error.

The kit script is loaded as it ships (it is copied byte-identically into every Gazebo bundle);
the service clients are fakes, since IFRA_LinkAttacher is not part of TSA's test environment.
"""
import importlib.util
import time
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

KIT = (Path(__file__).resolve().parents[1] / 'trainit_setup_assistant' / 'resources' /
       'gazebo_cell_kit' / 'scripts' / 'link_attacher_bridge.py')
REFUSED = 'Failed to find model with name: red_cube'
NO_JOINT = 'DETACHED -- ERROR (Joint does not exist!)'
DONE = 'ATTACHED: {MODEL , LINK} -> {ur , wrist_3_link} -- {red_cube , link_1}.'


@pytest.fixture
def rclpy_ctx():
    rclpy = pytest.importorskip('rclpy')
    # a topic nobody publishes: a live cell's latched grasp signal must not reach the bridge
    rclpy.init(args=['--ros-args', '-p', 'gripper_cmd_topic:=/_tsa_test_link_attacher_unused'])
    yield rclpy
    if rclpy.ok():
        rclpy.shutdown()


class _Service:
    """A service type: what the bridge builds its request from."""
    Request = SimpleNamespace


class _Client:
    def __init__(self, name, available=True):
        self.srv_name, self.available = name, available
        self.requests, self.futures = [], []

    def wait_for_service(self, timeout_sec):
        return self.available

    def call_async(self, req):
        from rclpy.task import Future
        self.requests.append(req)
        self.futures.append(Future())
        return self.futures[-1]


class _Log:
    def __init__(self):
        self.lines = []

    def info(self, text):
        self.lines.append(('info', text))

    def warn(self, text):
        self.lines.append(('warn', text))

    def error(self, text):
        self.lines.append(('error', text))

    def at(self, level):
        return [text for lvl, text in self.lines if lvl == level]


@pytest.fixture
def bridge(rclpy_ctx, monkeypatch):
    spec = importlib.util.spec_from_file_location('link_attacher_bridge_kit', KIT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    node = module.LinkAttacherBridge()
    monkeypatch.setattr(module, 'AttachLink', _Service)
    monkeypatch.setattr(module, 'DetachLink', _Service)
    node._attach, node._detach = _Client('/ATTACHLINK'), _Client('/DETACHLINK')
    log = _Log()
    node.get_logger = lambda: log
    yield SimpleNamespace(node=node, log=log, attach=node._attach, detach=node._detach)
    node.destroy_node()


def _answer(future, success, message):
    with warnings.catch_warnings():
        warnings.simplefilter('error')      # an exception in the bridge's callback must surface
        future.set_result(SimpleNamespace(success=success, message=message))


def _spin_until(rclpy, node, done, timeout=3.0):
    end = time.monotonic() + timeout
    while not done() and time.monotonic() < end:
        rclpy.spin_once(node, timeout_sec=0.1)
    return done()


def _spin_past_the_retry(rclpy, node):
    for _ in range(10):                                  # 1 s, the retry waits 0.5 s
        rclpy.spin_once(node, timeout_sec=0.1)


def test_a_weld_the_plugin_confirms_is_logged_as_done(bridge):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    req = bridge.attach.requests[0]
    assert (req.model1_name, req.link1_name, req.model2_name, req.link2_name) == (
        'robot', 'ee_link', 'red_box', 'link')          # the node's parameter defaults
    _answer(bridge.attach.futures[0], True, DONE)
    assert bridge.log.at('info')[-1] == f'ATTACH done: {DONE}'
    assert not bridge.log.at('warn') and not bridge.log.at('error')
    assert len(bridge.attach.requests) == 1


def test_a_refused_weld_is_tried_once_more_then_reported_with_the_plugins_reason(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    _answer(bridge.attach.futures[0], False, REFUSED)
    assert bridge.log.at('warn') == [f'ATTACH refused by Gazebo ({REFUSED}): trying once more']
    assert _spin_until(rclpy_ctx, bridge.node, lambda: len(bridge.attach.requests) == 2)
    _answer(bridge.attach.futures[1], False, REFUSED)
    errors = bridge.log.at('error')
    assert len(errors) == 1 and REFUSED in errors[0] and 'weld was not made' in errors[0]
    _spin_past_the_retry(rclpy_ctx, bridge.node)          # once more only: no third request
    assert len(bridge.attach.requests) == 2


def test_a_refusal_then_a_weld_ends_done_with_no_error(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    _answer(bridge.attach.futures[0], False, REFUSED)
    assert _spin_until(rclpy_ctx, bridge.node, lambda: len(bridge.attach.requests) == 2)
    _answer(bridge.attach.futures[1], True, DONE)
    assert bridge.log.at('info')[-1] == f'ATTACH done: {DONE}' and not bridge.log.at('error')


def test_a_release_before_the_retry_drops_the_retry(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    _answer(bridge.attach.futures[0], False, REFUSED)
    bridge.node._on_cmd(SimpleNamespace(data=False))      # the cycle moved on: release
    assert len(bridge.detach.requests) == 1
    _spin_past_the_retry(rclpy_ctx, bridge.node)
    assert len(bridge.attach.requests) == 1               # no weld after the release


def test_a_retry_of_an_older_grasp_never_follows_a_new_one(bridge, rclpy_ctx):
    """grasp, refused, release, grasp again within the 0.5 s: the new grasp's weld only -- the old
    retry would be refused as "already attached" and logged as a failure of a healthy weld."""
    bridge.node._on_cmd(SimpleNamespace(data=True))
    _answer(bridge.attach.futures[0], False, REFUSED)
    bridge.node._on_cmd(SimpleNamespace(data=False))
    bridge.node._on_cmd(SimpleNamespace(data=True))
    assert len(bridge.attach.requests) == 2
    _answer(bridge.attach.futures[1], True, DONE)
    _spin_past_the_retry(rclpy_ctx, bridge.node)
    assert len(bridge.attach.requests) == 2 and not bridge.log.at('error')


def test_an_older_grasps_late_answer_is_dropped(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    bridge.node._on_cmd(SimpleNamespace(data=False))
    bridge.node._on_cmd(SimpleNamespace(data=True))
    _answer(bridge.attach.futures[0], False, REFUSED)     # the FIRST grasp's answer, late
    _spin_past_the_retry(rclpy_ctx, bridge.node)
    assert len(bridge.attach.requests) == 2 and not bridge.log.at('warn')


def test_a_late_refusal_of_a_superseded_request_is_not_retried(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    bridge.node._on_cmd(SimpleNamespace(data=False))
    _answer(bridge.attach.futures[0], False, REFUSED)     # answers after the release was sent
    _spin_past_the_retry(rclpy_ctx, bridge.node)
    assert len(bridge.attach.requests) == 1 and not bridge.log.at('error')


def test_a_refused_release_of_a_weld_is_reported_as_maybe_still_welded(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    _answer(bridge.attach.futures[0], True, DONE)
    bridge.node._on_cmd(SimpleNamespace(data=False))
    _answer(bridge.detach.futures[0], False, NO_JOINT)
    assert _spin_until(rclpy_ctx, bridge.node, lambda: len(bridge.detach.requests) == 2)
    _answer(bridge.detach.futures[1], False, NO_JOINT)
    errors = bridge.log.at('error')
    assert len(errors) == 1 and NO_JOINT in errors[0] and 'may still be welded' in errors[0]


def test_a_refused_release_with_nothing_welded_is_no_error(bridge, rclpy_ctx):
    """The opening command at start-up, or the release after a weld that was never made."""
    bridge.node._on_cmd(SimpleNamespace(data=False))
    _answer(bridge.detach.futures[0], False, NO_JOINT)
    _spin_past_the_retry(rclpy_ctx, bridge.node)
    assert len(bridge.detach.requests) == 1
    assert not bridge.log.at('warn') and not bridge.log.at('error')
    assert 'nothing this bridge welded' in bridge.log.at('info')[-1]


def test_a_missing_service_is_an_error(bridge):
    bridge.attach.available = False
    bridge.node._on_cmd(SimpleNamespace(data=True))
    assert bridge.log.at('error') == ['/ATTACHLINK not available: the part is not welded to the gripper']
    assert not bridge.attach.requests


def test_a_failed_call_is_reported_with_its_exception(bridge, rclpy_ctx):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        bridge.attach.futures[0].set_exception(RuntimeError('service call failed'))
    assert 'service call failed' in bridge.log.at('warn')[0]
    assert _spin_until(rclpy_ctx, bridge.node, lambda: len(bridge.attach.requests) == 2)


def test_the_same_signal_twice_sends_one_request(bridge):
    bridge.node._on_cmd(SimpleNamespace(data=True))
    bridge.node._on_cmd(SimpleNamespace(data=True))      # a latched topic re-delivers on connect
    assert len(bridge.attach.requests) == 1
