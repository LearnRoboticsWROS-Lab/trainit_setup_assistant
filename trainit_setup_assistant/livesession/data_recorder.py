"""TSA's runtime link to the TrainIt data recorder (D0.8, ADR-0029 Decision 7).

The recorder (``trainit_data_ros``, a separate process: the bundle's bring-up or a launched command) owns
acquisition, sessions and the canonical store. TSA, the control plane, has EXACTLY three clients of it -- the
recorder's ``std_srvs/Trigger`` services ``~/start_session``, ``~/capture`` and ``~/close_session`` -- created
here, only by the Data panel, resolved in the robot slot (ADR-0023); and it OBSERVES the recorder's 1 Hz
``diagnostic_msgs/DiagnosticArray`` (the relative ``diagnostics`` topic of that slot). Nothing else: no store
access, no recorder logic, no import of the data stack.

The ROS side runs on a background executor thread; results are handed to the Qt side through futures and a
lock-protected cache the GUI POLLS (the LiveCameraCapture pattern), so a widget is never touched from ROS.
rclpy is imported in ``__init__``, so importing this module never needs ROS.
"""

from __future__ import annotations

import threading
import time
from typing import Dict, Optional

NODE = 'trainit_data_recorder'
SERVICES = ('start_session', 'capture', 'close_session')


def recorder_base(namespace: str) -> str:
    """The recorder node's absolute name in a robot slot: ``/<slot>/trainit_data_recorder``."""
    slot = '/'.join(p for p in (namespace or '').split('/') if p)
    return f'/{slot}/{NODE}' if slot else f'/{NODE}'


def diagnostics_topic(namespace: str) -> str:
    slot = '/'.join(p for p in (namespace or '').split('/') if p)
    return f'/{slot}/diagnostics' if slot else '/diagnostics'


class RecorderLink:
    """The three recorder services + the recorder's diagnostics, for one robot slot."""

    def __init__(self, namespace: str = '', node_name: str = 'trainit_data_panel'):
        import rclpy
        from diagnostic_msgs.msg import DiagnosticArray
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from std_srvs.srv import Trigger

        self._rclpy = rclpy
        self._Trigger = Trigger
        # the process's ONE rclpy context, shared with the wizard's other observers: initialised if needed,
        # never shut down here (another observer -- the live preview, a link rebuilt for a new slot -- may be
        # using it); it ends with the process
        if not rclpy.ok():
            rclpy.init()
        self.base = recorder_base(namespace)
        self._node = Node(node_name)
        self._lock = threading.Lock()
        self._status: Optional[dict] = None
        # the ONLY clients TSA has of the data recorder (ADR-0029 Decision 7): three Trigger services
        self._clients = {name: self._node.create_client(Trigger, f'{self.base}/{name}') for name in SERVICES}
        self._node.create_subscription(DiagnosticArray, diagnostics_topic(namespace), self._on_diagnostics, 10)
        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spinning = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        while self._spinning and self._rclpy.ok():
            self._executor.spin_once(timeout_sec=0.1)

    def _on_diagnostics(self, msg):
        prefix = f'{NODE}: {self.base}'
        for status in msg.status:
            if status.name == prefix:
                values = {kv.key: kv.value for kv in status.values}
                with self._lock:
                    self._status = {'level': int.from_bytes(status.level, 'little')
                                    if isinstance(status.level, bytes) else int(status.level),
                                    'message': status.message, 'backend': status.hardware_id,
                                    'values': values, 'received': time.monotonic()}

    # --- Qt-side polling --------------------------------------------------------------------------------
    def reachable(self) -> bool:
        """The recorder's services are on the graph (observation: nothing is called)."""
        return all(client.service_is_ready() for client in self._clients.values())

    def status(self, max_age_s: float = 3.0) -> Optional[dict]:
        """The recorder's latest diagnostics ``{level, message, backend, values, age_s}``, or None when none
        arrived within ``max_age_s`` (the recorder is gone or not publishing)."""
        with self._lock:
            status = dict(self._status) if self._status else None
        if status is None:
            return None
        status['age_s'] = time.monotonic() - status.pop('received')
        return status if status['age_s'] <= max_age_s else None

    def call(self, service: str):
        """Call one of the three services asynchronously: an rclpy Future the GUI polls (``done()``), whose
        result is the Trigger response (``success``, ``message``)."""
        if service not in SERVICES:
            raise ValueError(f'{service!r} is not a recorder service TSA calls ({", ".join(SERVICES)})')
        return self._clients[service].call_async(self._Trigger.Request())

    def alive(self) -> bool:
        """The link still spins (another owner of the context may have shut it down: rebuild the link then)."""
        return self._thread.is_alive() and self._rclpy.ok()

    def forget(self, service: str, future) -> None:
        """Give up waiting for a call the recorder never answered (it died, or was restarted)."""
        try:
            self._clients[service].remove_pending_request(future)
        except Exception:  # noqa: BLE001 - already answered or gone
            pass

    def close(self) -> None:
        """Stop this link's node (the shared context stays: see ``__init__``)."""
        self._spinning = False
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)
        try:
            self._executor.remove_node(self._node)
            self._node.destroy_node()
        except Exception:  # noqa: BLE001
            pass


def parse_capture_reply(message: str) -> Dict[str, str]:
    """``~/capture``'s reply ``"<capture_id> <OUTCOME>"`` -> ``{capture_id, outcome}`` (the recorder's contract)."""
    parts = (message or '').split()
    if len(parts) == 2:
        return {'capture_id': parts[0], 'outcome': parts[1]}
    return {'capture_id': '', 'outcome': '', 'raw': message or ''}


__all__ = ['NODE', 'SERVICES', 'RecorderLink', 'diagnostics_topic', 'parse_capture_reply', 'recorder_base']
