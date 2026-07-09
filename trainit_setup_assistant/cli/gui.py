"""``trainit_setup_assistant`` — launch the RViz-native Setup Assistant wizard.

The wizard (Qt) edits a project via the controller and generates the bundle. To move
the end-effector with the real MoveIt gizmo, run the live session in parallel:
``ros2 launch trainit_setup_assistant setup_assistant.launch.py
moveit_config_package:=<...> robot_name:=<...>`` after generating+building a bootstrap.
"""

from __future__ import annotations

import sys


def main(argv=None) -> int:
    try:
        from python_qt_binding.QtWidgets import QApplication
        from ..gui.wizard import SetupWizard
    except Exception as exc:  # noqa: BLE001
        print(f'error: GUI deps unavailable: {exc}', file=sys.stderr)
        return 1

    app = QApplication(sys.argv if argv is None else argv)
    wizard = SetupWizard()
    wizard.show()
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())
