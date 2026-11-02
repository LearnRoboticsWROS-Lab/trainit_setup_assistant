"""The YAML section (vNext I0.12, I0_DESIGN.md §20): the canonical file, exactly as a save writes it.

The YAML and the GUI are two views of ONE model. This panel shows the bytes a save would write
(read-only: hand edits go in the file, then "Reload from file"). It saves under D4: comments are
announced, and a ``.bak`` is written first. It also says when the file changed on disk since TSA read it.
"""

from __future__ import annotations

import os
from typing import Callable, Optional

from python_qt_binding.QtGui import QFont, QFontDatabase
from python_qt_binding.QtWidgets import (QFileDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                                         QPushButton, QVBoxLayout, QWidget)

from ...model.io import project_yaml


class YamlPanel(QWidget):
    def __init__(self, ctrl, reloaded: Callable[[], None] = lambda: None,
                 confirm: Callable[[str], bool] = None, parent=None):
        super().__init__(parent)
        self.ctrl, self._reloaded = ctrl, reloaded
        self._confirm = confirm or self._ask
        self._choose_path = self._ask_path
        v = QVBoxLayout(self)
        self.where = QLabel('')
        self.where.setWordWrap(True)
        v.addWidget(self.where)
        self.changed = QLabel('')
        self.changed.setWordWrap(True)
        self.changed.setStyleSheet('color:#8a6d3b;')
        v.addWidget(self.changed)
        self.text = QPlainTextEdit()
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setStyleHint(QFont.TypeWriter)
        self.text.setFont(mono)
        v.addWidget(self.text, 1)
        row = QHBoxLayout()
        self.save_button = QPushButton('Save project')
        self.save_button.clicked.connect(self.save)
        self.reload_button = QPushButton('Reload from file')
        self.reload_button.clicked.connect(self.reload)
        row.addWidget(self.save_button)
        row.addWidget(self.reload_button)
        row.addStretch(1)
        v.addLayout(row)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        v.addWidget(self.message)

    def _ask(self, question: str) -> bool:  # pragma: no cover - a modal dialog needs a user
        return QMessageBox.question(self, 'Save project', question) == QMessageBox.Yes

    def _ask_path(self) -> Optional[str]:  # pragma: no cover - a modal dialog needs a user
        name = f'{self.ctrl.project.project_name}.yaml' if self.ctrl.project is not None else 'project.yaml'
        path, _ = QFileDialog.getSaveFileName(self, 'Save project', os.path.join(os.path.expanduser('~'), name),
                                              'project (*.yaml *.yml)')
        return path or None

    def refresh(self) -> None:
        if self.ctrl.project is None:
            return
        self.text.setPlainText(project_yaml(self.ctrl.project))
        path = self.ctrl.project_path
        self.where.setText(f'<b>{path}</b> — the YAML a save writes (read-only here: edit the file, then '
                           'Reload)' if path else '<b>not saved yet</b> — the YAML a save writes')
        self.changed.setText('⚠ The file changed on disk since TSA read it: "Reload from file" shows it '
                             '(unsaved edits here would be replaced)' if self.ctrl.file_changed_on_disk() else '')

    def save(self) -> bool:
        if not self.ctrl.project_path:
            # never saved yet: choose where, so taught waypoints survive a TSA restart (owner Test B: the
            # save was only possible from Step 8)
            path = self._choose_path()
            if not path:
                self.message.setText('not saved: no file chosen')
                return False
            out = self.ctrl.save(path)
            self.message.setStyleSheet('')
            self.message.setText(f'saved to {out}: open it at Step 1 ("Open project.yaml") to continue later')
            self.refresh()
            return True
        saved, _backup, text = self.ctrl.save_project_file(self._confirm)
        self.message.setStyleSheet('' if saved else 'color:#8a6d3b;')
        self.message.setText(text)
        self.refresh()
        return saved

    def reload(self) -> bool:
        if self.ctrl.has_unsaved_changes() and not self._confirm(
                'Reload from file DISCARDS the edits made here since the last save. Reload?'):
            self.message.setText('not reloaded: your edits are kept')
            return False
        try:
            self.ctrl.reload_project()
        except Exception as exc:  # noqa: BLE001 - a hand edit the model refuses is shown, not raised
            self.message.setStyleSheet('color:#c62828;')
            self.message.setText(f'the file cannot be loaded (the GUI keeps the last valid project): {exc}')
            return False
        self.message.setStyleSheet('')
        self.message.setText('reloaded from the file')
        self._reloaded()
        self.refresh()
        return True
