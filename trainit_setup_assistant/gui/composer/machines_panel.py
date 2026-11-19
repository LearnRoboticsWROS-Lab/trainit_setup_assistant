"""The Machine services section (TSA 8.10.0, P0.b, P0B_DESIGN §7): the LOGICAL machine services a program uses.

A machine service is what a cell's machine serves -- a station's cycle, a line index -- named by an id, with an
optional one-line description. That is all the robot application says of a machine (owner decision 3): what serves
the service (the cell's machine simulation or its PLC) is the cell's, set up and started apart. The Tasks section's
"Machine request" and "Wait for machine" steps name these services.

Every edit is one Composer operation: add a service, change its description, remove one no step uses (a used one is
refused, naming its users). The panel holds no copy of the program.
"""

from __future__ import annotations

from typing import Callable, Optional

from python_qt_binding.QtWidgets import (QFormLayout, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
                                         QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget)

from ...composer.ops import ComposerError


class MachinesPanel(QWidget):
    def __init__(self, composer: Callable[[], object], parent=None):
        super().__init__(parent)
        self._composer = composer
        self.selected: Optional[str] = None
        v = QVBoxLayout(self)
        intro = QLabel('<b>Machine services</b> — the cell machines this program asks for work, by name. The cell '
                       'serves them (its machine simulation, or its PLC), started apart; a "Machine request" step '
                       'asks a service for one cycle and a "Wait for machine" step waits for it.')
        intro.setWordWrap(True)
        v.addWidget(intro)
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(['Id', 'Description', 'Used by'])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_select)
        v.addWidget(self.table, 1)
        self.editor = QGroupBox('Service')
        f = QFormLayout(self.editor)
        self.service_id = QLineEdit()
        self.service_id.setPlaceholderText('turner, indexer, ...')
        self.service_id.setToolTip('The service\'s name, as the cell serves it: a letter, then letters, digits or _')
        f.addRow('Id', self.service_id)
        self.description = QLineEdit()
        self.description.setPlaceholderText('optional: what the machine does (one line)')
        f.addRow('Description', self.description)
        row = QHBoxLayout()
        self.add_button = QPushButton('Add')
        self.add_button.clicked.connect(self.add)
        self.apply_button = QPushButton('Apply description')
        self.apply_button.clicked.connect(self.apply_description)
        self.remove_button = QPushButton('Remove')
        self.remove_button.clicked.connect(lambda: self.remove(self.selected))
        for b in (self.add_button, self.apply_button, self.remove_button):
            row.addWidget(b)
        row.addStretch(1)
        f.addRow(row)
        v.addWidget(self.editor)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        v.addWidget(self.message)

    @property
    def composer(self):
        return self._composer()

    def services(self) -> list:
        return [s for s in self.composer.document().get('machine_services') or [] if isinstance(s, dict)]

    def refresh(self) -> None:
        services = self.services()
        self.table.blockSignals(True)
        self.table.setRowCount(len(services))
        for i, s in enumerate(services):
            sid = s.get('id') or ''
            cells = (sid, s.get('description') or '—', ', '.join(self.composer.machine_service_users(sid)) or '—')
            for col, text in enumerate(cells):
                self.table.setItem(i, col, QTableWidgetItem(text))
        self.table.blockSignals(False)
        ids = [s.get('id') for s in services]
        if self.selected not in ids:
            self.selected = None
        self.apply_button.setEnabled(self.selected is not None)
        self.remove_button.setEnabled(self.selected is not None)

    def select(self, service_id: str) -> None:
        """Select a service: its id and description go to the editor (what the user types there is kept until then)."""
        services = {s.get('id'): s for s in self.services()}
        if service_id not in services:
            return
        self.selected = service_id
        row = list(services).index(service_id)
        self.table.blockSignals(True)
        self.table.selectRow(row)
        self.table.blockSignals(False)
        self.service_id.setText(service_id)
        self.description.setText(services[service_id].get('description') or '')
        self.apply_button.setEnabled(True)
        self.remove_button.setEnabled(True)

    def _on_select(self) -> None:
        rows = {i.row() for i in self.table.selectedItems()}
        if rows:
            self.select(self.table.item(min(rows), 0).text())

    def add(self) -> bool:
        sid, text = self.service_id.text().strip(), self.description.text().strip()
        if not sid:
            self._error('name the service first (its id)')
            return False
        if self._do(lambda: self.composer.add_machine_service(sid, text or None)):
            self.message.setText(f"machine service '{sid}' added")
            self.selected = sid
            self.refresh()
            self.select(sid)
            return True
        return False

    def apply_description(self) -> bool:
        if self.selected is None:
            return False
        sid, text = self.selected, self.description.text().strip()
        if self._do(lambda: self.composer.set_machine_service_description(sid, text or None)):
            self.message.setText(f"'{sid}' saved")
            self.refresh()
            return True
        return False

    def remove(self, service_id: Optional[str]) -> bool:
        if not service_id:
            return False
        if self._do(lambda: self.composer.remove_machine_service(service_id)):
            self.message.setText(f"machine service '{service_id}' removed")
            self.selected = None
            self.service_id.clear()
            self.description.clear()
            self.refresh()
            return True
        return False

    def _do(self, op) -> bool:
        try:
            op()
        except ComposerError as exc:
            self._error(str(exc))
            return False
        self.message.setStyleSheet('')
        return True

    def _error(self, text: str) -> None:
        self.message.setStyleSheet('color:#c62828;')
        self.message.setText(text)


__all__ = ['MachinesPanel']
