"""The Tasks section (vNext I0.7, I0_DESIGN.md §13, §16): the application structure, visually.

The tree IS the canonical nesting:
- a task's steps;
- a Repeat's or a Retry's body;
- each alternative of a "Try alternatives" (fallback).

A node is addressed by its canonical label, so a validator message ("task 'cell' step 4.2.1") points at
exactly one node. Every operation (add, move, delete, wrap, unwrap, make-one-path, edit) is one Composer
operation. The panel holds no copy of the program. The Route view lists the task's motions as
FROM → TO : HOW.
"""

from __future__ import annotations

import copy
from typing import Callable, List, Optional, Tuple

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtWidgets import (QAbstractItemView, QComboBox, QFormLayout, QGroupBox, QHBoxLayout,
                                         QLabel, QLineEdit, QListWidget, QPlainTextEdit, QPushButton, QSpinBox,
                                         QStackedWidget, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget)

from ...composer.describe import label_text, route, step_text
from ...composer.ops import CONTROL_KINDS, ComposerError, step_bodies, step_kind

Label = Tuple[int, ...]
ADDABLE = ('move', 'path', 'process', 'capture', 'dwell', 'detect', 'grasp', 'release', 'tool_close',
           'tool_open', 'tool', 'task')
ADD_TITLES = {'move': 'Move', 'path': 'Path', 'process': 'Process event', 'capture': 'Camera capture',
              'dwell': 'Wait', 'detect': 'Detect', 'grasp': 'Grasp', 'release': 'Release',
              'tool_close': 'Close gripper', 'tool_open': 'Open gripper', 'tool': 'Tool (actuator / verify)',
              'task': 'Run task'}


class TasksPanel(QWidget):
    def __init__(self, composer: Callable[[], object], parent=None):
        super().__init__(parent)
        self._composer = composer
        self.task: Optional[str] = None
        self._edit_label: Optional[Label] = None
        self._edit_original: Optional[dict] = None
        self.items = {}                          # canonical label -> tree node
        h = QHBoxLayout(self)
        # ---- tasks ----
        left = QVBoxLayout()
        left.addWidget(QLabel('<b>Tasks</b>'))
        self.task_list = QListWidget()
        self.task_list.currentTextChanged.connect(self._on_task)
        left.addWidget(self.task_list, 1)
        self.task_name = QLineEdit()
        self.task_name.setPlaceholderText('task name')
        left.addWidget(self.task_name)
        for text, slot in (('New task', lambda: self.new_task(self.task_name.text().strip())),
                           ('Rename', lambda: self.rename_task(self.task, self.task_name.text().strip())),
                           ('Delete', lambda: self.delete_task(self.task)),
                           ('Set as main', lambda: self.set_main(self.task))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            left.addWidget(b)
        h.addLayout(left, 1)
        # ---- tree + route ----
        mid = QVBoxLayout()
        self.title = QLabel('')
        mid.addWidget(self.title)
        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(['Step', '#'])
        self.tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.tree.itemSelectionChanged.connect(self._on_tree)
        mid.addWidget(self.tree, 3)
        bar = QHBoxLayout()
        self.add_kind = QComboBox()
        for k in ADDABLE:
            self.add_kind.addItem(ADD_TITLES[k], k)
        bar.addWidget(self.add_kind)
        self.add_after_button = QPushButton('Add after')
        self.add_after_button.clicked.connect(lambda: self.add_step(self.add_kind.currentData(), inside=False))
        self.add_inside_button = QPushButton('Add inside')
        self.add_inside_button.setToolTip('Into the selected Repeat / Retry body, or the selected alternative')
        self.add_inside_button.clicked.connect(lambda: self.add_step(self.add_kind.currentData(), inside=True))
        for b in (self.add_after_button, self.add_inside_button):
            bar.addWidget(b)
        bar.addStretch(1)
        for text, slot in (('▲', lambda: self.move(-1)), ('▼', lambda: self.move(1)),
                           ('Delete', self.delete_selected)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)
        mid.addLayout(bar)
        bar2 = QHBoxLayout()
        self.wrap_kind = QComboBox()
        self.wrap_kind.addItem('Repeat', 'repeat')
        self.wrap_kind.addItem('Retry', 'retry')
        self.wrap_kind.addItem('Try alternatives', 'fallback')
        self.wrap_times = QSpinBox()
        self.wrap_times.setRange(1, 1000)
        self.wrap_times.setValue(2)
        self.alternative_task = QComboBox()
        self.alternative_task.setToolTip('Try alternatives: the alternative that runs when the selected steps '
                                         'fail (a task)')
        wrap = QPushButton('Wrap selection')
        wrap.clicked.connect(self.wrap_selection)
        unwrap = QPushButton('Unwrap')
        unwrap.clicked.connect(self.unwrap_selected)
        make_path = QPushButton('Make one path')
        make_path.setToolTip('The selected LIN / CIRC moves become ONE blended path (the move before them is '
                             'its start)')
        make_path.clicked.connect(lambda: self.make_path())
        for w in (self.wrap_kind, QLabel('×'), self.wrap_times, QLabel('else'), self.alternative_task, wrap, unwrap,
                  make_path):
            bar2.addWidget(w)
        mid.addLayout(bar2)
        # owner Test C (friction row 30): why an Add / Wrap was refused, next to the buttons
        self.bar_message = QLabel('')
        self.bar_message.setWordWrap(True)
        self.bar_message.setStyleSheet('color:#c62828;')
        mid.addWidget(self.bar_message)
        mid.addWidget(QLabel('Route (FROM → TO : HOW)'))
        self.route_view = QPlainTextEdit()
        self.route_view.setReadOnly(True)
        mid.addWidget(self.route_view, 1)
        h.addLayout(mid, 3)
        # ---- inspector ----
        self.inspector = QGroupBox('Inspector')
        iv = QVBoxLayout(self.inspector)
        self.forms = QStackedWidget()
        self._build_forms()
        iv.addWidget(self.forms)
        self.apply_button = QPushButton('Apply')
        self.apply_button.clicked.connect(self.apply)
        iv.addWidget(self.apply_button)
        h.addWidget(self.inspector, 2)
        self.message = QLabel('')
        self.message.setWordWrap(True)
        iv.addWidget(self.message)
        self.inspector.setEnabled(False)

    # ---- inspector forms ---------------------------------------------------------------------------------
    def _build_forms(self) -> None:
        self.form_widgets = {}
        w = QWidget()
        f = QFormLayout(w)
        self.move_to, self.move_type, self.move_via, self.move_profile = (QComboBox(), QComboBox(), QComboBox(),
                                                                          QComboBox())
        self.move_type.addItems(['ptp', 'lin', 'circ', 'free'])
        f.addRow('To (WHERE)', self.move_to)
        f.addRow('How', self.move_type)
        f.addRow('Via (CIRC)', self.move_via)
        f.addRow('Profile', self.move_profile)
        self._add_form('move', w)
        w = QWidget()
        f = QFormLayout(w)
        self.path_name = QComboBox()
        f.addRow('Path', self.path_name)
        self._add_form('path', w)
        w = QWidget()
        f = QFormLayout(w)
        self.call_task = QComboBox()
        f.addRow('Task', self.call_task)
        self._add_form('task', w)
        w = QWidget()
        f = QFormLayout(w)
        self.dwell_ms = QSpinBox()
        self.dwell_ms.setRange(0, 3600000)
        self.dwell_ms.setSuffix(' ms')
        f.addRow('Wait', self.dwell_ms)
        self._add_form('dwell', w)
        w = QWidget()
        f = QFormLayout(w)
        self.count = QSpinBox()
        self.count.setRange(-1, 100000)
        self.count.setSpecialValueText('forever')          # repeat cycles -1: until the program stops
        f.addRow('Times', self.count)
        self._add_form('count', w)
        # vNext I0.8: process / capture, detect, grasp, release, tool (M3 scope)
        from .step_forms import FORMS
        self.step_forms = {}
        for kind, cls in FORMS.items():
            self.step_forms[kind] = cls()
            self._add_form(kind, self.step_forms[kind])
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel('This step, as the canonical program writes it:'))
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        v.addWidget(self.raw)
        self._add_form('other', w)

    def _add_form(self, name: str, w: QWidget) -> None:
        self.form_widgets[name] = w
        self.forms.addWidget(w)

    # ---- data --------------------------------------------------------------------------------------------
    @property
    def composer(self):
        return self._composer()

    def _doc(self) -> dict:
        return self.composer.document()

    def _task_doc(self) -> Optional[dict]:
        return next((t for t in self._doc().get('tasks', []) if t['name'] == self.task), None)

    def refresh(self) -> None:
        doc = self._doc()
        names = [t['name'] for t in doc.get('tasks', [])]
        main = doc.get('main_task')
        self.task_list.blockSignals(True)
        self.task_list.clear()
        for n in names:
            self.task_list.addItem(f'{n}  ★ main' if n == main else n)
            self.task_list.item(self.task_list.count() - 1).setData(Qt.UserRole, n)
        self.task_list.blockSignals(False)
        if self.task not in names:
            self.task = main if main in names else (names[0] if names else None)
        if self.task in names:
            self.task_list.setCurrentRow(names.index(self.task))
        self.alternative_task.clear()
        self.alternative_task.addItems([n for n in names if n != self.task])
        self._fill_tree()

    def _on_task(self, _text: str) -> None:
        item = self.task_list.currentItem()
        if item is not None and item.data(Qt.UserRole) != self.task:
            self.task = item.data(Qt.UserRole)
            self._edit_label = None
            self.refresh()

    def _fill_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        task = self._task_doc()
        if task is None:
            self.title.setText('')
            self.route_view.setPlainText('')
            self.tree.blockSignals(False)
            return
        self.title.setText(f"<b>{task['name']}</b>" + (f" — {task['description']}" if task.get('description') else ''))
        self.items = {}

        def add(parent, steps, prefix: Label):
            for i, step in enumerate(steps, start=1):
                label = prefix + (i,)
                item = QTreeWidgetItem([step_text(step), label_text(label)])
                item.setData(0, Qt.UserRole, ('step', label))
                parent.addChild(item) if parent is not None else self.tree.addTopLevelItem(item)
                self.items[label] = item
                kind = step_kind(step)
                if kind == 'fallback':
                    for b, body in enumerate(step['fallback'].get('bodies') or [], start=1):
                        alt = QTreeWidgetItem([f'Alternative {b}' + (' (tried first)' if b == 1 else ''),
                                               label_text(label + (b,))])
                        alt.setData(0, Qt.UserRole, ('alt', label, b))
                        item.addChild(alt)
                        add(alt, body, label + (b,))
                elif kind in ('repeat', 'retry'):
                    add(item, step_bodies(copy.deepcopy(step))[0], label)
        add(None, task.get('steps') or [], ())
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.blockSignals(False)
        paths = {p['name']: p for p in self._doc().get('paths') or []}
        self.route_view.setPlainText('\n'.join(route(task, paths=paths)) or '(no motion)')
        if self._edit_label in self.items:
            self.select(self._edit_label)

    # ---- selection ---------------------------------------------------------------------------------------
    def select(self, label: Label) -> None:
        item = self.items.get(tuple(label))
        if item is None:
            return
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        item.setSelected(True)
        self.tree.setCurrentItem(item)
        self.tree.blockSignals(False)
        self._load(tuple(label))

    def selected_labels(self) -> List[Label]:
        out = []
        for item in self.tree.selectedItems():
            data = item.data(0, Qt.UserRole)
            if data and data[0] == 'step':
                out.append(tuple(data[1]))
        return sorted(out)

    def _selected_alt(self):
        items = self.tree.selectedItems()
        if len(items) == 1:
            data = items[0].data(0, Qt.UserRole)
            if data and data[0] == 'alt':
                return tuple(data[1]), data[2]
        return None

    def _on_tree(self) -> None:
        labels = self.selected_labels()
        if len(labels) == 1:
            self._load(labels[0])
        else:
            self.inspector.setEnabled(False)

    def _load(self, label: Label) -> None:
        step = self.composer.step(self.task, label)
        kind = step_kind(step)
        self._edit_label, self._edit_original = label, step
        self.inspector.setEnabled(True)
        self.inspector.setTitle(f'Step {label_text(label)} — {step_text(step)}')
        doc = self._doc()
        wps = [w['name'] for w in doc.get('waypoints') or []]
        if kind == 'move':
            mv = step['move']
            self._combo(self.move_to, wps, mv['to'])
            self.move_type.setCurrentText(mv['type'])
            self._combo(self.move_via, [''] + wps, mv.get('via') if isinstance(mv.get('via'), str) else '')
            self._combo(self.move_profile, [''] + list(doc.get('motion_profiles') or {}), mv.get('profile') or '')
            self.forms.setCurrentWidget(self.form_widgets['move'])
        elif kind == 'path':
            self._combo(self.path_name, [p['name'] for p in doc.get('paths') or []], step['path'])
            self.forms.setCurrentWidget(self.form_widgets['path'])
        elif kind == 'task':
            self._combo(self.call_task, [t['name'] for t in doc.get('tasks', []) if t['name'] != self.task],
                        step['task'])
            self.forms.setCurrentWidget(self.form_widgets['task'])
        elif kind == 'dwell':
            self.dwell_ms.setValue(int(step['dwell'].get('ms') or 0))
            self.forms.setCurrentWidget(self.form_widgets['dwell'])
        elif kind in ('repeat', 'retry'):
            self.count.setMinimum(-1 if kind == 'repeat' else 1)
            self.count.setValue(int(step[kind].get('cycles' if kind == 'repeat' else 'attempts')))
            self.forms.setCurrentWidget(self.form_widgets['count'])
        elif kind in self.step_forms:
            from .step_forms import context
            self.step_forms[kind].load(step, context(doc, self.composer.project))
            self.forms.setCurrentWidget(self.step_forms[kind])
        else:
            import yaml
            shown = copy.deepcopy(step)
            if kind == 'fallback':
                shown = {'fallback': {'alternatives': len(step['fallback'].get('bodies') or [])}}
            self.raw.setPlainText(yaml.safe_dump(shown, sort_keys=False))
            self.forms.setCurrentWidget(self.form_widgets['other'])

    @staticmethod
    def _combo(combo: QComboBox, items, current) -> None:
        combo.clear()
        combo.addItems(list(items))
        if current and combo.findText(current) < 0:
            combo.addItem(current)
        combo.setCurrentText(current or '')

    # ---- edits -------------------------------------------------------------------------------------------
    def spec_from_inspector(self) -> dict:
        step = copy.deepcopy(self._edit_original)
        kind = step_kind(step)
        if kind == 'move':
            mv = step['move']
            mv['to'], mv['type'] = self.move_to.currentText(), self.move_type.currentText()
            via = self.move_via.currentText()
            if mv['type'] == 'circ' and via:
                mv['via'] = via
            elif isinstance(mv.get('via'), str) or mv['type'] != 'circ':
                mv.pop('via', None)
            if mv['type'] != 'circ':
                mv.pop('center', None)
            if self.move_profile.currentText():
                mv['profile'] = self.move_profile.currentText()
            else:
                mv.pop('profile', None)
        elif kind == 'path':
            step['path'] = self.path_name.currentText()
        elif kind == 'task':
            step['task'] = self.call_task.currentText()
        elif kind == 'dwell':
            step['dwell']['ms'] = self.dwell_ms.value()
        elif kind in ('repeat', 'retry'):
            step[kind]['cycles' if kind == 'repeat' else 'attempts'] = self.count.value()
        elif kind in self.step_forms:
            return self.step_forms[kind].spec()
        return step

    def apply(self) -> bool:
        if self._edit_label is None:
            return False
        spec = self.spec_from_inspector()
        label = self._edit_label
        return self._do(lambda: self.composer.replace_step(self.task, label, spec), select=label)

    def default_step(self, kind: str) -> dict:
        doc = self._doc()
        wps = [w['name'] for w in doc.get('waypoints') or []]
        if kind == 'move':
            if not wps:
                raise ComposerError('teach a waypoint first: a move goes TO a waypoint')
            move = {'to': wps[0], 'type': 'ptp'}
            profiles = list(doc.get('motion_profiles') or {})
            if profiles:                       # every motion runs with explicit limits (R-M3-07)
                move['profile'] = 'transit' if 'transit' in profiles else profiles[0]
            return {'move': move}
        if kind == 'path':
            paths = [p['name'] for p in doc.get('paths') or []]
            if not paths:
                raise ComposerError('create a path first (Paths section)')
            return {'path': paths[0]}
        if kind == 'task':
            others = [t['name'] for t in doc.get('tasks', []) if t['name'] != self.task]
            if not others:
                raise ComposerError('create the task to run first')
            return {'task': others[0]}
        if kind == 'dwell':
            return {'dwell': {'ms': 500}}
        if kind == 'process':
            return {'process': {'kind': 'process_on', 'channel': 'process'}}
        from .step_forms import context, default_step
        try:
            return default_step(kind, context(doc, self.composer.project), self.composer.project)
        except ValueError as exc:
            raise ComposerError(str(exc)) from None

    def add_step(self, kind: str, inside: bool = False) -> Optional[Label]:
        try:
            step = self.default_step(kind)
        except ComposerError as exc:
            self._error(str(exc))
            return None
        labels = self.selected_labels()
        alt = self._selected_alt()
        c = self.composer
        if inside:
            if alt is not None:
                parent, body = alt
                target = ('append', parent, body)
            elif len(labels) == 1 and step_kind(c.step(self.task, labels[0])) in ('repeat', 'retry'):
                target = ('append', labels[0], 1)
            else:
                self._error('select a Repeat / Retry, or an alternative of a Try alternatives, to add inside it')
                return None
            new = []
            ok = self._do(lambda: new.append(c.append_step(self.task, step, target[1], target[2])))
            return new[0] if ok else None
        at = (labels[-1][:-1] + (labels[-1][-1] + 1,)) if labels else None
        if at is None:
            new = []
            ok = self._do(lambda: new.append(c.append_step(self.task, step)))
            if ok:
                self.select(new[0])
            return new[0] if ok else None
        return at if self._do(lambda: c.insert_step(self.task, at, step), select=at) else None

    def delete_selected(self) -> bool:
        labels = self.selected_labels()
        if len(labels) != 1:
            self._error('select one step to delete')
            return False
        self._edit_label = None
        return self._do(lambda: self.composer.delete_step(self.task, labels[0]))

    def move(self, delta: int) -> bool:
        labels = self.selected_labels()
        if len(labels) != 1:
            return False
        new = []
        ok = self._do(lambda: new.append(self.composer.move_step(self.task, labels[0], delta)))
        if ok:
            self.select(new[0])
        return ok

    def _contiguous(self) -> Optional[Tuple[Label, int]]:
        labels = self.selected_labels()
        if not labels:
            return None
        parent = labels[0][:-1]
        idx = [l[-1] for l in labels]
        if any(l[:-1] != parent for l in labels) or idx != list(range(idx[0], idx[0] + len(idx))):
            self._error('select consecutive steps of ONE list')
            return None
        return labels[0], len(labels)

    def wrap_selection(self) -> bool:
        sel = self._contiguous()
        if sel is None:
            return False
        label, count = sel
        kind = self.wrap_kind.currentData()
        alternatives = []
        if kind == 'fallback':
            alt = self.alternative_task.currentText()
            if not alt:
                self._error('Try alternatives needs an alternative: create a task for it, then choose it')
                return False
            alternatives = [[{'task': alt}]]
        return self._do(lambda: self.composer.wrap_steps(self.task, label, count, kind, self.wrap_times.value(),
                                                         alternatives), select=label)

    def unwrap_selected(self) -> bool:
        labels = self.selected_labels()
        if len(labels) != 1:
            return False
        return self._do(lambda: self.composer.unwrap_step(self.task, labels[0]), select=labels[0])

    def make_path(self, name: Optional[str] = None, continuity: str = 'blend', radius_m: float = 0.005) -> bool:
        sel = self._contiguous()
        if sel is None:
            return False
        label, count = sel
        existing = {p['name'] for p in self._doc().get('paths') or []}
        if not name:
            i = 1
            while f'{self.task}_path{i}' in existing:
                i += 1
            name = f'{self.task}_path{i}'
        return self._do(lambda: self.composer.moves_to_path(self.task, label, count, name, continuity, radius_m),
                        select=label)

    # ---- tasks ---------------------------------------------------------------------------------------------
    def new_task(self, name: str, first_kind: str = 'dwell') -> bool:
        if not name:
            self._error('name the task first')
            return False
        try:
            first = self.default_step(first_kind)
        except ComposerError as exc:
            self._error(str(exc))
            return False
        ok = self._do(lambda: self.composer.add_task(name, steps=[first]))
        if ok:
            self.task = name
            self.task_name.clear()
            self.refresh()
        return ok

    def rename_task(self, old: Optional[str], new: str) -> bool:
        if not old or not new:
            return False
        ok = self._do(lambda: self.composer.rename_task(old, new))
        if ok:
            self.task = new
            self.refresh()
        return ok

    def delete_task(self, name: Optional[str]) -> bool:
        if not name:
            return False
        ok = self._do(lambda: self.composer.delete_task(name))
        if ok:
            self.task = None
            self.refresh()
        return ok

    def set_main(self, name: Optional[str]) -> bool:
        return bool(name) and self._do(lambda: self.composer.set_main_task(name))

    def _do(self, op, select: Optional[Label] = None) -> bool:
        try:
            op()
        except ComposerError as exc:
            self._error(str(exc))
            return False
        self.message.setStyleSheet('')
        self.message.setText('')
        self.bar_message.setText('')
        if select is not None:
            self._edit_label = tuple(select)
        self.refresh()
        return True

    def _error(self, text: str) -> None:
        self.message.setStyleSheet('color:#c62828;')
        self.message.setText(text)
        self.bar_message.setText(text)
