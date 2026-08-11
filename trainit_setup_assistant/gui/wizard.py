"""The RViz-native Setup Assistant wizard (Qt).

A thin QWizard over :class:`AssistantController`: each page reads its widgets and pushes
into the controller on ``validatePage()``. Kept logic-light so it is drivable headless
(``QT_QPA_PLATFORM=offscreen``) and so a future web GUI can reuse the controller.

M6 pages: Load Robot (S0), Group/Frames (S1), Named States (S2), Generate (S6).
Scene (S3) and Waypoints (S4/S5) pages are added in M7/M8.
"""

from __future__ import annotations

import os
from typing import Dict, Optional

from python_qt_binding.QtCore import Qt
from python_qt_binding.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWizard,
    QWizardPage,
)

from .controller import AssistantController


def _parse_joint_csv(text: str) -> Dict[str, float]:
    """Parse 'j1=0, j2=-1.08, ...' into {joint: value}."""
    out: Dict[str, float] = {}
    for tok in text.replace('\n', ',').split(','):
        tok = tok.strip()
        if not tok:
            continue
        key, _, val = tok.partition('=')
        out[key.strip()] = float(val.strip())
    return out


def _parse_floats(text: str) -> list:
    return [float(t.strip()) for t in text.replace(' ', '').split(',') if t.strip()]


def _expand_path(text: str) -> str:
    """Expand ~ and $VARS in a user-typed path (Qt line edits don't do this)."""
    return os.path.expanduser(os.path.expandvars(text.strip()))


class LoadRobotPage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('S0 — Load robot')
        self.setSubTitle('Open an existing project.yaml (keeps its collision matrix + cell '
                         'bridges), OR load a robot xacro fresh (group/frames auto-detected).')
        form = QFormLayout(self)
        # Option 1: open an existing complete project (recommended to start from a
        # working base — carries the self-collision matrix + deployment bridges).
        self.project_file = QLineEdit()
        pbrowse = QPushButton('Browse…')
        pbrowse.clicked.connect(self._browse_project)
        prow = QHBoxLayout()
        prow.addWidget(self.project_file)
        prow.addWidget(pbrowse)
        form.addRow('Open project.yaml', prow)
        # Option 2: load a robot xacro fresh (a bootstrap; you must add the collision
        # matrix + cell bridges separately).
        self.xacro = QLineEdit()
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.xacro)
        row.addWidget(browse)
        form.addRow('— or — Robot xacro', row)
        self.robot_name = QLineEdit()
        form.addRow('Robot name (optional)', self.robot_name)
        self.group_name = QLineEdit()
        form.addRow('Group name (optional)', self.group_name)
        self.meshes = QLineEdit()
        form.addRow('Meshes dir (optional)', self.meshes)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        form.addRow('Detected', self.summary)

    def _browse(self):  # pragma: no cover - needs a display
        path, _ = QFileDialog.getOpenFileName(self, 'Select robot xacro', '',
                                              'xacro/urdf (*.xacro *.urdf)')
        if path:
            self.xacro.setText(path)

    def _browse_project(self):  # pragma: no cover - needs a display
        path, _ = QFileDialog.getOpenFileName(self, 'Open project.yaml', '',
                                              'project (*.yaml *.yml)')
        if path:
            self.project_file.setText(path)

    def validatePage(self) -> bool:
        try:
            if self.project_file.text().strip():
                self.ctrl.open_project(_expand_path(self.project_file.text()))
            else:
                self.ctrl.new_from_robot(
                    _expand_path(self.xacro.text()),
                    robot_name=self.robot_name.text().strip() or None,
                    group_name=self.group_name.text().strip() or None,
                    meshes_dir=(_expand_path(self.meshes.text())
                                if self.meshes.text().strip() else None),
                )
        except Exception as exc:  # noqa: BLE001
            self.summary.setText(f'ERROR: {exc}')
            return False
        s = self.ctrl.robot_summary()
        matrix = 'yes' if (self.ctrl.project.robot.disable_collisions) else 'MISSING (bootstrap)'
        self.summary.setText(
            f"{s['robot_name']}: base={s['base_frame']} tip={s['tip_link']} "
            f"joints={s['arm_joints']} gripper={s['gripper_kind']} | collision-matrix: {matrix}")
        return True


class GroupFramesPage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('S1 — Group & frames')
        self.setSubTitle('Confirm or edit the planning group and frames.')
        form = QFormLayout(self)
        self.base = QLineEdit()
        self.tip = QLineEdit()
        self.group = QLineEdit()
        self.joints = QLineEdit()
        form.addRow('Base frame', self.base)
        form.addRow('Tip link', self.tip)
        form.addRow('Group name', self.group)
        form.addRow('Arm joints (csv)', self.joints)
        # Self-collision matrix: import it from an existing SRDF (e.g. one made by the
        # real MoveIt Setup Assistant). Needed for a fresh bootstrap to plan correctly.
        self.matrix_status = QLabel('')
        form.addRow('Collision matrix', self.matrix_status)
        mbtn = QPushButton('Import collision matrix from SRDF…')
        mbtn.clicked.connect(self._import_matrix)
        form.addRow('', mbtn)

    def initializePage(self) -> None:
        s = self.ctrl.robot_summary()
        self.base.setText(s['base_frame'])
        self.tip.setText(s['tip_link'])
        self.group.setText(s['group'])
        self.joints.setText(', '.join(s['arm_joints']))
        self._update_matrix_status()

    def _update_matrix_status(self) -> None:
        n = len(self.ctrl.project.robot.disable_collisions or [])
        self.matrix_status.setText(f'{n} pairs' if n else 'MISSING — planning will fail '
                                   '(import from an SRDF, e.g. MoveIt Setup Assistant)')

    def _import_matrix(self):  # pragma: no cover - file dialog needs a display
        path, _ = QFileDialog.getOpenFileName(self, 'Import collision matrix from SRDF',
                                              '', 'SRDF (*.srdf *.xml)')
        if not path:
            return
        try:
            self.ctrl.import_collision_matrix_from_srdf(_expand_path(path))
        except Exception as exc:  # noqa: BLE001
            self.matrix_status.setText(f'import failed: {exc}')
            return
        self._update_matrix_status()

    def validatePage(self) -> bool:
        joints = [j.strip() for j in self.joints.text().split(',') if j.strip()]
        if not joints:
            return False
        self.ctrl.set_frames(self.base.text().strip(), self.tip.text().strip())
        self.ctrl.set_group(self.group.text().strip(), joints)
        return True


class ControllersPage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('S2 — Controllers')
        self.setSubTitle('The robot controllers (ros2_control + MoveIt). Defaults match the '
                         'TrainIt runtime — edit only if your robot uses different names.')
        form = QFormLayout(self)
        self.arm_name = QLineEdit()
        self.arm_type = QLineEdit()
        self.arm_action_ns = QLineEdit()
        self.arm_update_rate = QSpinBox()
        self.arm_update_rate.setRange(1, 2000)
        self.arm_cmd_if = QLineEdit()
        self.arm_state_if = QLineEdit()
        form.addRow('Arm controller name', self.arm_name)
        form.addRow('Arm controller type', self.arm_type)
        form.addRow('Arm action ns', self.arm_action_ns)
        form.addRow('Update rate (Hz)', self.arm_update_rate)
        form.addRow('Command interfaces (csv)', self.arm_cmd_if)
        form.addRow('State interfaces (csv)', self.arm_state_if)
        self.grip_label = QLabel('')
        form.addRow('Gripper kind', self.grip_label)
        self.grip_name = QLineEdit()
        self.grip_type = QLineEdit()
        self.grip_action_ns = QLineEdit()
        form.addRow('Gripper controller name', self.grip_name)
        form.addRow('Gripper controller type', self.grip_type)
        form.addRow('Gripper action ns', self.grip_action_ns)

    def initializePage(self) -> None:
        arm = self.ctrl.project.robot.arm_controller
        self.arm_name.setText(arm.name)
        self.arm_type.setText(arm.type)
        self.arm_action_ns.setText(arm.action_ns)
        self.arm_update_rate.setValue(arm.update_rate)
        self.arm_cmd_if.setText(', '.join(arm.command_interfaces))
        self.arm_state_if.setText(', '.join(arm.state_interfaces))
        grip = self.ctrl.project.robot.gripper
        self.grip_label.setText(grip.kind.value)
        self.grip_name.setText(grip.controller_name or '')
        self.grip_type.setText(grip.controller_type)
        self.grip_action_ns.setText(grip.action_ns)

    def validatePage(self) -> bool:
        self.ctrl.set_arm_controller(
            name=self.arm_name.text().strip(),
            ctrl_type=self.arm_type.text().strip(),
            action_ns=self.arm_action_ns.text().strip(),
            update_rate=self.arm_update_rate.value(),
            command_interfaces=[s.strip() for s in self.arm_cmd_if.text().split(',') if s.strip()],
            state_interfaces=[s.strip() for s in self.arm_state_if.text().split(',') if s.strip()])
        self.ctrl.set_gripper_controller(
            controller_name=self.grip_name.text().strip(),
            controller_type=self.grip_type.text().strip(),
            action_ns=self.grip_action_ns.text().strip())
        return True


class NamedStatesPage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('S3 — Named states')
        self.setSubTitle('Capture named joint configurations (e.g. home). '
                         'Live capture reads /joint_states; offline, type the values.')
        layout = QVBoxLayout(self)
        self.list = QListWidget()
        layout.addWidget(self.list)
        form = QFormLayout()
        self.state_name = QLineEdit()
        self.state_joints = QLineEdit()
        form.addRow('Name', self.state_name)
        form.addRow('Joints (j1=0, j2=-1.08, …)', self.state_joints)
        layout.addLayout(form)
        btns = QHBoxLayout()
        capture = QPushButton('Capture current (live)')
        capture.clicked.connect(self.capture_live)
        add = QPushButton('Add / update')
        add.clicked.connect(self.add_state)
        remove = QPushButton('Remove selected')
        remove.clicked.connect(self.remove_selected)
        btns.addWidget(capture)
        btns.addWidget(add)
        btns.addWidget(remove)
        layout.addLayout(btns)

    def capture_live(self) -> None:  # pragma: no cover - needs a live ROS session
        """Read the current arm joint values from the running config session."""
        joints = self.ctrl.robot_summary()['arm_joints']
        try:
            values = self.wizard().live_capture().current_joint_values(joints)
        except Exception as exc:  # noqa: BLE001
            self.state_joints.setText(f'# capture failed: {exc}')
            return
        self.state_joints.setText(', '.join(f'{k}={v:.4f}' for k, v in values.items()))

    def initializePage(self) -> None:
        self._refresh()

    def _refresh(self) -> None:
        self.list.clear()
        for name in self.ctrl.robot_summary()['named_states']:
            self.list.addItem(name)

    def add_state(self) -> None:
        name = self.state_name.text().strip()
        if not name:
            return
        try:
            joints = _parse_joint_csv(self.state_joints.text())
        except ValueError:
            return
        self.ctrl.add_named_state(name, joints)
        self._refresh()

    def remove_selected(self) -> None:
        item = self.list.currentItem()
        if item:
            self.ctrl.remove_named_state(item.text())
            self._refresh()

    def validatePage(self) -> bool:
        return True  # named states are optional


class GeneratePage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 7 — Generate the bundle')
        self.setSubTitle('Name the bundle and generate the 3 packages (app + trainit_config '
                         '+ description). The bundle README has the one-command run.')
        form = QFormLayout(self)
        self.project_name = QLineEdit()
        form.addRow('Project / bundle name', self.project_name)
        self.output_dir = QLineEdit()
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        row = QHBoxLayout()
        row.addWidget(self.output_dir)
        row.addWidget(browse)
        form.addRow('Output dir', row)
        gen = QPushButton('Generate')
        gen.clicked.connect(self.generate)
        form.addRow(gen)
        self.result = QTextEdit()
        self.result.setReadOnly(True)
        form.addRow('Result', self.result)

    def initializePage(self) -> None:
        self.project_name.setText(self.ctrl.robot_summary()['robot_name'] + '_app')

    def _browse(self):  # pragma: no cover - needs a display
        path = QFileDialog.getExistingDirectory(self, 'Output directory')
        if path:
            self.output_dir.setText(path)

    def generate(self) -> None:
        name = self.project_name.text().strip()
        if name:
            self.ctrl.set_project_name(name)
        problems = self.ctrl.validate()
        out_dir = _expand_path(self.output_dir.text())
        if not out_dir:
            self.result.setPlainText('ERROR: please set an output directory')
            return
        try:
            manifest = self.ctrl.generate(out_dir)
        except Exception as exc:  # noqa: BLE001
            self.result.setPlainText(f'ERROR: {exc}')
            return
        info = manifest.as_dict()
        lines = [f"Generated {info['file_count']} files into {out_dir}"]
        # also persist the project so you can REOPEN it (e.g. to capture poses with the
        # gizmo after building this bootstrap) — the two-pass workflow.
        try:
            proj_path = os.path.join(out_dir, 'project.yaml')
            self.ctrl.save(proj_path)
            lines.append(f'Saved project.yaml -> {proj_path} (reopen to continue)')
        except Exception as exc:  # noqa: BLE001
            lines.append(f'(could not save project.yaml: {exc})')
        if problems:
            lines.append('Validation notes:')
            lines += [f'  - {p}' for p in problems]
        self.result.setPlainText('\n'.join(lines))


class ScenePage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 2 — Cell scene (from the USD)')
        self.setSubTitle('Load the USD + the mesh package: the TSA reads the cell prims + '
                         'base-frame poses and AUTO-SUGGESTS a collision mesh per group '
                         '(scanning <pkg>/meshes). Review the small table (category / grasp / '
                         'mesh), then "Apply mapping". static/actuated = CHECKED collision; '
                         'dynamic = allowed; grasp targets attach to the tool on close.')
        layout = QVBoxLayout(self)

        # --- USD import + auto-suggested mesh mapping (the recommended path) ---
        usd_form = QFormLayout()
        self.usd_path = QLineEdit()
        ubrowse = QPushButton('Browse…')
        ubrowse.clicked.connect(self._browse_usd)
        urow = QHBoxLayout(); urow.addWidget(self.usd_path); urow.addWidget(ubrowse)
        usd_form.addRow('USD scene', urow)
        self.mesh_pkg = QLineEdit()
        self.mesh_pkg.setPlaceholderText('mesh package, e.g. big1500_isaac')
        usd_form.addRow('Mesh package', self.mesh_pkg)
        # scene-loader attach param (not in the USD): links near the tool allowed to
        # touch a grasped object (self-collision relief). BIG1500: end_effector, tcp, wrist3_link.
        self.touch_links = QLineEdit('tcp')
        usd_form.addRow('Tool touch links (csv)', self.touch_links)
        load_usd = QPushButton('Load USD → auto-suggest meshes')
        load_usd.clicked.connect(self.load_usd_mapping)
        usd_form.addRow(load_usd)
        layout.addLayout(usd_form)
        # the mapping table: one row per prim GROUP (bottle_* collapses to one row)
        self._rules = []
        self.map_table = QTableWidget(0, 5)
        self.map_table.setHorizontalHeaderLabels(
            ['include', 'group (count)', 'category', 'grasp', 'mesh resource'])
        self.map_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        layout.addWidget(self.map_table)
        apply_map = QPushButton('Apply mapping → build scene')
        apply_map.clicked.connect(self.apply_usd_mapping_table)
        layout.addWidget(apply_map)
        # how a GRASPED object is shown while held (meshes stay meshes):
        #   remove  = disappears on grasp, reappears at the tool on release (simplest);
        #   attach_box = attaches as its AABB box so attached_collision_check works.
        grow = QHBoxLayout()
        grow.addWidget(QLabel('Grasp handling:'))
        self.grasp_mode = QComboBox()
        self.grasp_mode.addItems(['remove', 'attach_box'])
        grow.addWidget(self.grasp_mode)
        grow.addStretch(1)
        layout.addLayout(grow)

        layout.addWidget(QLabel('— resulting scene objects (edit individually below) —'))
        self.list = QListWidget()
        self.list.itemClicked.connect(self._on_select)
        layout.addWidget(self.list)
        form = QFormLayout()
        self.obj_id = QLineEdit('table')
        form.addRow('Object id', self.obj_id)
        self.shape = QComboBox()
        self.shape.addItems(['box', 'sphere', 'cylinder', 'cone', 'mesh'])
        form.addRow('Shape', self.shape)
        self.dims = QLineEdit('2.0, 2.0, 0.10')
        form.addRow('Dims (box x,y,z | sphere r | cyl/cone r,h)', self.dims)
        # mesh objects carry a package:// STL loaded by the scene loader (not AABB'd)
        self.mesh_resource = QLineEdit()
        self.mesh_resource.setPlaceholderText('shape=mesh: package://<pkg>/meshes/…stl')
        form.addRow('Mesh resource (mesh)', self.mesh_resource)
        self.position = QLineEdit('0.0, 0.0, -0.08')
        form.addRow('Position (x,y,z)', self.position)
        # cell role: static | actuated (URDF, adapter-driven) | dynamic (manipulated)
        self.category = QComboBox()
        self.category.addItems(['static', 'actuated', 'dynamic'])
        self.category.currentTextChanged.connect(self._category_changed)
        form.addRow('Category', self.category)
        # dynamic-object grasp handling (Step 8): does the gripper grasp THIS object
        # (attach on close), and what does it do on release (freeze | gravity)?
        self.grasp_target = QCheckBox('grasp target (attaches to the tool on gripper close)')
        form.addRow('Grasp', self.grasp_target)
        self.release_policy = QComboBox()
        self.release_policy.addItems(['freeze', 'gravity'])
        form.addRow('On release (Isaac)', self.release_policy)
        layout.addLayout(form)
        btns = QHBoxLayout()
        add = QPushButton('Add / update object')
        add.clicked.connect(self.add_object)
        remove = QPushButton('Remove selected')
        remove.clicked.connect(self.remove_selected)
        scene_yaml = QPushButton('Import scene.yaml (alt)')
        scene_yaml.clicked.connect(self.import_scene_yaml)
        preview = QPushButton('Preview in RViz (live)')
        preview.clicked.connect(self.preview_live)
        for b in (add, remove, scene_yaml, preview):
            btns.addWidget(b)
        layout.addLayout(btns)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

    def import_scene_yaml(self):  # pragma: no cover - file dialog needs a display
        path, _ = QFileDialog.getOpenFileName(self, 'Import scene.yaml (scene_manager_node)',
                                              '', 'scene (*.yaml *.yml)')
        if not path:
            return
        try:
            n = self.ctrl.import_scene_yaml(_expand_path(path))
            self.status.setText(f'Imported {n} objects from scene.yaml (meshes + poses + '
                                'attach ids) — matches the baseline scene.')
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f'import failed: {exc}')
        self._refresh()

    def _browse_usd(self):  # pragma: no cover - needs a display
        path, _ = QFileDialog.getOpenFileName(self, 'Load USD scene', '',
                                              'USD (*.usd *.usda *.usdc)')
        if path:
            self.usd_path.setText(path)

    def load_usd_mapping(self):
        usd = _expand_path(self.usd_path.text())
        pkg = self.mesh_pkg.text().strip()
        if not usd or not pkg:
            self.status.setText('set the USD path + mesh package first')
            return
        try:
            self._rules = self.ctrl.import_usd_cell(usd, pkg)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f'USD load failed: {exc}')
            return
        self._fill_table(self._rules)
        scan = getattr(self.ctrl, 'mesh_scan', {}) or {}
        mdir, stls = scan.get('dir'), scan.get('stls', [])
        matched = sum(1 for r in self._rules if r['mesh'])
        if not mdir or not stls:
            self.status.setText(
                f'⚠ NO MESHES FOUND for package "{pkg}" — every suggestion is EMPTY, so the '
                f'scene would not render. Check the Mesh package name (it must be the ROS '
                f'package that ships meshes/, e.g. big1500_isaac).')
        else:
            self.status.setText(
                f'{len(self._rules)} prim groups · scanned {mdir} ({len(stls)} STLs) · '
                f'{matched}/{len(self._rules)} groups matched a mesh — review, then "Apply mapping".')

    def _fill_table(self, rules):
        self.map_table.setRowCount(len(rules))
        for i, r in enumerate(rules):
            inc = QTableWidgetItem()
            inc.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            inc.setCheckState(Qt.Checked if r['include'] else Qt.Unchecked)
            self.map_table.setItem(i, 0, inc)
            g = QTableWidgetItem(f"{r['group']}  (x{r['count']})")
            g.setFlags(Qt.ItemIsEnabled)
            self.map_table.setItem(i, 1, g)
            cat = QComboBox()
            cat.addItems(['static', 'actuated', 'dynamic'])
            cat.setCurrentText(r['category'])
            self.map_table.setCellWidget(i, 2, cat)
            gr = QTableWidgetItem()
            gr.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
            gr.setCheckState(Qt.Checked if r['grasp'] else Qt.Unchecked)
            self.map_table.setItem(i, 3, gr)
            self.map_table.setItem(i, 4, QTableWidgetItem(r['mesh']))

    def _read_table(self):
        rules = []
        for i, base in enumerate(self._rules):
            r = dict(base)
            r['include'] = self.map_table.item(i, 0).checkState() == Qt.Checked
            cat = self.map_table.cellWidget(i, 2)
            r['category'] = cat.currentText() if cat else r['category']
            r['grasp'] = self.map_table.item(i, 3).checkState() == Qt.Checked
            r['mesh'] = self.map_table.item(i, 4).text().strip()
            rules.append(r)
        return rules

    def apply_usd_mapping_table(self):
        if not self._rules:
            self.status.setText('load a USD first')
            return
        try:
            n = self.ctrl.apply_usd_mapping(self._read_table())
            self.ctrl.set_scene_loader_params(grasp_attach_mode=self.grasp_mode.currentText())
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f'apply failed: {exc}')
            return
        self.status.setText(f'built {n} scene objects (meshes) — grasp handling: '
                            f'{self.grasp_mode.currentText()}.')
        self._refresh()

    def _on_select(self, item):  # pragma: no cover - needs a display
        """Populate the form from a selected object so it can be edited / re-flagged."""
        oid = item.text().split(' ')[0]
        obj = next((o for o in self.ctrl.project.scene.objects if o.id == oid), None)
        if obj is None:
            return
        self.obj_id.setText(obj.id)
        self.shape.setCurrentText(obj.shape.value)
        self.dims.setText(', '.join(str(d) for d in obj.dims))
        self.mesh_resource.setText(obj.mesh_resource or '')
        self.position.setText(', '.join(str(p) for p in obj.position))
        self.category.blockSignals(True)       # don't fire _category_changed on populate
        self.category.setCurrentText(obj.category.value)
        self.category.blockSignals(False)
        self.grasp_target.setChecked(obj.grasp_target)
        self.release_policy.setCurrentText(obj.release_policy.value)

    def _category_changed(self, category):  # pragma: no cover - needs a display
        """Changing the category applies immediately to the object named in the form."""
        oid = self.obj_id.text().strip()
        if any(o.id == oid for o in self.ctrl.project.scene.objects):
            self.ctrl.set_object_category(oid, category)
            self._refresh()

    def initializePage(self):
        # touch_links were derived from the robot chain at Step 1 — show them, don't ask.
        try:
            self.touch_links.setText(', '.join(self.ctrl.project.scene.touch_links))
        except Exception:  # noqa: BLE001
            pass
        self._refresh()

    def _refresh(self):
        self.list.clear()
        for o in self.ctrl.project.scene.objects:
            self.list.addItem(f'{o.id} [{o.shape.value}, {o.category.value}]')

    def add_object(self):
        oid = self.obj_id.text().strip()
        if not oid:
            return
        shape = self.shape.currentText()
        try:
            dims = _parse_floats(self.dims.text()) if self.dims.text().strip() else [0.1, 0.1, 0.1]
            position = _parse_floats(self.position.text())
        except ValueError:
            return
        cat = self.category.currentText()
        is_dyn = (cat == 'dynamic')
        self.ctrl.add_scene_object(
            oid, dims, position, shape=shape, category=cat,
            mesh_resource=(self.mesh_resource.text().strip() or None) if shape == 'mesh' else None,
            grasp_target=(is_dyn and self.grasp_target.isChecked()),
            release_policy=self.release_policy.currentText())
        self._refresh()

    def remove_selected(self):
        item = self.list.currentItem()
        if item:
            oid = item.text().split(' ')[0]
            self.ctrl.project.scene.objects = [
                o for o in self.ctrl.project.scene.objects if o.id != oid]
            self._refresh()

    def preview_live(self):  # pragma: no cover - needs a live ROS session
        try:
            self.wizard().planning_scene().publish_scene(self.ctrl.project.scene)
        except Exception:  # noqa: BLE001
            pass

    def validatePage(self) -> bool:
        tl = [s.strip() for s in self.touch_links.text().split(',') if s.strip()]
        if tl:
            self.ctrl.set_scene_loader_params(touch_links=tl)
        return True


class ApplicationPage(QWizardPage):
    # Every application the TrainIt Motion Runtime can host (only pick_and_place is
    # wired end-to-end in this MVP; the rest are shown greyed = coming / PRO).
    APPS = [('pick_and_place', True), ('gluing', False), ('follow_path', False),
            ('waypoint_replay', False), ('cnc', False)]

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 5 — Application type')
        self.setSubTitle('Pick the robotic application. The greyed ones are hosted by the '
                         'TrainIt Motion Runtime but not yet wired in this MVP (PRO / '
                         'roadmap). Motion type + planner are chosen PER WAYPOINT in the '
                         'next step — there is no single global planner. The manipulated '
                         'objects come from the scene (grasp targets), not from here.')
        form = QFormLayout(self)
        self.app_type = QComboBox()
        for name, enabled in self.APPS:
            self.app_type.addItem(name if enabled else f'{name}  (coming soon)')
        model = self.app_type.model()          # grey out the not-yet-available apps
        for i, (_, enabled) in enumerate(self.APPS):
            if not enabled:
                model.item(i).setEnabled(False)
        form.addRow('Application', self.app_type)

    def validatePage(self) -> bool:
        name = self.APPS[max(0, self.app_type.currentIndex())][0]
        if not self.APPS[self.app_type.currentIndex()][1]:
            name = 'pick_and_place'            # guard: only the available one is set
        # planner is per-waypoint; keep a sensible fallback for segments that don't set one
        self.ctrl.set_application(name, 'ompl')
        return True


class WaypointsPage(QWizardPage):
    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 6 — Waypoints, motions & dynamic-object flags')
        self.setSubTitle('The heart: for each move pick a named SRDF state (or capture a '
                         'TCP pose with the gizmo) AND choose motion / planner / speed + '
                         'the per-move attached-collision-check. Add grasp/release where '
                         'the gripper acts. Re-add a waypoint to re-visit it.')
        layout = QVBoxLayout(self)
        self.seq = QListWidget()
        layout.addWidget(self.seq)
        form = QFormLayout()
        self.move_name = QLineEdit()
        form.addRow('Move name', self.move_name)
        self.target_mode = QComboBox()
        self.target_mode.addItems(['tcp', 'named'])
        form.addRow('Target', self.target_mode)
        self.pos = QLineEdit()
        self.quat = QLineEdit('0, 0, 0, 1')
        cap = QPushButton('Capture pose (live)')
        cap.clicked.connect(self.capture_pose)
        posrow = QHBoxLayout()
        posrow.addWidget(self.pos)
        posrow.addWidget(cap)
        form.addRow('Position (x,y,z)', posrow)
        form.addRow('Orientation (qx,qy,qz,qw)', self.quat)
        self.named = QLineEdit()
        form.addRow('Named state (if target=named)', self.named)
        self.motion = QComboBox()
        self.motion.addItems(['ptp', 'lin', 'circ', 'free'])
        form.addRow('Motion', self.motion)
        self.move_planner = QComboBox()
        self.move_planner.addItems(['', 'pilz', 'ompl', 'ompl_chomp'])
        form.addRow('Planner (blank=global)', self.move_planner)
        self.speed = QSpinBox()
        self.speed.setRange(1, 100)
        self.speed.setValue(50)
        form.addRow('Speed %', self.speed)
        # per-waypoint start-state tolerance (rad); 0.0 disables the check. Bump it for
        # tight-space moves where the reported start state drifts (e.g. into the prewash).
        self.start_tol = QDoubleSpinBox()
        self.start_tol.setRange(0.0, 3.1416)
        self.start_tol.setSingleStep(0.01)
        self.start_tol.setDecimals(3)
        self.start_tol.setValue(0.1)
        form.addRow('Allowed start tol (rad)', self.start_tol)
        self.role = QComboBox()
        self.role.addItems(['generic', 'home', 'pre_pick', 'pick', 'post_pick',
                            'pre_place', 'place', 'post_place'])
        form.addRow('Role', self.role)
        # per-move planning-collision check for GRASPED objects: ON before a transfer
        # that must route the held payload around the static meshes (e.g. into the
        # prewash); OFF near the pick (payload on the belt); inherit = leave as-is.
        self.attached_check = QComboBox()
        self.attached_check.addItems(['inherit', 'on', 'off'])
        form.addRow('Attached collision check', self.attached_check)
        # CIRC only: an auxiliary point (a point on the arc, or the circle centre)
        self.aux = QLineEdit()
        self.aux.setPlaceholderText('circ only: x, y, z')
        form.addRow('Aux point (circ)', self.aux)
        self.aux_is_center = QCheckBox('aux is the circle centre (else a point on the arc)')
        form.addRow('', self.aux_is_center)
        layout.addLayout(form)
        # tool actions to fire AT this move
        self.tool_cbs = {k: QCheckBox(k) for k in ('grasp', 'release', 'attach', 'detach')}
        trow = QHBoxLayout()
        for cb in self.tool_cbs.values():
            trow.addWidget(cb)
        self.payload_ref = QLineEdit('cube')
        trow.addWidget(QLabel('payload:'))
        trow.addWidget(self.payload_ref)
        layout.addLayout(trow)
        addbtn = QPushButton('Add move to sequence')
        addbtn.clicked.connect(self.add_move)
        layout.addWidget(addbtn)

    def capture_pose(self):  # pragma: no cover - needs a live ROS session
        s = self.ctrl.robot_summary()
        try:
            pos, quat = self.wizard().live_capture().current_pose(
                s['base_frame'], s['tip_link'])
        except Exception as exc:  # noqa: BLE001
            self.pos.setText(f'# capture failed: {exc}')
            return
        self.pos.setText(', '.join(f'{v:.4f}' for v in pos))
        self.quat.setText(', '.join(f'{v:.4f}' for v in quat))
        self.target_mode.setCurrentText('tcp')

    def add_move(self):
        name = self.move_name.text().strip()
        if not name:
            return
        acc = {'inherit': None, 'on': True, 'off': False}[self.attached_check.currentText()]
        kwargs = dict(name=name, motion=self.motion.currentText(),
                      planner=self.move_planner.currentText() or None,
                      speed=self.speed.value(), role=self.role.currentText(),
                      allowed_start_tolerance=self.start_tol.value(),
                      attached_collision_check=acc)
        try:
            if self.target_mode.currentText() == 'named':
                kwargs['named'] = self.named.text().strip()
            else:
                kwargs['position'] = _parse_floats(self.pos.text())
                kwargs['orientation'] = _parse_floats(self.quat.text())
            if self.motion.currentText() == 'circ' and self.aux.text().strip():
                kwargs['aux'] = _parse_floats(self.aux.text())
                kwargs['aux_is_center'] = self.aux_is_center.isChecked()
        except ValueError:
            return
        if self.motion.currentText() == 'circ' and not kwargs.get('aux'):
            self.seq.addItem(f'(skipped {name}: CIRC needs an aux point)')
            return
        self.ctrl.add_move(**kwargs)
        for kind, cb in self.tool_cbs.items():
            if cb.isChecked():
                ref = self.payload_ref.text().strip() or None if kind in ('attach', 'detach') else None
                self.ctrl.add_tool_action(name, kind, ref)
        self._refresh()

    def _refresh(self):
        self.seq.clear()
        for n in self.ctrl.project.application.sequence:
            self.seq.addItem(n)

    def validatePage(self) -> bool:
        return True


class BaseConfigPage(QWizardPage):
    """MVP entry (Steps 1-2): start a project + load the hand-made BASE moveit_config,
    which auto-configures the robot (group/frames/named-states/gripper/controllers) and
    switches generation to the standalone copy-base flow."""

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 1 — Robot & base config')
        self.setSubTitle('Load your hand-made base moveit_config: robot, SRDF waypoints, '
                         'controllers and the mock/isaac/real bridges are read from it.')
        form = QFormLayout(self)
        self.project_file = QLineEdit()
        pbrowse = QPushButton('Browse…')
        pbrowse.clicked.connect(self._browse_project)
        prow = QHBoxLayout(); prow.addWidget(self.project_file); prow.addWidget(pbrowse)
        form.addRow('Open project.yaml', prow)
        self.project_name = QLineEdit('big1500')
        form.addRow('— or new — Project name', self.project_name)
        self.base_pkg = QLineEdit('fr30_eef_moveit_config')
        form.addRow('Base moveit_config package', self.base_pkg)
        self.base_path = QLineEdit()
        bbrowse = QPushButton('Browse…')
        bbrowse.clicked.connect(self._browse_base)
        brow = QHBoxLayout(); brow.addWidget(self.base_path); brow.addWidget(bbrowse)
        form.addRow('Base package path', brow)
        self.summary = QLabel('')
        self.summary.setWordWrap(True)
        form.addRow('Loaded', self.summary)

    def _browse_project(self):  # pragma: no cover - needs a display
        path, _ = QFileDialog.getOpenFileName(self, 'Open project.yaml', '', 'project (*.yaml *.yml)')
        if path:
            self.project_file.setText(path)

    def _browse_base(self):  # pragma: no cover - needs a display
        path = QFileDialog.getExistingDirectory(self, 'Base moveit_config package dir')
        if path:
            self.base_path.setText(path)

    def validatePage(self) -> bool:
        try:
            if self.project_file.text().strip():
                self.ctrl.open_project(_expand_path(self.project_file.text()))
            else:
                self.ctrl.new_blank_project(self.project_name.text().strip() or 'robot_app')
                info = self.ctrl.load_base_moveit_config(
                    self.base_pkg.text().strip(), _expand_path(self.base_path.text()))
                self.summary.setText(
                    f"{info['robot_name']}: group={info['group']}, "
                    f"named states={len(info['named_states'])}, "
                    f"arm={info['arm_controller']}, gripper={info['gripper_controller']}")
                return True
        except Exception as exc:  # noqa: BLE001
            self.summary.setText(f'ERROR: {exc}')
            return False
        s = self.ctrl.robot_summary()
        self.summary.setText(f"opened project: {s['robot_name']} group={s['group']}")
        return True


class GenerateConfigPage(QWizardPage):
    """Step 3-4: generate the intermediate scene+planner config (to configure against a
    faithful RViz) and show the build snippet."""

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 3 — Generate scene+planner config')
        self.setSubTitle('Emit <robot>_scene_loader_moveit_config (base + planners + '
                         'scene.yaml), then build it. Configure the application against it.')
        form = QFormLayout(self)
        self.out_dir = QLineEdit()
        browse = QPushButton('Browse…')
        browse.clicked.connect(self._browse)
        row = QHBoxLayout(); row.addWidget(self.out_dir); row.addWidget(browse)
        form.addRow('Output dir (your src/)', row)
        self.pkg_name = QLineEdit()
        form.addRow('Config package name', self.pkg_name)
        gen = QPushButton('Generate scene+planner config')
        gen.clicked.connect(self.generate)
        form.addRow(gen)
        self.result = QTextEdit(); self.result.setReadOnly(True)
        form.addRow('Result', self.result)

    def initializePage(self):
        try:
            self.pkg_name.setText(self.ctrl.scene_loader_package_name())
        except Exception:  # noqa: BLE001
            pass

    def _browse(self):  # pragma: no cover - needs a display
        path = QFileDialog.getExistingDirectory(self, 'Output directory (src/)')
        if path:
            self.out_dir.setText(path)

    def generate(self):
        out = _expand_path(self.out_dir.text())
        pkg = self.pkg_name.text().strip() or None
        if not out:
            self.result.setPlainText('ERROR: set an output directory (your ros2_ws/src)')
            return
        # Overwrite guard: the default name equals a hand-made package would-be name, so a
        # blind Generate could clobber an existing package. Require an explicit re-click.
        target = os.path.join(out, pkg or self.ctrl.scene_loader_package_name())
        if os.path.isdir(target) and getattr(self, '_confirm_overwrite', None) != target:
            self._confirm_overwrite = target
            self.result.setPlainText(
                f'⚠ "{os.path.basename(target)}" ALREADY EXISTS at {out}.\n'
                f'Generating will OVERWRITE it. If that is intended, click "Generate" '
                f'again. Otherwise change the "Config package name" first.')
            return
        self._confirm_overwrite = None
        try:
            manifest = self.ctrl.generate_scene_loader_config(out, pkg)
        except Exception as exc:  # noqa: BLE001
            self.result.setPlainText(f'ERROR: {exc}')
            return
        pkg = pkg or self.ctrl.scene_loader_package_name()
        snippet = self.ctrl.build_snippet(pkg)   # ws_root derived at generate time
        lines = [f"Generated {manifest.as_dict()['file_count']} files -> {out}/{pkg}", '',
                 'Build it, then bring it up to configure the application:', snippet]
        for w in manifest.as_dict().get('warnings', []):
            lines.append(f'  warn: {w}')
        self.result.setPlainText('\n'.join(lines))

    def validatePage(self) -> bool:
        return True


class ModeBringupPage(QWizardPage):
    """Step 5-6: choose the mode and show the guided bring-up procedure."""

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 4 — Mode & bring-up (for configuration)')
        self.setSubTitle('Pick how you will run, then follow the procedure to BUILD + bring '
                         'up the scene+planner config so you can configure the application '
                         'against RViz. (The application itself is launched only at the end.)')
        form = QFormLayout(self)
        self.mode = QComboBox()
        self.mode.addItems(['isaac', 'mock', 'real'])
        self.mode.currentTextChanged.connect(self._refresh)
        form.addRow('Mode', self.mode)
        self.usd_path = QLineEdit()
        self.usd_path.setPlaceholderText('isaac: path to your cell .usd (optional)')
        form.addRow('USD scene (isaac)', self.usd_path)
        self.procedure = QTextEdit(); self.procedure.setReadOnly(True)
        form.addRow('Procedure', self.procedure)

    def initializePage(self):
        self._refresh()

    def _refresh(self, *_):
        cfg = self.ctrl.scene_loader_pkg      # the name set at Step 3
        if not cfg:
            try:
                cfg = self.ctrl.scene_loader_package_name()
            except Exception:  # noqa: BLE001
                cfg = '<the config you generated at Step 3>'
        proc = self.ctrl.bringup_procedure(
            self.mode.currentText(), cfg, usd_path=self.usd_path.text().strip() or None)
        self.procedure.setPlainText(proc)

    def validatePage(self) -> bool:
        try:
            self.ctrl.set_mode(self.mode.currentText())
        except Exception:  # noqa: BLE001
            pass
        return True


class SetupWizard(QWizard):
    def __init__(self, controller: Optional[AssistantController] = None):
        super().__init__()
        self.controller = controller or AssistantController()
        self._live = None
        self.setWindowTitle('TrainIt Setup Assistant')
        # MVP entry: load the hand-made base config (auto-configures the robot).
        self.base_page = BaseConfigPage(self.controller)
        self.gen_config_page = GenerateConfigPage(self.controller)
        self.mode_page = ModeBringupPage(self.controller)
        self.scene_page = ScenePage(self.controller)
        self.application_page = ApplicationPage(self.controller)
        self.waypoints_page = WaypointsPage(self.controller)
        self.generate_page = GeneratePage(self.controller)
        # From-scratch / advanced pages: constructed (used by the CLI + tests, and as
        # manual overrides of the base-derived values), not in the default MVP flow.
        self.load_page = LoadRobotPage(self.controller)
        self.group_page = GroupFramesPage(self.controller)
        self.controllers_page = ControllersPage(self.controller)
        self.states_page = NamedStatesPage(self.controller)
        # The two-phase MVP flow (Phase A: build the faithful env; Phase B: the app).
        for page in (self.base_page, self.scene_page, self.gen_config_page,
                     self.mode_page, self.application_page, self.waypoints_page,
                     self.generate_page):
            self.addPage(page)
        self._scene_pub = None

    def live_capture(self):  # pragma: no cover - needs a live ROS session
        """Lazily create the shared LiveCapture reader for the running session."""
        if self._live is None:
            from ..livesession import LiveCapture
            self._live = LiveCapture()
        return self._live

    def planning_scene(self):  # pragma: no cover - needs a live ROS session
        """Lazily create the planning-scene publisher for live preview."""
        if self._scene_pub is None:
            from ..livesession import PlanningScenePublisher
            self._scene_pub = PlanningScenePublisher()
        return self._scene_pub

    def closeEvent(self, event):  # pragma: no cover - GUI lifecycle
        if self._live is not None:
            self._live.close()
        if self._scene_pub is not None:
            self._scene_pub.close()
        super().closeEvent(event)
