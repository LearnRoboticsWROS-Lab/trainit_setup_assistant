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

from python_qt_binding.QtCore import Qt, QTimer
from python_qt_binding.QtGui import QBrush, QColor, QImage, QPixmap
from python_qt_binding.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QWizard,
    QWizardPage,
)

from ..model.enums import (EndEffectorActuation, GripperJointTarget,
                          SimGraspAdapter)
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
        self.setTitle('Step 8 — Generate the bundle')
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
        # Bundle prefix = the ROBOT name -> <robot>_description / _trainit_config / _app.
        robot = self.ctrl.robot_summary()['robot_name']
        self.project_name.setText(robot)
        # Default output: NEXT TO the base config package (generate() adds the
        # <name>_bundle/ folder itself), so the bundle lands in the colcon src tree.
        base = self.ctrl.project.robot.base_moveit_config_path
        if base and not self.output_dir.text().strip():
            self.output_dir.setText(os.path.dirname(str(base)))

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
        # the bundle is ONE folder holding the three packages (colcon finds nested
        # packages, so <src>/<name>_bundle/{_description,_trainit_config,_app} builds)
        bundle_dirname = f'{name}_bundle' if name else 'bundle'
        if os.path.basename(os.path.normpath(out_dir)) != bundle_dirname:
            out_dir = os.path.join(out_dir, bundle_dirname)
        try:
            manifest = self.ctrl.generate(out_dir)
        except Exception as exc:  # noqa: BLE001
            import traceback
            self.result.setPlainText(f'ERROR: {exc}\n\n{traceback.format_exc()}')
            return
        info = manifest.as_dict()
        b = self.ctrl.project.bundle
        pkgs = f'{b.description_package} {b.moveit_config_package} {b.app_package}'
        ws = self.ctrl._workspace_root(out_dir)
        lines = [
            f"Generated {info['file_count']} files into {out_dir}",
            f'Bundle: {b.description_package} + {b.moveit_config_package} + {b.app_package}',
            '',
            'Build & run:',
            f'  cd {ws}',
            f'  colcon build --packages-select {pkgs}',
            '  source install/setup.bash',
            f'  # robot alone:   ros2 launch {b.description_package} view_robot.launch.py',
            f'  # cell + RViz:   ros2 launch {b.moveit_config_package} bringup.launch.py '
            f'mode:={self.ctrl.project.deployment.default_mode}',
            f'  # application:  ros2 launch {b.app_package} bringup.launch.py '
            f'mode:={self.ctrl.project.deployment.default_mode}',
        ]
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
        # The two ROS contract topics scene_manager_node needs. They are NOT derivable
        # from the base moveit_config (they live as declare_parameter defaults inside the
        # cell bridge's own source), but the USD ActionGraph names the gripper one — so
        # "Load USD" fills the combo with what it found. Editable: the user always wins.
        self.gripper_topic = QComboBox()
        self.gripper_topic.setEditable(True)
        self.gripper_topic.setToolTip(
            'Bool, true = gripper CLOSED. scene_manager_node attaches the grasp targets '
            'when this fires. A wrong name fails SILENTLY: the payload never attaches.')
        usd_form.addRow('Gripper cmd topic', self.gripper_topic)
        self.reset_topic = QLineEdit()
        self.reset_topic.setToolTip(
            'Bool, latched. scene_manager_node publishes here on ~/reset_scene so the '
            'hand-authored Isaac adapter can teleport its dynamic prims home.')
        usd_form.addRow('Scene reset topic', self.reset_topic)
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
        # how a GRASPED object is represented while held:
        #   attach_box = DEFAULT. Attaches to the tool as a universal AABB cuboid (one rule
        #                for ANY shape -> scalable for a client) that fits the mesh (origin
        #                offset handled). Payload-aware planning (attached_collision_check),
        #                moves with the EE, robust in RViz. The full mesh stays in the scene
        #                until grasp; the real object is simulated in Isaac.
        #   remove     = debug fallback: disappears on grasp, reappears at the tool on release
        #                (NO payload collision-awareness during the transfer).
        grow = QHBoxLayout()
        grow.addWidget(QLabel('Grasp handling:'))
        self.grasp_mode = QComboBox()
        self.grasp_mode.addItems(['attach_box', 'remove'])
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
        # The ActionGraph named the gripper signal — offer it, and show what got applied.
        for t in (getattr(self.ctrl, 'usd_bool_topics', None) or []):
            if self.gripper_topic.findText(t) < 0:
                self.gripper_topic.addItem(t)
        self._prefill_topics()
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
        # append the gripper-topic outcome to whichever status line was just written
        if not getattr(self.ctrl, 'usd_base_prim_ok', True):
            tried = getattr(self.ctrl, 'usd_base_prim_tried', '?')
            self.status.setText(
                f'STOP — the robot was NOT found at "{tried}". Object poses are in STAGE '
                f'coordinates (wrong by the robot mount height) and the robot itself is '
                f'listed as a scene object. The path comes from the SRDF of the base '
                f'moveit_config loaded at Step 1 — go Back and check you pointed at the '
                f'RIGHT package.')
            return
        sniffed = getattr(self.ctrl, 'usd_bool_topics', None) or []
        self.status.setText(self.status.text() + (
            f'  ·  gripper topic from USD: {", ".join(sniffed)}' if sniffed else
            '  ·  ⚠ no std_msgs/Bool subscriber in the USD — set the Gripper cmd topic '
            'by hand or the payload never attaches.'))

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
            self.ctrl.set_scene_loader_params(
                grasp_attach_mode=self.grasp_mode.currentText(),
                gripper_cmd_topic=self.gripper_topic.currentText(),
                scene_reset_topic=self.reset_topic.text())
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

    def _prefill_topics(self):
        """Show what the project currently holds. Guarded: self.ctrl.project is None on a
        fresh wizard, and initializePage must not raise."""
        try:
            sc = self.ctrl.project.scene
        except Exception:  # noqa: BLE001
            return
        cur = sc.gripper_cmd_topic
        if self.gripper_topic.findText(cur) < 0:
            self.gripper_topic.addItem(cur)
        self.gripper_topic.setCurrentText(cur)
        self.reset_topic.setText(sc.scene_reset_topic)

    def initializePage(self):
        self._prefill_topics()
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


class PerceptionPage(QWizardPage):
    """Step 5 — Perception: WHAT the camera sees, HOW, and what to extract (D-015).

    Detectors are NAMED RESOURCES configured here once; the application step only
    BINDS them to waypoints. The live tuner runs trainit_perception's pure
    ``detect()`` on frames from the running bring-up — the SAME function the
    generated bundle runs, so what you tune is what you deploy.
    """

    # method token, label, enabled. Greyed = roadmap: shape mask (OpenCV contours),
    # PCL 3D cluster (an external C++ node behind the same contract), custom node.
    METHODS = [('color_mask', 'Colour mask (HSV)', True),
               ('shape_mask', 'Shape mask (contours)', False),
               ('pcl_cluster', 'PCL 3D cluster (C++ node)', False),
               ('custom', 'Custom node (expert)', False)]

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 5 — Perception (camera & detectors)')
        self.setSubTitle('Configure what the camera must SEE and what to EXTRACT: pick a '
                         'method, tune it LIVE against the running bring-up (Step 4), and '
                         'save it as a named detector. The application step then binds '
                         'detectors to waypoints. Skip this step for a blind application.')
        self._detectors: Dict[str, dict] = {}      # name -> DetectorSpec-shaped dict
        self._deleted: set = set()                 # names removed here, to reconcile
        self._last_detection = None
        self._tuner_topics = None                  # snapshot while the tuner runs
        root = QHBoxLayout(self)

        # ---- left: detector list + camera + timing ---------------------------
        left = QVBoxLayout()
        left.addWidget(QLabel('<b>Detectors</b>'))
        self.det_list = QListWidget()
        self.det_list.currentTextChanged.connect(self._on_pick_detector)
        left.addWidget(self.det_list, 1)
        addrow = QHBoxLayout()
        self.det_name = QLineEdit()
        self.det_name.setPlaceholderText('name (e.g. cube)')
        addb = QPushButton('Add')
        addb.clicked.connect(self._add_detector)
        rmb = QPushButton('Remove')
        rmb.clicked.connect(self._remove_detector)
        addrow.addWidget(self.det_name)
        addrow.addWidget(addb)
        addrow.addWidget(rmb)
        left.addLayout(addrow)

        cam = QGroupBox('Camera')
        cf = QFormLayout(cam)
        self.c_rgb = QLineEdit('/camera/color/image_raw')
        self.c_depth = QLineEdit('/camera/depth/image_rect_raw')
        self.c_info = QLineEdit('/camera/color/camera_info')
        self.c_frame = QLineEdit('camera_color_optical_frame')
        self.c_link = QLineEdit('camera_link')
        cf.addRow('RGB topic', self.c_rgb)
        cf.addRow('Depth topic', self.c_depth)
        cf.addRow('CameraInfo', self.c_info)
        cf.addRow('Optical frame', self.c_frame)
        cf.addRow('Camera link', self.c_link)
        self.c_replace = QLineEdit()
        self.c_replace.setPlaceholderText('URDF include to replace (base model)')
        self.c_with = QLineEdit()
        self.c_with.setPlaceholderText('camera-variant include (with the camera)')
        cf.addRow('Base include', self.c_replace)
        cf.addRow('Camera include', self.c_with)
        self.c_cloud = QCheckBox('build the coloured cloud in sim (depth_image_proc; '
                                 'in real the driver publishes it)')
        self.c_cloud.setChecked(True)
        cf.addRow(self.c_cloud)
        sniff = QPushButton('Sniff live topics')
        sniff.setToolTip('Read the running ROS graph and fill the image/depth/'
                         'camera_info topics from what the camera actually publishes.')
        sniff.clicked.connect(self._sniff_topics)
        cf.addRow(sniff)
        left.addWidget(cam)

        timing = QGroupBox('Timing (cycle rules)')
        tf_ = QFormLayout(timing)
        self.t_settle = QSpinBox()
        self.t_settle.setRange(0, 60000)
        self.t_settle.setValue(1500)
        self.t_settle.setSuffix(' ms')
        self.t_settle.setToolTip('Settle barrier emitted AFTER a scene reset that '
                                 'precedes a detection in the loop: the reset returns '
                                 'before the camera has re-rendered the world '
                                 '(measured 0.4 s on the reference cell).')
        tf_.addRow('Settle after reset', self.t_settle)
        self.t_timeout = QSpinBox()
        self.t_timeout.setRange(100, 60000)
        self.t_timeout.setValue(3000)
        self.t_timeout.setSuffix(' ms')
        tf_.addRow('Detect timeout', self.t_timeout)
        left.addWidget(timing)
        root.addLayout(left, 3)

        # ---- right: method + params + live tuner -----------------------------
        right = QVBoxLayout()
        mrow = QFormLayout()
        self.method = QComboBox()
        for _, label, enabled in self.METHODS:
            self.method.addItem(label if enabled else f'{label}  (roadmap)')
        mmodel = self.method.model()
        for i, (_, _, enabled) in enumerate(self.METHODS):
            if not enabled:
                mmodel.item(i).setEnabled(False)
        mrow.addRow('Method', self.method)
        self.p_class = QLineEdit('object')
        mrow.addRow('Class id', self.p_class)
        right.addLayout(mrow)

        params = QGroupBox('Colour mask parameters (tuned live)')
        pf = QFormLayout(params)
        self.h_lo, r = self._slider_row(179, 170)
        pf.addRow('H low', r)
        self.h_hi, r = self._slider_row(179, 10)
        pf.addRow('H high (lo>hi wraps 0 = red)', r)
        self.s_lo, r = self._slider_row(255, 40)
        pf.addRow('S low', r)
        self.s_hi, r = self._slider_row(255, 255)
        pf.addRow('S high', r)
        self.v_lo, r = self._slider_row(255, 60)
        pf.addRow('V low', r)
        self.v_hi, r = self._slider_row(255, 255)
        pf.addRow('V high', r)
        nrow = QHBoxLayout()
        self.p_area = QSpinBox(); self.p_area.setRange(1, 100000); self.p_area.setValue(60)
        self.p_max = QSpinBox(); self.p_max.setRange(1, 20); self.p_max.setValue(1)
        self.p_morph = QSpinBox(); self.p_morph.setRange(0, 15); self.p_morph.setValue(3)
        self.p_win = QSpinBox(); self.p_win.setRange(1, 15); self.p_win.setValue(5)
        self.p_area.setToolTip('Blobs smaller than this many pixels are ignored '
                               '(noise floor).')
        self.p_max.setToolTip('How many objects to report at most, biggest first.')
        self.p_morph.setToolTip('Morphological open+close kernel (px): removes '
                                'speckle, closes small holes. 0 = off.')
        self.p_win.setToolTip('Median window (px) around the centroid when reading '
                              'Z from the depth image — rescues depth holes.')
        for lbl, w in (('min area px', self.p_area), ('max obj', self.p_max),
                       ('morph', self.p_morph), ('depth win', self.p_win)):
            nrow.addWidget(QLabel(lbl)); nrow.addWidget(w)
        pf.addRow(nrow)
        frow = QHBoxLayout()
        self.p_blur = QSpinBox(); self.p_blur.setRange(0, 31)
        self.p_blur.setToolTip('Gaussian blur (px, 0=off): melts single-pixel colour noise')
        self.p_zmin = QDoubleSpinBox(); self.p_zmin.setRange(0.0, 10.0)
        self.p_zmin.setDecimals(2); self.p_zmin.setSingleStep(0.05)
        self.p_zmax = QDoubleSpinBox(); self.p_zmax.setRange(0.0, 10.0)
        self.p_zmax.setDecimals(2); self.p_zmax.setSingleStep(0.05)
        self.p_zmax.setToolTip('Depth pass-through band [min, max) m; max 0 = off. '
                               'Cuts colour noise outside the working distance.')
        for lbl, w in (('blur px', self.p_blur), ('z min m', self.p_zmin),
                       ('z max m', self.p_zmax)):
            frow.addWidget(QLabel(lbl)); frow.addWidget(w)
        pf.addRow(frow)
        right.addWidget(params)

        trow = QHBoxLayout()
        self.p_cont = QCheckBox('continuous')
        self.p_cont.setChecked(True)
        self.p_cont.setToolTip('Publish at rate (what the generated tree consumes). '
                               'Off = publish only on the /perception/<name>/detect '
                               'Trigger service — NOTE: the generated tree does NOT '
                               'call it, so keep continuous for tree-driven apps.')
        self.p_rate = QDoubleSpinBox(); self.p_rate.setRange(0.1, 60.0)
        self.p_rate.setValue(10.0); self.p_rate.setSuffix(' Hz')
        trow.addWidget(self.p_cont)
        trow.addWidget(self.p_rate)
        save = QPushButton('Save detector')
        save.setToolTip('Store the tuned parameters under the selected name')
        save.clicked.connect(self._save_detector)
        trow.addWidget(save)
        right.addLayout(trow)

        tuner = QGroupBox('Live tuner (needs the Step-4 bring-up running)')
        tv = QVBoxLayout(tuner)
        imgrow = QHBoxLayout()
        self.img_view = QLabel('camera')
        self.mask_view = QLabel('mask')
        for v in (self.img_view, self.mask_view):
            v.setMinimumSize(240, 140)
            v.setAlignment(Qt.AlignCenter)
            v.setStyleSheet('background:#222; color:#888;')
        imgrow.addWidget(self.img_view)
        imgrow.addWidget(self.mask_view)
        tv.addLayout(imgrow)
        brow = QHBoxLayout()
        self.tune_btn = QPushButton('Start live tuning')
        self.tune_btn.setCheckable(True)
        self.tune_btn.toggled.connect(self._toggle_tuner)
        cap = QPushButton('Capture centroid')
        cap.setToolTip('Freeze the current detection and show its 3D point: pixel -> '
                       'metres via the intrinsics (X,Y), depth image at the centroid (Z), '
                       'then TF into the robot base frame')
        cap.clicked.connect(self._capture_centroid)
        brow.addWidget(self.tune_btn)
        brow.addWidget(cap)
        tv.addLayout(brow)
        self.tuner_status = QLabel('')
        self.tuner_status.setWordWrap(True)
        tv.addWidget(self.tuner_status)
        right.addWidget(tuner, 1)
        root.addLayout(right, 5)

        self._timer = QTimer(self)
        self._timer.setInterval(120)          # ~8 Hz, matching the runtime default
        self._timer.timeout.connect(self._tick)

    # ---- widgets helpers -----------------------------------------------------
    def _status(self, msg: str, ok: bool = False) -> None:
        self.tuner_status.setStyleSheet(
            'color:#2e7d32; font-weight:bold;' if ok else '')
        self.tuner_status.setText(('\u2713 ' if ok else '') + msg)

    def _slider_row(self, top: int, value: int):
        s = QSlider(Qt.Horizontal)
        s.setRange(0, top)
        s.setValue(value)
        val = QLabel(str(value))
        val.setMinimumWidth(30)
        s.valueChanged.connect(lambda v, lab=val: lab.setText(str(v)))
        row = QHBoxLayout()
        row.addWidget(s)
        row.addWidget(val)
        return s, row

    def _params_from_form(self) -> dict:
        p = {'class_id': self.p_class.text().strip() or 'object',
             'h': [self.h_lo.value(), self.h_hi.value()],
             's': [self.s_lo.value(), self.s_hi.value()],
             'v': [self.v_lo.value(), self.v_hi.value()],
             'min_area_px': self.p_area.value(),
             'max_objects': self.p_max.value(),
             'morph_kernel': self.p_morph.value(),
             'depth_window_px': self.p_win.value()}
        # noise filters land in the file only when ON, so an untuned profile stays
        # byte-stable with the runtime defaults
        if self.p_blur.value() > 0:
            p['blur_px'] = self.p_blur.value()
        if self.p_zmax.value() > 0:
            p['depth_min_m'] = round(self.p_zmin.value(), 3)
            p['depth_max_m'] = round(self.p_zmax.value(), 3)
        return p

    def _form_from_params(self, p: dict) -> None:
        self.p_class.setText(str(p.get('class_id', 'object')))
        h = p.get('h', [170, 10]); s = p.get('s', [40, 255]); v = p.get('v', [60, 255])
        self.h_lo.setValue(int(h[0])); self.h_hi.setValue(int(h[1]))
        self.s_lo.setValue(int(s[0])); self.s_hi.setValue(int(s[1]))
        self.v_lo.setValue(int(v[0])); self.v_hi.setValue(int(v[1]))
        self.p_area.setValue(int(p.get('min_area_px', 60)))
        self.p_max.setValue(int(p.get('max_objects', 1)))
        self.p_morph.setValue(int(p.get('morph_kernel', 3)))
        self.p_win.setValue(int(p.get('depth_window_px', 5)))
        self.p_blur.setValue(int(p.get('blur_px', 0)))
        self.p_zmin.setValue(float(p.get('depth_min_m', 0.0)))
        self.p_zmax.setValue(float(p.get('depth_max_m', 0.0)))

    # ---- detector list -------------------------------------------------------
    def _add_detector(self) -> None:
        name = self.det_name.text().strip()
        if not name:
            self._status('Give the detector a name first.')
            return
        import re
        if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', name):
            self._status(
                f'"{name}" is not a valid detector name: it becomes a ROS node and '
                'topic name — use letters, digits and underscores, starting with a '
                'letter (e.g. red_cube).')
            return
        self._deleted.discard(name)
        self._detectors[name] = {'method': 'color_mask',
                                 'params': self._params_from_form(),
                                 'continuous': self.p_cont.isChecked(),
                                 'rate_hz': self.p_rate.value()}
        self._refresh_list(select=name)

    def _save_detector(self) -> None:
        name = self.det_list.currentItem().text() if self.det_list.currentItem() else ''
        name = name or self.det_name.text().strip()
        if not name:
            self._status('Select (or name) a detector to save into.')
            return
        self._detectors[name] = {'method': 'color_mask',
                                 'params': self._params_from_form(),
                                 'continuous': self.p_cont.isChecked(),
                                 'rate_hz': self.p_rate.value()}
        self._refresh_list(select=name)
        self._status(f'saved detector "{name}" — it becomes perception.yaml entry, '
                     f'node detector_{name} and topic /perception/{name}/detections',
                     ok=True)

    def _remove_detector(self) -> None:
        item = self.det_list.currentItem()
        if item:
            self._detectors.pop(item.text(), None)
            self._deleted.add(item.text())    # reconciled into the project on Next
            self._refresh_list()

    def _on_pick_detector(self, name: str) -> None:
        d = self._detectors.get(name)
        if d:
            self._form_from_params(d['params'])
            self.p_cont.setChecked(bool(d.get('continuous', True)))
            self.p_rate.setValue(float(d.get('rate_hz', 10.0)))

    def _refresh_list(self, select: str = '') -> None:
        self.det_list.blockSignals(True)
        self.det_list.clear()
        for name in sorted(self._detectors):
            self.det_list.addItem(name)
        self.det_list.blockSignals(False)
        if select:
            hits = self.det_list.findItems(select, Qt.MatchExactly)
            if hits:
                self.det_list.setCurrentItem(hits[0])

    # ---- live tuner ----------------------------------------------------------
    def _toggle_tuner(self, on: bool):  # pragma: no cover - needs a live ROS session
        if on:
            # topic snapshot: editing a topic field mid-run must not churn the ROS
            # node once per keystroke — a change applies on the next explicit Start
            self._tuner_topics = (self.c_rgb.text().strip(),
                                  self.c_depth.text().strip(),
                                  self.c_info.text().strip())
            self.tune_btn.setText('Stop live tuning')
            self._timer.start()
        else:
            self.tune_btn.setText('Start live tuning')
            self._timer.stop()

    def hideEvent(self, event):  # pragma: no cover - GUI lifecycle
        # Back/Next/close: never leave the tuner ticking on a hidden page
        self.tune_btn.setChecked(False)
        super().hideEvent(event)

    def _tick(self):  # pragma: no cover - needs a live ROS session
        try:
            cap = self.wizard().camera_capture(*(self._tuner_topics or
                                                 ('/camera/color/image_raw',
                                                  '/camera/depth/image_rect_raw',
                                                  '/camera/color/camera_info')))
        except Exception as exc:  # noqa: BLE001
            self._status(f'camera capture failed: {exc}')
            self.tune_btn.setChecked(False)
            return
        frame = cap.latest()
        if frame is None:
            err = cap.last_error()
            self._status(
                f'frames arriving but undecodable: {err}' if err else
                'waiting for frames… (is the Step-4 bring-up running, with the '
                'camera publishing?)')
            return
        rgb, depth_m, K, frame_id = frame
        self.tuner_status.setStyleSheet('')
        try:
            from trainit_perception.detectors import make_detector
            det = make_detector('color_mask', self._params_from_form())
            detections = det.detect(rgb, depth_m, K)
            mask = det.debug_mask(rgb, depth_m)
            self._paint(self.img_view, rgb)
            if mask is not None:
                self._paint(self.mask_view, mask)
        except Exception as exc:  # noqa: BLE001
            self._status(f'detector failed: {exc}')
            self.tune_btn.setChecked(False)
            return
        if detections:
            d = detections[0]
            self._last_detection = (d, frame_id)
            x, y, z = d.position
            self.tuner_status.setText(
                f'{len(detections)} object(s) — best "{d.class_id}" score {d.score:.2f}'
                f'  px ({d.pixel[0]:.1f}, {d.pixel[1]:.1f})  area {d.area_px} px'
                f'  optical ({x:.4f}, {y:.4f}, {z:.4f}) m'
                f'  size ({d.size[0] * 1000:.1f} × {d.size[1] * 1000:.1f}) mm')
        else:
            self._last_detection = None
            self._status('no detection — widen the HSV window or lower '
                         'the min area')

    def _paint(self, view: QLabel, arr) -> None:  # pragma: no cover - display only
        import numpy as np
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        arr = np.ascontiguousarray(arr)
        h, w, _ = arr.shape
        img = QImage(arr.data, w, h, 3 * w, QImage.Format_RGB888)
        view.setPixmap(QPixmap.fromImage(img).scaled(
            view.width(), view.height(), Qt.KeepAspectRatio))

    def _capture_centroid(self):  # pragma: no cover - needs a live ROS session
        if self._last_detection is None:
            self._status('no detection to capture — start the tuner first')
            return
        d, frame_id = self._last_detection
        base = self.ctrl.robot_summary()['base_frame']
        optical = frame_id or self.c_frame.text().strip()
        try:
            import numpy as np
            pos, quat = self.wizard().live_capture().current_pose(base, optical)
            qx, qy, qz, qw = quat
            R = np.array([
                [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
                [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
                [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)]])
            p = np.array(pos) + R @ np.array(d.position)
            self._status(
                f'CAPTURED centroid: optical ({d.position[0]:.4f}, {d.position[1]:.4f}, '
                f'{d.position[2]:.4f}) -> {base} ({p[0]:.4f}, {p[1]:.4f}, {p[2]:.4f}) m. '
                f'Pick this detector on a Move block in Step 7 (Camera guidance).',
                ok=True)
        except Exception as exc:  # noqa: BLE001
            self._status(f'TF {base} -> {optical} failed: {exc}')

    def _sniff_topics(self):  # pragma: no cover - needs a live ROS session
        try:
            from ..livesession.camera_capture import sniff_camera_topics
            found = sniff_camera_topics(timeout_s=0.8)
        except Exception as exc:  # noqa: BLE001
            self._status(f'sniff failed: {exc}')
            return
        if not found:
            self._status('no image topics found — is the bring-up (or the camera) '
                         'publishing?')
            return
        for key, widget in (('rgb', self.c_rgb), ('depth', self.c_depth),
                            ('camera_info', self.c_info)):
            if found.get(key):
                widget.setText(found[key])
        self._status('topics filled from the live graph: '
                     + ', '.join(v for v in found.values() if v), ok=True)

    # ---- lifecycle -----------------------------------------------------------
    def initializePage(self) -> None:
        try:
            per = self.ctrl.project.perception
        except Exception:  # noqa: BLE001
            return
        if per is None:
            return
        if per.camera is not None:
            c = per.camera
            self.c_rgb.setText(c.rgb_topic)
            self.c_depth.setText(c.depth_topic)
            self.c_info.setText(c.camera_info_topic)
            self.c_frame.setText(c.optical_frame)
            self.c_link.setText(c.link)
            self.c_replace.setText(c.replace_include or '')
            self.c_with.setText(c.with_include or '')
            self.c_cloud.setChecked(c.synthetic_cloud_in_sim)
        self.t_settle.setValue(per.settle_ms)
        self.t_timeout.setValue(per.detect_timeout_ms)
        # merge by name: never resurrect a locally removed detector on back-navigation
        for d in per.detectors:
            if d.name not in self._detectors and d.name not in self._deleted:
                self._detectors[d.name] = {'method': d.method.value,
                                           'params': dict(d.params),
                                           'continuous': d.continuous,
                                           'rate_hz': d.rate_hz}
        self._refresh_list()

    def validatePage(self) -> bool:
        self.tune_btn.setChecked(False)        # stops the timer via the toggle handler
        project_has = bool(self.ctrl.detector_names()) if self.ctrl.project else False
        if not self._detectors and not project_has and not self._deleted:
            return True                        # blind flow: nothing to push
        # deletions first (also unbinds any waypoint using the removed detector)
        for name in set(self.ctrl.detector_names()) - set(self._detectors):
            self.ctrl.remove_detector(name)
        self._deleted.clear()
        self.ctrl.set_camera(
            rgb_topic=self.c_rgb.text().strip() or None,
            depth_topic=self.c_depth.text().strip() or None,
            camera_info_topic=self.c_info.text().strip() or None,
            optical_frame=self.c_frame.text().strip() or None,
            link=self.c_link.text().strip() or None,
            replace_include=self.c_replace.text().strip(),
            with_include=self.c_with.text().strip(),
            synthetic_cloud_in_sim=self.c_cloud.isChecked())
        self.ctrl.set_perception_timing(self.t_settle.value(), self.t_timeout.value())
        for name, d in self._detectors.items():
            try:
                self.ctrl.upsert_detector(name, method=d['method'], params=d['params'],
                                          continuous=d['continuous'],
                                          rate_hz=d['rate_hz'])
            except ValueError as exc:
                self._status(str(exc))
                return False
        return True


class ApplicationPage(QWizardPage):
    # Every application the TrainIt Motion Runtime can host. Tokens are frozen
    # (saved projects carry them); labels are the product names (D-015).
    APPS = [('pick_and_place', 'Blind pick and place', True),
            ('vision_guided_motion', 'Vision guided motion', True),
            ('gluing', 'Gluing', False), ('follow_path', 'Follow path', False),
            ('waypoint_replay', 'Waypoint replay', False), ('cnc', 'CNC tending', False)]

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 6 — Application type')
        self.setSubTitle('Pick the robotic application. "Vision guided motion" is the '
                         'editable sequence with vision blocks: waypoints, EEF trigger, '
                         'per-segment planner, and the detectors from Step 5 bound to '
                         'waypoints. Greyed entries are roadmap. Motion type + planner '
                         'are chosen PER WAYPOINT in the next step.')
        form = QFormLayout(self)
        self.app_type = QComboBox()
        for _, label, enabled in self.APPS:
            self.app_type.addItem(label if enabled else f'{label}  (coming soon)')
        model = self.app_type.model()          # grey out the not-yet-available apps
        for i, (_, _, enabled) in enumerate(self.APPS):
            if not enabled:
                model.item(i).setEnabled(False)
        form.addRow('Application', self.app_type)

    def initializePage(self) -> None:
        try:
            cur = self.ctrl.project.application.type.value
            per = self.ctrl.project.perception
        except Exception:  # noqa: BLE001
            return
        tokens = [t for t, _, _ in self.APPS]
        if getattr(self, '_user_chose', False):
            # re-entry: mirror what the user already chose, never override it
            self.app_type.setCurrentIndex(tokens.index(cur) if cur in tokens else 0)
            return
        # first entry: suggest Vision guided motion when detectors exist
        if per is not None and per.detectors:
            self.app_type.setCurrentIndex(1)

    def validatePage(self) -> bool:
        idx = max(0, self.app_type.currentIndex())
        token, _, enabled = self.APPS[idx]
        if not enabled:
            token = 'pick_and_place'           # guard: a disabled row cannot be chosen
        # planner is per-waypoint; keep a sensible fallback for segments that don't set one
        self.ctrl.set_application(token, 'ompl')
        self._user_chose = True
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
        capj = QPushButton('Capture joints (live)')
        capj.setToolTip('Blind mode: jog the robot with the RViz gizmo (Plan & Execute), '
                        'then capture — the CURRENT joints become a named state with '
                        'this move\'s name, targeted by this move.')
        capj.clicked.connect(self.capture_named)
        namedrow = QHBoxLayout()
        namedrow.addWidget(self.named)
        namedrow.addWidget(capj)
        form.addRow('Named state (if target=named)', namedrow)
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

    def capture_named(self):  # pragma: no cover - needs a live ROS session
        """Blind-mode: save the robot's CURRENT joints as a named state + target it.
        (Jog with the RViz gizmo, Plan & Execute, then capture.) The generator merges
        these captured states into the bundle's SRDF so they resolve at runtime."""
        name = self.move_name.text().strip() or self.named.text().strip()
        if not name:
            self.seq.addItem('(set a Move name first, then capture)')
            return
        joints = self.ctrl.robot_summary()['arm_joints']
        try:
            values = self.wizard().live_capture().current_joint_values(joints)
        except Exception as exc:  # noqa: BLE001
            self.named.setText(f'# capture failed: {exc}')
            return
        self.ctrl.add_named_state(name, values)
        self.named.setText(name)
        self.target_mode.setCurrentText('named')
        self.seq.addItem(f'(captured joints -> named state "{name}")')

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
                nj = len(info.get('arm_joints') or [])
                warn = '' if nj else '  ⚠ NO ARM JOINTS — captured named states would be EMPTY'
                self.summary.setText(
                    f"{info['robot_name']}: group={info['group']}, "
                    f"joints={nj}, "
                    f"named states={len(info['named_states'])}, "
                    f"arm={info['arm_controller']}, gripper={info['gripper_controller']}{warn}")
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


# ═══ Step 7 (v3 UX, was 6): block-based application editor ══════════════════════════
# Palette | Sequence | Inspector — the MoveIt-Pro-style trittico. Blocks belong to
# LAYERS: 1 robot motion (blue), 2 gripper/objects (green), 3 process (orange).
# Greyed palette entries are roadmap (vision, PLC, policy). Layer 4 (adapters per
# mock/isaac/real) is shown read-only in the Deployment dialog. The page FOLDS the
# block list into the existing canonical model (moves + tool actions + wait/loop),
# so bundle generation is untouched.

_BLOCK_META = {
    'move':    ('\U0001F9BE', '#1565c0', 'Move (robot)'),
    'gripper': ('✊',     '#2e7d32', 'Gripper'),
    'reset':   ('♻',     '#2e7d32', 'Reset scene'),
    'detect':  ('\U0001F4F7', '#6a1b9a', 'Detect'),
    'policy':  ('\U0001F9E0', '#00838f', 'Policy (learned)'),
    'wait':    ('⏱',     '#ef6c00', 'Wait'),
    'loop':    ('\U0001F501', '#ef6c00', 'Loop'),
}

# D-016: vision is a PROPERTY of the Move block (a detector dropdown on its
# inspector), not a block of its own — the sequence shows an automatic read-only
# "detect" row at cycle start instead. Policy labels <-> model orientation tokens:
_VORI_KEEP, _VORI_SAME, _VORI_ALIGN = 0, 1, 2


class DeploymentDialog(QDialog):
    """Layer 4, READ-ONLY: what is wired under the hood for mock / isaac / real."""

    def __init__(self, ctrl: AssistantController, parent=None):
        super().__init__(parent)
        self.setWindowTitle('Layer 4 — Deployment (read-only)')
        self.resize(680, 420)
        p = ctrl.project
        dep, robot = p.deployment, p.robot
        lines = [
            'Derived from the base moveit_config — editing comes in a later release.\n',
            f'Modes: {", ".join(dep.modes)}   (default: {dep.default_mode})',
            f'Arm controller: {robot.arm_controller.name} '
            f'(action: /{robot.arm_controller.name}/follow_joint_trajectory)',
            f'Gripper action: {robot.gripper.gripper_action_ns() or "-"}',
            f'Arm /joint_states remap: {dep.arm_joint_states_remap_to or "-"}',
            '',
            'Cell bridges / adapters:',
        ]
        for b in dep.bridges:
            modes = ",".join(b.modes) if getattr(b, 'modes', None) else 'mock,isaac'
            lines.append(f'  [{modes:16s}] {b.package} / {b.executable}')
        if dep.real_include:
            lines.append(f'  [real hardware  ] include {dep.real_include.package}'
                         f'/launch/{dep.real_include.launch_file}')
        text = QTextEdit()
        text.setReadOnly(True)
        text.setPlainText('\n'.join(lines))
        lay = QVBoxLayout(self)
        lay.addWidget(text)
        close = QPushButton('Close')
        close.clicked.connect(self.accept)
        lay.addWidget(close)


class EndEffectorDialog(QDialog):
    """How the end-effector GRIPS — the two ADR-0010 axes, for the WHOLE robot (one shared
    GripperSpec), opened from Step 7. The green *Gripper block* says WHEN the tool opens/closes;
    THIS dialog says HOW: the actuation (an on/off trigger, or a joint driven to open/closed
    targets) and, per target backend, which sim-physics trick makes a grasped object stick
    (Isaac SurfaceGripper / Gazebo LinkAttacher / none). Progressive disclosure keeps it minimal:
    a suction cell sees only the actuation combo + the adapter matrix."""

    _ACT = [('Trigger (on/off signal)', EndEffectorActuation.TRIGGER.value),
            ('Joint position (command a joint to open/closed)',
             EndEffectorActuation.JOINT_POSITION.value)]
    _TGT = [('SRDF named state (open / closed)', GripperJointTarget.SRDF_STATE.value),
            ('Explicit angle (capture live or type)', GripperJointTarget.ANGLE.value)]
    _ADAPTERS = [SimGraspAdapter.SURFACE_GRIPPER.value, SimGraspAdapter.LINK_ATTACHER.value,
                 SimGraspAdapter.NONE.value]

    def __init__(self, ctrl: AssistantController, wiz, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.wiz = wiz                       # for live_capture(); a QDialog has no .wizard()
        self.setWindowTitle('End-effector — how the gripper grips')
        self.resize(560, 620)
        grip = ctrl.project.robot.gripper
        root = QVBoxLayout(self)

        # header: what the base config already knows (read-only)
        head = QFormLayout()
        head.addRow('Kind', QLabel(f'{grip.kind.value}   (from the base config)'))
        head.addRow('Command joint', QLabel(f'{grip.command_joint or "-"}   (from the base config)'))
        root.addLayout(head)

        # axis 1 — actuation
        self.g_act = QComboBox()
        for label, tok in self._ACT:
            self.g_act.addItem(label, tok)
        self.g_act.setToolTip('Trigger = an on/off signal (suction, weld, on/off tool). '
                              'Joint position = drive a gripper joint to an open and a closed target.')
        act_form = QFormLayout()
        act_form.addRow('Actuation', self.g_act)
        root.addLayout(act_form)

        # axis 1 (cont.) — joint targets, shown only for joint_position
        self.g_jp = QGroupBox('Joint targets')
        jp = QVBoxLayout(self.g_jp)
        self.g_target = QComboBox()
        for label, tok in self._TGT:
            self.g_target.addItem(label, tok)
        tgt_form = QFormLayout()
        tgt_form.addRow('Targets by', self.g_target)
        jp.addLayout(tgt_form)
        # SRDF-state sub-panel
        self.g_srdf = QWidget()
        srdf = QFormLayout(self.g_srdf)
        srdf.setContentsMargins(0, 0, 0, 0)
        self.g_open_state = QComboBox()
        self.g_open_state.setEditable(True)
        self.g_closed_state = QComboBox()
        self.g_closed_state.setEditable(True)
        srdf.addRow('Open state', self.g_open_state)
        srdf.addRow('Closed state', self.g_closed_state)
        jp.addWidget(self.g_srdf)
        # explicit-angle sub-panel, each with a live-capture button
        self.g_ang = QWidget()
        ang = QFormLayout(self.g_ang)
        ang.setContentsMargins(0, 0, 0, 0)
        self.g_open_ang = QDoubleSpinBox()
        self.g_closed_ang = QDoubleSpinBox()
        for sb in (self.g_open_ang, self.g_closed_ang):
            sb.setRange(-6.2832, 6.2832)     # ±2π rad — covers any single gripper joint
            sb.setDecimals(4)
            sb.setSingleStep(0.01)
            sb.setSuffix(' rad')
        cap_open = QPushButton('Capture current angle (live)')
        cap_open.setToolTip('Jog the gripper (Plan & Execute in RViz), then capture the '
                            'current joint angle from /joint_states — like a Move block captures a pose.')
        cap_open.clicked.connect(self.capture_open_angle)
        cap_closed = QPushButton('Capture current angle (live)')
        cap_closed.clicked.connect(self.capture_closed_angle)
        orow = QHBoxLayout(); orow.addWidget(self.g_open_ang); orow.addWidget(cap_open)
        crow = QHBoxLayout(); crow.addWidget(self.g_closed_ang); crow.addWidget(cap_closed)
        ang.addRow('Open angle', orow)
        ang.addRow('Closed angle', crow)
        jp.addWidget(self.g_ang)
        root.addWidget(self.g_jp)

        # axis 2 — per-backend sim grasp adapter (ALWAYS shown: orthogonal to actuation)
        self.g_adapters = QGroupBox('Sim grasp adapter — per target backend')
        self.g_adapters.setToolTip('Which sim-physics trick makes a grasped object stick. '
                                   'Independent of the actuation: a suction gripper in Isaac still '
                                   'needs surface_gripper. real/mock = none (real physics / no sim).')
        adl = QFormLayout(self.g_adapters)
        self._adapter_combos: Dict[str, QComboBox] = {}
        for backend in ctrl.project.deployment.modes:
            tok = str(backend)
            combo = QComboBox()
            combo.addItems(self._ADAPTERS)
            current = grip.grasp_adapter_for(backend)
            combo.setCurrentText(current.value)
            implicit = tok not in grip.sim_grasp_adapter
            adl.addRow(f'{tok}{"  (default)" if implicit else ""}', combo)
            self._adapter_combos[tok] = combo
        root.addWidget(self.g_adapters)

        self.g_status = QLabel('')
        self.g_status.setWordWrap(True)
        root.addWidget(self.g_status)

        btns = QHBoxLayout()
        btns.addStretch(1)
        cancel = QPushButton('Cancel')
        cancel.clicked.connect(self.reject)
        save = QPushButton('Save')
        save.setDefault(True)
        save.clicked.connect(self.accept)
        btns.addWidget(cancel)
        btns.addWidget(save)
        root.addLayout(btns)

        self._fill(grip)
        # signals drive progressive disclosure — connected AFTER _fill so its setCurrentIndex
        # calls don't redundantly re-fire _reveal (_fill calls _reveal() itself once at the end).
        self.g_act.currentIndexChanged.connect(self._reveal)
        self.g_target.currentIndexChanged.connect(self._reveal)

    # ---- model -> widgets ----------------------------------------------------
    def _fill(self, grip) -> None:
        i = self.g_act.findData(grip.actuation.value)
        self.g_act.setCurrentIndex(i if i >= 0 else 0)
        j = self.g_target.findData(grip.joint_target.value)
        self.g_target.setCurrentIndex(j if j >= 0 else 0)
        # seed the editable SRDF-state combos with ALL the EEF group_state names parsed from
        # the base moveit_config's SRDF (open / closed / partial / ...), so the user picks from
        # what the cell actually declares; still editable to type another.
        states = self.ctrl.gripper_state_names()
        for combo, val in ((self.g_open_state, grip.open_state),
                           (self.g_closed_state, grip.closed_state)):
            combo.clear()
            for s in states:
                combo.addItem(s)
            if val and val not in states:
                combo.addItem(val)
            combo.setCurrentText(val or '')
        if grip.open_angle is not None:
            self.g_open_ang.setValue(grip.open_angle)
        if grip.closed_angle is not None:
            self.g_closed_ang.setValue(grip.closed_angle)
        self._reveal()

    def _reveal(self) -> None:
        is_joint = self.g_act.currentData() == EndEffectorActuation.JOINT_POSITION.value
        self.g_jp.setVisible(is_joint)
        by = self.g_target.currentData()
        self.g_srdf.setVisible(is_joint and by == GripperJointTarget.SRDF_STATE.value)
        self.g_ang.setVisible(is_joint and by == GripperJointTarget.ANGLE.value)

    # ---- live capture (mirrors BlocksPage.capture_joints) --------------------
    def _capture_angle(self, spinbox):  # pragma: no cover - needs a live ROS session
        joint = self.ctrl.robot_summary().get('gripper_joint')
        if not joint:
            self.g_status.setText('Set the gripper command_joint (Step 1 / base config) first, '
                                  'then capture.')
            return
        try:
            values = self.wiz.live_capture().current_joint_values([joint])
        except Exception as exc:  # noqa: BLE001
            self.g_status.setText(f'capture failed: {exc}')
            return
        if joint not in values:
            self.g_status.setText(f'joint "{joint}" not in /joint_states')
            return
        v = values[joint]
        spinbox.setValue(v)
        self.g_status.setText(f'captured {v:.4f} rad')

    def capture_open_angle(self):  # pragma: no cover - needs a live ROS session
        self._capture_angle(self.g_open_ang)

    def capture_closed_angle(self):  # pragma: no cover - needs a live ROS session
        self._capture_angle(self.g_closed_ang)

    # ---- widgets -> model (only through the controller setters) --------------
    def accept(self):
        by = self.g_target.currentData()
        is_angle = by == GripperJointTarget.ANGLE.value
        is_srdf = by == GripperJointTarget.SRDF_STATE.value
        # write only the branch the user is on: angles when ANGLE, state names when SRDF_STATE,
        # keeping the other branch's values untouched (symmetric keep/clear/set contract). Guard
        # like every other page's save so a future GripperSpec validator can't crash the slot.
        try:
            self.ctrl.set_gripper_actuation(
                actuation=self.g_act.currentData(),
                joint_target=by,
                open_angle=self.g_open_ang.value() if is_angle else '__keep__',
                closed_angle=self.g_closed_ang.value() if is_angle else '__keep__',
                open_state=self.g_open_state.currentText().strip() if is_srdf else '__keep__',
                closed_state=self.g_closed_state.currentText().strip() if is_srdf else '__keep__')
            self.ctrl.set_sim_grasp_adapter(
                {tok: combo.currentText() for tok, combo in self._adapter_combos.items()})
        except Exception as exc:  # noqa: BLE001 - keep the dialog open, report to the user
            self.g_status.setText(f'could not save: {exc}')
            return
        super().accept()


class BlocksPage(QWizardPage):
    """Step 6 — the application as ordered BLOCKS (palette | sequence | inspector)."""

    def __init__(self, ctrl: AssistantController):
        super().__init__()
        self.ctrl = ctrl
        self.setTitle('Step 7 — Application blocks (robot · gripper · vision · process)')
        self.setSubTitle('Drag blocks into the sequence and order them. Jog the robot with '
                         'the RViz gizmo, then Capture. Layers: blue=robot motion, '
                         'green=gripper, purple=vision (binds a Step-5 detector to '
                         'waypoints), orange=process. Greyed blocks are roadmap.')
        self.blocks: list = []          # ordered [{kind:..., ...}]
        self._selected: int = -1

        root = QHBoxLayout(self)

        # ---- left: palette ---------------------------------------------------
        pal = QVBoxLayout()
        pal.addWidget(QLabel('<b>Blocks</b>'))

        def pal_btn(kind: str, label: str) -> QPushButton:
            icon, color, _ = _BLOCK_META[kind]
            b = QPushButton(f'{icon}  {label}')
            b.setStyleSheet(f'text-align:left; color:{color}; font-weight:bold;')
            b.clicked.connect(lambda _=False, k=kind: self.add_block(k))
            return b

        pal.addWidget(QLabel('Layer 1 — Robot'))
        pal.addWidget(pal_btn('move', 'Move'))
        pal.addWidget(QLabel('Layer 2 — Gripper / objects'))
        pal.addWidget(pal_btn('gripper', 'Gripper'))
        # The Gripper BLOCK says WHEN the tool opens/closes; this button says HOW it grips
        # (actuation + per-backend sim grasp adapter) for the whole robot — one shared setting.
        eef_btn = QPushButton('⚙  End-effector (how it grips)…')
        eef_btn.setToolTip('Actuation (trigger vs joint position), open/closed joint targets, '
                           'and the per-backend sim grasp adapter — applies to the whole robot.')
        eef_btn.setStyleSheet('text-align:left;')
        eef_btn.clicked.connect(self._show_end_effector)
        pal.addWidget(eef_btn)
        pal.addWidget(pal_btn('reset', 'Reset scene (sim)'))
        pal.addWidget(QLabel('Layer 3 — Vision'))
        pal.addWidget(pal_btn('detect', 'Detect (sample the camera)'))
        note = QLabel('WHERE this block sits is WHEN the camera is sampled; the '
                      'Move blocks bound to its detector (Camera guidance) update '
                      'there. No Detect block = automatic at cycle start.')
        note.setWordWrap(True)
        note.setStyleSheet('color:#6a1b9a;')
        pal.addWidget(note)
        pal.addWidget(QLabel('Layer 4 — Process'))
        pal.addWidget(pal_btn('wait', 'Wait / delay'))
        pal.addWidget(pal_btn('loop', 'Loop sequence'))
        pal.addWidget(QLabel('Layer 5 — Physical AI'))
        pal.addWidget(pal_btn('policy', 'Policy (learned)'))
        pnote = QLabel('A trained policy produces the move into the NEXT Move block: '
                       'hybrid decides the target (deterministic runtime executes it), '
                       'pure drives the robot, residual corrects a nominal move. Load '
                       'its .pt + card; a robot-state check is added after.')
        pnote.setWordWrap(True)
        pnote.setStyleSheet('color:#00838f;')
        pal.addWidget(pnote)
        pal.addWidget(QLabel('Roadmap'))
        for soon in ('PLC trigger (in/out)', 'Modbus / TCP-IP'):
            g = QPushButton(f'⚪  {soon}')
            g.setEnabled(False)
            g.setToolTip('Coming soon — the UX is ready for it')
            g.setStyleSheet('text-align:left; color:#9e9e9e;')
            pal.addWidget(g)
        pal.addStretch(1)
        dep_btn = QPushButton('Layer 4 — Deployment…')
        dep_btn.setToolTip('What is wired for mock / isaac / real (read-only)')
        dep_btn.clicked.connect(self._show_deployment)
        pal.addWidget(dep_btn)
        root.addLayout(pal, 2)

        # ---- centre: the sequence -------------------------------------------
        mid = QVBoxLayout()
        mid.addWidget(QLabel('<b>Sequence</b> (drag to reorder)'))
        # automatic detect row (D-016): shows WHEN the camera fires without the
        # user managing a block — the detection is always emitted at cycle start.
        self.auto_detect = QLabel('')
        self.auto_detect.setWordWrap(True)
        self.auto_detect.setStyleSheet('color:#6a1b9a; font-weight:bold;')
        self.auto_detect.setVisible(False)
        mid.addWidget(self.auto_detect)
        self.seq = QListWidget()
        self.seq.setDragDropMode(QAbstractItemView.InternalMove)
        self.seq.currentRowChanged.connect(self._on_select)
        self.seq.model().rowsMoved.connect(self._on_reorder)
        mid.addWidget(self.seq, 1)
        rm = QPushButton('Remove selected block')
        rm.clicked.connect(self.remove_selected)
        mid.addWidget(rm)
        self.status = QLabel('')
        self.status.setWordWrap(True)
        mid.addWidget(self.status)
        root.addLayout(mid, 4)

        # ---- right: inspector ------------------------------------------------
        insp = QVBoxLayout()
        insp.addWidget(QLabel('<b>Block properties</b>'))
        self.stack = QStackedWidget()
        self.stack.addWidget(self._empty_form())     # 0
        self.stack.addWidget(self._move_form())      # 1
        self.stack.addWidget(self._gripper_form())   # 2
        self.stack.addWidget(self._wait_form())      # 3
        self.stack.addWidget(self._loop_form())      # 4
        self.stack.addWidget(self._reset_form())     # 5
        self.stack.addWidget(self._detect_form())    # 6
        self.stack.addWidget(self._policy_form())    # 7
        insp.addWidget(self.stack, 1)
        apply_btn = QPushButton('Apply to block')
        apply_btn.clicked.connect(self.apply_inspector)
        insp.addWidget(apply_btn)
        root.addLayout(insp, 3)

    # ---- inspector forms -----------------------------------------------------
    def _empty_form(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        v.addWidget(QLabel('Select a block, or add one from the palette.'))
        v.addStretch(1)
        return w

    def _move_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.m_name = QLineEdit()
        f.addRow('Name', self.m_name)
        self.m_target = QComboBox()
        self.m_target.addItems(['named (captured joints)', 'tcp (pose)'])
        f.addRow('Target', self.m_target)
        self.m_named = QLineEdit()
        capj = QPushButton('Capture joints (live)')
        capj.setToolTip('Jog with the RViz gizmo (Plan & Execute), then capture: the '
                        'CURRENT joints become a named state with this block\'s name.')
        capj.clicked.connect(self.capture_joints)
        r1 = QHBoxLayout(); r1.addWidget(self.m_named); r1.addWidget(capj)
        f.addRow('Named state', r1)
        self.m_pos = QLineEdit()
        capp = QPushButton('Capture pose (live)')
        capp.clicked.connect(self.capture_pose)
        r2 = QHBoxLayout(); r2.addWidget(self.m_pos); r2.addWidget(capp)
        f.addRow('Position x,y,z', r2)
        self.m_quat = QLineEdit('0, 0, 0, 1')
        f.addRow('Orientation qx,qy,qz,qw', self.m_quat)
        self.m_motion = QComboBox()
        self.m_motion.addItems(['ptp', 'lin', 'free', 'circ'])
        self.m_motion.setToolTip('ptp/free = joint-space (robust). lin/circ = CARTESIAN '
                                 'straight/arc: to a named state the pose is resolved via '
                                 'FK; the straight line itself must be feasible.')
        f.addRow('Motion', self.m_motion)
        self.m_planner = QComboBox()
        self.m_planner.addItems(['', 'pilz', 'ompl', 'ompl_chomp'])
        f.addRow('Planner (blank=global)', self.m_planner)
        self.m_speed = QSpinBox(); self.m_speed.setRange(1, 100); self.m_speed.setValue(50)
        f.addRow('Speed %', self.m_speed)
        self.m_tol = QDoubleSpinBox()
        self.m_tol.setRange(0.0, 3.1416); self.m_tol.setDecimals(3)
        self.m_tol.setSingleStep(0.01); self.m_tol.setValue(0.1)
        f.addRow('Start tol (rad)', self.m_tol)
        self.m_check = QComboBox()
        self.m_check.addItems(['inherit', 'on', 'off'])
        self.m_check.setToolTip('Held-payload collision check for THIS move: ON when the '
                                'carried objects must avoid the static meshes.')
        f.addRow('Attached collision check', self.m_check)

        # --- camera guidance (D-016): take what the detector sees, TF2 it into the
        # base frame, and the EEF goes there — this move's POSITION comes from the
        # detection at run time; the pose above stays as the recorded fallback.
        f.addRow(QLabel('<b>Camera guidance</b>'))
        self.m_detector = QComboBox()
        self.m_detector.setToolTip('The Step-5 detector that feeds this move. '
                                   '(none) = a normal taught waypoint.')
        f.addRow('Guided by camera', self.m_detector)
        offrow = QHBoxLayout()
        self.m_dx = QDoubleSpinBox(); self.m_dy = QDoubleSpinBox()
        self.m_dz = QDoubleSpinBox()
        for sb in (self.m_dx, self.m_dy, self.m_dz):
            sb.setRange(-1.0, 1.0); sb.setDecimals(3); sb.setSingleStep(0.005)
        self.m_dz.setToolTip('Offset above the DETECTED point (the visible top '
                             'face): the tool standoff. Base-frame axes.')
        for lbl, sb in (('dx', self.m_dx), ('dy', self.m_dy), ('dz', self.m_dz)):
            offrow.addWidget(QLabel(lbl)); offrow.addWidget(sb)
        f.addRow('Offset from detection (m)', offrow)
        self.m_vori = QComboBox()
        self.m_vori.addItems(['keep the taught orientation',
                              'same as waypoint…',
                              'align to detected object'])
        f.addRow('EEF orientation', self.m_vori)
        self.m_vori_ref = QComboBox()
        f.addRow('…same as', self.m_vori_ref)

        # --- relative to step (D-017): this move's pose is DERIVED at run time
        # from another step's final pose (vision included) + offsets — the retreat
        # case: post_pick = pick + 8 cm, wherever the camera sent pick, with no
        # second detection (the arm occludes the object at retreat time anyway).
        f.addRow(QLabel('<b>Relative to step</b>'))
        self.m_rel = QComboBox()
        self.m_rel.setToolTip('The step this move follows. (none) = not relative. '
                              'Mutually exclusive with camera guidance.')
        f.addRow('Relative to', self.m_rel)
        roff = QHBoxLayout()
        self.m_rdx = QDoubleSpinBox(); self.m_rdy = QDoubleSpinBox()
        self.m_rdz = QDoubleSpinBox()
        for sb in (self.m_rdx, self.m_rdy, self.m_rdz):
            sb.setRange(-1.0, 1.0); sb.setDecimals(3); sb.setSingleStep(0.005)
        for lbl, sb in (('dx', self.m_rdx), ('dy', self.m_rdy), ('dz', self.m_rdz)):
            roff.addWidget(QLabel(lbl)); roff.addWidget(sb)
        f.addRow('Offset from step (m)', roff)
        rang = QHBoxLayout()
        self.m_rroll = QDoubleSpinBox(); self.m_rpitch = QDoubleSpinBox()
        self.m_ryaw = QDoubleSpinBox()
        for sb in (self.m_rroll, self.m_rpitch, self.m_ryaw):
            sb.setRange(-180.0, 180.0); sb.setDecimals(1); sb.setSingleStep(5.0)
        for lbl, sb in (('Δroll', self.m_rroll), ('Δpitch', self.m_rpitch),
                        ('Δyaw', self.m_ryaw)):
            rang.addWidget(QLabel(lbl)); rang.addWidget(sb)
        rang_note = QHBoxLayout()
        f.addRow('Δ orientation (°)', rang)
        return w

    def _fill_move_vision(self, b: dict) -> None:
        """Populate the Camera-guidance widgets for the selected move block."""
        names = []
        try:
            names = self.ctrl.detector_names()
        except Exception:  # noqa: BLE001
            pass
        self.m_detector.blockSignals(True)
        self.m_detector.clear()
        self.m_detector.addItem('(none)')
        self.m_detector.addItems(names)
        want = b.get('detector', '')
        self.m_detector.setCurrentText(want if want in names else '(none)')
        self.m_detector.blockSignals(False)
        self.m_dx.setValue(float(b.get('vdx', 0.0)))
        self.m_dy.setValue(float(b.get('vdy', 0.0)))
        self.m_dz.setValue(float(b.get('vdz', 0.0)))
        moves = [x['name'] for x in self.blocks
                 if x['kind'] == 'move' and x['name'] != b.get('name')]
        self.m_vori_ref.blockSignals(True)
        self.m_vori_ref.clear()
        self.m_vori_ref.addItems(moves or [''])
        ref = b.get('vori_ref', '')
        if ref and ref not in moves:
            # dangling reference (renamed/removed move): keep it selectable so an
            # untouched Apply round-trips instead of silently re-pointing it
            self.m_vori_ref.addItem(ref)
            self.status.setText(f'orientation references missing waypoint "{ref}" '
                                '— fix or re-pick it')
        if ref:
            self.m_vori_ref.setCurrentText(ref)
        self.m_vori_ref.blockSignals(False)
        self.m_vori.setCurrentIndex({'keep': _VORI_KEEP, 'same': _VORI_SAME,
                                     'align': _VORI_ALIGN}.get(b.get('vori', 'keep'),
                                                               _VORI_KEEP))
        self._filled_vori = self.m_vori.currentIndex()
        # "align to detected object" only when the detector gives an orientation
        # (a colour mask / 3D-only detector publishes identity — object-relative
        # grasping is the future learned-policy path, D-016)
        gives = False
        try:
            per = self.ctrl.project.perception
            d = per.detector_by_name(want) if per else None
            gives = bool(d and d.gives_orientation)
        except Exception:  # noqa: BLE001
            pass
        item = self.m_vori.model().item(_VORI_ALIGN)
        item.setEnabled(gives)
        item.setToolTip('' if gives else
                        'This detector publishes no object orientation (3D-only). '
                        'Object-relative grasping will come from a learned grasp '
                        'policy (roadmap).')
        # relative-to-step widgets (D-017) + mutual exclusion with camera guidance
        rel_step = b.get('rel_step', '')
        self.m_rel.blockSignals(True)
        self.m_rel.clear()
        self.m_rel.addItem('(none)')
        self.m_rel.addItems(moves)
        if rel_step and rel_step not in moves:
            self.m_rel.addItem(rel_step)      # dangling ref stays selectable
            self.status.setText(f'relative step "{rel_step}" no longer exists — '
                                'fix or re-pick it')
        self.m_rel.setCurrentText(rel_step if rel_step else '(none)')
        self.m_rel.blockSignals(False)
        self.m_rdx.setValue(float(b.get('rdx', 0.0)))
        self.m_rdy.setValue(float(b.get('rdy', 0.0)))
        self.m_rdz.setValue(float(b.get('rdz', 0.0)))
        self.m_rroll.setValue(float(b.get('rroll', 0.0)))
        self.m_rpitch.setValue(float(b.get('rpitch', 0.0)))
        self.m_ryaw.setValue(float(b.get('ryaw', 0.0)))
        # one master per move: camera or step, not both — the other side greys out
        self.m_rel.setEnabled(not b.get('detector'))
        self.m_detector.setEnabled(not rel_step)

    def _gripper_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.g_action = QComboBox()
        self.g_action.addItems(['close (grasp)', 'open (release)',
                                'attach payload', 'detach payload'])
        f.addRow('Action', self.g_action)
        self.g_payload = QLineEdit()
        self.g_payload.setPlaceholderText('attach/detach only: payload id')
        f.addRow('Payload', self.g_payload)
        note = QLabel('Object dynamics (attach_box cuboid, freeze/gravity on release) '
                      'come from the SCENE (Step 2).')
        note.setWordWrap(True)
        f.addRow(note)
        return w

    def _wait_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.w_ms = QSpinBox()
        self.w_ms.setRange(1, 600000); self.w_ms.setValue(500); self.w_ms.setSuffix(' ms')
        f.addRow('Pause', self.w_ms)
        return w

    def _reset_form(self) -> QWidget:
        w = QWidget()
        v = QVBoxLayout(w)
        note = QLabel('Puts every DYNAMIC object back at its initial (scene.yaml) pose — '
                      'the cycle boundary of a looping app in SIMULATION: the next cycle '
                      'picks where cycle 1 did. Detaches anything still held. In Isaac the '
                      'adapter is signalled on /isaac_scene_reset. No parameters. '
                      'With camera-guided moves in the loop, the Step-5 settle pause is '
                      'emitted automatically after the reset.')
        note.setWordWrap(True)
        v.addWidget(note)
        v.addStretch(1)
        return w

    def _detect_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.dp_detector = QComboBox()
        f.addRow('Detector (Step 5)', self.dp_detector)
        note = QLabel('Samples the camera AT THIS POINT of the flow: DetectObject '
                      'runs here and updates every Move guided by this detector. '
                      'Place it where the scene is visible (e.g. at ready, before '
                      'the approach). A cycle may hold several Detect blocks with '
                      'different detectors.')
        note.setWordWrap(True)
        f.addRow(note)
        return w

    def _fill_detect_point(self, b: dict) -> None:
        names = []
        try:
            names = self.ctrl.detector_names()
        except Exception:  # noqa: BLE001
            pass
        self.dp_detector.blockSignals(True)
        self.dp_detector.clear()
        self.dp_detector.addItems(names or ['(configure one in Step 5)'])
        if b.get('detector') in names:
            self.dp_detector.setCurrentText(b['detector'])
        self.dp_detector.blockSignals(False)

    def _policy_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.p_name = QLineEdit()
        f.addRow('Name', self.p_name)
        self.p_mode = QComboBox()
        self.p_mode.addItems(['hybrid', 'pure', 'residual'])
        # residual is SHOWN but NOT selectable — it needs a residual-trained policy and a
        # correction-aware executor (ADR-0006).
        _ridx = self.p_mode.findText('residual')
        if _ridx >= 0:
            self.p_mode.model().item(_ridx).setEnabled(False)
            self.p_mode.setItemData(_ridx, 'requires a residual-trained policy (not yet)',
                                    Qt.ToolTipRole)
        self.p_mode.setToolTip('hybrid: the policy decides the target and the '
                               'deterministic runtime moves there (recommended). '
                               'pure: the policy drives the robot. residual: the '
                               'policy corrects a nominal move.')
        f.addRow('Mode', self.p_mode)
        self.p_card = QLineEdit()
        self.p_card.setReadOnly(True)
        browse = QPushButton('Load policy card (.yaml, .pt beside it)…')
        browse.clicked.connect(self._browse_policy_card)
        r = QHBoxLayout(); r.addWidget(self.p_card); r.addWidget(browse)
        f.addRow('Policy card', r)
        self.p_info = QLabel('Load a policy card to see what it was trained for and '
                             'where it leaves the robot.')
        self.p_info.setWordWrap(True)
        self.p_info.setStyleSheet('color:#00838f;')
        f.addRow(self.p_info)
        self.p_check = QCheckBox('Check the tcp reaches the card end pose afterwards')
        self.p_check.setChecked(True)
        f.addRow(self.p_check)
        self.p_attach = QCheckBox('Require the object attached (suction) afterwards')
        f.addRow(self.p_attach)
        self.p_attach_topic = QLineEdit('/suction/attached')
        f.addRow('Attached topic (std_msgs/Bool)', self.p_attach_topic)
        self.p_calib = QLineEdit()
        self.p_calib.setPlaceholderText('dx, dy, dz, droll, dpitch, dyaw — empty = none')
        f.addRow('Deploy calibration (optional)', self.p_calib)
        self.p_calib_frame = QComboBox()
        self.p_calib_frame.addItems(['base_link', 'tcp'])
        f.addRow('Calibration frame', self.p_calib_frame)
        self.p_calib_note = QLineEdit()
        self.p_calib_note.setPlaceholderText('why (e.g. systematic +2cm x bias, run-7)')
        f.addRow('Calibration note', self.p_calib_note)
        cnote = QLabel('Calibration is a POST-HOC fix for a SYSTEMATIC policy bias — use '
                       'only if the error is consistent; the clean fix is retraining.')
        cnote.setWordWrap(True)
        cnote.setStyleSheet('color:#8a6d3b;')
        f.addRow(cnote)
        note = QLabel('Place this block ABOVE the Move it produces: the policy brings '
                      'the robot into that next Move (e.g. pre-place with the cube '
                      'held). A CheckRobotState is emitted right after to confirm the '
                      'end state before the next step runs.')
        note.setWordWrap(True)
        f.addRow(note)
        return w

    def _browse_policy_card(self):  # pragma: no cover - file dialog
        path, _ = QFileDialog.getOpenFileName(
            self, 'Select the policy card (YAML)', '',
            'Policy card (*.yaml *.yml);;All files (*)')
        if not path:
            return
        self.p_card.setText(path)
        self.p_info.setText(self._policy_card_summary(path))

    @staticmethod
    def _policy_card_summary(path: str) -> str:
        import yaml
        try:
            card = (yaml.safe_load(open(path)) or {}).get('policy_card', {})
        except Exception:  # noqa: BLE001
            return 'Could not read this card.'
        trained = (card.get('trained_for') or '').strip()
        cat = card.get('category', 'reach_grasp')
        eef = card.get('end_effector', '')
        head = f"Category: {cat}" + (f" · end-effector: {eef}" if eef else "")
        es = card.get('end_state', {}) or {}
        tcp = es.get('tcp_pose_base')
        end = ''
        if tcp and len(tcp) >= 3:
            end = (f'\nEnds ~({tcp[0]:.3f}, {tcp[1]:.3f}, {tcp[2]:.3f}) m'
                   + (', object HELD' if es.get('object_attached') else ', empty'))
        return f"{head}\nTrained for: {trained or '(unnamed)'}{end}"

    def _fill_policy(self, b: dict) -> None:
        self.p_name.setText(b.get('name', ''))
        self.p_mode.setCurrentText(b.get('mode', 'hybrid'))
        self.p_card.setText(b.get('card', ''))
        self.p_check.setChecked(bool(b.get('check_position', True)))
        self.p_attach.setChecked(bool(b.get('require_attached', False)))
        self.p_attach_topic.setText(b.get('attached_topic', '') or '/suction/attached')
        self.p_calib.setText(b.get('calibration', ''))
        self.p_calib_frame.setCurrentText(b.get('calibration_frame', 'base_link'))
        self.p_calib_note.setText(b.get('calibration_note', ''))
        self.p_info.setText(self._policy_card_summary(b['card']) if b.get('card')
                            else 'Load a policy card to see its training + end state.')

    def _fill_loop_to(self, current=''):
        """Move blocks the loop may end on. Rebuilt on selection so it always matches
        the current sequence."""
        names = [b['name'] for b in self.blocks if b['kind'] == 'move']
        self.l_to.blockSignals(True)
        self.l_to.clear()
        self.l_to.addItem('(end)')
        self.l_to.addItems(names)
        self.l_to.setCurrentText(current if current in names else '(end)')
        self.l_to.blockSignals(False)

    def _loop_form(self) -> QWidget:
        w = QWidget()
        f = QFormLayout(w)
        self.l_forever = QCheckBox('repeat forever')
        self.l_forever.setChecked(True)
        f.addRow(self.l_forever)
        self.l_cycles = QSpinBox(); self.l_cycles.setRange(1, 100000); self.l_cycles.setValue(2)
        f.addRow('…or cycles', self.l_cycles)
        self.l_to = QComboBox()
        self.l_to.setEditable(False)
        f.addRow('…up to (incl.)', self.l_to)
        note = QLabel('The loop repeats the moves FROM this block DOWN TO the one chosen '
                      'above — drag the Loop block to set where it starts. Blocks above '
                      'it (a homing move, say) stay outside. "(end)" = to the last block.')
        note.setWordWrap(True)
        f.addRow(note)
        return w

    # ---- palette / list operations ------------------------------------------
    def add_block(self, kind: str) -> None:
        if kind == 'loop' and any(b['kind'] == 'loop' for b in self.blocks):
            self.status.setText('Only one Loop block per sequence.')
            return
        defaults = {
            'move':    {'kind': 'move', 'name': f'move_{sum(1 for b in self.blocks if b["kind"] == "move") + 1}',
                        'target': 'named', 'named': '', 'pos': '', 'quat': '0, 0, 0, 1',
                        'motion': 'ptp', 'planner': '', 'speed': 50, 'tol': 0.1, 'check': 'inherit',
                        'detector': '', 'vdx': 0.0, 'vdy': 0.0, 'vdz': 0.0,
                        'vori': 'keep', 'vori_ref': '', 'vori_raw': '',
                        'rel_step': '', 'rdx': 0.0, 'rdy': 0.0, 'rdz': 0.0,
                        'rroll': 0.0, 'rpitch': 0.0, 'ryaw': 0.0},
            'gripper': {'kind': 'gripper', 'action': 'close', 'payload': ''},
            'reset':   {'kind': 'reset'},
            'detect':  {'kind': 'detect', 'detector': ''},
            'policy':  {'kind': 'policy',
                        'name': f'policy_{sum(1 for b in self.blocks if b["kind"] == "policy") + 1}',
                        'mode': 'hybrid', 'card': '', 'check_position': True,
                        'require_attached': False, 'attached_topic': '/suction/attached',
                        'calibration': '', 'calibration_frame': 'base_link',
                        'calibration_note': ''},
            'wait':    {'kind': 'wait', 'ms': 500},
            'loop':    {'kind': 'loop', 'cycles': -1},
        }[kind]
        self.blocks.append(dict(defaults))
        self._refresh(select=len(self.blocks) - 1)

    def remove_selected(self) -> None:
        row = self.seq.currentRow()
        if 0 <= row < len(self.blocks):
            del self.blocks[row]
            self._refresh(select=min(row, len(self.blocks) - 1))

    def _on_reorder(self, *_args) -> None:
        order = []
        for i in range(self.seq.count()):
            order.append(self.seq.item(i).data(Qt.UserRole))
        # Rebuilding the list widget INSIDE the drop's rowsMoved signal is fragile
        # (stale persistent indexes mid-dropEvent) — defer it out of the drop.
        QTimer.singleShot(0, lambda: self._apply_reorder(order))

    def _apply_reorder(self, order) -> None:
        self.blocks = [self.blocks[j] for j in order]
        # The Loop block's POSITION is its meaning: the region runs from the first Move
        # below it down to the block named in '...up to'. It used to be force-moved back
        # to the end on every reorder (when a loop always wrapped the whole sequence),
        # which made it the one block the user could not place. Leave it where it lands.
        self._refresh(select=self.seq.currentRow())

    def _on_select(self, row: int) -> None:
        self._selected = row
        if not (0 <= row < len(self.blocks)):
            self.stack.setCurrentIndex(0)
            return
        b = self.blocks[row]
        idx = {'move': 1, 'gripper': 2, 'wait': 3, 'loop': 4, 'reset': 5,
               'detect': 6, 'policy': 7}[b['kind']]
        self.stack.setCurrentIndex(idx)
        if b['kind'] == 'detect':
            self._fill_detect_point(b)
        if b['kind'] == 'policy':
            self._fill_policy(b)
        if b['kind'] == 'move':
            self._fill_move_vision(b)
            self.m_name.setText(b['name'])
            self.m_target.setCurrentIndex(0 if b['target'] == 'named' else 1)
            self.m_named.setText(b['named'])
            self.m_pos.setText(b['pos'])
            self.m_quat.setText(b['quat'])
            self.m_motion.setCurrentText(b['motion'])
            self.m_planner.setCurrentText(b['planner'])
            self.m_speed.setValue(int(b['speed']))
            self.m_tol.setValue(float(b['tol']))
            self.m_check.setCurrentText(b['check'])
        elif b['kind'] == 'gripper':
            labels = {'close': 0, 'open': 1, 'attach': 2, 'detach': 3}
            self.g_action.setCurrentIndex(labels.get(b['action'], 0))
            self.g_payload.setText(b.get('payload', ''))
        elif b['kind'] == 'wait':
            self.w_ms.setValue(int(b['ms']))
        elif b['kind'] == 'loop':
            self._fill_loop_to(b.get('to', ''))
            self.l_forever.setChecked(b['cycles'] == -1)
            if b['cycles'] > 0:
                self.l_cycles.setValue(int(b['cycles']))

    def apply_inspector(self) -> None:
        row = self._selected
        if not (0 <= row < len(self.blocks)):
            return
        b = self.blocks[row]
        if b['kind'] == 'move':
            det = self.m_detector.currentText()
            det = '' if det.startswith('(') else det
            rel_step = self.m_rel.currentText()
            rel_step = '' if rel_step.startswith('(') else rel_step
            if det and rel_step:
                # one master per move: keep the camera, drop the relative selection
                rel_step = ''
                self.status.setText(f"'{b['name']}': a move follows EITHER the "
                                    'camera OR a step — kept the camera; set the '
                                    'detector to (none) first to make it relative.')
            old_det = b.get('detector', '')
            vori = {_VORI_KEEP: 'keep', _VORI_SAME: 'same',
                    _VORI_ALIGN: 'align'}[self.m_vori.currentIndex()]
            # a literal-quaternion orientation (hand-edited project) is carried in
            # vori_raw; it survives an untouched Apply and is dropped only when the
            # user actually changes the policy combo
            if self.m_vori.currentIndex() != getattr(self, '_filled_vori',
                                                     self.m_vori.currentIndex()):
                b.pop('vori_raw', None)
            b.update(name=self.m_name.text().strip() or b['name'],
                     target='named' if self.m_target.currentIndex() == 0 else 'tcp',
                     named=self.m_named.text().strip(),
                     pos=self.m_pos.text().strip(), quat=self.m_quat.text().strip(),
                     motion=self.m_motion.currentText(),
                     planner=self.m_planner.currentText(),
                     speed=self.m_speed.value(), tol=self.m_tol.value(),
                     check=self.m_check.currentText(),
                     detector=det, vdx=self.m_dx.value(), vdy=self.m_dy.value(),
                     vdz=self.m_dz.value(), vori=vori,
                     vori_ref=self.m_vori_ref.currentText().strip(),
                     rel_step=rel_step, rdx=self.m_rdx.value(),
                     rdy=self.m_rdy.value(), rdz=self.m_rdz.value(),
                     rroll=self.m_rroll.value(), rpitch=self.m_rpitch.value(),
                     ryaw=self.m_ryaw.value())
            # a vision goal lands anywhere: the travel into it must be
            # collision-aware — so when the camera FIRST takes this move over,
            # default a resolved ptp/pilz to free/OMPL. A later deliberate
            # ptp/pilz re-choice is respected (validate() still flags it).
            switch_msg = ''
            try:
                eff = (b['planner'] or
                       self.ctrl.project.application.global_planner_mode.value)
            except Exception:  # noqa: BLE001
                eff = b['planner'] or 'pilz'
            if det and not old_det and b['motion'] == 'ptp' and eff == 'pilz':
                b.update(motion='free', planner='ompl')
                switch_msg = (f"'{b['name']}': motion set to free/OMPL — a "
                              'vision-driven move needs a collision-aware '
                              'planner (Pilz PTP cannot avoid the camera).')
        elif b['kind'] == 'gripper':
            b.update(action=('close', 'open', 'attach', 'detach')[self.g_action.currentIndex()],
                     payload=self.g_payload.text().strip())
        elif b['kind'] == 'wait':
            b.update(ms=self.w_ms.value())
        elif b['kind'] == 'detect':
            det = self.dp_detector.currentText()
            b.update(detector='' if det.startswith('(') else det)
        elif b['kind'] == 'policy':
            b.update(name=self.p_name.text().strip() or b['name'],
                     mode=self.p_mode.currentText(),
                     card=self.p_card.text().strip(),
                     check_position=self.p_check.isChecked(),
                     require_attached=self.p_attach.isChecked(),
                     attached_topic=self.p_attach_topic.text().strip(),
                     calibration=self.p_calib.text().strip(),
                     calibration_frame=self.p_calib_frame.currentText(),
                     calibration_note=self.p_calib_note.text().strip())
        elif b['kind'] == 'loop':
            to = self.l_to.currentText()
            b.update(cycles=-1 if self.l_forever.isChecked() else self.l_cycles.value(),
                     to='' if to == '(end)' else to)
        self._refresh(select=row)
        if b['kind'] == 'move' and switch_msg:
            self.status.setText(switch_msg)   # after _refresh, or the fold clears it

    # ---- live capture (blind mode) ------------------------------------------
    def capture_joints(self):  # pragma: no cover - needs a live ROS session
        name = self.m_name.text().strip()
        if not name:
            self.status.setText('Set the block Name first, then capture.')
            return
        joints = self.ctrl.robot_summary()['arm_joints']
        try:
            values = self.wizard().live_capture().current_joint_values(joints)
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f'capture failed: {exc}')
            return
        self.ctrl.add_named_state(name, values)
        self.m_named.setText(name)
        self.m_target.setCurrentIndex(0)
        self.status.setText(f'captured joints -> named state "{name}"')

    def capture_pose(self):  # pragma: no cover - needs a live ROS session
        s = self.ctrl.robot_summary()
        try:
            pos, quat = self.wizard().live_capture().current_pose(s['base_frame'], s['tip_link'])
        except Exception as exc:  # noqa: BLE001
            self.status.setText(f'capture failed: {exc}')
            return
        self.m_pos.setText(', '.join(f'{v:.4f}' for v in pos))
        self.m_quat.setText(', '.join(f'{v:.4f}' for v in quat))
        self.m_target.setCurrentIndex(1)
        self.status.setText('captured TCP pose')

    def _show_deployment(self):  # pragma: no cover - simple dialog
        DeploymentDialog(self.ctrl, self).exec_()

    def _show_end_effector(self):  # pragma: no cover - exec loop; dialog is unit-tested
        EndEffectorDialog(self.ctrl, self.wizard(), self).exec_()

    # ---- rendering + model sync ---------------------------------------------
    def _label(self, b: dict) -> str:
        if b['kind'] == 'move':
            tgt = b['named'] or b['name'] if b['target'] == 'named' else 'tcp pose'
            chk = '' if b['check'] == 'inherit' else f"  · chk {b['check'].upper()}"
            cam = ''
            if b.get('detector'):
                vdz = b.get('vdz', 0.0)
                cam = (f"  · \U0001F4F7 {b['detector']}"
                       + (f" {vdz * 1000:+.0f}mm" if vdz else ''))
            elif b.get('rel_step'):
                rdz = b.get('rdz', 0.0)
                cam = (f"  · ⇢ {b['rel_step']}"
                       + (f" {rdz * 1000:+.0f}mm" if rdz else ''))
            return (f"{b['name']}   [{b['motion']}"
                    f"{('/' + b['planner']) if b['planner'] else ''} {b['speed']}%]"
                    f"  → {tgt}{chk}{cam}")
        if b['kind'] == 'gripper':
            names = {'close': 'CLOSE (grasp)', 'open': 'OPEN (release)',
                     'attach': f"ATTACH {b.get('payload', '')}",
                     'detach': f"DETACH {b.get('payload', '')}"}
            return f"Gripper {names[b['action']]}"
        if b['kind'] == 'reset':
            return 'Reset scene → initial poses (sim)'
        if b['kind'] == 'detect':
            return f"Detect {b.get('detector') or '?'} — sample the camera here"
        if b['kind'] == 'policy':
            card = b.get('card', '')
            cname = card.rsplit('/', 1)[-1] if card else 'no card'
            return (f"\U0001F9E0 Policy {b.get('name', '?')} "
                    f"[{b.get('mode', 'hybrid')}] → next move · {cname}")
        if b['kind'] == 'wait':
            return f"Wait {b['ms']} ms"
        times = 'forever' if b['cycles'] == -1 else f"{b['cycles']}×"
        upto = b.get('to') or 'end'
        return f'LOOP starts here → repeats down to {upto} — {times}'

    def _refresh(self, select: int = -1) -> None:
        # container rendering: with a Loop present, the contained blocks carry a │
        # bar and the Loop closes the bracket (└─) — the loop's SCOPE reads at a glance.
        # the region runs from the block AFTER the Loop down to its 'to' move
        # (inclusive), so the bracket shows exactly what repeats.
        loop_i = next((i for i, b in enumerate(self.blocks) if b['kind'] == 'loop'), None)
        end_i = len(self.blocks) - 1
        if loop_i is not None:
            to = self.blocks[loop_i].get('to') or ''
            if to:
                end_i = next((i for i, b in enumerate(self.blocks)
                              if b['kind'] == 'move' and b['name'] == to), end_i)
        self.seq.blockSignals(True)
        self.seq.clear()
        for i, b in enumerate(self.blocks):
            icon, color, _ = _BLOCK_META[b['kind']]
            inside = loop_i is not None and loop_i < i <= end_i
            if b['kind'] == 'loop':
                text = f'┌─ {icon}  {self._label(b)}'
            elif inside:
                text = f'│  {icon}  {self._label(b)}'
                if i == end_i:
                    text = f'└─ {icon}  {self._label(b)}'
            else:
                text = f'{icon}  {self._label(b)}'
            item = QListWidgetItem(text)
            item.setForeground(QBrush(QColor(color)))
            item.setData(Qt.UserRole, i)
            self.seq.addItem(item)
        self.seq.blockSignals(False)
        if 0 <= select < self.seq.count():
            self.seq.setCurrentRow(select)
        bound = {b['detector'] for b in self.blocks
                 if b['kind'] == 'move' and b.get('detector')}
        explicit = {b['detector'] for b in self.blocks
                    if b['kind'] == 'detect' and b.get('detector')}
        dets = sorted(bound - explicit)
        if dets:
            self.auto_detect.setText(
                '\U0001F4F7 Detect ' + ', '.join(dets) + ' — automatic, at cycle '
                'start (add a Detect block to choose the moment); after a Reset '
                'scene the Step-5 settle pause is emitted.')
        self.auto_detect.setVisible(bool(dets))
        self._sync_model()

    def _sync_model(self) -> None:
        """FOLD the block list into the canonical model (the generator's contract)."""
        ctrl = self.ctrl
        try:
            ctrl.clear_application()
        except Exception:  # noqa: BLE001 - no project yet
            return
        prev_move = None
        waits: Dict[str, int] = {}
        loop = 0
        loop_from = loop_to = None
        bindings = []                 # (waypoint, detector, dx, dy, dz, orientation)
        relatives = []                # (waypoint, step, dx, dy, dz, dr, dp, dyw)
        warn = ''
        for b in self.blocks:
            if b['kind'] == 'move':
                kwargs = dict(name=b['name'], motion=b['motion'],
                              planner=b['planner'] or None, speed=int(b['speed']),
                              allowed_start_tolerance=float(b['tol']),
                              attached_collision_check={'inherit': None, 'on': True,
                                                        'off': False}[b['check']],
                              # carried through the block dict (no widgets): the fold
                              # must not silently strip a reopened project's data
                              role=b.get('role', 'generic'),
                              aux=b.get('aux'),
                              aux_is_center=bool(b.get('aux_is_center', False)))
                try:
                    if b['target'] == 'named':
                        kwargs['named'] = b['named'] or b['name']
                    else:
                        # an EMPTY pose is legal for a camera-guided or relative
                        # move (the pose is written at run time); it stays tcp
                        kwargs['position'] = _parse_floats(b['pos']) or None
                        kwargs['orientation'] = _parse_floats(b['quat']) or None
                        kwargs['tcp'] = True
                    ctrl.add_move(**kwargs)
                    prev_move = b['name']
                    if b.get('detector'):
                        vori = b.get('vori', 'keep')
                        ref = b.get('vori_ref', '')
                        if b.get('vori_raw'):
                            orientation = b['vori_raw']   # literal quaternion, verbatim
                        elif vori == 'same' and ref:
                            orientation = f'from:{ref}'
                        elif vori == 'align':
                            orientation = 'detected'
                        else:
                            orientation = 'keep'
                        bindings.append((b['name'], b['detector'],
                                         float(b.get('vdx', 0.0)),
                                         float(b.get('vdy', 0.0)),
                                         float(b.get('vdz', 0.0)), orientation))
                    elif b.get('rel_step'):
                        relatives.append((b['name'], b['rel_step'],
                                          float(b.get('rdx', 0.0)),
                                          float(b.get('rdy', 0.0)),
                                          float(b.get('rdz', 0.0)),
                                          float(b.get('rroll', 0.0)),
                                          float(b.get('rpitch', 0.0)),
                                          float(b.get('ryaw', 0.0))))
                except Exception:  # noqa: BLE001 - incomplete block, keep editing
                    warn = f"block '{b['name']}': incomplete target (set pose or named state)"
            elif b['kind'] == 'gripper':
                if prev_move is None:
                    warn = 'a Gripper block needs a Move before it'
                    continue
                kind = {'close': 'grasp', 'open': 'release',
                        'attach': 'attach', 'detach': 'detach'}[b['action']]
                ctrl.add_tool_action(prev_move, kind, b.get('payload') or None)
            elif b['kind'] == 'reset':
                if prev_move is None:
                    warn = 'a Reset scene block needs a Move before it'
                    continue
                ctrl.add_tool_action(prev_move, 'reset_scene')
            elif b['kind'] == 'wait':
                if prev_move is None:
                    warn = 'a Wait block needs a Move before it'
                    continue
                waits[prev_move] = waits.get(prev_move, 0) + int(b['ms'])
            elif b['kind'] == 'loop':
                loop = int(b['cycles'])
                loop_to = b.get('to') or None
                # the loop starts at the first Move AFTER the block, so dragging it below
                # a homing move leaves that move outside the cycle
                loop_from = next((x['name'] for x in self.blocks[self.blocks.index(b) + 1:]
                                  if x['kind'] == 'move'), None)
        for wp, ms in waits.items():
            ctrl.set_wait_after(wp, ms)
        ctrl.set_loop(loop, start=loop_from, end=loop_to)
        # vision bindings (D-016): applied AFTER the moves exist; the detection
        # itself is emitted automatically at cycle start by the application template.
        for name, det, dx, dy, dz, orientation in bindings:
            try:
                ctrl.bind_vision(name, det, dx=dx, dy=dy, dz=dz,
                                 orientation=orientation)
            except Exception as exc:  # noqa: BLE001 - dangling names while editing
                warn = f'vision binding: {exc}'
        for name, step, dx, dy, dz, dr, dp, dyw in relatives:
            try:
                ctrl.bind_relative(name, step, dx=dx, dy=dy, dz=dz,
                                   droll=dr, dpitch=dp, dyaw=dyw)
            except Exception as exc:  # noqa: BLE001 - dangling names while editing
                warn = f'relative binding: {exc}'
        # explicit Detect blocks (D-018): the anchor is the NEXT move below them
        for i, b in enumerate(self.blocks):
            if b['kind'] != 'detect':
                continue
            if not b.get('detector'):
                warn = 'a Detect block needs a detector (Step 5)'
                continue
            nxt = next((x['name'] for x in self.blocks[i + 1:]
                        if x['kind'] == 'move'), None)
            if nxt is None:
                warn = f"Detect '{b['detector']}': no Move below it — put it above " \
                       'the move it should precede'
                continue
            ctrl.add_detect_point(b['detector'], nxt)
        # learned-policy steps (ADR-0005): the policy produces the move into the NEXT
        # move below it (same anchoring as a Detect block).
        for i, b in enumerate(self.blocks):
            if b['kind'] != 'policy':
                continue
            nxt = next((x['name'] for x in self.blocks[i + 1:]
                        if x['kind'] == 'move'), None)
            if nxt is None:
                warn = (f"Policy '{b.get('name', '?')}': no Move below it — put it "
                        'above the move it should produce')
                continue
            try:
                ctrl.add_policy(b.get('name') or f'policy_{i}', nxt,
                                mode=b.get('mode', 'hybrid'), card=b.get('card', ''),
                                require_attached=bool(b.get('require_attached', False)),
                                attached_topic=b.get('attached_topic', ''),
                                check_position=bool(b.get('check_position', True)),
                                calibration=b.get('calibration', ''),
                                calibration_frame=b.get('calibration_frame', 'base_link'),
                                calibration_note=b.get('calibration_note', ''))
            except Exception as exc:  # noqa: BLE001 - dangling names while editing
                warn = f'policy step: {exc}'
        self.status.setText(warn)

    # ---- lifecycle -----------------------------------------------------------
    def initializePage(self) -> None:
        """Rebuild the block list from the project (reopen / back-navigation)."""
        try:
            app = self.ctrl.project.application
        except Exception:  # noqa: BLE001
            return
        if self.blocks or not app.sequence:
            return                        # keep in-progress edits
        blocks = []
        for name in app.sequence:
            wp = app.waypoint_by_name(name)
            seg = app.segment_for(name)
            if wp is None:
                continue
            blocks.append({'kind': 'move', 'name': name,
                           'target': 'named' if wp.type.value == 'joint' else 'tcp',
                           'named': wp.named or '',
                           'pos': ', '.join(format(v, 'g') for v in (wp.position or [])),
                           'quat': ', '.join(format(v, 'g') for v in (wp.orientation or [0, 0, 0, 1])),
                           'motion': seg.motion.value if seg else 'ptp',
                           'planner': (seg.planner.value if seg and seg.planner else ''),
                           'speed': seg.speed if seg else 50,
                           'tol': wp.allowed_start_tolerance,
                           'check': ('inherit' if not seg or seg.attached_collision_check is None
                                     else ('on' if seg.attached_collision_check else 'off')),
                           'role': wp.role.value,
                           'aux': (list(seg.aux) if seg and seg.aux else None),
                           'aux_is_center': (seg.aux_is_center if seg else False),
                           'detector': (wp.vision.detector if wp.vision else ''),
                           'vdx': (wp.vision.dx if wp.vision else 0.0),
                           'vdy': (wp.vision.dy if wp.vision else 0.0),
                           'vdz': (wp.vision.dz if wp.vision else 0.0),
                           'vori': ('align' if wp.vision and wp.vision.orientation == 'detected'
                                    else ('same' if wp.vision and
                                          wp.vision.orientation.startswith('from:')
                                          else 'keep')),
                           'vori_ref': (wp.vision.orientation.split(':', 1)[1]
                                        if wp.vision and
                                        wp.vision.orientation.startswith('from:')
                                        else ''),
                           'vori_raw': (wp.vision.orientation
                                        if wp.vision and wp.vision.orientation
                                        not in ('keep', 'detected') and not
                                        wp.vision.orientation.startswith('from:')
                                        else ''),
                           'rel_step': (wp.relative.step if wp.relative else ''),
                           'rdx': (wp.relative.dx if wp.relative else 0.0),
                           'rdy': (wp.relative.dy if wp.relative else 0.0),
                           'rdz': (wp.relative.dz if wp.relative else 0.0),
                           'rroll': (wp.relative.droll if wp.relative else 0.0),
                           'rpitch': (wp.relative.dpitch if wp.relative else 0.0),
                           'ryaw': (wp.relative.dyaw if wp.relative else 0.0)})
            acts = app.actions_at(name)
            for act in acts:
                if act.kind.value == 'reset_scene':
                    continue                      # rendered AFTER the wait (cycle boundary)
                rev = {'grasp': 'close', 'release': 'open',
                       'attach': 'attach', 'detach': 'detach'}
                blocks.append({'kind': 'gripper', 'action': rev.get(act.kind.value, 'close'),
                               'payload': act.payload_ref or ''})
            if seg and seg.wait_after_ms > 0:
                blocks.append({'kind': 'wait', 'ms': seg.wait_after_ms})
            for act in acts:
                if act.kind.value == 'reset_scene':
                    blocks.append({'kind': 'reset'})
        if app.loop_cycles != 0:
            lb = {'kind': 'loop', 'cycles': app.loop_cycles, 'to': app.loop_end or ''}
            # The Loop block's POSITION is its meaning (the region starts at the first
            # Move after it) — so it must be REBUILT before the loop_start move, or
            # merely entering this page would fold loop_start back to None and a
            # regenerated bundle would wrap the homing move into the cycle.
            idx = None
            if app.loop_start:
                idx = next((i for i, blk in enumerate(blocks)
                            if blk['kind'] == 'move' and blk['name'] == app.loop_start),
                           None)
            if idx is not None:
                blocks.insert(idx, lb)
            else:
                blocks.append(lb)
        # rebuild the explicit Detect blocks (D-018) before their anchor moves;
        # reversed so repeated inserts at the same index keep the declared order
        for pt in reversed(app.detections):
            idx = next((i for i, blk in enumerate(blocks)
                        if blk['kind'] == 'move' and blk['name'] == pt.before_waypoint),
                       None)
            db = {'kind': 'detect', 'detector': pt.detector}
            if idx is None:
                blocks.append(db)
            else:
                blocks.insert(idx, db)
        # rebuild the learned-policy steps (ADR-0005) before their anchor moves
        for pol in reversed(app.policies):
            idx = next((i for i, blk in enumerate(blocks)
                        if blk['kind'] == 'move' and blk['name'] == pol.before_waypoint),
                       None)
            pb = {'kind': 'policy', 'name': pol.name, 'mode': pol.mode.value,
                  'card': pol.card, 'check_position': pol.check_position,
                  'require_attached': pol.require_attached,
                  'attached_topic': pol.attached_topic or ''}
            if idx is None:
                blocks.append(pb)
            else:
                blocks.insert(idx, pb)
        self.blocks = blocks
        self._refresh()

    def validatePage(self) -> bool:
        self._sync_model()
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
        self.perception_page = PerceptionPage(self.controller)
        self.application_page = ApplicationPage(self.controller)
        self.blocks_page = BlocksPage(self.controller)
        self.generate_page = GeneratePage(self.controller)
        # From-scratch / advanced pages: constructed (used by the CLI + tests, and as
        # manual overrides of the base-derived values), not in the default MVP flow.
        # waypoints_page is the LEGACY Step 6 (form-based), kept for tests/back-compat;
        # the MVP flow now uses the block-based BlocksPage.
        self.waypoints_page = WaypointsPage(self.controller)
        self.load_page = LoadRobotPage(self.controller)
        self.group_page = GroupFramesPage(self.controller)
        self.controllers_page = ControllersPage(self.controller)
        self.states_page = NamedStatesPage(self.controller)
        # The two-phase MVP flow (Phase A: build the faithful env; Phase B: the app).
        for page in (self.base_page, self.scene_page, self.gen_config_page,
                     self.mode_page, self.perception_page, self.application_page,
                     self.blocks_page, self.generate_page):
            self.addPage(page)
        self._scene_pub = None
        self._camera = None
        self._camera_topics = None

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

    def camera_capture(self, rgb_topic, depth_topic, info_topic):  # pragma: no cover
        """Lazily create the shared camera reader; recreate it if the topics change."""
        topics = (rgb_topic, depth_topic, info_topic)
        if self._camera is not None and self._camera_topics != topics:
            self._camera.close()
            self._camera = None
        if self._camera is None:
            from ..livesession import LiveCameraCapture
            self._camera = LiveCameraCapture(rgb_topic, depth_topic, info_topic)
            self._camera_topics = topics
        return self._camera

    def closeEvent(self, event):  # pragma: no cover - GUI lifecycle
        if self._live is not None:
            self._live.close()
        if self._scene_pub is not None:
            self._scene_pub.close()
        if self._camera is not None:
            self._camera.close()
        super().closeEvent(event)
