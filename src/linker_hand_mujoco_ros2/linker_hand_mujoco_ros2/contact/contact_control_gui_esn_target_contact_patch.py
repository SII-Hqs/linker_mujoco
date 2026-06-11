#!/usr/bin/env python3
"""PyQt5 GUI for target q/qdot ESN control with injected contact-patch forces."""

import argparse
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import mujoco
import mujoco.viewer
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from ..mujoco_esn_controller_target_q_dq import (
        ControlRateAdapter,
        ESNTorqueController,
    )
    from ..position_command_mapper import PositionCommandMapper
    from .contact_control_demo import default_finger_vec, load_finger_frame
    from .contact_control_demo_esn_target import default_esn_model_path
    from .contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from .contact_control_gui_esn import ReservoirHeatmapWidget
    from .contact_patch import ContactPatchCalculator
    from .contact_visualizer import ContactVisualizer
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn_target import (
        default_esn_model_path,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui_esn import (
        ReservoirHeatmapWidget,
    )
    from linker_hand_mujoco_ros2.contact.contact_patch import ContactPatchCalculator
    from linker_hand_mujoco_ros2.contact.contact_visualizer import ContactVisualizer
    from linker_hand_mujoco_ros2.mujoco_esn_controller_target_q_dq import (
        ControlRateAdapter,
        ESNTorqueController,
    )
    from linker_hand_mujoco_ros2.position_command_mapper import PositionCommandMapper


def contact_force_g_to_n(contact_force_g):
    """Convert gram-force to newtons."""
    return float(contact_force_g) * 9.80665 / 1000.0


def _find_actuator_ids_for_joints(model, joint_names):
    """Return actuator IDs whose transmission targets the given joints."""
    target_names = set(joint_names)
    actuator_ids = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model.joint(joint_id).name
        if joint_name in target_names:
            actuator_ids.append(actuator_id)
    if not actuator_ids:
        raise ValueError(f"no actuators found for joints: {joint_names}")
    return actuator_ids


def _apply_contact_params(model, solref, solimp):
    """Override solref/solimp for all geom pairs at runtime."""
    solref = np.asarray(solref, dtype=float)
    solimp = np.asarray(solimp, dtype=float)
    for geom_id in range(model.ngeom):
        model.geom_solref[geom_id, : len(solref)] = solref
        model.geom_solimp[geom_id, : len(solimp)] = solimp


@dataclass
class ESNTargetContactPatchSimulation:
    """Target q/qdot ESN simulation with GUI-injected contact-patch force."""

    model: object
    data: object
    contact_patch: ContactPatchCalculator
    contact_visualizer: ContactVisualizer
    controller: ControlRateAdapter
    target_qpos: np.ndarray
    target_qvel: np.ndarray
    contact_force_g: float
    index_dof_ids: List[int] = field(default_factory=list)
    index_actuator_ids: List[int] = field(default_factory=list)
    last_contact_force_g: float | None = None
    force_profile_mode: str = "Manual"
    force_min_g: float = 0.0
    force_max_g: float = 100.0
    force_period_s: float = 4.0
    force_profile_start_time: float = 0.0

    def step(self):
        """Advance simulation and inject the patch force into MuJoCo dynamics."""
        contact_force_g = self.current_contact_force_g()
        self.contact_force_g = contact_force_g
        force_update = (
            self.last_contact_force_g is None
            or abs(contact_force_g - self.last_contact_force_g) > 1e-9
        )
        if self.contact_patch.update(contact_force_g, force_update=force_update):
            self.last_contact_force_g = contact_force_g

        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.qfrc_applied[:] += self.contact_patch.contact_generalized_force

        qpos = self.current_qpos()
        qvel = self.data.qvel[self.index_dof_ids]
        force_n = contact_force_g_to_n(contact_force_g)
        tau = self.controller.step(
            qpos,
            qvel,
            force_n,
            self.target_qpos,
            self.target_qvel,
        )

        self.data.ctrl[:] = 0.0
        for i, act_id in enumerate(self.index_actuator_ids):
            self.data.ctrl[act_id] = tau[i]

        mujoco.mj_step(self.model, self.data)

    def current_qpos(self):
        """Return the controlled joint positions."""
        return self.data.qpos[self.index_dof_ids].copy()

    def tracking_error(self):
        """Return qpos - target_qpos for the controlled joints."""
        return self.current_qpos() - self.target_qpos

    def reset_esn(self):
        """Reset the target q/qdot ESN controller state."""
        self.controller.reset()

    def set_force_profile(self, mode, min_g, max_g, period_s):
        """Set the contact-force generator parameters."""
        self.force_profile_mode = str(mode)
        self.force_min_g = float(min(min_g, max_g))
        self.force_max_g = float(max(min_g, max_g))
        self.force_period_s = max(float(period_s), 1e-6)
        self.force_profile_start_time = float(self.data.time)

    def current_contact_force_g(self):
        """Return the manual or generated contact force in gram-force."""
        mode = self.force_profile_mode
        if mode == "Manual":
            return float(self.contact_force_g)

        min_g = float(self.force_min_g)
        max_g = float(self.force_max_g)
        span = max_g - min_g
        if abs(span) < 1e-12:
            return min_g

        phase = (
            (float(self.data.time) - self.force_profile_start_time)
            / max(self.force_period_s, 1e-6)
        ) % 1.0
        if mode == "Sine":
            value01 = 0.5 - 0.5 * np.cos(2.0 * np.pi * phase)
        elif mode == "Triangle":
            value01 = 2.0 * phase if phase < 0.5 else 2.0 * (1.0 - phase)
        elif mode == "Square":
            value01 = 0.0 if phase < 0.5 else 1.0
        elif mode == "Ramp":
            value01 = phase
        else:
            value01 = 0.0
        return min_g + span * value01


class TargetContactPatchSimulationWorker:
    """Run MuJoCo and its passive viewer outside the Qt event loop."""

    def __init__(self, simulation, launch_viewer=True):
        self.simulation = simulation
        self.launch_viewer = launch_viewer
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        """Start the simulation worker thread."""
        self.thread.start()

    def stop(self):
        """Stop simulation stepping and wait briefly for the worker."""
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

    def set_contact_force_g(self, value):
        """Set the GUI-controlled contact force in gram-force."""
        with self.state_lock:
            self.simulation.contact_force_g = float(value)

    def set_force_profile(self, mode, min_g, max_g, period_s):
        """Set the GUI-controlled force profile."""
        with self.state_lock:
            self.simulation.set_force_profile(mode, min_g, max_g, period_s)

    def set_target_state(self, target_qpos, target_qvel):
        """Set the runtime target q/qdot state."""
        with self.state_lock:
            self.simulation.target_qpos[:] = target_qpos
            self.simulation.target_qvel[:] = target_qvel

    def set_horizon(self, value):
        """Set the ESN prediction horizon."""
        with self.state_lock:
            self.simulation.controller.horizon = int(value)

    def reset_esn(self):
        """Reset the ESN controller."""
        with self.state_lock:
            self.simulation.reset_esn()

    def snapshot(self):
        """Return a compact copy of the current simulation status."""
        with self.state_lock:
            contact_force_g = float(self.simulation.contact_force_g)
            return {
                "contact_force_g": contact_force_g,
                "contact_force_n": contact_force_g_to_n(contact_force_g),
                "contact_torque": self.simulation.contact_patch.index_contact_torque.copy(),
                "esn_tau": self.simulation.controller.last_tau.copy(),
                "point_count": int(len(self.simulation.contact_patch.points)),
                "qfrc_norm": float(np.linalg.norm(self.simulation.data.qfrc_applied)),
                "tracking_error": float(np.linalg.norm(self.simulation.tracking_error())),
                "force_profile_mode": self.simulation.force_profile_mode,
            }

    def _step_once(self):
        with self.state_lock:
            self.simulation.step()

    def _run(self):
        try:
            if self.launch_viewer:
                self._run_with_viewer()
            else:
                self._run_without_viewer()
        except Exception as exc:
            self.error = exc
            self.stop_event.set()

    def _run_with_viewer(self):
        with mujoco.viewer.launch_passive(
            self.simulation.model,
            self.simulation.data,
        ) as viewer:
            while not self.stop_event.is_set() and viewer.is_running():
                step_start = time.time()
                with viewer.lock():
                    self._step_once()
                    self.simulation.contact_visualizer.draw(
                        viewer,
                        self.simulation.contact_patch,
                    )
                viewer.sync()
                self._sleep_to_realtime(step_start)
        self.stop_event.set()

    def _run_without_viewer(self):
        while not self.stop_event.is_set():
            step_start = time.time()
            self._step_once()
            self._sleep_to_realtime(step_start)

    def _sleep_to_realtime(self, step_start):
        sleep_time = self.simulation.model.opt.timestep - (time.time() - step_start)
        if sleep_time > 0.0:
            time.sleep(sleep_time)


class ESNTargetContactPatchControlWindow(QMainWindow):
    """GUI for target q/qdot ESN control with injected patch forces."""

    def __init__(self, worker, refresh_ms, contact_force_max_g):
        super().__init__()
        self.worker = worker
        self.contact_force_max_g = float(contact_force_max_g)
        self.target_qpos_inputs = []
        self.target_qvel_inputs = []

        self.setWindowTitle("L20a ESN Target Contact-Patch Force Control")
        self.setMinimumWidth(540)
        self.resize(640, 900)
        self.setCentralWidget(self._build_content())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(max(int(refresh_ms), 20))

    def _build_content(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        root_layout = QVBoxLayout(content)

        force_group = QGroupBox("Contact Force")
        force_layout = QFormLayout(force_group)
        self.contact_force_input = QDoubleSpinBox()
        self.contact_force_input.setDecimals(3)
        self.contact_force_input.setSingleStep(1.0)
        self.contact_force_input.setRange(-self.contact_force_max_g, self.contact_force_max_g)
        self.contact_force_input.setSuffix(" g")
        self.contact_force_input.setValue(float(self.worker.simulation.contact_force_g))
        self.contact_force_input.valueChanged.connect(self.update_contact_force)
        self.contact_force_n_label = QLabel("0.0000 N")
        self.contact_force_n_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        force_layout.addRow("Contact force:", self.contact_force_input)
        force_layout.addRow("Force in N:", self.contact_force_n_label)
        root_layout.addWidget(force_group)

        profile_group = QGroupBox("Force Profile")
        profile_layout = QFormLayout(profile_group)
        self.force_profile_input = QComboBox()
        self.force_profile_input.addItems(["Manual", "Sine", "Triangle", "Square", "Ramp"])
        self.force_profile_input.setCurrentText(self.worker.simulation.force_profile_mode)
        self.force_profile_input.currentTextChanged.connect(self.update_force_profile)
        self.force_min_input = QDoubleSpinBox()
        self.force_min_input.setDecimals(3)
        self.force_min_input.setSingleStep(1.0)
        self.force_min_input.setRange(-self.contact_force_max_g, self.contact_force_max_g)
        self.force_min_input.setSuffix(" g")
        self.force_min_input.setValue(float(self.worker.simulation.force_min_g))
        self.force_min_input.valueChanged.connect(self.update_force_profile)
        self.force_max_input = QDoubleSpinBox()
        self.force_max_input.setDecimals(3)
        self.force_max_input.setSingleStep(1.0)
        self.force_max_input.setRange(-self.contact_force_max_g, self.contact_force_max_g)
        self.force_max_input.setSuffix(" g")
        self.force_max_input.setValue(float(self.worker.simulation.force_max_g))
        self.force_max_input.valueChanged.connect(self.update_force_profile)
        self.force_period_input = QDoubleSpinBox()
        self.force_period_input.setDecimals(3)
        self.force_period_input.setSingleStep(0.1)
        self.force_period_input.setRange(0.01, 120.0)
        self.force_period_input.setSuffix(" s")
        self.force_period_input.setValue(float(self.worker.simulation.force_period_s))
        self.force_period_input.valueChanged.connect(self.update_force_profile)
        profile_layout.addRow("Mode:", self.force_profile_input)
        profile_layout.addRow("Min force:", self.force_min_input)
        profile_layout.addRow("Max force:", self.force_max_input)
        profile_layout.addRow("Period:", self.force_period_input)
        root_layout.addWidget(profile_group)
        self.update_force_profile()

        target_group = QGroupBox("Target Joint State")
        target_layout = QFormLayout(target_group)
        for i, value in enumerate(self.worker.simulation.target_qpos):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setSingleStep(0.01)
            spin_box.setRange(0.0, 1.5)
            spin_box.setSuffix(" rad")
            spin_box.setValue(float(value))
            spin_box.valueChanged.connect(self.update_target_state)
            self.target_qpos_inputs.append(spin_box)
            target_layout.addRow(f"target_qpos[{i}]:", spin_box)
        for i, value in enumerate(self.worker.simulation.target_qvel):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setSingleStep(0.1)
            spin_box.setRange(-2.0, 2.0)
            spin_box.setSuffix(" rad/s")
            spin_box.setValue(float(value))
            spin_box.valueChanged.connect(self.update_target_state)
            self.target_qvel_inputs.append(spin_box)
            target_layout.addRow(f"target_qvel[{i}]:", spin_box)
        root_layout.addWidget(target_group)

        esn_group = QGroupBox("ESN Controller")
        esn_layout = QFormLayout(esn_group)
        esn_ctrl = self.worker.simulation.controller
        self.horizon_input = QSpinBox()
        horizons = esn_ctrl.controller.horizons
        self.horizon_input.setRange(min(horizons), max(horizons))
        self.horizon_input.setValue(esn_ctrl.horizon)
        self.horizon_input.valueChanged.connect(self.update_horizon)
        esn_layout.addRow("Horizon:", self.horizon_input)
        root_layout.addWidget(esn_group)

        self.reservoir_widget = ReservoirHeatmapWidget(esn_ctrl.controller)
        root_layout.addWidget(self.reservoir_widget)

        button_layout = QHBoxLayout()
        reset_force_button = QPushButton("接触力清零")
        reset_force_button.clicked.connect(self.reset_contact_force)
        button_layout.addWidget(reset_force_button)
        reset_esn_button = QPushButton("重置 ESN")
        reset_esn_button.clicked.connect(self.reset_esn)
        button_layout.addWidget(reset_esn_button)
        root_layout.addLayout(button_layout)

        state_group = QGroupBox("Contact Patch State")
        state_layout = QFormLayout(state_group)
        self.point_count_label = QLabel("0")
        self.point_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.esn_tau_label = QLabel("0.0000, 0.0000, 0.0000")
        self.esn_tau_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.contact_torque_label = QLabel("0.0000, 0.0000, 0.0000, 0.0000")
        self.contact_torque_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tracking_error_label = QLabel("0.00000 rad")
        self.tracking_error_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.qfrc_norm_label = QLabel("0.0000")
        self.qfrc_norm_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        state_layout.addRow("Patch points:", self.point_count_label)
        state_layout.addRow("ESN torque:", self.esn_tau_label)
        state_layout.addRow("Contact torque:", self.contact_torque_label)
        state_layout.addRow("Tracking error:", self.tracking_error_label)
        state_layout.addRow("Applied qfrc norm:", self.qfrc_norm_label)
        root_layout.addWidget(state_group)
        root_layout.addStretch(1)

        scroll_area.setWidget(content)
        return scroll_area

    def update_contact_force(self, value):
        """Update the injected contact force from the GUI spin box."""
        self.worker.set_contact_force_g(value)
        self.contact_force_n_label.setText(f"{contact_force_g_to_n(value):.4f} N")

    def update_force_profile(self, *_args):
        """Update the automatic contact-force profile parameters."""
        mode = self.force_profile_input.currentText()
        self.contact_force_input.setEnabled(mode == "Manual")
        self.worker.set_force_profile(
            mode,
            self.force_min_input.value(),
            self.force_max_input.value(),
            self.force_period_input.value(),
        )

    def update_target_state(self, *_args):
        """Update runtime target q/qdot values from the spin boxes."""
        target_qpos = [spin_box.value() for spin_box in self.target_qpos_inputs]
        target_qvel = [spin_box.value() for spin_box in self.target_qvel_inputs]
        self.worker.set_target_state(target_qpos, target_qvel)

    def update_horizon(self, value):
        """Update the ESN prediction horizon at runtime."""
        self.worker.set_horizon(value)

    def reset_contact_force(self, *_args):
        """Set the injected contact force to zero."""
        self.force_profile_input.setCurrentText("Manual")
        self.contact_force_input.setValue(0.0)

    def reset_esn(self, *_args):
        """Reset the ESN reservoir and filters."""
        self.worker.reset_esn()

    def refresh_status(self):
        """Refresh displayed contact-patch state and worker health."""
        if self.worker.error is not None:
            self.point_count_label.setText(f"Error: {self.worker.error}")
            self.refresh_timer.stop()
            return

        snapshot = self.worker.snapshot()
        self.contact_force_input.blockSignals(True)
        self.contact_force_input.setValue(snapshot["contact_force_g"])
        self.contact_force_input.blockSignals(False)
        self.contact_force_n_label.setText(f"{snapshot['contact_force_n']:.4f} N")
        self.point_count_label.setText(str(snapshot["point_count"]))
        self.esn_tau_label.setText(
            ", ".join(f"{value:.4f}" for value in snapshot["esn_tau"])
        )
        self.contact_torque_label.setText(
            ", ".join(f"{value:.4f}" for value in snapshot["contact_torque"])
        )
        self.tracking_error_label.setText(f"{snapshot['tracking_error']:.5f} rad")
        self.qfrc_norm_label.setText(f"{snapshot['qfrc_norm']:.4f}")
        self.reservoir_widget.update_from_esn()

    def closeEvent(self, event):
        """Stop the simulation thread when the GUI closes."""
        self.refresh_timer.stop()
        self.worker.stop()
        event.accept()


def build_esn_target_contact_patch_simulation(args):
    """Build the target q/qdot ESN simulation with contact-patch injection."""
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    model.dof_damping[:] = args.damping
    model.opt.gravity[:] = np.array([0.0, 0.0, args.gravity_z], dtype=float)

    if args.solref is not None or args.solimp is not None:
        solref = args.solref if args.solref is not None else [0.02, 1.0]
        solimp = args.solimp if args.solimp is not None else [0.9, 0.95, 0.001]
        _apply_contact_params(model, solref, solimp)

    joint_names, initial_command = load_finger_frame(
        args.finger_vec,
        args.initial_frame,
    )
    initial_command = clip_to_joint_ranges(model, joint_names, initial_command)
    mapper = PositionCommandMapper(model, joint_names)
    mapper.apply_qpos(data, initial_command)
    data.ctrl[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    esn = ESNTorqueController(
        model_path=args.esn_model,
        torque_limit=args.torque_limit,
        torque_rate_limit=args.torque_rate_limit,
        filter_cutoff_hz=args.filter_cutoff_hz,
    )
    controller = ControlRateAdapter(
        esn,
        sim_dt=model.opt.timestep,
        horizon=args.horizon,
    )

    esn_joint_names = [f"index_joint{i}" for i in esn.joint_ids]
    index_dof_ids = []
    for name in esn_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"ESN joint not found in model: {name}")
        index_dof_ids.append(int(model.jnt_dofadr[jid]))
    index_actuator_ids = _find_actuator_ids_for_joints(model, esn_joint_names)

    index_joint_names = ["index_joint0", "index_joint1", "index_joint2", "index_joint3"]
    index_patch_dof_ids = []
    for name in index_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"index joint not found in model: {name}")
        index_patch_dof_ids.append(int(model.jnt_dofadr[jid]))

    contact_patch = ContactPatchCalculator(
        model,
        data,
        args.contact_patch_link,
        args.contact_patch_mesh,
        args.contact_patch_radius,
        args.contact_patch_update_hz,
        args.contact_patch_max_points,
        args.contact_force_sigma,
        index_joint_names,
        index_patch_dof_ids,
    )
    contact_visualizer = ContactVisualizer(args.contact_force_arrow_scale)

    target_qpos = np.asarray(args.target_qpos, dtype=float).reshape(3).copy()
    target_qvel = np.asarray(args.target_qvel, dtype=float).reshape(3).copy()

    return ESNTargetContactPatchSimulation(
        model,
        data,
        contact_patch,
        contact_visualizer,
        controller,
        target_qpos,
        target_qvel,
        args.contact_force_g,
        index_dof_ids,
        index_actuator_ids,
        None,
        args.force_profile,
        args.force_min_g,
        args.force_max_g,
        args.force_period,
    )


def add_esn_target_contact_patch_arguments(parser):
    """Add target ESN contact-patch GUI arguments to a parser."""
    parser.add_argument("--model-xml", type=Path, default=default_torque_model_xml())
    parser.add_argument("--esn-model", type=Path, default=default_esn_model_path())
    parser.add_argument("--finger-vec", type=Path, default=default_finger_vec())
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--torque-limit", type=float, default=2.0)
    parser.add_argument("--torque-rate-limit", type=float, default=None)
    parser.add_argument("--filter-cutoff-hz", type=float, default=None)
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument(
        "--target-qpos",
        type=float,
        nargs=3,
        default=[0.332, 0.254, 0.271],
        metavar=("Q1", "Q2", "Q3"),
        help="Target joint positions for the controlled joints in rad.",
    )
    parser.add_argument(
        "--target-qvel",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DQ1", "DQ2", "DQ3"),
        help="Target joint velocities for the controlled joints in rad/s.",
    )
    parser.add_argument("--contact-force-g", type=float, default=0.0)
    parser.add_argument("--contact-force-max-g", type=float, default=1000.0)
    parser.add_argument(
        "--force-profile",
        choices=["Manual", "Sine", "Triangle", "Square", "Ramp"],
        default="Manual",
    )
    parser.add_argument("--force-min-g", type=float, default=0.0)
    parser.add_argument("--force-max-g", type=float, default=100.0)
    parser.add_argument("--force-period", type=float, default=4.0)
    parser.add_argument("--contact-patch-link", default="index_link3")
    parser.add_argument("--contact-patch-mesh", default="index_link3")
    parser.add_argument("--contact-patch-radius", type=float, default=0.004)
    parser.add_argument("--contact-patch-update-hz", type=float, default=20.0)
    parser.add_argument("--contact-patch-max-points", type=int, default=120)
    parser.add_argument("--contact-force-sigma", type=float, default=0.002)
    parser.add_argument("--contact-force-arrow-scale", type=float, default=0.2)
    parser.add_argument(
        "--solref",
        type=float,
        nargs=2,
        default=None,
        metavar=("TIMECONST", "DAMPRATIO"),
        help="Contact solref [time_constant, damping_ratio]. Default: model XML values.",
    )
    parser.add_argument(
        "--solimp",
        type=float,
        nargs=3,
        default=None,
        metavar=("DMIN", "DMAX", "WIDTH"),
        help="Contact solimp [dmin, dmax, width]. Default: model XML values.",
    )
    parser.add_argument(
        "--gravity-z",
        type=float,
        default=0.0,
        help="World Z gravity. Defaults to 0.0 so no-contact motion stays quiet.",
    )
    parser.add_argument("--refresh-ms", type=int, default=100)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run the Qt controller without launching the MuJoCo viewer.",
    )
    return parser


def parse_args(argv):
    """Parse target ESN contact-patch GUI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactively control a target q/qdot ESN hand simulation with "
            "GUI-injected contact-patch forces."
        )
    )
    add_esn_target_contact_patch_arguments(parser)
    return parser.parse_args(argv)


def main():
    """Launch the target ESN contact-patch force GUI."""
    args = parse_args(sys.argv[1:])
    simulation = build_esn_target_contact_patch_simulation(args)
    worker = TargetContactPatchSimulationWorker(
        simulation,
        launch_viewer=not args.no_viewer,
    )

    app = QApplication(sys.argv[:1])
    window = ESNTargetContactPatchControlWindow(
        worker,
        args.refresh_ms,
        args.contact_force_max_g,
    )
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
