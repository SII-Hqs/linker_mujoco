#!/usr/bin/env python3
"""PyQt5 GUI for PID position-tracking control with injected contact-patch forces.

This script uses a joint-space PID controller to track target joint positions
(same control objective as the ESN target controller), while external forces are
injected via the same ContactPatchCalculator mechanism.  This enables a fair
comparison between PID position control and ESN position control under identical
contact conditions.
"""

import argparse
import pickle
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
    QAbstractSpinBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from ..position_command_mapper import PositionCommandMapper
    from .contact_control_demo import default_finger_vec, default_model_xml, load_finger_frame
    from .contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from .contact_patch import ContactPatchCalculator
    from .contact_visualizer import ContactVisualizer
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo import (
        default_finger_vec,
        default_model_xml,
        load_finger_frame,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from linker_hand_mujoco_ros2.contact.contact_patch import ContactPatchCalculator
    from linker_hand_mujoco_ros2.contact.contact_visualizer import ContactVisualizer
    from linker_hand_mujoco_ros2.position_command_mapper import PositionCommandMapper


def contact_force_g_to_n(contact_force_g):
    """Convert gram-force to newtons."""
    return float(contact_force_g) * 9.80665 / 1000.0


def _apply_contact_params(model, solref, solimp):
    """Override solref/solimp for all geom pairs at runtime."""
    solref = np.asarray(solref, dtype=float)
    solimp = np.asarray(solimp, dtype=float)
    for geom_id in range(model.ngeom):
        model.geom_solref[geom_id, : len(solref)] = solref
        model.geom_solimp[geom_id, : len(solimp)] = solimp


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


def _load_index_target_trajectory(model, trajectory_pkl, trajectory_fps):
    """Load index_joint1/2/3 target q/qdot arrays from an L20a joints pkl."""
    trajectory_fps = float(trajectory_fps)
    if trajectory_fps <= 0.0:
        raise ValueError("trajectory_fps must be positive")

    with Path(trajectory_pkl).open("rb") as file:
        payload = pickle.load(file)

    joint_data = np.asarray(payload["data"], dtype=float)
    if joint_data.ndim != 2 or joint_data.shape[0] == 0:
        raise ValueError("trajectory data must be a non-empty 2-D array")

    target_joint_names = ["index_joint1", "index_joint2", "index_joint3"]
    joint_names = payload.get("meta_data", {}).get("joint_names", [])
    if all(name in joint_names for name in target_joint_names):
        target_columns = [joint_names.index(name) for name in target_joint_names]
    else:
        if joint_data.shape[1] < 4:
            raise ValueError("trajectory data must contain at least 4 joint columns")
        target_columns = [1, 2, 3]

    target_qpos = joint_data[:, target_columns].copy()
    target_qpos = np.vstack([
        clip_to_joint_ranges(model, target_joint_names, frame)
        for frame in target_qpos
    ])
    target_qvel = np.zeros_like(target_qpos)
    if len(target_qpos) > 1:
        target_qvel[:-1] = np.diff(target_qpos, axis=0) * trajectory_fps
    return target_qpos, target_qvel


def _default_record_path():
    """Return an auto-generated runtime record path under the current workspace."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return Path.cwd() / "runs" / "contact_patch_records" / f"hybrid_{stamp}.npz"


class JointPositionPIDController:
    """PID controller that tracks target joint positions and outputs torques."""

    def __init__(self, ndof, kp, ki, kd, torque_limit, integral_limit):
        self.ndof = int(ndof)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.torque_limit = float(torque_limit)
        self.integral_limit = float(integral_limit)
        self.integral = np.zeros(self.ndof, dtype=float)
        self.prev_error = np.zeros(self.ndof, dtype=float)
        self.last_tau = np.zeros(self.ndof, dtype=float)

    def reset(self):
        """Reset PID internal state."""
        self.integral[:] = 0.0
        self.prev_error[:] = 0.0
        self.last_tau[:] = 0.0

    def compute(self, qpos, qvel, target_qpos, target_qvel, dt):
        """Compute torque command from position/velocity error."""
        dt = max(float(dt), 1e-9)
        pos_error = target_qpos - qpos
        vel_error = target_qvel - qvel

        self.integral += pos_error * dt
        self.integral = np.clip(
            self.integral, -self.integral_limit, self.integral_limit
        )

        tau = (
            self.kp * pos_error
            + self.ki * self.integral
            + self.kd * vel_error
        )
        tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        self.prev_error = pos_error.copy()
        self.last_tau = tau.copy()
        return tau

@dataclass
class PIDContactPatchSimulation:
    """PID position-tracking control with virtual contact-patch force injection."""

    model: object
    data: object
    contact_patch: ContactPatchCalculator
    contact_visualizer: ContactVisualizer
    controller: JointPositionPIDController
    target_qpos: np.ndarray
    target_qvel: np.ndarray
    index_dof_ids: List[int] = field(default_factory=list)
    index_actuator_ids: List[int] = field(default_factory=list)
    contact_force_g: float = 0.0
    last_contact_force_g: float | None = None
    force_profile_mode: str = "Manual"
    force_min_g: float = 0.0
    force_max_g: float = 100.0
    force_period_s: float = 4.0
    force_profile_start_time: float = 0.0
    trajectory_qpos: np.ndarray | None = None
    trajectory_qvel: np.ndarray | None = None
    trajectory_fps: float = 30.0
    trajectory_loop: bool = False
    trajectory_frame: int = 0
    trajectory_playing: bool = False
    trajectory_start_time: float = 0.0
    record_enabled: bool = False
    record_path: Path | None = None
    record_every: int = 1
    record_step_count: int = 0
    record_saved: bool = False
    records: dict = field(default_factory=dict)

    def __post_init__(self):
        """Initialize optional trajectory playback state."""
        self.record_every = max(int(self.record_every), 1)
        if self.record_enabled:
            self.record_path = Path(self.record_path or _default_record_path())
            self._init_records()
        if self.trajectory_active:
            self.trajectory_fps = float(self.trajectory_fps)
            if self.trajectory_fps <= 0.0:
                raise ValueError("trajectory_fps must be positive")
            if self.trajectory_qpos.shape != self.trajectory_qvel.shape:
                raise ValueError("trajectory qpos/qvel arrays must have matching shapes")
            if self.trajectory_qpos.ndim != 2 or self.trajectory_qpos.shape[1] != 3:
                raise ValueError("trajectory arrays must have shape (N, 3)")
            self.trajectory_playing = True
            self.trajectory_start_time = float(self.data.time)
            self._apply_trajectory_frame(0)

    @property
    def trajectory_active(self):
        """Return True when target trajectory playback is configured."""
        return self.trajectory_qpos is not None and self.trajectory_qvel is not None

    def step(self):
        """Advance simulation: inject patch force, compute PID torque, step."""
        self.update_trajectory_target()
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

        qpos = self.data.qpos[self.index_dof_ids]
        qvel = self.data.qvel[self.index_dof_ids]
        tau = self.controller.compute(
            qpos, qvel, self.target_qpos, self.target_qvel, self.model.opt.timestep
        )

        self.data.ctrl[:] = 0.0
        for i, act_id in enumerate(self.index_actuator_ids):
            self.data.ctrl[act_id] = tau[i]

        mujoco.mj_step(self.model, self.data)
        self._record_step(tau)

    def current_qpos(self):
        """Return the controlled joint positions."""
        return self.data.qpos[self.index_dof_ids].copy()

    def tracking_error(self):
        """Return qpos - target_qpos for the controlled joints."""
        return self.current_qpos() - self.target_qpos

    def _apply_trajectory_frame(self, frame):
        """Set target q/qdot from one trajectory frame."""
        if not self.trajectory_active:
            return
        frame = int(np.clip(frame, 0, len(self.trajectory_qpos) - 1))
        self.trajectory_frame = frame
        self.target_qpos[:] = self.trajectory_qpos[frame]
        self.target_qvel[:] = self.trajectory_qvel[frame]

    def update_trajectory_target(self):
        """Advance trajectory target according to MuJoCo simulation time."""
        if not self.trajectory_active:
            return
        if self.trajectory_playing:
            elapsed = max(0.0, float(self.data.time) - self.trajectory_start_time)
            raw_frame = int(elapsed * self.trajectory_fps)
            frame_count = len(self.trajectory_qpos)
            if self.trajectory_loop:
                frame = raw_frame % frame_count
            else:
                frame = min(raw_frame, frame_count - 1)
                if raw_frame >= frame_count - 1:
                    self.trajectory_playing = False
            self._apply_trajectory_frame(frame)

    def set_trajectory_playing(self, playing):
        """Pause or resume trajectory playback."""
        if not self.trajectory_active:
            return
        self.update_trajectory_target()
        playing = bool(playing)
        if playing and not self.trajectory_playing:
            self.trajectory_start_time = (
                float(self.data.time) - self.trajectory_frame / self.trajectory_fps
            )
        self.trajectory_playing = playing

    def toggle_trajectory_playing(self):
        """Toggle trajectory playback and return the new playing state."""
        self.set_trajectory_playing(not self.trajectory_playing)
        return self.trajectory_playing

    def reset_trajectory(self):
        """Restart trajectory playback from frame 0."""
        if not self.trajectory_active:
            return
        self.trajectory_start_time = float(self.data.time)
        self.trajectory_playing = True
        self._apply_trajectory_frame(0)

    def trajectory_snapshot(self):
        """Return a compact trajectory playback status dictionary."""
        frame_count = 0 if not self.trajectory_active else len(self.trajectory_qpos)
        return {
            "active": self.trajectory_active,
            "frame": self.trajectory_frame,
            "frame_count": frame_count,
            "playing": self.trajectory_playing,
            "loop": self.trajectory_loop,
            "fps": self.trajectory_fps,
        }

    def _init_records(self):
        """Initialize per-step runtime recording buffers."""
        self.records = {
            "time": [],
            "trajectory_frame": [],
            "trajectory_playing": [],
            "target_qpos": [],
            "target_qvel": [],
            "qpos": [],
            "qvel": [],
            "pid_tau": [],
            "contact_force_g": [],
            "contact_force_n": [],
            "contact_torque": [],
            "qfrc_norm": [],
            "tracking_error": [],
        }

    def _record_step(self, tau):
        """Record one simulation step into memory for later saving."""
        if not self.record_enabled:
            return
        self.record_step_count += 1
        if self.record_step_count % self.record_every != 0:
            return

        qpos = self.current_qpos()
        qvel = self.data.qvel[self.index_dof_ids].copy()
        tracking_error = qpos - self.target_qpos
        self.records["time"].append(float(self.data.time))
        self.records["trajectory_frame"].append(int(self.trajectory_frame))
        self.records["trajectory_playing"].append(bool(self.trajectory_playing))
        self.records["target_qpos"].append(self.target_qpos.copy())
        self.records["target_qvel"].append(self.target_qvel.copy())
        self.records["qpos"].append(qpos)
        self.records["qvel"].append(qvel)
        self.records["pid_tau"].append(np.asarray(tau, dtype=float).copy())
        self.records["contact_force_g"].append(float(self.contact_force_g))
        self.records["contact_force_n"].append(contact_force_g_to_n(self.contact_force_g))
        self.records["contact_torque"].append(
            self.contact_patch.index_contact_torque.copy()
        )
        self.records["qfrc_norm"].append(float(np.linalg.norm(self.data.qfrc_applied)))
        self.records["tracking_error"].append(tracking_error)

    def save_records(self):
        """Save accumulated runtime records as a compressed NumPy archive."""
        if not self.record_enabled or self.record_saved:
            return None
        self.record_saved = True
        if not self.records or not self.records["time"]:
            return None

        path = Path(self.record_path or _default_record_path())
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: np.asarray(values) for name, values in self.records.items()}
        np.savez_compressed(
            path,
            **arrays,
            sim_dt=np.asarray(float(self.model.opt.timestep)),
            trajectory_fps=np.asarray(float(self.trajectory_fps)),
            trajectory_loop=np.asarray(bool(self.trajectory_loop)),
            record_every=np.asarray(int(self.record_every)),
        )
        self.record_path = path
        return path

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


class PIDContactPatchWorker:
    """Run MuJoCo and its passive viewer outside the Qt event loop."""

    def __init__(self, simulation, launch_viewer=True):
        self.simulation = simulation
        self.launch_viewer = launch_viewer
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
        record_path = self.simulation.save_records()
        if record_path is not None:
            print(f"Saved hybrid contact-patch record: {record_path}", flush=True)

    def set_contact_force_g(self, value):
        with self.state_lock:
            self.simulation.contact_force_g = float(value)

    def set_force_profile(self, mode, min_g, max_g, period_s):
        with self.state_lock:
            self.simulation.set_force_profile(mode, min_g, max_g, period_s)

    def set_target_state(self, target_qpos, target_qvel):
        with self.state_lock:
            if self.simulation.trajectory_active:
                return
            self.simulation.target_qpos[:] = target_qpos
            self.simulation.target_qvel[:] = target_qvel

    def set_pid_gains(self, kp, ki, kd):
        with self.state_lock:
            self.simulation.controller.kp = float(kp)
            self.simulation.controller.ki = float(ki)
            self.simulation.controller.kd = float(kd)

    def reset_pid(self):
        with self.state_lock:
            self.simulation.controller.reset()

    def toggle_trajectory_playing(self):
        """Pause or resume trajectory playback."""
        with self.state_lock:
            return self.simulation.toggle_trajectory_playing()

    def reset_trajectory(self):
        """Restart trajectory playback from the first frame."""
        with self.state_lock:
            self.simulation.reset_trajectory()

    def snapshot(self):
        with self.state_lock:
            self.simulation.update_trajectory_target()
            contact_force_g = float(self.simulation.contact_force_g)
            return {
                "contact_force_g": contact_force_g,
                "contact_force_n": contact_force_g_to_n(contact_force_g),
                "contact_torque": self.simulation.contact_patch.index_contact_torque.copy(),
                "pid_tau": self.simulation.controller.last_tau.copy(),
                "point_count": int(len(self.simulation.contact_patch.points)),
                "qfrc_norm": float(np.linalg.norm(self.simulation.data.qfrc_applied)),
                "tracking_error": float(np.linalg.norm(self.simulation.tracking_error())),
                "current_qpos": self.simulation.current_qpos(),
                "force_profile_mode": self.simulation.force_profile_mode,
                "target_qpos": self.simulation.target_qpos.copy(),
                "target_qvel": self.simulation.target_qvel.copy(),
                "trajectory": self.simulation.trajectory_snapshot(),
                "record_enabled": self.simulation.record_enabled,
                "record_count": len(self.simulation.records.get("time", [])),
                "record_path": (
                    None
                    if self.simulation.record_path is None
                    else str(self.simulation.record_path)
                ),
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


class PIDContactPatchControlWindow(QMainWindow):
    """GUI for PID position-tracking control with injected patch forces."""

    def __init__(self, worker, refresh_ms, contact_force_max_g):
        super().__init__()
        self.worker = worker
        self.contact_force_max_g = float(contact_force_max_g)
        self.target_qpos_inputs = []
        self.target_qvel_inputs = []
        self.trajectory_active = self.worker.simulation.trajectory_active

        self.setWindowTitle("L20a PID Position Contact-Patch Control")
        self.setMinimumWidth(540)
        self.resize(620, 880)
        self.setCentralWidget(self._build_content())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(max(int(refresh_ms), 20))

    def _build_content(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        content = QWidget()
        root_layout = QVBoxLayout(content)

        # --- Contact Force ---
        force_group = QGroupBox("Contact Force (Patch Injection)")
        force_layout = QFormLayout(force_group)
        self.contact_force_input = QDoubleSpinBox()
        self.contact_force_input.setDecimals(3)
        self.contact_force_input.setSingleStep(1.0)
        self.contact_force_input.setRange(
            -self.contact_force_max_g, self.contact_force_max_g
        )
        self.contact_force_input.setSuffix(" g")
        self.contact_force_input.setValue(
            float(self.worker.simulation.contact_force_g)
        )
        self.contact_force_input.valueChanged.connect(self.update_contact_force)
        self.contact_force_n_label = QLabel("0.0000 N")
        self.contact_force_n_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        force_layout.addRow("Contact force:", self.contact_force_input)
        force_layout.addRow("Force in N:", self.contact_force_n_label)
        root_layout.addWidget(force_group)

        # --- Force Profile ---
        profile_group = QGroupBox("Force Profile")
        profile_layout = QFormLayout(profile_group)
        self.force_profile_input = QComboBox()
        self.force_profile_input.addItems(
            ["Manual", "Sine", "Triangle", "Square", "Ramp"]
        )
        self.force_profile_input.setCurrentText(
            self.worker.simulation.force_profile_mode
        )
        self.force_profile_input.currentTextChanged.connect(self.update_force_profile)
        self.force_min_input = QDoubleSpinBox()
        self.force_min_input.setDecimals(3)
        self.force_min_input.setSingleStep(1.0)
        self.force_min_input.setRange(
            -self.contact_force_max_g, self.contact_force_max_g
        )
        self.force_min_input.setSuffix(" g")
        self.force_min_input.setValue(float(self.worker.simulation.force_min_g))
        self.force_min_input.valueChanged.connect(self.update_force_profile)
        self.force_max_input = QDoubleSpinBox()
        self.force_max_input.setDecimals(3)
        self.force_max_input.setSingleStep(1.0)
        self.force_max_input.setRange(
            -self.contact_force_max_g, self.contact_force_max_g
        )
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

        # --- Target Joint State ---
        target_group = QGroupBox("Target Joint State")
        target_layout = QFormLayout(target_group)
        for i, value in enumerate(self.worker.simulation.target_qpos):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setSingleStep(0.01)
            spin_box.setRange(0.0, 1.5)
            if self.trajectory_active:
                qpos_values = self.worker.simulation.trajectory_qpos[:, i]
                spin_box.setRange(
                    min(0.0, float(np.min(qpos_values))),
                    max(1.5, float(np.max(qpos_values))),
                )
            spin_box.setSuffix(" rad")
            spin_box.setValue(float(value))
            if self.trajectory_active:
                spin_box.setReadOnly(True)
                spin_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin_box.valueChanged.connect(self.update_target_state)
            self.target_qpos_inputs.append(spin_box)
            target_layout.addRow(f"target_qpos[{i}]:", spin_box)
        for i, value in enumerate(self.worker.simulation.target_qvel):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setSingleStep(0.1)
            spin_box.setRange(-2.0, 2.0)
            if self.trajectory_active:
                qvel_values = self.worker.simulation.trajectory_qvel[:, i]
                spin_box.setRange(
                    min(-2.0, float(np.min(qvel_values))),
                    max(2.0, float(np.max(qvel_values))),
                )
            spin_box.setSuffix(" rad/s")
            spin_box.setValue(float(value))
            if self.trajectory_active:
                spin_box.setReadOnly(True)
                spin_box.setButtonSymbols(QAbstractSpinBox.NoButtons)
            spin_box.valueChanged.connect(self.update_target_state)
            self.target_qvel_inputs.append(spin_box)
            target_layout.addRow(f"target_qvel[{i}]:", spin_box)
        root_layout.addWidget(target_group)

        if self.trajectory_active:
            trajectory_group = QGroupBox("Trajectory")
            trajectory_layout = QFormLayout(trajectory_group)
            self.trajectory_frame_label = QLabel("0 / 0")
            self.trajectory_frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            self.trajectory_status_label = QLabel("Stopped")
            self.trajectory_status_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            trajectory_button_layout = QHBoxLayout()
            self.trajectory_pause_button = QPushButton("暂停")
            self.trajectory_pause_button.clicked.connect(self.toggle_trajectory_playback)
            trajectory_button_layout.addWidget(self.trajectory_pause_button)
            self.trajectory_reset_button = QPushButton("重置")
            self.trajectory_reset_button.clicked.connect(self.reset_trajectory_playback)
            trajectory_button_layout.addWidget(self.trajectory_reset_button)
            trajectory_layout.addRow("Frame:", self.trajectory_frame_label)
            trajectory_layout.addRow("Status:", self.trajectory_status_label)
            trajectory_layout.addRow("Controls:", trajectory_button_layout)
            root_layout.addWidget(trajectory_group)

        # --- PID Gains ---
        pid_group = QGroupBox("PID Gains")
        pid_layout = QFormLayout(pid_group)
        ctrl = self.worker.simulation.controller
        self.kp_input = QDoubleSpinBox()
        self.kp_input.setDecimals(3)
        self.kp_input.setSingleStep(0.1)
        self.kp_input.setRange(0.0, 100.0)
        self.kp_input.setValue(ctrl.kp)
        self.kp_input.valueChanged.connect(self.update_pid_gains)
        self.ki_input = QDoubleSpinBox()
        self.ki_input.setDecimals(4)
        self.ki_input.setSingleStep(0.01)
        self.ki_input.setRange(0.0, 50.0)
        self.ki_input.setValue(ctrl.ki)
        self.ki_input.valueChanged.connect(self.update_pid_gains)
        self.kd_input = QDoubleSpinBox()
        self.kd_input.setDecimals(4)
        self.kd_input.setSingleStep(0.01)
        self.kd_input.setRange(0.0, 50.0)
        self.kd_input.setValue(ctrl.kd)
        self.kd_input.valueChanged.connect(self.update_pid_gains)
        pid_layout.addRow("Kp:", self.kp_input)
        pid_layout.addRow("Ki:", self.ki_input)
        pid_layout.addRow("Kd:", self.kd_input)
        root_layout.addWidget(pid_group)

        # --- Buttons ---
        button_layout = QHBoxLayout()
        reset_force_button = QPushButton("接触力清零")
        reset_force_button.clicked.connect(self.reset_contact_force)
        button_layout.addWidget(reset_force_button)
        reset_pid_button = QPushButton("重置 PID")
        reset_pid_button.clicked.connect(self.reset_pid)
        button_layout.addWidget(reset_pid_button)
        root_layout.addLayout(button_layout)

        # --- State Display ---
        state_group = QGroupBox("Controller State")
        state_layout = QFormLayout(state_group)
        self.pid_tau_label = QLabel("0.0000, 0.0000, 0.0000")
        self.pid_tau_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tracking_error_label = QLabel("0.00000 rad")
        self.tracking_error_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.point_count_label = QLabel("0")
        self.point_count_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.contact_torque_label = QLabel("0.0000, 0.0000, 0.0000, 0.0000")
        self.contact_torque_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.qfrc_norm_label = QLabel("0.0000")
        self.qfrc_norm_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.record_label = QLabel("Disabled")
        self.record_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.record_label.setWordWrap(True)
        state_layout.addRow("PID torque:", self.pid_tau_label)
        state_layout.addRow("Tracking error:", self.tracking_error_label)
        state_layout.addRow("Patch points:", self.point_count_label)
        state_layout.addRow("Contact torque:", self.contact_torque_label)
        state_layout.addRow("Applied qfrc norm:", self.qfrc_norm_label)
        state_layout.addRow("Record:", self.record_label)
        root_layout.addWidget(state_group)
        root_layout.addStretch(1)

        scroll_area.setWidget(content)
        return scroll_area

    def update_contact_force(self, value):
        self.worker.set_contact_force_g(value)
        self.contact_force_n_label.setText(f"{contact_force_g_to_n(value):.4f} N")

    def update_force_profile(self, *_args):
        mode = self.force_profile_input.currentText()
        self.contact_force_input.setEnabled(mode == "Manual")
        self.worker.set_force_profile(
            mode,
            self.force_min_input.value(),
            self.force_max_input.value(),
            self.force_period_input.value(),
        )

    def update_target_state(self, *_args):
        if self.trajectory_active:
            return
        target_qpos = [s.value() for s in self.target_qpos_inputs]
        target_qvel = [s.value() for s in self.target_qvel_inputs]
        self.worker.set_target_state(target_qpos, target_qvel)

    def update_pid_gains(self, *_args):
        self.worker.set_pid_gains(
            self.kp_input.value(),
            self.ki_input.value(),
            self.kd_input.value(),
        )

    def reset_contact_force(self, *_args):
        self.force_profile_input.setCurrentText("Manual")
        self.contact_force_input.setValue(0.0)

    def reset_pid(self, *_args):
        self.worker.reset_pid()

    def toggle_trajectory_playback(self, *_args):
        """Pause or resume trajectory playback."""
        self.worker.toggle_trajectory_playing()
        self.refresh_status()

    def reset_trajectory_playback(self, *_args):
        """Restart trajectory playback from the first frame."""
        self.worker.reset_trajectory()
        self.refresh_status()

    def _refresh_target_spin_boxes(self, target_qpos, target_qvel):
        """Update target spin boxes without emitting GUI edit callbacks."""
        for spin_box, value in zip(self.target_qpos_inputs, target_qpos):
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        for spin_box, value in zip(self.target_qvel_inputs, target_qvel):
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)

    def refresh_status(self):
        if self.worker.error is not None:
            self.pid_tau_label.setText(f"Error: {self.worker.error}")
            self.refresh_timer.stop()
            return

        snapshot = self.worker.snapshot()
        self.contact_force_input.blockSignals(True)
        self.contact_force_input.setValue(snapshot["contact_force_g"])
        self.contact_force_input.blockSignals(False)
        self.contact_force_n_label.setText(f"{snapshot['contact_force_n']:.4f} N")
        self.pid_tau_label.setText(
            ", ".join(f"{v:.4f}" for v in snapshot["pid_tau"])
        )
        self.tracking_error_label.setText(f"{snapshot['tracking_error']:.5f} rad")
        self.point_count_label.setText(str(snapshot["point_count"]))
        self.contact_torque_label.setText(
            ", ".join(f"{v:.4f}" for v in snapshot["contact_torque"])
        )
        self.qfrc_norm_label.setText(f"{snapshot['qfrc_norm']:.4f}")
        if snapshot["record_enabled"]:
            self.record_label.setText(
                f"{snapshot['record_count']} samples -> {snapshot['record_path']}"
            )
        else:
            self.record_label.setText("Disabled")
        if self.trajectory_active:
            self._refresh_target_spin_boxes(
                snapshot["target_qpos"],
                snapshot["target_qvel"],
            )
            trajectory = snapshot["trajectory"]
            frame_count = trajectory["frame_count"]
            frame_text = "0 / 0"
            if frame_count > 0:
                frame_text = f"{trajectory['frame'] + 1} / {frame_count}"
            if not trajectory["playing"]:
                status_text = "Stopped"
            elif trajectory["loop"]:
                status_text = "Looping"
            else:
                status_text = "Playing"
            self.trajectory_frame_label.setText(frame_text)
            self.trajectory_status_label.setText(status_text)
            self.trajectory_pause_button.setText(
                "暂停" if trajectory["playing"] else "继续"
            )

    def closeEvent(self, event):
        self.refresh_timer.stop()
        self.worker.stop()
        event.accept()


def build_pid_contact_patch_simulation(args):
    """Build the PID position-tracking simulation with contact-patch injection."""
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

    # Controlled joints: index_joint1, index_joint2, index_joint3 (same as ESN)
    controlled_joint_names = ["index_joint1", "index_joint2", "index_joint3"]
    index_dof_ids = []
    for name in controlled_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"controlled joint not found in model: {name}")
        index_dof_ids.append(int(model.jnt_dofadr[jid]))
    index_actuator_ids = _find_actuator_ids_for_joints(model, controlled_joint_names)

    controller = JointPositionPIDController(
        ndof=len(controlled_joint_names),
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        torque_limit=args.torque_limit,
        integral_limit=args.integral_limit,
    )

    # Contact patch uses all 4 index joints for generalized force mapping
    index_patch_joint_names = [
        "index_joint0", "index_joint1", "index_joint2", "index_joint3"
    ]
    index_patch_dof_ids = []
    for name in index_patch_joint_names:
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
        index_patch_joint_names,
        index_patch_dof_ids,
    )
    contact_visualizer = ContactVisualizer(args.contact_force_arrow_scale)

    target_qpos = np.asarray(args.target_qpos, dtype=float).reshape(3).copy()
    target_qvel = np.asarray(args.target_qvel, dtype=float).reshape(3).copy()
    trajectory_qpos = None
    trajectory_qvel = None
    if args.trajectory_pkl is not None:
        trajectory_qpos, trajectory_qvel = _load_index_target_trajectory(
            model,
            args.trajectory_pkl,
            args.trajectory_fps,
        )
        target_qpos = trajectory_qpos[0].copy()
        target_qvel = trajectory_qvel[0].copy()
    record_enabled = (
        not args.no_record
        and (args.record_path is not None or args.trajectory_pkl is not None)
    )

    return PIDContactPatchSimulation(
        model=model,
        data=data,
        contact_patch=contact_patch,
        contact_visualizer=contact_visualizer,
        controller=controller,
        target_qpos=target_qpos,
        target_qvel=target_qvel,
        index_dof_ids=index_dof_ids,
        index_actuator_ids=index_actuator_ids,
        contact_force_g=args.contact_force_g,
        last_contact_force_g=None,
        force_profile_mode=args.force_profile,
        force_min_g=args.force_min_g,
        force_max_g=args.force_max_g,
        force_period_s=args.force_period,
        force_profile_start_time=0.0,
        trajectory_qpos=trajectory_qpos,
        trajectory_qvel=trajectory_qvel,
        trajectory_fps=args.trajectory_fps,
        trajectory_loop=args.trajectory_loop,
        record_enabled=record_enabled,
        record_path=args.record_path,
        record_every=args.record_every,
    )


def add_pid_contact_patch_arguments(parser):
    """Add PID contact-patch GUI arguments to a parser."""
    parser.add_argument("--model-xml", type=Path, default=default_torque_model_xml())
    parser.add_argument("--finger-vec", type=Path, default=default_finger_vec())
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--damping", type=float, default=0.8)
    # PID parameters
    parser.add_argument("--kp", type=float, default=5.0)
    parser.add_argument("--ki", type=float, default=0.1)
    parser.add_argument("--kd", type=float, default=0.5)
    parser.add_argument("--torque-limit", type=float, default=2.0)
    parser.add_argument("--integral-limit", type=float, default=5.0)
    parser.add_argument(
        "--target-qpos",
        type=float,
        nargs=3,
        default=[0.332, 0.254, 0.271],
        metavar=("Q1", "Q2", "Q3"),
        help="Target joint positions for index_joint1/2/3 in rad.",
    )
    parser.add_argument(
        "--target-qvel",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DQ1", "DQ2", "DQ3"),
        help="Target joint velocities for index_joint1/2/3 in rad/s.",
    )
    # Contact-patch parameters
    parser.add_argument("--contact-force-g", type=float, default=0.0)
    parser.add_argument("--contact-force-max-g", type=float, default=1000.0)
    parser.add_argument(
        "--trajectory-pkl",
        type=Path,
        default=None,
        help="Optional L20a *_joints.pkl target trajectory to replay.",
    )
    parser.add_argument(
        "--trajectory-fps",
        type=float,
        default=30.0,
        help="Target trajectory playback frame rate in frames per second.",
    )
    parser.add_argument(
        "--trajectory-loop",
        action="store_true",
        help="Loop the target trajectory instead of stopping at the last frame.",
    )
    parser.add_argument(
        "--record-path",
        type=Path,
        default=None,
        help=(
            "Output .npz path for runtime records. Defaults to "
            "runs/contact_patch_records/hybrid_<timestamp>.npz when a trajectory is used."
        ),
    )
    parser.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record every N simulation steps.",
    )
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Disable automatic runtime recording.",
    )
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
    )
    parser.add_argument(
        "--solimp",
        type=float,
        nargs=3,
        default=None,
        metavar=("DMIN", "DMAX", "WIDTH"),
    )
    parser.add_argument("--gravity-z", type=float, default=0.0)
    parser.add_argument("--refresh-ms", type=int, default=100)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run the Qt controller without launching the MuJoCo viewer.",
    )
    return parser


def parse_args(argv):
    """Parse PID contact-patch GUI arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Interactively control a PID position-tracking hand simulation with "
            "GUI-injected contact-patch forces."
        )
    )
    add_pid_contact_patch_arguments(parser)
    return parser.parse_args(argv)


def main():
    """Launch the PID position contact-patch force GUI."""
    args = parse_args(sys.argv[1:])
    simulation = build_pid_contact_patch_simulation(args)
    worker = PIDContactPatchWorker(
        simulation,
        launch_viewer=not args.no_viewer,
    )

    app = QApplication(sys.argv[:1])
    window = PIDContactPatchControlWindow(
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
