#!/usr/bin/env python3
"""Interactive PyQt5 GUI for PID target q/qdot contact tracking."""

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import mujoco
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
    QAbstractSpinBox,
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
    from ..controllers.platform import MovingPlatformController
    from ..controllers.position_mapper import PositionCommandMapper
    from .contact_control_demo import default_finger_vec, load_finger_frame
    from .contact_control_demo_esn_target import (
        _apply_contact_params,
        apply_initial_index_qpos,
        default_initial_qpos_path,
    )
    from .contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from .contact_control_gui import SimulationWorker
    from .contact_control_gui_esn import (
        MujocoTrajectoryVisualizer,
        _find_mesh_geom_id,
        _get_link3_link4_midpoint,
    )
    from .contact_control_gui_esn_target import (
        TrajectoryTrackingWindow,
        _load_target_trajectory,
    )
    from .observer import ContactObserver
    from .tracking_metrics_logger import TrackingMetricsLogger
    from .viewer_camera import ViewerCameraMixin
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn_target import (
        _apply_contact_params,
        apply_initial_index_qpos,
        default_initial_qpos_path,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui import SimulationWorker
    from linker_hand_mujoco_ros2.contact.contact_control_gui_esn import (
        MujocoTrajectoryVisualizer,
        _find_mesh_geom_id,
        _get_link3_link4_midpoint,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui_esn_target import (
        TrajectoryTrackingWindow,
        _load_target_trajectory,
    )
    from linker_hand_mujoco_ros2.contact.observer import ContactObserver
    from linker_hand_mujoco_ros2.contact.tracking_metrics_logger import (
        TrackingMetricsLogger,
    )
    from linker_hand_mujoco_ros2.contact.viewer_camera import ViewerCameraMixin
    from linker_hand_mujoco_ros2.controllers.platform import MovingPlatformController
    from linker_hand_mujoco_ros2.controllers.position_mapper import PositionCommandMapper


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


class JointPIDController:
    """PID controller that tracks target joint positions and velocities."""

    def __init__(
        self,
        ndof,
        kp,
        ki,
        kd,
        torque_limit,
        integral_limit,
        torque_rate_limit=None,
    ):
        self.ndof = int(ndof)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.torque_limit = self._as_joint_vector(torque_limit, "torque_limit")
        self.integral_limit = self._as_joint_vector(
            integral_limit,
            "integral_limit",
        )
        self.torque_rate_limit = self._as_joint_vector(
            torque_rate_limit,
            "torque_rate_limit",
        )
        self.integral = np.zeros(self.ndof, dtype=float)
        self.last_tau = np.zeros(self.ndof, dtype=float)

    def _as_joint_vector(self, value, name):
        if value is None:
            return None
        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(self.ndof, float(arr), dtype=float)
        if arr.shape != (self.ndof,):
            raise ValueError(f"{name} must be a scalar or length-{self.ndof} vector")
        return arr

    def reset(self):
        """Reset PID integral and torque memory."""
        self.integral[:] = 0.0
        self.last_tau[:] = 0.0

    def compute(self, qpos, qvel, target_qpos, target_qvel, dt):
        """Return torque command for one MuJoCo step."""
        dt = max(float(dt), 1e-9)
        qpos = np.asarray(qpos, dtype=float).reshape(self.ndof)
        qvel = np.asarray(qvel, dtype=float).reshape(self.ndof)
        target_qpos = np.asarray(target_qpos, dtype=float).reshape(self.ndof)
        target_qvel = np.asarray(target_qvel, dtype=float).reshape(self.ndof)

        pos_error = target_qpos - qpos
        vel_error = target_qvel - qvel
        self.integral += pos_error * dt
        if self.integral_limit is not None:
            self.integral = np.clip(
                self.integral,
                -self.integral_limit,
                self.integral_limit,
            )

        tau = self.kp * pos_error + self.ki * self.integral + self.kd * vel_error
        if self.torque_rate_limit is not None:
            max_delta = self.torque_rate_limit * dt
            tau = self.last_tau + np.clip(tau - self.last_tau, -max_delta, max_delta)
        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)
        self.last_tau = tau.copy()
        return tau


@dataclass
class PIDTargetContactSimulation:
    """Objects required to step PID target-tracking contact simulation."""

    model: object
    data: object
    platform: MovingPlatformController
    observer: ContactObserver
    controller: JointPIDController
    target_qpos: np.ndarray
    target_qvel: np.ndarray
    index_dof_ids: List[int] = field(default_factory=list)
    index_actuator_ids: List[int] = field(default_factory=list)

    def step(self):
        """Advance the PID closed-loop simulation."""
        self.platform.update(self.data.time)
        contact_state = self.observer.read()

        qpos = self.current_qpos()
        qvel = self.data.qvel[self.index_dof_ids]
        tau = self.controller.compute(
            qpos,
            qvel,
            self.target_qpos,
            self.target_qvel,
            self.model.opt.timestep,
        )

        self.data.ctrl[:] = 0.0
        for i, act_id in enumerate(self.index_actuator_ids):
            self.data.ctrl[act_id] = tau[i]

        mujoco.mj_step(self.model, self.data)
        return contact_state

    def current_qpos(self):
        """Return the controlled joint positions."""
        return self.data.qpos[self.index_dof_ids].copy()

    def tracking_error(self):
        """Return qpos - target_qpos for the controlled joints."""
        return self.current_qpos() - self.target_qpos


class PIDTargetTrajectoryPlayer:
    """Advance PID simulation target q/qdot ordered as index_joint1/2/3."""

    def __init__(self, simulation, target_qpos, target_qvel, fps, loop=False):
        self.simulation = simulation
        self.target_qpos = np.asarray(target_qpos, dtype=float)
        self.target_qvel = np.asarray(target_qvel, dtype=float)
        self.fps = float(fps)
        self.loop = bool(loop)
        self.frame = 0
        self.playing = True
        self.start_time = float(simulation.data.time)
        self.apply_frame(0)

    def _resolve_frame(self, frame):
        frame = int(frame)
        frame_count = len(self.target_qpos)
        if self.loop:
            return frame % frame_count
        return int(np.clip(frame, 0, frame_count - 1))

    def apply_frame(self, frame):
        """Apply one target trajectory frame."""
        frame = self._resolve_frame(frame)
        self.frame = frame
        self.simulation.target_qpos[:] = self.target_qpos[frame]
        target_qvel = self.target_qvel[frame]
        if not self.loop and not self.playing and frame == len(self.target_qvel) - 1:
            target_qvel = np.zeros_like(target_qvel)
        self.simulation.target_qvel[:] = target_qvel

    def update(self):
        """Advance target according to MuJoCo simulation time."""
        if not self.playing:
            return
        elapsed = max(0.0, float(self.simulation.data.time) - self.start_time)
        raw_frame = int(elapsed * self.fps)
        frame_count = len(self.target_qpos)
        if self.loop:
            frame = raw_frame % frame_count
        else:
            frame = min(raw_frame, frame_count - 1)
            reached_end = raw_frame >= frame_count - 1
            if reached_end:
                self.playing = False
        self.apply_frame(frame)

    def toggle_playing(self):
        """Pause or resume trajectory playback."""
        self.update()
        if not self.playing:
            self.start_time = float(self.simulation.data.time) - self.frame / self.fps
        self.playing = not self.playing
        return self.playing

    def reset(self):
        """Restart trajectory playback from frame 0."""
        self.start_time = float(self.simulation.data.time)
        self.playing = True
        self.apply_frame(0)

    def snapshot(self):
        """Return compact playback status."""
        tracking_qvel = self.target_qvel[self.frame].copy()
        if not self.loop and not self.playing and self.frame == len(self.target_qvel) - 1:
            tracking_qvel[:] = 0.0
        return {
            "frame": self.frame,
            "feature_frame": self.frame,
            "frame_count": len(self.target_qpos),
            "playing": self.playing,
            "loop": self.loop,
            "feature_offset": 0,
            "tracking_qpos": self.target_qpos[self.frame].copy(),
            "tracking_qvel": tracking_qvel,
            "feature_qpos": self.target_qpos[self.frame].copy(),
            "feature_qvel": tracking_qvel.copy(),
        }


class PIDTargetSimulationWorker(ViewerCameraMixin, SimulationWorker):
    """Simulation worker with PID target playback and fingertip overlay."""

    def __init__(
        self,
        simulation,
        trajectory_player=None,
        launch_viewer=True,
        camera_path=None,
        metrics_logger=None,
    ):
        super().__init__(simulation, launch_viewer=launch_viewer)
        self._init_camera_controls(camera_path)
        self.trajectory_player = trajectory_player
        self.metrics_logger = metrics_logger
        self.tip_body_id = mujoco.mj_name2id(
            self.simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "index_link3",
        )
        if self.tip_body_id < 0:
            raise ValueError("body not found: index_link3")
        self.tip_geom_id = _find_mesh_geom_id(self.simulation.model, "index_link4")
        self.trajectory_visualizer = MujocoTrajectoryVisualizer()

    @property
    def trajectory_active(self):
        """Return True when target trajectory playback is configured."""
        return self.trajectory_player is not None

    def _step_once(self):
        if self.trajectory_player is not None:
            self.trajectory_player.update()
        super()._step_once()
        self._log_tracking_metrics()
        tip_pos = _get_link3_link4_midpoint(
            self.simulation.data,
            self.tip_body_id,
            self.tip_geom_id,
        ).copy()
        self.trajectory_visualizer.update_position(tip_pos)

    def _log_tracking_metrics(self):
        """Record one target-tracking sample after the simulation step."""
        if self.metrics_logger is None:
            return
        if self.trajectory_player is None:
            trajectory = {
                "frame": 0,
                "tracking_qpos": self.simulation.target_qpos.copy(),
            }
        else:
            trajectory = self.trajectory_player.snapshot()
        self.metrics_logger.sample(
            self.simulation.data.time,
            trajectory["frame"],
            trajectory["tracking_qpos"],
            self.simulation.current_qpos(),
            self.simulation.controller.last_tau,
            self.simulation.model.opt.timestep,
        )

    def stop(self):
        """Stop simulation stepping and close the optional metrics logger."""
        super().stop()
        if self.metrics_logger is not None:
            self.metrics_logger.close()

    def clear_tip_trajectory(self):
        """Clear the MuJoCo viewer fingertip trajectory."""
        self.trajectory_visualizer.clear_trajectory()

    def toggle_tip_trajectory_visible(self):
        """Toggle MuJoCo fingertip trajectory visibility."""
        return self.trajectory_visualizer.toggle_visible()

    def _draw_viewer_overlays(self, viewer):
        """Draw fingertip trajectory in the MuJoCo viewer."""
        self.trajectory_visualizer.draw(viewer)

    def toggle_trajectory_playing(self):
        """Pause or resume trajectory playback."""
        if self.trajectory_player is None:
            return False
        return self.trajectory_player.toggle_playing()

    def reset_trajectory(self):
        """Restart trajectory playback from the first frame."""
        if self.trajectory_player is not None:
            self.trajectory_player.reset()

    def trajectory_snapshot(self):
        """Return target trajectory playback status."""
        if self.trajectory_player is None:
            return {
                "frame": 0,
                "feature_frame": 0,
                "frame_count": 0,
                "playing": False,
                "loop": False,
                "feature_offset": 0,
                "tracking_qpos": self.simulation.target_qpos.copy(),
                "tracking_qvel": self.simulation.target_qvel.copy(),
                "feature_qpos": self.simulation.target_qpos.copy(),
                "feature_qvel": self.simulation.target_qvel.copy(),
            }
        self.trajectory_player.update()
        return self.trajectory_player.snapshot()

    def set_pid_gains(self, kp, ki, kd):
        """Update PID gains at runtime."""
        self.simulation.controller.kp = float(kp)
        self.simulation.controller.ki = float(ki)
        self.simulation.controller.kd = float(kd)

    def reset_pid(self):
        """Reset PID integral and torque memory."""
        self.simulation.controller.reset()


class PIDTargetContactControlWindow(QMainWindow):
    """GUI for PID platform control and target tracking."""

    def __init__(self, worker, position_range, refresh_ms, plot_window):
        super().__init__()
        self.worker = worker
        self.initial_pos = self.worker.simulation.platform.initial_pos.copy()
        self.position_range = float(position_range)
        self.position_inputs = {}
        self.target_qpos_inputs = []
        self.target_qvel_inputs = []
        self.target_joint_names = ["index_joint1", "index_joint2", "index_joint3"]
        self.trajectory_active = self.worker.trajectory_active
        self.tracking_window = TrajectoryTrackingWindow(
            self.target_joint_names,
            window_s=plot_window,
        )

        self.setWindowTitle("L20a PID Target-Tracking Contact Platform Control")
        self.setMinimumWidth(520)
        self.resize(620, 820)
        self.setCentralWidget(self._build_content())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(max(int(refresh_ms), 20))

        self.reset_platform()

    def _build_content(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)

        content = QWidget()
        root_layout = QVBoxLayout(content)

        position_group = QGroupBox("Platform Position")
        position_layout = QFormLayout(position_group)
        for axis, value in zip(("X", "Y", "Z"), self.initial_pos):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(5)
            spin_box.setSingleStep(0.001)
            spin_box.setSuffix(" m")
            spin_box.setRange(
                float(value - self.position_range),
                float(value + self.position_range),
            )
            spin_box.setValue(float(value))
            spin_box.valueChanged.connect(self.update_platform_target)
            self.position_inputs[axis] = spin_box
            position_layout.addRow(f"{axis}:", spin_box)
        root_layout.addWidget(position_group)

        target_group = QGroupBox("Target Joint State")
        target_layout = QFormLayout(target_group)
        for i, value in enumerate(self.worker.simulation.target_qpos):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setSingleStep(0.01)
            spin_box.setRange(0.0, 1.5)
            if self.trajectory_active:
                qpos_values = self.worker.trajectory_player.target_qpos[:, i]
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
            target_layout.addRow(f"{self.target_joint_names[i]} qpos:", spin_box)
        for i, value in enumerate(self.worker.simulation.target_qvel):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(4)
            spin_box.setSingleStep(0.1)
            spin_box.setRange(-2.0, 2.0)
            if self.trajectory_active:
                qvel_values = self.worker.trajectory_player.target_qvel[:, i]
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
            target_layout.addRow(f"{self.target_joint_names[i]} qvel:", spin_box)
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

        ctrl = self.worker.simulation.controller
        pid_group = QGroupBox("PID Gains")
        pid_layout = QFormLayout(pid_group)
        self.kp_input = QDoubleSpinBox()
        self.kp_input.setDecimals(3)
        self.kp_input.setSingleStep(0.1)
        self.kp_input.setRange(0.0, 200.0)
        self.kp_input.setValue(ctrl.kp)
        self.kp_input.valueChanged.connect(self.update_pid_gains)
        self.ki_input = QDoubleSpinBox()
        self.ki_input.setDecimals(4)
        self.ki_input.setSingleStep(0.01)
        self.ki_input.setRange(0.0, 100.0)
        self.ki_input.setValue(ctrl.ki)
        self.ki_input.valueChanged.connect(self.update_pid_gains)
        self.kd_input = QDoubleSpinBox()
        self.kd_input.setDecimals(4)
        self.kd_input.setSingleStep(0.01)
        self.kd_input.setRange(0.0, 100.0)
        self.kd_input.setValue(ctrl.kd)
        self.kd_input.valueChanged.connect(self.update_pid_gains)
        pid_layout.addRow("Kp:", self.kp_input)
        pid_layout.addRow("Ki:", self.ki_input)
        pid_layout.addRow("Kd:", self.kd_input)
        root_layout.addWidget(pid_group)

        button_layout = QHBoxLayout()
        reset_button = QPushButton("复位")
        reset_button.clicked.connect(self.reset_platform)
        button_layout.addWidget(reset_button)
        reset_pid_button = QPushButton("重置 PID")
        reset_pid_button.clicked.connect(self.reset_pid)
        button_layout.addWidget(reset_pid_button)
        clear_trajectory_button = QPushButton("清除轨迹")
        clear_trajectory_button.clicked.connect(self.clear_trajectory)
        button_layout.addWidget(clear_trajectory_button)
        self.toggle_trajectory_button = QPushButton("隐藏轨迹")
        self.toggle_trajectory_button.clicked.connect(self.toggle_trajectory_visibility)
        button_layout.addWidget(self.toggle_trajectory_button)
        save_camera_button = QPushButton("保存视角")
        save_camera_button.clicked.connect(self.save_camera)
        button_layout.addWidget(save_camera_button)
        load_camera_button = QPushButton("加载视角")
        load_camera_button.clicked.connect(self.load_camera)
        button_layout.addWidget(load_camera_button)
        root_layout.addLayout(button_layout)

        contact_group = QGroupBox("Contact State")
        contact_layout = QFormLayout(contact_group)
        self.contact_label = QLabel("No contact")
        self.contact_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.force_label = QLabel("0.0000 N")
        self.force_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.raw_force_label = QLabel("0.0000 N")
        self.raw_force_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tau_label = QLabel("0.0000, 0.0000, 0.0000")
        self.tau_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.tracking_error_label = QLabel("0.00000 rad")
        self.tracking_error_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.geom_label = QLabel("none")
        self.geom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.geom_label.setWordWrap(True)
        contact_layout.addRow("Status:", self.contact_label)
        contact_layout.addRow("Normal force:", self.force_label)
        contact_layout.addRow("Raw force:", self.raw_force_label)
        contact_layout.addRow("PID torque:", self.tau_label)
        contact_layout.addRow("Tracking error:", self.tracking_error_label)
        contact_layout.addRow("Geoms:", self.geom_label)
        root_layout.addWidget(contact_group)
        root_layout.addStretch(1)
        scroll_area.setWidget(content)
        return scroll_area

    def update_platform_target(self, *_args):
        """Send current XYZ inputs to the simulation worker."""
        self.worker.set_platform_pos(
            self.position_inputs["X"].value(),
            self.position_inputs["Y"].value(),
            self.position_inputs["Z"].value(),
        )

    def update_target_state(self, *_args):
        """Update runtime target q/qdot values from the spin boxes."""
        if self.trajectory_active:
            return
        self.worker.simulation.target_qpos[:] = [
            spin_box.value() for spin_box in self.target_qpos_inputs
        ]
        self.worker.simulation.target_qvel[:] = [
            spin_box.value() for spin_box in self.target_qvel_inputs
        ]

    def update_pid_gains(self, *_args):
        """Update PID gains from GUI spin boxes."""
        self.worker.set_pid_gains(
            self.kp_input.value(),
            self.ki_input.value(),
            self.kd_input.value(),
        )

    def reset_platform(self, *_args):
        """Reset platform inputs and target to the initial position."""
        for axis, value in zip(("X", "Y", "Z"), self.initial_pos):
            spin_box = self.position_inputs[axis]
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        self.worker.simulation.platform.reset_target_pos()

    def reset_pid(self, *_args):
        """Reset PID integral and torque memory."""
        self.worker.reset_pid()

    def clear_trajectory(self, *_args):
        """Clear the MuJoCo viewer fingertip trajectory."""
        self.worker.clear_tip_trajectory()

    def toggle_trajectory_visibility(self, *_args):
        """Show or hide the MuJoCo viewer fingertip trajectory."""
        visible = self.worker.toggle_tip_trajectory_visible()
        self.toggle_trajectory_button.setText("隐藏轨迹" if visible else "显示轨迹")

    def save_camera(self, *_args):
        """Save current MuJoCo viewer camera."""
        try:
            path = self.worker.save_camera()
            self.contact_label.setText(f"Camera saved: {path}")
        except Exception as exc:
            self.contact_label.setText(f"Camera save failed: {exc}")

    def load_camera(self, *_args):
        """Load saved MuJoCo viewer camera."""
        try:
            path = self.worker.load_camera()
            self.contact_label.setText(f"Camera loaded: {path}")
        except Exception as exc:
            self.contact_label.setText(f"Camera load failed: {exc}")

    def toggle_trajectory_playback(self, *_args):
        """Pause or resume trajectory playback."""
        self.worker.toggle_trajectory_playing()
        self.refresh_status()

    def reset_trajectory_playback(self, *_args):
        """Restart trajectory playback from the first frame."""
        self.worker.reset_trajectory()
        self.refresh_status()

    def _refresh_target_spin_boxes(self):
        """Update target spin boxes from simulation without emitting callbacks."""
        for spin_box, value in zip(
            self.target_qpos_inputs,
            self.worker.simulation.target_qpos,
        ):
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        for spin_box, value in zip(
            self.target_qvel_inputs,
            self.worker.simulation.target_qvel,
        ):
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)

    def refresh_status(self):
        """Refresh displayed contact state and worker health."""
        if self.worker.error is not None:
            self.contact_label.setText(f"Error: {self.worker.error}")
            self.refresh_timer.stop()
            return

        state = self.worker.get_contact_state()
        self.contact_label.setText("Contact" if state.in_contact else "No contact")
        self.force_label.setText(f"{state.normal_force:.4f} N")
        self.raw_force_label.setText(f"{state.raw_normal_force:.4f} N")
        tau = self.worker.simulation.controller.last_tau
        self.tau_label.setText(", ".join(f"{t:.4f}" for t in tau))

        actual_qpos = self.worker.simulation.current_qpos()
        tracking_qpos = self.worker.simulation.target_qpos.copy()
        trajectory = None
        if self.trajectory_active:
            trajectory = self.worker.trajectory_snapshot()
            tracking_qpos = trajectory["tracking_qpos"]
        error = np.linalg.norm(actual_qpos - tracking_qpos)
        self.tracking_error_label.setText(f"{error:.5f} rad")
        self.geom_label.setText(", ".join(state.geoms) if state.geoms else "none")
        self.tracking_window.sample(
            self.worker.simulation.data.time,
            tracking_qpos,
            actual_qpos,
        )

        if self.trajectory_active:
            self._refresh_target_spin_boxes()
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
        """Stop the simulation thread when the GUI closes."""
        self.refresh_timer.stop()
        self.tracking_window.close()
        self.worker.stop()
        event.accept()


def build_pid_target_simulation(args):
    """Build and initialize the PID target contact simulation."""
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
    apply_initial_index_qpos(model, data, args.initial_qpos_pkl)
    mujoco.mj_forward(model, data)

    platform = MovingPlatformController(
        model,
        data,
        body_name="moving_platform",
        target_name=args.target,
        top_offset=args.platform_top_offset,
        stroke=args.platform_stroke,
        period=args.platform_period,
        initial_pos=[
            args.platform_initial_x,
            args.platform_initial_y,
            args.platform_initial_z,
        ],
        high_z_offset=args.high_z_offset,
        low_z_offset=args.low_z_offset,
    )
    observer = ContactObserver(
        model,
        data,
        "moving_platform_geom",
        force_clip=args.force_clip,
    )

    index_joint_names = ["index_joint1", "index_joint2", "index_joint3"]
    index_dof_ids = []
    for name in index_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"PID joint not found in model: {name}")
        index_dof_ids.append(int(model.jnt_dofadr[jid]))
    index_actuator_ids = _find_actuator_ids_for_joints(model, index_joint_names)

    target_qpos = np.asarray(args.target_qpos, dtype=float).reshape(3).copy()
    target_qvel = np.asarray(args.target_qvel, dtype=float).reshape(3).copy()
    controller = JointPIDController(
        ndof=3,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        torque_limit=args.torque_limit,
        integral_limit=args.integral_limit,
        torque_rate_limit=args.torque_rate_limit,
    )

    return PIDTargetContactSimulation(
        model,
        data,
        platform,
        observer,
        controller,
        target_qpos,
        target_qvel,
        index_dof_ids,
        index_actuator_ids,
    )


def add_pid_target_arguments(parser):
    """Add PID target GUI arguments to a parser."""
    parser.add_argument("--model-xml", type=Path, default=default_torque_model_xml())
    parser.add_argument("--finger-vec", type=Path, default=default_finger_vec())
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--initial-qpos-pkl", type=Path, default=default_initial_qpos_path())
    parser.add_argument("--target", default="index_link1")
    parser.add_argument("--platform-top-offset", type=float, default=0.075)
    parser.add_argument("--platform-initial-x", type=float, default=0.24212)
    parser.add_argument("--platform-initial-y", type=float, default=0.03445)
    parser.add_argument("--platform-initial-z", type=float, default=0.2)
    parser.add_argument("--platform-stroke", type=float, default=0.004)
    parser.add_argument("--platform-period", type=float, default=4.0)
    parser.add_argument(
        "--high-z-offset",
        type=float,
        default=0.0,
        help="Lower the highest point by this amount (meters).",
    )
    parser.add_argument(
        "--low-z-offset",
        type=float,
        default=0.0,
        help="Raise the lowest point by this amount (meters).",
    )
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
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--torque-limit", type=float, default=10.0)
    parser.add_argument("--torque-rate-limit", type=float, default=None)
    parser.add_argument("--integral-limit", type=float, default=1.0)
    parser.add_argument("--force-clip", type=float, default=5.0)
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument(
        "--solref",
        type=float,
        nargs=2,
        default=None,
        metavar=("TIMECONST", "DAMPRATIO"),
        help="Contact solref [time_constant, damping_ratio].",
    )
    parser.add_argument(
        "--solimp",
        type=float,
        nargs=3,
        default=None,
        metavar=("DMIN", "DMAX", "WIDTH"),
        help="Contact solimp [dmin, dmax, width].",
    )
    parser.add_argument(
        "--gravity-z",
        type=float,
        default=-9.81,
        help="World Z gravity in m/s^2.",
    )
    parser.add_argument(
        "--position-range",
        type=float,
        default=0.12,
        help="XYZ control range around the initial platform position in meters.",
    )
    parser.add_argument("--refresh-ms", type=int, default=100)
    parser.add_argument(
        "--trajectory-pkl",
        type=Path,
        default=None,
        help="Optional target q/qdot trajectory pkl to replay.",
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
        "--init-to-trajectory",
        action="store_true",
        help="Initialize controlled joints to the first trajectory frame.",
    )
    parser.add_argument(
        "--plot-window",
        type=float,
        default=8.0,
        help="Rolling time window in seconds for target-vs-actual qpos plots.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run the Qt controller without launching the MuJoCo viewer.",
    )
    parser.add_argument(
        "--camera-view",
        type=Path,
        default=Path("mujoco_camera_view_pid.json"),
        help="JSON file used by the save/load camera buttons.",
    )
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=None,
        help="Optional CSV path for qpos tracking and torque metrics.",
    )
    parser.add_argument(
        "--metrics-summary",
        type=Path,
        default=None,
        help="Optional JSON path for aggregate tracking metrics.",
    )
    parser.add_argument(
        "--metrics-start-time",
        type=float,
        default=None,
        help="Optional simulation time to start recording metrics.",
    )
    parser.add_argument(
        "--metrics-end-time",
        type=float,
        default=None,
        help="Optional simulation time to stop recording metrics.",
    )


def parse_args(argv):
    """Parse PID target tracking GUI arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively control PID target q/qdot contact tracking."
    )
    add_pid_target_arguments(parser)
    return parser.parse_args(argv)


def _build_metrics_metadata(args):
    """Return compact experiment metadata stored with metrics summaries."""
    return {
        "trajectory_pkl": args.trajectory_pkl,
        "trajectory_fps": args.trajectory_fps,
        "trajectory_loop": args.trajectory_loop,
        "init_to_trajectory": args.init_to_trajectory,
        "model_xml": args.model_xml,
        "kp": args.kp,
        "ki": args.ki,
        "kd": args.kd,
        "torque_limit": args.torque_limit,
        "torque_rate_limit": args.torque_rate_limit,
        "integral_limit": args.integral_limit,
        "solref": args.solref,
        "solimp": args.solimp,
        "damping": args.damping,
        "gravity_z": args.gravity_z,
    }


def main():
    """Launch the PID target tracking GUI and simulation worker."""
    args = parse_args(sys.argv[1:])
    simulation = build_pid_target_simulation(args)
    trajectory_player = None
    if args.trajectory_pkl is not None:
        target_qpos, target_qvel = _load_target_trajectory(
            args.trajectory_pkl,
            args.trajectory_fps,
        )
        if args.init_to_trajectory:
            simulation.data.qpos[simulation.index_dof_ids] = target_qpos[0]
            simulation.data.qvel[simulation.index_dof_ids] = target_qvel[0]
            mujoco.mj_forward(simulation.model, simulation.data)
        trajectory_player = PIDTargetTrajectoryPlayer(
            simulation,
            target_qpos,
            target_qvel,
            args.trajectory_fps,
            loop=args.trajectory_loop,
        )

    worker = PIDTargetSimulationWorker(
        simulation,
        trajectory_player=trajectory_player,
        launch_viewer=not args.no_viewer,
        camera_path=args.camera_view,
        metrics_logger=(
            TrackingMetricsLogger(
                "pid",
                csv_path=args.metrics_csv,
                summary_path=args.metrics_summary,
                metadata=_build_metrics_metadata(args),
                start_time=args.metrics_start_time,
                end_time=args.metrics_end_time,
            )
            if args.metrics_csv is not None or args.metrics_summary is not None
            else None
        ),
    )

    app = QApplication(sys.argv[:1])
    window = PIDTargetContactControlWindow(
        worker,
        args.position_range,
        args.refresh_ms,
        args.plot_window,
    )
    worker.start()
    window.show()
    window.tracking_window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
