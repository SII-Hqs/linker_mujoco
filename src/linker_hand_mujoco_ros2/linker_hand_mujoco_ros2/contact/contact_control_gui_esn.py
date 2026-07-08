#!/usr/bin/env python3
"""Interactive PyQt5 GUI for the L20a ESN torque-control contact demo."""

import argparse
import collections
import pickle
import sys
import threading
import time as _time
from pathlib import Path

import mujoco
import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )
    from .contact_control_gui import SimulationWorker
    from .viewer_camera import ViewerCameraMixin
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui import SimulationWorker
    from linker_hand_mujoco_ros2.contact.viewer_camera import ViewerCameraMixin


def _find_mesh_geom_id(model, mesh_name):
    """Return the first geom using the named mesh."""
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, mesh_name)
    if geom_id >= 0:
        return geom_id

    mesh_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
    if mesh_id < 0:
        raise ValueError(f"mesh not found: {mesh_name}")

    for geom_id in range(model.ngeom):
        if int(model.geom_dataid[geom_id]) == mesh_id:
            return geom_id
    raise ValueError(f"geom using mesh not found: {mesh_name}")


def _get_link3_link4_midpoint(data, link3_body_id, link4_geom_id):
    """Return the world-space midpoint between index_link3 and index_link4."""
    link3_pos = data.xpos[link3_body_id]
    link4_pos = data.geom_xpos[link4_geom_id]
    return 0.5 * (link3_pos + link4_pos)


def _set_white_viewer_background(model):
    """Use a white MuJoCo viewer background and floor for this model."""
    model.vis.rgba.haze[:] = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    floor_geom_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "floor",
    )
    if floor_geom_id >= 0:
        model.geom_matid[floor_geom_id] = -1
        model.geom_rgba[floor_geom_id] = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)


class ReservoirHeatmapWidget(QGroupBox):
    """Qt heatmap for ESN reservoir activations."""

    def __init__(self, esn, parent=None):
        super().__init__("ESN Reservoir", parent)
        self.esn = esn
        self.nr = int(esn.nr)
        self.rows = int(np.ceil(np.sqrt(self.nr)))
        self.cols = int(np.ceil(self.nr / self.rows))

        layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(360, 260)
        self.image_label.setStyleSheet("background: #111; border: 1px solid #555;")
        self.stats_label = QLabel("Nr=0 step=0 min=0.000 max=0.000")
        self.stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.image_label)
        layout.addWidget(self.stats_label)

        self.update_from_esn()

    def update_from_esn(self):
        """Read the ESN state and refresh the heatmap image."""
        flat_state = np.asarray(self.esn.x, dtype=float).reshape(-1).copy()
        padded = np.zeros(self.rows * self.cols, dtype=float)
        padded[: min(self.nr, flat_state.size)] = flat_state[: self.nr]
        grid = padded.reshape(self.rows, self.cols)
        rgb = self._activation_to_rgb(grid)
        image = QImage(
            rgb.data,
            self.cols,
            self.rows,
            3 * self.cols,
            QImage.Format_RGB888,
        ).copy()
        pixmap = QPixmap.fromImage(image).scaled(
            self.image_label.size(),
            Qt.KeepAspectRatio,
            Qt.FastTransformation,
        )
        self.image_label.setPixmap(pixmap)
        self.stats_label.setText(
            f"Nr={self.nr} step={self.esn.step_count} "
            f"min={grid.min(): .3f} max={grid.max(): .3f}"
        )

    def _activation_to_rgb(self, grid):
        """Map activation values in [-1, 1] to a blue-white-red heatmap."""
        values = np.clip(grid, -1.0, 1.0)
        rgb = np.empty((*values.shape, 3), dtype=np.uint8)

        blue = np.array([59.0, 76.0, 192.0])
        white = np.array([221.0, 221.0, 221.0])
        red = np.array([180.0, 4.0, 38.0])

        negative = values < 0.0
        neg_weight = (values[negative] + 1.0)[:, None]
        pos_weight = values[~negative][:, None]
        rgb[negative] = (blue + neg_weight * (white - blue)).astype(np.uint8)
        rgb[~negative] = (white + pos_weight * (red - white)).astype(np.uint8)
        return rgb

    def resizeEvent(self, event):
        """Keep the heatmap crisp when the window is resized."""
        super().resizeEvent(event)
        self.update_from_esn()


class MujocoTrajectoryVisualizer:
    """Draw the index fingertip trajectory in MuJoCo viewer."""

    def __init__(self, max_points=None, max_segments=1200):
        maxlen = None if max_points is None else int(max_points)
        self.positions = collections.deque(maxlen=maxlen)
        self.max_segments = int(max_segments)
        self.lock = threading.Lock()
        self.visible = True

    def update_position(self, pos):
        """Append one world-space position."""
        with self.lock:
            self.positions.append(np.asarray(pos, dtype=float).reshape(3).copy())

    def clear_trajectory(self):
        """Clear all stored trajectory points."""
        with self.lock:
            self.positions.clear()

    def toggle_visible(self):
        """Toggle trajectory visibility and return the new state."""
        with self.lock:
            self.visible = not self.visible
            return self.visible

    def draw(self, viewer):
        """Render the trajectory into viewer.user_scn."""
        scene = viewer.user_scn
        scene.ngeom = 0

        with self.lock:
            if not self.visible or not self.positions:
                return
            points = np.asarray(self.positions, dtype=float)

        max_geoms = getattr(scene, "maxgeom", None)
        if max_geoms is None:
            max_geoms = len(scene.geoms)
        if max_geoms <= 0:
            return

        segment_limit = min(max_geoms - 1, self.max_segments, max(len(points) - 1, 0))
        if segment_limit > 0:
            if len(points) > segment_limit + 1:
                sample_idx = np.linspace(0, len(points) - 1, segment_limit + 1).astype(
                    int
                )
                draw_points = points[sample_idx]
            else:
                draw_points = points
            old_color = np.array([0.02, 0.05, 0.22, 0.35], dtype=float)
            new_color = np.array([0.0, 1.0, 1.0, 0.95], dtype=float)
            denom = max(len(draw_points) - 2, 1)
            for idx, (start, end) in enumerate(zip(draw_points[:-1], draw_points[1:])):
                if scene.ngeom >= max_geoms - 1:
                    break
                mujoco.mjv_connector(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    0.0010,
                    start,
                    end,
                )
                t = float(idx) / float(denom)
                scene.geoms[scene.ngeom].rgba[:] = old_color + t * (
                    new_color - old_color
                )
                scene.ngeom += 1

        if scene.ngeom < max_geoms:
            identity = np.eye(3).reshape(-1)
            marker_size = np.array([0.003, 0.003, 0.003], dtype=float)
            marker_color = np.array([0.0, 1.0, 1.0, 0.18], dtype=float)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                marker_size,
                points[-1],
                identity,
                marker_color,
            )
            scene.ngeom += 1


class ESNTrajectoryWorker(ViewerCameraMixin, SimulationWorker):
    """Simulation worker with MuJoCo viewer fingertip trajectory overlay."""

    def __init__(self, simulation, launch_viewer=True, camera_path=None):
        super().__init__(simulation, launch_viewer=launch_viewer)
        self._init_camera_controls(camera_path)
        self.tip_body_id = mujoco.mj_name2id(
            self.simulation.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "index_link3",
        )
        if self.tip_body_id < 0:
            raise ValueError("body not found: index_link3")
        self.tip_geom_id = _find_mesh_geom_id(self.simulation.model, "index_link4")
        self.trajectory_visualizer = MujocoTrajectoryVisualizer()

    def clear_tip_trajectory(self):
        """Clear the MuJoCo viewer trajectory."""
        self.trajectory_visualizer.clear_trajectory()

    def toggle_tip_trajectory_visible(self):
        """Toggle MuJoCo trajectory visibility."""
        return self.trajectory_visualizer.toggle_visible()

    def _step_once(self):
        super()._step_once()
        tip_pos = _get_link3_link4_midpoint(
            self.simulation.data,
            self.tip_body_id,
            self.tip_geom_id,
        ).copy()
        self.trajectory_visualizer.update_position(tip_pos)

    def _draw_viewer_overlays(self, viewer):
        """Draw fingertip trajectory in the MuJoCo viewer."""
        self.trajectory_visualizer.draw(viewer)


class TrajectoryRecorder:
    """Record index finger qpos and qvel at each simulation step."""

    def __init__(self):
        self.recording = False
        self.qpos_list = []
        self.qvel_list = []

    def start(self):
        """Start recording (clears previous data)."""
        self.qpos_list = []
        self.qvel_list = []
        self.recording = True

    def stop(self):
        """Stop recording."""
        self.recording = False

    def record(self, qpos, qvel):
        """Append one frame if recording is active."""
        if self.recording:
            self.qpos_list.append(np.asarray(qpos, dtype=float).copy())
            self.qvel_list.append(np.asarray(qvel, dtype=float).copy())

    @property
    def frame_count(self):
        return len(self.qpos_list)

    def export(self, path):
        """Save recorded trajectory to a pkl file."""
        data = {
            "qpos": np.array(self.qpos_list),
            "qvel": np.array(self.qvel_list),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)


class ESNContactControlWindow(QMainWindow):
    """GUI for XYZ platform control and ESN torque monitoring."""

    def __init__(self, worker, position_range, refresh_ms):
        super().__init__()
        self.worker = worker
        self.initial_pos = self.worker.simulation.platform.initial_pos.copy()
        self.position_range = float(position_range)
        self.position_inputs = {}
        self.recorder = TrajectoryRecorder()
        self.setWindowTitle("L20a ESN Contact Platform Control")
        self.setMinimumWidth(520)
        self.setCentralWidget(self._build_content())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(max(int(refresh_ms), 20))

        self.reset_platform()

    def _build_content(self):
        content = QWidget()
        root_layout = QVBoxLayout(content)

        # Platform position controls
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

        # ESN horizon selector
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

        # Buttons
        button_layout = QHBoxLayout()
        reset_button = QPushButton("复位")
        reset_button.clicked.connect(self.reset_platform)
        button_layout.addWidget(reset_button)
        reset_esn_button = QPushButton("重置 ESN")
        reset_esn_button.clicked.connect(self.reset_esn)
        button_layout.addWidget(reset_esn_button)
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

        # Trajectory recording
        record_group = QGroupBox("轨迹录制")
        record_layout = QFormLayout(record_group)
        self.record_frame_label = QLabel("0 帧")
        self.record_frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        record_layout.addRow("已录制:", self.record_frame_label)
        record_btn_layout = QHBoxLayout()
        self.record_start_button = QPushButton("开始录制")
        self.record_start_button.clicked.connect(self.toggle_recording)
        record_btn_layout.addWidget(self.record_start_button)
        self.record_export_button = QPushButton("导出 pkl")
        self.record_export_button.clicked.connect(self.export_trajectory)
        self.record_export_button.setEnabled(False)
        record_btn_layout.addWidget(self.record_export_button)
        record_layout.addRow(record_btn_layout)
        root_layout.addWidget(record_group)

        # Contact state display
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
        self.geom_label = QLabel("none")
        self.geom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.geom_label.setWordWrap(True)
        contact_layout.addRow("Status:", self.contact_label)
        contact_layout.addRow("Normal force:", self.force_label)
        contact_layout.addRow("Raw force:", self.raw_force_label)
        contact_layout.addRow("Torque:", self.tau_label)
        contact_layout.addRow("Geoms:", self.geom_label)
        root_layout.addWidget(contact_group)
        root_layout.addStretch(1)
        return content

    def update_platform_target(self, *_args):
        """Send current XYZ inputs to the simulation worker."""
        self.worker.set_platform_pos(
            self.position_inputs["X"].value(),
            self.position_inputs["Y"].value(),
            self.position_inputs["Z"].value(),
        )

    def update_horizon(self, value):
        """Update the ESN prediction horizon at runtime."""
        self.worker.simulation.controller.horizon = value

    def reset_platform(self, *_args):
        """Reset platform inputs and target to the initial position."""
        for axis, value in zip(("X", "Y", "Z"), self.initial_pos):
            spin_box = self.position_inputs[axis]
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        self.worker.simulation.platform.reset_target_pos()

    def reset_esn(self, *_args):
        """Reset the ESN reservoir and filters."""
        self.worker.simulation.controller.reset()

    def clear_trajectory(self, *_args):
        """Clear the MuJoCo viewer trajectory."""
        if hasattr(self.worker, "clear_tip_trajectory"):
            self.worker.clear_tip_trajectory()

    def toggle_trajectory_visibility(self, *_args):
        """Show or hide the MuJoCo viewer trajectory."""
        if not hasattr(self.worker, "toggle_tip_trajectory_visible"):
            return
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

    def toggle_recording(self, *_args):
        """Start or stop trajectory recording."""
        if self.recorder.recording:
            self.recorder.stop()
            self.record_start_button.setText("开始录制")
            self.record_export_button.setEnabled(self.recorder.frame_count > 0)
        else:
            self.recorder.start()
            self.record_start_button.setText("停止录制")
            self.record_export_button.setEnabled(False)

    def export_trajectory(self, *_args):
        """Export recorded q/dq trajectory to a pkl file."""
        if self.recorder.frame_count == 0:
            return
        default_name = f"trajectory_{int(_time.time())}.pkl"
        path, _ = QFileDialog.getSaveFileName(
            self, "导出轨迹", default_name, "Pickle Files (*.pkl)"
        )
        if path:
            self.recorder.export(path)

    def refresh_status(self):
        """Refresh displayed contact state and worker health."""
        if self.worker.error is not None:
            self.contact_label.setText(f"Error: {self.worker.error}")
            self.refresh_timer.stop()
            return

        sim = self.worker.simulation
        qpos = sim.data.qpos[sim.index_dof_ids]
        qvel = sim.data.qvel[sim.index_dof_ids]
        self.recorder.record(qpos, qvel)
        self.record_frame_label.setText(f"{self.recorder.frame_count} 帧")

        state = self.worker.get_contact_state()
        self.contact_label.setText("Contact" if state.in_contact else "No contact")
        self.force_label.setText(f"{state.normal_force:.4f} N")
        self.raw_force_label.setText(f"{state.raw_normal_force:.4f} N")
        tau = self.worker.simulation.controller.last_tau
        self.tau_label.setText(", ".join(f"{t:.4f}" for t in tau))
        self.geom_label.setText(", ".join(state.geoms) if state.geoms else "none")
        self.reservoir_widget.update_from_esn()

    def closeEvent(self, event):
        """Stop the simulation thread when the GUI closes."""
        self.refresh_timer.stop()
        self.worker.stop()
        event.accept()


def parse_args(argv):
    """Parse ESN GUI and shared simulation arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively control the L20a ESN contact platform."
    )
    add_esn_arguments(parser)
    parser.add_argument(
        "--position-range",
        type=float,
        default=0.12,
        help="XYZ control range around the initial platform position in meters.",
    )
    parser.add_argument("--refresh-ms", type=int, default=100)
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Run the Qt controller without launching the MuJoCo viewer.",
    )
    parser.add_argument(
        "--camera-view",
        type=Path,
        default=Path("mujoco_camera_view.json"),
        help="JSON file used by the save/load camera buttons.",
    )
    return parser.parse_args(argv)


def main():
    """Launch the ESN-control GUI and simulation worker."""
    args = parse_args(sys.argv[1:])
    simulation = build_esn_simulation(args)
    _set_white_viewer_background(simulation.model)
    worker = ESNTrajectoryWorker(
        simulation,
        launch_viewer=not args.no_viewer,
        camera_path=args.camera_view,
    )

    app = QApplication(sys.argv[:1])
    window = ESNContactControlWindow(worker, args.position_range, args.refresh_ms)
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
