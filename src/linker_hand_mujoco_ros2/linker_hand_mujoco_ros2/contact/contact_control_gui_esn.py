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
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
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

# Import PyQt5 first so pyqtgraph cannot auto-select the installed PyQt6.
import pyqtgraph as pg

if __package__:
    from .contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )
    from .contact_control_gui import SimulationWorker
    from .viewer_camera import ViewerCameraMixin
    from .reservoir_inspector import ReservoirInspectorWindow
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui import SimulationWorker
    from linker_hand_mujoco_ros2.contact.viewer_camera import ViewerCameraMixin
    from linker_hand_mujoco_ros2.contact.reservoir_inspector import (
        ReservoirInspectorWindow,
    )


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
        self.image_label = QLabel()
        self.stats_label = QLabel()
        self.set_esn(esn)

        layout = QVBoxLayout(self)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(320, 180)
        self.image_label.setStyleSheet("background: #111; border: 1px solid #555;")
        self.stats_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.image_label)
        layout.addWidget(self.stats_label)

        self.update_from_esn()

    def set_esn(self, esn):
        """Switch the heatmap to another ESN reservoir."""
        self.esn = esn
        self.nr = int(esn.nr)
        self.rows = int(np.ceil(np.sqrt(self.nr)))
        self.cols = int(np.ceil(self.nr / self.rows))

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


class ForceHistoryPlotWidget(QGroupBox):
    """Plot the force value sent to the ESN controller over simulation time."""

    def __init__(self, force_clip, history_seconds=10.0, parent=None):
        super().__init__("ESN Force Input F(t)", parent)
        self.history_seconds = float(history_seconds)
        self.force_clip = float(force_clip)
        self.times = collections.deque()
        self.forces = collections.deque()

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(155)
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "Simulation time", units="s")
        self.plot_widget.setLabel("left", "F(t)", units="N")
        self.curve = self.plot_widget.plot(
            pen=pg.mkPen(color=(0, 114, 178), width=2)
        )
        if np.isfinite(self.force_clip):
            self.clip_line = pg.InfiniteLine(
                pos=self.force_clip,
                angle=0,
                pen=pg.mkPen(color=(213, 94, 0), width=1, style=Qt.DashLine),
                label="force clip",
                labelOpts={"position": 0.92},
            )
            self.plot_widget.addItem(self.clip_line)
            self.plot_widget.setYRange(0.0, max(1.0, self.force_clip * 1.08))

        self.value_label = QLabel("F(t) = 0.0000 N")
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.value_label)

    def set_force_clip(self, force_clip):
        """Move the force-limit line and update the plot range."""
        self.force_clip = float(force_clip)
        self.clip_line.setValue(self.force_clip)
        self.plot_widget.setYRange(
            0.0, max(1.0, self.force_clip * 1.08), padding=0.0
        )

    def add_sample(self, sim_time, force):
        """Add or refresh one force sample and keep a rolling time window."""
        sim_time = float(sim_time)
        force = float(force)
        if not np.isfinite(sim_time) or not np.isfinite(force):
            return

        if self.times and sim_time <= self.times[-1]:
            if sim_time == self.times[-1]:
                self.forces[-1] = force
            else:
                self.times.clear()
                self.forces.clear()
        else:
            self.times.append(sim_time)
            self.forces.append(force)

        cutoff_time = sim_time - self.history_seconds
        while self.times and self.times[0] < cutoff_time:
            self.times.popleft()
            self.forces.popleft()

        self.curve.setData(
            np.asarray(self.times, dtype=float),
            np.asarray(self.forces, dtype=float),
        )
        left = max(0.0, sim_time - self.history_seconds)
        right = max(self.history_seconds, sim_time)
        self.plot_widget.setXRange(left, right, padding=0.0)
        self.value_label.setText(f"F(t) = {force:.4f} N")


class TorqueHistoryPlotWidget(QGroupBox):
    """Plot the three ESN output torques over simulation time."""

    def __init__(self, history_seconds=10.0, parent=None):
        super().__init__("ESN Joint Torques", parent)
        self.history_seconds = float(history_seconds)
        self.times = collections.deque()
        self.torques = collections.deque()

        layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setMinimumHeight(155)
        self.plot_widget.setBackground("w")
        self.plot_widget.showGrid(x=True, y=True, alpha=0.25)
        self.plot_widget.setLabel("bottom", "Simulation time", units="s")
        self.plot_widget.setLabel("left", "Torque", units="N m")
        self.plot_widget.addLegend(offset=(8, 8))
        self.curves = (
            self.plot_widget.plot(
                pen=pg.mkPen(color=(0, 114, 178), width=2),
                name="index_joint1",
            ),
            self.plot_widget.plot(
                pen=pg.mkPen(color=(230, 159, 0), width=2),
                name="index_joint2",
            ),
            self.plot_widget.plot(
                pen=pg.mkPen(color=(0, 158, 115), width=2),
                name="index_joint3",
            ),
        )
        self.plot_widget.addItem(
            pg.InfiniteLine(
                pos=0.0,
                angle=0,
                pen=pg.mkPen(color=(100, 100, 100), width=1, style=Qt.DashLine),
            )
        )
        self.plot_widget.setYRange(-1.0, 1.0)

        self.value_label = QLabel("tau = [0.0000, 0.0000, 0.0000] N m")
        self.value_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.plot_widget)
        layout.addWidget(self.value_label)

    def add_sample(self, sim_time, torque):
        """Add or refresh one torque sample and keep a rolling time window."""
        sim_time = float(sim_time)
        torque = np.asarray(torque, dtype=float).reshape(-1)
        if torque.shape != (3,):
            return
        if not np.isfinite(sim_time) or not np.all(np.isfinite(torque)):
            return

        if self.times and sim_time <= self.times[-1]:
            if sim_time == self.times[-1]:
                self.torques[-1] = torque.copy()
            else:
                self.times.clear()
                self.torques.clear()
        else:
            self.times.append(sim_time)
            self.torques.append(torque.copy())

        cutoff_time = sim_time - self.history_seconds
        while self.times and self.times[0] < cutoff_time:
            self.times.popleft()
            self.torques.popleft()

        time_values = np.asarray(self.times, dtype=float)
        torque_values = np.asarray(self.torques, dtype=float).reshape(-1, 3)
        for joint_index, curve in enumerate(self.curves):
            curve.setData(time_values, torque_values[:, joint_index])

        left = max(0.0, sim_time - self.history_seconds)
        right = max(self.history_seconds, sim_time)
        self.plot_widget.setXRange(left, right, padding=0.0)
        peak = max(1.0, float(np.max(np.abs(torque_values))) * 1.10)
        self.plot_widget.setYRange(-peak, peak, padding=0.0)
        self.value_label.setText(
            "tau = [" + ", ".join(f"{value:.4f}" for value in torque) + "] N m"
        )


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
        self.pause_event = threading.Event()
        self.straighten_event = threading.Event()
        self.speed_lock = threading.Lock()
        self.obstacle_request_lock = threading.Lock()
        self.obstacle_request = None
        self.obstacle_removed = self.simulation.platform.is_removed()
        self.single_step_lock = threading.Lock()
        self.single_step_targets = collections.deque(maxlen=1)
        self.realtime_factor = 1.0
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

    def toggle_paused(self):
        """Pause or resume simulation stepping and return the paused state."""
        self.set_paused(not self.pause_event.is_set())
        return self.pause_event.is_set()

    def set_paused(self, paused):
        if paused:
            self.pause_event.set()
        else:
            self.pause_event.clear()

    def is_paused(self):
        """Return whether simulation stepping is paused."""
        return self.pause_event.is_set()

    def request_straighten_index_finger(self):
        """Request a straight index-finger pose on the worker thread."""
        self.straighten_event.set()

    def request_set_obstacle_removed(self, removed):
        """Request obstacle removal or restoration on the worker thread."""
        with self.obstacle_request_lock:
            self.obstacle_request = bool(removed)

    def is_obstacle_removed(self):
        """Return the last obstacle state applied by the worker thread."""
        with self.obstacle_request_lock:
            return self.obstacle_removed

    def _take_obstacle_request(self):
        with self.obstacle_request_lock:
            request = self.obstacle_request
            self.obstacle_request = None
        return request

    def _apply_obstacle_removed(self, removed):
        self.simulation.platform.set_removed(removed)
        mujoco.mj_forward(self.simulation.model, self.simulation.data)
        contact_state = self.simulation.observer.read()
        with self.state_lock:
            self.latest_contact_state = contact_state
        with self.obstacle_request_lock:
            self.obstacle_removed = bool(removed)

    def _apply_straight_index_pose(self):
        model = self.simulation.model
        data = self.simulation.data
        for joint_number in range(4):
            joint_name = f"index_joint{joint_number}"
            joint_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0:
                continue
            qpos_address = int(model.jnt_qposadr[joint_id])
            dof_address = int(model.jnt_dofadr[joint_id])
            joint_min, joint_max = model.jnt_range[joint_id]
            data.qpos[qpos_address] = np.clip(0.0, joint_min, joint_max)
            data.qvel[dof_address] = 0.0

        data.ctrl[:] = 0.0
        self.simulation.controller.reset()
        mujoco.mj_forward(model, data)
        contact_state = self.simulation.observer.read()
        with self.state_lock:
            self.latest_contact_state = contact_state

    def get_realtime_factor(self):
        with self.speed_lock:
            return self.realtime_factor

    def request_single_esn_step(self):
        self.set_paused(True)
        target_count = int(
            self.simulation.controller.current_controller.step_count
        ) + 1
        with self.single_step_lock:
            if not self.single_step_targets:
                self.single_step_targets.append(target_count)

    def cancel_single_esn_steps(self):
        with self.single_step_lock:
            self.single_step_targets.clear()

    def _take_single_esn_step_request(self):
        with self.single_step_lock:
            if not self.single_step_targets:
                return None
            return self.single_step_targets.popleft()

    def set_realtime_factor(self, factor):
        """Set simulation-time speed relative to wall-clock time."""
        factor = float(factor)
        if factor <= 0.0:
            raise ValueError("realtime factor must be positive")
        with self.speed_lock:
            self.realtime_factor = factor

    def _sleep_to_realtime(self, step_start):
        with self.speed_lock:
            factor = self.realtime_factor
        wall_step = self.simulation.model.opt.timestep / factor
        sleep_time = wall_step - (_time.time() - step_start)
        if sleep_time > 0.0:
            _time.sleep(sleep_time)

    def _record_tip_position(self):
        tip_pos = _get_link3_link4_midpoint(
            self.simulation.data,
            self.tip_body_id,
            self.tip_geom_id,
        ).copy()
        self.trajectory_visualizer.update_position(tip_pos)

    def _advance_one_esn_update(self, target_count):
        controller = self.simulation.controller
        if int(controller.horizon) != 0:
            return
        start_count = int(controller.current_controller.step_count)
        if start_count == target_count:
            return
        if target_count != start_count + 1:
            raise RuntimeError(
                f"single ESN step target {target_count}, current {start_count}"
            )
        max_steps = int(np.ceil(controller.update_period / controller.sim_dt)) + 2
        for _ in range(max_steps):
            super()._step_once()
            if int(controller.current_controller.step_count) > start_count:
                break
        end_count = int(controller.current_controller.step_count)
        if end_count != start_count + 1:
            raise RuntimeError(
                f"single ESN step expected {start_count + 1}, got {end_count}"
            )
        self._record_tip_position()

    def _step_once(self):
        obstacle_request = self._take_obstacle_request()
        if obstacle_request is not None:
            self._apply_obstacle_removed(obstacle_request)
        if self.straighten_event.is_set():
            self.straighten_event.clear()
            self._apply_straight_index_pose()
        single_step_target = self._take_single_esn_step_request()
        if single_step_target is not None:
            self.set_paused(True)
            self._advance_one_esn_update(single_step_target)
            return
        if self.pause_event.is_set():
            return
        super()._step_once()
        self._record_tip_position()

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
        self.inspector_window = None
        self.heatmap_window = None
        self.setWindowTitle("L20a ESN Contact Platform Control")
        self.setMinimumWidth(520)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll_area.setWidget(self._build_content())
        self.setCentralWidget(self.scroll_area)

        screen = QApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            width = min(980, max(520, int(available.width() * 0.90)))
            height = min(900, max(560, int(available.height() * 0.90)))
            self.resize(width, height)
        else:
            self.resize(900, 760)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(max(int(refresh_ms), 20))

        self.reset_platform()

    def _build_content(self):
        content = QWidget()
        root_layout = QVBoxLayout(content)
        root_layout.setContentsMargins(8, 8, 8, 8)
        root_layout.setSpacing(6)

        # Platform position controls
        position_group = QGroupBox("Platform Position")
        position_layout = QFormLayout(position_group)
        position_layout.setContentsMargins(8, 8, 8, 8)
        position_layout.setVerticalSpacing(4)
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
        esn_layout.setContentsMargins(8, 8, 8, 8)
        esn_layout.setVerticalSpacing(4)
        esn_ctrl = self.worker.simulation.controller
        self.horizon_input = QSpinBox()
        horizons = esn_ctrl.horizons
        self.horizon_input.setRange(min(horizons), max(horizons))
        self.horizon_input.setValue(esn_ctrl.horizon)
        self.horizon_input.valueChanged.connect(self.update_horizon)
        esn_layout.addRow("Horizon:", self.horizon_input)

        self.recurrence_k_input = QDoubleSpinBox()
        self.recurrence_k_input.setDecimals(4)
        self.recurrence_k_input.setSingleStep(0.01)
        self.recurrence_k_input.setRange(0.0, 1.0)
        self.recurrence_k_input.setValue(
            esn_ctrl.current_controller.recurrence_k
        )
        self.recurrence_k_input.valueChanged.connect(
            self.update_recurrence_k
        )
        esn_layout.addRow("Recurrence k:", self.recurrence_k_input)

        self.leak_alpha_input = QDoubleSpinBox()
        self.leak_alpha_input.setDecimals(4)
        self.leak_alpha_input.setSingleStep(0.01)
        self.leak_alpha_input.setRange(0.0, 1.0)
        self.leak_alpha_input.setValue(esn_ctrl.current_controller.leak_alpha)
        self.leak_alpha_input.setToolTip("Leaky-integrator alpha: 0 keeps old state; 1 uses the new tanh state.")
        self.leak_alpha_input.valueChanged.connect(self.update_leak_alpha)
        esn_layout.addRow("Leak alpha:", self.leak_alpha_input)

        self.win_scale_input = QDoubleSpinBox()
        self.win_scale_input.setDecimals(3)
        self.win_scale_input.setSingleStep(0.1)
        self.win_scale_input.setRange(0.0, 10.0)
        self.win_scale_input.setValue(
            esn_ctrl.current_controller.input_weight_scale
        )
        self.win_scale_input.setToolTip(
            "Multiply Win @ u at runtime. Saved Win and Wout are unchanged."
        )
        self.win_scale_input.valueChanged.connect(self.update_win_scale)
        esn_layout.addRow("Win scale:", self.win_scale_input)

        self.extreme_zero_n_input = QSpinBox()
        self.extreme_zero_n_input.setRange(
            0, esn_ctrl.current_controller.nr // 2
        )
        self.extreme_zero_n_input.setValue(
            esn_ctrl.current_controller.extreme_zero_n
        )
        self.extreme_zero_n_input.setToolTip(
            "After each recurrence, zero the n largest and n smallest neurons."
        )
        self.extreme_zero_n_input.valueChanged.connect(
            self.update_extreme_zero_n
        )
        esn_layout.addRow("Extreme zero n:", self.extreme_zero_n_input)

        self.force_gated_n_checkbox = QCheckBox()
        self.force_gated_n_checkbox.setChecked(
            esn_ctrl.current_controller.force_gate_enabled
        )
        self.force_gated_n_checkbox.setToolTip(
            "Use n=0 below the force threshold and the configured n above it."
        )
        self.force_gated_n_checkbox.stateChanged.connect(
            self.update_force_gated_n
        )
        esn_layout.addRow("Force-gated n:", self.force_gated_n_checkbox)

        self.force_n_threshold_input = QDoubleSpinBox()
        self.force_n_threshold_input.setDecimals(2)
        self.force_n_threshold_input.setSingleStep(0.5)
        self.force_n_threshold_input.setRange(0.0, 1000.0)
        self.force_n_threshold_input.setSuffix(" N")
        self.force_n_threshold_input.setValue(
            esn_ctrl.current_controller.force_gate_threshold
        )
        self.force_n_threshold_input.valueChanged.connect(
            self.update_force_n_threshold
        )
        esn_layout.addRow("n activation force:", self.force_n_threshold_input)

        self.effective_n_label = QLabel("0")
        self.effective_n_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        esn_layout.addRow("Effective n:", self.effective_n_label)

        self.force_limit_input = QDoubleSpinBox()
        self.force_limit_input.setDecimals(2)
        self.force_limit_input.setSingleStep(1.0)
        self.force_limit_input.setRange(0.01, 1000.0)
        self.force_limit_input.setSuffix(" N")
        self.force_limit_input.setValue(
            float(self.worker.simulation.observer.force_clip)
        )
        self.force_limit_input.valueChanged.connect(
            self.update_force_limit
        )
        esn_layout.addRow("Force input limit:", self.force_limit_input)

        self.torque_limit_input = QDoubleSpinBox()
        self.torque_limit_input.setDecimals(2)
        self.torque_limit_input.setSingleStep(0.10)
        self.torque_limit_input.setRange(0.01, 1000.0)
        self.torque_limit_input.setSuffix(" N m")
        initial_limit = esn_ctrl.current_controller.torque_limit
        if initial_limit is None or not np.all(np.isfinite(initial_limit)):
            initial_limit_value = 1000.0
        else:
            initial_limit_value = float(np.max(np.abs(initial_limit)))
        self.torque_limit_input.setValue(initial_limit_value)
        self.torque_limit_input.valueChanged.connect(
            self.update_torque_limit
        )
        esn_layout.addRow("Torque limit:", self.torque_limit_input)
        self.update_torque_limit(self.torque_limit_input.value())

        self.speed_input = QDoubleSpinBox()
        self.speed_input.setDecimals(2)
        self.speed_input.setSingleStep(0.25)
        self.speed_input.setRange(0.05, 4.0)
        self.speed_input.setSuffix(" x")
        self.speed_input.setValue(1.0)
        self.speed_input.valueChanged.connect(
            self.update_simulation_speed
        )
        esn_layout.addRow("Simulation speed:", self.speed_input)
        self.update_simulation_speed(self.speed_input.value())
        root_layout.addWidget(esn_group)

        self.reservoir_widget = ReservoirHeatmapWidget(esn_ctrl.active_controller)
        self.open_heatmap_button = QPushButton("Open separate reservoir heatmap")
        self.open_heatmap_button.clicked.connect(self.open_heatmap_window)
        root_layout.addWidget(self.open_heatmap_button)

        self.force_history_widget = ForceHistoryPlotWidget(
            self.worker.simulation.observer.force_clip
        )
        root_layout.addWidget(self.force_history_widget)

        self.torque_history_widget = TorqueHistoryPlotWidget()
        root_layout.addWidget(self.torque_history_widget)

        # Buttons
        button_layout = QHBoxLayout()
        button_layout.setSpacing(4)
        self.pause_button = QPushButton("Pause simulation")
        self.pause_button.clicked.connect(self.toggle_pause)
        button_layout.addWidget(self.pause_button)
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

        pose_button_layout = QHBoxLayout()
        pose_button_layout.setSpacing(4)
        self.straighten_index_button = QPushButton("食指全部打直")
        self.straighten_index_button.clicked.connect(
            self.straighten_index_finger
        )
        pose_button_layout.addWidget(self.straighten_index_button)
        self.open_inspector_button = QPushButton("打开神经元递推检查器")
        self.open_inspector_button.clicked.connect(self.open_reservoir_inspector)
        pose_button_layout.addWidget(self.open_inspector_button)
        self.obstacle_button = QPushButton("移除障碍物")
        self.obstacle_button.clicked.connect(self.toggle_obstacle)
        pose_button_layout.addWidget(self.obstacle_button)
        pose_button_layout.addStretch(1)
        root_layout.addLayout(pose_button_layout)

        # Trajectory recording
        record_group = QGroupBox("轨迹录制")
        record_layout = QFormLayout(record_group)
        record_layout.setContentsMargins(8, 8, 8, 8)
        record_layout.setVerticalSpacing(4)
        self.record_frame_label = QLabel("0 帧")
        self.record_frame_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        record_layout.addRow("已录制:", self.record_frame_label)
        record_btn_layout = QHBoxLayout()
        record_btn_layout.setSpacing(4)
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
        contact_layout.setContentsMargins(8, 8, 8, 8)
        contact_layout.setVerticalSpacing(4)
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
        self.reservoir_widget.set_esn(
            self.worker.simulation.controller.active_controller
        )
        self.reservoir_widget.update_from_esn()

    def update_force_limit(self, value):
        """Apply the force-input limit and update its plot marker."""
        force_limit = float(value)
        self.worker.simulation.observer.force_clip = force_limit
        if hasattr(self, "force_history_widget"):
            self.force_history_widget.set_force_clip(force_limit)

    def update_torque_limit(self, value):
        """Apply one symmetric torque limit to all three joints."""
        limit = np.full(3, float(value), dtype=float)
        esn_ctrl = self.worker.simulation.controller
        esn_ctrl.current_controller.torque_limit = limit.copy()
        esn_ctrl.horizon_controller.torque_limit = limit.copy()

    def update_simulation_speed(self, value):
        """Apply the selected simulation-time speed factor."""
        self.worker.set_realtime_factor(value)

    def update_recurrence_k(self, value):
        """Apply the recurrence correction strength to the current ESN."""
        current_esn = self.worker.simulation.controller.current_controller
        current_esn.set_recurrence_k(value)

    def update_leak_alpha(self, value):
        """Apply the reservoir leak alpha at runtime."""
        current_esn = self.worker.simulation.controller.current_controller
        current_esn.set_leak_alpha(value)

    def update_win_scale(self, value):
        """Scale the current-input contribution without changing Wout."""
        current_esn = self.worker.simulation.controller.current_controller
        current_esn.set_input_weight_scale(value)

    def update_extreme_zero_n(self, value):
        """Set how many largest and smallest neurons are zeroed."""
        current_esn = self.worker.simulation.controller.current_controller
        current_esn.set_extreme_zero_n(value)

    def update_force_gated_n(self, state):
        """Enable force-gated extreme zeroing."""
        current_esn = self.worker.simulation.controller.current_controller
        current_esn.set_force_gate_enabled(bool(state))

    def update_force_n_threshold(self, value):
        """Set the force that activates the configured extreme n."""
        current_esn = self.worker.simulation.controller.current_controller
        current_esn.set_force_gate_threshold(value, hysteresis=0.5)

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
        self.reservoir_widget.update_from_esn()

    def toggle_pause(self, *_args):
        """Pause or resume MuJoCo and ESN stepping."""
        paused = self.worker.toggle_paused()
        self.pause_button.setText(
            "Resume simulation" if paused else "Pause simulation"
        )

    def open_reservoir_inspector(self, *_args):
        if self.inspector_window is None:
            self.inspector_window = ReservoirInspectorWindow(self.worker, owner=self)
        self.inspector_window.show()
        self.inspector_window.raise_()
        self.inspector_window.activateWindow()

    def open_heatmap_window(self, *_args):
        """Show the live reservoir heatmap in its own window."""
        if self.heatmap_window is None:
            self.heatmap_window = QMainWindow(self)
            self.heatmap_window.setAttribute(Qt.WA_DeleteOnClose, False)
            self.heatmap_window.setWindowTitle("ESN Reservoir Heatmap")
            self.heatmap_window.resize(460, 380)
            self.heatmap_window.setCentralWidget(self.reservoir_widget)
        self.heatmap_window.show()
        self.heatmap_window.raise_()
        self.heatmap_window.activateWindow()

    def sync_worker_controls(self):
        self.pause_button.setText(
            "Resume simulation" if self.worker.is_paused() else "Pause simulation"
        )
        obstacle_removed = self.worker.is_obstacle_removed()
        self.obstacle_button.setText(
            "恢复障碍物" if obstacle_removed else "移除障碍物"
        )
        for spin_box in self.position_inputs.values():
            spin_box.setEnabled(not obstacle_removed)
        speed = self.worker.get_realtime_factor()
        self.speed_input.blockSignals(True)
        self.speed_input.setValue(speed)
        self.speed_input.blockSignals(False)

    def toggle_obstacle(self, *_args):
        """Remove or restore the platform without touching MuJoCo from Qt."""
        removed = self.worker.is_obstacle_removed()
        self.worker.request_set_obstacle_removed(not removed)

    def straighten_index_finger(self, *_args):
        """Pause and request a fully straight index-finger pose."""
        if not self.worker.is_paused():
            self.worker.toggle_paused()
        self.pause_button.setText("Resume simulation")
        self.worker.request_straighten_index_finger()

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
        current_esn = self.worker.simulation.controller.current_controller
        self.effective_n_label.setText(
            str(current_esn.effective_extreme_zero_n)
        )
        self.tau_label.setText(", ".join(f"{t:.4f}" for t in tau))
        self.geom_label.setText(", ".join(state.geoms) if state.geoms else "none")
        self.force_history_widget.add_sample(sim.data.time, state.normal_force)
        self.torque_history_widget.add_sample(sim.data.time, tau)
        self.reservoir_widget.update_from_esn()
        self.sync_worker_controls()

    def closeEvent(self, event):
        """Stop the simulation thread when the GUI closes."""
        self.refresh_timer.stop()
        if self.inspector_window is not None:
            self.inspector_window.close()
        if self.heatmap_window is not None:
            self.heatmap_window.close()
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

    # Qt requires its application object before anything in the construction
    # chain can create a QWidget (including optional diagnostic components).
    app = QApplication(sys.argv[:1])

    simulation = build_esn_simulation(args)
    _set_white_viewer_background(simulation.model)
    worker = ESNTrajectoryWorker(
        simulation,
        launch_viewer=not args.no_viewer,
        camera_path=args.camera_view,
    )

    window = ESNContactControlWindow(worker, args.position_range, args.refresh_ms)
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
