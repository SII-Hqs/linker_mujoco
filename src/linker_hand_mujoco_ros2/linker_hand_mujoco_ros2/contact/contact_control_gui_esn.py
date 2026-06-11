#!/usr/bin/env python3
"""Interactive PyQt5 GUI for the L20a ESN torque-control contact demo."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
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
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui import SimulationWorker


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


class ESNContactControlWindow(QMainWindow):
    """GUI for XYZ platform control and ESN torque monitoring."""

    def __init__(self, worker, position_range, refresh_ms):
        super().__init__()
        self.worker = worker
        self.initial_pos = self.worker.simulation.platform.initial_pos.copy()
        self.position_range = float(position_range)
        self.position_inputs = {}

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
        root_layout.addLayout(button_layout)

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
    return parser.parse_args(argv)


def main():
    """Launch the ESN-control GUI and simulation worker."""
    args = parse_args(sys.argv[1:])
    simulation = build_esn_simulation(args)
    worker = SimulationWorker(simulation, launch_viewer=not args.no_viewer)

    app = QApplication(sys.argv[:1])
    window = ESNContactControlWindow(worker, args.position_range, args.refresh_ms)
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
