#!/usr/bin/env python3
"""Interactive PyQt5 GUI for the target q/qdot ESN contact demo."""

import argparse
import sys
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtWidgets import (
    QApplication,
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
    from .contact_control_demo_esn_target import (
        add_esn_target_arguments,
        build_esn_target_simulation,
    )
    from .contact_control_gui import SimulationWorker
    from .contact_control_gui_esn import ReservoirHeatmapWidget
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn_target import (
        add_esn_target_arguments,
        build_esn_target_simulation,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui import SimulationWorker
    from linker_hand_mujoco_ros2.contact.contact_control_gui_esn import (
        ReservoirHeatmapWidget,
    )


class ESNTargetContactControlWindow(QMainWindow):
    """GUI for target q/qdot ESN platform control and monitoring."""

    def __init__(self, worker, position_range, refresh_ms):
        super().__init__()
        self.worker = worker
        self.initial_pos = self.worker.simulation.platform.initial_pos.copy()
        self.position_range = float(position_range)
        self.position_inputs = {}
        self.target_qpos_inputs = []
        self.target_qvel_inputs = []

        self.setWindowTitle("L20a ESN Target-Tracking Contact Platform Control")
        self.setMinimumWidth(520)
        self.resize(620, 860)
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
        reset_button = QPushButton("复位")
        reset_button.clicked.connect(self.reset_platform)
        button_layout.addWidget(reset_button)
        reset_esn_button = QPushButton("重置 ESN")
        reset_esn_button.clicked.connect(self.reset_esn)
        button_layout.addWidget(reset_esn_button)
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
        contact_layout.addRow("Torque:", self.tau_label)
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
        self.worker.simulation.target_qpos[:] = [
            spin_box.value() for spin_box in self.target_qpos_inputs
        ]
        self.worker.simulation.target_qvel[:] = [
            spin_box.value() for spin_box in self.target_qvel_inputs
        ]

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
        error = np.linalg.norm(self.worker.simulation.tracking_error())
        self.tracking_error_label.setText(f"{error:.5f} rad")
        self.geom_label.setText(", ".join(state.geoms) if state.geoms else "none")
        self.reservoir_widget.update_from_esn()

    def closeEvent(self, event):
        """Stop the simulation thread when the GUI closes."""
        self.refresh_timer.stop()
        self.worker.stop()
        event.accept()


def parse_args(argv):
    """Parse target q/qdot ESN GUI and shared simulation arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively control the L20a target q/qdot ESN platform."
    )
    add_esn_target_arguments(parser)
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
    """Launch the target q/qdot ESN GUI and simulation worker."""
    args = parse_args(sys.argv[1:])
    simulation = build_esn_target_simulation(args)
    worker = SimulationWorker(simulation, launch_viewer=not args.no_viewer)

    app = QApplication(sys.argv[:1])
    window = ESNTargetContactControlWindow(
        worker,
        args.position_range,
        args.refresh_ms,
    )
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
