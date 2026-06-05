#!/usr/bin/env python3
"""Interactive PyQt5 GUI for the L20a torque-control contact demo."""

import argparse
import sys
from pathlib import Path

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
    QVBoxLayout,
    QWidget,
)

if __package__:
    from .contact_control_demo_torque import (
        add_torque_arguments,
        build_torque_simulation,
    )
    from .contact_control_gui import SimulationWorker
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_torque import (
        add_torque_arguments,
        build_torque_simulation,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_gui import SimulationWorker

class TorqueContactControlWindow(QMainWindow):
    """GUI for XYZ platform control and torque force-target tuning."""

    def __init__(self, worker, position_range, refresh_ms):
        super().__init__()
        self.worker = worker
        self.initial_pos = self.worker.simulation.platform.initial_pos.copy()
        self.position_range = float(position_range)
        self.position_inputs = {}

        self.setWindowTitle("L20a Torque Contact Platform Control")
        self.setMinimumWidth(420)
        self.setCentralWidget(self._build_content())

        self.refresh_timer = QTimer(self)
        self.refresh_timer.timeout.connect(self.refresh_status)
        self.refresh_timer.start(max(int(refresh_ms), 20))

        self.reset_platform()

    def _build_content(self):
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

        force_group = QGroupBox("Force Control")
        force_layout = QFormLayout(force_group)
        self.target_force_input = QDoubleSpinBox()
        self.target_force_input.setDecimals(3)
        self.target_force_input.setSingleStep(0.1)
        self.target_force_input.setRange(0.0, 50.0)
        self.target_force_input.setSuffix(" N")
        self.target_force_input.setValue(
            float(self.worker.simulation.controller.target_force)
        )
        self.target_force_input.valueChanged.connect(self.update_target_force)
        force_layout.addRow("Target force:", self.target_force_input)
        root_layout.addWidget(force_group)

        button_layout = QHBoxLayout()
        reset_button = QPushButton("复位")
        reset_button.clicked.connect(self.reset_platform)
        button_layout.addWidget(reset_button)
        root_layout.addLayout(button_layout)

        contact_group = QGroupBox("Contact State")
        contact_layout = QFormLayout(contact_group)
        self.contact_label = QLabel("No contact")
        self.contact_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.force_label = QLabel("0.0000 N")
        self.force_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.raw_force_label = QLabel("0.0000 N")
        self.raw_force_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.geom_label = QLabel("none")
        self.geom_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.geom_label.setWordWrap(True)
        contact_layout.addRow("Status:", self.contact_label)
        contact_layout.addRow("Normal force:", self.force_label)
        contact_layout.addRow("Raw force:", self.raw_force_label)
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

    def update_target_force(self, *_args):
        """Update the runtime force target."""
        self.worker.simulation.controller.set_target_force(
            self.target_force_input.value()
        )

    def reset_platform(self, *_args):
        """Reset platform inputs and target to the initial position."""
        for axis, value in zip(("X", "Y", "Z"), self.initial_pos):
            spin_box = self.position_inputs[axis]
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        self.worker.simulation.platform.reset_target_pos()

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
        self.geom_label.setText(", ".join(state.geoms) if state.geoms else "none")

    def closeEvent(self, event):
        """Stop the simulation thread when the GUI closes."""
        self.refresh_timer.stop()
        self.worker.stop()
        event.accept()


def parse_args(argv):
    """Parse torque GUI and shared simulation arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively control the L20a torque contact platform."
    )
    add_torque_arguments(parser)
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
    """Launch the torque-control GUI and simulation worker."""
    args = parse_args(sys.argv[1:])
    simulation = build_torque_simulation(args)
    worker = SimulationWorker(simulation, launch_viewer=not args.no_viewer)

    app = QApplication(sys.argv[:1])
    window = TorqueContactControlWindow(worker, args.position_range, args.refresh_ms)
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
