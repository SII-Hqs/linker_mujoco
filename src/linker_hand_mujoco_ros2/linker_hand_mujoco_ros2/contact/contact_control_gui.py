#!/usr/bin/env python3
"""Interactive PyQt5 GUI for the L20a platform contact-control demo."""

import argparse
import sys
import threading
import time
from pathlib import Path

import mujoco.viewer
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
    from .contact_control_demo import add_simulation_arguments, build_simulation
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo import add_simulation_arguments, build_simulation


class SimulationWorker:
    """Run MuJoCo and its passive viewer outside the Qt event loop."""

    def __init__(self, simulation, launch_viewer=True):
        self.simulation = simulation
        self.launch_viewer = launch_viewer
        self.stop_event = threading.Event()
        self.state_lock = threading.Lock()
        self.latest_contact_state = self.simulation.observer.read()
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

    def set_platform_pos(self, x, y, z):
        """Set the external platform target."""
        self.simulation.platform.set_target_pos(x, y, z)

    def get_contact_state(self):
        """Return the most recently observed contact state."""
        with self.state_lock:
            return self.latest_contact_state

    def _step_once(self):
        contact_state = self.simulation.step()
        with self.state_lock:
            self.latest_contact_state = contact_state

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


class ContactControlWindow(QMainWindow):
    """GUI for direct XYZ platform control and contact-force monitoring."""

    def __init__(self, worker, position_range, refresh_ms):
        super().__init__()
        self.worker = worker
        self.initial_pos = self.worker.simulation.platform.initial_pos.copy()
        self.position_range = float(position_range)
        self.position_inputs = {}

        self.setWindowTitle("L20a Contact Platform Control")
        self.setMinimumWidth(380)
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
        """Send the current XYZ inputs to the simulation worker."""
        self.worker.set_platform_pos(
            self.position_inputs["X"].value(),
            self.position_inputs["Y"].value(),
            self.position_inputs["Z"].value(),
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
    """Parse GUI and shared simulation arguments."""
    parser = argparse.ArgumentParser(
        description="Interactively control the L20a MuJoCo contact platform."
    )
    add_simulation_arguments(parser)
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
    """Launch the contact-control GUI and simulation worker."""
    args = parse_args(sys.argv[1:])
    simulation = build_simulation(args)
    worker = SimulationWorker(simulation, launch_viewer=not args.no_viewer)

    app = QApplication(sys.argv[:1])
    window = ContactControlWindow(worker, args.position_range, args.refresh_ms)
    worker.start()
    window.show()
    exit_code = app.exec_()
    worker.stop()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
