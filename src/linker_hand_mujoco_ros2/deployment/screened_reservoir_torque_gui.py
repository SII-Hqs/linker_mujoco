#!/usr/bin/env python3
"""Qt launcher for screened-reservoir torque models in MuJoCo.

The GUI deliberately does not construct or run a simulation at startup.  It
starts ``screened_reservoir_torque_mujoco.py`` only after the user selects a
candidate and presses the run button.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QStyle,
    QVBoxLayout,
    QWidget,
)


DEPLOYMENT_DIR = Path(__file__).resolve().parent
DEFAULT_CANDIDATE_DIR = DEPLOYMENT_DIR / "candidate_models_extended_2s"
DEFAULT_MODEL = DEFAULT_CANDIDATE_DIR / "candidate_0004_review.npz"
DEFAULT_DEPLOYMENT_SCRIPT = DEPLOYMENT_DIR / "screened_reservoir_torque_mujoco.py"


def _first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.is_file():
            return path.resolve()
    return paths[0].resolve()


def default_xml_path() -> Path:
    package_root = DEPLOYMENT_DIR.parent
    return _first_existing(
        package_root
        / "linker_hand_mujoco_ros2/urdf/L20/linker_hand_l20_right"
        / "linker_hand_l20_right.xml",
        package_root
        / "urdf/L20/linker_hand_l20_right/linker_hand_l20_right.xml",
    )


def default_reference_csv() -> Path | None:
    repo_root = DEPLOYMENT_DIR.parents[2]
    search_dirs = (
        Path.cwd() / "results/joint_logs/movement_without_thumb_joint0_100hz",
        repo_root / "results/joint_logs/movement_without_thumb_joint0_100hz",
    )
    for directory in search_dirs:
        candidates = sorted(directory.glob("*.csv"))
        if candidates:
            return candidates[0].resolve()
    return None


def discover_candidate_models(directory: Path) -> list[Path]:
    directory = Path(directory).expanduser().resolve()
    if not directory.is_dir():
        return []
    return sorted(path.resolve() for path in directory.glob("*.npz") if path.is_file())


@dataclass(frozen=True)
class CandidateMetadata:
    candidate_id: int
    status: str
    pareto_core: bool
    reservoir_size: int
    seed: int
    regularization: float
    spectral_radius: float
    leak_rate: float
    input_scaling: float
    connectivity: float
    bias_scaling: float
    washout_frames: int
    sample_rate_hz: float
    future_horizon_seconds: float


def read_candidate_metadata(path: Path) -> CandidateMetadata:
    fields = tuple(CandidateMetadata.__dataclass_fields__)
    archive_names = {
        "status": "screening_status",
    }
    with np.load(path, allow_pickle=False) as archive:
        missing = [archive_names.get(name, name) for name in fields]
        missing = [name for name in missing if name not in archive.files]
        if missing:
            raise ValueError(f"missing fields: {', '.join(missing)}")
        values = {
            name: np.asarray(archive[archive_names.get(name, name)]).item()
            for name in fields
        }

    metadata = CandidateMetadata(
        candidate_id=int(values["candidate_id"]),
        status=str(values["status"]),
        pareto_core=bool(values["pareto_core"]),
        reservoir_size=int(values["reservoir_size"]),
        seed=int(values["seed"]),
        regularization=float(values["regularization"]),
        spectral_radius=float(values["spectral_radius"]),
        leak_rate=float(values["leak_rate"]),
        input_scaling=float(values["input_scaling"]),
        connectivity=float(values["connectivity"]),
        bias_scaling=float(values["bias_scaling"]),
        washout_frames=int(values["washout_frames"]),
        sample_rate_hz=float(values["sample_rate_hz"]),
        future_horizon_seconds=float(values["future_horizon_seconds"]),
    )
    if not np.isclose(metadata.sample_rate_hz, 100.0, atol=1e-12):
        raise ValueError(f"expected 100 Hz, got {metadata.sample_rate_hz:g} Hz")
    if not np.isclose(metadata.future_horizon_seconds, 0.1, atol=1e-12):
        raise ValueError(
            "expected a 0.1 s horizon, got "
            f"{metadata.future_horizon_seconds:g} s"
        )
    return metadata


class PathRow(QWidget):
    def __init__(self, value: Path | None, file_filter: str, parent=None):
        super().__init__(parent)
        self.file_filter = file_filter
        self.edit = QLineEdit(str(value) if value is not None else "")
        self.edit.setClearButtonEnabled(True)
        self.button = QPushButton()
        self.button.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.button.setToolTip("Choose file")
        self.button.setFixedWidth(36)
        self.button.clicked.connect(self.choose_file)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(self.button)

    def choose_file(self):
        current = self.edit.text().strip()
        start = str(Path(current).expanduser().parent) if current else str(Path.cwd())
        selected, _ = QFileDialog.getOpenFileName(
            self, "Choose file", start, self.file_filter
        )
        if selected:
            self.edit.setText(selected)

    def path(self) -> Path | None:
        value = self.edit.text().strip()
        return Path(value).expanduser().resolve() if value else None

    def set_controls_enabled(self, enabled: bool) -> None:
        self.edit.setEnabled(enabled)
        self.button.setEnabled(enabled)


class ScreenedReservoirTorqueWindow(QMainWindow):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.process = QProcess(self)
        self.stop_requested = False
        self._connect_process()

        self.setWindowTitle("Screened Reservoir Torque - MuJoCo")
        self.setMinimumSize(780, 720)
        self.setCentralWidget(self._build_content())
        self._apply_style()
        self._load_models(args.model)

    def _connect_process(self) -> None:
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self._read_process_output)
        self.process.started.connect(self._process_started)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)

    def _build_content(self) -> QWidget:
        content = QWidget()
        root = QVBoxLayout(content)
        root.setContentsMargins(18, 16, 18, 16)
        root.setSpacing(12)

        title = QLabel("Screened Reservoir Torque Controller")
        title.setObjectName("title")
        subtitle = QLabel("100 Hz control | 100 ms future-pose horizon")
        subtitle.setObjectName("subtitle")
        root.addWidget(title)
        root.addWidget(subtitle)

        model_group = QGroupBox("Candidate model")
        model_layout = QFormLayout(model_group)
        self.model_combo = QComboBox()
        self.model_combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.model_combo.setMinimumContentsLength(34)
        self.model_combo.currentIndexChanged.connect(self._model_changed)
        model_layout.addRow("Model", self.model_combo)

        metadata_widget = QWidget()
        metadata_layout = QGridLayout(metadata_widget)
        metadata_layout.setContentsMargins(0, 4, 0, 0)
        self.identity_label = QLabel("-")
        self.network_label = QLabel("-")
        self.hyperparameters_label = QLabel("-")
        self.hyperparameters_label.setWordWrap(True)
        metadata_layout.addWidget(QLabel("Identity"), 0, 0)
        metadata_layout.addWidget(self.identity_label, 0, 1)
        metadata_layout.addWidget(QLabel("Network"), 1, 0)
        metadata_layout.addWidget(self.network_label, 1, 1)
        metadata_layout.addWidget(QLabel("Parameters"), 2, 0)
        metadata_layout.addWidget(self.hyperparameters_label, 2, 1)
        metadata_layout.setColumnStretch(1, 1)
        model_layout.addRow(metadata_widget)
        root.addWidget(model_group)

        files_group = QGroupBox("Simulation input")
        files_layout = QFormLayout(files_group)
        self.reference_row = PathRow(
            self.args.reference_csv, "CSV files (*.csv);;All files (*)"
        )
        self.initial_row = PathRow(
            self.args.initial_csv, "CSV files (*.csv);;All files (*)"
        )
        self.xml_row = PathRow(
            self.args.xml, "MuJoCo XML (*.xml);;All files (*)"
        )
        self.log_row = PathRow(
            self.args.log, "CSV files (*.csv);;All files (*)"
        )
        files_layout.addRow("Reference CSV", self.reference_row)
        files_layout.addRow("Initial CSV (optional)", self.initial_row)
        files_layout.addRow("MuJoCo XML", self.xml_row)
        self.write_log_checkbox = QCheckBox("Write simulation log")
        self.write_log_checkbox.setChecked(self.args.log is not None)
        self.write_log_checkbox.toggled.connect(self._update_log_enabled)
        files_layout.addRow(self.write_log_checkbox, self.log_row)
        root.addWidget(files_group)

        settings_group = QGroupBox("Run settings")
        settings_layout = QGridLayout(settings_group)
        self.torque_limit = self._double_spin(
            0.01, 100.0, self.args.torque_limit, 0.05, 3
        )
        self.torque_limit.setSuffix(" N m")
        self.ramp_seconds = self._double_spin(
            0.0, 60.0, self.args.ramp_seconds, 0.05, 3
        )
        self.ramp_seconds.setSuffix(" s")
        self.duration = self._double_spin(
            0.0, 3600.0, self.args.duration or 0.0, 1.0, 2
        )
        self.duration.setSpecialValueText("Automatic")
        self.duration.setSuffix(" s")
        self.max_joint_speed = self._double_spin(
            0.1, 1000.0, self.args.max_joint_speed, 1.0, 1
        )
        self.max_joint_speed.setSuffix(" rad/s")
        self.max_input_zscore = self._double_spin(
            0.1, 1000.0, self.args.max_input_zscore, 1.0, 1
        )
        self.warmup_steps = QSpinBox()
        self.warmup_steps.setRange(-1, 1_000_000)
        self.warmup_steps.setSpecialValueText("Model washout")
        self.warmup_steps.setValue(
            self.args.reservoir_warmup_steps
            if self.args.reservoir_warmup_steps is not None
            else -1
        )
        self.initial_row_index = QSpinBox()
        self.initial_row_index.setRange(0, 100_000_000)
        self.initial_row_index.setValue(self.args.initial_row)
        self.contacts_checkbox = QCheckBox("Enable contacts")
        self.contacts_checkbox.setChecked(self.args.enable_contacts)
        self.headless_checkbox = QCheckBox("Headless (no MuJoCo viewer)")
        self.headless_checkbox.setChecked(self.args.headless)
        self.live_visualization_checkbox = QCheckBox(
            "Live reservoir visualization"
        )
        self.live_visualization_checkbox.setChecked(self.args.live_visualization)
        self.visualization_fps = self._double_spin(
            1.0, 60.0, self.args.visualization_fps, 1.0, 1
        )
        self.visualization_fps.setSuffix(" FPS")
        self.visualization_window_seconds = self._double_spin(
            0.2, 30.0, self.args.visualization_window_seconds, 0.5, 1
        )
        self.visualization_window_seconds.setSuffix(" s")
        self.live_visualization_checkbox.toggled.connect(
            self._update_visualization_enabled
        )
        self._update_visualization_enabled(self.args.live_visualization)

        controls = (
            ("Torque limit", self.torque_limit),
            ("Ramp", self.ramp_seconds),
            ("Duration", self.duration),
            ("Max joint speed", self.max_joint_speed),
            ("Max input |z|", self.max_input_zscore),
            ("Warmup steps", self.warmup_steps),
            ("Initial CSV row", self.initial_row_index),
            ("Visualization FPS", self.visualization_fps),
            ("Visualization window", self.visualization_window_seconds),
        )
        split = 5
        for index, (label, widget) in enumerate(controls):
            column = 0 if index < split else 2
            row = index if index < split else index - split
            settings_layout.addWidget(QLabel(label), row, column)
            settings_layout.addWidget(widget, row, column + 1)
        settings_layout.addWidget(self.contacts_checkbox, 5, 0, 1, 2)
        settings_layout.addWidget(self.headless_checkbox, 5, 2, 1, 2)
        settings_layout.addWidget(
            self.live_visualization_checkbox, 6, 0, 1, 4
        )
        settings_layout.setColumnStretch(1, 1)
        settings_layout.setColumnStretch(3, 1)
        root.addWidget(settings_group)

        action_layout = QHBoxLayout()
        self.run_button = QPushButton("Run selected model")
        self.run_button.setObjectName("runButton")
        self.run_button.setIcon(self.style().standardIcon(QStyle.SP_MediaPlay))
        self.run_button.clicked.connect(self.start_selected_model)
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setIcon(self.style().standardIcon(QStyle.SP_MediaStop))
        self.stop_button.clicked.connect(self.stop_model)
        self.stop_button.setEnabled(False)
        self.status_label = QLabel("Ready - no model is running")
        self.status_label.setObjectName("status")
        action_layout.addWidget(self.run_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addWidget(self.status_label, 1)
        root.addLayout(action_layout)

        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setPlaceholderText("Process output")
        self.output.setMaximumBlockCount(5000)
        self.output.setFont(QFont("Monospace", 9))
        root.addWidget(self.output, 1)

        self._update_log_enabled(self.write_log_checkbox.isChecked())
        return content

    @staticmethod
    def _double_spin(
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> QDoubleSpinBox:
        widget = QDoubleSpinBox()
        widget.setRange(minimum, maximum)
        widget.setValue(value)
        widget.setSingleStep(step)
        widget.setDecimals(decimals)
        return widget

    def _apply_style(self) -> None:
        self.setStyleSheet(
            """
            QMainWindow, QWidget { background: #f4f6f8; color: #20252b; }
            QLabel#title { font-size: 20px; font-weight: 600; }
            QLabel#subtitle { color: #5f6975; margin-bottom: 4px; }
            QLabel#status { color: #46515d; padding-left: 8px; }
            QGroupBox {
                background: #ffffff; border: 1px solid #d6dce2;
                border-radius: 6px; margin-top: 10px; padding: 12px 10px 8px;
                font-weight: 600;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
                background: #ffffff; border: 1px solid #b9c1c9;
                border-radius: 4px; padding: 5px; selection-background-color: #16725c;
            }
            QPushButton { min-height: 28px; padding: 3px 12px; }
            QPushButton#runButton {
                color: #ffffff; background: #16725c;
                border: 0; border-radius: 4px;
            }
            QPushButton#runButton:disabled { background: #9aa6a1; }
            QPushButton#stopButton { color: #9b2c2c; }
            """
        )

    def _load_models(self, initial_model: Path | None) -> None:
        models = discover_candidate_models(self.args.candidate_dir)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for path in models:
            self.model_combo.addItem(path.name, str(path))
        self.model_combo.blockSignals(False)

        if not models:
            self.run_button.setEnabled(False)
            self.status_label.setText(
                f"No .npz candidates found in {self.args.candidate_dir}"
            )
            self._model_changed(-1)
            return

        if initial_model is not None:
            wanted = initial_model.expanduser().resolve()
            for index, path in enumerate(models):
                if path == wanted:
                    self.model_combo.setCurrentIndex(index)
                    break
        self._model_changed(self.model_combo.currentIndex())

    def _model_changed(self, index: int) -> None:
        if index < 0:
            self.identity_label.setText("-")
            self.network_label.setText("-")
            self.hyperparameters_label.setText("-")
            return
        path = Path(self.model_combo.itemData(index))
        try:
            metadata = read_candidate_metadata(path)
        except (OSError, TypeError, ValueError) as exc:
            self.identity_label.setText(f"Invalid candidate: {exc}")
            self.network_label.setText("-")
            self.hyperparameters_label.setText("-")
            self.run_button.setEnabled(False)
            return

        pareto = " | Pareto core" if metadata.pareto_core else ""
        self.identity_label.setText(
            f"ID {metadata.candidate_id:04d} | {metadata.status}{pareto}"
        )
        self.network_label.setText(
            f"reservoir={metadata.reservoir_size} | seed={metadata.seed} | "
            f"{metadata.sample_rate_hz:g} Hz | "
            f"horizon={metadata.future_horizon_seconds:g} s"
        )
        self.hyperparameters_label.setText(
            f"rho={metadata.spectral_radius:g}, leak={metadata.leak_rate:g}, "
            f"input={metadata.input_scaling:g}, "
            f"connectivity={metadata.connectivity:g}, "
            f"bias={metadata.bias_scaling:g}, lambda={metadata.regularization:g}, "
            f"washout={metadata.washout_frames}"
        )
        self.run_button.setEnabled(self.process.state() == QProcess.NotRunning)

    def _update_log_enabled(self, enabled: bool) -> None:
        self.log_row.set_controls_enabled(enabled)

    def _update_visualization_enabled(self, enabled: bool) -> None:
        self.visualization_fps.setEnabled(enabled)
        self.visualization_window_seconds.setEnabled(enabled)

    def _required_file(self, row: PathRow, label: str) -> Path:
        path = row.path()
        if path is None or not path.is_file():
            raise ValueError(f"{label} does not exist: {path or '(empty)'}")
        return path

    def _deployment_arguments(self) -> list[str]:
        model_value = self.model_combo.currentData()
        if not model_value:
            raise ValueError("Select a candidate model")
        model = Path(model_value)
        script = Path(self.args.deployment_script).expanduser().resolve()
        if not model.is_file():
            raise ValueError(f"Candidate model does not exist: {model}")
        if not script.is_file():
            raise ValueError(f"Deployment script does not exist: {script}")

        reference = self._required_file(self.reference_row, "Reference CSV")
        xml = self._required_file(self.xml_row, "MuJoCo XML")
        arguments = [
            str(script),
            "--model",
            str(model),
            "--xml",
            str(xml),
            "--reference-csv",
            str(reference),
            "--initial-row",
            str(self.initial_row_index.value()),
            "--torque-limit",
            f"{self.torque_limit.value():g}",
            "--ramp-seconds",
            f"{self.ramp_seconds.value():g}",
            "--max-joint-speed",
            f"{self.max_joint_speed.value():g}",
            "--max-input-zscore",
            f"{self.max_input_zscore.value():g}",
        ]
        initial = self.initial_row.path()
        if initial is not None:
            if not initial.is_file():
                raise ValueError(f"Initial CSV does not exist: {initial}")
            arguments.extend(["--initial-csv", str(initial)])
        if self.duration.value() > 0.0:
            arguments.extend(["--duration", f"{self.duration.value():g}"])
        if self.warmup_steps.value() >= 0:
            arguments.extend(
                ["--reservoir-warmup-steps", str(self.warmup_steps.value())]
            )
        if self.contacts_checkbox.isChecked():
            arguments.append("--enable-contacts")
        if self.headless_checkbox.isChecked():
            arguments.append("--headless")
        if self.live_visualization_checkbox.isChecked():
            arguments.extend(
                [
                    "--live-visualization",
                    "--visualization-fps",
                    f"{self.visualization_fps.value():g}",
                    "--visualization-window-seconds",
                    f"{self.visualization_window_seconds.value():g}",
                    "--visualization-backend",
                    self.args.visualization_backend,
                ]
            )
        if self.write_log_checkbox.isChecked():
            log_path = self.log_row.path()
            if log_path is None:
                raise ValueError("Choose a simulation log path or disable logging")
            arguments.extend(["--log", str(log_path)])
        return arguments

    def start_selected_model(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            return
        try:
            arguments = self._deployment_arguments()
        except ValueError as exc:
            self.status_label.setText(str(exc))
            if self.args.auto_start:
                QTimer.singleShot(0, QApplication.instance().quit)
            else:
                QMessageBox.warning(self, "Cannot start", str(exc))
            return

        self.output.clear()
        self.output.appendPlainText(
            f"$ {sys.executable} " + " ".join(arguments) + "\n"
        )
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        self.process.setProcessEnvironment(environment)
        self.stop_requested = False
        self.status_label.setText("Starting selected model...")
        self.run_button.setEnabled(False)
        self.model_combo.setEnabled(False)
        self.process.start(sys.executable, arguments)

    def stop_model(self) -> None:
        if self.process.state() == QProcess.NotRunning:
            return
        self.stop_requested = True
        self.status_label.setText("Stopping...")
        self.stop_button.setEnabled(False)
        self.process.terminate()
        QTimer.singleShot(2000, self._kill_if_running)

    def _kill_if_running(self) -> None:
        if self.process.state() != QProcess.NotRunning:
            self.process.kill()

    def _process_started(self) -> None:
        name = self.model_combo.currentText()
        self.status_label.setText(f"Running {name}")
        self.stop_button.setEnabled(True)

    def _read_process_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode(
            errors="replace"
        )
        if data:
            cursor = self.output.textCursor()
            cursor.movePosition(cursor.End)
            cursor.insertText(data)
            self.output.setTextCursor(cursor)
            self.output.ensureCursorVisible()

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_process_output()
        if self.stop_requested:
            self.status_label.setText("Stopped - ready for another model")
        elif exit_code == 0:
            self.status_label.setText("Finished - ready for another model")
        else:
            self.status_label.setText(f"Deployment failed (exit code {exit_code})")
        self.output.appendPlainText(f"\n[process exited with code {exit_code}]")
        self.stop_button.setEnabled(False)
        self.model_combo.setEnabled(True)
        self._model_changed(self.model_combo.currentIndex())
        if self.args.auto_start:
            QTimer.singleShot(0, QApplication.instance().quit)

    def _process_error(self, error) -> None:
        if error == QProcess.FailedToStart:
            self.status_label.setText(f"Failed to start: {self.process.errorString()}")
            self.stop_button.setEnabled(False)
            self.model_combo.setEnabled(True)
            self._model_changed(self.model_combo.currentIndex())
            if self.args.auto_start:
                QTimer.singleShot(0, QApplication.instance().quit)

    def closeEvent(self, event) -> None:
        if self.process.state() != QProcess.NotRunning:
            self.process.terminate()
            if not self.process.waitForFinished(1500):
                self.process.kill()
                self.process.waitForFinished(500)
        event.accept()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    reference = default_reference_csv()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-dir", type=Path, default=DEFAULT_CANDIDATE_DIR)
    parser.add_argument(
        "--deployment-script", type=Path, default=DEFAULT_DEPLOYMENT_SCRIPT
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--xml", type=Path, default=default_xml_path())
    parser.add_argument("--reference-csv", type=Path, default=reference)
    parser.add_argument("--initial-csv", type=Path)
    parser.add_argument("--initial-row", type=int, default=0)
    parser.add_argument("--log", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--torque-limit", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.1)
    parser.add_argument("--reservoir-warmup-steps", type=int)
    parser.add_argument("--max-joint-speed", type=float, default=30.0)
    parser.add_argument("--max-input-zscore", type=float, default=20.0)
    parser.add_argument("--enable-contacts", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--live-visualization", action="store_true")
    parser.add_argument("--visualization-fps", type=float, default=15.0)
    parser.add_argument(
        "--visualization-window-seconds", type=float, default=2.5
    )
    parser.add_argument("--visualization-backend", default="QtAgg")
    parser.add_argument(
        "--auto-start",
        action="store_true",
        help="start the selected model immediately and exit after it finishes",
    )
    args = parser.parse_args(argv)

    positive = (
        args.torque_limit,
        args.max_joint_speed,
        args.max_input_zscore,
        args.visualization_fps,
        args.visualization_window_seconds,
    )
    if any(value <= 0.0 for value in positive):
        parser.error(
            "torque-limit, max-joint-speed, max-input-zscore, and visualization "
            "values must be positive"
        )
    if args.initial_row < 0 or args.ramp_seconds < 0.0:
        parser.error("initial-row and ramp-seconds must be non-negative")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("duration must be positive")
    if args.reservoir_warmup_steps is not None and args.reservoir_warmup_steps < 0:
        parser.error("reservoir-warmup-steps must be non-negative")
    return args


def main() -> None:
    args = parse_args(sys.argv[1:])
    app = QApplication(sys.argv[:1])
    window = ScreenedReservoirTorqueWindow(args)
    window.show()
    if args.auto_start:
        QTimer.singleShot(0, window.start_selected_model)
    raise SystemExit(app.exec_())


if __name__ == "__main__":
    main()
