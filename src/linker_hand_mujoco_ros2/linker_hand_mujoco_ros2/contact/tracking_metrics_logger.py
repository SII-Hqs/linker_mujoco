"""Small CSV/summary logger for target tracking experiments."""

import csv
import json
import math
import threading
from pathlib import Path

import numpy as np


class TrackingMetricsLogger:
    """Record qpos tracking error and torque effort with a shared schema."""

    FIELDNAMES = [
        "controller",
        "time",
        "step",
        "trajectory_frame",
        "target_qpos_1",
        "target_qpos_2",
        "target_qpos_3",
        "actual_qpos_1",
        "actual_qpos_2",
        "actual_qpos_3",
        "qpos_error_1",
        "qpos_error_2",
        "qpos_error_3",
        "error_norm",
        "tau_1",
        "tau_2",
        "tau_3",
        "tau_norm",
    ]

    def __init__(
        self,
        controller_name,
        csv_path=None,
        summary_path=None,
        metadata=None,
        joint_names=("index_joint1", "index_joint2", "index_joint3"),
        start_time=None,
        end_time=None,
    ):
        self.controller_name = str(controller_name)
        self.csv_path = Path(csv_path) if csv_path is not None else None
        if summary_path is not None:
            self.summary_path = Path(summary_path)
        elif self.csv_path is not None:
            self.summary_path = self.csv_path.with_suffix(".summary.json")
        else:
            self.summary_path = None
        self.metadata = self._jsonable(metadata or {})
        self.joint_names = list(joint_names)
        self.start_time_filter = None if start_time is None else float(start_time)
        self.end_time_filter = None if end_time is None else float(end_time)
        self.lock = threading.Lock()
        self.closed = False
        self.csv_file = None
        self.writer = None

        self.sample_count = 0
        self.first_time = None
        self.last_time = None
        self.sum_error_norm = 0.0
        self.sum_error_norm_sq = 0.0
        self.max_error_norm = 0.0
        self.sum_tau_norm_sq = 0.0
        self.torque_integral = 0.0

        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            self.csv_file = self.csv_path.open("w", newline="")
            self.writer = csv.DictWriter(self.csv_file, fieldnames=self.FIELDNAMES)
            self.writer.writeheader()

    def sample(self, sim_time, trajectory_frame, target_qpos, actual_qpos, tau, dt):
        """Append one tracking sample and update summary accumulators."""
        target_qpos = np.asarray(target_qpos, dtype=float).reshape(3)
        actual_qpos = np.asarray(actual_qpos, dtype=float).reshape(3)
        tau = np.asarray(tau, dtype=float).reshape(3)
        qpos_error = actual_qpos - target_qpos
        error_norm = float(np.linalg.norm(qpos_error))
        tau_norm = float(np.linalg.norm(tau))
        sim_time = float(sim_time)
        dt = max(float(dt), 0.0)
        if (
            self.start_time_filter is not None
            and sim_time < self.start_time_filter
        ):
            return
        if self.end_time_filter is not None and sim_time > self.end_time_filter:
            return

        with self.lock:
            if self.closed:
                return
            step = self.sample_count
            self.sample_count += 1
            if self.first_time is None:
                self.first_time = sim_time
            self.last_time = sim_time
            self.sum_error_norm += error_norm
            self.sum_error_norm_sq += error_norm * error_norm
            self.max_error_norm = max(self.max_error_norm, error_norm)
            self.sum_tau_norm_sq += tau_norm * tau_norm
            self.torque_integral += tau_norm * dt

            if self.writer is not None:
                row = {
                    "controller": self.controller_name,
                    "time": sim_time,
                    "step": step,
                    "trajectory_frame": int(trajectory_frame),
                    "error_norm": error_norm,
                    "tau_norm": tau_norm,
                }
                for i in range(3):
                    suffix = i + 1
                    row[f"target_qpos_{suffix}"] = float(target_qpos[i])
                    row[f"actual_qpos_{suffix}"] = float(actual_qpos[i])
                    row[f"qpos_error_{suffix}"] = float(qpos_error[i])
                    row[f"tau_{suffix}"] = float(tau[i])
                self.writer.writerow(row)
                if step % 100 == 0:
                    self.csv_file.flush()

    def close(self):
        """Flush files and write the summary JSON once."""
        with self.lock:
            if self.closed:
                return
            self.closed = True
            summary = self.summary()
            if self.summary_path is not None:
                self.summary_path.parent.mkdir(parents=True, exist_ok=True)
                with self.summary_path.open("w") as file:
                    json.dump(summary, file, indent=2, sort_keys=True)
                    file.write("\n")
            if self.csv_file is not None:
                self.csv_file.flush()
                self.csv_file.close()

    def summary(self):
        """Return aggregate metrics for the collected samples."""
        if self.sample_count:
            tracking_rmse = math.sqrt(self.sum_error_norm_sq / self.sample_count)
            tracking_mae = self.sum_error_norm / self.sample_count
            torque_rms = math.sqrt(self.sum_tau_norm_sq / self.sample_count)
        else:
            tracking_rmse = 0.0
            tracking_mae = 0.0
            torque_rms = 0.0

        duration = 0.0
        if self.first_time is not None and self.last_time is not None:
            duration = max(0.0, self.last_time - self.first_time)

        return {
            "controller": self.controller_name,
            "csv_path": str(self.csv_path) if self.csv_path is not None else None,
            "joint_names": self.joint_names,
            "metadata": self.metadata,
            "metrics_start_time": self.start_time_filter,
            "metrics_end_time": self.end_time_filter,
            "sample_count": self.sample_count,
            "start_time": self.first_time,
            "end_time": self.last_time,
            "duration": duration,
            "tracking_rmse": tracking_rmse,
            "tracking_mae": tracking_mae,
            "tracking_max_error": self.max_error_norm,
            "torque_integral": self.torque_integral,
            "torque_rms": torque_rms,
        }

    def _jsonable(self, value):
        if isinstance(value, dict):
            return {str(key): self._jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._jsonable(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        return value
