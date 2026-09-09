"""Shared processing helpers for numeric MuJoCo joint-log CSV files."""

import csv
import math
from pathlib import Path

import numpy as np
from scipy.ndimage import median_filter
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter
from scipy.signal import savgol_filter
from scipy.signal import sosfiltfilt


INTERPOLATION_METHODS = ("linear", "nearest", "cubic", "pchip", "akima")
FILTER_METHODS = ("butterworth", "savgol", "moving-average", "median")


def read_numeric_csv(input_path):
    """Read a CSV whose columns are all finite numeric values."""
    input_path = Path(input_path)
    with input_path.open(newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if len(rows) < 2:
        raise ValueError(f"At least two samples are required: {input_path}")
    if "sim_time" not in fieldnames:
        raise ValueError(f"Missing sim_time column: {input_path}")

    try:
        values = np.asarray(
            [[float(row[field]) for field in fieldnames] for row in rows],
            dtype=float,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"All CSV columns must be numeric: {input_path}"
        ) from error
    if not np.all(np.isfinite(values)):
        raise ValueError(f"CSV contains non-finite values: {input_path}")
    return fieldnames, values


def write_numeric_csv(output_path, fieldnames, values):
    """Write a numeric matrix to a CSV and create its parent directory."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(fieldnames)
        writer.writerows(values)
    return output_path


def qpos_columns(fieldnames):
    """Return joint-position columns in their source order."""
    return [field for field in fieldnames if field.startswith("qpos_")]


def validate_timestamps(timestamps, require_uniform=False):
    """Validate a finite, increasing timestamp vector and return its rate."""
    timestamps = np.asarray(timestamps, dtype=float)
    if not np.all(np.isfinite(timestamps)):
        raise ValueError("sim_time must contain only finite values")
    intervals = np.diff(timestamps)
    if np.any(intervals <= 0.0):
        raise ValueError("sim_time must be strictly increasing")
    interval = float(np.median(intervals))
    if require_uniform:
        tolerance = max(1e-9, interval * 1e-5)
        if not np.allclose(intervals, interval, rtol=0.0, atol=tolerance):
            raise ValueError(
                "sim_time is not uniformly sampled; resample it first"
            )
    return 1.0 / interval


def differentiate_positions(positions, timestamps):
    """Calculate velocity and acceleration from positions and timestamps."""
    positions = np.asarray(positions, dtype=float)
    timestamps = np.asarray(timestamps, dtype=float)
    validate_timestamps(timestamps)
    edge_order = 2 if len(positions) > 2 else 1
    qvel = np.gradient(
        positions, timestamps, axis=0, edge_order=edge_order
    )
    qacc = np.gradient(qvel, timestamps, axis=0, edge_order=edge_order)
    return qvel, qacc


def validated_filter_window(window, sample_count, method, polyorder=0):
    """Return a valid odd window size for a sample-domain filter."""
    window = int(window)
    if window < 3:
        raise ValueError(f"{method} window must be at least 3")
    if window % 2 == 0:
        window += 1
    if window > sample_count:
        raise ValueError(
            f"{method} window {window} exceeds sample count {sample_count}"
        )
    if method == "savgol" and polyorder >= window:
        raise ValueError("savgol polyorder must be smaller than its window")
    return window


def moving_average(values, window):
    """Apply an edge-padded centered moving average to numeric columns."""
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.column_stack(
        [
            np.convolve(padded[:, index], kernel, mode="valid")
            for index in range(values.shape[1])
        ]
    )


def filter_traces(
    values,
    sample_rate,
    method="butterworth",
    cutoff_hz=10.0,
    order=4,
    window=11,
    polyorder=3,
):
    """Filter a sample-by-channel matrix with an offline filter."""
    values = np.asarray(values, dtype=float)
    if method not in FILTER_METHODS:
        raise ValueError(f"Unsupported filter method: {method}")
    if values.ndim != 2 or not len(values):
        raise ValueError("values must be a non-empty two-dimensional array")

    if method == "butterworth":
        if not np.isfinite(cutoff_hz) or cutoff_hz <= 0.0:
            raise ValueError("cutoff-hz must be a positive finite number")
        nyquist = sample_rate / 2.0
        if cutoff_hz >= nyquist:
            raise ValueError(
                f"cutoff-hz must be below Nyquist frequency {nyquist:g} Hz"
            )
        if int(order) < 1:
            raise ValueError("Butterworth order must be at least 1")
        sections = butter(
            int(order), cutoff_hz, fs=sample_rate, output="sos"
        )
        try:
            return sosfiltfilt(sections, values, axis=0)
        except ValueError as error:
            raise ValueError(
                "Input is too short for the requested Butterworth order"
            ) from error

    window = validated_filter_window(
        window, len(values), method, polyorder
    )
    if method == "savgol":
        if int(polyorder) < 0:
            raise ValueError("savgol polyorder cannot be negative")
        return savgol_filter(
            values,
            window_length=window,
            polyorder=int(polyorder),
            axis=0,
            mode="interp",
        )
    if method == "moving-average":
        return moving_average(values, window)
    return median_filter(values, size=(window, 1), mode="nearest")


def replace_joint_state(fieldnames, values, positions):
    """Replace qpos and regenerate matching qvel and qacc columns."""
    qpos_fields = qpos_columns(fieldnames)
    if not qpos_fields:
        raise ValueError("No qpos_* columns found")
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(values), len(qpos_fields)):
        raise ValueError(
            "Position shape does not match the CSV samples and qpos columns"
        )

    time_index = fieldnames.index("sim_time")
    timestamps = values[:, time_index]
    qvel, qacc = differentiate_positions(positions, timestamps)
    output_fields = list(fieldnames)
    columns = {
        field: values[:, index].copy()
        for index, field in enumerate(fieldnames)
    }

    for index, qpos_field in enumerate(qpos_fields):
        joint_name = qpos_field.removeprefix("qpos_")
        qvel_field = f"qvel_{joint_name}"
        qacc_field = f"qacc_{joint_name}"
        columns[qpos_field] = positions[:, index]
        columns[qvel_field] = qvel[:, index]
        columns[qacc_field] = qacc[:, index]
        if qvel_field not in output_fields:
            output_fields.append(qvel_field)
        if qacc_field not in output_fields:
            output_fields.append(qacc_field)

    output_values = np.column_stack(
        [columns[field] for field in output_fields]
    )
    return output_fields, output_values


def interpolate_trace(source_time, source_values, target_time, method):
    """Interpolate one trace without extrapolating beyond source endpoints."""
    if method not in INTERPOLATION_METHODS:
        raise ValueError(f"Unsupported interpolation method: {method}")
    if method == "linear":
        return np.interp(target_time, source_time, source_values)
    if method == "nearest":
        upper = np.searchsorted(source_time, target_time, side="left")
        upper = np.clip(upper, 0, len(source_time) - 1)
        lower = np.clip(upper - 1, 0, len(source_time) - 1)
        use_upper = (
            np.abs(source_time[upper] - target_time)
            < np.abs(target_time - source_time[lower])
        )
        indices = np.where(use_upper, upper, lower)
        return source_values[indices]
    if method == "cubic":
        if len(source_time) < 4:
            raise ValueError(
                "cubic interpolation requires at least four samples"
            )
        return CubicSpline(source_time, source_values)(target_time)
    if method == "pchip":
        return PchipInterpolator(source_time, source_values)(target_time)
    if len(source_time) < 3:
        raise ValueError("akima interpolation requires at least three samples")
    return Akima1DInterpolator(source_time, source_values)(target_time)


def resample_joint_log(fieldnames, values, frequency, method="linear"):
    """Resample qpos and regenerate matching velocity and acceleration."""
    if not np.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency must be a positive finite number")
    time_index = fieldnames.index("sim_time")
    source_time = values[:, time_index]
    validate_timestamps(source_time)
    source_offset = source_time - source_time[0]
    duration = float(source_offset[-1])
    interval = 1.0 / float(frequency)
    step_count = int(np.floor(duration * frequency + 1e-6))
    target_offset = np.arange(step_count + 1, dtype=float) * interval
    target_values = np.empty(
        (len(target_offset), len(fieldnames)), dtype=float
    )
    qpos_fields = set(qpos_columns(fieldnames))

    for column_index, field in enumerate(fieldnames):
        if field == "sim_time":
            target_values[:, column_index] = source_time[0] + target_offset
        elif field in qpos_fields:
            target_values[:, column_index] = interpolate_trace(
                source_offset,
                values[:, column_index],
                target_offset,
                method,
            )
        else:
            target_values[:, column_index] = np.interp(
                target_offset, source_offset, values[:, column_index]
            )

    ordered_positions = np.column_stack(
        [
            target_values[:, fieldnames.index(field)]
            for field in qpos_columns(fieldnames)
        ]
    )
    output_fields, output_values = replace_joint_state(
        fieldnames, target_values, ordered_positions
    )
    return output_fields, output_values, duration


def comparison_plot(
    output_path,
    source_time,
    source_positions,
    processed_time,
    processed_positions,
    joint_names,
    source_label,
    processed_label,
    source_points=False,
):
    """Draw one position-comparison subplot per joint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column_count = 5
    row_count = math.ceil(len(joint_names) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(20, max(3.0 * row_count, 4.0)),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)
    source_relative = np.asarray(source_time) - source_time[0]
    processed_relative = np.asarray(processed_time) - processed_time[0]

    for index, joint_name in enumerate(joint_names):
        axis = axes[index]
        if source_points:
            axis.scatter(
                source_relative,
                source_positions[:, index],
                s=8,
                alpha=0.65,
                label=source_label,
            )
        else:
            axis.plot(
                source_relative,
                source_positions[:, index],
                linewidth=1.0,
                alpha=0.75,
                label=source_label,
            )
        axis.plot(
            processed_relative,
            processed_positions[:, index],
            linewidth=1.2,
            label=processed_label,
        )
        axis.set_title(joint_name, fontsize=9)
        axis.set_ylabel("Angle (rad)")
        axis.grid(True, alpha=0.25)

    for axis in axes[len(joint_names):]:
        axis.set_visible(False)
    for axis in axes[:len(joint_names)]:
        axis.set_xlabel("Time (s)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path
