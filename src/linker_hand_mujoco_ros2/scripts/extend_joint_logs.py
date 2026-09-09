#!/usr/bin/env python3
"""Extend numeric joint-log CSV files with a static final-pose hold."""

import argparse
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))
SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from linker_hand_mujoco_ros2.io.joint_log_processing import (  # noqa: E402
    read_numeric_csv,
    validate_timestamps,
    write_numeric_csv,
)
from reconstruct_joint_state_from_torque import (  # noqa: E402
    plot_reconstruction,
    smoothing_joint_indices,
)


DEFAULT_INPUT_DIR = (
    Path.cwd()
    / "results"
    / "joint_state_reconstruction"
    / "movement_without_thumb_joint0_flexion_2hz"
)
DEFAULT_OUTPUT_DIR = Path(f"{DEFAULT_INPUT_DIR}_extended_2s")


def extend_final_pose(fieldnames, values, duration):
    """Append a static final-pose hold and return data and hold metadata."""
    duration = float(duration)
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be a positive finite number")

    time_index = fieldnames.index("sim_time")
    sample_rate = validate_timestamps(
        values[:, time_index], require_uniform=True
    )
    interval = 1.0 / sample_rate
    extension_count = int(round(duration * sample_rate))
    if extension_count < 1:
        raise ValueError("duration is shorter than one sample interval")
    actual_duration = extension_count * interval

    extension = np.repeat(values[-1:, :], extension_count, axis=0)
    offsets = interval * np.arange(1, extension_count + 1, dtype=float)
    for time_field in ("sim_time", "wall_time"):
        if time_field in fieldnames:
            index = fieldnames.index(time_field)
            extension[:, index] = values[-1, index] + offsets

    for index, field in enumerate(fieldnames):
        if field.startswith(("qvel_", "qacc_")):
            extension[:, index] = 0.0

    return (
        np.vstack((values, extension)),
        extension_count,
        sample_rate,
        actual_duration,
    )


def extend_csv(input_path, output_path, duration):
    """Extend one CSV while preserving every original sample."""
    fieldnames, values = read_numeric_csv(input_path)
    extended, count, sample_rate, actual_duration = extend_final_pose(
        fieldnames, values, duration
    )
    write_numeric_csv(output_path, fieldnames, extended)
    return len(values), len(extended), count, sample_rate, actual_duration


def plot_extended_csv(input_path, output_path):
    """Plot torque smoothing and reconstructed angles from an extended CSV."""
    fieldnames, values = read_numeric_csv(input_path)
    joint_names = [
        field.removeprefix("tau_inverse_")
        for field in fieldnames
        if field.startswith("tau_inverse_")
    ]
    if not joint_names:
        raise ValueError(f"No tau_inverse_* columns found: {input_path}")
    plot_indices = smoothing_joint_indices(joint_names, scope="flexion")

    def columns(prefix):
        names = [f"{prefix}_{name}" for name in joint_names]
        missing = [name for name in names if name not in fieldnames]
        if missing:
            raise ValueError(f"Missing columns in {input_path}: {missing}")
        return values[:, [fieldnames.index(name) for name in names]]

    selected_names = [joint_names[index] for index in plot_indices]
    timestamps = values[:, fieldnames.index("sim_time")]
    plot_reconstruction(
        output_path,
        timestamps,
        selected_names,
        columns("qpos")[:, plot_indices],
        columns("qpos_reconstructed")[:, plot_indices],
        columns("tau_inverse")[:, plot_indices],
        columns("tau_smoothed")[:, plot_indices],
        source_name=Path(input_path).name,
    )
    return output_path


def parse_args(argv=None):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT_DIR
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Final-pose hold duration in seconds (default: 2).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Plot original/smoothed torque and original/reconstructed angle.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Plot directory (default: <output directory>/plots).",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Extend one joint-log CSV or all CSV files in one directory."""
    args = parse_args(argv)
    inputs = (
        [args.input]
        if args.input.is_file()
        else sorted(args.input.glob("*.csv"))
    )
    if not inputs:
        raise FileNotFoundError(f"No CSV files found: {args.input}")
    if len(inputs) > 1 and args.output.suffix.lower() == ".csv":
        raise ValueError(
            "Output must be a directory when processing multiple files"
        )

    for index, input_path in enumerate(inputs, start=1):
        output_path = (
            args.output
            if len(inputs) == 1 and args.output.suffix.lower() == ".csv"
            else args.output / input_path.name
        )
        source_count, output_count, added, rate, duration = extend_csv(
            input_path, output_path, args.duration
        )
        plot_path = None
        if args.plot:
            plot_dir = args.plot_dir or output_path.parent / "plots"
            plot_path = plot_dir / f"{output_path.stem}.png"
            plot_extended_csv(output_path, plot_path)
        print(
            f"[{index}/{len(inputs)}] {input_path.name}: "
            f"{source_count} -> {output_count} samples "
            f"(+{added} at {rate:.6g} Hz, {duration:.6g} s): "
            f"{output_path}"
            + (f"; plot={plot_path}" if plot_path is not None else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
