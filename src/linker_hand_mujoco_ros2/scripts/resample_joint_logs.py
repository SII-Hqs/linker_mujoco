#!/usr/bin/env python3
"""Resample MuJoCo joint-state CSV logs to a fixed frequency."""

import argparse
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from linker_hand_mujoco_ros2.io.joint_log_processing import (  # noqa: E402
    INTERPOLATION_METHODS,
    comparison_plot,
    qpos_columns,
    read_numeric_csv,
    resample_joint_log,
    write_numeric_csv,
)


DEFAULT_INPUT_DIR = Path.cwd() / "results" / "joint_logs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "results" / "joint_logs_1000hz"


def resample_csv(input_path, output_path, frequency, method="linear"):
    """Resample one CSV and return source/output counts and duration."""
    fieldnames, values = read_numeric_csv(input_path)
    output_fields, output_values, duration = resample_joint_log(
        fieldnames, values, frequency, method
    )
    write_numeric_csv(output_path, output_fields, output_values)
    return len(values), len(output_values), duration


def parse_args(argv=None):
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, default=DEFAULT_INPUT_DIR
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=DEFAULT_OUTPUT_DIR
    )
    parser.add_argument("--frequency", type=float, default=1000.0)
    parser.add_argument(
        "--method",
        choices=INTERPOLATION_METHODS,
        default="linear",
        help="Interpolation method for qpos columns (default: linear).",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save one original/interpolated position comparison per CSV.",
    )
    parser.add_argument(
        "--plot-dir",
        type=Path,
        help="Plot directory (default: <output directory>/plots).",
    )
    return parser.parse_args(argv)


def frequency_label(frequency):
    """Format a sampling frequency for output filenames."""
    if float(frequency).is_integer():
        return f"{int(frequency)}hz"
    return f"{frequency:g}hz".replace(".", "p")


def main(argv=None):
    """Resample one joint-log CSV or all CSV files in a directory."""
    args = parse_args(argv)
    if not np.isfinite(args.frequency) or args.frequency <= 0.0:
        raise ValueError("frequency must be a positive finite number")

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

    output_suffix = f"{frequency_label(args.frequency)}_{args.method}"

    for index, input_path in enumerate(inputs, start=1):
        output_path = (
            args.output
            if len(inputs) == 1 and args.output.suffix.lower() == ".csv"
            else args.output / f"{input_path.stem}_{output_suffix}.csv"
        )
        source_count, output_count, duration = resample_csv(
            input_path, output_path, args.frequency, args.method
        )
        if args.plot:
            source_fields, source_values = read_numeric_csv(input_path)
            output_fields, output_values = read_numeric_csv(output_path)
            source_qpos = qpos_columns(source_fields)
            output_qpos = qpos_columns(output_fields)
            if source_qpos != output_qpos:
                raise ValueError(
                    "Output qpos columns differ from input columns"
                )
            plot_dir = args.plot_dir or output_path.parent / "plots"
            source_indices = [
                source_fields.index(field) for field in source_qpos
            ]
            output_indices = [
                output_fields.index(field) for field in output_qpos
            ]
            comparison_plot(
                plot_dir / f"{output_path.stem}_comparison.png",
                source_values[:, source_fields.index("sim_time")],
                source_values[:, source_indices],
                output_values[:, output_fields.index("sim_time")],
                output_values[:, output_indices],
                [field.removeprefix("qpos_") for field in source_qpos],
                "original",
                f"{args.method} {args.frequency:g} Hz",
                source_points=True,
            )
        print(
            f"[{index}/{len(inputs)}] {input_path.name}: {source_count} -> "
            f"{output_count} samples over {duration:.6f} s: {output_path}",
            flush=True,
        )


if __name__ == "__main__":
    main()
