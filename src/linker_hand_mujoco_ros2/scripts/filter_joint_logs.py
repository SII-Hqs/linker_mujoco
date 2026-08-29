#!/usr/bin/env python3
"""Filter MuJoCo joint-log qpos columns and regenerate qvel and qacc."""

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from linker_hand_mujoco_ros2.io.joint_log_processing import (  # noqa: E402
    FILTER_METHODS,
    comparison_plot,
    filter_traces,
    qpos_columns,
    read_numeric_csv,
    replace_joint_state,
    validate_timestamps,
    write_numeric_csv,
)


DEFAULT_INPUT_DIR = Path.cwd() / "results" / "joint_logs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "results" / "joint_logs_filtered"


def filter_positions(
    positions,
    sample_rate,
    method="butterworth",
    cutoff_hz=10.0,
    order=4,
    window=11,
    polyorder=3,
):
    """Filter a position matrix with the requested offline method."""
    return filter_traces(
        positions,
        sample_rate,
        method=method,
        cutoff_hz=cutoff_hz,
        order=order,
        window=window,
        polyorder=polyorder,
    )


def filter_csv(input_path, output_path, args):
    """Filter one CSV, regenerate derivatives, and optionally plot it."""
    fieldnames, values = read_numeric_csv(input_path)
    fields = qpos_columns(fieldnames)
    if not fields:
        raise ValueError(f"No qpos_* columns found: {input_path}")
    time_index = fieldnames.index("sim_time")
    timestamps = values[:, time_index]
    sample_rate = validate_timestamps(timestamps, require_uniform=True)
    indices = [fieldnames.index(field) for field in fields]
    source_positions = values[:, indices]
    filtered_positions = filter_positions(
        source_positions,
        sample_rate,
        method=args.method,
        cutoff_hz=args.cutoff_hz,
        order=args.order,
        window=args.window,
        polyorder=args.polyorder,
    )
    output_fields, output_values = replace_joint_state(
        fieldnames, values, filtered_positions
    )
    write_numeric_csv(output_path, output_fields, output_values)

    plot_path = None
    if args.plot:
        plot_dir = args.plot_dir or output_path.parent / "plots"
        plot_path = comparison_plot(
            plot_dir / f"{output_path.stem}_comparison.png",
            timestamps,
            source_positions,
            timestamps,
            filtered_positions,
            [field.removeprefix("qpos_") for field in fields],
            "original",
            args.method,
        )
    return len(values), len(fields), sample_rate, plot_path


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
        "--method", choices=FILTER_METHODS, default="butterworth"
    )
    parser.add_argument("--cutoff-hz", type=float, default=10.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path)
    return parser.parse_args(argv)


def output_path_for(input_path, output, method, input_count):
    """Resolve the output CSV path for one file in a batch."""
    if input_count == 1 and output.suffix.lower() == ".csv":
        return output
    return output / f"{input_path.stem}_{method}.csv"


def main(argv=None):
    """Filter one joint-log CSV or all CSV files in one directory."""
    args = parse_args(argv)
    inputs = (
        [args.input]
        if args.input.is_file()
        else sorted(args.input.glob("*.csv"))
    )
    if not inputs:
        raise FileNotFoundError(f"No CSV files found: {args.input}")
    if len(inputs) > 1 and args.output.suffix.lower() == ".csv":
        raise ValueError("Output must be a directory for multiple input files")

    for index, input_path in enumerate(inputs, start=1):
        output_path = output_path_for(
            input_path, args.output, args.method, len(inputs)
        )
        sample_count, joint_count, sample_rate, plot_path = filter_csv(
            input_path, output_path, args
        )
        message = (
            f"[{index}/{len(inputs)}] {input_path.name}: "
            f"{sample_count} samples, "
            f"{joint_count} joints at {sample_rate:.6g} Hz: {output_path}"
        )
        if plot_path is not None:
            message += f"; plot={plot_path}"
        print(message, flush=True)


if __name__ == "__main__":
    main()
