#!/usr/bin/env python3
"""Smooth inverse torque and reconstruct joint state with MuJoCo dynamics."""

import argparse
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from linker_hand_mujoco_ros2.io.joint_log_processing import (  # noqa: E402
    FILTER_METHODS,
    differentiate_positions,
    filter_traces,
    read_numeric_csv,
    validate_timestamps,
    write_numeric_csv,
)
from linker_hand_mujoco_ros2.tools.calculate_joint_log_torque import (  # noqa
    infer_hand_from_log_name,
    resolve_joint_addresses,
    resolve_model_xml,
)


DEFAULT_INPUT_DIR = (
    Path.cwd()
    / "results"
    / "joint_torques"
    / "movement_without_thumb_joint0_100hz_pchip_filtered"
)
DEFAULT_OUTPUT_DIR = Path.cwd() / "results" / "joint_state_reconstruction"
FLEXION_JOINT = re.compile(r"^(index|middle|ring|little)_joint[1-9][0-9]*$")


def smoothing_joint_indices(joint_names, scope="flexion"):
    """Return joints selected for torque smoothing."""
    if scope == "all":
        return list(range(len(joint_names)))
    if scope != "flexion":
        raise ValueError(f"Unsupported joint scope: {scope}")
    indices = [
        index
        for index, joint_name in enumerate(joint_names)
        if FLEXION_JOINT.fullmatch(joint_name)
    ]
    if not indices:
        raise ValueError("No non-thumb finger joint1+ joints found")
    return indices


def read_torque_state(fieldnames, values):
    """Extract matched torque, position, and optional velocity columns."""
    torque_fields = [
        field for field in fieldnames if field.startswith("tau_inverse_")
    ]
    if not torque_fields:
        raise ValueError("No tau_inverse_* columns found")
    joint_names = [
        field.removeprefix("tau_inverse_") for field in torque_fields
    ]
    qpos_fields = [f"qpos_{name}" for name in joint_names]
    missing_qpos = [field for field in qpos_fields if field not in fieldnames]
    if missing_qpos:
        raise ValueError(f"Missing position columns: {missing_qpos}")

    qvel_fields = [f"qvel_{name}" for name in joint_names]
    qpos = values[:, [fieldnames.index(field) for field in qpos_fields]]
    torque = values[:, [fieldnames.index(field) for field in torque_fields]]
    qvel = None
    if all(field in fieldnames for field in qvel_fields):
        qvel = values[:, [fieldnames.index(field) for field in qvel_fields]]
    timestamps = values[:, fieldnames.index("sim_time")]
    return timestamps, joint_names, qpos, qvel, torque


def reconstruct_state(
    model_xml,
    joint_names,
    timestamps,
    initial_qpos,
    initial_qvel,
    torque,
    damping=0.8,
    integration_timestep=None,
):
    """Integrate MuJoCo forward dynamics from one measured initial state."""
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    model.dof_damping[:] = float(damping)
    model.opt.disableflags = (
        1 | int(mujoco.mjtDisableBit.mjDSBL_ACTUATION)
    )
    native_timestep = float(model.opt.timestep)
    step_limit = (
        native_timestep
        if integration_timestep is None
        else float(integration_timestep)
    )
    if not np.isfinite(step_limit) or step_limit <= 0.0:
        raise ValueError("integration-timestep must be positive and finite")

    data = mujoco.MjData(model)
    qpos_ids, dof_ids = resolve_joint_addresses(model, joint_names)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[qpos_ids] = initial_qpos
    data.qvel[dof_ids] = initial_qvel

    sample_count = len(timestamps)
    reconstructed_qpos = np.empty((sample_count, len(joint_names)))
    reconstructed_qvel = np.empty_like(reconstructed_qpos)
    reconstructed_qacc = np.empty_like(reconstructed_qpos)

    def store_state(index, applied_torque):
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof_ids] = applied_torque
        mujoco.mj_forward(model, data)
        reconstructed_qpos[index] = data.qpos[qpos_ids]
        reconstructed_qvel[index] = data.qvel[dof_ids]
        reconstructed_qacc[index] = data.qacc[dof_ids]

    store_state(0, torque[0])
    for sample_index in range(sample_count - 1):
        interval = float(
            timestamps[sample_index + 1] - timestamps[sample_index]
        )
        substep_count = max(1, int(math.ceil(interval / step_limit)))
        model.opt.timestep = interval / substep_count
        for substep_index in range(substep_count):
            weight = (substep_index + 0.5) / substep_count
            applied_torque = (
                (1.0 - weight) * torque[sample_index]
                + weight * torque[sample_index + 1]
            )
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[dof_ids] = applied_torque
            mujoco.mj_step(model, data)
        store_state(sample_index + 1, torque[sample_index + 1])
        if not (
            np.all(np.isfinite(reconstructed_qpos[sample_index + 1]))
            and np.all(np.isfinite(reconstructed_qvel[sample_index + 1]))
            and np.all(np.isfinite(reconstructed_qacc[sample_index + 1]))
        ):
            raise FloatingPointError(
                f"Non-finite reconstructed state at sample {sample_index + 1}"
            )

    return reconstructed_qpos, reconstructed_qvel, reconstructed_qacc


def append_reconstruction_columns(
    fieldnames,
    values,
    joint_names,
    smoothed_torque,
    qpos,
    qvel,
    qacc,
):
    """Append smoothed torque and reconstructed state to source columns."""
    output_fields = list(fieldnames)
    columns = {
        field: values[:, index].copy()
        for index, field in enumerate(fieldnames)
    }
    groups = (
        ("tau_smoothed", smoothed_torque),
        ("qpos_reconstructed", qpos),
        ("qvel_reconstructed", qvel),
        ("qacc_reconstructed", qacc),
    )
    for prefix, group_values in groups:
        for joint_index, joint_name in enumerate(joint_names):
            field = f"{prefix}_{joint_name}"
            output_fields.append(field)
            columns[field] = group_values[:, joint_index]
    output_values = np.column_stack(
        [columns[field] for field in output_fields]
    )
    return output_fields, output_values


def plot_reconstruction(
    output_path,
    timestamps,
    joint_names,
    original_qpos,
    reconstructed_qpos,
    original_torque,
    smoothed_torque,
    source_name="",
):
    """Plot original/reconstructed angle and raw/smoothed torque per joint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column_count = 4
    row_count = math.ceil(len(joint_names) / column_count)
    figure, axes = plt.subplots(
        row_count * 2,
        column_count,
        figsize=(20, max(4.5 * row_count, 6.0)),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(row_count * 2, column_count)
    relative_time = timestamps - timestamps[0]
    legend_handles = None

    for joint_index, joint_name in enumerate(joint_names):
        group_row = joint_index // column_count
        column = joint_index % column_count
        torque_axis = axes[group_row * 2, column]
        angle_axis = axes[group_row * 2 + 1, column]
        raw_torque = torque_axis.plot(
            relative_time,
            original_torque[:, joint_index],
            color="salmon",
            linewidth=0.9,
            alpha=0.75,
            label="raw torque",
        )[0]
        filtered_torque = torque_axis.plot(
            relative_time,
            smoothed_torque[:, joint_index],
            color="darkred",
            linewidth=1.4,
            label="smoothed torque",
        )[0]
        torque_axis.fill_between(
            relative_time,
            original_torque[:, joint_index],
            smoothed_torque[:, joint_index],
            color="goldenrod",
            alpha=0.20,
            linewidth=0.0,
        )
        raw_roughness = float(
            np.sqrt(
                np.mean(
                    np.square(np.diff(original_torque[:, joint_index]))
                )
            )
        )
        smooth_roughness = float(
            np.sqrt(
                np.mean(
                    np.square(np.diff(smoothed_torque[:, joint_index]))
                )
            )
        )
        roughness_reduction = (
            100.0 * (1.0 - smooth_roughness / raw_roughness)
            if raw_roughness > 0.0
            else 0.0
        )
        original_angle = angle_axis.plot(
            relative_time,
            original_qpos[:, joint_index],
            color="tab:blue",
            linewidth=0.9,
            alpha=0.55,
            label="original angle",
        )[0]
        reconstructed_angle = angle_axis.plot(
            relative_time,
            reconstructed_qpos[:, joint_index],
            color="navy",
            linewidth=1.1,
            linestyle="--",
            label="reconstructed angle",
        )[0]
        legend_handles = (
            raw_torque,
            filtered_torque,
            original_angle,
            reconstructed_angle,
        )
        torque_axis.set_title(
            f"{joint_name} | roughness -{roughness_reduction:.1f}%",
            fontsize=9,
        )
        torque_axis.set_ylabel("Torque (N*m)", color="darkred", fontsize=8)
        torque_axis.tick_params(axis="y", labelcolor="darkred", labelsize=7)
        torque_axis.grid(True, alpha=0.25)
        angle_axis.set_xlabel("Time (s)")
        angle_axis.set_ylabel("Angle (rad)", color="navy", fontsize=8)
        angle_axis.tick_params(axis="y", labelcolor="navy", labelsize=7)
        angle_axis.grid(True, alpha=0.25)

    for unused_index in range(len(joint_names), row_count * column_count):
        group_row = unused_index // column_count
        column = unused_index % column_count
        axes[group_row * 2, column].set_visible(False)
        axes[group_row * 2 + 1, column].set_visible(False)
    title = "Smoothed torque and forward-dynamics state reconstruction"
    if source_name:
        title += f": {source_name}"
    figure.suptitle(title, fontsize=12)
    figure.legend(
        legend_handles,
        [
            "raw torque",
            "smoothed torque",
            "original angle",
            "reconstructed angle",
        ],
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 0.975),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def process_csv(input_path, output_path, args, plot_path=None):
    """Smooth, reconstruct, save, and optionally plot one torque CSV."""
    fieldnames, values = read_numeric_csv(input_path)
    timestamps, joint_names, qpos, input_qvel, torque = read_torque_state(
        fieldnames, values
    )
    sample_rate = validate_timestamps(timestamps, require_uniform=True)
    smooth_indices = smoothing_joint_indices(
        joint_names, scope=args.joint_scope
    )
    smoothed_torque = torque.copy()
    smoothed_torque[:, smooth_indices] = filter_traces(
        torque[:, smooth_indices],
        sample_rate,
        method=args.method,
        cutoff_hz=args.cutoff_hz,
        order=args.order,
        window=args.window,
        polyorder=args.polyorder,
    )
    if input_qvel is None:
        input_qvel, _ = differentiate_positions(qpos, timestamps)

    inferred_joint, inferred_type = infer_hand_from_log_name(input_path)
    hand_joint = args.hand_joint or inferred_joint
    hand_type = args.hand_type or inferred_type
    model_xml = resolve_model_xml(hand_joint, hand_type, args.model_xml)
    reconstructed = reconstruct_state(
        model_xml,
        joint_names,
        timestamps,
        qpos[0],
        input_qvel[0],
        smoothed_torque,
        damping=args.damping,
        integration_timestep=args.integration_timestep,
    )
    reconstructed_qpos, reconstructed_qvel, reconstructed_qacc = reconstructed
    output_fields, output_values = append_reconstruction_columns(
        fieldnames,
        values,
        joint_names,
        smoothed_torque,
        reconstructed_qpos,
        reconstructed_qvel,
        reconstructed_qacc,
    )
    write_numeric_csv(output_path, output_fields, output_values)
    if plot_path is not None:
        plot_indices = (
            list(range(len(joint_names)))
            if args.plot_all_joints
            else smooth_indices
        )
        plot_reconstruction(
            plot_path,
            timestamps,
            [joint_names[index] for index in plot_indices],
            qpos[:, plot_indices],
            reconstructed_qpos[:, plot_indices],
            torque[:, plot_indices],
            smoothed_torque[:, plot_indices],
            source_name=Path(input_path).name,
        )
    rmse = float(np.sqrt(np.mean(np.square(reconstructed_qpos - qpos))))
    return (
        len(values),
        len(joint_names),
        len(smooth_indices),
        sample_rate,
        model_xml,
        rmse,
    )


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
    parser.add_argument("--cutoff-hz", type=float, default=2.0)
    parser.add_argument("--order", type=int, default=4)
    parser.add_argument("--window", type=int, default=11)
    parser.add_argument("--polyorder", type=int, default=3)
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument("--integration-timestep", type=float, default=None)
    parser.add_argument("--model-xml", default="")
    parser.add_argument("--hand-joint", default="")
    parser.add_argument("--hand-type", default="")
    parser.add_argument(
        "--joint-scope",
        choices=("flexion", "all"),
        default="flexion",
        help=(
            "Joints to smooth: flexion selects index/middle/ring/little "
            "joint1+ and excludes the thumb and joint0 (default: flexion)."
        ),
    )
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot-dir", type=Path)
    parser.add_argument(
        "--plot-all-joints",
        action="store_true",
        help="Plot all joints instead of only the smoothed joints.",
    )
    return parser.parse_args(argv)


def main(argv=None):
    """Process one torque CSV or every torque CSV in one directory."""
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
        suffix = f"torque_{args.method}_reconstruction"
        input_stem = input_path.stem.removesuffix("_inverse_torque")
        output_path = (
            args.output
            if len(inputs) == 1 and args.output.suffix.lower() == ".csv"
            else args.output / f"{input_stem}_{suffix}.csv"
        )
        plot_path = None
        if args.plot:
            plot_dir = args.plot_dir or output_path.parent / "plots"
            plot_path = plot_dir / f"{output_path.stem}.png"
        result = process_csv(input_path, output_path, args, plot_path)
        (
            sample_count,
            joint_count,
            smooth_count,
            sample_rate,
            model_xml,
            rmse,
        ) = result
        print(
            f"[{index}/{len(inputs)}] {input_path.name}: "
            f"{sample_count} samples, "
            f"{smooth_count}/{joint_count} smoothed joints at "
            f"{sample_rate:.6g} Hz, "
            f"angle RMSE={rmse:.6g} rad: {output_path} "
            f"(model={model_xml})"
            + (f"; plot={plot_path}" if plot_path is not None else ""),
            flush=True,
        )


if __name__ == "__main__":
    main()
