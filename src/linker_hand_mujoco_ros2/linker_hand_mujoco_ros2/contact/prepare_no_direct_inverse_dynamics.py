#!/usr/bin/env python3
"""Prepare 60 Hz no-direct-readout inputs and MuJoCo torque targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

if __package__:
    from .contact_control_demo_esn import add_esn_arguments, build_esn_simulation
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo_esn import (
        add_esn_arguments,
        build_esn_simulation,
    )


EXPECTED_COLUMNS = (
    "qpos_index_joint1",
    "qpos_index_joint2",
    "qpos_index_joint3",
    "qvel_index_joint1",
    "qvel_index_joint2",
    "qvel_index_joint3",
    "force_g",
)
JOINT_NAMES = ("index_joint1", "index_joint2", "index_joint3")


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Validate a dense seven-column capture, estimate acceleration, "
            "resample to 60 Hz, and calculate MuJoCo inverse-dynamics torques."
        )
    )
    add_esn_arguments(parser)
    parser.set_defaults(horizon=0)
    parser.add_argument("input", type=Path, help="Collector NPZ file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Prepared NPZ path. Defaults next to the input file.",
    )
    parser.add_argument(
        "--savgol-window",
        type=int,
        default=51,
        help="Odd 500 Hz smoothing window length (default: 51 samples).",
    )
    parser.add_argument(
        "--savgol-order",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order (default: 3).",
    )
    parser.add_argument(
        "--forward-replay",
        action="store_true",
        help="Also run a full open-loop forward replay of the torque sequence.",
    )
    return parser.parse_args(argv)


def _odd_window(requested, sample_count, order):
    window = min(int(requested), int(sample_count))
    if window % 2 == 0:
        window -= 1
    minimum = int(order) + 2
    if minimum % 2 == 0:
        minimum += 1
    if window < minimum:
        raise ValueError(
            f"not enough samples for Savitzky-Golay order {order}: {sample_count}"
        )
    return window


def _local_polynomial_filter(values, window, order, dt, derivative=0):
    """Apply a centered Savitzky-Golay filter using NumPy only."""
    values = np.asarray(values, dtype=float)
    half_window = window // 2
    offsets = np.arange(-half_window, half_window + 1, dtype=float) * float(dt)
    design = np.vander(offsets, N=order + 1, increasing=True)
    coefficients = np.linalg.pinv(design)[int(derivative)]
    padded = np.pad(values, ((half_window, half_window), (0, 0)), mode="reflect")
    windows = np.lib.stride_tricks.sliding_window_view(
        padded, window_shape=window, axis=0
    )
    return np.tensordot(windows, coefficients, axes=([-1], [0]))


def _check_time(name, times):
    times = np.asarray(times, dtype=float).reshape(-1)
    if len(times) < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    differences = np.diff(times)
    if not np.all(np.isfinite(times)) or np.any(differences <= 0.0):
        raise ValueError(f"{name} timestamps must be finite and strictly increasing")
    return differences


def load_and_validate(path):
    path = path.expanduser().resolve()
    with np.load(path, allow_pickle=False) as model:
        required = {
            "warmup_time",
            "warmup_inputs",
            "training_time",
            "training_inputs",
            "feature_columns",
            "simulation_timestep",
            "esn_fs_hz",
            "obstacle_removed",
        }
        missing = sorted(required.difference(model.files))
        if missing:
            raise KeyError(f"capture is missing keys: {missing}")
        capture = {key: np.asarray(model[key]).copy() for key in model.files}

    columns = tuple(str(value) for value in capture["feature_columns"].tolist())
    if columns != EXPECTED_COLUMNS:
        raise ValueError(f"unexpected feature columns: {columns}")
    for name in ("warmup_inputs", "training_inputs"):
        values = np.asarray(capture[name], dtype=float)
        if values.ndim != 2 or values.shape[1] != 7:
            raise ValueError(f"{name} must have shape (N, 7), got {values.shape}")
        if not np.all(np.isfinite(values)):
            raise FloatingPointError(f"{name} contains NaN or infinite values")
        capture[name] = values

    warmup_dt = _check_time("warmup_time", capture["warmup_time"])
    training_dt = _check_time("training_time", capture["training_time"])
    if len(capture["warmup_time"]) != len(capture["warmup_inputs"]):
        raise ValueError("warmup timestamps and inputs have different lengths")
    if len(capture["training_time"]) != len(capture["training_inputs"]):
        raise ValueError("training timestamps and inputs have different lengths")
    if not bool(capture["obstacle_removed"]):
        raise ValueError("capture metadata says the obstacle was not removed")
    if float(np.max(np.abs(capture["training_inputs"][:, 6]))) > 1e-9:
        raise ValueError("training force is not zero within tolerance 1e-9")

    expected_dt = float(capture["simulation_timestep"])
    maximum_dt_error = max(
        float(np.max(np.abs(warmup_dt - expected_dt))),
        float(np.max(np.abs(training_dt - expected_dt))),
    )
    if maximum_dt_error > 1e-9:
        raise ValueError(
            f"capture timing differs from timestep by {maximum_dt_error:.3g} s"
        )
    if abs(float(capture["esn_fs_hz"]) - 60.0) > 1e-9:
        raise ValueError("capture does not use the required 60 Hz ESN rate")
    return path, capture


def smooth_dense_capture(capture, window, order):
    warmup_time = np.asarray(capture["warmup_time"], dtype=float)
    training_time = np.asarray(capture["training_time"], dtype=float)
    warmup_inputs = capture["warmup_inputs"]
    training_inputs = capture["training_inputs"]
    dense_time = np.concatenate([warmup_time, training_time])
    dense_inputs = np.vstack([warmup_inputs, training_inputs])

    if training_time[0] <= warmup_time[-1]:
        raise ValueError("training segment must start after the warmup segment")
    dense_dt = float(capture["simulation_timestep"])
    actual_window = _odd_window(window, len(dense_time), order)
    qpos_smooth = _local_polynomial_filter(
        dense_inputs[:, :3], actual_window, order, dense_dt
    )
    qvel_smooth = _local_polynomial_filter(
        dense_inputs[:, 3:6], actual_window, order, dense_dt
    )
    qacc_smooth = _local_polynomial_filter(
        dense_inputs[:, 3:6],
        actual_window,
        order,
        dense_dt,
        derivative=1,
    )
    force = dense_inputs[:, 6]
    return {
        "time": dense_time,
        "qpos": qpos_smooth,
        "qvel": qvel_smooth,
        "qacc": qacc_smooth,
        "force": force,
        "training_start_time": float(training_time[0]),
        "window": actual_window,
        "order": int(order),
    }


def _uniform_times(start, stop, fs_hz, include_stop=True):
    duration = float(stop) - float(start)
    count = int(np.floor(duration * fs_hz + 1e-9))
    if include_stop:
        count += 1
    return float(start) + np.arange(count, dtype=float) / float(fs_hz)


def _interpolate_matrix(source_time, values, target_time):
    values = np.asarray(values, dtype=float)
    return np.column_stack(
        [np.interp(target_time, source_time, values[:, i]) for i in range(values.shape[1])]
    )


def resample_60hz(smoothed, fs_hz=60.0):
    dense_time = smoothed["time"]
    training_start = smoothed["training_start_time"]
    warmup_time = _uniform_times(dense_time[0], training_start, fs_hz, False)
    training_time = _uniform_times(training_start, dense_time[-1], fs_hz, True)

    def sample(times):
        qpos = _interpolate_matrix(dense_time, smoothed["qpos"], times)
        qvel = _interpolate_matrix(dense_time, smoothed["qvel"], times)
        qacc = _interpolate_matrix(dense_time, smoothed["qacc"], times)
        force = np.interp(times, dense_time, smoothed["force"])
        inputs = np.column_stack([qpos, qvel, force])
        return inputs, qacc

    warmup_inputs, warmup_qacc = sample(warmup_time)
    training_inputs, training_qacc = sample(training_time)
    return {
        "warmup_time": warmup_time,
        "warmup_inputs": warmup_inputs,
        "warmup_qacc": warmup_qacc,
        "training_time": training_time,
        "training_inputs": training_inputs,
        "training_qacc": training_qacc,
    }


def _joint_and_actuator_ids(model):
    qpos_ids = []
    dof_ids = []
    actuator_ids = []
    for name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"joint not found: {name}")
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
        matches = [
            actuator_id
            for actuator_id in range(model.nu)
            if int(model.actuator_trnid[actuator_id, 0]) == joint_id
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one actuator for {name}, got {matches}")
        actuator_ids.append(matches[0])

    for actuator_id in actuator_ids:
        gear = np.asarray(model.actuator_gear[actuator_id], dtype=float)
        if not np.isclose(gear[0], 1.0) or not np.allclose(gear[1:], 0.0):
            raise ValueError(f"actuator {actuator_id} does not have unit scalar gear")
        if int(model.actuator_dyntype[actuator_id]) != int(mujoco.mjtDyn.mjDYN_NONE):
            raise ValueError(f"actuator {actuator_id} has unsupported activation dynamics")
    return np.asarray(qpos_ids), np.asarray(dof_ids), np.asarray(actuator_ids)


def inverse_dynamics_targets(simulation, resampled):
    model = simulation.model
    simulation.platform.set_removed(True)
    qpos_ids, dof_ids, actuator_ids = _joint_and_actuator_ids(model)
    base_qpos = simulation.data.qpos.copy()
    base_mocap_pos = simulation.data.mocap_pos.copy()
    base_mocap_quat = simulation.data.mocap_quat.copy()
    inverse_data = mujoco.MjData(model)
    local_data = mujoco.MjData(model)
    inputs = resampled["training_inputs"]
    desired_qacc = resampled["training_qacc"]
    joint_ranges = np.asarray(
        [
            model.jnt_range[
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            ]
            for name in JOINT_NAMES
        ],
        dtype=float,
    )
    limit_violation = (
        (inputs[:, :3] < joint_ranges[:, 0])
        | (inputs[:, :3] > joint_ranges[:, 1])
    )
    targets = np.empty((len(inputs), 3), dtype=float)
    local_qacc = np.empty_like(targets)

    for index, (row, qacc3) in enumerate(zip(inputs, desired_qacc)):
        inverse_data.qpos[:] = base_qpos
        inverse_data.qvel[:] = 0.0
        inverse_data.qacc[:] = 0.0
        inverse_data.qpos[qpos_ids] = row[:3]
        inverse_data.qvel[dof_ids] = row[3:6]
        inverse_data.qacc[dof_ids] = qacc3
        inverse_data.mocap_pos[:] = base_mocap_pos
        inverse_data.mocap_quat[:] = base_mocap_quat
        mujoco.mj_inverse(model, inverse_data)
        targets[index] = inverse_data.qfrc_inverse[dof_ids]

        local_data.qpos[:] = inverse_data.qpos
        local_data.qvel[:] = inverse_data.qvel
        local_data.ctrl[:] = 0.0
        local_data.ctrl[actuator_ids] = targets[index]
        local_data.mocap_pos[:] = base_mocap_pos
        local_data.mocap_quat[:] = base_mocap_quat
        mujoco.mj_forward(model, local_data)
        local_qacc[index] = local_data.qacc[dof_ids]

    if not np.all(np.isfinite(targets)):
        raise FloatingPointError("MuJoCo inverse dynamics produced non-finite torque")
    local_error = local_qacc - desired_qacc
    inside_all_limits = ~np.any(limit_violation, axis=1)
    if not np.any(inside_all_limits):
        raise ValueError("no 60 Hz training samples are inside all joint limits")
    return {
        "target_torque": targets,
        "local_forward_qacc": local_qacc,
        "local_qacc_rmse": np.sqrt(np.mean(local_error ** 2, axis=0)),
        "local_qacc_mae": np.mean(np.abs(local_error), axis=0),
        "local_qacc_rmse_inside_limits": np.sqrt(
            np.mean(local_error[inside_all_limits] ** 2, axis=0)
        ),
        "joint_limit_violation_fraction": np.mean(limit_violation, axis=0),
        "qpos_ids": qpos_ids,
        "dof_ids": dof_ids,
        "actuator_ids": actuator_ids,
        "base_qpos": base_qpos,
    }


def forward_replay(simulation, resampled, inverse):
    model = simulation.model
    data = mujoco.MjData(model)
    qpos_ids = inverse["qpos_ids"]
    dof_ids = inverse["dof_ids"]
    actuator_ids = inverse["actuator_ids"]
    desired = resampled["training_inputs"][:, :3]
    targets = inverse["target_torque"]
    data.qpos[:] = inverse["base_qpos"]
    data.qpos[qpos_ids] = desired[0]
    data.qvel[:] = 0.0
    data.qvel[dof_ids] = resampled["training_inputs"][0, 3:6]
    data.mocap_pos[:] = simulation.data.mocap_pos
    data.mocap_quat[:] = simulation.data.mocap_quat
    mujoco.mj_forward(model, data)

    sample_times = resampled["training_time"]
    start_time = float(data.time)
    actual = np.empty_like(desired)
    sample_index = 0
    while sample_index < len(sample_times):
        relative_time = float(data.time) - start_time
        desired_time = float(sample_times[sample_index] - sample_times[0])
        if relative_time + 1e-12 >= desired_time:
            actual[sample_index] = data.qpos[qpos_ids]
            sample_index += 1
            if sample_index >= len(sample_times):
                break
        control_index = min(max(sample_index - 1, 0), len(targets) - 1)
        data.ctrl[:] = 0.0
        data.ctrl[actuator_ids] = targets[control_index]
        mujoco.mj_step(model, data)

    error = actual - desired
    return {
        "forward_qpos": actual,
        "forward_qpos_rmse": np.sqrt(np.mean(error ** 2, axis=0)),
        "forward_qpos_mae": np.mean(np.abs(error), axis=0),
        "forward_qpos_max_abs": np.max(np.abs(error), axis=0),
    }


def _range_stats(values):
    values = np.asarray(values, dtype=float)
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
    }


def build_report(input_path, capture, smoothed, resampled, inverse, replay):
    training_inputs = capture["training_inputs"]
    report = {
        "input": str(input_path),
        "steps_completed": [
            "validate_dense_capture",
            "smooth_and_estimate_acceleration",
            "resample_to_60hz",
            "mujoco_inverse_dynamics",
        ],
        "dense": {
            "warmup_samples": int(len(capture["warmup_inputs"])),
            "training_samples": int(len(training_inputs)),
            "sample_hz": 1.0 / float(capture["simulation_timestep"]),
            "training_input_stats": _range_stats(training_inputs),
            "force_max_abs": float(np.max(np.abs(training_inputs[:, 6]))),
        },
        "smoothing": {
            "method": "Savitzky-Golay",
            "window_samples": int(smoothed["window"]),
            "polynomial_order": int(smoothed["order"]),
            "qacc_stats": _range_stats(resampled["training_qacc"]),
        },
        "resampled": {
            "sample_hz": 60.0,
            "warmup_samples": int(len(resampled["warmup_inputs"])),
            "training_samples": int(len(resampled["training_inputs"])),
            "training_input_stats": _range_stats(resampled["training_inputs"]),
        },
        "inverse_dynamics": {
            "target_torque_stats": _range_stats(inverse["target_torque"]),
            "local_qacc_rmse": inverse["local_qacc_rmse"].tolist(),
            "local_qacc_mae": inverse["local_qacc_mae"].tolist(),
            "local_qacc_rmse_inside_limits": inverse[
                "local_qacc_rmse_inside_limits"
            ].tolist(),
            "joint_limit_violation_fraction": inverse[
                "joint_limit_violation_fraction"
            ].tolist(),
            "qpos_ids": inverse["qpos_ids"].tolist(),
            "dof_ids": inverse["dof_ids"].tolist(),
            "actuator_ids": inverse["actuator_ids"].tolist(),
            "actuator_gear": "unit scalar",
        },
    }
    if replay is not None:
        report["forward_replay"] = {
            "qpos_rmse": replay["forward_qpos_rmse"].tolist(),
            "qpos_mae": replay["forward_qpos_mae"].tolist(),
            "qpos_max_abs": replay["forward_qpos_max_abs"].tolist(),
        }
    return report


def save_plot(path, resampled, inverse, replay):
    time_values = resampled["training_time"] - resampled["training_time"][0]
    qpos = resampled["training_inputs"][:, :3]
    qvel = resampled["training_inputs"][:, 3:6]
    qacc = resampled["training_qacc"]
    torque = inverse["target_torque"]
    colors = ("tab:blue", "tab:orange", "tab:green")
    labels = JOINT_NAMES
    row_count = 5 if replay is not None else 4
    fig, axes = plt.subplots(row_count, 1, figsize=(13, 3.0 * row_count), sharex=True)
    for joint in range(3):
        axes[0].plot(time_values, qpos[:, joint], color=colors[joint], label=labels[joint])
        axes[1].plot(time_values, qvel[:, joint], color=colors[joint])
        axes[2].plot(time_values, qacc[:, joint], color=colors[joint])
        axes[3].plot(time_values, torque[:, joint], color=colors[joint])
    axes[0].set_ylabel("q (rad)")
    axes[1].set_ylabel("dq (rad/s)")
    axes[2].set_ylabel("ddq (rad/s^2)")
    axes[3].set_ylabel("inverse tau (N m)")
    axes[0].legend(loc="upper right")
    if replay is not None:
        for joint in range(3):
            axes[4].plot(time_values, qpos[:, joint], color=colors[joint], linestyle="--")
            axes[4].plot(time_values, replay["forward_qpos"][:, joint], color=colors[joint])
        axes[4].set_ylabel("q replay (rad)")
    axes[-1].set_xlabel("Training time (s)")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(args):
    input_path, capture = load_and_validate(args.input)
    smoothed = smooth_dense_capture(capture, args.savgol_window, args.savgol_order)
    resampled = resample_60hz(smoothed, fs_hz=60.0)
    simulation = build_esn_simulation(args)
    inverse = inverse_dynamics_targets(simulation, resampled)
    replay = forward_replay(simulation, resampled, inverse) if args.forward_replay else None

    output_path = args.output
    if output_path is None:
        output_path = input_path.with_name(input_path.stem + "_prepared_60hz.npz")
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report_path = output_path.with_name(output_path.stem + "_report.json")
    plot_path = output_path.with_name(output_path.stem + "_curves.png")
    report = build_report(input_path, capture, smoothed, resampled, inverse, replay)

    payload = {
        "warmup_time": resampled["warmup_time"],
        "warmup_inputs": resampled["warmup_inputs"],
        "training_time": resampled["training_time"],
        "training_inputs": resampled["training_inputs"],
        "training_qacc": resampled["training_qacc"],
        "target_torque": inverse["target_torque"],
        "feature_columns": np.asarray(EXPECTED_COLUMNS),
        "target_columns": np.asarray(
            [f"tau_inverse_{joint_name}" for joint_name in JOINT_NAMES]
        ),
        "fs_hz": np.asarray(60.0),
        "source_capture": np.asarray(str(input_path)),
        "savgol_window": np.asarray(smoothed["window"]),
        "savgol_order": np.asarray(smoothed["order"]),
        "qpos_ids": inverse["qpos_ids"],
        "dof_ids": inverse["dof_ids"],
        "actuator_ids": inverse["actuator_ids"],
    }
    if replay is not None:
        payload["forward_qpos"] = replay["forward_qpos"]
    np.savez_compressed(output_path, **payload)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_plot(plot_path, resampled, inverse, replay)

    print(f"Prepared data: {output_path}")
    print(f"Report:        {report_path}")
    print(f"Curves:        {plot_path}")
    print(f"60 Hz warmup samples:  {len(resampled['warmup_inputs'])}")
    print(f"60 Hz training samples: {len(resampled['training_inputs'])}")
    print("Torque min:", np.min(inverse["target_torque"], axis=0))
    print("Torque max:", np.max(inverse["target_torque"], axis=0))
    print("Local qacc RMSE:", inverse["local_qacc_rmse"])
    if replay is not None:
        print("Forward replay qpos RMSE:", replay["forward_qpos_rmse"])


def main():
    run(parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
