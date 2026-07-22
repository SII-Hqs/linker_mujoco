#!/usr/bin/env python3
"""Collect dense seven-dimensional ESN inputs with the obstacle removed."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
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


FEATURE_COLUMNS = (
    "qpos_index_joint1",
    "qpos_index_joint2",
    "qpos_index_joint3",
    "qvel_index_joint1",
    "qvel_index_joint2",
    "qvel_index_joint3",
    "force_g",
)


def _default_output_dir() -> Path:
    return Path.cwd() / "runs" / "esn_training_inputs"


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Hide the obstacle, warm up the original ESN, then record dense "
            "[q1,q2,q3,dq1,dq2,dq3,force] samples."
        )
    )
    add_esn_arguments(parser)
    parser.set_defaults(horizon=0)
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=10.0,
        help="Simulation seconds to run before the training segment starts.",
    )
    parser.add_argument(
        "--record-seconds",
        type=float,
        default=10.0,
        help="Simulation seconds to record after warmup.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help="Directory receiving NPZ, CSV, and JSON output files.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output basename. A timestamped name is used when omitted.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening the MuJoCo viewer.",
    )
    return parser.parse_args(argv)


def _validate_args(args):
    if args.warmup_seconds < 0.0:
        raise ValueError("--warmup-seconds must be non-negative")
    if args.record_seconds <= 0.0:
        raise ValueError("--record-seconds must be positive")
    if int(args.horizon) != 0:
        raise ValueError("data collection requires --horizon 0")


def _sample(simulation):
    contact_state = simulation.observer.read()
    qpos = np.asarray(
        simulation.data.qpos[simulation.index_dof_ids], dtype=float
    ).reshape(3)
    qvel = np.asarray(
        simulation.data.qvel[simulation.index_dof_ids], dtype=float
    ).reshape(3)
    features = np.concatenate(
        [qpos, qvel, [float(contact_state.normal_force)]]
    )
    return float(simulation.data.time), features


def _advance(simulation, viewer=None):
    step_start = time.perf_counter()
    simulation.step()
    if viewer is not None:
        viewer.sync()
    sleep_time = float(simulation.model.opt.timestep) - (
        time.perf_counter() - step_start
    )
    if viewer is not None and sleep_time > 0.0:
        time.sleep(sleep_time)


def _run_phase(simulation, duration, viewer, label):
    start_time = float(simulation.data.time)
    end_time = start_time + float(duration)
    times = []
    inputs = []
    next_log_time = start_time

    while simulation.data.time < end_time:
        if viewer is not None and not viewer.is_running():
            raise RuntimeError("MuJoCo viewer was closed before collection finished")
        sample_time, features = _sample(simulation)
        times.append(sample_time)
        inputs.append(features)
        _advance(simulation, viewer)

        if simulation.data.time >= next_log_time:
            elapsed = simulation.data.time - start_time
            print(
                f"{label}: {min(elapsed, duration):6.2f}/{duration:.2f} s",
                end="\r",
                flush=True,
            )
            next_log_time += 1.0

    print(f"{label}: {duration:.2f}/{duration:.2f} s")
    input_array = np.asarray(inputs, dtype=float).reshape(-1, len(FEATURE_COLUMNS))
    return np.asarray(times, dtype=float), input_array


def _write_csv(path, times, inputs):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("simulation_time", *FEATURE_COLUMNS))
        for sample_time, row in zip(times, inputs):
            writer.writerow((f"{sample_time:.12g}", *[f"{v:.12g}" for v in row]))


def _save_results(args, simulation, warmup_time, warmup_inputs, record_time, record_inputs):
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    basename = args.name or (
        "zero_force_closed_trajectory_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    npz_path = output_dir / f"{basename}.npz"
    warmup_csv = output_dir / f"{basename}_warmup.csv"
    training_csv = output_dir / f"{basename}_training.csv"
    metadata_path = output_dir / f"{basename}_metadata.json"

    controller = simulation.controller.current_controller
    np.savez_compressed(
        npz_path,
        warmup_time=warmup_time,
        warmup_inputs=warmup_inputs,
        training_time=record_time,
        training_inputs=record_inputs,
        feature_columns=np.asarray(FEATURE_COLUMNS),
        simulation_timestep=np.asarray(simulation.model.opt.timestep),
        esn_fs_hz=np.asarray(controller.fs_hz),
        obstacle_removed=np.asarray(True),
        recurrence_k=np.asarray(controller.recurrence_k),
        extreme_zero_n=np.asarray(controller.extreme_zero_n),
    )
    _write_csv(warmup_csv, warmup_time, warmup_inputs)
    _write_csv(training_csv, record_time, record_inputs)

    metadata = {
        "npz": str(npz_path),
        "warmup_csv": str(warmup_csv),
        "training_csv": str(training_csv),
        "feature_columns": list(FEATURE_COLUMNS),
        "warmup_samples": int(len(warmup_inputs)),
        "training_samples": int(len(record_inputs)),
        "simulation_timestep": float(simulation.model.opt.timestep),
        "physics_sample_hz": 1.0 / float(simulation.model.opt.timestep),
        "esn_fs_hz": float(controller.fs_hz),
        "warmup_seconds": float(args.warmup_seconds),
        "record_seconds": float(args.record_seconds),
        "obstacle_removed": True,
        "recurrence_k": float(controller.recurrence_k),
        "extreme_zero_n": int(controller.extreme_zero_n),
        "warmup_force_max_abs": (
            float(np.max(np.abs(warmup_inputs[:, 6])))
            if len(warmup_inputs)
            else 0.0
        ),
        "training_force_max_abs": float(np.max(np.abs(record_inputs[:, 6]))),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    return npz_path, warmup_csv, training_csv, metadata_path, metadata


def run(args):
    _validate_args(args)
    simulation = build_esn_simulation(args)
    simulation.platform.set_removed(True)
    simulation.controller.horizon = 0
    current_esn = simulation.controller.current_controller
    current_esn.set_recurrence_k(0.0)
    current_esn.set_extreme_zero_n(0)
    mujoco.mj_forward(simulation.model, simulation.data)

    print("Obstacle hidden; collisions with the platform are disabled.")
    print(f"MuJoCo timestep: {simulation.model.opt.timestep:.6f} s")
    print(f"Dense sampling: {1.0 / simulation.model.opt.timestep:.1f} Hz")
    print(f"ESN control rate: {current_esn.fs_hz:.1f} Hz")
    print("Saved features: q1, q2, q3, dq1, dq2, dq3, force")

    if args.headless:
        warmup_time, warmup_inputs = _run_phase(
            simulation, args.warmup_seconds, None, "Warmup"
        )
        record_time, record_inputs = _run_phase(
            simulation, args.record_seconds, None, "Recording"
        )
    else:
        with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
            warmup_time, warmup_inputs = _run_phase(
                simulation, args.warmup_seconds, viewer, "Warmup"
            )
            record_time, record_inputs = _run_phase(
                simulation, args.record_seconds, viewer, "Recording"
            )

    arrays = (warmup_inputs, record_inputs)
    if any(a.ndim != 2 or a.shape[1] != 7 for a in arrays):
        raise RuntimeError("collector produced an invalid feature matrix")
    if any(not np.all(np.isfinite(a)) for a in arrays):
        raise FloatingPointError("collector produced NaN or infinite values")

    paths = _save_results(
        args,
        simulation,
        warmup_time,
        warmup_inputs,
        record_time,
        record_inputs,
    )
    npz_path, warmup_csv, training_csv, metadata_path, metadata = paths
    print("\nCollection complete.")
    print(f"NPZ:          {npz_path}")
    print(f"Warmup CSV:   {warmup_csv}")
    print(f"Training CSV: {training_csv}")
    print(f"Metadata:     {metadata_path}")
    print(f"Training samples: {metadata['training_samples']}")
    print(
        "Maximum absolute training force: "
        f"{metadata['training_force_max_abs']:.9g}"
    )
    if metadata["training_force_max_abs"] > 1e-9:
        print("WARNING: force was not exactly zero; inspect the saved force column.")


def main():
    run(parse_args(sys.argv[1:]))


if __name__ == "__main__":
    main()
