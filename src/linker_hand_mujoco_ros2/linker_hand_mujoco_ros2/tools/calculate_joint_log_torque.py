#!/usr/bin/env python3
import argparse
import csv
import math
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

try:
    from linker_hand_mujoco_ros2.io.package_paths import resource_path
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from linker_hand_mujoco_ros2.io.package_paths import resource_path

from linker_hand_mujoco_ros2.io.joint_log_processing import (
    differentiate_positions,
)


def default_joint_log_root():
    return Path.cwd() / "results" / "joint_logs"


def default_output_root():
    return Path.cwd() / "results" / "joint_torques"


def infer_hand_from_log_name(log_path):
    match = re.match(
        r"(?P<hand_joint>[a-z0-9]+)_(?P<hand_type>left|right)_",
        Path(log_path).name.lower(),
    )
    if match is None:
        return None, None
    return match.group("hand_joint"), match.group("hand_type")


def resolve_model_xml(hand_joint, hand_type, model_xml=""):
    if model_xml:
        return Path(model_xml).expanduser().resolve()

    if not hand_joint or not hand_type:
        raise ValueError(
            "Could not infer hand_joint/hand_type from the CSV name. "
            "Pass --hand-joint/--hand-type or --model-xml."
        )

    hand_joint_upper = hand_joint.upper()
    hand_type = hand_type.lower()
    if hand_joint_upper == "L20A":
        model_name = f"linker_hand_l20a_{hand_type}"
        return resource_path("urdf", "L20a", model_name, f"{model_name}.xml")

    model_name = f"linker_hand_{hand_joint.lower()}_{hand_type}"
    return resource_path(
        "urdf", hand_joint_upper, model_name, f"{model_name}.xml"
    )


def read_joint_log(csv_path):
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    if not rows:
        raise ValueError(f"No samples found in {csv_path}")

    qpos_cols = [name for name in fieldnames if name.startswith("qpos_")]
    if not qpos_cols:
        raise ValueError(f"No qpos_* columns found in {csv_path}")

    joint_names = [name[len("qpos_"):] for name in qpos_cols]
    time_col = "sim_time" if "sim_time" in fieldnames else "wall_time"
    if time_col not in fieldnames:
        raise ValueError(
            f"Expected sim_time or wall_time column in {csv_path}"
        )

    timestamps = np.asarray(
        [float(row[time_col]) for row in rows], dtype=float
    )
    qpos = np.asarray(
        [[float(row[col]) for col in qpos_cols] for row in rows],
        dtype=float,
    )
    qvel_cols = [f"qvel_{name}" for name in joint_names]
    qacc_cols = [f"qacc_{name}" for name in joint_names]
    qvel = None
    qacc = None
    if all(column in fieldnames for column in qvel_cols):
        qvel = np.asarray(
            [[float(row[column]) for column in qvel_cols] for row in rows],
            dtype=float,
        )
    if all(column in fieldnames for column in qacc_cols):
        qacc = np.asarray(
            [[float(row[column]) for column in qacc_cols] for row in rows],
            dtype=float,
        )
    return rows, fieldnames, joint_names, timestamps, qpos, qvel, qacc


def differentiate_qvel(qvel, timestamps):
    if len(qvel) < 2:
        return np.zeros_like(qvel)
    if (
        not np.all(np.isfinite(timestamps))
        or np.any(np.diff(timestamps) <= 0.0)
    ):
        raise ValueError("Timestamps must be finite and strictly increasing")
    edge_order = 2 if len(qvel) > 2 else 1
    return np.gradient(qvel, timestamps, axis=0, edge_order=edge_order)


def resolve_joint_addresses(model, joint_names):
    qpos_ids = []
    dof_ids = []
    missing = []
    for joint_name in joint_names:
        joint_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
        )
        if joint_id < 0:
            missing.append(joint_name)
            continue
        qpos_ids.append(int(model.jnt_qposadr[joint_id]))
        dof_ids.append(int(model.jnt_dofadr[joint_id]))
    if missing:
        raise ValueError(f"Joints not found in model: {missing}")
    return np.asarray(qpos_ids, dtype=int), np.asarray(dof_ids, dtype=int)


def compute_inverse_torque(
    model_xml, joint_names, qpos, qvel, qacc, damping=0.8
):
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    model.dof_damping[:] = float(damping)
    model.opt.disableflags = 1
    data = mujoco.MjData(model)
    qpos_ids, dof_ids = resolve_joint_addresses(model, joint_names)

    tau = np.zeros((len(qpos), len(joint_names)), dtype=float)
    for i in range(len(qpos)):
        data.qpos[:] = model.qpos0
        data.qvel[:] = 0.0
        data.qacc[:] = 0.0
        data.qfrc_applied[:] = 0.0
        data.xfrc_applied[:] = 0.0
        data.ctrl[:] = 0.0

        data.qpos[qpos_ids] = qpos[i]
        data.qvel[dof_ids] = qvel[i]
        data.qacc[dof_ids] = qacc[i]

        mujoco.mj_inverse(model, data)
        tau[i] = data.qfrc_inverse[dof_ids]

    return tau


def output_path_for(
    input_csv, output, source_root=None, preserve_source_dir=False
):
    input_csv = Path(input_csv)
    if output is None:
        return input_csv.with_name(f"{input_csv.stem}_inverse_torque.csv")

    output = Path(output).expanduser()
    if output.suffix.lower() == ".csv":
        return output
    if preserve_source_dir and source_root is not None:
        output = output / Path(source_root).name
    return output / f"{input_csv.stem}_inverse_torque.csv"


def write_output(output_csv, rows, fieldnames, joint_names, qvel, qacc, tau):
    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    extra_fields = (
        [f"qvel_{name}" for name in joint_names]
        + [f"qacc_{name}" for name in joint_names]
        + [f"tau_inverse_{name}" for name in joint_names]
    )
    out_fields = list(fieldnames)
    for field in extra_fields:
        if field not in out_fields:
            out_fields.append(field)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            out_row = dict(row)
            for j, joint_name in enumerate(joint_names):
                out_row[f"qvel_{joint_name}"] = float(qvel[i, j])
                out_row[f"qacc_{joint_name}"] = float(qacc[i, j])
                out_row[f"tau_inverse_{joint_name}"] = float(tau[i, j])
            writer.writerow(out_row)
    return output_csv


def plot_qpos_torque(
    output_path, timestamps, joint_names, qpos, tau, source_name=""
):
    """Plot joint angle and inverse torque together for every joint."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    column_count = 5
    row_count = math.ceil(len(joint_names) / column_count)
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(21, max(3.2 * row_count, 4.0)),
        sharex=True,
    )
    axes = np.asarray(axes).reshape(-1)
    relative_time = timestamps - timestamps[0]
    angle_handle = None
    torque_handle = None

    for index, joint_name in enumerate(joint_names):
        angle_axis = axes[index]
        torque_axis = angle_axis.twinx()
        angle_handle = angle_axis.plot(
            relative_time,
            qpos[:, index],
            color="tab:blue",
            linewidth=1.0,
            label="angle",
        )[0]
        torque_handle = torque_axis.plot(
            relative_time,
            tau[:, index],
            color="tab:red",
            linewidth=1.0,
            alpha=0.85,
            label="inverse torque",
        )[0]
        angle_axis.set_title(joint_name, fontsize=9)
        angle_axis.set_xlabel("Time (s)")
        angle_axis.set_ylabel("Angle (rad)", color="tab:blue", fontsize=8)
        torque_axis.set_ylabel(
            "Torque (N*m)", color="tab:red", fontsize=8
        )
        angle_axis.tick_params(axis="y", labelcolor="tab:blue", labelsize=7)
        torque_axis.tick_params(axis="y", labelcolor="tab:red", labelsize=7)
        angle_axis.grid(True, alpha=0.25)

    for axis in axes[len(joint_names):]:
        axis.set_visible(False)
    title = "Joint angles and inverse-dynamics torques"
    if source_name:
        title += f": {source_name}"
    figure.suptitle(title, fontsize=12)
    figure.legend(
        [angle_handle, torque_handle],
        ["angle", "inverse torque"],
        loc="upper center",
        ncol=2,
        bbox_to_anchor=(0.5, 0.975),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return output_path


def calculate_joint_log_torque(
    input_csv,
    output_csv,
    model_xml="",
    hand_joint="",
    hand_type="",
    damping=0.8,
    use_input_derivatives=False,
    plot_path=None,
):
    input_csv = Path(input_csv).expanduser().resolve()
    inferred_joint, inferred_type = infer_hand_from_log_name(input_csv)
    hand_joint = hand_joint or inferred_joint
    hand_type = hand_type or inferred_type
    model_xml = resolve_model_xml(hand_joint, hand_type, model_xml)

    (
        rows,
        fieldnames,
        joint_names,
        timestamps,
        qpos,
        input_qvel,
        input_qacc,
    ) = read_joint_log(input_csv)
    if use_input_derivatives:
        if input_qvel is None:
            raise ValueError(
                "--use-input-derivatives requires all matching qvel_* columns"
            )
        qvel = input_qvel
        qacc = (
            input_qacc
            if input_qacc is not None
            else differentiate_qvel(qvel, timestamps)
        )
    else:
        qvel, qacc = differentiate_positions(qpos, timestamps)
    tau = compute_inverse_torque(
        model_xml, joint_names, qpos, qvel, qacc, damping
    )
    output_csv = write_output(
        output_csv, rows, fieldnames, joint_names, qvel, qacc, tau
    )
    if plot_path is not None:
        plot_qpos_torque(
            plot_path,
            timestamps,
            joint_names,
            qpos,
            tau,
            source_name=input_csv.name,
        )
    return output_csv, len(rows), len(joint_names), model_xml


def collect_input_csvs(input_path):
    input_path = Path(input_path).expanduser().resolve()
    if input_path.is_file():
        return [(input_path, None)]
    if input_path.is_dir():
        csvs = sorted(
            path for path in input_path.glob("*.csv")
            if not path.name.endswith("_inverse_torque.csv")
        )
        if csvs:
            return [(path, input_path) for path in csvs]
    raise FileNotFoundError(f"No input CSV found: {input_path}")


def collect_all_input_csvs(input_paths):
    csvs = []
    for input_path in input_paths:
        csvs.extend(collect_input_csvs(input_path))
    return csvs


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Calculate MuJoCo inverse-dynamics torques from joint log CSV "
            "files."
        )
    )
    parser.add_argument(
        "input",
        nargs="*",
        help=(
            "Joint-log CSV files or directories. "
            "Defaults to results/joint_logs when omitted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help=(
            "Output CSV path for one input file, or output directory for "
            "multiple files. Defaults to <input>_inverse_torque.csv for a "
            "file and results/joint_torques for a directory."
        ),
    )
    parser.add_argument(
        "--model-xml", default="", help="MuJoCo XML model path."
    )
    parser.add_argument(
        "--hand-joint", default="", help="Model name, e.g. L20 or L20a."
    )
    parser.add_argument(
        "--hand-type", default="", help="Hand side: left or right."
    )
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument(
        "--use-input-derivatives",
        action="store_true",
        help="Use input qvel/qacc instead of regenerating them from qpos.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save one qpos/inverse-torque overview PNG per input CSV.",
    )
    parser.add_argument(
        "--plot-dir",
        default=None,
        help="Plot directory (default: a plots directory beside output CSVs).",
    )
    return parser.parse_args(sys.argv[1:] if argv is None else argv)


def main(args=None):
    parsed = parse_args(args)
    input_paths = parsed.input or [str(default_joint_log_root())]
    input_csvs = collect_all_input_csvs(input_paths)

    output_arg = parsed.output
    if output_arg is None and len(input_csvs) > 1:
        output_arg = default_output_root()
    if (
        output_arg is not None
        and Path(output_arg).expanduser().suffix.lower() == ".csv"
        and len(input_csvs) > 1
    ):
        raise ValueError(
            "A single output CSV path can only be used with one input CSV"
        )

    preserve_source_dir = len(input_paths) > 1

    for input_csv, source_root in input_csvs:
        output_csv = output_path_for(
            input_csv,
            output_arg,
            source_root=source_root,
            preserve_source_dir=preserve_source_dir,
        )
        plot_path = None
        if parsed.plot:
            plot_dir = (
                Path(parsed.plot_dir).expanduser()
                if parsed.plot_dir
                else Path(output_csv).parent / "plots"
            )
            plot_stem = Path(output_csv).stem.removesuffix("_inverse_torque")
            plot_path = plot_dir / f"{plot_stem}_qpos_torque.png"
        result = calculate_joint_log_torque(
            input_csv=input_csv,
            output_csv=output_csv,
            model_xml=parsed.model_xml,
            hand_joint=parsed.hand_joint,
            hand_type=parsed.hand_type,
            damping=parsed.damping,
            use_input_derivatives=parsed.use_input_derivatives,
            plot_path=plot_path,
        )
        result_path, sample_count, joint_count, model_xml = result
        print(
            f"Wrote {sample_count} samples, {joint_count} joints: "
            f"{result_path} "
            f"(model={model_xml})"
            + (f"; plot={plot_path}" if plot_path is not None else "")
        )


if __name__ == "__main__":
    main()
