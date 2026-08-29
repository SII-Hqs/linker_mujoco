#!/usr/bin/env python3
"""Render recorded MuJoCo joint-state CSV files to consistently framed MP4s."""

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import mujoco
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOG_DIR = Path.cwd() / "results" / "joint_logs"
DEFAULT_OUTPUT_DIR = Path.cwd() / "results" / "joint_log_videos"
FINGER_JOINT0_NAMES = {
    "index_joint0",
    "middle_joint0",
    "ring_joint0",
    "little_joint0",
}


def infer_model_path(log_path):
    match = re.match(
        r"(?P<model>[a-z0-9]+)_(?P<side>left|right)_",
        log_path.name.lower(),
    )
    if match is None:
        raise ValueError(f"Cannot infer model and side from {log_path.name}")
    model = match.group("model")
    side = match.group("side")
    model_dir = model.upper() if model != "l20a" else "L20a"
    model_name = f"linker_hand_{model}_{side}"
    return (
        PACKAGE_ROOT
        / "linker_hand_mujoco_ros2"
        / "urdf"
        / model_dir
        / model_name
        / f"{model_name}.xml"
    )


def qpos_columns(fields, prefix):
    """Return qpos columns for exactly one trajectory representation."""
    columns = [field for field in fields if field.startswith(prefix)]
    if prefix == "qpos_":
        columns = [
            field
            for field in columns
            if not field.startswith("qpos_reconstructed_")
        ]
    return columns


def read_trajectory(
    log_path, qpos_prefix="qpos_", zero_thumb=False, zero_finger_joint0=False
):
    with log_path.open(newline="") as log_file:
        reader = csv.DictReader(log_file)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"No samples in {log_path}")

    qpos_fields = qpos_columns(fields, qpos_prefix)
    if not qpos_fields:
        raise ValueError(
            f"No columns with qpos prefix {qpos_prefix!r} in {log_path}"
        )
    time_field = "sim_time" if "sim_time" in fields else "wall_time"
    timestamps = np.asarray(
        [float(row[time_field]) for row in rows], dtype=float
    )
    positions = np.asarray(
        [[float(row[field]) for field in qpos_fields] for row in rows],
        dtype=float,
    )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValueError(
            f"Timestamps are not strictly increasing in {log_path}"
        )
    joint_names = [field.removeprefix(qpos_prefix) for field in qpos_fields]
    zero_names = set()
    if zero_thumb:
        zero_names.update(
            name for name in joint_names if name.startswith("thumb_joint")
        )
    if zero_finger_joint0:
        zero_names.update(FINGER_JOINT0_NAMES)
    for joint_index, joint_name in enumerate(joint_names):
        if joint_name in zero_names:
            positions[:, joint_index] = 0.0
    return timestamps, joint_names, positions


def joint_qpos_addresses(model, joint_names):
    addresses = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(
                f"Joint {name!r} is missing from the MuJoCo model"
            )
        addresses.append(int(model.jnt_qposadr[joint_id]))
    return np.asarray(addresses, dtype=int)


def configure_camera(camera, args):
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = args.lookat
    camera.distance = args.distance
    camera.azimuth = args.azimuth
    camera.elevation = args.elevation


def set_interpolated_pose(
    model, data, qpos_addresses, timestamps, positions, sample_time
):
    upper = int(np.searchsorted(timestamps, sample_time, side="right"))
    if upper <= 0:
        pose = positions[0]
    elif upper >= len(timestamps):
        pose = positions[-1]
    else:
        lower = upper - 1
        span = timestamps[upper] - timestamps[lower]
        weight = (sample_time - timestamps[lower]) / span
        pose = positions[lower] + weight * (
            positions[upper] - positions[lower]
        )
    data.qpos[qpos_addresses] = pose
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def render_log(log_path, output_path, args, preview_path=None):
    timestamps, joint_names, positions = read_trajectory(
        log_path,
        qpos_prefix=args.qpos_prefix,
        zero_thumb=args.zero_thumb,
        zero_finger_joint0=args.zero_finger_joint0,
    )
    model_path = (
        Path(args.model_xml) if args.model_xml else infer_model_path(log_path)
    )
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos_addresses = joint_qpos_addresses(model, joint_names)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    camera = mujoco.MjvCamera()
    configure_camera(camera, args)

    if preview_path is not None:
        sample_time = timestamps[len(timestamps) // 2]
        set_interpolated_pose(
            model, data, qpos_addresses, timestamps, positions, sample_time
        )
        renderer.update_scene(data, camera=camera)
        image = renderer.render()
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        if not cv2.imwrite(str(preview_path), bgr):
            raise RuntimeError(f"Failed to write {preview_path}")
        renderer.close()
        return 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not open the MP4 video writer")

    duration = max(float(timestamps[-1] - timestamps[0]), 0.0)
    frame_count = max(int(np.ceil(duration * args.fps)) + 1, 1)
    try:
        for frame_index in range(frame_count):
            sample_time = min(
                timestamps[0] + frame_index / args.fps,
                timestamps[-1],
            )
            set_interpolated_pose(
                model, data, qpos_addresses, timestamps, positions, sample_time
            )
            renderer.update_scene(data, camera=camera)
            rgb = renderer.render()
            writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()
        renderer.close()
    return frame_count


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-xml", default="")
    parser.add_argument(
        "--qpos-prefix",
        default="qpos_",
        help="CSV angle-column prefix (for example: qpos_reconstructed_)",
    )
    parser.add_argument(
        "--zero-thumb",
        action="store_true",
        help="Force every thumb joint angle to zero during replay",
    )
    parser.add_argument(
        "--zero-finger-joint0",
        action="store_true",
        help=(
            "Force index/middle/ring/little joint0 angles to zero "
            "during replay"
        ),
    )
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--lookat", type=float, nargs=3, default=(0.13, 0.035, 0.29)
    )
    parser.add_argument("--distance", type=float, default=0.30)
    parser.add_argument("--azimuth", type=float, default=125.0)
    parser.add_argument("--elevation", type=float, default=-25.0)
    parser.add_argument("--preview", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    logs = args.inputs or sorted(args.input_dir.glob("*.csv"))
    if not logs:
        raise SystemExit(f"No CSV logs found in {args.input_dir}")

    for index, log_path in enumerate(logs, start=1):
        suffix = ".png" if args.preview else ".mp4"
        output_path = args.output_dir / f"{log_path.stem}{suffix}"
        frame_count = render_log(
            log_path,
            output_path,
            args,
            preview_path=output_path if args.preview else None,
        )
        print(
            f"[{index}/{len(logs)}] {output_path} ({frame_count} frames)",
            flush=True,
        )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        raise
