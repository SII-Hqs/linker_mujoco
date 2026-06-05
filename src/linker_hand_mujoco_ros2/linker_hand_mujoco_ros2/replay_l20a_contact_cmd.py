#!/usr/bin/env python3
"""Replay L20a retargeting data as /cb_right_hand_l20a_contact_cmd."""

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

sys.modules.setdefault("numpy._core", np.core)
sys.modules.setdefault("numpy._core.multiarray", np.core.multiarray)
sys.modules.setdefault("numpy._core.numeric", np.core.numeric)


class ReplayL20aContactCmd(Node):
    def __init__(
        self,
        data_dir: Path,
        topic: str,
        rate: float | None,
        start_frame: int,
        end_frame: int | None,
        loop: bool,
        log_every: int,
        pressure_scale: float,
        pressure_offset: float,
    ):
        super().__init__("replay_l20a_contact_cmd")
        self.data_dir = data_dir
        self.topic = topic
        self.loop = loop
        self.log_every = max(log_every, 1)
        self.pressure_scale = pressure_scale
        self.pressure_offset = pressure_offset

        pkl_files = list(data_dir.glob("*_joints.pkl"))
        if not pkl_files:
            self.get_logger().fatal(f"No *_joints.pkl found in {data_dir}")
            raise SystemExit(1)
        pkl_path = sorted(pkl_files)[0]

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        self.joint_frames = [np.asarray(frame, dtype=np.float32) for frame in pkl_data["data"]]
        self.joint_names = pkl_data["meta_data"]["joint_names"]
        if len(self.joint_names) != 21:
            self.get_logger().fatal(f"Expected 21 joint names, got {len(self.joint_names)}")
            raise SystemExit(1)
        self.get_logger().info(
            f"Loaded {len(self.joint_frames)} joint frames from {pkl_path.name}"
        )

        frame_log_path = data_dir / "frame_log.csv"
        if not frame_log_path.exists():
            self.get_logger().fatal(f"frame_log.csv not found in {data_dir}")
            raise SystemExit(1)

        self.pressures = []
        with open(frame_log_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.pressures.append(float(row["matched_pressure"]))
        self.get_logger().info(
            f"Loaded {len(self.pressures)} matched_pressure values from frame_log.csv"
        )

        frame_count = min(len(self.joint_frames), len(self.pressures))
        start_frame = min(max(start_frame, 0), frame_count)
        if end_frame is None:
            end_frame = frame_count
        end_frame = min(max(end_frame, start_frame), frame_count)
        self.joint_frames = self.joint_frames[:frame_count]
        self.pressures = self.pressures[:frame_count]
        self.start_frame = start_frame
        self.end_frame = end_frame

        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            source_fps = float(metadata.get("fps", 30.0))
        else:
            source_fps = 30.0
        self.fps = float(rate) if rate is not None else source_fps

        self.get_logger().info(
            f"Replaying frames [{self.start_frame}, {self.end_frame}) at {self.fps} fps "
            f"to {self.topic}"
        )

        self.publisher = self.create_publisher(Float32MultiArray, self.topic, 10)

        self.frame_idx = self.start_frame
        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

    def timer_cb(self):
        if self.frame_idx >= self.end_frame:
            if self.loop:
                self.frame_idx = self.start_frame
            else:
                self.get_logger().info("Replay finished.")
                self.timer.cancel()
                rclpy.shutdown()
                return

        joints = self.joint_frames[self.frame_idx]
        pressure = self.pressures[self.frame_idx] * self.pressure_scale + self.pressure_offset

        msg = Float32MultiArray()
        msg.data = joints.astype(float).tolist() + [float(pressure)]
        self.publisher.publish(msg)

        if (self.frame_idx - self.start_frame) % self.log_every == 0:
            self.get_logger().info(
                f"Frame {self.frame_idx}/{self.end_frame - 1}, "
                f"pressure={pressure:.1f}g"
            )

        self.frame_idx += 1


def default_retargeting_root():
    return Path(__file__).resolve().parents[1] / "retargeting_data"


def latest_session(root: Path):
    if not root.exists():
        return None
    sessions = [path for path in root.iterdir() if path.is_dir() and (path / "frame_log.csv").exists()]
    if not sessions:
        return None
    return sorted(sessions, key=lambda path: path.stat().st_mtime)[-1]


def main(args=None):
    parser = argparse.ArgumentParser(description="Replay L20a retargeting data")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Path to retargeting data directory",
    )
    parser.add_argument("--topic", default="/cb_right_hand_l20a_contact_cmd")
    parser.add_argument("--rate", type=float, default=None, help="Replay FPS. Defaults to metadata fps.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--log-every", type=int, default=30)
    parser.add_argument("--pressure-scale", type=float, default=1.0)
    parser.add_argument("--pressure-offset", type=float, default=0.0)
    parser.add_argument(
        "--latest",
        action="store_true",
        help="Use the newest session under src/linker_hand_mujoco_ros2/retargeting_data.",
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])

    rclpy.init(args=remaining)

    if parsed.latest:
        data_dir = latest_session(default_retargeting_root())
        if data_dir is None:
            print(f"Error: no replay sessions found under {default_retargeting_root()}")
            rclpy.shutdown()
            return
    elif parsed.data_dir:
        data_dir = Path(parsed.data_dir).expanduser().resolve()
    else:
        tmp_node = rclpy.create_node("_tmp_param_reader")
        tmp_node.declare_parameter("data_dir", "")
        val = tmp_node.get_parameter("data_dir").value
        tmp_node.destroy_node()
        if val:
            data_dir = Path(val).expanduser().resolve()
        else:
            print("Error: data_dir not specified. Pass as argument or ROS2 param.")
            rclpy.shutdown()
            return

    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        rclpy.shutdown()
        return

    node = ReplayL20aContactCmd(
        data_dir=data_dir,
        topic=parsed.topic,
        rate=parsed.rate,
        start_frame=parsed.start_frame,
        end_frame=parsed.end_frame,
        loop=parsed.loop,
        log_every=parsed.log_every,
        pressure_scale=parsed.pressure_scale,
        pressure_offset=parsed.pressure_offset,
    )
    rclpy.spin(node)


if __name__ == "__main__":
    main()
