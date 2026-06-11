#!/usr/bin/env python3
"""Replay retargeting data as Float32MultiArray on /cb_right_hand_l20a_contact_cmd.

Usage:
    ros2 run linker_hand_mujoco_ros2 replay_l20a_contact_cmd --ros-args \
        -p data_dir:=/path/to/retargeting_data/2026051902

Or run directly:
    python3 replay_l20a_contact_cmd.py ./retargeting_data/2026051902
"""

import argparse
import csv
import pickle
import sys
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class ReplayL20aContactCmd(Node):
    def __init__(self, data_dir: Path):
        super().__init__("replay_l20a_contact_cmd")
        self.data_dir = data_dir

        # Load joint data from pkl
        pkl_files = list(data_dir.glob("*_joints.pkl"))
        if not pkl_files:
            self.get_logger().fatal(f"No *_joints.pkl found in {data_dir}")
            raise SystemExit(1)
        pkl_path = pkl_files[0]

        with open(pkl_path, "rb") as f:
            pkl_data = pickle.load(f)
        self.joint_frames = [np.asarray(frame, dtype=np.float32) for frame in pkl_data["data"]]
        self.joint_names = pkl_data["meta_data"]["joint_names"]
        self.get_logger().info(
            f"Loaded {len(self.joint_frames)} joint frames from {pkl_path.name}"
        )

        # Load pressure from frame_log.csv
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
            f"Loaded {len(self.pressures)} pressure values from frame_log.csv"
        )

        # Ensure frame counts match
        frame_count = min(len(self.joint_frames), len(self.pressures))
        self.joint_frames = self.joint_frames[:frame_count]
        self.pressures = self.pressures[:frame_count]

        # Load fps from metadata.json
        import json
        metadata_path = data_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            self.fps = float(metadata.get("fps", 30.0))
        else:
            self.fps = 30.0

        self.get_logger().info(
            f"Replaying {frame_count} frames at {self.fps} fps"
        )

        # Publisher
        self.publisher = self.create_publisher(
            Float32MultiArray, "/cb_right_hand_l20a_contact_cmd", 10
        )

        # Timer
        self.frame_idx = 0
        self.timer = self.create_timer(1.0 / self.fps, self.timer_cb)

    def timer_cb(self):
        if self.frame_idx >= len(self.joint_frames):
            self.get_logger().info("Replay finished.")
            self.timer.cancel()
            rclpy.shutdown()
            return

        joints = self.joint_frames[self.frame_idx]
        pressure = self.pressures[self.frame_idx]

        msg = Float32MultiArray()
        msg.data = joints.tolist() + [pressure]
        self.publisher.publish(msg)

        if self.frame_idx % 30 == 0:
            self.get_logger().info(
                f"Frame {self.frame_idx}/{len(self.joint_frames)}, "
                f"pressure={pressure:.1f}g"
            )

        self.frame_idx += 1


def main(args=None):
    # Support both ROS2 param and CLI argument for data_dir
    parser = argparse.ArgumentParser(description="Replay L20a retargeting data")
    parser.add_argument(
        "data_dir",
        nargs="?",
        default=None,
        help="Path to retargeting data directory",
    )
    parsed, remaining = parser.parse_known_args(sys.argv[1:])

    rclpy.init(args=remaining)

    if parsed.data_dir:
        data_dir = Path(parsed.data_dir).expanduser().resolve()
    else:
        # Fallback: try ROS2 parameter
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

    node = ReplayL20aContactCmd(data_dir)
    rclpy.spin(node)


if __name__ == "__main__":
    main()
