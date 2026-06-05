import threading
import time
from pathlib import Path

import mujoco.viewer
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from linker_hand_mujoco_ros2.contact.contact_patch import ContactPatchCalculator
from linker_hand_mujoco_ros2.contact.contact_visualizer import ContactVisualizer
from linker_hand_mujoco_ros2.finger_vec_loader import (
    load_finger_vec,
    resolve_finger_vec_path,
    resolve_model_xml,
)
from linker_hand_mujoco_ros2.mujoco_sim import MujocoSim


class LinkerHandL20aContactNode(Node):
    def __init__(self):
        super().__init__("linker_hand_l20a_contact_node")
        self.declare_parameter("hand_type", "right")
        self.declare_parameter("model_xml", "")
        self.declare_parameter("finger_vec_path", "")
        self.declare_parameter("initial_frame_index", 0)
        self.declare_parameter("contact_patch_link", "index_link3")
        self.declare_parameter("contact_patch_mesh", "index_link3")
        self.declare_parameter("contact_patch_radius", 0.004)
        self.declare_parameter("contact_patch_update_hz", 20.0)
        self.declare_parameter("contact_patch_max_points", 120)
        self.declare_parameter("contact_force_sigma", 0.002)
        self.declare_parameter("contact_force_arrow_scale", 0.1)
        self.declare_parameter("clip_joint_command", True)

        self.hand_type = self.get_parameter("hand_type").value
        self.package_root = Path(__file__).resolve().parents[2]
        self.model_xml = self.resolve_model_xml()
        self.finger_vec_path = self.resolve_finger_vec_path()
        self.finger_joint_names, self.finger_frames = self.load_finger_vec(self.finger_vec_path)

        self.sim = MujocoSim(
            self.model_xml,
            self.finger_joint_names,
            self.finger_frames,
            self.get_parameter("clip_joint_command").value,
        )
        self.model = self.sim.model
        self.data = self.sim.data
        self.ctrl_values = self.sim.ctrl_values
        self.actuator_joint_names = self.sim.actuator_joint_names
        self.finger_joint_index = self.sim.finger_joint_index

        self.index_joint_names = ["index_joint0", "index_joint1", "index_joint2", "index_joint3"]
        self.index_dof_ids = self.get_joint_dof_ids(self.index_joint_names)
        self.contact_force_g = 0.0
        self.last_contact_cmd_log_time = 0.0

        self.setup_contact_patch_visualization()
        self.apply_initial_frame()
        self.sim.forward()

        topic_prefix = f"/cb_{self.hand_type}_hand_l20a"
        self.create_subscription(
            Float32MultiArray,
            f"{topic_prefix}_contact_cmd",
            self.hand_contact_cb,
            10,
        )
        self.index_torque_pub = self.create_publisher(
            JointState,
            f"{topic_prefix}_index_contact_torque",
            10,
        )

        sim_thread = threading.Thread(target=self.mujoco_thread, daemon=True)
        sim_thread.start()
        self.get_logger().info(f"L20a contact node loaded model: {self.model_xml}")
        self.get_logger().info(f"L20a command format loaded from: {self.finger_vec_path}")

    def resolve_model_xml(self):
        return resolve_model_xml(self.package_root, self.get_parameter("model_xml").value)

    def resolve_finger_vec_path(self):
        return resolve_finger_vec_path(
            self.package_root,
            self.get_parameter("finger_vec_path").value,
        )

    def load_finger_vec(self, path):
        return load_finger_vec(path)

    def get_actuator_joint_names(self):
        return self.sim.get_actuator_joint_names()

    def apply_initial_frame(self):
        self.sim.apply_initial_frame(self.get_parameter("initial_frame_index").value)

    def set_joint_command(self, command):
        self.sim.set_joint_command(command)

    def hand_contact_cb(self, data):
        values = np.asarray(list(data.data), dtype=float)
        expected_len = len(self.finger_joint_names) + 1
        if len(values) != expected_len:
            self.get_logger().error(
                f"Expected {expected_len} values: 21 joint radians + contact_force_g, got {len(values)}"
            )
            return

        try:
            self.set_joint_command(values[:-1])
            self.contact_force_g = float(values[-1])
            now = time.time()
            if now - self.last_contact_cmd_log_time > 1.0:
                self.last_contact_cmd_log_time = now
                self.get_logger().info(
                    f"Received L20a contact cmd: contact_force_g={self.contact_force_g}"
                )
        except Exception as exc:
            self.get_logger().error(f"Error in L20a contact command: {exc}")

    def setup_contact_patch_visualization(self):
        self.contact_patch = ContactPatchCalculator(
            self.model,
            self.data,
            self.get_parameter("contact_patch_link").value,
            self.get_parameter("contact_patch_mesh").value,
            self.get_parameter("contact_patch_radius").value,
            self.get_parameter("contact_patch_update_hz").value,
            self.get_parameter("contact_patch_max_points").value,
            self.get_parameter("contact_force_sigma").value,
            self.index_joint_names,
            self.index_dof_ids,
        )
        self.contact_visualizer = ContactVisualizer(
            self.get_parameter("contact_force_arrow_scale").value,
        )

    def mujoco_thread(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("MuJoCo L20a viewer running...", flush=True)
            while viewer.is_running():
                with viewer.lock():
                    self.sim.step()
                    self.update_contact_patch()
                    self.draw_contact_patch(viewer)
                viewer.sync()
                time.sleep(0.001)

    def update_contact_patch(self):
        if self.contact_patch.update(self.contact_force_g):
            self.publish_index_contact_torque()

    def distribute_contact_force(self, points, center):
        return self.contact_patch.distribute_contact_force(points, center, self.contact_force_g)

    def compute_index_contact_torque(self, points, forces):
        return self.contact_patch.compute_index_contact_torque(points, forces)

    def get_joint_dof_ids(self, joint_names):
        return self.sim.get_joint_dof_ids(joint_names)

    def publish_index_contact_torque(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.index_joint_names
        msg.effort = self.contact_patch.index_contact_torque.tolist()
        self.index_torque_pub.publish(msg)

    def draw_contact_patch(self, viewer):
        self.contact_visualizer.draw(viewer, self.contact_patch)


def main(args=None):
    rclpy.init(args=args)
    node = LinkerHandL20aContactNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
