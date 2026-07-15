import os
import tempfile
import threading
import time
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from linker_hand_mujoco_ros2.io.package_paths import resource_path
from linker_hand_mujoco_ros2.utils.mapping import (
    L10_JOINT_ARC,
    L10_JOINT_MAP,
    L20_JOINT_ARC,
    L20_JOINT_MAP,
    L21_JOINT_ARC,
    L21_JOINT_MAP,
    L6_JOINT_ARC,
    L6_JOINT_MAP,
    L7_JOINT_ARC,
    L7_JOINT_MAP,
    range_to_arc_left,
    range_to_arc_right,
)


JOINT_CONFIG = {
    "L6": {"map": L6_JOINT_MAP, "arc": L6_JOINT_ARC},
    "L7": {"map": L7_JOINT_MAP, "arc": L7_JOINT_ARC},
    "L10": {"map": L10_JOINT_MAP, "arc": L10_JOINT_ARC},
    "L20": {"map": L20_JOINT_MAP, "arc": L20_JOINT_ARC},
    "L21": {"map": L21_JOINT_MAP, "arc": L21_JOINT_ARC},
}

REFERENCE_ATTRS = {
    "mesh": "mesh",
    "material": "material",
    "texture": "texture",
    "joint": "joint",
    "body": "body",
    "geom": "geom",
    "site": "site",
    "tendon": "tendon",
}


def parse_vector(value, expected_len, default):
    if value is None:
        return default
    if isinstance(value, str):
        values = value.replace(",", " ").split()
    else:
        values = list(value)
    if not values:
        return default
    if len(values) != expected_len:
        raise ValueError(f"Expected {expected_len} values, got {len(values)}: {value}")
    return " ".join(str(float(item)) for item in values)


def hand_xml_path(hand_joint, hand_type):
    hand_joint = hand_joint.upper()
    hand_type = hand_type.lower()
    model_name = f"linker_hand_{hand_joint.lower()}_{hand_type}"
    return resource_path("urdf", hand_joint, model_name, f"{model_name}.xml")


def prefix_mjcf_tree(root, prefix, source_dir):
    name_maps = {}
    for elem in root.iter():
        name = elem.get("name")
        if name:
            name_maps.setdefault(elem.tag, {})[name] = f"{prefix}_{name}"

    for elem in root.iter():
        name = elem.get("name")
        if name:
            elem.set("name", name_maps[elem.tag][name])

        if elem.tag == "mesh" and elem.get("file"):
            mesh_file = Path(elem.get("file"))
            if not mesh_file.is_absolute():
                elem.set("file", str((source_dir / mesh_file).resolve()))

        for attr, tag in REFERENCE_ATTRS.items():
            old_value = elem.get(attr)
            if old_value in name_maps.get(tag, {}):
                elem.set(attr, name_maps[tag][old_value])


def is_world_fixture(elem):
    if elem.tag == "light":
        return True
    if elem.tag == "geom" and elem.get("type") == "plane":
        return True
    return False


def append_children(dst, src):
    if src is None:
        return
    for child in src:
        dst.append(deepcopy(child))


def append_hand_world(dst_worldbody, source_worldbody, wrapper_name, pos, quat):
    wrapper = ET.SubElement(dst_worldbody, "body", {"name": wrapper_name, "pos": pos, "quat": quat})
    for child in source_worldbody:
        if is_world_fixture(child):
            continue
        wrapper.append(deepcopy(child))


def make_dual_hand_xml(left_joint, right_joint, left_pos, right_pos, left_quat, right_quat):
    left_path = hand_xml_path(left_joint, "left")
    right_path = hand_xml_path(right_joint, "right")

    left_tree = ET.parse(left_path)
    right_tree = ET.parse(right_path)
    left_root = left_tree.getroot()
    right_root = right_tree.getroot()

    prefix_mjcf_tree(left_root, "left", left_path.parent)
    prefix_mjcf_tree(right_root, "right", right_path.parent)

    root = ET.Element("mujoco", {"model": "linker_hand_dual"})
    ET.SubElement(root, "compiler", {"angle": "radian"})
    ET.SubElement(root, "size", {"njmax": "1200", "nconmax": "300"})

    asset = ET.SubElement(root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "name": "dual_grid_tex",
            "type": "2d",
            "builtin": "checker",
            "rgb1": "0.2 0.3 0.8",
            "rgb2": "1 1 1",
            "width": "512",
            "height": "512",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "dual_grid_material",
            "texture": "dual_grid_tex",
            "texrepeat": "10 10",
            "specular": "0.3",
            "shininess": "0.5",
        },
    )
    append_children(asset, left_root.find("asset"))
    append_children(asset, right_root.find("asset"))

    visual = left_root.find("visual")
    if visual is not None:
        root.append(deepcopy(visual))

    default = left_root.find("default")
    if default is not None:
        root.append(deepcopy(default))

    worldbody = ET.SubElement(root, "worldbody")
    ET.SubElement(
        worldbody,
        "geom",
        {
            "name": "floor",
            "type": "plane",
            "pos": "0 0 -0.01",
            "size": "5 5 0.1",
            "material": "dual_grid_material",
            "conaffinity": "1",
            "condim": "3",
        },
    )
    ET.SubElement(
        worldbody,
        "light",
        {
            "directional": "true",
            "diffuse": ".4 .4 .4",
            "specular": ".1 .1 .1",
            "pos": "0 0 5",
            "dir": "0 0 -1",
        },
    )

    append_hand_world(worldbody, left_root.find("worldbody"), "left_dual_root", left_pos, left_quat)
    append_hand_world(worldbody, right_root.find("worldbody"), "right_dual_root", right_pos, right_quat)

    actuator = ET.SubElement(root, "actuator")
    append_children(actuator, left_root.find("actuator"))
    append_children(actuator, right_root.find("actuator"))

    ET.indent(root, space="    ")
    output_path = Path(tempfile.gettempdir()) / (
        f"linker_hand_dual_{left_joint.lower()}_{right_joint.lower()}_{os.getpid()}.xml"
    )
    ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


class HandController:
    def __init__(self, side, hand_joint, model):
        self.side = side
        self.hand_joint = hand_joint.upper()
        joint_config = JOINT_CONFIG.get(self.hand_joint)
        if joint_config is None:
            raise ValueError(f"Unsupported hand_joint for {side}: {hand_joint}")

        self.joint_map = joint_config["map"]
        self.actuator_ids = self.find_actuator_ids(model, side)
        self.last_contact_cmd_log_time = 0.0

    def find_actuator_ids(self, model, side):
        actuator_ids = []
        prefix = f"{side}_"
        for actuator_id in range(model.nu):
            name = model.actuator(actuator_id).name
            if name.startswith(prefix):
                actuator_ids.append(actuator_id)
        if not actuator_ids:
            raise ValueError(f"No actuators found for {side} hand")
        return actuator_ids

    def control_position_to_actuator(self, position):
        if self.side == "left":
            hand_arc = range_to_arc_left(position, self.hand_joint)
        else:
            hand_arc = range_to_arc_right(position, self.hand_joint)
        return self.map_position_array(hand_arc)

    def map_position_array(self, position):
        mapped_array = [0.0] * len(self.joint_map)
        for target_idx, source_idx in self.joint_map.items():
            if source_idx < len(position):
                mapped_array[target_idx] = position[source_idx]
        return mapped_array

    def apply_command(self, position, ctrl_values):
        command = self.control_position_to_actuator(position)
        count = min(len(command), len(self.actuator_ids))
        for local_idx in range(count):
            ctrl_values[self.actuator_ids[local_idx]] = command[local_idx]


class MujocoDualNode(Node):
    def __init__(self):
        super().__init__("linker_hand_mujoco_ros2_dual_node")
        self.declare_parameter("hand_joint", "L20")
        self.declare_parameter("left_hand_joint", "")
        self.declare_parameter("right_hand_joint", "")
        self.declare_parameter("left_root_pos", "-0.18 0 0")
        self.declare_parameter("right_root_pos", "0.18 0 0")
        self.declare_parameter("left_root_quat", "1 0 0 0")
        self.declare_parameter("right_root_quat", "1 0 0 0")
        self.declare_parameter("left_initial_position", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("right_initial_position", [], ParameterDescriptor(dynamic_typing=True))

        hand_joint = self.get_parameter("hand_joint").value.upper()
        left_joint = self.get_parameter("left_hand_joint").value or hand_joint
        right_joint = self.get_parameter("right_hand_joint").value or hand_joint
        left_joint = left_joint.upper()
        right_joint = right_joint.upper()

        left_pos = parse_vector(self.get_parameter("left_root_pos").value, 3, "-0.18 0 0")
        right_pos = parse_vector(self.get_parameter("right_root_pos").value, 3, "0.18 0 0")
        left_quat = parse_vector(self.get_parameter("left_root_quat").value, 4, "1 0 0 0")
        right_quat = parse_vector(self.get_parameter("right_root_quat").value, 4, "1 0 0 0")

        xml_path = make_dual_hand_xml(left_joint, right_joint, left_pos, right_pos, left_quat, right_quat)
        self.get_logger().info(f"Loading dual hand MuJoCo model: {xml_path}")

        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.model.dof_damping[:] = 0.8
        self.data = mujoco.MjData(self.model)
        self.ctrl_values = np.zeros(self.model.nu, dtype=float)
        self.model.opt.disableflags = 1

        self.left_hand = HandController("left", left_joint, self.model)
        self.right_hand = HandController("right", right_joint, self.model)
        self.apply_initial_position("left", self.left_hand)
        self.apply_initial_position("right", self.right_hand)
        mujoco.mj_forward(self.model, self.data)

        self.create_subscription(JointState, "/cb_left_hand_control_cmd", self.left_hand_cb, 10)
        self.create_subscription(JointState, "/cb_right_hand_control_cmd", self.right_hand_cb, 10)
        self.create_subscription(Float32MultiArray, "/cb_left_hand_contact_cmd", self.left_contact_cb, 10)
        self.create_subscription(Float32MultiArray, "/cb_right_hand_contact_cmd", self.right_contact_cb, 10)

        self.print_model_summary()
        sim_thread = threading.Thread(target=self.mujoco_thread, daemon=True)
        sim_thread.start()

    def print_model_summary(self):
        print("=" * 20, flush=True)
        print(mujoco.mj_versionString(), flush=True)
        print(f"Dual hand model: nq={self.model.nq}, nv={self.model.nv}, nu={self.model.nu}", flush=True)
        print("=" * 20, flush=True)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = self.model.joint(joint_id).name
            actuator_name = self.model.actuator(actuator_id).name
            print(f"Actuator {actuator_id}: {actuator_name} -> {joint_name}", flush=True)

    def apply_initial_position(self, side, hand_controller):
        position = self.get_parameter(f"{side}_initial_position").value
        if not position:
            return
        try:
            hand_controller.apply_command(position, self.ctrl_values)
            for actuator_id in hand_controller.actuator_ids:
                joint_id = int(self.model.actuator_trnid[actuator_id, 0])
                qpos_id = int(self.model.jnt_qposadr[joint_id])
                if qpos_id < len(self.data.qpos):
                    self.data.qpos[qpos_id] = self.ctrl_values[actuator_id]
        except Exception as exc:
            self.get_logger().error(f"Error applying {side}_initial_position: {exc}")

    def mujoco_thread(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("Dual hand MuJoCo viewer running...", flush=True)
            while viewer.is_running() and rclpy.ok():
                with viewer.lock():
                    self.data.ctrl[:] = self.ctrl_values
                    mujoco.mj_step(self.model, self.data)
                viewer.sync()
                time.sleep(0.001)

    def left_hand_cb(self, msg):
        self.handle_hand_cmd(self.left_hand, msg.position)

    def right_hand_cb(self, msg):
        self.handle_hand_cmd(self.right_hand, msg.position)

    def left_contact_cb(self, msg):
        self.handle_contact_cmd(self.left_hand, msg.data)

    def right_contact_cb(self, msg):
        self.handle_contact_cmd(self.right_hand, msg.data)

    def handle_hand_cmd(self, hand_controller, position):
        try:
            hand_controller.apply_command(position, self.ctrl_values)
        except Exception as exc:
            self.get_logger().error(f"Error in {hand_controller.side} hand command: {exc}")

    def handle_contact_cmd(self, hand_controller, data):
        values = list(data)
        if len(values) < 2:
            self.get_logger().error(
                f"{hand_controller.side}_hand_contact_cmd requires joint positions plus contact_force_g"
            )
            return
        try:
            hand_controller.apply_command(values[:-1], self.ctrl_values)
            now = time.time()
            if now - hand_controller.last_contact_cmd_log_time > 1.0:
                hand_controller.last_contact_cmd_log_time = now
                self.get_logger().info(
                    f"Received {hand_controller.side} contact cmd: "
                    f"joints={len(values) - 1}, contact_force_g={float(values[-1])}"
                )
        except Exception as exc:
            self.get_logger().error(f"Error in {hand_controller.side} contact command: {exc}")


def main(args=None):
    rclpy.init(args=args)
    node = MujocoDualNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
