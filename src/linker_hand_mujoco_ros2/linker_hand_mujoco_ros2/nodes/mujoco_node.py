
import rclpy,math,sys                                     # ROS2 Python接口库
from rclpy.node import Node                      # ROS2 节点类
from rcl_interfaces.msg import ParameterDescriptor
import rclpy.time
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import JointState
import csv
import time,threading, json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path


import sys,os
import threading
from sensor_msgs.msg import JointState
import numpy as np
import mujoco, time
import mujoco.viewer
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtCore import Qt, QTimer
from linker_hand_mujoco_ros2.io.package_paths import resource_path
from linker_hand_mujoco_ros2.utils.mapping import *

JOINT_CONFIG = {
    "L6": {
        "map": L6_JOINT_MAP,
        "arc": L6_JOINT_ARC
    },
    "L7": {
        "map": L7_JOINT_MAP,
        "arc": L7_JOINT_ARC
    },
    "L10": {
        "map": L10_JOINT_MAP,
        "arc": L10_JOINT_ARC
    },
    "L20": {
        "map": L20_JOINT_MAP,
        "arc": L20_JOINT_ARC
    },
    "L21": {
        "map": L21_JOINT_MAP,
        "arc": L21_JOINT_ARC,
    }
}

MIN_MUJOCO_MASS = 1e-6
MIN_MUJOCO_INERTIA = 1e-9


def _to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _temp_path(suffix, prefix):
    temp_dir = Path(tempfile.gettempdir()) / "linker_hand_mujoco_ros2"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix, dir=temp_dir)
    os.close(fd)
    return Path(path)


def _resolve_model_path(hand_joint, hand_type):
    hand_joint = hand_joint.upper()
    hand_type = hand_type.lower()
    if hand_joint == "O20":
        return resource_path("urdf", f"o20_{hand_type}", f"linkerhand_o20_{hand_type}.urdf")
    if hand_joint == "L20A":
        model_name = f"linker_hand_l20a_{hand_type}"
        return resource_path("urdf", "L20a", model_name, f"{model_name}.xml")

    model_name = f"linker_hand_{hand_joint.lower()}_{hand_type}"
    return resource_path("urdf", hand_joint, model_name, f"{model_name}.xml")


def _sanitize_urdf_for_mujoco(urdf_path):
    urdf_path = Path(urdf_path)
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.get("filename")
        if not filename or filename.startswith("package://") or Path(filename).is_absolute():
            continue
        mesh.set("filename", str((urdf_path.parent / filename).resolve()))

    for inertial in root.iter("inertial"):
        mass = inertial.find("mass")
        if mass is not None and _to_float(mass.get("value")) <= MIN_MUJOCO_MASS:
            mass.set("value", f"{MIN_MUJOCO_MASS:g}")

        inertia = inertial.find("inertia")
        if inertia is None:
            continue
        for attr in ("ixx", "iyy", "izz"):
            if _to_float(inertia.get(attr)) <= MIN_MUJOCO_INERTIA:
                inertia.set(attr, f"{MIN_MUJOCO_INERTIA:g}")

    output_path = _temp_path(".urdf", f"{urdf_path.stem}_mujoco_")
    ET.indent(root, space="  ")
    tree.write(output_path, encoding="utf-8", xml_declaration=True)
    return output_path


def _make_actuated_mjcf_from_urdf(urdf_path):
    sanitized_urdf_path = _sanitize_urdf_for_mujoco(urdf_path)
    model = mujoco.MjModel.from_xml_path(str(sanitized_urdf_path))
    mjcf_path = _temp_path(".xml", f"{Path(urdf_path).stem}_actuated_")
    mujoco.mj_saveLastXML(str(mjcf_path), model)

    tree = ET.parse(mjcf_path)
    root = tree.getroot()
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.SubElement(root, "actuator")

    hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
    slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) not in (hinge_type, slide_type):
            continue
        joint_name = model.joint(joint_id).name
        if not joint_name:
            continue
        ctrl_min, ctrl_max = model.jnt_range[joint_id]
        if not model.jnt_limited[joint_id] or ctrl_min == ctrl_max:
            ctrl_min, ctrl_max = -1.57, 1.57
        ET.SubElement(
            actuator,
            "position",
            {
                "name": f"{joint_name}_pos",
                "joint": joint_name,
                "kp": "10",
                "ctrllimited": "true",
                "ctrlrange": f"{ctrl_min:g} {ctrl_max:g}",
            },
        )

    ET.indent(root, space="    ")
    tree.write(mjcf_path, encoding="utf-8", xml_declaration=True)
    return mjcf_path


def _prepare_mujoco_model_path(hand_joint, hand_type):
    model_path = _resolve_model_path(hand_joint, hand_type)
    if hand_joint.upper() == "O20":
        return _make_actuated_mjcf_from_urdf(model_path)
    return model_path


def _mapping_joint_name(hand_joint):
    hand_joint = hand_joint.upper()
    if hand_joint == "L20A":
        return "L20"
    return hand_joint


class MujocoNode(Node):
    def __init__(self):
        super().__init__('linker_hand_mujoco_ros2_node')
        self.declare_parameter("hand_type", "right")
        self.hand_type = self.get_parameter('hand_type').get_parameter_value().string_value.lower()
        self.declare_parameter("hand_joint", "L10")
        self.hand_joint = self.get_parameter('hand_joint').get_parameter_value().string_value.upper()
        self.mapping_joint = _mapping_joint_name(self.hand_joint)
        self.declare_parameter("initial_position", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("show_contact_patch", True)
        self.declare_parameter("contact_patch_link", "index_link3")
        self.declare_parameter("contact_patch_mesh", "index_link3")
        self.declare_parameter("contact_patch_radius", 0.004)
        self.declare_parameter("contact_patch_update_hz", 20.0)
        self.declare_parameter("contact_patch_max_points", 120)
        self.declare_parameter("contact_force_sigma", 0.002)
        self.declare_parameter("contact_force_arrow_scale", 0.2)
        self.declare_parameter("record_joint_state", False)
        self.declare_parameter("joint_state_log_path", "")
        self.declare_parameter("joint_state_log_hz", 60.0)
        self.declare_parameter("record_control_gui", True)
        self.create_subscription(JointState,f"/cb_{self.hand_type}_hand_control_cmd",self.hand_cb,10)
        self.create_subscription(Float32MultiArray,f"/cb_{self.hand_type}_hand_contact_cmd",self.hand_contact_cb,10)
        self.index_torque_pub = self.create_publisher(JointState, f"/cb_{self.hand_type}_hand_index_contact_torque", 10)
        # 直接通过字典获取配置
        joint_config = JOINT_CONFIG.get(self.mapping_joint)
        if joint_config:
            self.joint_map = joint_config["map"]
            self.joint_arc = joint_config["arc"]
        else:
            # 处理未匹配的情况（可选）
            self.joint_map = None
            self.joint_arc = None
        XML_PATH = _prepare_mujoco_model_path(self.hand_joint, self.hand_type)
        self.get_logger().info(f"Loading MuJoCo model: {XML_PATH}")

        # --- 加载模型 ---
        self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        self.model.dof_damping[:] = 0.8  # 所有关节都设置为 1.0 阻尼
        self.data = mujoco.MjData(self.model)
        joint_count = self.model.nu
        self.ctrl_values = np.zeros(joint_count)
        # 获取 actuator 控制范围（注意：actuator 不是 joint 本体）
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()


        print("=" * 20,flush=True)
        print(mujoco.mj_versionString(),flush=True)  # 查看MuJoCo版本
        print("=" * 20,flush=True)
        self.data.qpos[:] = 0
        self.data.qvel[:] = 0
        self.apply_initial_position()
        self.model.opt.disableflags = 1
        mujoco.mj_forward(self.model, self.data)

        joint_names = []
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name  # 获取第i个关节的名称
            joint_names.append(joint_name)
            print(f"Joint {i}: {joint_name}",flush=True)
        self.setup_joint_state_recording()
        self.setup_contact_patch_visualization()
        sim_thread = threading.Thread(target=self.mujoco_thread, daemon=True)
        sim_thread.start()
        


    # --- MuJoCo 模拟线程 ---
    def mujoco_thread(self):
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                print("MuJoCo viewer running...")
                while viewer.is_running():
                    with viewer.lock():
                        self.data.ctrl[:] = self.ctrl_values
                        mujoco.mj_step(self.model, self.data)
                        self.record_joint_state_sample()
                        self.update_contact_patch()
                        self.draw_contact_patch(viewer)
                    viewer.sync()
                    time.sleep(0.001)
        finally:
            self.close_joint_state_log()


    def hand_cb(self,data):
        position = data.position
        try:
            self.ctrl_values[:] = self.control_position_to_actuator(position)
        except Exception as e:
            self.get_logger().error(f"Error in hand_cb: {e}")
        time.sleep(0.01)  # 确保数据处理的间隔时间


    def hand_contact_cb(self, data):
        values = list(data.data)
        if len(values) < 2:
            self.get_logger().error("hand_contact_cmd requires joint positions followed by contact_force_g")
            return

        position = values[:-1]
        try:
            self.ctrl_values[:] = self.control_position_to_actuator(position)
            self.contact_force_g = float(values[-1])
            now = time.time()
            if now - self.last_contact_cmd_log_time > 1.0:
                self.last_contact_cmd_log_time = now
                self.get_logger().info(
                    f"Received contact cmd: joints={len(position)}, contact_force_g={self.contact_force_g}"
                )
        except Exception as e:
            self.get_logger().error(f"Error in hand_contact_cb: {e}")


    def apply_initial_position(self):
        position = self.get_parameter("initial_position").value
        if not position:
            return

        try:
            initial_ctrl = np.array(self.control_position_to_actuator(position), dtype=float)
            ctrl_count = min(len(initial_ctrl), len(self.ctrl_values))
            qpos_count = min(len(initial_ctrl), len(self.data.qpos))
            self.ctrl_values[:ctrl_count] = initial_ctrl[:ctrl_count]
            self.data.qpos[:qpos_count] = initial_ctrl[:qpos_count]
        except Exception as e:
            self.get_logger().error(f"Error applying initial_position: {e}")


    def control_position_to_actuator(self, position):
        if self.joint_map is None:
            return self.direct_position_to_actuator(position)

        if self.hand_type == "left":
            tmp = range_to_arc_left(position, self.mapping_joint)
        elif self.hand_type == "right":
            tmp = range_to_arc_right(position, self.mapping_joint)
        else:
            raise ValueError(f"Unsupported hand_type: {self.hand_type}")

        return self.map_position_array(tmp, self.joint_map)


    def direct_position_to_actuator(self, position):
        mapped_array = [0.0] * len(self.ctrl_ranges)
        count = min(len(position), len(self.ctrl_ranges))
        for idx in range(count):
            ctrl_min, ctrl_max = self.ctrl_ranges[idx]
            value = is_within_range(position[idx], 0, 255)
            mapped_array[idx] = scale_value(value, 0, 255, ctrl_min, ctrl_max)
        return mapped_array


    def map_position_array(self, position, joint_map):
        mapped_array = [0.0] * len(joint_map)  # 初始化20长度的数组
        
        for target_idx, source_idx in joint_map.items():
            if source_idx < len(position):
                mapped_array[target_idx] = position[source_idx]
        
        return mapped_array


    def setup_joint_state_recording(self):
        self.joint_state_log_lock = threading.Lock()
        initial_recording = _to_bool(self.get_parameter("record_joint_state").value)
        self.record_joint_state = False
        self.joint_state_log_file = None
        self.joint_state_log_writer = None
        self.joint_state_log_path = None
        self.joint_state_log_base_path = str(self.get_parameter("joint_state_log_path").value).strip()
        self.joint_state_log_specs = []
        self.joint_state_log_count = 0
        self.joint_state_log_interval = 1.0 / max(
            _to_float(self.get_parameter("joint_state_log_hz").value, 60.0),
            1e-6,
        )
        self.last_joint_state_log_time = -self.joint_state_log_interval

        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for joint_id in range(self.model.njnt):
            joint_name = self.model.joint(joint_id).name
            if not joint_name or "thumb" in joint_name.lower():
                continue
            if int(self.model.jnt_type[joint_id]) not in (hinge_type, slide_type):
                continue
            self.joint_state_log_specs.append(
                (
                    joint_name,
                    int(self.model.jnt_qposadr[joint_id]),
                    int(self.model.jnt_dofadr[joint_id]),
                )
            )

        if not self.joint_state_log_specs:
            self.get_logger().warning("Joint-state recording disabled: no non-thumb joints found")
            return

        if initial_recording:
            self.start_joint_state_recording()


    def _next_joint_state_log_path(self):
        if self.joint_state_log_base_path:
            log_path = Path(self.joint_state_log_base_path).expanduser()
            if not log_path.is_absolute():
                log_path = Path.cwd() / log_path
            if not log_path.exists():
                return log_path

            stem = log_path.stem
            suffix = log_path.suffix or ".csv"
            parent = log_path.parent
            index = 1
            while True:
                candidate = parent / f"{stem}_{index}{suffix}"
                if not candidate.exists():
                    return candidate
                index += 1

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        msec = int((time.time() % 1.0) * 1000.0)
        return (
            Path.cwd()
            / "results"
            / "joint_logs"
            / f"{self.hand_joint.lower()}_{self.hand_type}_{timestamp}_{msec:03d}.csv"
        )


    def start_joint_state_recording(self):
        with self.joint_state_log_lock:
            if self.record_joint_state:
                return self.joint_state_log_path
            if not self.joint_state_log_specs:
                self.get_logger().warning("Cannot start joint-state recording: no non-thumb joints found")
                return None

            log_path = self._next_joint_state_log_path()
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self.joint_state_log_file = log_path.open("w", newline="")
            self.joint_state_log_writer = csv.writer(self.joint_state_log_file)

            joint_names = [item[0] for item in self.joint_state_log_specs]
            self.joint_state_log_writer.writerow(
                ["sim_time", "wall_time"]
                + [f"qpos_{name}" for name in joint_names]
                + [f"qvel_{name}" for name in joint_names]
            )
            self.joint_state_log_file.flush()
            self.joint_state_log_path = log_path
            self.joint_state_log_count = 0
            self.last_joint_state_log_time = -self.joint_state_log_interval
            self.record_joint_state = True
            self.get_logger().info(
                f"Started recording {len(joint_names)} non-thumb joints to {log_path} "
                f"at {1.0 / self.joint_state_log_interval:.2f} Hz"
            )
            return log_path


    def stop_joint_state_recording(self):
        with self.joint_state_log_lock:
            self.record_joint_state = False
            if self.joint_state_log_file is None:
                return self.joint_state_log_path, self.joint_state_log_count
            try:
                self.joint_state_log_file.flush()
                self.joint_state_log_file.close()
                self.get_logger().info(
                    f"Stopped joint-state recording: {self.joint_state_log_path} "
                    f"({self.joint_state_log_count} samples)"
                )
                return self.joint_state_log_path, self.joint_state_log_count
            finally:
                self.joint_state_log_file = None
                self.joint_state_log_writer = None


    def toggle_joint_state_recording(self):
        if self.record_joint_state:
            return False, self.stop_joint_state_recording()
        return True, (self.start_joint_state_recording(), self.joint_state_log_count)


    def record_joint_state_sample(self):
        with self.joint_state_log_lock:
            if (
                not self.record_joint_state
                or self.joint_state_log_writer is None
                or self.data.time - self.last_joint_state_log_time < self.joint_state_log_interval
            ):
                return

            self.last_joint_state_log_time = float(self.data.time)
            qpos_values = [
                float(self.data.qpos[qpos_id]) for _, qpos_id, _ in self.joint_state_log_specs
            ]
            qvel_values = [
                float(self.data.qvel[dof_id]) for _, _, dof_id in self.joint_state_log_specs
            ]
            self.joint_state_log_writer.writerow(
                [float(self.data.time), time.time()] + qpos_values + qvel_values
            )
            self.joint_state_log_count += 1
            if self.joint_state_log_count % 60 == 0:
                self.joint_state_log_file.flush()


    def close_joint_state_log(self):
        if getattr(self, "joint_state_log_lock", None) is None:
            return
        self.stop_joint_state_recording()


    def setup_contact_patch_visualization(self):
        self.show_contact_patch = self.get_parameter("show_contact_patch").value
        self.contact_patch_radius = float(self.get_parameter("contact_patch_radius").value)
        self.contact_patch_update_interval = 1.0 / max(
            float(self.get_parameter("contact_patch_update_hz").value), 1.0
        )
        self.contact_patch_max_points = int(self.get_parameter("contact_patch_max_points").value)
        self.contact_patch_points = np.empty((0, 3), dtype=float)
        self.contact_patch_forces = np.empty((0, 3), dtype=float)
        self.contact_patch_lowest_point = None
        self.contact_force_g = 0.0
        self.index_contact_torque = np.zeros(4, dtype=float)
        self.index_joint_names = ["index_joint0", "index_joint1", "index_joint2", "index_joint3"]
        self.index_dof_ids = []
        self.contact_force_sigma = float(self.get_parameter("contact_force_sigma").value)
        self.contact_force_arrow_scale = float(self.get_parameter("contact_force_arrow_scale").value)
        self.last_contact_patch_update_time = 0.0
        self.last_contact_cmd_log_time = 0.0
        self.contact_patch_geom_id = None
        self.contact_patch_body_id = None
        self.contact_patch_vertices = None
        self.contact_patch_faces = None

        if not self.show_contact_patch:
            return

        link_name = self.get_parameter("contact_patch_link").value
        mesh_name = self.get_parameter("contact_patch_mesh").value
        try:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, link_name)
            mesh_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_MESH, mesh_name)
            if body_id < 0:
                raise ValueError(f"body not found: {link_name}")
            if mesh_id < 0:
                raise ValueError(f"mesh not found: {mesh_name}")

            geom_id = None
            for i in range(self.model.ngeom):
                if self.model.geom_bodyid[i] == body_id and self.model.geom_dataid[i] == mesh_id:
                    geom_id = i
                    break
            if geom_id is None:
                raise ValueError(f"geom not found for body={link_name}, mesh={mesh_name}")

            vert_adr = self.model.mesh_vertadr[mesh_id]
            vert_num = self.model.mesh_vertnum[mesh_id]
            face_adr = self.model.mesh_faceadr[mesh_id]
            face_num = self.model.mesh_facenum[mesh_id]
            vertices = self.model.mesh_vert[vert_adr:vert_adr + vert_num].copy()
            faces = self.model.mesh_face[face_adr:face_adr + face_num].copy()
            if len(faces) and faces.max() >= vert_num:
                faces = faces - vert_adr

            self.contact_patch_geom_id = geom_id
            self.contact_patch_body_id = body_id
            self.contact_patch_vertices = vertices
            self.contact_patch_faces = faces.astype(np.int32)
            self.index_dof_ids = self.get_joint_dof_ids(self.index_joint_names)
            self.get_logger().info(
                f"Contact patch enabled: link={link_name}, mesh={mesh_name}, "
                f"radius={self.contact_patch_radius}m"
            )
        except Exception as e:
            self.show_contact_patch = False
            self.get_logger().error(f"Contact patch visualization disabled: {e}")


    def update_contact_patch(self):
        if (
            not self.show_contact_patch
            or self.contact_patch_geom_id is None
            or self.contact_patch_body_id is None
            or self.contact_patch_vertices is None
            or self.contact_patch_faces is None
        ):
            return

        now = time.time()
        if now - self.last_contact_patch_update_time < self.contact_patch_update_interval:
            return
        self.last_contact_patch_update_time = now

        geom_id = self.contact_patch_geom_id
        rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
        position = self.data.geom_xpos[geom_id]
        world_vertices = self.contact_patch_vertices @ rotation.T + position

        lowest_idx = int(np.argmin(world_vertices[:, 2]))
        lowest_point = world_vertices[lowest_idx]
        face_centroids = world_vertices[self.contact_patch_faces].mean(axis=1)
        distances = np.linalg.norm(face_centroids - lowest_point, axis=1)
        selected = np.where(distances <= self.contact_patch_radius)[0]
        if len(selected) == 0:
            selected = np.array([int(np.argmin(distances))])

        points = face_centroids[selected]
        if len(points) > self.contact_patch_max_points:
            sample_idx = np.linspace(0, len(points) - 1, self.contact_patch_max_points).astype(int)
            points = points[sample_idx]

        self.contact_patch_lowest_point = lowest_point
        self.contact_patch_points = points
        self.contact_patch_forces = self.distribute_contact_force(points, lowest_point)
        self.index_contact_torque = self.compute_index_contact_torque(
            points, self.contact_patch_forces
        )
        self.publish_index_contact_torque()


    def distribute_contact_force(self, points, center):
        if len(points) == 0:
            return np.empty((0, 3), dtype=float)

        total_force_n = self.contact_force_g * 9.80665 / 1000.0
        if abs(total_force_n) < 1e-12:
            return np.zeros((len(points), 3), dtype=float)

        sigma = max(self.contact_force_sigma, 1e-6)
        distances = np.linalg.norm(points - center, axis=1)
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
        weight_sum = weights.sum()
        if weight_sum <= 0:
            weights[:] = 1.0 / len(points)
        else:
            weights /= weight_sum

        forces = np.zeros((len(points), 3), dtype=float)
        forces[:, 2] = total_force_n * weights
        return forces


    def compute_index_contact_torque(self, points, forces):
        if len(points) == 0 or len(forces) == 0 or self.contact_patch_body_id is None:
            return np.zeros(4, dtype=float)

        tau_total = np.zeros(self.model.nv, dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for point, force in zip(points, forces):
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.contact_patch_body_id)
            tau_total += jacp.T @ force

        torque = np.zeros(len(self.index_joint_names), dtype=float)
        for i, dof_id in enumerate(self.index_dof_ids):
            if 0 <= dof_id < len(tau_total):
                torque[i] = tau_total[dof_id]
        return torque


    def publish_index_contact_torque(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.index_joint_names
        msg.effort = self.index_contact_torque.tolist()
        self.index_torque_pub.publish(msg)


    def get_joint_dof_ids(self, joint_names):
        dof_ids = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                dof_ids.append(-1)
                continue
            dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
        return dof_ids


    def draw_contact_patch(self, viewer):
        if not self.show_contact_patch or self.contact_patch_lowest_point is None:
            return

        scene = viewer.user_scn
        scene.ngeom = 0
        max_geoms = getattr(scene, "maxgeom", None)
        if max_geoms is None:
            max_geoms = len(scene.geoms)
        force_count = min(len(self.contact_patch_points), len(self.contact_patch_forces))
        draw_forces = (
            force_count > 0
            and np.any(np.linalg.norm(self.contact_patch_forces[:force_count], axis=1) > 1e-12)
        )
        if draw_forces:
            marker_count = min(force_count, max((max_geoms - 1) // 2, 0))
        else:
            marker_count = min(len(self.contact_patch_points), max_geoms - 1)
        if marker_count <= 0:
            return

        identity = np.eye(3).reshape(-1)
        red = np.array([1.0, 0.05, 0.02, 0.85], dtype=float)
        blue = np.array([0.0, 0.25, 1.0, 1.0], dtype=float)
        green = np.array([0.1, 1.0, 0.2, 0.9], dtype=float)
        patch_size = np.array([0.0012, 0.0012, 0.0012], dtype=float)
        lowest_size = np.array([0.0025, 0.0025, 0.0025], dtype=float)

        for point in self.contact_patch_points[:marker_count]:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                patch_size,
                point,
                identity,
                red,
            )
            scene.ngeom += 1

        if draw_forces:
            for point, force in zip(
                self.contact_patch_points[:marker_count],
                self.contact_patch_forces[:marker_count],
            ):
                force_norm = float(np.linalg.norm(force))
                if force_norm < 1e-12 or scene.ngeom >= max_geoms:
                    continue
                direction = force / force_norm
                arrow_length = min(max(force_norm * self.contact_force_arrow_scale, 0.1), 0.5)
                arrow_end = point + direction * arrow_length
                mujoco.mjv_connector(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.002,
                    point,
                    arrow_end,
                )
                scene.geoms[scene.ngeom].rgba[:] = green
                scene.ngeom += 1

        if scene.ngeom < max_geoms:
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                lowest_size,
                self.contact_patch_lowest_point,
                identity,
                blue,
            )
            scene.ngeom += 1


class JointStateRecordControl(QWidget):
    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("Joint State Recorder")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.status_label = QLabel()
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.toggle_button = QPushButton()
        self.toggle_button.clicked.connect(self.toggle_recording)

        layout.addWidget(self.status_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.toggle_button)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)
        self.refresh()

    def toggle_recording(self):
        self.node.toggle_joint_state_recording()
        self.refresh()

    def refresh(self):
        recording = bool(getattr(self.node, "record_joint_state", False))
        sample_count = int(getattr(self.node, "joint_state_log_count", 0))
        joint_count = len(getattr(self.node, "joint_state_log_specs", []))
        path = getattr(self.node, "joint_state_log_path", None)
        self.status_label.setText(
            f"状态: {'记录中' if recording else '未记录'} | "
            f"关节数: {joint_count} | 样本数: {sample_count}"
        )
        self.path_label.setText(f"文件: {path if path is not None else '尚未创建'}")
        self.toggle_button.setText("停止记录" if recording else "开始记录")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoNode()  # 创建 MujocoNode 实例
    use_control_gui = _to_bool(node.get_parameter("record_control_gui").value)

    if use_control_gui:
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        app = QApplication(sys.argv[:1])
        window = JointStateRecordControl(node)
        window.show()
        try:
            app.exec_()
        finally:
            node.close_joint_state_log()
            rclpy.shutdown() # 关闭rclpy
            spin_thread.join(timeout=1.0)
            node.destroy_node()
        return

    try:
        rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    finally:
        node.close_joint_state_log()
        node.destroy_node()
        rclpy.shutdown() # 关闭rclpy
    


if __name__ == '__main__':
    main()
