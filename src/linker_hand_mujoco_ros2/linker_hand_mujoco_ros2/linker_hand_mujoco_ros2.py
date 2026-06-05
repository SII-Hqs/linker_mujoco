
import rclpy,math,sys                                     # ROS2 Python接口库
from rclpy.node import Node                      # ROS2 节点类
from rcl_interfaces.msg import ParameterDescriptor
import rclpy.time
from std_msgs.msg import String, Header, Float32MultiArray
from sensor_msgs.msg import JointState
import time,threading, json


import sys,os
import threading
from sensor_msgs.msg import JointState
import numpy as np
import mujoco, time
import mujoco.viewer
from PyQt5.QtWidgets import QApplication, QWidget, QSlider, QVBoxLayout, QLabel
from PyQt5.QtCore import Qt
from .utils.mapping import *

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
class MujocoNode(Node):
    def __init__(self):
        super().__init__('linker_hand_mujoco_ros2_node')
        self.declare_parameter("hand_type", "right")
        self.hand_type = self.get_parameter('hand_type').get_parameter_value().string_value
        self.declare_parameter("hand_joint", "L10")
        self.hand_joint = self.get_parameter('hand_joint').get_parameter_value().string_value
        self.declare_parameter("initial_position", [], ParameterDescriptor(dynamic_typing=True))
        self.declare_parameter("show_contact_patch", True)
        self.declare_parameter("contact_patch_link", "index_link3")
        self.declare_parameter("contact_patch_mesh", "index_link3")
        self.declare_parameter("contact_patch_radius", 0.004)
        self.declare_parameter("contact_patch_update_hz", 20.0)
        self.declare_parameter("contact_patch_max_points", 120)
        self.declare_parameter("contact_force_sigma", 0.002)
        self.declare_parameter("contact_force_arrow_scale", 0.2)
        self.create_subscription(JointState,f"/cb_{self.hand_type}_hand_control_cmd",self.hand_cb,10)
        self.create_subscription(Float32MultiArray,f"/cb_{self.hand_type}_hand_contact_cmd",self.hand_contact_cb,10)
        self.index_torque_pub = self.create_publisher(JointState, f"/cb_{self.hand_type}_hand_index_contact_torque", 10)
        # 直接通过字典获取配置
        joint_config = JOINT_CONFIG.get(self.hand_joint)
        if joint_config:
            self.joint_map = joint_config["map"]
            self.joint_arc = joint_config["arc"]
        else:
            # 处理未匹配的情况（可选）
            self.joint_map = None
            self.joint_arc = None
        XML_PATH = os.path.dirname(os.path.abspath(__file__))+f"/urdf/{self.hand_joint.upper()}/linker_hand_{self.hand_joint.lower()}_{self.hand_type}/linker_hand_{self.hand_joint.lower()}_{self.hand_type}.xml"

        # --- 加载模型 ---
        self.model = mujoco.MjModel.from_xml_path(XML_PATH)
        self.model.dof_damping[:] = 0.8  # 所有关节都设置为 1.0 阻尼
        self.data = mujoco.MjData(self.model)
        joint_count = self.model.nu
        self.ctrl_values = np.zeros(joint_count)


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
        # 获取 actuator 控制范围（注意：actuator 不是 joint 本体）
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()
        self.setup_contact_patch_visualization()
        sim_thread = threading.Thread(target=self.mujoco_thread)
        sim_thread.start()
        


    # --- MuJoCo 模拟线程 ---
    def mujoco_thread(self):
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            print("MuJoCo viewer running...")
            while viewer.is_running():
                with viewer.lock():
                    self.data.ctrl[:] = self.ctrl_values
                    mujoco.mj_step(self.model, self.data)
                    self.update_contact_patch()
                    self.draw_contact_patch(viewer)
                viewer.sync()
                time.sleep(0.001)


    def hand_cb(self,data):
        position = data.position
        try:
            if self.joint_map is not None:
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
            if self.joint_map is not None:
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
        if self.hand_type == "left":
            tmp = range_to_arc_left(position, self.hand_joint)
        elif self.hand_type == "right":
            tmp = range_to_arc_right(position, self.hand_joint)
        else:
            raise ValueError(f"Unsupported hand_type: {self.hand_type}")

        return self.map_position_array(tmp, self.joint_map)


    def map_position_array(self, position, joint_map):
        mapped_array = [0.0] * len(joint_map)  # 初始化20长度的数组
        
        for target_idx, source_idx in joint_map.items():
            if source_idx < len(position):
                mapped_array[target_idx] = position[source_idx]
        
        return mapped_array


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


def main(args=None):
    rclpy.init(args=args)
    node = MujocoNode()  # 创建 MujocoNode 实例
    rclpy.spin(node) # 保持节点运行，检测是否收到退出指令（Ctrl+C）
    rclpy.shutdown() # 关闭rclpy
    


if __name__ == '__main__':
    main()
