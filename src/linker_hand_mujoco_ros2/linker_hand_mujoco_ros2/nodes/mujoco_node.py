
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
from PyQt5.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QShortcut,
    QVBoxLayout,
    QWidget,
)
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QKeySequence
from linker_hand_mujoco_ros2.io.package_paths import resource_path
from linker_hand_mujoco_ros2.grasp.evaluator import (
    ContactMonitor,
    GraspEvaluator,
    GraspResult,
)
from linker_hand_mujoco_ros2.grasp.scene import (
    add_test_object,
    configure_collision_groups,
    configure_grasp_contact_physics,
)
from linker_hand_mujoco_ros2.controllers.deployment_esn import (
    CascadedESNTorqueController,
    ESNTorqueController,
)
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


def _to_float_list(value, length, default):
    if isinstance(value, str):
        value = value.replace(",", " ").split()
    try:
        result = [float(item) for item in value]
    except (TypeError, ValueError):
        result = []
    return result if len(result) == length else list(default)


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
    RECORD_TOGGLE_KEYCODES = {ord("R"), ord("r"), 32}
    L20_NON_THUMB_JOINT0_ACTUATOR_IDS = (5, 9, 13, 17)

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
        self.declare_parameter("joint_snapshot_path", "")
        self.declare_parameter("record_control_gui", True)
        self.declare_parameter("grasp_object_control_gui", True)
        self.declare_parameter("grasp_object_control_range", 0.30)
        self.declare_parameter("enable_grasp_evaluation", False)
        self.declare_parameter("grasp_object_name", "grasp_object")
        self.declare_parameter("grasp_object_position", "0.19 0 0.30")
        self.declare_parameter("grasp_object_size", "0.025 0.025 0.025")
        self.declare_parameter("grasp_object_mass", 0.1)
        self.declare_parameter("grasp_object_friction", "0.6 0.02 0.001")
        self.declare_parameter("enable_hand_self_collision", True)
        self.declare_parameter("grasp_contact_timestep", 0.002)
        self.declare_parameter("grasp_contact_solref", "0.005 1.0")
        self.declare_parameter("grasp_contact_solimp", "0.90 0.99 0.002")
        self.declare_parameter("grasp_contact_iterations", 80)
        self.declare_parameter("grasp_contact_ls_iterations", 20)
        self.declare_parameter("grasp_contact_noslip_iterations", 3)
        self.declare_parameter("grasp_contact_impratio", 10.0)
        self.declare_parameter("grasp_min_fingers", 2)
        self.declare_parameter("grasp_min_normal_force", 0.05)
        self.declare_parameter("grasp_max_normal_force", 5.0)
        self.declare_parameter("grasp_hard_max_normal_force", 15.0)
        self.declare_parameter("grasp_candidate_hold_time", 0.25)
        self.declare_parameter("grasp_gravity_test_time", 0.75)
        self.declare_parameter("grasp_settle_time", 0.3)
        self.declare_parameter("grasp_axis_test_time", 0.2)
        self.declare_parameter("grasp_translation_threshold", 0.05)
        self.declare_parameter("grasp_rotation_threshold_deg", 15.0)
        self.declare_parameter("grasp_disturbance_scale", 1.0)
        self.declare_parameter("grasp_publish_hz", 30.0)
        self.declare_parameter("enable_esn_controller", False)
        self.declare_parameter("esn_model_variant", "combined")
        self.declare_parameter("esn_combined_model", "")
        self.declare_parameter("esn_open_model", "")
        self.declare_parameter("esn_grasp_model", "")
        self.declare_parameter("cascaded_esn_controller", "")
        self.declare_parameter("cascaded_goal_model", "")
        self.declare_parameter("cascaded_torque_model", "")
        self.declare_parameter("cascaded_grasp_target_qpos", "")
        self.declare_parameter("esn_control_period", 0.018)
        self.declare_parameter("esn_torque_limit", 0.20)
        self.declare_parameter("esn_ramp_seconds", 0.50)
        self.declare_parameter("esn_max_joint_speed", 5.0)
        self.declare_parameter("esn_enable_object_contact", False)
        self.declare_parameter("esn_contact_torque_scale", 0.60)
        self.declare_parameter("esn_contact_force_target", 1.5)
        self.declare_parameter("esn_contact_force_limit", 5.0)
        self.declare_parameter("esn_contact_force_filter_time", 0.05)
        self.declare_parameter("esn_hold_kp", 0.10)
        self.declare_parameter("esn_hold_kd", 0.01)
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
        self.grasp_evaluation_enabled = _to_bool(
            self.get_parameter("enable_grasp_evaluation").value
        )
        if self.grasp_evaluation_enabled:
            self.model = add_test_object(
                XML_PATH,
                name=str(self.get_parameter("grasp_object_name").value),
                position=_to_float_list(
                    self.get_parameter("grasp_object_position").value, 3, (0.19, 0.0, 0.30)
                ),
                size=_to_float_list(
                    self.get_parameter("grasp_object_size").value, 3, (0.025, 0.025, 0.025)
                ),
                mass=_to_float(self.get_parameter("grasp_object_mass").value, 0.1),
                friction=_to_float_list(
                    self.get_parameter("grasp_object_friction").value, 3, (0.6, 0.02, 0.001)
                ),
                enable_hand_self_collision=_to_bool(
                    self.get_parameter("enable_hand_self_collision").value
                ),
            )
            configure_grasp_contact_physics(
                self.model,
                str(self.get_parameter("grasp_object_name").value),
                timestep=_to_float(
                    self.get_parameter("grasp_contact_timestep").value, 0.002
                ),
                solref=_to_float_list(
                    self.get_parameter("grasp_contact_solref").value,
                    2,
                    (0.005, 1.0),
                ),
                solimp=_to_float_list(
                    self.get_parameter("grasp_contact_solimp").value,
                    3,
                    (0.90, 0.99, 0.002),
                ),
                iterations=int(self.get_parameter("grasp_contact_iterations").value),
                ls_iterations=int(
                    self.get_parameter("grasp_contact_ls_iterations").value
                ),
                noslip_iterations=int(
                    self.get_parameter("grasp_contact_noslip_iterations").value
                ),
                impratio=_to_float(
                    self.get_parameter("grasp_contact_impratio").value, 10.0
                ),
            )
        else:
            self.model = mujoco.MjModel.from_xml_path(str(XML_PATH))
            configure_collision_groups(
                self.model,
                enable_hand_self_collision=_to_bool(
                    self.get_parameter("enable_hand_self_collision").value
                ),
            )
        self.model.dof_damping[:] = 0.8  # 所有关节都设置为 1.0 阻尼
        self.data = mujoco.MjData(self.model)
        joint_count = self.model.nu
        self.ctrl_values = np.zeros(joint_count)
        # 获取 actuator 控制范围（注意：actuator 不是 joint 本体）
        self.ctrl_ranges = self.model.actuator_ctrlrange.copy()


        print("=" * 20,flush=True)
        print(mujoco.mj_versionString(),flush=True)  # 查看MuJoCo版本
        print("=" * 20,flush=True)
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0
        self.apply_initial_position()
        self.model.opt.disableflags &= ~int(mujoco.mjtDisableBit.mjDSBL_CONSTRAINT)
        mujoco.mj_forward(self.model, self.data)

        joint_names = []
        for i in range(self.model.njnt):
            joint_name = self.model.joint(i).name  # 获取第i个关节的名称
            joint_names.append(joint_name)
            print(f"Joint {i}: {joint_name}",flush=True)
        self.setup_joint_state_recording()
        self.setup_contact_patch_visualization()
        self.setup_grasp_evaluation()
        self.setup_esn_controller()
        self.last_record_toggle_key_time = 0.0
        self.simulation_data_lock = threading.RLock()
        sim_thread = threading.Thread(target=self.mujoco_thread, daemon=True)
        sim_thread.start()
        


    # --- MuJoCo 模拟线程 ---
    def mujoco_thread(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                key_callback=self.handle_viewer_key,
            ) as viewer:
                print("MuJoCo viewer running...")
                while viewer.is_running():
                    with viewer.lock():
                        with self.simulation_data_lock:
                            self.update_hand_control()
                            mujoco.mj_step(self.model, self.data)
                            self.apply_grasp_object_position_command()
                            self.update_grasp_evaluation()
                            self.record_joint_state_sample()
                            self.update_contact_patch()
                            self.draw_contact_patch(viewer)
                    viewer.sync()
                    time.sleep(0.001)
        finally:
            self.close_joint_state_log()


    def setup_esn_controller(self):
        self.esn_available = _to_bool(
            self.get_parameter("enable_esn_controller").value
        )
        self.esn_controller = None
        self.esn_control_lock = threading.Lock()
        self.esn_pending_command = None
        self.esn_pending_object_contact = None
        self.esn_running = False
        self.esn_hold_active = False
        self.esn_hold_qpos = None
        self.esn_output_scale = 1.0
        self.esn_object_contact_enabled = True
        self.esn_action = "open"
        self.esn_torque = np.zeros(0, dtype=float)
        self.esn_qpos_ids = np.zeros(0, dtype=int)
        self.esn_dof_ids = np.zeros(0, dtype=int)
        self.esn_next_control_time = 0.0
        self.esn_started_time = 0.0
        self.esn_status = {
            "available": self.esn_available,
            "running": False,
            "action": "open",
            "variant": str(self.get_parameter("esn_model_variant").value),
            "max_torque": 0.0,
            "max_speed": 0.0,
            "state": "disabled" if not self.esn_available else "ready",
            "object_contact": False,
        }
        self.esn_original_gain = self.model.actuator_gainprm.copy()
        self.esn_original_bias = self.model.actuator_biasprm.copy()
        self.esn_object_collision = []
        self.esn_contact_force_target = max(
            _to_float(self.get_parameter("esn_contact_force_target").value, 1.5), 0.0
        )
        self.esn_contact_force_limit = max(
            _to_float(self.get_parameter("esn_contact_force_limit").value, 5.0),
            self.esn_contact_force_target + 1e-6,
        )
        self.esn_contact_force_filter_time = max(
            _to_float(self.get_parameter("esn_contact_force_filter_time").value, 0.05),
            0.0,
        )
        if not self.esn_available:
            return
        if self.hand_joint != "L20A" or self.hand_type != "right":
            raise ValueError("deployment ESN currently supports only L20A right hand")

        def configured_path(parameter_name, default_name):
            requested = str(self.get_parameter(parameter_name).value).strip()
            return Path(requested).expanduser().resolve() if requested else resource_path(
                "deployment", default_name
            )

        variant = str(self.get_parameter("esn_model_variant").value).strip().lower()
        torque_limit = _to_float(self.get_parameter("esn_torque_limit").value, 0.20)
        if variant == "cascaded":
            requested_target = _to_float_list(
                self.get_parameter("cascaded_grasp_target_qpos").value,
                21,
                (),
            )
            self.esn_controller = CascadedESNTorqueController(
                controller_path=configured_path(
                    "cascaded_esn_controller", "cascaded_grasp_esn_controller.py"
                ),
                goal_model_path=configured_path(
                    "cascaded_goal_model", "grasp_goal_conditioned_esn_model.npz"
                ),
                torque_model_path=configured_path(
                    "cascaded_torque_model", "grasp_acceleration_torque_esn_model.npz"
                ),
                torque_limit=torque_limit,
                target_qpos=requested_target or None,
            )
        else:
            self.esn_controller = ESNTorqueController(
                variant=variant,
                torque_limit=torque_limit,
                combined_path=configured_path(
                    "esn_combined_model", "open_grasp_onehot_esn_model.npz"
                ),
                open_path=configured_path("esn_open_model", "open_separate_esn_model.npz"),
                grasp_path=configured_path(
                    "esn_grasp_model", "grasp_separate_esn_model.npz"
                ),
            )
        self.esn_control_period = max(
            _to_float(self.get_parameter("esn_control_period").value, 0.018), 1e-4
        )
        self.esn_ramp_seconds = max(
            _to_float(self.get_parameter("esn_ramp_seconds").value, 0.50), 0.0
        )
        self.esn_max_joint_speed = max(
            _to_float(self.get_parameter("esn_max_joint_speed").value, 5.0), 1e-3
        )
        self.esn_torque_limit = max(
            _to_float(self.get_parameter("esn_torque_limit").value, 0.20), 1e-6
        )
        self.esn_contact_torque_scale = float(np.clip(
            _to_float(self.get_parameter("esn_contact_torque_scale").value, 0.60),
            0.0,
            1.0,
        ))
        self.esn_hold_kp = max(
            _to_float(self.get_parameter("esn_hold_kp").value, 0.10), 0.0
        )
        self.esn_hold_kd = max(
            _to_float(self.get_parameter("esn_hold_kd").value, 0.01), 0.0
        )
        self.esn_output_scale = 1.0
        self.esn_object_contact_enabled = (
            True
            if variant == "cascaded"
            else _to_bool(self.get_parameter("esn_enable_object_contact").value)
        )
        qpos_ids, dof_ids = [], []
        for joint_name in self.esn_controller.joint_names:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name
            )
            if joint_id < 0 or int(self.model.jnt_type[joint_id]) != int(
                mujoco.mjtJoint.mjJNT_HINGE
            ):
                raise ValueError(f"ESN hinge joint not found: {joint_name}")
            qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
        self.esn_qpos_ids = np.asarray(qpos_ids, dtype=int)
        self.esn_dof_ids = np.asarray(dof_ids, dtype=int)
        self.esn_torque = np.zeros(len(dof_ids), dtype=float)
        if self.grasp_monitor is not None:
            for geom_id in self.grasp_monitor.object_geom_ids:
                self.esn_object_collision.append(
                    (
                        geom_id,
                        int(self.model.geom_contype[geom_id]),
                        int(self.model.geom_conaffinity[geom_id]),
                    )
                )
            self._set_esn_object_contact(self.esn_object_contact_enabled)
        self.esn_status.update(
            variant=variant,
            state="ready",
            object_contact=self.esn_object_contact_enabled,
        )
        self.get_logger().info(
            f"ESN torque controller ready: variant={variant}, "
            f"period={self.esn_control_period:.3f}s, object_contact={self.esn_object_contact_enabled}"
        )


    def _set_esn_object_contact(self, enabled):
        for geom_id, contype, conaffinity in self.esn_object_collision:
            self.model.geom_contype[geom_id] = contype if enabled else 0
            self.model.geom_conaffinity[geom_id] = conaffinity if enabled else 0


    def queue_esn_command(self, command):
        with self.esn_control_lock:
            self.esn_pending_command = command


    def queue_esn_object_contact(self, enabled):
        with self.esn_control_lock:
            self.esn_pending_object_contact = bool(enabled)


    def get_esn_status_snapshot(self):
        with self.esn_control_lock:
            return dict(self.esn_status)


    def _update_esn_status(self, **values):
        with self.esn_control_lock:
            self.esn_status.update(values)


    def _hold_current_position(self):
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            if joint_id < 0:
                continue
            qpos_id = int(self.model.jnt_qposadr[joint_id])
            value = float(self.data.qpos[qpos_id])
            if self.model.actuator_ctrllimited[actuator_id]:
                value = float(np.clip(value, *self.model.actuator_ctrlrange[actuator_id]))
            self.ctrl_values[actuator_id] = value


    def _stop_esn(self, state="stopped"):
        self.esn_running = False
        self.esn_hold_active = False
        self.esn_hold_qpos = None
        self.esn_torque[:] = 0.0
        self.data.qfrc_applied[self.esn_dof_ids] = 0.0
        self._hold_current_position()
        self.model.actuator_gainprm[:] = self.esn_original_gain
        self.model.actuator_biasprm[:] = self.esn_original_bias
        self._update_esn_status(running=False, state=state, max_torque=0.0)


    def _apply_pending_esn_command(self):
        with self.esn_control_lock:
            command = self.esn_pending_command
            self.esn_pending_command = None
            contact_enabled = self.esn_pending_object_contact
            self.esn_pending_object_contact = None
        if contact_enabled is not None:
            self.esn_object_contact_enabled = contact_enabled
            self._set_esn_object_contact(contact_enabled)
            self.grasp_candidate_since = None
            self.grasp_candidate_ready = False
            self.grasp_failure_latched = False
            self._update_esn_status(object_contact=contact_enabled)
            mujoco.mj_forward(self.model, self.data)
        if command is None:
            return
        if command == "stop":
            self._stop_esn()
            return
        if command == "reset":
            self.esn_controller.reset()
            self.esn_torque[:] = 0.0
            self._update_esn_status(state="reset")
            return
        if command not in ("open", "grasp"):
            return
        if command == "open" and getattr(self.esn_controller, "variant", "") == "cascaded":
            self._stop_esn(state="cascaded controller supports Grasp only")
            return
        self.esn_controller.set_action(command, reset=True)
        self.esn_action = command
        self.esn_torque[:] = 0.0
        self.esn_running = True
        self.esn_hold_active = False
        self.esn_hold_qpos = None
        self.esn_output_scale = 1.0
        self.grasp_candidate_since = None
        self.grasp_candidate_ready = False
        self.grasp_release_time = None
        self.grasp_release_pose = None
        self.grasp_failure_latched = False
        self.esn_started_time = float(self.data.time)
        self.esn_next_control_time = float(self.data.time)
        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_biasprm[:] = 0.0
        self._update_esn_status(running=True, action=command, state="running")


    def _enter_esn_hold(self):
        if self.esn_hold_active:
            return
        self.esn_running = False
        self.esn_hold_active = True
        self.esn_hold_qpos = self.data.qpos[self.esn_qpos_ids].copy()
        self.esn_torque[:] = 0.0
        self.model.actuator_gainprm[:] = 0.0
        self.model.actuator_biasprm[:] = 0.0
        self._update_esn_status(running=False, state="stable hold")


    def update_hand_control(self):
        if not self.esn_available:
            self.data.ctrl[:] = self.ctrl_values
            return
        self._apply_pending_esn_command()
        self.data.qfrc_applied[self.esn_dof_ids] = 0.0
        if self.esn_hold_active:
            position_error = self.esn_hold_qpos - self.data.qpos[self.esn_qpos_ids]
            torque = (
                self.esn_hold_kp * position_error
                - self.esn_hold_kd * self.data.qvel[self.esn_dof_ids]
            )
            self.esn_torque[:] = np.clip(
                torque, -self.esn_torque_limit, self.esn_torque_limit
            )
            self.data.qfrc_applied[self.esn_dof_ids] = self.esn_torque
            self._update_esn_status(
                max_torque=float(np.max(np.abs(self.esn_torque))),
                max_speed=float(np.max(np.abs(self.data.qvel[self.esn_dof_ids]))),
            )
            return
        if not self.esn_running:
            self.data.ctrl[:] = self.ctrl_values
            return

        speed = float(np.max(np.abs(self.data.qvel[self.esn_dof_ids])))
        if speed > self.esn_max_joint_speed or not np.all(np.isfinite(self.data.qpos)):
            self._stop_esn(state=f"safety stop: speed={speed:.3f} rad/s")
            self.get_logger().error(self.esn_status["state"])
            return
        if self.data.time + 1e-12 >= self.esn_next_control_time:
            try:
                qpos = self.data.qpos[self.esn_qpos_ids].copy()
                qvel = self.data.qvel[self.esn_dof_ids].copy()
                if getattr(self.esn_controller, "requires_qacc", False):
                    torque = self.esn_controller.predict(
                        qpos,
                        qvel,
                        self.data.qacc[self.esn_dof_ids].copy(),
                    )
                else:
                    torque = self.esn_controller.predict(qpos, qvel)
            except (ValueError, FloatingPointError) as exc:
                self._stop_esn(state=f"ESN error: {exc}")
                self.get_logger().error(self.esn_status["state"])
                return
            elapsed = float(self.data.time) - self.esn_started_time
            ramp = min(1.0, elapsed / self.esn_ramp_seconds) if self.esn_ramp_seconds else 1.0
            self.esn_torque[:] = torque * ramp * self.esn_output_scale
            self.esn_next_control_time += self.esn_control_period
        self.data.qfrc_applied[self.esn_dof_ids] = self.esn_torque
        self._update_esn_status(
            max_torque=float(np.max(np.abs(self.esn_torque))),
            max_speed=speed,
        )


    def handle_viewer_key(self, keycode):
        if keycode not in self.RECORD_TOGGLE_KEYCODES:
            return

        now = time.monotonic()
        if now - self.last_record_toggle_key_time < 0.25:
            return

        self.last_record_toggle_key_time = now
        recording, result = self.toggle_joint_state_recording()
        log_path, sample_count = result
        state = "started" if recording else "stopped"
        self.get_logger().info(
            f"Joint-state recording {state} by keyboard: "
            f"{log_path if log_path is not None else 'no file'} ({sample_count} samples)"
        )


    def setup_grasp_evaluation(self):
        self.grasp_monitor = None
        self.grasp_evaluator = None
        self.grasp_object_body_id = -1
        self.grasp_last_publish_time = -np.inf
        self.grasp_publish_interval = 1.0 / max(
            _to_float(self.get_parameter("grasp_publish_hz").value, 30.0), 1e-6
        )
        self.grasp_status_pub = self.create_publisher(
            String, f"/cb_{self.hand_type}_hand_grasp_status", 10
        )
        self.grasp_contact_pub = self.create_publisher(
            Float32MultiArray, f"/cb_{self.hand_type}_hand_object_contact", 10
        )
        self.grasp_object_position_lock = threading.Lock()
        self.pending_grasp_object_position = None
        self.grasp_object_position_target = None
        self.grasp_object_position_control_active = False
        self.grasp_status_snapshot = {
            "state": "disabled",
            "stable": False,
            "contact_count": 0,
            "normal_force_n": 0.0,
        }
        self.grasp_candidate_since = None
        self.grasp_candidate_ready = False
        self.grasp_release_time = None
        self.grasp_release_pose = None
        self.grasp_last_position_control_active = True
        self.grasp_failure_latched = False
        self.grasp_filtered_normal_force = 0.0
        self.grasp_last_contact_sample_time = None
        if not self.grasp_evaluation_enabled:
            return

        object_name = str(self.get_parameter("grasp_object_name").value)
        self.grasp_monitor = ContactMonitor(self.model, self.data, object_name)
        self.grasp_object_body_id = self.grasp_monitor.object_body_id
        freejoint_name = f"{object_name}_freejoint"
        self.grasp_object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, freejoint_name
        )
        if self.grasp_object_joint_id < 0:
            raise ValueError(f"grasp object freejoint not found: {freejoint_name}")
        self.grasp_object_qpos_id = int(
            self.model.jnt_qposadr[self.grasp_object_joint_id]
        )
        self.grasp_object_dof_id = int(
            self.model.jnt_dofadr[self.grasp_object_joint_id]
        )
        qpos_id = self.grasp_object_qpos_id
        self.grasp_object_position_target = self.data.qpos[qpos_id:qpos_id + 3].copy()
        self.grasp_object_position_control_active = True
        self.create_subscription(
            Float32MultiArray,
            f"/cb_{self.hand_type}_hand_grasp_object_position_cmd",
            self.grasp_object_position_cb,
            10,
        )
        self.grasp_evaluator = GraspEvaluator(
            min_fingers=int(self.get_parameter("grasp_min_fingers").value),
            min_normal_force=_to_float(
                self.get_parameter("grasp_min_normal_force").value, 0.05
            ),
            settle_time=_to_float(self.get_parameter("grasp_settle_time").value, 0.3),
            axis_test_time=_to_float(
                self.get_parameter("grasp_axis_test_time").value, 0.2
            ),
            translation_threshold=_to_float(
                self.get_parameter("grasp_translation_threshold").value, 0.05
            ),
            rotation_threshold_deg=_to_float(
                self.get_parameter("grasp_rotation_threshold_deg").value, 15.0
            ),
            disturbance_scale=_to_float(
                self.get_parameter("grasp_disturbance_scale").value, 1.0
            ),
        )
        self.grasp_max_normal_force = max(
            _to_float(self.get_parameter("grasp_max_normal_force").value, 5.0),
            self.grasp_evaluator.min_normal_force,
        )
        self.grasp_hard_max_normal_force = max(
            _to_float(
                self.get_parameter("grasp_hard_max_normal_force").value, 15.0
            ),
            self.grasp_max_normal_force,
        )
        self.grasp_candidate_hold_time = max(
            _to_float(self.get_parameter("grasp_candidate_hold_time").value, 0.25), 0.0
        )
        self.grasp_gravity_test_time = max(
            _to_float(self.get_parameter("grasp_gravity_test_time").value, 0.75), 0.0
        )
        self.get_logger().info(
            f"Grasp evaluation enabled for {object_name}: "
            f"body_id={self.grasp_object_body_id}, six-axis disturbance test active"
        )


    def grasp_object_position_cb(self, msg):
        if len(msg.data) != 3:
            self.get_logger().error(
                "grasp object position command requires [x, y, z] in metres"
            )
            return
        position = np.asarray(msg.data, dtype=float)
        if not np.all(np.isfinite(position)):
            self.get_logger().error("grasp object position must contain finite values")
            return
        self.set_grasp_object_position_target(position)


    def set_grasp_object_position_target(self, position):
        """Queue an XYZ target for application by the simulation thread."""
        position = np.asarray(position, dtype=float).reshape(3)
        with self.grasp_object_position_lock:
            self.pending_grasp_object_position = position.copy()
            self.grasp_object_position_control_active = True


    def toggle_grasp_object_position_control(self):
        """Toggle GUI position hold and return whether hold is active."""
        with self.grasp_object_position_lock:
            self.grasp_object_position_control_active = (
                not self.grasp_object_position_control_active
            )
            return self.grasp_object_position_control_active


    def get_grasp_status_snapshot(self):
        """Return the latest GUI-safe grasp status values."""
        with self.grasp_object_position_lock:
            return dict(self.grasp_status_snapshot)


    def apply_grasp_object_position_command(self):
        if self.grasp_monitor is None:
            return
        with self.grasp_object_position_lock:
            position = self.pending_grasp_object_position
            self.pending_grasp_object_position = None
            if position is not None:
                self.grasp_object_position_target = position.copy()
            target = (
                None
                if self.grasp_object_position_target is None
                else self.grasp_object_position_target.copy()
            )
            control_active = self.grasp_object_position_control_active
        if not control_active or target is None:
            return

        qpos_id = self.grasp_object_qpos_id
        dof_id = self.grasp_object_dof_id
        self.data.qpos[qpos_id:qpos_id + 3] = target
        self.data.qvel[dof_id:dof_id + 6] = 0.0
        self.data.xfrc_applied[self.grasp_object_body_id] = 0.0
        self.grasp_evaluator.reset()
        mujoco.mj_forward(self.model, self.data)
        if position is not None:
            self.get_logger().info(
                f"Moved grasp object to [{target[0]:.4f}, {target[1]:.4f}, {target[2]:.4f}] m"
            )


    def update_grasp_evaluation(self):
        if self.grasp_monitor is None or self.grasp_evaluator is None:
            return

        body_id = self.grasp_object_body_id
        contact = self.grasp_monitor.read()
        with self.grasp_object_position_lock:
            position_control_active = self.grasp_object_position_control_active
        object_pose = np.concatenate((self.data.xpos[body_id], self.data.xquat[body_id]))
        object_weight = float(self.model.body_subtreemass[body_id]) * 9.80665
        sim_time = float(self.data.time)
        if self.grasp_last_contact_sample_time is None:
            contact_dt = float(self.model.opt.timestep)
        else:
            contact_dt = max(sim_time - self.grasp_last_contact_sample_time, 0.0)
        self.grasp_last_contact_sample_time = sim_time
        if self.esn_contact_force_filter_time <= 0.0:
            force_alpha = 1.0
        else:
            force_alpha = 1.0 - np.exp(
                -contact_dt / self.esn_contact_force_filter_time
            )
        self.grasp_filtered_normal_force += force_alpha * (
            contact.normal_force - self.grasp_filtered_normal_force
        )
        force_scale = float(np.clip(
            (self.esn_contact_force_limit - self.grasp_filtered_normal_force)
            / (self.esn_contact_force_limit - self.esn_contact_force_target),
            0.0,
            1.0,
        ))
        fingers = set(contact.fingers)
        opposition_contact = (
            self.esn_object_contact_enabled
            and "thumb" in fingers
            and bool(fingers - {"thumb", "palm"})
            and contact.normal_force >= self.grasp_evaluator.min_normal_force
        )
        envelope_contact = (
            opposition_contact
            and contact.normal_force <= self.grasp_max_normal_force
        )

        if contact.normal_force > self.grasp_hard_max_normal_force:
            self.esn_output_scale = 0.0
            if not self.grasp_failure_latched:
                self._stop_esn(
                    state=(
                        "safety stop: hard contact force="
                        f"{contact.normal_force:.3f} N"
                    )
                )
                self.esn_object_contact_enabled = False
                self._set_esn_object_contact(False)
                self._update_esn_status(object_contact=False)
                self.grasp_failure_latched = True
            result = GraspResult(
                "failed", False, False, -1, 0.0, 0.0,
                "contact force exceeded safety limit", np.zeros(3, dtype=float),
            )
        elif (
            opposition_contact
            and contact.normal_force > self.grasp_max_normal_force
            and (self.grasp_candidate_ready or not position_control_active)
        ):
            self.esn_output_scale = 0.0
            if not self.grasp_failure_latched:
                self._stop_esn(
                    state=f"safety stop: grasp force={contact.normal_force:.3f} N"
                )
                self.grasp_failure_latched = True
            result = GraspResult(
                "failed", False, False, -1, 0.0, 0.0,
                "thumb-opposition force exceeded grasp limit",
                np.zeros(3, dtype=float),
            )
        elif position_control_active:
            self.grasp_release_time = None
            self.grasp_release_pose = None
            self.grasp_evaluator.reset()
            if envelope_contact and self.esn_running and self.esn_action == "grasp":
                if self.grasp_candidate_since is None:
                    self.grasp_candidate_since = sim_time
                held_time = sim_time - self.grasp_candidate_since
                self.grasp_candidate_ready = held_time >= self.grasp_candidate_hold_time
                self.esn_output_scale = min(
                    self.esn_contact_torque_scale, force_scale
                )
                state = "candidate" if self.grasp_candidate_ready else "contact settling"
                reason = (
                    "release the object to start gravity test"
                    if self.grasp_candidate_ready
                    else "waiting for sustained thumb-opposition contact"
                )
            else:
                self.grasp_candidate_since = None
                self.grasp_candidate_ready = False
                if contact.in_contact and self.esn_running and self.esn_action == "grasp":
                    self.esn_output_scale = (
                        min(self.esn_contact_torque_scale, 0.25)
                        if contact.normal_force > self.grasp_max_normal_force
                        else min(self.esn_contact_torque_scale, force_scale)
                    )
                    state = "waiting for thumb"
                    reason = "non-thumb contact detected; waiting for opposition"
                else:
                    self.esn_output_scale = 1.0
                    state = "position_control"
                    reason = "enable contact and run Grasp while the object is held"
            result = GraspResult(
                state, self.grasp_candidate_ready, False, -1, 0.0, 0.0,
                reason, np.zeros(3, dtype=float),
            )
        else:
            if self.grasp_last_position_control_active:
                self.grasp_release_time = sim_time
                self.grasp_release_pose = object_pose.copy()
                self.grasp_evaluator.reset()
            if not self.grasp_candidate_ready:
                if not self.grasp_failure_latched:
                    self._stop_esn(state="grasp failed: released without candidate")
                    self.grasp_failure_latched = True
                result = GraspResult(
                    "failed", False, False, -1, 0.0, 0.0,
                    "object was released before a valid grasp candidate",
                    np.zeros(3, dtype=float),
                )
            elif not envelope_contact:
                if not self.grasp_failure_latched:
                    self._stop_esn(state="grasp failed: contact lost")
                    self.grasp_failure_latched = True
                result = GraspResult(
                    "failed", False, False, -1, 0.0, 0.0,
                    "thumb-opposition contact was lost after release",
                    np.zeros(3, dtype=float),
                )
            elif sim_time - self.grasp_release_time < self.grasp_gravity_test_time:
                translation, rotation = self.grasp_evaluator._pose_error(
                    self.grasp_release_pose, object_pose
                )
                if (
                    translation >= self.grasp_evaluator.translation_threshold
                    or rotation >= self.grasp_evaluator.rotation_threshold_deg
                ):
                    if not self.grasp_failure_latched:
                        self._stop_esn(state="grasp failed: gravity test")
                        self.grasp_failure_latched = True
                    result = GraspResult(
                        "failed", False, False, -1, translation, rotation,
                        "object moved beyond threshold during gravity test",
                        np.zeros(3, dtype=float),
                    )
                else:
                    result = GraspResult(
                        "gravity_test", True, False, -1, translation, rotation,
                        "holding under gravity before disturbance test",
                        np.zeros(3, dtype=float),
                    )
            else:
                result = self.grasp_evaluator.update(
                    sim_time, object_pose, object_weight, contact
                )
                if result.stable:
                    self._enter_esn_hold()
                elif result.state == "failed" and not self.grasp_failure_latched:
                    self._stop_esn(state="grasp failed: disturbance test")
                    self.grasp_failure_latched = True
        self.grasp_last_position_control_active = position_control_active
        self.data.xfrc_applied[body_id, :3] = result.disturbance_force
        self.data.xfrc_applied[body_id, 3:] = 0.0

        if self.data.time - self.grasp_last_publish_time < self.grasp_publish_interval:
            return
        self.grasp_last_publish_time = float(self.data.time)
        status = {
            "state": result.state,
            "candidate": result.candidate,
            "stable": result.stable,
            "test_axis": result.test_axis,
            "translation_error_m": result.translation_error,
            "rotation_error_deg": result.rotation_error_deg,
            "reason": result.reason,
            "contact_count": contact.contact_count,
            "normal_force_n": contact.normal_force,
            "filtered_normal_force_n": self.grasp_filtered_normal_force,
            "max_contact_force_n": contact.max_normal_force,
            "max_penetration_m": contact.max_penetration,
            "fingers": list(contact.fingers),
            "hand_bodies": list(contact.hand_bodies),
        }
        with self.grasp_object_position_lock:
            self.grasp_status_snapshot = dict(status)
        status_msg = String()
        status_msg.data = json.dumps(status, ensure_ascii=True)
        self.grasp_status_pub.publish(status_msg)

        contact_msg = Float32MultiArray()
        contact_msg.data = [
            float(contact.in_contact),
            float(contact.contact_count),
            float(contact.normal_force),
            float(contact.max_penetration),
        ]
        self.grasp_contact_pub.publish(contact_msg)


    def hand_cb(self,data):
        position = data.position
        try:
            self.ctrl_values[:] = self.received_position_to_actuator(position)
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
            self.ctrl_values[:] = self.received_position_to_actuator(position)
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


    def received_position_to_actuator(self, position):
        actuator_position = self.control_position_to_actuator(position)
        if self.hand_joint == "L20":
            for actuator_id in self.L20_NON_THUMB_JOINT0_ACTUATOR_IDS:
                if actuator_id < len(actuator_position):
                    actuator_position[actuator_id] = 0.0
        return actuator_position


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
        self.joint_snapshot_path = None
        self.joint_snapshot_count = 0
        self.joint_snapshot_base_path = str(
            self.get_parameter("joint_snapshot_path").value
        ).strip()
        self.joint_state_log_interval = 1.0 / max(
            _to_float(self.get_parameter("joint_state_log_hz").value, 60.0),
            1e-6,
        )
        self.last_joint_state_log_time = -self.joint_state_log_interval

        hinge_type = int(mujoco.mjtJoint.mjJNT_HINGE)
        slide_type = int(mujoco.mjtJoint.mjJNT_SLIDE)
        for joint_id in range(self.model.njnt):
            joint_name = self.model.joint(joint_id).name
            if not joint_name:
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
            self.get_logger().warning("Joint-state recording disabled: no hinge/slide joints found")
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


    def _joint_snapshot_output_path(self):
        if self.joint_snapshot_base_path:
            snapshot_path = Path(self.joint_snapshot_base_path).expanduser()
            if not snapshot_path.is_absolute():
                snapshot_path = Path.cwd() / snapshot_path
            if not snapshot_path.suffix:
                snapshot_path = snapshot_path.with_suffix(".csv")
            return snapshot_path

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        msec = int((time.time() % 1.0) * 1000.0)
        return (
            Path.cwd()
            / "results"
            / "joint_snapshots"
            / f"{self.hand_joint.lower()}_{self.hand_type}_{timestamp}_{msec:03d}.csv"
        )


    def save_current_joint_snapshot(self):
        if not self.joint_state_log_specs:
            self.get_logger().warning("Cannot save joint snapshot: no hinge/slide joints found")
            return None

        with self.simulation_data_lock:
            sim_time = float(self.data.time)
            qpos_values = [
                float(self.data.qpos[qpos_id])
                for _, qpos_id, _ in self.joint_state_log_specs
            ]
        wall_time = time.time()
        wall_time_iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(wall_time))
        wall_time_iso += f".{int((wall_time % 1.0) * 1000.0):03d}"

        with self.joint_state_log_lock:
            if self.joint_snapshot_path is None:
                self.joint_snapshot_path = self._joint_snapshot_output_path()
            snapshot_path = self.joint_snapshot_path
            snapshot_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not snapshot_path.exists() or snapshot_path.stat().st_size == 0
            with snapshot_path.open("a", newline="") as snapshot_file:
                writer = csv.writer(snapshot_file)
                if write_header:
                    joint_names = [item[0] for item in self.joint_state_log_specs]
                    writer.writerow(
                        ["wall_time_iso", "wall_time_unix", "sim_time"]
                        + [f"qpos_{name}_rad" for name in joint_names]
                    )
                writer.writerow([wall_time_iso, wall_time, sim_time] + qpos_values)
            self.joint_snapshot_count += 1

        self.get_logger().info(
            f"Saved joint snapshot #{self.joint_snapshot_count} to {snapshot_path}"
        )
        return snapshot_path


    def start_joint_state_recording(self):
        with self.joint_state_log_lock:
            if self.record_joint_state:
                return self.joint_state_log_path
            if not self.joint_state_log_specs:
                self.get_logger().warning("Cannot start joint-state recording: no hinge/slide joints found")
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
                f"Started recording {len(joint_names)} joints to {log_path} "
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
        self.object_position_inputs = {}
        self.object_initial_position = None
        if node.grasp_evaluation_enabled:
            self._add_object_controls(layout)
        if node.esn_available:
            self._add_esn_controls(layout)

        self.status_label = QLabel()
        self.path_label = QLabel()
        self.path_label.setWordWrap(True)
        self.snapshot_label = QLabel()
        self.snapshot_label.setWordWrap(True)
        self.toggle_button = QPushButton()
        self.toggle_button.clicked.connect(self.toggle_recording)
        self.toggle_button.setToolTip("R / Space")
        self.snapshot_button = QPushButton("保存当前帧关节角")
        self.snapshot_button.clicked.connect(self.save_snapshot)

        self.shortcuts = []
        for key in (Qt.Key_R, Qt.Key_Space):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.setContext(Qt.ApplicationShortcut)
            shortcut.activated.connect(self.toggle_recording)
            self.shortcuts.append(shortcut)

        layout.addWidget(self.status_label)
        layout.addWidget(self.path_label)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.snapshot_button)
        layout.addWidget(self.snapshot_label)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(250)
        self.refresh()

    def _add_esn_controls(self, layout):
        self.setWindowTitle("MuJoCo ESN Torque Control")
        group = QGroupBox("ESN Torque Controller")
        form = QFormLayout(group)
        variant = self.node.get_esn_status_snapshot().get("variant", "unknown")
        variant_label = QLabel("混训 one-hot" if variant == "combined" else "单训 open/grasp")
        variant_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.addRow("Model:", variant_label)

        buttons = QHBoxLayout()
        open_button = QPushButton("Open")
        open_button.clicked.connect(lambda: self.node.queue_esn_command("open"))
        if variant == "cascaded":
            open_button.setEnabled(False)
            open_button.setToolTip("The cascaded controller supports Grasp only")
        buttons.addWidget(open_button)
        grasp_button = QPushButton("Grasp")
        grasp_button.clicked.connect(lambda: self.node.queue_esn_command("grasp"))
        buttons.addWidget(grasp_button)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.node.queue_esn_command("stop"))
        buttons.addWidget(stop_button)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.node.queue_esn_command("reset"))
        buttons.addWidget(reset_button)
        form.addRow(buttons)

        self.esn_contact_button = QPushButton("Enable Object Contact")
        self.esn_contact_button.clicked.connect(self.toggle_esn_object_contact)
        form.addRow(self.esn_contact_button)

        self.esn_state_label = QLabel("ready")
        self.esn_state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.esn_action_label = QLabel("open")
        self.esn_action_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.esn_torque_label = QLabel("0.0000 N m")
        self.esn_torque_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.esn_speed_label = QLabel("0.0000 rad/s")
        self.esn_speed_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.addRow("State:", self.esn_state_label)
        form.addRow("Action:", self.esn_action_label)
        form.addRow("Max torque:", self.esn_torque_label)
        form.addRow("Max speed:", self.esn_speed_label)
        layout.addWidget(group)

    def toggle_esn_object_contact(self, *_args):
        enabled = bool(
            self.node.get_esn_status_snapshot().get("object_contact", False)
        )
        self.node.queue_esn_object_contact(not enabled)

    def _add_object_controls(self, layout):
        self.setWindowTitle("MuJoCo Grasp Object Control")
        group = QGroupBox("Grasp Object Position")
        form = QFormLayout(group)
        initial = _to_float_list(
            self.node.get_parameter("grasp_object_position").value,
            3,
            (0.19, 0.0, 0.30),
        )
        self.object_initial_position = np.asarray(initial, dtype=float)
        control_range = max(
            _to_float(
                self.node.get_parameter("grasp_object_control_range").value,
                0.30,
            ),
            0.001,
        )
        for axis, value in zip(("X", "Y", "Z"), self.object_initial_position):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(5)
            spin_box.setSingleStep(0.001)
            spin_box.setSuffix(" m")
            spin_box.setRange(float(value - control_range), float(value + control_range))
            spin_box.setValue(float(value))
            spin_box.valueChanged.connect(self.update_object_target)
            self.object_position_inputs[axis] = spin_box
            form.addRow(f"{axis}:", spin_box)

        reset_button = QPushButton("复位方块")
        reset_button.clicked.connect(self.reset_object_position)
        button_layout = QHBoxLayout()
        button_layout.addWidget(reset_button)
        self.object_hold_button = QPushButton("释放方块")
        self.object_hold_button.clicked.connect(self.toggle_object_hold)
        button_layout.addWidget(self.object_hold_button)
        form.addRow(button_layout)
        self.grasp_state_label = QLabel("idle")
        self.grasp_state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.grasp_force_label = QLabel("0.0000 N")
        self.grasp_force_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.grasp_fingers_label = QLabel("none")
        self.grasp_fingers_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.grasp_reason_label = QLabel("-")
        self.grasp_reason_label.setWordWrap(True)
        self.grasp_reason_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        form.addRow("Grasp state:", self.grasp_state_label)
        form.addRow("Normal force:", self.grasp_force_label)
        form.addRow("Contact fingers:", self.grasp_fingers_label)
        form.addRow("Reason:", self.grasp_reason_label)
        layout.addWidget(group)

    def update_object_target(self, *_args):
        if not self.object_position_inputs:
            return
        self.node.set_grasp_object_position_target(
            [self.object_position_inputs[axis].value() for axis in ("X", "Y", "Z")]
        )

    def reset_object_position(self, *_args):
        if self.object_initial_position is None:
            return
        for axis, value in zip(("X", "Y", "Z"), self.object_initial_position):
            spin_box = self.object_position_inputs[axis]
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        self.node.set_grasp_object_position_target(self.object_initial_position)
        self.object_hold_button.setText("释放方块")

    def toggle_object_hold(self, *_args):
        active = self.node.toggle_grasp_object_position_control()
        self.object_hold_button.setText("释放方块" if active else "保持方块")

    def toggle_recording(self):
        self.node.toggle_joint_state_recording()
        self.refresh()

    def save_snapshot(self):
        self.node.save_current_joint_snapshot()
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
        snapshot_path = getattr(self.node, "joint_snapshot_path", None)
        snapshot_count = int(getattr(self.node, "joint_snapshot_count", 0))
        self.snapshot_label.setText(
            f"当前帧文件: {snapshot_path if snapshot_path is not None else '尚未创建'} | "
            f"已保存: {snapshot_count}"
        )
        if self.object_position_inputs:
            grasp = self.node.get_grasp_status_snapshot()
            stable_text = "stable" if grasp.get("stable") else grasp.get("state", "idle")
            self.grasp_state_label.setText(str(stable_text))
            self.grasp_force_label.setText(
                f"{float(grasp.get('normal_force_n', 0.0)):.4f} N"
            )
            fingers = grasp.get("fingers", [])
            self.grasp_fingers_label.setText(
                ", ".join(fingers) if fingers else "none"
            )
            self.grasp_reason_label.setText(str(grasp.get("reason", "-")))
        if self.node.esn_available:
            esn = self.node.get_esn_status_snapshot()
            self.esn_state_label.setText(str(esn.get("state", "unknown")))
            self.esn_action_label.setText(str(esn.get("action", "open")))
            self.esn_torque_label.setText(
                f"{float(esn.get('max_torque', 0.0)):.4f} N m"
            )
            self.esn_speed_label.setText(
                f"{float(esn.get('max_speed', 0.0)):.4f} rad/s"
            )
            contact_enabled = bool(esn.get("object_contact", False))
            self.esn_contact_button.setText(
                "Disable Object Contact" if contact_enabled else "Enable Object Contact"
            )

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


class CascadedGraspControl(QWidget):
    """Focused controls for the grasp-only cascaded ESN deployment."""

    def __init__(self, node):
        super().__init__()
        self.node = node
        self.setWindowTitle("Cascaded Grasp ESN Control")
        self.setMinimumWidth(420)
        layout = QVBoxLayout(self)

        self.object_position_inputs = {}
        self.object_initial_position = None
        if node.grasp_evaluation_enabled:
            self._add_object_controls(layout)
        self._add_controller_controls(layout)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(200)
        self.refresh()

    def _add_object_controls(self, layout):
        group = QGroupBox("Grasp Object Position")
        form = QFormLayout(group)
        initial = _to_float_list(
            self.node.get_parameter("grasp_object_position").value,
            3,
            (0.19, 0.0, 0.30),
        )
        self.object_initial_position = np.asarray(initial, dtype=float)
        control_range = max(
            _to_float(
                self.node.get_parameter("grasp_object_control_range").value,
                0.30,
            ),
            0.001,
        )
        for axis, value in zip(("X", "Y", "Z"), self.object_initial_position):
            spin_box = QDoubleSpinBox()
            spin_box.setDecimals(5)
            spin_box.setSingleStep(0.001)
            spin_box.setSuffix(" m")
            spin_box.setRange(float(value - control_range), float(value + control_range))
            spin_box.setValue(float(value))
            spin_box.valueChanged.connect(self.update_object_target)
            self.object_position_inputs[axis] = spin_box
            form.addRow(f"{axis}:", spin_box)

        buttons = QHBoxLayout()
        reset_button = QPushButton("复位方块")
        reset_button.clicked.connect(self.reset_object_position)
        buttons.addWidget(reset_button)
        self.object_hold_button = QPushButton("释放方块")
        self.object_hold_button.clicked.connect(self.toggle_object_hold)
        buttons.addWidget(self.object_hold_button)
        form.addRow(buttons)
        layout.addWidget(group)

    def _add_controller_controls(self, layout):
        group = QGroupBox("Cascaded Grasp ESN")
        form = QFormLayout(group)

        buttons = QHBoxLayout()
        grasp_button = QPushButton("Grasp")
        grasp_button.clicked.connect(lambda: self.node.queue_esn_command("grasp"))
        buttons.addWidget(grasp_button)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(lambda: self.node.queue_esn_command("stop"))
        buttons.addWidget(stop_button)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(lambda: self.node.queue_esn_command("reset"))
        buttons.addWidget(reset_button)
        form.addRow(buttons)

        contact_label = QLabel("Enabled")
        contact_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controller_state_label = QLabel("ready")
        self.controller_state_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.controller_torque_label = QLabel("0.0000 N m")
        self.controller_torque_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.contact_force_label = QLabel("0.0000 N")
        self.contact_force_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.contact_fingers_label = QLabel("none")
        self.contact_fingers_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.reason_label = QLabel("-")
        self.reason_label.setWordWrap(True)
        self.reason_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)

        form.addRow("Object contact:", contact_label)
        form.addRow("Controller state:", self.controller_state_label)
        form.addRow("Max torque:", self.controller_torque_label)
        form.addRow("Normal force:", self.contact_force_label)
        form.addRow("Contact fingers:", self.contact_fingers_label)
        form.addRow("Grasp state:", self.reason_label)
        layout.addWidget(group)

    def update_object_target(self, *_args):
        if not self.object_position_inputs:
            return
        self.node.set_grasp_object_position_target(
            [self.object_position_inputs[axis].value() for axis in ("X", "Y", "Z")]
        )

    def reset_object_position(self, *_args):
        if self.object_initial_position is None:
            return
        for axis, value in zip(("X", "Y", "Z"), self.object_initial_position):
            spin_box = self.object_position_inputs[axis]
            spin_box.blockSignals(True)
            spin_box.setValue(float(value))
            spin_box.blockSignals(False)
        self.node.set_grasp_object_position_target(self.object_initial_position)
        self.object_hold_button.setText("释放方块")

    def toggle_object_hold(self, *_args):
        active = self.node.toggle_grasp_object_position_control()
        self.object_hold_button.setText("释放方块" if active else "保持方块")

    def refresh(self):
        esn = self.node.get_esn_status_snapshot()
        grasp = self.node.get_grasp_status_snapshot()
        self.controller_state_label.setText(str(esn.get("state", "unknown")))
        self.controller_torque_label.setText(
            f"{float(esn.get('max_torque', 0.0)):.4f} N m"
        )
        self.contact_force_label.setText(
            f"{float(grasp.get('normal_force_n', 0.0)):.4f} N"
        )
        fingers = grasp.get("fingers", [])
        self.contact_fingers_label.setText(", ".join(fingers) if fingers else "none")
        state = "stable" if grasp.get("stable") else grasp.get("state", "idle")
        reason = grasp.get("reason", "-")
        self.reason_label.setText(f"{state}: {reason}")

    def closeEvent(self, event):
        self.timer.stop()
        super().closeEvent(event)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoNode()  # 创建 MujocoNode 实例
    esn_variant = str(node.get_parameter("esn_model_variant").value).strip().lower()
    use_cascaded_gui = node.esn_available and esn_variant == "cascaded"
    use_record_gui = _to_bool(node.get_parameter("record_control_gui").value)
    use_object_gui = (
        node.grasp_evaluation_enabled
        and _to_bool(node.get_parameter("grasp_object_control_gui").value)
    )
    use_control_gui = use_cascaded_gui or use_record_gui or use_object_gui or node.esn_available

    if use_control_gui:
        spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
        spin_thread.start()
        app = QApplication(sys.argv[:1])
        window = (
            CascadedGraspControl(node)
            if use_cascaded_gui
            else JointStateRecordControl(node)
        )
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
