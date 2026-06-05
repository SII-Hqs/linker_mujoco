import csv
import json
import pickle
from pathlib import Path

import mujoco
import numpy as np

from linker_hand_mujoco_ros2.contact.contact_patch import ContactPatchCalculator
from linker_hand_mujoco_ros2.mujoco_sim import MujocoSim


def load_retargeting_session(data_dir, pressure_scale=1.0, pressure_offset=0.0):
    data_dir = Path(data_dir).expanduser().resolve()
    pkl_files = sorted(data_dir.glob("*_joints.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No *_joints.pkl found in {data_dir}")

    with open(pkl_files[0], "rb") as f:
        joint_payload = pickle.load(f)
    joint_names = list(joint_payload["meta_data"]["joint_names"])
    joint_frames = np.asarray(joint_payload["data"], dtype=float)
    if joint_frames.ndim != 2 or joint_frames.shape[1] != len(joint_names):
        raise ValueError(f"Invalid joint frame shape: {joint_frames.shape}")

    frame_log_path = data_dir / "frame_log.csv"
    if not frame_log_path.exists():
        raise FileNotFoundError(f"frame_log.csv not found in {data_dir}")

    frame_indices = []
    timestamps = []
    pressures = []
    with open(frame_log_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row_idx, row in enumerate(reader):
            frame_indices.append(int(row.get("frame_idx", row_idx)))
            if row.get("frame_monotonic_ts"):
                timestamps.append(float(row["frame_monotonic_ts"]))
            elif row.get("frame_timestamp_ms"):
                timestamps.append(float(row["frame_timestamp_ms"]) / 1000.0)
            else:
                timestamps.append(np.nan)
            pressures.append(float(row["matched_pressure"]) * pressure_scale + pressure_offset)

    metadata_path = data_dir / "metadata.json"
    fps = 30.0
    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        fps = float(metadata.get("fps", fps))

    frame_count = min(len(joint_frames), len(pressures))
    joint_frames = joint_frames[:frame_count]
    pressures = np.asarray(pressures[:frame_count], dtype=float)
    frame_indices = np.asarray(frame_indices[:frame_count], dtype=int)
    timestamps = np.asarray(timestamps[:frame_count], dtype=float)
    if len(timestamps) != frame_count or not np.all(np.isfinite(timestamps)):
        timestamps = np.arange(frame_count, dtype=float) / fps
    if frame_count > 1 and np.any(np.diff(timestamps) <= 0.0):
        timestamps = np.arange(frame_count, dtype=float) / fps

    return {
        "data_dir": data_dir,
        "joint_names": joint_names,
        "joint_frames": joint_frames,
        "pressures": pressures,
        "frame_indices": frame_indices,
        "timestamps": timestamps,
        "fps": fps,
        "joints_path": pkl_files[0],
    }


def smooth_positions(positions, window):
    window = int(window)
    if window <= 1:
        return positions.copy()
    if window % 2 == 0:
        window += 1
    if len(positions) < 3:
        return positions.copy()

    pad = window // 2
    kernel = np.ones(window, dtype=float) / float(window)
    padded = np.pad(positions, ((pad, pad), (0, 0)), mode="edge")
    smoothed = np.empty_like(positions, dtype=float)
    for col in range(positions.shape[1]):
        smoothed[:, col] = np.convolve(padded[:, col], kernel, mode="valid")
    return smoothed


def differentiate_positions(positions, timestamps):
    if len(positions) < 2:
        return np.zeros_like(positions), np.zeros_like(positions)

    edge_order = 2 if len(positions) > 2 else 1
    qvel = np.gradient(positions, timestamps, axis=0, edge_order=edge_order)
    qacc = np.gradient(qvel, timestamps, axis=0, edge_order=edge_order)
    return qvel, qacc


class DynamicsTorqueCalculator:
    def __init__(
        self,
        model_xml,
        joint_names,
        contact_patch_link="index_link3",
        contact_patch_mesh="index_link3",
        contact_patch_radius=0.004,
        contact_patch_max_points=120,
        contact_force_sigma=0.002,
        index_joint_names=None,
    ):
        if index_joint_names is None:
            index_joint_names = ["index_joint0", "index_joint1", "index_joint2", "index_joint3"]

        self.sim = MujocoSim(model_xml, joint_names, [np.zeros(len(joint_names))], True)
        self.model = self.sim.model
        self.data = self.sim.data
        self.joint_names = joint_names
        self.index_joint_names = index_joint_names
        self.index_dof_ids = self.sim.get_joint_dof_ids(index_joint_names)
        self.joint_qpos_ids, self.joint_dof_ids = self._resolve_joint_addresses(joint_names)

        self.contact_patch = ContactPatchCalculator(
            self.model,
            self.data,
            contact_patch_link,
            contact_patch_mesh,
            contact_patch_radius,
            1.0,
            contact_patch_max_points,
            contact_force_sigma,
            index_joint_names,
            self.index_dof_ids,
        )

    def _resolve_joint_addresses(self, joint_names):
        qpos_ids = []
        dof_ids = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                raise ValueError(f"joint not found: {joint_name}")
            qpos_ids.append(int(self.model.jnt_qposadr[joint_id]))
            dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
        return np.asarray(qpos_ids, dtype=int), np.asarray(dof_ids, dtype=int)

    def set_kinematic_state(self, qpos, qvel, qacc):
        self.data.qpos[:] = self.model.qpos0
        self.data.qvel[:] = 0.0
        self.data.qacc[:] = 0.0
        self.data.qfrc_applied[:] = 0.0
        self.data.xfrc_applied[:] = 0.0
        self.data.ctrl[:] = 0.0

        self.data.qpos[self.joint_qpos_ids] = qpos
        self.data.qvel[self.joint_dof_ids] = qvel
        self.data.qacc[self.joint_dof_ids] = qacc

    def compute_frame(self, qpos, qvel, qacc, contact_force_g):
        self.set_kinematic_state(qpos, qvel, qacc)
        mujoco.mj_inverse(self.model, self.data)
        tau_motion = self._index_values(self.data.qfrc_inverse)

        self.contact_patch.update(contact_force_g, force_update=True)
        tau_contact = self.contact_patch.index_contact_torque.copy()
        tau_required_with_fsr = tau_motion - tau_contact

        return {
            "tau_motion_no_contact": tau_motion,
            "tau_contact_fsr": tau_contact,
            "tau_required_with_fsr": tau_required_with_fsr,
            "contact_point_count": len(self.contact_patch.points),
        }

    def _index_values(self, values):
        out = np.zeros(len(self.index_joint_names), dtype=float)
        for i, dof_id in enumerate(self.index_dof_ids):
            if 0 <= dof_id < len(values):
                out[i] = values[dof_id]
        return out


def calculate_session_dynamics(
    data_dir,
    model_xml,
    output_csv,
    start_frame=0,
    end_frame=None,
    pressure_scale=1.0,
    pressure_offset=0.0,
    smoothing_window=5,
    contact_patch_link="index_link3",
    contact_patch_mesh="index_link3",
    contact_patch_radius=0.004,
    contact_patch_max_points=120,
    contact_force_sigma=0.002,
):
    session = load_retargeting_session(data_dir, pressure_scale, pressure_offset)
    frame_count = len(session["joint_frames"])
    start_frame = min(max(int(start_frame), 0), frame_count)
    if end_frame is None:
        end_frame = frame_count
    end_frame = min(max(int(end_frame), start_frame), frame_count)

    qpos = smooth_positions(session["joint_frames"], smoothing_window)
    qvel, qacc = differentiate_positions(qpos, session["timestamps"])

    calculator = DynamicsTorqueCalculator(
        model_xml,
        session["joint_names"],
        contact_patch_link,
        contact_patch_mesh,
        contact_patch_radius,
        contact_patch_max_points,
        contact_force_sigma,
    )

    output_csv = Path(output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    index_names = calculator.index_joint_names

    fieldnames = ["frame_idx", "timestamp", "fsr_force_g", "contact_point_count"]
    for prefix in ("qpos", "qvel", "qacc"):
        fieldnames.extend(f"{prefix}_{name}" for name in index_names)
    for prefix in ("tau_motion_no_contact", "tau_contact_fsr", "tau_required_with_fsr"):
        fieldnames.extend(f"{prefix}_{name}" for name in index_names)

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(start_frame, end_frame):
            result = calculator.compute_frame(qpos[i], qvel[i], qacc[i], session["pressures"][i])
            row = {
                "frame_idx": int(session["frame_indices"][i]),
                "timestamp": float(session["timestamps"][i]),
                "fsr_force_g": float(session["pressures"][i]),
                "contact_point_count": int(result["contact_point_count"]),
            }
            for local_idx, name in enumerate(index_names):
                source_idx = session["joint_names"].index(name)
                row[f"qpos_{name}"] = float(qpos[i, source_idx])
                row[f"qvel_{name}"] = float(qvel[i, source_idx])
                row[f"qacc_{name}"] = float(qacc[i, source_idx])
                row[f"tau_motion_no_contact_{name}"] = float(
                    result["tau_motion_no_contact"][local_idx]
                )
                row[f"tau_contact_fsr_{name}"] = float(result["tau_contact_fsr"][local_idx])
                row[f"tau_required_with_fsr_{name}"] = float(
                    result["tau_required_with_fsr"][local_idx]
                )
            writer.writerow(row)

    return output_csv, end_frame - start_frame
