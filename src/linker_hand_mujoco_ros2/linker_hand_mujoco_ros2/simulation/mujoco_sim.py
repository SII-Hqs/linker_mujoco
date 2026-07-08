import mujoco
import numpy as np


class MujocoSim:
    def __init__(self, model_xml, finger_joint_names, finger_frames, clip_joint_command=True):
        self.model = mujoco.MjModel.from_xml_path(str(model_xml))
        self.model.dof_damping[:] = 0.8
        self.data = mujoco.MjData(self.model)
        self.model.opt.disableflags = 1

        self.finger_joint_names = finger_joint_names
        self.finger_frames = finger_frames
        self.clip_joint_command = clip_joint_command
        self.finger_joint_index = {
            joint_name: idx for idx, joint_name in enumerate(self.finger_joint_names)
        }

        self.actuator_joint_names = self.get_actuator_joint_names()
        self.ctrl_values = np.zeros(self.model.nu, dtype=float)

    def get_actuator_joint_names(self):
        joint_names = []
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_names.append(self.model.joint(joint_id).name)
        return joint_names

    def apply_initial_frame(self, frame_index):
        frame_index = min(max(int(frame_index), 0), len(self.finger_frames) - 1)
        command = np.asarray(self.finger_frames[frame_index], dtype=float)
        self.set_joint_command(command)
        self.data.qvel[:] = 0.0
        for joint_name, value in zip(self.finger_joint_names, command):
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                continue
            qpos_id = int(self.model.jnt_qposadr[joint_id])
            if qpos_id < len(self.data.qpos):
                self.data.qpos[qpos_id] = value

    def set_joint_command(self, command):
        if len(command) != len(self.finger_joint_names):
            raise ValueError(
                f"Expected {len(self.finger_joint_names)} joint values, got {len(command)}"
            )

        for actuator_id, joint_name in enumerate(self.actuator_joint_names):
            source_idx = self.finger_joint_index.get(joint_name)
            if source_idx is None:
                continue
            value = float(command[source_idx])
            if self.clip_joint_command:
                ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
                value = float(np.clip(value, ctrl_min, ctrl_max))
            self.ctrl_values[actuator_id] = value

    def get_joint_dof_ids(self, joint_names):
        dof_ids = []
        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id < 0:
                dof_ids.append(-1)
                continue
            dof_ids.append(int(self.model.jnt_dofadr[joint_id]))
        return dof_ids

    def forward(self):
        mujoco.mj_forward(self.model, self.data)

    def step(self):
        self.data.ctrl[:] = self.ctrl_values
        mujoco.mj_step(self.model, self.data)
