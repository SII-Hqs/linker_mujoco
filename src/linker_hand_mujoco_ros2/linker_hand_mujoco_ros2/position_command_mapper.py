"""Map named joint position commands to MuJoCo actuator controls."""

import mujoco
import numpy as np


class PositionCommandMapper:
    """Convert joint-ordered commands into the model actuator order."""

    def __init__(self, model, joint_names):
        self.model = model
        self.joint_index = {name: i for i, name in enumerate(joint_names)}
        self.actuator_joint_names = []
        for actuator_id in range(model.nu):
            joint_id = int(model.actuator_trnid[actuator_id, 0])
            self.actuator_joint_names.append(model.joint(joint_id).name)

    def command_to_ctrl(self, command):
        """Return a clipped MuJoCo ctrl vector for the given joint command."""
        ctrl = np.zeros(self.model.nu, dtype=float)
        for actuator_id, joint_name in enumerate(self.actuator_joint_names):
            source_idx = self.joint_index.get(joint_name)
            if source_idx is None:
                continue
            value = float(command[source_idx])
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
            ctrl[actuator_id] = np.clip(value, ctrl_min, ctrl_max)
        return ctrl

    def clip_command(self, command):
        """Return a joint command clipped to the matching actuator ranges."""
        clipped = np.asarray(command, dtype=float).copy()
        for actuator_id, joint_name in enumerate(self.actuator_joint_names):
            source_idx = self.joint_index.get(joint_name)
            if source_idx is None:
                continue
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
            clipped[source_idx] = np.clip(clipped[source_idx], ctrl_min, ctrl_max)
        return clipped

    def apply_qpos(self, data, command):
        """Apply a joint command directly to qpos for initialization."""
        for joint_name, value in zip(self.joint_index.keys(), command):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                continue
            qpos_id = int(self.model.jnt_qposadr[joint_id])
            if qpos_id < len(data.qpos):
                data.qpos[qpos_id] = float(value)
