"""Contact-force reflex controller for the L20a index finger."""

import mujoco
import numpy as np


INDEX_JOINTS = ("index_joint0", "index_joint1", "index_joint2", "index_joint3")


class ReflexIndexController:
    """Smoothly retract the index finger in response to contact force."""

    def __init__(
        self,
        model,
        joint_names,
        base_command,
        gain,
        max_delta,
        filter_tau,
        rate_limit,
    ):
        self.model = model
        self.joint_names = joint_names
        self.base_command = base_command.copy()
        self.current_command = base_command.copy()
        self.filtered_force = 0.0
        self.gain = float(gain)
        self.max_delta = float(max_delta)
        self.filter_tau = max(float(filter_tau), 1e-6)
        self.rate_limit = max(float(rate_limit), 1e-6)
        self.index_ids = [
            joint_names.index(name)
            for name in INDEX_JOINTS
            if name in joint_names
        ]

    def compute(self, contact_state, dt):
        """Return the next rate-limited joint position command."""
        alpha = float(dt) / (self.filter_tau + float(dt))
        self.filtered_force += alpha * (
            contact_state.normal_force - self.filtered_force
        )

        target_command = self.base_command.copy()
        delta = min(self.gain * self.filtered_force, self.max_delta)
        for source_id in self.index_ids[1:]:
            target_command[source_id] -= delta

        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                continue
            joint_range = self.model.jnt_range[joint_id]
            target_command[i] = np.clip(
                target_command[i],
                joint_range[0],
                joint_range[1],
            )

        max_step = self.rate_limit * float(dt)
        self.current_command += np.clip(
            target_command - self.current_command,
            -max_step,
            max_step,
        )
        return self.current_command.copy()
