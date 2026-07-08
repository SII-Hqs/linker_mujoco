"""PID contact-force tracking controller that outputs motor torques."""

import mujoco
import numpy as np


INDEX_JOINTS = ("index_joint0", "index_joint1", "index_joint2", "index_joint3")


class ForceTrackingController:
    """Track a target contact force by commanding index-finger motor torques."""

    def __init__(
        self,
        model,
        target_force,
        kp,
        ki,
        kd,
        integral_limit,
        torque_limit,
        joint_names=INDEX_JOINTS,
    ):
        self.model = model
        self.target_force = float(target_force)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = abs(float(integral_limit))
        self.torque_limit = abs(float(torque_limit))
        self.integral = 0.0
        self.prev_error = 0.0
        self.index_actuator_ids = self._find_actuator_ids(joint_names)

    def set_target_force(self, target_force):
        """Set the force target at runtime."""
        self.target_force = float(target_force)

    def reset(self):
        """Reset PID memory."""
        self.integral = 0.0
        self.prev_error = 0.0

    def compute(self, contact_state, dt):
        """Return a full MuJoCo ctrl vector containing motor torques."""
        dt = max(float(dt), 1e-9)
        measured_force = float(contact_state.normal_force)
        error = self.target_force - measured_force
        self.integral += error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )
        derivative = (error - self.prev_error) / dt
        self.prev_error = error

        torque_delta = (
            self.kp * error
            + self.ki * self.integral
            + self.kd * derivative
        )
        torque_delta = float(
            np.clip(torque_delta, -self.torque_limit, self.torque_limit)
        )

        ctrl = np.zeros(self.model.nu, dtype=float)
        for actuator_id in self.index_actuator_ids:
            ctrl_min, ctrl_max = self.model.actuator_ctrlrange[actuator_id]
            ctrl[actuator_id] = np.clip(torque_delta, ctrl_min, ctrl_max)
        return ctrl

    def _find_actuator_ids(self, joint_names):
        actuator_ids = []
        target_names = set(joint_names)
        for actuator_id in range(self.model.nu):
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])
            joint_name = self.model.joint(joint_id).name
            if joint_name in target_names:
                actuator_ids.append(actuator_id)
        if not actuator_ids:
            raise ValueError("no index-finger actuators found")
        return actuator_ids
