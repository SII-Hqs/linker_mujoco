"""Hybrid force-position controller with force PID and position inner loop."""

import mujoco
import numpy as np


INDEX_JOINTS = ("index_joint0", "index_joint1", "index_joint2", "index_joint3")


class HybridForcePositionController:
    """Track contact force by changing index-finger position targets."""

    def __init__(
        self,
        model,
        joint_names,
        base_position,
        target_force,
        kp,
        ki,
        kd,
        integral_limit,
        max_position_delta,
        filter_tau,
        rate_limit,
    ):
        self.model = model
        self.joint_names = joint_names
        self.base_position = np.asarray(base_position, dtype=float).copy()
        self.current_position = self.base_position.copy()
        self.target_force = float(target_force)
        self.kp = float(kp)
        self.ki = float(ki)
        self.kd = float(kd)
        self.integral_limit = abs(float(integral_limit))
        self.max_position_delta = abs(float(max_position_delta))
        self.filter_tau = max(float(filter_tau), 1e-6)
        self.rate_limit = max(float(rate_limit), 1e-6)
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_force = 0.0
        self.last_position_delta = 0.0
        self.index_ids = [
            joint_names.index(name)
            for name in INDEX_JOINTS
            if name in joint_names
        ]
        if not self.index_ids:
            raise ValueError("no index-finger joints found")

    def set_target_force(self, force):
        """Set the runtime force target."""
        self.target_force = float(force)

    def set_base_position(self, command):
        """Update the base posture used by the force outer loop."""
        self.base_position = np.asarray(command, dtype=float).copy()
        self.current_position = self._clip_joint_ranges(self.base_position)

    def reset(self):
        """Reset PID memory and return output to the base posture."""
        self.integral = 0.0
        self.prev_error = 0.0
        self.filtered_force = 0.0
        self.last_position_delta = 0.0
        self.current_position = self.base_position.copy()

    def compute(self, contact_state, dt):
        """Return a full joint position command."""
        dt = max(float(dt), 1e-9)
        alpha = dt / (self.filter_tau + dt)
        self.filtered_force += alpha * (
            float(contact_state.normal_force) - self.filtered_force
        )

        force_error = self.target_force - self.filtered_force
        self.integral += force_error * dt
        self.integral = float(
            np.clip(self.integral, -self.integral_limit, self.integral_limit)
        )
        derivative = (force_error - self.prev_error) / dt
        self.prev_error = force_error

        position_delta = (
            self.kp * force_error
            + self.ki * self.integral
            + self.kd * derivative
        )
        position_delta = float(
            np.clip(
                position_delta,
                -self.max_position_delta,
                self.max_position_delta,
            )
        )
        self.last_position_delta = position_delta

        target_position = self.base_position.copy()
        for source_id in self.index_ids[1:]:
            target_position[source_id] += position_delta
        target_position = self._clip_joint_ranges(target_position)

        max_step = self.rate_limit * dt
        self.current_position += np.clip(
            target_position - self.current_position,
            -max_step,
            max_step,
        )
        return self.current_position.copy()

    def _clip_joint_ranges(self, command):
        clipped = np.asarray(command, dtype=float).copy()
        for i, joint_name in enumerate(self.joint_names):
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )
            if joint_id < 0:
                continue
            joint_range = self.model.jnt_range[joint_id]
            clipped[i] = np.clip(clipped[i], joint_range[0], joint_range[1])
        return clipped
