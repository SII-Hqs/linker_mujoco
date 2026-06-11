"""Control the mocap-driven contact platform position."""

import threading

import mujoco
import numpy as np


class MovingPlatformController:
    """Drive a mocap platform automatically or from an external XYZ target."""

    def __init__(
        self,
        model,
        data,
        body_name,
        target_name,
        top_offset,
        stroke,
        period,
        initial_pos=None,
        high_z_offset=0.0,
        low_z_offset=0.0,
    ):
        self.model = model
        self.data = data
        self.body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        self.target_geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            target_name,
        )
        self.target_body_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            target_name,
        )
        if self.body_id < 0:
            raise ValueError(f"body not found: {body_name}")
        if self.target_geom_id < 0 and self.target_body_id < 0:
            raise ValueError(f"target geom/body not found: {target_name}")
        if self.target_geom_id < 0:
            self.target_geom_id = self._first_geom_on_body(self.target_body_id)

        self.mocap_id = int(model.body_mocapid[self.body_id])
        if self.mocap_id < 0:
            raise ValueError(f"body is not mocap-controlled: {body_name}")

        mujoco.mj_forward(model, data)
        target_pos = self._target_position()
        platform_geom_id = self._first_geom_on_body(self.body_id)
        platform_half_z = 0.006
        if platform_geom_id >= 0:
            platform_half_z = float(model.geom_size[platform_geom_id, 2])

        if initial_pos is None:
            self.initial_pos = np.array(
                [target_pos[0], target_pos[1], target_pos[2] + top_offset],
                dtype=float,
            )
        else:
            self.initial_pos = np.asarray(initial_pos, dtype=float).copy()
            if self.initial_pos.shape != (3,):
                raise ValueError("initial_pos must contain exactly 3 values")

        self.center_xy = self.initial_pos[:2]
        self.high_z = float(self.initial_pos[2]) - abs(float(high_z_offset))
        self.low_z = float(target_pos[2] + platform_half_z - stroke) + abs(float(low_z_offset))
        self.period = max(float(period), 0.1)
        self._target_pos = None
        self._target_lock = threading.Lock()
        self.data.mocap_quat[self.mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
        self.data.mocap_pos[self.mocap_id] = self.initial_pos

    def set_target_pos(self, x, y, z):
        """Override the automatic trajectory with an external XYZ target."""
        with self._target_lock:
            self._target_pos = np.array([x, y, z], dtype=float)

    def reset_target_pos(self):
        """Set the external target back to the initial platform position."""
        self.set_target_pos(*self.initial_pos)

    def clear_target_override(self):
        """Resume the automatic sinusoidal trajectory."""
        with self._target_lock:
            self._target_pos = None

    def get_target_pos(self):
        """Return the external target or the initial position if automatic."""
        with self._target_lock:
            if self._target_pos is None:
                return self.initial_pos.copy()
            return self._target_pos.copy()

    def update(self, t):
        """Write the current automatic or external target into mocap_pos."""
        with self._target_lock:
            target_pos = None if self._target_pos is None else self._target_pos.copy()
        if target_pos is None:
            phase = 0.5 - 0.5 * np.cos(2.0 * np.pi * t / self.period)
            z = (1.0 - phase) * self.high_z + phase * self.low_z
            target_pos = np.array([self.center_xy[0], self.center_xy[1], z])
        self.data.mocap_pos[self.mocap_id] = target_pos

    def _first_geom_on_body(self, body_id):
        for geom_id in range(self.model.ngeom):
            if int(self.model.geom_bodyid[geom_id]) == body_id:
                return geom_id
        return -1

    def _target_position(self):
        if self.target_geom_id < 0:
            return self.data.xpos[self.target_body_id].copy()

        mesh_id = int(self.model.geom_dataid[self.target_geom_id])
        if mesh_id < 0:
            return self.data.geom_xpos[self.target_geom_id].copy()

        vert_adr = int(self.model.mesh_vertadr[mesh_id])
        vert_num = int(self.model.mesh_vertnum[mesh_id])
        vertices = self.model.mesh_vert[vert_adr:vert_adr + vert_num]
        rotation = self.data.geom_xmat[self.target_geom_id].reshape(3, 3)
        world_vertices = (
            vertices @ rotation.T + self.data.geom_xpos[self.target_geom_id]
        )
        highest_idx = int(np.argmax(world_vertices[:, 2]))
        return world_vertices[highest_idx].copy()
