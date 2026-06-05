import time

import mujoco
import numpy as np


class ContactPatchCalculator:
    def __init__(
        self,
        model,
        data,
        link_name,
        mesh_name,
        radius,
        update_hz,
        max_points,
        force_sigma,
        index_joint_names,
        index_dof_ids,
    ):
        self.model = model
        self.data = data
        self.radius = float(radius)
        self.update_interval = 1.0 / max(float(update_hz), 1.0)
        self.max_points = int(max_points)
        self.force_sigma = float(force_sigma)
        self.index_joint_names = index_joint_names
        self.index_dof_ids = index_dof_ids

        self.points = np.empty((0, 3), dtype=float)
        self.forces = np.empty((0, 3), dtype=float)
        self.lowest_point = None
        self.index_contact_torque = np.zeros(4, dtype=float)
        self.last_update_time = 0.0

        self.geom_id, self.body_id, self.vertices, self.faces = self._load_mesh_patch(
            link_name,
            mesh_name,
        )

    def _load_mesh_patch(self, link_name, mesh_name):
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
        faces = self.model.mesh_face[face_adr:face_adr + face_num].copy()
        if len(faces) and faces.max() >= vert_num:
            faces = faces - vert_adr

        vertices = self.model.mesh_vert[vert_adr:vert_adr + vert_num].copy()
        return geom_id, body_id, vertices, faces.astype(np.int32)

    def update(self, contact_force_g, force_update=False):
        now = time.time()
        if not force_update and now - self.last_update_time < self.update_interval:
            return False
        self.last_update_time = now

        rotation = self.data.geom_xmat[self.geom_id].reshape(3, 3)
        position = self.data.geom_xpos[self.geom_id]
        world_vertices = self.vertices @ rotation.T + position

        lowest_idx = int(np.argmin(world_vertices[:, 2]))
        lowest_point = world_vertices[lowest_idx]
        face_centroids = world_vertices[self.faces].mean(axis=1)
        distances = np.linalg.norm(face_centroids - lowest_point, axis=1)
        selected = np.where(distances <= self.radius)[0]
        if len(selected) == 0:
            selected = np.array([int(np.argmin(distances))])

        points = face_centroids[selected]
        if len(points) > self.max_points:
            sample_idx = np.linspace(0, len(points) - 1, self.max_points).astype(int)
            points = points[sample_idx]

        self.lowest_point = lowest_point
        self.points = points
        self.forces = self.distribute_contact_force(points, lowest_point, contact_force_g)
        self.index_contact_torque = self.compute_index_contact_torque(points, self.forces)
        return True

    def distribute_contact_force(self, points, center, contact_force_g):
        if len(points) == 0:
            return np.empty((0, 3), dtype=float)

        total_force_n = contact_force_g * 9.80665 / 1000.0
        if abs(total_force_n) < 1e-12:
            return np.zeros((len(points), 3), dtype=float)

        sigma = max(self.force_sigma, 1e-6)
        distances = np.linalg.norm(points - center, axis=1)
        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
        weights /= max(weights.sum(), 1e-12)

        forces = np.zeros((len(points), 3), dtype=float)
        forces[:, 2] = total_force_n * weights
        return forces

    def compute_index_contact_torque(self, points, forces):
        if len(points) == 0 or len(forces) == 0:
            return np.zeros(4, dtype=float)

        tau_total = np.zeros(self.model.nv, dtype=float)
        jacp = np.zeros((3, self.model.nv), dtype=float)
        jacr = np.zeros((3, self.model.nv), dtype=float)
        for point, force in zip(points, forces):
            jacp[:] = 0.0
            jacr[:] = 0.0
            mujoco.mj_jac(self.model, self.data, jacp, jacr, point, self.body_id)
            tau_total += jacp.T @ force

        torque = np.zeros(len(self.index_joint_names), dtype=float)
        for i, dof_id in enumerate(self.index_dof_ids):
            if 0 <= dof_id < len(tau_total):
                torque[i] = tau_total[dof_id]
        return torque
