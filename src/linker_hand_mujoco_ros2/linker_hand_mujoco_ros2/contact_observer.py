"""Read platform contact forces and contact points from MuJoCo."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ContactState:
    """Aggregated contacts between the controlled platform and the hand."""

    in_contact: bool
    normal_force: float
    raw_normal_force: float
    force_world: np.ndarray
    points: np.ndarray
    geoms: tuple


class ContactObserver:
    """Observe contacts involving one named platform geom."""

    def __init__(
        self,
        model,
        data,
        platform_geom_name,
        ignored_geom_names=("floor",),
        force_clip=20.0,
    ):
        self.model = model
        self.data = data
        self.force_clip = float(force_clip)
        self.platform_geom_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_GEOM,
            platform_geom_name,
        )
        if self.platform_geom_id < 0:
            raise ValueError(f"platform geom not found: {platform_geom_name}")
        self.ignored_geom_ids = {
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in ignored_geom_names
        }
        self.ignored_geom_ids.discard(-1)

    def read(self):
        """Return the current aggregate contact state."""
        normal_force = 0.0
        force_world = np.zeros(3, dtype=float)
        points = []
        geoms = []

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1 = int(contact.geom1)
            geom2 = int(contact.geom2)
            if self.platform_geom_id not in (geom1, geom2):
                continue
            other_geom = geom2 if geom1 == self.platform_geom_id else geom1
            if other_geom in self.ignored_geom_ids:
                continue

            force6 = np.zeros(6, dtype=float)
            mujoco.mj_contactForce(self.model, self.data, contact_id, force6)
            frame = contact.frame.reshape(3, 3)
            force_world += frame.T @ force6[:3]
            normal_force += max(float(force6[0]), 0.0)
            points.append(contact.pos.copy())

            geom_name = self.model.geom(other_geom).name
            if not geom_name:
                body_id = int(self.model.geom_bodyid[other_geom])
                geom_name = self.model.body(body_id).name
            geoms.append(geom_name)

        point_array = np.vstack(points) if points else np.empty((0, 3), dtype=float)
        return ContactState(
            in_contact=bool(points),
            normal_force=min(normal_force, self.force_clip),
            raw_normal_force=normal_force,
            force_world=force_world,
            points=point_array,
            geoms=tuple(sorted(set(geoms))),
        )
