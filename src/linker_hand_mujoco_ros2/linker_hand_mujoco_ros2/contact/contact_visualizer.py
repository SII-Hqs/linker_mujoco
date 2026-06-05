import mujoco
import numpy as np


class ContactVisualizer:
    def __init__(self, force_arrow_scale):
        self.force_arrow_scale = float(force_arrow_scale)

    def draw(self, viewer, contact_patch):
        if contact_patch.lowest_point is None:
            return

        scene = viewer.user_scn
        scene.ngeom = 0
        max_geoms = getattr(scene, "maxgeom", None)
        if max_geoms is None:
            max_geoms = len(scene.geoms)

        force_count = min(len(contact_patch.points), len(contact_patch.forces))
        draw_forces = (
            force_count > 0
            and np.any(np.linalg.norm(contact_patch.forces[:force_count], axis=1) > 1e-12)
        )
        if draw_forces:
            marker_count = min(force_count, max((max_geoms - 1) // 2, 0))
        else:
            marker_count = min(len(contact_patch.points), max_geoms - 1)
        if marker_count <= 0:
            return

        identity = np.eye(3).reshape(-1)
        red = np.array([1.0, 0.05, 0.02, 0.85], dtype=float)
        blue = np.array([0.0, 0.25, 1.0, 1.0], dtype=float)
        green = np.array([0.1, 1.0, 0.2, 0.9], dtype=float)
        patch_size = np.array([0.0012, 0.0012, 0.0012], dtype=float)
        lowest_size = np.array([0.0025, 0.0025, 0.0025], dtype=float)

        for point in contact_patch.points[:marker_count]:
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
                contact_patch.points[:marker_count],
                contact_patch.forces[:marker_count],
            ):
                force_norm = float(np.linalg.norm(force))
                if force_norm < 1e-12 or scene.ngeom >= max_geoms:
                    continue
                direction = force / force_norm
                arrow_length = min(max(force_norm * self.force_arrow_scale, 0.05), 0.10)
                arrow_end = point + direction * arrow_length
                mujoco.mjv_connector(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    0.0005,
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
                contact_patch.lowest_point,
                identity,
                blue,
            )
            scene.ngeom += 1
