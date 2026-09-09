"""MuJoCo contact monitoring and disturbance-based grasp evaluation."""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass(frozen=True)
class HandObjectContact:
    """Aggregated contacts between the hand and the active object."""

    in_contact: bool
    contact_count: int
    normal_force: float
    points: np.ndarray
    hand_bodies: tuple
    fingers: tuple
    max_penetration: float = 0.0
    max_normal_force: float = 0.0


@dataclass(frozen=True)
class GraspResult:
    """Current state of the online grasp evaluation."""

    state: str
    candidate: bool
    stable: bool
    test_axis: int
    translation_error: float
    rotation_error_deg: float
    reason: str
    disturbance_force: np.ndarray


class ContactMonitor:
    """Read contacts between a named object body and every non-world body."""

    FINGER_NAMES = ("thumb", "index", "middle", "ring", "little")

    def __init__(self, model, data, object_body_name):
        self.model = model
        self.data = data
        self.object_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
        )
        if self.object_body_id < 0:
            raise ValueError(f"object body not found: {object_body_name}")
        self.object_geom_ids = {
            geom_id
            for geom_id in range(model.ngeom)
            if self._is_descendant(int(model.geom_bodyid[geom_id]), self.object_body_id)
        }

    def _is_descendant(self, body_id, ancestor_id):
        while body_id > 0:
            if body_id == ancestor_id:
                return True
            body_id = int(self.model.body_parentid[body_id])
        return False

    @classmethod
    def _finger_name(cls, body_name):
        lower_name = body_name.lower()
        for finger in cls.FINGER_NAMES:
            if finger in lower_name:
                return finger
        return "palm"

    def read(self):
        normal_force = 0.0
        max_normal_force = 0.0
        max_penetration = 0.0
        points = []
        hand_bodies = set()
        fingers = set()
        force6 = np.zeros(6, dtype=float)

        for contact_id in range(self.data.ncon):
            contact = self.data.contact[contact_id]
            geom1, geom2 = int(contact.geom1), int(contact.geom2)
            geom1_is_object = geom1 in self.object_geom_ids
            geom2_is_object = geom2 in self.object_geom_ids
            if geom1_is_object == geom2_is_object:
                continue
            hand_geom = geom2 if geom1_is_object else geom1
            hand_body_id = int(self.model.geom_bodyid[hand_geom])
            if hand_body_id == 0:
                continue

            force6[:] = 0.0
            mujoco.mj_contactForce(self.model, self.data, contact_id, force6)
            contact_normal_force = max(float(force6[0]), 0.0)
            normal_force += contact_normal_force
            max_normal_force = max(max_normal_force, contact_normal_force)
            max_penetration = max(max_penetration, -float(contact.dist))
            points.append(contact.pos.copy())
            body_name = self.model.body(hand_body_id).name or f"body_{hand_body_id}"
            hand_bodies.add(body_name)
            fingers.add(self._finger_name(body_name))

        return HandObjectContact(
            in_contact=bool(points),
            contact_count=len(points),
            normal_force=normal_force,
            points=np.vstack(points) if points else np.empty((0, 3), dtype=float),
            hand_bodies=tuple(sorted(hand_bodies)),
            fingers=tuple(sorted(fingers)),
            max_penetration=max_penetration,
            max_normal_force=max_normal_force,
        )


class GraspEvaluator:
    """Validate a grasp by holding it under disturbances along six axes."""

    AXES = np.asarray(
        ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)),
        dtype=float,
    )

    def __init__(
        self,
        min_fingers=2,
        min_normal_force=0.05,
        settle_time=0.3,
        axis_test_time=0.2,
        translation_threshold=0.05,
        rotation_threshold_deg=15.0,
        disturbance_scale=1.0,
    ):
        self.min_fingers = int(min_fingers)
        self.min_normal_force = float(min_normal_force)
        self.settle_time = float(settle_time)
        self.axis_test_time = float(axis_test_time)
        self.translation_threshold = float(translation_threshold)
        self.rotation_threshold_deg = float(rotation_threshold_deg)
        self.disturbance_scale = float(disturbance_scale)
        self.reset()

    def reset(self):
        self.state = "idle"
        self.state_started = 0.0
        self.reference_pose = None
        self.test_axis = -1
        self.reason = "waiting for contact"

    @staticmethod
    def _pose_error(reference_pose, pose):
        translation = float(np.linalg.norm(pose[:3] - reference_pose[:3]))
        quat_dot = float(np.clip(abs(np.dot(pose[3:], reference_pose[3:])), 0.0, 1.0))
        rotation_deg = float(np.degrees(2.0 * np.arccos(quat_dot)))
        return translation, rotation_deg

    def _is_candidate(self, contact):
        finger_count = len(set(contact.fingers) - {"palm"})
        return (
            contact.in_contact
            and finger_count >= self.min_fingers
            and contact.normal_force >= self.min_normal_force
        )

    def update(self, sim_time, object_pose, object_weight, contact):
        candidate = self._is_candidate(contact)
        disturbance = np.zeros(3, dtype=float)

        if not candidate:
            if self.state not in ("idle", "failed"):
                self.reason = "required hand-object contacts were lost"
            self.state = "idle"
            self.state_started = float(sim_time)
            self.reference_pose = None
            self.test_axis = -1
            return self._result(candidate, False, 0.0, 0.0, disturbance)

        if self.state == "failed":
            translation, rotation = self._pose_error(self.reference_pose, object_pose)
            return self._result(candidate, False, translation, rotation, disturbance)

        if self.state == "idle":
            self.state = "settling"
            self.state_started = float(sim_time)
            self.reference_pose = np.asarray(object_pose, dtype=float).copy()
            self.test_axis = -1
            self.reason = "contact candidate is settling"

        translation, rotation = self._pose_error(self.reference_pose, object_pose)
        if translation >= self.translation_threshold or rotation >= self.rotation_threshold_deg:
            self.state = "failed"
            self.reason = "object pose exceeded the stability threshold"
            return self._result(candidate, False, translation, rotation, disturbance)

        elapsed = float(sim_time) - self.state_started
        if self.state == "settling" and elapsed >= self.settle_time:
            self.state = "testing"
            self.state_started = float(sim_time)
            self.test_axis = 0
            self.reason = "running six-axis disturbance test"
            elapsed = 0.0

        if self.state == "testing":
            if elapsed >= self.axis_test_time:
                self.test_axis += 1
                self.state_started = float(sim_time)
            if self.test_axis >= len(self.AXES):
                self.state = "stable"
                self.test_axis = -1
                self.reason = "passed all six disturbance directions"
            else:
                disturbance = self.AXES[self.test_axis] * float(object_weight) * self.disturbance_scale

        stable = self.state == "stable"
        return self._result(candidate, stable, translation, rotation, disturbance)

    def _result(self, candidate, stable, translation, rotation, disturbance):
        return GraspResult(
            state=self.state,
            candidate=bool(candidate),
            stable=bool(stable),
            test_axis=self.test_axis,
            translation_error=float(translation),
            rotation_error_deg=float(rotation),
            reason=self.reason,
            disturbance_force=disturbance,
        )
