"""Helpers for composing a graspable object into an existing MuJoCo model."""

import mujoco


HAND_COLLISION = 1
OBJECT_COLLISION = 2
ENVIRONMENT_COLLISION = 4
FINGER_COLLISIONS = {
    "thumb": 8,
    "index": 16,
    "middle": 32,
    "ring": 64,
    "little": 128,
}
ALL_FINGER_COLLISIONS = sum(FINGER_COLLISIONS.values())
ALL_HAND_COLLISIONS = HAND_COLLISION | ALL_FINGER_COLLISIONS


def configure_grasp_contact_physics(
    model,
    object_body_name,
    timestep=0.002,
    solref=(0.005, 1.0),
    solimp=(0.90, 0.99, 0.002),
    iterations=80,
    ls_iterations=20,
    noslip_iterations=3,
    impratio=10.0,
):
    """Apply a stable contact profile to hand-object collision geoms."""
    object_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
    )
    if object_body_id < 0:
        raise ValueError(f"object body not found: {object_body_name}")

    model.opt.timestep = max(float(timestep), 1e-5)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    model.opt.iterations = max(int(iterations), 1)
    model.opt.ls_iterations = max(int(ls_iterations), 0)
    model.opt.noslip_iterations = max(int(noslip_iterations), 0)
    model.opt.impratio = max(float(impratio), 1.0)

    solref = tuple(float(value) for value in solref)
    solimp = tuple(float(value) for value in solimp)
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        is_object = _is_descendant(model, body_id, object_body_id)
        is_hand = body_id != 0 and not is_object
        if not (is_hand or is_object):
            continue
        model.geom_solref[geom_id, :len(solref)] = solref
        model.geom_solimp[geom_id, :len(solimp)] = solimp
        model.geom_condim[geom_id] = 4

    return model


def _is_descendant(model, body_id, ancestor_id):
    while body_id > 0:
        if body_id == ancestor_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


def _finger_collision_category(model, body_id):
    while body_id > 0:
        body_name = (model.body(body_id).name or "").lower()
        for finger_name, category in FINGER_COLLISIONS.items():
            if finger_name in body_name:
                return category
        body_id = int(model.body_parentid[body_id])
    return HAND_COLLISION


def _is_thumb_distal_body(model, body_id):
    body_name = (model.body(body_id).name or "").lower()
    return "thumb" in body_name and any(
        marker in body_name for marker in ("link3", "link4", "distal", "tip")
    )


def configure_collision_groups(
    model, object_body_name=None, enable_hand_self_collision=False
):
    """Configure hand/object collision and optional collision between fingers."""
    object_body_id = -1
    if object_body_name:
        object_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, object_body_name
        )
        if object_body_id < 0:
            raise ValueError(f"object body not found: {object_body_name}")

    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        if body_id == 0:
            category = ENVIRONMENT_COLLISION
            affinity = ALL_HAND_COLLISIONS | OBJECT_COLLISION
        elif object_body_id >= 0 and _is_descendant(model, body_id, object_body_id):
            category = OBJECT_COLLISION
            affinity = ALL_HAND_COLLISIONS | ENVIRONMENT_COLLISION
        else:
            category = _finger_collision_category(model, body_id)
            affinity = OBJECT_COLLISION | ENVIRONMENT_COLLISION
            if enable_hand_self_collision and category in FINGER_COLLISIONS.values():
                affinity |= ALL_FINGER_COLLISIONS & ~category
                if (
                    category == FINGER_COLLISIONS["thumb"]
                    and _is_thumb_distal_body(model, body_id)
                ):
                    affinity |= HAND_COLLISION
        model.geom_contype[geom_id] = category
        model.geom_conaffinity[geom_id] = affinity

    return model


def add_test_object(
    model_path,
    name="grasp_object",
    position=(0.19, 0.0, 0.30),
    size=(0.025, 0.025, 0.025),
    mass=0.1,
    friction=(0.6, 0.02, 0.001),
    enable_hand_self_collision=False,
):
    """Compile a model with one free box used for contact and grasp tests."""
    spec = mujoco.MjSpec.from_file(str(model_path))
    body = spec.worldbody.add_body(name=name, pos=position)
    body.add_freejoint(name=f"{name}_freejoint")
    body.add_geom(
        name=f"{name}_geom",
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=size,
        mass=float(mass),
        friction=friction,
        condim=4,
        rgba=(0.95, 0.45, 0.08, 1.0),
    )
    model = spec.compile()
    return configure_collision_groups(model, name, enable_hand_self_collision)
