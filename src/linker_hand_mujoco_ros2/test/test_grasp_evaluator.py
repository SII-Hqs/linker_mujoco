import numpy as np
from pathlib import Path

try:
    import mujoco
except ImportError:
    import pytest

    pytest.skip("MuJoCo is installed in the linker_mujoco environment", allow_module_level=True)

from linker_hand_mujoco_ros2.grasp.evaluator import (
    ContactMonitor,
    GraspEvaluator,
    HandObjectContact,
)
from linker_hand_mujoco_ros2.grasp.scene import (
    configure_collision_groups,
    configure_grasp_contact_physics,
)
from linker_hand_mujoco_ros2.controllers.deployment_esn import (
    CascadedESNTorqueController,
)


def candidate_contact():
    return HandObjectContact(
        in_contact=True,
        contact_count=2,
        normal_force=1.0,
        points=np.zeros((2, 3)),
        hand_bodies=("thumb_link", "index_link"),
        fingers=("thumb", "index"),
    )


def test_grasp_evaluator_passes_six_axes():
    evaluator = GraspEvaluator(settle_time=0.1, axis_test_time=0.1)
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])

    result = evaluator.update(0.0, pose, 1.0, candidate_contact())
    assert result.state == "settling"
    for sim_time in np.arange(0.11, 0.9, 0.11):
        result = evaluator.update(sim_time, pose, 1.0, candidate_contact())

    assert result.state == "stable"
    assert result.stable


def test_grasp_evaluator_rejects_object_motion():
    evaluator = GraspEvaluator(translation_threshold=0.05)
    pose = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    evaluator.update(0.0, pose, 1.0, candidate_contact())
    moved_pose = pose.copy()
    moved_pose[0] = 0.051

    result = evaluator.update(0.01, moved_pose, 1.0, candidate_contact())

    assert result.state == "failed"
    assert not result.stable


def test_contact_monitor_reads_mujoco_contact_force():
    xml = """
    <mujoco>
      <option gravity="0 0 0"/>
      <worldbody>
        <body name="hand-index" pos="0 0 0">
          <geom name="hand_geom" type="sphere" size="0.02"/>
        </body>
        <body name="object" pos="0.035 0 0">
          <freejoint/>
          <geom name="object_geom" type="sphere" size="0.02" mass="0.1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_step(model, data)

    contact = ContactMonitor(model, data, "object").read()

    assert contact.in_contact
    assert contact.contact_count >= 1
    assert contact.normal_force > 0.0
    assert contact.max_normal_force > 0.0
    assert contact.max_penetration > 0.0
    assert contact.fingers == ("index",)


def test_collision_groups_disable_only_same_group_contacts():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <geom name="floor" type="plane" size="1 1 0.1"/>
            <body name="hand-a"><geom type="sphere" size="0.02"/></body>
            <body name="hand-b"><geom type="sphere" size="0.02"/></body>
            <body name="object"><freejoint/><geom type="sphere" size="0.02" mass="0.1"/></body>
          </worldbody>
        </mujoco>
        """
    )
    configure_collision_groups(model, "object")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contact_pairs = {
        frozenset((int(contact.geom1), int(contact.geom2))) for contact in data.contact
    }

    assert frozenset((1, 2)) not in contact_pairs
    assert frozenset((1, 3)) in contact_pairs
    assert frozenset((0, 3)) in contact_pairs


def test_grasp_contact_physics_profile_is_applied():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <worldbody>
            <body name="hand"><geom type="sphere" size="0.02"/></body>
            <body name="object">
              <freejoint/>
              <geom type="box" size="0.02 0.02 0.02" mass="0.1"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )

    configure_grasp_contact_physics(model, "object")

    assert model.opt.timestep == 0.002
    assert model.opt.solver == mujoco.mjtSolver.mjSOL_NEWTON
    assert model.opt.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert np.all(model.geom_condim == 4)
    assert np.allclose(model.geom_solref, (0.005, 1.0))
    assert np.allclose(model.geom_solimp[:, :3], (0.90, 0.99, 0.002))


def test_self_collision_is_enabled_only_between_different_fingers():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="index_link0">
              <geom name="index_base" type="sphere" size="0.02"/>
              <body name="index_link1">
                <joint type="slide" axis="1 0 0"/>
                <geom name="index_child" type="sphere" size="0.02"/>
              </body>
            </body>
            <body name="middle_link0">
              <joint type="slide" axis="0 1 0"/>
              <geom name="middle_base" type="sphere" size="0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    configure_collision_groups(model, enable_hand_self_collision=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    contact_pairs = {
        frozenset((int(contact.geom1), int(contact.geom2))) for contact in data.contact
    }

    assert frozenset((0, 2)) in contact_pairs
    assert frozenset((1, 2)) in contact_pairs
    assert frozenset((0, 1)) not in contact_pairs


def test_thumb_collides_with_palm_when_self_collision_is_enabled():
    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco>
          <option gravity="0 0 0"/>
          <worldbody>
            <body name="hand_root">
              <geom name="palm" type="sphere" size="0.02"/>
            </body>
            <body name="thumb_link3">
              <joint type="slide" axis="1 0 0"/>
              <geom name="thumb" type="sphere" size="0.02"/>
            </body>
          </worldbody>
        </mujoco>
        """
    )
    configure_collision_groups(model, enable_hand_self_collision=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert any(
        frozenset((int(contact.geom1), int(contact.geom2))) == frozenset((0, 1))
        for contact in data.contact
    )


def test_cascaded_grasp_models_load_and_predict_finite_torque():
    deployment_dir = Path(__file__).resolve().parents[1] / "deployment"
    controller = CascadedESNTorqueController(
        deployment_dir / "cascaded_grasp_esn_controller.py",
        deployment_dir / "grasp_goal_conditioned_esn_model.npz",
        deployment_dir / "grasp_acceleration_torque_esn_model.npz",
        torque_limit=0.2,
    )

    torque = controller.predict(
        controller.target_qpos.copy(),
        np.zeros(21),
        np.zeros(21),
    )

    assert len(controller.joint_names) == 21
    assert torque.shape == (21,)
    assert np.all(np.isfinite(torque))
    assert np.max(np.abs(torque)) <= 0.2
