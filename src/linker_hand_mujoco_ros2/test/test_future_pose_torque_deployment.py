import importlib.util
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "deployment" / "future_pose_torque_mujoco.py"
MODEL_PATH = PACKAGE_ROOT / "deployment" / "future_pose_10ms_torque_esn_model.npz"


def load_deployment_module():
    spec = importlib.util.spec_from_file_location(
        "test_future_pose_torque_deployment_module", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_future_pose_model_contract_and_prediction():
    module = load_deployment_module()
    controller = module.FuturePoseTorqueController(MODEL_PATH, torque_limit=1.0)

    assert controller.sample_rate_hz == 1000.0
    assert controller.future_horizon == 0.01
    assert len(controller.joint_names) == 12
    assert all(not name.startswith("thumb_") for name in controller.joint_names)
    assert all(not name.endswith("joint0") for name in controller.joint_names)

    current = controller.input_mean[:12]
    future = controller.input_mean[12:]
    output = controller.predict(current, future)

    assert output.raw_torque.shape == (12,)
    assert output.clipped_torque.shape == (12,)
    assert np.all(np.isfinite(output.raw_torque))
    assert np.all(np.abs(output.clipped_torque) <= 1.0)
    assert output.max_abs_input_zscore == 0.0
    assert controller.activation.shape == controller.state.shape
    assert np.all(np.isfinite(controller.activation))

    controller.reset()
    assert np.all(controller.state == 0.0)
    assert np.all(controller.activation == 0.0)


def test_object_control_state_updates_transform_and_status():
    module = load_deployment_module()
    state = module.ObjectControlState(
        np.asarray([0.19, 0.0, 0.10]), np.asarray([0.05, 0.05, 0.05])
    )

    state.set_position(np.asarray([0.20, 0.01, 0.31]))
    position, size = state.transform()
    state.set_held(False)
    state.set_contacts_enabled(True)
    state.update_status(1.25, True, 2, 0.75, ("index", "thumb"))

    assert np.allclose(position, [0.20, 0.01, 0.31])
    assert np.allclose(size, [0.05, 0.05, 0.05])
    assert state.interaction() == (False, True, 2)
    assert state.status() == (1.25, True, 2, 0.75, ("index", "thumb"))

    state.request_reset()
    assert state.consume_reset_request() is True
    assert state.consume_reset_request() is False
