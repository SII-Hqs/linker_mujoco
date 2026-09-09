import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from linker_hand_mujoco_ros2.io import joint_log_processing  # noqa: E402
from linker_hand_mujoco_ros2.tools import (  # noqa: E402
    calculate_joint_log_torque,
)


FILTER_SCRIPT = PACKAGE_ROOT / "scripts" / "filter_joint_logs.py"
RECONSTRUCTION_SCRIPT = (
    PACKAGE_ROOT / "scripts" / "reconstruct_joint_state_from_torque.py"
)
EXTENSION_SCRIPT = PACKAGE_ROOT / "scripts" / "extend_joint_logs.py"


def load_filter_module():
    spec = importlib.util.spec_from_file_location(
        "test_filter_joint_logs_module", FILTER_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_reconstruction_module():
    spec = importlib.util.spec_from_file_location(
        "test_reconstruct_joint_state_module", RECONSTRUCTION_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_extension_module():
    spec = importlib.util.spec_from_file_location(
        "test_extend_joint_logs_module", EXTENSION_SCRIPT
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_final_pose_extension_advances_time_and_stops_derivatives():
    module = load_extension_module()
    fields = [
        "sim_time",
        "wall_time",
        "qpos_joint1",
        "qvel_joint1",
        "qacc_joint1",
        "tau_smoothed_joint1",
    ]
    values = np.asarray(
        [
            [0.00, 100.00, 0.1, 0.3, 0.4, 0.5],
            [0.01, 100.01, 0.2, 0.2, 0.3, 0.6],
            [0.02, 100.02, 0.3, 0.1, 0.2, 0.7],
        ]
    )

    extended, count, rate, duration = module.extend_final_pose(
        fields, values, duration=0.02
    )

    assert count == 2
    assert rate == pytest.approx(100.0)
    assert duration == pytest.approx(0.02)
    np.testing.assert_array_equal(extended[: len(values)], values)
    np.testing.assert_allclose(extended[-2:, 0], [0.03, 0.04])
    np.testing.assert_allclose(extended[-2:, 1], [100.03, 100.04])
    np.testing.assert_allclose(extended[-2:, 2], 0.3)
    np.testing.assert_allclose(extended[-2:, 3:5], 0.0)
    np.testing.assert_allclose(extended[-2:, 5], 0.7)


@pytest.mark.parametrize(
    "method", joint_log_processing.INTERPOLATION_METHODS
)
def test_interpolation_methods_preserve_source_samples(method):
    source_time = np.arange(6, dtype=float)
    source_values = np.asarray([0.0, 1.0, 0.5, 2.0, 1.5, 3.0])

    result = joint_log_processing.interpolate_trace(
        source_time, source_values, source_time, method
    )

    np.testing.assert_allclose(result, source_values)


def test_resampling_regenerates_velocity_and_acceleration():
    timestamps = np.arange(0.0, 1.01, 0.1)
    positions = 1.0 + 2.0 * timestamps
    values = np.column_stack(
        [
            timestamps,
            1000.0 + timestamps,
            positions,
            np.full(len(timestamps), 99.0),
        ]
    )
    fields = ["sim_time", "wall_time", "qpos_test_joint0", "qvel_test_joint0"]

    output_fields, output_values, _ = joint_log_processing.resample_joint_log(
        fields, values, frequency=20.0, method="pchip"
    )

    qvel = output_values[:, output_fields.index("qvel_test_joint0")]
    qacc = output_values[:, output_fields.index("qacc_test_joint0")]
    assert len(output_values) == 21
    np.testing.assert_allclose(qvel, 2.0, atol=1e-10)
    np.testing.assert_allclose(qacc, 0.0, atol=1e-9)


def test_butterworth_filter_reduces_high_frequency_noise():
    module = load_filter_module()
    sample_rate = 100.0
    timestamps = np.arange(0.0, 4.0, 1.0 / sample_rate)
    low_frequency = np.sin(2.0 * np.pi * timestamps)
    noisy = low_frequency + 0.25 * np.sin(2.0 * np.pi * 20.0 * timestamps)

    filtered = module.filter_positions(
        noisy[:, None],
        sample_rate,
        method="butterworth",
        cutoff_hz=5.0,
        order=4,
    )[:, 0]

    raw_error = np.sqrt(np.mean(np.square(noisy - low_frequency)))
    filtered_error = np.sqrt(np.mean(np.square(filtered - low_frequency)))
    assert filtered_error < raw_error * 0.2


@pytest.mark.parametrize(
    "method", ("butterworth", "savgol", "moving-average", "median")
)
def test_filter_methods_return_finite_position_matrix(method):
    module = load_filter_module()
    timestamps = np.arange(0.0, 2.0, 0.01)
    positions = np.column_stack(
        [
            np.sin(2.0 * np.pi * timestamps),
            np.cos(2.0 * np.pi * timestamps),
        ]
    )

    filtered = module.filter_positions(
        positions,
        sample_rate=100.0,
        method=method,
        cutoff_hz=10.0,
        order=4,
        window=11,
        polyorder=3,
    )

    assert filtered.shape == positions.shape
    assert np.all(np.isfinite(filtered))


def test_qpos_torque_plot_is_written(tmp_path):
    timestamps = np.linspace(0.0, 1.0, 20)
    qpos = np.column_stack([timestamps, timestamps ** 2])
    tau = np.column_stack([timestamps * 0.1, -timestamps * 0.2])
    output = tmp_path / "qpos_torque.png"

    result = calculate_joint_log_torque.plot_qpos_torque(
        output,
        timestamps,
        ["joint0", "joint1"],
        qpos,
        tau,
        source_name="synthetic.csv",
    )

    assert result == output.resolve()
    assert output.stat().st_size > 1000


def test_reconstruction_disables_position_actuator(tmp_path):
    module = load_reconstruction_module()
    model_path = tmp_path / "one_joint.xml"
    model_path.write_text(
        """
<mujoco>
  <option timestep="0.001" gravity="0 0 0"/>
  <worldbody>
    <body>
      <joint name="joint0" type="hinge" damping="0.1"/>
      <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.01"/>
    </body>
  </worldbody>
  <actuator><position joint="joint0" kp="100"/></actuator>
</mujoco>
""".strip()
    )
    timestamps = np.arange(0.0, 0.101, 0.01)
    torque = np.zeros((len(timestamps), 1))

    qpos, qvel, qacc = module.reconstruct_state(
        model_path,
        ["joint0"],
        timestamps,
        initial_qpos=np.asarray([0.4]),
        initial_qvel=np.asarray([0.0]),
        torque=torque,
        damping=0.1,
    )

    np.testing.assert_allclose(qpos[:, 0], 0.4, atol=1e-12)
    np.testing.assert_allclose(qvel[:, 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(qacc[:, 0], 0.0, atol=1e-12)


def test_flexion_scope_excludes_thumb_and_joint0():
    module = load_reconstruction_module()
    joint_names = [
        "thumb_joint0",
        "thumb_joint1",
        "index_joint0",
        "index_joint1",
        "index_joint2",
        "middle_joint0",
        "middle_joint3",
        "ring_joint1",
        "little_joint3",
    ]

    indices = module.smoothing_joint_indices(joint_names, "flexion")

    assert [joint_names[index] for index in indices] == [
        "index_joint1",
        "index_joint2",
        "middle_joint3",
        "ring_joint1",
        "little_joint3",
    ]
