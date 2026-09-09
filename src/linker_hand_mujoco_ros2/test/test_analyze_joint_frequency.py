import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PACKAGE_ROOT / "scripts" / "analyze_joint_frequency.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "test_analyze_joint_frequency_module", SCRIPT_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_target_columns_exclude_thumb_and_joint0():
    module = load_module()
    fields = [
        "sim_time",
        "qpos_thumb_joint1",
        "qpos_index_joint0",
        "qpos_index_joint1",
        "qpos_little_joint3",
        "qvel_middle_joint0",
        "qvel_middle_joint2",
        "tau_inverse_thumb_joint2",
        "tau_inverse_ring_joint0",
        "tau_inverse_ring_joint3",
    ]

    assert module.target_columns(fields) == [
        "qpos_index_joint1",
        "qpos_little_joint3",
        "qvel_middle_joint2",
    ]
    assert module.target_columns(fields, ("tau_inverse",)) == [
        "tau_inverse_ring_joint3"
    ]


def test_welch_analysis_detects_synthetic_frequency():
    module = load_module()
    sample_rate = 100.0
    time = np.arange(0.0, 12.0, 1.0 / sample_rate)
    values = 0.3 * time + np.sin(2.0 * np.pi * 3.0 * time)

    frequencies, _, summary = module.analyze_trace(
        values, sample_rate, nperseg=512
    )

    assert abs(summary["peak_frequency_hz"] - 3.0) <= frequencies[1]
    assert summary["frequency_95pct_hz"] < 4.0


def test_read_log_rejects_nonuniform_sim_time(tmp_path):
    module = load_module()
    path = tmp_path / "nonuniform.csv"
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(("sim_time", "qpos_index_joint1"))
        writer.writerows(((0.0, 0.0), (0.01, 1.0), (0.02, 0.0), (0.031, -1.0)))

    try:
        module.read_log(path)
    except ValueError as error:
        assert "not uniformly sampled" in str(error)
    else:
        raise AssertionError("expected nonuniform sampling to be rejected")
