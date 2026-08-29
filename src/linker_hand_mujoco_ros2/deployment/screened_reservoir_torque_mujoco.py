#!/usr/bin/env python3
"""Deploy any retained 100 ms screened-reservoir torque ESN in MuJoCo.

The controller applies only ESN-predicted torque to the 12 non-thumb,
non-joint0 joints:

    ESN([q(t), commanded_q(t+0.1s)]) -> tau_inverse(t)

Use ``--model`` to switch candidates listed in ``all_models_manifest.csv``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from . import future_pose_torque_mujoco as core
except ImportError:  # Direct execution: python deployment/screened_....py
    import future_pose_torque_mujoco as core


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "L20/linker_hand_l20_right/linker_hand_l20_right.xml"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results/screened_reservoir_torque_esn_100ms"
    / "screened_reservoir_torque_esn_100ms_model.npz"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/movement_without_thumb_joint0_100hz"
EXPECTED_FUTURE_HORIZON_SECONDS = 0.1
EXPECTED_SAMPLE_RATE_HZ = 100.0


class ScreenedReservoirTorqueController(core.FuturePoseTorqueController):
    """Inference wrapper that enforces the screened 100 ms model contract."""

    CANDIDATE_FIELDS = {
        "candidate_id",
        "screening_status",
        "pareto_core",
        "reservoir_size",
        "seed",
        "regularization",
        "spectral_radius",
        "leak_rate",
        "input_scaling",
        "connectivity",
        "bias_scaling",
        "washout_frames",
    }

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        torque_limit: float | np.ndarray | None = 1.0,
        max_input_zscore: float | None = 20.0,
    ) -> None:
        super().__init__(model_path, torque_limit, max_input_zscore)
        missing = self.CANDIDATE_FIELDS.difference(self.params)
        if missing:
            raise ValueError(
                f"{self.path} is not a screened candidate model; missing: "
                f"{sorted(missing)}"
            )
        if not np.isclose(
            self.future_horizon, EXPECTED_FUTURE_HORIZON_SECONDS, atol=1e-12
        ):
            raise ValueError(
                "Expected a 0.1s future horizon, "
                f"got {self.future_horizon:g}s"
            )
        if not np.isclose(self.sample_rate_hz, EXPECTED_SAMPLE_RATE_HZ, atol=1e-12):
            raise ValueError(
                f"Expected a 100 Hz model, got {self.sample_rate_hz:g} Hz"
            )

    @property
    def candidate_id(self) -> int:
        return int(self.params["candidate_id"])

    @property
    def regularization(self) -> float:
        return float(self.params["regularization"])


def parse_args() -> argparse.Namespace:
    core.DEFAULT_MODEL = DEFAULT_MODEL
    core.DEFAULT_XML = DEFAULT_XML
    core.DEFAULT_DATA_DIR = DEFAULT_DATA_DIR
    return core.parse_args()


def run(args: argparse.Namespace) -> None:
    core.run(args, controller_class=ScreenedReservoirTorqueController)


def main() -> None:
    try:
        run(parse_args())
    except (
        FileNotFoundError,
        FloatingPointError,
        IndexError,
        ValueError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()


ReferenceTrajectory = core.ReferenceTrajectory
configure_position_actuators = core.configure_position_actuators
mujoco_addresses = core.mujoco_addresses

__all__ = [
    "DEFAULT_DATA_DIR",
    "DEFAULT_MODEL",
    "DEFAULT_XML",
    "ReferenceTrajectory",
    "ScreenedReservoirTorqueController",
    "configure_position_actuators",
    "mujoco_addresses",
]
