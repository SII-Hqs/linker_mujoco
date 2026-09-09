#!/usr/bin/env python3
"""Cascade the grasp trajectory ESN (ESN1) and torque ESN (ESN2).

External input:
    current_qpos, target_qpos, current_qacc (for ESN2.2)

Pipeline:
    ESN1([current_qpos, target_qpos]) -> predicted_next_qpos
    ESN2.2([current_qpos, predicted_next_qpos, current_qacc]) -> current_torque

ESN2 uses the feature order from training even though both components are
conceptually its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ESN1_MODEL = (
    PROJECT_ROOT
    / "results/grasp_goal_conditioned_esn/grasp_goal_conditioned_esn_model.npz"
)
DEFAULT_ESN2_MODEL = (
    PROJECT_ROOT
    / "results/grasp_transition_torque_esn/grasp_transition_torque_esn_model.npz"
)
DEFAULT_ESN21_MODEL = (
    PROJECT_ROOT
    / "results/grasp_predicted_transition_torque_esn/"
    "grasp_predicted_transition_torque_esn_model.npz"
)
DEFAULT_ESN21_TUNED_MODEL = (
    PROJECT_ROOT
    / "results/esn21_hyperparameter_experiments/"
    "grasp_predicted_transition_torque_esn_tuned_model.npz"
)
DEFAULT_ESN22_MODEL = (
    PROJECT_ROOT
    / "results/grasp_acceleration_torque_esn/"
    "grasp_acceleration_torque_esn_model.npz"
)
N_JOINTS = 21


def _strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in values]


class _LoadedESN:
    """Stateful inference for one exported ESN."""

    def __init__(self, path: Path):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ESN model does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            self.params = {key: np.asarray(archive[key]) for key in archive.files}

        required = {
            "Win",
            "Wb",
            "Wr",
            "Wout",
            "dt",
            "input_mean",
            "input_std",
            "target_mean",
            "target_std",
            "input_dim",
        }
        missing = required.difference(self.params)
        if missing:
            raise ValueError(f"{path} is missing model fields: {sorted(missing)}")

        self.path = path
        self.Win = self.params["Win"].astype(float)
        self.Wb = self.params["Wb"].astype(float)
        self.Wr = self.params["Wr"].astype(float)
        self.Wout = self.params["Wout"].astype(float)
        self.dt = float(self.params["dt"])
        self.input_mean = self.params["input_mean"].astype(float)
        self.input_std = self.params["input_std"].astype(float)
        self.target_mean = self.params["target_mean"].astype(float)
        self.target_std = self.params["target_std"].astype(float)
        self.input_dim = int(self.params["input_dim"])
        self.state = np.zeros((self.Win.shape[0], 1), dtype=float)
        self._validate_shapes()

    def _validate_shapes(self) -> None:
        reservoir_size = self.Win.shape[0]
        output_dim = len(self.target_mean)
        expected = {
            "Win": (reservoir_size, self.input_dim),
            "Wb": (reservoir_size, 1),
            "Wr": (reservoir_size, reservoir_size),
            "Wout": (output_dim, 1 + self.input_dim + reservoir_size),
            "input_mean": (self.input_dim,),
            "input_std": (self.input_dim,),
            "target_std": (output_dim,),
        }
        actual = {
            "Win": self.Win.shape,
            "Wb": self.Wb.shape,
            "Wr": self.Wr.shape,
            "Wout": self.Wout.shape,
            "input_mean": self.input_mean.shape,
            "input_std": self.input_std.shape,
            "target_std": self.target_std.shape,
        }
        errors = [
            f"{name}: expected {shape}, got {actual[name]}"
            for name, shape in expected.items()
            if actual[name] != shape
        ]
        if errors:
            raise ValueError(f"Invalid ESN model {self.path}: " + "; ".join(errors))
        if np.any(self.input_std <= 0.0) or np.any(self.target_std <= 0.0):
            raise ValueError(f"Model {self.path} contains non-positive scaling values")

    def reset(self) -> None:
        self.state.fill(0.0)

    def predict(self, raw_input: np.ndarray) -> np.ndarray:
        values = np.asarray(raw_input, dtype=float)
        if values.shape != (self.input_dim,):
            raise ValueError(
                f"Expected {self.input_dim} ESN inputs, got shape {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("ESN input contains non-finite values")

        normalized = ((values - self.input_mean) / self.input_std).reshape(-1, 1)
        derivative = -self.state + np.tanh(
            self.Win @ normalized + self.Wr @ self.state + self.Wb
        )
        self.state += derivative * self.dt
        readout = np.vstack([np.ones((1, 1)), normalized, self.state])
        scaled_output = (self.Wout @ readout).ravel()
        output = scaled_output * self.target_std + self.target_mean
        if not np.all(np.isfinite(output)):
            raise FloatingPointError(f"ESN model produced non-finite output: {self.path}")
        return output


@dataclass(frozen=True)
class ControllerOutput:
    predicted_next_qpos: np.ndarray
    current_torque: np.ndarray


class CascadedGraspESNController:
    """Stateful ESN1 -> ESN2 grasp controller."""

    def __init__(
        self,
        esn1_model: Path = DEFAULT_ESN1_MODEL,
        esn2_model: Path = DEFAULT_ESN22_MODEL,
        torque_limit: float | np.ndarray | None = None,
    ):
        self.esn1 = _LoadedESN(Path(esn1_model))
        self.esn2 = _LoadedESN(Path(esn2_model))
        self.torque_limit = self._torque_limit(torque_limit)
        self._validate_contract()

    def _torque_limit(
        self, value: float | np.ndarray | None
    ) -> np.ndarray | None:
        if value is None:
            return None
        limits = np.asarray(value, dtype=float)
        if limits.ndim == 0:
            limits = np.full(N_JOINTS, float(limits))
        if limits.shape != (N_JOINTS,) or np.any(limits <= 0.0):
            raise ValueError("torque_limit must be positive and scalar or 21-dimensional")
        return limits

    def _validate_contract(self) -> None:
        if self.esn1.input_dim != 2 * N_JOINTS:
            raise ValueError("ESN1 must accept [current_qpos, target_qpos]")
        if self.esn2.input_dim not in (2 * N_JOINTS, 3 * N_JOINTS):
            raise ValueError(
                "ESN2 must accept [current_qpos, next_qpos] with optional qacc"
            )
        if len(self.esn1.target_mean) != N_JOINTS:
            raise ValueError("ESN1 must output 21 predicted joint positions")
        if len(self.esn2.target_mean) != N_JOINTS:
            raise ValueError("ESN2 must output 21 joint torques")

        esn1_layout = _strings(self.esn1.params.get("input_layout", np.array([])))
        esn2_layout = _strings(self.esn2.params.get("input_layout", np.array([])))
        if esn1_layout != ["current_qpos", "averaged_final_qpos"]:
            raise ValueError(f"Unexpected ESN1 input layout: {esn1_layout}")
        valid_esn2_layouts = (
            ["current_qpos", "next_qpos"],
            ["current_qpos", "esn1_predicted_next_qpos"],
            ["current_qpos", "esn1_predicted_next_qpos", "current_qacc"],
        )
        if esn2_layout not in valid_esn2_layouts:
            raise ValueError(f"Unexpected ESN2 input layout: {esn2_layout}")
        if str(self.esn2.params["target_layout"].item()) != "current_original_tau_inverse":
            raise ValueError("ESN2 does not predict tau_inverse(t)")

        self.qpos_columns = _strings(self.esn1.params["qpos_cols"])
        esn2_qpos_columns = _strings(self.esn2.params["qpos_cols"])
        if self.qpos_columns != esn2_qpos_columns:
            raise ValueError("ESN1 and ESN2 use different joint orders")
        self.tau_columns = _strings(self.esn2.params["tau_cols"])

    def reset(self) -> None:
        """Reset both reservoirs at the start of a new trajectory."""
        self.esn1.reset()
        self.esn2.reset()

    def predict(
        self,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
        current_qacc: np.ndarray | None = None,
    ) -> ControllerOutput:
        """Predict the next pose and current torque for one control step."""
        current = np.asarray(current_qpos, dtype=float)
        target = np.asarray(target_qpos, dtype=float)
        if current.shape != (N_JOINTS,) or target.shape != (N_JOINTS,):
            raise ValueError("current_qpos and target_qpos must each have shape (21,)")

        predicted_next = self.esn1.predict(np.concatenate([current, target]))
        # ESN2 was trained in this exact order: [qpos(t), qpos(t+1)].
        esn2_parts = [current, predicted_next]
        if self.esn2.input_dim == 3 * N_JOINTS:
            if current_qacc is None:
                raise ValueError("current_qacc is required by the ESN2.2 model")
            acceleration = np.asarray(current_qacc, dtype=float)
            if acceleration.shape != (N_JOINTS,):
                raise ValueError("current_qacc must have shape (21,)")
            esn2_parts.append(acceleration)
        elif current_qacc is not None:
            raise ValueError("current_qacc was provided to a model not trained with it")
        torque = self.esn2.predict(np.concatenate(esn2_parts))
        if self.torque_limit is not None:
            torque = np.clip(torque, -self.torque_limit, self.torque_limit)
        return ControllerOutput(predicted_next, torque)

    def compute_torque(
        self,
        current_qpos: np.ndarray,
        target_qpos: np.ndarray,
        current_qacc: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return only the torque command expected by a control loop."""
        return self.predict(current_qpos, target_qpos, current_qacc).current_torque


__all__ = [
    "CascadedGraspESNController",
    "ControllerOutput",
    "DEFAULT_ESN1_MODEL",
    "DEFAULT_ESN2_MODEL",
    "DEFAULT_ESN21_MODEL",
    "DEFAULT_ESN21_TUNED_MODEL",
    "DEFAULT_ESN22_MODEL",
]
