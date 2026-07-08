"""Runtime ESN torque controller for MuJoCo experiments.

Typical use:

    controller = ESNTorqueController(
        "results/esn_controller_no_tau_history_seed67_60hz.npz",
        torque_limit=[2.0, 2.0, 2.0],
    )

    # Call at 60 Hz, matching the training data.
    tau = controller.step(qpos3, qvel3, force_g, horizon=1)
    data.ctrl[actuator_ids] = tau

The controller keeps its own reservoir state. Do not call reset() during a
continuous rollout unless you intentionally want to restart the ESN memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class OnePoleLowpass:
    """Simple causal low-pass filter for online control."""

    cutoff_hz: float
    sample_hz: float
    state: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.cutoff_hz <= 0:
            raise ValueError("cutoff_hz must be positive")
        if self.sample_hz <= 0:
            raise ValueError("sample_hz must be positive")
        dt = 1.0 / self.sample_hz
        self.alpha = 1.0 - np.exp(-2.0 * np.pi * self.cutoff_hz * dt)

    def reset(self) -> None:
        """Clear the filter memory."""
        self.state = None

    def step(self, value: np.ndarray) -> np.ndarray:
        """Filter one scalar/vector sample."""
        value = np.asarray(value, dtype=float)
        if self.state is None:
            self.state = value.copy()
        else:
            self.state = self.state + self.alpha * (value - self.state)
        return self.state.copy()


class ESNTorqueController:
    """Online ESN controller exported by save_esn_controller_seed67.py."""

    def __init__(
        self,
        model_path: str | Path,
        torque_limit: float | Iterable[float] | None = None,
        torque_rate_limit: float | Iterable[float] | None = None,
        filter_cutoff_hz: float | None = None,
    ) -> None:
        model = np.load(Path(model_path), allow_pickle=False)

        self.Win = model["Win"]
        self.Wr = model["Wr"]
        self.Wb = model["Wb"]
        self.Wout = model["Wout"]
        self.mean = model["standardizer_mean"]
        self.std = model["standardizer_std"]
        self.horizons = tuple(int(k) for k in model["horizons"])
        self.joint_ids = tuple(int(joint) for joint in model["joint_ids"])
        self.fs_hz = float(model["fs_hz"])
        self.force_cutoff_hz = float(model["force_cutoff_hz"])
        self.dt = float(model["dt"])
        self.nr = int(model["Nr"])
        self.input_dimension = int(model["input_dimension"])

        cutoff = self.force_cutoff_hz if filter_cutoff_hz is None else float(filter_cutoff_hz)
        self.qvel_filter = OnePoleLowpass(cutoff_hz=cutoff, sample_hz=self.fs_hz)
        self.force_filter = OnePoleLowpass(cutoff_hz=cutoff, sample_hz=self.fs_hz)
        self.torque_limit = self._as_joint_vector(torque_limit, "torque_limit")
        self.torque_rate_limit = self._as_joint_vector(torque_rate_limit, "torque_rate_limit")
        self.x = np.zeros((self.nr, 1), dtype=float)
        self.last_tau: np.ndarray | None = None
        self.step_count = 0

        if self.input_dimension != 7:
            raise ValueError(f"Expected input_dimension=7, got {self.input_dimension}")
        if self.Wout.shape[1] != 1 + self.input_dimension + self.nr:
            raise ValueError("Wout shape does not match [bias, input, reservoir] dimension")

    def _as_joint_vector(
        self,
        value: float | Iterable[float] | None,
        name: str,
    ) -> np.ndarray | None:
        if value is None:
            return None

        arr = np.asarray(value, dtype=float)
        if arr.ndim == 0:
            return np.full(3, float(arr), dtype=float)
        if arr.shape != (3,):
            raise ValueError(f"{name} must be a scalar or a length-3 vector")
        return arr

    def reset(self) -> None:
        """Reset reservoir, filters, and output-rate limiter state."""
        self.x = np.zeros((self.nr, 1), dtype=float)
        self.qvel_filter.reset()
        self.force_filter.reset()
        self.last_tau = None
        self.step_count = 0

    def build_input(
        self,
        qpos: Iterable[float],
        qvel: Iterable[float],
        force_g: float,
    ) -> np.ndarray:
        """Build and standardize the 7-D ESN input vector."""
        qpos = np.asarray(qpos, dtype=float).reshape(-1)
        qvel = np.asarray(qvel, dtype=float).reshape(-1)
        if qpos.shape != (3,):
            raise ValueError("qpos must contain exactly 3 joint positions")
        if qvel.shape != (3,):
            raise ValueError("qvel must contain exactly 3 joint velocities")

        qvel_filtered = self.qvel_filter.step(qvel)
        force_filtered = float(self.force_filter.step(np.asarray(force_g, dtype=float)))
        raw = np.concatenate([qpos, qvel_filtered, [force_filtered]])
        return (raw - self.mean) / self.std

    def predict_all_horizons(
        self,
        qpos: Iterable[float],
        qvel: Iterable[float],
        force_g: float,
    ) -> np.ndarray:
        """Advance the reservoir by one control tick and return all 15 outputs."""
        u = self.build_input(qpos, qvel, force_g).reshape(self.input_dimension, 1)
        x_dot = -self.x + np.tanh(self.Win @ u + self.Wr @ self.x + self.Wb)
        self.x = self.x + x_dot * self.dt
        readout = np.vstack([np.ones((1, 1)), u, self.x])
        y = (self.Wout @ readout).reshape(-1)
        self.step_count += 1
        return y

    def step(
        self,
        qpos: Iterable[float],
        qvel: Iterable[float],
        force_g: float,
        horizon: int = 1,
    ) -> np.ndarray:
        """Return a 3-D torque command for the requested prediction horizon."""
        y = self.predict_all_horizons(qpos, qvel, force_g)
        tau = self.select_horizon(y, horizon)
        tau = self.apply_safety_limits(tau)
        self.last_tau = tau.copy()
        return tau

    def select_horizon(self, y: np.ndarray, horizon: int) -> np.ndarray:
        """Select joint torques for one horizon from the 15-D ESN output."""
        if horizon not in self.horizons:
            raise ValueError(f"horizon must be one of {self.horizons}")
        horizon_index = self.horizons.index(horizon)
        start = horizon_index * 3
        return y[start : start + 3].copy()

    def apply_safety_limits(self, tau: np.ndarray) -> np.ndarray:
        """Apply optional torque and torque-rate limits."""
        tau = np.asarray(tau, dtype=float)
        if not np.all(np.isfinite(tau)):
            raise FloatingPointError(f"ESN produced non-finite torque: {tau}")

        if self.torque_rate_limit is not None and self.last_tau is not None:
            max_delta = self.torque_rate_limit / self.fs_hz
            tau = self.last_tau + np.clip(tau - self.last_tau, -max_delta, max_delta)

        if self.torque_limit is not None:
            tau = np.clip(tau, -self.torque_limit, self.torque_limit)

        return tau


class ControlRateAdapter:
    """Call an ESNTorqueController at 60 Hz while MuJoCo may step faster."""

    def __init__(self, controller: ESNTorqueController, sim_dt: float, horizon: int = 1) -> None:
        if sim_dt <= 0:
            raise ValueError("sim_dt must be positive")
        self.controller = controller
        self.sim_dt = float(sim_dt)
        self.horizon = horizon
        self.update_period = 1.0 / controller.fs_hz
        self.elapsed = 0.0
        self.last_tau = np.zeros(3, dtype=float)

    def reset(self) -> None:
        """Reset timing state and the wrapped controller."""
        self.controller.reset()
        self.elapsed = 0.0
        self.last_tau = np.zeros(3, dtype=float)

    def step(self, qpos: Iterable[float], qvel: Iterable[float], force_g: float) -> np.ndarray:
        """Update torque only when the trained controller period has elapsed."""
        self.elapsed += self.sim_dt
        if self.elapsed >= self.update_period:
            self.elapsed -= self.update_period
            self.last_tau = self.controller.step(qpos, qvel, force_g, horizon=self.horizon)
        return self.last_tau.copy()
