"""Runtime current-torque ESN controller for MuJoCo experiments.

Typical use:

    controller = ESNTorqueController(
        "results/esn_controller_current_torque_seed52_60hz.npz",
        torque_limit=[2.0, 2.0, 2.0],
    )

    # Call at 60 Hz, matching the training data.
    tau = controller.step(qpos3, qvel3, force_g)
    data.ctrl[actuator_ids] = tau

The controller keeps its own reservoir state. Do not call reset() during a
continuous rollout unless you intentionally want to restart the ESN memory.
"""

from __future__ import annotations

import collections
import threading
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
    """Online ESN controller exported by save_esn_controller_current_torque.py."""

    def __init__(
        self,
        model_path: str | Path,
        torque_limit: float | Iterable[float] | None = None,
        torque_rate_limit: float | Iterable[float] | None = None,
        filter_cutoff_hz: float | None = None,
        recurrence_k: float = 0.0,
        extreme_zero_n: int | None = None,
        input_weight_scale: float = 1.0,
        force_gate_enabled: bool = True,
        force_gate_threshold: float = 5.0,
        force_gate_hysteresis: float = 0.5,
        leak_alpha: float | None = None,
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
        self.target_torque_offset = int(model["target_torque_offset"])
        self.dt = float(model["dt"])
        # alpha=dt preserves the previous Euler update by default.
        self.set_leak_alpha(self.dt if leak_alpha is None else leak_alpha)
        self.nr = int(model["Nr"])
        self.input_dimension = int(model["input_dimension"])
        self.output_dimension = int(model["output_dimension"])

        full_shape = (3, 1 + self.input_dimension + self.nr)
        reservoir_shape = (3, 1 + self.nr)
        if self.Wout.shape == full_shape:
            inferred_layout = "bias_input_reservoir"
            inferred_includes_input = True
        elif self.Wout.shape == reservoir_shape:
            inferred_layout = "bias_reservoir"
            inferred_includes_input = False
        else:
            raise ValueError(
                f"Unsupported Wout shape {self.Wout.shape}; expected "
                f"{full_shape} or {reservoir_shape}"
            )
        metadata_layout = (
            str(model["readout_layout"].item())
            if "readout_layout" in model.files
            else inferred_layout
        )
        metadata_includes_input = (
            bool(model["readout_includes_input"].item())
            if "readout_includes_input" in model.files
            else inferred_includes_input
        )
        if metadata_layout != inferred_layout:
            raise ValueError(
                f"readout_layout={metadata_layout!r} conflicts with "
                f"Wout shape {self.Wout.shape}"
            )
        if metadata_includes_input != inferred_includes_input:
            raise ValueError(
                "readout_includes_input metadata conflicts with Wout shape"
            )
        self.readout_layout = inferred_layout
        self.readout_includes_input = inferred_includes_input
        model_extreme_n = (
            int(model["extreme_n"].item()) if "extreme_n" in model.files else 0
        )

        cutoff = self.force_cutoff_hz if filter_cutoff_hz is None else float(filter_cutoff_hz)
        self.qvel_filter = OnePoleLowpass(cutoff_hz=cutoff, sample_hz=self.fs_hz)
        self.force_filter = OnePoleLowpass(cutoff_hz=cutoff, sample_hz=self.fs_hz)
        self.torque_limit = self._as_joint_vector(torque_limit, "torque_limit")
        self.torque_rate_limit = self._as_joint_vector(torque_rate_limit, "torque_rate_limit")
        self.set_recurrence_k(recurrence_k)
        self.set_input_weight_scale(input_weight_scale)
        self.set_extreme_zero_n(
            model_extreme_n if extreme_zero_n is None else extreme_zero_n
        )
        self.set_force_gate_threshold(
            force_gate_threshold, hysteresis=force_gate_hysteresis
        )
        self.force_gate_active = False
        self.set_force_gate_enabled(force_gate_enabled)
        self.x = np.zeros((self.nr, 1), dtype=float)
        self.last_tau: np.ndarray | None = None
        self.step_count = 0
        self._debug_lock = threading.Lock()
        self._debug_enabled = False
        self._debug_snapshots = collections.deque(maxlen=2000)
        self._pending_debug_snapshot = None

        if self.input_dimension != 7:
            raise ValueError(f"Expected input_dimension=7, got {self.input_dimension}")
        if self.output_dimension != 3:
            raise ValueError(f"Expected output_dimension=3, got {self.output_dimension}")
        if self.horizons != (0,):
            raise ValueError(f"Expected horizons=(0,), got {self.horizons}")
        if self.target_torque_offset != 0:
            raise ValueError(
                f"Expected target_torque_offset=0, got {self.target_torque_offset}"
            )

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

    def set_recurrence_k(self, value: float) -> None:
        """Set the strength of the arctanh reservoir correction."""
        value = float(value)
        if not np.isfinite(value):
            raise ValueError("recurrence_k must be finite")
        self.recurrence_k = value

    def set_leak_alpha(self, value: float) -> None:
        """Set the leaky-integrator mixing factor in [0, 1]."""
        value = float(value)
        if not np.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("leak_alpha must be finite and in [0, 1]")
        self.leak_alpha = value

    def set_input_weight_scale(self, value: float) -> None:
        """Scale the Win contribution without modifying the saved weights."""
        value = float(value)
        if not np.isfinite(value) or value < 0.0:
            raise ValueError("input_weight_scale must be finite and non-negative")
        self.input_weight_scale = value

    def set_extreme_zero_n(self, value: int) -> None:
        """Zero the n largest and n smallest reservoir values each step."""
        value = int(value)
        maximum = self.nr // 2
        if value < 0 or value > maximum:
            raise ValueError(f"extreme_zero_n must be in [0, {maximum}]")
        self.extreme_zero_n = value
        if hasattr(self, "force_gate_enabled"):
            self.effective_extreme_zero_n = (
                value
                if not self.force_gate_enabled or self.force_gate_active
                else 0
            )

    def set_force_gate_enabled(self, enabled: bool) -> None:
        """Gate extreme zeroing with the raw contact force."""
        self.force_gate_enabled = bool(enabled)
        if not self.force_gate_enabled:
            self.force_gate_active = False
        self.effective_extreme_zero_n = (
            self.extreme_zero_n
            if not self.force_gate_enabled or self.force_gate_active
            else 0
        )

    def set_force_gate_threshold(
        self, threshold: float, hysteresis: float = 0.5
    ) -> None:
        """Set the n=1 turn-on force and lower turn-off force in newtons."""
        threshold = float(threshold)
        hysteresis = float(hysteresis)
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("force_gate_threshold must be finite and non-negative")
        if not np.isfinite(hysteresis) or hysteresis < 0.0:
            raise ValueError("force_gate_hysteresis must be finite and non-negative")
        self.force_gate_threshold = threshold
        self.force_gate_hysteresis = hysteresis
        self.force_gate_off_threshold = max(0.0, threshold - hysteresis)

    def _update_force_gate(self, force_value: float) -> int:
        if self.force_gate_enabled:
            if (
                not self.force_gate_active
                and force_value >= self.force_gate_threshold
            ):
                self.force_gate_active = True
            elif (
                self.force_gate_active
                and force_value <= self.force_gate_off_threshold
            ):
                self.force_gate_active = False
        self.effective_extreme_zero_n = (
            self.extreme_zero_n
            if not self.force_gate_enabled or self.force_gate_active
            else 0
        )
        return self.effective_extreme_zero_n

    def set_debug_enabled(self, enabled: bool) -> None:
        """Enable bounded recurrence snapshots, clearing prior debug data."""
        with self._debug_lock:
            self._debug_enabled = bool(enabled)
            self._debug_snapshots.clear()
            self._pending_debug_snapshot = None

    def drain_debug_snapshots(self) -> list:
        """Return and remove all captured recurrence snapshots."""
        with self._debug_lock:
            snapshots = list(self._debug_snapshots)
            self._debug_snapshots.clear()
        return snapshots

    def debug_snapshot_count(self) -> int:
        with self._debug_lock:
            return len(self._debug_snapshots)

    def reset(self) -> None:
        """Reset reservoir, filters, and output-rate limiter state."""
        self.x = np.zeros((self.nr, 1), dtype=float)
        self.qvel_filter.reset()
        self.force_filter.reset()
        self.last_tau = None
        self.step_count = 0
        self.force_gate_active = False
        self.effective_extreme_zero_n = (
            0 if self.force_gate_enabled else self.extreme_zero_n
        )
        with self._debug_lock:
            self._debug_snapshots.clear()
            self._pending_debug_snapshot = None

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

    def predict(
        self,
        qpos: Iterable[float],
        qvel: Iterable[float],
        force_g: float,
    ) -> np.ndarray:
        """Advance the reservoir by one control tick and return 3-D torque."""
        force_value = float(np.asarray(force_g, dtype=float))
        if not np.isfinite(force_value):
            raise ValueError("force_g must be finite")
        effective_extreme_zero_n = self._update_force_gate(force_value)
        u = self.build_input(qpos, qvel, force_g).reshape(self.input_dimension, 1)
        x_old = self.x.copy()
        input_term = self.input_weight_scale * (self.Win @ u)
        recurrent_term = self.Wr @ x_old
        bias_term = self.Wb
        pre_activation = input_term + recurrent_term + bias_term
        tanh_value = np.tanh(pre_activation)
        x_dot = -x_old + tanh_value
        x_base_next = x_old + self.leak_alpha * x_dot
        x_safe = np.clip(x_old, -1.0 + 1e-7, 1.0 - 1e-7)
        raw_correction = np.arctanh(x_safe) - x_safe
        scaled_correction = self.recurrence_k * raw_correction
        x_before_zeroing = x_base_next - scaled_correction
        self.x = x_before_zeroing.copy()
        zeroed_indices = np.empty(0, dtype=int)
        extreme_zero_n = int(effective_extreme_zero_n)
        if extreme_zero_n > 0:
            values = self.x.reshape(-1)
            order = np.argsort(values, kind="stable")
            zeroed_indices = np.concatenate(
                [order[:extreme_zero_n], order[-extreme_zero_n:]]
            )
            values[zeroed_indices] = 0.0
        if not np.all(np.isfinite(self.x)):
            raise FloatingPointError("ESN produced a non-finite reservoir state")
        if self.readout_includes_input:
            readout = np.vstack([np.ones((1, 1)), u, self.x])
        else:
            readout = np.vstack([np.ones((1, 1)), self.x])
        y = (self.Wout @ readout).reshape(-1)
        self.step_count += 1
        with self._debug_lock:
            if self._debug_enabled:
                self._pending_debug_snapshot = {
                    "step_count": int(self.step_count),
                    "u": u.copy(),
                    "x_old": x_old.copy(),
                    "input_term": input_term.copy(),
                    "recurrent_term": recurrent_term.copy(),
                    "bias_term": bias_term.copy(),
                    "pre_activation": pre_activation.copy(),
                    "tanh_value": tanh_value.copy(),
                    "x_dot": x_dot.copy(),
                    "x_base_next": x_base_next.copy(),
                    "raw_correction": raw_correction.copy(),
                    "scaled_correction": scaled_correction.copy(),
                    "x_before_zeroing": x_before_zeroing.copy(),
                    "zeroed_indices": zeroed_indices.copy(),
                    "extreme_zero_n": extreme_zero_n,
                    "configured_extreme_zero_n": int(self.extreme_zero_n),
                    "effective_extreme_zero_n": extreme_zero_n,
                    "force_raw": force_value,
                    "force_gate_enabled": bool(self.force_gate_enabled),
                    "force_gate_active": bool(self.force_gate_active),
                    "force_gate_threshold": float(self.force_gate_threshold),
                    "force_gate_off_threshold": float(
                        self.force_gate_off_threshold
                    ),
                    "x_next": self.x.copy(),
                    "k": float(self.recurrence_k),
                    "input_weight_scale": float(self.input_weight_scale),
                    "leak_alpha": float(self.leak_alpha),
                }
            else:
                self._pending_debug_snapshot = None
        return y.copy()

    def step(
        self,
        qpos: Iterable[float],
        qvel: Iterable[float],
        force_g: float,
    ) -> np.ndarray:
        """Return a 3-D torque command for the current frame."""
        tau = self.predict(qpos, qvel, force_g)
        tau = self.apply_safety_limits(tau)
        self.last_tau = tau.copy()
        with self._debug_lock:
            if self._debug_enabled and self._pending_debug_snapshot is not None:
                snapshot = self._pending_debug_snapshot
                snapshot["tau"] = tau.copy()
                self._debug_snapshots.append(snapshot)
            self._pending_debug_snapshot = None
        return tau

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

    def __init__(self, controller: ESNTorqueController, sim_dt: float) -> None:
        if sim_dt <= 0:
            raise ValueError("sim_dt must be positive")
        self.controller = controller
        self.sim_dt = float(sim_dt)
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
            self.last_tau = self.controller.step(qpos, qvel, force_g)
        return self.last_tau.copy()
