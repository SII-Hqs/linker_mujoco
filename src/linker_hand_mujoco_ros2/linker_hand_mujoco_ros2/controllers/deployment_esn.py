"""Runtime inference for combined and separately trained open/grasp ESNs."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi


def _joint_name(column, prefix):
    if not column.startswith(prefix):
        raise ValueError(f"unexpected model column {column!r}; expected {prefix!r}")
    return column.removeprefix(prefix)


class VectorLowpass:
    """Stateful Butterworth filter for vector samples."""

    def __init__(self, channels, sample_rate_hz, cutoff_hz, order):
        self.sos = butter(order, cutoff_hz, fs=sample_rate_hz, output="sos")
        self.zi_template = sosfilt_zi(self.sos)
        self.channels = int(channels)
        self.state = None

    def reset(self):
        self.state = None

    def apply(self, values):
        values = np.asarray(values, dtype=float)
        if self.state is None:
            self.state = np.repeat(
                self.zi_template[:, :, None], self.channels, axis=2
            ) * values[None, None, :]
        filtered, self.state = sosfilt(
            self.sos, values.reshape(1, -1), axis=0, zi=self.state
        )
        return filtered[0]


class ESNTorqueModel:
    """One exported ESN with optional open/grasp action selection."""

    def __init__(self, model_path, torque_limit):
        self.model_path = Path(model_path)
        with np.load(self.model_path) as archive:
            self.params = {key: np.asarray(archive[key]) for key in archive.files}

        self.Win = self.params["Win"].astype(float)
        self.Wb = self.params["Wb"].astype(float)
        self.Wr = self.params["Wr"].astype(float)
        self.Wout = self.params["Wout"].astype(float)
        self.dt = float(self.params["dt"])
        self.target_mean = self.params["target_mean"].astype(float)
        self.target_std = self.params["target_std"].astype(float)
        self.qpos_columns = [str(value) for value in self.params["qpos_cols"]]
        self.qvel_columns = [str(value) for value in self.params["qvel_cols"]]
        self.joint_names = [
            _joint_name(value, "qpos_") for value in self.qpos_columns
        ]
        self.state_names = [str(value) for value in self.params["state_names"]]
        self.state_vectors = self.params["state_one_hot"].astype(float)
        self.trained_action = (
            str(self.params["state"].item()) if "state" in self.params else None
        )
        self.torque_limit = float(torque_limit)
        self.state = np.zeros((self.Win.shape[0], 1), dtype=float)
        self.action = self.trained_action or "open"

        if self.Win.shape[1] != 44 or len(self.joint_names) != 21:
            raise ValueError("expected a 44-input, 21-joint open/grasp ESN model")
        if self.trained_action is not None and self.trained_action not in self.state_names:
            raise ValueError(f"invalid separately trained action: {self.trained_action}")

        use_filter = bool(self.params.get("use_lowpass_filter", False))
        self.qvel_filter = None
        self.torque_filter = None
        if use_filter:
            sample_rate = float(self.params["filter_sample_rate_hz"])
            cutoff = float(self.params["filter_cutoff_hz"])
            order = int(self.params["filter_order"])
            if bool(self.params.get("filter_qvel", False)):
                self.qvel_filter = VectorLowpass(21, sample_rate, cutoff, order)
            if bool(self.params.get("filter_tau", False)):
                self.torque_filter = VectorLowpass(21, sample_rate, cutoff, order)

    @property
    def is_separate(self):
        return self.trained_action is not None

    def set_action(self, action):
        if action not in self.state_names:
            raise ValueError(f"unknown ESN action {action!r}")
        if self.is_separate and action != self.trained_action:
            raise ValueError(
                f"model was trained for {self.trained_action!r}, not {action!r}"
            )
        self.action = action

    def reset(self):
        self.state.fill(0.0)
        if self.qvel_filter is not None:
            self.qvel_filter.reset()
        if self.torque_filter is not None:
            self.torque_filter.reset()

    def _normalize(self, qpos, qvel):
        if self.qvel_filter is not None:
            qvel = self.qvel_filter.apply(qvel)
        continuous = np.concatenate([qpos, qvel]).astype(float)
        if "qpos_lower" in self.params and "qvel_mean" in self.params:
            lower = self.params["qpos_lower"].astype(float)
            upper = self.params["qpos_upper"].astype(float)
            qpos = np.clip(qpos, lower, upper)
            scaled_qpos = 2.0 * (qpos - lower) / (upper - lower) - 1.0
            scaled_qvel = (
                qvel - self.params["qvel_mean"]
            ) / self.params["qvel_std"]
            continuous = np.concatenate([scaled_qpos, scaled_qvel])
        else:
            continuous = (
                continuous - self.params["continuous_input_mean"]
            ) / self.params["continuous_input_std"]
        one_hot = self.state_vectors[self.state_names.index(self.action)]
        return np.concatenate([continuous, one_hot])

    def predict(self, qpos, qvel):
        inputs = self._normalize(qpos, qvel).reshape(-1, 1)
        derivative = -self.state + np.tanh(
            self.Win @ inputs + self.Wr @ self.state + self.Wb
        )
        self.state += derivative * self.dt
        features = np.vstack([np.ones((1, 1)), inputs, self.state])
        scaled_torque = (self.Wout @ features).ravel()
        torque = scaled_torque * self.target_std + self.target_mean
        if self.torque_filter is not None:
            torque = self.torque_filter.apply(torque)
        if not np.all(np.isfinite(torque)):
            raise FloatingPointError("ESN produced non-finite torque")
        return np.clip(torque, -self.torque_limit, self.torque_limit)


class ESNTorqueController:
    """Select a combined model or action-specific separately trained models."""

    def __init__(self, variant, torque_limit, combined_path=None, open_path=None, grasp_path=None):
        if variant not in ("combined", "separate"):
            raise ValueError("ESN model variant must be 'combined' or 'separate'")
        self.variant = variant
        if variant == "combined":
            model = ESNTorqueModel(combined_path, torque_limit)
            if model.is_separate:
                raise ValueError("combined ESN path points to a separately trained model")
            self.models = {"open": model, "grasp": model}
        else:
            open_model = ESNTorqueModel(open_path, torque_limit)
            grasp_model = ESNTorqueModel(grasp_path, torque_limit)
            if open_model.trained_action != "open":
                raise ValueError("open ESN path is not an open-only model")
            if grasp_model.trained_action != "grasp":
                raise ValueError("grasp ESN path is not a grasp-only model")
            self.models = {"open": open_model, "grasp": grasp_model}
        self.action = "open"
        self.active_model.set_action(self.action)

    @property
    def active_model(self):
        return self.models[self.action]

    @property
    def joint_names(self):
        return self.active_model.joint_names

    def set_action(self, action, reset=True):
        if action not in self.models:
            raise ValueError(f"unknown ESN action {action!r}")
        changed = action != self.action
        self.action = action
        self.active_model.set_action(action)
        if reset and changed:
            self.active_model.reset()

    def reset(self):
        for model in set(self.models.values()):
            model.reset()

    def predict(self, qpos, qvel):
        return self.active_model.predict(qpos, qvel)


class CascadedESNTorqueController:
    """Adapter for the goal-conditioned trajectory and acceleration-torque ESNs."""

    variant = "cascaded"
    requires_qacc = True

    def __init__(
        self,
        controller_path,
        goal_model_path,
        torque_model_path,
        torque_limit,
        target_qpos=None,
    ):
        controller_path = Path(controller_path).resolve()
        spec = importlib.util.spec_from_file_location(
            "linker_hand_cascaded_grasp_esn", controller_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load cascaded ESN controller: {controller_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(spec.name, None)
            raise

        self.controller = module.CascadedGraspESNController(
            esn1_model=Path(goal_model_path),
            esn2_model=Path(torque_model_path),
            torque_limit=float(torque_limit),
        )
        self.joint_names = [
            _joint_name(column, "qpos_")
            for column in self.controller.qpos_columns
        ]
        if target_qpos is None:
            target_qpos = self.controller.esn1.input_mean[len(self.joint_names):]
        self.target_qpos = np.asarray(target_qpos, dtype=float)
        if self.target_qpos.shape != (len(self.joint_names),):
            raise ValueError("cascaded grasp target must contain 21 joint positions")
        if not np.all(np.isfinite(self.target_qpos)):
            raise ValueError("cascaded grasp target contains non-finite values")
        self.action = "grasp"
        self.predicted_next_qpos = np.zeros_like(self.target_qpos)

    def set_action(self, action, reset=True):
        if action != "grasp":
            raise ValueError("cascaded ESN supports only the grasp action")
        self.action = action
        if reset:
            self.reset()

    def reset(self):
        self.controller.reset()
        self.predicted_next_qpos.fill(0.0)

    def predict(self, qpos, qvel, qacc=None):
        del qvel
        output = self.controller.predict(qpos, self.target_qpos, qacc)
        self.predicted_next_qpos[:] = output.predicted_next_qpos
        return output.current_torque
