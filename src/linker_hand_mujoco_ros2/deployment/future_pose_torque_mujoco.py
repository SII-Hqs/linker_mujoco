#!/usr/bin/env python3
"""Deploy a future-pose torque ESN in the L20 right-hand MuJoCo model.

At every model-rate control step:

    ESN([q(t), commanded_q(t+horizon)]) -> tau_inverse(t)

The 12 modeled joints are torque controlled. Thumb joints and every joint0
remain on the XML position servos at their initialized positions.
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "L20/linker_hand_l20_right/linker_hand_l20_right.xml"
DEFAULT_MODEL = (
    PROJECT_ROOT
    / "results/future_pose_10ms_torque_esn/future_pose_10ms_torque_esn_model.npz"
)
DEFAULT_DATA_DIR = PROJECT_ROOT / "data/movement_without_thumb_joint0"
EXPECTED_JOINT_COUNT = 12


def _strings(values: np.ndarray) -> list[str]:
    return [str(value) for value in values]


def _joint_name(column: str, prefix: str) -> str:
    if not column.startswith(prefix):
        raise ValueError(f"Unexpected model column {column!r}; expected {prefix!r}")
    return column.removeprefix(prefix)


@dataclass(frozen=True)
class ControllerOutput:
    raw_torque: np.ndarray
    clipped_torque: np.ndarray
    max_abs_input_zscore: float


class FuturePoseTorqueController:
    """Stateful inference wrapper for the exported 12-joint ESN."""

    REQUIRED_FIELDS = {
        "Win",
        "Wb",
        "Wr",
        "Wout",
        "input_dimension",
        "output_dimension",
        "input_mean",
        "input_std",
        "target_mean",
        "target_std",
        "qpos_cols",
        "tau_cols",
        "input_layout",
        "target_layout",
        "sample_rate_hz",
        "future_horizon_seconds",
    }

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL,
        torque_limit: float | np.ndarray | None = 1.0,
        max_input_zscore: float | None = 20.0,
    ):
        path = Path(model_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"ESN model does not exist: {path}")
        with np.load(path, allow_pickle=False) as archive:
            self.params = {key: np.asarray(archive[key]) for key in archive.files}

        missing = self.REQUIRED_FIELDS.difference(self.params)
        if missing:
            raise ValueError(f"{path} is missing model fields: {sorted(missing)}")

        self.path = path
        self.Win = self.params["Win"].astype(float)
        self.Wb = self.params["Wb"].astype(float)
        self.Wr = self.params["Wr"].astype(float)
        self.Wout = self.params["Wout"].astype(float)
        leak_field = "leak_rate" if "leak_rate" in self.params else "dt"
        washout_field = (
            "washout_frames" if "washout_frames" in self.params else "washout"
        )
        if leak_field not in self.params or washout_field not in self.params:
            raise ValueError(
                f"{path} must contain leak_rate/dt and washout_frames/washout"
            )
        self.dt = float(self.params[leak_field])
        self.input_dimension = int(self.params["input_dimension"])
        self.output_dimension = int(self.params["output_dimension"])
        self.input_mean = self.params["input_mean"].astype(float)
        self.input_std = self.params["input_std"].astype(float)
        self.target_mean = self.params["target_mean"].astype(float)
        self.target_std = self.params["target_std"].astype(float)
        self.qpos_columns = _strings(self.params["qpos_cols"])
        self.tau_columns = _strings(self.params["tau_cols"])
        self.joint_names = [
            _joint_name(column, "qpos_") for column in self.qpos_columns
        ]
        self.sample_rate_hz = float(self.params["sample_rate_hz"])
        self.sample_period = 1.0 / self.sample_rate_hz
        self.future_horizon = float(self.params["future_horizon_seconds"])
        self.washout = int(self.params[washout_field])
        self.state = np.zeros((self.Win.shape[0], 1), dtype=float)
        self.activation = np.zeros_like(self.state)
        self.torque_limit = self._torque_limit(torque_limit)
        self.max_input_zscore = max_input_zscore
        self._validate_contract()

    def _torque_limit(
        self, value: float | np.ndarray | None
    ) -> np.ndarray | None:
        if value is None:
            return None
        limits = np.asarray(value, dtype=float)
        if limits.ndim == 0:
            limits = np.full(EXPECTED_JOINT_COUNT, float(limits))
        if limits.shape != (EXPECTED_JOINT_COUNT,) or np.any(limits <= 0.0):
            raise ValueError("torque_limit must be positive and scalar or 12-dimensional")
        return limits

    def _validate_contract(self) -> None:
        reservoir_size = self.Win.shape[0]
        expected_shapes = {
            "Win": (reservoir_size, self.input_dimension),
            "Wb": (reservoir_size, 1),
            "Wr": (reservoir_size, reservoir_size),
            "Wout": (
                self.output_dimension,
                1 + self.input_dimension + reservoir_size,
            ),
            "input_mean": (self.input_dimension,),
            "input_std": (self.input_dimension,),
            "target_mean": (self.output_dimension,),
            "target_std": (self.output_dimension,),
        }
        actual_shapes = {
            "Win": self.Win.shape,
            "Wb": self.Wb.shape,
            "Wr": self.Wr.shape,
            "Wout": self.Wout.shape,
            "input_mean": self.input_mean.shape,
            "input_std": self.input_std.shape,
            "target_mean": self.target_mean.shape,
            "target_std": self.target_std.shape,
        }
        errors = [
            f"{name}: expected {shape}, got {actual_shapes[name]}"
            for name, shape in expected_shapes.items()
            if actual_shapes[name] != shape
        ]
        if errors:
            raise ValueError(f"Invalid ESN model {self.path}: " + "; ".join(errors))
        if self.input_dimension != 2 * EXPECTED_JOINT_COUNT:
            raise ValueError("Model must accept 24 inputs: [q(t), q(t+horizon)]")
        if self.output_dimension != EXPECTED_JOINT_COUNT:
            raise ValueError("Model must output 12 joint torques")
        if len(self.qpos_columns) != EXPECTED_JOINT_COUNT:
            raise ValueError("Model must contain 12 qpos columns")
        expected_tau = [
            f"tau_inverse_{name}" for name in self.joint_names
        ]
        if self.tau_columns != expected_tau:
            raise ValueError("Model qpos and torque joint orders do not match")
        if any(
            name.startswith("thumb_") or name.endswith("joint0")
            for name in self.joint_names
        ):
            raise ValueError("Model unexpectedly contains thumb or joint0")
        expected_layout = [
            "qpos(t)",
            f"qpos(t+{self.future_horizon:g}s)",
        ]
        if _strings(self.params["input_layout"]) != expected_layout:
            raise ValueError(f"Unexpected input layout: {self.params['input_layout']}")
        if str(self.params["target_layout"].item()) != "tau_inverse(t)":
            raise ValueError("Model target is not tau_inverse(t)")
        if np.any(self.input_std <= 0.0) or np.any(self.target_std <= 0.0):
            raise ValueError("Model contains non-positive scaling values")
        if not np.isfinite(self.sample_rate_hz) or self.sample_rate_hz <= 0.0:
            raise ValueError("Model sample_rate_hz must be positive")
        if not np.isfinite(self.dt) or self.dt <= 0.0 or self.dt > 1.0:
            raise ValueError("Model leak rate must be in (0, 1]")
        horizon_frames = self.future_horizon * self.sample_rate_hz
        if self.future_horizon <= 0.0 or not np.isclose(
            horizon_frames, round(horizon_frames), atol=1e-9
        ):
            raise ValueError("Model future horizon must be a positive whole frame count")
        if self.washout < 0:
            raise ValueError("Model washout must be non-negative")
        if self.max_input_zscore is not None and self.max_input_zscore <= 0.0:
            raise ValueError("max_input_zscore must be positive or None")

    def reset(self) -> None:
        self.state.fill(0.0)
        self.activation.fill(0.0)

    def predict(
        self, current_qpos: np.ndarray, future_target_qpos: np.ndarray
    ) -> ControllerOutput:
        current = np.asarray(current_qpos, dtype=float)
        target = np.asarray(future_target_qpos, dtype=float)
        expected_shape = (EXPECTED_JOINT_COUNT,)
        if current.shape != expected_shape or target.shape != expected_shape:
            raise ValueError("current_qpos and future_target_qpos must have shape (12,)")
        values = np.concatenate([current, target])
        if not np.all(np.isfinite(values)):
            raise ValueError("ESN input contains non-finite values")

        normalized = (values - self.input_mean) / self.input_std
        max_abs_zscore = float(np.max(np.abs(normalized)))
        if (
            self.max_input_zscore is not None
            and max_abs_zscore > self.max_input_zscore
        ):
            raise FloatingPointError(
                "ESN input is outside the configured training-distribution guard: "
                f"max |z|={max_abs_zscore:.3f} > {self.max_input_zscore:.3f}"
            )

        u = normalized.reshape(-1, 1)
        self.activation = np.tanh(self.Win @ u + self.Wr @ self.state + self.Wb)
        derivative = -self.state + self.activation
        self.state += derivative * self.dt
        features = np.vstack([np.ones((1, 1)), u, self.state])
        scaled_torque = (self.Wout @ features).ravel()
        raw_torque = scaled_torque * self.target_std + self.target_mean
        if not np.all(np.isfinite(raw_torque)):
            raise FloatingPointError("ESN produced non-finite torque")
        clipped = raw_torque.copy()
        if self.torque_limit is not None:
            clipped = np.clip(clipped, -self.torque_limit, self.torque_limit)
        return ControllerOutput(raw_torque, clipped, max_abs_zscore)

    def compute_torque(
        self, current_qpos: np.ndarray, future_target_qpos: np.ndarray
    ) -> np.ndarray:
        return self.predict(current_qpos, future_target_qpos).clipped_torque

@dataclass(frozen=True)
class ReferenceTrajectory:
    path: Path
    time: np.ndarray
    qpos: np.ndarray
    qpos_columns: list[str]
    initial_row: pd.Series

    @classmethod
    def load(cls, path: Path, qpos_columns: list[str]) -> "ReferenceTrajectory":
        path = Path(path).resolve()
        frame = pd.read_csv(path)
        required = ["sim_time", *qpos_columns]
        missing = [column for column in required if column not in frame.columns]
        if missing:
            raise ValueError(f"Reference CSV is missing columns: {', '.join(missing)}")
        if frame[required].isna().any().any():
            raise ValueError(f"Reference CSV contains missing values: {path}")
        absolute_time = frame["sim_time"].to_numpy(dtype=float)
        if len(absolute_time) < 2 or np.any(np.diff(absolute_time) <= 0.0):
            raise ValueError("Reference sim_time must be strictly increasing")
        relative_time = absolute_time - absolute_time[0]
        qpos = frame[qpos_columns].to_numpy(dtype=float)
        return cls(
            path=path,
            time=relative_time,
            qpos=qpos,
            qpos_columns=qpos_columns,
            initial_row=frame.iloc[0],
        )

    @property
    def duration(self) -> float:
        return float(self.time[-1])

    def sample(self, query_time: float) -> np.ndarray:
        if not np.isfinite(query_time):
            raise ValueError("Reference query time must be finite")
        clipped_time = float(np.clip(query_time, self.time[0], self.time[-1]))
        return np.asarray(
            [
                np.interp(clipped_time, self.time, self.qpos[:, index])
                for index in range(self.qpos.shape[1])
            ]
        )

def default_reference_csv() -> Path:
    candidates = sorted(DEFAULT_DATA_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No reference trajectory in {DEFAULT_DATA_DIR}")
    return candidates[0]


def mujoco_addresses(mujoco, model, names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    joint_ids = []
    qpos_addresses = []
    dof_addresses = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model has no joint named {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Joint {name!r} is not a hinge joint")
        joint_ids.append(joint_id)
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    return (
        np.asarray(joint_ids, dtype=int),
        np.asarray(qpos_addresses, dtype=int),
        np.asarray(dof_addresses, dtype=int),
    )


def all_joint_names(mujoco, model) -> list[str]:
    names = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        if name is None:
            raise ValueError(f"MuJoCo joint {joint_id} has no name")
        names.append(name)
    return names


def initial_qpos_from_csv(
    path: Path, row_index: int, joint_names: list[str]
) -> np.ndarray:
    frame = pd.read_csv(path)
    columns = [f"qpos_{name}" for name in joint_names]
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Initial CSV is missing columns: {', '.join(missing)}")
    if row_index < 0 or row_index >= len(frame):
        raise IndexError(f"Initial CSV {path} has no row {row_index}")
    values = frame.iloc[row_index][columns].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Initial qpos contains non-finite values")
    return values


def configure_position_actuators(
    mujoco,
    model,
    data,
    controlled_joint_ids: np.ndarray,
    initial_qpos: np.ndarray,
) -> tuple[list[int], list[int]]:
    """Disable servos on ESN joints and hold all other joints at initialization."""
    controlled = set(int(value) for value in controlled_joint_ids)
    disabled = []
    holding = []
    for actuator_id in range(model.nu):
        if model.actuator_trntype[actuator_id] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        qpos_address = int(model.jnt_qposadr[joint_id])
        if joint_id in controlled:
            model.actuator_gainprm[actuator_id] = 0.0
            model.actuator_biasprm[actuator_id] = 0.0
            disabled.append(actuator_id)
        else:
            target = initial_qpos[qpos_address]
            if model.actuator_ctrllimited[actuator_id]:
                target = np.clip(
                    target,
                    model.actuator_ctrlrange[actuator_id, 0],
                    model.actuator_ctrlrange[actuator_id, 1],
                )
            data.ctrl[actuator_id] = target
            holding.append(actuator_id)
    return disabled, holding


def disable_hand_contacts(mujoco, model) -> None:
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name != "floor":
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0


def validate_targets_in_joint_ranges(
    values: np.ndarray,
    joint_ids: np.ndarray,
    model,
    label: str,
) -> None:
    lower = model.jnt_range[joint_ids, 0]
    upper = model.jnt_range[joint_ids, 1]
    below = np.min(values - lower, axis=0) < -1e-6 if values.ndim == 2 else values < lower - 1e-6
    above = np.max(values - upper, axis=0) > 1e-6 if values.ndim == 2 else values > upper + 1e-6
    if np.any(below) or np.any(above):
        names = [
            str(name)
            for name, invalid in zip(
                np.asarray([model.joint(int(joint_id)).name for joint_id in joint_ids]),
                np.logical_or(below, above),
                strict=True,
            )
            if invalid
        ]
        raise ValueError(f"{label} exceeds MuJoCo ranges for: {', '.join(names)}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    target_group = parser.add_mutually_exclusive_group()
    target_group.add_argument(
        "--reference-csv",
        type=Path,
        help="trajectory interpolated at the model's future horizon as the ESN target",
    )
    target_group.add_argument(
        "--target-qpos",
        type=float,
        nargs=EXPECTED_JOINT_COUNT,
        metavar="Q",
        help="fixed 12-value future target in model joint order",
    )
    parser.add_argument("--initial-csv", type=Path)
    parser.add_argument("--initial-row", type=int, default=0)
    parser.add_argument(
        "--duration",
        type=float,
        help="simulation duration; defaults to reference duration minus model horizon",
    )
    parser.add_argument("--torque-limit", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.1)
    parser.add_argument(
        "--reservoir-warmup-steps",
        type=int,
        help="ESN state warmup before simulation; defaults to model washout",
    )
    parser.add_argument("--max-joint-speed", type=float, default=30.0)
    parser.add_argument("--max-input-zscore", type=float, default=20.0)
    parser.add_argument(
        "--enable-contacts",
        action="store_true",
        help="enable mesh contacts from the XML",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--live-visualization",
        action="store_true",
        help="show joint trajectories and live reservoir dynamics",
    )
    parser.add_argument("--visualization-fps", type=float, default=15.0)
    parser.add_argument(
        "--visualization-window-seconds", type=float, default=2.5
    )
    parser.add_argument("--visualization-backend", default="QtAgg")
    parser.add_argument("--log", type=Path, help="optional simulation CSV log")
    args = parser.parse_args(argv)
    positive = (
        args.torque_limit,
        args.max_joint_speed,
        args.max_input_zscore,
    )
    if any(value <= 0.0 for value in positive):
        parser.error("torque-limit, max-joint-speed, and max-input-zscore must be positive")
    if args.initial_row < 0 or args.ramp_seconds < 0.0:
        parser.error("initial-row and ramp-seconds must be non-negative")
    if args.reservoir_warmup_steps is not None and args.reservoir_warmup_steps < 0:
        parser.error("reservoir-warmup-steps must be non-negative")
    if args.duration is not None and args.duration <= 0.0:
        parser.error("duration must be positive")
    if args.visualization_fps <= 0.0 or args.visualization_window_seconds <= 0.0:
        parser.error("visualization-fps and visualization-window-seconds must be positive")
    return args


def run(
    args: argparse.Namespace,
    controller_class: type[FuturePoseTorqueController] = FuturePoseTorqueController,
) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is not installed. Run: conda env update -n python_esn -f environment.yml"
        ) from exc

    controller = controller_class(
        args.model,
        torque_limit=args.torque_limit,
        max_input_zscore=args.max_input_zscore,
    )
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    model.opt.timestep = controller.sample_period
    data = mujoco.MjData(model)
    controlled_joint_ids, qpos_addresses, dof_addresses = mujoco_addresses(
        mujoco, model, controller.joint_names
    )
    joint_names = all_joint_names(mujoco, model)

    reference_path = args.reference_csv
    if reference_path is None and args.target_qpos is None:
        reference_path = default_reference_csv()
    reference = (
        ReferenceTrajectory.load(reference_path, controller.qpos_columns)
        if reference_path is not None
        else None
    )
    fixed_target = (
        np.asarray(args.target_qpos, dtype=float)
        if args.target_qpos is not None
        else None
    )
    if reference is not None:
        validate_targets_in_joint_ranges(
            reference.qpos, controlled_joint_ids, model, "Reference trajectory"
        )
    else:
        assert fixed_target is not None
        validate_targets_in_joint_ranges(
            fixed_target, controlled_joint_ids, model, "Fixed target"
        )

    initial_csv = Path(
        args.initial_csv
        or (reference.path if reference is not None else default_reference_csv())
    ).resolve()
    initial_qpos = initial_qpos_from_csv(
        initial_csv, args.initial_row, joint_names
    )
    all_joint_ids, all_qpos_addresses, all_dof_addresses = mujoco_addresses(
        mujoco, model, joint_names
    )
    clipped_initial = np.clip(
        initial_qpos,
        model.jnt_range[all_joint_ids, 0],
        model.jnt_range[all_joint_ids, 1],
    )
    data.qpos[all_qpos_addresses] = clipped_initial
    data.qvel[all_dof_addresses] = 0.0
    disabled_actuators, holding_actuators = configure_position_actuators(
        mujoco, model, data, controlled_joint_ids, data.qpos
    )
    if not args.enable_contacts:
        disable_hand_contacts(mujoco, model)
    mujoco.mj_forward(model, data)

    warmup_steps = (
        controller.washout
        if args.reservoir_warmup_steps is None
        else args.reservoir_warmup_steps
    )
    initial_current = data.qpos[qpos_addresses].copy()
    initial_future_target = (
        reference.sample(controller.future_horizon)
        if reference is not None
        else fixed_target
    )
    assert initial_future_target is not None
    controller.reset()
    for _ in range(warmup_steps):
        controller.predict(initial_current, initial_future_target)

    visualizer = None
    if args.live_visualization:
        try:
            from .reservoir_live_visualizer import LiveReservoirVisualizer
        except ImportError:
            from reservoir_live_visualizer import LiveReservoirVisualizer

        visualizer = LiveReservoirVisualizer(
            controller,
            reference,
            visualization_fps=args.visualization_fps,
            window_seconds=args.visualization_window_seconds,
            backend=args.visualization_backend,
        )

    if args.duration is not None:
        duration = args.duration
    elif reference is not None:
        duration = max(controller.sample_period, reference.duration - controller.future_horizon)
    else:
        duration = 5.0

    log_stream = None
    writer = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_stream = args.log.open("w", newline="")
        fields = [
            "time",
            "future_target_time",
            "ramp",
            "max_abs_input_zscore",
            "max_abs_tracking_error",
        ]
        fields += controller.qpos_columns
        fields += [f"target_{column}" for column in controller.qpos_columns]
        fields += [f"reference_{column}" for column in controller.qpos_columns]
        fields += [f"qvel_{name}" for name in controller.joint_names]
        fields += [f"raw_{column}" for column in controller.tau_columns]
        fields += [f"applied_{column}" for column in controller.tau_columns]
        writer = csv.DictWriter(log_stream, fieldnames=fields)
        writer.writeheader()

    applied_torque = np.zeros(EXPECTED_JOINT_COUNT)
    max_abs_qvel = 0.0
    max_abs_tracking_error = 0.0
    max_abs_raw_torque = 0.0
    max_abs_input_zscore = 0.0

    def physics_step() -> None:
        nonlocal applied_torque
        nonlocal max_abs_qvel, max_abs_tracking_error
        nonlocal max_abs_raw_torque, max_abs_input_zscore

        current_qpos = data.qpos[qpos_addresses].copy()
        current_qvel = data.qvel[dof_addresses].copy()
        future_time = data.time + controller.future_horizon
        if reference is not None:
            future_target = reference.sample(future_time)
            current_reference = reference.sample(data.time)
            tracking_error = current_qpos - current_reference
            step_tracking_error = float(np.max(np.abs(tracking_error)))
        else:
            assert fixed_target is not None
            future_target = fixed_target
            current_reference = np.full(EXPECTED_JOINT_COUNT, np.nan)
            step_tracking_error = float("nan")

        output = controller.predict(current_qpos, future_target)
        if visualizer is not None:
            visualizer.update(
                data.time,
                current_qpos,
                future_target,
                controller.state,
                controller.activation,
                output.max_abs_input_zscore,
                output.raw_torque,
            )
        ramp = min(1.0, data.time / args.ramp_seconds) if args.ramp_seconds else 1.0
        applied_torque = output.clipped_torque * ramp
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof_addresses] = applied_torque

        speed = float(np.max(np.abs(current_qvel)))
        max_abs_qvel = max(max_abs_qvel, speed)
        max_abs_raw_torque = max(
            max_abs_raw_torque, float(np.max(np.abs(output.raw_torque)))
        )
        max_abs_input_zscore = max(
            max_abs_input_zscore, output.max_abs_input_zscore
        )
        if np.isfinite(step_tracking_error):
            max_abs_tracking_error = max(
                max_abs_tracking_error, step_tracking_error
            )

        if writer:
            values = np.concatenate(
                [
                    current_qpos,
                    future_target,
                    current_reference,
                    current_qvel,
                    output.raw_torque,
                    applied_torque,
                ]
            )
            value_fields = (
                controller.qpos_columns
                + [f"target_{column}" for column in controller.qpos_columns]
                + [f"reference_{column}" for column in controller.qpos_columns]
                + [f"qvel_{name}" for name in controller.joint_names]
                + [f"raw_{column}" for column in controller.tau_columns]
                + [f"applied_{column}" for column in controller.tau_columns]
            )
            row = {
                "time": data.time,
                "future_target_time": future_time,
                "ramp": ramp,
                "max_abs_input_zscore": output.max_abs_input_zscore,
                "max_abs_tracking_error": step_tracking_error,
            }
            row.update(dict(zip(value_fields, values, strict=True)))
            writer.writerow(row)

        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError(f"Simulation became non-finite at t={data.time:.4f}s")
        post_step_speed = float(np.max(np.abs(data.qvel[dof_addresses])))
        if post_step_speed > args.max_joint_speed:
            raise FloatingPointError(
                f"joint speed safety limit exceeded at t={data.time:.4f}s: "
                f"{post_step_speed:.3f} > {args.max_joint_speed:.3f} rad/s"
            )

    source = str(reference.path) if reference is not None else "fixed target"
    print(f"Model: {controller.path}")
    print(f"XML: {args.xml.resolve()}")
    print(f"Target source: {source}")
    print(f"Initialization: {initial_csv} row {args.initial_row}")
    print(f"Controlled joints ({len(controller.joint_names)}): {controller.joint_names}")
    print(
        f"MuJoCo/model period: {controller.sample_period:g}s; "
        f"future horizon: {controller.future_horizon:g}s; duration: {duration:g}s; "
        f"reservoir warmup: {warmup_steps} steps"
    )
    print(
        f"Disabled position actuators: {len(disabled_actuators)}; "
        f"holding unmodeled actuators: {len(holding_actuators)}"
    )
    print(
        f"Torque limit: {args.torque_limit:g} N m; "
        f"contacts: {'enabled' if args.enable_contacts else 'disabled'}"
    )
    if visualizer is not None:
        print(
            f"Live reservoir visualization: {args.visualization_fps:g} FPS, "
            f"{args.visualization_window_seconds:g}s rolling window"
        )
    try:
        if args.headless:
            while data.time < duration:
                physics_step()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and data.time < duration:
                    started = time.perf_counter()
                    physics_step()
                    viewer.sync()
                    remaining = model.opt.timestep - (time.perf_counter() - started)
                    if remaining > 0.0:
                        time.sleep(remaining)
    finally:
        if visualizer is not None:
            visualizer.close()
        if log_stream:
            log_stream.close()

    tracking_text = (
        f"; max |tracking error|={max_abs_tracking_error:.4f} rad"
        if reference is not None
        else ""
    )
    print(
        f"Finished at t={data.time:.3f}s; max |qvel|={max_abs_qvel:.3f} rad/s; "
        f"max |raw torque|={max_abs_raw_torque:.3f} N m; "
        f"max |input z|={max_abs_input_zscore:.3f}{tracking_text}"
    )


def main(
    controller_class: type[FuturePoseTorqueController] = FuturePoseTorqueController,
) -> None:
    try:
        run(parse_args(), controller_class=controller_class)
    except (
        FileNotFoundError,
        FloatingPointError,
        IndexError,
        ValueError,
    ) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()


__all__ = [
    "ControllerOutput",
    "FuturePoseTorqueController",
    "ReferenceTrajectory",
]
