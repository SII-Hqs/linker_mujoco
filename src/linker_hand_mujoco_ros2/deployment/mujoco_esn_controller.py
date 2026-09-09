#!/usr/bin/env python3
"""Run a trained open/grasp ESN as a torque controller in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_XML = PROJECT_ROOT / "L20/linker_hand_l20_right/linker_hand_l20_right.xml"
DEFAULT_JOINT_MODEL = (
    PROJECT_ROOT / "results/open_grasp_onehot_esn/open_grasp_onehot_esn_model.npz"
)
DEFAULT_CONTROL_PERIOD = 0.018


def joint_name(column: str, prefix: str) -> str:
    if not column.startswith(prefix):
        raise ValueError(f"Unexpected model column {column!r}; expected prefix {prefix!r}")
    return column.removeprefix(prefix)


class ESNTorqueController:
    """Stateful one-step inference for an exported ESN model."""

    def __init__(self, model_path: Path, action: str, torque_limit: float):
        with np.load(model_path) as archive:
            self.params = {key: np.asarray(archive[key]) for key in archive.files}

        self.model_path = model_path
        self.action = action
        self.torque_limit = torque_limit
        self.Win = self.params["Win"]
        self.Wb = self.params["Wb"]
        self.Wr = self.params["Wr"]
        self.Wout = self.params["Wout"]
        self.dt = float(self.params["dt"])
        self.target_mean = self.params["target_mean"].astype(float)
        self.target_std = self.params["target_std"].astype(float)
        self.qpos_columns = [str(value) for value in self.params["qpos_cols"]]
        self.qvel_columns = [str(value) for value in self.params["qvel_cols"]]
        self.tau_columns = [str(value) for value in self.params["tau_cols"]]
        self.joint_names = [joint_name(value, "qpos_") for value in self.qpos_columns]
        self.state = np.zeros((self.Win.shape[0], 1), dtype=float)

        state_names = [str(value) for value in self.params["state_names"]]
        if action not in state_names:
            raise ValueError(f"Action {action!r} is not present in model: {state_names}")
        state_vectors = self.params["state_one_hot"].astype(float)
        self.one_hot = state_vectors[state_names.index(action)]

        if "state" in self.params and str(self.params["state"].item()) != action:
            trained_action = str(self.params["state"].item())
            raise ValueError(
                f"Separate model was trained for {trained_action!r}, not {action!r}"
            )
        if self.Win.shape[1] != 44 or len(self.joint_names) != 21:
            raise ValueError("Expected a 44-input, 21-joint open/grasp ESN model")

    def reset(self) -> None:
        self.state.fill(0.0)

    def normalize(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        continuous = np.concatenate([qpos, qvel]).astype(float)
        if "qpos_lower" in self.params and "qvel_mean" in self.params:
            lower = self.params["qpos_lower"].astype(float)
            upper = self.params["qpos_upper"].astype(float)
            clipped = np.clip(qpos, lower, upper)
            scaled_qpos = 2.0 * (clipped - lower) / (upper - lower) - 1.0
            scaled_qvel = (
                qvel - self.params["qvel_mean"]
            ) / self.params["qvel_std"]
            continuous = np.concatenate([scaled_qpos, scaled_qvel])
        else:
            continuous = (
                continuous - self.params["continuous_input_mean"]
            ) / self.params["continuous_input_std"]
        return np.concatenate([continuous, self.one_hot])

    def predict(self, qpos: np.ndarray, qvel: np.ndarray) -> np.ndarray:
        u = self.normalize(qpos, qvel).reshape(-1, 1)
        derivative = -self.state + np.tanh(
            self.Win @ u + self.Wr @ self.state + self.Wb
        )
        self.state += derivative * self.dt
        features = np.vstack([np.ones((1, 1)), u, self.state])
        scaled_torque = (self.Wout @ features).ravel()
        torque = scaled_torque * self.target_std + self.target_mean
        if not np.all(np.isfinite(torque)):
            raise FloatingPointError("ESN produced a non-finite torque")
        return np.clip(torque, -self.torque_limit, self.torque_limit)


def model_for_action(action: str, requested: Path | None, separate: bool) -> Path:
    if requested is not None:
        return requested
    if separate:
        return (
            PROJECT_ROOT
            / "results/open_grasp_separate_esn"
            / action
            / f"{action}_separate_esn_model.npz"
        )
    return DEFAULT_JOINT_MODEL


def default_initial_csv(action: str) -> Path:
    candidates = sorted((PROJECT_ROOT / "data" / action).glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No initialization trajectory in data/{action}")
    return candidates[0]


def read_initial_qpos(path: Path, columns: Iterable[str], row_index: int) -> np.ndarray:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        missing = [column for column in columns if column not in reader.fieldnames]
        if missing:
            raise ValueError(f"CSV is missing model columns: {', '.join(missing)}")
        for index, row in enumerate(reader):
            if index == row_index:
                return np.asarray([float(row[column]) for column in columns])
    raise IndexError(f"CSV {path} has no row {row_index}")


def mujoco_addresses(mujoco, model, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses = []
    dof_addresses = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise ValueError(f"MuJoCo model has no joint named {name!r}")
        if model.jnt_type[joint_id] != mujoco.mjtJoint.mjJNT_HINGE:
            raise ValueError(f"Joint {name!r} is not a hinge joint")
        qpos_addresses.append(model.jnt_qposadr[joint_id])
        dof_addresses.append(model.jnt_dofadr[joint_id])
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def disable_xml_actuators(model) -> None:
    """The bundled XML has position servos; ESN output is joint torque."""
    model.actuator_gainprm[:] = 0.0
    model.actuator_biasprm[:] = 0.0


def disable_hand_contacts(mujoco, model) -> None:
    """Disable unstable mesh self-collisions while retaining the floor geom."""
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if name != "floor":
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("open", "grasp"), required=True)
    parser.add_argument("--model", type=Path, help="Exported ESN .npz model")
    parser.add_argument(
        "--separate", action="store_true", help="Use the action-specific model"
    )
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--initial-csv", type=Path)
    parser.add_argument("--initial-row", type=int, default=0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--control-period", type=float, default=DEFAULT_CONTROL_PERIOD)
    parser.add_argument("--torque-limit", type=float, default=1.0)
    parser.add_argument("--ramp-seconds", type=float, default=0.25)
    parser.add_argument("--max-joint-speed", type=float, default=30.0)
    parser.add_argument(
        "--enable-contacts",
        action="store_true",
        help="Enable XML mesh contacts (may be unstable without collision exclusions)",
    )
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--log", type=Path, help="Optional simulation CSV output")
    args = parser.parse_args()
    if min(
        args.duration, args.control_period, args.torque_limit, args.max_joint_speed
    ) <= 0:
        parser.error(
            "duration, control-period, torque-limit, and max-joint-speed must be positive"
        )
    if args.initial_row < 0 or args.ramp_seconds < 0:
        parser.error("initial-row and ramp-seconds must be non-negative")
    return args


def run(args: argparse.Namespace) -> None:
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "MuJoCo is not installed. Run: conda env update -n python_esn -f environment.yml"
        ) from exc

    model_path = model_for_action(args.action, args.model, args.separate).resolve()
    controller = ESNTorqueController(model_path, args.action, args.torque_limit)
    model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
    data = mujoco.MjData(model)
    disable_xml_actuators(model)
    if not args.enable_contacts:
        disable_hand_contacts(mujoco, model)
    qpos_adr, dof_adr = mujoco_addresses(mujoco, model, controller.joint_names)

    initial_csv = (args.initial_csv or default_initial_csv(args.action)).resolve()
    initial_qpos = read_initial_qpos(
        initial_csv, controller.qpos_columns, args.initial_row
    )
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in controller.joint_names
    ]
    data.qpos[qpos_adr] = np.clip(
        initial_qpos, model.jnt_range[joint_ids, 0], model.jnt_range[joint_ids, 1]
    )
    data.qvel[dof_adr] = 0.0
    mujoco.mj_forward(model, data)

    log_stream = None
    writer = None
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        log_stream = args.log.open("w", newline="")
        fields = ["time", "action"] + controller.qpos_columns + controller.qvel_columns
        fields += controller.tau_columns
        writer = csv.DictWriter(log_stream, fieldnames=fields)
        writer.writeheader()

    torque = np.zeros(len(dof_adr))
    next_control_time = 0.0
    max_abs_qvel = 0.0

    def physics_step() -> None:
        nonlocal torque, next_control_time, max_abs_qvel
        if data.time + 1e-12 >= next_control_time:
            qpos = data.qpos[qpos_adr].copy()
            qvel = data.qvel[dof_adr].copy()
            torque = controller.predict(qpos, qvel)
            ramp = min(1.0, data.time / args.ramp_seconds) if args.ramp_seconds else 1.0
            torque *= ramp
            next_control_time += args.control_period
            if writer:
                values = np.concatenate([qpos, qvel, torque])
                row = {"time": data.time, "action": args.action}
                row.update(
                    dict(
                        zip(
                            controller.qpos_columns
                            + controller.qvel_columns
                            + controller.tau_columns,
                            values,
                        )
                    )
                )
                writer.writerow(row)
        data.qfrc_applied[:] = 0.0
        data.qfrc_applied[dof_adr] = torque
        mujoco.mj_step(model, data)
        max_abs_qvel = max(max_abs_qvel, float(np.max(np.abs(data.qvel[dof_adr]))))
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError(f"Simulation became non-finite at t={data.time:.4f}s")
        if max_abs_qvel > args.max_joint_speed:
            raise FloatingPointError(
                f"joint speed safety limit exceeded at t={data.time:.4f}s: "
                f"{max_abs_qvel:.3f} > {args.max_joint_speed:.3f} rad/s"
            )

    print(f"Model: {model_path}")
    print(f"Action: {args.action}; initialization: {initial_csv} row {args.initial_row}")
    print(f"Control period: {args.control_period:g}s; torque limit: {args.torque_limit:g} N*m")
    print(f"Hand contacts: {'enabled' if args.enable_contacts else 'disabled'}")
    try:
        if args.headless:
            while data.time < args.duration:
                physics_step()
        else:
            import mujoco.viewer

            with mujoco.viewer.launch_passive(model, data) as viewer:
                while viewer.is_running() and data.time < args.duration:
                    started = time.perf_counter()
                    physics_step()
                    viewer.sync()
                    remaining = model.opt.timestep - (time.perf_counter() - started)
                    if remaining > 0:
                        time.sleep(remaining)
    finally:
        if log_stream:
            log_stream.close()

    print(
        f"Finished at t={data.time:.3f}s; max |qvel|={max_abs_qvel:.3f} rad/s; "
        f"final max |torque|={np.max(np.abs(torque)):.3f} N*m"
    )


def main() -> None:
    try:
        run(parse_args())
    except (FileNotFoundError, ValueError, IndexError, FloatingPointError) as exc:
        raise SystemExit(f"error: {exc}") from exc


if __name__ == "__main__":
    main()
