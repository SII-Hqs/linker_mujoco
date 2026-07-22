#!/usr/bin/env python3
"""Run the L20a hand contact-control demo with an ESN torque controller."""

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import mujoco
import mujoco.viewer
import numpy as np

if __package__:
    from ..io.package_paths import resource_path
    from .observer import ContactObserver
    from ..controllers.esn import ESNTorqueController as HorizonESNTorqueController
    from ..controllers.mujoco_esn_controller_current_torque import (
        ESNTorqueController as CurrentESNTorqueController,
    )
    from ..controllers.platform import MovingPlatformController
    from ..controllers.position_mapper import PositionCommandMapper
    from .contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
    from .contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.io.package_paths import resource_path
    from linker_hand_mujoco_ros2.contact.contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from linker_hand_mujoco_ros2.contact.observer import ContactObserver
    from linker_hand_mujoco_ros2.controllers.esn import (
        ESNTorqueController as HorizonESNTorqueController,
    )
    from linker_hand_mujoco_ros2.controllers.mujoco_esn_controller_current_torque import (
        ESNTorqueController as CurrentESNTorqueController,
    )
    from linker_hand_mujoco_ros2.controllers.platform import MovingPlatformController
    from linker_hand_mujoco_ros2.controllers.position_mapper import PositionCommandMapper


def default_current_esn_model_path():
    """Return the current-torque ESN model path used by horizon 0."""
    return resource_path(
        "resources",
        "weights",
        "esn_controller_current_torque_seed52_60hz.npz",
    )


def default_horizon_esn_model_path():
    """Return the previous multi-horizon ESN model path used by horizon 1."""
    return resource_path(
        "resources",
        "weights",
        "esn_controller_no_tau_history_seed67_60hz.npz",
    )


def default_esn_model_path():
    """Return the default ESN model path for backward-compatible CLI use."""
    return default_current_esn_model_path()


def _find_actuator_ids_for_joints(model, joint_names):
    """Return actuator IDs whose transmission targets the given joints."""
    target_names = set(joint_names)
    actuator_ids = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        joint_name = model.joint(joint_id).name
        if joint_name in target_names:
            actuator_ids.append(actuator_id)
    if not actuator_ids:
        raise ValueError(f"no actuators found for joints: {joint_names}")
    return actuator_ids


def _close_non_index_fingers(model, joint_names, command):
    """Return an initial command with all non-index fingers closed."""
    closed = np.asarray(command, dtype=float).copy()
    for i, joint_name in enumerate(joint_names):
        if joint_name.startswith("index_"):
            continue
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        joint_min, joint_max = model.jnt_range[joint_id]
        if joint_name.endswith("_joint0"):
            closed[i] = np.clip(0.0, joint_min, joint_max)
        else:
            closed[i] = joint_max
    return closed


class HorizonSwitchingControlRateAdapter:
    """Switch between the current-torque ESN and the previous horizon ESN."""

    horizons = (0, 1)

    def __init__(self, current_controller, horizon_controller, sim_dt, horizon=0):
        if sim_dt <= 0:
            raise ValueError("sim_dt must be positive")
        if current_controller.joint_ids != horizon_controller.joint_ids:
            raise ValueError(
                "current-torque and horizon ESN joint_ids must match: "
                f"{current_controller.joint_ids} != {horizon_controller.joint_ids}"
            )
        if 1 not in horizon_controller.horizons:
            raise ValueError(
                f"previous ESN must support horizon 1, got {horizon_controller.horizons}"
            )

        self.current_controller = current_controller
        self.horizon_controller = horizon_controller
        self.joint_ids = current_controller.joint_ids
        self.sim_dt = float(sim_dt)
        self.update_period = 1.0 / float(current_controller.fs_hz)
        self.elapsed = 0.0
        self.last_tau = np.zeros(3, dtype=float)
        self._horizon = 0
        self.horizon = horizon

    @property
    def horizon(self):
        return self._horizon

    @horizon.setter
    def horizon(self, value):
        value = int(value)
        if value not in self.horizons:
            raise ValueError(f"horizon must be one of {self.horizons}")
        changed = value != self._horizon
        self._horizon = value
        if changed:
            self.elapsed = self.update_period

    @property
    def active_controller(self):
        if self.horizon == 0:
            return self.current_controller
        return self.horizon_controller

    def reset(self):
        """Reset both ESN reservoirs and the output-rate limiter state."""
        self.current_controller.reset()
        self.horizon_controller.reset()
        self.elapsed = 0.0
        self.last_tau = np.zeros(3, dtype=float)

    def step(self, qpos, qvel, force_g):
        """Update torque only when the trained ESN control period has elapsed."""
        self.elapsed += self.sim_dt
        if self.elapsed >= self.update_period:
            self.elapsed -= self.update_period
            if self.horizon == 0:
                self.last_tau = self.current_controller.step(qpos, qvel, force_g)
            else:
                self.last_tau = self.horizon_controller.step(
                    qpos,
                    qvel,
                    force_g,
                    horizon=1,
                )
        return self.last_tau.copy()


@dataclass
class ESNContactSimulation:
    """Objects required to step the ESN contact-control simulation."""

    model: object
    data: object
    platform: MovingPlatformController
    observer: ContactObserver
    controller: HorizonSwitchingControlRateAdapter
    index_dof_ids: List[int] = field(default_factory=list)
    index_actuator_ids: List[int] = field(default_factory=list)

    def step(self):
        """Advance the ESN closed-loop simulation."""
        self.platform.update(self.data.time)
        contact_state = self.observer.read()

        qpos = self.data.qpos[self.index_dof_ids]
        qvel = self.data.qvel[self.index_dof_ids]
        force_g = contact_state.normal_force

        tau = self.controller.step(qpos, qvel, force_g)

        self.data.ctrl[:] = 0.0
        for i, act_id in enumerate(self.index_actuator_ids):
            self.data.ctrl[act_id] = tau[i]

        mujoco.mj_step(self.model, self.data)
        return contact_state


def _apply_contact_params(model, solref, solimp):
    """Override solref/solimp for all geom pairs at runtime."""
    solref = np.asarray(solref, dtype=float)
    solimp = np.asarray(solimp, dtype=float)
    for geom_id in range(model.ngeom):
        model.geom_solref[geom_id, : len(solref)] = solref
        model.geom_solimp[geom_id, : len(solimp)] = solimp


def build_esn_simulation(args):
    """Build and initialize the ESN contact-control simulation."""
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    model.dof_damping[:] = args.damping
    model.opt.gravity[:] = np.array([0.0, 0.0, args.gravity_z], dtype=float)

    # Soften contact if requested via --solref / --solimp.
    if args.solref is not None or args.solimp is not None:
        solref = args.solref if args.solref is not None else [0.02, 1.0]
        solimp = args.solimp if args.solimp is not None else [0.9, 0.95, 0.001]
        _apply_contact_params(model, solref, solimp)

    joint_names, initial_command = load_finger_frame(
        args.finger_vec,
        args.initial_frame,
    )
    initial_command = _close_non_index_fingers(model, joint_names, initial_command)
    initial_command = clip_to_joint_ranges(model, joint_names, initial_command)
    mapper = PositionCommandMapper(model, joint_names)
    mapper.apply_qpos(data, initial_command)
    data.ctrl[:] = 0.0
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    platform = MovingPlatformController(
        model,
        data,
        body_name="moving_platform",
        target_name=args.target,
        top_offset=args.platform_top_offset,
        stroke=args.platform_stroke,
        period=args.platform_period,
        initial_pos=[
            args.platform_initial_x,
            args.platform_initial_y,
            args.platform_initial_z,
        ],
        high_z_offset=args.high_z_offset,
        low_z_offset=args.low_z_offset,
    )
    observer = ContactObserver(
        model,
        data,
        "moving_platform_geom",
        force_clip=args.force_clip,
        force_scale=args.force_scale,
    )

    current_esn_model = args.esn_model if args.esn_model is not None else args.current_esn_model
    current_esn = CurrentESNTorqueController(
        model_path=current_esn_model,
        torque_limit=args.torque_limit,
        torque_rate_limit=args.torque_rate_limit,
        filter_cutoff_hz=args.filter_cutoff_hz,
        leak_alpha=getattr(args, "leak_alpha", None),
    )
    horizon_esn = HorizonESNTorqueController(
        model_path=args.horizon_esn_model,
        torque_limit=args.torque_limit,
        torque_rate_limit=args.torque_rate_limit,
        filter_cutoff_hz=args.filter_cutoff_hz,
    )
    controller = HorizonSwitchingControlRateAdapter(
        current_esn,
        horizon_esn,
        sim_dt=model.opt.timestep,
        horizon=args.horizon,
    )

    # Resolve the 3 controlled joints from the ESN model metadata.
    esn_joint_names = [f"index_joint{i}" for i in controller.joint_ids]
    index_dof_ids = []
    for name in esn_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"ESN joint not found in model: {name}")
        index_dof_ids.append(int(model.jnt_dofadr[jid]))
    index_actuator_ids = _find_actuator_ids_for_joints(model, esn_joint_names)

    return ESNContactSimulation(
        model, data, platform, observer, controller,
        index_dof_ids, index_actuator_ids,
    )


def run_demo(args):
    """Run the ESN-control demo with a viewer or headless."""
    simulation = build_esn_simulation(args)
    print("ESN contact-control demo started.")
    print(f"model: {args.model_xml}")
    print(f"current ESN: {args.esn_model if args.esn_model is not None else args.current_esn_model}")
    print(f"horizon ESN: {args.horizon_esn_model}")
    print(f"target: {args.target}")
    print(f"horizon: {simulation.controller.horizon}")
    print("Press Ctrl+C in the terminal to stop.", flush=True)

    last_log = -args.log_interval

    def step_once():
        nonlocal last_log
        contact_state = simulation.step()
        if simulation.data.time - last_log >= args.log_interval:
            last_log = simulation.data.time
            names = ",".join(contact_state.geoms) if contact_state.geoms else "none"
            tau = simulation.controller.last_tau
            tau_str = ",".join(f"{t:7.4f}" for t in tau)
            print(
                f"t={simulation.data.time:6.3f}s "
                f"contact={int(contact_state.in_contact)} "
                f"normal_force={contact_state.normal_force:8.4f}N "
                f"raw={contact_state.raw_normal_force:8.4f}N "
                f"tau=[{tau_str}] geoms={names}",
                flush=True,
            )

    if args.headless:
        end_time = simulation.data.time + args.duration
        while simulation.data.time < end_time:
            step_once()
        return

    with mujoco.viewer.launch_passive(simulation.model, simulation.data) as viewer:
        while viewer.is_running():
            step_start = time.time()
            with viewer.lock():
                step_once()
            viewer.sync()
            sleep_time = simulation.model.opt.timestep - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def add_esn_arguments(parser):
    """Add ESN-control demo arguments to an argument parser."""
    parser.add_argument("--model-xml", type=Path, default=default_torque_model_xml())
    parser.add_argument(
        "--esn-model",
        type=Path,
        default=None,
        help="Backward-compatible alias for --current-esn-model.",
    )
    parser.add_argument(
        "--current-esn-model",
        type=Path,
        default=default_current_esn_model_path(),
        help="ESN weight used when Horizon is 0.",
    )
    parser.add_argument(
        "--horizon-esn-model",
        type=Path,
        default=default_horizon_esn_model_path(),
        help="Previous multi-horizon ESN weight used when Horizon is 1.",
    )
    parser.add_argument("--finger-vec", type=Path, default=default_finger_vec())
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--target", default="index_link1")
    parser.add_argument("--platform-top-offset", type=float, default=0.075)
    parser.add_argument("--platform-initial-x", type=float, default=0.24212)
    parser.add_argument("--platform-initial-y", type=float, default=0.03445)
    parser.add_argument("--platform-initial-z", type=float, default=0.31972)
    parser.add_argument("--platform-stroke", type=float, default=0.004)
    parser.add_argument("--platform-period", type=float, default=4.0)
    parser.add_argument(
        "--high-z-offset",
        type=float,
        default=0.0,
        help="Lower the highest point by this amount (meters).",
    )
    parser.add_argument(
        "--low-z-offset",
        type=float,
        default=0.0,
        help="Raise the lowest point by this amount (meters).",
    )
    parser.add_argument("--horizon", type=int, choices=(0, 1), default=0)
    parser.add_argument("--torque-limit", type=float, default=2.0)
    parser.add_argument("--torque-rate-limit", type=float, default=None)
    parser.add_argument("--filter-cutoff-hz", type=float, default=None)
    parser.add_argument("--force-clip", type=float, default=5.0)
    parser.add_argument("--force-scale", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument("--leak-alpha", type=float, default=None, metavar="ALPHA", help="Reservoir leaky-integrator alpha in [0, 1]. Default: model dt.")
    parser.add_argument(
        "--solref",
        type=float,
        nargs=2,
        default=None,
        metavar=("TIMECONST", "DAMPRATIO"),
        help="Contact solref [time_constant, damping_ratio]. Larger time_constant = softer contact. Default: model XML values.",
    )
    parser.add_argument(
        "--solimp",
        type=float,
        nargs=3,
        default=None,
        metavar=("DMIN", "DMAX", "WIDTH"),
        help="Contact solimp [dmin, dmax, width]. Default: model XML values.",
    )
    parser.add_argument(
        "--gravity-z",
        type=float,
        default=0.0,
        help="World Z gravity. Defaults to 0.0 so no-contact motion stays quiet.",
    )
    return parser


def parse_args(argv):
    """Parse ESN-control demo command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an L20a hand/platform MuJoCo ESN torque-control demo."
    )
    add_esn_arguments(parser)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    return parser.parse_args(argv)


def main():
    """Run the ESN torque-control demo."""
    args = parse_args(sys.argv[1:])
    run_demo(args)


if __name__ == "__main__":
    main()
