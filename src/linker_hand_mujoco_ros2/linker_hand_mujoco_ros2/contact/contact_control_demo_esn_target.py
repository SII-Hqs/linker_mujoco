#!/usr/bin/env python3
"""Run the L20a contact demo with a target q/qdot ESN torque controller."""

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
    from ..contact_observer import ContactObserver
    from ..mujoco_esn_controller_target_q_dq import (
        ControlRateAdapter,
        ESNTorqueController,
    )
    from ..platform_controller import MovingPlatformController
    from ..position_command_mapper import PositionCommandMapper
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
    from linker_hand_mujoco_ros2.contact.contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
    from linker_hand_mujoco_ros2.contact.contact_control_demo_torque import (
        clip_to_joint_ranges,
        default_torque_model_xml,
    )
    from linker_hand_mujoco_ros2.contact_observer import ContactObserver
    from linker_hand_mujoco_ros2.mujoco_esn_controller_target_q_dq import (
        ControlRateAdapter,
        ESNTorqueController,
    )
    from linker_hand_mujoco_ros2.platform_controller import MovingPlatformController
    from linker_hand_mujoco_ros2.position_command_mapper import PositionCommandMapper


def default_esn_model_path():
    """Return the default target q/qdot ESN model .npz path."""
    return (
        Path(__file__).resolve().parents[1]
        / "esn_controller_target_q_dq_7_seed30_60hz.npz"
    )


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


@dataclass
class ESNTargetContactSimulation:
    """Objects required to step the target q/qdot ESN contact simulation."""

    model: object
    data: object
    platform: MovingPlatformController
    observer: ContactObserver
    controller: ControlRateAdapter
    target_qpos: np.ndarray
    target_qvel: np.ndarray
    index_dof_ids: List[int] = field(default_factory=list)
    index_actuator_ids: List[int] = field(default_factory=list)

    def step(self):
        """Advance the target q/qdot ESN closed-loop simulation."""
        self.platform.update(self.data.time)
        contact_state = self.observer.read()

        qpos = self.current_qpos()
        qvel = self.data.qvel[self.index_dof_ids]
        force_g = contact_state.normal_force

        tau = self.controller.step(
            qpos,
            qvel,
            force_g,
            self.target_qpos,
            self.target_qvel,
        )

        self.data.ctrl[:] = 0.0
        for i, act_id in enumerate(self.index_actuator_ids):
            self.data.ctrl[act_id] = tau[i]

        mujoco.mj_step(self.model, self.data)
        return contact_state

    def current_qpos(self):
        """Return the controlled joint positions."""
        return self.data.qpos[self.index_dof_ids].copy()

    def tracking_error(self):
        """Return qpos - target_qpos for the controlled joints."""
        return self.current_qpos() - self.target_qpos


def _apply_contact_params(model, solref, solimp):
    """Override solref/solimp for all geom pairs at runtime."""
    solref = np.asarray(solref, dtype=float)
    solimp = np.asarray(solimp, dtype=float)
    for geom_id in range(model.ngeom):
        model.geom_solref[geom_id, : len(solref)] = solref
        model.geom_solimp[geom_id, : len(solimp)] = solimp


def build_esn_target_simulation(args):
    """Build and initialize the target q/qdot ESN contact simulation."""
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    model.dof_damping[:] = args.damping
    model.opt.gravity[:] = np.array([0.0, 0.0, args.gravity_z], dtype=float)

    if args.solref is not None or args.solimp is not None:
        solref = args.solref if args.solref is not None else [0.02, 1.0]
        solimp = args.solimp if args.solimp is not None else [0.9, 0.95, 0.001]
        _apply_contact_params(model, solref, solimp)

    joint_names, initial_command = load_finger_frame(
        args.finger_vec,
        args.initial_frame,
    )
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
    )

    esn = ESNTorqueController(
        model_path=args.esn_model,
        torque_limit=args.torque_limit,
        torque_rate_limit=args.torque_rate_limit,
        filter_cutoff_hz=args.filter_cutoff_hz,
    )
    controller = ControlRateAdapter(
        esn,
        sim_dt=model.opt.timestep,
        horizon=args.horizon,
    )

    esn_joint_names = [f"index_joint{i}" for i in esn.joint_ids]
    index_dof_ids = []
    for name in esn_joint_names:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"ESN joint not found in model: {name}")
        index_dof_ids.append(int(model.jnt_dofadr[jid]))
    index_actuator_ids = _find_actuator_ids_for_joints(model, esn_joint_names)

    target_qpos = np.asarray(args.target_qpos, dtype=float).reshape(3).copy()
    target_qvel = np.asarray(args.target_qvel, dtype=float).reshape(3).copy()

    return ESNTargetContactSimulation(
        model,
        data,
        platform,
        observer,
        controller,
        target_qpos,
        target_qvel,
        index_dof_ids,
        index_actuator_ids,
    )


def _format_vector(values):
    """Format a 3-D vector for compact logging."""
    return ",".join(f"{v:7.4f}" for v in values)


def run_demo(args):
    """Run the target q/qdot ESN demo with a viewer or headless."""
    simulation = build_esn_target_simulation(args)
    print("Target q/qdot ESN contact-control demo started.")
    print(f"model: {args.model_xml}")
    print(f"esn: {args.esn_model}")
    print(f"target: {args.target}")
    print(f"horizon: {args.horizon}")
    print(f"target_qpos: [{_format_vector(simulation.target_qpos)}]")
    print(f"target_qvel: [{_format_vector(simulation.target_qvel)}]")
    print("Press Ctrl+C in the terminal to stop.", flush=True)

    last_log = -args.log_interval

    def step_once():
        nonlocal last_log
        contact_state = simulation.step()
        if simulation.data.time - last_log >= args.log_interval:
            last_log = simulation.data.time
            names = ",".join(contact_state.geoms) if contact_state.geoms else "none"
            tau = simulation.controller.last_tau
            qpos = simulation.current_qpos()
            qerr = qpos - simulation.target_qpos
            print(
                f"t={simulation.data.time:6.3f}s "
                f"contact={int(contact_state.in_contact)} "
                f"normal_force={contact_state.normal_force:8.4f}N "
                f"raw={contact_state.raw_normal_force:8.4f}N "
                f"tau=[{_format_vector(tau)}] "
                f"qpos=[{_format_vector(qpos)}] "
                f"target_qpos=[{_format_vector(simulation.target_qpos)}] "
                f"qerr=[{_format_vector(qerr)}] "
                f"qerr_norm={np.linalg.norm(qerr):8.5f}rad "
                f"geoms={names}",
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


def add_esn_target_arguments(parser):
    """Add target q/qdot ESN-control demo arguments to an argument parser."""
    parser.add_argument("--model-xml", type=Path, default=default_torque_model_xml())
    parser.add_argument("--esn-model", type=Path, default=default_esn_model_path())
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
    parser.add_argument(
        "--target-qpos",
        type=float,
        nargs=3,
        default=[0.332, 0.254, 0.271],
        metavar=("Q1", "Q2", "Q3"),
        help="Target joint positions for the controlled joints in rad.",
    )
    parser.add_argument(
        "--target-qvel",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        metavar=("DQ1", "DQ2", "DQ3"),
        help="Target joint velocities for the controlled joints in rad/s.",
    )
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--torque-limit", type=float, default=2.0)
    parser.add_argument("--torque-rate-limit", type=float, default=None)
    parser.add_argument("--filter-cutoff-hz", type=float, default=None)
    parser.add_argument("--force-clip", type=float, default=5.0)
    parser.add_argument("--damping", type=float, default=0.8)
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
    """Parse target q/qdot ESN demo command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run an L20a hand/platform MuJoCo target q/qdot ESN "
            "torque-control demo."
        )
    )
    add_esn_target_arguments(parser)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    return parser.parse_args(argv)


def main():
    """Run the target q/qdot ESN torque-control demo."""
    args = parse_args(sys.argv[1:])
    run_demo(args)


if __name__ == "__main__":
    main()
