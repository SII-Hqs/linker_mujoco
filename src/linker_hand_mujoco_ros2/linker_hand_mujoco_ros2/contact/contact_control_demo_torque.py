#!/usr/bin/env python3
"""Run the L20a hand contact-control demo with direct torque control."""

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

if __package__:
    from ..contact_observer import ContactObserver
    from ..force_controller import ForceTrackingController
    from ..platform_controller import MovingPlatformController
    from ..position_command_mapper import PositionCommandMapper
    from .contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact.contact_control_demo import (
        default_finger_vec,
        load_finger_frame,
    )
    from linker_hand_mujoco_ros2.contact_observer import ContactObserver
    from linker_hand_mujoco_ros2.force_controller import ForceTrackingController
    from linker_hand_mujoco_ros2.platform_controller import MovingPlatformController
    from linker_hand_mujoco_ros2.position_command_mapper import PositionCommandMapper


@dataclass
class TorqueContactSimulation:
    """Objects required to step the torque contact-control simulation."""

    model: object
    data: object
    platform: MovingPlatformController
    observer: ContactObserver
    controller: ForceTrackingController

    def step(self):
        """Advance the direct-torque closed-loop simulation."""
        self.platform.update(self.data.time)
        contact_state = self.observer.read()
        self.data.ctrl[:] = self.controller.compute(
            contact_state,
            self.model.opt.timestep,
        )
        mujoco.mj_step(self.model, self.data)
        return contact_state


def default_torque_model_xml():
    """Return the standalone torque-control platform model path."""
    return (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "L20a"
        / "linker_hand_l20a_right"
        / "linker_hand_l20a_right_contact_platform_torque.xml"
    )


def clip_to_joint_ranges(model, joint_names, command):
    """Clip a joint command to the model joint ranges."""
    clipped = np.asarray(command, dtype=float).copy()
    for i, joint_name in enumerate(joint_names):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        joint_range = model.jnt_range[joint_id]
        clipped[i] = np.clip(clipped[i], joint_range[0], joint_range[1])
    return clipped


def build_torque_simulation(args):
    """Build and initialize the shared torque-control simulation."""
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    model.dof_damping[:] = args.damping
    model.opt.gravity[:] = np.array([0.0, 0.0, args.gravity_z], dtype=float)

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
    )
    observer = ContactObserver(
        model,
        data,
        "moving_platform_geom",
        force_clip=args.force_clip,
    )
    controller = ForceTrackingController(
        model,
        target_force=args.target_force,
        kp=args.kp,
        ki=args.ki,
        kd=args.kd,
        integral_limit=args.integral_limit,
        torque_limit=args.torque_limit,
    )
    return TorqueContactSimulation(model, data, platform, observer, controller)


def run_demo(args):
    """Run the torque-control demo with a viewer or headless."""
    simulation = build_torque_simulation(args)
    print("Torque contact-control demo started.")
    print(f"model: {args.model_xml}")
    print(f"target: {args.target}")
    print(f"target_force: {args.target_force:.4f} N")
    print("Press Ctrl+C in the terminal to stop.", flush=True)

    last_log = -args.log_interval

    def step_once():
        nonlocal last_log
        contact_state = simulation.step()
        if simulation.data.time - last_log >= args.log_interval:
            last_log = simulation.data.time
            names = ",".join(contact_state.geoms) if contact_state.geoms else "none"
            print(
                f"t={simulation.data.time:6.3f}s "
                f"contact={int(contact_state.in_contact)} "
                f"normal_force={contact_state.normal_force:8.4f}N "
                f"raw={contact_state.raw_normal_force:8.4f}N geoms={names}",
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


def add_torque_arguments(parser):
    """Add torque-control demo arguments to an argument parser."""
    parser.add_argument("--model-xml", type=Path, default=default_torque_model_xml())
    parser.add_argument("--finger-vec", type=Path, default=default_finger_vec())
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--target", default="index_link1")
    parser.add_argument("--platform-top-offset", type=float, default=0.075)
    parser.add_argument("--platform-initial-x", type=float, default=0.24212)
    parser.add_argument("--platform-initial-y", type=float, default=0.03445)
    parser.add_argument("--platform-initial-z", type=float, default=0.31972)
    parser.add_argument("--platform-stroke", type=float, default=0.004)
    parser.add_argument("--platform-period", type=float, default=4.0)
    parser.add_argument("--target-force", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=0.05)
    parser.add_argument("--ki", type=float, default=0.0)
    parser.add_argument("--kd", type=float, default=0.0)
    parser.add_argument("--integral-limit", type=float, default=10.0)
    parser.add_argument("--torque-limit", type=float, default=2.0)
    parser.add_argument("--force-clip", type=float, default=5.0)
    parser.add_argument("--damping", type=float, default=0.8)
    parser.add_argument(
        "--gravity-z",
        type=float,
        default=0.0,
        help="World Z gravity. Defaults to 0.0 so no-contact motion stays quiet.",
    )
    return parser


def parse_args(argv):
    """Parse torque-control demo command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run an L20a hand/platform MuJoCo torque-control demo."
    )
    add_torque_arguments(parser)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    return parser.parse_args(argv)


def main():
    """Run the direct torque-control demo."""
    args = parse_args(sys.argv[1:])
    run_demo(args)


if __name__ == "__main__":
    main()
