#!/usr/bin/env python3
"""Run the L20a hand contact-control demo with an automatic platform."""

import argparse
import pickle
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

if __package__:
    from ..contact_observer import ContactObserver
    from ..platform_controller import MovingPlatformController
    from ..position_command_mapper import PositionCommandMapper
    from ..reflex_controller import ReflexIndexController
else:
    package_source = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(package_source))
    from linker_hand_mujoco_ros2.contact_observer import ContactObserver
    from linker_hand_mujoco_ros2.platform_controller import MovingPlatformController
    from linker_hand_mujoco_ros2.position_command_mapper import PositionCommandMapper
    from linker_hand_mujoco_ros2.reflex_controller import ReflexIndexController

numpy_core = getattr(np, "_core", np.core)
sys.modules.setdefault("numpy._core", numpy_core)
sys.modules.setdefault("numpy._core.multiarray", numpy_core.multiarray)
sys.modules.setdefault("numpy._core.numeric", numpy_core.numeric)


@dataclass
class ContactSimulation:
    """Objects required to step the contact-control simulation."""

    model: object
    data: object
    mapper: PositionCommandMapper
    platform: MovingPlatformController
    observer: ContactObserver
    controller: ReflexIndexController

    def step(self):
        """Advance the closed-loop simulation by one MuJoCo timestep."""
        self.platform.update(self.data.time)
        contact_state = self.observer.read()
        command = self.controller.compute(contact_state, self.model.opt.timestep)
        self.data.ctrl[:] = self.mapper.command_to_ctrl(command)
        mujoco.mj_step(self.model, self.data)
        return contact_state


def package_root():
    """Return the ROS2 package source root."""
    return Path(__file__).resolve().parents[2]


def default_model_xml():
    """Return the standalone platform demo model path."""
    return (
        Path(__file__).resolve().parents[1]
        / "urdf"
        / "L20a"
        / "linker_hand_l20a_right"
        / "linker_hand_l20a_right_contact_platform.xml"
    )


def default_finger_vec():
    """Return the default saved L20a joint command path."""
    return package_root() / "finger_l20_vec.pkl"


def load_finger_frame(path, frame_index):
    """Load one initial joint command frame."""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    joint_names = list(payload["meta_data"]["joint_names"])
    frames = payload["data"]
    frame_index = min(max(int(frame_index), 0), len(frames) - 1)
    return joint_names, np.asarray(frames[frame_index], dtype=float)


def build_simulation(args):
    """Build and initialize the shared contact-control simulation."""
    model = mujoco.MjModel.from_xml_path(str(args.model_xml))
    data = mujoco.MjData(model)
    model.dof_damping[:] = args.damping
    model.opt.gravity[:] = np.array([0.0, 0.0, args.gravity_z], dtype=float)

    joint_names, initial_command = load_finger_frame(
        args.finger_vec,
        args.initial_frame,
    )
    mapper = PositionCommandMapper(model, joint_names)
    initial_command = mapper.clip_command(initial_command)
    mapper.apply_qpos(data, initial_command)
    data.ctrl[:] = mapper.command_to_ctrl(initial_command)
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
    controller = ReflexIndexController(
        model,
        joint_names,
        initial_command,
        gain=args.reflex_gain,
        max_delta=args.reflex_max_delta,
        filter_tau=args.force_filter_tau,
        rate_limit=args.command_rate_limit,
    )
    return ContactSimulation(model, data, mapper, platform, observer, controller)


def run_demo(args):
    """Run the automatic platform demo with a viewer or headless."""
    simulation = build_simulation(args)
    print("Contact-control demo started.")
    print(f"model: {args.model_xml}")
    print(f"target: {args.target}")
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


def add_simulation_arguments(parser):
    """Add common model and controller arguments to an argument parser."""
    parser.add_argument("--model-xml", type=Path, default=default_model_xml())
    parser.add_argument("--finger-vec", type=Path, default=default_finger_vec())
    parser.add_argument("--initial-frame", type=int, default=0)
    parser.add_argument("--target", default="index_link1")
    parser.add_argument("--platform-top-offset", type=float, default=0.075)
    parser.add_argument("--platform-initial-x", type=float, default=0.24212)
    parser.add_argument("--platform-initial-y", type=float, default=0.03445)
    parser.add_argument("--platform-initial-z", type=float, default=0.31972)
    parser.add_argument("--platform-stroke", type=float, default=0.004)
    parser.add_argument("--platform-period", type=float, default=4.0)
    parser.add_argument("--reflex-gain", type=float, default=0.01)
    parser.add_argument("--reflex-max-delta", type=float, default=0.12)
    parser.add_argument("--force-filter-tau", type=float, default=0.08)
    parser.add_argument("--command-rate-limit", type=float, default=0.6)
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
    """Parse automatic demo command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run a minimal L20a hand/platform MuJoCo contact-control demo."
    )
    add_simulation_arguments(parser)
    parser.add_argument("--log-interval", type=float, default=0.25)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--duration", type=float, default=6.0)
    return parser.parse_args(argv)


def main():
    """Run the command-line demo."""
    args = parse_args(sys.argv[1:])
    run_demo(args)


if __name__ == "__main__":
    main()
