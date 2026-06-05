#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

from linker_hand_mujoco_ros2.finger_vec_loader import resolve_model_xml


def default_retargeting_root():
    return Path(__file__).resolve().parents[1] / "retargeting_data"


def latest_session(root):
    root = Path(root)
    if not root.exists():
        return None
    sessions = [path for path in root.iterdir() if path.is_dir() and (path / "frame_log.csv").exists()]
    if not sessions:
        return None
    return sorted(sessions, key=lambda path: path.stat().st_mtime)[-1]


def default_output_path(data_dir):
    return Path(data_dir) / "l20a_dynamics_torque.csv"


def main(args=None):
    parser = argparse.ArgumentParser(
        description="Calculate L20a index contact and inverse-dynamics torques from retargeting data."
    )
    parser.add_argument("data_dir", nargs="?", default=None)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-xml", default="")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--pressure-scale", type=float, default=1.0)
    parser.add_argument("--pressure-offset", type=float, default=0.0)
    parser.add_argument("--smoothing-window", type=int, default=5)
    parser.add_argument("--contact-patch-link", default="index_link3")
    parser.add_argument("--contact-patch-mesh", default="index_link3")
    parser.add_argument("--contact-patch-radius", type=float, default=0.004)
    parser.add_argument("--contact-patch-max-points", type=int, default=120)
    parser.add_argument("--contact-force-sigma", type=float, default=0.002)
    parsed = parser.parse_args(sys.argv[1:] if args is None else args)

    if parsed.latest:
        data_dir = latest_session(default_retargeting_root())
        if data_dir is None:
            raise SystemExit(f"No replay sessions found under {default_retargeting_root()}")
    elif parsed.data_dir:
        data_dir = Path(parsed.data_dir).expanduser().resolve()
    else:
        raise SystemExit("data_dir is required unless --latest is used")

    if not data_dir.is_dir():
        raise SystemExit(f"{data_dir} is not a directory")

    package_root = Path(__file__).resolve().parents[1]
    model_xml = resolve_model_xml(package_root, parsed.model_xml)
    output = Path(parsed.output).expanduser().resolve() if parsed.output else default_output_path(data_dir)

    from linker_hand_mujoco_ros2.dynamics_torque import calculate_session_dynamics

    output_csv, written = calculate_session_dynamics(
        data_dir=data_dir,
        model_xml=model_xml,
        output_csv=output,
        start_frame=parsed.start_frame,
        end_frame=parsed.end_frame,
        pressure_scale=parsed.pressure_scale,
        pressure_offset=parsed.pressure_offset,
        smoothing_window=parsed.smoothing_window,
        contact_patch_link=parsed.contact_patch_link,
        contact_patch_mesh=parsed.contact_patch_mesh,
        contact_patch_radius=parsed.contact_patch_radius,
        contact_patch_max_points=parsed.contact_patch_max_points,
        contact_force_sigma=parsed.contact_force_sigma,
    )
    print(f"Wrote {written} frames to {output_csv}")


if __name__ == "__main__":
    main()
