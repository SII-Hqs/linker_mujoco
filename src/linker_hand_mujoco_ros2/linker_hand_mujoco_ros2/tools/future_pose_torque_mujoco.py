"""ROS 2 console entry point for the packaged future-pose torque deployment."""

import importlib.util
import sys

from linker_hand_mujoco_ros2.io.package_paths import resource_path


def main():
    script_path = resource_path("deployment", "future_pose_torque_mujoco.py")
    spec = importlib.util.spec_from_file_location(
        "linker_hand_future_pose_torque_deployment", script_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load future-pose deployment: {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.modules.pop(spec.name, None)


if __name__ == "__main__":
    main()
