import pickle
from pathlib import Path

from linker_hand_mujoco_ros2.io.package_paths import (
    package_share_root,
    resource_path,
    source_package_root,
)


def resolve_model_xml(package_root, model_xml):
    if model_xml:
        return Path(model_xml).expanduser().resolve()
    return resource_path(
        "urdf",
        "L20a",
        "linker_hand_l20a_right",
        "linker_hand_l20a_right.xml",
    )


def resolve_finger_vec_path(package_root, finger_vec_path):
    if finger_vec_path:
        return Path(finger_vec_path).expanduser().resolve()

    candidates = [
        resource_path("finger_l20_vec.pkl"),
        package_root / "finger_l20_vec.pkl",
        source_package_root() / "finger_l20_vec.pkl",
        Path.cwd() / "src" / "linker_hand_mujoco_ros2" / "finger_l20_vec.pkl",
        Path.cwd() / "finger_l20_vec.pkl",
    ]
    share_root = package_share_root()
    if share_root is not None:
        candidates.append(share_root / "finger_l20_vec.pkl")
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def load_finger_vec(path):
    with open(path, "rb") as f:
        payload = pickle.load(f)
    meta_data = payload.get("meta_data", {})
    joint_names = list(meta_data.get("joint_names", []))
    frames = payload.get("data", [])
    if not joint_names or int(meta_data.get("dof", len(joint_names))) != 21:
        raise ValueError(f"Invalid finger vec metadata in {path}")
    if not frames:
        raise ValueError(f"No finger frames found in {path}")
    return joint_names, frames
