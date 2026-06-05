import pickle
from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


def resolve_model_xml(package_root, model_xml):
    if model_xml:
        return Path(model_xml).expanduser().resolve()
    return (
        package_root
        / "linker_hand_mujoco_ros2"
        / "urdf"
        / "L20a"
        / "linker_hand_l20a_right"
        / "linker_hand_l20a_right.xml"
    )


def resolve_finger_vec_path(package_root, finger_vec_path):
    if finger_vec_path:
        return Path(finger_vec_path).expanduser().resolve()

    candidates = [
        package_root / "finger_l20_vec.pkl",
        Path.cwd() / "src" / "linker_hand_mujoco_ros2" / "finger_l20_vec.pkl",
        Path.cwd() / "finger_l20_vec.pkl",
    ]
    if get_package_share_directory is not None:
        try:
            candidates.append(
                Path(get_package_share_directory("linker_hand_mujoco_ros2"))
                / "finger_l20_vec.pkl"
            )
        except Exception:
            pass
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
