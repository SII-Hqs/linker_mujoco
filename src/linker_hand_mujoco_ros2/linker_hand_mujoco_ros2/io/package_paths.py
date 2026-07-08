from pathlib import Path

try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None


PACKAGE_NAME = "linker_hand_mujoco_ros2"


def python_package_root():
    return Path(__file__).resolve().parents[1]


def source_package_root():
    return python_package_root().parent


def package_share_root():
    if get_package_share_directory is None:
        return None
    try:
        return Path(get_package_share_directory(PACKAGE_NAME))
    except Exception:
        return None


def first_existing_path(*paths):
    for path in paths:
        if path is not None and Path(path).exists():
            return Path(path).resolve()
    return Path(paths[0]).resolve()


def resource_path(*parts):
    rel_path = Path(*parts)
    return first_existing_path(
        python_package_root() / rel_path,
        source_package_root() / rel_path,
        package_share_root() / rel_path if package_share_root() is not None else None,
    )


def retargeting_data_root():
    return first_existing_path(
        source_package_root() / "retargeting_data",
        package_share_root() / "retargeting_data" if package_share_root() is not None else None,
    )
