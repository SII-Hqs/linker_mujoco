from pathlib import Path

from ament_index_python.packages import get_package_prefix
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = "linker_hand_mujoco_ros2"


def _default_reference_csv():
    data_dir = (
        Path.cwd()
        / "results"
        / "joint_logs"
        / "movement_without_thumb_joint0_1000hz"
    )
    candidates = sorted(data_dir.glob("*.csv"))
    return str(candidates[0]) if candidates else ""


def _launch_deployment(context):
    def value(name):
        return LaunchConfiguration(name).perform(context).strip()

    arguments = [
        "--model", value("model"),
        "--xml", value("xml"),
        "--torque-limit", value("torque_limit"),
        "--ramp-seconds", value("ramp_seconds"),
        "--max-joint-speed", value("max_joint_speed"),
        "--max-input-zscore", value("max_input_zscore"),
        "--timestep", value("timestep"),
        "--object-position", *value("object_position").replace(",", " ").split(),
        "--object-size", *value("object_size").replace(",", " ").split(),
        "--object-mass", value("object_mass"),
        "--object-friction",
        *value("object_friction").replace(",", " ").split(),
        "--object-control-range", value("object_control_range"),
        "--grasp-contact-solref",
        *value("grasp_contact_solref").replace(",", " ").split(),
        "--grasp-contact-solimp",
        *value("grasp_contact_solimp").replace(",", " ").split(),
    ]
    optional_values = {
        "reference_csv": "--reference-csv",
        "initial_csv": "--initial-csv",
        "duration": "--duration",
        "log": "--log",
    }
    for name, flag in optional_values.items():
        configured = value(name)
        if configured:
            arguments.extend([flag, configured])
    if value("headless").lower() in ("1", "true", "yes", "on"):
        arguments.append("--headless")
    if value("enable_contacts").lower() in ("1", "true", "yes", "on"):
        arguments.append("--enable-contacts")
    if value("enable_hand_self_collision").lower() in ("1", "true", "yes", "on"):
        arguments.append("--enable-hand-self-collision")

    executable = (
        Path(get_package_prefix(PACKAGE_NAME))
        / "lib"
        / PACKAGE_NAME
        / "future_pose_torque_mujoco"
    )
    return [
        ExecuteProcess(
            cmd=[
                "conda", "run", "-n", "linker_mujoco", "--no-capture-output",
                "python", str(executable), *arguments,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    package_share = FindPackageShare(PACKAGE_NAME)
    default_model = PathJoinSubstitution(
        [package_share, "deployment", "future_pose_10ms_torque_esn_model.npz"]
    )
    default_xml = PathJoinSubstitution(
        [
            package_share,
            "urdf",
            "L20",
            "linker_hand_l20_right",
            "linker_hand_l20_right.xml",
        ]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("model", default_value=default_model),
            DeclareLaunchArgument("xml", default_value=default_xml),
            DeclareLaunchArgument(
                "reference_csv",
                default_value=_default_reference_csv(),
                description="1000 Hz qpos trajectory used for q(t+10 ms).",
            ),
            DeclareLaunchArgument("initial_csv", default_value=""),
            DeclareLaunchArgument("duration", default_value=""),
            DeclareLaunchArgument("torque_limit", default_value="1.0"),
            DeclareLaunchArgument("ramp_seconds", default_value="0.1"),
            DeclareLaunchArgument("max_joint_speed", default_value="30.0"),
            DeclareLaunchArgument("max_input_zscore", default_value="20.0"),
            DeclareLaunchArgument("timestep", default_value="0.001"),
            DeclareLaunchArgument("enable_contacts", default_value="true"),
            DeclareLaunchArgument("object_position", default_value="0.19 0 0.10"),
            DeclareLaunchArgument("object_size", default_value="0.05 0.05 0.05"),
            DeclareLaunchArgument("object_mass", default_value="0.1"),
            DeclareLaunchArgument(
                "object_friction", default_value="0.6 0.02 0.001"
            ),
            DeclareLaunchArgument("object_control_range", default_value="0.30"),
            DeclareLaunchArgument(
                "enable_hand_self_collision", default_value="true"
            ),
            DeclareLaunchArgument(
                "grasp_contact_solref", default_value="0.005 1.0"
            ),
            DeclareLaunchArgument(
                "grasp_contact_solimp", default_value="0.90 0.99 0.002"
            ),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument(
                "log",
                default_value=str(
                    Path.cwd() / "results" / "future_pose_torque_deployment.csv"
                ),
            ),
            OpaqueFunction(function=_launch_deployment),
        ]
    )
