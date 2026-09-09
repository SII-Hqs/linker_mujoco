from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = "linker_hand_mujoco_ros2"


def _default_reference_csv() -> str:
    directory = (
        Path.cwd()
        / "results"
        / "joint_logs"
        / "movement_without_thumb_joint0_100hz"
    )
    candidates = sorted(directory.glob("*.csv"))
    return str(candidates[0]) if candidates else ""


def _launch_gui(context):
    def value(name):
        return LaunchConfiguration(name).perform(context).strip()

    arguments = [
        "--candidate-dir",
        value("candidate_dir"),
        "--deployment-script",
        value("deployment_script"),
        "--xml",
        value("xml"),
        "--torque-limit",
        value("torque_limit"),
        "--ramp-seconds",
        value("ramp_seconds"),
        "--max-joint-speed",
        value("max_joint_speed"),
        "--max-input-zscore",
        value("max_input_zscore"),
        "--initial-row",
        value("initial_row"),
        "--visualization-fps",
        value("visualization_fps"),
        "--visualization-window-seconds",
        value("visualization_window_seconds"),
        "--visualization-backend",
        value("visualization_backend"),
    ]
    optional_values = {
        "model": "--model",
        "reference_csv": "--reference-csv",
        "initial_csv": "--initial-csv",
        "duration": "--duration",
        "reservoir_warmup_steps": "--reservoir-warmup-steps",
        "log": "--log",
    }
    for name, flag in optional_values.items():
        configured = value(name)
        if configured:
            arguments.extend([flag, configured])
    if value("enable_contacts").lower() in ("1", "true", "yes", "on"):
        arguments.append("--enable-contacts")
    if value("headless").lower() in ("1", "true", "yes", "on"):
        arguments.append("--headless")
    if value("live_visualization").lower() in ("1", "true", "yes", "on"):
        arguments.append("--live-visualization")
    if value("auto_start").lower() in ("1", "true", "yes", "on"):
        arguments.append("--auto-start")

    return [
        ExecuteProcess(
            cmd=[
                "conda",
                "run",
                "-n",
                "linker_mujoco",
                "--no-capture-output",
                "python",
                value("gui_script"),
                *arguments,
            ],
            output="screen",
        )
    ]


def generate_launch_description():
    package_share = FindPackageShare(PACKAGE_NAME)
    candidate_dir = PathJoinSubstitution(
        [package_share, "deployment", "candidate_models_extended_2s"]
    )
    default_model = PathJoinSubstitution(
        [
            package_share,
            "deployment",
            "candidate_models_extended_2s",
            "candidate_0004_review.npz",
        ]
    )
    deployment_script = PathJoinSubstitution(
        [package_share, "deployment", "screened_reservoir_torque_mujoco.py"]
    )
    gui_script = PathJoinSubstitution(
        [package_share, "deployment", "screened_reservoir_torque_gui.py"]
    )
    xml = PathJoinSubstitution(
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
            DeclareLaunchArgument("candidate_dir", default_value=candidate_dir),
            DeclareLaunchArgument("deployment_script", default_value=deployment_script),
            DeclareLaunchArgument("gui_script", default_value=gui_script),
            DeclareLaunchArgument("model", default_value=default_model),
            DeclareLaunchArgument(
                "reference_csv",
                default_value=_default_reference_csv(),
                description="100 Hz qpos trajectory used as the ESN future target.",
            ),
            DeclareLaunchArgument("initial_csv", default_value=""),
            DeclareLaunchArgument("initial_row", default_value="0"),
            DeclareLaunchArgument("xml", default_value=xml),
            DeclareLaunchArgument("duration", default_value=""),
            DeclareLaunchArgument("torque_limit", default_value="1.0"),
            DeclareLaunchArgument("ramp_seconds", default_value="0.1"),
            DeclareLaunchArgument("reservoir_warmup_steps", default_value=""),
            DeclareLaunchArgument("max_joint_speed", default_value="30.0"),
            DeclareLaunchArgument("max_input_zscore", default_value="20.0"),
            DeclareLaunchArgument("enable_contacts", default_value="false"),
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("live_visualization", default_value="true"),
            DeclareLaunchArgument("visualization_fps", default_value="15.0"),
            DeclareLaunchArgument(
                "visualization_window_seconds", default_value="2.5"
            ),
            DeclareLaunchArgument("visualization_backend", default_value="QtAgg"),
            DeclareLaunchArgument(
                "auto_start",
                default_value="false",
                description="Start the selected model immediately and exit when it finishes.",
            ),
            DeclareLaunchArgument(
                "log",
                default_value=str(
                    Path.cwd()
                    / "results"
                    / "screened_reservoir_torque_deployment.csv"
                ),
            ),
            OpaqueFunction(function=_launch_gui),
        ]
    )
