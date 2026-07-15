from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hand_type = LaunchConfiguration("hand_type")
    hand_joint = LaunchConfiguration("hand_joint")
    record_joint_state = LaunchConfiguration("record_joint_state")
    joint_state_log_path = LaunchConfiguration("joint_state_log_path")
    joint_state_log_hz = LaunchConfiguration("joint_state_log_hz")
    record_control_gui = LaunchConfiguration("record_control_gui")

    return LaunchDescription([
        DeclareLaunchArgument(
            "hand_type",
            default_value="left",
            description="Hand side. Use 'left' for /cb_left_hand_control_cmd or 'right' for /cb_right_hand_control_cmd.",
        ),
        DeclareLaunchArgument(
            "hand_joint",
            default_value="O20",
            description="LinkerHand model type, for example L10, L20, L20a, L21, or O20.",
        ),
        DeclareLaunchArgument(
            "record_joint_state",
            default_value="false",
            description="Start non-thumb joint qpos/qvel CSV recording immediately.",
        ),
        DeclareLaunchArgument(
            "record_control_gui",
            default_value="true",
            description="Show a small Start/Stop window for joint-state recording.",
        ),
        DeclareLaunchArgument(
            "joint_state_log_path",
            default_value="",
            description="CSV output path. Empty means results/joint_logs/<model>_<side>_<timestamp>.csv.",
        ),
        DeclareLaunchArgument(
            "joint_state_log_hz",
            default_value="60.0",
            description="Non-thumb joint qpos/qvel CSV sampling rate in Hz.",
        ),
        Node(
            package="linker_hand_mujoco_ros2",
            executable="linker_hand_mujoco_ros2_node",
            name="linker_hand_mujoco_ros2_node",
            output="screen",
            # Keep the same runtime style as linker_hand_l20a_contact.launch.py.
            prefix="conda run -n linker_mujoco --no-capture-output python",
            parameters=[{
                "hand_type": hand_type,
                "hand_joint": hand_joint,
                "show_contact_patch": False,
                "record_joint_state": record_joint_state,
                "joint_state_log_path": joint_state_log_path,
                "joint_state_log_hz": joint_state_log_hz,
                "record_control_gui": record_control_gui,
            }],
        ),
    ])
