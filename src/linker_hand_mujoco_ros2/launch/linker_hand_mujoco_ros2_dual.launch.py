from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    hand_joint = LaunchConfiguration("hand_joint")
    left_hand_joint = LaunchConfiguration("left_hand_joint")
    right_hand_joint = LaunchConfiguration("right_hand_joint")
    left_root_pos = LaunchConfiguration("left_root_pos")
    right_root_pos = LaunchConfiguration("right_root_pos")
    left_root_quat = LaunchConfiguration("left_root_quat")
    right_root_quat = LaunchConfiguration("right_root_quat")

    return LaunchDescription([
        DeclareLaunchArgument(
            "hand_joint",
            default_value="L20",
            description="Default LinkerHand model type for both hands, for example L10, L20, or L21.",
        ),
        DeclareLaunchArgument(
            "left_hand_joint",
            default_value="",
            description="Optional left hand model type. Empty means use hand_joint.",
        ),
        DeclareLaunchArgument(
            "right_hand_joint",
            default_value="",
            description="Optional right hand model type. Empty means use hand_joint.",
        ),
        DeclareLaunchArgument(
            "left_root_pos",
            default_value="-0.18 0 0",
            description="Left hand root position in the merged MuJoCo world.",
        ),
        DeclareLaunchArgument(
            "right_root_pos",
            default_value="0.18 0 0",
            description="Right hand root position in the merged MuJoCo world.",
        ),
        DeclareLaunchArgument(
            "left_root_quat",
            default_value="1 0 0 0",
            description="Left hand root quaternion in w x y z order.",
        ),
        DeclareLaunchArgument(
            "right_root_quat",
            default_value="1 0 0 0",
            description="Right hand root quaternion in w x y z order.",
        ),
        Node(
            package="linker_hand_mujoco_ros2",
            executable="linker_hand_mujoco_ros2_dual_node",
            name="linker_hand_mujoco_ros2_dual_node",
            output="screen",
            prefix="conda run -n linker_mujoco --no-capture-output python",
            parameters=[{
                "hand_joint": hand_joint,
                "left_hand_joint": left_hand_joint,
                "right_hand_joint": right_hand_joint,
                "left_root_pos": left_root_pos,
                "right_root_pos": right_root_pos,
                "left_root_quat": left_root_quat,
                "right_root_quat": right_root_quat,
            }],
        ),
    ])
