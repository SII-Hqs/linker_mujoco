from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package="linker_hand_mujoco_ros2",
            executable="linker_hand_l20a_contact_node",
            name="linker_hand_l20a_contact_node",
            output="screen",
            # prefix="conda run -n linker_mujoco --no-capture-output",
            prefix="conda run -n linker_mujoco --no-capture-output python",
            parameters=[{
                "hand_type": "right",
                "initial_frame_index": 0,
                "contact_patch_link": "index_link3",
                "contact_patch_mesh": "index_link3",
                "contact_patch_radius": 0.004,
                "contact_patch_update_hz": 20.0,
                "contact_patch_max_points": 120,
                "contact_force_sigma": 0.002,
                "contact_force_arrow_scale": 0.2,
                "clip_joint_command": True,
            }],
        ),
    ])
