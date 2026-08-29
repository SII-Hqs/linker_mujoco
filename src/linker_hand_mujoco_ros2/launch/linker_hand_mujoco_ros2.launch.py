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
    joint_snapshot_path = LaunchConfiguration("joint_snapshot_path")
    record_control_gui = LaunchConfiguration("record_control_gui")
    enable_grasp_evaluation = LaunchConfiguration("enable_grasp_evaluation")
    grasp_object_position = LaunchConfiguration("grasp_object_position")
    grasp_object_control_gui = LaunchConfiguration("grasp_object_control_gui")
    grasp_object_control_range = LaunchConfiguration("grasp_object_control_range")
    enable_hand_self_collision = LaunchConfiguration("enable_hand_self_collision")
    grasp_contact_timestep = LaunchConfiguration("grasp_contact_timestep")
    grasp_contact_solref = LaunchConfiguration("grasp_contact_solref")
    grasp_contact_solimp = LaunchConfiguration("grasp_contact_solimp")
    enable_esn_controller = LaunchConfiguration("enable_esn_controller")
    esn_model_variant = LaunchConfiguration("esn_model_variant")
    cascaded_grasp_target_qpos = LaunchConfiguration("cascaded_grasp_target_qpos")
    esn_torque_limit = LaunchConfiguration("esn_torque_limit")
    esn_enable_object_contact = LaunchConfiguration("esn_enable_object_contact")
    esn_contact_force_target = LaunchConfiguration("esn_contact_force_target")
    esn_contact_force_limit = LaunchConfiguration("esn_contact_force_limit")

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
            description="Start all joint qpos/qvel CSV recording immediately.",
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
            description="All joint qpos/qvel CSV sampling rate in Hz.",
        ),
        DeclareLaunchArgument(
            "joint_snapshot_path",
            default_value="",
            description="CSV path for timestamped current-frame joint snapshots.",
        ),
        DeclareLaunchArgument(
            "enable_grasp_evaluation",
            default_value="false",
            description="Add a free test object and evaluate grasp stability.",
        ),
        DeclareLaunchArgument(
            "grasp_object_position",
            default_value="0.19 0 0.30",
            description="Initial test-object XYZ position in metres.",
        ),
        DeclareLaunchArgument(
            "grasp_object_control_gui",
            default_value="true",
            description="Show the interactive XYZ grasp-object control panel.",
        ),
        DeclareLaunchArgument(
            "grasp_object_control_range",
            default_value="0.30",
            description="XYZ adjustment range around the initial object position in metres.",
        ),
        DeclareLaunchArgument(
            "enable_hand_self_collision",
            default_value="true",
            description="Enable collision between different fingers.",
        ),
        DeclareLaunchArgument(
            "grasp_contact_timestep",
            default_value="0.002",
            description="MuJoCo physics timestep used for grasp contact in seconds.",
        ),
        DeclareLaunchArgument(
            "grasp_contact_solref",
            default_value="0.005 1.0",
            description="Hand-object contact solref time constant and damping ratio.",
        ),
        DeclareLaunchArgument(
            "grasp_contact_solimp",
            default_value="0.90 0.99 0.002",
            description="Hand-object contact solimp dmin, dmax, and width.",
        ),
        DeclareLaunchArgument(
            "enable_esn_controller",
            default_value="false",
            description="Enable manual open/grasp ESN torque controls.",
        ),
        DeclareLaunchArgument(
            "esn_model_variant",
            default_value="combined",
            description="Use 'combined', 'separate', or the grasp-only 'cascaded' controller.",
        ),
        DeclareLaunchArgument(
            "cascaded_grasp_target_qpos",
            default_value="",
            description="Optional 21-angle grasp goal; empty uses the ESN1 training mean goal.",
        ),
        DeclareLaunchArgument(
            "esn_torque_limit",
            default_value="0.20",
            description="Per-joint ESN torque limit in N m.",
        ),
        DeclareLaunchArgument(
            "esn_enable_object_contact",
            default_value="false",
            description="Allow the test object to contact the hand during ESN control.",
        ),
        DeclareLaunchArgument(
            "esn_contact_force_target",
            default_value="1.5",
            description="Normal force where ESN torque starts decreasing, in N.",
        ),
        DeclareLaunchArgument(
            "esn_contact_force_limit",
            default_value="5.0",
            description="Normal force where ESN grasp torque reaches zero, in N.",
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
                "joint_snapshot_path": joint_snapshot_path,
                "record_control_gui": record_control_gui,
                "enable_grasp_evaluation": enable_grasp_evaluation,
                "grasp_object_position": grasp_object_position,
                "grasp_object_control_gui": grasp_object_control_gui,
                "grasp_object_control_range": grasp_object_control_range,
                "enable_hand_self_collision": enable_hand_self_collision,
                "grasp_contact_timestep": grasp_contact_timestep,
                "grasp_contact_solref": grasp_contact_solref,
                "grasp_contact_solimp": grasp_contact_solimp,
                "enable_esn_controller": enable_esn_controller,
                "esn_model_variant": esn_model_variant,
                "cascaded_grasp_target_qpos": cascaded_grasp_target_qpos,
                "esn_torque_limit": esn_torque_limit,
                "esn_enable_object_contact": esn_enable_object_contact,
                "esn_contact_force_target": esn_contact_force_target,
                "esn_contact_force_limit": esn_contact_force_limit,
            }],
        ),
    ])
