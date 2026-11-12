from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription

from unitree_description import (
    GO2_DESCRIPTION_URDF_PATH,
    G1_LOCKED_WAIST_DESCRIPTION_URDF_PATH,
    G1_UNLOCKED_WAIST_DESCRIPTION_URDF_PATH,
)


def launch_setup(context, *args, **kwargs):
    # Conditional URDF file depending on robot launch argument
    robot_arg_value = LaunchConfiguration("robot").perform(context)
    lock_waist_arg_value = LaunchConfiguration("lock_waist").perform(context)

    # map the differents launch options to the correct URDFs
    urdf_map = {
        "g1_True": G1_LOCKED_WAIST_DESCRIPTION_URDF_PATH,
        "g1_False": G1_UNLOCKED_WAIST_DESCRIPTION_URDF_PATH,
        "go2": GO2_DESCRIPTION_URDF_PATH,
    }
    urdf_path = urdf_map.get(
        str(robot_arg_value) + "_" + str(lock_waist_arg_value) if str(robot_arg_value) == "g1" else str(robot_arg_value)
    )

    with open(urdf_path, "r") as f:
        robot_description = f.read()

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[{"robot_description": robot_description}],
    )

    # Robot simulation
    go2_simulation_node = Node(
        package="unitree_simulation",
        executable="simulation_node",
        name="simulation_node",
        output="screen",
        parameters=[
            {
                "simulator": LaunchConfiguration("simulator"),
                "robot": LaunchConfiguration("robot"),
                "lock_waist": LaunchConfiguration("lock_waist"),
                "unlock_base": LaunchConfiguration("unlock_base"),
            }
        ],
    )

    return [robot_state_publisher, go2_simulation_node]


def generate_launch_description():
    # Declare launch arguments
    simulator_arg = DeclareLaunchArgument(
        "simulator", default_value="pybullet", description="Which simulator to use 'pybullet'"
    )

    robot_arg = DeclareLaunchArgument("robot", description="Which robot to simulate 'go2' or 'g1'")

    lock_waist_arg = DeclareLaunchArgument(
        "lock_waist",
        default_value="True",
        description="Whether to unlock or not the pitch and roll waist DOF, False by default",
    )

    unlock_base_arg = DeclareLaunchArgument(
        "unlock_base",
        default_value="False",
        description="should the robot base be free from start, or should it simulate being hanged first",
    )
    return LaunchDescription(
        [simulator_arg, robot_arg, lock_waist_arg, unlock_base_arg, OpaqueFunction(function=launch_setup)]
    )
