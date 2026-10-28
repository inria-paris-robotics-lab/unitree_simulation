import os

from ament_index_python.packages import get_package_share_directory
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from launch import LaunchDescription


def launch_setup(context, *args, **kwargs):
    # Conditional URDF file depending on robot launch argument
    robot_arg_value = LaunchConfiguration("robot").perform(context)
    pkg_path = get_package_share_directory("unitree_description")

    urdf_map = {
        "g1": "g1/g1.urdf",
        "go2": "go2/go2.urdf",
    }
    urdf_file = urdf_map.get(str(robot_arg_value))
    urdf_path = os.path.join(pkg_path, "model", urdf_file)
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

    unlock_base_arg = DeclareLaunchArgument(
        "unlock_base",
        default_value="False",
        description="should the robot base be free from start, or should it simulate being hanged first",
    )
    return LaunchDescription([simulator_arg, robot_arg, unlock_base_arg, OpaqueFunction(function=launch_setup)])
