from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # Declare launch arguments for parameters
    simulator_arg = DeclareLaunchArgument(
        "simulator", default_value="pybullet", description="Which simulator to use 'pybullet'"
    )

    robot_arg = DeclareLaunchArgument("robot", description="Which robot to simulate 'go2' or 'g1'")

    unlock_base_arg = DeclareLaunchArgument(
        "unlock_base",
        default_value="False",
        description="should the robot base be free from start, or should it simulate being hanged first",
    )

    # Node configuration
    go2_simulation_node = Node(
        package="unitree_simulation",  # Replace with the actual package name
        executable="simulation_node",  # Replace with the actual node executable name
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

    # Launch description
    return LaunchDescription([simulator_arg, robot_arg, unlock_base_arg, go2_simulation_node])
