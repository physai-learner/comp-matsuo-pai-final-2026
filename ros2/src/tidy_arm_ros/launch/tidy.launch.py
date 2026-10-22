"""sim と planner を起動する。指示は別の端末から `ros2 run tidy_arm_ros tidy "..."` で出す。

ros2 launch tidy_arm_ros tidy.launch.py seed:=3 realtime:=false
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description() -> LaunchDescription:
    seed = LaunchConfiguration("seed")
    realtime = LaunchConfiguration("realtime")
    effort = LaunchConfiguration("effort")
    return LaunchDescription(
        [
            DeclareLaunchArgument("seed", default_value="0"),
            DeclareLaunchArgument("realtime", default_value="true"),
            DeclareLaunchArgument("effort", default_value="medium"),
            Node(
                package="tidy_arm_ros",
                executable="sim_node",
                name="sim",
                output="screen",
                parameters=[
                    {
                        "seed": ParameterValue(seed, value_type=int),
                        "realtime": ParameterValue(realtime, value_type=bool),
                    }
                ],
            ),
            Node(
                package="tidy_arm_ros",
                executable="planner_node",
                name="planner",
                output="screen",
                parameters=[{"effort": ParameterValue(effort, value_type=str)}],
            ),
        ]
    )
