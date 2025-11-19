#!/usr/bin/env python3

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    """
    Launch file for autonomous frontier exploration.
    
    This launch file starts:
    1. RViz for visualization
    2. RViz goal relay node
    3. State publisher for turtlebot
    4. Navigator node for path planning and control
    5. Frontier exploration node (with optional delay)
    
    Note: This does not launch the simulator or hardware bringup.
    """
    
    use_sim_time = LaunchConfiguration("use_sim_time")
    exploration_delay = LaunchConfiguration("exploration_delay")
    
    # Declare launch arguments
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation time if true",
    )
    
    declare_exploration_delay = DeclareLaunchArgument(
        "exploration_delay",
        default_value="5.0",
        description="Delay (in seconds) before starting exploration node",
    )
    
    # RViz visualization
    rviz_launch = IncludeLaunchDescription(
        PathJoinSubstitution(
            [FindPackageShare("asl_tb3_sim"), "launch", "rviz.launch.py"]
        ),
        launch_arguments={
            "config": PathJoinSubstitution(
                [
                    FindPackageShare("autonomy_repo"),
                    "rviz",
                    "default.rviz",
                ]
            ),
            "use_sim_time": use_sim_time,
        }.items(),
    )
    
    # Relay RVIZ goal pose to navigation channel
    rviz_goal_relay = Node(
        executable="rviz_goal_relay.py",
        package="asl_tb3_lib",
        parameters=[
            {"output_channel": "/cmd_nav"},
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )
    
    # State publisher for turtlebot
    state_publisher = Node(
        executable="state_publisher.py",
        package="asl_tb3_lib",
        parameters=[
            {"use_sim_time": use_sim_time},
        ],
        output="screen",
    )
    
    # Navigator node (from your HW2)
    navigator = Node(
        executable="navigator.py",
        package="autonomy_repo",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",
    )
    
    # Frontier exploration node
    frontier_exploration = Node(
        executable="frontier_exploration.py",
        package="autonomy_repo",
        parameters=[{"use_sim_time": use_sim_time}],
        output="screen",  # Show logs in terminal
    )
    
    # Optionally delay the exploration node to allow map initialization
    # This gives time for the map to build up a bit before starting exploration
    delayed_exploration = TimerAction(
        period=exploration_delay,
        actions=[frontier_exploration],
    )
    
    return LaunchDescription(
        [
            # Launch arguments
            declare_use_sim_time,
            declare_exploration_delay,
            
            # Launch nodes
            rviz_launch,
            state_publisher,
            rviz_goal_relay,
            navigator,
            delayed_exploration,
        ]
    )
