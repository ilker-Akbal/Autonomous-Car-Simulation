from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    ego_role_name = LaunchConfiguration("ego_role_name")

    enable_camera_follow = LaunchConfiguration("enable_camera_follow")
    camera_view = LaunchConfiguration("camera_view")
    camera_follow_rate_hz = LaunchConfiguration("camera_follow_rate_hz")
    camera_distance_m = LaunchConfiguration("camera_distance_m")
    camera_height_m = LaunchConfiguration("camera_height_m")
    camera_pitch_deg = LaunchConfiguration("camera_pitch_deg")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("ego_role_name", default_value="ego_vehicle"),

        DeclareLaunchArgument("enable_camera_follow", default_value="true"),
        DeclareLaunchArgument("camera_view", default_value="chase"),
        DeclareLaunchArgument("camera_follow_rate_hz", default_value="60.0"),
        DeclareLaunchArgument("camera_distance_m", default_value="7.0"),
        DeclareLaunchArgument("camera_height_m", default_value="3.2"),
        DeclareLaunchArgument("camera_pitch_deg", default_value="-12.0"),

        LogInfo(msg="Minimal CARLA sensor skeleton started"),
        LogInfo(msg="Decision, route, control, traffic-light stop/go and WASD nodes are removed"),

        Node(
            package="autonomous_driving",
            executable="carla_world_manager_node",
            name="carla_world_manager_node",
            output="screen",
            parameters=[{
                "carla_root": carla_root,
                "host": host,
                "port": port,
                "town": town,
                "timeout": 120.0,
                "ego_role_name": ego_role_name,
                "status_period_s": 0.1,
            }],
        ),

        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="carla_sensor_bridge_node",
                    name="carla_sensor_bridge_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": ego_role_name,
                        "camera_width": 640,
                        "camera_height": 360,
                        "camera_fov": 72.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.25,
                        "camera_pitch": -1.0,
                        "zed_enabled": True,
                        "depth_enabled": True,
                        "lidar_enabled": True,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="viewport_camera_follow_node",
                    name="viewport_camera_follow_node",
                    output="screen",
                    condition=IfCondition(enable_camera_follow),
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "ego_role_name": ego_role_name,
                        "camera_view": camera_view,
                        "camera_follow_rate_hz": camera_follow_rate_hz,
                        "camera_distance_m": camera_distance_m,
                        "camera_height_m": camera_height_m,
                        "camera_pitch_deg": camera_pitch_deg,
                    }],
                ),
            ],
        ),
    ])
