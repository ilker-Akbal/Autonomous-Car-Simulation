from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    ego_role_name = LaunchConfiguration("ego_role_name")

    enable_camera_follow = LaunchConfiguration("enable_camera_follow")
    enable_lane_detector = LaunchConfiguration("enable_lane_detector")
    enable_ekf_localizer = LaunchConfiguration("enable_ekf_localizer")
    enable_diagnostics = LaunchConfiguration("enable_diagnostics")
    camera_view = LaunchConfiguration("camera_view")
    camera_follow_rate_hz = LaunchConfiguration("camera_follow_rate_hz")
    camera_distance_m = LaunchConfiguration("camera_distance_m")
    camera_height_m = LaunchConfiguration("camera_height_m")
    camera_pitch_deg = LaunchConfiguration("camera_pitch_deg")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_sensor_tick = LaunchConfiguration("camera_sensor_tick")
    front_rgb_separate_enabled = LaunchConfiguration("front_rgb_separate_enabled")
    front_rgb_from_zed_left = LaunchConfiguration("front_rgb_from_zed_left")
    zed_enabled = LaunchConfiguration("zed_enabled")
    depth_enabled = LaunchConfiguration("depth_enabled")
    lidar_enabled = LaunchConfiguration("lidar_enabled")
    zed_point_cloud_enabled = LaunchConfiguration("zed_point_cloud_enabled")
    lidar_points_per_second = LaunchConfiguration("lidar_points_per_second")
    lidar_sensor_tick = LaunchConfiguration("lidar_sensor_tick")
    lidar_channels = LaunchConfiguration("lidar_channels")
    enable_sync_mode = LaunchConfiguration("enable_sync_mode")
    auto_tick_sync_world = LaunchConfiguration("auto_tick_sync_world")
    fixed_delta_seconds = LaunchConfiguration("fixed_delta_seconds")
    tick_rate_hz = LaunchConfiguration("tick_rate_hz")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("ego_role_name", default_value="ego_vehicle"),

        DeclareLaunchArgument("enable_camera_follow", default_value="true"),
        DeclareLaunchArgument("enable_lane_detector", default_value="true"),
        DeclareLaunchArgument("enable_ekf_localizer", default_value="true"),
        DeclareLaunchArgument("enable_diagnostics", default_value="true"),
        DeclareLaunchArgument("enable_sync_mode", default_value="true"),
        DeclareLaunchArgument("auto_tick_sync_world", default_value="true"),
        DeclareLaunchArgument("fixed_delta_seconds", default_value="0.05"),
        DeclareLaunchArgument("tick_rate_hz", default_value="20.0"),
        DeclareLaunchArgument("zed_enabled", default_value="false"),
        DeclareLaunchArgument("depth_enabled", default_value="false"),
        DeclareLaunchArgument("lidar_enabled", default_value="false"),
        DeclareLaunchArgument("zed_point_cloud_enabled", default_value="false"),
        DeclareLaunchArgument("front_rgb_separate_enabled", default_value="true"),
        DeclareLaunchArgument("front_rgb_from_zed_left", default_value="false"),
        DeclareLaunchArgument("camera_width", default_value="320"),
        DeclareLaunchArgument("camera_height", default_value="180"),
        DeclareLaunchArgument("camera_sensor_tick", default_value="0.1"),
        DeclareLaunchArgument("lidar_points_per_second", default_value="120000"),
        DeclareLaunchArgument("lidar_sensor_tick", default_value="0.2"),
        DeclareLaunchArgument("lidar_channels", default_value="16"),

        DeclareLaunchArgument("camera_view", default_value="chase"),
        DeclareLaunchArgument("camera_follow_rate_hz", default_value="60.0"),
        DeclareLaunchArgument("camera_distance_m", default_value="7.0"),
        DeclareLaunchArgument("camera_height_m", default_value="3.2"),
        DeclareLaunchArgument("camera_pitch_deg", default_value="-12.0"),

        LogInfo(msg="Phase 1 CARLA sensor + perception + localization stack starting."),

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
                "enable_sync_mode": ParameterValue(enable_sync_mode, value_type=bool),
                "auto_tick_sync_world": ParameterValue(auto_tick_sync_world, value_type=bool),
                "fixed_delta_seconds": ParameterValue(fixed_delta_seconds, value_type=float),
                "tick_rate_hz": ParameterValue(tick_rate_hz, value_type=float),
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
                        "camera_width": ParameterValue(camera_width, value_type=int),
                        "camera_height": ParameterValue(camera_height, value_type=int),
                        "camera_fov": 72.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.25,
                        "camera_pitch": -1.0,
                        "camera_sensor_tick": ParameterValue(camera_sensor_tick, value_type=float),
                        "zed_enabled": ParameterValue(zed_enabled, value_type=bool),
                        "depth_enabled": ParameterValue(depth_enabled, value_type=bool),
                        "lidar_enabled": ParameterValue(lidar_enabled, value_type=bool),
                        "zed_point_cloud_enabled": ParameterValue(zed_point_cloud_enabled, value_type=bool),
                        "front_rgb_separate_enabled": ParameterValue(front_rgb_separate_enabled, value_type=bool),
                        "front_rgb_from_zed_left": ParameterValue(front_rgb_from_zed_left, value_type=bool),
                        "lidar_points_per_second": ParameterValue(lidar_points_per_second, value_type=int),
                        "lidar_sensor_tick": ParameterValue(lidar_sensor_tick, value_type=float),
                        "lidar_channels": ParameterValue(lidar_channels, value_type=int),
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

        Node(
            package="autonomous_driving",
            executable="teknofest_diagnostics_node",
            name="teknofest_diagnostics_node",
            output="screen",
            condition=IfCondition(enable_diagnostics),
        ),

        Node(
            package="autonomous_driving",
            executable="lane_detector_node",
            name="lane_detector_node",
            output="screen",
            condition=IfCondition(enable_lane_detector),
        ),

        Node(
            package="autonomous_driving",
            executable="ekf_localizer_node",
            name="ekf_localizer_node",
            output="screen",
            condition=IfCondition(enable_ekf_localizer),
        ),
    ])
