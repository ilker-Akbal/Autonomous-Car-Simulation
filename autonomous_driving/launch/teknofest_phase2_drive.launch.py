from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    # Phase 1 sensor args (lite defaults)
    zed_enabled = LaunchConfiguration("zed_enabled")
    depth_enabled = LaunchConfiguration("depth_enabled")
    lidar_enabled = LaunchConfiguration("lidar_enabled")
    zed_point_cloud_enabled = LaunchConfiguration("zed_point_cloud_enabled")
    front_rgb_separate_enabled = LaunchConfiguration("front_rgb_separate_enabled")
    front_rgb_from_zed_left = LaunchConfiguration("front_rgb_from_zed_left")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_sensor_tick = LaunchConfiguration("camera_sensor_tick")

    enable_sync_mode = LaunchConfiguration("enable_sync_mode")
    auto_tick_sync_world = LaunchConfiguration("auto_tick_sync_world")
    fixed_delta_seconds = LaunchConfiguration("fixed_delta_seconds")
    tick_rate_hz = LaunchConfiguration("tick_rate_hz")

    # Phase2 args
    target_speed_mps = LaunchConfiguration("target_speed_mps")
    cruise_speed_mps = LaunchConfiguration("cruise_speed_mps")
    max_speed_mps = LaunchConfiguration("max_speed_mps")
    min_turn_speed_mps = LaunchConfiguration("min_turn_speed_mps")
    sharp_turn_yaw_deg = LaunchConfiguration("sharp_turn_yaw_deg")
    moderate_turn_yaw_deg = LaunchConfiguration("moderate_turn_yaw_deg")
    speed_slew_up_mps_per_s = LaunchConfiguration("speed_slew_up_mps_per_s")
    speed_slew_down_mps_per_s = LaunchConfiguration("speed_slew_down_mps_per_s")
    dynamic_lookahead_enabled = LaunchConfiguration("dynamic_lookahead_enabled")
    base_lookahead_m = LaunchConfiguration("base_lookahead_m")
    lookahead_gain = LaunchConfiguration("lookahead_gain")
    min_lookahead_m = LaunchConfiguration("min_lookahead_m")
    min_speed_for_throttle_floor_mps = LaunchConfiguration("min_speed_for_throttle_floor_mps")
    throttle_floor_when_moving = LaunchConfiguration("throttle_floor_when_moving")
    uphill_speed_error_boost = LaunchConfiguration("uphill_speed_error_boost")
    throttle_slew_limit = LaunchConfiguration("throttle_slew_limit")
    integral_limit = LaunchConfiguration("integral_limit")
    route_horizon_m = LaunchConfiguration("route_horizon_m")
    route_step_m = LaunchConfiguration("route_step_m")
    enable_phase2_drive = LaunchConfiguration("enable_phase2_drive")
    max_throttle = LaunchConfiguration("max_throttle")
    max_brake = LaunchConfiguration("max_brake")


    # Phase 2B tuning LaunchConfigurations
    max_lookahead_m = LaunchConfiguration("max_lookahead_m")


    # Auto-added missing launch configurations
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    ego_role_name = LaunchConfiguration("ego_role_name")

    return LaunchDescription([
        DeclareLaunchArgument("zed_enabled", default_value="false"),
        DeclareLaunchArgument("depth_enabled", default_value="false"),
        DeclareLaunchArgument("lidar_enabled", default_value="false"),
        DeclareLaunchArgument("zed_point_cloud_enabled", default_value="false"),
        DeclareLaunchArgument("front_rgb_separate_enabled", default_value="true"),
        DeclareLaunchArgument("front_rgb_from_zed_left", default_value="false"),
        DeclareLaunchArgument("camera_width", default_value="320"),
        DeclareLaunchArgument("camera_height", default_value="180"),
        DeclareLaunchArgument("camera_sensor_tick", default_value="0.1"),

        DeclareLaunchArgument("enable_sync_mode", default_value="true"),
        DeclareLaunchArgument("auto_tick_sync_world", default_value="true"),
        DeclareLaunchArgument("fixed_delta_seconds", default_value="0.05"),
        DeclareLaunchArgument("tick_rate_hz", default_value="20.0"),

        DeclareLaunchArgument("target_speed_mps", default_value="2.5"),
        DeclareLaunchArgument("cruise_speed_mps", default_value="4.5"),
        DeclareLaunchArgument("max_speed_mps", default_value="6.0"),
        DeclareLaunchArgument("min_turn_speed_mps", default_value="2.0"),
        DeclareLaunchArgument("sharp_turn_yaw_deg", default_value="45.0"),
        DeclareLaunchArgument("moderate_turn_yaw_deg", default_value="18.0"),
        DeclareLaunchArgument("speed_slew_up_mps_per_s", default_value="0.8"),
        DeclareLaunchArgument("speed_slew_down_mps_per_s", default_value="2.0"),
        DeclareLaunchArgument("dynamic_lookahead_enabled", default_value="true"),
        DeclareLaunchArgument("min_lookahead_m", default_value="4.5"),
        DeclareLaunchArgument("max_lookahead_m", default_value="14.0"),
        DeclareLaunchArgument("base_lookahead_m", default_value="5.0"),
        DeclareLaunchArgument("lookahead_gain", default_value="1.2"),
        DeclareLaunchArgument("min_speed_for_throttle_floor_mps", default_value="0.5"),
        DeclareLaunchArgument("throttle_floor_when_moving", default_value="0.12"),
        DeclareLaunchArgument("uphill_speed_error_boost", default_value="0.10"),
        DeclareLaunchArgument("throttle_slew_limit", default_value="0.04"),
        DeclareLaunchArgument("integral_limit", default_value="3.0"),
        DeclareLaunchArgument("enable_phase2_drive", default_value="true"),
        DeclareLaunchArgument("max_throttle", default_value="0.45"),
        DeclareLaunchArgument("max_brake", default_value="0.75"),

        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("ego_role_name", default_value="ego_vehicle"),
        DeclareLaunchArgument("route_horizon_m", default_value="80.0"),
        DeclareLaunchArgument("route_step_m", default_value="2.0"),

        LogInfo(msg="Starting Phase 2 minimal drive stack"),

        # Start Phase 1 nodes (embedded here to ensure proper ordering)
        Node(
            package="autonomous_driving",
            executable="carla_world_manager_node",
            name="carla_world_manager_node",
            output="screen",
            parameters=[{
                "timeout": 120.0,
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
                        "camera_width": ParameterValue(camera_width, value_type=int),
                        "camera_height": ParameterValue(camera_height, value_type=int),
                        "camera_sensor_tick": ParameterValue(camera_sensor_tick, value_type=float),
                        "zed_enabled": ParameterValue(zed_enabled, value_type=bool),
                        "depth_enabled": ParameterValue(depth_enabled, value_type=bool),
                        "lidar_enabled": ParameterValue(lidar_enabled, value_type=bool),
                        "zed_point_cloud_enabled": ParameterValue(zed_point_cloud_enabled, value_type=bool),
                        "front_rgb_separate_enabled": ParameterValue(front_rgb_separate_enabled, value_type=bool),
                        "front_rgb_from_zed_left": ParameterValue(front_rgb_from_zed_left, value_type=bool),
                    }],
                ),
                Node(
                    package="autonomous_driving",
                    executable="teknofest_diagnostics_node",
                    name="teknofest_diagnostics_node",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                ),
                Node(
                    package="autonomous_driving",
                    executable="ekf_localizer_node",
                    name="ekf_localizer_node",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                ),
            ],
        ),

        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="simple_route_planner_node",
                    name="simple_route_planner",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "horizon_m": ParameterValue(route_horizon_m, value_type=float),
                        "step_m": ParameterValue(route_step_m, value_type=float),
                        "rate_hz": 5.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=9.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="lane_follower_node",
                    name="lane_follower",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "rate_hz": 20.0,
                        "target_speed_mps": ParameterValue(target_speed_mps, value_type=float),
                        "cruise_speed_mps": ParameterValue(cruise_speed_mps, value_type=float),
                        "max_speed_mps": ParameterValue(max_speed_mps, value_type=float),
                        "min_turn_speed_mps": ParameterValue(min_turn_speed_mps, value_type=float),
                        "sharp_turn_yaw_deg": ParameterValue(sharp_turn_yaw_deg, value_type=float),
                        "moderate_turn_yaw_deg": ParameterValue(moderate_turn_yaw_deg, value_type=float),
                        "speed_slew_up_mps_per_s": ParameterValue(speed_slew_up_mps_per_s, value_type=float),
                        "speed_slew_down_mps_per_s": ParameterValue(speed_slew_down_mps_per_s, value_type=float),
                        "dynamic_lookahead_enabled": ParameterValue(dynamic_lookahead_enabled, value_type=bool),
                        "base_lookahead_m": ParameterValue(base_lookahead_m, value_type=float),
                        "lookahead_gain": ParameterValue(lookahead_gain, value_type=float),
                        "min_lookahead_m": ParameterValue(min_lookahead_m, value_type=float),
                        "max_lookahead_m": ParameterValue(max_lookahead_m, value_type=float),
                    }],
                ),
            ],
        ),

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="control_node",
                    name="control_node",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "max_throttle": ParameterValue(max_throttle, value_type=float),
                        "max_brake": ParameterValue(max_brake, value_type=float),
                        "throttle_floor_when_moving": ParameterValue(throttle_floor_when_moving, value_type=float),
                        "uphill_speed_error_boost": ParameterValue(uphill_speed_error_boost, value_type=float),
                        "min_speed_for_throttle_floor_mps": ParameterValue(min_speed_for_throttle_floor_mps, value_type=float),
                        "throttle_slew_limit": ParameterValue(throttle_slew_limit, value_type=float),
                        "integral_limit": ParameterValue(integral_limit, value_type=float),
                    }],
                ),
                Node(
                    package="autonomous_driving",
                    executable="carla_control_adapter_node",
                    name="carla_control_adapter_node",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                ),
            ],
        ),
    ])
