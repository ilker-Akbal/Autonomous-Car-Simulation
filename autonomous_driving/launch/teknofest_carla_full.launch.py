from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition, UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def lane_follower_parameters(lane_plan_topic, carla_root, host, port, log_root, log_session_id):
    return {
        "route_topic": "/adas/planning/route",
        "mission_topic": "/adas/teknofest/mission",
        "status_topic": "/adas/carla/status",
        "lane_plan_topic": lane_plan_topic,
        "carla_root": carla_root,
        "host": host,
        "port": port,
        "timeout": 120.0,
        "cruise_speed_mps": 5.2,
        "turn_speed_mps": 3.2,
        "approach_speed_mps": 2.0,
        "min_drive_speed_mps": 2.2,
        "lookahead_base_m": 4.5,
        "lookahead_gain": 0.55,
        "lookahead_min_m": 3.5,
        "lookahead_max_m": 10.0,
        "startup_duration_s": 5.0,
        "startup_lane_lock_s": 3.0,
        "startup_speed_mps": 2.2,
        "unstable_lane_speed_mps": 2.8,
        "startup_lane_target_m": 3.2,
        "lane_target_jump_threshold_m": 6.0,
        "heading_error_slowdown_deg": 15.0,
        "cross_track_slowdown_m": 0.5,
        "upcoming_turn_lookahead_m": 25.0,
        "turn_slowdown_start_m": 22.0,
        "turn_speed_limit_mps": 2.8,
        "approach_turn_speed_mps": 2.8,
        "junction_turn_speed_mps": 2.2,
        "exit_turn_speed_mps": 2.6,
        "post_turn_speed_mps": 3.2,
        "hard_alignment_speed_mps": 1.8,
        "target_forward_min_m": 1.0,
        "target_jump_reject_m": 5.0,
        "junction_lane_change_distance_m": 8.0,
        "post_turn_stabilize_s": 1.2,
        "cruise_cte_threshold_m": 0.35,
        "recovery_cte_threshold_m": 0.70,
        "recovery_heading_threshold_deg": 30.0,
        "recovery_speed_mps": 1.7,
        "speed_setpoint_accel_mps2": 1.1,
        "speed_setpoint_decel_mps2": 1.0,
        "log_root": log_root,
        "log_session_id": log_session_id,
    }


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    mission_geojson = LaunchConfiguration("mission_geojson")
    round_name = LaunchConfiguration("round_name")
    log_root = LaunchConfiguration("log_root")
    log_session_id = LaunchConfiguration("log_session_id")
    log_session_id_param = ParameterValue(log_session_id, value_type=str)
    enable_traffic_light_system = LaunchConfiguration("enable_traffic_light_system")
    traffic_light_model_path = LaunchConfiguration("traffic_light_model_path")
    traffic_light_classes_path = LaunchConfiguration("traffic_light_classes_path")
    comfort_decel = LaunchConfiguration("comfort_decel")
    max_decel = LaunchConfiguration("max_decel")
    safe_stop_buffer = LaunchConfiguration("safe_stop_buffer")
    reaction_margin = LaunchConfiguration("reaction_margin")
    post_light_ignore_s = LaunchConfiguration("post_light_ignore_s")
    roi_center_margin = LaunchConfiguration("roi_center_margin")
    route_corridor_width_m = LaunchConfiguration("route_corridor_width_m")
    traffic_light_debug_draw_stop_points = LaunchConfiguration("traffic_light_debug_draw_stop_points")
    traffic_light_debug_draw_stop_points_param = ParameterValue(
        traffic_light_debug_draw_stop_points,
        value_type=bool,
    )
    generated_log_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("round_name", default_value="round_3"),
        DeclareLaunchArgument("log_root", default_value="autonomous_driving/outputs/teknofest_sim_logs"),
        DeclareLaunchArgument("log_session_id", default_value=generated_log_session_id),
        DeclareLaunchArgument("enable_traffic_light_system", default_value="false"),
        DeclareLaunchArgument(
            "traffic_light_model_path",
            default_value="autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/best.pt",
        ),
        DeclareLaunchArgument(
            "traffic_light_classes_path",
            default_value="autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/classes.json",
        ),
        DeclareLaunchArgument("comfort_decel", default_value="1.4"),
        DeclareLaunchArgument("max_decel", default_value="3.0"),
        DeclareLaunchArgument("safe_stop_buffer", default_value="1.8"),
        DeclareLaunchArgument("reaction_margin", default_value="1.5"),
        DeclareLaunchArgument("post_light_ignore_s", default_value="3.0"),
        DeclareLaunchArgument("roi_center_margin", default_value="0.35"),
        DeclareLaunchArgument("route_corridor_width_m", default_value="4.5"),
        DeclareLaunchArgument("traffic_light_debug_draw_stop_points", default_value="true"),
        DeclareLaunchArgument(
            "mission_geojson",
            default_value="autonomous_driving/missions/teknofest_town03_competition_v4_tasks_only.geojson",
        ),

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
                "ego_role_name": "ego_vehicle",
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
                        "ego_role_name": "ego_vehicle",
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
            period=6.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_mission_node",
                    name="teknofest_mission_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "mission_geojson": mission_geojson,
                        "round_name": round_name,
                        "competition_mode": True,
                        "gnss_topic": "/adas/localization/gnss",
                        "mission_topic": "/adas/teknofest/mission",
                        "event_topic": "/adas/teknofest/events",
                        "point_pass_tolerance_m": 1.0,
                        "task_stop_tolerance_m": 1.0,
                        "distance_reference": "front_bumper",
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="route_planner_node",
                    name="route_planner_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "mission_topic": "/adas/teknofest/mission",
                        "route_topic": "/adas/planning/route",
                        "sampling_resolution_m": 1.2,
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.3,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="lane_follower_node",
                    name="lane_follower_node",
                    output="screen",
                    condition=UnlessCondition(enable_traffic_light_system),
                    parameters=[lane_follower_parameters(
                        "/adas/planning/lane_plan",
                        carla_root,
                        host,
                        port,
                        log_root,
                        log_session_id_param,
                    )],
                ),
                Node(
                    package="autonomous_driving",
                    executable="lane_follower_node",
                    name="lane_follower_node_tl_raw",
                    output="screen",
                    condition=IfCondition(enable_traffic_light_system),
                    parameters=[lane_follower_parameters(
                        "/adas/planning/lane_plan_raw",
                        carla_root,
                        host,
                        port,
                        log_root,
                        log_session_id_param,
                    )],
                ),
            ],
        ),

        TimerAction(
            period=7.45,
            actions=[
                LogInfo(
                    msg="TL SYSTEM: launching traffic_light_detector_node",
                    condition=IfCondition(enable_traffic_light_system),
                ),
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_detector_node",
                    name="traffic_light_detector_node",
                    output="screen",
                    emulate_tty=True,
                    condition=IfCondition(enable_traffic_light_system),
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "image_topic": "/adas/camera/front/image_raw",
                        "traffic_light_topic": "/adas/perception/traffic_lights",
                        "traffic_light_model_path": traffic_light_model_path,
                        "traffic_light_classes_path": traffic_light_classes_path,
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.50,
            actions=[
                LogInfo(
                    msg="TL SYSTEM: launching traffic_light_stopline_detector_node",
                    condition=IfCondition(enable_traffic_light_system),
                ),
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_stopline_detector_node",
                    name="traffic_light_stopline_detector_node",
                    output="screen",
                    emulate_tty=True,
                    condition=IfCondition(enable_traffic_light_system),
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "depth_topic": "/zed/zed_node/depth/depth_registered",
                        "stopline_topic": "/adas/perception/traffic_light_stopline",
                        "output_topic": "/adas/perception/traffic_light_stopline",
                        "debug_image_topic": "/adas/debug/traffic_light_stopline_image",
                        "camera_width": 640,
                        "camera_height": 360,
                        "camera_fov_deg": 72.0,
                        "camera_height_m": 2.25,
                        "camera_pitch_deg": -1.0,
                        "front_bumper_offset_m": 1.35,
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.55,
            actions=[
                LogInfo(
                    msg="TL SYSTEM: launching traffic_light_manager_node",
                    condition=IfCondition(enable_traffic_light_system),
                ),
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_manager_node",
                    name="traffic_light_manager_node",
                    output="screen",
                    emulate_tty=True,
                    condition=IfCondition(enable_traffic_light_system),
                    parameters=[{
                        "lane_plan_input_topic": "/adas/planning/lane_plan_raw",
                        "lane_plan_output_topic": "/adas/planning/lane_plan",
                        "traffic_light_topic": "/adas/perception/traffic_lights",
                        "stopline_topic": "/adas/perception/traffic_light_stopline",
                        "route_topic": "/adas/planning/route",
                        "status_topic": "/adas/carla/status",
                        "comfort_decel": comfort_decel,
                        "max_decel": max_decel,
                        "safe_stop_buffer": safe_stop_buffer,
                        "reaction_margin": reaction_margin,
                        "post_light_ignore_s": post_light_ignore_s,
                        "roi_center_margin": roi_center_margin,
                        "route_corridor_width_m": route_corridor_width_m,
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "traffic_light_debug_draw_stop_points": traffic_light_debug_draw_stop_points_param,
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.6,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="control_node",
                    name="control_node",
                    output="screen",
                    parameters=[{
                        "lane_plan_topic": "/adas/planning/lane_plan",
                        "status_topic": "/adas/carla/status",
                        "vehicle_command_topic": "/adas/control/vehicle_command",
                        "plan_timeout_s": 1.8,
                        "status_timeout_s": 3.0,
                        "max_throttle": 0.62,
                        "launch_throttle": 0.24,
                        "max_steer_delta_per_s": 1.4,
                        "steer_low_pass_alpha": 0.35,
                        "startup_steer_limit_s": 5.0,
                        "startup_max_steer": 0.14,
                        "startup_lane_jump_max_steer": 0.08,
                        "high_heading_max_steer": 0.25,
                        "high_heading_limit_deg": 25.0,
                        "turn_max_steer": 0.28,
                        "junction_turn_max_steer": 0.36,
                        "startup_throttle_limit_s": 3.0,
                        "startup_max_throttle": 0.28,
                        "low_speed_turn_max_throttle": 0.40,
                        "coast_overspeed_margin_mps": 0.35,
                        "speed_setpoint_accel_mps2": 1.1,
                        "speed_setpoint_decel_mps2": 1.0,
                        "throttle_slew_rate_per_s": 1.0,
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="carla_control_adapter_node",
                    name="carla_control_adapter_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "vehicle_command_topic": "/adas/control/vehicle_command",
                        "command_hold_s": 0.7,
                        "command_timeout_s": 3.0,
                        "log_root": log_root,
                        "log_session_id": log_session_id_param,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=8.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="teknofest_spectator_follow_node",
                    name="teknofest_spectator_follow_node",
                    output="screen",
                    parameters=[{
                        "host": host,
                        "port": port,
                        "distance_m": 8.0,
                        "height_m": 3.2,
                        "target_forward_m": 3.0,
                        "target_height_m": 1.2,
                        "timer_period_sec": 0.05,
                    }],
                ),
            ],
        ),
    ])
