import os
from datetime import datetime
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, LogInfo, SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node
from launch.conditions import IfCondition


def generate_launch_description():
    package_root = Path(__file__).resolve().parent.parent
    log_session_id = f"phase2_drive_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
    log_root = package_root / "outputs" / "teknofest_sim_logs"
    default_mission_geojson = package_root / "missions" / "teknofest_town03_competition_v4_tasks_only.geojson"
    if not default_mission_geojson.exists():
        default_mission_geojson = package_root / "missions" / "teknofest_round3.geojson"
    default_sign_plan_geojson = package_root / "missions" / "town03_competition_v4_sign_plan.geojson"
    default_sign_plan_json = package_root / "missions" / "town03_competition_v4_sign_plan.json"
    default_slalom_plan_json = package_root / "config" / "town03_round3_slalom_plan.json"
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
    enable_viewport_camera_follow = LaunchConfiguration("enable_viewport_camera_follow")
    viewport_camera_follow_distance_m = LaunchConfiguration("follow_distance_m")
    viewport_camera_follow_height_m = LaunchConfiguration("follow_height_m")
    viewport_camera_follow_pitch_deg = LaunchConfiguration("follow_pitch_deg")
    viewport_camera_follow_update_hz = LaunchConfiguration("follow_update_hz")
    viewport_camera_carla_host = LaunchConfiguration("carla_host")
    viewport_camera_carla_port = LaunchConfiguration("carla_port")
    enable_demo_weather = LaunchConfiguration("enable_demo_weather")
    demo_weather_preset = LaunchConfiguration("demo_weather_preset")
    enable_ego_lights = LaunchConfiguration("enable_ego_lights")
    enable_ego_label = LaunchConfiguration("enable_ego_label")
    ego_label_text = LaunchConfiguration("ego_label_text")

    enable_sync_mode = LaunchConfiguration("enable_sync_mode")
    auto_tick_sync_world = LaunchConfiguration("auto_tick_sync_world")
    fixed_delta_seconds = LaunchConfiguration("fixed_delta_seconds")
    tick_rate_hz = LaunchConfiguration("tick_rate_hz")

    # Phase2 args
    target_speed_mps = LaunchConfiguration("target_speed_mps")
    cruise_speed_mps = LaunchConfiguration("cruise_speed_mps")
    max_speed_mps = LaunchConfiguration("max_speed_mps")
    min_turn_speed_mps = LaunchConfiguration("min_turn_speed_mps")
    speed_boost_enabled = LaunchConfiguration("speed_boost_enabled")
    nominal_speed_boost_mps = LaunchConfiguration("nominal_speed_boost_mps")
    sharp_turn_yaw_deg = LaunchConfiguration("sharp_turn_yaw_deg")
    moderate_turn_yaw_deg = LaunchConfiguration("moderate_turn_yaw_deg")
    speed_slew_up_mps_per_s = LaunchConfiguration("speed_slew_up_mps_per_s")
    speed_slew_down_mps_per_s = LaunchConfiguration("speed_slew_down_mps_per_s")
    dynamic_lookahead_enabled = LaunchConfiguration("dynamic_lookahead_enabled")
    base_lookahead_m = LaunchConfiguration("base_lookahead_m")
    lookahead_gain = LaunchConfiguration("lookahead_gain")
    lane_assist_only = LaunchConfiguration("lane_assist_only")
    right_lane_lateral_bias_enabled = LaunchConfiguration("right_lane_lateral_bias_enabled")
    right_lane_lateral_bias_m = LaunchConfiguration("right_lane_lateral_bias_m")
    right_lane_lateral_bias_min_speed_mps = LaunchConfiguration("right_lane_lateral_bias_min_speed_mps")
    right_lane_lateral_bias_disable_in_junction = LaunchConfiguration("right_lane_lateral_bias_disable_in_junction")
    right_lane_lateral_bias_disable_in_turn = LaunchConfiguration("right_lane_lateral_bias_disable_in_turn")
    right_lane_lateral_bias_safety_margin_m = LaunchConfiguration("right_lane_lateral_bias_safety_margin_m")
    vehicle_half_width_m = LaunchConfiguration("vehicle_half_width_m")
    lane_departure_guard_enabled = LaunchConfiguration("lane_departure_guard_enabled")
    lane_departure_lateral_threshold_m = LaunchConfiguration("lane_departure_lateral_threshold_m")
    route_recovery_speed_mps = LaunchConfiguration("route_recovery_speed_mps")
    route_conflict_heading_threshold_deg = LaunchConfiguration("route_conflict_heading_threshold_deg")
    route_index_hysteresis_enabled = LaunchConfiguration("route_index_hysteresis_enabled")
    max_route_index_jump = LaunchConfiguration("max_route_index_jump")
    steering_rate_limit_enabled = LaunchConfiguration("steering_rate_limit_enabled")
    max_steer_delta = LaunchConfiguration("max_steer_delta")
    min_nonzero_target_speed_mps = LaunchConfiguration("min_nonzero_target_speed_mps")
    task_pull_over_start_distance_m = LaunchConfiguration("task_pull_over_start_distance_m")
    task_pull_over_final_distance_m = LaunchConfiguration("task_pull_over_final_distance_m")
    task_pull_over_lateral_offset_m = LaunchConfiguration("task_pull_over_lateral_offset_m")
    task_stop_reached_distance_m = LaunchConfiguration("task_stop_reached_distance_m")
    task_stop_final_phase_latch_enabled = LaunchConfiguration("task_stop_final_phase_latch_enabled")
    task_stop_final_latch_distance_m = LaunchConfiguration("task_stop_final_latch_distance_m")
    task_stop_overshoot_guard_distance_m = LaunchConfiguration("task_stop_overshoot_guard_distance_m")
    task_stop_overshoot_guard_speed_mps = LaunchConfiguration("task_stop_overshoot_guard_speed_mps")
    task_stop_alignment_enabled = LaunchConfiguration("task_stop_alignment_enabled")
    task_stop_alignment_start_distance_m = LaunchConfiguration("task_stop_alignment_start_distance_m")
    task_stop_alignment_yaw_tolerance_deg = LaunchConfiguration("task_stop_alignment_yaw_tolerance_deg")
    task_stop_alignment_speed_mps = LaunchConfiguration("task_stop_alignment_speed_mps")
    task_stop_alignment_target_ahead_m = LaunchConfiguration("task_stop_alignment_target_ahead_m")
    task_stop_approach_cruise_speed_mps = LaunchConfiguration("task_stop_approach_cruise_speed_mps")
    task_stop_pre_align_speed_mps = LaunchConfiguration("task_stop_pre_align_speed_mps")
    task_stop_final_align_speed_mps = LaunchConfiguration("task_stop_final_align_speed_mps")
    task_stop_min_creep_speed_mps = LaunchConfiguration("task_stop_min_creep_speed_mps")
    task_stop_no_stop_before_final_distance_m = LaunchConfiguration("task_stop_no_stop_before_final_distance_m")
    task_stop_phase_hysteresis_m = LaunchConfiguration("task_stop_phase_hysteresis_m")
    task_pull_over_approach_speed_mps = LaunchConfiguration("task_pull_over_approach_speed_mps")
    task_pull_over_final_speed_mps = LaunchConfiguration("task_pull_over_final_speed_mps")
    task_pull_over_crawl_speed_mps = LaunchConfiguration("task_pull_over_crawl_speed_mps")
    task_pull_over_keep_bias_until_reached = LaunchConfiguration("task_pull_over_keep_bias_until_reached")
    min_lookahead_m = LaunchConfiguration("min_lookahead_m")
    min_speed_for_throttle_floor_mps = LaunchConfiguration("min_speed_for_throttle_floor_mps")
    throttle_floor_when_moving = LaunchConfiguration("throttle_floor_when_moving")
    uphill_speed_error_boost = LaunchConfiguration("uphill_speed_error_boost")
    throttle_slew_limit = LaunchConfiguration("throttle_slew_limit")
    integral_limit = LaunchConfiguration("integral_limit")
    route_horizon_m = LaunchConfiguration("route_horizon_m")
    route_step_m = LaunchConfiguration("route_step_m")
    local_route_horizon_m = LaunchConfiguration("local_route_horizon_m")
    global_sampling_resolution_m = LaunchConfiguration("global_sampling_resolution_m")
    global_route_publish_hz = LaunchConfiguration("global_route_publish_hz")
    replan_distance_threshold_m = LaunchConfiguration("replan_distance_threshold_m")
    right_lane_policy_enabled = LaunchConfiguration("right_lane_policy_enabled")
    preferred_lane_side = LaunchConfiguration("preferred_lane_side")
    right_lane_projection_max_distance_m = LaunchConfiguration("right_lane_projection_max_distance_m")
    right_lane_policy_disable_in_junction = LaunchConfiguration("right_lane_policy_disable_in_junction")
    route_initial_wrong_way_reject_deg = LaunchConfiguration("route_initial_wrong_way_reject_deg")
    sign_constraints_enabled = LaunchConfiguration("sign_constraints_enabled")
    sign_plan_geojson = LaunchConfiguration("sign_plan_geojson")
    sign_plan_json = LaunchConfiguration("sign_plan_json")
    sign_constraint_effective_radius_m = LaunchConfiguration("sign_constraint_effective_radius_m")
    sign_constraint_debug = LaunchConfiguration("sign_constraint_debug")
    route_source_mode = LaunchConfiguration("route_source_mode")
    fallback_to_simple_forward_route = LaunchConfiguration("fallback_to_simple_forward_route")
    disable_fallback_driving = LaunchConfiguration("disable_fallback_driving")
    disable_fallback_driving_when_mission_missing = LaunchConfiguration("disable_fallback_driving_when_mission_missing")
    global_route_stale_timeout_s = LaunchConfiguration("global_route_stale_timeout_s")
    mission_goal_near_distance_m = LaunchConfiguration("mission_goal_near_distance_m")
    route_end_requires_goal_near = LaunchConfiguration("route_end_requires_goal_near")
    replan_when_local_route_exhausted = LaunchConfiguration("replan_when_local_route_exhausted")
    local_route_short_goal_far_replan_distance_m = LaunchConfiguration("local_route_short_goal_far_replan_distance_m")
    hold_last_route_s = LaunchConfiguration("hold_last_route_s")
    enable_route_events = LaunchConfiguration("enable_route_events")
    route_event_horizon_m = LaunchConfiguration("route_event_horizon_m")
    route_lateral_margin_m = LaunchConfiguration("route_lateral_margin_m")
    vehicle_follow_distance_m = LaunchConfiguration("vehicle_follow_distance_m")
    vehicle_stop_distance_m = LaunchConfiguration("vehicle_stop_distance_m")
    pedestrian_stop_distance_m = LaunchConfiguration("pedestrian_stop_distance_m")
    follow_time_gap_s = LaunchConfiguration("follow_time_gap_s")
    enable_traffic_light_events = LaunchConfiguration("enable_traffic_light_events")
    traffic_light_horizon_m = LaunchConfiguration("traffic_light_horizon_m")
    traffic_light_lateral_margin_m = LaunchConfiguration("traffic_light_lateral_margin_m")
    red_detection_horizon_m = LaunchConfiguration("red_detection_horizon_m")
    red_approach_distance_m = LaunchConfiguration("red_approach_distance_m")
    red_stop_distance_m = LaunchConfiguration("red_stop_distance_m")
    red_stop_trigger_base_m = LaunchConfiguration("red_stop_trigger_base_m")
    red_stop_trigger_max_m = LaunchConfiguration("red_stop_trigger_max_m")
    red_stop_trigger_speed_gain_s = LaunchConfiguration("red_stop_trigger_speed_gain_s")
    red_stop_trigger_speed_buffer_m = LaunchConfiguration("red_stop_trigger_speed_buffer_m")
    red_creep_distance_m = LaunchConfiguration("red_creep_distance_m")
    red_approach_speed_mps = LaunchConfiguration("red_approach_speed_mps")
    red_creep_speed_mps = LaunchConfiguration("red_creep_speed_mps")
    yellow_slow_distance_m = LaunchConfiguration("yellow_slow_distance_m")
    yellow_stop_distance_m = LaunchConfiguration("yellow_stop_distance_m")
    yellow_slow_speed_mps = LaunchConfiguration("yellow_slow_speed_mps")
    traffic_light_stop_buffer_m = LaunchConfiguration("traffic_light_stop_buffer_m")
    traffic_light_stop_front_bumper_offset_m = LaunchConfiguration("traffic_light_stop_front_bumper_offset_m")
    traffic_light_stop_line_buffer_m = LaunchConfiguration("traffic_light_stop_line_buffer_m")
    traffic_light_stop_distance_tolerance_m = LaunchConfiguration("traffic_light_stop_distance_tolerance_m")
    traffic_light_stop_debug_enabled = LaunchConfiguration("traffic_light_stop_debug_enabled")
    traffic_light_stop_anchor_forward_offset_m = LaunchConfiguration("traffic_light_stop_anchor_forward_offset_m")
    traffic_light_post_green_ignore_s = LaunchConfiguration("traffic_light_post_green_ignore_s")
    traffic_light_passed_ignore_distance_m = LaunchConfiguration("traffic_light_passed_ignore_distance_m")
    green_release_distance_m = LaunchConfiguration("green_release_distance_m")
    green_ignore_after_pass_m = LaunchConfiguration("green_ignore_after_pass_m")
    tl_hold_state_memory_s = LaunchConfiguration("tl_hold_state_memory_s")
    tl_lost_grace_s = LaunchConfiguration("tl_lost_grace_s")
    green_release_grace_s = LaunchConfiguration("green_release_grace_s")
    stopped_speed_threshold_mps = LaunchConfiguration("stopped_speed_threshold_mps")
    stopline_reached_distance_m = LaunchConfiguration("stopline_reached_distance_m")
    tl_stop_line_buffer_m = LaunchConfiguration("tl_stop_line_buffer_m")
    tl_decel_max_mps2 = LaunchConfiguration("tl_decel_max_mps2")
    tl_slow_speed_mps = LaunchConfiguration("tl_slow_speed_mps")
    tl_min_profile_speed_mps = LaunchConfiguration("tl_min_profile_speed_mps")
    tl_hard_stop_distance_m = LaunchConfiguration("tl_hard_stop_distance_m")
    tl_profile_horizon_m = LaunchConfiguration("tl_profile_horizon_m")
    tl_fence_width_m = LaunchConfiguration("tl_fence_width_m")
    yellow_pass_time_s = LaunchConfiguration("yellow_pass_time_s")
    stopped_vehicle_speed_mps = LaunchConfiguration("stopped_vehicle_speed_mps")
    stopped_vehicle_stop_distance_m = LaunchConfiguration("stopped_vehicle_stop_distance_m")
    red_approach_hard_stop_distance_m = LaunchConfiguration(
        "red_approach_hard_stop_distance_m"
    )
    mission_geojson = LaunchConfiguration("mission_geojson")
    target_reached_distance_m = LaunchConfiguration("target_reached_distance_m")
    mission_position_tolerance_m = LaunchConfiguration("mission_position_tolerance_m")
    mission_yaw_tolerance_deg = LaunchConfiguration("mission_yaw_tolerance_deg")
    front_bumper_offset_m = LaunchConfiguration("front_bumper_offset_m")
    task_stop_position_tolerance_m = LaunchConfiguration("task_stop_position_tolerance_m")
    task_stop_front_tolerance_m = LaunchConfiguration("task_stop_front_tolerance_m")
    task_stop_yaw_tolerance_deg = LaunchConfiguration("task_stop_yaw_tolerance_deg")
    task_stop_close_enough_distance_m = LaunchConfiguration("task_stop_close_enough_distance_m")
    task_stop_close_enough_ignore_yaw = LaunchConfiguration("task_stop_close_enough_ignore_yaw")
    task_stop_close_enough_max_yaw_error_deg = LaunchConfiguration("task_stop_close_enough_max_yaw_error_deg")
    task_stop_completion_yaw_tolerance_deg = LaunchConfiguration("task_stop_completion_yaw_tolerance_deg")
    task_stop_completion_position_tolerance_m = LaunchConfiguration("task_stop_completion_position_tolerance_m")
    task_stop_use_side_projection = LaunchConfiguration("task_stop_use_side_projection")
    task_stop_side_projection_lateral_m = LaunchConfiguration("task_stop_side_projection_lateral_m")
    task_stop_side_projection_forward_m = LaunchConfiguration("task_stop_side_projection_forward_m")
    task_stop_side_projection_clamp_to_road = LaunchConfiguration("task_stop_side_projection_clamp_to_road")
    task_stop_raw_override_enabled = LaunchConfiguration("task_stop_raw_override_enabled")
    task_stop_min_road_edge_clearance_m = LaunchConfiguration("task_stop_min_road_edge_clearance_m")
    task_stop_max_side_projection_m = LaunchConfiguration("task_stop_max_side_projection_m")
    task_pull_over_hold_requires_effective_stop = LaunchConfiguration("task_pull_over_hold_requires_effective_stop")
    pickup_hold_s = LaunchConfiguration("pickup_hold_s")
    dropoff_hold_s = LaunchConfiguration("dropoff_hold_s")
    loop_mission = LaunchConfiguration("loop_mission")
    mission_publish_rate_hz = LaunchConfiguration("mission_publish_rate_hz")
    min_route_points = LaunchConfiguration("min_route_points")
    enable_phase2_drive = LaunchConfiguration("enable_phase2_drive")
    max_throttle = LaunchConfiguration("max_throttle")
    max_brake = LaunchConfiguration("max_brake")
    enable_slalom = LaunchConfiguration("enable_slalom")
    slalom_plan_json = LaunchConfiguration("slalom_plan_json")
    slalom_start_side = LaunchConfiguration("slalom_start_side")
    slalom_clearance_m = LaunchConfiguration("slalom_clearance_m")
    slalom_speed_mps = LaunchConfiguration("slalom_speed_mps")
    slalom_activation_horizon_m = LaunchConfiguration("slalom_activation_horizon_m")
    slalom_route_corridor_m = LaunchConfiguration("slalom_route_corridor_m")


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
        DeclareLaunchArgument("speed_boost_enabled", default_value="true"),
        DeclareLaunchArgument("nominal_speed_boost_mps", default_value="2.0"),
        DeclareLaunchArgument("sharp_turn_yaw_deg", default_value="45.0"),
        DeclareLaunchArgument("moderate_turn_yaw_deg", default_value="18.0"),
        DeclareLaunchArgument("speed_slew_up_mps_per_s", default_value="0.8"),
        DeclareLaunchArgument("speed_slew_down_mps_per_s", default_value="2.0"),
        DeclareLaunchArgument("dynamic_lookahead_enabled", default_value="true"),
        DeclareLaunchArgument("min_lookahead_m", default_value="4.5"),
        DeclareLaunchArgument("max_lookahead_m", default_value="14.0"),
        DeclareLaunchArgument("base_lookahead_m", default_value="5.0"),
        DeclareLaunchArgument("lookahead_gain", default_value="1.2"),
        DeclareLaunchArgument("lane_assist_only", default_value="true"),
        DeclareLaunchArgument("right_lane_lateral_bias_enabled", default_value="true"),
        DeclareLaunchArgument("right_lane_lateral_bias_m", default_value="0.65"),
        DeclareLaunchArgument("right_lane_lateral_bias_min_speed_mps", default_value="0.5"),
        DeclareLaunchArgument("right_lane_lateral_bias_disable_in_junction", default_value="true"),
        DeclareLaunchArgument("right_lane_lateral_bias_disable_in_turn", default_value="true"),
        DeclareLaunchArgument("right_lane_lateral_bias_safety_margin_m", default_value="0.35"),
        DeclareLaunchArgument("vehicle_half_width_m", default_value="0.95"),
        DeclareLaunchArgument("lane_departure_guard_enabled", default_value="true"),
        DeclareLaunchArgument("lane_departure_lateral_threshold_m", default_value="1.2"),
        DeclareLaunchArgument("route_recovery_speed_mps", default_value="2.5"),
        DeclareLaunchArgument("route_conflict_heading_threshold_deg", default_value="35.0"),
        DeclareLaunchArgument("route_index_hysteresis_enabled", default_value="true"),
        DeclareLaunchArgument("max_route_index_jump", default_value="8"),
        DeclareLaunchArgument("steering_rate_limit_enabled", default_value="true"),
        DeclareLaunchArgument("max_steer_delta", default_value="0.08"),
        DeclareLaunchArgument("min_nonzero_target_speed_mps", default_value="1.2"),
        DeclareLaunchArgument("task_pull_over_start_distance_m", default_value="18.0"),
        DeclareLaunchArgument("task_pull_over_final_distance_m", default_value="5.0"),
        DeclareLaunchArgument("task_pull_over_lateral_offset_m", default_value="1.0"),
        DeclareLaunchArgument("task_stop_reached_distance_m", default_value="2.0"),
        DeclareLaunchArgument("task_stop_final_phase_latch_enabled", default_value="true"),
        DeclareLaunchArgument("task_stop_final_latch_distance_m", default_value="1.0"),
        DeclareLaunchArgument("task_stop_overshoot_guard_distance_m", default_value="0.75"),
        DeclareLaunchArgument("task_stop_overshoot_guard_speed_mps", default_value="0.0"),
        DeclareLaunchArgument("task_stop_alignment_enabled", default_value="true"),
        DeclareLaunchArgument("task_stop_alignment_start_distance_m", default_value="3.0"),
        DeclareLaunchArgument("task_stop_alignment_yaw_tolerance_deg", default_value="12.0"),
        DeclareLaunchArgument("task_stop_alignment_speed_mps", default_value="0.8"),
        DeclareLaunchArgument("task_stop_alignment_target_ahead_m", default_value="2.0"),
        DeclareLaunchArgument("task_stop_approach_cruise_speed_mps", default_value="2.0"),
        DeclareLaunchArgument("task_stop_pre_align_speed_mps", default_value="1.2"),
        DeclareLaunchArgument("task_stop_final_align_speed_mps", default_value="0.8"),
        DeclareLaunchArgument("task_stop_min_creep_speed_mps", default_value="0.6"),
        DeclareLaunchArgument("task_stop_no_stop_before_final_distance_m", default_value="1.0"),
        DeclareLaunchArgument("task_stop_phase_hysteresis_m", default_value="0.75"),
        DeclareLaunchArgument("task_pull_over_approach_speed_mps", default_value="2.0"),
        DeclareLaunchArgument("task_pull_over_final_speed_mps", default_value="1.2"),
        DeclareLaunchArgument("task_pull_over_crawl_speed_mps", default_value="0.6"),
        DeclareLaunchArgument("task_pull_over_keep_bias_until_reached", default_value="true"),
        DeclareLaunchArgument("min_speed_for_throttle_floor_mps", default_value="0.5"),
        DeclareLaunchArgument("throttle_floor_when_moving", default_value="0.12"),
        DeclareLaunchArgument("uphill_speed_error_boost", default_value="0.10"),
        DeclareLaunchArgument("throttle_slew_limit", default_value="0.04"),
        DeclareLaunchArgument("integral_limit", default_value="3.0"),
        DeclareLaunchArgument("enable_phase2_drive", default_value="true"),
        DeclareLaunchArgument("max_throttle", default_value="0.45"),
        DeclareLaunchArgument("max_brake", default_value="0.75"),
        DeclareLaunchArgument("enable_slalom", default_value="false"),
        DeclareLaunchArgument("slalom_plan_json", default_value=str(default_slalom_plan_json)),
        DeclareLaunchArgument("slalom_start_side", default_value="right"),
        DeclareLaunchArgument("slalom_clearance_m", default_value="1.5"),
        DeclareLaunchArgument("slalom_speed_mps", default_value="1.6"),
        DeclareLaunchArgument("slalom_activation_horizon_m", default_value="80.0"),
        DeclareLaunchArgument("slalom_route_corridor_m", default_value="6.0"),

        DeclareLaunchArgument(
            "carla_root",
            default_value="/home/ilker/simulators/CARLA_0.9.15",
        ),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("carla_host", default_value="127.0.0.1"),
        DeclareLaunchArgument("carla_port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("ego_role_name", default_value="ego_vehicle"),
        DeclareLaunchArgument("enable_viewport_camera_follow", default_value="true"),
        DeclareLaunchArgument("follow_distance_m", default_value="10.0"),
        DeclareLaunchArgument("follow_height_m", default_value="5.0"),
        DeclareLaunchArgument("follow_pitch_deg", default_value="-18.0"),
        DeclareLaunchArgument("follow_update_hz", default_value="30.0"),
        DeclareLaunchArgument("enable_demo_weather", default_value="true"),
        DeclareLaunchArgument("demo_weather_preset", default_value="clear_sunset"),
        DeclareLaunchArgument("enable_ego_lights", default_value="true"),
        DeclareLaunchArgument("enable_ego_label", default_value="true"),
        DeclareLaunchArgument("ego_label_text", default_value="ROTA TEKNOFEST"),
        DeclareLaunchArgument("route_horizon_m", default_value="80.0"),
        DeclareLaunchArgument("route_step_m", default_value="2.0"),
        DeclareLaunchArgument("local_route_horizon_m", default_value="80.0"),
        DeclareLaunchArgument("global_sampling_resolution_m", default_value="2.0"),
        DeclareLaunchArgument("global_route_publish_hz", default_value="1.0"),
        DeclareLaunchArgument("replan_distance_threshold_m", default_value="8.0"),
        DeclareLaunchArgument("right_lane_policy_enabled", default_value="true"),
        DeclareLaunchArgument("preferred_lane_side", default_value="right"),
        DeclareLaunchArgument("right_lane_projection_max_distance_m", default_value="3.5"),
        DeclareLaunchArgument("right_lane_policy_disable_in_junction", default_value="true"),
        DeclareLaunchArgument("route_initial_wrong_way_reject_deg", default_value="120.0"),
        DeclareLaunchArgument("sign_constraints_enabled", default_value="false"),
        DeclareLaunchArgument("sign_plan_geojson", default_value=str(default_sign_plan_geojson)),
        DeclareLaunchArgument("sign_plan_json", default_value=str(default_sign_plan_json)),
        DeclareLaunchArgument("sign_constraint_effective_radius_m", default_value="12.0"),
        DeclareLaunchArgument("sign_constraint_debug", default_value="true"),
        DeclareLaunchArgument("route_source_mode", default_value="global"),
        DeclareLaunchArgument("fallback_to_simple_forward_route", default_value="false"),
        DeclareLaunchArgument("disable_fallback_driving", default_value="true"),
        DeclareLaunchArgument("disable_fallback_driving_when_mission_missing", default_value="true"),
        DeclareLaunchArgument("global_route_stale_timeout_s", default_value="8.0"),
        DeclareLaunchArgument("mission_goal_near_distance_m", default_value="3.0"),
        DeclareLaunchArgument("route_end_requires_goal_near", default_value="true"),
        DeclareLaunchArgument("replan_when_local_route_exhausted", default_value="true"),
        DeclareLaunchArgument("local_route_short_goal_far_replan_distance_m", default_value="8.0"),
        DeclareLaunchArgument("hold_last_route_s", default_value="2.0"),
        DeclareLaunchArgument("enable_route_events", default_value="true"),
        DeclareLaunchArgument("route_event_horizon_m", default_value="45.0"),
        DeclareLaunchArgument("route_lateral_margin_m", default_value="3.0"),
        DeclareLaunchArgument("vehicle_follow_distance_m", default_value="10.0"),
        DeclareLaunchArgument("vehicle_stop_distance_m", default_value="6.0"),
        DeclareLaunchArgument("pedestrian_stop_distance_m", default_value="8.0"),
        DeclareLaunchArgument("follow_time_gap_s", default_value="1.5"),
        DeclareLaunchArgument("enable_traffic_light_events", default_value="true"),
        DeclareLaunchArgument("traffic_light_horizon_m", default_value="60.0"),
        DeclareLaunchArgument("traffic_light_lateral_margin_m", default_value="4.0"),
        DeclareLaunchArgument("red_detection_horizon_m", default_value="60.0"),
        DeclareLaunchArgument("red_approach_distance_m", default_value="45.0"),
        DeclareLaunchArgument("red_stop_distance_m", default_value="8.0"),
        DeclareLaunchArgument("red_stop_trigger_base_m", default_value="1.5"),
        DeclareLaunchArgument("red_stop_trigger_max_m", default_value="3.0"),
        DeclareLaunchArgument("red_stop_trigger_speed_gain_s", default_value="0.6"),
        DeclareLaunchArgument("red_stop_trigger_speed_buffer_m", default_value="0.8"),
        DeclareLaunchArgument("red_creep_distance_m", default_value="3.0"),
        DeclareLaunchArgument("red_approach_speed_mps", default_value="2.0"),
        DeclareLaunchArgument("red_creep_speed_mps", default_value="0.8"),
        DeclareLaunchArgument("yellow_slow_distance_m", default_value="30.0"),
        DeclareLaunchArgument("yellow_stop_distance_m", default_value="8.0"),
        DeclareLaunchArgument("yellow_slow_speed_mps", default_value="1.5"),
        DeclareLaunchArgument("traffic_light_stop_buffer_m", default_value="1.0"),
        DeclareLaunchArgument("traffic_light_stop_front_bumper_offset_m", default_value="2.0"),
        DeclareLaunchArgument("traffic_light_stop_line_buffer_m", default_value="1.0"),
        DeclareLaunchArgument("traffic_light_stop_distance_tolerance_m", default_value="0.4"),
        DeclareLaunchArgument("traffic_light_stop_debug_enabled", default_value="true"),
        DeclareLaunchArgument("traffic_light_stop_anchor_forward_offset_m", default_value="0.0"),
        DeclareLaunchArgument("traffic_light_post_green_ignore_s", default_value="6.0"),
        DeclareLaunchArgument("traffic_light_passed_ignore_distance_m", default_value="8.0"),
        DeclareLaunchArgument("green_release_distance_m", default_value="8.0"),
        DeclareLaunchArgument("green_ignore_after_pass_m", default_value="6.0"),
        DeclareLaunchArgument("tl_hold_state_memory_s", default_value="2.0"),
        DeclareLaunchArgument("tl_lost_grace_s", default_value="0.75"),
        DeclareLaunchArgument("green_release_grace_s", default_value="1.0"),
        DeclareLaunchArgument("stopped_speed_threshold_mps", default_value="0.25"),
        DeclareLaunchArgument("stopline_reached_distance_m", default_value="1.5"),
        DeclareLaunchArgument("tl_stop_line_buffer_m", default_value="1.0"),
        DeclareLaunchArgument("tl_decel_max_mps2", default_value="1.2"),
        DeclareLaunchArgument("tl_slow_speed_mps", default_value="0.8"),
        DeclareLaunchArgument("tl_min_profile_speed_mps", default_value="0.4"),
        DeclareLaunchArgument("tl_hard_stop_distance_m", default_value="1.2"),
        DeclareLaunchArgument("tl_profile_horizon_m", default_value="45.0"),
        DeclareLaunchArgument("tl_fence_width_m", default_value="8.0"),
        DeclareLaunchArgument("yellow_pass_time_s", default_value="1.0"),
        DeclareLaunchArgument("stopped_vehicle_speed_mps", default_value="0.4"),
        DeclareLaunchArgument("stopped_vehicle_stop_distance_m", default_value="12.0"),
        DeclareLaunchArgument("red_approach_hard_stop_distance_m", default_value="2.0"),
        DeclareLaunchArgument("mission_geojson", default_value=str(default_mission_geojson)),
        DeclareLaunchArgument("target_reached_distance_m", default_value="4.0"),
        DeclareLaunchArgument("mission_position_tolerance_m", default_value="2.0"),
        DeclareLaunchArgument("mission_yaw_tolerance_deg", default_value="20.0"),
        DeclareLaunchArgument("front_bumper_offset_m", default_value="2.0"),
        DeclareLaunchArgument("task_stop_position_tolerance_m", default_value="1.0"),
        DeclareLaunchArgument("task_stop_front_tolerance_m", default_value="1.0"),
        DeclareLaunchArgument("task_stop_yaw_tolerance_deg", default_value="16.0"),
        DeclareLaunchArgument("task_stop_close_enough_distance_m", default_value="0.5"),
        DeclareLaunchArgument("task_stop_close_enough_ignore_yaw", default_value="true"),
        DeclareLaunchArgument("task_stop_close_enough_max_yaw_error_deg", default_value="25.0"),
        DeclareLaunchArgument("task_stop_completion_yaw_tolerance_deg", default_value="14.0"),
        DeclareLaunchArgument("task_stop_completion_position_tolerance_m", default_value="0.75"),
        DeclareLaunchArgument("task_stop_use_side_projection", default_value="false"),
        DeclareLaunchArgument("task_stop_side_projection_lateral_m", default_value="2.2"),
        DeclareLaunchArgument("task_stop_side_projection_forward_m", default_value="0.0"),
        DeclareLaunchArgument("task_stop_side_projection_clamp_to_road", default_value="true"),
        DeclareLaunchArgument("task_stop_raw_override_enabled", default_value="true"),
        DeclareLaunchArgument("task_stop_min_road_edge_clearance_m", default_value="0.4"),
        DeclareLaunchArgument("task_stop_max_side_projection_m", default_value="3.0"),
        DeclareLaunchArgument("task_pull_over_hold_requires_effective_stop", default_value="true"),
        DeclareLaunchArgument("pickup_hold_s", default_value="16.0"),
        DeclareLaunchArgument("dropoff_hold_s", default_value="16.0"),
        DeclareLaunchArgument("loop_mission", default_value="false"),
        DeclareLaunchArgument("mission_publish_rate_hz", default_value="2.0"),
        DeclareLaunchArgument("min_route_points", default_value="8"),
        SetEnvironmentVariable("TEKNOFEST_LOG_SESSION", log_session_id),
        SetEnvironmentVariable("TEKNOFEST_LOG_ROOT", str(log_root)),

        LogInfo(msg="Starting Phase 2 minimal drive stack"),
        LogInfo(msg=f"Phase 2 debug session: {log_root / log_session_id}"),

        # Start Phase 1 nodes (embedded here to ensure proper ordering)
        Node(
            package="autonomous_driving",
            executable="carla_world_manager_node",
            name="carla_world_manager_node",
            output="screen",
            parameters=[{
                "carla_root": ParameterValue(carla_root, value_type=str),
                "host": ParameterValue(host, value_type=str),
                "port": ParameterValue(port, value_type=int),
                "town": ParameterValue(town, value_type=str),
                "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                "timeout": 120.0,
                "status_period_s": 0.1,
                "enable_sync_mode": ParameterValue(enable_sync_mode, value_type=bool),
                "auto_tick_sync_world": ParameterValue(auto_tick_sync_world, value_type=bool),
                "fixed_delta_seconds": ParameterValue(fixed_delta_seconds, value_type=float),
                "tick_rate_hz": ParameterValue(tick_rate_hz, value_type=float),
            }],
        ),

        TimerAction(
            period=6.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="viewport_camera_follow_node",
                    name="viewport_camera_follow_node",
                    output="screen",
                    condition=IfCondition(enable_viewport_camera_follow),
                    parameters=[{
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "carla_host": ParameterValue(viewport_camera_carla_host, value_type=str),
                        "carla_port": ParameterValue(viewport_camera_carla_port, value_type=int),
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                        "camera_view": "chase",
                        "follow_distance_m": ParameterValue(
                            viewport_camera_follow_distance_m,
                            value_type=float,
                        ),
                        "follow_height_m": ParameterValue(
                            viewport_camera_follow_height_m,
                            value_type=float,
                        ),
                        "follow_pitch_deg": ParameterValue(
                            viewport_camera_follow_pitch_deg,
                            value_type=float,
                        ),
                        "follow_update_hz": ParameterValue(
                            viewport_camera_follow_update_hz,
                            value_type=float,
                        ),
                        "enable_demo_weather": ParameterValue(
                            enable_demo_weather,
                            value_type=bool,
                        ),
                        "demo_weather_preset": ParameterValue(
                            demo_weather_preset,
                            value_type=str,
                        ),
                        "enable_ego_lights": ParameterValue(
                            enable_ego_lights,
                            value_type=bool,
                        ),
                        "enable_ego_label": ParameterValue(
                            enable_ego_label,
                            value_type=bool,
                        ),
                        "ego_label_text": ParameterValue(
                            ego_label_text,
                            value_type=str,
                        ),
                        "ego_retry_timeout_s": 60.0,
                    }],
                ),
            ],
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
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
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
                    executable="mission_route_manager_node",
                    name="mission_route_manager",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "mission_geojson": ParameterValue(mission_geojson, value_type=str),
                        "target_reached_distance_m": ParameterValue(target_reached_distance_m, value_type=float),
                        "position_tolerance_m": ParameterValue(mission_position_tolerance_m, value_type=float),
                        "yaw_tolerance_deg": ParameterValue(mission_yaw_tolerance_deg, value_type=float),
                        "front_bumper_offset_m": ParameterValue(front_bumper_offset_m, value_type=float),
                        "task_stop_position_tolerance_m": ParameterValue(task_stop_position_tolerance_m, value_type=float),
                        "task_stop_front_tolerance_m": ParameterValue(task_stop_front_tolerance_m, value_type=float),
                        "task_stop_yaw_tolerance_deg": ParameterValue(task_stop_yaw_tolerance_deg, value_type=float),
                        "task_stop_close_enough_distance_m": ParameterValue(task_stop_close_enough_distance_m, value_type=float),
                        "task_stop_close_enough_ignore_yaw": ParameterValue(task_stop_close_enough_ignore_yaw, value_type=bool),
                        "task_stop_close_enough_max_yaw_error_deg": ParameterValue(task_stop_close_enough_max_yaw_error_deg, value_type=float),
                        "task_stop_completion_yaw_tolerance_deg": ParameterValue(task_stop_completion_yaw_tolerance_deg, value_type=float),
                        "task_stop_completion_position_tolerance_m": ParameterValue(task_stop_completion_position_tolerance_m, value_type=float),
                        "task_stop_use_side_projection": ParameterValue(task_stop_use_side_projection, value_type=bool),
                        "task_stop_side_projection_lateral_m": ParameterValue(task_stop_side_projection_lateral_m, value_type=float),
                        "task_stop_side_projection_forward_m": ParameterValue(task_stop_side_projection_forward_m, value_type=float),
                        "task_stop_side_projection_clamp_to_road": ParameterValue(task_stop_side_projection_clamp_to_road, value_type=bool),
                        "task_stop_raw_override_enabled": ParameterValue(task_stop_raw_override_enabled, value_type=bool),
                        "task_stop_min_road_edge_clearance_m": ParameterValue(task_stop_min_road_edge_clearance_m, value_type=float),
                        "task_stop_max_side_projection_m": ParameterValue(task_stop_max_side_projection_m, value_type=float),
                        "task_pull_over_lateral_offset_m": ParameterValue(task_pull_over_lateral_offset_m, value_type=float),
                        "task_pull_over_hold_requires_effective_stop": ParameterValue(task_pull_over_hold_requires_effective_stop, value_type=bool),
                        "pickup_hold_s": ParameterValue(pickup_hold_s, value_type=float),
                        "dropoff_hold_s": ParameterValue(dropoff_hold_s, value_type=float),
                        "publish_rate_hz": ParameterValue(mission_publish_rate_hz, value_type=float),
                        "loop_mission": ParameterValue(loop_mission, value_type=bool),
                        "competition_mode": True,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=9.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="global_route_planner_node",
                    name="global_route_planner",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                        "global_sampling_resolution_m": ParameterValue(global_sampling_resolution_m, value_type=float),
                        "global_route_publish_hz": ParameterValue(global_route_publish_hz, value_type=float),
                        "replan_distance_threshold_m": ParameterValue(replan_distance_threshold_m, value_type=float),
                        "right_lane_policy_enabled": ParameterValue(right_lane_policy_enabled, value_type=bool),
                        "preferred_lane_side": ParameterValue(preferred_lane_side, value_type=str),
                        "right_lane_projection_max_distance_m": ParameterValue(right_lane_projection_max_distance_m, value_type=float),
                        "right_lane_policy_disable_in_junction": ParameterValue(right_lane_policy_disable_in_junction, value_type=bool),
                        "route_initial_wrong_way_reject_deg": ParameterValue(route_initial_wrong_way_reject_deg, value_type=float),
                        "sign_constraints_enabled": ParameterValue(sign_constraints_enabled, value_type=bool),
                        "sign_plan_geojson": ParameterValue(sign_plan_geojson, value_type=str),
                        "sign_plan_json": ParameterValue(sign_plan_json, value_type=str),
                        "sign_constraint_effective_radius_m": ParameterValue(sign_constraint_effective_radius_m, value_type=float),
                        "sign_constraint_debug": ParameterValue(sign_constraint_debug, value_type=bool),
                        "goal_change_replan": True,
                        "route_timeout_s": 10.0,
                        "min_route_points": ParameterValue(min_route_points, value_type=int),
                    }],
                ),
            ],
        ),

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="route_sampler_node",
                    name="route_sampler",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                        "route_source_mode": ParameterValue(route_source_mode, value_type=str),
                        "fallback_to_simple_forward_route": ParameterValue(fallback_to_simple_forward_route, value_type=bool),
                        "disable_fallback_driving": ParameterValue(disable_fallback_driving, value_type=bool),
                        "disable_fallback_driving_when_mission_missing": ParameterValue(disable_fallback_driving_when_mission_missing, value_type=bool),
                        "global_route_stale_timeout_s": ParameterValue(global_route_stale_timeout_s, value_type=float),
                        "mission_goal_near_distance_m": ParameterValue(mission_goal_near_distance_m, value_type=float),
                        "route_end_requires_goal_near": ParameterValue(route_end_requires_goal_near, value_type=bool),
                        "replan_when_local_route_exhausted": ParameterValue(replan_when_local_route_exhausted, value_type=bool),
                        "local_route_short_goal_far_replan_distance_m": ParameterValue(local_route_short_goal_far_replan_distance_m, value_type=float),
                        "hold_last_route_s": ParameterValue(hold_last_route_s, value_type=float),
                        "local_route_horizon_m": ParameterValue(local_route_horizon_m, value_type=float),
                        "min_route_points": ParameterValue(min_route_points, value_type=int),
                        "rate_hz": 5.0,
                    }],
                    remappings=[
                        ("/adas/planning/route", "/adas/planning/route_base"),
                        ("/adas/planning/route_debug", "/adas/planning/route_base_debug"),
                    ],
                ),
                Node(
                    package="autonomous_driving",
                    executable="slalom_overlay_node",
                    name="slalom_overlay",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "enable_slalom": ParameterValue(enable_slalom, value_type=bool),
                        "slalom_plan_json": ParameterValue(slalom_plan_json, value_type=str),
                        "slalom_start_side": ParameterValue(slalom_start_side, value_type=str),
                        "slalom_clearance_m": ParameterValue(slalom_clearance_m, value_type=float),
                        "slalom_speed_mps": ParameterValue(slalom_speed_mps, value_type=float),
                        "slalom_activation_horizon_m": ParameterValue(
                            slalom_activation_horizon_m,
                            value_type=float,
                        ),
                        "slalom_route_corridor_m": ParameterValue(
                            slalom_route_corridor_m,
                            value_type=float,
                        ),
                    }],
                ),
            ],
        ),

        TimerAction(
            period=10.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="route_event_analyzer_node",
                    name="route_event_analyzer",
                    output="screen",
                    condition=IfCondition(enable_route_events),
                    parameters=[{
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                        "publish_rate_hz": 10.0,
                        "event_horizon_m": ParameterValue(route_event_horizon_m, value_type=float),
                        "route_lateral_margin_m": ParameterValue(route_lateral_margin_m, value_type=float),
                        "vehicle_follow_distance_m": ParameterValue(vehicle_follow_distance_m, value_type=float),
                        "vehicle_stop_distance_m": ParameterValue(vehicle_stop_distance_m, value_type=float),
                        "pedestrian_stop_distance_m": ParameterValue(pedestrian_stop_distance_m, value_type=float),
                        "follow_time_gap_s": ParameterValue(follow_time_gap_s, value_type=float),
                        "stopped_vehicle_speed_mps": ParameterValue(stopped_vehicle_speed_mps, value_type=float),
                        "stopped_vehicle_stop_distance_m": ParameterValue(stopped_vehicle_stop_distance_m, value_type=float),
                        "enable_traffic_light_events": ParameterValue(enable_traffic_light_events, value_type=bool),
                        "traffic_light_horizon_m": ParameterValue(traffic_light_horizon_m, value_type=float),
                        "traffic_light_lateral_margin_m": ParameterValue(traffic_light_lateral_margin_m, value_type=float),
                        "red_detection_horizon_m": ParameterValue(red_detection_horizon_m, value_type=float),
                        "red_approach_distance_m": ParameterValue(red_approach_distance_m, value_type=float),
                        "red_stop_distance_m": ParameterValue(red_stop_distance_m, value_type=float),
                        "red_stop_trigger_base_m": ParameterValue(red_stop_trigger_base_m, value_type=float),
                        "red_stop_trigger_max_m": ParameterValue(red_stop_trigger_max_m, value_type=float),
                        "red_stop_trigger_speed_gain_s": ParameterValue(red_stop_trigger_speed_gain_s, value_type=float),
                        "red_stop_trigger_speed_buffer_m": ParameterValue(red_stop_trigger_speed_buffer_m, value_type=float),
                        "red_creep_distance_m": ParameterValue(red_creep_distance_m, value_type=float),
                        "red_approach_speed_mps": ParameterValue(red_approach_speed_mps, value_type=float),
                        "red_creep_speed_mps": ParameterValue(red_creep_speed_mps, value_type=float),
                        "yellow_slow_distance_m": ParameterValue(yellow_slow_distance_m, value_type=float),
                        "yellow_stop_distance_m": ParameterValue(yellow_stop_distance_m, value_type=float),
                        "yellow_slow_speed_mps": ParameterValue(yellow_slow_speed_mps, value_type=float),
                        "traffic_light_stop_buffer_m": ParameterValue(traffic_light_stop_buffer_m, value_type=float),
                        "traffic_light_stop_front_bumper_offset_m": ParameterValue(traffic_light_stop_front_bumper_offset_m, value_type=float),
                        "traffic_light_stop_line_buffer_m": ParameterValue(traffic_light_stop_line_buffer_m, value_type=float),
                        "traffic_light_stop_distance_tolerance_m": ParameterValue(traffic_light_stop_distance_tolerance_m, value_type=float),
                        "traffic_light_stop_debug_enabled": ParameterValue(traffic_light_stop_debug_enabled, value_type=bool),
                        "traffic_light_stop_anchor_forward_offset_m": ParameterValue(traffic_light_stop_anchor_forward_offset_m, value_type=float),
                        "traffic_light_post_green_ignore_s": ParameterValue(traffic_light_post_green_ignore_s, value_type=float),
                        "traffic_light_passed_ignore_distance_m": ParameterValue(traffic_light_passed_ignore_distance_m, value_type=float),
                        "green_release_distance_m": ParameterValue(green_release_distance_m, value_type=float),
                        "green_ignore_after_pass_m": ParameterValue(green_ignore_after_pass_m, value_type=float),
                        "tl_hold_state_memory_s": ParameterValue(tl_hold_state_memory_s, value_type=float),
                        "tl_lost_grace_s": ParameterValue(tl_lost_grace_s, value_type=float),
                        "green_release_grace_s": ParameterValue(green_release_grace_s, value_type=float),
                        "stopped_speed_threshold_mps": ParameterValue(stopped_speed_threshold_mps, value_type=float),
                        "stopline_reached_distance_m": ParameterValue(stopline_reached_distance_m, value_type=float),
                        "cruise_speed_mps": ParameterValue(cruise_speed_mps, value_type=float),
                        "tl_stop_line_buffer_m": ParameterValue(tl_stop_line_buffer_m, value_type=float),
                        "tl_decel_max_mps2": ParameterValue(tl_decel_max_mps2, value_type=float),
                        "tl_slow_speed_mps": ParameterValue(tl_slow_speed_mps, value_type=float),
                        "tl_min_profile_speed_mps": ParameterValue(tl_min_profile_speed_mps, value_type=float),
                        "tl_hard_stop_distance_m": ParameterValue(tl_hard_stop_distance_m, value_type=float),
                        "tl_profile_horizon_m": ParameterValue(tl_profile_horizon_m, value_type=float),
                        "tl_fence_width_m": ParameterValue(tl_fence_width_m, value_type=float),
                        "yellow_pass_time_s": ParameterValue(yellow_pass_time_s, value_type=float),
                        "stale_route_timeout_s": 1.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=11.0,
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
                        "speed_boost_enabled": ParameterValue(speed_boost_enabled, value_type=bool),
                        "nominal_speed_boost_mps": ParameterValue(nominal_speed_boost_mps, value_type=float),
                        "sharp_turn_yaw_deg": ParameterValue(sharp_turn_yaw_deg, value_type=float),
                        "moderate_turn_yaw_deg": ParameterValue(moderate_turn_yaw_deg, value_type=float),
                        "speed_slew_up_mps_per_s": ParameterValue(speed_slew_up_mps_per_s, value_type=float),
                        "speed_slew_down_mps_per_s": ParameterValue(speed_slew_down_mps_per_s, value_type=float),
                        "dynamic_lookahead_enabled": ParameterValue(dynamic_lookahead_enabled, value_type=bool),
                        "base_lookahead_m": ParameterValue(base_lookahead_m, value_type=float),
                        "lookahead_gain": ParameterValue(lookahead_gain, value_type=float),
                        "lane_assist_only": ParameterValue(lane_assist_only, value_type=bool),
                        "right_lane_lateral_bias_enabled": ParameterValue(right_lane_lateral_bias_enabled, value_type=bool),
                        "right_lane_lateral_bias_m": ParameterValue(right_lane_lateral_bias_m, value_type=float),
                        "right_lane_lateral_bias_min_speed_mps": ParameterValue(right_lane_lateral_bias_min_speed_mps, value_type=float),
                        "right_lane_lateral_bias_disable_in_junction": ParameterValue(right_lane_lateral_bias_disable_in_junction, value_type=bool),
                        "right_lane_lateral_bias_disable_in_turn": ParameterValue(right_lane_lateral_bias_disable_in_turn, value_type=bool),
                        "right_lane_lateral_bias_safety_margin_m": ParameterValue(right_lane_lateral_bias_safety_margin_m, value_type=float),
                        "vehicle_half_width_m": ParameterValue(vehicle_half_width_m, value_type=float),
                        "lane_departure_guard_enabled": ParameterValue(lane_departure_guard_enabled, value_type=bool),
                        "lane_departure_lateral_threshold_m": ParameterValue(lane_departure_lateral_threshold_m, value_type=float),
                        "route_recovery_speed_mps": ParameterValue(route_recovery_speed_mps, value_type=float),
                        "route_conflict_heading_threshold_deg": ParameterValue(route_conflict_heading_threshold_deg, value_type=float),
                        "route_index_hysteresis_enabled": ParameterValue(route_index_hysteresis_enabled, value_type=bool),
                        "max_route_index_jump": ParameterValue(max_route_index_jump, value_type=int),
                        "steering_rate_limit_enabled": ParameterValue(steering_rate_limit_enabled, value_type=bool),
                        "max_steer_delta": ParameterValue(max_steer_delta, value_type=float),
                        "min_nonzero_target_speed_mps": ParameterValue(min_nonzero_target_speed_mps, value_type=float),
                        "task_pull_over_start_distance_m": ParameterValue(task_pull_over_start_distance_m, value_type=float),
                        "task_pull_over_final_distance_m": ParameterValue(task_pull_over_final_distance_m, value_type=float),
                        "task_pull_over_lateral_offset_m": ParameterValue(task_pull_over_lateral_offset_m, value_type=float),
                        "task_stop_reached_distance_m": ParameterValue(task_stop_reached_distance_m, value_type=float),
                        "task_stop_final_phase_latch_enabled": ParameterValue(task_stop_final_phase_latch_enabled, value_type=bool),
                        "task_stop_final_latch_distance_m": ParameterValue(task_stop_final_latch_distance_m, value_type=float),
                        "task_stop_overshoot_guard_distance_m": ParameterValue(task_stop_overshoot_guard_distance_m, value_type=float),
                        "task_stop_overshoot_guard_speed_mps": ParameterValue(task_stop_overshoot_guard_speed_mps, value_type=float),
                        "task_stop_alignment_enabled": ParameterValue(task_stop_alignment_enabled, value_type=bool),
                        "task_stop_alignment_start_distance_m": ParameterValue(task_stop_alignment_start_distance_m, value_type=float),
                        "task_stop_alignment_yaw_tolerance_deg": ParameterValue(task_stop_alignment_yaw_tolerance_deg, value_type=float),
                        "task_stop_alignment_speed_mps": ParameterValue(task_stop_alignment_speed_mps, value_type=float),
                        "task_stop_alignment_target_ahead_m": ParameterValue(task_stop_alignment_target_ahead_m, value_type=float),
                        "task_stop_approach_cruise_speed_mps": ParameterValue(task_stop_approach_cruise_speed_mps, value_type=float),
                        "task_stop_pre_align_speed_mps": ParameterValue(task_stop_pre_align_speed_mps, value_type=float),
                        "task_stop_final_align_speed_mps": ParameterValue(task_stop_final_align_speed_mps, value_type=float),
                        "task_stop_min_creep_speed_mps": ParameterValue(task_stop_min_creep_speed_mps, value_type=float),
                        "task_stop_no_stop_before_final_distance_m": ParameterValue(task_stop_no_stop_before_final_distance_m, value_type=float),
                        "task_stop_phase_hysteresis_m": ParameterValue(task_stop_phase_hysteresis_m, value_type=float),
                        "task_pull_over_approach_speed_mps": ParameterValue(task_pull_over_approach_speed_mps, value_type=float),
                        "task_pull_over_final_speed_mps": ParameterValue(task_pull_over_final_speed_mps, value_type=float),
                        "task_pull_over_crawl_speed_mps": ParameterValue(task_pull_over_crawl_speed_mps, value_type=float),
                        "task_pull_over_keep_bias_until_reached": ParameterValue(task_pull_over_keep_bias_until_reached, value_type=bool),
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
                        "red_approach_hard_stop_distance_m": ParameterValue(
                            red_approach_hard_stop_distance_m,
                            value_type=float,
                        ),
                    }],
                ),
                Node(
                    package="autonomous_driving",
                    executable="carla_control_adapter_node",
                    name="carla_control_adapter_node",
                    output="screen",
                    condition=IfCondition(enable_phase2_drive),
                    parameters=[{
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                    }],
                ),
            ],
        ),
    ])
