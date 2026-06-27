from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_root = Path(__file__).resolve().parent.parent
    phase2_launch_path = package_root / "launch" / "teknofest_phase2_drive.launch.py"
    default_mission_geojson = package_root / "missions" / "teknofest_town03_competition_v4_tasks_only.geojson"
    default_slalom_plan_json = package_root / "config" / "town03_round3_slalom_plan.json"

    town = LaunchConfiguration("town")
    mission_geojson = LaunchConfiguration("mission_geojson")
    round_name = LaunchConfiguration("round_name")
    enable_phase2_drive = LaunchConfiguration("enable_phase2_drive")
    enable_traffic_light_events = LaunchConfiguration("enable_traffic_light_events")
    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_sensor_tick = LaunchConfiguration("camera_sensor_tick")
    red_stop_trigger_base_m = LaunchConfiguration("red_stop_trigger_base_m")
    red_stop_trigger_max_m = LaunchConfiguration("red_stop_trigger_max_m")
    red_stop_trigger_speed_gain_s = LaunchConfiguration("red_stop_trigger_speed_gain_s")
    red_stop_trigger_speed_buffer_m = LaunchConfiguration("red_stop_trigger_speed_buffer_m")
    red_approach_hard_stop_distance_m = LaunchConfiguration(
        "red_approach_hard_stop_distance_m"
    )
    task_stop_yaw_tolerance_deg = LaunchConfiguration("task_stop_yaw_tolerance_deg")
    task_stop_close_enough_distance_m = LaunchConfiguration("task_stop_close_enough_distance_m")
    task_stop_close_enough_ignore_yaw = LaunchConfiguration("task_stop_close_enough_ignore_yaw")
    task_stop_close_enough_max_yaw_error_deg = LaunchConfiguration(
        "task_stop_close_enough_max_yaw_error_deg"
    )
    task_stop_completion_yaw_tolerance_deg = LaunchConfiguration(
        "task_stop_completion_yaw_tolerance_deg"
    )
    task_stop_completion_position_tolerance_m = LaunchConfiguration(
        "task_stop_completion_position_tolerance_m"
    )
    task_stop_use_side_projection = LaunchConfiguration("task_stop_use_side_projection")
    task_stop_side_projection_lateral_m = LaunchConfiguration(
        "task_stop_side_projection_lateral_m"
    )
    task_stop_side_projection_forward_m = LaunchConfiguration(
        "task_stop_side_projection_forward_m"
    )
    task_stop_side_projection_clamp_to_road = LaunchConfiguration(
        "task_stop_side_projection_clamp_to_road"
    )
    task_stop_raw_override_enabled = LaunchConfiguration("task_stop_raw_override_enabled")
    task_stop_min_road_edge_clearance_m = LaunchConfiguration(
        "task_stop_min_road_edge_clearance_m"
    )
    task_stop_max_side_projection_m = LaunchConfiguration("task_stop_max_side_projection_m")
    task_stop_final_phase_latch_enabled = LaunchConfiguration(
        "task_stop_final_phase_latch_enabled"
    )
    task_stop_final_latch_distance_m = LaunchConfiguration("task_stop_final_latch_distance_m")
    task_stop_overshoot_guard_distance_m = LaunchConfiguration(
        "task_stop_overshoot_guard_distance_m"
    )
    task_stop_overshoot_guard_speed_mps = LaunchConfiguration(
        "task_stop_overshoot_guard_speed_mps"
    )
    task_stop_alignment_enabled = LaunchConfiguration("task_stop_alignment_enabled")
    task_stop_alignment_start_distance_m = LaunchConfiguration(
        "task_stop_alignment_start_distance_m"
    )
    task_stop_alignment_yaw_tolerance_deg = LaunchConfiguration(
        "task_stop_alignment_yaw_tolerance_deg"
    )
    task_stop_alignment_speed_mps = LaunchConfiguration("task_stop_alignment_speed_mps")
    task_stop_alignment_target_ahead_m = LaunchConfiguration(
        "task_stop_alignment_target_ahead_m"
    )
    task_stop_approach_cruise_speed_mps = LaunchConfiguration(
        "task_stop_approach_cruise_speed_mps"
    )
    task_stop_pre_align_speed_mps = LaunchConfiguration("task_stop_pre_align_speed_mps")
    task_stop_final_align_speed_mps = LaunchConfiguration("task_stop_final_align_speed_mps")
    task_stop_min_creep_speed_mps = LaunchConfiguration("task_stop_min_creep_speed_mps")
    task_stop_no_stop_before_final_distance_m = LaunchConfiguration(
        "task_stop_no_stop_before_final_distance_m"
    )
    task_stop_phase_hysteresis_m = LaunchConfiguration("task_stop_phase_hysteresis_m")
    enable_slalom = LaunchConfiguration("enable_slalom")
    slalom_plan_json = LaunchConfiguration("slalom_plan_json")
    slalom_start_side = LaunchConfiguration("slalom_start_side")
    slalom_clearance_m = LaunchConfiguration("slalom_clearance_m")
    slalom_speed_mps = LaunchConfiguration("slalom_speed_mps")
    slalom_activation_horizon_m = LaunchConfiguration("slalom_activation_horizon_m")
    slalom_route_corridor_m = LaunchConfiguration("slalom_route_corridor_m")

    enable_sign_perception = LaunchConfiguration("enable_sign_perception")
    detection_roi_enabled = LaunchConfiguration("detection_roi_enabled")
    detection_roi_x_min_ratio = LaunchConfiguration("detection_roi_x_min_ratio")
    detection_roi_y_min_ratio = LaunchConfiguration("detection_roi_y_min_ratio")
    detection_roi_x_max_ratio = LaunchConfiguration("detection_roi_x_max_ratio")
    detection_roi_y_max_ratio = LaunchConfiguration("detection_roi_y_max_ratio")
    detection_roi_resize_width = LaunchConfiguration("detection_roi_resize_width")
    detection_roi_resize_height = LaunchConfiguration("detection_roi_resize_height")
    detector_conf_threshold = LaunchConfiguration("detector_conf_threshold")
    classifier_conf_threshold = LaunchConfiguration("classifier_conf_threshold")
    min_bbox_width_px = LaunchConfiguration("min_bbox_width_px")
    min_bbox_height_px = LaunchConfiguration("min_bbox_height_px")
    min_bbox_area_px = LaunchConfiguration("min_bbox_area_px")
    min_bbox_area_ratio = LaunchConfiguration("min_bbox_area_ratio")
    max_bbox_area_ratio = LaunchConfiguration("max_bbox_area_ratio")
    min_aspect_ratio = LaunchConfiguration("min_aspect_ratio")
    max_aspect_ratio = LaunchConfiguration("max_aspect_ratio")
    publish_rejected_debug = LaunchConfiguration("publish_rejected_debug")
    draw_rejected_detections = LaunchConfiguration("draw_rejected_detections")
    detector_imgsz = LaunchConfiguration("detector_imgsz")
    process_every_n_frames = LaunchConfiguration("process_every_n_frames")
    debug_publish_rate_hz = LaunchConfiguration("debug_publish_rate_hz")

    return LaunchDescription([
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("mission_geojson", default_value=str(default_mission_geojson)),
        DeclareLaunchArgument("round_name", default_value="round_3"),
        DeclareLaunchArgument("enable_phase2_drive", default_value="true"),
        DeclareLaunchArgument("enable_traffic_light_events", default_value="true"),
        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="360"),
        DeclareLaunchArgument("camera_sensor_tick", default_value="0.1"),
        DeclareLaunchArgument("red_stop_trigger_base_m", default_value="1.5"),
        DeclareLaunchArgument("red_stop_trigger_max_m", default_value="3.0"),
        DeclareLaunchArgument("red_stop_trigger_speed_gain_s", default_value="0.6"),
        DeclareLaunchArgument("red_stop_trigger_speed_buffer_m", default_value="0.8"),
        DeclareLaunchArgument("red_approach_hard_stop_distance_m", default_value="2.0"),
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
        DeclareLaunchArgument("enable_slalom", default_value="false"),
        DeclareLaunchArgument("slalom_plan_json", default_value=str(default_slalom_plan_json)),
        DeclareLaunchArgument("slalom_start_side", default_value="right"),
        DeclareLaunchArgument("slalom_clearance_m", default_value="1.5"),
        DeclareLaunchArgument("slalom_speed_mps", default_value="1.6"),
        DeclareLaunchArgument("slalom_activation_horizon_m", default_value="80.0"),
        DeclareLaunchArgument("slalom_route_corridor_m", default_value="6.0"),

        DeclareLaunchArgument("enable_sign_perception", default_value="true"),
        DeclareLaunchArgument("detection_roi_enabled", default_value="true"),
        DeclareLaunchArgument("detection_roi_x_min_ratio", default_value="0.45"),
        DeclareLaunchArgument("detection_roi_y_min_ratio", default_value="0.10"),
        DeclareLaunchArgument("detection_roi_x_max_ratio", default_value="1.00"),
        DeclareLaunchArgument("detection_roi_y_max_ratio", default_value="0.90"),
        DeclareLaunchArgument("detection_roi_resize_width", default_value="960"),
        DeclareLaunchArgument("detection_roi_resize_height", default_value="540"),
        DeclareLaunchArgument("detector_conf_threshold", default_value="0.15"),
        DeclareLaunchArgument("classifier_conf_threshold", default_value="0.50"),
        DeclareLaunchArgument("min_bbox_width_px", default_value="16"),
        DeclareLaunchArgument("min_bbox_height_px", default_value="16"),
        DeclareLaunchArgument("min_bbox_area_px", default_value="250"),
        DeclareLaunchArgument("min_bbox_area_ratio", default_value="0.001"),
        DeclareLaunchArgument("max_bbox_area_ratio", default_value="0.15"),
        DeclareLaunchArgument("min_aspect_ratio", default_value="0.25"),
        DeclareLaunchArgument("max_aspect_ratio", default_value="4.0"),
        DeclareLaunchArgument("publish_rejected_debug", default_value="true"),
        DeclareLaunchArgument("draw_rejected_detections", default_value="true"),
        DeclareLaunchArgument("detector_imgsz", default_value="512"),
        DeclareLaunchArgument("process_every_n_frames", default_value="4"),
        DeclareLaunchArgument("debug_publish_rate_hz", default_value="3.0"),

        LogInfo(msg="Starting Phase 2 drive stack with isolated traffic sign perception/debug."),

        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(phase2_launch_path)),
            launch_arguments={
                "town": town,
                "mission_geojson": mission_geojson,
                "round_name": round_name,
                "enable_phase2_drive": enable_phase2_drive,
                "enable_traffic_light_events": enable_traffic_light_events,
                "camera_width": camera_width,
                "camera_height": camera_height,
                "camera_sensor_tick": camera_sensor_tick,
                "red_stop_trigger_base_m": red_stop_trigger_base_m,
                "red_stop_trigger_max_m": red_stop_trigger_max_m,
                "red_stop_trigger_speed_gain_s": red_stop_trigger_speed_gain_s,
                "red_stop_trigger_speed_buffer_m": red_stop_trigger_speed_buffer_m,
                "red_approach_hard_stop_distance_m": red_approach_hard_stop_distance_m,
                "task_stop_yaw_tolerance_deg": task_stop_yaw_tolerance_deg,
                "task_stop_close_enough_distance_m": task_stop_close_enough_distance_m,
                "task_stop_close_enough_ignore_yaw": task_stop_close_enough_ignore_yaw,
                "task_stop_close_enough_max_yaw_error_deg": task_stop_close_enough_max_yaw_error_deg,
                "task_stop_completion_yaw_tolerance_deg": task_stop_completion_yaw_tolerance_deg,
                "task_stop_completion_position_tolerance_m": task_stop_completion_position_tolerance_m,
                "task_stop_use_side_projection": task_stop_use_side_projection,
                "task_stop_side_projection_lateral_m": task_stop_side_projection_lateral_m,
                "task_stop_side_projection_forward_m": task_stop_side_projection_forward_m,
                "task_stop_side_projection_clamp_to_road": task_stop_side_projection_clamp_to_road,
                "task_stop_raw_override_enabled": task_stop_raw_override_enabled,
                "task_stop_min_road_edge_clearance_m": task_stop_min_road_edge_clearance_m,
                "task_stop_max_side_projection_m": task_stop_max_side_projection_m,
                "task_stop_final_phase_latch_enabled": task_stop_final_phase_latch_enabled,
                "task_stop_final_latch_distance_m": task_stop_final_latch_distance_m,
                "task_stop_overshoot_guard_distance_m": task_stop_overshoot_guard_distance_m,
                "task_stop_overshoot_guard_speed_mps": task_stop_overshoot_guard_speed_mps,
                "task_stop_alignment_enabled": task_stop_alignment_enabled,
                "task_stop_alignment_start_distance_m": task_stop_alignment_start_distance_m,
                "task_stop_alignment_yaw_tolerance_deg": task_stop_alignment_yaw_tolerance_deg,
                "task_stop_alignment_speed_mps": task_stop_alignment_speed_mps,
                "task_stop_alignment_target_ahead_m": task_stop_alignment_target_ahead_m,
                "task_stop_approach_cruise_speed_mps": task_stop_approach_cruise_speed_mps,
                "task_stop_pre_align_speed_mps": task_stop_pre_align_speed_mps,
                "task_stop_final_align_speed_mps": task_stop_final_align_speed_mps,
                "task_stop_min_creep_speed_mps": task_stop_min_creep_speed_mps,
                "task_stop_no_stop_before_final_distance_m": task_stop_no_stop_before_final_distance_m,
                "task_stop_phase_hysteresis_m": task_stop_phase_hysteresis_m,
                "enable_slalom": enable_slalom,
                "slalom_plan_json": slalom_plan_json,
                "slalom_start_side": slalom_start_side,
                "slalom_clearance_m": slalom_clearance_m,
                "slalom_speed_mps": slalom_speed_mps,
                "slalom_activation_horizon_m": slalom_activation_horizon_m,
                "slalom_route_corridor_m": slalom_route_corridor_m,
            }.items(),
        ),

        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_sign_perception_node",
                    name="traffic_sign_perception_node",
                    output="screen",
                    condition=IfCondition(enable_sign_perception),
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "detections_topic": "/adas/perception/traffic_sign_detections",
                        "viz_topic": "/adas/perception/traffic_sign_viz",
                        "detection_roi_enabled": ParameterValue(detection_roi_enabled, value_type=bool),
                        "detection_roi_x_min_ratio": ParameterValue(detection_roi_x_min_ratio, value_type=float),
                        "detection_roi_y_min_ratio": ParameterValue(detection_roi_y_min_ratio, value_type=float),
                        "detection_roi_x_max_ratio": ParameterValue(detection_roi_x_max_ratio, value_type=float),
                        "detection_roi_y_max_ratio": ParameterValue(detection_roi_y_max_ratio, value_type=float),
                        "detection_roi_resize_width": ParameterValue(detection_roi_resize_width, value_type=int),
                        "detection_roi_resize_height": ParameterValue(detection_roi_resize_height, value_type=int),
                        "detector_conf_threshold": ParameterValue(detector_conf_threshold, value_type=float),
                        "classifier_conf_threshold": ParameterValue(classifier_conf_threshold, value_type=float),
                        "min_bbox_width_px": ParameterValue(min_bbox_width_px, value_type=float),
                        "min_bbox_height_px": ParameterValue(min_bbox_height_px, value_type=float),
                        "min_bbox_area_px": ParameterValue(min_bbox_area_px, value_type=float),
                        "min_bbox_area_ratio": ParameterValue(min_bbox_area_ratio, value_type=float),
                        "max_bbox_area_ratio": ParameterValue(max_bbox_area_ratio, value_type=float),
                        "min_aspect_ratio": ParameterValue(min_aspect_ratio, value_type=float),
                        "max_aspect_ratio": ParameterValue(max_aspect_ratio, value_type=float),
                        "publish_rejected_debug": ParameterValue(publish_rejected_debug, value_type=bool),
                        "draw_rejected_detections": ParameterValue(draw_rejected_detections, value_type=bool),
                        "detector_imgsz": ParameterValue(detector_imgsz, value_type=int),
                        "process_every_n_frames": ParameterValue(process_every_n_frames, value_type=int),
                        "debug_publish_rate_hz": ParameterValue(debug_publish_rate_hz, value_type=float),
                    }],
                ),
            ],
        ),
    ])
