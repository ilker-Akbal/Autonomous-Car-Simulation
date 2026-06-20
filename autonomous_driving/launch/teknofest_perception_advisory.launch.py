from datetime import datetime

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    mission_geojson = LaunchConfiguration("mission_geojson")
    round_name = LaunchConfiguration("round_name")
    ego_role_name = LaunchConfiguration("ego_role_name")
    log_root = LaunchConfiguration("log_root")
    log_session_id = LaunchConfiguration("log_session_id")
    advisory_log_root = LaunchConfiguration("advisory_log_root")
    tl_model_path = LaunchConfiguration("tl_model_path")
    tl_classes_path = LaunchConfiguration("tl_classes_path")
    adas_model_path = LaunchConfiguration("adas_model_path")
    enable_spectator_follow = LaunchConfiguration("enable_spectator_follow")
    camera_view = LaunchConfiguration("camera_view")
    camera_follow_rate_hz = LaunchConfiguration("camera_follow_rate_hz")
    camera_distance_m = LaunchConfiguration("camera_distance_m")
    camera_height_m = LaunchConfiguration("camera_height_m")
    camera_pitch_deg = LaunchConfiguration("camera_pitch_deg")
    draw_mission_points = LaunchConfiguration("draw_mission_points")
    draw_model_detections = LaunchConfiguration("draw_model_detections")
    draw_bbox_only = LaunchConfiguration("draw_bbox_only")
    draw_model_labels = LaunchConfiguration("draw_model_labels")
    draw_model_confidence = LaunchConfiguration("draw_model_confidence")
    draw_detection_text = LaunchConfiguration("draw_detection_text")
    draw_traffic_light_text = LaunchConfiguration("draw_traffic_light_text")
    draw_tl_proxy_detections = LaunchConfiguration("draw_tl_proxy_detections")
    draw_stopline_overlay = LaunchConfiguration("draw_stopline_overlay")
    draw_tl_actor_fallback = LaunchConfiguration("draw_tl_actor_fallback")
    draw_sign_actor_fallback = LaunchConfiguration("draw_sign_actor_fallback")
    draw_carla_actor_debug = LaunchConfiguration("draw_carla_actor_debug")
    log_session_id_value = ParameterValue(log_session_id, value_type=str)
    generated_log_session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("round_name", default_value="round_3"),
        DeclareLaunchArgument("ego_role_name", default_value="ego_vehicle"),
        DeclareLaunchArgument("log_root", default_value="autonomous_driving/outputs/teknofest_sim_logs"),
        DeclareLaunchArgument("advisory_log_root", default_value="autonomous_driving/outputs/perception_advisory_logs"),
        DeclareLaunchArgument("log_session_id", default_value=generated_log_session_id),
        DeclareLaunchArgument(
            "mission_geojson",
            default_value="autonomous_driving/missions/teknofest_town03_competition_v4_tasks_only.geojson",
        ),
        DeclareLaunchArgument(
            "tl_model_path",
            default_value="autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/best.pt",
        ),
        DeclareLaunchArgument(
            "tl_classes_path",
            default_value="autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/classes.json",
        ),
        DeclareLaunchArgument(
            "adas_model_path",
            default_value="/home/ilker/Masaüstü/Autonomous-Car-Simulation/autonomous_driving/outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt",
        ),
        DeclareLaunchArgument("enable_spectator_follow", default_value="true"),
        DeclareLaunchArgument("camera_view", default_value="chase"),
        DeclareLaunchArgument("camera_follow_rate_hz", default_value="90.0"),
        DeclareLaunchArgument("camera_distance_m", default_value="7.0"),
        DeclareLaunchArgument("camera_height_m", default_value="3.2"),
        DeclareLaunchArgument("camera_pitch_deg", default_value="-12.0"),
        DeclareLaunchArgument("draw_mission_points", default_value="true"),
        DeclareLaunchArgument("draw_model_detections", default_value="false"),
        DeclareLaunchArgument("draw_bbox_only", default_value="false"),
        DeclareLaunchArgument("draw_model_labels", default_value="false"),
        DeclareLaunchArgument("draw_model_confidence", default_value="false"),
        DeclareLaunchArgument("draw_detection_text", default_value="false"),
        DeclareLaunchArgument("draw_traffic_light_text", default_value="false"),
        DeclareLaunchArgument("draw_tl_proxy_detections", default_value="false"),
        DeclareLaunchArgument("draw_stopline_overlay", default_value="false"),
        DeclareLaunchArgument("draw_tl_actor_fallback", default_value="false"),
        DeclareLaunchArgument("draw_sign_actor_fallback", default_value="false"),
        DeclareLaunchArgument("draw_carla_actor_debug", default_value="false"),

        LogInfo(msg="Perception advisory visualization started"),
        LogInfo(msg="Actuation nodes are disabled"),
        LogInfo(msg="Perception/logging only"),

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
                        "ego_role_name": ego_role_name,
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
                        "log_session_id": log_session_id_value,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=6.5,
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
                        "ego_role_name": ego_role_name,
                        "mission_topic": "/adas/teknofest/mission",
                        "route_topic": "/adas/planning/route",
                        "sampling_resolution_m": 1.2,
                        "log_root": log_root,
                        "log_session_id": log_session_id_value,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.2,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_detector_node",
                    name="traffic_light_detector_node",
                    output="screen",
                    emulate_tty=True,
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "image_topic": "/adas/camera/front/image_raw",
                        "traffic_light_topic": "/adas/perception/traffic_lights",
                        "traffic_light_model_path": tl_model_path,
                        "traffic_light_classes_path": tl_classes_path,
                        "camera_width": 640,
                        "camera_height": 360,
                        "camera_fov_deg": 72.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.25,
                        "camera_pitch_deg": -1.0,
                        "log_root": log_root,
                        "log_session_id": log_session_id_value,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.23,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_sign_detector_node",
                    name="traffic_sign_detector_node",
                    output="screen",
                    emulate_tty=True,
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "traffic_sign_topic": "/adas/perception/traffic_signs",
                        "model_detections_topic": "/adas/perception/model_detections",
                        "model_path": adas_model_path,
                        "confidence_threshold": 0.35,
                        "image_size": 1024,
                        "process_every_n_frames": 3,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.25,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_stopline_detector_node",
                    name="traffic_light_stopline_detector_node",
                    output="screen",
                    emulate_tty=True,
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
                        "log_session_id": log_session_id_value,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.3,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_light_manager_node",
                    name="traffic_light_manager_node",
                    output="screen",
                    emulate_tty=True,
                    parameters=[{
                        "traffic_light_topic": "/adas/perception/traffic_lights",
                        "route_topic": "/adas/planning/route",
                        "status_topic": "/adas/carla/status",
                        "tl_event_topic": "/adas/planning/tl_event",
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "log_root": log_root,
                        "log_session_id": log_session_id_value,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="perception_advisory_visualizer_node",
                    name="perception_advisory_visualizer_node",
                    output="screen",
                    emulate_tty=True,
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "image_topic_fallback": "/zed/zed_node/left/image_rect_color",
                        "traffic_light_topic": "/adas/perception/traffic_lights",
                        "model_detections_topic": "/adas/perception/model_detections",
                        "tl_event_topic": "/adas/planning/tl_event",
                        "status_topic": "/adas/carla/status",
                        "mission_topic": "/adas/teknofest/mission",
                        "annotated_image_topic": "/adas/perception/annotated_image",
                        "enable_perception_window": True,
                        "perception_window_scale": 0.75,
                        "visualizer_fps": 10.0,
                        "draw_labels": True,
                        "draw_confidence": False,
                        "draw_model_detections": True,
                        "draw_traffic_signs_topic": False,
                        "draw_traffic_lights": False,
                        "publish_annotated_image": True,
                        "draw_tl_proxy_detections": False,
                        "sign_topics": ["/adas/perception/traffic_signs"],
                        "session_id": log_session_id_value,
                        "log_root": advisory_log_root,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.7,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="carla_viewport_overlay_node",
                    name="carla_viewport_overlay_node",
                    output="screen",
                    emulate_tty=True,
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "ego_role_name": ego_role_name,
                        "mission_topic": "/adas/teknofest/mission",
                        "traffic_light_topic": "/adas/perception/traffic_lights",
                        "model_detections_topic": "/adas/perception/model_detections",
                        "depth_topic": "/zed/zed_node/depth/depth_registered",
                        "camera_width": 640,
                        "camera_height": 360,
                        "camera_fov_deg": 72.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.25,
                        "camera_pitch_deg": -1.0,
                        "sign_topics": ["/adas/perception/traffic_signs"],
                        "overlay_rate_hz": 14.0,
                        "draw_lifetime_s": 0.10,
                        "draw_mission_points": draw_mission_points,
                        "draw_model_detections": draw_model_detections,
                        "draw_bbox_only": draw_bbox_only,
                        "draw_model_labels": draw_model_labels,
                        "draw_model_confidence": draw_model_confidence,
                        "draw_detection_text": draw_detection_text,
                        "draw_traffic_light_text": draw_traffic_light_text,
                        "draw_tl_proxy_detections": draw_tl_proxy_detections,
                        "draw_stopline_overlay": draw_stopline_overlay,
                        "draw_tl_actor_fallback": draw_tl_actor_fallback,
                        "draw_sign_actor_fallback": draw_sign_actor_fallback,
                        "draw_carla_actor_debug": draw_carla_actor_debug,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.8,
            actions=[
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "ego_role_name": ego_role_name,
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
                    condition=IfCondition(enable_spectator_follow),
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
