from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    package_root = Path(__file__).resolve().parent.parent
    default_detector_model_path = package_root / "models" / "traffic_sign_detector" / "adas5_sign_yolo_best.pt"
    default_classifier_model_path = (
        package_root / "models" / "traffic_sign_classifier" / "sign_classifier_resnet18_v2_best.pt"
    )

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

    camera_width = LaunchConfiguration("camera_width")
    camera_height = LaunchConfiguration("camera_height")
    camera_sensor_tick = LaunchConfiguration("camera_sensor_tick")
    enable_sync_mode = LaunchConfiguration("enable_sync_mode")
    auto_tick_sync_world = LaunchConfiguration("auto_tick_sync_world")
    fixed_delta_seconds = LaunchConfiguration("fixed_delta_seconds")
    tick_rate_hz = LaunchConfiguration("tick_rate_hz")

    traffic_sign_detector_model_path = LaunchConfiguration("traffic_sign_detector_model_path")
    traffic_sign_classifier_model_path = LaunchConfiguration("traffic_sign_classifier_model_path")
    detector_conf_threshold = LaunchConfiguration("detector_conf_threshold")
    classifier_conf_threshold = LaunchConfiguration("classifier_conf_threshold")
    traffic_sign_iou_threshold = LaunchConfiguration("traffic_sign_iou_threshold")
    traffic_sign_inference_period_s = LaunchConfiguration("traffic_sign_inference_period_s")
    traffic_sign_publish_viz = LaunchConfiguration("traffic_sign_publish_viz")
    publish_rejected_debug = LaunchConfiguration("publish_rejected_debug")
    draw_rejected_detections = LaunchConfiguration("draw_rejected_detections")
    input_min_width_warn = LaunchConfiguration("input_min_width_warn")
    input_min_height_warn = LaunchConfiguration("input_min_height_warn")
    min_bbox_width_px = LaunchConfiguration("min_bbox_width_px")
    min_bbox_height_px = LaunchConfiguration("min_bbox_height_px")
    min_bbox_area_px = LaunchConfiguration("min_bbox_area_px")
    min_bbox_area_ratio = LaunchConfiguration("min_bbox_area_ratio")
    max_bbox_area_ratio = LaunchConfiguration("max_bbox_area_ratio")
    min_aspect_ratio = LaunchConfiguration("min_aspect_ratio")
    max_aspect_ratio = LaunchConfiguration("max_aspect_ratio")
    detection_roi_x_min_ratio = LaunchConfiguration("detection_roi_x_min_ratio")
    detection_roi_y_min_ratio = LaunchConfiguration("detection_roi_y_min_ratio")
    detection_roi_x_max_ratio = LaunchConfiguration("detection_roi_x_max_ratio")
    detection_roi_y_max_ratio = LaunchConfiguration("detection_roi_y_max_ratio")
    detection_roi_resize_width = LaunchConfiguration("detection_roi_resize_width")
    detection_roi_resize_height = LaunchConfiguration("detection_roi_resize_height")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("ego_role_name", default_value="ego_vehicle"),

        DeclareLaunchArgument("camera_width", default_value="640"),
        DeclareLaunchArgument("camera_height", default_value="360"),
        DeclareLaunchArgument("camera_sensor_tick", default_value="0.1"),
        DeclareLaunchArgument("enable_sync_mode", default_value="true"),
        DeclareLaunchArgument("auto_tick_sync_world", default_value="true"),
        DeclareLaunchArgument("fixed_delta_seconds", default_value="0.05"),
        DeclareLaunchArgument("tick_rate_hz", default_value="20.0"),

        DeclareLaunchArgument("traffic_sign_detector_model_path", default_value=str(default_detector_model_path)),
        DeclareLaunchArgument("traffic_sign_classifier_model_path", default_value=str(default_classifier_model_path)),
        DeclareLaunchArgument("detector_conf_threshold", default_value="0.08"),
        DeclareLaunchArgument("classifier_conf_threshold", default_value="0.50"),
        DeclareLaunchArgument("traffic_sign_iou_threshold", default_value="0.45"),
        DeclareLaunchArgument("traffic_sign_inference_period_s", default_value="0.15"),
        DeclareLaunchArgument("traffic_sign_publish_viz", default_value="true"),
        DeclareLaunchArgument("publish_rejected_debug", default_value="true"),
        DeclareLaunchArgument("draw_rejected_detections", default_value="true"),
        DeclareLaunchArgument("input_min_width_warn", default_value="640"),
        DeclareLaunchArgument("input_min_height_warn", default_value="360"),
        DeclareLaunchArgument("min_bbox_width_px", default_value="8"),
        DeclareLaunchArgument("min_bbox_height_px", default_value="8"),
        DeclareLaunchArgument("min_bbox_area_px", default_value="80"),
        DeclareLaunchArgument("min_bbox_area_ratio", default_value="0.00025"),
        DeclareLaunchArgument("max_bbox_area_ratio", default_value="0.15"),
        DeclareLaunchArgument("min_aspect_ratio", default_value="0.25"),
        DeclareLaunchArgument("max_aspect_ratio", default_value="4.0"),
        DeclareLaunchArgument("detection_roi_x_min_ratio", default_value="0.45"),
        DeclareLaunchArgument("detection_roi_y_min_ratio", default_value="0.10"),
        DeclareLaunchArgument("detection_roi_x_max_ratio", default_value="1.00"),
        DeclareLaunchArgument("detection_roi_y_max_ratio", default_value="0.90"),
        DeclareLaunchArgument("detection_roi_resize_width", default_value="960"),
        DeclareLaunchArgument("detection_roi_resize_height", default_value="540"),

        DeclareLaunchArgument("enable_camera_follow", default_value="true"),
        DeclareLaunchArgument("camera_view", default_value="chase"),
        DeclareLaunchArgument("camera_follow_rate_hz", default_value="60.0"),
        DeclareLaunchArgument("camera_distance_m", default_value="7.0"),
        DeclareLaunchArgument("camera_height_m", default_value="3.2"),
        DeclareLaunchArgument("camera_pitch_deg", default_value="-12.0"),

        LogInfo(msg="Traffic sign perception debug stack starting."),

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
                "timeout": 120.0,
                "ego_role_name": ParameterValue(ego_role_name, value_type=str),
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
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
                        "timeout": 120.0,
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                        "camera_width": ParameterValue(camera_width, value_type=int),
                        "camera_height": ParameterValue(camera_height, value_type=int),
                        "camera_fov": 72.0,
                        "camera_x": 1.6,
                        "camera_y": 0.0,
                        "camera_z": 2.25,
                        "camera_pitch": -1.0,
                        "camera_sensor_tick": ParameterValue(camera_sensor_tick, value_type=float),
                        "zed_enabled": False,
                        "depth_enabled": False,
                        "lidar_enabled": False,
                        "zed_point_cloud_enabled": False,
                        "front_rgb_separate_enabled": True,
                        "front_rgb_from_zed_left": False,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="traffic_sign_perception_node",
                    name="traffic_sign_perception_node",
                    output="screen",
                    parameters=[{
                        "detector_model_path": ParameterValue(traffic_sign_detector_model_path, value_type=str),
                        "classifier_model_path": ParameterValue(traffic_sign_classifier_model_path, value_type=str),
                        "detector_conf_threshold": ParameterValue(detector_conf_threshold, value_type=float),
                        "classifier_conf_threshold": ParameterValue(classifier_conf_threshold, value_type=float),
                        "iou_threshold": ParameterValue(traffic_sign_iou_threshold, value_type=float),
                        "inference_period_s": ParameterValue(traffic_sign_inference_period_s, value_type=float),
                        "publish_viz": ParameterValue(traffic_sign_publish_viz, value_type=bool),
                        "publish_rejected_debug": ParameterValue(publish_rejected_debug, value_type=bool),
                        "draw_rejected_detections": ParameterValue(draw_rejected_detections, value_type=bool),
                        "input_min_width_warn": ParameterValue(input_min_width_warn, value_type=int),
                        "input_min_height_warn": ParameterValue(input_min_height_warn, value_type=int),
                        "min_bbox_width_px": ParameterValue(min_bbox_width_px, value_type=float),
                        "min_bbox_height_px": ParameterValue(min_bbox_height_px, value_type=float),
                        "min_bbox_area_px": ParameterValue(min_bbox_area_px, value_type=float),
                        "min_bbox_area_ratio": ParameterValue(min_bbox_area_ratio, value_type=float),
                        "max_bbox_area_ratio": ParameterValue(max_bbox_area_ratio, value_type=float),
                        "min_aspect_ratio": ParameterValue(min_aspect_ratio, value_type=float),
                        "max_aspect_ratio": ParameterValue(max_aspect_ratio, value_type=float),
                        "detection_roi_x_min_ratio": ParameterValue(detection_roi_x_min_ratio, value_type=float),
                        "detection_roi_y_min_ratio": ParameterValue(detection_roi_y_min_ratio, value_type=float),
                        "detection_roi_x_max_ratio": ParameterValue(detection_roi_x_max_ratio, value_type=float),
                        "detection_roi_y_max_ratio": ParameterValue(detection_roi_y_max_ratio, value_type=float),
                        "detection_roi_resize_width": ParameterValue(detection_roi_resize_width, value_type=int),
                        "detection_roi_resize_height": ParameterValue(detection_roi_resize_height, value_type=int),
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
                        "carla_root": ParameterValue(carla_root, value_type=str),
                        "host": ParameterValue(host, value_type=str),
                        "port": ParameterValue(port, value_type=int),
                        "ego_role_name": ParameterValue(ego_role_name, value_type=str),
                        "camera_view": ParameterValue(camera_view, value_type=str),
                        "camera_follow_rate_hz": ParameterValue(camera_follow_rate_hz, value_type=float),
                        "camera_distance_m": ParameterValue(camera_distance_m, value_type=float),
                        "camera_height_m": ParameterValue(camera_height_m, value_type=float),
                        "camera_pitch_deg": ParameterValue(camera_pitch_deg, value_type=float),
                    }],
                ),
            ],
        ),
    ])
