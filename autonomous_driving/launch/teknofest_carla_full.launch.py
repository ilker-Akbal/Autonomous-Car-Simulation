from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, SetEnvironmentVariable, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    model_path = LaunchConfiguration("model_path")
    tl_model_path = LaunchConfiguration("tl_model_path")
    sign_model_path = LaunchConfiguration("sign_model_path")
    mission_geojson = LaunchConfiguration("mission_geojson")
    log_dir = LaunchConfiguration("log_dir")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument(
            "model_path",
            default_value="autonomous_driving/outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt",
        ),
        DeclareLaunchArgument(
            "tl_model_path",
            default_value="autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/best.pt",
        ),
        DeclareLaunchArgument(
            "sign_model_path",
            default_value="autonomous_driving/sign_classifier/outputs_v2/sign_classifier_resnet18_v2_best.pt",
        ),
        DeclareLaunchArgument(
            "mission_geojson",
            default_value="autonomous_driving/missions/teknofest_town03_competition_v4_tasks_only.geojson",
        ),
        DeclareLaunchArgument("log_dir", default_value="outputs/teknofest_sim_logs"),

        SetEnvironmentVariable("ADAS_HEADLESS", "1"),
        SetEnvironmentVariable("SHOW_DEBUG", "0"),
        SetEnvironmentVariable("MODEL_PATH", model_path),
        SetEnvironmentVariable("TRAFFIC_LIGHT_STATE_MODEL_PATH", tl_model_path),
        SetEnvironmentVariable("TRAFFIC_LIGHT_STATE_CLASSIFIER_ENABLED", "1"),
        SetEnvironmentVariable("TRAFFIC_LIGHT_STATE_USE_HSV_FALLBACK", "1"),
        SetEnvironmentVariable("SIGN_CLASSIFIER_ENABLED", "1"),
        SetEnvironmentVariable("SIGN_CLASSIFIER_MODEL_PATH", sign_model_path),

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
            period=8.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="perception_node",
                    name="perception_node",
                    output="screen",
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "detections_topic": "/adas/perception/detections_json",
                        "decision_events_topic": "/adas/perception/decision_events_json",
                        "route_constraints_topic": "/adas/perception/route_constraints_json",
                        "annotated_topic": "/adas/perception/annotated_image",
                        "model_path": model_path,
                        "traffic_light_state_model_path": tl_model_path,
                        "traffic_light_state_classifier_enabled": True,
                        "traffic_light_state_use_hsv_fallback": True,
                        "traffic_light_state_device": "cuda",
                        "sign_classifier_enabled": True,
                        "sign_classifier_model_path": sign_model_path,
                        "show_debug": False,
                        "imgsz": 960,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="phase1_route_node",
                    name="phase1_route_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "mission_geojson": mission_geojson,
                        "competition_mode": False,
                        "route_topic": "/adas/phase1/route",
                        "lookahead_distance_m": 8.0,
                        "prefer_right_lane": True,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=9.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="phase1_lane_follower_node",
                    name="phase1_lane_follower_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "route_topic": "/adas/phase1/route",
                        "lane_command_topic": "/adas/phase1/lane_command",
                    }],
                ),
            ],
        ),

        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="phase1_behavior_node",
                    name="phase1_behavior_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "route_topic": "/adas/phase1/route",
                        "lane_command_topic": "/adas/phase1/lane_command",
                        "perception_events_topic": "/adas/perception/decision_events_json",
                        "behavior_topic": "/adas/phase1/behavior",
                        "base_speed_mps": 6.0,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=11.0,
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
                        "behavior_topic": "/adas/phase1/behavior",
                        "debug_topic": "/adas/carla/control_debug",
                    }],
                ),
            ],
        ),

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
    ])
