from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    carla_root = LaunchConfiguration("carla_root")
    host = LaunchConfiguration("host")
    port = LaunchConfiguration("port")
    town = LaunchConfiguration("town")
    mission_geojson = LaunchConfiguration("mission_geojson")
    round_name = LaunchConfiguration("round_name")

    return LaunchDescription([
        DeclareLaunchArgument("carla_root", default_value="/home/ilker/simulators/CARLA_0.9.15"),
        DeclareLaunchArgument("host", default_value="127.0.0.1"),
        DeclareLaunchArgument("port", default_value="2000"),
        DeclareLaunchArgument("town", default_value="Town03"),
        DeclareLaunchArgument("round_name", default_value="round_3"),
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
                        "point_pass_tolerance_m": 2.5,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="clean_lane_vision_node",
                    name="clean_lane_vision_node",
                    output="screen",
                    parameters=[{
                        "image_topic": "/adas/camera/front/image_raw",
                        "vision_topic": "/adas/phase1/lane_vision_json",
                        "debug_image_topic": "/adas/phase1/lane_vision_debug_image",
                    }],
                ),
            ],
        ),

        TimerAction(
            period=7.5,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="clean_phase1_driver_node",
                    name="clean_phase1_driver_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "command_topic": "/adas/phase1/command",
                        "mission_topic": "/adas/teknofest/mission",
                        "sign_facts_topic": "/adas/phase1/sign_facts_json",
                        "lane_vision_topic": "/adas/phase1/lane_vision_json",
                        "lookahead_m": 8.0,
                        "cruise_speed_mps": 5.8,
                        "turn_speed_mps": 3.2,
                    }],
                ),
            ],
        ),

        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package="autonomous_driving",
                    executable="clean_carla_control_node",
                    name="clean_carla_control_node",
                    output="screen",
                    parameters=[{
                        "carla_root": carla_root,
                        "host": host,
                        "port": port,
                        "timeout": 120.0,
                        "ego_role_name": "ego_vehicle",
                        "command_topic": "/adas/phase1/command",
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
