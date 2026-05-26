from launch import LaunchDescription
from launch.actions import SetEnvironmentVariable
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        SetEnvironmentVariable("ADAS_HEADLESS", "0"),
        SetEnvironmentVariable("QT_QPA_PLATFORM", "xcb"),
        SetEnvironmentVariable("GDK_BACKEND", "x11"),

        SetEnvironmentVariable("SIGN_CLASSIFIER_ENABLED", "1"),
        SetEnvironmentVariable("SIGN_CLASSIFIER_CONF_THRESHOLD", "0.45"),
        SetEnvironmentVariable(
            "SIGN_CLASSIFIER_MODEL_PATH",
            "autonomous_driving/sign_classifier/outputs_v2/sign_classifier_resnet18_v2_best.pt",
        ),

        SetEnvironmentVariable("RAW_CONF_THRESHOLD", "0.05"),
        SetEnvironmentVariable("TRAFFIC_SIGN_CONF_THRESHOLD", "0.15"),
        SetEnvironmentVariable("TRAFFIC_LIGHT_CONF_THRESHOLD", "0.60"),
        SetEnvironmentVariable("PERSON_CONF_THRESHOLD", "0.40"),
        SetEnvironmentVariable("VEHICLE_CONF_THRESHOLD", "0.40"),

        SetEnvironmentVariable("YOLO_IMGSZ", "960"),
        SetEnvironmentVariable("YOLO_IOU", "0.50"),
        SetEnvironmentVariable("YOLO_MAX_DET", "80"),

        Node(
            package="autonomous_driving",
            executable="teknofest_sign_static_test_node",
            name="teknofest_sign_static_test_node",
            output="screen",
            parameters=[
                {
                    "image_topic": "/adas/camera/front/image_raw",
                    "fps": 5.0,
                    "raw_sign_dir": "autonomous_driving/sign_classifier/dataset_v2/raw",
                }
            ],
        ),

        Node(
            package="autonomous_driving",
            executable="perception_node",
            name="perception_node",
            output="screen",
            parameters=[
                {
                    "image_topic": "/adas/camera/front/image_raw",
                    "detections_topic": "/adas/perception/detections_json",
                    "annotated_topic": "/adas/perception/annotated_image",
                }
            ],
        ),
    ])
