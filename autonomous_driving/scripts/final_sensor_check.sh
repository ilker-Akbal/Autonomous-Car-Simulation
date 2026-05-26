#!/usr/bin/env bash
set -euo pipefail

OUT="outputs/teknofest_sim_logs/final_sensor_check_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p outputs/teknofest_sim_logs

{
  echo "================================================="
  echo "TEKNOFEST FINAL SENSOR CHECK"
  echo "DATE: $(date)"
  echo "PWD: $(pwd)"
  echo "================================================="

  echo
  echo "===== 1) ROS2 NODE LIST ====="
  ros2 node list || true

  echo
  echo "===== 2) SENSOR / PERCEPTION TOPIC LIST ====="
  ros2 topic list -t | grep -Ei "image|camera|depth|zed|lidar|point|cloud|scan|imu|gps|gnss|camera_info|annotated|perception|lane|decision|route_agent" || true

  echo
  echo "===== 3) PERCEPTION NODE INFO ====="
  ros2 node info /perception_node || true

  echo
  echo "===== 4) DECISION NODE INFO ====="
  ros2 node info /decision_node || true

  echo
  echo "===== 5) ROUTE AGENT NODE INFO ====="
  ros2 node info /teknofest_route_agent_node || true

  echo
  echo "===== 6) IMPORTANT TOPIC INFO ====="

  for T in \
    "/adas/camera/front/image_raw" \
    "/adas/camera/front/camera_info" \
    "/adas/perception/annotated_image" \
    "/adas/perception/detections_json" \
    "/adas/lidar/points" \
    "/zed/zed_node/depth/depth_registered" \
    "/zed/zed_node/left/image_rect_color" \
    "/zed/zed_node/right/image_rect_color" \
    "/zed/zed_node/point_cloud/cloud_registered" \
    "/adas/localization/gnss" \
    "/adas/localization/imu" \
    "/adas/lane/assist" \
    "/adas/decision" \
    "/adas/teknofest/route_agent_debug"
  do
    echo
    echo "---- $T ----"
    if ros2 topic list | grep -qx "$T"; then
      ros2 topic info -v "$T" || true
    else
      echo "[MISSING] $T"
    fi
  done

  echo
  echo "===== 7) FREQUENCY CHECK ====="

  hz_check () {
    local TOPIC="$1"
    echo
    echo "---- HZ: $TOPIC ----"
    if ros2 topic list | grep -qx "$TOPIC"; then
      timeout 8 ros2 topic hz "$TOPIC" || true
    else
      echo "[SKIP] Topic yok: $TOPIC"
    fi
  }

  hz_check "/adas/camera/front/image_raw"
  hz_check "/adas/perception/annotated_image"
  hz_check "/zed/zed_node/left/image_rect_color"
  hz_check "/zed/zed_node/right/image_rect_color"
  hz_check "/zed/zed_node/depth/depth_registered"
  hz_check "/zed/zed_node/point_cloud/cloud_registered"
  hz_check "/adas/lidar/points"
  hz_check "/adas/lane/assist"

  echo
  echo "===== 8) DETECTIONS JSON SAMPLE ====="

  timeout 10 python3 - << 'PY' || true
import json
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Once(Node):
    def __init__(self):
        super().__init__("final_check_detections_once")
        self.done = False
        self.sub = self.create_subscription(
            String,
            "/adas/perception/detections_json",
            self.cb,
            10,
        )

    def cb(self, msg):
        if self.done:
            return
        self.done = True

        try:
            data = json.loads(msg.data)
        except Exception as e:
            print("detections_json parse error:", e)
            rclpy.shutdown()
            return

        print("image:", data.get("image_width"), "x", data.get("image_height"))
        print("fusion_enabled:", data.get("fusion_enabled"))
        print("fusion_sources:", data.get("fusion_sources"))
        print("front_lidar_obstacle_m:", data.get("front_lidar_obstacle_m"))

        dets = data.get("detections", [])
        print("detections_count:", len(dets))

        for d in dets[:5]:
            print(json.dumps(d, ensure_ascii=False, indent=2))

        rclpy.shutdown()

rclpy.init()
node = Once()
try:
    rclpy.spin(node)
finally:
    if rclpy.ok():
        rclpy.shutdown()
PY

  echo
  echo "===== 9) LIDAR SAMPLE ====="
  timeout 5 ros2 topic echo /adas/lidar/points --once | head -80 || true

  echo
  echo "===== 10) DEPTH SAMPLE ====="
  timeout 5 ros2 topic echo /zed/zed_node/depth/depth_registered --once | head -60 || true

  echo
  echo "===== 11) LANE ASSIST SAMPLE ====="
  timeout 5 ros2 topic echo /adas/lane/assist --once || true

  echo
  echo "===== 12) ROUTE AGENT DEBUG SAMPLE ====="
  timeout 5 ros2 topic echo /adas/teknofest/route_agent_debug --once || true

  echo
  echo "===== 13) DECISION SAMPLE ====="
  timeout 5 ros2 topic echo /adas/decision --once || true

  echo
  echo "===== 14) SUMMARY GUIDE ====="
  echo "Beklenen:"
  echo "- perception_node subscribers: RGB image + /adas/lidar/points + /zed depth"
  echo "- detections_json: fusion_enabled=True"
  echo "- Nesne varsa detection içinde distance_m / distance_est / distance_source"
  echo "- /adas/perception/annotated_image publisher count 1"
  echo "- rqt_image_view açılırsa annotated image subscription count 1 olur"

} | tee "$OUT"

echo
echo "Final kontrol dosyası:"
echo "$OUT"
