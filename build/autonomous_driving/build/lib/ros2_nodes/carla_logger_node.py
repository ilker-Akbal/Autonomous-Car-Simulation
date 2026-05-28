import csv
import json
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CarlaLoggerNode(Node):
    def __init__(self):
        super().__init__("carla_logger_node")

        self.declare_parameter(
            "log_dir",
            "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/logs/carla",
        )

        self.log_dir = self.get_parameter("log_dir").value
        os.makedirs(self.log_dir, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_dir = os.path.join(self.log_dir, f"carla_run_{stamp}")
        os.makedirs(self.run_dir, exist_ok=True)

        self.decision_path = os.path.join(self.run_dir, "decisions.jsonl")
        self.control_path = os.path.join(self.run_dir, "control.jsonl")
        self.status_path = os.path.join(self.run_dir, "status.jsonl")
        self.collision_path = os.path.join(self.run_dir, "collisions.jsonl")

        self.create_subscription(String, "/adas/decision", self.decision_cb, 10)
        self.create_subscription(String, "/adas/carla/control_debug", self.control_cb, 10)
        self.create_subscription(String, "/adas/carla/status", self.status_cb, 10)
        self.create_subscription(String, "/adas/events/collision", self.collision_cb, 10)

        self.get_logger().info(f"CARLA logger hazır: {self.run_dir}")

    def append_jsonl(self, path, raw_data):
        try:
            data = json.loads(raw_data)
        except Exception:
            data = {"raw": raw_data}

        data["_logger_stamp"] = time.time()

        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def decision_cb(self, msg):
        self.append_jsonl(self.decision_path, msg.data)

    def control_cb(self, msg):
        self.append_jsonl(self.control_path, msg.data)

    def status_cb(self, msg):
        self.append_jsonl(self.status_path, msg.data)

    def collision_cb(self, msg):
        self.append_jsonl(self.collision_path, msg.data)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaLoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()