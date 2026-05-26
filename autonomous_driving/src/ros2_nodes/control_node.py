import json
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")

        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("cmd_topic", "/cmd_vel")

        self.declare_parameter("default_stop_speed", 0.0)
        self.declare_parameter("default_slow_speed", 0.8)
        self.declare_parameter("default_go_speed", 1.5)
        self.declare_parameter("max_speed", 3.0)

        self.decision_topic = self.get_parameter("decision_topic").value
        self.cmd_topic = self.get_parameter("cmd_topic").value

        self.default_stop_speed = float(self.get_parameter("default_stop_speed").value)
        self.default_slow_speed = float(self.get_parameter("default_slow_speed").value)
        self.default_go_speed = float(self.get_parameter("default_go_speed").value)
        self.max_speed = float(self.get_parameter("max_speed").value)

        self.publisher = self.create_publisher(Twist, self.cmd_topic, 10)

        self.subscription = self.create_subscription(
            String,
            self.decision_topic,
            self.decision_callback,
            10,
        )

        self.get_logger().info(f"control_node başladı: {self.decision_topic} -> {self.cmd_topic}")

    def clamp_speed(self, speed):
        return max(0.0, min(float(speed), self.max_speed))

    def decision_callback(self, msg: String) -> None:
        cmd = Twist()

        try:
            data = json.loads(msg.data)
            decision = str(data.get("decision", "STOP")).upper()
            risk = data.get("risk", "UNKNOWN")
            distance_est = data.get("distance_est", None)
            target_speed = data.get("target_speed", None)
            reason = data.get("reason", "unknown")
        except Exception:
            decision = msg.data.strip().upper()
            risk = "UNKNOWN"
            distance_est = None
            target_speed = None
            reason = "raw_text_decision"

        if target_speed is not None:
            linear_speed = self.clamp_speed(target_speed)
        else:
            if decision == "STOP":
                linear_speed = self.default_stop_speed
            elif decision == "SLOW":
                linear_speed = self.default_slow_speed
            elif decision == "GO":
                linear_speed = self.default_go_speed
            elif decision in ["LEFT", "RIGHT"]:
                linear_speed = self.default_slow_speed
            else:
                linear_speed = self.default_stop_speed

        angular_speed = 0.0

        if decision == "STOP":
            linear_speed = 0.0
            angular_speed = 0.0
        elif decision == "LEFT":
            angular_speed = 0.5
        elif decision == "RIGHT":
            angular_speed = -0.5

        cmd.linear.x = linear_speed
        cmd.angular.z = angular_speed

        self.publisher.publish(cmd)

        self.get_logger().info(
            f"[CONTROL] decision={decision} risk={risk} target_speed={target_speed} "
            f"distance_est={distance_est} reason={reason} "
            f"linear={cmd.linear.x} angular={cmd.angular.z}",
            throttle_duration_sec=0.5,
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControlNode()
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