import json
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _clamp(v, a, b):
    return max(a, min(b, v))


class ControlNode(Node):
    def __init__(self):
        super().__init__("control_node")

        self.declare_parameter("kp", 0.35)
        self.declare_parameter("ki", 0.02)
        self.declare_parameter("brake_kp", 0.45)
        self.declare_parameter("max_throttle", 0.45)
        self.declare_parameter("max_brake", 0.75)
        self.declare_parameter("steer_rate_limit", 0.08)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("rate_hz", 20.0)

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.brake_kp = float(self.get_parameter("brake_kp").value)
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)
        self.steer_rate_limit = float(self.get_parameter("steer_rate_limit").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.last_plan = None
        self.last_plan_time = 0.0
        self.last_status = None
        self.last_status_time = 0.0

        self.integral = 0.0
        self.last_steer = 0.0

        self.create_subscription(String, "/adas/planning/lane_plan", self._plan_cb, 10)
        self.create_subscription(String, "/adas/carla/status", self._status_cb, 10)

        self.cmd_pub = self.create_publisher(String, "/adas/control/vehicle_command", 10)
        self.debug_pub = self.create_publisher(String, "/adas/control/debug", 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._run)

    def _plan_cb(self, msg: String):
        try:
            self.last_plan = json.loads(msg.data)
            self.last_plan_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse lane_plan JSON")

    def _status_cb(self, msg: String):
        try:
            self.last_status = json.loads(msg.data)
            self.last_status_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse carla status JSON")

    def _get_current_speed(self):
        if not self.last_status:
            return 0.0
        s = self.last_status.get("speed_mps")
        if s is not None:
            return float(s)
        s = self.last_status.get("speed")
        if s is not None:
            return float(s)
        s = self.last_status.get("speed_kmh")
        if s is not None:
            return float(s) / 3.6
        return 0.0

    def _run(self):
        now = time.time()
        plan_ok = (self.last_plan is not None) and (now - self.last_plan_time) < 1.0
        status_ok = (self.last_status is not None) and (now - self.last_status_time) < 1.0

        if not plan_ok or not status_ok:
            # publish emergency brake
            cmd = {"stamp": now, "source": "phase2_control", "target_speed_mps": 0.0, "current_speed_mps": self._get_current_speed(), "throttle": 0.0, "brake": 1.0, "steer": 0.0, "reverse": False, "hand_brake": False}
            m = String(); m.data = json.dumps(cmd); self.cmd_pub.publish(m)
            dbg = String(); dbg.data = json.dumps({"status_ok": status_ok, "plan_ok": plan_ok}); self.debug_pub.publish(dbg)
            return

        target_speed = float(self.last_plan.get("target_speed_mps", 0.0))
        target_steer = float(self.last_plan.get("steer", 0.0))

        current_speed = self._get_current_speed()

        error = target_speed - current_speed
        self.integral += error * (1.0 / max(1.0, self.rate_hz))

        throttle = _clamp(self.kp * error + self.ki * self.integral, 0.0, self.max_throttle)
        brake = 0.0
        if error < 0:
            brake = _clamp(-self.brake_kp * error, 0.0, self.max_brake)

        # steering rate limit
        max_delta = self.steer_rate_limit
        steer = _clamp(target_steer, -1.0, 1.0)
        delta = steer - self.last_steer
        if abs(delta) > max_delta:
            steer = self.last_steer + (max_delta if delta > 0 else -max_delta)
        self.last_steer = steer

        cmd = {
            "stamp": now,
            "source": "phase2_control",
            "target_speed_mps": target_speed,
            "current_speed_mps": current_speed,
            "throttle": round(float(throttle), 3),
            "brake": round(float(brake), 3),
            "steer": round(float(steer), 3),
            "reverse": False,
            "hand_brake": False,
        }

        m = String(); m.data = json.dumps(cmd); self.cmd_pub.publish(m)
        dbg = String(); dbg.data = json.dumps({"error": error, "throttle": throttle, "brake": brake, "steer": steer}); self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = ControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
