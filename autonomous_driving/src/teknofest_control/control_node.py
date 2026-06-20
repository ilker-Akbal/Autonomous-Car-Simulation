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
        self.declare_parameter("throttle_floor_when_moving", 0.12)
        self.declare_parameter("uphill_speed_error_boost", 0.10)
        self.declare_parameter("min_speed_for_throttle_floor_mps", 0.5)
        self.declare_parameter("throttle_slew_limit", 0.04)
        self.declare_parameter("integral_limit", 3.0)
        self.declare_parameter("command_timeout_s", 0.5)
        self.declare_parameter("rate_hz", 20.0)

        self.kp = float(self.get_parameter("kp").value)
        self.ki = float(self.get_parameter("ki").value)
        self.brake_kp = float(self.get_parameter("brake_kp").value)
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)
        self.steer_rate_limit = float(self.get_parameter("steer_rate_limit").value)
        self.throttle_floor_when_moving = float(self.get_parameter("throttle_floor_when_moving").value)
        self.uphill_speed_error_boost = float(self.get_parameter("uphill_speed_error_boost").value)
        self.min_speed_for_throttle_floor_mps = float(self.get_parameter("min_speed_for_throttle_floor_mps").value)
        self.throttle_slew_limit = float(self.get_parameter("throttle_slew_limit").value)
        self.integral_limit = float(self.get_parameter("integral_limit").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.last_plan = None
        self.last_plan_time = 0.0
        self.last_status = None
        self.last_status_time = 0.0

        self.integral = 0.0
        self.last_throttle = 0.0
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
        timeout_stop = not plan_ok or not status_ok

        current_speed = self._get_current_speed()
        target_speed = 0.0
        target_steer = 0.0
        raw_throttle = 0.0

        if plan_ok:
            target_speed = float(self.last_plan.get("target_speed_mps", 0.0))
            target_steer = float(self.last_plan.get("steer", 0.0))

        if timeout_stop:
            self.integral = 0.0
            throttle = 0.0
            brake = 1.0
            steer = 0.0
        else:
            error = target_speed - current_speed
            dt = 1.0 / max(1.0, self.rate_hz)
            if target_speed <= 0.05 or error < 0:
                self.integral += error * dt * 0.5
            else:
                self.integral += error * dt

            self.integral = _clamp(self.integral, -self.integral_limit, self.integral_limit)

            raw_throttle = self.kp * error + self.ki * self.integral

            if current_speed > self.min_speed_for_throttle_floor_mps and target_speed > 0.2:
                raw_throttle = max(raw_throttle, self.throttle_floor_when_moving)

            if error > 0.5:
                raw_throttle += self.uphill_speed_error_boost * min(error, 1.0)

            throttle = _clamp(raw_throttle, 0.0, self.max_throttle)
            brake = 0.0

            if error < -0.05:
                brake = _clamp(-self.brake_kp * error, 0.0, self.max_brake)
                throttle = 0.0

            if target_speed <= 0.05 and current_speed > 0.1:
                brake = self.max_brake
                throttle = 0.0

            delta = throttle - self.last_throttle
            delta = _clamp(delta, -self.throttle_slew_limit, self.throttle_slew_limit)
            throttle = _clamp(self.last_throttle + delta, 0.0, 1.0)

            if brake > 0.05:
                throttle = 0.0
            elif throttle > 0.05:
                brake = 0.0

            if brake > 0.05 or target_speed <= 0.05:
                self.integral = 0.0

            steer = _clamp(target_steer, -1.0, 1.0)
            steer_delta = steer - self.last_steer
            steer = self.last_steer + _clamp(steer_delta, -self.steer_rate_limit, self.steer_rate_limit)
            steer = _clamp(steer, -1.0, 1.0)
            self.last_steer = steer

        self.last_throttle = throttle

        cmd = {
            "stamp": now,
            "source": "phase2_control",
            "target_speed_mps": target_speed,
            "current_speed_mps": current_speed,
            "throttle": round(float(throttle), 3),
            "brake": round(float(brake), 3),
            "steer": round(float(self.last_steer), 3),
            "reverse": False,
            "hand_brake": False,
        }

        m = String()
        m.data = json.dumps(cmd)
        self.cmd_pub.publish(m)

        dbg = String()
        dbg.data = json.dumps({
            "target_speed_mps": target_speed,
            "current_speed_mps": current_speed,
            "speed_error": round(target_speed - current_speed, 3),
            "integral": round(self.integral, 3),
            "raw_throttle": round(raw_throttle if not timeout_stop else 0.0, 3),
            "throttle": round(float(throttle), 3),
            "brake": round(float(brake), 3),
            "steer": round(float(self.last_steer), 3),
            "timeout_stop": timeout_stop,
        })
        self.debug_pub.publish(dbg)


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
