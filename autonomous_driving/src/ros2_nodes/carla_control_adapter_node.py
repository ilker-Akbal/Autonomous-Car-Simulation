import glob
import json
import math
import os
import sys
import time


def load_carla(carla_root: str):
    egg_pattern = os.path.join(
        carla_root,
        "PythonAPI",
        "carla",
        "dist",
        "carla-*%d.%d-%s.egg" % (
            sys.version_info.major,
            sys.version_info.minor,
            "linux-x86_64",
        ),
    )

    eggs = glob.glob(egg_pattern)
    if eggs and eggs[0] not in sys.path:
        sys.path.append(eggs[0])

    import carla
    return carla


import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class CarlaControlAdapterNode(Node):
    def __init__(self):
        super().__init__("carla_control_adapter_node")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("behavior_topic", "/adas/phase1/behavior")
        self.declare_parameter("debug_topic", "/adas/carla/control_debug")

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_throttle", 0.55)
        self.declare_parameter("max_brake", 1.0)
        self.declare_parameter("speed_kp", 0.45)
        self.declare_parameter("speed_ki", 0.02)
        self.declare_parameter("speed_kd", 0.03)
        self.declare_parameter("max_steer", 0.55)
        self.declare_parameter("steer_alpha", 0.35)
        self.declare_parameter("throttle_slew_per_sec", 1.0)
        self.declare_parameter("brake_slew_per_sec", 2.5)
        self.declare_parameter("behavior_timeout_s", 1.0)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.behavior_topic = self.get_parameter("behavior_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value

        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)
        self.speed_kp = float(self.get_parameter("speed_kp").value)
        self.speed_ki = float(self.get_parameter("speed_ki").value)
        self.speed_kd = float(self.get_parameter("speed_kd").value)
        self.max_steer = float(self.get_parameter("max_steer").value)
        self.steer_alpha = float(self.get_parameter("steer_alpha").value)
        self.throttle_slew_per_sec = float(self.get_parameter("throttle_slew_per_sec").value)
        self.brake_slew_per_sec = float(self.get_parameter("brake_slew_per_sec").value)
        self.behavior_timeout_s = float(self.get_parameter("behavior_timeout_s").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()

        self.ego_vehicle = self.wait_for_ego_vehicle()

        self.current_behavior = {
            "decision": "STOP",
            "target_speed": 0.0,
            "steering_target": 0.0,
            "reason": "initial_stop",
        }
        self.last_behavior_stamp = time.time()

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()
        self.filtered_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0

        self.sub = self.create_subscription(
            String,
            self.behavior_topic,
            self.behavior_callback,
            10,
        )

        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.control_loop)

        self.get_logger().info("CARLA control adapter hazır")
        self.get_logger().info(f"{self.behavior_topic} -> CARLA VehicleControl")

    def wait_for_ego_vehicle(self):
        for _ in range(100):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_world_manager_node çalışmalı.")

    def behavior_callback(self, msg):
        try:
            data = json.loads(msg.data)
            decision = str(data.get("decision", "STOP")).upper()
            self.current_behavior = {
                "decision": decision,
                "target_speed": max(0.0, float(data.get("target_speed", 0.0))),
                "steering_target": self.clamp(
                    data.get("steering_target", 0.0),
                    -self.max_steer,
                    self.max_steer,
                ),
                "reason": str(data.get("reason", "unknown")),
                "traffic_light_state": str(data.get("traffic_light_state", "unknown")),
                "stopline_distance_m": data.get("stopline_distance_m"),
                "stop_required": bool(data.get("stop_required", False)),
            }
            self.last_behavior_stamp = time.time()

        except Exception as exc:
            self.get_logger().warn(f"behavior parse hata: {exc}")
            self.current_behavior = {
                "decision": "STOP",
                "target_speed": 0.0,
                "steering_target": 0.0,
                "reason": "behavior_parse_error",
            }

    def get_speed_mps(self):
        velocity = self.ego_vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(float(value), max_value))

    def compute_speed_control(self, target_speed, current_speed, dt):
        error = target_speed - current_speed
        self.integral_error += error * dt
        derivative = (error - self.prev_error) / dt if dt > 1e-4 else 0.0

        raw = (
            self.speed_kp * error
            + self.speed_ki * self.integral_error
            + self.speed_kd * derivative
        )

        self.prev_error = error

        if raw >= 0.0:
            throttle = self.clamp(raw, 0.0, self.max_throttle)
            brake = 0.0
        else:
            throttle = 0.0
            brake = self.clamp(abs(raw), 0.0, self.max_brake)

        return throttle, brake, error

    def slew(self, target, previous, rate_per_sec, dt):
        step = max(0.0, float(rate_per_sec)) * max(0.0, float(dt))
        if target > previous + step:
            return previous + step
        if target < previous - step:
            return previous - step
        return target

    def control_loop(self):
        now = time.time()
        dt = max(1e-3, now - self.prev_time)
        self.prev_time = now

        current_speed = self.get_speed_mps()

        control = self.carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False

        behavior_age = now - self.last_behavior_stamp

        if behavior_age > self.behavior_timeout_s:
            decision = "STOP"
            target_speed = 0.0
            steering_target = 0.0
            reason = "behavior_timeout"
        else:
            decision = str(self.current_behavior.get("decision", "STOP")).upper()
            target_speed = float(self.current_behavior.get("target_speed", 0.0))
            steering_target = self.clamp(
                self.current_behavior.get("steering_target", 0.0),
                -self.max_steer,
                self.max_steer,
            )
            reason = str(self.current_behavior.get("reason", "unknown"))

        if decision == "STOP":
            throttle = 0.0
            brake = 1.0 if current_speed > 0.05 else 0.7
            self.integral_error = 0.0
        else:
            throttle, brake, speed_error = self.compute_speed_control(
                target_speed,
                current_speed,
                dt,
            )

        self.filtered_steer = (
            self.steer_alpha * steering_target
            + (1.0 - self.steer_alpha) * self.filtered_steer
        )
        steer = self.clamp(self.filtered_steer, -self.max_steer, self.max_steer)

        throttle = self.slew(
            self.clamp(throttle, 0.0, self.max_throttle),
            self.prev_throttle,
            self.throttle_slew_per_sec,
            dt,
        )
        brake = self.slew(
            self.clamp(brake, 0.0, self.max_brake),
            self.prev_brake,
            self.brake_slew_per_sec,
            dt,
        )
        self.prev_throttle = throttle
        self.prev_brake = brake

        if brake > 0.05:
            throttle = 0.0

        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steer)

        self.ego_vehicle.apply_control(control)

        payload = {
            "stamp": now,
            "decision": decision,
            "reason": reason,
            "target_speed": round(target_speed, 3),
            "current_speed": round(current_speed, 3),
            "throttle": round(control.throttle, 3),
            "brake": round(control.brake, 3),
            "steer": round(control.steer, 3),
            "steering_target": round(float(steering_target), 3),
            "behavior_age": round(behavior_age, 3),
            "traffic_light_state": self.current_behavior.get("traffic_light_state", "unknown"),
            "stopline_distance_m": self.current_behavior.get("stopline_distance_m"),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.debug_pub.publish(msg)

        self.get_logger().info(
            f"[CARLA CONTROL] decision={decision} target={target_speed:.2f} "
            f"speed={current_speed:.2f} throttle={control.throttle:.2f} "
            f"brake={control.brake:.2f} steer={control.steer:.2f} reason={reason}",
            throttle_duration_sec=0.5,
        )


def main(args=None):
    rclpy.init(args=args)
    node = CarlaControlAdapterNode()

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
