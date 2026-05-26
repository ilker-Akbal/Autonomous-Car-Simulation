import glob
import json
import math
import os
import sys
import time
from typing import Optional


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

        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("debug_topic", "/adas/carla/control_debug")

        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_throttle", 0.55)
        self.declare_parameter("max_brake", 1.0)
        self.declare_parameter("speed_kp", 0.45)
        self.declare_parameter("speed_ki", 0.02)
        self.declare_parameter("speed_kd", 0.03)
        self.declare_parameter("default_go_speed", 1.5)
        self.declare_parameter("default_slow_speed", 0.8)

        self.declare_parameter("enable_lane_keep", True)
        self.declare_parameter("lookahead_distance", 6.0)
        self.declare_parameter("steer_kp", 0.025)
        self.declare_parameter("max_steer", 0.45)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value

        self.decision_topic = self.get_parameter("decision_topic").value
        self.debug_topic = self.get_parameter("debug_topic").value
    
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_brake = float(self.get_parameter("max_brake").value)
        self.speed_kp = float(self.get_parameter("speed_kp").value)
        self.speed_ki = float(self.get_parameter("speed_ki").value)
        self.speed_kd = float(self.get_parameter("speed_kd").value)
        self.default_go_speed = float(self.get_parameter("default_go_speed").value)
        self.default_slow_speed = float(self.get_parameter("default_slow_speed").value)

        self.enable_lane_keep = bool(self.get_parameter("enable_lane_keep").value)
        self.lookahead_distance = float(self.get_parameter("lookahead_distance").value)
        self.steer_kp = float(self.get_parameter("steer_kp").value)
        self.max_steer = float(self.get_parameter("max_steer").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        self.ego_vehicle = self.wait_for_ego_vehicle()

        self.current_decision = "STOP"
        self.current_target_speed = 0.0
        self.current_reason = "initial_stop"
        self.current_risk = "UNKNOWN"
        self.last_decision_stamp = time.time()

        self.integral_error = 0.0
        self.prev_error = 0.0
        self.prev_time = time.time()

        self.sub = self.create_subscription(
            String,
            self.decision_topic,
            self.decision_callback,
            10,
        )

        self.debug_pub = self.create_publisher(String, self.debug_topic, 10)

        rate = float(self.get_parameter("control_rate_hz").value)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self.control_loop)

        self.get_logger().info("CARLA control adapter hazır")
        self.get_logger().info(f"{self.decision_topic} -> CARLA VehicleControl")

    def wait_for_ego_vehicle(self):
        for _ in range(100):
            vehicles = self.world.get_actors().filter("vehicle.*")
            for vehicle in vehicles:
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    return vehicle
            time.sleep(0.2)

        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_world_manager_node çalışmalı.")

    def decision_callback(self, msg):
        try:
            data = json.loads(msg.data)
            decision = str(data.get("decision", "STOP")).upper()
            target_speed = data.get("target_speed", None)

            if target_speed is None:
                if decision == "GO":
                    target_speed = self.default_go_speed
                elif decision == "SLOW":
                    target_speed = self.default_slow_speed
                else:
                    target_speed = 0.0

            self.current_decision = decision
            self.current_target_speed = max(0.0, float(target_speed))
            self.current_reason = str(data.get("reason", "unknown"))
            self.current_risk = str(data.get("risk", "UNKNOWN"))
            self.last_decision_stamp = time.time()

        except Exception as exc:
            self.get_logger().warn(f"decision parse hata: {exc}")
            self.current_decision = "STOP"
            self.current_target_speed = 0.0
            self.current_reason = "decision_parse_error"
            self.current_risk = "UNKNOWN"

    def get_speed_mps(self):
        velocity = self.ego_vehicle.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def clamp(self, value, min_value, max_value):
        return max(min_value, min(float(value), max_value))

    def normalize_angle(self, angle_deg):
        while angle_deg > 180.0:
            angle_deg -= 360.0
        while angle_deg < -180.0:
            angle_deg += 360.0
        return angle_deg

    def compute_lane_keep_steer(self):
        if not self.enable_lane_keep:
            return 0.0

        transform = self.ego_vehicle.get_transform()
        location = transform.location
        vehicle_yaw = transform.rotation.yaw

        waypoint = self.map.get_waypoint(
            location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if waypoint is None:
            return 0.0

        next_waypoints = waypoint.next(self.lookahead_distance)

        if not next_waypoints:
            return 0.0

        target_wp = next_waypoints[0]
        target_yaw = target_wp.transform.rotation.yaw

        heading_error = self.normalize_angle(target_yaw - vehicle_yaw)
        steer = self.clamp(self.steer_kp * heading_error, -self.max_steer, self.max_steer)
        return steer

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

    def control_loop(self):
        now = time.time()
        dt = now - self.prev_time
        self.prev_time = now

        current_speed = self.get_speed_mps()

        control = self.carla.VehicleControl()
        control.hand_brake = False
        control.manual_gear_shift = False

        decision_age = now - self.last_decision_stamp

        if decision_age > 2.0:
            decision = "STOP"
            target_speed = 0.0
            reason = "decision_timeout"
        else:
            decision = self.current_decision
            target_speed = self.current_target_speed
            reason = self.current_reason

        if decision == "STOP":
            throttle = 0.0
            brake = 1.0
            steer = 0.0 if current_speed < 0.2 else self.compute_lane_keep_steer()
            self.integral_error = 0.0

        else:
            throttle, brake, speed_error = self.compute_speed_control(
                target_speed,
                current_speed,
                dt,
            )
            steer = self.compute_lane_keep_steer()

            if decision == "LEFT":
                steer = self.max_steer * 0.7
            elif decision == "RIGHT":
                steer = -self.max_steer * 0.7

        control.throttle = float(throttle)
        control.brake = float(brake)
        control.steer = float(steer)

        self.ego_vehicle.apply_control(control)

        payload = {
            "stamp": now,
            "decision": decision,
            "risk": self.current_risk,
            "reason": reason,
            "target_speed": round(target_speed, 3),
            "current_speed": round(current_speed, 3),
            "throttle": round(control.throttle, 3),
            "brake": round(control.brake, 3),
            "steer": round(control.steer, 3),
            "decision_age": round(decision_age, 3),
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