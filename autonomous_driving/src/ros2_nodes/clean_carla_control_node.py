#!/usr/bin/env python3
import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


def clamp(value, low, high):
    return max(low, min(high, value))


def rate_limit(current, target, max_delta):
    delta = clamp(target - current, -max_delta, max_delta)
    return current + delta


class CleanCarlaControlNode(Node):
    def __init__(self):
        super().__init__("clean_carla_control_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("command_topic", "/adas/phase1/command")
        self.declare_parameter("control_hz", 20.0)
        self.declare_parameter("command_timeout_s", 0.8)

        self.declare_parameter("speed_kp", 0.22)
        self.declare_parameter("speed_ki", 0.035)
        self.declare_parameter("speed_kd", 0.015)
        self.declare_parameter("integral_limit", 6.0)
        self.declare_parameter("max_throttle", 0.62)
        self.declare_parameter("max_service_brake", 0.55)
        self.declare_parameter("stop_brake", 0.82)
        self.declare_parameter("emergency_brake", 1.0)

        self.declare_parameter("steer_limit", 0.70)
        self.declare_parameter("steer_lowpass_alpha", 0.18)
        self.declare_parameter("steer_rate_limit_per_s", 0.85)
        self.declare_parameter("throttle_rate_limit_per_s", 1.20)
        self.declare_parameter("brake_rate_limit_per_s", 2.20)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.command_topic = str(self.get_parameter("command_topic").value)
        self.control_hz = float(self.get_parameter("control_hz").value)
        self.command_timeout_s = float(self.get_parameter("command_timeout_s").value)

        self.speed_kp = float(self.get_parameter("speed_kp").value)
        self.speed_ki = float(self.get_parameter("speed_ki").value)
        self.speed_kd = float(self.get_parameter("speed_kd").value)
        self.integral_limit = float(self.get_parameter("integral_limit").value)
        self.max_throttle = float(self.get_parameter("max_throttle").value)
        self.max_service_brake = float(self.get_parameter("max_service_brake").value)
        self.stop_brake = float(self.get_parameter("stop_brake").value)
        self.emergency_brake = float(self.get_parameter("emergency_brake").value)

        self.steer_limit = float(self.get_parameter("steer_limit").value)
        self.steer_lowpass_alpha = float(self.get_parameter("steer_lowpass_alpha").value)
        self.steer_rate_limit_per_s = float(self.get_parameter("steer_rate_limit_per_s").value)
        self.throttle_rate_limit_per_s = float(
            self.get_parameter("throttle_rate_limit_per_s").value
        )
        self.brake_rate_limit_per_s = float(self.get_parameter("brake_rate_limit_per_s").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.ego = None
        self.last_ego_lookup_s = 0.0

        self.latest_command = None
        self.latest_command_s = 0.0
        self.speed_integral = 0.0
        self.last_speed_error = 0.0
        self.last_control_s = time.time()
        self.filtered_steer = 0.0
        self.last_throttle = 0.0
        self.last_brake = 0.0
        self.last_log_s = 0.0

        self.create_subscription(String, self.command_topic, self.command_cb, 10)
        self.timer = self.create_timer(1.0 / max(1.0, self.control_hz), self.tick)

        self.get_logger().info(
            f"clean_carla_control_node ready: {self.host}:{self.port} command={self.command_topic}"
        )

    def command_cb(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"Command JSON parse failed: {exc}")
            return

        self.latest_command = data
        self.latest_command_s = time.time()

    def find_ego(self):
        now = time.time()

        if self.ego is not None:
            try:
                if self.ego.is_alive:
                    return self.ego
            except Exception:
                self.ego = None

        if now - self.last_ego_lookup_s < 1.0:
            return self.ego

        self.last_ego_lookup_s = now
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self.ego = vehicle
                self.get_logger().info(f"Clean control ego found: id={vehicle.id}")
                return self.ego

        return None

    def speed_mps(self, ego):
        velocity = ego.get_velocity()
        return math.sqrt(velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2)

    def build_control(self, ego, command, dt):
        decision = str(command.get("decision", "STOP")).upper()
        emergency = bool(command.get("emergency", False))
        brake_required = bool(command.get("brake_required", False))
        target_speed = float(command.get("target_speed", 0.0) or 0.0)
        steering_target = clamp(
            float(command.get("steering_target", 0.0) or 0.0),
            -self.steer_limit,
            self.steer_limit,
        )

        steer_lp = (
            (1.0 - self.steer_lowpass_alpha) * self.filtered_steer
            + self.steer_lowpass_alpha * steering_target
        )
        self.filtered_steer = rate_limit(
            self.filtered_steer,
            steer_lp,
            self.steer_rate_limit_per_s * dt,
        )

        speed = self.speed_mps(ego)

        if decision == "STOP" or emergency:
            self.speed_integral = 0.0
            self.last_speed_error = 0.0
            throttle_target = 0.0
            brake_target = self.emergency_brake if emergency else self.stop_brake
        else:
            error = target_speed - speed
            self.speed_integral = clamp(
                self.speed_integral + error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
            derivative = (error - self.last_speed_error) / max(1e-3, dt)
            self.last_speed_error = error

            pid = (
                self.speed_kp * error
                + self.speed_ki * self.speed_integral
                + self.speed_kd * derivative
            )

            if brake_required:
                throttle_target = 0.0
                brake_target = max(0.35, min(self.max_service_brake, -pid + 0.20))
            elif pid >= 0.0:
                throttle_target = clamp(pid, 0.0, self.max_throttle)
                brake_target = 0.0
            else:
                throttle_target = 0.0
                brake_target = clamp(-pid * 0.45, 0.0, self.max_service_brake)

            if speed < 0.35 and target_speed > 0.8 and throttle_target < 0.24:
                throttle_target = 0.24

        self.last_throttle = rate_limit(
            self.last_throttle,
            throttle_target,
            self.throttle_rate_limit_per_s * dt,
        )
        self.last_brake = rate_limit(
            self.last_brake,
            brake_target,
            self.brake_rate_limit_per_s * dt,
        )

        if (
            decision != "STOP"
            and not emergency
            and not brake_required
            and target_speed > speed + 0.5
            and brake_target <= 0.0
        ):
            if speed < 0.5:
                self.last_brake = 0.0
            else:
                self.last_brake = rate_limit(
                    self.last_brake,
                    0.0,
                    max(self.brake_rate_limit_per_s * dt, 0.12),
                )

        if self.last_brake > 0.03:
            self.last_throttle = 0.0

        control = self.carla.VehicleControl()
        control.throttle = float(clamp(self.last_throttle, 0.0, self.max_throttle))
        control.brake = float(clamp(self.last_brake, 0.0, 1.0))
        control.steer = float(clamp(self.filtered_steer, -self.steer_limit, self.steer_limit))
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False

        return control, speed

    def stop_for_missing_command(self, ego, dt):
        self.speed_integral = 0.0
        self.last_throttle = rate_limit(self.last_throttle, 0.0, self.throttle_rate_limit_per_s * dt)
        self.last_brake = rate_limit(self.last_brake, 0.45, self.brake_rate_limit_per_s * dt)
        self.filtered_steer = rate_limit(
            self.filtered_steer,
            0.0,
            self.steer_rate_limit_per_s * dt,
        )

        control = self.carla.VehicleControl()
        control.throttle = float(clamp(self.last_throttle, 0.0, self.max_throttle))
        control.brake = float(clamp(self.last_brake, 0.0, 1.0))
        control.steer = float(clamp(self.filtered_steer, -self.steer_limit, self.steer_limit))
        control.hand_brake = False
        control.reverse = False
        control.manual_gear_shift = False
        ego.apply_control(control)

    def tick(self):
        ego = self.find_ego()
        if ego is None:
            return

        now = time.time()
        dt = clamp(now - self.last_control_s, 0.01, 0.20)
        self.last_control_s = now

        if self.latest_command is None or now - self.latest_command_s > self.command_timeout_s:
            self.stop_for_missing_command(ego, dt)
            return

        control, speed = self.build_control(ego, self.latest_command, dt)
        ego.apply_control(control)

        if now - self.last_log_s >= 1.0:
            self.last_log_s = now
            self.get_logger().info(
                "CLEAN_CONTROL "
                f"speed={speed:.2f} target={float(self.latest_command.get('target_speed', 0.0) or 0.0):.2f} "
                f"throttle={control.throttle:.2f} brake={control.brake:.2f} "
                f"steer={control.steer:.3f} decision={self.latest_command.get('decision')} "
                f"reason={self.latest_command.get('reason')}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = CleanCarlaControlNode()

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
