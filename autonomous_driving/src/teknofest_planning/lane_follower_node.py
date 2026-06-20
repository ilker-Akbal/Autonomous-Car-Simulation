import json
import math
import time
from typing import Any

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_planning.velocity_profile import (
    clamp,
    compute_target_speed_from_route,
)


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("base_lookahead_m", 5.0)
        self.declare_parameter("lookahead_gain", 1.2)
        self.declare_parameter("min_lookahead_m", 4.5)
        self.declare_parameter("max_lookahead_m", 14.0)
        self.declare_parameter("dynamic_lookahead_enabled", True)
        self.declare_parameter("wheel_base_m", 2.8)
        self.declare_parameter("max_steer_angle_rad", 0.65)
        self.declare_parameter("target_speed_mps", 3.0)
        self.declare_parameter("cruise_speed_mps", 4.5)
        self.declare_parameter("max_speed_mps", 6.0)
        self.declare_parameter("min_turn_speed_mps", 2.0)
        self.declare_parameter("sharp_turn_yaw_deg", 45.0)
        self.declare_parameter("moderate_turn_yaw_deg", 18.0)
        self.declare_parameter("speed_slew_up_mps_per_s", 0.8)
        self.declare_parameter("speed_slew_down_mps_per_s", 2.0)
        self.declare_parameter("rate_hz", 20.0)

        self.base_lookahead_m = float(self.get_parameter("base_lookahead_m").value)
        self.lookahead_gain = float(self.get_parameter("lookahead_gain").value)
        self.min_lookahead_m = float(self.get_parameter("min_lookahead_m").value)
        self.max_lookahead_m = float(self.get_parameter("max_lookahead_m").value)
        self.dynamic_lookahead_enabled = bool(self.get_parameter("dynamic_lookahead_enabled").value)
        self.wheel_base_m = float(self.get_parameter("wheel_base_m").value)
        self.max_steer_angle_rad = float(self.get_parameter("max_steer_angle_rad").value)
        self.target_speed_mps_param = float(self.get_parameter("target_speed_mps").value)
        self.cruise_speed_mps = float(self.get_parameter("cruise_speed_mps").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.min_turn_speed_mps = float(self.get_parameter("min_turn_speed_mps").value)
        self.sharp_turn_yaw_deg = float(self.get_parameter("sharp_turn_yaw_deg").value)
        self.moderate_turn_yaw_deg = float(self.get_parameter("moderate_turn_yaw_deg").value)
        self.speed_slew_up_mps_per_s = float(self.get_parameter("speed_slew_up_mps_per_s").value)
        self.speed_slew_down_mps_per_s = float(self.get_parameter("speed_slew_down_mps_per_s").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        if self.target_speed_mps_param != 3.0 and self.cruise_speed_mps == 4.5:
            self.cruise_speed_mps = self.target_speed_mps_param

        self.route = None
        self.ego = None
        self.last_route_time = 0.0
        self.last_status_time = 0.0
        self.last_target_speed = 0.0

        self.create_subscription(String, "/adas/planning/route", self._route_cb, 10)
        self.create_subscription(String, "/adas/carla/status", self._status_cb, 10)

        self.plan_pub = self.create_publisher(String, "/adas/planning/lane_plan", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/lane_debug", 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._run)

    def _route_cb(self, msg: String):
        try:
            self.route = json.loads(msg.data)
            self.last_route_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse route JSON")

    def _status_cb(self, msg: String):
        try:
            self.ego = json.loads(msg.data)
            self.last_status_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse carla status JSON")

    def _find_nearest_index(self, points: list[dict[str, Any]], ego_x: float, ego_y: float) -> int:
        nearest_index = 0
        nearest_dist = float("inf")
        for index, pt in enumerate(points):
            dx = float(pt.get("x", 0.0)) - ego_x
            dy = float(pt.get("y", 0.0)) - ego_y
            dist = math.hypot(dx, dy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_index = index
        return nearest_index

    def _run(self):
        now = time.time()
        status_ok = (now - self.last_status_time) < 1.0
        route_ok = (self.route is not None) and (now - self.last_route_time) < 2.0

        if not status_ok or not route_ok:
            plan = {
                "stamp": now,
                "source": "phase2b_pure_pursuit",
                "cruise_speed_mps": self.cruise_speed_mps,
                "target_speed_mps": 0.0,
                "turn_intensity": 0.0,
                "speed_reason": "route_invalid",
                "nearest_index": None,
                "selected_target_index": None,
                "lookahead_m": self.base_lookahead_m,
                "steer": 0.0,
                "route_ok": bool(route_ok),
                "status_ok": bool(status_ok),
                "target": None,
            }
            msg = String()
            msg.data = json.dumps(plan)
            self.plan_pub.publish(msg)
            return

        loc = self.ego.get("location", {})
        rot = self.ego.get("rotation", {})
        ego_x = float(loc.get("x", 0.0))
        ego_y = float(loc.get("y", 0.0))
        ego_yaw_deg = float(rot.get("yaw", 0.0))
        ego_yaw = math.radians(ego_yaw_deg)
        speed = float(self.ego.get("speed_mps", 0.0))

        points = self.route.get("points", [])
        if not points:
            plan = {
                "stamp": now,
                "source": "phase2b_pure_pursuit",
                "cruise_speed_mps": self.cruise_speed_mps,
                "target_speed_mps": 0.0,
                "turn_intensity": 0.0,
                "speed_reason": "route_invalid",
                "nearest_index": None,
                "selected_target_index": None,
                "lookahead_m": self.base_lookahead_m,
                "steer": 0.0,
                "route_ok": False,
                "status_ok": True,
                "target": None,
            }
            msg = String(); msg.data = json.dumps(plan); self.plan_pub.publish(msg)
            return

        nearest_index = self._find_nearest_index(points, ego_x, ego_y)
        profile = compute_target_speed_from_route(
            points,
            nearest_index,
            cruise_speed_mps=self.cruise_speed_mps,
            min_turn_speed_mps=self.min_turn_speed_mps,
            max_speed_mps=self.max_speed_mps,
            moderate_turn_yaw_deg=self.moderate_turn_yaw_deg,
            sharp_turn_yaw_deg=self.sharp_turn_yaw_deg,
        )

        target_speed = float(profile["target_speed_mps"])
        speed_error = target_speed - self.last_target_speed
        if speed_error > 0:
            max_delta = self.speed_slew_up_mps_per_s / max(1.0, self.rate_hz)
        else:
            max_delta = self.speed_slew_down_mps_per_s / max(1.0, self.rate_hz)

        self.last_target_speed = clamp(
            self.last_target_speed + clamp(speed_error, -max_delta, max_delta),
            0.0,
            self.max_speed_mps,
        )

        lookahead_m = self.base_lookahead_m
        if self.dynamic_lookahead_enabled:
            lookahead_m = self.base_lookahead_m + min(
                self.max_lookahead_m - self.base_lookahead_m,
                self.last_target_speed * self.lookahead_gain,
            )
            if profile["turn_intensity"] >= self.moderate_turn_yaw_deg:
                lookahead_m = clamp(lookahead_m * 0.75, self.min_lookahead_m, self.max_lookahead_m)

        lookahead_m = clamp(lookahead_m, self.min_lookahead_m, self.max_lookahead_m)

        target = None
        selected_target_index = None
        for index, pt in enumerate(points):
            dx = float(pt.get("x", 0.0)) - ego_x
            dy = float(pt.get("y", 0.0)) - ego_y
            dist = math.hypot(dx, dy)
            if dist >= lookahead_m:
                target = pt
                selected_target_index = index
                break

        if target is None:
            target = points[-1]
            selected_target_index = len(points) - 1

        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))

        dx = tx - ego_x
        dy = ty - ego_y
        local_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        local_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy

        if lookahead_m <= 0.0:
            steering_angle = 0.0
        else:
            steering_angle = math.atan2(
                2.0 * self.wheel_base_m * local_y,
                max(1e-6, lookahead_m * lookahead_m),
            )

        steer_norm = clamp(steering_angle / self.max_steer_angle_rad, -1.0, 1.0)

        if abs(steer_norm) > 0.35:
            self.last_target_speed = min(self.last_target_speed, self.min_turn_speed_mps)

        plan = {
            "stamp": now,
            "source": "phase2b_pure_pursuit",
            "cruise_speed_mps": self.cruise_speed_mps,
            "target_speed_mps": round(self.last_target_speed, 3),
            "turn_intensity": round(float(profile["turn_intensity"]), 3),
            "speed_reason": profile["speed_reason"],
            "nearest_index": nearest_index,
            "selected_target_index": selected_target_index,
            "lookahead_m": round(lookahead_m, 3),
            "steer": float(steer_norm),
            "route_ok": True,
            "status_ok": True,
            "target": {"x": tx, "y": ty},
        }

        msg = String()
        msg.data = json.dumps(plan)
        self.plan_pub.publish(msg)

        dbg = String()
        dbg.data = json.dumps({
            "stamp": now,
            "local_x": local_x,
            "local_y": local_y,
            "steer": steer_norm,
            "cruise_speed_mps": self.cruise_speed_mps,
            "target_speed_mps": self.last_target_speed,
            "turn_intensity": profile["turn_intensity"],
            "speed_reason": profile["speed_reason"],
        })
        self.debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
