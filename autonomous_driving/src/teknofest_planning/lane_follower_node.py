import json
import math
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


def _clamp(v, a, b):
    return max(a, min(b, v))


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("base_lookahead_m", 4.0)
        self.declare_parameter("lookahead_gain", 0.8)
        self.declare_parameter("min_lookahead_m", 4.0)
        self.declare_parameter("max_lookahead_m", 10.0)
        self.declare_parameter("wheel_base_m", 2.8)
        self.declare_parameter("max_steer_angle_rad", 0.65)
        self.declare_parameter("target_speed_mps", 3.0)
        self.declare_parameter("turn_speed_mps", 2.0)
        self.declare_parameter("rate_hz", 20.0)

        self.base_lookahead_m = float(self.get_parameter("base_lookahead_m").value)
        self.lookahead_gain = float(self.get_parameter("lookahead_gain").value)
        self.min_lookahead_m = float(self.get_parameter("min_lookahead_m").value)
        self.max_lookahead_m = float(self.get_parameter("max_lookahead_m").value)
        self.wheel_base_m = float(self.get_parameter("wheel_base_m").value)
        self.max_steer_angle_rad = float(self.get_parameter("max_steer_angle_rad").value)
        self.target_speed_mps_default = float(self.get_parameter("target_speed_mps").value)
        self.turn_speed_mps = float(self.get_parameter("turn_speed_mps").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.route = None
        self.ego = None
        self.last_route_time = 0.0
        self.last_status_time = 0.0

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

    def _run(self):
        now = time.time()
        status_ok = (now - self.last_status_time) < 1.0
        route_ok = (self.route is not None) and (now - self.last_route_time) < 2.0

        if not status_ok or not route_ok:
            # publish stop plan
            plan = {
                "stamp": now,
                "source": "phase2_pure_pursuit",
                "target_speed_mps": 0.0,
                "steer": 0.0,
                "lookahead_m": self.base_lookahead_m,
                "target": None,
                "route_ok": bool(route_ok),
                "status_ok": bool(status_ok),
            }
            m = String(); m.data = json.dumps(plan); self.plan_pub.publish(m)
            return

        # parse ego
        loc = self.ego.get("location", {})
        rot = self.ego.get("rotation", {})
        ego_x = float(loc.get("x", 0.0))
        ego_y = float(loc.get("y", 0.0))
        ego_yaw_deg = float(rot.get("yaw", 0.0))
        ego_yaw = math.radians(ego_yaw_deg)
        speed = float(self.ego.get("speed_mps", 0.0))

        # select target from route points
        points = self.route.get("points", [])
        if not points:
            plan = {"stamp": now, "source": "phase2_pure_pursuit", "target_speed_mps": 0.0, "steer": 0.0, "lookahead_m": self.base_lookahead_m, "target": None, "route_ok": False, "status_ok": True}
            m = String(); m.data = json.dumps(plan); self.plan_pub.publish(m); return

        # determine lookahead
        lookahead = self.base_lookahead_m + speed * self.lookahead_gain
        lookahead = _clamp(lookahead, self.min_lookahead_m, self.max_lookahead_m)

        # find first point at distance >= lookahead
        target = None
        for pt in points:
            dx = float(pt.get("x", 0.0)) - ego_x
            dy = float(pt.get("y", 0.0)) - ego_y
            dist = math.hypot(dx, dy)
            if dist >= lookahead:
                target = pt
                break

        if target is None:
            target = points[-1]

        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))

        # transform to vehicle frame
        dx = tx - ego_x
        dy = ty - ego_y
        # vehicle local x forward, y left -> rotate by -yaw
        local_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        local_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy

        # pure pursuit steering
        if lookahead <= 0.0:
            steering_angle = 0.0
        else:
            steering_angle = math.atan2(2.0 * self.wheel_base_m * local_y, max(1e-6, lookahead * lookahead))

        steer_norm = _clamp(steering_angle / self.max_steer_angle_rad, -1.0, 1.0)

        target_speed = float(self.route.get("target_speed_mps", self.target_speed_mps_default)) if self.route.get("target_speed_mps") is not None else self.target_speed_mps_default
        if abs(steer_norm) > 0.35:
            target_speed = min(target_speed, self.turn_speed_mps)

        plan = {
            "stamp": now,
            "source": "phase2_pure_pursuit",
            "target_speed_mps": target_speed,
            "steer": float(steer_norm),
            "lookahead_m": lookahead,
            "target": {"x": tx, "y": ty},
            "route_ok": True,
            "status_ok": True,
        }

        m = String(); m.data = json.dumps(plan); self.plan_pub.publish(m)
        dbg = String(); dbg.data = json.dumps({"stamp": now, "local_x": local_x, "local_y": local_y, "steer": steer_norm}); self.debug_pub.publish(dbg)


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
