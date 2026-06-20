import json
import math
import time
from typing import List

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


def _yaw_diff(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


class SimpleRoutePlanner(Node):
    def __init__(self):
        super().__init__("simple_route_planner")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("horizon_m", 80.0)
        self.declare_parameter("step_m", 2.0)
        self.declare_parameter("rate_hz", 5.0)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.horizon_m = float(self.get_parameter("horizon_m").value)
        self.step_m = float(self.get_parameter("step_m").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self.route_pub = self.create_publisher(String, "/adas/planning/route", 10)
        self.route_debug_pub = self.create_publisher(String, "/adas/planning/route_debug", 10)

        # Subscribe to CARLA status for ego location updates
        self.create_subscription(String, "/adas/carla/status", self._status_cb, 10)

        self._last_status = None
        self._last_map = None

        # Connect to CARLA
        try:
            carla = load_carla(self.carla_root)
            self.client = carla.Client(self.host, self.port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        except Exception:
            self.client = None
            self.world = None
            self.map = None

        self.timer = self.create_timer(1.0 / max(0.1, self.rate_hz), self.publish_route)

    def _status_cb(self, msg: String):
        try:
            self._last_status = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("Failed to parse /adas/carla/status JSON")

    def _get_ego_location(self):
        if not self._last_status:
            return None
        loc = self._last_status.get("location", {})
        x = loc.get("x")
        y = loc.get("y")
        z = loc.get("z", 0.0)
        if x is None or y is None:
            return None
        return float(x), float(y), float(z)

    def publish_route(self):
        ego_loc = self._get_ego_location()
        if ego_loc is None:
            return

        if self.map is None:
            # attempt to connect
            try:
                carla = load_carla(self.carla_root)
                self.client = carla.Client(self.host, self.port)
                self.client.set_timeout(10.0)
                self.world = self.client.get_world()
                self.map = self.world.get_map()
            except Exception as exc:
                self.get_logger().warn(f"Route planner: cannot connect to CARLA: {exc}")
                return

        carla = load_carla(self.carla_root)
        Location = carla.Location
        try:
            wp = self.map.get_waypoint(Location(x=ego_loc[0], y=ego_loc[1], z=ego_loc[2]), project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception as exc:
            self.get_logger().warn(f"get_waypoint failed: {exc}")
            return

        points = []
        s = 0.0
        current = wp
        prev_yaw = current.transform.rotation.yaw
        max_steps = int(self.horizon_m / max(0.001, self.step_m))

        for i in range(max_steps):
            # advance
            next_wps = current.next(self.step_m)
            if not next_wps:
                break
            # choose best by yaw change
            best = None
            best_delta = None
            for cand in next_wps:
                dyaw = _yaw_diff(cand.transform.rotation.yaw, prev_yaw)
                if best is None or dyaw < best_delta:
                    best = cand
                    best_delta = dyaw

            if best is None:
                break

            current = best
            prev_yaw = current.transform.rotation.yaw
            s += self.step_m

            pt = {
                "x": round(current.transform.location.x, 3),
                "y": round(current.transform.location.y, 3),
                "z": round(current.transform.location.z, 3),
                "yaw": round(current.transform.rotation.yaw, 3),
                "road_id": current.road_id,
                "lane_id": current.lane_id,
                "s": round(s, 3),
            }
            points.append(pt)

        payload = {
            "stamp": time.time(),
            "frame": "carla_map",
            "points": points,
            "ego_road_id": wp.road_id if wp is not None else None,
            "ego_lane_id": wp.lane_id if wp is not None else None,
            "route_len": len(points),
        }

        msg = String()
        msg.data = json.dumps(payload)
        self.route_pub.publish(msg)

        dbg = String()
        dbg.data = json.dumps({"stamp": time.time(), "points": len(points)})
        self.route_debug_pub.publish(dbg)


def main(args=None):
    rclpy.init(args=args)
    node = SimpleRoutePlanner()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
