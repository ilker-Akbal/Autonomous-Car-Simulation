import copy
import json
import time
from typing import Any, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_planning.route_geometry import (
    build_local_route_segment,
    forward_window_search,
)
from teknofest_sim.carla_loader import load_carla


class RouteSampler(Node):
    def __init__(self):
        super().__init__("route_sampler")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("route_source_mode", "global")
        self.declare_parameter("fallback_to_simple_forward_route", True)
        self.declare_parameter("local_route_horizon_m", 80.0)
        self.declare_parameter("min_route_points", 8)
        self.declare_parameter("hold_last_route_s", 2.0)
        self.declare_parameter("rate_hz", 5.0)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.route_source_mode = self.get_parameter("route_source_mode").value
        self.fallback_to_simple_forward_route = bool(self.get_parameter("fallback_to_simple_forward_route").value)
        self.local_route_horizon_m = float(self.get_parameter("local_route_horizon_m").value)
        self.min_route_points = int(self.get_parameter("min_route_points").value)
        self.hold_last_route_s = float(self.get_parameter("hold_last_route_s").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self._last_status: Optional[dict[str, Any]] = None
        self._last_global_route: Optional[dict[str, Any]] = None
        self._last_route_time = 0.0
        self._last_index = 0
        self._last_valid_local_route: Optional[dict[str, Any]] = None
        self._last_valid_local_route_time = 0.0

        self._carla = None
        self._client = None
        self._world = None
        self._map = None

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_subscription(String, "/adas/planning/global_route", self._global_route_callback, 10)

        self.route_pub = self.create_publisher(String, "/adas/planning/route", 10)
        self.route_debug_pub = self.create_publisher(String, "/adas/planning/route_debug", 10)

        self._connect_to_carla()
        timer_period_s = 1.0 / max(0.1, self.rate_hz)
        self.timer = self.create_timer(timer_period_s, self._tick)

    def _connect_to_carla(self) -> None:
        try:
            self._carla = load_carla(self.carla_root)
            self._client = self._carla.Client(self.host, self.port)
            self._client.set_timeout(10.0)
            self._world = self._client.get_world()
            self._map = self._world.get_map()
        except Exception as exc:
            self.get_logger().warn(f"RouteSampler: CARLA connection failed: {exc}")
            self._carla = None
            self._client = None
            self._world = None
            self._map = None

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("RouteSampler: failed to parse /adas/carla/status JSON")

    def _global_route_callback(self, msg: String) -> None:
        try:
            self._last_global_route = json.loads(msg.data)
            self._last_route_time = time.time()
        except Exception:
            self.get_logger().warn("RouteSampler: failed to parse /adas/planning/global_route JSON")

    def _get_ego_location(self) -> Optional[tuple[float, float, float]]:
        if self._last_status is None:
            return None
        loc = self._last_status.get("location", {})
        if "x" not in loc or "y" not in loc:
            return None
        return float(loc["x"]), float(loc["y"]), float(loc.get("z", 0.0))

    def _build_forward_route(self) -> dict[str, Any]:
        if self._map is None or self._last_status is None:
            return self._empty_route_payload("fallback")

        ego_loc = self._get_ego_location()
        if ego_loc is None:
            return self._empty_route_payload("fallback")

        carla = self._carla
        if carla is None:
            return self._empty_route_payload("fallback")

        try:
            start_waypoint = self._map.get_waypoint(
                carla.Location(x=ego_loc[0], y=ego_loc[1], z=ego_loc[2]),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception as exc:
            self.get_logger().warn(f"RouteSampler: fallback waypoint lookup failed: {exc}")
            return self._empty_route_payload("fallback")

        points: List[dict[str, Any]] = []
        distance = 0.0
        previous = start_waypoint
        step_m = 2.0
        max_steps = int(self.local_route_horizon_m / max(0.001, step_m))
        yaw = previous.transform.rotation.yaw

        for _ in range(max_steps):
            next_waypoints = previous.next(step_m)
            if not next_waypoints:
                break
            best = None
            best_delta = None
            for candidate in next_waypoints:
                delta = abs((candidate.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0)
                if best is None or delta < best_delta:
                    best = candidate
                    best_delta = delta
            if best is None:
                break
            distance += step_m
            previous = best
            yaw = previous.transform.rotation.yaw
            points.append(
                {
                    "x": round(previous.transform.location.x, 3),
                    "y": round(previous.transform.location.y, 3),
                    "z": round(previous.transform.location.z, 3),
                    "yaw": round(previous.transform.rotation.yaw, 3),
                    "road_id": previous.road_id,
                    "lane_id": previous.lane_id,
                    "s": round(distance, 3),
                }
            )

        return {
            "stamp": time.time(),
            "source": "fallback_forward_route",
            "route_source": "fallback",
            "points": points,
            "route_len": len(points),
            "route_ok": len(points) >= self.min_route_points,
            "status_ok": True,
            "fallback_used": True,
            "global_route_ok": False,
            "held_last_route": False,
        }

    def _empty_route_payload(self, source: str) -> dict[str, Any]:
        return {
            "stamp": time.time(),
            "source": source,
            "route_source": source,
            "points": [],
            "route_len": 0,
            "route_ok": False,
            "status_ok": self._last_status is not None,
            "fallback_used": source == "fallback",
            "global_route_ok": False,
            "held_last_route": False,
        }

    def _route_is_valid(self, payload: dict[str, Any]) -> bool:
        points = payload.get("points", [])
        return bool(payload.get("route_ok")) and len(points) >= self.min_route_points

    def _remember_valid_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._last_valid_local_route = copy.deepcopy(payload)
        self._last_valid_local_route_time = time.time()
        return payload

    def _hold_last_valid_route(self, global_route_ok: bool) -> Optional[dict[str, Any]]:
        if self._last_valid_local_route is None:
            return None

        route_age_s = time.time() - self._last_valid_local_route_time
        if route_age_s > self.hold_last_route_s:
            return None

        payload = copy.deepcopy(self._last_valid_local_route)
        payload["stamp"] = time.time()
        payload["source"] = "route_sampler_hold_last"
        payload["held_last_route"] = True
        payload["held_route_age_s"] = round(route_age_s, 3)
        payload["global_route_ok"] = global_route_ok
        return payload

    def _select_local_route(self) -> dict[str, Any]:
        ego_loc = self._get_ego_location()
        if ego_loc is None:
            held_route = self._hold_last_valid_route(global_route_ok=False)
            return held_route or self._empty_route_payload("route_sampler")

        x, y, _ = ego_loc
        route = self._last_global_route or {}
        points = route.get("points", []) if isinstance(route, dict) else []
        global_route_ok = bool(
            isinstance(route, dict)
            and route.get("route_ok", False)
            and points
        )

        if self.route_source_mode != "global" or not global_route_ok:
            if self.fallback_to_simple_forward_route:
                fallback_route = self._build_forward_route()
                if self._route_is_valid(fallback_route):
                    return self._remember_valid_route(fallback_route)

            held_route = self._hold_last_valid_route(global_route_ok=False)
            if held_route is not None:
                return held_route
            return self._empty_route_payload(
                "fallback" if self.fallback_to_simple_forward_route else "route_invalid"
            )

        nearest_index = forward_window_search(points, x, y, self._last_index, window=12, reset_threshold_m=10.0)
        self._last_index = nearest_index
        local_points = build_local_route_segment(points, nearest_index, self.local_route_horizon_m)

        payload = {
            "stamp": time.time(),
            "source": "phase2c_route_sampler",
            "route_source": "global_route",
            "nearest_index": nearest_index,
            "points": local_points,
            "route_len": len(local_points),
            "route_ok": len(local_points) >= self.min_route_points,
            "status_ok": self._last_status is not None,
            "fallback_used": False,
            "global_route_ok": True,
            "held_last_route": False,
        }
        if self._route_is_valid(payload):
            return self._remember_valid_route(payload)

        if self.fallback_to_simple_forward_route:
            fallback_route = self._build_forward_route()
            if self._route_is_valid(fallback_route):
                return self._remember_valid_route(fallback_route)

        held_route = self._hold_last_valid_route(global_route_ok=True)
        return held_route or payload

    def _tick(self) -> None:
        if self._client is None or self._map is None:
            self._connect_to_carla()

        route_payload = self._select_local_route()
        self.route_pub.publish(String(data=json.dumps(route_payload)))

        debug_payload = {
            "stamp": time.time(),
            "source": route_payload.get("source"),
            "route_source": route_payload.get("route_source"),
            "route_len": route_payload.get("route_len"),
            "nearest_index": route_payload.get("nearest_index"),
            "route_ok": route_payload.get("route_ok"),
            "status_ok": route_payload.get("status_ok"),
            "fallback_used": route_payload.get("fallback_used", False),
            "global_route_ok": route_payload.get("global_route_ok", False),
            "held_last_route": route_payload.get("held_last_route", False),
            "held_route_age_s": route_payload.get("held_route_age_s"),
            "hold_last_route_s": self.hold_last_route_s,
            "fallback_to_simple_forward_route": self.fallback_to_simple_forward_route,
        }
        self.route_debug_pub.publish(String(data=json.dumps(debug_payload)))


def main(args=None):
    rclpy.init(args=args)
    node = RouteSampler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
