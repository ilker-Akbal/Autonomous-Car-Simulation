import glob
import json
import os
import sys
import time
from typing import Any, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla
from teknofest_planning.route_geometry import distance_2d, nearest_point_distance


class GlobalRoutePlannerNode(Node):
    def __init__(self):
        super().__init__("global_route_planner")

        self.declare_parameter(
            "carla_root",
            "/home/ilker/simulators/CARLA_0.9.15",
        )
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("global_sampling_resolution_m", 2.0)
        self.declare_parameter("replan_period_s", 3.0)
        self.declare_parameter("replan_distance_threshold_m", 8.0)
        self.declare_parameter("goal_change_replan", True)
        self.declare_parameter("route_timeout_s", 10.0)
        self.declare_parameter("min_route_points", 8)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.global_sampling_resolution_m = float(self.get_parameter("global_sampling_resolution_m").value)
        self.replan_period_s = float(self.get_parameter("replan_period_s").value)
        self.replan_distance_threshold_m = float(self.get_parameter("replan_distance_threshold_m").value)
        self.goal_change_replan = bool(self.get_parameter("goal_change_replan").value)
        self.route_timeout_s = float(self.get_parameter("route_timeout_s").value)
        self.min_route_points = int(self.get_parameter("min_route_points").value)

        self._last_status: Optional[dict[str, Any]] = None
        self._last_goal: Optional[dict[str, Any]] = None
        self._last_goal_time = 0.0
        self._last_route_time = 0.0
        self._route_points: List[dict[str, Any]] = []

        self._client = None
        self._world = None
        self._map = None
        self._route_planner = None
        self._planner_api = "failed"
        self._replan_reason = "planner_not_initialized"
        self._carla_root_used: Optional[str] = None
        self._agents_import_path: Optional[str] = None
        self._last_warning_times: dict[str, float] = {}
        self._carla = None

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_subscription(String, "/adas/mission/current_goal", self._goal_callback, 10)

        self.route_pub = self.create_publisher(String, "/adas/planning/global_route", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/global_route_debug", 10)

        self._connect_to_carla()
        self.timer = self.create_timer(max(0.1, self.replan_period_s), self._tick)

    def _candidate_carla_roots(self) -> list[str]:
        roots = [
            str(self.carla_root),
            "/home/ilker/simulators/CARLA_0.9.15",
            "/mnt/carla/CARLA_0.9.15",
        ]
        unique_roots = []
        for root in roots:
            normalized = os.path.abspath(os.path.expanduser(root))
            if normalized not in unique_roots:
                unique_roots.append(normalized)
        return unique_roots

    def _prepare_carla_python_paths(self) -> None:
        self._agents_import_path = None
        for root in self._candidate_carla_roots():
            python_api = os.path.join(root, "PythonAPI", "carla")
            if os.path.isdir(os.path.join(python_api, "agents")):
                if python_api not in sys.path:
                    sys.path.insert(0, python_api)
                if self._agents_import_path is None:
                    self._agents_import_path = python_api
                    self._carla_root_used = root

            egg_pattern = os.path.join(python_api, "dist", "*.egg")
            for egg_path in sorted(glob.glob(egg_pattern)):
                if egg_path not in sys.path:
                    sys.path.append(egg_path)

    def _load_carla_with_fallback(self):
        self._prepare_carla_python_paths()
        errors = []
        for root in self._candidate_carla_roots():
            try:
                carla = load_carla(root)
                if self._carla_root_used is None:
                    self._carla_root_used = root
                return carla
            except Exception as exc:
                errors.append(f"{root}: {exc}")
        raise RuntimeError("; ".join(errors))

    def _connect_to_carla(self) -> None:
        try:
            self._carla = self._load_carla_with_fallback()
            self._client = self._carla.Client(self.host, self.port)
            self._client.set_timeout(10.0)
            self._world = self._client.get_world()
            self._map = self._world.get_map()
        except Exception as exc:
            self.get_logger().warn(f"GlobalRoutePlanner: CARLA connection failed: {exc}")
            self._client = None
            self._world = None
            self._map = None

    def _warn_throttled(self, key: str, message: str, period_s: float = 10.0) -> None:
        now = time.monotonic()
        if now - self._last_warning_times.get(key, 0.0) >= period_s:
            self.get_logger().warn(message)
            self._last_warning_times[key] = now

    def _ensure_route_planner(self) -> bool:
        if self._route_planner is not None:
            return True
        if self._map is None:
            self._planner_api = "failed"
            self._replan_reason = "carla_map_unavailable"
            return False

        self._prepare_carla_python_paths()
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except Exception as exc:
            self._planner_api = "failed"
            self._replan_reason = f"planner_import_failed: {exc}"
            self._warn_throttled(
                "planner_import",
                f"GlobalRoutePlanner: import failed: {exc}",
            )
            return False

        try:
            self._route_planner = GlobalRoutePlanner(
                self._map,
                self.global_sampling_resolution_m,
            )
            self._planner_api = "direct_map"
            self._replan_reason = "planner_ready"
            return True
        except Exception as direct_exc:
            try:
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(
                    self._map,
                    self.global_sampling_resolution_m,
                )
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                self._route_planner = planner
                self._planner_api = "dao"
                self._replan_reason = "planner_ready"
                return True
            except Exception as dao_exc:
                self._route_planner = None
                self._planner_api = "failed"
                self._replan_reason = (
                    f"planner_setup_failed: direct_map={direct_exc}; dao={dao_exc}"
                )
                self._warn_throttled(
                    "planner_setup",
                    "GlobalRoutePlanner: setup failed for direct-map and DAO APIs: "
                    f"direct-map={direct_exc}; DAO={dao_exc}",
                )
                return False

    def _clear_route(self, reason: str) -> None:
        self._route_points = []
        self._last_route_time = 0.0
        self._replan_reason = reason

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("GlobalRoutePlanner: failed to parse /adas/carla/status JSON")

    def _goal_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self._last_goal = payload.get("current_goal")
            self._last_goal_time = time.time()
        except Exception:
            self.get_logger().warn("GlobalRoutePlanner: failed to parse /adas/mission/current_goal JSON")

    def _is_route_stale(self) -> bool:
        return (time.time() - self._last_route_time) > self.route_timeout_s

    def _route_has_enough_points(self) -> bool:
        return len(self._route_points) >= self.min_route_points

    def _distance_to_route(self, x: float, y: float) -> float:
        if not self._route_points:
            return float("inf")
        _, distance = nearest_point_distance(self._route_points, x, y)
        return distance

    def _get_ego_waypoint(self) -> Optional[Any]:
        if self._map is None or self._carla is None:
            return None

        location = None
        if self._last_status is not None:
            loc = self._last_status.get("location", {})
            if "x" in loc and "y" in loc:
                location = self._carla.Location(x=float(loc["x"]), y=float(loc["y"]), z=float(loc.get("z", 0.0)))

        if location is None:
            return None

        try:
            waypoint = self._map.get_waypoint(location, project_to_road=True, lane_type=self._carla.LaneType.Driving)
            return waypoint
        except Exception as exc:
            self.get_logger().warn(f"GlobalRoutePlanner: waypoint lookup failed: {exc}")
            return None

    def _get_goal_waypoint(self) -> Optional[Any]:
        if self._map is None or self._carla is None or self._last_goal is None:
            return None

        goal_x = self._last_goal.get("carla_x")
        goal_y = self._last_goal.get("carla_y")
        goal_z = self._last_goal.get("carla_z")
        if goal_x is None or goal_y is None:
            return None

        try:
            location = self._carla.Location(x=float(goal_x), y=float(goal_y), z=float(goal_z or 0.0))
            return self._map.get_waypoint(location, project_to_road=True, lane_type=self._carla.LaneType.Driving)
        except Exception as exc:
            self.get_logger().warn(f"GlobalRoutePlanner: goal waypoint lookup failed: {exc}")
            return None

    def _need_replan(self) -> bool:
        if self._last_goal is None or self._last_status is None:
            return False

        if not self._route_has_enough_points():
            return True

        if self._is_route_stale():
            return True

        if self.goal_change_replan:
            current_goal_name = self._last_goal.get("name")
            previous_goal_name = None
            if self._route_points:
                previous_goal_name = self._route_points[-1].get("goal_name")
            if current_goal_name != previous_goal_name:
                return True

        ego_loc = self._last_status.get("location", {})
        x = float(ego_loc.get("x", 0.0))
        y = float(ego_loc.get("y", 0.0))
        distance_to_route = self._distance_to_route(x, y)
        if distance_to_route > self.replan_distance_threshold_m:
            return True

        return False

    def _plan_route(self) -> None:
        if not self._ensure_route_planner():
            self._clear_route(self._replan_reason)
            return

        start_wp = self._get_ego_waypoint()
        goal_wp = self._get_goal_waypoint()
        if start_wp is None:
            self._clear_route("start_waypoint_unavailable")
            return
        if goal_wp is None:
            self._clear_route("goal_waypoint_unavailable")
            return

        try:
            start_location = start_wp.transform.location
            goal_location = goal_wp.transform.location
            route = self._route_planner.trace_route(start_location, goal_location)
        except Exception as exc:
            self._clear_route(f"trace_route_failed: {exc}")
            self._warn_throttled(
                "trace_route",
                f"GlobalRoutePlanner: trace_route failed: {exc}",
            )
            return

        if not route:
            self._clear_route("trace_route_returned_empty")
            return

        points: List[dict[str, Any]] = []
        s = 0.0
        previous = None
        for waypoint, _ in route:
            if previous is not None:
                s += distance_2d(
                    float(previous.transform.location.x),
                    float(previous.transform.location.y),
                    float(waypoint.transform.location.x),
                    float(waypoint.transform.location.y),
                )
            previous = waypoint
            points.append(
                {
                    "x": round(float(waypoint.transform.location.x), 3),
                    "y": round(float(waypoint.transform.location.y), 3),
                    "z": round(float(waypoint.transform.location.z), 3),
                    "yaw": round(float(waypoint.transform.rotation.yaw), 3),
                    "road_id": waypoint.road_id,
                    "lane_id": waypoint.lane_id,
                    "s": round(s, 3),
                    "goal_name": self._last_goal.get("name"),
                    "goal_index": self._last_goal.get("nokta_id"),
                }
            )

        if points:
            self._route_points = points
            self._last_route_time = time.time()
            self._replan_reason = "route_ready"
        else:
            self._clear_route("trace_route_had_no_waypoints")

    def _publish_route(self) -> None:
        now = time.time()
        route_ok = len(self._route_points) >= self.min_route_points

        payload = {
            "stamp": now,
            "source": "phase2c_global_route_planner",
            "goal_name": self._last_goal.get("name") if self._last_goal else None,
            "goal_index": self._last_goal.get("nokta_id") if self._last_goal else None,
            "route_len": len(self._route_points),
            "route_source": "global",
            "points": self._route_points,
            "route_ok": route_ok,
            "status_ok": self._last_status is not None,
            "planner_api": self._planner_api,
            "carla_root_used": self._carla_root_used,
            "agents_import_path": self._agents_import_path,
            "replan_reason": None if route_ok else self._replan_reason,
        }

        self.route_pub.publish(String(data=json.dumps(payload)))

        debug_payload = {
            "stamp": now,
            "route_ok": route_ok,
            "status_ok": self._last_status is not None,
            "goal_name": self._last_goal.get("name") if self._last_goal else None,
            "goal_index": self._last_goal.get("nokta_id") if self._last_goal else None,
            "route_len": len(self._route_points),
            "planner_api": self._planner_api,
            "carla_root_used": self._carla_root_used,
            "agents_import_path": self._agents_import_path,
            "replan_reason": None if route_ok else self._replan_reason,
            "distance_to_route_m": self._distance_to_route(
                float(self._last_status.get("location", {}).get("x", 0.0)),
                float(self._last_status.get("location", {}).get("y", 0.0)),
            ) if self._last_status is not None else None,
            "route_age_s": time.time() - self._last_route_time if self._last_route_time else None,
        }
        self.debug_pub.publish(String(data=json.dumps(debug_payload)))

    def _tick(self) -> None:
        if self._client is None or self._map is None:
            self._connect_to_carla()

        if self._need_replan():
            self._plan_route()

        self._publish_route()


def main(args=None):
    rclpy.init(args=args)
    node = GlobalRoutePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
