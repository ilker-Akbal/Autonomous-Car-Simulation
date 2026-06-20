import json
import math
import time
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_planning.route_geometry import (
    cumulative_route_s,
    is_ahead_on_route,
    project_actor_to_route,
)
from teknofest_sim.carla_loader import load_carla


class RouteEventAnalyzer(Node):
    def __init__(self):
        super().__init__("route_event_analyzer")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("route_lateral_margin_m", 3.0)
        self.declare_parameter("event_horizon_m", 45.0)
        self.declare_parameter("vehicle_follow_distance_m", 10.0)
        self.declare_parameter("vehicle_stop_distance_m", 6.0)
        self.declare_parameter("pedestrian_stop_distance_m", 8.0)
        self.declare_parameter("min_event_speed_mps", 0.0)
        self.declare_parameter("follow_time_gap_s", 1.5)
        self.declare_parameter("stale_route_timeout_s", 1.0)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.route_lateral_margin_m = float(self.get_parameter("route_lateral_margin_m").value)
        self.event_horizon_m = float(self.get_parameter("event_horizon_m").value)
        self.vehicle_follow_distance_m = float(self.get_parameter("vehicle_follow_distance_m").value)
        self.vehicle_stop_distance_m = float(self.get_parameter("vehicle_stop_distance_m").value)
        self.pedestrian_stop_distance_m = float(self.get_parameter("pedestrian_stop_distance_m").value)
        self.min_event_speed_mps = float(self.get_parameter("min_event_speed_mps").value)
        self.follow_time_gap_s = float(self.get_parameter("follow_time_gap_s").value)
        self.stale_route_timeout_s = float(self.get_parameter("stale_route_timeout_s").value)

        self._last_status: Optional[dict[str, Any]] = None
        self._last_route: Optional[dict[str, Any]] = None
        self._last_status_time = 0.0
        self._last_route_time = 0.0
        self._last_warning_times: dict[str, float] = {}

        self._carla = None
        self._client = None
        self._world = None
        self._ego_vehicle = None

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_subscription(String, "/adas/planning/route", self._route_callback, 10)
        self.event_pub = self.create_publisher(String, "/adas/planning/route_events", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/route_events_debug", 10)

        self._connect_to_carla()
        self.timer = self.create_timer(
            1.0 / max(0.1, self.publish_rate_hz),
            self._tick,
        )

    def _warn_throttled(self, key: str, message: str, period_s: float = 10.0) -> None:
        now = time.monotonic()
        if now - self._last_warning_times.get(key, 0.0) >= period_s:
            self.get_logger().warn(message)
            self._last_warning_times[key] = now

    def _connect_to_carla(self) -> None:
        try:
            self._carla = load_carla(self.carla_root)
            self._client = self._carla.Client(self.host, self.port)
            self._client.set_timeout(5.0)
            self._world = self._client.get_world()
            self._ego_vehicle = None
        except Exception as exc:
            self._client = None
            self._world = None
            self._ego_vehicle = None
            self._warn_throttled(
                "carla_connection",
                f"RouteEventAnalyzer: CARLA connection failed: {exc}",
            )

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
            self._last_status_time = time.time()
        except Exception:
            self._warn_throttled(
                "status_json",
                "RouteEventAnalyzer: failed to parse /adas/carla/status JSON",
            )

    def _route_callback(self, msg: String) -> None:
        try:
            self._last_route = json.loads(msg.data)
            self._last_route_time = time.time()
        except Exception:
            self._warn_throttled(
                "route_json",
                "RouteEventAnalyzer: failed to parse /adas/planning/route JSON",
            )

    def _find_ego_vehicle(self):
        if self._world is None:
            return None

        status_ego_id = None
        if self._last_status is not None:
            status_ego_id = self._last_status.get("ego_id")

        if self._ego_vehicle is not None and self._ego_vehicle.is_alive:
            if status_ego_id is None or self._ego_vehicle.id == int(status_ego_id):
                return self._ego_vehicle

        for vehicle in self._world.get_actors().filter("vehicle.*"):
            if status_ego_id is not None and vehicle.id == int(status_ego_id):
                self._ego_vehicle = vehicle
                return vehicle
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self._ego_vehicle = vehicle
                return vehicle
        return None

    def _base_event(self, ok: bool, reason: str) -> dict[str, Any]:
        return {
            "stamp": time.time(),
            "ok": ok,
            "event": "clear",
            "target_speed_limit_mps": None,
            "stop_required": False,
            "distance_m": None,
            "actor_id": None,
            "actor_type": None,
            "route_index": None,
            "route_lateral_distance_m": None,
            "reason": reason,
        }

    def _candidate(
        self,
        event: str,
        actor,
        projection: dict[str, float | int],
        distance_ahead_m: float,
        target_speed_limit_mps: float,
        reason: str,
    ) -> dict[str, Any]:
        stop_required = event in ("vehicle_stop", "pedestrian_stop")
        return {
            "event": event,
            "target_speed_limit_mps": (
                0.0
                if stop_required
                else max(self.min_event_speed_mps, target_speed_limit_mps)
            ),
            "stop_required": stop_required,
            "distance_m": round(distance_ahead_m, 3),
            "actor_id": int(actor.id),
            "actor_type": str(actor.type_id),
            "route_index": int(projection["route_index"]),
            "route_lateral_distance_m": round(
                float(projection["lateral_distance_m"]),
                3,
            ),
            "reason": reason,
        }

    def _analyze(self) -> tuple[dict[str, Any], dict[str, Any]]:
        now = time.time()
        route_age_s = now - self._last_route_time if self._last_route_time else None
        status_age_s = now - self._last_status_time if self._last_status_time else None
        debug = {
            "stamp": now,
            "route_age_s": route_age_s,
            "status_age_s": status_age_s,
            "actors_checked": 0,
            "actors_in_route_corridor": 0,
            "candidates": 0,
            "ego_found": False,
        }

        if self._world is None:
            return self._base_event(False, "carla_unavailable"), debug
        if self._last_status is None or status_age_s is None or status_age_s > 1.0:
            return self._base_event(False, "status_missing_or_stale"), debug
        if self._last_route is None or route_age_s is None or route_age_s > self.stale_route_timeout_s:
            return self._base_event(False, "route_missing_or_stale"), debug

        points = self._last_route.get("points", [])
        if not self._last_route.get("route_ok", False) or len(points) < 2:
            return self._base_event(False, "route_invalid"), debug

        ego_vehicle = self._find_ego_vehicle()
        debug["ego_found"] = ego_vehicle is not None
        if ego_vehicle is None:
            return self._base_event(False, "ego_vehicle_not_found"), debug

        ego_location = self._last_status.get("location", {})
        route_s = cumulative_route_s(points)
        ego_projection = project_actor_to_route(
            points,
            float(ego_location.get("x", 0.0)),
            float(ego_location.get("y", 0.0)),
            route_s,
        )
        if ego_projection is None:
            return self._base_event(False, "ego_route_projection_failed"), debug

        candidates: list[dict[str, Any]] = []
        world_actors = self._world.get_actors()
        actors = list(world_actors.filter("vehicle.*"))
        actors.extend(world_actors.filter("walker.pedestrian.*"))

        for actor in actors:
            if actor.id == ego_vehicle.id:
                continue
            debug["actors_checked"] += 1

            try:
                location = actor.get_location()
                projection = project_actor_to_route(
                    points,
                    float(location.x),
                    float(location.y),
                    route_s,
                )
            except Exception:
                continue

            if projection is None:
                continue
            lateral_distance = float(projection["lateral_distance_m"])
            if lateral_distance > self.route_lateral_margin_m:
                continue

            distance_ahead = (
                float(projection["route_s_m"])
                - float(ego_projection["route_s_m"])
            )
            if not is_ahead_on_route(
                float(ego_projection["route_s_m"]),
                float(projection["route_s_m"]),
                tolerance_m=0.25,
            ):
                continue
            distance_ahead = max(0.0, distance_ahead)
            if distance_ahead > self.event_horizon_m:
                continue

            debug["actors_in_route_corridor"] += 1
            if actor.type_id.startswith("walker.pedestrian."):
                if distance_ahead <= self.pedestrian_stop_distance_m:
                    candidates.append(
                        self._candidate(
                            "pedestrian_stop",
                            actor,
                            projection,
                            distance_ahead,
                            0.0,
                            "pedestrian_ahead_in_route_corridor",
                        )
                    )
                continue

            if not actor.type_id.startswith("vehicle."):
                continue
            if distance_ahead <= self.vehicle_stop_distance_m:
                candidates.append(
                    self._candidate(
                        "vehicle_stop",
                        actor,
                        projection,
                        distance_ahead,
                        0.0,
                        "vehicle_inside_stop_distance",
                    )
                )
                continue

            try:
                velocity = actor.get_velocity()
            except Exception:
                continue
            front_speed_mps = math.sqrt(
                velocity.x * velocity.x
                + velocity.y * velocity.y
                + velocity.z * velocity.z
            )
            available_gap_m = max(
                0.0,
                distance_ahead - self.vehicle_follow_distance_m,
            )
            safe_follow_speed_mps = (
                front_speed_mps
                + available_gap_m / max(0.1, self.follow_time_gap_s)
            )
            candidates.append(
                self._candidate(
                    "vehicle_follow",
                    actor,
                    projection,
                    distance_ahead,
                    safe_follow_speed_mps,
                    "lead_vehicle_ahead_in_route_corridor",
                )
            )

        debug["candidates"] = len(candidates)
        if not candidates:
            return self._base_event(True, "route_corridor_clear"), debug

        priority = {
            "pedestrian_stop": 0,
            "vehicle_stop": 1,
            "vehicle_follow": 2,
        }
        selected = min(
            candidates,
            key=lambda item: (
                priority.get(item["event"], 99),
                float(item["distance_m"]),
            ),
        )
        payload = self._base_event(True, selected["reason"])
        payload.update(selected)
        return payload, debug

    def _tick(self) -> None:
        if self._world is None:
            self._connect_to_carla()

        try:
            payload, debug = self._analyze()
        except Exception as exc:
            payload = self._base_event(False, f"analysis_failed: {exc}")
            debug = {
                "stamp": time.time(),
                "analysis_error": str(exc),
            }
            self._warn_throttled(
                "analysis",
                f"RouteEventAnalyzer: analysis failed: {exc}",
            )

        debug.update({
            "ok": payload["ok"],
            "event": payload["event"],
            "distance_m": payload["distance_m"],
            "actor_id": payload["actor_id"],
            "reason": payload["reason"],
        })
        self.event_pub.publish(String(data=json.dumps(payload)))
        self.debug_pub.publish(String(data=json.dumps(debug)))


def main(args=None):
    rclpy.init(args=args)
    node = RouteEventAnalyzer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
