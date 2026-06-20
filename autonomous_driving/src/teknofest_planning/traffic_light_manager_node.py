#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import os
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla


def _package_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def _distance_xy(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


class TrafficLightManagerNode(Node):
    def __init__(self):
        super().__init__("traffic_light_manager_node")

        default_waypoints = os.path.join(_package_root(), "config", "captured_tl_waypoints.json")
        self.declare_parameter("tl_event_topic", "/adas/planning/tl_event")
        self.declare_parameter("traffic_light_topic", "/adas/perception/traffic_lights")
        self.declare_parameter("route_topic", "/adas/planning/route")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("traffic_light_max_age_s", 0.50)
        self.declare_parameter("traffic_light_min_confidence", 0.50)
        self.declare_parameter("route_corridor_width_m", 4.5)
        self.declare_parameter("front_bumper_offset_m", 2.39589)
        self.declare_parameter("captured_tl_waypoints_path", default_waypoints)
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("debug_draw_waypoints", True)
        self.declare_parameter("debug_draw_life_time_s", 0.20)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.route_payload: Optional[dict] = None
        self.status_payload: Optional[dict] = None
        self.traffic_light_payload: Optional[dict] = None
        self.last_traffic_light_s = 0.0
        self.last_ros_log_s = 0.0

        self.carla = None
        self.world = None
        self.ego_vehicle = None
        self.last_ego_lookup_s = 0.0
        self.carla_connect_attempted = False
        self.captured_waypoints = self.load_captured_waypoints()

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="traffic_light_manager_node",
            file_name="traffic_light.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        self.event_pub = self.create_publisher(
            String,
            str(self.get_parameter("tl_event_topic").value),
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("traffic_light_topic").value),
            self.traffic_light_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("route_topic").value),
            self.route_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self.status_cb,
            10,
        )
        self.create_timer(float(self.get_parameter("publish_period_s").value), self.tick)
        self.create_timer(0.10, self.draw_waypoint_marker)
        self.get_logger().info("Traffic light manager node ready.")

    def load_captured_waypoints(self) -> list[dict]:
        path = str(self.get_parameter("captured_tl_waypoints_path").value)
        try:
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            self.get_logger().warning(f"Captured TL waypoint file unavailable: {exc}")
            return []

        if isinstance(payload, dict) and isinstance(payload.get("waypoints"), list):
            payload = payload["waypoints"]

        out = []
        items = []
        if isinstance(payload, list):
            for index, item in enumerate(payload):
                items.append((f"waypoint_{index}", item))
        elif isinstance(payload, dict):
            for key, value in payload.items():
                items.append((str(key), value))

        for fallback_id, item in items:
            if not isinstance(item, dict):
                continue
            try:
                out.append({
                    "id": str(item.get("id") or item.get("waypoint_id") or fallback_id),
                    "x": float(item["x"]),
                    "y": float(item["y"]),
                    "z": float(item.get("z", 0.0) or 0.0),
                    "yaw_deg": float(item.get("yaw_deg", item.get("yaw", 0.0)) or 0.0),
                    "road_id": item.get("road_id"),
                    "lane_id": item.get("lane_id"),
                    "is_junction": item.get("is_junction"),
                })
            except (KeyError, TypeError, ValueError):
                continue

        self.get_logger().info(f"Loaded {len(out)} captured TL waypoints from {path}")
        if not out:
            self.get_logger().warning(
                "No captured TL waypoints loaded; traffic light manager will not emit route-relevant TL events"
            )
        return out

    def parse_json(self, message: String) -> Optional[dict]:
        try:
            return json.loads(message.data)
        except (json.JSONDecodeError, TypeError):
            return None

    def traffic_light_cb(self, message: String):
        payload = self.parse_json(message)
        if payload is not None:
            self.traffic_light_payload = payload
            self.last_traffic_light_s = time.time()

    def route_cb(self, message: String):
        payload = self.parse_json(message)
        if payload is not None:
            self.route_payload = payload

    def status_cb(self, message: String):
        payload = self.parse_json(message)
        if payload is not None:
            self.status_payload = payload

    def ensure_carla_world(self):
        if self.world is not None or self.carla_connect_attempted:
            return self.world
        self.carla_connect_attempted = True
        try:
            self.carla = load_carla(str(self.get_parameter("carla_root").value))
            client = self.carla.Client(
                str(self.get_parameter("host").value),
                int(self.get_parameter("port").value),
            )
            client.set_timeout(2.0)
            self.world = client.get_world()
        except Exception as exc:
            self.get_logger().warning(f"Traffic light CARLA unavailable: {exc}")
            self.world = None
        return self.world

    def find_ego_vehicle(self):
        ego_vehicle = self.ego_vehicle
        if ego_vehicle is not None:
            try:
                if ego_vehicle.is_alive:
                    return ego_vehicle
            except Exception:
                self.ego_vehicle = None

        world = self.ensure_carla_world()
        if world is None:
            return None

        now = time.time()
        if now - self.last_ego_lookup_s < 0.5:
            return self.ego_vehicle
        self.last_ego_lookup_s = now

        status_ego_id = (self.status_payload or {}).get("ego_id")
        if status_ego_id is not None:
            try:
                actor = world.get_actor(int(status_ego_id))
                if actor is not None:
                    self.ego_vehicle = actor
                    return actor
            except Exception:
                pass

        role_name = str(self.get_parameter("ego_role_name").value)
        try:
            for vehicle in world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == role_name:
                    self.ego_vehicle = vehicle
                    return vehicle
        except Exception:
            return None
        return None

    def front_bumper_point(self) -> Optional[dict]:
        status = self.status_payload or {}
        location = status.get("location") or {}
        rotation = status.get("rotation") or {}
        try:
            x = float(location["x"])
            y = float(location["y"])
            yaw = math.radians(float(rotation.get("yaw", 0.0)))
        except (KeyError, TypeError, ValueError):
            return None
        offset = max(0.0, float(self.get_parameter("front_bumper_offset_m").value))
        return {
            "x": x + math.cos(yaw) * offset,
            "y": y + math.sin(yaw) * offset,
            "z": float(location.get("z", 0.0) or 0.0),
            "yaw_deg": float(rotation.get("yaw", 0.0) or 0.0),
        }

    def detector_traffic_light_color(self) -> tuple[str, str]:
        payload = self.traffic_light_payload or {}
        max_age = max(0.0, float(self.get_parameter("traffic_light_max_age_s").value))
        if self.last_traffic_light_s <= 0.0 or time.time() - self.last_traffic_light_s > max_age:
            return "unknown", "traffic_light_detector_stale"
        minimum_confidence = float(self.get_parameter("traffic_light_min_confidence").value)
        for key in ("selected_detection", "primary_detection"):
            item = payload.get(key)
            if not isinstance(item, dict):
                continue
            color = str(item.get("tl_color_filtered") or item.get("tl_color_raw") or item.get("color") or "unknown").lower()
            confidence = float(item.get("tl_confidence", 1.0) or 0.0)
            if color in {"red", "yellow", "green"} and confidence >= minimum_confidence:
                return color, f"detector_{key}"
        return "unknown", "traffic_light_detector_no_reliable_color"

    @staticmethod
    def normalize_traffic_light_state(state) -> str:
        if state is None:
            return "unknown"
        text = str(getattr(state, "name", state)).strip().lower().rsplit(".", 1)[-1]
        if text in {"red", "yellow", "green"}:
            return text
        return "unknown"

    def carla_traffic_light_color(self) -> tuple[str, str]:
        ego_vehicle = self.find_ego_vehicle()
        if ego_vehicle is None:
            return "unknown", "carla_ego_vehicle_unavailable"
        try:
            color = self.normalize_traffic_light_state(ego_vehicle.get_traffic_light_state())
            if color != "unknown":
                return color, "carla_ego_traffic_light_state"
        except Exception:
            pass
        try:
            traffic_light = ego_vehicle.get_traffic_light()
            if traffic_light is not None:
                color = self.normalize_traffic_light_state(traffic_light.get_state())
                if color != "unknown":
                    return color, "carla_ego_traffic_light_actor_state"
        except Exception:
            pass
        return "unknown", "carla_ego_traffic_light_unknown"

    def route_progress(self, point: dict) -> tuple[int, float]:
        route_points = list((self.route_payload or {}).get("points", []))
        nearest_index = 0
        nearest_distance = float("inf")
        for index, route_point in enumerate(route_points):
            try:
                distance = _distance_xy(route_point, point)
            except (KeyError, TypeError, ValueError):
                continue
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        return nearest_index, nearest_distance

    def route_distance(self, start_index: int, end_index: int) -> float:
        route_points = list((self.route_payload or {}).get("points", []))
        if not route_points:
            return 0.0
        start = max(0, min(start_index, len(route_points) - 1))
        end = max(start, min(end_index, len(route_points) - 1))
        total = 0.0
        for index in range(start, end):
            total += _distance_xy(route_points[index], route_points[index + 1])
        return total

    def relevant_waypoint(self, front_point: dict | None) -> tuple[Optional[dict], Optional[float], Optional[float]]:
        route_points = list((self.route_payload or {}).get("points", []))
        if not route_points or front_point is None or not self.captured_waypoints:
            return None, None, None
        ego_index, _ = self.route_progress(front_point)
        corridor_limit = float(self.get_parameter("route_corridor_width_m").value)

        best = None
        for waypoint in self.captured_waypoints:
            waypoint_index, lateral_distance = self.route_progress(waypoint)
            if lateral_distance > corridor_limit:
                continue
            if waypoint_index < ego_index:
                continue
            route_distance = self.route_distance(ego_index, waypoint_index)
            yaw = math.radians(float(front_point.get("yaw_deg", 0.0) or 0.0))
            dx = float(waypoint["x"]) - float(front_point["x"])
            dy = float(waypoint["y"]) - float(front_point["y"])
            forward_projection = math.cos(yaw) * dx + math.sin(yaw) * dy
            if forward_projection < -1.0:
                continue
            candidate = (route_distance, lateral_distance, waypoint)
            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is None:
            return None, None, None
        return best[2], best[0], best[1]

    def final_traffic_light_color(self) -> tuple[str, str]:
        carla_color, carla_source = self.carla_traffic_light_color()
        if carla_color in {"red", "yellow", "green"}:
            return carla_color, carla_source
        detector_color, detector_source = self.detector_traffic_light_color()
        if detector_color in {"red", "yellow", "green"}:
            return detector_color, detector_source
        return "unknown", "no_reliable_traffic_light_color"

    def draw_waypoint_marker(self):
        if not bool(self.get_parameter("debug_draw_waypoints").value):
            return
        world = self.ensure_carla_world()
        if world is None or self.carla is None:
            return
        front_point = self.front_bumper_point()
        waypoint, _, _ = self.relevant_waypoint(front_point)
        if waypoint is None:
            return
        life_time = float(self.get_parameter("debug_draw_life_time_s").value)
        location = self.carla.Location(
            x=float(waypoint["x"]),
            y=float(waypoint["y"]),
            z=float(waypoint.get("z", 0.0)),
        )
        world.debug.draw_point(
            location,
            size=0.18,
            color=self.carla.Color(255, 0, 0),
            life_time=life_time,
        )
        world.debug.draw_string(
            location + self.carla.Location(z=0.35),
            str(waypoint["id"]),
            draw_shadow=True,
            color=self.carla.Color(255, 255, 255),
            life_time=life_time,
        )

    def tick(self):
        front_point = self.front_bumper_point()
        waypoint, route_distance_m, lateral_distance_m = self.relevant_waypoint(front_point)
        color, color_source = self.final_traffic_light_color()
        has_relevant_light = waypoint is not None and color in {"red", "yellow", "green"}

        output = {
            "stamp": time.time(),
            "has_relevant_light": bool(has_relevant_light),
            "color": color if has_relevant_light else "unknown",
            "distance_m": round(float(route_distance_m), 3) if route_distance_m is not None else None,
            "stop_point": {
                "x": round(float(waypoint["x"]), 4),
                "y": round(float(waypoint["y"]), 4),
                "z": round(float(waypoint.get("z", 0.0)), 4),
                "yaw_deg": round(float(waypoint.get("yaw_deg", 0.0)), 4),
                "road_id": waypoint.get("road_id"),
                "lane_id": waypoint.get("lane_id"),
            } if waypoint is not None else None,
            "source": "captured_tl_waypoint" if waypoint is not None else color_source,
            "tl_id": waypoint.get("id") if waypoint is not None else None,
            "reason": (
                f"route_relevant_{color}"
                if has_relevant_light
                else "no_route_relevant_traffic_light"
            ),
            "route_corridor_lateral_m": round(float(lateral_distance_m), 3) if lateral_distance_m is not None else None,
            "color_source": color_source,
        }

        msg = String()
        msg.data = json.dumps(output, ensure_ascii=False)
        self.event_pub.publish(msg)
        self.runtime_logger.write(output)

        period = float(self.get_parameter("ros_log_period_s").value)
        now = time.time()
        if now - self.last_ros_log_s >= period:
            self.last_ros_log_s = now
            self.get_logger().info(
                "traffic_light_manager "
                f"relevant={output['has_relevant_light']} "
                f"color={output['color']} "
                f"dist={output['distance_m']} "
                f"tl_id={output['tl_id']} "
                f"reason={output['reason']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightManagerNode()
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
