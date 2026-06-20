#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla


def _distance_xy(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _normalize_angle_deg(angle: float) -> float:
    while angle > 180.0:
        angle -= 360.0
    while angle < -180.0:
        angle += 360.0
    return angle


class TrafficLightManagerNode(Node):
    def __init__(self):
        super().__init__("traffic_light_manager_node")

        self.declare_parameter("tl_event_topic", "/adas/planning/tl_event")
        self.declare_parameter("traffic_light_topic", "/adas/perception/traffic_lights")
        self.declare_parameter("route_topic", "/adas/planning/route")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("traffic_light_max_age_s", 0.50)
        self.declare_parameter("traffic_light_min_confidence", 0.50)
        self.declare_parameter("route_corridor_width_m", 4.5)
        self.declare_parameter("traffic_light_max_distance_m", 45.0)
        self.declare_parameter("traffic_light_lateral_limit_m", 5.0)
        self.declare_parameter("stop_before_line_m", 1.0)
        self.declare_parameter("green_release_confirm_s", 0.25)
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
        self.map = None
        self.ego_vehicle = None
        self.last_ego_lookup_s = 0.0
        self.carla_connect_attempted = False

        self.held_tl_id: Optional[int] = None
        self.held_stop_point: Optional[dict] = None
        self.green_seen_since_s: Optional[float] = None

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
        self.create_timer(0.10, self.draw_selected_marker)
        self.get_logger().info("Traffic light manager node ready.")

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
            self.map = self.world.get_map()
        except Exception as exc:
            self.get_logger().warning(f"Traffic light CARLA unavailable: {exc}")
            self.world = None
            self.map = None
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

    def local_xy(self, ego_transform, location) -> tuple[float, float]:
        yaw = math.radians(float(ego_transform.rotation.yaw))
        dx = float(location.x) - float(ego_transform.location.x)
        dy = float(location.y) - float(ego_transform.location.y)
        forward = math.cos(yaw) * dx + math.sin(yaw) * dy
        lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
        return forward, lateral

    def stopline_waypoints_for_light(self, light) -> list:
        try:
            waypoints = light.get_stop_waypoints()
            if waypoints:
                return list(waypoints)
        except Exception:
            pass

        if self.map is None:
            return []
        try:
            transform = light.get_transform()
            waypoint = self.map.get_waypoint(
                transform.location,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
            return [waypoint] if waypoint is not None else []
        except Exception:
            return []

    def _route_progress(self, point: dict) -> tuple[int, float]:
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

    def _route_distance(self, start_index: int, end_index: int) -> float:
        route_points = list((self.route_payload or {}).get("points", []))
        if not route_points:
            return 0.0
        start = max(0, min(start_index, len(route_points) - 1))
        end = max(start, min(end_index, len(route_points) - 1))
        total = 0.0
        for index in range(start, end):
            total += _distance_xy(route_points[index], route_points[index + 1])
        return total

    def _stop_point_from_waypoint(self, waypoint) -> dict:
        transform = waypoint.transform
        return {
            "x": round(float(transform.location.x), 4),
            "y": round(float(transform.location.y), 4),
            "z": round(float(transform.location.z), 4),
            "yaw_deg": round(float(transform.rotation.yaw), 4),
            "road_id": int(waypoint.road_id),
            "lane_id": int(waypoint.lane_id),
        }

    def relevant_stopline_ahead(self, ego_vehicle) -> Optional[dict]:
        world = self.ensure_carla_world()
        if world is None or self.map is None or ego_vehicle is None:
            return None

        ego_transform = ego_vehicle.get_transform()
        current_wp = self.map.get_waypoint(
            ego_transform.location,
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )
        route_points = list((self.route_payload or {}).get("points", []))
        corridor_limit = float(self.get_parameter("route_corridor_width_m").value)
        max_distance = float(self.get_parameter("traffic_light_max_distance_m").value)
        lateral_limit = float(self.get_parameter("traffic_light_lateral_limit_m").value)
        stop_before_line_m = float(self.get_parameter("stop_before_line_m").value)

        ego_route_index = None
        if route_points:
            ego_route_index, _ = self._route_progress({
                "x": float(ego_transform.location.x),
                "y": float(ego_transform.location.y),
            })

        best = None
        for light in world.get_actors().filter("traffic.traffic_light*"):
            for waypoint in self.stopline_waypoints_for_light(light):
                if waypoint is None:
                    continue
                forward, lateral = self.local_xy(ego_transform, waypoint.transform.location)
                stopline_distance = forward - stop_before_line_m
                if stopline_distance < -2.0 or forward > max_distance:
                    continue
                if abs(lateral) > lateral_limit:
                    continue
                if current_wp is not None:
                    yaw_diff = abs(
                        _normalize_angle_deg(
                            float(current_wp.transform.rotation.yaw) - float(waypoint.transform.rotation.yaw)
                        )
                    )
                    same_lane = (
                        int(current_wp.road_id) == int(waypoint.road_id)
                        and int(current_wp.lane_id) == int(waypoint.lane_id)
                    )
                    if yaw_diff > 70.0 and not same_lane:
                        continue

                stop_point = self._stop_point_from_waypoint(waypoint)
                route_distance = None
                route_lateral = None
                if route_points and ego_route_index is not None:
                    stop_route_index, route_lateral = self._route_progress(stop_point)
                    if route_lateral > corridor_limit or stop_route_index < ego_route_index:
                        continue
                    route_distance = self._route_distance(ego_route_index, stop_route_index)

                item = {
                    "traffic_light_id": int(light.id),
                    "distance_to_line_m": round(float(forward), 3),
                    "distance_to_stop_m": round(float(stopline_distance), 3),
                    "route_distance_m": round(float(route_distance), 3) if route_distance is not None else None,
                    "route_corridor_lateral_m": round(float(route_lateral), 3) if route_lateral is not None else None,
                    "lateral_m": round(float(lateral), 3),
                    "road_id": int(waypoint.road_id),
                    "lane_id": int(waypoint.lane_id),
                    "stop_point": stop_point,
                    "actor": light,
                }
                score = (
                    item["route_distance_m"] if item["route_distance_m"] is not None else item["distance_to_line_m"],
                    abs(item["lateral_m"]),
                )
                if best is None or score < best[0]:
                    best = (score, item)
        return None if best is None else best[1]

    def _normalize_traffic_light_state(self, state) -> str:
        if state is None:
            return "unknown"
        text = str(getattr(state, "name", state)).strip().lower().rsplit(".", 1)[-1]
        if text in {"red", "yellow", "green"}:
            return text
        return "unknown"

    def _detector_detection_for_id(self, light_id: int) -> Optional[dict]:
        payload = self.traffic_light_payload or {}
        max_age = max(0.0, float(self.get_parameter("traffic_light_max_age_s").value))
        if self.last_traffic_light_s <= 0.0 or time.time() - self.last_traffic_light_s > max_age:
            return None
        for item in payload.get("detections") or []:
            if not isinstance(item, dict):
                continue
            try:
                if int(item.get("actor_id")) == int(light_id):
                    return item
            except (TypeError, ValueError):
                continue
        for key in ("selected_detection", "primary_detection"):
            item = payload.get(key)
            if not isinstance(item, dict):
                continue
            try:
                if int(item.get("actor_id")) == int(light_id):
                    return item
            except (TypeError, ValueError):
                continue
        return None

    def light_color_for_actor(self, light) -> tuple[str, str]:
        light_id = int(light.id)
        minimum_confidence = float(self.get_parameter("traffic_light_min_confidence").value)
        detection = self._detector_detection_for_id(light_id)
        if detection is not None:
            color = str(
                detection.get("tl_color_filtered")
                or detection.get("tl_color_raw")
                or detection.get("traffic_light_state")
                or detection.get("color")
                or "unknown"
            ).lower()
            try:
                confidence = float(detection.get("tl_confidence", 1.0) or 0.0)
            except (TypeError, ValueError):
                confidence = 0.0
            if color in {"red", "yellow", "green"} and confidence >= minimum_confidence:
                source = str(detection.get("source") or "traffic_light_detector_node")
                return color, f"detector_match:{source}"

        try:
            color = self._normalize_traffic_light_state(light.get_state())
            if color in {"red", "yellow", "green"}:
                return color, "carla_light_actor_state"
        except Exception:
            pass
        return "unknown", "unknown_light_color"

    def held_light_context(self, ego_vehicle) -> Optional[dict]:
        if self.held_tl_id is None:
            return None
        world = self.ensure_carla_world()
        if world is None:
            return None
        light = world.get_actor(int(self.held_tl_id))
        if light is None:
            self.held_tl_id = None
            self.held_stop_point = None
            self.green_seen_since_s = None
            return None

        stop_point = dict(self.held_stop_point or {})
        distance_m = None
        if ego_vehicle is not None and stop_point:
            ego_transform = ego_vehicle.get_transform()
            location = self.carla.Location(
                x=float(stop_point["x"]),
                y=float(stop_point["y"]),
                z=float(stop_point.get("z", 0.0) or 0.0),
            )
            forward, lateral = self.local_xy(ego_transform, location)
            stop_before_line_m = float(self.get_parameter("stop_before_line_m").value)
            distance_m = round(float(forward - stop_before_line_m), 3)
            stop_point["lateral_m"] = round(float(lateral), 3)

        color, color_source = self.light_color_for_actor(light)
        return {
            "traffic_light_id": int(light.id),
            "distance_to_stop_m": distance_m,
            "stop_point": stop_point or None,
            "actor": light,
            "color": color,
            "color_source": color_source,
        }

    def publish_event(self, output: dict):
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
                f"reason={output['reason']} "
                f"color_source={output['color_source']}"
            )

    def build_output(
        self,
        *,
        relevant: bool,
        color: str,
        distance_m: float | None,
        stop_point: dict | None,
        tl_id: int | None,
        reason: str,
        color_source: str,
        source: str,
    ) -> dict:
        return {
            "stamp": time.time(),
            "has_relevant_light": bool(relevant),
            "color": color if relevant else "unknown",
            "distance_m": round(float(distance_m), 3) if distance_m is not None else None,
            "stop_point": stop_point,
            "source": source,
            "tl_id": str(tl_id) if tl_id is not None else None,
            "reason": reason,
            "color_source": color_source,
        }

    def draw_selected_marker(self):
        if not bool(self.get_parameter("debug_draw_waypoints").value):
            return
        world = self.ensure_carla_world()
        if world is None or self.carla is None:
            return
        context = self.held_light_context(self.find_ego_vehicle())
        if context is None or not context.get("stop_point"):
            return
        point = context["stop_point"]
        life_time = float(self.get_parameter("debug_draw_life_time_s").value)
        location = self.carla.Location(
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point.get("z", 0.0)),
        )
        world.debug.draw_point(
            location,
            size=0.18,
            color=self.carla.Color(255, 0, 0),
            life_time=life_time,
        )
        world.debug.draw_string(
            location + self.carla.Location(z=0.35),
            str(context["traffic_light_id"]),
            draw_shadow=True,
            color=self.carla.Color(255, 255, 255),
            life_time=life_time,
        )

    def tick(self):
        ego_vehicle = self.find_ego_vehicle()
        selected = self.relevant_stopline_ahead(ego_vehicle)
        held = self.held_light_context(ego_vehicle)
        now = time.time()

        if selected is not None:
            color, color_source = self.light_color_for_actor(selected["actor"])
            selected["color"] = color
            selected["color_source"] = color_source
        elif held is not None:
            color = held["color"]
        else:
            color = "unknown"

        if selected is not None and selected["color"] in {"red", "yellow"}:
            self.held_tl_id = int(selected["traffic_light_id"])
            self.held_stop_point = dict(selected["stop_point"])
            self.green_seen_since_s = None
            output = self.build_output(
                relevant=True,
                color=selected["color"],
                distance_m=selected["distance_to_stop_m"],
                stop_point=selected["stop_point"],
                tl_id=selected["traffic_light_id"],
                reason=f"{selected['color']}_approach_hold",
                color_source=selected["color_source"],
                source="carla_stopline_ahead",
            )
            self.publish_event(output)
            return

        if held is not None and held["color"] == "green":
            if self.green_seen_since_s is None:
                self.green_seen_since_s = now
            release_active = now - self.green_seen_since_s >= float(self.get_parameter("green_release_confirm_s").value)
            output = self.build_output(
                relevant=True,
                color="green",
                distance_m=held["distance_to_stop_m"],
                stop_point=held["stop_point"],
                tl_id=held["traffic_light_id"],
                reason="green_release" if release_active else "green_seen_wait_confirm",
                color_source=held["color_source"],
                source="held_light_release",
            )
            if release_active:
                self.held_tl_id = None
                self.held_stop_point = None
                self.green_seen_since_s = None
            self.publish_event(output)
            return

        if held is not None and held["color"] in {"red", "yellow"}:
            self.green_seen_since_s = None
            output = self.build_output(
                relevant=True,
                color=held["color"],
                distance_m=held["distance_to_stop_m"],
                stop_point=held["stop_point"],
                tl_id=held["traffic_light_id"],
                reason=f"{held['color']}_hold",
                color_source=held["color_source"],
                source="held_light",
            )
            self.publish_event(output)
            return

        if selected is not None and selected["color"] == "green":
            self.green_seen_since_s = None
            output = self.build_output(
                relevant=True,
                color="green",
                distance_m=selected["distance_to_stop_m"],
                stop_point=selected["stop_point"],
                tl_id=selected["traffic_light_id"],
                reason="green_light_ahead",
                color_source=selected["color_source"],
                source="carla_stopline_ahead",
            )
            self.publish_event(output)
            return

        self.green_seen_since_s = None
        output = self.build_output(
            relevant=False,
            color="unknown",
            distance_m=None,
            stop_point=None,
            tl_id=None,
            reason="no_route_relevant_traffic_light",
            color_source="no_reliable_traffic_light_color",
            source="traffic_light_manager_node",
        )
        self.publish_event(output)


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
