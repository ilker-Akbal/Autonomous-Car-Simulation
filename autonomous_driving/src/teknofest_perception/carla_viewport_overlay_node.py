#!/usr/bin/env python3
from __future__ import annotations

import json
import math
from typing import Optional

from cv_bridge import CvBridge
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _distance_xyz(a, b) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


class CarlaViewportOverlayNode(Node):
    def __init__(self):
        super().__init__("carla_viewport_overlay_node")

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("overlay_rate_hz", 12.0)
        self.declare_parameter("draw_lifetime_s", 0.10)
        self.declare_parameter("draw_mission_points", True)
        self.declare_parameter("draw_model_detections", False)
        self.declare_parameter("draw_stopline_overlay", False)
        self.declare_parameter("draw_tl_actor_fallback", False)
        self.declare_parameter("draw_sign_actor_fallback", False)
        self.declare_parameter("draw_carla_actor_debug", False)
        self.declare_parameter("draw_bbox_only", False)
        self.declare_parameter("draw_model_labels", False)
        self.declare_parameter("draw_model_confidence", False)
        self.declare_parameter("draw_detection_text", False)
        self.declare_parameter("draw_traffic_light_text", False)
        self.declare_parameter("draw_tl_proxy_detections", False)
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("traffic_light_topic", "/adas/perception/traffic_lights")
        self.declare_parameter("model_detections_topic", "/adas/perception/model_detections")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 360)
        self.declare_parameter("camera_fov_deg", 72.0)
        self.declare_parameter("camera_x", 1.6)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 2.25)
        self.declare_parameter("camera_pitch_deg", -1.0)
        self.declare_parameter("sign_topics", ["/adas/perception/traffic_signs"])
        self.declare_parameter("object_topics", [])

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.overlay_period_s = 1.0 / max(5.0, float(self.get_parameter("overlay_rate_hz").value))
        self.draw_lifetime_s = float(self.get_parameter("draw_lifetime_s").value)
        self.draw_mission_points_enabled = bool(self.get_parameter("draw_mission_points").value)
        self.draw_model_detections_enabled = bool(self.get_parameter("draw_model_detections").value)
        self.draw_bbox_only = bool(self.get_parameter("draw_bbox_only").value)
        self.draw_model_labels = bool(self.get_parameter("draw_model_labels").value)
        self.draw_model_confidence = bool(self.get_parameter("draw_model_confidence").value)
        self.draw_detection_text = bool(self.get_parameter("draw_detection_text").value)
        self.draw_traffic_light_text = bool(self.get_parameter("draw_traffic_light_text").value)
        self.draw_tl_proxy_detections = bool(self.get_parameter("draw_tl_proxy_detections").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fov_deg = float(self.get_parameter("camera_fov_deg").value)
        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)
        self.camera_pitch_deg = float(self.get_parameter("camera_pitch_deg").value)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()
        self.ego_vehicle = None

        self.mission_payload: Optional[dict] = None
        self.traffic_light_payload: Optional[dict] = None
        self.model_detections_payload: Optional[dict] = None
        self.sign_payloads: dict[str, dict | list] = {}
        self.object_payloads: dict[str, dict | list] = {}

        self.create_subscription(String, str(self.get_parameter("mission_topic").value), self.mission_cb, 10)
        self.create_subscription(String, str(self.get_parameter("traffic_light_topic").value), self.traffic_light_cb, 10)
        self.create_subscription(String, str(self.get_parameter("model_detections_topic").value), self.model_detections_cb, 10)
        for topic in list(self.get_parameter("sign_topics").value):
            self.create_subscription(String, str(topic), self.make_payload_cb(self.sign_payloads, str(topic)), 10)
        for topic in list(self.get_parameter("object_topics").value):
            if str(topic) == str(self.get_parameter("model_detections_topic").value):
                continue
            self.create_subscription(String, str(topic), self.make_payload_cb(self.object_payloads, str(topic)), 10)

        self.create_timer(self.overlay_period_s, self.tick)
        self.get_logger().info("CARLA viewport overlay ready")

    def parse_json(self, data: str) -> Optional[dict | list]:
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    def mission_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.mission_payload = payload

    def traffic_light_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.traffic_light_payload = payload

    def model_detections_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.model_detections_payload = payload

    def make_payload_cb(self, store: dict, topic: str):
        def _cb(msg: String):
            payload = self.parse_json(msg.data)
            if isinstance(payload, (dict, list)):
                store[topic] = payload
        return _cb

    def find_ego_vehicle(self):
        if self.ego_vehicle is not None:
            try:
                if self.ego_vehicle.is_alive:
                    return self.ego_vehicle
            except Exception:
                self.ego_vehicle = None
        try:
            self.world = self.client.get_world()
        except Exception:
            return None
        for vehicle in self.world.get_actors().filter("vehicle.*"):
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self.ego_vehicle = vehicle
                return vehicle
        return None

    def draw_string(self, location, text: str, color, z_offset: float = 1.2):
        if location is None:
            return
        self.world.debug.draw_string(
            location + self.carla.Location(z=z_offset),
            text,
            draw_shadow=False,
            color=color,
            life_time=self.draw_lifetime_s,
            persistent_lines=False,
        )

    def draw_vertical_marker(self, location, color, height_m: float = 3.5):
        top = location + self.carla.Location(z=height_m)
        self.world.debug.draw_line(
            location,
            top,
            thickness=0.10,
            color=color,
            life_time=self.draw_lifetime_s,
            persistent_lines=False,
        )

    def mission_locations(self) -> list[dict]:
        payload = self.mission_payload or {}
        mission = payload.get("mission") if isinstance(payload, dict) else None
        points = []
        if not isinstance(mission, dict):
            return points
        task_points = mission.get("task_points") or []
        for item in task_points:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "")
            if name not in {"gorev_1", "gorev_2"}:
                continue
            x = _safe_float(item.get("carla_x"))
            y = _safe_float(item.get("carla_y"))
            z = _safe_float(item.get("carla_z"))
            if None in {x, y}:
                continue
            points.append({
                "name": name.upper(),
                "location": self.carla.Location(x=x, y=y, z=0.5 if z is None else z),
            })
        return points

    def draw_mission_points(self, ego_transform):
        if not self.draw_mission_points_enabled:
            return
        ego_loc = ego_transform.location
        for item in self.mission_locations():
            location = item["location"]
            distance_m = _distance_xyz(ego_loc, location)
            color = self.carla.Color(0, 255, 255)
            self.draw_vertical_marker(location, color)
            self.draw_string(location, f"{item['name']} {distance_m:.1f}m", color, z_offset=3.8)

    def tick(self):
        ego_vehicle = self.find_ego_vehicle()
        if ego_vehicle is None:
            return
        try:
            ego_transform = ego_vehicle.get_transform()
        except Exception:
            self.ego_vehicle = None
            return
        self.draw_mission_points(ego_transform)


def main(args=None):
    rclpy.init(args=args)
    node = CarlaViewportOverlayNode()
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
