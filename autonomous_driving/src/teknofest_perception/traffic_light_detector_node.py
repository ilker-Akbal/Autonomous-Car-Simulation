#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla


class TrafficLightDetectorNode(Node):
    def __init__(self):
        super().__init__("traffic_light_detector_node")

        # -------------------------
        # Config / parameter block
        # -------------------------
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("traffic_light_topic", "/adas/perception/traffic_lights")
        self.declare_parameter(
            "traffic_light_model_path",
            "autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/best.pt",
        )
        self.declare_parameter(
            "traffic_light_classes_path",
            "autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/classes.json",
        )
        self.declare_parameter("publish_period_s", 0.1)
        self.declare_parameter("max_lights", 32)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 2.0)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.model_path = Path(str(self.get_parameter("traffic_light_model_path").value))
        self.classes_path = Path(str(self.get_parameter("traffic_light_classes_path").value))
        self.max_lights = int(self.get_parameter("max_lights").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        # -------------------------
        # Runtime state block
        # -------------------------
        self.carla = None
        self.client = None
        self.world = None
        self.last_image_stamp_s = 0.0
        self.last_ros_log_s = 0.0
        self.model_status = self.inspect_model()

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="traffic_light_detector_node",
            file_name="traffic_light.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        # -------------------------
        # Publisher block
        # -------------------------
        self.pub = self.create_publisher(
            String,
            str(self.get_parameter("traffic_light_topic").value),
            10,
        )

        # -------------------------
        # Subscriber block
        # -------------------------
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self.image_cb,
            10,
        )

        # -------------------------
        # Timer / startup block
        # -------------------------
        self.connect_to_carla()
        self.create_timer(float(self.get_parameter("publish_period_s").value), self.tick)
        self.get_logger().info(
            "Traffic light detector ready. Model analysis: "
            f"{self.model_status['model_type']} ({self.model_status['load_status']})"
        )

    # -------------------------
    # Model / CARLA helper block
    # -------------------------
    def inspect_model(self) -> dict:
        classes = []
        img_size = None
        if self.classes_path.exists():
            try:
                payload = json.loads(self.classes_path.read_text(encoding="utf-8"))
                classes = list(payload.get("class_names") or [])
                img_size = payload.get("img_size")
            except Exception as exc:
                return {
                    "model_exists": self.model_path.exists(),
                    "classes_exists": True,
                    "model_type": "unknown",
                    "load_status": f"classes_read_failed:{exc}",
                    "class_names": [],
                    "img_size": None,
                }

        classifier_like = set(classes) >= {"red", "yellow", "green", "unknown"}
        load_status = "not_loaded"
        try:
            import torch  # noqa: F401

            load_status = "torch_available_classifier_not_detector"
        except Exception as exc:
            load_status = f"torch_unavailable:{type(exc).__name__}"

        return {
            "model_exists": self.model_path.exists(),
            "classes_exists": self.classes_path.exists(),
            "model_type": "color_classifier" if classifier_like else "unknown",
            "load_status": load_status,
            "class_names": classes,
            "img_size": img_size,
            "note": "best.pt is treated as a color classifier; ROI/bbox detection is not inferred.",
        }

    def connect_to_carla(self):
        carla_python_api = os.path.join(self.carla_root, "PythonAPI", "carla")
        if os.path.isdir(carla_python_api) and carla_python_api not in sys.path:
            sys.path.append(carla_python_api)

        self.carla = load_carla(self.carla_root)
        self.client = self.carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        self.world = self.client.get_world()

    def image_cb(self, _msg: Image):
        self.last_image_stamp_s = time.time()

    def color_from_state(self, state) -> str:
        name = getattr(state, "name", str(state)).lower()
        if "red" in name:
            return "red"
        if "yellow" in name:
            return "yellow"
        if "green" in name:
            return "green"
        return "unknown"

    def actor_location_dict(self, loc) -> dict:
        return {
            "x": round(float(loc.x), 4),
            "y": round(float(loc.y), 4),
            "z": round(float(loc.z), 4),
        }

    def stop_waypoints(self, actor) -> list[dict]:
        out = []
        try:
            waypoints = actor.get_stop_waypoints()
        except Exception:
            waypoints = []

        for index, waypoint in enumerate(waypoints or []):
            transform = waypoint.transform
            out.append({
                "index": index,
                "x": round(float(transform.location.x), 4),
                "y": round(float(transform.location.y), 4),
                "z": round(float(transform.location.z), 4),
                "yaw_deg": round(float(transform.rotation.yaw), 4),
                "road_id": int(waypoint.road_id),
                "lane_id": int(waypoint.lane_id),
            })
        return out

    # -------------------------
    # Detection publish block
    # -------------------------
    def tick(self):
        now = time.time()
        try:
            actors = list(self.world.get_actors().filter("traffic.traffic_light*"))
        except Exception as exc:
            self.get_logger().warning(f"Traffic light actor query failed: {exc}")
            actors = []

        lights = []
        for actor in actors[: self.max_lights]:
            try:
                transform = actor.get_transform()
                color = self.color_from_state(actor.state)
                stop_waypoints = self.stop_waypoints(actor)
                lights.append({
                    "actor_id": int(actor.id),
                    "tl_detected": True,
                    "tl_color_raw": color,
                    "tl_confidence": 1.0,
                    "source": "carla_actor_state",
                    "location": self.actor_location_dict(transform.location),
                    "yaw_deg": round(float(transform.rotation.yaw), 4),
                    "stop_waypoints": stop_waypoints,
                })
            except Exception as exc:
                self.get_logger().debug(f"Traffic light actor skipped: {exc}")

        payload = {
            "stamp": now,
            "source": "traffic_light_detector_node",
            "model_status": self.model_status,
            "image_age_ms": int(max(0.0, now - self.last_image_stamp_s) * 1000.0)
            if self.last_image_stamp_s > 0.0 else None,
            "tl_detected": bool(lights),
            "detections": lights,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.pub.publish(msg)
        self.log_runtime(payload)

    def log_runtime(self, payload: dict):
        detections = payload.get("detections") or []
        first = detections[0] if detections else {}
        record = {
            "tl_detected": payload.get("tl_detected"),
            "tl_color_raw": first.get("tl_color_raw"),
            "tl_confidence": first.get("tl_confidence"),
            "candidate_count": len(detections),
            "model_type": self.model_status.get("model_type"),
            "model_load_status": self.model_status.get("load_status"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "traffic_light_detector "
                f"count={len(detections)} first={record['tl_color_raw']} "
                f"model={record['model_type']} load={record['model_load_status']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightDetectorNode()
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
