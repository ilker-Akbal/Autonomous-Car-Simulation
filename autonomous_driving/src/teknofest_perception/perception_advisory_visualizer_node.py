#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value) -> int | None:
    try:
        if value is None:
            return None
        return int(round(float(value)))
    except (TypeError, ValueError):
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class JsonlSink:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict):
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


class PerceptionAdvisoryVisualizerNode(Node):
    def __init__(self):
        super().__init__("perception_advisory_visualizer_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("image_topic_fallback", "/zed/zed_node/left/image_rect_color")
        self.declare_parameter("traffic_light_topic", "/adas/perception/traffic_lights")
        self.declare_parameter("model_detections_topic", "/adas/perception/model_detections")
        self.declare_parameter("tl_event_topic", "/adas/planning/tl_event")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("annotated_image_topic", "/adas/perception/annotated_image")
        self.declare_parameter("enable_perception_window", True)
        self.declare_parameter("perception_window_scale", 0.75)
        self.declare_parameter("visualizer_fps", 10.0)
        self.declare_parameter("draw_labels", True)
        self.declare_parameter("draw_confidence", False)
        self.declare_parameter("draw_traffic_lights", False)
        self.declare_parameter("draw_traffic_signs_topic", False)
        self.declare_parameter("draw_model_detections", True)
        self.declare_parameter("publish_annotated_image", True)
        self.declare_parameter("draw_tl_proxy_detections", False)
        self.declare_parameter("sign_topics", [
            "/adas/perception/signs",
            "/adas/perception/traffic_signs",
            "/adas/teknofest/signs",
            "/adas/route/signs",
        ])
        self.declare_parameter("object_topics", [
            "/adas/perception/objects",
            "/adas/perception/detections",
            "/adas/perception/obstacles",
        ])
        self.declare_parameter("session_id", time.strftime("%Y%m%d_%H%M%S"))
        self.declare_parameter(
            "log_root",
            "autonomous_driving/outputs/perception_advisory_logs",
        )
        self.declare_parameter("write_frame_log_period_s", 0.20)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.bridge = CvBridge()
        self.window_name = "TEKNOFEST Camera Perception"
        self.write_frame_log_period_s = float(self.get_parameter("write_frame_log_period_s").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)
        self.enable_perception_window = bool(self.get_parameter("enable_perception_window").value)
        self.perception_window_scale = max(0.1, float(self.get_parameter("perception_window_scale").value))
        self.visualizer_fps = max(1.0, float(self.get_parameter("visualizer_fps").value))
        self.draw_labels = bool(self.get_parameter("draw_labels").value)
        self.draw_confidence = bool(self.get_parameter("draw_confidence").value)
        self.draw_traffic_lights = bool(self.get_parameter("draw_traffic_lights").value)
        self.draw_traffic_signs_topic = bool(self.get_parameter("draw_traffic_signs_topic").value)
        self.draw_model_detections_enabled = bool(self.get_parameter("draw_model_detections").value)
        self.publish_annotated_image_enabled = bool(self.get_parameter("publish_annotated_image").value)
        self.draw_tl_proxy_detections = bool(self.get_parameter("draw_tl_proxy_detections").value)
        self.last_frame_log_s = 0.0
        self.last_ros_log_s = 0.0

        session_id = str(self.get_parameter("session_id").value)
        log_root = Path(str(self.get_parameter("log_root").value))
        self.session_dir = log_root / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.perception_log = JsonlSink(self.session_dir / "perception_advisory.jsonl")
        self.tl_events_log = JsonlSink(self.session_dir / "traffic_light_events.jsonl")
        self.sign_events_log = JsonlSink(self.session_dir / "sign_events.jsonl")
        self.object_events_log = JsonlSink(self.session_dir / "object_events.jsonl")
        self.mission_log = JsonlSink(self.session_dir / "mission_status.jsonl")

        self.image_message: Optional[Image] = None
        self.last_image_s = 0.0
        self.tl_event_payload: Optional[dict] = None
        self.traffic_light_payload: Optional[dict] = None
        self.model_detections_payload: Optional[dict] = None
        self.status_payload: Optional[dict] = None
        self.mission_payload: Optional[dict] = None
        self.sign_payloads: dict[str, dict] = {}
        self.object_payloads: dict[str, dict] = {}
        self.last_tl_event_signature = None

        self.annotated_image_pub = self.create_publisher(
            Image,
            str(self.get_parameter("annotated_image_topic").value),
            10,
        )

        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic").value),
            self.image_cb,
            10,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("image_topic_fallback").value),
            self.image_cb_fallback,
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
            str(self.get_parameter("model_detections_topic").value),
            self.model_detections_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("tl_event_topic").value),
            self.tl_event_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self.status_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mission_topic").value),
            self.mission_cb,
            10,
        )

        for topic in list(self.get_parameter("sign_topics").value):
            self.create_subscription(String, str(topic), self.make_sign_cb(str(topic)), 10)
        for topic in list(self.get_parameter("object_topics").value):
            self.create_subscription(String, str(topic), self.make_object_cb(str(topic)), 10)

        self.create_timer(1.0 / self.visualizer_fps, self.render_tick)
        self.get_logger().info("Perception advisory visualization started")

    def parse_json(self, data: str) -> Optional[dict | list]:
        try:
            return json.loads(data)
        except (json.JSONDecodeError, TypeError):
            return None

    def image_cb(self, msg: Image):
        self.image_message = msg
        self.last_image_s = time.time()

    def image_cb_fallback(self, msg: Image):
        if self.image_message is None:
            self.image_message = msg
            self.last_image_s = time.time()

    def traffic_light_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.traffic_light_payload = payload

    def tl_event_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.tl_event_payload = payload

    def model_detections_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.model_detections_payload = payload

    def status_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.status_payload = payload

    def mission_cb(self, msg: String):
        payload = self.parse_json(msg.data)
        if isinstance(payload, dict):
            self.mission_payload = payload
            self.mission_log.write({
                "stamp": time.time(),
                "mission": payload,
            })

    def make_sign_cb(self, topic: str):
        def _cb(msg: String):
            payload = self.parse_json(msg.data)
            if isinstance(payload, (dict, list)):
                self.sign_payloads[topic] = {
                    "stamp": time.time(),
                    "payload": payload,
                }
                for detection in self.extract_detections(payload, default_type="traffic_sign", source=topic):
                    self.sign_events_log.write({
                        "stamp": time.time(),
                        "label": detection["label"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "distance_m": detection["distance_m"],
                        "decision": self.sign_decision(detection["label"]),
                    })
        return _cb

    def make_object_cb(self, topic: str):
        def _cb(msg: String):
            payload = self.parse_json(msg.data)
            if isinstance(payload, (dict, list)):
                self.object_payloads[topic] = {
                    "stamp": time.time(),
                    "payload": payload,
                }
                for detection in self.extract_detections(payload, default_type="object", source=topic):
                    self.object_events_log.write({
                        "stamp": time.time(),
                        "label": detection["label"],
                        "confidence": detection["confidence"],
                        "bbox": detection["bbox"],
                        "distance_m": detection["distance_m"],
                    })
        return _cb

    def advisory_from_tl_event(self) -> tuple[dict, dict]:
        payload = self.tl_event_payload or {}
        has_relevant_light = bool(payload.get("has_relevant_light", False))
        color = str(payload.get("color") or "unknown").lower()
        distance_m = _safe_float(payload.get("distance_m"))
        color_source = payload.get("color_source")
        reason = str(payload.get("reason") or "no_relevant_light")

        if (not has_relevant_light or color == "unknown") and isinstance(self.traffic_light_payload, dict):
            detections = self.traffic_light_payload.get("detections") or []
            if detections and isinstance(detections[0], dict):
                first = detections[0]
                color = str(first.get("label") or first.get("tl_color_raw") or color).lower()
                color_source = first.get("source") or "traffic_light_detector_node"
                reason = "detector_fallback"
                distance_m = _safe_float(first.get("distance_m")) or distance_m
                has_relevant_light = color in {"red", "yellow", "green"}

        decision_type = "UNKNOWN"
        recommended_brake = 0.0

        if has_relevant_light and color == "red":
            if distance_m is not None and distance_m <= 8.0:
                decision_type = "STOP"
                recommended_brake = 1.0
            elif distance_m is not None and distance_m <= 25.0:
                decision_type = "SLOW"
                recommended_brake = 0.35
            else:
                decision_type = "PREPARE"
                recommended_brake = 0.0
        elif has_relevant_light and color == "yellow":
            if distance_m is not None and distance_m <= 15.0:
                decision_type = "STOP_OR_SLOW"
                recommended_brake = 0.6
            else:
                decision_type = "SLOW"
                recommended_brake = 0.35
        elif has_relevant_light and color == "green":
            decision_type = "GO"
            recommended_brake = 0.0

        traffic_light = {
            "has_relevant_light": has_relevant_light,
            "color": color,
            "distance_m": distance_m,
            "tl_id": payload.get("tl_id"),
            "reason": reason,
            "color_source": color_source,
        }
        decision = {
            "type": decision_type,
            "recommended_brake": recommended_brake,
            "stop_required": decision_type in {"STOP", "STOP_OR_SLOW"},
            "reason": reason,
        }
        return traffic_light, decision

    def tl_bbox_detections(self) -> list[dict]:
        payload = self.traffic_light_payload
        if not isinstance(payload, dict):
            return []
        detections = self.extract_detections(payload, default_type="traffic_light", source=str(payload.get("source") or "traffic_light_topic"))
        filtered = []
        for detection in detections:
            source = str(detection.get("source") or "").lower()
            if source in {"traffic_light_model_proxy", "carla_actor_state", "actor_fallback"} and not self.draw_tl_proxy_detections:
                continue
            filtered.append(detection)
        color = str((self.tl_event_payload or {}).get("color") or "unknown").lower()
        for detection in filtered:
            if detection["label"] in {"unknown", "traffic_light"} and color in {"red", "yellow", "green"}:
                detection["label"] = f"traffic_light:{color}"
        return filtered

    def sign_decision(self, label: str) -> str:
        text = str(label or "unknown").lower()
        if "stop" in text or "dur" in text:
            return "STOP"
        if "speed" in text or "hiz" in text:
            return "SPEED_LIMIT"
        if "no entry" in text or "girilmez" in text:
            return "DO_NOT_ENTER"
        if "left" in text or "right" in text or "turn" in text or "donulmez" in text:
            return "TURN_RESTRICTION"
        if "pedestrian" in text or "crosswalk" in text or "yaya" in text:
            return "SLOW"
        return "SIGN_DETECTED"

    def extract_bbox(self, payload) -> dict | None:
        if payload is None:
            return None
        if isinstance(payload, list) and len(payload) >= 4:
            first = [_safe_float(v) for v in payload[:4]]
            if all(v is not None for v in first):
                x1, y1, x2, y2 = first
                if x2 >= x1 and y2 >= y1:
                    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
                return {"x1": x1, "y1": y1, "x2": x1 + x2, "y2": y1 + y2}
        if isinstance(payload, dict):
            if "bbox" in payload:
                return self.extract_bbox(payload["bbox"])
            if "box" in payload:
                return self.extract_bbox(payload["box"])
            xmin = _safe_float(payload.get("xmin"))
            ymin = _safe_float(payload.get("ymin"))
            xmax = _safe_float(payload.get("xmax"))
            ymax = _safe_float(payload.get("ymax"))
            if None not in {xmin, ymin, xmax, ymax}:
                return {"x1": xmin, "y1": ymin, "x2": xmax, "y2": ymax}
            x = _safe_float(payload.get("x"))
            y = _safe_float(payload.get("y"))
            w = _safe_float(payload.get("w"))
            h = _safe_float(payload.get("h"))
            if None not in {x, y, w, h}:
                return {"x1": x, "y1": y, "x2": x + w, "y2": y + h}
        return None

    def extract_detections(self, payload, *, default_type: str, source: str) -> list[dict]:
        candidates = []
        if isinstance(payload, dict):
            for key in ("detections", "boxes", "objects", "signs", "items"):
                value = payload.get(key)
                if isinstance(value, list):
                    candidates.extend(value)
            if not candidates:
                candidates.append(payload)
        elif isinstance(payload, list):
            candidates.extend(payload)

        detections = []
        for item in candidates:
            if not isinstance(item, dict):
                continue
            bbox = self.extract_bbox(item)
            label = (
                item.get("label")
                or item.get("class")
                or item.get("name")
                or item.get("traffic_light_state")
                or item.get("tl_color_raw")
                or default_type
            )
            confidence = (
                _safe_float(item.get("confidence"))
                or _safe_float(item.get("score"))
                or _safe_float(item.get("tl_confidence"))
            )
            distance_m = (
                _safe_float(item.get("distance_m"))
                or _safe_float(item.get("distance_to_stop_m"))
                or _safe_float(item.get("distance_est"))
            )
            detections.append({
                "type": default_type,
                "label": str(label),
                "confidence": confidence,
                "bbox": bbox,
                "distance_m": distance_m,
                "source": source,
            })
        return detections

    def aggregate_sign_detections(self) -> list[dict]:
        detections = []
        for topic, payload in self.sign_payloads.items():
            detections.extend(self.extract_detections(payload.get("payload"), default_type="traffic_sign", source=topic))
        deduped = []
        seen = set()
        for detection in detections:
            bbox = detection.get("bbox") or {}
            key = (
                str(detection.get("label") or ""),
                _safe_int(bbox.get("x1")),
                _safe_int(bbox.get("y1")),
                _safe_int(bbox.get("x2")),
                _safe_int(bbox.get("y2")),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(detection)
        return deduped

    def aggregate_object_detections(self) -> list[dict]:
        detections = []
        if isinstance(self.model_detections_payload, dict):
            detections.extend(
                self.extract_detections(
                    self.model_detections_payload,
                    default_type="object",
                    source=str(self.model_detections_payload.get("source") or "model_detections"),
                )
            )
        for topic, payload in self.object_payloads.items():
            detections.extend(self.extract_detections(payload.get("payload"), default_type="object", source=topic))
        return detections

    def ego_speed_mps(self) -> float | None:
        if not isinstance(self.status_payload, dict):
            return None
        return _safe_float(self.status_payload.get("speed_mps"))

    def draw_detection(self, image, detection: dict, color: tuple[int, int, int]):
        bbox = detection.get("bbox")
        if not isinstance(bbox, dict):
            return
        x1 = _safe_int(bbox.get("x1"))
        y1 = _safe_int(bbox.get("y1"))
        x2 = _safe_int(bbox.get("x2"))
        y2 = _safe_int(bbox.get("y2"))
        if None in {x1, y1, x2, y2}:
            return
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(image.shape[1] - 1, x2)
        y2 = min(image.shape[0] - 1, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        if not self.draw_labels:
            return
        label = str(detection["label"])
        text = label
        confidence = detection.get("confidence")
        if self.draw_confidence and confidence is not None:
            text = f"{label} {confidence:.2f}"
        cv2.putText(
            image,
            text,
            (x1, max(18, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    def render_tick(self):
        if self.image_message is None:
            return
        try:
            image = self.bridge.imgmsg_to_cv2(self.image_message, desired_encoding="bgr8")
        except Exception:
            return

        annotated = image.copy()
        traffic_light, decision = self.advisory_from_tl_event()
        tl_detections = self.tl_bbox_detections() if self.draw_traffic_lights else []
        sign_detections = self.aggregate_sign_detections() if self.draw_traffic_signs_topic else []
        object_detections = self.aggregate_object_detections() if self.draw_model_detections_enabled else []

        sign_keys = {
            (
                str(det.get("label") or ""),
                _safe_int((det.get("bbox") or {}).get("x1")),
                _safe_int((det.get("bbox") or {}).get("y1")),
                _safe_int((det.get("bbox") or {}).get("x2")),
                _safe_int((det.get("bbox") or {}).get("y2")),
            )
            for det in sign_detections
        }
        filtered_objects = []
        for detection in object_detections:
            bbox = detection.get("bbox") or {}
            key = (
                str(detection.get("label") or ""),
                _safe_int(bbox.get("x1")),
                _safe_int(bbox.get("y1")),
                _safe_int(bbox.get("x2")),
                _safe_int(bbox.get("y2")),
            )
            if key in sign_keys:
                continue
            filtered_objects.append(detection)
        object_detections = filtered_objects

        for detection in tl_detections:
            self.draw_detection(annotated, detection, (0, 0, 255))
        for detection in sign_detections:
            self.draw_detection(annotated, detection, (0, 255, 255))
        for detection in object_detections:
            self.draw_detection(annotated, detection, (0, 255, 0))

        if self.publish_annotated_image_enabled:
            self.annotated_image_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        if self.enable_perception_window:
            display = annotated
            if abs(self.perception_window_scale - 1.0) > 1e-6:
                display = cv2.resize(
                    annotated,
                    None,
                    fx=self.perception_window_scale,
                    fy=self.perception_window_scale,
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow(self.window_name, display)
            cv2.waitKey(1)
        detections = tl_detections + sign_detections + object_detections
        frame_record = {
            "stamp": time.time(),
            "speed_mps": self.ego_speed_mps(),
            "detections": detections,
            "traffic_light": traffic_light,
            "decision": decision,
            "bbox_count": sum(1 for item in detections if item.get("bbox") is not None),
        }

        now = time.time()
        if now - self.last_frame_log_s >= self.write_frame_log_period_s:
            self.last_frame_log_s = now
            self.perception_log.write(frame_record)

        tl_signature = (
            traffic_light.get("has_relevant_light"),
            traffic_light.get("color"),
            traffic_light.get("distance_m"),
            traffic_light.get("tl_id"),
            decision.get("type"),
        )
        if tl_signature != self.last_tl_event_signature:
            self.last_tl_event_signature = tl_signature
            self.tl_events_log.write({
                "stamp": now,
                "color": traffic_light.get("color"),
                "distance_m": traffic_light.get("distance_m"),
                "tl_id": traffic_light.get("tl_id"),
                "decision": decision.get("type"),
                "recommended_brake": decision.get("recommended_brake"),
                "reason": decision.get("reason"),
                "source": traffic_light.get("color_source"),
            })

        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "perception_advisory "
                f"decision={decision['type']} "
                f"bbox_count={frame_record['bbox_count']} "
                f"speed_mps={frame_record['speed_mps']} "
                f"tl_color={traffic_light.get('color')}"
            )

    def destroy_node(self):
        if self.enable_perception_window:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        return super().destroy_node()

def main(args=None):
    rclpy.init(args=args)
    node = PerceptionAdvisoryVisualizerNode()
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
