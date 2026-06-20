#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover
    YOLO = None


def _safe_float(value) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


class TrafficSignDetectorNode(Node):
    def __init__(self):
        super().__init__("traffic_sign_detector_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("traffic_sign_topic", "/adas/perception/traffic_signs")
        self.declare_parameter("model_detections_topic", "/adas/perception/model_detections")
        self.declare_parameter(
            "model_path",
            "/home/ilker/Masaüstü/Autonomous-Car-Simulation/autonomous_driving/outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt",
        )
        self.declare_parameter(
            "sign_plan_geojson",
            "autonomous_driving/missions/town03_competition_v4_sign_plan.geojson",
        )
        self.declare_parameter("confidence_threshold", 0.35)
        self.declare_parameter("image_size", 1024)
        self.declare_parameter("process_every_n_frames", 3)
        self.declare_parameter("ros_log_period_s", 2.0)
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 20.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 360)
        self.declare_parameter("camera_fov_deg", 72.0)
        self.declare_parameter("camera_x", 1.6)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 2.25)
        self.declare_parameter("camera_pitch_deg", -1.0)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.model_path = Path(str(self.get_parameter("model_path").value))
        self.sign_plan_geojson = Path(str(self.get_parameter("sign_plan_geojson").value))
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.image_size = int(self.get_parameter("image_size").value)
        self.process_every_n_frames = max(1, int(self.get_parameter("process_every_n_frames").value))
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)
        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fov_deg = float(self.get_parameter("camera_fov_deg").value)
        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)
        self.camera_pitch_deg = float(self.get_parameter("camera_pitch_deg").value)

        self.bridge = CvBridge()
        self.model = None
        self.model_names = {}
        self.last_image_msg: Optional[Image] = None
        self.frame_count = 0
        self.last_ros_log_s = 0.0
        self.carla = None
        self.client = None
        self.world = None
        self.ego_vehicle = None
        self.sign_plan_entries = self.load_sign_plan()

        self.sign_pub = self.create_publisher(
            String,
            str(self.get_parameter("traffic_sign_topic").value),
            10,
        )
        self.model_pub = self.create_publisher(
            String,
            str(self.get_parameter("model_detections_topic").value),
            10,
        )

        self.create_subscription(Image, self.image_topic, self.image_cb, 10)
        self.create_timer(0.05, self.tick)

        self.load_model()
        self.connect_to_carla()

    def load_model(self):
        if YOLO is None:
            self.get_logger().warn(
                "ultralytics not installed; install with: python3 -m pip install ultralytics --user"
            )
            return
        if not self.model_path.exists():
            self.get_logger().warn(f"traffic_sign_detector_node: model not found: {self.model_path}")
            return
        try:
            self.model = YOLO(str(self.model_path))
            self.model_names = dict(getattr(self.model, "names", {}) or {})
            self.get_logger().info(
                f"traffic_sign_detector_node loaded model={self.model_path} classes={self.model_names}"
            )
        except Exception as exc:
            self.model = None
            self.model_names = {}
            self.get_logger().warn(f"traffic_sign_detector_node model load failed: {exc}")

    def load_sign_plan(self) -> list[dict]:
        if not self.sign_plan_geojson.exists():
            return []
        try:
            payload = json.loads(self.sign_plan_geojson.read_text(encoding="utf-8"))
        except Exception:
            return []
        entries = []
        for feature in payload.get("features", []):
            props = feature.get("properties") or {}
            coords = ((feature.get("geometry") or {}).get("coordinates") or [])
            if len(coords) < 2:
                continue
            entries.append({
                "sign": str(props.get("sign") or "traffic_sign"),
                "location": (float(coords[0]), float(coords[1]), float(coords[2]) if len(coords) > 2 else 0.0),
            })
        return entries

    def connect_to_carla(self):
        try:
            self.carla = load_carla(self.carla_root)
            self.client = self.carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
        except Exception:
            self.world = None

    def find_ego_vehicle(self):
        if self.world is None:
            return None
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

    def camera_transform(self, ego_vehicle):
        ego_transform = ego_vehicle.get_transform()
        base = ego_transform.location
        forward = ego_transform.get_forward_vector()
        right = ego_transform.get_right_vector()
        camera_location = self.carla.Location(
            x=base.x + forward.x * self.camera_x + right.x * self.camera_y,
            y=base.y + forward.y * self.camera_x + right.y * self.camera_y,
            z=base.z + self.camera_z,
        )
        camera_rotation = self.carla.Rotation(
            pitch=float(ego_transform.rotation.pitch) + self.camera_pitch_deg,
            yaw=float(ego_transform.rotation.yaw),
            roll=0.0,
        )
        return self.carla.Transform(camera_location, camera_rotation)

    def project_point(self, point, camera_transform) -> tuple[float, float] | None:
        matrix = camera_transform.get_inverse_matrix()
        px = float(point.x)
        py = float(point.y)
        pz = float(point.z)
        cam_x = matrix[0][0] * px + matrix[0][1] * py + matrix[0][2] * pz + matrix[0][3]
        cam_y = matrix[1][0] * px + matrix[1][1] * py + matrix[1][2] * pz + matrix[1][3]
        cam_z = matrix[2][0] * px + matrix[2][1] * py + matrix[2][2] * pz + matrix[2][3]
        if cam_x <= 0.1:
            return None
        focal = self.camera_width / (2.0 * math.tan(math.radians(self.camera_fov_deg) / 2.0))
        u = focal * (cam_y / cam_x) + self.camera_width / 2.0
        v = focal * (-cam_z / cam_x) + self.camera_height / 2.0
        return u, v

    def bbox_center(self, bbox: dict) -> tuple[float, float]:
        return ((float(bbox["xmin"]) + float(bbox["xmax"])) * 0.5, (float(bbox["ymin"]) + float(bbox["ymax"])) * 0.5)

    def infer_sign_from_actor_type(self, type_id: str) -> str | None:
        text = str(type_id or "").lower()
        if "speed_limit" in text:
            return text.split(".")[-1]
        if "no_entry" in text:
            return "no_entry"
        if "stop" in text:
            return "stop"
        if "yield" in text:
            return "yol_ver"
        if "turn" in text or "mandatory" in text or "mecburi" in text:
            return text.split(".")[-1]
        return None

    def refine_sign_label(self, detection: dict, camera_transform) -> tuple[str, str]:
        if str(detection.get("label") or "") != "traffic_sign":
            return str(detection.get("label") or "traffic_sign"), "model_class"
        bbox = detection.get("bbox") or {}
        if not isinstance(bbox, dict):
            return "traffic_sign", "model_class"
        center_u, center_v = self.bbox_center(bbox)
        best_label = "traffic_sign"
        best_source = "model_class"
        best_score = 1e9

        if self.world is not None:
            try:
                for actor in self.world.get_actors().filter("traffic.*"):
                    actor_label = self.infer_sign_from_actor_type(actor.type_id)
                    if actor_label is None:
                        continue
                    projected = self.project_point(actor.get_transform().location, camera_transform)
                    if projected is None:
                        continue
                    score = math.hypot(projected[0] - center_u, projected[1] - center_v)
                    if score < best_score and score <= 90.0:
                        best_score = score
                        best_label = actor_label
                        best_source = "carla_actor_match"
            except Exception:
                pass

        for entry in self.sign_plan_entries:
            projected = self.project_point(
                self.carla.Location(x=entry["location"][0], y=entry["location"][1], z=entry["location"][2]),
                camera_transform,
            )
            if projected is None:
                continue
            score = math.hypot(projected[0] - center_u, projected[1] - center_v)
            if score < best_score and score <= 90.0:
                best_score = score
                best_label = entry["sign"]
                best_source = "sign_plan_match"
        return best_label, best_source

    def image_cb(self, msg: Image):
        self.last_image_msg = msg
        self.frame_count += 1

    def is_sign_like_label(self, label: str) -> bool:
        text = str(label or "").lower()
        keywords = [
            "dur",
            "stop",
            "entry",
            "turn",
            "donulmez",
            "mandatory",
            "speed",
            "hiz",
            "sign",
            "yaya",
            "pedestrian",
            "crosswalk",
        ]
        return any(keyword in text for keyword in keywords)

    def empty_payload(self, stamp: float) -> dict:
        return {
            "stamp": stamp,
            "model_path": str(self.model_path),
            "source": "adas5_yolo_model",
            "detections": [],
        }

    def publish_payloads(self, all_payload: dict, sign_payload: dict):
        model_msg = String()
        model_msg.data = json.dumps(all_payload, ensure_ascii=False)
        self.model_pub.publish(model_msg)

        sign_msg = String()
        sign_msg.data = json.dumps(sign_payload, ensure_ascii=False)
        self.sign_pub.publish(sign_msg)

    def run_inference(self, image_bgr, camera_transform) -> list[dict]:
        if self.model is None:
            return []
        results = self.model.predict(
            source=image_bgr,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            verbose=False,
        )
        detections = []
        for result in results or []:
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            xyxy = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []
            confs = boxes.conf.tolist() if hasattr(boxes, "conf") else []
            classes = boxes.cls.tolist() if hasattr(boxes, "cls") else []
            for bbox, conf, class_id in zip(xyxy, confs, classes):
                if len(bbox) < 4:
                    continue
                class_id_int = int(class_id)
                label = str(self.model_names.get(class_id_int, class_id_int))
                detections.append({
                    "label": label,
                    "class_id": class_id_int,
                    "confidence": round(float(conf), 4),
                    "bbox": {
                        "xmin": round(float(bbox[0]), 2),
                        "ymin": round(float(bbox[1]), 2),
                        "xmax": round(float(bbox[2]), 2),
                        "ymax": round(float(bbox[3]), 2),
                    },
                    "distance_m": None,
                    "source": "adas5_yolo_model",
                    "bbox_source": "adas5_yolo_model",
                    "label_source": "model_class",
                })
                detections[-1]["label"], detections[-1]["label_source"] = self.refine_sign_label(detections[-1], camera_transform)
        return detections

    def tick(self):
        now = time.time()
        if self.last_image_msg is None:
            return
        if self.frame_count % self.process_every_n_frames != 0:
            return

        if self.model is None:
            payload = self.empty_payload(now)
            self.publish_payloads(payload, payload)
            return

        try:
            image = self.bridge.imgmsg_to_cv2(self.last_image_msg, desired_encoding="bgr8")
        except Exception:
            return
        ego_vehicle = self.find_ego_vehicle()
        camera_transform = self.camera_transform(ego_vehicle) if ego_vehicle is not None else None
        if camera_transform is None:
            return
        detections = self.run_inference(image, camera_transform)
        all_payload = {
            "stamp": now,
            "model_path": str(self.model_path),
            "source": "adas5_yolo_model",
            "detections": detections,
        }
        sign_detections = [item for item in detections if self.is_sign_like_label(item.get("label", ""))]
        if not sign_detections:
            sign_detections = list(detections)
        sign_payload = {
            "stamp": now,
            "model_path": str(self.model_path),
            "source": "adas5_yolo_model",
            "detections": sign_detections,
        }
        self.publish_payloads(all_payload, sign_payload)

        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                f"traffic_sign_detector detections={len(detections)} classes={self.model_names}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TrafficSignDetectorNode()
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
