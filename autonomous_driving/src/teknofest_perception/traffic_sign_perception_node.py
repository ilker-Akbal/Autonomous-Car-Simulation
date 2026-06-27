from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    import numpy as np
except Exception as exc:  # pragma: no cover - depends on runtime environment
    np = None
    NUMPY_IMPORT_ERROR = exc
else:
    NUMPY_IMPORT_ERROR = None

try:
    import cv2
except Exception as exc:  # pragma: no cover - depends on runtime environment
    cv2 = None
    CV2_IMPORT_ERROR = exc
else:
    CV2_IMPORT_ERROR = None

try:
    from ultralytics import YOLO
except Exception as exc:  # pragma: no cover - depends on runtime environment
    YOLO = None
    YOLO_IMPORT_ERROR = exc
else:
    YOLO_IMPORT_ERROR = None

try:
    import torch
    import torch.nn as nn
    from torchvision import models
except Exception as exc:  # pragma: no cover - depends on runtime environment
    torch = None
    nn = None
    models = None
    TORCH_IMPORT_ERROR = exc
else:
    TORCH_IMPORT_ERROR = None


PACKAGE_NAME = "autonomous_driving"
DETECTOR_MODEL_RELATIVE_PATH = Path("models") / "traffic_sign_detector" / "adas5_sign_yolo_best.pt"
CLASSIFIER_MODEL_RELATIVE_PATH = (
    Path("models") / "traffic_sign_classifier" / "sign_classifier_resnet18_v2_best.pt"
)
DEFAULT_DETECTOR_MODEL_PATH = str(Path(PACKAGE_NAME) / DETECTOR_MODEL_RELATIVE_PATH)
DEFAULT_CLASSIFIER_MODEL_PATH = str(Path(PACKAGE_NAME) / CLASSIFIER_MODEL_RELATIVE_PATH)

CLASS_NAMES = [
    "ada_etrafinda_donunuz",
    "dikkat",
    "dur",
    "girisi_olmayan_yol",
    "hiz_siniri_20",
    "hiz_siniri_30",
    "hiz_siniri_40",
    "hiz_siniri_50",
    "iki_yonlu_yol",
    "ileri_mecburi_yon",
    "ileri_ve_saga_mecburi_yon",
    "ileri_ve_sola_mecburi_yon",
    "isikli_isaret_cihazi",
    "okul_gecidi",
    "park_etmek_yasaktir",
    "park_yeri",
    "saga_donulmez",
    "saga_mecburi_yon",
    "sagdan_gidiniz",
    "sola_donulmez",
    "sola_mecburi_yon",
    "soldan_gidiniz",
    "tunel",
    "yaya_gecidi",
    "yol_calismasi",
    "yol_ver",
]


def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() not in ("false", "0", "no", "off", "none", "n", "f", "")
    return False


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def _bbox_dict(x1: float, y1: float, x2: float, y2: float) -> dict[str, float]:
    return {
        "x_min": float(x1),
        "y_min": float(y1),
        "x_max": float(x2),
        "y_max": float(y2),
    }


def _bbox_list(box: dict[str, float]) -> list[float]:
    return [box["x_min"], box["y_min"], box["x_max"], box["y_max"]]


def _iou(box_a: dict[str, float], box_b: dict[str, float]) -> float:
    ax1, ay1, ax2, ay2 = _bbox_list(box_a)
    bx1, by1, bx2, by2 = _bbox_list(box_b)
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class TrafficSignPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("traffic_sign_perception_node")
        self.bridge = CvBridge()

        self.declare_parameter("detector_model_path", DEFAULT_DETECTOR_MODEL_PATH)
        self.declare_parameter("model_path", DEFAULT_DETECTOR_MODEL_PATH)
        self.declare_parameter("classifier_model_path", DEFAULT_CLASSIFIER_MODEL_PATH)
        self.declare_parameter("classifier_enabled", True)
        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("detections_topic", "/adas/perception/traffic_sign_detections")
        self.declare_parameter("viz_topic", "/adas/perception/traffic_sign_viz")
        self.declare_parameter("detector_conf_threshold", 0.08)
        self.declare_parameter("conf_threshold", 0.08)
        self.declare_parameter("classifier_conf_threshold", 0.50)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("max_detections", 20)
        self.declare_parameter("publish_viz", True)
        self.declare_parameter("publish_rejected_debug", True)
        self.declare_parameter("draw_rejected_detections", True)
        self.declare_parameter("detector_imgsz", 512)
        self.declare_parameter("process_every_n_frames", 4)
        self.declare_parameter("debug_publish_rate_hz", 3.0)
        self.declare_parameter("device", "auto")
        self.declare_parameter("inference_period_s", 0.15)
        self.declare_parameter("log_every_n", 30)
        self.declare_parameter("input_min_width_warn", 640)
        self.declare_parameter("input_min_height_warn", 360)
        self.declare_parameter("min_bbox_width_px", 8.0)
        self.declare_parameter("min_bbox_height_px", 8.0)
        self.declare_parameter("min_bbox_area_px", 80.0)
        self.declare_parameter("min_bbox_area_ratio", 0.00025)
        self.declare_parameter("max_bbox_area_ratio", 0.15)
        self.declare_parameter("min_aspect_ratio", 0.25)
        self.declare_parameter("max_aspect_ratio", 4.0)
        self.declare_parameter("detection_roi_enabled", True)
        self.declare_parameter("full_frame_fallback", False)
        self.declare_parameter("detection_roi_x_min_ratio", 0.45)
        self.declare_parameter("detection_roi_y_min_ratio", 0.10)
        self.declare_parameter("detection_roi_x_max_ratio", 1.00)
        self.declare_parameter("detection_roi_y_max_ratio", 0.90)
        self.declare_parameter("detection_roi_resize_width", 960)
        self.declare_parameter("detection_roi_resize_height", 540)

        detector_path_value = self.get_parameter("detector_model_path").value
        legacy_model_path = self.get_parameter("model_path").value
        if str(legacy_model_path) != DEFAULT_DETECTOR_MODEL_PATH:
            detector_path_value = legacy_model_path

        self.detector_model_path = self._resolve_package_path(
            detector_path_value,
            DEFAULT_DETECTOR_MODEL_PATH,
            DETECTOR_MODEL_RELATIVE_PATH,
        )
        self.classifier_model_path = self._resolve_package_path(
            self.get_parameter("classifier_model_path").value,
            DEFAULT_CLASSIFIER_MODEL_PATH,
            CLASSIFIER_MODEL_RELATIVE_PATH,
        )

        self.classifier_enabled = _parse_bool(self.get_parameter("classifier_enabled").value)
        self.image_topic = str(self.get_parameter("image_topic").value)
        self.detections_topic = str(self.get_parameter("detections_topic").value)
        self.viz_topic = str(self.get_parameter("viz_topic").value)
        self.detector_conf_threshold = _safe_float(self.get_parameter("detector_conf_threshold").value, 0.08)
        self.classifier_conf_threshold = _safe_float(self.get_parameter("classifier_conf_threshold").value, 0.50)
        self.iou_threshold = _safe_float(self.get_parameter("iou_threshold").value, 0.45)
        self.max_detections = max(1, _safe_int(self.get_parameter("max_detections").value, 20))
        self.publish_viz = _parse_bool(self.get_parameter("publish_viz").value)
        self.publish_rejected_debug = _parse_bool(self.get_parameter("publish_rejected_debug").value)
        self.draw_rejected_detections = _parse_bool(self.get_parameter("draw_rejected_detections").value)
        self.detector_imgsz = max(0, _safe_int(self.get_parameter("detector_imgsz").value, 512))
        self.process_every_n_frames = max(1, _safe_int(self.get_parameter("process_every_n_frames").value, 4))
        self.debug_publish_rate_hz = max(0.0, _safe_float(self.get_parameter("debug_publish_rate_hz").value, 3.0))
        self.device = str(self.get_parameter("device").value).strip()
        self.inference_period_s = max(0.0, _safe_float(self.get_parameter("inference_period_s").value, 0.15))
        self.log_every_n = _safe_int(self.get_parameter("log_every_n").value, 30)

        self.input_min_width_warn = max(1, _safe_int(self.get_parameter("input_min_width_warn").value, 640))
        self.input_min_height_warn = max(1, _safe_int(self.get_parameter("input_min_height_warn").value, 360))
        self.min_bbox_width_px = max(0.0, _safe_float(self.get_parameter("min_bbox_width_px").value, 8.0))
        self.min_bbox_height_px = max(0.0, _safe_float(self.get_parameter("min_bbox_height_px").value, 8.0))
        self.min_bbox_area_px = max(0.0, _safe_float(self.get_parameter("min_bbox_area_px").value, 80.0))
        self.min_bbox_area_ratio = max(0.0, _safe_float(self.get_parameter("min_bbox_area_ratio").value, 0.00025))
        self.max_bbox_area_ratio = max(0.0, _safe_float(self.get_parameter("max_bbox_area_ratio").value, 0.15))
        self.min_aspect_ratio = max(0.0, _safe_float(self.get_parameter("min_aspect_ratio").value, 0.25))
        self.max_aspect_ratio = max(self.min_aspect_ratio, _safe_float(self.get_parameter("max_aspect_ratio").value, 4.0))
        self.detection_roi_enabled = _parse_bool(self.get_parameter("detection_roi_enabled").value)
        self.full_frame_fallback = _parse_bool(self.get_parameter("full_frame_fallback").value)
        self.roi_x_min_ratio = _clamp(
            _safe_float(self.get_parameter("detection_roi_x_min_ratio").value, 0.45),
            0.0,
            1.0,
        )
        self.roi_y_min_ratio = _clamp(
            _safe_float(self.get_parameter("detection_roi_y_min_ratio").value, 0.10),
            0.0,
            1.0,
        )
        self.roi_x_max_ratio = _clamp(
            _safe_float(self.get_parameter("detection_roi_x_max_ratio").value, 1.0),
            0.0,
            1.0,
        )
        self.roi_y_max_ratio = _clamp(
            _safe_float(self.get_parameter("detection_roi_y_max_ratio").value, 0.90),
            0.0,
            1.0,
        )
        self.roi_resize_width = max(0, _safe_int(self.get_parameter("detection_roi_resize_width").value, 960))
        self.roi_resize_height = max(0, _safe_int(self.get_parameter("detection_roi_resize_height").value, 540))

        self.classifier_class_names = CLASS_NAMES
        self.classifier_img_size = 224
        if np is not None:
            self.classifier_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            self.classifier_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        else:
            self.classifier_mean = [0.485, 0.456, 0.406]
            self.classifier_std = [0.229, 0.224, 0.225]
        self.classifier_device = self._torch_device()
        self.detector = self._load_detector()
        self.classifier = self._load_classifier()

        self._last_inference_time = 0.0
        self._last_viz_publish_time = 0.0
        self._received_frames = 0
        self._processed_frames = 0

        self.subscription = self.create_subscription(Image, self.image_topic, self._image_callback, 10)
        self.detections_pub = self.create_publisher(String, self.detections_topic, 10)
        self.viz_pub = self.create_publisher(Image, self.viz_topic, 10)

        self.get_logger().info(
            "Traffic sign perception/debug active: "
            f"image_topic={self.image_topic}, detections_topic={self.detections_topic}, "
            f"viz_topic={self.viz_topic}, detector_model_path={self.detector_model_path}, "
            f"classifier_model_path={self.classifier_model_path}, "
            f"detector_conf_threshold={self.detector_conf_threshold}, "
            f"classifier_conf_threshold={self.classifier_conf_threshold}, "
            f"detector_imgsz={self.detector_imgsz}, "
            f"process_every_n_frames={self.process_every_n_frames}, "
            f"debug_publish_rate_hz={self.debug_publish_rate_hz}, "
            f"roi_enabled={self.detection_roi_enabled}, full_frame_fallback={self.full_frame_fallback}"
        )

    def _resolve_package_path(self, raw_value: Any, default_value: str, package_relative: Path) -> Path:
        raw_text = str(raw_value or default_value).strip() or default_value
        raw_path = Path(raw_text).expanduser()
        if raw_path.is_absolute():
            return raw_path.resolve()

        parts = raw_path.parts
        relative_path = Path(*parts[1:]) if parts and parts[0] == PACKAGE_NAME else raw_path
        candidates: list[Path] = []

        def add(path: Path) -> None:
            resolved = path.expanduser().resolve()
            if resolved not in candidates:
                candidates.append(resolved)

        add(Path.cwd() / raw_path)
        for root in self._package_root_candidates():
            add(root / relative_path)
            add(root / package_relative)
            add(root.parent / raw_path)

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else raw_path.resolve()

    def _package_root_candidates(self) -> list[Path]:
        candidates: list[Path] = []

        def add(path: Path) -> None:
            resolved = path.expanduser().resolve()
            if resolved not in candidates:
                candidates.append(resolved)

        here = Path(__file__).resolve()
        for parent in here.parents:
            if parent.name == PACKAGE_NAME and ((parent / "package.xml").exists() or (parent / "models").exists()):
                add(parent)

        try:
            from ament_index_python.packages import get_package_share_directory

            add(Path(get_package_share_directory(PACKAGE_NAME)))
        except Exception:
            pass

        cwd = Path.cwd()
        if (cwd / PACKAGE_NAME).exists():
            add(cwd / PACKAGE_NAME)
        add(cwd)
        return candidates

    def _torch_device(self):
        if torch is None:
            return None

        device_text = (self.device or "auto").strip().lower()
        if device_text == "auto":
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if device_text.isdigit():
            return torch.device(f"cuda:{device_text}" if torch.cuda.is_available() else "cpu")
        if device_text.startswith("cuda") and not torch.cuda.is_available():
            self.get_logger().warn("CUDA requested for classifier but unavailable; using CPU.")
            return torch.device("cpu")
        try:
            return torch.device(device_text)
        except Exception:
            self.get_logger().warn(f"Unsupported classifier device '{self.device}'; using CPU.")
            return torch.device("cpu")

    def _load_detector(self):
        self.detector_error: str | None = None
        if YOLO is None:
            self.detector_error = f"Ultralytics YOLO import failed: {YOLO_IMPORT_ERROR}"
            self.get_logger().error(self.detector_error)
            return None

        if not self.detector_model_path.exists():
            self.detector_error = f"Traffic sign YOLO model file not found: {self.detector_model_path}"
            self.get_logger().error(self.detector_error)
            return None

        try:
            detector = YOLO(str(self.detector_model_path))
        except Exception as exc:
            self.detector_error = f"Traffic sign YOLO model load failed: {exc}"
            self.get_logger().error(self.detector_error)
            return None

        self.get_logger().info(f"Traffic sign detector loaded: {self.detector_model_path}")
        return detector

    def _load_classifier(self):
        self.classifier_error: str | None = None
        if not self.classifier_enabled:
            return None

        if torch is None or nn is None or models is None:
            self.classifier_error = f"Torch/torchvision import failed: {TORCH_IMPORT_ERROR}"
            self.get_logger().error(self.classifier_error)
            return None
        if np is None:
            self.classifier_error = f"Numpy import failed: {NUMPY_IMPORT_ERROR}"
            self.get_logger().error(self.classifier_error)
            return None

        if not self.classifier_model_path.exists():
            self.classifier_error = f"Traffic sign classifier model file not found: {self.classifier_model_path}"
            self.get_logger().error(self.classifier_error)
            return None

        try:
            checkpoint = torch.load(str(self.classifier_model_path), map_location="cpu")
            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
                class_names = checkpoint.get("class_names") or CLASS_NAMES
                img_size = int(checkpoint.get("img_size") or 224)
                normalization = checkpoint.get("normalization") or {}
                mean = normalization.get("mean") or [0.485, 0.456, 0.406]
                std = normalization.get("std") or [0.229, 0.224, 0.225]
            else:
                state_dict = checkpoint
                class_names = CLASS_NAMES
                img_size = 224
                mean = [0.485, 0.456, 0.406]
                std = [0.229, 0.224, 0.225]

            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(0.30),
                nn.Linear(in_features, len(class_names)),
            )
            model.load_state_dict(state_dict)
            self.classifier_class_names = list(class_names)
            self.classifier_img_size = img_size
            self.classifier_mean = np.array(mean, dtype=np.float32)
            self.classifier_std = np.array(std, dtype=np.float32)
            model.to(self._torch_device())
            model.eval()
        except Exception as exc:
            self.classifier_error = f"Traffic sign classifier load failed: {exc}"
            self.get_logger().error(self.classifier_error)
            return None

        self.get_logger().info(f"Traffic sign classifier loaded: {self.classifier_model_path}")
        return model

    def _to_bgr(self, msg: Image) -> np.ndarray | None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        except CvBridgeError as exc:
            self.get_logger().error(f"CV bridge conversion failed: {exc}")
            return None

        encoding = (msg.encoding or "").lower()
        try:
            if encoding in ("bgr8", "bgr"):
                bgr = frame
            elif encoding == "rgb8":
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            elif encoding == "rgba8":
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            elif encoding == "bgra8":
                bgr = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            elif encoding in ("mono8", "8uc1"):
                bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            else:
                try:
                    bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
                except CvBridgeError as exc:
                    self.get_logger().error(f"Unsupported image encoding {msg.encoding}: {exc}")
                    return None
        except cv2.error as exc:
            self.get_logger().error(f"Image encoding conversion failed for {msg.encoding}: {exc}")
            return None

        if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
            self.get_logger().error(f"Unsupported image shape after conversion: {None if bgr is None else bgr.shape}")
            return None

        if bgr.dtype != np.uint8:
            try:
                bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            except CvBridgeError as exc:
                self.get_logger().error(f"Unsupported non-uint8 image encoding {msg.encoding}: {exc}")
                return None

        return np.ascontiguousarray(bgr)

    def _roi_info(self, width: int, height: int) -> dict[str, Any]:
        if self.detection_roi_enabled:
            x_min_ratio = min(self.roi_x_min_ratio, self.roi_x_max_ratio)
            y_min_ratio = min(self.roi_y_min_ratio, self.roi_y_max_ratio)
            x_max_ratio = max(self.roi_x_min_ratio, self.roi_x_max_ratio)
            y_max_ratio = max(self.roi_y_min_ratio, self.roi_y_max_ratio)
        else:
            x_min_ratio = 0.0
            y_min_ratio = 0.0
            x_max_ratio = 1.0
            y_max_ratio = 1.0

        x1 = int(round(width * x_min_ratio))
        y1 = int(round(height * y_min_ratio))
        x2 = int(round(width * x_max_ratio))
        y2 = int(round(height * y_max_ratio))
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        roi_width = max(1, x2 - x1)
        roi_height = max(1, y2 - y1)
        resize_width = self.roi_resize_width or roi_width
        resize_height = self.roi_resize_height or roi_height
        scale_x = resize_width / float(roi_width)
        scale_y = resize_height / float(roi_height)
        roi_upscale_warning = scale_x > 1.01 or scale_y > 1.01
        low_res_input = width < self.input_min_width_warn or height < self.input_min_height_warn

        return {
            "enabled": self.detection_roi_enabled,
            "full_frame_fallback": self.full_frame_fallback,
            "image_px": _bbox_dict(x1, y1, x2, y2),
            "x_min_ratio": x_min_ratio,
            "y_min_ratio": y_min_ratio,
            "x_max_ratio": x_max_ratio,
            "y_max_ratio": y_max_ratio,
            "width_px": roi_width,
            "height_px": roi_height,
            "detection_roi_resize_width": int(resize_width),
            "detection_roi_resize_height": int(resize_height),
            "roi_upscale_factor_x": float(scale_x),
            "roi_upscale_factor_y": float(scale_y),
            "roi_upscale_warning": bool(roi_upscale_warning),
            "low_res_input": bool(low_res_input),
        }

    def _detector_input(self, frame: np.ndarray, roi_info: dict[str, Any]) -> np.ndarray:
        roi_box = roi_info["image_px"]
        x1, y1, x2, y2 = [int(round(value)) for value in _bbox_list(roi_box)]
        crop = frame[y1:y2, x1:x2]
        resize_width = int(roi_info["detection_roi_resize_width"])
        resize_height = int(roi_info["detection_roi_resize_height"])
        if crop.shape[1] == resize_width and crop.shape[0] == resize_height:
            return crop
        return cv2.resize(crop, (resize_width, resize_height), interpolation=cv2.INTER_LINEAR)

    def _run_detector(self, detector_frame: np.ndarray) -> tuple[list[dict[str, Any]], str | None]:
        if self.detector is None:
            return [], self.detector_error or "Traffic sign detector unavailable"

        kwargs: dict[str, Any] = {
            "conf": 0.001,
            "iou": 0.99,
            "max_det": max(self.max_detections * 3, self.max_detections),
            "verbose": False,
        }
        if self.detector_imgsz > 0:
            kwargs["imgsz"] = self.detector_imgsz
        if self.device and self.device.lower() != "auto":
            kwargs["device"] = self.device

        try:
            results = self.detector.predict(source=detector_frame, **kwargs)
        except Exception as exc:
            error = f"Traffic sign detector inference failed: {exc}"
            self.get_logger().error(error)
            return [], error

        if not results:
            return [], None

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], None

        try:
            xyxy = boxes.xyxy.detach().cpu().numpy()
            confidences = boxes.conf.detach().cpu().numpy()
            class_ids = boxes.cls.detach().cpu().numpy().astype(int)
        except Exception as exc:
            error = f"Traffic sign detector output parse failed: {exc}"
            self.get_logger().error(error)
            return [], error

        raw: list[dict[str, Any]] = []
        for idx, (bbox, confidence, class_id) in enumerate(zip(xyxy, confidences, class_ids)):
            class_name = CLASS_NAMES[class_id] if 0 <= int(class_id) < len(CLASS_NAMES) else f"unknown_{int(class_id)}"
            raw.append({
                "raw_index": int(idx),
                "detector_class_id": int(class_id),
                "detector_class_name": class_name,
                "det_confidence": float(confidence),
                "bbox_detector_input_px": _bbox_dict(
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3]),
                ),
            })
        return raw, None

    def _map_raw_to_image(self, raw: list[dict[str, Any]], roi_info: dict[str, Any], image_width: int, image_height: int) -> None:
        roi_box = roi_info["image_px"]
        roi_x1 = roi_box["x_min"]
        roi_y1 = roi_box["y_min"]
        scale_x = max(1e-6, roi_info["roi_upscale_factor_x"])
        scale_y = max(1e-6, roi_info["roi_upscale_factor_y"])

        for detection in raw:
            det_box = detection["bbox_detector_input_px"]
            rx1 = det_box["x_min"] / scale_x
            ry1 = det_box["y_min"] / scale_y
            rx2 = det_box["x_max"] / scale_x
            ry2 = det_box["y_max"] / scale_y
            if rx2 < rx1:
                rx1, rx2 = rx2, rx1
            if ry2 < ry1:
                ry1, ry2 = ry2, ry1

            ix1 = _clamp(roi_x1 + rx1, 0.0, float(image_width))
            iy1 = _clamp(roi_y1 + ry1, 0.0, float(image_height))
            ix2 = _clamp(roi_x1 + rx2, 0.0, float(image_width))
            iy2 = _clamp(roi_y1 + ry2, 0.0, float(image_height))
            if ix2 < ix1:
                ix1, ix2 = ix2, ix1
            if iy2 < iy1:
                iy1, iy2 = iy2, iy1

            width_px = max(0.0, ix2 - ix1)
            height_px = max(0.0, iy2 - iy1)
            area_px = width_px * height_px
            image_area = max(1.0, float(image_width * image_height))
            aspect_ratio = width_px / height_px if height_px > 0.0 else 0.0
            cx = ix1 + width_px * 0.5
            cy = iy1 + height_px * 0.5

            detection.update({
                "bbox_roi_px": _bbox_dict(rx1, ry1, rx2, ry2),
                "bbox_image_px": _bbox_dict(ix1, iy1, ix2, iy2),
                "bbox_xyxy": [ix1, iy1, ix2, iy2],
                "bbox_cxcywh": [cx, cy, width_px, height_px],
                "width_px": float(width_px),
                "height_px": float(height_px),
                "area_px": float(area_px),
                "area_ratio": float(area_px / image_area),
                "aspect_ratio": float(aspect_ratio),
                "effective_min_area_px": float(max(self.min_bbox_area_px, image_area * self.min_bbox_area_ratio)),
                "reject_reasons": [],
                "detector_filter_passed": False,
                "filter_passed": False,
                "classifier_label_raw": None,
                "classifier_confidence": None,
                "final_label": "traffic_sign_unknown",
                "final_confidence": None,
            })

    def _apply_detector_filters(self, raw: list[dict[str, Any]], roi_info: dict[str, Any]) -> None:
        roi_box = roi_info["image_px"]
        for detection in raw:
            reasons = detection["reject_reasons"]
            width_px = detection["width_px"]
            height_px = detection["height_px"]
            area_px = detection["area_px"]
            area_ratio = detection["area_ratio"]
            aspect_ratio = detection["aspect_ratio"]
            bbox = detection["bbox_image_px"]
            cx = bbox["x_min"] + width_px * 0.5
            cy = bbox["y_min"] + height_px * 0.5

            if detection["det_confidence"] < self.detector_conf_threshold:
                reasons.append("det_conf_below_threshold")
            if width_px < self.min_bbox_width_px:
                reasons.append("bbox_too_small_width")
            if height_px < self.min_bbox_height_px:
                reasons.append("bbox_too_small_height")
            if area_px < detection["effective_min_area_px"]:
                reasons.append("bbox_too_small_area")
            if self.max_bbox_area_ratio > 0.0 and area_ratio > self.max_bbox_area_ratio:
                reasons.append("bbox_too_large_area")
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                reasons.append("aspect_ratio_out_of_range")
            if not (
                roi_box["x_min"] <= cx <= roi_box["x_max"]
                and roi_box["y_min"] <= cy <= roi_box["y_max"]
            ):
                reasons.append("outside_roi")

            detection["detector_filter_passed"] = not reasons

        selected: list[dict[str, Any]] = []
        candidates = [item for item in raw if item["detector_filter_passed"]]
        candidates.sort(key=lambda item: item["det_confidence"], reverse=True)
        for detection in candidates:
            if len(selected) >= self.max_detections:
                detection["reject_reasons"].append("max_detections_exceeded")
                detection["detector_filter_passed"] = False
                continue
            if any(_iou(detection["bbox_image_px"], kept["bbox_image_px"]) > self.iou_threshold for kept in selected):
                detection["reject_reasons"].append("nms_suppressed")
                detection["detector_filter_passed"] = False
                continue
            selected.append(detection)

    def _classify_crop(self, frame: np.ndarray, bbox: dict[str, float]) -> dict[str, Any]:
        if not self.classifier_enabled:
            return {"enabled": False, "label": None, "confidence": None, "error": None}
        if self.classifier is None or torch is None:
            return {"enabled": True, "label": None, "confidence": None, "error": self.classifier_error}

        x1, y1, x2, y2 = [int(round(value)) for value in _bbox_list(bbox)]
        x1 = max(0, min(frame.shape[1] - 1, x1))
        y1 = max(0, min(frame.shape[0] - 1, y1))
        x2 = max(x1 + 1, min(frame.shape[1], x2))
        y2 = max(y1 + 1, min(frame.shape[0], y2))
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return {"enabled": True, "label": None, "confidence": None, "error": "empty_classifier_crop"}

        try:
            rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb,
                (self.classifier_img_size, self.classifier_img_size),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            normalized = resized / 255.0
            normalized = (normalized - self.classifier_mean) / self.classifier_std
            tensor = torch.from_numpy(normalized.transpose(2, 0, 1)).unsqueeze(0).float()
            tensor = tensor.to(self.classifier_device)
            with torch.no_grad():
                outputs = self.classifier(tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, class_idx = torch.max(probabilities, dim=1)
            idx = int(class_idx.item())
            label = (
                self.classifier_class_names[idx]
                if 0 <= idx < len(self.classifier_class_names)
                else f"unknown_{idx}"
            )
            return {
                "enabled": True,
                "label": label,
                "class_id": idx,
                "confidence": float(confidence.item()),
                "error": None,
            }
        except Exception as exc:
            error = f"classifier_inference_failed: {exc}"
            self.get_logger().error(error)
            return {"enabled": True, "label": None, "confidence": None, "error": error}

    def _apply_classifier(self, frame: np.ndarray, raw: list[dict[str, Any]], errors: list[str]) -> None:
        for detection in raw:
            if not detection["detector_filter_passed"]:
                continue

            result = self._classify_crop(frame, detection["bbox_image_px"])
            if result["error"]:
                detection["classifier_error"] = result["error"]
                if result["error"] not in errors:
                    errors.append(result["error"])
                detection["final_label"] = detection["detector_class_name"]
                detection["final_confidence"] = detection["det_confidence"]
                detection["filter_passed"] = True
                continue

            if not result["enabled"]:
                detection["final_label"] = detection["detector_class_name"]
                detection["final_confidence"] = detection["det_confidence"]
                detection["filter_passed"] = True
                continue

            detection["classifier_label_raw"] = result["label"]
            detection["classifier_class_id"] = result.get("class_id")
            detection["classifier_confidence"] = result["confidence"]
            if result["confidence"] is None or result["confidence"] < self.classifier_conf_threshold:
                detection["reject_reasons"].append("classifier_low_confidence")
                detection["final_label"] = "traffic_sign_unknown"
                detection["final_confidence"] = result["confidence"]
                detection["filter_passed"] = False
                continue

            detection["final_label"] = result["label"]
            detection["final_confidence"] = result["confidence"]
            detection["filter_passed"] = True

    def _finalize_detection_schema(self, raw: list[dict[str, Any]]) -> None:
        for detection in raw:
            if detection["filter_passed"]:
                final_label = detection["final_label"]
                final_confidence = detection["final_confidence"]
            else:
                final_label = detection.get("final_label") or "traffic_sign_unknown"
                final_confidence = detection.get("final_confidence")

            detection["class_id"] = detection.get("classifier_class_id", detection["detector_class_id"])
            detection["class_name"] = final_label
            detection["confidence"] = (
                float(final_confidence)
                if final_confidence is not None
                else float(detection["det_confidence"])
            )
            detection["filter_passed"] = bool(detection["filter_passed"])

    def _base_payload_from_size(self, msg: Image, width: int, height: int, processing_ms: float = 0.0) -> dict[str, Any]:
        width = max(1, int(width))
        height = max(1, int(height))
        roi_info = self._roi_info(width, height)
        warnings: list[str] = []
        warning = None
        if roi_info["low_res_input"]:
            warnings.append("LOW_RES_INPUT")
        if roi_info["low_res_input"] and roi_info["roi_upscale_warning"]:
            warning = "ROI is upscaled from low resolution input; this does not add real detail"
            warnings.append("ROI_UPSCALED_FROM_LOW_RES_INPUT")
        elif roi_info["roi_upscale_warning"]:
            warnings.append("ROI_UPSCALED")

        return {
            "stamp": {
                "sec": int(msg.header.stamp.sec),
                "nanosec": int(msg.header.stamp.nanosec),
            },
            "frame_id": str(msg.header.frame_id),
            "source": "adas5_sign_yolo",
            "detector_model_path": str(self.detector_model_path),
            "model_path": str(self.detector_model_path),
            "classifier_model_path": str(self.classifier_model_path),
            "image_width": int(width),
            "image_height": int(height),
            "input_resolution": {"width": int(width), "height": int(height)},
            "low_res_input": bool(roi_info["low_res_input"]),
            "input_min_width_warn": int(self.input_min_width_warn),
            "input_min_height_warn": int(self.input_min_height_warn),
            "detection_roi": roi_info,
            "detection_roi_resize_width": roi_info["detection_roi_resize_width"],
            "detection_roi_resize_height": roi_info["detection_roi_resize_height"],
            "roi_upscale_factor_x": roi_info["roi_upscale_factor_x"],
            "roi_upscale_factor_y": roi_info["roi_upscale_factor_y"],
            "roi_upscale_warning": roi_info["roi_upscale_warning"],
            "warning": warning,
            "warnings": warnings,
            "detector_conf_threshold": float(self.detector_conf_threshold),
            "classifier_conf_threshold": float(self.classifier_conf_threshold),
            "iou_threshold": float(self.iou_threshold),
            "detector_imgsz": int(self.detector_imgsz),
            "process_every_n_frames": int(self.process_every_n_frames),
            "debug_publish_rate_hz": float(self.debug_publish_rate_hz),
            "bbox_filter_config": {
                "min_bbox_width_px": float(self.min_bbox_width_px),
                "min_bbox_height_px": float(self.min_bbox_height_px),
                "min_bbox_area_px": float(self.min_bbox_area_px),
                "min_bbox_area_ratio": float(self.min_bbox_area_ratio),
                "max_bbox_area_ratio": float(self.max_bbox_area_ratio),
                "min_aspect_ratio": float(self.min_aspect_ratio),
                "max_aspect_ratio": float(self.max_aspect_ratio),
            },
            "raw_detector_count": 0,
            "filtered_detector_count": 0,
            "rejected_detector_count": 0,
            "final_detection_count": 0,
            "raw_detections": [],
            "rejected_detections": [],
            "accepted_detections": [],
            "detections": [],
            "processing_ms": float(processing_ms),
            "error": None,
        }

    def _base_payload(self, msg: Image, frame: Any, processing_ms: float = 0.0) -> dict[str, Any]:
        height, width = frame.shape[:2]
        return self._base_payload_from_size(msg, width, height, processing_ms)

    def _process_frame(self, msg: Image, frame: np.ndarray, start_time: float) -> tuple[dict[str, Any], dict[str, Any]]:
        height, width = frame.shape[:2]
        roi_info = self._roi_info(width, height)
        detector_frame = self._detector_input(frame, roi_info)
        raw, detector_error = self._run_detector(detector_frame)
        errors: list[str] = []
        if detector_error:
            errors.append(detector_error)

        self._map_raw_to_image(raw, roi_info, width, height)
        self._apply_detector_filters(raw, roi_info)
        self._apply_classifier(frame, raw, errors)
        self._finalize_detection_schema(raw)

        accepted = [item for item in raw if item["filter_passed"]]
        rejected = [item for item in raw if not item["filter_passed"]]
        detector_filtered_count = sum(1 for item in raw if item["detector_filter_passed"])
        processing_ms = (time.monotonic() - start_time) * 1000.0

        payload = self._base_payload(msg, frame, processing_ms)
        payload.update({
            "raw_detector_count": len(raw),
            "filtered_detector_count": detector_filtered_count,
            "rejected_detector_count": len(rejected),
            "final_detection_count": len(accepted),
            "raw_detections": raw,
            "rejected_detections": rejected if self.publish_rejected_debug else [],
            "accepted_detections": accepted,
            "detections": accepted,
            "processing_ms": float(processing_ms),
            "error": "; ".join(errors) if errors else None,
        })

        viz_context = {
            "roi_info": roi_info,
            "raw": raw,
            "accepted": accepted,
            "rejected": rejected,
            "processing_ms": processing_ms,
            "errors": errors,
            "payload": payload,
        }
        return payload, viz_context

    def _publish_detections(self, payload: dict[str, Any]) -> None:
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        self.detections_pub.publish(msg)

    def _draw_label(self, image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
        text = text[:90]
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)

    def _draw_summary(self, image: np.ndarray, payload: dict[str, Any], errors: list[str]) -> None:
        lines = [
            f"input {payload['image_width']}x{payload['image_height']}",
            f"det_thr {self.detector_conf_threshold:.2f} cls_thr {self.classifier_conf_threshold:.2f}",
            (
                f"raw {payload['raw_detector_count']} rejected {payload['rejected_detector_count']} "
                f"final {payload['final_detection_count']}"
            ),
            f"processing_ms {payload['processing_ms']:.1f}",
        ]
        if payload["low_res_input"]:
            lines.append("LOW_RES_INPUT")
        if payload["warning"]:
            lines.append(payload["warning"])
        if errors:
            lines.append(f"ERROR {errors[0]}")

        line_height = 18
        block_height = line_height * len(lines) + 8
        max_width = min(image.shape[1], 760)
        cv2.rectangle(image, (4, 4), (max_width, block_height), (0, 0, 0), -1)
        y = 20
        for line in lines:
            color = (0, 255, 255) if "LOW_RES" in line or "ERROR" in line else (255, 255, 255)
            cv2.putText(image, line[:115], (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            y += line_height

    def _draw_viz(self, frame: np.ndarray, context: dict[str, Any]) -> np.ndarray:
        viz = frame.copy()
        payload = context["payload"]
        roi_box = context["roi_info"]["image_px"]
        rx1, ry1, rx2, ry2 = [int(round(value)) for value in _bbox_list(roi_box)]
        cv2.rectangle(viz, (rx1, ry1), (rx2, ry2), (255, 0, 255), 2)
        self._draw_label(viz, "ROI", rx1 + 4, max(16, ry1 + 16), (255, 0, 255))

        if self.draw_rejected_detections:
            for detection in context["rejected"]:
                x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 80, 255), 1)
                reasons = detection.get("reject_reasons") or ["rejected"]
                reason_text = ",".join(reasons[:2])
                if detection.get("classifier_label_raw") is not None:
                    cls_conf = detection.get("classifier_confidence")
                    cls_conf_text = "nan" if cls_conf is None else f"{cls_conf:.2f}"
                    reason_text = f"UNK raw={detection['classifier_label_raw']}:{cls_conf_text} {reason_text}"
                self._draw_label(viz, reason_text, x1, max(14, y1 - 5), (0, 140, 255))

        for detection in context["accepted"]:
            x1, y1, x2, y2 = [int(round(value)) for value in detection["bbox_xyxy"]]
            label = detection["final_label"]
            confidence = detection.get("final_confidence")
            conf_text = f"{confidence:.2f}" if confidence is not None else f"{detection['det_confidence']:.2f}"
            raw_label = detection.get("classifier_label_raw") or detection.get("detector_class_name")
            cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 220, 0), 2)
            self._draw_label(viz, f"{label} {conf_text} raw={raw_label}", x1, max(14, y1 - 5), (0, 255, 0))

        self._draw_summary(viz, payload, context["errors"])
        return viz

    def _publish_viz(self, frame: np.ndarray, context: dict[str, Any], source_msg: Image) -> None:
        if not self.publish_viz:
            return
        now = time.monotonic()
        if self.debug_publish_rate_hz > 0.0:
            min_period = 1.0 / self.debug_publish_rate_hz
            if now - self._last_viz_publish_time < min_period:
                return
            self._last_viz_publish_time = now

        viz = frame.copy() if cv2 is None else self._draw_viz(frame, context)
        try:
            viz_msg = self.bridge.cv2_to_imgmsg(viz, "bgr8")
        except CvBridgeError as exc:
            self.get_logger().error(f"Traffic sign visualization conversion failed: {exc}")
            return

        viz_msg.header = source_msg.header
        self.viz_pub.publish(viz_msg)

    def _publish_error(self, msg: Image, error: str, frame: np.ndarray | None = None) -> None:
        if frame is None:
            height = max(1, int(msg.height or 360))
            width = max(1, int(msg.width or 640))
            if np is None:
                payload = self._base_payload_from_size(msg, width, height, 0.0)
                payload["error"] = error
                self._publish_detections(payload)
                return
            frame = np.zeros((height, width, 3), dtype=np.uint8)

        payload = self._base_payload(msg, frame, 0.0)
        payload["error"] = error
        context = {
            "roi_info": self._roi_info(frame.shape[1], frame.shape[0]),
            "raw": [],
            "accepted": [],
            "rejected": [],
            "processing_ms": 0.0,
            "errors": [error],
            "payload": payload,
        }
        self._publish_detections(payload)
        self._publish_viz(frame, context, msg)

    def _image_callback(self, msg: Image) -> None:
        self._received_frames += 1
        if self.process_every_n_frames > 1 and self._received_frames % self.process_every_n_frames != 0:
            return

        now = time.monotonic()
        if self.inference_period_s > 0.0 and now - self._last_inference_time < self.inference_period_s:
            return
        self._last_inference_time = now
        start_time = time.monotonic()

        if np is None:
            self._publish_error(msg, f"numpy import failed: {NUMPY_IMPORT_ERROR}")
            return
        if cv2 is None:
            self._publish_error(msg, f"opencv import failed: {CV2_IMPORT_ERROR}")
            return

        frame = self._to_bgr(msg)
        if frame is None:
            self._publish_error(msg, "image_conversion_failed")
            return

        try:
            payload, context = self._process_frame(msg, frame, start_time)
        except Exception as exc:
            error = f"traffic_sign_callback_failed: {exc}"
            self.get_logger().error(error)
            self._publish_error(msg, error, frame)
            return

        self._processed_frames += 1
        self._publish_detections(payload)
        self._publish_viz(frame, context, msg)

        if self.log_every_n > 0 and self._processed_frames % self.log_every_n == 0:
            self.get_logger().info(
                f"Traffic sign debug published: raw={payload['raw_detector_count']}, "
                f"rejected={payload['rejected_detector_count']}, final={payload['final_detection_count']}, "
                f"processing_ms={payload['processing_ms']:.1f}"
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TrafficSignPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
