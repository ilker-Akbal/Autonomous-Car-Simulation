import json
import time
import cv2
import os
import numpy as np

if os.environ.get("ADAS_HEADLESS", "0") == "1":
    cv2.imshow = lambda *args, **kwargs: None
    cv2.waitKey = lambda *args, **kwargs: -1
    cv2.namedWindow = lambda *args, **kwargs: None
    cv2.destroyAllWindows = lambda *args, **kwargs: None

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image as PILImage

import rclpy
from rclpy.qos import qos_profile_sensor_data
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import String
from cv_bridge import CvBridge
from ultralytics import YOLO

from ros2_nodes.perception_tl_pipeline import TrafficLightPipeline


def env_float(name, default):
    try:
        return float(os.environ.get(name, default))
    except Exception:
        return float(default)


def env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except Exception:
        return int(default)


def env_bool(name, default):
    value = os.environ.get(name, str(default)).lower().strip()
    return value in ["1", "true", "yes", "on"]

def resolve_runtime_path(path_value, label="path"):
    """
    ROS launch bazen relative path'i farklı cwd ile geçiriyor.
    Bu helper model/ckpt dosyalarını workspace köküne göre de arar.
    """
    raw = str(path_value or "").strip()
    expanded = os.path.expanduser(raw)

    candidates = []

    if expanded:
        candidates.append(expanded)

    cwd = os.getcwd()

    if expanded and not os.path.isabs(expanded):
        candidates.append(os.path.join(cwd, expanded))
        candidates.append(os.path.join(cwd, "autonomous_driving", expanded))

    # Bu dosya install/site-packages içinden çalışsa bile kaynak workspace'i çoğunlukla cwd.
    # Yine de duplicate path'leri temizleyelim.
    seen = set()
    unique = []
    for c in candidates:
        c = os.path.abspath(c)
        if c not in seen:
            seen.add(c)
            unique.append(c)

    for c in unique:
        if os.path.exists(c):
            return c

    return expanded



class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")

        self.declare_parameter("image_topic", os.environ.get("IMAGE_TOPIC", "/adas/camera/front/image_raw"))
        self.declare_parameter("detections_topic", "/adas/perception/detections_json")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("lidar_topic", "/adas/lidar/points")
        self.declare_parameter("depth_lidar_fusion_enabled", True)
        self.declare_parameter("depth_roi_half_size", 4)
        self.declare_parameter("depth_min_m", 0.2)
        self.declare_parameter("depth_max_m", 80.0)
        self.declare_parameter("lidar_front_min_x", 0.5)
        self.declare_parameter("lidar_front_max_x", 40.0)
        self.declare_parameter("lidar_front_abs_y", 2.0)
        self.declare_parameter("lidar_min_z", -2.0)
        self.declare_parameter("lidar_max_z", 3.0)
        self.declare_parameter("annotated_topic", "/adas/perception/annotated_image")

        self.declare_parameter(
            "model_path",
            os.environ.get(
                "MODEL_PATH",
                "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/outputs/models/adas5_targeted_aug_finetune_from_old_img1024_b8_ep50/weights/best.pt",
            ),
        )

        self.declare_parameter("conf_threshold", env_float("CONF_THRESHOLD", 0.25))
        self.declare_parameter("raw_conf_threshold", env_float("RAW_CONF_THRESHOLD", 0.05))
        self.declare_parameter("person_conf_threshold", env_float("PERSON_CONF_THRESHOLD", 0.40))
        self.declare_parameter("vehicle_conf_threshold", env_float("VEHICLE_CONF_THRESHOLD", 0.25))
        self.declare_parameter("motorcycle_conf_threshold", env_float("MOTORCYCLE_CONF_THRESHOLD", 0.05))
        self.declare_parameter("traffic_light_conf_threshold", env_float("TRAFFIC_LIGHT_CONF_THRESHOLD", 0.50))
        self.declare_parameter("traffic_sign_conf_threshold", env_float("TRAFFIC_SIGN_CONF_THRESHOLD", 0.20))

        # Traffic sign fine-grained classifier V2
        self.declare_parameter(
            "sign_classifier_enabled",
            env_bool("SIGN_CLASSIFIER_ENABLED", True),
        )
        self.declare_parameter(
            "sign_classifier_model_path",
            os.environ.get(
                "SIGN_CLASSIFIER_MODEL_PATH",
                "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier/outputs_v2/sign_classifier_resnet18_v2_best.pt",
            ),
        )
        self.declare_parameter(
            "sign_classifier_conf_threshold",
            env_float("SIGN_CLASSIFIER_CONF_THRESHOLD", 0.45),
        )
        self.declare_parameter(
            "sign_classifier_device",
            os.environ.get("SIGN_CLASSIFIER_DEVICE", "auto"),
        )

        self.declare_parameter("iou_threshold", env_float("YOLO_IOU", 0.50))
        self.declare_parameter("imgsz", env_int("YOLO_IMGSZ", 960))
        self.declare_parameter("max_det", env_int("YOLO_MAX_DET", 80))
        self.declare_parameter("show_debug", env_bool("SHOW_DEBUG", True))

        self.declare_parameter(
            "traffic_light_state_classifier_enabled",
            env_bool("TRAFFIC_LIGHT_STATE_CLASSIFIER_ENABLED", True),
        )
        self.declare_parameter(
            "traffic_light_state_model_path",
            os.environ.get(
                "TRAFFIC_LIGHT_STATE_MODEL_PATH",
                "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/outputs/models/traffic_light_state_resnet18_carla/best.pt",
            ),
        )
        self.declare_parameter(
            "traffic_light_state_conf_threshold",
            env_float("TRAFFIC_LIGHT_STATE_CONF_THRESHOLD", 0.60),
        )
        self.declare_parameter(
            "traffic_light_state_device",
            os.environ.get("TRAFFIC_LIGHT_STATE_DEVICE", "auto"),
        )
        self.declare_parameter(
            "traffic_light_state_use_hsv_fallback",
            env_bool("TRAFFIC_LIGHT_STATE_USE_HSV_FALLBACK", True),
        )

        self.image_topic = self.get_parameter("image_topic").value
        self.detections_topic = self.get_parameter("detections_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.model_path = self.get_parameter("model_path").value

        self.conf_threshold = float(self.get_parameter("conf_threshold").value)
        self.raw_conf_threshold = float(self.get_parameter("raw_conf_threshold").value)
        self.person_conf_threshold = float(self.get_parameter("person_conf_threshold").value)
        self.vehicle_conf_threshold = float(self.get_parameter("vehicle_conf_threshold").value)
        self.motorcycle_conf_threshold = float(self.get_parameter("motorcycle_conf_threshold").value)
        self.traffic_light_conf_threshold = float(self.get_parameter("traffic_light_conf_threshold").value)
        self.traffic_sign_conf_threshold = float(self.get_parameter("traffic_sign_conf_threshold").value)

        self.sign_classifier_enabled = bool(
            self.get_parameter("sign_classifier_enabled").value
        )
        self.sign_classifier_model_path = str(
            self.get_parameter("sign_classifier_model_path").value
        )
        self.sign_classifier_conf_threshold = float(
            self.get_parameter("sign_classifier_conf_threshold").value
        )
        self.sign_classifier_device_name = str(
            self.get_parameter("sign_classifier_device").value
        )

        self.iou_threshold = float(self.get_parameter("iou_threshold").value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.max_det = int(self.get_parameter("max_det").value)
        self.show_debug = bool(self.get_parameter("show_debug").value)

        self.tl_state_classifier_enabled = bool(
            self.get_parameter("traffic_light_state_classifier_enabled").value
        )
        self.tl_state_model_path = str(
            self.get_parameter("traffic_light_state_model_path").value
        )
        self.tl_state_conf_threshold = float(
            self.get_parameter("traffic_light_state_conf_threshold").value
        )
        self.tl_state_device_name = str(
            self.get_parameter("traffic_light_state_device").value
        )
        self.tl_state_use_hsv_fallback = bool(
            self.get_parameter("traffic_light_state_use_hsv_fallback").value
        )

        self.bridge = CvBridge()

        # Inline ZED-depth + LiDAR fusion state
        self.depth_topic = self.get_parameter("depth_topic").value
        self.lidar_topic = self.get_parameter("lidar_topic").value
        self.latest_depth = None
        self.latest_lidar_front_m = None
        self.model_path = resolve_runtime_path(self.model_path, "MODEL_PATH")
        self.tl_state_model_path = resolve_runtime_path(self.tl_state_model_path, "TRAFFIC_LIGHT_STATE_MODEL_PATH")
        self.sign_classifier_model_path = resolve_runtime_path(self.sign_classifier_model_path, "SIGN_CLASSIFIER_MODEL_PATH")

        self.get_logger().info(
            "MODEL_PATH_RESOLVED "
            f"yolo={self.model_path} exists={os.path.exists(self.model_path)} | "
            f"tl={self.tl_state_model_path} exists={os.path.exists(self.tl_state_model_path)} | "
            f"sign={self.sign_classifier_model_path} exists={os.path.exists(self.sign_classifier_model_path)}"
        )

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                "YOLO model bulunamadı. "
                f"model_path={self.model_path} cwd={os.getcwd()} "
                "Çözüm: launch'ta model_path:=autonomous_driving/outputs/models/.../best.pt ver "
                "veya model dosyasını bu path'e koy."
            )

        self.model = YOLO(self.model_path)

        self.tl_state_model = None
        self.tl_state_transform = None
        self.tl_state_class_names = []
        self.tl_state_img_size = 128
        self.tl_state_device = "cpu"
        self.tl_state_ready = False

        # Traffic sign classifier V2 state
        self.sign_model = None
        self.sign_transform = None
        self.sign_class_names = []
        self.sign_img_size = 224
        self.sign_device = "cpu"
        self.sign_ready = False

        self.load_traffic_light_state_classifier()
        self.load_sign_classifier()

        # PERCEPTION_REWRITE_V1:
        # Trafik ışığı kararını tek geçişli, state mutate etmeyen ayrı pipeline verir.
        self.tl_pipeline = TrafficLightPipeline(self)

        self.window_name = "ADAS PERCEPTION DEBUG"

        self.sub = self.create_subscription(Image, self.image_topic, self.image_callback, qos_profile_sensor_data, )

        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            qos_profile_sensor_data,
        )
        self.lidar_sub = self.create_subscription(
            PointCloud2,
            self.lidar_topic,
            self.lidar_callback,
            qos_profile_sensor_data,
        )
        self.get_logger().info(f"Depth/LiDAR fusion subscribers: {self.depth_topic}, {self.lidar_topic}")

        self.det_pub = self.create_publisher(
            String,
            self.detections_topic,
            10,
        )

        self.annotated_pub = self.create_publisher(
            Image,
            self.annotated_topic,
            10,
        )

        if self.show_debug:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, 1400, 800)

        self.get_logger().info("perception_node başladı - MODEL ONLY + TL STATE CLASSIFIER + HSV FALLBACK")
        self.get_logger().info(f"YOLO class names={self.model.names}")
        self.get_logger().info(f"image_topic={self.image_topic}")
        self.get_logger().info(f"detections_topic={self.detections_topic}")
        self.get_logger().info(f"annotated_topic={self.annotated_topic}")
        self.get_logger().info(f"model_path={self.model_path}")
        self.get_logger().info(f"raw_conf_threshold={self.raw_conf_threshold}")
        self.get_logger().info(f"person_conf_threshold={self.person_conf_threshold}")
        self.get_logger().info(f"vehicle_conf_threshold={self.vehicle_conf_threshold}")
        self.get_logger().info(f"motorcycle_conf_threshold={self.motorcycle_conf_threshold}")
        self.get_logger().info(f"traffic_light_conf_threshold={self.traffic_light_conf_threshold}")
        self.get_logger().info(f"traffic_sign_conf_threshold={self.traffic_sign_conf_threshold}")
        self.get_logger().info(f"iou_threshold={self.iou_threshold}")
        self.get_logger().info(f"imgsz={self.imgsz}")
        self.get_logger().info(f"max_det={self.max_det}")
        self.get_logger().info(f"tl_state_classifier_enabled={self.tl_state_classifier_enabled}")
        self.get_logger().info(f"tl_state_ready={self.tl_state_ready}")
        self.get_logger().info(f"tl_state_model_path={self.tl_state_model_path}")
        self.get_logger().info(f"tl_state_conf_threshold={self.tl_state_conf_threshold}")
        self.get_logger().info(f"tl_state_device={self.tl_state_device}")
        self.get_logger().info(f"sign_classifier_enabled={self.sign_classifier_enabled}")
        self.get_logger().info(f"sign_ready={self.sign_ready}")
        self.get_logger().info(f"sign_classifier_model_path={self.sign_classifier_model_path}")
        self.get_logger().info(f"sign_classifier_conf_threshold={self.sign_classifier_conf_threshold}")
        self.get_logger().info(f"sign_classifier_device={self.sign_device}")

    def build_sign_classifier_model(self, num_classes):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.30),
            nn.Linear(in_features, num_classes),
        )
        return model

    def load_sign_classifier(self):
        if not self.sign_classifier_enabled:
            self.get_logger().info("sign classifier disabled")
            return

        if not os.path.exists(self.sign_classifier_model_path):
            self.get_logger().warning(
                f"sign classifier model bulunamadı: {self.sign_classifier_model_path}"
            )
            return

        try:
            if self.sign_classifier_device_name == "auto":
                self.sign_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.sign_device = self.sign_classifier_device_name

            ckpt = torch.load(self.sign_classifier_model_path, map_location=self.sign_device)

            class_names = ckpt.get("class_names", None)
            if not class_names:
                raise RuntimeError("sign checkpoint içinde class_names yok")

            self.sign_class_names = list(class_names)
            self.sign_img_size = int(ckpt.get("img_size", 224))

            model = self.build_sign_classifier_model(len(self.sign_class_names))

            state_dict = (
                ckpt.get("model_state_dict", None)
                or ckpt.get("model_state", None)
                or ckpt.get("state_dict", None)
            )

            if state_dict is None:
                raise RuntimeError("sign checkpoint içinde model_state_dict/model_state yok")

            model.load_state_dict(state_dict)
            model.to(self.sign_device)
            model.eval()

            self.sign_model = model
            self.sign_transform = transforms.Compose([
                transforms.Resize((self.sign_img_size, self.sign_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.sign_ready = True

            self.get_logger().info(
                f"sign classifier loaded: classes={self.sign_class_names}"
            )

        except Exception as exc:
            self.sign_ready = False
            self.sign_model = None
            self.sign_transform = None
            self.get_logger().error(f"sign classifier yüklenemedi: {exc}")

    def classify_traffic_sign_crop(self, frame, bbox):
        if not self.sign_classifier_enabled or not self.sign_ready:
            return {
                "sign_type": "unknown",
                "sign_name": self.pretty_sign_name("unknown"),
                "sign_confidence": 0.0,
                "sign_probs": {},
                "sign_source": "classifier_not_ready",
            }

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if bw < 8 or bh < 8:
            return {
                "sign_type": "unknown",
                "sign_name": self.pretty_sign_name("unknown"),
                "sign_confidence": 0.0,
                "sign_probs": {},
                "sign_source": f"classifier_too_small:{bw}x{bh}",
            }

        pad_x = int(bw * env_float("SIGN_CROP_PAD_X", 0.12))
        pad_y = int(bh * env_float("SIGN_CROP_PAD_Y", 0.12))

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w - 1, x2 + pad_x)
        cy2 = min(frame_h - 1, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return {
                "sign_type": "unknown",
                "sign_name": self.pretty_sign_name("unknown"),
                "sign_confidence": 0.0,
                "sign_probs": {},
                "sign_source": "classifier_empty_crop",
            }

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        x = self.sign_transform(pil_img)
        x = x.unsqueeze(0).to(self.sign_device)

        with torch.no_grad():
            logits = self.sign_model(x)
            prob_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()

        probs = {}
        for i, name in enumerate(self.sign_class_names):
            probs[str(name)] = float(prob_tensor[i].item())

        idx = int(torch.argmax(prob_tensor).item())
        sign_type = str(self.sign_class_names[idx])
        confidence = float(prob_tensor[idx].item())

        if confidence < self.sign_classifier_conf_threshold:
            return {
                "sign_type": "unknown",
                "sign_name": self.pretty_sign_name("unknown"),
                "sign_confidence": confidence,
                "sign_probs": probs,
                "sign_source": f"classifier_low_conf:{sign_type}:{confidence:.3f}",
            }

        return {
            "sign_type": sign_type,
            "sign_name": self.pretty_sign_name(sign_type),
            "sign_confidence": confidence,
            "sign_probs": probs,
            "sign_source": f"classifier:{sign_type}:{confidence:.3f}",
        }


    def build_tl_state_model(self, num_classes):
        model = models.resnet18(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(in_features, num_classes),
        )
        return model

    def load_traffic_light_state_classifier(self):
        if not self.tl_state_classifier_enabled:
            self.get_logger().info("traffic light state classifier disabled")
            return

        if not os.path.exists(self.tl_state_model_path):
            self.get_logger().warning(
                f"traffic light state model bulunamadı, HSV fallback kullanılacak: {self.tl_state_model_path}"
            )
            return

        try:
            if self.tl_state_device_name == "auto":
                self.tl_state_device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.tl_state_device = self.tl_state_device_name

            ckpt = torch.load(self.tl_state_model_path, map_location=self.tl_state_device)

            class_names = ckpt.get("class_names", None)
            if not class_names:
                raise RuntimeError("checkpoint içinde class_names yok")

            self.tl_state_class_names = list(class_names)
            self.tl_state_img_size = int(ckpt.get("img_size", 128))

            model = self.build_tl_state_model(len(self.tl_state_class_names))
            model.load_state_dict(ckpt["model_state"])
            model.to(self.tl_state_device)
            model.eval()

            self.tl_state_model = model
            self.tl_state_transform = transforms.Compose([
                transforms.Resize((self.tl_state_img_size, self.tl_state_img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

            self.tl_state_ready = True

            self.get_logger().info(
                f"traffic light state classifier loaded: classes={self.tl_state_class_names}"
            )

        except Exception as exc:
            self.tl_state_ready = False
            self.tl_state_model = None
            self.tl_state_transform = None
            self.get_logger().error(
                f"traffic light state classifier yüklenemedi, HSV fallback kullanılacak: {exc}"
            )

    def get_yolo_name(self, cls_id):
        names = self.model.names

        if isinstance(names, dict):
            return names.get(cls_id, str(cls_id))

        if isinstance(names, list) and 0 <= cls_id < len(names):
            return names[cls_id]

        return str(cls_id)

    def normalize_text(self, text):
        text = str(text).lower().strip()
        text = text.replace("-", "_")
        text = text.replace(" ", "_")
        return text

    def normalize_label(self, original_label):
        label = self.normalize_text(original_label)

        class_map = {
            "motorcycle": "__drop__",
            "motorbike": "__drop__",
            "bike": "__drop__",

            "pedestrian": "person",
            "person": "person",
            "human": "person",
            "insan": "person",
            "yaya": "person",

            "traffic_light": "traffic_light",
            "trafficlight": "traffic_light",
            "red_light": "traffic_light",
            "yellow_light": "traffic_light",
            "green_light": "traffic_light",
            "traffic_light_red": "traffic_light",
            "traffic_light_yellow": "traffic_light",
            "traffic_light_green": "traffic_light",

            "traffic_sign": "traffic_sign",
            "trafficsign": "traffic_sign",
            "sign": "traffic_sign",

            "vehicle": "vehicle",
            "car": "vehicle",
            "truck": "vehicle",
            "bus": "vehicle",
        }

        return class_map.get(label, label)

    def get_class_threshold(self, label):
        if label == "person":
            return self.person_conf_threshold

        if label == "vehicle":
            return self.vehicle_conf_threshold

        if label == "__drop__":
            return 999.0

        if label == "traffic_light":
            return self.traffic_light_conf_threshold

        if label == "traffic_sign":
            return self.traffic_sign_conf_threshold

        return self.conf_threshold

    def get_sign_type_from_label(self, original_label):
        label = self.normalize_text(original_label)

        sign_classes = {
            "ada_etrafinda_donunuz",
            "dikkat",
            "dur",
            "duraklamak_park_yasaktir",
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
            "tasit_giremez",
            "tunel",
            "yaya_gecidi",
            "yol_calismasi",
            "yol_ver",
        }

        if label in sign_classes:
            return label

        if label == "stop_sign":
            return "dur"

        return "unknown"

    def pretty_sign_name(self, sign_type):
        names = {
            "dikkat": "Dikkat",
            "dur": "Dur",
            "duraklamak_park_yasaktir": "Duraklamak/Park Yasak",
            "girisi_olmayan_yol": "Girişi Olmayan Yol",
            "hiz_siniri_20": "Hız Sınırı 20",
            "hiz_siniri_30": "Hız Sınırı 30",
            "hiz_siniri_40": "Hız Sınırı 40",
            "hiz_siniri_50": "Hız Sınırı 50",
            "isikli_isaret_cihazi": "Işıklı İşaret Cihazı",
            "okul_gecidi": "Okul Geçidi",
            "park_etmek_yasaktir": "Park Yasak",
            "park_yeri": "Park Yeri",
            "saga_donulmez": "Sağa Dönülmez",
            "saga_mecburi_yon": "Sağa Mecburi Yön",
            "sagdan_gidiniz": "Sağdan Gidiniz",
            "sola_donulmez": "Sola Dönülmez",
            "sola_mecburi_yon": "Sola Mecburi Yön",
            "soldan_gidiniz": "Soldan Gidiniz",
            "tasit_giremez": "Taşıt Giremez",
            "tunel": "Tünel",
            "yaya_gecidi": "Yaya Geçidi",
            "yol_calismasi": "Yol Çalışması",
            "yol_ver": "Yol Ver",
            "ada_etrafinda_donunuz": "Ada Etrafında Dönünüz",
            "iki_yonlu_yol": "İki Yönlü Yol",
            "ileri_mecburi_yon": "İleri Mecburi Yön",
            "ileri_ve_saga_mecburi_yon": "İleri ve Sağa Mecburi Yön",
            "ileri_ve_sola_mecburi_yon": "İleri ve Sola Mecburi Yön",
            "unknown": "Bilinmiyor",
        }

        return names.get(sign_type, sign_type)

    def get_component_info(self, mask):
        mask = mask.astype("uint8")
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)

        if num_labels <= 1:
            return {
                "area": 0,
                "x": 0,
                "y": 0,
                "w": 0,
                "h": 0,
                "touches_border": False,
            }

        best = {
            "area": 0,
            "x": 0,
            "y": 0,
            "w": 0,
            "h": 0,
            "touches_border": False,
        }

        H, W = mask.shape[:2]

        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]

            if int(area) > best["area"]:
                touches_border = (
                    x <= 1 or
                    y <= 1 or
                    x + w >= W - 2 or
                    y + h >= H - 2
                )

                best = {
                    "area": int(area),
                    "x": int(x),
                    "y": int(y),
                    "w": int(w),
                    "h": int(h),
                    "touches_border": bool(touches_border),
                }

        return best

    def classify_traffic_light_hsv_strict(self, frame, bbox):
        frame_h, frame_w = frame.shape[:2]

        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)
        area = bw * bh

        scores = {
            "red": 0,
            "yellow": 0,
            "green": 0,
        }

        if bw < 10 or bh < 18 or area < 180:
            return "unknown", scores, f"too_small:bw={bw},bh={bh},area={area}"

        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return "unknown", scores, "empty_crop"

        crop = cv2.resize(crop, (60, 120), interpolation=cv2.INTER_LINEAR)

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        h = hsv[:, :, 0]
        s = hsv[:, :, 1]
        v = hsv[:, :, 2]

        red_mask = (
            (((h >= 0) & (h <= 12)) | ((h >= 168) & (h <= 179))) &
            (s >= 80) &
            (v >= 100)
        )

        yellow_mask = (
            (h >= 18) &
            (h <= 38) &
            (s >= 90) &
            (v >= 130)
        )

        green_mask = (
            (h >= 42) &
            (h <= 95) &
            (s >= 60) &
            (v >= 80)
        )

        H = crop.shape[0]

        top = slice(0, int(H * 0.40))
        mid = slice(int(H * 0.30), int(H * 0.72))
        bot = slice(int(H * 0.60), H)

        red_region = red_mask[top, :]
        yellow_region = yellow_mask[mid, :]
        green_region = green_mask[bot, :]

        red_score = int(red_region.sum())
        yellow_score = int(yellow_region.sum())
        green_score = int(green_region.sum())

        scores = {
            "red": red_score,
            "yellow": yellow_score,
            "green": green_score,
        }

        total = red_score + yellow_score + green_score

        if total < 20:
            return "unknown", scores, f"not_enough_signal:{total}"

        red_comp = self.get_component_info(red_region)
        yellow_comp = self.get_component_info(yellow_region)
        green_comp = self.get_component_info(green_region)

        best_color = max(scores, key=scores.get)
        best_score = scores[best_color]

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        second_score = sorted_scores[1][1]

        if best_score < 18:
            return "unknown", scores, f"best_too_low:{best_score}"

        if second_score > 0 and best_score < second_score * 1.6:
            return "unknown", scores, f"not_dominant:best={best_score},second={second_score}"

        if red_score >= 18 and red_comp["area"] >= 10 and red_score >= yellow_score * 0.45:
            return "red", scores, "red_region_priority"

        if best_color == "yellow":
            if yellow_score < 45:
                return "unknown", scores, f"yellow_weak:{yellow_score}"

            if yellow_comp["area"] < 15:
                return "unknown", scores, f"yellow_component_too_small:{yellow_comp['area']}"

            if yellow_comp["touches_border"]:
                return "unknown", scores, "yellow_touches_border_probably_body_or_backside"

            yellow_region_area = max(1, yellow_region.shape[0] * yellow_region.shape[1])
            yellow_ratio = yellow_comp["area"] / yellow_region_area

            if yellow_ratio > 0.35:
                return "unknown", scores, f"yellow_area_too_large_probably_body:{yellow_ratio:.3f}"

        if best_color == "green":
            if green_score < 20:
                return "unknown", scores, f"green_weak:{green_score}"

            if green_comp["area"] < 10:
                return "unknown", scores, f"green_component_too_small:{green_comp['area']}"

        if best_color == "red":
            if red_score < 18:
                return "unknown", scores, f"red_weak:{red_score}"

            if red_comp["area"] < 10:
                return "unknown", scores, f"red_component_too_small:{red_comp['area']}"

        return best_color, scores, "region_hsv_ok"

    def classify_traffic_light_by_lit_position(self, crop):
        if crop is None or crop.size == 0:
            return {
                "state": "unknown",
                "confidence": 0.0,
                "reason": "empty_crop_position",
                "scores": {},
            }

        try:
            h, w = crop.shape[:2]

            if h < 20 or w < 10:
                return {
                    "state": "unknown",
                    "confidence": 0.0,
                    "reason": f"crop_too_small_position:{w}x{h}",
                    "scores": {},
                }

            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


            mx = max(1, int(w * 0.18))
            my = max(1, int(h * 0.04))
            inner = gray[my:h - my, mx:w - mx]

            if inner.size == 0:
                inner = gray

            ih, iw = inner.shape[:2]


            blur = cv2.GaussianBlur(inner, (3, 3), 0)
            thresh_val = max(120, int(np.percentile(blur, 97)))
            bright = (blur >= thresh_val).astype("uint8")

            bright_count = int(bright.sum())
            if bright_count < 5:
                return {
                    "state": "unknown",
                    "confidence": 0.0,
                    "reason": f"not_enough_bright_pixels:{bright_count}",
                    "scores": {},
                }

            third1 = int(ih / 3)
            third2 = int(2 * ih / 3)

            top_score = int(bright[:third1, :].sum())
            mid_score = int(bright[third1:third2, :].sum())
            bot_score = int(bright[third2:, :].sum())

            scores = {
                "red": top_score,
                "yellow": mid_score,
                "green": bot_score,
            }

            best_state = max(scores, key=scores.get)
            best_score = scores[best_state]
            total = max(1, top_score + mid_score + bot_score)

            ordered = sorted(scores.values(), reverse=True)
            second = ordered[1] if len(ordered) > 1 else 0


            if best_score < 5:
                return {
                    "state": "unknown",
                    "confidence": 0.0,
                    "reason": f"best_position_score_too_low:{best_score}",
                    "scores": scores,
                }

            confidence = float(best_score / total)

            if second > 0 and best_score < second * 1.35:
                return {
                    "state": "unknown",
                    "confidence": confidence,
                    "reason": f"ambiguous_position:best={best_score},second={second},scores={scores}",
                    "scores": scores,
                }

            return {
                "state": best_state,
                "confidence": confidence,
                "reason": f"lit_position_ok:{best_state}:scores={scores}",
                "scores": scores,
            }

        except Exception as exc:
            return {
                "state": "unknown",
                "confidence": 0.0,
                "reason": f"position_error:{exc}",
                "scores": {},
            }

    def save_tl_debug_crop(self, frame, crop, bbox, det_conf=None, cls_result=None, pos_result=None):
            if not env_bool("TL_DEBUG_SAVE", False):
                return

            try:
                save_dir = os.environ.get("TL_DEBUG_DIR", "/tmp/adas_tl_debug")
                os.makedirs(save_dir, exist_ok=True)

                self.tl_debug_count = getattr(self, "tl_debug_count", 0) + 1
                every_n = max(1, env_int("TL_DEBUG_EVERY_N", 1))

                if self.tl_debug_count % every_n != 0:
                    return

                x1, y1, x2, y2 = [int(v) for v in bbox]
                bw = max(1, x2 - x1)
                bh = max(1, y2 - y1)

                ts = int(time.time() * 1000)

                state = "unknown"
                conf = 0.0

                if isinstance(cls_result, dict):
                    state = str(cls_result.get("state", "unknown"))
                    conf = float(cls_result.get("confidence", 0.0))
                elif isinstance(pos_result, dict):
                    state = str(pos_result.get("state", "unknown"))
                    conf = float(pos_result.get("confidence", 0.0))

                filename_base = (
                    f"{ts}_state-{state}_conf-{conf:.3f}_"
                    f"x{x1}_y{y1}_w{bw}_h{bh}"
                )

                crop_path = os.path.join(save_dir, filename_base + "_crop.png")
                cv2.imwrite(crop_path, crop)

                if frame is not None and frame.size > 0:
                    vis = frame.copy()
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(
                        vis,
                        f"{state} {conf:.2f}",
                        (max(0, x1), max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    frame_path = os.path.join(save_dir, filename_base + "_frame.png")
                    cv2.imwrite(frame_path, vis)

                meta = {
                    "time": ts,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_w": bw,
                    "bbox_h": bh,
                    "det_conf": det_conf,
                    "cls_result": cls_result,
                    "pos_result": pos_result,
                }

                meta_path = os.path.join(save_dir, filename_base + "_meta.json")
                with open(meta_path, "w", encoding="utf-8") as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

                self.get_logger().info(
                    f"TL_DEBUG_SAVED crop={crop_path}",
                    throttle_duration_sec=0.5,
                )

            except Exception as exc:
                self.get_logger().warning(
                    f"TL_DEBUG_SAVE_ERROR: {exc}",
                    throttle_duration_sec=1.0,
                )
    def classify_traffic_light_state_model(self, frame, bbox):
        if not self.tl_state_ready or self.tl_state_model is None or self.tl_state_transform is None:
            return None

        frame_h, frame_w = frame.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in bbox]

        x1 = max(0, min(frame_w - 1, x1))
        y1 = max(0, min(frame_h - 1, y1))
        x2 = max(0, min(frame_w - 1, x2))
        y2 = max(0, min(frame_h - 1, y2))

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        if bw < 4 or bh < 4:
            return {
                "state": "unknown",
                "confidence": 0.0,
                "probs": {},
                "reason": f"classifier_too_small:bw={bw},bh={bh}",
            }

        pad_x = int(bw * 0.08)
        pad_y = int(bh * 0.08)

        cx1 = max(0, x1 - pad_x)
        cy1 = max(0, y1 - pad_y)
        cx2 = min(frame_w - 1, x2 + pad_x)
        cy2 = min(frame_h - 1, y2 + pad_y)

        crop = frame[cy1:cy2, cx1:cx2]

        if crop.size == 0:
            return {
                "state": "unknown",
                "confidence": 0.0,
                "probs": {},
                "reason": "classifier_empty_crop",
            }


        if env_bool("SAVE_TL_STATE_CROPS", False):
            try:
                self.tl_crop_save_count = getattr(self, "tl_crop_save_count", 0) + 1

                save_every = max(1, env_int("TL_STATE_CROP_SAVE_EVERY_N", 10))

                if self.tl_crop_save_count % save_every == 0:
                    save_dir = os.environ.get(
                        "TL_STATE_CROP_SAVE_DIR",
                        "/tmp/adas_tl_state_crops",
                    )

                    label = self.normalize_text(
                        os.environ.get("TL_STATE_CROP_LABEL", "unknown")
                    )

                    os.makedirs(save_dir, exist_ok=True)

                    ts = int(time.time() * 1000)

                    crop_path = os.path.join(
                        save_dir,
                        (
                            f"{label}_tl_crop_{ts}_"
                            f"x{int(x1)}_y{int(y1)}_"
                            f"w{int(bw)}_h{int(bh)}.png"
                        ),
                    )

                    ok = cv2.imwrite(crop_path, crop)

                    if ok:
                        self.get_logger().info(
                            f"TL_STATE_CROP_SAVED label={label} path={crop_path}",
                            throttle_duration_sec=1.0,
                        )
                    else:
                        self.get_logger().warning(
                            f"TL_STATE_CROP_SAVE_FAILED path={crop_path}",
                            throttle_duration_sec=1.0,
                        )

            except Exception as exc:
                self.get_logger().warning(
                    f"TL_STATE_CROP_SAVE_ERROR: {exc}",
                    throttle_duration_sec=1.0,
                )

        if env_bool("TL_POSITION_STATE_ENABLED", False):
            pos_result = self.classify_traffic_light_by_lit_position(crop)
            pos_state = pos_result.get("state", "unknown")
            pos_conf = float(pos_result.get("confidence", 0.0))

            if pos_state in ["red", "yellow", "green"] and pos_conf >= env_float("TL_POSITION_STATE_CONF_THRESHOLD", 0.45):
                return {
                    "state": pos_state,
                    "confidence": pos_conf,
                    "probs": {
                        "red": 1.0 if pos_state == "red" else 0.0,
                        "yellow": 1.0 if pos_state == "yellow" else 0.0,
                        "green": 1.0 if pos_state == "green" else 0.0,
                        "unknown": 0.0,
                    },
                    "reason": f"position_classifier:{pos_result.get('reason', '-')}",
                }

        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        pil_img = PILImage.fromarray(rgb)

        x = self.tl_state_transform(pil_img)
        x = x.unsqueeze(0).to(self.tl_state_device)

        with torch.no_grad():
            logits = self.tl_state_model(x)
            prob_tensor = torch.softmax(logits, dim=1)[0].detach().cpu()

        probs = {}
        for i, name in enumerate(self.tl_state_class_names):
            probs[str(name)] = float(prob_tensor[i].item())

        idx = int(torch.argmax(prob_tensor).item())
        state = str(self.tl_state_class_names[idx])
        confidence = float(prob_tensor[idx].item())

        if state not in ["red", "yellow", "green", "unknown"]:
            state = "unknown"

        return {
            "state": state,
            "confidence": confidence,
            "probs": probs,
            "reason": f"classifier_pred:{state}:{confidence:.3f}",
        }

    def is_vehicle(self, label):
        return label == "vehicle"

    def is_person(self, label):
        return label == "person"

    def is_motorcycle(self, label):
        return label == "motorcycle"

    def is_traffic_light(self, label):
        return label == "traffic_light"

    def is_traffic_sign(self, label):
        return label == "traffic_sign"





    def bbox_iou(self, a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

        union = max(1.0, area_a + area_b - inter)
        return float(inter / union)

    def bbox_intersection_over_min_area(self, a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        inter = iw * ih

        area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1.0, (bx2 - bx1) * (by2 - by1))

        return float(inter / max(1.0, min(area_a, area_b)))

    def bbox_center_distance_ratio(self, a, b):
        ax1, ay1, ax2, ay2 = [float(v) for v in a]
        bx1, by1, bx2, by2 = [float(v) for v in b]

        acx = (ax1 + ax2) / 2.0
        acy = (ay1 + ay2) / 2.0
        bcx = (bx1 + bx2) / 2.0
        bcy = (by1 + by2) / 2.0

        dx = acx - bcx
        dy = acy - bcy
        dist = (dx * dx + dy * dy) ** 0.5

        aw = max(1.0, ax2 - ax1)
        ah = max(1.0, ay2 - ay1)
        bw = max(1.0, bx2 - bx1)
        bh = max(1.0, by2 - by1)

        scale = max(1.0, min(max(aw, ah), max(bw, bh)))
        return float(dist / scale)

    def resolve_person_motorcycle_conflicts(self, detections):

        if not env_bool("PERSON_MOTOR_CONFLICT_FILTER", True):
            return detections

        persons = [d for d in detections if d.get("label") == "person"]
        motorcycles = [d for d in detections if d.get("label") == "motorcycle"]

        if not persons or not motorcycles:
            return detections

        iou_thr = env_float("PERSON_MOTOR_IOU_THR", 0.12)
        cover_thr = env_float("PERSON_MOTOR_COVER_THR", 0.45)
        center_thr = env_float("PERSON_MOTOR_CENTER_THR", 0.35)

        filtered = []
        dropped = []

        for det in detections:
            if det.get("label") != "motorcycle":
                filtered.append(det)
                continue

            mbox = det.get("bbox", None)

            if not mbox or len(mbox) != 4:
                filtered.append(det)
                continue

            drop_motor = False
            best_reason = "-"

            for p in persons:
                pbox = p.get("bbox", None)

                if not pbox or len(pbox) != 4:
                    continue

                iou = self.bbox_iou(mbox, pbox)
                cover = self.bbox_intersection_over_min_area(mbox, pbox)
                center = self.bbox_center_distance_ratio(mbox, pbox)

                same_object = (
                    iou >= iou_thr or
                    cover >= cover_thr or
                    center <= center_thr
                )

                if same_object:
                    drop_motor = True
                    best_reason = (
                        f"person_motor_conflict:"
                        f"iou={iou:.3f},cover={cover:.3f},center={center:.3f},"
                        f"person_conf={float(p.get('confidence', 0.0)):.3f},"
                        f"motor_conf={float(det.get('confidence', 0.0)):.3f}"
                    )
                    break

            if drop_motor:
                det["dropped"] = True
                det["drop_reason"] = best_reason
                dropped.append(best_reason)
                continue

            filtered.append(det)

        if dropped:
            self.get_logger().info(
                f"DROP_FALSE_MOTORCYCLE_ON_PERSON count={len(dropped)} reasons={dropped[:3]}",
                throttle_duration_sec=0.5,
            )

        return filtered



    def get_tl_state_confidence(self, det):
        for key in [
            "traffic_light_state_confidence",
            "tl_state_confidence",
            "state_confidence",
            "state_conf",
            "classifier_confidence",
        ]:
            value = det.get(key, None)
            if value is not None:
                try:
                    return float(value)
                except Exception:
                    pass

        probs = det.get("traffic_light_state_probs", None)
        state = det.get("traffic_light_state", "unknown")

        if isinstance(probs, dict) and state in probs:
            try:
                return float(probs[state])
            except Exception:
                pass

        return 0.0


    def draw_all_red_lights_on_screen(self, frame):
        if not env_bool("TL_SHOW_ALL_RED_ON_SCREEN", True):
            return frame

        reds = getattr(self, "screen_red_lights", [])

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.48
        thickness = 2

        def draw_label(text, x, y, fg=(255, 255, 255), bg=(0, 0, 180)):
            text = str(text)
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)

            x1 = max(0, int(x))
            y1 = max(0, int(y - th - 8))
            x2 = min(frame.shape[1] - 1, int(x + tw + 10))
            y2 = min(frame.shape[0] - 1, int(y + 4))

            cv2.rectangle(frame, (x1, y1), (x2, y2), bg, -1)
            cv2.putText(
                frame,
                text,
                (x1 + 5, y),
                font,
                font_scale,
                fg,
                thickness,
                cv2.LINE_AA,
            )

        # Üst-sol ekranda toplam kırmızı ışık listesi.
        draw_label(
            f"ALL RED LIGHTS SEEN: {len(reds)}",
            10,
            28,
            fg=(255, 255, 255),
            bg=(0, 0, 200),
        )

        y = 55
        for i, r in enumerate(reds[:8], start=1):
            x1, y1, x2, y2 = r["bbox"]

            line = (
                f"R{i} det={r['det_conf']:.2f} "
                f"state={r['state_conf']:.2f} "
                f"x={int(x1)} y={int(y1)} "
                f"w={int(r['w'])} h={int(r['h'])}"
            )

            draw_label(
                line,
                10,
                y,
                fg=(255, 255, 255),
                bg=(0, 0, 120),
            )

            y += 25

            x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

            cv2.rectangle(
                frame,
                (x1i, y1i),
                (x2i, y2i),
                (0, 0, 255),
                3,
            )

            tag = f"RED#{i} {r['det_conf']:.2f}/{r['state_conf']:.2f}"

            tx = max(0, x1i)
            ty = max(18, y1i - 8)

            draw_label(
                tag,
                tx,
                ty,
                fg=(255, 255, 255),
                bg=(0, 0, 220),
            )

        if len(reds) > 8:
            draw_label(
                f"+{len(reds) - 8} more red lights",
                10,
                y,
                fg=(255, 255, 255),
                bg=(0, 0, 120),
            )

        return frame


    def _tl_env_float(self, name, default):
        try:
            return float(env_float(name, default))
        except Exception:
            return float(default)

    def _tl_env_bool(self, name, default):
        try:
            return bool(env_bool(name, default))
        except Exception:
            v = str(os.environ.get(name, str(int(default)))).lower().strip()
            return v in ("1", "true", "yes", "on")

    def draw_detection(self, frame, det):
        # PERCEPTION_REWRITE_V1:
        # Draw fonksiyonu detection state değiştirmez. Trafik ışığı kararı sadece
        # TrafficLightPipeline.process() içinde bir kez verilir.
        x1, y1, x2, y2 = [int(v) for v in det["bbox"]]

        label = det.get("label", "unknown")
        original_label = det.get("original_label", label)
        conf = float(det.get("confidence", 0.0))

        if self.is_person(label):
            color = (255, 0, 0)
            text = f"person/{original_label} {conf:.2f}"

        elif self.is_vehicle(label):
            color = (0, 255, 0)
            text = f"vehicle/{original_label} {conf:.2f}"

        elif self.is_motorcycle(label):
            color = (255, 0, 255)
            text = f"motorcycle/{original_label} {conf:.2f}"

        elif self.is_traffic_light(label):
            state = det.get("traffic_light_state", "unknown")
            state_conf = det.get("traffic_light_state_confidence", None)

            if state == "red":
                color = (0, 0, 255)
            elif state == "yellow":
                color = (0, 255, 255)
            elif state == "green":
                color = (0, 255, 0)
            else:
                color = (180, 180, 180)

            if state_conf is not None:
                text = f"traffic_light {state} {conf:.2f}/{float(state_conf):.2f}"
            else:
                text = f"traffic_light {state} {conf:.2f}"

        elif self.is_traffic_sign(label):
            color = (0, 255, 255)
            sign_type = det.get("sign_type", "unknown")
            sign_name = self.pretty_sign_name(sign_type)

            if sign_type != "unknown":
                text = f"LEVHA: {sign_name} {conf:.2f}"
            else:
                text = f"traffic_sign/{original_label} {conf:.2f}"

        else:
            color = (255, 255, 255)
            text = f"{label}/{original_label} {conf:.2f}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.50
        font_thickness = 2

        (tw, th), _ = cv2.getTextSize(text, font, font_scale, font_thickness)

        tx1 = max(0, x1)
        ty1 = max(0, y1 - th - 8)
        tx2 = min(frame.shape[1] - 1, x1 + tw + 8)
        ty2 = max(0, y1)

        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), color, -1)

        cv2.putText(
            frame,
            text,
            (x1 + 4, max(15, y1 - 6)),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness,
            cv2.LINE_AA,
        )

    def draw_text(self, frame, text, x, y, color=(255, 255, 255), scale=0.55, thickness=2):
        cv2.putText(
            frame,
            str(text),
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    def make_debug_canvas(self, frame, detections):
        h, w = frame.shape[:2]
        panel_w = 430

        canvas = cv2.copyMakeBorder(
            frame,
            0,
            0,
            0,
            panel_w,
            cv2.BORDER_CONSTANT,
            value=(45, 45, 45),
        )

        px = w + 20

        vehicles = len([d for d in detections if d.get("label") == "vehicle"])
        persons = len([d for d in detections if d.get("label") == "person"])
        motorcycles = 0
        traffic_lights = len([d for d in detections if d.get("label") == "traffic_light"])
        traffic_signs = len([d for d in detections if d.get("label") == "traffic_sign"])

        others = len([
            d for d in detections
            if d.get("label") not in [
                "vehicle",
                "person",
                "motorcycle",
                "traffic_light",
                "traffic_sign",
            ]
        ])

        light_states = [
            d.get("traffic_light_state", "unknown")
            for d in detections
            if d.get("label") == "traffic_light"
        ]

        sign_names = [
            self.pretty_sign_name(d.get("sign_type", "unknown"))
            for d in detections
            if d.get("label") == "traffic_sign"
        ]

        light_text = ",".join(light_states[:5]) if light_states else "-"
        sign_text = ",".join(sign_names[:5]) if sign_names else "-"

        self.draw_text(canvas, "ADAS PERCEPTION", px, 35, (255, 255, 255), 0.70, 2)
        self.draw_text(canvas, "YOLO + TL STATE CLASSIFIER", px, 65, (0, 255, 255), 0.48, 2)

        self.draw_text(canvas, f"Model imgsz   : {self.imgsz}", px, 105)
        self.draw_text(canvas, f"Raw Conf      : {self.raw_conf_threshold:.3f}", px, 132)
        self.draw_text(canvas, f"Person Conf   : {self.person_conf_threshold:.3f}", px, 159)
        self.draw_text(canvas, f"Vehicle Conf  : {self.vehicle_conf_threshold:.2f}", px, 186)
        self.draw_text(canvas, f"Motor Conf    : {self.motorcycle_conf_threshold:.2f}", px, 213)
        self.draw_text(canvas, f"Light Conf    : {self.traffic_light_conf_threshold:.2f}", px, 240)
        self.draw_text(canvas, f"TL State Thr  : {self.tl_state_conf_threshold:.2f}", px, 267)
        self.draw_text(canvas, f"TL Cls Ready  : {self.tl_state_ready}", px, 294)

        cv2.line(canvas, (w + 15, 320), (w + panel_w - 15, 320), (100, 100, 100), 1)

        self.draw_text(canvas, "DETECTIONS", px, 355, (200, 200, 200), 0.60, 2)
        self.draw_text(canvas, f"Vehicles      : {vehicles}", px, 392)
        self.draw_text(canvas, f"Persons       : {persons}", px, 419)
        self.draw_text(canvas, "Motorcycles   : disabled", px, 446)
        self.draw_text(canvas, f"TrafficLight  : {traffic_lights}", px, 473)
        self.draw_text(canvas, f"TrafficSign   : {traffic_signs}", px, 500)
        self.draw_text(canvas, f"Others        : {others}", px, 527)

        cv2.line(canvas, (w + 15, 555), (w + panel_w - 15, 555), (100, 100, 100), 1)

        self.draw_text(canvas, "MODEL OUTPUT", px, 590, (200, 200, 200), 0.60, 2)
        self.draw_text(canvas, f"Light State   : {light_text}", px, 625)
        self.draw_text(canvas, f"Signs         : {sign_text}", px, 655, (255, 255, 255), 0.45, 1)

        y = 690
        for det in detections[:7]:
            label = det.get("label", "-")
            original = det.get("original_label", "-")
            conf = float(det.get("confidence", 0.0))

            if label == "traffic_light":
                state = det.get("traffic_light_state", "unknown")
                source = det.get("traffic_light_state_source", "-")
                state_conf = det.get("traffic_light_state_confidence", None)
                hsv_state = det.get("traffic_light_hsv_state", "-")
                probs = det.get("traffic_light_state_probs", {})
                reason = det.get("traffic_light_color_reason", "-")

                state_conf_text = "-" if state_conf is None else f"{float(state_conf):.2f}"
                p_red = float(probs.get("red", 0.0))
                p_yellow = float(probs.get("yellow", 0.0))
                p_green = float(probs.get("green", 0.0))
                p_unknown = float(probs.get("unknown", 0.0))

                line = f"TL {state} det={conf:.2f} cls={state_conf_text} src={source[:12]}"
                self.draw_text(canvas, line, px, y, (255, 255, 255), 0.32, 1)
                y += 18

                line2 = f"P R{p_red:.2f} Y{p_yellow:.2f} G{p_green:.2f} U{p_unknown:.2f} HSV={hsv_state}"
                self.draw_text(canvas, line2, px, y, (220, 220, 220), 0.29, 1)
                y += 18

                self.draw_text(canvas, f"reason: {str(reason)[:36]}", px, y, (220, 220, 220), 0.28, 1)
                y += 22

            else:
                line = f"{label}/{original} {conf:.2f}"
                self.draw_text(canvas, line, px, y, (255, 255, 255), 0.36, 1)
                y += 22

        return canvas


    # ==========================================================
    # Inline depth + LiDAR fusion helpers
    # ==========================================================

    def depth_callback(self, msg):
        if not bool(self.get_parameter("depth_lidar_fusion_enabled").value):
            return

        if msg.encoding != "32FC1":
            self.get_logger().warn_once(f"Depth encoding 32FC1 değil: {msg.encoding}")
            return

        try:
            depth = np.frombuffer(msg.data, dtype=np.float32).reshape((msg.height, msg.width))
            self.latest_depth = depth.copy()
        except Exception as e:
            self.get_logger().warn(f"Depth parse hata: {e}")

    def lidar_callback(self, msg):
        if not bool(self.get_parameter("depth_lidar_fusion_enabled").value):
            return

        try:
            if len(msg.data) == 0 or msg.point_step < 12:
                self.latest_lidar_front_m = None
                return

            cols = max(3, msg.point_step // 4)
            arr = np.frombuffer(msg.data, dtype=np.float32)

            usable = (arr.size // cols) * cols
            if usable <= 0:
                self.latest_lidar_front_m = None
                return

            arr = arr[:usable].reshape((-1, cols))
            xyz = arr[:, :3]

            x = xyz[:, 0]
            y = xyz[:, 1]
            z = xyz[:, 2]

            valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)

            min_x = float(self.get_parameter("lidar_front_min_x").value)
            max_x = float(self.get_parameter("lidar_front_max_x").value)
            abs_y = float(self.get_parameter("lidar_front_abs_y").value)
            min_z = float(self.get_parameter("lidar_min_z").value)
            max_z = float(self.get_parameter("lidar_max_z").value)

            front_mask = (
                valid
                & (x >= min_x)
                & (x <= max_x)
                & (np.abs(y) <= abs_y)
                & (z >= min_z)
                & (z <= max_z)
            )

            front_x = x[front_mask]

            if front_x.size == 0:
                self.latest_lidar_front_m = None
                return

            self.latest_lidar_front_m = float(np.percentile(front_x, 5.0))

        except Exception as e:
            self.get_logger().warn(f"LiDAR parse hata: {e}")
            self.latest_lidar_front_m = None

    def _extract_detections_list_for_fusion(self, payload):
        if isinstance(payload, list):
            return payload

        if not isinstance(payload, dict):
            return None

        for key in ("detections", "objects", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value

        return None

    def _extract_image_size_for_fusion(self, payload):
        if not isinstance(payload, dict):
            return None, None

        pairs = (
            ("image_width", "image_height"),
            ("frame_width", "frame_height"),
            ("width", "height"),
        )

        for wk, hk in pairs:
            if wk in payload and hk in payload:
                try:
                    return int(payload[wk]), int(payload[hk])
                except Exception:
                    pass

        shape = payload.get("frame_shape") or payload.get("image_shape")
        if isinstance(shape, list) and len(shape) >= 2:
            try:
                return int(shape[1]), int(shape[0])
            except Exception:
                pass

        return None, None

    def _extract_bbox_for_fusion(self, det):
        if not isinstance(det, dict):
            return None

        bbox = det.get("bbox") or det.get("box")
        if isinstance(bbox, list) and len(bbox) >= 4:
            try:
                return [float(v) for v in bbox[:4]]
            except Exception:
                return None

        keys = ("x1", "y1", "x2", "y2")
        if all(k in det for k in keys):
            try:
                return [float(det[k]) for k in keys]
            except Exception:
                return None

        return None

    def _depth_distance_for_bbox(self, bbox, image_w=None, image_h=None):
        if self.latest_depth is None:
            return None

        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        except Exception:
            return None

        depth = self.latest_depth
        dh, dw = depth.shape

        if image_w and image_h and image_w > 0 and image_h > 0:
            sx = dw / float(image_w)
            sy = dh / float(image_h)
            x1 *= sx
            x2 *= sx
            y1 *= sy
            y2 *= sy

        cx = int(round((x1 + x2) / 2.0))
        cy = int(round((y1 + y2) / 2.0))

        half = int(self.get_parameter("depth_roi_half_size").value)

        x0 = max(0, cx - half)
        x3 = min(dw, cx + half + 1)
        y0 = max(0, cy - half)
        y3 = min(dh, cy + half + 1)

        if x0 >= x3 or y0 >= y3:
            return None

        roi = depth[y0:y3, x0:x3]

        min_m = float(self.get_parameter("depth_min_m").value)
        max_m = float(self.get_parameter("depth_max_m").value)

        valid = roi[np.isfinite(roi)]
        valid = valid[(valid >= min_m) & (valid <= max_m)]

        if valid.size == 0:
            return None

        return float(np.median(valid))

    def apply_depth_lidar_fusion_to_payload(self, payload):
        if not bool(self.get_parameter("depth_lidar_fusion_enabled").value):
            return payload

        detections = self._extract_detections_list_for_fusion(payload)
        image_w, image_h = self._extract_image_size_for_fusion(payload)

        if isinstance(payload, dict):
            payload["fusion_enabled"] = True
            payload["fusion_sources"] = {
                "zed_depth": self.latest_depth is not None,
                "lidar": self.latest_lidar_front_m is not None,
            }
            payload["front_lidar_obstacle_m"] = (
                round(float(self.latest_lidar_front_m), 3)
                if self.latest_lidar_front_m is not None
                else None
            )

        if detections is None:
            return payload

        for det in detections:
            if not isinstance(det, dict):
                continue

            bbox = self._extract_bbox_for_fusion(det)
            depth_m = None

            if bbox is not None:
                depth_m = self._depth_distance_for_bbox(bbox, image_w, image_h)

            if depth_m is not None:
                det["distance_m"] = round(float(depth_m), 3)
                det["distance_est"] = round(float(depth_m), 3)
                det["distance_source"] = "zed_depth"
            else:
                det["distance_source"] = det.get("distance_source", "none")

            if self.latest_lidar_front_m is not None:
                det["front_lidar_obstacle_m"] = round(float(self.latest_lidar_front_m), 3)

        return payload


    def image_callback(self, msg):
        """
        PERCEPTION_REWRITE_V1 ana akış.

        Eski JSON sözleşmesini korur ama trafik ışığını tek bir pipeline üzerinden işler.
        Tabela, araç, yaya, depth/LiDAR ve annotated yayınları korunur.
        """
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge hata: {exc}")
            return

        frame_h, frame_w = frame.shape[:2]
        annotated = frame.copy()
        detections = []

        try:
            results = self.model.predict(
                source=frame,
                conf=self.raw_conf_threshold,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                max_det=self.max_det,
                verbose=False,
            )
        except Exception as exc:
            self.get_logger().error(f"YOLO predict hata: {exc}")
            return

        if len(results) > 0:
            result = results[0]

            if result.boxes is not None:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    x1, y1, x2, y2 = box.xyxy[0].tolist()

                    original_label = self.get_yolo_name(cls_id)
                    label = self.normalize_label(original_label)

                    if label == "__drop__":
                        continue

                    class_threshold = self.get_class_threshold(label)

                    if conf < class_threshold:
                        self.get_logger().info(
                            f"FILTERED_LOW_CONF "
                            f"class_id={cls_id} "
                            f"original={original_label} "
                            f"mapped_label={label} "
                            f"conf={conf:.3f} "
                            f"required={class_threshold:.3f}",
                            throttle_duration_sec=0.5,
                        )
                        continue

                    bbox_w = max(1.0, float(x2 - x1))
                    bbox_h = max(1.0, float(y2 - y1))
                    area_ratio = (bbox_w * bbox_h) / float(frame_w * frame_h)

                    det = {
                        "class_id": cls_id,
                        "label": label,
                        "original_label": original_label,
                        "confidence": conf,
                        "bbox": [
                            float(x1),
                            float(y1),
                            float(x2),
                            float(y2),
                        ],
                        "center_x": float((x1 + x2) / 2.0),
                        "center_y": float((y1 + y2) / 2.0),
                        "bbox_width": float(bbox_w),
                        "bbox_height": float(bbox_h),
                        "area_ratio": float(area_ratio),
                        "source": "model_only_yolo",
                    }

                    if label == "traffic_sign":
                        sign_type_from_label = self.get_sign_type_from_label(original_label)
                        sign_result = self.classify_traffic_sign_crop(frame, det["bbox"])

                        cls_sign_type = sign_result.get("sign_type", "unknown")
                        cls_sign_conf = float(sign_result.get("sign_confidence", 0.0))

                        if cls_sign_type != "unknown":
                            sign_type = cls_sign_type
                            sign_conf = cls_sign_conf
                            sign_source = sign_result.get("sign_source", "classifier")
                        else:
                            sign_type = sign_type_from_label
                            sign_conf = conf if sign_type != "unknown" else cls_sign_conf
                            sign_source = sign_result.get("sign_source", "classifier_unknown")

                        det["sign_type"] = sign_type
                        det["sign_name"] = self.pretty_sign_name(sign_type)
                        det["sign_confidence"] = float(sign_conf)
                        det["sign_probs"] = sign_result.get("sign_probs", {})
                        det["sign_source"] = sign_source

                        self.get_logger().info(
                            f"SIGN_CLASSIFIER "
                            f"yolo_conf={conf:.3f} "
                            f"original={original_label} "
                            f"type={det.get('sign_type')} "
                            f"name={det.get('sign_name')} "
                            f"sign_conf={det.get('sign_confidence'):.3f} "
                            f"source={det.get('sign_source')} "
                            f"bbox={[round(float(v), 1) for v in det['bbox']]}",
                            throttle_duration_sec=0.3,
                        )

                    detections.append(det)

                    model_name = (
                        self.model.names.get(cls_id, str(cls_id))
                        if isinstance(self.model.names, dict)
                        else self.model.names[cls_id]
                    )

                    self.get_logger().info(
                        f"MODEL_DET "
                        f"class_id={cls_id} "
                        f"model_name={model_name} "
                        f"original={original_label} "
                        f"mapped_label={label} "
                        f"conf={conf:.3f} "
                        f"bbox={[round(float(v), 1) for v in det['bbox']]}",
                        throttle_duration_sec=0.3,
                    )

        detections = self.resolve_person_motorcycle_conflicts(detections)

        detections, tl_info, tl_candidates, tl_rejected = self.tl_pipeline.process(
            frame,
            detections,
            frame_w,
            frame_h,
        )

        self.screen_red_lights = []

        if tl_rejected:
            self.get_logger().info(
                "TL_PIPELINE_REJECTED "
                f"count={len(tl_rejected)} "
                f"reasons={[d.get('tl_pipeline_reject_reason', '-') for d in tl_rejected[:5]]}",
                throttle_duration_sec=0.5,
            )

        if tl_info.get("state", "unknown") != "unknown":
            bbox = tl_info.get("bbox") or []
            self.get_logger().info(
                "ACTIVE_TRAFFIC_LIGHT_V2 "
                f"state={tl_info.get('state')} "
                f"det_conf={float(tl_info.get('confidence') or 0.0):.3f} "
                f"state_conf={float(tl_info.get('state_confidence') or 0.0):.3f} "
                f"source={tl_info.get('state_source')} "
                f"roi={tl_info.get('roi')} "
                f"score={tl_info.get('score')} "
                f"bbox={[round(float(v), 1) for v in bbox]} "
                f"reason={str(tl_info.get('color_reason'))[:120]}",
                throttle_duration_sec=0.3,
            )

        for det in detections:
            self.draw_detection(annotated, det)

        self.draw_all_red_lights_on_screen(annotated)
        debug_canvas = self.make_debug_canvas(annotated, detections)

        payload = {
            "stamp": time.time(),
            "image_width": frame_w,
            "image_height": frame_h,
            "model_path": self.model_path,
            "model_only": True,
            "raw_conf_threshold": self.raw_conf_threshold,
            "person_conf_threshold": self.person_conf_threshold,
            "vehicle_conf_threshold": self.vehicle_conf_threshold,
            "motorcycle_conf_threshold": self.motorcycle_conf_threshold,
            "traffic_light_conf_threshold": self.traffic_light_conf_threshold,
            "traffic_sign_conf_threshold": self.traffic_sign_conf_threshold,
            "iou_threshold": self.iou_threshold,
            "imgsz": self.imgsz,
            "max_det": self.max_det,

            "traffic_light_state_classifier_enabled": self.tl_state_classifier_enabled,
            "traffic_light_state_classifier_ready": self.tl_state_ready,
            "traffic_light_state_model_path": self.tl_state_model_path,
            "traffic_light_state_conf_threshold": self.tl_state_conf_threshold,

            "detections": detections,

            "traffic_light_state": tl_info.get("state", "unknown"),
            "traffic_light_confidence": tl_info.get("confidence"),
            "traffic_light_state_confidence": tl_info.get("state_confidence"),
            "traffic_light_state_source": tl_info.get("state_source"),
            "traffic_light_state_probs": tl_info.get("state_probs"),
            "traffic_light_color_scores": tl_info.get("color_scores"),
            "traffic_light_color_reason": tl_info.get("color_reason"),
            "traffic_light_active_bbox": tl_info.get("bbox"),
            "traffic_light_candidate_count": tl_info.get("candidate_count"),
            "traffic_light_rejected_count": tl_info.get("rejected_count"),
            "perception_rewrite_v1": True,
        }

        msg_out = String()
        payload = self.apply_depth_lidar_fusion_to_payload(payload)
        msg_out.data = json.dumps(payload, ensure_ascii=False)
        self.det_pub.publish(msg_out)

        try:
            annotated_msg = self.bridge.cv2_to_imgmsg(debug_canvas, encoding="bgr8")
            annotated_msg.header = msg.header
            self.annotated_pub.publish(annotated_msg)
        except Exception as exc:
            self.get_logger().error(f"annotated image publish hata: {exc}")

        if self.show_debug:
            cv2.imshow(self.window_name, debug_canvas)
            cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show_debug:
            cv2.destroyAllWindows()

        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
