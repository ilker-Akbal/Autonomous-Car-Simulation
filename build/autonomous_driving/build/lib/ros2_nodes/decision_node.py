import json
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.behavior_contracts import normalize_decision_command
from teknofest_sim.traffic_light_state_machine import TrafficLightStoplineStateMachine


class DecisionNode(Node):
    """
    TEKNOFEST/CARLA için kural tabanlı karar katmanı.

    Amaç:
    - Gereksiz false STOP azaltmak.
    - Yaya sadece ego şeridi/ön güvenlik koridorundaysa STOP.
    - Sağ/sol kaldırım yayası aracı yolun ortasında kilitlemesin.
    - Öndeki araç/engel için erken SLOW/STOP.
    - Trafik ışığı, levha, hız sınırı ve kişi/araç öncelikleri net olsun.
    """

    def __init__(self):
        super().__init__("decision_node")

        self.declare_parameter("detections_topic", "/adas/perception/detections_json")
        self.declare_parameter("decision_events_topic", "/adas/perception/decision_events_json")
        self.declare_parameter("route_intent_topic", "/adas/planning/route_intent")
        self.declare_parameter("decision_topic", "/adas/decision")
        self.declare_parameter("decision_command_topic", "/adas/decision/command")
        self.declare_parameter("decision_debug_topic", "/adas/decision/debug")

        # Mesafe yaklaşımı: bbox yüksekliğine göre basit tahmin.
        self.declare_parameter("distance_k", 1200.0)

        # Daha güvenli durma/yavaşlama mesafeleri.
        self.declare_parameter("stop_distance", 10.0)
        self.declare_parameter("slow_distance", 18.0)

        # Hızlar.
        self.declare_parameter("default_go_speed", 0.8)
        self.declare_parameter("slow_speed", 0.35)
        self.declare_parameter("creep_speed", 0.20)
        self.declare_parameter("stop_speed", 0.0)

        # İnsan/yarışma tarafı km/h kullanır.
        # Eski m/s parametreleri geriye uyumluluk için durur.
        # *_kmh >= 0 verilirse km/h değeri esas alınır ve içte m/s'ye çevrilir.
        self.declare_parameter("default_go_speed_kmh", -1.0)
        self.declare_parameter("slow_speed_kmh", -1.0)
        self.declare_parameter("creep_speed_kmh", -1.0)
        self.declare_parameter("stop_speed_kmh", -1.0)

        # Algı güven eşikleri.
        self.declare_parameter("vehicle_conf_threshold", 0.25)
        self.declare_parameter("person_conf_threshold", 0.30)
        self.declare_parameter("traffic_light_conf_threshold", 0.40)
        self.declare_parameter("traffic_light_state_conf_threshold", 0.55)
        self.declare_parameter("traffic_sign_conf_threshold", 0.20)
        self.declare_parameter("sign_classifier_conf_threshold", 0.25)

        # Levha seçim filtresi:
        # Çok uzaktaki/minik levha yanlış karar üretmesin.
        self.declare_parameter("sign_min_bottom_ratio", 0.18)
        self.declare_parameter("sign_min_area_ratio", 0.00020)
        self.declare_parameter("sign_center_bonus", 0.20)

        # DUR levhası sonsuza kadar aracı kilitlemesin.
        # Görüldüğünde kısa süre durur, sonra cooldown süresince aynı tip STOP tekrar tetiklenmez.
        self.declare_parameter("stop_sign_hold_seconds", 2.0)
        self.declare_parameter("stop_sign_cooldown_seconds", 7.0)

        # Ego şeridi / sürüş koridoru.
        # Yakın bölgede koridor daha geniş, uzak bölgede daha dar.
        self.declare_parameter("path_x_center", 0.50)
        self.declare_parameter("path_half_width_top", 0.14)
        self.declare_parameter("path_half_width_bottom", 0.34)
        self.declare_parameter("path_y_top", 0.30)
        self.declare_parameter("path_y_bottom", 1.00)

        # Yaya kuralları.
        self.declare_parameter("person_stop_bottom_ratio", 0.58)
        self.declare_parameter("person_slow_bottom_ratio", 0.42)
        self.declare_parameter("person_stop_hold_seconds", 1.2)
        self.declare_parameter("person_release_seconds", 0.8)

        # Öndeki araç/engel filtreleri.
        self.declare_parameter("min_vehicle_bbox_height_ratio", 0.05)
        self.declare_parameter("min_vehicle_bbox_width_ratio", 0.04)
        self.declare_parameter("min_vehicle_area_ratio", 0.0025)
        self.declare_parameter("max_missing_front", 5)

        # Trafik ışığı aktif ROI.
        self.declare_parameter("tl_active_roi_x_min", 0.32)
        self.declare_parameter("tl_active_roi_x_max", 0.88)
        self.declare_parameter("tl_active_roi_y_min", 0.05)
        self.declare_parameter("tl_active_roi_y_max", 0.58)
        # Trafik ışığı çok uzaktayken/minik bbox iken STOP üretmesin.
        # Yeşil ise serbest bırakmak için kabul edilebilir; kırmızı/sarı için yakınlık şartı var.
        self.declare_parameter("tl_stop_min_bottom_ratio", 0.28)
        self.declare_parameter("tl_stop_min_height_ratio", 0.035)
        self.declare_parameter("tl_stop_min_area_ratio", 0.00015)
        self.declare_parameter("tl_green_release_bypass_hold", True)
        self.declare_parameter("tl_prefer_semantic_event", True)

        # KIRMIZI 1. ÖNCELİK:
        # Kırmızı bir kere görüldüyse unknown frame'lerde GO'ya düşme.
        # Net yeşil gelene kadar veya latch süresi dolana kadar STOP kal.
        self.declare_parameter("red_light_latch_enabled", True)
        self.declare_parameter("red_light_latch_seconds", 3.0)
        self.declare_parameter("red_light_min_green_release_count", 1)
        self.declare_parameter("yellow_as_slow", True)
        self.declare_parameter("tl_green_release_min_det_conf", 0.12)
        self.declare_parameter("tl_green_release_min_state_conf", 0.60)
        self.declare_parameter("tl_green_release_min_area_ratio", 0.00030)
        self.declare_parameter("tl_green_release_min_height_ratio", 0.035)
        self.declare_parameter("tl_green_release_max_distance_m", 60.0)
        self.declare_parameter("tl_green_release_min_count", 2)
        self.declare_parameter("tl_red_stop_distance_m", 12.0)
        self.declare_parameter("tl_red_crawl_distance_m", 18.0)
        self.declare_parameter("tl_red_slow_distance_m", 24.0)
        self.declare_parameter("tl_red_stop_offset_m", 3.0)
        self.declare_parameter("tl_red_no_distance_stop_bottom_ratio", 0.54)
        self.declare_parameter("tl_red_no_distance_stop_height_ratio", 0.045)
        self.declare_parameter("tl_red_no_distance_stop_area_ratio", 0.00045)
        self.declare_parameter("tl_red_distance_memory_seconds", 6.0)
        self.declare_parameter("tl_red_distance_outlier_jump_m", 18.0)
        self.declare_parameter("tl_red_distance_far_outlier_m", 55.0)
        self.declare_parameter("tl_red_far_approach_speed_kmh", 8.0)
        self.declare_parameter("tl_red_memory_decay_mps", 4.2)
        self.declare_parameter("tl_red_raw_stall_epsilon_m", 0.70)
        # Trafik ışığı final kararı decision_node içindedir.
        self.declare_parameter("handle_traffic_lights", True)
        self.declare_parameter("tl_stop_before_line_m", 1.0)
        self.declare_parameter("tl_brake_decel_mps2", 3.0)
        self.declare_parameter("tl_reaction_time_s", 0.75)
        self.declare_parameter("tl_brake_safety_margin_m", 2.0)
        self.declare_parameter("tl_green_confirm_count", 1)
        self.declare_parameter("tl_red_confirm_count", 1)
        self.declare_parameter("tl_yellow_confirm_count", 1)
        self.declare_parameter("tl_unknown_precaution_distance_m", 18.0)
        self.declare_parameter("tl_passed_ignore_s", 2.0)

        # Karar stabilizasyonu.
        self.declare_parameter("stop_hold_seconds", 0.8)
        self.declare_parameter("slow_hold_seconds", 0.5)
        self.declare_parameter("decision_timeout_seconds", 1.5)
        self.declare_parameter("decision_events_timeout_s", 1.5)
        self.declare_parameter("route_intent_timeout_s", 1.5)
        self.declare_parameter("route_constraint_wait_stop", False)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.decision_events_topic = self.get_parameter("decision_events_topic").value
        self.route_intent_topic = self.get_parameter("route_intent_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value
        self.decision_command_topic = self.get_parameter("decision_command_topic").value
        self.decision_debug_topic = self.get_parameter("decision_debug_topic").value

        self.distance_k = float(self.get_parameter("distance_k").value)
        self.stop_distance = float(self.get_parameter("stop_distance").value)
        self.slow_distance = float(self.get_parameter("slow_distance").value)

        self.default_go_speed = float(self.get_parameter("default_go_speed").value)
        self.slow_speed = float(self.get_parameter("slow_speed").value)
        self.creep_speed = float(self.get_parameter("creep_speed").value)
        self.stop_speed = float(self.get_parameter("stop_speed").value)

        self.default_go_speed_kmh = float(self.get_parameter("default_go_speed_kmh").value)
        self.slow_speed_kmh = float(self.get_parameter("slow_speed_kmh").value)
        self.creep_speed_kmh = float(self.get_parameter("creep_speed_kmh").value)
        self.stop_speed_kmh = float(self.get_parameter("stop_speed_kmh").value)

        if self.default_go_speed_kmh >= 0.0:
            self.default_go_speed = self.default_go_speed_kmh / 3.6
        if self.slow_speed_kmh >= 0.0:
            self.slow_speed = self.slow_speed_kmh / 3.6
        if self.creep_speed_kmh >= 0.0:
            self.creep_speed = self.creep_speed_kmh / 3.6
        if self.stop_speed_kmh >= 0.0:
            self.stop_speed = self.stop_speed_kmh / 3.6

        self.vehicle_conf_threshold = float(self.get_parameter("vehicle_conf_threshold").value)
        self.person_conf_threshold = float(self.get_parameter("person_conf_threshold").value)
        self.traffic_light_conf_threshold = float(self.get_parameter("traffic_light_conf_threshold").value)
        self.traffic_light_state_conf_threshold = float(
            self.get_parameter("traffic_light_state_conf_threshold").value
        )
        self.traffic_sign_conf_threshold = float(self.get_parameter("traffic_sign_conf_threshold").value)
        self.sign_classifier_conf_threshold = float(self.get_parameter("sign_classifier_conf_threshold").value)

        self.sign_min_bottom_ratio = float(self.get_parameter("sign_min_bottom_ratio").value)
        self.sign_min_area_ratio = float(self.get_parameter("sign_min_area_ratio").value)
        self.sign_center_bonus = float(self.get_parameter("sign_center_bonus").value)
        self.stop_sign_hold_seconds = float(self.get_parameter("stop_sign_hold_seconds").value)
        self.stop_sign_cooldown_seconds = float(self.get_parameter("stop_sign_cooldown_seconds").value)



        self.path_x_center = float(self.get_parameter("path_x_center").value)
        self.path_half_width_top = float(self.get_parameter("path_half_width_top").value)
        self.path_half_width_bottom = float(self.get_parameter("path_half_width_bottom").value)
        self.path_y_top = float(self.get_parameter("path_y_top").value)
        self.path_y_bottom = float(self.get_parameter("path_y_bottom").value)

        self.person_stop_bottom_ratio = float(self.get_parameter("person_stop_bottom_ratio").value)
        self.person_slow_bottom_ratio = float(self.get_parameter("person_slow_bottom_ratio").value)
        self.person_stop_hold_seconds = float(self.get_parameter("person_stop_hold_seconds").value)
        self.person_release_seconds = float(self.get_parameter("person_release_seconds").value)

        self.min_vehicle_bbox_height_ratio = float(
            self.get_parameter("min_vehicle_bbox_height_ratio").value
        )
        self.min_vehicle_bbox_width_ratio = float(
            self.get_parameter("min_vehicle_bbox_width_ratio").value
        )
        self.min_vehicle_area_ratio = float(self.get_parameter("min_vehicle_area_ratio").value)
        self.max_missing_front = int(self.get_parameter("max_missing_front").value)

        self.tl_active_roi_x_min = float(self.get_parameter("tl_active_roi_x_min").value)
        self.tl_active_roi_x_max = float(self.get_parameter("tl_active_roi_x_max").value)
        self.tl_active_roi_y_min = float(self.get_parameter("tl_active_roi_y_min").value)
        self.tl_active_roi_y_max = float(self.get_parameter("tl_active_roi_y_max").value)
        self.tl_stop_min_bottom_ratio = float(self.get_parameter("tl_stop_min_bottom_ratio").value)
        self.tl_stop_min_height_ratio = float(self.get_parameter("tl_stop_min_height_ratio").value)
        self.tl_stop_min_area_ratio = float(self.get_parameter("tl_stop_min_area_ratio").value)
        self.tl_green_release_bypass_hold = bool(self.get_parameter("tl_green_release_bypass_hold").value)
        self.tl_prefer_semantic_event = bool(self.get_parameter("tl_prefer_semantic_event").value)
        self.red_light_latch_enabled = bool(self.get_parameter("red_light_latch_enabled").value)
        self.red_light_latch_seconds = float(self.get_parameter("red_light_latch_seconds").value)
        self.red_light_min_green_release_count = int(
            self.get_parameter("red_light_min_green_release_count").value
        )
        self.yellow_as_slow = bool(self.get_parameter("yellow_as_slow").value)
        self.tl_green_release_min_det_conf = float(self.get_parameter("tl_green_release_min_det_conf").value)
        self.tl_green_release_min_state_conf = float(self.get_parameter("tl_green_release_min_state_conf").value)
        self.tl_green_release_min_area_ratio = float(self.get_parameter("tl_green_release_min_area_ratio").value)
        self.tl_green_release_min_height_ratio = float(self.get_parameter("tl_green_release_min_height_ratio").value)
        self.tl_green_release_max_distance_m = float(self.get_parameter("tl_green_release_max_distance_m").value)
        self.tl_green_release_min_count = int(self.get_parameter("tl_green_release_min_count").value)
        self._qualified_green_count = 0
        self.tl_red_stop_distance_m = float(self.get_parameter("tl_red_stop_distance_m").value)
        self.tl_red_crawl_distance_m = float(self.get_parameter("tl_red_crawl_distance_m").value)
        self.tl_red_slow_distance_m = float(self.get_parameter("tl_red_slow_distance_m").value)
        self.tl_red_stop_offset_m = float(self.get_parameter("tl_red_stop_offset_m").value)
        self.tl_red_no_distance_stop_bottom_ratio = float(
            self.get_parameter("tl_red_no_distance_stop_bottom_ratio").value
        )
        self.tl_red_no_distance_stop_height_ratio = float(
            self.get_parameter("tl_red_no_distance_stop_height_ratio").value
        )
        self.tl_red_no_distance_stop_area_ratio = float(
            self.get_parameter("tl_red_no_distance_stop_area_ratio").value
        )
        self.tl_red_distance_memory_seconds = float(
            self.get_parameter("tl_red_distance_memory_seconds").value
        )
        self.tl_red_distance_outlier_jump_m = float(
            self.get_parameter("tl_red_distance_outlier_jump_m").value
        )
        self.tl_red_distance_far_outlier_m = float(
            self.get_parameter("tl_red_distance_far_outlier_m").value
        )
        self.tl_red_far_approach_speed = self.kmh_to_mps(
            float(self.get_parameter("tl_red_far_approach_speed_kmh").value)
        )
        self.tl_red_memory_decay_mps = float(
            self.get_parameter("tl_red_memory_decay_mps").value
        )
        self.tl_red_raw_stall_epsilon_m = float(
            self.get_parameter("tl_red_raw_stall_epsilon_m").value
        )
        self.handle_traffic_lights = bool(self.get_parameter("handle_traffic_lights").value)
        self.tl_stop_before_line_m = float(self.get_parameter("tl_stop_before_line_m").value)
        self.tl_brake_decel_mps2 = float(self.get_parameter("tl_brake_decel_mps2").value)
        self.tl_reaction_time_s = float(self.get_parameter("tl_reaction_time_s").value)
        self.tl_brake_safety_margin_m = float(self.get_parameter("tl_brake_safety_margin_m").value)
        self.tl_green_confirm_count = int(self.get_parameter("tl_green_confirm_count").value)
        self.tl_red_confirm_count = int(self.get_parameter("tl_red_confirm_count").value)
        self.tl_yellow_confirm_count = int(self.get_parameter("tl_yellow_confirm_count").value)
        self.tl_unknown_precaution_distance_m = float(self.get_parameter("tl_unknown_precaution_distance_m").value)
        self.tl_passed_ignore_s = float(self.get_parameter("tl_passed_ignore_s").value)

        self.stop_hold_seconds = float(self.get_parameter("stop_hold_seconds").value)
        self.slow_hold_seconds = float(self.get_parameter("slow_hold_seconds").value)
        self.decision_timeout_seconds = float(self.get_parameter("decision_timeout_seconds").value)
        self.decision_events_timeout_s = float(self.get_parameter("decision_events_timeout_s").value)
        self.route_intent_timeout_s = float(self.get_parameter("route_intent_timeout_s").value)
        self.route_constraint_wait_stop = bool(self.get_parameter("route_constraint_wait_stop").value)

        self.vehicle_labels = {
            "vehicle",
            "car",
            "truck",
            "bus",
            "van",
            "suv",
            "motorcycle",
        }

        self.stop_signs = {
            "dur",
        }

        self.slow_signs = {
            "yaya_gecidi",
            "okul_gecidi",
            "yol_calismasi",
            "dikkat",
            "yol_ver",
        }

        self.warning_signs = set()

        # Hız sınırı levhaları insan standardında km/h tutulur.
        # İç fizik/BasicAgent tarafına giderken m/s'ye çevrilir.
        self.speed_limit_kmh_map = {
            "hiz_siniri_20": 20.0,
            "hiz_siniri_30": 30.0,
            "hiz_siniri_40": 40.0,
            "hiz_siniri_50": 50.0,
        }
        self.speed_limit_map = {
            key: self.kmh_to_mps(value_kmh)
            for key, value_kmh in self.speed_limit_kmh_map.items()
        }

        self.last_front_vehicle = None
        self.missing_front_count = 0

        self.last_speed_limit_sign = None
        self.current_speed_limit = None

        # Stop-sign temporal state
        self.stop_sign_hold_until = 0.0
        self.stop_sign_cooldown_until = 0.0
        self.last_stop_sign_type = None

        self.last_near_person_time = 0.0
        self.last_near_person = None

        self.last_output = None
        self.last_output_time = 0.0

        self.latest_decision_events = []
        self.last_decision_events_time = 0.0
        self.latest_route_intent = None
        self.last_route_intent_time = 0.0

        # Trafik ışığı kırmızı hafızası.
        self.red_light_latch_until = 0.0
        self.red_light_latch_info = None
        self.red_light_green_seen_count = 0
        self.red_light_first_seen_time = 0.0
        self.last_red_distance_m = None
        self.last_red_distance_time = 0.0
        self.last_red_distance_source = None
        self.tl_sm_state = "CLEAR"
        self.tl_sm_stopline_id = None
        self.tl_sm_ignore_until = 0.0
        self.tl_sm_last_green_time = 0.0
        self.tl_sm_counts = {"red": 0, "yellow": 0, "green": 0, "unknown": 0}
        self.tl_hold_active = False
        self.tl_hold_stopline_id = None
        self.tl_hold_since = 0.0
        self.tl_hold_reason = None
        self.last_non_green_light_state = "unknown"
        self.last_confirmed_green_time = 0.0
        self.last_tl_active_light_marker = "TL_ACTIVE_LIGHT_REJECTED:not_evaluated"

        self.pub = self.create_publisher(String, self.decision_topic, 10)
        self.command_pub = self.create_publisher(String, self.decision_command_topic, 10)
        self.debug_pub = self.create_publisher(String, self.decision_debug_topic, 10)
        self.sub = self.create_subscription(
            String,
            self.detections_topic,
            self.callback,
            10,
        )
        self.create_subscription(
            String,
            self.decision_events_topic,
            self.decision_events_cb,
            10,
        )
        self.create_subscription(
            String,
            self.route_intent_topic,
            self.route_intent_cb,
            10,
        )

        self.get_logger().info(
            f"decision_node başladı: detections={self.detections_topic} "
            f"decision_events={self.decision_events_topic} "
            f"route_intent={self.route_intent_topic} -> {self.decision_topic}, {self.decision_command_topic}"
        )


    def decision_events_cb(self, msg):
        try:
            data = json.loads(msg.data)
            events = data.get("events", [])
            if not isinstance(events, list):
                events = []
            self.latest_decision_events = events
            self.last_decision_events_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"decision events parse hatası: {exc}")

    def route_intent_cb(self, msg):
        try:
            self.latest_route_intent = json.loads(msg.data)
            self.last_route_intent_time = time.time()
        except Exception as exc:
            self.get_logger().warning(f"route intent parse hatası: {exc}")

    def get_fresh_decision_events(self):
        if not self.latest_decision_events:
            return []
        if time.time() - self.last_decision_events_time > self.decision_events_timeout_s:
            return []
        return list(self.latest_decision_events)

    def get_fresh_route_intent(self):
        if self.latest_route_intent is None:
            return None
        if time.time() - self.last_route_intent_time > self.route_intent_timeout_s:
            return None
        return dict(self.latest_route_intent)

    def route_intent_requires_wait(self, route_intent):
        if not isinstance(route_intent, dict):
            return False, "route_intent_missing"

        status = str(route_intent.get("route_decision_status", "clear")).lower().strip()
        reason = str(route_intent.get("route_decision_reason", status or "clear"))

        if status == "blocked":
            return True, f"ROUTE_CONSTRAINT_BLOCKED:{reason}"

        if status in {"violation", "violation_warning"}:
            if self.route_constraint_wait_stop:
                return True, reason
            return False, f"route_constraint_warning_only:{reason}"

        return False, status or "clear"

    def clamp(self, value, mn, mx):
        return max(mn, min(float(value), mx))

    def kmh_to_mps(self, value):
        try:
            return float(value) / 3.6
        except Exception:
            return 0.0

    def mps_to_kmh(self, value):
        try:
            return float(value) * 3.6
        except Exception:
            return 0.0

    def bbox_metrics(self, det, frame_width, frame_height):
        bbox = det.get("bbox", None)
        if bbox is None or len(bbox) != 4:
            return None

        x1, y1, x2, y2 = map(float, bbox)
        if x2 <= x1 or y2 <= y1:
            return None

        w = x2 - x1
        h = y2 - y1
        area = w * h
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        bottom_y = y2

        return {
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "w": w,
            "h": h,
            "area": area,
            "cx": cx,
            "cy": cy,
            "bottom_y": bottom_y,
            "cx_ratio": cx / max(1.0, float(frame_width)),
            "cy_ratio": cy / max(1.0, float(frame_height)),
            "bottom_ratio": bottom_y / max(1.0, float(frame_height)),
            "w_ratio": w / max(1.0, float(frame_width)),
            "h_ratio": h / max(1.0, float(frame_height)),
            "area_ratio": area / max(1.0, float(frame_width * frame_height)),
        }

    def path_half_width_at_y(self, y_ratio):
        if y_ratio <= self.path_y_top:
            return self.path_half_width_top

        if y_ratio >= self.path_y_bottom:
            return self.path_half_width_bottom

        t = (y_ratio - self.path_y_top) / max(1e-6, self.path_y_bottom - self.path_y_top)
        return self.path_half_width_top + t * (
            self.path_half_width_bottom - self.path_half_width_top
        )

    def is_in_ego_path(self, metrics):
        y = metrics["bottom_ratio"]
        x = metrics["cx_ratio"]

        if y < self.path_y_top:
            return False

        half_width = self.path_half_width_at_y(y)
        return abs(x - self.path_x_center) <= half_width

    def estimate_distance(self, bbox_height):
        if bbox_height <= 1:
            return None
        return self.distance_k / float(bbox_height)

    def get_state_confidence(self, det):
        keys = [
            "traffic_light_state_confidence",
            "tl_state_confidence",
            "state_confidence",
            "state_conf",
            "classifier_confidence",
        ]

        for key in keys:
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

    def is_vehicle_candidate(self, det, frame_width, frame_height):
        label = str(det.get("label", "")).strip()
        conf = float(det.get("confidence", 0.0))

        if label not in self.vehicle_labels:
            return False

        if conf < self.vehicle_conf_threshold:
            return False

        m = self.bbox_metrics(det, frame_width, frame_height)
        if m is None:
            return False

        # Çok küçük araç bbox'ları genelde uzaktaki/yan şeritteki araçtır.
        if m["h_ratio"] < self.min_vehicle_bbox_height_ratio:
            return False

        if m["w_ratio"] < self.min_vehicle_bbox_width_ratio:
            return False

        if m["area_ratio"] < self.min_vehicle_area_ratio:
            return False

        # Kritik düzeltme:
        # Görüntü sağ/sol kenarına yapışan büyük bbox'lar çoğunlukla yan araç,
        # kesilmiş obje veya ego/spectator kaynaklı false front_vehicle oluyor.
        edge_margin_px = 3.0
        touches_left = m["x1"] <= edge_margin_px
        touches_right = m["x2"] >= float(frame_width) - edge_margin_px
        touches_edge = touches_left or touches_right

        if touches_edge and (m["w_ratio"] > 0.10 or m["area_ratio"] > 0.010):
            return False

        # Çok yakında ama şerit merkezinden belirgin uzaksa ön araç değil, yan obje say.
        center_error = abs(m["cx_ratio"] - self.path_x_center)

        if m["bottom_ratio"] >= 0.78 and center_error > 0.20:
            return False

        # Ön araç için koridoru eskiye göre daraltıyoruz.
        # Önceden bottom bölgede çok genişti; sağ kaldırım/yan araç da front sanılıyordu.
        dynamic_half_width = self.path_half_width_at_y(m["bottom_ratio"])
        strict_half_width = min(dynamic_half_width, 0.22)

        if center_error > strict_half_width:
            return False

        # Araç görüntünün en altına değiyor ve genişse bu çoğu zaman gerçek ön araçtan çok
        # yan/kesilmiş araç parçası oluyor. Merkezdeyse izin ver, kenardaysa reddet.
        if m["bottom_ratio"] >= 0.96 and m["w_ratio"] > 0.22 and center_error > 0.12:
            return False

        return True

    def select_front_vehicle(self, detections, frame_width, frame_height):
        candidates = []

        for det in detections:
            if not self.is_vehicle_candidate(det, frame_width, frame_height):
                continue

            m = self.bbox_metrics(det, frame_width, frame_height)
            distance_est = self.estimate_distance(m["h"])
            if distance_est is None:
                continue

            cx_error = abs(m["cx_ratio"] - self.path_x_center)
            half_width = self.path_half_width_at_y(m["bottom_ratio"])
            center_score = 1.0 - min(1.0, cx_error / max(half_width, 1e-6))

            selected = dict(det)
            selected.update({
                "center_x": m["cx"],
                "bbox_width": m["w"],
                "bbox_height": m["h"],
                "area_ratio": m["area_ratio"],
                "bottom_ratio": m["bottom_ratio"],
                "distance_est": distance_est,
                "in_ego_path": True,
            })

            score = (
                float(det.get("confidence", 0.0)) * 2.0
                + m["area_ratio"] * 20.0
                + m["bottom_ratio"] * 2.0
                + center_score * 3.0
                - min(distance_est, 50.0) * 0.02
            )

            selected["front_score"] = score
            candidates.append((score, selected))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def apply_front_vehicle_memory(self, front_vehicle):
        used_memory = False

        if front_vehicle is not None:
            self.last_front_vehicle = front_vehicle
            self.missing_front_count = 0
            return front_vehicle, used_memory

        self.missing_front_count += 1

        if self.last_front_vehicle is not None and self.missing_front_count <= self.max_missing_front:
            used_memory = True
            remembered = dict(self.last_front_vehicle)
            remembered["used_memory"] = True
            return remembered, used_memory

        self.last_front_vehicle = None
        return None, used_memory

    def evaluate_vehicle_rule(self, front_vehicle, used_memory):
        if front_vehicle is None:
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "distance_est": None,
                "reason": "front_vehicle_not_found",
            }

        distance_est = float(front_vehicle["distance_est"])

        if distance_est <= self.stop_distance:
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "distance_est": distance_est,
                "reason": "front_vehicle_too_close_memory" if used_memory else "front_vehicle_too_close",
            }

        if distance_est <= self.slow_distance:
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "distance_est": distance_est,
                "reason": "front_vehicle_near_memory" if used_memory else "front_vehicle_near",
            }

        return {
            "decision": "GO",
            "risk": "LOW",
            "target_speed": self.default_go_speed,
            "distance_est": distance_est,
            "reason": "front_vehicle_safe",
        }


    def traffic_light_geometry_ok(self, item, frame_width, frame_height):
        """
        Kırmızı/sarı ışık için mesafe filtresi.
        Çok küçük/uzak ışıklar STOP üretirse araç kavşaktan gereksiz uzakta kalıyor.
        Yeşil ışık release için serbest bırakılır; kırmızı/sarı ise yakınlık şartı ister.
        """
        state = str(
            item.get("traffic_light_state", item.get("state", "unknown"))
        ).lower().strip()

        if state not in {"red", "yellow"}:
            return True, "green_or_unknown_no_stop_geometry_gate"

        bbox = item.get("bbox")
        m = self.bbox_metrics({"bbox": bbox}, frame_width, frame_height)
        if m is None:
            return False, "tl_stop_reject_no_bbox"

        if m["bottom_ratio"] < self.tl_stop_min_bottom_ratio:
            return False, f"tl_stop_reject_far_bottom:{m['bottom_ratio']:.3f}"

        if m["h_ratio"] < self.tl_stop_min_height_ratio:
            return False, f"tl_stop_reject_small_height:{m['h_ratio']:.4f}"

        if m["area_ratio"] < self.tl_stop_min_area_ratio:
            return False, f"tl_stop_reject_small_area:{m['area_ratio']:.6f}"

        return True, "tl_stop_geometry_ok"


    def select_priority_red_light(self, decision_events, detections, frame_width, frame_height):
        """
        Güvenlik kuralı:
        Aynı anda yeşil ve kırmızı görülürse, kırmızı önceliklidir.
        Semantic pipeline bazen yeşil seçse bile raw detection içinde geçerli kırmızı varsa STOP üretir.
        """
        candidates = []

        def add_candidate(item, source_name):
            if not isinstance(item, dict):
                return

            state = str(
                item.get("traffic_light_state", item.get("state", "unknown"))
            ).lower().strip()

            if state != "red":
                return

            ok, gate_reason = self.traffic_light_geometry_ok(item, frame_width, frame_height)
            if not ok:
                return

            m = self.bbox_metrics({"bbox": item.get("bbox")}, frame_width, frame_height)
            if m is None:
                return

            try:
                state_conf = float(
                    item.get(
                        "traffic_light_state_confidence",
                        item.get("state_confidence", item.get("confidence", 0.0)),
                    )
                    or 0.0
                )
            except Exception:
                state_conf = 0.0

            try:
                det_conf = float(item.get("det_confidence", item.get("confidence", 0.0)) or 0.0)
            except Exception:
                det_conf = 0.0

            # Kırmızı için merkezden ziyade alt-yakınlık + güven + alan önemli.
            score = (
                state_conf * 4.0
                + det_conf * 2.0
                + float(m.get("bottom_ratio", 0.0)) * 1.5
                + float(m.get("area_ratio", 0.0)) * 80.0
            )

            selected = dict(item)
            selected["label"] = "traffic_light"
            selected["traffic_light_state"] = "red"
            selected["state_confidence"] = state_conf
            selected["traffic_light_state_confidence"] = state_conf
            selected["priority_red_source"] = source_name
            selected["traffic_light_geometry_gate"] = gate_reason
            selected["priority_red_score"] = round(score, 4)
            candidates.append((score, selected))

        for event in decision_events or []:
            if str(event.get("event_type", "")).strip() == "traffic_light":
                add_candidate(event, "decision_event")

        for det in detections or []:
            if str(det.get("label", "")).strip() == "traffic_light":
                add_candidate(det, "raw_detection")

        if not candidates:
            return None

        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[0][1]

    def select_traffic_light_from_decision_events(self, decision_events, frame_width, frame_height):
        """
        Perception TL pipeline zaten aktif ışığı seçip filtreliyor.
        Decision burada tekrar küçük bbox / geometry gate ile kırmızıyı öldürmemeli.

        Kritik kural:
        - decision_events içindeki traffic_light event'i ana kaynaktır.
        - Red > Yellow > Green önceliği vardır.
        - Geometry sadece perception tarafında yapılır; decision burada güvenlik kararını uygular.
        """
        if not self.tl_prefer_semantic_event:
            self.last_tl_active_light_marker = "TL_ACTIVE_LIGHT_REJECTED:semantic_event_disabled"
            return "unknown", None

        candidates = []

        for event in decision_events or []:
            if not isinstance(event, dict):
                continue

            if str(event.get("event_type", "")).strip() != "traffic_light":
                continue

            state = str(
                event.get("traffic_light_state", event.get("state", "unknown"))
            ).lower().strip()

            if state not in {"red", "yellow", "green"}:
                continue

            try:
                state_conf = float(
                    event.get(
                        "traffic_light_state_confidence",
                        event.get(
                            "state_confidence",
                            event.get("state_conf", event.get("confidence", 0.0)),
                        ),
                    )
                    or 0.0
                )
            except Exception:
                state_conf = 0.0

            try:
                det_conf = float(
                    event.get(
                        "det_confidence",
                        event.get("detection_confidence", event.get("confidence", 0.0)),
                    )
                    or 0.0
                )
            except Exception:
                det_conf = 0.0

            selected = dict(event)
            selected["label"] = "traffic_light"
            selected["traffic_light_state"] = state
            selected["state_confidence"] = state_conf
            selected["traffic_light_state_confidence"] = state_conf
            selected["traffic_light_geometry_gate"] = "trusted_perception_decision_event"
            selected["selected_by"] = "decision_events_trusted_active_tl"

            if state == "red":
                priority = 3.0
            elif state == "yellow":
                priority = 2.0
            else:
                priority = 1.0

            score = priority * 10.0 + state_conf * 2.0 + det_conf
            selected["decision_tl_score"] = round(float(score), 4)

            candidates.append((score, selected))

        if not candidates:
            self.last_tl_active_light_marker = "TL_ACTIVE_LIGHT_REJECTED:no_active_decision_event"
            return "unknown", None

        candidates.sort(key=lambda x: x[0], reverse=True)
        best = candidates[0][1]
        self.last_tl_active_light_marker = (
            "TL_ACTIVE_LIGHT_SELECTED:"
            f"state={best.get('traffic_light_state')},score={best.get('decision_tl_score')},"
            f"source={best.get('selected_by')}"
        )
        best["traffic_light_selection_marker"] = self.last_tl_active_light_marker
        return str(best.get("traffic_light_state", "unknown")).lower().strip(), best


    def get_best_sign(self, detections, frame_width=None, frame_height=None):
        """
        Decision log / active sign fallback.

        Uzak/minik/yanlış tabelalar decision'ı etkilemesin.
        sign_min_bottom_ratio, sign_min_area_ratio ve sign_center_bonus burada gerçekten kullanılır.
        """
        best = None
        best_score = -1.0

        for det in detections or []:
            if det.get("label") != "traffic_sign":
                continue

            sign_type = str(
                det.get("traffic_sign_type")
                or det.get("sign_type")
                or det.get("type")
                or "unknown"
            ).lower().strip()

            if sign_type in {"", "unknown", "none", "bilinmiyor"}:
                continue

            source = str(det.get("sign_source") or det.get("source") or "").lower()
            if "classifier_low_conf" in source:
                continue

            try:
                det_conf = float(det.get("confidence", 0.0) or 0.0)
            except Exception:
                det_conf = 0.0

            try:
                sign_conf = float(
                    det.get("sign_confidence")
                    or det.get("traffic_sign_confidence")
                    or det.get("classifier_confidence")
                    or det_conf
                    or 0.0
                )
            except Exception:
                sign_conf = 0.0

            if det_conf < self.traffic_sign_conf_threshold:
                continue

            if sign_conf < self.sign_classifier_conf_threshold:
                continue

            m = None
            area_ratio = 0.0
            bottom_ratio = 0.0
            center_score = 0.0

            if frame_width and frame_height:
                try:
                    m = self.bbox_metrics(det, frame_width, frame_height)
                except Exception:
                    m = None

                if m is None:
                    continue

                area_ratio = float(m.get("area_ratio", 0.0) or 0.0)
                bottom_ratio = float(m.get("bottom_ratio", 0.0) or 0.0)
                cx_ratio = float(m.get("cx_ratio", 0.5) or 0.5)

                if area_ratio < self.sign_min_area_ratio:
                    continue

                if bottom_ratio < self.sign_min_bottom_ratio:
                    continue

                if cx_ratio < 0.08 or cx_ratio > 0.98:
                    continue

                center_score = 1.0 - min(1.0, abs(cx_ratio - 0.50) / 0.50)

            distance_m = det.get("distance_m") or det.get("distance_est")
            try:
                distance_m = float(distance_m) if distance_m is not None else None
            except Exception:
                distance_m = None

            if distance_m is not None and distance_m > 45.0:
                continue

            if sign_type in self.stop_signs:
                if sign_conf < max(0.80, self.sign_classifier_conf_threshold):
                    continue

                # Uzak/sahte DUR tabelası yeşil ışıkta ani fren yaptırmasın.
                if distance_m is not None and distance_m > 18.0:
                    continue

                if m is not None and bottom_ratio < max(0.62, self.sign_min_bottom_ratio):
                    continue

            score = (
                det_conf * 0.35
                + sign_conf * 0.50
                + area_ratio * 35.0
                + bottom_ratio * 0.20
                + center_score * float(self.sign_center_bonus)
            )

            if score > best_score:
                best_score = score
                best = dict(det)
                best["traffic_sign_type"] = sign_type
                best["active_sign_score"] = round(float(score), 4)
                best["active_sign_filter"] = "decision_near_sign_filter"
                if m is not None:
                    best["bottom_ratio"] = round(bottom_ratio, 4)
                    best["area_ratio"] = round(area_ratio, 6)

        return best


    def select_active_traffic_light(self, detections, frame_width, frame_height):
        """
        Decision fallback trafik ışığı seçimi.

        Normal mimaride aktif trafik ışığını perception_tl_pipeline.select_active()
        seçer ve decision_events içinde decision'a gönderir.

        Bu fonksiyon sadece decision_events gelmezse fallback olarak kullanılmalı.
        Bu yüzden burada küçük/uzak/yanlış bbox'lara karşı çok daha sıkı filtre var.
        """

        if not getattr(self, "handle_traffic_lights", False):
            return "unknown", {"traffic_light_disabled": True}

        candidates = []

        # Perception ile aynı mantığa yakın tutuyoruz.
        target_x = float(getattr(self, "tl_active_target_x_ratio", 0.64))
        target_y = float(getattr(self, "tl_active_target_y_ratio", 0.24))

        # Kırmızı/sarı STOP için küçük bbox kabul edilmesin.
        min_width_ratio = float(getattr(self, "tl_stop_min_width_ratio", 0.012))
        min_height_ratio = float(getattr(self, "tl_stop_min_height_ratio", 0.040))
        min_area_ratio = float(getattr(self, "tl_stop_min_area_ratio", 0.00050))
        min_bottom_ratio = float(getattr(self, "tl_stop_min_bottom_ratio", 0.10))

        for det in detections:
            if det.get("label") != "traffic_light":
                continue

            det_conf = float(det.get("confidence", 0.0) or 0.0)
            if det_conf < self.traffic_light_conf_threshold:
                continue

            state = str(det.get("traffic_light_state", "unknown")).lower().strip()
            if state not in {"red", "yellow", "green"}:
                continue

            state_conf = self.get_state_confidence(det)
            if state_conf < self.traffic_light_state_conf_threshold:
                continue

            m = self.bbox_metrics(det, frame_width, frame_height)
            if m is None:
                continue

            ok_geom, geom_reason = self.traffic_light_geometry_ok(
                det,
                frame_width,
                frame_height,
            )
            if not ok_geom:
                det["traffic_light_geometry_rejected"] = True
                det["traffic_light_geometry_gate"] = geom_reason
                continue

            # Decision fallback'te de ROI dışı ışığı aktif sayma.
            if not (
                self.tl_active_roi_x_min <= m["cx_ratio"] <= self.tl_active_roi_x_max
                and self.tl_active_roi_y_min <= m["cy_ratio"] <= self.tl_active_roi_y_max
            ):
                det["traffic_light_geometry_rejected"] = True
                det["traffic_light_geometry_gate"] = (
                    f"tl_reject_outside_roi:"
                    f"x={m['cx_ratio']:.3f},y={m['cy_ratio']:.3f}"
                )
                continue

            # Green release için küçük bbox daha tolere edilebilir.
            # Ama red/yellow STOP için küçük bbox kesinlikle kabul edilmemeli.
            if state in {"red", "yellow"}:
                if m["w_ratio"] < min_width_ratio:
                    det["traffic_light_geometry_rejected"] = True
                    det["traffic_light_geometry_gate"] = (
                        f"tl_stop_reject_small_width:{m['w_ratio']:.4f}"
                    )
                    continue

                if m["h_ratio"] < min_height_ratio:
                    det["traffic_light_geometry_rejected"] = True
                    det["traffic_light_geometry_gate"] = (
                        f"tl_stop_reject_small_height:{m['h_ratio']:.4f}"
                    )
                    continue

                if m["area_ratio"] < min_area_ratio:
                    det["traffic_light_geometry_rejected"] = True
                    det["traffic_light_geometry_gate"] = (
                        f"tl_stop_reject_small_area:{m['area_ratio']:.6f}"
                    )
                    continue

                if m["bottom_ratio"] < min_bottom_ratio:
                    det["traffic_light_geometry_rejected"] = True
                    det["traffic_light_geometry_gate"] = (
                        f"tl_stop_reject_far_bottom:{m['bottom_ratio']:.3f}"
                    )
                    continue

            x_score = 1.0 - min(1.0, abs(m["cx_ratio"] - target_x) / 0.38)
            y_score = 1.0 - min(1.0, abs(m["cy_ratio"] - target_y) / 0.34)
            size_score = min(1.0, m["area_ratio"] / 0.00150)
            bottom_score = min(1.0, m["bottom_ratio"] / 0.34)

            # Decision fallback'te güven skorunu tamamen baskın yapmıyoruz.
            # Aksi halde küçük bbox + yüksek classifier confidence yine yanlış STOP üretir.
            score = (
                x_score * 0.34
                + y_score * 0.16
                + size_score * 0.22
                + bottom_score * 0.10
                + state_conf * 0.10
                + det_conf * 0.08
            )

            selected = dict(det)
            selected["state_confidence"] = state_conf
            selected["active_tl_score"] = round(float(score), 4)
            selected["active_tl_source"] = "decision_fallback"
            selected["tl_cx_ratio"] = round(float(m["cx_ratio"]), 4)
            selected["tl_cy_ratio"] = round(float(m["cy_ratio"]), 4)
            selected["tl_width_ratio"] = round(float(m["w_ratio"]), 4)
            selected["tl_height_ratio"] = round(float(m["h_ratio"]), 4)
            selected["tl_bottom_ratio"] = round(float(m["bottom_ratio"]), 4)
            selected["tl_area_ratio"] = round(float(m["area_ratio"]), 6)
            selected["traffic_light_geometry_gate"] = "decision_fallback_geometry_ok"

            candidates.append((score, selected))

        if not candidates:
            return "unknown", None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        return str(best.get("traffic_light_state", "unknown")).lower().strip(), best

    def get_person_risk(self, detections, frame_width, frame_height):
        persons = []

        for det in detections:
            if det.get("label") != "person":
                continue

            conf = float(det.get("confidence", 0.0))
            if conf < self.person_conf_threshold:
                continue

            m = self.bbox_metrics(det, frame_width, frame_height)
            if m is None:
                continue

            in_path = self.is_in_ego_path(m)

            copied = dict(det)
            copied.update({
                "bottom_ratio": m["bottom_ratio"],
                "cx_ratio": m["cx_ratio"],
                "cy_ratio": m["cy_ratio"],
                "area_ratio": m["area_ratio"],
                "in_ego_path": bool(in_path),
            })

            persons.append(copied)

        if not persons:
            return "none", None

        # Önce ego yol koridorundaki kişileri değerlendir.
        in_path_persons = [p for p in persons if p.get("in_ego_path")]
        side_persons = [p for p in persons if not p.get("in_ego_path")]

        if in_path_persons:
            in_path_persons.sort(
                key=lambda p: (
                    float(p.get("bottom_ratio", 0.0)),
                    float(p.get("confidence", 0.0)),
                    float(p.get("area_ratio", 0.0)),
                ),
                reverse=True,
            )

            nearest = in_path_persons[0]
            bottom = float(nearest.get("bottom_ratio", 0.0))

            if bottom >= self.person_stop_bottom_ratio:
                return "near", nearest

            if bottom >= self.person_slow_bottom_ratio:
                return "far", nearest

            return "far", nearest

        # Şerit dışında kişi varsa artık STOP yok; en fazla düşük risk/slow.
        side_persons.sort(
            key=lambda p: (
                float(p.get("bottom_ratio", 0.0)),
                float(p.get("confidence", 0.0)),
            ),
            reverse=True,
        )

        nearest_side = side_persons[0]
        if float(nearest_side.get("bottom_ratio", 0.0)) >= 0.70:
            return "side_near", nearest_side

        return "side", nearest_side

    def apply_person_stop_hold(self, person_risk, active_person):
        now = time.time()

        if person_risk == "near" and active_person is not None:
            self.last_near_person_time = now
            self.last_near_person = dict(active_person)
            return person_risk, active_person, False

        recently_near = (
            self.last_near_person is not None
            and (now - self.last_near_person_time) <= self.person_stop_hold_seconds
        )

        if recently_near:
            held = dict(self.last_near_person)
            held["held"] = True
            held["hold_seconds_left"] = round(
                self.person_stop_hold_seconds - (now - self.last_near_person_time),
                2,
            )
            return "near", held, True

        self.last_near_person = None
        return person_risk, active_person, False


    def should_stop_for_stop_sign(self, active_sign):
        """
        STOP/DUR tabelası için konservatif karar + hold/cooldown.

        Eski hata:
        - stop_sign_hold_seconds / stop_sign_cooldown_seconds parametreleri tanımlıydı
          ama pratikte kullanılmıyordu.
        - Gerçek DUR tabelası kamerada kaldığında tekrar tekrar STOP üretebiliyordu.
        """
        if not active_sign:
            return False, None

        sign_type = str(
            active_sign.get("traffic_sign_type")
            or active_sign.get("sign_type")
            or active_sign.get("type")
            or ""
        ).lower().strip()

        if sign_type not in {"dur", "stop", "stop_sign"}:
            return False, None

        source = str(active_sign.get("source", "")).lower()
        if "classifier_low_conf" in source:
            return False, "stop_sign_rejected_low_classifier_conf"

        try:
            det_conf = float(active_sign.get("confidence", 0.0) or 0.0)
        except Exception:
            det_conf = 0.0

        try:
            sign_conf = float(
                active_sign.get("sign_confidence")
                or active_sign.get("traffic_sign_confidence")
                or active_sign.get("classifier_confidence")
                or det_conf
                or 0.0
            )
        except Exception:
            sign_conf = 0.0

        if det_conf < 0.25:
            return False, f"stop_sign_rejected_low_yolo_conf:{det_conf:.3f}"

        if sign_conf < 0.55:
            return False, f"stop_sign_rejected_low_sign_conf:{sign_conf:.3f}"

        now = time.time()
        same_stop = self.last_stop_sign_type in {"dur", "stop", "stop_sign"}

        if same_stop and now < float(self.stop_sign_hold_until or 0.0):
            return True, "stop_sign_hold_active"

        if same_stop and now < float(self.stop_sign_cooldown_until or 0.0):
            return False, "stop_sign_cooldown_active"

        self.last_stop_sign_type = sign_type
        self.stop_sign_hold_until = now + float(self.stop_sign_hold_seconds)
        self.stop_sign_cooldown_until = self.stop_sign_hold_until + float(self.stop_sign_cooldown_seconds)

        return True, "stop_sign_detected"



    def _float_or_none(self, value):
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None

    def get_red_light_distance_context(self, tl):
        """
        Trafik ışığı mesafesinde ana kaynak sensör distance_m'dir.

        Kurallar:
        - distance_m / distance_est / tl_distance_m varsa direkt kullanılır.
        - Memory decay yok.
        - Bbox/area/bottom sensör mesafesini ezemez.
        - Sensör yoksa distance_m None döner; bbox sadece fallback kararında kullanılır.
        """
        tl = tl or {}

        raw_distance = self._float_or_none(
            tl.get("distance_m") or tl.get("distance_est") or tl.get("tl_distance_m")
        )
        raw_source = str(tl.get("distance_source") or "none")

        bottom_ratio = self._float_or_none(
            tl.get("tl_bottom_ratio") or tl.get("bottom_ratio") or tl.get("bbox_bottom_ratio")
        )
        height_ratio = self._float_or_none(
            tl.get("tl_height_ratio") or tl.get("height_ratio") or tl.get("bbox_height_ratio")
        )
        area_ratio = self._float_or_none(
            tl.get("tl_area_ratio") or tl.get("area_ratio") or tl.get("bbox_area_ratio")
        )

        bbox = tl.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            try:
                x1, y1, x2, y2 = [float(x) for x in bbox[:4]]
                if bottom_ratio is None:
                    bottom_ratio = y2 / 360.0
                if height_ratio is None:
                    height_ratio = max(0.0, y2 - y1) / 360.0
                if area_ratio is None:
                    area_ratio = (
                        max(0.0, x2 - x1)
                        * max(0.0, y2 - y1)
                        / (640.0 * 360.0)
                    )
            except Exception:
                pass

        stable_distance = None
        source = "none"

        if raw_distance is not None and 0.2 <= raw_distance <= 80.0:
            stable_distance = float(raw_distance)
            source = raw_source if raw_source and raw_source != "none" else "sensor_distance"

            # Sadece debug için tutuyoruz. Karar mesafesi memory'den üretilmiyor.
            self.last_red_distance_m = stable_distance
            self.last_red_distance_time = time.time()
            self.last_red_distance_source = source

        else:
            # SENSOR_HOLD_NO_DECAY:
            # Depth anlık kaybolursa no_sensor_stop'a düşüp gereksiz STOP verme.
            # Mesafeyi azaltmıyoruz/uydurmuyoruz; sadece son gerçek sensör mesafesini
            # kısa süre koruyoruz.
            try:
                now = time.time()
                last_d = getattr(self, "last_red_distance_m", None)
                last_t = getattr(self, "last_red_distance_time", None)
                last_src = getattr(self, "last_red_distance_source", "sensor_distance")

                if last_d is not None and last_t is not None:
                    age = now - float(last_t)
                    if 0.0 <= age <= 1.5 and 0.2 <= float(last_d) <= 80.0:
                        stable_distance = float(last_d)
                        raw_distance = float(last_d)
                        source = f"recent_sensor_hold:{last_src},age={age:.1f}"
            except Exception:
                pass

        return {
            "distance_m": stable_distance,
            "raw_distance_m": raw_distance,
            "distance_source": source,
            "raw_distance_source": raw_source,
            "bottom_ratio": bottom_ratio,
            "height_ratio": height_ratio,
            "area_ratio": area_ratio,
        }

    def route_stopline_context(self, route_intent):
        if not isinstance(route_intent, dict):
            return {
                "valid": False,
                "distance_m": None,
                "target_dist_m": None,
                "id": None,
                "stop_before_m": self.tl_stop_before_line_m,
            }

        dist = self._float_or_none(route_intent.get("distance_to_stopline_m"))
        stop_before = self._float_or_none(route_intent.get("stop_before_m"))
        if stop_before is None:
            stop_before = self.tl_stop_before_line_m

        target_dist = self._float_or_none(route_intent.get("stop_target_dist_m"))
        if target_dist is None and dist is not None:
            target_dist = float(dist) - float(stop_before)

        stopline_ctx = route_intent.get("stopline_context") if isinstance(route_intent.get("stopline_context"), dict) else {}
        explicit_route_valid = route_intent.get("stopline_route_valid", stopline_ctx.get("route_valid"))
        if explicit_route_valid is None:
            explicit_route_valid = bool(route_intent.get("stopline_valid", False)) and not bool(
                route_intent.get("stopline_selection_reject_reason", stopline_ctx.get("selection_reject_reason"))
            )
        route_valid = bool(explicit_route_valid)
        return {
            "valid": bool(route_intent.get("stopline_valid", False)) and route_valid and dist is not None,
            "distance_m": dist,
            "target_dist_m": target_dist,
            "id": route_intent.get("stopline_id"),
            "stop_before_m": stop_before,
            "pose": route_intent.get("stopline_pose"),
            "release_after_pass_m": route_intent.get("release_after_pass_m"),
            "road_id": route_intent.get("stopline_road_id", route_intent.get("road_id")),
            "lane_id": route_intent.get("stopline_lane_id", route_intent.get("lane_id")),
            "lateral_m": route_intent.get("stopline_lateral_m"),
            "route_valid": route_valid,
            "road_match": bool(route_intent.get("stopline_road_match", stopline_ctx.get("road_match", False))),
            "lane_match": bool(route_intent.get("stopline_lane_match", stopline_ctx.get("lane_match", False))),
            "route_connected_match": bool(route_intent.get("stopline_route_connected_match", stopline_ctx.get("route_connected_match", False))),
            "route_valid_reason": route_intent.get("stopline_route_valid_reason", stopline_ctx.get("route_valid_reason")),
            "selection_reject_reason": route_intent.get("stopline_selection_reject_reason", stopline_ctx.get("selection_reject_reason")),
        }

    def normalize_decision_output(self, output):
        output = normalize_decision_command(output)
        should_stop = bool(output.get("should_stop", False))
        emergency = bool(output.get("emergency_brake", False))
        decision_stop = str(output.get("decision", "")).upper() == "STOP"
        reason = str(output.get("reason", "") or "")
        tl_hold_reason = (
            self.tl_hold_active
            and "TL_GREEN_RELEASE" not in reason
            and str(output.get("release_condition", "")) != "confirmed_green"
        )

        if should_stop or emergency or decision_stop:
            output["decision"] = "STOP"
            output["target_speed"] = 0.0
            output["target_speed_mps"] = 0.0
            output["target_speed_kmh"] = 0.0
            output["should_stop"] = True
            output["stop_required"] = True
            output["risk"] = "HIGH" if emergency else output.get("risk", "MEDIUM_STOP")
            if not output.get("stop_reason"):
                output["stop_reason"] = reason or "structured_stop"

            marker = "TL_DECISION_NORMALIZED_SHOULD_STOP"
            if marker not in reason:
                output["reason"] = f"{reason}|{marker}" if reason else marker
            if self.tl_hold_active and "TL_STOPLINE_HOLD_ACTIVE" not in output["reason"]:
                output["reason"] = f"{output['reason']}|TL_STOPLINE_HOLD_ACTIVE"
        elif tl_hold_reason and str(output.get("decision", "")).upper() == "SLOW":
            marker = "TL_HOLD_APPROACH_ACTIVE"
            if marker not in reason:
                output["reason"] = f"{reason}|{marker}" if reason else marker

        return output

    def evaluate_traffic_light_state_machine(self, traffic_light_state, route_intent, traffic_light_det=None):
        if not self.handle_traffic_lights:
            return None

        if not hasattr(self, "tl_stopline_sm"):
            self.tl_stopline_sm = TrafficLightStoplineStateMachine(
                default_go_speed=self.default_go_speed,
                approach_speed=min(self.default_go_speed, 4.0),
                crawl_speed=self.creep_speed,
                stop_speed=self.stop_speed,
                green_confirm_count=max(1, int(getattr(self, "tl_green_confirm_count", 1))),
                red_confirm_count=max(1, int(getattr(self, "tl_red_confirm_count", 1))),
                yellow_confirm_count=max(1, int(getattr(self, "tl_yellow_confirm_count", 1))),
                decel_mps2=self.tl_brake_decel_mps2,
                reaction_time_s=self.tl_reaction_time_s,
                safety_margin_m=self.tl_brake_safety_margin_m,
            )

        route = route_intent if isinstance(route_intent, dict) else {}
        stopline = self.route_stopline_context(route)
        current_speed = self._float_or_none(route.get("current_speed_mps")) or 0.0
        rule = self.tl_stopline_sm.evaluate(
            traffic_light_state,
            stopline,
            current_speed,
            traffic_light_det=traffic_light_det,
            pitch_deg=route.get("ego_pitch_deg", 0.0),
        )
        self.tl_sm_state = self.tl_stopline_sm.state
        self.tl_sm_stopline_id = self.tl_stopline_sm.stopline_id
        self.tl_sm_last_green_time = self.tl_stopline_sm.last_green_time
        self.tl_hold_active = self.tl_stopline_sm.hold_active
        self.tl_hold_stopline_id = self.tl_stopline_sm.hold_stopline_id
        self.tl_hold_since = self.tl_stopline_sm.hold_since
        self.tl_hold_reason = self.tl_stopline_sm.hold_reason
        self.last_non_green_light_state = self.tl_stopline_sm.last_non_green_light_state
        self.last_confirmed_green_time = self.tl_stopline_sm.last_green_time
        return rule

    def build_rule_decision(
        self,
        vehicle_rule,
        traffic_light_state,
        traffic_light_det,
        active_sign,
        person_risk,
        active_person,
        person_hold_active,
        route_intent=None,
    ):
        def with_priority(rule, winner, suppressed):
            out = dict(rule)
            out["priority_winner"] = winner
            out["suppressed_sources"] = list(suppressed)
            if "release_condition" not in out:
                out["release_condition"] = None
            if "should_stop" not in out:
                out["should_stop"] = str(out.get("decision", "GO")).upper() == "STOP"
            if out["should_stop"] or str(out.get("decision", "")).upper() == "STOP":
                out["decision"] = "STOP"
                out["target_speed"] = self.stop_speed
                out["should_stop"] = True
            return out

        suppressed_sources = []
        active_sign_type = None
        active_sign_name = None

        if active_sign is not None:
            active_sign_type = str(active_sign.get("sign_type", "unknown"))
            active_sign_name = active_sign.get("sign_name", active_sign_type)

        if active_sign_type in self.speed_limit_map:
            self.current_speed_limit = self.speed_limit_map[active_sign_type]
            self.last_speed_limit_sign = active_sign_type

        route_wait, route_wait_reason = self.route_intent_requires_wait(route_intent)

        tl_rule = self.evaluate_traffic_light_state_machine(traffic_light_state, route_intent, traffic_light_det)
        if tl_rule is not None:
            tl_rule = dict(tl_rule)
            reason = str(tl_rule.get("reason", "") or "")
            if "DECISION_TL_FINAL" not in reason:
                tl_rule["reason"] = f"{reason}|DECISION_TL_FINAL" if reason else "DECISION_TL_FINAL"
            suppressed = ["route_blocked", "stop_sign", "pedestrian", "front_vehicle", "speed_limit", "lane_keep", "normal_route"]
            return with_priority(tl_rule, "traffic_light_stopline", suppressed)

        route_tl_ctx = self.route_stopline_context(route_intent)
        tl_route_rejected = bool(
            self.handle_traffic_lights
            and traffic_light_state in {"red", "yellow"}
            and not bool(route_tl_ctx.get("valid"))
        )
        tl_route_reject_reason = None
        if tl_route_rejected:
            tl_route_reject_reason = (
                "TL_ACTIVE_LIGHT_REJECTED|TL_ROUTE_VALIDATION|TL_RELEASE_REJECTED_REASON:no_route_valid_stopline"
                f":state={traffic_light_state}"
                f":stopline_id={route_tl_ctx.get('id')}"
                f":road_match={int(bool(route_tl_ctx.get('road_match')))}"
                f":lane_match={int(bool(route_tl_ctx.get('lane_match')))}"
                f":route_connected_match={int(bool(route_tl_ctx.get('route_connected_match')))}"
                f":route_valid_reason={route_tl_ctx.get('route_valid_reason')}"
                f":selection_reject_reason={route_tl_ctx.get('selection_reject_reason')}"
            )
            suppressed_sources.append(tl_route_reject_reason)

        if traffic_light_state in {"red", "yellow"} and tl_route_rejected:
            self._qualified_green_count = 0
            return with_priority({
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": min(self.slow_speed, self.kmh_to_mps(10.0)),
                "reason": (
                    f"TL_NO_ROUTE_VALID_STOPLINE_PRECAUTION:{tl_route_reject_reason}"
                    "|DECISION_TL_FINAL"
                ),
                "should_stop": False,
                "stop_reason": None,
                "release_condition": "route_valid_stopline_required",
                "active_stopline_id": route_tl_ctx.get("id"),
                "distance_to_stopline_m": route_tl_ctx.get("distance_m"),
                "stop_target_distance_m": route_tl_ctx.get("target_dist_m"),
                "tl_applicable": False,
                "active_light_state": traffic_light_state,
                "stop_required": False,
                "release_allowed": False,
                "violation": False,
            }, "traffic_light_route_validation", ["speed_limit", "lane_keep", "normal_route"])

        if person_risk == "near":
            return with_priority({
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": "near_person_hold" if person_hold_active else "near_person_in_path",
                "should_stop": True,
                "stop_reason": "pedestrian",
            }, "emergency_pedestrian", ["traffic_light_stopline", "route_blocked", "stop_sign", "front_vehicle", "speed_limit", "lane_keep", "normal_route"])

        if route_wait:
            return with_priority({
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": f"route_constraint_wait:{route_wait_reason}",
                "should_stop": True,
                "stop_reason": "route_blocked",
                "route_status": "blocked",
            }, "route_blocked", ["stop_sign", "front_vehicle", "speed_limit", "lane_keep", "normal_route"])

        if vehicle_rule["decision"] == "STOP":
            rule = dict(vehicle_rule)
            rule["should_stop"] = True
            rule["stop_reason"] = "front_vehicle"
            return with_priority(rule, "emergency_front_vehicle", ["traffic_light_stopline", "route_blocked", "stop_sign", "speed_limit", "lane_keep", "normal_route"])

        if active_sign_type in self.stop_signs:
            should_stop, stop_reason = self.should_stop_for_stop_sign(active_sign)
            if should_stop:
                return with_priority({
                    "decision": "STOP",
                    "risk": "HIGH",
                    "target_speed": self.stop_speed,
                    "reason": stop_reason,
                    "should_stop": True,
                    "stop_reason": "stop_sign",
                    "release_condition": "stop_sign_hold_elapsed",
                }, "stop_sign", ["speed_limit", "lane_keep", "normal_route"])

        if active_sign_type in self.slow_signs:
            return with_priority({
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": f"slow_sign_detected:{active_sign_type}",
            }, "slow_sign", ["speed_limit", "lane_keep", "normal_route"])

        if person_risk == "far":
            return with_priority({
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": "person_in_path_far",
            }, "pedestrian_slow", ["speed_limit", "lane_keep", "normal_route"])

        if vehicle_rule["decision"] == "SLOW":
            return with_priority(vehicle_rule, "front_vehicle_slow", ["speed_limit", "lane_keep", "normal_route"])

        if person_risk in {"side", "side_near"}:
            if person_risk == "side_near":
                return with_priority({
                    "decision": "SLOW",
                    "risk": "MEDIUM",
                    "target_speed": self.slow_speed,
                    "reason": "side_person_near_slow",
                }, "side_person_slow", ["speed_limit", "lane_keep", "normal_route"])

        if self.current_speed_limit is not None and traffic_light_state != "green":
            return with_priority({
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.current_speed_limit,
                "reason": f"speed_limit_active:{self.last_speed_limit_sign}"
                + (f"|{tl_route_reject_reason}" if tl_route_reject_reason else ""),
                "should_stop": False,
            }, "speed_limit", suppressed_sources + ["lane_keep", "normal_route"])

        if active_sign_type in self.warning_signs:
            return with_priority({
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "reason": f"warning_sign_detected:{active_sign_type}",
            }, "warning_sign", suppressed_sources + ["lane_keep", "normal_route"])

        if tl_route_reject_reason and vehicle_rule.get("decision") != "STOP":
            out = dict(vehicle_rule)
            out["reason"] = f"{vehicle_rule.get('reason', 'vehicle_rule')}|{tl_route_reject_reason}"
            return with_priority(out, "normal_route", suppressed_sources)

        normal = dict(vehicle_rule)
        if normal.get("decision") not in {"GO", "SLOW", "STOP"}:
            normal["decision"] = "GO"
        normal["reason"] = str(normal.get("reason", "normal_route_clear"))
        if traffic_light_state == "green":
            normal["reason"] = f"{normal['reason']}|green_light_no_hold"
        return with_priority(normal, "normal_route", suppressed_sources)

    def stabilize_decision(self, output):
        now = time.time()

        if self.last_output is None:
            self.last_output = dict(output)
            self.last_output_time = now
            return output

        prev_decision = self.last_output.get("decision", "GO")
        age = now - self.last_output_time

        if prev_decision == "STOP" and output["decision"] != "STOP":
            prev_reason = str(self.last_output.get("reason", ""))
            current_reason = str(output.get("reason", ""))
            current_tl = str(output.get("traffic_light_state", "unknown")).lower().strip()

            if "route_constraint_wait" in prev_reason and output.get("decision") != "STOP":
                self.last_output_time = now
                self.last_output = dict(output)
                return output

            if (
                self.tl_green_release_bypass_hold
                and "red_light_" in prev_reason
                and "green_light_confirmed_stable" in current_reason
            ):
                self.last_output_time = now
                self.last_output = dict(output)
                return output

            if age < self.stop_hold_seconds:
                release_confirmed = (
                    str(output.get("release_condition", "") or "").lower() == "confirmed_green"
                    or str(output.get("tl_state_machine_state", "") or "").upper() == "GREEN_RELEASED"
                    or "TL_RELEASE_CONFIRMED_GREEN" in current_reason
                    or "TL_RELEASE_GREEN_CONFIRMED" in current_reason
                )
                if release_confirmed:
                    self.last_output_time = now
                    self.last_output = dict(output)
                    out = dict(output)
                    out["reason"] = (
                        f"{current_reason}|TL_STATE_CONFLICT_FIXED:green_release_bypassed_decision_hold"
                    )
                    return out

                held = dict(output)
                held["decision"] = "STOP"
                held["risk"] = "HIGH"
                held["target_speed"] = self.stop_speed
                held["target_speed_mps"] = self.stop_speed
                held["reason"] = f"decision_hold_after_stop:{self.last_output.get('reason')}"
                return held

        if prev_decision == "SLOW" and output["decision"] == "GO":
            if age < self.slow_hold_seconds:
                held = dict(output)
                held["decision"] = "SLOW"
                held["risk"] = "MEDIUM"
                prev_speed = float(
                    self.last_output.get(
                        "target_speed_mps",
                        self.last_output.get("target_speed", self.slow_speed),
                    )
                    or self.slow_speed
                )
                held_speed = min(self.slow_speed, prev_speed, float(output.get("target_speed", self.slow_speed)))
                held["target_speed"] = held_speed
                held["target_speed_mps"] = held_speed
                held["reason"] = f"decision_hold_after_slow:{self.last_output.get('reason')}"
                return held

        if output["decision"] != prev_decision or output.get("reason") != self.last_output.get("reason"):
            self.last_output_time = now

        self.last_output = dict(output)
        return output

    def fix_traffic_light_contract_conflicts(self, output):
        reason = str(output.get("reason", "") or "")
        release_confirmed = (
            str(output.get("release_condition", "") or "").lower() == "confirmed_green"
            or str(output.get("tl_state_machine_state", "") or "").upper() == "GREEN_RELEASED"
            or "TL_RELEASE_CONFIRMED_GREEN" in reason
            or "TL_RELEASE_GREEN_CONFIRMED" in reason
        )
        traffic_light_winner = str(output.get("priority_winner", "") or "") == "traffic_light_stopline"
        hard_safety = bool(output.get("emergency_brake", False)) or str(
            output.get("route_decision_status", "")
        ).lower() == "blocked"

        if release_confirmed and traffic_light_winner and not hard_safety:
            if bool(output.get("should_stop", False)) or str(output.get("decision", "")).upper() == "STOP":
                output = dict(output)
                output["decision"] = "GO"
                output["should_stop"] = False
                output["stop_reason"] = None
                output["emergency_brake"] = False
                output["target_speed"] = round(float(self.default_go_speed), 3)
                output["target_speed_mps"] = round(float(self.default_go_speed), 3)
                output["target_speed_kmh"] = round(self.mps_to_kmh(self.default_go_speed), 1)
                output["green_release_allowed"] = True
                output["release_allowed"] = True
                output["stop_required"] = False
                output["reason"] = f"{reason}|TL_STATE_CONFLICT_FIXED:green_release_forced_go"

        return output

    def callback(self, msg):
        try:
            data = json.loads(msg.data)
        except Exception as exc:
            self.get_logger().warn(f"JSON parse hatası: {exc}")
            return

        frame_width = int(data.get("image_width", data.get("frame_width", 800)))
        frame_height = int(data.get("image_height", data.get("frame_height", 600)))
        detections = data.get("detections", [])

        front_vehicle = self.select_front_vehicle(detections, frame_width, frame_height)
        front_vehicle, used_memory = self.apply_front_vehicle_memory(front_vehicle)
        vehicle_rule = self.evaluate_vehicle_rule(front_vehicle, used_memory)

        # CLEAN_TL_SINGLE_SOURCE:
        # Trafik ışığında tek karar kaynağı perception'ın aynı frame içinde seçtiği
        # active traffic-light decision_event olmalı. Raw detections içinden tekrar
        # aktif ışık seçmek ve priority_red override yapmak, uzaktaki/yan taraftaki
        # kırmızının yeşili ezmesine sebep oluyordu.
        decision_events = data.get("decision_events", [])
        if not isinstance(decision_events, list):
            decision_events = []

        # Geriye uyumluluk/debug için topic fallback kalsın; ama normal akışta
        # aynı detections_json frame'i içindeki decision_events kullanılır.
        if not decision_events:
            decision_events = self.get_fresh_decision_events()

        traffic_light_state, traffic_light_det = self.select_traffic_light_from_decision_events(
            decision_events,
            frame_width,
            frame_height,
        )

        # Eğer aynı frame payload içinde aktif ışık varsa, decision_events kaçsa bile
        # perception'ın seçtiği aktif ışığı kullan. Burada tekrar geometry gate yapmıyoruz;
        # aksi halde gerçek küçük kırmızı decision tarafında unknown'a düşüyor.
        if traffic_light_state == "unknown":
            payload_tl_state = str(data.get("traffic_light_state", "unknown")).lower().strip()
            if payload_tl_state in {"red", "yellow", "green"}:
                try:
                    payload_state_conf = float(data.get("traffic_light_state_confidence") or 0.0)
                except Exception:
                    payload_state_conf = 0.0

                try:
                    payload_det_conf = float(data.get("traffic_light_confidence") or 0.0)
                except Exception:
                    payload_det_conf = 0.0

                candidate = {
                    "label": "traffic_light",
                    "traffic_light_state": payload_tl_state,
                    "state_confidence": payload_state_conf,
                    "traffic_light_state_confidence": payload_state_conf,
                    "confidence": payload_det_conf,
                    "bbox": data.get("traffic_light_active_bbox"),
                    "traffic_light_state_source": data.get("traffic_light_state_source"),
                    "traffic_light_color_reason": data.get("traffic_light_color_reason"),
                    "traffic_light_geometry_gate": "trusted_perception_payload_active_tl",
                    "selected_by": "perception_payload_active_tl",
                }
                self.last_tl_active_light_marker = (
                    "TL_ACTIVE_LIGHT_SELECTED:"
                    f"state={payload_tl_state},source=perception_payload_active_tl"
                )
                candidate["traffic_light_selection_marker"] = self.last_tl_active_light_marker

                traffic_light_state = payload_tl_state
                traffic_light_det = candidate

        active_sign = self.get_best_sign(detections, frame_width, frame_height)

        raw_person_risk, raw_active_person = self.get_person_risk(
            detections,
            frame_width,
            frame_height,
        )

        person_risk, active_person, person_hold_active = self.apply_person_stop_hold(
            raw_person_risk,
            raw_active_person,
        )

        route_intent = self.get_fresh_route_intent()

        final_rule = self.build_rule_decision(
            vehicle_rule=vehicle_rule,
            traffic_light_state=traffic_light_state,
            traffic_light_det=traffic_light_det,
            active_sign=active_sign,
            person_risk=person_risk,
            active_person=active_person,
            person_hold_active=person_hold_active,
            route_intent=route_intent,
        )

        distance_est = vehicle_rule.get("distance_est", None)


        output = {
            "stamp": time.time(),
            "source": "decision_node",
            "ttl_sec": 1.0,
            "command_version": "decision_command_v1",
            "decision": final_rule["decision"],
            "risk": final_rule["risk"],
            # target_speed eski uyumluluk için m/s.
            # target_speed_kmh insan/yarışma/log tarafında ana hızdır.
            "target_speed": round(float(final_rule["target_speed"]), 3),
            "target_speed_mps": round(float(final_rule["target_speed"]), 3),
            "target_speed_kmh": round(self.mps_to_kmh(final_rule["target_speed"]), 1),
            "distance_est": round(float(distance_est), 2) if distance_est is not None else None,
            "front_vehicle": front_vehicle,
            "traffic_light_state": traffic_light_state,
            "traffic_light": traffic_light_det,
            "traffic_light_selection_marker": self.last_tl_active_light_marker,
            "tl_state_machine_state": getattr(self, "tl_sm_state", "UNKNOWN"),
            "active_stopline_id": final_rule.get("active_stopline_id") or (route_intent.get("stopline_id") if isinstance(route_intent, dict) else None),
            "stopline_id": final_rule.get("active_stopline_id") or (route_intent.get("stopline_id") if isinstance(route_intent, dict) else None),
            "distance_to_stopline_m": final_rule.get("distance_to_stopline_m", route_intent.get("distance_to_stopline_m") if isinstance(route_intent, dict) else None),
            "stopline_dist_m": route_intent.get("distance_to_stopline_m") if isinstance(route_intent, dict) else None,
            "stop_before_m": route_intent.get("stop_before_m") if isinstance(route_intent, dict) else None,
            "stop_target_dist_m": final_rule.get("stop_target_distance_m", route_intent.get("stop_target_dist_m") if isinstance(route_intent, dict) else None),
            "stop_target_distance_m": final_rule.get("stop_target_distance_m", route_intent.get("stop_target_dist_m") if isinstance(route_intent, dict) else None),
            "stopline_route_valid": route_intent.get("stopline_route_valid") if isinstance(route_intent, dict) else None,
            "stopline_context": route_intent.get("stopline_context") if isinstance(route_intent, dict) else None,
            "current_speed_mps": route_intent.get("current_speed_mps") if isinstance(route_intent, dict) else None,
            "should_stop": bool(final_rule.get("should_stop", final_rule["decision"] == "STOP")),
            "stop_reason": final_rule.get("stop_reason", final_rule["reason"] if final_rule["decision"] == "STOP" else None),
            "stop_distance_m": final_rule.get("stop_distance_m"),
            "emergency_brake": bool(final_rule.get("emergency_brake", False)),
            "release_condition": final_rule.get("release_condition"),
            "green_release_allowed": final_rule.get("release_condition") == "confirmed_green",
            "tl_applicable": bool(final_rule.get("tl_applicable", final_rule.get("priority_winner") == "traffic_light_stopline")),
            "active_light_state": final_rule.get("active_light_state", traffic_light_state),
            "stop_required": bool(final_rule.get("stop_required", final_rule.get("should_stop", False))),
            "release_allowed": bool(final_rule.get("release_allowed", final_rule.get("release_condition") == "confirmed_green")),
            "violation": bool(final_rule.get("violation", False)),
            "debug_reason": final_rule["reason"],
            "desired_maneuver": route_intent.get("desired_maneuver", route_intent.get("route_intent"))
            if isinstance(route_intent, dict) else None,
            "lane_keep_required": route_intent.get("lane_keep_required", True)
            if isinstance(route_intent, dict) else True,
            "preferred_lane": route_intent.get("preferred_side", "right")
            if isinstance(route_intent, dict) else "right",
            "route_allowed_maneuvers": route_intent.get("allowed_maneuvers", [])
            if isinstance(route_intent, dict) else [],
            "route_forbidden_maneuvers": route_intent.get("forbidden_maneuvers", [])
            if isinstance(route_intent, dict) else [],
            "lane_keep_status": route_intent.get("lane_keep_status")
            if isinstance(route_intent, dict) else None,
            "active_sign_constraints": route_intent.get("active_route_constraints", route_intent.get("active_constraints", []))
            if isinstance(route_intent, dict) else [],
            "red_light_latch_active": bool(time.time() < float(self.red_light_latch_until or 0.0)),
            "red_light_latch_left_s": round(
                max(0.0, float(self.red_light_latch_until or 0.0) - time.time()),
                2,
            ),
            "active_sign": active_sign,
            "person_risk": person_risk,
            "raw_person_risk": raw_person_risk,
            "person_hold_active": person_hold_active,
            "active_person": active_person,
            "speed_limit_active": self.last_speed_limit_sign,
            "decision_events": decision_events,
            "route_intent": route_intent,
            "route_decision_status": route_intent.get("route_decision_status") if isinstance(route_intent, dict) else None,
            "route_decision_reason": route_intent.get("route_decision_reason") if isinstance(route_intent, dict) else None,
            "route_status": route_intent.get("route_status") if isinstance(route_intent, dict) else None,
            "route_status_decision": final_rule.get("route_status"),
            "route_intent_summary": route_intent.get("route_intent") if isinstance(route_intent, dict) else None,
            "priority_winner": final_rule.get("priority_winner", "normal_route"),
            "suppressed_sources": final_rule.get("suppressed_sources", []),
            "reason": final_rule["reason"],
            "debug_state": {
                "traffic_light_state": traffic_light_state,
                "tl_state_machine_state": getattr(self, "tl_sm_state", "UNKNOWN"),
                "tl_rule": final_rule.get("tl_debug"),
            },
            "debug_policy": {
                "path_x_center": self.path_x_center,
                "path_half_width_top": self.path_half_width_top,
                "path_half_width_bottom": self.path_half_width_bottom,
                "person_stop_bottom_ratio": self.person_stop_bottom_ratio,
                "stop_distance": self.stop_distance,
                "slow_distance": self.slow_distance,
            },
        }

        output = self.stabilize_decision(output)
        output = self.fix_traffic_light_contract_conflicts(output)
        output = self.normalize_decision_output(output)
        output = self.fix_traffic_light_contract_conflicts(output)
        output["target_speed_mps"] = round(float(output.get("target_speed", 0.0)), 3)
        output["target_speed_kmh"] = round(self.mps_to_kmh(output.get("target_speed", 0.0)), 1)
        output["speed_limit_active_kmh"] = round(self.mps_to_kmh(self.current_speed_limit), 1) if self.current_speed_limit is not None else None
        self.last_output = dict(output)

        out_msg = String()
        out_msg.data = json.dumps(output, ensure_ascii=False)
        self.pub.publish(out_msg)
        self.command_pub.publish(out_msg)

        debug_msg = String()
        debug_msg.data = json.dumps({
            "stamp": output.get("stamp"),
            "source": "decision_node",
            "traffic_light_state": output.get("traffic_light_state"),
            "tl_state_machine_state": output.get("tl_state_machine_state"),
            "stopline_id": output.get("stopline_id"),
            "distance_to_stopline_m": output.get("distance_to_stopline_m"),
            "stopline_dist_m": output.get("stopline_dist_m"),
            "stop_before_m": output.get("stop_before_m"),
            "stop_target_dist_m": output.get("stop_target_dist_m"),
            "stopline_route_valid": output.get("stopline_route_valid"),
            "target_speed_mps": output.get("target_speed"),
            "current_speed_mps": output.get("current_speed_mps"),
            "should_stop": output.get("should_stop"),
            "emergency_brake": output.get("emergency_brake"),
            "tl_hold_active": self.tl_hold_active,
            "tl_hold_stopline_id": self.tl_hold_stopline_id,
            "green_release_age": (
                round(time.time() - self.tl_sm_last_green_time, 3)
                if self.tl_sm_last_green_time else None
            ),
            "route_allowed_maneuvers": output.get("route_allowed_maneuvers"),
            "route_forbidden_maneuvers": output.get("route_forbidden_maneuvers"),
            "lane_keep_status": output.get("lane_keep_status"),
            "active_sign_constraints": output.get("active_sign_constraints"),
            "reason": output.get("reason"),
            "tl_debug": output.get("debug_state", {}).get("tl_rule"),
        }, ensure_ascii=False)
        self.debug_pub.publish(debug_msg)

        reason_text = str(output.get("reason", ""))
        if "TL_" in reason_text or output.get("route_decision_status") == "blocked":
            self.get_logger().warning(
                "TL_STATE_MACHINE "
                f"state={output.get('tl_state_machine_state')} "
                f"light={output.get('traffic_light_state')} "
                f"stopline_id={output.get('stopline_id')} "
                f"dist={output.get('distance_to_stopline_m')} "
                f"target_dist={output.get('stop_target_dist_m')} "
                f"target_speed={output.get('target_speed_mps')} "
                f"should_stop={output.get('should_stop')} "
                f"emergency={output.get('emergency_brake')} "
                f"reason={reason_text}",
                throttle_duration_sec=0.25,
            )
            tl_debug = output.get("debug_state", {}).get("tl_rule") or {}
            if isinstance(tl_debug, dict):
                markers = tl_debug.get("markers") or []
                for marker in markers:
                    if marker:
                        self.get_logger().info(str(marker), throttle_duration_sec=0.5)

        active_person = output.get("active_person")
        active_sign = output.get("active_sign")
        traffic_light = output.get("traffic_light")

        person_conf = active_person.get("confidence") if active_person else None
        person_x = active_person.get("cx_ratio") if active_person else None
        person_bottom = active_person.get("bottom_ratio") if active_person else None
        person_in_path = active_person.get("in_ego_path") if active_person else None

        sign_name = active_sign.get("sign_name") if active_sign else None
        sign_type = active_sign.get("sign_type") if active_sign else None

        speed_limit_kmh = output.get("speed_limit_active_kmh")
        route_status = output.get("route_decision_status")
        route_reason = output.get("route_decision_reason")
        route_maneuver = None
        if isinstance(output.get("route_intent"), dict):
            route_maneuver = output["route_intent"].get("route_intent")

        light_conf = None
        if traffic_light:
            light_conf = self.get_state_confidence(traffic_light)

        decision_log = (
            "\n"
            "================ ADAS DECISION LOG ================\n"
            f"DECISION        : {output.get('decision')}\n"
            f"RISK            : {output.get('risk')}\n"
            f"TARGET SPEED    : {output.get('target_speed_kmh')} km/h | {output.get('target_speed')} m/s\n"
            f"REASON          : {output.get('reason')}\n"
            "---------------------------------------------------\n"
            f"FRONT DISTANCE  : {output.get('distance_est')}\n"
            f"FRONT VEHICLE   : {front_vehicle is not None}\n"
            f"TRAFFIC LIGHT   : {output.get('traffic_light_state')} | state_conf={light_conf}\n"
            f"TRAFFIC SIGN    : {sign_name} | type={sign_type}\n"
            f"PERSON RISK     : {output.get('person_risk')} | conf={person_conf} | x={person_x} | bottom={person_bottom} | in_path={person_in_path}\n"
            f"SPEED LIMIT     : {output.get('speed_limit_active')} | {speed_limit_kmh} km/h\n"
            f"ROUTE INTENT    : {route_maneuver} | status={route_status} | reason={route_reason}\n"
            "===================================================\n"
        )

        self.get_logger().info(decision_log, throttle_duration_sec=0.5)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
