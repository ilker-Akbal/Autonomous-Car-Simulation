import json
import time
from collections import deque

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


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
        self.declare_parameter("decision_topic", "/adas/decision")

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
        self.declare_parameter("yellow_as_slow", True)
        # Trafik ışığı final kararı decision_node içindedir.
        self.declare_parameter("handle_traffic_lights", True)

        # Karar stabilizasyonu.
        self.declare_parameter("stop_hold_seconds", 0.8)
        self.declare_parameter("slow_hold_seconds", 0.5)
        self.declare_parameter("decision_timeout_seconds", 1.5)

        self.detections_topic = self.get_parameter("detections_topic").value
        self.decision_topic = self.get_parameter("decision_topic").value

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
        self.yellow_as_slow = bool(self.get_parameter("yellow_as_slow").value)
        self.handle_traffic_lights = bool(self.get_parameter("handle_traffic_lights").value)

        self.stop_hold_seconds = float(self.get_parameter("stop_hold_seconds").value)
        self.slow_hold_seconds = float(self.get_parameter("slow_hold_seconds").value)
        self.decision_timeout_seconds = float(self.get_parameter("decision_timeout_seconds").value)

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

        self.pub = self.create_publisher(String, self.decision_topic, 10)
        self.sub = self.create_subscription(
            String,
            self.detections_topic,
            self.callback,
            10,
        )

        self.get_logger().info(
            f"decision_node başladı: {self.detections_topic} -> {self.decision_topic}"
        )

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

    def select_active_traffic_light(self, detections, frame_width, frame_height):
        if not getattr(self, "handle_traffic_lights", False):
            return "unknown", {"traffic_light_disabled": True}

        candidates = []

        for det in detections:
            if det.get("label") != "traffic_light":
                continue

            det_conf = float(det.get("confidence", 0.0))
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

            if not (
                self.tl_active_roi_x_min <= m["cx_ratio"] <= self.tl_active_roi_x_max
                and self.tl_active_roi_y_min <= m["cy_ratio"] <= self.tl_active_roi_y_max
            ):
                continue

            # Aktif ışık seçiminde yanlış arkadaki kırmızıyı abartmamak için
            # önce konum/güven/alan ağırlıklı seçiyoruz.
            center_score = 1.0 - min(1.0, abs(m["cx_ratio"] - 0.55) / 0.55)
            score = (
                state_conf * 3.0
                + det_conf * 2.0
                + m["area_ratio"] * 25.0
                + center_score
            )

            selected = dict(det)
            selected["state_confidence"] = state_conf
            selected["active_tl_score"] = score
            candidates.append((score, selected))

        if not candidates:
            return "unknown", None

        candidates.sort(key=lambda item: item[0], reverse=True)
        best = candidates[0][1]
        return str(best.get("traffic_light_state", "unknown")).lower().strip(), best

    def get_best_sign(self, detections, frame_width, frame_height):
        """
        En güvenilir/aktif trafik levhasını seçer.

        Eski mantık sadece güven skoruna bakıyordu. Bu da çok uzaktaki veya
        görüntünün köşesindeki levhayı karar sebebi yapabiliyordu.

        Yeni mantık:
        - traffic_sign label ister
        - YOLO + classifier eşiği ister
        - bbox alanı ve görüntüde yeterince aşağı/yakın olma şartı ister
        - STOP/SLOW levhalarına öncelik verir
        """
        candidates = []

        for det in detections:
            if det.get("label") != "traffic_sign":
                continue

            yolo_conf = float(det.get("confidence", 0.0))
            sign_type = str(det.get("sign_type", "unknown")).strip()

            if yolo_conf < self.traffic_sign_conf_threshold:
                continue

            if sign_type == "unknown":
                continue

            sign_conf = float(det.get("sign_confidence", yolo_conf))
            if sign_conf < self.sign_classifier_conf_threshold:
                continue

            m = self.bbox_metrics(det, frame_width, frame_height)
            if m is None:
                continue

            # Çok küçük/uzak levhalar karar üretmesin.
            if m["area_ratio"] < self.sign_min_area_ratio:
                continue

            # Tabela genelde yanlarda olur; merkez koridor şartı koymuyoruz.
            # Ama çok tepede kalan levha uzaktır, karar üretmesin.
            if m["bottom_ratio"] < self.sign_min_bottom_ratio:
                continue

            # Aşırı kenara yapışmış/kesilmiş false detection'ları azalt.
            if not (0.02 <= m["cx_ratio"] <= 0.98):
                continue

            selected = dict(det)
            selected.update({
                "cx_ratio": round(float(m["cx_ratio"]), 4),
                "cy_ratio": round(float(m["cy_ratio"]), 4),
                "bottom_ratio": round(float(m["bottom_ratio"]), 4),
                "area_ratio": round(float(m["area_ratio"]), 6),
                "bbox_width_ratio": round(float(m["w_ratio"]), 4),
                "bbox_height_ratio": round(float(m["h_ratio"]), 4),
            })

            center_score = 1.0 - min(1.0, abs(m["cx_ratio"] - 0.5) / 0.5)

            if sign_type in self.stop_signs:
                type_priority = 1.50
            elif sign_type in self.slow_signs:
                type_priority = 0.90
            elif sign_type in self.speed_limit_map:
                type_priority = 0.70
            elif sign_type in self.warning_signs:
                type_priority = 0.30
            else:
                type_priority = 0.10

            score = (
                sign_conf * 2.0
                + yolo_conf
                + m["bottom_ratio"] * 1.5
                + min(m["area_ratio"] * 120.0, 1.5)
                + center_score * self.sign_center_bonus
                + type_priority
            )

            selected["active_sign_score"] = round(float(score), 4)
            candidates.append((score, selected))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]

    def should_stop_for_stop_sign(self, active_sign):
        """
        DUR levhası için:
        - İlk görüldüğünde belirli süre STOP üretir.
        - Sonra cooldown süresince tekrar STOP üretmez.
        Böylece araç levhayı gördüğü yerde sonsuza kadar kilitlenmez.
        """
        now = time.time()

        if active_sign is None:
            return False, "no_stop_sign"

        sign_type = str(active_sign.get("sign_type", "unknown"))
        # FAR_STOP_SIGN_FILTER:
        # Logda araç çok uzakta, görüntünün üst kısmındaki minik/kararsız levhayı
        # dur gibi okuyup gereksiz STOP üretti.
        # DUR levhası ancak yakın/güvenilir/geometrik olarak yeterliyse STOP sayılır.
        if sign_type == "dur":
            try:
                sign_conf = float(active_sign.get("sign_confidence", active_sign.get("confidence", 0.0)) or 0.0)
            except Exception:
                sign_conf = 0.0

            try:
                bottom_ratio = float(active_sign.get("bottom_ratio", 0.0) or 0.0)
            except Exception:
                bottom_ratio = 0.0

            try:
                area_ratio = float(active_sign.get("area_ratio", 0.0) or 0.0)
            except Exception:
                area_ratio = 0.0

            # Bu eşikler özellikle false positive'i kesmek için:
            # - logdaki yanlış dur levhası görüntünün üst/uzak bölgesindeydi
            # - gerçek yakın DUR daha büyük ve daha aşağıda olur
            if sign_conf < 0.55:
                active_sign["stop_sign_rejected"] = "low_sign_conf"
                active_sign["stop_sign_reject_detail"] = f"conf={sign_conf:.3f}<0.55"
                return False, f"stop_sign_rejected_low_conf:dur:{sign_conf:.3f}"

            if bottom_ratio < 0.32:
                active_sign["stop_sign_rejected"] = "too_far_top"
                active_sign["stop_sign_reject_detail"] = f"bottom={bottom_ratio:.3f}<0.32"
                return False, f"stop_sign_rejected_too_far:dur:bottom={bottom_ratio:.3f}"

            if area_ratio < 0.00045:
                active_sign["stop_sign_rejected"] = "too_small"
                active_sign["stop_sign_reject_detail"] = f"area={area_ratio:.6f}<0.00045"
                return False, f"stop_sign_rejected_too_small:dur:area={area_ratio:.6f}"

        if now < self.stop_sign_hold_until:
            active_sign["stop_sign_state"] = "holding"
            active_sign["stop_sign_hold_left_s"] = round(self.stop_sign_hold_until - now, 2)
            return True, f"stop_sign_holding:{sign_type}"

        if now < self.stop_sign_cooldown_until:
            active_sign["stop_sign_state"] = "cooldown"
            active_sign["stop_sign_cooldown_left_s"] = round(self.stop_sign_cooldown_until - now, 2)
            return False, f"stop_sign_cooldown:{sign_type}"

        self.last_stop_sign_type = sign_type
        self.stop_sign_hold_until = now + self.stop_sign_hold_seconds
        self.stop_sign_cooldown_until = (
            now + self.stop_sign_hold_seconds + self.stop_sign_cooldown_seconds
        )

        active_sign["stop_sign_state"] = "new_stop"
        active_sign["stop_sign_hold_left_s"] = round(self.stop_sign_hold_seconds, 2)
        active_sign["stop_sign_cooldown_left_s"] = round(
            self.stop_sign_hold_seconds + self.stop_sign_cooldown_seconds,
            2,
        )

        return True, f"stop_sign_detected:{sign_type}"

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

    def build_rule_decision(
        self,
        vehicle_rule,
        traffic_light_state,
        traffic_light_det,
        active_sign,
        person_risk,
        active_person,
        person_hold_active,
    ):
        active_sign_type = None
        active_sign_name = None

        if active_sign is not None:
            active_sign_type = str(active_sign.get("sign_type", "unknown"))
            active_sign_name = active_sign.get("sign_name", active_sign_type)

        if active_sign_type in self.speed_limit_map:
            self.current_speed_limit = self.speed_limit_map[active_sign_type]
            self.last_speed_limit_sign = active_sign_type

        # Öncelik 1: kırmızı ışık / dur levhası / yakın yaya / çok yakın engel.
        if traffic_light_state == "red":
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": "red_light_detected",
            }

        if active_sign_type in self.stop_signs:
            should_stop, stop_reason = self.should_stop_for_stop_sign(active_sign)
            if should_stop:
                return {
                    "decision": "STOP",
                    "risk": "HIGH",
                    "target_speed": self.stop_speed,
                    "reason": stop_reason,
                }

        if person_risk == "near":
            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": "near_person_hold" if person_hold_active else "near_person_in_path",
            }

        if vehicle_rule["decision"] == "STOP":
            return vehicle_rule

        # Öncelik 2: sarı ışık, yavaş levhası, yol koridorunda uzak yaya, yakın araç.
        if traffic_light_state == "yellow":
            if self.yellow_as_slow:
                return {
                    "decision": "SLOW",
                    "risk": "MEDIUM",
                    "target_speed": self.slow_speed,
                    "reason": "yellow_light_detected",
                }

            return {
                "decision": "STOP",
                "risk": "HIGH",
                "target_speed": self.stop_speed,
                "reason": "yellow_light_stop_policy",
            }

        if active_sign_type in self.slow_signs:
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": f"slow_sign_detected:{active_sign_type}",
            }

        if person_risk == "far":
            return {
                "decision": "SLOW",
                "risk": "MEDIUM",
                "target_speed": self.slow_speed,
                "reason": "person_in_path_far",
            }

        if vehicle_rule["decision"] == "SLOW":
            return vehicle_rule

        # Şerit dışında kişi: durma değil, kontrollü sürüş.
        if person_risk in {"side", "side_near"}:
            if person_risk == "side_near":
                return {
                    "decision": "SLOW",
                    "risk": "MEDIUM",
                    "target_speed": self.slow_speed,
                    "reason": "side_person_near_slow",
                }

        # Hız limiti varsa uygula.
        # hiz_siniri_30 => target_speed_kmh=30.0, target_speed=8.33 m/s.
        if self.current_speed_limit is not None:
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.current_speed_limit,
                "reason": f"speed_limit_active:{self.last_speed_limit_sign}",
            }

        if active_sign_type in self.warning_signs:
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "reason": f"warning_sign_detected:{active_sign_type}",
            }

        if traffic_light_state == "green":
            return {
                "decision": "GO",
                "risk": "LOW",
                "target_speed": self.default_go_speed,
                "reason": "green_light_detected",
            }

        return vehicle_rule

    def stabilize_decision(self, output):
        now = time.time()

        if self.last_output is None:
            self.last_output = dict(output)
            self.last_output_time = now
            return output

        prev_decision = self.last_output.get("decision", "GO")
        age = now - self.last_output_time

        if prev_decision == "STOP" and output["decision"] != "STOP":
            if age < self.stop_hold_seconds:
                held = dict(output)
                held["decision"] = "STOP"
                held["risk"] = "HIGH"
                held["target_speed"] = self.stop_speed
                held["reason"] = f"decision_hold_after_stop:{self.last_output.get('reason')}"
                return held

        if prev_decision == "SLOW" and output["decision"] == "GO":
            if age < self.slow_hold_seconds:
                held = dict(output)
                held["decision"] = "SLOW"
                held["risk"] = "MEDIUM"
                held["target_speed"] = min(self.slow_speed, float(output.get("target_speed", self.slow_speed)))
                held["reason"] = f"decision_hold_after_slow:{self.last_output.get('reason')}"
                return held

        if output["decision"] != prev_decision or output.get("reason") != self.last_output.get("reason"):
            self.last_output_time = now

        self.last_output = dict(output)
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

        traffic_light_state, traffic_light_det = self.select_active_traffic_light(
            detections,
            frame_width,
            frame_height,
        )

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

        final_rule = self.build_rule_decision(
            vehicle_rule=vehicle_rule,
            traffic_light_state=traffic_light_state,
            traffic_light_det=traffic_light_det,
            active_sign=active_sign,
            person_risk=person_risk,
            active_person=active_person,
            person_hold_active=person_hold_active,
        )

        distance_est = vehicle_rule.get("distance_est", None)


        output = {
            "stamp": time.time(),
            "decision": final_rule["decision"],
            "risk": final_rule["risk"],
            # target_speed eski uyumluluk için m/s.
            # target_speed_kmh insan/yarışma/log tarafında ana hızdır.
            "target_speed": round(float(final_rule["target_speed"]), 3),
            "target_speed_kmh": round(self.mps_to_kmh(final_rule["target_speed"]), 1),
            "distance_est": round(float(distance_est), 2) if distance_est is not None else None,
            "front_vehicle": front_vehicle,
            "traffic_light_state": traffic_light_state,
            "traffic_light": traffic_light_det,
            "active_sign": active_sign,
            "person_risk": person_risk,
            "raw_person_risk": raw_person_risk,
            "person_hold_active": person_hold_active,
            "active_person": active_person,
            "speed_limit_active": self.last_speed_limit_sign,
            "reason": final_rule["reason"],
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
        output["target_speed_kmh"] = round(self.mps_to_kmh(output.get("target_speed", 0.0)), 1)
        output["speed_limit_active_kmh"] = round(self.mps_to_kmh(self.current_speed_limit), 1) if self.current_speed_limit is not None else None

        out_msg = String()
        out_msg.data = json.dumps(output, ensure_ascii=False)
        self.pub.publish(out_msg)

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


