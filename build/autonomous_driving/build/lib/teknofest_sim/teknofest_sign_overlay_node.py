import json
import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla


@dataclass
class VirtualSign:
    sign_type: str
    sign_name: str
    location: object
    yaw: float
    side_m: float
    forward_m: float


SIGN_SEQUENCE = [
    "yaya_gecidi",
    "ada_etrafinda_donunuz",
    "isikli_isaret_cihazi",
    "saga_donulmez",
    "sola_donulmez",
    "girisi_olmayan_yol",
    "sagdan_gidiniz",
    "soldan_gidiniz",
    "saga_mecburi_yon",
    "sola_mecburi_yon",
    "ileri_mecburi_yon",
    "ileri_ve_saga_mecburi_yon",
    "ileri_ve_sola_mecburi_yon",
    "ileriden_saga_mecburi_yon",
    "ileriden_sola_mecburi_yon",
    "serit_duzenleme_levhasi_sag",
    "serit_duzenleme_levhasi_sol",
    "iki_yonlu_yol",
    "park_etmek_yasaktir",
    "park_yeri",
    "tunel",
    "dur",
    "yol_ver",
    "dikkat",
    "okul_gecidi",
    "yol_calismasi",
    "hiz_siniri_20",
    "hiz_siniri_30",
    "hiz_siniri_40",
    "hiz_siniri_50",
]


PRETTY_NAMES = {
    "yaya_gecidi": "Yaya Geçidi",
    "ada_etrafinda_donunuz": "Ada Etrafında Dönünüz",
    "isikli_isaret_cihazi": "Işıklı İşaret Cihazı",
    "saga_donulmez": "Sağa Dönülmez",
    "sola_donulmez": "Sola Dönülmez",
    "girisi_olmayan_yol": "Girişi Olmayan Yol",
    "sagdan_gidiniz": "Sağdan Gidiniz",
    "soldan_gidiniz": "Soldan Gidiniz",
    "saga_mecburi_yon": "Sağa Mecburi Yön",
    "sola_mecburi_yon": "Sola Mecburi Yön",
    "ileri_mecburi_yon": "İleri Mecburi Yön",
    "ileri_ve_saga_mecburi_yon": "İleri ve Sağa Mecburi Yön",
    "ileri_ve_sola_mecburi_yon": "İleri ve Sola Mecburi Yön",
    "ileriden_saga_mecburi_yon": "İleriden Sağa Mecburi Yön",
    "ileriden_sola_mecburi_yon": "İleriden Sola Mecburi Yön",
    "serit_duzenleme_levhasi_sag": "Şerit Düzenleme Sağ",
    "serit_duzenleme_levhasi_sol": "Şerit Düzenleme Sol",
    "iki_yonlu_yol": "İki Yönlü Yol",
    "park_etmek_yasaktir": "Park Etmek Yasaktır",
    "park_yeri": "Park Yeri",
    "tunel": "Tünel",
    "dur": "Dur",
    "yol_ver": "Yol Ver",
    "dikkat": "Dikkat",
    "okul_gecidi": "Okul Geçidi",
    "yol_calismasi": "Yol Çalışması",
    "hiz_siniri_20": "Hız Sınırı 20",
    "hiz_siniri_30": "Hız Sınırı 30",
    "hiz_siniri_40": "Hız Sınırı 40",
    "hiz_siniri_50": "Hız Sınırı 50",
}


class TeknofestSignOverlayNode(Node):
    """
    Şartname tabelalarını CARLA kamera görüntüsüne sanal/perspektifli olarak basar.

    Neden böyle?
    - CARLA runtime'da custom texture'lı tabela spawn etmek pratik değil.
    - Unreal Editor ile custom prop üretmeden şartname görsellerini kamera görüntüsünde
      göstermek için en hızlı ve kontrollü yöntem bu overlay node'dur.
    - Karar/rota sistemi daha sonra /adas/perception/detections_json üzerinden bağlanabilir.
    """

    def __init__(self):
        super().__init__("teknofest_sign_overlay_node")

        self.declare_parameter("enabled", True)

        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)
        self.declare_parameter("ego_role_name", "ego_vehicle")

        self.declare_parameter("input_image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("output_image_topic", "/adas/camera/front/sign_overlay_image")
        self.declare_parameter("status_topic", "/adas/teknofest/sign_overlay_status")

        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 360)
        self.declare_parameter("camera_fov", 72.0)
        self.declare_parameter("camera_x", 1.6)
        self.declare_parameter("camera_y", 0.0)
        self.declare_parameter("camera_z", 2.25)
        self.declare_parameter("camera_pitch", -1.0)

        self.declare_parameter("sign_count", len(SIGN_SEQUENCE))
        self.declare_parameter("first_distance_m", 12.0)
        self.declare_parameter("distance_step_m", 7.0)
        self.declare_parameter("side_offset_m", 2.15)
        self.declare_parameter("sign_z_m", 1.55)
        self.declare_parameter("sign_world_height_m", 1.20)
        self.declare_parameter("min_render_distance_m", 3.0)
        self.declare_parameter("max_render_distance_m", 85.0)
        self.declare_parameter("route_waypoint_step_m", 5.0)
        self.declare_parameter("route_start_index", 3)
        self.declare_parameter("route_stride", 2)
        self.declare_parameter("draw_debug_label", True)
        self.declare_parameter("draw_world_debug", True)

        self.enabled = bool(self.get_parameter("enabled").value)
        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)

        self.input_image_topic = str(self.get_parameter("input_image_topic").value)
        self.output_image_topic = str(self.get_parameter("output_image_topic").value)
        self.status_topic = str(self.get_parameter("status_topic").value)

        self.camera_width = int(self.get_parameter("camera_width").value)
        self.camera_height = int(self.get_parameter("camera_height").value)
        self.camera_fov = float(self.get_parameter("camera_fov").value)
        self.camera_x = float(self.get_parameter("camera_x").value)
        self.camera_y = float(self.get_parameter("camera_y").value)
        self.camera_z = float(self.get_parameter("camera_z").value)
        self.camera_pitch = float(self.get_parameter("camera_pitch").value)

        self.sign_count = int(self.get_parameter("sign_count").value)
        self.first_distance_m = float(self.get_parameter("first_distance_m").value)
        self.distance_step_m = float(self.get_parameter("distance_step_m").value)
        self.side_offset_m = float(self.get_parameter("side_offset_m").value)
        self.sign_z_m = float(self.get_parameter("sign_z_m").value)
        self.sign_world_height_m = float(self.get_parameter("sign_world_height_m").value)
        self.min_render_distance_m = float(self.get_parameter("min_render_distance_m").value)
        self.max_render_distance_m = float(self.get_parameter("max_render_distance_m").value)
        self.route_waypoint_step_m = float(self.get_parameter("route_waypoint_step_m").value)
        self.route_start_index = int(self.get_parameter("route_start_index").value)
        self.route_stride = int(self.get_parameter("route_stride").value)
        self.draw_debug_label = bool(self.get_parameter("draw_debug_label").value)
        self.draw_world_debug = bool(self.get_parameter("draw_world_debug").value)

        self.bridge = CvBridge()

        self.fx, self.fy, self.cx, self.cy = self.compute_intrinsics(
            self.camera_width,
            self.camera_height,
            self.camera_fov,
        )

        self.carla = None
        self.client = None
        self.world = None
        self.map = None
        self.ego = None
        self.virtual_signs: List[VirtualSign] = []
        self.icons: Dict[str, np.ndarray] = {}

        self.pub = self.create_publisher(Image, self.output_image_topic, 10)
        self.status_pub = self.create_publisher(String, self.status_topic, 10)

        self.sub = self.create_subscription(
            Image,
            self.input_image_topic,
            self.image_callback,
            qos_profile_sensor_data,
        )

        self.connect_carla()
        self.icons = self.build_all_icons()
        self.spawn_virtual_signs()

        self.status_timer = self.create_timer(1.0, self.publish_status)

        self.get_logger().info("TEKNOFEST sign overlay node hazır.")
        self.get_logger().info(f"enabled={self.enabled}")
        self.get_logger().info(f"input={self.input_image_topic}")
        self.get_logger().info(f"output={self.output_image_topic}")
        self.get_logger().info(f"virtual_sign_count={len(self.virtual_signs)}")

    @staticmethod
    def compute_intrinsics(width: int, height: int, horizontal_fov_deg: float):
        fov_rad = math.radians(horizontal_fov_deg)
        fx = width / (2.0 * math.tan(fov_rad / 2.0))
        fy = fx
        cx = width / 2.0
        cy = height / 2.0
        return fx, fy, cx, cy

    def connect_carla(self):
        try:
            self.carla = load_carla(self.carla_root)
            self.client = self.carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
            self.ego = self.wait_for_ego()
            self.get_logger().info(f"CARLA bağlantısı hazır: {self.host}:{self.port}")
        except Exception as exc:
            self.get_logger().error(f"CARLA bağlantısı kurulamadı: {exc}")
            raise

    def wait_for_ego(self):
        for _ in range(150):
            for vehicle in self.world.get_actors().filter("vehicle.*"):
                if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                    self.get_logger().info(f"Ego bulundu: id={vehicle.id}, type={vehicle.type_id}")
                    return vehicle
            time.sleep(0.2)
        raise RuntimeError("Ego vehicle bulunamadı. Önce carla_world_manager_node çalışmalı.")

    def get_route_waypoints_ahead(self, count: int):
        ego_wp = self.map.get_waypoint(
            self.ego.get_location(),
            project_to_road=True,
            lane_type=self.carla.LaneType.Driving,
        )

        if ego_wp is None:
            return []

        waypoints = [ego_wp]
        current = ego_wp

        for _ in range(count - 1):
            nxt = current.next(self.route_waypoint_step_m)
            if not nxt:
                break

            # Rastgele dal seçmek yerine en düz devam eden yolu seçmeye çalış.
            nxt_sorted = sorted(
                nxt,
                key=lambda wp: abs(wp.transform.rotation.yaw - current.transform.rotation.yaw),
            )
            current = nxt_sorted[0]
            waypoints.append(current)

        return waypoints

    def spawn_virtual_signs(self):
        total = min(self.sign_count, len(SIGN_SEQUENCE))
        needed_wp = self.route_start_index + total * max(1, self.route_stride) + 5
        waypoints = self.get_route_waypoints_ahead(count=max(needed_wp, 80))

        if len(waypoints) < 5:
            self.get_logger().warning("Yeterli waypoint bulunamadı; tabela yerleşimi yapılamadı.")
            return

        self.virtual_signs.clear()

        for i, sign_type in enumerate(SIGN_SEQUENCE[:total]):
            wp_index = min(
                len(waypoints) - 1,
                self.route_start_index + i * max(1, self.route_stride),
            )
            wp = waypoints[wp_index]
            right = wp.transform.get_right_vector()

            # Çoğu gerçek tabelada sağ tarafta olur; ama şartname çeşitlilik dediği için
            # bazılarını sol tarafa da koyuyoruz.
            side_sign = 1.0 if i % 3 != 1 else -1.0
            side_m = self.side_offset_m * side_sign

            loc = wp.transform.location + self.carla.Location(
                x=right.x * side_m,
                y=right.y * side_m,
                z=self.sign_z_m,
            )

            sign = VirtualSign(
                sign_type=sign_type,
                sign_name=PRETTY_NAMES.get(sign_type, sign_type),
                location=loc,
                yaw=float(wp.transform.rotation.yaw),
                side_m=side_m,
                forward_m=float(wp_index * self.route_waypoint_step_m),
            )
            self.virtual_signs.append(sign)

            if self.draw_world_debug:
                try:
                    self.world.debug.draw_string(
                        loc + self.carla.Location(z=1.5),
                        f"{i + 1}:{sign_type}",
                        draw_shadow=True,
                        color=self.carla.Color(0, 255, 255),
                        life_time=9999.0,
                        persistent_lines=True,
                    )
                except Exception:
                    pass

        self.get_logger().info(
            "Şartname tabela overlay yerleşimi tamamlandı: "
            + ", ".join([s.sign_type for s in self.virtual_signs])
        )

    # -----------------------------
    # Icon drawing helpers
    # -----------------------------

    def blank_icon(self, size=256, bg=(255, 255, 255)):
        img = np.zeros((size, size, 3), dtype=np.uint8)
        img[:] = bg
        return img

    def put_center_text(self, img, text, y, scale=1.0, color=(0, 0, 0), thickness=2):
        text = str(text)
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        x = max(0, int((img.shape[1] - tw) / 2))
        cv2.putText(img, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)

    def draw_red_triangle(self, title="!", subtitle=None):
        img = self.blank_icon()
        pts = np.array([[128, 22], [28, 218], [228, 218]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))
        cv2.polylines(img, [pts], True, (0, 0, 230), 18, cv2.LINE_AA)
        self.put_center_text(img, title, 154, 2.4, (0, 0, 0), 5)
        if subtitle:
            self.put_center_text(img, subtitle, 205, 0.55, (0, 0, 0), 2)
        return img

    def draw_red_circle(self, title, subtitle=None, slash=False):
        img = self.blank_icon()
        cv2.circle(img, (128, 128), 104, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (128, 128), 78, (255, 255, 255), -1, cv2.LINE_AA)
        self.put_center_text(img, title, 138, 1.65, (0, 0, 0), 4)
        if subtitle:
            self.put_center_text(img, subtitle, 177, 0.55, (0, 0, 0), 2)
        if slash:
            cv2.line(img, (58, 198), (198, 58), (0, 0, 230), 18, cv2.LINE_AA)
        return img

    def draw_blue_circle(self, title, subtitle=None):
        img = self.blank_icon()
        cv2.circle(img, (128, 128), 106, (210, 90, 0), -1, cv2.LINE_AA)
        self.put_center_text(img, title, 132, 1.45, (255, 255, 255), 4)
        if subtitle:
            self.put_center_text(img, subtitle, 174, 0.55, (255, 255, 255), 2)
        return img

    def draw_blue_square(self, title, subtitle=None):
        img = self.blank_icon()
        cv2.rectangle(img, (28, 28), (228, 228), (210, 90, 0), -1)
        cv2.rectangle(img, (28, 28), (228, 228), (255, 255, 255), 6)
        self.put_center_text(img, title, 136, 1.55, (255, 255, 255), 4)
        if subtitle:
            self.put_center_text(img, subtitle, 178, 0.55, (255, 255, 255), 2)
        return img

    def draw_stop(self):
        img = self.blank_icon()
        pts = []
        for k in range(8):
            ang = math.radians(22.5 + k * 45)
            pts.append((int(128 + 105 * math.cos(ang)), int(128 + 105 * math.sin(ang))))
        pts = np.array(pts, dtype=np.int32)
        cv2.fillPoly(img, [pts], (0, 0, 220))
        cv2.polylines(img, [pts], True, (255, 255, 255), 7, cv2.LINE_AA)
        self.put_center_text(img, "DUR", 148, 1.75, (255, 255, 255), 5)
        return img

    def draw_yol_ver(self):
        img = self.blank_icon()
        pts = np.array([[128, 226], [28, 42], [228, 42]], dtype=np.int32)
        cv2.fillPoly(img, [pts], (255, 255, 255))
        cv2.polylines(img, [pts], True, (0, 0, 230), 18, cv2.LINE_AA)
        self.put_center_text(img, "YOL", 116, 1.05, (0, 0, 0), 3)
        self.put_center_text(img, "VER", 160, 1.05, (0, 0, 0), 3)
        return img

    def draw_no_entry(self):
        img = self.blank_icon()
        cv2.circle(img, (128, 128), 106, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.rectangle(img, (48, 105), (208, 151), (255, 255, 255), -1)
        return img

    def draw_parking(self):
        img = self.draw_blue_square("P")
        return img

    def draw_no_parking(self):
        img = self.draw_red_circle("P", slash=True)
        return img

    def draw_speed(self, speed):
        return self.draw_red_circle(str(speed))

    def draw_traffic_light_warning(self):
        img = self.draw_red_triangle("", None)
        x = 128
        cv2.rectangle(img, (100, 70), (156, 190), (25, 25, 25), -1)
        cv2.circle(img, (128, 95), 16, (0, 0, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (128, 130), 16, (0, 230, 230), -1, cv2.LINE_AA)
        cv2.circle(img, (128, 165), 16, (0, 190, 0), -1, cv2.LINE_AA)
        return img

    def draw_lane_regulation(self, direction="sag"):
        img = self.draw_blue_square("", None)
        # Basit şerit düzenleme sembolü
        cv2.line(img, (78, 210), (78, 54), (255, 255, 255), 10, cv2.LINE_AA)
        cv2.line(img, (128, 210), (128, 54), (255, 255, 255), 10, cv2.LINE_AA)
        cv2.line(img, (178, 210), (178, 54), (255, 255, 255), 10, cv2.LINE_AA)

        if direction == "sag":
            cv2.arrowedLine(img, (82, 170), (132, 104), (255, 255, 255), 10, cv2.LINE_AA, tipLength=0.35)
        else:
            cv2.arrowedLine(img, (174, 170), (124, 104), (255, 255, 255), 10, cv2.LINE_AA, tipLength=0.35)

        return img

    def draw_tunnel(self):
        img = self.draw_blue_square("", None)
        cv2.rectangle(img, (62, 110), (194, 208), (255, 255, 255), -1)
        cv2.ellipse(img, (128, 112), (66, 68), 180, 0, 180, (255, 255, 255), -1)
        cv2.rectangle(img, (88, 112), (168, 208), (210, 90, 0), -1)
        return img

    def draw_pedestrian(self):
        img = self.draw_blue_square("", None)
        cv2.rectangle(img, (38, 38), (218, 218), (255, 255, 255), -1)
        cv2.rectangle(img, (48, 48), (208, 208), (210, 90, 0), -1)
        self.put_center_text(img, "YAYA", 118, 1.1, (255, 255, 255), 3)
        cv2.line(img, (62, 180), (194, 180), (255, 255, 255), 5)
        cv2.line(img, (78, 160), (178, 160), (255, 255, 255), 5)
        return img

    def build_all_icons(self):
        icons = {
            "dur": self.draw_stop(),
            "yol_ver": self.draw_yol_ver(),
            "girisi_olmayan_yol": self.draw_no_entry(),
            "dikkat": self.draw_red_triangle("!"),
            "okul_gecidi": self.draw_red_triangle("OKUL", "GECIDI"),
            "yol_calismasi": self.draw_red_triangle("IS", "YOL"),
            "yaya_gecidi": self.draw_pedestrian(),
            "isikli_isaret_cihazi": self.draw_traffic_light_warning(),
            "park_yeri": self.draw_parking(),
            "park_etmek_yasaktir": self.draw_no_parking(),
            "hiz_siniri_20": self.draw_speed(20),
            "hiz_siniri_30": self.draw_speed(30),
            "hiz_siniri_40": self.draw_speed(40),
            "hiz_siniri_50": self.draw_speed(50),
            "saga_donulmez": self.draw_red_circle("R", slash=True),
            "sola_donulmez": self.draw_red_circle("L", slash=True),
            "sagdan_gidiniz": self.draw_blue_circle("SAG"),
            "soldan_gidiniz": self.draw_blue_circle("SOL"),
            "saga_mecburi_yon": self.draw_blue_circle("SAG"),
            "sola_mecburi_yon": self.draw_blue_circle("SOL"),
            "ileri_mecburi_yon": self.draw_blue_circle("ILERI"),
            "ileri_ve_saga_mecburi_yon": self.draw_blue_circle("I+SAG"),
            "ileri_ve_sola_mecburi_yon": self.draw_blue_circle("I+SOL"),
            "ileriden_saga_mecburi_yon": self.draw_blue_circle("ILER", "SAG"),
            "ileriden_sola_mecburi_yon": self.draw_blue_circle("ILER", "SOL"),
            "ada_etrafinda_donunuz": self.draw_blue_circle("ADA"),
            "iki_yonlu_yol": self.draw_red_triangle("<->"),
            "serit_duzenleme_levhasi_sag": self.draw_lane_regulation("sag"),
            "serit_duzenleme_levhasi_sol": self.draw_lane_regulation("sol"),
            "tunel": self.draw_tunnel(),
        }

        return icons

    # -----------------------------
    # Projection + rendering
    # -----------------------------

    def camera_pose_from_ego(self):
        tf = self.ego.get_transform()

        fwd = tf.get_forward_vector()
        right = tf.get_right_vector()

        cam_loc = tf.location + self.carla.Location(
            x=fwd.x * self.camera_x + right.x * self.camera_y,
            y=fwd.y * self.camera_x + right.y * self.camera_y,
            z=self.camera_z,
        )

        return tf, cam_loc, fwd, right

    def project_world_to_image(self, loc, frame_w, frame_h):
        ego_tf, cam_loc, fwd, right = self.camera_pose_from_ego()

        dx = loc.x - cam_loc.x
        dy = loc.y - cam_loc.y
        dz = loc.z - cam_loc.z

        x_cam = dx * fwd.x + dy * fwd.y
        y_cam = dx * right.x + dy * right.y

        # Kamera pitch'i küçük olduğu için basit düzeltme yeterli.
        pitch = math.radians(self.camera_pitch)
        z_cam = dz * math.cos(pitch) - x_cam * math.sin(pitch)
        x_cam_p = x_cam * math.cos(pitch) + dz * math.sin(pitch)

        if x_cam_p <= self.min_render_distance_m or x_cam_p > self.max_render_distance_m:
            return None

        u = self.cx + (y_cam * self.fx / x_cam_p)
        v = self.cy - (z_cam * self.fy / x_cam_p)

        if u < -frame_w * 0.3 or u > frame_w * 1.3 or v < -frame_h * 0.5 or v > frame_h * 1.5:
            return None

        size_px = int(max(18, min(180, self.sign_world_height_m * self.fy / x_cam_p)))
        return int(u), int(v), size_px, float(x_cam_p)

    def paste_icon(self, frame, icon, center_x, center_y, size_px, label):
        if icon is None or icon.size == 0:
            return

        h, w = frame.shape[:2]
        size_px = int(size_px)

        if size_px <= 8:
            return

        icon_resized = cv2.resize(icon, (size_px, size_px), interpolation=cv2.INTER_AREA)

        # Tabela direği
        pole_top = center_y + size_px // 2
        pole_bot = min(h - 1, pole_top + int(size_px * 1.35))
        if 0 <= center_x < w and pole_top < h:
            cv2.line(
                frame,
                (center_x, max(0, pole_top)),
                (center_x, pole_bot),
                (45, 45, 45),
                max(2, size_px // 18),
                cv2.LINE_AA,
            )

        x1 = int(center_x - size_px / 2)
        y1 = int(center_y - size_px / 2)
        x2 = x1 + size_px
        y2 = y1 + size_px

        if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h:
            return

        src_x1 = max(0, -x1)
        src_y1 = max(0, -y1)
        dst_x1 = max(0, x1)
        dst_y1 = max(0, y1)

        dst_x2 = min(w, x2)
        dst_y2 = min(h, y2)

        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)

        roi = frame[dst_y1:dst_y2, dst_x1:dst_x2]
        icon_crop = icon_resized[src_y1:src_y2, src_x1:src_x2]

        if roi.size == 0 or icon_crop.size == 0:
            return

        # Hafif alpha ile gerçek sahneye yedir.
        alpha = 0.96
        blended = cv2.addWeighted(icon_crop, alpha, roi, 1.0 - alpha, 0)
        frame[dst_y1:dst_y2, dst_x1:dst_x2] = blended

        if self.draw_debug_label:
            text = label[:22]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.32, min(0.55, size_px / 150.0))
            thickness = 1
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            tx = max(0, min(w - tw - 2, x1))
            ty = max(th + 2, min(h - 4, y2 + th + 4))
            cv2.rectangle(frame, (tx, ty - th - 3), (tx + tw + 4, ty + 3), (245, 245, 245), -1)
            cv2.putText(frame, text, (tx + 2, ty), font, scale, (0, 0, 0), thickness, cv2.LINE_AA)

    def image_callback(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"cv_bridge image parse hatası: {exc}")
            return

        if self.enabled:
            h, w = frame.shape[:2]

            # Uzak tabela arkada kalmasın diye yakından uzağa değil,
            # uzaktan yakına çiziyoruz; yakındaki üstte kalır.
            projections = []
            for sign in self.virtual_signs:
                projected = self.project_world_to_image(sign.location, w, h)
                if projected is None:
                    continue

                u, v, size_px, dist_m = projected
                projections.append((dist_m, sign, u, v, size_px))

            projections.sort(key=lambda x: x[0], reverse=True)

            for _, sign, u, v, size_px in projections:
                icon = self.icons.get(sign.sign_type)
                self.paste_icon(frame, icon, u, v, size_px, sign.sign_type)

            cv2.putText(
                frame,
                f"TEKNOFEST SIGN OVERLAY | visible={len(projections)} total={len(self.virtual_signs)}",
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        out = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        out.header = msg.header
        out.header.frame_id = "front_camera_with_teknofest_signs"
        self.pub.publish(out)

    def publish_status(self):
        signs = []
        for i, s in enumerate(self.virtual_signs):
            alive_info = {
                "idx": i,
                "sign_type": s.sign_type,
                "sign_name": s.sign_name,
                "x": round(float(s.location.x), 3),
                "y": round(float(s.location.y), 3),
                "z": round(float(s.location.z), 3),
                "side_m": round(float(s.side_m), 3),
                "forward_m": round(float(s.forward_m), 3),
            }
            signs.append(alive_info)

        payload = {
            "stamp": round(time.time(), 3),
            "enabled": self.enabled,
            "input_image_topic": self.input_image_topic,
            "output_image_topic": self.output_image_topic,
            "sign_count": len(self.virtual_signs),
            "signs": signs,
        }

        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.status_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestSignOverlayNode()

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
