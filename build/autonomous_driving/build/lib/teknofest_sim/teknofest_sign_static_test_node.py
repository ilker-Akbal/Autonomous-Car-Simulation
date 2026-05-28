import os
import time
import random
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class TeknofestSignStaticTestNode(Node):
    def __init__(self):
        super().__init__("teknofest_sign_static_test_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("fps", 5.0)
        self.declare_parameter(
            "raw_sign_dir",
            "/home/huseyindgn/Masaüstü/Autonomous-Driving-Perception-and-Decision-System/autonomous_driving/sign_classifier/dataset_v2/raw",
        )

        self.image_topic = self.get_parameter("image_topic").value
        self.fps = float(self.get_parameter("fps").value)
        self.raw_sign_dir = Path(self.get_parameter("raw_sign_dir").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(Image, self.image_topic, 10)

        self.sign_classes = [
            "dur",
            "yol_ver",
            "girisi_olmayan_yol",
            "saga_donulmez",
            "sola_donulmez",
            "park_yeri",
            "park_etmek_yasaktir",
            "yaya_gecidi",
            "isikli_isaret_cihazi",
            "hiz_siniri_20",
            "hiz_siniri_30",
            "hiz_siniri_40",
            "hiz_siniri_50",
            "sagdan_gidiniz",
            "soldan_gidiniz",
            "saga_mecburi_yon",
            "sola_mecburi_yon",
            "ileri_mecburi_yon",
            "ileri_ve_saga_mecburi_yon",
            "ileri_ve_sola_mecburi_yon",
            "ada_etrafinda_donunuz",
            "iki_yonlu_yol",
            "tunel",
            "dikkat",
            "okul_gecidi",
            "yol_calismasi",
        ]

        self.sign_images = self.load_sign_images()
        self.frame = self.build_scene()

        self.timer = self.create_timer(1.0 / max(1.0, self.fps), self.publish_frame)

        self.get_logger().info(f"Sign static test publisher başladı: {self.image_topic}")
        self.get_logger().info(f"Raw sign dir: {self.raw_sign_dir}")
        self.get_logger().info(f"Loaded signs: {list(self.sign_images.keys())}")

    def load_sign_images(self):
        exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        result = {}

        for cls in self.sign_classes:
            cls_dir = self.raw_sign_dir / cls
            if not cls_dir.exists():
                continue

            files = [
                p for p in sorted(cls_dir.iterdir())
                if p.is_file() and p.suffix.lower() in exts
            ]

            if not files:
                continue

            # Her sınıftan ilk düzgün okunabilen görseli al.
            for fp in files:
                img = cv2.imread(str(fp), cv2.IMREAD_COLOR)
                if img is not None and img.size > 0:
                    result[cls] = img
                    break

        return result

    def draw_road_background(self, frame):
        h, w = frame.shape[:2]

        # Gökyüzü
        frame[:] = (190, 210, 230)

        # Zemin / yol
        road = np.array([
            [int(w * 0.28), h],
            [int(w * 0.43), int(h * 0.36)],
            [int(w * 0.57), int(h * 0.36)],
            [int(w * 0.72), h],
        ], dtype=np.int32)

        cv2.fillPoly(frame, [road], (55, 55, 55))

        # Yol kenarı yeşil alan
        left_grass = np.array([
            [0, h],
            [0, int(h * 0.45)],
            [int(w * 0.43), int(h * 0.36)],
            [int(w * 0.28), h],
        ], dtype=np.int32)

        right_grass = np.array([
            [w, h],
            [w, int(h * 0.45)],
            [int(w * 0.57), int(h * 0.36)],
            [int(w * 0.72), h],
        ], dtype=np.int32)

        cv2.fillPoly(frame, [left_grass], (70, 130, 70))
        cv2.fillPoly(frame, [right_grass], (70, 130, 70))

        # Şerit çizgileri
        for i in range(8):
            y1 = int(h * 0.42 + i * h * 0.07)
            y2 = y1 + int(h * 0.035)

            x_mid1 = int(w * 0.50)
            x_mid2 = int(w * 0.50)

            cv2.line(frame, (x_mid1, y1), (x_mid2, y2), (240, 240, 240), 5)

        return frame

    def paste_sign(self, frame, sign_img, x, y, size, cls_name):
        h, w = frame.shape[:2]

        if sign_img is None or sign_img.size == 0:
            return

        # Kareye yakın crop/pad
        src = sign_img.copy()
        sh, sw = src.shape[:2]

        scale = size / max(1, max(sw, sh))
        nw = max(8, int(sw * scale))
        nh = max(8, int(sh * scale))

        src = cv2.resize(src, (nw, nh), interpolation=cv2.INTER_AREA)

        # Beyaz tabela paneli
        pad = max(4, int(size * 0.08))
        panel_w = nw + pad * 2
        panel_h = nh + pad * 2

        px1 = int(x - panel_w / 2)
        py1 = int(y - panel_h / 2)
        px2 = px1 + panel_w
        py2 = py1 + panel_h

        if px1 < 0 or py1 < 0 or px2 >= w or py2 >= h:
            return

        # Direk
        pole_x = int(x)
        cv2.line(
            frame,
            (pole_x, py2),
            (pole_x, min(h - 1, py2 + int(size * 1.1))),
            (40, 40, 40),
            5,
        )

        cv2.rectangle(frame, (px1, py1), (px2, py2), (245, 245, 245), -1)
        cv2.rectangle(frame, (px1, py1), (px2, py2), (25, 25, 25), 2)

        ix1 = px1 + pad
        iy1 = py1 + pad
        ix2 = ix1 + nw
        iy2 = iy1 + nh

        frame[iy1:iy2, ix1:ix2] = src

        # Küçük debug etiketi. Perception bunu bbox dışında görsün diye alta koyuyoruz.
        cv2.putText(
            frame,
            cls_name[:18],
            (px1, min(h - 5, py2 + 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    def build_scene(self):
        w, h = 1280, 720
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame = self.draw_road_background(frame)

        # Tabelaları iç içe koymamak için yol boyunca sağ/sol sıra.
        layout = [
            ("dur", 180, 155, 78),
            ("yol_ver", 355, 155, 78),
            ("girisi_olmayan_yol", 535, 155, 78),
            ("saga_donulmez", 720, 155, 78),
            ("sola_donulmez", 900, 155, 78),
            ("park_yeri", 1080, 155, 78),

            ("park_etmek_yasaktir", 170, 330, 88),
            ("yaya_gecidi", 350, 330, 88),
            ("isikli_isaret_cihazi", 535, 330, 88),
            ("hiz_siniri_20", 720, 330, 88),
            ("hiz_siniri_30", 900, 330, 88),
            ("hiz_siniri_40", 1080, 330, 88),

            ("hiz_siniri_50", 170, 530, 100),
            ("sagdan_gidiniz", 350, 530, 100),
            ("soldan_gidiniz", 535, 530, 100),
            ("saga_mecburi_yon", 720, 530, 100),
            ("sola_mecburi_yon", 900, 530, 100),
            ("ileri_mecburi_yon", 1080, 530, 100),

            ("ileri_ve_saga_mecburi_yon", 250, 660, 76),
            ("ileri_ve_sola_mecburi_yon", 455, 660, 76),
            ("ada_etrafinda_donunuz", 660, 660, 76),
            ("iki_yonlu_yol", 865, 660, 76),
            ("tunel", 1070, 660, 76),
        ]

        for cls, x, y, size in layout:
            img = self.sign_images.get(cls)
            if img is None:
                self.get_logger().warning(f"Sign image yok, atlanıyor: {cls}")
                continue

            self.paste_sign(frame, img, x, y, size, cls)

        # Başlık
        cv2.rectangle(frame, (0, 0), (1280, 42), (20, 20, 20), -1)
        cv2.putText(
            frame,
            "TEKNOFEST STATIC TRAFFIC SIGN TEST - ego stationary / signs spaced",
            (20, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        return frame

    def publish_frame(self):
        # Hafif parlaklık titreşimi ekle; görüntü sabit ama model farklı frame görsün.
        frame = self.frame.copy()

        alpha = 1.0 + random.uniform(-0.02, 0.02)
        beta = random.uniform(-3, 3)
        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)

        msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "sign_static_test_camera"
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = TeknofestSignStaticTestNode()

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
