import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import String


class LaneAssistNode(Node):
    def __init__(self):
        super().__init__("lane_assist_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("lane_topic", "/adas/lane/assist")
        self.declare_parameter("annotated_topic", "/adas/lane/annotated_image")
        self.declare_parameter("publish_annotated", True)

        self.declare_parameter("roi_top_ratio", 0.45)
        self.declare_parameter("canny_low", 60)
        self.declare_parameter("canny_high", 160)
        self.declare_parameter("hough_threshold", 35)
        self.declare_parameter("min_line_length", 35)
        self.declare_parameter("max_line_gap", 50)

        self.declare_parameter("lane_steer_gain", 0.35)
        self.declare_parameter("max_lane_steer", 0.25)
        self.declare_parameter("ema_alpha", 0.65)

        self.declare_parameter("min_lane_width_ratio", 0.28)
        self.declare_parameter("max_lane_width_ratio", 0.90)

        self.image_topic = self.get_parameter("image_topic").value
        self.lane_topic = self.get_parameter("lane_topic").value
        self.annotated_topic = self.get_parameter("annotated_topic").value
        self.publish_annotated = bool(self.get_parameter("publish_annotated").value)

        self.roi_top_ratio = float(self.get_parameter("roi_top_ratio").value)
        self.canny_low = int(self.get_parameter("canny_low").value)
        self.canny_high = int(self.get_parameter("canny_high").value)
        self.hough_threshold = int(self.get_parameter("hough_threshold").value)
        self.min_line_length = int(self.get_parameter("min_line_length").value)
        self.max_line_gap = int(self.get_parameter("max_line_gap").value)

        self.lane_steer_gain = float(self.get_parameter("lane_steer_gain").value)
        self.max_lane_steer = float(self.get_parameter("max_lane_steer").value)
        self.ema_alpha = float(self.get_parameter("ema_alpha").value)

        self.min_lane_width_ratio = float(self.get_parameter("min_lane_width_ratio").value)
        self.max_lane_width_ratio = float(self.get_parameter("max_lane_width_ratio").value)

        self.bridge = CvBridge()

        self.prev_offset = None
        self.prev_lane_steer = None

        self.pub = self.create_publisher(String, self.lane_topic, 10)
        self.ann_pub = self.create_publisher(Image, self.annotated_topic, 10) if self.publish_annotated else None

        self.create_subscription(
            Image,
            self.image_topic,
            self.image_cb,
            qos_profile_sensor_data,
        )

        self.get_logger().info(
            f"LaneAssistNode hazır. image={self.image_topic}, lane={self.lane_topic}"
        )

    def clamp(self, value, mn, mx):
        return max(mn, min(mx, float(value)))

    def fit_side_line(self, lines, side):
        points_x = []
        points_y = []

        for x1, y1, x2, y2 in lines:
            dx = x2 - x1
            dy = y2 - y1

            if abs(dx) < 1:
                continue

            slope = dy / dx

            # Image coordinate: y aşağı doğru artar.
            # Sol çizgi genelde negatif slope, sağ çizgi pozitif slope.
            if side == "left" and slope >= -0.35:
                continue
            if side == "right" and slope <= 0.35:
                continue

            if abs(slope) > 3.5:
                continue

            points_x.extend([x1, x2])
            points_y.extend([y1, y2])

        if len(points_x) < 4:
            return None

        # x = a*y + b fit ediyoruz. Bu, dikeye yakın çizgilerde daha stabil.
        a, b = np.polyfit(points_y, points_x, 1)
        return float(a), float(b), len(points_x)

    def x_at_y(self, fit, y):
        a, b, _ = fit
        return a * y + b

    def process_lane(self, frame):
        h, w = frame.shape[:2]
        roi_y0 = int(h * self.roi_top_ratio)
        roi = frame[roi_y0:h, 0:w]
        rh = roi.shape[0]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        hls = cv2.cvtColor(roi, cv2.COLOR_BGR2HLS)

        # Beyaz ve sarı şerit maskesi
        white_mask = cv2.inRange(hls, np.array([0, 150, 0]), np.array([180, 255, 255]))
        yellow_mask = cv2.inRange(hls, np.array([12, 40, 60]), np.array([45, 230, 255]))
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)

        masked_gray = cv2.bitwise_and(blur, blur, mask=color_mask)
        edges_color = cv2.Canny(masked_gray, self.canny_low, self.canny_high)
        edges_gray = cv2.Canny(blur, self.canny_low, self.canny_high)
        edges = cv2.bitwise_or(edges_color, edges_gray)

        # Alt trapez ROI
        poly = np.array([[
            (int(w * 0.05), rh),
            (int(w * 0.38), int(rh * 0.35)),
            (int(w * 0.62), int(rh * 0.35)),
            (int(w * 0.95), rh),
        ]], dtype=np.int32)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, poly, 255)
        edges = cv2.bitwise_and(edges, mask)

        raw_lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        lines = []
        if raw_lines is not None:
            for l in raw_lines:
                x1, y1, x2, y2 = l[0]
                lines.append((int(x1), int(y1), int(x2), int(y2)))

        left_fit = self.fit_side_line(lines, "left")
        right_fit = self.fit_side_line(lines, "right")

        y_bottom = rh - 1
        y_top = int(rh * 0.42)

        lane_detected = False
        confidence = 0.0
        reason = "no_lane"
        lane_center_x = None
        left_bottom = None
        right_bottom = None

        if left_fit is not None and right_fit is not None:
            left_bottom = self.x_at_y(left_fit, y_bottom)
            right_bottom = self.x_at_y(right_fit, y_bottom)

            if left_bottom > right_bottom:
                left_bottom, right_bottom = right_bottom, left_bottom

            lane_width = right_bottom - left_bottom
            lane_width_ratio = lane_width / max(w, 1)

            if self.min_lane_width_ratio <= lane_width_ratio <= self.max_lane_width_ratio:
                lane_center_x = (left_bottom + right_bottom) / 2.0
                lane_detected = True

                point_score = min((left_fit[2] + right_fit[2]) / 20.0, 1.0)
                width_score = 1.0 - min(abs(lane_width_ratio - 0.48) / 0.48, 1.0)
                confidence = self.clamp(0.55 + 0.30 * point_score + 0.15 * width_score, 0.0, 1.0)
                reason = "both_lanes_ok"
            else:
                reason = f"bad_lane_width_ratio:{lane_width_ratio:.3f}"

        # Tek çizgi fallback. Varsayılan güveni düşük. Route agent default 0.60 üstünü kullanır.
        elif left_fit is not None:
            left_bottom = self.x_at_y(left_fit, y_bottom)
            expected_width = w * 0.48
            lane_center_x = left_bottom + expected_width / 2.0
            lane_detected = True
            confidence = 0.50
            reason = "left_only_low_conf"

        elif right_fit is not None:
            right_bottom = self.x_at_y(right_fit, y_bottom)
            expected_width = w * 0.48
            lane_center_x = right_bottom - expected_width / 2.0
            lane_detected = True
            confidence = 0.50
            reason = "right_only_low_conf"

        if lane_detected and lane_center_x is not None:
            image_center_x = w / 2.0

            # Pozitif offset: şerit merkezi görüntüde sağda => araç sola kayık => sağa kır.
            offset_px = lane_center_x - image_center_x
            offset_norm = offset_px / max(w / 2.0, 1.0)
            offset_norm = self.clamp(offset_norm, -0.35, 0.35)

            lane_steer = self.clamp(
                self.lane_steer_gain * offset_norm,
                -self.max_lane_steer,
                self.max_lane_steer,
            )

            if self.prev_offset is not None:
                a = self.ema_alpha
                offset_norm = a * self.prev_offset + (1.0 - a) * offset_norm
                lane_steer = a * self.prev_lane_steer + (1.0 - a) * lane_steer

            self.prev_offset = offset_norm
            self.prev_lane_steer = lane_steer
        else:
            offset_px = None

            # Şerit kısa süre kaybolursa eski EMA bilgisini hemen silme.
            if self.prev_offset is not None:
                offset_norm = self.prev_offset * 0.92
                lane_steer = self.prev_lane_steer * 0.92
            else:
                offset_norm = 0.0
                lane_steer = 0.0

        debug = {
            "roi_y0": roi_y0,
            "line_count": len(lines),
            "left_found": left_fit is not None,
            "right_found": right_fit is not None,
            "left_bottom_x": None if left_bottom is None else round(float(left_bottom), 2),
            "right_bottom_x": None if right_bottom is None else round(float(right_bottom), 2),
            "lane_center_x": None if lane_center_x is None else round(float(lane_center_x), 2),
            "reason": reason,
        }

        payload = {
            "stamp": time.time(),
            "lane_detected": bool(lane_detected),
            "confidence": round(float(confidence), 3),
            "offset_px": None if offset_px is None else round(float(offset_px), 2),
            "offset_norm": round(float(offset_norm), 4),
            "lane_steer": round(float(lane_steer), 4),
            "image_width": int(w),
            "image_height": int(h),
            "debug": debug,
        }

        annotated = None
        if self.publish_annotated:
            annotated = frame.copy()
            cv2.line(annotated, (w // 2, h), (w // 2, roi_y0), (255, 255, 255), 2)

            if lane_center_x is not None:
                cx = int(lane_center_x)
                cv2.line(annotated, (cx, h), (cx, roi_y0), (0, 255, 255), 2)

            for x1, y1, x2, y2 in lines:
                cv2.line(
                    annotated,
                    (x1, y1 + roi_y0),
                    (x2, y2 + roi_y0),
                    (0, 255, 0),
                    2,
                )

            txt = f"lane={lane_detected} conf={confidence:.2f} steer={lane_steer:.3f} {reason}"
            cv2.putText(
                annotated,
                txt,
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

        return payload, annotated

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            payload, annotated = self.process_lane(frame)

            out = String()
            out.data = json.dumps(payload, ensure_ascii=False)
            self.pub.publish(out)

            if self.publish_annotated and annotated is not None:
                ann_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                ann_msg.header = msg.header
                self.ann_pub.publish(ann_msg)

        except Exception as e:
            self.get_logger().warning(f"Lane assist hata: {repr(e)}")


def main(args=None):
    rclpy.init(args=args)
    node = LaneAssistNode()

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
