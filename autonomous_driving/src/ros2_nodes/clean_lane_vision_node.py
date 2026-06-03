#!/usr/bin/env python3
import json
import math
import time

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String


def clamp(value, low, high):
    return max(low, min(high, value))


class CleanLaneVisionNode(Node):
    def __init__(self):
        super().__init__("clean_lane_vision_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("vision_topic", "/adas/phase1/lane_vision_json")
        self.declare_parameter("debug_image_topic", "/adas/phase1/lane_vision_debug_image")
        self.declare_parameter("roi_top_ratio", 0.52)
        self.declare_parameter("canny_low", 50)
        self.declare_parameter("canny_high", 140)
        self.declare_parameter("hough_threshold", 28)
        self.declare_parameter("min_line_length", 28)
        self.declare_parameter("max_line_gap", 65)
        self.declare_parameter("lane_center_alpha", 0.22)
        self.declare_parameter("offset_alpha", 0.24)
        self.declare_parameter("hold_last_valid_s", 0.65)
        self.declare_parameter("confidence_enter", 0.55)
        self.declare_parameter("confidence_exit", 0.25)

        self.image_topic = str(self.get_parameter("image_topic").value)
        self.vision_topic = str(self.get_parameter("vision_topic").value)
        self.debug_image_topic = str(self.get_parameter("debug_image_topic").value)
        self.roi_top_ratio = float(self.get_parameter("roi_top_ratio").value)
        self.canny_low = int(self.get_parameter("canny_low").value)
        self.canny_high = int(self.get_parameter("canny_high").value)
        self.hough_threshold = int(self.get_parameter("hough_threshold").value)
        self.min_line_length = int(self.get_parameter("min_line_length").value)
        self.max_line_gap = int(self.get_parameter("max_line_gap").value)
        self.lane_center_alpha = float(self.get_parameter("lane_center_alpha").value)
        self.offset_alpha = float(self.get_parameter("offset_alpha").value)
        self.hold_last_valid_s = float(self.get_parameter("hold_last_valid_s").value)
        self.confidence_enter = float(self.get_parameter("confidence_enter").value)
        self.confidence_exit = float(self.get_parameter("confidence_exit").value)

        self.bridge = CvBridge()
        self.pub = self.create_publisher(String, self.vision_topic, 10)
        self.debug_pub = self.create_publisher(Image, self.debug_image_topic, 5)
        self.create_subscription(Image, self.image_topic, self.image_cb, 10)

        self.filtered_center_px = None
        self.filtered_offset_norm = 0.0
        self.filtered_heading_deg = 0.0
        self.last_lane_width_px = None
        self.last_two_line_center_px = None
        self.last_two_line_width_px = None
        self.last_two_line_heading_deg = 0.0
        self.last_two_line_s = 0.0
        self.last_valid_payload = None
        self.last_valid_s = 0.0
        self.vision_active = False
        self.last_log_s = 0.0

        self.get_logger().info(
            f"clean_lane_vision_node ready: image={self.image_topic} vision={self.vision_topic}"
        )

    def image_cb(self, msg):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warn(f"Image conversion failed: {exc}")
            return

        payload, debug = self.process_frame(frame)
        out = String()
        out.data = json.dumps(payload)
        self.pub.publish(out)

        try:
            debug_msg = self.bridge.cv2_to_imgmsg(debug, encoding="bgr8")
            debug_msg.header = msg.header
            self.debug_pub.publish(debug_msg)
        except Exception as exc:
            self.get_logger().warn(f"Debug image publish failed: {exc}")

        now = time.time()
        if now - self.last_log_s >= 1.0:
            self.last_log_s = now
            self.get_logger().info(
                "CLEAN_LANE_VISION "
                f"valid={payload['valid']} conf={payload['confidence']:.3f} "
                f"left={payload['left_line_found']} right={payload['right_line_found']} "
                f"center_px={payload['lane_center_px']} "
                f"offset_norm={payload['vision_offset_norm']:.4f} "
                f"heading={payload['vision_heading_error_deg']:.3f} "
                f"source={payload['source']} "
                f"lane_width_px={payload['lane_width_px']} "
                f"center_jump_px={payload['center_jump_px']} "
                f"heading_jump_deg={payload['heading_jump_deg']} "
                f"rejected_reason={payload['rejected_reason']} "
                f"used_last_valid={payload['used_last_valid']} "
                f"reason={payload['reason']}"
            )

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        roi_y = int(h * clamp(self.roi_top_ratio, 0.35, 0.80))
        y_bottom = h - 1
        y_top = roi_y
        vehicle_center_px = w * 0.5

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        hls_h = hls[:, :, 0]
        hls_l = hls[:, :, 1]
        hls_s = hls[:, :, 2]

        # CARLA yol çizgileri çoğu sahnede beyaz/sarı. Sadece gri Canny,
        # asfalt dokusu veya gölge ile çizgiyi kaçırabiliyor; renk maskesiyle
        # destekleyip yine edge tabanlı kalıyoruz.
        white_mask = cv2.inRange(hls_l, 165, 255)
        yellow_mask = cv2.inRange(hls_h, 15, 38) & cv2.inRange(hls_s, 55, 255) & cv2.inRange(hls_l, 70, 255)
        color_mask = cv2.bitwise_or(white_mask, yellow_mask)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges_gray = cv2.Canny(blur, self.canny_low, self.canny_high)
        edges_color = cv2.Canny(color_mask, 40, 120)
        edges = cv2.bitwise_or(edges_gray, edges_color)

        mask = np.zeros_like(edges)
        polygon = np.array([[
            (int(w * 0.06), h),
            (int(w * 0.42), roi_y),
            (int(w * 0.58), roi_y),
            (int(w * 0.94), h),
        ]], dtype=np.int32)
        cv2.fillPoly(mask, polygon, 255)
        roi_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            roi_edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )

        left_points, right_points = self.collect_lane_points(lines, w, h, roi_y)
        left_fit = self.fit_line_x_of_y(left_points)
        right_fit = self.fit_line_x_of_y(right_points)

        left_found = left_fit is not None
        right_found = right_fit is not None

        left_bottom = self.eval_fit(left_fit, y_bottom) if left_found else None
        right_bottom = self.eval_fit(right_fit, y_bottom) if right_found else None
        left_top = self.eval_fit(left_fit, y_top) if left_found else None
        right_top = self.eval_fit(right_fit, y_top) if right_found else None

        lane_width_px = None
        source = "no_lines"
        reason = "no_lines"
        rejected_reason = None
        center_jump_px = None
        heading_jump_deg = None
        used_last_valid = False
        confidence = 0.0
        lane_center_bottom = None
        lane_center_top = None
        now = time.time()

        if left_found and right_found and right_bottom is not None and left_bottom is not None:
            lane_width_px = float(right_bottom - left_bottom)
            candidate_center_bottom = 0.5 * (left_bottom + right_bottom)
            candidate_center_top = 0.5 * (left_top + right_top)
            candidate_heading = math.degrees(
                math.atan2(candidate_center_top - candidate_center_bottom, max(1.0, y_bottom - y_top))
            )
            center_jump_px = (
                abs(candidate_center_bottom - self.last_two_line_center_px)
                if self.last_two_line_center_px is not None else 0.0
            )
            heading_jump_deg = (
                abs(candidate_heading - self.last_two_line_heading_deg)
                if self.last_two_line_center_px is not None else 0.0
            )

            if not (w * 0.22 <= lane_width_px <= w * 0.90):
                rejected_reason = "bad_lane_width"
                reason = "bad_lane_width"
            elif not (left_bottom < right_bottom and left_top < right_top):
                rejected_reason = "left_right_crossed"
                reason = "left_right_crossed"
            elif not (0.0 <= candidate_center_bottom <= float(w - 1)):
                rejected_reason = "center_out_of_image"
                reason = "center_out_of_image"
            elif abs(candidate_heading) > 14.0:
                rejected_reason = "two_lines_rejected_heading"
                source = "two_lines_rejected_heading"
                reason = "two_lines_rejected_heading"
                confidence = 0.20
            elif center_jump_px > 95.0:
                rejected_reason = "two_lines_center_jump"
                reason = "two_lines_center_jump"
                confidence = 0.25
            elif heading_jump_deg > 22.0:
                rejected_reason = "two_lines_heading_jump"
                reason = "two_lines_heading_jump"
                confidence = 0.30
            else:
                lane_center_bottom = candidate_center_bottom
                lane_center_top = candidate_center_top
                confidence = 0.86 if abs(candidate_heading) <= 8.0 else 0.45
                source = "two_lines_stable"
                reason = "two_lines_stable"
                self.last_lane_width_px = lane_width_px
                self.last_two_line_center_px = lane_center_bottom
                self.last_two_line_width_px = lane_width_px
                self.last_two_line_heading_deg = candidate_heading
                self.last_two_line_s = now

        if lane_center_bottom is None and (left_found or right_found):
            recent_two_line = (
                self.last_two_line_width_px is not None
                and now - self.last_two_line_s <= 1.00
            )
            if not recent_two_line:
                rejected_reason = rejected_reason or "one_line_rejected_no_recent_width"
                source = "one_line_rejected_no_recent_width"
                reason = "one_line_rejected_no_recent_width"
                confidence = min(confidence, 0.15)
            else:
                estimated_width = self.last_two_line_width_px
                if left_found and left_bottom is not None:
                    right_bottom = left_bottom + estimated_width
                    right_top = left_top + estimated_width
                    candidate_source = "one_line_left"
                    candidate_reason = "right_line_estimated"
                elif right_found and right_bottom is not None:
                    left_bottom = right_bottom - estimated_width
                    left_top = right_top - estimated_width
                    candidate_source = "one_line_right"
                    candidate_reason = "left_line_estimated"
                else:
                    candidate_source = None
                    candidate_reason = None

                if candidate_source is not None:
                    candidate_center_bottom = 0.5 * (left_bottom + right_bottom)
                    candidate_center_top = 0.5 * (left_top + right_top)
                    candidate_heading = math.degrees(
                        math.atan2(
                            candidate_center_top - candidate_center_bottom,
                            max(1.0, y_bottom - y_top),
                        )
                    )
                    center_jump_px = abs(candidate_center_bottom - self.last_two_line_center_px)
                    heading_jump_deg = abs(candidate_heading - self.last_two_line_heading_deg)

                    if abs(candidate_heading) > 15.0:
                        rejected_reason = "one_line_rejected_heading"
                        source = "one_line_rejected_heading"
                        reason = "one_line_rejected_heading"
                        confidence = 0.15
                    elif center_jump_px > 40.0:
                        rejected_reason = "one_line_rejected_center_jump"
                        source = "one_line_rejected_center_jump"
                        reason = "one_line_rejected_center_jump"
                        confidence = 0.18
                    elif heading_jump_deg > 12.0:
                        rejected_reason = "one_line_rejected_heading_jump"
                        source = "one_line_rejected_heading_jump"
                        reason = "one_line_rejected_heading_jump"
                        confidence = 0.18
                    else:
                        lane_center_bottom = candidate_center_bottom
                        lane_center_top = candidate_center_top
                        lane_width_px = estimated_width
                        confidence = 0.30
                        source = "one_line_estimated_low_conf"
                        reason = candidate_reason

        raw_offset_norm = 0.0
        raw_heading_deg = 0.0
        valid = lane_center_bottom is not None

        if valid:
            lane_center_bottom = float(clamp(lane_center_bottom, 0.0, float(w - 1)))
            lane_center_top = float(clamp(lane_center_top, 0.0, float(w - 1)))
            raw_offset_norm = (lane_center_bottom - vehicle_center_px) / float(w)
            raw_heading_deg = math.degrees(
                math.atan2(lane_center_top - lane_center_bottom, max(1.0, y_bottom - y_top))
            )

            if self.filtered_center_px is None:
                self.filtered_center_px = lane_center_bottom
                self.filtered_offset_norm = raw_offset_norm
                self.filtered_heading_deg = raw_heading_deg
            else:
                self.filtered_center_px = (
                    (1.0 - self.lane_center_alpha) * self.filtered_center_px
                    + self.lane_center_alpha * lane_center_bottom
                )
                self.filtered_offset_norm = (
                    (1.0 - self.offset_alpha) * self.filtered_offset_norm
                    + self.offset_alpha * raw_offset_norm
                )
                self.filtered_heading_deg = 0.75 * self.filtered_heading_deg + 0.25 * raw_heading_deg

            payload = self.make_payload(
                valid=True,
                confidence=confidence,
                left_found=left_found,
                right_found=right_found,
                lane_center_px=self.filtered_center_px,
                vehicle_center_px=vehicle_center_px,
                offset_norm=self.filtered_offset_norm,
                heading_deg=self.filtered_heading_deg,
                lane_width_px=lane_width_px,
                source=source,
                reason=reason,
                rejected_reason=rejected_reason,
                center_jump_px=center_jump_px,
                heading_jump_deg=heading_jump_deg,
                used_last_valid=used_last_valid,
            )
            if source == "two_lines_stable":
                self.last_valid_payload = payload
                self.last_valid_s = now
        elif self.last_valid_payload is not None and now - self.last_valid_s <= self.hold_last_valid_s:
            payload = dict(self.last_valid_payload)
            age = now - self.last_valid_s
            payload["stamp"] = now
            payload["source"] = "last_valid"
            payload["reason"] = f"held_last_valid_{age:.2f}s"
            payload["confidence"] = min(
                0.45,
                max(0.20, float(payload["confidence"]) * (1.0 - age / self.hold_last_valid_s)),
            )
            payload["valid"] = True
            payload["used_last_valid"] = True
            if rejected_reason is not None:
                payload["rejected_reason"] = rejected_reason
        else:
            payload = self.make_payload(
                valid=False,
                confidence=0.0,
                left_found=False,
                right_found=False,
                lane_center_px=None,
                vehicle_center_px=vehicle_center_px,
                offset_norm=0.0,
                heading_deg=0.0,
                lane_width_px=None,
                source="no_lines_map_fallback",
                reason="no_lines_map_fallback",
                rejected_reason=rejected_reason,
                center_jump_px=center_jump_px,
                heading_jump_deg=heading_jump_deg,
                used_last_valid=used_last_valid,
            )

        self.update_confidence_hysteresis(payload)
        debug = self.draw_debug(
            frame,
            roi_y,
            left_fit,
            right_fit,
            payload,
            y_top,
            y_bottom,
        )
        return payload, debug

    def collect_lane_points(self, lines, width, height, roi_y):
        left_points = []
        right_points = []
        if lines is None:
            return left_points, right_points

        for line in lines[:, 0, :]:
            x1, y1, x2, y2 = [float(v) for v in line]
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 4.0:
                continue
            slope = dy / dx
            length = math.hypot(dx, dy)
            if length < self.min_line_length:
                continue
            if abs(slope) < 0.45 or abs(slope) > 5.0:
                continue

            mid_x = 0.5 * (x1 + x2)
            if slope < 0.0 and mid_x < width * 0.62:
                left_points.extend([(x1, y1), (x2, y2)])
            elif slope > 0.0 and mid_x > width * 0.38:
                right_points.extend([(x1, y1), (x2, y2)])

        return left_points, right_points

    def fit_line_x_of_y(self, points):
        if len(points) < 4:
            return None
        ys = np.array([p[1] for p in points], dtype=np.float32)
        xs = np.array([p[0] for p in points], dtype=np.float32)
        try:
            a, b = np.polyfit(ys, xs, 1)
        except Exception:
            return None
        if not np.isfinite(a) or not np.isfinite(b):
            return None
        return float(a), float(b)

    def eval_fit(self, fit, y):
        if fit is None:
            return None
        a, b = fit
        return float(a * y + b)

    def make_payload(
        self,
        valid,
        confidence,
        left_found,
        right_found,
        lane_center_px,
        vehicle_center_px,
        offset_norm,
        heading_deg,
        lane_width_px,
        source,
        reason,
        rejected_reason=None,
        center_jump_px=None,
        heading_jump_deg=None,
        used_last_valid=False,
    ):
        return {
            "stamp": time.time(),
            "valid": bool(valid),
            "confidence": round(float(clamp(confidence, 0.0, 1.0)), 3),
            "left_line_found": bool(left_found),
            "right_line_found": bool(right_found),
            "lane_center_px": round(float(lane_center_px), 3) if lane_center_px is not None else None,
            "vehicle_center_px": round(float(vehicle_center_px), 3),
            "vision_offset_norm": round(float(offset_norm), 6),
            "vision_heading_error_deg": round(float(heading_deg), 3),
            "lane_width_px": round(float(lane_width_px), 3) if lane_width_px is not None else None,
            "source": source,
            "reason": reason,
            "rejected_reason": rejected_reason,
            "center_jump_px": round(float(center_jump_px), 3) if center_jump_px is not None else None,
            "heading_jump_deg": round(float(heading_jump_deg), 3) if heading_jump_deg is not None else None,
            "used_last_valid": bool(used_last_valid),
        }

    def update_confidence_hysteresis(self, payload):
        conf = float(payload.get("confidence", 0.0))
        if self.vision_active:
            if conf < self.confidence_exit:
                self.vision_active = False
        elif conf >= self.confidence_enter:
            self.vision_active = True

        if (
            self.vision_active
            and payload["valid"]
            and str(payload.get("source", "")).startswith("two_lines")
        ):
            payload["confidence"] = round(max(conf, self.confidence_enter), 3)

    def draw_debug(self, frame, roi_y, left_fit, right_fit, payload, y_top, y_bottom):
        debug = frame.copy()
        h, w = debug.shape[:2]
        cv2.rectangle(debug, (0, roi_y), (w - 1, h - 1), (40, 40, 40), 1)

        def draw_fit(fit, color):
            if fit is None:
                return
            x_bottom = int(clamp(self.eval_fit(fit, y_bottom), 0, w - 1))
            x_top = int(clamp(self.eval_fit(fit, y_top), 0, w - 1))
            cv2.line(debug, (x_bottom, y_bottom), (x_top, y_top), color, 3)

        draw_fit(left_fit, (255, 0, 0))
        draw_fit(right_fit, (0, 255, 255))

        vehicle_center = int(payload["vehicle_center_px"])
        cv2.line(debug, (vehicle_center, h), (vehicle_center, roi_y), (255, 255, 255), 2)

        if payload["lane_center_px"] is not None:
            lane_center = int(payload["lane_center_px"])
            cv2.line(debug, (lane_center, h), (lane_center, roi_y), (0, 255, 0), 2)

        text = (
            f"valid={payload['valid']} conf={payload['confidence']:.2f} "
            f"offset={payload['vision_offset_norm']:.3f} "
            f"heading={payload['vision_heading_error_deg']:.1f}"
        )
        cv2.putText(debug, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(debug, payload["source"], (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)
        return debug


def main(args=None):
    rclpy.init(args=args)
    node = CleanLaneVisionNode()

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