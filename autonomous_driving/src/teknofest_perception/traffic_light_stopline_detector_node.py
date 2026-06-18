#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
import traceback
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger


class TrafficLightStoplineDetectorNode(Node):
    def __init__(self):
        super().__init__("traffic_light_stopline_detector_node")

        self.declare_parameter("image_topic", "/adas/camera/front/image_raw")
        self.declare_parameter("depth_topic", "/zed/zed_node/depth/depth_registered")
        self.declare_parameter("stopline_topic", "/adas/perception/traffic_light_stopline")
        self.declare_parameter("output_topic", "")
        self.declare_parameter("debug_image_topic", "/adas/debug/traffic_light_stopline_image")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 360)
        self.declare_parameter("camera_fov_deg", 72.0)
        self.declare_parameter("camera_height_m", 2.25)
        self.declare_parameter("camera_pitch_deg", -1.0)
        self.declare_parameter("front_bumper_offset_m", 1.35)
        self.declare_parameter("roi_x_min_ratio", 0.08)
        self.declare_parameter("roi_x_max_ratio", 0.92)
        self.declare_parameter("roi_y_min_ratio", 0.40)
        self.declare_parameter("roi_y_max_ratio", 0.98)
        self.declare_parameter("white_hsv_min_value", 145)
        self.declare_parameter("white_hsv_max_saturation", 110)
        self.declare_parameter("white_gray_min_value", 165)
        self.declare_parameter("horizontal_angle_tolerance_deg", 20.0)
        self.declare_parameter("min_stopline_width_ratio", 0.18)
        self.declare_parameter("max_stopline_thickness_ratio", 0.28)
        self.declare_parameter("min_stable_frames", 3)
        self.declare_parameter("confidence_threshold", 0.55)
        self.declare_parameter("max_depth_age_s", 0.25)
        self.declare_parameter("max_stopline_distance_m", 35.0)
        self.declare_parameter("publish_debug_image", True)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.bridge = CvBridge()
        self.last_depth: Optional[np.ndarray] = None
        self.last_depth_s = 0.0
        self.image_received = False
        self.depth_received = False
        self.stable_count = 0
        self.last_ros_log_s = 0.0

        self.width = int(self.get_parameter("camera_width").value)
        self.height = int(self.get_parameter("camera_height").value)
        self.fov_deg = float(self.get_parameter("camera_fov_deg").value)
        self.camera_height_m = float(self.get_parameter("camera_height_m").value)
        self.camera_pitch_deg = float(self.get_parameter("camera_pitch_deg").value)
        self.front_bumper_offset_m = float(self.get_parameter("front_bumper_offset_m").value)
        self.roi_x_min_ratio = float(self.get_parameter("roi_x_min_ratio").value)
        self.roi_x_max_ratio = float(self.get_parameter("roi_x_max_ratio").value)
        self.roi_y_min_ratio = float(self.get_parameter("roi_y_min_ratio").value)
        self.roi_y_max_ratio = float(self.get_parameter("roi_y_max_ratio").value)
        self.white_hsv_min_value = int(self.get_parameter("white_hsv_min_value").value)
        self.white_hsv_max_saturation = int(self.get_parameter("white_hsv_max_saturation").value)
        self.white_gray_min_value = int(self.get_parameter("white_gray_min_value").value)
        self.horizontal_angle_tolerance_deg = float(self.get_parameter("horizontal_angle_tolerance_deg").value)
        self.min_stopline_width_ratio = float(self.get_parameter("min_stopline_width_ratio").value)
        self.max_stopline_thickness_ratio = float(self.get_parameter("max_stopline_thickness_ratio").value)
        self.min_stable_frames = int(self.get_parameter("min_stable_frames").value)
        self.confidence_threshold = float(self.get_parameter("confidence_threshold").value)
        self.max_depth_age_s = float(self.get_parameter("max_depth_age_s").value)
        self.max_stopline_distance_m = float(self.get_parameter("max_stopline_distance_m").value)
        self.publish_debug_image_enabled = bool(self.get_parameter("publish_debug_image").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        output_topic = str(self.get_parameter("output_topic").value or "").strip()
        stopline_topic = output_topic or str(self.get_parameter("stopline_topic").value)
        self.stopline_pub = self.create_publisher(String, stopline_topic, 10)
        self.debug_pub = self.create_publisher(Image, str(self.get_parameter("debug_image_topic").value), 2)
        self.image_sub = self.create_subscription(Image, str(self.get_parameter("image_topic").value), self.image_cb, 10)
        self.depth_sub = self.create_subscription(Image, str(self.get_parameter("depth_topic").value), self.depth_cb, 10)
        self.heartbeat_timer = self.create_timer(1.0, self.heartbeat_cb)

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="traffic_light_stopline_detector_node",
            file_name="traffic_light_stopline.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        self.get_logger().info("Traffic light stopline detector node ready.")

    def depth_cb(self, msg: Image):
        try:
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            self.last_depth = np.asarray(depth, dtype=np.float32)
            self.last_depth_s = time.time()
            self.depth_received = True
        except Exception as exc:
            self.get_logger().warning(f"Depth image decode failed: {exc}")

    def image_cb(self, msg: Image):
        now = time.time()
        self.image_received = True
        image_stamp_s = self.ros_stamp_to_float(msg.header.stamp)
        image_age_ms = None if image_stamp_s <= 0.0 else max(0.0, (now - image_stamp_s) * 1000.0)
        try:
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.publish_payload(now, self.empty_detection(f"image_decode_failed:{type(exc).__name__}", image_age_ms))
            return

        try:
            detection = self.detect_stopline(image, now, image_age_ms)
        except Exception as exc:
            self.get_logger().error(f"stopline detection exception: {exc}")
            self.get_logger().error(traceback.format_exc())
            detection = self.empty_detection("detect_exception", image_age_ms)
            detection.update(self.base_debug_fields(now, image_age_ms, None))
        self.publish_payload(now, detection)

    def heartbeat_cb(self):
        if self.image_received:
            return
        now = time.time()
        detection = self.empty_detection("waiting_for_image")
        detection.update(self.base_debug_fields(now, None, None))
        self.publish_payload(now, detection)

    def detect_stopline(self, image: np.ndarray, now: float, image_age_ms: Optional[float]) -> dict:
        height, width = image.shape[:2]
        y0 = int(height * np.clip(self.roi_y_min_ratio, 0.0, 1.0))
        y1 = int(height * np.clip(self.roi_y_max_ratio, 0.0, 1.0))
        x0 = int(width * np.clip(self.roi_x_min_ratio, 0.0, 1.0))
        x1 = int(width * np.clip(self.roi_x_max_ratio, 0.0, 1.0))
        roi = image[y0:y1, x0:x1]
        debug = self.base_debug_fields(now, image_age_ms, roi.shape if roi.size else None)
        if roi.size == 0:
            self.stable_count = 0
            out = self.empty_detection("empty_roi", image_age_ms)
            out.update(debug)
            self.publish_debug_image(image, None, None, (x0, y0, x1, y1), out)
            return out

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        hsv_mask = cv2.inRange(
            hsv,
            np.array([0, 0, self.white_hsv_min_value], dtype=np.uint8),
            np.array([180, self.white_hsv_max_saturation, 255], dtype=np.uint8),
        )
        gray_mask = cv2.inRange(gray, self.white_gray_min_value, 255)
        mask = cv2.bitwise_or(hsv_mask, gray_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 5)), iterations=2)
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3)), iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)), iterations=1)
        debug["white_mask_pixels"] = int(cv2.countNonZero(mask))

        roi_width = max(1, x1 - x0)
        roi_height = max(1, y1 - y0)
        min_width_px = max(18.0, roi_width * self.min_stopline_width_ratio)
        max_thickness_px = max(8.0, roi_height * self.max_stopline_thickness_ratio)

        candidates = self.find_contour_candidates(mask, x0, y0, roi_width, roi_height, min_width_px, max_thickness_px, debug)
        candidates.extend(self.find_hough_candidates(mask, x0, y0, roi_width, roi_height, min_width_px, debug))
        debug["candidate_line_count"] = len(candidates)

        best = max(candidates, key=lambda c: float(c["confidence"]), default=None)
        if best is not None:
            debug["best_line_angle_deg"] = best.get("stopline_angle_deg", best.get("angle_deg"))
            debug["best_line_width_px"] = best.get("stopline_width_px", best.get("width_px"))
            debug["best_line_y"] = best.get("stopline_pixel_y", best.get("pixel_y"))
            debug["best_line_confidence"] = best.get("confidence", 0.0)

        if best is None:
            self.stable_count = 0
            out = self.empty_detection("no_horizontal_white_line", image_age_ms)
            out.update(debug)
            self.publish_debug_image(image, mask, None, (x0, y0, x1, y1), out)
            return out

        distance_m, source, distance_reason = self.estimate_distance(best, width, height, now)
        if distance_m is None or distance_m <= 0.0 or distance_m > self.max_stopline_distance_m:
            self.stable_count = 0
            out = self.empty_detection(distance_reason, image_age_ms)
            out.update(best)
            out.update(debug)
            self.publish_debug_image(image, mask, best, (x0, y0, x1, y1), out)
            return out

        confidence = float(best.get("confidence", 0.0))
        if confidence >= self.confidence_threshold:
            self.stable_count += 1
        else:
            self.stable_count = 0
        detected = bool(self.stable_count >= self.min_stable_frames and confidence >= self.confidence_threshold)

        out = {
            "stopline_detected": detected,
            "stopline_confidence": round(confidence, 4),
            "stopline_distance_m": round(float(distance_m), 3),
            "front_bumper_to_stopline_m": round(float(distance_m), 3),
            "stopline_source": source,
            "stopline_pixel_y": best.get("stopline_pixel_y", best.get("pixel_y")),
            "stopline_pixel_x1": best.get("stopline_pixel_x1", best.get("pixel_x1")),
            "stopline_pixel_x2": best.get("stopline_pixel_x2", best.get("pixel_x2")),
            "stopline_width_px": best.get("stopline_width_px", best.get("width_px")),
            "stopline_angle_deg": best.get("stopline_angle_deg", best.get("angle_deg")),
            "roi_valid": True,
            "reject_reason": "" if detected else "waiting_for_stable_frames",
            "stable_frame_count": int(self.stable_count),
            "required_stable_frames": int(self.min_stable_frames),
        }
        out.update(best)
        out.update(debug)
        self.publish_debug_image(image, mask, best, (x0, y0, x1, y1), out)
        return out

    def find_contour_candidates(
        self,
        mask: np.ndarray,
        x0: int,
        y0: int,
        roi_width: int,
        roi_height: int,
        min_width_px: float,
        max_thickness_px: float,
        debug: dict,
    ) -> list[dict]:
        contours, _hier = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            (_cx, _cy), (rw, rh), _angle = rect
            long_side = max(rw, rh)
            short_side = max(1.0, min(rw, rh))
            if long_side < min_width_px:
                debug["rejected_by_width_count"] += 1
                continue
            if short_side > max_thickness_px:
                debug["rejected_by_roi_count"] += 1
                continue

            line = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            vx = float(line[0][0])
            vy = float(line[1][0])
            angle_deg = math.degrees(math.atan2(vy, vx))
            angle_abs = self.horizontal_angle_abs(angle_deg)
            if angle_abs > self.horizontal_angle_tolerance_deg:
                debug["rejected_by_angle_count"] += 1
                continue

            x, y, w, h = cv2.boundingRect(contour)
            candidates.append(self.score_line_candidate(
                x0=x0,
                y0=y0,
                roi_width=roi_width,
                roi_height=roi_height,
                x=x,
                y=y,
                w=w,
                h=h,
                angle_deg=angle_deg,
                angle_abs=angle_abs,
                area=float(cv2.contourArea(contour)),
                source="contour",
            ))
        return candidates

    def find_hough_candidates(
        self,
        mask: np.ndarray,
        x0: int,
        y0: int,
        roi_width: int,
        roi_height: int,
        min_width_px: float,
        debug: dict,
    ) -> list[dict]:
        candidates = []
        lines = cv2.HoughLinesP(
            mask,
            rho=1,
            theta=np.pi / 180.0,
            threshold=18,
            minLineLength=int(min_width_px),
            maxLineGap=32,
        )
        if lines is None:
            return candidates
        for line in lines[:, 0, :]:
            lx1, ly1, lx2, ly2 = [int(v) for v in line]
            dx = float(lx2 - lx1)
            dy = float(ly2 - ly1)
            length = math.hypot(dx, dy)
            if length < min_width_px:
                debug["rejected_by_width_count"] += 1
                continue
            angle_deg = math.degrees(math.atan2(dy, dx))
            angle_abs = self.horizontal_angle_abs(angle_deg)
            if angle_abs > self.horizontal_angle_tolerance_deg:
                debug["rejected_by_angle_count"] += 1
                continue

            x = min(lx1, lx2)
            y = min(ly1, ly2)
            w = max(1, abs(lx2 - lx1))
            h = max(3, abs(ly2 - ly1) + 3)
            candidates.append(self.score_line_candidate(
                x0=x0,
                y0=y0,
                roi_width=roi_width,
                roi_height=roi_height,
                x=x,
                y=y,
                w=w,
                h=h,
                angle_deg=angle_deg,
                angle_abs=angle_abs,
                area=length * 3.0,
                source="hough",
            ))
        return candidates

    def score_line_candidate(
        self,
        *,
        x0: int,
        y0: int,
        roi_width: int,
        roi_height: int,
        x: int,
        y: int,
        w: int,
        h: int,
        angle_deg: float,
        angle_abs: float,
        area: float,
        source: str,
    ) -> dict:
        width_score = min(1.0, float(w) / max(1.0, roi_width * 0.50))
        angle_score = max(0.0, 1.0 - angle_abs / max(1.0, self.horizontal_angle_tolerance_deg))
        area_score = min(1.0, area / max(1.0, roi_width * 4.0))
        bottom_preference = min(1.0, (float(y + h) / max(1.0, roi_height)))
        score = 0.48 * width_score + 0.30 * angle_score + 0.12 * area_score + 0.10 * bottom_preference
        return {
            "pixel_x1": int(x0 + x),
            "pixel_x2": int(x0 + x + w),
            "pixel_y": int(y0 + y + h / 2),
            "width_px": int(w),
            "angle_deg": round(float(angle_deg), 3),
            "stopline_pixel_x1": int(x0 + x),
            "stopline_pixel_x2": int(x0 + x + w),
            "stopline_pixel_y": int(y0 + y + h / 2),
            "stopline_width_px": int(w),
            "stopline_angle_deg": round(float(angle_deg), 3),
            "confidence": round(float(score), 4),
            "mask_area_px": int(area),
            "candidate_source": source,
        }

    def estimate_distance(self, best: dict, width: int, height: int, now: float) -> tuple[Optional[float], str, str]:
        if self.last_depth is not None and now - self.last_depth_s <= self.max_depth_age_s:
            depth = self.last_depth
            y = int(np.clip(best.get("stopline_pixel_y", best.get("pixel_y", 0)), 0, depth.shape[0] - 1))
            x1 = int(np.clip(best.get("stopline_pixel_x1", best.get("pixel_x1", 0)), 0, depth.shape[1] - 1))
            x2 = int(np.clip(best.get("stopline_pixel_x2", best.get("pixel_x2", x1 + 1)), x1 + 1, depth.shape[1]))
            band = depth[max(0, y - 2): min(depth.shape[0], y + 3), x1:x2]
            valid = band[np.isfinite(band) & (band > 0.2) & (band < self.max_stopline_distance_m + 10.0)]
            if valid.size >= 8:
                distance = float(np.median(valid)) - self.front_bumper_offset_m
                return max(0.0, distance), "camera_depth", ""

        # Fallback: rough ground projection from image row. This is only used
        # when depth is stale/unavailable and is logged as camera_ipm.
        vertical_fov_rad = 2.0 * math.atan(math.tan(math.radians(self.fov_deg) / 2.0) * (height / max(1.0, width)))
        cy = height / 2.0
        pixel_y = float(best.get("stopline_pixel_y", best.get("pixel_y", cy)))
        norm_y = (pixel_y - cy) / max(1.0, height / 2.0)
        ray_down_angle = math.radians(-self.camera_pitch_deg) + norm_y * (vertical_fov_rad / 2.0)
        if ray_down_angle <= math.radians(1.0):
            return None, "camera_ipm", "ipm_ray_above_ground"
        distance = self.camera_height_m / math.tan(ray_down_angle) - self.front_bumper_offset_m
        return max(0.0, float(distance)), "camera_ipm", ""

    def empty_detection(self, reason: str, image_age_ms: Optional[float] = None) -> dict:
        return {
            "stopline_detected": False,
            "stopline_confidence": 0.0,
            "stopline_distance_m": None,
            "front_bumper_to_stopline_m": None,
            "stopline_source": "none",
            "stopline_pixel_y": None,
            "stopline_pixel_x1": None,
            "stopline_pixel_x2": None,
            "stopline_width_px": None,
            "stopline_angle_deg": None,
            "roi_valid": False,
            "reject_reason": reason,
            "stable_frame_count": int(self.stable_count),
            "required_stable_frames": int(self.min_stable_frames),
            "image_age_ms": None if image_age_ms is None else round(float(image_age_ms), 3),
        }

    def base_debug_fields(self, now: float, image_age_ms: Optional[float], roi_shape: Optional[tuple]) -> dict:
        depth_age_ms = None if self.last_depth_s <= 0.0 else max(0.0, (now - self.last_depth_s) * 1000.0)
        return {
            "image_age_ms": None if image_age_ms is None else round(float(image_age_ms), 3),
            "depth_age_ms": None if depth_age_ms is None else round(float(depth_age_ms), 3),
            "roi_shape": None if roi_shape is None else [int(v) for v in roi_shape[:2]],
            "white_mask_pixels": 0,
            "candidate_line_count": 0,
            "rejected_by_angle_count": 0,
            "rejected_by_width_count": 0,
            "rejected_by_roi_count": 0,
            "best_line_angle_deg": None,
            "best_line_width_px": None,
            "best_line_y": None,
            "best_line_confidence": 0.0,
        }

    @staticmethod
    def horizontal_angle_abs(angle_deg: float) -> float:
        return abs((angle_deg + 90.0) % 180.0 - 90.0)

    @staticmethod
    def ros_stamp_to_float(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def publish_debug_image(self, image: np.ndarray, mask: Optional[np.ndarray], best: Optional[dict], roi_box: tuple[int, int, int, int], payload: dict):
        if not self.publish_debug_image_enabled:
            return
        annotated = image.copy()
        x0, y0, x1, y1 = roi_box
        cv2.rectangle(annotated, (x0, y0), (x1, y1), (255, 180, 0), 2)
        if mask is not None and mask.size:
            overlay = np.zeros_like(annotated[y0:y1, x0:x1])
            overlay[:, :, 1] = mask
            annotated[y0:y1, x0:x1] = cv2.addWeighted(annotated[y0:y1, x0:x1], 0.78, overlay, 0.22, 0.0)
        if best is not None:
            x_left = int(best.get("stopline_pixel_x1", best.get("pixel_x1", 0)))
            x_right = int(best.get("stopline_pixel_x2", best.get("pixel_x2", x_left)))
            y = int(best.get("stopline_pixel_y", best.get("pixel_y", 0)))
            cv2.line(annotated, (x_left, y), (x_right, y), (0, 0, 255), 3)
        text = (
            f"det={payload.get('stopline_detected')} conf={payload.get('stopline_confidence')} "
            f"mask={payload.get('white_mask_pixels')} cand={payload.get('candidate_line_count')} "
            f"reason={payload.get('reject_reason')}"
        )
        cv2.putText(annotated, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(annotated, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        try:
            self.debug_pub.publish(self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8"))
        except Exception as exc:
            self.get_logger().debug(f"Stopline debug image publish failed: {exc}")

    def publish_payload(self, stamp: float, detection: dict):
        payload = {
            "stamp": stamp,
            "source": "traffic_light_stopline_detector_node",
            "image_received": bool(self.image_received),
            "depth_received": bool(self.depth_received),
            **detection,
        }
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.stopline_pub.publish(msg)
        self.log_runtime(payload)

    def log_runtime(self, payload: dict):
        self.runtime_logger.write(payload)
        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "traffic_light_stopline "
                f"det={payload.get('stopline_detected')} conf={payload.get('stopline_confidence')} "
                f"dist={payload.get('front_bumper_to_stopline_m')} source={payload.get('stopline_source')} "
                f"mask={payload.get('white_mask_pixels')} cand={payload.get('candidate_line_count')} "
                f"rej_angle={payload.get('rejected_by_angle_count')} rej_width={payload.get('rejected_by_width_count')} "
                f"best_w={payload.get('best_line_width_px')} best_y={payload.get('best_line_y')} "
                f"reason={payload.get('reject_reason')}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightStoplineDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as exc:
        node.get_logger().error(f"traffic_light_stopline_detector_node crashed: {exc}")
        node.get_logger().error(traceback.format_exc())
        raise
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
