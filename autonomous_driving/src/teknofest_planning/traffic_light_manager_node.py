#!/usr/bin/env python3
from __future__ import annotations

import json
import time
from collections import deque
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_perception.traffic_light_distance_estimator import (
    EgoStatus,
    TrafficLightDistanceEstimator,
)
from teknofest_sim.carla_loader import load_carla
from teknofest_planning.traffic_light_state_machine import (
    GREEN_RELEASE,
    NO_RELEVANT_LIGHT,
    PASSED_LIGHT,
    POST_LIGHT_IGNORE,
    RED_DECEL,
    RED_LOCK_OVERRUN_HOLD,
    RED_STOP_COMMIT,
    RED_STOP_CREEP,
    STOPPED_AT_RED,
    TrafficLightStateMachine,
)


class TrafficLightManagerNode(Node):
    def __init__(self):
        super().__init__("traffic_light_manager_node")

        # -------------------------
        # Config / parameter block
        # -------------------------
        self.declare_parameter("lane_plan_input_topic", "/adas/planning/lane_plan_raw")
        self.declare_parameter("lane_plan_output_topic", "/adas/planning/lane_plan")
        self.declare_parameter("traffic_light_topic", "/adas/perception/traffic_lights")
        self.declare_parameter("stopline_topic", "/adas/perception/traffic_light_stopline")
        self.declare_parameter("route_topic", "/adas/planning/route")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("publish_period_s", 0.05)
        self.declare_parameter("comfort_decel", 1.4)
        self.declare_parameter("max_decel", 3.0)
        self.declare_parameter("safe_stop_buffer", 1.8)
        self.declare_parameter("reaction_margin", 1.5)
        self.declare_parameter("post_light_ignore_s", 3.0)
        self.declare_parameter("roi_center_margin", 0.35)
        self.declare_parameter("route_corridor_width_m", 4.5)
        self.declare_parameter("traffic_light_distance_reference", "front_bumper")
        self.declare_parameter("traffic_light_front_bumper_offset_m", 1.35)
        self.declare_parameter("visual_stopline_confidence_threshold", 0.55)
        self.declare_parameter("visual_stopline_max_age_s", 0.35)
        self.declare_parameter("visual_stopline_association_max_stop_gap_m", 6.0)
        self.declare_parameter("visual_stopline_association_max_actor_ahead_m", 28.0)
        self.declare_parameter("visual_stopline_far_carla_gap_m", 15.0)
        self.declare_parameter("visual_red_association_latch_s", 0.9)
        self.declare_parameter("locked_visual_stopline_acquire_max_m", 30.0)
        self.declare_parameter("locked_visual_stopline_consistency_tolerance_m", 2.0)
        self.declare_parameter("visual_red_commit_distance_m", 2.5)
        self.declare_parameter("red_visual_slow_distance_m", 10.0)
        self.declare_parameter("red_visual_stop_distance_m", 2.0)
        self.declare_parameter("red_visual_hard_stop_distance_m", 1.5)
        self.declare_parameter("red_visual_approach_speed_mps", 0.5)
        self.declare_parameter("red_visual_near_light_failsafe_min_m", 3.0)
        self.declare_parameter("red_visual_near_light_failsafe_max_m", 6.5)
        self.declare_parameter("red_visual_near_light_failsafe_speed_mps", 1.2)
        self.declare_parameter("red_visual_near_light_failsafe_light_distance_m", 7.0)
        self.declare_parameter("red_visual_stuck_stop_min_s", 1.2)
        self.declare_parameter("red_visual_stuck_stop_min_speed_mps", 0.5)
        self.declare_parameter("traffic_light_debug_draw_stop_points", True)
        self.declare_parameter("traffic_light_debug_draw_life_time_s", 0.25)
        self.declare_parameter("traffic_light_debug_draw_z_offset_m", 0.45)
        self.declare_parameter("traffic_light_stop_point_suspect_visual_error_m", 4.0)
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)

        # -------------------------
        # Runtime state block
        # -------------------------
        self.lane_plan_payload: Optional[dict] = None
        self.tl_payload: Optional[dict] = None
        self.stopline_payload: Optional[dict] = None
        self.route_payload: Optional[dict] = None
        self.status_payload: Optional[dict] = None
        self.last_lane_plan_s = 0.0
        self.last_tl_s = 0.0
        self.last_stopline_s = 0.0
        self.last_route_s = 0.0
        self.last_status_s = 0.0
        self.last_ros_log_s = 0.0
        self.locked_red_light: Optional[dict] = None
        self.recent_visual_stopline_distances = deque(maxlen=5)
        self.last_visual_red_association: Optional[dict] = None
        self.locked_visual_stopline: Optional[dict] = None
        self.red_visual_approach_started_s: Optional[float] = None
        self.red_stop_hold_active = False
        self.red_stop_hold_reason = ""
        self.last_good_visual_stop_distance_m: Optional[float] = None
        self.last_good_visual_stop_s: Optional[float] = None
        self.last_good_visual_stop_point: Optional[dict] = None
        self.carla = None
        self.carla_client = None
        self.carla_world = None
        self.carla_debug_connect_attempted = False

        self.estimator = TrafficLightDistanceEstimator(
            route_corridor_width_m=float(self.get_parameter("route_corridor_width_m").value),
        )
        self.state_machine = TrafficLightStateMachine(
            comfort_decel=float(self.get_parameter("comfort_decel").value),
            max_decel=float(self.get_parameter("max_decel").value),
            safe_stop_buffer=float(self.get_parameter("safe_stop_buffer").value),
            reaction_margin=float(self.get_parameter("reaction_margin").value),
            post_light_ignore_s=float(self.get_parameter("post_light_ignore_s").value),
        )

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="traffic_light_manager_node",
            file_name="traffic_light.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        # -------------------------
        # Publisher block
        # -------------------------
        self.plan_pub = self.create_publisher(
            String,
            str(self.get_parameter("lane_plan_output_topic").value),
            10,
        )

        # -------------------------
        # Subscriber block
        # -------------------------
        self.create_subscription(String, str(self.get_parameter("lane_plan_input_topic").value), self.lane_plan_cb, 10)
        self.create_subscription(String, str(self.get_parameter("traffic_light_topic").value), self.tl_cb, 10)
        self.create_subscription(String, str(self.get_parameter("stopline_topic").value), self.stopline_cb, 10)
        self.create_subscription(String, str(self.get_parameter("route_topic").value), self.route_cb, 10)
        self.create_subscription(String, str(self.get_parameter("status_topic").value), self.status_cb, 10)

        # -------------------------
        # Timer block
        # -------------------------
        self.create_timer(float(self.get_parameter("publish_period_s").value), self.tick)
        self.get_logger().info("Traffic light manager node ready.")

    # -------------------------
    # Subscriber callbacks
    # -------------------------
    def lane_plan_cb(self, msg: String):
        try:
            self.lane_plan_payload = json.loads(msg.data)
            self.last_lane_plan_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid lane plan JSON ignored: {exc}")

    def tl_cb(self, msg: String):
        try:
            self.tl_payload = json.loads(msg.data)
            self.last_tl_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid traffic light JSON ignored: {exc}")

    def stopline_cb(self, msg: String):
        try:
            self.stopline_payload = json.loads(msg.data)
            self.last_stopline_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid stopline JSON ignored: {exc}")

    def route_cb(self, msg: String):
        try:
            self.route_payload = json.loads(msg.data)
            self.last_route_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid route JSON ignored: {exc}")

    def status_cb(self, msg: String):
        try:
            self.status_payload = json.loads(msg.data)
            self.last_status_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid status JSON ignored: {exc}")

    # -------------------------
    # Decision functions
    # -------------------------
    def ego_status(self) -> Optional[EgoStatus]:
        status = self.status_payload or {}
        loc = status.get("location") or {}
        rot = status.get("rotation") or {}
        if loc.get("x") is None or loc.get("y") is None:
            return None
        return EgoStatus(
            x=float(loc["x"]),
            y=float(loc["y"]),
            yaw_deg=float(rot.get("yaw", 0.0)),
            speed_mps=float(status.get("speed_mps", 0.0)),
            front_bumper_offset_m=float(self.get_parameter("traffic_light_front_bumper_offset_m").value),
            distance_reference=str(self.get_parameter("traffic_light_distance_reference").value),
        )

    def route_points(self) -> list[dict]:
        return list((self.route_payload or {}).get("points") or [])

    def filtered_color(self, candidate: Optional[dict]) -> tuple[str, str, float, bool]:
        if not candidate:
            return "unknown", "unknown", 0.0, False
        raw = str(candidate.get("tl_color_raw") or "unknown").lower()
        if raw not in {"red", "yellow", "green"}:
            raw = "unknown"
        confidence = float(candidate.get("tl_confidence", 0.0) or 0.0)
        return raw, raw, confidence, bool(candidate.get("tl_detected", True))

    def default_distance_result(self):
        return self.estimator.empty("missing_inputs")

    def visual_stopline_fields(
        self,
        *,
        relevant: bool,
        color: str,
        carla_distance_m,
        visual_decision_allowed: bool = True,
    ):
        payload = self.stopline_payload or {}
        now = time.time()
        age_s = now - self.last_stopline_s if self.last_stopline_s > 0.0 else None
        confidence = float(payload.get("stopline_confidence", 0.0) or 0.0)
        distance = payload.get("front_bumper_to_stopline_m")
        threshold = float(self.get_parameter("visual_stopline_confidence_threshold").value)
        max_age_s = float(self.get_parameter("visual_stopline_max_age_s").value)
        detected = bool(payload.get("stopline_detected", False))
        fresh = age_s is not None and age_s <= max_age_s
        usable_color = str(color).lower() in {"red", "yellow", "green"}
        distance_valid = distance is not None and float(distance) > 0.0
        visual_candidate_valid = bool(detected and fresh and confidence >= threshold and distance_valid)
        using = bool(visual_decision_allowed and relevant and usable_color and visual_candidate_valid)

        raw_visual_distance = float(distance) if distance is not None else None
        filtered_visual_distance = raw_visual_distance
        visual_distance_jump_rejected = False
        if visual_candidate_valid and raw_visual_distance is not None:
            recent = list(self.recent_visual_stopline_distances)
            recent_close = [d for d in recent if 4.0 <= d <= 6.0]
            if recent_close and raw_visual_distance >= 20.0:
                visual_distance_jump_rejected = True
                filtered_visual_distance = min(recent)
            else:
                self.recent_visual_stopline_distances.append(raw_visual_distance)
                filtered_visual_distance = min(self.recent_visual_stopline_distances)
        elif not visual_candidate_valid:
            self.recent_visual_stopline_distances.clear()

        selected_distance = filtered_visual_distance if using else carla_distance_m
        source = "visual_stopline" if using else "carla_stop_waypoint_fallback"
        if not relevant or not usable_color:
            reject_reason = "not_relevant_red_or_yellow"
        elif not detected:
            reject_reason = payload.get("reject_reason") or "stopline_not_detected"
        elif not fresh:
            reject_reason = "stale_stopline"
        elif confidence < threshold:
            reject_reason = "low_stopline_confidence"
        elif not distance_valid:
            reject_reason = "invalid_stopline_distance"
        elif not visual_decision_allowed:
            reject_reason = "visual_stopline_light_association_pending"
        else:
            reject_reason = ""

        return selected_distance, {
            "distance_source": source,
            "visual_stopline_candidate_valid": visual_candidate_valid,
            "visual_stopline_detected": detected,
            "visual_stopline_confidence": round(confidence, 4),
            "front_bumper_to_visual_stopline_m": round(float(distance), 3) if distance is not None else None,
            "visual_dist_raw": round(float(raw_visual_distance), 3) if raw_visual_distance is not None else None,
            "visual_dist_filtered": (
                round(float(filtered_visual_distance), 3)
                if filtered_visual_distance is not None
                else None
            ),
            "visual_distance_jump_rejected": bool(visual_distance_jump_rejected),
            "carla_stop_point_distance_m": carla_distance_m,
            "selected_distance_m": round(float(selected_distance), 3) if selected_distance is not None else None,
            "stopline_pixel_y": payload.get("stopline_pixel_y"),
            "stopline_pixel_x1": payload.get("stopline_pixel_x1"),
            "stopline_pixel_x2": payload.get("stopline_pixel_x2"),
            "stopline_width_px": payload.get("stopline_width_px"),
            "stopline_angle_deg": payload.get("stopline_angle_deg"),
            "stopline_reject_reason": reject_reason,
            "stopline_source": payload.get("stopline_source", "none"),
            "using_visual_stopline": using,
            "visual_stopline_age_ms": int(age_s * 1000.0) if age_s is not None else None,
        }

    def default_visual_association_fields(self) -> dict:
        return {
            "associated_light_id": None,
            "associated_light_color": "unknown",
            "associated_light_source": "none",
            "associated_light_distance": None,
            "associated_light_roi_score": 0.0,
            "associated_light_above_stopline": False,
            "associated_light_horizontal_overlap": False,
            "visual_stopline_light_association_valid": False,
            "visual_stopline_light_association_reason": "no_visual_stopline",
            "far_carla_candidate_ignored": False,
            "far_carla_candidate_distance_m": None,
            "carla_candidate_suspect": False,
            "carla_candidate_suspect_reason": "",
            "visual_association_latched": False,
            "visual_association_age_ms": None,
        }

    def apply_visual_red_association_latch(
        self,
        *,
        associated_candidate: Optional[dict],
        association_fields: dict,
        stopline_fields: dict,
    ) -> tuple[Optional[dict], dict]:
        now = time.time()
        fields = dict(association_fields)
        association_valid = bool(fields.get("visual_stopline_light_association_valid"))
        associated_color = str(fields.get("associated_light_color") or "unknown").lower()

        if association_valid:
            fields["visual_association_latched"] = False
            fields["visual_association_age_ms"] = 0
            if associated_color == "red" and associated_candidate is not None:
                self.last_visual_red_association = {
                    "stamp": now,
                    "candidate": dict(associated_candidate),
                    "fields": dict(fields),
                    "visual_dist_filtered": stopline_fields.get("visual_dist_filtered"),
                    "visual_dist_raw": stopline_fields.get("visual_dist_raw"),
                    "front_bumper_to_visual_stopline_m": stopline_fields.get("front_bumper_to_visual_stopline_m"),
                }
            elif associated_color in {"green", "yellow"}:
                self.last_visual_red_association = None
            return associated_candidate, fields

        latch = self.last_visual_red_association
        if not latch:
            return associated_candidate, fields

        age_s = now - float(latch.get("stamp", 0.0))
        latch_s = float(self.get_parameter("visual_red_association_latch_s").value)
        if age_s > latch_s:
            self.last_visual_red_association = None
            return associated_candidate, fields

        latched_fields = dict(latch.get("fields") or {})
        latched_fields.update({
            "visual_stopline_light_association_valid": True,
            "visual_stopline_light_association_reason": "latched_visual_red_association",
            "visual_association_latched": True,
            "visual_association_age_ms": int(age_s * 1000.0),
            "associated_light_color": "red",
        })
        if stopline_fields.get("visual_dist_filtered") is None:
            stopline_fields["visual_dist_filtered"] = latch.get("visual_dist_filtered")
        if stopline_fields.get("visual_dist_raw") is None:
            stopline_fields["visual_dist_raw"] = latch.get("visual_dist_raw")
        if stopline_fields.get("front_bumper_to_visual_stopline_m") is None:
            stopline_fields["front_bumper_to_visual_stopline_m"] = latch.get(
                "front_bumper_to_visual_stopline_m"
            )
        stopline_fields["visual_stopline_candidate_valid"] = True
        stopline_fields["visual_stopline_detected"] = bool(stopline_fields.get("visual_stopline_detected", False))
        return dict(latch.get("candidate") or {}), latched_fields

    def visual_stopline_light_association(
        self,
        *,
        ego: Optional[EgoStatus],
        route_points: list[dict],
        detections: list[dict],
        visual_distance_m,
        carla_distance_m,
    ) -> tuple[Optional[dict], dict]:
        fields = self.default_visual_association_fields()
        if visual_distance_m is None:
            return None, fields
        if ego is None:
            fields["visual_stopline_light_association_reason"] = "missing_ego"
            return None, fields
        if not route_points:
            fields["visual_stopline_light_association_reason"] = "missing_route"
            return None, fields
        if not detections:
            fields["visual_stopline_light_association_reason"] = "no_traffic_light_detection"
            return None, fields

        visual_distance = float(visual_distance_m)
        max_stop_gap = float(self.get_parameter("visual_stopline_association_max_stop_gap_m").value)
        max_actor_ahead = float(self.get_parameter("visual_stopline_association_max_actor_ahead_m").value)
        far_gap = float(self.get_parameter("visual_stopline_far_carla_gap_m").value)
        corridor_limit = float(self.get_parameter("route_corridor_width_m").value)
        reference_point = self.estimator.distance_reference_point(ego)
        cumulative = self.estimator.cumulative_route_distances(route_points)
        reference_s, _reference_lateral, _reference_index, _reference_yaw = self.estimator.project_point_to_route(
            reference_point,
            route_points,
            cumulative,
        )

        if carla_distance_m is not None and float(carla_distance_m) - visual_distance >= far_gap:
            fields["carla_candidate_suspect"] = True
            fields["carla_candidate_suspect_reason"] = "far_stop_waypoint_but_near_visual_stopline"
            fields["far_carla_candidate_distance_m"] = round(float(carla_distance_m), 3)

        best_candidate = None
        best_score = -1.0
        best_debug = None
        for candidate in detections:
            location = candidate.get("location") or {}
            if location.get("x") is None or location.get("y") is None:
                continue

            actor_s, actor_lateral, _actor_index, _actor_yaw = self.estimator.project_point_to_route(
                location,
                route_points,
                cumulative,
            )
            actor_delta = actor_s - reference_s
            best_stop_gap = float("inf")
            best_stop_delta = None
            best_stop_lateral = float("inf")
            for stop_point, _source, _index in self.estimator.candidate_stop_points(candidate):
                if stop_point.get("x") is None or stop_point.get("y") is None:
                    continue
                stop_s, stop_lateral, _stop_index, _stop_yaw = self.estimator.project_point_to_route(
                    stop_point,
                    route_points,
                    cumulative,
                )
                stop_delta = stop_s - reference_s
                stop_gap = abs(stop_delta - visual_distance)
                if stop_gap < best_stop_gap:
                    best_stop_gap = stop_gap
                    best_stop_delta = stop_delta
                    best_stop_lateral = stop_lateral

            actor_above_stopline = (
                actor_delta >= visual_distance - 1.5
                and actor_delta <= visual_distance + max_actor_ahead
            )
            stop_distance_match = best_stop_gap <= max_stop_gap
            corridor_match = min(float(actor_lateral), float(best_stop_lateral)) <= corridor_limit
            association_valid = bool((actor_above_stopline or stop_distance_match) and corridor_match)
            if not association_valid:
                continue

            actor_gap = abs(max(0.0, actor_delta - visual_distance))
            stop_score = max(0.0, 1.0 - min(best_stop_gap, max_stop_gap) / max(0.1, max_stop_gap))
            actor_score = max(0.0, 1.0 - min(actor_gap, max_actor_ahead) / max(0.1, max_actor_ahead))
            lateral_score = max(0.0, 1.0 - min(float(actor_lateral), corridor_limit) / max(0.1, corridor_limit))
            score = 0.45 * stop_score + 0.40 * actor_score + 0.15 * lateral_score
            if actor_above_stopline:
                score += 0.20
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_debug = {
                    "actor_delta": actor_delta,
                    "actor_lateral": actor_lateral,
                    "best_stop_delta": best_stop_delta,
                    "best_stop_gap": best_stop_gap,
                    "best_stop_lateral": best_stop_lateral,
                    "actor_above_stopline": actor_above_stopline,
                    "corridor_match": corridor_match,
                    "score": min(1.0, score),
                }

        if best_candidate is None or best_debug is None:
            fields["visual_stopline_light_association_reason"] = "not_associated_with_visual_stopline"
            return None, fields

        associated_color = str(best_candidate.get("tl_color_raw") or "unknown").lower()
        if associated_color not in {"red", "yellow", "green"}:
            associated_color = "unknown"
        associated_distance = best_debug["best_stop_delta"]
        if associated_distance is None or abs(float(associated_distance) - visual_distance) > max_stop_gap:
            associated_distance = best_debug["actor_delta"]

        fields.update({
            "associated_light_id": best_candidate.get("actor_id"),
            "associated_light_color": associated_color,
            "associated_light_source": str(best_candidate.get("source") or "traffic_light_detection"),
            "associated_light_distance": round(float(associated_distance), 3),
            "associated_light_roi_score": round(float(best_debug["score"]), 3),
            "associated_light_above_stopline": bool(best_debug["actor_above_stopline"]),
            "associated_light_horizontal_overlap": bool(best_debug["corridor_match"]),
            "visual_stopline_light_association_valid": True,
            "visual_stopline_light_association_reason": "associated_with_visual_stopline",
            "far_carla_candidate_ignored": bool(fields["carla_candidate_suspect"]),
        })
        return best_candidate, fields

    def make_lock_meta(self, *, active: bool = False) -> dict:
        lock = self.locked_red_light or {}
        return {
            "locked_light_id": lock.get("locked_light_id"),
            "locked_stop_point_x": lock.get("locked_stop_point_x"),
            "locked_stop_point_y": lock.get("locked_stop_point_y"),
            "locked_route_s": lock.get("locked_route_s"),
            "locked_light_active": bool(active),
            "locked_light_color": "",
            "candidate_switch_blocked": False,
            "candidate_switch_block_reason": "",
            "green_release_source": "",
            "red_lock_release_reason": "",
            "lock_match_source": "",
        }

    def stop_point_distance_to_lock(self, stop_point: dict) -> float:
        lock = self.locked_red_light or {}
        if lock.get("locked_stop_point_x") is None or lock.get("locked_stop_point_y") is None:
            return float("inf")
        return self.estimator.distance_xy(
            stop_point,
            {"x": lock["locked_stop_point_x"], "y": lock["locked_stop_point_y"]},
        )

    def locked_candidate_copy(self, candidate: dict, route_points: list[dict], cumulative: list[float]) -> dict:
        if not self.locked_red_light:
            return dict(candidate)

        best_stop_point = None
        best_score = float("inf")
        locked_route_s = self.locked_red_light.get("locked_route_s")
        for stop_point, _source, _index in self.estimator.candidate_stop_points(candidate):
            if stop_point.get("x") is None or stop_point.get("y") is None:
                continue
            route_s, _lateral, _route_index, _yaw = self.estimator.project_point_to_route(
                stop_point,
                route_points,
                cumulative,
            )
            route_score = abs(float(route_s) - float(locked_route_s)) if locked_route_s is not None else 0.0
            score = route_score + self.stop_point_distance_to_lock(stop_point)
            if score < best_score:
                best_score = score
                best_stop_point = dict(stop_point)

        locked_candidate = dict(candidate)
        if best_stop_point is not None:
            locked_candidate["stop_waypoints"] = [best_stop_point]
        return locked_candidate

    def find_locked_candidate(self, detections: list[dict], route_points: list[dict]) -> tuple[Optional[dict], str]:
        if not self.locked_red_light:
            return None, ""

        locked_id = self.locked_red_light.get("locked_light_id")
        cumulative = self.estimator.cumulative_route_distances(route_points)

        for candidate in detections:
            if candidate.get("actor_id") == locked_id:
                return self.locked_candidate_copy(candidate, route_points, cumulative), "exact_actor"

        best_candidate = None
        best_score = float("inf")
        locked_route_s = self.locked_red_light.get("locked_route_s")
        for candidate in detections:
            for stop_point, _source, _index in self.estimator.candidate_stop_points(candidate):
                if stop_point.get("x") is None or stop_point.get("y") is None:
                    continue
                route_s, _lateral, _route_index, _yaw = self.estimator.project_point_to_route(
                    stop_point,
                    route_points,
                    cumulative,
                )
                route_gap = abs(float(route_s) - float(locked_route_s)) if locked_route_s is not None else float("inf")
                point_gap = self.stop_point_distance_to_lock(stop_point)
                score = min(route_gap, point_gap)
                if score < best_score:
                    best_score = score
                    best_candidate = candidate

        if best_candidate is not None and best_score <= 3.0:
            return self.locked_candidate_copy(best_candidate, route_points, cumulative), "same_stop_point"

        lock = self.locked_red_light
        if lock.get("locked_stop_point_x") is None or lock.get("locked_stop_point_y") is None:
            return None, "missing_locked_stop_point"
        synthetic = {
            "actor_id": lock.get("locked_light_id"),
            "tl_detected": True,
            "tl_color_raw": lock.get("locked_light_color") or "red",
            "tl_confidence": 1.0,
            "source": "locked_red_light_memory",
            "location": {
                "x": lock["locked_stop_point_x"],
                "y": lock["locked_stop_point_y"],
                "z": 0.0,
            },
            "yaw_deg": 0.0,
            "stop_waypoints": [{
                "x": lock["locked_stop_point_x"],
                "y": lock["locked_stop_point_y"],
                "z": 0.0,
                "yaw_deg": 0.0,
            }],
        }
        return synthetic, "memory"

    def choose_distance_with_lock(
        self,
        *,
        ego: EgoStatus,
        route_points: list[dict],
        detections: list[dict],
    ):
        normal_distance = self.estimator.choose_candidate(
            ego=ego,
            route_points=route_points,
            detections=detections,
        )
        lock_meta = self.make_lock_meta(active=False)
        if not self.locked_red_light:
            return normal_distance, lock_meta

        locked_candidate, match_source = self.find_locked_candidate(detections, route_points)
        if locked_candidate is None:
            return normal_distance, lock_meta

        locked_distance = self.estimator.choose_candidate(
            ego=ego,
            route_points=route_points,
            detections=[locked_candidate],
        )
        lock_meta = self.make_lock_meta(active=True)
        lock_meta["lock_match_source"] = match_source
        lock_meta["locked_light_color"] = str(locked_candidate.get("tl_color_raw") or "unknown").lower()

        if (
            normal_distance.selected_light_id is not None
            and normal_distance.selected_light_id != locked_distance.selected_light_id
        ):
            lock_meta["candidate_switch_blocked"] = True
            lock_meta["candidate_switch_block_reason"] = (
                f"locked_red_light_active:blocked_selected_id={normal_distance.selected_light_id}"
            )

        return locked_distance, lock_meta

    def acquire_red_lock(self, distance, color_filtered: str):
        if (
            not distance.candidate
            or distance.selected_light_id is None
            or distance.selected_stop_point_x is None
            or distance.selected_stop_point_y is None
            or distance.stop_point_route_distance_m is None
        ):
            return
        self.locked_red_light = {
            "locked_light_id": distance.selected_light_id,
            "locked_stop_point_x": distance.selected_stop_point_x,
            "locked_stop_point_y": distance.selected_stop_point_y,
            "locked_route_s": distance.stop_point_route_distance_m,
            "locked_light_color": color_filtered,
        }

    def acquire_visual_corrected_red_lock(
        self,
        *,
        distance,
        association_fields: dict,
        corrected_point: Optional[dict],
        corrected_route_s,
        color_filtered: str,
    ):
        if not corrected_point or corrected_point.get("x") is None or corrected_point.get("y") is None:
            return
        locked_id = association_fields.get("associated_light_id")
        if locked_id is None:
            locked_id = distance.selected_light_id
        if locked_id is None and self.locked_red_light:
            locked_id = self.locked_red_light.get("locked_light_id")
        if locked_id is None:
            return
        self.locked_red_light = {
            "locked_light_id": locked_id,
            "locked_stop_point_x": round(float(corrected_point["x"]), 3),
            "locked_stop_point_y": round(float(corrected_point["y"]), 3),
            "locked_route_s": (
                round(float(corrected_route_s), 3)
                if corrected_route_s is not None
                else None
            ),
            "locked_light_color": color_filtered,
        }

    def lock_fields(self, lock_meta: dict, release_reason: str) -> dict:
        fields = dict(lock_meta)
        if self.locked_red_light:
            fields.update({
                "locked_light_id": self.locked_red_light.get("locked_light_id"),
                "locked_stop_point_x": self.locked_red_light.get("locked_stop_point_x"),
                "locked_stop_point_y": self.locked_red_light.get("locked_stop_point_y"),
                "locked_route_s": self.locked_red_light.get("locked_route_s"),
                "locked_light_active": True,
                "locked_light_color": fields.get("locked_light_color") or self.locked_red_light.get("locked_light_color"),
            })
        fields["red_lock_release_reason"] = release_reason
        return fields

    def default_locked_visual_stopline_fields(self) -> dict:
        return {
            "locked_visual_stopline_active": False,
            "locked_visual_stopline_route_s": None,
            "locked_visual_distance_m": None,
            "raw_visual_distance_m": None,
            "distance_source_for_decision": "carla_stop_waypoint_fallback",
            "distance_source_before_consistency": "carla_stop_waypoint_fallback",
            "distance_source_after_consistency": "carla_stop_waypoint_fallback",
            "locked_visual_consistency_valid": False,
            "locked_visual_consistency_reject_reason": "",
            "locked_visual_distance_vs_carla_error_m": None,
            "locked_visual_distance_vs_raw_error_m": None,
            "locked_visual_reanchored_to_carla_stop": False,
            "visual_distance_tracking_error_m": None,
            "visual_distance_monotonic_violation": False,
            "simple_red_visual_rule": False,
            "red_visual_decision_distance_m": None,
            "red_visual_slow_threshold_m": None,
            "red_visual_stop_threshold_m": None,
            "red_visual_hard_stop_threshold_m": None,
            "carla_stop_ignored_because_visual_rule": False,
            "locked_visual_ignored_because_visual_rule": False,
            "carla_aligned_lock_ignored_because_visual_rule": False,
            "visual_lock_active": False,
            "visual_lock_route_s": None,
            "visual_lock_distance_m": None,
            "visual_lock_created": False,
            "visual_lock_cleared_reason": "",
            "visual_lock_suspect": False,
            "visual_lock_reject_reason": "",
            "visual_lock_vs_raw_error_m": None,
            "visual_lock_vs_front_error_m": None,
            "visual_lock_used_for_commit": False,
            "raw_visual_used_instead_of_bad_lock": False,
        }

    def default_red_visual_failsafe_fields(self) -> dict:
        return {
            "red_visual_near_light_failsafe": False,
            "red_visual_near_light_failsafe_reason": "",
            "red_visual_approach_elapsed_s": 0.0,
            "red_visual_distance_stuck": False,
            "red_visual_stuck_stop": False,
            "associated_light_distance_for_failsafe": None,
            "red_stop_2m_any_valid_distance": False,
            "red_stop_2m_any_valid_distance_source": "",
            "red_stop_2m_any_valid_distance_value": None,
            "red_stop_2m_any_valid_distance_reason": "",
            "red_corrected_point_overrun_full_brake": False,
            "red_corrected_point_overrun_source": "",
        }

    @staticmethod
    def default_stop_point_correction_fields() -> dict:
        return {
            "stop_point_corrected_from_bad_carla": False,
            "stop_point_correction_reason": "",
            "original_carla_stop_distance": None,
            "original_selected_stop_distance": None,
            "visual_stopline_distance_for_correction": None,
            "corrected_stop_point_source": "",
            "corrected_stop_point_world_x": None,
            "corrected_stop_point_world_y": None,
            "corrected_stop_point_world_z": None,
            "corrected_point_ignored_no_valid_association": False,
            "corrected_point_ignored_reason": "",
            "corrected_point_would_have_distance": None,
            "tl_speed_override_cleared_for_no_association": False,
            "visual_correction_rejected_associated_light_too_far": False,
            "visual_correction_reject_assoc_distance": None,
            "visual_correction_reject_visual_distance": None,
            "correction_applied_to_decision": False,
        }

    def red_stop_hold_fields(self, *, cleared_by_green: bool = False) -> dict:
        return {
            "red_stop_hold_active": bool(self.red_stop_hold_active),
            "red_stop_hold_reason": self.red_stop_hold_reason if self.red_stop_hold_active else "",
            "red_stop_hold_cleared_by_green": bool(cleared_by_green),
        }

    @staticmethod
    def default_effective_stop_context_fields() -> dict:
        return {
            "effective_stop_distance_m": None,
            "effective_stop_source": "",
            "effective_stop_point_x": None,
            "effective_stop_point_y": None,
            "effective_stop_valid": False,
            "effective_stop_reason": "",
            "distance_conflict_detected": False,
            "ignored_distance_sources": "",
            "visual_distance_primary": False,
            "last_good_visual_distance_m": None,
            "last_good_visual_age_s": None,
            "visual_last_good_hold_active": False,
            "visual_stuck_overrun_detected": False,
            "visual_stuck_overrun_reason": "",
            "fallback_blocked_by_recent_visual": False,
            "fallback_blocked_source": "",
        }

    def resolve_red_stop_context(
        self,
        *,
        color_filtered: str,
        association_fields: dict,
        stopline_fields: dict,
        locked_visual_fields: dict,
        debug_fields: dict,
        stop_point_correction_fields: dict,
        front_dist_for_commit,
        selected_stop_point_suspect: bool,
        red_visual_approach_elapsed_s: float,
        distance,
    ) -> dict:
        fields = self.default_effective_stop_context_fields()
        now_s = time.time()
        color = str(color_filtered or "unknown").lower()
        associated_color = str(association_fields.get("associated_light_color") or "unknown").lower()
        association_valid = bool(association_fields.get("visual_stopline_light_association_valid"))
        valid_red_association = bool(
            associated_color == "red"
            or association_valid
            or bool(association_fields.get("visual_association_latched"))
            or bool(association_fields.get("associated_light_above_stopline"))
        )

        if color == "green" or associated_color == "green":
            fields["effective_stop_reason"] = "green_release"
            return fields

        if color != "red":
            fields["effective_stop_reason"] = "not_red"
            return fields

        if not valid_red_association:
            fields["effective_stop_reason"] = "no_valid_red_association"
            return fields

        if stop_point_correction_fields.get("visual_correction_rejected_associated_light_too_far"):
            fields["effective_stop_reason"] = "visual_correction_rejected_associated_light_too_far"
            return fields

        visual_dist = self.optional_float(stopline_fields.get("visual_dist_filtered"))
        associated_dist = self.optional_float(association_fields.get("associated_light_distance"))
        front_dist = self.optional_float(front_dist_for_commit)
        locked_dist = self.optional_float(locked_visual_fields.get("locked_visual_distance_m"))
        carla_dist = self.optional_float(distance.stop_point_distance_m)
        corrected_active = bool(stop_point_correction_fields.get("stop_point_corrected_from_bad_carla"))
        ignored_sources: list[str] = []
        raw_visual_point = debug_fields.get("raw_visual_stopline_estimated_world")
        last_good_age_s = (
            max(0.0, now_s - float(self.last_good_visual_stop_s))
            if self.last_good_visual_stop_s is not None
            else None
        )
        last_good_distance = self.optional_float(self.last_good_visual_stop_distance_m)
        fields["last_good_visual_distance_m"] = (
            round(float(last_good_distance), 3) if last_good_distance is not None else None
        )
        fields["last_good_visual_age_s"] = (
            round(float(last_good_age_s), 3) if last_good_age_s is not None else None
        )

        def set_effective(source: str, value, reason: str, point: Optional[dict] = None) -> dict:
            value_f = self.optional_float(value)
            fields["effective_stop_distance_m"] = round(float(value_f), 3) if value_f is not None else None
            fields["effective_stop_source"] = source
            fields["effective_stop_valid"] = value_f is not None
            fields["effective_stop_reason"] = reason
            if point:
                fields["effective_stop_point_x"] = point.get("x")
                fields["effective_stop_point_y"] = point.get("y")
            return fields

        visual_primary = bool(
            visual_dist is not None
            and 0.0 < visual_dist <= 10.0
            and association_valid
            and bool(stopline_fields.get("visual_stopline_detected"))
        )
        if visual_primary:
            self.last_good_visual_stop_distance_m = visual_dist
            self.last_good_visual_stop_s = now_s
            self.last_good_visual_stop_point = raw_visual_point
            for name, source_dist in (
                ("associated_light", associated_dist),
                ("front", front_dist),
                ("locked_visual", locked_dist),
            ):
                if source_dist is not None and abs(visual_dist - source_dist) > 1.5:
                    ignored_sources.append(name)
            associated_overrun_valid = bool(
                associated_dist is not None and associated_dist <= 0.0
            )
            front_overrun_valid = bool(
                front_dist is not None
                and front_dist <= 0.0
                and not selected_stop_point_suspect
                and not bool(association_fields.get("carla_candidate_suspect"))
            )
            if (
                associated_color == "red"
                and visual_dist > 3.0
                and red_visual_approach_elapsed_s >= 1.0
                and (associated_overrun_valid or front_overrun_valid)
            ):
                source = "associated_light" if associated_overrun_valid else "front"
                fields["visual_stuck_overrun_detected"] = True
                fields["visual_stuck_overrun_reason"] = f"{source}_overrun_while_visual_stuck"
                fields["distance_conflict_detected"] = True
                fields["ignored_distance_sources"] = ",".join(ignored_sources)
                fields["visual_distance_primary"] = True
                return set_effective(
                    "visual_stuck_overrun",
                    0.0,
                    "red_visual_stuck_overrun_full_brake",
                    raw_visual_point,
                )
            fields["distance_conflict_detected"] = bool(ignored_sources)
            fields["ignored_distance_sources"] = ",".join(ignored_sources)
            fields["visual_distance_primary"] = True
            source = "visual_corrected" if corrected_active else "visual"
            return set_effective(
                source,
                visual_dist,
                "visual_primary",
                raw_visual_point,
            )

        recent_visual_available = bool(
            last_good_distance is not None
            and 0.0 < last_good_distance <= 10.0
            and last_good_age_s is not None
            and last_good_age_s <= 0.75
        )
        if recent_visual_available:
            blocked = []
            for name, source_dist in (
                ("locked_visual", locked_dist),
                ("front", front_dist),
                ("associated_light", associated_dist),
            ):
                if source_dist is not None and source_dist <= 2.0:
                    blocked.append(name)
            fields["visual_last_good_hold_active"] = True
            fields["fallback_blocked_by_recent_visual"] = bool(blocked)
            fields["fallback_blocked_source"] = ",".join(blocked)
            fields["ignored_distance_sources"] = ",".join(blocked)
            return set_effective(
                "visual_last_good_hold",
                last_good_distance,
                "recent_visual_last_good_hold",
                self.last_good_visual_stop_point,
            )

        fallback_sources = (
            ("locked_visual", locked_dist, debug_fields.get("locked_stop_point_world")),
            ("front", front_dist, debug_fields.get("selected_stop_point_world")),
            ("associated_light", associated_dist, debug_fields.get("selected_stop_point_world")),
            ("carla", carla_dist, debug_fields.get("selected_stop_point_world")),
        )
        for source, value, point in fallback_sources:
            value_f = self.optional_float(value)
            if value_f is None or value_f <= 0.0:
                continue
            if source in {"front", "carla"} and selected_stop_point_suspect:
                ignored_sources.append(source)
                continue
            if source == "front" and bool(association_fields.get("carla_candidate_suspect")):
                ignored_sources.append(source)
                continue
            fields["ignored_distance_sources"] = ",".join(ignored_sources)
            return set_effective(source, value_f, "fallback_" + source, point)

        fields["ignored_distance_sources"] = ",".join(ignored_sources)
        fields["effective_stop_reason"] = "no_valid_stop_distance"
        return fields

    @staticmethod
    def optional_float(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def world_point_dict(point: Optional[dict]) -> Optional[dict]:
        if not point or point.get("x") is None or point.get("y") is None:
            return None
        return {
            "x": round(float(point["x"]), 3),
            "y": round(float(point["y"]), 3),
            "z": round(float(point.get("z", 0.0) or 0.0), 3),
        }

    def route_point_at_s(self, route_points: list[dict], target_s) -> Optional[dict]:
        s = self.optional_float(target_s)
        if s is None or not route_points:
            return None
        if len(route_points) == 1:
            point = route_points[0]
            return {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "z": float(point.get("z", 0.0) or 0.0),
            }

        cumulative = self.estimator.cumulative_route_distances(route_points)
        if not cumulative:
            return None

        if s <= cumulative[0]:
            point = route_points[0]
            return {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "z": float(point.get("z", 0.0) or 0.0),
            }
        if s >= cumulative[-1]:
            point = route_points[-1]
            return {
                "x": float(point["x"]),
                "y": float(point["y"]),
                "z": float(point.get("z", 0.0) or 0.0),
            }

        for index in range(len(route_points) - 1):
            start_s = cumulative[index]
            end_s = cumulative[index + 1]
            if not (start_s <= s <= end_s):
                continue
            start = route_points[index]
            end = route_points[index + 1]
            seg_len = max(1e-6, end_s - start_s)
            t = (s - start_s) / seg_len
            return {
                "x": float(start["x"]) + (float(end["x"]) - float(start["x"])) * t,
                "y": float(start["y"]) + (float(end["y"]) - float(start["y"])) * t,
                "z": float(start.get("z", 0.0) or 0.0)
                + (float(end.get("z", 0.0) or 0.0) - float(start.get("z", 0.0) or 0.0)) * t,
            }
        return None

    def debug_draw_fields(
        self,
        *,
        route_points: list[dict],
        distance,
        stopline_fields: dict,
        locked_visual_fields: dict,
    ) -> dict:
        front_point = None
        if distance.front_bumper_x is not None and distance.front_bumper_y is not None:
            front_point = {
                "x": float(distance.front_bumper_x),
                "y": float(distance.front_bumper_y),
                "z": 0.0,
            }

        selected_stop_point = None
        if distance.selected_stop_point_x is not None and distance.selected_stop_point_y is not None:
            selected_stop_point = {
                "x": float(distance.selected_stop_point_x),
                "y": float(distance.selected_stop_point_y),
                "z": 0.0,
            }

        locked_stop_point = None
        lock = self.locked_red_light or {}
        if lock.get("locked_stop_point_x") is not None and lock.get("locked_stop_point_y") is not None:
            locked_stop_point = {
                "x": float(lock["locked_stop_point_x"]),
                "y": float(lock["locked_stop_point_y"]),
                "z": 0.0,
            }

        front_s = self.optional_float(distance.front_bumper_route_s)
        raw_dist = self.optional_float(stopline_fields.get("visual_dist_filtered"))
        raw_visual_route_s = front_s + raw_dist if front_s is not None and raw_dist is not None else None
        raw_visual_point = self.route_point_at_s(route_points, raw_visual_route_s)

        carla_aligned_route_s = None
        if self.locked_visual_stopline and self.locked_visual_stopline.get("anchored_to_carla_stop"):
            carla_aligned_route_s = self.locked_visual_stopline.get("route_s")
        elif locked_visual_fields.get("distance_source_after_consistency") == "carla_aligned_locked_visual_stopline":
            carla_aligned_route_s = locked_visual_fields.get("locked_visual_stopline_route_s")
        elif distance.stop_point_route_distance_m is not None:
            carla_aligned_route_s = distance.stop_point_route_distance_m
        carla_aligned_point = self.route_point_at_s(route_points, carla_aligned_route_s)

        alignment_error = None
        if distance.stop_point_distance_m is not None and raw_dist is not None:
            alignment_error = abs(float(distance.stop_point_distance_m) - raw_dist)

        suspect_threshold = float(self.get_parameter("traffic_light_stop_point_suspect_visual_error_m").value)
        selected_stop_point_suspect = bool(
            alignment_error is not None
            and alignment_error > suspect_threshold
            and bool(stopline_fields.get("visual_stopline_candidate_valid"))
            and bool(stopline_fields.get("visual_stopline_light_association_valid"))
        )

        return {
            "debug_draw_stop_points_enabled": bool(
                self.get_parameter("traffic_light_debug_draw_stop_points").value
            ),
            "selected_stop_point_world": self.world_point_dict(selected_stop_point),
            "final_stop_point_world": self.world_point_dict(selected_stop_point),
            "locked_stop_point_world": self.world_point_dict(locked_stop_point),
            "front_bumper_world": self.world_point_dict(front_point),
            "raw_visual_stopline_estimated_world": self.world_point_dict(raw_visual_point),
            "carla_aligned_visual_stopline_world": self.world_point_dict(carla_aligned_point),
            "stop_point_visual_alignment_error_m": (
                round(float(alignment_error), 3) if alignment_error is not None else None
            ),
            "selected_stop_point_suspect": selected_stop_point_suspect,
        }

    def ensure_carla_debug_world(self):
        if self.carla_world is not None:
            return self.carla_world
        if self.carla_debug_connect_attempted:
            return None
        self.carla_debug_connect_attempted = True
        try:
            self.carla = load_carla(str(self.get_parameter("carla_root").value))
            self.carla_client = self.carla.Client(
                str(self.get_parameter("host").value),
                int(self.get_parameter("port").value),
            )
            self.carla_client.set_timeout(2.0)
            self.carla_world = self.carla_client.get_world()
            return self.carla_world
        except Exception as exc:
            self.get_logger().warning(f"CARLA debug draw unavailable: {exc}")
            return None

    def draw_debug_point(self, point: Optional[dict], *, color, label: str):
        if not point:
            return
        world = self.ensure_carla_debug_world()
        if world is None or self.carla is None:
            return
        z_offset = float(self.get_parameter("traffic_light_debug_draw_z_offset_m").value)
        life_time = float(self.get_parameter("traffic_light_debug_draw_life_time_s").value)
        location = self.carla.Location(
            x=float(point["x"]),
            y=float(point["y"]),
            z=float(point.get("z", 0.0) or 0.0) + z_offset,
        )
        world.debug.draw_point(location, size=0.16, color=color, life_time=life_time)
        world.debug.draw_string(
            location + self.carla.Location(z=0.35),
            label,
            draw_shadow=True,
            color=color,
            life_time=life_time,
        )

    def draw_stop_point_debug(self, debug_fields: dict):
        if not bool(debug_fields.get("debug_draw_stop_points_enabled")):
            return
        world = self.ensure_carla_debug_world()
        if world is None or self.carla is None:
            return
        colors = {
            "selected": self.carla.Color(255, 0, 0),
            "locked": self.carla.Color(180, 0, 255),
            "front": self.carla.Color(0, 80, 255),
            "carla_aligned": self.carla.Color(255, 140, 0),
            "raw_visual": self.carla.Color(0, 255, 0),
        }
        selected_label_distance = (
            debug_fields.get("visual_stopline_distance_for_correction")
            if debug_fields.get("stop_point_corrected_from_bad_carla")
            else debug_fields.get("front_dist_for_commit")
        )
        self.draw_debug_point(
            debug_fields.get("selected_stop_point_world"),
            color=colors["selected"],
            label=f"sel_stop d={selected_label_distance}",
        )
        self.draw_debug_point(
            debug_fields.get("locked_stop_point_world"),
            color=colors["locked"],
            label="locked",
        )
        self.draw_debug_point(
            debug_fields.get("front_bumper_world"),
            color=colors["front"],
            label="front",
        )
        self.draw_debug_point(
            debug_fields.get("carla_aligned_visual_stopline_world"),
            color=colors["carla_aligned"],
            label=f"locked_visual={debug_fields.get('locked_dist_for_commit')}",
        )
        self.draw_debug_point(
            debug_fields.get("raw_visual_stopline_estimated_world"),
            color=colors["raw_visual"],
            label=f"visual raw={debug_fields.get('raw_visual_dist_for_commit')}",
        )

    def update_locked_visual_stopline(
        self,
        *,
        front_bumper_route_s,
        carla_stop_point_distance_m,
        stop_point_route_distance_m,
        selected_delta_s_m,
        stopline_fields: dict,
        association_fields: dict,
        color_filtered: str,
    ) -> dict:
        fields = self.default_locked_visual_stopline_fields()
        raw_visual_distance = stopline_fields.get("visual_dist_filtered")
        fields["raw_visual_distance_m"] = raw_visual_distance
        fields["red_visual_slow_threshold_m"] = float(self.get_parameter("red_visual_slow_distance_m").value)
        fields["red_visual_stop_threshold_m"] = float(self.get_parameter("red_visual_stop_distance_m").value)
        fields["red_visual_hard_stop_threshold_m"] = float(
            self.get_parameter("red_visual_hard_stop_distance_m").value
        )
        now = time.time()

        associated_color = str(association_fields.get("associated_light_color") or color_filtered or "").lower()
        if associated_color == "green":
            if self.locked_visual_stopline:
                fields["visual_lock_cleared_reason"] = "associated_light_green"
            self.locked_visual_stopline = None

        front_s = self.optional_float(front_bumper_route_s)
        raw_dist = self.optional_float(raw_visual_distance)
        carla_dist = self.optional_float(carla_stop_point_distance_m)
        carla_route_s = self.optional_float(stop_point_route_distance_m)
        selected_delta_s = self.optional_float(selected_delta_s_m)

        association_valid = bool(association_fields.get("visual_stopline_light_association_valid"))
        associated_light_id = association_fields.get("associated_light_id")
        acquire_max_m = float(self.get_parameter("locked_visual_stopline_acquire_max_m").value)
        tolerance_m = float(self.get_parameter("locked_visual_stopline_consistency_tolerance_m").value)
        carla_candidate_suspect = bool(association_fields.get("carla_candidate_suspect"))
        carla_reliable_close = bool(
            not carla_candidate_suspect
            and carla_dist is not None
            and 0.0 <= carla_dist <= 12.0
            and carla_route_s is not None
            and selected_delta_s is not None
        )
        can_acquire = bool(
            front_s is not None
            and raw_dist is not None
            and associated_color in {"red", "yellow"}
            and association_valid
            and bool(stopline_fields.get("visual_stopline_detected"))
            and raw_dist <= acquire_max_m
        )

        if can_acquire and (
            not self.locked_visual_stopline
            or self.locked_visual_stopline.get("light_id") != associated_light_id
        ):
            lock_route_s = front_s + raw_dist if associated_color == "red" else (
                carla_route_s if carla_reliable_close else front_s + raw_dist
            )
            self.locked_visual_stopline = {
                "route_s": lock_route_s,
                "light_id": associated_light_id,
                "created_s": now,
                "last_distance_m": (lock_route_s - front_s) if lock_route_s is not None else raw_dist,
                "last_front_bumper_route_s": front_s,
                "anchored_to_carla_stop": bool(carla_reliable_close and associated_color != "red"),
            }
            fields["visual_lock_created"] = True

        lock = self.locked_visual_stopline
        if not lock or front_s is None:
            fields["distance_source_for_decision"] = (
                "visual_stopline_simple_red_rule"
                if associated_color == "red" and stopline_fields.get("using_visual_stopline")
                else "raw_visual_stopline"
                if stopline_fields.get("using_visual_stopline")
                else "carla_stop_waypoint_fallback"
            )
            fields["distance_source_before_consistency"] = fields["distance_source_for_decision"]
            fields["distance_source_after_consistency"] = fields["distance_source_for_decision"]
            return fields

        locked_distance = float(lock["route_s"]) - front_s
        previous_locked_distance = lock.get("last_distance_m")
        vehicle_moved_forward = (
            lock.get("last_front_bumper_route_s") is not None
            and front_s > float(lock.get("last_front_bumper_route_s")) + 0.05
        )
        monotonic_violation = bool(
            vehicle_moved_forward
            and previous_locked_distance is not None
            and locked_distance > float(previous_locked_distance) + 0.25
        )

        tracking_error = None
        if raw_dist is not None:
            tracking_error = abs(raw_dist - locked_distance)

        carla_error = abs(locked_distance - carla_dist) if carla_dist is not None else None
        raw_error = abs(locked_distance - raw_dist) if raw_dist is not None else None
        if associated_color == "red":
            stop_threshold_m = float(self.get_parameter("red_visual_stop_distance_m").value)
            raw_says_far = raw_dist is not None and raw_dist > 3.0
            front_says_far = carla_dist is not None and carla_dist > 3.0
            lock_says_stop = locked_distance <= stop_threshold_m
            raw_disagrees_far = raw_error is not None and raw_error > tolerance_m and raw_says_far
            front_disagrees_far = carla_error is not None and carla_error > tolerance_m and front_says_far
            visual_lock_suspect = bool(
                (lock_says_stop and raw_says_far and front_says_far)
                or raw_disagrees_far
                or front_disagrees_far
            )
            if lock_says_stop and raw_says_far and front_says_far:
                visual_lock_reject_reason = "lock_disagrees_with_raw_and_front"
            elif raw_disagrees_far and front_disagrees_far:
                visual_lock_reject_reason = "lock_disagrees_with_raw_and_front"
            elif raw_disagrees_far:
                visual_lock_reject_reason = "lock_disagrees_with_raw"
            elif front_disagrees_far:
                visual_lock_reject_reason = "lock_disagrees_with_front"
            else:
                visual_lock_reject_reason = ""

            lock["last_distance_m"] = locked_distance
            lock["last_front_bumper_route_s"] = front_s
            source = "visual_stopline_locked_red_rule"
            fields.update({
                "locked_visual_stopline_active": True,
                "locked_visual_stopline_route_s": round(float(lock["route_s"]), 3),
                "locked_visual_distance_m": round(float(locked_distance), 3),
                "distance_source_for_decision": source,
                "distance_source_before_consistency": source,
                "distance_source_after_consistency": source,
                "locked_visual_consistency_valid": not visual_lock_suspect,
                "locked_visual_consistency_reject_reason": visual_lock_reject_reason,
                "locked_visual_distance_vs_carla_error_m": (
                    round(float(carla_error), 3) if carla_error is not None else None
                ),
                "locked_visual_distance_vs_raw_error_m": (
                    round(float(raw_error), 3) if raw_error is not None else None
                ),
                "locked_visual_reanchored_to_carla_stop": False,
                "visual_distance_tracking_error_m": round(float(tracking_error), 3)
                if tracking_error is not None
                else None,
                "visual_distance_monotonic_violation": monotonic_violation,
                "simple_red_visual_rule": True,
                "red_visual_decision_distance_m": round(float(locked_distance), 3),
                "carla_stop_ignored_because_visual_rule": carla_dist is not None,
                "locked_visual_ignored_because_visual_rule": False,
                "carla_aligned_lock_ignored_because_visual_rule": bool(lock.get("anchored_to_carla_stop")),
                "visual_lock_active": True,
                "visual_lock_route_s": round(float(lock["route_s"]), 3),
                "visual_lock_distance_m": round(float(locked_distance), 3),
                "visual_lock_suspect": visual_lock_suspect,
                "visual_lock_reject_reason": visual_lock_reject_reason,
                "visual_lock_vs_raw_error_m": (
                    round(float(raw_error), 3) if raw_error is not None else None
                ),
                "visual_lock_vs_front_error_m": (
                    round(float(carla_error), 3) if carla_error is not None else None
                ),
            })
            return fields

        visual_candidate_valid = bool(stopline_fields.get("visual_stopline_candidate_valid"))
        stopline_detected = bool(stopline_fields.get("visual_stopline_detected"))
        consistency_valid = True
        reject_reason = ""
        if associated_color not in {"red", "yellow"}:
            consistency_valid = False
            reject_reason = "associated_light_not_red_or_yellow"
        elif not association_valid:
            consistency_valid = False
            reject_reason = "visual_stopline_light_association_invalid"
        elif not visual_candidate_valid or not stopline_detected:
            consistency_valid = False
            reject_reason = "visual_stopline_not_currently_detected"
        elif carla_error is not None and carla_error > tolerance_m:
            consistency_valid = False
            if raw_error is not None and raw_error > tolerance_m:
                reject_reason = "locked_distance_disagrees_with_carla_and_raw_visual"
            else:
                reject_reason = "locked_distance_disagrees_with_carla"
        elif raw_error is not None and raw_error > tolerance_m:
            consistency_valid = False
            reject_reason = "locked_distance_disagrees_with_raw_visual"
        elif monotonic_violation:
            consistency_valid = False
            reject_reason = "locked_visual_distance_monotonic_violation"

        source_before = "locked_visual_stopline"
        source_after = (
            "carla_aligned_locked_visual_stopline"
            if consistency_valid and bool(lock.get("anchored_to_carla_stop"))
            else "locked_visual_stopline"
            if consistency_valid
            else "raw_visual_stopline_consistency_fallback"
            if raw_dist is not None and association_valid and visual_candidate_valid
            else "carla_stop_waypoint_fallback"
            if carla_dist is not None
            else "carla_fallback"
        )

        lock["last_distance_m"] = locked_distance
        lock["last_front_bumper_route_s"] = front_s

        fields.update({
            "locked_visual_stopline_active": True,
            "locked_visual_stopline_route_s": round(float(lock["route_s"]), 3),
            "locked_visual_distance_m": round(float(locked_distance), 3),
            "distance_source_for_decision": source_after,
            "distance_source_before_consistency": source_before,
            "distance_source_after_consistency": source_after,
            "locked_visual_consistency_valid": bool(consistency_valid),
            "locked_visual_consistency_reject_reason": reject_reason,
            "locked_visual_distance_vs_carla_error_m": (
                round(float(carla_error), 3) if carla_error is not None else None
            ),
            "locked_visual_distance_vs_raw_error_m": (
                round(float(raw_error), 3) if raw_error is not None else None
            ),
            "visual_distance_tracking_error_m": round(float(tracking_error), 3)
            if tracking_error is not None
            else None,
            "visual_distance_monotonic_violation": monotonic_violation,
        })
        if not consistency_valid and carla_reliable_close:
            self.locked_visual_stopline.update({
                "route_s": carla_route_s,
                "last_distance_m": carla_dist,
                "last_front_bumper_route_s": front_s,
                "anchored_to_carla_stop": True,
                "reanchored_s": now,
            })
            fields["locked_visual_reanchored_to_carla_stop"] = True
        return fields

    def clear_locked_visual_stopline_if_needed(
        self,
        *,
        color_filtered: str,
        decision,
        locked_fields: dict,
    ) -> str:
        locked_distance = locked_fields.get("locked_visual_distance_m")
        try:
            locked_distance_f = float(locked_distance)
        except (TypeError, ValueError):
            locked_distance_f = None

        clear_reason = ""
        if str(color_filtered).lower() == "green":
            clear_reason = "associated_light_green"
        elif decision.state == GREEN_RELEASE:
            clear_reason = "green_release"
        elif bool(decision.passed_light):
            clear_reason = "passed_light"
        elif bool(decision.post_light_ignore_active):
            clear_reason = "post_light_ignore"
        elif locked_distance_f is not None and locked_distance_f < -1.0:
            clear_reason = "visual_lock_distance_below_minus_1m"

        if clear_reason:
            self.locked_visual_stopline = None
        return clear_reason

    def tick(self):
        if not self.lane_plan_payload:
            return

        lane_plan = dict(self.lane_plan_payload)
        ego = self.ego_status()
        route_points = self.route_points()
        detections = list((self.tl_payload or {}).get("detections") or [])

        if ego is not None and route_points:
            distance, lock_meta = self.choose_distance_with_lock(
                ego=ego,
                route_points=route_points,
                detections=detections,
            )
        else:
            distance = self.default_distance_result()
            lock_meta = self.make_lock_meta(active=bool(self.locked_red_light))

        lane_speed = float(lane_plan.get("target_speed_mps", 0.0) or 0.0)
        current_speed = ego.speed_mps if ego is not None else float(lane_plan.get("current_speed_mps", 0.0) or 0.0)
        selected_stop_distance, stopline_fields = self.visual_stopline_fields(
            relevant=True,
            color="red",
            carla_distance_m=distance.stop_point_distance_m,
            visual_decision_allowed=False,
        )
        associated_candidate = None
        association_fields = self.default_visual_association_fields()
        if stopline_fields.get("visual_stopline_candidate_valid"):
            associated_candidate, association_fields = self.visual_stopline_light_association(
                ego=ego,
                route_points=route_points,
                detections=detections,
                visual_distance_m=stopline_fields.get("visual_dist_filtered"),
                carla_distance_m=distance.stop_point_distance_m,
            )
        associated_candidate, association_fields = self.apply_visual_red_association_latch(
            associated_candidate=associated_candidate,
            association_fields=association_fields,
            stopline_fields=stopline_fields,
        )

        association_valid = bool(association_fields.get("visual_stopline_light_association_valid"))
        decision_candidate = associated_candidate if association_valid else distance.candidate
        color_raw, color_filtered, confidence, detected = self.filtered_color(decision_candidate)
        relevant = True if association_valid else bool(distance.tl_is_front_relevant)
        visual_stopline_decision_active = bool(
            association_valid
            and stopline_fields.get("visual_stopline_candidate_valid")
            and color_filtered in {"red", "yellow", "green"}
        )
        selected_stop_distance = (
            stopline_fields.get("visual_dist_filtered")
            if visual_stopline_decision_active
            else distance.stop_point_distance_m
        )
        stopline_fields.update(association_fields)
        stopline_fields["using_visual_stopline"] = bool(visual_stopline_decision_active)
        stopline_fields["distance_source"] = (
            "visual_stopline_latched"
            if visual_stopline_decision_active and association_fields.get("visual_association_latched")
            else "visual_stopline"
            if visual_stopline_decision_active
            else "carla_stop_waypoint_fallback"
        )
        stopline_fields["selected_distance_m"] = (
            round(float(selected_stop_distance), 3) if selected_stop_distance is not None else None
        )
        if visual_stopline_decision_active:
            stopline_fields["stopline_reject_reason"] = ""
        elif stopline_fields.get("visual_stopline_candidate_valid"):
            stopline_fields["stopline_reject_reason"] = (
                association_fields.get("visual_stopline_light_association_reason")
                or "not_associated_with_visual_stopline"
            )

        locked_visual_fields = self.update_locked_visual_stopline(
            front_bumper_route_s=distance.front_bumper_route_s,
            carla_stop_point_distance_m=distance.stop_point_distance_m,
            stop_point_route_distance_m=distance.stop_point_route_distance_m,
            selected_delta_s_m=distance.selected_delta_s_m,
            stopline_fields=stopline_fields,
            association_fields=association_fields,
            color_filtered=color_filtered,
        )

        red_visual_raw_valid = bool(
            color_filtered == "red"
            and association_valid
            and stopline_fields.get("visual_stopline_candidate_valid")
            and stopline_fields.get("visual_dist_filtered") is not None
        )
        red_visual_lock_active = bool(
            color_filtered == "red"
            and locked_visual_fields.get("locked_visual_stopline_active")
            and locked_visual_fields.get("locked_visual_distance_m") is not None
        )
        visual_lock_suspect = bool(locked_visual_fields.get("visual_lock_suspect"))
        red_visual_lock_usable = bool(red_visual_lock_active and not visual_lock_suspect)
        red_visual_stop_threshold_for_lock_m = float(self.get_parameter("red_visual_stop_distance_m").value)
        simple_red_visual_rule = bool(red_visual_lock_usable or red_visual_raw_valid)
        if color_filtered == "green":
            visual_stopline_decision_active = False
            stopline_fields["using_visual_stopline"] = False
            stopline_fields["distance_source"] = "green_release_visual_stopline_ignored"
            selected_stop_distance = distance.stop_point_distance_m
            stopline_fields["selected_distance_m"] = (
                round(float(selected_stop_distance), 3) if selected_stop_distance is not None else None
            )
            locked_visual_fields["distance_source_for_decision"] = "green_release"
            locked_visual_fields["distance_source_after_consistency"] = "green_release"
        elif simple_red_visual_rule:
            if red_visual_lock_usable:
                selected_stop_distance = float(locked_visual_fields["locked_visual_distance_m"])
                decision_source = "visual_stopline_locked_red_rule"
            else:
                selected_stop_distance = self.optional_float(stopline_fields.get("visual_dist_filtered"))
                decision_source = "visual_stopline_simple_red_rule"
            visual_stopline_decision_active = True
            relevant = True
            stopline_fields["using_visual_stopline"] = True
            stopline_fields["distance_source"] = decision_source
            stopline_fields["selected_distance_m"] = round(float(selected_stop_distance), 3)
            locked_visual_fields["simple_red_visual_rule"] = True
            locked_visual_fields["red_visual_decision_distance_m"] = round(float(selected_stop_distance), 3)
            locked_visual_fields["distance_source_for_decision"] = decision_source
            locked_visual_fields["distance_source_before_consistency"] = decision_source
            locked_visual_fields["distance_source_after_consistency"] = decision_source
            locked_visual_fields["carla_stop_ignored_because_visual_rule"] = (
                distance.stop_point_distance_m is not None
            )
            locked_visual_fields["locked_visual_ignored_because_visual_rule"] = bool(visual_lock_suspect)
            locked_visual_fields["carla_aligned_lock_ignored_because_visual_rule"] = False
            locked_visual_fields["visual_lock_active"] = bool(red_visual_lock_active)
            locked_visual_fields["visual_lock_route_s"] = locked_visual_fields.get(
                "locked_visual_stopline_route_s"
            )
            locked_visual_fields["visual_lock_distance_m"] = (
                locked_visual_fields.get("locked_visual_distance_m") if red_visual_lock_active else None
            )
            locked_visual_fields["visual_lock_used_for_commit"] = bool(
                red_visual_lock_usable
                and selected_stop_distance is not None
                and float(selected_stop_distance) <= red_visual_stop_threshold_for_lock_m
            )
            locked_visual_fields["raw_visual_used_instead_of_bad_lock"] = bool(
                visual_lock_suspect and red_visual_raw_valid
            )
        elif (
            locked_visual_fields.get("locked_visual_stopline_active")
            and color_filtered == "yellow"
            and locked_visual_fields.get("locked_visual_distance_m") is not None
            and locked_visual_fields.get("locked_visual_consistency_valid")
        ):
            selected_stop_distance = float(locked_visual_fields["locked_visual_distance_m"])
            visual_stopline_decision_active = True
            relevant = True
            stopline_fields["using_visual_stopline"] = True
            stopline_fields["distance_source"] = locked_visual_fields.get(
                "distance_source_after_consistency",
                "locked_visual_stopline",
            )
            stopline_fields["selected_distance_m"] = round(float(selected_stop_distance), 3)
        elif (
            locked_visual_fields.get("locked_visual_stopline_active")
            and color_filtered == "yellow"
            and locked_visual_fields.get("locked_visual_consistency_valid") is False
        ):
            fallback_source = locked_visual_fields.get("distance_source_after_consistency")
            raw_fallback_distance = self.optional_float(stopline_fields.get("visual_dist_filtered"))
            if fallback_source == "raw_visual_stopline_consistency_fallback" and raw_fallback_distance is not None:
                selected_stop_distance = raw_fallback_distance
                visual_stopline_decision_active = True
                relevant = True
                stopline_fields["using_visual_stopline"] = True
                stopline_fields["distance_source"] = fallback_source
                stopline_fields["selected_distance_m"] = round(float(selected_stop_distance), 3)
            elif distance.stop_point_distance_m is not None:
                selected_stop_distance = distance.stop_point_distance_m
                visual_stopline_decision_active = False
                relevant = True
                stopline_fields["using_visual_stopline"] = False
                stopline_fields["distance_source"] = "carla_stop_waypoint_fallback"
                stopline_fields["selected_distance_m"] = round(float(selected_stop_distance), 3)
                locked_visual_fields["distance_source_for_decision"] = "carla_stop_waypoint_fallback"
                locked_visual_fields["distance_source_after_consistency"] = "carla_stop_waypoint_fallback"
        elif visual_stopline_decision_active:
            locked_visual_fields["distance_source_for_decision"] = (
                "raw_visual_stopline"
                if not association_fields.get("visual_association_latched")
                else "raw_visual_stopline_latched"
            )
            locked_visual_fields["distance_source_before_consistency"] = locked_visual_fields[
                "distance_source_for_decision"
            ]
            locked_visual_fields["distance_source_after_consistency"] = locked_visual_fields[
                "distance_source_for_decision"
            ]

        debug_fields = self.debug_draw_fields(
            route_points=route_points,
            distance=distance,
            stopline_fields=stopline_fields,
            locked_visual_fields=locked_visual_fields,
        )
        stop_point_correction_fields = self.default_stop_point_correction_fields()
        original_carla_stop_distance = self.optional_float(distance.stop_point_distance_m)
        original_selected_stop_distance = self.optional_float(selected_stop_distance)
        visual_distance_for_correction = self.optional_float(stopline_fields.get("visual_dist_filtered"))
        stop_point_correction_fields["original_carla_stop_distance"] = (
            round(float(original_carla_stop_distance), 3)
            if original_carla_stop_distance is not None
            else None
        )
        stop_point_correction_fields["original_selected_stop_distance"] = (
            round(float(original_selected_stop_distance), 3)
            if original_selected_stop_distance is not None
            else None
        )
        stop_point_correction_fields["visual_stopline_distance_for_correction"] = (
            round(float(visual_distance_for_correction), 3)
            if visual_distance_for_correction is not None
            else None
        )

        corrected_stop_point = debug_fields.get("raw_visual_stopline_estimated_world")
        front_route_s_for_correction = self.optional_float(distance.front_bumper_route_s)
        corrected_stop_point_route_s = (
            front_route_s_for_correction + visual_distance_for_correction
            if front_route_s_for_correction is not None and visual_distance_for_correction is not None
            else None
        )
        associated_light_id = association_fields.get("associated_light_id")
        selected_light_id = distance.selected_light_id
        visual_more_reliable_than_selected = bool(
            association_valid
            and associated_light_id is not None
            and bool(stopline_fields.get("visual_stopline_candidate_valid"))
            and bool(stopline_fields.get("visual_stopline_detected"))
        )
        selected_associated_light_mismatch = bool(
            visual_more_reliable_than_selected
            and selected_light_id is not None
            and str(selected_light_id) != str(associated_light_id)
        )
        associated_light_color_for_correction = str(
            association_fields.get("associated_light_color") or ""
        ).lower()
        corrected_point_has_valid_red_association = bool(
            associated_light_color_for_correction == "red"
            or association_valid
            or bool(association_fields.get("visual_association_latched"))
            or bool(association_fields.get("associated_light_above_stopline"))
        )
        correction_reasons = []
        if original_carla_stop_distance is not None and original_carla_stop_distance < 0.0:
            correction_reasons.append("carla_stop_point_behind_vehicle")
        if original_selected_stop_distance is not None and original_selected_stop_distance < 0.0:
            correction_reasons.append("selected_stop_point_behind_vehicle")
        if bool(debug_fields.get("selected_stop_point_suspect")):
            correction_reasons.append("selected_stop_point_suspect")
        if bool(association_fields.get("carla_candidate_suspect")):
            correction_reasons.append("carla_candidate_suspect")
        if (
            original_carla_stop_distance is not None
            and visual_distance_for_correction is not None
            and abs(original_carla_stop_distance - visual_distance_for_correction) > 2.5
        ):
            correction_reasons.append("carla_visual_distance_mismatch")
        if selected_associated_light_mismatch:
            correction_reasons.append("selected_associated_light_mismatch_visual_reliable")

        associated_distance_for_correction = self.optional_float(
            association_fields.get("associated_light_distance")
        )
        visual_correction_reject_assoc_far = bool(
            associated_light_color_for_correction == "red"
            and association_valid
            and visual_distance_for_correction is not None
            and associated_distance_for_correction is not None
            and associated_distance_for_correction > 12.0
            and visual_distance_for_correction < 6.0
        )
        if visual_correction_reject_assoc_far:
            stop_point_correction_fields.update({
                "visual_correction_rejected_associated_light_too_far": True,
                "visual_correction_reject_assoc_distance": round(
                    float(associated_distance_for_correction), 3
                ),
                "visual_correction_reject_visual_distance": round(
                    float(visual_distance_for_correction), 3
                ),
            })

        stop_point_correction_allowed = bool(
            color_filtered == "red"
            and bool(stopline_fields.get("visual_stopline_detected"))
            and bool(stopline_fields.get("visual_stopline_candidate_valid"))
            and visual_distance_for_correction is not None
            and 0.0 < visual_distance_for_correction <= 10.0
            and corrected_stop_point is not None
            and correction_reasons
            and not visual_correction_reject_assoc_far
        )
        if stop_point_correction_allowed:
            selected_stop_distance = visual_distance_for_correction
            visual_stopline_decision_active = bool(corrected_point_has_valid_red_association)
            relevant = True
            correction_reason = ",".join(correction_reasons)
            correction_decision_source = (
                "visual_stopline_corrected_point"
                if corrected_point_has_valid_red_association
                else "visual_corrected_ignored_no_association"
            )
            stopline_fields["using_visual_stopline"] = True
            stopline_fields["distance_source"] = correction_decision_source
            stopline_fields["selected_distance_m"] = round(float(selected_stop_distance), 3)
            locked_visual_fields["distance_source_for_decision"] = correction_decision_source
            locked_visual_fields["distance_source_before_consistency"] = correction_decision_source
            locked_visual_fields["distance_source_after_consistency"] = correction_decision_source
            locked_visual_fields["red_visual_decision_distance_m"] = round(float(selected_stop_distance), 3)
            locked_visual_fields["carla_stop_ignored_because_visual_rule"] = True
            stop_point_correction_fields.update({
                "stop_point_corrected_from_bad_carla": True,
                "stop_point_correction_reason": correction_reason,
                "corrected_stop_point_source": "visual_stopline_corrected",
                "corrected_stop_point_world_x": corrected_stop_point.get("x"),
                "corrected_stop_point_world_y": corrected_stop_point.get("y"),
                "corrected_stop_point_world_z": corrected_stop_point.get("z"),
                "corrected_point_ignored_no_valid_association": (
                    not corrected_point_has_valid_red_association
                ),
                "corrected_point_ignored_reason": (
                    "associated_light_unknown_no_latched_visual_association"
                    if not corrected_point_has_valid_red_association
                    else ""
                ),
                "corrected_point_would_have_distance": round(float(selected_stop_distance), 3),
                "correction_applied_to_decision": bool(corrected_point_has_valid_red_association),
            })
            debug_fields["selected_stop_point_world"] = corrected_stop_point
            debug_fields["final_stop_point_world"] = corrected_stop_point
            if self.locked_red_light and corrected_point_has_valid_red_association:
                self.acquire_visual_corrected_red_lock(
                    distance=distance,
                    association_fields=association_fields,
                    corrected_point=corrected_stop_point,
                    corrected_route_s=corrected_stop_point_route_s,
                    color_filtered=color_filtered,
                )
                lock_meta.update(self.lock_fields(lock_meta, ""))
                lock_meta["lock_match_source"] = "visual_stopline_corrected"

        red_visual_slow_distance_m = float(self.get_parameter("red_visual_slow_distance_m").value)
        visual_red_commit_distance_m = float(self.get_parameter("red_visual_stop_distance_m").value)
        red_visual_hard_stop_distance_m = float(self.get_parameter("red_visual_hard_stop_distance_m").value)
        red_visual_approach_speed_mps = float(self.get_parameter("red_visual_approach_speed_mps").value)
        front_dist_for_commit = self.optional_float(distance.stop_point_distance_m)
        locked_dist_for_commit = self.optional_float(locked_visual_fields.get("locked_visual_distance_m"))
        raw_visual_dist_for_commit = self.optional_float(stopline_fields.get("visual_dist_filtered"))
        current_decision_distance = self.optional_float(selected_stop_distance)
        front_commit_ready = bool(
            front_dist_for_commit is not None
            and 0.0 < front_dist_for_commit <= visual_red_commit_distance_m
        )
        locked_commit_ready = bool(
            locked_dist_for_commit is not None
            and 0.0 < locked_dist_for_commit <= visual_red_commit_distance_m
        )
        selected_distance_would_not_commit = bool(
            current_decision_distance is None
            or current_decision_distance > visual_red_commit_distance_m
        )
        raw_visual_source_active = str(stopline_fields.get("distance_source") or "").startswith(
            "raw_visual_stopline"
        )
        commit_blocked_by_raw_visual = bool(
            color_filtered == "red"
            and raw_visual_source_active
            and selected_distance_would_not_commit
            and (front_commit_ready or locked_commit_ready)
        )
        commit_overrode_raw_visual_mismatch = False
        commit_distance_source = str(stopline_fields.get("distance_source") or "unknown")
        selected_stop_point_suspect = bool(debug_fields.get("selected_stop_point_suspect"))
        carla_commit_allowed = bool(
            color_filtered == "red"
            and not simple_red_visual_rule
            and str(association_fields.get("associated_light_color") or color_filtered).lower() == "red"
            and not bool(association_fields.get("carla_candidate_suspect"))
            and not selected_stop_point_suspect
            and bool(distance.distance_valid)
            and bool(relevant)
            and distance.selected_light_id is not None
            and (front_commit_ready or locked_commit_ready)
        )
        if carla_commit_allowed and selected_distance_would_not_commit:
            if front_commit_ready:
                selected_stop_distance = front_dist_for_commit
                commit_distance_source = "carla_stop_waypoint_commit"
            else:
                selected_stop_distance = locked_dist_for_commit
                commit_distance_source = "carla_aligned_locked_visual"
            visual_stopline_decision_active = True
            relevant = True
            stopline_fields["using_visual_stopline"] = True
            stopline_fields["distance_source"] = commit_distance_source
            stopline_fields["selected_distance_m"] = round(float(selected_stop_distance), 3)
            locked_visual_fields["distance_source_for_decision"] = commit_distance_source
            locked_visual_fields["distance_source_after_consistency"] = commit_distance_source
            commit_overrode_raw_visual_mismatch = bool(commit_blocked_by_raw_visual)

        debug_fields.update({
            "front_dist_for_commit": (
                round(float(front_dist_for_commit), 3) if front_dist_for_commit is not None else None
            ),
            "locked_dist_for_commit": (
                round(float(locked_dist_for_commit), 3) if locked_dist_for_commit is not None else None
            ),
            "raw_visual_dist_for_commit": (
                round(float(raw_visual_dist_for_commit), 3) if raw_visual_dist_for_commit is not None else None
            ),
            "commit_distance_source": commit_distance_source,
            "commit_blocked_by_raw_visual": commit_blocked_by_raw_visual,
            "commit_overrode_raw_visual_mismatch": commit_overrode_raw_visual_mismatch,
        })

        resolver_approach_elapsed_s = (
            max(0.0, time.time() - float(self.red_visual_approach_started_s))
            if self.red_visual_approach_started_s is not None
            else 0.0
        )
        effective_stop_fields = self.resolve_red_stop_context(
            color_filtered=color_filtered,
            association_fields=association_fields,
            stopline_fields=stopline_fields,
            locked_visual_fields=locked_visual_fields,
            debug_fields=debug_fields,
            stop_point_correction_fields=stop_point_correction_fields,
            front_dist_for_commit=front_dist_for_commit,
            selected_stop_point_suspect=selected_stop_point_suspect,
            red_visual_approach_elapsed_s=resolver_approach_elapsed_s,
            distance=distance,
        )
        effective_stop_distance = self.optional_float(effective_stop_fields.get("effective_stop_distance_m"))
        if color_filtered == "red":
            if bool(effective_stop_fields.get("effective_stop_valid")):
                selected_stop_distance = effective_stop_distance
                relevant = True
                visual_stopline_decision_active = bool(
                    effective_stop_fields.get("effective_stop_source")
                    in {"visual", "visual_corrected", "visual_last_good_hold", "visual_stuck_overrun"}
                )
                stopline_fields["using_visual_stopline"] = bool(visual_stopline_decision_active)
                stopline_fields["selected_distance_m"] = (
                    round(float(selected_stop_distance), 3)
                    if selected_stop_distance is not None
                    else None
                )
                stopline_fields["distance_source"] = str(effective_stop_fields.get("effective_stop_source") or "")
                locked_visual_fields["distance_source_for_decision"] = str(
                    effective_stop_fields.get("effective_stop_source") or ""
                )
                locked_visual_fields["distance_source_after_consistency"] = str(
                    effective_stop_fields.get("effective_stop_source") or ""
                )
                locked_visual_fields["red_visual_decision_distance_m"] = (
                    round(float(selected_stop_distance), 3)
                    if selected_stop_distance is not None
                    else None
                )
            else:
                selected_stop_distance = None
                relevant = False
                visual_stopline_decision_active = False

        decision = self.state_machine.update(
            relevant=relevant,
            color=color_filtered,
            stop_point_distance_m=selected_stop_distance,
            current_speed_mps=current_speed,
            cruise_speed_mps=lane_speed,
            distance_valid=bool(
                selected_stop_distance is not None
                and (distance.distance_valid or visual_stopline_decision_active)
            ),
            visual_stopline_active=bool(visual_stopline_decision_active),
            visual_stopline_distance_m=selected_stop_distance
            if visual_stopline_decision_active
            else None,
            visual_red_brake_distance_m=red_visual_slow_distance_m,
            visual_red_commit_distance_m=visual_red_commit_distance_m,
            visual_red_hard_commit_distance_m=red_visual_hard_stop_distance_m,
            visual_red_approach_target_mps=red_visual_approach_speed_mps,
        )
        if commit_overrode_raw_visual_mismatch and decision.state in {RED_STOP_COMMIT, STOPPED_AT_RED}:
            decision.reason = "red_stop_at_carla_aligned_visual_stopline"
            decision.stop_commit_reason = "red_stop_at_carla_aligned_visual_stopline"

        if stop_point_correction_fields["visual_correction_rejected_associated_light_too_far"]:
            reject_reason = "visual_correction_rejected_associated_light_too_far"
            self.state_machine.state = NO_RELEVANT_LIGHT
            self.state_machine.reset_speed(lane_speed)
            decision.state = NO_RELEVANT_LIGHT
            decision.desired_speed_mps = lane_speed
            decision.desired_speed_raw_mps = lane_speed
            decision.stop_request = False
            decision.reason = reject_reason
            decision.red_stop_commit_active = False
            decision.stop_commit_reason = ""
            decision.visual_red_brake = False
            decision.visual_red_commit = False
            decision.visual_red_hard_commit = False
            decision.visual_red_approach = False
            decision.visual_red_approach_target_mps = 0.0
        elif stop_point_correction_fields["corrected_point_ignored_no_valid_association"]:
            ignored_reason = "red_visual_corrected_point_ignored_no_valid_association"
            self.state_machine.state = NO_RELEVANT_LIGHT
            self.state_machine.reset_speed(lane_speed)
            decision.state = NO_RELEVANT_LIGHT
            decision.desired_speed_mps = lane_speed
            decision.desired_speed_raw_mps = lane_speed
            decision.stop_request = False
            decision.reason = ignored_reason
            decision.red_stop_commit_active = False
            decision.stop_commit_reason = ""
            decision.visual_red_brake = False
            decision.visual_red_commit = False
            decision.visual_red_hard_commit = False
            decision.visual_red_approach = False
            decision.visual_red_approach_target_mps = 0.0
            stop_point_correction_fields["tl_speed_override_cleared_for_no_association"] = True
        elif stop_point_correction_fields["stop_point_corrected_from_bad_carla"]:
            corrected_decision_distance = self.optional_float(selected_stop_distance)
            if corrected_decision_distance is not None and corrected_decision_distance <= 2.0:
                correction_reason = "red_visual_corrected_stop_point_2m_full_brake"
                decision.state = RED_STOP_COMMIT
                decision.desired_speed_mps = 0.0
                decision.desired_speed_raw_mps = 0.0
                decision.stop_request = True
                decision.reason = correction_reason
                decision.red_stop_commit_active = True
                decision.stop_commit_reason = correction_reason
                decision.visual_red_brake = True
                decision.visual_red_commit = True
                decision.visual_red_hard_commit = bool(
                    corrected_decision_distance <= red_visual_hard_stop_distance_m
                )
                decision.visual_red_approach = False
            elif corrected_decision_distance is not None and corrected_decision_distance <= 10.0:
                correction_reason = "red_visual_corrected_stop_point_slow"
                corrected_target_speed = max(0.0, red_visual_approach_speed_mps)
                decision.state = RED_DECEL
                decision.desired_speed_mps = corrected_target_speed
                decision.desired_speed_raw_mps = corrected_target_speed
                decision.stop_request = False
                decision.reason = correction_reason
                decision.red_stop_commit_active = False
                decision.stop_commit_reason = ""
                decision.visual_red_brake = True
                decision.visual_red_commit = False
                decision.visual_red_hard_commit = False
                decision.visual_red_approach = True
                decision.visual_red_approach_target_mps = corrected_target_speed

        failsafe_fields = self.default_red_visual_failsafe_fields()
        associated_light_distance_f = self.optional_float(association_fields.get("associated_light_distance"))
        failsafe_fields["associated_light_distance_for_failsafe"] = (
            round(float(associated_light_distance_f), 3)
            if associated_light_distance_f is not None
            else None
        )
        decision_distance_f = self.optional_float(locked_visual_fields.get("red_visual_decision_distance_m"))
        if decision_distance_f is None:
            decision_distance_f = self.optional_float(selected_stop_distance)

        red_stop_2m_any_valid_distance_reason = "red_stop_2m_any_valid_distance_full_brake"
        associated_light_color = str(association_fields.get("associated_light_color") or "").lower()
        carla_candidate_suspect = bool(association_fields.get("carla_candidate_suspect"))
        stop_point_corrected_from_bad_carla = bool(
            stop_point_correction_fields["stop_point_corrected_from_bad_carla"]
        )
        red_stop_2m_base_allowed = bool(
            associated_light_color == "red"
            and association_valid
            and simple_red_visual_rule
            and decision.state == RED_DECEL
        )
        red_stop_2m_source = ""
        red_stop_2m_value = None
        red_stop_2m_reject_reasons = []
        stop_point_uses_visual_primary = bool(effective_stop_fields.get("visual_distance_primary"))
        corrected_visual_distance_for_fallback = (
            effective_stop_distance
            if stop_point_uses_visual_primary
            else self.optional_float(stopline_fields.get("visual_dist_filtered"))
            if stop_point_corrected_from_bad_carla
            else decision_distance_f
        )
        if corrected_visual_distance_for_fallback is None:
            corrected_visual_distance_for_fallback = self.optional_float(selected_stop_distance)

        def distance_agrees_with_corrected_visual(source_distance) -> bool:
            source_distance_f = self.optional_float(source_distance)
            agreement_required = bool(
                stop_point_corrected_from_bad_carla or stop_point_uses_visual_primary
            )
            return bool(
                not agreement_required
                or (
                    corrected_visual_distance_for_fallback is not None
                    and source_distance_f is not None
                    and abs(corrected_visual_distance_for_fallback - source_distance_f) <= 1.5
                )
            )

        if red_stop_2m_base_allowed:
            if decision_distance_f is not None:
                if 0.0 <= decision_distance_f <= 2.0:
                    red_stop_2m_source = (
                        "corrected_visual"
                        if stop_point_corrected_from_bad_carla
                        else "visual"
                    )
                    red_stop_2m_value = decision_distance_f
                elif decision_distance_f < 0.0:
                    red_stop_2m_reject_reasons.append("negative_visual_distance_ignored")

            if red_stop_2m_source == "" and associated_light_distance_f is not None:
                if 0.0 <= associated_light_distance_f <= 2.0:
                    if distance_agrees_with_corrected_visual(associated_light_distance_f):
                        red_stop_2m_source = "associated_light"
                        red_stop_2m_value = associated_light_distance_f
                    else:
                        red_stop_2m_reject_reasons.append(
                            "associated_distance_conflicts_with_corrected_visual"
                        )
                elif associated_light_distance_f < 0.0:
                    red_stop_2m_reject_reasons.append("negative_associated_light_distance_ignored")

            if red_stop_2m_source == "" and front_dist_for_commit is not None:
                if selected_stop_point_suspect or carla_candidate_suspect:
                    red_stop_2m_reject_reasons.append("front_distance_suspect_ignored")
                elif 0.0 <= front_dist_for_commit <= 2.0:
                    if distance_agrees_with_corrected_visual(front_dist_for_commit):
                        red_stop_2m_source = "front"
                        red_stop_2m_value = front_dist_for_commit
                    else:
                        red_stop_2m_reject_reasons.append(
                            "front_distance_conflicts_with_corrected_visual"
                        )
                elif front_dist_for_commit < 0.0:
                    red_stop_2m_reject_reasons.append("negative_front_distance_ignored")

        red_corrected_point_overrun_source = ""
        red_corrected_point_overrun_allowed = bool(
            associated_light_color == "red"
            and association_valid
            and simple_red_visual_rule
            and stop_point_corrected_from_bad_carla
            and decision.state in {RED_DECEL, RED_STOP_COMMIT}
        )
        if (
            red_corrected_point_overrun_allowed
            and associated_light_distance_f is not None
            and associated_light_distance_f < 0.0
        ):
            red_corrected_point_overrun_source = "associated_light"
        elif (
            red_corrected_point_overrun_allowed
            and front_dist_for_commit is not None
            and front_dist_for_commit < 0.0
            and not selected_stop_point_suspect
            and not carla_candidate_suspect
        ):
            red_corrected_point_overrun_source = "front"

        if red_corrected_point_overrun_source:
            failsafe_fields["red_corrected_point_overrun_full_brake"] = True
            failsafe_fields["red_corrected_point_overrun_source"] = red_corrected_point_overrun_source
            decision.state = RED_STOP_COMMIT
            decision.desired_speed_mps = 0.0
            decision.desired_speed_raw_mps = 0.0
            decision.stop_request = True
            decision.reason = "red_corrected_point_overrun_full_brake"
            decision.red_stop_commit_active = True
            decision.stop_commit_reason = "red_corrected_point_overrun_full_brake"
        elif red_stop_2m_source:
            failsafe_fields["red_stop_2m_any_valid_distance"] = True
            failsafe_fields["red_stop_2m_any_valid_distance_source"] = red_stop_2m_source
            failsafe_fields["red_stop_2m_any_valid_distance_value"] = round(float(red_stop_2m_value), 3)
            failsafe_fields["red_stop_2m_any_valid_distance_reason"] = red_stop_2m_any_valid_distance_reason
            decision.state = RED_STOP_COMMIT
            decision.desired_speed_mps = 0.0
            decision.desired_speed_raw_mps = 0.0
            decision.stop_request = True
            decision.reason = red_stop_2m_any_valid_distance_reason
            decision.red_stop_commit_active = True
            decision.stop_commit_reason = red_stop_2m_any_valid_distance_reason
        elif red_stop_2m_reject_reasons:
            failsafe_fields["red_stop_2m_any_valid_distance_reason"] = ",".join(red_stop_2m_reject_reasons)

        now_s = time.time()
        in_red_visual_decel = bool(
            color_filtered == "red"
            and association_valid
            and simple_red_visual_rule
            and visual_stopline_decision_active
            and decision.state == RED_DECEL
            and decision.reason in {
                "red_visual_slow_10m_rule",
                "red_visual_corrected_stop_point_slow",
                "red_effective_stop_distance_slow",
            }
            and decision_distance_f is not None
        )
        if in_red_visual_decel:
            if self.red_visual_approach_started_s is None:
                self.red_visual_approach_started_s = now_s
            approach_elapsed_s = max(0.0, now_s - float(self.red_visual_approach_started_s))
        else:
            self.red_visual_approach_started_s = None
            approach_elapsed_s = 0.0

        failsafe_fields["red_visual_approach_elapsed_s"] = round(float(approach_elapsed_s), 3)
        failsafe_min_m = float(self.get_parameter("red_visual_near_light_failsafe_min_m").value)
        failsafe_max_m = float(self.get_parameter("red_visual_near_light_failsafe_max_m").value)
        failsafe_speed_mps = float(self.get_parameter("red_visual_near_light_failsafe_speed_mps").value)
        failsafe_light_distance_m = float(
            self.get_parameter("red_visual_near_light_failsafe_light_distance_m").value
        )
        stuck_stop_min_s = float(self.get_parameter("red_visual_stuck_stop_min_s").value)
        stuck_stop_min_speed_mps = float(self.get_parameter("red_visual_stuck_stop_min_speed_mps").value)
        visual_distance_in_stuck_band = bool(
            decision_distance_f is not None
            and failsafe_min_m <= decision_distance_f <= failsafe_max_m
        )
        near_associated_light = bool(
            associated_light_distance_f is not None
            and associated_light_distance_f <= failsafe_light_distance_m
        )
        associated_above_stopline = bool(association_fields.get("associated_light_above_stopline"))
        low_speed_near_light = bool(
            in_red_visual_decel
            and visual_distance_in_stuck_band
            and current_speed <= failsafe_speed_mps
            and (near_associated_light or associated_above_stopline)
        )
        stuck_elapsed_stop = bool(
            in_red_visual_decel
            and visual_distance_in_stuck_band
            and approach_elapsed_s >= stuck_stop_min_s
            and stuck_stop_min_speed_mps <= current_speed <= failsafe_speed_mps
        )
        failsafe_fields["red_visual_distance_stuck"] = bool(
            in_red_visual_decel and visual_distance_in_stuck_band and approach_elapsed_s >= stuck_stop_min_s
        )
        failsafe_fields["red_visual_stuck_stop"] = bool(stuck_elapsed_stop)
        if low_speed_near_light or stuck_elapsed_stop:
            failsafe_reason = (
                "near_associated_light"
                if low_speed_near_light and near_associated_light
                else "associated_light_above_stopline"
                if low_speed_near_light and associated_above_stopline
                else "visual_distance_stuck_timeout"
            )
            failsafe_fields["red_visual_near_light_failsafe"] = True
            failsafe_fields["red_visual_near_light_failsafe_reason"] = failsafe_reason
            decision.state = RED_STOP_COMMIT
            decision.desired_speed_mps = 0.0
            decision.desired_speed_raw_mps = 0.0
            decision.stop_request = True
            decision.reason = "red_visual_near_light_failsafe_stop"
            decision.red_stop_commit_active = True
            decision.stop_commit_reason = "red_visual_near_light_failsafe_stop"

        visual_lock_cleared_reason = self.clear_locked_visual_stopline_if_needed(
            color_filtered=color_filtered,
            decision=decision,
            locked_fields=locked_visual_fields,
        )
        if visual_lock_cleared_reason:
            locked_visual_fields["visual_lock_cleared_reason"] = visual_lock_cleared_reason
            locked_visual_fields["visual_lock_active"] = False

        lock_release_reason = ""
        locked_color = str(lock_meta.get("locked_light_color") or color_filtered or "").lower()
        locked_red_or_yellow = locked_color in {"red", "yellow"}
        locked_overrun = (
            bool(self.locked_red_light)
            and bool(lock_meta.get("locked_light_active"))
            and color_filtered != "green"
            and decision.state != GREEN_RELEASE
            and not visual_lock_suspect
            and locked_red_or_yellow
            and distance.stop_point_distance_m is not None
            and float(selected_stop_distance if selected_stop_distance is not None else distance.stop_point_distance_m) <= 0.0
        )

        if locked_overrun:
            decision.state = RED_LOCK_OVERRUN_HOLD
            decision.desired_speed_mps = 0.0
            decision.desired_speed_raw_mps = 0.0
            decision.stop_request = True
            decision.reason = "locked_red_stop_point_overrun"
            decision.red_stop_commit_active = True
            decision.stop_commit_reason = "locked_red_stop_point_overrun"
            lock_meta["candidate_switch_blocked"] = True
            lock_meta["candidate_switch_block_reason"] = "locked_red_stop_point_overrun"
        elif self.locked_red_light and decision.state in {PASSED_LIGHT, POST_LIGHT_IGNORE}:
            lock_release_reason = "passed_light"
        elif self.locked_red_light and decision.state == GREEN_RELEASE and lock_meta.get("locked_light_active"):
            lock_release_reason = "locked_light_green"
            lock_meta["green_release_source"] = "same_locked_light"
        elif decision.state == GREEN_RELEASE:
            lock_meta["green_release_source"] = "new_candidate"

        red_stop_hold_start_allowed = False
        if color_filtered == "red":
            failsafe_fields["red_stop_2m_any_valid_distance"] = False
            failsafe_fields["red_stop_2m_any_valid_distance_source"] = ""
            failsafe_fields["red_stop_2m_any_valid_distance_value"] = None
            failsafe_fields["red_corrected_point_overrun_full_brake"] = False
            failsafe_fields["red_corrected_point_overrun_source"] = ""

            effective_valid = bool(effective_stop_fields.get("effective_stop_valid"))
            effective_source = str(effective_stop_fields.get("effective_stop_source") or "")
            effective_reason = str(effective_stop_fields.get("effective_stop_reason") or "no_valid_stop_distance")
            effective_distance = self.optional_float(effective_stop_fields.get("effective_stop_distance_m"))
            if not effective_valid or effective_distance is None:
                self.state_machine.state = NO_RELEVANT_LIGHT
                self.state_machine.reset_speed(lane_speed)
                decision.state = NO_RELEVANT_LIGHT
                decision.desired_speed_mps = lane_speed
                decision.desired_speed_raw_mps = lane_speed
                decision.stop_request = False
                decision.reason = effective_reason
                decision.red_stop_commit_active = False
                decision.stop_commit_reason = ""
                decision.visual_red_brake = False
                decision.visual_red_commit = False
                decision.visual_red_hard_commit = False
                decision.visual_red_approach = False
                decision.visual_red_approach_target_mps = 0.0
                failsafe_fields["red_visual_near_light_failsafe"] = False
                failsafe_fields["red_visual_near_light_failsafe_reason"] = ""
            elif effective_distance <= visual_red_commit_distance_m:
                commit_reason = (
                    "red_visual_stuck_overrun_full_brake"
                    if effective_source == "visual_stuck_overrun"
                    else
                    "red_visual_corrected_stop_point_2m_full_brake"
                    if effective_source == "visual_corrected"
                    else "red_effective_stop_distance_2m_full_brake"
                )
                decision.state = RED_STOP_COMMIT
                decision.desired_speed_mps = 0.0
                decision.desired_speed_raw_mps = 0.0
                decision.stop_request = True
                decision.reason = commit_reason
                decision.red_stop_commit_active = True
                decision.stop_commit_reason = commit_reason
                decision.visual_red_brake = True
                decision.visual_red_commit = True
                decision.visual_red_hard_commit = bool(effective_distance <= red_visual_hard_stop_distance_m)
                decision.visual_red_approach = False
                decision.visual_red_approach_target_mps = 0.0
                recent_visual_hold_active = bool(
                    effective_stop_fields.get("visual_last_good_hold_active")
                )
                primary_commit_sources = {
                    "visual",
                    "visual_corrected",
                    "visual_last_good_hold",
                    "visual_stuck_overrun",
                }
                fallback_commit_sources = {"locked_visual", "front", "associated_light", "carla"}
                red_stop_hold_start_allowed = bool(
                    effective_source in primary_commit_sources
                    or (
                        effective_source in fallback_commit_sources
                        and not recent_visual_hold_active
                    )
                )
            elif effective_distance <= red_visual_slow_distance_m:
                slow_reason = (
                    "red_visual_corrected_stop_point_slow"
                    if effective_source == "visual_corrected"
                    else "red_effective_stop_distance_slow"
                )
                target = max(0.0, red_visual_approach_speed_mps)
                decision.state = RED_DECEL
                decision.desired_speed_mps = target
                decision.desired_speed_raw_mps = target
                decision.stop_request = False
                decision.reason = slow_reason
                decision.red_stop_commit_active = False
                decision.stop_commit_reason = ""
                decision.visual_red_brake = True
                decision.visual_red_commit = False
                decision.visual_red_hard_commit = False
                decision.visual_red_approach = True
                decision.visual_red_approach_target_mps = target
                failsafe_fields["red_visual_near_light_failsafe"] = False
                failsafe_fields["red_visual_near_light_failsafe_reason"] = ""
            else:
                self.state_machine.reset_speed(lane_speed)
                decision.state = NO_RELEVANT_LIGHT
                decision.desired_speed_mps = lane_speed
                decision.desired_speed_raw_mps = lane_speed
                decision.stop_request = False
                decision.reason = "red_effective_stop_distance_far"
                decision.red_stop_commit_active = False
                decision.stop_commit_reason = ""
                decision.visual_red_brake = False
                decision.visual_red_commit = False
                decision.visual_red_hard_commit = False
                decision.visual_red_approach = False
                decision.visual_red_approach_target_mps = 0.0
                failsafe_fields["red_visual_near_light_failsafe"] = False
                failsafe_fields["red_visual_near_light_failsafe_reason"] = ""

        red_stop_hold_cleared_by_green = False
        red_stop_hold_green_release = bool(
            decision.state == GREEN_RELEASE or associated_light_color == "green"
        )
        if red_stop_hold_green_release and self.red_stop_hold_active:
            self.red_stop_hold_active = False
            self.red_stop_hold_reason = ""
            red_stop_hold_cleared_by_green = True

        red_stop_hold_red_context = bool(
            not red_stop_hold_green_release
            and (
                color_filtered == "red"
                or associated_light_color == "red"
                or locked_color == "red"
                or bool(self.locked_red_light)
            )
        )
        if (
            decision.state in {RED_STOP_COMMIT, STOPPED_AT_RED}
            and red_stop_hold_red_context
            and red_stop_hold_start_allowed
        ):
            self.red_stop_hold_active = True
            self.red_stop_hold_reason = "red_stop_hold_until_green"

        if self.red_stop_hold_active and red_stop_hold_red_context:
            stopped_for_hold = current_speed <= self.state_machine.full_stop_speed_mps
            decision.state = STOPPED_AT_RED if stopped_for_hold else RED_STOP_COMMIT
            decision.desired_speed_mps = 0.0
            decision.desired_speed_raw_mps = 0.0
            decision.stop_request = True
            decision.reason = "red_stop_hold_until_green"
            decision.red_stop_commit_active = True
            decision.stop_commit_reason = "red_stop_hold_until_green"
            decision.visual_red_approach = False
            decision.visual_red_approach_target_mps = 0.0

        if (
            decision.state in {RED_STOP_COMMIT, STOPPED_AT_RED, RED_STOP_CREEP}
            and color_filtered == "red"
            and relevant
        ):
            if stop_point_correction_fields["stop_point_corrected_from_bad_carla"]:
                self.acquire_visual_corrected_red_lock(
                    distance=distance,
                    association_fields=association_fields,
                    corrected_point=debug_fields.get("final_stop_point_world"),
                    corrected_route_s=corrected_stop_point_route_s,
                    color_filtered=color_filtered,
                )
            else:
                self.acquire_red_lock(distance, color_filtered)
            lock_meta.update(self.lock_fields(lock_meta, ""))
            lock_release_reason = ""

        lock_debug_fields = self.lock_fields(lock_meta, lock_release_reason)
        if lock_release_reason:
            self.locked_red_light = None
        red_stop_hold_debug_fields = self.red_stop_hold_fields(
            cleared_by_green=red_stop_hold_cleared_by_green
        )

        mission_stop = bool(lane_plan.get("stop_request", False))
        final_stop_request = bool(mission_stop or decision.stop_request)
        final_speed = min(lane_speed, decision.desired_speed_mps)
        reason = str(lane_plan.get("reason", "lane_follow"))
        if mission_stop:
            final_reason = reason
            stop_reason = lane_plan.get("stop_reason") or reason
        elif decision.stop_request:
            final_reason = "traffic_light_stop"
            stop_reason = decision.reason
        elif color_filtered == "red" and decision.reason in {
            "no_valid_red_association",
            "visual_correction_rejected_associated_light_too_far",
            "no_valid_stop_distance",
            "red_effective_stop_distance_far",
        }:
            final_reason = decision.reason
            stop_reason = ""
        elif stop_point_correction_fields["visual_correction_rejected_associated_light_too_far"]:
            final_reason = decision.reason
            stop_reason = ""
        elif stop_point_correction_fields["corrected_point_ignored_no_valid_association"]:
            final_reason = decision.reason
            stop_reason = ""
        elif final_speed < lane_speed - 0.05:
            final_reason = decision.reason
            stop_reason = ""
        else:
            final_reason = reason
            stop_reason = lane_plan.get("stop_reason", "")

        tl_fields = {
            "tl_detected": bool(detected),
            "tl_color_raw": color_raw,
            "tl_color_filtered": color_filtered,
            "tl_confidence": round(float(confidence), 4),
            **distance.to_dict(),
            **stopline_fields,
            **locked_visual_fields,
            **failsafe_fields,
            **debug_fields,
            **stop_point_correction_fields,
            **effective_stop_fields,
            "selected_stop_point_x": (
                stop_point_correction_fields["corrected_stop_point_world_x"]
                if stop_point_correction_fields["stop_point_corrected_from_bad_carla"]
                else distance.selected_stop_point_x
            ),
            "selected_stop_point_y": (
                stop_point_correction_fields["corrected_stop_point_world_y"]
                if stop_point_correction_fields["stop_point_corrected_from_bad_carla"]
                else distance.selected_stop_point_y
            ),
            "stop_point_route_distance_m": (
                round(float(corrected_stop_point_route_s), 3)
                if stop_point_correction_fields["stop_point_corrected_from_bad_carla"]
                and corrected_stop_point_route_s is not None
                else distance.stop_point_route_distance_m
            ),
            "selected_delta_s_m": (
                round(float(selected_stop_distance), 3)
                if stop_point_correction_fields["stop_point_corrected_from_bad_carla"]
                and selected_stop_distance is not None
                else distance.selected_delta_s_m
            ),
            "stop_point_distance_m": round(float(selected_stop_distance), 3) if selected_stop_distance is not None else None,
            "tl_state_machine_state": decision.state,
            "required_stop_distance_m": round(float(decision.required_stop_distance_m), 3),
            "desired_speed_raw_mps": round(float(decision.desired_speed_raw_mps), 3),
            "desired_speed_smoothed_mps": round(float(decision.desired_speed_mps), 3),
            "tl_desired_speed_mps": round(float(decision.desired_speed_mps), 3),
            "tl_stop_request": bool(decision.stop_request),
            "red_stop_commit_threshold_m": round(float(decision.red_stop_commit_threshold_m), 3),
            "red_stop_commit_active": bool(decision.red_stop_commit_active),
            "visual_red_brake": bool(decision.visual_red_brake),
            "visual_red_commit": bool(decision.visual_red_commit),
            "visual_red_hard_commit": bool(decision.visual_red_hard_commit),
            "visual_red_approach": bool(decision.visual_red_approach),
            "visual_red_approach_target_mps": round(float(decision.visual_red_approach_target_mps), 3),
            "stop_commit_reason": decision.stop_commit_reason,
            "red_creep_active": bool(decision.red_creep_active),
            "red_creep_target_mps": round(float(decision.red_creep_target_mps), 3),
            "red_creep_target_speed_mps": round(float(decision.red_creep_target_mps), 3),
            "red_creep_remaining_m": round(float(decision.red_creep_remaining_m), 3),
            "red_creep_elapsed_s": round(float(decision.red_creep_elapsed_s), 3),
            "red_creep_stop_threshold_m": round(float(decision.red_creep_stop_threshold_m), 3),
            "red_creep_reason": decision.red_creep_reason,
            "stopped_too_far_from_stop_point": bool(decision.stopped_too_far_from_stop_point),
            "final_stop_distance_m": round(float(decision.final_stop_distance_m), 3),
            "point_of_no_return": bool(decision.point_of_no_return),
            "passed_light": bool(decision.passed_light),
            "post_light_ignore_active": bool(decision.post_light_ignore_active),
            "tl_decision_reason": decision.reason,
            "tl_is_front_relevant": bool(relevant),
            "current_speed_mps": round(float(current_speed), 3),
            **lock_debug_fields,
            **red_stop_hold_debug_fields,
        }

        lane_plan.update(tl_fields)
        lane_plan["target_speed_mps"] = round(float(final_speed), 3)
        lane_plan["stop_request"] = final_stop_request
        lane_plan["reason"] = final_reason
        lane_plan["stop_reason"] = stop_reason
        lane_plan["lane_target_speed_mps"] = round(float(lane_speed), 3)
        lane_plan["final_target_speed_source"] = (
            "mission_stop" if mission_stop else "traffic_light" if final_speed < lane_speed - 0.05 else "lane_follow"
        )

        self.draw_stop_point_debug(lane_plan)

        msg = String()
        msg.data = json.dumps(lane_plan, ensure_ascii=False)
        self.plan_pub.publish(msg)
        self.log_runtime(lane_plan)

    # -------------------------
    # Debug / log block
    # -------------------------
    def log_runtime(self, payload: dict):
        record = {
            "ego_x": payload.get("ego_x"),
            "ego_y": payload.get("ego_y"),
            "ego_yaw": payload.get("ego_yaw"),
            "current_speed_mps": payload.get("current_speed_mps"),
            "active_mission_target": payload.get("active_mission_target"),
            "mission_state": payload.get("mission_state"),
            "route_index": payload.get("route_index"),
            "route_length": payload.get("route_length"),
            "lane_target_speed_mps": payload.get("lane_target_speed_mps"),
            "target_speed_mps": payload.get("target_speed_mps"),
            "stop_request": payload.get("stop_request"),
            "front_bumper_x": payload.get("front_bumper_x"),
            "front_bumper_y": payload.get("front_bumper_y"),
            "debug_draw_stop_points_enabled": payload.get("debug_draw_stop_points_enabled"),
            "selected_stop_point_world": payload.get("selected_stop_point_world"),
            "final_stop_point_world": payload.get("final_stop_point_world"),
            "locked_stop_point_world": payload.get("locked_stop_point_world"),
            "front_bumper_world": payload.get("front_bumper_world"),
            "raw_visual_stopline_estimated_world": payload.get("raw_visual_stopline_estimated_world"),
            "carla_aligned_visual_stopline_world": payload.get("carla_aligned_visual_stopline_world"),
            "stop_point_visual_alignment_error_m": payload.get("stop_point_visual_alignment_error_m"),
            "selected_stop_point_suspect": payload.get("selected_stop_point_suspect"),
            "stop_point_corrected_from_bad_carla": payload.get("stop_point_corrected_from_bad_carla"),
            "stop_point_correction_reason": payload.get("stop_point_correction_reason"),
            "original_carla_stop_distance": payload.get("original_carla_stop_distance"),
            "original_selected_stop_distance": payload.get("original_selected_stop_distance"),
            "visual_stopline_distance_for_correction": payload.get("visual_stopline_distance_for_correction"),
            "corrected_stop_point_source": payload.get("corrected_stop_point_source"),
            "corrected_stop_point_world_x": payload.get("corrected_stop_point_world_x"),
            "corrected_stop_point_world_y": payload.get("corrected_stop_point_world_y"),
            "corrected_stop_point_world_z": payload.get("corrected_stop_point_world_z"),
            "corrected_point_ignored_no_valid_association": payload.get(
                "corrected_point_ignored_no_valid_association"
            ),
            "corrected_point_ignored_reason": payload.get("corrected_point_ignored_reason"),
            "corrected_point_would_have_distance": payload.get("corrected_point_would_have_distance"),
            "tl_speed_override_cleared_for_no_association": payload.get(
                "tl_speed_override_cleared_for_no_association"
            ),
            "visual_correction_rejected_associated_light_too_far": payload.get(
                "visual_correction_rejected_associated_light_too_far"
            ),
            "visual_correction_reject_assoc_distance": payload.get(
                "visual_correction_reject_assoc_distance"
            ),
            "visual_correction_reject_visual_distance": payload.get(
                "visual_correction_reject_visual_distance"
            ),
            "correction_applied_to_decision": payload.get("correction_applied_to_decision"),
            "effective_stop_distance_m": payload.get("effective_stop_distance_m"),
            "effective_stop_source": payload.get("effective_stop_source"),
            "effective_stop_point_x": payload.get("effective_stop_point_x"),
            "effective_stop_point_y": payload.get("effective_stop_point_y"),
            "effective_stop_valid": payload.get("effective_stop_valid"),
            "effective_stop_reason": payload.get("effective_stop_reason"),
            "distance_conflict_detected": payload.get("distance_conflict_detected"),
            "ignored_distance_sources": payload.get("ignored_distance_sources"),
            "visual_distance_primary": payload.get("visual_distance_primary"),
            "last_good_visual_distance_m": payload.get("last_good_visual_distance_m"),
            "last_good_visual_age_s": payload.get("last_good_visual_age_s"),
            "visual_last_good_hold_active": payload.get("visual_last_good_hold_active"),
            "visual_stuck_overrun_detected": payload.get("visual_stuck_overrun_detected"),
            "visual_stuck_overrun_reason": payload.get("visual_stuck_overrun_reason"),
            "fallback_blocked_by_recent_visual": payload.get("fallback_blocked_by_recent_visual"),
            "fallback_blocked_source": payload.get("fallback_blocked_source"),
            "front_dist_for_commit": payload.get("front_dist_for_commit"),
            "locked_dist_for_commit": payload.get("locked_dist_for_commit"),
            "raw_visual_dist_for_commit": payload.get("raw_visual_dist_for_commit"),
            "commit_distance_source": payload.get("commit_distance_source"),
            "commit_blocked_by_raw_visual": payload.get("commit_blocked_by_raw_visual"),
            "commit_overrode_raw_visual_mismatch": payload.get("commit_overrode_raw_visual_mismatch"),
            "euclidean_ego_to_stop_m": payload.get("euclidean_ego_to_stop_m"),
            "euclidean_front_bumper_to_stop_m": payload.get("euclidean_front_bumper_to_stop_m"),
            "route_delta_s_m": payload.get("route_delta_s_m"),
            "ego_route_s": payload.get("ego_route_s"),
            "front_bumper_route_s": payload.get("front_bumper_route_s"),
            "distance_reference": payload.get("distance_reference"),
            "stop_point_source": payload.get("stop_point_source"),
            "carla_stop_waypoint_visual_mismatch": payload.get("carla_stop_waypoint_visual_mismatch"),
            "locked_light_id": payload.get("locked_light_id"),
            "locked_stop_point_x": payload.get("locked_stop_point_x"),
            "locked_stop_point_y": payload.get("locked_stop_point_y"),
            "locked_route_s": payload.get("locked_route_s"),
            "locked_light_active": payload.get("locked_light_active"),
            "locked_light_color": payload.get("locked_light_color"),
            "candidate_switch_blocked": payload.get("candidate_switch_blocked"),
            "candidate_switch_block_reason": payload.get("candidate_switch_block_reason"),
            "green_release_source": payload.get("green_release_source"),
            "red_lock_release_reason": payload.get("red_lock_release_reason"),
            "lock_match_source": payload.get("lock_match_source"),
            "red_stop_hold_active": payload.get("red_stop_hold_active"),
            "red_stop_hold_reason": payload.get("red_stop_hold_reason"),
            "red_stop_hold_cleared_by_green": payload.get("red_stop_hold_cleared_by_green"),
            "tl_detected": payload.get("tl_detected"),
            "tl_color_raw": payload.get("tl_color_raw"),
            "tl_color_filtered": payload.get("tl_color_filtered"),
            "tl_confidence": payload.get("tl_confidence"),
            "candidate_count": payload.get("candidate_count"),
            "valid_route_candidate_count": payload.get("valid_route_candidate_count"),
            "selected_candidate_rank": payload.get("selected_candidate_rank"),
            "selected_delta_s_m": payload.get("selected_delta_s_m"),
            "selected_lateral_m": payload.get("selected_lateral_m"),
            "stop_point_lateral_to_route_m": payload.get("stop_point_lateral_to_route_m"),
            "top_candidates_debug": payload.get("top_candidates_debug"),
            "rejected_candidate_count": payload.get("rejected_candidate_count"),
            "direction_mismatch_soft": payload.get("direction_mismatch_soft"),
            "direction_reject_disabled_for_route_candidate": payload.get("direction_reject_disabled_for_route_candidate"),
            "tl_distance_m": payload.get("tl_distance_m"),
            "stop_point_distance_m": payload.get("stop_point_distance_m"),
            "distance_source": payload.get("distance_source"),
            "visual_stopline_detected": payload.get("visual_stopline_detected"),
            "visual_stopline_confidence": payload.get("visual_stopline_confidence"),
            "front_bumper_to_visual_stopline_m": payload.get("front_bumper_to_visual_stopline_m"),
            "visual_dist_raw": payload.get("visual_dist_raw"),
            "visual_dist_filtered": payload.get("visual_dist_filtered"),
            "visual_distance_jump_rejected": payload.get("visual_distance_jump_rejected"),
            "locked_visual_stopline_active": payload.get("locked_visual_stopline_active"),
            "locked_visual_stopline_route_s": payload.get("locked_visual_stopline_route_s"),
            "locked_visual_distance_m": payload.get("locked_visual_distance_m"),
            "raw_visual_distance_m": payload.get("raw_visual_distance_m"),
            "distance_source_for_decision": payload.get("distance_source_for_decision"),
            "distance_source_before_consistency": payload.get("distance_source_before_consistency"),
            "distance_source_after_consistency": payload.get("distance_source_after_consistency"),
            "locked_visual_consistency_valid": payload.get("locked_visual_consistency_valid"),
            "locked_visual_consistency_reject_reason": payload.get("locked_visual_consistency_reject_reason"),
            "locked_visual_distance_vs_carla_error_m": payload.get("locked_visual_distance_vs_carla_error_m"),
            "locked_visual_distance_vs_raw_error_m": payload.get("locked_visual_distance_vs_raw_error_m"),
            "locked_visual_reanchored_to_carla_stop": payload.get("locked_visual_reanchored_to_carla_stop"),
            "visual_distance_tracking_error_m": payload.get("visual_distance_tracking_error_m"),
            "visual_distance_monotonic_violation": payload.get("visual_distance_monotonic_violation"),
            "simple_red_visual_rule": payload.get("simple_red_visual_rule"),
            "red_visual_decision_distance_m": payload.get("red_visual_decision_distance_m"),
            "red_visual_slow_threshold_m": payload.get("red_visual_slow_threshold_m"),
            "red_visual_stop_threshold_m": payload.get("red_visual_stop_threshold_m"),
            "red_visual_hard_stop_threshold_m": payload.get("red_visual_hard_stop_threshold_m"),
            "carla_stop_ignored_because_visual_rule": payload.get("carla_stop_ignored_because_visual_rule"),
            "locked_visual_ignored_because_visual_rule": payload.get("locked_visual_ignored_because_visual_rule"),
            "carla_aligned_lock_ignored_because_visual_rule": payload.get(
                "carla_aligned_lock_ignored_because_visual_rule"
            ),
            "visual_lock_active": payload.get("visual_lock_active"),
            "visual_lock_route_s": payload.get("visual_lock_route_s"),
            "visual_lock_distance_m": payload.get("visual_lock_distance_m"),
            "visual_lock_created": payload.get("visual_lock_created"),
            "visual_lock_cleared_reason": payload.get("visual_lock_cleared_reason"),
            "visual_lock_suspect": payload.get("visual_lock_suspect"),
            "visual_lock_reject_reason": payload.get("visual_lock_reject_reason"),
            "visual_lock_vs_raw_error_m": payload.get("visual_lock_vs_raw_error_m"),
            "visual_lock_vs_front_error_m": payload.get("visual_lock_vs_front_error_m"),
            "visual_lock_used_for_commit": payload.get("visual_lock_used_for_commit"),
            "raw_visual_used_instead_of_bad_lock": payload.get("raw_visual_used_instead_of_bad_lock"),
            "red_visual_near_light_failsafe": payload.get("red_visual_near_light_failsafe"),
            "red_visual_near_light_failsafe_reason": payload.get("red_visual_near_light_failsafe_reason"),
            "red_visual_approach_elapsed_s": payload.get("red_visual_approach_elapsed_s"),
            "red_visual_distance_stuck": payload.get("red_visual_distance_stuck"),
            "red_visual_stuck_stop": payload.get("red_visual_stuck_stop"),
            "associated_light_distance_for_failsafe": payload.get("associated_light_distance_for_failsafe"),
            "red_stop_2m_any_valid_distance": payload.get("red_stop_2m_any_valid_distance"),
            "red_stop_2m_any_valid_distance_source": payload.get("red_stop_2m_any_valid_distance_source"),
            "red_stop_2m_any_valid_distance_value": payload.get("red_stop_2m_any_valid_distance_value"),
            "red_stop_2m_any_valid_distance_reason": payload.get("red_stop_2m_any_valid_distance_reason"),
            "red_corrected_point_overrun_full_brake": payload.get(
                "red_corrected_point_overrun_full_brake"
            ),
            "red_corrected_point_overrun_source": payload.get("red_corrected_point_overrun_source"),
            "associated_light_id": payload.get("associated_light_id"),
            "associated_light_color": payload.get("associated_light_color"),
            "associated_light_source": payload.get("associated_light_source"),
            "associated_light_distance": payload.get("associated_light_distance"),
            "associated_light_roi_score": payload.get("associated_light_roi_score"),
            "associated_light_above_stopline": payload.get("associated_light_above_stopline"),
            "associated_light_horizontal_overlap": payload.get("associated_light_horizontal_overlap"),
            "visual_stopline_light_association_valid": payload.get("visual_stopline_light_association_valid"),
            "visual_stopline_light_association_reason": payload.get("visual_stopline_light_association_reason"),
            "far_carla_candidate_ignored": payload.get("far_carla_candidate_ignored"),
            "far_carla_candidate_distance_m": payload.get("far_carla_candidate_distance_m"),
            "carla_candidate_suspect": payload.get("carla_candidate_suspect"),
            "carla_candidate_suspect_reason": payload.get("carla_candidate_suspect_reason"),
            "visual_association_latched": payload.get("visual_association_latched"),
            "visual_association_age_ms": payload.get("visual_association_age_ms"),
            "carla_stop_point_distance_m": payload.get("carla_stop_point_distance_m"),
            "selected_distance_m": payload.get("selected_distance_m"),
            "stopline_pixel_y": payload.get("stopline_pixel_y"),
            "stopline_pixel_x1": payload.get("stopline_pixel_x1"),
            "stopline_pixel_x2": payload.get("stopline_pixel_x2"),
            "stopline_width_px": payload.get("stopline_width_px"),
            "stopline_angle_deg": payload.get("stopline_angle_deg"),
            "stopline_reject_reason": payload.get("stopline_reject_reason"),
            "stopline_source": payload.get("stopline_source"),
            "using_visual_stopline": payload.get("using_visual_stopline"),
            "visual_stopline_age_ms": payload.get("visual_stopline_age_ms"),
            "distance_valid": payload.get("distance_valid"),
            "selected_light_id": payload.get("selected_light_id"),
            "selected_stop_point_x": payload.get("selected_stop_point_x"),
            "selected_stop_point_y": payload.get("selected_stop_point_y"),
            "stop_point_route_distance_m": payload.get("stop_point_route_distance_m"),
            "actor_route_distance_m": payload.get("actor_route_distance_m"),
            "actor_vs_stop_point_used": payload.get("actor_vs_stop_point_used"),
            "closest_route_index": payload.get("closest_route_index"),
            "route_corridor_width_m": payload.get("route_corridor_width_m"),
            "relevance_basis": payload.get("relevance_basis"),
            "reject_reason_detail": payload.get("reject_reason_detail"),
            "is_in_front_of_ego": payload.get("is_in_front_of_ego"),
            "is_on_route_corridor": payload.get("is_on_route_corridor"),
            "is_same_direction_relevant": payload.get("is_same_direction_relevant"),
            "tl_is_front_relevant": payload.get("tl_is_front_relevant"),
            "roi_rejected_reason": payload.get("roi_rejected_reason"),
            "tl_state_machine_state": payload.get("tl_state_machine_state"),
            "required_stop_distance_m": payload.get("required_stop_distance_m"),
            "desired_speed_raw_mps": payload.get("desired_speed_raw_mps"),
            "desired_speed_smoothed_mps": payload.get("desired_speed_smoothed_mps"),
            "tl_stop_request": payload.get("tl_stop_request"),
            "red_stop_commit_threshold_m": payload.get("red_stop_commit_threshold_m"),
            "red_stop_commit_active": payload.get("red_stop_commit_active"),
            "visual_red_brake": payload.get("visual_red_brake"),
            "visual_red_commit": payload.get("visual_red_commit"),
            "visual_red_hard_commit": payload.get("visual_red_hard_commit"),
            "visual_red_approach": payload.get("visual_red_approach"),
            "visual_red_approach_target_mps": payload.get("visual_red_approach_target_mps"),
            "stop_commit_reason": payload.get("stop_commit_reason"),
            "red_creep_active": payload.get("red_creep_active"),
            "red_creep_target_mps": payload.get("red_creep_target_mps"),
            "red_creep_target_speed_mps": payload.get("red_creep_target_speed_mps"),
            "red_creep_remaining_m": payload.get("red_creep_remaining_m"),
            "red_creep_elapsed_s": payload.get("red_creep_elapsed_s"),
            "red_creep_stop_threshold_m": payload.get("red_creep_stop_threshold_m"),
            "red_creep_reason": payload.get("red_creep_reason"),
            "stopped_too_far_from_stop_point": payload.get("stopped_too_far_from_stop_point"),
            "final_stop_distance_m": payload.get("final_stop_distance_m"),
            "point_of_no_return": payload.get("point_of_no_return"),
            "passed_light": payload.get("passed_light"),
            "post_light_ignore_active": payload.get("post_light_ignore_active"),
            "tl_decision_reason": payload.get("tl_decision_reason"),
            "final_target_speed_source": payload.get("final_target_speed_source"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "traffic_light_manager "
                f"color={record['tl_color_filtered']} state={record['tl_state_machine_state']} "
                f"rel={record['tl_is_front_relevant']} dist={record['stop_point_distance_m']} "
                f"ref={record['distance_reference']} front_dist={record['euclidean_front_bumper_to_stop_m']} "
                f"debug_draw_stop_points_enabled={record['debug_draw_stop_points_enabled']} "
                f"selected_stop_point_world={record['selected_stop_point_world']} "
                f"final_stop_point_world={record['final_stop_point_world']} "
                f"locked_stop_point_world={record['locked_stop_point_world']} "
                f"front_bumper_world={record['front_bumper_world']} "
                f"raw_visual_stopline_estimated_world={record['raw_visual_stopline_estimated_world']} "
                f"carla_aligned_visual_stopline_world={record['carla_aligned_visual_stopline_world']} "
                f"stop_point_visual_alignment_error_m={record['stop_point_visual_alignment_error_m']} "
                f"selected_stop_point_suspect={record['selected_stop_point_suspect']} "
                f"stop_point_corrected_from_bad_carla={record['stop_point_corrected_from_bad_carla']} "
                f"stop_point_correction_reason={record['stop_point_correction_reason']} "
                f"original_carla_stop_distance={record['original_carla_stop_distance']} "
                f"original_selected_stop_distance={record['original_selected_stop_distance']} "
                f"visual_stopline_distance_for_correction={record['visual_stopline_distance_for_correction']} "
                f"corrected_stop_point_source={record['corrected_stop_point_source']} "
                f"corrected_stop_point_world=({record['corrected_stop_point_world_x']},"
                f"{record['corrected_stop_point_world_y']},"
                f"{record['corrected_stop_point_world_z']}) "
                f"corrected_point_ignored_no_valid_association="
                f"{record['corrected_point_ignored_no_valid_association']} "
                f"corrected_point_ignored_reason={record['corrected_point_ignored_reason']} "
                f"corrected_point_would_have_distance={record['corrected_point_would_have_distance']} "
                f"tl_speed_override_cleared_for_no_association="
                f"{record['tl_speed_override_cleared_for_no_association']} "
                f"visual_correction_rejected_associated_light_too_far="
                f"{record['visual_correction_rejected_associated_light_too_far']} "
                f"visual_correction_reject_assoc_distance="
                f"{record['visual_correction_reject_assoc_distance']} "
                f"visual_correction_reject_visual_distance="
                f"{record['visual_correction_reject_visual_distance']} "
                f"correction_applied_to_decision={record['correction_applied_to_decision']} "
                f"effective_stop_distance_m={record['effective_stop_distance_m']} "
                f"effective_stop_source={record['effective_stop_source']} "
                f"effective_stop_point=({record['effective_stop_point_x']},"
                f"{record['effective_stop_point_y']}) "
                f"effective_stop_valid={record['effective_stop_valid']} "
                f"effective_stop_reason={record['effective_stop_reason']} "
                f"distance_conflict_detected={record['distance_conflict_detected']} "
                f"ignored_distance_sources={record['ignored_distance_sources']} "
                f"visual_distance_primary={record['visual_distance_primary']} "
                f"last_good_visual_distance_m={record['last_good_visual_distance_m']} "
                f"last_good_visual_age_s={record['last_good_visual_age_s']} "
                f"visual_last_good_hold_active={record['visual_last_good_hold_active']} "
                f"visual_stuck_overrun_detected={record['visual_stuck_overrun_detected']} "
                f"visual_stuck_overrun_reason={record['visual_stuck_overrun_reason']} "
                f"fallback_blocked_by_recent_visual={record['fallback_blocked_by_recent_visual']} "
                f"fallback_blocked_source={record['fallback_blocked_source']} "
                f"front_dist_for_commit={record['front_dist_for_commit']} "
                f"locked_dist_for_commit={record['locked_dist_for_commit']} "
                f"raw_visual_dist_for_commit={record['raw_visual_dist_for_commit']} "
                f"commit_distance_source={record['commit_distance_source']} "
                f"commit_blocked_by_raw_visual={record['commit_blocked_by_raw_visual']} "
                f"commit_overrode_raw_visual_mismatch={record['commit_overrode_raw_visual_mismatch']} "
                f"src={record['distance_source']} vis={record['using_visual_stopline']} "
                f"visual_dist_raw={record['visual_dist_raw']} "
                f"visual_dist_filtered={record['visual_dist_filtered']} "
                f"visual_distance_jump_rejected={record['visual_distance_jump_rejected']} "
                f"locked_visual_stopline_active={record['locked_visual_stopline_active']} "
                f"locked_visual_distance_m={record['locked_visual_distance_m']} "
                f"distance_source_for_decision={record['distance_source_for_decision']} "
                f"distance_source_before_consistency={record['distance_source_before_consistency']} "
                f"distance_source_after_consistency={record['distance_source_after_consistency']} "
                f"locked_visual_consistency_valid={record['locked_visual_consistency_valid']} "
                f"locked_visual_consistency_reject_reason={record['locked_visual_consistency_reject_reason']} "
                f"locked_visual_distance_vs_carla_error_m={record['locked_visual_distance_vs_carla_error_m']} "
                f"locked_visual_distance_vs_raw_error_m={record['locked_visual_distance_vs_raw_error_m']} "
                f"locked_visual_reanchored_to_carla_stop={record['locked_visual_reanchored_to_carla_stop']} "
                f"visual_distance_tracking_error_m={record['visual_distance_tracking_error_m']} "
                f"visual_distance_monotonic_violation={record['visual_distance_monotonic_violation']} "
                f"simple_red_visual_rule={record['simple_red_visual_rule']} "
                f"red_visual_decision_distance_m={record['red_visual_decision_distance_m']} "
                f"red_visual_thresholds=({record['red_visual_slow_threshold_m']},"
                f"{record['red_visual_stop_threshold_m']},"
                f"{record['red_visual_hard_stop_threshold_m']}) "
                f"carla_stop_ignored_because_visual_rule={record['carla_stop_ignored_because_visual_rule']} "
                f"visual_lock_active={record['visual_lock_active']} "
                f"visual_lock_distance_m={record['visual_lock_distance_m']} "
                f"visual_lock_created={record['visual_lock_created']} "
                f"visual_lock_cleared_reason={record['visual_lock_cleared_reason']} "
                f"visual_lock_suspect={record['visual_lock_suspect']} "
                f"visual_lock_reject_reason={record['visual_lock_reject_reason']} "
                f"visual_lock_vs_raw_error_m={record['visual_lock_vs_raw_error_m']} "
                f"visual_lock_vs_front_error_m={record['visual_lock_vs_front_error_m']} "
                f"visual_lock_used_for_commit={record['visual_lock_used_for_commit']} "
                f"raw_visual_used_instead_of_bad_lock={record['raw_visual_used_instead_of_bad_lock']} "
                f"red_visual_near_light_failsafe={record['red_visual_near_light_failsafe']} "
                f"red_visual_near_light_failsafe_reason={record['red_visual_near_light_failsafe_reason']} "
                f"red_visual_approach_elapsed_s={record['red_visual_approach_elapsed_s']} "
                f"red_visual_distance_stuck={record['red_visual_distance_stuck']} "
                f"red_visual_stuck_stop={record['red_visual_stuck_stop']} "
                f"associated_light_distance_for_failsafe={record['associated_light_distance_for_failsafe']} "
                f"red_stop_2m_any_valid_distance={record['red_stop_2m_any_valid_distance']} "
                f"red_stop_2m_any_valid_distance_source={record['red_stop_2m_any_valid_distance_source']} "
                f"red_stop_2m_any_valid_distance_value={record['red_stop_2m_any_valid_distance_value']} "
                f"red_stop_2m_any_valid_distance_reason={record['red_stop_2m_any_valid_distance_reason']} "
                f"red_corrected_point_overrun_full_brake="
                f"{record['red_corrected_point_overrun_full_brake']} "
                f"red_corrected_point_overrun_source={record['red_corrected_point_overrun_source']} "
                f"associated_light_id={record['associated_light_id']} "
                f"associated_light_color={record['associated_light_color']} "
                f"visual_stopline_light_association_valid={record['visual_stopline_light_association_valid']} "
                f"visual_stopline_light_association_reason={record['visual_stopline_light_association_reason']} "
                f"visual_association_latched={record['visual_association_latched']} "
                f"visual_association_age_ms={record['visual_association_age_ms']} "
                f"far_carla_candidate_ignored={record['far_carla_candidate_ignored']} "
                f"carla_candidate_suspect={record['carla_candidate_suspect']} "
                f"valid={record['valid_route_candidate_count']}/{record['candidate_count']} "
                f"sel_id={record['selected_light_id']} delta_s={record['selected_delta_s_m']} "
                f"lock={record['locked_light_active']} lock_id={record['locked_light_id']} "
                f"red_stop_hold_active={record['red_stop_hold_active']} "
                f"red_stop_hold_reason={record['red_stop_hold_reason']} "
                f"red_stop_hold_cleared_by_green={record['red_stop_hold_cleared_by_green']} "
                f"switch_blocked={record['candidate_switch_blocked']} "
                f"lat={record['selected_lateral_m']} basis={record['relevance_basis']} "
                f"dir_soft={record['direction_mismatch_soft']} "
                f"reject={record['reject_reason_detail']} "
                f"lane_v={record['lane_target_speed_mps']} final_v={record['target_speed_mps']} "
                f"stop={record['tl_stop_request']} red_commit={record['red_stop_commit_active']} "
                f"visual_red_brake={record['visual_red_brake']} "
                f"visual_red_commit={record['visual_red_commit']} "
                f"visual_red_approach={record['visual_red_approach']} "
                f"creep={record['red_creep_active']} too_far={record['stopped_too_far_from_stop_point']} "
                f"creep_t={record['red_creep_elapsed_s']} "
                f"commit_thr={record['red_stop_commit_threshold_m']} reason={record['tl_decision_reason']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = TrafficLightManagerNode()
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
