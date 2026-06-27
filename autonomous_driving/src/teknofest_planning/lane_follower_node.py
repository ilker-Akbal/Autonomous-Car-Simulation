import json
import math
import time
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_planning.route_geometry import signed_angle_diff_deg
from teknofest_planning.velocity_profile import (
    clamp,
    compute_target_speed_from_route,
)


class LaneFollower(Node):
    def __init__(self):
        super().__init__("lane_follower")

        self.declare_parameter("base_lookahead_m", 5.0)
        self.declare_parameter("lookahead_gain", 1.2)
        self.declare_parameter("min_lookahead_m", 4.5)
        self.declare_parameter("max_lookahead_m", 14.0)
        self.declare_parameter("dynamic_lookahead_enabled", True)
        self.declare_parameter("wheel_base_m", 2.8)
        self.declare_parameter("max_steer_angle_rad", 0.65)
        self.declare_parameter("target_speed_mps", 3.0)
        self.declare_parameter("cruise_speed_mps", 4.5)
        self.declare_parameter("max_speed_mps", 6.0)
        self.declare_parameter("min_turn_speed_mps", 2.0)
        self.declare_parameter("speed_boost_enabled", True)
        self.declare_parameter("nominal_speed_boost_mps", 2.0)
        self.declare_parameter("sharp_turn_yaw_deg", 45.0)
        self.declare_parameter("moderate_turn_yaw_deg", 18.0)
        self.declare_parameter("speed_slew_up_mps_per_s", 0.8)
        self.declare_parameter("speed_slew_down_mps_per_s", 2.0)
        self.declare_parameter("route_event_timeout_s", 0.75)
        self.declare_parameter("event_stop_deceleration_mps2", 1.5)
        self.declare_parameter("event_stop_buffer_m", 1.0)
        self.declare_parameter("lane_assist_only", True)
        self.declare_parameter("right_lane_lateral_bias_enabled", True)
        self.declare_parameter("right_lane_lateral_bias_m", 0.65)
        self.declare_parameter("right_lane_lateral_bias_min_speed_mps", 0.5)
        self.declare_parameter("right_lane_lateral_bias_disable_in_junction", True)
        self.declare_parameter("right_lane_lateral_bias_disable_in_turn", True)
        self.declare_parameter("right_lane_lateral_bias_safety_margin_m", 0.35)
        self.declare_parameter("vehicle_half_width_m", 0.95)
        self.declare_parameter("lane_departure_guard_enabled", True)
        self.declare_parameter("lane_departure_lateral_threshold_m", 1.2)
        self.declare_parameter("route_recovery_speed_mps", 2.5)
        self.declare_parameter("route_conflict_heading_threshold_deg", 35.0)
        self.declare_parameter("route_index_hysteresis_enabled", True)
        self.declare_parameter("max_route_index_jump", 8)
        self.declare_parameter("steering_rate_limit_enabled", True)
        self.declare_parameter("max_steer_delta", 0.08)
        self.declare_parameter("min_nonzero_target_speed_mps", 1.2)
        self.declare_parameter("nominal_min_speed_mps", 1.2)
        self.declare_parameter("junction_recovery_min_speed_mps", 1.0)
        self.declare_parameter("junction_exit_min_speed_mps", 1.0)
        self.declare_parameter("task_pull_over_start_distance_m", 18.0)
        self.declare_parameter("task_pull_over_final_distance_m", 5.0)
        self.declare_parameter("task_pull_over_lateral_offset_m", 1.0)
        self.declare_parameter("task_stop_reached_distance_m", 2.0)
        self.declare_parameter("task_pull_over_approach_speed_mps", 2.0)
        self.declare_parameter("task_pull_over_final_speed_mps", 1.2)
        self.declare_parameter("task_pull_over_crawl_speed_mps", 0.6)
        self.declare_parameter("task_pull_over_keep_bias_until_reached", True)
        self.declare_parameter("task_stop_final_phase_latch_enabled", True)
        self.declare_parameter("task_stop_final_latch_distance_m", 1.0)
        self.declare_parameter("task_stop_overshoot_guard_distance_m", 0.75)
        self.declare_parameter("task_stop_overshoot_guard_speed_mps", 0.0)
        self.declare_parameter("task_stop_alignment_enabled", True)
        self.declare_parameter("task_stop_alignment_start_distance_m", 3.0)
        self.declare_parameter("task_stop_alignment_yaw_tolerance_deg", 12.0)
        self.declare_parameter("task_stop_alignment_speed_mps", 0.8)
        self.declare_parameter("task_stop_alignment_target_ahead_m", 2.0)
        self.declare_parameter("task_stop_approach_cruise_speed_mps", 2.0)
        self.declare_parameter("task_stop_pre_align_speed_mps", 1.2)
        self.declare_parameter("task_stop_final_align_speed_mps", 0.8)
        self.declare_parameter("task_stop_min_creep_speed_mps", 0.6)
        self.declare_parameter("task_stop_no_stop_before_final_distance_m", 1.0)
        self.declare_parameter("task_stop_phase_hysteresis_m", 0.75)
        self.declare_parameter("rate_hz", 20.0)

        self.base_lookahead_m = float(self.get_parameter("base_lookahead_m").value)
        self.lookahead_gain = float(self.get_parameter("lookahead_gain").value)
        self.min_lookahead_m = float(self.get_parameter("min_lookahead_m").value)
        self.max_lookahead_m = float(self.get_parameter("max_lookahead_m").value)
        self.dynamic_lookahead_enabled = bool(self.get_parameter("dynamic_lookahead_enabled").value)
        self.wheel_base_m = float(self.get_parameter("wheel_base_m").value)
        self.max_steer_angle_rad = float(self.get_parameter("max_steer_angle_rad").value)
        self.target_speed_mps_param = float(self.get_parameter("target_speed_mps").value)
        self.cruise_speed_mps = float(self.get_parameter("cruise_speed_mps").value)
        self.max_speed_mps = float(self.get_parameter("max_speed_mps").value)
        self.min_turn_speed_mps = float(self.get_parameter("min_turn_speed_mps").value)
        self.speed_boost_enabled = bool(self.get_parameter("speed_boost_enabled").value)
        self.nominal_speed_boost_mps = float(self.get_parameter("nominal_speed_boost_mps").value)
        self.sharp_turn_yaw_deg = float(self.get_parameter("sharp_turn_yaw_deg").value)
        self.moderate_turn_yaw_deg = float(self.get_parameter("moderate_turn_yaw_deg").value)
        self.speed_slew_up_mps_per_s = float(self.get_parameter("speed_slew_up_mps_per_s").value)
        self.speed_slew_down_mps_per_s = float(self.get_parameter("speed_slew_down_mps_per_s").value)
        self.route_event_timeout_s = float(self.get_parameter("route_event_timeout_s").value)
        self.event_stop_deceleration_mps2 = float(
            self.get_parameter("event_stop_deceleration_mps2").value
        )
        self.event_stop_buffer_m = float(self.get_parameter("event_stop_buffer_m").value)
        self.lane_assist_only = bool(self.get_parameter("lane_assist_only").value)
        self.right_lane_lateral_bias_enabled = bool(
            self.get_parameter("right_lane_lateral_bias_enabled").value
        )
        self.right_lane_lateral_bias_m = float(
            self.get_parameter("right_lane_lateral_bias_m").value
        )
        self.right_lane_lateral_bias_min_speed_mps = float(
            self.get_parameter("right_lane_lateral_bias_min_speed_mps").value
        )
        self.right_lane_lateral_bias_disable_in_junction = bool(
            self.get_parameter("right_lane_lateral_bias_disable_in_junction").value
        )
        self.right_lane_lateral_bias_disable_in_turn = bool(
            self.get_parameter("right_lane_lateral_bias_disable_in_turn").value
        )
        self.right_lane_lateral_bias_safety_margin_m = float(
            self.get_parameter("right_lane_lateral_bias_safety_margin_m").value
        )
        self.vehicle_half_width_m = float(self.get_parameter("vehicle_half_width_m").value)
        self.lane_departure_guard_enabled = bool(
            self.get_parameter("lane_departure_guard_enabled").value
        )
        self.lane_departure_lateral_threshold_m = float(
            self.get_parameter("lane_departure_lateral_threshold_m").value
        )
        self.route_recovery_speed_mps = float(self.get_parameter("route_recovery_speed_mps").value)
        self.route_conflict_heading_threshold_deg = float(
            self.get_parameter("route_conflict_heading_threshold_deg").value
        )
        self.route_index_hysteresis_enabled = bool(
            self.get_parameter("route_index_hysteresis_enabled").value
        )
        self.max_route_index_jump = int(self.get_parameter("max_route_index_jump").value)
        self.steering_rate_limit_enabled = bool(
            self.get_parameter("steering_rate_limit_enabled").value
        )
        self.max_steer_delta = float(self.get_parameter("max_steer_delta").value)
        self.min_nonzero_target_speed_mps = float(
            self.get_parameter("min_nonzero_target_speed_mps").value
        )
        self.nominal_min_speed_mps = float(
            self.get_parameter("nominal_min_speed_mps").value
        )
        self.junction_recovery_min_speed_mps = float(
            self.get_parameter("junction_recovery_min_speed_mps").value
        )
        self.junction_exit_min_speed_mps = float(
            self.get_parameter("junction_exit_min_speed_mps").value
        )
        self.task_pull_over_start_distance_m = float(
            self.get_parameter("task_pull_over_start_distance_m").value
        )
        self.task_pull_over_final_distance_m = float(
            self.get_parameter("task_pull_over_final_distance_m").value
        )
        self.task_pull_over_lateral_offset_m = float(
            self.get_parameter("task_pull_over_lateral_offset_m").value
        )
        self.task_stop_reached_distance_m = float(
            self.get_parameter("task_stop_reached_distance_m").value
        )
        self.task_pull_over_approach_speed_mps = float(
            self.get_parameter("task_pull_over_approach_speed_mps").value
        )
        self.task_pull_over_final_speed_mps = float(
            self.get_parameter("task_pull_over_final_speed_mps").value
        )
        self.task_pull_over_crawl_speed_mps = float(
            self.get_parameter("task_pull_over_crawl_speed_mps").value
        )
        self.task_pull_over_keep_bias_until_reached = bool(
            self.get_parameter("task_pull_over_keep_bias_until_reached").value
        )
        self.task_stop_final_phase_latch_enabled = bool(
            self.get_parameter("task_stop_final_phase_latch_enabled").value
        )
        self.task_stop_final_latch_distance_m = float(
            self.get_parameter("task_stop_final_latch_distance_m").value
        )
        self.task_stop_overshoot_guard_distance_m = float(
            self.get_parameter("task_stop_overshoot_guard_distance_m").value
        )
        self.task_stop_overshoot_guard_speed_mps = float(
            self.get_parameter("task_stop_overshoot_guard_speed_mps").value
        )
        self.task_stop_alignment_enabled = bool(
            self.get_parameter("task_stop_alignment_enabled").value
        )
        self.task_stop_alignment_start_distance_m = float(
            self.get_parameter("task_stop_alignment_start_distance_m").value
        )
        self.task_stop_alignment_yaw_tolerance_deg = float(
            self.get_parameter("task_stop_alignment_yaw_tolerance_deg").value
        )
        self.task_stop_alignment_speed_mps = float(
            self.get_parameter("task_stop_alignment_speed_mps").value
        )
        self.task_stop_alignment_target_ahead_m = float(
            self.get_parameter("task_stop_alignment_target_ahead_m").value
        )
        self.task_stop_approach_cruise_speed_mps = float(
            self.get_parameter("task_stop_approach_cruise_speed_mps").value
        )
        self.task_stop_pre_align_speed_mps = float(
            self.get_parameter("task_stop_pre_align_speed_mps").value
        )
        self.task_stop_final_align_speed_mps = float(
            self.get_parameter("task_stop_final_align_speed_mps").value
        )
        self.task_stop_min_creep_speed_mps = float(
            self.get_parameter("task_stop_min_creep_speed_mps").value
        )
        self.task_stop_no_stop_before_final_distance_m = float(
            self.get_parameter("task_stop_no_stop_before_final_distance_m").value
        )
        self.task_stop_phase_hysteresis_m = float(
            self.get_parameter("task_stop_phase_hysteresis_m").value
        )
        self.task_pose_approach_start_distance_m = min(
            self.task_pull_over_start_distance_m,
            10.0,
        )
        self.task_pose_pre_stop_distance_m = 6.0
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        if self.target_speed_mps_param != 3.0 and self.cruise_speed_mps == 4.5:
            self.cruise_speed_mps = self.target_speed_mps_param

        self.route = None
        self.ego = None
        self.route_event = None
        self.mission_goal = None
        self.last_route_time = 0.0
        self.last_status_time = 0.0
        self.last_route_event_time = 0.0
        self.last_mission_goal_time = 0.0
        self.last_lane_cte = None
        self.last_lane_cte_time = 0.0
        self.last_target_speed = 0.0
        self.last_nearest_index = None
        self.last_steer_cmd = 0.0
        self._last_logged_route_event = None
        self._final_task_stop_latch_active = False
        self._final_task_stop_latch_key: Optional[tuple[Any, Any, Any]] = None
        self._final_task_stop_latch_reason: Optional[str] = None
        self.runtime_logger = RuntimeJsonlLogger(
            node_name="lane_follower",
            file_name="lane_follower.jsonl",
        )
        self.runtime_logger.update_summary({
            "lane_follower_log": self.runtime_logger.path(),
        })
        self.get_logger().info(
            f"LaneFollower JSONL logging -> {self.runtime_logger.path()}"
        )

        self.create_subscription(String, "/adas/planning/route", self._route_cb, 10)
        self.create_subscription(String, "/adas/carla/status", self._status_cb, 10)
        self.create_subscription(String, "/adas/planning/route_events", self._route_event_cb, 10)
        self.create_subscription(String, "/adas/mission/current_goal", self._mission_goal_cb, 10)
        self.create_subscription(Float32, "/adas/perception/lane_cte", self._lane_cte_cb, 10)

        self.plan_pub = self.create_publisher(String, "/adas/planning/lane_plan", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/lane_debug", 10)

        self.timer = self.create_timer(1.0 / max(1.0, self.rate_hz), self._run)

    def _route_cb(self, msg: String):
        try:
            self.route = json.loads(msg.data)
            self.last_route_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse route JSON")

    def _status_cb(self, msg: String):
        try:
            self.ego = json.loads(msg.data)
            self.last_status_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse carla status JSON")

    def _route_event_cb(self, msg: String):
        try:
            self.route_event = json.loads(msg.data)
            self.last_route_event_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse route_events JSON")

    def _mission_goal_cb(self, msg: String):
        try:
            self.mission_goal = json.loads(msg.data)
            self.last_mission_goal_time = time.time()
        except Exception:
            self.get_logger().warn("Failed to parse mission current_goal JSON")

    def _lane_cte_cb(self, msg: Float32):
        self.last_lane_cte = float(msg.data)
        self.last_lane_cte_time = time.time()

    def _mission_goal_latch_key(self) -> Optional[tuple[Any, Any, Any]]:
        if self.mission_goal is None:
            return None
        return (
            self.mission_goal.get("goal_name"),
            self.mission_goal.get("goal_kind"),
            self.mission_goal.get("goal_index"),
        )

    def _reset_final_task_stop_latch(self) -> None:
        self._final_task_stop_latch_active = False
        self._final_task_stop_latch_key = None
        self._final_task_stop_latch_reason = None

    def _route_only_debug_defaults(self) -> dict[str, Any]:
        return {
            "target_source": "fallback_lane",
            "route_locked": True,
            "lane_assist_only": self.lane_assist_only,
            "lane_preference": "right",
            "route_lane_id": None,
            "requested_right_lane_id": None,
            "selected_lane_id": None,
            "selected_road_id": None,
            "right_lane_selected": False,
            "right_lane_reason": None,
            "right_lane_projection_status": None,
            "right_lane_projection_rejected_reason": None,
            "right_lane_fallback_used": False,
            "fallback_kept_right_lane": False,
            "lane_jump_disabled": True,
            "selected_lane_lateral_right_m": None,
            "candidate_lane_ids": [],
            "candidate_lane_lateral_right_m": [],
            "right_lane_calibration_source": None,
            "task_stop_side_lateral_m": None,
            "right_lane_lateral_bias_enabled": self.right_lane_lateral_bias_enabled,
            "right_lane_lateral_bias_requested_m": self.right_lane_lateral_bias_m,
            "right_lane_lateral_bias_applied_m": 0.0,
            "right_lane_lateral_bias_reason": "not_evaluated",
            "lane_width_m": None,
            "vehicle_half_width_m": self.vehicle_half_width_m,
            "preferred_lane_side": "right",
            "biased_target_x": None,
            "biased_target_y": None,
            "raw_target_x": None,
            "raw_target_y": None,
            "cte_used": False,
            "cte_source": None,
            "route_heading_error_deg": None,
            "route_lateral_error_m": None,
            "centerline_lateral_error_m": None,
            "biased_lateral_error_m": None,
            "route_recovery_allowed": True,
            "route_recovery_active": False,
            "route_target_recovery_active": False,
            "lane_departure_risk": False,
            "lane_departure_speed_clamp_applied": False,
            "min_nonzero_speed_floor_applied": False,
            "min_nonzero_target_speed_mps": self.min_nonzero_target_speed_mps,
            "min_speed_floor_applied": False,
            "min_speed_floor_reason": None,
            "nominal_min_speed_mps": self.nominal_min_speed_mps,
            "junction_recovery_min_speed_mps": self.junction_recovery_min_speed_mps,
            "junction_exit_min_speed_mps": self.junction_exit_min_speed_mps,
            "route_conflict_with_lane_detection": False,
            "target_speed_raw_mps": None,
            "target_speed_final_mps": None,
            "target_speed_final": None,
            "zero_speed_reason": None,
            "junction_locked": False,
            "junction_exit_recovery_active": False,
            "selected_route_index": None,
            "lookahead_distance_m": self.base_lookahead_m,
            "steering_limited": False,
            "steering_rate_limited": False,
            "mission_stop_active": False,
            "mission_stop_reason": None,
            "task_pull_over_mode": False,
            "task_stop_x": None,
            "task_stop_y": None,
            "task_stop_yaw": None,
            "pre_stop_x": None,
            "pre_stop_y": None,
            "pre_stop_yaw": None,
            "task_pose_phase": "route_lane",
            "task_pull_over_target_x": None,
            "task_pull_over_target_y": None,
            "task_pull_over_blended_target_x": None,
            "task_pull_over_blended_target_y": None,
            "task_pull_over_lateral_offset_m": self.task_pull_over_lateral_offset_m,
            "task_pull_over_keep_bias_until_reached": self.task_pull_over_keep_bias_until_reached,
            "task_stop_distance_m": None,
            "task_stop_distance_source": None,
            "task_stop_reached": False,
            "task_stop_reached_by_mission": False,
            "task_stop_reached_reason": None,
            "task_stop_yaw_tolerance_deg": None,
            "task_stop_completion_yaw_tolerance_deg": None,
            "task_stop_completion_position_tolerance_m": None,
            "task_stop_completion_yaw_ok": None,
            "task_stop_completion_position_ok": False,
            "task_stop_close_enough_distance_m": None,
            "task_stop_close_enough_reached": False,
            "yaw_error_deg": None,
            "task_stop_yaw_error_deg": None,
            "task_stop_yaw_within_tolerance": None,
            "final_task_stop_latch_enabled": self.task_stop_final_phase_latch_enabled,
            "final_task_stop_latch_active": False,
            "final_task_stop_latch_reason": None,
            "task_stop_final_latch_distance_m": self.task_stop_final_latch_distance_m,
            "task_stop_overshoot_guard_distance_m": self.task_stop_overshoot_guard_distance_m,
            "task_stop_overshoot_guard_speed_mps": self.task_stop_overshoot_guard_speed_mps,
            "task_stop_overshoot_guard_active": False,
            "task_stop_alignment_active": False,
            "task_stop_alignment_reason": None,
            "task_stop_alignment_target_x": None,
            "task_stop_alignment_target_y": None,
            "task_stop_alignment_enabled": self.task_stop_alignment_enabled,
            "task_stop_alignment_start_distance_m": self.task_stop_alignment_start_distance_m,
            "task_stop_alignment_yaw_tolerance_deg": self.task_stop_alignment_yaw_tolerance_deg,
            "task_stop_alignment_speed_mps": self.task_stop_alignment_speed_mps,
            "task_stop_alignment_target_ahead_m": self.task_stop_alignment_target_ahead_m,
            "task_stop_approach_speed_phase": None,
            "task_stop_approach_speed_mps": None,
            "task_stop_approach_cruise_speed_mps": self.task_stop_approach_cruise_speed_mps,
            "task_stop_pre_align_speed_mps": self.task_stop_pre_align_speed_mps,
            "task_stop_final_align_speed_mps": self.task_stop_final_align_speed_mps,
            "task_stop_min_creep_speed_mps": self.task_stop_min_creep_speed_mps,
            "task_stop_no_stop_before_final_distance_m": self.task_stop_no_stop_before_final_distance_m,
            "task_stop_phase_hysteresis_m": self.task_stop_phase_hysteresis_m,
            "task_stop_phase_hysteresis_active": False,
            "task_stop_safety_hold_active": False,
            "task_stop_safety_hold_reason": None,
            "task_hold_active": False,
            "task_hold_remaining_s": None,
            "base_goal_x": None,
            "base_goal_y": None,
            "base_goal_yaw": None,
            "raw_task_stop_x": None,
            "raw_task_stop_y": None,
            "effective_task_stop_x": None,
            "effective_task_stop_y": None,
            "effective_task_stop_yaw": None,
            "effective_task_stop_source": None,
            "task_stop_projection_enabled": False,
            "task_stop_projection_reason": None,
            "task_stop_projection_lateral_m": None,
            "task_stop_projection_forward_m": None,
            "task_stop_on_road": None,
            "task_stop_road_id": None,
            "task_stop_lane_id": None,
            "distance_to_road_edge_m": None,
            "task_pose_approach_start_distance_m": self.task_pose_approach_start_distance_m,
            "task_pose_pre_stop_distance_m": self.task_pose_pre_stop_distance_m,
        }

    def _route_tracking_metrics(
        self,
        points: list[dict[str, Any]],
        nearest_index: int,
        ego_x: float,
        ego_y: float,
        ego_yaw_deg: float,
        now: float,
    ) -> dict[str, Any]:
        debug = self._route_only_debug_defaults()
        debug["cte_used"] = True
        debug["cte_source"] = "route_points"
        debug["selected_route_index"] = nearest_index

        if not points or nearest_index < 0 or nearest_index >= len(points):
            return debug

        nearest = points[nearest_index]
        route_yaw_deg = float(nearest.get("yaw", ego_yaw_deg))
        route_yaw_rad = math.radians(route_yaw_deg)
        dx = ego_x - float(nearest.get("x", ego_x))
        dy = ego_y - float(nearest.get("y", ego_y))
        lateral_error_m = -math.sin(route_yaw_rad) * dx + math.cos(route_yaw_rad) * dy
        heading_error_deg = signed_angle_diff_deg(ego_yaw_deg, route_yaw_deg)
        lane_cte_fresh = (now - self.last_lane_cte_time) < 0.5 and self.last_lane_cte is not None
        route_conflict = False
        if lane_cte_fresh:
            normalized_route_cte = lateral_error_m / max(0.1, self.lane_departure_lateral_threshold_m)
            route_conflict = (
                abs(heading_error_deg) > self.route_conflict_heading_threshold_deg
                or (
                    abs(normalized_route_cte) > 0.4
                    and abs(float(self.last_lane_cte)) > 0.4
                    and normalized_route_cte * float(self.last_lane_cte) < 0.0
                )
            )

        lane_departure_risk = (
            self.lane_departure_guard_enabled
            and abs(lateral_error_m) >= self.lane_departure_lateral_threshold_m
        )
        debug.update(
            {
                "route_heading_error_deg": round(float(heading_error_deg), 3),
                "route_lateral_error_m": round(float(lateral_error_m), 3),
                "centerline_lateral_error_m": round(float(lateral_error_m), 3),
                "route_recovery_allowed": True,
                "route_recovery_active": lane_departure_risk,
                "lane_departure_risk": lane_departure_risk,
                "route_conflict_with_lane_detection": route_conflict,
            }
        )
        return debug

    def _bias_target_right(
        self,
        target: dict[str, Any],
        route_event_name: str,
        mission_stop_active: bool,
        speed_mps: float,
        bias_disabled_reason: Optional[str] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw_x = float(target.get("x", 0.0))
        raw_y = float(target.get("y", 0.0))
        lane_width = target.get("lane_width_m", target.get("lane_width"))
        debug = {
            "right_lane_lateral_bias_enabled": self.right_lane_lateral_bias_enabled,
            "right_lane_lateral_bias_requested_m": self.right_lane_lateral_bias_m,
            "right_lane_lateral_bias_applied_m": 0.0,
            "right_lane_lateral_bias_reason": None,
            "lane_width_m": None,
            "vehicle_half_width_m": self.vehicle_half_width_m,
            "preferred_lane_side": "right",
            "raw_target_x": round(raw_x, 3),
            "raw_target_y": round(raw_y, 3),
            "biased_target_x": round(raw_x, 3),
            "biased_target_y": round(raw_y, 3),
        }

        if not self.right_lane_lateral_bias_enabled:
            debug["right_lane_lateral_bias_reason"] = "disabled"
            return target, debug
        if bias_disabled_reason:
            debug["right_lane_lateral_bias_reason"] = bias_disabled_reason
            return target, debug
        if mission_stop_active:
            debug["right_lane_lateral_bias_reason"] = "mission_stop_active"
            return target, debug
        if speed_mps < self.right_lane_lateral_bias_min_speed_mps:
            debug["right_lane_lateral_bias_reason"] = "below_min_speed"
            return target, debug
        if self.right_lane_lateral_bias_disable_in_junction and bool(target.get("is_junction", False)):
            debug["right_lane_lateral_bias_reason"] = "junction_disabled"
            return target, debug
        turn_direction = str(target.get("turn_direction", "unknown"))
        if (
            self.right_lane_lateral_bias_disable_in_turn
            and turn_direction in ("left", "right", "u_turn")
        ):
            debug["right_lane_lateral_bias_reason"] = f"turn_disabled:{turn_direction}"
            return target, debug
        if route_event_name in (
            "traffic_light_red_approach",
            "traffic_light_red_stop",
            "traffic_light_yellow_slow",
            "traffic_light_yellow_stop",
        ):
            debug["right_lane_lateral_bias_reason"] = "traffic_light_stopline_protected"
            return target, debug
        try:
            lane_width_m = float(lane_width)
        except Exception:
            lane_width_m = 0.0
        debug["lane_width_m"] = round(lane_width_m, 3) if lane_width_m > 0.0 else None
        if lane_width_m <= 0.0:
            debug["right_lane_lateral_bias_reason"] = "lane_width_unavailable"
            return target, debug

        max_bias = (
            lane_width_m / 2.0
            - self.vehicle_half_width_m
            - self.right_lane_lateral_bias_safety_margin_m
        )
        applied_bias = clamp(self.right_lane_lateral_bias_m, 0.0, max(0.0, max_bias))
        if applied_bias <= 1e-6:
            debug["right_lane_lateral_bias_reason"] = "safety_margin_no_room"
            return target, debug

        yaw_rad = math.radians(float(target.get("yaw", 0.0)))
        biased = dict(target)
        biased_x = raw_x + math.sin(yaw_rad) * applied_bias
        biased_y = raw_y - math.cos(yaw_rad) * applied_bias
        biased["x"] = biased_x
        biased["y"] = biased_y
        biased["raw_x"] = raw_x
        biased["raw_y"] = raw_y
        biased["right_lane_lateral_bias_applied_m"] = applied_bias
        debug.update(
            {
                "right_lane_lateral_bias_applied_m": round(applied_bias, 3),
                "right_lane_lateral_bias_reason": "applied",
                "biased_target_x": round(biased_x, 3),
                "biased_target_y": round(biased_y, 3),
            }
        )
        return biased, debug

    def _active_route_event(self, now: float) -> tuple[Optional[dict[str, Any]], Optional[float]]:
        if self.route_event is None or self.last_route_event_time <= 0.0:
            return None, None

        age_s = now - self.last_route_event_time
        if age_s > self.route_event_timeout_s:
            return None, age_s
        if not self.route_event.get("ok", False):
            return None, age_s
        return self.route_event, age_s

    def _apply_route_event_speed_limit(
        self,
        target_speed_mps: float,
        speed_reason: str,
        route_event: Optional[dict[str, Any]],
    ) -> tuple[float, str]:
        if route_event is None:
            return target_speed_mps, speed_reason

        event = str(route_event.get("event", "clear"))
        speed_limit = route_event.get("target_speed_limit_mps")
        if event in (
            "clear",
            "traffic_light_green_clear",
            "traffic_light_green_release",
        ):
            return target_speed_mps, f"{speed_reason}+{event}"

        if event in (
            "vehicle_follow",
            "traffic_light_red_approach",
            "traffic_light_yellow_slow",
        ) and speed_limit is not None:
            return (
                min(target_speed_mps, max(0.0, float(speed_limit))),
                f"{speed_reason}+{event}",
            )

        if event not in (
            "vehicle_stop",
            "pedestrian_stop",
            "traffic_light_red_stop",
            "traffic_light_yellow_stop",
        ):
            return target_speed_mps, speed_reason

        if (
            event in ("traffic_light_red_stop", "traffic_light_yellow_stop")
            and bool(route_event.get("stop_required", False))
        ):
            return 0.0, f"{speed_reason}+{event}"

        distance_m = max(0.0, float(route_event.get("distance_m") or 0.0))
        distance_is_buffered = bool(route_event.get("distance_is_buffered", False))
        if distance_is_buffered:
            remaining_distance_m = distance_m
            stop_threshold_m = 0.5
        else:
            stop_buffer_m = max(
                0.0,
                float(route_event.get("stop_buffer_m") or self.event_stop_buffer_m),
            )
            remaining_distance_m = max(0.0, distance_m - stop_buffer_m)
            stop_threshold_m = stop_buffer_m
        approach_speed_mps = math.sqrt(
            2.0
            * max(0.1, self.event_stop_deceleration_mps2)
            * remaining_distance_m
        )
        if distance_m <= stop_threshold_m:
            approach_speed_mps = 0.0

        return (
            min(target_speed_mps, approach_speed_mps),
            f"{speed_reason}+{event}",
        )

    @staticmethod
    def _red_approach_speed_cap(distance_to_stop_m: Optional[float]) -> Optional[float]:
        if distance_to_stop_m is None:
            return None
        distance = max(0.0, float(distance_to_stop_m))
        if distance <= 4.0:
            return 0.0
        if distance <= 8.0:
            return 0.8
        if distance <= 15.0:
            return 1.2
        if distance <= 25.0:
            return 1.8
        if distance <= 35.0:
            return 2.5
        return 3.0

    @staticmethod
    def _event_zero_speed_reason(
        route_event_name: str,
        route_event: Optional[dict[str, Any]],
    ) -> Optional[str]:
        if route_event is None:
            return None

        if route_event_name in ("emergency_stop", "collision_stop"):
            return route_event_name

        if route_event_name in (
            "traffic_light_red_stop",
            "traffic_light_red_approach",
            "traffic_light_yellow_stop",
            "traffic_light_yellow_slow",
        ):
            speed_limit = route_event.get("target_speed_limit_mps")
            if bool(route_event.get("stop_required", False)):
                return route_event_name
            if speed_limit is not None:
                try:
                    if float(speed_limit) <= 0.0:
                        return route_event_name
                except Exception:
                    pass

        if route_event_name in ("vehicle_stop", "pedestrian_stop"):
            return route_event_name

        return None

    def _minimum_drive_speed(
        self,
        speed_context: str,
        junction_locked: bool,
        route_recovery_active: bool,
    ) -> float:
        if junction_locked or route_recovery_active:
            return min(self.max_speed_mps, max(0.0, self.junction_exit_min_speed_mps))
        if speed_context == "nominal":
            return min(self.max_speed_mps, max(0.0, self.nominal_min_speed_mps))
        return 0.0

    @staticmethod
    def _minimum_drive_speed_reason(
        speed_context: str,
        junction_exit_recovery_active: bool,
        route_recovery_active: bool,
    ) -> Optional[str]:
        if junction_exit_recovery_active:
            return "junction_exit_min_speed"
        if route_recovery_active:
            return "route_recovery_min_speed"
        if speed_context == "nominal":
            return "nominal_min_speed"
        return None

    def _find_nearest_index(self, points: list[dict[str, Any]], ego_x: float, ego_y: float) -> int:
        nearest_index = 0
        nearest_dist = float("inf")
        for index, pt in enumerate(points):
            dx = float(pt.get("x", 0.0)) - ego_x
            dy = float(pt.get("y", 0.0)) - ego_y
            dist = math.hypot(dx, dy)
            if dist < nearest_dist:
                nearest_dist = dist
                nearest_index = index
        return nearest_index

    def _nearest_index_with_hysteresis(
        self,
        points: list[dict[str, Any]],
        ego_x: float,
        ego_y: float,
    ) -> int:
        raw_index = self._find_nearest_index(points, ego_x, ego_y)
        if (
            not self.route_index_hysteresis_enabled
            or self.last_nearest_index is None
            or self.max_route_index_jump <= 0
        ):
            self.last_nearest_index = raw_index
            return raw_index

        min_allowed = max(0, self.last_nearest_index - self.max_route_index_jump)
        max_allowed = min(len(points) - 1, self.last_nearest_index + self.max_route_index_jump)
        nearest_index = max(min_allowed, min(max_allowed, raw_index))
        self.last_nearest_index = nearest_index
        return nearest_index

    def _active_mission_stop(self, now: float) -> tuple[bool, Optional[str]]:
        if self.mission_goal is None or now - self.last_mission_goal_time > 2.0:
            return False, None
        active = bool(self.mission_goal.get("mission_stop_active", False))
        reason = self.mission_goal.get("mission_stop_reason")
        return active, str(reason) if reason is not None else None

    def _task_pull_over_target(
        self,
        raw_target: dict[str, Any],
        distance_to_goal_m: Optional[float],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        debug = {
            "task_pull_over_mode": False,
            "task_stop_x": None,
            "task_stop_y": None,
            "task_stop_yaw": None,
            "pre_stop_x": None,
            "pre_stop_y": None,
            "pre_stop_yaw": None,
            "task_pose_phase": "route_lane",
            "task_pull_over_target_x": None,
            "task_pull_over_target_y": None,
            "task_pull_over_blended_target_x": None,
            "task_pull_over_blended_target_y": None,
            "task_pull_over_lateral_offset_m": self.task_pull_over_lateral_offset_m,
            "task_pull_over_keep_bias_until_reached": self.task_pull_over_keep_bias_until_reached,
            "task_stop_distance_m": distance_to_goal_m,
            "task_stop_distance_source": "route_distance_to_goal",
            "task_stop_reached": False,
            "task_stop_reached_by_mission": False,
            "task_stop_reached_reason": None,
            "task_stop_yaw_tolerance_deg": None,
            "task_stop_completion_yaw_tolerance_deg": None,
            "task_stop_completion_position_tolerance_m": None,
            "task_stop_completion_yaw_ok": None,
            "task_stop_completion_position_ok": False,
            "task_stop_close_enough_distance_m": None,
            "task_stop_close_enough_reached": False,
            "yaw_error_deg": None,
            "task_stop_yaw_error_deg": None,
            "task_stop_yaw_within_tolerance": None,
            "final_task_stop_latch_enabled": self.task_stop_final_phase_latch_enabled,
            "final_task_stop_latch_active": False,
            "final_task_stop_latch_reason": None,
            "task_stop_final_latch_distance_m": self.task_stop_final_latch_distance_m,
            "task_stop_overshoot_guard_distance_m": self.task_stop_overshoot_guard_distance_m,
            "task_stop_overshoot_guard_speed_mps": self.task_stop_overshoot_guard_speed_mps,
            "task_stop_overshoot_guard_active": False,
            "task_stop_alignment_active": False,
            "task_stop_alignment_reason": None,
            "task_stop_alignment_target_x": None,
            "task_stop_alignment_target_y": None,
            "task_stop_alignment_enabled": self.task_stop_alignment_enabled,
            "task_stop_alignment_start_distance_m": self.task_stop_alignment_start_distance_m,
            "task_stop_alignment_yaw_tolerance_deg": self.task_stop_alignment_yaw_tolerance_deg,
            "task_stop_alignment_speed_mps": self.task_stop_alignment_speed_mps,
            "task_stop_alignment_target_ahead_m": self.task_stop_alignment_target_ahead_m,
            "task_stop_approach_speed_phase": None,
            "task_stop_approach_speed_mps": None,
            "task_stop_approach_cruise_speed_mps": self.task_stop_approach_cruise_speed_mps,
            "task_stop_pre_align_speed_mps": self.task_stop_pre_align_speed_mps,
            "task_stop_final_align_speed_mps": self.task_stop_final_align_speed_mps,
            "task_stop_min_creep_speed_mps": self.task_stop_min_creep_speed_mps,
            "task_stop_no_stop_before_final_distance_m": self.task_stop_no_stop_before_final_distance_m,
            "task_stop_phase_hysteresis_m": self.task_stop_phase_hysteresis_m,
            "task_stop_phase_hysteresis_active": False,
            "task_stop_safety_hold_active": False,
            "task_stop_safety_hold_reason": None,
            "task_hold_active": False,
            "task_hold_remaining_s": None,
            "base_goal_x": None,
            "base_goal_y": None,
            "base_goal_yaw": None,
            "raw_task_stop_x": None,
            "raw_task_stop_y": None,
            "effective_task_stop_x": None,
            "effective_task_stop_y": None,
            "effective_task_stop_yaw": None,
            "effective_task_stop_source": None,
            "task_stop_projection_enabled": False,
            "task_stop_projection_reason": None,
            "task_stop_projection_lateral_m": None,
            "task_stop_projection_forward_m": None,
            "task_stop_on_road": None,
            "task_stop_road_id": None,
            "task_stop_lane_id": None,
            "distance_to_road_edge_m": None,
            "task_pose_approach_start_distance_m": self.task_pose_approach_start_distance_m,
            "task_pose_pre_stop_distance_m": self.task_pose_pre_stop_distance_m,
        }
        if self.mission_goal is None or time.time() - self.last_mission_goal_time > 2.0:
            self._reset_final_task_stop_latch()
            return raw_target, debug
        goal_kind = str(self.mission_goal.get("goal_kind", ""))
        task_required = bool(self.mission_goal.get("task_stop_required", False))
        mission_hold_active = bool(
            self.mission_goal.get(
                "mission_hold_active",
                self.mission_goal.get("mission_stop_active", False),
            )
        )
        task_reached_by_mission = bool(
            self.mission_goal.get("task_stop_reached_by_mission", False)
        )
        task_stop_yaw_error = self.mission_goal.get("task_stop_yaw_error_deg")
        task_stop_yaw_within_tolerance = self.mission_goal.get(
            "task_stop_yaw_within_tolerance"
        )
        hold_active = mission_hold_active and task_reached_by_mission
        debug["task_hold_active"] = hold_active
        debug["task_hold_remaining_s"] = self.mission_goal.get("mission_hold_remaining_s")
        debug["yaw_error_deg"] = task_stop_yaw_error
        debug["task_stop_yaw_error_deg"] = task_stop_yaw_error
        debug["task_stop_yaw_within_tolerance"] = task_stop_yaw_within_tolerance
        debug["task_stop_yaw_tolerance_deg"] = self.mission_goal.get(
            "task_stop_yaw_tolerance_deg"
        )
        debug["task_stop_completion_yaw_tolerance_deg"] = self.mission_goal.get(
            "task_stop_completion_yaw_tolerance_deg"
        )
        debug["task_stop_completion_position_tolerance_m"] = self.mission_goal.get(
            "task_stop_completion_position_tolerance_m"
        )
        debug["task_stop_completion_yaw_ok"] = self.mission_goal.get(
            "task_stop_completion_yaw_ok"
        )
        debug["task_stop_completion_position_ok"] = bool(
            self.mission_goal.get("task_stop_completion_position_ok", False)
        )
        debug["task_stop_close_enough_distance_m"] = self.mission_goal.get(
            "task_stop_close_enough_distance_m"
        )
        debug["task_stop_close_enough_reached"] = bool(
            self.mission_goal.get("task_stop_close_enough_reached", False)
        )
        debug["task_stop_safety_hold_active"] = bool(
            self.mission_goal.get("task_stop_safety_hold_active", False)
        )
        debug["task_stop_safety_hold_reason"] = self.mission_goal.get(
            "task_stop_safety_hold_reason"
        )
        debug["task_stop_reached_reason"] = self.mission_goal.get(
            "task_stop_reached_reason"
        )
        if goal_kind not in ("pickup", "dropoff") or not task_required:
            self._reset_final_task_stop_latch()
            return raw_target, debug

        latch_key = self._mission_goal_latch_key()
        if latch_key != self._final_task_stop_latch_key:
            self._reset_final_task_stop_latch()
            self._final_task_stop_latch_key = latch_key

        base_goal_x = self.mission_goal.get("base_goal_x", self.mission_goal.get("target_x"))
        base_goal_y = self.mission_goal.get("base_goal_y", self.mission_goal.get("target_y"))
        base_goal_yaw = self.mission_goal.get("base_goal_yaw", self.mission_goal.get("target_yaw"))
        raw_task_stop_x = self.mission_goal.get("raw_task_stop_x")
        raw_task_stop_y = self.mission_goal.get("raw_task_stop_y")
        effective_x = self.mission_goal.get("effective_task_stop_x")
        effective_y = self.mission_goal.get("effective_task_stop_y")
        effective_yaw = self.mission_goal.get("effective_task_stop_yaw")
        effective_source = self.mission_goal.get("effective_task_stop_source")
        task_stop_x = effective_x if effective_x is not None else self.mission_goal.get("task_stop_x")
        task_stop_y = effective_y if effective_y is not None else self.mission_goal.get("task_stop_y")
        task_stop_yaw = effective_yaw if effective_yaw is not None else self.mission_goal.get("task_stop_yaw")
        debug.update(
            {
                "task_stop_x": task_stop_x,
                "task_stop_y": task_stop_y,
                "task_stop_yaw": task_stop_yaw,
                "base_goal_x": base_goal_x,
                "base_goal_y": base_goal_y,
                "base_goal_yaw": base_goal_yaw,
                "raw_task_stop_x": raw_task_stop_x,
                "raw_task_stop_y": raw_task_stop_y,
                "effective_task_stop_x": effective_x,
                "effective_task_stop_y": effective_y,
                "effective_task_stop_yaw": effective_yaw,
                "effective_task_stop_source": effective_source,
                "task_stop_projection_enabled": bool(
                    self.mission_goal.get("task_stop_projection_enabled", False)
                ),
                "task_stop_projection_reason": self.mission_goal.get(
                    "task_stop_projection_reason"
                ),
                "task_stop_projection_lateral_m": self.mission_goal.get(
                    "task_stop_projection_lateral_m"
                ),
                "task_stop_projection_forward_m": self.mission_goal.get(
                    "task_stop_projection_forward_m"
                ),
                "task_stop_side": self.mission_goal.get("task_stop_side"),
                "task_stop_side_lateral_m": self.mission_goal.get(
                    "task_stop_side_lateral_m"
                ),
                "task_stop_on_road": self.mission_goal.get("task_stop_on_road"),
                "task_stop_road_id": self.mission_goal.get("task_stop_road_id"),
                "task_stop_lane_id": self.mission_goal.get("task_stop_lane_id"),
                "distance_to_road_edge_m": self.mission_goal.get(
                    "distance_to_road_edge_m"
                ),
                "task_stop_reached_by_mission": task_reached_by_mission,
            }
        )
        if task_stop_x is None or task_stop_y is None:
            self._reset_final_task_stop_latch()
            return raw_target, debug
        effective_center_distance = self.mission_goal.get("center_distance_to_effective_task_stop_m")
        effective_front_distance = self.mission_goal.get("front_bumper_distance_to_effective_task_stop_m")
        effective_distances = [
            float(d)
            for d in (effective_center_distance, effective_front_distance)
            if d is not None
        ]
        if effective_distances:
            distance_to_goal_m = min(effective_distances)
            debug["task_stop_distance_source"] = "mission_effective_task_stop"
        elif distance_to_goal_m is None:
            distance_to_goal_m = self.mission_goal.get("distance_to_goal_m")
            debug["task_stop_distance_source"] = "mission_distance_to_goal"
        if distance_to_goal_m is None:
            return raw_target, debug
        distance_to_goal_m = float(distance_to_goal_m)
        debug["task_stop_distance_m"] = round(distance_to_goal_m, 3)
        if self.mission_goal.get("task_stop_reached_by_mission") is not None:
            debug["task_stop_reached"] = task_reached_by_mission
        else:
            debug["task_stop_reached"] = distance_to_goal_m <= self.task_stop_reached_distance_m

        pull_x = float(task_stop_x)
        pull_y = float(task_stop_y)
        pre_stop_x = pull_x
        pre_stop_y = pull_y
        pre_stop_yaw = task_stop_yaw
        forward_x = None
        forward_y = None
        if task_stop_yaw is not None:
            yaw_rad = math.radians(float(task_stop_yaw))
            forward_x = math.cos(yaw_rad)
            forward_y = math.sin(yaw_rad)
            pre_stop_x = pull_x - forward_x * self.task_pose_pre_stop_distance_m
            pre_stop_y = pull_y - forward_y * self.task_pose_pre_stop_distance_m
        debug.update(
            {
                "pre_stop_x": round(float(pre_stop_x), 3),
                "pre_stop_y": round(float(pre_stop_y), 3),
                "pre_stop_yaw": pre_stop_yaw,
            }
        )

        if distance_to_goal_m > self.task_pose_approach_start_distance_m and not hold_active:
            return raw_target, debug

        if self.task_stop_final_phase_latch_enabled and not hold_active:
            should_enter_final_phase = distance_to_goal_m <= self.task_pull_over_final_distance_m
            should_lock_near_stop = distance_to_goal_m <= self.task_stop_final_latch_distance_m
            if should_enter_final_phase or self._final_task_stop_latch_active:
                self._final_task_stop_latch_active = True
                self._final_task_stop_latch_key = latch_key
                if should_lock_near_stop:
                    self._final_task_stop_latch_reason = "within_final_latch_distance"
                elif self._final_task_stop_latch_reason is None:
                    self._final_task_stop_latch_reason = "final_phase_entered"

        yaw_error_for_alignment = None
        if task_stop_yaw_error is not None:
            try:
                yaw_error_for_alignment = float(task_stop_yaw_error)
            except Exception:
                yaw_error_for_alignment = None
        completion_yaw_ok = debug.get("task_stop_completion_yaw_ok")
        completion_position_ok = bool(debug.get("task_stop_completion_position_ok", False))
        local_safety_hold_active = (
            not hold_active
            and not task_reached_by_mission
            and distance_to_goal_m <= self.task_stop_overshoot_guard_distance_m
            and (
                completion_yaw_ok is False
                or not completion_position_ok
            )
        )
        if local_safety_hold_active:
            debug["task_stop_safety_hold_active"] = True
            debug["task_stop_safety_hold_reason"] = (
                "safety_hold_yaw_not_aligned"
                if completion_yaw_ok is False
                else "safety_hold_position_not_complete"
            )
        alignment_active = (
            self.task_stop_alignment_enabled
            and not hold_active
            and not local_safety_hold_active
            and forward_x is not None
            and forward_y is not None
            and yaw_error_for_alignment is not None
            and distance_to_goal_m <= self.task_stop_alignment_start_distance_m
            and yaw_error_for_alignment > self.task_stop_alignment_yaw_tolerance_deg
        )
        alignment_target_x = None
        alignment_target_y = None
        if alignment_active:
            alignment_target_x = pull_x + forward_x * self.task_stop_alignment_target_ahead_m
            alignment_target_y = pull_y + forward_y * self.task_stop_alignment_target_ahead_m

        if hold_active:
            phase = f"{goal_kind}_hold"
            target_x = pull_x
            target_y = pull_y
        elif alignment_active:
            phase = "final_task_stop"
            target_x = alignment_target_x
            target_y = alignment_target_y
        elif self._final_task_stop_latch_active:
            phase = "final_task_stop"
            target_x = pull_x
            target_y = pull_y
        elif distance_to_goal_m > self.task_pull_over_final_distance_m and task_stop_yaw is not None:
            phase = "pre_stop_align"
            target_x = pre_stop_x
            target_y = pre_stop_y
        else:
            phase = "final_task_stop"
            target_x = pull_x
            target_y = pull_y

        target = dict(raw_target)
        target["x"] = float(target_x)
        target["y"] = float(target_y)
        if task_stop_yaw is not None:
            target["yaw"] = float(task_stop_yaw)
        overshoot_guard_active = (
            self._final_task_stop_latch_active
            and distance_to_goal_m <= self.task_stop_overshoot_guard_distance_m
        )
        if local_safety_hold_active:
            overshoot_guard_active = True
        final_latch_reason = (
            "mission_hold_active"
            if hold_active
            else self._final_task_stop_latch_reason
        )
        if local_safety_hold_active:
            final_latch_reason = debug.get("task_stop_safety_hold_reason")
        elif overshoot_guard_active:
            final_latch_reason = "final_stop_latch_hold"
        debug.update(
            {
                "task_pull_over_mode": True,
                "task_pose_phase": phase,
                "task_pull_over_target_x": round(float(pull_x), 3),
                "task_pull_over_target_y": round(float(pull_y), 3),
                "task_pull_over_blended_target_x": round(float(target["x"]), 3),
                "task_pull_over_blended_target_y": round(float(target["y"]), 3),
                "final_task_stop_latch_enabled": self.task_stop_final_phase_latch_enabled,
                "final_task_stop_latch_active": self._final_task_stop_latch_active,
                "final_task_stop_latch_reason": final_latch_reason,
                "task_stop_final_latch_distance_m": self.task_stop_final_latch_distance_m,
                "task_stop_overshoot_guard_distance_m": self.task_stop_overshoot_guard_distance_m,
                "task_stop_overshoot_guard_speed_mps": self.task_stop_overshoot_guard_speed_mps,
                "task_stop_overshoot_guard_active": overshoot_guard_active,
                "task_stop_alignment_active": alignment_active,
                "task_stop_alignment_reason": (
                    "yaw_error_above_alignment_tolerance"
                    if alignment_active
                    else None
                ),
                "task_stop_alignment_target_x": (
                    round(float(alignment_target_x), 3)
                    if alignment_target_x is not None
                    else None
                ),
                "task_stop_alignment_target_y": (
                    round(float(alignment_target_y), 3)
                    if alignment_target_y is not None
                    else None
                ),
                "task_stop_phase_hysteresis_active": self._final_task_stop_latch_active,
            }
        )
        return target, debug

    def _run(self):
        now = time.time()
        status_ok = (now - self.last_status_time) < 1.0
        route_ok = (self.route is not None) and (now - self.last_route_time) < 2.0
        active_route_event, route_event_age_s = self._active_route_event(now)
        route_event_name = (
            str(active_route_event.get("event", "clear"))
            if active_route_event is not None
            else "clear"
        )
        route_event_distance_m = (
            active_route_event.get("distance_m")
            if active_route_event is not None
            else None
        )
        route_event_distance_to_stop_m = (
            active_route_event.get(
                "distance_to_stop_m",
                route_event_distance_m,
            )
            if active_route_event is not None
            else None
        )
        route_event_front_bumper_to_stopline_m = (
            active_route_event.get("front_bumper_to_stopline_m")
            if active_route_event is not None
            else None
        )
        traffic_light_distance_m = (
            route_event_front_bumper_to_stopline_m
            if route_event_front_bumper_to_stopline_m is not None
            else route_event_distance_to_stop_m
        )
        route_event_speed_limit_mps = (
            active_route_event.get("target_speed_limit_mps")
            if active_route_event is not None
            else None
        )
        red_stop_trigger_m = (
            active_route_event.get("red_stop_trigger_m")
            if active_route_event is not None
            else None
        )
        red_stop_triggered_by_distance = bool(
            active_route_event.get("red_stop_triggered_by_distance", False)
            if active_route_event is not None
            else False
        )
        route_event_traffic_light_state = (
            active_route_event.get("traffic_light_state")
            if active_route_event is not None
            else None
        )
        route_event_reason = (
            active_route_event.get("reason")
            if active_route_event is not None
            else None
        )
        route_event_stop_required = bool(
            active_route_event.get("stop_required", False)
            if active_route_event is not None
            else False
        )
        route_event_stop_point_source = (
            active_route_event.get("stop_point_source")
            if active_route_event is not None
            else None
        )
        route_event_confidence = (
            active_route_event.get("confidence")
            if active_route_event is not None
            else None
        )
        tl_stop_route_index = (
            active_route_event.get(
                "stop_route_index",
                active_route_event.get("route_index"),
            )
            if active_route_event is not None
            else None
        )
        tl_stop_s = (
            active_route_event.get("stop_s")
            if active_route_event is not None
            else None
        )

        route_source = self.route.get("route_source", "unknown") if self.route else "unknown"
        route_point_count = len(self.route.get("points", [])) if self.route else 0
        route_payload_ok = bool(self.route.get("route_ok", False)) if self.route else False
        route_distance_to_goal_m = self.route.get("distance_to_goal_m") if self.route else None
        route_goal_near_distance_m = (
            self.route.get("mission_goal_near_distance_m", 3.0)
            if self.route
            else 3.0
        )
        route_target_recovery_active = False
        if route_source == "route_end_near_goal" and route_point_count > 0:
            try:
                route_target_recovery_active = (
                    route_distance_to_goal_m is not None
                    and float(route_distance_to_goal_m) > float(route_goal_near_distance_m)
                )
            except Exception:
                route_target_recovery_active = False
        route_source_allows_drive = route_source == "global_route" or route_target_recovery_active
        if route_target_recovery_active:
            route_payload_ok = True
        route_ok = route_ok and route_payload_ok and route_source_allows_drive
        mission_stop_active, mission_stop_reason = self._active_mission_stop(now)
        if not status_ok or not route_ok:
            route_only_debug = self._route_only_debug_defaults()
            route_only_debug["mission_stop_active"] = mission_stop_active
            route_only_debug["mission_stop_reason"] = mission_stop_reason
            route_only_debug["route_target_recovery_active"] = route_target_recovery_active
            route_only_debug["target_speed_raw_mps"] = 0.0
            route_only_debug["target_speed_final_mps"] = 0.0
            route_only_debug["zero_speed_reason"] = f"route_invalid:{route_source}"
            plan = {
                "stamp": now,
                "source": "phase2b_pure_pursuit",
                "route_source": route_source,
                "distance_to_goal_m": route_distance_to_goal_m,
                "mission_goal_near_distance_m": route_goal_near_distance_m,
                "cruise_speed_mps": self.cruise_speed_mps,
                "target_speed_mps": 0.0,
                "turn_intensity": 0.0,
                "speed_reason": f"route_invalid:{route_source}",
                "speed_boost_enabled": self.speed_boost_enabled,
                "speed_boost_mps": self.nominal_speed_boost_mps,
                "speed_boost_applied": False,
                "boost_applied": False,
                "speed_context": "route_invalid",
                "pre_boost_speed_mps": 0.0,
                "post_boost_speed_mps": 0.0,
                "speed_limit_clamped": False,
                "clamp_reason": "route_invalid",
                "turn_speed_protected": True,
                "nearest_index": None,
                "selected_target_index": None,
                "lookahead_m": self.base_lookahead_m,
                "steer": 0.0,
                "route_ok": bool(route_ok),
                "status_ok": bool(status_ok),
                "target": None,
                "route_event": route_event_name,
                "route_event_distance_m": route_event_distance_m,
                "route_event_speed_limit_mps": route_event_speed_limit_mps,
                "route_event_traffic_light_state": route_event_traffic_light_state,
                "route_event_reason": route_event_reason,
                "route_event_stop_required": route_event_stop_required,
                "route_event_stop_point_source": route_event_stop_point_source,
                "route_event_confidence": route_event_confidence,
                "route_event_age_s": route_event_age_s,
                **route_only_debug,
            }
            msg = String()
            msg.data = json.dumps(plan)
            self.plan_pub.publish(msg)
            self.runtime_logger.write({
                "kind": "lane_plan",
                "route_point_count": route_point_count,
                "plan": plan,
                "target_speed_before_route_events_mps": None,
                "target_speed_after_route_events_mps": 0.0,
            })
            return

        loc = self.ego.get("location", {})
        rot = self.ego.get("rotation", {})
        ego_x = float(loc.get("x", 0.0))
        ego_y = float(loc.get("y", 0.0))
        ego_yaw_deg = float(rot.get("yaw", 0.0))
        ego_yaw = math.radians(ego_yaw_deg)
        speed = float(self.ego.get("speed_mps", 0.0))

        points = self.route.get("points", [])
        if not points:
            route_only_debug = self._route_only_debug_defaults()
            route_only_debug["mission_stop_active"] = mission_stop_active
            route_only_debug["mission_stop_reason"] = mission_stop_reason
            route_only_debug["route_target_recovery_active"] = route_target_recovery_active
            route_only_debug["target_speed_raw_mps"] = 0.0
            route_only_debug["target_speed_final_mps"] = 0.0
            route_only_debug["zero_speed_reason"] = f"route_invalid:{route_source}"
            plan = {
                "stamp": now,
                "source": "phase2b_pure_pursuit",
                "route_source": route_source,
                "distance_to_goal_m": route_distance_to_goal_m,
                "mission_goal_near_distance_m": route_goal_near_distance_m,
                "cruise_speed_mps": self.cruise_speed_mps,
                "target_speed_mps": 0.0,
                "turn_intensity": 0.0,
                "speed_reason": f"route_invalid:{route_source}",
                "speed_boost_enabled": self.speed_boost_enabled,
                "speed_boost_mps": self.nominal_speed_boost_mps,
                "speed_boost_applied": False,
                "boost_applied": False,
                "speed_context": "route_invalid",
                "pre_boost_speed_mps": 0.0,
                "post_boost_speed_mps": 0.0,
                "speed_limit_clamped": False,
                "clamp_reason": "route_invalid",
                "turn_speed_protected": True,
                "nearest_index": None,
                "selected_target_index": None,
                "lookahead_m": self.base_lookahead_m,
                "steer": 0.0,
                "route_ok": False,
                "status_ok": True,
                "target": None,
                "route_event": route_event_name,
                "route_event_distance_m": route_event_distance_m,
                "route_event_speed_limit_mps": route_event_speed_limit_mps,
                "route_event_traffic_light_state": route_event_traffic_light_state,
                "route_event_reason": route_event_reason,
                "route_event_stop_required": route_event_stop_required,
                "route_event_stop_point_source": route_event_stop_point_source,
                "route_event_confidence": route_event_confidence,
                "route_event_age_s": route_event_age_s,
                **route_only_debug,
            }
            msg = String(); msg.data = json.dumps(plan); self.plan_pub.publish(msg)
            self.runtime_logger.write({
                "kind": "lane_plan",
                "route_point_count": route_point_count,
                "plan": plan,
                "target_speed_before_route_events_mps": None,
                "target_speed_after_route_events_mps": 0.0,
            })
            return

        nearest_index = self._nearest_index_with_hysteresis(points, ego_x, ego_y)
        route_only_debug = self._route_tracking_metrics(
            points,
            nearest_index,
            ego_x,
            ego_y,
            ego_yaw_deg,
            now,
        )
        profile = compute_target_speed_from_route(
            points,
            nearest_index,
            cruise_speed_mps=self.cruise_speed_mps,
            min_turn_speed_mps=self.min_turn_speed_mps,
            max_speed_mps=self.max_speed_mps,
            moderate_turn_yaw_deg=self.moderate_turn_yaw_deg,
            sharp_turn_yaw_deg=self.sharp_turn_yaw_deg,
            speed_boost_enabled=self.speed_boost_enabled,
            nominal_speed_boost_mps=self.nominal_speed_boost_mps,
        )
        target_speed_before_route_events = float(profile["target_speed_mps"])

        target_speed, speed_reason = self._apply_route_event_speed_limit(
            target_speed_before_route_events,
            str(profile["speed_reason"]),
            active_route_event,
        )
        red_approach_speed_cap_mps = None
        if route_event_name == "traffic_light_red_approach":
            red_approach_speed_cap_mps = self._red_approach_speed_cap(
                traffic_light_distance_m
            )
            if red_approach_speed_cap_mps is not None:
                target_speed = min(target_speed, red_approach_speed_cap_mps)
                speed_reason = f"{speed_reason}+red_approach_distance_cap"
        target_speed_after_route_events = float(target_speed)
        if mission_stop_active:
            target_speed = 0.0
            speed_reason = f"{speed_reason}+{mission_stop_reason or 'mission_stop'}"
            target_speed_after_route_events = 0.0
        green_release_active = route_event_name in (
            "clear",
            "traffic_light_green_clear",
            "traffic_light_green_release",
        )
        force_event_stop = (
            active_route_event is not None
            and route_event_name in (
                "traffic_light_red_stop",
                "traffic_light_yellow_stop",
            )
            and route_event_stop_required
        )
        route_event_zero_reason = self._event_zero_speed_reason(
            route_event_name,
            active_route_event,
        )
        speed_error = target_speed - self.last_target_speed
        if speed_error > 0:
            max_delta = self.speed_slew_up_mps_per_s / max(1.0, self.rate_hz)
        else:
            max_delta = self.speed_slew_down_mps_per_s / max(1.0, self.rate_hz)

        if force_event_stop:
            self.last_target_speed = 0.0
        else:
            self.last_target_speed = clamp(
                self.last_target_speed + clamp(speed_error, -max_delta, max_delta),
                0.0,
                self.max_speed_mps,
            )
        if (
            route_event_name == "traffic_light_red_approach"
            and red_approach_speed_cap_mps is not None
        ):
            self.last_target_speed = min(self.last_target_speed, red_approach_speed_cap_mps)
        if green_release_active and target_speed_after_route_events > 0.0:
            self.last_target_speed = max(
                self.last_target_speed,
                min(target_speed_after_route_events, max_delta),
            )
        min_nonzero_speed_floor_applied = False
        if (
            target_speed_after_route_events > 0.0
            and route_event_name in (
                "clear",
                "traffic_light_green_clear",
                "traffic_light_green_release",
            )
            and not mission_stop_active
            and not force_event_stop
        ):
            before_floor_speed = self.last_target_speed
            self.last_target_speed = max(
                self.last_target_speed,
                min(
                    self.min_nonzero_target_speed_mps,
                    target_speed_after_route_events,
                    self.max_speed_mps,
                ),
            )
            min_nonzero_speed_floor_applied = self.last_target_speed > before_floor_speed

        lookahead_m = self.base_lookahead_m
        if self.dynamic_lookahead_enabled:
            lookahead_m = self.base_lookahead_m + min(
                self.max_lookahead_m - self.base_lookahead_m,
                self.last_target_speed * self.lookahead_gain,
            )
            if profile["turn_intensity"] >= self.moderate_turn_yaw_deg:
                lookahead_m = clamp(lookahead_m * 0.75, self.min_lookahead_m, self.max_lookahead_m)

        lookahead_m = clamp(lookahead_m, self.min_lookahead_m, self.max_lookahead_m)

        target = None
        selected_target_index = None
        for index in range(nearest_index, len(points)):
            pt = points[index]
            dx = float(pt.get("x", 0.0)) - ego_x
            dy = float(pt.get("y", 0.0)) - ego_y
            dist = math.hypot(dx, dy)
            if dist >= lookahead_m:
                target = pt
                selected_target_index = index
                break

        if target is None:
            target = points[-1]
            selected_target_index = len(points) - 1

        tl_target_clamped = False
        gate_events = (
            "traffic_light_red_approach",
            "traffic_light_red_stop",
            "traffic_light_yellow_slow",
            "traffic_light_yellow_stop",
        )
        gate_cap_allowed = (
            active_route_event is not None
            and (
                (
                    bool(
                        active_route_event.get(
                            "fence_intersection_found",
                            False,
                        )
                    )
                    and route_event_confidence in ("high", "medium")
                )
                or (
                    route_event_distance_to_stop_m is not None
                    and float(route_event_distance_to_stop_m) <= 1.0
                )
            )
        )
        if (
            active_route_event is not None
            and route_event_name in gate_events
            and tl_stop_route_index is not None
            and gate_cap_allowed
        ):
            stop_index = max(
                nearest_index,
                min(len(points) - 1, int(tl_stop_route_index)),
            )
            if selected_target_index is None or selected_target_index >= stop_index:
                target = dict(points[stop_index])
                stop_x = active_route_event.get("stop_x")
                stop_y = active_route_event.get("stop_y")
                stop_z = active_route_event.get("stop_z")
                if stop_x is not None and stop_y is not None:
                    target["x"] = float(stop_x)
                    target["y"] = float(stop_y)
                    if stop_z is not None:
                        target["z"] = float(stop_z)
                selected_target_index = stop_index
                tl_target_clamped = True

        target, task_debug = self._task_pull_over_target(
            target,
            route_distance_to_goal_m,
        )
        if task_debug["task_pull_over_mode"]:
            task_distance = task_debug.get("task_stop_distance_m")
            task_pose_phase = str(task_debug.get("task_pose_phase", "route_lane"))
            if task_debug.get("task_stop_reached") or task_debug.get("task_hold_active"):
                target_speed = 0.0
                target_speed_after_route_events = 0.0
                speed_reason = f"{speed_reason}+task_stop_reached"
                profile["speed_context"] = (
                    f"{self.mission_goal.get('goal_kind', 'task')}_hold"
                    if self.mission_goal is not None and task_debug.get("task_hold_active")
                    else "task_stop"
                )
                task_debug["task_stop_approach_speed_phase"] = "hold"
                task_debug["task_stop_approach_speed_mps"] = 0.0
            elif task_debug.get("task_stop_safety_hold_active"):
                guard_speed = max(0.0, self.task_stop_overshoot_guard_speed_mps)
                target_speed = min(target_speed, guard_speed)
                target_speed_after_route_events = min(
                    target_speed_after_route_events,
                    guard_speed,
                )
                safety_reason = task_debug.get("task_stop_safety_hold_reason") or "safety_hold"
                speed_reason = f"{speed_reason}+{safety_reason}"
                profile["speed_context"] = "task_stop_safety_hold"
                task_debug["task_stop_approach_speed_phase"] = "safety_hold"
                task_debug["task_stop_approach_speed_mps"] = guard_speed
            elif task_debug.get("task_stop_overshoot_guard_active"):
                guard_speed = max(0.0, self.task_stop_overshoot_guard_speed_mps)
                target_speed = min(target_speed, guard_speed)
                target_speed_after_route_events = min(
                    target_speed_after_route_events,
                    guard_speed,
                )
                speed_reason = f"{speed_reason}+final_stop_latch_hold"
                profile["speed_context"] = "final_stop_latch_hold"
                task_debug["task_stop_approach_speed_phase"] = "final_stop_latch_hold"
                task_debug["task_stop_approach_speed_mps"] = guard_speed
            elif task_distance is not None and float(task_distance) <= self.task_pull_over_final_distance_m:
                task_distance_f = float(task_distance)
                if task_debug.get("task_stop_alignment_active"):
                    phase_cap = self.task_stop_alignment_speed_mps
                    approach_phase = "alignment"
                    speed_context = "task_stop_alignment"
                    speed_suffix = "task_stop_alignment"
                elif task_distance_f <= self.task_stop_no_stop_before_final_distance_m:
                    phase_cap = self.task_stop_min_creep_speed_mps
                    approach_phase = "final_creep"
                    speed_context = "task_stop_final_creep"
                    speed_suffix = "task_stop_final_creep"
                elif task_distance_f <= self.task_stop_alignment_start_distance_m:
                    phase_cap = self.task_stop_final_align_speed_mps
                    approach_phase = "final_align"
                    speed_context = "task_stop_final_align"
                    speed_suffix = "task_stop_final_align"
                else:
                    phase_cap = self.task_stop_pre_align_speed_mps
                    approach_phase = "pre_align"
                    speed_context = (
                        "final_task_stop"
                        if task_pose_phase == "final_task_stop"
                        else "pre_stop_align"
                    )
                    speed_suffix = (
                        "final_task_stop"
                        if task_pose_phase == "final_task_stop"
                        else "pre_stop_align"
                    )
                target_speed = min(target_speed, phase_cap)
                target_speed_after_route_events = min(
                    target_speed_after_route_events,
                    phase_cap,
                )
                speed_reason = f"{speed_reason}+{speed_suffix}"
                profile["speed_context"] = speed_context
                task_debug["task_stop_approach_speed_phase"] = approach_phase
                task_debug["task_stop_approach_speed_mps"] = phase_cap
            else:
                approach_cap = self.task_stop_approach_cruise_speed_mps
                target_speed = min(target_speed, approach_cap)
                target_speed_after_route_events = min(
                    target_speed_after_route_events,
                    approach_cap,
                )
                if task_pose_phase == "pre_stop_align":
                    speed_reason = f"{speed_reason}+pre_stop_align"
                    profile["speed_context"] = "pre_stop_align"
                else:
                    speed_reason = f"{speed_reason}+task_pull_over"
                    profile["speed_context"] = "task_pull_over"
                task_debug["task_stop_approach_speed_phase"] = "approach_cruise"
                task_debug["task_stop_approach_speed_mps"] = approach_cap
            self.last_target_speed = min(self.last_target_speed, target_speed_after_route_events)

        raw_target_for_bias = dict(target)
        turn_direction = str(target.get("turn_direction", "unknown"))
        if task_debug.get("task_pull_over_mode"):
            target_source = "mission_task_stop"
        elif bool(target.get("is_junction", False)) or turn_direction in ("left", "right", "u_turn"):
            target_source = "junction_route"
        elif bool(target.get("right_lane_selected", False)):
            target_source = "right_lane_route"
        elif bool(target.get("lane_jump_disabled", False)):
            target_source = "route_lane_center"
        else:
            target_source = "fallback_lane"

        route_only_debug["target_source"] = target_source
        junction_locked = (
            target_source == "junction_route"
            or target.get("right_lane_reason") == "junction_route_lane_center_locked"
        )
        route_only_debug["junction_locked"] = junction_locked
        route_only_debug["lane_preference"] = str(target.get("lane_preference", "right"))
        route_only_debug["route_lane_id"] = target.get("route_lane_id", target.get("lane_id"))
        route_only_debug["requested_right_lane_id"] = target.get("requested_right_lane_id")
        route_only_debug["selected_lane_id"] = target.get(
            "selected_lane_id",
            target.get("lane_id"),
        )
        route_only_debug["selected_road_id"] = target.get(
            "selected_road_id",
            target.get("road_id"),
        )
        route_only_debug["right_lane_selected"] = bool(target.get("right_lane_selected", False))
        route_only_debug["right_lane_reason"] = target.get(
            "right_lane_reason",
            target.get("right_lane_projection_failed_reason"),
        )
        route_only_debug["right_lane_projection_status"] = target.get(
            "right_lane_projection_status"
        )
        route_only_debug["right_lane_projection_rejected_reason"] = target.get(
            "right_lane_projection_rejected_reason",
            target.get("right_lane_projection_failed_reason"),
        )
        route_only_debug["right_lane_fallback_used"] = bool(
            target.get("right_lane_fallback_used", False)
        )
        route_only_debug["fallback_kept_right_lane"] = bool(
            target.get("fallback_kept_right_lane", False)
        )
        route_only_debug["lane_jump_disabled"] = bool(
            target.get("lane_jump_disabled", True)
        )
        route_only_debug["selected_lane_lateral_right_m"] = target.get(
            "selected_lane_lateral_right_m"
        )
        route_only_debug["candidate_lane_ids"] = target.get("candidate_lane_ids", [])
        route_only_debug["candidate_lane_lateral_right_m"] = target.get(
            "candidate_lane_lateral_right_m",
            [],
        )
        route_only_debug["right_lane_calibration_source"] = target.get(
            "right_lane_calibration_source"
        )
        route_only_debug["task_stop_side_lateral_m"] = target.get(
            "task_stop_side_lateral_m"
        )

        suppress_bias_for_hold = bool(task_debug.get("task_hold_active", mission_stop_active))
        if (
            self.task_pull_over_keep_bias_until_reached
            and task_debug.get("task_pull_over_mode")
            and not task_debug.get("task_stop_reached")
        ):
            suppress_bias_for_hold = False
        bias_disabled_reason = "mission_task_stop_protected" if task_debug.get("task_pull_over_mode") else "route_lane_center"
        target, bias_debug = self._bias_target_right(
            target,
            route_event_name,
            suppress_bias_for_hold,
            speed,
            bias_disabled_reason=bias_disabled_reason,
        )
        tx = float(target.get("x", 0.0))
        ty = float(target.get("y", 0.0))

        dx = tx - ego_x
        dy = ty - ego_y
        local_x = math.cos(ego_yaw) * dx + math.sin(ego_yaw) * dy
        local_y = -math.sin(ego_yaw) * dx + math.cos(ego_yaw) * dy

        if lookahead_m <= 0.0:
            steering_angle = 0.0
        else:
            steering_angle = math.atan2(
                2.0 * self.wheel_base_m * local_y,
                max(1e-6, lookahead_m * lookahead_m),
            )

        raw_steer_norm = clamp(steering_angle / self.max_steer_angle_rad, -1.0, 1.0)
        steer_norm = raw_steer_norm
        steering_rate_limited = False
        if self.steering_rate_limit_enabled:
            max_steer_delta = max(0.01, self.max_steer_delta)
            steer_delta = raw_steer_norm - self.last_steer_cmd
            limited_delta = clamp(steer_delta, -max_steer_delta, max_steer_delta)
            steer_norm = clamp(self.last_steer_cmd + limited_delta, -1.0, 1.0)
            steering_rate_limited = abs(limited_delta - steer_delta) > 1e-6
        self.last_steer_cmd = steer_norm

        if abs(steer_norm) > 0.35:
            self.last_target_speed = min(self.last_target_speed, self.min_turn_speed_mps)

        lane_departure_speed_clamp_applied = False
        final_clamp_reason = profile.get("clamp_reason")
        if route_only_debug["lane_departure_risk"]:
            heading_error = abs(float(route_only_debug.get("route_heading_error_deg") or 0.0))
            recovery_cap = self.route_recovery_speed_mps
            if heading_error < self.route_conflict_heading_threshold_deg:
                recovery_cap = max(self.route_recovery_speed_mps, self.min_nonzero_target_speed_mps)
            clamped_speed = min(self.last_target_speed, recovery_cap)
            lane_departure_speed_clamp_applied = clamped_speed < self.last_target_speed
            self.last_target_speed = clamped_speed
            if lane_departure_speed_clamp_applied:
                final_clamp_reason = "lane_departure"
        zero_speed_reason = None
        task_hold_active = bool(task_debug.get("task_hold_active", False))
        task_stop_reached = bool(task_debug.get("task_stop_reached", False))
        task_reached_by_mission = bool(task_debug.get("task_stop_reached_by_mission", False))
        if mission_stop_active:
            zero_speed_reason = mission_stop_reason or "mission_stop"
        elif task_hold_active:
            zero_speed_reason = str(task_debug.get("task_pose_phase") or "task_hold")
        elif task_stop_reached or task_reached_by_mission:
            zero_speed_reason = "task_stop_reached"
        elif task_debug.get("task_stop_safety_hold_active"):
            zero_speed_reason = (
                str(task_debug.get("task_stop_safety_hold_reason"))
                if task_debug.get("task_stop_safety_hold_reason") is not None
                else "task_stop_safety_hold"
            )
        elif route_event_zero_reason is not None:
            zero_speed_reason = route_event_zero_reason

        route_recovery_active = (
            bool(route_only_debug.get("route_recovery_active", False))
            or bool(route_only_debug.get("route_target_recovery_active", False))
        )
        junction_exit_recovery_active = (
            junction_locked
            and zero_speed_reason is None
            and route_event_name in (
                "clear",
                "traffic_light_green_clear",
                "traffic_light_green_release",
            )
        )
        min_drive_speed = self._minimum_drive_speed(
            str(profile.get("speed_context", "nominal")),
            junction_exit_recovery_active,
            route_recovery_active,
        )
        min_speed_floor_reason = self._minimum_drive_speed_reason(
            str(profile.get("speed_context", "nominal")),
            junction_exit_recovery_active,
            route_recovery_active,
        )
        min_speed_floor_applied = False
        min_speed_floor_skipped_reason = None
        red_light_speed_cap_active = route_event_name in (
            "traffic_light_red_approach",
            "traffic_light_red_stop",
        )
        if (
            zero_speed_reason is None
            and not red_light_speed_cap_active
            and min_drive_speed > 0.0
            and self.last_target_speed < min_drive_speed
        ):
            before_floor_speed = self.last_target_speed
            self.last_target_speed = max(
                self.last_target_speed,
                min_drive_speed,
            )
            self.last_target_speed = clamp(self.last_target_speed, 0.0, self.max_speed_mps)
            min_speed_floor_applied = self.last_target_speed > before_floor_speed
            min_nonzero_speed_floor_applied = (
                min_nonzero_speed_floor_applied
                or min_speed_floor_applied
            )
        elif red_light_speed_cap_active:
            min_speed_floor_skipped_reason = "traffic_light_red_cap_active"
        if self.last_target_speed > 1e-3:
            zero_speed_reason = None
        route_only_debug["lane_departure_speed_clamp_applied"] = lane_departure_speed_clamp_applied
        route_only_debug["min_nonzero_speed_floor_applied"] = min_nonzero_speed_floor_applied
        route_only_debug["min_nonzero_target_speed_mps"] = self.min_nonzero_target_speed_mps
        route_only_debug["min_speed_floor_applied"] = min_speed_floor_applied
        route_only_debug["min_speed_floor_reason"] = (
            min_speed_floor_reason if min_speed_floor_applied else None
        )
        route_only_debug["min_speed_floor_skipped_reason"] = min_speed_floor_skipped_reason
        route_only_debug["nominal_min_speed_mps"] = self.nominal_min_speed_mps
        route_only_debug["junction_recovery_min_speed_mps"] = self.junction_recovery_min_speed_mps
        route_only_debug["junction_exit_min_speed_mps"] = self.junction_exit_min_speed_mps
        route_only_debug["junction_exit_recovery_active"] = junction_exit_recovery_active
        route_only_debug["target_speed_raw_mps"] = round(target_speed_before_route_events, 3)
        route_only_debug["target_speed_final_mps"] = round(self.last_target_speed, 3)
        route_only_debug["target_speed_final"] = round(self.last_target_speed, 3)
        route_only_debug["zero_speed_reason"] = zero_speed_reason
        route_only_debug["current_speed_mps"] = round(speed, 3)
        route_only_debug["traffic_light_distance_m"] = (
            round(float(traffic_light_distance_m), 3)
            if traffic_light_distance_m is not None
            else None
        )
        route_only_debug["front_bumper_to_stopline_m"] = (
            round(float(route_event_front_bumper_to_stopline_m), 3)
            if route_event_front_bumper_to_stopline_m is not None
            else None
        )
        route_only_debug["red_approach_speed_cap_mps"] = (
            round(float(red_approach_speed_cap_mps), 3)
            if red_approach_speed_cap_mps is not None
            else None
        )
        route_only_debug["red_approach_profile_mode"] = (
            "soft" if red_approach_speed_cap_mps is not None else None
        )
        route_only_debug["red_stop_trigger_m"] = (
            round(float(red_stop_trigger_m), 3)
            if red_stop_trigger_m is not None
            else None
        )
        route_only_debug["red_stop_triggered_by_distance"] = red_stop_triggered_by_distance
        route_only_debug["selected_route_index"] = selected_target_index
        route_only_debug["lookahead_distance_m"] = round(lookahead_m, 3)
        route_only_debug["steering_limited"] = abs(raw_steer_norm) >= 1.0
        route_only_debug["steering_rate_limited"] = steering_rate_limited
        route_only_debug["mission_stop_active"] = mission_stop_active
        route_only_debug["mission_stop_reason"] = mission_stop_reason
        route_only_debug["route_target_recovery_active"] = route_target_recovery_active
        route_only_debug.update(task_debug)
        route_only_debug.update(bias_debug)
        route_yaw_rad = math.radians(float(raw_target_for_bias.get("yaw", ego_yaw_deg)))
        biased_lateral_error = (
            -math.sin(route_yaw_rad) * (ego_x - tx)
            + math.cos(route_yaw_rad) * (ego_y - ty)
        )
        route_only_debug["biased_lateral_error_m"] = round(float(biased_lateral_error), 3)

        plan = {
            "stamp": now,
            "source": "phase2b_pure_pursuit",
            "route_source": route_source,
            "distance_to_goal_m": route_distance_to_goal_m,
            "mission_goal_near_distance_m": route_goal_near_distance_m,
            "cruise_speed_mps": self.cruise_speed_mps,
            "target_speed_mps": round(self.last_target_speed, 3),
            "turn_intensity": round(float(profile["turn_intensity"]), 3),
            "speed_reason": speed_reason,
            "speed_boost_enabled": bool(profile["speed_boost_enabled"]),
            "speed_boost_mps": round(float(profile["speed_boost_mps"]), 3),
            "speed_boost_applied": bool(profile["speed_boost_applied"]),
            "boost_applied": bool(profile.get("boost_applied", profile["speed_boost_applied"])),
            "speed_context": profile["speed_context"],
            "pre_boost_speed_mps": round(float(profile["pre_boost_speed_mps"]), 3),
            "post_boost_speed_mps": round(float(profile["post_boost_speed_mps"]), 3),
            "speed_limit_clamped": bool(profile["speed_limit_clamped"]),
            "clamp_reason": final_clamp_reason,
            "turn_speed_protected": bool(profile["turn_speed_protected"]),
            "nearest_index": nearest_index,
            "selected_target_index": selected_target_index,
            "lookahead_m": round(lookahead_m, 3),
            "steer": float(steer_norm),
            "route_ok": True,
            "status_ok": True,
            "target": {"x": tx, "y": ty},
            "route_event": route_event_name,
            "route_event_distance_m": route_event_distance_m,
            "route_event_speed_limit_mps": route_event_speed_limit_mps,
            "route_event_traffic_light_state": route_event_traffic_light_state,
            "route_event_reason": route_event_reason,
            "route_event_stop_required": route_event_stop_required,
            "route_event_stop_point_source": route_event_stop_point_source,
            "route_event_confidence": route_event_confidence,
            "route_event_age_s": route_event_age_s,
            "tl_stop_route_index": tl_stop_route_index,
            "tl_target_clamped": tl_target_clamped,
            "tl_stop_s": tl_stop_s,
            "tl_distance_to_stop_m": route_event_distance_to_stop_m,
            **route_only_debug,
        }

        msg = String()
        msg.data = json.dumps(plan)
        self.plan_pub.publish(msg)
        if route_event_name != self._last_logged_route_event:
            self._last_logged_route_event = route_event_name
            self.get_logger().info(
                "LaneFollower route event update: "
                f"event={route_event_name} reason={route_event_reason} "
                f"target_speed_before={round(target_speed_before_route_events, 3)} "
                f"target_speed_after={round(target_speed_after_route_events, 3)} "
                f"target_speed_final={round(self.last_target_speed, 3)}"
            )

        dbg = String()
        debug_payload = {
            "stamp": now,
            "local_x": local_x,
            "local_y": local_y,
            "steer": steer_norm,
            "cruise_speed_mps": self.cruise_speed_mps,
            "target_speed_before_route_events_mps": round(target_speed_before_route_events, 3),
            "target_speed_after_route_events_mps": round(target_speed_after_route_events, 3),
            "target_speed_mps": self.last_target_speed,
            "turn_intensity": profile["turn_intensity"],
            "speed_reason": speed_reason,
            "speed_boost_enabled": bool(profile["speed_boost_enabled"]),
            "speed_boost_mps": profile["speed_boost_mps"],
            "speed_boost_applied": bool(profile["speed_boost_applied"]),
            "boost_applied": bool(profile.get("boost_applied", profile["speed_boost_applied"])),
            "speed_context": profile["speed_context"],
            "pre_boost_speed_mps": profile["pre_boost_speed_mps"],
            "post_boost_speed_mps": profile["post_boost_speed_mps"],
            "speed_limit_clamped": bool(profile["speed_limit_clamped"]),
            "clamp_reason": final_clamp_reason,
            "turn_speed_protected": bool(profile["turn_speed_protected"]),
            "route_event": route_event_name,
            "route_event_distance_m": route_event_distance_m,
            "route_event_speed_limit_mps": route_event_speed_limit_mps,
            "route_event_traffic_light_state": route_event_traffic_light_state,
            "route_event_reason": route_event_reason,
            "route_event_stop_required": route_event_stop_required,
            "route_event_stop_point_source": route_event_stop_point_source,
            "route_event_confidence": route_event_confidence,
            "route_event_age_s": route_event_age_s,
            "tl_stop_route_index": tl_stop_route_index,
            "tl_target_clamped": tl_target_clamped,
            "tl_stop_s": tl_stop_s,
            "tl_distance_to_stop_m": route_event_distance_to_stop_m,
            **route_only_debug,
        }
        dbg.data = json.dumps(debug_payload)
        self.debug_pub.publish(dbg)
        self.runtime_logger.write({
            "kind": "lane_plan",
            "route_point_count": route_point_count,
            "nearest_route_index": nearest_index,
            "target_speed_before_route_events_mps": round(target_speed_before_route_events, 3),
            "target_speed_after_route_events_mps": round(target_speed_after_route_events, 3),
            "plan": plan,
            "debug": debug_payload,
        })


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollower()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
