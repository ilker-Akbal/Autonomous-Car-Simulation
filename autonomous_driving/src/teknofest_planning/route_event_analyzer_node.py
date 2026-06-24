import json
import math
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_planning.route_geometry import (
    cumulative_route_s,
    is_ahead_on_route,
    project_actor_to_route,
)
from teknofest_sim.carla_loader import load_carla


@dataclass
class TrafficLightStopGate:
    light_id: int
    light_state: str
    stop_point_source: str
    stop_x: float
    stop_y: float
    stop_z: float
    stop_route_index: int
    stop_s: float
    distance_to_stop_m: float
    distance_to_light_m: Optional[float]
    lateral_distance_m: float
    confidence: str
    road_lane_match: Optional[bool]
    fence_intersection_found: bool
    projection: dict[str, float | int]
    center_to_stopline_m: Optional[float] = None
    front_bumper_to_stopline_m: Optional[float] = None
    ego_center_x: Optional[float] = None
    ego_center_y: Optional[float] = None
    ego_front_x: Optional[float] = None
    ego_front_y: Optional[float] = None
    applied_stop_line_buffer_m: Optional[float] = None
    stop_anchor_forward_offset_m: Optional[float] = None
    stopline_road_id: Optional[int] = None
    stopline_lane_id: Optional[int] = None
    ego_road_id: Optional[int] = None
    ego_lane_id: Optional[int] = None
    same_lane_or_compatible: Optional[bool] = None
    route_heading_deg: Optional[float] = None
    candidate_heading_deg: Optional[float] = None
    candidate_yaw_diff_deg: Optional[float] = None
    candidate_cross_track_m: Optional[float] = None
    candidate_along_track_m: Optional[float] = None
    candidate_on_route_corridor: Optional[bool] = None
    candidate_heading_ok: Optional[bool] = None
    candidate_ahead_ok: Optional[bool] = None
    candidate_rejected: bool = False
    candidate_reject_reason: Optional[str] = None


class RouteEventAnalyzer(Node):
    def __init__(self):
        super().__init__("route_event_analyzer")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("route_lateral_margin_m", 3.0)
        self.declare_parameter("event_horizon_m", 45.0)
        self.declare_parameter("vehicle_follow_distance_m", 10.0)
        self.declare_parameter("vehicle_stop_distance_m", 6.0)
        self.declare_parameter("stopped_vehicle_speed_mps", 0.4)
        self.declare_parameter("stopped_vehicle_stop_distance_m", 12.0)
        self.declare_parameter("pedestrian_stop_distance_m", 8.0)
        self.declare_parameter("min_event_speed_mps", 0.0)
        self.declare_parameter("follow_time_gap_s", 1.5)
        self.declare_parameter("stale_route_timeout_s", 1.0)
        self.declare_parameter("enable_traffic_light_events", True)
        self.declare_parameter("traffic_light_horizon_m", 60.0)
        self.declare_parameter("traffic_light_lateral_margin_m", 4.0)
        self.declare_parameter("red_detection_horizon_m", 60.0)
        self.declare_parameter("red_approach_distance_m", 45.0)
        self.declare_parameter("red_stop_distance_m", 8.0)
        self.declare_parameter("red_stop_trigger_base_m", 4.0)
        self.declare_parameter("red_stop_trigger_max_m", 8.0)
        self.declare_parameter("red_stop_trigger_speed_gain_s", 1.5)
        self.declare_parameter("red_stop_trigger_speed_buffer_m", 1.5)
        self.declare_parameter("red_creep_distance_m", 3.0)
        self.declare_parameter("red_approach_speed_mps", 2.0)
        self.declare_parameter("red_creep_speed_mps", 0.8)
        self.declare_parameter("yellow_slow_distance_m", 30.0)
        self.declare_parameter("yellow_stop_distance_m", 8.0)
        self.declare_parameter("traffic_light_stop_buffer_m", 1.0)
        self.declare_parameter("traffic_light_stop_front_bumper_offset_m", 2.0)
        self.declare_parameter("traffic_light_stop_line_buffer_m", 1.0)
        self.declare_parameter("traffic_light_stop_distance_tolerance_m", 0.4)
        self.declare_parameter("traffic_light_stop_debug_enabled", True)
        self.declare_parameter("traffic_light_stop_anchor_forward_offset_m", 0.0)
        self.declare_parameter("traffic_light_post_green_ignore_s", 6.0)
        self.declare_parameter("traffic_light_passed_ignore_distance_m", 8.0)
        self.declare_parameter("traffic_light_passed_stopline_threshold_m", 0.25)
        self.declare_parameter("traffic_light_min_stop_speed_mps", 0.0)
        self.declare_parameter("yellow_slow_speed_mps", 1.5)
        self.declare_parameter("green_release_distance_m", 8.0)
        self.declare_parameter("green_ignore_after_pass_m", 6.0)
        self.declare_parameter("tl_hold_state_memory_s", 2.0)
        self.declare_parameter("tl_lost_grace_s", 0.75)
        self.declare_parameter("green_release_grace_s", 1.0)
        self.declare_parameter("stopped_speed_threshold_mps", 0.25)
        self.declare_parameter("stopline_reached_distance_m", 1.5)
        self.declare_parameter("cruise_speed_mps", 4.5)
        self.declare_parameter("tl_stop_line_buffer_m", 1.0)
        self.declare_parameter("tl_decel_max_mps2", 1.2)
        self.declare_parameter("tl_slow_speed_mps", 0.8)
        self.declare_parameter("tl_min_profile_speed_mps", 0.4)
        self.declare_parameter("tl_hard_stop_distance_m", 1.2)
        self.declare_parameter("tl_profile_horizon_m", 45.0)
        self.declare_parameter("tl_fence_width_m", 8.0)
        self.declare_parameter("yellow_pass_time_s", 1.0)
        self.declare_parameter("traffic_light_debug", True)

        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = str(self.get_parameter("ego_role_name").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.route_lateral_margin_m = float(self.get_parameter("route_lateral_margin_m").value)
        self.event_horizon_m = float(self.get_parameter("event_horizon_m").value)
        self.vehicle_follow_distance_m = float(self.get_parameter("vehicle_follow_distance_m").value)
        self.vehicle_stop_distance_m = float(self.get_parameter("vehicle_stop_distance_m").value)
        self.stopped_vehicle_speed_mps = float(
            self.get_parameter("stopped_vehicle_speed_mps").value
        )
        self.stopped_vehicle_stop_distance_m = float(
            self.get_parameter("stopped_vehicle_stop_distance_m").value
        )
        self.pedestrian_stop_distance_m = float(self.get_parameter("pedestrian_stop_distance_m").value)
        self.min_event_speed_mps = float(self.get_parameter("min_event_speed_mps").value)
        self.follow_time_gap_s = float(self.get_parameter("follow_time_gap_s").value)
        self.stale_route_timeout_s = float(self.get_parameter("stale_route_timeout_s").value)
        self.enable_traffic_light_events = bool(
            self.get_parameter("enable_traffic_light_events").value
        )
        self.traffic_light_horizon_m = float(
            self.get_parameter("traffic_light_horizon_m").value
        )
        self.traffic_light_lateral_margin_m = float(
            self.get_parameter("traffic_light_lateral_margin_m").value
        )
        self.red_detection_horizon_m = float(
            self.get_parameter("red_detection_horizon_m").value
        )
        self.red_approach_distance_m = float(
            self.get_parameter("red_approach_distance_m").value
        )
        self.red_stop_distance_m = float(self.get_parameter("red_stop_distance_m").value)
        self.red_stop_trigger_base_m = float(
            self.get_parameter("red_stop_trigger_base_m").value
        )
        self.red_stop_trigger_max_m = float(
            self.get_parameter("red_stop_trigger_max_m").value
        )
        self.red_stop_trigger_speed_gain_s = float(
            self.get_parameter("red_stop_trigger_speed_gain_s").value
        )
        self.red_stop_trigger_speed_buffer_m = float(
            self.get_parameter("red_stop_trigger_speed_buffer_m").value
        )
        self.red_creep_distance_m = float(
            self.get_parameter("red_creep_distance_m").value
        )
        self.red_approach_speed_mps = float(
            self.get_parameter("red_approach_speed_mps").value
        )
        self.red_creep_speed_mps = float(
            self.get_parameter("red_creep_speed_mps").value
        )
        self.yellow_slow_distance_m = float(
            self.get_parameter("yellow_slow_distance_m").value
        )
        self.yellow_stop_distance_m = float(
            self.get_parameter("yellow_stop_distance_m").value
        )
        self.traffic_light_stop_buffer_m = float(
            self.get_parameter("traffic_light_stop_buffer_m").value
        )
        self.traffic_light_stop_front_bumper_offset_m = float(
            self.get_parameter("traffic_light_stop_front_bumper_offset_m").value
        )
        self.traffic_light_stop_line_buffer_m = float(
            self.get_parameter("traffic_light_stop_line_buffer_m").value
        )
        self.traffic_light_stop_distance_tolerance_m = float(
            self.get_parameter("traffic_light_stop_distance_tolerance_m").value
        )
        self.traffic_light_stop_debug_enabled = bool(
            self.get_parameter("traffic_light_stop_debug_enabled").value
        )
        self.traffic_light_stop_anchor_forward_offset_m = float(
            self.get_parameter("traffic_light_stop_anchor_forward_offset_m").value
        )
        self.traffic_light_post_green_ignore_s = float(
            self.get_parameter("traffic_light_post_green_ignore_s").value
        )
        self.traffic_light_passed_ignore_distance_m = float(
            self.get_parameter("traffic_light_passed_ignore_distance_m").value
        )
        self.traffic_light_passed_stopline_threshold_m = float(
            self.get_parameter("traffic_light_passed_stopline_threshold_m").value
        )
        self.traffic_light_min_stop_speed_mps = float(
            self.get_parameter("traffic_light_min_stop_speed_mps").value
        )
        self.yellow_slow_speed_mps = float(
            self.get_parameter("yellow_slow_speed_mps").value
        )
        self.green_release_distance_m = float(
            self.get_parameter("green_release_distance_m").value
        )
        self.green_ignore_after_pass_m = float(
            self.get_parameter("green_ignore_after_pass_m").value
        )
        self.tl_hold_state_memory_s = float(
            self.get_parameter("tl_hold_state_memory_s").value
        )
        self.tl_lost_grace_s = float(
            self.get_parameter("tl_lost_grace_s").value
        )
        self.green_release_grace_s = float(
            self.get_parameter("green_release_grace_s").value
        )
        self.stopped_speed_threshold_mps = float(
            self.get_parameter("stopped_speed_threshold_mps").value
        )
        self.stopline_reached_distance_m = float(
            self.get_parameter("stopline_reached_distance_m").value
        )
        self.cruise_speed_mps = float(
            self.get_parameter("cruise_speed_mps").value
        )
        self.tl_stop_line_buffer_m = float(
            self.get_parameter("tl_stop_line_buffer_m").value
        )
        self.tl_stop_line_buffer_m = self.traffic_light_stop_line_buffer_m
        self.tl_decel_max_mps2 = float(
            self.get_parameter("tl_decel_max_mps2").value
        )
        self.tl_slow_speed_mps = float(
            self.get_parameter("tl_slow_speed_mps").value
        )
        self.tl_min_profile_speed_mps = float(
            self.get_parameter("tl_min_profile_speed_mps").value
        )
        self.tl_hard_stop_distance_m = float(
            self.get_parameter("tl_hard_stop_distance_m").value
        )
        self.tl_profile_horizon_m = float(
            self.get_parameter("tl_profile_horizon_m").value
        )
        self.tl_fence_width_m = float(
            self.get_parameter("tl_fence_width_m").value
        )
        self.yellow_pass_time_s = float(
            self.get_parameter("yellow_pass_time_s").value
        )
        self.traffic_light_debug = bool(
            self.get_parameter("traffic_light_debug").value
        )

        self._last_status: Optional[dict[str, Any]] = None
        self._last_route: Optional[dict[str, Any]] = None
        self._last_status_time = 0.0
        self._last_route_time = 0.0
        self._last_warning_times: dict[str, float] = {}
        self._traffic_light_state_memory: dict[int, tuple[str, float]] = {}
        self._restrictive_traffic_lights: dict[int, float] = {}
        self._green_release_until: dict[int, float] = {}
        self._post_green_ignore_until: dict[int, float] = {}
        self._post_tl_ignore_until: dict[int, float] = {}
        self._last_green_clear_light_id: Optional[int] = None
        self._last_restrictive_tl_candidate: Optional[dict[str, Any]] = None
        self._last_restrictive_tl_time = 0.0
        self.active_tl_id: Optional[int] = None
        self.active_tl_state: Optional[str] = None
        self.active_tl_last_seen_time = 0.0
        self.active_stop_s: Optional[float] = None
        self.active_stop_route_index: Optional[int] = None
        self.active_stop_point_source: Optional[str] = None
        self.red_hold_active = False
        self.stopped_for_tl_id: Optional[int] = None
        self.last_tl_event = "clear"
        self.previous_tl_state: Optional[str] = None
        self.green_release_triggered = False
        self.red_hold_active_before = False
        self.red_hold_active_after = False
        self.stop_hold_cleared = False
        self.release_reason: Optional[str] = None
        self._green_release_details: dict[int, dict[str, Any]] = {}

        self._carla = None
        self._client = None
        self._world = None
        self._ego_vehicle = None
        self._startup_snapshot_emitted = False
        self._last_warning_debug_publish = 0.0
        self._last_debug_reason: Optional[str] = None
        self.runtime_logger = RuntimeJsonlLogger(
            node_name="route_event_analyzer",
            file_name="route_event_analyzer.jsonl",
        )
        self.runtime_logger.update_summary({
            "phase2_log_session": self.runtime_logger.session_id,
            "route_event_analyzer_log": self.runtime_logger.path(),
        })
        self.get_logger().info(
            f"RouteEventAnalyzer JSONL logging -> {self.runtime_logger.path()}"
        )

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_subscription(String, "/adas/planning/route", self._route_callback, 10)
        self.event_pub = self.create_publisher(String, "/adas/planning/route_events", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/route_events_debug", 10)

        self._connect_to_carla()
        self.timer = self.create_timer(
            1.0 / max(0.1, self.publish_rate_hz),
            self._tick,
        )

    def _warn_throttled(self, key: str, message: str, period_s: float = 10.0) -> None:
        now = time.monotonic()
        if now - self._last_warning_times.get(key, 0.0) >= period_s:
            self.get_logger().warn(message)
            self._last_warning_times[key] = now

    def _connect_to_carla(self) -> None:
        try:
            self._carla = load_carla(self.carla_root)
            self._client = self._carla.Client(self.host, self.port)
            self._client.set_timeout(5.0)
            self._world = self._client.get_world()
            self._ego_vehicle = None
        except Exception as exc:
            self._client = None
            self._world = None
            self._ego_vehicle = None
            self._warn_throttled(
                "carla_connection",
                f"RouteEventAnalyzer: CARLA connection failed: {exc}",
            )

    @staticmethod
    def _location_to_dict(location: Any) -> dict[str, float]:
        return {
            "x": round(float(getattr(location, "x", 0.0)), 3),
            "y": round(float(getattr(location, "y", 0.0)), 3),
            "z": round(float(getattr(location, "z", 0.0)), 3),
        }

    def _transform_to_dict(self, actor: Any) -> Optional[dict[str, Any]]:
        try:
            transform = actor.get_transform()
        except Exception:
            return None
        return {
            "location": self._location_to_dict(transform.location),
            "rotation": {
                "pitch": round(float(transform.rotation.pitch), 3),
                "yaw": round(float(transform.rotation.yaw), 3),
                "roll": round(float(transform.rotation.roll), 3),
            },
        }

    def _traffic_light_snapshot(self, traffic_light: Any) -> dict[str, Any]:
        state = self._get_traffic_light_state(traffic_light)
        snapshot = {
            "id": int(traffic_light.id),
            "state": state,
            "type_id": str(traffic_light.type_id),
        }
        transform = self._transform_to_dict(traffic_light)
        if transform is not None:
            snapshot["transform"] = transform
        return snapshot

    def _build_warning_payload(
        self,
        *,
        reason: str,
        map_name: Optional[str],
        ego_location: dict[str, Any],
        traffic_light_actor_count: int,
    ) -> dict[str, Any]:
        return {
            "stamp": time.time(),
            "ok": False,
            "event": None,
            "reason": reason,
            "traffic_light_actor_count": int(traffic_light_actor_count),
            "map": map_name,
            "ego_location": ego_location,
        }

    def _emit_startup_tl_diagnostics(self) -> None:
        if self._world is None or self._startup_snapshot_emitted:
            return
        try:
            world_actors = self._world.get_actors()
            traffic_lights = list(world_actors.filter("traffic.traffic_light*"))
            map_name = self._world.get_map().name
            snapshot = {
                "kind": "startup_world_snapshot",
                "map": map_name,
                "total_actor_count": len(world_actors),
                "traffic_light_actor_count": len(traffic_lights),
                "traffic_lights": [
                    self._traffic_light_snapshot(traffic_light)
                    for traffic_light in traffic_lights
                ],
            }
            self._startup_snapshot_emitted = True
            self.runtime_logger.write(snapshot)
            self.get_logger().info(
                f"RouteEventAnalyzer startup diagnostics: map={map_name}, "
                f"traffic_light_actor_count={len(traffic_lights)}"
            )
            if self.enable_traffic_light_events and not traffic_lights:
                self.get_logger().warn(
                    "TL EVENTS ENABLED BUT NO traffic.traffic_light ACTORS FOUND IN CARLA WORLD. "
                    "RED LIGHT STOP CANNOT WORK ON THIS MAP."
                )
        except Exception as exc:
            self._warn_throttled(
                "startup_snapshot",
                f"RouteEventAnalyzer: startup snapshot failed: {exc}",
            )

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
            self._last_status_time = time.time()
        except Exception:
            self._warn_throttled(
                "status_json",
                "RouteEventAnalyzer: failed to parse /adas/carla/status JSON",
            )

    def _route_callback(self, msg: String) -> None:
        try:
            self._last_route = json.loads(msg.data)
            self._last_route_time = time.time()
        except Exception:
            self._warn_throttled(
                "route_json",
                "RouteEventAnalyzer: failed to parse /adas/planning/route JSON",
            )

    def _find_ego_vehicle(self):
        if self._world is None:
            return None

        status_ego_id = None
        if self._last_status is not None:
            status_ego_id = self._last_status.get("ego_id")

        if self._ego_vehicle is not None and self._ego_vehicle.is_alive:
            if status_ego_id is None or self._ego_vehicle.id == int(status_ego_id):
                return self._ego_vehicle

        for vehicle in self._world.get_actors().filter("vehicle.*"):
            if status_ego_id is not None and vehicle.id == int(status_ego_id):
                self._ego_vehicle = vehicle
                return vehicle
            if vehicle.attributes.get("role_name", "") == self.ego_role_name:
                self._ego_vehicle = vehicle
                return vehicle
        return None

    def _base_event(self, ok: bool, reason: str) -> dict[str, Any]:
        return {
            "stamp": time.time(),
            "ok": ok,
            "event": "clear",
            "target_speed_limit_mps": None,
            "stop_required": False,
            "distance_m": None,
            "distance_to_stop_m": None,
            "distance_to_light_m": None,
            "actor_id": None,
            "traffic_light_id": None,
            "actor_type": None,
            "actor_speed_mps": None,
            "traffic_light_state": None,
            "route_index": None,
            "stop_route_index": None,
            "route_lateral_distance_m": None,
            "lateral_distance_m": None,
            "stop_buffer_m": None,
            "stop_point_source": None,
            "confidence": None,
            "fence_intersection_found": False,
            "stop_x": None,
            "stop_y": None,
            "stop_z": None,
            "stop_s": None,
            "distance_is_buffered": False,
            "reason": reason,
            "active_tl_id": self.active_tl_id,
            "active_tl_state": self.active_tl_state,
            "active_light_id": self.active_tl_id,
            "active_light_state": self.active_tl_state,
            "previous_tl_state": self.previous_tl_state,
            "stopped_for_tl_id": self.stopped_for_tl_id,
            "green_release_triggered": self.green_release_triggered,
            "red_hold_active_before": self.red_hold_active_before,
            "red_hold_active_after": self.red_hold_active_after,
            "stop_hold_cleared": self.stop_hold_cleared,
            "release_reason": self.release_reason,
            "stopline_x": None,
            "stopline_y": None,
            "ego_center_x": None,
            "ego_center_y": None,
            "ego_front_x": None,
            "ego_front_y": None,
            "center_to_stopline_m": None,
            "front_bumper_to_stopline_m": None,
            "passed_stopline": False,
            "passed_stopline_threshold_m": self.traffic_light_passed_stopline_threshold_m,
            "post_green_ignore_active": False,
            "post_green_ignore_light_id": None,
            "post_tl_ignore_active": False,
            "same_light_ignore_active": False,
            "last_green_clear_light_id": self._last_green_clear_light_id,
            "stale_red_stop_suppressed": False,
            "stale_red_stop_suppress_reason": None,
            "last_restrictive_light_id": (
                self._last_restrictive_tl_candidate.get("traffic_light_id")
                if self._last_restrictive_tl_candidate is not None
                else None
            ),
            "candidate_light_id": None,
            "candidate_road_id": None,
            "candidate_lane_id": None,
            "route_heading_deg": None,
            "candidate_heading_deg": None,
            "candidate_yaw_diff_deg": None,
            "candidate_cross_track_m": None,
            "candidate_along_track_m": None,
            "candidate_on_route_corridor": None,
            "candidate_heading_ok": None,
            "candidate_ahead_ok": None,
            "candidate_rejected": False,
            "candidate_reject_reason": None,
            "selected_light_id": None,
            "selected_light_reason": None,
            "current_candidate_light_id": None,
            "replay_candidate_light_id": None,
            "old_light_replay_suppressed": False,
            "applied_stop_line_buffer_m": self.tl_stop_line_buffer_m,
            "traffic_light_stop_front_bumper_offset_m": self.traffic_light_stop_front_bumper_offset_m,
            "traffic_light_stop_distance_tolerance_m": self.traffic_light_stop_distance_tolerance_m,
            "traffic_light_stop_anchor_forward_offset_m": self.traffic_light_stop_anchor_forward_offset_m,
            "red_stop_trigger_reason": None,
            "current_speed_mps": None,
            "red_stop_trigger_m": None,
            "red_stop_triggered_by_distance": False,
            "ego_road_id": None,
            "ego_lane_id": None,
            "active_light_road_id": None,
            "active_light_lane_id": None,
            "stopline_road_id": None,
            "stopline_lane_id": None,
            "same_lane_or_compatible": None,
            "rejected_light_reason": None,
            "desired_front_bumper_to_stopline_m": self.tl_stop_line_buffer_m,
            "predicted_stop_distance_m": None,
            "trigger_distance_m": None,
            "final_front_bumper_to_stopline_m": None,
            "stopline_anchor_source": None,
            "stopline_anchor_confidence": None,
            "visible_stopline_mismatch_possible": False,
        }

    def _candidate(
        self,
        event: str,
        actor,
        projection: dict[str, float | int],
        distance_ahead_m: float,
        target_speed_limit_mps: float,
        reason: str,
        actor_speed_mps: Optional[float] = None,
        traffic_light_state: Optional[str] = None,
        stop_buffer_m: Optional[float] = None,
        distance_to_light_m: Optional[float] = None,
        stop_point_source: Optional[str] = None,
        confidence: Optional[str] = None,
        gate: Optional[TrafficLightStopGate] = None,
        distance_is_buffered: bool = False,
        current_speed_mps: Optional[float] = None,
        red_stop_trigger_m: Optional[float] = None,
        red_stop_triggered_by_distance: bool = False,
    ) -> dict[str, Any]:
        stop_required = event in (
            "vehicle_stop",
            "pedestrian_stop",
            "traffic_light_red_stop",
            "traffic_light_yellow_stop",
        )
        now_monotonic = time.monotonic()
        traffic_light_id = (
            int(actor.id)
            if str(actor.type_id).startswith("traffic.traffic_light")
            else None
        )
        passed_stopline = self._gate_passed_stopline(gate)
        post_green_ignore_active = (
            traffic_light_id is not None
            and now_monotonic <= self._post_green_ignore_until.get(traffic_light_id, 0.0)
        )
        post_tl_ignore_active = (
            traffic_light_id is not None
            and now_monotonic <= self._post_tl_ignore_until.get(traffic_light_id, 0.0)
        )
        same_light_ignore_active = bool(
            traffic_light_id is not None
            and (
                traffic_light_id == self.active_tl_id
                or traffic_light_id == self.stopped_for_tl_id
                or traffic_light_id == self._last_green_clear_light_id
            )
            and (post_green_ignore_active or post_tl_ignore_active or passed_stopline)
        )
        lateral_distance_m = (
            gate.lateral_distance_m
            if gate is not None
            else float(projection["lateral_distance_m"])
        )
        return {
            "event": event,
            "target_speed_limit_mps": (
                None
                if event in (
                    "clear",
                    "traffic_light_green_clear",
                    "traffic_light_green_release",
                )
                else (
                    0.0
                    if stop_required
                    else max(self.min_event_speed_mps, target_speed_limit_mps)
                )
            ),
            "stop_required": stop_required,
            "distance_m": round(distance_ahead_m, 3),
            "distance_to_stop_m": round(distance_ahead_m, 3),
            "distance_to_light_m": (
                round(float(distance_to_light_m), 3)
                if distance_to_light_m is not None
                else None
            ),
            "actor_id": int(actor.id),
            "traffic_light_id": traffic_light_id,
            "actor_type": str(actor.type_id),
            "actor_speed_mps": (
                round(float(actor_speed_mps), 3)
                if actor_speed_mps is not None
                else None
            ),
            "traffic_light_state": traffic_light_state,
            "route_index": int(projection["route_index"]),
            "stop_route_index": int(projection["route_index"]),
            "route_lateral_distance_m": round(
                lateral_distance_m,
                3,
            ),
            "lateral_distance_m": round(
                lateral_distance_m,
                3,
            ),
            "stop_buffer_m": stop_buffer_m,
            "stop_point_source": stop_point_source,
            "confidence": confidence,
            "fence_intersection_found": (
                gate.fence_intersection_found if gate is not None else False
            ),
            "stop_x": gate.stop_x if gate is not None else None,
            "stop_y": gate.stop_y if gate is not None else None,
            "stop_z": gate.stop_z if gate is not None else None,
            "stop_s": gate.stop_s if gate is not None else None,
            "distance_is_buffered": distance_is_buffered,
            "reason": reason,
            "stopline_x": gate.stop_x if gate is not None else None,
            "stopline_y": gate.stop_y if gate is not None else None,
            "ego_center_x": gate.ego_center_x if gate is not None else None,
            "ego_center_y": gate.ego_center_y if gate is not None else None,
            "ego_front_x": gate.ego_front_x if gate is not None else None,
            "ego_front_y": gate.ego_front_y if gate is not None else None,
            "center_to_stopline_m": (
                round(float(gate.center_to_stopline_m), 3)
                if gate is not None and gate.center_to_stopline_m is not None
                else None
            ),
            "front_bumper_to_stopline_m": (
                round(float(gate.front_bumper_to_stopline_m), 3)
                if gate is not None and gate.front_bumper_to_stopline_m is not None
                else None
            ),
            "passed_stopline": passed_stopline,
            "post_green_ignore_active": post_green_ignore_active,
            "post_green_ignore_light_id": (
                traffic_light_id if post_green_ignore_active else None
            ),
            "post_tl_ignore_active": post_tl_ignore_active,
            "same_light_ignore_active": same_light_ignore_active,
            "last_green_clear_light_id": self._last_green_clear_light_id,
            "stale_red_stop_suppressed": False,
            "stale_red_stop_suppress_reason": None,
            "applied_stop_line_buffer_m": (
                round(float(gate.applied_stop_line_buffer_m), 3)
                if gate is not None and gate.applied_stop_line_buffer_m is not None
                else stop_buffer_m
            ),
            "traffic_light_stop_front_bumper_offset_m": self.traffic_light_stop_front_bumper_offset_m,
            "traffic_light_stop_distance_tolerance_m": self.traffic_light_stop_distance_tolerance_m,
            "traffic_light_stop_anchor_forward_offset_m": (
                round(float(gate.stop_anchor_forward_offset_m), 3)
                if gate is not None and gate.stop_anchor_forward_offset_m is not None
                else self.traffic_light_stop_anchor_forward_offset_m
            ),
            "red_stop_trigger_reason": reason if event in (
                "traffic_light_red_stop",
                "traffic_light_yellow_stop",
            ) else None,
            "current_speed_mps": (
                round(float(current_speed_mps), 3)
                if current_speed_mps is not None
                else None
            ),
            "red_stop_trigger_m": (
                round(float(red_stop_trigger_m), 3)
                if red_stop_trigger_m is not None
                else None
            ),
            "red_stop_triggered_by_distance": bool(red_stop_triggered_by_distance),
            "ego_road_id": gate.ego_road_id if gate is not None else None,
            "ego_lane_id": gate.ego_lane_id if gate is not None else None,
            "active_light_road_id": gate.stopline_road_id if gate is not None else None,
            "active_light_lane_id": gate.stopline_lane_id if gate is not None else None,
            "stopline_road_id": gate.stopline_road_id if gate is not None else None,
            "stopline_lane_id": gate.stopline_lane_id if gate is not None else None,
            "same_lane_or_compatible": gate.same_lane_or_compatible if gate is not None else None,
            "candidate_light_id": traffic_light_id,
            "candidate_road_id": gate.stopline_road_id if gate is not None else None,
            "candidate_lane_id": gate.stopline_lane_id if gate is not None else None,
            "route_heading_deg": (
                round(float(gate.route_heading_deg), 3)
                if gate is not None and gate.route_heading_deg is not None
                else None
            ),
            "candidate_heading_deg": (
                round(float(gate.candidate_heading_deg), 3)
                if gate is not None and gate.candidate_heading_deg is not None
                else None
            ),
            "candidate_yaw_diff_deg": (
                round(float(gate.candidate_yaw_diff_deg), 3)
                if gate is not None and gate.candidate_yaw_diff_deg is not None
                else None
            ),
            "candidate_cross_track_m": (
                round(float(gate.candidate_cross_track_m), 3)
                if gate is not None and gate.candidate_cross_track_m is not None
                else None
            ),
            "candidate_along_track_m": (
                round(float(gate.candidate_along_track_m), 3)
                if gate is not None and gate.candidate_along_track_m is not None
                else None
            ),
            "candidate_on_route_corridor": (
                gate.candidate_on_route_corridor if gate is not None else None
            ),
            "candidate_heading_ok": gate.candidate_heading_ok if gate is not None else None,
            "candidate_ahead_ok": gate.candidate_ahead_ok if gate is not None else None,
            "candidate_rejected": gate.candidate_rejected if gate is not None else False,
            "candidate_reject_reason": gate.candidate_reject_reason if gate is not None else None,
            "selected_light_id": traffic_light_id,
            "selected_light_reason": reason,
            "rejected_light_reason": None,
            "desired_front_bumper_to_stopline_m": self.tl_stop_line_buffer_m,
            "predicted_stop_distance_m": round(max(0.0, distance_ahead_m - self.tl_stop_line_buffer_m), 3),
            "trigger_distance_m": round(
                self.tl_stop_line_buffer_m + self.traffic_light_stop_distance_tolerance_m,
                3,
            ),
            "final_front_bumper_to_stopline_m": (
                round(float(gate.front_bumper_to_stopline_m), 3)
                if gate is not None and gate.front_bumper_to_stopline_m is not None
                else None
            ),
            "stopline_anchor_source": stop_point_source,
            "stopline_anchor_confidence": confidence,
            "visible_stopline_mismatch_possible": bool(
                gate is not None
                and gate.front_bumper_to_stopline_m is not None
                and abs(float(gate.front_bumper_to_stopline_m) - self.tl_stop_line_buffer_m) > 0.8
            ),
        }

    def _gate_passed_stopline(self, gate: Optional[TrafficLightStopGate]) -> bool:
        if gate is None or gate.front_bumper_to_stopline_m is None:
            return False
        return (
            float(gate.front_bumper_to_stopline_m)
            <= self.traffic_light_passed_stopline_threshold_m
        )

    def _candidate_passed_stopline(self, candidate: dict[str, Any]) -> bool:
        front_bumper_to_stopline_m = candidate.get("front_bumper_to_stopline_m")
        if front_bumper_to_stopline_m is not None:
            try:
                return (
                    float(front_bumper_to_stopline_m)
                    <= self.traffic_light_passed_stopline_threshold_m
                )
            except Exception:
                pass
        return bool(candidate.get("passed_stopline", False))

    def _mark_post_tl_ignore(self, traffic_light_id: int, now_monotonic: float) -> None:
        ignore_until = now_monotonic + self.traffic_light_post_green_ignore_s
        self._post_green_ignore_until[int(traffic_light_id)] = ignore_until
        self._post_tl_ignore_until[int(traffic_light_id)] = ignore_until

    def _same_light_ignore_active(
        self,
        traffic_light_id: int,
        gate: Optional[TrafficLightStopGate],
        now_monotonic: float,
    ) -> tuple[bool, Optional[str]]:
        post_green_active = now_monotonic <= self._post_green_ignore_until.get(
            int(traffic_light_id),
            0.0,
        )
        post_tl_active = now_monotonic <= self._post_tl_ignore_until.get(
            int(traffic_light_id),
            0.0,
        )
        same_light = (
            int(traffic_light_id) == self.active_tl_id
            or int(traffic_light_id) == self.stopped_for_tl_id
            or int(traffic_light_id) == self._last_green_clear_light_id
        )
        if same_light and post_green_active:
            return True, "post_green_same_light_ignore"
        if same_light and post_tl_active:
            return True, "post_tl_ignore_same_light"
        if same_light and self._gate_passed_stopline(gate):
            return True, "passed_stopline_clear"
        return False, None

    @staticmethod
    def _is_restrictive_tl_event(candidate: Optional[dict[str, Any]]) -> bool:
        if candidate is None:
            return False
        return str(candidate.get("event")) in (
            "traffic_light_red_stop",
            "traffic_light_red_approach",
            "traffic_light_yellow_stop",
            "traffic_light_yellow_slow",
        )

    def _suppress_stale_restrictive_candidate(
        self,
        candidate: Optional[dict[str, Any]],
        now_monotonic: float,
    ) -> tuple[Optional[dict[str, Any]], bool, Optional[str]]:
        if not self._is_restrictive_tl_event(candidate):
            return candidate, False, None
        traffic_light_id = candidate.get("traffic_light_id")
        if traffic_light_id is None:
            return candidate, False, None
        passed_stopline = self._candidate_passed_stopline(candidate)
        post_green_active = now_monotonic <= self._post_green_ignore_until.get(
            int(traffic_light_id),
            0.0,
        )
        post_tl_active = now_monotonic <= self._post_tl_ignore_until.get(
            int(traffic_light_id),
            0.0,
        )
        same_light = (
            int(traffic_light_id) == self.active_tl_id
            or int(traffic_light_id) == self.stopped_for_tl_id
            or int(traffic_light_id) == self._last_green_clear_light_id
        )
        reason = None
        if candidate.get("candidate_heading_ok") is False:
            reason = "rejected_opposite_direction_light"
        elif candidate.get("candidate_on_route_corridor") is False:
            reason = "rejected_out_of_route_corridor"
        elif candidate.get("candidate_ahead_ok") is False:
            reason = "rejected_not_ahead_light"
        if same_light and post_green_active:
            reason = "post_green_same_light_ignore"
        elif same_light and post_tl_active:
            reason = "post_tl_ignore_same_light"
        elif same_light and passed_stopline:
            if str(candidate.get("reason")) == "red_light_hold":
                reason = "stale_red_hold_passed_stopline_clear"
            else:
                reason = "passed_stopline_clear"
        if reason is None:
            return candidate, False, None

        clear_candidate = dict(candidate)
        clear_candidate.update(
            {
                "event": "clear",
                "target_speed_limit_mps": None,
                "stop_required": False,
                "reason": reason,
                "red_stop_trigger_reason": None,
                "stale_red_stop_suppressed": True,
                "stale_red_stop_suppress_reason": reason,
                "candidate_rejected": True,
                "candidate_reject_reason": reason,
                "passed_stopline": passed_stopline,
                "passed_stopline_threshold_m": self.traffic_light_passed_stopline_threshold_m,
                "same_light_ignore_active": True,
                "post_green_ignore_active": post_green_active,
                "post_green_ignore_light_id": (
                    int(traffic_light_id) if post_green_active else None
                ),
                "post_tl_ignore_active": post_tl_active,
            }
        )
        self.red_hold_active = False
        self.stopped_for_tl_id = None
        self._last_restrictive_tl_candidate = None
        self._last_restrictive_tl_time = 0.0
        if int(traffic_light_id) == self.active_tl_id and passed_stopline:
            self._clear_active_gate()
        return clear_candidate, True, reason

    @staticmethod
    def _angle_difference_deg(first: float, second: float) -> float:
        return abs((first - second + 180.0) % 360.0 - 180.0)

    @staticmethod
    def _traffic_light_state_name(state: Any) -> str:
        state_name = str(state).split(".")[-1]
        normalized = state_name.strip().lower()
        return {
            "red": "Red",
            "yellow": "Yellow",
            "green": "Green",
            "off": "Off",
            "unknown": "Unknown",
        }.get(normalized, "Unknown")

    def _get_traffic_light_state(self, traffic_light) -> str:
        now = time.monotonic()
        try:
            get_state = getattr(traffic_light, "get_state", None)
            raw_state = get_state() if callable(get_state) else traffic_light.state
            state = self._traffic_light_state_name(raw_state)
            self._traffic_light_state_memory[int(traffic_light.id)] = (state, now)
            return state
        except Exception:
            remembered = self._traffic_light_state_memory.get(int(traffic_light.id))
            if (
                remembered is not None
                and now - remembered[1] <= self.tl_hold_state_memory_s
            ):
                return remembered[0]
            return "Unknown"

    def _set_active_gate(
        self,
        gate: TrafficLightStopGate,
        state: str,
        seen_time: float,
    ) -> None:
        self.active_tl_id = gate.light_id
        self.active_tl_state = state
        self.active_tl_last_seen_time = seen_time
        self.active_stop_s = gate.stop_s
        self.active_stop_route_index = gate.stop_route_index
        self.active_stop_point_source = gate.stop_point_source

    def _clear_active_gate(self) -> None:
        self.active_tl_id = None
        self.active_tl_state = None
        self.active_stop_s = None
        self.active_stop_route_index = None
        self.active_stop_point_source = None
        self.red_hold_active = False
        self.stopped_for_tl_id = None

    @staticmethod
    def _find_actor_by_id(world_actors: Any, actor_id: int) -> Optional[Any]:
        try:
            actor = world_actors.find(int(actor_id))
            if actor is not None:
                return actor
        except Exception:
            pass
        try:
            return next(
                actor for actor in world_actors if int(actor.id) == int(actor_id)
            )
        except (StopIteration, TypeError):
            return None

    def _release_active_traffic_light_hold(
        self,
        traffic_light: Any,
        ego_route_s_m: float,
        now_monotonic: float,
        previous_state: Optional[str] = None,
    ) -> dict[str, Any]:
        traffic_light_id = int(traffic_light.id)
        if previous_state is None:
            previous_state = self.active_tl_state
        active_stop_s = self.active_stop_s
        active_stop_route_index = self.active_stop_route_index
        active_stop_point_source = self.active_stop_point_source
        stopped_for_tl_id = self.stopped_for_tl_id
        red_hold_active_before = self.red_hold_active
        stop_hold_cleared = bool(
            red_hold_active_before
            or stopped_for_tl_id is not None
            or active_stop_s is not None
            or active_stop_route_index is not None
        )

        self.previous_tl_state = previous_state
        self.active_tl_id = traffic_light_id
        self.active_tl_state = "Green"
        self.active_tl_last_seen_time = now_monotonic
        self.red_hold_active = False
        self.stopped_for_tl_id = None
        self.active_stop_route_index = None
        self.active_stop_s = None
        self.active_stop_point_source = None
        self._restrictive_traffic_lights.pop(traffic_light_id, None)
        self._last_restrictive_tl_candidate = None
        self._last_restrictive_tl_time = 0.0
        self._green_release_until[traffic_light_id] = (
            now_monotonic + self.green_release_grace_s
        )
        self._mark_post_tl_ignore(traffic_light_id, now_monotonic)
        self._last_green_clear_light_id = traffic_light_id
        self.last_tl_event = "traffic_light_green_clear"
        self.green_release_triggered = True
        self.red_hold_active_before = red_hold_active_before
        self.red_hold_active_after = self.red_hold_active
        self.stop_hold_cleared = stop_hold_cleared
        self.release_reason = "active_light_green_release"

        details = {
            "active_tl_id": traffic_light_id,
            "active_tl_state": "Green",
            "previous_tl_state": previous_state,
            "stopped_for_tl_id": None,
            "green_release_triggered": True,
            "red_hold_active_before": red_hold_active_before,
            "red_hold_active_after": False,
            "stop_hold_cleared": stop_hold_cleared,
            "release_reason": "active_light_green_release",
        }
        self._green_release_details[traffic_light_id] = details

        distance_to_stop_m = (
            max(0.0, float(active_stop_s) - ego_route_s_m)
            if active_stop_s is not None
            else 0.0
        )
        projection = {
            "route_index": int(active_stop_route_index or 0),
            "lateral_distance_m": 0.0,
        }
        candidate = self._candidate(
            "traffic_light_green_clear",
            traffic_light,
            projection,
            distance_to_stop_m,
            0.0,
            "active_light_green_release",
            traffic_light_state="Green",
            stop_buffer_m=self.traffic_light_stop_buffer_m,
            stop_point_source=active_stop_point_source,
            distance_is_buffered=True,
        )
        candidate.update(details)
        return candidate

    def _poll_active_traffic_light(
        self,
        world_actors: Any,
        ego_route_s_m: float,
        now_monotonic: float,
    ) -> Optional[dict[str, Any]]:
        tracked_tl_id = (
            self.stopped_for_tl_id
            if self.stopped_for_tl_id is not None
            else self.active_tl_id
        )
        if tracked_tl_id is None:
            return None

        traffic_light = self._find_actor_by_id(world_actors, tracked_tl_id)
        if traffic_light is None or not str(traffic_light.type_id).startswith(
            "traffic.traffic_light"
        ):
            return None

        previous_state = self.active_tl_state
        state = self._get_traffic_light_state(traffic_light)
        self.previous_tl_state = previous_state

        was_restrictive = (
            self.red_hold_active
            or self.stopped_for_tl_id == int(tracked_tl_id)
            or previous_state in ("Red", "Yellow")
            or self.last_tl_event
            in (
                "traffic_light_red_approach",
                "traffic_light_red_stop",
                "traffic_light_yellow_slow",
                "traffic_light_yellow_stop",
            )
        )
        if state == "Green" and was_restrictive:
            return self._release_active_traffic_light_hold(
                traffic_light,
                ego_route_s_m,
                now_monotonic,
                previous_state=previous_state,
            )
        self.active_tl_id = int(tracked_tl_id)
        self.active_tl_state = state
        self.active_tl_last_seen_time = now_monotonic
        return None

    def _traffic_light_locations(
        self,
        traffic_light,
    ) -> list[tuple[Any, Optional[float], str, Optional[int], Optional[int]]]:
        locations = []
        for method_name in ("get_stop_waypoints", "get_affected_lane_waypoints"):
            method = getattr(traffic_light, method_name, None)
            if not callable(method):
                continue
            try:
                for waypoint in method() or []:
                    locations.append((
                        waypoint.transform.location,
                        float(waypoint.transform.rotation.yaw),
                        (
                            "stop_waypoint"
                            if method_name == "get_stop_waypoints"
                            else "affected_lane_waypoint"
                        ),
                        int(waypoint.road_id),
                        int(waypoint.lane_id),
                    ))
            except Exception:
                continue

        try:
            transform = traffic_light.get_transform()
            trigger_location = transform.transform(traffic_light.trigger_volume.location)
            locations.append((
                trigger_location,
                float(transform.rotation.yaw),
                "trigger_volume",
                None,
                None,
            ))
        except Exception:
            pass

        try:
            transform = traffic_light.get_transform()
            locations.append((
                transform.location,
                float(transform.rotation.yaw),
                "actor_location_fallback",
                None,
                None,
            ))
        except Exception:
            pass
        return locations

    def build_tl_stop_fence(
        self,
        center_x: float,
        center_y: float,
        yaw_deg: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        perpendicular_yaw = math.radians(yaw_deg + 90.0)
        half_width = 0.5 * self.tl_fence_width_m
        offset_x = math.cos(perpendicular_yaw) * half_width
        offset_y = math.sin(perpendicular_yaw) * half_width
        return (
            (center_x - offset_x, center_y - offset_y),
            (center_x + offset_x, center_y + offset_y),
        )

    @staticmethod
    def _segment_intersection(
        first_start: tuple[float, float],
        first_end: tuple[float, float],
        second_start: tuple[float, float],
        second_end: tuple[float, float],
    ) -> Optional[tuple[float, float, float]]:
        p_x, p_y = first_start
        r_x = first_end[0] - p_x
        r_y = first_end[1] - p_y
        q_x, q_y = second_start
        s_x = second_end[0] - q_x
        s_y = second_end[1] - q_y
        cross_rs = r_x * s_y - r_y * s_x
        if abs(cross_rs) <= 1e-9:
            return None

        qmp_x = q_x - p_x
        qmp_y = q_y - p_y
        route_fraction = (qmp_x * s_y - qmp_y * s_x) / cross_rs
        fence_fraction = (qmp_x * r_y - qmp_y * r_x) / cross_rs
        if not (0.0 <= route_fraction <= 1.0 and 0.0 <= fence_fraction <= 1.0):
            return None
        return (
            p_x + route_fraction * r_x,
            p_y + route_fraction * r_y,
            route_fraction,
        )

    def intersect_route_with_stop_fence(
        self,
        points: list[dict[str, Any]],
        route_s: list[float],
        fence: tuple[tuple[float, float], tuple[float, float]],
        ego_route_s_m: float,
    ) -> Optional[dict[str, float | int]]:
        best = None
        for index in range(len(points) - 1):
            route_start = (
                float(points[index].get("x", 0.0)),
                float(points[index].get("y", 0.0)),
            )
            route_end = (
                float(points[index + 1].get("x", 0.0)),
                float(points[index + 1].get("y", 0.0)),
            )
            intersection = self._segment_intersection(
                route_start,
                route_end,
                fence[0],
                fence[1],
            )
            if intersection is None:
                continue

            segment_length = math.hypot(
                route_end[0] - route_start[0],
                route_end[1] - route_start[1],
            )
            intersection_s = route_s[index] + intersection[2] * segment_length
            distance_ahead = intersection_s - ego_route_s_m
            if distance_ahead < -self.green_ignore_after_pass_m:
                continue
            candidate = {
                "route_index": index if intersection[2] < 0.5 else index + 1,
                "route_s_m": intersection_s,
                "lateral_distance_m": 0.0,
                "projected_x": intersection[0],
                "projected_y": intersection[1],
            }
            rank = max(0.0, distance_ahead)
            if best is None or rank < best[0]:
                best = (rank, candidate)
        return best[1] if best is not None else None

    def compute_stop_profile_speed(
        self,
        current_speed_mps: float,
        distance_to_stop_m: float,
        cruise_speed_mps: Optional[float] = None,
    ) -> tuple[float, bool]:
        del current_speed_mps
        cruise_speed = (
            self.cruise_speed_mps
            if cruise_speed_mps is None
            else max(0.0, float(cruise_speed_mps))
        )
        effective_distance = max(
            0.0,
            distance_to_stop_m - self.tl_stop_line_buffer_m,
        )
        if effective_distance <= self.traffic_light_stop_distance_tolerance_m + 1e-6:
            return 0.0, True

        allowed_speed = math.sqrt(
            2.0 * max(0.1, self.tl_decel_max_mps2) * effective_distance
        )
        speed_limit = min(cruise_speed, allowed_speed)
        if effective_distance > self.tl_hard_stop_distance_m:
            profile_floor = (
                self.tl_slow_speed_mps
                if effective_distance > 2.0 * self.tl_hard_stop_distance_m
                else self.tl_min_profile_speed_mps
            )
            speed_limit = max(speed_limit, profile_floor)

        return max(0.0, speed_limit), False

    def red_stop_trigger_distance_m(self, current_speed_mps: float) -> float:
        raw_trigger_m = (
            max(0.0, float(current_speed_mps)) * self.red_stop_trigger_speed_gain_s
            + self.red_stop_trigger_speed_buffer_m
        )
        min_trigger_m = min(self.red_stop_trigger_base_m, self.red_stop_trigger_max_m)
        max_trigger_m = max(self.red_stop_trigger_base_m, self.red_stop_trigger_max_m)
        return min(max(raw_trigger_m, min_trigger_m), max_trigger_m)

    def _gate_allows_hard_stop(self, gate: TrafficLightStopGate) -> bool:
        if gate.fence_intersection_found and gate.confidence in ("high", "medium"):
            return True
        return (
            gate.distance_to_stop_m <= 1.0
            and gate.lateral_distance_m <= 1.5
        )

    def select_best_tl_gate(
        self,
        traffic_light,
        points: list[dict[str, Any]],
        route_s: list[float],
        ego_route_s_m: float,
    ) -> tuple[Optional[TrafficLightStopGate], list[str]]:
        best = None
        rejection_reasons = []
        for location, lane_yaw, source, road_id, lane_id in self._traffic_light_locations(
            traffic_light
        ):
            center_projection = project_actor_to_route(
                points,
                float(location.x),
                float(location.y),
                route_s,
            )
            if center_projection is None:
                rejection_reasons.append(f"{source}:projection_failed")
                continue

            route_index = int(center_projection["route_index"])
            route_yaw = float(points[route_index].get("yaw", 0.0))
            route_road_id = points[route_index].get("road_id")
            route_lane_id = points[route_index].get("lane_id")
            if (
                lane_yaw is None
                and source in ("stop_waypoint", "affected_lane_waypoint")
                and self._world is not None
            ):
                try:
                    waypoint = self._world.get_map().get_waypoint(location)
                    lane_yaw = float(waypoint.transform.rotation.yaw)
                    road_id = int(waypoint.road_id)
                    lane_id = int(waypoint.lane_id)
                except Exception:
                    lane_yaw = None
            if lane_yaw is None and self._world is not None:
                try:
                    waypoint = self._world.get_map().get_waypoint(
                        location,
                        project_to_road=True,
                        lane_type=self._carla.LaneType.Driving,
                    )
                    lane_yaw = float(waypoint.transform.rotation.yaw)
                    road_id = int(waypoint.road_id)
                    lane_id = int(waypoint.lane_id)
                except Exception:
                    lane_yaw = None
            road_lane_match = None
            if (
                road_id is not None
                and lane_id is not None
                and route_road_id is not None
                and route_lane_id is not None
            ):
                road_lane_match = (
                    int(road_id) == int(route_road_id)
                    and int(lane_id) == int(route_lane_id)
                )
            candidate_heading = lane_yaw
            candidate_yaw_diff = (
                self._angle_difference_deg(candidate_heading, route_yaw)
                if candidate_heading is not None
                else None
            )
            candidate_heading_ok = (
                candidate_yaw_diff is not None
                and candidate_yaw_diff <= 45.0
            )
            if candidate_yaw_diff is not None and candidate_yaw_diff > 90.0:
                rejection_reasons.append("rejected_opposite_direction_light")
                continue
            if not candidate_heading_ok:
                rejection_reasons.append(f"{source}:direction_mismatch")
                continue

            fence_yaw = lane_yaw if lane_yaw is not None else route_yaw
            fence = self.build_tl_stop_fence(
                float(location.x),
                float(location.y),
                fence_yaw,
            )
            fence_projection = self.intersect_route_with_stop_fence(
                points,
                route_s,
                fence,
                ego_route_s_m,
            )
            fence_intersection_found = fence_projection is not None
            projection = fence_projection or center_projection
            selected_source = (
                f"{source}_fence"
                if fence_intersection_found and source != "actor_location_fallback"
                else (
                    source
                    if source == "actor_location_fallback"
                    else f"{source}_route_projection_fallback"
                )
            )
            raw_stop_distance_m = float(projection["route_s_m"]) - ego_route_s_m
            candidate_cross_track_m = float(center_projection["lateral_distance_m"])
            candidate_on_route_corridor = (
                candidate_cross_track_m <= self.traffic_light_lateral_margin_m
            )
            candidate_ahead_ok = (
                0.0 <= raw_stop_distance_m <= self.traffic_light_horizon_m
            )
            if raw_stop_distance_m < 0.0:
                rejection_reasons.append("rejected_not_ahead_light")
                continue
            if raw_stop_distance_m > self.traffic_light_horizon_m:
                rejection_reasons.append(f"{source}:beyond_horizon")
                continue
            if not candidate_on_route_corridor:
                rejection_reasons.append("rejected_out_of_route_corridor")
                continue
            if road_lane_match is False:
                rejection_reasons.append(f"{source}:road_lane_mismatch")
                continue

            source_priority = {
                "stop_waypoint_fence": 0,
                "affected_lane_waypoint_fence": 1,
                "trigger_volume_fence": 2,
                "stop_waypoint_route_projection_fallback": 3,
                "affected_lane_waypoint_route_projection_fallback": 4,
                "trigger_volume_route_projection_fallback": 5,
                "actor_location_fallback": 6,
            }.get(selected_source, 99)
            rank = (
                source_priority,
                0 if road_lane_match is True else 1,
                max(0.0, raw_stop_distance_m),
                candidate_cross_track_m,
            )
            if best is None or rank < best[0]:
                best = (
                    rank,
                    projection,
                    raw_stop_distance_m,
                    selected_source,
                    (
                        SimpleNamespace(
                            x=float(projection["projected_x"]),
                            y=float(projection["projected_y"]),
                            z=float(location.z),
                        )
                        if fence_intersection_found
                        else location
                    ),
                    road_lane_match,
                    fence_intersection_found,
                    float(center_projection["lateral_distance_m"]),
                    road_id,
                    lane_id,
                    route_road_id,
                    route_lane_id,
                    route_yaw,
                    candidate_heading,
                    candidate_yaw_diff,
                    candidate_cross_track_m,
                    raw_stop_distance_m,
                    candidate_on_route_corridor,
                    candidate_heading_ok,
                    candidate_ahead_ok,
                )

        if best is None:
            return None, rejection_reasons

        light_projection = None
        try:
            light_location = traffic_light.get_location()
            light_projection = project_actor_to_route(
                points,
                float(light_location.x),
                float(light_location.y),
                route_s,
            )
        except Exception:
            pass

        distance_to_light_m = None
        if light_projection is not None:
            distance_to_light_m = (
                float(light_projection["route_s_m"]) - ego_route_s_m
            )

        raw_stop_distance_m = float(best[2])
        adjusted_stop_s = (
            float(best[1]["route_s_m"])
            + self.traffic_light_stop_anchor_forward_offset_m
        )
        center_to_stopline_m = adjusted_stop_s - ego_route_s_m
        front_bumper_to_stopline_m = (
            center_to_stopline_m - self.traffic_light_stop_front_bumper_offset_m
        )
        distance_to_stop_m = max(0.0, front_bumper_to_stopline_m)
        source = str(best[3])
        if not best[6]:
            confidence = "low"
        else:
            confidence = {
                "stop_waypoint_fence": "high",
                "affected_lane_waypoint_fence": (
                    "high" if best[5] is True else "medium"
                ),
                "trigger_volume_fence": "medium",
                "actor_location_fallback": "low",
            }.get(source, "low")
        location = best[4]
        ego_center_x = None
        ego_center_y = None
        ego_front_x = None
        ego_front_y = None
        if isinstance(self._last_status, dict):
            ego_loc = self._last_status.get("location", {})
            ego_rot = self._last_status.get("rotation", {})
            ego_center_x = float(ego_loc.get("x", 0.0))
            ego_center_y = float(ego_loc.get("y", 0.0))
            ego_yaw_rad = math.radians(float(ego_rot.get("yaw", 0.0)))
            ego_front_x = (
                ego_center_x
                + math.cos(ego_yaw_rad) * self.traffic_light_stop_front_bumper_offset_m
            )
            ego_front_y = (
                ego_center_y
                + math.sin(ego_yaw_rad) * self.traffic_light_stop_front_bumper_offset_m
            )
        gate = TrafficLightStopGate(
            light_id=int(traffic_light.id),
            light_state="Unknown",
            stop_point_source=source,
            stop_x=float(location.x),
            stop_y=float(location.y),
            stop_z=float(location.z),
            stop_route_index=int(best[1]["route_index"]),
            stop_s=adjusted_stop_s,
            distance_to_stop_m=distance_to_stop_m,
            distance_to_light_m=distance_to_light_m,
            lateral_distance_m=float(best[7]),
            confidence=confidence,
            road_lane_match=best[5],
            fence_intersection_found=bool(best[6]),
            projection=best[1],
            center_to_stopline_m=center_to_stopline_m,
            front_bumper_to_stopline_m=front_bumper_to_stopline_m,
            ego_center_x=round(ego_center_x, 3) if ego_center_x is not None else None,
            ego_center_y=round(ego_center_y, 3) if ego_center_y is not None else None,
            ego_front_x=round(ego_front_x, 3) if ego_front_x is not None else None,
            ego_front_y=round(ego_front_y, 3) if ego_front_y is not None else None,
            applied_stop_line_buffer_m=self.tl_stop_line_buffer_m,
            stop_anchor_forward_offset_m=self.traffic_light_stop_anchor_forward_offset_m,
            stopline_road_id=int(best[8]) if best[8] is not None else None,
            stopline_lane_id=int(best[9]) if best[9] is not None else None,
            ego_road_id=int(best[10]) if best[10] is not None else None,
            ego_lane_id=int(best[11]) if best[11] is not None else None,
            same_lane_or_compatible=(
                bool(best[5])
                if best[5] is not None
                else None
            ),
            route_heading_deg=float(best[12]) if best[12] is not None else None,
            candidate_heading_deg=float(best[13]) if best[13] is not None else None,
            candidate_yaw_diff_deg=float(best[14]) if best[14] is not None else None,
            candidate_cross_track_m=float(best[15]) if best[15] is not None else None,
            candidate_along_track_m=float(best[16]) if best[16] is not None else None,
            candidate_on_route_corridor=bool(best[17]),
            candidate_heading_ok=bool(best[18]),
            candidate_ahead_ok=bool(best[19]),
        )
        return gate, rejection_reasons

    def _analyze(self) -> tuple[dict[str, Any], dict[str, Any]]:
        now = time.time()
        self.green_release_triggered = False
        self.red_hold_active_before = self.red_hold_active
        self.red_hold_active_after = self.red_hold_active
        self.stop_hold_cleared = False
        self.release_reason = None
        route_age_s = now - self._last_route_time if self._last_route_time else None
        status_age_s = now - self._last_status_time if self._last_status_time else None
        debug = {
            "stamp": now,
            "route_age_s": route_age_s,
            "status_age_s": status_age_s,
            "map": self._last_status.get("map") if isinstance(self._last_status, dict) else None,
            "ego_id": self._last_status.get("ego_id") if isinstance(self._last_status, dict) else None,
            "ego_location": (
                self._last_status.get("location", {})
                if isinstance(self._last_status, dict)
                else {}
            ),
            "route_point_count": (
                len(self._last_route.get("points", []))
                if isinstance(self._last_route, dict)
                else 0
            ),
            "nearest_route_index": None,
            "total_actor_count": None,
            "traffic_light_actor_count": None,
            "traffic_lights": [],
            "traffic_light_system_enabled": self.enable_traffic_light_events,
            "traffic_light_system_reason": None,
            "warning_payload": None,
            "actors_checked": 0,
            "actors_in_route_corridor": 0,
            "vehicles_checked": 0,
            "pedestrians_checked": 0,
            "traffic_lights_checked": 0,
            "traffic_lights_in_route_corridor": 0,
            "rejected_lights_count": 0,
            "traffic_light_rejection_reasons": {},
            "candidates": 0,
            "ego_found": False,
            "route_source": (
                self._last_route.get("route_source")
                if isinstance(self._last_route, dict)
                else None
            ),
            "active_tl_id": self.active_tl_id,
            "active_tl_state": self.active_tl_state,
            "active_light_id": self.active_tl_id,
            "active_light_state": self.active_tl_state,
            "previous_tl_state": self.previous_tl_state,
            "active_stop_s": self.active_stop_s,
            "active_stop_route_index": self.active_stop_route_index,
            "active_stop_point_source": self.active_stop_point_source,
            "red_hold_active": self.red_hold_active,
            "stopped_for_tl_id": self.stopped_for_tl_id,
            "green_release_triggered": self.green_release_triggered,
            "red_hold_active_before": self.red_hold_active_before,
            "red_hold_active_after": self.red_hold_active_after,
            "stop_hold_cleared": self.stop_hold_cleared,
            "release_reason": self.release_reason,
            "last_tl_event": self.last_tl_event,
            "front_bumper_to_stopline_m": None,
            "center_to_stopline_m": None,
            "passed_stopline": False,
            "passed_stopline_threshold_m": self.traffic_light_passed_stopline_threshold_m,
            "post_green_ignore_active": False,
            "post_green_ignore_light_id": None,
            "post_tl_ignore_active": False,
            "same_light_ignore_active": False,
            "last_green_clear_light_id": self._last_green_clear_light_id,
            "stale_red_stop_suppressed": False,
            "stale_red_stop_suppress_reason": None,
            "last_restrictive_light_id": (
                self._last_restrictive_tl_candidate.get("traffic_light_id")
                if self._last_restrictive_tl_candidate is not None
                else None
            ),
            "current_candidate_light_id": None,
            "replay_candidate_light_id": None,
            "old_light_replay_suppressed": False,
        }

        if self._world is None:
            return self._base_event(False, "carla_unavailable"), debug
        if self._last_status is None or status_age_s is None or status_age_s > 1.0:
            return self._base_event(False, "status_missing_or_stale"), debug
        if self._last_route is None or route_age_s is None or route_age_s > self.stale_route_timeout_s:
            return self._base_event(False, "route_missing_or_stale"), debug

        points = self._last_route.get("points", [])
        if not self._last_route.get("route_ok", False) or len(points) < 2:
            return self._base_event(False, "route_invalid"), debug

        ego_vehicle = self._find_ego_vehicle()
        debug["ego_found"] = ego_vehicle is not None
        if ego_vehicle is None:
            return self._base_event(False, "ego_vehicle_not_found"), debug

        ego_location = self._last_status.get("location", {})
        route_s = cumulative_route_s(points)
        ego_projection = project_actor_to_route(
            points,
            float(ego_location.get("x", 0.0)),
            float(ego_location.get("y", 0.0)),
            route_s,
        )
        if ego_projection is None:
            return self._base_event(False, "ego_route_projection_failed"), debug
        debug["nearest_route_index"] = int(ego_projection["route_index"])

        candidates: list[dict[str, Any]] = []
        world_actors = self._world.get_actors()
        self._emit_startup_tl_diagnostics()
        debug["total_actor_count"] = len(world_actors)
        actors = list(world_actors.filter("vehicle.*"))
        actors.extend(world_actors.filter("walker.pedestrian.*"))

        for actor in actors:
            if actor.id == ego_vehicle.id:
                continue
            debug["actors_checked"] += 1
            if actor.type_id.startswith("vehicle."):
                debug["vehicles_checked"] += 1
            elif actor.type_id.startswith("walker.pedestrian."):
                debug["pedestrians_checked"] += 1

            try:
                location = actor.get_location()
                projection = project_actor_to_route(
                    points,
                    float(location.x),
                    float(location.y),
                    route_s,
                )
            except Exception:
                continue

            if projection is None:
                continue
            lateral_distance = float(projection["lateral_distance_m"])
            if lateral_distance > self.route_lateral_margin_m:
                continue

            distance_ahead = (
                float(projection["route_s_m"])
                - float(ego_projection["route_s_m"])
            )
            if not is_ahead_on_route(
                float(ego_projection["route_s_m"]),
                float(projection["route_s_m"]),
                tolerance_m=0.25,
            ):
                continue
            distance_ahead = max(0.0, distance_ahead)
            if distance_ahead > self.event_horizon_m:
                continue

            debug["actors_in_route_corridor"] += 1
            if actor.type_id.startswith("walker.pedestrian."):
                if distance_ahead <= self.pedestrian_stop_distance_m:
                    candidates.append(
                        self._candidate(
                            "pedestrian_stop",
                            actor,
                            projection,
                            distance_ahead,
                            0.0,
                            "pedestrian_ahead_in_route_corridor",
                        )
                    )
                continue

            if not actor.type_id.startswith("vehicle."):
                continue

            try:
                velocity = actor.get_velocity()
                front_speed_mps = math.sqrt(
                    velocity.x * velocity.x
                    + velocity.y * velocity.y
                    + velocity.z * velocity.z
                )
            except Exception:
                continue

            stopped_lead_vehicle = (
                front_speed_mps <= self.stopped_vehicle_speed_mps
                and distance_ahead <= self.stopped_vehicle_stop_distance_m
            )
            if stopped_lead_vehicle:
                candidates.append(
                    self._candidate(
                        "vehicle_stop",
                        actor,
                        projection,
                        distance_ahead,
                        0.0,
                        "stopped_lead_vehicle_ahead",
                        actor_speed_mps=front_speed_mps,
                    )
                )
                continue

            if distance_ahead <= self.vehicle_stop_distance_m:
                candidates.append(
                    self._candidate(
                        "vehicle_stop",
                        actor,
                        projection,
                        distance_ahead,
                        0.0,
                        "vehicle_inside_stop_distance",
                        actor_speed_mps=front_speed_mps,
                    )
                )
                continue

            available_gap_m = max(
                0.0,
                distance_ahead - self.vehicle_follow_distance_m,
            )
            safe_follow_speed_mps = (
                front_speed_mps
                + available_gap_m / max(0.1, self.follow_time_gap_s)
            )
            candidates.append(
                self._candidate(
                    "vehicle_follow",
                    actor,
                    projection,
                    distance_ahead,
                    safe_follow_speed_mps,
                    "lead_vehicle_ahead_in_route_corridor",
                    actor_speed_mps=front_speed_mps,
                )
            )

        if self.enable_traffic_light_events:
            now_monotonic = time.monotonic()
            traffic_light_candidates: list[dict[str, Any]] = []
            traffic_light_contexts: dict[
                int,
                tuple[TrafficLightStopGate, str],
            ] = {}
            active_tl_seen = False
            ego_speed_mps = float(self._last_status.get("speed_mps", 0.0))
            traffic_lights = list(world_actors.filter("traffic.traffic_light*"))
            debug["traffic_light_actor_count"] = len(traffic_lights)
            debug["traffic_lights"] = [
                self._traffic_light_snapshot(traffic_light)
                for traffic_light in traffic_lights
            ]
            green_release_candidate = self._poll_active_traffic_light(
                world_actors,
                float(ego_projection["route_s_m"]),
                now_monotonic,
            )
            if green_release_candidate is not None:
                traffic_light_candidates.append(green_release_candidate)
                active_tl_seen = True
            if not traffic_lights:
                debug["traffic_light_system_reason"] = "no_carla_traffic_light_actors"
                debug["warning_payload"] = self._build_warning_payload(
                    reason="no_carla_traffic_light_actors",
                    map_name=debug["map"],
                    ego_location=debug["ego_location"],
                    traffic_light_actor_count=0,
                )
            for traffic_light in traffic_lights:
                debug["actors_checked"] += 1
                debug["traffic_lights_checked"] += 1

                gate, rejection_reasons = self.select_best_tl_gate(
                    traffic_light,
                    points,
                    route_s,
                    float(ego_projection["route_s_m"]),
                )
                if gate is None:
                    debug["rejected_lights_count"] += 1
                    debug["candidate_light_id"] = int(traffic_light.id)
                    debug["current_candidate_light_id"] = int(traffic_light.id)
                    debug["candidate_rejected"] = True
                    debug["candidate_reject_reason"] = (
                        sorted(set(rejection_reasons))[0]
                        if rejection_reasons
                        else "rejected_no_valid_stop_gate"
                    )
                    for reason in set(rejection_reasons):
                        reason_counts = debug["traffic_light_rejection_reasons"]
                        reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    continue

                projection = gate.projection
                distance_to_stop_m = gate.distance_to_stop_m
                signed_stop_distance_m = (
                    gate.stop_s - float(ego_projection["route_s_m"])
                )
                distance_to_light_m = gate.distance_to_light_m
                stop_point_source = gate.stop_point_source
                debug["actors_in_route_corridor"] += 1
                debug["traffic_lights_in_route_corridor"] += 1
                state = self._get_traffic_light_state(traffic_light)
                gate.light_state = state
                traffic_light_id = int(traffic_light.id)
                passed_stopline = self._gate_passed_stopline(gate)
                if passed_stopline:
                    self._mark_post_tl_ignore(traffic_light_id, now_monotonic)
                same_light_ignore_active, same_light_ignore_reason = (
                    self._same_light_ignore_active(
                        traffic_light_id,
                        gate,
                        now_monotonic,
                    )
                )
                if (
                    state in ("Red", "Yellow")
                    and same_light_ignore_active
                ):
                    debug["rejected_lights_count"] += 1
                    reason_counts = debug["traffic_light_rejection_reasons"]
                    reason = same_light_ignore_reason or "post_tl_ignore_same_light"
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                    debug["stale_red_stop_suppressed"] = True
                    debug["stale_red_stop_suppress_reason"] = reason
                    debug["passed_stopline"] = passed_stopline
                    debug["post_green_ignore_active"] = now_monotonic <= self._post_green_ignore_until.get(
                        traffic_light_id,
                        0.0,
                    )
                    debug["post_green_ignore_light_id"] = (
                        traffic_light_id if debug["post_green_ignore_active"] else None
                    )
                    debug["post_tl_ignore_active"] = now_monotonic <= self._post_tl_ignore_until.get(
                        traffic_light_id,
                        0.0,
                    )
                    debug["same_light_ignore_active"] = same_light_ignore_active
                    debug["last_green_clear_light_id"] = self._last_green_clear_light_id
                    if reason == "passed_stopline_clear":
                        self._clear_active_gate()
                        self.last_tl_event = "clear"
                    continue
                if signed_stop_distance_m < 0.0 and not (
                    self.red_hold_active
                    and traffic_light_id == self.stopped_for_tl_id
                ):
                    if signed_stop_distance_m <= -self.traffic_light_passed_ignore_distance_m:
                        self._mark_post_tl_ignore(traffic_light_id, now_monotonic)
                    if (
                        traffic_light_id == self.active_tl_id
                        and signed_stop_distance_m
                        <= -self.green_ignore_after_pass_m
                    ):
                        self._clear_active_gate()
                        self.last_tl_event = "clear"
                    continue
                if traffic_light_id == self.active_tl_id:
                    active_tl_seen = True
                    if state == "Green":
                        self.active_tl_state = state
                        self.active_tl_last_seen_time = now_monotonic
                    else:
                        self._set_active_gate(gate, state, now_monotonic)

                candidate = None
                if state == "Red":
                    self._restrictive_traffic_lights[traffic_light_id] = now_monotonic
                    if (
                        self.red_hold_active
                        and traffic_light_id == self.stopped_for_tl_id
                    ):
                        candidate = self._candidate(
                            "traffic_light_red_stop",
                            traffic_light,
                            projection,
                            distance_to_stop_m,
                            0.0,
                            "red_light_hold",
                            traffic_light_state=state,
                            stop_buffer_m=self.tl_stop_line_buffer_m,
                            distance_to_light_m=distance_to_light_m,
                            stop_point_source=stop_point_source,
                            confidence=gate.confidence,
                            gate=gate,
                            distance_is_buffered=False,
                        )
                    elif distance_to_stop_m <= self.tl_profile_horizon_m:
                        red_stop_trigger_m = self.red_stop_trigger_distance_m(ego_speed_mps)
                        red_stop_triggered_by_distance = distance_to_stop_m <= red_stop_trigger_m
                        profile_speed, profile_stop = self.compute_stop_profile_speed(
                            ego_speed_mps,
                            distance_to_stop_m,
                            self.cruise_speed_mps,
                        )
                        hard_stop_allowed = self._gate_allows_hard_stop(gate)
                        if red_stop_triggered_by_distance or (profile_stop and hard_stop_allowed):
                            candidate = self._candidate(
                                "traffic_light_red_stop",
                                traffic_light,
                                projection,
                                distance_to_stop_m,
                                0.0,
                                (
                                    "red_light_distance_stop"
                                    if red_stop_triggered_by_distance
                                    else "red_light_stopline_reached"
                                ),
                                traffic_light_state=state,
                                stop_buffer_m=self.tl_stop_line_buffer_m,
                                distance_to_light_m=distance_to_light_m,
                                stop_point_source=stop_point_source,
                                confidence=gate.confidence,
                                gate=gate,
                                distance_is_buffered=False,
                                current_speed_mps=ego_speed_mps,
                                red_stop_trigger_m=red_stop_trigger_m,
                                red_stop_triggered_by_distance=red_stop_triggered_by_distance,
                            )
                        else:
                            candidate = self._candidate(
                                "traffic_light_red_approach",
                                traffic_light,
                                projection,
                                distance_to_stop_m,
                                max(profile_speed, self.tl_min_profile_speed_mps),
                                "red_light_profile_approach",
                                traffic_light_state=state,
                                stop_buffer_m=self.tl_stop_line_buffer_m,
                                distance_to_light_m=distance_to_light_m,
                                stop_point_source=stop_point_source,
                                confidence=gate.confidence,
                                gate=gate,
                                distance_is_buffered=False,
                                current_speed_mps=ego_speed_mps,
                                red_stop_trigger_m=red_stop_trigger_m,
                                red_stop_triggered_by_distance=False,
                            )

                elif state == "Yellow":
                    yellow_can_pass = (
                        ego_speed_mps > 0.2
                        and distance_to_stop_m / ego_speed_mps
                        < self.yellow_pass_time_s
                    )
                    if yellow_can_pass:
                        candidate = self._candidate(
                            "clear",
                            traffic_light,
                            projection,
                            distance_to_stop_m,
                            0.0,
                            "yellow_can_pass",
                            traffic_light_state=state,
                            stop_buffer_m=self.tl_stop_line_buffer_m,
                            distance_to_light_m=distance_to_light_m,
                            stop_point_source=stop_point_source,
                            confidence=gate.confidence,
                            gate=gate,
                            distance_is_buffered=False,
                        )
                    else:
                        self._restrictive_traffic_lights[traffic_light_id] = (
                            now_monotonic
                        )
                        profile_speed, profile_stop = self.compute_stop_profile_speed(
                            ego_speed_mps,
                            distance_to_stop_m,
                            self.cruise_speed_mps,
                        )
                    if (
                        not yellow_can_pass
                        and profile_stop
                        and self._gate_allows_hard_stop(gate)
                    ):
                        candidate = self._candidate(
                            "traffic_light_yellow_stop",
                            traffic_light,
                            projection,
                            distance_to_stop_m,
                            0.0,
                            "yellow_too_close_stop",
                            traffic_light_state=state,
                            stop_buffer_m=self.tl_stop_line_buffer_m,
                            distance_to_light_m=distance_to_light_m,
                            stop_point_source=stop_point_source,
                            confidence=gate.confidence,
                            gate=gate,
                            distance_is_buffered=False,
                        )
                    elif not yellow_can_pass and distance_to_stop_m <= self.tl_profile_horizon_m:
                        candidate = self._candidate(
                            "traffic_light_yellow_slow",
                            traffic_light,
                            projection,
                            distance_to_stop_m,
                            profile_speed,
                            "yellow_light_profile_slow",
                            traffic_light_state=state,
                            stop_buffer_m=self.tl_stop_line_buffer_m,
                            distance_to_light_m=distance_to_light_m,
                            stop_point_source=stop_point_source,
                            confidence=gate.confidence,
                            gate=gate,
                            distance_is_buffered=False,
                        )
                elif state == "Green":
                    last_restrictive = self._restrictive_traffic_lights.pop(
                        traffic_light_id,
                        None,
                    )
                    same_held_light = (
                        traffic_light_id == self.stopped_for_tl_id
                        or traffic_light_id == self.active_tl_id
                    )
                    restrictive_memory_active = (
                        self.red_hold_active
                        or self.stopped_for_tl_id == traffic_light_id
                        or self.last_tl_event
                        in (
                            "traffic_light_red_approach",
                            "traffic_light_red_stop",
                            "traffic_light_yellow_slow",
                            "traffic_light_yellow_stop",
                        )
                    )
                    if same_held_light and restrictive_memory_active:
                        if not self.green_release_triggered:
                            candidate = self._release_active_traffic_light_hold(
                                traffic_light,
                                float(ego_projection["route_s_m"]),
                                now_monotonic,
                            )
                        active_tl_seen = True
                    if (
                        same_held_light
                        and restrictive_memory_active
                    ) or (
                        last_restrictive is not None
                        and distance_to_stop_m <= self.green_release_distance_m
                    ):
                        self._green_release_until[traffic_light_id] = (
                            now_monotonic + self.green_release_grace_s
                        )
                        self._mark_post_tl_ignore(traffic_light_id, now_monotonic)
                        self._last_green_clear_light_id = traffic_light_id
                    if (
                        candidate is None
                        and now_monotonic <= self._green_release_until.get(
                        traffic_light_id,
                        0.0,
                        )
                    ):
                        candidate = self._candidate(
                            "traffic_light_green_clear",
                            traffic_light,
                            projection,
                            distance_to_stop_m,
                            0.0,
                            "active_light_green_release",
                            traffic_light_state=state,
                            stop_buffer_m=self.traffic_light_stop_buffer_m,
                            distance_to_light_m=distance_to_light_m,
                            stop_point_source=stop_point_source,
                            confidence=gate.confidence,
                            gate=gate,
                            distance_is_buffered=True,
                        )
                        candidate.update(
                            self._green_release_details.get(
                                traffic_light_id,
                                {
                                    "active_tl_id": traffic_light_id,
                                    "active_tl_state": "Green",
                                    "previous_tl_state": self.previous_tl_state,
                                    "stopped_for_tl_id": None,
                                    "green_release_triggered": False,
                                    "red_hold_active_before": False,
                                    "red_hold_active_after": False,
                                    "stop_hold_cleared": False,
                                    "release_reason": "active_light_green_release",
                                },
                            )
                        )

                if candidate is not None:
                    debug["current_candidate_light_id"] = candidate.get("traffic_light_id")
                    candidate, suppressed, suppress_reason = (
                        self._suppress_stale_restrictive_candidate(
                            candidate,
                            now_monotonic,
                        )
                    )
                    if suppressed:
                        debug["stale_red_stop_suppressed"] = True
                        debug["stale_red_stop_suppress_reason"] = suppress_reason
                        debug["passed_stopline"] = bool(candidate.get("passed_stopline", False))
                        debug["post_green_ignore_active"] = bool(
                            candidate.get("post_green_ignore_active", False)
                        )
                        debug["post_green_ignore_light_id"] = candidate.get(
                            "post_green_ignore_light_id"
                        )
                        debug["post_tl_ignore_active"] = bool(
                            candidate.get("post_tl_ignore_active", False)
                        )
                        debug["same_light_ignore_active"] = bool(
                            candidate.get("same_light_ignore_active", False)
                        )
                        debug["last_green_clear_light_id"] = self._last_green_clear_light_id
                        if suppress_reason in (
                            "passed_stopline_clear",
                            "stale_red_hold_passed_stopline_clear",
                        ):
                            debug["old_light_replay_suppressed"] = True
                    traffic_light_candidates.append(candidate)
                    if self._is_restrictive_tl_event(candidate):
                        traffic_light_contexts[traffic_light_id] = (gate, state)

            if traffic_light_candidates:
                restrictive_events = {
                    "traffic_light_red_approach",
                    "traffic_light_red_stop",
                    "traffic_light_yellow_slow",
                    "traffic_light_yellow_stop",
                }
                restrictive_candidates = [
                    item
                    for item in traffic_light_candidates
                    if item["event"] in restrictive_events
                ]
                tracking_candidate = min(
                    restrictive_candidates or traffic_light_candidates,
                    key=lambda item: float(item["distance_m"]),
                )
                tracking_tl_id = tracking_candidate.get("traffic_light_id")
                tracking_context = (
                    traffic_light_contexts.get(int(tracking_tl_id))
                    if tracking_tl_id is not None
                    else None
                )
                if (
                    tracking_context is not None
                    and tracking_candidate["event"] in restrictive_events
                ):
                    tracking_gate, tracking_state = tracking_context
                    self._set_active_gate(
                        tracking_gate,
                        tracking_state,
                        now_monotonic,
                    )
                    active_tl_seen = True
                    if tracking_candidate["event"] in (
                        "traffic_light_red_stop",
                        "traffic_light_yellow_stop",
                    ):
                        self.stopped_for_tl_id = tracking_gate.light_id
                    if (
                        tracking_candidate["event"]
                        in (
                            "traffic_light_red_stop",
                            "traffic_light_yellow_stop",
                        )
                        and tracking_gate.distance_to_stop_m
                        <= self.stopline_reached_distance_m
                        and ego_speed_mps <= self.stopped_speed_threshold_mps
                        and self._gate_allows_hard_stop(tracking_gate)
                    ):
                        self.red_hold_active = True
                        self.stopped_for_tl_id = tracking_gate.light_id
                        tracking_candidate["target_speed_limit_mps"] = 0.0
                        tracking_candidate["stop_required"] = True
                        tracking_candidate["reason"] = (
                            "red_light_hold"
                            if tracking_state == "Red"
                            else "yellow_light_hold"
                        )

                if restrictive_candidates:
                    self._last_restrictive_tl_candidate = dict(
                        tracking_candidate
                    )
                    self._last_restrictive_tl_time = now_monotonic
                else:
                    self._last_restrictive_tl_candidate = None
                    self._last_restrictive_tl_time = 0.0
                candidates.extend(traffic_light_candidates)
                self.last_tl_event = str(tracking_candidate["event"])
            elif (
                self.red_hold_active
                and self._last_restrictive_tl_candidate is not None
                and not active_tl_seen
                and now_monotonic - self.active_tl_last_seen_time
                <= self.tl_lost_grace_s
            ):
                held_candidate = dict(self._last_restrictive_tl_candidate)
                held_candidate["event"] = "traffic_light_red_stop"
                held_candidate["target_speed_limit_mps"] = 0.0
                held_candidate["stop_required"] = True
                held_candidate["reason"] = "red_light_hold"
                debug["replay_candidate_light_id"] = held_candidate.get("traffic_light_id")
                held_candidate, suppressed, suppress_reason = (
                    self._suppress_stale_restrictive_candidate(
                        held_candidate,
                        now_monotonic,
                    )
                )
                if suppressed:
                    if suppress_reason in (
                        "passed_stopline_clear",
                        "stale_red_hold_passed_stopline_clear",
                    ):
                        held_candidate["reason"] = "passed_stopline_old_light_replay"
                        held_candidate["stale_red_stop_suppress_reason"] = (
                            "passed_stopline_old_light_replay"
                        )
                        suppress_reason = "passed_stopline_old_light_replay"
                    debug["stale_red_stop_suppressed"] = True
                    debug["stale_red_stop_suppress_reason"] = suppress_reason
                    debug["old_light_replay_suppressed"] = True
                candidates.append(held_candidate)
            elif (
                self._last_restrictive_tl_candidate is not None
                and now_monotonic - self._last_restrictive_tl_time
                <= self.tl_lost_grace_s
            ):
                replay_candidate = dict(self._last_restrictive_tl_candidate)
                debug["replay_candidate_light_id"] = replay_candidate.get("traffic_light_id")
                replay_candidate, suppressed, suppress_reason = (
                    self._suppress_stale_restrictive_candidate(
                        replay_candidate,
                        now_monotonic,
                    )
                )
                if suppressed:
                    if suppress_reason in (
                        "passed_stopline_clear",
                        "stale_red_hold_passed_stopline_clear",
                    ):
                        replay_candidate["reason"] = "passed_stopline_old_light_replay"
                        replay_candidate["stale_red_stop_suppress_reason"] = (
                            "passed_stopline_old_light_replay"
                        )
                        suppress_reason = "passed_stopline_old_light_replay"
                    debug["stale_red_stop_suppressed"] = True
                    debug["stale_red_stop_suppress_reason"] = suppress_reason
                    debug["old_light_replay_suppressed"] = True
                candidates.append(replay_candidate)
            else:
                self._last_restrictive_tl_candidate = None
                if (
                    self.red_hold_active
                    and now_monotonic - self.active_tl_last_seen_time
                    > self.tl_lost_grace_s
                ):
                    self._clear_active_gate()
                    self.last_tl_event = "clear"
        else:
            debug["traffic_light_system_reason"] = "traffic_light_events_disabled"

        debug["candidates"] = len(candidates)
        if not candidates:
            no_event_reason = (
                "no_carla_traffic_light_actors"
                if (
                    self.enable_traffic_light_events
                    and debug.get("traffic_light_actor_count") == 0
                )
                else "route_corridor_clear"
            )
            debug["event_decision"] = "none"
            debug["no_event_reason"] = no_event_reason
            return self._base_event(True, no_event_reason), debug

        priority = {
            "pedestrian_stop": 0,
            "traffic_light_red_stop": 1,
            "vehicle_stop": 2,
            "traffic_light_yellow_stop": 3,
            "traffic_light_green_clear": 4,
            "traffic_light_green_release": 4,
            "traffic_light_red_approach": 5,
            "traffic_light_yellow_slow": 6,
            "vehicle_follow": 7,
            "clear": 8,
        }
        selected = min(
            candidates,
            key=lambda item: (
                priority.get(item["event"], 99),
                float(item["distance_m"]),
            ),
        )
        payload = self._base_event(True, selected["reason"])
        payload.update(selected)
        if str(payload["event"]).startswith("traffic_light_"):
            self.last_tl_event = str(payload["event"])
        debug.update({
            "event_decision": payload["event"],
            "no_event_reason": None,
            "selected_event": payload["event"],
            "selected_actor_id": payload["actor_id"],
            "selected_traffic_light_state": payload["traffic_light_state"],
            "selected_actor_speed_mps": payload["actor_speed_mps"],
            "selected_distance_m": payload["distance_m"],
            "selected_distance_to_stop_m": payload["distance_to_stop_m"],
            "selected_distance_to_light_m": payload["distance_to_light_m"],
            "selected_stop_point_source": payload["stop_point_source"],
            "selected_confidence": payload["confidence"],
            "selected_lateral_distance_m": payload["route_lateral_distance_m"],
            "selected_route_index": payload["route_index"],
            "selected_stop_route_index": payload["stop_route_index"],
            "selected_fence_intersection_found": payload[
                "fence_intersection_found"
            ],
            "selected_stop_x": payload["stop_x"],
            "selected_stop_y": payload["stop_y"],
            "selected_stop_s": payload["stop_s"],
            "selected_reason": payload["reason"],
            "stopline_x": payload.get("stopline_x"),
            "stopline_y": payload.get("stopline_y"),
            "ego_center_x": payload.get("ego_center_x"),
            "ego_center_y": payload.get("ego_center_y"),
            "ego_front_x": payload.get("ego_front_x"),
            "ego_front_y": payload.get("ego_front_y"),
            "center_to_stopline_m": payload.get("center_to_stopline_m"),
            "front_bumper_to_stopline_m": payload.get("front_bumper_to_stopline_m"),
            "passed_stopline": payload.get("passed_stopline", False),
            "passed_stopline_threshold_m": self.traffic_light_passed_stopline_threshold_m,
            "post_green_ignore_active": payload.get("post_green_ignore_active", False),
            "post_green_ignore_light_id": payload.get("post_green_ignore_light_id"),
            "post_tl_ignore_active": payload.get("post_tl_ignore_active", False),
            "same_light_ignore_active": payload.get("same_light_ignore_active", False),
            "last_green_clear_light_id": self._last_green_clear_light_id,
            "stale_red_stop_suppressed": payload.get(
                "stale_red_stop_suppressed",
                debug.get("stale_red_stop_suppressed", False),
            ),
            "stale_red_stop_suppress_reason": payload.get(
                "stale_red_stop_suppress_reason",
                debug.get("stale_red_stop_suppress_reason"),
            ),
            "last_restrictive_light_id": debug.get("last_restrictive_light_id"),
            "candidate_light_id": payload.get("candidate_light_id"),
            "candidate_road_id": payload.get("candidate_road_id"),
            "candidate_lane_id": payload.get("candidate_lane_id"),
            "ego_road_id": payload.get("ego_road_id"),
            "ego_lane_id": payload.get("ego_lane_id"),
            "route_heading_deg": payload.get("route_heading_deg"),
            "candidate_heading_deg": payload.get("candidate_heading_deg"),
            "candidate_yaw_diff_deg": payload.get("candidate_yaw_diff_deg"),
            "candidate_cross_track_m": payload.get("candidate_cross_track_m"),
            "candidate_along_track_m": payload.get("candidate_along_track_m"),
            "candidate_on_route_corridor": payload.get("candidate_on_route_corridor"),
            "candidate_heading_ok": payload.get("candidate_heading_ok"),
            "candidate_ahead_ok": payload.get("candidate_ahead_ok"),
            "candidate_rejected": payload.get("candidate_rejected", False),
            "candidate_reject_reason": payload.get("candidate_reject_reason"),
            "selected_light_id": payload.get("selected_light_id"),
            "selected_light_reason": payload.get("selected_light_reason"),
            "current_candidate_light_id": debug.get("current_candidate_light_id"),
            "replay_candidate_light_id": debug.get("replay_candidate_light_id"),
            "old_light_replay_suppressed": debug.get("old_light_replay_suppressed", False),
            "applied_stop_line_buffer_m": payload.get("applied_stop_line_buffer_m"),
            "red_stop_trigger_reason": payload.get("red_stop_trigger_reason"),
            "current_speed_mps": payload.get("current_speed_mps"),
            "red_stop_trigger_m": payload.get("red_stop_trigger_m"),
            "red_stop_triggered_by_distance": payload.get(
                "red_stop_triggered_by_distance",
                False,
            ),
            "active_tl_id": self.active_tl_id,
            "active_tl_state": self.active_tl_state,
            "active_light_id": self.active_tl_id,
            "active_light_state": self.active_tl_state,
            "previous_tl_state": self.previous_tl_state,
            "active_stop_s": self.active_stop_s,
            "active_stop_route_index": self.active_stop_route_index,
            "active_stop_point_source": self.active_stop_point_source,
            "red_hold_active": self.red_hold_active,
            "stopped_for_tl_id": self.stopped_for_tl_id,
            "green_release_triggered": self.green_release_triggered,
            "red_hold_active_before": self.red_hold_active_before,
            "red_hold_active_after": self.red_hold_active_after,
            "stop_hold_cleared": self.stop_hold_cleared,
            "release_reason": self.release_reason,
            "last_tl_event": self.last_tl_event,
        })
        return payload, debug

    def _tick(self) -> None:
        if self._world is None:
            self._connect_to_carla()

        try:
            payload, debug = self._analyze()
        except Exception as exc:
            payload = self._base_event(False, f"analysis_failed: {exc}")
            debug = {
                "stamp": time.time(),
                "analysis_error": str(exc),
            }
            self._warn_throttled(
                "analysis",
                f"RouteEventAnalyzer: analysis failed: {exc}",
            )

        warning_payload = debug.pop("warning_payload", None)
        debug.update({
            "ok": payload["ok"],
            "event": payload["event"],
            "distance_m": payload["distance_m"],
            "distance_to_stop_m": payload["distance_to_stop_m"],
            "distance_to_light_m": payload["distance_to_light_m"],
            "actor_id": payload["actor_id"],
            "traffic_light_id": payload["traffic_light_id"],
            "actor_speed_mps": payload["actor_speed_mps"],
            "traffic_light_state": payload["traffic_light_state"],
            "reason": payload["reason"],
            "selected_event": payload["event"],
            "selected_actor_id": payload["actor_id"],
            "selected_traffic_light_state": payload["traffic_light_state"],
            "selected_actor_speed_mps": payload["actor_speed_mps"],
            "selected_distance_m": payload["distance_m"],
            "selected_distance_to_stop_m": payload["distance_to_stop_m"],
            "selected_distance_to_light_m": payload["distance_to_light_m"],
            "selected_stop_point_source": payload["stop_point_source"],
            "selected_confidence": payload["confidence"],
            "selected_lateral_distance_m": payload["route_lateral_distance_m"],
            "selected_route_index": payload["route_index"],
            "selected_stop_route_index": payload["stop_route_index"],
            "selected_fence_intersection_found": payload[
                "fence_intersection_found"
            ],
            "selected_stop_x": payload["stop_x"],
            "selected_stop_y": payload["stop_y"],
            "selected_stop_s": payload["stop_s"],
            "selected_reason": payload["reason"],
            "stopline_x": payload.get("stopline_x"),
            "stopline_y": payload.get("stopline_y"),
            "ego_center_x": payload.get("ego_center_x"),
            "ego_center_y": payload.get("ego_center_y"),
            "ego_front_x": payload.get("ego_front_x"),
            "ego_front_y": payload.get("ego_front_y"),
            "center_to_stopline_m": payload.get("center_to_stopline_m"),
            "front_bumper_to_stopline_m": payload.get("front_bumper_to_stopline_m"),
            "passed_stopline": payload.get("passed_stopline", False),
            "passed_stopline_threshold_m": self.traffic_light_passed_stopline_threshold_m,
            "post_green_ignore_active": payload.get("post_green_ignore_active", False),
            "post_green_ignore_light_id": payload.get("post_green_ignore_light_id"),
            "post_tl_ignore_active": payload.get("post_tl_ignore_active", False),
            "same_light_ignore_active": payload.get("same_light_ignore_active", False),
            "last_green_clear_light_id": self._last_green_clear_light_id,
            "stale_red_stop_suppressed": payload.get(
                "stale_red_stop_suppressed",
                debug.get("stale_red_stop_suppressed", False),
            ),
            "stale_red_stop_suppress_reason": payload.get(
                "stale_red_stop_suppress_reason",
                debug.get("stale_red_stop_suppress_reason"),
            ),
            "last_restrictive_light_id": debug.get("last_restrictive_light_id"),
            "candidate_light_id": payload.get("candidate_light_id"),
            "candidate_road_id": payload.get("candidate_road_id"),
            "candidate_lane_id": payload.get("candidate_lane_id"),
            "ego_road_id": payload.get("ego_road_id"),
            "ego_lane_id": payload.get("ego_lane_id"),
            "route_heading_deg": payload.get("route_heading_deg"),
            "candidate_heading_deg": payload.get("candidate_heading_deg"),
            "candidate_yaw_diff_deg": payload.get("candidate_yaw_diff_deg"),
            "candidate_cross_track_m": payload.get("candidate_cross_track_m"),
            "candidate_along_track_m": payload.get("candidate_along_track_m"),
            "candidate_on_route_corridor": payload.get("candidate_on_route_corridor"),
            "candidate_heading_ok": payload.get("candidate_heading_ok"),
            "candidate_ahead_ok": payload.get("candidate_ahead_ok"),
            "candidate_rejected": payload.get("candidate_rejected", False),
            "candidate_reject_reason": payload.get("candidate_reject_reason"),
            "selected_light_id": payload.get("selected_light_id"),
            "selected_light_reason": payload.get("selected_light_reason"),
            "current_candidate_light_id": debug.get("current_candidate_light_id"),
            "replay_candidate_light_id": debug.get("replay_candidate_light_id"),
            "old_light_replay_suppressed": debug.get("old_light_replay_suppressed", False),
            "applied_stop_line_buffer_m": payload.get("applied_stop_line_buffer_m"),
            "red_stop_trigger_reason": payload.get("red_stop_trigger_reason"),
            "current_speed_mps": payload.get("current_speed_mps"),
            "red_stop_trigger_m": payload.get("red_stop_trigger_m"),
            "red_stop_triggered_by_distance": payload.get(
                "red_stop_triggered_by_distance",
                False,
            ),
            "active_tl_id": self.active_tl_id,
            "active_tl_state": self.active_tl_state,
            "active_light_id": self.active_tl_id,
            "active_light_state": self.active_tl_state,
            "previous_tl_state": self.previous_tl_state,
            "active_stop_s": self.active_stop_s,
            "active_stop_route_index": self.active_stop_route_index,
            "active_stop_point_source": self.active_stop_point_source,
            "red_hold_active": self.red_hold_active,
            "stopped_for_tl_id": self.stopped_for_tl_id,
            "green_release_triggered": self.green_release_triggered,
            "red_hold_active_before": self.red_hold_active_before,
            "red_hold_active_after": self.red_hold_active_after,
            "stop_hold_cleared": self.stop_hold_cleared,
            "release_reason": self.release_reason,
            "last_tl_event": self.last_tl_event,
        })
        if warning_payload is not None and time.monotonic() - self._last_warning_debug_publish >= 1.0:
            self._last_warning_debug_publish = time.monotonic()
            self.debug_pub.publish(String(data=json.dumps(warning_payload)))
            self.runtime_logger.write({
                "kind": "traffic_light_warning",
                **warning_payload,
            })
            self._warn_throttled(
                "no_traffic_light_actors",
                "TL EVENTS ENABLED BUT NO traffic.traffic_light ACTORS FOUND IN CARLA WORLD. "
                "RED LIGHT STOP CANNOT WORK ON THIS MAP.",
                period_s=5.0,
            )

        if payload.get("reason") != self._last_debug_reason:
            self._last_debug_reason = str(payload.get("reason"))
            self.get_logger().info(
                "RouteEventAnalyzer decision: "
                f"event={payload.get('event')} reason={payload.get('reason')} "
                f"traffic_light_actor_count={debug.get('traffic_light_actor_count')}"
            )

        self.event_pub.publish(String(data=json.dumps(payload)))
        self.debug_pub.publish(String(data=json.dumps(debug)))
        self.runtime_logger.write({
            "kind": "route_event_tick",
            "payload": payload,
            "debug": debug,
        })


def main(args=None):
    rclpy.init(args=args)
    node = RouteEventAnalyzer()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
