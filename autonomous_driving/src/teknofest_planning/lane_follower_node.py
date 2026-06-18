#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_common.runtime_logging import RuntimeJsonlLogger
from teknofest_sim.carla_loader import load_carla
from teknofest_planning.lane_policy import (
    EgoPose,
    LanePolicyConfig,
    RoutePoint,
    build_lane_plan,
)


class LaneFollowerNode(Node):
    def __init__(self):
        super().__init__("lane_follower_node")

        # -------------------------
        # Config / parameter block
        # -------------------------
        self.declare_parameter("route_topic", "/adas/planning/route")
        self.declare_parameter("mission_topic", "/adas/teknofest/mission")
        self.declare_parameter("status_topic", "/adas/carla/status")
        self.declare_parameter("lane_plan_topic", "/adas/planning/lane_plan")
        self.declare_parameter("carla_root", "/home/ilker/simulators/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("timeout", 120.0)

        self.declare_parameter("cruise_speed_mps", 6.0)
        self.declare_parameter("turn_speed_mps", 3.2)
        self.declare_parameter("approach_speed_mps", 2.0)
        self.declare_parameter("min_drive_speed_mps", 1.2)
        self.declare_parameter("stop_distance_m", 1.8)
        self.declare_parameter("approach_distance_m", 14.0)
        self.declare_parameter("lookahead_base_m", 7.0)
        self.declare_parameter("lookahead_gain", 0.65)
        self.declare_parameter("lookahead_min_m", 5.0)
        self.declare_parameter("lookahead_max_m", 14.0)
        self.declare_parameter("turn_angle_slowdown_deg", 35.0)
        self.declare_parameter("route_timeout_s", 2.5)
        self.declare_parameter("status_timeout_s", 3.0)
        self.declare_parameter("mission_timeout_s", 3.0)
        self.declare_parameter("startup_duration_s", 5.0)
        self.declare_parameter("startup_lane_lock_s", 3.0)
        self.declare_parameter("startup_speed_mps", 2.2)
        self.declare_parameter("unstable_lane_speed_mps", 2.8)
        self.declare_parameter("startup_lane_target_m", 3.2)
        self.declare_parameter("lane_target_jump_threshold_m", 6.0)
        self.declare_parameter("heading_error_slowdown_deg", 15.0)
        self.declare_parameter("cross_track_slowdown_m", 0.5)
        self.declare_parameter("upcoming_turn_lookahead_m", 25.0)
        self.declare_parameter("turn_slowdown_start_m", 22.0)
        self.declare_parameter("turn_speed_limit_mps", 2.8)
        self.declare_parameter("approach_turn_speed_mps", 2.8)
        self.declare_parameter("junction_turn_speed_mps", 2.2)
        self.declare_parameter("exit_turn_speed_mps", 2.6)
        self.declare_parameter("post_turn_speed_mps", 3.2)
        self.declare_parameter("hard_alignment_speed_mps", 1.8)
        self.declare_parameter("target_forward_min_m", 1.0)
        self.declare_parameter("target_jump_reject_m", 5.0)
        self.declare_parameter("junction_lane_change_distance_m", 8.0)
        self.declare_parameter("turn_state_min_s", 2.0)
        self.declare_parameter("post_turn_stabilize_s", 1.2)
        self.declare_parameter("cruise_cte_threshold_m", 0.35)
        self.declare_parameter("cruise_heading_threshold_deg", 6.0)
        self.declare_parameter("lane_confirm_cte_threshold_m", 0.4)
        self.declare_parameter("lane_confirm_heading_threshold_deg", 10.0)
        self.declare_parameter("recovery_cte_threshold_m", 0.7)
        self.declare_parameter("recovery_heading_threshold_deg", 30.0)
        self.declare_parameter("recovery_speed_mps", 1.7)
        self.declare_parameter("junction_offroute_safety_cte_m", 3.0)
        self.declare_parameter("junction_route_recovery_speed_mps", 0.8)
        self.declare_parameter("turn_arc_lookahead_m", 3.2)
        self.declare_parameter("turn_approach_lookahead_m", 4.0)
        self.declare_parameter("speed_setpoint_accel_mps2", 1.1)
        self.declare_parameter("speed_setpoint_decel_mps2", 1.0)
        self.declare_parameter("log_root", "autonomous_driving/outputs/teknofest_sim_logs")
        self.declare_parameter("log_session_id", "")
        self.declare_parameter("jsonl_logging_enabled", True)
        self.declare_parameter("ros_log_period_s", 1.0)

        self.config = LanePolicyConfig(
            cruise_speed_mps=float(self.get_parameter("cruise_speed_mps").value),
            turn_speed_mps=float(self.get_parameter("turn_speed_mps").value),
            approach_speed_mps=float(self.get_parameter("approach_speed_mps").value),
            min_drive_speed_mps=float(self.get_parameter("min_drive_speed_mps").value),
            stop_distance_m=float(self.get_parameter("stop_distance_m").value),
            approach_distance_m=float(self.get_parameter("approach_distance_m").value),
            lookahead_base_m=float(self.get_parameter("lookahead_base_m").value),
            lookahead_gain=float(self.get_parameter("lookahead_gain").value),
            lookahead_min_m=float(self.get_parameter("lookahead_min_m").value),
            lookahead_max_m=float(self.get_parameter("lookahead_max_m").value),
            turn_angle_slowdown_deg=float(
                self.get_parameter("turn_angle_slowdown_deg").value
            ),
            lateral_slowdown_error_m=float(
                self.declare_parameter("lateral_slowdown_error_m", 0.9).value
            ),
            lateral_slowdown_speed_mps=float(
                self.declare_parameter("lateral_slowdown_speed_mps", 3.0).value
            ),
        )
        self.route_timeout_s = float(self.get_parameter("route_timeout_s").value)
        self.status_timeout_s = float(self.get_parameter("status_timeout_s").value)
        self.mission_timeout_s = float(self.get_parameter("mission_timeout_s").value)
        self.ros_log_period_s = float(self.get_parameter("ros_log_period_s").value)
        self.carla_root = str(self.get_parameter("carla_root").value)
        self.host = str(self.get_parameter("host").value)
        self.port = int(self.get_parameter("port").value)
        self.timeout = float(self.get_parameter("timeout").value)
        self.startup_duration_s = float(self.get_parameter("startup_duration_s").value)
        self.startup_lane_lock_s = float(self.get_parameter("startup_lane_lock_s").value)
        self.startup_speed_mps = float(self.get_parameter("startup_speed_mps").value)
        self.unstable_lane_speed_mps = float(self.get_parameter("unstable_lane_speed_mps").value)
        self.startup_lane_target_m = float(self.get_parameter("startup_lane_target_m").value)
        self.lane_target_jump_threshold_m = float(
            self.get_parameter("lane_target_jump_threshold_m").value
        )
        self.heading_error_slowdown_deg = float(
            self.get_parameter("heading_error_slowdown_deg").value
        )
        self.cross_track_slowdown_m = float(self.get_parameter("cross_track_slowdown_m").value)
        self.upcoming_turn_lookahead_m = float(
            self.get_parameter("upcoming_turn_lookahead_m").value
        )
        self.turn_slowdown_start_m = float(self.get_parameter("turn_slowdown_start_m").value)
        self.turn_speed_limit_mps = float(self.get_parameter("turn_speed_limit_mps").value)
        self.approach_turn_speed_mps = float(self.get_parameter("approach_turn_speed_mps").value)
        self.junction_turn_speed_mps = float(self.get_parameter("junction_turn_speed_mps").value)
        self.exit_turn_speed_mps = float(self.get_parameter("exit_turn_speed_mps").value)
        self.post_turn_speed_mps = float(self.get_parameter("post_turn_speed_mps").value)
        self.hard_alignment_speed_mps = float(self.get_parameter("hard_alignment_speed_mps").value)
        self.target_forward_min_m = float(self.get_parameter("target_forward_min_m").value)
        self.target_jump_reject_m = float(self.get_parameter("target_jump_reject_m").value)
        self.junction_lane_change_distance_m = float(
            self.get_parameter("junction_lane_change_distance_m").value
        )
        self.turn_state_min_s = float(self.get_parameter("turn_state_min_s").value)
        self.post_turn_stabilize_s = float(self.get_parameter("post_turn_stabilize_s").value)
        self.cruise_cte_threshold_m = float(self.get_parameter("cruise_cte_threshold_m").value)
        self.cruise_heading_threshold_deg = float(
            self.get_parameter("cruise_heading_threshold_deg").value
        )
        self.lane_confirm_cte_threshold_m = float(
            self.get_parameter("lane_confirm_cte_threshold_m").value
        )
        self.lane_confirm_heading_threshold_deg = float(
            self.get_parameter("lane_confirm_heading_threshold_deg").value
        )
        self.recovery_cte_threshold_m = float(self.get_parameter("recovery_cte_threshold_m").value)
        self.recovery_heading_threshold_deg = float(
            self.get_parameter("recovery_heading_threshold_deg").value
        )
        self.recovery_speed_mps = float(self.get_parameter("recovery_speed_mps").value)
        self.junction_offroute_safety_cte_m = float(
            self.get_parameter("junction_offroute_safety_cte_m").value
        )
        self.junction_route_recovery_speed_mps = float(
            self.get_parameter("junction_route_recovery_speed_mps").value
        )
        self.turn_arc_lookahead_m = float(self.get_parameter("turn_arc_lookahead_m").value)
        self.turn_approach_lookahead_m = float(
            self.get_parameter("turn_approach_lookahead_m").value
        )
        self.speed_setpoint_accel_mps2 = float(
            self.get_parameter("speed_setpoint_accel_mps2").value
        )
        self.speed_setpoint_decel_mps2 = float(
            self.get_parameter("speed_setpoint_decel_mps2").value
        )

        # -------------------------
        # Runtime state block
        # -------------------------
        self.carla = None
        self.client = None
        self.world = None
        self.map = None
        self.route_payload: Optional[dict] = None
        self.mission_payload: Optional[dict] = None
        self.status_payload: Optional[dict] = None
        self.last_route_time_s = 0.0
        self.last_mission_time_s = 0.0
        self.last_status_time_s = 0.0
        self.last_nearest_index = 0
        self.last_objective_index = None
        self.last_route_signature = None
        self.last_ros_log_s = 0.0
        self.started_at_s = time.time()
        self.previous_road_id = None
        self.previous_lane_id = None
        self.previous_target_x = None
        self.previous_target_y = None
        self.previous_target_road_id = None
        self.previous_target_lane_id = None
        self.turn_state = "NORMAL_LANE_FOLLOW"
        self.turn_state_started_at_s = time.time()
        self.stable_after_turn_since_s = None
        self.previous_abs_cte = None
        self.last_tick_s = time.time()
        self.speed_setpoint_smoothed = 0.0
        self.junction_entry_index = None
        self.junction_exit_index = None

        self.runtime_logger = RuntimeJsonlLogger(
            node_name="lane_follower_node",
            file_name="lane.jsonl",
            log_root=str(self.get_parameter("log_root").value),
            session_id=str(self.get_parameter("log_session_id").value) or None,
            enabled=bool(self.get_parameter("jsonl_logging_enabled").value),
        )

        # -------------------------
        # Publisher block
        # -------------------------
        self.plan_pub = self.create_publisher(
            String,
            str(self.get_parameter("lane_plan_topic").value),
            10,
        )

        # -------------------------
        # Subscriber block
        # -------------------------
        self.create_subscription(
            String,
            str(self.get_parameter("route_topic").value),
            self.route_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("mission_topic").value),
            self.mission_cb,
            10,
        )
        self.create_subscription(
            String,
            str(self.get_parameter("status_topic").value),
            self.status_cb,
            10,
        )

        # -------------------------
        # Timer block
        # -------------------------
        self.connect_to_carla()
        self.create_timer(0.1, self.tick)
        self.get_logger().info("Lane follower node ready.")

    # -------------------------
    # CARLA map helper functions
    # -------------------------
    def connect_to_carla(self):
        try:
            self.carla = load_carla(self.carla_root)
            self.client = self.carla.Client(self.host, self.port)
            self.client.set_timeout(self.timeout)
            self.world = self.client.get_world()
            self.map = self.world.get_map()
        except Exception as exc:
            self.get_logger().warning(f"Lane follower CARLA map unavailable: {exc}")
            self.carla = None
            self.map = None

    # -------------------------
    # Subscriber callbacks
    # -------------------------
    def route_cb(self, msg: String):
        try:
            payload = json.loads(msg.data)
            signature = (
                payload.get("objective_index"),
                (payload.get("active_mission_target") or {}).get("name"),
                payload.get("route_id"),
            )
            if signature != self.last_route_signature:
                self.last_nearest_index = 0
                self.last_route_signature = signature
            self.route_payload = payload
            self.last_route_time_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid route JSON ignored: {exc}")

    def mission_cb(self, msg: String):
        try:
            self.mission_payload = json.loads(msg.data)
            self.last_mission_time_s = time.time()
            objective_index = self.mission_payload.get("objective_index")
            if objective_index != self.last_objective_index:
                self.last_nearest_index = 0
                self.last_objective_index = objective_index
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid mission JSON ignored: {exc}")

    def status_cb(self, msg: String):
        try:
            self.status_payload = json.loads(msg.data)
            self.last_status_time_s = time.time()
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"Invalid CARLA status JSON ignored: {exc}")

    # -------------------------
    # Data conversion functions
    # -------------------------
    def route_points(self) -> list[RoutePoint]:
        payload = self.route_payload or {}
        points = payload.get("points", [])
        out = []

        for index, point in enumerate(points):
            try:
                out.append(
                    RoutePoint(
                        x=float(point["x"]),
                        y=float(point["y"]),
                        z=float(point.get("z", 0.0)),
                        yaw_deg=float(point.get("yaw_deg", 0.0)),
                        road_id=point.get("road_id"),
                        lane_id=point.get("lane_id"),
                        lane_width=point.get("lane_width"),
                        is_junction=bool(point.get("is_junction", False)),
                        route_index=int(point.get("index", index)),
                        road_option=str(point.get("road_option", "LANEFOLLOW")),
                    )
                )
            except (KeyError, TypeError, ValueError):
                continue

        return out

    def ego_pose(self) -> Optional[EgoPose]:
        status = self.status_payload or {}
        location = status.get("location") or {}
        rotation = status.get("rotation") or {}

        try:
            return EgoPose(
                x=float(location["x"]),
                y=float(location["y"]),
                yaw_deg=float(rotation.get("yaw", 0.0)),
                speed_mps=float(status.get("speed_mps", 0.0)),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def mission_must_stop(self) -> bool:
        if not self.mission_payload:
            return False

        return bool(self.mission_payload.get("must_stop", False))

    def distance_to_goal_m(self) -> Optional[float]:
        if self.mission_payload and self.mission_payload.get("distance_to_objective_m") is not None:
            return float(self.mission_payload["distance_to_objective_m"])

        if self.route_payload and self.route_payload.get("distance_to_objective_m") is not None:
            return float(self.route_payload["distance_to_objective_m"])

        return None

    def current_carla_waypoint_info(self, ego: EgoPose) -> dict:
        if self.map is None or self.carla is None:
            return {}

        try:
            loc = self.carla.Location(x=float(ego.x), y=float(ego.y), z=0.0)
            waypoint = self.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
            transform = waypoint.transform
            yaw = math.radians(float(transform.rotation.yaw))
            dx = float(ego.x) - float(transform.location.x)
            dy = float(ego.y) - float(transform.location.y)
            lateral = -math.sin(yaw) * dx + math.cos(yaw) * dy
            return {
                "current_road_id": int(waypoint.road_id),
                "current_lane_id": int(waypoint.lane_id),
                "current_is_junction": bool(waypoint.is_junction),
                "lane_width": round(float(waypoint.lane_width), 3),
                "actual_lateral_distance_to_lane_center": round(float(lateral), 3),
                "current_lane_center_x": round(float(transform.location.x), 4),
                "current_lane_center_y": round(float(transform.location.y), 4),
            }
        except Exception as exc:
            return {"waypoint_error": str(exc)}

    def same_lane_target_point(self, ego: EgoPose, distance_m: float) -> Optional[RoutePoint]:
        if self.map is None or self.carla is None:
            return None

        try:
            loc = self.carla.Location(x=float(ego.x), y=float(ego.y), z=0.0)
            current_wp = self.map.get_waypoint(
                loc,
                project_to_road=True,
                lane_type=self.carla.LaneType.Driving,
            )
            candidates = current_wp.next(max(1.0, float(distance_m)))
            same_lane = [
                wp for wp in candidates
                if wp.road_id == current_wp.road_id and wp.lane_id == current_wp.lane_id
            ]
            target_wp = same_lane[0] if same_lane else current_wp
            transform = target_wp.transform
            return RoutePoint(
                x=float(transform.location.x),
                y=float(transform.location.y),
                z=float(transform.location.z),
                yaw_deg=float(transform.rotation.yaw),
                road_id=int(target_wp.road_id),
                lane_id=int(target_wp.lane_id),
                route_index=-1,
                road_option="STARTUP_LANE_HOLD",
            )
        except Exception:
            return None

    def heading_error_to_target(self, ego: EgoPose, target: RoutePoint) -> float:
        delta = (float(target.yaw_deg) - float(ego.yaw_deg) + 180.0) % 360.0 - 180.0
        return delta

    def dot_to_target(self, ego: EgoPose, target: RoutePoint) -> float:
        yaw = math.radians(float(ego.yaw_deg))
        dx = float(target.x) - float(ego.x)
        dy = float(target.y) - float(ego.y)
        return math.cos(yaw) * dx + math.sin(yaw) * dy

    def distance_to_previous_target(self, target: RoutePoint) -> Optional[float]:
        if self.previous_target_x is None or self.previous_target_y is None:
            return None

        return math.hypot(float(target.x) - self.previous_target_x, float(target.y) - self.previous_target_y)

    def update_previous_target(self, target: RoutePoint):
        self.previous_target_x = float(target.x)
        self.previous_target_y = float(target.y)
        self.previous_target_road_id = target.road_id
        self.previous_target_lane_id = target.lane_id

    def route_segment_distance(self, points: list[RoutePoint], start: int, end: int) -> float:
        if not points:
            return 0.0

        start = max(0, min(start, len(points) - 1))
        end = max(start, min(end, len(points) - 1))
        total = 0.0

        for index in range(start, end):
            total += math.hypot(points[index + 1].x - points[index].x, points[index + 1].y - points[index].y)

        return total

    def filter_route_option(self, route_option: str) -> str:
        option = str(route_option or "").upper()
        if option in {"CHANGELANELEFT", "CHANGELANERIGHT"}:
            return "LANEFOLLOW"
        return option or "LANEFOLLOW"

    def real_turn_option(self, route_option: str) -> bool:
        return self.filter_route_option(route_option) in {"LEFT", "RIGHT"}

    def upcoming_turn_info(self, points: list[RoutePoint], nearest_index: int) -> dict:
        if not points:
            return {
                "upcoming_turn_type": "NONE",
                "upcoming_turn_distance_m": None,
                "route_option": "",
                "route_option_raw": "",
                "route_option_filtered": "",
                "is_junction": False,
            }

        traveled = 0.0
        base_yaw = points[max(0, min(nearest_index, len(points) - 1))].yaw_deg
        last = points[max(0, min(nearest_index, len(points) - 1))]
        first_raw_option = str(last.road_option or "").upper()
        first_filtered_option = self.filter_route_option(first_raw_option)

        for index in range(max(0, nearest_index), len(points)):
            point = points[index]
            if index > nearest_index:
                traveled += math.hypot(point.x - last.x, point.y - last.y)
            last = point

            if traveled > self.upcoming_turn_lookahead_m:
                break

            route_option_raw = str(point.road_option or "").upper()
            route_option = self.filter_route_option(route_option_raw)
            yaw_delta = abs((point.yaw_deg - base_yaw + 180.0) % 360.0 - 180.0)

            if route_option_raw in {"CHANGELANELEFT", "CHANGELANERIGHT"}:
                continue

            if route_option in {"LEFT", "RIGHT"}:
                return {
                    "upcoming_turn_type": route_option,
                    "upcoming_turn_distance_m": round(traveled, 3),
                    "route_option": route_option,
                    "route_option_raw": route_option_raw,
                    "route_option_filtered": route_option,
                    "is_junction": bool(point.is_junction),
                }

            if point.is_junction and yaw_delta >= 12.0:
                return {
                    "upcoming_turn_type": "JUNCTION_CURVE",
                    "upcoming_turn_distance_m": round(traveled, 3),
                    "route_option": route_option,
                    "route_option_raw": route_option_raw,
                    "route_option_filtered": route_option,
                    "is_junction": True,
                }

            if yaw_delta >= 22.0:
                return {
                    "upcoming_turn_type": "CURVE",
                    "upcoming_turn_distance_m": round(traveled, 3),
                    "route_option": route_option,
                    "route_option_raw": route_option_raw,
                    "route_option_filtered": route_option,
                    "is_junction": bool(point.is_junction),
                }

        return {
            "upcoming_turn_type": "NONE",
            "upcoming_turn_distance_m": None,
            "route_option": first_filtered_option,
            "route_option_raw": first_raw_option,
            "route_option_filtered": first_filtered_option,
            "is_junction": bool(points[nearest_index].is_junction if points else False),
        }

    def target_lane_differs(self, current_road_id, current_lane_id, target: RoutePoint) -> bool:
        if current_road_id is None or current_lane_id is None:
            return False
        if target.road_id is None or target.lane_id is None:
            return False
        return current_road_id != target.road_id or current_lane_id != target.lane_id

    def set_turn_state(self, new_state: str, now: float, nearest_index: int):
        if new_state == self.turn_state:
            return

        self.turn_state = new_state
        self.turn_state_started_at_s = now
        if new_state == "ENTERING_JUNCTION":
            self.junction_entry_index = nearest_index
        elif new_state == "EXITING_JUNCTION":
            self.junction_exit_index = nearest_index
        if new_state != "STABILIZE_AFTER_TURN":
            self.stable_after_turn_since_s = None

    def update_turn_state(
        self,
        *,
        now: float,
        nearest_index: int,
        turn_info: dict,
        current_is_junction: bool,
        cte: float,
        heading_error: float,
    ):
        turn_type = str(turn_info.get("upcoming_turn_type") or "NONE")
        turn_distance = turn_info.get("upcoming_turn_distance_m")
        real_turn_ahead = (
            turn_type in {"LEFT", "RIGHT", "JUNCTION_CURVE", "CURVE"}
            and turn_distance is not None
            and float(turn_distance) <= self.turn_slowdown_start_m
        )
        state_age = now - self.turn_state_started_at_s
        aligned_for_lane_confirm = (
            abs(cte) <= self.lane_confirm_cte_threshold_m
            and abs(heading_error) <= self.lane_confirm_heading_threshold_deg
        )
        cruise_stable = (
            abs(cte) <= self.cruise_cte_threshold_m
            and abs(heading_error) <= self.cruise_heading_threshold_deg
        )

        if self.turn_state == "NORMAL_LANE_FOLLOW":
            if real_turn_ahead:
                self.set_turn_state("APPROACHING_TURN", now, nearest_index)
        elif self.turn_state == "APPROACHING_TURN":
            if current_is_junction or (turn_distance is not None and float(turn_distance) <= 6.0):
                self.set_turn_state("ENTERING_JUNCTION", now, nearest_index)
            elif not real_turn_ahead and state_age >= self.turn_state_min_s:
                self.set_turn_state("NORMAL_LANE_FOLLOW", now, nearest_index)
        elif self.turn_state == "ENTERING_JUNCTION":
            if current_is_junction or (turn_distance is not None and float(turn_distance) <= 2.0):
                self.set_turn_state("IN_JUNCTION_TURN", now, nearest_index)
            elif state_age >= self.turn_state_min_s and not real_turn_ahead:
                self.set_turn_state("EXITING_JUNCTION", now, nearest_index)
        elif self.turn_state == "IN_JUNCTION_TURN":
            if not current_is_junction and state_age >= self.turn_state_min_s:
                self.set_turn_state("EXITING_JUNCTION", now, nearest_index)
        elif self.turn_state == "EXITING_JUNCTION":
            if aligned_for_lane_confirm:
                self.set_turn_state("STABILIZE_AFTER_TURN", now, nearest_index)
        elif self.turn_state == "STABILIZE_AFTER_TURN":
            if cruise_stable:
                if self.stable_after_turn_since_s is None:
                    self.stable_after_turn_since_s = now
                elif now - self.stable_after_turn_since_s >= self.post_turn_stabilize_s:
                    self.set_turn_state("NORMAL_LANE_FOLLOW", now, nearest_index)
            else:
                self.stable_after_turn_since_s = None

    def route_forward_target(
        self,
        points: list[RoutePoint],
        ego: EgoPose,
        start_index: int,
        lookahead_m: float,
    ) -> Optional[RoutePoint]:
        if not points:
            return None

        start_index = max(0, min(start_index, len(points) - 1))
        traveled = 0.0
        previous = points[start_index]
        best = None

        for index in range(start_index + 1, len(points)):
            point = points[index]
            traveled += math.hypot(point.x - previous.x, point.y - previous.y)
            previous = point

            if self.dot_to_target(ego, point) < self.target_forward_min_m:
                continue

            best = point
            if traveled >= lookahead_m:
                return point

        return best

    def select_turn_arc_target(
        self,
        points: list[RoutePoint],
        ego: EgoPose,
        nearest_index: int,
    ) -> Optional[RoutePoint]:
        lookahead = (
            self.turn_approach_lookahead_m
            if self.turn_state == "APPROACHING_TURN"
            else self.turn_arc_lookahead_m
        )
        return self.route_forward_target(points, ego, nearest_index, lookahead)

    def smooth_speed_setpoint(self, raw_speed: float, dt: float) -> float:
        if self.speed_setpoint_smoothed <= 0.0:
            self.speed_setpoint_smoothed = min(raw_speed, self.startup_speed_mps)

        delta = raw_speed - self.speed_setpoint_smoothed
        if delta >= 0.0:
            max_delta = self.speed_setpoint_accel_mps2 * dt
        else:
            max_delta = self.speed_setpoint_decel_mps2 * dt

        if abs(delta) <= max_delta:
            self.speed_setpoint_smoothed = raw_speed
        else:
            self.speed_setpoint_smoothed += math.copysign(max_delta, delta)

        return self.speed_setpoint_smoothed

    def age_ms(self, now: float, stamp_s: float):
        if stamp_s <= 0.0:
            return None
        return int(max(0.0, now - stamp_s) * 1000.0)

    # -------------------------
    # Decision / state machine functions
    # -------------------------
    def tick(self):
        now = time.time()
        dt = max(1e-3, now - self.last_tick_s)
        self.last_tick_s = now
        ego = self.ego_pose()

        if ego is None or now - self.last_status_time_s > self.status_timeout_s:
            self.publish_stop_plan("missing_status")
            return

        if self.mission_payload is None or now - self.last_mission_time_s > self.mission_timeout_s:
            self.publish_stop_plan("missing_mission", ego)
            return

        route_points = self.route_points()
        if not route_points or now - self.last_route_time_s > self.route_timeout_s:
            self.publish_stop_plan("missing_route", ego)
            return

        plan = build_lane_plan(
            route_points=route_points,
            ego=ego,
            config=self.config,
            last_nearest_index=self.last_nearest_index,
            mission_must_stop=self.mission_must_stop(),
            distance_to_goal_m=self.distance_to_goal_m(),
        )
        self.last_nearest_index = plan.nearest_index
        waypoint_info = self.current_carla_waypoint_info(ego)
        route_age_ms = self.age_ms(now, self.last_route_time_s)
        mission_status_age_ms = self.age_ms(now, self.last_mission_time_s)
        ego_status_age_ms = self.age_ms(now, self.last_status_time_s)
        startup_phase = now - self.started_at_s <= self.startup_duration_s
        startup_lane_lock_active = now - self.started_at_s <= self.startup_lane_lock_s
        previous_road_id = self.previous_road_id
        previous_lane_id = self.previous_lane_id
        current_road_id = waypoint_info.get("current_road_id")
        current_lane_id = waypoint_info.get("current_lane_id")
        turn_info = self.upcoming_turn_info(route_points, plan.nearest_index)
        route_option_raw = turn_info.get("route_option_raw")
        route_option_filtered = turn_info.get("route_option_filtered")
        route_target_jump_distance = self.distance_to_previous_target(plan.target_point)
        original_dot_to_target = self.dot_to_target(ego, plan.target_point)
        target_is_behind_ego = original_dot_to_target < self.target_forward_min_m
        is_junction_context = (
            bool(turn_info.get("is_junction"))
            or bool(route_points[plan.nearest_index].is_junction)
            or bool(waypoint_info.get("current_is_junction"))
        )
        lane_transition_detected = (
            (previous_road_id is not None and current_road_id is not None and previous_road_id != current_road_id)
            or (previous_lane_id is not None and current_lane_id is not None and previous_lane_id != current_lane_id)
            or (
                current_road_id is not None
                and plan.target_point.road_id is not None
                and current_road_id != plan.target_point.road_id
            )
            or (
                current_lane_id is not None
                and plan.target_point.lane_id is not None
                and current_lane_id != plan.target_point.lane_id
            )
        )
        target_jump_detected = (
            route_target_jump_distance is not None
            and route_target_jump_distance >= self.target_jump_reject_m
        )
        unstable_alignment = (
            abs(plan.heading_error_deg) >= self.heading_error_slowdown_deg
            or abs(plan.lateral_error_m) >= self.cross_track_slowdown_m
        )
        self.update_turn_state(
            now=now,
            nearest_index=plan.nearest_index,
            turn_info=turn_info,
            current_is_junction=bool(waypoint_info.get("current_is_junction")),
            cte=plan.lateral_error_m,
            heading_error=plan.heading_error_deg,
        )
        selected_target = plan.target_point
        target_override_reason = ""
        lane_jump_rejected = False
        target_jump_rejected = False
        selected_fallback_reason = ""
        turn_arc_target_selected = False
        fallback_target_selected = False
        fallback_reason = ""
        lane_transition_confirmed = (
            abs(plan.lateral_error_m) <= self.lane_confirm_cte_threshold_m
            and abs(plan.heading_error_deg) <= self.lane_confirm_heading_threshold_deg
        )
        lane_sign_flip = (
            current_lane_id is not None
            and plan.target_point.lane_id is not None
            and int(current_lane_id) * int(plan.target_point.lane_id) < 0
        )
        abs_cte = abs(plan.lateral_error_m)
        abs_heading_error = abs(plan.heading_error_deg)
        cte_increasing = (
            self.previous_abs_cte is not None
            and abs_cte > self.previous_abs_cte + 0.05
        )
        target_lane_differs = self.target_lane_differs(current_road_id, current_lane_id, plan.target_point)
        junction_allows_transition = (
            is_junction_context
            and turn_info.get("upcoming_turn_distance_m") is not None
            and float(turn_info["upcoming_turn_distance_m"]) <= self.junction_lane_change_distance_m
        )

        if startup_lane_lock_active and target_lane_differs:
            same_lane_target = self.same_lane_target_point(ego, self.startup_lane_target_m)
            if same_lane_target is not None:
                selected_target = same_lane_target
                target_override_reason = "startup_lane_lock"
                selected_fallback_reason = "same_lane_target"
                fallback_reason = "startup_lane_lock"
                fallback_target_selected = True
                lane_jump_rejected = True

        if self.turn_state in {"APPROACHING_TURN", "ENTERING_JUNCTION", "IN_JUNCTION_TURN"}:
            arc_target = self.select_turn_arc_target(route_points, ego, plan.nearest_index)
            if arc_target is not None:
                selected_target = arc_target
                turn_arc_target_selected = True

        junction_arc_tracking = (
            self.turn_state in {"ENTERING_JUNCTION", "IN_JUNCTION_TURN"}
            and turn_arc_target_selected
            and not lane_sign_flip
        )
        severe_cte_recovery = (
            lane_sign_flip
            or (self.turn_state == "EXITING_JUNCTION" and not lane_transition_confirmed)
            or abs_cte > 0.75
            or (abs_heading_error > 40.0 and cte_increasing and abs_cte > 0.45)
        )
        if not junction_arc_tracking:
            severe_cte_recovery = severe_cte_recovery or (
                abs_cte > self.recovery_cte_threshold_m
                or abs_heading_error > self.recovery_heading_threshold_deg
            )
        soft_turn_alignment_active = (
            junction_arc_tracking
            and not severe_cte_recovery
            and (abs_cte > 0.45 or abs_heading_error > 30.0)
        )
        cte_recovery_active = bool(severe_cte_recovery)

        if target_lane_differs and (not junction_allows_transition or (lane_sign_flip and self.turn_state not in {"ENTERING_JUNCTION", "IN_JUNCTION_TURN"})):
            same_lane_target = self.same_lane_target_point(ego, self.startup_lane_target_m)
            if same_lane_target is not None:
                selected_target = same_lane_target
                target_override_reason = "lane_jump_rejected"
                selected_fallback_reason = "same_lane_target"
                fallback_reason = "lane_jump_rejected"
                fallback_target_selected = True
                lane_jump_rejected = True

        if target_is_behind_ego and not target_override_reason and not turn_arc_target_selected:
            forward_target = self.route_forward_target(route_points, ego, plan.nearest_index, 2.5)
            same_lane_target = forward_target or self.same_lane_target_point(ego, self.startup_lane_target_m)
            if same_lane_target is not None:
                selected_target = same_lane_target
                target_override_reason = "target_behind_rejected"
                selected_fallback_reason = "route_forward_target" if forward_target else "same_lane_forward_target"
                fallback_reason = "target_behind_rejected"
                fallback_target_selected = True

        if target_jump_detected and not target_override_reason and not turn_arc_target_selected:
            forward_target = self.route_forward_target(route_points, ego, plan.nearest_index, 2.8)
            same_lane_target = forward_target or self.same_lane_target_point(ego, self.startup_lane_target_m)
            if same_lane_target is not None:
                selected_target = same_lane_target
                target_override_reason = "target_jump_rejected"
                selected_fallback_reason = "route_small_step_target" if forward_target else "same_lane_target"
                fallback_reason = "target_jump_rejected"
                fallback_target_selected = True
                target_jump_rejected = True

        if startup_phase and (lane_transition_detected or target_jump_detected or unstable_alignment) and not target_override_reason:
            same_lane_target = self.same_lane_target_point(ego, self.startup_lane_target_m)
            if same_lane_target is not None:
                selected_target = same_lane_target
                target_override_reason = "startup_same_lane_hold"
                selected_fallback_reason = "same_lane_target"
                fallback_reason = "startup_same_lane_hold"
                fallback_target_selected = True

        if cte_recovery_active and not turn_arc_target_selected:
            recovery_target = self.same_lane_target_point(ego, 2.8)
            if recovery_target is not None:
                selected_target = recovery_target
                target_override_reason = target_override_reason or "cte_recovery"
                selected_fallback_reason = "lane_center_recovery"
                fallback_reason = fallback_reason or "cte_recovery"
                fallback_target_selected = True

        selected_heading_error = self.heading_error_to_target(ego, selected_target)
        selected_dot_to_target = self.dot_to_target(ego, selected_target)
        speed_setpoint_raw = plan.target_speed_mps
        target_speed = plan.target_speed_mps
        speed_reason = plan.reason
        base_speed_reason = plan.reason
        turn_slowdown_active = False
        route_replan_reason = str((self.route_payload or {}).get("replan_reason", ""))
        off_route = route_replan_reason == "off_route"
        junction_offroute_safety_stop = False
        safety_reason = ""
        stop_request_out = bool(plan.stop_request)
        stop_reason_out = plan.reason if plan.stop_request else ""
        cruise_allowed = (
            self.turn_state == "NORMAL_LANE_FOLLOW"
            and not cte_recovery_active
            and abs(plan.lateral_error_m) <= self.cruise_cte_threshold_m
            and abs(selected_heading_error) <= self.cruise_heading_threshold_deg
        )

        def state_speed_limit(limit_mps: float, floor_mps: float) -> float:
            if base_speed_reason in {"mission_approach", "goal_approach"}:
                return min(target_speed, limit_mps)
            return min(max(target_speed, floor_mps), limit_mps)

        if startup_phase and not plan.stop_request:
            target_speed = min(target_speed, self.startup_speed_mps)
            speed_reason = "startup_ramp"

        if self.turn_state == "APPROACHING_TURN" and not plan.stop_request:
            target_speed = state_speed_limit(self.approach_turn_speed_mps, 2.6)
            speed_reason = "turn_approach"
            turn_slowdown_active = True

        if self.turn_state == "ENTERING_JUNCTION" and not plan.stop_request:
            target_speed = state_speed_limit(max(self.junction_turn_speed_mps, 2.4), 2.0)
            speed_reason = "turn_entry"
            turn_slowdown_active = True

        if self.turn_state == "IN_JUNCTION_TURN" and not plan.stop_request:
            target_speed = state_speed_limit(self.junction_turn_speed_mps, 2.0)
            speed_reason = "junction_turn"
            turn_slowdown_active = True

        if self.turn_state == "EXITING_JUNCTION" and not plan.stop_request:
            target_speed = state_speed_limit(self.exit_turn_speed_mps, 2.4)
            speed_reason = "turn_exit"
            turn_slowdown_active = True

        if self.turn_state == "STABILIZE_AFTER_TURN" and not plan.stop_request:
            target_speed = state_speed_limit(self.post_turn_speed_mps, 2.8)
            speed_reason = "post_turn_stabilize"

        soft_alignment_handled = False
        if soft_turn_alignment_active and not plan.stop_request:
            target_speed = state_speed_limit(2.3, 2.1)
            speed_reason = "soft_turn_alignment"
            turn_slowdown_active = True
            cruise_allowed = False
            soft_alignment_handled = True

        if (unstable_alignment or target_override_reason) and not plan.stop_request and not soft_alignment_handled:
            hard_limit = self.hard_alignment_speed_mps if abs(selected_heading_error) >= 30.0 else self.unstable_lane_speed_mps
            target_speed = min(target_speed, hard_limit)
            if not startup_phase:
                speed_reason = "alignment_slowdown"

        if severe_cte_recovery and not plan.stop_request:
            target_speed = min(target_speed, self.recovery_speed_mps)
            speed_reason = "severe_cte_recovery"
            cruise_allowed = False

        if (
            self.turn_state == "IN_JUNCTION_TURN"
            and not plan.stop_request
            and (
                off_route
                or (severe_cte_recovery and abs_cte > self.junction_offroute_safety_cte_m)
            )
        ):
            junction_offroute_safety_stop = True
            safety_reason = (
                "route_planner_off_route"
                if off_route
                else "junction_severe_cte_over_threshold"
            )
            target_speed = 0.0
            speed_reason = "junction_offroute_safety_stop"
            stop_request_out = True
            stop_reason_out = "junction_offroute_safety_stop"
            cruise_allowed = False
            turn_slowdown_active = False
        elif (
            self.turn_state == "IN_JUNCTION_TURN"
            and not plan.stop_request
            and severe_cte_recovery
        ):
            target_speed = min(target_speed, self.junction_route_recovery_speed_mps)
            speed_reason = "junction_route_recovery_slow"
            safety_reason = "junction_route_recovery_slow"
            cruise_allowed = False

        if cruise_allowed and not stop_request_out:
            target_speed = min(target_speed, self.config.cruise_speed_mps)
        elif speed_reason == "cruise":
            target_speed = min(target_speed, self.unstable_lane_speed_mps)
            speed_reason = "cruise_blocked"

        speed_setpoint_raw = target_speed
        if junction_offroute_safety_stop:
            self.speed_setpoint_smoothed = 0.0
            speed_setpoint_smoothed = 0.0
        else:
            speed_setpoint_smoothed = self.smooth_speed_setpoint(speed_setpoint_raw, dt)
        speed_state = speed_reason

        payload = {
            "stamp": now,
            "source": "lane_follower_node",
            "target_speed_mps": round(speed_setpoint_smoothed, 3),
            "stop_request": bool(stop_request_out),
            "stop_reason": stop_reason_out,
            "reason": speed_reason,
            "turn_state": self.turn_state,
            "speed_state": speed_state,
            "speed_setpoint_raw": round(float(speed_setpoint_raw), 3),
            "speed_setpoint_smoothed": round(float(speed_setpoint_smoothed), 3),
            "cruise_allowed": bool(cruise_allowed),
            "lane_transition_confirmed": bool(lane_transition_confirmed),
            "junction_entry_index": self.junction_entry_index,
            "junction_exit_index": self.junction_exit_index,
            "turn_arc_target_selected": bool(turn_arc_target_selected),
            "fallback_target_selected": bool(fallback_target_selected),
            "fallback_reason": fallback_reason,
            "cte_recovery_active": bool(cte_recovery_active),
            "soft_turn_alignment_active": bool(soft_turn_alignment_active),
            "severe_cte_recovery": bool(severe_cte_recovery),
            "junction_offroute_safety_stop": bool(junction_offroute_safety_stop),
            "off_route": bool(off_route),
            "route_replan_reason": route_replan_reason,
            "safety_reason": safety_reason,
            "cte_increasing": bool(cte_increasing),
            "startup_phase": bool(startup_phase),
            "startup_lane_lock_active": bool(startup_lane_lock_active),
            "lane_transition_detected": bool(lane_transition_detected),
            "target_override_reason": target_override_reason,
            "selected_fallback_reason": selected_fallback_reason,
            "is_junction": bool(is_junction_context),
            "upcoming_turn_type": turn_info.get("upcoming_turn_type"),
            "upcoming_turn_distance_m": turn_info.get("upcoming_turn_distance_m"),
            "route_option": turn_info.get("route_option"),
            "route_option_raw": route_option_raw,
            "route_option_filtered": route_option_filtered,
            "turn_slowdown_active": bool(turn_slowdown_active),
            "turn_speed_limit_mps": self.turn_speed_limit_mps,
            "approach_turn_speed_mps": self.approach_turn_speed_mps,
            "junction_turn_speed_mps": self.junction_turn_speed_mps,
            "exit_turn_speed_mps": self.exit_turn_speed_mps,
            "post_turn_speed_mps": self.post_turn_speed_mps,
            "lane_jump_rejected": bool(lane_jump_rejected),
            "target_jump_rejected": bool(target_jump_rejected),
            "target_is_behind_ego": bool(target_is_behind_ego),
            "dot_to_target": round(float(selected_dot_to_target), 3),
            "original_dot_to_target": round(float(original_dot_to_target), 3),
            "previous_road_id": previous_road_id,
            "previous_lane_id": previous_lane_id,
            "selected_target_road_id": selected_target.road_id,
            "selected_target_lane_id": selected_target.lane_id,
            "route_target_jump_distance": round(route_target_jump_distance, 3)
            if route_target_jump_distance is not None else None,
            "nearest_index": int(plan.nearest_index),
            "lookahead_index": int(plan.lookahead_index),
            "route_index": (self.route_payload or {}).get("objective_index"),
            "route_length": len(route_points),
            "route_age_ms": route_age_ms,
            "mission_status_age_ms": mission_status_age_ms,
            "ego_status_age_ms": ego_status_age_ms,
            "timeout_source": "",
            "timeout_threshold_ms": {
                "route": int(self.route_timeout_s * 1000.0),
                "mission_status": int(self.mission_timeout_s * 1000.0),
                "ego_status": int(self.status_timeout_s * 1000.0),
            },
            "lateral_error_m": round(plan.lateral_error_m, 3),
            "heading_error_deg": round(selected_heading_error, 3),
            "route_remaining_m": round(plan.route_remaining_m, 3),
            "distance_to_goal_m": plan.distance_to_goal_m,
            "lane_change_ahead": bool(plan.lane_change_ahead),
            "lane_change_reason": plan.lane_change_reason,
            "active_mission_target": self.active_mission_target_name(),
            "mission_state": (self.mission_payload or {}).get("stage"),
            "current_speed_mps": round(float(ego.speed_mps), 3),
            "ego_x": round(float(ego.x), 4),
            "ego_y": round(float(ego.y), 4),
            "ego_yaw": round(float(ego.yaw_deg), 4),
            **waypoint_info,
            "target_point": {
                "x": round(selected_target.x, 4),
                "y": round(selected_target.y, 4),
                "z": round(selected_target.z, 4),
                "yaw_deg": round(selected_target.yaw_deg, 4),
                "road_id": selected_target.road_id,
                "lane_id": selected_target.lane_id,
                "route_index": selected_target.route_index,
                "road_option": selected_target.road_option,
            },
        }
        self.publish_json(payload)
        self.log_runtime(payload)
        self.previous_abs_cte = abs_cte
        self.previous_road_id = current_road_id
        self.previous_lane_id = current_lane_id
        self.update_previous_target(selected_target)

    # -------------------------
    # Debug / publish block
    # -------------------------
    def publish_stop_plan(self, reason: str, ego: Optional[EgoPose] = None):
        point = {
            "x": ego.x if ego else 0.0,
            "y": ego.y if ego else 0.0,
            "z": 0.0,
            "yaw_deg": ego.yaw_deg if ego else 0.0,
        }
        now = time.time()
        route_age_ms = self.age_ms(now, self.last_route_time_s)
        mission_status_age_ms = self.age_ms(now, self.last_mission_time_s)
        ego_status_age_ms = self.age_ms(now, self.last_status_time_s)
        payload = {
            "stamp": time.time(),
            "source": "lane_follower_node",
            "target_speed_mps": 0.0,
            "stop_request": True,
            "stop_reason": reason,
            "reason": reason,
            "route_index": (self.mission_payload or {}).get("objective_index"),
            "route_length": len((self.route_payload or {}).get("points", [])),
            "route_age_ms": route_age_ms,
            "mission_status_age_ms": mission_status_age_ms,
            "ego_status_age_ms": ego_status_age_ms,
            "timeout_source": reason,
            "timeout_threshold_ms": {
                "route": int(self.route_timeout_s * 1000.0),
                "mission_status": int(self.mission_timeout_s * 1000.0),
                "ego_status": int(self.status_timeout_s * 1000.0),
            },
            "lateral_error_m": None,
            "heading_error_deg": None,
            "active_mission_target": self.active_mission_target_name(),
            "mission_state": (self.mission_payload or {}).get("stage"),
            "current_speed_mps": round(float(ego.speed_mps), 3) if ego else None,
            "ego_x": round(float(ego.x), 4) if ego else None,
            "ego_y": round(float(ego.y), 4) if ego else None,
            "ego_yaw": round(float(ego.yaw_deg), 4) if ego else None,
            **(self.current_carla_waypoint_info(ego) if ego else {}),
            "target_point": point,
        }
        self.publish_json(payload)
        self.log_runtime(payload)

    def publish_json(self, payload: dict):
        msg = String()
        msg.data = json.dumps(payload, ensure_ascii=False)
        self.plan_pub.publish(msg)

    def active_mission_target_name(self):
        target = (self.mission_payload or {}).get("objective_target") or {}
        return target.get("name")

    def log_runtime(self, payload: dict):
        target = payload.get("target_point") or {}
        record = {
            "ego_x": payload.get("ego_x"),
            "ego_y": payload.get("ego_y"),
            "ego_yaw": payload.get("ego_yaw"),
            "current_speed_mps": payload.get("current_speed_mps"),
            "active_mission_target": payload.get("active_mission_target"),
            "mission_state": payload.get("mission_state"),
            "route_index": payload.get("route_index"),
            "route_length": payload.get("route_length"),
            "route_age_ms": payload.get("route_age_ms"),
            "mission_status_age_ms": payload.get("mission_status_age_ms"),
            "ego_status_age_ms": payload.get("ego_status_age_ms"),
            "timeout_source": payload.get("timeout_source"),
            "timeout_threshold_ms": payload.get("timeout_threshold_ms"),
            "startup_phase": payload.get("startup_phase"),
            "startup_lane_lock_active": payload.get("startup_lane_lock_active"),
            "turn_state": payload.get("turn_state"),
            "speed_state": payload.get("speed_state"),
            "speed_setpoint_raw": payload.get("speed_setpoint_raw"),
            "speed_setpoint_smoothed": payload.get("speed_setpoint_smoothed"),
            "cruise_allowed": payload.get("cruise_allowed"),
            "lane_transition_confirmed": payload.get("lane_transition_confirmed"),
            "junction_entry_index": payload.get("junction_entry_index"),
            "junction_exit_index": payload.get("junction_exit_index"),
            "turn_arc_target_selected": payload.get("turn_arc_target_selected"),
            "fallback_target_selected": payload.get("fallback_target_selected"),
            "fallback_reason": payload.get("fallback_reason"),
            "cte_recovery_active": payload.get("cte_recovery_active"),
            "soft_turn_alignment_active": payload.get("soft_turn_alignment_active"),
            "severe_cte_recovery": payload.get("severe_cte_recovery"),
            "junction_offroute_safety_stop": payload.get("junction_offroute_safety_stop"),
            "off_route": payload.get("off_route"),
            "route_replan_reason": payload.get("route_replan_reason"),
            "safety_reason": payload.get("safety_reason"),
            "cte_increasing": payload.get("cte_increasing"),
            "lane_transition_detected": payload.get("lane_transition_detected"),
            "target_override_reason": payload.get("target_override_reason"),
            "previous_road_id": payload.get("previous_road_id"),
            "previous_lane_id": payload.get("previous_lane_id"),
            "selected_target_road_id": payload.get("selected_target_road_id"),
            "selected_target_lane_id": payload.get("selected_target_lane_id"),
            "is_junction": payload.get("is_junction"),
            "upcoming_turn_type": payload.get("upcoming_turn_type"),
            "upcoming_turn_distance_m": payload.get("upcoming_turn_distance_m"),
            "route_option": payload.get("route_option"),
            "route_option_raw": payload.get("route_option_raw"),
            "route_option_filtered": payload.get("route_option_filtered"),
            "turn_slowdown_active": payload.get("turn_slowdown_active"),
            "turn_speed_limit_mps": payload.get("turn_speed_limit_mps"),
            "approach_turn_speed_mps": payload.get("approach_turn_speed_mps"),
            "junction_turn_speed_mps": payload.get("junction_turn_speed_mps"),
            "exit_turn_speed_mps": payload.get("exit_turn_speed_mps"),
            "post_turn_speed_mps": payload.get("post_turn_speed_mps"),
            "lane_jump_rejected": payload.get("lane_jump_rejected"),
            "target_jump_rejected": payload.get("target_jump_rejected"),
            "target_is_behind_ego": payload.get("target_is_behind_ego"),
            "dot_to_target": payload.get("dot_to_target"),
            "original_dot_to_target": payload.get("original_dot_to_target"),
            "selected_fallback_reason": payload.get("selected_fallback_reason"),
            "route_target_jump_distance": payload.get("route_target_jump_distance"),
            "lane_target_x": target.get("x"),
            "lane_target_y": target.get("y"),
            "current_road_id": payload.get("current_road_id"),
            "current_lane_id": payload.get("current_lane_id"),
            "target_road_id": target.get("road_id"),
            "target_lane_id": target.get("lane_id"),
            "lane_width": payload.get("lane_width"),
            "actual_lateral_distance_to_lane_center": payload.get(
                "actual_lateral_distance_to_lane_center"
            ),
            "current_lane_center_x": payload.get("current_lane_center_x"),
            "current_lane_center_y": payload.get("current_lane_center_y"),
            "cross_track_error": payload.get("lateral_error_m"),
            "heading_error": payload.get("heading_error_deg"),
            "target_speed_mps": payload.get("target_speed_mps"),
            "speed_reason": payload.get("reason"),
            "stop_request": payload.get("stop_request"),
            "stop_reason": payload.get("stop_reason"),
            "lane_change_ahead": payload.get("lane_change_ahead"),
            "lane_change_reason": payload.get("lane_change_reason"),
            "nearest_index": payload.get("nearest_index"),
            "lookahead_index": payload.get("lookahead_index"),
        }
        self.runtime_logger.write(record)

        now = time.time()
        if now - self.last_ros_log_s >= self.ros_log_period_s:
            self.last_ros_log_s = now
            self.get_logger().info(
                "lane "
                f"state={record['mission_state']} target={record['active_mission_target']} "
                f"speed={record['target_speed_mps']} reason={record['speed_reason']} "
                f"stop={record['stop_request']} cte={record['cross_track_error']} "
                f"hdg={record['heading_error']} lane={record['current_road_id']}/"
                f"{record['current_lane_id']} target_lane={record['target_road_id']}/"
                f"{record['target_lane_id']} actual_lat="
                f"{record['actual_lateral_distance_to_lane_center']} turn_state="
                f"{record['turn_state']} turn={record['upcoming_turn_type']}@"
                f"{record['upcoming_turn_distance_m']} speed_state={record['speed_state']} "
                f"cruise={record['cruise_allowed']} fallback={record['fallback_reason']} "
                f"junction_offroute_safety_stop={record['junction_offroute_safety_stop']} "
                f"off_route={record['off_route']} safety_reason={record['safety_reason']}"
            )


def main(args=None):
    rclpy.init(args=args)
    node = LaneFollowerNode()

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
