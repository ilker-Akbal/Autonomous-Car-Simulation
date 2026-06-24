import copy
import json
import math
import time
from typing import Any, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_planning.route_geometry import (
    build_local_route_segment,
    forward_window_search,
)
from teknofest_sim.carla_loader import load_carla


class RouteSampler(Node):
    def __init__(self):
        super().__init__("route_sampler")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("route_source_mode", "global")
        self.declare_parameter("fallback_to_simple_forward_route", False)
        self.declare_parameter("disable_fallback_driving", True)
        self.declare_parameter("disable_fallback_driving_when_mission_missing", True)
        self.declare_parameter("global_route_stale_timeout_s", 8.0)
        self.declare_parameter("mission_goal_near_distance_m", 3.0)
        self.declare_parameter("route_end_requires_goal_near", True)
        self.declare_parameter("replan_when_local_route_exhausted", True)
        self.declare_parameter("local_route_short_goal_far_replan_distance_m", 8.0)
        self.declare_parameter("local_route_horizon_m", 80.0)
        self.declare_parameter("min_route_points", 8)
        self.declare_parameter("hold_last_route_s", 2.0)
        self.declare_parameter("rate_hz", 5.0)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.route_source_mode = self.get_parameter("route_source_mode").value
        self.fallback_to_simple_forward_route = bool(self.get_parameter("fallback_to_simple_forward_route").value)
        self.disable_fallback_driving = bool(self.get_parameter("disable_fallback_driving").value)
        self.disable_fallback_driving_when_mission_missing = bool(
            self.get_parameter("disable_fallback_driving_when_mission_missing").value
        )
        self.global_route_stale_timeout_s = float(self.get_parameter("global_route_stale_timeout_s").value)
        self.mission_goal_near_distance_m = float(self.get_parameter("mission_goal_near_distance_m").value)
        self.route_end_requires_goal_near = bool(self.get_parameter("route_end_requires_goal_near").value)
        self.replan_when_local_route_exhausted = bool(
            self.get_parameter("replan_when_local_route_exhausted").value
        )
        self.local_route_short_goal_far_replan_distance_m = float(
            self.get_parameter("local_route_short_goal_far_replan_distance_m").value
        )
        self.local_route_horizon_m = float(self.get_parameter("local_route_horizon_m").value)
        self.min_route_points = int(self.get_parameter("min_route_points").value)
        self.hold_last_route_s = float(self.get_parameter("hold_last_route_s").value)
        self.rate_hz = float(self.get_parameter("rate_hz").value)

        self._last_status: Optional[dict[str, Any]] = None
        self._last_mission_goal: Optional[dict[str, Any]] = None
        self._last_mission_goal_time = 0.0
        self._last_global_route: Optional[dict[str, Any]] = None
        self._last_route_time = 0.0
        self._last_valid_global_route: Optional[dict[str, Any]] = None
        self._last_valid_global_route_time = 0.0
        self._last_index = 0
        self._last_valid_local_route: Optional[dict[str, Any]] = None
        self._last_valid_local_route_time = 0.0

        self._carla = None
        self._client = None
        self._world = None
        self._map = None

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_subscription(String, "/adas/mission/current_goal", self._mission_goal_callback, 10)
        self.create_subscription(String, "/adas/planning/global_route", self._global_route_callback, 10)

        self.route_pub = self.create_publisher(String, "/adas/planning/route", 10)
        self.route_debug_pub = self.create_publisher(String, "/adas/planning/route_debug", 10)

        self._connect_to_carla()
        timer_period_s = 1.0 / max(0.1, self.rate_hz)
        self.timer = self.create_timer(timer_period_s, self._tick)

    def _connect_to_carla(self) -> None:
        try:
            self._carla = load_carla(self.carla_root)
            self._client = self._carla.Client(self.host, self.port)
            self._client.set_timeout(10.0)
            self._world = self._client.get_world()
            self._map = self._world.get_map()
        except Exception as exc:
            self.get_logger().warn(f"RouteSampler: CARLA connection failed: {exc}")
            self._carla = None
            self._client = None
            self._world = None
            self._map = None

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("RouteSampler: failed to parse /adas/carla/status JSON")

    def _global_route_callback(self, msg: String) -> None:
        try:
            self._last_global_route = json.loads(msg.data)
            self._last_route_time = time.time()
            points = self._last_global_route.get("points", [])
            route_ok = bool(
                self._last_global_route.get("ok", self._last_global_route.get("route_ok", False))
            )
            if route_ok and len(points) > 0:
                self._last_valid_global_route = copy.deepcopy(self._last_global_route)
                self._last_valid_global_route_time = self._last_route_time
        except Exception:
            self.get_logger().warn("RouteSampler: failed to parse /adas/planning/global_route JSON")

    def _mission_goal_callback(self, msg: String) -> None:
        try:
            self._last_mission_goal = json.loads(msg.data)
            self._last_mission_goal_time = time.time()
        except Exception:
            self.get_logger().warn("RouteSampler: failed to parse /adas/mission/current_goal JSON")

    def _get_ego_location(self) -> Optional[tuple[float, float, float]]:
        if self._last_status is None:
            return None
        loc = self._last_status.get("location", {})
        if "x" not in loc or "y" not in loc:
            return None
        return float(loc["x"]), float(loc["y"]), float(loc.get("z", 0.0))

    def _goal_xy(self) -> tuple[Optional[float], Optional[float]]:
        if self._last_mission_goal is None:
            return None, None
        goal = self._last_mission_goal.get("current_goal")
        if not isinstance(goal, dict):
            goal = self._last_mission_goal
        x = goal.get("carla_x", self._last_mission_goal.get("target_x"))
        y = goal.get("carla_y", self._last_mission_goal.get("target_y"))
        if x is None or y is None:
            return None, None
        return float(x), float(y)

    def _mission_goal_fields(self) -> dict[str, Any]:
        if self._last_mission_goal is None:
            return {
                "goal_kind": None,
                "task_stop_required": False,
                "task_stop_x": None,
                "task_stop_y": None,
                "task_stop_z": None,
                "task_stop_yaw": None,
                "task_stop_side": None,
                "task_stop_mode": None,
                "task_stop_source": None,
                "task_hold_s": None,
                "mission_approach_active": False,
                "mission_hold_active": False,
                "mission_reached": False,
                "task_hold_remaining_s": None,
                "base_goal_x": None,
                "base_goal_y": None,
                "base_goal_yaw": None,
                "effective_task_stop_x": None,
                "effective_task_stop_y": None,
                "effective_task_stop_z": None,
                "effective_task_stop_yaw": None,
                "effective_task_stop_source": None,
                "task_stop_yaw_error_deg": None,
                "task_stop_yaw_within_tolerance": None,
                "task_stop_yaw_tolerance_deg": None,
                "center_distance_to_effective_task_stop_m": None,
                "front_bumper_distance_to_effective_task_stop_m": None,
                "task_stop_reached_by_mission": False,
                "mission_hold_start_allowed": False,
                "mission_hold_block_reason": None,
            }
        payload = self._last_mission_goal
        goal = payload.get("current_goal")
        if not isinstance(goal, dict):
            goal = payload
        goal_kind = payload.get("goal_kind", goal.get("kind"))
        return {
            "goal_kind": goal_kind,
            "task_stop_required": bool(payload.get("task_stop_required", False)),
            "task_stop_x": payload.get("task_stop_x", goal.get("task_stop_x", goal.get("carla_x"))),
            "task_stop_y": payload.get("task_stop_y", goal.get("task_stop_y", goal.get("carla_y"))),
            "task_stop_z": payload.get("task_stop_z", goal.get("task_stop_z", goal.get("carla_z"))),
            "task_stop_yaw": payload.get("task_stop_yaw", goal.get("task_stop_yaw", goal.get("carla_yaw"))),
            "task_stop_side": payload.get("task_stop_side", goal.get("task_stop_side")),
            "task_stop_mode": payload.get("task_stop_mode", goal.get("task_stop_mode")),
            "task_stop_source": payload.get("task_stop_source", goal.get("task_stop_source")),
            "task_hold_s": payload.get("task_hold_s", goal.get("task_hold_s")),
            "mission_approach_active": bool(payload.get("mission_approach_active", False)),
            "mission_hold_active": bool(payload.get("mission_hold_active", False)),
            "mission_reached": bool(payload.get("mission_reached", False)),
            "task_hold_remaining_s": payload.get("mission_hold_remaining_s"),
            "base_goal_x": payload.get("base_goal_x", goal.get("carla_x")),
            "base_goal_y": payload.get("base_goal_y", goal.get("carla_y")),
            "base_goal_yaw": payload.get("base_goal_yaw", goal.get("carla_yaw")),
            "effective_task_stop_x": payload.get("effective_task_stop_x"),
            "effective_task_stop_y": payload.get("effective_task_stop_y"),
            "effective_task_stop_z": payload.get("effective_task_stop_z"),
            "effective_task_stop_yaw": payload.get("effective_task_stop_yaw"),
            "effective_task_stop_source": payload.get("effective_task_stop_source"),
            "task_stop_yaw_error_deg": payload.get("task_stop_yaw_error_deg"),
            "task_stop_yaw_within_tolerance": payload.get("task_stop_yaw_within_tolerance"),
            "task_stop_yaw_tolerance_deg": payload.get("task_stop_yaw_tolerance_deg"),
            "center_distance_to_effective_task_stop_m": payload.get(
                "center_distance_to_effective_task_stop_m"
            ),
            "front_bumper_distance_to_effective_task_stop_m": payload.get(
                "front_bumper_distance_to_effective_task_stop_m"
            ),
            "task_stop_reached_by_mission": bool(
                payload.get("task_stop_reached_by_mission", False)
            ),
            "mission_hold_start_allowed": bool(
                payload.get("mission_hold_start_allowed", False)
            ),
            "mission_hold_block_reason": payload.get("mission_hold_block_reason"),
        }

    def _distance_to_goal(self, ego_x: float, ego_y: float) -> tuple[Optional[float], Optional[float], Optional[float]]:
        goal_x, goal_y = self._goal_xy()
        if goal_x is None or goal_y is None:
            return None, None, None
        return math.hypot(goal_x - ego_x, goal_y - ego_y), goal_x, goal_y

    def _mission_stop_active(self) -> bool:
        if self._last_mission_goal is None:
            return False
        if time.time() - self._last_mission_goal_time > 2.0:
            return False
        return bool(self._last_mission_goal.get("mission_stop_active", False))

    def _build_forward_route(self) -> dict[str, Any]:
        if self._map is None or self._last_status is None:
            return self._empty_route_payload("fallback")

        ego_loc = self._get_ego_location()
        if ego_loc is None:
            return self._empty_route_payload("fallback")

        carla = self._carla
        if carla is None:
            return self._empty_route_payload("fallback")

        try:
            start_waypoint = self._map.get_waypoint(
                carla.Location(x=ego_loc[0], y=ego_loc[1], z=ego_loc[2]),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
        except Exception as exc:
            self.get_logger().warn(f"RouteSampler: fallback waypoint lookup failed: {exc}")
            return self._empty_route_payload("fallback")

        points: List[dict[str, Any]] = []
        distance = 0.0
        previous = start_waypoint
        step_m = 2.0
        max_steps = int(self.local_route_horizon_m / max(0.001, step_m))
        yaw = previous.transform.rotation.yaw

        for _ in range(max_steps):
            next_waypoints = previous.next(step_m)
            if not next_waypoints:
                break
            best = None
            best_delta = None
            for candidate in next_waypoints:
                delta = abs((candidate.transform.rotation.yaw - yaw + 180.0) % 360.0 - 180.0)
                if best is None or delta < best_delta:
                    best = candidate
                    best_delta = delta
            if best is None:
                break
            distance += step_m
            previous = best
            yaw = previous.transform.rotation.yaw
            points.append(
                {
                    "x": round(previous.transform.location.x, 3),
                    "y": round(previous.transform.location.y, 3),
                    "z": round(previous.transform.location.z, 3),
                    "yaw": round(previous.transform.rotation.yaw, 3),
                    "road_id": previous.road_id,
                    "lane_id": previous.lane_id,
                    "selected_road_id": previous.road_id,
                    "selected_lane_id": previous.lane_id,
                    "lane_preference": "right",
                    "right_lane_selected": False,
                    "right_lane_reason": "fallback_forward_route",
                    "selected_lane_lateral_right_m": 0.0,
                    "candidate_lane_ids": [previous.lane_id],
                    "candidate_lane_lateral_right_m": [0.0],
                    "s": round(distance, 3),
                    "right_lane_policy_enabled": False,
                    "original_lane_id": previous.lane_id,
                    "is_junction": bool(getattr(previous, "is_junction", False)),
                    "right_lane_candidate_found": False,
                    "wrong_way_rejected": False,
                    "right_lane_projection_applied": False,
                    "right_lane_projection_failed_reason": "fallback_forward_route",
                    "route_continuity_ok": True,
                }
            )

        return {
            "stamp": time.time(),
            "source": "fallback_forward_route",
            "route_source": "fallback",
            "points": points,
            "route_len": len(points),
            "route_ok": len(points) >= self.min_route_points,
            "status_ok": True,
            "fallback_used": True,
            "global_route_ok": False,
            "held_last_route": False,
            "right_lane_policy_enabled": False,
            "right_lane_projection_failed_reason": "fallback_forward_route",
            "route_continuity_ok": True,
            "sign_constraints_enabled": False,
            "sign_constraints_loaded": False,
            "sign_constraints_count": 0,
            "active_sign_constraints": [],
            "sign_constraint_replan_requested": False,
            "sign_constraint_replan_reason": None,
            "forbidden_road_lane_rejected": False,
            "forbidden_turn_rejected": False,
            "speed_limit_annotation": False,
            "stop_yield_annotation": False,
            "park_restriction_annotation": False,
        }

    def _empty_route_payload(self, source: str) -> dict[str, Any]:
        now = time.time()
        ego_loc = self._get_ego_location()
        if ego_loc is not None:
            distance_to_goal_m, goal_x, goal_y = self._distance_to_goal(ego_loc[0], ego_loc[1])
            ego_x = ego_loc[0]
            ego_y = ego_loc[1]
        else:
            distance_to_goal_m, goal_x, goal_y = None, None, None
            ego_x, ego_y = None, None
        last_global_route_age_s = now - self._last_route_time if self._last_route_time else None
        last_valid_global_route_age_s = (
            now - self._last_valid_global_route_time
            if self._last_valid_global_route_time
            else None
        )
        last_global_route_len = (
            len(self._last_global_route.get("points", []))
            if isinstance(self._last_global_route, dict)
            else 0
        )
        return {
            "stamp": now,
            "source": source,
            "route_source": source,
            "route_missing_reason": source,
            "ego_x": round(ego_x, 3) if ego_x is not None else None,
            "ego_y": round(ego_y, 3) if ego_y is not None else None,
            "goal_x": round(goal_x, 3) if goal_x is not None else None,
            "goal_y": round(goal_y, 3) if goal_y is not None else None,
            "distance_to_goal_m": round(distance_to_goal_m, 3) if distance_to_goal_m is not None else None,
            "mission_goal_near_distance_m": self.mission_goal_near_distance_m,
            "terminal_route": False,
            "route_end_near_goal": False,
            "route_end_reason": source,
            "replan_recommended": False,
            "replan_reason": None,
            "points": [],
            "route_len": 0,
            "route_ok": False,
            "status_ok": self._last_status is not None,
            "fallback_used": False,
            "global_route_ok": False,
            "global_route_cache_valid": False,
            "last_global_route_age_s": round(last_global_route_age_s, 3) if last_global_route_age_s is not None else None,
            "last_valid_global_route_age_s": round(last_valid_global_route_age_s, 3) if last_valid_global_route_age_s is not None else None,
            "last_global_route_len": last_global_route_len,
            "held_last_route": False,
            "right_lane_policy_enabled": False,
            "right_lane_projection_failed_reason": source,
            "route_continuity_ok": False,
            "sign_constraints_enabled": False,
            "sign_constraints_loaded": False,
            "sign_constraints_count": 0,
            "active_sign_constraints": [],
            "sign_constraint_replan_requested": False,
            "sign_constraint_replan_reason": None,
            "forbidden_road_lane_rejected": False,
            "forbidden_turn_rejected": False,
            "speed_limit_annotation": False,
            "stop_yield_annotation": False,
            "park_restriction_annotation": False,
        }

    def _missing_route_source(self, route: dict[str, Any]) -> str:
        if not isinstance(route, dict) or not route:
            return "global_route_missing"
        if isinstance(route, dict) and route:
            route_ok = bool(route.get("ok", route.get("route_ok", False)))
            if not route_ok:
                return "global_route_invalid"
        reason = str(route.get("replan_reason") or "")
        if "mission" in reason or route.get("goal_name") is None:
            return "mission_missing"
        return "global_route_missing"

    def _valid_global_route_cache(self) -> Optional[dict[str, Any]]:
        if self._last_valid_global_route is None:
            return None
        age_s = time.time() - self._last_valid_global_route_time
        if age_s > self.global_route_stale_timeout_s:
            return None
        cached = copy.deepcopy(self._last_valid_global_route)
        cached["global_route_cache_valid"] = True
        cached["global_route_cache_age_s"] = round(age_s, 3)
        return cached

    def _route_is_valid(self, payload: dict[str, Any]) -> bool:
        points = payload.get("points", [])
        return bool(payload.get("route_ok")) and len(points) >= self.min_route_points

    def _remember_valid_route(self, payload: dict[str, Any]) -> dict[str, Any]:
        self._last_valid_local_route = copy.deepcopy(payload)
        self._last_valid_local_route_time = time.time()
        return payload

    def _hold_last_valid_route(self, global_route_ok: bool) -> Optional[dict[str, Any]]:
        if self._last_valid_local_route is None:
            return None

        route_age_s = time.time() - self._last_valid_local_route_time
        if route_age_s > self.hold_last_route_s:
            return None

        payload = copy.deepcopy(self._last_valid_local_route)
        payload["stamp"] = time.time()
        payload["source"] = "route_sampler_hold_last"
        payload["held_last_route"] = True
        payload["held_route_age_s"] = round(route_age_s, 3)
        payload["global_route_ok"] = global_route_ok
        return payload

    def _select_local_route(self) -> dict[str, Any]:
        ego_loc = self._get_ego_location()
        if ego_loc is None:
            held_route = self._hold_last_valid_route(global_route_ok=False)
            return held_route or self._empty_route_payload("route_sampler")

        x, y, _ = ego_loc
        route = self._last_global_route or {}
        points = route.get("points", []) if isinstance(route, dict) else []
        global_route_ok = bool(
            isinstance(route, dict)
            and route.get("ok", route.get("route_ok", False))
            and points
        )

        if self.route_source_mode != "global" or not global_route_ok:
            cached_route = self._valid_global_route_cache()
            if cached_route is not None:
                route = cached_route
                points = route.get("points", [])
                global_route_ok = bool(points)
            elif self.disable_fallback_driving or self.disable_fallback_driving_when_mission_missing:
                source = self._missing_route_source(route)
                payload = self._empty_route_payload(source)
                payload["route_missing_reason"] = source
                return payload
            else:
                global_route_ok = False

        if self.route_source_mode != "global" or not global_route_ok:
            if self.fallback_to_simple_forward_route:
                fallback_route = self._build_forward_route()
                if self._route_is_valid(fallback_route):
                    return self._remember_valid_route(fallback_route)

            held_route = self._hold_last_valid_route(global_route_ok=False)
            if held_route is not None:
                return held_route
            return self._empty_route_payload(
                "fallback" if self.fallback_to_simple_forward_route else "route_invalid"
            )

        nearest_index = forward_window_search(points, x, y, self._last_index, window=12, reset_threshold_m=10.0)
        self._last_index = nearest_index
        local_points = build_local_route_segment(points, nearest_index, self.local_route_horizon_m)
        distance_to_goal_m, goal_x, goal_y = self._distance_to_goal(x, y)
        mission_fields = self._mission_goal_fields()
        mission_stop_active = self._mission_stop_active()
        task_stop_required = bool(mission_fields.get("task_stop_required", False))
        goal_kind = str(mission_fields.get("goal_kind") or "")
        task_goal = task_stop_required and goal_kind in ("pickup", "dropoff")
        task_stop_reached = bool(mission_fields.get("task_stop_reached_by_mission", False))
        task_pull_over_incomplete = task_goal and not task_stop_reached
        terminal_route = nearest_index >= max(0, len(points) - self.min_route_points) or len(local_points) < self.min_route_points
        goal_near = (
            distance_to_goal_m is not None
            and distance_to_goal_m <= self.mission_goal_near_distance_m
        )
        route_end_near = terminal_route and (
            mission_stop_active
            or goal_near
            or not self.route_end_requires_goal_near
        )
        if task_pull_over_incomplete:
            route_end_near = False
        local_route_short_goal_far = (
            terminal_route
            and not route_end_near
            and distance_to_goal_m is not None
            and distance_to_goal_m > self.mission_goal_near_distance_m
        )
        replan_recommended = (
            self.replan_when_local_route_exhausted
            and local_route_short_goal_far
            and distance_to_goal_m >= self.local_route_short_goal_far_replan_distance_m
        )
        route_end_reason = None
        if route_end_near:
            route_end_reason = "mission_stop_active" if mission_stop_active else "goal_near"
        elif local_route_short_goal_far:
            route_end_reason = "local_route_short_but_goal_far"
        global_route_cache_age_s = route.get("global_route_cache_age_s")
        global_route_cache_valid = bool(route.get("global_route_cache_valid", False))
        last_global_route_age_s = time.time() - self._last_route_time if self._last_route_time else None

        payload = {
            "stamp": time.time(),
            "source": "phase2c_route_sampler",
            "route_source": "route_end_near_goal" if route_end_near else "global_route",
            "route_missing_reason": "route_end_near_goal" if route_end_near else (
                "local_route_short_but_goal_far" if local_route_short_goal_far else None
            ),
            "nearest_index": nearest_index,
            "points": local_points,
            "route_len": len(local_points),
            "route_ok": (len(local_points) >= self.min_route_points) or (
                local_route_short_goal_far and len(local_points) > 0
            ) or (
                task_pull_over_incomplete and len(local_points) > 0
            ),
            "status_ok": self._last_status is not None,
            "ego_x": round(x, 3),
            "ego_y": round(y, 3),
            "goal_x": round(goal_x, 3) if goal_x is not None else None,
            "goal_y": round(goal_y, 3) if goal_y is not None else None,
            "distance_to_goal_m": round(distance_to_goal_m, 3) if distance_to_goal_m is not None else None,
            "mission_goal_near_distance_m": self.mission_goal_near_distance_m,
            "terminal_route": terminal_route,
            "route_end_near_goal": route_end_near,
            "route_end_reason": route_end_reason,
            "mission_stop_active": mission_stop_active,
            **mission_fields,
            "replan_recommended": replan_recommended,
            "replan_reason": "local_route_exhausted_before_goal" if replan_recommended else None,
            "fallback_used": False,
            "global_route_ok": True,
            "global_route_cache_valid": global_route_cache_valid,
            "global_route_cache_age_s": global_route_cache_age_s,
            "last_global_route_age_s": round(last_global_route_age_s, 3) if last_global_route_age_s is not None else None,
            "last_global_route_len": len(points),
            "held_last_route": False,
            "right_lane_policy_enabled": route.get("right_lane_policy_enabled", False),
            "right_lane_projection_count": route.get("right_lane_projection_count", 0),
            "right_lane_projection_failed_count": route.get("right_lane_projection_failed_count", 0),
            "right_lane_projection_partial_fallback_count": route.get("right_lane_projection_partial_fallback_count", 0),
            "right_lane_candidate_count": route.get("right_lane_candidate_count", 0),
            "wrong_way_rejected_count": route.get("wrong_way_rejected_count", 0),
            "right_lane_projection_failed_reason": route.get("right_lane_projection_failed_reason"),
            "route_continuity_ok": route.get("route_continuity_ok", False),
            "right_lane_route_continuity_ok": route.get("right_lane_route_continuity_ok", route.get("route_continuity_ok", False)),
            "sign_constraints_enabled": route.get("sign_constraints_enabled", False),
            "sign_constraints_loaded": route.get("sign_constraints_loaded", False),
            "sign_constraints_count": route.get("sign_constraints_count", 0),
            "active_sign_constraints": route.get("active_sign_constraints", []),
            "sign_constraint_replan_requested": route.get("sign_constraint_replan_requested", False),
            "sign_constraint_replan_reason": route.get("sign_constraint_replan_reason"),
            "forbidden_road_lane_rejected": route.get("forbidden_road_lane_rejected", False),
            "forbidden_turn_rejected": route.get("forbidden_turn_rejected", False),
            "speed_limit_annotation": route.get("speed_limit_annotation", False),
            "stop_yield_annotation": route.get("stop_yield_annotation", False),
            "park_restriction_annotation": route.get("park_restriction_annotation", False),
        }
        if route_end_near:
            payload["route_ok"] = False
            payload["global_route_ok"] = True
            payload["route_missing_reason"] = "route_end_near_goal"
            return payload

        if local_route_short_goal_far and len(local_points) > 0:
            return payload

        if self._route_is_valid(payload):
            return self._remember_valid_route(payload)

        if not self.disable_fallback_driving and self.fallback_to_simple_forward_route:
            fallback_route = self._build_forward_route()
            if self._route_is_valid(fallback_route):
                return self._remember_valid_route(fallback_route)

        held_route = self._hold_last_valid_route(global_route_ok=True)
        return held_route or payload

    def _tick(self) -> None:
        if self._client is None or self._map is None:
            self._connect_to_carla()

        route_payload = self._select_local_route()
        self.route_pub.publish(String(data=json.dumps(route_payload)))

        debug_payload = {
            "stamp": time.time(),
            "source": route_payload.get("source"),
            "route_source": route_payload.get("route_source"),
            "route_missing_reason": route_payload.get("route_missing_reason"),
            "route_len": route_payload.get("route_len"),
            "nearest_index": route_payload.get("nearest_index"),
            "route_ok": route_payload.get("route_ok"),
            "status_ok": route_payload.get("status_ok"),
            "ego_x": route_payload.get("ego_x"),
            "ego_y": route_payload.get("ego_y"),
            "goal_x": route_payload.get("goal_x"),
            "goal_y": route_payload.get("goal_y"),
            "distance_to_goal_m": route_payload.get("distance_to_goal_m"),
            "mission_goal_near_distance_m": route_payload.get("mission_goal_near_distance_m"),
            "terminal_route": route_payload.get("terminal_route", False),
            "route_end_near_goal": route_payload.get("route_end_near_goal", False),
            "route_end_reason": route_payload.get("route_end_reason"),
            "replan_recommended": route_payload.get("replan_recommended", False),
            "replan_reason": route_payload.get("replan_reason"),
            "fallback_used": route_payload.get("fallback_used", False),
            "global_route_ok": route_payload.get("global_route_ok", False),
            "global_route_cache_valid": route_payload.get("global_route_cache_valid", False),
            "global_route_cache_age_s": route_payload.get("global_route_cache_age_s"),
            "last_global_route_age_s": route_payload.get("last_global_route_age_s"),
            "last_valid_global_route_age_s": route_payload.get("last_valid_global_route_age_s"),
            "last_global_route_len": route_payload.get("last_global_route_len"),
            "global_route_stale_timeout_s": self.global_route_stale_timeout_s,
            "held_last_route": route_payload.get("held_last_route", False),
            "held_route_age_s": route_payload.get("held_route_age_s"),
            "hold_last_route_s": self.hold_last_route_s,
            "fallback_to_simple_forward_route": self.fallback_to_simple_forward_route,
            "disable_fallback_driving": self.disable_fallback_driving,
            "disable_fallback_driving_when_mission_missing": self.disable_fallback_driving_when_mission_missing,
            "right_lane_policy_enabled": route_payload.get("right_lane_policy_enabled", False),
            "right_lane_projection_count": route_payload.get("right_lane_projection_count", 0),
            "right_lane_projection_failed_count": route_payload.get("right_lane_projection_failed_count", 0),
            "right_lane_projection_partial_fallback_count": route_payload.get("right_lane_projection_partial_fallback_count", 0),
            "right_lane_candidate_count": route_payload.get("right_lane_candidate_count", 0),
            "wrong_way_rejected_count": route_payload.get("wrong_way_rejected_count", 0),
            "right_lane_projection_failed_reason": route_payload.get("right_lane_projection_failed_reason"),
            "route_continuity_ok": route_payload.get("route_continuity_ok", False),
            "right_lane_route_continuity_ok": route_payload.get("right_lane_route_continuity_ok", False),
            "sign_constraints_enabled": route_payload.get("sign_constraints_enabled", False),
            "sign_constraints_loaded": route_payload.get("sign_constraints_loaded", False),
            "sign_constraints_count": route_payload.get("sign_constraints_count", 0),
            "active_sign_constraints": route_payload.get("active_sign_constraints", []),
            "sign_constraint_replan_requested": route_payload.get("sign_constraint_replan_requested", False),
            "sign_constraint_replan_reason": route_payload.get("sign_constraint_replan_reason"),
            "forbidden_road_lane_rejected": route_payload.get("forbidden_road_lane_rejected", False),
            "forbidden_turn_rejected": route_payload.get("forbidden_turn_rejected", False),
            "speed_limit_annotation": route_payload.get("speed_limit_annotation", False),
            "stop_yield_annotation": route_payload.get("stop_yield_annotation", False),
            "park_restriction_annotation": route_payload.get("park_restriction_annotation", False),
        }
        self.route_debug_pub.publish(String(data=json.dumps(debug_payload)))


def main(args=None):
    rclpy.init(args=args)
    node = RouteSampler()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
