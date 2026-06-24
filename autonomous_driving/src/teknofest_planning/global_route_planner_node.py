import glob
import json
import math
import os
import sys
import time
from typing import Any, List, Optional

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from std_msgs.msg import String

from teknofest_sim.carla_loader import load_carla
from teknofest_planning.route_geometry import (
    angle_diff_deg,
    classify_turn_direction,
    distance_2d,
    nearest_point_distance,
    route_continuity_ok,
)
from teknofest_planning.sign_constraints import (
    HARD_SEGMENT_CONSTRAINTS,
    SignConstraintLoader,
    SignConstraintSet,
)


class GlobalRoutePlannerNode(Node):
    def __init__(self):
        super().__init__("global_route_planner")

        self.declare_parameter(
            "carla_root",
            "/home/ilker/simulators/CARLA_0.9.15",
        )
        self.declare_parameter("host", "127.0.0.1")
        self.declare_parameter("port", 2000)
        self.declare_parameter("ego_role_name", "ego_vehicle")
        self.declare_parameter("global_sampling_resolution_m", 2.0)
        self.declare_parameter("global_route_publish_hz", 1.0)
        self.declare_parameter("replan_period_s", 3.0)
        self.declare_parameter("replan_distance_threshold_m", 8.0)
        self.declare_parameter("goal_change_replan", True)
        self.declare_parameter("route_timeout_s", 10.0)
        self.declare_parameter("min_route_points", 8)
        self.declare_parameter("right_lane_policy_enabled", True)
        self.declare_parameter("preferred_lane_side", "right")
        self.declare_parameter("right_lane_projection_max_distance_m", 3.5)
        self.declare_parameter("right_lane_policy_disable_in_junction", True)
        self.declare_parameter("route_initial_wrong_way_reject_deg", 120.0)
        self.declare_parameter("sign_constraints_enabled", False)
        self.declare_parameter("sign_plan_geojson", "")
        self.declare_parameter("sign_plan_json", "")
        self.declare_parameter("sign_constraint_effective_radius_m", 12.0)
        self.declare_parameter("sign_constraint_debug", True)

        self.carla_root = self.get_parameter("carla_root").value
        self.host = self.get_parameter("host").value
        self.port = int(self.get_parameter("port").value)
        self.ego_role_name = self.get_parameter("ego_role_name").value
        self.global_sampling_resolution_m = float(self.get_parameter("global_sampling_resolution_m").value)
        self.global_route_publish_hz = float(self.get_parameter("global_route_publish_hz").value)
        self.replan_period_s = float(self.get_parameter("replan_period_s").value)
        self.replan_distance_threshold_m = float(self.get_parameter("replan_distance_threshold_m").value)
        self.goal_change_replan = bool(self.get_parameter("goal_change_replan").value)
        self.route_timeout_s = float(self.get_parameter("route_timeout_s").value)
        self.min_route_points = int(self.get_parameter("min_route_points").value)
        self.right_lane_policy_enabled = bool(self.get_parameter("right_lane_policy_enabled").value)
        self.preferred_lane_side = str(self.get_parameter("preferred_lane_side").value or "right").lower()
        self.right_lane_projection_max_distance_m = float(
            self.get_parameter("right_lane_projection_max_distance_m").value
        )
        self.right_lane_policy_disable_in_junction = bool(
            self.get_parameter("right_lane_policy_disable_in_junction").value
        )
        self.route_initial_wrong_way_reject_deg = float(
            self.get_parameter("route_initial_wrong_way_reject_deg").value
        )
        self.sign_constraints_enabled = bool(self.get_parameter("sign_constraints_enabled").value)
        self.sign_plan_geojson = str(self.get_parameter("sign_plan_geojson").value)
        self.sign_plan_json = str(self.get_parameter("sign_plan_json").value)
        self.sign_constraint_effective_radius_m = float(
            self.get_parameter("sign_constraint_effective_radius_m").value
        )
        self.sign_constraint_debug = bool(self.get_parameter("sign_constraint_debug").value)

        self._last_status: Optional[dict[str, Any]] = None
        self._last_status_time = 0.0
        self._last_odom: Optional[Odometry] = None
        self._last_odom_time = 0.0
        self._last_goal: Optional[dict[str, Any]] = None
        self._last_goal_ok = False
        self._last_goal_time = 0.0
        self._last_route_time = 0.0
        self._route_points: List[dict[str, Any]] = []
        self._force_replan = False
        self._force_replan_reason: Optional[str] = None

        self._client = None
        self._world = None
        self._map = None
        self._route_planner = None
        self._planner_api = "failed"
        self._replan_reason = "planner_not_initialized"
        self._carla_root_used: Optional[str] = None
        self._agents_import_path: Optional[str] = None
        self._last_warning_times: dict[str, float] = {}
        self._carla = None
        self._route_validation_debug: dict[str, Any] = {
            "raw_route_len": 0,
            "rejected_wrong_way_count": 0,
            "route_initial_heading_error_deg": None,
            "route_initial_wrong_way": False,
            "candidate_wrong_way": False,
            "route_valid": False,
        }
        self._right_lane_policy_debug: dict[str, Any] = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "preferred_lane_side": self.preferred_lane_side,
            "lane_jump_disabled": True,
            "right_lane_projection_count": 0,
            "right_lane_projection_failed_count": 0,
            "right_lane_projection_partial_fallback_count": 0,
            "right_lane_projection_failed_reason": None,
            "route_continuity_ok": True,
            "right_lane_route_continuity_ok": True,
        }
        self._sign_constraints = SignConstraintSet(
            default_effective_radius_m=self.sign_constraint_effective_radius_m,
        )
        self._sign_constraint_debug: dict[str, Any] = {
            "sign_constraints_enabled": self.sign_constraints_enabled,
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

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)
        self.create_subscription(Odometry, "/adas/localization/odom", self._odom_callback, 10)
        self.create_subscription(String, "/adas/mission/current_goal", self._goal_callback, 10)
        self.create_subscription(String, "/adas/planning/route_debug", self._route_debug_callback, 10)

        self.route_pub = self.create_publisher(String, "/adas/planning/global_route", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/global_route_debug", 10)

        self._connect_to_carla()
        self._load_sign_constraints()
        self.timer = self.create_timer(
            1.0 / max(1.0, self.global_route_publish_hz),
            self._tick,
        )

    def _candidate_carla_roots(self) -> list[str]:
        roots = [
            str(self.carla_root),
            "/home/ilker/simulators/CARLA_0.9.15",
            "/mnt/carla/CARLA_0.9.15",
        ]
        unique_roots = []
        for root in roots:
            normalized = os.path.abspath(os.path.expanduser(root))
            if normalized not in unique_roots:
                unique_roots.append(normalized)
        return unique_roots

    def _prepare_carla_python_paths(self) -> None:
        self._agents_import_path = None
        for root in self._candidate_carla_roots():
            python_api = os.path.join(root, "PythonAPI", "carla")
            if os.path.isdir(os.path.join(python_api, "agents")):
                if python_api not in sys.path:
                    sys.path.insert(0, python_api)
                if self._agents_import_path is None:
                    self._agents_import_path = python_api
                    self._carla_root_used = root

            egg_pattern = os.path.join(python_api, "dist", "*.egg")
            for egg_path in sorted(glob.glob(egg_pattern)):
                if egg_path not in sys.path:
                    sys.path.append(egg_path)

    def _load_carla_with_fallback(self):
        self._prepare_carla_python_paths()
        errors = []
        for root in self._candidate_carla_roots():
            try:
                carla = load_carla(root)
                if self._carla_root_used is None:
                    self._carla_root_used = root
                return carla
            except Exception as exc:
                errors.append(f"{root}: {exc}")
        raise RuntimeError("; ".join(errors))

    def _connect_to_carla(self) -> None:
        try:
            self._carla = self._load_carla_with_fallback()
            self._client = self._carla.Client(self.host, self.port)
            self._client.set_timeout(10.0)
            self._world = self._client.get_world()
            self._map = self._world.get_map()
        except Exception as exc:
            self.get_logger().warn(f"GlobalRoutePlanner: CARLA connection failed: {exc}")
            self._client = None
            self._world = None
            self._map = None

    def _load_sign_constraints(self) -> None:
        if not self.sign_constraints_enabled:
            self._sign_constraints = SignConstraintSet(
                default_effective_radius_m=self.sign_constraint_effective_radius_m,
            )
            self._sign_constraint_debug = {
                **self._sign_constraint_debug,
                "sign_constraints_enabled": False,
                "sign_constraints_loaded": False,
                "sign_constraints_count": 0,
                "active_sign_constraints": [],
            }
            return

        loader = SignConstraintLoader(
            default_effective_radius_m=self.sign_constraint_effective_radius_m,
            carla_map=self._map,
            carla_module=self._carla,
        )
        self._sign_constraints = loader.load(
            geojson_path=self.sign_plan_geojson,
            json_path=self.sign_plan_json,
        )
        self._sign_constraint_debug = {
            **self._sign_constraint_debug,
            "sign_constraints_enabled": True,
            **self._sign_constraints.debug_payload(),
            "sign_constraint_replan_requested": False,
            "sign_constraint_replan_reason": None,
            "forbidden_road_lane_rejected": False,
            "forbidden_turn_rejected": False,
            "speed_limit_annotation": False,
            "stop_yield_annotation": False,
            "park_restriction_annotation": False,
        }
        if not self._sign_constraints.loaded:
            self.get_logger().info(
                "GlobalRoutePlanner: sign constraints loaded=false "
                f"geojson={self.sign_plan_geojson!r} json={self.sign_plan_json!r}"
            )

    def _warn_throttled(self, key: str, message: str, period_s: float = 10.0) -> None:
        now = time.monotonic()
        if now - self._last_warning_times.get(key, 0.0) >= period_s:
            self.get_logger().warn(message)
            self._last_warning_times[key] = now

    def _ensure_route_planner(self) -> bool:
        if self._route_planner is not None:
            return True
        if self._map is None:
            self._planner_api = "failed"
            self._replan_reason = "carla_map_unavailable"
            return False

        self._prepare_carla_python_paths()
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner
        except Exception as exc:
            self._planner_api = "failed"
            self._replan_reason = f"planner_import_failed: {exc}"
            self._warn_throttled(
                "planner_import",
                f"GlobalRoutePlanner: import failed: {exc}",
            )
            return False

        try:
            self._route_planner = GlobalRoutePlanner(
                self._map,
                self.global_sampling_resolution_m,
            )
            self._planner_api = "direct_map"
            self._replan_reason = "planner_ready"
            return True
        except Exception as direct_exc:
            try:
                from agents.navigation.global_route_planner_dao import GlobalRoutePlannerDAO

                dao = GlobalRoutePlannerDAO(
                    self._map,
                    self.global_sampling_resolution_m,
                )
                planner = GlobalRoutePlanner(dao)
                planner.setup()
                self._route_planner = planner
                self._planner_api = "dao"
                self._replan_reason = "planner_ready"
                return True
            except Exception as dao_exc:
                self._route_planner = None
                self._planner_api = "failed"
                self._replan_reason = (
                    f"planner_setup_failed: direct_map={direct_exc}; dao={dao_exc}"
                )
                self._warn_throttled(
                    "planner_setup",
                    "GlobalRoutePlanner: setup failed for direct-map and DAO APIs: "
                    f"direct-map={direct_exc}; DAO={dao_exc}",
                )
                return False

    def _clear_route(self, reason: str) -> None:
        self._route_points = []
        self._last_route_time = 0.0
        self._replan_reason = reason
        self._route_validation_debug = {
            **self._route_validation_debug,
            "raw_route_len": 0,
            "route_valid": False,
        }

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
            self._last_status_time = time.time()
        except Exception:
            self.get_logger().warn("GlobalRoutePlanner: failed to parse /adas/carla/status JSON")

    def _odom_callback(self, msg: Odometry) -> None:
        self._last_odom = msg
        self._last_odom_time = time.time()

    @staticmethod
    def _first_present(payload: dict[str, Any], keys: tuple[str, ...]) -> Any:
        for key in keys:
            value = payload.get(key)
            if value is not None:
                return value
        return None

    def _normalize_goal_payload(self, payload: dict[str, Any]) -> Optional[dict[str, Any]]:
        goal = payload.get("current_goal")
        if not isinstance(goal, dict):
            goal = payload

        name = self._first_present(goal, ("name", "goal_name"))
        kind = self._first_present(goal, ("kind", "goal_kind", "mission_stage"))
        x = self._first_present(goal, ("carla_x", "target_x", "x"))
        y = self._first_present(goal, ("carla_y", "target_y", "y"))
        z = self._first_present(goal, ("carla_z", "target_z", "z"))
        yaw = self._first_present(goal, ("carla_yaw", "target_yaw", "yaw"))
        index = self._first_present(goal, ("nokta_id", "goal_index", "index"))

        if x is None or y is None:
            return None

        normalized = dict(goal)
        for key in (
            "effective_task_stop_x",
            "effective_task_stop_y",
            "effective_task_stop_z",
            "effective_task_stop_yaw",
            "effective_task_stop_source",
            "task_stop_x",
            "task_stop_y",
            "task_stop_z",
            "task_stop_yaw",
            "task_stop_source",
            "task_stop_mode",
            "task_stop_side",
        ):
            if key not in normalized and key in payload:
                normalized[key] = payload.get(key)
        normalized.update(
            {
                "name": name,
                "kind": kind,
                "nokta_id": index,
                "carla_x": float(x),
                "carla_y": float(y),
                "carla_z": float(z or 0.0),
                "carla_yaw": float(yaw) if yaw is not None else None,
                "mission_stage": payload.get("mission_stage", goal.get("mission_stage", kind)),
                "mission_sequence": payload.get("mission_sequence", goal.get("mission_sequence")),
            }
        )
        return normalized

    def _goal_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
            self._last_goal_ok = bool(payload.get("ok", False))
            self._last_goal = self._normalize_goal_payload(payload) if self._last_goal_ok else None
            self._last_goal_time = time.time()
            if not self._last_goal_ok or self._last_goal is None:
                reason = "mission_goal_unavailable"
                if self._last_goal_ok:
                    reason = "mission_goal_parse_failed"
                self._clear_route(reason)
        except Exception:
            self.get_logger().warn("GlobalRoutePlanner: failed to parse /adas/mission/current_goal JSON")
            self._last_goal_ok = False
            self._clear_route("mission_goal_parse_failed")

    def _route_debug_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            return
        if bool(payload.get("replan_recommended", False)):
            self._force_replan = True
            self._force_replan_reason = str(
                payload.get("replan_reason") or "route_sampler_replan_recommended"
            )

    def _ego_pose(self) -> tuple[Optional[dict[str, float]], dict[str, Any]]:
        now = time.time()
        status_age_s = now - self._last_status_time if self._last_status_time else None
        odom_age_s = now - self._last_odom_time if self._last_odom_time else None
        debug = {
            "pose_source": "missing",
            "carla_status_received": self._last_status is not None,
            "carla_status_age_s": round(status_age_s, 3) if status_age_s is not None else None,
            "odom_received": self._last_odom is not None,
            "odom_age_s": round(odom_age_s, 3) if odom_age_s is not None else None,
            "ego_x": None,
            "ego_y": None,
            "ego_yaw": None,
        }

        if self._last_status is not None:
            loc = self._last_status.get("location", {})
            rot = self._last_status.get("rotation", {})
            if "x" in loc and "y" in loc:
                pose = {
                    "x": float(loc["x"]),
                    "y": float(loc["y"]),
                    "z": float(loc.get("z", 0.0)),
                    "yaw": float(rot.get("yaw", 0.0)),
                }
                debug.update(
                    {
                        "pose_source": "carla_status",
                        "ego_x": round(pose["x"], 3),
                        "ego_y": round(pose["y"], 3),
                        "ego_yaw": round(pose["yaw"], 3),
                    }
                )
                return pose, debug

        if self._last_odom is not None:
            pose_msg = self._last_odom.pose.pose
            pose = {
                "x": float(pose_msg.position.x),
                "y": float(pose_msg.position.y),
                "z": float(pose_msg.position.z),
                "yaw": 0.0,
            }
            debug.update(
                {
                    "pose_source": "odom_fallback",
                    "ego_x": round(pose["x"], 3),
                    "ego_y": round(pose["y"], 3),
                    "ego_yaw": 0.0,
                }
            )
            return pose, debug

        return None, debug

    def _is_route_stale(self) -> bool:
        return (time.time() - self._last_route_time) > self.route_timeout_s

    def _route_has_enough_points(self) -> bool:
        return len(self._route_points) >= self.min_route_points

    def _distance_to_route(self, x: float, y: float) -> float:
        if not self._route_points:
            return float("inf")
        _, distance = nearest_point_distance(self._route_points, x, y)
        return distance

    def _get_ego_waypoint(self) -> Optional[Any]:
        if self._map is None or self._carla is None:
            return None

        pose, _ = self._ego_pose()
        if pose is None:
            return None

        try:
            location = self._carla.Location(
                x=float(pose["x"]),
                y=float(pose["y"]),
                z=float(pose.get("z", 0.0)),
            )
            waypoint = self._map.get_waypoint(location, project_to_road=True, lane_type=self._carla.LaneType.Driving)
            return waypoint
        except Exception as exc:
            self.get_logger().warn(f"GlobalRoutePlanner: waypoint lookup failed: {exc}")
            return None

    def _get_goal_waypoint(self) -> Optional[Any]:
        if self._map is None or self._carla is None or self._last_goal is None:
            return None

        goal_x = self._last_goal.get("carla_x")
        goal_y = self._last_goal.get("carla_y")
        goal_z = self._last_goal.get("carla_z")
        if goal_x is None or goal_y is None:
            return None

        try:
            location = self._carla.Location(x=float(goal_x), y=float(goal_y), z=float(goal_z or 0.0))
            return self._map.get_waypoint(location, project_to_road=True, lane_type=self._carla.LaneType.Driving)
        except Exception as exc:
            self.get_logger().warn(f"GlobalRoutePlanner: goal waypoint lookup failed: {exc}")
            return None

    def _is_driving_waypoint(self, waypoint: Any) -> bool:
        if waypoint is None or self._carla is None:
            return False

        try:
            driving = self._carla.LaneType.Driving
            lane_type = waypoint.lane_type
            if lane_type == driving:
                return True
            return bool(lane_type & driving)
        except Exception:
            return False

    def _waypoint_distance(self, a: Any, b: Any) -> float:
        return distance_2d(
            float(a.transform.location.x),
            float(a.transform.location.y),
            float(b.transform.location.x),
            float(b.transform.location.y),
        )

    def _right_lane_projection_for_waypoint(
        self,
        waypoint: Any,
        protect_junction_approach: bool = False,
    ) -> tuple[Any, dict[str, Any]]:
        original_lane_id = getattr(waypoint, "lane_id", None)
        debug = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "lane_preference": "right",
            "original_lane_id": original_lane_id,
            "selected_lane_id": original_lane_id,
            "selected_road_id": getattr(waypoint, "road_id", None),
            "road_id": getattr(waypoint, "road_id", None),
            "is_junction": bool(getattr(waypoint, "is_junction", False)),
            "right_lane_candidate_found": False,
            "right_lane_selected": False,
            "right_lane_reason": "not_evaluated",
            "wrong_way_rejected": False,
            "right_lane_projection_applied": False,
            "right_lane_projection_failed_reason": None,
            "right_lane_projection_distance_m": 0.0,
            "selected_lane_lateral_right_m": 0.0,
            "candidate_lane_ids": [original_lane_id],
            "candidate_lane_lateral_right_m": [0.0],
            "right_lane_calibration_source": None,
            "task_stop_side_lateral_m": None,
            "lane_jump_disabled": True,
            "route_continuity_ok": True,
        }

        if not self.right_lane_policy_enabled:
            debug["right_lane_projection_failed_reason"] = "right_lane_policy_disabled"
            debug["right_lane_reason"] = "right_lane_policy_disabled"
            return waypoint, debug

        if self.preferred_lane_side != "right":
            debug["right_lane_projection_failed_reason"] = "preferred_lane_side_not_right"
            debug["right_lane_reason"] = "preferred_lane_side_not_right"
            return waypoint, debug

        reason = "route_lane_center_locked"
        if protect_junction_approach or (
            self.right_lane_policy_disable_in_junction
            and bool(getattr(waypoint, "is_junction", False))
        ):
            reason = "junction_route_lane_center_locked"
        debug.update(
            {
                "right_lane_selected": True,
                "right_lane_reason": reason,
                "right_lane_projection_applied": False,
                "right_lane_projection_failed_reason": None,
                "right_lane_candidate_found": False,
                "wrong_way_rejected": False,
                "selected_lane_id": original_lane_id,
                "selected_road_id": getattr(waypoint, "road_id", None),
                "selected_lane_lateral_right_m": 0.0,
                "candidate_lane_ids": [original_lane_id],
                "candidate_lane_lateral_right_m": [0.0],
                "right_lane_calibration_source": None,
                "task_stop_side_lateral_m": None,
                "lane_jump_disabled": True,
            }
        )
        return waypoint, debug

    def _waypoint_to_route_point(
        self,
        waypoint: Any,
        route_s_m: float,
        policy_debug: dict[str, Any],
    ) -> dict[str, Any]:
        point = {
            "x": round(float(waypoint.transform.location.x), 3),
            "y": round(float(waypoint.transform.location.y), 3),
            "z": round(float(waypoint.transform.location.z), 3),
            "yaw": round(float(waypoint.transform.rotation.yaw), 3),
            "road_id": waypoint.road_id,
            "lane_id": waypoint.lane_id,
            "selected_road_id": policy_debug.get("selected_road_id", waypoint.road_id),
            "selected_lane_id": policy_debug.get("selected_lane_id", waypoint.lane_id),
            "lane_preference": policy_debug.get("lane_preference", "right"),
            "right_lane_selected": bool(policy_debug.get("right_lane_selected", False)),
            "right_lane_reason": policy_debug.get("right_lane_reason"),
            "is_junction": bool(getattr(waypoint, "is_junction", False)),
            "lane_change": str(getattr(waypoint, "lane_change", "None")),
            "lane_width": round(float(getattr(waypoint, "lane_width", 0.0) or 0.0), 3),
            "lane_width_m": round(float(getattr(waypoint, "lane_width", 0.0) or 0.0), 3),
            "s": round(route_s_m, 3),
            "goal_name": self._last_goal.get("name"),
            "goal_index": self._last_goal.get("nokta_id"),
            "goal_kind": self._last_goal.get("kind"),
        }
        point.update(policy_debug)
        return point

    def _build_route_points(
        self,
        waypoints: list[Any],
        policy_debugs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        points: List[dict[str, Any]] = []
        route_s_m = 0.0
        previous = None
        for waypoint, policy_debug in zip(waypoints, policy_debugs):
            if previous is not None:
                route_s_m += distance_2d(
                    float(previous.transform.location.x),
                    float(previous.transform.location.y),
                    float(waypoint.transform.location.x),
                    float(waypoint.transform.location.y),
                )
            previous = waypoint
            points.append(self._waypoint_to_route_point(waypoint, route_s_m, policy_debug))
        return points

    def _add_turn_direction_annotations(self, points: list[dict[str, Any]]) -> None:
        for point in points:
            point["turn_direction"] = "unknown"

        if len(points) < 3:
            return

        for index in range(1, len(points) - 1):
            current = points[index]
            if not (
                bool(current.get("is_junction", False))
                or bool(points[index - 1].get("is_junction", False))
                or bool(points[index + 1].get("is_junction", False))
            ):
                continue
            current["turn_direction"] = classify_turn_direction(
                float(points[index - 1].get("yaw", current.get("yaw", 0.0))),
                float(points[index + 1].get("yaw", current.get("yaw", 0.0))),
            )

    def _has_hard_sign_constraint(self, point: dict[str, Any]) -> bool:
        annotations = self._sign_constraints.annotate_point(point)
        for constraint in annotations.get("sign_constraints", []):
            if constraint.get("sign_type") in HARD_SEGMENT_CONSTRAINTS:
                return True
        return False

    def _point_to_carla_waypoint(self, point: dict[str, Any]) -> Optional[Any]:
        if self._map is None or self._carla is None:
            return None
        try:
            location = self._carla.Location(
                x=float(point.get("x", 0.0)),
                y=float(point.get("y", 0.0)),
                z=float(point.get("z", 0.0)),
            )
            return self._map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=self._carla.LaneType.Driving,
            )
        except Exception:
            return None

    def _candidate_waypoint_to_point(
        self,
        source_point: dict[str, Any],
        waypoint: Any,
    ) -> dict[str, Any]:
        point = dict(source_point)
        point.update(
            {
                "x": round(float(waypoint.transform.location.x), 3),
                "y": round(float(waypoint.transform.location.y), 3),
                "z": round(float(waypoint.transform.location.z), 3),
                "yaw": round(float(waypoint.transform.rotation.yaw), 3),
                "road_id": waypoint.road_id,
                "lane_id": waypoint.lane_id,
                "selected_lane_id": waypoint.lane_id,
                "sign_constraint_lane_projection_applied": True,
            }
        )
        return point

    def _try_avoid_forbidden_sign_segments(
        self,
        points: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], bool, str | None]:
        if not self.sign_constraints_enabled or not self._sign_constraints.active_constraints():
            return points, False, None

        adjusted_points = [dict(point) for point in points]
        changed = False
        failed_reason = None

        for index, point in enumerate(points):
            if not self._has_hard_sign_constraint(point):
                continue

            base_waypoint = self._point_to_carla_waypoint(point)
            if base_waypoint is None:
                failed_reason = "forbidden_segment_waypoint_unavailable"
                continue

            candidates = []
            for getter_name in ("get_right_lane", "get_left_lane"):
                try:
                    getter = getattr(base_waypoint, getter_name)
                    candidate = getter()
                except Exception:
                    candidate = None
                if candidate is not None:
                    candidates.append(candidate)

            replacement = None
            for candidate in candidates:
                if not self._is_driving_waypoint(candidate):
                    continue
                if getattr(candidate, "road_id", None) != getattr(base_waypoint, "road_id", None):
                    continue
                if angle_diff_deg(
                    float(candidate.transform.rotation.yaw),
                    float(base_waypoint.transform.rotation.yaw),
                ) > 60.0:
                    continue
                candidate_point = self._candidate_waypoint_to_point(point, candidate)
                if self._has_hard_sign_constraint(candidate_point):
                    continue
                replacement = candidate_point
                break

            if replacement is None:
                failed_reason = "forbidden_segment_no_valid_adjacent_lane"
                continue

            adjusted_points[index] = replacement
            changed = True

        if not changed:
            return points, False, failed_reason

        continuity_ok = route_continuity_ok(adjusted_points)
        if not continuity_ok:
            return points, False, "sign_constraint_projection_continuity_failed"

        return adjusted_points, True, None

    def _apply_sign_constraints(self, points: list[dict[str, Any]]) -> list[dict[str, Any]]:
        self._add_turn_direction_annotations(points)
        if not self.sign_constraints_enabled:
            return points

        points, projection_applied, projection_failed_reason = (
            self._try_avoid_forbidden_sign_segments(points)
        )
        if projection_applied:
            self._add_turn_direction_annotations(points)

        decision = self._sign_constraints.evaluate_route(points)
        for index, annotations in decision.annotations_by_index.items():
            if index < 0 or index >= len(points):
                continue
            points[index].update(annotations)

        self._sign_constraint_debug = {
            **self._sign_constraint_debug,
            "sign_constraints_enabled": self.sign_constraints_enabled,
            **self._sign_constraints.debug_payload(),
            **decision.debug_payload(),
        }
        if projection_failed_reason and decision.sign_constraint_replan_requested:
            self._sign_constraint_debug["sign_constraint_replan_reason"] = (
                f"{decision.sign_constraint_replan_reason};{projection_failed_reason}"
            )
        if projection_applied:
            self._sign_constraint_debug["sign_constraint_lane_projection_applied"] = True
        else:
            self._sign_constraint_debug["sign_constraint_lane_projection_applied"] = False

        if decision.sign_constraint_replan_requested:
            self._warn_throttled(
                "sign_constraint_replan",
                "GlobalRoutePlanner: sign constraint requested route alternative, "
                f"reason={self._sign_constraint_debug.get('sign_constraint_replan_reason')}. "
                "Using safe current route fallback.",
            )

        return points

    def _apply_right_lane_policy(self, route: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
        original_waypoints = [waypoint for waypoint, _ in route]
        if not original_waypoints:
            self._right_lane_policy_debug = {
                "right_lane_policy_enabled": self.right_lane_policy_enabled,
                "preferred_lane_side": self.preferred_lane_side,
                "lane_jump_disabled": True,
                "right_lane_projection_count": 0,
                "right_lane_projection_failed_count": 0,
                "right_lane_projection_partial_fallback_count": 0,
                "right_lane_projection_failed_reason": "empty_route",
                "route_continuity_ok": False,
                "right_lane_route_continuity_ok": False,
            }
            return []

        projected_waypoints: list[Any] = []
        projected_debugs: list[dict[str, Any]] = []
        for index, waypoint in enumerate(original_waypoints):
            protect_junction_approach = False
            if self.right_lane_policy_disable_in_junction:
                lookahead = original_waypoints[index:min(len(original_waypoints), index + 4)]
                protect_junction_approach = (
                    not bool(getattr(waypoint, "is_junction", False))
                    and any(bool(getattr(candidate, "is_junction", False)) for candidate in lookahead[1:])
                )
            selected, policy_debug = self._right_lane_projection_for_waypoint(
                waypoint,
                protect_junction_approach=protect_junction_approach,
            )
            projected_waypoints.append(selected)
            projected_debugs.append(policy_debug)

        projected_points = self._build_route_points(projected_waypoints, projected_debugs)
        max_segment_distance_m = max(
            8.0,
            self.global_sampling_resolution_m * 3.0
            + self.right_lane_projection_max_distance_m,
        )
        continuity_ok = route_continuity_ok(
            projected_points,
            max_segment_distance_m=max_segment_distance_m,
        )

        projection_count = sum(
            1 for debug in projected_debugs if debug.get("right_lane_projection_applied")
        )
        candidate_count = sum(
            1 for debug in projected_debugs if debug.get("right_lane_candidate_found")
        )
        wrong_way_rejected_count = sum(
            1 for debug in projected_debugs if debug.get("wrong_way_rejected")
        )
        failed_count = sum(
            1 for debug in projected_debugs if debug.get("right_lane_projection_failed_reason")
        )

        if continuity_ok:
            for point in projected_points:
                point["route_continuity_ok"] = True
            self._right_lane_policy_debug = {
                "right_lane_policy_enabled": self.right_lane_policy_enabled,
                "preferred_lane_side": self.preferred_lane_side,
                "lane_jump_disabled": True,
                "right_lane_projection_count": projection_count,
                "right_lane_projection_failed_count": failed_count,
                "right_lane_projection_partial_fallback_count": 0,
                "right_lane_candidate_count": candidate_count,
                "wrong_way_rejected_count": wrong_way_rejected_count,
                "right_lane_projection_failed_reason": None,
                "route_continuity_ok": True,
                "right_lane_route_continuity_ok": True,
            }
            return self._apply_sign_constraints(projected_points)

        mixed_waypoints: list[Any] = []
        mixed_debugs: list[dict[str, Any]] = []
        partial_fallback_count = 0
        for index, (original, projected, projected_debug) in enumerate(
            zip(original_waypoints, projected_waypoints, projected_debugs)
        ):
            use_projected = bool(projected_debug.get("right_lane_projection_applied"))
            selected = projected if use_projected else original
            debug = dict(projected_debug)

            if use_projected and mixed_waypoints:
                previous_point = self._waypoint_to_route_point(
                    mixed_waypoints[-1],
                    0.0,
                    mixed_debugs[-1],
                )
                candidate_point = self._waypoint_to_route_point(
                    projected,
                    0.0,
                    projected_debug,
                )
                if not route_continuity_ok(
                    [previous_point, candidate_point],
                    max_segment_distance_m=max_segment_distance_m,
                ):
                    selected = original
                    debug = dict(projected_debug)
                    debug["selected_lane_id"] = getattr(original, "lane_id", None)
                    debug["selected_road_id"] = getattr(original, "road_id", None)
                    debug["right_lane_selected"] = False
                    debug["right_lane_reason"] = "partial_route_continuity_failed"
                    debug["right_lane_projection_applied"] = False
                    debug["right_lane_projection_failed_reason"] = "partial_route_continuity_failed"
                    partial_fallback_count += 1

            if use_projected and index + 1 < len(original_waypoints):
                candidate_point = self._waypoint_to_route_point(
                    selected,
                    0.0,
                    debug,
                )
                next_debug = projected_debugs[index + 1]
                next_waypoint = (
                    projected_waypoints[index + 1]
                    if next_debug.get("right_lane_projection_applied")
                    else original_waypoints[index + 1]
                )
                next_point = self._waypoint_to_route_point(next_waypoint, 0.0, next_debug)
                if not route_continuity_ok(
                    [candidate_point, next_point],
                    max_segment_distance_m=max_segment_distance_m,
                ):
                    selected = original
                    debug = dict(projected_debug)
                    debug["selected_lane_id"] = getattr(original, "lane_id", None)
                    debug["selected_road_id"] = getattr(original, "road_id", None)
                    debug["right_lane_selected"] = False
                    debug["right_lane_reason"] = "partial_route_continuity_failed"
                    debug["right_lane_projection_applied"] = False
                    debug["right_lane_projection_failed_reason"] = "partial_route_continuity_failed"
                    partial_fallback_count += 1

            mixed_waypoints.append(selected)
            debug["route_continuity_ok"] = True
            mixed_debugs.append(debug)

        fallback_points = self._build_route_points(mixed_waypoints, mixed_debugs)
        mixed_continuity_ok = route_continuity_ok(
            fallback_points,
            max_segment_distance_m=max_segment_distance_m,
        )
        if not mixed_continuity_ok:
            fallback_debugs: list[dict[str, Any]] = []
            for waypoint, projected_debug in zip(original_waypoints, projected_debugs):
                fallback_debug = dict(projected_debug)
                fallback_debug["selected_lane_id"] = getattr(waypoint, "lane_id", None)
                fallback_debug["selected_road_id"] = getattr(waypoint, "road_id", None)
                fallback_debug["right_lane_selected"] = False
                fallback_debug["right_lane_reason"] = "route_continuity_failed"
                fallback_debug["right_lane_projection_applied"] = False
                fallback_debug["right_lane_projection_failed_reason"] = "route_continuity_failed"
                fallback_debug["route_continuity_ok"] = False
                fallback_debugs.append(fallback_debug)
            fallback_points = self._build_route_points(original_waypoints, fallback_debugs)
            partial_fallback_count = projection_count

        self._right_lane_policy_debug = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "preferred_lane_side": self.preferred_lane_side,
            "lane_jump_disabled": True,
            "right_lane_projection_count": max(0, projection_count - partial_fallback_count),
            "right_lane_projection_failed_count": failed_count,
            "right_lane_projection_partial_fallback_count": partial_fallback_count,
            "right_lane_candidate_count": candidate_count,
            "wrong_way_rejected_count": wrong_way_rejected_count,
            "right_lane_projection_failed_reason": (
                "route_continuity_failed"
                if not mixed_continuity_ok
                else "partial_route_continuity_fallback"
            ),
            "route_continuity_ok": mixed_continuity_ok,
            "right_lane_route_continuity_ok": mixed_continuity_ok,
        }
        self._warn_throttled(
            "right_lane_route_continuity",
            "GlobalRoutePlanner: right lane projection broke route continuity; "
            "using segment-level partial fallback.",
        )
        return self._apply_sign_constraints(fallback_points)

    def _need_replan(self) -> bool:
        if not self._last_goal_ok:
            return False
        pose, _ = self._ego_pose()
        if self._last_goal is None or pose is None:
            return False

        if self._force_replan:
            self._replan_reason = self._force_replan_reason or "route_sampler_replan_recommended"
            return True

        if not self._route_has_enough_points():
            return True

        if self._is_route_stale():
            return True

        if self.goal_change_replan:
            current_goal_name = self._last_goal.get("name")
            previous_goal_name = None
            if self._route_points:
                previous_goal_name = self._route_points[-1].get("goal_name")
            if current_goal_name != previous_goal_name:
                return True

        distance_to_route = self._distance_to_route(float(pose["x"]), float(pose["y"]))
        if distance_to_route > self.replan_distance_threshold_m:
            return True

        return False

    def _plan_route(self) -> None:
        if not self._last_goal_ok or self._last_goal is None:
            self._clear_route("mission_goal_unavailable")
            return

        if not self._ensure_route_planner():
            self._clear_route(self._replan_reason)
            return

        start_wp = self._get_ego_waypoint()
        goal_wp = self._get_goal_waypoint()
        if start_wp is None:
            self._clear_route("start_waypoint_unavailable")
            return
        if goal_wp is None:
            self._clear_route("goal_waypoint_unavailable")
            return

        try:
            start_location = start_wp.transform.location
            goal_location = goal_wp.transform.location
            route = self._route_planner.trace_route(start_location, goal_location)
        except Exception as exc:
            self._clear_route(f"trace_route_failed: {exc}")
            self._warn_throttled(
                "trace_route",
                f"GlobalRoutePlanner: trace_route failed: {exc}",
            )
            return

        if not route:
            self._clear_route("trace_route_returned_empty")
            return

        self._route_validation_debug["raw_route_len"] = len(route)
        if self._route_initial_wrong_way(route):
            raw_route_len = len(route)
            heading_error = self._route_validation_debug.get("route_initial_heading_error_deg")
            self._clear_route("route_initial_wrong_way")
            self._route_validation_debug["raw_route_len"] = raw_route_len
            self._route_validation_debug["route_initial_heading_error_deg"] = heading_error
            return

        points = self._apply_right_lane_policy(route)

        if points:
            self._route_points = points
            self._last_route_time = time.time()
            self._replan_reason = "route_ready"
            self._force_replan = False
            self._force_replan_reason = None
            self._route_validation_debug["route_valid"] = True
        else:
            self._clear_route("trace_route_had_no_waypoints")

    def _route_initial_wrong_way(self, route: list[tuple[Any, Any]]) -> bool:
        self._route_validation_debug["route_initial_heading_error_deg"] = None
        self._route_validation_debug["route_initial_wrong_way"] = False
        self._route_validation_debug["candidate_wrong_way"] = False
        pose, _ = self._ego_pose()
        if not route or len(route) < 2 or pose is None:
            return False

        ego_yaw = float(pose["yaw"])
        first = route[0][0]
        route_yaw = float(first.transform.rotation.yaw)
        fallback_waypoint = None
        for waypoint, _ in route[1:]:
            distance_from_start = distance_2d(
                float(first.transform.location.x),
                float(first.transform.location.y),
                float(waypoint.transform.location.x),
                float(waypoint.transform.location.y),
            )
            if distance_from_start >= 2.0:
                fallback_waypoint = waypoint
            if 10.0 <= distance_from_start <= 20.0:
                fallback_waypoint = waypoint
                break

        if fallback_waypoint is not None:
            route_yaw = math.degrees(
                math.atan2(
                    float(fallback_waypoint.transform.location.y) - float(first.transform.location.y),
                    float(fallback_waypoint.transform.location.x) - float(first.transform.location.x),
                )
            )

        heading_error = angle_diff_deg(route_yaw, ego_yaw)
        self._route_validation_debug["route_initial_heading_error_deg"] = round(heading_error, 3)
        self._route_validation_debug["candidate_wrong_way"] = (
            heading_error > self.route_initial_wrong_way_reject_deg
        )
        if heading_error <= self.route_initial_wrong_way_reject_deg:
            return False

        self._route_validation_debug["rejected_wrong_way_count"] = (
            int(self._route_validation_debug.get("rejected_wrong_way_count", 0)) + 1
        )
        self._route_validation_debug["route_initial_wrong_way"] = True
        self._route_validation_debug["route_valid"] = False
        self._warn_throttled(
            "route_initial_wrong_way",
            "GlobalRoutePlanner: rejecting route_initial_wrong_way "
            f"heading_error={round(heading_error, 3)} deg goal={self._last_goal.get('name')}",
        )
        return True

    def _publish_route(self) -> None:
        now = time.time()
        route_valid = bool(self._route_validation_debug.get("route_valid", False)) and bool(self._route_points)
        route_ok = route_valid
        pose, pose_debug = self._ego_pose()
        route_invalid_reason = None if route_ok else self._replan_reason
        target_x = self._last_goal.get("carla_x") if self._last_goal else None
        target_y = self._last_goal.get("carla_y") if self._last_goal else None
        target_z = self._last_goal.get("carla_z") if self._last_goal else None
        target_yaw = self._last_goal.get("carla_yaw") if self._last_goal else None

        payload = {
            "stamp": now,
            "source": "phase2c_global_route_planner",
            "ok": route_ok,
            "goal_name": self._last_goal.get("name") if self._last_goal else None,
            "goal_index": self._last_goal.get("nokta_id") if self._last_goal else None,
            "goal_kind": self._last_goal.get("kind") if self._last_goal else None,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "target_yaw": target_yaw,
            "route_len": len(self._route_points),
            "route_source": "global_route",
            "points": self._route_points,
            "route_ok": route_ok,
            "status_ok": self._last_status is not None,
            "current_goal_ok": self._last_goal_ok,
            "planner_api": self._planner_api,
            "carla_root_used": self._carla_root_used,
            "agents_import_path": self._agents_import_path,
            "replan_reason": route_invalid_reason,
            "route_invalid_reason": route_invalid_reason,
            "global_route_publish_skipped_reason": None if route_ok else route_invalid_reason,
            **self._route_validation_debug,
            **self._right_lane_policy_debug,
            **self._sign_constraint_debug,
            **pose_debug,
        }

        self.route_pub.publish(String(data=json.dumps(payload)))

        debug_payload = {
            "stamp": now,
            "ok": route_ok,
            "route_ok": route_ok,
            "status_ok": self._last_status is not None,
            "current_goal_ok": self._last_goal_ok,
            "goal_name": self._last_goal.get("name") if self._last_goal else None,
            "goal_index": self._last_goal.get("nokta_id") if self._last_goal else None,
            "goal_kind": self._last_goal.get("kind") if self._last_goal else None,
            "target_x": target_x,
            "target_y": target_y,
            "target_z": target_z,
            "target_yaw": target_yaw,
            "route_len": len(self._route_points),
            "planner_api": self._planner_api,
            "carla_root_used": self._carla_root_used,
            "agents_import_path": self._agents_import_path,
            "replan_reason": route_invalid_reason,
            "route_invalid_reason": route_invalid_reason,
            "global_route_publish_skipped_reason": None if route_ok else route_invalid_reason,
            **self._route_validation_debug,
            **self._right_lane_policy_debug,
            **self._sign_constraint_debug,
            **pose_debug,
            "distance_to_route_m": self._distance_to_route(
                float(pose["x"]),
                float(pose["y"]),
            ) if pose is not None else None,
            "route_age_s": time.time() - self._last_route_time if self._last_route_time else None,
        }
        self.debug_pub.publish(String(data=json.dumps(debug_payload)))

    def _tick(self) -> None:
        if self._client is None or self._map is None:
            self._connect_to_carla()

        if self._need_replan():
            self._plan_route()

        self._publish_route()


def main(args=None):
    rclpy.init(args=args)
    node = GlobalRoutePlannerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
