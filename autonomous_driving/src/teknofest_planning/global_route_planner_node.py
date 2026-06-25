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
        self.declare_parameter("max_initial_trim_distance_m", 12.0)
        self.declare_parameter("route_safety_max_point_gap_m", 8.0)
        self.declare_parameter("route_safety_max_lateral_jump_m", 3.5)
        self.declare_parameter("route_safety_max_heading_jump_deg", 70.0)
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
        self.max_initial_trim_distance_m = float(
            self.get_parameter("max_initial_trim_distance_m").value
        )
        self.route_safety_max_point_gap_m = float(
            self.get_parameter("route_safety_max_point_gap_m").value
        )
        self.route_safety_max_lateral_jump_m = float(
            self.get_parameter("route_safety_max_lateral_jump_m").value
        )
        self.route_safety_max_heading_jump_deg = float(
            self.get_parameter("route_safety_max_heading_jump_deg").value
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
            "route_initial_heading_deg": None,
            "ego_yaw_deg": None,
            "ego_waypoint_yaw_deg": None,
            "route_reference_yaw_deg": None,
            "initial_heading_error_deg": None,
            "route_initial_heading_error_deg": None,
            "route_initial_wrong_way": False,
            "route_initial_wrong_way_detected": False,
            "route_initial_wrong_way_action": None,
            "trimmed_initial_points_count": 0,
            "first_forward_route_index": None,
            "first_forward_route_x": None,
            "first_forward_route_y": None,
            "first_forward_route_distance_m": None,
            "seed_replan_attempted": False,
            "seed_replan_success": False,
            "seed_replan_distance_m": None,
            "seed_replan_reason": None,
            "candidate_wrong_way": False,
            "route_valid": False,
            "route_reject_reason": None,
            "route_safety_validated": False,
            "route_safety_reject_reason": None,
            "max_point_gap_m": self.route_safety_max_point_gap_m,
            "max_lateral_jump_m": self.route_safety_max_lateral_jump_m,
            "max_heading_jump_deg": self.route_safety_max_heading_jump_deg,
            "final_route_source": None,
            "published_route_points_count": 0,
        }
        self._right_lane_policy_debug: dict[str, Any] = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "preferred_lane_side": self.preferred_lane_side,
            "lane_jump_disabled": True,
            "route_lane_id": None,
            "requested_right_lane_id": None,
            "selected_lane_id": None,
            "selected_road_id": None,
            "right_lane_projection_status": "not_evaluated",
            "right_lane_projection_rejected_reason": None,
            "right_lane_fallback_used": False,
            "fallback_kept_right_lane": False,
            "right_lane_reason": None,
            "right_lane_projection_count": 0,
            "right_lane_projection_failed_count": 0,
            "right_lane_projection_partial_fallback_count": 0,
            "right_lane_projection_failed_reason": None,
            "right_lane_projection_attempted": False,
            "right_lane_projection_result": "not_evaluated",
            "right_lane_projection_reject_reason": None,
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
            "route_reject_reason": reason,
            "route_safety_validated": False,
            "route_safety_reject_reason": reason,
            "final_route_source": None,
            "published_route_points_count": 0,
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

    @staticmethod
    def _lane_id_int(waypoint: Any) -> Optional[int]:
        try:
            return int(getattr(waypoint, "lane_id", 0))
        except Exception:
            return None

    def _same_direction_waypoint(
        self,
        waypoint: Any,
        reference: Any,
        max_yaw_delta_deg: float = 60.0,
    ) -> bool:
        try:
            return angle_diff_deg(
                float(waypoint.transform.rotation.yaw),
                float(reference.transform.rotation.yaw),
            ) <= max_yaw_delta_deg
        except Exception:
            return False

    def _lane_lateral_right_m(self, waypoint: Any, reference: Any) -> float:
        yaw_rad = math.radians(float(reference.transform.rotation.yaw))
        dx = float(waypoint.transform.location.x) - float(reference.transform.location.x)
        dy = float(waypoint.transform.location.y) - float(reference.transform.location.y)
        return dx * math.sin(yaw_rad) - dy * math.cos(yaw_rad)

    def _adjacent_driving_lane_candidates(self, waypoint: Any) -> list[Any]:
        candidates: list[Any] = []
        seen: set[tuple[Any, Any]] = set()

        def add_candidate(candidate: Any) -> None:
            if candidate is None:
                return
            key = (getattr(candidate, "road_id", None), getattr(candidate, "lane_id", None))
            if key in seen:
                return
            seen.add(key)
            if getattr(candidate, "road_id", None) != getattr(waypoint, "road_id", None):
                return
            if not self._is_driving_waypoint(candidate):
                return
            if not self._same_direction_waypoint(candidate, waypoint):
                return
            candidates.append(candidate)

        add_candidate(waypoint)
        for getter_name in ("get_right_lane", "get_left_lane"):
            current = waypoint
            for _ in range(8):
                try:
                    getter = getattr(current, getter_name)
                    current = getter()
                except Exception:
                    current = None
                if current is None:
                    break
                add_candidate(current)

        return candidates

    def _task_stop_side_waypoint_for(self, waypoint: Any) -> Optional[Any]:
        if self._last_goal is None or self._map is None or self._carla is None:
            return None

        task_x = self._last_goal.get("task_stop_x")
        task_y = self._last_goal.get("task_stop_y")
        task_z = self._last_goal.get("task_stop_z", self._last_goal.get("carla_z", 0.0))
        if task_x is None or task_y is None:
            task_x = self._last_goal.get("effective_task_stop_x")
            task_y = self._last_goal.get("effective_task_stop_y")
            task_z = self._last_goal.get("effective_task_stop_z", task_z)
        if task_x is None or task_y is None:
            return None

        try:
            task_waypoint = self._map.get_waypoint(
                self._carla.Location(
                    x=float(task_x),
                    y=float(task_y),
                    z=float(task_z or 0.0),
                ),
                project_to_road=True,
                lane_type=self._carla.LaneType.Driving,
            )
        except Exception:
            return None

        if task_waypoint is None:
            return None
        if getattr(task_waypoint, "road_id", None) != getattr(waypoint, "road_id", None):
            return None
        if not self._is_driving_waypoint(task_waypoint):
            return None
        if not self._same_direction_waypoint(task_waypoint, waypoint, max_yaw_delta_deg=75.0):
            return None
        return task_waypoint

    def _select_right_lane_candidate(
        self,
        waypoint: Any,
        candidates: list[Any],
    ) -> tuple[Any, Optional[int], str, Optional[str]]:
        original_lane_id = self._lane_id_int(waypoint)
        requested_lane_id = None
        task_side_waypoint = self._task_stop_side_waypoint_for(waypoint)
        if task_side_waypoint is not None:
            requested_lane_id = self._lane_id_int(task_side_waypoint)
            for candidate in candidates:
                if self._lane_id_int(candidate) == requested_lane_id:
                    return candidate, requested_lane_id, "task_side_right_lane_locked", None
            return waypoint, requested_lane_id, "route_lane_center_locked", "requested_right_lane_unavailable"

        usable = [candidate for candidate in candidates if self._lane_id_int(candidate) is not None]
        if original_lane_id is not None and original_lane_id != 0:
            same_side = [
                candidate
                for candidate in usable
                if self._lane_id_int(candidate) is not None
                and self._lane_id_int(candidate) * original_lane_id > 0
            ]
            if same_side:
                usable = same_side

        if not usable:
            return waypoint, None, "route_lane_center_locked", "right_lane_candidate_unavailable"

        selected = max(
            usable,
            key=lambda candidate: (
                abs(self._lane_id_int(candidate) or 0),
                self._lane_lateral_right_m(candidate, waypoint),
            ),
        )
        return selected, self._lane_id_int(selected), "route_right_lane_locked", None

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
            "route_lane_id": original_lane_id,
            "requested_right_lane_id": None,
            "selected_lane_id": original_lane_id,
            "selected_road_id": getattr(waypoint, "road_id", None),
            "road_id": getattr(waypoint, "road_id", None),
            "is_junction": bool(getattr(waypoint, "is_junction", False)),
            "right_lane_candidate_found": False,
            "right_lane_selected": False,
            "right_lane_reason": "not_evaluated",
            "wrong_way_rejected": False,
            "right_lane_projection_applied": False,
            "right_lane_projection_status": "not_evaluated",
            "right_lane_projection_failed_reason": None,
            "right_lane_projection_rejected_reason": None,
            "right_lane_fallback_used": False,
            "fallback_kept_right_lane": False,
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
            debug["right_lane_projection_rejected_reason"] = "right_lane_policy_disabled"
            debug["right_lane_projection_status"] = "disabled"
            debug["right_lane_reason"] = "right_lane_policy_disabled"
            return waypoint, debug

        if self.preferred_lane_side != "right":
            debug["right_lane_projection_failed_reason"] = "preferred_lane_side_not_right"
            debug["right_lane_projection_rejected_reason"] = "preferred_lane_side_not_right"
            debug["right_lane_projection_status"] = "disabled"
            debug["right_lane_reason"] = "preferred_lane_side_not_right"
            return waypoint, debug

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
                    "right_lane_projection_status": "kept_original",
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

        candidates = self._adjacent_driving_lane_candidates(waypoint)
        selected, requested_lane_id, reason, rejected_reason = self._select_right_lane_candidate(
            waypoint,
            candidates,
        )
        wrong_way_rejected = False
        if selected is not waypoint and not self._same_direction_waypoint(
            selected,
            waypoint,
            max_yaw_delta_deg=90.0,
        ):
            selected = waypoint
            reason = "route_lane_center_locked"
            rejected_reason = "right_lane_projection_wrong_way"
            wrong_way_rejected = True
        candidate_lane_ids = [getattr(candidate, "lane_id", None) for candidate in candidates]
        candidate_lateral = [
            round(self._lane_lateral_right_m(candidate, waypoint), 3) for candidate in candidates
        ]
        projection_distance = self._waypoint_distance(waypoint, selected)
        projection_applied = selected is not waypoint and projection_distance > 0.05
        selected_lane_lateral = self._lane_lateral_right_m(selected, waypoint)
        task_side_waypoint = self._task_stop_side_waypoint_for(waypoint)
        task_stop_side_lateral = (
            round(self._lane_lateral_right_m(task_side_waypoint, waypoint), 3)
            if task_side_waypoint is not None
            else None
        )
        status = "projected" if projection_applied else "already_right_lane"
        if rejected_reason is not None:
            status = "rejected"

        debug.update(
            {
                "right_lane_selected": rejected_reason is None,
                "right_lane_reason": reason,
                "right_lane_projection_applied": projection_applied and rejected_reason is None,
                "right_lane_projection_status": status,
                "right_lane_projection_failed_reason": rejected_reason,
                "right_lane_projection_rejected_reason": rejected_reason,
                "wrong_way_rejected": wrong_way_rejected,
                "route_lane_id": original_lane_id,
                "requested_right_lane_id": requested_lane_id,
                "selected_lane_id": getattr(selected, "lane_id", original_lane_id),
                "selected_road_id": getattr(selected, "road_id", getattr(waypoint, "road_id", None)),
                "right_lane_projection_distance_m": round(projection_distance, 3),
                "selected_lane_lateral_right_m": round(selected_lane_lateral, 3),
                "candidate_lane_ids": candidate_lane_ids,
                "candidate_lane_lateral_right_m": candidate_lateral,
                "right_lane_candidate_found": len(candidates) > 1 or selected is not waypoint,
                "right_lane_calibration_source": (
                    "task_stop_waypoint" if requested_lane_id is not None else "same_road_outer_driving_lane"
                ),
                "task_stop_side_lateral_m": task_stop_side_lateral,
                "lane_jump_disabled": True,
            }
        )
        return selected, debug

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
            "lane_type": str(getattr(waypoint, "lane_type", "")),
            "lane_width": round(float(getattr(waypoint, "lane_width", 0.0) or 0.0), 3),
            "lane_width_m": round(float(getattr(waypoint, "lane_width", 0.0) or 0.0), 3),
            "s": round(route_s_m, 3),
            "goal_name": self._last_goal.get("name"),
            "goal_index": self._last_goal.get("nokta_id"),
            "goal_kind": self._last_goal.get("kind"),
        }
        point.update(policy_debug)
        return point

    def _original_route_policy_debug(
        self,
        waypoint: Any,
        reason: str,
        status: str,
        rejected_reason: Optional[str],
    ) -> dict[str, Any]:
        lane_id = getattr(waypoint, "lane_id", None)
        road_id = getattr(waypoint, "road_id", None)
        return {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "lane_preference": "right",
            "original_lane_id": lane_id,
            "route_lane_id": lane_id,
            "requested_right_lane_id": None,
            "selected_lane_id": lane_id,
            "selected_road_id": road_id,
            "road_id": road_id,
            "is_junction": bool(getattr(waypoint, "is_junction", False)),
            "right_lane_candidate_found": False,
            "right_lane_selected": False,
            "right_lane_reason": reason,
            "wrong_way_rejected": False,
            "right_lane_projection_applied": False,
            "right_lane_projection_status": status,
            "right_lane_projection_failed_reason": rejected_reason,
            "right_lane_projection_rejected_reason": rejected_reason,
            "right_lane_fallback_used": rejected_reason is not None,
            "fallback_kept_right_lane": False,
            "right_lane_projection_distance_m": 0.0,
            "selected_lane_lateral_right_m": 0.0,
            "candidate_lane_ids": [lane_id],
            "candidate_lane_lateral_right_m": [0.0],
            "right_lane_calibration_source": None,
            "task_stop_side_lateral_m": None,
            "lane_jump_disabled": True,
            "route_continuity_ok": True,
        }

    def _densify_waypoints(self, waypoints: list[Any]) -> list[Any]:
        if len(waypoints) < 2:
            return waypoints

        densified: list[Any] = [waypoints[0]]
        step_m = max(1.0, min(self.global_sampling_resolution_m, self.route_safety_max_point_gap_m * 0.5))
        stop_gap_m = max(2.0, self.route_safety_max_point_gap_m * 0.75)

        for target in waypoints[1:]:
            current = densified[-1]
            guard = 0
            while self._waypoint_distance(current, target) > stop_gap_m and guard < 32:
                guard += 1
                try:
                    candidates = list(current.next(step_m))
                except Exception:
                    break
                if not candidates:
                    break

                current_gap = self._waypoint_distance(current, target)
                usable = [
                    candidate
                    for candidate in candidates
                    if candidate is not None
                    and self._is_driving_waypoint(candidate)
                    and self._same_direction_waypoint(candidate, current, max_yaw_delta_deg=90.0)
                    and self._waypoint_distance(candidate, target) < current_gap - 0.1
                ]
                if not usable:
                    break

                best = min(
                    usable,
                    key=lambda candidate: (
                        self._waypoint_distance(candidate, target),
                        angle_diff_deg(
                            float(candidate.transform.rotation.yaw),
                            float(target.transform.rotation.yaw),
                        ),
                    ),
                )
                if self._waypoint_distance(best, current) < 0.1:
                    break
                densified.append(best)
                current = best

            if self._waypoint_distance(densified[-1], target) > 0.1:
                densified.append(target)

        return densified

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

    def _should_keep_projected_right_lane_on_fallback(
        self,
        index: int,
        original_waypoints: list[Any],
        projected_waypoints: list[Any],
        projected_debugs: list[dict[str, Any]],
    ) -> bool:
        if index < 0 or index >= len(projected_waypoints):
            return False
        debug = projected_debugs[index]
        if not bool(debug.get("right_lane_projection_applied", False)):
            return False

        original = original_waypoints[index]
        projected = projected_waypoints[index]
        if bool(getattr(original, "is_junction", False)) or bool(getattr(projected, "is_junction", False)):
            return False

        road_id = getattr(projected, "road_id", None)
        for neighbor_index in (index - 1, index + 1):
            if neighbor_index < 0 or neighbor_index >= len(projected_waypoints):
                continue
            neighbor_original = original_waypoints[neighbor_index]
            neighbor_projected = projected_waypoints[neighbor_index]
            if (
                bool(getattr(neighbor_original, "is_junction", False))
                or bool(getattr(neighbor_projected, "is_junction", False))
            ):
                return False
            if getattr(neighbor_projected, "road_id", None) != road_id:
                return False

        return True

    def _apply_right_lane_policy(self, route: list[tuple[Any, Any]]) -> list[dict[str, Any]]:
        original_waypoints = self._densify_waypoints([waypoint for waypoint, _ in route])
        if not original_waypoints:
            self._right_lane_policy_debug = {
                "right_lane_policy_enabled": self.right_lane_policy_enabled,
                "preferred_lane_side": self.preferred_lane_side,
                "lane_jump_disabled": True,
                "route_lane_id": None,
                "requested_right_lane_id": None,
                "selected_lane_id": None,
                "selected_road_id": None,
                "right_lane_projection_status": "empty_route",
                "right_lane_projection_rejected_reason": "empty_route",
                "right_lane_fallback_used": False,
                "fallback_kept_right_lane": False,
                "right_lane_reason": "empty_route",
                "right_lane_projection_count": 0,
                "right_lane_projection_failed_count": 0,
                "right_lane_projection_partial_fallback_count": 0,
                "right_lane_projection_failed_reason": "empty_route",
                "right_lane_projection_attempted": False,
                "right_lane_projection_result": "empty_route",
                "right_lane_projection_reject_reason": "empty_route",
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
                "route_lane_id": projected_points[0].get("route_lane_id"),
                "requested_right_lane_id": projected_points[0].get("requested_right_lane_id"),
                "selected_lane_id": projected_points[0].get("selected_lane_id"),
                "selected_road_id": projected_points[0].get("selected_road_id"),
                "right_lane_projection_status": "ok",
                "right_lane_projection_rejected_reason": None,
                "right_lane_fallback_used": False,
                "fallback_kept_right_lane": False,
                "right_lane_reason": projected_points[0].get("right_lane_reason"),
                "right_lane_projection_count": projection_count,
                "right_lane_projection_failed_count": failed_count,
                "right_lane_projection_partial_fallback_count": 0,
                "right_lane_candidate_count": candidate_count,
                "wrong_way_rejected_count": wrong_way_rejected_count,
                "right_lane_projection_failed_reason": None,
                "right_lane_projection_attempted": projection_count > 0,
                "right_lane_projection_result": "ok",
                "right_lane_projection_reject_reason": None,
                "route_continuity_ok": True,
                "right_lane_route_continuity_ok": True,
            }
            return self._apply_sign_constraints(projected_points)

        original_debugs = [
            self._original_route_policy_debug(
                waypoint,
                reason="route_lane_center_locked",
                status="fallback_to_original_continuity_failed",
                rejected_reason="projected_route_continuity_failed",
            )
            for waypoint in original_waypoints
        ]
        original_points = self._build_route_points(original_waypoints, original_debugs)
        original_continuity_ok = route_continuity_ok(
            original_points,
            max_segment_distance_m=max_segment_distance_m,
        )
        for point in original_points:
            point["route_continuity_ok"] = original_continuity_ok
            point["right_lane_projection_status"] = "fallback_to_original_continuity_failed"
            point["right_lane_projection_rejected_reason"] = "projected_route_continuity_failed"
            point["right_lane_projection_failed_reason"] = "projected_route_continuity_failed"

        self._right_lane_policy_debug = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "preferred_lane_side": self.preferred_lane_side,
            "lane_jump_disabled": True,
            "route_lane_id": original_points[0].get("route_lane_id") if original_points else None,
            "requested_right_lane_id": original_points[0].get("requested_right_lane_id") if original_points else None,
            "selected_lane_id": original_points[0].get("selected_lane_id") if original_points else None,
            "selected_road_id": original_points[0].get("selected_road_id") if original_points else None,
            "right_lane_projection_status": "fallback_to_original_continuity_failed",
            "right_lane_projection_rejected_reason": "projected_route_continuity_failed",
            "right_lane_fallback_used": True,
            "fallback_kept_right_lane": False,
            "right_lane_reason": original_points[0].get("right_lane_reason") if original_points else None,
            "right_lane_projection_count": 0,
            "right_lane_projection_failed_count": failed_count,
            "right_lane_projection_partial_fallback_count": 0,
            "right_lane_candidate_count": candidate_count,
            "wrong_way_rejected_count": wrong_way_rejected_count,
            "right_lane_projection_failed_reason": "projected_route_continuity_failed",
            "right_lane_projection_attempted": projection_count > 0,
            "right_lane_projection_result": (
                "fallback_to_original" if original_continuity_ok else "rejected"
            ),
            "right_lane_projection_reject_reason": "projected_route_continuity_failed",
            "route_continuity_ok": original_continuity_ok,
            "right_lane_route_continuity_ok": False,
        }
        self._warn_throttled(
            "right_lane_route_continuity",
            "GlobalRoutePlanner: right lane projection broke route continuity; "
            "using original CARLA route fallback.",
        )
        return self._apply_sign_constraints(original_points)

    def _build_original_route_fallback_points(
        self,
        route: list[tuple[Any, Any]],
        reject_reason: str,
    ) -> list[dict[str, Any]]:
        original_waypoints = self._densify_waypoints([waypoint for waypoint, _ in route])
        original_debugs = [
            self._original_route_policy_debug(
                waypoint,
                reason="route_lane_center_locked",
                status="fallback_to_original_safety_failed",
                rejected_reason=reject_reason,
            )
            for waypoint in original_waypoints
        ]
        original_points = self._build_route_points(original_waypoints, original_debugs)
        max_segment_distance_m = max(
            8.0,
            self.global_sampling_resolution_m * 3.0
            + self.right_lane_projection_max_distance_m,
        )
        original_continuity_ok = route_continuity_ok(
            original_points,
            max_segment_distance_m=max_segment_distance_m,
        )
        for point in original_points:
            point["route_continuity_ok"] = original_continuity_ok
            point["right_lane_projection_status"] = "fallback_to_original_safety_failed"
            point["right_lane_projection_rejected_reason"] = reject_reason
            point["right_lane_projection_failed_reason"] = reject_reason

        self._right_lane_policy_debug = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "preferred_lane_side": self.preferred_lane_side,
            "lane_jump_disabled": True,
            "route_lane_id": original_points[0].get("route_lane_id") if original_points else None,
            "requested_right_lane_id": original_points[0].get("requested_right_lane_id") if original_points else None,
            "selected_lane_id": original_points[0].get("selected_lane_id") if original_points else None,
            "selected_road_id": original_points[0].get("selected_road_id") if original_points else None,
            "right_lane_projection_status": "fallback_to_original_safety_failed",
            "right_lane_projection_rejected_reason": reject_reason,
            "right_lane_fallback_used": True,
            "fallback_kept_right_lane": False,
            "right_lane_reason": original_points[0].get("right_lane_reason") if original_points else None,
            "right_lane_projection_count": 0,
            "right_lane_projection_failed_count": 1,
            "right_lane_projection_partial_fallback_count": 0,
            "right_lane_candidate_count": 0,
            "wrong_way_rejected_count": 0,
            "right_lane_projection_failed_reason": reject_reason,
            "right_lane_projection_attempted": True,
            "right_lane_projection_result": (
                "fallback_to_original" if original_continuity_ok else "rejected"
            ),
            "right_lane_projection_reject_reason": reject_reason,
            "route_continuity_ok": original_continuity_ok,
            "right_lane_route_continuity_ok": False,
        }
        return self._apply_sign_constraints(original_points)

        mixed_waypoints: list[Any] = []
        mixed_debugs: list[dict[str, Any]] = []
        partial_fallback_count = 0
        fallback_kept_right_lane_count = 0
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
                    if self._should_keep_projected_right_lane_on_fallback(
                        index,
                        original_waypoints,
                        projected_waypoints,
                        projected_debugs,
                    ):
                        debug = dict(projected_debug)
                        debug["right_lane_fallback_used"] = True
                        debug["fallback_kept_right_lane"] = True
                        debug["right_lane_projection_status"] = "continuity_fallback_kept_right_lane"
                        debug["right_lane_projection_rejected_reason"] = (
                            "partial_route_continuity_failed_same_road_kept_right_lane"
                        )
                        debug["right_lane_reason"] = "route_right_lane_locked"
                        fallback_kept_right_lane_count += 1
                    else:
                        selected = original
                        debug = dict(projected_debug)
                        debug["selected_lane_id"] = getattr(original, "lane_id", None)
                        debug["selected_road_id"] = getattr(original, "road_id", None)
                        debug["right_lane_selected"] = False
                        debug["right_lane_reason"] = "partial_route_continuity_failed"
                        debug["right_lane_projection_applied"] = False
                        debug["right_lane_projection_status"] = "fallback_to_original"
                        debug["right_lane_projection_failed_reason"] = "partial_route_continuity_failed"
                        debug["right_lane_projection_rejected_reason"] = "partial_route_continuity_failed"
                        debug["right_lane_fallback_used"] = True
                        debug["fallback_kept_right_lane"] = False
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
                    if self._should_keep_projected_right_lane_on_fallback(
                        index,
                        original_waypoints,
                        projected_waypoints,
                        projected_debugs,
                    ):
                        selected = projected
                        debug = dict(debug)
                        debug["right_lane_fallback_used"] = True
                        debug["fallback_kept_right_lane"] = True
                        debug["right_lane_projection_status"] = "continuity_fallback_kept_right_lane"
                        debug["right_lane_projection_rejected_reason"] = (
                            "partial_route_continuity_failed_same_road_kept_right_lane"
                        )
                        if debug.get("right_lane_reason") == "partial_route_continuity_failed":
                            debug["right_lane_reason"] = "route_right_lane_locked"
                        fallback_kept_right_lane_count += 1
                    else:
                        selected = original
                        debug = dict(projected_debug)
                        debug["selected_lane_id"] = getattr(original, "lane_id", None)
                        debug["selected_road_id"] = getattr(original, "road_id", None)
                        debug["right_lane_selected"] = False
                        debug["right_lane_reason"] = "partial_route_continuity_failed"
                        debug["right_lane_projection_applied"] = False
                        debug["right_lane_projection_status"] = "fallback_to_original"
                        debug["right_lane_projection_failed_reason"] = "partial_route_continuity_failed"
                        debug["right_lane_projection_rejected_reason"] = "partial_route_continuity_failed"
                        debug["right_lane_fallback_used"] = True
                        debug["fallback_kept_right_lane"] = False
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
            final_waypoints: list[Any] = []
            fallback_debugs: list[dict[str, Any]] = []
            forced_original_count = 0
            for index, (waypoint, projected_waypoint, projected_debug) in enumerate(
                zip(original_waypoints, projected_waypoints, projected_debugs)
            ):
                fallback_debug = dict(projected_debug)
                if self._should_keep_projected_right_lane_on_fallback(
                    index,
                    original_waypoints,
                    projected_waypoints,
                    projected_debugs,
                ):
                    final_waypoints.append(projected_waypoint)
                    fallback_debug["right_lane_fallback_used"] = True
                    fallback_debug["fallback_kept_right_lane"] = True
                    fallback_debug["right_lane_projection_status"] = "continuity_fallback_kept_right_lane"
                    fallback_debug["right_lane_projection_rejected_reason"] = (
                        "route_continuity_failed_same_road_kept_right_lane"
                    )
                    fallback_debug["right_lane_reason"] = "route_right_lane_locked"
                    fallback_kept_right_lane_count += 1
                else:
                    final_waypoints.append(waypoint)
                    fallback_debug["selected_lane_id"] = getattr(waypoint, "lane_id", None)
                    fallback_debug["selected_road_id"] = getattr(waypoint, "road_id", None)
                    fallback_debug["right_lane_selected"] = False
                    fallback_debug["right_lane_reason"] = "route_continuity_failed"
                    fallback_debug["right_lane_projection_applied"] = False
                    fallback_debug["right_lane_projection_status"] = "fallback_to_original"
                    fallback_debug["right_lane_projection_failed_reason"] = "route_continuity_failed"
                    fallback_debug["right_lane_projection_rejected_reason"] = "route_continuity_failed"
                    fallback_debug["right_lane_fallback_used"] = True
                    fallback_debug["fallback_kept_right_lane"] = False
                    if projected_debug.get("right_lane_projection_applied"):
                        forced_original_count += 1
                fallback_debug["route_continuity_ok"] = False
                fallback_debugs.append(fallback_debug)
            fallback_points = self._build_route_points(final_waypoints, fallback_debugs)
            partial_fallback_count += forced_original_count

        self._right_lane_policy_debug = {
            "right_lane_policy_enabled": self.right_lane_policy_enabled,
            "preferred_lane_side": self.preferred_lane_side,
            "lane_jump_disabled": True,
            "route_lane_id": fallback_points[0].get("route_lane_id") if fallback_points else None,
            "requested_right_lane_id": fallback_points[0].get("requested_right_lane_id") if fallback_points else None,
            "selected_lane_id": fallback_points[0].get("selected_lane_id") if fallback_points else None,
            "selected_road_id": fallback_points[0].get("selected_road_id") if fallback_points else None,
            "right_lane_projection_status": (
                "fallback_kept_right_lane"
                if fallback_kept_right_lane_count > 0
                else "partial_fallback_to_original"
            ),
            "right_lane_projection_rejected_reason": (
                "route_continuity_failed_same_road_kept_right_lane"
                if fallback_kept_right_lane_count > 0
                else "route_continuity_failed"
            ),
            "right_lane_fallback_used": partial_fallback_count > 0 or fallback_kept_right_lane_count > 0,
            "fallback_kept_right_lane": fallback_kept_right_lane_count > 0,
            "right_lane_reason": fallback_points[0].get("right_lane_reason") if fallback_points else None,
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

    def _seeded_forward_route_from_ego_lane(
        self,
        start_wp: Any,
        goal_wp: Any,
    ) -> tuple[list[tuple[Any, Any]], dict[str, Any]]:
        debug: dict[str, Any] = {
            "seed_replan_attempted": True,
            "seed_replan_success": False,
            "seed_replan_distance_m": None,
            "seed_replan_reason": "no_seed_candidate",
        }
        if self._route_planner is None or start_wp is None or goal_wp is None:
            debug["seed_replan_reason"] = "seed_replan_unavailable"
            return [], debug

        goal_location = goal_wp.transform.location
        seed_distances_m = (2.0, 4.0, 6.0, 8.0, 10.0)
        for seed_distance_m in seed_distances_m:
            try:
                candidates = list(start_wp.next(seed_distance_m))
            except Exception as exc:
                debug["seed_replan_reason"] = f"seed_next_failed:{exc}"
                continue

            same_lane_candidates = [
                candidate
                for candidate in candidates
                if candidate is not None
                and getattr(candidate, "road_id", None) == getattr(start_wp, "road_id", None)
                and getattr(candidate, "lane_id", None) == getattr(start_wp, "lane_id", None)
                and self._is_driving_waypoint(candidate)
                and self._same_direction_waypoint(candidate, start_wp, max_yaw_delta_deg=60.0)
            ]
            compatible_candidates = same_lane_candidates or [
                candidate
                for candidate in candidates
                if candidate is not None
                and self._is_driving_waypoint(candidate)
                and self._same_direction_waypoint(candidate, start_wp, max_yaw_delta_deg=75.0)
            ]
            compatible_candidates.sort(
                key=lambda candidate: (
                    getattr(candidate, "road_id", None) != getattr(start_wp, "road_id", None),
                    getattr(candidate, "lane_id", None) != getattr(start_wp, "lane_id", None),
                    angle_diff_deg(
                        float(candidate.transform.rotation.yaw),
                        float(start_wp.transform.rotation.yaw),
                    ),
                )
            )

            for seed_wp in compatible_candidates:
                try:
                    route = self._route_planner.trace_route(
                        seed_wp.transform.location,
                        goal_location,
                    )
                except Exception as exc:
                    debug["seed_replan_reason"] = f"seed_trace_failed:{exc}"
                    continue

                if route:
                    debug.update(
                        {
                            "seed_replan_success": True,
                            "seed_replan_distance_m": seed_distance_m,
                            "seed_replan_reason": "ego_lane_forward_seed",
                        }
                    )
                    return route, debug

        return [], debug

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
        route = self._repair_initial_wrong_way_route(route)
        if not route:
            original_validation_debug = dict(self._route_validation_debug)
            seeded_route, seed_debug = self._seeded_forward_route_from_ego_lane(
                start_wp,
                goal_wp,
            )
            if seeded_route:
                self._route_validation_debug["raw_route_len"] = len(seeded_route)
                route = self._repair_initial_wrong_way_route(seeded_route)
                self._route_validation_debug.update(seed_debug)
            if not route:
                raw_route_len = int(original_validation_debug.get("raw_route_len", 0))
                validation_debug = {
                    **original_validation_debug,
                    **seed_debug,
                }
                self._clear_route("route_initial_wrong_way")
                self._route_validation_debug.update(validation_debug)
                self._route_validation_debug["raw_route_len"] = raw_route_len
                self._route_validation_debug["route_valid"] = False
                self._route_validation_debug["route_reject_reason"] = "route_initial_wrong_way"
                return

        points = self._apply_right_lane_policy(route)
        route_ready = bool(points) and self._validate_final_route_points(points)
        if (
            not route_ready
            and points
            and bool(self._right_lane_policy_debug.get("right_lane_projection_attempted", False))
            and self._right_lane_policy_debug.get("right_lane_projection_status") == "ok"
        ):
            projected_reject_reason = str(
                self._route_validation_debug.get("route_safety_reject_reason")
                or "projected_route_safety_failed"
            )
            points = self._build_original_route_fallback_points(
                route,
                projected_reject_reason,
            )
            route_ready = bool(points) and self._validate_final_route_points(points)

        if route_ready:
            self._route_points = points
            self._last_route_time = time.time()
            self._replan_reason = "route_ready"
            self._force_replan = False
            self._force_replan_reason = None
            self._route_validation_debug["route_valid"] = True
            self._route_validation_debug["route_reject_reason"] = None
        elif points:
            validation_debug = dict(self._route_validation_debug)
            reject_reason = str(
                validation_debug.get("route_safety_reject_reason")
                or "route_safety_failed"
            )
            self._clear_route(reject_reason)
            self._route_validation_debug.update(validation_debug)
            self._route_validation_debug["route_valid"] = False
            self._route_validation_debug["route_reject_reason"] = reject_reason
        else:
            self._clear_route("trace_route_had_no_waypoints")

    def _reset_initial_route_validation_debug(self) -> None:
        self._route_validation_debug.update(
            {
                "route_initial_heading_deg": None,
                "ego_yaw_deg": None,
                "ego_waypoint_yaw_deg": None,
                "route_reference_yaw_deg": None,
                "initial_heading_error_deg": None,
                "route_initial_heading_error_deg": None,
                "route_initial_wrong_way": False,
                "route_initial_wrong_way_detected": False,
                "route_initial_wrong_way_action": "not_checked",
                "trimmed_initial_points_count": 0,
                "first_forward_route_index": None,
                "first_forward_route_x": None,
                "first_forward_route_y": None,
                "first_forward_route_distance_m": None,
                "seed_replan_attempted": False,
                "seed_replan_success": False,
                "seed_replan_distance_m": None,
                "seed_replan_reason": None,
                "candidate_wrong_way": False,
            }
        )

    def _route_heading_from_index(
        self,
        route: list[tuple[Any, Any]],
        start_index: int,
    ) -> Optional[float]:
        if start_index < 0 or start_index >= len(route):
            return None

        first = route[start_index][0]
        route_yaw = float(first.transform.rotation.yaw)
        fallback_waypoint = None
        for waypoint, _ in route[start_index + 1:]:
            distance_from_start = distance_2d(
                float(first.transform.location.x),
                float(first.transform.location.y),
                float(waypoint.transform.location.x),
                float(waypoint.transform.location.y),
            )
            if distance_from_start >= 2.0 and fallback_waypoint is None:
                fallback_waypoint = waypoint
            if 10.0 <= distance_from_start <= 20.0:
                fallback_waypoint = waypoint
                break
            if distance_from_start > 20.0 and fallback_waypoint is not None:
                break

        if fallback_waypoint is not None:
            route_yaw = math.degrees(
                math.atan2(
                    float(fallback_waypoint.transform.location.y)
                    - float(first.transform.location.y),
                    float(fallback_waypoint.transform.location.x)
                    - float(first.transform.location.x),
                )
            )

        return route_yaw

    def _ego_waypoint_yaw(self) -> Optional[float]:
        waypoint = self._get_ego_waypoint()
        if waypoint is None:
            return None
        try:
            return float(waypoint.transform.rotation.yaw)
        except Exception:
            return None

    def _first_forward_route_index(
        self,
        route: list[tuple[Any, Any]],
        pose: dict[str, float],
        reference_yaw_deg: Optional[float] = None,
    ) -> tuple[Optional[int], Optional[float], Optional[float]]:
        ego_yaw = float(pose["yaw"] if reference_yaw_deg is None else reference_yaw_deg)
        ego_yaw_rad = math.radians(ego_yaw)
        forward_x = math.cos(ego_yaw_rad)
        forward_y = math.sin(ego_yaw_rad)
        heading_limit_deg = min(90.0, self.route_initial_wrong_way_reject_deg)
        max_scan = min(len(route), 40)

        for index in range(max_scan):
            waypoint = route[index][0]
            dx = float(waypoint.transform.location.x) - float(pose["x"])
            dy = float(waypoint.transform.location.y) - float(pose["y"])
            along_track_m = dx * forward_x + dy * forward_y
            if along_track_m <= 0.5:
                continue
            if along_track_m > self.max_initial_trim_distance_m:
                break
            lateral_m = abs(-dx * forward_y + dy * forward_x)
            if lateral_m > self.route_safety_max_lateral_jump_m:
                continue

            route_yaw = self._route_heading_from_index(route, index)
            if route_yaw is None:
                continue

            heading_error = angle_diff_deg(route_yaw, ego_yaw)
            if heading_error <= heading_limit_deg:
                return index, route_yaw, heading_error

        return None, None, None

    def _repair_initial_wrong_way_route(
        self,
        route: list[tuple[Any, Any]],
    ) -> list[tuple[Any, Any]]:
        self._reset_initial_route_validation_debug()
        pose, _ = self._ego_pose()
        if not route or len(route) < 2 or pose is None:
            self._route_validation_debug["route_initial_wrong_way_action"] = "skipped"
            return route

        ego_yaw = float(pose["yaw"])
        ego_waypoint_yaw = self._ego_waypoint_yaw()
        route_yaw = self._route_heading_from_index(route, 0)
        if route_yaw is None:
            self._route_validation_debug["route_initial_wrong_way_action"] = "skipped"
            return route

        heading_error = angle_diff_deg(route_yaw, ego_yaw)
        waypoint_heading_error = (
            angle_diff_deg(route_yaw, ego_waypoint_yaw)
            if ego_waypoint_yaw is not None
            else None
        )
        reference_yaw = (
            ego_waypoint_yaw
            if waypoint_heading_error is not None and waypoint_heading_error <= 75.0
            else ego_yaw
        )
        reference_heading_error = angle_diff_deg(route_yaw, reference_yaw)
        self._route_validation_debug["route_initial_heading_deg"] = round(route_yaw, 3)
        self._route_validation_debug["ego_yaw_deg"] = round(ego_yaw, 3)
        self._route_validation_debug["ego_waypoint_yaw_deg"] = (
            round(ego_waypoint_yaw, 3) if ego_waypoint_yaw is not None else None
        )
        self._route_validation_debug["route_reference_yaw_deg"] = round(reference_yaw, 3)
        self._route_validation_debug["initial_heading_error_deg"] = round(reference_heading_error, 3)
        self._route_validation_debug["route_initial_heading_error_deg"] = round(reference_heading_error, 3)
        self._route_validation_debug["candidate_wrong_way"] = (
            reference_heading_error > self.route_initial_wrong_way_reject_deg
        )
        if reference_heading_error <= self.route_initial_wrong_way_reject_deg:
            self._route_validation_debug["route_initial_wrong_way_action"] = (
                "accepted_ego_waypoint_heading"
                if heading_error > self.route_initial_wrong_way_reject_deg
                and reference_yaw == ego_waypoint_yaw
                else "accepted"
            )
            return route

        self._route_validation_debug["route_initial_wrong_way"] = True
        self._route_validation_debug["route_initial_wrong_way_detected"] = True

        forward_index, forward_yaw, forward_error = self._first_forward_route_index(
            route,
            pose,
            reference_yaw_deg=reference_yaw,
        )
        if forward_index is not None and forward_index > 0 and len(route) - forward_index >= 2:
            forward_waypoint = route[forward_index][0]
            self._route_validation_debug["route_initial_wrong_way_action"] = (
                "initial_wrong_way_trimmed_to_forward_segment"
            )
            self._route_validation_debug["trimmed_initial_points_count"] = forward_index
            self._route_validation_debug["first_forward_route_index"] = forward_index
            self._route_validation_debug["first_forward_route_x"] = round(
                float(forward_waypoint.transform.location.x),
                3,
            )
            self._route_validation_debug["first_forward_route_y"] = round(
                float(forward_waypoint.transform.location.y),
                3,
            )
            self._route_validation_debug["first_forward_route_distance_m"] = round(
                distance_2d(
                    float(pose["x"]),
                    float(pose["y"]),
                    float(forward_waypoint.transform.location.x),
                    float(forward_waypoint.transform.location.y),
                ),
                3,
            )
            if forward_yaw is not None:
                self._route_validation_debug["route_initial_heading_deg"] = round(
                    forward_yaw,
                    3,
                )
            if forward_error is not None:
                self._route_validation_debug["initial_heading_error_deg"] = round(
                    forward_error,
                    3,
                )
                self._route_validation_debug["route_initial_heading_error_deg"] = round(
                    forward_error,
                    3,
                )
            self._warn_throttled(
                "route_initial_wrong_way_trimmed",
                "GlobalRoutePlanner: initial wrong-way segment trimmed "
                f"heading_error={round(reference_heading_error, 3)} deg "
                f"trimmed_points={forward_index} goal={self._last_goal.get('name')}",
            )
            return route[forward_index:]

        self._route_validation_debug["rejected_wrong_way_count"] = (
            int(self._route_validation_debug.get("rejected_wrong_way_count", 0)) + 1
        )
        self._route_validation_debug["route_valid"] = False
        self._route_validation_debug["route_initial_wrong_way_action"] = (
            "rejected_no_forward_segment"
        )
        self._route_validation_debug["route_reject_reason"] = "route_initial_wrong_way"
        self._warn_throttled(
            "route_initial_wrong_way",
            "GlobalRoutePlanner: rejecting route_initial_wrong_way "
            f"heading_error={round(reference_heading_error, 3)} deg goal={self._last_goal.get('name')}",
        )
        return []

    def _reset_route_safety_debug(self) -> None:
        self._route_validation_debug.update(
            {
                "route_safety_validated": False,
                "route_safety_reject_reason": None,
                "max_point_gap_m": self.route_safety_max_point_gap_m,
                "max_lateral_jump_m": self.route_safety_max_lateral_jump_m,
                "max_heading_jump_deg": self.route_safety_max_heading_jump_deg,
                "final_route_source": None,
                "published_route_points_count": 0,
            }
        )

    def _validate_final_route_points(self, points: list[dict[str, Any]]) -> bool:
        self._reset_route_safety_debug()
        if not points:
            self._route_validation_debug["route_safety_reject_reason"] = "empty_route"
            return False

        pose, _ = self._ego_pose()
        if pose is not None:
            first = points[0]
            reference_yaw = self._ego_waypoint_yaw()
            if reference_yaw is None:
                reference_yaw = float(pose["yaw"])
            self._route_validation_debug["route_reference_yaw_deg"] = round(reference_yaw, 3)
            ego_yaw_rad = math.radians(float(reference_yaw))
            dx = float(first.get("x", 0.0)) - float(pose["x"])
            dy = float(first.get("y", 0.0)) - float(pose["y"])
            along_track_m = dx * math.cos(ego_yaw_rad) + dy * math.sin(ego_yaw_rad)
            distance_m = math.hypot(dx, dy)
            if along_track_m < -2.0:
                self._route_validation_debug["route_safety_reject_reason"] = (
                    "first_point_behind_ego"
                )
                return False
            if distance_m > self.max_initial_trim_distance_m:
                self._route_validation_debug["route_safety_reject_reason"] = (
                    "first_point_too_far_from_ego"
                )
                return False

        for index, point in enumerate(points):
            lane_type = str(point.get("lane_type", ""))
            if lane_type and "Driving" not in lane_type:
                self._route_validation_debug["route_safety_reject_reason"] = (
                    f"non_driving_lane_at_{index}"
                )
                return False

            if index == 0:
                continue

            previous = points[index - 1]
            dx = float(point.get("x", 0.0)) - float(previous.get("x", 0.0))
            dy = float(point.get("y", 0.0)) - float(previous.get("y", 0.0))
            gap_m = math.hypot(dx, dy)
            if gap_m > self.route_safety_max_point_gap_m:
                self._route_validation_debug["route_safety_reject_reason"] = (
                    f"point_gap_too_large_at_{index}"
                )
                return False

            in_junction = bool(point.get("is_junction", False)) or bool(
                previous.get("is_junction", False)
            )
            if in_junction:
                continue

            previous_yaw = float(previous.get("yaw", 0.0))
            previous_yaw_rad = math.radians(previous_yaw)
            lateral_jump_m = abs(
                -dx * math.sin(previous_yaw_rad)
                + dy * math.cos(previous_yaw_rad)
            )
            if lateral_jump_m > self.route_safety_max_lateral_jump_m:
                self._route_validation_debug["route_safety_reject_reason"] = (
                    f"lateral_jump_too_large_at_{index}"
                )
                return False

            heading_jump_deg = angle_diff_deg(float(point.get("yaw", 0.0)), previous_yaw)
            if heading_jump_deg > self.route_safety_max_heading_jump_deg:
                self._route_validation_debug["route_safety_reject_reason"] = (
                    f"heading_jump_too_large_at_{index}"
                )
                return False

        self._route_validation_debug["route_safety_validated"] = True
        self._route_validation_debug["route_safety_reject_reason"] = None
        self._route_validation_debug["published_route_points_count"] = len(points)
        projection_status = str(
            self._right_lane_policy_debug.get("right_lane_projection_status") or ""
        )
        if projection_status == "ok":
            self._route_validation_debug["final_route_source"] = "right_lane_projected_route"
        elif projection_status == "fallback_to_original_continuity_failed":
            self._route_validation_debug["final_route_source"] = "original_carla_route_fallback"
        else:
            self._route_validation_debug["final_route_source"] = "global_route"
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
