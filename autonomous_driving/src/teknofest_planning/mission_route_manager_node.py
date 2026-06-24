import json
import math
import os
from pathlib import Path
import time
from typing import Any, List, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
try:
    from ament_index_python.packages import get_package_share_directory
except Exception:
    get_package_share_directory = None

from teknofest_sim.geojson_mission import load_mission_geojson


class MissionRouteManager(Node):
    def __init__(self):
        super().__init__("mission_route_manager")

        self.declare_parameter("carla_root", "/mnt/carla/CARLA_0.9.15")
        self.declare_parameter("mission_geojson", "")
        self.declare_parameter("target_reached_distance_m", 4.0)
        self.declare_parameter("position_tolerance_m", 2.0)
        self.declare_parameter("yaw_tolerance_deg", 20.0)
        self.declare_parameter("front_bumper_offset_m", 2.0)
        self.declare_parameter("task_stop_position_tolerance_m", 1.0)
        self.declare_parameter("task_stop_front_tolerance_m", 1.0)
        self.declare_parameter("task_stop_yaw_tolerance_deg", 12.0)
        self.declare_parameter("task_pull_over_lateral_offset_m", 1.0)
        self.declare_parameter("task_pull_over_hold_requires_effective_stop", True)
        self.declare_parameter("pickup_hold_s", 16.0)
        self.declare_parameter("dropoff_hold_s", 16.0)
        self.declare_parameter("publish_rate_hz", 2.0)
        self.declare_parameter("loop_mission", False)
        self.declare_parameter("competition_mode", True)

        self.carla_root = self.get_parameter("carla_root").value
        self.mission_geojson = self.get_parameter("mission_geojson").value
        self.target_reached_distance_m = float(self.get_parameter("target_reached_distance_m").value)
        self.position_tolerance_m = float(self.get_parameter("position_tolerance_m").value)
        self.yaw_tolerance_deg = float(self.get_parameter("yaw_tolerance_deg").value)
        self.front_bumper_offset_m = float(self.get_parameter("front_bumper_offset_m").value)
        self.task_stop_position_tolerance_m = float(
            self.get_parameter("task_stop_position_tolerance_m").value
        )
        self.task_stop_front_tolerance_m = float(
            self.get_parameter("task_stop_front_tolerance_m").value
        )
        self.task_stop_yaw_tolerance_deg = float(
            self.get_parameter("task_stop_yaw_tolerance_deg").value
        )
        self.task_pull_over_lateral_offset_m = float(
            self.get_parameter("task_pull_over_lateral_offset_m").value
        )
        self.task_pull_over_hold_requires_effective_stop = bool(
            self.get_parameter("task_pull_over_hold_requires_effective_stop").value
        )
        self.pickup_hold_s = float(self.get_parameter("pickup_hold_s").value)
        self.dropoff_hold_s = float(self.get_parameter("dropoff_hold_s").value)
        self.publish_rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.loop_mission = bool(self.get_parameter("loop_mission").value)
        self.competition_mode = bool(self.get_parameter("competition_mode").value)

        self._mission_spec = None
        self._targets: List[dict[str, Any]] = []
        self._current_target_index: int = 0
        self._resolved_mission_geojson: Optional[str] = None
        self._mission_attempted_paths: list[str] = []
        self._hold_target_index: Optional[int] = None
        self._hold_started_at: float = 0.0
        self._pickup_done = False
        self._dropoff_done = False
        self._park_entry_reached = False
        self._last_status: Optional[dict[str, Any]] = None
        self._last_status_time: float = 0.0
        self._load_mission()

        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)

        self.targets_pub = self.create_publisher(String, "/adas/mission/targets", 10)
        self.current_goal_pub = self.create_publisher(String, "/adas/mission/current_goal", 10)
        self.status_pub = self.create_publisher(String, "/adas/mission/status", 10)

        self.timer = self.create_timer(1.0 / max(0.1, self.publish_rate_hz), self._publish)

    def _candidate_mission_paths(self) -> list[str]:
        if not self.mission_geojson:
            return []

        raw_path = Path(os.path.expanduser(str(self.mission_geojson)))
        candidates: list[Path] = []
        if raw_path.is_absolute():
            candidates.append(raw_path)
        else:
            candidates.append(Path.cwd() / raw_path)
            package_root = Path(__file__).resolve().parents[2]
            candidates.append(package_root / "missions" / raw_path.name)
            candidates.append(package_root / raw_path)
            try:
                install_share = Path.cwd() / "install" / "autonomous_driving" / "share" / "autonomous_driving" / "missions"
                candidates.append(install_share / raw_path.name)
            except Exception:
                pass
            if get_package_share_directory is not None:
                try:
                    package_share = Path(get_package_share_directory("autonomous_driving"))
                    candidates.append(package_share / "missions" / raw_path.name)
                except Exception:
                    pass

        unique = []
        for candidate in candidates:
            resolved = str(candidate)
            if resolved not in unique:
                unique.append(resolved)
        return unique

    def _resolve_mission_geojson(self) -> Optional[str]:
        self._mission_attempted_paths = self._candidate_mission_paths()
        for candidate in self._mission_attempted_paths:
            if os.path.exists(candidate):
                return candidate
        return None

    def _load_mission(self) -> None:
        if not self.mission_geojson:
            self.get_logger().warn("MissionRouteManager: mission_geojson parameter is empty")
            return

        resolved_path = self._resolve_mission_geojson()
        if resolved_path is None:
            self.get_logger().warn(
                "MissionRouteManager: mission_geojson_resolved=false "
                f"mission_geojson={self.mission_geojson} "
                f"attempted_paths={self._mission_attempted_paths}"
            )
            return

        try:
            self._resolved_mission_geojson = resolved_path
            self._mission_spec = load_mission_geojson(
                path=resolved_path,
                competition_mode=self.competition_mode,
            )
            self._targets = self._build_targets()
            self._current_target_index = 0
            self.get_logger().info(
                "MissionRouteManager: mission_geojson_resolved=true "
                f"resolved_path={resolved_path} targets={len(self._targets)}"
            )
        except Exception as exc:
            self._mission_spec = None
            self._targets = []
            self.get_logger().error(f"Failed to load mission geojson: {exc}")

    def _build_targets(self) -> List[dict[str, Any]]:
        if self._mission_spec is None:
            return []

        targets: List[dict[str, Any]] = []
        for task in self._mission_spec.task_points:
            kind = str(task.kind or "task")
            if task.name == "gorev_1":
                kind = "pickup"
            elif task.name == "gorev_2":
                kind = "dropoff"
            if kind not in ("pickup", "dropoff"):
                continue
            targets.append({"role": kind, "mission_stage": kind, **task.__dict__, "kind": kind})
        park_entry = self._mission_spec.park_entry
        targets.append(
            {
                "role": "park_entry",
                "mission_stage": "park_entry",
                **park_entry.__dict__,
                "kind": "park_entry",
            }
        )
        return targets

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
            self._last_status_time = time.time()
        except Exception:
            self.get_logger().warn("MissionRouteManager: failed to parse /adas/carla/status JSON")

    def _is_status_fresh(self) -> bool:
        return (time.time() - self._last_status_time) < 2.0

    @staticmethod
    def _angle_difference_deg(a: float, b: float) -> float:
        return abs((a - b + 180.0) % 360.0 - 180.0)

    def _front_bumper_xy(self) -> Optional[tuple[float, float]]:
        if self._last_status is None:
            return None

        ego_loc = self._last_status.get("location", {})
        ego_rot = self._last_status.get("rotation", {})
        ego_x = float(ego_loc.get("x", 0.0))
        ego_y = float(ego_loc.get("y", 0.0))
        ego_yaw = math.radians(float(ego_rot.get("yaw", 0.0)))
        return (
            ego_x + math.cos(ego_yaw) * self.front_bumper_offset_m,
            ego_y + math.sin(ego_yaw) * self.front_bumper_offset_m,
        )

    def _center_xy(self) -> Optional[tuple[float, float]]:
        if self._last_status is None:
            return None

        ego_loc = self._last_status.get("location", {})
        return (
            float(ego_loc.get("x", 0.0)),
            float(ego_loc.get("y", 0.0)),
        )

    def _distance_to_point(self, x: float, y: float, point: dict[str, Any]) -> float:
        px = float(point.get("carla_x", point.get("x", 0.0)) or 0.0)
        py = float(point.get("carla_y", point.get("y", 0.0)) or 0.0)
        return math.hypot(px - x, py - y)

    def _distance_to_xy(self, x: float, y: float, px: float, py: float) -> float:
        return math.hypot(float(px) - x, float(py) - y)

    def _is_pull_over_task(self, target: dict[str, Any]) -> bool:
        kind = str(target.get("kind", target.get("role", "")))
        mode = str(target.get("task_stop_mode", "pull_over_pose"))
        return kind in ("pickup", "dropoff") and mode == "pull_over_pose"

    def _effective_task_stop_pose(self, target: dict[str, Any]) -> dict[str, Any]:
        base_x = target.get("carla_x", target.get("x"))
        base_y = target.get("carla_y", target.get("y"))
        base_z = target.get("carla_z", target.get("z"))
        base_yaw = target.get("carla_yaw")
        if base_x is None or base_y is None:
            return {
                "base_goal_x": base_x,
                "base_goal_y": base_y,
                "base_goal_yaw": base_yaw,
                "effective_task_stop_x": None,
                "effective_task_stop_y": None,
                "effective_task_stop_z": base_z,
                "effective_task_stop_yaw": base_yaw,
                "effective_task_stop_source": "missing_base_goal",
            }

        raw_x = target.get("task_stop_x")
        raw_y = target.get("task_stop_y")
        raw_yaw = target.get("task_stop_yaw", base_yaw)
        if raw_x is not None and raw_y is not None:
            return {
                "base_goal_x": base_x,
                "base_goal_y": base_y,
                "base_goal_yaw": base_yaw,
                "effective_task_stop_x": float(raw_x),
                "effective_task_stop_y": float(raw_y),
                "effective_task_stop_z": target.get("task_stop_z", base_z),
                "effective_task_stop_yaw": raw_yaw,
                "effective_task_stop_source": "geojson_task_stop",
            }

        yaw_for_offset = float(base_yaw if base_yaw is not None else 0.0)
        yaw_rad = math.radians(yaw_for_offset)
        effective_x = float(base_x) + math.sin(yaw_rad) * self.task_pull_over_lateral_offset_m
        effective_y = float(base_y) - math.cos(yaw_rad) * self.task_pull_over_lateral_offset_m
        return {
            "base_goal_x": float(base_x),
            "base_goal_y": float(base_y),
            "base_goal_yaw": base_yaw,
            "effective_task_stop_x": effective_x,
            "effective_task_stop_y": effective_y,
            "effective_task_stop_z": base_z,
            "effective_task_stop_yaw": base_yaw,
            "effective_task_stop_source": "computed_pull_over",
        }

    def _target_yaw_error_deg(self, target: dict[str, Any]) -> Optional[float]:
        if self._last_status is None:
            return None
        target_yaw = target.get("carla_yaw")
        if target_yaw is None:
            return None
        ego_yaw = float(self._last_status.get("rotation", {}).get("yaw", 0.0))
        return self._angle_difference_deg(ego_yaw, float(target_yaw))

    def _yaw_error_to_deg(self, target_yaw: Any) -> Optional[float]:
        if self._last_status is None or target_yaw is None:
            return None
        ego_yaw = float(self._last_status.get("rotation", {}).get("yaw", 0.0))
        return self._angle_difference_deg(ego_yaw, float(target_yaw))

    def _target_reached(
        self,
        front_bumper_distance_m: Optional[float],
        center_distance_m: Optional[float],
    ) -> bool:
        front_reached = (
            front_bumper_distance_m is not None
            and front_bumper_distance_m <= self.position_tolerance_m
        )
        center_reached = (
            center_distance_m is not None
            and center_distance_m <= self.position_tolerance_m
        )
        return front_reached or center_reached

    def _task_stop_reached(
        self,
        front_bumper_distance_m: Optional[float],
        center_distance_m: Optional[float],
        yaw_error_deg: Optional[float] = None,
    ) -> bool:
        front_reached = (
            front_bumper_distance_m is not None
            and front_bumper_distance_m <= self.task_stop_front_tolerance_m
        )
        center_reached = (
            center_distance_m is not None
            and center_distance_m <= self.task_stop_position_tolerance_m
        )
        yaw_reached = (
            yaw_error_deg is None
            or yaw_error_deg <= self.task_stop_yaw_tolerance_deg
        )
        return (front_reached or center_reached) and yaw_reached

    def _hold_duration_for_target(self, target: dict[str, Any]) -> tuple[float, Optional[str]]:
        kind = str(target.get("kind", target.get("role", "")))
        if kind == "pickup":
            return self.pickup_hold_s, "pickup_hold"
        if kind == "dropoff":
            return self.dropoff_hold_s, "dropoff_hold"
        return 0.0, None

    def _advance_target(self) -> None:
        if self._current_target_index < len(self._targets):
            target = self._targets[self._current_target_index]
            kind = str(target.get("kind", target.get("role", "")))
            if kind == "pickup":
                self._pickup_done = True
            elif kind == "dropoff":
                self._dropoff_done = True
            elif kind == "park_entry":
                self._park_entry_reached = True

        if self._current_target_index >= len(self._targets) - 1:
            if self.loop_mission:
                self._current_target_index = 0
                self._pickup_done = False
                self._dropoff_done = False
                self._park_entry_reached = False
            return
        self._current_target_index += 1

    def _select_current_goal(self, now: float) -> tuple[Optional[dict[str, Any]], Optional[float], dict[str, Any]]:
        state = {
            "mission_stop_active": False,
            "mission_stop_reason": None,
            "mission_hold_remaining_s": 0.0,
            "front_bumper_distance_m": None,
            "center_distance_m": None,
            "distance_to_goal_m": None,
            "yaw_error_deg": None,
            "yaw_within_tolerance": None,
            "task_stop_yaw_error_deg": None,
            "task_stop_yaw_within_tolerance": None,
            "task_stop_yaw_tolerance_deg": self.task_stop_yaw_tolerance_deg,
            "base_goal_x": None,
            "base_goal_y": None,
            "base_goal_yaw": None,
            "effective_task_stop_x": None,
            "effective_task_stop_y": None,
            "effective_task_stop_z": None,
            "effective_task_stop_yaw": None,
            "effective_task_stop_source": None,
            "center_distance_to_effective_task_stop_m": None,
            "front_bumper_distance_to_effective_task_stop_m": None,
            "task_stop_reached_by_mission": False,
            "mission_hold_start_allowed": False,
            "mission_hold_block_reason": None,
        }
        if not self._targets:
            return None, None, state

        if self._last_status is None:
            return self._targets[self._current_target_index], None, state

        front_bumper = self._front_bumper_xy()
        center = self._center_xy()
        if front_bumper is None and center is None:
            return self._targets[self._current_target_index], None, state

        if self._current_target_index < 0:
            self._current_target_index = 0
        if self._current_target_index >= len(self._targets):
            self._current_target_index = max(0, len(self._targets) - 1)

        while self._current_target_index < len(self._targets):
            target = self._targets[self._current_target_index]
            front_distance = (
                self._distance_to_point(front_bumper[0], front_bumper[1], target)
                if front_bumper is not None
                else None
            )
            center_distance = (
                self._distance_to_point(center[0], center[1], target)
                if center is not None
                else None
            )
            distance = min(
                d for d in (front_distance, center_distance) if d is not None
            )
            yaw_error = self._target_yaw_error_deg(target)
            effective_pose = self._effective_task_stop_pose(target)
            task_pull_over = self._is_pull_over_task(target)
            effective_x = effective_pose.get("effective_task_stop_x")
            effective_y = effective_pose.get("effective_task_stop_y")
            effective_yaw = effective_pose.get("effective_task_stop_yaw")
            task_yaw_error = self._yaw_error_to_deg(effective_yaw)
            effective_front_distance = None
            effective_center_distance = None
            if effective_x is not None and effective_y is not None:
                effective_front_distance = (
                    self._distance_to_xy(front_bumper[0], front_bumper[1], effective_x, effective_y)
                    if front_bumper is not None
                    else None
                )
                effective_center_distance = (
                    self._distance_to_xy(center[0], center[1], effective_x, effective_y)
                    if center is not None
                    else None
                )
            task_reached = (
                self._task_stop_reached(
                    effective_front_distance,
                    effective_center_distance,
                    task_yaw_error,
                )
                if task_pull_over
                else False
            )
            hold_requires_effective = task_pull_over and self.task_pull_over_hold_requires_effective_stop
            target_reached = (
                task_reached
                if hold_requires_effective
                else self._target_reached(front_distance, center_distance)
            )
            hold_block_reason = None
            if hold_requires_effective and not task_reached:
                distance_reached_for_debug = self._task_stop_reached(
                    effective_front_distance,
                    effective_center_distance,
                    None,
                )
                yaw_ok_for_debug = (
                    task_yaw_error is None
                    or task_yaw_error <= self.task_stop_yaw_tolerance_deg
                )
                if distance_reached_for_debug and not yaw_ok_for_debug:
                    hold_block_reason = "task_stop_yaw_not_aligned"
                else:
                    hold_block_reason = "effective_task_stop_not_reached"
            state.update(
                {
                    "front_bumper_distance_m": round(front_distance, 3) if front_distance is not None else None,
                    "center_distance_m": round(center_distance, 3) if center_distance is not None else None,
                    "distance_to_goal_m": round(distance, 3),
                    "yaw_error_deg": round(yaw_error, 3) if yaw_error is not None else None,
                    "yaw_within_tolerance": (
                        yaw_error <= self.yaw_tolerance_deg
                        if yaw_error is not None
                        else None
                    ),
                    "task_stop_yaw_error_deg": (
                        round(task_yaw_error, 3)
                        if task_yaw_error is not None
                        else None
                    ),
                    "task_stop_yaw_within_tolerance": (
                        task_yaw_error <= self.task_stop_yaw_tolerance_deg
                        if task_yaw_error is not None
                        else None
                    ),
                    "task_stop_yaw_tolerance_deg": self.task_stop_yaw_tolerance_deg,
                    **effective_pose,
                    "center_distance_to_effective_task_stop_m": (
                        round(effective_center_distance, 3)
                        if effective_center_distance is not None
                        else None
                    ),
                    "front_bumper_distance_to_effective_task_stop_m": (
                        round(effective_front_distance, 3)
                        if effective_front_distance is not None
                        else None
                    ),
                    "task_stop_reached_by_mission": task_reached,
                    "mission_hold_start_allowed": target_reached,
                    "mission_hold_block_reason": hold_block_reason,
                }
            )
            if not target_reached:
                return target, distance, state

            hold_s, hold_reason = self._hold_duration_for_target(target)
            if hold_s > 0.0:
                if self._hold_target_index != self._current_target_index:
                    self._hold_target_index = self._current_target_index
                    self._hold_started_at = now
                elapsed = now - self._hold_started_at
                remaining = max(0.0, hold_s - elapsed)
                if remaining > 0.0:
                    state.update(
                        {
                            "mission_stop_active": True,
                            "mission_stop_reason": hold_reason,
                            "mission_hold_remaining_s": round(remaining, 3),
                        }
                    )
                    return target, distance, state
                self._hold_target_index = None

            self._advance_target()
            if self._current_target_index >= len(self._targets) - 1 and self._park_entry_reached:
                return self._targets[-1], distance, state
            if not self.loop_mission and self._current_target_index == len(self._targets) - 1 and self._park_entry_reached:
                return self._targets[-1], distance, state
            if self._current_target_index < len(self._targets):
                continue
            break

        target = self._targets[-1]
        if center is not None:
            distance = self._distance_to_point(center[0], center[1], target)
        elif front_bumper is not None:
            distance = self._distance_to_point(front_bumper[0], front_bumper[1], target)
        else:
            distance = None
        return target, distance, state

    def _current_goal_payload_fields(
        self,
        current_goal: Optional[dict[str, Any]],
        mission_stop_state: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        mission_stop_state = mission_stop_state or {}
        mission_sequence = [target.get("name") for target in self._targets]
        if current_goal is None:
            return {
                "mission_stage": None,
                "goal_name": None,
                "goal_kind": None,
                "goal_index": None,
                "target_x": None,
                "target_y": None,
                "target_z": None,
                "target_yaw": None,
                "mission_sequence": mission_sequence,
                "pickup_done": self._pickup_done,
                "dropoff_done": self._dropoff_done,
                "park_entry_reached": self._park_entry_reached,
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
                "task_stop_yaw_tolerance_deg": self.task_stop_yaw_tolerance_deg,
                "center_distance_to_effective_task_stop_m": None,
                "front_bumper_distance_to_effective_task_stop_m": None,
                "task_stop_reached_by_mission": False,
                "mission_hold_start_allowed": False,
                "mission_hold_block_reason": None,
            }

        goal_kind = current_goal.get("kind")
        task_stop_required = goal_kind in ("pickup", "dropoff")
        task_hold_s, _ = self._hold_duration_for_target(current_goal)
        distance_to_goal_m = mission_stop_state.get("distance_to_goal_m")
        task_stop_reached_by_mission = bool(
            mission_stop_state.get("task_stop_reached_by_mission", False)
        )
        mission_reached = (
            task_stop_reached_by_mission
            if task_stop_required
            else bool(
                distance_to_goal_m is not None
                and float(distance_to_goal_m) <= self.position_tolerance_m
            )
        )
        mission_hold_active = bool(mission_stop_state.get("mission_stop_active", False))
        mission_approach_active = bool(task_stop_required and not mission_hold_active and not mission_reached)
        effective_pose = self._effective_task_stop_pose(current_goal)
        task_stop_mode = current_goal.get("task_stop_mode")
        if task_stop_mode is None and task_stop_required:
            task_stop_mode = "pull_over_pose"
        task_stop_side = current_goal.get("task_stop_side")
        if task_stop_side is None and task_stop_required:
            task_stop_side = "right"
        task_stop_source = current_goal.get("task_stop_source")
        if (
            task_stop_source is None
            and current_goal.get("task_stop_x") is not None
            and current_goal.get("task_stop_y") is not None
        ):
            task_stop_source = "geojson_task_stop"
        return {
            "mission_stage": current_goal.get("mission_stage", current_goal.get("kind")),
            "goal_name": current_goal.get("name"),
            "goal_kind": current_goal.get("kind"),
            "goal_index": current_goal.get("nokta_id"),
            "target_x": current_goal.get("carla_x"),
            "target_y": current_goal.get("carla_y"),
            "target_z": current_goal.get("carla_z"),
            "target_yaw": current_goal.get("carla_yaw"),
            "task_stop_required": task_stop_required,
            "task_stop_x": current_goal.get("task_stop_x", current_goal.get("carla_x")),
            "task_stop_y": current_goal.get("task_stop_y", current_goal.get("carla_y")),
            "task_stop_z": current_goal.get("task_stop_z", current_goal.get("carla_z")),
            "task_stop_yaw": current_goal.get("task_stop_yaw", current_goal.get("carla_yaw")),
            "task_stop_side": task_stop_side,
            "task_stop_mode": task_stop_mode,
            "task_stop_source": task_stop_source,
            "task_hold_s": task_hold_s if task_stop_required else None,
            **effective_pose,
            "mission_approach_active": mission_approach_active,
            "mission_hold_active": mission_hold_active,
            "mission_reached": mission_reached,
            "center_distance_to_task_stop_m": mission_stop_state.get("center_distance_m"),
            "front_bumper_distance_to_task_stop_m": mission_stop_state.get("front_bumper_distance_m"),
            "center_distance_to_effective_task_stop_m": mission_stop_state.get(
                "center_distance_to_effective_task_stop_m"
            ),
            "front_bumper_distance_to_effective_task_stop_m": mission_stop_state.get(
                "front_bumper_distance_to_effective_task_stop_m"
            ),
            "task_stop_yaw_error_deg": mission_stop_state.get("task_stop_yaw_error_deg"),
            "task_stop_yaw_within_tolerance": mission_stop_state.get(
                "task_stop_yaw_within_tolerance"
            ),
            "task_stop_yaw_tolerance_deg": self.task_stop_yaw_tolerance_deg,
            "task_stop_reached_by_mission": task_stop_reached_by_mission,
            "mission_hold_start_allowed": bool(
                mission_stop_state.get("mission_hold_start_allowed", False)
            ),
            "mission_hold_block_reason": mission_stop_state.get("mission_hold_block_reason"),
            "mission_sequence": mission_sequence,
            "pickup_done": self._pickup_done,
            "dropoff_done": self._dropoff_done,
            "park_entry_reached": self._park_entry_reached,
        }

    def _publish(self) -> None:
        now = time.time()
        mission_targets_payload = {
            "stamp": now,
            "mission_geojson": self.mission_geojson,
            "mission_geojson_resolved": self._resolved_mission_geojson is not None,
            "resolved_path": self._resolved_mission_geojson,
            "attempted_paths": self._mission_attempted_paths,
            "ok": self._mission_spec is not None,
            "targets": self._targets,
        }

        current_goal, distance_to_goal, mission_stop_state = self._select_current_goal(now)
        goal_fields = self._current_goal_payload_fields(current_goal, mission_stop_state)
        current_goal_payload = {
            "stamp": now,
            "mission_geojson": self.mission_geojson,
            "mission_geojson_resolved": self._resolved_mission_geojson is not None,
            "resolved_path": self._resolved_mission_geojson,
            "ok": self._mission_spec is not None and current_goal is not None,
            "current_index": self._current_target_index,
            "current_goal": current_goal,
            "distance_to_goal_m": distance_to_goal,
            "front_bumper_distance_m": mission_stop_state.get("front_bumper_distance_m"),
            "center_distance_m": mission_stop_state.get("center_distance_m"),
            "status_fresh": self._is_status_fresh(),
            **goal_fields,
            **mission_stop_state,
        }

        status_payload = {
            "stamp": now,
            "mission_geojson": self.mission_geojson,
            "mission_geojson_resolved": self._resolved_mission_geojson is not None,
            "resolved_path": self._resolved_mission_geojson,
            "attempted_paths": self._mission_attempted_paths,
            "ok": self._mission_spec is not None,
            "status_fresh": self._is_status_fresh(),
            "mission_loaded": self._mission_spec is not None,
            "target_index": self._current_target_index,
            "position_tolerance_m": self.position_tolerance_m,
            "yaw_tolerance_deg": self.yaw_tolerance_deg,
            "front_bumper_offset_m": self.front_bumper_offset_m,
            "distance_to_goal_m": distance_to_goal,
            "front_bumper_distance_m": mission_stop_state.get("front_bumper_distance_m"),
            "center_distance_m": mission_stop_state.get("center_distance_m"),
            **goal_fields,
            **mission_stop_state,
        }

        self.targets_pub.publish(String(data=json.dumps(mission_targets_payload)))
        self.current_goal_pub.publish(String(data=json.dumps(current_goal_payload)))
        self.status_pub.publish(String(data=json.dumps(status_payload)))


def main(args=None):
    rclpy.init(args=args)
    node = MissionRouteManager()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
