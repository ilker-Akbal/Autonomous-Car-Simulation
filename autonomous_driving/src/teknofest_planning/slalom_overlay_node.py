import copy
import json
import math
import time
from typing import Any, Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


Point2 = tuple[float, float]
Point3 = tuple[float, float, float]


def _finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result):
        return None
    return result


def _distance_xy(a: Point2, b: Point2) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _yaw_deg(x0: float, y0: float, x1: float, y1: float, fallback: float = 0.0) -> float:
    dx = x1 - x0
    dy = y1 - y0
    if dx * dx + dy * dy <= 1e-12:
        return fallback
    return math.degrees(math.atan2(dy, dx))


def _catmull_rom(p0: Point3, p1: Point3, p2: Point3, p3: Point3, t: float) -> Point3:
    t2 = t * t
    t3 = t2 * t
    return (
        0.5
        * (
            (2.0 * p1[0])
            + (-p0[0] + p2[0]) * t
            + (2.0 * p0[0] - 5.0 * p1[0] + 4.0 * p2[0] - p3[0]) * t2
            + (-p0[0] + 3.0 * p1[0] - 3.0 * p2[0] + p3[0]) * t3
        ),
        0.5
        * (
            (2.0 * p1[1])
            + (-p0[1] + p2[1]) * t
            + (2.0 * p0[1] - 5.0 * p1[1] + 4.0 * p2[1] - p3[1]) * t2
            + (-p0[1] + 3.0 * p1[1] - 3.0 * p2[1] + p3[1]) * t3
        ),
        0.5
        * (
            (2.0 * p1[2])
            + (-p0[2] + p2[2]) * t
            + (2.0 * p0[2] - 5.0 * p1[2] + 4.0 * p2[2] - p3[2]) * t2
            + (-p0[2] + 3.0 * p1[2] - 3.0 * p2[2] + p3[2]) * t3
        ),
    )


class SlalomOverlay(Node):
    def __init__(self) -> None:
        super().__init__("slalom_overlay")

        self.declare_parameter("enable_slalom", False)
        self.declare_parameter("slalom_plan_json", "")
        self.declare_parameter("slalom_start_side", "left")
        self.declare_parameter("slalom_clearance_m", 1.45)
        self.declare_parameter("slalom_speed_mps", 1.2)
        self.declare_parameter("slalom_activation_horizon_m", 80.0)
        self.declare_parameter("slalom_route_corridor_m", 6.0)

        self.enable_slalom = bool(self.get_parameter("enable_slalom").value)
        self.plan_json = str(self.get_parameter("slalom_plan_json").value)
        self.param_start_side = str(self.get_parameter("slalom_start_side").value).strip().lower()
        self.param_clearance_m = float(self.get_parameter("slalom_clearance_m").value)
        self.param_speed_mps = float(self.get_parameter("slalom_speed_mps").value)
        self.param_activation_distance_m = float(
            self.get_parameter("slalom_activation_horizon_m").value
        )

        self.config = self._load_config(self.plan_json)
        self._last_status: Optional[dict[str, Any]] = None
        self._slalom_active = False

        self.route_pub = self.create_publisher(String, "/adas/planning/route", 10)
        self.debug_pub = self.create_publisher(String, "/adas/planning/slalom_debug", 10)

        self.create_subscription(String, "/adas/planning/route_base", self._route_callback, 10)
        self.create_subscription(String, "/adas/carla/status", self._status_callback, 10)

    def _load_config(self, path: str) -> dict[str, Any]:
        defaults = {
            "enabled": True,
            "source": "fixed_world_spline",
            "coordinate_unit": "meters",
            "start_side": "left",
            "slalom_clearance_m": 1.45,
            "slalom_speed_mps": 1.2,
            "override_slalom_speed_limit": False,
            "entry_margin_m": 24.0,
            "exit_margin_m": 16.0,
            "pre_pass_lead_m": 7.0,
            "post_pass_hold_m": 4.0,
            "sample_step_m": 0.35,
            "activation_distance_m": 80.0,
            "max_slalom_segment_m": 1.2,
            "max_join_segment_m": 10.0,
            "max_yaw_jump_deg": 60.0,
            "min_output_route_len": 40,
            "use_base_route_entry_exit": True,
            "neutral_lane_y_m": -5.2,
            "lane_y_min": -6.95,
            "lane_y_max": -3.45,
            "vehicle_half_width_m": 0.95,
            "cone_radius_m": 0.18,
            "cone_safety_margin_m": 0.15,
            "min_required_center_clearance_m": 1.25,
            "keep_vehicle_body_inside_lane": False,
            "allow_slalom_if_geometry_infeasible": True,
            "cones": [],
        }
        if not path:
            self.get_logger().warn("SlalomOverlay: slalom_plan_json is empty")
            return defaults
        try:
            with open(path, "r", encoding="utf-8") as handle:
                loaded = json.load(handle)
        except Exception as exc:
            self.get_logger().warn(f"SlalomOverlay: failed to load plan JSON: {exc}")
            return defaults
        if not isinstance(loaded, dict):
            self.get_logger().warn("SlalomOverlay: plan JSON root must be an object")
            return defaults
        merged = dict(defaults)
        merged.update(loaded)
        return merged

    def _status_callback(self, msg: String) -> None:
        try:
            self._last_status = json.loads(msg.data)
        except Exception:
            self.get_logger().warn("SlalomOverlay: failed to parse /adas/carla/status JSON")

    def _ego_xy(self, payload: dict[str, Any]) -> tuple[Optional[float], Optional[float]]:
        if isinstance(self._last_status, dict):
            loc = self._last_status.get("location", {})
            if isinstance(loc, dict):
                x = _finite_float(loc.get("x"))
                y = _finite_float(loc.get("y"))
                if x is not None and y is not None:
                    return x, y
        return _finite_float(payload.get("ego_x")), _finite_float(payload.get("ego_y"))

    def _runtime_settings(self) -> dict[str, Any]:
        start_side = str(self.config.get("start_side", self.param_start_side)).strip().lower()
        if start_side not in ("right", "left"):
            start_side = "left"
        clearance_m = _finite_float(self.config.get("slalom_clearance_m"))
        speed_mps = _finite_float(self.config.get("slalom_speed_mps"))
        entry_margin_m = _finite_float(self.config.get("entry_margin_m"))
        exit_margin_m = _finite_float(self.config.get("exit_margin_m"))
        pre_pass_lead_m = _finite_float(self.config.get("pre_pass_lead_m"))
        post_pass_hold_m = _finite_float(self.config.get("post_pass_hold_m"))
        sample_step_m = _finite_float(self.config.get("sample_step_m"))
        activation_distance_m = _finite_float(self.config.get("activation_distance_m"))
        max_slalom_segment_m = _finite_float(self.config.get("max_slalom_segment_m"))
        max_join_segment_m = _finite_float(self.config.get("max_join_segment_m"))
        max_yaw_jump_deg = _finite_float(self.config.get("max_yaw_jump_deg"))
        min_output_route_len = _finite_float(self.config.get("min_output_route_len"))
        neutral_lane_y_m = _finite_float(self.config.get("neutral_lane_y_m"))
        lane_y_min = _finite_float(self.config.get("lane_y_min"))
        lane_y_max = _finite_float(self.config.get("lane_y_max"))
        lane_min = lane_y_min if lane_y_min is not None else -6.95
        lane_max = lane_y_max if lane_y_max is not None else -3.45
        if lane_min > lane_max:
            lane_min, lane_max = lane_max, lane_min
        return {
            "start_side": start_side,
            "clearance_m": max(0.1, clearance_m if clearance_m is not None else self.param_clearance_m),
            "speed_mps": max(0.05, speed_mps if speed_mps is not None else self.param_speed_mps),
            "override_slalom_speed_limit": bool(
                self.config.get("override_slalom_speed_limit", False)
            ),
            "entry_margin_m": max(0.0, entry_margin_m if entry_margin_m is not None else 24.0),
            "exit_margin_m": max(0.0, exit_margin_m if exit_margin_m is not None else 16.0),
            "pre_pass_lead_m": max(
                0.0,
                pre_pass_lead_m if pre_pass_lead_m is not None else 7.0,
            ),
            "post_pass_hold_m": max(
                0.0,
                post_pass_hold_m if post_pass_hold_m is not None else 4.0,
            ),
            "sample_step_m": max(0.1, sample_step_m if sample_step_m is not None else 0.35),
            "activation_distance_m": max(
                0.0,
                activation_distance_m
                if activation_distance_m is not None
                else self.param_activation_distance_m,
            ),
            "max_slalom_segment_m": max(
                0.1,
                max_slalom_segment_m if max_slalom_segment_m is not None else 1.2,
            ),
            "max_join_segment_m": max(
                0.1,
                max_join_segment_m if max_join_segment_m is not None else 10.0,
            ),
            "max_yaw_jump_deg": max(
                1.0,
                max_yaw_jump_deg if max_yaw_jump_deg is not None else 60.0,
            ),
            "min_output_route_len": max(
                2,
                int(min_output_route_len if min_output_route_len is not None else 40),
            ),
            "use_base_route_entry_exit": bool(
                self.config.get("use_base_route_entry_exit", True)
            ),
            "neutral_lane_y_m": (
                neutral_lane_y_m if neutral_lane_y_m is not None else -5.2
            ),
            "lane_y_min": lane_min,
            "lane_y_max": lane_max,
        }

    def _config_cones(self) -> list[dict[str, Any]]:
        cones = self.config.get("cones", [])
        if not isinstance(cones, list):
            return []
        result: list[dict[str, Any]] = []
        for cone in cones:
            if not isinstance(cone, dict):
                continue
            x = _finite_float(cone.get("x"))
            y = _finite_float(cone.get("y"))
            z = _finite_float(cone.get("z")) or 0.0
            if x is None or y is None:
                continue
            result.append(
                {
                    "name": str(cone.get("name", f"cone_{len(result) + 1}")),
                    "x": x,
                    "y": y,
                    "z": z,
                }
            )
        return result

    def _clamp_point_y(self, point: Point3, settings: dict[str, Any]) -> tuple[Point3, bool]:
        clamped_y = max(settings["lane_y_min"], min(settings["lane_y_max"], point[1]))
        return (point[0], clamped_y, point[2]), abs(clamped_y - point[1]) > 1e-9

    def _base_route_y_near(
        self,
        points: list[dict[str, Any]],
        x: float,
        y: float,
        fallback_y: float,
    ) -> tuple[float, Optional[int]]:
        index = self._nearest_index(points, x, y)
        if index is None:
            return fallback_y, None
        route_y = _finite_float(points[index].get("y"))
        if route_y is None:
            return fallback_y, index
        return route_y, index

    def _slalom_geometry(
        self,
        cones: list[dict[str, Any]],
        settings: dict[str, Any],
        base_points: list[dict[str, Any]],
    ) -> tuple[Optional[dict[str, Any]], str]:
        if len(cones) < 3:
            return None, "not_enough_config_cones"

        cone1 = cones[0]
        cone2 = cones[1]
        cone3 = cones[2]
        dx = float(cone3["x"]) - float(cone1["x"])
        dy = float(cone3["y"]) - float(cone1["y"])
        length = math.hypot(dx, dy)
        if length <= 1e-6:
            return None, "invalid_cone_direction"

        dir_x = dx / length
        dir_y = dy / length
        left = (-dir_y, dir_x)
        right = (dir_y, -dir_x)

        clamped_count = 0

        def clamp(point: Point3) -> Point3:
            nonlocal clamped_count
            clamped, changed = self._clamp_point_y(point, settings)
            if changed:
                clamped_count += 1
            return clamped

        def shifted(cone: dict[str, Any], forward_m: float, normal: Point2) -> Point3:
            return clamp(
                (
                    float(cone["x"]) + dir_x * forward_m + normal[0] * settings["clearance_m"],
                    float(cone["y"]) + dir_y * forward_m + normal[1] * settings["clearance_m"],
                    float(cone.get("z", 0.0)),
                )
            )

        nominal_entry = (
            float(cone1["x"]) - dir_x * settings["entry_margin_m"],
            float(cone1["y"]) - dir_y * settings["entry_margin_m"],
            float(cone1.get("z", 0.0)),
        )
        nominal_exit = (
            float(cone3["x"]) + dir_x * settings["exit_margin_m"],
            float(cone3["y"]) + dir_y * settings["exit_margin_m"],
            float(cone3.get("z", 0.0)),
        )
        if settings["use_base_route_entry_exit"]:
            entry_y, entry_base_index = self._base_route_y_near(
                base_points,
                nominal_entry[0],
                nominal_entry[1],
                settings["neutral_lane_y_m"],
            )
            exit_y, exit_base_index = self._base_route_y_near(
                base_points,
                nominal_exit[0],
                nominal_exit[1],
                settings["neutral_lane_y_m"],
            )
        else:
            entry_y = settings["neutral_lane_y_m"]
            exit_y = settings["neutral_lane_y_m"]
            entry_base_index = None
            exit_base_index = None

        entry = clamp((nominal_entry[0], entry_y, nominal_entry[2]))
        exit_point = clamp((nominal_exit[0], exit_y, nominal_exit[2]))
        if settings["start_side"] == "left":
            sides = [left, right, left]
        else:
            sides = [right, left, right]

        pre_pass_points: list[Point3] = []
        pass_points: list[Point3] = []
        post_pass_points: list[Point3] = []
        controls: list[Point3] = [entry]
        for cone, side in zip((cone1, cone2, cone3), sides):
            pre_pass = shifted(cone, -settings["pre_pass_lead_m"], side)
            pass_point = shifted(cone, 0.0, side)
            post_pass = shifted(cone, settings["post_pass_hold_m"], side)
            pre_pass_points.append(pre_pass)
            pass_points.append(pass_point)
            post_pass_points.append(post_pass)
            controls.extend([pre_pass, pass_point, post_pass])
        controls.append(exit_point)

        return {
            "direction": (dir_x, dir_y),
            "left": left,
            "right": right,
            "entry": entry,
            "pre_passes": pre_pass_points,
            "passes": pass_points,
            "post_passes": post_pass_points,
            "exit": exit_point,
            "entry_base_route_y": entry_y,
            "exit_base_route_y": exit_y,
            "entry_base_index": entry_base_index,
            "exit_base_index": exit_base_index,
            "controls": controls,
            "clamped_points_count": clamped_count,
        }, "ok"

    def _sample_spline(
        self,
        controls: list[Point3],
        sample_step_m: float,
        settings: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], int]:
        if len(controls) < 2:
            return [], 0

        padded = [controls[0], *controls, controls[-1]]
        samples: list[Point3] = []
        clamped_count = 0
        for index in range(1, len(padded) - 2):
            p0 = padded[index - 1]
            p1 = padded[index]
            p2 = padded[index + 1]
            p3 = padded[index + 2]
            chord = _distance_xy((p1[0], p1[1]), (p2[0], p2[1]))
            steps = max(1, int(math.ceil(chord / sample_step_m)))
            start_step = 0 if index == 1 else 1
            for step in range(start_step, steps + 1):
                t = step / steps
                point, changed = self._clamp_point_y(_catmull_rom(p0, p1, p2, p3, t), settings)
                if changed:
                    clamped_count += 1
                samples.append(point)

        points = [
            {
                "x": round(point[0], 3),
                "y": round(point[1], 3),
                "z": round(point[2], 3),
                "slalom_overlay": True,
                "target_source": "fixed_world_slalom_spline",
            }
            for point in samples
        ]
        self._set_yaws(points)
        return points, clamped_count

    def _set_yaws(self, points: list[dict[str, Any]]) -> None:
        for index, point in enumerate(points):
            previous_index = max(0, index - 1)
            next_index = min(len(points) - 1, index + 1)
            previous = points[previous_index]
            next_point = points[next_index]
            fallback = _finite_float(point.get("yaw")) or 0.0
            point["yaw"] = round(
                _yaw_deg(
                    float(previous.get("x", point.get("x", 0.0))),
                    float(previous.get("y", point.get("y", 0.0))),
                    float(next_point.get("x", point.get("x", 0.0))),
                    float(next_point.get("y", point.get("y", 0.0))),
                    fallback,
                ),
                3,
            )

    def _max_yaw_jump(self, points: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {"jump": 0.0, "index": None}
        previous_yaw: Optional[float] = None
        for index, point in enumerate(points):
            yaw = _finite_float(point.get("yaw"))
            if yaw is None:
                continue
            if previous_yaw is not None:
                jump = abs((yaw - previous_yaw + 180.0) % 360.0 - 180.0)
                if jump > float(result["jump"]):
                    result = {"jump": jump, "index": index}
            previous_yaw = yaw
        return result

    def _nearest_index(self, points: list[dict[str, Any]], x: float, y: float) -> Optional[int]:
        best_index = None
        best_distance = float("inf")
        for index, point in enumerate(points):
            px = _finite_float(point.get("x"))
            py = _finite_float(point.get("y"))
            if px is None or py is None:
                continue
            distance = math.hypot(px - x, py - y)
            if distance < best_distance:
                best_distance = distance
                best_index = index
        return best_index

    def _first_index_after_exit(
        self,
        points: list[dict[str, Any]],
        exit_point: Point3,
        direction: Point2,
    ) -> Optional[int]:
        best_index = None
        best_ahead = float("inf")
        best_distance = float("inf")
        for index, point in enumerate(points):
            px = _finite_float(point.get("x"))
            py = _finite_float(point.get("y"))
            if px is None or py is None:
                continue
            ahead = (px - exit_point[0]) * direction[0] + (py - exit_point[1]) * direction[1]
            distance = math.hypot(px - exit_point[0], py - exit_point[1])
            if ahead >= -0.5 and (ahead < best_ahead or (abs(ahead - best_ahead) < 1e-6 and distance < best_distance)):
                best_index = index
                best_ahead = ahead
                best_distance = distance
        if best_index is not None:
            return best_index
        return self._nearest_index(points, exit_point[0], exit_point[1])

    def _copy_template_fields(
        self,
        slalom_points: list[dict[str, Any]],
        base_points: list[dict[str, Any]],
        speed_mps: float,
        override_speed_limit: bool,
    ) -> None:
        if not base_points:
            for point in slalom_points:
                self._apply_or_clear_slalom_speed_limit(point, speed_mps, override_speed_limit)
            return
        for point in slalom_points:
            nearest = self._nearest_index(base_points, float(point["x"]), float(point["y"]))
            template = copy.deepcopy(base_points[nearest]) if nearest is not None else {}
            template.update(point)
            self._apply_or_clear_slalom_speed_limit(template, speed_mps, override_speed_limit)
            template["slalom_overlay"] = True
            template["target_source"] = "fixed_world_slalom_spline"
            point.clear()
            point.update(template)

    def _apply_or_clear_slalom_speed_limit(
        self,
        point: dict[str, Any],
        speed_mps: float,
        override_speed_limit: bool,
    ) -> None:
        if override_speed_limit:
            point["speed_limit_mps"] = round(max(0.05, speed_mps), 3)
            return
        for field in (
            "speed_limit_mps",
            "speed_limit_kmh",
            "target_speed_mps",
            "target_speed_limit_mps",
        ):
            point.pop(field, None)

    def _apply_slalom_speeds(
        self,
        slalom_points: list[dict[str, Any]],
        speed_mps: float,
        override_speed_limit: bool,
    ) -> None:
        for point in slalom_points:
            self._apply_or_clear_slalom_speed_limit(point, speed_mps, override_speed_limit)

    def _slalom_points_with_speed_limit_count(self, slalom_points: list[dict[str, Any]]) -> int:
        return sum(1 for point in slalom_points if point.get("speed_limit_mps") is not None)

    def _recompute_route_s(self, points: list[dict[str, Any]]) -> None:
        distance = 0.0
        previous_xy: Optional[Point2] = None
        for point in points:
            x = _finite_float(point.get("x"))
            y = _finite_float(point.get("y"))
            if x is None or y is None:
                point["s"] = round(distance, 3)
                continue
            if previous_xy is not None:
                distance += math.hypot(x - previous_xy[0], y - previous_xy[1])
            point["s"] = round(distance, 3)
            previous_xy = (x, y)

    def _assign_slalom_s(
        self,
        slalom_points: list[dict[str, Any]],
        start_s: Optional[float],
        end_s: Optional[float],
    ) -> None:
        if not slalom_points:
            return
        distances = [0.0]
        for previous, current in zip(slalom_points, slalom_points[1:]):
            distances.append(
                distances[-1]
                + _distance_xy(
                    (float(previous.get("x", 0.0)), float(previous.get("y", 0.0))),
                    (float(current.get("x", 0.0)), float(current.get("y", 0.0))),
                )
            )
        total = distances[-1]
        if start_s is not None and end_s is not None and end_s > start_s + 1e-6:
            span = end_s - start_s
            for point, distance in zip(slalom_points, distances):
                fraction = 0.0 if total <= 1e-6 else distance / total
                point["s"] = round(start_s + fraction * span, 3)
            return
        cursor = start_s if start_s is not None else 0.0
        previous_distance = 0.0
        for point, distance in zip(slalom_points, distances):
            cursor += distance - previous_distance
            point["s"] = round(cursor, 3)
            previous_distance = distance

    def _max_segment_distance(self, points: list[dict[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "distance": 0.0,
            "index": None,
            "from": None,
            "to": None,
        }
        for index, (previous, current) in enumerate(zip(points, points[1:])):
            px = _finite_float(previous.get("x"))
            py = _finite_float(previous.get("y"))
            cx = _finite_float(current.get("x"))
            cy = _finite_float(current.get("y"))
            if px is None or py is None or cx is None or cy is None:
                continue
            distance = math.hypot(cx - px, cy - py)
            if distance <= float(result["distance"]):
                continue
            result = {
                "distance": distance,
                "index": index,
                "from": {
                    "x": previous.get("x"),
                    "y": previous.get("y"),
                    "slalom_overlay": bool(previous.get("slalom_overlay", False)),
                },
                "to": {
                    "x": current.get("x"),
                    "y": current.get("y"),
                    "slalom_overlay": bool(current.get("slalom_overlay", False)),
                },
            }
        return result

    def _point_payload(self, point: Point3) -> dict[str, float]:
        return {
            "x": round(point[0], 3),
            "y": round(point[1], 3),
            "z": round(point[2], 3),
        }

    def _base_debug(self, payload: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        points = payload.get("points", []) if isinstance(payload, dict) else []
        output_route_len = len(points) if isinstance(points, list) else None
        settings = self._runtime_settings()
        return {
            "stamp": time.time(),
            "slalom_active": False,
            "slalom_reason": "not_evaluated",
            "mode": "fixed_world_spline",
            "enable_slalom": self.enable_slalom,
            "config_enabled": bool(self.config.get("enabled", False)),
            "preserve_classic_route_outside_slalom": True,
            "preserve_classic_speed_profile": True,
            "use_base_route_entry_exit": settings["use_base_route_entry_exit"],
            "ego_x": None,
            "ego_y": None,
            "cone_world_points": self._config_cones(),
            "entry_point": None,
            "entry_base_route_y": None,
            "pre_pass_points": [],
            "pass_points": [],
            "post_pass_points": [],
            "exit_point": None,
            "exit_base_route_y": None,
            "start_side": settings["start_side"],
            "slalom_clearance_m": round(settings["clearance_m"], 3),
            "slalom_speed_mps": round(settings["speed_mps"], 3),
            "override_slalom_speed_limit": settings["override_slalom_speed_limit"],
            "slalom_speed_limit_applied": False,
            "slalom_points_with_speed_limit_count": 0,
            "lane_y_min": round(settings["lane_y_min"], 3),
            "lane_y_max": round(settings["lane_y_max"], 3),
            "clamped_points_count": 0,
            "spline_point_count": 0,
            "spline_remaining_count": 0,
            "base_suffix_count": 0,
            "max_slalom_segment_m": round(settings["max_slalom_segment_m"], 3),
            "max_join_segment_m": round(settings["max_join_segment_m"], 3),
            "min_output_route_len": settings["min_output_route_len"],
            "route_invalid_prevented": False,
            "route_source": payload.get("route_source") if isinstance(payload, dict) else None,
            "route_ok": payload.get("route_ok") if isinstance(payload, dict) else None,
            "route_safety_validated": (
                payload.get("route_safety_validated") if isinstance(payload, dict) else None
            ),
            "final_route_source": (
                payload.get("final_route_source") if isinstance(payload, dict) else None
            ),
            "output_route_len": output_route_len,
            "activation_distance_m": round(settings["activation_distance_m"], 3),
            "distance_to_entry_m": None,
            "distance_to_cone_1_m": None,
            "exit_progress_m": None,
            "entry_index": None,
            "exit_index": None,
            "prefix_base_count": 0,
            "suffix_base_count": 0,
            "slalom_replaced_count": 0,
            "join_start_distance_m": None,
            "join_exit_distance_m": None,
            "max_spline_segment_distance_m": None,
            "max_spline_segment_index": None,
            "max_yaw_jump_deg": None,
            "yaw_jump_index": None,
        }

    def _apply_overlay(
        self,
        payload: dict[str, Any],
    ) -> tuple[Optional[dict[str, Any]], dict[str, Any], str]:
        debug = self._base_debug(payload)
        settings = self._runtime_settings()

        if not self.enable_slalom:
            self._slalom_active = False
            debug["slalom_reason"] = "disabled_by_launch"
            return None, debug, "disabled_by_launch"
        if not bool(self.config.get("enabled", False)):
            self._slalom_active = False
            debug["slalom_reason"] = "disabled_by_config"
            return None, debug, "disabled_by_config"
        if str(self.config.get("source", "fixed_world_spline")) != "fixed_world_spline":
            self._slalom_active = False
            debug["slalom_reason"] = "config_source_not_fixed_world_spline"
            return None, debug, "config_source_not_fixed_world_spline"
        if payload.get("route_source") != "global_route":
            self._slalom_active = False
            debug["slalom_reason"] = "route_source_not_global_route"
            return None, debug, "route_source_not_global_route"
        if not bool(payload.get("route_ok", False)):
            self._slalom_active = False
            debug["slalom_reason"] = "route_not_ok"
            return None, debug, "route_not_ok"

        points = payload.get("points", [])
        if not isinstance(points, list) or len(points) < 2:
            self._slalom_active = False
            debug["slalom_reason"] = "not_enough_route_points"
            return None, debug, "not_enough_route_points"
        if not all(isinstance(point, dict) for point in points):
            self._slalom_active = False
            debug["slalom_reason"] = "invalid_route_points"
            return None, debug, "invalid_route_points"

        ego_x, ego_y = self._ego_xy(payload)
        if ego_x is None or ego_y is None:
            self._slalom_active = False
            debug["slalom_reason"] = "ego_position_missing"
            return None, debug, "ego_position_missing"
        debug["ego_x"] = round(ego_x, 3)
        debug["ego_y"] = round(ego_y, 3)

        cones = self._config_cones()
        geometry, geometry_reason = self._slalom_geometry(cones, settings, points)
        if geometry is None:
            self._slalom_active = False
            debug["slalom_reason"] = geometry_reason
            return None, debug, geometry_reason

        entry = geometry["entry"]
        pre_pass_points = geometry["pre_passes"]
        pass_points = geometry["passes"]
        post_pass_points = geometry["post_passes"]
        exit_point = geometry["exit"]
        direction = geometry["direction"]
        debug.update(
            {
                "entry_point": self._point_payload(entry),
                "entry_base_route_y": round(float(geometry["entry_base_route_y"]), 3),
                "pre_pass_points": [self._point_payload(point) for point in pre_pass_points],
                "pass_points": [self._point_payload(point) for point in pass_points],
                "post_pass_points": [self._point_payload(point) for point in post_pass_points],
                "exit_point": self._point_payload(exit_point),
                "exit_base_route_y": round(float(geometry["exit_base_route_y"]), 3),
                "clamped_points_count": int(geometry["clamped_points_count"]),
            }
        )

        ego_xy = (ego_x, ego_y)
        distance_to_entry = _distance_xy(ego_xy, (entry[0], entry[1]))
        distance_to_cone_1 = _distance_xy(ego_xy, (float(cones[0]["x"]), float(cones[0]["y"])))
        exit_progress = (ego_x - exit_point[0]) * direction[0] + (ego_y - exit_point[1]) * direction[1]
        debug["distance_to_entry_m"] = round(distance_to_entry, 3)
        debug["distance_to_cone_1_m"] = round(distance_to_cone_1, 3)
        debug["exit_progress_m"] = round(exit_progress, 3)

        if exit_progress > 1.0:
            self._slalom_active = False
            debug["slalom_reason"] = "slalom_exit_passed"
            return None, debug, "slalom_exit_passed"

        can_activate = (
            distance_to_entry <= settings["activation_distance_m"]
            or distance_to_cone_1 <= settings["activation_distance_m"]
        )
        if not self._slalom_active and not can_activate:
            debug["slalom_reason"] = "outside_activation_distance"
            return None, debug, "outside_activation_distance"
        self._slalom_active = True

        full_slalom_points, spline_clamped_count = self._sample_spline(
            geometry["controls"],
            settings["sample_step_m"],
            settings,
        )
        debug["clamped_points_count"] = (
            int(debug["clamped_points_count"]) + spline_clamped_count
        )
        if len(full_slalom_points) < 2:
            self._slalom_active = False
            debug["slalom_reason"] = "slalom_sampling_failed"
            return None, debug, "slalom_sampling_failed"
        self._copy_template_fields(
            full_slalom_points,
            points,
            settings["speed_mps"],
            settings["override_slalom_speed_limit"],
        )
        self._apply_slalom_speeds(
            full_slalom_points,
            settings["speed_mps"],
            settings["override_slalom_speed_limit"],
        )
        debug["slalom_points_with_speed_limit_count"] = (
            self._slalom_points_with_speed_limit_count(full_slalom_points)
        )
        debug["slalom_speed_limit_applied"] = bool(
            settings["override_slalom_speed_limit"]
            and debug["slalom_points_with_speed_limit_count"] > 0
        )

        spline_segment_debug = self._max_segment_distance(full_slalom_points)
        yaw_jump_debug = self._max_yaw_jump(full_slalom_points)
        debug["spline_point_count"] = len(full_slalom_points)
        debug["max_spline_segment_distance_m"] = round(float(spline_segment_debug["distance"]), 3)
        debug["max_spline_segment_index"] = spline_segment_debug["index"]
        debug["max_yaw_jump_deg"] = round(float(yaw_jump_debug["jump"]), 3)
        debug["yaw_jump_index"] = yaw_jump_debug["index"]
        if float(spline_segment_debug["distance"]) > settings["max_slalom_segment_m"]:
            debug["slalom_reason"] = "fixed_spline_segment_large"
            debug["route_invalid_prevented"] = True
            return None, debug, "fixed_spline_segment_large"
        if float(yaw_jump_debug["jump"]) > settings["max_yaw_jump_deg"]:
            debug["slalom_reason"] = "fixed_spline_yaw_jump_passthrough"
            debug["route_invalid_prevented"] = True
            return None, debug, "fixed_spline_yaw_jump_passthrough"

        entry_index = self._nearest_index(points, entry[0], entry[1])
        exit_index = self._first_index_after_exit(points, exit_point, direction)
        if entry_index is None or exit_index is None:
            debug["slalom_reason"] = "base_route_join_failed"
            debug["route_invalid_prevented"] = True
            return None, debug, "base_route_join_failed"
        debug["entry_index"] = entry_index
        debug["exit_index"] = exit_index
        if exit_index <= entry_index:
            debug["slalom_reason"] = "base_route_window_invalid"
            debug["route_invalid_prevented"] = True
            return None, debug, "base_route_window_invalid"

        entry_join = points[entry_index]
        spline_start_index = self._nearest_index(
            full_slalom_points,
            float(entry_join.get("x", entry[0])),
            float(entry_join.get("y", entry[1])),
        )
        if spline_start_index is None:
            debug["slalom_reason"] = "base_route_join_failed"
            debug["route_invalid_prevented"] = True
            return None, debug, "base_route_join_failed"
        slalom_points = full_slalom_points[spline_start_index:]
        if len(slalom_points) < 2:
            debug["slalom_reason"] = "slalom_remaining_too_short"
            debug["route_invalid_prevented"] = True
            return None, debug, "slalom_remaining_too_short"
        debug["slalom_points_with_speed_limit_count"] = (
            self._slalom_points_with_speed_limit_count(slalom_points)
        )
        debug["slalom_speed_limit_applied"] = bool(
            settings["override_slalom_speed_limit"]
            and debug["slalom_points_with_speed_limit_count"] > 0
        )
        prefix_points = copy.deepcopy(points[: entry_index + 1])
        if not prefix_points:
            debug["slalom_reason"] = "base_route_join_failed"
            debug["route_invalid_prevented"] = True
            return None, debug, "base_route_join_failed"
        base_suffix = copy.deepcopy(points[exit_index:])
        debug["spline_remaining_count"] = len(slalom_points)
        debug["base_suffix_count"] = len(base_suffix)
        debug["prefix_base_count"] = len(prefix_points)
        debug["suffix_base_count"] = len(base_suffix)
        debug["slalom_replaced_count"] = max(0, exit_index - entry_index - 1)
        exit_join_point = points[exit_index]
        start_join_distance = _distance_xy(
            (
                float(prefix_points[-1].get("x", ego_x)),
                float(prefix_points[-1].get("y", ego_y)),
            ),
            (float(slalom_points[0]["x"]), float(slalom_points[0]["y"])),
        )
        exit_join_distance = _distance_xy(
            (float(slalom_points[-1]["x"]), float(slalom_points[-1]["y"])),
            (float(exit_join_point.get("x", exit_point[0])), float(exit_join_point.get("y", exit_point[1]))),
        )
        debug["join_start_distance_m"] = round(start_join_distance, 3)
        debug["join_exit_distance_m"] = round(exit_join_distance, 3)
        if (
            start_join_distance > settings["max_join_segment_m"]
            or exit_join_distance > settings["max_join_segment_m"]
        ):
            debug["slalom_reason"] = "fixed_spline_join_large_passthrough"
            debug["route_invalid_prevented"] = True
            return None, debug, "fixed_spline_join_large_passthrough"

        self._assign_slalom_s(
            slalom_points,
            _finite_float(prefix_points[-1].get("s")),
            _finite_float(exit_join_point.get("s")),
        )

        final_points: list[dict[str, Any]] = []
        final_points.extend(prefix_points)
        final_points.extend(copy.deepcopy(slalom_points))
        final_points.extend(base_suffix)
        if len(final_points) < 2:
            debug["slalom_reason"] = "final_route_too_short"
            debug["route_invalid_prevented"] = True
            return None, debug, "final_route_too_short"
        if len(final_points) < settings["min_output_route_len"]:
            debug["slalom_reason"] = "min_output_route_len_passthrough"
            debug["output_route_len"] = len(final_points)
            debug["route_invalid_prevented"] = True
            return None, debug, "min_output_route_len_passthrough"

        overlay = copy.deepcopy(payload)
        overlay["points"] = final_points
        overlay["route_len"] = len(final_points)
        overlay["route_source"] = "global_route"
        overlay["final_route_source"] = "slalom_fixed_world_spline"
        overlay["slalom_active"] = True
        overlay["slalom_reason"] = "active"
        overlay["slalom_mode"] = "fixed_world_spline"
        overlay["slalom_cones"] = copy.deepcopy(cones[:3])
        slalom_waypoints = []
        for point in slalom_points:
            waypoint = {
                "s": point.get("s"),
                "x": point.get("x"),
                "y": point.get("y"),
                "yaw": point.get("yaw"),
            }
            if settings["override_slalom_speed_limit"] and point.get("speed_limit_mps") is not None:
                waypoint["speed_limit_mps"] = point.get("speed_limit_mps")
            slalom_waypoints.append(waypoint)
        overlay["slalom_waypoints"] = slalom_waypoints

        debug.update(
            {
                "slalom_active": True,
                "slalom_reason": "active",
                "spline_point_count": len(slalom_points),
                "final_route_source": "slalom_fixed_world_spline",
                "output_route_len": len(final_points),
            }
        )
        return overlay, debug, "active"

    def _publish_debug(self, debug: dict[str, Any]) -> None:
        self.debug_pub.publish(String(data=json.dumps(debug)))

    def _route_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except Exception:
            self.route_pub.publish(msg)
            debug = self._base_debug()
            debug["slalom_reason"] = "route_base_json_parse_failed"
            self._publish_debug(debug)
            return
        if not isinstance(payload, dict):
            self.route_pub.publish(msg)
            debug = self._base_debug()
            debug["slalom_reason"] = "route_base_not_object"
            self._publish_debug(debug)
            return

        overlay, debug, _ = self._apply_overlay(payload)
        if overlay is None:
            self.route_pub.publish(msg)
            self._publish_debug(debug)
            return
        self.route_pub.publish(String(data=json.dumps(overlay)))
        self._publish_debug(debug)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SlalomOverlay()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
