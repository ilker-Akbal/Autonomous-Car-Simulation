from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass
class RoutePoint:
    x: float
    y: float
    z: float = 0.0
    yaw_deg: float = 0.0
    road_id: Optional[int] = None
    lane_id: Optional[int] = None
    lane_width: Optional[float] = None
    is_junction: bool = False
    route_index: int = 0
    road_option: str = "LANEFOLLOW"


@dataclass
class EgoPose:
    x: float
    y: float
    yaw_deg: float
    speed_mps: float


@dataclass
class LanePlan:
    target_point: RoutePoint
    nearest_index: int
    lookahead_index: int
    distance_to_goal_m: Optional[float]
    target_speed_mps: float
    stop_request: bool
    lateral_error_m: float
    heading_error_deg: float
    route_remaining_m: float
    lane_change_ahead: bool
    lane_change_reason: str
    reason: str


@dataclass
class LanePolicyConfig:
    cruise_speed_mps: float = 6.0
    turn_speed_mps: float = 3.2
    approach_speed_mps: float = 2.0
    min_drive_speed_mps: float = 2.2
    stop_distance_m: float = 1.8
    approach_distance_m: float = 14.0
    lookahead_base_m: float = 4.5
    lookahead_gain: float = 0.55
    lookahead_min_m: float = 3.5
    lookahead_max_m: float = 10.0
    turn_angle_slowdown_deg: float = 35.0
    lateral_slowdown_error_m: float = 0.9
    lateral_slowdown_speed_mps: float = 3.0


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance_xy(a: RoutePoint | EgoPose, b: RoutePoint | EgoPose) -> float:
    return math.hypot(float(a.x) - float(b.x), float(a.y) - float(b.y))


def route_distance(points: list[RoutePoint], start_index: int, end_index: int) -> float:
    if not points:
        return 0.0

    start = max(0, min(start_index, len(points) - 1))
    end = max(0, min(end_index, len(points) - 1))

    if end <= start:
        return 0.0

    total = 0.0
    for index in range(start, end):
        total += distance_xy(points[index], points[index + 1])
    return total


def nearest_route_index(points: list[RoutePoint], ego: EgoPose, last_index: int = 0) -> int:
    if not points:
        return 0

    start = max(0, min(last_index - 2, len(points) - 1))
    end = min(len(points), max(start + 1, last_index + 80))
    search_points: Iterable[tuple[int, RoutePoint]] = enumerate(points[start:end], start)

    nearest_index = start
    nearest_distance = float("inf")

    for index, point in search_points:
        dist = distance_xy(point, ego)
        if dist < nearest_distance:
            nearest_distance = dist
            nearest_index = index

    return nearest_index


def lookahead_route_index(
    points: list[RoutePoint],
    nearest_index: int,
    lookahead_m: float,
) -> int:
    if not points:
        return 0

    total = 0.0
    index = max(0, min(nearest_index, len(points) - 1))

    while index < len(points) - 1 and total < lookahead_m:
        total += distance_xy(points[index], points[index + 1])
        index += 1

    return index


def signed_lateral_error_m(ego: EgoPose, reference: RoutePoint) -> float:
    yaw = math.radians(reference.yaw_deg)
    dx = float(ego.x) - float(reference.x)
    dy = float(ego.y) - float(reference.y)
    return -math.sin(yaw) * dx + math.cos(yaw) * dy


def heading_change_deg(points: list[RoutePoint], start_index: int, end_index: int) -> float:
    if not points or end_index <= start_index:
        return 0.0

    first = points[max(0, min(start_index, len(points) - 1))].yaw_deg
    last = points[max(0, min(end_index, len(points) - 1))].yaw_deg
    delta = (last - first + 180.0) % 360.0 - 180.0
    return abs(delta)


def heading_error_deg(ego: EgoPose, target: RoutePoint) -> float:
    delta = (float(target.yaw_deg) - float(ego.yaw_deg) + 180.0) % 360.0 - 180.0
    return delta


def planned_lane_change(route_option: str) -> bool:
    option = str(route_option or "").upper()
    return option in {"LEFT", "RIGHT", "CHANGELANELEFT", "CHANGELANERIGHT"}


def lane_key(point: RoutePoint) -> tuple[Optional[int], Optional[int]]:
    return point.road_id, point.lane_id


def stable_lookahead_route_index(
    points: list[RoutePoint],
    nearest_index: int,
    desired_index: int,
) -> tuple[int, bool, str]:
    if not points:
        return 0, False, "empty_route"

    nearest_index = max(0, min(nearest_index, len(points) - 1))
    desired_index = max(nearest_index, min(desired_index, len(points) - 1))
    current_key = lane_key(points[nearest_index])
    desired = points[desired_index]

    if lane_key(desired) == current_key:
        return desired_index, False, str(desired.road_option)

    fallback_index = desired_index
    for index in range(nearest_index, desired_index + 1):
        point = points[index]
        if lane_key(point) == current_key:
            fallback_index = index
        else:
            break

    if fallback_index <= nearest_index and desired_index > nearest_index:
        fallback_index = nearest_index

    point = points[fallback_index]
    return fallback_index, lane_key(point) != current_key, "lane_preserve"


def speed_for_context(
    *,
    config: LanePolicyConfig,
    mission_must_stop: bool,
    distance_to_goal_m: Optional[float],
    heading_change_ahead_deg: float,
) -> tuple[float, bool, str]:
    if mission_must_stop:
        if distance_to_goal_m is not None and distance_to_goal_m <= config.stop_distance_m:
            return 0.0, True, "mission_stop"

        if distance_to_goal_m is not None and distance_to_goal_m <= config.approach_distance_m:
            ratio = clamp(distance_to_goal_m / config.approach_distance_m, 0.0, 1.0)
            speed = max(config.approach_speed_mps, config.cruise_speed_mps * ratio)
            return speed, False, "mission_approach"

    if distance_to_goal_m is not None and distance_to_goal_m <= config.approach_distance_m:
        ratio = clamp(distance_to_goal_m / config.approach_distance_m, 0.0, 1.0)
        speed = max(config.min_drive_speed_mps, config.cruise_speed_mps * ratio)
        return speed, False, "goal_approach"

    if heading_change_ahead_deg >= config.turn_angle_slowdown_deg:
        return config.turn_speed_mps, False, "turn_slowdown"

    return config.cruise_speed_mps, False, "cruise"


def apply_lateral_speed_guard(
    *,
    config: LanePolicyConfig,
    target_speed_mps: float,
    reason: str,
    lateral_error_m: float,
) -> tuple[float, str]:
    if abs(lateral_error_m) < config.lateral_slowdown_error_m:
        return target_speed_mps, reason

    guarded_speed = max(config.min_drive_speed_mps, config.lateral_slowdown_speed_mps)
    return min(target_speed_mps, guarded_speed), "cross_track_slowdown"


def build_lane_plan(
    *,
    route_points: list[RoutePoint],
    ego: EgoPose,
    config: LanePolicyConfig,
    last_nearest_index: int,
    mission_must_stop: bool,
    distance_to_goal_m: Optional[float],
) -> LanePlan:
    if not route_points:
        fallback = RoutePoint(x=ego.x, y=ego.y, yaw_deg=ego.yaw_deg)
        return LanePlan(
            target_point=fallback,
            nearest_index=0,
            lookahead_index=0,
            distance_to_goal_m=distance_to_goal_m,
            target_speed_mps=0.0,
            stop_request=True,
            lateral_error_m=0.0,
            heading_error_deg=0.0,
            route_remaining_m=0.0,
            lane_change_ahead=False,
            lane_change_reason="empty_route",
            reason="empty_route",
        )

    nearest_index = nearest_route_index(route_points, ego, last_nearest_index)
    lookahead_m = clamp(
        config.lookahead_base_m + ego.speed_mps * config.lookahead_gain,
        config.lookahead_min_m,
        config.lookahead_max_m,
    )
    desired_lookahead_index = lookahead_route_index(route_points, nearest_index, lookahead_m)
    lookahead_index, lane_change_ahead, lane_change_reason = stable_lookahead_route_index(
        route_points,
        nearest_index,
        desired_lookahead_index,
    )
    target_point = route_points[lookahead_index]
    remaining = route_distance(route_points, nearest_index, len(route_points) - 1)
    ahead_heading = heading_change_deg(route_points, nearest_index, lookahead_index)
    lateral_error = signed_lateral_error_m(ego, route_points[nearest_index])
    target_speed, stop_request, reason = speed_for_context(
        config=config,
        mission_must_stop=mission_must_stop,
        distance_to_goal_m=distance_to_goal_m if distance_to_goal_m is not None else remaining,
        heading_change_ahead_deg=ahead_heading,
    )
    if not stop_request:
        target_speed, reason = apply_lateral_speed_guard(
            config=config,
            target_speed_mps=target_speed,
            reason=reason,
            lateral_error_m=lateral_error,
        )

    return LanePlan(
        target_point=target_point,
        nearest_index=nearest_index,
        lookahead_index=lookahead_index,
        distance_to_goal_m=distance_to_goal_m,
        target_speed_mps=target_speed,
        stop_request=stop_request,
        lateral_error_m=lateral_error,
        heading_error_deg=heading_error_deg(ego, target_point),
        route_remaining_m=remaining,
        lane_change_ahead=lane_change_ahead,
        lane_change_reason=lane_change_reason,
        reason=reason,
    )
