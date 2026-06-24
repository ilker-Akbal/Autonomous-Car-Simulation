import math
from typing import Any


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def angle_diff_deg(a: float, b: float) -> float:
    d = (a - b + 180.0) % 360.0 - 180.0
    return abs(d)


def estimate_route_turn_intensity(points: list[dict[str, Any]], start_index: int, window: int = 8) -> float:
    if not points or start_index < 0 or start_index >= len(points):
        return 0.0

    total_change = 0.0
    count = 0
    last_yaw = float(points[start_index].get("yaw", 0.0))

    for next_index in range(start_index + 1, min(len(points), start_index + 1 + window)):
        next_yaw = float(points[next_index].get("yaw", last_yaw))
        total_change += angle_diff_deg(next_yaw, last_yaw)
        last_yaw = next_yaw
        count += 1

    if count == 0:
        return 0.0

    # average degrees per segment over the window
    return total_change / count


def route_speed_limit_mps(
    points: list[dict[str, Any]],
    start_index: int,
    window: int = 8,
) -> float | None:
    speed_limits = []
    if not points or start_index < 0 or start_index >= len(points):
        return None

    for index in range(start_index, min(len(points), start_index + 1 + window)):
        point = points[index]
        limit_mps = point.get("speed_limit_mps")
        if limit_mps is None and point.get("speed_limit_kmh") is not None:
            limit_mps = float(point.get("speed_limit_kmh")) / 3.6
        if limit_mps is not None:
            speed_limits.append(max(0.0, float(limit_mps)))

    return min(speed_limits) if speed_limits else None


def protected_turn_context(points: list[dict[str, Any]], start_index: int, window: int = 4) -> bool:
    if not points or start_index < 0 or start_index >= len(points):
        return False

    protected_turns = {"left", "right", "u_turn"}
    for index in range(start_index, min(len(points), start_index + 1 + window)):
        turn_direction = str(points[index].get("turn_direction", "unknown"))
        if turn_direction in protected_turns:
            return True
    return False


def route_end_context(
    points: list[dict[str, Any]],
    start_index: int,
    approach_distance_m: float = 15.0,
) -> str | None:
    if not points or start_index < 0 or start_index >= len(points):
        return None

    current_s = points[start_index].get("s")
    final_s = points[-1].get("s")
    if current_s is None or final_s is None:
        return None

    remaining_m = max(0.0, float(final_s) - float(current_s))
    if remaining_m > approach_distance_m:
        return None

    goal_name = str(points[-1].get("goal_name", points[start_index].get("goal_name", ""))).lower()
    goal_kind = str(points[-1].get("goal_kind", points[start_index].get("goal_kind", ""))).lower()
    if "park" in goal_name or "park" in goal_kind:
        return "park_approach"
    if "pickup" in goal_kind or goal_name == "gorev_1":
        return "pickup_approach"
    if "dropoff" in goal_kind or goal_name == "gorev_2":
        return "dropoff_approach"
    if "gorev" in goal_name or "görev" in goal_name or "task" in goal_name:
        return "mission_approach"
    return None


def compute_target_speed_from_route(
    points: list[dict[str, Any]],
    nearest_index: int,
    cruise_speed_mps: float = 4.5,
    min_turn_speed_mps: float = 2.0,
    max_speed_mps: float = 6.0,
    moderate_turn_yaw_deg: float = 18.0,
    sharp_turn_yaw_deg: float = 45.0,
    speed_boost_enabled: bool = True,
    nominal_speed_boost_mps: float = 2.0,
) -> dict[str, Any]:
    if not points or nearest_index < 0 or nearest_index >= len(points):
        return {
            "target_speed_mps": 0.0,
            "turn_intensity": 0.0,
            "speed_reason": "route_invalid",
            "speed_boost_enabled": bool(speed_boost_enabled),
            "speed_boost_mps": float(nominal_speed_boost_mps),
            "speed_boost_applied": False,
            "boost_applied": False,
            "speed_context": "route_invalid",
            "pre_boost_speed_mps": 0.0,
            "post_boost_speed_mps": 0.0,
            "speed_limit_clamped": False,
            "clamp_reason": "route_invalid",
            "turn_speed_protected": True,
        }

    intensity = estimate_route_turn_intensity(points, nearest_index)
    intensity = clamp(intensity, 0.0, sharp_turn_yaw_deg)
    junction_turn_protected = protected_turn_context(points, nearest_index)
    protected_end_context = route_end_context(points, nearest_index)

    if intensity >= sharp_turn_yaw_deg:
        speed = min_turn_speed_mps
        reason = "sharp_turn"
        speed_context = "sharp_turn"
    elif intensity >= moderate_turn_yaw_deg:
        ratio = (intensity - moderate_turn_yaw_deg) / max(1e-6, sharp_turn_yaw_deg - moderate_turn_yaw_deg)
        speed = cruise_speed_mps - (cruise_speed_mps - min_turn_speed_mps) * ratio
        reason = "moderate_turn"
        speed_context = "moderate_turn"
    elif junction_turn_protected:
        speed = cruise_speed_mps
        reason = "junction_turn"
        speed_context = "junction_turn"
    elif protected_end_context is not None:
        speed = min(cruise_speed_mps, min_turn_speed_mps)
        reason = protected_end_context
        speed_context = protected_end_context
    else:
        speed = cruise_speed_mps
        reason = "straight"
        speed_context = "nominal"

    speed = clamp(speed, min_turn_speed_mps, max_speed_mps)
    pre_boost_speed = speed
    clamp_reason = None
    turn_speed_protected = speed_context != "nominal"
    speed_boost_applied = False
    if (
        speed_boost_enabled
        and not turn_speed_protected
        and nominal_speed_boost_mps > 0.0
    ):
        speed = min(max_speed_mps, speed + float(nominal_speed_boost_mps))

    speed_limit = route_speed_limit_mps(points, nearest_index)
    speed_limit_clamped = False
    if speed_limit is not None and speed_limit > 0.0 and speed > speed_limit:
        speed = float(speed_limit)
        speed_limit_clamped = True
        clamp_reason = "speed_limit"
    elif speed_limit is not None and speed_limit <= 0.0:
        clamp_reason = "ignored_nonpositive_speed_limit"

    clamped_speed = clamp(speed, 0.0, max_speed_mps)
    if clamped_speed < speed and clamp_reason is None:
        clamp_reason = "max_speed"
    speed = clamped_speed
    speed_boost_applied = speed > pre_boost_speed

    return {
        "target_speed_mps": speed,
        "turn_intensity": intensity,
        "speed_reason": reason,
        "speed_boost_enabled": bool(speed_boost_enabled),
        "speed_boost_mps": float(nominal_speed_boost_mps),
        "speed_boost_applied": speed_boost_applied,
        "boost_applied": speed_boost_applied,
        "speed_context": speed_context,
        "pre_boost_speed_mps": pre_boost_speed,
        "post_boost_speed_mps": speed,
        "speed_limit_clamped": speed_limit_clamped,
        "clamp_reason": clamp_reason,
        "turn_speed_protected": turn_speed_protected,
    }
