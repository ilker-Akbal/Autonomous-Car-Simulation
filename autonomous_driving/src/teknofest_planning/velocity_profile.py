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


def compute_target_speed_from_route(
    points: list[dict[str, Any]],
    nearest_index: int,
    cruise_speed_mps: float = 4.5,
    min_turn_speed_mps: float = 2.0,
    max_speed_mps: float = 6.0,
    moderate_turn_yaw_deg: float = 18.0,
    sharp_turn_yaw_deg: float = 45.0,
) -> dict[str, Any]:
    if not points or nearest_index < 0 or nearest_index >= len(points):
        return {
            "target_speed_mps": 0.0,
            "turn_intensity": 0.0,
            "speed_reason": "route_invalid",
        }

    intensity = estimate_route_turn_intensity(points, nearest_index)
    intensity = clamp(intensity, 0.0, sharp_turn_yaw_deg)

    if intensity >= sharp_turn_yaw_deg:
        speed = min_turn_speed_mps
        reason = "sharp_turn"
    elif intensity >= moderate_turn_yaw_deg:
        ratio = (intensity - moderate_turn_yaw_deg) / max(1e-6, sharp_turn_yaw_deg - moderate_turn_yaw_deg)
        speed = cruise_speed_mps - (cruise_speed_mps - min_turn_speed_mps) * ratio
        reason = "moderate_turn"
    else:
        speed = cruise_speed_mps
        reason = "straight"

    speed = clamp(speed, min_turn_speed_mps, max_speed_mps)

    return {
        "target_speed_mps": speed,
        "turn_intensity": intensity,
        "speed_reason": reason,
    }
