import math
from typing import Any


def angle_diff_deg(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def signed_angle_diff_deg(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def classify_turn_direction(
    entry_yaw_deg: float,
    exit_yaw_deg: float,
    straight_threshold_deg: float = 25.0,
    u_turn_threshold_deg: float = 135.0,
) -> str:
    delta = signed_angle_diff_deg(exit_yaw_deg, entry_yaw_deg)
    abs_delta = abs(delta)
    if abs_delta >= u_turn_threshold_deg:
        return "u_turn"
    if abs_delta <= straight_threshold_deg:
        return "straight"
    if delta > 0.0:
        return "left"
    if delta < 0.0:
        return "right"
    return "unknown"


def distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def nearest_point_index(points: list[dict[str, Any]], x: float, y: float) -> int:
    best_index = 0
    best_distance = float("inf")
    for index, point in enumerate(points):
        dx = float(point.get("x", 0.0)) - x
        dy = float(point.get("y", 0.0)) - y
        dist = math.hypot(dx, dy)
        if dist < best_distance:
            best_distance = dist
            best_index = index
    return best_index


def nearest_point_distance(points: list[dict[str, Any]], x: float, y: float) -> tuple[int, float]:
    best_index = 0
    best_distance = float("inf")
    for index, point in enumerate(points):
        dx = float(point.get("x", 0.0)) - x
        dy = float(point.get("y", 0.0)) - y
        dist = math.hypot(dx, dy)
        if dist < best_distance:
            best_distance = dist
            best_index = index
    return best_index, best_distance


def cumulative_route_s(points: list[dict[str, Any]]) -> list[float]:
    """Return cumulative arc length for a JSON route polyline."""
    if not points:
        return []

    cumulative = [0.0]
    for previous, current in zip(points, points[1:]):
        cumulative.append(
            cumulative[-1]
            + distance_2d(
                float(previous.get("x", 0.0)),
                float(previous.get("y", 0.0)),
                float(current.get("x", 0.0)),
                float(current.get("y", 0.0)),
            )
        )
    return cumulative


def project_actor_to_route(
    points: list[dict[str, Any]],
    x: float,
    y: float,
    route_s: list[float] | None = None,
) -> dict[str, float | int] | None:
    """Project a point onto the route polyline and return route-relative data."""
    if not points:
        return None

    if route_s is None or len(route_s) != len(points):
        route_s = cumulative_route_s(points)

    if len(points) == 1:
        point_x = float(points[0].get("x", 0.0))
        point_y = float(points[0].get("y", 0.0))
        return {
            "route_index": 0,
            "route_s_m": 0.0,
            "lateral_distance_m": distance_2d(x, y, point_x, point_y),
            "projected_x": point_x,
            "projected_y": point_y,
        }

    best_projection = None
    best_distance = float("inf")
    for index in range(len(points) - 1):
        start_x = float(points[index].get("x", 0.0))
        start_y = float(points[index].get("y", 0.0))
        end_x = float(points[index + 1].get("x", 0.0))
        end_y = float(points[index + 1].get("y", 0.0))
        segment_x = end_x - start_x
        segment_y = end_y - start_y
        segment_length_sq = segment_x * segment_x + segment_y * segment_y

        if segment_length_sq <= 1e-12:
            fraction = 0.0
        else:
            fraction = (
                (x - start_x) * segment_x + (y - start_y) * segment_y
            ) / segment_length_sq
            fraction = max(0.0, min(1.0, fraction))

        projected_x = start_x + fraction * segment_x
        projected_y = start_y + fraction * segment_y
        lateral_distance = distance_2d(x, y, projected_x, projected_y)
        if lateral_distance >= best_distance:
            continue

        segment_length = math.sqrt(segment_length_sq)
        best_distance = lateral_distance
        best_projection = {
            "route_index": index if fraction < 0.5 else index + 1,
            "route_s_m": route_s[index] + fraction * segment_length,
            "lateral_distance_m": lateral_distance,
            "projected_x": projected_x,
            "projected_y": projected_y,
        }

    return best_projection


def is_ahead_on_route(
    ego_route_s_m: float,
    actor_route_s_m: float,
    tolerance_m: float = 0.0,
) -> bool:
    return actor_route_s_m + tolerance_m >= ego_route_s_m


def route_distance_between_indices(
    points: list[dict[str, Any]],
    start_index: int,
    end_index: int,
) -> float:
    if not points:
        return 0.0

    route_s = cumulative_route_s(points)
    start_index = max(0, min(len(points) - 1, start_index))
    end_index = max(0, min(len(points) - 1, end_index))
    return route_s[end_index] - route_s[start_index]


def route_continuity_ok(
    points: list[dict[str, Any]],
    max_segment_distance_m: float = 8.0,
    max_yaw_delta_deg: float = 120.0,
) -> bool:
    if len(points) < 2:
        return True

    for previous, current in zip(points, points[1:]):
        segment_distance = distance_2d(
            float(previous.get("x", 0.0)),
            float(previous.get("y", 0.0)),
            float(current.get("x", 0.0)),
            float(current.get("y", 0.0)),
        )
        if segment_distance > max_segment_distance_m:
            return False

        if bool(previous.get("is_junction", False)) or bool(current.get("is_junction", False)):
            continue

        yaw_delta = angle_diff_deg(
            float(previous.get("yaw", 0.0)),
            float(current.get("yaw", 0.0)),
        )
        if yaw_delta > max_yaw_delta_deg:
            return False

    return True


def build_local_route_segment(
    points: list[dict[str, Any]],
    start_index: int,
    horizon_m: float,
) -> list[dict[str, Any]]:
    if start_index < 0:
        start_index = 0
    if start_index >= len(points):
        return []

    segment = []
    distance_accum = 0.0
    previous = points[start_index]
    segment.append(previous)

    for point in points[start_index + 1 :]:
        this_distance = distance_2d(
            float(previous.get("x", 0.0)),
            float(previous.get("y", 0.0)),
            float(point.get("x", 0.0)),
            float(point.get("y", 0.0)),
        )
        distance_accum += this_distance
        if distance_accum > horizon_m:
            break
        segment.append(point)
        previous = point

    return segment


def forward_window_search(
    points: list[dict[str, Any]],
    x: float,
    y: float,
    initial_index: int,
    window: int = 10,
    reset_threshold_m: float = 10.0,
) -> int:
    if not points:
        return 0

    length = len(points)
    if initial_index < 0:
        initial_index = 0
    if initial_index >= length:
        initial_index = length - 1

    start = max(0, initial_index - window)
    end = min(length, initial_index + window + 1)
    best_index, best_distance = nearest_point_distance(points[start:end], x, y)
    best_index += start

    if best_distance > reset_threshold_m:
        return nearest_point_index(points, x, y)

    return best_index
