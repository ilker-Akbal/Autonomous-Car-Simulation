import math
from typing import Any, Iterable


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
