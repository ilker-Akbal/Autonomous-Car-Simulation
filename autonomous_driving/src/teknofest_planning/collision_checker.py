import math
from dataclasses import dataclass
from typing import Any


@dataclass
class CollisionInfo:
    obstacle_index: int
    path_index: int
    distance: float
    description: str


def circle_collision(point_x: float, point_y: float, circle_x: float, circle_y: float, radius: float) -> bool:
    return math.hypot(point_x - circle_x, point_y - circle_y) <= radius


def check_path_collision(
    path_points: list[dict[str, Any]],
    obstacle_points: list[dict[str, Any]],
    vehicle_radius_m: float = 1.0,
    obstacle_radius_m: float = 0.5,
    clearance_m: float = 0.2,
) -> tuple[bool, CollisionInfo | None]:
    if not path_points or not obstacle_points:
        return True, None

    threshold = vehicle_radius_m + obstacle_radius_m + clearance_m

    for path_index, path_point in enumerate(path_points):
        px = float(path_point.get("x", 0.0))
        py = float(path_point.get("y", 0.0))
        for obs_index, obstacle in enumerate(obstacle_points):
            ox = float(obstacle.get("x", 0.0))
            oy = float(obstacle.get("y", 0.0))
            distance = math.hypot(px - ox, py - oy)
            if distance <= threshold:
                info = CollisionInfo(
                    obstacle_index=obs_index,
                    path_index=path_index,
                    distance=distance,
                    description="path point within obstacle clearance",
                )
                return False, info

    return True, None


class CollisionChecker:
    def __init__(self, vehicle_radius_m: float = 1.0, obstacle_radius_m: float = 0.5, clearance_m: float = 0.2):
        self.vehicle_radius_m = vehicle_radius_m
        self.obstacle_radius_m = obstacle_radius_m
        self.clearance_m = clearance_m

    def check(self, path_points: list[dict[str, Any]], obstacle_points: list[dict[str, Any]]) -> dict[str, Any]:
        collision_free, collision_info = check_path_collision(
            path_points,
            obstacle_points,
            vehicle_radius_m=self.vehicle_radius_m,
            obstacle_radius_m=self.obstacle_radius_m,
            clearance_m=self.clearance_m,
        )
        return {
            "collision_free": collision_free,
            "collision_info": {
                "path_index": collision_info.path_index,
                "obstacle_index": collision_info.obstacle_index,
                "distance": collision_info.distance,
                "description": collision_info.description,
            } if collision_info is not None else None,
        }


# TODO: wire obstacle points from perception or external obstacle detectors in Phase 2C.
