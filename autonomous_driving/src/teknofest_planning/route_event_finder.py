from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RouteEvent:
    event_type: str
    distance_m: float | None
    stop_point: dict | None
    color: str
    source: str
    priority: int
    reason: str


def _distance(a: dict, b: dict) -> float:
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


def _route_progress_and_lateral(route_points: list[dict], point: dict) -> tuple[int, float]:
    nearest_index = 0
    nearest_distance = float("inf")
    for index, route_point in enumerate(route_points):
        try:
            distance = _distance(route_point, point)
        except (KeyError, TypeError, ValueError):
            continue
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_index = index
    return nearest_index, nearest_distance


def _route_distance(route_points: list[dict], start_index: int, end_index: int) -> float:
    if not route_points:
        return 0.0
    start = max(0, min(start_index, len(route_points) - 1))
    end = max(start, min(end_index, len(route_points) - 1))
    total = 0.0
    for index in range(start, end):
        total += _distance(route_points[index], route_points[index + 1])
    return total


class RouteEventFinder:
    PRIORITY = {
        "EMERGENCY": 0,
        "MISSION_STOP": 1,
        "RED_LIGHT": 2,
        "YELLOW_LIGHT_STOP": 3,
        "LEAD_VEHICLE_STOPPED": 4,
        "PARKING": 5,
        "LEAD_VEHICLE": 6,
        "NORMAL": 100,
    }

    def __init__(self, route_corridor_width_m: float = 4.5):
        self.route_corridor_width_m = max(1.0, float(route_corridor_width_m))

    def _event_distance_on_route(
        self,
        ego_status: dict,
        route_points: list[dict],
        stop_point: dict | None,
    ) -> float | None:
        if not route_points or not stop_point:
            return None
        ego_point = {
            "x": float(((ego_status or {}).get("location") or {}).get("x")),
            "y": float(((ego_status or {}).get("location") or {}).get("y")),
        }
        ego_index, _ = _route_progress_and_lateral(route_points, ego_point)
        event_index, event_lateral = _route_progress_and_lateral(route_points, stop_point)
        if event_lateral > self.route_corridor_width_m:
            return None
        if event_index < ego_index:
            return None
        return _route_distance(route_points, ego_index, event_index)

    def _mission_event(self, ego_status: dict, route_points: list[dict], mission_payload: dict | None) -> RouteEvent | None:
        mission = mission_payload or {}
        if not bool(mission.get("must_stop", False)):
            return None
        target = mission.get("objective_target") or mission.get("target")
        point = None
        if isinstance(target, dict) and target.get("carla_x") is not None and target.get("carla_y") is not None:
            point = {
                "x": float(target["carla_x"]),
                "y": float(target["carla_y"]),
                "z": float(target.get("carla_z", 0.0) or 0.0),
            }
        distance_m = mission.get("distance_to_objective_m")
        if distance_m is None:
            distance_m = self._event_distance_on_route(ego_status, route_points, point)
        return RouteEvent(
            event_type="MISSION_STOP",
            distance_m=float(distance_m) if distance_m is not None else None,
            stop_point=point,
            color="mission",
            source="mission_node",
            priority=self.PRIORITY["MISSION_STOP"],
            reason="mission_must_stop",
        )

    def _parking_event(self, ego_status: dict, route_points: list[dict], mission_payload: dict | None) -> RouteEvent | None:
        mission = mission_payload or {}
        stage = str(mission.get("stage") or "")
        if stage != "PARKING":
            return None
        target = mission.get("objective_target") or {}
        if target.get("carla_x") is None or target.get("carla_y") is None:
            return None
        point = {
            "x": float(target["carla_x"]),
            "y": float(target["carla_y"]),
            "z": float(target.get("carla_z", 0.0) or 0.0),
        }
        distance_m = mission.get("distance_to_objective_m")
        if distance_m is None:
            distance_m = self._event_distance_on_route(ego_status, route_points, point)
        return RouteEvent(
            event_type="PARKING",
            distance_m=float(distance_m) if distance_m is not None else None,
            stop_point=point,
            color="parking",
            source="mission_node",
            priority=self.PRIORITY["PARKING"],
            reason="mission_parking_stage",
        )

    def _traffic_light_event(self, ego_status: dict, route_points: list[dict], tl_event: dict | None) -> RouteEvent | None:
        payload = tl_event or {}
        if not bool(payload.get("has_relevant_light", False)):
            return None
        color = str(payload.get("color") or "unknown").lower()
        if color not in {"red", "yellow", "green"}:
            return None
        stop_point = payload.get("stop_point") if isinstance(payload.get("stop_point"), dict) else None
        distance_m = payload.get("distance_m")
        if distance_m is None:
            distance_m = self._event_distance_on_route(ego_status, route_points, stop_point)
        event_type = "RED_LIGHT"
        priority_key = "RED_LIGHT"
        if color == "yellow":
            event_type = "YELLOW_LIGHT_STOP"
            priority_key = "YELLOW_LIGHT_STOP"
        elif color == "green":
            event_type = "NORMAL"
            priority_key = "NORMAL"
        return RouteEvent(
            event_type=event_type,
            distance_m=float(distance_m) if distance_m is not None else None,
            stop_point=stop_point,
            color=color,
            source=str(payload.get("source") or "traffic_light_manager_node"),
            priority=self.PRIORITY[priority_key],
            reason=str(payload.get("reason") or f"traffic_light_{color}"),
        )

    def _lead_vehicle_event(self, lead_vehicle: dict | None) -> RouteEvent | None:
        payload = lead_vehicle or {}
        if payload.get("distance_m") is None:
            return None
        speed_mps = float(payload.get("speed_mps", 0.0) or 0.0)
        distance_m = float(payload["distance_m"])
        event_type = "LEAD_VEHICLE_STOPPED" if speed_mps <= 0.3 else "LEAD_VEHICLE"
        return RouteEvent(
            event_type=event_type,
            distance_m=distance_m,
            stop_point=None,
            color="vehicle",
            source=str(payload.get("source") or "lead_vehicle"),
            priority=self.PRIORITY.get(event_type, self.PRIORITY["LEAD_VEHICLE"]),
            reason="lead_vehicle_on_route",
        )

    def find_event(
        self,
        ego_status: dict,
        route_points: list[dict],
        mission_payload: dict | None,
        tl_event: dict | None,
        lead_vehicle: dict | None = None,
    ) -> RouteEvent:
        if not route_points:
            return RouteEvent(
                event_type="NORMAL",
                distance_m=None,
                stop_point=None,
                color="unknown",
                source="route_event_finder",
                priority=self.PRIORITY["NORMAL"],
                reason="missing_route",
            )

        candidates = []
        for builder in (
            lambda: self._mission_event(ego_status, route_points, mission_payload),
            lambda: self._parking_event(ego_status, route_points, mission_payload),
            lambda: self._traffic_light_event(ego_status, route_points, tl_event),
            lambda: self._lead_vehicle_event(lead_vehicle),
        ):
            event = builder()
            if event is None:
                continue
            if event.distance_m is not None and event.distance_m < -0.1:
                continue
            candidates.append(event)

        if not candidates:
            return RouteEvent(
                event_type="NONE",
                distance_m=None,
                stop_point=None,
                color="none",
                source="route_event_finder",
                priority=self.PRIORITY["NORMAL"],
                reason="no_relevant_event",
            )

        return min(
            candidates,
            key=lambda item: (
                int(item.priority),
                float(item.distance_m) if item.distance_m is not None else float("inf"),
            ),
        )
