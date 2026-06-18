from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional


@dataclass
class EgoStatus:
    x: float
    y: float
    yaw_deg: float
    speed_mps: float
    front_bumper_offset_m: float = 0.0
    distance_reference: str = "ego_center"


@dataclass
class TrafficLightDistanceResult:
    candidate: Optional[dict]
    tl_distance_m: Optional[float]
    stop_point_distance_m: Optional[float]
    distance_source: str
    distance_valid: bool
    is_in_front_of_ego: bool
    is_on_route_corridor: bool
    is_same_direction_relevant: bool
    tl_is_front_relevant: bool
    roi_rejected_reason: str
    route_index: Optional[int]
    lateral_distance_m: Optional[float]
    selected_light_id: Optional[int]
    selected_stop_point_x: Optional[float]
    selected_stop_point_y: Optional[float]
    stop_point_route_distance_m: Optional[float]
    actor_route_distance_m: Optional[float]
    actor_vs_stop_point_used: str
    closest_route_index: Optional[int]
    route_corridor_width_m: float
    relevance_basis: str
    reject_reason_detail: str
    candidate_count: int
    valid_route_candidate_count: int
    selected_candidate_rank: Optional[int]
    selected_delta_s_m: Optional[float]
    selected_lateral_m: Optional[float]
    stop_point_lateral_to_route_m: Optional[float]
    top_candidates_debug: list[dict]
    rejected_candidate_count: int
    direction_mismatch_soft: bool
    direction_reject_disabled_for_route_candidate: bool
    ego_x: Optional[float]
    ego_y: Optional[float]
    ego_yaw: Optional[float]
    front_bumper_x: Optional[float]
    front_bumper_y: Optional[float]
    euclidean_ego_to_stop_m: Optional[float]
    euclidean_front_bumper_to_stop_m: Optional[float]
    route_delta_s_m: Optional[float]
    ego_route_s: Optional[float]
    front_bumper_route_s: Optional[float]
    distance_reference: str
    stop_point_source: str
    carla_stop_waypoint_visual_mismatch: bool

    def to_dict(self) -> dict:
        return {
            "tl_distance_m": self.tl_distance_m,
            "stop_point_distance_m": self.stop_point_distance_m,
            "distance_source": self.distance_source,
            "distance_valid": self.distance_valid,
            "is_in_front_of_ego": self.is_in_front_of_ego,
            "is_on_route_corridor": self.is_on_route_corridor,
            "is_same_direction_relevant": self.is_same_direction_relevant,
            "tl_is_front_relevant": self.tl_is_front_relevant,
            "roi_rejected_reason": self.roi_rejected_reason,
            "route_index": self.route_index,
            "lateral_distance_m": self.lateral_distance_m,
            "selected_light_id": self.selected_light_id,
            "selected_stop_point_x": self.selected_stop_point_x,
            "selected_stop_point_y": self.selected_stop_point_y,
            "stop_point_route_distance_m": self.stop_point_route_distance_m,
            "actor_route_distance_m": self.actor_route_distance_m,
            "actor_vs_stop_point_used": self.actor_vs_stop_point_used,
            "closest_route_index": self.closest_route_index,
            "route_corridor_width_m": self.route_corridor_width_m,
            "relevance_basis": self.relevance_basis,
            "reject_reason_detail": self.reject_reason_detail,
            "candidate_count": self.candidate_count,
            "valid_route_candidate_count": self.valid_route_candidate_count,
            "selected_candidate_rank": self.selected_candidate_rank,
            "selected_delta_s_m": self.selected_delta_s_m,
            "selected_lateral_m": self.selected_lateral_m,
            "stop_point_lateral_to_route_m": self.stop_point_lateral_to_route_m,
            "top_candidates_debug": self.top_candidates_debug,
            "rejected_candidate_count": self.rejected_candidate_count,
            "direction_mismatch_soft": self.direction_mismatch_soft,
            "direction_reject_disabled_for_route_candidate": self.direction_reject_disabled_for_route_candidate,
            "ego_x": self.ego_x,
            "ego_y": self.ego_y,
            "ego_yaw": self.ego_yaw,
            "front_bumper_x": self.front_bumper_x,
            "front_bumper_y": self.front_bumper_y,
            "euclidean_ego_to_stop_m": self.euclidean_ego_to_stop_m,
            "euclidean_front_bumper_to_stop_m": self.euclidean_front_bumper_to_stop_m,
            "route_delta_s_m": self.route_delta_s_m,
            "ego_route_s": self.ego_route_s,
            "front_bumper_route_s": self.front_bumper_route_s,
            "distance_reference": self.distance_reference,
            "stop_point_source": self.stop_point_source,
            "carla_stop_waypoint_visual_mismatch": self.carla_stop_waypoint_visual_mismatch,
        }


class TrafficLightDistanceEstimator:
    def __init__(
        self,
        *,
        route_corridor_width_m: float = 4.5,
        same_direction_yaw_threshold_deg: float = 85.0,
        max_relevant_distance_m: float = 90.0,
    ):
        self.route_corridor_width_m = float(route_corridor_width_m)
        self.same_direction_yaw_threshold_deg = float(same_direction_yaw_threshold_deg)
        self.max_relevant_distance_m = float(max_relevant_distance_m)

    @staticmethod
    def normalize_angle_deg(angle: float) -> float:
        return (float(angle) + 180.0) % 360.0 - 180.0

    @staticmethod
    def distance_xy(a: dict, b: dict) -> float:
        return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))

    @staticmethod
    def dot_forward(ego: EgoStatus, point: dict) -> float:
        yaw = math.radians(ego.yaw_deg)
        dx = float(point["x"]) - ego.x
        dy = float(point["y"]) - ego.y
        return math.cos(yaw) * dx + math.sin(yaw) * dy

    @staticmethod
    def front_bumper_point(ego: EgoStatus) -> dict:
        yaw = math.radians(ego.yaw_deg)
        offset = max(0.0, float(ego.front_bumper_offset_m))
        return {
            "x": ego.x + math.cos(yaw) * offset,
            "y": ego.y + math.sin(yaw) * offset,
        }

    @staticmethod
    def distance_reference_point(ego: EgoStatus) -> dict:
        if str(ego.distance_reference).lower() == "front_bumper":
            return TrafficLightDistanceEstimator.front_bumper_point(ego)
        return {"x": ego.x, "y": ego.y}

    def cumulative_route_distances(self, route_points: list[dict]) -> list[float]:
        if not route_points:
            return []

        distances = [0.0]
        total = 0.0
        previous = route_points[0]
        for point in route_points[1:]:
            total += self.distance_xy(previous, point)
            distances.append(total)
            previous = point
        return distances

    def nearest_route_index(self, point: dict, route_points: list[dict]) -> tuple[int, float]:
        nearest_index = 0
        nearest_distance = float("inf")
        for index, route_point in enumerate(route_points):
            try:
                distance = self.distance_xy(point, route_point)
            except Exception:
                continue
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_index = index
        return nearest_index, nearest_distance

    def project_point_to_route(
        self,
        point: dict,
        route_points: list[dict],
        cumulative: list[float],
    ) -> tuple[float, float, int, float]:
        if len(route_points) < 2:
            index, lateral = self.nearest_route_index(point, route_points)
            route_s = cumulative[index] if cumulative else 0.0
            yaw = float(route_points[index].get("yaw_deg", 0.0)) if route_points else 0.0
            return route_s, lateral, index, yaw

        px = float(point["x"])
        py = float(point["y"])
        best_s = 0.0
        best_lateral = float("inf")
        best_index = 0
        best_yaw = 0.0

        for index in range(len(route_points) - 1):
            start = route_points[index]
            end = route_points[index + 1]
            ax = float(start["x"])
            ay = float(start["y"])
            bx = float(end["x"])
            by = float(end["y"])
            vx = bx - ax
            vy = by - ay
            seg_len_sq = vx * vx + vy * vy
            if seg_len_sq <= 1e-9:
                continue

            t = ((px - ax) * vx + (py - ay) * vy) / seg_len_sq
            t = max(0.0, min(1.0, t))
            proj_x = ax + t * vx
            proj_y = ay + t * vy
            lateral = math.hypot(px - proj_x, py - proj_y)
            if lateral < best_lateral:
                seg_len = math.sqrt(seg_len_sq)
                best_lateral = lateral
                best_s = (cumulative[index] if cumulative else 0.0) + t * seg_len
                best_index = index if t < 0.5 else index + 1
                best_yaw = math.degrees(math.atan2(vy, vx))

        return best_s, best_lateral, best_index, best_yaw

    def candidate_stop_points(self, candidate: dict) -> list[tuple[dict, str, int]]:
        stop_waypoints = candidate.get("stop_waypoints") or []
        if stop_waypoints:
            return [
                (dict(stop_waypoint), "carla_stop_waypoint", int(index))
                for index, stop_waypoint in enumerate(stop_waypoints)
            ]

        location = candidate.get("location") or {}
        return [(dict(location), "carla_actor_location", 0)]

    @staticmethod
    def compact_candidate_debug(evaluations: list[dict], limit: int = 5) -> list[dict]:
        ordered = sorted(
            evaluations,
            key=lambda item: (
                0 if item["reason"] == "" else 1,
                0 if bool(item.get("same_direction", False)) else 1,
                abs(float(item["delta_s_m"])),
                float(item["lateral_m"]),
            ),
        )
        return [
            {
                "rank": index + 1,
                "actor_id": item["actor_id"],
                "stop_waypoint_index": item["stop_waypoint_index"],
                "delta_s_m": round(float(item["delta_s_m"]), 3),
                "lateral_m": round(float(item["lateral_m"]), 3),
                "route_index": item["route_index"],
                "reason": item["reason"] or "accepted",
                "same_direction": bool(item.get("same_direction", False)),
                "direction_mismatch_soft": bool(item.get("direction_mismatch_soft", False)),
            }
            for index, item in enumerate(ordered[:limit])
        ]

    def choose_candidate(
        self,
        *,
        ego: EgoStatus,
        route_points: list[dict],
        detections: list[dict],
    ) -> TrafficLightDistanceResult:
        if not detections:
            return self.empty("no_detection")

        if not route_points:
            return self.empty("missing_route")

        ego_point = {"x": ego.x, "y": ego.y}
        front_bumper_point = self.front_bumper_point(ego)
        reference_point = self.distance_reference_point(ego)
        cumulative = self.cumulative_route_distances(route_points)
        ego_s, _ego_lateral, ego_index, _ego_route_yaw = self.project_point_to_route(
            ego_point,
            route_points,
            cumulative,
        )
        front_bumper_s, _front_lateral, _front_index, _front_route_yaw = self.project_point_to_route(
            front_bumper_point,
            route_points,
            cumulative,
        )
        reference_s, _reference_lateral, _reference_index, _reference_route_yaw = self.project_point_to_route(
            reference_point,
            route_points,
            cumulative,
        )

        evaluations = []
        first_rejection = ""
        candidate_count = len(detections)

        for candidate in detections:
            location = candidate.get("location") or {}
            if location.get("x") is None or location.get("y") is None:
                first_rejection = first_rejection or "missing_candidate_location"
                continue

            for stop_point, source, stop_waypoint_index in self.candidate_stop_points(candidate):
                if stop_point.get("x") is None or stop_point.get("y") is None:
                    first_rejection = first_rejection or "missing_stop_point"
                    continue

                stop_s, lateral, stop_index, route_yaw = self.project_point_to_route(
                    stop_point,
                    route_points,
                    cumulative,
                )
                stop_distance = stop_s - reference_s
                actor_s, _actor_lateral, _actor_index, _actor_route_yaw = self.project_point_to_route(
                    location,
                    route_points,
                    cumulative,
                )
                tl_distance = math.hypot(float(location["x"]) - ego.x, float(location["y"]) - ego.y)
                euclidean_ego_to_stop = self.distance_xy(ego_point, stop_point)
                euclidean_front_to_stop = self.distance_xy(front_bumper_point, stop_point)

                forward_dot = self.dot_forward(ego, stop_point)
                in_front = stop_distance > 0.0
                on_corridor = lateral <= self.route_corridor_width_m
                relevance_basis = (
                    "route_projected_stop_waypoint"
                    if source == "carla_stop_waypoint"
                    else "route_projected_actor_location"
                )

                light_yaw = float(candidate.get("yaw_deg", route_yaw))
                yaw_delta = abs(self.normalize_angle_deg(light_yaw - route_yaw))
                same_direction = (
                    yaw_delta <= self.same_direction_yaw_threshold_deg
                    or yaw_delta >= 180.0 - self.same_direction_yaw_threshold_deg
                )
                direction_mismatch_soft = not same_direction

                reason = ""
                if not in_front:
                    reason = "not_in_front"
                elif stop_distance > self.max_relevant_distance_m:
                    reason = "too_far"
                elif not on_corridor:
                    reason = "outside_route_corridor"

                detail = reason
                if reason == "not_in_front":
                    detail = f"stop_point_not_in_front:delta_s_m={stop_distance:.3f},forward_dot={forward_dot:.3f}"
                elif reason == "outside_route_corridor":
                    detail = (
                        "stop_point_outside_route_corridor:"
                        f"lateral_m={lateral:.3f},limit_m={self.route_corridor_width_m:.3f},basis={relevance_basis}"
                    )
                elif reason == "direction_mismatch":
                    detail = f"direction_mismatch:yaw_delta_deg={yaw_delta:.3f},limit_deg={self.same_direction_yaw_threshold_deg:.3f}"
                elif reason == "too_far":
                    detail = f"too_far:delta_s_m={stop_distance:.3f},limit_m={self.max_relevant_distance_m:.3f}"
                elif direction_mismatch_soft:
                    detail = (
                        "accepted_direction_mismatch_soft:"
                        f"yaw_delta_deg={yaw_delta:.3f},limit_deg={self.same_direction_yaw_threshold_deg:.3f},"
                        "direction_reject_disabled_for_route_candidate=true"
                    )
                else:
                    detail = "accepted"

                evaluations.append(
                    {
                        "candidate": candidate,
                        "actor_id": candidate.get("actor_id"),
                        "stop_point": stop_point,
                        "source": source,
                        "stop_waypoint_index": stop_waypoint_index,
                        "tl_distance_m": tl_distance,
                        "delta_s_m": stop_distance,
                        "stop_s_m": stop_s,
                        "actor_s_m": actor_s,
                        "ego_s_m": ego_s,
                        "front_bumper_s_m": front_bumper_s,
                        "reference_s_m": reference_s,
                        "ego_point": ego_point,
                        "ego_yaw": ego.yaw_deg,
                        "front_bumper_point": front_bumper_point,
                        "euclidean_ego_to_stop_m": euclidean_ego_to_stop,
                        "euclidean_front_bumper_to_stop_m": euclidean_front_to_stop,
                        "lateral_m": lateral,
                        "route_index": int(stop_index),
                        "in_front": bool(in_front),
                        "on_corridor": bool(on_corridor),
                        "same_direction": bool(same_direction),
                        "direction_mismatch_soft": bool(direction_mismatch_soft),
                        "direction_reject_disabled_for_route_candidate": bool(direction_mismatch_soft and in_front and on_corridor),
                        "reason": reason,
                        "detail": detail,
                        "relevance_basis": relevance_basis,
                        "distance_reference": str(ego.distance_reference).lower(),
                    }
                )

        if not evaluations:
            return self.empty(first_rejection or "no_valid_candidate", candidate_count=candidate_count)

        valid = [item for item in evaluations if item["reason"] == ""]
        top_debug = self.compact_candidate_debug(evaluations)

        if valid:
            ranked_valid = sorted(
                valid,
                key=lambda item: (
                    0 if bool(item["same_direction"]) else 1,
                    float(item["delta_s_m"]),
                    float(item["lateral_m"]),
                ),
            )
            selected = ranked_valid[0]
            selected_rank = next(
                (
                    item["rank"]
                    for item in top_debug
                    if item["actor_id"] == selected["actor_id"]
                    and item["stop_waypoint_index"] == selected["stop_waypoint_index"]
                    and item["reason"] == "accepted"
                ),
                1,
            )
            return self.result_from_evaluation(
                selected,
                candidate_count=candidate_count,
                valid_route_candidate_count=len(valid),
                selected_candidate_rank=selected_rank,
                top_candidates_debug=top_debug,
                rejected_candidate_count=len(evaluations) - len(valid),
            )

        best_rejected = sorted(
            evaluations,
            key=lambda item: (
                0 if float(item["delta_s_m"]) > 0.0 else 1,
                abs(float(item["delta_s_m"])),
                float(item["lateral_m"]),
            ),
        )[0]
        best_rejected = dict(best_rejected)
        best_rejected["reason"] = "no_route_relevant_candidate"
        best_rejected["detail"] = (
            "no_route_relevant_candidate:"
            f"evaluated={len(evaluations)},top_candidates={top_debug}"
        )
        return self.result_from_evaluation(
            best_rejected,
            candidate_count=candidate_count,
            valid_route_candidate_count=0,
            selected_candidate_rank=None,
            top_candidates_debug=top_debug,
            rejected_candidate_count=len(evaluations),
        )

        return self.empty(first_rejection or "no_valid_candidate")

    def result_from_evaluation(
        self,
        item: dict,
        *,
        candidate_count: int,
        valid_route_candidate_count: int,
        selected_candidate_rank: Optional[int],
        top_candidates_debug: list[dict],
        rejected_candidate_count: int,
    ) -> TrafficLightDistanceResult:
        stop_point = item["stop_point"]
        relevant = item["reason"] == ""
        return TrafficLightDistanceResult(
            candidate=item["candidate"],
            tl_distance_m=round(float(item["tl_distance_m"]), 3),
            stop_point_distance_m=round(float(item["delta_s_m"]), 3),
            distance_source=item["source"],
            distance_valid=True,
            is_in_front_of_ego=bool(item["in_front"]),
            is_on_route_corridor=bool(item["on_corridor"]),
            is_same_direction_relevant=bool(item["same_direction"]),
            tl_is_front_relevant=bool(relevant),
            roi_rejected_reason=item["reason"],
            route_index=int(item["route_index"]),
            lateral_distance_m=round(float(item["lateral_m"]), 3),
            selected_light_id=item["actor_id"],
            selected_stop_point_x=round(float(stop_point["x"]), 3),
            selected_stop_point_y=round(float(stop_point["y"]), 3),
            stop_point_route_distance_m=round(float(item["stop_s_m"]), 3),
            actor_route_distance_m=round(float(item["actor_s_m"]), 3),
            actor_vs_stop_point_used="stop_point" if item["source"] == "carla_stop_waypoint" else "actor_location",
            closest_route_index=int(item["route_index"]),
            route_corridor_width_m=round(float(self.route_corridor_width_m), 3),
            relevance_basis=item["relevance_basis"],
            reject_reason_detail=item["detail"],
            candidate_count=int(candidate_count),
            valid_route_candidate_count=int(valid_route_candidate_count),
            selected_candidate_rank=selected_candidate_rank,
            selected_delta_s_m=round(float(item["delta_s_m"]), 3),
            selected_lateral_m=round(float(item["lateral_m"]), 3),
            stop_point_lateral_to_route_m=round(float(item["lateral_m"]), 3),
            top_candidates_debug=top_candidates_debug,
            rejected_candidate_count=int(rejected_candidate_count),
            direction_mismatch_soft=bool(item.get("direction_mismatch_soft", False)),
            direction_reject_disabled_for_route_candidate=bool(
                item.get("direction_reject_disabled_for_route_candidate", False)
            ),
            ego_x=round(float(item["ego_point"]["x"]), 3),
            ego_y=round(float(item["ego_point"]["y"]), 3),
            ego_yaw=round(float(item["ego_yaw"]), 3),
            front_bumper_x=round(float(item["front_bumper_point"]["x"]), 3),
            front_bumper_y=round(float(item["front_bumper_point"]["y"]), 3),
            euclidean_ego_to_stop_m=round(float(item["euclidean_ego_to_stop_m"]), 3),
            euclidean_front_bumper_to_stop_m=round(float(item["euclidean_front_bumper_to_stop_m"]), 3),
            route_delta_s_m=round(float(item["delta_s_m"]), 3),
            ego_route_s=round(float(item["ego_s_m"]), 3),
            front_bumper_route_s=round(float(item["front_bumper_s_m"]), 3),
            distance_reference=item["distance_reference"],
            stop_point_source=item["source"],
            carla_stop_waypoint_visual_mismatch=False,
        )

    @staticmethod
    def empty(reason: str, candidate_count: int = 0) -> TrafficLightDistanceResult:
        return TrafficLightDistanceResult(
            candidate=None,
            tl_distance_m=None,
            stop_point_distance_m=None,
            distance_source="none",
            distance_valid=False,
            is_in_front_of_ego=False,
            is_on_route_corridor=False,
            is_same_direction_relevant=False,
            tl_is_front_relevant=False,
            roi_rejected_reason=reason,
            route_index=None,
            lateral_distance_m=None,
            selected_light_id=None,
            selected_stop_point_x=None,
            selected_stop_point_y=None,
            stop_point_route_distance_m=None,
            actor_route_distance_m=None,
            actor_vs_stop_point_used="none",
            closest_route_index=None,
            route_corridor_width_m=0.0,
            relevance_basis="none",
            reject_reason_detail=reason,
            candidate_count=int(candidate_count),
            valid_route_candidate_count=0,
            selected_candidate_rank=None,
            selected_delta_s_m=None,
            selected_lateral_m=None,
            stop_point_lateral_to_route_m=None,
            top_candidates_debug=[],
            rejected_candidate_count=0,
            direction_mismatch_soft=False,
            direction_reject_disabled_for_route_candidate=False,
            ego_x=None,
            ego_y=None,
            ego_yaw=None,
            front_bumper_x=None,
            front_bumper_y=None,
            euclidean_ego_to_stop_m=None,
            euclidean_front_bumper_to_stop_m=None,
            route_delta_s_m=None,
            ego_route_s=None,
            front_bumper_route_s=None,
            distance_reference="none",
            stop_point_source="none",
            carla_stop_waypoint_visual_mismatch=False,
        )
