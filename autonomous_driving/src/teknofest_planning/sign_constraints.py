import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Optional


TURN_ALIASES = {
    "right": "right",
    "sag": "right",
    "sağ": "right",
    "left": "left",
    "sol": "left",
    "straight": "straight",
    "duz": "straight",
    "düz": "straight",
    "u_turn": "u_turn",
    "uturn": "u_turn",
}


SIGN_TYPE_ALIASES = {
    "girilmez": "no_entry",
    "no_entry": "no_entry",
    "do_not_enter": "no_entry",
    "road_closed": "road_closed",
    "closed_road": "road_closed",
    "yol_kapali": "road_closed",
    "yol_kapalı": "road_closed",
    "road_work": "road_work",
    "roadworks": "road_work",
    "construction": "road_work",
    "yol_calismasi": "road_work",
    "yol_çalışması": "road_work",
    "no_right_turn": "no_right_turn",
    "saga_donulmez": "no_right_turn",
    "sağa_dönülmez": "no_right_turn",
    "no_left_turn": "no_left_turn",
    "sola_donulmez": "no_left_turn",
    "sola_dönülmez": "no_left_turn",
    "no_straight": "no_straight",
    "duz_gidilmez": "no_straight",
    "düz_gidilmez": "no_straight",
    "mandatory_right": "mandatory_right",
    "mecburi_sag": "mandatory_right",
    "mecburi_sağ": "mandatory_right",
    "mandatory_left": "mandatory_left",
    "mecburi_sol": "mandatory_left",
    "mandatory_straight": "mandatory_straight",
    "mecburi_duz": "mandatory_straight",
    "mecburi_düz": "mandatory_straight",
    "speed_limit": "speed_limit",
    "hiz_siniri": "speed_limit",
    "hız_sınırı": "speed_limit",
    "stop": "stop",
    "dur": "stop",
    "yield": "yield",
    "yol_ver": "yield",
    "no_parking": "no_parking",
    "park_yasagi": "no_parking",
    "park_yasağı": "no_parking",
    "traffic_light": "traffic_light",
    "trafik_isigi": "traffic_light",
    "trafik_ışığı": "traffic_light",
}


HARD_SEGMENT_CONSTRAINTS = {"no_entry", "road_closed", "road_work"}
TURN_RESTRICTION_CONSTRAINTS = {
    "no_right_turn",
    "no_left_turn",
    "no_straight",
    "mandatory_right",
    "mandatory_left",
    "mandatory_straight",
}
ANNOTATION_CONSTRAINTS = {"speed_limit", "stop", "yield", "no_parking"}
IGNORED_CONSTRAINTS = {"traffic_light"}


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip().lower().replace(" ", "_").replace("-", "_")


def normalize_sign_type(value: Any) -> str:
    text = _normalize_text(value)
    if text.startswith("hiz_siniri_") or text.startswith("hız_sınırı_") or text.startswith("speed_limit_"):
        return "speed_limit"
    if "mecburi" in text and ("sag" in text or "sağ" in text):
        return "mandatory_right"
    if "mecburi" in text and "sol" in text:
        return "mandatory_left"
    if "mecburi" in text and ("duz" in text or "düz" in text):
        return "mandatory_straight"
    if text in ("park_etmek_yasaktir", "park_etmek_yasaktır"):
        return "no_parking"
    return SIGN_TYPE_ALIASES.get(text, text)


def normalize_turn(value: Any) -> Optional[str]:
    text = _normalize_text(value)
    return TURN_ALIASES.get(text, text or None)


def _to_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            return None


def _speed_limit_kmh_from_text(value: Any) -> Optional[float]:
    text = _normalize_text(value)
    for prefix in ("hiz_siniri_", "hız_sınırı_", "speed_limit_"):
        if text.startswith(prefix):
            return _to_float(text[len(prefix) :])
    return None


def _normalize_turn_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = [part.strip() for part in value.replace(";", ",").split(",")]
    elif isinstance(value, (list, tuple, set)):
        raw_values = list(value)
    else:
        raw_values = [value]

    turns = []
    for item in raw_values:
        turn = normalize_turn(item)
        if turn and turn not in turns:
            turns.append(turn)
    return turns


@dataclass
class SignConstraint:
    sign_type: str
    raw_sign_type: str = ""
    constraint_id: str = ""
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    road_id: Optional[int] = None
    lane_id: Optional[int] = None
    yaw: Optional[float] = None
    affected_road_id: Optional[int] = None
    affected_lane_id: Optional[int] = None
    affected_junction_id: Optional[int] = None
    forbidden_turn: Optional[str] = None
    allowed_turns: list[str] = field(default_factory=list)
    speed_limit_kmh: Optional[float] = None
    speed_limit_mps: Optional[float] = None
    effective_radius_m: Optional[float] = None
    active: bool = True
    inactive_reason: Optional[str] = None
    source_path: Optional[str] = None
    raw: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return {
            "id": self.constraint_id,
            "sign_type": self.sign_type,
            "road_id": self.road_id,
            "lane_id": self.lane_id,
            "affected_road_id": self.affected_road_id,
            "affected_lane_id": self.affected_lane_id,
            "affected_junction_id": self.affected_junction_id,
            "forbidden_turn": self.forbidden_turn,
            "allowed_turns": self.allowed_turns,
            "speed_limit_mps": self.speed_limit_mps,
            "speed_limit_kmh": self.speed_limit_kmh,
            "active": self.active,
            "inactive_reason": self.inactive_reason,
        }

    def road_id_for_match(self) -> Optional[int]:
        return self.affected_road_id if self.affected_road_id is not None else self.road_id

    def lane_id_for_match(self) -> Optional[int]:
        return self.affected_lane_id if self.affected_lane_id is not None else self.lane_id

    def applies_to_point(self, point: dict[str, Any], default_radius_m: float) -> bool:
        if not self.active:
            return False

        road_id = self.road_id_for_match()
        lane_id = self.lane_id_for_match()
        if road_id is not None and int(point.get("road_id", -999999)) != int(road_id):
            return False
        if lane_id is not None and int(point.get("lane_id", -999999)) != int(lane_id):
            return False

        if road_id is not None:
            return True

        if self.x is None or self.y is None:
            return False

        radius = self.effective_radius_m or default_radius_m
        px = float(point.get("x", 0.0))
        py = float(point.get("y", 0.0))
        return math.hypot(px - self.x, py - self.y) <= radius


@dataclass
class ConstraintDecision:
    route_allowed: bool = True
    sign_constraint_replan_requested: bool = False
    sign_constraint_replan_reason: Optional[str] = None
    forbidden_road_lane_rejected: bool = False
    forbidden_turn_rejected: bool = False
    speed_limit_annotation: bool = False
    stop_yield_annotation: bool = False
    park_restriction_annotation: bool = False
    active_sign_constraints: list[dict[str, Any]] = field(default_factory=list)
    annotations_by_index: dict[int, dict[str, Any]] = field(default_factory=dict)

    def debug_payload(self) -> dict[str, Any]:
        return {
            "active_sign_constraints": self.active_sign_constraints,
            "sign_constraint_replan_requested": self.sign_constraint_replan_requested,
            "sign_constraint_replan_reason": self.sign_constraint_replan_reason,
            "forbidden_road_lane_rejected": self.forbidden_road_lane_rejected,
            "forbidden_turn_rejected": self.forbidden_turn_rejected,
            "speed_limit_annotation": self.speed_limit_annotation,
            "stop_yield_annotation": self.stop_yield_annotation,
            "park_restriction_annotation": self.park_restriction_annotation,
        }


class SignConstraintSet:
    def __init__(
        self,
        constraints: Optional[list[SignConstraint]] = None,
        loaded: bool = False,
        load_errors: Optional[list[str]] = None,
        default_effective_radius_m: float = 12.0,
    ):
        self.constraints = constraints or []
        self.loaded = loaded
        self.load_errors = load_errors or []
        self.default_effective_radius_m = default_effective_radius_m

    def __len__(self) -> int:
        return len(self.constraints)

    def active_constraints(self) -> list[SignConstraint]:
        return [constraint for constraint in self.constraints if constraint.active]

    def debug_payload(self) -> dict[str, Any]:
        return {
            "sign_constraints_loaded": self.loaded,
            "sign_constraints_count": len(self.constraints),
            "active_sign_constraints": [
                constraint.compact() for constraint in self.active_constraints()
            ],
            "sign_constraint_load_errors": self.load_errors,
        }

    def _constraints_for_point(self, point: dict[str, Any]) -> list[SignConstraint]:
        return [
            constraint
            for constraint in self.active_constraints()
            if constraint.applies_to_point(point, self.default_effective_radius_m)
        ]

    def annotate_point(self, point: dict[str, Any]) -> dict[str, Any]:
        annotations: dict[str, Any] = {
            "sign_constraints": [],
            "speed_limit_mps": None,
            "speed_limit_kmh": None,
            "forbidden_turn_info": None,
            "stop_or_yield_control": None,
            "park_restriction": None,
        }

        for constraint in self._constraints_for_point(point):
            annotations["sign_constraints"].append(constraint.compact())
            if constraint.sign_type == "speed_limit":
                annotations["speed_limit_mps"] = constraint.speed_limit_mps
                annotations["speed_limit_kmh"] = constraint.speed_limit_kmh
            elif constraint.sign_type in ("stop", "yield"):
                annotations["stop_or_yield_control"] = constraint.sign_type
            elif constraint.sign_type == "no_parking":
                annotations["park_restriction"] = "no_parking"
            elif constraint.sign_type in TURN_RESTRICTION_CONSTRAINTS:
                annotations["forbidden_turn_info"] = {
                    "sign_type": constraint.sign_type,
                    "forbidden_turn": constraint.forbidden_turn,
                    "allowed_turns": constraint.allowed_turns,
                }

        return annotations

    def evaluate_route(self, points: list[dict[str, Any]]) -> ConstraintDecision:
        decision = ConstraintDecision(
            active_sign_constraints=[
                constraint.compact() for constraint in self.active_constraints()
            ]
        )

        for index, point in enumerate(points):
            annotations = self.annotate_point(point)
            if annotations["sign_constraints"]:
                decision.annotations_by_index[index] = annotations

            if annotations["speed_limit_mps"] is not None or annotations["speed_limit_kmh"] is not None:
                decision.speed_limit_annotation = True
            if annotations["stop_or_yield_control"] is not None:
                decision.stop_yield_annotation = True
            if annotations["park_restriction"] is not None:
                decision.park_restriction_annotation = True

            for constraint in annotations["sign_constraints"]:
                if constraint["sign_type"] in HARD_SEGMENT_CONSTRAINTS:
                    decision.route_allowed = False
                    decision.sign_constraint_replan_requested = True
                    decision.sign_constraint_replan_reason = (
                        f"forbidden_road_lane:{constraint['sign_type']}"
                    )
                    decision.forbidden_road_lane_rejected = True
                    return decision

        for index, point in enumerate(points):
            turn_direction = point.get("turn_direction")
            if not turn_direction or turn_direction == "unknown":
                continue
            for constraint in self._constraints_for_point(point):
                if constraint.sign_type not in TURN_RESTRICTION_CONSTRAINTS:
                    continue
                forbidden_turn = constraint.forbidden_turn
                allowed_turns = constraint.allowed_turns
                sign_type = constraint.sign_type
                is_forbidden = (
                    forbidden_turn == turn_direction
                    or sign_type == f"no_{turn_direction}_turn"
                    or (sign_type == "no_straight" and turn_direction == "straight")
                    or (allowed_turns and turn_direction not in allowed_turns)
                )
                if is_forbidden:
                    decision.route_allowed = False
                    decision.sign_constraint_replan_requested = True
                    decision.sign_constraint_replan_reason = f"forbidden_turn:{turn_direction}"
                    decision.forbidden_turn_rejected = True
                    decision.annotations_by_index[index] = self.annotate_point(point)
                    return decision

        return decision


class SignConstraintLoader:
    def __init__(
        self,
        default_effective_radius_m: float = 12.0,
        carla_map: Any = None,
        carla_module: Any = None,
    ):
        self.default_effective_radius_m = default_effective_radius_m
        self.carla_map = carla_map
        self.carla_module = carla_module

    def load(self, geojson_path: str = "", json_path: str = "") -> SignConstraintSet:
        constraints: list[SignConstraint] = []
        load_errors: list[str] = []
        loaded_any_file = False

        for path in (geojson_path, json_path):
            if not path:
                continue
            expanded_path = os.path.abspath(os.path.expanduser(path))
            if not os.path.exists(expanded_path) or os.path.getsize(expanded_path) == 0:
                continue
            loaded_any_file = True
            try:
                with open(expanded_path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                constraints.extend(self._parse_payload(payload, expanded_path))
            except Exception as exc:
                load_errors.append(f"{expanded_path}: {exc}")

        constraints = self._dedupe_constraints(constraints)
        return SignConstraintSet(
            constraints=constraints,
            loaded=loaded_any_file and not load_errors,
            load_errors=load_errors,
            default_effective_radius_m=self.default_effective_radius_m,
        )

    def _dedupe_constraints(self, constraints: list[SignConstraint]) -> list[SignConstraint]:
        deduped: list[SignConstraint] = []
        seen = set()
        for constraint in constraints:
            key = (
                constraint.sign_type,
                constraint.road_id,
                constraint.lane_id,
                constraint.affected_road_id,
                constraint.affected_lane_id,
                round(constraint.x, 3) if constraint.x is not None else None,
                round(constraint.y, 3) if constraint.y is not None else None,
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(constraint)
        return deduped

    def _parse_payload(self, payload: Any, source_path: str) -> list[SignConstraint]:
        if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
            records = payload.get("features", [])
            return [
                constraint
                for constraint in (
                    self._constraint_from_feature(feature, source_path)
                    for feature in records
                )
                if constraint is not None
            ]

        if isinstance(payload, dict):
            records = payload.get("signs", payload.get("constraints", payload.get("features", [])))
            if not isinstance(records, list):
                records = [payload]
        elif isinstance(payload, list):
            records = payload
        else:
            records = []

        return [
            constraint
            for constraint in (
                self._constraint_from_record(record, source_path)
                for record in records
            )
            if constraint is not None
        ]

    def _constraint_from_feature(self, feature: Any, source_path: str) -> Optional[SignConstraint]:
        if not isinstance(feature, dict):
            return None
        properties = dict(feature.get("properties") or {})
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") if isinstance(geometry, dict) else None
        if isinstance(coordinates, list) and len(coordinates) >= 2:
            properties.setdefault("x", coordinates[0])
            properties.setdefault("y", coordinates[1])
            if len(coordinates) >= 3:
                properties.setdefault("z", coordinates[2])
        return self._constraint_from_record(properties, source_path, raw=feature)

    def _constraint_from_record(
        self,
        record: Any,
        source_path: str,
        raw: Optional[dict[str, Any]] = None,
    ) -> Optional[SignConstraint]:
        if not isinstance(record, dict):
            return None

        raw_sign_type = (
            record.get("sign_type")
            or record.get("sign")
            or record.get("type")
            or record.get("kind")
            or record.get("name")
            or ""
        )
        sign_type = normalize_sign_type(raw_sign_type)
        if not sign_type or sign_type in IGNORED_CONSTRAINTS:
            return None

        speed_limit_kmh = (
            _to_float(record.get("speed_limit_kmh", record.get("speed_limit")))
            or _speed_limit_kmh_from_text(raw_sign_type)
        )
        speed_limit_mps = _to_float(record.get("speed_limit_mps"))
        if sign_type == "speed_limit" and speed_limit_mps is None and speed_limit_kmh is not None:
            speed_limit_mps = speed_limit_kmh / 3.6

        constraint = SignConstraint(
            sign_type=sign_type,
            raw_sign_type=str(raw_sign_type),
            constraint_id=str(record.get("id", record.get("name", ""))),
            x=_to_float(record.get("x", record.get("carla_x", record.get("route_x", record.get("lon"))))),
            y=_to_float(record.get("y", record.get("carla_y", record.get("route_y", record.get("lat"))))),
            z=_to_float(record.get("z", record.get("carla_z", record.get("route_z")))),
            road_id=_to_int(record.get("road_id")),
            lane_id=_to_int(record.get("lane_id")),
            yaw=_to_float(record.get("yaw", record.get("direction"))),
            affected_road_id=_to_int(record.get("affected_road_id")),
            affected_lane_id=_to_int(record.get("affected_lane_id")),
            affected_junction_id=_to_int(record.get("affected_junction_id")),
            forbidden_turn=normalize_turn(record.get("forbidden_turn")),
            allowed_turns=_normalize_turn_list(record.get("allowed_turns")),
            speed_limit_kmh=speed_limit_kmh,
            speed_limit_mps=speed_limit_mps,
            effective_radius_m=_to_float(record.get("effective_radius_m", record.get("distance"))),
            source_path=source_path,
            raw=raw or dict(record),
        )

        if sign_type == "no_right_turn":
            constraint.forbidden_turn = "right"
        elif sign_type == "no_left_turn":
            constraint.forbidden_turn = "left"
        elif sign_type == "no_straight":
            constraint.forbidden_turn = "straight"
        elif sign_type == "mandatory_right":
            constraint.allowed_turns = ["right"]
        elif sign_type == "mandatory_left":
            constraint.allowed_turns = ["left"]
        elif sign_type == "mandatory_straight":
            constraint.allowed_turns = ["straight"]

        self._resolve_missing_road_lane(constraint)
        self._validate_constraint(constraint)
        return constraint

    def _resolve_missing_road_lane(self, constraint: SignConstraint) -> None:
        if (
            constraint.road_id is not None
            or constraint.x is None
            or constraint.y is None
            or self.carla_map is None
            or self.carla_module is None
        ):
            return

        try:
            location = self.carla_module.Location(
                x=float(constraint.x),
                y=float(constraint.y),
                z=float(constraint.z or 0.0),
            )
            waypoint = self.carla_map.get_waypoint(
                location,
                project_to_road=True,
                lane_type=self.carla_module.LaneType.Driving,
            )
            if waypoint is not None:
                constraint.road_id = int(waypoint.road_id)
                constraint.lane_id = int(waypoint.lane_id)
        except Exception as exc:
            constraint.active = False
            constraint.inactive_reason = f"waypoint_resolution_failed:{exc}"

    def _validate_constraint(self, constraint: SignConstraint) -> None:
        if constraint.sign_type not in (
            HARD_SEGMENT_CONSTRAINTS
            | TURN_RESTRICTION_CONSTRAINTS
            | ANNOTATION_CONSTRAINTS
        ):
            constraint.active = False
            constraint.inactive_reason = "unsupported_sign_type"
            return

        needs_location = constraint.sign_type in (
            HARD_SEGMENT_CONSTRAINTS
            | TURN_RESTRICTION_CONSTRAINTS
            | ANNOTATION_CONSTRAINTS
        )
        has_road_lane = constraint.road_id is not None or constraint.affected_road_id is not None
        has_position = constraint.x is not None and constraint.y is not None
        if needs_location and not has_road_lane and not has_position:
            constraint.active = False
            constraint.inactive_reason = "missing_location_or_road_lane"
