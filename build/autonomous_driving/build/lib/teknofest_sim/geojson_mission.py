import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class MissionPoint:
    name: str
    description: str
    nokta_id: int

    # Legacy alanlar. Gerçek GPS verilirse kullanılır.
    lon: float
    lat: float
    yaw: Optional[float] = None

    # CARLA local koordinatları. Town03 simülasyonunda asıl kullanılacak alanlar.
    carla_x: Optional[float] = None
    carla_y: Optional[float] = None
    carla_z: Optional[float] = None
    carla_yaw: Optional[float] = None

    road_id: Optional[int] = None
    lane_id: Optional[int] = None
    kind: Optional[str] = None

    # Legacy/debug alanları. Yarışma modunda bunların görevi rota vermek değildir.
    route_index: Optional[int] = None
    route_distance_m: Optional[float] = None


@dataclass
class MissionSpec:
    round_name: str
    start: MissionPoint
    task_points: List[MissionPoint]
    park_entry: MissionPoint
    raw_points: Dict[str, MissionPoint]


def _float_or_none(value):
    if value is None:
        return None

    try:
        return float(value)
    except Exception:
        return None


def _int_or_none(value):
    if value is None:
        return None

    try:
        return int(value)
    except Exception:
        return None


def _feature_name(feature: dict) -> str:
    props = feature.get("properties", {}) or {}
    return str(props.get("name", "")).strip()


def _feature_kind(feature: dict) -> Optional[str]:
    props = feature.get("properties", {}) or {}
    name = str(props.get("name", "")).strip().lower()
    kind = str(props.get("kind", "") or "").strip().lower()

    if kind:
        return kind

    if name == "start":
        return "start"

    if name == "park_giris" or name.startswith("park"):
        return "park"

    if name.startswith("gorev_") or name.startswith("passenger_"):
        return "task"

    if name.startswith("via_"):
        return "via"

    return None


def validate_geojson_schema(data: dict):
    if data.get("type") != "FeatureCollection":
        raise ValueError("GEOJSON kökü FeatureCollection olmalı.")

    features = data.get("features", [])
    if not isinstance(features, list):
        raise ValueError("GEOJSON features alanı liste olmalı.")

    names = []

    for feature in features:
        if feature.get("type") != "Feature":
            raise ValueError("Her GEOJSON elemanı Feature olmalı.")

        name = _feature_name(feature)
        if not name:
            raise ValueError("Her feature properties.name içermeli.")

        if name in names:
            raise ValueError(f"Tekrarlı feature name bulundu: {name}")

        names.append(name)

        geom = feature.get("geometry", {}) or {}
        if geom.get("type") != "Point":
            raise ValueError(f"Sadece Point geometry destekleniyor: {name} geometry={geom.get('type')}")

        coords = geom.get("coordinates", []) or []
        if len(coords) < 2:
            raise ValueError(f"{name}: GEOJSON Point coordinates [x/lon, y/lat] formatında olmalı.")

    if "start" not in names:
        raise ValueError("GEOJSON içinde name=start bulunmalı.")

    if "park_giris" not in names:
        raise ValueError("GEOJSON içinde name=park_giris bulunmalı.")


def validate_competition_geojson(data: dict):
    """
    Yarışma modu:
    GeoJSON sadece kritik görev girdisidir.
    Rota, şerit ve geçiş noktaları GeoJSON ile verilmez.

    Kabul edilenler:
      - start
      - gorev_*
      - passenger_*
      - park_giris

    Yasaklananlar:
      - via_*
      - kind=via
      - route_index / route_distance_m ile route sırası saklamak
    """
    validate_geojson_schema(data)

    illegal_route_points = []
    illegal_route_fields = []
    task_count = 0

    for feature in data.get("features", []):
        name = _feature_name(feature)
        kind = _feature_kind(feature)
        props = feature.get("properties", {}) or {}

        if name.lower().startswith("via_") or kind == "via":
            illegal_route_points.append(name)

        for key in ("route_index", "route_distance_m"):
            if key in props:
                illegal_route_fields.append(f"{name}.{key}")

        allowed = (
            name == "start"
            or name == "park_giris"
            or name.startswith("gorev_")
            or name.startswith("passenger_")
        )

        if not allowed:
            raise ValueError(
                "Yarışma modunda sadece start, gorev_*, passenger_* ve park_giris kabul edilir. "
                f"Geçersiz nokta: {name}"
            )

        if name.startswith("gorev_") or name.startswith("passenger_"):
            task_count += 1

    if illegal_route_points:
        raise ValueError(
            "Yarışma modunda GeoJSON rota/via noktası içeremez. "
            f"Bulunan noktalar: {illegal_route_points}"
        )

    if illegal_route_fields:
        raise ValueError(
            "Yarışma modunda GeoJSON rota sırası/mesafesi içeremez. "
            f"Bulunan alanlar: {illegal_route_fields}"
        )

    if task_count <= 0:
        raise ValueError("Yarışma modunda en az bir gorev_* veya passenger_* noktası olmalı.")


def _point_from_feature(feature: dict) -> MissionPoint:
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", []) or []

    if geom.get("type") != "Point":
        raise ValueError(f"Sadece Point geometry destekleniyor: {geom.get('type')}")

    if len(coords) < 2:
        raise ValueError("GEOJSON Point coordinates [x/lon, y/lat] formatında olmalı.")

    name = str(props.get("name", "")).strip()
    description = str(props.get("description", "")).strip()
    nokta_id = int(props.get("nokta_id", 0))

    # Bizim Town03 dosyalarında properties.carla_x/carla_y var.
    # Yoksa geometry.coordinates fallback olarak kullanılır.
    carla_x = _float_or_none(props.get("carla_x", coords[0]))
    carla_y = _float_or_none(props.get("carla_y", coords[1]))
    carla_z = _float_or_none(props.get("carla_z", 0.2))
    carla_yaw = _float_or_none(props.get("carla_yaw", props.get("yaw", None)))

    # Legacy lat/lon alanları. Gerçek GPS dosyası gelirse yine desteklenir.
    lon = float(props.get("lon", coords[0]))
    lat = float(props.get("lat", coords[1]))

    kind = str(props.get("kind", "")).strip() or _feature_kind(feature)

    return MissionPoint(
        name=name,
        description=description,
        nokta_id=nokta_id,
        lon=lon,
        lat=lat,
        yaw=_float_or_none(props.get("yaw", None)),
        carla_x=carla_x,
        carla_y=carla_y,
        carla_z=carla_z,
        carla_yaw=carla_yaw,
        road_id=_int_or_none(props.get("road_id", None)),
        lane_id=_int_or_none(props.get("lane_id", None)),
        kind=kind,
        route_index=_int_or_none(props.get("route_index", None)),
        route_distance_m=_float_or_none(props.get("route_distance_m", None)),
    )


def load_mission_geojson(
    path: str,
    round_name: str = "round_3",
    competition_mode: bool = False,
) -> MissionSpec:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if competition_mode:
        validate_competition_geojson(data)
    else:
        validate_geojson_schema(data)

    points: Dict[str, MissionPoint] = {}

    for feature in data.get("features", []):
        point = _point_from_feature(feature)

        if not point.name:
            raise ValueError("Her feature properties.name içermeli.")

        points[point.name] = point

    task_points = [
        p for name, p in sorted(points.items(), key=lambda item: item[1].nokta_id)
        if name.startswith("gorev_") or name.startswith("passenger_")
    ]

    # Eski/debug mod uyumluluğu.
    # Yarışma modunda buraya düşülmez; çünkü task zorunlu.
    if not task_points and not competition_mode:
        task_points = [
            p for name, p in sorted(points.items(), key=lambda item: item[1].nokta_id)
            if name not in {"start", "park_giris"} and not name.startswith("via_")
        ]

    return MissionSpec(
        round_name=round_name,
        start=points["start"],
        task_points=task_points,
        park_entry=points["park_giris"],
        raw_points=points,
    )


def haversine_meters(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    dx = lon2 - lon1
    dy = lat2 - lat1
    return math.sqrt(dx * dx + dy * dy) * 100000.0


def carla_xy_meters(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(float(x2) - float(x1), float(y2) - float(y1))


def mission_to_dict(mission: MissionSpec) -> dict:
    def p2d(p: MissionPoint):
        return {
            "name": p.name,
            "description": p.description,
            "nokta_id": p.nokta_id,
            "lat": p.lat,
            "lon": p.lon,
            "yaw": p.yaw,

            "carla_x": p.carla_x,
            "carla_y": p.carla_y,
            "carla_z": p.carla_z,
            "carla_yaw": p.carla_yaw,

            "road_id": p.road_id,
            "lane_id": p.lane_id,
            "kind": p.kind,
            "route_index": p.route_index,
            "route_distance_m": p.route_distance_m,
        }

    return {
        "round_name": mission.round_name,
        "start": p2d(mission.start),
        "task_points": [p2d(p) for p in mission.task_points],
        "park_entry": p2d(mission.park_entry),
    }
