#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple


TURN_OPTIONS = {"LEFT", "RIGHT"}
AVOID_OPTIONS_FOR_SIGN_BASE = {"LEFT", "RIGHT", "CHANGELANELEFT", "CHANGELANERIGHT"}

KNOWN_SIGNS = {
    "hiz_siniri_20",
    "hiz_siniri_30",
    "hiz_siniri_40",
    "hiz_siniri_50",
    "dur",
    "yol_ver",
    "dikkat",
    "park_yeri",
    "park_etmek_yasaktir",
    "ileriden_saga_mecburi_yon",
    "ileriden_sola_mecburi_yon",
    "saga_mecburi_yon",
    "sola_mecburi_yon",
    "ileri_mecburi_yon",
    "ileri_ve_saga_mecburi_yon",
    "ileri_ve_sola_mecburi_yon",
    "serit_duzenleme_levhasi_sag",
    "serit_duzenleme_levhasi_sol",
    "tunel",
    "yaya_gecidi",
    "okul_gecidi",
    "iki_yonlu_yol",
    "yol_calismasi",
}

FALLBACK_MISSION_POINTS = [
    {"name": "start", "role": "start", "x": 220.41, "y": -5.19, "nokta_id": 0},
    {"name": "gorev_1", "role": "task", "x": 82.60, "y": -60.28, "nokta_id": 1},
    {"name": "gorev_2", "role": "task", "x": 99.19, "y": -191.51, "nokta_id": 2},
    {"name": "park_giris", "role": "parking", "x": 234.81, "y": -145.41, "nokta_id": 100},
]


def as_float(value, default=0.0) -> float:
    try:
        if value is None or value == "":
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def as_int(value, default=0) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(float(value))
    except Exception:
        return int(default)


def as_bool(value) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "evet"}


def norm_name(value) -> str:
    return str(value or "").strip()


def lower_name(value) -> str:
    return norm_name(value).lower().replace("ı", "i")


def classify_mission_point(name: str, props: Optional[dict] = None) -> str:
    props = props or {}
    lname = lower_name(name)
    kind = lower_name(props.get("kind", ""))
    role = lower_name(props.get("role", props.get("mission_role", props.get("point_role", ""))))

    if not lname:
        return "ignored"

    if lname.upper().startswith("route_"):
        return "route"

    control_prefixes = (
        "via",
        "via_",
        "ara",
        "ara_",
        "wp",
        "waypoint",
        "waypoint_",
        "gecis",
        "gecis_",
        "kontrol",
        "kontrol_",
    )

    if lname.startswith(control_prefixes):
        return "control"

    if role in {"control", "via", "route", "route_control", "intermediate"}:
        return "control"

    if lname == "start" or role == "start" or kind == "start":
        return "start"

    if lname.startswith("park") or role in {"park", "parking"} or "park" in lname:
        return "parking"

    if (
        lname.startswith("gorev")
        or lname.startswith("görev")
        or lname.startswith("task")
        or lname.startswith("durak")
        or "yolcu" in lname
        or role in {"task", "mission", "passenger", "stop"}
        or kind in {"task", "mission", "passenger", "stop"}
    ):
        return "task"

    return "control"


def read_route_csv(path: str) -> List[dict]:
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"route_index", "distance_m", "road_option", "carla_x", "carla_y"}
        missing = required - set(reader.fieldnames or [])

        if missing:
            raise RuntimeError(f"Route CSV eksik kolonlar içeriyor: {sorted(missing)} path={path}")

        for row in reader:
            rows.append(
                {
                    "name": row.get("name", ""),
                    "kind": row.get("kind", ""),
                    "route_index": as_int(row.get("route_index"), len(rows)),
                    "distance_m": as_float(row.get("distance_m")),
                    "road_option": str(row.get("road_option", "LANEFOLLOW")).upper().strip(),
                    "carla_x": as_float(row.get("carla_x")),
                    "carla_y": as_float(row.get("carla_y")),
                    "carla_z": as_float(row.get("carla_z")),
                    "carla_yaw": as_float(row.get("carla_yaw")),
                    "road_id": as_int(row.get("road_id"), -1),
                    "section_id": as_int(row.get("section_id"), 0),
                    "lane_id": as_int(row.get("lane_id"), 0),
                    "lane_width": as_float(row.get("lane_width"), 3.5),
                    "s": as_float(row.get("s"), 0.0),
                    "is_junction": as_bool(row.get("is_junction")),
                }
            )

    rows.sort(key=lambda r: (r["route_index"], r["distance_m"]))

    if len(rows) < 10:
        raise RuntimeError(f"Route CSV yeterli waypoint içermiyor: count={len(rows)} path={path}")

    return rows


def parse_point_from_feature(feature: dict) -> Optional[dict]:
    props = feature.get("properties", {}) or {}
    geom = feature.get("geometry", {}) or {}
    coords = geom.get("coordinates", []) or []

    name = norm_name(props.get("name", ""))

    if not name:
        return None

    if "carla_x" in props:
        x = as_float(props.get("carla_x"))
    elif len(coords) >= 1:
        x = as_float(coords[0])
    else:
        return None

    if "carla_y" in props:
        y = as_float(props.get("carla_y"))
    elif len(coords) >= 2:
        y = as_float(coords[1])
    else:
        return None

    role = classify_mission_point(name, props)

    return {
        "name": name,
        "role": role,
        "x": x,
        "y": y,
        "z": as_float(props.get("carla_z", props.get("z", 0.2))),
        "yaw": as_float(props.get("carla_yaw", props.get("yaw", 0.0))),
        "nokta_id": as_int(props.get("nokta_id"), 9999),
    }


def order_semantic_points(points: List[dict]) -> List[dict]:
    starts = [p for p in points if p["role"] == "start"]
    tasks = [p for p in points if p["role"] == "task"]
    parks = [p for p in points if p["role"] == "parking"]

    starts = sorted(starts, key=lambda p: p["nokta_id"])
    tasks = sorted(tasks, key=lambda p: p["nokta_id"])
    parks = sorted(parks, key=lambda p: p["nokta_id"])

    ordered = []

    if starts:
        ordered.append(starts[0])

    ordered.extend(tasks)
    ordered.extend(parks)

    return ordered


def read_mission_points(path: Optional[str]) -> Tuple[List[dict], List[dict]]:
    if not path or not os.path.exists(path):
        points = list(FALLBACK_MISSION_POINTS)
        return points, []

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    raw_points = []

    for feature in data.get("features", []):
        point = parse_point_from_feature(feature)

        if point is not None:
            raw_points.append(point)

    semantic = order_semantic_points(raw_points)

    if not semantic:
        semantic = list(FALLBACK_MISSION_POINTS)

    return semantic, raw_points


def nearest_route_distance(route: List[dict], x: float, y: float) -> Tuple[float, int, float]:
    best_i = 0
    best_d = 1e18

    for i, r in enumerate(route):
        d = math.hypot(r["carla_x"] - x, r["carla_y"] - y)

        if d < best_d:
            best_d = d
            best_i = i

    return route[best_i]["distance_m"], best_i, best_d


def group_consecutive(route: List[dict], key: str) -> List[dict]:
    groups = []
    start = 0
    prev = route[0][key]

    for i in range(1, len(route) + 1):
        cur = route[i][key] if i < len(route) else None

        if cur == prev:
            continue

        seg = route[start:i]

        groups.append(
            {
                "key": prev,
                "start_i": start,
                "end_i": i - 1,
                "start_distance_m": round(seg[0]["distance_m"], 3),
                "end_distance_m": round(seg[-1]["distance_m"], 3),
                "length_m": round(seg[-1]["distance_m"] - seg[0]["distance_m"], 3),
                "road_id": seg[0]["road_id"],
                "lane_id": seg[0]["lane_id"],
                "contains_junction": any(x["is_junction"] for x in seg),
            }
        )

        start = i
        prev = cur

    return groups


def is_candidate_good(row: dict, avoid_junction: bool = True) -> bool:
    if avoid_junction and row["is_junction"]:
        return False

    if row["road_option"] in AVOID_OPTIONS_FOR_SIGN_BASE:
        return False

    return True


def spacing_ok(distance_m: float, signs: List[dict], min_spacing_m: float) -> bool:
    return all(abs(distance_m - s["distance_m"]) >= min_spacing_m for s in signs)


def choose_safe_route_row(
    route: List[dict],
    signs: List[dict],
    target_d: float,
    min_spacing_m: float,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
    prefer_before: bool = False,
    prefer_after: bool = False,
    avoid_junction: bool = True,
) -> Optional[dict]:
    route_len = route[-1]["distance_m"]
    min_d = 8.0 if min_d is None else max(0.0, float(min_d))
    max_d = route_len - 8.0 if max_d is None else min(route_len, float(max_d))
    target_d = max(min_d, min(float(target_d), max_d))

    candidates = [r for r in route if min_d <= r["distance_m"] <= max_d]

    if prefer_before:
        candidates.sort(key=lambda r: (0 if r["distance_m"] <= target_d else 1, abs(r["distance_m"] - target_d)))
    elif prefer_after:
        candidates.sort(key=lambda r: (0 if r["distance_m"] >= target_d else 1, abs(r["distance_m"] - target_d)))
    else:
        candidates.sort(key=lambda r: abs(r["distance_m"] - target_d))

    for r in candidates:
        if not is_candidate_good(r, avoid_junction=avoid_junction):
            continue

        if not spacing_ok(r["distance_m"], signs, min_spacing_m):
            continue

        return r

    for r in candidates:
        if r["road_option"] in AVOID_OPTIONS_FOR_SIGN_BASE:
            continue

        if not spacing_ok(r["distance_m"], signs, min_spacing_m):
            continue

        return r

    return None


def add_sign(
    route: List[dict],
    signs: List[dict],
    skipped: List[dict],
    sign: str,
    target_d: float,
    reason: str,
    source_event: str,
    side: str = "R",
    required: bool = False,
    min_spacing_m: float = 18.0,
    min_d: Optional[float] = None,
    max_d: Optional[float] = None,
    prefer_before: bool = False,
    prefer_after: bool = False,
) -> bool:
    sign = str(sign).strip().lower()

    if sign not in KNOWN_SIGNS:
        raise RuntimeError(f"Bilinmeyen tabela adı: {sign}")

    row = choose_safe_route_row(
        route=route,
        signs=signs,
        target_d=target_d,
        min_spacing_m=min_spacing_m,
        min_d=min_d,
        max_d=max_d,
        prefer_before=prefer_before,
        prefer_after=prefer_after,
    )

    if row is None:
        payload = {
            "sign": sign,
            "target_distance_m": round(float(target_d), 3),
            "side": side,
            "reason": reason,
            "source_event": source_event,
            "required": bool(required),
            "skip_reason": "no_safe_route_row",
            "search_min_m": None if min_d is None else round(float(min_d), 3),
            "search_max_m": None if max_d is None else round(float(max_d), 3),
        }
        skipped.append(payload)

        if required:
            raise RuntimeError(
                "Zorunlu tabela için güvenli rota satırı bulunamadı: "
                + json.dumps(payload, ensure_ascii=False)
            )

        return False

    item = {
        "sign": sign,
        "distance_m": round(row["distance_m"], 3),
        "side": "L" if str(side).strip().upper().startswith("L") else "R",
        "reason": reason,
        "source_event": source_event,
        "required": bool(required),
        "route_index": row["route_index"],
        "route_x": round(row["carla_x"], 3),
        "route_y": round(row["carla_y"], 3),
        "route_yaw": round(row["carla_yaw"], 3),
        "road_id": row["road_id"],
        "lane_id": row["lane_id"],
        "road_option": row["road_option"],
        "is_junction": row["is_junction"],
        "target_distance_m": round(float(target_d), 3),
    }

    signs.append(item)
    signs.sort(key=lambda s: s["distance_m"])
    return True


def relevant_turn_events(route: List[dict]) -> List[dict]:
    groups = group_consecutive(route, "road_option")
    turns = []

    for g in groups:
        if g["key"] not in TURN_OPTIONS:
            continue

        if g["length_m"] < 6.0:
            continue

        if g["start_distance_m"] < 55.0:
            continue

        if route[-1]["distance_m"] - g["start_distance_m"] < 45.0:
            continue

        turns.append(g)

    return turns


def build_plan(route: List[dict], semantic_points: List[dict], raw_mission_points: List[dict]) -> dict:
    route_len = route[-1]["distance_m"]
    signs: List[dict] = []
    skipped: List[dict] = []
    omitted_by_design: List[dict] = []

    mission_distances: Dict[str, dict] = {}

    for p in semantic_points:
        d, idx, err = nearest_route_distance(route, p["x"], p["y"])
        mission_distances[p["name"]] = {
            "role": p["role"],
            "distance_m": round(d, 3),
            "route_index": idx,
            "xy_error_m": round(err, 3),
        }

    ignored_points = [
        {
            "name": p["name"],
            "role": p["role"],
            "x": round(p["x"], 3),
            "y": round(p["y"], 3),
        }
        for p in raw_mission_points
        if p.get("role") in {"control", "route", "ignored"}
    ]

    turns = relevant_turn_events(route)

    add_sign(
        route,
        signs,
        skipped,
        "hiz_siniri_30",
        target_d=34.0,
        reason="route_start_speed_limit_after_initial_lane_change",
        source_event="route_start",
        side="R",
        required=True,
        min_spacing_m=18.0,
        min_d=18.0,
        max_d=60.0,
        prefer_after=True,
    )

    for turn in turns:
        direction = turn["key"]
        turn_start = float(turn["start_distance_m"])
        turn_sign = "ileriden_saga_mecburi_yon" if direction == "RIGHT" else "ileriden_sola_mecburi_yon"
        source = f"turn_{direction.lower()}_{turn_start:.1f}m"

        add_sign(
            route,
            signs,
            skipped,
            "yol_ver",
            target_d=turn_start - 48.0,
            reason=f"yield_before_{direction.lower()}_turn",
            source_event=source,
            side="R",
            required=False,
            min_spacing_m=16.0,
            min_d=max(8.0, turn_start - 70.0),
            max_d=max(8.0, turn_start - 30.0),
            prefer_before=False,
        )

        add_sign(
            route,
            signs,
            skipped,
            turn_sign,
            target_d=turn_start - 26.0,
            reason=f"mandatory_direction_before_{direction.lower()}_turn",
            source_event=source,
            side="R",
            required=True,
            min_spacing_m=18.0,
            min_d=max(8.0, turn_start - 45.0),
            max_d=max(8.0, turn_start - 12.0),
            prefer_before=True,
        )

    task_points = [p for p in semantic_points if p["role"] == "task"]

    for p in task_points:
        md = mission_distances[p["name"]]["distance_m"]

        if md < 70.0:
            omitted_by_design.append(
                {
                    "sign": "dur",
                    "mission_point": p["name"],
                    "mission_distance_m": round(md, 3),
                    "reason": "task_too_close_to_route_start",
                }
            )
            continue

        nearby_turn = None
        for turn in turns:
            turn_start = float(turn["start_distance_m"])
            turn_end = float(turn["end_distance_m"])

            # Final kural:
            # Görev noktası dönüşün içinde veya dönüş çıkışına çok yakınsa dur tabelası koyma.
            # Ama dönüş bittikten sonra yeterli düz yaklaşım mesafesi varsa dur tabelası koyulabilir.
            # gorev_1 örneği: turn_end≈147m, mission≈181m -> 34m boşluk var, dur koyulmalı.
            # gorev_2 örneği: turn_end≈329m, mission≈329m -> dönüş çıkışıyla aynı, dur koyulmamalı.
            if (turn_start - 8.0) <= md <= (turn_end + 20.0):
                nearby_turn = turn
                break

        if nearby_turn is not None:
            omitted_by_design.append(
                {
                    "sign": "dur",
                    "mission_point": p["name"],
                    "mission_distance_m": round(md, 3),
                    "reason": "task_is_inside_or_immediately_after_turn_context",
                    "near_turn": {
                        "direction": nearby_turn["key"],
                        "start_distance_m": nearby_turn["start_distance_m"],
                        "end_distance_m": nearby_turn["end_distance_m"],
                    },
                }
            )
            continue

        add_sign(
            route,
            signs,
            skipped,
            "dur",
            target_d=md - 22.0,
            reason=f"stop_before_real_mission_point_{p['name']}",
            source_event=f"mission_{p['name']}",
            side="R",
            required=False,
            min_spacing_m=18.0,
            min_d=max(8.0, md - 55.0),
            max_d=max(8.0, md - 10.0),
            prefer_before=True,
        )

    if task_points:
        first_task_d = mission_distances[task_points[0]["name"]]["distance_m"]

        add_sign(
            route,
            signs,
            skipped,
            "hiz_siniri_20",
            target_d=first_task_d + 34.0,
            reason="slow_zone_after_first_real_mission_point",
            source_event=f"after_{task_points[0]['name']}",
            side="R",
            required=False,
            min_spacing_m=18.0,
            min_d=min(route_len - 8.0, first_task_d + 12.0),
            max_d=min(route_len - 8.0, first_task_d + 70.0),
            prefer_after=True,
        )

    parking_points = [p for p in semantic_points if p["role"] == "parking"]

    parking_candidates = []
    for p in parking_points:
        info = mission_distances.get(p["name"])
        if info is None:
            continue

        parking_candidates.append(
            {
                "name": p["name"],
                "distance_m": float(info["distance_m"]),
                "xy_error_m": float(info["xy_error_m"]),
            }
        )

    # Final kullanım kuralı:
    # Park tabelaları ancak rota sonuna yakın gerçek park anchor'ına göre üretilir.
    # Mission içindeki park isimli bir nokta rota başına/ortasına düşerse anchor kabul edilmez.
    if parking_candidates:
        best_parking = max(parking_candidates, key=lambda x: x["distance_m"])

        if best_parking["distance_m"] >= route_len * 0.72:
            park_name = best_parking["name"]
            park_d = best_parking["distance_m"]
            parking_selection = {
                "mode": "mission_parking_point",
                "selected": best_parking,
                "candidates": parking_candidates,
            }
        else:
            park_name = "route_end"
            park_d = route_len
            parking_selection = {
                "mode": "route_end_fallback",
                "reason": "no_parking_candidate_close_enough_to_route_end",
                "route_end_distance_m": round(route_len, 3),
                "candidates": parking_candidates,
            }
    else:
        park_name = "route_end"
        park_d = route_len
        parking_selection = {
            "mode": "route_end_fallback",
            "reason": "no_parking_point_in_mission",
            "route_end_distance_m": round(route_len, 3),
            "candidates": [],
        }

    add_sign(
        route,
        signs,
        skipped,
        "park_etmek_yasaktir",
        target_d=park_d - 72.0,
        reason="no_parking_zone_before_parking_entry",
        source_event=f"parking_{park_name}",
        side="L",
        required=False,
        min_spacing_m=18.0,
        min_d=max(8.0, park_d - 105.0),
        max_d=max(8.0, park_d - 45.0),
        prefer_before=False,
    )

    add_sign(
        route,
        signs,
        skipped,
        "park_yeri",
        target_d=park_d - 34.0,
        reason="parking_entry_sign_before_parking_target",
        source_event=f"parking_{park_name}",
        side="R",
        required=True,
        min_spacing_m=18.0,
        min_d=max(8.0, park_d - 55.0),
        max_d=max(8.0, park_d - 15.0),
        prefer_before=False,
    )

    semantic_summary = [
        {
            "name": p["name"],
            "role": p["role"],
            "distance_m": mission_distances.get(p["name"], {}).get("distance_m"),
            "xy_error_m": mission_distances.get(p["name"], {}).get("xy_error_m"),
        }
        for p in semantic_points
    ]

    return {
        "version": 2,
        "planner": "town03_route_semantic_sign_plan",
        "route_length_m": round(route_len, 3),
        "semantic_mission_points": semantic_summary,
        "ignored_route_control_points": ignored_points,
        "parking_selection": parking_selection,
        "turn_events": turns,
        "signs": signs,
        "skipped": skipped,
        "omitted_by_design": omitted_by_design,
    }


def write_geojson(plan: dict, out_json: str):
    features = []

    for i, sign in enumerate(plan.get("signs", []), start=1):
        features.append(
            {
                "type": "Feature",
                "properties": {
                    "id": f"SIGN_{i:03d}",
                    "sign": sign["sign"],
                    "distance_m": sign["distance_m"],
                    "side": sign["side"],
                    "reason": sign["reason"],
                    "source_event": sign["source_event"],
                    "required": sign["required"],
                    "road_id": sign["road_id"],
                    "lane_id": sign["lane_id"],
                    "road_option": sign["road_option"],
                },
                "geometry": {
                    "type": "Point",
                    "coordinates": [sign["route_x"], sign["route_y"], 0.0],
                },
            }
        )

    geo = {"type": "FeatureCollection", "features": features}
    out_geojson = os.path.splitext(out_json)[0] + ".geojson"

    with open(out_geojson, "w", encoding="utf-8") as f:
        json.dump(geo, f, ensure_ascii=False, indent=2)

    return out_geojson


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--route-csv", required=True)
    parser.add_argument("--mission-geojson", default="")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    route_csv = os.path.expanduser(args.route_csv)
    mission_geojson = os.path.expanduser(args.mission_geojson) if args.mission_geojson else ""
    out_json = os.path.expanduser(args.out)

    route = read_route_csv(route_csv)
    semantic_points, raw_points = read_mission_points(mission_geojson)
    plan = build_plan(route, semantic_points, raw_points)

    os.makedirs(os.path.dirname(out_json), exist_ok=True)

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(plan, f, ensure_ascii=False, indent=2)

    out_geojson = write_geojson(plan, out_json)

    print(f"[OK] sign plan written: {out_json}")
    print(f"[OK] sign plan geojson written: {out_geojson}")
    print(f"route_length_m={plan['route_length_m']}")
    print("parking_selection=" + json.dumps(plan.get("parking_selection", {}), ensure_ascii=False))
    print(f"semantic_mission_points={len(plan['semantic_mission_points'])}")
    print(f"ignored_route_control_points={len(plan['ignored_route_control_points'])}")
    print(f"turn_events={len(plan['turn_events'])}")
    print(f"signs={len(plan['signs'])}")
    print(f"skipped={len(plan['skipped'])}")
    print(f"omitted_by_design={len(plan.get('omitted_by_design', []))}")

    print("\n[SIGNS]")
    for i, s in enumerate(plan["signs"], 1):
        print(
            f"{i:02d} {s['distance_m']:7.1f}m "
            f"{s['sign']:<34} side={s['side']} "
            f"route_idx={s['route_index']:<4} "
            f"road={s['road_id']:<5} lane={s['lane_id']:<3} "
            f"opt={s['road_option']:<10} reason={s['reason']}"
        )

    if plan["skipped"]:
        print("\n[SKIPPED]")
        for s in plan["skipped"]:
            print(json.dumps(s, ensure_ascii=False))

    if plan.get("omitted_by_design"):
        print("\n[OMITTED_BY_DESIGN]")
        for s in plan["omitted_by_design"]:
            print(json.dumps(s, ensure_ascii=False))


if __name__ == "__main__":
    main()
